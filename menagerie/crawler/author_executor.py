"""Supervised headless author executor: the author lane with no managing session.

Invoked by the existing :class:`CommandAuthorLane` subprocess contract (argv +
one absolute request path; result at the envelope's required path; typed exit
codes ``0/64/75/76/78``), this program owns one model's complete author
lifecycle:

- **stage 1** runs as ``claude -p`` with the verified pinned research
  configuration (every flag is load-bearing; a session that cannot reach its
  tools must fail loudly, never degrade quietly);
- the machine-owned **source broker** turns the typed discovery result into
  manifest rows whose exact strings (commit SHAs, digests, citation metadata)
  are derived from receipts, never authored;
- **stage 2** runs as ``claude -p --resume <stage-1 session>`` for context
  retention without a managing session, with at most one typed supplementary
  broker round.

Every artifact of an attempt lives under its own attempt directory
(:mod:`menagerie.crawler.author_attempts`); the executor is the only publisher
to the envelope's required path, publication is bound to the attempt nonce the
executor opened, and the published digest is verified and printed as a receipt
the lane re-verifies. A killed executor leaves a durable attempt record; the
retry resumes the recorded provider session when the record proves stage 1
completed, and otherwise opens attempt N+1 with the prior attempt in the
feedback channel.

Pause classification authority is structured signals only: the ``claude -p``
JSON error fields decide quota exits, and this program's own diagnostics use
closed reason codes, never verbatim provider text, so a GitHub rate-limit
message in a dying author's output can never masquerade as an Anthropic quota
event.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.author_attempts import (
    AttemptHandle,
    latest_attempt,
    list_attempts,
    new_attempt,
    prior_attempts_summary,
)
from menagerie.crawler.capability_probe import CAPABILITY_PROBE_FORMAT, derive_challenge
from menagerie.crawler.discovery import (
    DiscoveryError,
    FoundDiscovery,
    HigherTierDiscovery,
    NegativeDiscovery,
    RetryableToolFailureDiscovery,
    validate_source_discovery,
)
from menagerie.crawler.identity import hash_bytes, stable_hash, utc_now
from menagerie.crawler.models import JsonObject
from menagerie.crawler.constants import (
    AUTHOR_WALL_EXTERNAL_KILL_FACTOR,
    AUTHOR_WALL_SECONDS_ENV,
    resolve_author_wall_seconds,
)
from menagerie.crawler.source_broker import (
    BrokerPack,
    SourceBrokerError,
    broker_source_pack,
    default_transport,
    write_broker_outputs,
)

EXECUTOR_VERSION = "menagerie-author-executor 1.0.0"
RECEIPT_VERSION = "menagerie.crawler.author-executor-receipt.v1"
SUPPLEMENT_VERSION = "menagerie.crawler.author-supplement-request.v1"

# Operator protocol exit codes (identical to the lane's expectations).
EXIT_OK = 0
EXIT_PERMANENT = 64
EXIT_RETRYABLE = 75
EXIT_BACKOFF = 76
EXIT_UNAVAILABLE = 78

#: The verified pinned web-tool recipe. Every flag is load-bearing: without
#: ``--mcp-config`` Exa is absent, without ``--allowedTools`` every tool is
#: permission-blocked, and a name mismatch reads as unavailability -- all three
#: complete "successfully" while researching nothing.
EXA_MCP_CONFIG = '{"mcpServers":{"exa":{"type":"http","url":"https://mcp.exa.ai/mcp"}}}'
RESEARCH_TOOLS = (
    "WebSearch",
    "mcp__exa__web_search_exa",
    "mcp__exa__web_fetch_exa",
    "ToolSearch",
)


def stage_tool_rules(
    *,
    write_root: Path,
    read_roots: Sequence[Path],
) -> tuple[str, ...]:
    """Return the path-scoped ``--allowedTools`` rules for one session stage.

    **The per-attempt directory is the sole writable path.** Confinement is
    synchronous, not post-hoc: sessions carry no Bash, so the only file
    surface is the harness ``Read``/``Write``/``Edit`` tools, and those are
    granted as **path-scoped permission specifiers** rather than bare tool
    names. In ``-p`` mode an unmatched permission auto-denies, so an
    absolute-path write outside the attempt directory — ``..`` traversal, a
    sibling live attempt's directory, the authority root — is denied by the
    harness before bytes change. Rename/link escapes have no tool to ride
    (no Bash), and the executor remains the only publisher to any
    lane-visible path: staged model code, source packs, results, and probe
    evidence all leave the attempt directory only through executor-owned
    publication.

    The escape acceptance suite (Sol's six cases, adopted verbatim) lives in
    ``tests/test_author_executor_escapes.py``; if the harness rules do not
    deny every case before bytes change, the mechanism behind this ONE
    injection point moves down the ladder (write-only Seatbelt profile, then
    executor-owned file RPCs) without touching call sites.

    Parameters
    ----------
    write_root:
        The attempt directory — the only tree the session may write.
    read_roots:
        Directories the session may read (the model's author root; for stage
        2 also the frozen prompt directory the envelope names). Read scoping
        is a deferred residual by design; write scoping is the bar.
    """

    rules: list[str] = list(RESEARCH_TOOLS)
    for root in read_roots:
        rules.append(f"Read({Path(root)}/**)")
    root = Path(write_root)
    rules.append(f"Read({root}/**)")
    rules.append(f"Write({root}/**)")
    rules.append(f"Edit({root}/**)")
    return tuple(rules)

#: The external kill fires at grant +10%; the brief carries the deadline so a
#: session watching its clock can land a typed ``BLOCKED`` instead of a SIGKILL.
#: Both this factor and the per-campaign grants are single-sourced in
#: :mod:`menagerie.crawler.constants` so the driver's lane and this executor can
#: never disagree about one campaign's budget.
EXTERNAL_KILL_FACTOR = AUTHOR_WALL_EXTERNAL_KILL_FACTOR
_KILL_GRACE_SECONDS = 10.0

#: Structured harness signals that mean a provider usage pause. Free-text
#: marker scanning is deliberately absent: classification authority is the
#: parsed harness JSON only (SEAM_REDESIGN section 3.5).
_LIMIT_SUBTYPES = frozenset(
    {
        "error_rate_limit",
        "rate_limit",
        "usage_limit",
        "usage_limit_reached",
        "error_usage_limit",
    }
)
_LIMIT_ERROR_TYPES = frozenset({"rate_limit_error", "overloaded_error"})

_PROMPT_ROOT = Path(__file__).with_name("prompts")
_EXECUTOR_PROMPTS = _PROMPT_ROOT / "executor"
_POOL_PROMPTS = _PROMPT_ROOT / "pool"

_SOURCE_REQUEST_VERSION = "menagerie.crawler.author-source-request.v1"
_AUTHOR_ENVELOPE_PREFIX = "menagerie.crawler.author-envelope"


class AuthorExecutorError(RuntimeError):
    """Raised when a request cannot be served at all (typed unavailability)."""


@dataclass(frozen=True)
class ExecutorConfig:
    """Frozen executor configuration, resolved once per invocation.

    Parameters
    ----------
    claude_command:
        Harness command prefix (``MENAGERIE_AUTHOR_CLAUDE_BIN``, shlex-split).
    author_model:
        Optional ``--model`` value (``MENAGERIE_AUTHOR_MODEL``).
    campaign_id:
        Campaign identity (``MENAGERIE_CAMPAIGN_ID`` or ``--campaign``).
    wall_seconds_override:
        Optional wall-grant override (``MENAGERIE_AUTHOR_WALL_SECONDS``).
    pause_after:
        Test-only crash-injection hook (``MENAGERIE_EXECUTOR_PAUSE_AFTER``):
        the executor sleeps after the named phase so a harness test can
        ``kill -9`` it at an exact durable-state boundary. Never set in
        production.
    """

    claude_command: tuple[str, ...]
    author_model: Optional[str]
    campaign_id: Optional[str]
    wall_seconds_override: Optional[float]
    pause_after: Optional[str]

    @classmethod
    def from_env(cls, *, campaign_id: Optional[str] = None) -> "ExecutorConfig":
        """Resolve the configuration from the environment."""

        raw_bin = os.environ.get("MENAGERIE_AUTHOR_CLAUDE_BIN", "claude")
        command = tuple(shlex.split(raw_bin)) or ("claude",)
        raw_wall = os.environ.get(AUTHOR_WALL_SECONDS_ENV)
        return cls(
            claude_command=command,
            author_model=os.environ.get("MENAGERIE_AUTHOR_MODEL") or None,
            campaign_id=campaign_id or os.environ.get("MENAGERIE_CAMPAIGN_ID") or None,
            wall_seconds_override=float(raw_wall) if raw_wall else None,
            pause_after=os.environ.get("MENAGERIE_EXECUTOR_PAUSE_AFTER") or None,
        )

    def wall_seconds(self) -> float:
        """Return the campaign wall grant in seconds.

        Resolution is delegated so this executor and the driver's author lane
        read one table. A driver-published override always names the exact grant
        the lane also sized its stall bound from.
        """

        return resolve_author_wall_seconds(self.campaign_id, self.wall_seconds_override)


@dataclass(frozen=True)
class SessionOutcome:
    """One ``claude -p`` round trip, machine-observed.

    Parameters
    ----------
    harness:
        Parsed harness JSON from stdout, or ``None`` when unparseable.
    returncode:
        Harness process exit status (``-1`` after an external kill).
    timed_out:
        Whether the external wall kill fired.
    wall_seconds:
        Executor-observed session wall time (excludes broker time).
    session_id:
        Harness-reported session identity when available.
    argv:
        Exact argv, recorded for the attempt record.
    stderr_tail:
        Bounded stderr tail, recorded but never used for classification.
    """

    harness: Optional[JsonObject]
    returncode: int
    timed_out: bool
    wall_seconds: float
    session_id: Optional[str]
    argv: tuple[str, ...]
    stderr_tail: str


def run_claude_session(
    prompt: str,
    *,
    cwd: Path,
    wall_seconds: float,
    config: ExecutorConfig,
    write_root: Path,
    read_roots: Sequence[Path] = (),
    session_id: Optional[str] = None,
    resume: Optional[str] = None,
) -> SessionOutcome:
    """Run one headless harness session under the pinned recipe.

    This is the ONE spawn-call injection point for session confinement: the
    write mechanism (currently harness path-scoped permission rules, per
    :func:`stage_tool_rules`) is applied here and can be swapped for a
    write-only OS sandbox profile without touching call sites.

    Parameters
    ----------
    prompt:
        Complete rendered brief.
    cwd:
        Attempt scratch directory (the session's working directory).
    wall_seconds:
        Wall grant; the external kill fires at grant +10%.
    config:
        Frozen executor configuration.
    write_root:
        The attempt directory — the session's sole writable tree.
    read_roots:
        Additional readable roots for this stage.
    session_id:
        Pre-generated session identity (durable before launch) for stage 1.
    resume:
        Prior session to resume for stage 2 / supplement rounds.

    Returns
    -------
    SessionOutcome
        Machine-observed outcome; never raises for harness failures.
    """

    argv: list[str] = [
        *config.claude_command,
        "-p",
        prompt,
        "--setting-sources",
        "",
        "--mcp-config",
        EXA_MCP_CONFIG,
        "--allowedTools",
        *stage_tool_rules(write_root=write_root, read_roots=read_roots),
        "--output-format",
        "json",
    ]
    if config.author_model:
        argv.extend(["--model", config.author_model])
    if resume is not None:
        argv.extend(["--resume", resume])
    elif session_id is not None:
        argv.extend(["--session-id", session_id])
    started = time.monotonic()
    cwd.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        argv,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=wall_seconds * EXTERNAL_KILL_FACTOR)
    except subprocess.TimeoutExpired:
        timed_out = True
        stdout, stderr = _kill_session(process)
    wall = time.monotonic() - started
    harness = _parse_harness_json(stdout or "")
    reported = harness.get("session_id") if harness else None
    return SessionOutcome(
        harness=harness,
        returncode=process.returncode if process.returncode is not None else -1,
        timed_out=timed_out,
        wall_seconds=wall,
        session_id=str(reported) if reported else session_id,
        argv=tuple(argv),
        stderr_tail=(stderr or "")[-2000:],
    )


def _kill_session(process: subprocess.Popen) -> tuple[str, str]:
    """Terminate a session's process group: SIGTERM, grace, SIGKILL."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        return process.communicate(timeout=_KILL_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            return process.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return "", ""


def _parse_harness_json(stdout: str) -> Optional[JsonObject]:
    """Parse the harness JSON document from session stdout."""

    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def structured_limit_signal(harness: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the structured usage-limit signal, if the harness declared one.

    Only parsed harness JSON fields decide a quota exit. Free text never does.
    """

    if not harness:
        return None
    subtype = str(harness.get("subtype", ""))
    if subtype in _LIMIT_SUBTYPES:
        return subtype
    error = harness.get("error")
    if isinstance(error, Mapping) and str(error.get("type", "")) in _LIMIT_ERROR_TYPES:
        return str(error["type"])
    return None


def effort_from_session(outcome: SessionOutcome) -> JsonObject:
    """Extract the machine-observed effort record for one session.

    Every value is taken verbatim from the harness JSON or the executor's own
    clock; nothing is self-reported by the model.
    """

    harness = outcome.harness or {}
    usage = harness.get("usage")
    return {
        "wall_seconds_observed": round(outcome.wall_seconds, 3),
        "duration_ms": harness.get("duration_ms"),
        "duration_api_ms": harness.get("duration_api_ms"),
        "num_turns": harness.get("num_turns"),
        "total_cost_usd": harness.get("total_cost_usd"),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "timed_out": outcome.timed_out,
        "returncode": outcome.returncode,
    }


# -- brief rendering -------------------------------------------------------


def _read_prompt(root: Path, name: str) -> str:
    """Read one prompt fragment, failing typed when it is missing."""

    try:
        return (root / name).read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AuthorExecutorError(f"executor prompt {name} is unavailable: {exc}") from exc


def _campaign_fragment(campaign_id: Optional[str]) -> Optional[str]:
    """Return the campaign standards fragment, failing loudly on a bad id."""

    if campaign_id is None:
        return None
    path = _POOL_PROMPTS / f"campaign_{campaign_id}.md"
    if not path.is_file():
        raise AuthorExecutorError(
            f"campaign {campaign_id!r} has no standards fragment at {path}"
        )
    return path.read_text(encoding="utf-8").strip()


def _prior_attempts_section(author_root: Path) -> Optional[str]:
    """Render the WHAT WENT WRONG LAST TIME feedback block."""

    summaries = prior_attempts_summary(author_root)
    relevant = [
        summary
        for summary in summaries
        if summary.get("status") in ("failed", "superseded")
        or summary.get("failure_reason")
        or summary.get("checker_findings")
    ]
    if not relevant:
        return None
    lines = [
        "## WHAT WENT WRONG LAST TIME (binding feedback -- do not repeat it)",
        "",
    ]
    for summary in relevant:
        lines.append(
            f"- attempt {summary.get('attempt_number')} "
            f"({summary.get('kind')}, {summary.get('status')}): "
            f"stage={summary.get('failure_stage')!r} "
            f"reason={summary.get('failure_reason')!r}"
        )
        findings = summary.get("checker_findings")
        if findings:
            lines.append(f"  checker findings, verbatim: {json.dumps(findings)}")
    return "\n".join(lines)


def _deadline_iso(wall_seconds: float) -> str:
    """Return the wall deadline as an ISO instant the brief can carry."""

    deadline = datetime.now(timezone.utc) + timedelta(seconds=wall_seconds)
    return deadline.isoformat().replace("+00:00", "Z")


def _facts(lines: Sequence[str]) -> str:
    """Render the binding JOB FACTS block."""

    return "\n".join(["## JOB FACTS (binding)", "", *lines])


def render_stage1_brief(
    *,
    request_path: Path,
    request: Mapping[str, Any],
    attempt: AttemptHandle,
    config: ExecutorConfig,
    author_root: Path,
) -> str:
    """Render the complete stage-1 discovery brief."""

    wall = config.wall_seconds()
    facts = _facts(
        [
            f"- stable_id: `{request.get('stable_id')}`",
            f"- work_id: `{request.get('work_id')}`",
            f"- campaign: `{config.campaign_id}`",
            f"- attempt: `{attempt.number}` nonce `{attempt.nonce}`",
            f"- REQUEST envelope to read first: `{request_path}`",
            f"- DISCOVERY output path, exact: `{attempt.paths.directory / 'discovery.json'}`",
            f"- max_sources: {request.get('max_sources')}",
            f"- wall deadline: `{_deadline_iso(wall)}` (external kill at +10%)",
        ]
    )
    sections = [facts, _read_prompt(_EXECUTOR_PROMPTS, "stage1_discovery.md")]
    campaign = _campaign_fragment(config.campaign_id)
    if campaign is not None:
        sections.append(campaign)
    feedback = _prior_attempts_section(author_root)
    if feedback is not None:
        sections.append(feedback)
    return "\n\n---\n\n".join(sections) + "\n"


def render_stage2_brief(
    *,
    request_path: Path,
    request: Mapping[str, Any],
    attempt: AttemptHandle,
    config: ExecutorConfig,
    author_root: Path,
    cold_start_discovery: Optional[str] = None,
) -> str:
    """Render the complete stage-2 authoring brief."""

    wall = config.wall_seconds()
    facts = _facts(
        [
            f"- stable_id: `{request.get('stable_id')}`",
            f"- work_id: `{request.get('work_id')}`",
            f"- campaign: `{config.campaign_id}`",
            f"- attempt: `{attempt.number}` nonce `{attempt.nonce}`",
            f"- REQUEST envelope to read first: `{request_path}`",
            f"- RESULT output path, exact: `{attempt.paths.directory / 'result.json'}`",
            f"- STAGED MODEL dir, exact: `{attempt.paths.directory / 'model'}`",
            f"- envelope allowed_model_dir (executor-mirrored at publication): "
            f"`{request.get('allowed_model_dir')}`",
            f"- wall deadline: `{_deadline_iso(wall)}` (external kill at +10%)",
        ]
    )
    sections = [facts, _read_prompt(_EXECUTOR_PROMPTS, "stage2_author.md")]
    campaign = _campaign_fragment(config.campaign_id)
    if campaign is not None:
        sections.append(campaign)
    if cold_start_discovery is not None:
        sections.append(
            "## COLD START -- the stage-1 session was lost\n\n"
            "The recorded stage-1 discovery output follows verbatim:\n\n"
            "```json\n" + cold_start_discovery + "\n```"
        )
    feedback = _prior_attempts_section(author_root)
    if feedback is not None:
        sections.append(feedback)
    return "\n\n---\n\n".join(sections) + "\n"


# -- publication -----------------------------------------------------------


def _publish_bytes(
    local_path: Path,
    required_path: Path,
    *,
    nonce: str,
    kind: str,
    stable_id: Optional[str] = None,
    attempt_number: Optional[int] = None,
) -> str:
    """Atomically publish one executor-local artifact and print its receipt.

    The executor is the only publisher: sessions write only their own attempt
    (or probe) directory, and this function moves bytes across that boundary.
    The published bytes are re-read and digest-verified, and the nonce-bound
    receipt is printed on stdout for the lane to verify again.

    Returns
    -------
    str
        The published digest.
    """

    data = local_path.read_bytes()
    digest = hash_bytes(data)
    required_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = required_path.with_name(f".{required_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, required_path)
    finally:
        temporary.unlink(missing_ok=True)
    published = hash_bytes(required_path.read_bytes())
    if published != digest:
        raise AuthorExecutorError(
            f"published bytes at {required_path} do not match the attempt artifact"
        )
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "kind": kind,
        "stable_id": stable_id,
        "attempt_number": attempt_number,
        "attempt_nonce": nonce,
        "result_sha256": digest,
        "published_path": str(required_path),
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return digest


def _publish(
    attempt: AttemptHandle,
    local_path: Path,
    required_path: Path,
    *,
    kind: str,
) -> str:
    """Publish one attempt-local artifact, binding the attempt record to it."""

    digest = _publish_bytes(
        local_path,
        required_path,
        nonce=attempt.nonce,
        kind=kind,
        stable_id=str(attempt.record.get("stable_id") or "") or None,
        attempt_number=attempt.number,
    )
    attempt.update(
        published={
            "kind": kind,
            "path": str(required_path),
            "sha256": digest,
            "published_at": utc_now(),
        }
    )
    attempt.event("published", kind=kind, path=str(required_path), sha256=digest)
    return digest


def _mirror_staged_model(attempt: AttemptHandle, model_dir: Path) -> None:
    """Mirror the attempt's staged ``model/`` tree into the envelope's model dir.

    Sessions never write the shared model directory (the attempt directory is
    the sole writable path); the executor performs this copy as part of
    publication, so cross-attempt contamination of staged code is
    structurally impossible.
    """

    staged = attempt.paths.directory / "model"
    if not staged.is_dir():
        return
    import shutil

    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(staged, model_dir, dirs_exist_ok=True)
    attempt.event("staged-model-mirrored", destination=str(model_dir))


def _pause_hook(config: ExecutorConfig, phase: str) -> None:
    """Test-only crash-injection point: sleep after a durable-state boundary."""

    if config.pause_after == phase:
        time.sleep(600.0)


def _fail(
    attempt: AttemptHandle,
    *,
    stage: str,
    reason: str,
    exit_code: int,
    detail: JsonObject | None = None,
) -> tuple[int, str]:
    """Record one typed attempt failure and return its exit."""

    attempt.update(
        status="failed",
        outcome={
            "kind": "failure",
            "failure_stage": stage,
            "failure_reason": reason,
            "detail": detail or {},
        },
    )
    attempt.event("failed", stage=stage, reason=reason)
    return exit_code, f"author executor {stage} failed: {reason} (attempt {attempt.nonce})"


# -- source-request (stage 1 + broker) -------------------------------------


def serve_source_request(
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Serve one stage-1 discovery round: session, broker, publication."""

    author_root = request_path.parent
    stable_id = str(request.get("stable_id", ""))
    required_output = Path(str(request.get("required_output_path", "")))
    if not stable_id or not str(required_output):
        return EXIT_PERMANENT, "source request lacks stable_id or required_output_path"
    max_sources = int(request.get("max_sources", 20) or 20)

    resumable = latest_attempt(
        author_root, stable_id=stable_id, statuses=frozenset({"stage1-complete"})
    )
    if resumable is not None:
        # Crash recovery: stage 1 already completed durably; replay the broker
        # from the recorded discovery output without re-running the session.
        resumable.event("resumed-broker-from-record")
        return _complete_source_discovery(
            resumable,
            request=request,
            required_output=required_output,
            max_sources=max_sources,
            config=config,
        )

    attempt = new_attempt(
        author_root,
        stable_id=stable_id,
        campaign_id=config.campaign_id,
        kind="source-request",
    )
    brief = render_stage1_brief(
        request_path=request_path,
        request=request,
        attempt=attempt,
        config=config,
        author_root=author_root,
    )
    session_id = str(uuid.uuid4())
    attempt.update(
        status="stage1-running",
        stage1={"session_id": session_id, "started_at": utc_now()},
    )
    outcome = run_claude_session(
        brief,
        cwd=attempt.paths.scratch,
        wall_seconds=config.wall_seconds(),
        config=config,
        write_root=attempt.paths.directory,
        read_roots=[author_root],
        session_id=session_id,
    )
    stage1 = {
        "session_id": outcome.session_id or session_id,
        "started_at": attempt.record["stage1"]["started_at"],
        "completed_at": utc_now(),
        "effort": effort_from_session(outcome),
        "argv": list(outcome.argv),
        "stderr_tail": outcome.stderr_tail,
    }
    if outcome.timed_out:
        attempt.update(stage1=stage1)
        return _fail(
            attempt, stage="stage1", reason="wall-exceeded", exit_code=EXIT_RETRYABLE
        )
    limit = structured_limit_signal(outcome.harness)
    if limit is not None:
        attempt.update(stage1=stage1)
        _fail(attempt, stage="stage1", reason="provider-usage-pause", exit_code=EXIT_BACKOFF)
        return EXIT_BACKOFF, f"author provider pause (structured signal {limit})"
    if outcome.returncode != 0:
        attempt.update(stage1=stage1)
        return _fail(
            attempt, stage="stage1", reason="session-crashed", exit_code=EXIT_RETRYABLE
        )

    discovery_path = attempt.paths.directory / "discovery.json"
    if _read_json_file(discovery_path) is None:
        attempt.update(stage1=stage1)
        return _fail(
            attempt, stage="stage1", reason="no-discovery-output", exit_code=EXIT_RETRYABLE
        )
    attempt.update(status="stage1-complete", stage1=stage1, discovery_path=str(discovery_path))
    _pause_hook(config, "stage1")
    return _complete_source_discovery(
        attempt,
        request=request,
        required_output=required_output,
        max_sources=max_sources,
        config=config,
    )


def _complete_source_discovery(
    attempt: AttemptHandle,
    *,
    request: Mapping[str, Any],
    required_output: Path,
    max_sources: int,
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Validate one recorded envelope and publish its governed machine product.

    Parameters
    ----------
    attempt:
        Attempt carrying the exact stage-1 discovery file.
    request:
        Machine source request supplying stable/work binding authority.
    required_output:
        Lane-visible publication path.
    max_sources:
        Fetch-target ceiling.
    config:
        Executor configuration supplying broker collaborators.

    Returns
    -------
    tuple[int, str]
        Operator exit and diagnostic.
    """

    raw = _read_json_file(attempt.paths.directory / "discovery.json")
    if raw is None:
        return _fail(
            attempt,
            stage="stage1",
            reason="no-discovery-output",
            exit_code=EXIT_RETRYABLE,
        )
    try:
        discovery = validate_source_discovery(
            raw,
            stable_id=str(request.get("stable_id", "")),
            work_id=str(request.get("work_id", "")),
        )
    except DiscoveryError as exc:
        return _fail(
            attempt,
            stage="stage1",
            reason="discovery-contract-invalid",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    attempt.event("stage1-complete", arm=str(raw["arm"]))
    if isinstance(discovery, RetryableToolFailureDiscovery):
        return _fail(
            attempt,
            stage="stage1",
            reason="research-tools-unavailable",
            exit_code=EXIT_RETRYABLE,
            detail={
                "tool_name": discovery.tool_name,
                "tool_spelling": discovery.tool_spelling,
                "error": discovery.error,
            },
        )
    if isinstance(discovery, (NegativeDiscovery, HigherTierDiscovery)):
        local = attempt.paths.directory / "discovery.json"
        _publish(attempt, local, required_output, kind="source-discovery")
        attempt.update(
            status="completed",
            outcome={"kind": "discovery-published", "arm": str(raw["arm"])},
        )
        return EXIT_OK, f"source discovery published for {request.get('stable_id')}"
    return _broker_and_publish(
        attempt,
        discovery=discovery,
        request=request,
        required_output=required_output,
        max_sources=max_sources,
        config=config,
    )


def _broker_and_publish(
    attempt: AttemptHandle,
    *,
    discovery: FoundDiscovery,
    request: Mapping[str, Any],
    required_output: Path,
    max_sources: int,
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Run the broker over a recorded FOUND discovery and publish the pack."""

    try:
        pack = broker_source_pack(
            [descriptor.to_mapping() for descriptor in discovery.descriptors],
            broker_dir=attempt.paths.broker,
            transport=default_transport(),
        )
    except SourceBrokerError as exc:
        return _fail(
            attempt,
            stage="broker",
            reason="descriptor-rejected",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    write_broker_outputs(pack, attempt.paths.broker)
    attempt.update(
        broker={
            "targets": len(pack.outcomes),
            "total_bytes": pack.total_bytes,
            "outcomes": {item.source_id: item.outcome for item in pack.outcomes},
            "receipts_path": str(attempt.paths.broker / "receipts.json"),
        }
    )
    if not pack.implementation_rows():
        return _fail(
            attempt,
            stage="broker",
            reason="primary-implementation-unfetchable",
            exit_code=EXIT_RETRYABLE,
            detail={"outcomes": [item.to_dict() for item in pack.outcomes]},
        )
    rows = _capped_rows(pack, max_sources)
    local = attempt.paths.directory / "source-targets.json"
    payload = {
        **pack.to_dict(),
        "sources": rows,
        "discovery": discovery.raw_result,
        "discovery_sha256": stable_hash(discovery.raw_result),
    }
    local.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _pause_hook(config, "broker")
    _publish(attempt, local, required_output, kind="source-targets")
    attempt.update(status="sources-published", outcome={"kind": "sources-published"})
    return EXIT_OK, f"source pack published for {request.get('stable_id')}"


def _capped_rows(pack: BrokerPack, max_sources: int) -> list[JsonObject]:
    """Return implementation-first rows, capped at the lane's fetch grant."""

    implementation = pack.implementation_rows()
    rest = [row for row in pack.rows if row not in implementation]
    return (implementation + rest)[:max_sources]


# -- author (stage 2, resume, supplement) ----------------------------------


def serve_author(
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Serve one stage-2 authoring round on the retained stage-1 session."""

    author_root = request_path.parent
    stable_id = str(request.get("stable_id", ""))
    required_output = Path(str(request.get("required_output_path", "")))
    if not stable_id or not str(required_output):
        return EXIT_PERMANENT, "author envelope lacks stable_id or required_output_path"

    attempt, resume_session, cold_discovery = _stage2_attempt(author_root, stable_id, config)
    brief = render_stage2_brief(
        request_path=request_path,
        request=request,
        attempt=attempt,
        config=config,
        author_root=author_root,
        cold_start_discovery=cold_discovery,
    )
    attempt.update(
        status="stage2-running",
        stage2={
            "started_at": utc_now(),
            "resumed_from": resume_session,
            "cold_start_reason": None if resume_session else "no-recoverable-stage1-session",
        },
    )
    stage2_read_roots = _stage2_read_roots(author_root, request)
    outcome = run_claude_session(
        brief,
        cwd=attempt.paths.scratch,
        wall_seconds=config.wall_seconds(),
        config=config,
        write_root=attempt.paths.directory,
        read_roots=stage2_read_roots,
        resume=resume_session,
    )
    if resume_session is not None and outcome.returncode != 0 and not outcome.timed_out:
        limit = structured_limit_signal(outcome.harness)
        if limit is None and not (attempt.paths.directory / "result.json").is_file():
            # Provider resume failure: record it and rerun once, cold, with the
            # recorded stage-1 discovery inlined (SEAM_REDESIGN 3.1 step 4).
            attempt.event(
                "resume-failed",
                returncode=outcome.returncode,
                resumed_from=resume_session,
            )
            stage2 = dict(attempt.record.get("stage2") or {})
            stage2["cold_start_reason"] = f"resume-exit-{outcome.returncode}"
            attempt.update(stage2=stage2)
            cold_brief = render_stage2_brief(
                request_path=request_path,
                request=request,
                attempt=attempt,
                config=config,
                author_root=author_root,
                cold_start_discovery=_recorded_discovery_text(attempt),
            )
            outcome = run_claude_session(
                cold_brief,
                cwd=attempt.paths.scratch,
                wall_seconds=config.wall_seconds(),
                config=config,
                write_root=attempt.paths.directory,
                read_roots=stage2_read_roots,
            )
    stage2 = dict(attempt.record.get("stage2") or {})
    stage2.update(
        {
            "session_id": outcome.session_id,
            "completed_at": utc_now(),
            "effort": effort_from_session(outcome),
            "argv": list(outcome.argv),
            "stderr_tail": outcome.stderr_tail,
        }
    )
    attempt.update(stage2=stage2)
    if outcome.timed_out:
        # The provider session may still be live; the attempt is superseded so
        # any late output it writes is quarantined, never read as a result.
        attempt.event("stage2-timeout-superseding")
        attempt.mark_superseded(by_nonce="", reason="timeout")
        return (
            EXIT_RETRYABLE,
            f"author executor stage2 failed: wall-exceeded (attempt {attempt.nonce})",
        )
    limit = structured_limit_signal(outcome.harness)
    if limit is not None:
        _fail(attempt, stage="stage2", reason="provider-usage-pause", exit_code=EXIT_BACKOFF)
        return EXIT_BACKOFF, f"author provider pause (structured signal {limit})"
    if outcome.returncode != 0:
        return _fail(
            attempt, stage="stage2", reason="session-crashed", exit_code=EXIT_RETRYABLE
        )
    _pause_hook(config, "stage2")

    result_path = attempt.paths.directory / "result.json"
    if not result_path.is_file():
        supplement_exit = _maybe_supplement_round(
            attempt, request_path, request, config, outcome
        )
        if supplement_exit is not None:
            return supplement_exit
    if not result_path.is_file():
        return _fail(
            attempt, stage="stage2", reason="no-result-output", exit_code=EXIT_RETRYABLE
        )
    parsed = _read_json_file(result_path)
    if parsed is None or "kind" not in parsed:
        return _fail(
            attempt, stage="stage2", reason="result-not-json", exit_code=EXIT_RETRYABLE
        )
    model_dir = str(request.get("allowed_model_dir") or "")
    if model_dir:
        _mirror_staged_model(attempt, Path(model_dir))
    _publish(attempt, result_path, required_output, kind="author-result")
    attempt.update(status="completed", outcome={"kind": "result-published"})
    return EXIT_OK, f"author result published for {stable_id}"


def _stage2_attempt(
    author_root: Path,
    stable_id: str,
    config: ExecutorConfig,
) -> tuple[AttemptHandle, Optional[str], Optional[str]]:
    """Select or open the attempt stage 2 runs in.

    Returns
    -------
    tuple[AttemptHandle, str | None, str | None]
        The live attempt, the stage-1 session to resume (when recoverable),
        and the recorded discovery text for a cold start (when needed).
    """

    # ``stage1-complete`` is reachable when the executor died between
    # publishing the source pack and recording ``sources-published``; the lane
    # only dispatches the author round after consuming a published pack, so
    # both states mean "stage 1 is durable, resume its session in place".
    ready = latest_attempt(
        author_root,
        stable_id=stable_id,
        statuses=frozenset({"sources-published", "stage1-complete"}),
    )
    if ready is not None:
        return ready, _stage1_session(ready.record), None
    # No in-place attempt. A killed or timed-out stage 2 never reuses its
    # directory — the orphaned provider session may still write there — but
    # its recorded stage-1 session identity is still the cheapest context we
    # have, so the newest attempt that recorded one donates it. Only when no
    # attempt ever recorded a stage-1 session (provably unrecoverable) does
    # stage 2 run cold.
    donor: Optional[AttemptHandle] = None
    for handle in reversed(
        [
            candidate
            for candidate in list_attempts(author_root)
            if candidate.record.get("stable_id") == stable_id
        ]
    ):
        if _stage1_session(handle.record) is not None:
            donor = handle
            break
    fresh = new_attempt(
        author_root,
        stable_id=stable_id,
        campaign_id=config.campaign_id,
        kind="author",
        inherit_from=donor.record if donor is not None else None,
    )
    if donor is not None:
        return fresh, _stage1_session(donor.record), None
    return fresh, None, None


def _stage2_read_roots(author_root: Path, request: Mapping[str, Any]) -> list[Path]:
    """Return stage-2 readable roots: author root + the envelope's prompt dir.

    The envelope's ``allowed_model_dir`` is deliberately NOT writable by the
    session: staged code goes under the attempt's ``model/`` tree and the
    executor mirrors it into the envelope's model dir at publication, so the
    per-attempt directory stays the sole writable path.
    """

    read_roots: list[Path] = [author_root]
    prompt = request.get("prompt")
    if isinstance(prompt, Mapping) and prompt.get("path"):
        read_roots.append(Path(str(prompt["path"])).parent)
    return read_roots


def _stage1_session(record: Mapping[str, Any]) -> Optional[str]:
    """Return the recorded stage-1 session identity, walking inheritance."""

    stage1 = record.get("stage1")
    if isinstance(stage1, Mapping) and stage1.get("session_id"):
        return str(stage1["session_id"])
    inherited = record.get("inherited")
    if isinstance(inherited, Mapping):
        stage1 = inherited.get("stage1")
        if isinstance(stage1, Mapping) and stage1.get("session_id"):
            return str(stage1["session_id"])
    return None


def _recorded_discovery_text(attempt: AttemptHandle) -> Optional[str]:
    """Return the recorded discovery JSON text for cold-start inlining."""

    for candidate in (
        attempt.paths.directory / "discovery.json",
        Path(str((attempt.record.get("inherited") or {}).get("discovery_path") or "")),
    ):
        if candidate and candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError:
                continue
    return None


def _maybe_supplement_round(
    attempt: AttemptHandle,
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
    stage2_outcome: SessionOutcome,
) -> Optional[tuple[int, str]]:
    """Grant at most one typed supplementary broker round, then resume once.

    Returns
    -------
    tuple[int, str] | None
        A typed exit when the supplement round itself failed, else ``None``
        (the caller re-checks for the result).
    """

    supplement_path = attempt.paths.directory / "supplement-request.json"
    scratch_supplement = attempt.paths.scratch / "supplement-request.json"
    if not supplement_path.is_file() and scratch_supplement.is_file():
        supplement_path = scratch_supplement
    if not supplement_path.is_file():
        return None
    if attempt.record.get("supplement") is not None:
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-round-already-consumed",
            exit_code=EXIT_RETRYABLE,
        )
    parsed = _read_json_file(supplement_path)
    sources = parsed.get("sources") if parsed else None
    if not isinstance(sources, list) or not sources:
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-request-untyped",
            exit_code=EXIT_RETRYABLE,
        )
    attempt.update(status="supplement-running", supplement={"requested_at": utc_now()})
    try:
        pack = broker_source_pack(
            sources,
            broker_dir=attempt.paths.broker / "supplement",
            transport=default_transport(),
        )
    except SourceBrokerError as exc:
        return _fail(
            attempt,
            stage="supplement",
            reason="descriptor-rejected",
            exit_code=EXIT_RETRYABLE,
            detail={"error": str(exc)},
        )
    write_broker_outputs(pack, attempt.paths.broker / "supplement")
    manifest_path = attempt.paths.directory / "supplement-manifest.json"
    manifest_path.write_text(
        json.dumps(pack.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    continuation = "\n".join(
        [
            "## SUPPLEMENTARY SOURCE ROUND (the only one)",
            "",
            f"- supplementary manifest, exact: `{manifest_path}`",
            f"- RESULT output path, exact: `{attempt.paths.directory / 'result.json'}`",
            "",
            "The broker fetched your supplementary request; per-target outcomes and",
            "receipts are in the manifest. Write the complete result now -- no further",
            "rounds are granted.",
        ]
    )
    outcome = run_claude_session(
        continuation,
        cwd=attempt.paths.scratch,
        wall_seconds=config.wall_seconds() / 2,
        config=config,
        write_root=attempt.paths.directory,
        read_roots=_stage2_read_roots(request_path.parent, request),
        resume=stage2_outcome.session_id,
    )
    attempt.update(
        supplement={
            "requested_at": attempt.record["supplement"]["requested_at"],
            "manifest_path": str(manifest_path),
            "session_id": outcome.session_id,
            "effort": effort_from_session(outcome),
            "completed_at": utc_now(),
        },
        status="stage2-running",
    )
    if outcome.timed_out or outcome.returncode != 0:
        return _fail(
            attempt,
            stage="supplement",
            reason="supplement-session-failed",
            exit_code=EXIT_RETRYABLE,
        )
    return None


# -- capability probe ------------------------------------------------------


def serve_capability_probe(
    request_path: Path,
    request: Mapping[str, Any],
    config: ExecutorConfig,
) -> tuple[int, str]:
    """Serve the doctor's live web-tools probe through the production path."""

    nonce = str(request.get("nonce", ""))
    required_output = Path(str(request.get("required_output_path", "")))
    if not nonce or not str(required_output):
        return EXIT_PERMANENT, "capability probe lacks nonce or required_output_path"
    deadline = float(request.get("deadline_seconds", 120) or 120)
    challenge = derive_challenge(nonce)
    probe_root = request_path.parent / f"executor-probe-{nonce[:12]}"
    evidence_path = probe_root / "evidence.json"
    facts = _facts(
        [
            f"- probe nonce: `{nonce}`",
            f"- challenge package: `{challenge.package}`",
            f"- challenge metadata URL: `{challenge.metadata_url}`",
            f"- challenge project URL: `{challenge.project_url}`",
            f"- challenge_id: `{challenge.challenge_id}`",
            f"- REQUIRED output path, exact: `{evidence_path}`",
            f"- deadline: {deadline:g}s from dispatch",
        ]
    )
    brief = "\n\n---\n\n".join(
        [facts, _read_prompt(_POOL_PROMPTS, "stage_capability_probe.md")]
    )
    outcome = run_claude_session(
        brief,
        cwd=probe_root / "scratch",
        wall_seconds=deadline,
        config=config,
        write_root=probe_root,
        read_roots=[request_path.parent],
        session_id=str(uuid.uuid4()),
    )
    limit = structured_limit_signal(outcome.harness)
    if limit is not None:
        return EXIT_BACKOFF, f"author provider pause (structured signal {limit})"
    if outcome.timed_out or outcome.returncode != 0:
        return EXIT_RETRYABLE, "capability probe session did not complete"
    if not evidence_path.is_file():
        return EXIT_RETRYABLE, "capability probe session published no evidence"
    # The session writes only its probe root; the executor publishes to the
    # doctor's required path and prints the nonce-bound receipt.
    _publish_bytes(evidence_path, required_output, nonce=nonce, kind="capability-probe")
    return EXIT_OK, f"capability probe evidence published for nonce {nonce[:12]}"


# -- CLI -------------------------------------------------------------------


def classify_request(request: Mapping[str, Any]) -> str:
    """Return the request kind one envelope asks for.

    Mirrors the operator wrapper's closed classification.
    """

    if str(request.get("format", "")) == CAPABILITY_PROBE_FORMAT:
        return "capability-probe"
    version = str(request.get("envelope_version", ""))
    if version == _SOURCE_REQUEST_VERSION:
        return "source-request"
    if version.startswith(_AUTHOR_ENVELOPE_PREFIX):
        return "author"
    raise AuthorExecutorError(
        f"request is neither a capability probe nor an author envelope: "
        f"format={request.get('format')!r} envelope_version={version!r}"
    )


def _read_json_file(path: Path) -> Optional[JsonObject]:
    """Read one JSON object, returning ``None`` when absent or invalid."""

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_parser() -> argparse.ArgumentParser:
    """Build the executor CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.author_executor",
        description="Serve one author request headless via claude -p.",
    )
    parser.add_argument("--version", action="store_true", help="print the executor version")
    parser.add_argument("--campaign", default=None, help="campaign identity")
    parser.add_argument("request", nargs="?", default=None, help="absolute request path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the author executor.

    Returns
    -------
    int
        Operator protocol exit status.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.version:
        print(EXECUTOR_VERSION)
        return EXIT_OK
    if args.request is None:
        print("author executor requires exactly one request path", file=sys.stderr)
        return EXIT_PERMANENT
    request_path = Path(args.request).expanduser().resolve()
    request = _read_json_file(request_path)
    if request is None:
        print(f"author request at {request_path} is not a JSON object", file=sys.stderr)
        return EXIT_PERMANENT
    config = ExecutorConfig.from_env(campaign_id=args.campaign or request.get("campaign_id"))
    try:
        kind = classify_request(request)
        if kind == "capability-probe":
            code, detail = serve_capability_probe(request_path, request, config)
        elif kind == "source-request":
            code, detail = serve_source_request(request_path, request, config)
        else:
            code, detail = serve_author(request_path, request, config)
    except AuthorExecutorError as exc:
        print(f"author executor error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except OSError as exc:
        print(f"author executor io error: {exc}", file=sys.stderr)
        return EXIT_RETRYABLE
    if code != EXIT_OK:
        print(detail, file=sys.stderr)
    return code


def wrapper_command(*, campaign_id: Optional[str] = None) -> str:
    """Return the shell-quoted ``MENAGERIE_AUTHOR_COMMAND`` for this executor.

    Pair it with ``MENAGERIE_AUTHOR_REQUIRE_RECEIPT=1`` so the lane demands
    the executor's attempt-bound publication receipt on every round trip.
    """

    argv = [sys.executable, "-m", "menagerie.crawler.author_executor"]
    if campaign_id:
        argv.extend(["--campaign", campaign_id])
    return shlex.join(argv)


if __name__ == "__main__":  # pragma: no cover -- executor entry point
    raise SystemExit(main())
