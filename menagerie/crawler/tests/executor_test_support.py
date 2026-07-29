"""Shared support for author-executor tests: fake harness, fixtures, envelopes.

The fake ``claude`` binary honors the executor's real invocation contract: it
parses ``-p``/``--session-id``/``--resume``, extracts its output paths from the
rendered brief exactly as a real session would, records every invocation, and
prints a harness-shaped JSON result. Behavior is selected by environment
variables so kill/timeout/limit scenarios are scriptable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RESOLVED_SHA = "a" * 39 + "b"
COMMITS_URL = "https://api.github.com/repos/acme/widgets/commits/v1.0.0"
RAW_URL = f"https://raw.githubusercontent.com/acme/widgets/{RESOLVED_SHA}/models/net.py"
IMPL_BODY = "class Net:\n    pass\n"
SUPPLEMENT_URL = "https://example.org/extra.txt"

DEFAULT_DISCOVERY = {
    "arm": "FOUND",
    "sources": [
        {
            "source_id": "impl-net",
            "kind": "forge-file",
            "repo": "github.com/acme/widgets",
            "path": "models/net.py",
            "ref": "v1.0.0",
            "requested_role": "implementation",
            "media_type_hint": "text/x-python",
            "basis": "The fixture repository owns the model implementation.",
        }
    ],
}

DEFAULT_RESULT = {
    "kind": "BLOCKED",
    "payload": {
        "stage": "source",
        "reason_code": "missing-material-source",
        "prerequisite_ids": ["source-needed"],
        "evidence_ids": ["evidence-gap"],
    },
}

_FAKE_CLAUDE_SOURCE = r'''
import json, os, re, sys, time, uuid

argv = sys.argv[1:]


def flag(name):
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return None


prompt = flag("-p") or ""
mode = os.environ.get("FAKE_CLAUDE_MODE", "happy")


def extract(label):
    matched = re.search(re.escape(label) + r"[^`]*`([^`]+)`", prompt)
    return matched.group(1) if matched else None


discovery_path = extract("- DISCOVERY output path, exact:")
result_path = extract("- RESULT output path, exact:")
required_path = extract("- REQUIRED output path, exact:")
if discovery_path:
    stage = "stage1"
elif "SUPPLEMENTARY SOURCE ROUND" in prompt:
    stage = "supplement"
elif result_path:
    stage = "stage2"
elif required_path:
    stage = "probe"
else:
    stage = "unknown"

log_dir = os.environ.get("FAKE_CLAUDE_LOG")
if log_dir:
    with open(os.path.join(log_dir, "invocations.jsonl"), "a") as fh:
        fh.write(
            json.dumps(
                {"pid": os.getpid(), "stage": stage, "argv": argv, "prompt": prompt}
            )
            + "\n"
        )
marker_dir = os.environ.get("FAKE_CLAUDE_MARKER")
if marker_dir:
    with open(os.path.join(marker_dir, stage + ".started"), "w") as fh:
        fh.write(str(os.getpid()))


def emit_harness(rc=0):
    sid = flag("--session-id") or str(uuid.uuid4())
    print(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 1234,
                "duration_api_ms": 987,
                "num_turns": 7,
                "session_id": sid,
                "total_cost_usd": 0.0123,
                "usage": {
                    "input_tokens": 111,
                    "output_tokens": 222,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20,
                },
                "result": "done",
            }
        )
    )
    sys.exit(rc)


sleep_s = float(os.environ.get("FAKE_CLAUDE_SLEEP", "120"))

if mode == "sleep-" + stage:
    time.sleep(sleep_s)
    sys.exit(0)
if mode == "late-write-stage2" and stage == "stage2":
    time.sleep(sleep_s)
    if result_path:
        with open(result_path, "w") as fh:
            fh.write(os.environ.get("FAKE_CLAUDE_LATE_RESULT", '{"kind": "STALE"}'))
    sys.exit(0)
if mode == "limit":
    # Real Claude Code result envelopes, not invented ones. FAKE_CLAUDE_LIMIT_SHAPE
    # selects which observed shape to replay; see _LIMIT_TERMINAL_REASONS in
    # author_executor.py for the citation of every field used here.
    shape = os.environ.get("FAKE_CLAUDE_LIMIT_SHAPE", "blocking_limit")
    payload = {
        "type": "result",
        "is_error": True,
        "session_id": str(uuid.uuid4()),
        "duration_ms": 1234,
        "num_turns": 1,
        "total_cost_usd": 0.0,
    }
    if shape == "blocking_limit":
        payload.update(subtype="error_during_execution", terminal_reason="blocking_limit")
    elif shape == "rapid_refill_breaker":
        payload.update(
            subtype="error_during_execution", terminal_reason="rapid_refill_breaker"
        )
    elif shape in ("five_hour", "seven_day"):
        payload.update(
            subtype="error_during_execution",
            terminal_reason="blocking_limit",
            rate_limit_info={
                "status": "rejected",
                "rateLimitType": shape,
                "resetsAt": int(os.environ["FAKE_CLAUDE_LIMIT_RESETS_AT"]),
            },
        )
    elif shape == "rate_limit_info_only":
        payload.update(
            subtype="error_during_execution",
            rate_limit_info={"status": "rejected", "rateLimitType": "seven_day"},
        )
    elif shape == "api_error_status":
        payload.update(subtype="success", api_error_status=429, result="")
    elif shape == "raw_api_error":
        payload = {"type": "error", "error": {"type": "rate_limit_error"}}
    elif shape == "generic_crash":
        # The generic failure subtype with NO limit terminal reason: an ordinary
        # session error that must stay a retry, never a campaign-wide pause.
        payload.update(subtype="error_during_execution", terminal_reason="model_error")
    else:
        raise SystemExit(f"unknown FAKE_CLAUDE_LIMIT_SHAPE {shape!r}")
    print(json.dumps(payload))
    sys.exit(1)
if mode == "crash":
    print("boom, not json")
    print("GitHub API rate limit exceeded for 1.2.3.4", file=sys.stderr)
    sys.exit(3)
if mode == "resume-fail" and "--resume" in argv:
    sys.exit(5)

if stage == "stage1" and discovery_path:
    payload = os.environ.get("FAKE_CLAUDE_DISCOVERY")
    if not payload:
        discovery = json.loads(os.environ["FAKE_CLAUDE_DEFAULT_DISCOVERY"])
        payload = json.dumps(discovery)
    with open(discovery_path, "w") as fh:
        fh.write(payload)
elif stage == "stage2" and result_path:
    if mode == "supplement-request":
        supp = os.path.join(os.path.dirname(result_path), "supplement-request.json")
        with open(supp, "w") as fh:
            fh.write(
                json.dumps(
                    {
                        "sources": [
                            {
                                "source_id": "supp-1",
                                "kind": "raw-url",
                                "url": os.environ["FAKE_CLAUDE_SUPPLEMENT_URL"],
                                "requested_role": "documentation",
                                "basis": "Supplementary fixture documentation.",
                            }
                        ],
                        "why": "one more file",
                    }
                )
            )
    else:
        with open(result_path, "w") as fh:
            fh.write(
                os.environ.get("FAKE_CLAUDE_RESULT")
                or os.environ["FAKE_CLAUDE_DEFAULT_RESULT"]
            )
elif stage == "supplement" and result_path:
    with open(result_path, "w") as fh:
        fh.write(
            os.environ.get("FAKE_CLAUDE_RESULT")
            or os.environ["FAKE_CLAUDE_DEFAULT_RESULT"]
        )
elif stage == "probe" and required_path:
    # Default: evidence shaped like a session that GENUINELY exercised all
    # three tools, derived from the prompt's own challenge facts so the
    # executor-side proof (nonce echo, live URLs, digest consistency,
    # version corroboration) can pass. FAKE_CLAUDE_PROBE=hollow writes the
    # evidence of a session that researched nothing, which the executor must
    # REFUSE to turn into a receipt.
    if os.environ.get("FAKE_CLAUDE_PROBE") == "hollow":
        evidence = {"challenge_id": "test", "tools": {}}
    else:
        import hashlib
        from datetime import datetime, timezone

        package = extract("- challenge package:") or "requests"
        metadata_url = extract("- challenge metadata URL:") or ""
        challenge_id = extract("- challenge_id:") or ""
        nonce_fact = extract("- probe nonce:") or ""
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        version = "9.9.9"
        serial = 20260728
        content = json.dumps(
            {"info": {"version": version}, "last_serial": serial, "pad": "x" * 900}
        )
        common = {"nonce": nonce_fact, "observed_at": now_iso}
        evidence = {
            "challenge_id": challenge_id,
            "tools": {
                "WebSearch": dict(
                    common,
                    registered_tool_name="WebSearch",
                    query=package + " pypi latest version",
                    reported_version=version,
                    results=[
                        {
                            "url": "https://pypi.org/project/" + package + "/",
                            "title": package + " on PyPI",
                            "excerpt": "latest release " + version,
                        },
                        {
                            "url": "https://libraries.io/pypi/" + package,
                            "title": package + " release history",
                            "excerpt": "current version " + version,
                        },
                    ],
                ),
                "mcp__exa__web_search_exa": dict(
                    common,
                    registered_tool_name="mcp__exa__web_search_exa",
                    query=package + " latest release",
                    reported_version=version,
                    results=[
                        {
                            "url": "https://snyk.io/advisor/python/" + package,
                            "title": package + " package health",
                            "text": (
                                "The package "
                                + package
                                + " has current released version "
                                + version
                                + ". "
                            )
                            * 8,
                        },
                        {
                            "url": "https://pypi.org/project/"
                            + package
                            + "/"
                            + version
                            + "/",
                            "title": package + " " + version,
                            "text": "release page",
                        },
                    ],
                ),
                "mcp__exa__web_fetch_exa": dict(
                    common,
                    registered_tool_name="mcp__exa__web_fetch_exa",
                    url=metadata_url,
                    content=content,
                    content_sha256=hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    reported_version=version,
                    reported_last_serial=serial,
                ),
            },
        }
    with open(required_path, "w") as fh:
        fh.write(json.dumps(evidence))
if os.environ.get("FAKE_CLAUDE_STDERR_NOISE"):
    print(os.environ["FAKE_CLAUDE_STDERR_NOISE"], file=sys.stderr)
emit_harness()
'''


def write_fake_claude(directory: Path) -> Path:
    """Write the fake harness script and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_claude.py"
    script.write_text(_FAKE_CLAUDE_SOURCE, encoding="utf-8")
    return script


def write_broker_fixtures(directory: Path) -> Path:
    """Write the hermetic broker transport fixtures and return the root."""

    directory.mkdir(parents=True, exist_ok=True)
    index = {
        COMMITS_URL: {"status": 200, "body_text": json.dumps({"sha": RESOLVED_SHA})},
        RAW_URL: {"status": 200, "body_text": IMPL_BODY},
        SUPPLEMENT_URL: {"status": 200, "body_text": "extra documentation\n"},
    }
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return directory


def executor_environment(
    monkeypatch,
    *,
    fake_claude: Path,
    fixtures: Path,
    log_dir: Path,
    campaign: str = "c1-mech",
) -> None:
    """Point the executor at the fake harness and hermetic broker fixtures."""

    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(
        "MENAGERIE_AUTHOR_CLAUDE_BIN", f'"{sys.executable}" "{fake_claude}"'
    )
    monkeypatch.setenv("MENAGERIE_BROKER_FIXTURES", str(fixtures))
    monkeypatch.setenv("MENAGERIE_CAMPAIGN_ID", campaign)
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_CLAUDE_DEFAULT_DISCOVERY", json.dumps(DEFAULT_DISCOVERY))
    monkeypatch.setenv("FAKE_CLAUDE_DEFAULT_RESULT", json.dumps(DEFAULT_RESULT))
    monkeypatch.setenv("FAKE_CLAUDE_SUPPLEMENT_URL", SUPPLEMENT_URL)
    monkeypatch.delenv("FAKE_CLAUDE_MODE", raising=False)
    monkeypatch.delenv("FAKE_CLAUDE_PROBE", raising=False)
    monkeypatch.delenv("MENAGERIE_EXECUTOR_PAUSE_AFTER", raising=False)


def subprocess_environment(
    *,
    fake_claude: Path,
    fixtures: Path,
    log_dir: Path,
    campaign: str = "c1-mech",
    **extra: str,
) -> dict:
    """Build the environment for running the executor as a real subprocess."""

    log_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "PYTHONPATH": str(repo_root),
        "MENAGERIE_AUTHOR_CLAUDE_BIN": f'"{sys.executable}" "{fake_claude}"',
        "MENAGERIE_BROKER_FIXTURES": str(fixtures),
        "MENAGERIE_CAMPAIGN_ID": campaign,
        "FAKE_CLAUDE_LOG": str(log_dir),
        "FAKE_CLAUDE_DEFAULT_DISCOVERY": json.dumps(DEFAULT_DISCOVERY),
        "FAKE_CLAUDE_DEFAULT_RESULT": json.dumps(DEFAULT_RESULT),
        "FAKE_CLAUDE_SUPPLEMENT_URL": SUPPLEMENT_URL,
    }
    env.pop("MENAGERIE_EXECUTOR_PAUSE_AFTER", None)
    env.pop("FAKE_CLAUDE_MODE", None)
    env.update(extra)
    return env


def write_source_request(root: Path, stable_id: str) -> Path:
    """Write one lane-shaped source-request envelope under the author root."""

    root.mkdir(parents=True, exist_ok=True)
    request = {
        "envelope_version": "menagerie.crawler.author-source-request.v1",
        "work_id": f"work-{stable_id}",
        "stable_id": stable_id,
        "untrusted_hints": {"name": stable_id, "zoo": "test-zoo"},
        "required_output_path": str(root / "source-targets.json"),
        "max_sources": 8,
        "discovery_schema_version": "menagerie.crawler.source-discovery.v1",
    }
    path = root / "source-request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def write_author_envelope(root: Path, stable_id: str) -> Path:
    """Write one minimal author envelope stub under the author root.

    The executor treats the envelope as opaque authority (the lane built and
    validated it); only the transport fields the executor reads are present.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "model").mkdir(exist_ok=True)
    request = {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "c1-mech",
        "expected_result": {
            "schema_version": "menagerie.crawler.author-result.v4",
            "stable_id": stable_id,
            "work_id": f"work-{stable_id}",
            "campaign_id": "c1-mech",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-test",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {
            "manifest_sha256": "sha256:" + "6" * 64,
            "sources": [{"source_id": "impl-net"}],
        },
        "required_output_path": str(root / "result.json"),
        "allowed_model_dir": str(root / "model"),
        "prompt": {"path": str(root / "prompts" / "author.txt")},
    }
    (root / "prompts").mkdir(exist_ok=True)
    (root / "prompts" / "author.txt").write_text("canonical author prompt", encoding="utf-8")
    path = root / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def read_invocations(log_dir: Path) -> list:
    """Return every recorded fake-harness invocation."""

    path = log_dir / "invocations.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
