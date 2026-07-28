"""Author subagent pool: services the author queue from inside a live Claude session.

Why a pool and not a subprocess per model
-----------------------------------------
The engine's author boundary is a subprocess contract. Satisfying it with a fresh
``claude -p`` per model was measured at 54,933 cache-creation tokens per invocation just
to load context -- roughly 1.5B tokens of pure overhead across the 28,482-model roster,
before a single word of research. The production pool is instead a set of **subagents
dispatched inside one live managing session**: context loads once, and dispatches run in
parallel natively.

This module is the machinery that session drives. It does not itself call the Agent tool;
it claims jobs, renders the exact dispatch brief for one, and -- on the way back -- turns a
subagent's work into the queue files the lane is waiting for, with honest accounting.

How each queue invariant is honored
-----------------------------------
*Nonce echo.* Every file the pool publishes goes through
:func:`menagerie.crawler.author_queue.stamp`, which is the only place ``job_id`` and
``attempt_nonce`` are written. A stall triggers a lane-side retry with a fresh nonce, so a
late file from the superseded attempt is always possible; because the pool cannot publish
an unstamped file, the lane can always tell the two apart.

*Receipt as commit point.* :meth:`AuthorPool.complete` verifies the result parses, fsyncs
it and its directory, and only then writes the receipt. The lane reads the result only
after seeing the receipt, so it can never observe a partial write.

*Honest consumption.* The lane refuses a receipt that omits declared consumption or
declares more than the grant. The pool therefore measures what it can rather than
believing what it is told: wall time comes from the pool's own claim timestamp, and
``fetch_targets`` for a source-request job is **counted from the published result**, never
taken on trust. Only ``tool_calls`` is operator-declared, and it is mandatory -- there is
no default -- because the pool is the only boundary that observes Agent-tool events.
When a measurement exceeds its grant the pool refuses to publish a receipt at all and
publishes a typed failure instead. Clamping the number to fit would defeat the audit the
lane is performing, which is the whole point of the declaration.

*Typed signals.* Anthropic usage exhaustion is a backoff sidecar routed to the scheduler's
pause path, never a model failure. Everything else is a failure sidecar with an explicit
``retryable`` boolean; the lane refuses a sidecar without one rather than guessing.

*Tier selection is never inferred from the job.* A descriptor carries two campaign
identities (see :mod:`menagerie.crawler.author_queue`): its **repair scope**
(``repair_campaign_id``, per item, e.g. ``campaign-m1706``) and its **tier campaign**
(``tier_campaign_id``, one of the frozen four). Only the tier campaign selects the author
model and the standards prompt. The pool resolves the tier from the descriptor and from
its own ``--campaign`` configuration, refuses when the two disagree, and refuses when
neither supplies one. A guessed tier would author a model under the wrong frozen
``author_model_identity`` for the whole run, which is strictly worse than a loud stop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any, Callable, Mapping, Optional, Sequence

from menagerie.crawler.author_queue import (
    AuthorQueueError,
    Consumption,
    QueueJob,
    author_queue_directories,
    build_backoff,
    build_claim,
    build_failure,
    build_receipt,
    ensure_author_queue,
    job_paths,
    read_json,
    write_json_atomic,
)
from menagerie.crawler.capability_probe import (
    CAPABILITY_PROBE_FORMAT,
    CapabilityProbeError,
    derive_challenge,
    validate_capability_evidence,
)
from menagerie.crawler.constants import (
    TIER_CAMPAIGN_AUTHOR_MODELS,
    TIER_CAMPAIGN_ENV,
    TIER_CAMPAIGN_IDS,
)
from menagerie.crawler.identity import hash_bytes, utc_now
from menagerie.crawler.models import JsonObject

PROMPT_ROOT = Path(__file__).with_name("prompts") / "pool"

#: Tier campaign identities produced by the partitioner, each bound to its frozen author
#: tier. The tier is a *tier campaign* property, never a per-item one: a campaign's
#: ``author_model_identity`` is frozen for the whole run, so a model a sonnet campaign
#: finds genuinely hard is emitted as a typed BLOCKED recommendation and requeued into the
#: opus campaign, never escalated in place.
CAMPAIGN_AUTHOR_MODELS = dict(TIER_CAMPAIGN_AUTHOR_MODELS)

#: Default lease width. Short enough that a dead session returns work promptly, long
#: enough to cover a real author session with renewals.
DEFAULT_LEASE_SECONDS = 600.0

#: Provider text that means "stop dispatching", not "this model failed".
_USAGE_LIMIT_PATTERNS = (
    re.compile(r"usage limit", re.IGNORECASE),
    re.compile(r"rate[ _-]?limit", re.IGNORECASE),
    re.compile(r"quota", re.IGNORECASE),
)
_RESET_PATTERNS = (
    re.compile(r"try again at ([^\n.]+)", re.IGNORECASE),
    re.compile(r"resets? at ([^\n.]+)", re.IGNORECASE),
    re.compile(r"available again at ([^\n.]+)", re.IGNORECASE),
)


class AuthorPoolError(RuntimeError):
    """Raised when a pool operation would violate the queue contract."""


@dataclass(frozen=True)
class ClaimedJob:
    """One leased job plus the pool-measured facts its receipt will carry.

    Parameters
    ----------
    job:
        Parsed pending descriptor.
    owner:
        Managing-session identity holding the lease.
    claimed_at:
        Pool-measured lease start. Wall consumption is measured from here, not
        self-reported.
    deadline_at:
        ``claimed_at`` plus the grant's wall budget.
    """

    job: QueueJob
    owner: str
    claimed_at: datetime
    deadline_at: datetime


def default_owner() -> str:
    """Return the conventional managing-session identity.

    Returns
    -------
    str
        ``<host>:<pid>`` identity, or an operator override from
        ``MENAGERIE_AUTHOR_POOL_OWNER``.
    """

    override = os.environ.get("MENAGERIE_AUTHOR_POOL_OWNER")
    if override and override.strip():
        return override.strip()
    return f"{socket.gethostname()}:{os.getpid()}"


def classify_usage_limit(text: str) -> Optional[JsonObject]:
    """Classify operator-observed text as a provider usage limit.

    Parameters
    ----------
    text:
        Verbatim provider or harness text.

    Returns
    -------
    dict[str, Any] | None
        ``{"reason": ..., "reset_at": ...}`` when the text names a provider limit,
        otherwise ``None``. A usage limit is a scheduler pause; treating it as a model
        failure would burn the model for an outage it did not cause.
    """

    body = str(text or "")
    if not any(pattern.search(body) for pattern in _USAGE_LIMIT_PATTERNS):
        return None
    reason = "rate-limit" if re.search(r"rate[ _-]?limit", body, re.IGNORECASE) else (
        "quota-exhausted"
    )
    reset_at: Optional[str] = None
    for pattern in _RESET_PATTERNS:
        matched = pattern.search(body)
        if matched:
            reset_at = matched.group(1).strip()
            break
    return {"reason": reason, "reset_at": reset_at}


def configured_tier_campaign(explicit: Optional[str] = None) -> Optional[str]:
    """Return the tier campaign this operator process is configured to serve.

    Parameters
    ----------
    explicit:
        Operator-supplied tier campaign, typically ``--campaign``.

    Returns
    -------
    str | None
        The validated tier campaign, or ``None`` when neither the argument nor
        ``MENAGERIE_CAMPAIGN_ID`` names one.

    Raises
    ------
    AuthorPoolError
        When a value is supplied but is outside the partitioner's closed four. An
        unrecognized tier is refused here rather than ignored: silently falling back to
        the descriptor, or to a default, is how a run acquires the wrong frozen author
        identity.
    """

    value = explicit if explicit is not None else os.environ.get(TIER_CAMPAIGN_ENV)
    if value is None or not str(value).strip():
        return None
    tier = str(value).strip()
    if tier not in TIER_CAMPAIGN_IDS:
        raise AuthorPoolError(
            f"configured tier campaign {tier!r} is not one of {sorted(TIER_CAMPAIGN_IDS)}"
        )
    return tier


def resolve_tier_campaign(job: QueueJob, *, configured: Optional[str] = None) -> str:
    """Resolve the one tier campaign that governs this job's author tier.

    The descriptor's own repair scope (``job.repair_campaign_id``, e.g.
    ``campaign-m1706``) is deliberately not consulted: it is per-item lineage and carries
    no tier information at all.

    Parameters
    ----------
    job:
        Job whose tier is being resolved.
    configured:
        Tier campaign this operator process was configured with, already validated by
        :func:`configured_tier_campaign`.

    Returns
    -------
    str
        The resolved tier campaign.

    Raises
    ------
    AuthorPoolError
        When the descriptor and the configuration disagree, when the descriptor names an
        unknown tier, or when neither supplies one. All three are refusals, never
        defaults: authoring under the wrong tier corrupts the campaign's frozen
        ``author_model_identity``, which no later step can detect or repair.
    """

    declared = job.tier_campaign_id
    if declared is not None and declared not in TIER_CAMPAIGN_IDS:
        raise AuthorPoolError(
            f"author job {job.job_id} names an unknown tier campaign {declared!r}; "
            f"expected one of {sorted(TIER_CAMPAIGN_IDS)}"
        )
    if declared is not None and configured is not None and declared != configured:
        raise AuthorPoolError(
            f"author job {job.job_id} is bound to tier campaign {declared!r} but this "
            f"pool is configured for {configured!r}; servicing it would author under the "
            "wrong frozen author identity"
        )
    tier = declared or configured
    if tier is None:
        raise AuthorPoolError(
            f"author job {job.job_id} names no tier campaign and this pool was not "
            f"configured with one; pass --campaign (or set {TIER_CAMPAIGN_ENV}) to one of "
            f"{sorted(TIER_CAMPAIGN_IDS)}. The job's repair scope "
            f"({job.repair_campaign_id!r}) is per-item lineage and never selects a tier"
        )
    return tier


class AuthorPool:
    """Queue servicer for one campaign's author subagent pool.

    Parameters
    ----------
    queue_root:
        Campaign-local author-queue root.
    owner:
        Managing-session identity recorded on every lease.
    lease_seconds:
        Lease width; an expired lease returns the job to the queue.
    tier_campaign_id:
        Frozen tier campaign this pool serves; defaults to ``MENAGERIE_CAMPAIGN_ID``. The
        tier is a property of the campaign run, not of an individual job, so it is
        configured once here and cross-checked against every descriptor that declares one.
    clock:
        Injectable wall clock for deterministic tests.
    """

    def __init__(
        self,
        queue_root: Path,
        *,
        owner: Optional[str] = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        tier_campaign_id: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        """Bind the queue root, the lease policy, the tier, and the session identity."""

        if lease_seconds <= 0:
            raise ValueError("author pool lease must be positive")
        self.queue_root = Path(queue_root)
        self.owner = owner or default_owner()
        self.lease_seconds = float(lease_seconds)
        self.tier_campaign_id = configured_tier_campaign(tier_campaign_id)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.directories = ensure_author_queue(self.queue_root)

    def tier_campaign(self, job: QueueJob) -> str:
        """Return the frozen tier campaign governing one job.

        Parameters
        ----------
        job:
            Job to resolve.

        Returns
        -------
        str
            Resolved tier campaign.

        Raises
        ------
        AuthorPoolError
            When the tier is unknown, absent, or disagrees with this pool's configuration.
        """

        return resolve_tier_campaign(job, configured=self.tier_campaign_id)

    # -- reading the queue -------------------------------------------------

    def pending(self) -> list[QueueJob]:
        """Return every parseable pending job, oldest first.

        Returns
        -------
        list[QueueJob]
            Pending jobs sorted by enqueue time then job identity.
        """

        jobs: list[QueueJob] = []
        for path in sorted(self.directories["pending"].glob("*.json")):
            payload = read_json(path)
            if payload is None:
                continue
            try:
                jobs.append(QueueJob.from_mapping(payload))
            except AuthorQueueError:
                continue
        return sorted(jobs, key=lambda job: (job.enqueued_at, job.job_id))

    def load(self, job_id: str) -> QueueJob:
        """Return one pending job by identity.

        Parameters
        ----------
        job_id:
            Job identity.

        Returns
        -------
        QueueJob
            Parsed descriptor.

        Raises
        ------
        AuthorPoolError
            When the job is absent or unparseable.
        """

        payload = read_json(job_paths(self.queue_root, job_id)["pending"])
        if payload is None:
            raise AuthorPoolError(f"author job {job_id} is not pending")
        try:
            return QueueJob.from_mapping(payload)
        except AuthorQueueError as exc:
            raise AuthorPoolError(f"author job {job_id} is malformed: {exc}") from exc

    def active_claim(self, job: QueueJob) -> Optional[JsonObject]:
        """Return a live lease for one job, if any.

        Parameters
        ----------
        job:
            Job whose lease is inspected.

        Returns
        -------
        dict[str, Any] | None
            The lease when it is bound to this attempt and unexpired.
        """

        payload = read_json(job.claim_path)
        if payload is None or payload.get("attempt_nonce") != job.attempt_nonce:
            return None
        try:
            expires = _parse_instant(payload.get("lease_expires_at"))
        except ValueError:
            return None
        return payload if expires > self._clock() else None

    # -- leasing -----------------------------------------------------------

    def claim(self, job_id: str, *, force: bool = False) -> ClaimedJob:
        """Lease one pending job for dispatch.

        Parameters
        ----------
        job_id:
            Job identity.
        force:
            Take over a live lease held by another servicer.

        Returns
        -------
        ClaimedJob
            Lease with the pool-measured start and the grant deadline.

        Raises
        ------
        AuthorPoolError
            When the job is already leased and ``force`` is not set, or when its tier
            campaign is unresolvable. The tier is checked *before* the lease is written:
            leasing a job this session cannot brief would only park it under a lease
            nobody can discharge.
        """

        job = self.load(job_id)
        self.tier_campaign(job)
        existing = self.active_claim(job)
        if existing is not None and not force and existing.get("owner") != self.owner:
            raise AuthorPoolError(
                f"author job {job_id} is leased by {existing.get('owner')} until "
                f"{existing.get('lease_expires_at')}"
            )
        now = self._clock()
        write_json_atomic(
            job.claim_path,
            build_claim(job, owner=self.owner, lease_expires_at=_iso(now + timedelta(
                seconds=self.lease_seconds
            ))),
        )
        return ClaimedJob(
            job=job,
            owner=self.owner,
            claimed_at=now,
            deadline_at=now + timedelta(seconds=job.effort_grant.wall_seconds),
        )

    def renew(self, job_id: str) -> str:
        """Extend this session's lease on one job.

        Parameters
        ----------
        job_id:
            Job identity.

        Returns
        -------
        str
            New lease expiry.

        Raises
        ------
        AuthorPoolError
            When no lease for this attempt exists.
        """

        job = self.load(job_id)
        payload = read_json(job.claim_path)
        if payload is None or payload.get("attempt_nonce") != job.attempt_nonce:
            raise AuthorPoolError(f"author job {job_id} holds no lease for this attempt")
        expires = _iso(self._clock() + timedelta(seconds=self.lease_seconds))
        write_json_atomic(job.claim_path, build_claim(job, owner=self.owner, lease_expires_at=expires))
        return expires

    def release(self, job_id: str) -> None:
        """Drop this session's lease without answering the job.

        Parameters
        ----------
        job_id:
            Job identity.
        """

        job_paths(self.queue_root, job_id)["claim"].unlink(missing_ok=True)

    def expired(self) -> list[QueueJob]:
        """Return pending jobs whose lease has lapsed and which are unanswered.

        Returns
        -------
        list[QueueJob]
            Jobs safe to re-dispatch. A job with a published receipt is excluded even if
            its lease lapsed: a completion always outranks a stale lease.
        """

        stale: list[QueueJob] = []
        for job in self.pending():
            if job.receipt_path.is_file() or job.failure_path.is_file():
                continue
            if self.active_claim(job) is None and job.claim_path.is_file():
                stale.append(job)
        return stale

    # -- dispatch ----------------------------------------------------------

    def brief(self, job: QueueJob, *, claimed: Optional[ClaimedJob] = None) -> str:
        """Render the exact prompt to hand one author subagent.

        Parameters
        ----------
        job:
            Job to dispatch.
        claimed:
            Active lease, when the deadline should be concrete.

        Returns
        -------
        str
            Fully rendered dispatch brief.

        Raises
        ------
        AuthorPoolError
            When the job's tier campaign is unresolvable or has no bound prompt.
        """

        return render_dispatch_brief(
            job,
            claimed=claimed,
            repo_root=_repo_root(),
            tier_campaign_id=self.tier_campaign_id,
        )

    def subagent_model(self, job: QueueJob) -> str:
        """Return the Agent-tool model tier for one job.

        Parameters
        ----------
        job:
            Job to dispatch.

        Returns
        -------
        str
            ``opus`` for the classics campaign, ``sonnet`` otherwise. The tier follows the
            *tier campaign*, never the individual model or the job's repair scope, because
            the campaign's author identity is frozen for its whole run.

        Raises
        ------
        AuthorPoolError
            When the tier campaign is unresolvable, or when the descriptor declares an
            author model that contradicts the tier campaign's frozen one. The latter means
            the producer and this pool disagree about which model authors this run, and
            dispatching either one would be a coin flip on the run's identity.
        """

        tier = self.tier_campaign(job)
        frozen = CAMPAIGN_AUTHOR_MODELS[tier]
        declared = job.author_model
        if declared and declared != frozen:
            raise AuthorPoolError(
                f"author job {job.job_id} declares author model {declared!r} but tier "
                f"campaign {tier!r} is frozen to {frozen!r}"
            )
        return "opus" if "opus" in frozen else "sonnet"

    # -- answering ---------------------------------------------------------

    def complete(
        self,
        claimed: ClaimedJob,
        *,
        tool_calls: int,
        notes: Sequence[str] = (),
    ) -> JsonObject:
        """Publish the receipt that commits one finished job.

        Parameters
        ----------
        claimed:
            Active lease.
        tool_calls:
            Agent tool calls the pool observed for this session. Mandatory: the pool is
            the only boundary that sees them, so there is no defensible default.
        notes:
            Bounded operator notes.

        Returns
        -------
        dict[str, Any]
            The published receipt.

        Raises
        ------
        AuthorPoolError
            When the result is absent or unparseable, or when measured consumption
            exceeds the grant. In the latter case a typed failure sidecar is published
            first: an over-grant receipt would be refused by the lane, and shrinking the
            number to fit would be a lie.
        """

        job = claimed.job
        if int(tool_calls) < 0:
            raise AuthorPoolError("declared tool calls cannot be negative")
        payload = self._commit_result(job)
        consumption = Consumption(
            tool_calls=int(tool_calls),
            fetch_targets=_count_fetch_targets(job, payload),
            wall_seconds=max(0.0, (self._clock() - claimed.claimed_at).total_seconds()),
        )
        try:
            receipt = build_receipt(
                job,
                consumption=consumption,
                author_model=job.author_model,
                result_sha256=hash_bytes(job.required_output_path.read_bytes()),
                notes=notes,
            )
        except AuthorQueueError as exc:
            self.fail(
                job,
                reason="effort-cap-exhausted",
                retryable=False,
                detail=(
                    f"{exc}; measured consumption {consumption.to_dict()} against grant "
                    f"{job.effort_grant.to_dict()}"
                ),
            )
            raise AuthorPoolError(f"author job {job.job_id} exceeded its grant: {exc}") from exc
        write_json_atomic(job.receipt_path, receipt)
        return receipt

    def complete_capability_probe(
        self,
        claimed: ClaimedJob,
        evidence: Mapping[str, Any],
        *,
        tool_calls: int,
    ) -> JsonObject:
        """Validate capability evidence and publish the nonce-bound receipt.

        The receipt is written **only** if the evidence survives every check in
        :mod:`menagerie.crawler.capability_probe`. A probe that cannot be proven produces
        no receipt, and the doctor's strict check fails -- which is the correct outcome,
        because an author path without live web tools cannot ground a proposal.

        Parameters
        ----------
        claimed:
            Active lease on a ``capability-probe`` job.
        evidence:
            Session-reported per-tool evidence.
        tool_calls:
            Agent tool calls the pool observed.

        Returns
        -------
        dict[str, Any]
            The queue receipt committing the probe.

        Raises
        ------
        AuthorPoolError
            When the job is not a capability probe or the evidence fails validation.
        """

        job = claimed.job
        if job.kind != "capability-probe":
            raise AuthorPoolError(f"author job {job.job_id} is not a capability probe")
        request = job.read_request()
        nonce = str(request.get("nonce", ""))
        try:
            receipt = validate_capability_evidence(
                nonce=nonce,
                evidence=evidence,
                requested_at=_parse_instant(request.get("requested_at")),
                deadline_seconds=int(request.get("deadline_seconds", 120)),
                now=self._clock(),
            )
        except (CapabilityProbeError, ValueError) as exc:
            self.fail(job, reason="capability-probe-unproven", retryable=True, detail=str(exc))
            raise AuthorPoolError(f"capability probe {job.job_id} is unproven: {exc}") from exc
        write_json_atomic(job.required_output_path, receipt)
        return self.complete(claimed, tool_calls=tool_calls, notes=("capability probe proven",))

    def fail(
        self,
        job: QueueJob,
        *,
        reason: str,
        retryable: bool,
        detail: str = "",
    ) -> JsonObject:
        """Publish one typed failure sidecar.

        Parameters
        ----------
        job:
            Failed job.
        reason:
            Short machine-readable label.
        retryable:
            Explicit classification; the lane refuses a sidecar without one.
        detail:
            Bounded diagnostic detail.

        Returns
        -------
        dict[str, Any]
            The published failure payload.
        """

        payload = build_failure(job, reason=reason, retryable=retryable, detail=detail)
        write_json_atomic(job.failure_path, payload)
        return payload

    def backoff(
        self,
        job: QueueJob,
        *,
        excerpt: str,
        reset_at: Optional[str] = None,
        reason: Optional[str] = None,
        provider: str = "anthropic",
    ) -> JsonObject:
        """Publish one typed provider-pause sidecar.

        Parameters
        ----------
        job:
            Job interrupted by a provider limit.
        excerpt:
            Verbatim provider text.
        reset_at:
            Explicit reset instant; otherwise parsed from ``excerpt``.
        reason:
            Explicit pause reason; otherwise classified from ``excerpt``.
        provider:
            Usage-limit provider. Author sessions pause on ``anthropic``.

        Returns
        -------
        dict[str, Any]
            The published backoff payload.
        """

        classified = classify_usage_limit(excerpt) or {}
        payload = build_backoff(
            job,
            provider=provider,
            reason=reason or str(classified.get("reason") or "quota-exhausted"),
            response_excerpt=excerpt,
            reset_at=reset_at or classified.get("reset_at"),
        )
        write_json_atomic(job.backoff_path, payload)
        return payload

    # -- internals ---------------------------------------------------------

    def _commit_result(self, job: QueueJob) -> JsonObject:
        """Verify and durably commit the published result.

        Parameters
        ----------
        job:
            Job whose result is being committed.

        Returns
        -------
        dict[str, Any]
            Parsed result payload.

        Raises
        ------
        AuthorPoolError
            When the result is absent or does not parse. Publishing a receipt for an
            unreadable result would hand the lane a broken commit point.
        """

        path = job.required_output_path
        if not path.is_file():
            raise AuthorPoolError(
                f"author job {job.job_id} has no result at its required output path {path}"
            )
        payload = read_json(path)
        if payload is None:
            raise AuthorPoolError(f"author job {job.job_id} published an unparseable result")
        _fsync_file(path)
        return payload


def _count_fetch_targets(job: QueueJob, payload: Mapping[str, Any]) -> int:
    """Count pinned controlled-fetch targets from the published result.

    Counting rather than believing is deliberate: this is the one consumption dimension
    the pool can observe exactly, so it is never operator-declared.

    Parameters
    ----------
    job:
        Answered job.
    payload:
        Parsed result payload.

    Returns
    -------
    int
        Number of named source targets; zero for stages that name none.
    """

    if job.kind != "source-request":
        return 0
    sources = payload.get("sources")
    return len(sources) if isinstance(sources, list) else 0


def _fsync_file(path: Path) -> None:
    """Flush one published file and its directory entry to stable storage.

    Parameters
    ----------
    path:
        Published file.
    """

    handle = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _iso(value: datetime) -> str:
    """Return one UTC instant in the campaign's canonical spelling.

    Parameters
    ----------
    value:
        Instant to format.

    Returns
    -------
    str
        ``...Z`` ISO-8601 spelling.
    """

    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_instant(value: Any) -> datetime:
    """Parse one ISO-8601 instant.

    Parameters
    ----------
    value:
        Raw timestamp.

    Returns
    -------
    datetime
        Timezone-aware instant.

    Raises
    ------
    ValueError
        When the timestamp is absent or unparseable.
    """

    text = str(value or "").strip()
    if not text:
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00" if text.endswith("Z") else text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _repo_root() -> Path:
    """Return the repository root containing this package.

    Returns
    -------
    Path
        Repository root.
    """

    return Path(__file__).resolve().parents[2]


def _read_prompt(name: str) -> str:
    """Read one pool prompt fragment.

    Parameters
    ----------
    name:
        File name under ``prompts/pool``.

    Returns
    -------
    str
        Prompt text.

    Raises
    ------
    AuthorPoolError
        When the fragment is missing from the installed package.
    """

    path = PROMPT_ROOT / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise AuthorPoolError(f"author pool prompt {name} is unavailable: {exc}") from exc


def campaign_prompt_name(tier_campaign_id: Optional[str]) -> str:
    """Return the standards fragment bound to one tier campaign.

    Parameters
    ----------
    tier_campaign_id:
        Resolved tier campaign -- never a job's per-item repair scope.

    Returns
    -------
    str
        Prompt file name.

    Raises
    ------
    AuthorPoolError
        When the tier campaign is outside the partitioner's closed set. Guessing one
        would silently author a hard model with the wrong tier and the wrong standards.
    """

    if tier_campaign_id not in CAMPAIGN_AUTHOR_MODELS:
        raise AuthorPoolError(
            f"author job names an unknown tier campaign {tier_campaign_id!r}; expected "
            f"one of {sorted(CAMPAIGN_AUTHOR_MODELS)}"
        )
    return f"campaign_{tier_campaign_id}.md"


def render_dispatch_brief(
    job: QueueJob,
    *,
    claimed: Optional[ClaimedJob] = None,
    repo_root: Optional[Path] = None,
    tier_campaign_id: Optional[str] = None,
) -> str:
    """Render the complete brief for one author subagent dispatch.

    Parameters
    ----------
    job:
        Job to dispatch.
    claimed:
        Active lease, so the brief can state a concrete wall deadline.
    repo_root:
        Repository root quoted in the brief.
    tier_campaign_id:
        Tier campaign this servicer is configured for. Cross-checked against the
        descriptor's own binding; one of the two must supply it.

    Returns
    -------
    str
        Rendered brief: job facts, the stage instructions, and the tier campaign's
        standards.

    Raises
    ------
    AuthorPoolError
        When the job kind has no bound prompt, or when the tier campaign is unresolvable.
    """

    stage = {
        "source-request": "stage_source_request.md",
        "author": "stage_author.md",
        "capability-probe": "stage_capability_probe.md",
    }.get(job.kind)
    if stage is None:
        raise AuthorPoolError(f"author job names an unsupported kind: {job.kind!r}")
    tier = resolve_tier_campaign(job, configured=tier_campaign_id)
    grant = job.effort_grant
    facts = [
        "## JOB FACTS (binding)",
        "",
        f"- job_id: `{job.job_id}`",
        f"- kind: `{job.kind}`",
        f"- stable_id: `{job.stable_id}`",
        f"- work_id: `{job.work_id}`",
        f"- tier campaign: `{tier}` (frozen author tier `{CAMPAIGN_AUTHOR_MODELS[tier]}`)",
        f"- repair scope: `{job.repair_campaign_id}`",
        f"- repository root (READ ONLY): `{repo_root or _repo_root()}`",
        f"- REQUEST envelope to read first: `{job.request_path}`",
        f"- REQUIRED output path, exact: `{job.required_output_path}`",
        "- Effort grant, hard: "
        f"{grant.tool_calls} tool calls, {grant.fetch_targets} fetch targets, "
        f"{grant.wall_seconds:g}s wall",
    ]
    if claimed is not None:
        facts.append(f"- wall deadline: `{_iso(claimed.deadline_at)}`")
    if job.kind == "capability-probe":
        request = job.read_request()
        challenge = derive_challenge(str(request.get("nonce", "")))
        facts.extend(
            [
                f"- probe nonce: `{request.get('nonce')}`",
                f"- challenge package: `{challenge.package}`",
                f"- challenge metadata URL: `{challenge.metadata_url}`",
                f"- challenge project URL: `{challenge.project_url}`",
                f"- challenge_id: `{challenge.challenge_id}`",
            ]
        )
    sections = ["\n".join(facts), _read_prompt(stage)]
    if job.kind != "capability-probe":
        sections.append(_read_prompt(campaign_prompt_name(tier)))
    return "\n\n---\n\n".join(sections) + "\n"


# -- operator CLI ---------------------------------------------------------


def _emit(payload: Any) -> None:
    """Print one JSON payload for the managing session to read.

    Parameters
    ----------
    payload:
        JSON-serializable value.
    """

    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the author-pool operator CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.author_pool",
        description="Service one campaign's author queue from the managing Claude session.",
    )
    parser.add_argument("--queue", required=True, type=Path, help="campaign author-queue root")
    parser.add_argument("--owner", default=None, help="managing-session identity")
    parser.add_argument(
        "--campaign",
        dest="tier_campaign_id",
        default=None,
        help=(
            "frozen tier campaign this session serves, one of "
            f"{sorted(TIER_CAMPAIGN_IDS)}; defaults to {TIER_CAMPAIGN_ENV}. This is the "
            "run's tier, never a job's per-item repair scope"
        ),
    )
    parser.add_argument(
        "--lease-seconds", type=float, default=DEFAULT_LEASE_SECONDS, help="lease width"
    )
    actions = parser.add_subparsers(dest="action", required=True)

    actions.add_parser("list", help="list pending jobs")
    actions.add_parser("expired", help="list jobs whose lease lapsed and are unanswered")

    for name, help_text in (
        ("claim", "lease one job and print its dispatch brief"),
        ("brief", "print one job's dispatch brief without leasing"),
        ("renew", "extend this session's lease"),
        ("release", "drop this session's lease"),
    ):
        sub = actions.add_parser(name, help=help_text)
        sub.add_argument("--job", required=True, help="job identity")
        if name == "claim":
            sub.add_argument("--force", action="store_true", help="take over a live lease")

    complete = actions.add_parser("complete", help="commit a finished job with a receipt")
    complete.add_argument("--job", required=True)
    complete.add_argument("--claimed-at", required=True, help="lease start echoed from claim")
    complete.add_argument("--tool-calls", required=True, type=int, help="observed tool calls")
    complete.add_argument("--note", action="append", default=[], help="bounded operator note")
    complete.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="capability-probe evidence JSON produced by the subagent",
    )

    fail = actions.add_parser("fail", help="publish a typed failure sidecar")
    fail.add_argument("--job", required=True)
    fail.add_argument("--reason", required=True)
    fail.add_argument("--detail", default="")
    retry = fail.add_mutually_exclusive_group(required=True)
    retry.add_argument("--retryable", dest="retryable", action="store_true")
    retry.add_argument("--permanent", dest="retryable", action="store_false")

    backoff = actions.add_parser("backoff", help="publish a typed provider-pause sidecar")
    backoff.add_argument("--job", required=True)
    backoff.add_argument("--excerpt", required=True, help="verbatim provider text")
    backoff.add_argument("--reset-at", default=None)
    backoff.add_argument("--provider", default="anthropic")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one author-pool operator action.

    Parameters
    ----------
    argv:
        Command-line arguments.

    Returns
    -------
    int
        Process exit status.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        pool = AuthorPool(
            args.queue,
            owner=args.owner,
            lease_seconds=args.lease_seconds,
            tier_campaign_id=args.tier_campaign_id,
        )
        return _dispatch_action(pool, args)
    except (AuthorPoolError, AuthorQueueError, CapabilityProbeError) as exc:
        print(f"author-pool error: {exc}", file=sys.stderr)
        return 1


def _dispatch_action(pool: AuthorPool, args: argparse.Namespace) -> int:
    """Execute one parsed operator action.

    Parameters
    ----------
    pool:
        Bound pool.
    args:
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.
    """

    if args.action == "list":
        _emit([_job_summary(pool, job) for job in pool.pending()])
        return 0
    if args.action == "expired":
        _emit([_job_summary(pool, job) for job in pool.expired()])
        return 0
    if args.action == "brief":
        print(pool.brief(pool.load(args.job)))
        return 0
    if args.action == "claim":
        claimed = pool.claim(args.job, force=args.force)
        _emit(
            {
                "job_id": claimed.job.job_id,
                "kind": claimed.job.kind,
                "stable_id": claimed.job.stable_id,
                "repair_campaign_id": claimed.job.repair_campaign_id,
                "tier_campaign_id": pool.tier_campaign(claimed.job),
                "subagent_model": pool.subagent_model(claimed.job),
                "claimed_at": _iso(claimed.claimed_at),
                "deadline_at": _iso(claimed.deadline_at),
                "required_output_path": str(claimed.job.required_output_path),
                "brief": pool.brief(claimed.job, claimed=claimed),
            }
        )
        return 0
    if args.action == "renew":
        _emit({"job_id": args.job, "lease_expires_at": pool.renew(args.job)})
        return 0
    if args.action == "release":
        pool.release(args.job)
        _emit({"job_id": args.job, "released": True})
        return 0
    if args.action == "complete":
        return _complete_action(pool, args)
    if args.action == "fail":
        _emit(
            pool.fail(
                pool.load(args.job),
                reason=args.reason,
                retryable=bool(args.retryable),
                detail=args.detail,
            )
        )
        return 0
    if args.action == "backoff":
        _emit(
            pool.backoff(
                pool.load(args.job),
                excerpt=args.excerpt,
                reset_at=args.reset_at,
                provider=args.provider,
            )
        )
        return 0
    raise AuthorPoolError(f"unsupported action {args.action!r}")


def _complete_action(pool: AuthorPool, args: argparse.Namespace) -> int:
    """Commit one finished job from the CLI.

    Parameters
    ----------
    pool:
        Bound pool.
    args:
        Parsed arguments.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    AuthorPoolError
        When the lease start is unparseable or the probe evidence is missing.
    """

    job = pool.load(args.job)
    try:
        claimed_at = _parse_instant(args.claimed_at)
    except ValueError as exc:
        raise AuthorPoolError(f"--claimed-at is not an ISO-8601 instant: {exc}") from exc
    claimed = ClaimedJob(
        job=job,
        owner=pool.owner,
        claimed_at=claimed_at,
        deadline_at=claimed_at + timedelta(seconds=job.effort_grant.wall_seconds),
    )
    if job.kind == "capability-probe":
        if args.evidence is None:
            raise AuthorPoolError("a capability probe requires --evidence")
        evidence = read_json(Path(args.evidence))
        if evidence is None:
            raise AuthorPoolError(f"capability evidence at {args.evidence} is unreadable")
        _emit(
            pool.complete_capability_probe(claimed, evidence, tool_calls=args.tool_calls)
        )
        return 0
    _emit(pool.complete(claimed, tool_calls=args.tool_calls, notes=args.note))
    return 0


def _job_summary(pool: AuthorPool, job: QueueJob) -> JsonObject:
    """Return one compact job row for the managing session.

    Parameters
    ----------
    pool:
        Bound pool.
    job:
        Pending job.

    Returns
    -------
    dict[str, Any]
        Compact summary.
    """

    claim = pool.active_claim(job)
    # ``list`` is the operator's situational awareness, so an unresolvable tier is
    # reported on the row rather than aborting the whole listing. It is still refused at
    # every point that would act on it -- brief, claim's printed dispatch, and
    # subagent_model -- so nothing is dispatched on a guess.
    tier: Optional[str] = None
    subagent_model: Optional[str] = None
    tier_error: Optional[str] = None
    try:
        tier = pool.tier_campaign(job)
        subagent_model = pool.subagent_model(job)
    except AuthorPoolError as exc:
        tier_error = str(exc)
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "stable_id": job.stable_id,
        "repair_campaign_id": job.repair_campaign_id,
        "tier_campaign_id": tier,
        "tier_error": tier_error,
        "author_model": job.author_model,
        "subagent_model": subagent_model,
        "enqueued_at": job.enqueued_at,
        "leased_by": None if claim is None else claim.get("owner"),
        "lease_expires_at": None if claim is None else claim.get("lease_expires_at"),
        "required_output_path": str(job.required_output_path),
    }


__all__ = [
    "AuthorPool",
    "AuthorPoolError",
    "CAMPAIGN_AUTHOR_MODELS",
    "CAPABILITY_PROBE_FORMAT",
    "ClaimedJob",
    "TIER_CAMPAIGN_ENV",
    "TIER_CAMPAIGN_IDS",
    "author_queue_directories",
    "campaign_prompt_name",
    "classify_usage_limit",
    "configured_tier_campaign",
    "default_owner",
    "main",
    "render_dispatch_brief",
    "resolve_tier_campaign",
    "utc_now",
]


if __name__ == "__main__":  # pragma: no cover -- operator entry point
    raise SystemExit(main())
