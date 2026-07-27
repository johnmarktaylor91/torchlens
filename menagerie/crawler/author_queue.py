"""Author-queue wire protocol shared by the engine lane and the subagent pool.

The engine side of this RPC is ``QueueAuthorLane`` (``driver_admission``): it publishes a
job descriptor and blocks until the operator publishes a result at the job's exact
``required_output_path``, gated by a receipt. This module is the *protocol* half of that
contract -- pure data, no driver imports -- so the pool, the author operator wrapper, and
the capability probe can all speak it without dragging in the scheduler.

Layout, under the campaign-local queue root, all paths absolute, all writes atomic::

    <queue_root>/pending/<job_id>.json            lane -> pool   job descriptor
    <queue_root>/claimed/<job_id>.json            pool -> lane   lease, liveness only
    <queue_root>/receipts/<job_id>.json           pool -> lane   completion receipt
    <queue_root>/signals/<job_id>.backoff.json    pool -> lane   typed provider pause
    <queue_root>/signals/<job_id>.failure.json    pool -> lane   typed failure

Three invariants are load-bearing and are enforced by the builders below rather than left
to prose:

1. **Every pool-published file echoes the job's ``attempt_nonce``.** A stall triggers a
   retry, so a late file from a superseded attempt is always possible; the lane discards
   any file whose nonce does not match, and it can only do that if the pool always stamps
   one. :func:`stamp` is the single chokepoint that stamps it.
2. **The receipt is the commit point.** The result is written to
   ``required_output_path`` and fsynced *before* the receipt exists, so the lane never
   observes a half-written result.
3. **A receipt declares consumption, and a failure declares retryability.** The lane
   refuses a receipt that omits consumption or declares more than the published grant, and
   refuses a failure sidecar without an explicit boolean ``retryable``. Those refusals are
   what convert a prompt instruction into something the engine can hold the pool to, so
   the builders here require both rather than defaulting them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    AUTHOR_MAX_FETCH_TARGETS,
    AUTHOR_MAX_TOOL_CALLS,
    AUTHOR_SESSION_WALL_SECONDS,
    USAGE_LIMIT_PROVIDERS,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    fsync_directory,
    hash_bytes,
    stable_hash,
    utc_now,
)
from menagerie.crawler.models import JsonObject

AUTHOR_QUEUE_DIRECTORIES = ("pending", "claimed", "receipts", "signals")
AUTHOR_JOB_VERSION = "menagerie.crawler.author-queue-job.v1"
AUTHOR_CLAIM_VERSION = "menagerie.crawler.author-queue-claim.v1"
AUTHOR_RECEIPT_VERSION = "menagerie.crawler.author-queue-receipt.v1"
AUTHOR_BACKOFF_VERSION = "menagerie.crawler.author-queue-backoff.v1"
AUTHOR_FAILURE_VERSION = "menagerie.crawler.author-queue-failure.v1"

#: Round-trip labels the lane and the wrapper may enqueue. ``source-request`` and
#: ``author`` are the two stages of one model's authoring; ``capability-probe`` is the
#: doctor's nonce-bound proof that the author path has live web grounding.
AUTHOR_JOB_KINDS = ("source-request", "author", "capability-probe")

#: Operator-protocol exit codes (``PLAN_RECONCILED`` section 3.0).
EXIT_OK = 0
EXIT_PERMANENT = 64
EXIT_RETRYABLE = 75
EXIT_QUOTA = 76
EXIT_UNAVAILABLE = 78


class AuthorQueueError(RuntimeError):
    """Raised when a queue payload violates the published protocol."""


@dataclass(frozen=True)
class EffortGrant:
    """Per-session author effort budget published with a job.

    Parameters
    ----------
    tool_calls:
        Maximum agent tool calls the pool may spend on one session.
    fetch_targets:
        Maximum pinned controlled-fetch source targets.
    wall_seconds:
        Maximum author session wall time.
    """

    tool_calls: int
    fetch_targets: int
    wall_seconds: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EffortGrant":
        """Parse a published grant.

        Parameters
        ----------
        value:
            ``effort_grant`` object from the job descriptor.

        Returns
        -------
        EffortGrant
            Parsed grant.

        Raises
        ------
        AuthorQueueError
            When a dimension is missing or not strictly positive.
        """

        try:
            grant = cls(
                tool_calls=int(value["tool_calls"]),
                fetch_targets=int(value["fetch_targets"]),
                wall_seconds=float(value["wall_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthorQueueError(f"author job publishes an unparseable effort grant: {exc}")
        if min(grant.tool_calls, grant.fetch_targets, grant.wall_seconds) <= 0:
            raise AuthorQueueError("author effort grant dimensions must be positive")
        return grant

    def to_dict(self) -> JsonObject:
        """Return the JSON grant payload.

        Returns
        -------
        dict[str, Any]
            Closed grant payload.
        """

        return {
            "tool_calls": self.tool_calls,
            "fetch_targets": self.fetch_targets,
            "wall_seconds": self.wall_seconds,
        }


#: ``PLAN.md`` LP-13.2 author-session ceiling, used when a job is enqueued by the operator
#: wrapper rather than by the lane. The values come directly from the engine constants so
#: the two enqueue paths cannot drift.
DEFAULT_EFFORT_GRANT = EffortGrant(
    tool_calls=AUTHOR_MAX_TOOL_CALLS,
    fetch_targets=AUTHOR_MAX_FETCH_TARGETS,
    wall_seconds=float(AUTHOR_SESSION_WALL_SECONDS),
)


@dataclass(frozen=True)
class Consumption:
    """Declared effort actually spent on one author session.

    The lane audits this against the job's grant, so an over-declaration is refused with
    ``AuthorEffortCapExceeded`` and an omission with ``DriverIntegrationError``. Declaring
    honestly is therefore the only way to get a completion accepted; see
    :meth:`validate_against` for the pool-side mirror of that audit.

    Parameters
    ----------
    tool_calls:
        Agent tool calls spent, as observed by the pool.
    fetch_targets:
        Pinned controlled-fetch targets named by the session. For a ``source-request``
        job this is *counted from the published result*, not taken on trust.
    wall_seconds:
        Session wall time, measured by the pool from claim to completion.
    """

    tool_calls: int
    fetch_targets: int
    wall_seconds: float

    def to_dict(self) -> JsonObject:
        """Return the JSON consumption payload.

        Returns
        -------
        dict[str, Any]
            Closed consumption payload.
        """

        return {
            "tool_calls": self.tool_calls,
            "fetch_targets": self.fetch_targets,
            "wall_seconds": round(float(self.wall_seconds), 3),
        }

    def validate_against(self, grant: EffortGrant) -> None:
        """Refuse a declaration the lane would reject anyway.

        Parameters
        ----------
        grant:
            Published per-session grant.

        Raises
        ------
        AuthorQueueError
            When any dimension is negative or exceeds its grant.
        """

        pairs = (
            ("tool_calls", float(self.tool_calls), float(grant.tool_calls)),
            ("fetch_targets", float(self.fetch_targets), float(grant.fetch_targets)),
            ("wall_seconds", float(self.wall_seconds), float(grant.wall_seconds)),
        )
        for metric, spent, limit in pairs:
            if spent < 0:
                raise AuthorQueueError(f"declared {metric} consumption cannot be negative")
            if spent > limit:
                raise AuthorQueueError(
                    f"declared {metric} consumption {spent:g} exceeds the {limit:g} grant"
                )


@dataclass(frozen=True)
class QueueJob:
    """One parsed pending job descriptor.

    Parameters
    ----------
    job_id, attempt_nonce, kind:
        Job identity, the per-attempt nonce every pool file must echo, and the round-trip
        label.
    stable_id, work_id, campaign_id, author_model:
        Model, work generation, campaign, and the campaign's frozen author model.
    request_path, request_sha256:
        Absolute request envelope path and its exact digest.
    required_output_path:
        Exact path the result must be published to.
    receipt_path, backoff_path, failure_path, claim_path:
        Absolute pool-published control paths.
    effort_grant:
        Published per-session budget.
    stall_timeout_seconds:
        Lane deadline. Exceeding it is retryable infrastructure, never a model failure.
    enqueued_at:
        Lane-side enqueue timestamp.
    raw:
        The verbatim descriptor, retained for diagnostics.
    """

    job_id: str
    attempt_nonce: str
    kind: str
    stable_id: str
    work_id: str
    campaign_id: Optional[str]
    author_model: Optional[str]
    request_path: Path
    request_sha256: str
    required_output_path: Path
    receipt_path: Path
    backoff_path: Path
    failure_path: Path
    claim_path: Path
    effort_grant: EffortGrant
    stall_timeout_seconds: float
    enqueued_at: str
    raw: JsonObject

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "QueueJob":
        """Parse and structurally validate one pending descriptor.

        Parameters
        ----------
        value:
            Raw ``pending/<job_id>.json`` payload.

        Returns
        -------
        QueueJob
            Parsed job.

        Raises
        ------
        AuthorQueueError
            When a required field is missing, the kind is outside the closed vocabulary,
            or a published path is not absolute.
        """

        required = (
            "job_id",
            "attempt_nonce",
            "kind",
            "stable_id",
            "work_id",
            "request_path",
            "request_sha256",
            "required_output_path",
            "receipt_path",
            "backoff_path",
            "failure_path",
            "claim_path",
            "effort_grant",
        )
        missing = [name for name in required if value.get(name) in (None, "")]
        if missing:
            raise AuthorQueueError(f"author job descriptor omits {sorted(missing)}")
        kind = str(value["kind"])
        if kind not in AUTHOR_JOB_KINDS:
            raise AuthorQueueError(f"author job names an unsupported kind: {kind!r}")
        paths: dict[str, Path] = {}
        for name in (
            "request_path",
            "required_output_path",
            "receipt_path",
            "backoff_path",
            "failure_path",
            "claim_path",
        ):
            path = Path(str(value[name]))
            if not path.is_absolute():
                raise AuthorQueueError(f"author job {name} must be absolute: {path}")
            paths[name] = path
        campaign = value.get("campaign_id")
        model = value.get("author_model")
        return cls(
            job_id=str(value["job_id"]),
            attempt_nonce=str(value["attempt_nonce"]),
            kind=kind,
            stable_id=str(value["stable_id"]),
            work_id=str(value["work_id"]),
            campaign_id=None if campaign is None else str(campaign),
            author_model=None if model is None else str(model),
            request_path=paths["request_path"],
            request_sha256=str(value["request_sha256"]),
            required_output_path=paths["required_output_path"],
            receipt_path=paths["receipt_path"],
            backoff_path=paths["backoff_path"],
            failure_path=paths["failure_path"],
            claim_path=paths["claim_path"],
            effort_grant=EffortGrant.from_mapping(value["effort_grant"]),
            stall_timeout_seconds=float(value.get("stall_timeout_seconds", 0.0) or 0.0),
            enqueued_at=str(value.get("enqueued_at", "")),
            raw=dict(value),
        )

    def read_request(self) -> JsonObject:
        """Return the request envelope, verifying its published digest.

        Returns
        -------
        dict[str, Any]
            Parsed request envelope.

        Raises
        ------
        AuthorQueueError
            When the request is unreadable or its bytes do not match
            ``request_sha256``.
        """

        try:
            data = self.request_path.read_bytes()
        except OSError as exc:
            raise AuthorQueueError(f"author job {self.job_id} request is unreadable: {exc}")
        observed = hash_bytes(data)
        if observed != self.request_sha256:
            raise AuthorQueueError(
                f"author job {self.job_id} request digest {observed} does not match the "
                f"published {self.request_sha256}"
            )
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthorQueueError(f"author job {self.job_id} request is not JSON: {exc}")
        if not isinstance(value, dict):
            raise AuthorQueueError(f"author job {self.job_id} request is not an object")
        return value


def author_queue_directories(queue_root: Path) -> dict[str, Path]:
    """Return the fixed author-queue subdirectory layout.

    Parameters
    ----------
    queue_root:
        Campaign-local author-queue root.

    Returns
    -------
    dict[str, Path]
        ``pending``, ``claimed``, ``receipts``, and ``signals`` roots.
    """

    return {name: Path(queue_root) / name for name in AUTHOR_QUEUE_DIRECTORIES}


def ensure_author_queue(queue_root: Path) -> dict[str, Path]:
    """Create the queue layout if absent and return it.

    Parameters
    ----------
    queue_root:
        Campaign-local author-queue root.

    Returns
    -------
    dict[str, Path]
        Created subdirectory roots.
    """

    directories = author_queue_directories(queue_root)
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def job_paths(queue_root: Path, job_id: str) -> dict[str, Path]:
    """Return every queue path owned by one job.

    Parameters
    ----------
    queue_root:
        Campaign-local author-queue root.
    job_id:
        Job identity.

    Returns
    -------
    dict[str, Path]
        ``pending``, ``claim``, ``receipt``, ``backoff``, and ``failure`` paths.
    """

    directories = author_queue_directories(queue_root)
    return {
        "pending": directories["pending"] / f"{job_id}.json",
        "claim": directories["claimed"] / f"{job_id}.json",
        "receipt": directories["receipts"] / f"{job_id}.json",
        "backoff": directories["signals"] / f"{job_id}.backoff.json",
        "failure": directories["signals"] / f"{job_id}.failure.json",
    }


def author_job_id(kind: str, work_id: str, request_sha256: str) -> str:
    """Return the deterministic filesystem-safe job identity.

    Parameters
    ----------
    kind, work_id, request_sha256:
        Round-trip label, work generation, and exact request digest.

    Returns
    -------
    str
        ``<kind>-<digest>`` identity, stable across retries of one request.
    """

    digest = stable_hash({"kind": kind, "work_id": work_id, "request": request_sha256})
    return f"{kind}-{digest.removeprefix('sha256:')[:24]}"


def build_job_descriptor(
    *,
    queue_root: Path,
    kind: str,
    stable_id: str,
    work_id: str,
    request_path: Path,
    required_output_path: Path,
    effort_grant: EffortGrant,
    stall_timeout_seconds: float,
    attempt_nonce: str,
    author_model: Optional[str],
    campaign_id: Optional[str],
    enqueued_at: str,
) -> JsonObject:
    """Build the canonical pending-job wire payload.

    Parameters
    ----------
    queue_root:
        Campaign-local author-queue root.
    kind, stable_id, work_id:
        Round-trip label and trusted model/work identities.
    request_path, required_output_path:
        Request envelope and the exact result publication path.
    effort_grant:
        Per-session effort budget published to the pool.
    stall_timeout_seconds:
        Outer lane deadline.
    attempt_nonce:
        Per-attempt nonce every pool-published file must echo.
    author_model, campaign_id:
        Frozen campaign bindings when available.
    enqueued_at:
        Lane-side enqueue timestamp.

    Returns
    -------
    dict[str, Any]
        Complete descriptor including its self-check digest.
    """

    resolved_request_path = Path(request_path).resolve()
    request_sha256 = hash_bytes(resolved_request_path.read_bytes())
    job_id = author_job_id(kind, work_id, request_sha256)
    paths = job_paths(Path(queue_root).resolve(), job_id)
    job: JsonObject = {
        "envelope_version": AUTHOR_JOB_VERSION,
        "job_id": job_id,
        "attempt_nonce": attempt_nonce,
        "kind": kind,
        "stable_id": stable_id,
        "work_id": work_id,
        "author_model": author_model,
        "campaign_id": campaign_id,
        "request_path": str(resolved_request_path),
        "request_sha256": request_sha256,
        "required_output_path": str(Path(required_output_path).resolve()),
        "receipt_path": str(paths["receipt"]),
        "backoff_path": str(paths["backoff"]),
        "failure_path": str(paths["failure"]),
        "claim_path": str(paths["claim"]),
        "effort_grant": effort_grant.to_dict(),
        "stall_timeout_seconds": float(stall_timeout_seconds),
        "enqueued_at": enqueued_at,
    }
    job["job_sha256"] = stable_hash(job)
    QueueJob.from_mapping(job)
    return job


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Publish one JSON payload atomically and durably.

    Parameters
    ----------
    path:
        Destination path.
    payload:
        JSON-serializable payload.

    Returns
    -------
    Path
        The written path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, canonical_json_bytes(payload) + b"\n")
    fsync_directory(path.parent)
    return path


def read_json(path: Path) -> Optional[JsonObject]:
    """Read one JSON object, tolerating a not-yet-complete write.

    Parameters
    ----------
    path:
        Candidate file.

    Returns
    -------
    dict[str, Any] | None
        Parsed object, or ``None`` when absent, unreadable, partially written, or not an
        object. A partial read is transient by construction: the caller re-polls.
    """

    try:
        data = path.read_bytes()
    except (OSError, ValueError):
        return None
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def stamp(job: QueueJob, payload: Mapping[str, Any]) -> JsonObject:
    """Bind one pool-published payload to this job attempt.

    This is the single chokepoint for the nonce-echo invariant. Every builder below routes
    through it, so no pool file can be published without the identity the lane matches on.

    Parameters
    ----------
    job:
        Job being answered.
    payload:
        Payload body.

    Returns
    -------
    dict[str, Any]
        Payload carrying ``job_id`` and ``attempt_nonce``.
    """

    return {**dict(payload), "job_id": job.job_id, "attempt_nonce": job.attempt_nonce}


def build_claim(job: QueueJob, *, owner: str, lease_expires_at: str) -> JsonObject:
    """Build one lease payload.

    Parameters
    ----------
    job:
        Job being claimed.
    owner:
        Managing-session identity servicing the job.
    lease_expires_at:
        ISO-8601 instant after which another servicer may re-claim.

    Returns
    -------
    dict[str, Any]
        Claim payload. Liveness only: it never authorizes a result.
    """

    return stamp(
        job,
        {
            "envelope_version": AUTHOR_CLAIM_VERSION,
            "kind": job.kind,
            "stable_id": job.stable_id,
            "owner": owner,
            "claimed_at": utc_now(),
            "lease_expires_at": lease_expires_at,
        },
    )


def build_receipt(
    job: QueueJob,
    *,
    consumption: Consumption,
    author_model: Optional[str] = None,
    result_sha256: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
) -> JsonObject:
    """Build one completion receipt.

    Parameters
    ----------
    job:
        Completed job.
    consumption:
        Honestly measured effort spent. Validated against the job's grant here so the pool
        never publishes a receipt the lane would refuse.
    author_model:
        Model that actually authored, for audit against the campaign's frozen identity.
    result_sha256:
        Digest of the bytes published at ``required_output_path``.
    notes:
        Bounded free-form operator notes.

    Returns
    -------
    dict[str, Any]
        Receipt payload.

    Raises
    ------
    AuthorQueueError
        When the declared consumption exceeds the published grant.
    """

    consumption.validate_against(job.effort_grant)
    payload: JsonObject = {
        "envelope_version": AUTHOR_RECEIPT_VERSION,
        "kind": job.kind,
        "stable_id": job.stable_id,
        "completed_at": utc_now(),
        "required_output_path": str(job.required_output_path),
        "consumption": consumption.to_dict(),
        "effort_grant": job.effort_grant.to_dict(),
    }
    if author_model is not None:
        payload["author_model"] = author_model
    if result_sha256 is not None:
        payload["result_sha256"] = result_sha256
    if notes:
        payload["notes"] = [str(note)[:500] for note in notes][:10]
    return stamp(job, payload)


def build_backoff(
    job: QueueJob,
    *,
    provider: str = "anthropic",
    reason: str = "quota-exhausted",
    response_excerpt: str = "",
    reset_at: Optional[str] = None,
    retry_after_seconds: Optional[int] = None,
) -> JsonObject:
    """Build one typed provider-pause sidecar.

    Parameters
    ----------
    job:
        Job that hit a provider limit.
    provider:
        Usage-limit provider, inside the closed vocabulary.
    reason:
        ``rate-limit`` or ``quota-exhausted``.
    response_excerpt:
        Bounded verbatim provider text; the lane parses a reset instant from it when
        ``reset_at`` is absent.
    reset_at:
        Provider-reported reset instant.
    retry_after_seconds:
        Provider-reported retry delay.

    Returns
    -------
    dict[str, Any]
        Backoff payload.

    Raises
    ------
    AuthorQueueError
        When the provider or reason is outside its closed vocabulary.
    """

    if provider not in USAGE_LIMIT_PROVIDERS:
        raise AuthorQueueError(f"author backoff names an unsupported provider: {provider!r}")
    if reason not in {"rate-limit", "quota-exhausted"}:
        raise AuthorQueueError(f"author backoff names an unsupported reason: {reason!r}")
    payload: JsonObject = {
        "envelope_version": AUTHOR_BACKOFF_VERSION,
        "kind": job.kind,
        "stable_id": job.stable_id,
        "provider": provider,
        "reason": reason,
        "response_excerpt": str(response_excerpt)[:1_500],
        "signalled_at": utc_now(),
    }
    if reset_at is not None:
        payload["reset_at"] = str(reset_at)
    if retry_after_seconds is not None:
        payload["retry_after_seconds"] = int(retry_after_seconds)
    return stamp(job, payload)


def build_failure(job: QueueJob, *, reason: str, retryable: bool, detail: str = "") -> JsonObject:
    """Build one typed failure sidecar.

    Parameters
    ----------
    job:
        Failed job.
    reason:
        Short machine-readable failure label.
    retryable:
        Explicit classification. The lane refuses a sidecar without one rather than
        guessing, so this argument is required and must be a real ``bool``.
    detail:
        Bounded diagnostic detail.

    Returns
    -------
    dict[str, Any]
        Failure payload.

    Raises
    ------
    AuthorQueueError
        When ``retryable`` is not a boolean or ``reason`` is empty.
    """

    if not isinstance(retryable, bool):
        raise AuthorQueueError("author failure requires an explicit boolean retryable")
    if not str(reason).strip():
        raise AuthorQueueError("author failure requires a nonempty reason")
    return stamp(
        job,
        {
            "envelope_version": AUTHOR_FAILURE_VERSION,
            "kind": job.kind,
            "stable_id": job.stable_id,
            "reason": str(reason),
            "retryable": retryable,
            "detail": str(detail)[:4_000],
            "signalled_at": utc_now(),
        },
    )


def matches_attempt(payload: Optional[Mapping[str, Any]], job: QueueJob) -> bool:
    """Return whether one pool file belongs to this exact attempt.

    Parameters
    ----------
    payload:
        Parsed candidate payload, or ``None``.
    job:
        Job whose attempt nonce is authoritative.

    Returns
    -------
    bool
        ``True`` only when both the job identity and the attempt nonce match. This mirrors
        the lane's own rule so the wrapper and the pool cannot disagree about which file
        answers which attempt.
    """

    if payload is None:
        return False
    return (
        payload.get("job_id") == job.job_id
        and payload.get("attempt_nonce") == job.attempt_nonce
    )
