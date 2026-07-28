"""Closed crawler vocabularies and fixed contract values."""

from __future__ import annotations

from enum import Enum

MODEL_SCHEMA_VERSION = "menagerie.crawler.model.v2"
ATTEMPT_SCHEMA_VERSION = "menagerie.crawler.attempt.v2"
GATE_SCHEMA_VERSION = "menagerie.crawler.gate.v2"
AUTHOR_PROPOSAL_SCHEMA_VERSION = "menagerie.crawler.author-proposal.v2"
OPERATIONAL_EVENT_SCHEMA_VERSION = "menagerie.crawler.operational-event.v1"
EXECUTION_READ_MANIFEST_VERSION_V2 = "menagerie.crawler.execution-read-manifest.v2"
EXECUTION_READ_MANIFEST_VERSION_V3 = "menagerie.crawler.execution-read-manifest.v3"
ENVIRONMENT_AUTHORITY_VERSION_V1 = "menagerie.crawler.environment-authority.v1"
ENVIRONMENT_CONTENT_MANIFEST_VERSION_V1 = "menagerie.crawler.environment-content-manifest.v1"
ENVIRONMENT_GENERATION_VERSION_V2 = "menagerie.crawler.environment-generation.v2"
DRIVER_SHUTDOWN_STATUS = "interrupted:shutdown"

# Round-14 contracts are additive during the interface freeze. Existing producers
# continue to emit v2 until the authority kernel switches atomically; v3 writers
# use these explicit current-version discriminators and may read immutable v2 history.
MODEL_SCHEMA_VERSION_V3 = "menagerie.crawler.model.v3"
ATTEMPT_SCHEMA_VERSION_V3 = "menagerie.crawler.attempt.v3"
GATE_SCHEMA_VERSION_V3 = "menagerie.crawler.gate.v3"
AUTHOR_PROPOSAL_SCHEMA_VERSION_V3 = "menagerie.crawler.author-proposal.v3"
AUTHOR_RESULT_SCHEMA_VERSION_V3 = "menagerie.crawler.author-result.v3"
AUTHOR_RESULT_SCHEMA_VERSION = "menagerie.crawler.author-result.v4"
PROMOTION_SCHEMA_VERSION = "menagerie.crawler.promotion.v1"
SOURCE_DISCOVERY_SCHEMA_VERSION = "menagerie.crawler.source-discovery.v1"
ARTIFACT_EVENT_SCHEMA_VERSION = "menagerie.crawler.artifact-event.v1"

CURRENT_SCHEMA_VERSIONS = frozenset(
    {
        MODEL_SCHEMA_VERSION_V3,
        ATTEMPT_SCHEMA_VERSION_V3,
        GATE_SCHEMA_VERSION_V3,
        AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
        AUTHOR_RESULT_SCHEMA_VERSION,
        ARTIFACT_EVENT_SCHEMA_VERSION,
        OPERATIONAL_EVENT_SCHEMA_VERSION,
    }
)

LEGACY_UNTRUSTED_SCHEMA_VERSIONS = frozenset(
    {
        MODEL_SCHEMA_VERSION,
        ATTEMPT_SCHEMA_VERSION,
        GATE_SCHEMA_VERSION,
        AUTHOR_PROPOSAL_SCHEMA_VERSION,
        AUTHOR_RESULT_SCHEMA_VERSION_V3,
    }
)

AUTHOR_PROMPT_NAME = "claude_crawler_author_v2"
CHECKER_PROMPT_NAME = "codex_accuracy_checker_v2"

METADATA_BATCH_MIN = 10
METADATA_BATCH_MAX = 20
METADATA_FINAL_TAIL_MIN = 1
STDIO_TAIL_MAX_CHARS = 1_500
STABLE_ID_DIGEST_CHARS = 20
DEFAULT_FORWARD_TIMEOUT_SECONDS = 300
MAX_FORWARD_TIMEOUT_SECONDS = 1_800
DEFAULT_NOTIFY_TIMEOUT_SECONDS = 5
DEFAULT_REVIEW_CHECKPOINT_MODELS = 1_000
DEFAULT_PROGRESS_NOTIFICATION_MILESTONES = (
    900,
    950,
    1_000,
    2_000,
    3_000,
    5_000,
    10_000,
    15_000,
    20_000,
)

# `PLAN.md` LP-13.2 author-session effort ceiling. The pool enforces the tool-call
# and wall budgets because it is the only boundary that observes Agent-tool events;
# the lane enforces the fetch-target count, which it observes directly, and audits
# the pool's declared consumption against the same grant.
AUTHOR_MAX_TOOL_CALLS = 30
AUTHOR_MAX_FETCH_TARGETS = 20
AUTHOR_SESSION_WALL_SECONDS = 30 * 60
# Outer stall guard. A managing session that dies must look like a stalled queue
# (retryable infrastructure), never a failed model.
AUTHOR_QUEUE_STALL_SECONDS = 45 * 60
AUTHOR_QUEUE_POLL_SECONDS = 2.0

# ---------------------------------------------------------------------------
# The author wall budget, single-sourced (SEAM_REDESIGN 3.7)
#
# Wall grants are PER CAMPAIGN, not a global constant: c3-classics carries the
# no-prior-code classics whose mean session is ~25 min with a long tail, and
# truncating that tail both loses the models and corrupts the very p95 the
# month's go/no-go depends on. Everything downstream -- the grant published to
# the session, the executor's own kill, and the lane's outer stall bound --
# derives from `resolve_author_wall_seconds` so no lower level can silently
# impose a stricter number of its own.
# ---------------------------------------------------------------------------

#: Per-campaign wall grants (seconds). Absent campaigns take the default grant.
AUTHOR_CAMPAIGN_WALL_SECONDS: dict[str, float] = {"c3-classics": 60.0 * 60.0}

#: The executor SIGKILLs its own `claude -p` child at grant x this factor. The
#: brief carries the deadline itself, so a session watching its clock lands a
#: typed BLOCKED-with-partial inside the grant and this only catches the ones
#: that do not.
AUTHOR_WALL_EXTERNAL_KILL_FACTOR = 1.10

#: Worst-case number of grant-sized `claude -p` sessions ONE executor invocation
#: may legitimately run. Stage 2 is the worst case: the primary run (1.0), one
#: cold rerun after a provider resume failure (1.0), and at most one typed
#: supplementary broker round at half the grant (0.5). The lane's bound must
#: clear this, or a stage 2 that legitimately took the cold-rerun path is killed
#: from outside while still inside its budget. `test_author_wall_budget.py` pins
#: this against the executor's actual session call sites.
AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET = 2.5

#: Slack for the work an invocation does OUTSIDE its `claude -p` children --
#: broker fetches, hashing, attempt-record writes, publication.
AUTHOR_LANE_WALL_MARGIN_SECONDS = 120.0

#: Environment variable carrying the resolved grant to the executor subprocess.
#: The driver PUBLISHES it from the campaign's authoritative grant rather than
#: trusting the operator to set it consistently by hand; a pre-set value that
#: disagrees is a startup failure, never a silent winner.
AUTHOR_WALL_SECONDS_ENV = "MENAGERIE_AUTHOR_WALL_SECONDS"


def resolve_author_wall_seconds(
    campaign_id: "str | None", override: "float | None" = None
) -> float:
    """Return the authoritative wall grant for one campaign, in seconds.

    This is the ONLY place a wall grant is chosen. An explicit operator
    override wins; otherwise the campaign's own grant; otherwise the default.

    Parameters
    ----------
    campaign_id:
        Tier campaign identity, or ``None`` outside a tier campaign.
    override:
        Explicit operator grant, when one was configured.

    Returns
    -------
    float
        Strictly positive wall grant in seconds.

    Raises
    ------
    ValueError
        If an override is not a finite, strictly positive number of seconds.
    """

    if override is not None:
        value = float(override)
        if not value > 0.0 or value != value or value in (float("inf"), float("-inf")):
            raise ValueError(
                f"author wall grant must be a finite positive number of seconds, not {override!r}"
            )
        return value
    return float(
        AUTHOR_CAMPAIGN_WALL_SECONDS.get(campaign_id or "", float(AUTHOR_SESSION_WALL_SECONDS))
    )


def author_lane_wall_bound(grant_seconds: float) -> float:
    """Return the lane's outer stall bound for a given per-session grant.

    The lane bound is a STALL GUARD, not a budget. The budget is the grant, and
    the executor enforces it per session; the lane only exists to stop a wedged
    wrapper blocking the driver forever. It must therefore be strictly greater
    than every inner limit, or it preempts the typed, recoverable outcomes the
    inner layers produce -- which is exactly the silent 30-minute truncation
    this derivation replaced.

    Parameters
    ----------
    grant_seconds:
        Authoritative per-session wall grant.

    Returns
    -------
    float
        Bound covering one invocation's worst-case session budget plus the
        non-session work around it.
    """

    return (
        float(grant_seconds)
        * AUTHOR_EXECUTOR_INVOCATION_SESSION_BUDGET
        * AUTHOR_WALL_EXTERNAL_KILL_FACTOR
        + AUTHOR_LANE_WALL_MARGIN_SECONDS
    )

# Bounded author-session fan-out per wave. The author lane is ~74% of the campaign's
# projected work, and a serial lane caps a four-campaign fleet at four concurrent
# sessions -- below the ~6.2 sustained (~9.5 in flight) the reconciled schedule needs,
# before anything else goes wrong. Four per campaign puts ~16 sessions in flight across
# the fleet, ~1.7x the requirement, which absorbs stragglers and quota stalls without
# assuming a perfect duty cycle. It is also a modest per-host budget: four concurrent
# `claude -p` subprocesses (command lane) or four concurrent in-session subagents (queue
# lane) sit well inside both the host's memory and the harness's own subagent ceiling.
# The right value genuinely differs per lane and per host, so it is configurable; this
# is only the defensible default.
DEFAULT_AUTHOR_WAVE_CONCURRENCY = 4
# Hard ceiling. Past this the bound stops being a bound: a wave would fan out further
# than any provider tier or host can service, and the failure mode is mass quota
# exhaustion rather than throughput.
MAX_AUTHOR_WAVE_CONCURRENCY = 32

# Closed usage-limit provider vocabulary shared by the pause path and the wakeup
# layer. The checker lane pauses on `openai`, the author lane on `anthropic`.
USAGE_LIMIT_PROVIDERS = frozenset({"anthropic", "openai"})

# The four frozen TIER campaigns the partitioner emits, each bound to its frozen
# author model. This is deliberately NOT the same concept as a *repair* campaign
# (`campaign-<stable_id>` / `campaign-<work_id>`), which is the driver's per-item
# authority lineage. A tier campaign is a property of the whole campaign run: its
# `author_model_identity` is frozen for the run, so a model a sonnet campaign finds
# genuinely hard is emitted as a typed BLOCKED recommendation and requeued into the
# opus campaign, never escalated in place. Mixing the two identities up would author
# a model with the wrong tier and corrupt the frozen identity for the whole run, so
# every boundary that selects an author tier validates against this closed set.
# `menagerie.crawler.partitioner.CAMPAIGN_SPECS` carries the same binding for the
# partitioner's own purposes; the two are asserted to agree in the test suite.
TIER_CAMPAIGN_AUTHOR_MODELS: dict[str, str] = {
    "c1-mech": "claude-sonnet",
    "c2-disco": "claude-sonnet",
    "c3-classics": "claude-opus-5",
    "c4-native": "claude-sonnet",
}
TIER_CAMPAIGN_IDS = frozenset(TIER_CAMPAIGN_AUTHOR_MODELS)

#: Environment variable naming the tier campaign this operator process serves.
TIER_CAMPAIGN_ENV = "MENAGERIE_CAMPAIGN_ID"


class StrEnum(str, Enum):
    """String-valued enum compatible with all supported Python versions."""


class InvocationOrigin(StrEnum):
    """Closed origin of one driver invocation for wake-episode transitions."""

    ORDINARY_RUN = "ordinary-run"
    MANUAL_RESUME = "manual-resume"
    WAKE_CALLBACK = "wake-callback"


class AuthoredMetadataState(StrEnum):
    """State of source-read fields in a model revision."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    FAILED = "failed"


class StatusKind(StrEnum):
    """Closed public terminal status kinds."""

    RUNS = "runs"
    DEFERRED = "deferred"
    SKIPPED = "skipped"
    FAILED = "failed"


class FailureStage(StrEnum):
    """Closed failure stages from the canonical plan."""

    INTAKE = "intake"
    SOURCE = "source"
    FETCH = "fetch"
    AUTHOR = "author"
    EVIDENCE = "evidence"
    ACCURACY_GATE = "accuracy-gate"
    ENVIRONMENT = "environment"
    IMPORT = "import"
    CONSTRUCTOR = "constructor"
    INPUT = "input"
    FORWARD = "forward"
    FIDELITY = "fidelity"
    RESOURCE = "resource"
    POLICY = "policy"
    SANDBOX_UNAVAILABLE = "policy"
    RUNNER = "runner"


class SourceRung(StrEnum):
    """Ordered source-resolution ladder."""

    LIBRARY = "R1_LIBRARY"
    VENDOR = "R2_VENDOR"
    PORT = "R3_PORT"
    REIMPLEMENT = "R4_REIMPLEMENT"
    SKIP = "R5_SKIP"


class GateKind(StrEnum):
    """Checker envelope kinds."""

    METADATA_BATCH = "metadata_batch"
    FIDELITY = "fidelity"
    TERMINAL_DISPOSITION = "terminal_disposition"


class AccuracyVerdict(StrEnum):
    """Closed metadata/integrity checker verdicts."""

    ACCURATE = "accurate"
    INACCURATE = "inaccurate"
    CANNOT_VERIFY = "cannot-verify"


class FidelityVerdict(StrEnum):
    """Closed fidelity checker verdicts."""

    MATCH = "match"
    MINOR_DRIFT = "minor-drift"
    MAJOR_DRIFT = "major-drift"
    SLOP = "slop"
    CANNOT_VERIFY = "cannot-verify"


class RetrievalStatus(StrEnum):
    """Closed controlled-fetch retrieval outcomes."""

    FETCHED = "fetched"
    ALREADY_PRESENT = "already-present"


class GateRoute(StrEnum):
    """Closed deterministic routes after a checker verdict."""

    ACCEPT = "accept"
    REQUEUE_NEXT_BATCH = "requeue-next-batch"
    HUMAN_FAIL = "human-fail"
    BLOCK_FIDELITY = "block-fidelity"


class CheckerPauseReason(StrEnum):
    """Closed checker responses that require a scheduler pause."""

    RATE_LIMIT = "rate-limit"
    QUOTA_EXHAUSTED = "quota-exhausted"


class AuthorPauseReason(StrEnum):
    """Closed author responses that require a scheduler pause.

    The author-side analogue of :class:`CheckerPauseReason`. Anthropic usage
    exhaustion is a provider pause with a reset time, never a model failure.
    """

    RATE_LIMIT = "rate-limit"
    QUOTA_EXHAUSTED = "quota-exhausted"


class RunMode(StrEnum):
    """Meaningful model runtime modes."""

    TRAIN = "train"
    EVAL = "eval"


class AttemptResult(StrEnum):
    """Immutable attempt outcomes."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OBSERVED = "observed"


class EnvironmentPhase(StrEnum):
    """Ordered environment execution phases."""

    PYTORCH = "pytorch"
    NATIVE_TAIL = "native-tail"


class PlatformRequirement(StrEnum):
    """Platform capabilities that can support an evidenced deferral."""

    CUDA = "cuda"
    X86 = "x86"


SKIPPED_STATUS_CODES = frozenset(
    {
        "skipped:insufficient-description",
        "skipped:no-description",
        "skipped:not-a-real-NN",
    }
)

TERMINAL_STATUS_CODES = frozenset(
    {
        "runs",
        "deferred:needs-cuda",
        "deferred:needs-x86",
        "deferred:needs-opus-tier",
        *SKIPPED_STATUS_CODES,
        *(f"failed:{stage.value}" for stage in FailureStage),
    }
)

FAILURE_REASON_CODES: dict[str, frozenset[str]] = {
    "intake": frozenset(
        {
            "schema-invalid",
            "stable-id-conflict",
            "duplicate-revision-conflict",
            "migration-invariant",
        }
    ),
    "source": frozenset(
        {
            "identity-unresolved",
            "missing-mandatory-link",
            "source-model-mismatch",
            "source-target-invalid",
            "higher-rung-unresolved",
            "effort-cap-exhausted",
        }
    ),
    "fetch": frozenset(
        {
            "unreachable",
            "revision-missing",
            "hash-mismatch",
            "access-denied",
            "artifact-missing",
            "effort-cap-exhausted",
        }
    ),
    "author": frozenset(
        {
            "effort-exhausted:tool-calls",
            "effort-exhausted:fetch-targets",
            "effort-exhausted:wall-seconds",
            "wall-exceeded",
            "session-crashed",
            "research-tools-unavailable",
            "repair-exhausted",
        }
    ),
    "evidence": frozenset(
        {
            "locator-missing",
            "excerpt-mismatch",
            "insufficient-detail",
            "coverage-incomplete",
            "search-incomplete",
            "effort-cap-exhausted",
        }
    ),
    "accuracy-gate": frozenset(
        {
            "inaccurate-cap-exhausted",
            "cannot-verify-cap-exhausted",
            "identity-mismatch",
            "checker-contract-invalid",
            "effort-cap-exhausted",
        }
    ),
    "environment": frozenset(
        {
            "solve-failed",
            "lock-missing",
            "artifact-hash-mismatch",
            "build-failed",
            "probe-failed",
            "resolved-export-mismatch",
            "island-cap",
            "below-minimum-island-size",
            "effort-cap-exhausted",
        }
    ),
    "import": frozenset(
        {
            "module-missing",
            "symbol-missing",
            "abi-load-failed",
            "import-exception",
            "effort-cap-exhausted",
        }
    ),
    "constructor": frozenset(
        {
            "exception",
            "requires-checkpoint",
            "requires-weight-asset",
            "invalid-model-object",
            "effort-cap-exhausted",
        }
    ),
    "input": frozenset(
        {
            "contract-invalid",
            "source-invalid-shape",
            "generation-exception",
            "semantic-constraint",
            "effort-cap-exhausted",
        }
    ),
    "forward": frozenset(
        {
            "exception",
            "mode-run",
            "incomplete-receipt",
            "invalid-output-signature",
            "confirmation-mismatch",
            "effort-cap-exhausted",
        }
    ),
    "fidelity": frozenset(
        {
            "major-drift-cap-exhausted",
            "slop-cap-exhausted",
            "cannot-verify-cap-exhausted",
            "identity-mismatch",
            "effort-cap-exhausted",
        }
    ),
    "resource": frozenset(
        {"timeout", "oom", "disk-floor", "scratch-cap", "rss-cap", "effort-cap-exhausted"}
    ),
    "policy": frozenset(
        {
            "network-attempt",
            "checkpoint-read",
            "write-outside-scratch",
            "credentials-exposed",
            "torchlens-import",
            "opaque-code",
            "sandbox-unavailable-v1",
            "effort-cap-exhausted",
        }
    ),
    "runner": frozenset(
        {
            "native-crash",
            "signal",
            "missing-receipt",
            "protocol-violation",
            "ledger-corruption",
            "internal-error",
            "effort-cap-exhausted",
        }
    ),
}

WORKFLOW_STATES = frozenset(
    {
        "UNTRIAGED",
        "queued",
        "authoring",
        "awaiting-gate",
        "fidelity-pending",
        "environment-pending",
        "forward-observed-but-blocked",
        "paused:usage-limit",
    }
)

INPUT_KINDS = frozenset(
    {
        "standard-image",
        "standard-text",
        "standard-audio",
        "standard-video",
        "standard-tabular",
        "standard-pointcloud",
        "random-fallback",
    }
)


class OperationalEventKind(StrEnum):
    """Closed operational-event kinds."""

    USAGE_PAUSE = "usage-pause"
    USAGE_RESUME = "usage-resume"
    WAKEUP = "wakeup"
    CHECKPOINT = "checkpoint"
    CAMPAIGN_HEALTH = "campaign-health"
    WAKE_NOOP_ALREADY_RUNNING = "wake-noop-already-running"
    CHECKPOINT_REVIEW = "checkpoint-review"
    REVIEW_SIGNOFF = "review-signoff"
    PROGRESS_NOTIFICATION = "progress-notification"
    NOTIFICATION_DELIVERY = "notification-delivery"
    REQUEUE_GRANT_CONSUMED = "requeue-grant-consumed"
    WORKER_LEASE_OPENED = "worker-lease-opened"
    WORKER_LEASE_STARTED = "worker-lease-started"
    WORKER_LEASE_CLOSED = "worker-lease-closed"
    WORKER_LEASE_REAPED = "worker-lease-reaped"
    ORPHAN_WORKER_RECOVERED = "orphan-worker-recovered"
    WAKEUP_INSTALLED = "wakeup-installed"
    WAKEUP_REPAIRED = "wakeup-repaired"
    WAKEUP_FIRED = "wakeup-fired"
    WAKEUP_DEACTIVATED = "wakeup-deactivated"
    WAKEUP_HEALTH_DEGRADED = "wakeup-health-degraded"
    WAKEUP_FAILED = "wakeup-failed"
    CAMPAIGN_COMPLETED = "campaign-completed"
    OPERATOR_CANCELLED = "operator-cancelled"
    WORKER_SHUTDOWN_INTERRUPTED = "worker-shutdown-interrupted"


class OperationalEventStatus(StrEnum):
    """Closed operational-event dispositions."""

    USAGE_PAUSED = "paused:usage-limit"
    USAGE_RESUMED = "resumed:usage-limit"
    WAKEUP_SCHEDULED = "wakeup-scheduled"
    WAKEUP_FIRED = "wakeup-fired"
    WAKE_NOOP_ALREADY_RUNNING = "wake-noop-already-running"
    CHECKPOINT_COMPLETE = "checkpoint-complete"
    CHECKPOINT_FAILED = "checkpoint-failed"
    HEALTHY = "healthy"
    RUNNER_FAILED = "failed:runner"
    RETRYABLE_INFRASTRUCTURE = "retryable:infrastructure"
    CHECKPOINT_REVIEW_PAUSED = "paused:checkpoint-review"
    REVIEW_SIGNED_OFF = "resumed:checkpoint-review"
    PROGRESS_NOTIFIED = "progress-notified"
    PROGRESS_RECORDED = "progress-recorded"
    NOTIFICATION_DELIVERED = "notification-delivered"
    NOTIFICATION_FAILED = "notification-failed"
    REQUEUE_GRANT_CONSUMED = "requeue-grant-consumed"
    WORKER_LEASE_OPEN = "worker-lease-open"
    WORKER_LEASE_ACTIVE = "worker-lease-active"
    WORKER_LEASE_CLOSED = "worker-lease-closed"
    WORKER_LEASE_REAPED = "worker-lease-reaped"
    ORPHAN_WORKER_RECOVERED = "orphan-worker-recovered"
    WAKEUP_INSTALLED = "wakeup-installed"
    WAKEUP_REPAIRED = "wakeup-repaired"
    WAKEUP_DEACTIVATED = "wakeup-deactivated"
    WAKEUP_HEALTH_DEGRADED = "wakeup-health-degraded"
    WAKEUP_FAILED = "wakeup-failed"
    CAMPAIGN_COMPLETED = "campaign-completed"
    OPERATOR_CANCELLED = "operator-cancelled"
    TERMINATED = "paused:terminated"
    SHUTDOWN_INTERRUPTED = DRIVER_SHUTDOWN_STATUS


# Slice-F scheduler configuration defaults.  The earlier names remain the
# compatibility surface used by Slice E's event builders.
DEFAULT_REVIEW_CHECKPOINT_AT = DEFAULT_REVIEW_CHECKPOINT_MODELS
DEFAULT_PROGRESS_MILESTONES = DEFAULT_PROGRESS_NOTIFICATION_MILESTONES
DEFAULT_NOTIFY_COMMAND: str | None = None
