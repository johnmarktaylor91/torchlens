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
DEFAULT_PROGRESS_NOTIFICATION_MILESTONES = (2_000, 3_000, 5_000, 10_000, 15_000, 20_000)


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
