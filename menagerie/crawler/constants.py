"""Closed crawler vocabularies and fixed contract values."""

from __future__ import annotations

from enum import Enum

MODEL_SCHEMA_VERSION = "menagerie.crawler.model.v2"
ATTEMPT_SCHEMA_VERSION = "menagerie.crawler.attempt.v2"
GATE_SCHEMA_VERSION = "menagerie.crawler.gate.v2"
AUTHOR_PROPOSAL_SCHEMA_VERSION = "menagerie.crawler.author-proposal.v2"
OPERATIONAL_EVENT_SCHEMA_VERSION = "menagerie.crawler.operational-event.v1"

AUTHOR_PROMPT_NAME = "claude_crawler_author_v2"
CHECKER_PROMPT_NAME = "codex_accuracy_checker_v2"

METADATA_BATCH_MIN = 10
METADATA_BATCH_MAX = 20
STDIO_TAIL_MAX_CHARS = 1_500
STABLE_ID_DIGEST_CHARS = 20
DEFAULT_FORWARD_TIMEOUT_SECONDS = 300
MAX_FORWARD_TIMEOUT_SECONDS = 1_800
DEFAULT_NOTIFY_TIMEOUT_SECONDS = 5
DEFAULT_REVIEW_CHECKPOINT_MODELS = 1_000
DEFAULT_PROGRESS_NOTIFICATION_MILESTONES = (2_000, 3_000, 5_000, 10_000, 15_000, 20_000)


class StrEnum(str, Enum):
    """String-valued enum compatible with all supported Python versions."""


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


# Slice-F scheduler configuration defaults.  The earlier names remain the
# compatibility surface used by Slice E's event builders.
DEFAULT_REVIEW_CHECKPOINT_AT = DEFAULT_REVIEW_CHECKPOINT_MODELS
DEFAULT_PROGRESS_MILESTONES = DEFAULT_PROGRESS_NOTIFICATION_MILESTONES
DEFAULT_NOTIFY_COMMAND: str | None = None
