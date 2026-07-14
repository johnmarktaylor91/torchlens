"""Never-push validation and allowlisted crawler checkpoint staging."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.licenses import (
    LicenseSweepReport,
    LicensedArtifact,
    PublicMergeRejected,
    pre_public_merge_sweep,
)
from menagerie.crawler.mirrors import ArtifactManifest, MirrorStore
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.wakeup import OperationalContext, record_operational_event

CRAWLER_BRANCH = "menagerie/crawler-pipeline"


class CheckpointError(RuntimeError):
    """Base class for typed checkpoint refusals."""


class WrongCheckpointBranch(CheckpointError):
    """Raised when checkpointing is attempted outside the crawler branch."""


class NonAllowlistedPath(CheckpointError):
    """Raised when a candidate or already-staged path is not checkpoint-safe."""


class SecretBearingPath(CheckpointError):
    """Raised when a candidate path or its bytes appear to contain credentials."""


class CheckpointValidationError(CheckpointError):
    """Raised when a ledger, view, mirror, or license validation fails."""


class RestrictedPublicArtifact(CheckpointValidationError):
    """Raised when the public staged set fails the mandatory license sweep."""


@dataclass(frozen=True)
class FunnelSnapshot:
    """Exact four-bucket current terminal counts."""

    runs: int
    deferred: int
    skipped: int
    failed: int

    def __post_init__(self) -> None:
        """Reject negative funnel counts.

        Raises
        ------
        ValueError
            If any count is negative.
        """

        if min(self.runs, self.deferred, self.skipped, self.failed) < 0:
            raise ValueError("funnel counts cannot be negative")

    def to_dict(self) -> JsonObject:
        """Return the strict schema funnel object.

        Returns
        -------
        dict[str, Any]
            Four exact terminal buckets.
        """

        return {
            "runs": self.runs,
            "deferred": self.deferred,
            "skipped": self.skipped,
            "failed": self.failed,
        }


@dataclass(frozen=True)
class GitCommandResult:
    """Minimal typed result returned by checkpoint Git commands."""

    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CheckpointSet:
    """Validated paths staged without committing or pushing."""

    paths: tuple[Path, ...]
    branch: str
    license_report: LicenseSweepReport


GitRunner = Callable[[Sequence[str], Path], GitCommandResult]
ValidationCheck = Callable[[], None]


_ALLOWLIST_ROOTS = (
    PurePosixPath("menagerie/crawler/records"),
    PurePosixPath("menagerie/crawler/source_manifests"),
    PurePosixPath("menagerie/crawler/mirrors"),
    PurePosixPath("menagerie/crawler/evidence"),
    PurePosixPath("menagerie/crawler/views"),
    PurePosixPath("menagerie/crawler/license_reports"),
)
_ALLOWLIST_SUFFIXES = frozenset({".json", ".jsonl", ".sha256", ".txt"})
_SECRET_PATH_PARTS = frozenset(
    {".env", "credential", "credentials", "secret", "secrets", "token", "private_key"}
)
_SECRET_BYTE_MARKERS = (
    b"-----begin private key-----",
    b"-----begin rsa private key-----",
    b"aws_secret_access_key",
    b"github_token=",
    b"openai_api_key=",
    b"anthropic_api_key=",
)


def build_checkpoint_review_event(
    *,
    models_completed: int,
    funnel_snapshot: FunnelSnapshot,
    report_path: str,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct the blocking human-review checkpoint event.

    Parameters
    ----------
    models_completed, funnel_snapshot, report_path:
        Review threshold, current funnel, and report presented to the human.
    context:
        Common operational context.
    created_at:
        Exact UTC event timestamp.

    Returns
    -------
    dict[str, Any]
        Strict logical operational event.
    """

    _validate_models_completed(models_completed, funnel_snapshot)
    if not report_path.strip():
        raise ValueError("checkpoint review report_path must be non-empty")
    event = _base_milestone_event(
        event_kind=OperationalEventKind.CHECKPOINT_REVIEW,
        status=OperationalEventStatus.CHECKPOINT_REVIEW_PAUSED,
        identity_payload={"models_completed": models_completed, "report_path": report_path},
        context=context,
        created_at=created_at,
    )
    event.update(
        {
            "models_completed": models_completed,
            "funnel_snapshot": funnel_snapshot.to_dict(),
            "report_path": report_path,
        }
    )
    return event


def build_review_signoff_event(
    *,
    approved_by_note: str,
    resume_after: int,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct a human continuation approval event.

    Parameters
    ----------
    approved_by_note:
        Non-empty human approval note.
    resume_after:
        Completed-model checkpoint cleared by the approval.
    context:
        Common operational context.
    created_at:
        Exact UTC event timestamp.

    Returns
    -------
    dict[str, Any]
        Strict logical operational event.
    """

    if not approved_by_note.strip() or resume_after < 1:
        raise ValueError("review signoff requires a note and positive resume_after checkpoint")
    event = _base_milestone_event(
        event_kind=OperationalEventKind.REVIEW_SIGNOFF,
        status=OperationalEventStatus.REVIEW_SIGNED_OFF,
        identity_payload={
            "approved_by_note": approved_by_note,
            "resume_after": resume_after,
        },
        context=context,
        created_at=created_at,
    )
    event.update({"approved_by_note": approved_by_note, "resume_after": resume_after})
    return event


def build_progress_notification_event(
    *,
    models_completed: int,
    milestone: int,
    funnel_snapshot: FunnelSnapshot,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct a non-blocking progress milestone event.

    Parameters
    ----------
    models_completed, milestone, funnel_snapshot:
        Current terminal count, announced threshold, and exact funnel.
    context:
        Common operational context.
    created_at:
        Exact UTC event timestamp.

    Returns
    -------
    dict[str, Any]
        Strict logical operational event.
    """

    _validate_models_completed(models_completed, funnel_snapshot)
    if milestone < 1 or models_completed < milestone:
        raise ValueError("progress milestone must be positive and already reached")
    event = _base_milestone_event(
        event_kind=OperationalEventKind.PROGRESS_NOTIFICATION,
        status=OperationalEventStatus.PROGRESS_NOTIFIED,
        identity_payload={"milestone": milestone},
        context=context,
        created_at=created_at,
    )
    event.update(
        {
            "models_completed": models_completed,
            "milestone": milestone,
            "funnel_snapshot": funnel_snapshot.to_dict(),
        }
    )
    return event


def record_checkpoint_review(
    ledger: JsonlLedger,
    *,
    models_completed: int,
    funnel_snapshot: FunnelSnapshot,
    report_path: str,
    context: OperationalContext,
    created_at: str,
) -> AppendResult:
    """Construct and append one checkpoint-review event.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    models_completed, funnel_snapshot, report_path, context, created_at:
        Fields accepted by ``build_checkpoint_review_event``.

    Returns
    -------
    AppendResult
        Idempotent append result.
    """

    event = build_checkpoint_review_event(
        models_completed=models_completed,
        funnel_snapshot=funnel_snapshot,
        report_path=report_path,
        context=context,
        created_at=created_at,
    )
    return record_operational_event(ledger, event)


def record_review_signoff(
    ledger: JsonlLedger,
    *,
    approved_by_note: str,
    resume_after: int,
    context: OperationalContext,
    created_at: str,
) -> AppendResult:
    """Construct and append one review-signoff event.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    approved_by_note, resume_after, context, created_at:
        Fields accepted by ``build_review_signoff_event``.

    Returns
    -------
    AppendResult
        Idempotent append result.
    """

    event = build_review_signoff_event(
        approved_by_note=approved_by_note,
        resume_after=resume_after,
        context=context,
        created_at=created_at,
    )
    return record_operational_event(ledger, event)


def record_progress_notification(
    ledger: JsonlLedger,
    *,
    models_completed: int,
    milestone: int,
    funnel_snapshot: FunnelSnapshot,
    context: OperationalContext,
    created_at: str,
) -> AppendResult:
    """Construct and append one progress-notification event.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    models_completed, milestone, funnel_snapshot, context, created_at:
        Fields accepted by ``build_progress_notification_event``.

    Returns
    -------
    AppendResult
        Idempotent append result.
    """

    event = build_progress_notification_event(
        models_completed=models_completed,
        milestone=milestone,
        funnel_snapshot=funnel_snapshot,
        context=context,
        created_at=created_at,
    )
    return record_operational_event(ledger, event)


def create_checkpoint_set(
    repo_root: Path,
    candidate_paths: Iterable[Path],
    *,
    ledger_paths: Iterable[Path],
    derived_view_checks: Iterable[ValidationCheck],
    public_artifacts: Iterable[LicensedArtifact],
    mirrors: MirrorStore,
    mirror_manifests: Iterable[ArtifactManifest] = (),
    expected_branch: str = CRAWLER_BRANCH,
    branch: Optional[str] = None,
    git_runner: Optional[GitRunner] = None,
) -> CheckpointSet:
    """Validate and stage an allowlisted crawler checkpoint without pushing.

    Parameters
    ----------
    repo_root:
        Git worktree root.
    candidate_paths:
        Complete set requested for staging, relative to ``repo_root``.
    ledger_paths:
        Canonical ledgers that must scan cleanly.
    derived_view_checks:
        Rebuild/digest validators; each raises on mismatch.
    public_artifacts:
        License-bound staged public artifacts.
    mirrors:
        Separated stores used for public-byte reverification.
    mirror_manifests:
        Complete retention manifests that must remain fetchable by hash.
    expected_branch:
        Exact crawler branch.
    branch:
        Optional already-resolved branch for deterministic orchestration/tests.
    git_runner:
        Injectable argv-only Git runner.

    Returns
    -------
    CheckpointSet
        Validated paths staged by one ``git add --`` command.

    Raises
    ------
    WrongCheckpointBranch
        If the current branch is not the crawler branch.
    NonAllowlistedPath
        If a candidate or already-staged path violates the allowlist.
    SecretBearingPath
        If a candidate path/bytes appear secret-bearing.
    CheckpointValidationError
        If ledgers, views, mirrors, licenses, or Git validation fails.
    """

    runner = git_runner or _run_git
    actual_branch = branch or _current_branch(repo_root, runner)
    if actual_branch != expected_branch or actual_branch in {"main", "master"}:
        raise WrongCheckpointBranch(
            f"checkpoint requires branch {expected_branch!r}, got {actual_branch!r}"
        )
    candidates = tuple(sorted({_normalize_path(path) for path in candidate_paths}, key=str))
    existing_staged = _staged_paths(repo_root, runner)
    for path in (*existing_staged, *candidates):
        _validate_allowlisted_path(repo_root, path, require_exists=path in candidates)
    try:
        for ledger_path in ledger_paths:
            scan_jsonl(ledger_path)
        for check in derived_view_checks:
            check()
        for manifest in mirror_manifests:
            mirrors.fetch(manifest)
        license_report = pre_public_merge_sweep(public_artifacts, mirrors)
    except PublicMergeRejected as exc:
        raise RestrictedPublicArtifact(str(exc)) from exc
    except Exception as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointValidationError(str(exc)) from exc
    if candidates:
        result = runner(["git", "add", "--", *(path.as_posix() for path in candidates)], repo_root)
        if result.returncode != 0:
            raise CheckpointValidationError(f"git add failed: {result.stderr.strip()}")
    return CheckpointSet(candidates, actual_branch, license_report)


def _base_milestone_event(
    *,
    event_kind: OperationalEventKind,
    status: OperationalEventStatus,
    identity_payload: Mapping[str, object],
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Build common fields for review/progress vocabulary.

    Parameters
    ----------
    event_kind, status, identity_payload, context, created_at:
        Type-specific identity and common event fields.

    Returns
    -------
    dict[str, Any]
        Strict common event envelope.
    """

    identity = stable_hash(
        {
            "event_kind": event_kind.value,
            "run_id": context.run_id,
            **identity_payload,
        }
    ).removeprefix("sha256:")
    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": f"{event_kind.value}-{identity[:24]}",
        "created_at": created_at,
        "event_kind": event_kind.value,
        "status": status.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": dict(context.queued_work_counts),
        "current_environment": context.current_environment,
        "run_id": context.run_id,
        "machine_id": context.machine_id,
        "details": {"blocking": event_kind is OperationalEventKind.CHECKPOINT_REVIEW},
    }


def _validate_models_completed(models_completed: int, snapshot: FunnelSnapshot) -> None:
    """Require the headline count to equal the exact funnel sum.

    Parameters
    ----------
    models_completed:
        Reported terminal count.
    snapshot:
        Exact terminal buckets.

    Raises
    ------
    ValueError
        If the count is negative or inconsistent.
    """

    total = snapshot.runs + snapshot.deferred + snapshot.skipped + snapshot.failed
    if models_completed < 0 or models_completed != total:
        raise ValueError(f"models_completed {models_completed} does not equal funnel total {total}")


def _normalize_path(path: Path) -> Path:
    """Normalize a candidate as a safe repository-relative path.

    Parameters
    ----------
    path:
        Candidate path.

    Returns
    -------
    pathlib.Path
        Normalized relative path.

    Raises
    ------
    NonAllowlistedPath
        If the path is absolute or traverses upward.
    """

    pure = PurePosixPath(path.as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise NonAllowlistedPath(f"checkpoint path must be repository-relative: {path}")
    return Path(pure.as_posix())


def _validate_allowlisted_path(repo_root: Path, path: Path, *, require_exists: bool) -> None:
    """Enforce checkpoint roots, file types, runtime exclusions, and secret scan.

    Parameters
    ----------
    repo_root, path:
        Worktree root and normalized candidate.
    require_exists:
        Whether bytes must exist for staging.

    Raises
    ------
    NonAllowlistedPath
        If the path does not match the checkpoint allowlist.
    SecretBearingPath
        If its name or bytes contain a credential marker.
    """

    pure = PurePosixPath(path.as_posix())
    under_root = any(pure == root or root in pure.parents for root in _ALLOWLIST_ROOTS)
    if (
        not under_root
        or pure.suffix not in _ALLOWLIST_SUFFIXES
        or ".crawl-local" in pure.parts
        or "__pycache__" in pure.parts
    ):
        raise NonAllowlistedPath(f"checkpoint path is not allowlisted: {path}")
    lowered_parts = {part.lower() for part in pure.parts}
    if lowered_parts & _SECRET_PATH_PARTS:
        raise SecretBearingPath(f"secret-bearing checkpoint path refused: {path}")
    absolute = repo_root / path
    if require_exists and not absolute.is_file():
        raise NonAllowlistedPath(f"checkpoint candidate is not a file: {path}")
    if absolute.is_file():
        lowered = absolute.read_bytes().lower()
        if any(marker in lowered for marker in _SECRET_BYTE_MARKERS):
            raise SecretBearingPath(f"secret-bearing checkpoint bytes refused: {path}")


def _current_branch(repo_root: Path, runner: GitRunner) -> str:
    """Read the exact current Git branch.

    Parameters
    ----------
    repo_root, runner:
        Worktree root and argv-only Git runner.

    Returns
    -------
    str
        Exact branch name.

    Raises
    ------
    CheckpointValidationError
        If Git cannot resolve the branch.
    """

    result = runner(["git", "branch", "--show-current"], repo_root)
    if result.returncode != 0 or not result.stdout.strip():
        raise CheckpointValidationError(f"cannot determine checkpoint branch: {result.stderr}")
    return result.stdout.strip()


def _staged_paths(repo_root: Path, runner: GitRunner) -> tuple[Path, ...]:
    """Return paths already staged before this checkpoint.

    Parameters
    ----------
    repo_root, runner:
        Worktree root and argv-only Git runner.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Normalized staged paths.

    Raises
    ------
    CheckpointValidationError
        If Git cannot inspect the index.
    """

    result = runner(["git", "diff", "--cached", "--name-only", "-z"], repo_root)
    if result.returncode != 0:
        raise CheckpointValidationError(f"cannot inspect staged paths: {result.stderr}")
    return tuple(_normalize_path(Path(value)) for value in result.stdout.split("\0") if value)


def _run_git(argv: Sequence[str], cwd: Path) -> GitCommandResult:
    """Run one non-interactive Git command and capture text output.

    Parameters
    ----------
    argv, cwd:
        Exact command and worktree root.

    Returns
    -------
    GitCommandResult
        Captured exit status and output.
    """

    completed = subprocess.run(list(argv), cwd=cwd, check=False, capture_output=True, text=True)
    return GitCommandResult(completed.returncode, completed.stdout, completed.stderr)
