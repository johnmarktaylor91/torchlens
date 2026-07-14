"""Never-push validation and allowlisted crawler checkpoint staging."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.licenses import (
    LicenseDecision,
    LicenseSweepReport,
    LicensedArtifact,
    PublicMergeRejected,
    RedistributionClass,
    pre_public_merge_sweep,
)
from menagerie.crawler.mirrors import ArtifactManifest, MirrorStore
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths, materialize_current
from menagerie.crawler.status import completeness_report
from menagerie.crawler.tools.rebuild_views import rebuild_views
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
_CANONICAL_MANIFEST_NAMES = ("public-manifest.jsonl", "private-manifest.jsonl")
_DERIVED_VIEW_PATHS = (
    Path("current-models/current.jsonl"),
    Path("release-models.jsonl"),
    Path("deferred-linux.jsonl"),
    Path("status-summary.json"),
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
    unexpected_staged = tuple(path for path in existing_staged if path not in candidates)
    if unexpected_staged:
        raise NonAllowlistedPath(
            "checkpoint index contains paths outside the derived candidate set: "
            f"{[path.as_posix() for path in unexpected_staged]}"
        )
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


def create_canonical_checkpoint(
    repo_root: Path,
    intake_root: Path,
    *,
    records_root: Optional[Path] = None,
    mirrors: Optional[MirrorStore] = None,
    expected_branch: str = CRAWLER_BRANCH,
    branch: Optional[str] = None,
    git_runner: Optional[GitRunner] = None,
) -> CheckpointSet:
    """Run the canonical transaction and normalize every validation failure.

    Parameters
    ----------
    repo_root:
        Git worktree root.
    intake_root:
        Canonical immutable intake snapshot directory.
    records_root:
        Optional canonical records root; defaults inside the crawler tree.
    mirrors:
        Optional separated byte stores. Defaults to the conventional runtime mirror roots.
    expected_branch, branch, git_runner:
        Branch policy and deterministic Git injection accepted by ``create_checkpoint_set``.

    Returns
    -------
    CheckpointSet
        Fully validated derived paths staged without commit or push.

    Raises
    ------
    CheckpointError
        If any branch, partition, completeness, view, mirror, license-report, or staging gate fails.
    """

    try:
        return _create_canonical_checkpoint(
            repo_root,
            intake_root,
            records_root=records_root,
            mirrors=mirrors,
            expected_branch=expected_branch,
            branch=branch,
            git_runner=git_runner,
        )
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(str(exc)) from exc


def _create_canonical_checkpoint(
    repo_root: Path,
    intake_root: Path,
    *,
    records_root: Optional[Path],
    mirrors: Optional[MirrorStore],
    expected_branch: str,
    branch: Optional[str],
    git_runner: Optional[GitRunner],
) -> CheckpointSet:
    """Implement the canonical fail-closed transaction.

    Parameters
    ----------
    repo_root, intake_root, records_root, mirrors, expected_branch, branch, git_runner:
        Fully normalized transaction inputs documented by ``create_canonical_checkpoint``.

    Returns
    -------
    CheckpointSet
        Fully validated derived paths staged without commit or push.
    """

    root = repo_root.resolve()
    runner = git_runner or _run_git
    actual_branch = branch or _current_branch(root, runner)
    if actual_branch != expected_branch or actual_branch in {"main", "master"}:
        raise WrongCheckpointBranch(
            f"checkpoint requires branch {expected_branch!r}, got {actual_branch!r}"
        )
    canonical_root = root / "menagerie" / "crawler"
    canonical_records = (records_root or canonical_root / "records").resolve()
    _require_under_root(canonical_records, canonical_root / "records", "records root")
    snapshot = load_intake_snapshot(intake_root.resolve())
    ledgers = default_ledger_paths(canonical_records)
    current = materialize_current(ledgers)
    completeness = completeness_report((item.stable_id for item in snapshot.items), current)
    if not completeness.complete:
        raise CheckpointValidationError(
            "crawler checkpoint is incomplete: "
            f"missing={sorted(completeness.partition.missing_ids)}, "
            f"extra={sorted(completeness.partition.extra_ids)}, "
            f"duplicates={sorted(completeness.partition.duplicate_ids)}, "
            f"issues={dict(completeness.incomplete_by_issue)}, "
            f"workflows={dict(completeness.workflow_counts)}"
        )

    views_root = canonical_root / "views"
    view_check = _derived_view_check(snapshot.root / "items.jsonl", canonical_records, views_root)
    mirror_store = mirrors or MirrorStore(
        root / ".crawl-local" / "mirrors" / "public",
        root / ".crawl-local" / "mirrors" / "private",
        root / ".crawl-local" / "mirrors" / "local",
    )
    manifest_root = canonical_root / "mirrors"
    public_artifacts, mirror_manifests = _derive_mirror_facts(manifest_root)
    report = _validate_persisted_license_report(
        canonical_root / "license_reports", public_artifacts, mirror_store
    )
    candidates = _derive_candidate_paths(root, canonical_root)
    if not any(
        path.parts[:3] == ("menagerie", "crawler", "license_reports") for path in candidates
    ):
        raise CheckpointValidationError("checkpoint requires a persisted passing license report")

    result = create_checkpoint_set(
        root,
        candidates,
        ledger_paths=(ledgers.models, ledgers.attempts, ledgers.gates),
        derived_view_checks=(view_check,),
        public_artifacts=public_artifacts,
        mirrors=mirror_store,
        mirror_manifests=mirror_manifests,
        expected_branch=expected_branch,
        branch=actual_branch,
        git_runner=runner,
    )
    if result.license_report != report:
        raise CheckpointValidationError(
            "persisted license report changed during checkpoint validation"
        )
    return result


def _require_under_root(path: Path, root: Path, label: str) -> None:
    """Require a resolved path to remain under its canonical repository root.

    Parameters
    ----------
    path, root:
        Resolved candidate and required ancestor.
    label:
        Diagnostic name for the candidate.
    """

    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise NonAllowlistedPath(f"checkpoint {label} is outside {resolved_root}: {path}")


def _derived_view_check(
    intake_path: Path, records_root: Path, committed_views_root: Path
) -> ValidationCheck:
    """Build an exact isolated view comparison closure.

    Parameters
    ----------
    intake_path, records_root, committed_views_root:
        Canonical inputs and committed derived-view destination.

    Returns
    -------
    ValidationCheck
        Validator that rebuilds and compares every canonical view byte-for-byte.
    """

    def check() -> None:
        """Rebuild views in isolation and reject missing, extra, or changed output."""

        with tempfile.TemporaryDirectory(prefix="torchlens-crawler-checkpoint-") as temporary:
            temporary_root = Path(temporary)
            rebuilt_root = temporary_root / "views"
            digests = rebuild_views(
                intake_path, records_root, rebuilt_root, temporary_root / "state.sqlite"
            )
            expected_digest_keys = {"current", "release", "deferred", "status"}
            if set(digests) != expected_digest_keys:
                raise CheckpointValidationError(
                    f"view rebuild returned incomplete digests: {sorted(digests)}"
                )
            committed_files = {
                path.relative_to(committed_views_root)
                for path in committed_views_root.rglob("*")
                if path.is_file() and path.suffix in _ALLOWLIST_SUFFIXES
            }
            if committed_files != set(_DERIVED_VIEW_PATHS):
                raise CheckpointValidationError(
                    "committed derived-view set does not match canonical rebuild: "
                    f"{sorted(path.as_posix() for path in committed_files)}"
                )
            for relative in _DERIVED_VIEW_PATHS:
                committed = committed_views_root / relative
                rebuilt = rebuilt_root / relative
                if committed.read_bytes() != rebuilt.read_bytes():
                    raise CheckpointValidationError(
                        f"derived view does not match canonical rebuild: {relative.as_posix()}"
                    )

    return check


def _derive_mirror_facts(
    manifest_root: Path,
) -> tuple[tuple[LicensedArtifact, ...], tuple[ArtifactManifest, ...]]:
    """Parse public license artifacts and all hash-verifiable mirror manifests.

    Parameters
    ----------
    manifest_root:
        Canonical committed mirror-manifest directory.

    Returns
    -------
    tuple[tuple[LicensedArtifact, ...], tuple[ArtifactManifest, ...]]
        Public sweep inputs and complete public/private mirror manifests.
    """

    public_artifacts: list[LicensedArtifact] = []
    manifests: list[ArtifactManifest] = []
    for name in _CANONICAL_MANIFEST_NAMES:
        path = manifest_root / name
        if not path.is_file():
            raise CheckpointValidationError(f"missing canonical mirror manifest: {path}")
        for row in scan_jsonl(path, validate=False):
            artifact = _licensed_artifact(row)
            manifests.append(artifact.manifest)
            if name == "public-manifest.jsonl":
                public_artifacts.append(artifact)
    return tuple(public_artifacts), tuple(manifests)


def _licensed_artifact(payload: Mapping[str, Any]) -> LicensedArtifact:
    """Parse one canonical hash- and license-bound mirror row.

    Parameters
    ----------
    payload:
        Row containing ``staged_path``, ``manifest``, and ``decision``.

    Returns
    -------
    LicensedArtifact
        Typed artifact suitable for mirror and license verification.
    """

    manifest_raw = payload.get("manifest")
    decision_raw = payload.get("decision")
    if not isinstance(manifest_raw, Mapping) or not isinstance(decision_raw, Mapping):
        raise CheckpointValidationError("mirror row requires manifest and decision objects")
    try:
        manifest = ArtifactManifest.from_dict(manifest_raw)
        decision = LicenseDecision(
            content_sha256=str(decision_raw["content_sha256"]),
            redistribution_class=RedistributionClass(str(decision_raw["redistribution_class"])),
            evidence_ids=tuple(str(item) for item in decision_raw.get("evidence_ids", [])),
            rationale=str(decision_raw["rationale"]),
        )
        staged_path = _normalize_path(Path(str(payload["staged_path"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"invalid licensed mirror artifact: {exc}") from exc
    return LicensedArtifact(staged_path, manifest, decision)


def _validate_persisted_license_report(
    report_root: Path,
    public_artifacts: Sequence[LicensedArtifact],
    mirrors: MirrorStore,
) -> LicenseSweepReport:
    """Require exactly one persisted report equal to a fresh passing sweep.

    Parameters
    ----------
    report_root:
        Canonical committed report directory.
    public_artifacts:
        Artifacts derived from the public mirror manifest.
    mirrors:
        Store used for fresh hash verification.

    Returns
    -------
    LicenseSweepReport
        Fresh passing report matching persisted bytes.
    """

    reports = sorted(path for path in report_root.glob("*.json") if path.is_file())
    if len(reports) != 1:
        raise CheckpointValidationError(
            f"checkpoint requires exactly one persisted license report, found {len(reports)}"
        )
    try:
        report = pre_public_merge_sweep(public_artifacts, mirrors)
    except PublicMergeRejected as exc:
        raise RestrictedPublicArtifact(str(exc)) from exc
    try:
        persisted = scan_jsonl(reports[0], validate=False)
    except Exception as exc:
        raise CheckpointValidationError(f"invalid persisted license report: {exc}") from exc
    if len(persisted) != 1 or persisted[0] != report.to_dict():
        raise CheckpointValidationError(
            "persisted license report does not match the freshly derived passing sweep"
        )
    return report


def _derive_candidate_paths(repo_root: Path, canonical_root: Path) -> tuple[Path, ...]:
    """Derive the complete checkpoint set solely from canonical public roots.

    Parameters
    ----------
    repo_root, canonical_root:
        Git worktree and canonical crawler roots.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted repository-relative files accepted by the checkpoint allowlist.
    """

    paths: set[Path] = set()
    for allowed_root in _ALLOWLIST_ROOTS:
        absolute_root = repo_root / Path(allowed_root.as_posix())
        for path in absolute_root.rglob("*"):
            if path.is_file() and path.suffix in _ALLOWLIST_SUFFIXES:
                paths.add(path.relative_to(repo_root))
    if not paths:
        raise CheckpointValidationError(
            f"no canonical checkpoint facts found under {canonical_root}"
        )
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


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
