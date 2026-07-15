"""Never-push validation and allowlisted crawler checkpoint staging."""

from __future__ import annotations

import json
import os
import re
import shutil
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
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
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
from menagerie.crawler.status import checkpoint_consistency_report
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


@dataclass(frozen=True)
class GeneratedMetadataDisposition:
    """Hash-bound safe generated/package metadata provenance."""

    staged_path: Path
    content_sha256: str
    byte_count: int
    disposition: str
    generator: str
    provenance: str


GitRunner = Callable[[Sequence[str], Path], GitCommandResult]
ValidationCheck = Callable[[], None]


_ALLOWLIST_ROOTS = (
    PurePosixPath("menagerie/crawler/records"),
    PurePosixPath("menagerie/crawler/source_manifests"),
    PurePosixPath("menagerie/crawler/mirrors"),
    PurePosixPath("menagerie/crawler/evidence"),
    PurePosixPath("menagerie/crawler/views"),
    PurePosixPath("menagerie/crawler/license_reports"),
    PurePosixPath("menagerie/crawler/envs"),
    PurePosixPath("menagerie/crawler/adapters"),
    PurePosixPath("menagerie/crawler/ports"),
    PurePosixPath("menagerie/crawler/patches"),
    PurePosixPath("menagerie/crawler/reconstruction"),
    PurePosixPath("menagerie/crawler/source_cas"),
)
_LICENSE_CANDIDATE_ROOTS = (
    PurePosixPath("menagerie/crawler/evidence"),
    PurePosixPath("menagerie/crawler/adapters"),
    PurePosixPath("menagerie/crawler/ports"),
    PurePosixPath("menagerie/crawler/patches"),
    PurePosixPath("menagerie/crawler/source_cas"),
)
_GENERATED_METADATA_ROOTS = (
    PurePosixPath("menagerie/crawler/records"),
    PurePosixPath("menagerie/crawler/source_manifests"),
    PurePosixPath("menagerie/crawler/reconstruction"),
    PurePosixPath("menagerie/crawler/envs"),
)
_ALLOWLIST_SUFFIXES = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".diff",
        ".h",
        ".hpp",
        ".json",
        ".jsonl",
        ".lock",
        ".md",
        ".patch",
        ".py",
        ".rs",
        ".sha256",
        ".source",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
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
_GENERATED_METADATA_MANIFEST = "generated-metadata-manifest.jsonl"
_SAFE_METADATA_DISPOSITIONS = frozenset(
    {
        "safe-composite-metadata-v1",
        "safe-generated-metadata-v1",
        "safe-package-metadata-v1",
    }
)
_SAFE_EXCERPT_DISPOSITIONS = frozenset(
    {"public-compatible", "public-domain", "short-excerpt-committed"}
)
_PRIVATE_URL_MARKERS = (
    b"git@",
    b"ssh://",
    b"https://private.",
    b"http://private.",
)
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
    license_inventory: Iterable[LicensedArtifact] = (),
    generated_metadata_inventory: Iterable[GeneratedMetadataDisposition] = (),
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
    license_inventory:
        Complete public/private license-decision inventory used to prove candidate
        coverage and exclude every restricted or unknown digest.
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
    requested_artifacts = tuple(public_artifacts)
    inventory = tuple(license_inventory) or requested_artifacts
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
        sweep_artifacts = _validate_candidate_license_coverage(
            repo_root,
            candidates,
            requested_artifacts,
            inventory,
            tuple(generated_metadata_inventory),
        )
        license_report = pre_public_merge_sweep(sweep_artifacts, mirrors)
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
    canonical_snapshot_root = canonical_records / "intake" / snapshot.snapshot_id
    _copy_canonical_tree(snapshot.root, canonical_snapshot_root)
    snapshot = load_intake_snapshot(canonical_snapshot_root)
    ledgers = default_ledger_paths(canonical_records)
    current = materialize_current(ledgers)
    consistency = checkpoint_consistency_report(
        (item.stable_id for item in snapshot.items), current
    )
    if not consistency.complete:
        raise CheckpointValidationError(
            "crawler checkpoint prefix is inconsistent: "
            f"extra={sorted(consistency.partition.extra_ids)}, "
            f"duplicates={sorted(consistency.partition.duplicate_ids)}, "
            f"issues={dict(consistency.incomplete_by_issue)}"
        )

    views_root = canonical_root / "views"
    view_check = _derived_view_check(snapshot.root / "items.jsonl", canonical_records, views_root)
    mirror_store = mirrors or MirrorStore(
        root / ".crawl-local" / "mirrors" / "public",
        root / ".crawl-local" / "mirrors" / "private",
        root / ".crawl-local" / "mirrors" / "local",
    )
    manifest_root = canonical_root / "mirrors"
    public_artifacts, license_inventory, mirror_manifests = _derive_mirror_facts(manifest_root)
    candidates = _derive_candidate_paths(root, canonical_root)
    generated_inventory = _publish_generated_metadata_inventory(root, canonical_root, candidates)
    candidates = _derive_candidate_paths(root, canonical_root)
    sweep_artifacts = _validate_candidate_license_coverage(
        root,
        candidates,
        public_artifacts,
        license_inventory,
        generated_inventory,
    )
    report = _validate_persisted_license_report(
        canonical_root / "license_reports", sweep_artifacts, mirror_store
    )
    if not any(
        path.parts[:3] == ("menagerie", "crawler", "license_reports") for path in candidates
    ):
        raise CheckpointValidationError("checkpoint requires a persisted passing license report")

    result = create_checkpoint_set(
        root,
        candidates,
        ledger_paths=(ledgers.models, ledgers.attempts, ledgers.gates),
        derived_view_checks=(view_check,),
        public_artifacts=sweep_artifacts,
        mirrors=mirror_store,
        mirror_manifests=mirror_manifests,
        license_inventory=license_inventory,
        generated_metadata_inventory=generated_inventory,
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
) -> tuple[
    tuple[LicensedArtifact, ...],
    tuple[LicensedArtifact, ...],
    tuple[ArtifactManifest, ...],
]:
    """Parse public license artifacts and all hash-verifiable mirror manifests.

    Parameters
    ----------
    manifest_root:
        Canonical committed mirror-manifest directory.

    Returns
    -------
    tuple[tuple[LicensedArtifact, ...], tuple[LicensedArtifact, ...],
    tuple[ArtifactManifest, ...]]
        Public sweep inputs, complete license inventory, and complete
        public/private mirror manifests.
    """

    public_artifacts: list[LicensedArtifact] = []
    license_inventory: list[LicensedArtifact] = []
    manifests: list[ArtifactManifest] = []
    for name in _CANONICAL_MANIFEST_NAMES:
        path = manifest_root / name
        if not path.is_file():
            raise CheckpointValidationError(f"missing canonical mirror manifest: {path}")
        for row in scan_jsonl(path, validate=False):
            artifact = _licensed_artifact(row)
            license_inventory.append(artifact)
            manifests.append(artifact.manifest)
            if name == "public-manifest.jsonl":
                public_artifacts.append(artifact)
    return tuple(public_artifacts), tuple(license_inventory), tuple(manifests)


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


def _validate_candidate_license_coverage(
    repo_root: Path,
    candidates: Sequence[Path],
    requested_artifacts: Sequence[LicensedArtifact],
    inventory: Sequence[LicensedArtifact],
    generated_metadata_inventory: Sequence[GeneratedMetadataDisposition] = (),
) -> tuple[LicensedArtifact, ...]:
    """Bind every candidate code/excerpt file to one hash-bound decision.

    Records, source manifests, evidence, and promoted authored-code roots may
    carry literal third-party code or excerpts. Every non-empty candidate in
    those roots therefore needs one decision in the complete public/private
    manifest inventory. Independently,
    every candidate file is hashed and compared with all restricted or unknown
    inventory digests so renaming a private artifact cannot make it public.

    Parameters
    ----------
    repo_root:
        Git worktree root containing candidate bytes.
    candidates:
        Complete normalized checkpoint set.
    requested_artifacts:
        Existing public mirror entries that must always enter the sweep.
    inventory:
        Complete public/private license-decision inventory.

    Returns
    -------
    tuple[LicensedArtifact, ...]
        Deduplicated sweep input containing every public entry and every
        candidate-bound decision.

    Raises
    ------
    RestrictedPublicArtifact
        If coverage, hashes, uniqueness, or restricted-digest exclusion fails.
    """

    by_path: dict[Path, LicensedArtifact] = {}
    duplicates: set[Path] = set()
    for artifact in inventory:
        staged_path = _normalize_path(artifact.staged_path)
        if artifact.decision.content_sha256 != artifact.manifest.content_sha256:
            raise RestrictedPublicArtifact(
                "license inventory decision hash does not match its mirror manifest: "
                f"{staged_path.as_posix()}"
            )
        if staged_path in by_path:
            duplicates.add(staged_path)
        else:
            by_path[staged_path] = artifact
    if duplicates:
        raise RestrictedPublicArtifact(
            "license inventory must contain exactly one decision per staged path: "
            f"duplicates={[path.as_posix() for path in sorted(duplicates, key=str)]}"
        )

    license_candidates = tuple(
        path
        for path in candidates
        if _is_license_candidate(path) and (repo_root / path).stat().st_size > 0
    )
    missing = tuple(path for path in license_candidates if path not in by_path)
    if missing:
        raise RestrictedPublicArtifact(
            "checkpoint candidate code/excerpt paths lack a license decision: "
            f"{[path.as_posix() for path in missing]}"
        )

    generated_by_path = {
        _normalize_path(disposition.staged_path): disposition
        for disposition in generated_metadata_inventory
    }
    if len(generated_by_path) != len(generated_metadata_inventory):
        raise RestrictedPublicArtifact(
            "generated metadata inventory must contain exactly one disposition per path"
        )
    generated_candidates = tuple(
        path
        for path in candidates
        if _is_generated_metadata(path) and (repo_root / path).stat().st_size > 0
    )
    missing_generated = tuple(
        path
        for path in generated_candidates
        if path not in generated_by_path and (_is_package_metadata(path) or path not in by_path)
    )
    if missing_generated:
        raise RestrictedPublicArtifact(
            "checkpoint generated/package metadata lacks provenance disposition: "
            f"{[path.as_posix() for path in missing_generated]}"
        )
    for path in generated_candidates:
        if path not in generated_by_path and path in by_path and not _is_package_metadata(path):
            continue
        disposition = generated_by_path[path]
        content = (repo_root / path).read_bytes()
        if disposition.disposition not in _SAFE_METADATA_DISPOSITIONS:
            raise RestrictedPublicArtifact(
                f"unsafe generated metadata disposition for {path.as_posix()}"
            )
        if disposition.content_sha256 != hash_bytes(content):
            raise RestrictedPublicArtifact(
                f"generated metadata hash is stale for {path.as_posix()}"
            )
        _validate_generated_metadata_bytes(path, content)

    for path in license_candidates:
        artifact = by_path[path]
        candidate_digest = hash_bytes((repo_root / path).read_bytes())
        if (
            candidate_digest != artifact.decision.content_sha256
            or candidate_digest != artifact.manifest.content_sha256
        ):
            raise RestrictedPublicArtifact(
                f"checkpoint candidate hash is not bound to its license decision: {path.as_posix()}"
            )

    restricted_digests = {
        artifact.decision.content_sha256
        for artifact in inventory
        if artifact.decision.redistribution_class
        in {RedistributionClass.RESTRICTED_PRIVATE, RedistributionClass.UNKNOWN}
    }
    restricted_paths = [
        path
        for path in candidates
        if hash_bytes((repo_root / path).read_bytes()) in restricted_digests
    ]
    if restricted_paths:
        raise RestrictedPublicArtifact(
            "restricted or unknown-license digest appears in checkpoint candidate set: "
            f"{[path.as_posix() for path in restricted_paths]}"
        )

    sweep_by_path: dict[Path, LicensedArtifact] = {}
    for artifact in (*requested_artifacts, *(by_path[path] for path in license_candidates)):
        path = _normalize_path(artifact.staged_path)
        previous = sweep_by_path.get(path)
        if previous is not None and previous != artifact:
            raise RestrictedPublicArtifact(
                f"conflicting license decisions enter the checkpoint sweep: {path.as_posix()}"
            )
        sweep_by_path[path] = artifact
    return tuple(sweep_by_path[path] for path in sorted(sweep_by_path, key=str))


def _is_license_candidate(path: Path) -> bool:
    """Return whether a checkpoint path may contain committed code/excerpts.

    Parameters
    ----------
    path:
        Normalized repository-relative checkpoint candidate.

    Returns
    -------
    bool
        True for records, source manifests, evidence, and accepted-code files.
    """

    pure = PurePosixPath(path.as_posix())
    return any(pure == root or root in pure.parents for root in _LICENSE_CANDIDATE_ROOTS)


def _is_generated_metadata(path: Path) -> bool:
    """Return whether a path requires the explicit safe metadata disposition.

    Parameters
    ----------
    path:
        Normalized repository-relative checkpoint path.

    Returns
    -------
    bool
        True for generated facts and environment package metadata.
    """

    pure = PurePosixPath(path.as_posix())
    return any(pure == root or root in pure.parents for root in _GENERATED_METADATA_ROOTS)


def _is_package_metadata(path: Path) -> bool:
    """Return whether a path is environment lock/export package metadata.

    Parameters
    ----------
    path:
        Normalized repository-relative candidate.

    Returns
    -------
    bool
        True only below the canonical environment root.
    """

    pure = PurePosixPath(path.as_posix())
    root = PurePosixPath("menagerie/crawler/envs")
    return pure == root or root in pure.parents


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


def _copy_canonical_tree(source: Path, destination: Path) -> None:
    """Durably copy immutable canonical facts without rewriting an existing tree.

    Parameters
    ----------
    source, destination:
        Verified source tree and deterministic repository destination.
    """

    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        return
    expected = {
        path.relative_to(source): hash_bytes(path.read_bytes())
        for path in source.rglob("*")
        if path.is_file()
    }
    if destination.is_dir():
        observed = {
            path.relative_to(destination): hash_bytes(path.read_bytes())
            for path in destination.rglob("*")
            if path.is_file()
        }
        if observed != expected:
            raise CheckpointValidationError("canonical immutable tree conflicts with source facts")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    os.replace(temporary, destination)


def _publish_generated_metadata_inventory(
    repo_root: Path, canonical_root: Path, candidates: Sequence[Path]
) -> tuple[GeneratedMetadataDisposition, ...]:
    """Publish the complete hash-bound generated/package metadata inventory.

    Parameters
    ----------
    repo_root, canonical_root, candidates:
        Repository roots and the complete pre-publication candidate set.

    Returns
    -------
    tuple[GeneratedMetadataDisposition, ...]
        Canonical safe dispositions in path order.
    """

    dispositions: list[GeneratedMetadataDisposition] = []
    for path in candidates:
        if not _is_generated_metadata(path):
            continue
        content = (repo_root / path).read_bytes()
        has_excerpts = _validate_generated_metadata_bytes(path, content)
        pure = PurePosixPath(path.as_posix())
        package_metadata = PurePosixPath("menagerie/crawler/envs") in pure.parents
        dispositions.append(
            GeneratedMetadataDisposition(
                staged_path=path,
                content_sha256=hash_bytes(content),
                byte_count=len(content),
                disposition=(
                    "safe-package-metadata-v1"
                    if package_metadata
                    else "safe-composite-metadata-v1"
                    if has_excerpts
                    else "safe-generated-metadata-v1"
                ),
                generator="menagerie.crawler.checkpoint.v1",
                provenance=(
                    "exact environment intent/lock/export package metadata"
                    if package_metadata
                    else "deterministic crawler canonical metadata"
                ),
            )
        )
    payloads = [
        {
            "staged_path": item.staged_path.as_posix(),
            "content_sha256": item.content_sha256,
            "byte_count": item.byte_count,
            "disposition": item.disposition,
            "generator": item.generator,
            "provenance": item.provenance,
        }
        for item in dispositions
    ]
    manifest_path = canonical_root / "mirrors" / _GENERATED_METADATA_MANIFEST
    data = b"".join(canonical_json_bytes(payload) + b"\n" for payload in payloads)
    _atomic_write_bytes(manifest_path, data)
    return tuple(dispositions)


def _validate_generated_metadata_bytes(path: Path, content: bytes) -> bool:
    """Reject secrets and unsafe embedded excerpts in composite metadata.

    Parameters
    ----------
    path, content:
        Candidate repository path and exact bytes.

    Returns
    -------
    bool
        Whether the candidate contains safely classified embedded excerpts.
    """

    lowered = content.lower()
    if any(marker in lowered for marker in _SECRET_BYTE_MARKERS):
        raise SecretBearingPath(f"generated metadata contains a credential marker: {path}")
    if any(marker in lowered for marker in _PRIVATE_URL_MARKERS) or re.search(
        rb"https?://[^/\s:@]+:[^/\s@]+@", lowered
    ):
        raise SecretBearingPath(f"generated metadata contains a private/credential URL: {path}")
    excerpts = _embedded_excerpts(path, content)
    unsafe = sorted(
        {
            str(excerpt.get("license_disposition") or "unknown")
            for excerpt in excerpts
            if excerpt.get("license_disposition") not in _SAFE_EXCERPT_DISPOSITIONS
        }
    )
    if unsafe:
        raise RestrictedPublicArtifact(
            "composite checkpoint metadata contains unknown/restricted embedded excerpts: "
            f"{path.as_posix()} dispositions={unsafe}"
        )
    return bool(excerpts)


def _embedded_excerpts(path: Path, content: bytes) -> tuple[Mapping[str, Any], ...]:
    """Extract every recursively embedded evidence excerpt from JSON metadata.

    Parameters
    ----------
    path, content:
        Candidate metadata path and exact bytes.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Recursively discovered excerpt objects across JSON and JSONL rows.

    Raises
    ------
    RestrictedPublicArtifact
        If a generated JSON candidate cannot be parsed conservatively.
    """

    if path.suffix not in {".json", ".jsonl"}:
        return ()
    try:
        values = (
            [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [json.loads(content.decode("utf-8"))]
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RestrictedPublicArtifact(
            f"composite checkpoint metadata is not valid JSON: {path.as_posix()}"
        ) from exc
    found: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        """Collect excerpt-shaped mappings from one nested JSON value."""

        if isinstance(value, Mapping):
            excerpts = value.get("excerpts")
            if isinstance(excerpts, list):
                found.extend(item for item in excerpts if isinstance(item, Mapping))
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for value in values:
        visit(value)
    return tuple(found)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace one generated canonical file and fsync its directory.

    Parameters
    ----------
    path, data:
        Destination and exact bytes.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    reconstruction_root = canonical_root / "reconstruction"
    for reconstruction in reconstruction_root.rglob("*.json"):
        if reconstruction.name.endswith(".commit.json"):
            continue
        try:
            payload = json.loads(reconstruction.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointValidationError(
                f"invalid canonical reconstruction: {reconstruction}"
            ) from exc
        if isinstance(payload, Mapping) and payload.get("schema_version") == (
            "menagerie.crawler.reconstruction.v2"
        ):
            marker = reconstruction.with_name(f"{str(payload.get('stable_id'))}.commit.json")
            if not marker.is_file():
                raise CheckpointValidationError(
                    f"checkpoint refuses uncommitted promotion: {reconstruction}"
                )
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
