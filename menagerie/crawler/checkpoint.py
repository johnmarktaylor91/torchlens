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
from typing import Any, Callable, Collection, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.artifact_transactions import (
    ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION,
    ArtifactCheckpointReference,
    ArtifactCheckpointProjection,
    ArtifactCheckpointError,
    ArtifactEventKind,
    artifact_reconstruction_paths,
    validate_artifact_checkpoint,
)
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.constants import (
    FAILURE_REASON_CODES,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.env_lifecycle import (
    EnvironmentExactnessError,
    EnvironmentProbeError,
    materialized_environment_generation,
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import EnvironmentSpecError, load_environment_registry
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    validate_size_variant_derivation,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import IntakeError, load_intake_snapshot
from menagerie.crawler.licenses import (
    AuthorizedArtifact,
    LicenseDecision,
    LicenseSweepReport,
    LicensedArtifact,
    PublicMergeRejected,
    RedistributionClass,
    pre_public_merge_sweep,
)
from menagerie.crawler.mirrors import ArtifactManifest, ArtifactOrigin, MirrorStore
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import (
    default_ledger_paths,
    project_dependency_current,
)
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


class ReconstructionValidationError(CheckpointValidationError):
    """Raised when a canonical reconstruction transaction is not reproducible."""


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
_REGISTERED_STANDALONE_ENVIRONMENT_CANDIDATES = frozenset(
    {
        Path("specs/round21-release.probes.json"),
        Path("specs/round21-release.virtual-packages.yml"),
        Path("specs/round21-release.yml"),
    }
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
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_DIAGNOSTIC_REDACTION_MARKER = "externally-controlled-text-v1"
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "
_EXTERNALLY_CONTROLLED_RECORD_FIELDS = frozenset(
    {
        "message",
        "mode_error",
        "observed_response",
        "receipt_error",
        "response_excerpt",
        "stderr_tail",
        "stdout_tail",
        "traceback",
    }
)
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
    canonical_ledger_path: Optional[Path] = None,
    policy_identity: Optional[Mapping[str, object]] = None,
) -> AppendResult:
    """Construct and append one checkpoint-review event.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    models_completed, funnel_snapshot, report_path, context, created_at:
        Fields accepted by ``build_checkpoint_review_event``.
    canonical_ledger_path:
        Optional canonical append-only operational ledger mirrored before runtime state.
    policy_identity:
        Stable intake/campaign identity for one-shot review policy.

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
    if policy_identity is not None:
        event["details"].update(dict(policy_identity))
    if canonical_ledger_path is not None:
        append_canonical_operational_event(canonical_ledger_path, event)
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
    canonical_path = _canonical_operational_path_for_runtime(ledger.path)
    canonical_events = scan_jsonl(canonical_path) if canonical_path is not None else []
    combined_by_id = {
        str(record.get("event_id")): record for record in (*canonical_events, *ledger.records)
    }
    combined = tuple(combined_by_id.values())
    signed_policy_keys = {
        str(record.get("details", {}).get("policy_key"))
        for record in combined
        if record.get("event_kind") == OperationalEventKind.REVIEW_SIGNOFF.value
        and isinstance(record.get("details"), Mapping)
        and isinstance(record.get("details", {}).get("policy_key"), str)
    }
    pending = [
        record
        for record in combined
        if record.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
        and (
            not isinstance(record.get("details"), Mapping)
            or record.get("details", {}).get("policy_key") not in signed_policy_keys
        )
    ]
    if pending:
        policy = pending[-1].get("details", {})
        if isinstance(policy, Mapping):
            for key in ("policy_key", "intake_snapshot_id", "review_threshold"):
                if key in policy:
                    event["details"][key] = policy[key]
    if canonical_path is not None and canonical_path != ledger.path:
        append_canonical_operational_event(canonical_path, event)
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


def canonical_records_root(models_path: Path) -> Path:
    """Return the canonical records root containing a model ledger.

    Parameters
    ----------
    models_path:
        Canonical model-ledger path.

    Returns
    -------
    pathlib.Path
        Canonical records root for sibling durable operational ledgers.
    """

    parent = models_path.resolve().parent
    return parent.parent if parent.name == "models" else parent


def canonical_operational_ledger_path(models_path: Path) -> Path:
    """Return the canonical append-only operational event ledger path.

    Parameters
    ----------
    models_path:
        Canonical model-ledger path.

    Returns
    -------
    pathlib.Path
        Canonical operational event ledger.
    """

    return canonical_records_root(models_path) / "operational" / "events.jsonl"


def canonical_requeue_grants_path(models_path: Path) -> Path:
    """Return the canonical append-only requeue grant ledger path.

    Parameters
    ----------
    models_path:
        Canonical model-ledger path.

    Returns
    -------
    pathlib.Path
        Canonical requeue grant ledger.
    """

    return canonical_records_root(models_path) / "operational" / "requeue-grants.jsonl"


def append_canonical_operational_event(path: Path, event: Mapping[str, Any]) -> AppendResult:
    """Append one logical event to the canonical operational ledger.

    Parameters
    ----------
    path:
        Canonical operational ledger path.
    event:
        Logical or persisted operational event.

    Returns
    -------
    AppendResult
        Idempotent canonical append result.
    """

    logical = {
        key: value for key, value in event.items() if key not in {"ledger_seq", "payload_sha256"}
    }
    with JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        return ledger.append(logical)


def append_canonical_requeue_grant(path: Path, grant: Mapping[str, Any]) -> bool:
    """Append one immutable grant to a canonical raw JSONL ledger once.

    Parameters
    ----------
    path:
        Canonical grant ledger path.
    grant:
        Already validated grant payload.

    Returns
    -------
    bool
        Whether new bytes were appended.

    Raises
    ------
    CheckpointValidationError
        If an existing grant identity has conflicting bytes.
    """

    grant_id = grant.get("grant_id")
    records = scan_jsonl(path, validate=False)
    for existing in records:
        if existing.get("grant_id") != grant_id:
            continue
        if existing != dict(grant):
            raise CheckpointValidationError(f"conflicting canonical requeue grant: {grant_id}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(dict(grant)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def build_canonical_requeue_grant(
    path: Path,
    *,
    stable_id: str,
    stage: str,
    reason: str,
    attempts: int,
    granted_by: str,
    target_intent: Optional[str] = None,
) -> JsonObject:
    """Build the sole canonical requeue-grant schema and identity.

    Parameters
    ----------
    path:
        Canonical grant ledger used to allocate the next append generation.
    stable_id, stage, reason, attempts, granted_by:
        Validated operator authorization facts.
    target_intent:
        Optional dependency-evidenced environment correction for the new work
        generation.

    Returns
    -------
    dict[str, Any]
        Exact grant payload accepted by driver rehydration.

    Raises
    ------
    ValueError
        If any operator-controlled grant fact is invalid.
    """

    if not stable_id.strip():
        raise ValueError("stable_id must be non-empty")
    records_root = path.parent.parent if path.parent.name == "operational" else None
    snapshot_roots = (
        sorted((records_root / "intake").iterdir())
        if records_root is not None and (records_root / "intake").is_dir()
        else []
    )
    intake_ids: set[str] = set()
    for snapshot_root in snapshot_roots:
        if snapshot_root.is_dir():
            try:
                intake_ids.update(
                    item.stable_id for item in load_intake_snapshot(snapshot_root).items
                )
            except (OSError, KeyError, TypeError, ValueError, IntakeError) as exc:
                raise ValueError(f"canonical intake snapshot is invalid: {snapshot_root}") from exc
    if intake_ids and stable_id not in intake_ids:
        raise ValueError(f"stable_id is absent from canonical intake snapshots: {stable_id}")
    if stage not in FAILURE_REASON_CODES:
        raise ValueError(f"unknown failure stage: {stage}")
    if not reason.strip() or not granted_by.strip():
        raise ValueError("reason and granted_by must be non-empty")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("grant attempts must be a positive integer")
    if target_intent is not None and not target_intent.strip():
        raise ValueError("target_intent must be non-empty when provided")
    generation = len(scan_jsonl(path, validate=False)) + 1
    identity = {
        "generation": generation,
        "stable_id": stable_id,
        "stage": stage,
        "reason": reason,
        "attempts": attempts,
        "granted_by": granted_by,
    }
    if target_intent is not None:
        identity["target_intent"] = target_intent
    grant: JsonObject = {
        "grant_id": stable_hash(identity),
        "stable_id": stable_id,
        "stage": stage,
        "reason": reason,
        "attempts": attempts,
        "granted_by": granted_by,
        "new_work_generation": generation,
    }
    if target_intent is not None:
        grant["target_intent"] = target_intent
    return grant


def _canonical_operational_path_for_runtime(runtime_path: Path) -> Optional[Path]:
    """Infer the canonical operational ledger paired with a disposable runtime ledger.

    Parameters
    ----------
    runtime_path:
        Runtime ``.crawl-local`` operational event path.

    Returns
    -------
    pathlib.Path | None
        Conventional canonical path, or ``None`` for an unrelated test/custom ledger.
    """

    resolved = runtime_path.resolve()
    runtime_root = resolved.parent.parent
    sibling_records = runtime_root.parent / "records"
    if sibling_records.is_dir():
        return sibling_records / "operational" / "events.jsonl"
    if ".crawl-local" in resolved.parts:
        index = resolved.parts.index(".crawl-local")
        if index > 0:
            repo_root = Path(*resolved.parts[:index])
            records_root = repo_root / "menagerie" / "crawler" / "records"
            if records_root.is_dir():
                return records_root / "operational" / "events.jsonl"
    return None


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
    restricted_gated_digests: Iterable[str] = (),
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
    restricted_gated_digests:
        Restricted/unknown exact digests derived independently from current gated facts.
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
            tuple(restricted_gated_digests),
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


def _head_jsonl_paths(repo_root: Path, canonical_root: Path, runner: GitRunner) -> tuple[Path, ...]:
    """Return canonical JSONL paths tracked by ``HEAD``.

    Parameters
    ----------
    repo_root, canonical_root, runner:
        Worktree roots and the non-interactive Git boundary.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted repository-relative tracked JSONL paths.
    """

    relative_root = canonical_root.relative_to(repo_root).as_posix()
    result = runner(["git", "ls-tree", "-r", "--name-only", "HEAD", "--", relative_root], repo_root)
    if result.returncode != 0:
        raise CheckpointValidationError(
            f"cannot enumerate canonical JSONL history from HEAD: {result.stderr.strip()}"
        )
    return tuple(
        sorted(
            {
                _normalize_path(Path(value))
                for value in result.stdout.splitlines()
                if value.endswith(".jsonl")
            },
            key=str,
        )
    )


def _head_path_bytes(repo_root: Path, path: Path, runner: GitRunner) -> Optional[bytes]:
    """Read exact UTF-8 JSONL bytes for one path from ``HEAD``.

    Parameters
    ----------
    repo_root, path, runner:
        Worktree root, normalized repository path, and Git boundary.

    Returns
    -------
    bytes | None
        Committed bytes, or ``None`` when the path is not tracked by ``HEAD``.
    """

    object_name = f"HEAD:{path.as_posix()}"
    exists = runner(["git", "cat-file", "-e", object_name], repo_root)
    if exists.returncode != 0:
        return None
    result = runner(["git", "show", object_name], repo_root)
    if result.returncode != 0:
        raise CheckpointValidationError(
            f"cannot read committed canonical JSONL {path.as_posix()}: {result.stderr.strip()}"
        )
    return result.stdout.encode("utf-8")


def _tail_recovery_is_evidenced(path: Path, committed: bytes, working: bytes) -> bool:
    """Return whether a committed torn tail was narrowly and durably recovered.

    Parameters
    ----------
    path, committed, working:
        Absolute ledger path and committed/working bytes.

    Returns
    -------
    bool
        True only for exact prefix truncation backed by matching recovery evidence.
    """

    if not committed or committed.endswith(b"\n"):
        return False
    good_end = committed.rfind(b"\n") + 1
    prefix = committed[:good_end]
    tail = committed[good_end:]
    if not working.startswith(prefix):
        return False
    evidence_path = path.with_suffix(f"{path.suffix}.recovery.jsonl")
    try:
        evidence = scan_jsonl(evidence_path, validate=False)
    except Exception:
        return False
    return any(
        row.get("byte_offset") == good_end
        and row.get("byte_count") == len(tail)
        and row.get("tail_sha256") == hash_bytes(tail)
        for row in evidence
    )


def _validate_canonical_jsonl_append_only(
    repo_root: Path,
    canonical_root: Path,
    candidates: Sequence[Path],
    runner: GitRunner,
) -> None:
    """Prove every canonical ledger is an append-only extension of ``HEAD``.

    Parameters
    ----------
    repo_root, canonical_root, candidates, runner:
        Worktree roots, complete candidate set, and Git boundary.

    Raises
    ------
    CheckpointValidationError
        If committed history was deleted/rewritten or new lines are incomplete/invalid.
    """

    records_root = canonical_root.relative_to(repo_root) / "records"
    mirrors_root = canonical_root.relative_to(repo_root) / "mirrors"

    def is_append_only_path(path: Path) -> bool:
        """Select canonical facts whose byte history is immutable."""

        if path.suffix != ".jsonl":
            return False
        if path.name == _GENERATED_METADATA_MANIFEST:
            return False
        return path.is_relative_to(records_root) or (
            path.is_relative_to(mirrors_root) and path.name in _CANONICAL_MANIFEST_NAMES
        )

    working_paths = {path for path in candidates if is_append_only_path(path)}
    tracked_paths = {
        path
        for path in _head_jsonl_paths(repo_root, canonical_root, runner)
        if is_append_only_path(path)
    }
    for relative in sorted(working_paths | tracked_paths, key=str):
        absolute = repo_root / relative
        if not absolute.is_file():
            raise CheckpointValidationError(
                f"tracked canonical JSONL was deleted: {relative.as_posix()}"
            )
        working = absolute.read_bytes()
        committed = _head_path_bytes(repo_root, relative, runner)
        if committed is not None and not working.startswith(committed):
            if not _tail_recovery_is_evidenced(absolute, committed, working):
                raise CheckpointValidationError(
                    "canonical JSONL is not an append-only extension of HEAD: "
                    f"{relative.as_posix()}"
                )
        try:
            scan_jsonl(absolute)
        except Exception as exc:
            raise CheckpointValidationError(
                f"canonical JSONL has incomplete or invalid appended lines: {relative.as_posix()}"
            ) from exc


def _validate_artifact_reconstruction_append_only(
    repo_root: Path,
    artifact_ledger_paths: Sequence[Path],
    candidates: Sequence[Path],
    runner: GitRunner,
) -> None:
    """Prove every ledger-named reconstruction is immutable against ``HEAD``.

    Parameters
    ----------
    repo_root, artifact_ledger_paths, candidates, runner:
        Worktree root, complete artifact shards, checkpoint candidate set, and
        non-interactive Git boundary.

    Raises
    ------
    CheckpointValidationError
        If a named reconstruction is absent from the checkpoint set or rewrites
        bytes already committed at the same immutable path.
    """

    candidate_set = set(candidates)
    for absolute in artifact_reconstruction_paths(artifact_ledger_paths, repo_root):
        try:
            relative = _normalize_path(absolute.resolve().relative_to(repo_root.resolve()))
        except ValueError as exc:
            raise CheckpointValidationError(
                f"artifact reconstruction escapes repository: {absolute}"
            ) from exc
        if relative not in candidate_set or not absolute.is_file():
            raise CheckpointValidationError(
                f"ledger-named reconstruction is absent from checkpoint: {relative.as_posix()}"
            )
        committed = _head_path_bytes(repo_root, relative, runner)
        if committed is not None and absolute.read_bytes() != committed:
            raise CheckpointValidationError(
                f"ledger-named reconstruction is not immutable against HEAD: {relative.as_posix()}"
            )


def create_canonical_checkpoint(
    repo_root: Path,
    intake_root: Path,
    *,
    authority_context: AuthorityContext,
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
    authority_context:
        Mandatory active trust roots shared by projection and artifact validation.
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
            authority_context=authority_context,
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
    authority_context: AuthorityContext,
    expected_branch: str,
    branch: Optional[str],
    git_runner: Optional[GitRunner],
) -> CheckpointSet:
    """Implement the canonical fail-closed transaction.

    Parameters
    ----------
    repo_root, intake_root, records_root, mirrors, authority_context,
    expected_branch, branch, git_runner:
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
    projection = project_dependency_current(
        ledgers,
        context=authority_context,
    )
    if projection.stale_reasons:
        raise CheckpointValidationError(
            "crawler checkpoint semantic replay rejected current rows: "
            f"{dict(sorted(projection.stale_reasons.items()))}"
        )
    current = projection.current_records
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
    view_check = _derived_view_check(
        snapshot.root / "items.jsonl",
        canonical_records,
        views_root,
        authority_context,
    )
    mirror_store = mirrors or MirrorStore(
        root / ".crawl-local" / "mirrors" / "public",
        root / ".crawl-local" / "mirrors" / "private",
        root / ".crawl-local" / "mirrors" / "local",
    )
    artifact_ledgers = tuple(sorted((canonical_records / "artifacts").glob("*.jsonl")))
    has_artifact_events = any(path.stat().st_size > 0 for path in artifact_ledgers)
    if (
        authority_context.active_intake_snapshot_id != snapshot.snapshot_id
        or authority_context.active_intake_snapshot_sha256 != snapshot.snapshot_sha256
    ):
        raise CheckpointValidationError(
            "checkpoint AuthorityContext differs from the active intake snapshot"
        )
    artifact_projection = ArtifactCheckpointProjection(transactions={}, objects=(), claims=())
    if has_artifact_events:
        try:
            artifact_projection = validate_artifact_checkpoint(
                artifact_ledgers,
                context=authority_context,
                mirrors=mirror_store,
                canonical_root=canonical_root,
                repository_root=root,
            )
        except ArtifactCheckpointError as exc:
            raise CheckpointValidationError(str(exc)) from exc
    manifest_root = canonical_root / "mirrors"
    public_artifacts, license_inventory, mirror_manifests = _derive_mirror_facts(manifest_root)
    promoted_model_ids = frozenset(
        str(event["stable_id"])
        for ledger_path in artifact_ledgers
        for event in scan_jsonl(ledger_path)
        if event.get("event_kind") == ArtifactEventKind.PUBLISHED.value
    )
    authorized_artifacts = _validate_gated_license_decisions(
        license_inventory,
        current,
        artifact_projection,
        promoted_model_ids=promoted_model_ids,
    )
    restricted_gated_digests = tuple(
        artifact.content_sha256
        for artifact in authorized_artifacts
        if artifact.decision.redistribution_class
        in {RedistributionClass.RESTRICTED_PRIVATE, RedistributionClass.UNKNOWN}
    )
    candidates = _derive_candidate_paths(root, canonical_root, authority_context)
    _validate_canonical_jsonl_append_only(root, canonical_root, candidates, runner)
    if has_artifact_events:
        _validate_artifact_reconstruction_append_only(root, artifact_ledgers, candidates, runner)
    validated_environment_candidates = _validate_environment_candidates(
        canonical_root, candidates, current
    )
    generated_inventory = _publish_generated_metadata_inventory(
        root, canonical_root, candidates, validated_environment_candidates
    )
    candidates = _derive_candidate_paths(root, canonical_root, authority_context)
    sweep_artifacts = _validate_candidate_license_coverage(
        root,
        candidates,
        public_artifacts,
        license_inventory,
        generated_inventory,
        restricted_gated_digests,
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
        ledger_paths=(
            ledgers.models,
            ledgers.attempts,
            ledgers.gates,
            *artifact_ledgers,
            canonical_operational_ledger_path(ledgers.models),
            canonical_requeue_grants_path(ledgers.models),
        ),
        derived_view_checks=(view_check,),
        public_artifacts=sweep_artifacts,
        mirrors=mirror_store,
        mirror_manifests=mirror_manifests,
        license_inventory=license_inventory,
        generated_metadata_inventory=generated_inventory,
        restricted_gated_digests=restricted_gated_digests,
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
    intake_path: Path,
    records_root: Path,
    committed_views_root: Path,
    authority_context: AuthorityContext,
) -> ValidationCheck:
    """Build an exact isolated view comparison closure.

    Parameters
    ----------
    intake_path, records_root, committed_views_root, authority_context:
        Canonical inputs, committed view destination, and active authority.

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
                intake_path,
                records_root,
                rebuilt_root,
                temporary_root / "state.sqlite",
                context=authority_context,
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
    seen_paths: dict[Path, str] = {}
    for name in _CANONICAL_MANIFEST_NAMES:
        path = manifest_root / name
        if not path.is_file():
            raise CheckpointValidationError(f"missing canonical mirror manifest: {path}")
        for row in scan_jsonl(path, validate=False):
            artifact = _licensed_artifact(row)
            digest = artifact.decision.content_sha256
            previous = seen_paths.get(artifact.staged_path)
            if previous is not None:
                raise CheckpointValidationError(
                    "canonical mirror manifests contain a duplicate staged path"
                )
            seen_paths[artifact.staged_path] = digest
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
        decision = LicenseDecision.from_dict(decision_raw)
        staged_path = _normalize_path(Path(str(payload["staged_path"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"invalid licensed mirror artifact: {exc}") from exc
    artifact_role = payload.get("artifact_role")
    source_id = payload.get("source_id")
    fetch_recipe = payload.get("fetch_recipe")
    if any(
        not isinstance(value, str) or not value
        for value in (artifact_role, source_id, fetch_recipe)
    ):
        raise RestrictedPublicArtifact(
            "licensed mirror artifact requires role, source_id, and fetch_recipe"
        )
    return LicensedArtifact(
        staged_path,
        manifest,
        decision,
        str(artifact_role),
        str(source_id),
        str(fetch_recipe),
    )


def _validate_gated_license_decisions(
    artifacts: Sequence[LicensedArtifact],
    current_models: Mapping[str, Mapping[str, Any]],
    projection: ArtifactCheckpointProjection,
    *,
    promoted_model_ids: Collection[str],
) -> tuple[AuthorizedArtifact, ...]:
    """Validate the complete mirror inventory against a closed gated artifact map.

    Parameters
    ----------
    artifacts:
        Complete canonical public/private mirror inventory.
    current_models:
        Dependency-current records that passed reducer semantic replay.
    promoted_model_ids:
        Models with a durable artifact-ledger ``published`` event. Only their complete
        authorized artifact sets must already appear in the mirror inventory.

    Raises
    ------
    CheckpointValidationError
        If any promoted public/private row is missing, any row is extra or
        ambiguous, or any row differs in its exact path, role, digest, origin,
        evidence decision, or fetch recipe.
    """

    objects = {str(obj.object_id): obj for obj in projection.objects}
    claims = {str(claim.claim_id): claim for claim in projection.claims}
    selected_claim_ids: set[str] = set()
    promoted_paths: set[Path] = set()
    for stable_id, model in current_models.items():
        authority = model.get("artifact_authority")
        if not isinstance(authority, Mapping) or authority.get("state") == "not-applicable":
            continue
        if authority.get("state") == ArtifactEventKind.STAGED_PRIVATE.value:
            continue
        transaction_id = authority.get("transaction_id")
        claim_ids = authority.get("claim_ids")
        if not isinstance(transaction_id, str) or not isinstance(claim_ids, list):
            raise RestrictedPublicArtifact(
                f"dependency-current model has malformed artifact authority: {stable_id}"
            )
        checkpoint_reference = ArtifactCheckpointReference.from_authority_payload(authority)
        transaction_id = checkpoint_reference.transaction_id
        claim_ids = checkpoint_reference.claim_ids
        transactions = [
            transaction
            for (candidate_stable, _work_id, candidate_transaction), transaction in (
                projection.transactions.items()
            )
            if candidate_stable == stable_id and str(candidate_transaction) == transaction_id
        ]
        if len(transactions) != 1:
            raise RestrictedPublicArtifact(
                f"dependency-current artifact transaction is missing or ambiguous: {stable_id}"
            )
        transaction = transactions[0]
        expected_claim_ids = {str(claim.claim_id) for claim in transaction.claims}
        if set(str(value) for value in claim_ids) != expected_claim_ids:
            raise RestrictedPublicArtifact(
                f"dependency-current artifact claim set differs from transaction: {stable_id}"
            )
        selected_claim_ids.update(expected_claim_ids)
        if stable_id in promoted_model_ids:
            promoted_paths.update(
                _normalize_path(Path(claim.logical_path)) for claim in transaction.claims
            )

    authorized: list[AuthorizedArtifact] = []
    objects_by_path: dict[Path, str] = {}
    for claim_id in sorted(selected_claim_ids):
        claim = claims.get(claim_id)
        if claim is None:
            raise RestrictedPublicArtifact(
                f"dependency-current artifact claim is absent from projection: {claim_id}"
            )
        obj = objects.get(str(claim.object_id))
        if obj is None:
            raise RestrictedPublicArtifact(f"artifact claim has no intrinsic object: {claim_id}")
        path = _normalize_path(Path(claim.logical_path))
        prior_object = objects_by_path.setdefault(path, str(claim.object_id))
        if prior_object != str(claim.object_id):
            raise RestrictedPublicArtifact(
                f"gated claims conflict at canonical artifact path: {path.as_posix()}"
            )
        redistribution = RedistributionClass(claim.license_disposition)
        decision = LicenseDecision(
            content_sha256=obj.content_sha256,
            redistribution_class=redistribution,
            evidence_ids=claim.evidence_ids,
            rationale="normalized artifact claim accepted by canonical transaction",
        )
        authorized.append(
            AuthorizedArtifact(
                staged_path=path,
                artifact_role=claim.logical_role,
                content_sha256=obj.content_sha256,
                origin=ArtifactOrigin(claim.origin, claim.revision),
                decision=decision,
                source_id=claim.source_id,
                fetch_recipe=claim.fetch_recipe_sha256,
            )
        )

    authorized_by_path: dict[Path, list[AuthorizedArtifact]] = {}
    for value in authorized:
        authorized_by_path.setdefault(_normalize_path(value.staged_path), []).append(value)

    inventory_by_path = {_normalize_path(artifact.staged_path): artifact for artifact in artifacts}
    for path, artifact in inventory_by_path.items():
        expected_values = authorized_by_path.get(path, [])
        if not expected_values or all(
            artifact.manifest.content_sha256 != expected.content_sha256
            or artifact.decision.redistribution_class is not expected.decision.redistribution_class
            for expected in expected_values
        ):
            raise RestrictedPublicArtifact(
                "mirror row is outside the closed dependency-current authorized-artifact map: "
                f"{path.as_posix()}"
            )
        expected_mirror = (
            "public"
            if artifact.decision.redistribution_class is RedistributionClass.PUBLIC_OK
            else "private"
        )
        if artifact.manifest.mirror_class.value != expected_mirror:
            raise RestrictedPublicArtifact(
                "mirror row is stored on the wrong license boundary: "
                f"{path.as_posix()} expected={expected_mirror}"
            )

    missing = tuple(path for path in promoted_paths if path not in inventory_by_path)
    if missing:
        raise RestrictedPublicArtifact(
            "canonical mirror manifests are incomplete for promoted dependency-current "
            "artifacts: "
            f"{[path.as_posix() for path in sorted(missing, key=str)]}"
        )
    return tuple(authorized)


def _validate_candidate_license_coverage(
    repo_root: Path,
    candidates: Sequence[Path],
    requested_artifacts: Sequence[LicensedArtifact],
    inventory: Sequence[LicensedArtifact],
    generated_metadata_inventory: Sequence[GeneratedMetadataDisposition] = (),
    restricted_gated_digests: Sequence[str] = (),
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
    restricted_gated_digests:
        Restricted/unknown digests recomputed from dependency-current facts rather
        than inferred from the possibly incomplete manifest inventory.

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
    restricted_digests.update(restricted_gated_digests)
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


def _validate_lock_package_hashes(path: Path, content: bytes) -> None:
    """Require every concrete target-lock package entry to carry a SHA-256 hash.

    Parameters
    ----------
    path, content:
        Exact target-lock path and bytes.

    Raises
    ------
    CheckpointValidationError
        If the lock is not UTF-8, has no packages, or has an unhashed entry.
    """

    lowered = content.lower()
    if any(marker in lowered for marker in _PRIVATE_URL_MARKERS) or re.search(
        rb"https?://[^/\s:@]+:[^/\s@]+@", lowered
    ):
        raise SecretBearingPath(f"generated metadata contains a private/credential URL: {path}")
    try:
        parse_exact_lock(content)
    except EnvironmentExactnessError as exc:
        raise CheckpointValidationError(f"invalid exact environment lock {path}: {exc}") from exc


def _validate_environment_candidates(
    canonical_root: Path,
    candidates: Sequence[Path],
    current_models: Mapping[str, Mapping[str, Any]],
) -> frozenset[Path]:
    """Strictly validate the registry-derived environment candidate class.

    Parameters
    ----------
    canonical_root, candidates, current_models:
        Canonical crawler root and complete repository-relative checkpoint set.

    Returns
    -------
    frozenset[pathlib.Path]
        Environment candidates proven to be registry/lock metadata.

    Raises
    ------
    CheckpointValidationError
        If registry intent membership, lock hashes, export bytes, or probe contracts fail.
    """

    repo_root = canonical_root.parents[1]
    env_root = canonical_root / "envs"
    env_relative = env_root.relative_to(repo_root)
    environment_candidates = frozenset(
        path for path in candidates if path == env_relative or path.is_relative_to(env_relative)
    )
    attempt_rows: list[JsonObject] = []
    for attempt_path in sorted((canonical_root / "records" / "attempts").glob("*.jsonl")):
        attempt_rows.extend(scan_jsonl(attempt_path))
    attempts_by_id = {str(row.get("attempt_id")): row for row in attempt_rows}
    current_generation_attempt_ids: set[str] = set()
    if not environment_candidates:
        if any(isinstance(row.get("environment"), Mapping) for row in attempt_rows):
            raise CheckpointValidationError(
                "observed environments require committed exact registry artifacts"
            )
        return frozenset()
    try:
        baseline = load_environment_registry(env_root)
    except (OSError, ValueError, EnvironmentSpecError) as exc:
        raise CheckpointValidationError(f"environment registry is invalid: {exc}") from exc
    registered = frozenset(baseline.intents)
    for path in environment_candidates:
        relative = path.relative_to(env_relative)
        if (
            len(relative.parts) > 1
            and relative.parts[0] not in registered
            and relative.parts[0] != "locks"
            and relative not in _REGISTERED_STANDALONE_ENVIRONMENT_CANDIDATES
        ):
            raise CheckpointValidationError(
                f"environment candidate is outside the registry: {path.as_posix()}"
            )

    target_names: set[str] = set()
    for intent_name in registered:
        lock_root = env_root / intent_name / "locks"
        if not lock_root.is_dir():
            continue
        for path in lock_root.iterdir():
            if path.name.endswith(".resolved.json"):
                target_names.add(path.name.removesuffix(".resolved.json"))
            elif path.name.endswith(".resolved.sha256"):
                target_names.add(path.name.removesuffix(".resolved.sha256"))
            elif path.suffix == ".lock":
                target_names.add(path.stem)
            elif path.name.endswith(".probes.json"):
                target_names.add(path.name.removesuffix(".probes.json"))

    for target in sorted(target_names):
        try:
            registry = load_environment_registry(env_root, target=target)
        except (OSError, ValueError, EnvironmentSpecError) as exc:
            raise CheckpointValidationError(
                f"environment registry cannot load target {target}: {exc}"
            ) from exc
        for intent in registry.intents.values():
            lock = intent.lock
            members_present = (
                lock.lock_path.is_file(),
                lock.export_path.is_file(),
                lock.export_hash_path.is_file(),
                lock.lock_path.with_name(f"{target}.probes.json").is_file(),
            )
            if not any(members_present):
                continue
            if not all(members_present) or lock.status != "locked":
                raise CheckpointValidationError(
                    f"environment target artifacts are incomplete or stale: {intent.name}/{target}"
                )
            assert lock.lock_bytes is not None
            assert lock.export_bytes is not None
            _validate_lock_package_hashes(lock.lock_path, lock.lock_bytes)
            if (
                not isinstance(lock.declared_export_hash, str)
                or _HASH_PATTERN.fullmatch(lock.declared_export_hash) is None
                or hash_bytes(lock.export_bytes) != lock.declared_export_hash
            ):
                raise CheckpointValidationError(
                    f"environment resolved-export digest is invalid: {intent.name}/{target}"
                )
            try:
                package_bytes = parse_resolved_export(lock.export_bytes)
                probe_results = parse_probe_receipt_bytes(
                    intent.probes,
                    lock.lock_path.with_name(f"{target}.probes.json").read_bytes(),
                )
            except (OSError, EnvironmentExactnessError, EnvironmentProbeError) as exc:
                raise CheckpointValidationError(
                    f"environment exact export/probe facts are invalid: {intent.name}/{target}"
                ) from exc
            lock_sha256 = hash_bytes(lock.lock_bytes)
            attested = False
            for row in attempt_rows:
                environment = row.get("environment")
                identities = row.get("identities")
                if (
                    not isinstance(environment, Mapping)
                    or not isinstance(identities, Mapping)
                    or environment.get("family") != intent.name
                    or environment.get("target") != target
                    or not isinstance(environment.get("python"), str)
                    or not isinstance(environment.get("compiler_identity"), str)
                    or not isinstance(environment.get("sdk_identity"), str)
                ):
                    continue
                try:
                    generation = materialized_environment_generation(
                        intent,
                        lock_bytes=lock.lock_bytes,
                        export_bytes=lock.export_bytes,
                        package_bytes=package_bytes,
                        python_version=str(environment["python"]),
                        compiler_identity=str(environment["compiler_identity"]),
                        sdk_identity=str(environment["sdk_identity"]),
                        probe_results=probe_results,
                    )
                except (EnvironmentExactnessError, EnvironmentProbeError):
                    continue
                if (
                    environment.get("lock_sha256") == lock_sha256
                    and environment.get("resolved_export_sha256") == lock.declared_export_hash
                    and environment.get("packages_manifest_sha256") == hash_bytes(package_bytes)
                    and identities.get("environment") == generation
                ):
                    attested = True
                    current_generation_attempt_ids.add(str(row.get("attempt_id")))
            if not attested:
                raise CheckpointValidationError(
                    "environment target lacks a canonical compiler/SDK/probe-bound runtime "
                    f"attestation: {intent.name}/{target}"
                )
    for stable_id, model in current_models.items():
        if model.get("status", {}).get("kind") != "runs":
            continue
        execution = model.get("execution")
        if not isinstance(execution, Mapping):
            raise CheckpointValidationError(f"current run has no execution block: {stable_id}")
        accepted_ids = {str(value) for value in execution.get("accepted_attempt_ids", [])}
        if not accepted_ids or not accepted_ids <= current_generation_attempt_ids:
            raise CheckpointValidationError(
                "current run was not executed entirely under the current environment generation: "
                f"{stable_id}"
            )
        if any(
            attempts_by_id[attempt_id].get("identities", {}).get("environment")
            != execution.get("env_generation")
            for attempt_id in accepted_ids
        ):
            raise CheckpointValidationError(
                f"current run execution.env_generation is stale: {stable_id}"
            )
    return environment_candidates


def _publish_generated_metadata_inventory(
    repo_root: Path,
    canonical_root: Path,
    candidates: Sequence[Path],
    validated_environment_candidates: frozenset[Path],
) -> tuple[GeneratedMetadataDisposition, ...]:
    """Publish the complete hash-bound generated/package metadata inventory.

    Parameters
    ----------
    repo_root, canonical_root, candidates, validated_environment_candidates:
        Repository roots, complete candidate set, and strict environment validation result.

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
        target_artifact = path.suffix == ".lock" or path.name.endswith(
            (".resolved.json", ".resolved.sha256")
        )
        if package_metadata and path not in validated_environment_candidates:
            raise CheckpointValidationError(
                f"environment metadata was not derived from the validated registry: {path}"
            )
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
                    "strict target lock/export with canonical runtime toolchain/probe attestation"
                    if package_metadata and target_artifact
                    else "registry-derived environment intent and probe contract"
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
    atomic_replace_bytes(manifest_path, data)
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
    external_text = _externally_controlled_record_text(path, content)
    if external_text:
        raise RestrictedPublicArtifact(
            "canonical record contains unredacted externally controlled text: "
            f"{path.as_posix()} fields={list(external_text)}"
        )
    return bool(excerpts)


def _externally_controlled_record_text(path: Path, content: bytes) -> tuple[str, ...]:
    """Find unredacted external stdio, traceback, exception, and response text.

    Parameters
    ----------
    path, content:
        Candidate generated-metadata path and exact bytes.

    Returns
    -------
    tuple[str, ...]
        JSON field paths whose values are neither hashes, local locators, nor licensed excerpts.
    """

    pure = PurePosixPath(path.as_posix())
    records_root = PurePosixPath("menagerie/crawler/records")
    if path.suffix not in {".json", ".jsonl"} or not (
        pure == records_root or records_root in pure.parents
    ):
        return ()
    try:
        values = (
            [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else [json.loads(content.decode("utf-8"))]
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ()
    findings: list[str] = []

    def is_safe_redaction(value: Any) -> bool:
        """Return whether externally controlled text was replaced by bounded metadata."""

        if value is None or value == "":
            return True
        if isinstance(value, Mapping):
            if value.get("license_disposition") in _SAFE_EXCERPT_DISPOSITIONS:
                return bool(
                    _HASH_PATTERN.fullmatch(str(value.get("text_sha256", "")))
                    and isinstance(value.get("locator"), str)
                )
            required = {"redaction", "content_sha256", "local_path", "diagnostic_key"}
            allowed = required | {"stream_sha256"}
            local_path = str(value.get("local_path", ""))
            pure_local = PurePosixPath(local_path)
            stream_sha256 = value.get("stream_sha256")
            return bool(
                set(value) <= allowed
                and required <= set(value)
                and value.get("redaction") == _DIAGNOSTIC_REDACTION_MARKER
                and _HASH_PATTERN.fullmatch(str(value.get("content_sha256", "")))
                and pure_local.parts[:2] == (".crawl-local", "diagnostics")
                and ".." not in pure_local.parts
                and pure_local.suffix == ".json"
                and re.fullmatch(r"\$[A-Za-z0-9_.\[\]-]+", str(value.get("diagnostic_key", "")))
                and (
                    stream_sha256 is None or _HASH_PATTERN.fullmatch(str(stream_sha256)) is not None
                )
            )
        return False

    def is_safe_completion_line(value: Any) -> bool:
        """Return whether a line is the closed TorchLens worker-completion shape."""

        if value is None:
            return True
        if not isinstance(value, str) or not value.startswith(_WORKER_COMPLETION_PREFIX):
            return False
        try:
            payload = json.loads(value[len(_WORKER_COMPLETION_PREFIX) :])
        except json.JSONDecodeError:
            return False
        return bool(
            isinstance(payload, Mapping)
            and set(payload) == {"raw_award_receipt_sha256", "request_nonce", "request_sha256"}
            and _HASH_PATTERN.fullmatch(str(payload.get("raw_award_receipt_sha256", "")))
            and isinstance(payload.get("request_nonce"), str)
            and bool(payload.get("request_nonce"))
            and _HASH_PATTERN.fullmatch(str(payload.get("request_sha256", "")))
        )

    def visit(value: Any, location: str = "$") -> None:
        """Collect unsafe values at externally controlled record fields."""

        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                failed_status_detail = (
                    key == "detail"
                    and location.endswith(".status")
                    and value.get("kind") == "failed"
                )
                if (
                    key in _EXTERNALLY_CONTROLLED_RECORD_FIELDS or failed_status_detail
                ) and not is_safe_redaction(nested):
                    findings.append(nested_location)
                if key == "stdout_completion_line" and not is_safe_completion_line(nested):
                    findings.append(nested_location)
                visit(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{location}[{index}]")

    for index, value in enumerate(values):
        visit(value, f"$[{index}]")
    return tuple(findings)


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


def _read_json_object(path: Path, label: str) -> JsonObject:
    """Read one JSON object without modifying its source.

    Parameters
    ----------
    path:
        JSON file to read.
    label:
        Diagnostic label used on failure.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"{label} must be a JSON object: {path}")
    return value


def reconstruction_transaction_id(
    stable_id: str,
    proposal_sha256: str,
    source_manifest: Mapping[str, Any],
    intake_item_sha256: str,
) -> str:
    """Recompute the exact promotion transaction identity.

    Parameters
    ----------
    stable_id, proposal_sha256, source_manifest, intake_item_sha256:
        Exact producer inputs used by accepted-artifact promotion.

    Returns
    -------
    str
        Canonical SHA-256 transaction identity.
    """

    return stable_hash(
        {
            "stable_id": stable_id,
            "proposal": proposal_sha256,
            "source_manifest": dict(source_manifest),
            "intake": intake_item_sha256,
        }
    )


def _reconstruction_has_canonical_anchor(
    manifest: Mapping[str, Any],
    proposal: Mapping[str, Any],
    proposal_digest: str,
    gates: Sequence[Mapping[str, Any]],
    current_models: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether immutable canonical history names this exact reconstruction.

    Parameters
    ----------
    manifest, proposal, proposal_digest:
        Reconstruction transaction facts.
    gates, current_models:
        Append-only accepted gate rows and current model revisions.

    Returns
    -------
    bool
        True for an exact accepted gate anchor or explicitly recorded pending proposal.
    """

    stable_id = proposal.get("stable_id")
    work_id = proposal.get("work_id")
    campaign_root = manifest.get("campaign_root_work_id")
    current = current_models.get(str(stable_id))
    if isinstance(current, Mapping):
        untrusted = current.get("untrusted_attempt")
        recorded = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        if (
            isinstance(untrusted, Mapping)
            and isinstance(recorded, Mapping)
            and dict(recorded) == dict(proposal)
            and untrusted.get("proposal_sha256") == proposal_digest
            and campaign_root == work_id
        ):
            return True
    current_gate_id = (
        current.get("accuracy_gate", {}).get("gate_id") if isinstance(current, Mapping) else None
    )
    if isinstance(current, Mapping):
        website = current.get("website")
        intake_item = manifest.get("intake_item")
        proposed_facts = proposal.get("proposed_facts")
        representative_id = current.get("identity", {}).get("family_representative_id")
        representative = current_models.get(str(representative_id))
        derivation = current.get("family_variant_derivation")
        variant_token = intake_item.get("variant") if isinstance(intake_item, Mapping) else None
        family_variant = bool(
            isinstance(website, Mapping)
            and website.get("kind") == "size-variant-template"
            and isinstance(intake_item, Mapping)
            and intake_item.get("stable_id") == stable_id
            and intake_item.get("variant_scope", "family") == "family"
            and intake_item.get("family_representative_id") == representative_id
            and isinstance(derivation, Mapping)
            and isinstance(representative, Mapping)
            and derivation.get("template_source_revision") == representative.get("record_revision")
            and campaign_root == work_id
            and proposal.get("proposal_id")
            == stable_hash(
                {
                    "template_source_revision": representative.get("record_revision"),
                    "stable_id": stable_id,
                    "work_id": work_id,
                }
            )
            and isinstance(proposed_facts, Mapping)
            and all(
                proposed_facts.get(field) == current.get(field)
                for field in ("identity", "implementation", "input_contract")
            )
            and proposal.get("recipe_revision")
            == current.get("implementation", {}).get("recipe_revision")
            and proposal.get("evidence_identity")
            == current.get("evidence", {}).get("evidence_identity")
            and proposal.get("verified_hashes", {}).get("family_template")
            == website.get("template_hash")
        )
        if family_variant:
            assert isinstance(representative, Mapping)
            try:
                validate_size_variant_derivation(
                    representative,
                    current,
                    str(representative_id),
                    trusted_variant_token=str(variant_token),
                )
            except FamilyTemplateError:
                family_variant = False
        if family_variant:
            assert isinstance(representative, Mapping)
            assert isinstance(proposed_facts, Mapping)
            representative_gate_id = representative.get("accuracy_gate", {}).get("gate_id")
            authorization = manifest.get("family_variant_authorization")
            authorization_valid = bool(
                isinstance(authorization, Mapping)
                and authorization.get("authorization_sha256")
                == stable_hash(
                    {
                        key: value
                        for key, value in authorization.items()
                        if key != "authorization_sha256"
                    }
                )
                and authorization.get("representative_stable_id") == representative_id
                and authorization.get("representative_record_revision")
                == representative.get("record_revision")
                and authorization.get("representative_gate_id") == representative_gate_id
                and authorization.get("derived_proposal_sha256") == proposal_digest
                and authorization.get("derived_source_manifest_sha256")
                == manifest.get("source_manifest", {}).get("manifest_sha256")
                and authorization.get("derived_source_facts_sha256")
                == stable_hash(proposed_facts.get("source_resolution"))
                and authorization.get("derived_evidence_facts_sha256")
                == stable_hash(proposed_facts.get("evidence"))
                and all(
                    proposed_facts.get(field) == current.get(field)
                    for field in (
                        "identity",
                        "taxonomy",
                        "external_metadata",
                        "people_and_origin",
                        "dates",
                        "citation",
                        "licenses",
                        "source_resolution",
                        "evidence",
                        "implementation",
                        "input_contract",
                    )
                )
            )
            if (
                authorization_valid
                and current_gate_id == representative_gate_id
                and current.get("accuracy_gate") == representative.get("accuracy_gate")
            ):
                assert isinstance(authorization, Mapping)
                for gate in gates:
                    if (
                        gate.get("gate_id") != representative_gate_id
                        or gate.get("gate_kind") != "metadata_batch"
                    ):
                        continue
                    if any(
                        isinstance(item, Mapping)
                        and item.get("stable_id") == representative_id
                        and item.get("verified_hashes", {}).get("proposal")
                        == authorization.get("representative_proposal_sha256")
                        and item.get("verified_hashes", {}).get("source_manifest")
                        == proposal.get("verified_hashes", {}).get("source_manifest")
                        and item.get("verdict") == "accurate"
                        and item.get("integrity", {}).get("verdict") == "accurate"
                        and item.get("rung_check", {}).get("verdict") == "accurate"
                        for item in gate.get("items", [])
                    ):
                        return True
    for gate in gates:
        if gate.get("gate_kind") != "metadata_batch":
            continue
        if current_gate_id is not None and gate.get("gate_id") != current_gate_id:
            continue
        for item in gate.get("items", []):
            if not isinstance(item, Mapping):
                continue
            if (
                item.get("stable_id") == stable_id
                and item.get("work_id") == work_id
                and item.get("campaign_root_work_id") == campaign_root
                and item.get("verified_hashes", {}).get("proposal") == proposal_digest
                and item.get("verified_hashes", {}).get("source_manifest")
                == proposal.get("verified_hashes", {}).get("source_manifest")
                and item.get("verdict") == "accurate"
                and item.get("integrity", {}).get("verdict") == "accurate"
                and item.get("rung_check", {}).get("verdict") == "accurate"
            ):
                return True
    return False


def _derive_candidate_paths(
    repo_root: Path,
    canonical_root: Path,
    authority_context: AuthorityContext,
) -> tuple[Path, ...]:
    """Derive the complete checkpoint set solely from canonical public roots.

    Parameters
    ----------
    repo_root, canonical_root, authority_context:
        Git worktree, canonical crawler root, and mandatory active authority.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted repository-relative files accepted by the checkpoint allowlist.
    """

    for manifest_path in sorted((canonical_root / "records" / "intake").glob("*/manifest.json")):
        try:
            load_intake_snapshot(manifest_path.parent)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, IntakeError) as exc:
            raise CheckpointValidationError(
                f"canonical intake snapshot is invalid: {manifest_path.parent}"
            ) from exc
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
        payload = _read_json_object(reconstruction, "canonical reconstruction")
        if payload.get("schema_version") == ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION:
            # The caller validates v1 documents from the independent artifact
            # ledger plus mandatory AuthorityContext before candidate derivation.
            continue
        raise CheckpointValidationError(
            f"legacy reconstruction is not artifact-ledger authority: {reconstruction}"
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
