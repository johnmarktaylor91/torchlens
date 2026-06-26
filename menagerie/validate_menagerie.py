"""Dependency-aware, disk-safe validator for the TorchLens model menagerie."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ContextManager, Mapping, Sequence, cast

from menagerie import cluster_runner
from menagerie.catalog import CatalogRow, load_rows
from menagerie import envs
from menagerie.recipe import (
    build_input_for_row,
    instantiate_model,
)
from menagerie.runtime import (
    CACHE_ROOTS,
    DependencyPlan,
    assert_min_free,
    combine_notes,
    cleanup_runtime,
    cuda_is_available,
    default_jobs,
    dependency_plan,
    device_note,
    disk_free_gb,
    group_by_dependency,
    install_dependency_plan,
    is_device_related_error,
    is_featured,
    log_event,
    move_model_and_input_to_device,
    purge_new_cache_entries,
    safe_path_part,
    select_rows,
    snapshot_cache,
    unrenderable_reason,
)
from menagerie.ledger import (
    ENV_VERIFICATION_DB,
    Scope,
    Status,
    VerificationRun,
    VerificationTarget,
    _resolve_verification_db,
    append_verification_run,
    base_lock_hash,
    base_env_hash,
    connect as connect_ledger,
    python_version,
    runner_host,
    torchlens_source_hash,
    utc_now,
)


DEFAULT_OUT_DIR = Path("/tmp/torchlens_menagerie_validation")
MB_PER_GB = 1024
DEFAULT_UNKNOWN_MEMORY_MB = 4 * MB_PER_GB
HEAVY_UNKNOWN_MEMORY_MB = 12 * MB_PER_GB
DEFAULT_MEMORY_FLOOR_GB = 12.0
AUTO_MEMORY_BUDGET_FRACTION = 0.7
FALLBACK_MEMORY_BUDGET_GB = 8.0
HEAVY_MEMORY_PATTERNS = (
    "deeplab",
    "mask2former",
    "segmentation",
    "segformer",
    "smp_",
    "u-net",
    "unet",
)
WORKER_MEMORY_CAP_STATUS = "failed:memory_cap"
WORKER_MEMORY_CAP_EXIT_CODE = 99
WORKER_MEMORY_CAP_POLL_INTERVAL_SEC = 0.5
WORKER_MEMORY_CAP_AS_BACKSTOP_MULTIPLIER = 1.5
WORKER_MEMORY_TEST_ALLOC_ENV = "TORCHLENS_MENAGERIE_WORKER_TEST_ALLOC_MB"
MANIFEST_STATUS_VALUES = frozenset(
    {
        "validated",
        "failed:exception",
        "failed:killed",
        "failed:memory_cap",
        "failed:native_crash",
        "failed:oom",
        "failed:replay",
        "failed:timeout",
        "failed:trace_summary",
        "skipped:cluster_unavailable",
        "skipped:dependency_unavailable",
        "skipped:dry_run",
        "skipped:unsupported_input_recipe",
    }
)
MANIFEST_COLUMNS = (
    "name",
    "model_id",
    "stable_id",
    "recipe_revision_sha256",
    "status",
    "n_ops",
    "validate_metadata_ok",
    "scope",
    "elapsed",
    "dependency_cluster",
    "error",
    "graph_shape_hash",
    "peak_rss_mb",
    "input_scale",
)
SUMMARY_JSON = "validation_summary.json"
REPORT_MD = "VALIDATION_REPORT.md"
RUNNER_CHOICES = ("auto", "local", "cluster")


@dataclass(frozen=True)
class SmokeCaseSettings:
    """Per-row smoke settings forwarded from the smoke manifest.

    Parameters
    ----------
    timeout_sec:
        Row-specific validation timeout.
    input_scale:
        Row-specific input scale.
    """

    timeout_sec: float
    input_scale: float


@dataclass(frozen=True)
class ValidationResult:
    """One model validation result.

    Parameters
    ----------
    name:
        Catalog model name.
    model_id:
        Catalog model identifier.
    status:
        Validation status.
    n_ops:
        Number of traced forward ops, when available.
    validate_metadata_ok:
        Whether forward metadata validation completed cleanly.
    scope:
        Requested validation scope.
    elapsed:
        Elapsed seconds.
    dependency_cluster:
        Dependency cluster used for this row.
    error:
        Error text or skip note.
    graph_shape_hash:
        TorchLens architecture hash for deduplication.
    stable_id:
        Opaque durable model identity.
    recipe_revision_sha256:
        Frozen recipe fingerprint for the row's current construction recipe.
    peak_rss_mb:
        Peak resident set size observed by the isolated worker process, in MB.
    input_scale:
        Input scaling factor used for this validation.
    """

    name: str
    model_id: int
    status: str
    n_ops: int | None
    validate_metadata_ok: bool
    scope: str
    elapsed: float
    dependency_cluster: str
    error: str
    graph_shape_hash: str = ""
    stable_id: str = ""
    recipe_revision_sha256: str = ""
    peak_rss_mb: int | None = None
    input_scale: float = 1.0


@dataclass(frozen=True)
class MemoryEstimate:
    """Memory estimate used by the validation scheduler.

    Parameters
    ----------
    estimated_mb:
        Estimated peak resident set size, in MB.
    source:
        Estimate source: ``"ledger"``, ``"heavy_default"``, or ``"default"``.
    """

    estimated_mb: int
    source: str


@dataclass(frozen=True)
class ValidationWorkItem:
    """One runnable validation item with its memory estimate.

    Parameters
    ----------
    plan:
        Dependency plan for the row's cluster.
    row:
        Catalog row to validate.
    estimated_memory_mb:
        Estimated peak resident set size for scheduler admission, in MB.
    estimate_source:
        Origin of the estimate used for logging and tests.
    """

    plan: DependencyPlan
    row: CatalogRow
    estimated_memory_mb: int
    estimate_source: str


@dataclass(frozen=True)
class AdmissionDecision:
    """Validation work admitted in one scheduler pass.

    Parameters
    ----------
    admitted:
        Items selected for submission.
    forced_oversized:
        Items admitted alone despite exceeding the memory budget.
    throttled:
        Whether pending work could not be admitted because in-flight estimated
        memory already consumed the remaining budget or actual free memory is
        below the admission floor.
    throttle_reason:
        Stable reason code for throttling, either ``"estimate_budget"``,
        ``"actual_free"``, or ``None`` when not throttled.
    """

    admitted: tuple[ValidationWorkItem, ...]
    forced_oversized: tuple[ValidationWorkItem, ...]
    throttled: bool
    throttle_reason: str | None = None


def manifest_records(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Read latest validation manifest rows keyed by stable model identity.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.

    Returns
    -------
    dict[str, dict[str, str]]
        Latest manifest rows.
    """

    if not manifest_path.exists():
        return {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["stable_id"]: row for row in reader if row.get("stable_id")}


def completed_stable_ids(manifest_path: Path, revalidate_failed: bool) -> set[str]:
    """Return stable IDs that should be skipped for resumable validation.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.
    revalidate_failed:
        Whether non-validated rows should be retried.

    Returns
    -------
    set[str]
        Stable IDs to skip.
    """

    records = manifest_records(manifest_path)
    if not revalidate_failed:
        return set(records)
    return {stable_id for stable_id, row in records.items() if row.get("status") == "validated"}


def _select_rows_for_validation(args: argparse.Namespace) -> list[CatalogRow]:
    """Select catalog rows without building the catalog when requested.

    Parameters
    ----------
    args:
        Parsed validator arguments.

    Returns
    -------
    list[CatalogRow]
        Selected catalog rows.
    """

    if not args.no_build_catalog:
        return select_rows(args)
    rows = cluster_runner.load_catalog_rows_ro(args.db)
    return _apply_catalog_filters_ro(rows, args)


def _apply_catalog_filters_ro(
    rows: Sequence[CatalogRow], args: argparse.Namespace
) -> list[CatalogRow]:
    """Apply validator catalog filters to read-only-loaded rows.

    Parameters
    ----------
    rows:
        Candidate catalog rows loaded from an existing database.
    args:
        Parsed validator arguments.

    Returns
    -------
    list[CatalogRow]
        Filtered rows in catalog order.
    """

    filtered = list(rows)
    if args.family:
        family = str(args.family).lower()
        filtered = [row for row in filtered if family in row.family_normalized.lower()]
    if args.domain:
        domain = str(args.domain).lower()
        filtered = [row for row in filtered if domain in row.domain.lower()]
    if args.zoo:
        zoo = str(args.zoo).lower()
        filtered = [row for row in filtered if zoo in row.zoo.lower()]
    if args.verified_only:
        filtered = [row for row in filtered if row.verified]
    if args.name:
        terms = [term.lower() for term in args.name]
        filtered = [row for row in filtered if any(term in row.name.lower() for term in terms)]
    if args.model_id:
        model_ids = set(args.model_id)
        filtered = [row for row in filtered if row.model_id in model_ids]
    if args.stable_ids:
        stable_ids = set(args.stable_ids)
        filtered = [row for row in filtered if row.stable_id in stable_ids]
    if args.featured_only:
        filtered = [row for row in filtered if is_featured(row)]
    if args.since is not None:
        filtered = [row for row in filtered if row.model_id > args.since]
    if args.subset is not None:
        filtered = filtered[: args.subset]
    if args.max_models is not None:
        filtered = filtered[: args.max_models]
    return filtered


def append_manifest(manifest_path: Path, result: ValidationResult) -> None:
    """Append one result row to the validation manifest.

    Parameters
    ----------
    manifest_path:
        Validation manifest path.
    result:
        Validation result.
    """

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        if write_header:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(
            (
                result.name,
                result.model_id,
                result.stable_id,
                result.recipe_revision_sha256,
                result.status,
                "" if result.n_ops is None else result.n_ops,
                str(result.validate_metadata_ok).lower(),
                result.scope,
                f"{result.elapsed:.3f}",
                result.dependency_cluster,
                result.error.replace("\n", " | "),
                result.graph_shape_hash,
                "" if result.peak_rss_mb is None else result.peak_rss_mb,
                result.input_scale,
            )
        )


def _package_version(package: str, fallback: str) -> str:
    """Return an installed package version with a safe fallback.

    Parameters
    ----------
    package:
        Distribution package name.
    fallback:
        Version string to return when distribution metadata is unavailable.

    Returns
    -------
    str
        Package version.
    """

    try:
        return version(package)
    except PackageNotFoundError:
        return fallback


def _torchlens_version() -> str:
    """Return the current TorchLens version.

    Returns
    -------
    str
        TorchLens package version.
    """

    import torchlens as tl

    return _package_version("torchlens", str(getattr(tl, "__version__", "unknown")))


def _torch_version() -> str:
    """Return the current PyTorch version.

    Returns
    -------
    str
        PyTorch package version.
    """

    import torch

    return str(torch.__version__)


def _ledger_status(status: str) -> str:
    """Map validation manifest status to ledger status.

    Parameters
    ----------
    status:
        Validation manifest status.

    Returns
    -------
    str
        Ledger status.
    """

    if status == "validated":
        return "passed"
    if status == "failed:timeout":
        return "timeout"
    if status == "failed:oom":
        return "oom"
    if status == "failed:native_crash":
        return "native_crash"
    if status == "failed:killed":
        return "killed"
    if status.startswith("skipped:"):
        return "skipped"
    if status.startswith("failed:"):
        return "failed"
    return "error"


def _forward_pass_value(result: ValidationResult, passed: bool, ledger_status: str) -> int | None:
    """Return the ledger forward-pass value for a validation result.

    Parameters
    ----------
    result:
        Validation result.
    passed:
        Whether the ledger status is a pass.
    ledger_status:
        Ledger status string.

    Returns
    -------
    int | None
        ``1`` when forward validation passed, ``0`` when it failed, otherwise ``None``.
    """

    if passed:
        return 1
    if ledger_status == "skipped":
        return None
    if result.scope == "forward+backward" and result.validate_metadata_ok:
        return 1
    return 0


def _backward_pass_value(result: ValidationResult, passed: bool) -> int | None:
    """Return the ledger backward-pass value for a validation result.

    Parameters
    ----------
    result:
        Validation result.
    passed:
        Whether the full requested validation passed.

    Returns
    -------
    int | None
        ``1`` when backward validation passed, ``0`` when it was attempted and failed, otherwise
        ``None``.
    """

    if result.scope != "forward+backward":
        return None
    if passed:
        return 1
    if result.validate_metadata_ok and "backward" in result.error.casefold():
        return 0
    return None


def _actual_device(result: ValidationResult, requested_device: str) -> str:
    """Infer the actual validation device from the result note.

    Parameters
    ----------
    result:
        Validation result.
    requested_device:
        Requested device mode.

    Returns
    -------
    str
        Actual device string.
    """

    match = re.search(r"(?:^|; )device=([^; |]+)", result.error)
    if match is not None:
        return match.group(1)
    return "cpu" if requested_device in {"cpu", "auto"} else requested_device


def _connect_verification_ledger(db_path: Path | None = None) -> Any:
    """Open the configured verification ledger.

    Parameters
    ----------
    db_path:
        Optional verification ledger path. When omitted, the ledger module
        resolves the active path at connection time.

    Returns
    -------
    Any
        Ledger connection context manager.
    """

    if db_path is not None:
        return connect_ledger(_resolve_verification_db(db_path))
    try:
        return connect_ledger()
    except TypeError:
        return connect_ledger(_resolve_verification_db(db_path))


def append_validation_ledger(
    row: CatalogRow,
    result: ValidationResult,
    device: str,
    input_scale: float,
    verification_db: Path | None = None,
) -> None:
    """Append one validation result to the verification ledger.

    Parameters
    ----------
    row:
        Catalog row.
    result:
        Validation result.
    device:
        Requested validation device.
    input_scale:
        Input scaling factor used for this validation.
    verification_db:
        Optional verification ledger path.
    """

    ledger_status = _ledger_status(result.status)
    env_hash = os.environ.get("TORCHLENS_MENAGERIE_ENV_HASH") or base_env_hash()
    lock_hash = os.environ.get("TORCHLENS_MENAGERIE_LOCK_HASH") or base_lock_hash()
    source_hash = os.environ.get("TORCHLENS_SOURCE_HASH") or torchlens_source_hash()
    passed = ledger_status == "passed"
    started_at = utc_now()
    finished_at = utc_now()
    with _connect_verification_ledger(verification_db) as conn:
        append_verification_run(
            conn,
            VerificationRun(
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                name=row.name,
                zoo=row.zoo,
                variant=row.variant,
                scope=cast(Scope, result.scope),
                status=cast(Status, ledger_status),
                forward_pass=_forward_pass_value(result, passed, ledger_status),
                backward_pass=_backward_pass_value(result, passed),
                backward_na_reason=None,
                metadata_ok=(
                    int(result.validate_metadata_ok)
                    if not result.status.startswith("skipped:")
                    else None
                ),
                n_ops=result.n_ops if passed else None,
                graph_shape_hash=result.graph_shape_hash or None,
                svg_sha256=None,
                torchlens_version=_torchlens_version(),
                torch_version=_torch_version(),
                python_version=python_version(),
                device_requested=device,
                device_actual=_actual_device(result, device),
                env_hash=env_hash,
                lock_hash=lock_hash,
                torchlens_source_hash=source_hash,
                input_scale=input_scale,
                runner_host=runner_host(),
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=result.elapsed,
                peak_rss_mb=result.peak_rss_mb,
                error_class=None if passed else result.status,
                error_message=None if passed else result.error,
            ),
        )


def result_from_payload(payload: Mapping[str, Any]) -> ValidationResult:
    """Build a validation result from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible result payload.

    Returns
    -------
    ValidationResult
        Parsed validation result.
    """

    raw_n_ops = payload.get("n_ops")
    n_ops = None if raw_n_ops in {None, ""} else int(raw_n_ops)
    raw_peak_rss_mb = payload.get("peak_rss_mb")
    peak_rss_mb = None if raw_peak_rss_mb in {None, ""} else int(raw_peak_rss_mb)
    return ValidationResult(
        name=str(payload["name"]),
        model_id=int(payload["model_id"]),
        status=str(payload["status"]),
        n_ops=n_ops,
        validate_metadata_ok=bool(payload["validate_metadata_ok"]),
        scope=str(payload["scope"]),
        elapsed=float(payload["elapsed"]),
        dependency_cluster=str(payload["dependency_cluster"]),
        error=str(payload["error"]),
        graph_shape_hash=str(payload.get("graph_shape_hash", "")),
        stable_id=str(payload.get("stable_id", "")),
        recipe_revision_sha256=str(payload.get("recipe_revision_sha256", "")),
        peak_rss_mb=peak_rss_mb,
        input_scale=float(payload.get("input_scale", 1.0) or 1.0),
    )


def _manifest_status_from_ledger_run(run: VerificationRun) -> str:
    """Return the validation-manifest status for a ledger run.

    Parameters
    ----------
    run:
        Verification ledger row.

    Returns
    -------
    str
        Manifest status string.
    """

    if run.status == "passed":
        return "validated"
    if run.error_class in MANIFEST_STATUS_VALUES:
        return str(run.error_class)
    if run.status == "timeout":
        return "failed:timeout"
    if run.status == "oom":
        return "failed:oom"
    if run.status == "native_crash":
        return "failed:native_crash"
    if run.status == "killed":
        return "failed:killed"
    if run.status == "env_unavailable":
        return (
            str(run.error_class)
            if run.error_class in MANIFEST_STATUS_VALUES
            else "skipped:cluster_unavailable"
        )
    if run.status == "skipped":
        return "skipped:dependency_unavailable"
    if run.status == "failed":
        return "failed:exception"
    return "failed:exception"


def _result_from_ledger_run(row: CatalogRow, run: VerificationRun) -> ValidationResult:
    """Build a validation manifest result from a merged ledger row.

    Parameters
    ----------
    row:
        Catalog row for model metadata not carried by the ledger.
    run:
        Verification ledger row.

    Returns
    -------
    ValidationResult
        Result suitable for appending to the validation manifest.
    """

    status = _manifest_status_from_ledger_run(run)
    return ValidationResult(
        name=run.name,
        model_id=row.model_id,
        status=status,
        n_ops=run.n_ops,
        validate_metadata_ok=bool(run.metadata_ok),
        scope=run.scope,
        elapsed=run.duration_sec,
        dependency_cluster="cluster",
        error=run.error_message or "",
        graph_shape_hash=run.graph_shape_hash or "",
        stable_id=run.stable_id,
        recipe_revision_sha256=run.recipe_revision_sha256,
        peak_rss_mb=run.peak_rss_mb,
        input_scale=float(run.input_scale or 1.0),
    )


def _latest_ledger_status(stable_id: str, ledger_db: Path | None = None) -> str | None:
    """Return the latest ledger status for a stable ID.

    Parameters
    ----------
    stable_id:
        Durable model identity.
    ledger_db:
        Optional verification ledger path.

    Returns
    -------
    str | None
        Latest status, or ``None`` when no current row exists.
    """

    with connect_ledger(_resolve_verification_db(ledger_db)) as conn:
        row = conn.execute(
            "SELECT status FROM current_verification WHERE stable_id = ?",
            (stable_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["status"])


def _verification_targets_for_rows(
    rows: Sequence[CatalogRow],
    *,
    scope: str,
    device: str,
) -> dict[str, VerificationTarget]:
    """Build current verification identity targets for catalog rows.

    Parameters
    ----------
    rows:
        Catalog rows.
    scope:
        Requested validation scope.
    device:
        Requested validation device policy.

    Returns
    -------
    dict[str, VerificationTarget]
        Targets keyed by stable ID.
    """

    env_hash = os.environ.get("TORCHLENS_MENAGERIE_ENV_HASH") or base_env_hash()
    lock_hash = os.environ.get("TORCHLENS_MENAGERIE_LOCK_HASH") or base_lock_hash()
    source_hash = os.environ.get("TORCHLENS_SOURCE_HASH") or torchlens_source_hash()
    return {
        row.stable_id: VerificationTarget(
            recipe_revision_sha256=row.recipe_revision_sha256,
            torchlens_source_hash=source_hash,
            env_hash=env_hash,
            lock_hash=lock_hash,
            device_requested=device,
            scope=cast(Scope, scope),
        )
        for row in rows
    }


def _base_env_rows(
    rows: Sequence[CatalogRow],
    *,
    env_registry: Path | None = None,
) -> tuple[CatalogRow, ...]:
    """Return only rows assigned to the base environment (no pixi island).

    Pixi-island models depend on packages that are intentionally absent from the
    base/cluster environment, so they MUST validate locally in their island env
    and can never be routed to the cluster. Filtering them out here keeps both
    ``auto`` and explicit ``cluster`` routing from sending an island model to a
    node that lacks its dependencies.

    Parameters
    ----------
    rows:
        Candidate catalog rows.
    env_registry:
        Optional environment registry path.

    Returns
    -------
    tuple[CatalogRow, ...]
        Rows whose island assignment is the base environment.
    """

    if not rows:
        return ()
    registry = envs.load_registry(env_registry) if env_registry else envs.load_registry()
    assignments = envs.assign(rows, registry)
    return tuple(row for row in rows if assignments.get(row.stable_id, "base") == "base")


def _cluster_candidate_rows(
    rows: Sequence[CatalogRow],
    runner: str,
    *,
    ledger_db: Path | None = None,
    env_registry: Path | None = None,
) -> tuple[CatalogRow, ...]:
    """Return rows that are candidates for cluster handling.

    Pixi-island rows are excluded before any RAM-based giant routing: an island
    model's dependencies are not present on the cluster, so it must validate in
    its island env locally regardless of its memory footprint.

    Parameters
    ----------
    rows:
        Candidate rows after manifest resume filtering.
    runner:
        Runner policy: ``"auto"``, ``"local"``, or ``"cluster"``.
    ledger_db:
        Optional verification ledger path.
    env_registry:
        Optional environment registry path used to identify island rows.

    Returns
    -------
    tuple[CatalogRow, ...]
        Rows that should not run locally.
    """

    if runner == "local":
        return ()
    base_rows = _base_env_rows(rows, env_registry=env_registry)
    if runner == "auto":
        return cluster_runner.route_giants(
            base_rows,
            ledger=_resolve_verification_db(ledger_db),
            local_ram_threshold_gb=cluster_runner.LOCAL_RAM_THRESHOLD_GB,
        )
    if runner == "cluster":
        return tuple(
            row
            for row in base_rows
            if _latest_ledger_status(row.stable_id, ledger_db) != "native_crash"
        )
    raise ValueError(f"unsupported runner {runner!r}")


def _completed_cluster_rows(
    rows: Sequence[CatalogRow],
    *,
    scope: str,
    device: str,
    ledger_db: Path | None = None,
) -> tuple[CatalogRow, ...]:
    """Return cluster candidate rows already completed in the ledger.

    Parameters
    ----------
    rows:
        Cluster candidate rows.
    scope:
        Requested validation scope.
    device:
        Requested validation device policy.
    ledger_db:
        Optional verification ledger path.

    Returns
    -------
    tuple[CatalogRow, ...]
        Rows with a current terminal ledger row.
    """

    targets = _verification_targets_for_rows(rows, scope=scope, device=device)
    completed = cluster_runner.ledger_completed_stable_ids(
        targets,
        ledger_db=_resolve_verification_db(ledger_db),
    )
    return tuple(row for row in rows if row.stable_id in completed)


def _cluster_route_rows(
    rows: Sequence[CatalogRow],
    runner: str,
    *,
    scope: str = "forward",
    device: str = "cpu",
    ledger_db: Path | None = None,
    env_registry: Path | None = None,
) -> tuple[CatalogRow, ...]:
    """Return ledger-pending rows that should be dispatched to the cluster.

    Parameters
    ----------
    rows:
        Candidate rows after manifest resume filtering.
    runner:
        Runner policy: ``"auto"``, ``"local"``, or ``"cluster"``.
    scope:
        Requested validation scope.
    device:
        Requested validation device policy.
    ledger_db:
        Optional verification ledger path.
    env_registry:
        Optional environment registry path used to exclude pixi-island rows.

    Returns
    -------
    tuple[CatalogRow, ...]
        Rows to dispatch to the cluster.
    """

    candidates = _cluster_candidate_rows(
        rows, runner, ledger_db=ledger_db, env_registry=env_registry
    )
    completed = {
        row.stable_id
        for row in _completed_cluster_rows(
            candidates, scope=scope, device=device, ledger_db=ledger_db
        )
    }
    return tuple(row for row in candidates if row.stable_id not in completed)


def _append_cluster_rows_from_ledger(
    rows: Sequence[CatalogRow],
    manifest_path: Path,
    *,
    ledger_db: Path | None = None,
) -> None:
    """Append merged cluster ledger rows to the validation manifest.

    Parameters
    ----------
    rows:
        Cluster-routed catalog rows.
    manifest_path:
        Validation manifest path.
    ledger_db:
        Optional verification ledger path.
    """

    for row in rows:
        run = cluster_runner.latest_verification_run_for_stable_id(
            row.stable_id, ledger_db=_resolve_verification_db(ledger_db)
        )
        append_manifest(manifest_path, _result_from_ledger_run(row, run))


def _append_cluster_dry_run_results(
    rows: Sequence[CatalogRow],
    manifest_path: Path,
    args: argparse.Namespace,
) -> None:
    """Append dry-run skips for cluster-routed rows.

    Parameters
    ----------
    rows:
        Cluster-routed catalog rows.
    manifest_path:
        Validation manifest path.
    args:
        Parsed validator arguments.
    """

    for row in rows:
        append_manifest(
            manifest_path,
            ValidationResult(
                row.name,
                row.model_id,
                "skipped:dry_run",
                0,
                False,
                args.scope,
                0.0,
                "cluster",
                "cluster dispatch dry run",
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                input_scale=args.input_scale,
            ),
        )


def _append_cluster_unavailable_rows(
    rows: Sequence[CatalogRow],
    manifest_path: Path,
    args: argparse.Namespace,
    *,
    reason: str,
    detail: str,
) -> None:
    """Append terminal env-unavailable ledger rows for cluster transport failures.

    Parameters
    ----------
    rows:
        Cluster-routed catalog rows.
    manifest_path:
        Validation manifest path.
    args:
        Parsed validator arguments.
    reason:
        Stable reason code.
    detail:
        Transport failure detail.
    """

    env_hash = os.environ.get("TORCHLENS_MENAGERIE_ENV_HASH") or base_env_hash()
    lock_hash = os.environ.get("TORCHLENS_MENAGERIE_LOCK_HASH") or base_lock_hash()
    source_hash = os.environ.get("TORCHLENS_SOURCE_HASH") or torchlens_source_hash()
    started_at = utc_now()
    finished_at = utc_now()
    message = f"reason={reason}; {detail}".strip()
    with _connect_verification_ledger(args.verification_db) as conn:
        for row in rows:
            run = VerificationRun(
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                name=row.name,
                zoo=row.zoo,
                variant=row.variant,
                scope=cast(Scope, args.scope),
                status="env_unavailable",
                forward_pass=None,
                backward_pass=None,
                backward_na_reason=None,
                metadata_ok=None,
                n_ops=None,
                graph_shape_hash=None,
                svg_sha256=None,
                torchlens_version=_torchlens_version(),
                torch_version=_torch_version(),
                python_version=python_version(),
                device_requested=args.device,
                device_actual=None,
                env_hash=env_hash,
                lock_hash=lock_hash,
                torchlens_source_hash=source_hash,
                input_scale=args.input_scale,
                runner_host=runner_host(),
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=0.0,
                peak_rss_mb=None,
                error_class="skipped:cluster_unavailable",
                error_message=message,
            )
            append_verification_run(conn, run)
            append_manifest(manifest_path, _result_from_ledger_run(row, run))


def _append_cluster_task_failed_rows(
    rows: Sequence[CatalogRow],
    manifest_path: Path,
    args: argparse.Namespace,
    *,
    detail: str,
    error_message_by_stable_id: Mapping[str, str] | None = None,
) -> None:
    """Append honest per-model failure rows for cluster tasks with no valid result.

    Each model passed here is attributed by its OWN task outcome, never by the
    array-job-level return code: it produced no valid, verified result artifact,
    so it failed/crashed/never-started on the cluster. The recorded message is the
    task's own ``.err`` tail when available (``error_message_by_stable_id``),
    falling back to the compact array ``detail`` only when the task left no log.

    A model whose worker DID write an honest result (already merged into the
    ledger for the current identity tuple) is never overwritten -- its real
    per-model status (validated, or its honest ``failed:*``) is reflected
    straight from the ledger. The tripwire stays armed: a genuine validation
    failure on the cluster still surfaces its honest ``failed:*``.

    Parameters
    ----------
    rows:
        Cluster-routed catalog rows whose tasks produced no valid result.
    manifest_path:
        Validation manifest path.
    args:
        Parsed validator arguments.
    detail:
        Fallback failure detail (return code / array-level tail) used only when a
        task left no usable ``.err`` log.
    error_message_by_stable_id:
        Optional per-model honest error message recovered from the task's own
        ``.err`` log.
    """

    if not rows:
        return
    env_hash = os.environ.get("TORCHLENS_MENAGERIE_ENV_HASH") or base_env_hash()
    lock_hash = os.environ.get("TORCHLENS_MENAGERIE_LOCK_HASH") or base_lock_hash()
    source_hash = os.environ.get("TORCHLENS_SOURCE_HASH") or torchlens_source_hash()
    started_at = utc_now()
    finished_at = utc_now()
    per_model = error_message_by_stable_id or {}
    targets = _verification_targets_for_rows(rows, scope=args.scope, device=args.device)
    already_merged = cluster_runner.ledger_completed_stable_ids(
        targets, ledger_db=_resolve_verification_db(args.verification_db)
    )
    with _connect_verification_ledger(args.verification_db) as conn:
        for row in rows:
            if row.stable_id in already_merged:
                run = cluster_runner.latest_verification_run_for_stable_id(
                    row.stable_id, ledger_db=_resolve_verification_db(args.verification_db)
                )
                append_manifest(manifest_path, _result_from_ledger_run(row, run))
                continue
            task_error = per_model.get(row.stable_id)
            if task_error:
                message = f"cluster task failed; {task_error}".strip()
            else:
                message = f"cluster task failed; no result returned; {detail}".strip()
            run = VerificationRun(
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                name=row.name,
                zoo=row.zoo,
                variant=row.variant,
                scope=cast(Scope, args.scope),
                status="failed",
                forward_pass=None,
                backward_pass=None,
                backward_na_reason=None,
                metadata_ok=None,
                n_ops=None,
                graph_shape_hash=None,
                svg_sha256=None,
                torchlens_version=_torchlens_version(),
                torch_version=_torch_version(),
                python_version=python_version(),
                device_requested=args.device,
                device_actual=None,
                env_hash=env_hash,
                lock_hash=lock_hash,
                torchlens_source_hash=source_hash,
                input_scale=args.input_scale,
                runner_host=runner_host(),
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=0.0,
                peak_rss_mb=None,
                error_class="failed:cluster_task_failed",
                error_message=message,
            )
            append_verification_run(conn, run)
            append_manifest(manifest_path, _result_from_ledger_run(row, run))


def _attribute_cluster_results_per_model(
    rows: Sequence[CatalogRow],
    args: argparse.Namespace,
    manifest_path: Path,
    dispatch: cluster_runner.DispatchResult,
    *,
    detail: str,
    job_ids: Sequence[str] = (),
) -> bool:
    """Collect, merge, and attribute cluster results PER-MODEL.

    This is the single source of truth for turning a dispatched array into ledger
    rows, used both when ``sbatch --wait`` returned zero AND when it returned
    non-zero (a partial array failure). Attribution is per-model, by each task's
    own artifact -- never by the array-job-level return code:

    * A model whose task wrote a valid, verified result -> its real merged status
      (validated, or its honest per-model ``failed:*``).
    * A model whose task produced no valid result -> an honest per-model
      ``failed:cluster_task_failed`` carrying that task's own ``.err`` tail when
      available.

    Parameters
    ----------
    rows:
        All cluster-routed catalog rows for this dispatch.
    args:
        Parsed validator arguments.
    manifest_path:
        Validation manifest path.
    dispatch:
        Dispatch metadata (assignments + remote artifact dir).
    detail:
        Compact context for the fallback message (array return code / tail).
    job_ids:
        SLURM array job IDs used to locate per-task ``.err`` logs.

    Returns
    -------
    bool
        ``True`` when at least one per-task result artifact was collected (the
        normal partial/full case). ``False`` when ZERO results returned -- the
        caller decides whether that is a genuine transport failure.

    Raises
    ------
    cluster_runner.ClusterMergeConflict
        If a present result conflicts with an existing ledger payload (fail-loud).
    """

    collected = cluster_runner.collect_cluster_results_partial(dispatch)
    if collected.result_rows_path is not None and collected.result_manifest_path is not None:
        cluster_runner.merge_cluster_results(
            collected.result_rows_path,
            collected.result_manifest_path,
            local_ledger_db=_resolve_verification_db(args.verification_db),
        )
    rows_by_id = {row.stable_id: row for row in rows}
    present_ids = {assignment.stable_id for assignment in collected.present_assignments}
    present_rows = [row for row in rows if row.stable_id in present_ids]
    _append_cluster_rows_from_ledger(present_rows, manifest_path, ledger_db=args.verification_db)
    missing_rows = [
        rows_by_id[assignment.stable_id]
        for assignment in collected.missing_assignments
        if assignment.stable_id in rows_by_id
    ]
    error_message_by_stable_id: dict[str, str] = {}
    for assignment in collected.missing_assignments:
        task_error = cluster_runner.read_task_error_log(
            collected.log_dir,
            job_ids,
            assignment.array_index,
        )
        if task_error:
            error_message_by_stable_id[assignment.stable_id] = task_error
    _append_cluster_task_failed_rows(
        missing_rows,
        manifest_path,
        args,
        detail=detail,
        error_message_by_stable_id=error_message_by_stable_id,
    )
    return bool(collected.present_assignments) or bool(collected.missing_assignments)


def _cluster_transport_failure_detail(error: BaseException) -> str:
    """Return a compact transport failure detail string.

    Parameters
    ----------
    error:
        Transport exception.

    Returns
    -------
    str
        Human-readable detail suitable for the ledger.
    """

    if isinstance(error, subprocess.CalledProcessError):
        stderr = (error.stderr or "").strip()
        stdout = (error.stdout or "").strip()
        tail = stderr or stdout or repr(error.cmd)
        return f"returncode={error.returncode}; {tail}"
    if isinstance(error, subprocess.TimeoutExpired):
        return f"timeout={error.timeout}; cmd={error.cmd!r}"
    return repr(error)


def _run_cluster_validation(
    rows: Sequence[CatalogRow],
    args: argparse.Namespace,
    out_dir: Path,
    manifest_path: Path,
    smoke_settings: Mapping[str, SmokeCaseSettings] | None = None,
) -> None:
    """Dispatch cluster-routed rows, merge results, and append report rows.

    Parameters
    ----------
    rows:
        Cluster-routed catalog rows.
    args:
        Parsed validator arguments.
    out_dir:
        Validation output directory.
    manifest_path:
        Validation manifest path.
    smoke_settings:
        Optional per-row smoke settings.
    """

    if not rows:
        return
    stable_ids = [row.stable_id for row in rows]
    row_settings = smoke_settings or {}
    log_event("cluster_dispatch_start", rows=len(stable_ids), stable_ids=stable_ids)
    try:
        dispatch = cluster_runner.dispatch_giants(
            stable_ids,
            catalog_db=args.db,
            ledger_db=_resolve_verification_db(args.verification_db),
            local_artifact_root=out_dir / "cluster",
            timeout_by_id={
                row.stable_id: _case_timeout(row, row_settings, args.timeout_sec) for row in rows
            },
            input_scale_by_id={
                row.stable_id: _case_input_scale(row, row_settings, args.input_scale)
                for row in rows
            },
            dry_run=args.dry_run,
        )
    except subprocess.TimeoutExpired as error:
        detail = _cluster_transport_failure_detail(error)
        _append_cluster_unavailable_rows(
            rows,
            manifest_path,
            args,
            reason="cluster_timeout",
            detail=detail,
        )
        log_event(
            "cluster_dispatch_unavailable",
            rows=len(stable_ids),
            reason="cluster_timeout",
            error=detail,
        )
        return
    except cluster_runner.ClusterJobFailed as error:
        # The job was SUBMITTED and RAN but exited non-zero. This is a real
        # job/validation failure, never a benign cluster-unavailable skip. Try
        # to collect any honest result rows the worker wrote, then surface an
        # honest failure for giants that produced nothing.
        _surface_cluster_job_failure(rows, args, manifest_path, error)
        log_event(
            "cluster_job_failed",
            rows=len(stable_ids),
            job_ids=list(error.job_ids),
            error=error.detail,
        )
        return
    except (OSError, subprocess.CalledProcessError) as error:
        detail = _cluster_transport_failure_detail(error)
        _append_cluster_unavailable_rows(
            rows,
            manifest_path,
            args,
            reason="cluster_unreachable",
            detail=detail,
        )
        log_event(
            "cluster_dispatch_unavailable",
            rows=len(stable_ids),
            reason="cluster_unreachable",
            error=detail,
        )
        return
    if args.dry_run:
        _append_cluster_dry_run_results(rows, manifest_path, args)
        log_event(
            "cluster_dispatch_dry_run",
            rows=len(stable_ids),
            artifact_dir=str(dispatch.local_artifact_dir),
        )
        return
    # The array `sbatch --wait` returned zero, but that is a BATCH-level signal;
    # attribute per-model anyway so a task that failed while the batch reported
    # success (or a partially-collected result tree) is still surfaced honestly,
    # never blanket-validated. Transport failures while collecting are the only
    # path to cluster_unavailable.
    try:
        attributed = _attribute_cluster_results_per_model(
            rows,
            args,
            manifest_path,
            dispatch,
            detail="sbatch --wait returncode=0",
            job_ids=dispatch.sbatch_job_ids,
        )
    except subprocess.TimeoutExpired as error:
        detail = _cluster_transport_failure_detail(error)
        _append_cluster_unavailable_rows(
            rows,
            manifest_path,
            args,
            reason="cluster_timeout",
            detail=detail,
        )
        log_event(
            "cluster_collect_unavailable",
            rows=len(stable_ids),
            reason="cluster_timeout",
            error=detail,
        )
        return
    except (OSError, subprocess.CalledProcessError) as error:
        detail = _cluster_transport_failure_detail(error)
        _append_cluster_unavailable_rows(
            rows,
            manifest_path,
            args,
            reason="cluster_unreachable",
            detail=detail,
        )
        log_event(
            "cluster_collect_unavailable",
            rows=len(stable_ids),
            reason="cluster_unreachable",
            error=detail,
        )
        return
    if not attributed:
        # Zero results came back despite a clean submit -- treat as a transport/
        # collection failure rather than blanket-failing every model.
        _append_cluster_unavailable_rows(
            rows,
            manifest_path,
            args,
            reason="cluster_no_results",
            detail="no per-model result artifacts returned from the cluster",
        )
        log_event(
            "cluster_collect_unavailable",
            rows=len(stable_ids),
            reason="cluster_no_results",
            error="no result artifacts returned",
        )
        return
    log_event(
        "cluster_dispatch_done",
        rows=len(stable_ids),
        artifact_dir=str(dispatch.local_artifact_dir),
    )


def _surface_cluster_job_failure(
    rows: Sequence[CatalogRow],
    args: argparse.Namespace,
    manifest_path: Path,
    error: cluster_runner.ClusterJobFailed,
) -> None:
    """Surface honest PER-MODEL results for an array job that exited non-zero.

    A non-zero ``sbatch --wait`` for a SLURM ARRAY means at least one task failed,
    NOT that every dispatched model failed: other tasks may have validated and
    written honest result artifacts. Results are therefore collected and
    attributed per-model regardless of the array-job return code:

    * a model whose task wrote a valid result keeps its real merged status, and
    * a model whose task produced no valid result gets an honest per-model
      ``failed:cluster_task_failed`` carrying that task's own ``.err`` tail.

    Only when the dispatch carried no usable context, or ZERO results came back,
    does this fall back to a single honest batch-level failure. Under no
    circumstance is a submitted + run array recorded as
    ``skipped:cluster_unavailable``, and a model that genuinely failed validation
    on the cluster still surfaces its honest ``failed:*``.

    Parameters
    ----------
    rows:
        Cluster-routed catalog rows.
    args:
        Parsed validator arguments.
    manifest_path:
        Validation manifest path.
    error:
        The raised job-failure exception, carrying best-effort dispatch context.
    """

    dispatch = error.dispatch
    if dispatch is None:
        # No dispatch context to collect per-model results from; fall back to a
        # single honest batch-level failure for every dispatched model.
        _append_cluster_task_failed_rows(rows, manifest_path, args, detail=error.detail)
        return
    job_ids = error.job_ids or dispatch.sbatch_job_ids
    try:
        _attribute_cluster_results_per_model(
            rows,
            args,
            manifest_path,
            dispatch,
            detail=error.detail,
            job_ids=job_ids,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        cluster_runner.ClusterResultIntegrityError,
        cluster_runner.ClusterMergeConflict,
    ) as collect_error:
        # Could not collect/merge per-model results at all; surface one honest
        # batch-level failure rather than masking it behind a benign skip.
        log_event(
            "cluster_job_failed_collect_skipped",
            rows=len(rows),
            error=repr(collect_error),
        )
        _append_cluster_task_failed_rows(rows, manifest_path, args, detail=error.detail)


def latest_peak_rss_estimates(ledger_db: Path | None = None) -> dict[str, int]:
    """Return latest recorded peak RSS values keyed by stable model ID.

    Parameters
    ----------
    ledger_db:
        Optional verification ledger path.

    Returns
    -------
    dict[str, int]
        Latest non-null peak RSS measurements from the verification ledger.
    """

    query = """
        WITH ranked AS (
            SELECT
                stable_id,
                peak_rss_mb,
                ROW_NUMBER() OVER (
                    PARTITION BY stable_id
                    ORDER BY finished_at DESC, run_id DESC
                ) AS rn
            FROM verification_runs
            WHERE peak_rss_mb IS NOT NULL
              AND scope = 'forward'
        )
        SELECT stable_id, peak_rss_mb
        FROM ranked
        WHERE rn = 1
    """
    with connect_ledger(_resolve_verification_db(ledger_db)) as conn:
        rows = conn.execute(query).fetchall()
    return {str(row["stable_id"]): int(row["peak_rss_mb"]) for row in rows}


def _looks_memory_heavy(row: CatalogRow) -> bool:
    """Return whether an unmeasured model should receive the high memory default.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    bool
        ``True`` for names/families/domains that suggest high-resolution segmentation.
    """

    haystack = " ".join((row.name, row.family, row.domain, row.zoo)).casefold()
    return any(pattern in haystack for pattern in HEAVY_MEMORY_PATTERNS)


def _memory_estimate_for_row(
    row: CatalogRow, ledger_estimates_mb: Mapping[str, int]
) -> MemoryEstimate:
    """Return the scheduler memory estimate for one catalog row.

    Parameters
    ----------
    row:
        Catalog row.
    ledger_estimates_mb:
        Latest ledger peak RSS measurements keyed by stable ID.

    Returns
    -------
    MemoryEstimate
        Estimate and source.
    """

    measured_mb = ledger_estimates_mb.get(row.stable_id)
    if measured_mb is not None and measured_mb > 0:
        return MemoryEstimate(measured_mb, "ledger")
    if _looks_memory_heavy(row):
        return MemoryEstimate(HEAVY_UNKNOWN_MEMORY_MB, "heavy_default")
    return MemoryEstimate(DEFAULT_UNKNOWN_MEMORY_MB, "default")


def _build_validation_work_items(
    runnable: Sequence[tuple[DependencyPlan, CatalogRow]],
    ledger_estimates_mb: Mapping[str, int],
) -> list[ValidationWorkItem]:
    """Build scheduler work items with memory estimates.

    Parameters
    ----------
    runnable:
        Dependency-qualified rows ready for validation.
    ledger_estimates_mb:
        Latest ledger peak RSS measurements keyed by stable ID.

    Returns
    -------
    list[ValidationWorkItem]
        Work items in their original runnable order.
    """

    items = []
    for plan, row in runnable:
        estimate = _memory_estimate_for_row(row, ledger_estimates_mb)
        items.append(
            ValidationWorkItem(
                plan=plan,
                row=row,
                estimated_memory_mb=estimate.estimated_mb,
                estimate_source=estimate.source,
            )
        )
    return items


def _resolve_memory_budget_gb(requested_budget_gb: float | None) -> float:
    """Resolve an explicit or automatic validation memory budget.

    Parameters
    ----------
    requested_budget_gb:
        User-provided budget in GB, or ``None`` for automatic detection.

    Returns
    -------
    float
        Memory budget in GB.
    """

    if requested_budget_gb is not None:
        return max(0.001, requested_budget_gb)
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        return FALLBACK_MEMORY_BUDGET_GB
    return max(0.001, available_gb * AUTO_MEMORY_BUDGET_FRACTION)


def _resolve_scheduler_memory_budget_gb(
    requested_budget_gb: float | None,
    worker_memory_cap_gb: float | None,
    effective_jobs: int,
) -> float:
    """Resolve the scheduler's estimated in-flight memory budget.

    Parameters
    ----------
    requested_budget_gb:
        User-provided soft scheduler budget, or ``None`` for the default policy.
    worker_memory_cap_gb:
        Per-worker hard RSS cap, or ``None`` when disabled.
    effective_jobs:
        Maximum number of workers that can run concurrently.

    Returns
    -------
    float
        Estimated in-flight memory budget in GB.
    """

    if worker_memory_cap_gb is None:
        return _resolve_memory_budget_gb(requested_budget_gb)
    cap_budget_gb = max(0.001, worker_memory_cap_gb) * max(1, effective_jobs)
    if requested_budget_gb is None:
        return cap_budget_gb
    return min(_resolve_memory_budget_gb(requested_budget_gb), cap_budget_gb)


def _resolve_memory_floor_gb(requested_floor_gb: float | None) -> float:
    """Resolve the actual-free-memory admission floor.

    Parameters
    ----------
    requested_floor_gb:
        User-provided floor in GB, or ``None`` for the default.

    Returns
    -------
    float
        Actual-free-memory floor in GB.
    """

    if requested_floor_gb is not None:
        return max(0.001, requested_floor_gb)
    return DEFAULT_MEMORY_FLOOR_GB


def _actual_available_memory_mb() -> int | None:
    """Return currently available system memory in MB.

    Returns
    -------
    int | None
        Available memory in MB, or ``None`` when it cannot be measured.
    """

    try:
        import psutil

        return int(psutil.virtual_memory().available // (1024**2))
    except Exception:
        return None


def _in_flight_memory_mb(
    in_flight: Mapping[Future[tuple[ValidationResult, int]], ValidationWorkItem],
) -> int:
    """Return total estimated memory for currently submitted validation work.

    Parameters
    ----------
    in_flight:
        Future-to-work-item mapping.

    Returns
    -------
    int
        Sum of in-flight estimated peak RSS values, in MB.
    """

    return sum(item.estimated_memory_mb for item in in_flight.values())


def _admit_memory_budgeted_items(
    pending: list[ValidationWorkItem],
    in_flight_memory_mb: int,
    in_flight_count: int,
    budget_mb: int,
    memory_floor_mb: int,
    actual_available_memory_mb: int | None,
    available_slots: int,
) -> AdmissionDecision:
    """Select pending validation work that fits memory admission gates.

    Parameters
    ----------
    pending:
        Mutable queue of not-yet-submitted work items.
    in_flight_memory_mb:
        Estimated memory already submitted and still running, in MB.
    in_flight_count:
        Number of submitted jobs still running.
    budget_mb:
        Maximum estimated in-flight memory, in MB.
    memory_floor_mb:
        Minimum actual free memory required before admitting another job, in MB.
        The effective floor for each item is ``max(memory_floor_mb,
        item.estimated_memory_mb)``.
    actual_available_memory_mb:
        Currently available system memory in MB, or ``None`` when unavailable.
    available_slots:
        Remaining concurrency slots under the hard job cap.

    Returns
    -------
    AdmissionDecision
        Items admitted, oversized items forced to run alone, and throttle state.
    """

    admitted: list[ValidationWorkItem] = []
    forced_oversized: list[ValidationWorkItem] = []
    throttled = False
    throttle_reason: str | None = None
    while pending and len(admitted) < available_slots:
        admitted_memory_mb = sum(item.estimated_memory_mb for item in admitted)
        remaining_mb = budget_mb - in_flight_memory_mb - admitted_memory_mb
        estimate_fit_indexes = [
            index for index, item in enumerate(pending) if item.estimated_memory_mb <= remaining_mb
        ]
        force_first_job = in_flight_count == 0 and not admitted

        def has_actual_headroom(item: ValidationWorkItem) -> bool:
            """Return whether actual free memory admits an item.

            Parameters
            ----------
            item:
                Candidate work item.

            Returns
            -------
            bool
                True when the actual-free-memory gate is satisfied.
            """

            if actual_available_memory_mb is None:
                return True
            if force_first_job:
                return True
            required_mb = max(memory_floor_mb, item.estimated_memory_mb)
            return actual_available_memory_mb >= required_mb

        fit_index = next(
            (index for index in estimate_fit_indexes if has_actual_headroom(pending[index])),
            None,
        )
        if fit_index is None:
            if not admitted and in_flight_count == 0:
                item = pending.pop(0)
                admitted.append(item)
                if item.estimated_memory_mb > budget_mb:
                    forced_oversized.append(item)
            elif pending and available_slots > len(admitted):
                throttled = True
                throttle_reason = "actual_free" if estimate_fit_indexes else "estimate_budget"
            break
        admitted.append(pending.pop(fit_index))
    return AdmissionDecision(
        tuple(admitted),
        tuple(forced_oversized),
        throttled,
        throttle_reason,
    )


def _refresh_pending_estimates(
    pending: list[ValidationWorkItem], stable_id: str, peak_rss_mb: int
) -> None:
    """Update pending duplicate rows with a fresh same-run memory measurement.

    Parameters
    ----------
    pending:
        Mutable pending work queue.
    stable_id:
        Stable ID whose estimate should be updated.
    peak_rss_mb:
        Fresh peak RSS measurement in MB.
    """

    for index, item in enumerate(pending):
        if item.row.stable_id != stable_id:
            continue
        pending[index] = ValidationWorkItem(
            plan=item.plan,
            row=item.row,
            estimated_memory_mb=peak_rss_mb,
            estimate_source="ledger",
        )


def catalog_row_from_payload(payload: Mapping[str, Any]) -> CatalogRow:
    """Build a catalog row from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible row payload.

    Returns
    -------
    CatalogRow
        Catalog row.
    """

    return CatalogRow(
        model_id=int(payload["model_id"]),
        display_index=int(payload.get("display_index", payload["model_id"])),
        stable_id=str(payload.get("stable_id", "")),
        name=str(payload["name"]),
        variant=str(payload.get("variant", "")),
        family=str(payload["family"]),
        family_normalized=str(payload["family_normalized"]),
        domain=str(payload["domain"]),
        zoo=str(payload["zoo"]),
        constructor_call=str(payload["constructor_call"]),
        input_shape=str(payload["input_shape"]),
        input_dtype=str(payload["input_dtype"]),
        era=str(payload["era"]),
        verified=bool(payload["verified"]),
        notes=str(payload["notes"]),
        source=str(payload.get("source", "catalog")),
        recipe_revision_sha256=str(payload.get("recipe_revision_sha256", "")),
    )


MIN_SCALED_SPATIAL_DIM = 32


def _build_input(row: CatalogRow, input_scale: float = 1.0) -> Any:
    """Build the example input for a catalog row.

    Parameters
    ----------
    row:
        Catalog row.
    input_scale:
        Spatial down-scaling factor applied to the built example input. ``1.0``
        (the default) leaves the input untouched. A value ``< 1.0`` shrinks the
        spatial dimensions so models that are too large at full resolution can be
        validated at reduced input size. The recipe/identity is NOT affected: the
        input is excluded from ``recipe_revision_sha256``.

    Returns
    -------
    Any
        Example input, spatially down-scaled when ``input_scale < 1.0``.
    """

    example_input = build_input_for_row(row)
    if input_scale >= 1.0:
        return example_input
    return _scale_example_input(example_input, input_scale)


def _scaled_spatial_size(size: int, input_scale: float) -> int:
    """Return one down-scaled spatial dimension, clamped to a safe minimum.

    Parameters
    ----------
    size:
        Original spatial dimension length.
    input_scale:
        Scaling factor in ``(0, 1)``.

    Returns
    -------
    int
        Scaled length, rounded and clamped to at least ``MIN_SCALED_SPATIAL_DIM``
        (never larger than the original, so scaling only ever shrinks).
    """

    scaled = int(round(size * input_scale))
    scaled = max(MIN_SCALED_SPATIAL_DIM, scaled)
    return min(size, scaled)


def _scale_example_input(example_input: Any, input_scale: float) -> Any:
    """Down-scale the spatial dimensions of an example input tree.

    Tensors with at least three dimensions are treated as having a leading batch
    dim (0) and channel dim (1); every dimension from index 2 onward is a spatial
    dimension and is scaled down. For a 4D image tensor ``(N, C, H, W)`` this
    scales ``H`` and ``W`` while keeping ``N`` and ``C``. Containers (tuple, list,
    dict) are scaled element-wise; everything else is returned unchanged.

    A fresh tensor of the reduced shape is allocated with the original dtype and
    device (example inputs are random placeholders, so values are irrelevant) --
    this avoids interpolation constraints on integer / non-image tensors.

    Parameters
    ----------
    example_input:
        Example input object or nested container thereof.
    input_scale:
        Scaling factor in ``(0, 1)``.

    Returns
    -------
    Any
        Spatially down-scaled copy of the input tree.
    """

    import torch

    if isinstance(example_input, torch.Tensor):
        if example_input.dim() < 3:
            return example_input
        new_shape = list(example_input.shape)
        for dim_index in range(2, example_input.dim()):
            new_shape[dim_index] = _scaled_spatial_size(new_shape[dim_index], input_scale)
        if tuple(new_shape) == tuple(example_input.shape):
            return example_input
        scaled = torch.zeros(new_shape, dtype=example_input.dtype, device=example_input.device)
        # requires_grad is only valid for floating/complex tensors; preserve it
        # when the original carried it and the dtype supports autograd.
        if example_input.requires_grad and (scaled.is_floating_point() or scaled.is_complex()):
            scaled.requires_grad_(True)
        return scaled
    if isinstance(example_input, tuple):
        return tuple(_scale_example_input(item, input_scale) for item in example_input)
    if isinstance(example_input, list):
        return [_scale_example_input(item, input_scale) for item in example_input]
    if isinstance(example_input, dict):
        return {
            key: _scale_example_input(value, input_scale) for key, value in example_input.items()
        }
    return example_input


def _sum_float_outputs(output: Any) -> Any:
    """Return a scalar sum over floating tensors in a nested output.

    Parameters
    ----------
    output:
        Model output object.

    Returns
    -------
    Any
        Scalar tensor loss.
    """

    import torch

    terms: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        """Collect floating tensor leaves from nested outputs."""

        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point or value.dtype.is_complex:
                terms.append(value.float().sum())
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(output)
    if not terms:
        raise ValueError("backward validation requires at least one floating tensor output")
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def _trace_n_ops_and_hash_from_trace(trace: Any) -> tuple[int | None, str, str]:
    """Count forward ops and compute the architecture hash from a validation trace.

    Parameters
    ----------
    trace:
        Completed TorchLens validation trace.

    Returns
    -------
    tuple[int | None, str, str]
        Number of traced ops, graph-shape hash, and error text. ``n_ops`` is ``None``
        when trace summarization fails.
    """

    try:
        n_ops = int(getattr(trace, "num_ops", 0) or len(getattr(trace, "layer_logs", {}) or {}))
        graph_shape_hash = str(getattr(trace, "graph_shape_hash", "") or "")
    except Exception as error:
        return None, "", f"{error!r}\n{traceback.format_exc(limit=8)}"
    return n_ops, graph_shape_hash, ""


def _peak_rss_mb() -> int:
    """Return this process's peak resident set size in MB.

    Returns
    -------
    int
        Peak resident set size rounded up to whole MB.
    """

    # Linux reports ru_maxrss in KiB. The validator runs on Linux workers.
    peak_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max(0, (peak_kib + 1023) // 1024)


def _bytes_to_mb(value: int) -> int:
    """Round a byte count up to whole MB.

    Parameters
    ----------
    value:
        Byte count.

    Returns
    -------
    int
        Whole-MB byte count rounded up.
    """

    return max(0, (value + (1024**2 - 1)) // (1024**2))


def _memory_cap_result(
    row: CatalogRow,
    scope: str,
    cap_gb: float,
    peak_rss_bytes: int,
    elapsed: float,
    input_scale: float = 1.0,
) -> ValidationResult:
    """Build the honest failure result for a worker RSS cap breach.

    Parameters
    ----------
    row:
        Catalog row being validated.
    scope:
        Validation scope requested by the parent.
    cap_gb:
        Per-worker RSS cap in GB.
    peak_rss_bytes:
        Highest RSS observed by the monitor, in bytes.
    elapsed:
        Elapsed worker time in seconds.
    input_scale:
        Input scaling factor used for this validation.

    Returns
    -------
    ValidationResult
        Memory-cap validation failure result.
    """

    plan = dependency_plan(row)
    peak_rss_mb = _bytes_to_mb(peak_rss_bytes)
    return ValidationResult(
        row.name,
        row.model_id,
        WORKER_MEMORY_CAP_STATUS,
        0,
        False,
        scope,
        elapsed,
        plan.cluster_key,
        (f"worker RSS exceeded --worker-memory-cap-gb={cap_gb:.3f}; peak_rss_mb={peak_rss_mb}"),
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        peak_rss_mb=peak_rss_mb,
        input_scale=input_scale,
    )


def _has_oom_evidence(stdout: str, stderr: str) -> bool:
    """Return whether worker output contains OOM evidence.

    Parameters
    ----------
    stdout:
        Worker standard output.
    stderr:
        Worker standard error.

    Returns
    -------
    bool
        ``True`` when output indicates kernel, cgroup, CUDA, or scheduler OOM.
    """

    evidence = f"{stdout}\n{stderr}".casefold()
    patterns = (
        r"\bout of memory\b",
        r"\boom\b",
        r"\boom-kill\b",
        r"\boom_kill\b",
        r"\boom killed\b",
        r"\boom-killer\b",
        r"\bcuda error: out of memory\b",
        r"\bkilled process\b",
        r"\bmemory cgroup\b",
        r"\bcgroup out of memory\b",
        r"\bslurmstepd: error: detected\b.*\boom\b",
        r"\bslurmstepd: error:.*out[ -]of[ -]memory\b",
    )
    return any(re.search(pattern, evidence) is not None for pattern in patterns)


def _worker_exit_status(returncode: int, stdout: str, stderr: str) -> str:
    """Classify a worker process exit status from its return code.

    Parameters
    ----------
    returncode:
        Subprocess return code.
    stdout:
        Worker standard output.
    stderr:
        Worker standard error.

    Returns
    -------
    str
        Manifest validation status.
    """

    if returncode == -signal.SIGKILL:
        return "failed:oom" if _has_oom_evidence(stdout, stderr) else "failed:killed"
    if returncode in {-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGBUS}:
        return "failed:native_crash"
    if returncode < 0:
        return "failed:killed"
    if returncode > 0:
        return "failed:exception"
    return "failed:exception"


def _emit_worker_result(result: ValidationResult) -> None:
    """Emit one JSON worker result event.

    Parameters
    ----------
    result:
        Validation result to send to the parent process.
    """

    print(json.dumps({"event": "worker_result", "result": result.__dict__}), flush=True)


def _set_worker_address_space_backstop(cap_bytes: int, current_vms_bytes: int) -> None:
    """Set a generous virtual-memory backstop for capped workers when possible.

    Parameters
    ----------
    cap_bytes:
        RSS cap in bytes.
    current_vms_bytes:
        Current virtual memory size in bytes.
    """

    if not hasattr(resource, "RLIMIT_AS"):
        return
    backstop_bytes = max(
        int(cap_bytes * WORKER_MEMORY_CAP_AS_BACKSTOP_MULTIPLIER),
        current_vms_bytes + int(cap_bytes * 2),
    )
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
        if hard_limit != resource.RLIM_INFINITY:
            backstop_bytes = min(backstop_bytes, hard_limit)
        if soft_limit != resource.RLIM_INFINITY:
            backstop_bytes = min(backstop_bytes, soft_limit)
        resource.setrlimit(resource.RLIMIT_AS, (backstop_bytes, hard_limit))
    except (OSError, ValueError):
        return


def _start_worker_memory_monitor(
    row: CatalogRow,
    scope: str,
    cap_gb: float | None,
    start_time: float,
    input_scale: float = 1.0,
) -> threading.Thread | None:
    """Start a daemon RSS monitor that records memory-cap failures.

    Parameters
    ----------
    row:
        Catalog row being validated.
    scope:
        Validation scope requested by the parent.
    cap_gb:
        Per-worker RSS cap in GB, or ``None`` to disable the monitor.
    start_time:
        Worker monotonic start time.
    input_scale:
        Input scaling factor used for this validation.

    Returns
    -------
    threading.Thread | None
        Started monitor thread, or ``None`` when no cap is active.
    """

    if cap_gb is None:
        return None
    cap_bytes = max(1, int(cap_gb * (1024**3)))
    import psutil

    process = psutil.Process()
    initial_memory = process.memory_info()
    peak_rss_bytes = int(initial_memory.rss)
    _set_worker_address_space_backstop(cap_bytes, int(initial_memory.vms))

    def monitor() -> None:
        """Hard-exit the worker after emitting a result if RSS exceeds the cap."""

        nonlocal peak_rss_bytes
        while True:
            try:
                rss_bytes = int(process.memory_info().rss)
            except psutil.Error:
                return
            peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            if rss_bytes > cap_bytes:
                result = _memory_cap_result(
                    row,
                    scope,
                    cap_gb,
                    peak_rss_bytes,
                    time.monotonic() - start_time,
                    input_scale,
                )
                _emit_worker_result(result)
                os._exit(WORKER_MEMORY_CAP_EXIT_CODE)
            time.sleep(WORKER_MEMORY_CAP_POLL_INTERVAL_SEC)

    thread = threading.Thread(target=monitor, name="menagerie-rss-cap", daemon=True)
    thread.start()
    return thread


def _run_worker_test_allocation_from_env() -> list[bytearray]:
    """Allocate test memory in worker subprocesses when requested by tests.

    Returns
    -------
    list[bytearray]
        Retained allocation chunks, empty outside test-hook usage.
    """

    requested_mb = int(os.environ.get(WORKER_MEMORY_TEST_ALLOC_ENV, "0") or "0")
    if requested_mb <= 0:
        return []
    chunks: list[bytearray] = []
    chunk_mb = 8
    remaining_mb = requested_mb
    while remaining_mb > 0:
        this_chunk_mb = min(chunk_mb, remaining_mb)
        chunks.append(bytearray(this_chunk_mb * 1024 * 1024))
        remaining_mb -= this_chunk_mb
        time.sleep(0.05)
    time.sleep(WORKER_MEMORY_CAP_POLL_INTERVAL_SEC * 2)
    return chunks


def _annotate_input_scale(result: ValidationResult, input_scale: float) -> ValidationResult:
    """Prepend an ``input_scale`` note to a result for reduced-input auditability.

    Parameters
    ----------
    result:
        Validation result to annotate.
    input_scale:
        Spatial scaling factor applied to the example input.

    Returns
    -------
    ValidationResult
        The result unchanged when ``input_scale`` is the full-resolution ``1.0``,
        otherwise a copy whose ``error`` note records the scaling so the manifest
        makes reduced-input validations auditable.
    """

    if input_scale >= 1.0:
        return result
    note = f"input_scale={input_scale:g}"
    annotated_error = combine_notes(note, result.error)
    return ValidationResult(
        result.name,
        result.model_id,
        result.status,
        result.n_ops,
        result.validate_metadata_ok,
        result.scope,
        result.elapsed,
        result.dependency_cluster,
        annotated_error,
        result.graph_shape_hash,
        stable_id=result.stable_id,
        recipe_revision_sha256=result.recipe_revision_sha256,
        peak_rss_mb=result.peak_rss_mb,
        input_scale=input_scale,
    )


def validate_one(
    row: CatalogRow,
    dry_run: bool,
    scope: str,
    device: str,
    input_scale: float = 1.0,
) -> ValidationResult:
    """Instantiate and validate one menagerie model.

    Parameters
    ----------
    row:
        Catalog row.
    dry_run:
        Build recipe only when true.
    scope:
        Validation scope, ``"forward"`` or ``"forward+backward"``.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.
    input_scale:
        Spatial down-scaling factor for the built example input (``1.0`` = full
        resolution). Values ``< 1.0`` validate too-large models at reduced input
        size; the scaling is recorded in the result note for auditability and does
        not alter the recipe identity (input is excluded from
        ``recipe_revision_sha256``).

    Returns
    -------
    ValidationResult
        Validation result.
    """

    return _annotate_input_scale(
        _validate_one_unscaled_note(row, dry_run, scope, device, input_scale),
        input_scale,
    )


def _validate_one_unscaled_note(
    row: CatalogRow,
    dry_run: bool,
    scope: str,
    device: str,
    input_scale: float,
) -> ValidationResult:
    """Instantiate and validate one model; the input-scale note is added by the caller.

    Parameters
    ----------
    row:
        Catalog row.
    dry_run:
        Build recipe only when true.
    scope:
        Validation scope.
    device:
        Device mode.
    input_scale:
        Spatial down-scaling factor for the built example input.

    Returns
    -------
    ValidationResult
        Validation result without the input-scale audit note.
    """

    start = time.monotonic()
    plan = dependency_plan(row)
    synthetic_result = _validate_smoke_synthetic(row, scope, input_scale, start, plan.cluster_key)
    if synthetic_result is not None:
        return synthetic_result
    skip_reason = unrenderable_reason(row)
    if skip_reason is not None:
        return ValidationResult(
            row.name,
            row.model_id,
            f"skipped:{skip_reason}",
            0,
            False,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            skip_reason,
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
            input_scale=input_scale,
        )
    try:
        input_tensor = _build_input(row, input_scale)
    except Exception as error:
        return ValidationResult(
            row.name,
            row.model_id,
            "skipped:unsupported_input_recipe",
            0,
            False,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            str(error),
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    if dry_run:
        return ValidationResult(
            row.name,
            row.model_id,
            "skipped:dry_run",
            0,
            False,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            "validated recipe",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )

    model = instantiate_model(row)
    if hasattr(model, "eval"):
        model.eval()

    def attempt_validation(
        attempt_model: Any, attempt_input: Any, actual_device: str
    ) -> ValidationResult:
        """Validate the model on one resolved device.

        Parameters
        ----------
        attempt_model:
            Model prepared for the attempt device.
        attempt_input:
            Example input prepared for the attempt device.
        actual_device:
            Device used by this attempt.

        Returns
        -------
        ValidationResult
            Validation result for this attempt.
        """

        import torchlens as tl
        from torchlens.user_funcs import _validate_forward_pass_torch

        n_ops: int | None = None
        graph_shape_hash = ""
        trace_error = ""
        replay_failure_summary = ""

        def observe_validation_trace(trace: Any) -> None:
            """Record summary fields from the trace already built for validation.

            Also lifts the structured replay-failure diagnostic off the live
            trace (before cleanup) so a ``failed:replay`` row carries the ACTUAL
            mismatch -- divergent op, shapes/dtypes, max abs/rel diff -- instead
            of the bare ``repr(False)``. Reading the diagnostic NEVER changes the
            pass/fail decision; it is a richer error message only.

            Parameters
            ----------
            trace:
                Completed TorchLens validation trace.
            """

            nonlocal graph_shape_hash, n_ops, trace_error, replay_failure_summary
            n_ops, graph_shape_hash, trace_error = _trace_n_ops_and_hash_from_trace(trace)
            try:
                from torchlens.validation.diagnostics import get_validation_failure

                failure = get_validation_failure(trace)
                if failure is not None:
                    replay_failure_summary = failure.summary()
            except Exception:
                # Best-effort: a diagnostics read must never break validation.
                replay_failure_summary = ""

        try:
            forward_result = _validate_forward_pass_torch(
                attempt_model,
                attempt_input,
                validate_metadata=True,
                _trace_observer=observe_validation_trace,
            )
        except Exception as error:
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                False,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                combine_notes(
                    device_note(device, actual_device),
                    f"{error!r}\n{traceback.format_exc(limit=8)}",
                ),
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        if not bool(forward_result):
            # Prefer the structured diagnostic captured in the observer (the real
            # mismatch); fall back to repr(forward_result) only if it is missing.
            replay_detail = replay_failure_summary or f"replay failed ({forward_result!r})"
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:replay",
                0,
                False,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                combine_notes(device_note(device, actual_device), replay_detail),
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )

        backward_error = ""
        if scope == "forward+backward":
            try:
                backward_result = tl.validate(
                    attempt_model,
                    attempt_input,
                    scope="backward",
                    loss_fn=_sum_float_outputs,
                    validate_metadata=True,
                )
            except Exception as error:
                return ValidationResult(
                    row.name,
                    row.model_id,
                    "failed:exception",
                    0,
                    True,
                    scope,
                    time.monotonic() - start,
                    plan.cluster_key,
                    combine_notes(
                        device_note(device, actual_device),
                        f"backward validation failed: {error!r}\n{traceback.format_exc(limit=8)}",
                    ),
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
            if not bool(backward_result):
                return ValidationResult(
                    row.name,
                    row.model_id,
                    "failed:replay",
                    0,
                    True,
                    scope,
                    time.monotonic() - start,
                    plan.cluster_key,
                    combine_notes(
                        device_note(device, actual_device),
                        f"backward validation returned {backward_result!r}",
                    ),
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
            backward_error = f"; backward={backward_result!r}"

        if n_ops is None:
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:trace_summary",
                None,
                True,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                combine_notes(device_note(device, actual_device), trace_error),
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        return ValidationResult(
            row.name,
            row.model_id,
            "validated",
            n_ops,
            True,
            scope,
            time.monotonic() - start,
            plan.cluster_key,
            combine_notes(
                device_note(device, actual_device),
                f"forward={forward_result!r}{backward_error}",
            ),
            graph_shape_hash,
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )

    if device == "cuda":
        try:
            model, input_tensor = move_model_and_input_to_device(model, input_tensor, "cuda")
        except Exception as error:
            return ValidationResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                False,
                scope,
                time.monotonic() - start,
                plan.cluster_key,
                f"device=cuda; {error!r}\n{traceback.format_exc(limit=8)}",
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
            )
        return attempt_validation(model, input_tensor, "cuda")

    if device == "auto":
        cpu_result = attempt_validation(model, input_tensor, "cpu")
        if (
            cpu_result.status == "failed:exception"
            and is_device_related_error(RuntimeError(cpu_result.error))
            and cuda_is_available()
        ):
            try:
                model, input_tensor = move_model_and_input_to_device(model, input_tensor, "cuda")
            except Exception as error:
                return ValidationResult(
                    row.name,
                    row.model_id,
                    "failed:exception",
                    0,
                    False,
                    scope,
                    time.monotonic() - start,
                    plan.cluster_key,
                    f"device=cuda; {error!r}\n{traceback.format_exc(limit=8)}",
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                )
            return attempt_validation(model, input_tensor, "cuda")
        if cpu_result.error:
            return ValidationResult(
                cpu_result.name,
                cpu_result.model_id,
                cpu_result.status,
                cpu_result.n_ops,
                cpu_result.validate_metadata_ok,
                cpu_result.scope,
                cpu_result.elapsed,
                cpu_result.dependency_cluster,
                combine_notes(device_note(device, "cpu"), cpu_result.error),
                cpu_result.graph_shape_hash,
                stable_id=cpu_result.stable_id,
                recipe_revision_sha256=cpu_result.recipe_revision_sha256,
                peak_rss_mb=cpu_result.peak_rss_mb,
                input_scale=cpu_result.input_scale,
            )
        return cpu_result

    return attempt_validation(model, input_tensor, "cpu")


def _validate_smoke_synthetic(
    row: CatalogRow,
    scope: str,
    input_scale: float,
    start: float,
    dependency_cluster: str,
) -> ValidationResult | None:
    """Run smoke-only synthetic validation primitives.

    Parameters
    ----------
    row:
        Catalog row.
    scope:
        Validation scope.
    input_scale:
        Input scale recorded for the row.
    start:
        Monotonic start time.
    dependency_cluster:
        Dependency cluster label.

    Returns
    -------
    ValidationResult | None
        Synthetic result, or ``None`` for ordinary catalog rows.
    """

    if not row.stable_id.startswith("smoke_"):
        return None
    if row.stable_id == "smoke_exc_1":
        raise RuntimeError("synthetic smoke constructor exception")
    if row.stable_id == "smoke_crash_1":
        import ctypes

        ctypes.memset(0, 0, 1)
    if row.stable_id == "smoke_memory_cap_1":
        return _memory_cap_result(row, scope, 0.001, 0, time.monotonic() - start, input_scale)
    if row.stable_id in {"smoke_vmap_1", "smoke_dataparallel_1"}:
        return ValidationResult(
            row.name,
            row.model_id,
            "validated",
            1,
            True,
            scope,
            time.monotonic() - start,
            dependency_cluster,
            "synthetic smoke capture primitive",
            graph_shape_hash=f"synthetic:{row.stable_id}",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
            input_scale=input_scale,
        )
    return ValidationResult(
        row.name,
        row.model_id,
        "failed:exception",
        0,
        False,
        scope,
        time.monotonic() - start,
        dependency_cluster,
        f"unknown synthetic smoke stable_id={row.stable_id}",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        input_scale=input_scale,
    )


def validate_with_timeout(
    row: CatalogRow,
    dry_run: bool,
    scope: str,
    device: str,
    timeout_sec: float,
    tmp_dir: Path | None = None,
    worker_memory_cap_gb: float | None = None,
    input_scale: float = 1.0,
    retry_worker_exception: bool = True,
) -> ValidationResult:
    """Run one validation in an isolated child process with a timeout.

    Parameters
    ----------
    row:
        Catalog row.
    dry_run:
        Build recipe only when true.
    scope:
        Validation scope.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.
    timeout_sec:
        Maximum wall time in seconds.
    tmp_dir:
        Optional per-model temporary directory routed to the worker via the
        ``TMPDIR``/``TEMP``/``TMP`` environment variables. Passed through the
        subprocess environment (not process globals) so concurrent workers each
        get an isolated scratch directory without mutating shared state.
    worker_memory_cap_gb:
        Optional per-worker RSS cap in GB enforced inside the child process.
    input_scale:
        Spatial down-scaling factor for the example input, forwarded to the worker
        via ``--input-scale`` (``1.0`` = full resolution).
    retry_worker_exception:
        Retry once when a capped worker reports a generic exception. The retry
        keeps memory-cap tests robust to transient broad-suite worker setup
        failures while preserving deterministic failures.

    Returns
    -------
    ValidationResult
        Validation result.
    """

    plan = dependency_plan(row)
    command = [
        sys.executable,
        "-m",
        "menagerie.validate_menagerie",
        "--worker-row-json",
        json.dumps(row.__dict__),
        "--scope",
        scope,
        "--device",
        device,
    ]
    if dry_run:
        command.append("--dry-run")
    if worker_memory_cap_gb is not None:
        command.extend(("--worker-memory-cap-gb", f"{worker_memory_cap_gb:.6f}"))
    if input_scale != 1.0:
        command.extend(("--input-scale", f"{input_scale:.6f}"))
    child_env = dict(os.environ)
    if device == "cpu":
        child_env["CUDA_VISIBLE_DEVICES"] = ""
    for thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_env[thread_var] = "1"
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TEMP", "TMP"):
            child_env[key] = str(tmp_dir)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            row.name,
            row.model_id,
            "failed:timeout",
            0,
            False,
            scope,
            timeout_sec,
            plan.cluster_key,
            f"timed out after {timeout_sec:.1f}s",
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "worker_result":
            result = result_from_payload(payload["result"])
            if (
                result.status == "failed:exception"
                and retry_worker_exception
                and worker_memory_cap_gb is not None
            ):
                return validate_with_timeout(
                    row,
                    dry_run,
                    scope,
                    device,
                    timeout_sec,
                    tmp_dir=tmp_dir,
                    worker_memory_cap_gb=worker_memory_cap_gb,
                    input_scale=input_scale,
                    retry_worker_exception=False,
                )
            return result
    if completed.returncode == WORKER_MEMORY_CAP_EXIT_CODE and worker_memory_cap_gb is not None:
        return _memory_cap_result(
            row,
            scope,
            worker_memory_cap_gb,
            0,
            0.0,
            input_scale,
        )
    if completed.returncode != 0:
        stderr_tail = " | ".join(completed.stderr.strip().splitlines()[-5:])
        stdout_tail = " | ".join(completed.stdout.strip().splitlines()[-5:])
        status = _worker_exit_status(completed.returncode, completed.stdout, completed.stderr)
        message = stderr_tail or stdout_tail or f"worker exited with code {completed.returncode}"
        return ValidationResult(
            row.name,
            row.model_id,
            status,
            0,
            False,
            scope,
            0.0,
            plan.cluster_key,
            message,
            stable_id=row.stable_id,
            recipe_revision_sha256=row.recipe_revision_sha256,
            input_scale=input_scale,
        )
    return ValidationResult(
        row.name,
        row.model_id,
        "failed:exception",
        0,
        False,
        scope,
        0.0,
        plan.cluster_key,
        "worker did not emit a worker_result event",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        input_scale=input_scale,
    )


def _status_bucket(status: str) -> str:
    """Return the high-level status bucket.

    Parameters
    ----------
    status:
        Manifest status.

    Returns
    -------
    str
        ``validated``, ``failed``, or ``skipped``.
    """

    if status == "validated":
        return "validated"
    if status.startswith("failed:"):
        return "failed"
    return "skipped"


def write_reports(
    out_dir: Path,
    manifest_path: Path,
    rows: Sequence[CatalogRow],
    *,
    catalog_db: Path | None = None,
    no_build_catalog: bool = False,
) -> None:
    """Write validation summary JSON and Markdown reports.

    Parameters
    ----------
    out_dir:
        Output directory.
    manifest_path:
        Validation manifest path.
    rows:
        Selected catalog rows used for report context.
    catalog_db:
        Catalog database path for report context.
    no_build_catalog:
        Whether to avoid build-capable catalog loading.
    """

    records = manifest_records(manifest_path)
    resolved_catalog = catalog_db or Path(__file__).parent / "data" / "catalog.db"
    catalog_rows = (
        cluster_runner.load_catalog_rows_ro(resolved_catalog)
        if no_build_catalog
        else load_rows(db_path=resolved_catalog)
    )
    row_by_stable_id = {row.stable_id: row for row in catalog_rows}
    row_by_stable_id.update({row.stable_id: row for row in rows})
    statuses = Counter(_status_bucket(row.get("status", "")) for row in records.values())
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_zoo: dict[str, Counter[str]] = defaultdict(Counter)
    failures = []
    for stable_id, record in records.items():
        catalog_row = row_by_stable_id.get(stable_id)
        domain = catalog_row.domain if catalog_row else "unknown"
        zoo = catalog_row.zoo if catalog_row else "unknown"
        bucket = _status_bucket(record.get("status", ""))
        by_domain[domain][bucket] += 1
        by_zoo[zoo][bucket] += 1
        if bucket == "failed":
            failures.append(
                {
                    "name": record.get("name", ""),
                    "stable_id": stable_id,
                    "model_id": record.get("model_id", ""),
                    "status": record.get("status", ""),
                    "error": record.get("error", ""),
                }
            )
    headline = (
        "TorchLens forward validation has algorithmically verified saved activation "
        f"replay for {statuses['validated']} menagerie models."
    )
    summary = {
        "totals": {
            "validated": statuses["validated"],
            "failed": statuses["failed"],
            "skipped": statuses["skipped"],
            "total": sum(statuses.values()),
        },
        "by_domain": {key: dict(value) for key, value in sorted(by_domain.items())},
        "by_zoo": {key: dict(value) for key, value in sorted(by_zoo.items())},
        "headline": headline,
        "failures": failures,
        "manifest": str(manifest_path),
    }
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# TorchLens Menagerie Validation Report",
        "",
        headline,
        "",
        "## Totals",
        "",
        f"- Validated: {statuses['validated']}",
        f"- Failed: {statuses['failed']}",
        f"- Skipped: {statuses['skipped']}",
        f"- Total manifest rows: {sum(statuses.values())}",
        "",
        "## Counts by Domain",
        "",
        "| Domain | Validated | Failed | Skipped |",
        "| --- | ---: | ---: | ---: |",
    ]
    for domain, counts in sorted(by_domain.items()):
        lines.append(
            f"| {domain} | {counts['validated']} | {counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(
        [
            "",
            "## Counts by Zoo",
            "",
            "| Zoo | Validated | Failed | Skipped |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for zoo, counts in sorted(by_zoo.items()):
        lines.append(
            f"| {zoo} | {counts['validated']} | {counts['failed']} | {counts['skipped']} |"
        )
    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- {failure['name']} ({failure['model_id']}): "
                f"{failure['status']} - {failure['error']}"
            )
    else:
        lines.append("No failures recorded.")
    (out_dir / REPORT_MD).write_text("\n".join(lines) + "\n")


def _island_validator_args(args: argparse.Namespace) -> list[str]:
    """Return validator CLI arguments forwarded inside an assigned island.

    Parameters
    ----------
    args:
        Parent validator arguments.

    Returns
    -------
    list[str]
        Extra CLI arguments for ``menagerie.validate_menagerie`` in the island env.
    """

    forwarded = [
        "--base-env-only",
        "--scope",
        str(args.scope),
        "--timeout-sec",
        f"{args.timeout_sec:.6f}",
        "--out-dir",
        str(args.out_dir),
        "--min-free-gb",
        f"{args.min_free_gb:.6f}",
        "--input-scale",
        f"{args.input_scale:.6f}",
        "--runner",
        str(args.runner),
    ]
    if args.manifest is not None:
        forwarded.extend(("--manifest", str(args.manifest)))
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.keep_cache:
        forwarded.append("--keep-cache")
    if args.revalidate_failed:
        forwarded.append("--revalidate-failed")
    if args.verification_db is not None:
        forwarded.extend(("--verification-db", str(args.verification_db)))
    if args.smoke_manifest is not None:
        forwarded.extend(("--smoke-manifest", str(args.smoke_manifest)))
    return forwarded


def run(args: argparse.Namespace) -> int:
    """Run the dependency-aware disk-safe validator.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    out_dir = args.out_dir.resolve()
    manifest_path = (args.manifest or out_dir / "validation_manifest.tsv").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_settings = _load_smoke_case_settings(args.smoke_manifest)
    selected = _select_rows_for_validation(args)
    if args.report_only:
        write_reports(
            out_dir,
            manifest_path,
            selected,
            catalog_db=args.db,
            no_build_catalog=args.no_build_catalog,
        )
        log_event("report_done", manifest=str(manifest_path), out_dir=str(out_dir))
        return 0

    run_cache_snapshots = [snapshot_cache(root) for root in CACHE_ROOTS]
    start_free_gb = disk_free_gb(out_dir)
    log_event(
        "validation_run_start",
        out_dir=str(out_dir),
        free_gb=round(start_free_gb, 3),
        input_scale=args.input_scale,
    )
    assert_min_free(out_dir, args.min_free_gb)

    done = completed_stable_ids(manifest_path, args.revalidate_failed)
    rows = [row for row in selected if row.stable_id not in done]
    log_event("selected", count=len(rows), skipped_existing=len(selected) - len(rows))
    # Base-env-only mode runs inside an island pixi env or the cluster worker;
    # those subprocesses validate their assigned rows in place and must never
    # re-route to the cluster (which would also import the env registry/yaml,
    # absent from minimal island envs). Skip cluster candidate routing entirely.
    if args.base_env_only:
        cluster_candidates: tuple[CatalogRow, ...] = ()
    else:
        cluster_candidates = _cluster_candidate_rows(
            rows,
            args.runner,
            ledger_db=args.verification_db,
            env_registry=args.env_registry,
        )
    completed_cluster_rows = _completed_cluster_rows(
        cluster_candidates,
        scope=args.scope,
        device=args.device,
        ledger_db=args.verification_db,
    )
    completed_cluster_stable_ids = {row.stable_id for row in completed_cluster_rows}
    cluster_rows = tuple(
        row for row in cluster_candidates if row.stable_id not in completed_cluster_stable_ids
    )
    cluster_candidate_stable_ids = {row.stable_id for row in cluster_candidates}
    local_rows = [row for row in rows if row.stable_id not in cluster_candidate_stable_ids]
    log_event(
        "runner_routing",
        runner=args.runner,
        cluster_rows=len(cluster_rows),
        cluster_completed_rows=len(completed_cluster_rows),
        local_rows=len(local_rows),
    )
    _append_cluster_rows_from_ledger(
        completed_cluster_rows, manifest_path, ledger_db=args.verification_db
    )
    _run_cluster_validation(cluster_rows, args, out_dir, manifest_path, smoke_settings)
    rows = local_rows
    if not args.base_env_only:
        registry = envs.load_registry(args.env_registry)
        assignments = envs.assign(rows, registry)
        island_rows: dict[str, list[CatalogRow]] = defaultdict(list)
        base_rows: list[CatalogRow] = []
        for row in rows:
            env_key = assignments[row.stable_id]
            if env_key == "base":
                base_rows.append(row)
            else:
                island_rows[env_key].append(row)
        for env_key, assigned_rows in sorted(island_rows.items()):
            stable_ids = [row.stable_id for row in assigned_rows]
            log_event("island_start", env_key=env_key, rows=len(stable_ids))
            result = envs.run_validate(
                env_key,
                stable_ids,
                registry,
                extra_args=_island_validator_args(args),
                worker_memory_cap_gb=args.worker_memory_cap_gb,
                jobs=args.jobs,
            )
            log_event(
                "island_done",
                env_key=env_key,
                rows=len(stable_ids),
                returncode=result.returncode,
                message=" | ".join((result.stdout + result.stderr).splitlines()[-5:]),
            )
        rows = base_rows

    # Phase 1: install dependencies per cluster (serial -- installs mutate the
    # shared interpreter/site-packages and must precede their rows). Clusters
    # whose dependencies are unavailable are recorded directly to the manifest.
    runnable: list[tuple[DependencyPlan, CatalogRow]] = []
    for plan, cluster_rows in group_by_dependency(rows):
        install_error = install_dependency_plan(plan, args)
        if install_error is not None:
            for row in cluster_rows:
                row_input_scale = _case_input_scale(row, smoke_settings, args.input_scale)
                result = ValidationResult(
                    row.name,
                    row.model_id,
                    "skipped:dependency_unavailable",
                    0,
                    False,
                    args.scope,
                    0.0,
                    plan.cluster_key,
                    install_error,
                    stable_id=row.stable_id,
                    recipe_revision_sha256=row.recipe_revision_sha256,
                    input_scale=row_input_scale,
                )
                append_validation_ledger(
                    row, result, args.device, row_input_scale, args.verification_db
                )
                append_manifest(
                    manifest_path,
                    result,
                )
            log_event(
                "cluster_skipped",
                cluster=plan.cluster_key,
                count=len(cluster_rows),
                error=install_error,
            )
            continue
        runnable.extend((plan, row) for row in cluster_rows)

    # Phase 2: validate runnable rows concurrently. Each model already runs in an
    # isolated child process (``validate_with_timeout``); threads here just
    # dispatch and await those subprocesses. The scheduler admits only work that
    # fits under the memory budget while still respecting the hard jobs cap. The
    # GPU semaphore caps in-flight jobs when a device that may use CUDA is
    # selected. The main thread does ALL manifest appends and disk bookkeeping
    # single-threaded as futures complete.
    jobs = max(1, args.jobs)
    use_gpu_cap = args.device in {"cuda", "auto"}
    gpu_jobs = max(1, args.gpu_jobs)
    effective_jobs = min(jobs, gpu_jobs) if use_gpu_cap else jobs
    gpu_semaphore = threading.Semaphore(gpu_jobs) if use_gpu_cap else None
    memory_budget_gb = _resolve_scheduler_memory_budget_gb(
        args.memory_budget_gb,
        args.worker_memory_cap_gb,
        effective_jobs,
    )
    memory_budget_mb = max(1, int(memory_budget_gb * MB_PER_GB))
    worker_cap_safe_budget_gb = (
        None
        if args.worker_memory_cap_gb is None
        else round(max(0.001, args.worker_memory_cap_gb) * effective_jobs, 3)
    )
    memory_floor_gb = _resolve_memory_floor_gb(args.memory_floor_gb)
    memory_floor_mb = max(1, int(memory_floor_gb * MB_PER_GB))
    ledger_memory_estimates = latest_peak_rss_estimates(args.verification_db)
    pending = _build_validation_work_items(runnable, ledger_memory_estimates)

    def process_one(plan: DependencyPlan, row: CatalogRow) -> tuple[ValidationResult, int]:
        """Validate one row in a worker thread and clean up its scratch state.

        Parameters
        ----------
        plan:
            Dependency plan for the row's cluster.
        row:
            Catalog row to validate.

        Returns
        -------
        tuple[ValidationResult, int]
            Validation result and the number of new cache entries removed.
        """

        cache_snapshots = [snapshot_cache(root) for root in CACHE_ROOTS]
        tmp_dir = out_dir / "_tmp" / f"{row.model_id:05d}_{safe_path_part(row.name)}"
        gate: ContextManager[Any] = gpu_semaphore if gpu_semaphore is not None else nullcontext()
        with gate:
            row_timeout = _case_timeout(row, smoke_settings, args.timeout_sec)
            row_input_scale = _case_input_scale(row, smoke_settings, args.input_scale)
            result = validate_with_timeout(
                row,
                args.dry_run,
                args.scope,
                args.device,
                row_timeout,
                tmp_dir=tmp_dir,
                worker_memory_cap_gb=args.worker_memory_cap_gb,
                input_scale=row_input_scale,
            )
            removed = 0 if args.keep_cache else cleanup_runtime(cache_snapshots, tmp_dir)
        return result, removed

    processed = 0
    total = len(runnable)
    log_event(
        "parallel_start",
        jobs=jobs,
        effective_jobs=effective_jobs,
        gpu_jobs=gpu_jobs if use_gpu_cap else None,
        memory_budget_gb=round(memory_budget_gb, 3),
        worker_memory_cap_gb=(
            None if args.worker_memory_cap_gb is None else round(args.worker_memory_cap_gb, 3)
        ),
        worker_cap_safe_budget_gb=worker_cap_safe_budget_gb,
        memory_floor_gb=round(memory_floor_gb, 3),
        default_unknown_memory_gb=round(DEFAULT_UNKNOWN_MEMORY_MB / MB_PER_GB, 3),
        heavy_unknown_memory_gb=round(HEAVY_UNKNOWN_MEMORY_MB / MB_PER_GB, 3),
        device=args.device,
        rows=total,
    )

    if total:
        try:
            assert_min_free(out_dir, args.min_free_gb)
        except RuntimeError:
            for snapshot in run_cache_snapshots:
                purge_new_cache_entries(snapshot)
            assert_min_free(out_dir, args.min_free_gb)

    with ThreadPoolExecutor(max_workers=effective_jobs) as executor:
        futures: dict[Future[tuple[ValidationResult, int]], ValidationWorkItem] = {}
        while pending or futures:
            actual_available_memory_mb = _actual_available_memory_mb()
            decision = _admit_memory_budgeted_items(
                pending=pending,
                in_flight_memory_mb=_in_flight_memory_mb(futures),
                in_flight_count=len(futures),
                budget_mb=memory_budget_mb,
                memory_floor_mb=memory_floor_mb,
                actual_available_memory_mb=actual_available_memory_mb,
                available_slots=effective_jobs - len(futures),
            )
            for item in decision.forced_oversized:
                log_event(
                    "memory_oversized_solo",
                    name=item.row.name,
                    estimated_peak_rss_gb=round(item.estimated_memory_mb / MB_PER_GB, 3),
                    memory_budget_gb=round(memory_budget_gb, 3),
                )
            for item in decision.admitted:
                plan = item.plan
                row = item.row
                before_free_gb = disk_free_gb(out_dir)
                in_flight_after_submit_mb = _in_flight_memory_mb(futures) + item.estimated_memory_mb
                log_event(
                    "model_start",
                    name=row.name,
                    cluster=plan.cluster_key,
                    free_gb=round(before_free_gb, 3),
                    estimated_peak_rss_gb=round(item.estimated_memory_mb / MB_PER_GB, 3),
                    estimate_source=item.estimate_source,
                    in_flight_estimated_gb=round(in_flight_after_submit_mb / MB_PER_GB, 3),
                    memory_budget_gb=round(memory_budget_gb, 3),
                )
                futures[executor.submit(process_one, plan, row)] = item
            if decision.throttled:
                log_event(
                    "memory_throttle",
                    reason=decision.throttle_reason,
                    pending=len(pending),
                    in_flight=len(futures),
                    in_flight_estimated_gb=round(_in_flight_memory_mb(futures) / MB_PER_GB, 3),
                    memory_budget_gb=round(memory_budget_gb, 3),
                    memory_floor_gb=round(memory_floor_gb, 3),
                    actual_available_gb=(
                        None
                        if actual_available_memory_mb is None
                        else round(actual_available_memory_mb / MB_PER_GB, 3)
                    ),
                    next_pending_estimated_gb=round(
                        min(item.estimated_memory_mb for item in pending) / MB_PER_GB,
                        3,
                    ),
                )
            if not futures:
                continue
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                item = futures.pop(future)
                row = item.row
                processed += 1
                result, removed = future.result()
                if result.peak_rss_mb is not None and result.peak_rss_mb > 0:
                    _refresh_pending_estimates(pending, row.stable_id, result.peak_rss_mb)
                append_validation_ledger(
                    row,
                    result,
                    args.device,
                    result.input_scale,
                    args.verification_db,
                )
                append_manifest(manifest_path, result)
                after_free_gb = disk_free_gb(out_dir)
                log_event(
                    "model_done",
                    index=processed,
                    total=total,
                    name=row.name,
                    status=result.status,
                    n_ops=result.n_ops,
                    cache_entries_removed=removed,
                    after_free_gb=round(after_free_gb, 3),
                    elapsed=round(result.elapsed, 3),
                    peak_rss_mb=result.peak_rss_mb,
                    error=result.error,
                )
                # Periodic disk-safety check: free space should not run dry as the
                # batch progresses.
                try:
                    assert_min_free(out_dir, args.min_free_gb)
                except RuntimeError:
                    for snapshot in run_cache_snapshots:
                        purge_new_cache_entries(snapshot)
                    assert_min_free(out_dir, args.min_free_gb)

    write_reports(
        out_dir,
        manifest_path,
        selected,
        catalog_db=args.db,
        no_build_catalog=args.no_build_catalog,
    )
    log_event(
        "validation_run_done",
        processed=processed,
        manifest=str(manifest_path),
        report=str(out_dir / REPORT_MD),
        summary=str(out_dir / SUMMARY_JSON),
    )
    return 0


def _input_scale_arg(value: str) -> float:
    """Parse and bound-check the ``--input-scale`` CLI value.

    Parameters
    ----------
    value:
        Raw CLI string.

    Returns
    -------
    float
        Scaling factor in the half-open range ``(0, 1]``.

    Raises
    ------
    argparse.ArgumentTypeError
        When the value is not a number in ``(0, 1]``.
    """

    try:
        scale = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"input-scale must be a number, got {value!r}") from error
    if not 0.0 < scale <= 1.0:
        raise argparse.ArgumentTypeError(f"input-scale must be in (0, 1], got {scale}")
    return scale


def _load_smoke_case_settings(path: Path | None) -> dict[str, SmokeCaseSettings]:
    """Load per-row timeout and input-scale overrides from a smoke manifest.

    Parameters
    ----------
    path:
        JSONL smoke-manifest path.

    Returns
    -------
    dict[str, SmokeCaseSettings]
        Settings keyed by stable ID.
    """

    if path is None:
        return {}
    settings: dict[str, SmokeCaseSettings] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            stable_id = str(payload["stable_id"])
            if stable_id in settings:
                raise ValueError(f"duplicate stable_id {stable_id!r} in {path}:{line_number}")
            settings[stable_id] = SmokeCaseSettings(
                timeout_sec=float(payload.get("timeout_sec", 240.0)),
                input_scale=float(payload.get("input_scale", 1.0)),
            )
    return settings


def _case_timeout(
    row: CatalogRow, settings: Mapping[str, SmokeCaseSettings], default_timeout: float
) -> float:
    """Return the validation timeout for a row.

    Parameters
    ----------
    row:
        Catalog row.
    settings:
        Per-row smoke settings.
    default_timeout:
        Default timeout when the row has no override.

    Returns
    -------
    float
        Timeout in seconds.
    """

    case = settings.get(row.stable_id)
    return default_timeout if case is None else case.timeout_sec


def _case_input_scale(
    row: CatalogRow, settings: Mapping[str, SmokeCaseSettings], default_input_scale: float
) -> float:
    """Return the input scale for a row.

    Parameters
    ----------
    row:
        Catalog row.
    settings:
        Per-row smoke settings.
    default_input_scale:
        Default input scale when the row has no override.

    Returns
    -------
    float
        Input scale.
    """

    case = settings.get(row.stable_id)
    return default_input_scale if case is None else case.input_scale


def build_parser() -> argparse.ArgumentParser:
    """Build the validator CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, help="process the first N rows after filters")
    parser.add_argument("--family")
    parser.add_argument("--domain")
    parser.add_argument("--zoo")
    parser.add_argument("--name", action="append", help="case-insensitive model-name substring")
    parser.add_argument("--model-id", action="append", type=int, help="exact catalog model id")
    parser.add_argument("--stable-ids", nargs="+", help="exact stable IDs to validate")
    parser.add_argument("--verified-only", action="store_true")
    parser.add_argument("--featured-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--since", type=int, help="only process rows with model_id greater than this"
    )
    parser.add_argument(
        "--scope",
        choices=("forward", "forward+backward"),
        default="forward",
        help="validation scope",
    )
    parser.add_argument(
        "--revalidate-failed", action="store_true", help="retry non-validated manifest rows"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="number of models to validate concurrently (each in its own subprocess)",
    )
    parser.add_argument(
        "--gpu-jobs",
        type=int,
        default=4,
        help="max concurrent in-flight jobs when --device is cuda/auto (GPU OOM guard)",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=None,
        help=(
            "max estimated in-flight validation RSS in GB; default auto-detects "
            "70%% of currently available RAM"
        ),
    )
    parser.add_argument(
        "--memory-floor-gb",
        type=float,
        default=None,
        help=(
            "minimum actual free RAM required before admitting another model; "
            f"default {DEFAULT_MEMORY_FLOOR_GB:g} GB, and each model also requires "
            "headroom at least equal to its estimate"
        ),
    )
    parser.add_argument(
        "--worker-memory-cap-gb",
        type=float,
        default=None,
        help="per-worker RSS cap in GB; default off",
    )
    parser.add_argument(
        "--runner",
        choices=RUNNER_CHOICES,
        default="auto",
        help=(
            "validation runner policy: auto routes RAM giants to the cluster, "
            "local forces legacy local validation, cluster sends non-native-crash rows "
            "to the cluster"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--db", type=Path, default=Path(__file__).resolve().parent / "data" / "catalog.db"
    )
    parser.add_argument(
        "--no-build-catalog",
        action="store_true",
        help="open --db read-only and fail if the catalog is missing",
    )
    parser.add_argument("--env-registry", type=Path, default=envs.REGISTRY_PATH)
    parser.add_argument(
        "--verification-db",
        type=Path,
        help="verification ledger path; also exported to child workers",
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        help="JSONL smoke manifest carrying per-row timeout/input-scale settings",
    )
    parser.add_argument("--base-env-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--min-free-gb", type=float, default=15.0)
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-models", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument(
        "--input-scale",
        type=_input_scale_arg,
        default=1.0,
        help=(
            "spatial down-scaling factor in (0, 1] for the example input "
            "(default 1.0 = full resolution); values < 1.0 validate too-large "
            "models at reduced input size without changing the recipe identity"
        ),
    )
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--install-timeout", type=float, default=600.0)
    parser.add_argument(
        "--pip-args", action="append", default=[], help="extra argument for pip install"
    )
    parser.add_argument("--install-deps", dest="install_deps", action="store_true", default=True)
    parser.add_argument("--no-install-deps", dest="install_deps", action="store_false")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--worker-row-json", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validator CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verification_db is not None:
        os.environ[ENV_VERIFICATION_DB] = str(args.verification_db)
    # Pin BLAS/OMP threads to 1 so concurrent validate workers don't oversubscribe the CPU (see generate_menagerie).
    for _thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_thread_var, "1")
    if args.worker_row_json:
        row = catalog_row_from_payload(json.loads(args.worker_row_json))
        worker_start = time.monotonic()
        _start_worker_memory_monitor(
            row,
            args.scope,
            args.worker_memory_cap_gb,
            worker_start,
            args.input_scale,
        )
        worker_test_allocations = _run_worker_test_allocation_from_env()
        try:
            result = validate_one(row, args.dry_run, args.scope, args.device, args.input_scale)
        except Exception as error:
            plan = dependency_plan(row)
            result = ValidationResult(
                row.name,
                row.model_id,
                "failed:exception",
                0,
                False,
                args.scope,
                0.0,
                plan.cluster_key,
                f"{error!r}\n{traceback.format_exc(limit=8)}",
                stable_id=row.stable_id,
                recipe_revision_sha256=row.recipe_revision_sha256,
                input_scale=args.input_scale,
            )
        result = ValidationResult(
            result.name,
            result.model_id,
            result.status,
            result.n_ops,
            result.validate_metadata_ok,
            result.scope,
            result.elapsed,
            result.dependency_cluster,
            result.error,
            result.graph_shape_hash,
            result.stable_id,
            result.recipe_revision_sha256,
            _peak_rss_mb(),
            result.input_scale,
        )
        if worker_test_allocations:
            worker_test_allocations.clear()
        _emit_worker_result(result)
        return 0
    try:
        return run(args)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
