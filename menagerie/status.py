"""Honest status funnel for the TorchLens menagerie."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from menagerie import envs
from menagerie.catalog import (
    CATALOG_DB,
    DEFERRED_JSONL,
    SOURCE_JSONL,
    CatalogRow,
    ensure_catalog,
    load_rows,
)
from menagerie.ledger import (
    VERIFICATION_DB,
    VerificationTarget,
    Scope,
    _load_verification_targets,
    connect as connect_ledger,
    torchlens_source_hash,
    verified_count,
)
from menagerie.provenance import SweepProvenance, read_sweeps
from menagerie.schema import (
    VerificationExpectation,
    load_jsonl,
    recipe_is_quarantined,
    recipe_uses_code_execution,
)
from menagerie.tools.distinct_report import (
    DistinctArchitectureReport,
    build_distinct_report,
    current_revisions_from_catalog,
    verified_hashes,
)


DEFAULT_RENDER_MANIFEST = Path("/tmp/torchlens_menagerie_gallery/manifest.tsv")
DEFERRAL_REASON_RE = re.compile(r"deferral_reason=([^;]+)")
CRAWL_HISTORY = Path(__file__).resolve().parent / "data" / "crawl_history.json"
DEFAULT_COMPLETENESS_ROOT = Path("/tmp/torchlens_menagerie_status")
CATALOG_SNAPSHOT_COLUMNS = (
    "stable_id",
    "recipe_revision_sha256",
    "name",
    "zoo",
    "source",
    "assigned_env",
)
TERMINAL_STATUSES = (
    "passed",
    "failed",
    "timeout",
    "not_applicable",
    "install_failed",
    "env_unavailable",
    "oom",
    "native_crash",
    "killed",
)
KNOWN_JUSTIFIED_MANIFEST_STATUSES = (
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
)
KNOWN_JUSTIFIED_SKIP_STATUSES = tuple(
    status for status in KNOWN_JUSTIFIED_MANIFEST_STATUSES if status.startswith("skipped:")
)


@dataclass(frozen=True)
class CatalogStatus:
    """Status metadata for one catalog model.

    Parameters
    ----------
    stable_id:
        Durable catalog identity.
    input_is_real:
        Whether the model is traced with a real input object.
    verification_expectation:
        Expected verification posture.
    quarantine:
        Whether the recipe is quarantined executable code.
    code_execution:
        Whether the recipe is built through Python code execution.
    deferred_reason:
        Human-readable deferral reason, when available.
    """

    stable_id: str
    input_is_real: bool
    verification_expectation: str
    quarantine: bool
    code_execution: bool
    deferred_reason: str = ""


@dataclass(frozen=True)
class FunnelStatus:
    """Honest menagerie reporting funnel.

    Parameters
    ----------
    catalog_db:
        Catalog database path used for the report.
    ledger_db:
        Verification ledger path used for the report.
    torchlens_version:
        TorchLens version used by the verified predicate.
    total_catalog_models:
        Total catalog model count.
    expected_models:
        Models expected to pass verification, excluding deferred rows.
    rendered_models:
        Models with rendered manifest entries.
    verified_models:
        Models with a current-version, current-recipe, full-predicate forward pass.
    headline_verified_real_input:
        Verified models traced with real input. This is the honest headline tier.
    verified_wrapper_input:
        Verified models using wrapper or sentinel input.
    deferred_models:
        Deferred model count.
    deferred_fraction:
        Deferred model fraction of total catalog models.
    deferred_by_reason:
        Deferred counts by reason.
    quarantined_models:
        Quarantined executable-recipe count.
    quarantine_fraction:
        Quarantined model fraction of total catalog models.
    code_execution_models:
        Models built through exec-string, expression, or statement recipes.
    code_execution_fraction:
        Code-execution model fraction of total catalog models.
    distinct:
        Distinct architecture report using shape-blind graph-shape hashes.
    render_manifest:
        Render manifest path consulted for render coverage.
    render_manifest_available:
        Whether the render manifest existed.
    """

    catalog_db: str
    ledger_db: str
    torchlens_version: str
    total_catalog_models: int
    expected_models: int
    rendered_models: int
    verified_models: int
    headline_verified_real_input: int
    verified_wrapper_input: int
    deferred_models: int
    deferred_fraction: float
    deferred_by_reason: dict[str, int]
    quarantined_models: int
    quarantine_fraction: float
    code_execution_models: int
    code_execution_fraction: float
    distinct: DistinctArchitectureReport
    render_manifest: str
    render_manifest_available: bool


@dataclass(frozen=True)
class ProvenanceStatus:
    """Structured provenance dashboard data.

    Parameters
    ----------
    ledger_db:
        Verification database path used for provenance.
    last_exhaustive_crawl:
        Last exhaustive crawl date from crawl history.
    sweep_count:
        Number of structured sweep provenance rows.
    sweeps:
        Structured sweep provenance rows.
    model_wave_counts:
        Catalog model counts by ``added_wave``.
    models_with_known_sweep:
        Models whose ``added_wave`` has a matching provenance row.
    models_without_wave:
        Models without an ``added_wave`` value.
    """

    ledger_db: str
    last_exhaustive_crawl: str
    sweep_count: int
    sweeps: list[SweepProvenance]
    model_wave_counts: dict[str, int]
    models_with_known_sweep: int
    models_without_wave: int


@dataclass(frozen=True)
class CompletenessIssue:
    """One catalog row missing a current terminal verification row.

    Parameters
    ----------
    stable_id:
        Durable catalog identity.
    recipe_revision_sha256:
        Current catalog recipe revision.
    name:
        Catalog model name.
    zoo:
        Catalog source zoo.
    source:
        Catalog source stream.
    assigned_env:
        Validation environment assigned to this catalog row.
    issue:
        Audit issue class.
    stale_recipe_revision_sha256:
        Latest stale terminal recipe revision, when available.
    stale_status:
        Latest stale terminal status, when available.
    """

    stable_id: str
    recipe_revision_sha256: str
    name: str
    zoo: str
    source: str
    assigned_env: str
    issue: str
    stale_recipe_revision_sha256: str = ""
    stale_status: str = ""


@dataclass(frozen=True)
class CompletenessStatus:
    """Completeness audit result.

    Parameters
    ----------
    catalog_db:
        Catalog database path used for the audit.
    ledger_db:
        Verification ledger path used for the audit.
    torchlens_version:
        TorchLens version used by the verified predicate.
    run_dir:
        Directory containing audit artifacts.
    catalog_snapshot:
        Catalog snapshot TSV artifact.
    total_catalog_models:
        Snapshot catalog row count.
    terminal_current_recipe_models:
        Catalog rows with a terminal ledger row at the current recipe revision.
    missing_terminal_models:
        Rows with no terminal ledger history.
    stale_recipe_models:
        Rows with terminal history only for stale recipe revisions.
    issues:
        Missing/stale catalog rows.
    terminal_by_status:
        Terminal current-recipe rows by ledger status.
    funnel:
        Honest status funnel, including the verified headline.
    """

    catalog_db: str
    ledger_db: str
    torchlens_version: str
    run_dir: str
    catalog_snapshot: str
    total_catalog_models: int
    terminal_current_recipe_models: int
    missing_terminal_models: int
    stale_recipe_models: int
    issues: list[CompletenessIssue]
    terminal_by_status: dict[str, int]
    funnel: FunnelStatus


@dataclass(frozen=True)
class FailureClassRecord:
    """Summary for one recurring validation failure class.

    Parameters
    ----------
    failure_class:
        Stable class label, usually the manifest status.
    count:
        Number of rows in the class.
    stable_ids:
        Stable IDs represented by the class, capped for report readability.
    sample_error:
        First non-empty error message seen for the class.
    kb_key:
        Machine-readable knowledge-base key for matching recurring failures.
    """

    failure_class: str
    count: int
    stable_ids: list[str]
    sample_error: str = ""
    kb_key: str = ""


@dataclass(frozen=True)
class RunHealth:
    """Run-all health signal derived from validation artifacts.

    Parameters
    ----------
    ok:
        Whether the run is clean enough for a zero exit.
    unexpected_failures:
        Count of unmasked failed rows.
    unexpected_skips:
        Count of skip rows outside the closed taxonomy.
    unknown_statuses:
        Manifest statuses not present in the closed taxonomy.
    worker_deaths:
        Count of killed/OOM/native-crash rows.
    failure_rate:
        Failed-row fraction across the manifest.
    alarms:
        Loud operator-facing health alarms.
    failure_classes:
        Structured recurring failure-class summary.
    """

    ok: bool
    unexpected_failures: int
    unexpected_skips: int
    unknown_statuses: list[str]
    worker_deaths: int
    failure_rate: float
    alarms: list[str]
    failure_classes: list[FailureClassRecord]


def classify_failure_records(
    rows: Sequence[Mapping[str, str]], *, sample_limit: int = 8
) -> list[FailureClassRecord]:
    """Classify failed validation rows into recurring failure records.

    Parameters
    ----------
    rows:
        Validation manifest rows.
    sample_limit:
        Maximum number of stable IDs retained per class.

    Returns
    -------
    list[FailureClassRecord]
        Failure classes sorted by frequency and class name.
    """

    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        status = str(row.get("status", ""))
        if not status.startswith("failed:"):
            continue
        grouped.setdefault(status, []).append(row)
    records: list[FailureClassRecord] = []
    for failure_class, class_rows in grouped.items():
        sample_error = next(
            (str(row.get("error", "")) for row in class_rows if row.get("error")), ""
        )
        stable_ids = [str(row.get("stable_id", "")) for row in class_rows[:sample_limit]]
        records.append(
            FailureClassRecord(
                failure_class=failure_class,
                count=len(class_rows),
                stable_ids=stable_ids,
                sample_error=sample_error,
                kb_key=f"{failure_class}:{sample_error[:96]}",
            )
        )
    return sorted(records, key=lambda record: (-record.count, record.failure_class))


def build_run_health(
    rows: Sequence[Mapping[str, str]],
    *,
    expected_status_by_id: Mapping[str, str] | None = None,
    failure_rate_alarm_threshold: float = 0.0,
    pending_cluster_tasks: Mapping[str, object] | None = None,
) -> RunHealth:
    """Build the fail-loud health block for a run-all report.

    Parameters
    ----------
    rows:
        Validation manifest rows.
    expected_status_by_id:
        Optional smoke manifest expectations. Rows matching an expected failed
        status are not counted as unexpected failures.
    failure_rate_alarm_threshold:
        Failure-rate threshold for a loud alarm. ``0`` means any unexpected
        failure trips the alarm.
    pending_cluster_tasks:
        Pending cluster task block from the validation summary.

    Returns
    -------
    RunHealth
        Structured run-health result.
    """

    expected = expected_status_by_id or {}
    unknown = sorted(
        {
            str(row.get("status", ""))
            for row in rows
            if str(row.get("status", "")) not in KNOWN_JUSTIFIED_MANIFEST_STATUSES
        }
    )
    unexpected_failures = 0
    unexpected_skips = 0
    worker_deaths = 0
    for row in rows:
        stable_id = str(row.get("stable_id", ""))
        status = str(row.get("status", ""))
        expected_status = expected.get(stable_id)
        if (
            status in {"failed:killed", "failed:oom", "failed:native_crash"}
            and expected_status != status
        ):
            worker_deaths += 1
        if status.startswith("failed:") and expected_status != status:
            unexpected_failures += 1
        if status.startswith("skipped:") and status not in KNOWN_JUSTIFIED_SKIP_STATUSES:
            unexpected_skips += 1
    total = len(rows)
    failure_rate = unexpected_failures / total if total else 0.0
    alarms: list[str] = []
    if unknown:
        alarms.append(f"UNKNOWN_STATUS: {', '.join(unknown)}")
    if unexpected_skips:
        alarms.append(f"UNMAPPED_SKIP: count={unexpected_skips}")
    if worker_deaths:
        alarms.append(f"WORKER_DEATH: count={worker_deaths}")
    if pending_cluster_tasks:
        alarms.append(f"CLUSTER_TASKS_PENDING: count={pending_cluster_tasks.get('count', 0)}")
    if unexpected_failures and failure_rate >= failure_rate_alarm_threshold:
        alarms.append(
            f"FAILURE_RATE: {failure_rate:.1%} exceeds threshold {failure_rate_alarm_threshold:.1%}"
        )
    failure_classes = classify_failure_records(rows)
    ok = not alarms and unexpected_failures == 0
    return RunHealth(
        ok=ok,
        unexpected_failures=unexpected_failures,
        unexpected_skips=unexpected_skips,
        unknown_statuses=unknown,
        worker_deaths=worker_deaths,
        failure_rate=failure_rate,
        alarms=alarms,
        failure_classes=failure_classes,
    )


def _catalog_rows(catalog_db: Path) -> list[sqlite3.Row]:
    """Load catalog rows needed for status reporting.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.

    Returns
    -------
    list[sqlite3.Row]
        Catalog rows.
    """

    ensure_catalog(catalog_db)
    conn = sqlite3.connect(catalog_db)
    conn.row_factory = sqlite3.Row
    with conn:
        return list(
            conn.execute(
                """
                SELECT *
                FROM models
                ORDER BY model_id
                """
            )
        )


def _jsonl_status_by_key(
    source_jsonl: Path = SOURCE_JSONL, deferred_jsonl: Path = DEFERRED_JSONL
) -> dict[tuple[str, str, str], CatalogStatus]:
    """Load status metadata from typed JSONL sources keyed by natural key.

    Parameters
    ----------
    source_jsonl:
        Forward-required catalog JSONL path.
    deferred_jsonl:
        Deferred catalog JSONL path.

    Returns
    -------
    dict[tuple[str, str, str], CatalogStatus]
        Status metadata keyed by ``(name, zoo, variant)``. Stable IDs are blank until joined
        against the catalog DB.
    """

    records = load_jsonl(source_jsonl)
    if deferred_jsonl.exists():
        records.extend(load_jsonl(deferred_jsonl))
    statuses: dict[tuple[str, str, str], CatalogStatus] = {}
    for record in records:
        reason = record.deferral.reason if record.deferral is not None else ""
        statuses[(record.name, record.zoo, record.variant)] = CatalogStatus(
            stable_id="",
            input_is_real=record.input_is_real,
            verification_expectation=record.verification_expectation.value,
            quarantine=recipe_is_quarantined(record.recipe),
            code_execution=recipe_uses_code_execution(record.recipe),
            deferred_reason=reason,
        )
    return statuses


def catalog_statuses(catalog_db: Path = CATALOG_DB) -> dict[str, CatalogStatus]:
    """Load honest-reporting status metadata keyed by stable ID.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.

    Returns
    -------
    dict[str, CatalogStatus]
        Status metadata keyed by stable model ID.
    """

    rows = _catalog_rows(catalog_db)
    has_db_status_columns = bool(rows) and {
        "input_is_real",
        "verification_expectation",
        "quarantine",
    }.issubset(set(rows[0].keys()))
    jsonl_by_key = {} if has_db_status_columns else _jsonl_status_by_key()
    statuses: dict[str, CatalogStatus] = {}
    for row in rows:
        key = (str(row["name"]), str(row["zoo"]), str(row["variant"]))
        source_status = jsonl_by_key.get(key)
        db_status = _status_from_db_row(row)
        notes_status = _status_from_notes(str(row["stable_id"]), str(row["notes"]))
        if has_db_status_columns or source_status is None:
            statuses[str(row["stable_id"])] = _merge_status(db_status, notes_status)
            continue
        verification_expectation = source_status.verification_expectation
        deferred_reason = source_status.deferred_reason
        if notes_status.verification_expectation == VerificationExpectation.deferred.value:
            verification_expectation = notes_status.verification_expectation
            deferred_reason = notes_status.deferred_reason
        statuses[str(row["stable_id"])] = CatalogStatus(
            stable_id=str(row["stable_id"]),
            input_is_real=source_status.input_is_real,
            verification_expectation=verification_expectation,
            quarantine=source_status.quarantine,
            code_execution=source_status.code_execution,
            deferred_reason=deferred_reason,
        )
    return statuses


def _status_from_db_row(row: sqlite3.Row) -> CatalogStatus:
    """Build status metadata from catalog DB columns when present.

    Parameters
    ----------
    row:
        Catalog database row.

    Returns
    -------
    CatalogStatus
        Status metadata.
    """

    keys = set(row.keys())
    return CatalogStatus(
        stable_id=str(row["stable_id"]),
        input_is_real=bool(row["input_is_real"]) if "input_is_real" in keys else True,
        verification_expectation=(
            str(row["verification_expectation"])
            if "verification_expectation" in keys
            else VerificationExpectation.forward_required.value
        ),
        quarantine=bool(row["quarantine"]) if "quarantine" in keys else False,
        code_execution=(
            False
            if str(row["source"]) == "classics"
            else _constructor_uses_code_execution(str(row["constructor_call"]))
        ),
        deferred_reason="",
    )


def _merge_status(primary: CatalogStatus, notes_status: CatalogStatus) -> CatalogStatus:
    """Merge DB metadata with deferred markers recovered from notes.

    Parameters
    ----------
    primary:
        Primary status metadata.
    notes_status:
        Notes-derived fallback status.

    Returns
    -------
    CatalogStatus
        Merged status metadata.
    """

    if notes_status.verification_expectation != VerificationExpectation.deferred.value:
        return primary
    return CatalogStatus(
        stable_id=primary.stable_id,
        input_is_real=primary.input_is_real,
        verification_expectation=notes_status.verification_expectation,
        quarantine=primary.quarantine,
        code_execution=primary.code_execution,
        deferred_reason=notes_status.deferred_reason,
    )


def _status_from_notes(stable_id: str, notes: str) -> CatalogStatus:
    """Build fallback status metadata from legacy catalog notes.

    Parameters
    ----------
    stable_id:
        Stable catalog identity.
    notes:
        Catalog notes text.

    Returns
    -------
    CatalogStatus
        Conservative status metadata inferred from notes.
    """

    if "verification_expectation=deferred" not in notes:
        return CatalogStatus(
            stable_id=stable_id,
            input_is_real=True,
            verification_expectation=VerificationExpectation.forward_required.value,
            quarantine=False,
            code_execution=False,
            deferred_reason="",
        )
    match = DEFERRAL_REASON_RE.search(notes)
    return CatalogStatus(
        stable_id=stable_id,
        input_is_real=True,
        verification_expectation=VerificationExpectation.deferred.value,
        quarantine=False,
        code_execution=False,
        deferred_reason=match.group(1).strip() if match else "deferred",
    )


def _constructor_uses_code_execution(constructor_call: str) -> bool:
    """Infer code-execution recipe status from a catalog constructor string.

    Parameters
    ----------
    constructor_call:
        Catalog constructor call.

    Returns
    -------
    bool
        Whether the constructor corresponds to exec-string, expression, or statement recipe forms.
    """

    code = constructor_call.strip()
    if "\n" in code or "exec(" in code:
        return True
    if ";" in code or code.startswith(("import ", "from ")):
        return True
    try:
        parsed = ast.parse(code, mode="eval")
    except SyntaxError:
        return True
    return not isinstance(parsed.body, ast.Call)


def rendered_stable_ids(manifest_path: Path = DEFAULT_RENDER_MANIFEST) -> set[str]:
    """Load stable IDs with rendered manifest rows.

    Parameters
    ----------
    manifest_path:
        Render manifest path.

    Returns
    -------
    set[str]
        Stable IDs whose latest manifest row is rendered.
    """

    if not manifest_path.exists():
        return set()
    latest: dict[str, dict[str, str]] = {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            stable_id = row.get("stable_id", "")
            if stable_id:
                latest[stable_id] = row
    return {
        stable_id
        for stable_id, row in latest.items()
        if row.get("status") == "rendered" or bool(row.get("render_path"))
    }


def build_status(
    catalog_db: Path = CATALOG_DB,
    ledger_db: Path = VERIFICATION_DB,
    torchlens_version: str | None = None,
    render_manifest: Path = DEFAULT_RENDER_MANIFEST,
    source_hash: str | None = None,
    device_requested: str = "cpu",
    env_registry: envs.EnvRegistry | None = None,
) -> FunnelStatus:
    """Build the honest menagerie status funnel.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.
    ledger_db:
        Verification ledger SQLite database path.
    torchlens_version:
        TorchLens version for the verified predicate. Defaults to installed TorchLens.
    render_manifest:
        Render manifest path for render coverage.
    source_hash:
        TorchLens source hash to require. Defaults to the current checkout hash.
    device_requested:
        Requested device policy to require.
    env_registry:
        Optional loaded validation environment registry.

    Returns
    -------
    FunnelStatus
        Honest reporting funnel.
    """

    if torchlens_version is None:
        import torchlens as tl

        torchlens_version = str(tl.__version__)
    statuses = catalog_statuses(catalog_db)
    total = len(statuses)
    deferred_statuses = [
        status
        for status in statuses.values()
        if status.verification_expectation == VerificationExpectation.deferred.value
    ]
    quarantined = [status for status in statuses.values() if status.quarantine]
    code_execution = [status for status in statuses.values() if status.code_execution]
    current_revisions = current_revisions_from_catalog(catalog_db)
    catalog_rows = load_rows(db_path=catalog_db)
    registry = env_registry or envs.load_registry()
    assignments = _assignment_map(catalog_rows, registry)
    current_targets = _verification_targets_for_rows(
        catalog_rows,
        assignments,
        source_hash or torchlens_source_hash(),
        device_requested,
        "forward",
    )
    with connect_ledger(ledger_db) as ledger_conn:
        verified_by_stable_id = verified_hashes(
            ledger_conn, torchlens_version, current_revisions, current_targets
        )
        verified_total = verified_count(
            ledger_conn, torchlens_version, current_revisions, current_targets
        )
    verified_ids = set(verified_by_stable_id)
    real_input_verified = sum(
        1
        for stable_id in verified_ids
        if statuses.get(stable_id, _missing_status(stable_id)).input_is_real
    )
    wrapper_verified = len(verified_ids) - real_input_verified
    deferred_by_reason = Counter(
        status.deferred_reason or "deferred" for status in deferred_statuses
    )
    distinct = build_distinct_report(
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        torchlens_version=torchlens_version,
        manifest_paths=[render_manifest] if render_manifest.exists() else (),
        current_targets=current_targets,
    )
    return FunnelStatus(
        catalog_db=str(catalog_db),
        ledger_db=str(ledger_db),
        torchlens_version=torchlens_version,
        total_catalog_models=total,
        expected_models=total - len(deferred_statuses),
        rendered_models=len(rendered_stable_ids(render_manifest)),
        verified_models=verified_total,
        headline_verified_real_input=real_input_verified,
        verified_wrapper_input=wrapper_verified,
        deferred_models=len(deferred_statuses),
        deferred_fraction=_fraction(len(deferred_statuses), total),
        deferred_by_reason=dict(sorted(deferred_by_reason.items())),
        quarantined_models=len(quarantined),
        quarantine_fraction=_fraction(len(quarantined), total),
        code_execution_models=len(code_execution),
        code_execution_fraction=_fraction(len(code_execution), total),
        distinct=distinct,
        render_manifest=str(render_manifest),
        render_manifest_available=render_manifest.exists(),
    )


def _default_completeness_run_dir() -> Path:
    """Return a timestamped completeness audit run directory.

    Returns
    -------
    Path
        Default run directory.
    """

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_COMPLETENESS_ROOT / timestamp


def _assignment_map(rows: Sequence[CatalogRow], registry: envs.EnvRegistry) -> dict[str, str]:
    """Return validation environment assignments keyed by stable ID.

    Parameters
    ----------
    rows:
        Catalog rows to assign.
    registry:
        Loaded environment registry.

    Returns
    -------
    dict[str, str]
        Assigned environment key by stable ID.
    """

    return envs.assign(rows, registry)


def _verification_targets_for_rows(
    rows: Sequence[CatalogRow],
    assignments: Mapping[str, str],
    source_hash: str,
    device_requested: str,
    scope: str,
) -> dict[str, VerificationTarget]:
    """Build current full-identity verification targets for catalog rows.

    Parameters
    ----------
    rows:
        Catalog rows to target.
    assignments:
        Assigned environment key by stable ID.
    source_hash:
        Current TorchLens source hash.
    device_requested:
        Requested device policy.
    scope:
        Verification scope.

    Returns
    -------
    dict[str, VerificationTarget]
        Verification targets keyed by stable ID.
    """

    targets: dict[str, VerificationTarget] = {}
    for row in rows:
        env_key = assignments[row.stable_id]
        lock_hash = envs.current_lock_hash(env_key)
        targets[row.stable_id] = VerificationTarget(
            recipe_revision_sha256=row.recipe_revision_sha256,
            torchlens_source_hash=source_hash,
            env_hash=envs.current_env_hash(env_key, lock_hash),
            lock_hash=lock_hash,
            device_requested=device_requested,
            scope=cast(Scope, scope),
        )
    return targets


def _write_catalog_snapshot(
    rows: list[sqlite3.Row], assignments: Mapping[str, str], run_dir: Path
) -> Path:
    """Write the current catalog snapshot artifact.

    Parameters
    ----------
    rows:
        Current catalog rows.
    assignments:
        Assigned environment key by stable ID.
    run_dir:
        Completeness audit run directory.

    Returns
    -------
    Path
        Snapshot TSV path.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = run_dir / "catalog_snapshot.tsv"
    with snapshot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(CATALOG_SNAPSHOT_COLUMNS)
        for row in rows:
            writer.writerow(
                (
                    row["stable_id"],
                    row["recipe_revision_sha256"],
                    row["name"],
                    row["zoo"],
                    row["source"],
                    assignments[str(row["stable_id"])],
                )
            )
    with snapshot_path.open(encoding="utf-8") as handle:
        written = sum(1 for _line in handle) - 1
    if written != len(rows):
        raise RuntimeError(
            f"catalog snapshot row-count mismatch: wrote {written}, expected {len(rows)}"
        )
    return snapshot_path


def _load_catalog_snapshot(
    conn: sqlite3.Connection, rows: list[sqlite3.Row], assignments: Mapping[str, str]
) -> None:
    """Load catalog rows into a temporary audit table.

    Parameters
    ----------
    conn:
        Ledger SQLite connection.
    rows:
        Current catalog rows.
    assignments:
        Assigned environment key by stable ID.
    """

    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS catalog_snapshot(
            stable_id TEXT PRIMARY KEY,
            recipe_revision_sha256 TEXT NOT NULL,
            name TEXT NOT NULL,
            zoo TEXT NOT NULL,
            source TEXT NOT NULL,
            assigned_env TEXT NOT NULL
        )
        """
    )
    conn.execute("DELETE FROM catalog_snapshot")
    conn.executemany(
        """
        INSERT INTO catalog_snapshot(
            stable_id,
            recipe_revision_sha256,
            name,
            zoo,
            source,
            assigned_env
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            (
                str(row["stable_id"]),
                str(row["recipe_revision_sha256"]),
                str(row["name"]),
                str(row["zoo"]),
                str(row["source"]),
                assignments[str(row["stable_id"])],
            )
            for row in rows
        ),
    )
    snapshot_count = conn.execute("SELECT COUNT(*) FROM catalog_snapshot").fetchone()[0]
    if int(snapshot_count) != len(rows):
        raise RuntimeError(
            f"catalog_snapshot temp row-count mismatch: loaded {snapshot_count}, expected {len(rows)}"
        )


def _terminal_status_placeholders() -> str:
    """Return SQL placeholders for terminal statuses.

    Returns
    -------
    str
        Comma-separated SQLite placeholders.
    """

    return ",".join("?" for _status in TERMINAL_STATUSES)


def _completeness_issues(conn: sqlite3.Connection) -> list[CompletenessIssue]:
    """Return catalog rows lacking a current-recipe terminal row.

    Parameters
    ----------
    conn:
        Ledger SQLite connection with ``catalog_snapshot`` loaded.

    Returns
    -------
    list[CompletenessIssue]
        Missing or stale rows.
    """

    placeholders = _terminal_status_placeholders()
    rows = conn.execute(
        f"""
        WITH current_terminal AS (
            SELECT
                current_verification.stable_id,
                current_verification.recipe_revision_sha256,
                current_verification.status
            FROM current_verification
            JOIN temp_current_verification_targets AS target
              ON target.stable_id = current_verification.stable_id
             AND target.recipe_revision_sha256 = current_verification.recipe_revision_sha256
             AND target.torchlens_source_hash = current_verification.torchlens_source_hash
             AND target.env_hash = current_verification.env_hash
             AND target.lock_hash = current_verification.lock_hash
             AND target.device_requested = current_verification.device_requested
             AND target.scope = current_verification.scope
            WHERE current_verification.status IN ({placeholders})
              AND current_verification.torchlens_source_hash != 'legacy-unknown'
              AND current_verification.lock_hash != 'legacy-unknown'
        ),
        latest_run AS (
            SELECT stable_id, recipe_revision_sha256, status
            FROM current_verification
        )
        SELECT
            catalog_snapshot.stable_id,
            catalog_snapshot.recipe_revision_sha256,
            catalog_snapshot.name,
            catalog_snapshot.zoo,
            catalog_snapshot.source,
            catalog_snapshot.assigned_env,
            CASE
                WHEN latest_run.stable_id IS NULL THEN 'missing_terminal'
                WHEN latest_run.recipe_revision_sha256 = catalog_snapshot.recipe_revision_sha256
                    THEN 'stale_identity_tuple'
                ELSE 'stale_recipe_revision'
            END AS issue,
            COALESCE(latest_run.recipe_revision_sha256, '') AS stale_recipe_revision_sha256,
            COALESCE(latest_run.status, '') AS stale_status
        FROM catalog_snapshot
        LEFT JOIN current_terminal
          ON current_terminal.stable_id = catalog_snapshot.stable_id
         AND current_terminal.recipe_revision_sha256 = catalog_snapshot.recipe_revision_sha256
        LEFT JOIN latest_run
          ON latest_run.stable_id = catalog_snapshot.stable_id
        WHERE current_terminal.stable_id IS NULL
        ORDER BY catalog_snapshot.stable_id
        """,
        TERMINAL_STATUSES,
    ).fetchall()
    return [
        CompletenessIssue(
            stable_id=str(row["stable_id"]),
            recipe_revision_sha256=str(row["recipe_revision_sha256"]),
            name=str(row["name"]),
            zoo=str(row["zoo"]),
            source=str(row["source"]),
            assigned_env=str(row["assigned_env"]),
            issue=str(row["issue"]),
            stale_recipe_revision_sha256=str(row["stale_recipe_revision_sha256"]),
            stale_status=str(row["stale_status"]),
        )
        for row in rows
    ]


def _terminal_by_status(conn: sqlite3.Connection) -> dict[str, int]:
    """Return current-recipe terminal row counts by status.

    Parameters
    ----------
    conn:
        Ledger SQLite connection with ``catalog_snapshot`` loaded.

    Returns
    -------
    dict[str, int]
        Status counts.
    """

    placeholders = _terminal_status_placeholders()
    rows = conn.execute(
        f"""
        WITH current_terminal AS (
            SELECT
                current_verification.stable_id,
                current_verification.recipe_revision_sha256,
                current_verification.status
            FROM current_verification
            JOIN temp_current_verification_targets AS target
              ON target.stable_id = current_verification.stable_id
             AND target.recipe_revision_sha256 = current_verification.recipe_revision_sha256
             AND target.torchlens_source_hash = current_verification.torchlens_source_hash
             AND target.env_hash = current_verification.env_hash
             AND target.lock_hash = current_verification.lock_hash
             AND target.device_requested = current_verification.device_requested
             AND target.scope = current_verification.scope
            WHERE current_verification.status IN ({placeholders})
              AND current_verification.torchlens_source_hash != 'legacy-unknown'
              AND current_verification.lock_hash != 'legacy-unknown'
        )
        SELECT current_terminal.status, COUNT(*) AS count
        FROM catalog_snapshot
        JOIN current_terminal
          ON current_terminal.stable_id = catalog_snapshot.stable_id
         AND current_terminal.recipe_revision_sha256 = catalog_snapshot.recipe_revision_sha256
        GROUP BY current_terminal.status
        ORDER BY current_terminal.status
        """,
        TERMINAL_STATUSES,
    ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def build_completeness_status(
    catalog_db: Path = CATALOG_DB,
    ledger_db: Path = VERIFICATION_DB,
    torchlens_version: str | None = None,
    render_manifest: Path = DEFAULT_RENDER_MANIFEST,
    run_dir: Path | None = None,
    source_hash: str | None = None,
    device_requested: str = "cpu",
    env_registry: envs.EnvRegistry | None = None,
) -> CompletenessStatus:
    """Build the auditable terminal-status completeness report.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.
    ledger_db:
        Verification ledger SQLite database path.
    torchlens_version:
        TorchLens version for the verified predicate. Defaults to installed TorchLens.
    render_manifest:
        Render manifest path for the underlying funnel.
    run_dir:
        Directory for audit artifacts. Defaults to a timestamped directory under ``/tmp``.
    source_hash:
        TorchLens source hash to require. Defaults to the current checkout hash.
    device_requested:
        Requested device policy to require.
    env_registry:
        Optional loaded validation environment registry.

    Returns
    -------
    CompletenessStatus
        Completeness audit result.
    """

    if torchlens_version is None:
        import torchlens as tl

        torchlens_version = str(tl.__version__)
    resolved_run_dir = run_dir or _default_completeness_run_dir()
    catalog_rows = _catalog_rows(catalog_db)
    catalog_model_rows = load_rows(db_path=catalog_db)
    registry = env_registry or envs.load_registry()
    assignments = _assignment_map(catalog_model_rows, registry)
    current_targets = _verification_targets_for_rows(
        catalog_model_rows,
        assignments,
        source_hash or torchlens_source_hash(),
        device_requested,
        "forward",
    )
    snapshot_path = _write_catalog_snapshot(catalog_rows, assignments, resolved_run_dir)
    funnel = build_status(
        catalog_db=catalog_db,
        ledger_db=ledger_db,
        torchlens_version=torchlens_version,
        render_manifest=render_manifest,
        source_hash=source_hash,
        device_requested=device_requested,
        env_registry=registry,
    )
    with connect_ledger(ledger_db) as ledger_conn:
        _load_catalog_snapshot(ledger_conn, catalog_rows, assignments)
        _load_verification_targets(ledger_conn, current_targets)
        issues = _completeness_issues(ledger_conn)
        terminal_by_status = _terminal_by_status(ledger_conn)
    missing = sum(1 for issue in issues if issue.issue == "missing_terminal")
    stale = sum(1 for issue in issues if issue.issue == "stale_recipe_revision")
    terminal_total = sum(terminal_by_status.values())
    if terminal_total + len(issues) != len(catalog_rows):
        raise RuntimeError(
            "completeness audit row-count mismatch: "
            f"terminal={terminal_total}, issues={len(issues)}, snapshot={len(catalog_rows)}"
        )
    return CompletenessStatus(
        catalog_db=str(catalog_db),
        ledger_db=str(ledger_db),
        torchlens_version=torchlens_version,
        run_dir=str(resolved_run_dir),
        catalog_snapshot=str(snapshot_path),
        total_catalog_models=len(catalog_rows),
        terminal_current_recipe_models=terminal_total,
        missing_terminal_models=missing,
        stale_recipe_models=stale,
        issues=issues,
        terminal_by_status=terminal_by_status,
        funnel=funnel,
    )


def _missing_status(stable_id: str) -> CatalogStatus:
    """Build a conservative fallback status for a missing stable ID.

    Parameters
    ----------
    stable_id:
        Stable ID missing from status metadata.

    Returns
    -------
    CatalogStatus
        Conservative real-input status.
    """

    return CatalogStatus(
        stable_id=stable_id,
        input_is_real=True,
        verification_expectation=VerificationExpectation.forward_required.value,
        quarantine=False,
        code_execution=False,
    )


def _fraction(count: int, total: int) -> float:
    """Return ``count / total`` with zero-safe handling.

    Parameters
    ----------
    count:
        Numerator.
    total:
        Denominator.

    Returns
    -------
    float
        Fraction, or ``0.0`` when total is zero.
    """

    return float(count / total) if total else 0.0


def _status_payload(status: FunnelStatus) -> dict[str, Any]:
    """Convert status dataclasses to a JSON-ready payload.

    Parameters
    ----------
    status:
        Funnel status.

    Returns
    -------
    dict[str, Any]
        JSON-serializable payload.
    """

    payload = asdict(status)
    payload["distinct"] = asdict(status.distinct)
    return payload


def _completeness_payload(status: CompletenessStatus) -> dict[str, Any]:
    """Convert completeness status dataclasses to a JSON-ready payload.

    Parameters
    ----------
    status:
        Completeness status.

    Returns
    -------
    dict[str, Any]
        JSON-serializable payload.
    """

    payload = asdict(status)
    payload["funnel"] = _status_payload(status.funnel)
    return payload


def build_provenance_status(
    ledger_db: Path = VERIFICATION_DB,
    source_jsonl: Path = SOURCE_JSONL,
    deferred_jsonl: Path = DEFERRED_JSONL,
    crawl_history: Path = CRAWL_HISTORY,
) -> ProvenanceStatus:
    """Build the provenance dashboard.

    Parameters
    ----------
    ledger_db:
        Verification database path used for provenance.
    source_jsonl:
        Forward-required catalog JSONL source.
    deferred_jsonl:
        Deferred catalog JSONL source.
    crawl_history:
        Machine-readable crawl history path.

    Returns
    -------
    ProvenanceStatus
        Structured provenance dashboard data.
    """

    sweeps = read_sweeps(ledger_db)
    sweep_ids = {sweep.sweep_id for sweep in sweeps}
    records = load_jsonl(source_jsonl)
    if deferred_jsonl.exists():
        records.extend(load_jsonl(deferred_jsonl))
    wave_counts = Counter(record.added_wave or "unassigned" for record in records)
    known_sweep_models = sum(
        count for wave, count in wave_counts.items() if wave != "unassigned" and wave in sweep_ids
    )
    crawl_data = json.loads(crawl_history.read_text(encoding="utf-8"))
    return ProvenanceStatus(
        ledger_db=str(ledger_db),
        last_exhaustive_crawl=str(crawl_data.get("last_exhaustive_crawl", "")),
        sweep_count=len(sweeps),
        sweeps=sweeps,
        model_wave_counts=dict(sorted(wave_counts.items())),
        models_with_known_sweep=known_sweep_models,
        models_without_wave=wave_counts.get("unassigned", 0),
    )


def _provenance_payload(status: ProvenanceStatus) -> dict[str, Any]:
    """Convert provenance status to a JSON-ready payload.

    Parameters
    ----------
    status:
        Provenance status.

    Returns
    -------
    dict[str, Any]
        JSON-serializable payload.
    """

    payload = asdict(status)
    payload["sweeps"] = [asdict(sweep) for sweep in status.sweeps]
    return payload


def format_provenance_status(status: ProvenanceStatus) -> str:
    """Format a human-readable provenance report.

    Parameters
    ----------
    status:
        Provenance status.

    Returns
    -------
    str
        Human-readable status text.
    """

    lines = [
        "TorchLens Menagerie Provenance",
        f"verification db: {status.ledger_db}",
        f"last_exhaustive_crawl: {status.last_exhaustive_crawl}",
        f"structured sweeps: {status.sweep_count}",
        "",
        "Sweeps:",
    ]
    if status.sweeps:
        lines.extend(
            f"  {sweep.date}: {sweep.sweep_id} [{len(sweep.sources)} sources]"
            for sweep in status.sweeps
        )
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Model added_wave coverage:",
            f"  models with known sweep: {status.models_with_known_sweep}",
            f"  models without added_wave: {status.models_without_wave}",
        ]
    )
    for wave, count in status.model_wave_counts.items():
        lines.append(f"  {count}: {wave}")
    return "\n".join(lines) + "\n"


def format_status(status: FunnelStatus) -> str:
    """Format a human-readable status report.

    Parameters
    ----------
    status:
        Funnel status.

    Returns
    -------
    str
        Human-readable status text.
    """

    render_note = (
        status.render_manifest
        if status.render_manifest_available
        else f"{status.render_manifest} (not found)"
    )
    lines = [
        "TorchLens Menagerie Status",
        f"catalog db: {status.catalog_db}",
        f"verification db: {status.ledger_db}",
        f"torchlens version for verified predicate: {status.torchlens_version}",
        "",
        "Honest funnel:",
        f"  total catalog models: {status.total_catalog_models}",
        f"  expected for verification (deferred excluded): {status.expected_models}",
        f"  rendered models: {status.rendered_models}  [manifest: {render_note}]",
        (f"  forward-validated @ current TL + current recipe: {status.verified_models}"),
        (f"  HEADLINE verified real-input models: {status.headline_verified_real_input}"),
        f"  verified wrapper/sentinel-input models: {status.verified_wrapper_input}",
        (
            "  deferred models: "
            f"{status.deferred_models} ({status.deferred_fraction:.2%} of catalog)"
        ),
        (
            "  quarantined arbitrary-exec models: "
            f"{status.quarantined_models} ({status.quarantine_fraction:.2%} of catalog)"
        ),
        (
            "  built via code execution: "
            f"{status.code_execution_models} ({status.code_execution_fraction:.2%} of catalog)"
        ),
        "",
        "Deferred reasons:",
    ]
    if status.deferred_by_reason:
        lines.extend(f"  {count}: {reason}" for reason, count in status.deferred_by_reason.items())
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Distinct architectures (shape-blind graph_shape_hash):",
            (f"  models with a hash recorded: {status.distinct.hashed_model_count}"),
            (f"  total distinct architectures: {status.distinct.total_distinct_architectures}"),
            (f"  distinct among verified: {status.distinct.verified_distinct_architectures}"),
            (f"  by-name vs by-architecture gap: {status.distinct.by_name_vs_architecture_gap}"),
            f"  hash source: {status.distinct.hash_source}",
        ]
    )
    return "\n".join(lines) + "\n"


def format_completeness_status(status: CompletenessStatus) -> str:
    """Format a human-readable completeness report.

    Parameters
    ----------
    status:
        Completeness status.

    Returns
    -------
    str
        Human-readable completeness report.
    """

    lines = [
        "TorchLens Menagerie Completeness",
        f"catalog db: {status.catalog_db}",
        f"verification db: {status.ledger_db}",
        f"torchlens version for verified predicate: {status.torchlens_version}",
        f"run dir: {status.run_dir}",
        f"catalog snapshot: {status.catalog_snapshot}",
        "",
        "Terminal-status audit:",
        f"  total catalog models: {status.total_catalog_models}",
        f"  terminal at current recipe: {status.terminal_current_recipe_models}",
        f"  missing_terminal: {status.missing_terminal_models}",
        f"  stale_recipe_revision: {status.stale_recipe_models}",
        "",
        "Terminal funnel:",
    ]
    if status.terminal_by_status:
        lines.extend(
            f"  {count}: {terminal_status}"
            for terminal_status, count in status.terminal_by_status.items()
        )
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "Verified headline:",
            (
                "  real-input forward-validated @ current TL + current recipe: "
                f"{status.funnel.headline_verified_real_input}"
            ),
            (
                "  all forward-validated @ current TL + current recipe: "
                f"{status.funnel.verified_models}"
            ),
        ]
    )
    if status.issues:
        lines.extend(["", "Completeness issues:"])
        for issue in status.issues[:50]:
            detail = (
                f" stale_recipe={issue.stale_recipe_revision_sha256}"
                f" stale_status={issue.stale_status}"
                if issue.issue == "stale_recipe_revision"
                else ""
            )
            lines.append(
                "  "
                f"{issue.issue}: {issue.stable_id} "
                f"{issue.name} [{issue.zoo}/{issue.source}]"
                f" current_recipe={issue.recipe_revision_sha256}{detail}"
            )
        if len(status.issues) > 50:
            lines.append(f"  ... {len(status.issues) - 50} more")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the status CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-db", type=Path, default=CATALOG_DB)
    parser.add_argument("--ledger-db", type=Path, default=VERIFICATION_DB)
    parser.add_argument("--torchlens-version")
    parser.add_argument("--render-manifest", type=Path, default=DEFAULT_RENDER_MANIFEST)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--provenance", action="store_true")
    parser.add_argument("--completeness", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-deferred-frac", type=float)
    parser.add_argument("--max-quarantine-frac", type=float)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the status command.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    if args.provenance:
        provenance = build_provenance_status(ledger_db=args.ledger_db)
        if args.json:
            print(json.dumps(_provenance_payload(provenance), indent=2, sort_keys=True))
        else:
            print(format_provenance_status(provenance), end="")
        return 0

    if args.completeness:
        completeness = build_completeness_status(
            catalog_db=args.catalog_db,
            ledger_db=args.ledger_db,
            torchlens_version=args.torchlens_version,
            render_manifest=args.render_manifest,
            run_dir=args.run_dir,
        )
        if args.json:
            print(json.dumps(_completeness_payload(completeness), indent=2, sort_keys=True))
        else:
            print(format_completeness_status(completeness), end="")
        return 1 if completeness.issues else 0

    status = build_status(
        catalog_db=args.catalog_db,
        ledger_db=args.ledger_db,
        torchlens_version=args.torchlens_version,
        render_manifest=args.render_manifest,
    )
    if args.json:
        print(json.dumps(_status_payload(status), indent=2, sort_keys=True))
    else:
        print(format_status(status), end="")
    failed = False
    if args.max_deferred_frac is not None and status.deferred_fraction > args.max_deferred_frac:
        failed = True
    if (
        args.max_quarantine_frac is not None
        and status.quarantine_fraction > args.max_quarantine_frac
    ):
        failed = True
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entry point.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
