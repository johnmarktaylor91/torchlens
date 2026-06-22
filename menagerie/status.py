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
from pathlib import Path
from typing import Any, Sequence

from menagerie.catalog import CATALOG_DB, DEFERRED_JSONL, SOURCE_JSONL, ensure_catalog
from menagerie.ledger import VERIFICATION_DB, connect as connect_ledger, verified_count
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
    with connect_ledger(ledger_db) as ledger_conn:
        verified_by_stable_id = verified_hashes(ledger_conn, torchlens_version, current_revisions)
        verified_total = verified_count(ledger_conn, torchlens_version, current_revisions)
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
