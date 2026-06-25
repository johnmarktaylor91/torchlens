"""Distinct architecture reporting for the TorchLens menagerie."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from menagerie.catalog import CATALOG_DB, ensure_catalog
from menagerie.dedup_report import HashRecord, read_manifest_records
from menagerie.ledger import (
    LEGACY_UNKNOWN,
    VERIFICATION_DB,
    VerificationTarget,
    _default_verification_targets,
    _load_verification_targets,
    connect as connect_ledger,
)


@dataclass(frozen=True)
class DistinctArchitectureReport:
    """Summary of shape-blind architecture distinctness.

    Parameters
    ----------
    catalog_model_count:
        Total catalog model count.
    hashed_model_count:
        Number of catalog models with any recorded graph-shape hash.
    total_distinct_architectures:
        Distinct shape-blind graph-shape hashes among all hashed catalog models.
    verified_model_count:
        Number of current-recipe, current-version models satisfying the frozen verified predicate.
    verified_distinct_architectures:
        Distinct shape-blind graph-shape hashes among verified models.
    by_name_vs_architecture_gap:
        Difference between hashed model names and distinct architecture hashes.
    verified_by_name_vs_architecture_gap:
        Difference between verified model names and verified distinct architecture hashes.
    hash_source:
        Human-readable source summary for all-model hashes.
    """

    catalog_model_count: int
    hashed_model_count: int
    total_distinct_architectures: int
    verified_model_count: int
    verified_distinct_architectures: int
    by_name_vs_architecture_gap: int
    verified_by_name_vs_architecture_gap: int
    hash_source: str


def current_revisions_from_catalog(catalog_db: Path = CATALOG_DB) -> dict[str, str]:
    """Load current recipe revisions keyed by stable ID from the catalog.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.

    Returns
    -------
    dict[str, str]
        Current recipe revision hash by stable model ID.
    """

    ensure_catalog(catalog_db)
    with sqlite3.connect(catalog_db) as conn:
        rows = conn.execute("SELECT stable_id, recipe_revision_sha256 FROM models").fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def catalog_model_count(catalog_db: Path = CATALOG_DB) -> int:
    """Return the number of catalog models.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.

    Returns
    -------
    int
        Catalog model count.
    """

    ensure_catalog(catalog_db)
    with sqlite3.connect(catalog_db) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM models").fetchone()[0])


def ledger_hash_records(ledger_conn: sqlite3.Connection) -> dict[str, HashRecord]:
    """Return latest ledger graph-shape hashes keyed by stable ID.

    Parameters
    ----------
    ledger_conn:
        Initialized verification ledger connection.

    Returns
    -------
    dict[str, HashRecord]
        Latest nonblank graph-shape hash observations by stable model ID.
    """

    rows = ledger_conn.execute(
        """
        WITH ranked AS (
            SELECT
                stable_id,
                recipe_revision_sha256,
                name,
                graph_shape_hash,
                status,
                ROW_NUMBER() OVER (
                    PARTITION BY stable_id
                    ORDER BY finished_at DESC, run_id DESC
                ) AS rn
            FROM verification_runs
            WHERE graph_shape_hash IS NOT NULL
              AND graph_shape_hash != ''
        )
        SELECT stable_id, recipe_revision_sha256, name, graph_shape_hash, status
        FROM ranked
        WHERE rn = 1
        """
    ).fetchall()
    return {
        str(row["stable_id"]): HashRecord(
            name=str(row["name"]),
            graph_shape_hash=str(row["graph_shape_hash"]),
            status=str(row["status"]),
            stable_id=str(row["stable_id"]),
            recipe_revision_sha256=str(row["recipe_revision_sha256"]),
        )
        for row in rows
    }


def manifest_hash_records(manifest_paths: Sequence[Path]) -> dict[str, HashRecord]:
    """Return fallback manifest hash records keyed by stable ID or name.

    Parameters
    ----------
    manifest_paths:
        Render or validation manifest paths.

    Returns
    -------
    dict[str, HashRecord]
        Hash records from manifests. Stable IDs are preferred when present.
    """

    records_by_key: dict[str, HashRecord] = {}
    for manifest_path in manifest_paths:
        records, _ = read_manifest_records(manifest_path)
        for record in records:
            if not record.graph_shape_hash.strip():
                continue
            key = record.stable_id or f"name:{record.name}"
            records_by_key[key] = record
    return records_by_key


def all_hash_records(
    ledger_conn: sqlite3.Connection, manifest_paths: Sequence[Path]
) -> tuple[dict[str, HashRecord], str]:
    """Merge ledger graph hashes with manifest fallback hashes.

    Parameters
    ----------
    ledger_conn:
        Initialized verification ledger connection.
    manifest_paths:
        Render or validation manifest paths used only when a stable ID has no ledger hash.

    Returns
    -------
    tuple[dict[str, HashRecord], str]
        Hash records keyed by stable ID or name fallback, and a source description.
    """

    records = ledger_hash_records(ledger_conn)
    ledger_count = len(records)
    fallback_records = manifest_hash_records(manifest_paths)
    for key, record in fallback_records.items():
        if key not in records:
            records[key] = record
    fallback_count = len(records) - ledger_count
    if manifest_paths:
        manifest_text = ", ".join(str(path) for path in manifest_paths)
        source = (
            f"ledger ({ledger_count}) + manifest fallback ({fallback_count}) from {manifest_text}"
        )
    else:
        source = f"ledger ({ledger_count}); no manifest fallback paths supplied"
    return records, source


def verified_hashes(
    ledger_conn: sqlite3.Connection,
    torchlens_version: str,
    current_revisions: dict[str, str],
    current_targets: dict[str, VerificationTarget] | None = None,
) -> dict[str, str]:
    """Return verified graph hashes keyed by stable ID.

    Parameters
    ----------
    ledger_conn:
        Initialized verification ledger connection.
    torchlens_version:
        TorchLens version that must match current verification rows.
    current_revisions:
        Catalog mapping from stable ID to current recipe revision.
    current_targets:
        Optional full identity targets keyed by stable ID. When omitted, the
        current base environment identity is used for all revisions.

    Returns
    -------
    dict[str, str]
        Verified graph-shape hash by stable model ID.
    """

    targets = (
        current_targets
        if current_targets is not None
        else _default_verification_targets(current_revisions)
    )
    if not targets:
        return {}
    _load_verification_targets(ledger_conn, targets)
    rows = ledger_conn.execute(
        """
        SELECT current_verification.stable_id, current_verification.graph_shape_hash
        FROM current_verification
        JOIN temp_current_verification_targets AS target
          ON target.stable_id = current_verification.stable_id
         AND target.recipe_revision_sha256 = current_verification.recipe_revision_sha256
         AND target.torchlens_source_hash = current_verification.torchlens_source_hash
         AND target.env_hash = current_verification.env_hash
         AND target.lock_hash = current_verification.lock_hash
         AND target.device_requested = current_verification.device_requested
         AND target.scope = current_verification.scope
        WHERE current_verification.status = 'passed'
          AND current_verification.forward_pass = 1
          AND current_verification.metadata_ok = 1
          AND current_verification.n_ops IS NOT NULL
          AND current_verification.graph_shape_hash IS NOT NULL
          AND current_verification.torchlens_version = ?
          AND current_verification.torchlens_source_hash != ?
          AND current_verification.lock_hash != ?
        """,
        (torchlens_version, LEGACY_UNKNOWN, LEGACY_UNKNOWN),
    ).fetchall()
    return {str(row["stable_id"]): str(row["graph_shape_hash"]) for row in rows}


def build_distinct_report(
    catalog_db: Path = CATALOG_DB,
    ledger_db: Path = VERIFICATION_DB,
    torchlens_version: str | None = None,
    manifest_paths: Sequence[Path] = (),
    current_targets: dict[str, VerificationTarget] | None = None,
) -> DistinctArchitectureReport:
    """Build the distinct-architecture summary.

    Parameters
    ----------
    catalog_db:
        Catalog SQLite database path.
    ledger_db:
        Verification ledger SQLite database path.
    torchlens_version:
        TorchLens version for the verified subset. Defaults to the installed package version.
    manifest_paths:
        Optional render or validation manifests used as fallback hash sources.
    current_targets:
        Optional full identity targets keyed by stable ID.

    Returns
    -------
    DistinctArchitectureReport
        Distinct architecture summary.
    """

    if torchlens_version is None:
        import torchlens as tl

        torchlens_version = str(tl.__version__)
    current_revisions = current_revisions_from_catalog(catalog_db)
    with connect_ledger(ledger_db) as ledger_conn:
        records_by_key, source = all_hash_records(ledger_conn, manifest_paths)
        verified_by_stable_id = verified_hashes(
            ledger_conn, torchlens_version, current_revisions, current_targets
        )
    all_hashes = {
        record.graph_shape_hash.strip()
        for record in records_by_key.values()
        if record.graph_shape_hash.strip()
    }
    verified_hash_values = {
        graph_shape_hash.strip()
        for graph_shape_hash in verified_by_stable_id.values()
        if graph_shape_hash.strip()
    }
    hashed_model_count = len(records_by_key)
    verified_model_count = len(verified_by_stable_id)
    return DistinctArchitectureReport(
        catalog_model_count=catalog_model_count(catalog_db),
        hashed_model_count=hashed_model_count,
        total_distinct_architectures=len(all_hashes),
        verified_model_count=verified_model_count,
        verified_distinct_architectures=len(verified_hash_values),
        by_name_vs_architecture_gap=hashed_model_count - len(all_hashes),
        verified_by_name_vs_architecture_gap=verified_model_count - len(verified_hash_values),
        hash_source=source,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the distinct report CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-db", type=Path, default=CATALOG_DB)
    parser.add_argument("--ledger-db", type=Path, default=VERIFICATION_DB)
    parser.add_argument("--torchlens-version")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the distinct report CLI.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    report = build_distinct_report(
        catalog_db=args.catalog_db,
        ledger_db=args.ledger_db,
        torchlens_version=args.torchlens_version,
        manifest_paths=args.manifest,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0
    print(f"catalog models: {report.catalog_model_count}")
    print(f"models with graph_shape_hash recorded: {report.hashed_model_count}")
    print(f"total distinct architectures: {report.total_distinct_architectures}")
    print(f"verified models: {report.verified_model_count}")
    print(f"distinct among verified: {report.verified_distinct_architectures}")
    print(f"by-name vs by-architecture gap: {report.by_name_vs_architecture_gap}")
    print(f"verified by-name vs by-architecture gap: {report.verified_by_name_vs_architecture_gap}")
    print(f"hash source: {report.hash_source}")
    return 0


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
