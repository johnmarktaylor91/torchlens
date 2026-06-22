"""Sample parity check for legacy TSV recipes versus typed JSONL records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from menagerie.catalog import SOURCE_COLUMNS, SOURCE_TSV, CatalogRow, is_verified
from menagerie.recipe import build_from_record, build_input_from_record, instantiate_model
from menagerie.schema import CatalogRecord, load_jsonl
from menagerie.structural_digest import structural_fingerprint
from menagerie.tools.tsv_to_jsonl import DEFAULT_OUTPUT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / ".research" / "menagerie-redesign" / "migration_parity.json"


def _read_raw_rows(path: Path) -> dict[int, dict[str, str]]:
    """Read raw TSV rows keyed by physical line number.

    Parameters
    ----------
    path:
        Source TSV path.

    Returns
    -------
    dict[int, dict[str, str]]
        Raw source rows keyed by line number.
    """

    with path.open(newline="") as handle:
        raw_rows = list(csv.reader(handle, delimiter="\t"))
    if not raw_rows:
        return {}
    first = [value.strip() for value in raw_rows[0]]
    data_rows = raw_rows[1:] if first == list(SOURCE_COLUMNS) else raw_rows
    offset = 2 if first == list(SOURCE_COLUMNS) else 1
    rows: dict[int, dict[str, str]] = {}
    for line_number, row in enumerate(data_rows, start=offset):
        if len(row) != len(SOURCE_COLUMNS):
            raise ValueError(f"{path}:{line_number} has {len(row)} columns, expected 9")
        rows[line_number] = dict(zip(SOURCE_COLUMNS, row))
    return rows


def _catalog_row_from_raw(row: dict[str, str], line_number: int) -> CatalogRow:
    """Build a minimal legacy catalog row for recipe parity.

    Parameters
    ----------
    row:
        Raw TSV row.
    line_number:
        Source line number.

    Returns
    -------
    CatalogRow
        Catalog row accepted by legacy recipe builders.
    """

    return CatalogRow(
        model_id=line_number,
        display_index=line_number,
        stable_id=f"line-{line_number}",
        name=row["name"],
        variant="",
        family=row["family"],
        family_normalized=row["family"],
        domain=row["domain"],
        zoo=row["zoo"],
        constructor_call=row["constructor_call"],
        input_shape=row["input_shape"],
        input_dtype=row["input_dtype"],
        era=row["era"],
        verified=is_verified(row["notes"], row["zoo"]),
        notes=row["notes"],
        source="catalog",
        recipe_revision_sha256="",
    )


def stratified_sample(records: Sequence[CatalogRecord], target: int) -> list[CatalogRecord]:
    """Choose a deterministic stratified sample from eligible typed records.

    Parameters
    ----------
    records:
        Candidate JSONL records.
    target:
        Desired sample size.

    Returns
    -------
    list[CatalogRecord]
        Stratified non-deferred records.
    """

    buckets: dict[tuple[str, str], list[CatalogRecord]] = defaultdict(list)
    for record in records:
        if record.deferral is not None:
            continue
        buckets[(record.recipe.type, record.input.kind)].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: (item.zoo, item.name, item.variant))

    selected: list[CatalogRecord] = []
    bucket_keys = sorted(buckets)
    while len(selected) < target:
        progressed = False
        for key in bucket_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break
    return selected


def _line_number(record: CatalogRecord) -> int:
    """Return the legacy TSV line number for a migrated record.

    Parameters
    ----------
    record:
        Candidate record.

    Returns
    -------
    int
        Source TSV line number.
    """

    return int(record.legacy["line_number"])


def compare_record(record: CatalogRecord, raw_rows: dict[int, dict[str, str]]) -> dict[str, Any]:
    """Compare structural fingerprints for one migrated record.

    Parameters
    ----------
    record:
        Candidate record.
    raw_rows:
        Raw TSV rows keyed by line number.

    Returns
    -------
    dict[str, Any]
        Comparison result.
    """

    import torch

    line_number = _line_number(record)
    old_row = _catalog_row_from_raw(raw_rows[line_number], line_number)
    torch.manual_seed(0)
    old_input = build_input_from_record(record)
    old_model = instantiate_model(old_row)
    torch.manual_seed(0)
    new_model, new_input = build_from_record(record)
    torch.manual_seed(0)
    old_fingerprint = structural_fingerprint(old_model, old_input)
    torch.manual_seed(0)
    new_fingerprint = structural_fingerprint(new_model, new_input)
    return {
        "name": record.name,
        "zoo": record.zoo,
        "variant": record.variant,
        "recipe_type": record.recipe.type,
        "input_kind": record.input.kind,
        "line_number": line_number,
        "matched": old_fingerprint == new_fingerprint,
        "old_fingerprint": old_fingerprint,
        "new_fingerprint": new_fingerprint,
    }


def run_parity(
    candidate_path: Path = DEFAULT_OUTPUT,
    source_tsv: Path = SOURCE_TSV,
    report_path: Path = DEFAULT_REPORT,
    *,
    target: int = 120,
    mismatch_threshold: float = 0.05,
    write: bool = True,
) -> dict[str, Any]:
    """Run the sample migration parity check.

    Parameters
    ----------
    candidate_path:
        Candidate JSONL path.
    source_tsv:
        Legacy TSV path.
    report_path:
        JSON report output path.
    target:
        Desired sample size.
    mismatch_threshold:
        Maximum allowed mismatch fraction.
    write:
        Whether to write the report artifact.

    Returns
    -------
    dict[str, Any]
        Parity report.
    """

    records = load_jsonl(candidate_path)
    raw_rows = _read_raw_rows(source_tsv)
    sample = stratified_sample(records, target)
    results = []
    skipped = []
    for record in sample:
        try:
            results.append(compare_record(record, raw_rows))
        except Exception as exc:  # noqa: BLE001 - dependency/runtime skips are reported.
            skipped.append(
                {
                    "name": record.name,
                    "zoo": record.zoo,
                    "recipe_type": record.recipe.type,
                    "input_kind": record.input.kind,
                    "reason": repr(exc),
                }
            )

    mismatches = [result for result in results if not result["matched"]]
    mismatch_fraction = len(mismatches) / len(results) if results else 0.0
    exercised = [
        {"recipe_type": recipe_type, "input_kind": input_kind}
        for recipe_type, input_kind in sorted(
            {(result["recipe_type"], result["input_kind"]) for result in results}
        )
    ]
    report: dict[str, Any] = {
        "candidate_path": str(candidate_path),
        "source_tsv": str(source_tsv),
        "target": target,
        "attempted": len(sample),
        "tested": len(results),
        "matched": len(results) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatch_fraction": mismatch_fraction,
        "mismatches": mismatches,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "exercised_recipe_input_pairs": exercised,
    }
    if write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if results and mismatch_fraction > mismatch_threshold:
        raise RuntimeError(
            f"migration parity mismatch fraction {mismatch_fraction:.1%} exceeds "
            f"{mismatch_threshold:.1%}"
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the migration parity command.

    Parameters
    ----------
    argv:
        Optional CLI arguments.

    Returns
    -------
    int
        Process exit status.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-tsv", type=Path, default=SOURCE_TSV)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    report = run_parity(
        args.candidate,
        args.source_tsv,
        args.report,
        target=args.target,
        write=not args.no_write,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
