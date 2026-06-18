"""Report TorchLens graph-shape-hash duplicates in menagerie manifests."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.catalog import CATALOG_DB, CatalogRow, load_rows
from menagerie.generate_menagerie import (
    classics_example_input,
    cuda_is_available,
    instantiate_model,
    is_classics_row,
    is_device_related_error,
    move_model_and_input_to_device,
    tensor_for_recipe,
    unrenderable_reason,
)


REPORT_MD = "dedup_report.md"
REPORT_JSON = "dedup_report.json"


@dataclass(frozen=True)
class HashRecord:
    """One model-name/hash observation.

    Parameters
    ----------
    name:
        Model name.
    graph_shape_hash:
        TorchLens architecture hash.
    status:
        Source or tracing status.
    error:
        Error text, when tracing failed.
    """

    name: str
    graph_shape_hash: str
    status: str = ""
    error: str = ""


def read_manifest_records(manifest_path: Path) -> tuple[list[HashRecord], bool]:
    """Read hash records from a manifest TSV.

    Parameters
    ----------
    manifest_path:
        Render or validation manifest path.

    Returns
    -------
    tuple[list[HashRecord], bool]
        Hash records and whether a named graph-shape-hash column exists.
    """

    if not manifest_path.exists():
        return [], False
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        has_hash_column = "graph_shape_hash" in (reader.fieldnames or [])
        records = [
            HashRecord(
                name=str(row.get("name", "")),
                graph_shape_hash=str(row.get("graph_shape_hash", "") or ""),
                status=str(row.get("status", "") or ""),
            )
            for row in reader
            if row.get("name")
        ]
    return records, has_hash_column


def build_input(row: CatalogRow) -> Any:
    """Build the example input for a catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    Any
        Example input object.
    """

    if is_classics_row(row):
        return classics_example_input(row)
    return tensor_for_recipe(row.input_shape, row.input_dtype)


def trace_graph_shape_hash(model: Any, input_value: Any) -> str:
    """Trace a model and return its graph-shape hash.

    Parameters
    ----------
    model:
        Instantiated model.
    input_value:
        Example input.

    Returns
    -------
    str
        TorchLens graph-shape hash.
    """

    import torch
    import torchlens as tl

    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        trace = tl.trace(
            model,
            input_value,
            layers_to_save=None,
            save=None,
            save_rng_states=False,
            inference_only=True,
        )
    return str(getattr(trace, "graph_shape_hash", "") or "")


def hash_catalog_row(row: CatalogRow, device: str) -> HashRecord:
    """Compute the graph-shape hash for one catalog row.

    Parameters
    ----------
    row:
        Catalog row.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.

    Returns
    -------
    HashRecord
        Hash record with status and error details.
    """

    skip_reason = unrenderable_reason(row)
    if skip_reason is not None:
        return HashRecord(row.name, "", f"skipped:{skip_reason}", skip_reason)
    try:
        input_value = build_input(row)
        model = instantiate_model(row)
    except Exception as error:
        return HashRecord(
            row.name,
            "",
            "failed:setup",
            f"{error!r}\n{traceback.format_exc(limit=4)}",
        )

    def attempt(attempt_model: Any, attempt_input: Any, actual_device: str) -> HashRecord:
        """Trace on a resolved device.

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
        HashRecord
            Hash record for the attempt.
        """

        graph_shape_hash = trace_graph_shape_hash(attempt_model, attempt_input)
        return HashRecord(row.name, graph_shape_hash, f"hashed:device={actual_device}", "")

    try:
        if device == "cuda":
            model, input_value = move_model_and_input_to_device(model, input_value, "cuda")
            return attempt(model, input_value, "cuda")
        if device == "auto":
            try:
                return attempt(model, input_value, "cpu")
            except Exception as error:
                if not is_device_related_error(error) or not cuda_is_available():
                    raise
                model, input_value = move_model_and_input_to_device(model, input_value, "cuda")
                return attempt(model, input_value, "cuda")
        return attempt(model, input_value, "cpu")
    except Exception as error:
        return HashRecord(
            row.name,
            "",
            "failed:trace",
            f"{error!r}\n{traceback.format_exc(limit=4)}",
        )


def catalog_hash_records(subset: int, device: str, db_path: Path) -> list[HashRecord]:
    """Compute hash records for a catalog sample.

    Parameters
    ----------
    subset:
        Number of catalog rows to sample from the start of the catalog.
    device:
        Device mode, one of ``"cpu"``, ``"cuda"``, or ``"auto"``.
    db_path:
        Catalog database path.

    Returns
    -------
    list[HashRecord]
        Hash records for sampled rows.
    """

    rows = load_rows(limit=subset, db_path=db_path)
    return [hash_catalog_row(row, device) for row in rows]


def duplicate_groups(records: Sequence[HashRecord]) -> dict[str, list[str]]:
    """Group distinct model names by nonblank graph-shape hash.

    Parameters
    ----------
    records:
        Hash records.

    Returns
    -------
    dict[str, list[str]]
        Hashes with more than one distinct model name.
    """

    names_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in records:
        graph_shape_hash = record.graph_shape_hash.strip()
        if graph_shape_hash:
            names_by_hash[graph_shape_hash].add(record.name)
    return {
        graph_shape_hash: sorted(names)
        for graph_shape_hash, names in sorted(names_by_hash.items())
        if len(names) > 1
    }


def summary_payload(
    records: Sequence[HashRecord], duplicates: Mapping[str, Sequence[str]], source: str
) -> dict[str, Any]:
    """Build a JSON-serializable dedup summary.

    Parameters
    ----------
    records:
        Hash records.
    duplicates:
        Duplicate hash groups.
    source:
        Description of the source data.

    Returns
    -------
    dict[str, Any]
        Report payload.
    """

    hashed_records = [record for record in records if record.graph_shape_hash.strip()]
    distinct_hashes = {record.graph_shape_hash.strip() for record in hashed_records}
    return {
        "source": source,
        "total_records": len(records),
        "total_hashed": len(hashed_records),
        "distinct_architectures": len(distinct_hashes),
        "architectural_duplicate_groups": len(duplicates),
        "architectural_duplicates": [
            {"graph_shape_hash": graph_shape_hash, "names": list(names)}
            for graph_shape_hash, names in duplicates.items()
        ],
        "unhashed": [
            {
                "name": record.name,
                "status": record.status,
                "error": record.error,
            }
            for record in records
            if not record.graph_shape_hash.strip()
        ],
    }


def markdown_report(payload: Mapping[str, Any]) -> str:
    """Render a dedup report as Markdown.

    Parameters
    ----------
    payload:
        Summary payload.

    Returns
    -------
    str
        Markdown report body.
    """

    lines = [
        "# Menagerie Dedup Report",
        "",
        f"Source: `{payload['source']}`",
        "",
        f"- Total records: {payload['total_records']}",
        f"- Total hashed: {payload['total_hashed']}",
        f"- Distinct architectures: {payload['distinct_architectures']}",
        f"- Architectural duplicate groups: {payload['architectural_duplicate_groups']}",
        "",
        "## Architectural Duplicates",
        "",
    ]
    duplicates = list(payload["architectural_duplicates"])
    if not duplicates:
        lines.append("No architectural duplicates with more than one distinct model name.")
    else:
        lines.extend(["| graph_shape_hash | distinct names |", "| --- | --- |"])
        for group in duplicates:
            names = ", ".join(f"`{name}`" for name in group["names"])
            lines.append(f"| `{group['graph_shape_hash']}` | {names} |")
    lines.extend(["", "## Unhashed", ""])
    unhashed = list(payload["unhashed"])
    if not unhashed:
        lines.append("No unhashed records.")
    else:
        lines.append(f"Unhashed records: {len(unhashed)}.")
    return "\n".join(lines) + "\n"


def write_reports(payload: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write Markdown and JSON dedup reports.

    Parameters
    ----------
    payload:
        Summary payload.
    out_dir:
        Output directory.

    Returns
    -------
    tuple[Path, Path]
        Markdown and JSON report paths.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / REPORT_MD
    json_path = out_dir / REPORT_JSON
    md_path.write_text(markdown_report(payload))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return md_path, json_path


def build_parser() -> argparse.ArgumentParser:
    """Build the dedup report CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        action="store_true",
        help="compute hashes from a catalog sample when the manifest lacks hash values",
    )
    parser.add_argument("--subset", type=int, help="catalog sample size for --catalog mode")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--db", type=Path, default=CATALOG_DB)
    parser.add_argument("--out-dir", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    """Run the dedup report.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    records, has_hash_column = read_manifest_records(args.manifest)
    hashed = [record for record in records if record.graph_shape_hash.strip()]
    source = str(args.manifest)
    if args.catalog and not hashed:
        if args.subset is None:
            raise RuntimeError("--catalog requires --subset when manifest hashes are unavailable")
        records = catalog_hash_records(args.subset, args.device, args.db)
        source = f"catalog sample: first {args.subset} rows from {args.db}"
    elif args.catalog and has_hash_column:
        source = f"{args.manifest} (manifest hashes present; catalog sampling skipped)"
    elif not has_hash_column:
        source = f"{args.manifest} (no graph_shape_hash column)"

    duplicates = duplicate_groups(records)
    payload = summary_payload(records, duplicates, source)
    out_dir = (args.out_dir or args.manifest.parent).resolve()
    md_path, json_path = write_reports(payload, out_dir)
    print(f"total hashed: {payload['total_hashed']}")
    print(f"distinct architectures: {payload['distinct_architectures']}")
    print(f"architectural duplicate groups: {payload['architectural_duplicate_groups']}")
    print(f"wrote: {md_path}")
    print(f"wrote: {json_path}")
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

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except RuntimeError as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
