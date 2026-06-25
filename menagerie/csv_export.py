"""Public CSV and side-table exporter for the model menagerie."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re
import sqlite3
import subprocess
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from packaging.version import InvalidVersion, Version

from menagerie.catalog import CATALOG_DB, CatalogRow, load_rows
from menagerie.csv_dictionary import DEFAULT_SCHEMA_PATH, SchemaColumn, parse_flagship_schema
from menagerie.ledger import STATUS_VALUES, VERIFICATION_DB
from menagerie.trace_summary import TRACE_SUMMARY_DB


DATASET_SCHEMA_VERSION = "2.0"
TRACE_METRICS_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "n_compute_ops",
    "n_unique_op_types",
    "n_inplace_ops",
    "has_custom_op",
    "dominant_op_type",
    "pct_conv",
    "pct_linear",
    "pct_attention",
    "pct_norm",
    "pct_elementwise",
    "pct_reduction",
    "pct_reshape",
    "pct_embedding",
    "pct_pooling",
    "graph_depth",
    "graph_max_width",
    "branching_factor",
    "max_fan_out",
    "max_fan_in",
    "is_branching",
    "is_recurrent",
    "max_recurrence_iters",
    "n_recurrent_layers",
    "has_conditional_branching",
    "n_conditionals",
    "is_dynamic_graph",
    "n_modules",
    "n_module_calls",
    "module_max_depth",
    "n_top_level_modules",
    "n_unique_module_types",
    "model_class_name",
    "model_class_qualname",
    "n_params",
    "n_params_trainable",
    "n_params_frozen",
    "n_param_tensors",
    "param_memory_bytes",
    "primary_param_dtype",
    "param_dtype_set_json",
    "quantized_param_tensor_count",
    "has_frozen_params",
    "n_buffers",
    "buffer_memory_bytes",
    "buffer_overwrite_count",
    "total_flops_forward",
    "total_macs_forward",
    "total_flops_backward",
    "total_macs_backward",
    "flops_coverage_pct",
    "n_unknown_flops_ops",
    "activation_memory_bytes",
    "forward_peak_memory_bytes",
    "largest_activation_bytes",
    "param_memory_mb",
    "activation_memory_mb",
    "forward_peak_memory_mb",
)
TRACE_HISTOGRAM_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "op_type_histogram",
    "module_type_histogram",
    "top_module_types_json",
    "top_level_block_sequence_json",
    "flops_by_op_type",
    "macs_by_op_type",
    "structural_barcode",
)
PAPERS_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "paper_title",
    "paper_url",
    "arxiv_id",
    "doi",
    "publication_venue",
    "venue_type",
    "authors_json",
    "first_author",
    "first_preprint_date",
    "first_publication_date",
    "huggingface_id",
    "model_card_url",
    "semantic_scholar_paper_id",
    "n_citations",
    "citations_snapshot_date",
    "citation_tier",
    "is_milestone",
)
LINEAGE_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "relation",
    "related_stable_id",
    "related_name",
    "variant_dimension",
    "is_canonical_variant",
    "source_note",
)
ARTIFACTS_COLUMNS: tuple[str, ...] = (
    "stable_id",
    "model_page_url",
    "svg_path",
    "svg_url",
    "svg_sha256",
    "has_svg",
    "tlspec_path",
    "model_card_url",
    "added_wave",
    "row_quality_flags_json",
    "dataset_schema_version",
    "dataset_as_of_date",
    "csv_generation_commit_sha",
)
_FLAGSHIP_TRACE_COLUMNS = {
    "modality_primary": None,
    "architecture_paradigm": None,
    "n_params": "n_params",
    "n_params_source": "n_params_source",
    "modality_output": None,
    "is_recurrent": "is_recurrent",
    "is_branching": "is_branching",
    "is_dynamic_graph": "is_dynamic_graph",
    "n_compute_ops": "n_compute_ops",
    "n_unique_op_types": "n_unique_op_types",
    "dominant_op_type": "dominant_op_type",
    "top_op_types_json": "top_op_types_json",
    "graph_depth": "graph_depth",
    "graph_max_width": "graph_max_width",
    "max_fan_out": "max_fan_out",
    "max_fan_in": "max_fan_in",
    "max_recurrence_iters": "max_recurrence_iters",
    "n_conditionals": "n_conditionals",
    "pct_conv": "pct_conv",
    "pct_linear": "pct_linear",
    "pct_attention": "pct_attention",
    "pct_norm": "pct_norm",
    "pct_elementwise": "pct_elementwise",
    "has_attention": "has_attention",
    "has_conv": "has_conv",
    "has_residual": "has_residual",
    "has_embedding": "has_embedding",
    "norm_type": "norm_type",
    "activation_fn_type": "activation_fn_type",
    "n_modules": "n_modules",
    "n_unique_module_types": "n_unique_module_types",
    "module_max_depth": "module_max_depth",
    "model_class_name": "model_class_name",
    "structural_barcode_json": "structural_barcode_json",
    "total_flops_forward": "total_flops_forward",
    "total_macs_forward": "total_macs_forward",
    "flops_coverage_pct": "flops_coverage_pct",
    "param_memory_bytes": "param_memory_bytes",
    "param_memory_mb": "param_memory_mb",
    "activation_memory_bytes": "activation_memory_bytes",
    "activation_memory_mb": "activation_memory_mb",
    "forward_peak_memory_mb": "forward_peak_memory_mb",
    "n_params_trainable": "n_params_trainable",
    "primary_param_dtype": "primary_param_dtype",
    "n_buffers": "n_buffers",
    "output_shape": "output_shape",
    "n_output_tensors": "n_output_tensors",
}
_EXTERNAL_FLAGSHIP_COLUMNS = {
    "task_primary",
    "architecture_subclass",
    "is_generative",
    "publication_year",
    "paper_url",
    "code_url",
    "license",
    "output_type",
}
_BOOL_COLUMNS = {
    "is_trustworthy",
    "is_multimodal",
    "is_generative",
    "is_recurrent",
    "is_branching",
    "is_dynamic_graph",
    "has_attention",
    "has_conv",
    "has_residual",
    "has_embedding",
    "validated_on_current_release",
    "catalog_verified_hint",
}
_INT_TYPES = {"int", "int (YYYY)", "int (FLOPs)", "int (MACs)", "int (bytes)"}
_FLOAT_TYPES = {"float (0-100)", "float (MB)"}
_SIZE_TOKEN_RE = re.compile(r"\b(nano|tiny|small|base|large|xl|xxl|huge|giant|mini|micro)\b", re.I)
_SHAPE_NUMBER_RE = re.compile(r"-?\d+")


def _current_torchlens_version() -> str:
    """Return the installed TorchLens version.

    Returns
    -------
    str
        Installed package version, or an empty string when unavailable.
    """

    try:
        return version("torchlens")
    except PackageNotFoundError:
        return ""


def _git_commit() -> str:
    """Return the current Git commit hash when available.

    Returns
    -------
    str
        Commit hash or an empty string outside Git.
    """

    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database for read-only row access.

    Parameters
    ----------
    db_path:
        SQLite database path.

    Returns
    -------
    sqlite3.Connection
        Row-factory connection.
    """

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Return available columns for a SQLite table or view.

    Parameters
    ----------
    connection:
        SQLite connection.
    table:
        Table or view name.

    Returns
    -------
    set[str]
        Available column names.
    """

    return {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}


def load_current_verification_rows(db_path: Path = VERIFICATION_DB) -> dict[str, dict[str, Any]]:
    """Load latest verification rows keyed by stable ID.

    Parameters
    ----------
    db_path:
        Verification ledger path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Latest verification rows keyed by ``stable_id``.
    """

    if not db_path.exists():
        return {}
    with _connect_readonly(db_path) as connection:
        columns = _table_columns(connection, "verification_runs")
        if not columns:
            return {}
        rows = connection.execute(
            """
            WITH ranked AS (
                SELECT
                    verification_runs.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY stable_id
                        ORDER BY finished_at DESC, run_id DESC
                    ) AS rn
                FROM verification_runs
            )
            SELECT * FROM ranked WHERE rn = 1
            """
        ).fetchall()
    return {str(row["stable_id"]): dict(row) for row in rows}


def load_trace_summary_rows(db_path: Path = TRACE_SUMMARY_DB) -> dict[str, dict[str, Any]]:
    """Load trace-summary rows keyed by stable ID.

    Parameters
    ----------
    db_path:
        Trace-summary SQLite path.

    Returns
    -------
    dict[str, dict[str, Any]]
        Trace-summary rows keyed by ``stable_id``.
    """

    if not db_path.exists():
        return {}
    with _connect_readonly(db_path) as connection:
        columns = _table_columns(connection, "trace_summaries")
        if not columns:
            return {}
        rows = connection.execute("SELECT * FROM trace_summaries").fetchall()
    return {str(row["stable_id"]): dict(row) for row in rows}


def _csv_value(value: Any) -> str:
    """Render one value for CSV output.

    Parameters
    ----------
    value:
        Python value.

    Returns
    -------
    str
        CSV cell value, with nulls as an empty string.
    """

    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _json_object(value: Any) -> Any:
    """Return a JSON value from a persisted SQLite value.

    Parameters
    ----------
    value:
        JSON text, Python object, or ``None``.

    Returns
    -------
    Any
        JSON value, or ``None`` when absent.
    """

    if value is None or value == "":
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json_dumps(value: Any) -> str:
    """Serialize JSON values deterministically.

    Parameters
    ----------
    value:
        JSON-serializable value.

    Returns
    -------
    str
        Compact JSON string.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _display_name(row: CatalogRow) -> str:
    """Return the public display name for a catalog row.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str
        Display name.
    """

    return row.name.replace("_", " ")


def _reference_batch_size(input_shape: str) -> int | None:
    """Parse the primary tensor batch size from an input-shape string.

    Parameters
    ----------
    input_shape:
        Catalog input shape string.

    Returns
    -------
    int | None
        First numeric dimension, or ``None`` when unavailable.
    """

    match = _SHAPE_NUMBER_RE.search(input_shape)
    if match is None:
        return None
    return int(match.group(0))


def _size_variant(row: CatalogRow) -> str:
    """Infer a natural size token from catalog identity text.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str
        Size token or an empty string.
    """

    match = _SIZE_TOKEN_RE.search(f"{row.name} {row.variant}")
    return match.group(1).lower() if match is not None else ""


def _model_scale_bucket(n_params: Any) -> str:
    """Bucket a parameter count by the schema edges.

    Parameters
    ----------
    n_params:
        Parameter count.

    Returns
    -------
    str
        Scale bucket, or an empty string when unavailable.
    """

    if n_params in (None, ""):
        return ""
    value = int(n_params)
    if value < 1_000_000:
        return "<1M"
    if value < 10_000_000:
        return "1-10M"
    if value < 100_000_000:
        return "10-100M"
    if value < 1_000_000_000:
        return "100M-1B"
    return "1B+"


def _install_difficulty(row: CatalogRow) -> str:
    """Infer install-difficulty from catalog dependency hints.

    Parameters
    ----------
    row:
        Catalog row.

    Returns
    -------
    str
        Schema install-difficulty enum.
    """

    if row.source == "classics" or row.zoo in {"torchvision", "torchvision.models"}:
        return "core"
    if row.quarantine:
        return "source-build"
    if row.verified:
        return "pip"
    return "unavailable"


def _row_quality_flags(ledger_row: Mapping[str, Any] | None, has_svg: bool) -> list[str]:
    """Return artifact row quality flags.

    Parameters
    ----------
    ledger_row:
        Latest verification row.
    has_svg:
        Whether a ledger SVG hash is present.

    Returns
    -------
    list[str]
        Deterministic flag list.
    """

    flags: list[str] = []
    if ledger_row is None:
        flags.append("not_verified")
    elif not has_svg:
        flags.append("no_artifact")
    if ledger_row is not None and ledger_row.get("status") not in {"passed", None}:
        flags.append("verification_not_passed")
    return flags


def _compatible_version(ledger_version: Any, current_version: str) -> bool:
    """Return whether a ledger TorchLens version is compatibility-trusted.

    Parameters
    ----------
    ledger_version:
        Ledger version value.
    current_version:
        Current TorchLens version.

    Returns
    -------
    bool
        Whether the versions are treated as compatible.
    """

    if ledger_version in (None, ""):
        return False
    try:
        ledger_parsed = Version(str(ledger_version))
        current_parsed = Version(current_version)
    except InvalidVersion:
        return True
    return ledger_parsed.major == current_parsed.major


def _is_trustworthy(ledger_row: Mapping[str, Any] | None, current_version: str) -> bool:
    """Return the schema trust predicate for a latest ledger row.

    Parameters
    ----------
    ledger_row:
        Latest verification row.
    current_version:
        Current TorchLens version.

    Returns
    -------
    bool
        Whether the row is trustworthy.
    """

    if ledger_row is None:
        return False
    return (
        ledger_row.get("forward_pass") == 1
        and ledger_row.get("metadata_ok") == 1
        and ledger_row.get("n_ops") is not None
        and ledger_row.get("graph_shape_hash") not in (None, "")
        and _compatible_version(ledger_row.get("torchlens_version"), current_version)
    )


def _validated_on_current_release(
    row: CatalogRow,
    ledger_row: Mapping[str, Any] | None,
    current_version: str,
) -> bool:
    """Return the strict current-release validation predicate.

    Parameters
    ----------
    row:
        Catalog row.
    ledger_row:
        Latest verification row.
    current_version:
        Current TorchLens version.

    Returns
    -------
    bool
        Whether the latest ledger run matches the current release and recipe.
    """

    if not _is_trustworthy(ledger_row, current_version) or ledger_row is None:
        return False
    return (
        ledger_row.get("torchlens_version") == current_version
        and ledger_row.get("recipe_revision_sha256") == row.recipe_revision_sha256
    )


def _trust_tier(
    row: CatalogRow,
    ledger_row: Mapping[str, Any] | None,
    current_version: str,
) -> str:
    """Return the graded public trust tier.

    Parameters
    ----------
    row:
        Catalog row.
    ledger_row:
        Latest verification row.
    current_version:
        Current TorchLens version.

    Returns
    -------
    str
        Trust-tier enum.
    """

    if _validated_on_current_release(row, ledger_row, current_version):
        return "current_verified"
    if _is_trustworthy(ledger_row, current_version):
        return "stale_recipe"
    if ledger_row is None:
        return "legacy_seed" if row.verified else "not_verified"
    status = str(ledger_row.get("status") or "not_verified")
    if status in {"failed", "timeout", "error", "install_failed", "oom", "native_crash", "killed"}:
        return "failed"
    if status in {"deferred", "skipped"}:
        return "deferred"
    if status in {"not_applicable", "env_unavailable"}:
        return "unsupported"
    return "not_verified"


def _ledger_status(ledger_row: Mapping[str, Any] | None) -> str:
    """Return the raw validation status with the schema sentinel.

    Parameters
    ----------
    ledger_row:
        Latest verification row.

    Returns
    -------
    str
        Ledger status or ``not_verified``.
    """

    if ledger_row is None:
        return "not_verified"
    status = str(ledger_row.get("status") or "not_verified")
    if status in STATUS_VALUES:
        return status
    return "not_verified"


def _stable_trace_value(trace_row: Mapping[str, Any] | None, key: str | None) -> Any:
    """Return one trace-summary value while preserving missing rows as null.

    Parameters
    ----------
    trace_row:
        Trace-summary row.
    key:
        Trace-summary key.

    Returns
    -------
    Any
        Value or ``None``.
    """

    if trace_row is None or key is None:
        return None
    return trace_row.get(key)


def build_flagship_row(
    row: CatalogRow,
    ledger_row: Mapping[str, Any] | None,
    trace_row: Mapping[str, Any] | None,
    schema_columns: Sequence[SchemaColumn],
    current_version: str,
) -> dict[str, Any]:
    """Build one flagship ``menagerie.csv`` row.

    Parameters
    ----------
    row:
        Catalog row.
    ledger_row:
        Latest ledger row.
    trace_row:
        Trace-summary row.
    schema_columns:
        Ordered schema columns.
    current_version:
        Current TorchLens version.

    Returns
    -------
    dict[str, Any]
        Flagship row keyed by schema column name.
    """

    values: dict[str, Any] = {
        "stable_id": row.stable_id,
        "display_name": _display_name(row),
        "family_normalized": row.family_normalized,
        "variant": row.variant,
        "zoo": row.zoo,
        "domain": row.domain,
        "era": row.era,
        "validation_status": _ledger_status(ledger_row),
        "is_trustworthy": _is_trustworthy(ledger_row, current_version),
        "n_ops": None if ledger_row is None else ledger_row.get("n_ops"),
        "family": row.family,
        "size_variant": _size_variant(row),
        "input_shape": row.input_shape,
        "input_dtype": row.input_dtype,
        "reference_batch_size": _reference_batch_size(row.input_shape),
        "install_difficulty": _install_difficulty(row),
        "trust_tier": _trust_tier(row, ledger_row, current_version),
        "validated_on_current_release": _validated_on_current_release(
            row, ledger_row, current_version
        ),
        "catalog_verified_hint": row.verified,
        "dedup_architecture_key": _stable_trace_value(trace_row, "graph_shape_hash"),
        "svg_url": "",
    }
    for column, trace_key in _FLAGSHIP_TRACE_COLUMNS.items():
        values[column] = _stable_trace_value(trace_row, trace_key)
    values["model_scale_bucket"] = _model_scale_bucket(values.get("n_params"))
    for column in _EXTERNAL_FLAGSHIP_COLUMNS:
        values[column] = ""
    return {column.name: values.get(column.name, "") for column in schema_columns}


def _csv_rows(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, str]]:
    """Render rows as CSV string dictionaries.

    Parameters
    ----------
    rows:
        Source rows.
    columns:
        Output columns.

    Returns
    -------
    list[dict[str, str]]
        CSV rows.
    """

    return [{column: _csv_value(row.get(column)) for column in columns} for row in rows]


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write a CSV file with deterministic null rendering.

    Parameters
    ----------
    path:
        Output path.
    columns:
        Header columns.
    rows:
        Row mappings.

    Returns
    -------
    Path
        Written path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_rows(rows, columns))
    return path


def _pa_type_for(column: str) -> pa.DataType:
    """Return a Parquet type for a trace-metrics column.

    Parameters
    ----------
    column:
        Column name.

    Returns
    -------
    pa.DataType
        Arrow data type.
    """

    if column == "stable_id" or column in {
        "dominant_op_type",
        "model_class_name",
        "model_class_qualname",
        "primary_param_dtype",
        "param_dtype_set_json",
    }:
        return pa.string()
    if column in {
        "branching_factor",
        "pct_conv",
        "pct_linear",
        "pct_attention",
        "pct_norm",
        "pct_elementwise",
        "pct_reduction",
        "pct_reshape",
        "pct_embedding",
        "pct_pooling",
        "flops_coverage_pct",
        "param_memory_mb",
        "activation_memory_mb",
        "forward_peak_memory_mb",
    }:
        return pa.float64()
    return pa.int64()


def _pa_value(value: Any, data_type: pa.DataType) -> Any:
    """Coerce one SQLite value for Arrow array construction.

    Parameters
    ----------
    value:
        SQLite value.
    data_type:
        Target Arrow type.

    Returns
    -------
    Any
        Coerced value, preserving nulls.
    """

    if value in (None, ""):
        return None
    if pa.types.is_integer(data_type):
        return int(value)
    if pa.types.is_floating(data_type):
        return float(value)
    return str(value)


def write_trace_metrics(
    path: Path,
    catalog_rows: Sequence[CatalogRow],
    trace_rows: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write ``trace_metrics.parquet``.

    Parameters
    ----------
    path:
        Output Parquet path.
    catalog_rows:
        Catalog rows defining row order.
    trace_rows:
        Trace-summary rows keyed by stable ID.

    Returns
    -------
    Path
        Written path.
    """

    arrays = {}
    for column in TRACE_METRICS_COLUMNS:
        data_type = _pa_type_for(column)
        values = []
        for row in catalog_rows:
            trace_row = trace_rows.get(row.stable_id)
            raw_value = (
                row.stable_id if column == "stable_id" else _stable_trace_value(trace_row, column)
            )
            values.append(_pa_value(raw_value, data_type))
        arrays[column] = pa.array(values, type=data_type)
    table = pa.table(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_trace_histograms(
    path: Path,
    trace_rows: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write ``trace_histograms.jsonl``.

    Parameters
    ----------
    path:
        Output JSONL path.
    trace_rows:
        Trace-summary rows keyed by stable ID.

    Returns
    -------
    Path
        Written path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for stable_id in sorted(trace_rows):
            trace_row = trace_rows[stable_id]
            record = {
                "stable_id": stable_id,
                "op_type_histogram": _json_object(trace_row.get("op_type_histogram")),
                "module_type_histogram": _json_object(trace_row.get("module_type_histogram")),
                "top_module_types_json": _json_object(trace_row.get("top_module_types_json")),
                "top_level_block_sequence_json": _json_object(
                    trace_row.get("top_level_block_sequence_json")
                ),
                "flops_by_op_type": _json_object(trace_row.get("flops_by_op_type")),
                "macs_by_op_type": _json_object(trace_row.get("macs_by_op_type")),
                "structural_barcode": _json_object(trace_row.get("structural_barcode_json")),
            }
            handle.write(_json_dumps(record) + "\n")
    return path


def build_artifact_rows(
    catalog_rows: Sequence[CatalogRow],
    ledger_rows: Mapping[str, Mapping[str, Any]],
    *,
    dataset_as_of_date: str,
    git_commit: str,
) -> list[dict[str, Any]]:
    """Build artifact side-table rows.

    Parameters
    ----------
    catalog_rows:
        Catalog rows.
    ledger_rows:
        Ledger rows keyed by stable ID.
    dataset_as_of_date:
        Build date.
    git_commit:
        Generation commit hash.

    Returns
    -------
    list[dict[str, Any]]
        Artifact rows.
    """

    rows: list[dict[str, Any]] = []
    for row in catalog_rows:
        ledger_row = ledger_rows.get(row.stable_id)
        svg_sha256 = "" if ledger_row is None else str(ledger_row.get("svg_sha256") or "")
        has_svg = bool(svg_sha256)
        rows.append(
            {
                "stable_id": row.stable_id,
                "model_page_url": "",
                "svg_path": "",
                "svg_url": "",
                "svg_sha256": svg_sha256,
                "has_svg": has_svg,
                "tlspec_path": "",
                "model_card_url": "",
                "added_wave": row.source,
                "row_quality_flags_json": _json_dumps(_row_quality_flags(ledger_row, has_svg)),
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "dataset_as_of_date": dataset_as_of_date,
                "csv_generation_commit_sha": git_commit,
            }
        )
    return rows


def export_menagerie_csvs(
    out_dir: Path,
    *,
    catalog_db: Path = CATALOG_DB,
    verification_db: Path = VERIFICATION_DB,
    trace_summary_db: Path = TRACE_SUMMARY_DB,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    dictionary_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Path]:
    """Export the flagship CSV, side-tables, and data dictionary.

    Parameters
    ----------
    out_dir:
        Output directory.
    catalog_db:
        Catalog database path.
    verification_db:
        Verification ledger path.
    trace_summary_db:
        Trace-summary database path.
    schema_path:
        Authoritative schema path.
    dictionary_path:
        Optional data dictionary path. Defaults inside ``out_dir``.
    limit:
        Optional catalog row limit for tests and smoke subsets.

    Returns
    -------
    dict[str, Path]
        Written artifact paths keyed by logical name.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    schema_columns = parse_flagship_schema(schema_path)
    catalog_rows = load_rows(limit=limit, db_path=catalog_db)
    ledger_rows = load_current_verification_rows(verification_db)
    trace_rows = load_trace_summary_rows(trace_summary_db)
    current_version = _current_torchlens_version()
    dataset_as_of_date = datetime.now(UTC).date().isoformat()
    git_commit = _git_commit()
    flagship_rows = [
        build_flagship_row(
            row,
            ledger_rows.get(row.stable_id),
            trace_rows.get(row.stable_id),
            schema_columns,
            current_version,
        )
        for row in catalog_rows
    ]
    dictionary_output = dictionary_path or out_dir / "DATA_DICTIONARY.md"
    from menagerie.csv_dictionary import write_data_dictionary

    paths = {
        "menagerie": write_csv(
            out_dir / "menagerie.csv",
            [column.name for column in schema_columns],
            flagship_rows,
        ),
        "trace_metrics": write_trace_metrics(
            out_dir / "trace_metrics.parquet", catalog_rows, trace_rows
        ),
        "trace_histograms": write_trace_histograms(out_dir / "trace_histograms.jsonl", trace_rows),
        "papers": write_csv(out_dir / "papers.csv", PAPERS_COLUMNS, []),
        "lineage": write_csv(out_dir / "lineage.csv", LINEAGE_COLUMNS, []),
        "artifacts": write_csv(
            out_dir / "artifacts.csv",
            ARTIFACTS_COLUMNS,
            build_artifact_rows(
                catalog_rows,
                ledger_rows,
                dataset_as_of_date=dataset_as_of_date,
                git_commit=git_commit,
            ),
        ),
        "dictionary": write_data_dictionary(dictionary_output, schema_path),
    }
    return paths


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--catalog-db", type=Path, default=CATALOG_DB, help="Catalog DB path.")
    parser.add_argument(
        "--verification-db",
        type=Path,
        default=VERIFICATION_DB,
        help="Verification ledger DB path.",
    )
    parser.add_argument(
        "--trace-summary-db",
        type=Path,
        default=TRACE_SUMMARY_DB,
        help="Trace-summary DB path.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Authoritative schema Markdown path.",
    )
    parser.add_argument(
        "--dictionary-path",
        type=Path,
        default=None,
        help="Optional data dictionary Markdown path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CSV exporter command line.

    Parameters
    ----------
    argv:
        Optional argument list.

    Returns
    -------
    int
        Process exit status.
    """

    args = _build_arg_parser().parse_args(argv)
    export_menagerie_csvs(
        args.out_dir,
        catalog_db=args.catalog_db,
        verification_db=args.verification_db,
        trace_summary_db=args.trace_summary_db,
        schema_path=args.schema_path,
        dictionary_path=args.dictionary_path,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
