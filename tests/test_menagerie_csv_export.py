"""Tests for public menagerie CSV and side-table export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sqlite3
from typing import Any

import pyarrow.parquet as pq

from menagerie.catalog import CatalogRow, write_catalog
from menagerie.csv_dictionary import DEFAULT_SCHEMA_PATH, SIDE_TABLE_COLUMNS, parse_flagship_schema
from menagerie.csv_export import (
    ARTIFACTS_COLUMNS,
    LINEAGE_COLUMNS,
    PAPERS_COLUMNS,
    TRACE_HISTOGRAM_COLUMNS,
    TRACE_METRICS_COLUMNS,
    export_menagerie_csvs,
    _is_trustworthy,
)
from menagerie.op_taxonomy import OP_TAXONOMY_VERSION


def _fixture_rows() -> list[CatalogRow]:
    """Return catalog rows for CSV export fixtures.

    Returns
    -------
    list[CatalogRow]
        Fixture catalog rows.
    """

    return [
        CatalogRow(
            model_id=1,
            display_index=1,
            stable_id="m8840",
            name="resnet18",
            variant="",
            family="resnet",
            family_normalized="ResNet",
            domain="vision/classification-backbone",
            zoo="torchvision",
            constructor_call="torchvision.models.resnet18(weights=None)",
            input_shape="(1,3,224,224)",
            input_dtype="float32",
            era="2015",
            verified=True,
            notes="fixture",
            source="catalog",
            recipe_revision_sha256="current-recipe",
            input_is_real=True,
            verification_expectation="forward_required",
            quarantine=False,
        ),
        CatalogRow(
            model_id=2,
            display_index=2,
            stable_id="m_missing",
            name="missing_summary",
            variant="tiny",
            family="missing",
            family_normalized="Missing",
            domain="vision/classification-backbone",
            zoo="example-zoo",
            constructor_call="model=Missing()",
            input_shape="(2, 3, 16, 16)",
            input_dtype="float32",
            era="2026",
            verified=False,
            notes="fixture",
            source="catalog",
            recipe_revision_sha256="missing-recipe",
            input_is_real=True,
            verification_expectation="forward_required",
            quarantine=True,
        ),
    ]


def _write_verification_db(path: Path) -> None:
    """Write a fixture verification ledger.

    Parameters
    ----------
    path:
        SQLite database path.
    """

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE verification_runs(
                run_id TEXT PRIMARY KEY,
                stable_id TEXT NOT NULL,
                recipe_revision_sha256 TEXT NOT NULL,
                name TEXT NOT NULL,
                zoo TEXT NOT NULL,
                variant TEXT NOT NULL DEFAULT '',
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                forward_pass INTEGER,
                backward_pass INTEGER,
                backward_na_reason TEXT,
                metadata_ok INTEGER,
                n_ops INTEGER,
                graph_shape_hash TEXT,
                svg_sha256 TEXT,
                torchlens_version TEXT NOT NULL,
                torch_version TEXT NOT NULL,
                python_version TEXT NOT NULL,
                device_requested TEXT NOT NULL,
                device_actual TEXT,
                env_hash TEXT,
                runner_host TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                peak_rss_mb INTEGER,
                error_class TEXT,
                error_message TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO verification_runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "old",
                    "m8840",
                    "current-recipe",
                    "resnet18",
                    "torchvision",
                    "",
                    "forward",
                    "failed",
                    0,
                    None,
                    "",
                    0,
                    None,
                    "",
                    "",
                    "2.27.0",
                    "2.9.0",
                    "CPython 3.11",
                    "cpu",
                    "cpu",
                    "",
                    "host",
                    "2026-06-23T00:00:00+00:00",
                    "2026-06-23T00:01:00+00:00",
                    60.0,
                    100,
                    "RuntimeError",
                    "old failure",
                ),
                (
                    "new",
                    "m8840",
                    "current-recipe",
                    "resnet18",
                    "torchvision",
                    "",
                    "forward",
                    "passed",
                    1,
                    None,
                    "",
                    1,
                    69,
                    "graph-hash",
                    "svg-hash",
                    "2.27.0",
                    "2.9.0",
                    "CPython 3.11",
                    "cpu",
                    "cpu",
                    "",
                    "host",
                    "2026-06-23T02:00:00+00:00",
                    "2026-06-23T02:01:00+00:00",
                    60.0,
                    110,
                    "",
                    "",
                ),
                (
                    "missing",
                    "m_missing",
                    "missing-recipe",
                    "missing_summary",
                    "example-zoo",
                    "tiny",
                    "forward",
                    "oom",
                    0,
                    None,
                    "",
                    0,
                    None,
                    "",
                    "",
                    "2.27.0",
                    "2.9.0",
                    "CPython 3.11",
                    "cpu",
                    "cpu",
                    "",
                    "host",
                    "2026-06-23T03:00:00+00:00",
                    "2026-06-23T03:01:00+00:00",
                    60.0,
                    120,
                    "OutOfMemoryError",
                    "oom",
                ),
            ],
        )


def _write_trace_summary_db(path: Path, recipe_sha256: str = "current-recipe") -> None:
    """Write a fixture trace-summary database with only m8840 populated.

    Parameters
    ----------
    path:
        SQLite database path.
    recipe_sha256:
        Recipe revision hash stamped on the trace summary.
    """

    columns = [
        "stable_id",
        "trace_summary_version",
        "op_taxonomy_version",
        "recipe_revision_sha256",
        "torchlens_version",
        "n_params_source",
        "n_compute_ops",
        "n_unique_op_types",
        "n_inplace_ops",
        "has_custom_op",
        "op_type_histogram",
        "dominant_op_type",
        "top_op_types_json",
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
        "module_type_histogram",
        "top_module_types_json",
        "top_level_block_sequence_json",
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
        "flops_by_op_type",
        "macs_by_op_type",
        "activation_memory_bytes",
        "forward_peak_memory_bytes",
        "largest_activation_bytes",
        "param_memory_mb",
        "activation_memory_mb",
        "forward_peak_memory_mb",
        "output_shape",
        "n_output_tensors",
        "output_container_kind",
        "has_attention",
        "has_conv",
        "has_embedding",
        "has_residual",
        "has_self_attention",
        "has_cross_attention",
        "norm_type",
        "activation_fn_type",
        "pooling_type",
        "structural_barcode_json",
        "graph_shape_hash",
        "structural_fingerprint_hash",
    ]
    values: dict[str, Any] = {column: None for column in columns}
    values.update(
        {
            "stable_id": "m8840",
            "trace_summary_version": "1.0.0",
            "op_taxonomy_version": OP_TAXONOMY_VERSION,
            "recipe_revision_sha256": recipe_sha256,
            "torchlens_version": "2.27.0",
            "n_params_source": "traced",
            "n_compute_ops": 69,
            "n_unique_op_types": 8,
            "n_inplace_ops": 25,
            "has_custom_op": 0,
            "op_type_histogram": json.dumps({"conv2d": 20, "batch_norm": 20}, sort_keys=True),
            "dominant_op_type": "conv",
            "top_op_types_json": json.dumps(["conv2d", "batch_norm"]),
            "pct_conv": 28.985507246376812,
            "pct_linear": 1.4492753623188406,
            "pct_attention": 0.0,
            "pct_norm": 28.985507246376812,
            "pct_elementwise": 36.231884057971016,
            "pct_reduction": 0.0,
            "pct_reshape": 1.4492753623188406,
            "pct_embedding": 0.0,
            "pct_pooling": 2.898550724637681,
            "graph_depth": 63,
            "graph_max_width": 2,
            "branching_factor": 1.0,
            "max_fan_out": 2,
            "max_fan_in": 2,
            "is_branching": 1,
            "is_recurrent": 0,
            "max_recurrence_iters": 1,
            "n_recurrent_layers": 0,
            "has_conditional_branching": 0,
            "n_conditionals": 0,
            "is_dynamic_graph": 0,
            "n_modules": 68,
            "n_module_calls": 68,
            "module_max_depth": 3,
            "n_top_level_modules": 10,
            "n_unique_module_types": 6,
            "module_type_histogram": json.dumps({"Conv2d": 20}, sort_keys=True),
            "top_module_types_json": json.dumps(["Conv2d"]),
            "top_level_block_sequence_json": json.dumps(["Conv2d", "BatchNorm2d"]),
            "model_class_name": "ResNet",
            "model_class_qualname": "torchvision.models.resnet.ResNet",
            "n_params": 11689512,
            "n_params_trainable": 11689512,
            "n_params_frozen": 0,
            "n_param_tensors": 62,
            "param_memory_bytes": 46758048,
            "primary_param_dtype": "torch.float32",
            "param_dtype_set_json": json.dumps(["torch.float32"]),
            "quantized_param_tensor_count": 0,
            "has_frozen_params": 0,
            "n_buffers": 60,
            "buffer_memory_bytes": 3840,
            "buffer_overwrite_count": 0,
            "total_flops_forward": 3628877824,
            "total_macs_forward": 1814438912,
            "total_flops_backward": None,
            "total_macs_backward": None,
            "flops_coverage_pct": 100.0,
            "n_unknown_flops_ops": 0,
            "flops_by_op_type": json.dumps({"conv": 3600000000}, sort_keys=True),
            "macs_by_op_type": json.dumps({"conv": 1800000000}, sort_keys=True),
            "activation_memory_bytes": 1000,
            "forward_peak_memory_bytes": 2000,
            "largest_activation_bytes": 300,
            "param_memory_mb": 44.591949462890625,
            "activation_memory_mb": 0.00095367431640625,
            "forward_peak_memory_mb": 0.0019073486328125,
            "output_shape": "(1,1000)",
            "n_output_tensors": 1,
            "output_container_kind": "tensor",
            "has_attention": 0,
            "has_conv": 1,
            "has_embedding": 0,
            "has_residual": 1,
            "has_self_attention": 0,
            "has_cross_attention": 0,
            "norm_type": "batch_norm",
            "activation_fn_type": "relu",
            "pooling_type": "adaptive_avg_pool",
            "structural_barcode_json": json.dumps(
                {"graph_depth": 63, "has_conv": True, "n_params": 11689512},
                sort_keys=True,
            ),
            "graph_shape_hash": "graph-hash",
            "structural_fingerprint_hash": "fingerprint-hash",
        }
    )
    column_defs = ", ".join(f"{column} TEXT" for column in columns)
    with sqlite3.connect(path) as connection:
        connection.execute(f"CREATE TABLE trace_summaries ({column_defs})")
        placeholders = ", ".join("?" for _column in columns)
        connection.execute(
            f"INSERT INTO trace_summaries ({', '.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV as dictionaries.

    Parameters
    ----------
    path:
        CSV path.

    Returns
    -------
    list[dict[str, str]]
        CSV rows.
    """

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_public_csv_export_schema_join_and_dictionary(tmp_path: Path) -> None:
    """CSV export preserves schema order, joins side tables, and documents every column."""

    catalog_db = tmp_path / "catalog.db"
    verification_db = tmp_path / "verification.db"
    trace_summary_db = tmp_path / "trace_summary.db"
    write_catalog(_fixture_rows(), canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)
    _write_verification_db(verification_db)
    _write_trace_summary_db(trace_summary_db)

    out_dir = tmp_path / "out"
    paths = export_menagerie_csvs(
        out_dir,
        catalog_db=catalog_db,
        verification_db=verification_db,
        trace_summary_db=trace_summary_db,
    )

    schema_columns = parse_flagship_schema(DEFAULT_SCHEMA_PATH)
    expected_header = [column.name for column in schema_columns]
    with paths["menagerie"].open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == expected_header
    assert len(expected_header) == 78

    rows = {row["stable_id"]: row for row in _read_csv(paths["menagerie"])}
    resnet = rows["m8840"]
    missing = rows["m_missing"]
    assert resnet["validation_status"] == "passed"
    assert resnet["is_trustworthy"] == "1"
    assert resnet["trust_tier"] == "current_verified"
    assert resnet["validated_on_current_release"] == "1"
    assert resnet["catalog_verified_hint"] == "1"
    assert resnet["n_ops"] == "69"
    assert resnet["n_params"] == "11689512"
    assert resnet["n_params_source"] == "traced"
    assert resnet["graph_depth"] == "63"
    assert resnet["has_conv"] == "1"
    assert resnet["dedup_architecture_key"] == "graph-hash"
    assert resnet["svg_url"].endswith("?sha256=svg-hash")

    assert missing["validation_status"] == "oom"
    assert missing["is_trustworthy"] == "0"
    assert missing["trust_tier"] == "failed"
    assert missing["n_params"] == ""
    assert missing["n_params_source"] == ""
    assert missing["graph_depth"] == ""
    assert missing["has_conv"] == ""

    metric_table = pq.read_table(paths["trace_metrics"])
    assert metric_table.column_names == list(TRACE_METRICS_COLUMNS)
    assert metric_table.column("stable_id").to_pylist() == ["m8840", "m_missing"]
    assert metric_table.column("n_params").to_pylist() == [11689512, None]

    histogram_lines = paths["trace_histograms"].read_text(encoding="utf-8").splitlines()
    assert len(histogram_lines) == 1
    histogram_row = json.loads(histogram_lines[0])
    assert set(histogram_row) == set(TRACE_HISTOGRAM_COLUMNS)
    assert histogram_row["stable_id"] == "m8840"
    assert histogram_row["op_type_histogram"]["conv2d"] == 20

    assert next(csv.reader(paths["papers"].open(encoding="utf-8", newline=""))) == list(
        PAPERS_COLUMNS
    )
    assert next(csv.reader(paths["lineage"].open(encoding="utf-8", newline=""))) == list(
        LINEAGE_COLUMNS
    )
    assert next(csv.reader(paths["artifacts"].open(encoding="utf-8", newline=""))) == list(
        ARTIFACTS_COLUMNS
    )
    artifact_rows = {row["stable_id"]: row for row in _read_csv(paths["artifacts"])}
    assert artifact_rows["m8840"]["has_svg"] == "1"
    assert artifact_rows["m8840"]["svg_url"].endswith("?sha256=svg-hash")
    assert artifact_rows["m8840"]["added_wave"] == ""
    assert artifact_rows["m8840"]["op_taxonomy_version"] == OP_TAXONOMY_VERSION
    assert artifact_rows["m_missing"]["has_svg"] == "0"

    dictionary = paths["dictionary"].read_text(encoding="utf-8")
    emitted_side_columns = {
        f"`{table}.{column}`" for table, columns in SIDE_TABLE_COLUMNS.items() for column in columns
    }
    assert dictionary.count("\n## ") == 78 + sum(
        len(columns) for columns in SIDE_TABLE_COLUMNS.values()
    )
    for column_entry in emitted_side_columns:
        assert column_entry in dictionary
    assert "graph_depth = max(op.max_distance_from_input)" in dictionary
    assert "is_trustworthy = forward_pass=1" in dictionary
    assert "display_name = catalog name with underscores replaced by spaces" in dictionary
    assert "install_difficulty = runtime dependency classification" in dictionary


def test_stale_trace_summary_nulls_retrace_fields_and_side_tables(tmp_path: Path) -> None:
    """Stale trace summaries must not populate authoritative retrace-derived measurements."""

    catalog_db = tmp_path / "catalog.db"
    verification_db = tmp_path / "verification.db"
    trace_summary_db = tmp_path / "trace_summary.db"
    write_catalog(_fixture_rows(), canonical_tsv=tmp_path / "catalog.tsv", db_path=catalog_db)
    _write_verification_db(verification_db)
    _write_trace_summary_db(trace_summary_db, recipe_sha256="old-recipe")

    paths = export_menagerie_csvs(
        tmp_path / "out",
        catalog_db=catalog_db,
        verification_db=verification_db,
        trace_summary_db=trace_summary_db,
    )

    rows = {row["stable_id"]: row for row in _read_csv(paths["menagerie"])}
    assert rows["m8840"]["is_trustworthy"] == "1"
    assert rows["m8840"]["n_ops"] == "69"
    assert rows["m8840"]["n_params"] == ""
    assert rows["m8840"]["n_params_source"] == ""
    assert rows["m8840"]["graph_depth"] == ""
    assert rows["m8840"]["has_conv"] == ""
    assert rows["m8840"]["dedup_architecture_key"] == ""

    metric_table = pq.read_table(paths["trace_metrics"])
    assert metric_table.column("stable_id").to_pylist() == ["m8840", "m_missing"]
    assert metric_table.column("n_params").to_pylist() == [None, None]
    assert metric_table.column("graph_depth").to_pylist() == [None, None]
    assert paths["trace_histograms"].read_text(encoding="utf-8") == ""


def test_version_compat_trust_gate_rejects_missing_and_invalid_versions() -> None:
    """Trust gating fails closed for missing or invalid TorchLens versions."""

    base_row: dict[str, Any] = {
        "forward_pass": 1,
        "metadata_ok": 1,
        "n_ops": 1,
        "graph_shape_hash": "graph-hash",
    }

    assert not _is_trustworthy({**base_row, "torchlens_version": ""}, "2.27.0")
    assert not _is_trustworthy({**base_row, "torchlens_version": "not-a-version"}, "2.27.0")
    assert not _is_trustworthy({**base_row, "torchlens_version": "2.27.0"}, "")
