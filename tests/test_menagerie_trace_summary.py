"""Tests for the menagerie trace-summary exporter."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest
import torch
from torch import nn

from menagerie.trace_summary import (
    TRACE_SUMMARY_COLUMNS,
    TRACE_SUMMARY_VERSION,
    ensure_store,
    load_summary,
    persist_summary,
    static_n_params,
    summarize_model,
)


class SmallResidualNet(nn.Module):
    """Small model covering conv, norm, activation, pooling, and residual motifs."""

    def __init__(self) -> None:
        """Initialize the small residual model."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.proj = nn.Conv2d(3, 4, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a small residual forward pass.

        Parameters
        ----------
        inputs:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Logits.
        """

        residual = self.proj(inputs)
        hidden = torch.relu(self.norm(self.conv(inputs)))
        hidden = hidden + residual
        pooled = self.pool(hidden).flatten(1)
        return self.head(pooled)


class ParallelAddNet(nn.Module):
    """Small model with a non-residual same-depth elementwise add."""

    def __init__(self) -> None:
        """Initialize parallel linear branches."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add two sibling branch outputs.

        Parameters
        ----------
        inputs:
            Input feature batch.

        Returns
        -------
        torch.Tensor
            Same-depth branch sum.
        """

        return self.left(inputs) + self.right(inputs)


def _json_column(row: dict[str, Any], key: str) -> Any:
    """Decode one persisted JSON column.

    Parameters
    ----------
    row:
        SQLite row.
    key:
        Column name.

    Returns
    -------
    Any
        Decoded JSON value.
    """

    return json.loads(str(row[key]))


def test_static_n_params_uses_named_parameters_without_trace() -> None:
    """Static fallback matches the direct named-parameter scalar count."""

    model = SmallResidualNet()
    expected = sum(parameter.numel() for _name, parameter in model.named_parameters())
    assert static_n_params(model) == expected


def test_trace_summary_small_model_fields_and_store(tmp_path: Path) -> None:
    """Small-model summaries populate schema fields and persist SQLite values."""

    model = SmallResidualNet()
    example = torch.randn(1, 3, 16, 16)
    summary = summarize_model(
        "small_residual",
        model,
        example,
        "recipe",
        compute_identity_hashes=False,
    )
    assert tuple(summary) == TRACE_SUMMARY_COLUMNS
    assert summary["trace_summary_version"] == TRACE_SUMMARY_VERSION
    assert summary["n_params_source"] == "traced"
    assert summary["n_params"] == static_n_params(model)
    assert summary["n_compute_ops"] > 0
    assert summary["n_unique_op_types"] > 0
    assert summary["graph_depth"] >= 1
    assert summary["graph_max_width"] >= 1
    assert summary["module_max_depth"] >= 1
    assert summary["n_modules"] >= 1
    assert summary["n_buffers"] >= 0
    assert summary["total_flops_forward"] > 0
    assert summary["total_macs_forward"] == summary["total_flops_forward"] // 2
    assert summary["param_memory_bytes"] == summary["param_memory_mb"] * 1_048_576
    assert summary["activation_memory_bytes"] == summary["activation_memory_mb"] * 1_048_576
    assert summary["pct_conv"] > 0.0
    assert summary["pct_norm"] > 0.0
    assert summary["pct_pooling"] > 0.0
    assert summary["has_conv"] is True
    assert summary["norm_type"] == "batch_norm"
    assert summary["activation_fn_type"] == "relu"
    assert isinstance(summary["op_type_histogram"], dict)
    assert isinstance(summary["module_type_histogram"], dict)

    db_path = tmp_path / "trace_summary.db"
    ensure_store(db_path)
    persist_summary(summary, db_path)
    persisted = load_summary("small_residual", db_path)
    assert persisted is not None
    assert persisted["stable_id"] == "small_residual"
    assert persisted["has_conv"] == 1
    assert persisted["n_params"] == summary["n_params"]
    assert _json_column(persisted, "op_type_histogram")
    assert _json_column(persisted, "module_type_histogram")

    with sqlite3.connect(db_path) as connection:
        table_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(trace_summaries)").fetchall()
        ]
    assert tuple(table_columns) == TRACE_SUMMARY_COLUMNS


def test_trace_summary_resnet18_is_deterministic() -> None:
    """ResNet-18 summaries are deterministic and populate core schema metrics."""

    torchvision_models = pytest.importorskip("torchvision.models")
    model_a = torchvision_models.resnet18(weights=None)
    model_b = torchvision_models.resnet18(weights=None)
    example = torch.randn(1, 3, 64, 64)

    summary_a = summarize_model(
        "resnet18",
        model_a,
        example,
        "recipe",
        compute_identity_hashes=False,
    )
    summary_b = summarize_model(
        "resnet18",
        model_b,
        example,
        "recipe",
        compute_identity_hashes=False,
    )

    # forward_peak_memory is a real runtime resource measurement (host RSS delta /
    # tracemalloc peak on CPU, device peak on CUDA), so it legitimately varies run
    # to run. Determinism is a property of the STRUCTURAL summary, not of runtime
    # memory, so compare every field except the memory measurements.
    _runtime_only = {"forward_peak_memory_bytes", "forward_peak_memory_mb"}
    structural_a = {k: v for k, v in summary_a.items() if k not in _runtime_only}
    structural_b = {k: v for k, v in summary_b.items() if k not in _runtime_only}
    assert structural_a == structural_b
    assert summary_a["n_params"] == static_n_params(model_a)
    assert summary_a["n_params_source"] == "traced"
    assert summary_a["has_conv"] is True
    assert summary_a["has_residual"] is True
    assert summary_a["pct_conv"] > 0.0
    assert summary_a["total_flops_forward"] > 0
    assert isinstance(summary_a["forward_peak_memory_bytes"], int)
    # The measurement is now wired (no longer hard-zero) for real CPU traces.
    assert summary_a["forward_peak_memory_bytes"] > 0


def test_same_depth_elementwise_add_is_not_residual() -> None:
    """A sibling branch add is not classified as a residual connection."""

    model = ParallelAddNet()
    example = torch.randn(1, 4)

    summary = summarize_model(
        "parallel_add",
        model,
        example,
        "recipe",
        compute_identity_hashes=False,
    )

    assert summary["has_residual"] is False
