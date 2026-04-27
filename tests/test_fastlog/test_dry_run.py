"""Comprehensive dry-run visualization tests for fastlog."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext


def _mlp() -> nn.Module:
    """Return a small static MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def _keep_linear(ctx: RecordContext) -> bool:
    """Keep linear operation events."""

    return ctx.kind == "op" and ctx.layer_type == "linear"


def _keep_relu(ctx: RecordContext) -> bool:
    """Keep relu operation events."""

    return ctx.kind == "op" and ctx.layer_type == "relu"


def test_print_tree_has_non_empty_output() -> None:
    """print_tree returns non-empty tree text."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)

    assert trace.print_tree().strip()


def test_to_pandas_has_expected_columns() -> None:
    """to_pandas returns the expected public columns."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)

    assert list(trace.to_pandas().columns) == [
        "pass_num",
        "step_num",
        "kind",
        "op_type",
        "module_address",
        "shape",
        "dtype",
    ]


def test_repredicate_changes_decisions_without_changing_events() -> None:
    """repredicate changes decisions while preserving event identity."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)
    updated = trace.repredicate(other_keep_op=_keep_relu)

    assert updated.events is trace.events
    assert updated.decisions != trace.decisions


def test_show_graph_renders_without_error(tmp_path: Path) -> None:
    """show_graph returns DOT and Graphviz can render it."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)
    dot = trace.show_graph(vis_outpath=str(tmp_path / "dry_run"), vis_fileformat="png")

    assert "digraph" in dot
    assert (tmp_path / "dry_run.png").exists()
