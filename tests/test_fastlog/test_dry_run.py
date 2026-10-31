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

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), save=_keep_linear)

    assert trace.print_tree().strip()


def test_to_pandas_has_expected_columns() -> None:
    """to_pandas returns the expected public columns."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), save=_keep_linear)

    assert list(trace.to_pandas().columns) == [
        "call_index",
        "step_num",
        "kind",
        "op_type",
        "address",
        "shape",
        "dtype",
    ]


def test_repredicate_changes_decisions_without_changing_events() -> None:
    """repredicate changes decisions while preserving event identity."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), save=_keep_linear)
    updated = trace.repredicate(other_keep_op=_keep_relu)

    assert updated.events is trace.events
    assert updated.decisions != trace.decisions


def test_show_graph_renders_without_error(tmp_path: Path) -> None:
    """show_graph returns DOT and Graphviz can render it."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), save=_keep_linear)
    dot = trace.draw(vis_outpath=str(tmp_path / "dry_run"), vis_fileformat="png")

    assert "digraph" in dot
    assert (tmp_path / "dry_run.png").exists()


def test_dry_run_uses_live_predicate_decisions_without_reinvoking() -> None:
    """dry_run reports the same stateful predicate decisions as record()."""

    calls = {"n": 0}

    def first_two_ops(ctx: RecordContext) -> bool:
        """Capture only the first two operation events the predicate sees."""

        calls["n"] += 1
        return calls["n"] <= 2

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    inputs = torch.randn(2, 4)

    trace = tl.fastlog.dry_run(model, inputs, save=first_two_ops)
    dry_run_calls = calls["n"]
    dry_run_labels = [
        ctx.label
        for ctx, decision in zip(trace.contexts, trace.decisions)
        if decision and ctx.kind == "op"
    ]

    calls["n"] = 0
    recording = tl.fastlog.record(model, inputs, save=first_two_ops)
    record_calls = calls["n"]
    kept_labels = [record.ctx.label for record in recording.records]

    assert dry_run_calls == record_calls
    assert dry_run_labels == kept_labels
