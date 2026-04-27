"""Smoke tests for fastlog dry-run live visualization helpers."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog.types import RecordContext


def _mlp() -> nn.Module:
    """Return a small two-layer MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def _keep_linear(ctx: RecordContext) -> bool:
    """Keep linear operation events."""

    return ctx.kind == "op" and ctx.layer_type == "linear"


def _keep_relu(ctx: RecordContext) -> bool:
    """Keep relu operation events."""

    return ctx.kind == "op" and ctx.layer_type == "relu"


def test_print_tree_outputs_unicode_rows() -> None:
    """Dry-run print_tree returns non-empty unicode-indented text."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)
    text = trace.print_tree()
    assert text
    assert "└─" in text


def test_to_pandas_expected_columns() -> None:
    """Dry-run to_pandas returns the public event columns."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)
    frame = trace.to_pandas()
    assert list(frame.columns) == [
        "pass_num",
        "step_num",
        "kind",
        "op_type",
        "module_address",
        "shape",
        "dtype",
    ]


def test_repredicate_updates_decisions_without_replacing_events() -> None:
    """Repredicate returns a new trace with shared events and changed decisions."""

    trace = tl.fastlog.dry_run(_mlp(), torch.randn(1, 4), keep_op=_keep_linear)
    updated = trace.repredicate(other_keep_op=_keep_relu)
    assert updated is not trace
    assert updated.events is trace.events
    assert updated.decisions != trace.decisions
