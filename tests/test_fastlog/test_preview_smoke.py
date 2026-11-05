"""Smoke tests for fastlog Trace preview rendering."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


def _mlp() -> nn.Module:
    """Return a small two-layer MLP."""

    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))


def test_preview_fastlog_renders_on_mlp(tmp_path: Path) -> None:
    """Preview rendering returns Graphviz source for a small MLP."""

    trace = tl.trace(_mlp(), torch.randn(1, 4))
    dot = trace.preview_fastlog(
        keep_op=lambda ctx: ctx.layer_type == "linear",
        path=str(tmp_path / "preview"),
    )
    assert "digraph" in dot.dot
    assert "fastlog" in dot.dot
