"""Regression tests for ``draw(show_orphans=...)`` island rendering."""

from __future__ import annotations

import graphviz
import torch
from torch import nn

import torchlens as tl


class WithOrphanIsland(nn.Module):
    """Model with a dead-end computation unreachable from inputs and outputs."""

    def __init__(self) -> None:
        """Initialize a small connected path alongside the island."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a connected path plus an orphan ``randn -> mul`` island."""

        _dead = torch.randn(4) * 2.0  # unreachable from both inputs and outputs
        return self.lin(x)


def _orphan_trace() -> tl.Trace:
    """Trace the island model with orphans retained (opt-in)."""

    torch.manual_seed(0)
    return tl.trace(WithOrphanIsland(), torch.ones(1, 4), keep_orphans=True)


def test_show_orphans_renders_island_cluster() -> None:
    """``show_orphans=True`` emits a dashed orphan cluster with the island nodes."""

    trace = _orphan_trace()
    assert [op.label for op in trace._orphan_logs]  # sanity: islands were retained

    dot = trace.draw(show_orphans=True, return_graph=True, vis_save_only=True)
    source = dot.source if isinstance(dot, graphviz.Digraph) else str(dot)

    assert "cluster_orphans" in source
    assert "orphans (unreachable from inputs & outputs)" in source
    assert "orphan__randn_1_1" in source


def test_orphans_hidden_by_default() -> None:
    """Without ``show_orphans`` the island stays out of the rendered graph."""

    trace = _orphan_trace()
    dot = trace.draw(return_graph=True, vis_save_only=True)
    source = dot.source if isinstance(dot, graphviz.Digraph) else str(dot)

    assert "cluster_orphans" not in source
    assert "orphan__randn_1_1" not in source
