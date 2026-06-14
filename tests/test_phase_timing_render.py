"""Tests for render phase timing buckets."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    """Tiny differentiable model for render timing tests."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        return torch.relu(self.linear(x)).sum()


def _assert_render_bucket(trace: tl.Trace, bucket: str) -> None:
    """Assert that a render timing bucket was recorded.

    Parameters
    ----------
    trace:
        Trace expected to contain timing data.
    bucket:
        Timing bucket name to inspect.
    """

    bucket_stats = trace._phase_timings[bucket]
    assert isinstance(bucket_stats["total_s"], float)
    assert bucket_stats["total_s"] >= 0.0
    assert bucket_stats["count"] >= 1


def test_graphviz_render_phase_timings_are_recorded(tmp_path: Path) -> None:
    """Forward, backward, and combined Graphviz renders populate timing buckets."""

    trace = tl.trace(_TinyModel(), torch.randn(2, 3, requires_grad=True), save_grads="all")
    trace.draw(
        vis_outpath=str(tmp_path / "forward"),
        vis_save_only=True,
        vis_fileformat="svg",
    )
    _assert_render_bucket(trace, "render:graphviz:forward")

    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    trace.draw_backward(
        vis_outpath=str(tmp_path / "backward"),
        vis_save_only=True,
        vis_fileformat="svg",
    )
    _assert_render_bucket(trace, "render:graphviz:backward")

    trace.draw_combined(
        vis_outpath=str(tmp_path / "combined"),
        vis_save_only=True,
        vis_fileformat="svg",
    )
    _assert_render_bucket(trace, "render:graphviz:combined")
