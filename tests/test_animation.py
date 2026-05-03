"""Tests for Phase 7 temporal visualization helpers."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class RecurrentLinear(nn.Module):
    """Small repeated-module model for animation tests."""

    def __init__(self) -> None:
        """Initialize the repeated linear layer."""

        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same cell multiple times.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final recurrent state.
        """

        state = x
        for _ in range(3):
            state = torch.relu(self.cell(state))
        return state


def test_animate_ops_returns_static_html() -> None:
    """animate_ops should expose pass frames for a repeated site."""

    log = tl.trace(RecurrentLinear(), torch.randn(1, 3))
    recurrent_label = next(label for label, count in log.layer_num_calls.items() if count > 1)

    html = log.animate_ops(recurrent_label)

    assert "tl-pass-animation" in html
    assert "Play" in html
    assert recurrent_label in html


def test_summary_waterfall_includes_timing_and_memory() -> None:
    """summary('waterfall') should render timing and memory columns."""

    log = tl.trace(RecurrentLinear(), torch.randn(1, 3))

    text = log.summary("waterfall")

    assert "Waterfall Summary" in text
    assert "Time (ms)" in text
    assert "Memory" in text
