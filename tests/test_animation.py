"""Tests for ``trace.draw(animate=)`` and the per-op profile."""

from __future__ import annotations

import pytest
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


def test_draw_animate_returns_an_html_drawing(tmp_path) -> None:
    """``draw(animate=site)`` is an HTML Drawing with one frame per pass; save writes it."""

    log = tl.trace(RecurrentLinear(), torch.randn(1, 3))
    recurrent_label = next(label for label, count in log.layer_num_calls.items() if count > 1)

    drawing = log.draw(animate=recurrent_label).dot

    assert isinstance(drawing, tl.viz.Drawing)
    assert drawing.file_format == "html"
    html = str(drawing)
    assert "tl-pass-animation" in html
    assert "Play" in html
    assert recurrent_label in html
    assert drawing._repr_html_() == html
    written = drawing.save(tmp_path / "passes.html")
    assert written.read_text(encoding="utf-8") == html


def test_draw_animate_unknown_site_raises() -> None:
    """An unknown site refuses with KeyError, as the old method did."""

    log = tl.trace(RecurrentLinear(), torch.randn(1, 3))
    with pytest.raises(KeyError):
        log.draw(animate="no_such_layer_1_1")


def test_op_profile_includes_timing_and_memory() -> None:
    """The per-op profile (successor of summary('waterfall')) has time and memory."""

    pytest.importorskip("pandas")
    log = tl.trace(RecurrentLinear(), torch.randn(1, 3))

    frame = log.profile(level="op").to_pandas()

    assert "time" in frame.columns
    assert "activation_memory" in frame.columns
