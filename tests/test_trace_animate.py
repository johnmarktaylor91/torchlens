"""Tests for ``Trace.animate(site)``, the one animation door, and ``Drawing.close``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import FacadeTeachingError
from torchlens.options import VisualizationOptions
from torchlens.viz import Drawing


class RecurrentLinear(nn.Module):
    """Small repeated-module model for animation tests."""

    def __init__(self) -> None:
        """Initialize the repeated linear layer."""

        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same cell three times and return the final state."""

        state = x
        for _ in range(3):
            state = torch.relu(self.cell(state))
        return state


def _recurrent_label(trace: tl.Trace) -> str:
    return next(layer.layer_label for layer in trace.layers if layer.num_passes > 1)


def test_animate_returns_an_html_drawing(tmp_path: Path) -> None:
    """``trace.animate(site)`` is an HTML Drawing with one frame per pass; save writes it."""

    trace = tl.trace(RecurrentLinear(), torch.randn(1, 3))
    label = _recurrent_label(trace)

    drawing = trace.animate(label)

    assert isinstance(drawing, Drawing)
    assert drawing.file_format == "html"
    assert drawing.path is None
    html = str(drawing)
    assert "tl-pass-animation" in html
    assert "Play" in html
    assert label in html
    assert drawing._repr_html_() == html
    written = drawing.save(tmp_path / "passes.html")
    assert written.read_text(encoding="utf-8") == html


def test_animate_path_writes_html(tmp_path: Path) -> None:
    """``path=`` writes the animation at once, appending ``.html`` when the suffix differs."""

    trace = tl.trace(RecurrentLinear(), torch.randn(1, 3))

    drawing = trace.animate(_recurrent_label(trace), path=str(tmp_path / "passes"))

    assert drawing.path == tmp_path / "passes.html"
    assert drawing.path.read_text(encoding="utf-8") == str(drawing)


def test_animate_unknown_site_raises() -> None:
    """An unknown site refuses with KeyError, as the old method did."""

    trace = tl.trace(RecurrentLinear(), torch.randn(1, 3))
    with pytest.raises(KeyError):
        trace.animate("no_such_layer_1_1")


def test_visualization_options_animate_redirects_to_trace_animate() -> None:
    """Reading the removed ``animate`` member names ``trace.animate`` as the remedy."""

    with pytest.raises(FacadeTeachingError) as excinfo:
        _ = VisualizationOptions().animate
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert "trace.animate" in str(excinfo.value)


def test_drawing_with_block_calls_close() -> None:
    """Leaving the ``with`` block releases the trace, raise or not."""

    drawing = Drawing(dot="digraph {}", trace=object())
    with drawing as entered:
        assert entered is drawing
        assert drawing.trace is not None
    assert drawing.trace is None

    drawing = Drawing(dot="digraph {}", trace=object())
    with pytest.raises(RuntimeError), drawing:
        raise RuntimeError("body failed")
    assert drawing.trace is None
