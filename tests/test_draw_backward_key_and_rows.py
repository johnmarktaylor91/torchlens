"""Backward drawing: key margin, grad rows and module focus.

Pins:
- the backward key's text cells are sized from the bundled face's measured
  width plus a margin, so the longest row stays inside the border when the
  viewer's face is a little wider than Graphviz's layout measurement;
- a grad_fn with no stored gradient prints no ``grad`` row (no ``grad N/A``
  placeholder), while nodes with a stored gradient keep their shape row;
- ``module=`` on a backward or combined graph refuses with
  ``visualization_mode_invalid`` instead of drawing the whole model.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._visualization._header_block import _text_width
from torchlens._visualization.themes import THEME_PRESETS

pytest.importorskip("graphviz")


class _ResidualBlock(nn.Module):
    """Two linears with a skip add: intervening and accumulate grad_fns."""

    def __init__(self) -> None:
        """Initialize the block."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus the two-layer branch."""

        return x + self.fc2(torch.relu(self.fc1(x)))


def _backward_trace() -> tl.Trace:
    """Capture the block with every gradient saved, then run backward."""

    torch.manual_seed(0)
    trace = tl.trace(
        _ResidualBlock(),
        torch.randn(4, 8),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.backward(trace[trace.output_layers[0]].out.sum())
    return trace


def test_nodes_without_a_stored_gradient_print_no_grad_row(tmp_path: Path) -> None:
    """Stored gradients keep their shape row; the rest print no placeholder."""

    trace = _backward_trace()
    try:
        dot = trace.draw(path=str(tmp_path / "bwd"), graph="backward", file_format="svg").dot
    finally:
        trace.cleanup()
    assert "grad N/A" not in dot
    assert "grad (4, 8)" in dot
    assert "grad (4, 16)" in dot
    # The intervening nodes are still drawn and still marked.
    assert "[i] accumulategrad_back" in dot


def test_backward_key_cells_carry_a_margin_over_the_measured_text(tmp_path: Path) -> None:
    """Every key text cell is fixed wider than its words in the bundled face."""

    trace = _backward_trace()
    try:
        drawing = trace.draw(
            path=str(tmp_path / "bwd"), graph="backward", file_format="svg", show_legend=True
        )
    finally:
        trace.cleanup()
    legend_line = next(line for line in drawing.dot.splitlines() if "backward key" in line)
    size = THEME_PRESETS["torchlens"].typography.secondary_size
    cells = re.findall(
        r'<TD (?:COLSPAN="2" )?FIXEDSIZE="TRUE" WIDTH="(\d+)" HEIGHT="\d+" ALIGN="LEFT">'
        r"<FONT[^>]*>(?:<B>)?([^<]+)",
        legend_line,
    )
    texts = {text for _, text in cells}
    assert "accum = gradient accumulation into a leaf (dotted edge)" in texts
    for width, text in cells:
        bold = text == "backward key"
        # 2 x 3 pt padding plus a margin of at least 2 pt over the measured words.
        assert int(width) >= _text_width(text, size, bold) + 8.0, (text, width)
    svg = (tmp_path / "bwd.svg").read_text(encoding="utf-8")
    assert "accum = gradient accumulation into a leaf (dotted edge)" in svg


def test_default_backward_drawing_is_title_and_graph_only(tmp_path: Path) -> None:
    """One pass: the title alone, no pass box, no key; the key opts in and lists drawn rows only."""

    trace = _backward_trace()
    try:
        default = trace.draw(path=str(tmp_path / "bwd"), graph="backward", file_format="svg").dot
        keyed = trace.draw(
            path=str(tmp_path / "keyed"), graph="backward", file_format="svg", show_legend=True
        ).dot
    finally:
        trace.cleanup()
    assert "_ResidualBlock backward graph" in default
    for absent in ("grad_fn_handle", "pass(es)", "mode:", "unrolled", "backward pass 1"):
        assert absent not in default, absent
    assert "cluster_backward_pass" not in default
    assert "backward key" not in default
    assert "backward key" in keyed
    assert "[i] = intervening grad_fn (no forward op)" in keyed
    # No node prints an ``order N`` or ``bwd N`` row, so the key explains neither.
    assert "order N" not in keyed
    assert "bwd N" not in keyed


@pytest.mark.parametrize("graph", ["backward", "combined"])
def test_module_focus_refuses_on_gradient_graphs(tmp_path: Path, graph: str) -> None:
    """module= with a backward or combined graph refuses instead of being ignored."""

    trace = _backward_trace()
    try:
        with pytest.raises(InvalidArgumentError) as excinfo:
            trace.draw(path=str(tmp_path / "g"), graph=graph, module="fc1")
        assert excinfo.value.fields["code"] == "visualization_mode_invalid"
        assert "module=" in str(excinfo.value)
        with pytest.raises(InvalidArgumentError):
            trace.modules["fc1"].draw(path=str(tmp_path / "m"), graph=graph)
        # The forward graph still honours the focus.
        trace.draw(path=str(tmp_path / "f"), module="fc1", file_format="svg")
    finally:
        trace.cleanup()
