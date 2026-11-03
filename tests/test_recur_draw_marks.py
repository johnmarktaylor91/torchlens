"""Recurrent and conditional drawing marks: edge annotation size and pass marks.

* Edge annotations (``In``/``Out`` pass ranges, argument labels, IF/THEN) follow
  the edge font size (``edge_overrides["fontsize"]`` or ``font_size``) instead of
  a fixed 8 pt; the default drawing keeps 8 pt and 18 pt.
* A rolled pass range on a conditional edge shares the branch label's midpoint
  slot rather than floating as a tail or head label.
* The ``"label"`` token of ``node_label_fields`` carries the pass marks the default
  title shows, and the ``"pass"`` token prints nothing on single-pass nodes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._visualization._typography import (
    DEFAULT_TYPOGRAPHY,
    resolve_edge_typography,
)

_POINT_SIZE = re.compile(r'POINT-SIZE="([^"]+)"')


class _ToyLoopIf(nn.Module):
    """Three passes of one recurrent cell, then a data-dependent branch."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.RNNCell(4, 4)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            h = self.cell(x, h)
        if h.mean() > -10:
            return h * 2
        return -h


class _RecurrentLinear(nn.Module):
    """One Linear reused three times, then a single-pass output op."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.fc(x))
        return torch.sigmoid(x)


def _dot(trace: tl.Trace, **kwargs: object) -> str:
    return trace.draw(file_format="dot", **kwargs).dot


def _edge_lines(dot: str) -> list[str]:
    return [line for line in dot.splitlines() if "->" in line]


def _edge_point_sizes(dot: str) -> set[str]:
    return {size for line in _edge_lines(dot) for size in _POINT_SIZE.findall(line)}


def _branch_edge(dot: str) -> str:
    lines = [line for line in _edge_lines(dot) if "<u>IF</u>" in line]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.fixture(scope="module")
def loop_if_trace() -> Iterator[tl.Trace]:
    torch.manual_seed(0)
    trace = tl.trace(_ToyLoopIf(), (torch.randn(1, 4), torch.zeros(1, 4)))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def linear_trace() -> Iterator[tl.Trace]:
    torch.manual_seed(0)
    trace = tl.trace(_RecurrentLinear(), torch.randn(1, 8))
    yield trace
    trace.cleanup()


def test_resolve_edge_typography_scales_annotation_and_emphasis() -> None:
    record = resolve_edge_typography(DEFAULT_TYPOGRAPHY, edge_overrides={"fontsize": 7})
    assert (record.annotation_pt, record.emphasis_pt) == ("7", "9")
    assert resolve_edge_typography(DEFAULT_TYPOGRAPHY) is DEFAULT_TYPOGRAPHY
    # A callable override cannot size a whole draw; the theme record stands.
    unsized = resolve_edge_typography(DEFAULT_TYPOGRAPHY, edge_overrides={"fontsize": len})
    assert unsized is DEFAULT_TYPOGRAPHY
    # edge_overrides wins over font_size, which wins over the theme edge size.
    ordered = resolve_edge_typography(
        DEFAULT_TYPOGRAPHY,
        edge_overrides={"fontsize": "11"},
        font_size=20,
        theme_edge={"fontsize": 30},
    )
    assert ordered.annotation_pt == "11"


def test_default_edge_annotations_keep_8_and_18_point(loop_if_trace: tl.Trace) -> None:
    dot = _dot(loop_if_trace, rolled=True)
    assert '<FONT POINT-SIZE="18"><b><u>IF</u>' in _branch_edge(dot)
    assert '"8"' in {f'"{size}"' for size in _edge_point_sizes(dot)}


@pytest.mark.parametrize(
    "kwargs",
    [{"edge_overrides": {"fontsize": 7}}, {"font_size": 7}],
    ids=["edge_overrides", "font_size"],
)
def test_edge_font_size_sizes_every_edge_annotation(
    loop_if_trace: tl.Trace, kwargs: dict[str, object]
) -> None:
    dot = _dot(loop_if_trace, rolled=True, **kwargs)
    assert _edge_point_sizes(dot) == {"7", "9"}
    assert '<FONT POINT-SIZE="9"><b><u>IF</u>' in _branch_edge(dot)
    assert "labelfontsize=8" not in dot


def test_edge_font_size_sizes_argument_labels(linear_trace: tl.Trace) -> None:
    default = _dot(linear_trace, rolled=True)
    sized = _dot(linear_trace, rolled=True, edge_overrides={"fontsize": 12})
    assert "8" in _edge_point_sizes(default)
    assert "8" not in _edge_point_sizes(sized)
    assert "12" in _edge_point_sizes(sized)


def test_rolled_pass_range_shares_the_branch_midpoint(loop_if_trace: tl.Trace) -> None:
    line = _branch_edge(_dot(loop_if_trace, rolled=True))
    label = line.split("[label=", 1)[1]
    assert "<u>IF</u>" in label and ">Out " in label
    assert "taillabel" not in line
    assert "headlabel" not in line


def test_label_token_carries_pass_marks(linear_trace: tl.Trace) -> None:
    unrolled = _dot(linear_trace, node_label_fields=["label", "shape"])
    rolled = _dot(linear_trace, rolled=True, node_label_fields=["label", "shape"])
    for pass_number in (1, 2, 3):
        assert f"linear_1_1:{pass_number}" in unrolled
    assert "linear_1_1 (x3)" in rolled
    # Single-pass nodes keep a bare label.
    assert re.search(r"sigmoid_\d+_\d+[<\"]", rolled)
    assert not re.search(r"sigmoid_\d+_\d+ \(x", rolled)


def test_pass_token_is_empty_on_single_pass_nodes(linear_trace: tl.Trace) -> None:
    rows = _dot(linear_trace, rolled=True, node_label_fields=["label", "pass"])
    unrolled = _dot(linear_trace, node_label_fields=["label", "pass"])
    sigmoid_lines = [line for line in rows.splitlines() if "sigmoid_" in line and "->" not in line]
    assert sigmoid_lines
    assert all(">1<" not in line for line in sigmoid_lines)
    assert any(">3<" in line for line in rows.splitlines() if "linear_1_1" in line)
    for pass_number in (1, 2, 3):
        assert f">{pass_number}<" in unrolled
