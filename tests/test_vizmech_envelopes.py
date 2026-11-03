"""Vizmech wave-3 item 20: usability-envelope metrics, calibrated both ways.

The collision oracle is structurally blind to a legend that triples page
width, 877 fused arrowheads, and a caption 15 inches from its members (each
demonstrated on a real render this sprint) -- the envelope is the second
oracle (memo D2). Every metric here has a known-bad AND a known-good, with
headroom, never tuned-tight.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._geometry_audit import (
    ARROWHEAD_GAP_FLOOR_PT,
    compute_usability_envelope,
    parse_layout_json,
    run_layout_json,
)


def _envelope_of(dot: str, engine: str = "dot"):
    """Parse + measure one DOT source through the named engine."""

    parsed = parse_layout_json(run_layout_json(dot, engine), engine=engine, dot_source=dot)
    return compute_usability_envelope(parsed)


class _FanStack(nn.Module):
    """N-way stack fan-in: the arrowhead-knot known-bad."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x * float(i + 1) for i in range(self.n)], 0).sum(0)


class _TwoInputAdd(nn.Module):
    """Two-way add: a healthy multi-arrowhead node (the known-good)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) + torch.sigmoid(x)


def _draw(tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor, **kwargs) -> str:
    """Render to SVG, return the DOT source."""

    trace = tl.trace(model, x)
    return trace.draw(
        file_format="svg",
        path=str(tmp_path / name),
        **kwargs,
    ).dot


def test_arrowhead_oracle_known_bad(tmp_path: Path) -> None:
    """Dense stack fan-in fuses arrowheads and breaks the 2.5 pt floor."""

    dot = _draw(tmp_path, "fan12", _FanStack(12), torch.randn(2, 3))
    envelope = _envelope_of(dot)
    assert envelope.arrowhead_fused_pairs >= 1
    assert envelope.arrowhead_min_gap < ARROWHEAD_GAP_FLOOR_PT


def test_legend_ratio_flags_the_historical_strip_form() -> None:
    """The six-free-node strip form (the 2.86x defect) reads as out of bound.

    Reconstructed verbatim from the retired emitter's topology: six
    disconnected nodes dot packs BESIDE a small chain.
    """

    strip_dot = """
digraph {
  rankdir=BT
  a -> b -> c
  subgraph cluster_torchlens_legend {
    label="TorchLens legend"
    tl_legend_0 [label="input is a rather long row" shape=oval]
    tl_legend_1 [label="output is a rather long row" shape=oval]
    tl_legend_2 [label="parameterized long row" shape=oval]
    tl_legend_3 [label="buffer long row" shape=cylinder]
    tl_legend_4 [label="boolean long row" shape=oval]
    tl_legend_5 [label="intervention/cone long row" shape=oval]
  }
}
"""
    envelope = _envelope_of(strip_dot)
    assert envelope.legend_ratio > 1.0, envelope
