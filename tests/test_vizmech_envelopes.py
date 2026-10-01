"""Vizmech wave-3 item 20: usability-envelope metrics, calibrated both ways.

The collision oracle is structurally blind to a legend that triples page
width, 877 fused arrowheads, and a caption 15 inches from its members (each
demonstrated on a real render this sprint) -- the envelope is the second
oracle (memo D2). Every metric here has a known-bad AND a known-good, with
headroom, never tuned-tight.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._geometry_audit import (
    ARROWHEAD_GAP_FLOOR_PT,
    compute_usability_envelope,
    parse_layout_json,
    run_layout_json,
)

pytestmark = pytest.mark.smoke


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
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / name),
        **kwargs,
    )


def test_arrowhead_oracle_known_bad(tmp_path: Path) -> None:
    """Dense stack fan-in fuses arrowheads and breaks the 2.5 pt floor."""

    dot = _draw(tmp_path, "fan12", _FanStack(12), torch.randn(2, 3))
    envelope = _envelope_of(dot)
    assert envelope.arrowhead_fused_pairs >= 1
    assert envelope.arrowhead_min_gap < ARROWHEAD_GAP_FLOOR_PT


def test_arrowhead_oracle_known_good(tmp_path: Path) -> None:
    """A two-input add keeps distinct arrowheads above the floor."""

    dot = _draw(tmp_path, "add2", _TwoInputAdd(), torch.randn(1, 4))
    envelope = _envelope_of(dot)
    assert envelope.arrowhead_fused_pairs == 0
    assert envelope.arrowhead_min_gap > ARROWHEAD_GAP_FLOOR_PT


def test_legend_ratio_bounded_in_table_form(tmp_path: Path) -> None:
    """The one-table legend stays under parity with content width.

    On the memo's measured defect the six-node strip was 839 pt beside a
    366 pt graph (ratio 2.29). The table form on a comparable tiny model
    stays under 1.0 -- generous headroom, not tuned-tight.
    """

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 3))
    dot = _draw(tmp_path, "legend_ratio", model, torch.randn(1, 3), show_legend=True)
    envelope = _envelope_of(dot)
    assert 0.0 < envelope.legend_ratio < 1.0, envelope


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


def test_caption_distance_measured_and_bounded(tmp_path: Path) -> None:
    """Caption-to-member distance is measured; the fixed form stays close.

    The defect measured p90 732.6 pt / max 1121.9 pt. The member-geometry
    placement on a small nested model stays under 100 pt -- an order of
    magnitude of headroom.
    """

    class Wrapped(nn.Module):
        """A cluster with a deep child so the caption has somewhere to drift."""

        def __init__(self) -> None:
            super().__init__()
            self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.inner(x))

    dot = _draw(tmp_path, "captions", Wrapped(), torch.randn(1, 4))
    envelope = _envelope_of(dot)
    assert envelope.caption_distances, "caption metric found no captions"
    assert envelope.caption_distance_max < 100.0, envelope.caption_distances


def test_canvas_and_ink_metrics_sane(tmp_path: Path) -> None:
    """Canvas aspect/area/ink-coverage read plausibly on a chain."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    dot = _draw(tmp_path, "canvas", model, torch.randn(1, 4))
    envelope = _envelope_of(dot)
    assert envelope.canvas_area > 0
    assert envelope.canvas_aspect >= 1.0
    assert 0.0 < envelope.ink_coverage <= 1.0
    assert 0.0 <= envelope.spine_drift <= 1.0
