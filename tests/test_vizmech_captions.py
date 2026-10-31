"""Module box captions sit in the top-left corner, always (JMT, 29 Oct 2026).

Pins here:

- every module cluster carries ``labelloc=t`` and ``labeljust=l``, whatever
  the flow direction or where its members sit;
- the triple-declared-subgraph emitter quirk stays harmless: a cluster's
  caption text is declared exactly once however many times the descent
  re-opens the subgraph block (N5 check).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization._geometry_audit import run_layout_json
from torchlens._visualization._render_utils import cluster_caption_attrs


class _TailHeavy(nn.Module):
    """A module whose DIRECT op runs AFTER its deep child block.

    The caption should sit at the module's exit end, next to the op it
    actually names, instead of floating at the entry corner of a box the
    child block dominates.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.inner(x))


class _HeadHeavy(nn.Module):
    """A module whose DIRECT op runs BEFORE its deep child block."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(torch.sigmoid(x))


class _Wrap(nn.Module):
    """Wrapper giving the module under test a named address (a cluster)."""

    def __init__(self, member: nn.Module) -> None:
        super().__init__()
        self.member = member

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.member(x)


def _draw(tmp_path: Path, name: str, model: nn.Module, direction: str = "bottomup") -> str:
    """Render and return DOT source (default BT direction)."""

    trace = tl.trace(model, torch.randn(1, 4))
    return trace.draw(
        direction=direction,  # type: ignore[arg-type]
        vis_save_only=True,
        file_format="svg",
        path=str(tmp_path / name),
    )


def test_caption_attrs_swap_where_dot_flips() -> None:
    """dot reads cluster captions in rank coordinates: BT swaps t/b, RL swaps l/r."""

    assert cluster_caption_attrs("TB") == {"labelloc": "t", "labeljust": "l"}
    assert cluster_caption_attrs("LR") == {"labelloc": "t", "labeljust": "l"}
    assert cluster_caption_attrs("BT") == {"labelloc": "b", "labeljust": "l"}
    assert cluster_caption_attrs("RL") == {"labelloc": "t", "labeljust": "r"}
    assert cluster_caption_attrs(None) == {"labelloc": "t", "labeljust": "l"}


@pytest.mark.parametrize("direction", ["bottomup", "topdown", "leftright", "rightleft"])
@pytest.mark.parametrize("model", [_TailHeavy, _HeadHeavy])
def test_caption_lands_top_left_on_the_page(
    tmp_path: Path, direction: str, model: type[nn.Module]
) -> None:
    """Measured from the layout: every module caption sits in its box's top-left quarter."""

    dot = _draw(tmp_path, f"{model.__name__}_{direction}", _Wrap(model()), direction=direction)
    layout = run_layout_json(dot, "dot")
    clusters = [obj for obj in layout["objects"] if str(obj.get("name", "")).startswith("cluster")]
    assert clusters
    for cluster in clusters:
        x0, y0, x1, y1 = (float(value) for value in cluster["bb"].split(","))
        label_x, label_y = (float(value) for value in cluster["lp"].split(","))
        # Graphviz JSON coordinates grow upward: the top half has the larger y.
        assert label_y > (y0 + y1) / 2, (direction, cluster["name"], "caption not at the top")
        assert label_x < (x0 + x1) / 2, (direction, cluster["name"], "caption not at the left")


def test_cluster_caption_declared_exactly_once(tmp_path: Path) -> None:
    """N5: the descent may re-open a subgraph block; the caption never dupes."""

    dot = _draw(tmp_path, "caption_once", _Wrap(_TailHeavy()))
    for marker in ("@member</B>", "@member.inner</B>"):
        assert dot.count(marker) == 1, f"caption {marker!r} declared {dot.count(marker)} times"
