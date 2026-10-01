"""Vizmech wave-2 item 15: cluster caption placement from member geometry (D18b).

Captions were emitted ``labelloc=b`` and INHERITED the graph-level
``labeljust=left``: a fixed corner of the cluster box, up to 1121.9 pt from
the nearest member (p90 732.6 pt over 1066 measured clusters). Pins here:

- every module cluster carries an EXPLICIT ``labeljust`` (centered when the
  chosen end is clear; cornered when boundary-crossing splines pierce it)
  and a ``labelloc`` chosen from where its DIRECT members sit in the flow;
- the entry/exit -> t/b mapping respects ``rankdir``;
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
from torchlens.visualization._render_utils import cluster_caption_attrs
from torchlens.visualization.render_ir import _region_caption_plan

pytestmark = pytest.mark.smoke


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


def _draw(tmp_path: Path, name: str, model: nn.Module) -> str:
    """Render and return DOT source (default BT direction)."""

    trace = tl.trace(model, torch.randn(1, 4))
    return trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / name),
    )


def _cluster_block(dot: str, cluster_marker: str) -> str:
    """Return the attr lines of the subgraph block that carries the caption."""

    blocks = []
    lines = dot.splitlines()
    for index, line in enumerate(lines):
        if cluster_marker in line and "subgraph" in line:
            depth = 0
            body = []
            for body_line in lines[index:]:
                depth += body_line.count("{") - body_line.count("}")
                body.append(body_line)
                if depth <= 0:
                    break
            blocks.append("\n".join(body))
    labeled = [block for block in blocks if "label=" in block]
    assert labeled, f"no labeled block for {cluster_marker}"
    return labeled[0]


def test_caption_attrs_mapping() -> None:
    """Entry/exit map onto page ends per rankdir; centered always."""

    assert cluster_caption_attrs(None, None) == {"labelloc": "b", "labeljust": "c"}
    assert cluster_caption_attrs("entry", "BT")["labelloc"] == "b"
    assert cluster_caption_attrs("exit", "BT")["labelloc"] == "t"
    assert cluster_caption_attrs("entry", "TB")["labelloc"] == "t"
    assert cluster_caption_attrs("exit", "TB")["labelloc"] == "b"
    assert cluster_caption_attrs("entry", "LR")["labeljust"] == "l"
    assert cluster_caption_attrs("exit", "LR")["labeljust"] == "r"
    assert cluster_caption_attrs("entry", "RL")["labeljust"] == "r"
    for end in ("entry", "exit"):
        for rankdir in ("TB", "BT"):
            assert cluster_caption_attrs(end, rankdir)["labeljust"] == "c"


def test_region_caption_plan_classification() -> None:
    """Member end from depth spans; pierced ends flagged; crossings can flip.

    The plan balances FINDABILITY (the member end) against LEGIBILITY (an
    end pierced by boundary-crossing splines strikes a centered caption --
    the widened audit's first live catch).
    """

    depths = {"a": 0, "b": 1, "c": 2, "d": 3}
    subtree = ("a", "b", "c", "d")
    no_edges: tuple = ()
    assert _region_caption_plan(
        direct_names=("a",), subtree_names=subtree, node_depths=depths, edge_endpoints=no_edges
    ) == ("entry", False)
    assert _region_caption_plan(
        direct_names=("d",), subtree_names=subtree, node_depths=depths, edge_endpoints=no_edges
    ) == ("exit", False)
    # Empty direct set: the subtree IS the drawn content (pruned leaf
    # children hoist their nodes into the parent cluster).
    assert _region_caption_plan(
        direct_names=(), subtree_names=subtree, node_depths=depths, edge_endpoints=no_edges
    ) == ("entry", False)
    assert _region_caption_plan(
        direct_names=("a",), subtree_names=("a",), node_depths=depths, edge_endpoints=no_edges
    ) == (None, False)
    # Crossings flip the end: member end is "entry" (a), but the entry side
    # takes three boundary crossings while the exit side takes one.
    depths2 = {"a": 0, "b": 1, "c": 2, "d": 3, "x": 0, "y": 4}
    crossings = (("x", "a"), ("x", "b"), ("x", "a"), ("d", "y"))
    end, pierced = _region_caption_plan(
        direct_names=("a",),
        subtree_names=subtree,
        node_depths=depths2,
        edge_endpoints=crossings,
    )
    assert end == "exit" and pierced is True


def test_tail_heavy_caption_moves_to_exit_end(tmp_path: Path) -> None:
    """BT flow: a tail-direct module captions at the top (its exit end)."""

    dot = _draw(tmp_path, "tail_heavy", _Wrap(_TailHeavy()))
    block = _cluster_block(dot, "cluster_member_pass1 ")
    assert "labelloc=t" in block
    # Both ends of this cluster are pierced by one boundary-crossing spline;
    # the caption dodges the spine channel into the corner (widened-audit
    # catch: a centered caption on a pierced end gets struck through).
    assert "labeljust=l" in block


def test_head_heavy_caption_stays_at_entry_end(tmp_path: Path) -> None:
    """BT flow: a head-direct module captions at the bottom (its entry end)."""

    dot = _draw(tmp_path, "head_heavy", _Wrap(_HeadHeavy()))
    block = _cluster_block(dot, "cluster_member_pass1 ")
    assert "labelloc=b" in block
    assert "labeljust=l" in block


def test_cluster_caption_declared_exactly_once(tmp_path: Path) -> None:
    """N5: the descent may re-open a subgraph block; the caption never dupes."""

    dot = _draw(tmp_path, "caption_once", _Wrap(_TailHeavy()))
    for marker in ("@member</B>", "@member.inner</B>"):
        assert dot.count(marker) == 1, f"caption {marker!r} declared {dot.count(marker)} times"
