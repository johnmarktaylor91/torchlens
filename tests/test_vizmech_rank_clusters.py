"""Vizmech wave-2 item 16: no one-node leaf-module clusters on the rank path.

``neato -n`` invents cluster boxes it was never designed to compute: on stock
densenet121, 366 one-node leaf clusters rendered at a median 91% of graph
width (50 above 99%) -- page-wide boxes claiming containment they do not have
(D18a). The rank path now hoists the single member node and emits NO subgraph
wrapper for such modules; multi-node and parent clusters are unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.visualization._rank_layout_internal.layout as layout_mod


class _LeafAndBlock(nn.Module):
    """One single-op leaf module beside a multi-op block."""

    def __init__(self) -> None:
        super().__init__()
        self.solo = nn.ReLU()  # one op -> one-node leaf module
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Tanh())  # two ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.solo(x))


def _rank_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Render through the rank path with neato stubbed; return DOT source."""

    captured: dict[str, str] = {}

    def _fake_neato(**kwargs: object) -> SimpleNamespace:
        captured["source"] = Path(str(kwargs["source_path"])).read_text()
        Path(str(kwargs["rendered_path"])).write_text("<svg></svg>")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(layout_mod, "_run_neato_with_fallbacks", _fake_neato)
    trace = tl.trace(_LeafAndBlock(), torch.randn(1, 4))
    trace.draw(
        vis_node_placement="rank",
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "rank_leaf_clusters"),
        show_containers=False,
    )
    return captured["source"]


def test_one_node_leaf_cluster_not_emitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The solo module gets no cluster; its node survives at parent scope."""

    dot = _rank_dot(tmp_path, monkeypatch)
    assert "cluster_solo" not in dot
    assert re.search(r"relu_\d+_\d+", dot), "the hoisted solo node must still be declared"
    # The multi-node block keeps its cluster.
    assert "cluster_block" in dot


def test_multi_node_clusters_keep_their_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix is scoped to one-node LEAF modules only."""

    dot = _rank_dot(tmp_path, monkeypatch)
    block_lines = [line for line in dot.splitlines() if "cluster_block" in line]
    assert block_lines, "multi-node module cluster disappeared"
