"""collapse 4: the rank layout scope split.

Segment half: a plan with segment boxes renders under the rank layout, with
one declared node per segment (the dot path already drew them). Buffer half:
the buffer-node rank crash no longer reproduces; buffer renders under the rank
layout are pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_collapse_segment_adjacency import AlternatingSiblings, ParentOpConnectors
from torchlens._visualization.collapse_optimizer import select_collapse_plan
from torchlens._visualization.collapse_plan import RenderContext

_SEGMENT_NODE = re.compile(r'^\s*"?(\S+__segment__[^"\s\[]+)"?\s*\[[^\]]*label=', re.MULTILINE)


def _source(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    return str(
        trace.draw(path=str(tmp_path / name), vis_save_only=True, file_format="svg", **kwargs)
    )


@pytest.mark.parametrize(
    ("factory", "mode"),
    [(AlternatingSiblings, 1.0), (ParentOpConnectors, "max")],
    ids=["alternating", "parent_op_connectors"],
)
def test_segment_plan_renders_under_rank_with_one_node_per_segment(
    tmp_path: Path, factory: type[nn.Module], mode: object
) -> None:
    """Rank declares every segment box the dot path draws."""
    torch.manual_seed(0)
    model = factory().eval()
    with torch.no_grad():
        trace = tl.trace(model, torch.randn(2, 8))
    try:
        result = select_collapse_plan(trace, RenderContext(), mode="max")
        assert result.segments, "fixture must produce a segment plan"
        dot = _source(trace, tmp_path, "dot", collapse=mode, vis_node_placement="dot")
        rank = _source(trace, tmp_path, "rank", collapse=mode, vis_node_placement="rank")
    finally:
        trace.cleanup()
    dot_segments = set(_SEGMENT_NODE.findall(dot))
    rank_segments = {name.replace(":", "pass") for name in _SEGMENT_NODE.findall(rank)}
    assert dot_segments
    assert rank_segments == {name.replace(":", "pass") for name in dot_segments}
    assert (tmp_path / "rank.svg").stat().st_size > 0


class _BatchNormNet(nn.Module):
    """Two conv-batchnorm stages: train mode writes the running buffers."""

    def __init__(self) -> None:
        """Initialize the stages."""
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3, padding=1)
        self.b1 = nn.BatchNorm2d(4)
        self.c2 = nn.Conv2d(4, 4, 3, padding=1)
        self.b2 = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both stages."""
        return self.b2(self.c2(torch.relu(self.b1(self.c1(x)))))


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("buffers", ["never", "meaningful", "always"])
def test_buffer_nodes_render_under_rank(tmp_path: Path, train: bool, buffers: str) -> None:
    """The buffer-node rank crash (collapse memo D2, the layout half) stays fixed."""
    torch.manual_seed(0)
    trace = tl.trace(_BatchNormNet().train(train), torch.randn(2, 3, 8, 8))
    try:
        _source(trace, tmp_path, "bn", vis_node_placement="rank", show_buffers=buffers)
    finally:
        trace.cleanup()
    assert (tmp_path / "bn.svg").stat().st_size > 0
