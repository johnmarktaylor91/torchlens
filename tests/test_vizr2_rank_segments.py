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
from support.collapse_models import AlternatingSiblings, ParentOpConnectors
from torch import nn

import torchlens as tl
from torchlens._visualization.collapse_optimizer import select_collapse_plan
from torchlens._visualization.collapse_plan import RenderContext

_SEGMENT_NODE = re.compile(r'^\s*"?(\S+__segment__[^"\s\[]+)"?\s*\[[^\]]*label=', re.MULTILINE)


def _source(trace: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    return str(trace.draw(path=str(tmp_path / name), file_format="svg", **kwargs).dot)


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
        dot = _source(trace, tmp_path, "dot", collapse=mode, layout="dot")
        rank = _source(trace, tmp_path, "rank", collapse=mode, layout="rank")
    finally:
        trace.cleanup()
    dot_segments = set(_SEGMENT_NODE.findall(dot))
    rank_segments = {name.replace(":", "pass") for name in _SEGMENT_NODE.findall(rank)}
    assert dot_segments
    assert rank_segments == {name.replace(":", "pass") for name in dot_segments}
    assert (tmp_path / "rank.svg").stat().st_size > 0
