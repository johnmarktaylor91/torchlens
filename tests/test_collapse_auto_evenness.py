"""``collapse="auto"`` evenness: stray small modules box, identical runs fold.

The ladder's first in-band point orders boxes by cost, so on its own it can
leave a small module expanded between boxed siblings and draw a run of
identical blocks as one box each. ``auto`` evens that point out; these
tests pin both passes and the invariant that evening never adds nodes.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import collapse_optimizer
from torchlens._visualization.auto_collapse import analyze_collapse
from torchlens._visualization.collapse_ladder import (
    EVEN_FOLD_MIN_RUN,
    _even_auto_boxes,
    _fold_identical_runs,
    _ladder_state_through,
    _source_graph,
    collapse_schedule,
)
from torchlens._visualization.collapse_plan import (
    ChildSegment,
    ModuleBox,
    RenderContext,
    collapse_plan_for_source_graph,
    count,
)


class _Block(nn.Module):
    """Residual MLP block (four ops)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class _Stage(nn.Module):
    """A small downsample module followed by a run of identical blocks."""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))
        self.blocks = nn.Sequential(*[_Block(width) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class _MiniConvNextLike(nn.Module):
    """Stem, three stages (2, 6, 2 blocks) and a head: convnext's shape."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8))
        self.stages = nn.Sequential(_Stage(8, 2), _Stage(8, 6), _Stage(8, 2))
        self.head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


@pytest.fixture(scope="module")
def mini_trace() -> Any:
    trace = tl.trace(_MiniConvNextLike(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _plan(trace: tl.Trace, boxes: set[str]) -> Any:
    context = RenderContext()
    return collapse_plan_for_source_graph(
        _source_graph(trace, context),
        collapse_optimizer._collapse_fn_from_selected(frozenset(boxes)),
        {},
    )


def _middle_blocks() -> set[str]:
    return {f"stages.1.blocks.{index}" for index in range(6)}


def test_small_expanded_modules_box_beside_condensed_siblings(mini_trace: tl.Trace) -> None:
    signals = analyze_collapse(mini_trace).signals
    boxes = _even_auto_boxes(mini_trace, signals, frozenset(_middle_blocks()))
    # The downsample sits beside the boxed blocks; stem and head beside the
    # stage container holding them. All three are no larger than a block box.
    assert {"stages.1.downsample", "stem", "head"} <= boxes
    # A module larger than the boxes beside it stays expanded.
    assert "stages.0" not in boxes and "stages.2" not in boxes
    # The parent the user wrote wins over its child.
    assert not any(box.startswith("stem.") for box in boxes)


def test_nothing_boxes_without_a_condensed_sibling(mini_trace: tl.Trace) -> None:
    signals = analyze_collapse(mini_trace).signals
    assert _even_auto_boxes(mini_trace, signals, frozenset()) == frozenset()


def test_identical_sibling_box_run_folds_into_one_segment(mini_trace: tl.Trace) -> None:
    plan = _plan(mini_trace, _middle_blocks())
    folded, segments = _fold_identical_runs(mini_trace, RenderContext(), plan)
    runs = [node for node in folded.nodes if isinstance(node, ChildSegment)]
    assert [tuple(run.members) for run in runs] == [tuple(sorted(_middle_blocks()))]
    assert len(segments) == 1
    assert count(folded) == count(plan) - (len(_middle_blocks()) - 1)
    assert not any(
        isinstance(node, ModuleBox) and node.call.startswith("stages.1.blocks.")
        for node in folded.nodes
    )


def test_short_runs_do_not_fold(mini_trace: tl.Trace) -> None:
    pair = {"stages.0.blocks.0", "stages.0.blocks.1"}
    assert len(pair) < EVEN_FOLD_MIN_RUN
    plan = _plan(mini_trace, pair)
    folded, segments = _fold_identical_runs(mini_trace, RenderContext(), plan)
    assert segments == {}
    assert folded is plan


def test_auto_never_has_more_nodes_than_its_ladder_point(mini_trace: tl.Trace) -> None:
    from torchlens._visualization.auto_collapse import _readable_band_high

    context = RenderContext()
    schedule = collapse_schedule(mini_trace, context)
    band_high = _readable_band_high(mini_trace)
    chosen = next(step for step in schedule.steps if step.visible_count <= band_high)
    auto = collapse_optimizer.select_collapse_plan(mini_trace, context, mode="auto")
    assert auto.visible_count <= chosen.visible_count
    assert auto.visible_count == count(auto.plan)
    if chosen is not schedule.steps[-1]:
        ladder_selected, _ = _ladder_state_through(schedule.steps, chosen)
        assert ladder_selected <= auto.selected


def test_auto_draw_renders_folded_run(mini_trace: tl.Trace, tmp_path: Any) -> None:
    dot = mini_trace.draw(
        collapse="auto", vis_save_only=True, file_format="svg", path=str(tmp_path / "g")
    )
    auto = collapse_optimizer.select_collapse_plan(mini_trace, RenderContext(), mode="auto")
    for descriptor in (auto.segments or {}).values():
        assert descriptor.name in dot
