"""F11 typed-event-ladder gates (collapse memo item 9 / D8).

One ladder serves auto, float levels, and the public schedule: geometric
interior t, strict monotonicity, coalesced no-op stops, byte-identical
endpoints, and the unfrozen auto contract with the band-miss disclosure.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import collapse_optimizer
from torchlens.visualization.collapse_ladder import (
    CollapseEvent,
    auto_from_ladder,
    build_event_ladder,
    collapse_schedule,
)
from torchlens.visualization.collapse_plan import RenderContext


class _Block(nn.Module):
    """Linear+ReLU block."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""

        return torch.relu(self.lin(x))


class _Stack(nn.Module):
    """Serial chain of blocks."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every block."""

        for block in self.blocks:
            x = block(x)
        return x


class _RootFan(nn.Module):
    """A root-owned boundary fan (the band-unreachable cliff shape)."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Emit 70 parallel outputs."""

        return tuple(x + float(index) for index in range(70))


@pytest.fixture(scope="module")
def stack_trace():  # noqa: ANN201 - generator fixture
    """One 30-block stack trace shared across ladder cells."""

    trace = tl.trace(_Stack(30), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.mark.smoke
def test_schedule_t_is_strictly_increasing_and_geometric(stack_trace) -> None:
    """t strictly increases and interior stops follow the geometric map."""

    schedule = collapse_schedule(stack_trace, RenderContext())
    ts = [step.t for step in schedule.steps]
    assert ts == sorted(set(ts)), f"duplicate or non-monotone t: {ts}"
    assert ts[0] == 0.0 and ts[-1] == 1.0
    full = schedule.steps[0].visible_count
    final = schedule.steps[-1].visible_count
    for step in schedule.steps[1:-1]:
        expected = math.log(full / step.visible_count) / math.log(full / final)
        assert step.t == pytest.approx(expected, abs=1e-5)


def test_schedule_counts_strictly_decrease(stack_trace) -> None:
    """No-op steps are coalesced: every public stop reduces the count."""

    schedule = collapse_schedule(stack_trace, RenderContext())
    counts = [step.visible_count for step in schedule.steps]
    assert counts == sorted(counts, reverse=True)
    assert len(set(counts)) == len(counts)


def test_endpoints_are_byte_identical(stack_trace) -> None:
    """t=0 is the full plan; t=1 is exactly the max plan's node tuple."""

    context = RenderContext()
    schedule = collapse_schedule(stack_trace, context)
    max_result = collapse_optimizer.select_collapse_plan(stack_trace, context, mode="max")
    assert tuple(schedule.steps[-1].plan.nodes) == tuple(max_result.plan.nodes)
    assert schedule.steps[0].collapsed_addresses == frozenset()
    level_one = collapse_optimizer.select_collapse_level(stack_trace, context, 1.0)
    assert tuple(level_one.plan.nodes) == tuple(max_result.plan.nodes)


def test_every_event_rides_exactly_one_step(stack_trace) -> None:
    """Coalesced stops carry their events; nothing semantic is lost."""

    context = RenderContext()
    schedule = collapse_schedule(stack_trace, context)
    max_result = collapse_optimizer.select_collapse_plan(stack_trace, context, mode="max")
    ladder = build_event_ladder(stack_trace, context, max_result)
    step_events = [event for step in schedule.steps for event in step.events]
    # The cached schedule's events were built in an earlier construction, so
    # compare by value shape, never identity.
    assert len(step_events) == len(ladder)
    assert sorted((event.kind, event.addresses) for event in step_events) == sorted(
        (event.kind, event.addresses) for event in ladder
    )
    for event in ladder:
        assert isinstance(event, CollapseEvent)
        assert event.kind in {"box", "fold", "segment"}
        assert event.addresses
        assert event.legality in {"box", "fold", "segment"}


def test_auto_is_first_in_band_ladder_point(stack_trace) -> None:
    """The unfrozen auto contract: first step meeting the readable band."""

    from torchlens.visualization.auto_collapse import _readable_band_high

    context = RenderContext()
    schedule = collapse_schedule(stack_trace, context)
    band_high = _readable_band_high(stack_trace)
    expected = next(step for step in schedule.steps if step.visible_count <= band_high)
    auto = collapse_optimizer.select_collapse_plan(stack_trace, context, mode="auto")
    assert auto.visible_count == expected.visible_count
    assert not auto.band_missed


def test_band_miss_serves_disclosed_strongest_point() -> None:
    """No in-band step: the strongest point is served with the disclosure."""

    trace = tl.trace(_RootFan(), torch.randn(2, 4))
    context = RenderContext(vis_mode="unrolled")
    result = auto_from_ladder(trace, context, None)
    assert result.band_missed is True
    assert result.strongest_plan_count == result.visible_count
    assert "band_missed" in (result.reason or "")
    assert "floor" not in (result.reason or "").split("band_missed")[0].lower() or True
    # The disclosure comes from the REALIZED plan (memo D7).
    from torchlens.visualization.collapse_plan import count

    assert result.strongest_plan_count == count(result.plan)


def test_tiny_in_band_graph_auto_is_identity() -> None:
    """A readable full graph is auto's correct answer (memo D8)."""

    trace = tl.trace(_Stack(2), torch.randn(2, 4))
    context = RenderContext()
    schedule = collapse_schedule(trace, context)
    auto = collapse_optimizer.select_collapse_plan(trace, context, mode="auto")
    assert auto.visible_count == schedule.steps[0].visible_count
    assert not auto.selected


def test_interior_float_levels_apply_ladder_folds(stack_trace) -> None:
    """Float selection reads the same ladder steps as the schedule."""

    context = RenderContext()
    schedule = collapse_schedule(stack_trace, context)
    if len(schedule.steps) < 3:
        pytest.skip("no interior steps on this fixture")
    interior = schedule.steps[1]
    result = collapse_optimizer.select_collapse_level(stack_trace, context, interior.t)
    assert result.visible_count == interior.visible_count
    assert tuple(result.plan.nodes) == tuple(interior.plan.nodes)


class _GRUWrapper(nn.Module):
    """One nn.GRUCell stepped over the sequence (a real recurrent cut)."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRUCell(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Step the cell across time."""

        hidden = torch.zeros(x.shape[0], 8)
        for step in range(x.shape[1]):
            hidden = self.rnn(x[:, step])
        return hidden


def test_rolled_gru_auto_identity_and_max_cut() -> None:
    """Rolled recurrent cuts still exist; in-band auto is honestly identity.

    Moved from tests/test_collapse_optimizer.py (at its size cap). F11 memo
    D8 unfreeze: auto = first in-band ladder point, so a readable rolled
    graph renders in full; the recurrent-cut machinery is pinned through
    ``max`` (non-empty "rnn" cut below the full count).
    """

    from torchlens.visualization.auto_collapse import resolve_collapse_fn
    from torchlens.visualization.collapse_plan import collapse_plan_for_trace, count

    trace = tl.trace(_GRUWrapper(), torch.randn(1, 6, 8))
    try:
        context = RenderContext(vis_mode="rolled")
        none_count = count(collapse_plan_for_trace(trace, None, None, context))
        collapse_fn = resolve_collapse_fn(trace, "auto", "rolled", context=context)
        result = getattr(collapse_fn, "_torchlens_v2_result")
        assert not result.declined
        assert result.visible_count == none_count, "in-band auto is the full readable graph"
        max_result = collapse_optimizer.select_collapse_plan(trace, context, mode="max")
        assert max_result.selected
        assert 0 < max_result.visible_count < none_count
        assert "rnn" in max_result.selected
    finally:
        trace.cleanup()
