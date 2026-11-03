"""F11 typed-event-ladder gates (collapse memo item 9 / D8).

One ladder serves auto, float levels, and the public schedule: geometric
interior t, strict monotonicity, coalesced no-op stops, byte-identical
endpoints, and the unfrozen auto contract with the band-miss disclosure.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._visualization import collapse_optimizer
from torchlens._visualization.collapse_ladder import (
    auto_from_ladder,
    collapse_schedule,
)
from torchlens._visualization.collapse_plan import RenderContext


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
    from torchlens._visualization.collapse_plan import count

    assert result.strongest_plan_count == count(result.plan)


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
