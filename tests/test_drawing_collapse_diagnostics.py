"""Collapse diagnostics live on the draw result, not on Trace (round-2 walk names.1)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import FacadeTeachingError, TraceCleanedUpError
from torchlens._user_funcs import render
from torchlens._visualization.collapse_plan import CollapsePlan, CollapseSchedule


class _Nested(nn.Module):
    """Two nested blocks so the collapse planner has modules to condense."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x))


def test_drawing_collapse_plan_and_schedule_read_the_retained_trace(tmp_path) -> None:
    """A drawing made with return_trace=True answers both diagnostics."""

    drawing = render(
        _Nested().eval(),
        torch.randn(1, 4),
        return_trace=True,
        file=str(tmp_path / "g.svg"),
    )
    with drawing:
        for mode in ("auto", "max", 0.5):
            assert isinstance(drawing.collapse_plan(mode), CollapsePlan)
        assert repr(drawing.collapse_plan(1.0)) == repr(drawing.collapse_plan("max"))
        schedule = drawing.collapse_schedule()
        assert isinstance(schedule, CollapseSchedule)
        counts = [step.visible_count for step in schedule.steps]
        assert counts == sorted(counts, reverse=True)


def test_drawing_without_a_retained_trace_refuses_typed(tmp_path) -> None:
    """A detached or closed drawing refuses with the husked-trace error, never a bare failure."""

    detached = render(_Nested().eval(), torch.randn(1, 4), file=str(tmp_path / "g.svg"))
    for call in (detached.collapse_plan, detached.collapse_schedule):
        with pytest.raises(TraceCleanedUpError) as excinfo:
            call()
        assert "return_trace=True" in str(excinfo.value)


def test_trace_no_longer_carries_the_collapse_diagnostics() -> None:
    """The Trace spellings refuse typed and name the draw result."""

    trace = tl.trace(_Nested().eval(), torch.randn(1, 4))
    try:
        for name in ("collapse_plan", "collapse_schedule"):
            with pytest.raises(FacadeTeachingError) as excinfo:
                getattr(trace, name)
            assert excinfo.value.fields["code"] == "facade_redirect"
            assert "draw result" in str(excinfo.value)
    finally:
        trace.cleanup()
