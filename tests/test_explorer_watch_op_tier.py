"""Op/facet tier, Route B via fastlog (explorer memo item 8; lane F25).

The differentiator: op-grain sites via one fastlog pass per logged step
under the P1 summary role and the P2 reduce-only path, folded into the
SAME artifact schema (site kinds ``op`` / ``facet``), with disclosed cost
(the plan table), selector-algebra scoping, training-graph neutrality
(the wrapped step returns the LIVE output), and per-head facet identity.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._observability import (
    StepTruth,
)
from torchlens._observability._op_tier import OpTierCollector, split_heads
from torchlens._observability._quantiles import WatchRenderError
from torchlens.errors import WatchLifecycleError, WatchPlanError


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))


def _run_steps(collector: OpTierCollector, model: torch.nn.Module, n: int = 3) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(n):
        with collector.step(step, truth=StepTruth()):
            output = collector.observe_step(torch.randn(4, 8))
            loss = output.square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


class TestOpTierLifecycle:
    def test_zero_match_selection_refuses_at_plan_time(self) -> None:
        collector = OpTierCollector(_model(), save=tl.func("conv2d"))
        with pytest.raises(WatchPlanError) as excinfo:
            collector.discover(torch.randn(4, 8))
        assert excinfo.value.fields["code"] == "watch_plan_empty"

    def test_step_before_discover_refuses(self) -> None:
        collector = OpTierCollector(_model())
        with pytest.raises(WatchLifecycleError) as excinfo, collector.step(0):
            pass
        assert excinfo.value.fields["code"] == "watch_lifecycle_invalid"

    def test_observe_outside_step_refuses(self) -> None:
        model = _model()
        collector = OpTierCollector(model)
        collector.discover(torch.randn(4, 8))
        with pytest.raises(WatchLifecycleError) as excinfo:
            collector.observe_step(torch.randn(4, 8))
        assert excinfo.value.fields["code"] == "watch_lifecycle_invalid"


class TestFacetTier:
    def test_missplit_head_axis_refuses(self) -> None:
        splitter = split_heads(4, dim=1)
        with pytest.raises(WatchRenderError) as excinfo:
            splitter(torch.randn(2, 5, 4))
        assert excinfo.value.fields["code"] == "watch_facet_shape_invalid"
