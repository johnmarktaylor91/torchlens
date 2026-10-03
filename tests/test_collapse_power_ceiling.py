"""F11 ceiling-replacement gates (collapse memo item 8 / D5).

Admission is U-gated and estimator-tiered; over-budget requests degrade to
the deterministic fallback planner, never an uncollapsed wall; the watchdog
abandons only TO the fallback and is disclosed as an estimator bug.
"""

from __future__ import annotations

import warnings as warnings_module

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.visualization import collapse_estimator, collapse_optimizer
from torchlens.visualization.collapse_plan import RenderContext, count


class _Block(nn.Module):
    """Linear+ReLU block for admission toys."""

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


def test_estimator_formula_and_constants() -> None:
    """The (U, W) power law reproduces the memo's fitted witnesses."""

    predicted = collapse_estimator.predicted_select_ms(713, 48)
    # densenet201 row: measured 4.8 s, predicted ~5.9 s (within the 2.05x
    # held-out envelope the memo reports).
    assert 4_000 < predicted < 9_000
    assert collapse_estimator.predicted_select_ms(1, 1) < 1.0
    assert collapse_estimator.ESTIMATOR_FORMULA_VERSION == "uw_power_v1"


def test_budget_fallback_is_a_real_disclosed_plan(monkeypatch) -> None:
    """Over-budget admission degrades to the fallback planner, never a wall."""

    # 30 Sequential blocks put U above the readable band AND give boxing
    # real content to hide (a Linear+ReLU interior with no kept own op);
    # honest per-call pricing makes a box of a 2-op block with a kept
    # atomic own op a no-op, which is correct but not this test's subject.
    class _BoxableStack(nn.Module):
        """Serial chain of boxable Sequential blocks."""

        def __init__(self, n: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(n))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply every block."""

            for block in self.blocks:
                x = block(x)
            return x

    trace = tl.trace(_BoxableStack(30), torch.randn(2, 4))
    # Constant 4: raw ops (62) stay under the 20x pathological gate (80)
    # while U (62) exceeds the constant -- the budget-degrade arm.
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 4)
    with pytest.warns(TorchLensWarning, match="compact fallback plan") as caught:
        result = collapse_optimizer.select_collapse_plan(trace, RenderContext(), mode="max")
    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "collapse_budget_fallback" in codes
    assert result.planner == "linear_fallback"
    assert not result.declined
    full_count = count(
        collapse_optimizer._collapse_plan_for_source_or_trace(
            trace, None, None, RenderContext(), None
        )
    )
    assert 0 < result.visible_count < full_count
    assert result.estimator is not None
    assert result.estimator.tier == "linear_fallback"
    assert result.scored_k is None  # no DP claim on fallback tiers
    # Parity: the visible count IS the realized plan count.
    assert result.visible_count == count(result.plan)


def test_watchdog_abandons_to_fallback_and_discloses(monkeypatch) -> None:
    """A fired watchdog serves the fallback and discloses the estimator bug."""

    trace = tl.trace(_Stack(6), torch.randn(2, 4))
    monkeypatch.setattr(collapse_estimator, "ACTIVE_BUDGET", collapse_estimator.ACTIVE_BUDGET)
    monkeypatch.setattr(collapse_optimizer, "watchdog_enabled", lambda: True)
    # Deadline lands in the past: the first frontier charge fires.
    monkeypatch.setattr(collapse_optimizer, "WATCHDOG_FLOOR_MS", -1e9)
    monkeypatch.setattr(collapse_optimizer, "WATCHDOG_SLACK", 0.0)
    with pytest.warns(TorchLensWarning, match="estimator bug") as caught:
        result = collapse_optimizer.select_collapse_plan(trace, RenderContext(), mode="max")
    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "collapse_watchdog_fallback" in codes
    assert result.planner == "linear_fallback"
    assert result.estimator is not None
    assert result.estimator.budget_fired == "watchdog"


def test_watchdog_is_off_in_ci_and_deterministic_mode(monkeypatch) -> None:
    """CI and deterministic runs never arm the wall clock (memo D5(v))."""

    monkeypatch.setenv("CI", "1")
    assert collapse_estimator.watchdog_enabled() is False
    monkeypatch.delenv("CI")
    monkeypatch.setenv("TORCHLENS_DETERMINISTIC", "1")
    assert collapse_estimator.watchdog_enabled() is False
    monkeypatch.delenv("TORCHLENS_DETERMINISTIC")
    monkeypatch.setenv("TORCHLENS_COLLAPSE_WATCHDOG", "0")
    assert collapse_estimator.watchdog_enabled() is False


def test_pathological_pre_gate_still_declines(monkeypatch) -> None:
    """Raw ops above 20x the constant decline outright (the ONE wall left)."""

    trace = tl.trace(_Stack(3), torch.randn(2, 4))
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 0)
    with pytest.warns(TorchLensWarning, match="skipping smart collapse") as caught:
        result = collapse_optimizer.select_collapse_plan(trace, RenderContext(), mode="max")
    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "collapse_pathological_skip" in codes
    assert result.declined
    assert "collapse_ops_ceiling" in (result.reason or "")


def test_frontier_allocation_is_instrumented() -> None:
    """Admitted selections record their peak frontier allocation (item 14)."""

    trace = tl.trace(_Stack(8), torch.randn(2, 4))
    result = collapse_optimizer.select_collapse_plan(trace, RenderContext(), mode="max")
    assert result.estimator is not None
    assert result.estimator.peak_frontier_records > 0
    assert result.estimator.budget_fired is None
    assert result.estimator.predicted_ms > 0
    assert result.estimator.actual_ms >= 0


def test_focused_context_is_a_real_remedy(monkeypatch) -> None:
    """The U-gate admits a small rendered universe on a large raw trace.

    The historical raw-op gate refused a seven-node ``vis_call_depth``
    problem citing compute (memo D5 witness). With the gate on U, a reduced
    context re-admits the quality planner even when raw ops exceed the
    constant.
    """

    class _SharedLoop(nn.Module):
        """ONE weight-shared block called ten times: rolled mode merges it."""

        def __init__(self) -> None:
            super().__init__()
            self.block = _Block()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the shared block ten times."""

            for _ in range(10):
                x = self.block(x)
            return x

    trace = tl.trace(_SharedLoop(), torch.randn(2, 4))
    # Constant 10: the unrolled universe exceeds it while the ROLLED
    # universe (one merged multi-pass block) passes the U-gate; raw ops stay
    # under the 20x pathological gate.
    monkeypatch.setattr(collapse_optimizer, "COLLAPSE_OPTIMIZER_MAX_OPS", 10)
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        unrolled = collapse_optimizer.select_collapse_plan(
            trace, RenderContext(vis_mode="unrolled"), mode="max"
        )
        rolled = collapse_optimizer.select_collapse_plan(
            trace, RenderContext(vis_mode="rolled"), mode="max"
        )
    assert unrolled.planner == "linear_fallback"  # unrolled U over the gate
    assert rolled.planner != "linear_fallback"  # rolled U re-admits
