"""Typed-code provocation tests for the C05 render-substrate disclosures.

Three C05 warning codes are disclosure-tier (never refusals):
``layout_engine_stderr`` (render_execution), ``collapse_floor_fallback``
and ``collapse_near_uncollapsed`` (auto_collapse, collapse memo D4).
The natural floor/near-uncollapsed provocations arise on huge graphs
(K_CAP exhaustion on cached decoders -- diagnostic-tier renders), so the
disclosure body is provoked at its unit seam with a real trace and a
real ``OptimizerResult`` shaped like the fallback cases.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.visualization._collapse_disclosures import (
    NEAR_UNCOLLAPSED_FRACTION,
    _warn_undisclosed_floor,
)
from torchlens.visualization.collapse_optimizer import (
    OptimizerResult,
    _optimizer_total_units,
)
from torchlens.visualization.collapse_plan import (
    RenderContext,
    collapse_plan_for_trace,
    count,
)
from torchlens.visualization.render_execution import surface_layout_stderr


class _TwoBlockModel(torch.nn.Module):
    """Small two-block model for real-trace disclosure provocations."""

    def __init__(self) -> None:
        """Initialize the two linear blocks."""

        super().__init__()
        self.block_a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.block_b = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks in sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the second block.
        """

        return self.block_b(self.block_a(x))


@pytest.fixture()
def two_block_trace():
    """Yield a traced two-block model and clean it up after the test."""

    trace = tl.trace(_TwoBlockModel(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _uncollapsed_result(trace: tl.Trace, context: RenderContext) -> OptimizerResult:
    """Build a real, segment-free ``OptimizerResult`` over the full universe.

    Parameters
    ----------
    trace:
        Trace whose full render universe seeds the plan.
    context:
        Render context the plan is derived under.

    Returns
    -------
    OptimizerResult
        Frontier-tagged result whose plan is the entire uncollapsed graph.
    """

    plan = collapse_plan_for_trace(trace, None, {}, context)
    return OptimizerResult(
        selected=frozenset(),
        repeat_folds={},
        plan=plan,
        visible_count=count(plan),
        analyze_ms=0.0,
        select_ms=0.0,
        g_star=None,
    )


def test_layout_engine_stderr_warning_carries_code() -> None:
    """A non-silent layout engine surfaces a typed ``layout_engine_stderr``."""

    message = b"Warning: graph is too large for one page; scaling by 0.5\n"
    with pytest.warns(TorchLensWarning) as caught:
        text = surface_layout_stderr(message, engine="dot")
    assert "scaling by 0.5" in text
    codes = [warning.message.fields["code"] for warning in caught]
    assert "layout_engine_stderr" in codes


def test_layout_engine_silence_stays_silent() -> None:
    """Empty or ``None`` stderr never warns and returns the empty string."""

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert surface_layout_stderr(None, engine="dot") == ""
        assert surface_layout_stderr(b"", engine="dot") == ""
        assert surface_layout_stderr("  \n", engine="dot") == ""


def test_floor_fallback_plan_discloses_typed_warning(two_block_trace: tl.Trace) -> None:
    """A published floor-fallback plan warns ``collapse_floor_fallback``."""

    context = RenderContext(vis_mode="unrolled")
    result = dataclasses.replace(
        _uncollapsed_result(two_block_trace, context),
        planner="floor_fallback",
        reason="frontier empty under K_CAP (provocation)",
    )
    with pytest.warns(TorchLensWarning) as caught:
        _warn_undisclosed_floor(two_block_trace, "auto", result, context)
    codes = [warning.message.fields["code"] for warning in caught]
    assert "collapse_floor_fallback" in codes
    disclosed = next(
        str(warning.message)
        for warning in caught
        if warning.message.fields["code"] == "collapse_floor_fallback"
    )
    assert "fell back to the conservative floor plan" in disclosed
    assert "frontier empty under K_CAP (provocation)" in disclosed


def test_near_uncollapsed_plan_discloses_typed_warning(
    two_block_trace: tl.Trace, monkeypatch
) -> None:
    """A nearly-uncollapsed frontier plan warns ``collapse_near_uncollapsed``.

    F11 (memo D8): the disclosure fires only under band pressure -- an
    in-band full graph is auto's CORRECT answer -- so the provocation pins
    the band below this toy's universe.
    """

    from torchlens.visualization import auto_collapse

    monkeypatch.setattr(auto_collapse, "_readable_band_high", lambda trace: 1)
    context = RenderContext(vis_mode="unrolled")
    total_units = _optimizer_total_units(two_block_trace, context)
    assert total_units > 0
    result = dataclasses.replace(
        _uncollapsed_result(two_block_trace, context),
        visible_count=total_units,
    )
    assert result.visible_count >= NEAR_UNCOLLAPSED_FRACTION * total_units
    with pytest.warns(TorchLensWarning) as caught:
        _warn_undisclosed_floor(two_block_trace, "auto", result, context)
    codes = [warning.message.fields["code"] for warning in caught]
    assert "collapse_near_uncollapsed" in codes


def test_collapsed_frontier_plan_stays_silent(two_block_trace: tl.Trace) -> None:
    """A genuinely collapsed frontier plan raises neither disclosure."""

    import warnings

    context = RenderContext(vis_mode="unrolled")
    total_units = _optimizer_total_units(two_block_trace, context)
    result = dataclasses.replace(
        _uncollapsed_result(two_block_trace, context),
        visible_count=max(1, int(total_units * 0.5)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_undisclosed_floor(two_block_trace, "auto", result, context)
