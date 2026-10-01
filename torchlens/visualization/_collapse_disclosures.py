"""Human-facing collapse disclosure warnings (collapse memo D4).

Extracted from ``auto_collapse.py`` (C05 fix cycle, R43 file-size ratchet):
the no-silent-floor rule's warning half. The typed facts ride
``OptimizerResult`` (``planner``/``k_cap_exhausted``/``root_own_units``) and
the render caption carries a visible notice; these warnings are the
human-facing disclosure at plan-selection time.
"""

from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING

from ..errors._base import TorchLensWarning
from ..utils.display import user_stacklevel

if TYPE_CHECKING:
    from .._literals import CollapseLiteral
    from ..data_classes.trace import Trace
    from .collapse_optimizer import OptimizerResult
    from .collapse_plan import RenderContext


#: Fraction of the full rendered universe above which a "collapsed" plan is
#: disclosed as near-uncollapsed (collapse memo D4 release invariant: ``auto``
#: never returns a plan within a small factor of U without disclosure).
NEAR_UNCOLLAPSED_FRACTION = 0.9


def _warn_undisclosed_floor(
    trace: Trace,
    collapse: CollapseLiteral,
    result: OptimizerResult,
    context: RenderContext,
) -> None:
    """Disclose floor-fallback and near-uncollapsed plans (memo D4).

    A 1,688-of-1,712-node "plan" returned with no message is a silent
    contract violation worse than any refusal: the user asked for
    ``collapse='auto'`` and got the entire graph. The typed facts ride
    ``OptimizerResult`` (``planner``/``k_cap_exhausted``/``root_own_units``);
    this warning is the human-facing half, and the render caption carries a
    visible notice.
    """

    from .collapse_optimizer import _optimizer_total_units

    if result.planner == "floor_fallback":
        detail = result.reason or "no optimizer frontier was produced"
        warnings.warn(
            TorchLensWarning(
                f"collapse={collapse!r} could not build an optimized plan and fell "
                f"back to the conservative floor plan ({result.visible_count} visible "
                f"nodes): {detail} "
                "Remedy: reduce the rendered graph with module= focus, "
                "vis_call_depth, or rolled mode",
                code="collapse_floor_fallback",
            ),
            stacklevel=user_stacklevel(),
        )
        return
    from .auto_collapse import _readable_band_high

    total_units = _optimizer_total_units(trace, context)
    # F11 auto unfreeze (memo D8): an in-band full graph is the CORRECT auto
    # answer (auto = first ladder point meeting the cap), so the
    # near-uncollapsed disclosure fires only under band pressure -- the
    # measured silent-violation shape it exists for (141-unit fan floods).
    if total_units <= _readable_band_high(trace):
        return
    if total_units > 0 and result.visible_count >= NEAR_UNCOLLAPSED_FRACTION * total_units:
        warnings.warn(
            TorchLensWarning(
                f"collapse={collapse!r} selected a plan with {result.visible_count} of "
                f"{total_units} rendered units visible -- nearly the uncollapsed graph. "
                "Remedy: reduce the rendered graph with module= focus, "
                "vis_call_depth, or rolled mode",
                code="collapse_near_uncollapsed",
            ),
            stacklevel=user_stacklevel(),
        )


def _quality_budget_ms() -> float:
    """Return the admission budget (runtime import: no module cycle)."""

    from .collapse_estimator import QUALITY_PLANNER_BUDGET_MS

    return QUALITY_PLANNER_BUDGET_MS


def _defensive_constant() -> int:
    """Return the defensive U constant (runtime import: no module cycle)."""

    from .collapse_optimizer import COLLAPSE_OPTIMIZER_MAX_OPS

    return int(COLLAPSE_OPTIMIZER_MAX_OPS)


#: Once-per-trace dedupe for the budget-fallback disclosure on the default
#: ``auto`` path (explicit ``max`` re-warns every call, N15).
_BUDGET_WARNED_TRACES: weakref.WeakSet = weakref.WeakSet()


def _warn_budget_fallback(trace: Trace, mode: str, result: OptimizerResult) -> None:
    """Disclose a budget/watchdog degrade to the fallback planner (memo D5).

    Explicit ``max`` requests re-warn on EVERY call (N15: an explicit
    compaction request must never silently degrade); the default ``auto``
    path dedupes once per trace.
    """

    if mode != "max" and trace in _BUDGET_WARNED_TRACES:
        return
    _BUDGET_WARNED_TRACES.add(trace)
    diagnostics = result.estimator
    fired = diagnostics.budget_fired if diagnostics is not None else None
    remedy = (
        "The render is a coarse-but-legal overview, never an uncollapsed "
        "wall; reduce the rendered graph with module= focus, vis_call_depth, "
        "or rolled mode to re-admit the quality planner."
    )
    if fired in {"watchdog", "allocation"}:
        detail = (
            f"the quality planner was abandoned mid-run ({fired}); this is an "
            f"estimator bug -- record (U={diagnostics.universe_count}, "
            f"W={diagnostics.max_sibling_width}, "
            f"predicted={diagnostics.predicted_ms:.0f} ms, "
            f"actual={diagnostics.actual_ms:.0f} ms)"
            if diagnostics is not None
            else f"the quality planner was abandoned mid-run ({fired})"
        )
        warnings.warn(
            TorchLensWarning(
                f"collapse degraded to the deterministic compact fallback plan "
                f"({result.visible_count} visible nodes): {detail}. {remedy}",
                code="collapse_watchdog_fallback",
            ),
            stacklevel=user_stacklevel(),
        )
        return
    detail = (
        f"the (U, W) work estimator refused admission (U="
        f"{diagnostics.universe_count}, W={diagnostics.max_sibling_width}, "
        f"predicted {diagnostics.predicted_ms:.0f} ms over the "
        f"{_quality_budget_ms():.0f} ms budget or U above "
        f"_defensive_constant()={_defensive_constant()})"
        if diagnostics is not None
        else "the work estimator refused admission"
    )
    warnings.warn(
        TorchLensWarning(
            f"collapse degraded to the deterministic compact fallback plan "
            f"({result.visible_count} visible nodes): {detail}. {remedy}",
            code="collapse_budget_fallback",
        ),
        stacklevel=user_stacklevel(),
    )
