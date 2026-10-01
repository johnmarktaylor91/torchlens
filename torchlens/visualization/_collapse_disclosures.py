"""Human-facing collapse disclosure warnings (collapse memo D4).

Extracted from ``auto_collapse.py`` (C05 fix cycle, R43 file-size ratchet):
the no-silent-floor rule's warning half. The typed facts ride
``OptimizerResult`` (``planner``/``k_cap_exhausted``/``root_own_units``) and
the render caption carries a visible notice; these warnings are the
human-facing disclosure at plan-selection time.
"""

from __future__ import annotations

import warnings
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
    total_units = _optimizer_total_units(trace, context)
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
