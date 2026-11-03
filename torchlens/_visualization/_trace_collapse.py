"""Trace collapse functions reached from the records through ``_data_classes._doors.visualization``.

Bodies of ``Trace.module_collapse_order`` and ``collapse_order`` live here as
``f(trace, ...)`` functions; the record keeps one thin method per name with its
public docstring and forwards to these through the door. ``collapse_plan`` and
``collapse_schedule`` are served by ``Drawing.collapse_plan`` and
``Drawing.collapse_schedule`` on the retained trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .._errors import InvalidArgumentError
from . import collapse_optimizer
from .auto_collapse import analyze_collapse, collapse_order as _collapse_order
from .collapse_estimator import PATHOLOGICAL_OP_MULTIPLIER
from .collapse_ladder import collapse_schedule as _collapse_schedule
from .collapse_optimizer import select_collapse_level, select_collapse_plan
from .collapse_plan import RenderContext

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .collapse_plan import CollapsePlan, CollapseSchedule

__all__: list[str] = []


def module_collapse_order(trace: Trace) -> list[tuple[str, float]]:
    """Body of ``Trace.module_collapse_order``; the record keeps the thin spelling and docstring."""

    ceiling = collapse_optimizer.COLLAPSE_OPTIMIZER_MAX_OPS * PATHOLOGICAL_OP_MULTIPLIER
    if len(trace._layer_list) > ceiling:
        return []
    signals = analyze_collapse(trace).signals
    largest = max((signal.hidden_ops for signal in signals.values()), default=0)
    scores: dict[str, float] = {}
    for module in trace.modules:
        address, signal = module.address, signals[module.address]
        # ``eligible`` carries every gate: the root, at most one Layer, and
        # the only child that hides its non-root parent's ops (the parent ranks).
        ranked = signal.eligible and largest > 0
        scores[address] = round(signal.hidden_ops / largest, 6) if ranked else 0.0
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def collapse_order(
    trace: Trace,
    mode: Literal["auto", "max"] = "auto",
) -> list[tuple[str, float]]:
    """Body of ``Trace.collapse_order``; the record keeps the thin spelling and docstring."""

    return _collapse_order(trace, mode=mode)


def collapse_plan(
    trace: Trace,
    mode: Literal["auto", "max"] | float = "auto",
    context: RenderContext | None = None,
) -> CollapsePlan:
    """Body of ``Drawing.collapse_plan``; the drawing carries the public docstring."""

    if isinstance(mode, float):
        if not 0.0 <= mode <= 1.0:
            raise InvalidArgumentError(
                f"collapse float level must be in [0.0, 1.0]; received {mode!r}",
                code="collapse_level_invalid",
                remedy="pass a collapse level between 0.0 and 1.0",
                argument="mode",
            )
    elif mode not in {"auto", "max"}:
        raise InvalidArgumentError(
            f"mode must be one of 'auto', 'max', or a float in [0.0, 1.0]; received {mode!r}",
            code="collapse_mode_invalid",
            remedy="pass mode='auto', 'max', or an in-range float",
            argument="mode",
        )

    resolved_context = RenderContext() if context is None else context
    result = (
        select_collapse_level(trace, resolved_context, mode)
        if isinstance(mode, float)
        else select_collapse_plan(trace, resolved_context, mode=mode)
    )
    if result.declined:
        reason = result.reason or "unsupported render context"
        raise InvalidArgumentError(
            f"collapse plan unavailable: {reason}",
            code="collapse_plan_unavailable",
            remedy="use a render context and mode the collapse optimizer supports",
            reason=reason,
        )
    return result.plan


def collapse_schedule(
    trace: Trace,
    context: RenderContext | None = None,
) -> CollapseSchedule:
    """Body of ``Drawing.collapse_schedule``; the drawing carries the public docstring."""

    resolved_context = RenderContext() if context is None else context
    return _collapse_schedule(trace, resolved_context)
