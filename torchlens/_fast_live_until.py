"""Early stop (``until=``) for the guarded fast live run.

``trace.run(inputs=x, until=site, fast=True)`` runs the native forward only as
far as ``site``: the collection hook of the stop site raises
:class:`FastUntilReached` once the site's output is collected, which cuts the
forward there, so nothing downstream executes. Every saved site before the stop
keeps the fast path's per-call path, shape and dtype guards and comes back
graph-connected under the default save; the sites after it, and the model
output, read ``None`` for that run (they did not run), and the report carries a
``RunTruncation``.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from ._fast_live_plans import _FastOutputPlan
from .errors import RunCapabilityUnavailableError
from .runnable import RunnableErrorCode, RunTruncation

__all__ = ["FastUntilReached", "function_prefix_count", "resolve_fast_until", "truncate_after"]


class FastUntilReached(BaseException):
    """Cut the native forward after the ``until=`` site's output was collected.

    A ``BaseException`` so a model's own ``except Exception`` cannot swallow it.
    """


def resolve_fast_until(
    trace: Any,
    module_plans: Sequence[_FastOutputPlan],
    function_plans: Sequence[_FastOutputPlan],
    until: Any,
) -> tuple[str, int]:
    """Return the plan the fast forward stops after.

    Parameters
    ----------
    trace:
        Source trace, to resolve a layer or raw label to its op label.
    module_plans:
        The session's module plans, in hook-firing order.
    function_plans:
        The session's functional plans, in capture order.
    until:
        A saved module address (``"layer2"``, stopping after its last call), a
        pass-qualified call (``"layer2:1"``), or the label of a saved op the
        fast path collects (stopping right after that op's call).

    Returns
    -------
    tuple[str, int]
        ``("module", index)`` or ``("function", index)`` into the plans.

    Raises
    ------
    RunCapabilityUnavailableError
        ``fast_until_site_unsaved`` when the fast path collects no such site,
        so it has no hook to stop at.
    """

    text = until if isinstance(until, str) else ""
    address, _sep, suffix = text.rpartition(":")
    call = int(suffix) if address and suffix.isdigit() else None
    if call is None:
        address = text
    by_address = [
        index
        for index, plan in enumerate(module_plans)
        if plan.address_or_name == address and (call is None or plan.call_index == call)
    ]
    if text and by_address:
        return "module", by_address[-1]
    op = trace.layer_dict_all_keys.get(text) if text else None
    op_label = str(getattr(op, "label", text))
    for kind, plans in (("module", module_plans), ("function", function_plans)):
        for index, plan in enumerate(plans):
            if text and op_label in plan.save_labels:
                return kind, index
    raise RunCapabilityUnavailableError(
        f"fast=True stops only at a site the capture saved; {until!r} is not one. "
        "Remedy: pass a saved module address or saved op label as until=, recapture "
        "with save= covering it, or drop fast=.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="fast_until_site_unsaved",
    )


def _raw_index(trace: Any, label: str) -> int:
    """Return an op's raw index by final label (``0`` when unknown)."""

    return int(getattr(trace.layer_dict_all_keys[label], "raw_index", 0) or 0)


def _stop_raw_index(trace: Any, stop_plan: _FastOutputPlan) -> int:
    """Return the raw index of the stop module's last output op."""

    return max(_raw_index(trace, label) for label in stop_plan.op_labels)


def function_prefix_count(
    trace: Any, stop_plan: _FastOutputPlan, function_plans: Sequence[_FastOutputPlan]
) -> int:
    """Count the functional plans that run up to the stop site.

    Parameters
    ----------
    trace:
        Source trace.
    stop_plan:
        The plan the forward stops after.
    function_plans:
        The session's functional plans, in capture order.

    Returns
    -------
    int
        How many functional calls a faithful prefix must have reported.
    """

    stop_raw = _stop_raw_index(trace, stop_plan)
    return sum(1 for plan in function_plans if _raw_index(trace, plan.op_labels[0]) <= stop_raw)


def _frontier_label(trace: Any, label: str) -> str:
    """Spell the stop site as the ordinary until= report does (no pass suffix on one pass)."""

    op = trace.layer_dict_all_keys[label]
    if int(getattr(op, "num_passes", 1) or 1) == 1:
        return str(getattr(op, "layer_label", label))
    return label


def truncate_after(
    trace: Any,
    stop_plan: _FastOutputPlan,
    until: Any,
) -> RunTruncation:
    """Clear the payloads of every saved op after the stop and describe the cut.

    Parameters
    ----------
    trace:
        The live Trace the fast run refreshes in place.
    stop_plan:
        The plan the forward stopped after.
    until:
        The caller's ``until=`` value.

    Returns
    -------
    RunTruncation
        The ``live_stop_after`` disclosure for the run report.
    """

    stop_raw = _stop_raw_index(trace, stop_plan)
    executed: list[str] = []
    skipped: list[str] = []
    for op in trace.layer_list:
        raw_label = str(getattr(op, "_label_raw", op.label))
        if int(getattr(op, "raw_index", 0) or 0) <= stop_raw:
            executed.append(raw_label)
            continue
        skipped.append(raw_label)
        if bool(getattr(op, "has_saved_activation", False)):
            # Not run this time: the not-available spelling, never the previous
            # run's value. The next full run refreshes the site again.
            op._internal_set("out", None)
            op._internal_set("transformed_out", None)
    trace.raw_output = None
    return RunTruncation(
        regime="live_stop_after",
        requested_sites=(str(until),),
        stopped_at=_frontier_label(trace, stop_plan.op_labels[-1]),
        executed_count=len(executed),
        skipped_count=len(skipped),
        skipped_digest=sha256("\n".join(skipped).encode("utf-8")).hexdigest()[:16],
    )
