"""Gradient sites at the trace doors: the armed read side and the plan seam.

``Trace.do`` routes a plan whose site addresses the backward pass
(``tl.grad(site)`` or an autograd-node selector) to the gradient-site plan
(``torchlens.intervention._grad_site_plan``): the plan is split into its
activation half and its backward half, preflighted whole, and armed, and
each backward pass end writes one settle row per arming transaction
(:func:`settle_armed_gradient_edits`, called from the torch backward door).
The plan lives in the intervention layer and is reached through
:func:`_plan`, imported on first use; this module keeps the arming metadata
keys, the engine choice and the read side of armed gradient rules. The
mixin methods in ``_trace_intervention`` delegate here.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .._errors import InvalidArgumentError
from .._trace_state import TraceState

if TYPE_CHECKING:
    import torch
    from torch import nn

    from ..intervention._grad_site_plan import _SignalPlan
    from ..options import InterventionOptions
    from .trace import Trace

#: HookSpec metadata naming the attach batch that armed a gradient rule.
_ARMING_GROUP_KEY = "arming_group"
#: HookSpec metadata naming the ``do`` arming row (``None`` for attach_hooks).
_ARMING_EVENT_KEY = "arming_event_id"
#: Arming group of the sticky backward hooks a capture-time ``intervene=``
#: staged (marked ``created_by`` by the capture door); their arming row is the
#: capture's own envelope (``door="trace"``).
_CAPTURE_ARMING_GROUP = "capture"
_CAPTURE_HOOK_MARK = "intervene_backward_selector"


class _SignalDoRequest(NamedTuple):
    """The caller-side arguments of one gradient-site ``do()``.

    ``model`` and ``x`` feed a rerun, ``intervention_options`` is the merged
    ``InterventionOptions``, and ``loss`` is the optional callable of the
    returned trace whose backward runs inside the ``do``.
    """

    model: nn.Module | None
    x: Any
    intervention_options: InterventionOptions
    loss: Callable[[Trace], torch.Tensor] | None


def _plan() -> ModuleType:
    """Return the gradient-site plan module, imported on first use."""

    return importlib.import_module("torchlens.intervention._grad_site_plan")


def signal_plan(hooks_or_site: Any, value_or_hook: Any) -> _SignalPlan | None:
    """Split a ``do()`` input into its activation and backward halves (``None``: no gradient site)."""

    return _plan().signal_plan(hooks_or_site, value_or_hook)


def refuse_invalid_do_loss(loss: Any, plan: _SignalPlan | None) -> None:
    """Refuse a ``loss=`` argument to ``do`` before any mutation."""

    _plan().refuse_invalid_do_loss(loss, plan)


def is_gradient_entry(entry: Any) -> bool:
    """Whether a normalized hook entry's site addresses the backward pass."""

    return bool(_plan().is_gradient_entry(entry))


def sticky_hook_entry(entry: Any, arming_group: str) -> tuple[dict[str, Any], Any, bool]:
    """Return the stored metadata and hook of one attached entry, and whether it is a gradient rule.

    Parameters
    ----------
    entry:
        One normalized hook entry of an ``attach_hooks`` batch.
    arming_group:
        The arming group id of that batch.

    Returns
    -------
    tuple[dict[str, Any], Any, bool]
        The metadata to store, the hook to store and whether the entry's site
        addresses the backward pass.
    """

    metadata = dict(entry.metadata)
    is_gradient = is_gradient_entry(entry)
    if is_gradient:
        # Settle contract: every gradient rule names the attach batch
        # that armed it, so the backward pass end writes one settle
        # row per arming transaction (settle_armed_gradient_edits).
        metadata[_ARMING_GROUP_KEY] = arming_group
        metadata.setdefault(_ARMING_EVENT_KEY, None)
    if metadata.get("facet_write"):
        # Facet-slice entries MUST store the scatter wrapper built by
        # expand_facet_hook_entries as the fire-time hook: storing the raw
        # helper would drop the wrapper, and rerun normalization would then
        # apply the helper to the whole home tensor instead of the selected
        # facet slice. The raw helper is kept in ``helper=`` as provenance.
        stored_hook: Any = entry.normalized_callable
    else:
        stored_hook = (
            entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable
        )
    return metadata, stored_hook, is_gradient


def do_signal_plan(trace: Trace, plan: _SignalPlan, request: _SignalDoRequest) -> Trace:
    """Run one ``do()`` whose plan has a gradient (or autograd-node) site; return the armed trace."""

    return _plan().do_signal_plan(trace, plan, request)


def armed_edits(trace: Trace) -> tuple[Any, ...]:
    """The gradient rules in this trace's recipe, as ``Rule`` clauses (read-only)."""

    return tuple(_plan().armed_edits(trace))


def settle_armed_gradient_edits(trace: Any, *, pass_index: int) -> None:
    """Write the settle rows of one finished backward pass (one per arming transaction)."""

    _plan().settle_armed_gradient_edits(trace, pass_index=pass_index)


def apply_gradient_site_edits(trace: Trace, grad: torch.Tensor, tensor_label: str) -> torch.Tensor:
    """Run the ``tl.grad(site)`` edits that address one op output's gradient (see the plan module)."""

    return cast("torch.Tensor", _plan().apply_gradient_site_edits(trace, grad, tensor_label))


def gradient_rule_sites(trace: Any) -> list[Any]:
    """Return the sites a capture-time ``tl.grad(site)`` rule addresses."""

    return list(_plan().gradient_rule_sites(trace))


def _arming_group(hook_spec: Any) -> str | None:
    """Return the arming group of one sticky hook, or ``None`` for a forward hook."""

    group = hook_spec.metadata.get(_ARMING_GROUP_KEY)
    if group is not None:
        return str(group)
    if hook_spec.metadata.get("created_by") == _CAPTURE_HOOK_MARK:
        return _CAPTURE_ARMING_GROUP
    return None


def _arming_event_id(trace: Any, hook_spec: Any) -> str | None:
    """Return the arming row id of one gradient rule (the capture envelope for capture rules)."""

    event_id = hook_spec.metadata.get(_ARMING_EVENT_KEY)
    if event_id is not None or _arming_group(hook_spec) != _CAPTURE_ARMING_GROUP:
        return None if event_id is None else str(event_id)
    for row in getattr(trace, "state_history", ()):
        if (
            isinstance(row, dict)
            and row.get("op") == "intervention_event"
            and row.get("door") == "trace"
        ):
            return str(row.get("event_id"))
    return None


def select_signal_plan_engine(
    trace: Trace,
    engine: str,
    *,
    model: nn.Module | None,
    x: Any,
    plan: _SignalPlan,
) -> str:
    """Pick the engine for a gradient plan, refusing replay before any mutation.

    Raises
    ------
    InvalidArgumentError
        ``intervention_engine_invalid`` for an unknown engine, and
        ``replay_gradient_edit_unsupported`` for ``engine="replay"`` or an
        ``auto`` engine with neither ``model=`` nor ``x=``.
    EngineDispatchError
        When only one of ``model=`` / ``x=`` is given.
    """

    if engine not in {"auto", "replay", "rerun", "set_only"}:
        raise InvalidArgumentError(
            "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'; "
            f"received {engine!r}",
            code="intervention_engine_invalid",
            remedy="pass engine='auto', 'replay', 'rerun', or 'set_only'",
            argument="engine",
        )
    if engine == "set_only":
        return engine
    if engine == "replay" or (engine == "auto" and model is None and x is None):
        message, remedy = _plan().gradient_edit_replay_text(plan.backward_repr)
        raise InvalidArgumentError(
            message,
            code="replay_gradient_edit_unsupported",
            remedy=remedy,
            argument="hooks_or_site",
        )
    return trace._select_do_engine(engine, model=model, x=x)


def live_backward_graph_available(trace: Trace) -> bool:
    """Whether a backward on this trace can reach its recorded forward.

    A loaded trace carries no live autograd graph, and a replay result's
    values were recomputed off the graph, so gradient rules attached there
    wait for the next rerun.
    """

    if bool(trace.__dict__.get("_loaded_from_bundle", False)):
        return False
    return bool(trace.state != TraceState.REPLAY_PROPAGATED)


def armed_summary_line(trace: Trace) -> str | None:
    """Summary text while any armed gradient rule has not fired here.

    Returns
    -------
    str | None
        ``"N gradient edits armed; call trace.backward(loss)"`` or ``None``.
    """

    if bool(trace.__dict__.get("_loaded_from_bundle", False)):
        annotations = getattr(trace, "annotations", None) or {}
        persisted = annotations.get("armed_gradient_edits")
        if isinstance(persisted, int) and persisted > 0:
            noun = "edit" if persisted == 1 else "edits"
            return f"{persisted} gradient {noun} armed; they cannot run until a rerun"
        return None
    pending = unfired_gradient_rule_count(trace)
    if not pending:
        return None
    noun = "edit" if pending == 1 else "edits"
    return f"{pending} gradient {noun} armed; call trace.backward(loss)"


def unfired_gradient_rule_count(trace: Trace) -> int | None:
    """Count the gradient rules in the recipe that never fired on this trace.

    Returns
    -------
    int | None
        The count, or ``None`` when the recipe holds no gradient rule.
    """

    spec = trace._intervention_spec
    hook_specs = [
        hook_spec
        for hook_spec in (spec.hook_specs if spec is not None else ())
        if _arming_group(hook_spec) is not None
    ]
    if not hook_specs:
        return None
    fired_groups = {
        str(row.get("arming_group"))
        for row in trace.state_history
        if isinstance(row, dict)
        and row.get("op") == "intervention_event"
        and row.get("door") == "backward"
        and row.get("status") == "fired"
    }
    return sum(1 for hook_spec in hook_specs if _arming_group(hook_spec) not in fired_groups)


def refresh_armed_annotation(trace: Trace) -> None:
    """Keep ``annotations["armed_gradient_edits"]`` equal to the unfired count.

    The annotation persists with the trace, so a saved armed result says
    how many gradient edits never ran (the recipe itself travels through
    ``save_intervention``).
    """

    count = unfired_gradient_rule_count(trace)
    annotations = getattr(trace, "annotations", None)
    if count is None:
        if isinstance(annotations, dict):
            annotations.pop("armed_gradient_edits", None)
        return
    if isinstance(annotations, dict):
        annotations["armed_gradient_edits"] = count


def _pass_filtered(target: Any, pass_index: int) -> bool:
    """Whether a target's backward-pass qualifiers exclude ``pass_index``.

    Walks the frozen target for ``backward_pass`` qualifiers; a target with
    none never filters a pass.
    """

    stack: list[Any] = [target]
    qualifiers: list[Any] = []
    while stack:
        node = stack.pop()
        if isinstance(node, (tuple, list)):
            stack.extend(node)
            continue
        kind = getattr(node, "selector_kind", None)
        if kind is None:
            continue
        value = getattr(node, "selector_value", None)
        if kind == "backward_pass":
            qualifiers.append(value)
        elif value is not None and not isinstance(value, (str, int, float)):
            stack.append(value)
    return bool(qualifiers) and all(_qualifier_pass(value) != pass_index for value in qualifiers)


def _qualifier_pass(value: Any) -> int | None:
    """Pass index carried by a ``backward_pass`` qualifier value."""

    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], int):
        return int(value[0])
    pass_index = getattr(value, "pass_number", getattr(value, "pass_index", None))
    return int(pass_index) if isinstance(pass_index, int) else None
