"""The canonical refusal for an attribute a ``Trace`` does not have.

``Trace.__getattr__`` routes every failed lookup here: a husked trace refuses
typed, a removed member names its replacement, and a plausible-but-absent name
points at the fields that answer it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils.facade import refuse_removed_member

if TYPE_CHECKING:
    from .trace import Trace

#: Where the live ``Trace`` class is defined; ``is_live_trace`` matches by identity.
_TRACE_MODULE = __name__.rsplit(".", 1)[0] + ".trace"

# Plausible-but-absent attribute names, mapped to the fields that answer them. A
# frontier-scale user's first question is "how big is this capture?", and the
# singular ``activation_memory`` spelling (which IS an ``Op`` field, meaning that
# one op's payload bytes) has no single correct Trace-level meaning: the whole
# forward's tensors and the subset ``save=`` retained are different numbers, and
# collapsing them into one alias would make the answer ambiguous rather than
# available. So the names route to the real fields instead of becoming one.
_MISSING_ATTR_HINTS: dict[str, str] = {
    "activation_memory": (
        "use total_activation_memory for every tensor computed in the forward, or "
        "saved_activation_memory for just the payloads save= retained "
        "(Op.activation_memory is the per-op figure; forward_peak_memory is the "
        "measured runtime peak)."
    ),
    "memory": (
        "use total_activation_memory / saved_activation_memory for activations, "
        "total_param_memory for parameters, or forward_peak_memory for the measured "
        "runtime peak."
    ),
    "total_memory": (
        "use total_activation_memory for activations and total_param_memory for "
        "parameters; forward_peak_memory is the measured runtime peak."
    ),
    "footprint": (
        "use total_activation_memory / saved_activation_memory / total_param_memory, "
        "or forward_peak_memory for the measured runtime peak."
    ),
}


#: Public Trace members removed by the 2026-10-01 shim removal (remove-and-
#: rename, no alias): a lookup names the replacement, typed.
_REMOVED_TRACE_MEMBERS: dict[str, str] = {
    "replay": "use trace.push(...) -- Trace.replay was renamed",
    "replay_from": "use trace.push_from(...) -- Trace.replay_from was renamed",
    "rerun": "use trace.run(model, x) -- Trace.rerun was renamed",
    "attach_hooks": (
        "use trace.do(site, edit, model=model, inputs=x) -- attach_hooks folded into do; "
        "stage without running with intervention=InterventionOptions(engine='set_only')"
    ),
    "clear_hooks": (
        "use trace.clear_edits(site) or trace.clear_edits(handle=...) -- clear_hooks was "
        "removed; a fresh fork() starts without staged edits"
    ),
    "collapse_plan": (
        "use trace.draw(collapse='auto', ...).collapse_plan -- the collapse plan is a member "
        "of the Drawing a forward draw returns"
    ),
    "collapse_schedule": (
        "use trace.draw(...).collapse_schedule -- the collapse schedule is a member of the "
        "Drawing a forward draw returns"
    ),
    "collapse_order": (
        "use trace.draw(collapse='auto', ...).collapse_plan -- the plan carries the ranking; "
        "trace.module_collapse_order still lists every module's collapse score"
    ),
    "save_new_outs": (
        "use trace.run(model, x) -- save_new_outs left the public surface; the refresh engine is internal"
    ),
    "validate_saved_outs": (
        "use trace.validate_forward_pass(...) -- validate_saved_outs was its removed alias"
    ),
    "conditional_then_entry_edges": (
        "use trace.graph.conditional_arm_entry_edges (one mapping for then, elif and else arms)"
    ),
    "conditional_elif_entry_edges": (
        "use trace.graph.conditional_arm_entry_edges (one mapping for then, elif and else arms)"
    ),
    "conditional_else_entry_edges": (
        "use trace.graph.conditional_arm_entry_edges (one mapping for then, elif and else arms)"
    ),
}


def _raise_missing_trace_attribute(trace: Trace, name: str) -> Any:
    """Raise the canonical error for one missing Trace attribute.

    Parameters
    ----------
    trace:
        Trace on which attribute lookup failed.
    name:
        Missing attribute name.

    Raises
    ------
    AttributeError
        Always, with an actionable memory-field hint when available.
    """

    if getattr(trace, "__dict__", {}).get("_tl_cleaned_up", False):
        # One typed refusal for every reader of a husked trace, instead of a
        # raw AttributeError naming whichever private field the reader hit
        # first (b6-opus R25: summary/iteration/getitem/draw/receptive_fields
        # each leaked a different private name).
        from .._errors import TraceCleanedUpError

        raise TraceCleanedUpError(
            f"this Trace was husked by cleanup(), so {name!r} (like every "
            "logged field) has been deleted",
            remedy="re-capture with tl.trace(...); cleanup() permanently empties a Trace",
            attribute=name,
        )
    refuse_removed_member(type(trace).__name__, name, _REMOVED_TRACE_MEMBERS)
    hint = _MISSING_ATTR_HINTS.get(name)
    if hint is not None:
        raise AttributeError(f"{type(trace).__name__!s} object has no attribute {name!r}; {hint}")
    raise AttributeError(f"{type(trace).__name__!s} object has no attribute {name!r}")


def is_live_trace(obj: Any) -> bool:
    """Return whether ``obj`` is a ``Trace`` that still holds its stores.

    Readers that accept any capture-shaped subject (a ``PartialTrace``, a
    layer/module accessor, an ``Op``, a ``Recording``, or a trace husked by
    ``cleanup()``) check this before reading a private Trace store, so only a
    live Trace reaches the direct, fail-closed read and every other subject
    keeps the reader's empty answer.
    """

    # Identity by class, not isinstance: importing ``trace`` here would make
    # this module part of the Trace import cycle, and every reader imports it.
    for cls in type(obj).__mro__:
        if cls.__name__ == "Trace" and cls.__module__ == _TRACE_MODULE:
            return not obj.__dict__.get("_tl_cleaned_up", False)
    return False
