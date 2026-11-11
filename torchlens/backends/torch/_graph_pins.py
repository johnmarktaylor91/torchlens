"""Release the autograd pins a trace holds for a forward pass it no longer records.

A trace pins every grad-fn node it registers (``_backward_gradfn_refs``) so a
node's id cannot be reused while ``_BACKWARD_GRAD_FN_REGISTRY`` maps that id to
the trace. A refresh projection (``Trace._save_new_outs`` and the live
re-saves built on it) replaces the trace's forward record with a new pass,
resets the hook owner map, and registers the new pass's nodes. The previous
pass's nodes were never unpinned: each refresh kept one more autograd graph
alive, with the gradient hooks on it (already disowned by the reset owner
map), for the trace's lifetime. A rerun's graph adoption (``intervention._rerun``)
re-registers the same way. ``release_superseded_pins`` clears them before the
new pass registers.
"""

from __future__ import annotations

import weakref
from typing import Any

__all__ = ["pin_epoch", "release_superseded_pins", "reset_backward_rebind_state"]

_PIN_EPOCHS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
"""Per-trace count of pin releases, keyed weakly; dies with the trace."""


def pin_epoch(trace: Any) -> int:
    """Return how many times ``trace`` released a superseded pass's pins.

    A cache keyed on the pinned-node list must include it: after a release the
    list can hold as many nodes as before, but different ones.

    Parameters
    ----------
    trace:
        Trace whose release count is read.

    Returns
    -------
    int
        Number of releases so far (0 for a trace never refreshed).
    """

    try:
        return _PIN_EPOCHS.get(trace, 0)
    except TypeError:  # Non-weakref-able stand-ins (tests) were never released.
        return 0


def reset_backward_rebind_state(trace: Any) -> None:
    """Clear a refresh target's hook bookkeeping and its superseded pass's pins.

    Resets the hooked-tensor keys and the one-owner-per-label map (the rebind
    re-registers ownership for the new pass), then ``release_superseded_pins``.

    Parameters
    ----------
    trace:
        Projection target about to rebind its backward hooks.
    """

    trace.__dict__["_tl_backward_hooked_tensor_keys"] = set()
    trace.__dict__["_tl_grad_hook_owner_by_label"] = {}
    release_superseded_pins(trace)


def release_superseded_pins(trace: Any) -> None:
    """Drop the pinned nodes and registry keys of the pass ``trace`` stops recording.

    Every pinned node and registry key the trace owns goes, except its
    model-state boundary nodes (later forwards chain onto those;
    ``_register_state_boundary_grad_fn``); the trace's pin epoch advances.
    The caller registers the new pass's nodes afterwards.

    Parameters
    ----------
    trace:
        Trace whose forward record a new pass replaces.
    """

    from .backward import _BACKWARD_GRAD_FN_REGISTRY, _BACKWARD_TRACE_SLOTS
    from .tensor_tracking import _STATE_BOUNDARY_GRAD_FNS

    slot = _BACKWARD_TRACE_SLOTS.get(trace)
    trace_ref = None if slot is None else slot[0]
    refs = trace.__dict__.get("_backward_gradfn_refs")
    kept = [
        node
        for node in refs or ()
        if trace_ref is not None and _STATE_BOUNDARY_GRAD_FNS.get(id(node)) is trace_ref
    ]
    if slot is not None:
        kept_ids = {id(node) for node in kept}
        owned_ids = slot[1]
        for grad_fn_object_id in owned_ids - kept_ids:
            # Only keys still mapped to THIS trace: an id may already name a
            # newer node registered by another trace.
            if _BACKWARD_GRAD_FN_REGISTRY.get(grad_fn_object_id) is trace_ref:
                del _BACKWARD_GRAD_FN_REGISTRY[grad_fn_object_id]
        owned_ids.intersection_update(kept_ids)
    if refs is not None:
        trace.__dict__["_backward_gradfn_refs"] = kept
    _PIN_EPOCHS[trace] = pin_epoch(trace) + 1
