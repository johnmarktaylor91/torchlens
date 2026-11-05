"""The user op's live autograd node, stamped on a logging copy for its op record.

A same-object in-place return or a pass-through container member is
logged against a safe copy whose own ``grad_fn`` is TorchLens's ``CloneBackward``
bookkeeping node. The wrapper stamps the USER op's live node on the copy
(``tl_user_grad_fn``) and the op record reads it back, so the copy's node is never
recorded as the operation's autograd handle.

An in-place write through a view also rebases the view's autograd history:
``rebase_history`` gives the base a fresh ``CopySlices`` node that takes the
place of the view's own node in the graph. ``rebind_rebased_view_producers``
moves the ops whose recorded node autograd discarded onto that ``CopySlices``.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch

from ..._state import pause_logging
from ..._utils._internal_read import internal_scalar_read
from ..._utils._tensor_label import get_tensor_label

_USER_GRAD_FN_ATTR = "tl_user_grad_fn"

# Bound on the autograd-node walks below (view chains and base lineage are short;
# the bound only keeps a malformed graph from looping).
_MAX_VIEW_REBASE_DEPTH = 64

NO_USER_GRAD_FN: object = object()
"""Stamp for a same-object in-place return whose live receiver has no ``grad_fn``.

The output of such an op (a leaf: ``x.detach().requires_grad_(True)``) is logged
against a safe copy whose own ``grad_fn`` is TorchLens's ``CloneBackward``. The
record reads ``tl_user_grad_fn`` for the user's live node; this sentinel says
"the user's node is ``None``" so the copy's bookkeeping node is never recorded
as the op's autograd handle.
"""


def stamp_user_grad_fn(logged: torch.Tensor, live_grad_fn: Any) -> None:
    """Stamp the user's live node (or ``NO_USER_GRAD_FN`` for none) on a logging copy.

    Parameters
    ----------
    logged:
        The safe copy the op is logged against.
    live_grad_fn:
        The user receiver's ``grad_fn`` read before the copy was made.
    """

    with suppress(AttributeError):
        setattr(
            logged,
            _USER_GRAD_FN_ATTR,
            NO_USER_GRAD_FN if live_grad_fn is None else live_grad_fn,
        )


def pop_user_grad_fn(t: torch.Tensor) -> Any:
    """Remove and return the stamped user node, or ``None`` when ``t`` carries no stamp."""

    user_grad_fn = getattr(t, _USER_GRAD_FN_ATTR, None)
    if user_grad_fn is not None:
        with suppress(AttributeError):
            delattr(t, _USER_GRAD_FN_ATTR)
    return user_grad_fn


def recorded_op_grad_fn(user_grad_fn: Any, t: torch.Tensor) -> Any:
    """Return the autograd handle an op record carries for output ``t``.

    The stamped user node when present; ``None`` when the stamp says the user's op
    returned a leaf (the copy's ``CloneBackward`` is bookkeeping, never the op's
    handle); otherwise ``t``'s own ``grad_fn``. The caller wraps the read in
    ``internal_scalar_read`` (a TorchLens bookkeeping read).
    """

    if user_grad_fn is NO_USER_GRAD_FN:
        return None
    return user_grad_fn if user_grad_fn is not None else t.grad_fn


def _first_input_node(node: Any) -> Any:
    """Return the node behind an autograd node's first input edge, or ``None``."""

    edges = getattr(node, "next_functions", None) or ()
    return edges[0][0] if edges else None


def _orphaned_view_nodes(start: Any, copy_slices: Any) -> list[Any]:
    """Return the nodes a view write's ``CopySlices`` cut out of the graph.

    ``rebase_history`` points ``CopySlices``'s first edge at the base's prior node
    and drops the view's node (and, for a view of a view, every intermediate view
    node) from the graph. Walking first-input edges from the receiver's recorded
    node, every node before the walk meets the base lineage behind ``CopySlices``
    is such a node. Any other shape (the walk reaches ``CopySlices`` itself, never
    meets the lineage, or exceeds the bound) proves nothing and returns ``[]``.
    """

    lineage: list[Any] = []
    node = _first_input_node(copy_slices)
    while node is not None and len(lineage) < _MAX_VIEW_REBASE_DEPTH:
        lineage.append(node)
        node = _first_input_node(node)
    chain: list[Any] = []
    node = start
    while node is not None and node is not copy_slices and len(chain) < _MAX_VIEW_REBASE_DEPTH:
        if any(node is base_node for base_node in lineage):
            return chain
        chain.append(node)
        node = _first_input_node(node)
    return []


def rebind_rebased_view_producers(trace: Any, receiver: Any) -> None:
    """Point ops whose node an in-place view write discarded at its ``CopySlices``.

    Called after an in-place op ran and before the receiver takes the op's label.
    When the receiver is a view, autograd's ``rebase_history`` gave the base a new
    ``CopySlices`` node that takes the place of the view's node: the op that
    produced the view (and any intermediate view op) recorded a node that no
    backward pass can reach. Each such op's recorded handle moves to the
    ``CopySlices`` node, so the recorded backward graph keeps the forward
    topology (the node between the view's consumers and the base's producer).
    Only a fresh ``CopySlices`` (no op records it yet) whose regenerated view node
    reads from it, reached by a provable view chain, triggers a rebind; anything
    else leaves the records unchanged.

    Parameters
    ----------
    trace:
        Active capture trace.
    receiver:
        The live tensor the in-place op wrote through.
    """

    handles = getattr(getattr(trace, "capture_events", None), "grad_fn_handles_by_label_raw", None)
    if handles is None or not isinstance(receiver, torch.Tensor):
        return
    with pause_logging(), internal_scalar_read():
        base = receiver._base if receiver.requires_grad and receiver._is_view() else None
        if base is None:
            return
        copy_slices = base.grad_fn
        view_node = receiver.grad_fn
        label = get_tensor_label(receiver)
    if (
        label is None
        or type(copy_slices).__name__ != "CopySlices"
        or _first_input_node(view_node) is not copy_slices
        or any(handle is copy_slices for handle in handles.values())
    ):
        return
    orphaned = _orphaned_view_nodes(handles.get(label), copy_slices)
    if not orphaned:
        return
    rebound = False
    for raw_label, handle in list(handles.items()):
        if any(handle is node for node in orphaned):
            handles[raw_label] = copy_slices
            rebound = True
    if rebound:
        # Pin the node for the trace's lifetime, as ``backward._strong_grad_fn_refs``
        # pins every recorded node, so its ``id()`` (the GradFn key) is never reused.
        trace.__dict__.setdefault("_backward_gradfn_refs", []).append(copy_slices)
