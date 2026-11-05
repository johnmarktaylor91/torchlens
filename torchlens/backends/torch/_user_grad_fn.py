"""The user op's live autograd node, stamped on a logging copy for its op record.

A same-object in-place return or a pass-through container member is
logged against a safe copy whose own ``grad_fn`` is TorchLens's ``CloneBackward``
bookkeeping node. The wrapper stamps the USER op's live node on the copy
(``tl_user_grad_fn``) and the op record reads it back, so the copy's node is never
recorded as the operation's autograd handle.

An in-place write through a view also rebases the view's autograd history:
``rebase_history`` gives the base a fresh ``CopySlices`` node that takes the
place of the view's own node in the graph. ``rebind_rebased_view_producers``
moves the ops whose recorded node autograd discarded onto that ``CopySlices``,
or, when a recorded op still reads the view's node, records ``CopySlices`` as
the writer's node.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import torch

from ..._state import pause_logging
from ..._utils._autograd_view_nodes import (
    autograd_node_sequence_nr,
    is_view_backward_class as _is_view_backward_class,
)
from ..._utils._internal_read import internal_scalar_read
from ..._utils._tensor_label import get_tensor_label

_USER_GRAD_FN_ATTR = "tl_user_grad_fn"

# Bound on the view-chain walk below (view chains are short; the bound only keeps
# a malformed graph from looping).
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


def _view_chain_to_prior(start: Any, copy_slices: Any) -> list[Any]:
    """Return the view nodes from ``start`` down to ``CopySlices``'s first input, or ``[]``.

    ``rebase_history`` points ``CopySlices``'s first edge at the base's prior node.
    The chain is walked by first-input edges from the receiver's recorded node;
    every node on it must be a view backward node, and the walk must land exactly
    on that prior node (a view taken before an earlier write of the same base
    meets the base lineage further back and is refused). Reaching
    ``CopySlices``, a non-view node or the depth bound returns ``[]``.
    """

    prior = _first_input_node(copy_slices)
    chain: list[Any] = []
    node = start
    while prior is not None and node is not None and len(chain) < _MAX_VIEW_REBASE_DEPTH:
        if node is copy_slices or not _is_view_backward_class(type(node).__name__):
            return []
        chain.append(node)
        node = _first_input_node(node)
        if node is prior:
            return chain
    return []


def _copy_slices_is_newest(copy_slices: Any, handles: Mapping[str, Any], skip: set[int]) -> bool:
    """Return whether ``CopySlices`` was created after every recorded node outside ``skip``.

    A node without a sequence number (or a torch without ``_sequence_nr``) proves
    nothing, so the answer is ``False``.
    """

    newest = autograd_node_sequence_nr(copy_slices)
    if newest is None:
        return False
    for handle in handles.values():
        if handle is None or id(handle) in skip:
            continue
        handle_nr = autograd_node_sequence_nr(handle)
        if handle_nr is None or handle_nr >= newest:
            return False
    return True


def _chain_has_recorded_reader(chain: list[Any], handles: Mapping[str, Any]) -> bool:
    """Return whether a recorded node outside ``chain`` has an edge into it."""

    chain_ids = {id(node) for node in chain}
    for handle in handles.values():
        if handle is None or id(handle) in chain_ids:
            continue
        edges = getattr(handle, "next_functions", None) or ()
        if any(id(edge[0]) in chain_ids for edge in edges if edge[0] is not None):
            return True
    return False


def _orphaned_view_nodes(start: Any, copy_slices: Any, handles: Mapping[str, Any]) -> list[Any]:
    """Return the view nodes a view write's ``CopySlices`` provably cut out of the graph.

    ``rebase_history`` drops the written view's node (for a view of a view, every
    intermediate view node) from the graph. The chain from the receiver's
    recorded node is returned only when all of this is proven:

    - ``_view_chain_to_prior`` walks it to ``CopySlices``'s first input;
    - no other recorded op's node has an edge into the chain (a reader of the
      view before the write keeps the view's node alive in the graph). Only
      recorded nodes are seen: a consumer whose op was not recorded is outside
      this proof;
    - ``CopySlices`` was created after every recorded op's node, so it is this
      write's node and not one an unlogged earlier write left behind.

    Anything else proves nothing and returns ``[]``.
    """

    chain = _view_chain_to_prior(start, copy_slices)
    if not chain or _chain_has_recorded_reader(chain, handles):
        return []
    skip = {id(node) for node in chain}
    return chain if _copy_slices_is_newest(copy_slices, handles, skip) else []


def _writer_node_for_read_view(
    start: Any, copy_slices: Any, handles: Mapping[str, Any]
) -> Any | None:
    """Return ``CopySlices`` as the writer's node when a recorded reader keeps the view's nodes.

    The view's chain is proven (``_view_chain_to_prior``) and ``CopySlices`` is
    this write's node (created after every recorded node, the chain included),
    but another recorded op reads a chain node, so the view producers keep their
    nodes. The write's gradient then flows only through ``CopySlices``: the
    receiver's regenerated view node is a view OF it that backward visits only
    when the view is read again. ``None`` when any of this is unproven.
    """

    chain = _view_chain_to_prior(start, copy_slices)
    if not chain or not _chain_has_recorded_reader(chain, handles):
        return None
    return copy_slices if _copy_slices_is_newest(copy_slices, handles, set()) else None


def rebind_rebased_view_producers(trace: Any, receiver: Any) -> Any | None:
    """Rebind ops to an in-place view write's ``CopySlices``; return the writer's node.

    Called during capture only, after an in-place op ran and before the receiver
    takes the op's label; postprocess reads the handle index and never writes it.
    When the receiver is a view, autograd's ``rebase_history`` gave the base a new
    ``CopySlices`` node that carries the write's gradient. Two shapes are bound:

    - no recorded op reads the view's nodes: the op that produced the view (and
      any intermediate view op) recorded a node no backward pass can reach. Each
      such op's recorded handle moves to ``CopySlices`` (for a view of a view,
      every view op on the chain shares it), so the recorded backward graph keeps
      the forward topology. The writer keeps its regenerated view node, which
      reads ``CopySlices``; the return value is ``None``.
    - a recorded op read the view before the write: the view producers keep their
      nodes, and the writer's node is ``CopySlices`` (returned), the node backward
      visits when the written base is read. The receiver's regenerated view node
      is then only a view of it.

    The binding is fail-closed: only under grad mode, for a fresh ``CopySlices``
    (no op records it yet) that the receiver's regenerated node reads from, and
    when the proof helpers establish the shape. Anything else returns ``None``
    and leaves the records unchanged, so the backward-graph invariant decides.

    Parameters
    ----------
    trace:
        Active capture trace.
    receiver:
        The live tensor the in-place op wrote through.

    Returns
    -------
    Any | None
        The node to record for the in-place op instead of the receiver's
        ``grad_fn``, or ``None`` to keep the receiver's ``grad_fn``.
    """

    handles = getattr(getattr(trace, "capture_events", None), "grad_fn_handles_by_label_raw", None)
    if handles is None or not isinstance(receiver, torch.Tensor) or not torch.is_grad_enabled():
        return None
    with pause_logging(), internal_scalar_read():
        base = receiver._base if receiver.requires_grad and receiver._is_view() else None
        if base is None:
            return None
        copy_slices = base.grad_fn
        view_node = receiver.grad_fn
        label = get_tensor_label(receiver)
    if (
        label is None
        or type(copy_slices).__name__ != "CopySlices"
        or _first_input_node(view_node) is not copy_slices
        or any(handle is copy_slices for handle in handles.values())
    ):
        return None
    orphaned = _orphaned_view_nodes(handles.get(label), copy_slices, handles)
    writer_node = (
        None if orphaned else _writer_node_for_read_view(handles.get(label), copy_slices, handles)
    )
    if not orphaned and writer_node is None:
        return None
    for raw_label, handle in list(handles.items()):
        if any(handle is node for node in orphaned):
            handles[raw_label] = copy_slices
    # Pin the node for the trace's lifetime, as ``backward._strong_grad_fn_refs``
    # pins every recorded node, so its ``id()`` (the GradFn key) is never reused.
    trace.__dict__.setdefault("_backward_gradfn_refs", []).append(copy_slices)
    return writer_node
