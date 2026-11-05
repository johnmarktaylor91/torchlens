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

import functools
import re
from collections.abc import Mapping
from contextlib import suppress
from typing import Any

import torch

from ..._state import pause_logging
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


_BACKWARD_CLASS = re.compile(r"([A-Z][A-Za-z0-9]*?)Backward\d+")


@functools.cache
def _is_view_backward_class(class_name: str) -> bool:
    """Return whether ``class_name`` is the backward node of an aten view op.

    The class name (``SliceBackward0``) names its forward op (``aten::slice``); the
    op is a view when one of its schemas takes an argument that its result
    aliases without writing it (``Tensor(a) self``). Read from torch's own
    schemas, so new view ops need no list here; any name that does not resolve
    is not a view (the caller then refuses to rebind).
    """

    match = _BACKWARD_CLASS.fullmatch(class_name)
    if match is None:
        return False
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", match.group(1)).lower()
    return _aten_op_is_view(snake) or _aten_op_is_view(f"_{snake}")


def _aten_op_is_view(op_name: str) -> bool:
    """Return whether ``aten::<op_name>`` has a schema whose result aliases an input unwritten."""

    try:
        packet = getattr(torch.ops.aten, op_name)
        schemas = [getattr(packet, overload)._schema for overload in packet.overloads()]
    except (AttributeError, RuntimeError):
        return False
    return any(
        arg.alias_info is not None and not arg.alias_info.is_write
        for schema in schemas
        for arg in schema.arguments
    )


def _sequence_nr(node: Any) -> int | None:
    """Return an autograd node's creation sequence number, or ``None``."""

    read = getattr(node, "_sequence_nr", None)
    return read() if callable(read) else None


def _orphaned_view_nodes(start: Any, copy_slices: Any, handles: Mapping[str, Any]) -> list[Any]:
    """Return the view nodes a view write's ``CopySlices`` provably cut out of the graph.

    ``rebase_history`` points ``CopySlices``'s first edge at the base's prior node
    and drops the written view's node (for a view of a view, every intermediate
    view node) from the graph. The chain from the receiver's recorded node is
    returned only when all of this is proven:

    - every node on it is a view backward node, walked by first-input edges, and
      the last one's first input IS ``CopySlices``'s first input (the base's node
      at the write). A view taken before an earlier write of the same base meets
      the base lineage further back and is refused;
    - no other recorded op's node has an edge into the chain (a reader of the
      view before the write keeps the view's node alive in the graph);
    - ``CopySlices`` was created after every recorded op's node, so it is this
      write's node and not one an unlogged earlier write left behind.

    Anything else (the walk reaches ``CopySlices`` or a non-view node, passes the
    bound, or a check fails) proves nothing and returns ``[]``.
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
            return chain if _chain_owned_by_this_write(chain, copy_slices, handles) else []
    return []


def _chain_owned_by_this_write(
    chain: list[Any], copy_slices: Any, handles: Mapping[str, Any]
) -> bool:
    """Return whether no other recorded node reads ``chain`` and ``CopySlices`` is newest.

    Parameters
    ----------
    chain:
        Candidate dropped view nodes, receiver's node first.
    copy_slices:
        The base's ``CopySlices`` node after the write.
    handles:
        The capture's recorded node per op label.
    """

    newest = _sequence_nr(copy_slices)
    if newest is None:
        return False
    chain_ids = {id(node) for node in chain}
    for handle in handles.values():
        if handle is None or id(handle) in chain_ids:
            continue
        handle_nr = _sequence_nr(handle)
        if handle_nr is None or handle_nr >= newest:
            return False
        edges = getattr(handle, "next_functions", None) or ()
        if any(id(edge[0]) in chain_ids for edge in edges if edge[0] is not None):
            return False
    return True


def rebind_rebased_view_producers(trace: Any, receiver: Any) -> None:
    """Point ops whose node an in-place view write discarded at its ``CopySlices``.

    Called during capture only, after an in-place op ran and before the receiver
    takes the op's label; postprocess reads the handle index and never writes it.
    When the receiver is a view, autograd's ``rebase_history`` gave the base a new
    ``CopySlices`` node that takes the place of the view's node: the op that
    produced the view (and any intermediate view op) recorded a node that no
    backward pass can reach. Each such op's recorded handle moves to the
    ``CopySlices`` node, so the recorded backward graph keeps the forward
    topology (the node between the view's consumers and the base's producer).
    For a view of a view, every view op on the chain shares that node.

    The rebind is fail-closed: it happens only under grad mode, for a fresh
    ``CopySlices`` (no op records it yet) that the receiver's regenerated node
    reads from, and when ``_orphaned_view_nodes`` proves the chain. Anything else
    leaves the records unchanged, so the backward-graph invariant still decides.

    Parameters
    ----------
    trace:
        Active capture trace.
    receiver:
        The live tensor the in-place op wrote through.
    """

    handles = getattr(getattr(trace, "capture_events", None), "grad_fn_handles_by_label_raw", None)
    if handles is None or not isinstance(receiver, torch.Tensor) or not torch.is_grad_enabled():
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
    orphaned = _orphaned_view_nodes(handles.get(label), copy_slices, handles)
    if not orphaned:
        return
    for raw_label, handle in list(handles.items()):
        if any(handle is node for node in orphaned):
            handles[raw_label] = copy_slices
    # Pin the node for the trace's lifetime, as ``backward._strong_grad_fn_refs``
    # pins every recorded node, so its ``id()`` (the GradFn key) is never reused.
    trace.__dict__.setdefault("_backward_gradfn_refs", []).append(copy_slices)
