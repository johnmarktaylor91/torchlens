"""The user op's live autograd node, stamped on a logging copy for its op record.

A same-object in-place return or a pass-through container member is
logged against a safe copy whose own ``grad_fn`` is TorchLens's ``CloneBackward``
bookkeeping node. The wrapper stamps the USER op's live node on the copy
(``tl_user_grad_fn``) and the op record reads it back, so the copy's node is never
recorded as the operation's autograd handle.

An in-place write through a view also rebases the view's autograd history:
``rebase_history`` gives the base a fresh ``CopySlices`` node that takes the
place of the view's own node in the graph. ``rebind_rebased_view_producers``
records ``CopySlices`` as the in-place op's node and moves the ops whose
recorded node autograd discarded onto it. Every other live view of the same base
is rebased too: autograd gives it a regenerated view node over ``CopySlices`` at
its next read, so its op moves onto that node.
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
from ..._utils._tensor_label import (
    get_tensor_label,
    session_storage_alias_candidates,
    storage_alias_index_key,
)

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


def _view_write_node(start: Any, copy_slices: Any, handles: Mapping[str, Any]) -> Any | None:
    """Return ``CopySlices`` as the in-place op's node when the view write is proven.

    The view's chain is proven (``_view_chain_to_prior``) and ``CopySlices`` is
    this write's node (created after every recorded node). ``CopySlices`` runs
    the write's backward for the base, so every gradient of the write passes
    through it; the receiver's regenerated view node is only a view OF it, which
    backward visits only when the written view is read again. ``None`` when any
    of this is unproven.
    """

    proven = bool(_view_chain_to_prior(start, copy_slices)) or _rewrites_previous_write(
        start, copy_slices
    )
    if not proven or not _copy_slices_is_newest(copy_slices, handles, set()):
        return None
    return copy_slices


def _rewrites_previous_write(start: Any, copy_slices: Any) -> bool:
    """Return whether the receiver's recorded node is the base's previous ``CopySlices``.

    A view written a second time carries the first write's label, whose node is
    the ``CopySlices`` of that write; the second write's ``CopySlices`` reads it as
    the base's prior node. A chain of zero view nodes: anything else is ``False``.
    """

    return (
        start is not None
        and type(start).__name__ == "CopySlices"
        and _first_input_node(copy_slices) is start
    )


def _fresh_view_write(trace: Any, receiver: Any) -> tuple[dict[str, Any], Any, Any] | None:
    """Return ``(handles, receiver's recorded node, CopySlices)`` for a fresh view write.

    Only under grad mode, for a grad-requiring view whose base now holds a
    ``CopySlices`` that no op records yet and that the receiver's regenerated
    node reads from; ``None`` otherwise.
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
    return handles, handles.get(label), copy_slices


def _live_sibling_views(receiver: Any) -> list[tuple[Any, str]]:
    """Return ``(view, label)`` for every other live labeled grad view of the receiver's base.

    Read from the session's storage-alias index; one entry per label, in session
    order. The caller holds ``pause_logging`` and ``internal_scalar_read``.
    """

    base = receiver._base
    try:
        storage_key = storage_alias_index_key(receiver.untyped_storage())
    except (RuntimeError, ValueError, AttributeError):
        return []
    if storage_key is None:
        return []
    found: list[tuple[Any, str]] = []
    seen_labels: set[str] = set()
    for alias in session_storage_alias_candidates(storage_key):
        if (
            alias is receiver
            or not isinstance(alias, torch.Tensor)
            or not alias.requires_grad
            or not alias._is_view()
            or alias._base is not base
        ):
            continue
        label = get_tensor_label(alias)
        if label is not None and label not in seen_labels:
            seen_labels.add(label)
            found.append((alias, label))
    return found


def _regenerated_sibling_node(
    alias: Any, label: str, copy_slices: Any, handles: Mapping[str, Any]
) -> Any | None:
    """Return the node a sibling view's op moves to after the write, or ``None``.

    Proven only when the op's recorded node is a pre-write view chain down to
    ``CopySlices``'s first input that no other recorded op reads (as for the
    written view's producers), and the sibling's regenerated node is a view node
    over ``CopySlices`` that no op records. Reading ``alias.grad_fn`` regenerates
    the node now, as autograd does at the sibling's next read.
    """

    chain = _view_chain_to_prior(handles.get(label), copy_slices)
    if not chain or _chain_has_recorded_reader(chain, handles):
        return None
    try:
        node = alias.grad_fn
    except RuntimeError:
        return None
    if (
        node is None
        or not _is_view_backward_class(type(node).__name__)
        or _first_input_node(node) is not copy_slices
        or any(handle is node for handle in handles.values())
    ):
        return None
    return node


def _rebind_sibling_views(receiver: Any, copy_slices: Any, handles: dict[str, Any]) -> list[Any]:
    """Move each provable sibling view's op onto its regenerated node; return the nodes.

    The write rebases the whole base, so a view of it taken before the write and
    read after it goes through a node autograd regenerates over ``CopySlices``;
    the view's pre-write node is no longer in the graph.
    """

    moved: list[Any] = []
    with pause_logging(), internal_scalar_read():
        for alias, label in _live_sibling_views(receiver):
            node = _regenerated_sibling_node(alias, label, copy_slices, handles)
            if node is not None:
                handles[label] = node
                moved.append(node)
    return moved


def rebind_rebased_view_producers(trace: Any, receiver: Any) -> Any | None:
    """Bind an in-place view write to the base's ``CopySlices``; return the writer's node.

    Called during capture only, after an in-place op ran and before the receiver
    takes the op's label; postprocess reads the handle index and never writes it.
    When the receiver is a view, autograd's ``rebase_history`` gave the base a new
    ``CopySlices`` node that runs the write's backward. The in-place op owns it:
    it is returned and recorded as the op's node, whichever of the base, the view
    or another alias is read afterwards (a later read of the view goes through a
    regenerated view node that reads ``CopySlices``).

    When no recorded op reads the view's nodes, the op that produced the view
    (and any intermediate view op) also recorded a node no backward pass can
    reach; each such op's handle moves to ``CopySlices`` too (for a view of a
    view, every view op on the chain shares it), since its backward is folded
    into the write's. A recorded reader of the view before the write keeps the
    view producers on their own nodes.

    Every other live view of the same base whose op recorded a pre-write view
    node moves onto the node autograd regenerates for it over ``CopySlices``
    (``_rebind_sibling_views``), under the same proof rules.

    The binding is fail-closed: ``_fresh_view_write`` and the proof helpers must
    establish the shape. Anything else returns ``None`` and leaves the records
    unchanged, so the backward-graph invariant decides.

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

    found = _fresh_view_write(trace, receiver)
    if found is None:
        return None
    handles, start, copy_slices = found
    writer_node = _view_write_node(start, copy_slices, handles)
    if writer_node is None:
        return None
    orphaned = _orphaned_view_nodes(start, copy_slices, handles)
    for raw_label, handle in list(handles.items()):
        if any(handle is node for node in orphaned):
            handles[raw_label] = copy_slices
    siblings = _rebind_sibling_views(receiver, copy_slices, handles)
    # Pin the nodes for the trace's lifetime, as ``backward._strong_grad_fn_refs``
    # pins every recorded node, so their ``id()`` (the GradFn key) is never reused.
    trace.__dict__.setdefault("_backward_gradfn_refs", []).extend([copy_slices, *siblings])
    return writer_node
