"""The GradFn-call marker stack and the one resolver of the executing node.

Every node a capture walked pushes a marker from its ATen prehook (the node,
the call and pass the capture predicts, the capture, the engine run it fired
in) and its posthook removes exactly that marker, so the stack nests even when
several captures hook one node and their hooks interleave (prehooks of the
first and second capture, then posthooks in the same order).

Three readers ask which GradFn call a dispatched call ran under: the
executed-backward census (``_backward_dispatch``), the private ATen recorder
(``_aten_capture``) and the checkpoint unpack-hook evidence (``backward``).
They all call :func:`executing_grad_fn_ref`, so they resolve the same node.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

_Marker = tuple[int, int, int, int, int | None]
"""``(node object id, call index, pass index, owner trace id, engine graph-task id)``."""

_ACTIVE_BACKWARD_GRAD_FN_REFS: ContextVar[tuple[_Marker, ...]] = ContextVar(
    "torchlens_active_aten_grad_fn_refs", default=()
)


def _graph_task_id() -> int | None:
    """Return the running engine's graph-task id, or ``None`` without one."""

    from .tensor_tracking import _current_backward_graph_task_id

    return _current_backward_graph_task_id()


def begin_grad_fn_marker(
    object_id: int, call_index: int, pass_index: int, owner_id: int
) -> _Marker:
    """Push the marker of one GradFn call about to run and return it.

    Parameters
    ----------
    object_id
        Python object id of the GradFn node.
    call_index
        One-based call index the owning capture predicts.
    pass_index
        The owning capture's one-based backward pass index.
    owner_id
        ``id()`` of the capture that walked the node.

    Returns
    -------
    tuple
        The pushed marker; pass it to :func:`end_grad_fn_marker`.
    """

    marker = (object_id, call_index, pass_index, owner_id, _graph_task_id())
    _ACTIVE_BACKWARD_GRAD_FN_REFS.set((*_ACTIVE_BACKWARD_GRAD_FN_REFS.get(), marker))
    return marker


def end_grad_fn_marker(marker: _Marker | None) -> None:
    """Remove exactly ``marker`` (its newest copy) from the stack.

    Removing the marker itself, not resetting to the value before its push,
    keeps every other capture's marker for the same node in place whatever
    order the posthooks run in.
    """

    if marker is None:
        return
    current = _ACTIVE_BACKWARD_GRAD_FN_REFS.get()
    for position in range(len(current) - 1, -1, -1):
        if current[position] == marker:
            _ACTIVE_BACKWARD_GRAD_FN_REFS.set(current[:position] + current[position + 1 :])
            return


def _newest(
    markers: tuple[_Marker, ...], owner_id: int, node_id: int | None, others: bool
) -> _Marker | None:
    """The newest marker (of ``node_id`` when given), the owner's first."""

    fallback = None
    for marker in reversed(markers):
        if node_id is not None and marker[0] != node_id:
            continue
        if marker[3] == owner_id:
            return marker
        if others and fallback is None:
            fallback = marker
    return fallback


def executing_grad_fn_ref(
    owner_id: int, *, resolver: Callable[[], Any] | None, other_captures: bool = False
) -> tuple[int, int, int] | None:
    """The GradFn call marker of the node executing now, or ``None``.

    The marker is matched to the node autograd reports as executing, the
    owner's own marker first when several captures hooked that node. A call
    made by a node no capture walked belongs to none of them, except inside an
    engine run nested under a walked node (a reentrant checkpoint's inner
    backward, a different graph-task id), whose calls are that node's work.
    Without an executing-node resolver nothing is charged: the innermost
    marker would be a guess.

    Parameters
    ----------
    owner_id
        ``id()`` of the capture asking.
    resolver
        ``torch._C._current_autograd_node`` as each reader's module resolves
        it through the compatibility layer, or ``None`` when torch lacks it.
    other_captures
        Whether a node only another capture walked may answer with that
        capture's marker. The census asks for it (such calls are another
        trace's work, counted apart from unattributed ones); the ATen recorder
        and the unpack evidence do not, since a parent they cannot resolve in
        their own trace would be left unresolved.

    Returns
    -------
    tuple[int, int, int] | None
        ``(node object id, call index, pass index)``.
    """

    markers = _ACTIVE_BACKWARD_GRAD_FN_REFS.get()
    if not markers:
        return None
    if resolver is None:
        return None
    node = resolver()
    if node is None:
        return None
    match = _newest(markers, owner_id, id(node), other_captures)
    if match is not None:
        return match[:3]
    top = _newest(markers, owner_id, None, other_captures)
    task = _graph_task_id()
    if top is not None and task is not None and top[4] not in (None, task):
        return top[:3]
    return None


__all__ = ["begin_grad_fn_marker", "end_grad_fn_marker", "executing_grad_fn_ref"]
