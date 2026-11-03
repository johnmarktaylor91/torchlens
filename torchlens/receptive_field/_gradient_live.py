"""Resolve the graph-connected tensor behind a same-object logging copy.

An op that returns its input object unchanged (eval-mode dropout, an identity
``to``) is logged against a clone so the record owns a distinct payload. The
clone's ``grad_fn`` is TorchLens's own ``CloneBackward0``; no downstream op
consumes it, so a gradient probe seeded there finds no path. The op's recorded
``grad_fn_class_name`` still names the user op's node. When the two disagree,
the live tensor is the parent payload whose ``grad_fn`` is the clone's single
next node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ._errors import ReceptiveFieldUnavailableError

if TYPE_CHECKING:
    from .._data_classes.op import Op

_MAX_DEPTH = 8

_RECAPTURE_RECIPE = (
    "tl.trace(model, x.requires_grad_(True), "
    'capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"))'
)


def _trace_op(op: Op, label: str) -> Any | None:
    """Resolve a parent label through the op's owning trace, or ``None``."""

    try:
        return op.source_trace.ops[label]
    except (AttributeError, KeyError, RuntimeError, ValueError, TypeError):
        return None


def _payload(op: Any) -> torch.Tensor | None:
    """Return an op's saved tensor payload, or ``None``."""

    try:
        value = op.out
    except (AttributeError, RuntimeError, ValueError, KeyError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def live_graph_tensor(op: Op, value: torch.Tensor, _depth: int = 0) -> torch.Tensor:
    """Return the graph-connected tensor an op's payload stands for.

    Parameters
    ----------
    op:
        Operation whose saved payload is ``value``.
    value:
        The saved payload.

    Returns
    -------
    torch.Tensor
        The parent payload the logging clone was taken from when ``value`` is
        such a clone, else ``value`` itself.
    """

    node = value.grad_fn
    if node is None or type(node).__name__ != "CloneBackward0" or _depth > _MAX_DEPTH:
        return value
    recorded = getattr(op, "grad_fn_class_name", None)
    if recorded is None or str(recorded).rsplit(".", 1)[-1] == "CloneBackward0":
        return value
    next_functions = tuple(node.next_functions)
    if len(next_functions) != 1 or next_functions[0][0] is None:
        return value
    target = next_functions[0][0]
    for label in tuple(getattr(op, "parents", ()) or ()):
        parent = _trace_op(op, label)
        if parent is None:
            continue
        payload = _payload(parent)
        if payload is None:
            continue
        candidate = live_graph_tensor(parent, payload, _depth + 1)
        if candidate.grad_fn is target and tuple(candidate.shape) == tuple(value.shape):
            return candidate
    return value


def _saved_tensor(op: Op, purpose: str) -> torch.Tensor:
    """Return one live saved tensor or raise the typed recapture diagnostic.

    Parameters
    ----------
    op:
        Operation whose payload is required.
    purpose:
        Human-readable purpose used in the error.

    Returns
    -------
    torch.Tensor
        Live saved tensor payload.

    Raises
    ------
    ReceptiveFieldUnavailableError
        If no graph-connected tensor payload is available.
    """

    try:
        value = op.out
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise ReceptiveFieldUnavailableError(
            f"The {purpose} op {op.label!r} has no saved payload. Recapture with "
            f"{_RECAPTURE_RECIPE}."
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise ReceptiveFieldUnavailableError(
            f"The {purpose} op {op.label!r} does not expose a tensor payload. Recapture with "
            f"{_RECAPTURE_RECIPE}."
        )
    return live_graph_tensor(op, value)


__all__ = ["live_graph_tensor"]
