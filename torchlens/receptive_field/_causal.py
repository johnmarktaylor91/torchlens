"""Structural causal-mask detection for attention receptive-field rules.

A causal attention row ``i`` has exactly zero softmax weight on every key
``k > i + offset``: the Jacobian of the row with respect to a masked logit is
``w * (delta - w) = 0`` and ``matmul(weights, value)`` has zero derivative with
respect to masked value rows. The engines express that as a sequence window
``[0, i + offset]``. This module recognizes the structure: an
``is_causal=True`` fused attention call needs no values; a masked softmax is
recognized from the saved weights' zero pattern, which must be the same
lower-triangular prefix pattern on every batch row and head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._capture.arg_positions import _normalize_func_name
from ._rules import ReceptiveFieldRuleContext

if TYPE_CHECKING:
    from .._data_classes.op import Op

_IDENTITY_SHAPED = frozenset(
    {
        "dropout",
        "type",
        "to",
        "contiguous",
        "clone",
        "detach",
        "float",
        "half",
        "double",
        "bfloat16",
    }
)
_SOFTMAX_NAMES = frozenset({"softmax", "logsoftmax"})
_MAX_WEIGHT_HOPS = 8


def saved_value(op: Op) -> torch.Tensor | None:
    """Return the op's saved tensor payload, or ``None`` when it was not retained."""

    try:
        value = op.out
    except (AttributeError, RuntimeError, ValueError, KeyError):
        return None
    return value if isinstance(value, torch.Tensor) else None


def prefix_offset(pattern: torch.Tensor) -> int | None:
    """Return the causal offset of a ``(queries, keys)`` support pattern, else ``None``.

    The pattern is causal when row ``i`` is exactly the prefix ``[0, i + offset]``
    (clipped to the key extent) for one integer ``offset``, at least one row is
    narrower than the key extent and no row is empty.
    """

    if pattern.ndim != 2 or pattern.numel() == 0:
        return None
    queries, keys = pattern.shape
    counts = pattern.sum(dim=1)
    if bool((counts == 0).any()):
        return None
    expected = torch.arange(keys, device=pattern.device)[None, :] < counts[:, None]
    if not torch.equal(pattern, expected):
        return None
    narrowed = counts < keys
    if not bool(narrowed.any()):
        return None
    rows = torch.arange(queries, device=pattern.device)
    offsets = (counts - 1 - rows)[narrowed]
    if int(offsets.unique().numel()) != 1:
        return None
    offset = int(offsets[0])
    if not bool(((rows + offset >= keys - 1) | narrowed).all()):
        return None
    return offset


def causal_offset_from_weights(op: Op, dim: int) -> int | None:
    """Return the causal offset encoded in a softmax op's saved output, else ``None``.

    Parameters
    ----------
    op:
        Captured softmax or log-softmax operation.
    dim:
        Normalized dimension the softmax ran over (the key axis); the query
        axis is the one before it.
    """

    value = saved_value(op)
    if value is None or value.ndim < 2:
        return None
    rank = value.ndim
    key_axis = dim % rank
    if key_axis == 0:
        return None
    query_axis = key_axis - 1
    with torch.no_grad():
        if _normalize_func_name(str(op.func_name)) == "logsoftmax":
            support = torch.isfinite(value)
        else:
            support = value != 0
        support = support.movedim((query_axis, key_axis), (-2, -1))
        support = support.reshape(-1, support.shape[-2], support.shape[-1])
        if support.shape[0] == 0:
            return None
        pattern = support[0]
        if not torch.equal(support, pattern.expand_as(support)):
            return None
        return prefix_offset(pattern)


def softmax_dim(op: Op) -> int | None:
    """Return a softmax op's recorded dimension, or ``None``."""

    dim = ReceptiveFieldRuleContext(op).arg("dim", op.func_config.get("dim", -1))
    return dim if isinstance(dim, int) else None


def _trace_op(op: Op, label: str) -> Any | None:
    """Resolve a parent label through the op's owning trace, or ``None``."""

    try:
        return op.source_trace.ops[label]
    except (AttributeError, KeyError, RuntimeError, ValueError, TypeError):
        return None


def causal_weights_source(op: Op) -> int | None:
    """Return the causal offset of the softmax feeding a weights-times-value product.

    Walks the first parent of ``op`` upward through shape-preserving identity
    ops (dropout, dtype casts, ``contiguous``) to a softmax whose normalized
    dimension is its last axis, and returns that softmax's causal offset.
    """

    if not op.parents:
        return None
    label = op.parents[0]
    shape = tuple(op.input_shapes[0]) if op.input_shapes and op.input_shapes[0] else None
    for _ in range(_MAX_WEIGHT_HOPS):
        parent = _trace_op(op, label)
        if parent is None:
            return None
        name = _normalize_func_name(str(parent.func_name))
        if name in _SOFTMAX_NAMES:
            dim = softmax_dim(parent)
            rank = len(parent.shape)
            if dim is None or rank == 0 or dim % rank != rank - 1:
                return None
            return causal_offset_from_weights(parent, dim)
        if (
            name in _IDENTITY_SHAPED
            and len(parent.parents) == 1
            and (shape is None or tuple(parent.shape) == shape)
        ):
            label = parent.parents[0]
            continue
        return None
    return None


__all__ = [
    "causal_offset_from_weights",
    "causal_weights_source",
    "prefix_offset",
    "saved_value",
    "softmax_dim",
]
