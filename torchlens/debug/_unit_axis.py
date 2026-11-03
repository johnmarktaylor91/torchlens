"""Pick the unit (feature) axis of an op's activation for ``dead_neurons``.

A unit is one feature a layer computes: a hidden unit of a linear layer, a
channel of a convolution. Which axis holds the units depends on the op, not on
a fixed index: a transformer block's ``(batch, tokens, features)`` activation
keeps its units on the LAST axis, while a convolution's ``(batch, channels,
*spatial)`` output keeps them on the channel axis. Elementwise ops (an
activation function after a convolution) inherit the axis of the same-shaped
op that feeds them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ._common import _op_from_label, _op_label

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace

#: Convolution and pooling functions: ``<name><k>d`` puts channels just before
#: the ``k`` spatial axes (axis 1 when batched, axis 0 when unbatched).
_SPATIAL_FUNC = re.compile(r"(conv|pool)\w*?([123])d")

#: Normalisations whose statistics run per channel on axis 1.
_CHANNEL_NORM_FUNCS = frozenset({"batch_norm", "instance_norm", "group_norm"})

#: How many same-shape parents to walk back before falling back to the last axis.
_MAX_INHERIT_HOPS = 64


def _own_axis(op: Op, ndim: int) -> int | None:
    """Return the unit axis the op's own function fixes, or ``None``."""

    func = str(getattr(op, "func_name", "") or "").lower()
    match = _SPATIAL_FUNC.search(func)
    if match is not None:
        axis = ndim - int(match.group(2)) - 1
        return axis if axis >= 0 else None
    if func in _CHANNEL_NORM_FUNCS and ndim >= 2:
        return 1
    return None


def _same_shape_parent(trace: Trace, op: Op, shape: tuple[int, ...]) -> Op | None:
    """Return the first parent op whose output has exactly ``shape``."""

    for label in op.parents or ():
        parent = _op_from_label(trace, str(label))
        if parent is not None and tuple(getattr(parent, "shape", None) or ()) == shape:
            return parent
    return None


def unit_axis(trace: Trace, op: Op, out: Any, memo: dict[str, int]) -> int | None:
    """Return the unit axis for one activation, or ``None`` when it has none.

    Parameters
    ----------
    trace:
        Completed TorchLens trace (resolves parent ops).
    op:
        The op that produced ``out``.
    out:
        Its saved activation tensor.
    memo:
        Per-call cache of decided axes keyed by op label.

    Returns
    -------
    int | None
        Axis 1 for four or more dimensions (channels first); the op's own
        channel axis for convolutions, pooling and per-channel norms; the axis
        of a same-shaped parent that has one (an activation after a
        convolution); otherwise the last axis. ``None`` below two dimensions,
        where one trace holds no batch of units to compare.
    """

    ndim = int(out.ndim)
    if ndim < 2:
        return None
    if ndim >= 4:
        return 1
    shape = tuple(out.shape)
    chain: list[str] = []
    current: Op | None = op
    axis: int | None = None
    for _ in range(_MAX_INHERIT_HOPS):
        if current is None:
            break
        label = _op_label(current)
        if label in memo:
            axis = memo[label]
            break
        chain.append(label)
        axis = _own_axis(current, ndim)
        if axis is not None:
            break
        current = _same_shape_parent(trace, current, shape)
    resolved = ndim - 1 if axis is None else axis
    for label in chain:
        memo[label] = resolved
    return resolved
