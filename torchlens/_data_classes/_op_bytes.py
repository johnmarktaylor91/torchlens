"""Theoretical read-once/write-once byte traffic of one operation record.

The record's ``cost`` group serves these numbers (``op.cost.bytes_read`` and
``op.cost.bytes_written``); the model lives beside the records so the group
reads it without reaching up into the debug package.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from .._quantities import Bytes
from .._utils.tensor_utils import get_memory_amount_from_metadata

if TYPE_CHECKING:
    from .op import Op

_NO_COPY_OPS = frozenset(
    {
        "adjoint",
        "alias",
        "as_strided",
        "conj",
        "detach",
        "expand",
        "flatten",
        "imag",
        "movedim",
        "narrow",
        "permute",
        "real",
        "reshape",
        "select",
        "slice",
        "squeeze",
        "swapaxes",
        "swapdims",
        "t",
        "transpose",
        "unflatten",
        "unsqueeze",
        "view",
    }
)


def _normalized_op_name(op: Op) -> str:
    """Return a normalized operation name for cost policy dispatch.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    str
        Lowercase name without leading/trailing double underscores.
    """

    return str(getattr(op, "func_name", "")).lower().strip("_")


def _metadata_bytes(shape: Any, dtype: Any) -> Bytes | None:
    """Compute dense tensor bytes from shape and native dtype metadata.

    Parameters
    ----------
    shape:
        Tensor shape metadata.
    dtype:
        Native torch dtype metadata.

    Returns
    -------
    Bytes | None
        Dense logical byte count, or ``None`` when metadata is insufficient.
    """

    if shape is None or not isinstance(dtype, torch.dtype):
        return None
    try:
        dimensions = tuple(int(dimension) for dimension in shape)
        if any(dimension < 0 for dimension in dimensions):
            return None
        element_size = int(dtype.itemsize)
    except (AttributeError, TypeError, ValueError):
        return None
    return Bytes(math.prod(dimensions) * element_size)


def _input_bytes(op: Op) -> Bytes | None:
    """Return ideal read-once bytes for distinct graph-parent tensors.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    Bytes | None
        Sum from parent shape/dtype metadata, or ``None`` when any parent cannot
        be resolved or sized.
    """

    try:
        parents = list(op.input_ops.values())
    except (AttributeError, KeyError, RuntimeError, ValueError):
        return None
    if len(parents) != len(getattr(op, "parents", ())):
        return None
    total = 0
    for parent in parents:
        amount = _metadata_bytes(getattr(parent, "shape", None), getattr(parent, "dtype", None))
        if amount is None:
            return None
        total += int(amount)
    return Bytes(total)


def _parameter_bytes(op: Op) -> Bytes | None:
    """Return ideal read-once bytes for parameters consumed by an operation.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    Bytes | None
        Sum from parameter shape/dtype metadata, or ``None`` when incomplete.
    """

    shapes = tuple(getattr(op, "param_shapes", ()) or ())
    try:
        dtypes = tuple(op.param_dtypes)
    except (AttributeError, RuntimeError, ValueError):
        return None
    if len(shapes) != len(dtypes):
        return None
    total = 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        amount = _metadata_bytes(shape, dtype)
        if amount is None:
            return None
        total += int(amount)
    return Bytes(total)


def theoretical_op_bytes(op: Op) -> tuple[Bytes | None, Bytes | None]:
    """Return theoretical ideal read-once/write-once traffic for an operation.

    View and alias operations return zero read and write traffic: this model
    treats them as logical metadata transformations, including ``reshape`` even
    though a particular non-contiguous eager execution can materialize a copy.
    This avoids presenting logical tensor size as measured traffic. For all
    other operations, distinct graph-parent and parameter tensors are read once
    and the dense output is written once. Kernel fusion, caches, allocator
    behavior, and hardware transactions are intentionally outside this model.

    Parameters
    ----------
    op:
        Operation metadata record.

    Returns
    -------
    tuple[Bytes | None, Bytes | None]
        Theoretical bytes read and written. Either side is ``None`` when its
        required shape/dtype metadata is insufficient.
    """

    if _normalized_op_name(op) in _NO_COPY_OPS:
        return Bytes(0), Bytes(0)
    input_bytes = _input_bytes(op)
    parameter_bytes = _parameter_bytes(op)
    bytes_read = (
        None
        if input_bytes is None or parameter_bytes is None
        else Bytes(int(input_bytes) + int(parameter_bytes))
    )
    bytes_written = _metadata_bytes(getattr(op, "shape", None), getattr(op, "dtype", None))
    return bytes_read, bytes_written


__all__ = ["theoretical_op_bytes"]


def activation_bytes(payload: torch.Tensor, shape: Any, dtype: Any) -> Bytes:
    """Return the size in bytes of a saved activation, read from its shape and dtype."""

    return Bytes(get_memory_amount_from_metadata(payload, shape, dtype))
