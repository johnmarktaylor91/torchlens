"""Recorded form of a call's arguments for the descriptive resolvers.

The function config (``salient_args.extract_salient_args``) and the forward FLOPs
(``flops.compute_forward_flops``) read only three things from a call's arguments:
each tensor's shape and dtype, and the non-tensor values. Both accept the live
arguments or this recorded form, which holds those three things and no tensor, so
either value can be resolved after the forward from the record and equal the value
the call produced (a freed or mutated input no longer matters).

``record_call_values`` replaces every tensor by a :class:`RecordedTensor` and copies
every container it walks through, so a later mutation of a caller-owned list cannot
reach the record. A tensor inside a container it does not rebuild (a set, a list or
dict subclass, an object attribute) makes it raise :class:`UnrecordableArgument`;
the caller then resolves at the call instead.
"""

from __future__ import annotations

from typing import Any

import torch

from ._arg_snapshot import snapshot_mutable_arg

__all__ = ("RecordedTensor", "UnrecordableArgument", "record_call_values")

_ATOMIC_TYPES = frozenset(
    {int, float, bool, str, bytes, type(None), complex, slice, type(Ellipsis)}
)


class UnrecordableArgument(Exception):
    """A tensor sits where :func:`record_call_values` cannot replace it."""


class RecordedTensor:
    """A tensor argument as the descriptive resolvers see it: shape and dtype at the call."""

    __slots__ = ("dtype", "shape")

    def __init__(self, tensor: torch.Tensor) -> None:
        """Read the tensor's shape (as a plain tuple) and dtype."""

        self.shape: tuple[int, ...] = tuple(tensor.shape)
        self.dtype: torch.dtype = tensor.dtype

    def __eq__(self, other: object) -> bool:
        """Equal when shape and dtype are equal."""

        if not isinstance(other, RecordedTensor):
            return NotImplemented
        return self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self) -> int:
        """Hash of shape and dtype."""

        return hash((self.shape, self.dtype))

    def __repr__(self) -> str:
        """Show shape and dtype."""

        return f"RecordedTensor(shape={self.shape}, dtype={self.dtype})"


def record_call_values(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return ``(args, kwargs)`` with every tensor recorded and every container copied.

    Raises:
        UnrecordableArgument: a tensor sits inside a container this does not rebuild.
    """

    return tuple(_record(value) for value in args), {
        key: _record(value) for key, value in kwargs.items()
    }


def _record(value: Any) -> Any:
    """Record one argument value (see :func:`record_call_values`)."""

    kind = type(value)
    if kind in _ATOMIC_TYPES:
        return value
    if isinstance(value, torch.Tensor):
        return RecordedTensor(value)
    if kind is tuple:
        return tuple(_record(item) for item in value)
    if kind is list:
        return [_record(item) for item in value]
    if kind is dict:
        return {key: _record(item) for key, item in value.items()}
    if isinstance(value, tuple) and hasattr(kind, "_fields"):
        return kind(*(_record(item) for item in value))
    if _holds_tensor(value):
        raise UnrecordableArgument(kind.__name__)
    return snapshot_mutable_arg(value)


def _holds_tensor(value: Any) -> bool:
    """Whether a value this module does not rebuild holds a tensor at any depth."""

    if isinstance(value, torch.Tensor):
        return True
    try:
        if isinstance(value, dict):
            return any(_holds_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple, set, frozenset)):
            return any(_holds_tensor(item) for item in value)
    except RecursionError:
        return True
    return False
