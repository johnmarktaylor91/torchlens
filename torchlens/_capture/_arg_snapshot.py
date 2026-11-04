"""Op-time snapshots of caller-owned mutable argument containers.

A capture stores an op's non-tensor arguments on its record. Lists, dicts,
sets and bytearrays among them belong to the caller, who may mutate them after
the call (``shape[0] = 4`` after ``x.reshape(shape)``). Storing the caller's
object would let that later mutation rewrite the record, so the record keeps a
copy of every mutable container, nested ones included, taken when the op ran.

Only containers are copied. Tensors, scalars, strings and other objects are
kept by reference, and a tuple whose elements need no copy is returned as is,
so the common scalar argument costs one type lookup.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ("snapshot_mutable_arg", "split_non_tensor_args")

# Exact types whose values never need a copy: the scalar arguments most ops pass.
_ATOMIC_TYPES = frozenset(
    {int, float, bool, str, bytes, type(None), complex, slice, type(Ellipsis)}
)
# Exact container types whose own ``copy`` method keeps the type.
_PLAIN_CONTAINERS = frozenset({list, dict, set})


def snapshot_mutable_arg(value: Any) -> Any:
    """Return ``value`` with every nested mutable container copied.

    Args:
        value: One non-tensor argument as the caller passed it.

    Returns:
        A value equal to ``value`` that shares no list, dict, set or bytearray
        with it. Anything else (tensors included) is the same object.
    """
    if type(value) in _ATOMIC_TYPES:
        return value
    return _snapshot(value, None)


def split_non_tensor_args(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    is_tensor: Callable[[Any], bool],
) -> tuple[list[Any], dict[str, Any]]:
    """Return the snapshotted non-tensor positional and keyword arguments of one call.

    Every backend that records ``non_tensor_pos_args`` / ``non_tensor_kwargs``
    builds them here, so each stores op-time copies of mutable containers.

    Args:
        args: Positional arguments as the caller passed them.
        kwargs: Keyword arguments as the caller passed them.
        is_tensor: The backend's predicate for arguments that are graph tensors.

    Returns:
        ``(positional, keyword)``: the non-tensor arguments in call order, each
        passed through ``snapshot_mutable_arg``.
    """
    # The exact-type test is inlined so the common scalar argument costs no extra call.
    atomic = _ATOMIC_TYPES
    positional = [
        arg if type(arg) in atomic else _snapshot(arg, None) for arg in args if not is_tensor(arg)
    ]
    keyword = {
        key: value if type(value) in atomic else _snapshot(value, None)
        for key, value in kwargs.items()
        if not is_tensor(value)
    }
    return positional, keyword


def _snapshot(value: Any, memo: dict[int, Any] | None) -> Any:
    """Copy the mutable containers in ``value``; ``memo`` maps id to copy for cycles.

    Args:
        value: Value to snapshot.
        memo: Copies made so far in this snapshot, keyed by the original's id.

    Returns:
        The snapshot of ``value``.
    """
    if type(value) in _ATOMIC_TYPES:
        return value
    if isinstance(value, (list, dict, set, bytearray)):
        if memo is None:
            memo = {}
        elif id(value) in memo:
            return memo[id(value)]
        return _copy_container(value, memo)
    if isinstance(value, tuple):
        return _snapshot_tuple(value, memo)
    return value


def _copy_container(
    value: list[Any] | dict[Any, Any] | set[Any] | bytearray, memo: dict[int, Any]
) -> Any:
    """Copy one mutable container (keeping its subclass) and snapshot its items.

    Args:
        value: List, dict, set or bytearray to copy.
        memo: Copies made so far, keyed by the original's id; updated in place.

    Returns:
        The copy, registered in ``memo`` before its items are visited.
    """
    if isinstance(value, bytearray):
        copied_bytes = bytearray(value)
        memo[id(value)] = copied_bytes
        return copied_bytes
    # ``list.copy`` on a list subclass returns a plain list; ``copy.copy`` keeps the subclass.
    copied: Any = value.copy() if type(value) in _PLAIN_CONTAINERS else copy.copy(value)
    memo[id(value)] = copied
    if isinstance(value, list):
        atomic = _ATOMIC_TYPES
        copied[:] = [item if type(item) in atomic else _snapshot(item, memo) for item in value]
    elif isinstance(value, dict):
        for key, item in value.items():
            copied[key] = _snapshot(item, memo)
    # Set members are hashable, so they hold no list, dict or set to copy.
    return copied


def _snapshot_tuple(value: tuple[Any, ...], memo: dict[int, Any] | None) -> tuple[Any, ...]:
    """Snapshot a tuple's items, returning the tuple itself when none changed.

    Args:
        value: Tuple (or tuple subclass such as a named tuple) to snapshot.
        memo: Copies made so far in this snapshot, keyed by the original's id.

    Returns:
        ``value`` when no item needed a copy, else a tuple of the same type.
    """
    atomic = _ATOMIC_TYPES
    items = [item if type(item) in atomic else _snapshot(item, memo) for item in value]
    if all(new is old for new, old in zip(items, value)):
        return value
    if type(value) is tuple:
        return tuple(items)
    make = getattr(type(value), "_make", None)
    if callable(make):
        return make(items)
    return type(value)(items)
