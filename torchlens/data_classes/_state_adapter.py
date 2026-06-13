"""Class-agnostic live-state enumeration and restoration helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

_T = TypeVar("_T")


def state_items(obj: Any) -> Iterable[tuple[str, Any]]:
    """Yield the live state fields stored on an object.

    Parameters
    ----------
    obj:
        Object whose live state should be enumerated.

    Returns
    -------
    Iterable[tuple[str, Any]]
        ``(field_name, field_value)`` pairs. Dict-backed objects return their
        ``vars(obj).items()`` view; slotted objects yield set slots in MRO slot
        order.
    """

    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict is not None:
        return instance_dict.items()
    return _slot_items(obj)


def state_new(cls: type[_T]) -> _T:
    """Create an uninitialized instance of ``cls``.

    Parameters
    ----------
    cls:
        Class to instantiate without calling ``__init__``.

    Returns
    -------
    _T
        New uninitialized instance.
    """

    return object.__new__(cls)


def state_restore(obj: _T, mapping: Mapping[str, Any]) -> _T:
    """Restore fields onto an object without using class-specific state storage.

    Parameters
    ----------
    obj:
        Object to mutate.
    mapping:
        Field values to install on ``obj``.

    Returns
    -------
    _T
        The same object, after field restoration.
    """

    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict is not None:
        instance_dict.update(mapping)
        return obj
    for field_name, field_value in mapping.items():
        object.__setattr__(obj, field_name, field_value)
    return obj


def _slot_items(obj: Any) -> Iterable[tuple[str, Any]]:
    """Yield set slot values for a slotted object."""

    for slot_name in _slot_names(type(obj)):
        try:
            yield slot_name, object.__getattribute__(obj, slot_name)
        except AttributeError:
            continue


def _slot_names(cls: type[Any]) -> tuple[str, ...]:
    """Return state-bearing slot names declared across the class MRO."""

    slot_names: list[str] = []
    seen: set[str] = set()
    for mro_cls in reversed(cls.__mro__):
        raw_slots = getattr(mro_cls, "__slots__", ())
        if isinstance(raw_slots, str):
            raw_slots = (raw_slots,)
        for slot_name in raw_slots:
            if slot_name in {"__dict__", "__weakref__"} or slot_name in seen:
                continue
            seen.add(slot_name)
            slot_names.append(slot_name)
    return tuple(slot_names)
