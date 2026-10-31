"""Class-agnostic live-state enumeration and restoration helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

_T = TypeVar("_T")


def state_items(obj: Any) -> Iterable[tuple[str, Any]]:
    """Yield the live state fields stored on an object.

    Classes whose state does not live in ``__dict__``/``__slots__`` (for
    example columnar-backed facades) opt in by defining
    ``__tl_state_items__(self)``, which must yield the complete
    ``(field_name, field_value)`` state and takes precedence over storage
    introspection.

    Parameters
    ----------
    obj:
        Object whose live state should be enumerated.

    Returns
    -------
    Iterable[tuple[str, Any]]
        ``(field_name, field_value)`` pairs. Dict-backed fields are yielded
        first, followed by any distinct set slots in MRO slot order.
    """

    items_hook = getattr(type(obj), "__tl_state_items__", None)
    if items_hook is not None:
        yield from items_hook(obj)
        return
    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict is not None:
        yield from instance_dict.items()
        dict_field_names = set(instance_dict)
        for field_name, field_value in _slot_items(obj):
            if field_name not in dict_field_names:
                yield field_name, field_value
        return
    yield from _slot_items(obj)


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

    Classes that define ``__tl_state_restore__(self, mapping)`` (the columnar
    opt-in counterpart of ``__tl_state_items__``) receive the mapping whole
    and install it into their backing store; the hook takes precedence over
    storage introspection.

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

    restore_hook = getattr(type(obj), "__tl_state_restore__", None)
    if restore_hook is not None:
        restore_hook(obj, mapping)
        return obj
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
