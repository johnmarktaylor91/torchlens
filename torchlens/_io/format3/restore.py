"""Record state restore for the 3.0 reader: one decoded row becomes one record object.

``restore_record`` runs the record's state hook on its row; ``true_up_record`` then puts
back what the hook's default fill and container coercion changed, so load then save is
lossless. Two causes, both reader-side: a saved ``null`` that the hook coerces to its typed
default (``_buffer_persistence`` ``null`` became ``{}``), and a column the save level
omitted (``forward_args`` at ``portable``) that leaves the attribute unset, so a property
over it (``has_saved_forward_args``) raises instead of answering ``False``. Nothing here
touches a class attribute or property: only instance slots the schema names.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import cache
from typing import Any, Final

from ..._data_classes._state_adapter import state_restore
from ..._data_classes.field_policy import StorageKind
from .schema import record_schema

__all__ = ["computed_columns", "restore_record", "true_up_record"]

TLSPEC_KEY: Final = "tlspec_version"
_UNSET: Final = object()


@cache
def computed_columns(kind: str) -> frozenset[str]:
    """Columns the writer fills from a read-only property; loaded objects recompute them."""

    return frozenset(
        name
        for name, column in record_schema(kind).items()
        if column.storage == StorageKind.COMPUTED.value
    )


@cache
def _stored_columns(kind: str) -> frozenset[str]:
    """Columns a loaded object stores itself (every column that is not computed)."""

    return frozenset(record_schema(kind)) - computed_columns(kind)


def restore_record(obj: Any, row: Mapping[str, Any], *, version: int, kind: str) -> None:
    """Run the record's state hook on its decoded row, computed columns dropped, then true up."""

    computed = computed_columns(kind)
    state = {name: value for name, value in row.items() if name not in computed}
    state[TLSPEC_KEY] = version
    setstate = getattr(type(obj), "__setstate__", None)
    if setstate is not None and setstate is not getattr(object, "__setstate__", None):
        setstate(obj, state)
    else:
        state.pop(TLSPEC_KEY)
        state_restore(obj, state)
    true_up_record(obj, row, kind)


def _writable_field(cls: type, name: str) -> bool:
    """Whether ``name`` is instance state the true-up may write: absent from the class, or
    a data descriptor that is not a property (a record's row-cell field). A property, a
    method or a class constant is never written: the writer recomputes those."""

    attr = inspect.getattr_static(cls, name, _UNSET)
    if attr is _UNSET:
        return True
    return not isinstance(attr, property) and hasattr(type(attr), "__set__")


def true_up_record(obj: Any, row: Mapping[str, Any], kind: str) -> None:
    """Restore the saved value of every stored column the state hook changed.

    A column saved ``null`` whose attribute the hook filled with a typed default goes back
    to ``None``; a stored column the save level omitted, left unset by the hook, is set to
    ``None`` so the properties over it answer. Row-cell fields (data descriptors) are
    written through the descriptor; a property, method or class constant is never written:
    the writer recomputes those.
    """

    if getattr(obj, "__dict__", None) is None:
        return
    cls = type(obj)
    for name in _stored_columns(kind):
        if not _writable_field(cls, name):
            continue
        current = getattr(obj, name, _UNSET)
        if name in row:
            if row[name] is None and current is not None and current is not _UNSET:
                setattr(obj, name, None)
        elif current is _UNSET:
            setattr(obj, name, None)
