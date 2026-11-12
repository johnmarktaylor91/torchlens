"""Per-op field reads for the finishing steps, one column at a time.

The finishing steps (the lookup-key postcondition, tensor undecoration and the
graph-shape hash) read a handful of fields of every op. Through the ``Op``
facade each read is one ``Op.__getattribute__`` call or more; here each field
is read once for all ops, as a column, when the ops share one store and the
field is stored under its own name. Every value is what ``getattr(op, name)``
returns: ``OpColumns.column`` returns a plain cell unchanged and reads every
other cell through the field's getter, which answers exactly as the facade.
A field the column view does not model, or a column read that refuses, is
read through the facade for every op.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from .._data_classes._op_columns import column_rows_by_label
from .._utils._record_group_table import op_attr

__all__ = ("ABSENT", "Bound", "bind_ops", "read_field", "row_views", "write_field")

#: Marker for an attribute the op does not have (``getattr`` would raise).
ABSENT: Any = object()

#: ``(OpColumns, store row per op)`` for ops sharing one store, else ``None``.
Bound = tuple[Any, list[int]] | None


def bind_ops(ops: Sequence[Any]) -> Bound:
    """Return the ops' column view and store rows, or ``None`` for the facade path."""

    lookup = column_rows_by_label(dict(enumerate(ops)))
    if lookup is None:
        return None
    columns, rows = lookup
    return columns, list(rows.values())


def read_field(ops: Sequence[Any], name: str, bound: Bound) -> list[Any]:
    """Return ``getattr(op, name, ABSENT)`` for every op in ``ops``.

    Parameters
    ----------
    ops:
        The ops, in the order the values are returned.
    name:
        Attribute name, as the facade reads it.
    bound:
        ``bind_ops(ops)``.

    Returns
    -------
    list[Any]
        One value per op; ``ABSENT`` where the op has no such attribute.
    """

    if bound is not None and op_attr(name) == name:
        columns, rows = bound
        try:
            return columns.column(name, rows)
        except (AttributeError, KeyError, ValueError):
            pass  # not a stored field, or a cell the facade refuses: read the facade
    return [getattr(op, name, ABSENT) for op in ops]


def write_field(ops: Sequence[Any], name: str, values: Sequence[Any], bound: Bound) -> None:
    """Set field ``name`` of every op in ``ops`` to its value in ``values``.

    A field stored under its own name and not guarded by the facade is written
    through the column view (``op._internal_set``, which for such a field is
    the facade's plain write); every other field is written through the facade.
    """

    put = None
    if bound is not None and op_attr(name) == name:
        try:
            put = bound[0].setter(name)
        except ValueError:
            put = None  # a field only the facade may write
    if put is None:
        for op, value in zip(ops, values, strict=True):
            setattr(op, name, value)
        return
    for row, value in zip(bound[1], values, strict=True):
        put(row, value)


def row_views(
    ops: Sequence[Any], names: Sequence[str], bound: Bound | None = None
) -> list[SimpleNamespace]:
    """Return one namespace per op holding the named attributes the op has.

    An attribute the op lacks is absent from its namespace too, so
    ``getattr(view, name, default)`` answers as ``getattr(op, name, default)``.
    ``bound`` is ``bind_ops(ops)`` when the caller already has it.
    """

    if bound is None:
        bound = bind_ops(ops)
    columns = [read_field(ops, name, bound) for name in names]
    views = []
    for values in zip(*columns, strict=True):
        view = SimpleNamespace()
        state = view.__dict__
        for name, value in zip(names, values, strict=True):
            if value is not ABSENT:
                state[name] = value
        views.append(view)
    if not columns:
        views = [SimpleNamespace() for _op in ops]
    return views
