"""Rows of ``Trace.to_pandas``: stored ``Op`` fields read a whole column at a time.

A table reads every field of every op. Through the facade each read costs a
record-member resolution, the ``Op`` attribute protocol and a store lookup;
here a stored field is resolved once per column and read from the store's
backing in one pass (:meth:`OpColumns.column`), and the sibling and co-parent
views resolve each label once per table. Every other field, and every field
of a table whose ops are not plain ``Op`` records on one store, reads per op
through the caller's ``cell``, exactly as before. Values are the facade's: a
column that cannot be read whole (an unset cell, a field the facade
intercepts or serves through a view) is read again per op, so its value or
its error is the facade's own.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from .._data_classes._op_columns import OpColumns, column_rows_by_label
from .._data_classes._op_relations import co_parents, memoized_relation_lookup, siblings
from .._data_classes._op_row_release import _OP_FACADE
from .._utils._record_group_table import OP_STORAGE_ATTRS, op_attr, stored_member

__all__ = ("table_rows",)

#: Op attribute -> the persisted field (store layout key) it reads.
_PERSISTED_BY_ATTR: dict[str, str] = {attr: field for field, attr in OP_STORAGE_ATTRS.items()}

#: Relation columns and the view each one reads (``has_*`` is its non-emptiness).
_RELATION_VIEWS: dict[str, Callable[..., list[str]]] = {
    "siblings": siblings,
    "co_parents": co_parents,
}

Cell = Callable[[Any, str], Any]


def table_rows(
    ops: Sequence[Any], fields: Sequence[str], cell: Cell, per_op: frozenset[str]
) -> list[dict[str, Any]]:
    """Return one ``{field: value}`` dict per op, keys in ``fields`` order.

    Parameters
    ----------
    ops:
        The table's ops, in row order.
    fields:
        Column names, in column order.
    cell:
        ``cell(op, field)``: the per-op read of one cell (the table's own
        special columns, else ``record_member``).
    per_op:
        Fields always read through ``cell``.

    Returns
    -------
    list[dict[str, Any]]
        The rows ``pandas.DataFrame`` is built from.
    """

    plan = _column_plan(ops)
    relations: dict[str, Any] = {}
    columns = [
        _column(ops, field, cell, plan, relations)
        if plan is not None and field not in per_op
        else [cell(op, field) for op in ops]
        for field in fields
    ]
    return [dict(zip(fields, values)) for values in zip(*columns)]


def _column_plan(ops: Sequence[Any]) -> tuple[OpColumns, list[int]] | None:
    """Return the column view and each op's row, or ``None`` to read per op."""

    if not _OP_FACADE or not ops:
        return None
    found = column_rows_by_label(cast(Mapping[str, Any], dict(enumerate(ops))))
    if found is None:
        return None
    view, rows = found
    return view, list(rows.values())


def _column(
    ops: Sequence[Any],
    field: str,
    cell: Cell,
    plan: tuple[OpColumns, list[int]],
    relations: dict[str, Any],
) -> list[Any]:
    """Return one column: a relation view, a whole stored column, or per-op reads."""

    base = field[4:] if field.startswith("has_") else field
    if base in _RELATION_VIEWS:
        values = _relation_column(ops, base, relations)
        return [len(value) > 0 for value in values] if base != field else values
    stored = _stored_column(field, plan)
    if stored is not None:
        return stored
    return [cell(op, field) for op in ops]


def _stored_column(field: str, plan: tuple[OpColumns, list[int]]) -> list[Any] | None:
    """Return ``field`` for every row read whole from the store, or ``None``.

    ``None`` when ``record_member`` would not read ``field`` through one
    stored ``Op`` attribute, when the facade intercepts that attribute, or
    when a cell of the column needs the facade's own read (it raised).
    """

    op_class = _OP_FACADE[0][0]
    attr = stored_member(op_class, field)
    if attr is None or attr in sys.modules[op_class.__module__]._INTERCEPTED_READ_FIELDS:
        return None
    persisted = _PERSISTED_BY_ATTR.get(attr, attr)
    view, rows = plan
    if op_attr(persisted) != attr or persisted not in view.store.layout.fid_by_name:
        return None
    try:
        return view.column(persisted, rows)
    except (AttributeError, KeyError, ValueError):
        return None


def _relation_column(ops: Sequence[Any], base: str, relations: dict[str, Any]) -> list[Any]:
    """Return the ``siblings`` or ``co_parents`` view of every op, computed once per table.

    Each source trace gets one memoized label lookup shared by both views; an
    op without a source trace reads ``[]``, as the per-op view does.
    """

    if base in relations:
        return cast(list[Any], relations[base])
    view = _RELATION_VIEWS[base]
    lookups: dict[int, Callable[[str], Any]] = relations.setdefault("lookups", {})
    values: list[Any] = []
    for op in ops:
        trace = op._source_trace
        if trace is None:
            values.append([])
            continue
        lookup = lookups.get(id(trace))
        if lookup is None:
            lookup = lookups[id(trace)] = memoized_relation_lookup(trace)
        values.append(view(op, lookup))
    relations[base] = values
    return values
