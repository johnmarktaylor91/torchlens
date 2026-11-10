"""Weak view tables for the row-facade record kinds (``Buffer`` first).

A facade record (``Buffer``, and later ``Param``, ``Module``, ``ModuleCall``,
``GradFn``) keeps every stored field in one row of a per-trace kind table
(``TraceCore.kind_rows[kind]``), so the object itself holds only the store and
its row and can be built again from what the trace owns. A trace container
that names such a record by row holds no record object; a door builds the view
on demand through the table here.

Each kind's table maps a row to a weak reference to the view handed out for
it, so identity holds while anyone keeps the view (``trace.buffers["a"] is
trace.buffers["a"]`` inside one expression, or for as long as a variable holds
it) and a dropped view is built again on the next read. A view handed out
from a built trace holds that trace in its ``_owner`` slot, which is what
makes a chained read work; the table holds the view weakly, so the reference
runs one way and a trace nobody holds is still freed by reference counting.

The tables live on the trace core's ``ViewTables`` (``kinds``), next to the
``Op`` and ``Layer`` tables, and follow their rule: while those tables are
pinned (the trace is being built) a view is handed out without an owner.
"""

from __future__ import annotations

import weakref
from typing import Any

from .record_rows import CORE_KEY, ROW_KEY

_ref = weakref.ref
_new = object.__new__
_set = object.__setattr__


def kind_table(core: Any, kind: str) -> dict[int, Any]:
    """Return ``core``'s weak view table for ``kind`` (row to weak reference), creating it."""

    kinds = core._views.kinds
    table = kinds.get(kind)
    if table is None:
        table = kinds[kind] = {}
    return table


def view_at(table: dict[int, Any], store: Any, cls: type, row: int, owner: Any) -> Any:
    """Return the live view of ``row`` cached in ``table``, building one over ``store`` if none is.

    Parameters
    ----------
    table:
        The kind's weak view table (``kind_table``).
    store:
        The kind's row store; a cached view over another store is replaced.
    cls:
        Facade class to build (it must have an ``_owner`` slot).
    row:
        Row of the record in ``store``.
    owner:
        Trace the view holds, or ``None`` (the caller passes ``None`` while
        the trace is built).

    Returns
    -------
    Any
        The facade record for ``row``.
    """

    reference = table.get(row)
    view = reference() if reference is not None else None
    if view is None or view.__dict__.get(CORE_KEY) is not store:
        view = _new(cls)
        state: dict[str, Any] = view.__dict__
        state[CORE_KEY] = store
        state[ROW_KEY] = row
        _set(view, "_owner", owner)
        table[row] = _ref(view)
    elif owner is not None and view._owner is None:
        _set(view, "_owner", owner)
    return view


def rows_in(records: Any, core: Any, kind: str) -> list[int] | None:
    """Return each record's row in ``core``'s ``kind`` table, or ``None`` if one has none there.

    A record bound to another store (a detached record, or a table a refresh
    left frozen) cannot be named by row, so the caller keeps the records.
    """

    store = None if core is None else core.kind_rows.get(kind)
    if store is None or getattr(core, "_views", None) is None:
        return None
    rows: list[int] = []
    for record in records:
        state = getattr(record, "__dict__", None)
        if state is None or state.get(CORE_KEY) is not store:
            return None
        rows.append(state[ROW_KEY])
    return rows
