"""Clear the Parameter tensor references ops hold once param finalization is done.

Kept apart from ``finalization`` (a size-capped module). An op whose
``parent_params`` cell is still default-omitted holds no reference and is
left unwritten: the freeze settles the same fresh empty list into it.
"""

from __future__ import annotations

from itertools import repeat
from typing import Any

from .._trace_core.cell_states import STATE_DEFAULT, LazyCell

__all__ = ("clear_parent_params",)

_object_getattribute = object.__getattribute__


def _shared_building_rows(ops: list[Any]) -> tuple[list[list[Any]], int] | None:
    """Return the building rows and ``parent_params`` field id when every op shares one store.

    The slot reads go through ``map`` so they add no per-op Python call.
    """

    if not ops:
        return None
    try:
        store_ids = set(map(id, map(_object_getattribute, ops, repeat("_core"))))
    except AttributeError:
        return None
    if len(store_ids) != 1:
        return None
    store = _object_getattribute(ops[0], "_core")
    rows = store.rows_building() if store is not None else None
    if rows is None:
        return None
    return rows, store.layout.fid_by_name["parent_params"]


def clear_parent_params(ops: list[Any]) -> None:
    """Write an empty ``parent_params`` list into every op that may hold one.

    Parameters
    ----------
    ops:
        The trace's ops (``Trace._layer_list``).
    """

    shared = _shared_building_rows(ops)
    if shared is None:
        for op in ops:
            op._graph_parent_params = []
        return
    rows, fid = shared
    for op, row in zip(ops, map(_object_getattribute, ops, repeat("_row"))):
        cell = rows[row][fid]
        if cell.__class__ is not LazyCell or cell.state != STATE_DEFAULT:
            op._graph_parent_params = []
