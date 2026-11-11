"""Clear the Parameter tensor references ops hold once param finalization is done.

Kept apart from ``finalization`` (a size-capped module). An op whose
``parent_params`` cell is still default-omitted holds no reference and is
left unwritten: the freeze settles the same fresh empty list into it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .._trace_core.cell_states import STATE_DEFAULT, LazyCell

__all__ = ("clear_parent_params",)

_object_getattribute = object.__getattribute__


def _holds_default(op: Any, fid_cache: dict[int, int]) -> bool:
    """Return whether ``op``'s building-store ``parent_params`` cell is default-omitted."""

    try:
        store = _object_getattribute(op, "_core")
        row = _object_getattribute(op, "_row")
    except AttributeError:
        return False
    rows = store.rows_building() if store is not None else None
    if rows is None:
        return False
    fid = fid_cache.get(id(store))
    if fid is None:
        fid = fid_cache[id(store)] = store.layout.fid_by_name["parent_params"]
    cell = rows[row][fid]
    return cell.__class__ is LazyCell and cell.state == STATE_DEFAULT


def clear_parent_params(ops: Iterable[Any]) -> None:
    """Write an empty ``parent_params`` list into every op that may hold one.

    Parameters
    ----------
    ops:
        The trace's ops (``Trace._layer_list``).
    """

    fid_cache: dict[int, int] = {}
    for op in ops:
        if not _holds_default(op, fid_cache):
            op._graph_parent_params = []
