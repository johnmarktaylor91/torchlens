"""Seal-pass column algorithms over the raw row cells of the op store.

The structure passes of postprocess (step 2's output ancestry, step 6's buffer
bookkeeping) run here as one loop over the store's raw row cells instead of
one facade or getter call per op. A pass takes this path only when every cell
it reads holds a value the facade would return unchanged (or a shared default
marker whose answer is known); otherwise it declines and the caller runs the
existing body. Both paths visit the same rows in the same order and write the
same cells with the same values, so the stored result is identical.

The row map is built with C-level ``map`` over the facade's slot descriptors,
so building it costs no Python call per op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from .._data_classes._op_columns import _bound_facade, _slow_cell_types
from .._trace_core.cell_states import STATE_DEFAULT, LazyCell
from .._trace_core.op_store import OpRowStore
from .._utils._record_group_table import op_attr
from ._column_passes import _row_or_refuse

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

    from .._data_classes.trace import Trace

__all__ = (
    "SealRows",
    "find_output_ancestors",
    "immutable_markers",
    "markers",
    "plain_column",
    "seal_rows",
    "trace_seal_rows",
)


class SealRows(NamedTuple):
    """One plain row store and the row of each label the caller asked for."""

    store: Any
    rows: dict[str, int]


def seal_rows(labels: Iterable[str], ops: Collection[Any]) -> SealRows | None:
    """Return the shared plain row store of ``ops`` and ``label -> row``, or ``None``.

    ``labels`` and ``ops`` are parallel (a raw layer dict's keys and values).
    ``None`` when ``ops`` is empty, holds a non-``Op`` record, spans more than
    one store, or the store is not a building-phase ``OpRowStore`` with row
    cells (the only shape whose raw cells the column bodies read and write).
    """

    if not ops:
        return None
    op_class, op_vars, _names, _module = _bound_facade()
    if set(map(type, ops)) != {op_class}:
        return None
    core_get = op_vars["_core"].__get__
    stores = set(map(id, map(core_get, ops)))
    if len(stores) != 1:
        return None
    store = core_get(next(iter(ops)))
    if store.__class__ is not OpRowStore or store._rows is None:
        return None
    return SealRows(store, dict(zip(labels, map(op_vars["_row"].__get__, ops), strict=True)))


def trace_seal_rows(trace: Trace) -> SealRows | None:
    """Return :func:`seal_rows` over the raw layer dict when the column bodies apply.

    The single seam the step 2 body consults (tests force the existing body
    through it). ``None`` while a live capture-event index serves lookups,
    as ``_column_passes._column_lookup`` declines.
    """

    if trace.__dict__.get("capture_events") is not None:
        return None
    raw = trace._raw_graph_ws.raw_layer_dict
    return seal_rows(raw.keys(), raw.values())


def _fid(store: Any, name: str) -> int:
    """Return the field id of persisted field ``name`` in ``store``'s layout."""

    return int(store.layout.fid_by_name[name])


def plain_column(store: Any, name: str, rows: list[int]) -> tuple[list[Any], set[type]] | None:
    """Return field ``name``'s raw cells for ``rows`` and their classes, or ``None``.

    ``None`` when a cell holds a value the facade transforms on read (a class
    the field's descriptor hydrates, or an unset or encoded sentinel); default
    markers (``LazyCell``) are returned as they are for the caller to answer.
    """

    _op_class, op_vars, _names, _module = _bound_facade()
    attribute = op_vars[op_attr(name)]
    slow_types, sentinel = _slow_cell_types(name, attribute)
    fid = _fid(store, name)
    cells = store._rows
    column = [cells[row][fid] for row in rows]
    classes = set(map(type, column))
    if object in classes or sentinel.__class__ in classes:
        return None  # the unset / CSR / fact sentinels are plain ``object()`` instances
    for cls in classes:
        if cls is not LazyCell and slow_types and issubclass(cls, slow_types):
            return None
    return column, classes


def markers(column: list[Any]) -> list[LazyCell]:
    """Return the distinct default markers in ``column``."""

    return list({cell for cell in column if cell.__class__ is LazyCell})


def immutable_markers(column: list[Any]) -> bool:
    """Whether every marker in ``column`` answers an immutable default without a write."""

    return all(not marker.make for marker in markers(column))


def _factory_markers(column: list[Any]) -> bool:
    """Whether every marker in ``column`` answers a fresh mutable default on read."""

    return all(
        marker.make and marker.state == STATE_DEFAULT and not marker.whole_column
        for marker in markers(column)
    )


def find_output_ancestors(trace: Trace) -> bool:
    """Step 2 over raw cells; return ``False`` (nothing written) when it declines.

    Same visit order and writes as ``_column_passes.find_output_ancestors``:
    reverse topological order over the raw labels; a child reaching an output
    marks its parent and merges its ``output_descendants`` into the parent's
    set in place. A parent's set still holding its default marker is answered
    as the facade's read answers it (a fresh set from the marker's factory,
    written back and registered as an alias site while building).
    """

    view = trace_seal_rows(trace)
    if view is None:
        return False
    store, rows = view
    every_row = list(rows.values())
    children = plain_column(store, "children", every_row)
    reached = plain_column(store, "has_output_descendant", every_row)
    descendants = plain_column(store, "output_descendants", every_row)
    if children is None or reached is None or descendants is None:
        return False
    if not children[1] <= {list, tuple}:
        return False
    if not reached[1] <= {bool, LazyCell} or not immutable_markers(reached[0]):
        return False
    if not descendants[1] <= {set, LazyCell} or not _factory_markers(descendants[0]):
        return False
    _propagate_output_reach(trace, store, rows)
    return True


def _propagate_output_reach(trace: Trace, store: Any, rows: dict[str, int]) -> None:
    """Run step 2's reverse sweep over the validated raw cells of ``store``."""

    cells = store._rows
    f_children = _fid(store, "children")
    f_reached = _fid(store, "has_output_descendant")
    f_descendants = _fid(store, "output_descendants")
    alias_sites = None if store._sealed else store._alias_sites
    for label in reversed(trace._raw_graph_ws.raw_layer_labels_list):
        try:
            row = rows[label]
        except KeyError:
            row = _row_or_refuse(trace, rows, label)
        own = cells[row]
        for child_label in own[f_children]:
            try:
                child = cells[rows[child_label]]
            except KeyError:
                child = cells[_row_or_refuse(trace, rows, child_label)]
            flag = child[f_reached]
            if flag.__class__ is LazyCell:
                flag = flag.source
            if not flag:
                continue
            own[f_reached] = True
            mine = own[f_descendants]
            if mine.__class__ is LazyCell:
                mine = own[f_descendants] = mine.source()
                if alias_sites is not None:
                    alias_sites[(row, f_descendants)] = mine
            theirs = child[f_descendants]
            if theirs.__class__ is LazyCell:
                theirs = child[f_descendants] = theirs.source()
                if alias_sites is not None:
                    alias_sites[(rows[child_label], f_descendants)] = theirs
            mine |= theirs
