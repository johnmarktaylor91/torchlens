"""Step 9 by column: the per-op part of final layer info over the building rows.

``labeling._log_final_info_for_layers`` visits every op through the ``Op``
facade: each read and write pays a descriptor, a store call and often an
attribute hook. While the trace's ops still live in one building
``OpRowStore`` the same work is done here column by column on the raw rows:
one list of cells per field, read once, then written back row by row.

The values, their container types, the sharing of group containers and the
op order of every tally are those of the per-op loop. A column that holds a
cell this pass cannot read as the facade would (a pooled cell, a group
reference, a CSR or fact sentinel, a required cell that is unset) makes the
trace not columnar: nothing has been written by then, and the per-op loop
runs instead. A default-omitted cell is answered through the store, as the
facade answers it.

Every column here stays eager. ``step_index`` depends on the op's position
among all the trace's ops and the others are read or rewritten by later
passes of the same trace, so none of them is a derived cell.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import TYPE_CHECKING, Any

from .._quantities import Bytes, Duration
from .._trace_core.cell_states import _MISSING, LazyCell
from .._trace_core.groups import GroupRef
from .._trace_core.op_store import OpRowStore, PooledCell
from .._utils._record_group_table import op_attr
from ._final_info_rename import RENAMED_ATTRS, RENAMED_MEMBERS, rename_label_columns
from ._module_containment import _log_module_hierarchy_info_for_layer

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Cell classes whose raw value is not what the facade would return.
_OPAQUE = frozenset({object, LazyCell, PooledCell, GroupRef})
_CONTAINERS = frozenset({list, tuple, set, frozenset, dict})
_MEMORY = "_cost_activation_memory"
_DURATION = "_cost_duration"
_ATOMIC_CALL = "_module_context_atomic_module_call"
_FX_PATH = "_source_fx_path"
_FX_INDEX = "_source_fx_call_index"
#: The closure fields: properties over a staging set while the trace builds.
#: Any other cell class (the compact bitset) is not read here.
_CLOSURES = ("root_ancestors", "input_ancestors", "output_descendants", "internal_source_ancestors")
_CLOSURE_CLASSES: frozenset[type] = frozenset({set, frozenset, type(None)})

#: Columns read by attribute name: ``name -> cell classes the pass accepts``
#: (``None`` accepts any class the facade returns as stored). An unset cell
#: of one of these raises in the per-op loop, so it makes the trace not columnar.
_REQUIRED: dict[str, frozenset[type] | None] = {
    "type": frozenset({str}),
    "layer_label": None,
    "_label_raw": None,
    "_param_barcodes": _CONTAINERS,
    _MEMORY: frozenset({int, Bytes}),
    _DURATION: frozenset({int, float, Duration}),
    _ATOMIC_CALL: None,
    "modules": frozenset({list, tuple}),
}
#: Columns only written, by attribute name.
_WRITTEN = ("is_input", "is_output", "_graph_is_buffer", "step_index", _FX_PATH, _FX_INDEX)
#: Every column the pass reaches by attribute name, then by member name.
_BY_ATTR = (*_REQUIRED, *_WRITTEN, *RENAMED_ATTRS)
_FX_COLUMNS = ("modules", _FX_PATH, _FX_INDEX)
#: Whether the per-op loop writes each field it reads by attribute under that
#: same attribute; the raw pass writes the cell it read, so it needs this.
_WRITTEN_IN_PLACE = all(op_attr(name) == name for name in RENAMED_ATTRS)


class _NotColumnar(Exception):
    """A column holds a cell the raw pass cannot read as the facade would."""


class _Table:
    """The trace's ops as raw building rows, in op order.

    Parameters
    ----------
    store:
        The building store every op lives in.
    ops:
        The op facades, in trace order.
    row_ids:
        Each op's row id in ``store``.
    rows:
        Each op's list of cells.
    fids:
        Field id by the name this pass uses for the column.
    """

    __slots__ = ("fids", "ops", "row_ids", "rows", "store")

    def __init__(
        self,
        store: Any,
        ops: list[Any],
        row_ids: list[int],
        rows: list[list[Any]],
        fids: dict[str, int],
    ) -> None:
        """Bind the rows to their store and their field ids."""

        self.store = store
        self.ops = ops
        self.row_ids = row_ids
        self.rows = rows
        self.fids = fids

    def read(
        self, name: str, optional: bool = True, classes: Collection[type] | None = None
    ) -> list[Any]:
        """Return one column as the facade would read it, indexed like ``rows``.

        Parameters
        ----------
        name:
            Column name.
        optional:
            Whether an unset cell reads as ``None`` (the per-op loop reads it
            with a ``None`` default) instead of making the trace not columnar.
        classes:
            Exact cell classes the caller's arithmetic accepts, or ``None``.

        Returns
        -------
        list[Any]
            The cell values in op order.
        """

        fid = self.fids[name]
        column = [row[fid] for row in self.rows]
        seen = set(map(type, column))
        if not seen.isdisjoint(_OPAQUE):
            self._answer(column, fid, optional)
            seen = set(map(type, column))
        if classes is not None and not seen.issubset(classes):
            raise _NotColumnar(name)
        return column

    def _answer(self, column: list[Any], fid: int, optional: bool) -> None:
        """Replace the marker cells of ``column`` by the values a read answers."""

        cell_get = self.store.cell_get
        for position, value in enumerate(column):
            if value.__class__ is LazyCell:
                value = column[position] = cell_get(self.row_ids[position], fid)
            if value is _MISSING and optional:
                column[position] = None
            elif value.__class__ in _OPAQUE:
                raise _NotColumnar(fid)


def _field_ids(
    op_cls: Any, layout: Any, by_attr: tuple[str, ...], by_member: tuple[str, ...]
) -> dict[str, int] | None:
    """Return the field id of each named column, or ``None`` when one is not stored.

    A column named by attribute is found through its descriptor on ``op_cls``
    (the class is taken from the ops: this module imports nothing from the
    ``Op`` module, which sits inside the package's import cycle). A column
    named by member is found in the store layout, which lists the member
    names in field order; that also covers the closure fields, whose
    attributes are properties over their original descriptors.
    """

    fids = {}
    for name in by_attr:
        fid = getattr(getattr(op_cls, name, None), "_fid", None)
        if fid.__class__ is not int:
            return None
        fids[name] = fid
    member_fids = layout.fid_by_name
    for name in by_member:
        if name not in member_fids:
            return None
        fids[name] = member_fids[name]
    return fids


def _open_table(
    ops: list[Any], by_attr: tuple[str, ...], by_member: tuple[str, ...]
) -> _Table | None:
    """Return the raw rows of ``ops``, or ``None`` when they are not one building store.

    ``by_attr`` and ``by_member`` name the columns the caller will read or write.
    """

    try:
        op_cls: Any = ops[0].__class__
        stores = list(map(op_cls._core.__get__, ops))
        row_ids = list(map(op_cls._row.__get__, ops))
    except (TypeError, AttributeError, IndexError):
        return None  # no ops, or test doubles in place of the row-store facade
    store = stores[0]
    if store.__class__ is not OpRowStore or stores.count(store) != len(stores):
        return None
    fids = _field_ids(op_cls, store.layout, by_attr, by_member)
    rows = store.rows_building()
    if fids is None or rows is None or store.frozen or store.dataflow_edges is not None:
        return None
    return _Table(store, ops, row_ids, list(map(rows.__getitem__, row_ids)), fids)


def _fold(start: Any, cells: list[Any], quantity: type, plain: type) -> Any:
    """Return ``start + cell + ...`` as the quantity operators compute it.

    ``Bytes`` and ``Duration`` add by unwrapping themselves to the plain
    number, adding natively and wrapping the result; the left fold below is
    that arithmetic in the same order without one operator call per op.

    Parameters
    ----------
    start:
        The running total before the first cell.
    cells:
        The values to add, in op order.
    quantity:
        The quantity class (``Bytes`` or ``Duration``).
    plain:
        Its plain number class (``int`` or ``float``).

    Returns
    -------
    Any
        The total, wrapped when ``start`` or any cell is a ``quantity``.
    """

    wrapped = start.__class__ is quantity or quantity in set(map(type, cells))
    total = plain(start) if start.__class__ is quantity else start
    for cell in cells:
        total += plain(cell) if cell.__class__ is quantity else cell
    return quantity(total) if wrapped else total


def _write_roles_and_steps(trace: Trace, table: _Table, kinds: list[str]) -> None:
    """Write each op's role flags and step index, and count the numbered ops.

    Inputs and buffers get step 0 and outputs ``None`` (numbered once the
    total is known); every other op takes the next step and is counted in
    ``trace._num_ops``.
    """

    fids = table.fids
    f_input, f_output, f_buffer = fids["is_input"], fids["is_output"], fids["_graph_is_buffer"]
    f_step = fids["step_index"]
    next_step = 1
    for row, kind in zip(table.rows, kinds, strict=True):
        is_input = row[f_input] = kind == "input"
        is_output = row[f_output] = kind == "output"
        is_buffer = row[f_buffer] = kind == "buffer"
        if is_input or is_buffer:
            row[f_step] = 0
        elif is_output:
            row[f_step] = None
        else:
            row[f_step] = next_step
            next_step += 1
    if next_step > 1:
        trace._num_ops += next_step - 1


def _log_module_facts(
    trace: Trace, table: _Table, columns: dict[str, list[Any]], shadow_sets: dict[str, Any]
) -> None:
    """Tally module containment and write each atomic module call as ``address:pass``."""

    for op, calls in zip(table.ops, columns["modules"], strict=True):
        if calls:
            _log_module_hierarchy_info_for_layer(trace, op, shadow_sets)
    fid = table.fids[_ATOMIC_CALL]
    for row, call in zip(table.rows, columns[_ATOMIC_CALL], strict=True):
        if call is not None:
            row[fid] = ":".join([str(part) for part in call])


def _tally(trace: Trace, columns: dict[str, list[Any]]) -> None:
    """Add the ops' activation memory, call time and parameter-using layers to the trace.

    Output rows alias their producer's tensor and add no memory. A layer
    counts once, at its first op, when that op used parameters.
    """

    kinds = columns["type"]
    memory = [cell for cell, kind in zip(columns[_MEMORY], kinds, strict=True) if kind != "output"]
    if memory:
        trace._cost_total_activation_memory = _fold(
            trace._cost_total_activation_memory, memory, Bytes, int
        )
    trace._cost_func_calls_duration = _fold(
        trace._cost_func_calls_duration, columns[_DURATION], Duration, float
    )
    seen: dict[Any, None] = {}
    layers_with_params = 0
    for label, barcodes in zip(columns["layer_label"], columns["_param_barcodes"], strict=True):
        if barcodes and label not in seen:
            layers_with_params += 1
        seen[label] = None
    if layers_with_params:
        trace._num_layers_with_params += layers_with_params


def log_final_info_by_column(
    trace: Trace,
    shadow_sets: dict[str, Any],
    parent_ref_cls: type[Any],
    project_views: Callable[[Any], tuple[Any, Any, Any]],
) -> bool:
    """Run the per-op part of step 9 column by column.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    shadow_sets:
        The membership sets of the module build data lists.
    parent_ref_cls:
        The ``ParentRef`` class.
    project_views:
        Projects the then, elif and else views of a ``conditional_arm_children``.

    Returns
    -------
    bool
        ``False`` when the trace is not columnar; nothing was written and the
        caller runs the per-op loop.
    """

    table = _open_table(list(trace), _BY_ATTR, RENAMED_MEMBERS) if _WRITTEN_IN_PLACE else None
    if table is None:
        return False
    try:
        columns = {name: table.read(name, False, classes) for name, classes in _REQUIRED.items()}
        for name in (*RENAMED_ATTRS, *RENAMED_MEMBERS):
            columns[name] = table.read(name, True, _CLOSURE_CLASSES if name in _CLOSURES else None)
    except _NotColumnar:
        return False
    if trace._cost_total_activation_memory.__class__ not in (int, Bytes):
        return False
    if trace._cost_func_calls_duration.__class__ not in (int, float, Duration):
        return False
    _write_roles_and_steps(trace, table, columns["type"])
    rename_label_columns(trace, table, columns, parent_ref_cls, project_views)
    _log_module_facts(trace, table, columns, shadow_sets)
    _tally(trace, columns)
    return True


def ops_inside_modules(trace: Trace) -> list[Any]:
    """Settle the FX fields of ops outside every module; return the ops still to settle.

    An op that ran inside no module call has no FX path and call index 0.
    Those are written here by column; the caller computes the rest per op.

    Parameters
    ----------
    trace:
        Trace being postprocessed.

    Returns
    -------
    list[Any]
        The ops whose FX fields the caller must compute, in trace order:
        the ops inside a module call, or every op when the trace is not columnar.
    """

    ops = list(trace)
    table = _open_table(ops, _FX_COLUMNS, ())
    if table is None:
        return ops
    try:
        modules = table.read("modules")
    except _NotColumnar:
        return ops
    f_path, f_index = table.fids[_FX_PATH], table.fids[_FX_INDEX]
    for row, calls in zip(table.rows, modules, strict=True):
        if not calls:
            row[f_path] = None
            row[f_index] = 0
    return [op for op, calls in zip(ops, modules, strict=True) if calls]
