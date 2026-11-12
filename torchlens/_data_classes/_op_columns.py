"""Private row-indexed column view over the ``Op`` row store.

Postprocess passes read and write ``Op`` fields millions of times per trace.
Through the attribute facade every read costs three to five Python-level calls
(``Op.__getattribute__``, the generated field descriptor or an overlay property,
``OpRowStore.cell_get``). :class:`OpColumns` hands a pass bound per-field
accessors indexed by row instead: one Python call per read or write.

Semantics are the facade's EXACTLY, by construction rather than by restatement:

- The fast path serves a cell only while the store is a plain BUILDING
  ``OpRowStore`` (the postprocess phase) and only when the raw cell is a value
  the facade would return unchanged.
- Every other cell goes to the facade's own class attribute (the generated
  descriptor or the overlay property) on a two-slot ``Op`` shell bound to the
  same ``(store, row)``: ``PooledCell`` hydrate-on-read with its cache-back,
  ``GroupRef`` views, CSR-backed dataflow tuples, shared-fact hydration, the
  ancestor-bitset frozen views, and the unset-cell ``AttributeError``.
- Other store shapes (sealed, columnar, fork views, the step-audit twins) read
  through ``store.cell_get`` so overlays, fork isolation and the audit read
  record behave as for a facade read; writes go through the facade attribute.

Fields the facade intercepts at the instance level (``Op.__getattribute__``
read hooks, ``Op.__setattr__`` dirty-marking writes) are refused at bind time:
those stay facade-only. The view is private to the postprocess passes; the
``Op`` facade remains the user read path.

The ``Op`` class and its descriptor classes are resolved at bind time from the
facade binding ``op.py`` registers in ``_op_row_release`` (the row-release fast
path's precedent), never imported here: this module and the passes using it
stay outside the record import cycle.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .._trace_core.cell_states import LazyCell
from .._trace_core.groups import GroupRef
from .._trace_core.op_store import (
    _CSR,
    _FACT,
    _MISSING,
    OpRowStore,
    PooledCell,
    _SealedRowMajorOpRowStore,
)
from .._trace_core.relation_views import OP_BITSET_VIEW_FIELDS
from .._utils._record_group_table import op_attr
from ._op_row_release import _OP_FACADE

__all__ = ("OpColumns", "column_rows_by_label")

#: Marker for "no default": an unset cell raises like the facade.
_NO_DEFAULT: Any = object()

#: Generated descriptor class name -> what its ``__get__`` transforms.
_DESCRIPTOR_KINDS = {
    "_OpField": "plain",
    "_RelationViewField": "relation",
    "_GroupViewField": "group",
    "_DataflowField": "dataflow",
    "_FactField": "fact",
}

_object_setattr = object.__setattr__


def _bound_facade() -> tuple[type, Any, frozenset[str], Any]:
    """Return ``(Op class, its class dict, stored names, its module)``.

    Raises
    ------
    RuntimeError
        When ``op.py`` has not bound the facade yet.
    """

    if not _OP_FACADE:
        raise RuntimeError("the Op facade is not bound; import torchlens first")
    op_class, op_vars, _slot_names, _field_classes, slot_set = _OP_FACADE[0]
    # The facade's module is imported by the time it binds; the import
    # machinery is only the fallback, not a per-read cost.
    module = sys.modules.get(op_class.__module__) or importlib.import_module(op_class.__module__)
    return op_class, op_vars, slot_set, module


def _slow_cell_types(name: str, attribute: Any) -> tuple[tuple[type, ...], Any]:
    """Return the cell classes and sentinel the facade transforms for ``name``.

    A raw cell whose class is in the returned tuple, or that IS the returned
    sentinel, must be read through the facade attribute; any other present
    value is what the facade returns unchanged.

    Parameters
    ----------
    name:
        Stored field name.
    attribute:
        The class attribute ``Op`` serves the field through.

    Returns
    -------
    tuple[tuple[type, ...], Any]
        ``(slow classes, slow sentinel)``.

    Raises
    ------
    ValueError
        When the attribute is neither a generated field descriptor nor the
        declared ancestor-bitset overlay property.
    """

    field_classes = _OP_FACADE[0][3]
    kind = _DESCRIPTOR_KINDS.get(type(attribute).__name__)
    if kind is not None and type(attribute) in field_classes:
        if kind == "group":
            return (GroupRef,), _MISSING
        if kind == "dataflow":
            return (), _CSR
        if kind == "fact":
            return (), _FACT
        return (PooledCell,), _MISSING
    slow = getattr(type(attribute), "COLUMN_SLOW_CELLS", None)
    if slow is not None:  # a descriptor that names the raw cells it transforms
        return slow, _MISSING
    if type(attribute) is property and name in OP_BITSET_VIEW_FIELDS:
        # The overlay reads through the original relation descriptor (PooledCell
        # hydration) and materializes the registered finished-cell encodings.
        encodings = _bound_facade()[3]._RELATION_CELL_ENCODINGS
        return (PooledCell, *encodings), _MISSING
    raise ValueError(
        f"Op field {name!r} is served by {type(attribute).__name__}, which the column "
        "view does not model; read it through the Op facade"
    )


def _is_relation_write(name: str, attribute: Any) -> bool:
    """Return whether writes to ``name`` normalize to an immutable view once finished."""

    kind = _DESCRIPTOR_KINDS.get(type(attribute).__name__)
    return kind in {"relation", "group", "dataflow"} or name in OP_BITSET_VIEW_FIELDS


def _stored_attribute(name: str) -> Any:
    """Return the ``Op`` class attribute serving stored field ``name`` (see ``OpColumns``)."""

    _op_class, op_vars, stored_names, _module = _bound_facade()
    attr = op_attr(name)
    if attr not in stored_names:
        raise ValueError(f"{name!r} is not a stored Op field")
    return op_vars[attr]


#: Stored field name -> ``(facade binding, attribute name, class attribute, slow
#: classes, sentinel)`` for a field served by a generated descriptor, whose slow cells
#: its class fixes (``_slow_cell_types``). An entry is used only while the bound
#: facade and the class attribute are the very objects it was built from; any other
#: attribute (a property, a ``COLUMN_SLOW_CELLS`` descriptor) is planned on every call.
_READ_PLANS: dict[str, tuple[Any, str, Any, tuple[type, ...], Any]] = {}


def _planned_attribute(name: str) -> tuple[Any, tuple[Any, ...] | None]:
    """Return ``(class attribute, live plan entry or None)`` for stored field ``name``."""

    entry = _READ_PLANS.get(name)
    if entry is not None and _OP_FACADE:
        binding = entry[0]
        if binding is _OP_FACADE[0] and binding[1].get(entry[1]) is entry[2]:
            return entry[2], entry
    return _stored_attribute(name), None


def _read_plan(name: str) -> tuple[Any, tuple[type, ...], Any]:
    """Return ``(class attribute, slow classes, sentinel)`` for stored field ``name``."""

    attribute, entry = _planned_attribute(name)
    if entry is not None:
        return attribute, entry[3], entry[4]
    slow_types, sentinel = _slow_cell_types(name, attribute)
    binding = _OP_FACADE[0]
    attribute_type = type(attribute)
    if attribute_type in binding[3] and attribute_type.__name__ in _DESCRIPTOR_KINDS:
        _READ_PLANS[name] = (binding, op_attr(name), attribute, slow_types, sentinel)
    return attribute, slow_types, sentinel


class OpColumns:
    """Bound per-field accessors over one ``Op`` row store.

    Parameters
    ----------
    store:
        The op row store (any store shape an ``Op`` facade can be bound to).
    """

    __slots__ = ("store",)

    def __init__(self, store: Any) -> None:
        """Bind the view to ``store``."""

        self.store = store

    def _attribute(self, name: str) -> Any:
        """Return the ``Op`` class attribute serving stored field ``name``.

        ``name`` is the persisted field name (the store layout key); a field in
        a concern group is served by its private storage attribute
        (``has_output_descendant`` by ``_graph_has_output_descendant``).
        """

        return _stored_attribute(name)

    def _shell(self, row: int) -> Any:
        """Return an ``Op`` facade over ``(self.store, row)`` for exact delegation."""

        op_class: Any = _OP_FACADE[0][0]
        shell = op_class.__new__(op_class)
        _object_setattr(shell, "_core", self.store)
        _object_setattr(shell, "_row", row)
        return shell

    def getter(self, name: str, default: Any = _NO_DEFAULT) -> Callable[[int], Any]:
        """Return a row-indexed reader for stored field ``name``.

        Parameters
        ----------
        name:
            Stored ``Op`` field name.
        default:
            Value returned for an unset cell, as ``getattr(op, name, default)``.
            Without it an unset cell raises the facade's ``AttributeError``.

        Returns
        -------
        Callable[[int], Any]
            ``get(row)`` returning what ``getattr(op, name)`` returns.

        Raises
        ------
        ValueError
            For an unknown name or a field whose reads the facade intercepts.
        """

        op_class, _op_vars, _names, op_module = _bound_facade()
        if name in op_module._INTERCEPTED_READ_FIELDS:
            raise ValueError(f"Op.{name} reads are intercepted by the facade; read the facade")
        attribute, slow_types, sentinel = _read_plan(name)
        store = self.store
        fid = store.layout.fid_by_name[name]
        facade_get = attribute.__get__
        shell = self._shell

        def get(row: int) -> Any:
            """Return field ``name`` of ``row`` exactly as the facade reads it."""

            if store.__class__ is OpRowStore and store._rows is not None:
                value = store._rows[row][fid]
                if value.__class__ is LazyCell:
                    value = store.cell_get(row, fid) if value.make else value.source
            else:
                value = store.cell_get(row, fid)
            if value is not _MISSING and value is not sentinel:
                if not isinstance(value, slow_types):
                    return value
            elif value is _MISSING and default is not _NO_DEFAULT:
                return default
            return facade_get(shell(row), op_class)

        return get

    def column(self, name: str, rows: Sequence[int]) -> list[Any]:
        """Return field ``name`` for every row in ``rows``, in one pass.

        The bulk readers' column read (``to_pandas``): each value is what
        ``getter(name)(row)`` returns, but a plain or sealed row store and a
        frozen columnar store are read straight from their backing, so a
        cell the facade returns unchanged costs no call. Every other cell
        (an unanswered lazy cell, an encoding the facade transforms, an
        unset cell) and every other store shape goes through ``getter``.

        Parameters
        ----------
        name:
            Stored ``Op`` field name.
        rows:
            Row indices, in the order the values are returned.

        Returns
        -------
        list[Any]
            One value per row.

        Raises
        ------
        ValueError
            As :meth:`getter`.
        """

        get = self.getter(name)
        raw = _raw_column(self.store, self.store.layout.fid_by_name[name], rows)
        if raw is None:
            return [get(row) for row in rows]
        _attribute, slow_types, sentinel = _read_plan(name)
        # The classes present, then the slow ones among them: one subclass test per
        # class, not per cell (the facade's ``isinstance`` on each cell).
        classes = {value.__class__ for value in raw}
        slow = {cls for cls in classes if cls is LazyCell or issubclass(cls, slow_types)}
        return [
            get(row) if value.__class__ in slow or value is _MISSING or value is sentinel else value
            for row, value in zip(rows, raw, strict=True)
        ]

    def setter(self, name: str) -> Callable[[int, Any], None]:
        """Return a row-indexed writer for stored field ``name``.

        Parameters
        ----------
        name:
            Stored ``Op`` field name.

        Returns
        -------
        Callable[[int, Any], None]
            ``put(row, value)`` with the effect of ``op._internal_set(name, value)``.

        Raises
        ------
        ValueError
            For an unknown name or a field whose writes the facade guards.
        """

        op_module = _bound_facade()[3]
        if name in op_module._DIRECT_WRITE_GUARDED_FIELDS or (
            name in op_module._INTERCEPTED_READ_FIELDS
        ):
            raise ValueError(f"Op.{name} writes are guarded by the facade; write the facade")
        # A field with no read plan takes the direct lookup: the plan probe would add
        # two calls to every unplanned write (``mark_tracing_finished``'s flag).
        if name in _READ_PLANS:
            attribute = _planned_attribute(name)[0]
        else:
            attribute = _stored_attribute(name)
        relation = _is_relation_write(name, attribute)
        store = self.store
        fid = store.layout.fid_by_name[name]
        facade_set = attribute.__set__
        shell = self._shell

        def put(row: int, value: Any) -> None:
            """Write field ``name`` of ``row`` exactly as the facade writes it."""

            if (
                store.__class__ is OpRowStore
                and store._rows is not None
                and not (relation and store.dataflow_edges is not None)
            ):
                store._rows[row][fid] = value
                return
            facade_set(shell(row), value)

        return put


def _raw_column(store: Any, fid: int, rows: Sequence[int]) -> list[Any] | None:
    """Return the raw cells ``store.cell_get(row, fid, raw=True)`` holds, or ``None``.

    ``None`` for a store shape this does not read directly (fork views, audit
    twins, detached stores); the caller then reads cell by cell. A packed
    numeric column converts in one ``tolist`` (the same Python scalars its
    per-cell ``item`` returns); an absent cell reads ``_MISSING`` and an
    overlay cell wins over the frozen backing, as in ``cell_get``.
    """

    if store.__class__ is not OpRowStore and store.__class__ is not _SealedRowMajorOpRowStore:
        return None
    cells = store._rows
    if cells is not None:
        return [cells[row][fid] for row in rows]
    columns = store._columns
    if columns is None:  # neither row cells nor columns: read row by row
        return None
    column = columns[fid]
    backing = column.values.tolist() if column.packed else column.values
    present = column.present
    raw = [backing[row] if present is None or present[row] else _MISSING for row in rows]
    overlay = store._overlay
    if overlay:
        n_fields = store.layout.n_fields
        hits = overlay.keys() & {row * n_fields + fid for row in rows}
        if hits:
            position = {row: index for index, row in enumerate(rows)}
            for key in hits:
                raw[position[key // n_fields]] = overlay[key]
    return raw


def column_rows_by_label(
    ops_by_label: Mapping[str, Any],
) -> tuple[OpColumns, dict[str, int]] | None:
    """Return a column view and ``label -> row`` for ops sharing one store.

    Parameters
    ----------
    ops_by_label:
        Raw label to ``Op`` mapping (the raw graph workspace's layer dict).

    Returns
    -------
    tuple[OpColumns, dict[str, int]] | None
        ``None`` when the mapping is empty, holds a non-``Op`` record, or its
        ops live on more than one store; the caller then keeps the facade path.
    """

    op_class, op_vars, _names, _module = _bound_facade()
    core_get = op_vars["_core"].__get__
    row_get = op_vars["_row"].__get__
    store = None
    rows: dict[str, int] = {}
    for label, op in ops_by_label.items():
        if type(op) is not op_class:  # ``op.__class__`` runs Op.__getattribute__
            return None
        op_store = core_get(op)
        if store is None:
            store = op_store
        elif op_store is not store:
            return None
        rows[label] = row_get(op)
    if store is None:
        return None
    return OpColumns(store), rows
