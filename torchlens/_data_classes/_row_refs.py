"""Child records held by row key in another record's cells.

A record that lists child records in one of its cells (a ``Buffer`` its
version ops, a ``Module`` its calls and its params) stores each child as the
child's row in the trace's shared store, never the child object, so
trace-owned storage has no strong path to the child. The key form is used
only when every child has a row in the trace's shared store; a cell with a
child that has none (a record built outside a captured run, or left detached
by a re-run of the build passes) keeps the objects, exactly as before.

Reading resolves a row through the owning trace: an op through the trace's
label dictionary (validated against the row, with a scan as fallback), a
module call or a param through the core's weak-valued facade cache (filled
from the trace's doors on a miss). A module cell also keeps a weak reference
to each child beside its row, read first by the record the keys were made
for, so that record still answers when another trace has taken it over (a
state dictionary restored into a second ``Trace`` without a copy). The public
attributes keep their names and values: ``Buffer.versions`` reads as a list of ops, ``Module.ops`` and
``Module.params`` as accessors.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .._errors import RecordBindingError
from .._trace_core.record_rows import CORE_KEY, ROW_KEY, RecordCellField, record_state_items

_GET = object.__getattribute__

#: Record kind -> (trace attribute holding that kind's door, attribute of the
#: door that maps to every record of the kind).
_DOORS: dict[str, tuple[str, str]] = {
    "module_call": ("_module_logs", "_pass_dict"),
    "param": ("_param_logs", "_dict"),
}


def _core_of(trace: Any) -> Any:
    """Return the trace's core, or ``None`` for no trace or a trace without one."""

    return None if trace is None else trace.__dict__.get("_trace_core")


def _live_trace(ref: Any, record_name: str) -> Any:
    """Return the trace behind ``ref``; refuse when it was never bound or is gone."""

    if ref is None:
        raise RecordBindingError(
            f"{record_name} not bound to a Trace",
            code="record_not_bound",
            remedy="keep the owning Trace alive and read records through it",
        )
    trace = ref()
    if trace is None:
        raise RecordBindingError(
            f"Trace has been garbage-collected; a {record_name} does not keep its Trace alive",
            code="trace_reference_collected",
            remedy="keep the Trace (t = tl.trace(...)) or fork (f = t.fork().do(...)) in a variable while reading its records",
        )
    return trace


def op_keys(ops: Iterable[Any], trace: Any) -> list[Any]:
    """Return the rows of ``ops`` in the trace's op store, or the ops when one has no row there."""

    ops = list(ops)
    core = _core_of(trace)
    if core is None or core.ops is None or not ops:
        return ops
    store = core.ops
    rows: list[Any] = []
    for op in ops:
        try:
            if _GET(op, "_core") is not store:
                return ops
            rows.append(_GET(op, "_row"))
        except AttributeError:
            return ops
    return rows


def ops_at(rows: list[int], trace_ref: Any, record_name: str) -> list[Any]:
    """Return the ops at ``rows`` of the owning trace's op store."""

    trace = _live_trace(trace_ref, record_name)
    core = _core_of(trace)
    if core is None or core.ops is None:
        _live_trace(None, record_name)
    store = core.ops
    label_fid = store.layout.fid_by_name["label"]
    by_label = trace._layer_dict_main_keys
    ops: list[Any] = []
    for row in rows:
        label = store.cell_get(row, label_fid)
        op = by_label.get(label) if label.__class__ is str else None
        if op is None or _GET(op, "_row") != row or _GET(op, "_core") is not store:
            op = _scan_for_op(trace, store, row, record_name)
        ops.append(op)
    return ops


def _scan_for_op(trace: Any, store: Any, row: int, record_name: str) -> Any:
    """Return the op at ``row`` by walking the trace's op list; refuse when none is."""

    for op in trace._layer_list:
        try:
            if _GET(op, "_row") == row and _GET(op, "_core") is store:
                return op
        except AttributeError:
            continue
    return _live_trace(None, f"{record_name} version op")


class RowKeys(dict[Any, int]):
    """A child cell in key form: lookup key -> the child's row in its kind store.

    ``build`` is the class of the accessor the cell held, called with
    ``key -> record`` to give a read the same accessor again. ``owner`` is a
    weak reference to the record whose cell this is and ``refs`` maps each
    key to a weak reference to the child it named: a fork's copy of the
    record shares this object but names its own trace's records, so only
    ``owner`` reads ``refs``.
    """

    __slots__ = ("build", "owner", "refs")

    build: Any
    owner: Any
    refs: dict[Any, Any]

    def kept(self, record: Any) -> dict[Any, Any] | None:
        """Return ``key -> child`` for ``owner`` while every child is alive, else ``None``."""

        if self.owner() is not record:
            return None
        children: dict[Any, Any] = {}
        for key, ref in self.refs.items():
            child = ref()
            if child is None or child.__dict__.get(ROW_KEY) != self[key]:
                return None
            children[key] = child
        return children


def child_keys(children: Any, kind: str, core: Any, owner: Any) -> RowKeys | None:
    """Return ``key -> row`` for ``owner``'s accessor of children, or ``None`` when one has no row.

    Each child is entered in the core's weak-valued facade cache, which is
    how a later read finds it.
    """

    items = getattr(children, "_dict", None)
    store = core.kind_rows.get(kind)
    if not items or store is None:
        return None
    keys = RowKeys()
    keys.build = type(children)
    keys.owner = weakref.ref(owner)
    for key, child in items.items():
        child_dict = getattr(child, "__dict__", None)
        if child_dict is None or child_dict.get(CORE_KEY) is not store:
            return None
        keys[key] = child_dict[ROW_KEY]
    facades = core._facades
    for child in items.values():
        facades[(kind, child.__dict__[ROW_KEY])] = child
    keys.refs = {key: weakref.ref(child) for key, child in items.items()}
    return keys


def children_at(keys: RowKeys, kind: str, trace_ref: Any, record_name: str) -> dict[Any, Any]:
    """Return ``key -> child record`` for a key-form cell of the owning trace."""

    trace = _live_trace(trace_ref, record_name)
    core = _core_of(trace)
    if core is None:
        _live_trace(None, record_name)
    facades = core._facades
    children: dict[Any, Any] = {}
    for key, row in keys.items():
        child = facades.get((kind, row))
        if child is None:
            child = _scan_for_child(trace, core, kind, row, record_name)
        children[key] = child
    return children


def _scan_for_child(trace: Any, core: Any, kind: str, row: int, record_name: str) -> Any:
    """Enter every ``kind`` record of the trace's door in the facade cache; return the one at ``row``."""

    door_name, mapping_name = _DOORS[kind]
    door = trace.__dict__.get(door_name)
    store = core.kind_rows.get(kind)
    facades = core._facades
    found = None
    for record in (getattr(door, mapping_name, None) or {}).values():
        record_dict = record.__dict__
        if record_dict.get(CORE_KEY) is store:
            facades[(kind, record_dict[ROW_KEY])] = record
            if record_dict[ROW_KEY] == row:
                found = record
    if found is None:
        _live_trace(None, f"{record_name} child record")
    return found


class VersionsField(RecordCellField):
    """The ``Buffer.versions`` cell: rows of the version ops, read as the ops."""

    __slots__ = ()

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the version ops (a new list when the cell holds rows)."""

        if record is None:
            return self
        held = RecordCellField.__get__(self, record, objtype)
        if not held or held[0].__class__ is not int:
            return held
        return ops_at(held, getattr(record, "_source_ref", None), "Buffer")

    def __set__(self, record: Any, value: Any) -> None:
        """Store the version ops by row when the owning trace's op store has them all."""

        ref = getattr(record, "_source_ref", None)
        RecordCellField.__set__(self, record, op_keys(value, ref() if ref is not None else None))


class ChildAccessorField(RecordCellField):
    """A ``Module`` cell of child records, read as an accessor.

    The cell holds the accessor itself until ``rekey_child_cells`` replaces
    it with the children's rows; from then on each read builds a new
    accessor over the records at those rows.
    """

    __slots__ = ("_kind",)

    def __init__(self, name: str, fid: int, kind: str) -> None:
        """Bind the cell to the record kind of its children."""

        super().__init__(name, fid)
        self._kind = kind

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return an accessor over the child records."""

        if record is None:
            return self
        held = RecordCellField.__get__(self, record, objtype)
        if held.__class__ is not RowKeys:
            return held
        children = held.kept(record)
        if children is None:
            ref = record.__dict__.get("_source_trace_ref")
            children = children_at(held, self._kind, ref, "Module")
        return held.build(children)


def install_child_fields(cls: type, fields: Mapping[str, str | None]) -> None:
    """Replace the plain cell descriptors of ``cls`` named in ``fields``.

    ``fields`` maps a stored field name to ``None`` (a list of version ops)
    or to the record kind of an accessor's children. Runs after
    ``install_record_facade``.
    """

    fid_by_name = cls._TL_LAYOUT.fid_by_name  # type: ignore[attr-defined]
    for name, spec in fields.items():
        if spec is None:
            setattr(cls, name, VersionsField(name, fid_by_name[name]))
        else:
            setattr(cls, name, ChildAccessorField(name, fid_by_name[name], spec))
    setattr(cls, "_TL_CHILD_FIELDS", tuple(fields))


def child_state_items(record: Any) -> Iterator[tuple[str, Any]]:
    """Yield the record's state pairs with each child cell as the records it names.

    State streams (pickle, deep copy, a saved file) carry the child records,
    as they did when the cells held them; row keys never leave the trace.
    """

    child_fields = type(record)._TL_CHILD_FIELDS
    for name, value in record_state_items(record):
        if name in child_fields:
            value = getattr(record, name)
        yield name, value


def rekey_child_cells(records: Iterable[Any], trace: Any) -> None:
    """Replace each record's child accessors with the children's rows.

    Runs once the children are rows of the trace's kind stores and before the
    records' own store is sealed; a cell already in key form is left alone.
    """

    core = _core_of(trace)
    if core is None:
        return
    for record in records:
        cls_dict = type(record).__dict__
        for name in cls_dict["_TL_CHILD_FIELDS"]:
            field = cls_dict[name]
            try:
                held = RecordCellField.__get__(field, record)
            except AttributeError:
                continue
            keys = (
                None if held.__class__ is RowKeys else child_keys(held, field._kind, core, record)
            )
            if keys is not None:
                RecordCellField.__set__(field, record, keys)
