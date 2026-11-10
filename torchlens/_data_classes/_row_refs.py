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
module call or a param through the trace's door for that kind, by the key the
child has there (validated against the row, with a scan as fallback). The
public attributes keep their names and values: ``Buffer.versions`` reads as a
list of ops, ``Module.ops`` and ``Module.params`` as accessors.

The first read of a key-form cell builds its container and keeps it in the
reading trace's side table (``TraceCore._strong_facades[CHILD_CELLS]``),
strong for the trace's life, so later reads return the same object and an
in-place write persists, as when the cell held the container itself.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .._errors import RecordBindingError
from .._trace_core.op_store import DetachedOpStore
from .._trace_core.record_rows import CORE_KEY, ROW_KEY, RecordCellField, record_state_items

_GET = object.__getattribute__

#: Side-table kind holding the containers built from key-form cells.
CHILD_CELLS = "child_cells"

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


def _built_once(trace: Any, record: Any, name: str, held: Any, build: Any) -> Any:
    """Return the container built from ``held`` for ``record.name``, building it once per trace.

    The container is kept in the trace's side table next to the store and the
    key-form value it was built from; a cell given a new value builds again.
    """

    core = _core_of(trace)
    facades = getattr(core, "_strong_facades", None)
    if not isinstance(facades, dict):
        return build()
    table = facades.get(CHILD_CELLS)
    if table is None:
        table = facades[CHILD_CELLS] = {}
    record_dict = record.__dict__
    store = record_dict.get(CORE_KEY)
    key = (id(store), record_dict.get(ROW_KEY), name)
    entry = table.get(key)
    if entry is not None and entry[0] is store and entry[1] is held:
        return entry[2]
    built = build()
    table[key] = (store, held, built)
    return built


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
    ``key -> record`` to give a read the same accessor again. ``door`` holds,
    in the order of the keys, the key each child has in its trace's door, and
    ``home`` is a weak reference to the trace the rows were read from. A
    fork's copy of the record shares this object and reads through its own
    trace, whose door hands out the fork's records under the same keys.
    """

    __slots__ = ("build", "door", "home")

    build: Any
    door: tuple[Any, ...]
    home: Any


def _door_of(trace: Any, kind: str) -> Any:
    """Return the mapping of the trace's door for ``kind``, or ``None`` when it has none."""

    door_name, mapping_name = _DOORS[kind]
    return getattr(trace.__dict__.get(door_name), mapping_name, None)


def child_keys(children: Any, kind: str, trace: Any, door_keys: Any) -> RowKeys | None:
    """Return ``key -> row`` for an accessor's children, or ``None`` when one cannot be keyed.

    ``door_keys`` maps the identity of each record the trace's door hands out
    to its key there. A child with no row in the trace's store for ``kind``,
    or one the door does not hand out, leaves the accessor in the cell.
    """

    items = getattr(children, "_dict", None)
    store = _core_of(trace).kind_rows.get(kind)
    if not items or store is None or door_keys is None:
        return None
    keys = RowKeys()
    door: list[Any] = []
    for key, child in items.items():
        child_dict = getattr(child, "__dict__", None)
        if (
            child_dict is None
            or child_dict.get(CORE_KEY) is not store
            or id(child) not in door_keys
        ):
            return None
        keys[key] = child_dict[ROW_KEY]
        door.append(door_keys[id(child)])
    keys.build = type(children)
    keys.door = tuple(door)
    keys.home = weakref.ref(trace)
    return keys


def reading_trace(keys: RowKeys, record: Any, record_name: str) -> Any:
    """Return the trace a key-form cell of ``record`` reads through.

    The record the keys were made for reads through the trace they were made
    in while that trace lives, so it still answers after another trace took
    the record over (a state dictionary restored into a second ``Trace``
    without a copy). A detached copy (a fork's) reads through its own trace.
    """

    record_dict = record.__dict__
    trace = None if record_dict[CORE_KEY].__class__ is DetachedOpStore else keys.home()
    if trace is None:
        trace = _live_trace(record_dict.get("_source_trace_ref"), record_name)
    return trace


def children_at(keys: RowKeys, kind: str, record: Any, record_name: str) -> dict[Any, Any]:
    """Return ``key -> child record`` for a key-form cell of ``record``."""

    return _children_in(reading_trace(keys, record, record_name), keys, kind, record_name)


def _children_in(trace: Any, keys: RowKeys, kind: str, record_name: str) -> dict[Any, Any]:
    """Return ``key -> child record`` for a key-form cell read through ``trace``."""

    mapping = _door_of(trace, kind)
    if mapping is None:
        _live_trace(None, record_name)
    children: dict[Any, Any] = {}
    for (key, row), door_key in zip(keys.items(), keys.door):
        try:
            child = mapping[door_key]
            if child.__dict__[ROW_KEY] != row:
                return _scan_for_children(keys, mapping, record_name)
        except KeyError:
            return _scan_for_children(keys, mapping, record_name)
        children[key] = child
    return children


def _scan_for_children(keys: RowKeys, mapping: Any, record_name: str) -> dict[Any, Any]:
    """Return ``key -> child`` by row when a door key no longer names the child; refuse a missing row."""

    by_row = {record.__dict__.get(ROW_KEY): record for record in mapping.values()}
    children: dict[Any, Any] = {}
    for key, row in keys.items():
        child = by_row.get(row)
        if child is None:
            _live_trace(None, f"{record_name} child record")
        children[key] = child
    return children


class VersionsField(RecordCellField):
    """The ``Buffer.versions`` cell: rows of the version ops, read as the ops."""

    __slots__ = ()

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the version ops (one list per trace when the cell holds rows)."""

        if record is None:
            return self
        held = RecordCellField.__get__(self, record, objtype)
        if not held or held[0].__class__ is not int:
            return held
        ref = getattr(record, "_source_ref", None)
        trace = _live_trace(ref, "Buffer")
        return _built_once(trace, record, self._name, held, lambda: ops_at(held, ref, "Buffer"))

    def __set__(self, record: Any, value: Any) -> None:
        """Store the version ops by row when the owning trace's op store has them all."""

        ref = getattr(record, "_source_ref", None)
        RecordCellField.__set__(self, record, op_keys(value, ref() if ref is not None else None))


class ChildAccessorField(RecordCellField):
    """A ``Module`` cell of child records, read as an accessor.

    The cell holds the accessor itself until ``rekey_child_cells`` replaces
    it with the children's rows; from then on the first read in a trace
    builds an accessor over the records at those rows and later reads in
    that trace return it.
    """

    __slots__ = ("_kind",)

    def __init__(self, name: str, fid: int, kind: str) -> None:
        """Bind the cell to the record kind of its children."""

        super().__init__(name, fid)
        self._kind = kind

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return an accessor over the child records (one accessor per trace)."""

        if record is None:
            return self
        held = RecordCellField.__get__(self, record, objtype)
        if held.__class__ is not RowKeys:
            return held
        trace = reading_trace(held, record, "Module")
        kind = self._kind
        return _built_once(
            trace,
            record,
            self._name,
            held,
            lambda: held.build(_children_in(trace, held, kind, "Module")),
        )


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

    Runs once the children are rows of the trace's kind stores and its doors
    hand them out, and before the records' own store is sealed; a cell
    already in key form is left alone.
    """

    if _core_of(trace) is None:
        return
    door_keys: dict[str, Any] = {}
    for record in records:
        cls_dict = type(record).__dict__
        for name in cls_dict["_TL_CHILD_FIELDS"]:
            field = cls_dict[name]
            try:
                held = RecordCellField.__get__(field, record)
            except AttributeError:
                continue
            if held.__class__ is RowKeys:
                continue
            kind = field._kind
            if kind not in door_keys:
                mapping = _door_of(trace, kind)
                door_keys[kind] = (
                    None if mapping is None else {id(child): key for key, child in mapping.items()}
                )
            keys = child_keys(held, kind, trace, door_keys[kind])
            if keys is not None:
                RecordCellField.__set__(field, record, keys)
