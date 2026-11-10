"""Records named by row from trace-owned storage.

Two kinds of trace-owned object name records. A facet view names the record
it describes, a facet spec names its home and a gradient record names its
owner; each of those lives in a cell of the record's own row. A record that
lists child records in one of its cells (a ``Buffer`` its version ops, a
``Module`` its calls and its params) names each child. Naming a record by
holding it made the trace's storage keep the record alive, often through a
reference cycle, so both keep the record's row instead and look the record up
through the trace's doors on every read. The public attributes keep their
names and values.

``RecordSlot`` keeps one name as a ``RowKey`` (the record's kind and row, with
a weak reference to its trace) and answers with the record itself. A value
that is not a row-backed record of a convertible kind, or a record the
trace's doors do not currently hand out for that row, is kept as it was
given. Record classes are told apart by name, so this module imports no
record class.

A child cell stores each child as its row in the trace's shared store, used
only when every child has a row there; a cell with a child that has none (a
record built outside a captured run, or left detached by a re-run of the
build passes) keeps the objects, exactly as before. Reading resolves a row
through the owning trace: an op through the trace's label dictionary, a
module call or a param through the trace's door for that kind, by the key the
child has there, each validated against the row with a scan as fallback.
``Buffer.versions`` reads as a list of ops, ``Module.ops`` and
``Module.params`` as accessors. The first read of a key-form cell builds its
container and keeps it in the reading trace's cell table
(``TraceCore._views.cells``, see ``view_tables.py``), strong for the trace's
life, so later reads return the same object and an in-place write persists, as
when the cell held the container itself. The kept ``Buffer.versions`` container
holds the ops by row (``VERSIONS_LIST``): the trace must hold no ``Op``, because
an ``Op`` holds its trace.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from .._errors import RecordBindingError
from .op_store import DetachedOpStore
from .record_rows import CORE_KEY, ROW_KEY, RecordCellField, record_state_items

_GET = object.__getattribute__

#: Builds the kept ``Buffer.versions`` container from ``(trace, ops)``. Bound at import by
#: the op containers (a list that holds row keys), so this module imports no record container.
VERSIONS_LIST: list[Any] = []

#: Record class name -> kind. Other kinds are held as given until they convert.
_KINDS: dict[str, str] = {
    "Op": "op",
    "Param": "param",
    "Module": "module",
    "ModuleCall": "module_call",
}
#: Kind -> the field whose value keys that kind's door (a lookup hint only).
_HINT_FIELDS: dict[str, str] = {
    "op": "label",
    "param": "address",
    "module": "address",
    "module_call": "label",
}
#: Kind other than ``op`` -> (trace attribute holding the kind's door, attribute
#: of the door that maps each key to its record).
_DOORS: dict[str, tuple[str, str]] = {
    "param": ("_param_logs", "_dict"),
    "module": ("_module_logs", "_dict"),
    "module_call": ("_module_logs", "_pass_dict"),
}


def _store_and_row(record: Any, kind: str) -> tuple[Any, Any]:
    """Return the row store and row a record of ``kind`` is bound to."""

    if kind == "op":
        return record._core, record._row
    cells = record.__dict__
    return cells.get(CORE_KEY), cells.get(ROW_KEY)


def _door(trace: Any, kind: str) -> tuple[Any, Any, Any]:
    """Return the kind's store on ``trace`` with its keyed and full doors."""

    cells = trace.__dict__
    core = cells.get("_trace_core")
    if core is None:
        return None, None, ()
    if kind == "op":
        return core.ops, cells.get("_layer_dict_main_keys"), cells.get("_layer_list") or ()
    keyed = _door_of(trace, kind)
    return core.kind_rows.get(kind), keyed, () if keyed is None else keyed.values()


def _door_of(trace: Any, kind: str) -> Any:
    """Return the mapping of the trace's door for ``kind``, or ``None`` when it has none."""

    door_name, mapping_name = _DOORS[kind]
    return getattr(trace.__dict__.get(door_name), mapping_name, None)


class RowKey:
    """Kind and row of one record, with a weak reference to its trace.

    Parameters
    ----------
    trace_ref:
        Weak reference to the trace that owns the row.
    kind:
        Record kind: ``"op"``, ``"param"``, ``"module"`` or ``"module_call"``.
    row:
        Row of the record in that kind's store.
    hint:
        Value that keyed the record in its door when the key was made.
    """

    __slots__ = ("hint", "record_kind", "row", "trace_ref")

    def __init__(self, trace_ref: Any, kind: str, row: int, hint: Any) -> None:
        """Store the parts of the key."""

        self.trace_ref = trace_ref
        self.record_kind = kind
        self.row = row
        self.hint = hint

    def find(self) -> Any | None:
        """Return the record the trace hands out for this row, or ``None``."""

        trace = self.trace_ref()
        if trace is None:
            return None
        store, keyed, every = _door(trace, self.record_kind)
        if store is None:
            return None
        found = None if keyed is None else keyed.get(self.hint)
        if found is not None:
            bound_store, bound_row = _store_and_row(found, self.record_kind)
            if bound_store is store and bound_row == self.row:
                return found
        for candidate in every:
            bound_store, bound_row = _store_and_row(candidate, self.record_kind)
            if bound_store is store and bound_row == self.row:
                return candidate
        return None

    def record(self) -> Any:
        """Return the record, refusing when its trace no longer hands it out."""

        found = self.find()
        if found is None:
            raise RecordBindingError(
                f"The {self.record_kind} record at row {self.row} is no longer reachable: its Trace "
                "has been garbage-collected or no longer holds that row",
                code="trace_reference_collected",
                remedy="keep the Trace (t = tl.trace(...)) in a variable while reading its "
                "facets and gradient records",
            )
        return found

    def __reduce__(self) -> tuple[Any, tuple[Any]]:
        """Pickle and copy as the record itself, so saved state keeps its shape."""

        return _as_given, (self.find(),)


def _as_given(value: Any) -> Any:
    """Return ``value``: a row key restores as the record it named."""

    return value


def row_key_or_value(value: Any, trace_ref: Any = None) -> Any:
    """Return a ``RowKey`` for a record its trace hands out, else ``value``.

    Parameters
    ----------
    value:
        Record to name, or any other value.
    trace_ref:
        Weak reference to the trace to look the record up in. Defaults to
        the record's own trace reference.

    Returns
    -------
    Any
        The row key, or ``value`` when it cannot be named by row.
    """

    kind = _KINDS.get(type(value).__name__)
    if kind is None:
        return value
    try:
        if trace_ref is None:
            trace_ref = value._source_trace_ref
        store, row = _store_and_row(value, kind)
        hint = getattr(value, _HINT_FIELDS[kind], None)
    except AttributeError:
        return value
    if trace_ref is None or store is None:
        return value
    trace = trace_ref()
    if trace is None:
        return value
    # Verify through the keyed door inline (the common case): a full ``find`` per
    # facet spec raised trace plus ``to_pandas`` calls past the C7 tolerance.
    door_store, keyed, _ = _door(trace, kind)
    if door_store is store and keyed is not None and keyed.get(hint) is value:
        return RowKey(trace_ref, kind, row, hint)
    key = RowKey(trace_ref, kind, row, hint)
    return key if key.find() is value else value


class RecordSlot:
    """Attribute that stores a record as a row key and reads back the record.

    Parameters
    ----------
    name:
        Attribute name; the stored form lives in the instance dictionary
        under the same name.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        """Bind the slot to its attribute name."""

        self._name = name

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """Return the named record, or the value that was stored as given."""

        if instance is None:
            return self
        try:
            stored = instance.__dict__[self._name]
        except KeyError:
            raise AttributeError(self._name) from None
        return stored.record() if stored.__class__ is RowKey else stored

    def __set__(self, instance: Any, value: Any) -> None:
        """Store ``value``; the slot itself, a dataclass default, means ``None``."""

        instance.__dict__[self._name] = None if value is self else row_key_or_value(value)


class OwnerByRow:
    """Base for objects whose ``owner`` record is kept as a row key."""

    owner = RecordSlot("owner")


def name_owner_by_row(trace_ref: Any, owner: Any) -> None:
    """Point every gradient record of ``owner`` at it by row in ``trace_ref``'s trace.

    Parameters
    ----------
    trace_ref:
        Weak reference to the trace that hands ``owner`` out.
    owner:
        Record whose ``_grad_records`` are rebound.
    """

    grad_records = getattr(owner, "_grad_records", None)
    if not grad_records:
        return  # nothing to rebind, so the owner's row is not looked up
    stored = row_key_or_value(owner, trace_ref)
    for grad_record in grad_records:
        grad_record.__dict__["owner"] = stored


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
    views = getattr(core, "_views", None)
    if views is None:
        return build()
    table = views.cells
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
        return _built_once(
            trace,
            record,
            self._name,
            held,
            lambda: VERSIONS_LIST[0](trace, ops_at(held, ref, "Buffer")),
        )

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
