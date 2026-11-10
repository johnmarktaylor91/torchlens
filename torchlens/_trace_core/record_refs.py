"""Row keys for records named from trace-owned storage.

A facet view names the record it describes, a facet spec names its home and
a gradient record names its owner. Each of those objects lives in a cell of
the record's own row, so naming the record by holding it made the trace's
storage keep the record alive through a reference cycle. ``RecordSlot``
keeps the name as a ``RowKey`` (the record's kind and row, with a weak
reference to its trace) and looks the record up on every read, so the
attribute still answers with the record itself.

A value that is not a row-backed record of a convertible kind, or a record
the trace's doors do not currently hand out for that row, is kept as it was
given. Record classes are told apart by name, so this module imports no
record class.
"""

from __future__ import annotations

from typing import Any

from .._errors import RecordBindingError

#: Record class name -> kind. Other kinds are held as given until they convert.
_KINDS: dict[str, str] = {"Op": "op", "Param": "param"}
#: Kind -> the field whose value keys that kind's door (a lookup hint only).
_HINT_FIELDS: dict[str, str] = {"op": "label", "param": "address"}


def _store_and_row(record: Any, kind: str) -> tuple[Any, Any]:
    """Return the row store and row a record of ``kind`` is bound to."""

    if kind == "op":
        return record._core, record._row
    cells = record.__dict__
    return cells.get("_tl_core"), cells.get("_tl_row")


def _door(trace: Any, kind: str) -> tuple[Any, Any, Any]:
    """Return the kind's store on ``trace`` with its keyed and full doors."""

    cells = trace.__dict__
    core = cells.get("_trace_core")
    if core is None:
        return None, None, ()
    if kind == "op":
        return core.ops, cells.get("_layer_dict_main_keys"), cells.get("_layer_list") or ()
    params = cells.get("_param_logs")
    return core.kind_rows.get(kind), getattr(params, "_dict", None), _members(params)


def _members(door: Any) -> Any:
    """Return an iterable over a door's records; an absent door has none."""

    return () if door is None else door


class RowKey:
    """Kind and row of one record, with a weak reference to its trace.

    Parameters
    ----------
    trace_ref:
        Weak reference to the trace that owns the row.
    kind:
        Record kind, ``"op"`` or ``"param"``.
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

    stored = row_key_or_value(owner, trace_ref)
    for grad_record in getattr(owner, "_grad_records", None) or ():
        grad_record.__dict__["owner"] = stored
