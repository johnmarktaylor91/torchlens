"""Trace-held ``Module`` and ``ModuleCall`` state, with record views built on demand.

A ``Module`` or ``ModuleCall`` object is a view: its ``__dict__`` is one state
dictionary, and every stored field is a cell of a row that state names. The
containers a trace keeps its modules and calls in (the ``_dict``, ``_list``
and index fields of the module accessors) therefore hold the *state* of each
record and build the record object when one is read. Two reads that overlap
return the same object, through any container, because the live view of each
state is remembered weakly in one table; once nobody holds a view, the next
read builds a fresh one over the same state, so attributes set on a record
are kept.

A view handed out holds its trace in its ``_owner`` slot, which is what makes
``tl.trace(model, x).modules["fc"].out`` work: the record keeps the trace, the
trace keeps the state. The reference runs one way only, so a trace nobody
holds is still freed by reference counting. A record copied or restored on
its own has no trace to hold and reads as it always did.

The accessor objects follow the same rule one level up. The accessor a trace
stores cannot hold the trace, so a door (``Trace.modules``,
``Trace.module_calls``, ``Module.calls``) hands out a second accessor object
over the same state that does, cached weakly on the stored one.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict, deque
from collections.abc import Iterable, Iterator
from itertools import repeat
from operator import is_ as _is
from typing import Any, TypeVar

from .._trace_core.record_refs import ChildAccessorField, RowKeys
from .._trace_core.record_rows import RecordCellField
from .._trace_core.view_tables import DEAD, DEREF

_ref = weakref.ref
_DEREF: Any = DEREF
_new = object.__new__
_set = object.__setattr__
_list_item = list.__getitem__
_TRACE_REF = "_source_trace_ref"
_T = TypeVar("_T")

#: The live view of each state, by the identity of the state dictionary. A
#: view holds its state, so an entry whose view is alive names a state that is
#: alive too and the identity is its own; an entry whose view is gone is
#: never trusted (every read checks ``view.__dict__ is state``) and is swept.
_VIEWS: dict[int, Any] = {}

#: The table size at which dead entries are swept next.
_SWEEP_AT = [4096]


def _sweep() -> None:
    """Drop the table entries whose view is gone."""

    for key, reference in list(_VIEWS.items()):
        if reference() is None and _VIEWS.get(key) is reference:
            del _VIEWS[key]
    _SWEEP_AT[0] = max(4096, 2 * len(_VIEWS))


def _owners(states: Iterable[dict[str, Any]]) -> list[Any]:
    """Return the live trace each state names, or ``None``."""

    references = map(dict.get, states, repeat(_TRACE_REF))
    return [None if reference is None else reference() for reference in references]


def adopt(records: list[Any]) -> list[dict[str, Any]]:
    """Make ``records`` the views of their states and return the states.

    Each record becomes the object later reads return while it is alive, and
    is given the trace its state names to hold.
    """

    states: list[Any] = list(map(vars, records))
    deque(map(_set, records, repeat("_owner"), _owners(states)), 0)
    _VIEWS.update(zip(map(id, states), map(_ref, records)))
    if len(_VIEWS) > _SWEEP_AT[0]:
        _sweep()
    return states


def _build(states: list[dict[str, Any]], cls: type) -> list[Any]:
    """Build a view of ``cls`` over each state and remember it; the loop runs in C."""

    views: list[Any] = list(map(_new, repeat(cls, len(states))))
    deque(map(_set, views, repeat("__dict__"), states), 0)
    deque(map(_set, views, repeat("_owner"), _owners(states)), 0)
    _VIEWS.update(zip(map(id, states), map(_ref, views)))
    if len(_VIEWS) > _SWEEP_AT[0]:
        _sweep()
    return views


def view_of(state: dict[str, Any], cls: type) -> Any:
    """Return the record of ``cls`` over ``state``: the live view, or a new one."""

    try:
        view = _VIEWS[id(state)]()
        if view.__dict__ is state:
            return view
    except (KeyError, AttributeError):
        pass  # no view of this state is alive: build one
    return _build([state], cls)[0]


def views_of(states: list[Any], cls: type) -> list[Any]:
    """Return the record over each of ``states`` in order.

    The loop runs in C, so reading every record of a container costs a fixed
    number of calls. An item that is not a state (see :class:`Kept`) is
    returned as the value it keeps.
    """

    found = list(map(_DEREF, map(_VIEWS.get, map(id, states), repeat(DEAD))))
    # Every item has a live view of its own state: the comparison runs in C.
    if False not in map(_is, map(getattr, found, repeat("__dict__"), repeat(None)), states):
        return found
    stale = [
        at
        for at, (view, state) in enumerate(zip(found, states, strict=True))
        if view is None or view.__dict__ is not state
    ]
    if stale:
        kept = [at for at in stale if states[at].__class__ is Kept]
        if kept:
            for at in kept:
                found[at] = states[at].value
            stale = [at for at in stale if states[at].__class__ is not Kept]
        missing = [states[at] for at in stale]
        built = _build(missing, cls)
        # Read back through the table, so a state named twice gets one view; a
        # table entry that names another state (an identity reported twice) is
        # not trusted, and the view just built is returned.
        fresh = list(map(_DEREF, map(_VIEWS.__getitem__, map(id, missing))))
        for at, view, own in zip(stale, fresh, built, strict=True):
            found[at] = view if view.__dict__ is own.__dict__ else own
    return found


class Kept:
    """A container item held as given: a value that is not a record of the container's class."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        """Keep ``value``."""

        self.value = value


def _held(value: Any, cls: type) -> Any:
    """Return what a container stores for ``value``: its state, or the value kept as given."""

    if value.__class__ is cls:
        return adopt([value])[0]
    return Kept(value)


def _all_held(values: list[Any], cls: type) -> list[Any]:
    """Return what a container stores for each of ``values``."""

    if set(map(type, values)) <= {cls}:
        return adopt(values)
    return [_held(value, cls) for value in values]


class RecordDict(dict[Any, Any]):
    """Mapping from key to record, holding each record's state.

    The mapping's own entries are ``key -> state dictionary``; every read
    hands out the record view of that state. ``kind`` is the container type
    it stands in for (``dict`` or ``OrderedDict``): its plain form, which is
    what a copy, a pickle, a comparison and a repr use, has that type.

    A read by key is the hot path, so it makes no call: ``raw`` is a plain
    ``dict`` with the same entries (a subscript of this mapping would come
    back here), ``vals`` its values, and ``live`` remembers weakly the view
    last handed out for each key. A ``live`` entry is used only while its view
    is alive and is the view of the state now at that key.
    """

    __slots__ = ("cls", "kind", "live", "raw", "vals")

    cls: type
    kind: type
    live: dict[Any, Any]
    raw: dict[Any, Any]
    vals: Any

    def __getitem__(self, key: Any) -> Any:
        """Return the record at ``key``."""

        state = self.raw[key]
        try:
            view = self.live[key]()
            if view.__dict__ is state:
                return view
        except (KeyError, AttributeError):
            pass  # no view was handed out for this key, or it is gone
        if state.__class__ is Kept:
            return state.value
        view = view_of(state, self.cls)
        self.live[key] = _ref(view)
        return view

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the record at ``key``, or ``default``."""

        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: Any, record: Any) -> None:
        """Hold ``record`` at ``key``."""

        state = _held(record, self.cls)
        dict.__setitem__(self, key, state)
        self.raw[key] = state

    def __delitem__(self, key: Any) -> None:
        """Remove ``key``."""

        dict.__delitem__(self, key)
        del self.raw[key]

    def clear(self) -> None:
        """Remove every item."""

        dict.clear(self)
        self.raw.clear()
        self.live.clear()

    def __iter__(self) -> Iterator[Any]:
        """Iterate the keys (defined here so a copy reads records, never states)."""

        return dict.__iter__(self)

    def values(self) -> Any:
        """Return every record in insertion order."""

        return views_of(list(self.vals), self.cls)

    def items(self) -> Any:
        """Return ``(key, record)`` pairs in insertion order."""

        return list(zip(self.raw, views_of(list(self.vals), self.cls), strict=True))

    def states(self) -> Any:
        """Return the stored states (no record is built)."""

        return self.vals

    def plain(self) -> Any:
        """Return the plain mapping (of the type this stands in for) from key to record."""

        return self.kind(self.items())

    def copy(self) -> Any:
        """Return the plain mapping of the records."""

        return self.plain()

    def pop(self, key: Any, *default: Any) -> Any:
        """Remove ``key`` and return its record (or the default)."""

        if key not in self:
            if default:
                return default[0]
            raise KeyError(key)
        record = self[key]
        del self[key]
        return record

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the last ``(key, record)`` pair."""

        if not self:
            raise KeyError("dictionary is empty")
        key = next(reversed(self))
        return key, self.pop(key)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Return the record at ``key``, holding ``default`` there when absent."""

        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *others: Any, **named: Any) -> None:
        """Hold every record of the given mappings or pairs."""

        for key, record in dict(*others, **named).items():
            self[key] = record

    def __eq__(self, other: object) -> bool:
        """Compare as the plain mapping of records."""

        if other.__class__ is RecordDict:
            other = other.plain()
        return bool(self.plain() == other)

    def __ne__(self, other: object) -> bool:
        """Compare as the plain mapping of records."""

        return not self == other

    def __or__(self, other: Any) -> Any:
        """Return the plain union with ``other``."""

        return self.plain() | other

    def __ror__(self, other: Any) -> Any:
        """Return the plain union of ``other`` with this mapping."""

        return other | self.plain()

    def __ior__(self, other: Any) -> Any:
        """Hold every record of ``other``."""

        self.update(other)
        return self

    def __repr__(self) -> str:
        """Render as the plain mapping this stands in for."""

        return repr(self.plain())

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the plain mapping of records."""

        return (self.kind, (), None, None, iter(self.items()))


def _item(state: Any, cls: type) -> Any:
    """Return the record for one stored item with no live view."""

    return state.value if state.__class__ is Kept else _build([state], cls)[0]


class RecordList(list[Any]):
    """List of records, holding each record's state; every read hands out the record view."""

    __slots__ = ("cls",)

    cls: type

    def __getitem__(self, index: Any) -> Any:
        """Return the record at ``index`` (a plain list of records for a slice)."""

        state = _list_item(self, index)
        if index.__class__ is slice:
            return views_of(state, self.cls)
        try:
            view = _VIEWS[id(state)]()
            if view.__dict__ is state:
                return view
        except (KeyError, AttributeError):
            pass  # no view of this state is alive: build one
        return _item(state, self.cls)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the records."""

        return iter(views_of(list(list.__iter__(self)), self.cls))

    def __reversed__(self) -> Iterator[Any]:
        """Iterate the records from the last."""

        return reversed(self.plain())

    def plain(self) -> list[Any]:
        """Return the plain list of records."""

        return views_of(list(list.__iter__(self)), self.cls)

    def copy(self) -> list[Any]:
        """Return the plain list of records."""

        return self.plain()

    def __setitem__(self, index: Any, record: Any) -> None:
        """Hold ``record`` at ``index`` (the records of an iterable for a slice)."""

        if index.__class__ is slice:
            list.__setitem__(self, index, _all_held(list(record), self.cls))
        else:
            list.__setitem__(self, index, _held(record, self.cls))

    def append(self, record: Any) -> None:
        """Hold ``record`` at the end."""

        list.append(self, _held(record, self.cls))

    def extend(self, records: Iterable[Any]) -> None:
        """Hold every record of ``records`` at the end."""

        list.extend(self, _all_held(list(records), self.cls))

    def insert(self, index: Any, record: Any) -> None:
        """Hold ``record`` before ``index``."""

        list.insert(self, index, _held(record, self.cls))

    def pop(self, index: Any = -1) -> Any:
        """Remove and return the record at ``index``."""

        record = self[index]
        list.__delitem__(self, index)
        return record

    def remove(self, record: Any) -> None:
        """Remove the first item equal to ``record``."""

        list.__delitem__(self, self.plain().index(record))

    def index(self, record: Any, *span: Any) -> int:
        """Return the position of the first item equal to ``record``."""

        return self.plain().index(record, *span)

    def count(self, record: Any) -> int:
        """Return how many items equal ``record``."""

        return self.plain().count(record)

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        """Sort the records in place."""

        self[:] = sorted(self.plain(), key=key, reverse=reverse)

    def __contains__(self, record: object) -> bool:
        """Return whether an item equals ``record``."""

        return record in self.plain()

    def __eq__(self, other: object) -> bool:
        """Compare as the plain list of records."""

        if other.__class__ is RecordList:
            other = other.plain()
        return bool(self.plain() == other)

    def __ne__(self, other: object) -> bool:
        """Compare as the plain list of records."""

        return not self == other

    def __add__(self, other: Any) -> Any:
        """Return the plain list of these records followed by ``other``."""

        return self.plain() + list(other)

    def __radd__(self, other: Any) -> Any:
        """Return ``other`` followed by the plain list of these records."""

        return list(other) + self.plain()

    def __iadd__(self, other: Any) -> Any:
        """Hold every record of ``other`` at the end."""

        self.extend(other)
        return self

    def __mul__(self, times: Any) -> Any:
        """Return the plain list of records repeated."""

        return self.plain() * times

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Render as the plain list of records."""

        return repr(self.plain())

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the plain list of records."""

        return (list, (self.plain(),))


#: The container classes above (``__class__`` membership costs no call).
HELD_CLASSES = (RecordDict, RecordList)


def record_dict(items: Any, cls: type) -> Any:
    """Return ``items`` (key -> record of ``cls``) as a :class:`RecordDict`.

    A ``RecordDict`` passes through (the caller shares it, as it shared the
    mapping before); a ``dict`` or ``OrderedDict`` is adopted and keeps its
    type as the plain form; anything else is returned as given.
    """

    if items.__class__ is not dict and items.__class__ is not OrderedDict:
        return items
    held = RecordDict()
    held.cls, held.kind, held.live = cls, items.__class__, {}
    held.raw = raw = dict(zip(items, _all_held(list(items.values()), cls), strict=True))
    held.vals = raw.values()
    dict.update(held, raw)
    return held


def record_list(items: Any, cls: type) -> Any:
    """Return ``items`` (records of ``cls``) as a :class:`RecordList`; a non-list passes through."""

    if items.__class__ is not list:
        return items
    held = RecordList()
    held.cls = cls
    list.extend(held, _all_held(items, cls))
    return held


def record_groups(groups: Any, cls: type) -> Any:
    """Return a ``key -> list of records`` index with each list as a :class:`RecordList`."""

    if groups.__class__ is not dict:
        return groups
    return {key: record_list(group, cls) for key, group in groups.items()}


def plain(value: Any) -> Any:
    """Return the plain form of a held container (or of an index of held lists)."""

    if value.__class__ in HELD_CLASSES:
        return value.plain()
    if value.__class__ is dict and value:
        if next(iter(value.values())).__class__ is RecordList:
            return {key: plain(group) for key, group in value.items()}
    return value


def handed(stored: _T, owner: Any) -> _T:
    """Return the accessor a door hands out for ``stored``: one that holds ``owner``.

    The object returned shares the stored accessor's state and keeps the
    trace, so records read from it are read while the trace is alive. It is
    cached weakly on the stored accessor: identity holds while anyone keeps
    it. With no owner (or an accessor of another class) the stored accessor
    is returned unchanged.
    """

    twin: Any
    try:
        twin = stored._handed()  # type: ignore[attr-defined]
        if twin._owner is owner:
            return twin
    except (AttributeError, TypeError):
        pass  # none handed out yet, or the one handed out is gone
    if owner is None or stored.__class__ not in _FIELDS:
        return stored
    twin = _new(stored.__class__)
    twin.__dict__ = stored.__dict__
    twin._owner = owner
    stored._handed = _ref(twin)  # type: ignore[attr-defined]
    return twin


class OwnedChildField(ChildAccessorField):
    """The ``Module.ops`` cell: the accessor of a module that holds its trace holds it too."""

    __slots__ = ()

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the module's call accessor (see :func:`handed`)."""

        stored = ChildAccessorField.__get__(self, record, objtype)
        try:
            owner = record._owner
        except AttributeError:
            return stored
        if owner is None:
            return stored
        try:
            twin = stored._handed()
            if twin._owner is owner:
                return twin
        except (AttributeError, TypeError):
            pass  # none handed out yet, or the one handed out is gone
        return handed(stored, owner)


def _source_trace(record: Any) -> Any:
    """Owning Trace, if still alive (named weakly in the state; a handed-out record holds it)."""

    reference = record.__dict__.get(_TRACE_REF)
    return None if reference is None else reference()


def _set_source_trace(record: Any, trace: Any) -> None:
    """Name ``trace`` as the owning Trace; a record a holder handed out holds the new one."""

    record.__dict__[_TRACE_REF] = _ref(trace) if trace is not None else None
    try:
        record._owner  # noqa: B018 - set only on a record a holder adopted or handed out
    except AttributeError:
        return
    _set(record, "_owner", trace)


#: The ``_source_trace`` property of ``Module`` and ``ModuleCall``.
SOURCE_TRACE = property(_source_trace, _set_source_trace)

#: Accessor class -> field name -> ``(converter, record class)`` for the fields that hold records.
_FIELDS: dict[type, dict[str, tuple[Any, type]]] = {}


def _hold_fields(accessor: Any) -> None:
    """Convert the record-holding fields in an accessor's state, in place."""

    state = accessor.__dict__
    for name, (convert, cls) in _FIELDS[accessor.__class__].items():
        value = state.get(name)
        if value is not None:
            state[name] = convert(value, cls)


def _plain_state(accessor: Any) -> dict[str, Any]:
    """Return a copy of an accessor's state with its containers in plain form."""

    state = dict(accessor.__dict__)
    for name in _FIELDS[accessor.__class__]:
        if name in state:
            state[name] = plain(state[name])
    return state


class _HeldField:
    """An accessor field that holds a container of records as states.

    The field has no ``__get__``, so a read finds the value in the instance
    dictionary at no cost; only an assignment comes here, and converts the
    container assigned.
    """

    __slots__ = ("cls", "convert", "name")

    def __init__(self, name: str, convert: Any, cls: type) -> None:
        """Bind the field to its name, its converter and the record class it holds."""

        self.name, self.convert, self.cls = name, convert, cls

    def __set__(self, accessor: Any, value: Any) -> None:
        """Store ``value``, a record container as states."""

        accessor.__dict__[self.name] = value if value is None else self.convert(value, self.cls)

    def __delete__(self, accessor: Any) -> None:
        """Delete the field from the accessor."""

        try:
            del accessor.__dict__[self.name]
        except KeyError:
            raise AttributeError(self.name) from None


def _install_accessor(accessor_cls: Any, fields: dict[str, tuple[Any, type]]) -> None:
    """Make ``accessor_cls`` hold states in ``fields`` and state-stream the plain containers."""

    _FIELDS[accessor_cls] = fields
    base_getstate = accessor_cls.__getstate__
    base_setstate = accessor_cls.__setstate__

    for name, (convert, cls) in fields.items():
        setattr(accessor_cls, name, _HeldField(name, convert, cls))

    def __getstate__(self: Any) -> dict[str, Any]:
        """Return the pickle state, with the plain containers of records the base carried."""

        state: dict[str, Any] = base_getstate(self)
        for name in fields:
            if name in state:
                state[name] = plain(state[name])
        return state

    def __setstate__(self: Any, state: dict[str, Any]) -> None:
        """Restore the pickle state and hold its record containers as states."""

        base_setstate(self, state)
        _hold_fields(self)

    def __tl_state_items__(self: Any) -> Any:
        """Yield the accessor's state fields with the plain containers of records."""

        return iter(_plain_state(self).items())

    for method in (__getstate__, __setstate__, __tl_state_items__):
        method.__qualname__ = f"{accessor_cls.__name__}.{method.__name__}"
        setattr(accessor_cls, method.__name__, method)


#: ``[Module, ModuleCall]``, bound by ``install`` so this module imports no record class.
_CLASSES: list[type] = []


def _records(container: Any) -> list[Any]:
    """Return the records of a held container (or of the plain one it stands in for)."""

    if container.__class__ is RecordDict:
        return list(container.values())
    if container.__class__ is RecordList:
        return list(container.plain())
    return list(container.values() if isinstance(container, dict) else container)


def reached(accessor: Any, *, in_order: bool = False) -> tuple[list[Any], list[Any]]:
    """Return the ``Module`` records of a module accessor and the ``ModuleCall`` records they reach.

    The calls are those of the accessor's call index, which is where a module
    whose ``ops`` cell names rows finds its calls. A module whose cell still
    holds an accessor adds the calls of that accessor. The loops run in C, so
    a caller that needs every record pays one cell read per module and a
    fixed number of calls besides.
    """

    state = accessor.__dict__
    modules = _records(state["_list"] if in_order else state["_dict"])
    calls = _records(state.get("_pass_dict") or ())
    cells = map(RecordCellField.__get__, repeat(_CLASSES[0].__dict__["ops"]), modules)
    loose = [cell for cell in cells if cell.__class__ is not RowKeys and cell]
    if loose:
        seen = set(map(id, calls))
        for cell in loose:
            for call in cell.values():
                if id(call) not in seen:
                    seen.add(id(call))
                    calls.append(call)
    return modules, calls


def rebind(trace: Any) -> None:
    """Name ``trace`` as the owner of every module and call it holds; drop module facet caches."""

    modules, calls = reached(trace._module_logs, in_order=True)
    reference = _ref(trace)
    module_states: list[Any] = list(map(vars, modules))
    states: list[Any] = module_states + list(map(vars, calls))
    deque(map(dict.__setitem__, states, repeat(_TRACE_REF), repeat(reference)), 0)
    deque(map(dict.pop, module_states, repeat("_facets_cache"), repeat(None)), 0)
    deque(map(_set, modules + calls, repeat("_owner"), repeat(trace)), 0)


def install(module_cls: Any, call_cls: Any, module_accessor: Any, call_accessor: Any) -> None:
    """Install the record holders for ``Module`` and ``ModuleCall`` (import time)."""

    _CLASSES[:] = [module_cls, call_cls]
    modules, calls = (record_dict, module_cls), (record_dict, call_cls)
    _install_accessor(
        module_accessor,
        {
            "_dict": modules,
            "_list": (record_list, module_cls),
            "_pass_dict": calls,
            "_alias_dict": modules,
        },
    )
    _install_accessor(call_accessor, {"_dict": calls, "_list": (record_list, call_cls)})
    field = module_cls.__dict__["ops"]
    setattr(module_cls, "ops", OwnedChildField(field._name, field._fid, field._kind))


def install_trace_calls(accessor_cls: Any) -> None:
    """Install the record holders of the trace-level ``ModuleCall`` accessor (import time)."""

    call_cls = _CLASSES[1]
    _install_accessor(
        accessor_cls,
        {
            "_dict": (record_dict, call_cls),
            "_list": (record_list, call_cls),
            "_address_index": (record_groups, call_cls),
        },
    )
