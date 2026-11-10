"""Records as views over state the trace holds.

A record of a kind handled here (``GradFn``, ``GradFnCall``) is a view over
one *state*: its instance dictionary, which carries the row binding and any
undeclared attributes. A trace keeps the state and never the record, so a
trace nobody reads holds no record object at all. A reader gets a view built
over the state; the view is cached weakly, so two reads return the identical
record for as long as anyone keeps it, and the next read after that builds a
fresh one over the same state.

A view handed out holds the trace its ``_source_trace_ref`` names in its
``_owner`` slot. The record keeps the trace and the trace keeps the state;
the trace never refers to the view, so the reference runs one way and a
trace nobody holds is still freed by reference counting. A record copied or
restored on its own names no trace and gets no owner.

``StateList`` and ``StateDict`` are the containers a trace-held structure
uses in place of a list or dictionary of records. Their own items are the
states; every method that hands out or takes items speaks records. Both
pickle and copy as the plain containers of records they stand for.
"""

from __future__ import annotations

import copy
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

_new = object.__new__
_set = object.__setattr__
_list_iter = list.__iter__
_list_get = list.__getitem__
_list_set = list.__setitem__
_dict_get = dict.get
_dict_set = dict.__setitem__
_MISSING: Any = object()

#: The live view of each state, by ``id(state)``. An entry counts only while
#: its view's ``__dict__`` is that state, and a live view keeps its state
#: alive, so the id of a state with an entry cannot be reused.
_VIEWS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

#: Trace fields that hold records as state, with the record class of each
#: (bound once, when the record class installs its facade).
TRACE_STATE_FIELDS: dict[str, type] = {}


class Foreign:
    """A stored plain ``dict`` that is a value of its own, not a record's state."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        """Wrap ``value``."""

        self.value = value


def own(record: Any) -> Any:
    """Give ``record`` the live trace its ``_source_trace_ref`` names as its owner; return it."""

    reference = getattr(record, "_source_trace_ref", None)
    trace = reference() if reference is not None else None
    if trace is not None and getattr(record, "_owner", None) is not trace:
        _set(record, "_owner", trace)
    return record


def set_owner(record: Any, trace: Any) -> None:
    """Make ``trace`` the owner of ``record`` (``None`` releases the one it has)."""

    if trace is not None:
        _set(record, "_owner", trace)
    elif getattr(record, "_owner", None) is not None:
        object.__delattr__(record, "_owner")


def hold(cls: type, value: Any) -> Any:
    """Return what a state container stores for ``value``.

    A record of exactly ``cls`` is stored as its state, and stays the view of
    that state while it lives. Anything else is stored as it is (a plain
    ``dict`` wrapped, so it is not read back as a state).
    """

    if value.__class__ is cls:
        state = value.__dict__
        _VIEWS[id(state)] = value
        return state
    if value.__class__ is dict:
        return Foreign(value)
    return value


def twin(cls: type, state: dict[str, Any]) -> Any:
    """Return the live ``cls`` view over ``state``, building one when there is none."""

    found = _VIEWS.get(id(state))
    if found is None or found.__dict__ is not state:
        found = _new(cls)
        _set(found, "__dict__", state)
        _VIEWS[id(state)] = found
    return found


def view(cls: type, stored: Any) -> Any:
    """Return the value a state container hands out for ``stored`` (see ``hold``)."""

    if stored.__class__ is dict:
        return own(twin(cls, stored))
    if stored.__class__ is Foreign:
        return stored.value
    return stored


class StateList(list[Any]):
    """A list of records stored as their states; reads return the records."""

    __slots__ = ("_cls",)

    def __init__(self, cls: type, records: Any = ()) -> None:
        """Hold ``records`` of class ``cls``."""

        self._cls = cls
        list.__init__(self, [hold(cls, record) for record in records])

    def plain(self) -> list[Any]:
        """Return the records as a plain list."""

        cls = self._cls
        return [view(cls, stored) for stored in _list_iter(self)]

    def __iter__(self) -> Iterator[Any]:
        """Iterate the records in order."""

        return iter(self.plain())

    def __reversed__(self) -> Iterator[Any]:
        """Iterate the records in reverse order."""

        return reversed(self.plain())

    def __contains__(self, value: object) -> bool:
        """Return whether ``value`` is one of the records."""

        return value in self.plain()

    def __getitem__(self, index: Any) -> Any:
        """Return one record, or a plain list of records for a slice."""

        stored = _list_get(self, index)
        if index.__class__ is slice:
            return [view(self._cls, item) for item in stored]
        return view(self._cls, stored)

    def __setitem__(self, index: Any, value: Any) -> None:
        """Replace one record or a slice of records."""

        if index.__class__ is slice:
            _list_set(self, index, [hold(self._cls, item) for item in value])
        else:
            _list_set(self, index, hold(self._cls, value))

    def append(self, value: Any) -> None:
        """Append a record."""

        list.append(self, hold(self._cls, value))

    def extend(self, values: Any) -> None:
        """Append every record of ``values``."""

        list.extend(self, [hold(self._cls, item) for item in values])

    def insert(self, index: Any, value: Any) -> None:
        """Insert a record before ``index``."""

        list.insert(self, index, hold(self._cls, value))

    def __iadd__(self, values: Any) -> Any:
        """Append every record of ``values`` in place."""

        self.extend(values)
        return self

    def pop(self, index: Any = -1) -> Any:
        """Remove and return the record at ``index``."""

        return view(self._cls, list.pop(self, index))

    def remove(self, value: Any) -> None:
        """Remove the first record equal to ``value``."""

        del self[self.plain().index(value)]

    def index(self, value: Any, *bounds: Any) -> int:
        """Return the position of the first record equal to ``value``."""

        return self.plain().index(value, *bounds)

    def count(self, value: Any) -> int:
        """Return how many records equal ``value``."""

        return self.plain().count(value)

    def copy(self) -> list[Any]:
        """Return the records as a plain list."""

        return self.plain()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        """Sort the records in place."""

        records = self.plain()
        records.sort(key=key, reverse=reverse)
        self[:] = records

    def __add__(self, other: Any) -> list[Any]:
        """Return a plain list of these records followed by ``other``."""

        return self.plain() + list(other)

    def __radd__(self, other: Any) -> list[Any]:
        """Return a plain list of ``other`` followed by these records."""

        return list(other) + self.plain()

    def __mul__(self, times: Any) -> list[Any]:
        """Return a plain list of the records repeated."""

        return self.plain() * times

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        """Compare as the plain list of records."""

        return bool(self.plain() == (other.plain() if isinstance(other, StateList) else other))

    def __ne__(self, other: object) -> bool:
        """Compare as the plain list of records."""

        return not self == other

    def __repr__(self) -> str:
        """Show the records, as a plain list does."""

        return repr(self.plain())

    def __reduce_ex__(self, protocol: Any) -> Any:
        """Pickle and copy as a plain list of the records."""

        return list, (self.plain(),)

    def __deepcopy__(self, memo: dict[int, Any]) -> StateList:
        """Copy the records and hold the copies the same way."""

        copied = StateList(self._cls)
        memo[id(self)] = copied
        copied.extend(copy.deepcopy(self.plain(), memo))
        return copied


class StateDict(dict[Any, Any]):
    """A dictionary of records stored as their states; reads return the records.

    ``ordered`` records whether it stands for an ``OrderedDict`` (it pickles,
    copies and compares as one) or a plain ``dict``.
    """

    __slots__ = ("_cls", "_ordered")

    def __init__(self, cls: type, items: Any = (), *, ordered: bool = False) -> None:
        """Hold the records of ``items`` (a mapping or pairs) of class ``cls``."""

        self._cls = cls
        self._ordered = ordered
        dict.__init__(self)
        self.update(items)

    def plain(self) -> Any:
        """Return the records in a plain ``dict`` or ``OrderedDict``."""

        cls = self._cls
        pairs = [(key, view(cls, stored)) for key, stored in dict.items(self)]
        return OrderedDict(pairs) if self._ordered else dict(pairs)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the keys (defined here so ``dict(self)`` reads through ``__getitem__``)."""

        return dict.__iter__(self)

    def __getitem__(self, key: Any) -> Any:
        """Return the record at ``key``."""

        return view(self._cls, dict.__getitem__(self, key))

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the record at ``key``, or ``default``."""

        stored = _dict_get(self, key, _MISSING)
        return default if stored is _MISSING else view(self._cls, stored)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Hold ``value`` at ``key``."""

        _dict_set(self, key, hold(self._cls, value))

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Return the record at ``key``, holding ``default`` there first when it is absent."""

        if key not in self:
            self[key] = default
        return self[key]

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Hold every record of the given mapping, pairs and keywords."""

        cls = self._cls
        for key, value in dict(*args, **kwargs).items():
            _dict_set(self, key, hold(cls, value))

    def __ior__(self, other: Any) -> Any:
        """Hold every record of ``other`` in place."""

        self.update(other)
        return self

    def pop(self, key: Any, *default: Any) -> Any:
        """Remove ``key`` and return its record (or the given default)."""

        stored = dict.pop(self, key, _MISSING)
        if stored is not _MISSING:
            return view(self._cls, stored)
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the last ``(key, record)`` pair."""

        key, stored = dict.popitem(self)
        return key, view(self._cls, stored)

    def values(self) -> Any:
        """Return the records in order, as a list."""

        cls = self._cls
        return [view(cls, stored) for stored in dict.values(self)]

    def items(self) -> Any:
        """Return the ``(key, record)`` pairs in order, as a list."""

        cls = self._cls
        return [(key, view(cls, stored)) for key, stored in dict.items(self)]

    def copy(self) -> Any:
        """Return the records in a plain dictionary."""

        return self.plain()

    def __eq__(self, other: object) -> bool:
        """Compare as the plain dictionary of records."""

        return bool(self.plain() == (other.plain() if isinstance(other, StateDict) else other))

    def __ne__(self, other: object) -> bool:
        """Compare as the plain dictionary of records."""

        return not self == other

    def __repr__(self) -> str:
        """Show the records, as the plain dictionary does."""

        return repr(self.plain())

    def __reduce_ex__(self, protocol: Any) -> Any:
        """Pickle and copy as the plain dictionary of the records."""

        return (OrderedDict if self._ordered else dict), (self.plain(),)

    def __deepcopy__(self, memo: dict[int, Any]) -> StateDict:
        """Copy the records and hold the copies the same way."""

        copied = StateDict(self._cls, ordered=self._ordered)
        memo[id(self)] = copied
        copied.update(copy.deepcopy(self.plain(), memo))
        return copied


def project_trace_states(state: dict[str, Any]) -> dict[str, Any]:
    """Replace the state containers in a trace's ``state`` with plain ones of records; return it."""

    for name in TRACE_STATE_FIELDS:
        held = state.get(name)
        if held.__class__ is StateDict:
            state[name] = held.plain()
    return state


def adopt_trace_states(state: dict[str, Any]) -> None:
    """Turn plain dictionaries of records in a trace's ``state`` into state containers."""

    for name, cls in TRACE_STATE_FIELDS.items():
        plain = state.get(name)
        if plain and plain.__class__ in (dict, OrderedDict):
            if any(value.__class__ is cls for value in plain.values()):
                state[name] = StateDict(cls, plain, ordered=plain.__class__ is OrderedDict)
