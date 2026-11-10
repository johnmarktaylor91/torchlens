"""Trace-side op containers that hold row keys instead of ``Op`` records.

``Trace._layer_list`` and the two label dictionaries used to hold every ``Op``
directly. They are now an ``OpSequence`` and two ``OpLabelMap`` objects that
store one integer key per entry and hand the record back on read, so code that
iterates, indexes, appends, inserts or assigns sees the same records as before.
They subclass ``list`` and ``dict`` with the keys as their own items, which
keeps ``len`` and membership at built-in speed.

A key is the op's row in the trace core's op store. A record that has no row
there (a detached single-row copy, or any record of a trace without a core)
gets the next free negative key. The record behind a row key is read through
the trace core's view table (``_trace_core/view_tables.py``): strongly held
while the trace is built, weakly afterwards, when a record nobody kept is
built again from its row on the next read. A record behind a negative key is
held strongly in the table's ``detached`` dictionary. A trace without a core
keeps a plain row list and dictionary on its ``OpRows`` resolver until a core
is bound.
"""

from __future__ import annotations

import copy
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from itertools import repeat
from typing import TYPE_CHECKING, Any, cast

from .._trace_core.record_refs import VERSIONS_LIST
from .._trace_core.view_tables import WeakViews

if TYPE_CHECKING:
    from .op import Op

CONTAINER_FIELDS = ("_layer_list", "_layer_dict_main_keys", "_layer_dict_all_keys")

_MISSING: Any = object()
_slot = object.__getattribute__
_list_iter = list.__iter__
_list_get = list.__getitem__
_list_set = list.__setitem__
_list_insert = list.insert
_list_append = list.append
_dict_iter = dict.__iter__
_dict_item = dict.__getitem__
_dict_get = dict.get
_dict_set = dict.__setitem__
_dict_values = dict.values


class OpRows:
    """Resolve op row keys to records for one trace's containers."""

    __slots__ = (
        "__weakref__",
        "_containers",
        "_core_ref",
        "_ids",
        "_last",
        "_store",
        "_tables",
        "_trace_ref",
    )

    def __init__(self, trace: Any = None) -> None:
        """Start unbound: records are cached locally until the trace has a core."""

        self._trace_ref = weakref.ref(trace) if trace is not None else None
        self._core_ref: Any = None
        self._store: Any = None
        self._tables: tuple[list[Any], dict[int, Any]] = ([], {})
        self._ids: dict[int, int] = {}
        # The row key ``hold`` last returned: a writer that registers one record
        # under several labels in a row finds its key again without a call.
        self._last = -1
        self._containers: list[weakref.ref[Any]] = []

    def _trace_core(self) -> Any:
        """Return the owning trace's current core, or ``None``."""

        trace = self._trace_ref() if self._trace_ref is not None else None
        return trace.__dict__.get("_trace_core") if trace is not None else None

    def register(self, container: Any) -> None:
        """Remember a container (weakly) so ``bind`` can move its keys."""

        live = [ref for ref in self._containers if ref() is not None]
        self._containers = [*live, weakref.ref(container)]

    def bind(self) -> None:
        """Move every held record to the side tables of the trace's current core."""

        core = self._trace_core()
        if core is None:
            return
        if self._core_ref is not None and self._core_ref() is core:
            self._store = core.ops
            return
        live = [
            container for container in (ref() for ref in self._containers) if container is not None
        ]
        contents = [(container, container._records()) for container in live]
        self._core_ref = weakref.ref(core)
        self._store = core.ops
        views = core._views
        views.set_owner(self._trace_ref() if self._trace_ref is not None else None)
        self._tables = (views.ops, views.detached)
        self._last = -1
        self._ids = {id(record): key for key, record in self._tables[1].items()}
        for container, records in contents:
            container._rekey(self.hold_many(records))

    def hold(self, record: Any) -> int:
        """Cache ``record`` in the side table and return its key."""

        try:
            store = _slot(record, "_core")
            row = _slot(record, "_row")
        except AttributeError:
            store = None
        if store is not None:
            if store is not self._store:
                self.bind()  # the trace may have gained its core since the last write
            if store is self._store:
                rows = self._tables[0]
                if rows.__class__ is WeakViews:
                    # ``WeakViews.claim`` written out: this runs once per op of
                    # every fork, so it makes no call while the row's view is
                    # alive. Never a plain read here: it would build a record
                    # for the row, and the one being held would lose its row.
                    weak: Any = rows
                    try:
                        held = weak.refs[row]()
                    except IndexError:
                        weak.pad(row + 1)
                        held = None
                    if held is None:
                        weak[row] = held = record
                else:
                    try:
                        held = rows[row]
                    except IndexError:
                        size = len(rows)
                        if row > size:
                            rows.extend(repeat(None, row - size))
                        rows.append(record)
                        self._last = row
                        return row
                    if held is None:
                        rows[row] = held = record
                if held is record:
                    self._last = row
                    return row
        return self._hold_detached(record)

    def _hold_detached(self, record: Any) -> int:
        """Cache a record that has no row in the bound store; return its negative key."""

        detached = self._tables[1]
        key = self._ids.get(id(record))
        if key is None:
            key = -1 - len(detached)
            while key in detached:
                key -= 1
            detached[key] = record
            self._ids[id(record)] = key
        return key

    def hold_many(self, records: Iterable[Any]) -> list[int]:
        """Cache every record and return the keys in order."""

        hold = self.hold
        return [hold(record) for record in records]

    def resolve(self, key: int) -> Any:
        """Return the record behind one key."""

        return self._tables[0][key] if key >= 0 else self._tables[1][key]

    def records(self, keys: Iterable[int]) -> Iterator[Any]:
        """Return an iterator over the records behind ``keys``."""

        rows, detached = self._tables
        if not detached:
            if rows.__class__ is WeakViews:
                return iter(cast("WeakViews", rows).many(keys))
            return map(rows.__getitem__, keys)
        if not rows:
            return map(detached.__getitem__, keys)
        return map(self.resolve, keys)

    def copy_for(self, memo: dict[Any, Any]) -> OpRows:
        """Return the resolver of the copied trace for one ``deepcopy`` memo."""

        known = memo.get(id(self))
        if known is None:
            trace = self._trace_ref() if self._trace_ref is not None else None
            known = memo[id(self)] = OpRows(memo.get(id(trace)) if trace is not None else None)
            memo.setdefault(id(memo), []).append(self)
        return known


def _copied(records: Iterable[Any], memo: dict[Any, Any]) -> list[Any]:
    """Return the ``deepcopy`` of each record under ``memo`` (a memo hit costs no call)."""

    get = memo.get
    out = []
    for record in records:
        known = get(id(record), _MISSING)
        out.append(copy.deepcopy(record, memo) if known is _MISSING else known)
    return out


class OpSequence(list[Any]):
    """A list of ops stored as row keys; reads return the ``Op`` records.

    The list's own items are the integer keys, so ``len`` and deletion run at
    list speed; every method that hands out or takes items is overridden to
    speak records.
    """

    __slots__ = ("__weakref__", "_owner")

    def __init__(self, owner: OpRows, records: Iterable[Any] = ()) -> None:
        """Hold ``records`` through ``owner``."""

        self._owner = owner
        list.__init__(self, owner.hold_many(records))
        owner.register(self)

    @property
    def _keys(self) -> list[int]:
        """Return a copy of the row keys (for tests and diagnostics)."""

        return list(_list_iter(self))

    def _records(self) -> list[Any]:
        """Return the records as a plain list."""

        return list(self._owner.records(_list_iter(self)))

    def _rekey(self, keys: list[int]) -> None:
        """Replace the keys in place after the owner moved its records.

        In place because a pending append already holds this list.
        """

        _list_set(self, slice(None), keys)

    def _key(self, record: Any) -> int:
        """Return the key for ``record``, holding it if it is not the one last held."""

        owner = self._owner
        row = owner._last
        return row if row >= 0 and owner._tables[0][row] is record else owner.hold(record)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the records in order."""

        return self._owner.records(_list_iter(self))

    def __reversed__(self) -> Iterator[Any]:
        """Iterate the records in reverse order."""

        return self._owner.records(_list_get(self, slice(None, None, -1)))

    def __contains__(self, value: object) -> bool:
        """Return whether ``value`` is one of the records."""

        return value in self._records()

    def __getitem__(self, index: Any) -> Any:
        """Return one record, or a plain list of records for a slice."""

        key = _list_get(self, index)
        owner = self._owner
        if key.__class__ is list:
            return list(owner.records(key))
        return owner._tables[0][key] if key >= 0 else owner._tables[1][key]

    def __setitem__(self, index: Any, value: Any) -> None:
        """Replace one record or a slice of records."""

        if index.__class__ is slice:
            _list_set(self, index, self._owner.hold_many(value))
        else:
            _list_set(self, index, self._key(value))

    def insert(self, index: Any, value: Any) -> None:
        """Insert a record before ``index``."""

        _list_insert(self, index, self._key(value))

    def append(self, value: Any) -> None:
        """Append a record."""

        owner = self._owner
        row = owner._last
        if row < 0 or owner._tables[0][row] is not value:
            row = owner.hold(value)
        _list_append(self, row)

    def extend(self, values: Iterable[Any]) -> None:
        """Append every record of ``values``."""

        list.extend(self, self._owner.hold_many(list(values)))

    def pop(self, index: Any = -1) -> Any:
        """Remove and return the record at ``index``."""

        return self._owner.resolve(list.pop(self, index))

    def remove(self, value: Any) -> None:
        """Remove the first position of ``value``."""

        del self[self.index(value)]

    def index(self, value: Any, *bounds: Any) -> int:
        """Return the first position of ``value``."""

        return self._records().index(value, *bounds)

    def count(self, value: Any) -> int:
        """Return how many positions hold ``value``."""

        return self._records().count(value)

    def copy(self) -> list[Any]:
        """Return a plain list of the records."""

        return self._records()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        """Sort the records in place."""

        self._rekey(self._owner.hold_many(sorted(self, key=key, reverse=reverse)))

    def __add__(self, other: Any) -> list[Any]:
        """Return a plain list: these records followed by ``other``."""

        return self._records() + list(other)

    def __radd__(self, other: Any) -> list[Any]:
        """Return a plain list: ``other`` followed by these records."""

        return list(other) + self._records()

    def __iadd__(self, other: Any) -> Any:
        """Append every record of ``other`` in place."""

        self.extend(other)
        return self

    def __mul__(self, count: Any) -> list[Any]:
        """Return a plain list of the records repeated ``count`` times."""

        return self._records() * count

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        """Compare as a list of records."""

        if isinstance(other, list):
            return self._records() == list(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        """Compare as a list of records."""

        if isinstance(other, list):
            return self._records() != list(other)
        return NotImplemented

    def _unordered(self, other: Any) -> Any:
        """Decline an ordering comparison: ops have no order."""

        return NotImplemented

    __lt__ = __le__ = __gt__ = __ge__ = _unordered

    def __repr__(self) -> str:
        """Return the list repr of the records."""

        return repr(self._records())

    def __copy__(self) -> OpSequence:
        """Return a new sequence over the same records."""

        return OpSequence(self._owner, self)

    def __deepcopy__(self, memo: dict[Any, Any]) -> OpSequence:
        """Copy for a fork or a deep copy: the copied trace's records, same order."""

        new = OpSequence(self._owner.copy_for(memo), _copied(self, memo))
        memo[id(self)] = new
        return new

    def __reduce_ex__(self, protocol: Any) -> Any:
        """Pickle as the plain list of records."""

        return list, (self._records(),)


class OpLabelMap(dict[Any, Any]):
    """A label dictionary stored as label to row key; reads return the ``Op`` records.

    The dictionary's own values are the integer keys, so ``len``, membership
    and deletion run at dictionary speed; every method that hands out or takes
    values is overridden to speak records.
    """

    __slots__ = ("__weakref__", "_ordered", "_owner")

    def __new__(cls, owner: OpRows, items: Any = (), *, ordered: bool = False) -> OpLabelMap:
        """Create the map; an ordered one is an ``OpOrderedLabelMap``."""

        del owner, items
        return dict.__new__(OpOrderedLabelMap if ordered and cls is OpLabelMap else cls)

    def __init__(self, owner: OpRows, items: Any = (), *, ordered: bool = False) -> None:
        """Hold the records of ``items`` through ``owner``; ``ordered`` names the plain type."""

        self._owner = owner
        self._ordered = ordered or type(self) is not OpLabelMap
        pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
        keys = owner.hold_many([record for _label, record in pairs])
        dict.__init__(self, zip((label for label, _record in pairs), keys, strict=True))
        owner.register(self)

    @property
    def _keys(self) -> dict[Any, int]:
        """Return a copy of the label to row key table (for tests and diagnostics)."""

        return dict(zip(_dict_iter(self), _dict_values(self), strict=True))

    def _records(self) -> list[Any]:
        """Return the records in label order."""

        return list(self._owner.records(_dict_values(self)))

    def _rekey(self, keys: list[int]) -> None:
        """Replace the keys in place after the owner moved its records."""

        dict.update(self, zip(list(_dict_iter(self)), keys, strict=True))

    def plain(self) -> dict[Any, Any]:
        """Return a plain dictionary (or ``OrderedDict``) from label to record."""

        pairs = zip(_dict_iter(self), self._records(), strict=True)
        return OrderedDict(pairs) if self._ordered else dict(pairs)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the labels in insertion order.

        Overridden so ``dict(this)`` and ``{**this}`` read through
        ``__getitem__`` (records) and never copy the row keys.
        """

        return _dict_iter(self)

    def __getitem__(self, label: Any) -> Any:
        """Return the record registered under ``label``."""

        key = _dict_item(self, label)
        tables = self._owner._tables
        return tables[0][key] if key >= 0 else tables[1][key]

    def __setitem__(self, label: Any, record: Any) -> None:
        """Register ``record`` under ``label``."""

        owner = self._owner
        row = owner._last
        if row < 0 or owner._tables[0][row] is not record:
            row = owner.hold(record)
        _dict_set(self, label, row)

    def get(self, label: Any, default: Any = None) -> Any:
        """Return the record under ``label``, or ``default``."""

        key = _dict_get(self, label)
        if key is None:
            return default
        tables = self._owner._tables
        return tables[0][key] if key >= 0 else tables[1][key]

    def values(self) -> Any:
        """Return the records in label order, as a list."""

        return self._records()

    def items(self) -> Any:
        """Return (label, record) pairs, as a list."""

        return list(zip(_dict_iter(self), self._records(), strict=True))

    def pop(self, label: Any, *default: Any) -> Any:
        """Remove ``label`` and return its record, or the default when it is absent."""

        key = dict.pop(self, label, None)
        if key is not None:
            return self._owner.resolve(key)
        if default:
            return default[0]
        raise KeyError(label)

    def popitem(self) -> tuple[Any, Any]:
        """Remove and return the last (label, record) pair."""

        label, key = dict.popitem(self)
        return label, self._owner.resolve(key)

    def setdefault(self, label: Any, default: Any = None) -> Any:
        """Return the record under ``label``, registering ``default`` first when absent."""

        if label not in self:
            self[label] = default
        return self[label]

    def update(self, *others: Any, **labelled: Any) -> None:
        """Register every (label, record) pair of the arguments."""

        for label, record in dict(*others, **labelled).items():
            self[label] = record

    def copy(self) -> dict[Any, Any]:
        """Return a plain dictionary from label to record."""

        return self.plain()

    def __or__(self, other: Any) -> dict[Any, Any]:
        """Return a plain dictionary: these pairs updated with ``other``."""

        return {**self.plain(), **other}

    def __ror__(self, other: Any) -> dict[Any, Any]:
        """Return a plain dictionary: ``other`` updated with these pairs."""

        return {**other, **self.plain()}

    def __ior__(self, other: Any) -> Any:
        """Register every pair of ``other`` in place."""

        self.update(other)
        return self

    def __eq__(self, other: object) -> bool:
        """Compare as a dictionary from label to record."""

        if isinstance(other, OpLabelMap):
            return self.plain() == other.plain()
        if isinstance(other, Mapping):
            return self.plain() == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        """Compare as a dictionary from label to record."""

        if isinstance(other, Mapping):
            return not self == other
        return NotImplemented

    def __repr__(self) -> str:
        """Return the dictionary repr."""

        return repr(self.plain())

    def __copy__(self) -> OpLabelMap:
        """Return a new map over the same records."""

        return OpLabelMap(self._owner, self.items(), ordered=self._ordered)

    def __deepcopy__(self, memo: dict[Any, Any]) -> OpLabelMap:
        """Copy for a fork or a deep copy: the copied trace's records, same labels."""

        pairs = zip(_dict_iter(self), _copied(self._records(), memo), strict=True)
        new = OpLabelMap(self._owner.copy_for(memo), pairs, ordered=self._ordered)
        memo[id(self)] = new
        return new

    def __reduce_ex__(self, protocol: Any) -> Any:
        """Pickle as the plain dictionary of records."""

        return (OrderedDict if self._ordered else dict), (self.plain(),)


class OpOrderedLabelMap(OpLabelMap):
    """The label map that stands in for an ``OrderedDict`` (its plain form is one).

    A separate type because the plain types were separate: a live trace's
    all-keys dictionary is ordered and a loaded file's is not, and a reader
    that tells them apart by type still can.
    """

    __slots__ = ()


def op_rows(trace: Any) -> OpRows:
    """Return the resolver shared by ``trace``'s op containers, creating one if none exists."""

    state = trace.__dict__
    for name in CONTAINER_FIELDS:
        container = state.get(name)
        if isinstance(container, (OpSequence, OpLabelMap)):
            return container._owner
    return OpRows(trace)


def op_holders(trace: Any) -> tuple[list[Op], dict[str, Op], dict[str, Op]]:
    """Return a new trace's three empty op containers.

    In order: ``_layer_list`` (every op, in order), ``_layer_dict_main_keys``
    (primary label to op) and ``_layer_dict_all_keys`` (every lookup key to op).
    They are typed as the plain list and dictionaries they stand in for.
    """

    owner = OpRows(trace)
    holders = (OpSequence(owner), OpLabelMap(owner, ordered=True), OpLabelMap(owner, ordered=True))
    return cast("tuple[list[Op], dict[str, Op], dict[str, Op]]", holders)


def restart_op_holders(trace: Any) -> None:
    """Give ``trace`` a fresh empty op list and a fresh plain primary-label dictionary.

    This is the labeling pass's restart (formerly ``_layer_list = []`` and
    ``_layer_dict_main_keys = {}``); the all-keys dictionary is left as it is.
    """

    owner = op_rows(trace)
    trace._layer_list = OpSequence(owner)
    trace._layer_dict_main_keys = OpLabelMap(owner)


def clear_op_holders(trace: Any) -> None:
    """Give ``trace`` a fresh empty op list and empty both label dictionaries in place."""

    trace._layer_list = OpSequence(op_rows(trace))
    trace._layer_dict_main_keys.clear()
    trace._layer_dict_all_keys.clear()


# The kept ``Buffer.versions`` container: the version ops held by row, read as the ops.
VERSIONS_LIST[:] = [lambda trace, ops: OpSequence(op_rows(trace), ops)]


def bind_op_containers(trace: Any, core: Any = None) -> None:
    """Install ``core`` on ``trace`` (when given), then adopt and bind its op containers.

    A fork arrives here with containers its field copy made (``__deepcopy__``),
    still keyed locally because the fork had no core while its fields were copied.
    """

    if core is not None:
        trace.__dict__["_trace_core"] = core
    adopt_op_containers(trace)


def adopt_op_containers(trace: Any, rebuild_core: Any = None) -> None:
    """Wrap plain containers found on ``trace`` (after a state restore) and bind them.

    ``rebuild_core`` (the restored trace's core rehydration) runs first, on the
    plain containers, so the records are keyed by the rows it gives them. An
    empty plain container holds no record and is left as it is (a field absent
    from the restored state keeps its plain default).
    """

    if rebuild_core is not None:
        rebuild_core(trace)
    state = trace.__dict__
    owner = op_rows(trace)
    plain_list = state.get("_layer_list")
    if plain_list and type(plain_list) is list:
        state["_layer_list"] = OpSequence(owner, plain_list)
    for name in CONTAINER_FIELDS[1:]:
        plain = state.get(name)
        if plain and type(plain) in (dict, OrderedDict):
            state[name] = OpLabelMap(owner, plain, ordered=isinstance(plain, OrderedDict))
    owner.bind()


def project_op_containers(state: dict[str, Any]) -> dict[str, Any]:
    """Replace the op containers in ``state`` with plain ones holding the records; return it."""

    sequence = state.get("_layer_list")
    if isinstance(sequence, OpSequence):
        state["_layer_list"] = sequence.copy()
    for name in CONTAINER_FIELDS[1:]:
        mapping = state.get(name)
        if isinstance(mapping, OpLabelMap):
            state[name] = mapping.plain()
    return state
