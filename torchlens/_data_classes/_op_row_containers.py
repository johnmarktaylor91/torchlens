"""Trace-side op containers that hold row keys instead of ``Op`` records.

``Trace._layer_list`` and the two label dictionaries used to hold every ``Op``
directly. They are now an ``OpSequence`` and two ``OpLabelMap`` objects that
store one integer key per entry and hand the record back on read, so code that
iterates, indexes, appends, inserts or assigns sees the same records as before.

A key is the op's row in the trace core's op store. A record that has no row
there (a detached single-row copy, or any record of a trace without a core)
gets the next free negative key. Until records are handed out weakly, the
record behind a key is cached strongly in one side table on the core:
``TraceCore._strong_facades["op"]`` is a list indexed by store row (``None``
where no record was cached) and ``TraceCore._strong_facades["op.detached"]``
is a dictionary from negative key to record. A trace without a core keeps the
same two tables on its ``OpRows`` resolver until a core is bound.
"""

from __future__ import annotations

import copy
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, MutableSequence
from itertools import repeat
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .op import Op

OP_KIND = "op"
DETACHED_KIND = "op.detached"
CONTAINER_FIELDS = ("_layer_list", "_layer_dict_main_keys", "_layer_dict_all_keys")

_MISSING: Any = object()
_slot = object.__getattribute__


def _side_tables(core: Any) -> tuple[list[Any], dict[int, Any]]:
    """Return the core's op side tables (rows list, detached dictionary), creating them once."""

    facades = core._strong_facades
    rows = facades.get(OP_KIND)
    if rows is None:
        rows = facades[OP_KIND] = []
    detached = facades.get(DETACHED_KIND)
    if detached is None:
        detached = facades[DETACHED_KIND] = {}
    return rows, detached


class OpRows:
    """Resolve op row keys to records for one trace's containers."""

    __slots__ = ("__weakref__", "_containers", "_core_ref", "_ids", "_tables", "_trace_ref")

    def __init__(self, trace: Any = None) -> None:
        """Start unbound: records are cached locally until the trace has a core."""

        self._trace_ref = weakref.ref(trace) if trace is not None else None
        self._core_ref: Any = None
        self._tables: tuple[list[Any], dict[int, Any]] = ([], {})
        self._ids: dict[int, int] = {}
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
        if core is None or (self._core_ref is not None and self._core_ref() is core):
            return
        live = [
            container for container in (ref() for ref in self._containers) if container is not None
        ]
        contents = [(container, container._records()) for container in live]
        self._core_ref = weakref.ref(core)
        self._tables = _side_tables(core)
        self._ids = {id(record): key for key, record in self._tables[1].items()}
        for container, records in contents:
            container._rekey(self.hold_many(records))

    def hold(self, record: Any) -> int:
        """Cache ``record`` in the side table and return its key."""

        core = self._core_ref() if self._core_ref is not None else None
        current = self._trace_core()
        if current is not None and current is not core:
            self.bind()
            core = current
        rows, detached = self._tables
        if core is not None:
            try:
                store = _slot(record, "_core")
                row = _slot(record, "_row")
            except AttributeError:
                store = None
            if store is not None and store is core.ops:
                if row >= len(rows):
                    rows.extend(repeat(None, row + 1 - len(rows)))
                held = rows[row]
                if held is None:
                    rows[row] = record
                    return row
                if held is record:
                    return row
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


class OpSequence(MutableSequence[Any]):
    """A list of ops stored as row keys; reads return the ``Op`` records."""

    __slots__ = ("__weakref__", "_keys", "_owner")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, owner: OpRows, records: Iterable[Any] = ()) -> None:
        """Hold ``records`` through ``owner``."""

        self._owner = owner
        self._keys: list[int] = owner.hold_many(records)
        owner.register(self)

    def _records(self) -> list[Any]:
        """Return the records as a plain list."""

        return list(self._owner.records(self._keys))

    def _rekey(self, keys: list[int]) -> None:
        """Replace the keys in place after the owner moved its records.

        In place because a pending ``self._keys.append(hold(...))`` already
        looked up the list before ``hold`` triggered the move.
        """

        self._keys[:] = keys

    def __len__(self) -> int:
        """Return the number of ops."""

        return len(self._keys)

    def __iter__(self) -> Iterator[Any]:
        """Iterate the records in order."""

        return self._owner.records(self._keys)

    def __reversed__(self) -> Iterator[Any]:
        """Iterate the records in reverse order."""

        return self._owner.records(self._keys[::-1])

    def __getitem__(self, index: Any) -> Any:
        """Return one record, or a plain list of records for a slice."""

        if isinstance(index, slice):
            return list(self._owner.records(self._keys[index]))
        return self._owner.resolve(self._keys[index])

    def __setitem__(self, index: Any, value: Any) -> None:
        """Replace one record or a slice of records."""

        if isinstance(index, slice):
            self._keys[index] = self._owner.hold_many(value)
        else:
            self._keys[index] = self._owner.hold(value)

    def __delitem__(self, index: Any) -> None:
        """Drop one position or a slice."""

        del self._keys[index]

    def insert(self, index: int, value: Any) -> None:
        """Insert a record before ``index``."""

        self._keys.insert(index, self._owner.hold(value))

    def append(self, value: Any) -> None:
        """Append a record."""

        self._keys.append(self._owner.hold(value))

    def extend(self, values: Iterable[Any]) -> None:
        """Append every record of ``values``."""

        self._keys.extend(self._owner.hold_many(list(values)))

    def clear(self) -> None:
        """Drop every position."""

        self._keys.clear()

    def index(self, value: Any, *bounds: Any) -> int:
        """Return the first position of ``value``."""

        return self._records().index(value, *bounds)

    def copy(self) -> list[Any]:
        """Return a plain list of the records."""

        return self._records()

    def sort(self, *, key: Any = None, reverse: bool = False) -> None:
        """Sort the records in place."""

        self._keys[:] = self._owner.hold_many(sorted(self, key=key, reverse=reverse))

    def __add__(self, other: Any) -> list[Any]:
        """Return a plain list: these records followed by ``other``."""

        return self._records() + list(other)

    def __radd__(self, other: Any) -> list[Any]:
        """Return a plain list: ``other`` followed by these records."""

        return list(other) + self._records()

    def __eq__(self, other: object) -> bool:
        """Compare as a list of records."""

        if isinstance(other, (list, OpSequence)):
            return self._records() == list(other)
        return NotImplemented

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


class OpLabelMap(MutableMapping[Any, Any]):
    """A label dictionary stored as label to row key; reads return the ``Op`` records."""

    __slots__ = ("__weakref__", "_keys", "_ordered", "_owner")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, owner: OpRows, items: Any = (), *, ordered: bool = False) -> None:
        """Hold the records of ``items`` through ``owner``; ``ordered`` names the plain type."""

        self._owner = owner
        self._ordered = ordered
        pairs = list(items.items()) if isinstance(items, Mapping) else list(items)
        keys = owner.hold_many([record for _label, record in pairs])
        self._keys: dict[Any, int] = dict(
            zip((label for label, _record in pairs), keys, strict=True)
        )
        owner.register(self)

    def _records(self) -> list[Any]:
        """Return the records in label order."""

        return list(self._owner.records(self._keys.values()))

    def _rekey(self, keys: list[int]) -> None:
        """Replace the keys in place after the owner moved its records."""

        self._keys.update(zip(list(self._keys), keys, strict=True))

    def plain(self) -> dict[Any, Any]:
        """Return a plain dictionary (or ``OrderedDict``) from label to record."""

        pairs = zip(self._keys, self._records(), strict=True)
        return OrderedDict(pairs) if self._ordered else dict(pairs)

    def __getitem__(self, label: Any) -> Any:
        """Return the record registered under ``label``."""

        key = self._keys[label]
        tables = self._owner._tables
        return tables[0][key] if key >= 0 else tables[1][key]

    def __setitem__(self, label: Any, record: Any) -> None:
        """Register ``record`` under ``label``."""

        self._keys[label] = self._owner.hold(record)

    def __delitem__(self, label: Any) -> None:
        """Drop ``label``."""

        del self._keys[label]

    def __iter__(self) -> Iterator[Any]:
        """Iterate the labels in insertion order."""

        return iter(self._keys)

    def __reversed__(self) -> Iterator[Any]:
        """Iterate the labels in reverse insertion order."""

        return reversed(self._keys)

    def __len__(self) -> int:
        """Return the number of labels."""

        return len(self._keys)

    def __contains__(self, label: object) -> bool:
        """Return whether ``label`` is registered."""

        return label in self._keys

    def get(self, label: Any, default: Any = None) -> Any:
        """Return the record under ``label``, or ``default``."""

        key = self._keys.get(label)
        return default if key is None else self._owner.resolve(key)

    def keys(self) -> Any:
        """Return the labels as a dictionary key view."""

        return self._keys.keys()

    def values(self) -> Any:
        """Return the records in label order, as a list."""

        return self._records()

    def items(self) -> Any:
        """Return (label, record) pairs, as a list."""

        return list(zip(self._keys, self._records(), strict=True))

    def clear(self) -> None:
        """Drop every label."""

        self._keys.clear()

    def copy(self) -> dict[Any, Any]:
        """Return a plain dictionary from label to record."""

        return self.plain()

    def __eq__(self, other: object) -> bool:
        """Compare as a dictionary from label to record."""

        if isinstance(other, OpLabelMap):
            return self.plain() == other.plain()
        if isinstance(other, Mapping):
            return self.plain() == other
        return NotImplemented

    def __repr__(self) -> str:
        """Return the dictionary repr."""

        return repr(self.plain())

    def __copy__(self) -> OpLabelMap:
        """Return a new map over the same records."""

        return OpLabelMap(self._owner, self.items(), ordered=self._ordered)

    def __deepcopy__(self, memo: dict[Any, Any]) -> OpLabelMap:
        """Copy for a fork or a deep copy: the copied trace's records, same labels."""

        pairs = zip(self._keys, _copied(self._records(), memo), strict=True)
        new = OpLabelMap(self._owner.copy_for(memo), pairs, ordered=self._ordered)
        memo[id(self)] = new
        return new

    def __reduce_ex__(self, protocol: Any) -> Any:
        """Pickle as the plain dictionary of records."""

        return (OrderedDict if self._ordered else dict), (self.plain(),)


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
    if plain_list and isinstance(plain_list, list):
        state["_layer_list"] = OpSequence(owner, plain_list)
    for name in CONTAINER_FIELDS[1:]:
        plain = state.get(name)
        if plain and isinstance(plain, dict):
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
