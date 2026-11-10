"""Trace-side ``Param`` containers that hold row keys, and the trace's ``Param`` views.

A ``Param`` is a row facade: its declared fields live in one row of the trace
core's param store, and its ``__dict__`` keeps only the store binding and any
extra attribute. So a ``Param`` can be built again from what the trace owns,
and the trace keeps no ``Param``. Every trace container that used to hold
``Param`` records (the ``trace.params`` accessor, the per-module lists, each
op's and each layer's parameter list, the accessor a module builds for its
parameters) holds one integer key per entry instead and hands the record back
on read, exactly as the op containers do (``_op_row_containers.py``).

The ``ParamTable`` maps each row to a weak reference to the view handed out
for it, and keeps the view's ``__dict__`` (the row binding plus extras), so a
view built again carries the extras the previous one had. Identity holds while
anyone keeps the view; once every reference is dropped the next read builds a
new one. Every view handed out holds its trace in its ``_owner`` slot, so
``tl.trace(model, x).params["fc.weight"].module`` reads a live trace. The
table is kept in the trace core's strong cell table (``ViewTables.cells``);
neither it nor the containers refer to the trace except weakly.

A ``Param`` with no row in the trace's param store gets a negative key and is
held strongly by the resolver, as before.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator
from typing import Any

from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from .._trace_core.view_tables import DEAD
from ._op_row_containers import OpLabelMap, OpSequence, _copied

#: The kind name of the param store in ``TraceCore.kind_rows``.
PARAM_KIND = "param"
#: The key of a trace's ``ParamRows`` in ``ViewTables.cells``.
TABLE_KEY = ("param", "rows")

_new = object.__new__
_set = object.__setattr__
_ref = weakref.ref
_list_iter = list.__iter__
_tuple_iter = tuple.__iter__
_tuple_get = tuple.__getitem__

#: The ``Param`` class, bound by ``param.py`` at import so this module imports no record class.
_PARAM: list[type] = []


class ParamTable:
    """The ``Param`` views of one trace, by row of its param store, held weakly."""

    __slots__ = ("__weakref__", "handed", "owner_ref", "refs", "states", "store")

    def __init__(self, store: Any) -> None:
        """Create an empty table over ``store``."""

        self.store = store
        # Per row: the ``__dict__`` the row's views share (``None`` until known).
        self.states: list[Any] = []
        # Per row: a weak reference to the live view, ``DEAD`` where none is.
        self.refs: list[Any] = []
        self.owner_ref: Any = None
        # A weak reference to the accessor ``trace.params`` handed out last.
        self.handed: Any = None

    def owner(self) -> Any:
        """Return the trace views of this table hold, or ``None``."""

        reference = self.owner_ref
        return reference() if reference is not None else None

    def set_owner(self, trace: Any) -> None:
        """Name ``trace`` as the owner unless a live one is named; give live views their owner."""

        if self.owner() is not None:
            return
        self.owner_ref = _ref(trace)
        for reference in self.refs:
            view = reference()
            if view is not None:
                _set(view, "_owner", trace)

    def _pad(self, size: int) -> None:
        """Grow to ``size`` rows, and at least to the store's row count (one growth per table)."""

        grow = max(size, len(self.store)) - len(self.refs)
        if grow > 0:
            self.states.extend([None] * grow)
            self.refs.extend([DEAD] * grow)

    def __getitem__(self, row: int) -> Any:
        """Return the view at ``row``, building one when none is alive."""

        try:
            view = self.refs[row]()
        except IndexError:
            self._pad(row + 1)
            view = None
        return self._build(row) if view is None else view

    def peek(self, row: int) -> Any:
        """Return the live view at ``row``, or ``None``; never builds."""

        return self.refs[row]() if 0 <= row < len(self.refs) else None

    def _build(self, row: int) -> Any:
        """Build, cache and return a new view of ``row``."""

        state = self.states[row]
        if state is None:
            state = self.states[row] = {CORE_KEY: self.store, ROW_KEY: row}
        view: Any = _new(_PARAM[0])
        view.__dict__ = state
        reference = self.owner_ref
        _set(view, "_owner", reference() if reference is not None else None)
        self.refs[row] = _ref(view)
        return view

    def claim(self, row: int, record: Any) -> Any:
        """Make ``record`` the view of ``row`` unless another view is alive there; return the view."""

        try:
            current = self.refs[row]()
        except IndexError:
            self._pad(row + 1)
            current = None
        if current is not None:
            return current
        self.states[row] = record.__dict__
        self.refs[row] = _ref(record)
        reference = self.owner_ref
        _set(record, "_owner", reference() if reference is not None else None)
        return record


class ParamRows:
    """Resolve param row keys to records for one trace's containers."""

    __slots__ = ("__weakref__", "_copies", "_ids", "_last", "_positions", "_tables")

    def __init__(self, table: ParamTable) -> None:
        """Resolve through ``table``; records with no row in it are held here."""

        self._tables: tuple[ParamTable, dict[int, Any]] = (table, {})
        self._ids: dict[int, int] = {}
        # id(parent sequence) -> (parent sequence, this fork's copy of it).
        self._copies: dict[int, tuple[Any, Any]] = {}
        # id(sequence) -> (sequence, key -> first position), for ``Param.index``.
        self._positions: dict[int, tuple[Any, dict[int, int]]] = {}
        # Never a row: the containers' one-step reuse of the last key is off.
        self._last = -1

    @property
    def table(self) -> ParamTable:
        """Return the view table."""

        return self._tables[0]

    def register(self, container: Any) -> None:
        """Accept a container (nothing to move later: keys never change)."""

    def hold(self, record: Any) -> int:
        """Return the key of ``record``, caching it as its row's view or holding it detached."""

        table, detached = self._tables
        try:
            state = record.__dict__
            on_store = state[CORE_KEY] is table.store
        except (AttributeError, KeyError, TypeError):
            on_store = False
        if on_store:
            row: int = state[ROW_KEY]
            try:
                current = table.refs[row]()
            except IndexError:
                current = None
            if current is not record:
                table.claim(row, record)
            return row
        key = self._ids.get(id(record))
        if key is None or detached.get(key) is not record:
            key = -1 - len(detached)
            while key in detached:
                key -= 1
            detached[key] = record
            self._ids[id(record)] = key
        return key

    def hold_many(self, records: Iterable[Any]) -> list[int]:
        """Return the keys of ``records`` in order."""

        hold = self.hold
        return [hold(record) for record in records]

    def key_of(self, record: Any) -> int | None:
        """Return the key ``record`` has here, or ``None``; never holds it."""

        table, detached = self._tables
        state = getattr(record, "__dict__", None)
        if state is not None and state.get(CORE_KEY) is table.store:
            return state[ROW_KEY]
        key = self._ids.get(id(record))
        return key if key is not None and detached.get(key) is record else None

    def forked(self, parent: Any) -> Any:
        """Return this fork's copy of a parent ``ParamSequence`` (one copy per parent sequence)."""

        entry = self._copies.get(id(parent))
        if entry is None or entry[0] is not parent:
            entry = self._copies[id(parent)] = (parent, self.rekeyed(parent))
        return entry[1]

    def rekeyed(self, container: Any) -> Any:
        """Return this resolver's copy of another resolver's ``ParamSequence`` or ``ParamTuple``."""

        keys = self.adopt_keys(container)
        if container.__class__ is ParamTuple:
            return ParamTuple.from_keys(self, keys)
        owner: Any = self  # ParamSequence types its owner as the OpRows it extends
        sequence = ParamSequence(owner)
        list.extend(sequence, keys)
        return sequence

    def position(self, sequence: Any, key: Any) -> int | None:
        """Return the first position of ``key`` in one of this resolver's sequences, or ``None``."""

        entry = self._positions.get(id(sequence))
        for fresh in (False, True):
            if fresh or entry is None or entry[0] is not sequence:
                table: dict[int, int] = {}
                for position, held in enumerate(_list_iter(sequence)):
                    table.setdefault(held, position)
                entry = self._positions[id(sequence)] = (sequence, table)
            index = entry[1].get(key)
            if (
                index is not None
                and index < len(sequence)
                and list.__getitem__(sequence, index) == key
            ):
                return index
        return None

    def resolve(self, key: int) -> Any:
        """Return the record behind one key."""

        if key < 0:
            return self._tables[1][key]
        table = self._tables[0]
        try:
            view = table.refs[key]()
        except IndexError:
            view = None
        return table[key] if view is None else view

    def records(self, keys: Iterable[int]) -> Iterator[Any]:
        """Return an iterator over the records behind ``keys``.

        Live views are read in place (no call per key); a missing one is built.
        """

        table, detached = self._tables
        refs = table.refs
        size = len(refs)
        return iter(
            [
                detached[key]
                if key < 0
                else (view if key < size and (view := refs[key]()) is not None else table[key])
                for key in keys
            ]
        )

    def adopt_keys(self, container: Any) -> list[int]:
        """Return this resolver's keys for the entries of another resolver's container.

        A fork's param store is a view over its parent's with the same rows,
        so a row key means the same parameter on both sides; a detached record
        is held again here.
        """

        other = container._owner
        if isinstance(container, dict):
            keys = list(dict.values(container))
        elif isinstance(container, tuple):
            keys = list(_tuple_iter(container))
        else:
            keys = list(_list_iter(container))
        return [key if key >= 0 else self.hold(other.resolve(key)) for key in keys]


class ParamSequence(OpSequence):
    """A list of ``Param`` records stored as row keys; reads return the records."""

    __slots__ = ()

    def __tl_plain__(self) -> list[Any]:
        """Return the plain list of records this sequence stands in for."""

        return self._records()

    def __copy__(self) -> ParamSequence:
        """Return a new sequence over the same records."""

        return ParamSequence(self._owner, self._records())

    def __deepcopy__(self, memo: dict[Any, Any]) -> list[Any]:  # type: ignore[override]
        """Copy as a plain list of the copied records (a copied trace keys them again)."""

        new = _copied(self._records(), memo)
        memo[id(self)] = new
        return new


class ParamMap(OpLabelMap):
    """A ``key -> Param`` dictionary stored as row keys; reads return the records."""

    __slots__ = ()

    def __init__(self, owner: Any, items: Any = (), *, ordered: bool = False) -> None:
        """Hold the records of ``items``; the plain form is a ``dict``."""

        super().__init__(owner, items, ordered=ordered)
        self._ordered = ordered

    def __tl_plain__(self) -> dict[Any, Any]:
        """Return the plain dictionary of records this map stands in for."""

        return self.plain()

    def __copy__(self) -> ParamMap:
        """Return a new map over the same records."""

        return ParamMap(self._owner, self.items(), ordered=self._ordered)

    def __deepcopy__(self, memo: dict[Any, Any]) -> dict[Any, Any]:  # type: ignore[override]
        """Copy as a plain dictionary of the copied records (a copied trace keys them again)."""

        plain = self.plain()
        new = type(plain)(zip(plain, _copied(plain.values(), memo), strict=True))
        memo[id(self)] = new
        return new


class ParamTuple(tuple):  # noqa: SLOT001 (a tuple subclass cannot declare slots)
    """A tuple of ``Param`` records stored as row keys; reads return the records.

    An op's parameter list is a tuple once its store is sealed. The tuple's own
    items are the keys; every method that hands out or compares items speaks
    records, and the plain form is a ``tuple``.
    """

    _owner: ParamRows

    def __new__(cls, owner: ParamRows, records: Iterable[Any] = ()) -> ParamTuple:
        """Hold ``records`` through ``owner``."""

        return cls.from_keys(owner, owner.hold_many(records))

    @classmethod
    def from_keys(cls, owner: ParamRows, keys: Iterable[int]) -> ParamTuple:
        """Return a tuple over keys ``owner`` already holds."""

        new = tuple.__new__(cls, keys)
        new._owner = owner
        return new

    def _records(self) -> tuple[Any, ...]:
        """Return the records as a plain tuple."""

        return tuple(self._owner.records(_tuple_iter(self)))

    def __tl_plain__(self) -> tuple[Any, ...]:
        """Return the plain tuple of records this tuple stands in for."""

        return self._records()

    def __iter__(self) -> Iterator[Any]:
        """Iterate the records in order."""

        return self._owner.records(_tuple_iter(self))

    def __reversed__(self) -> Iterator[Any]:
        """Iterate the records in reverse order."""

        return self._owner.records(reversed(_tuple_get(self, slice(None))))

    def __getitem__(self, index: Any) -> Any:
        """Return one record, or a plain tuple of records for a slice."""

        key = _tuple_get(self, index)
        if index.__class__ is slice:
            return tuple(self._owner.records(key))
        return self._owner.resolve(key)

    def __contains__(self, value: object) -> bool:
        """Return whether ``value`` is one of the records."""

        return value in self._records()

    def index(self, value: Any, *bounds: Any) -> int:
        """Return the first position of ``value``."""

        return self._records().index(value, *bounds)

    def count(self, value: Any) -> int:
        """Return how many times ``value`` occurs."""

        return self._records().count(value)

    def __eq__(self, other: object) -> bool:
        """Compare as a tuple of records."""

        return self._records() == other if isinstance(other, tuple) else NotImplemented

    def __ne__(self, other: object) -> bool:
        """Compare as a tuple of records."""

        return self._records() != other if isinstance(other, tuple) else NotImplemented

    def __lt__(self, other: Any) -> bool:
        """Order as a tuple of records."""

        return self._records() < other

    def __le__(self, other: Any) -> bool:
        """Order as a tuple of records."""

        return self._records() <= other

    def __gt__(self, other: Any) -> bool:
        """Order as a tuple of records."""

        return self._records() > other

    def __ge__(self, other: Any) -> bool:
        """Order as a tuple of records."""

        return self._records() >= other

    def __hash__(self) -> int:
        """Hash as the tuple of records."""

        return hash(self._records())

    def __add__(self, other: Any) -> tuple[Any, ...]:
        """Concatenate as a plain tuple of records."""

        return self._records() + other

    def __radd__(self, other: Any) -> tuple[Any, ...]:
        """Concatenate as a plain tuple of records."""

        return other + self._records()

    def __mul__(self, count: Any) -> tuple[Any, ...]:
        """Repeat as a plain tuple of records."""

        return self._records() * count

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Return the tuple repr of the records."""

        return repr(self._records())

    def __copy__(self) -> ParamTuple:
        """Return this tuple (immutable, as ``copy.copy`` of a tuple does)."""

        return self

    def __deepcopy__(self, memo: dict[Any, Any]) -> tuple[Any, ...]:
        """Copy as a plain tuple of the copied records (a copied trace keys them again)."""

        new = tuple(_copied(self._records(), memo))
        memo[id(self)] = new
        return new

    def __reduce_ex__(self, protocol: Any) -> Any:
        """Pickle as the plain tuple of records."""

        return tuple, (self._records(),)


#: The row-key containers, for exact class checks.
ROW_FORMS = (ParamSequence, ParamMap, ParamTuple)


def param_rows_of(core: Any) -> ParamRows | None:
    """Return the ``ParamRows`` of ``core``, creating it when the core has a param store."""

    views = getattr(core, "_views", None)
    if views is None:
        return None
    rows = views.cells.get(TABLE_KEY)
    if rows is None:
        store = core.kind_rows.get(PARAM_KIND)
        if store is None:
            return None
        rows = views.cells[TABLE_KEY] = ParamRows(ParamTable(store))
    return rows


def as_sequence(rows: ParamRows, value: Any, memo: dict[int, Any]) -> Any:
    """Return ``value`` (a list or tuple of records) in the row-key form of ``rows``.

    ``memo`` maps each plain list already converted to its sequence, so one
    list object reached from two places stays one object. An empty value,
    or one that is not a list or tuple, is returned unchanged.
    """

    cls = value.__class__
    if cls is ParamSequence or cls is ParamTuple:
        if value._owner is rows:
            return value
        known = memo.get(id(value))
        if known is None:
            known = memo[id(value)] = rows.rekeyed(value)
        return known
    if (cls is not list and cls is not tuple) or not value:
        return value
    known = memo.get(id(value))
    if known is None:
        owner: Any = rows  # ParamSequence types its owner as the OpRows it extends
        form = ParamSequence if cls is list else ParamTuple
        known = memo[id(value)] = form(owner, value)
    return known


def as_map(rows: ParamRows, value: Any) -> Any:
    """Return ``value`` (a ``key -> record`` dict) as a ``ParamMap`` of ``rows``."""

    cls = value.__class__
    if cls is ParamMap:
        if value._owner is rows:
            return value
        new = ParamMap(rows)
        dict.update(new, zip(list(dict.__iter__(value)), rows.adopt_keys(value), strict=True))
        return new
    if cls is not dict or not value:
        return value
    return ParamMap(rows, value)
