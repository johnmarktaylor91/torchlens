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

#: The ``Param`` class, bound on first use so this module imports no record class.
_PARAM: list[type] = []


def _param_class() -> type:
    """Return the ``Param`` class."""

    if not _PARAM:
        from .param import Param

        _PARAM.append(Param)
    return _PARAM[0]


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
        """Grow to ``size`` rows."""

        grow = size - len(self.refs)
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
        view = _new(_param_class())
        view.__dict__ = state
        _set(view, "_owner", self.owner())
        self.refs[row] = _ref(view)
        return view

    def claim(self, row: int, record: Any) -> Any:
        """Make ``record`` the view of ``row`` unless another view is alive there; return the view."""

        if row >= len(self.refs):
            self._pad(row + 1)
        current = self.refs[row]()
        if current is not None:
            return current
        self.states[row] = record.__dict__
        self.refs[row] = _ref(record)
        _set(record, "_owner", self.owner())
        return record


class ParamRows:
    """Resolve param row keys to records for one trace's containers."""

    __slots__ = ("__weakref__", "_copies", "_ids", "_last", "_tables")

    def __init__(self, table: ParamTable) -> None:
        """Resolve through ``table``; records with no row in it are held here."""

        self._tables: tuple[ParamTable, dict[int, Any]] = (table, {})
        self._ids: dict[int, int] = {}
        # id(parent sequence) -> (parent sequence, this fork's copy of it).
        self._copies: dict[int, tuple[Any, Any]] = {}
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
        state = getattr(record, "__dict__", None)
        if state is not None and state.get(CORE_KEY) is table.store:
            row: int = state[ROW_KEY]
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
            copy = ParamSequence(self)
            list.extend(copy, self.adopt_keys(parent))
            entry = self._copies[id(parent)] = (parent, copy)
        return entry[1]

    def resolve(self, key: int) -> Any:
        """Return the record behind one key."""

        return self._tables[0][key] if key >= 0 else self._tables[1][key]

    def records(self, keys: Iterable[int]) -> Iterator[Any]:
        """Return an iterator over the records behind ``keys``."""

        return map(self.resolve, keys)

    def adopt_keys(self, container: Any) -> list[int]:
        """Return this resolver's keys for the entries of another resolver's container.

        A fork's param store is a view over its parent's with the same rows,
        so a row key means the same parameter on both sides; a detached record
        is held again here.
        """

        other = container._owner
        keys = list(
            _list_iter(container) if isinstance(container, list) else dict.values(container)
        )
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

    def __deepcopy__(self, memo: dict[Any, Any]) -> list[Any]:
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

    def __deepcopy__(self, memo: dict[Any, Any]) -> dict[Any, Any]:
        """Copy as a plain dictionary of the copied records (a copied trace keys them again)."""

        plain = self.plain()
        new = type(plain)(zip(plain, _copied(plain.values(), memo), strict=True))
        memo[id(self)] = new
        return new


#: The row-key containers, for exact class checks.
ROW_FORMS = (ParamSequence, ParamMap)


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
    """Return ``value`` (a list of records) as a ``ParamSequence`` of ``rows``.

    ``memo`` maps each plain list already converted to its sequence, so one
    list object reached from two places stays one object. An empty or
    non-list value is returned unchanged.
    """

    cls = value.__class__
    if cls is ParamSequence:
        if value._owner is rows:
            return value
        known = memo.get(id(value))
        if known is None:
            known = memo[id(value)] = ParamSequence(rows)
            list.extend(known, rows.adopt_keys(value))
        return known
    if cls is not list or not value:
        return value
    known = memo.get(id(value))
    if known is None:
        known = memo[id(value)] = ParamSequence(rows, value)
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
