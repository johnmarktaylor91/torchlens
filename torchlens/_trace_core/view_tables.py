"""The per-trace view tables: where ``Op`` and ``Layer`` records are cached, weakly.

An ``Op`` is a view over one row of the trace's op store and a ``Layer`` is a
view over one trace-held state dictionary, so either can be built again from
what the trace owns. The trace therefore keeps no record: each table here maps
a row to a weak reference to the view handed out for it. Identity holds while
anyone keeps the view (``trace.ops[0] is trace.ops[0]`` inside one expression,
or for as long as a variable holds it); once every reference is dropped, the
next lookup builds a fresh view.

Every view handed out holds its trace in its ``_owner`` slot, which is what
makes ``tl.trace(model, x)["relu_1"].out`` work: the record keeps the trace,
the trace keeps the rows. The reference runs one way only (the table holds the
view weakly), so a trace nobody holds is still freed by reference counting.
``Trace.cleanup()`` empties the rows, so a record held past it keeps only the
emptied trace object.

While a trace is being built the tables are *pinned*: they hold the records
strongly and are plain lists as far as the build passes can tell, so a pass
that reads the same op a hundred times pays for no rebuilds. ``release`` ends
the build: it turns each table weak in place (the list objects other
containers already hold stay the same objects) and gives the surviving views
their owner. A trace that is restored, copied or forked starts weak.

Neither table refers back to the object that holds it, and the core and the
owning trace are named weakly, so the tables add no reference cycle.

The same object also carries the two strong side tables that remain: records
with no row in the store (``detached``), and the containers built once per
module or buffer cell (``cells``), which must keep in-place writes.
"""

from __future__ import annotations

import weakref
from collections import deque
from collections.abc import Iterable
from itertools import repeat
from typing import Any

_ref = weakref.ref
_deref = weakref.ref.__call__
#: Calls a weak reference (for ``map``): the referent, or ``None`` once it is gone.
DEREF = _deref
_new = object.__new__
_set = object.__setattr__
_list_get = list.__getitem__
_list_set = list.__setitem__
_list_extend = list.extend

OP_KIND = "op"
LAYER_KIND = "layer"

#: The ``Op`` class, bound at import (``bind_op_facade``) so this module imports no record class.
OP_CLASS: list[type] = []


class _Gone:
    """The referent of the shared dead reference."""

    __slots__ = ("__weakref__",)


def _dead_reference() -> Any:
    """Return a weak reference whose referent is already gone."""

    gone = _Gone()
    reference = _ref(gone)
    del gone
    return reference


#: What a weak table holds for a row with no live view: calling it gives ``None``.
DEAD: Any = _dead_reference()


class _Views(list[Any]):
    """Common layout of the pinned and weak tables, so one can become the other in place.

    ``core_ref`` and ``owner_ref`` are weak references (``owner_ref`` is
    ``None`` until a trace is named), ``kind`` is ``"op"`` or ``"layer"``.
    ``refs`` is ``None`` for a pinned table. A weak table keeps its weak
    references there, one per row, in a plain list: a reader on a hot path
    indexes that list and calls the reference without entering a method.
    """

    __slots__ = ("core_ref", "kind", "owner_ref", "refs")

    core_ref: Any
    kind: str
    owner_ref: Any
    refs: Any


class PinnedViews(_Views):
    """Views by row, held strongly while the trace is built (``None`` where none was cached).

    It overrides nothing a build pass uses, so indexing, assignment, ``append``
    and ``extend`` run at list speed.
    """

    __slots__ = ()

    def pad(self, size: int) -> None:
        """Grow to ``size`` rows."""

        if len(self) < size:
            _list_extend(self, repeat(None, size - len(self)))


class WeakViews(_Views):
    """Views by row, held weakly: a read returns the live view or builds one.

    The weak references are in ``refs`` (``DEAD`` where there is no live
    view); the list's own items are ``None`` and only give the table its
    length. An ``Op`` table builds a missing view from its row; a ``Layer``
    table cannot (the layer store builds from its state) and returns ``None``.
    """

    __slots__ = ()

    def __getitem__(self, row: Any) -> Any:
        """Return the view at ``row``, building an op view that is not alive."""

        view = self.refs[row]()
        if view is None and self.kind == OP_KIND:
            return self.build((row,))[0]
        return view

    def __setitem__(self, row: Any, view: Any) -> None:
        """Cache ``view`` at ``row`` (``None`` clears the row) and give it its owner."""

        if view is None:
            self.refs[row] = DEAD
            return
        self.refs[row] = _ref(view)
        reference = self.owner_ref
        owner = reference() if reference is not None else None
        if owner is not None:
            _set(view, "_owner", owner)

    def __iter__(self) -> Any:
        """Iterate the live views (``None`` where a row has none)."""

        return iter(self.live())

    def live(self) -> list[Any]:
        """Return the live view of every row (``None`` where a row has none); never builds."""

        return list(map(_deref, self.refs))

    def append(self, view: Any) -> None:
        """Add one row holding ``view``."""

        _list_extend(self, (None,))
        self.refs.append(DEAD)
        if view is not None:
            self[len(self) - 1] = view

    def extend(self, views: Iterable[Any]) -> None:
        """Add one row per item of ``views``."""

        for view in views:
            self.append(view)

    def pad(self, size: int) -> None:
        """Grow to ``size`` rows."""

        if len(self) < size:
            self.refs.extend(repeat(DEAD, size - len(self)))
            _list_extend(self, repeat(None, size - len(self)))

    def peek(self, row: int) -> Any:
        """Return the live view at ``row``, or ``None``; never builds."""

        return self.refs[row]() if 0 <= row < len(self) else None

    def claim(self, row: int, view: Any) -> Any:
        """Cache ``view`` at ``row`` unless another view is alive there; return the one held."""

        if row >= len(self):
            self.pad(row + 1)
        current = self.refs[row]()
        if current is not None:
            return current
        self[row] = view
        return view

    def many(self, rows: Iterable[int]) -> list[Any]:
        """Return the views at ``rows`` in order, building the op views that are not alive.

        The loop over rows runs in C: a reader of every op of a trace pays a
        fixed number of calls, not one per op.
        """

        rows = list(rows)
        found = list(map(_deref, map(self.refs.__getitem__, rows)))
        if self.kind != OP_KIND or None not in found:
            return found
        missing = list(dict.fromkeys([row for row, view in zip(rows, found) if view is None]))
        built = dict(zip(missing, self.build(missing)))
        return [built[row] if view is None else view for row, view in zip(rows, found)]

    def build(self, rows: Any) -> list[Any]:
        """Build an ``Op`` view for each of ``rows`` (no repeats), cache them weakly, return them."""

        core = self.core_ref()
        store = core.ops if core is not None else None
        views: list[Any] = list(map(_new, repeat(OP_CLASS[0], len(rows))))
        deque(map(_set, views, repeat("_core"), repeat(store)), 0)
        deque(map(_set, views, repeat("_row"), rows), 0)
        self.adopt(rows, views, None)
        return views

    def adopt(self, rows: Any, views: list[Any], unowned: Any = DEAD) -> None:
        """Cache ``views`` at ``rows`` (replacing what is there) and give each its owner.

        With no owner named yet, each view's owner slot is set to ``unowned``
        when one is given (a new op view has an unset slot) and left alone
        otherwise.
        """

        reference = self.owner_ref
        owner = reference() if reference is not None else None
        if owner is None:
            owner = unowned
        if owner is not DEAD:
            deque(map(_set, views, repeat("_owner"), repeat(owner)), 0)
        deque(map(_list_set, repeat(self.refs), rows, map(_ref, views)), 0)


class ViewTables:
    """The view tables of one trace core."""

    __slots__ = ("cells", "detached", "layers", "ops", "pinned")

    def __init__(self, core: Any, *, pinned: bool = False) -> None:
        """Create empty tables for ``core``; ``pinned`` starts them strong for a build."""

        self.pinned = pinned
        self.ops = self._table(core, OP_KIND)
        self.layers = self._table(core, LAYER_KIND)
        # Records with no row in the store, by negative key (held strongly).
        self.detached: dict[int, Any] = {}
        # Containers built once per module or buffer cell, and any other
        # per-trace object that must persist (held strongly: a dropped entry
        # would lose an in-place write).
        self.cells: dict[Any, Any] = {}

    def _table(self, core: Any, kind: str) -> Any:
        """Return an empty table of ``kind`` for ``core``."""

        table: Any = PinnedViews() if self.pinned else WeakViews()
        table.core_ref = _ref(core)
        table.kind = kind
        table.owner_ref = None
        table.refs = None if self.pinned else []
        return table

    def owner(self) -> Any:
        """Return the trace that views of these tables hold, or ``None``."""

        reference = self.ops.owner_ref
        return reference() if reference is not None else None

    def set_owner(self, trace: Any) -> None:
        """Name ``trace`` as the owner (the first live one wins); live views are given it."""

        if trace is None or self.owner() is not None:
            return
        self.ops.owner_ref = self.layers.owner_ref = _ref(trace)
        if not self.pinned:
            for table in (self.ops, self.layers):
                for view in table.live():
                    if view is not None:
                        _set(view, "_owner", trace)

    def release(self, trace: Any = None) -> None:
        """End the build: make the tables weak in place and give survivors their owner.

        A record somebody still holds (a hook kept it) stays the view of its
        row; every other record is dropped here and built again on lookup.
        """

        if not self.pinned:
            self.set_owner(trace)
            return
        if trace is not None and self.owner() is None:
            self.ops.owner_ref = self.layers.owner_ref = _ref(trace)
        self.pinned = False
        owner = self.owner()
        for table in (self.ops, self.layers):
            rows = [row for row, view in enumerate(list.__iter__(table)) if view is not None]
            references = [_ref(view) for view in list.__iter__(table) if view is not None]
            table.__class__ = WeakViews
            table.refs = [DEAD] * len(table)
            _list_set(table, slice(None), [None] * len(table))
            # Nothing above still holds a view: a reference that is alive now
            # is to a record somebody else kept.
            for row, reference, view in zip(rows, references, list(map(_deref, references))):
                if view is not None:
                    table.refs[row] = reference
                    _set(view, "_owner", owner)


def tables_of(trace: Any) -> ViewTables | None:
    """Return the view tables of ``trace``'s core, or ``None`` when it has no core."""

    core = getattr(trace, "__dict__", {}).get("_trace_core")
    return getattr(core, "_views", None)


def release_views(trace: Any) -> None:
    """End the build of ``trace``: its records are handed out weakly from here on."""

    tables = tables_of(trace)
    if tables is not None:
        tables.release(trace)
