"""Side-column families: derived fields whose raw values live beside the store.

A side-column family is a derived family whose raw material differs per row
and is observed before the family is installed (the replay templates of
``_postprocess/_replay_template_cells.py`` are the first). Each field has one
shared ``LazyCell`` marker, so an unread cell costs no object. The raw values
live in one side column per field, a list indexed by store row, and the
context every row's resolver shares is held once per store. Both are kept in
the family's table keyed weakly by the root ``OpRowStore``, so they live as
long as the store, which outlives its trace while a facade holds it; a fork
view reads through its root store, whose rows it shares. Row ids never change
for a store's life, which is what keeps a row-indexed side column valid.

A context may name the trace its resolver reads (the replay templates read
display keys from it), and a fork can outlive that trace. At fork time
``bind_fork_views`` gives each fork view its own side: the root's columns,
shared, with the context bound to the fork (``context.for_trace(fork)``).
The fork holds trace data, never a record of its parent.

A read of a marker answers ``resolver(raw, context, name)`` and caches it in
the cell, like any per-cell derived family. A state stream must not answer a
cell only to have its consumer drop it, so the op state stream yields a
``PendingCell`` instead (``pending_cell``). A consumer that keeps the value
answers it (``answer_pending``); pickling a pending cell carries the raw value
plus the context and answers on load. The shared markers still refuse to
pickle.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable
from typing import Any

from .cell_states import _MISSING, STATE_DERIVED, LazyCell

__all__ = (
    "SIDE_FAMILIES",
    "SIDE_FIELDS",
    "PendingCell",
    "SideColumnFamily",
    "answer_pending",
    "bind_fork_views",
    "pending_cell",
)

#: Every side-column family by name; the op state stream finds a marker's family here.
SIDE_FAMILIES: dict[str, SideColumnFamily] = {}
#: The stored field names some side-column family resolves.
SIDE_FIELDS: set[str] = set()


class _Side:
    """One store's side columns (field name to a list indexed by row) and their context."""

    __slots__ = ("columns", "context")

    def __init__(self, context: Any, columns: dict[str, list[Any]] | None = None) -> None:
        """Start with ``columns`` (another side's, shared), else with none."""

        self.context = context
        self.columns: dict[str, list[Any]] = {} if columns is None else columns


def _root(store: Any) -> Any:
    """Return the store whose rows ``store`` reads: a fork view's root store, else itself."""

    return getattr(store, "base", store)


class SideColumnFamily:
    """A derived family whose per-row raw values are held in side columns.

    Parameters
    ----------
    family:
        Family name.
    fields:
        The stored fields the family resolves.
    resolver:
        ``(raw, context, name) -> value``: the value today's pipeline would
        have stored for one row. It must be a module-level function, so a
        pickled pending cell can name it.
    """

    __slots__ = ("_sides", "family", "fields", "markers", "resolver")

    def __init__(self, family: str, fields: Iterable[str], resolver: Callable[..., Any]) -> None:
        """Register the family, its fields and one shared marker per field."""

        names = tuple(fields)
        if not names:
            raise ValueError(f"family {family!r} registers no fields")
        if family == "default" or family in SIDE_FAMILIES:
            raise ValueError(f"family {family!r} is already registered")
        self.family = family
        self.fields = names
        self.resolver = resolver
        self.markers = {
            name: LazyCell(STATE_DERIVED, name, family, self._answer, make=True) for name in names
        }
        self._sides: weakref.WeakKeyDictionary[Any, _Side] = weakref.WeakKeyDictionary()
        SIDE_FAMILIES[family] = self
        SIDE_FIELDS.update(names)

    def bind(self, store: Any, context: Any) -> _Side:
        """Return the side columns of ``store`` for ``context``, creating them once.

        Binding a new context starts empty columns. Raises ``TypeError`` for
        a store that cannot be weakly referenced (a detached single-row store):
        its cells are answered at once instead.
        """

        root = _root(store)
        side = self._sides.get(root)
        if side is None or side.context is not context:
            side = self._sides[root] = _Side(context)
        return side

    def hold(self, side: _Side, name: str, row: int, raw: Any) -> LazyCell:
        """Keep ``raw`` for ``(row, name)`` and return the marker its cell stores."""

        column = side.columns.get(name)
        if column is None:
            column = side.columns[name] = []
        if row >= len(column):
            column.extend([_MISSING] * (row + 1 - len(column)))
        column[row] = raw
        return self.markers[name]

    def hold_column(self, side: _Side, name: str, by_row: list[Any]) -> LazyCell:
        """Adopt ``by_row`` (raw values indexed by row) as the side column of ``name``."""

        side.columns[name] = by_row
        return self.markers[name]

    def material(self, store: Any, row: int, name: str) -> tuple[Any, Any]:
        """Return ``(raw value, context)`` of one row, raising when the store holds none."""

        sides = self._sides
        try:
            side = sides.get(store)  # a fork view's own side, else its root's
            if side is None:
                side = sides.get(_root(store))
        except TypeError:  # a store without weak references never holds side columns
            side = None
        if side is None:
            raise LookupError(f"family {self.family!r}: the store holds no side columns")
        column = side.columns.get(name, ())
        raw = column[row] if row < len(column) else _MISSING
        if raw is _MISSING:
            raise LookupError(f"family {self.family!r}: no raw {name!r} for row {row}")
        return raw, side.context

    def raw(self, store: Any, row: int, name: str) -> Any:
        """Return the raw value an unread cell would resolve from, else ``_MISSING``."""

        cell = store.cell_get(row, store.layout.fid_by_name[name], True)
        if cell is not self.markers[name]:
            return _MISSING
        return self.material(store, row, name)[0]

    def _answer(self, store: Any, row: int, name: str) -> Any:
        """The markers' resolver: answer one row from its raw value and the context."""

        raw, context = self.material(store, row, name)
        return self.resolver(raw, context, name)


def bind_fork_views(views: Iterable[Any], trace: Any) -> None:
    """Give each fork view whose root holds side columns its own context.

    ``views`` are a new fork's store views and ``trace`` the fork. The view's
    side shares its root's columns; its context is the root's, bound to
    ``trace`` by ``context.for_trace``, so a first read after the parent trace
    is gone answers as it would with the parent alive.
    """

    families = tuple(SIDE_FAMILIES.values())
    for view in views:
        root = view.base
        for family in families:
            side = family._sides.get(root)
            if side is not None:
                family._sides[view] = _Side(side.context.for_trace(trace), side.columns)


class PendingCell:
    """An unread side-column cell as a state stream yields it.

    ``cell`` is the field's marker and ``(store, row)`` the cell's place.
    Nothing is answered until a consumer keeps the value.
    """

    __slots__ = ("cell", "row", "store")

    cell: LazyCell
    row: int
    store: Any

    def __repr__(self) -> str:
        """Return a stable representation (never an address)."""

        return f"PendingCell({self.cell.family}.{self.cell.name}, row={self.row})"

    #: Unhashable, like the containers a template resolves to.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """Compare by answer, so two states that would hold equal values compare equal."""

        theirs = other.answer() if other.__class__ is PendingCell else other
        result: bool = self.answer() == theirs
        return result

    def answer(self) -> Any:
        """Return the cell's value without writing it back."""

        cell = self.cell
        return SIDE_FAMILIES[cell.family]._answer(self.store, self.row, cell.name)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Carry the raw value and the context; the load answers the cell."""

        cell = self.cell
        family = SIDE_FAMILIES[cell.family]
        raw, context = family.material(self.store, self.row, cell.name)
        return _answer_material, (family.resolver, raw, context, cell.name)


def _answer_material(resolver: Callable[..., Any], raw: Any, context: Any, name: str) -> Any:
    """Answer a pickled pending cell from its raw value and context."""

    return resolver(raw, context, name)


def pending_cell(store: Any, row: int, cell: Any) -> Any:
    """Return a ``PendingCell`` for a side-column marker at ``(store, row)``, else ``_MISSING``."""

    if cell.__class__ is not LazyCell or cell.family not in SIDE_FAMILIES:
        return _MISSING
    pending = PendingCell()
    pending.cell = cell
    pending.store = store
    pending.row = row
    return pending


def answer_pending(value: Any) -> Any:
    """Return a pending cell's answer, or ``value`` itself for any other value."""

    return value.answer() if value.__class__ is PendingCell else value
