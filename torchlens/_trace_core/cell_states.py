"""Cell states of the row stores: stored, default-omitted, derived and deleted.

A cell of an ``OpRowStore`` (or of its detached and fork-view twins) is in
exactly one of four states, and the store never conflates them:

* set: the cell holds a value (or one of the storage encodings below,
  which the facade descriptors decode);
* default-omitted: the cell holds the field's ``LazyCell`` in state
  ``"default"`` and a read answers the registered default. An immutable
  default is answered without a write; a mutable default is created for the
  reading row and written back, so per-row mutation stays isolated;
* derived: the cell holds the field's ``LazyCell`` in state ``"derived"``
  and a read answers the registered family resolver, either for the one
  cell or, for a whole-column family, by one batch pass that fills every
  row of the column still in this state;
* deleted (or never set): the cell holds ``_MISSING``. A read reports
  ``_MISSING`` and a delete reports "was not set". Deleting a cell in any
  other state moves it here without answering it, and nothing moves it
  back except an explicit write: a deleted cell never answers a default
  and never runs a resolver.

``LazyFields`` is the per-layout registry: it hands out one shared marker
per registered field and builds the seed row that starts default-capable
fields default-omitted and derived fields derived. Markers carry their own
answer, so the store read path needs no lookup: one exact-class test.

The exhaustive torch capture registers its constant-valued op cells as
defaults (``backends/torch/_ops_row_cells.py``): a draft row starts them
default-omitted instead of writing each value, and ``settle_default_cells``
writes the remaining ones at the freeze seam, so a sealed store holds the
values every reader of raw cells expects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, NoReturn

#: Unset-cell sentinel; distinct from None (a real storable value).
_MISSING = object()

#: CSR-backed relation cell sentinel: the value lives in the store's
#: dataflow edge table; the facade descriptor rematerializes an immutable
#: tuple view on first access and caches it back through ``cell_set``.
#: Distinct from ``_MISSING`` so present/absent semantics stay exact —
#: ``cell_del`` on a ``_CSR`` cell clears to genuinely absent.
_CSR = object()

#: Shared-fact cell sentinel: the value lives in the store's fact
#: blocks (the FunctionCall / ParamAlias group tables); the facade
#: descriptor hydrates the exact public container type for the row on first
#: access and caches it back. Same present/absent semantics as ``_CSR``.
_FACT = object()

#: The four states ``cell_state`` reports.
STATE_SET = "set"
STATE_DEFAULT = "default"
STATE_DERIVED = "derived"
STATE_DELETED = "deleted"

#: Exact types a default may share across rows without a factory.
_IMMUTABLE_ATOMS = frozenset({type(None), bool, int, float, complex, str, bytes})


def _is_immutable_default(value: Any) -> bool:
    """Return whether ``value`` can be shared by every row as a default."""

    cls = value.__class__
    if cls in _IMMUTABLE_ATOMS:
        return True
    if cls is tuple or cls is frozenset:
        return all(_is_immutable_default(member) for member in value)
    return False


def _cache_cell(store: Any, row: int, fid: int, value: Any) -> None:
    """Write an answered value back into its cell.

    The audited store twins name the unaudited writer as ``cell_cache``:
    caching the answer to a read is not a column write by the reading step.
    Every other store writes through its own ``cell_set``, which keeps the
    fork-isolation index of sealed stores and the overlay of fork views.
    """

    writer = getattr(type(store), "cell_cache", None)
    if writer is None:
        store.cell_set(row, fid, value)
    else:
        writer(store, row, fid, value)


class LazyCell:
    """Marker a cell holds while the store answers its value on read.

    One shared instance per registered field (``LazyFields`` creates them);
    the store read path recognizes the exact class and calls ``read``.

    Parameters
    ----------
    state:
        ``STATE_DEFAULT`` or ``STATE_DERIVED``.
    name:
        Field name the marker stands in for.
    family:
        Name of the registering family (``"default"`` for defaults).
    source:
        The immutable default value, the mutable-default factory, the
        per-cell resolver ``(store, row, name) -> value`` or the
        whole-column resolver ``(store, name) -> values indexed by row``.
    make:
        Whether ``source`` is called to produce the value (every kind but
        an immutable default).
    whole_column:
        Whether a derived read fills the whole column in one pass.
    cache:
        Whether a per-cell answer is written back (object-valued results
        must be, so ``op.x is op.x`` holds; cheap scalars need not be).
    """

    __slots__ = ("cache", "family", "make", "name", "source", "state", "whole_column")

    def __init__(
        self,
        state: str,
        name: str,
        family: str,
        source: Any,
        *,
        make: bool,
        whole_column: bool = False,
        cache: bool = True,
    ) -> None:
        """Bind the marker to its field and its answer."""

        self.state = state
        self.name = name
        self.family = family
        self.source = source
        self.make = make
        self.whole_column = whole_column
        self.cache = cache

    def __repr__(self) -> str:
        """Return a stable representation (never an address)."""

        return f"<lazy {self.state} cell {self.name!r} of family {self.family!r}>"

    def __copy__(self) -> LazyCell:
        """Return the marker itself: a copied row keeps the shared marker."""

        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> LazyCell:
        """Return the marker itself: a copied row keeps the shared marker."""

        return self

    def __reduce__(self) -> NoReturn:
        """Refuse: a state stream carries the answered value, never the marker."""

        raise TypeError(f"{self!r} is a storage marker and never enters a state stream")

    def read(self, store: Any, row: int, fid: int, keep: bool = True) -> Any:
        """Answer the cell at ``(row, fid)`` of ``store``, which holds this marker.

        ``keep=False`` is the state-stream read: a per-row answer (a mutable
        default, a per-cell derived value) is returned without being written
        back, so streaming a record does not materialize its cells. A
        whole-column family fills its column either way; the batch pass is
        the only way it answers.
        """

        if not self.make:
            return self.source
        if self.state == STATE_DEFAULT:
            value = self.source()
        elif self.whole_column:
            return self._fill_column(store, row, fid)
        else:
            value = self._checked(self.source(store, row, self.name))
            keep = keep and self.cache
        if keep:
            _cache_cell(store, row, fid, value)
        return value

    def _checked(self, value: Any) -> Any:
        """Return a resolver result, refusing the two non-values."""

        if value is _MISSING or value.__class__ is LazyCell:
            raise RuntimeError(
                f"resolver of family {self.family!r} returned no value for field {self.name!r}; "
                "a resolver that lacks its raw material must raise"
            )
        return value

    def _fill_column(self, store: Any, row: int, fid: int) -> Any:
        """Run the column resolver once and fill every row still derived.

        Rows that were written or deleted before the first touch keep their
        cell: only cells that still hold this marker are filled.
        """

        values = self.source(store, self.name)
        n_rows = len(store)
        if len(values) != n_rows:
            raise RuntimeError(
                f"column resolver of family {self.family!r} returned {len(values)} values "
                f"for field {self.name!r} of a {n_rows}-row store"
            )
        answer = self._checked(values[row])
        cell_get = store.cell_get
        for other in range(n_rows):
            if cell_get(other, fid, True) is self:
                _cache_cell(store, other, fid, self._checked(values[other]))
        return answer


def cell_state(store: Any, row: int, fid: int) -> str:
    """Return which of the four states the cell at ``(row, fid)`` is in."""

    value = store.cell_get(row, fid, True)
    if value is _MISSING:
        return STATE_DELETED
    if value.__class__ is LazyCell:
        return str(value.state)
    return STATE_SET


def default_answer(value: Any) -> Any:
    """Return what a default-omitted cell answers, or ``value`` for any other cell.

    Nothing is written: a mutable default is built fresh for the caller. This
    is the comparison form for code that inspects a raw cell.
    """

    if value.__class__ is LazyCell and value.state == STATE_DEFAULT:
        return value.source() if value.make else value.source
    return value


def settle_default_cells(
    rows: Iterable[list[Any]], share: Callable[[int, Any], Any] | None = None
) -> None:
    """Write the default into every cell of ``rows`` that is still default-omitted.

    Parameters
    ----------
    rows:
        Row cell lists of a building store (``rows_building()``).
    share:
        Optional ``(fid, value) -> cell`` hook for mutable defaults. It is
        called once per field with a fresh default and returns either that
        same object (the field gets a fresh container per row) or one shared
        storage cell that every settled row of the field then holds.
    """

    shared: dict[int, Any] = {}
    fresh: set[int] = set()
    for row in rows:
        # One comprehension finds the row's lazy cells (most cells hold plain
        # values); they are then visited in field order, as a full scan would.
        for fid in [fid for fid, cell in enumerate(row) if cell.__class__ is LazyCell]:
            value = row[fid]
            if value.state != STATE_DEFAULT:
                continue
            if not value.make:
                row[fid] = value.source
            elif fid in shared:
                row[fid] = shared[fid]
            elif fid in fresh or share is None:
                # Exact list and dict factories are built by display, not a call.
                source = value.source
                row[fid] = [] if source is list else {} if source is dict else source()
            else:
                made = value.source()
                cell = share(fid, made)
                if cell is made:
                    fresh.add(fid)
                else:
                    shared[fid] = cell
                row[fid] = cell


class LazyFields:
    """Per-layout registry of default-capable fields and derived families.

    Parameters
    ----------
    names:
        The layout's stored field names in layout order
        (``OpStoreLayout.names``).
    """

    __slots__ = ("_cells", "_families", "_seed", "names")

    def __init__(self, names: Sequence[str]) -> None:
        """Create an empty registry for one layout."""

        self.names = tuple(names)
        self._cells: dict[str, LazyCell] = {}
        self._families: dict[str, tuple[str, ...]] = {}
        self._seed: list[Any] | None = None

    def _claim(self, name: str, cell: LazyCell) -> LazyCell:
        """Record ``cell`` as the one marker of field ``name``."""

        if name not in self.names:
            raise KeyError(f"{name!r} is not a stored field of this layout")
        if name in self._cells:
            raise ValueError(f"field {name!r} is already registered: {self._cells[name]!r}")
        self._cells[name] = cell
        self._seed = None
        return cell

    def register_default(
        self, name: str, value: Any = None, *, factory: Callable[[], Any] | None = None
    ) -> LazyCell:
        """Register a default-capable field and return its marker.

        Pass an immutable ``value`` (shared by every row, answered without a
        write) or a ``factory`` for a mutable default (called once per row on
        first read, the result written back).
        """

        if factory is not None:
            if value is not None:
                raise ValueError(f"field {name!r}: pass a default value or a factory, not both")
            return self._claim(name, LazyCell(STATE_DEFAULT, name, "default", factory, make=True))
        if not _is_immutable_default(value):
            raise TypeError(
                f"field {name!r}: a {type(value).__name__} default would be shared by every "
                "row; register a factory instead"
            )
        return self._claim(name, LazyCell(STATE_DEFAULT, name, "default", value, make=False))

    def register_family(
        self,
        family: str,
        fields: Iterable[str],
        resolver: Callable[..., Any],
        *,
        whole_column: bool = False,
        cache: bool = True,
    ) -> None:
        """Register one derived family: its fields and the resolver for them.

        A per-cell resolver is called as ``resolver(store, row, name)`` and
        returns the value today's pipeline would have stored. A whole-column
        resolver is called as ``resolver(store, name)`` on the first read of
        any cell of the column and returns the values of every row, indexed
        by row; it must compute them from its inputs, never by reading its
        own column through the store. A resolver that lacks its raw material
        raises; it never returns a default.
        """

        if family in self._families or family == "default":
            raise ValueError(f"family {family!r} is already registered")
        names = tuple(fields)
        if not names:
            raise ValueError(f"family {family!r} registers no fields")
        for name in names:
            cell = LazyCell(
                STATE_DERIVED,
                name,
                family,
                resolver,
                make=True,
                whole_column=whole_column,
                cache=cache or whole_column,
            )
            self._claim(name, cell)
        self._families[family] = names

    def cell(self, name: str) -> LazyCell:
        """Return the marker registered for field ``name``."""

        return self._cells[name]

    def family_fields(self, family: str) -> tuple[str, ...]:
        """Return the fields one registered family resolves."""

        return self._families[family]

    def seed_row(self) -> list[Any]:
        """Return a fresh layout-ordered row for ``adopt_row``.

        Registered fields start default-omitted or derived; every other
        field starts ``_MISSING``, as ``new_row`` leaves it.
        """

        seed = self._seed
        if seed is None:
            cells = self._cells
            seed = self._seed = [cells.get(name, _MISSING) for name in self.names]
        return seed.copy()
