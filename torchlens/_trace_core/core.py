"""TraceCore: per-trace ownership and cross-domain coordination.

TraceCore supplies lifetime, row-id domains, transactions, and cross-domain
coordination ONLY — each domain (columns per kind, pools, edges, groups,
payloads, overlays) owns its typed table and narrow API, so the core never
becomes a generic dict-driven god store.

Fork is core-level COW: the fork shares the frozen base (columns, pools,
edges, payload registrations) and receives its OWN overlay and facade
cache; base immutability makes sharing safe by construction.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from typing import Any

from .columns import ColumnBuilder
from .groups import MembershipGroups
from .overlays import MISSING, RowOverlay, Transaction
from .payloads import PayloadArena
from .pools import ClosurePool, InternPool
from .relations import EdgeTable


class KindTable:
    """Typed column block for one record kind."""

    __slots__ = ("_columns", "_frozen", "_n_rows", "kind")

    def __init__(self, kind: str) -> None:
        """Create an empty table for ``kind``."""

        self.kind = kind
        self._columns: dict[str, ColumnBuilder] = {}
        self._n_rows = 0
        self._frozen = False

    def __len__(self) -> int:
        """Return the row count."""

        return self._n_rows

    @property
    def frozen(self) -> bool:
        """Return whether the table is frozen."""

        return self._frozen

    def column(self, field: str, codec: str = "object") -> ColumnBuilder:
        """Return (creating on first use) the column for ``field``."""

        column = self._columns.get(field)
        if column is None:
            if self._frozen:
                raise RuntimeError(f"{self.kind} table is frozen")
            column = ColumnBuilder(codec)
            for _ in range(self._n_rows):
                column.append_missing()
            self._columns[field] = column
        return column

    def columns(self) -> Iterator[tuple[str, ColumnBuilder]]:
        """Iterate (field, column) pairs."""

        return iter(self._columns.items())

    def new_row(self) -> int:
        """Append one row across every column and return its row id."""

        if self._frozen:
            raise RuntimeError(f"{self.kind} table is frozen")
        row = self._n_rows
        self._n_rows += 1
        for column in self._columns.values():
            column.append_missing()
        return row

    def set(self, row: int, field: str, value: Any) -> None:
        """Write one cell during BUILDING."""

        self.column(field).set(row, value)

    def get(self, row: int, field: str) -> Any:
        """Read one cell from the base (no overlay)."""

        column = self._columns.get(field)
        return None if column is None else column.get(row)

    def freeze(self) -> None:
        """Freeze every column; further writes go to the overlay."""

        for column in self._columns.values():
            column.freeze()
        self._frozen = True


class TraceCore:
    """One per-trace semantic store."""

    __slots__ = (
        "__weakref__",
        "_facade_factory",
        "_facades",
        "_record_translator",
        "_strong_facades",
        "backward_epochs",
        "closures",
        "edges",
        "groups",
        "kind_rows",
        "label_rows",
        "ops",
        "overlay",
        "payloads",
        "pool",
        "tables",
    )

    def __init__(self) -> None:
        """Create an empty core."""

        self.tables: dict[str, KindTable] = {}
        # The Op row store (op_store.OpRowStore) once the ingress binds it;
        # None until materialize step 0 creates it for a captured run.
        self.ops: Any = None
        # The non-Op kind tables (param/buffer/module/module_call/
        # func_call_location), populated by the build passes' adopt_records.
        self.kind_rows: dict[str, Any] = {}
        # The canonical label -> op-row index, bound at the relation freeze
        # (final layer_label per live core-backed op). The core-side index
        # the Trace decomposition resolves lookups through.
        self.label_rows: dict[str, int] = {}
        self.pool = InternPool()
        self.closures = ClosurePool()
        self.edges: dict[str, EdgeTable] = {}
        self.groups: dict[str, MembershipGroups] = {}
        self.payloads = PayloadArena()
        self.overlay = RowOverlay()
        # Weak-valued facade cache (the strong->weak flip): identity is
        # stable while ANY reference lives, and an uninspected row retains no
        # facade. Record lifetime stays pinned by the trace-side lookup
        # containers (aliases-v1 row 1b), never by this cache. ``Op`` is
        # deliberately NOT weak-referenceable (aliases-v1 pins the refusal),
        # so non-weakref-able facades fall back to the strong side table.
        self._facades: weakref.WeakValueDictionary[tuple[str, int], Any] = (
            weakref.WeakValueDictionary()
        )
        self._strong_facades: dict[tuple[str, int], Any] = {}
        self._facade_factory: Callable[[str, int], Any] | None = None
        # Strong anchor for a fork core's record translator: the store views
        # hold it WEAKLY (a retained fork Op must not root the fork graph
        # through it), so the translator lives exactly as long as its core.
        self._record_translator: Any = None
        self.backward_epochs: list[Any] = []

    def table(self, kind: str) -> KindTable:
        """Return (creating on first use) the table for ``kind``."""

        table = self.tables.get(kind)
        if table is None:
            table = KindTable(kind)
            self.tables[kind] = table
        return table

    def edge_table(self, family: str) -> EdgeTable:
        """Return (creating on first use) one edge family table."""

        edges = self.edges.get(family)
        if edges is None:
            edges = EdgeTable()
            self.edges[family] = edges
        return edges

    def read(self, kind: str, row: int, field: str) -> Any:
        """Overlay-aware cell read."""

        overlaid = self.overlay.read(row, f"{kind}.{field}", MISSING)
        if overlaid is not MISSING:
            return overlaid
        return self.table(kind).get(row, field)

    def write(self, kind: str, row: int, field: str, value: Any) -> None:
        """Public write: base while building, overlay after freeze."""

        table = self.table(kind)
        if table.frozen:
            self.overlay.write(row, f"{kind}.{field}", value)
        else:
            table.set(row, field, value)

    def set_facade_factory(self, factory: Callable[[str, int], Any]) -> None:
        """Install the facade constructor used by ``facade()``."""

        self._facade_factory = factory

    def facade(self, kind: str, row: int) -> Any:
        """Return the identity-cached facade for one row.

        Weak-valued: the cache guarantees ``facade(k, r) is facade(k, r)``
        while any reference to the facade lives, and never pins an
        unreferenced facade. Facades whose class refuses weak references
        (``Op``) are held in the strong side table instead.
        """

        key = (kind, row)
        cached = self._facades.get(key)
        if cached is None:
            cached = self._strong_facades.get(key)
        if cached is None:
            if self._facade_factory is None:
                raise RuntimeError("no facade factory installed")
            cached = self._facade_factory(kind, row)
            try:
                self._facades[key] = cached
            except TypeError:
                self._strong_facades[key] = cached
        return cached

    def freeze(self) -> None:
        """Freeze every kind table and edge family."""

        for table in self.tables.values():
            table.freeze()
        for row_store in self.kind_rows.values():
            row_store.freeze()
        for family, edges in self.edges.items():
            del family
            if len(edges):
                n_rows = max(len(table) for table in self.tables.values())
                edges.freeze(n_rows, n_rows)

    def fork(self) -> TraceCore:
        """COW fork: share the frozen base, detach every mutation surface.

        The fork receives ``OpStoreView`` wrappers over the sealed op store
        and every kind table (fork writes land in per-view overlays), cloned
        group tables (removal scrub on one trace never reaches the other),
        its own label index and core overlay snapshot, a fresh facade cache,
        and an empty backward-epoch list (fork projections rematerialize cold
        from the fork's detached event stream).
        """

        from .op_store import OpStoreView

        child = TraceCore.__new__(TraceCore)
        child.tables = self.tables
        group_tables: dict[int, MembershipGroups] = {}
        child.groups = {}
        for family, table in self.groups.items():
            clone = MembershipGroups(table.view_type)
            # Rebuild the view objects (not just the list): the parent and
            # fork must never hand out the SAME live view object, matching
            # the deepcopy fork's cross-trace distinctness. The inner list()
            # defeats CPython's identity shortcut for frozenset(frozenset)/
            # tuple(tuple).
            clone._views = [table.view_type(list(view)) for view in table._views]
            clone._source_tables = (*table._source_tables, table)
            child.groups[family] = clone
            # A GroupRef reachable through the fork's cells may bind THIS
            # table or any ancestor (fork chains flatten onto root storage
            # whose refs bind the root tables); all translate to the clone.
            for source_table in clone._source_tables:
                group_tables[id(source_table)] = clone
        child.ops = OpStoreView(self.ops, group_tables) if self.ops is not None else None
        if child.ops is not None:
            # The fork core is a co-owner of the shared sealed base: payload
            # eviction must wait for the LAST core -- parent or
            # any fork -- to be garbage-collected, or a live fork's reads
            # through the base would lose their payloads.
            child.ops.base.adopt_payload_owner(child)
        # An unfrozen kind table (born after a rehydrated load's seal and not
        # yet sealed itself) cannot back a view; its records take the fork
        # builder's detached-duplication fallback instead.
        child.kind_rows = {
            kind: OpStoreView(store, group_tables)
            for kind, store in self.kind_rows.items()
            if store.frozen
        }
        child.label_rows = dict(self.label_rows)
        child.pool = self.pool
        child.closures = self.closures
        child.edges = self.edges
        child.payloads = self.payloads
        child.overlay = RowOverlay()
        for key, value in self.overlay.snapshot().items():
            child.overlay.write(key[0], key[1], value)
        child._facades = weakref.WeakValueDictionary()
        child._strong_facades = {}
        child._facade_factory = self._facade_factory
        child._record_translator = None
        child.backward_epochs = []
        return child

    def store_views(self) -> Iterator[Any]:
        """Yield every COW store view owned by this core (fork cores only)."""

        from .op_store import OpStoreView

        if isinstance(self.ops, OpStoreView):
            yield self.ops
        for store in self.kind_rows.values():
            if isinstance(store, OpStoreView):
                yield store

    def view_for_store(self, store: Any) -> Any:
        """Return this core's view over ``store``, or ``None``."""

        for view in self.store_views():
            if view.base is store:
                return view
        return None

    def transaction(self) -> Transaction:
        """Checkpoint every mutation surface for one atomic rollback.

        Covers the core overlay, the op-store and kind-table overlays
        (base stores and fork views alike), and the backward-epoch list —
        the intervention-rollback substrate from the converged design.
        Call ``rollback()`` on the returned transaction to restore all of
        them atomically; dropping it commits.
        """

        txn = Transaction({"core": self.overlay})
        stores = [self.ops, *self.kind_rows.values()]
        for index, store in enumerate(stores):
            if store is None:
                continue

            def _restore_store(snapshot: dict, _store: Any = store) -> None:
                """Restore one store's overlay in place from a transaction snapshot."""

                _store._overlay = dict(snapshot)

            txn.stash(("store", index), dict(store._overlay), _restore_store)
            # Sealed row-major stores (below the transpose threshold) write
            # cells in place rather than through the overlay, so their rows
            # checkpoint too.
            rows = getattr(store, "_rows", None)
            if rows is not None and store.frozen:

                def _restore_rows(snapshot: list, _store: Any = store) -> None:
                    """Restore a sealed store's row cells from a transaction snapshot."""

                    _store._rows = [list(cells) for cells in snapshot]

                txn.stash(
                    ("store-rows", index),
                    [list(cells) for cells in rows],
                    _restore_rows,
                )

        def _restore_epochs(snapshot: list) -> None:
            """Restore the backward-epoch list in place from a transaction snapshot."""

            self.backward_epochs[:] = snapshot

        txn.stash("epochs", list(self.backward_epochs), _restore_epochs)
        return txn

    def weak_self(self) -> weakref.ref[TraceCore]:
        """Return a weak reference to this core (facade back-pointer)."""

        return weakref.ref(self)
