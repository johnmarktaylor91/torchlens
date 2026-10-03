"""Columnar row store backing the ``Op`` facade (the M5 seam).

One ``OpRowStore`` per captured trace holds every stored ``Op`` field cell;
``Op`` itself is a two-word ``(_core, _row)`` facade whose generated data
descriptors read and write cells here. ``DetachedOpStore`` is the single-row
twin used by ``Op.copy()``, pickle restoration, fork shells, and preview
backends that construct ops outside the event-materialize ingress.

Storage phases:

* BUILDING (postprocess steps 0-20): rows are plain Python lists (one
  ``[_MISSING] * n_fields`` allocation per op), so cell reads/writes are
  list indexing and the per-op footprint matches the former slot layout.
* FROZEN (after step 20, where the standalone compaction passes ran): rows
  transpose once into per-field columns. Numeric/bool columns pack into
  numpy arrays when every present value is an exact ``bool``/``int``/
  ``float`` (subclasses such as ``Bytes`` stay object-backed so reads keep
  their exact public type); everything else stays an object column. Later
  writes land in a sparse overlay; deletes on object columns release the
  cell value in place so op removal keeps freeing payloads.

An unset cell is ``_MISSING`` (never ``None`` -- ``None`` is a real stored
value); facade descriptors translate ``_MISSING`` into the exact
``AttributeError`` an unset ``__slots__`` member raised.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np

from .groups import GroupRef, MembershipGroups

#: Unset-cell sentinel; distinct from None (a real storable value).
_MISSING = object()

#: CSR-backed relation cell sentinel (M6): the value lives in the store's
#: dataflow edge table; the facade descriptor rematerializes an immutable
#: tuple view on first access and caches it back through ``cell_set``.
#: Distinct from ``_MISSING`` so present/absent semantics stay exact —
#: ``cell_del`` on a ``_CSR`` cell clears to genuinely absent.
_CSR = object()

#: Shared-fact cell sentinel (M7): the value lives in the store's fact
#: blocks (the FunctionCall / ParamAlias group tables); the facade
#: descriptor hydrates the exact public container type for the row on first
#: access and caches it back. Same present/absent semantics as ``_CSR``.
_FACT = object()

#: Overlay-miss sentinel local to cell reads.
_NO_OVERLAY = object()

#: Minimum row count for the freeze-time columnar transpose. Below this the
#: per-trace fixed cost of ~190 column objects (~29 KB measured) exceeds any
#: packing win, so small sealed stores keep their row-major cells (write and
#: delete semantics are identical either way; ``new_row`` refuses once sealed).
_TRANSPOSE_MIN_ROWS = 512

# Depth ceiling for the nested-container payload eviction walk (r8 R37).
_EVICT_WALK_DEPTH_CEILING = 8


def _evict_payloads_in_cell(
    value: Any,
    tensor_cls: type,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Return ``value`` with every reachable tensor replaced by ``None``.

    Bounded to builtin containers (list/tuple/dict/set/frozenset): a direct
    tensor becomes ``None`` (the payload-absent spelling), mutable containers
    evict in place, tuples/frozensets rebuild only when a member changed
    (named tuples reconstruct through their own type, falling back to the
    unchanged original -- keeping a pin -- rather than corrupting the
    container type). Tensors inside non-builtin custom objects remain the
    disclosed residual; cyclic containers terminate through the seen set.
    """

    if isinstance(value, tensor_cls):
        return None
    if depth >= _EVICT_WALK_DEPTH_CEILING:
        return value
    if isinstance(value, list):
        if seen is None:
            seen = set()
        if id(value) in seen:
            return value
        seen.add(id(value))
        for index, item in enumerate(value):
            value[index] = _evict_payloads_in_cell(item, tensor_cls, depth + 1, seen)
        return value
    if isinstance(value, tuple):
        rebuilt = [_evict_payloads_in_cell(item, tensor_cls, depth + 1, seen) for item in value]
        if all(new is old for new, old in zip(rebuilt, value, strict=True)):
            return value
        if type(value) is tuple:
            return tuple(rebuilt)
        try:
            # Named tuples and tuple subclasses reconstruct through their own
            # type; an unreconstructable subclass keeps the original (a
            # retained pin beats a corrupted container).
            return type(value)(*rebuilt)
        except Exception:
            try:
                return type(value)(rebuilt)
            except Exception:
                return value
    if isinstance(value, dict):
        if seen is None:
            seen = set()
        if id(value) in seen:
            return value
        seen.add(id(value))
        for key in list(value):
            value[key] = _evict_payloads_in_cell(value[key], tensor_cls, depth + 1, seen)
        return value
    if isinstance(value, (set, frozenset)):
        if not any(isinstance(item, tensor_cls) for item in value):
            return value
        survivors = {item for item in value if not isinstance(item, tensor_cls)}
        survivors.add(None)
        try:
            return type(value)(survivors)
        except Exception:
            return value
    return value


class OpStoreLayout:
    """Immutable field-name-to-field-id layout shared by every op store.

    Parameters
    ----------
    names:
        Stored field names in declared state order (the ``_OP_SLOT_NAMES``
        order: FIELD_ORDER minus computed property-backed names, plus the
        dynamic runtime slots).
    """

    __slots__ = ("fid_by_name", "n_fields", "names")

    def __init__(self, names: tuple[str, ...]) -> None:
        """Freeze the layout for ``names``."""

        self.names = tuple(names)
        self.fid_by_name = {name: fid for fid, name in enumerate(self.names)}
        self.n_fields = len(self.names)


class _FrozenColumn:
    """One frozen per-field column with presence tracking.

    Parameters
    ----------
    values:
        numpy array (packed) or Python list (object column).
    present:
        ``bytearray`` presence mask, or ``None`` when every row is present.
    packed:
        Whether ``values`` is a numpy array requiring ``.item()`` on read.
    """

    __slots__ = ("packed", "present", "values")

    def __init__(self, values: Any, present: bytearray | None, packed: bool) -> None:
        """Bind backing storage produced by ``OpRowStore.freeze``."""

        self.values = values
        self.present = present
        self.packed = packed

    def get(self, row: int) -> Any:
        """Return the cell value at ``row``, or ``_MISSING``."""

        present = self.present
        if present is not None and not present[row]:
            return _MISSING
        value = self.values[row]
        if self.packed:
            return value.item()
        return value

    def clear(self, row: int) -> None:
        """Release the cell at ``row`` (object columns drop the reference)."""

        if self.present is None:
            self.present = bytearray(b"\x01" * len(self.values))
        self.present[row] = 0
        if not self.packed:
            self.values[row] = None


class OpRowStore:
    """Per-trace row store for every ``Op`` of one captured run.

    Parameters
    ----------
    layout:
        Shared field layout (one module-level instance per process).
    """

    __slots__ = (
        "_columns",
        "_compacted_singletons",
        "_cow_shared",
        "_mutable_keys",
        "_n_rows",
        "_overlay",
        "_payload_owner_count",
        "_payload_views",
        "_sweep_plan",
        "_rows",
        "_sealed",
        "dataflow_edges",
        "fact_blocks",
        "layout",
        "ref_labels",
    )

    def __init__(self, layout: OpStoreLayout) -> None:
        """Create an empty building-phase store."""

        self.layout = layout
        self._rows: list[list[Any]] | None = []
        self._columns: list[_FrozenColumn] | None = None
        self._overlay: dict[int, Any] = {}
        self._n_rows = 0
        self._sealed = False
        # Sticky flag set by the first OpStoreView taken over this store
        # (M11 COW fork). Once shared, deletes must stop releasing object
        # column cells IN PLACE — a fork reads the same column backing —
        # and tombstone through the overlay instead. Memory-only effect:
        # a removed op's payloads on a forked-from trace are released with
        # the store rather than at removal time.
        self._cow_shared = False
        # Fork-isolation index (M11/F4): ``fid -> rows`` (tuple once
        # columnar, set while sealed row-major) of cells whose values may
        # need eager fork-time isolation (exact builtin mutable containers,
        # and immutable containers transitively nesting one).
        # Built lazily by the first ``OpStoreView.isolate_mutable_cells``
        # sweep and kept valid thereafter: frozen columnar cells never
        # mutate once ``_cow_shared`` (writes go to the overlay, deletes
        # tombstone), and sealed row-major stores swap to
        # ``_SealedRowMajorOpRowStore`` at freeze so post-seal container
        # writes register here.
        self._mutable_keys: dict[int, Any] | None = None
        # Cached per-store sweep plan derived from ``_mutable_keys`` (F4/F11):
        # flat parallel arrays of pre-resolved (key, allocation/copy) work so
        # the per-fork sweep is a tight guard+copy loop with no per-cell
        # backing fetch or classification. Built lazily with the index;
        # sealed row-major maintenance hooks drop it on any post-seal cell
        # REBINDING that could stale it (columnar cell bindings are immutable
        # once ``_cow_shared``). In-place mutation of a held container fires
        # no hook on either seal shape, so plan entries alias the observed
        # values and execution re-checks them (``_build_sweep_plan``).
        self._sweep_plan: tuple[Any, ...] | None = None
        # The M6 dataflow family: bound at the freeze-time relation
        # conversion (same EdgeTable object registered in the owning
        # TraceCore's edge registry) together with the row -> reference-label
        # table the facade descriptors use to rematerialize views.
        self.dataflow_edges: Any = None
        self.ref_labels: dict[int, str] | None = None
        # The M7 shared-fact blocks (FunctionCall / ParamAlias group
        # tables), bound by the same freeze-time conversion.
        self.fact_blocks: Any = None
        # M14 singleton-label compaction registry (kind tables only):
        # ``row * n_fields + fid -> the exact str object`` the freeze-seam
        # compaction stored in place of a one-element label list. Decode is
        # gated on OBJECT IDENTITY (``registry[key] is cell_value``), so a
        # later user write of any OTHER object to the cell self-invalidates
        # the entry without write-path hooks; entries are never popped
        # (identity-guarded stale entries are harmless, and the str is
        # already retained by the cell/hydrated list, so no extra pinning).
        self._compacted_singletons: dict[int, str] | None = None
        # Count of live owning TraceCores (the capture's own core plus one
        # per fork core viewing this base). When the LAST owner is
        # garbage-collected, tensor payload cells are evicted so a retained
        # ``Op`` facade -- which holds this store strongly -- keeps its
        # metadata readable without pinning every captured activation.
        self._payload_owner_count = 0
        # Weak references to every OpStoreView over this base: row-major
        # views SNAPSHOT the row lists (and every view has private
        # overlays), so last-owner payload eviction must sweep surviving
        # views too -- a retained fork Op reads through its view's
        # snapshot, never the live base rows. Weak so the base never
        # extends a dead fork's view lifetime.
        self._payload_views: list[Any] = []

    def adopt_payload_owner(self, owner: Any) -> None:
        """Register one owning ``TraceCore`` whose death releases payload pins.

        Called at the capture freeze seam for the original core and at
        ``TraceCore.fork()`` for each fork core sharing this base. The
        finalizer holds this store strongly, which adds no pinning beyond
        the owner's own strong reference and is dropped when it fires.

        Parameters
        ----------
        owner:
            Weak-referenceable owning object (a ``TraceCore``).
        """

        self._payload_owner_count += 1
        weakref.finalize(owner, self._release_payload_owner)

    def _release_payload_owner(self) -> None:
        """Drop one owner; evict tensor payloads when the last owner dies."""

        self._payload_owner_count -= 1
        if self._payload_owner_count == 0 and self._sealed:
            self._evict_tensor_payloads()

    def _evict_tensor_payloads(self) -> None:
        """Release tensor-valued cells once no owning core is alive.

        A retained ``Op`` is a two-word view over this shared store, so one
        live facade used to pin EVERY captured payload after its Trace died.
        With no owning TraceCore left, no store view can still serve another
        trace, so tensor cells (saved ``out`` payloads and friends) are
        replaced with ``None`` -- the existing "no payload retained"
        spelling -- while every metadata cell stays readable. Tensors nested
        inside builtin CONTAINER cells (``saved_args`` trees and friends)
        are evicted through a bounded walk too (r8 R37, sol repro: a tensor
        inside a saved-args tuple survived Trace death via any retained Op);
        tensors inside non-builtin custom objects remain the disclosed
        residual. Keep the source Trace alive (or clone the tensor) to keep
        payloads past the trace's lifetime.
        """

        import torch

        tensor_cls = torch.Tensor
        evict = _evict_payloads_in_cell
        rows = self._rows
        if rows is not None:
            for row_cells in rows:
                for fid, value in enumerate(row_cells):
                    row_cells[fid] = evict(value, tensor_cls)
        columns = self._columns
        if columns is not None:
            for column in columns:
                if column.packed:
                    continue
                values = column.values
                for index, value in enumerate(values):
                    values[index] = evict(value, tensor_cls)
        overlay = self._overlay
        for key, value in overlay.items():
            overlay[key] = evict(value, tensor_cls)
        # Surviving views (reachable only through retained fork Ops now that
        # every owning core is dead) read their own snapshot surfaces, so
        # they are swept too. Shared snapshot lists (fork chains) evict
        # idempotently.
        for view_ref in self._payload_views:
            view = view_ref()
            if view is None:
                continue
            snapshot_rows = view._base_rows
            if snapshot_rows is not None:
                for row_cells in snapshot_rows:
                    for fid, value in enumerate(row_cells):
                        row_cells[fid] = evict(value, tensor_cls)
            for view_overlay in (view._base_overlay, view._overlay):
                for key, value in view_overlay.items():
                    view_overlay[key] = evict(value, tensor_cls)

    def __len__(self) -> int:
        """Return the number of rows ever appended (removed rows included)."""

        return self._n_rows

    @property
    def frozen(self) -> bool:
        """Return whether the store has been sealed by ``freeze()``."""

        return self._sealed

    def new_row(self) -> int:
        """Append one all-``_MISSING`` row and return its row id."""

        rows = self._rows
        if rows is None or self._sealed:
            raise RuntimeError("op store is frozen; no new rows may be appended")
        row = self._n_rows
        self._n_rows = row + 1
        rows.append([_MISSING] * self.layout.n_fields)
        return row

    def adopt_row(self, cells: list[Any]) -> int:
        """Append one pre-built row (the bulk ingress path) and return its id.

        ``cells`` must be layout-ordered with ``_MISSING`` for unset fields;
        the store takes ownership of the list.
        """

        rows = self._rows
        if rows is None or self._sealed:
            raise RuntimeError("op store is frozen; no new rows may be appended")
        if len(cells) != self.layout.n_fields:
            raise ValueError("row cell count does not match the layout")
        row = self._n_rows
        self._n_rows = row + 1
        rows.append(cells)
        return row

    def cell_get(self, row: int, fid: int) -> Any:
        """Return one cell value, or ``_MISSING`` when unset."""

        rows = self._rows
        if rows is not None:
            return rows[row][fid]
        overlay = self._overlay
        if overlay:
            value = overlay.get(row * self.layout.n_fields + fid, _NO_OVERLAY)
            if value is not _NO_OVERLAY:
                return value
        columns = self._columns
        assert columns is not None
        return columns[fid].get(row)

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell (base while building, overlay after freeze)."""

        rows = self._rows
        if rows is not None:
            rows[row][fid] = value
        else:
            self._overlay[row * self.layout.n_fields + fid] = value

    def cell_del(self, row: int, fid: int) -> bool:
        """Delete one cell; return whether it was previously set.

        Frozen object columns release the value in place (so a removed op's
        payloads keep being freed); packed columns tombstone via the overlay.
        """

        rows = self._rows
        if rows is not None:
            row_cells = rows[row]
            if row_cells[fid] is _MISSING:
                return False
            row_cells[fid] = _MISSING
            return True
        if self.cell_get(row, fid) is _MISSING:
            return False
        key = row * self.layout.n_fields + fid
        columns = self._columns
        assert columns is not None
        column = columns[fid]
        if column.packed or self._cow_shared:
            self._overlay[key] = _MISSING
        else:
            self._overlay.pop(key, None)
            column.clear(row)
        return True

    def register_compacted_singleton(self, row: int, fid: int, element: str) -> None:
        """Record one singleton-label compaction (freeze-seam pass only)."""

        registry = self._compacted_singletons
        if registry is None:
            registry = self._compacted_singletons = {}
        registry[row * self.layout.n_fields + fid] = element

    def compacted_singleton(self, row: int, fid: int, value: Any) -> bool:
        """Return whether ``value`` is the compacted singleton for this cell.

        True only when ``value`` is the EXACT object the compaction pass
        stored (object identity), so any later cell write self-invalidates.
        """

        registry = self._compacted_singletons
        return registry is not None and registry.get(row * self.layout.n_fields + fid) is value

    def items(self, row: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(field_name, value)`` for every set cell in layout order.

        ``PooledCell`` cells hydrate to a fresh equal container, and
        compacted singleton-label cells decode to a fresh one-element list
        (state streams must never carry either internal encoding); the cell
        itself is left untouched — pickling a record must not materialize
        its containers.
        """

        names = self.layout.names
        for fid, name in enumerate(names):
            value = self.cell_get(row, fid)
            if value is not _MISSING:
                if value.__class__ is PooledCell:
                    value = value.hydrate()
                elif value.__class__ is str and self.compacted_singleton(row, fid, value):
                    value = [value]
                yield name, value

    def retained_bytes(self) -> int:
        """Return shallow structural bytes retained by this store.

        Counts the store object, row lists (building) or column
        backings/presence masks/overlay (frozen) -- never the cell VALUES,
        matching the former shallow per-op ``sys.getsizeof`` accounting.
        """

        import sys

        total = sys.getsizeof(self)
        rows = self._rows
        if rows is not None:
            total += sys.getsizeof(rows)
            for row_cells in rows:
                total += sys.getsizeof(row_cells)
            return total
        total += sys.getsizeof(self._overlay)
        columns = self._columns
        assert columns is not None
        total += sys.getsizeof(columns)
        for column in columns:
            total += sys.getsizeof(column)
            total += sys.getsizeof(column.values)
            if column.present is not None:
                total += sys.getsizeof(column.present)
        return total

    def rows_building(self) -> list[list[Any]] | None:
        """Return the raw building-phase rows, or ``None`` once frozen.

        Exposed for the trace-level metadata compaction sweep, which pools
        immutable values column-by-column without the attribute protocol.
        """

        return self._rows

    def freeze(self) -> None:
        """Seal the store; transpose into columns when the trace is large.

        Sealing always stops row appends. The physical columnar transpose
        (numeric packing) only pays for itself past ``_TRANSPOSE_MIN_ROWS``;
        smaller sealed stores keep their row-major cells with identical
        read/write/delete behavior.
        """

        rows = self._rows
        if rows is None or self._sealed:
            self._sealed = True
            return
        self._sealed = True
        if self._n_rows < _TRANSPOSE_MIN_ROWS:
            # Sealed row-major cells keep mutating IN PLACE on later writes,
            # so the sealed twin tracks post-seal container writes for the
            # fork-isolation index (columnar stores need no tracking: their
            # post-seal writes land in the overlay, which every fork
            # snapshots). Audits never see a sealed store (the trailing
            # postprocess audit window closes before the freeze seam).
            self.__class__ = _SealedRowMajorOpRowStore
            return
        n_fields = self.layout.n_fields
        columns: list[_FrozenColumn] = []
        for fid in range(n_fields):
            column_values = [row_cells[fid] for row_cells in rows]
            columns.append(_freeze_column(column_values))
        self._columns = columns
        self._rows = None


def _freeze_column(column_values: list[Any]) -> _FrozenColumn:
    """Build one frozen column, packing exact-typed numerics when safe.

    Packing requires every PRESENT value to be an exact ``bool``, exact
    ``int`` (within int64 range), or exact ``float`` -- one uniform type per
    column. Subclass instances (``Bytes``, ``Flops``, ``Duration``...),
    ``None`` values, and mixed types keep the object backing so every read
    returns the exact object/type that was stored.
    """

    present: bytearray | None = None
    any_missing = False
    for value in column_values:
        if value is _MISSING:
            any_missing = True
            break
    if any_missing:
        present = bytearray(0 if value is _MISSING else 1 for value in column_values)

    value_cls: type | None = None
    uniform = True
    for value in column_values:
        if value is _MISSING:
            continue
        cls = value.__class__
        if cls is not bool and cls is not int and cls is not float:
            uniform = False
            break
        if value_cls is None:
            value_cls = cls
        elif cls is not value_cls:
            uniform = False
            break
    if uniform and value_cls is not None:
        dtype = {bool: np.bool_, int: np.int64, float: np.float64}[value_cls]
        fill: Any = False if value_cls is bool else value_cls(0)
        try:
            packed_values: Any = np.array(
                [fill if value is _MISSING else value for value in column_values],
                dtype=dtype,
            )
        except OverflowError:
            pass
        else:
            return _FrozenColumn(packed_values, present, True)
    object_values = [None if value is _MISSING else value for value in column_values]
    return _FrozenColumn(object_values, present, False)


def _contains_mutable_container(value: Any) -> bool:
    """Return whether a tuple/frozenset transitively nests a dict/list/set."""

    for item in value:
        cls = item.__class__
        if cls is dict or cls is list or cls is set:
            return True
        if (cls is tuple or cls is frozenset) and _contains_mutable_container(item):
            return True
    return False


def _needs_eager_isolation(value: Any) -> bool:
    """Return whether a cell value must be copied at fork time.

    This is the minimal leak-closure set for the parent->fork direction:
    exact builtin mutable containers (a parent in-place mutation after the
    fork would otherwise be visible through the fork's copy-on-first-read
    window) and exact immutable containers transitively nesting one. Every
    other translation concern (record facades, ``GroupRef`` cells, interned
    immutable views of atomics) stays lazy on first read — translation is
    not mutation, so deferring it cannot leak parent state.
    """

    cls = value.__class__
    if cls is dict or cls is list or cls is set:
        return True
    if cls is tuple or cls is frozenset:
        return _contains_mutable_container(value)
    return False


def _build_mutable_key_index(store: OpRowStore) -> dict[int, Any]:
    """Scan a sealed base store once for its fork-isolation candidate cells.

    Returns ``fid -> rows`` (field-grouped so the per-fork sweep reads
    column backings by direct index, no per-cell method dispatch). Runs at
    the FIRST fork of a store (never at capture/freeze time, so unforked
    traces pay nothing) and is cached on the store: frozen columnar cells
    are immutable once ``_cow_shared`` (post-seal writes overlay, deletes
    tombstone) and sealed row-major stores register post-seal container
    writes through ``_SealedRowMajorOpRowStore.cell_set``, so the index
    never goes stale. Packed numeric columns are skipped wholesale.
    """

    index: dict[int, Any] = {}
    rows = store._rows
    if rows is not None:
        # Row-major indices stay sets: ``_SealedRowMajorOpRowStore.cell_set``
        # keeps extending them on post-seal container writes.
        for row_index, row_cells in enumerate(rows):
            for fid, value in enumerate(row_cells):
                if _needs_eager_isolation(value):
                    fid_rows = index.get(fid)
                    if fid_rows is None:
                        fid_rows = index[fid] = set()
                    fid_rows.add(row_index)
        return index
    columns = store._columns
    assert columns is not None
    # Columnar cells are immutable once ``_cow_shared``, so the per-field row
    # lists freeze as ascending tuples (faster iteration, better locality).
    for fid, column in enumerate(columns):
        if column.packed:
            continue
        present = column.present
        candidate_rows: list[int] = []
        if present is None:
            for row, value in enumerate(column.values):
                if _needs_eager_isolation(value):
                    candidate_rows.append(row)
        else:
            for row, value in enumerate(column.values):
                if present[row] and _needs_eager_isolation(value):
                    candidate_rows.append(row)
        if candidate_rows:
            index[fid] = tuple(candidate_rows)
    return index


def _build_sweep_plan(store: OpRowStore) -> tuple[Any, ...]:
    """Flatten the mutable-cell index into pre-resolved per-fork sweep work.

    Returns ``(alloc_entries, copy_entries)`` — two sequences of small
    RECORD tuples (``(key, cls, value)`` and ``(key, value, deep)``) so
    ``isolate_mutable_cells`` runs a tight guard+copy loop with no per-cell
    backing fetch, key arithmetic, or classification (the F11 small-trace
    fork constant). Records, not positionally-correlated parallel arrays: a
    one-sided append cannot silently truncate the sweep and leave tail fork
    cells sharing mutable containers with their parent (R23-5). Empty
    mutable containers (the census-dominant case, ~11/op) become bare class
    allocations; non-empty dict/list/set cells use ``_eager_copy``; nesting
    tuples/frozensets use the generic translating copier.

    Validity: cell BINDINGS cannot change under a cached plan — columnar
    cells are immutable once ``_cow_shared``, and the sealed row-major
    write/delete hooks drop the plan whenever a post-seal change touches
    an indexed cell or stores a new container (the next fork rebuilds it
    from the maintained index). The held container OBJECT can still mutate
    in place with no hook firing, which is why every plan entry aliases
    the observed value: copies read the live object, and each alloc entry
    re-checks emptiness at execution, falling back to a real copy after an
    in-place empty->non-empty transition (sol closure round 3). Plan
    values alias the store's own cell backing, so the plan retains nothing
    the store does not already retain.
    """

    index = store._mutable_keys
    assert index is not None
    n_fields = store.layout.n_fields
    alloc_entries: list[tuple[int, type, Any]] = []
    copy_entries: list[tuple[int, Any, bool]] = []
    rows = store._rows
    columns = store._columns
    for fid in sorted(index):
        fid_rows = index[fid]
        if rows is None:
            assert columns is not None
            backing = columns[fid].values
        else:
            backing = None
        for row in sorted(fid_rows):
            value = rows[row][fid] if rows is not None else backing[row]
            cls = value.__class__
            if cls is dict or cls is list or cls is set:
                if value:
                    copy_entries.append((row * n_fields + fid, value, False))
                else:
                    alloc_entries.append((row * n_fields + fid, cls, value))
            elif (cls is tuple or cls is frozenset) and _contains_mutable_container(value):
                copy_entries.append((row * n_fields + fid, value, True))
    return (tuple(alloc_entries), tuple(copy_entries))


def _eager_copy(value: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Copy one mutable container for the eager fork sweep.

    Same result as ``cow_copy_value``, restructured for the sweep's value
    population: empty containers and containers of exact atomics copy at C
    speed (the overwhelmingly common case — the F8 census counts ~11 empty
    plus ~16 small flat containers per op), atomic members skip the
    per-item call entirely, and nested mutable containers recurse through
    the same fast paths. Only tuples/frozensets and non-container members
    fall through to the generic translating copier (which owns the
    identity-sharing rules).
    """

    atomic = _COW_ATOMIC
    cls = value.__class__
    if cls is dict:
        for key, item in value.items():
            if key.__class__ not in atomic or item.__class__ not in atomic:
                return {
                    (key if key.__class__ in atomic else _eager_item(key, translate)): (
                        item if item.__class__ in atomic else _eager_item(item, translate)
                    )
                    for key, item in value.items()
                }
        return value.copy()
    if cls is list:
        for item in value:
            if item.__class__ not in atomic:
                return [
                    item if item.__class__ in atomic else _eager_item(item, translate)
                    for item in value
                ]
        return value.copy()
    if cls is set:
        for item in value:
            if item.__class__ not in atomic:
                return {
                    item if item.__class__ in atomic else _eager_item(item, translate)
                    for item in value
                }
        return value.copy()
    return cow_copy_value(value, translate)


def _eager_item(item: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Copy one non-atomic container member for ``_eager_copy``."""

    cls = item.__class__
    if cls is dict or cls is list or cls is set:
        return _eager_copy(item, translate)
    return cow_copy_value(item, translate)


class PooledCell:
    """Shared canonical cell for pooled duplicate/empty mutable containers.

    The M14 memory slice: the freeze-seam compaction replaces exact builtin
    mutable-container cells (``dict``/``list``/``set``/top-level
    ``defaultdict``) whose full content is provably immutable AND either
    empty or repeated across cells with ONE shared ``PooledCell`` per
    distinct content. The facade descriptors hydrate a fresh exact-type
    container for the reading row on first access and cache it back — the
    ``_FACT`` semantics: identity is stable across reads, per-row in-place
    mutation stays isolated, and an uninspected row retains no per-row
    container.

    ``prototype`` is a DETACHED deep copy taken at pool time (never one of
    the live cell values), so no external handle to a pre-pool container can
    reach it and every hydration is a consistent snapshot.
    """

    __slots__ = ("prototype",)

    def __init__(self, prototype: Any) -> None:
        """Bind the detached canonical prototype container."""

        self.prototype = prototype

    def hydrate(self) -> Any:
        """Return a fresh mutable container equal to the prototype.

        Immutable members (the only members the pool key admits) are shared
        by reference; nested exact builtin mutable containers are rebuilt
        per hydration, so no mutable state is ever shared across rows.
        """

        return _detached_container_copy(self.prototype)

    @classmethod
    def from_value(cls, value: Any) -> PooledCell:
        """Build a pooled cell around a DETACHED copy of ``value``."""

        return cls(_detached_container_copy(value))


def _detached_container_copy(value: Any) -> Any:
    """Deep-copy one exact builtin mutable container, sharing immutables."""

    cls = value.__class__
    if cls is defaultdict:
        fresh: Any = defaultdict(value.default_factory)
        for key, item in value.items():
            item_cls = item.__class__
            fresh[key] = (
                _eager_copy(item, None)
                if item_cls is dict or item_cls is list or item_cls is set
                else item
            )
        return fresh
    if not value:
        return cls()
    return _eager_copy(value, None)


class _SealedRowMajorOpRowStore(OpRowStore):
    """Sealed row-major store (under the transpose threshold).

    Installed by ``freeze()`` via class swap so the BUILDING hot path keeps
    zero per-write overhead. Post-seal writes still mutate rows in place
    (existing fork views hold row snapshots, so isolation is unaffected),
    but container writes must register in the fork-isolation index — a
    later fork's eager sweep would otherwise miss a container written after
    the index was built (parent ``op.field = [..]`` then in-place mutation
    after the next fork).
    """

    __slots__ = ()

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell in place, indexing post-seal container writes.

        Any write that could stale the cached sweep plan drops it (a new
        container, or replacement of an already-indexed cell); the next
        fork rebuilds the plan from the maintained index.
        """

        rows = self._rows
        assert rows is not None
        rows[row][fid] = value
        index = self._mutable_keys
        if index is not None:
            if _needs_eager_isolation(value):
                fid_rows = index.get(fid)
                if fid_rows is None:
                    fid_rows = index[fid] = set()
                fid_rows.add(row)
                self._sweep_plan = None
            elif self._sweep_plan is not None:
                fid_rows = index.get(fid)
                if fid_rows is not None and row in fid_rows:
                    self._sweep_plan = None

    def cell_del(self, row: int, fid: int) -> bool:
        """Delete one cell in place, dropping a staled sweep plan.

        Row-major deletes clear the cell in the live row list, so a cached
        plan referencing the deleted container would wrongly resurrect it
        in the next fork's overlay (and pin its value).
        """

        result = OpRowStore.cell_del(self, row, fid)
        if result and self._sweep_plan is not None:
            index = self._mutable_keys
            if index is not None:
                fid_rows = index.get(fid)
                if fid_rows is not None and row in fid_rows:
                    self._sweep_plan = None
        return result


#: Live write-audit collectors keyed by audited store id (M10 step
#: contracts). Populated only while a postprocess step audit is active.
_AUDIT_COLLECTORS: dict[int, set[int]] = {}

#: Per-window read collectors (design-ppdag-v3 combined audit): column ids
#: observed through ``cell_get`` while a COMBINED (read-recording) audit is
#: armed. Column-granular by design — never per ``(row, fid)`` (review note
#: N4: the matrix cost is measured, and row granularity buys nothing the
#: per-column classification uses).
_AUDIT_READS: dict[int, set[int]] = {}

#: Reads made inside a ``row_clone_scope`` — the mechanical row-clone access
#: kind (design-ppdag-v3 §2.4d). ``Op.copy()`` reads every schema column via
#: ``getattr``; recorded naively those would drown the findings report in
#: phantom per-column dependencies. Clone reads are legal iff the step's
#: ``row_effects`` contains ``"creates"`` and generate NO per-column edges
#: (ordering safety is carried by the two-sided row barrier). Clone WRITES
#: need no twin: ``Op.__init__`` delivers the cloned row through
#: ``store.adopt_row``, which bypasses cell interception entirely (verified;
#: review note N7). Deepcopy recursion into ``__repr__``/``__reduce__``
#: readers inside the clone loop also lands here, conservatively.
_AUDIT_CLONE_READS: dict[int, set[int]] = {}

#: Column ids with at least one CONTENT-EFFECTIVE intercepted write this
#: window (recording mode only): the effective-write evidence guard — a
#: permanent no-op write (``cell_set(c, old_value)``) cannot discharge a
#: read-before-write finding. In-place container mutations are judged by the
#: begin/end fingerprint diff instead (they never pass through ``cell_set``),
#: and a fingerprint change is content-effective by definition.
_AUDIT_WRITE_EFFECTS: dict[int, set[int]] = {}

#: Live ``row_clone_scope`` nesting depth per audited store id.
_CLONE_SCOPE_DEPTH: dict[int, int] = {}

#: Per-store ``(row_count, fingerprints)`` snapshots of mutable-container
#: cells captured at audit begin, diffed at audit end so IN-PLACE mutations
#: (invisible to the ``cell_set``/``cell_del`` interception) surface as
#: column writes. Rows born mid-window are excluded — whole-row creation is
#: a step's produces contract, not a column write.
_AUDIT_FINGERPRINTS: dict[int, tuple[int, dict[int, int]]] = {}

#: Rows RELEASED whole (op removal husking) during the live audit window,
#: keyed by audited store id. Whole-row release is the row-lifecycle twin of
#: row creation: its per-cell deletes are not column writes, so the audited
#: ``cell_del`` skips recording them and the fingerprint diff skips the row.
#: The release itself is still audited — ``end_cell_write_audit`` reports
#: the released-row count and the step contract must sanction removal
#: explicitly (``removes_rows``), so an unsanctioned whole-row removal
#: trips with a precise message instead of a wall of column names.
_AUDIT_ROW_RELEASES: dict[int, set[int]] = {}

#: Immutable empty fallback for the audited ``cell_del`` released-row probe.
_NO_RELEASES: frozenset[int] = frozenset()


def mark_op_row_released(store: Any, row: int) -> None:
    """Mark one row as released whole for the active audit window (if any).

    Called by the op-removal husking path BEFORE it deletes the row's
    cells. A no-op when the store is not under audit (including detached
    stores and fork views, which are never audited).
    """

    releases = _AUDIT_ROW_RELEASES.get(id(store))
    if releases is not None:
        releases.add(row)


def _value_fingerprint(value: Any) -> int:
    """Order-canonical recursive content fingerprint for audit snapshots.

    Unordered containers (``set``/``frozenset``/``dict``) hash their
    element fingerprints SORTED, so equal content always fingerprints
    equal regardless of iteration order — a ``hash(repr(...))`` snapshot
    was iteration-order sensitive (a mutate-and-revert that resized a
    set's table, or reordered equal dict keys, false-positively read as a
    write of the column). Ordered containers stay positional. Leaves keep
    the ``repr`` basis (in-place mutation anywhere inside still changes
    the fingerprint); mutables nested in NON-builtin custom objects remain
    the audit's disclosed residual. Env-gated enforcement-leg cost only:
    comparable to the former full-``repr`` walk (it recurses the same
    content, minus container string building; leaf ``repr`` — including
    tensor and record cells — still dominates).
    """

    cls = value.__class__
    if cls is dict:
        return hash(
            (
                1,
                tuple(
                    sorted(
                        hash((_value_fingerprint(key), _value_fingerprint(item)))
                        for key, item in value.items()
                    )
                ),
            )
        )
    if cls is list:
        return hash((2, tuple(_value_fingerprint(item) for item in value)))
    if cls is set or cls is frozenset:
        return hash((3, tuple(sorted(_value_fingerprint(item) for item in value))))
    if cls is tuple:
        return hash((4, tuple(_value_fingerprint(item) for item in value)))
    try:
        return hash((0, repr(value)))
    except Exception:
        # Fail CLOSED (R22-8): a shared sentinel here made every pair of
        # fingerprint-refusing values compare EQUAL, collapsing a real write
        # to "no effective write" and laundering the downstream read. The
        # object id keeps same-object comparisons stable within the audit
        # window while distinct refusing objects read as changed -- a false
        # alarm at worst, never a laundering pass.
        return id(value)


def _cell_content_fingerprint(value: Any) -> int | None:
    """Content fingerprint for one exact builtin mutable-container cell.

    Non-container values (and immutable views) return ``None`` — cell
    REPLACEMENT is already caught by the write interception. Containers
    whose content cannot be fingerprinted (cycles) conservatively
    fingerprint as their length so at least size changes are visible.
    """

    cls = value.__class__
    if cls is dict or cls is list or cls is set:
        try:
            return _value_fingerprint(value)
        except Exception:
            return len(value)
    return None


class _AuditedOpRowStore(OpRowStore):
    """Write-recording twin used ONLY during env-gated step audits.

    ``begin_cell_write_audit`` swaps a store's ``__class__`` to this subclass
    (layout-identical: empty ``__slots__``), so the un-audited hot path pays
    ZERO extra cost — no per-write branch exists on ``OpRowStore`` itself.
    The enforcing CI path keeps this write-only twin so it never pays a
    ``cell_get`` override; the read-recording combined twin below is armed
    only in read-audit modes. Swap pairs are disjoint per store; the two
    named silent-skip kinds (the ``is OpRowStore`` guard) are fork-backed
    ``OpStoreView``s and post-freeze ``_SealedRowMajorOpRowStore``s — sealing
    happens after step 20, outside every window, so the sealed skip is
    benign for the pipeline (disclosed residual, design-ppdag-v3 §1.4).
    """

    __slots__ = ()

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Record the written column id, then perform the write."""

        collector = _AUDIT_COLLECTORS.get(id(self))
        if collector is not None:
            collector.add(fid)
        OpRowStore.cell_set(self, row, fid, value)

    def cell_del(self, row: int, fid: int) -> bool:
        """Record the deleted column id, then perform the delete.

        Deletes belonging to a whole-row release (op removal husking) are
        row-lifecycle events, not column writes: they are excluded here and
        accounted through the released-row count instead.
        """

        collector = _AUDIT_COLLECTORS.get(id(self))
        if collector is not None and row not in _AUDIT_ROW_RELEASES.get(id(self), _NO_RELEASES):
            collector.add(fid)
        return OpRowStore.cell_del(self, row, fid)


def _write_is_content_effective(old: Any, new: Any) -> bool:
    """Classify one intercepted write as content-effective or no-op.

    FINDING-favoring (design-ppdag-v3 §2.4 guard 2): under guard 2 an
    EFFECTIVE verdict is the lenient one — effective writes discharge
    read-before-write findings — so a write counts effective only when that
    is PROVABLE: first write over the missing placeholder, class change,
    scalar inequality, or a builtin-container content-fingerprint change.
    Everything ambiguous defaults to NO-OP, which can only keep a writer in
    the permanent no-op table and keep a downstream read visible as a
    finding — a false alarm at worst, never a laundering pass. Honest
    residual: a genuinely value-changing rich-object rewrite (e.g. a tensor
    cell overwritten with a distinct same-class tensor holding different
    contents) is classified no-op here, so reads it feeds must be
    discharged by review (pinned finding/probe or a matrix axis whose
    first-write is observed), never by effectiveness evidence.
    """

    if old is new:
        # Rebinding the identical object writes nothing the cell did not
        # already hold; in-place mutation of it is judged by the begin/end
        # fingerprint diff, never here (review note N3).
        return False
    if old is _MISSING:
        return True
    old_cls = old.__class__
    if old_cls is not new.__class__:
        return True
    if old_cls in (bool, int, float, str, bytes, type(None)):
        return old != new
    if old_cls is dict or old_cls is list or old_cls is set:
        return _cell_content_fingerprint(old) != _cell_content_fingerprint(new)
    if old_cls is tuple or old_cls is frozenset:
        try:
            return _value_fingerprint(old) != _value_fingerprint(new)
        except Exception:
            # Fail CLOSED (R22-8): fingerprint machinery failure must read
            # as an EFFECTIVE write, never collapse to "no-op" -- a no-op
            # classification here would let the write discharge nothing
            # while hiding the step's real effect from the audit.
            return True
    return False


class _CombinedAuditOpRowStore(OpRowStore):
    """Read+write recording twin for the recording/read-audit modes.

    Same class-swap mechanics as ``_AuditedOpRowStore``; additionally
    records column-granular reads through ``cell_get`` (the one facade read
    chokepoint) and tags each intercepted write as content-effective or
    no-op. The old-value probe inside ``cell_set`` reads ``self._rows``
    directly — NEVER through ``cell_get`` — so the effectiveness guard
    cannot record a phantom read of the written column (review note N3).
    """

    __slots__ = ()

    def cell_get(self, row: int, fid: int) -> Any:
        """Record the read column id, then perform the read.

        Reads on rows already marked RELEASED this window (op removal
        husking walks and clears every set cell) are row-lifecycle events
        like the husking deletes — recording them would report the whole
        schema as step-3/6 reads. Reads BEFORE the release mark (e.g. the
        orphan-record construction) still record normally.
        """

        store_id = id(self)
        if _CLONE_SCOPE_DEPTH.get(store_id, 0):
            collector = _AUDIT_CLONE_READS.get(store_id)
        else:
            collector = _AUDIT_READS.get(store_id)
        if collector is not None and row not in _AUDIT_ROW_RELEASES.get(store_id, _NO_RELEASES):
            collector.add(fid)
        return OpRowStore.cell_get(self, row, fid)

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Record the write and its content-effectiveness, then write."""

        store_id = id(self)
        collector = _AUDIT_COLLECTORS.get(store_id)
        if collector is not None:
            collector.add(fid)
            effects = _AUDIT_WRITE_EFFECTS.get(store_id)
            if effects is not None and fid not in effects:
                rows = self._rows
                old = rows[row][fid] if rows is not None else _MISSING
                if _write_is_content_effective(old, value):
                    effects.add(fid)
        OpRowStore.cell_set(self, row, fid, value)

    def cell_del(self, row: int, fid: int) -> bool:
        """Record the deleted column id, then perform the delete.

        A delete of a previously-set cell is content-effective by
        definition; husking deletes stay row-lifecycle events.
        """

        store_id = id(self)
        collector = _AUDIT_COLLECTORS.get(store_id)
        if collector is not None and row not in _AUDIT_ROW_RELEASES.get(store_id, _NO_RELEASES):
            collector.add(fid)
            result = OpRowStore.cell_del(self, row, fid)
            if result:
                effects = _AUDIT_WRITE_EFFECTS.get(store_id)
                if effects is not None:
                    effects.add(fid)
            return result
        return OpRowStore.cell_del(self, row, fid)


@contextmanager
def row_clone_scope(store: Any) -> Iterator[None]:
    """Tag reads inside the scope as row-clone reads (category d).

    Free when unarmed (review note N7): ``Op.copy()`` also runs on pickle,
    fork-fallback, and preview paths where no audit exists — the fast path
    is one dict lookup. Detached stores and fork views are never audited,
    so their ids simply miss.
    """

    store_id = id(store)
    if store_id not in _AUDIT_READS:
        yield
        return
    _CLONE_SCOPE_DEPTH[store_id] = _CLONE_SCOPE_DEPTH.get(store_id, 0) + 1
    try:
        yield
    finally:
        depth = _CLONE_SCOPE_DEPTH.get(store_id, 1) - 1
        if depth:
            _CLONE_SCOPE_DEPTH[store_id] = depth
        else:
            _CLONE_SCOPE_DEPTH.pop(store_id, None)


class StepAuditResult(NamedTuple):
    """One closed audit window's observations (column NAMES, not fids)."""

    written_columns: set[str]
    released_rows: int
    read_columns: set[str]
    clone_read_columns: set[str]
    effective_write_columns: set[str]


def begin_cell_write_audit(store: OpRowStore, *, record_reads: bool = False) -> None:
    """Start recording column writes (and optionally reads) on ``store``.

    Ordering is load-bearing (design-ppdag-v3 §2.5, review 5): the
    mutable-container fingerprint sweep runs arbitrary ``__repr__`` code at
    the leaves, so the snapshot is built FIRST and the class swap happens
    LAST — arming before fingerprinting would let any repr path reaching
    ``cell_get`` self-pollute the window's read record and break
    ``observed <= declared``.
    """

    fingerprints: dict[int, int] = {}
    rows = store._rows
    baseline_rows = 0
    if rows is not None:
        baseline_rows = len(rows)
        n_fields = store.layout.n_fields
        for row_index, row_cells in enumerate(rows):
            base = row_index * n_fields
            for fid, value in enumerate(row_cells):
                fingerprint = _cell_content_fingerprint(value)
                if fingerprint is not None:
                    fingerprints[base + fid] = fingerprint
    _AUDIT_FINGERPRINTS[id(store)] = (baseline_rows, fingerprints)
    _AUDIT_COLLECTORS.setdefault(id(store), set())
    _AUDIT_ROW_RELEASES.setdefault(id(store), set())
    if record_reads:
        _AUDIT_READS.setdefault(id(store), set())
        _AUDIT_CLONE_READS.setdefault(id(store), set())
        _AUDIT_WRITE_EFFECTS.setdefault(id(store), set())
    if store.__class__ is OpRowStore:
        store.__class__ = _CombinedAuditOpRowStore if record_reads else _AuditedOpRowStore


def end_cell_write_audit(store: OpRowStore) -> StepAuditResult:
    """Stop recording; return the window's observations by column NAME.

    Columns whose mutable-container cells changed CONTENT since the audit
    began count as written (and content-effective) even without an
    intercepted ``cell_set`` — the in-place mutation path, judged ONLY by
    this begin/end fingerprint diff (review note N3). Rows released whole
    (op removal husking) are excluded from the column channels and reported
    as ``released_rows``, checked against the step contract's explicit
    ``row_effects`` sanction. Collectors and the snapshot are popped BEFORE
    the diff runs, so the diff's own reads cannot pollute the record.
    """

    store_id = id(store)
    observed = _AUDIT_COLLECTORS.pop(store_id, set())
    released = _AUDIT_ROW_RELEASES.pop(store_id, set())
    snapshot = _AUDIT_FINGERPRINTS.pop(store_id, None)
    reads = _AUDIT_READS.pop(store_id, set())
    clone_reads = _AUDIT_CLONE_READS.pop(store_id, set())
    tracked_effects = _AUDIT_WRITE_EFFECTS.pop(store_id, None)
    effects = tracked_effects if tracked_effects is not None else set()
    _CLONE_SCOPE_DEPTH.pop(store_id, None)
    if store.__class__ in (_AuditedOpRowStore, _CombinedAuditOpRowStore):
        store.__class__ = OpRowStore  # type: ignore[assignment]
    layout_names = store.layout.names
    rows = store._rows
    # The fingerprint diff only applies to stores this audit window ARMED
    # (a store born mid-window — the step-0 materialize ingress — has no
    # baseline and its whole-row bulk writes are its declared contract) and
    # to rows that existed at begin (row creation is not a column write;
    # released rows' emptied cells are not container mutations either).
    if rows is not None and snapshot is not None:
        baseline_rows, baseline = snapshot
        n_fields = store.layout.n_fields
        for row_index in range(min(baseline_rows, len(rows))):
            if row_index in released:
                continue
            base = row_index * n_fields
            for fid, value in enumerate(rows[row_index]):
                # Write-only mode keeps the historical skip on intercepted
                # columns; effects tracking additionally diffs intercepted-
                # but-not-yet-effective columns so an in-place mutation
                # behind a no-op rebind still reads as content-effective.
                if fid in observed and (tracked_effects is None or fid in effects):
                    continue
                if _cell_content_fingerprint(value) != baseline.get(base + fid):
                    observed.add(fid)
                    effects.add(fid)
    return StepAuditResult(
        {layout_names[fid] for fid in observed},
        len(released),
        {layout_names[fid] for fid in reads},
        {layout_names[fid] for fid in clone_reads},
        {layout_names[fid] for fid in effects},
    )


class DetachedOpStore:
    """Single-row op store for detached facades (copy/pickle/fork/preview).

    Parameters
    ----------
    layout:
        Shared field layout.
    """

    __slots__ = ("_cells", "layout")

    #: Detached rows never carry CSR-backed relations or shared-fact cells
    #: (class-level constants so the facade descriptors can probe both store
    #: kinds uniformly).
    dataflow_edges = None
    ref_labels = None
    fact_blocks = None

    def __init__(self, layout: OpStoreLayout) -> None:
        """Create an empty single-row store."""

        self.layout = layout
        self._cells: list[Any] = [_MISSING] * layout.n_fields

    def adopt_row(self, cells: list[Any]) -> int:
        """Adopt one pre-built layout-ordered cell list as the single row."""

        if len(cells) != self.layout.n_fields:
            raise ValueError("row cell count does not match the layout")
        self._cells = cells
        return 0

    def __len__(self) -> int:
        """Return the row count (always one)."""

        return 1

    @property
    def frozen(self) -> bool:
        """Return ``False``: detached rows never freeze."""

        return False

    def cell_get(self, row: int, fid: int) -> Any:
        """Return one cell value, or ``_MISSING`` when unset."""

        return self._cells[fid]

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell."""

        self._cells[fid] = value

    def cell_del(self, row: int, fid: int) -> bool:
        """Delete one cell; return whether it was previously set."""

        cells = self._cells
        if cells[fid] is _MISSING:
            return False
        cells[fid] = _MISSING
        return True

    def items(self, row: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(field_name, value)`` for every set cell in layout order."""

        names = self.layout.names
        cells = self._cells
        for fid, name in enumerate(names):
            value = cells[fid]
            if value is not _MISSING:
                yield name, value

    def compacted_singleton(self, row: int, fid: int, value: Any) -> bool:
        """Return ``False``: detached rows carry no compaction registry.

        ``detach_record`` decodes compacted singleton-label cells while
        copying, so a detached row never holds the internal encoding.
        """

        return False

    def retained_bytes(self) -> int:
        """Return shallow structural bytes retained by this store."""

        import sys

        return sys.getsizeof(self) + sys.getsizeof(self._cells)


#: Exact types returned uncopied (and untranslated) by the COW copier.
_COW_ATOMIC = frozenset({str, int, float, bool, bytes, complex, type(None)})


def cow_copy_value(value: Any, translate: Callable[[Any], Any] | None) -> Any:
    """Structurally copy one COW-read value for fork-side isolation.

    Exact builtin containers are rebuilt (so fork-side in-place mutation can
    never reach the shared base); ``translate`` maps parent record facades to
    their fork facades (returning ``None`` for non-records); every other
    object — tensors, callables, interned immutables, quantity subclasses —
    is returned by identity, preserving the fork's payload-sharing contract.
    All-identity tuples/frozensets return the original object so interned
    immutable views stay shared.
    """

    cls = value.__class__
    if cls in _COW_ATOMIC:
        return value
    if cls is dict:
        return {
            cow_copy_value(key, translate): cow_copy_value(item, translate)
            for key, item in value.items()
        }
    if cls is list:
        return [cow_copy_value(item, translate) for item in value]
    if cls is set:
        return {cow_copy_value(item, translate) for item in value}
    if cls is tuple or cls is frozenset:
        copied = [cow_copy_value(item, translate) for item in value]
        for original, item_copy in zip(value, copied, strict=True):
            if original is not item_copy:
                return cls(copied)
        return value
    if translate is not None:
        mapped = translate(value)
        if mapped is not None:
            return mapped
    return value


class _ViewFactBlocks:
    """Fact-block adapter isolating hydrated containers for one fork view."""

    __slots__ = ("_base", "_view")

    def __init__(self, base: Any, view: OpStoreView) -> None:
        """Bind the base fact blocks and the owning view."""

        self._base = base
        self._view = view

    def hydrate(self, row: int, name: str) -> Any:
        """Hydrate one shared fact, translating record members for the fork."""

        return cow_copy_value(self._base.hydrate(row, name), self._view.record_translator)


class OpStoreView:
    """Per-fork COW view over one sealed base store (the M11 fork substrate).

    The view shares the base's frozen storage and isolates everything mutable:

    * Fork writes and deletes land in the view's own overlay, never the base.
    * The base's post-freeze overlay is SNAPSHOT at construction (row-major
      sealed bases snapshot their row lists instead), so parent writes after
      the fork stay invisible in both directions.
    * Exact builtin mutable containers are eagerly copied into the view
      overlay at fork time (``isolate_mutable_cells``, run by the fork
      builder once the record translator is installed), so isolation holds
      in BOTH directions from fork time — the copy-on-first-read in
      ``_isolate`` remains as the read-cache backstop (tensors/callables
      inside stay shared by identity — the payload-sharing fork contract).
      The sweep is sparse: it visits only the base's cached mutable-cell
      index plus the base-overlay snapshot, never the whole store.
    * ``GroupRef`` cells translate to the fork core's cloned group tables, so
      removal scrub on either trace never reaches the other.
    * Record facades inside containers and hydrated fact blocks translate
      through ``record_translator`` (installed by the fork builder) to the
      fork's own facades.

    Non-builtin mutable cell values (custom objects) are shared by identity —
    the same residual the shallow fork path already accepted for
    replay-unaffected fields.
    """

    __slots__ = (
        "__weakref__",
        "_base_overlay",
        "_base_rows",
        "_group_refs",
        "_group_tables",
        "_overlay",
        "_record_translator_ref",
        "base",
        "fact_blocks",
    )

    @property
    def record_translator(self) -> Callable[[Any], Any] | None:
        """Return the fork's record translator while its owning core lives.

        Held WEAKLY (the translator is anchored on the owning fork
        ``TraceCore``): a retained fork ``Op`` must not root the whole fork
        graph through ``view -> translator -> fork shells -> fork Trace`` --
        that chain kept the fork core alive forever, so the last-owner
        payload eviction could never fire. After the fork dies, reads
        proceed untranslated (dead-trace view semantics).
        """

        ref = self._record_translator_ref
        return ref() if ref is not None else None

    @record_translator.setter
    def record_translator(self, translator: Callable[[Any], Any] | None) -> None:
        """Bind the translator weakly (``None`` clears it)."""

        self._record_translator_ref = None if translator is None else weakref.ref(translator)

    def __init__(
        self,
        base: OpRowStore | OpStoreView,
        group_tables: dict[int, MembershipGroups] | None = None,
    ) -> None:
        """Snapshot ``base`` (which must be sealed) into a COW view.

        ``base`` may itself be an ``OpStoreView`` (fork of a fork): the new
        view flattens onto the ROOT store, snapshotting the parent view's
        effective overlay (its private base snapshot plus its own writes) —
        the parent view never mutates its base snapshot, so sharing the
        row snapshot list is safe.
        """

        if not base.frozen:
            raise RuntimeError("OpStoreView requires a sealed base store")
        self.base: OpRowStore
        self._base_rows: list[list[Any]] | None
        self._base_overlay: dict[int, Any]
        if isinstance(base, OpStoreView):
            self.base = base.base
            self._base_rows = base._base_rows
            self._base_overlay = {**base._base_overlay, **base._overlay}
        else:
            self.base = base
            base._cow_shared = True
            base_rows = base._rows
            if base_rows is not None:
                # Sealed row-major base: writes/deletes mutate rows in
                # place, so the view snapshots the row lists (cheap under
                # the transpose threshold) and ignores the live cells
                # thereafter.
                self._base_rows = [list(row_cells) for row_cells in base_rows]
                self._base_overlay = {}
            else:
                self._base_rows = None
                self._base_overlay = dict(base._overlay)
        self._overlay: dict[int, Any] = {}
        self._group_tables = group_tables
        self._group_refs: dict[tuple[int, int], GroupRef] = {}
        # Last-owner payload eviction must reach this view's snapshot
        # surfaces (a retained fork Op reads through them, never the live
        # base rows), so the root store tracks its views weakly.
        self.base._payload_views.append(weakref.ref(self))
        self.record_translator: Callable[[Any], Any] | None = None
        self.fact_blocks = (
            _ViewFactBlocks(self.base.fact_blocks, self)
            if self.base.fact_blocks is not None
            else None
        )

    def __len__(self) -> int:
        """Return the base row count."""

        return len(self.base)

    @property
    def layout(self) -> OpStoreLayout:
        """Return the shared field layout."""

        return self.base.layout

    @property
    def dataflow_edges(self) -> Any:
        """Return the shared (frozen) dataflow edge table."""

        return self.base.dataflow_edges

    @property
    def ref_labels(self) -> Any:
        """Return the shared row -> reference-label table."""

        return self.base.ref_labels

    @property
    def frozen(self) -> bool:
        """Return ``True``: views only exist over sealed bases."""

        return True

    def new_row(self) -> int:
        """Refuse: fork views never append rows."""

        raise RuntimeError("op store view is frozen; no new rows may be appended")

    def adopt_row(self, cells: list[Any]) -> int:
        """Refuse: fork views never adopt rows."""

        raise RuntimeError("op store view is frozen; no new rows may be appended")

    def rows_building(self) -> None:
        """Return ``None``: a view is never in the building phase."""

        return

    def _translate_group_ref(self, ref: GroupRef) -> GroupRef:
        """Return the fork-side ref for one shared group cell."""

        tables = self._group_tables
        if tables is None:
            return ref
        clone = tables.get(id(ref.groups))
        if clone is None:
            return ref
        key = (id(ref.groups), ref.group_id)
        fork_ref = self._group_refs.get(key)
        if fork_ref is None:
            fork_ref = GroupRef(clone, ref.group_id)
            self._group_refs[key] = fork_ref
        return fork_ref

    def _isolate(self, key: int, value: Any) -> Any:
        """Isolate one base-read value, caching fork copies in the overlay."""

        if value is _MISSING or value is _CSR or value is _FACT:
            return value
        cls = value.__class__
        if cls is PooledCell:
            # Shared-by-construction (immutable prototype): the facade
            # descriptor hydrates a fork-local container on first read.
            return value
        if cls in _COW_ATOMIC:
            return value
        if cls is GroupRef:
            fork_ref = self._translate_group_ref(value)
            if fork_ref is not value:
                self._overlay[key] = fork_ref
            return fork_ref
        if cls is dict or cls is list or cls is set or cls is tuple or cls is frozenset:
            copied = cow_copy_value(value, self.record_translator)
            self._overlay[key] = copied
            return copied
        translate = self.record_translator
        if translate is not None:
            mapped = translate(value)
            if mapped is not None:
                self._overlay[key] = mapped
                return mapped
        return value

    def cell_get(self, row: int, fid: int) -> Any:
        """Return one cell value with fork-side isolation applied."""

        key = row * self.base.layout.n_fields + fid
        value = self._overlay.get(key, _NO_OVERLAY)
        if value is not _NO_OVERLAY:
            return value
        value = self._base_overlay.get(key, _NO_OVERLAY)
        if value is _NO_OVERLAY:
            base_rows = self._base_rows
            if base_rows is not None:
                value = base_rows[row][fid]
            else:
                columns = self.base._columns
                assert columns is not None
                value = columns[fid].get(row)
        return self._isolate(key, value)

    def cell_set(self, row: int, fid: int, value: Any) -> None:
        """Write one cell into the fork overlay."""

        self._overlay[row * self.base.layout.n_fields + fid] = value

    def cell_del(self, row: int, fid: int) -> bool:
        """Tombstone one cell in the fork overlay; report prior presence."""

        if self.cell_get(row, fid) is _MISSING:
            return False
        self._overlay[row * self.base.layout.n_fields + fid] = _MISSING
        return True

    def compacted_singleton(self, row: int, fid: int, value: Any) -> bool:
        """Delegate to the base registry (identity keeps view writes exact).

        A fork write lands in the view overlay as a DIFFERENT object, so the
        base's identity check already refuses it; a fork-side decode caches
        its hydrated list into the view overlay, leaving the parent's
        compacted cell untouched.
        """

        return self.base.compacted_singleton(row, fid, value)

    def items(self, row: int) -> Iterator[tuple[str, Any]]:
        """Yield ``(field_name, value)`` for every set cell in layout order.

        ``PooledCell`` cells hydrate fresh and compacted singleton-label
        cells decode fresh (never the internal encoding), matching the
        base-store ``items`` contract.
        """

        names = self.base.layout.names
        for fid, name in enumerate(names):
            value = self.cell_get(row, fid)
            if value is not _MISSING:
                if value.__class__ is PooledCell:
                    value = value.hydrate()
                elif value.__class__ is str and self.compacted_singleton(row, fid, value):
                    value = [value]
                yield name, value

    def isolate_mutable_cells(self) -> None:
        """Eagerly isolate every mutable-container cell into the fork overlay.

        Called once by the fork builder AFTER the record translator is
        installed. Copy-on-first-read alone left a window where a PARENT's
        in-place container mutation between fork time and the fork's first
        read of that cell leaked into the fork; pre-isolating restores the
        deepcopy fork's snapshot semantics in both directions (fork writes
        were already overlay-isolated).

        The sweep is SPARSE (sol closure review, F4 perf): it visits only
        the base store's cached mutable-cell index — pre-flattened into a
        per-store sweep PLAN (``_build_sweep_plan``, F11 small-trace
        constant) so the per-fork loop does no backing fetch or per-cell
        classification — plus this view's base-overlay snapshot, and copies
        only the minimal leak-closure set (``_needs_eager_isolation``).
        Record facades,
        ``GroupRef`` cells, and interned immutable views stay lazy on first
        read through ``_isolate`` — translation is not mutation, so
        deferring it cannot leak parent state. Tensors/callables inside
        copied containers stay shared by identity (the payload-sharing
        fork contract).
        """

        base = self.base
        plan = base._sweep_plan
        if plan is None:
            if base._mutable_keys is None:
                base._mutable_keys = _build_mutable_key_index(base)
            plan = base._sweep_plan = _build_sweep_plan(base)
        overlay = self._overlay
        base_overlay = self._base_overlay
        translate = self.record_translator
        # Post-freeze writes visible at fork time (the base-overlay snapshot;
        # for a fork-of-fork this includes the parent view's own writes).
        # Checked FIRST: an overlay tombstone/replacement supersedes the
        # plan's view of the frozen backing.
        for key, value in base_overlay.items():
            if key not in overlay and _needs_eager_isolation(value):
                overlay[key] = _eager_copy(value, translate)
        # Frozen cells, pre-resolved by the cached sweep plan (values and
        # copy kinds classified once at plan build — see ``_build_sweep_plan``
        # for the staleness contract). Empty containers dominate the census
        # (~11 of ~27 candidate cells per op), hence the dedicated bare
        # class-allocation loop.
        alloc_entries, copy_entries = plan
        for key, cls, value in alloc_entries:
            if key in overlay or key in base_overlay:
                continue
            # Emptiness re-check: an in-place empty->non-empty mutation of
            # the aliased container fires no maintenance hook, so the alloc
            # classification alone would resurrect an empty cell here.
            overlay[key] = _eager_copy(value, translate) if value else cls()
        for key, value, deep in copy_entries:
            if key in overlay or key in base_overlay:
                continue
            overlay[key] = (
                cow_copy_value(value, translate) if deep else _eager_copy(value, translate)
            )

    def retained_bytes(self) -> int:
        """Return shallow structural bytes retained by this view alone."""

        import sys

        total = sys.getsizeof(self) + sys.getsizeof(self._overlay)
        total += sys.getsizeof(self._base_overlay)
        if self._base_rows is not None:
            total += sys.getsizeof(self._base_rows)
            for row_cells in self._base_rows:
                total += sys.getsizeof(row_cells)
        return total
