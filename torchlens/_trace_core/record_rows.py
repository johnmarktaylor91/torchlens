"""Generic store-backed record facades for the non-Op record kinds (M8).

Converts a dict-backed record class (``Param``, ``Buffer``,
``FuncCallLocation``, ``ModuleCall``, ``Module``) into a row facade over the
columnar row stores: every declared stored field becomes a class-level data
descriptor reading and writing one row cell, so the per-record ``__dict__``
survives only for the two store-binding keys and undeclared user attributes
(FORK-7 default: arbitrary user attributes stay supported).

Storage reuses the M5 machinery verbatim: ``OpRowStore`` (layout-agnostic row
store, row-major while building, sealed after postprocess) holds the
per-trace kind tables registered in ``TraceCore.kind_rows``; a record built
outside a captured run (direct construction, pickle restore, fork shells)
binds a single-row ``DetachedOpStore`` on its first field write. The torch
build passes adopt freshly built records into the per-trace table
(``adopt_records``); preview backends stay detached, exactly like preview
``Op`` records.

Cell semantics match ``Op``'s facade: an unset cell reads as the exact
``AttributeError`` a missing ``__dict__`` key raised, writes store the object
reference unchanged (in-place mutation of a stored container stays visible to
every reader), and deletes release the cell.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .op_store import _MISSING, DetachedOpStore, OpRowStore, OpStoreLayout, PooledCell

#: Instance-dict key holding the backing store (kept out of state streams).
CORE_KEY = "_tl_core"

#: Instance-dict key holding the record's row id in the backing store.
ROW_KEY = "_tl_row"


class RecordCellField:
    """Data descriptor for one declared stored field of a facade record."""

    __slots__ = ("_name", "_fid")

    def __init__(self, name: str, fid: int) -> None:
        """Bind the descriptor to its field name and layout column id."""

        self._name = name
        self._fid = fid

    def __repr__(self) -> str:
        """Return a debugging repr naming the stored field."""

        return f"<record field descriptor {self._name!r}>"

    def __get__(self, record: Any, objtype: Any = None) -> Any:
        """Return the row cell value; unset cells raise ``AttributeError``.

        A ``PooledCell`` (M14 duplicate/empty-container pooling) hydrates a
        fresh exact-type container on first read and caches it back, so
        identity is stable across reads and per-row in-place mutation stays
        isolated. A compacted singleton-label cell (M14 slice 2: the exact
        str object registered by the freeze-seam pass) decodes to a fresh
        one-element list the same way — identity-stable after the first
        read, per-row mutation isolated, uninspected rows retain no list.
        """

        if record is None:
            return self
        instance_dict = record.__dict__
        store = instance_dict.get(CORE_KEY)
        if store is None:
            raise AttributeError(self._name)
        row = instance_dict[ROW_KEY]
        value = store.cell_get(row, self._fid)
        if value is _MISSING:
            raise AttributeError(self._name)
        if value.__class__ is PooledCell:
            value = value.hydrate()
            store.cell_set(row, self._fid, value)
        elif value.__class__ is str and store.compacted_singleton(row, self._fid, value):
            value = [value]
            store.cell_set(row, self._fid, value)
        return value

    def __set__(self, record: Any, value: Any) -> None:
        """Write the row cell, binding a detached store on first write."""

        instance_dict = record.__dict__
        store = instance_dict.get(CORE_KEY)
        if store is None:
            store = DetachedOpStore(type(record)._TL_LAYOUT)
            instance_dict[CORE_KEY] = store
            instance_dict[ROW_KEY] = 0
        store.cell_set(instance_dict[ROW_KEY], self._fid, value)

    def __delete__(self, record: Any) -> None:
        """Delete the row cell; a second delete raises ``AttributeError``."""

        instance_dict = record.__dict__
        store = instance_dict.get(CORE_KEY)
        if store is None or not store.cell_del(instance_dict[ROW_KEY], self._fid):
            raise AttributeError(self._name)


def install_record_facade(cls: type, stored_names: tuple[str, ...]) -> OpStoreLayout:
    """Install cell descriptors for ``stored_names`` on ``cls``.

    Binds the shared layout as ``cls._TL_LAYOUT`` and refuses to overwrite an
    existing class attribute (a collision with a hand-written ``@property``
    would silently change public behavior, so it fails at import time) —
    EXCEPT a plain ``@dataclass`` field default: dataclass processing leaves
    simple defaults as class attributes, but the generated ``__init__`` bakes
    them into its own signature and never reads the class attribute again, so
    replacing one with a descriptor is behavior-preserving.
    """

    layout = OpStoreLayout(stored_names)
    setattr(cls, "_TL_LAYOUT", layout)
    for name in stored_names:
        existing = vars(cls).get(name)
        if existing is not None and (
            callable(existing)
            or hasattr(existing, "__get__")
            or isinstance(existing, (classmethod, staticmethod, property))
        ):
            raise RuntimeError(
                f"{cls.__name__} stored field {name!r} collides with an existing class attribute"
            )
        setattr(cls, name, RecordCellField(name, layout.fid_by_name[name]))
    return layout


def record_state_items(record: Any) -> Iterator[tuple[str, Any]]:
    """Yield ``(field_name, value)`` state pairs for one facade record.

    Declared stored fields come first in layout order (unset cells skipped),
    followed by undeclared instance attributes in insertion order — the same
    complete state the dict-era ``__dict__`` enumeration yielded.

    Instance-dict entries that SHADOW a declared stored field are skipped:
    the cell descriptor is a data descriptor, so such a shadow is unreachable
    through attribute access, and streaming it would let a dead ``__dict__``
    write silently replace the live cell value on restore (the store cell is
    the single truth for declared fields).
    """

    instance_dict = record.__dict__
    store = instance_dict.get(CORE_KEY)
    declared = type(record)._TL_LAYOUT.fid_by_name
    if store is not None:
        row = instance_dict[ROW_KEY]
        for name, value in store.items(row):
            # Decode the M14 storage encodings exactly like the cell
            # descriptor does: pickle state must carry real containers, never
            # a raw PooledCell (restore refuses it as corrupted state) or a
            # compacted singleton-label str (restore would coerce it to a
            # list of CHARACTERS where a label list is declared).
            if value.__class__ is PooledCell:
                value = value.hydrate()
            elif value.__class__ is str and store.compacted_singleton(row, declared[name], value):
                value = [value]
            yield name, value
    for name, value in instance_dict.items():
        if name is CORE_KEY or name is ROW_KEY or name in declared:
            continue
        yield name, value


def record_state_restore(record: Any, mapping: Any) -> None:
    """Install a state mapping onto a facade record.

    Declared names route through their cell descriptors (binding a detached
    store on a bare ``state_new`` shell); undeclared names land in
    ``__dict__`` directly — including names shadowed by read-only properties,
    which the dict-era ``__dict__.update`` stored as inert dead weight.
    """

    cls = type(record)
    fid_by_name = cls._TL_LAYOUT.fid_by_name
    instance_dict = record.__dict__
    store = instance_dict.get(CORE_KEY)
    if store is None:
        store = DetachedOpStore(cls._TL_LAYOUT)
        instance_dict[CORE_KEY] = store
        instance_dict[ROW_KEY] = 0
    row = instance_dict[ROW_KEY]
    for name, value in mapping.items():
        fid = fid_by_name.get(name)
        if fid is not None:
            store.cell_set(row, fid, value)
        else:
            instance_dict[name] = value


def adopt_records(core: Any, kind: str, records: Any) -> None:
    """Adopt detached facade records into the per-trace kind table.

    Each record's single detached row moves into ``core.kind_rows[kind]``
    (created on first use, sharing the class layout); the record rebinds to
    the shared store. Records already bound to a shared store are left
    untouched (idempotent for re-entrant build passes).
    """

    adopt_rows(core.kind_rows, kind, records)


def adopt_rows(store_registry: dict, kind: str, records: Any) -> None:
    """Adopt detached facade records into ``store_registry[kind]``.

    The registry-level twin of ``adopt_records`` used by owners other than
    the forward core's kind tables (the M9 backward epochs).
    """

    records = list(records)
    if not records:
        return
    layout = type(records[0])._TL_LAYOUT
    store = store_registry.get(kind)
    if store is None:
        store = OpRowStore(layout)
        store_registry[kind] = store
    elif store.frozen:
        # A refresh re-runs the build passes after the first capture sealed
        # the table; records built by the re-run stay detached-backed (the
        # dict-era equivalent of their fresh per-record dicts).
        return
    for record in records:
        instance_dict = record.__dict__
        bound = instance_dict.get(CORE_KEY)
        if bound is None:
            instance_dict[CORE_KEY] = store
            instance_dict[ROW_KEY] = store.new_row()
            continue
        if isinstance(bound, DetachedOpStore):
            row = store.adopt_row(bound._cells)
            instance_dict[CORE_KEY] = store
            instance_dict[ROW_KEY] = row


class BackwardEpoch:
    """One atomic backward-projection generation (M9).

    Bound by the projection materializer AFTER a successful projection: a
    full rebuild atomically replaces the core's epoch list with one fresh
    epoch; a clean tail fold extends the live epoch in place. ``revision``
    and ``watermark`` mirror the trace-side lazy invalidation fields
    (``_backward_projection_revision`` / ``_backward_projection_event_count``)
    — the invalidation SEMANTICS stay on the trace, the epoch records the
    generation its rows belong to. ``stores`` holds the per-kind row stores
    (``grad_fn`` / ``grad_fn_call`` / ``backward_pass``) backing the
    projection's record facades; they stay unsealed because a later fold may
    append rows.
    """

    __slots__ = ("revision", "watermark", "stores")

    def __init__(self) -> None:
        """Create an empty epoch."""

        self.revision: Any = None
        self.watermark: Any = None
        self.stores: dict[str, Any] = {}


def detach_record(record: Any) -> None:
    """Rebind a record to a fresh detached single-row copy of its cells.

    Used when a record leaves its trace (removal husks, cleanup) so a
    user-held record cannot pin the trace's shared kind table.
    """

    instance_dict = record.__dict__
    store = instance_dict.get(CORE_KEY)
    if store is None or isinstance(store, DetachedOpStore):
        return
    row = instance_dict[ROW_KEY]
    layout = store.layout
    cells = [store.cell_get(row, fid) for fid in range(layout.n_fields)]
    for fid, value in enumerate(cells):
        # Decode compacted singleton-label cells while copying: a detached
        # record leaves its trace, so the internal encoding (whose decode
        # needs the source store's registry) must never travel with it.
        if value.__class__ is str and store.compacted_singleton(row, fid, value):
            cells[fid] = [value]
    detached = DetachedOpStore(layout)
    detached.adopt_row(cells)
    instance_dict[CORE_KEY] = detached
    instance_dict[ROW_KEY] = 0
