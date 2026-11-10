"""Freeze-seam Op metadata compaction.

``compact_op_metadata`` pools repeated immutable Op metadata onto shared
instances. It used to live on ``Trace`` as ``_compact_op_metadata``; it
moved here as a module function invoked from the core freeze seam
(torch postprocess) and the preview-backend finalizers, so the Trace surface
carries no compaction method and the freeze owns the compaction role
(trace_core_design.md section 3.3).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._quantities import Duration
from .._trace_core.cell_states import LazyCell, settle_default_cells
from .._trace_core.op_store import _MISSING, PooledCell
from ._op_container_census import _EMPTY_CONTAINER_POOL_KEYS

if TYPE_CHECKING:
    from .trace import Trace

# Traces whose Op metadata has already been pooled by ``compact_op_metadata``.
# Held weakly and OFF the Trace itself so no new field enters ``__dict__``,
# pickle state, or a portable artifact.
_COMPACTED_TRACES: weakref.WeakSet[Trace] = weakref.WeakSet()


def compact_op_metadata(trace: Trace) -> None:
    """Collapse repeated immutable Op metadata onto shared instances.

    A finished graph stores the same dtype name, module address, ancestor
    label, and zero-valued quantity once per op, so Python metadata grows
    with ``#ops x #repeated facts`` rather than with the number of distinct
    facts. One pass at the freeze seam pools those values; the pool is
    dropped on return, so nothing is retained process-wide.

    Field values are unchanged -- see :func:`~torchlens._data_classes.op._pool_key`
    for why pooling is injective, and ``Op._compact_metadata`` for the
    per-field walk. Running it more than once is a no-op beyond the first.
    Op cells still default-omitted are written last, in the form the pooling
    would have left them in (see :func:`_pool_core_containers`).
    """

    core = trace.__dict__.get("_trace_core")
    store = core.ops if core is not None else None
    ops = trace.__dict__.get("_layer_list")
    share: Callable[[int, Any], Any] | None = None
    if ops and trace not in _COMPACTED_TRACES:
        _COMPACTED_TRACES.add(trace)
        pool: dict[Any, Any] = {}
        if store is not None:
            # Core-backed ops pool in ONE column-major sweep over the row
            # cells (same ladder, same skips as the per-op walk) without
            # paying the attribute protocol per field.
            _compact_store_rows(store, pool)
        if core is not None:
            share = _pool_core_containers(core, store)
        seen_ops: set[int] = set()
        for op in ops:
            op_id = id(op)
            if op_id in seen_ops:
                continue
            seen_ops.add(op_id)
            if store is not None and getattr(op, "_core", None) is store:
                continue
            op._compact_metadata(pool)
    rows = store.rows_building() if store is not None else None
    if rows is not None:
        settle_default_cells(rows, share)


def _pool_core_containers(core: Any, store: Any) -> Callable[[int, Any], Any]:
    """Pool whole container cells of a core's stores; return the default-cell hook.

    Whole-cell container pooling (the memory slice): duplicate and empty
    provably-immutable mutable-container cells collapse onto one shared
    ``PooledCell`` per distinct content: allowlisted op fields, every
    kind-table field (Module hook lists, ``custom_attributes``...).

    The returned ``share(fid, value)`` hook is what ``settle_default_cells``
    calls for an op cell nothing read or wrote before the freeze seam: a
    fresh empty default of a sanctioned field becomes the shared
    ``PooledCell`` of its content, from the same table the pooling used, so
    the settled store equals one whose capture wrote every container. Any
    other default stays a fresh container per row.
    """

    from .op import _POOLED_CONTAINER_FIELDS, _container_pool_key, _pool_container_cells

    container_pool: dict[Any, Any] = {}
    container_stores: list[tuple[Any, Any]] = []
    pooled_fids: tuple[int, ...] = ()
    if store is not None:
        fid_by_name = store.layout.fid_by_name
        pooled_fids = tuple(
            fid_by_name[name] for name in _POOLED_CONTAINER_FIELDS if name in fid_by_name
        )
        container_stores.append((store, pooled_fids))
    for kind_store in core.kind_rows.values():
        container_stores.append((kind_store, None))
    if container_stores:
        _pool_container_cells(container_stores, container_pool)
    sanctioned = frozenset(pooled_fids)

    def share(fid: int, value: Any) -> Any:
        """Return the shared pooled cell for a sanctioned empty container."""

        if fid not in sanctioned:
            return value
        key = _EMPTY_CONTAINER_POOL_KEYS.get(value.__class__)
        if key is None:
            key = _container_pool_key(value, 0, [])
            if key is None:
                return value
        cell = container_pool.get(key)
        if cell is None:
            cell = container_pool[key] = PooledCell.from_value(value)
        return cell

    return share


def _compact_store_rows(store: Any, pool: dict[Any, Any]) -> None:
    """Pool repeated immutable metadata across a whole building-phase store.

    Column-major equivalent of ``Op._compact_metadata`` (same class ladder,
    same ``_UNPOOLED_SLOTS`` skips, same container-member handling) operating
    directly on the row cells, so a core-backed trace pools without paying
    the per-attribute descriptor protocol.

    Parameters
    ----------
    store:
        Building-phase ``OpRowStore`` (frozen stores are left untouched --
        pooling always precedes the physical freeze).
    pool:
        Pass-local ``pool key -> canonical instance`` table shared with any
        remaining per-op walks of the same trace.
    """

    # Imported lazily, as in ``compact_op_metadata``, so loading this module never loads ``op``.
    from .op import _POOLED_CLASSES, _UNPOOLED_SLOTS, _pool_container_members, _pool_value

    rows = store.rows_building()
    if rows is None:
        return
    pooled_classes = _POOLED_CLASSES
    pool_get = pool.get
    for fid, name in enumerate(store.layout.names):
        if name in _UNPOOLED_SLOTS:
            continue
        for row_cells in rows:
            value = row_cells[fid]
            if value is None or value is _MISSING:
                continue
            cls = value.__class__
            if cls is bool or cls is int or cls is LazyCell or cls is float:
                # Nothing below pools these: plain numbers are not pooled
                # values and a default-omitted cell holds no value yet.
                continue
            if cls in pooled_classes and (cls is not Duration or not value):
                key = (cls, value.hex()) if cls is Duration else (cls, value)
                pooled = pool_get(key)
                if pooled is None:
                    pool[key] = value
                elif pooled is not value:
                    row_cells[fid] = pooled
            elif cls is list or cls is set or cls is dict:
                _pool_container_members(value, pool, 0)
            elif cls is tuple or cls is frozenset:
                pooled = _pool_value(value, pool)
                if pooled is not value:
                    row_cells[fid] = pooled
            elif isinstance(value, dict):
                _pool_container_members(value, pool, 0)
