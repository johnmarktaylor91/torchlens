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
from .._trace_core.pools import ContainerTokens
from ._arg_position_keys import with_arg_position_keys
from ._container_key import make_content_key, members_inert
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
        tokens = with_arg_position_keys(store, _take_container_tokens(trace))
        if store is not None:
            # Core-backed ops pool in ONE column-major sweep over the row
            # cells (same ladder, same skips as the per-op walk) without
            # paying the attribute protocol per field.
            _compact_store_rows(store, pool, tokens)
        if core is not None:
            share = _pool_core_containers(core, store, tokens)
        seen_ops: set[int] = set()
        for op in ops:
            if store is not None and getattr(op, "_core", None) is store:
                continue
            op_id = id(op)
            if op_id in seen_ops:
                continue
            seen_ops.add(op_id)
            op._compact_metadata(pool)
    rows = store.rows_building() if store is not None else None
    if rows is not None:
        settle_default_cells(rows, share)


def _take_container_tokens(trace: Any) -> ContainerTokens | None:
    """Take the trace core's write-time container tokens (``None`` once taken).

    The workspace drop handed them to ``core.pool.tokens``
    (:func:`hand_off_container_tokens`); this freeze is their last reader.
    """

    core = trace.__dict__.get("_trace_core")
    pool = getattr(core, "pool", None)
    tokens = getattr(pool, "tokens", None)
    if pool is not None and tokens is not None:
        pool.tokens = None
    return tokens


def hand_off_container_tokens(trace: Any, workspace: Any) -> None:
    """Move the runtime workspace's token table to the trace core's pool.

    Called where the capture workspaces drop. ``False`` is left on the
    workspace so a snapshot logged after this registers nothing.
    """

    table = getattr(workspace, "container_tokens", None)
    if workspace is not None:
        workspace.container_tokens = False
    core = trace.__dict__.get("_trace_core")
    if table and core is not None:
        core.pool.tokens = ContainerTokens(table)


def _pool_core_containers(
    core: Any, store: Any, tokens: ContainerTokens | None = None
) -> Callable[[int, Any], Any]:
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
        content_key = make_content_key(_container_pool_key, tokens)
        _pool_container_cells(container_stores, container_pool, content_key)
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


def _compact_store_rows(
    store: Any, pool: dict[Any, Any], tokens: ContainerTokens | None = None
) -> None:
    """Pool repeated immutable metadata across a whole building-phase store.

    Column-major equivalent of ``Op._compact_metadata`` (same class ladder, same
    ``_UNPOOLED_SLOTS`` skips, same container-member handling) on the row cells.

    Parameters
    ----------
    store:
        Building-phase ``OpRowStore`` (frozen stores are left untouched --
        pooling always precedes the physical freeze).
    pool:
        Pass-local ``pool key -> canonical instance`` table shared with any
        remaining per-op walks of the same trace.
    tokens:
        The trace's write-time container tokens. A dict registered as inert is
        skipped: its members are values the pool can never replace (``ContainerTokens``).
    """

    # Imported lazily, as in ``compact_op_metadata``, so loading this module never loads ``op``.
    from .op import _POOLED_CLASSES, _UNPOOLED_SLOTS, _pool_container_members, _pool_value

    rows = store.rows_building()
    if rows is None:
        return
    helpers = (_POOLED_CLASSES, _pool_container_members, _pool_value)
    for fid, name in enumerate(store.layout.names):
        if name in _UNPOOLED_SLOTS:
            continue
        if name in _BY_OBJECT_FIELDS and _sweep_by_object(rows, fid, pool, tokens, helpers):
            continue
        _sweep_column(rows, fid, pool, tokens, helpers)


#: Cell classes the per-cell ladder never pools or descends into: ``None``, the
#: unset sentinel, a default-omitted cell and the plain numbers.
_INERT_CELL_CLASSES = frozenset({type(None), type(_MISSING), LazyCell, bool, int, float})


def _sweep_column(
    rows: list[list[Any]],
    fid: int,
    pool: dict[Any, Any],
    tokens: ContainerTokens | None,
    helpers: tuple[Any, Any, Any],
) -> None:
    """Pool one column cell by cell (the per-cell ladder of ``Op._compact_metadata``).

    A column whose cells are all of an inert class is skipped whole: the ladder
    leaves each such cell as it is, so the skip changes no cell and no pool entry.
    Any other column visits its cells in row order, as before.
    """

    # Essential complexity (D): the class ladder moved here whole from
    # _compact_store_rows, which drops to its column dispatch; it stays one hot
    # loop because a call per cell would cost more than the skip saves.
    column = [row_cells[fid] for row_cells in rows]
    if _INERT_CELL_CLASSES.issuperset(map(type, column)):
        return
    pooled_classes, pool_container_members, pool_value = helpers
    pool_get = pool.get
    for row_cells, value in zip(rows, column, strict=True):
        if value is None or value is _MISSING:
            continue
        cls = value.__class__
        if cls is bool or cls is int or cls is LazyCell or cls is float:
            continue  # plain numbers never pool; a default-omitted cell holds no value
        if cls in pooled_classes and (cls is not Duration or not value):
            key = (cls, value.hex()) if cls is Duration else (cls, value)
            pooled = pool_get(key)
            if pooled is None:
                pool[key] = value
            elif pooled is not value:
                row_cells[fid] = pooled
        elif cls is list or cls is set or cls is dict:
            if cls is dict and tokens is not None and members_inert(tokens, value):
                continue
            pool_container_members(value, pool, 0)
        elif cls is tuple or cls is frozenset:
            pooled = pool_value(value, pool)
            if pooled is not value:
                row_cells[fid] = pooled
        elif isinstance(value, dict):
            pool_container_members(value, pool, 0)


#: Op fields whose cells repeat a few value objects (part C1: row ingest interns
#: the quantities and references; the strings come from shared sources). The
#: sweep visits each distinct object of these columns once instead of each cell.
_BY_OBJECT_FIELDS = frozenset(
    {
        "dtype_ref",
        "device_ref",
        "activation_memory",
        "gradient_memory",
        "param_memory",
        "flops_forward",
        "flops_backward",
        "func_duration",
        "dtype",
        "output_device",
        "resolver_status",
        "func_name",
        "func_qualname",
        "type",
        "equivalence_class",
    }
)


def _sweep_by_object(
    rows: list[list[Any]],
    fid: int,
    pool: dict[Any, Any],
    tokens: ContainerTokens | None,
    helpers: tuple[Any, Any, Any],
) -> bool:
    """Sweep one column once per distinct cell object; ``False`` leaves it to the per-cell loop.

    Equal to the per-cell sweep: objects are visited in the order of their
    first cell, so the pool evolves as it would, and visiting one object again
    changes nothing (its pooled twin, or its members' twins, are already in the
    pool and pool entries never change), so every cell holding it gets the same
    replacement as its first cell. Columns with mostly distinct objects return
    ``False`` (nothing to save).
    """

    column = [row_cells[fid] for row_cells in rows]
    distinct = dict(zip(map(id, column), column, strict=True))
    if len(distinct) * 2 > len(column):
        return False
    for value in distinct.values():
        pooled = _pooled_cell(value, pool, tokens, helpers)
        if pooled is not value:
            for row_cells in rows:
                if row_cells[fid] is value:
                    row_cells[fid] = pooled
    return True


def _pooled_cell(
    value: Any, pool: dict[Any, Any], tokens: ContainerTokens | None, helpers: tuple[Any, Any, Any]
) -> Any:
    """Apply the per-cell sweep's ladder to one cell value; return what the cell should hold.

    ``helpers`` is ``op.py``'s ``(_POOLED_CLASSES, _pool_container_members, _pool_value)``.
    """

    pooled_classes, pool_container_members, pool_value = helpers
    if value is None or value is _MISSING:
        return value
    cls = value.__class__
    if cls is bool or cls is int or cls is LazyCell or cls is float:
        return value
    if cls in pooled_classes and (cls is not Duration or not value):
        key = (cls, value.hex()) if cls is Duration else (cls, value)
        pooled = pool.get(key)
        if pooled is None:
            pool[key] = value
            return value
        return pooled
    if cls is list or cls is set or cls is dict:
        if not (cls is dict and tokens is not None and members_inert(tokens, value)):
            pool_container_members(value, pool, 0)
    elif cls is tuple or cls is frozenset:
        return pool_value(value, pool)
    elif isinstance(value, dict):
        pool_container_members(value, pool, 0)
    return value
