"""Freeze-seam Op metadata compaction (M11 fold of the standalone pass).

``compact_op_metadata`` pools repeated immutable Op metadata onto shared
instances. It used to live on ``Trace`` as ``_compact_op_metadata``; the M11
fold moved it here as a module function invoked from the core freeze seam
(torch postprocess) and the preview-backend finalizers, so the Trace surface
carries no compaction method and the freeze owns the compaction role
(trace_core_design.md section 3.3).
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

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
    """

    if trace in _COMPACTED_TRACES:
        return
    ops = trace.__dict__.get("layer_list")
    if not ops:
        return
    _COMPACTED_TRACES.add(trace)
    pool: dict[Any, Any] = {}
    core = trace.__dict__.get("_trace_core")
    store = core.ops if core is not None else None
    if store is not None:
        # Core-backed ops pool in ONE column-major sweep over the row
        # cells (same ladder, same skips as the per-op walk) without
        # paying the attribute protocol per field.
        from .op import _compact_store_rows

        _compact_store_rows(store, pool)
    if core is not None:
        # Whole-cell container pooling (the M14 memory slice): duplicate and
        # empty provably-immutable mutable-container cells collapse onto one
        # shared PooledCell per distinct content — allowlisted op fields,
        # every kind-table field (Module hook lists, custom_attributes...).
        from .op import _POOLED_CONTAINER_FIELDS, _pool_container_cells

        container_stores: list[tuple[Any, Any]] = []
        if store is not None:
            fid_by_name = store.layout.fid_by_name
            container_stores.append(
                (
                    store,
                    tuple(
                        fid_by_name[name]
                        for name in _POOLED_CONTAINER_FIELDS
                        if name in fid_by_name
                    ),
                )
            )
        for kind_store in core.kind_rows.values():
            container_stores.append((kind_store, None))
        if container_stores:
            _pool_container_cells(container_stores, {})
    seen_ops: set[int] = set()
    for op in ops:
        op_id = id(op)
        if op_id in seen_ops:
            continue
        seen_ops.add(op_id)
        if store is not None and getattr(op, "_core", None) is store:
            continue
        op._compact_metadata(pool)
