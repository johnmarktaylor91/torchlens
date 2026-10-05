"""Rebuild weakref-backed model accessors after finalization or rehydrate.

Portable loads intentionally avoid rebuilding every weakref-backed accessor in
``__setstate__``. This module centralizes the post-load repair step that
reconnects module and buffer accessors to the owning ``Trace`` once the
rehydrated object graph is ready for normal user access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_classes._trace_accessors import _invalidate_trace_module_call_accessor_cache
from ..data_classes.buffer import Buffer, BufferAccessor
from ..data_classes.module import ModuleAccessor

if TYPE_CHECKING:
    from ..data_classes.module import Module, ModuleCall
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


def rebuild_trace_accessors(
    trace: Trace,
    module_dict: dict[str, Module],
    module_order: list[Module],
    pass_dict: dict[str, ModuleCall],
) -> None:
    """Rebuild the user-facing module and buffer accessors on a ``Trace``.

    Parameters
    ----------
    trace:
        Model log receiving the rebuilt accessors.
    module_dict:
        Mapping from primary module address to ``Module``.
    module_order:
        Ordered list of module logs for iteration and integer indexing.
    pass_dict:
        Mapping from ``"address:pass"`` labels to ``ModuleCall`` entries.
    """

    for module_log in module_dict.values():
        module_log._source_trace = trace
        module_log._buffer_accessor = None
        module_ops = getattr(module_log, "ops", None)
        if module_ops is None:
            continue
        for module_call in module_ops._dict.values():
            module_call._source_trace = trace

    _invalidate_trace_module_call_accessor_cache(trace)
    trace._module_logs = ModuleAccessor(module_dict, module_order, pass_dict)

    # Adopt Module/ModuleCall rows into the per-trace kind tables (M8). A
    # ``.tlspec`` load rebuilds the module accessor only HERE — after
    # ``__setstate__`` already rehydrated and sealed the core (F9) — which
    # used to leave every public Module/ModuleCall facade detached-backed
    # with empty kind tables (closure review F9 gap). Live captures arrive
    # already table-bound (postprocess/finalization adopts them just before
    # calling this), so adoption is an idempotent no-op there. Tables born
    # after a sealed core are sealed too, exactly like buffers below.
    _core = trace.__dict__.get("_trace_core")
    if _core is not None and _core.ops is not None:
        from torchlens._trace_core.record_rows import adopt_records

        for kind, records in (
            ("module", list(module_dict.values())),
            ("module_call", list(pass_dict.values())),
        ):
            if not records:
                continue
            adopt_records(_core, kind, records)
            kind_store = _core.kind_rows.get(kind)
            if kind_store is not None and not kind_store.frozen and _core.ops.frozen:
                kind_store.freeze()

    rebuild_buffer_accessor(trace)


def rebuild_buffer_accessor(trace: Trace) -> None:
    """Rebuild the user-facing buffer accessor on a ``Trace`` from its buffer ops.

    Split out of ``rebuild_trace_accessors`` so preview function-root traces,
    which build their single ``self`` module directly, still get a (possibly
    empty) ``trace.buffers`` accessor instead of ``None``.

    Parameters
    ----------
    trace:
        Model log receiving the rebuilt buffer accessor.
    """

    buffer_versions: dict[str, list[Op]] = {}
    for entry in trace.layer_list:
        for grad_record in getattr(entry, "_grad_records", ()):
            grad_record.owner = entry
        if getattr(entry, "is_buffer", False) and entry.address is not None:
            buffer_versions.setdefault(entry.address, []).append(entry)
    buffer_dict = {
        address: Buffer(
            address,
            versions,
            initial_value=(getattr(trace, "_buffer_initial_values", None) or {}).get(address),
            source_trace=trace,
        )
        for address, versions in buffer_versions.items()
    }
    # Adopt Buffer rows into the per-trace kind table (M8) when this trace is
    # core-backed (live captures AND rehydrated loads, F9).
    _core = trace.__dict__.get("_trace_core")
    if _core is not None and buffer_dict:
        from torchlens._trace_core.record_rows import adopt_records

        adopt_records(_core, "buffer", buffer_dict.values())
        # A rehydrated load's core is already sealed; seal a kind table born
        # after that seal too, so a later fork can view it (OpStoreView
        # requires sealed bases).
        buffer_store = _core.kind_rows.get("buffer")
        if (
            buffer_store is not None
            and not buffer_store.frozen
            and _core.ops is not None
            and _core.ops.frozen
        ):
            buffer_store.freeze()
    trace._buffer_accessor = BufferAccessor(buffer_dict, source_trace=trace)  # type: ignore[assignment]
