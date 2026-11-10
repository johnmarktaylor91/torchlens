"""Rebuild weakref-backed model accessors after finalization or rehydrate.

Portable loads intentionally avoid rebuilding every weakref-backed accessor in
``__setstate__``. This module centralizes the post-load repair step that
reconnects module and buffer accessors to the owning ``Trace`` once the
rehydrated object graph is ready for normal user access.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from .._data_classes._op_column_scan import ops_with
from .._data_classes._trace_accessors import (
    _TRACE_MODULE_CALL_ACCESSOR_ATTR,
    TraceModuleCallAccessor,
    _invalidate_trace_module_call_accessor_cache,
)
from .._data_classes.buffer import Buffer, BufferAccessor
from .._data_classes.module import ModuleAccessor
from .._trace_core.record_refs import name_owner_by_row, rekey_child_cells
from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.module import Module, ModuleCall
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def rebuild_trace_accessors(
    trace: Trace,
    module_dict: dict[str, Module],
    module_order: list[Module],
    pass_dict: dict[str, ModuleCall],
    *,
    built_for_trace: bool = False,
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
    built_for_trace:
        The records were just constructed for ``trace`` and are still
        detached (the step 16 build): they already name it, so they are not
        rebound, and they are adopted here even when the trace has no op store.
    """

    # The door is set first: a module whose ``ops`` cell already holds rows
    # (a second rebuild of the same records) reads its calls through it.
    _invalidate_trace_module_call_accessor_cache(trace)
    trace._module_logs = ModuleAccessor(module_dict, module_order, pass_dict)

    for module_log in () if built_for_trace else module_dict.values():
        module_log._source_trace = trace
        module_log._buffer_accessor = None
        module_ops = getattr(module_log, "ops", None)
        if module_ops is None:
            continue
        for module_call in module_ops._dict.values():
            module_call._source_trace = trace

    _adopt_module_rows(trace, module_dict, pass_dict, built_for_trace)
    rebuild_buffer_accessor(trace)
    if built_for_trace:
        _list_calls_in_build_order(trace, pass_dict)


def _list_calls_in_build_order(trace: Trace, pass_dict: dict[str, ModuleCall]) -> None:
    """Set the trace's flat module-call accessor from the calls the build just made.

    The build makes the root's calls first and then each module's calls in
    module order, which is the order ``Trace.module_calls`` lists them in, so
    the accessor needs no walk over every module's calls.
    """

    calls = TraceModuleCallAccessor(OrderedDict(pass_dict))
    trace.__dict__[_TRACE_MODULE_CALL_ACCESSOR_ATTR] = calls


def _adopt_module_rows(
    trace: Trace,
    module_dict: dict[str, Module],
    pass_dict: dict[str, ModuleCall],
    built_for_trace: bool,
) -> None:
    """Adopt the Module and ModuleCall rows into the trace's kind tables.

    A ``.tlspec`` load rebuilds the module accessor only after ``__setstate__``
    rehydrated and sealed the core, which would leave every public Module and
    ModuleCall facade detached-backed with empty kind tables; a live capture's
    step 16 hands its freshly built records here for the same adoption. Tables
    born after a sealed core are sealed too, exactly like buffers. Between
    adoption and the seal, each module's ``ops`` and ``params`` cells swap
    their accessors for the children's rows, so the module table holds no
    record (a cell written after the seal would leave the accessor in place).
    """

    _core = trace.__dict__.get("_trace_core")
    if _core is None or not (built_for_trace or _core.ops is not None):
        return
    from torchlens._trace_core.record_rows import adopt_records

    kinds = (("module", list(module_dict.values())), ("module_call", list(pass_dict.values())))
    for kind, records in kinds:
        if records:
            adopt_records(_core, kind, records)
    if _core.ops is None:
        return
    rekey_child_cells(module_dict.values(), trace)
    for kind, records in kinds:
        kind_store = _core.kind_rows.get(kind)
        if records and kind_store is not None and not kind_store.frozen and _core.ops.frozen:
            kind_store.freeze()


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
    trace_ref = weakref.ref(trace)
    for param in trace.__dict__.get("_param_logs") or ():
        name_owner_by_row(trace_ref, param)
    # Only an op with gradient records or a buffer flag has work here.
    for entry in ops_with(trace, "_grad_records", "is_buffer"):
        name_owner_by_row(trace_ref, entry)
        if record_member(entry, "is_buffer", False) and entry.buffer_address is not None:
            buffer_versions.setdefault(entry.buffer_address, []).append(entry)
    buffer_dict = {
        address: Buffer(
            address,
            versions,
            initial_value=(getattr(trace, "_buffer_initial_values", None) or {}).get(address),
            source_trace=trace,
        )
        for address, versions in buffer_versions.items()
    }
    # Adopt Buffer rows into the per-trace kind table when this trace is
    # core-backed (live captures AND rehydrated loads).
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


def rebuild_saved_trace_accessors(trace: Trace, module_accessor_state: Any) -> None:
    """Restore saved module accessors after their buffer payloads have materialized."""

    # rebuild_trace_accessors() MUST run AFTER _rehydrate_object(), not before.
    # accessor_rebuild.py bakes `trace._buffer_initial_values.get(address)`
    # into each Buffer's `_initial_value` at construction time (Buffer.__init__
    # stores it directly; `initial_value` is not a dynamically-recomputed
    # property). `_buffer_initial_values` is a FieldPolicy.BLOB_RECURSIVE field
    # that only _rehydrate_object() resolves from raw BlobRef placeholders into
    # real tensors (for a default eager load; lazy=True/materialize_nested=False
    # intentionally leaves it as BlobRef for the expert nested-materialization
    # flow). Building accessors first captured the still-unresolved BlobRef
    # permanently, so `Buffer.initial_value` returned a raw BlobRef forever
    # after a plain `tl.load()` instead of a torch.Tensor. Nothing
    # rebuild_trace_accessors() reads (trace._layer_list, module_accessor_state)
    # depends on _rehydrate_object() having run first, so this reorder is safe.
    if module_accessor_state is not None:
        rebuild_trace_accessors(
            trace,
            module_accessor_state._dict,
            module_accessor_state._list,
            module_accessor_state._pass_dict,
        )
        # An artifact written before the root call counted among saved module
        # calls stores a count without it; re-derive the stored scalar from the
        # loaded records so it matches ``saved_module_calls``.
        from .._postprocess.saved_summary import refresh_saved_module_call_count

        refresh_saved_module_call_count(trace)
