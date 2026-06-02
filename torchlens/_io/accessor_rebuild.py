"""Rebuild weakref-backed model accessors after finalization or rehydrate.

Portable loads intentionally avoid rebuilding every weakref-backed accessor in
``__setstate__``. This module centralizes the post-load repair step that
reconnects module and buffer accessors to the owning ``Trace`` once the
rehydrated object graph is ready for normal user access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_classes.buffer import BufferAccessor
from ..data_classes.module import ModuleAccessor

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ..data_classes.module import Module, ModuleCall


def rebuild_trace_accessors(
    trace: "Trace",
    module_dict: dict[str, "Module"],
    module_order: list["Module"],
    pass_dict: dict[str, "ModuleCall"],
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

    trace._module_logs = ModuleAccessor(module_dict, module_order, pass_dict)

    buffer_dict = {}
    for label in trace.buffer_layers:
        if label in trace.layer_dict_all_keys:
            entry = trace.layer_dict_all_keys[label]
            if entry.address is not None:
                buffer_dict[entry.address] = entry
    trace._buffer_accessor = BufferAccessor(buffer_dict, source_trace=trace)  # type: ignore[assignment, arg-type]
