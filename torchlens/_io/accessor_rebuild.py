"""Rebuild weakref-backed model accessors after finalization or rehydrate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..data_classes.buffer_log import BufferAccessor
from ..data_classes.module_log import ModuleAccessor

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog
    from ..data_classes.module_log import ModuleLog, ModulePassLog


def rebuild_model_log_accessors(
    model_log: "ModelLog",
    module_dict: dict[str, "ModuleLog"],
    module_order: list["ModuleLog"],
    pass_dict: dict[str, "ModulePassLog"],
) -> None:
    """Rebuild the user-facing module and buffer accessors on a ``ModelLog``.

    Parameters
    ----------
    model_log:
        Model log receiving the rebuilt accessors.
    module_dict:
        Mapping from primary module address to ``ModuleLog``.
    module_order:
        Ordered list of module logs for iteration and integer indexing.
    pass_dict:
        Mapping from ``"address:pass"`` labels to ``ModulePassLog`` entries.
    """

    for module_log in module_dict.values():
        module_log._source_model_log = model_log
        module_log._buffer_accessor = None

    model_log._module_logs = ModuleAccessor(module_dict, module_order, pass_dict)

    buffer_dict = {}
    for label in model_log.buffer_layers:
        if label in model_log.layer_dict_all_keys:
            entry = model_log.layer_dict_all_keys[label]
            if entry.buffer_address is not None:
                buffer_dict[entry.buffer_address] = entry
    model_log._buffer_accessor = BufferAccessor(buffer_dict, source_model_log=model_log)  # type: ignore[assignment, arg-type]
