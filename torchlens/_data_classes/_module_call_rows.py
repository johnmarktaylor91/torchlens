"""DataFrame row converter for ModuleCall records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._constants import MODULE_PASS_LOG_FIELD_ORDER

if TYPE_CHECKING:
    from .module import ModuleCall


def _module_call_log_to_row(module_call_log: ModuleCall) -> dict[str, Any]:
    """Convert a ModuleCall into one DataFrame row.

    Parameters
    ----------
    module_call_log:
        Module-pass metadata entry to export.

    Returns
    -------
    Dict[str, Any]
        Mapping from canonical field name to exported value.
    """
    row = {field: getattr(module_call_log, field) for field in MODULE_PASS_LOG_FIELD_ORDER}
    row["output_ops"] = list(module_call_log.output_ops)
    row["output_structure"] = (
        repr(module_call_log.output_structure)
        if module_call_log.output_structure is not None
        else None
    )
    return row
