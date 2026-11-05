"""Roofline traffic fields and the ideal per-op memory traffic they derive."""

from __future__ import annotations

from typing import Any

from .._utils._record_group_table import record_member

#: The Op record fields the roofline traffic model reads (D21: read-once /
#: write-once ideal traffic = parent outputs + own output + parameters
#: consumed). Pinned against the Op field vocabulary at import by
#: :func:`_pin_traffic_fields`, so a field rename can never silently zero a
#: traffic term again (AUD-CODE 2.11: a ``params_memory`` misspelling read
#: ``None`` through ``getattr`` and dropped ALL parameter traffic, reporting a
#: Linear(256, 512) at 85 FLOP/B instead of 1.96 FLOP/B).
ROOFLINE_TRAFFIC_FIELDS: tuple[str, ...] = ("activation_memory", "param_memory")
_TRAFFIC_OUTPUT_FIELD, _TRAFFIC_PARAM_FIELD = ROOFLINE_TRAFFIC_FIELDS


def _op_ideal_traffic(op: Any, trace: Any) -> int | None:
    """Read-once/write-once logical traffic for one op, if derivable."""

    output_bytes = record_member(op, _TRAFFIC_OUTPUT_FIELD, None)
    if output_bytes is None:
        return None
    total = int(output_bytes)
    for parent_label in getattr(op, "parents", ()) or ():
        try:
            parent = trace.ops[str(parent_label)]
        except (KeyError, ValueError):
            return None
        parent_bytes = record_member(parent, _TRAFFIC_OUTPUT_FIELD, None)
        if parent_bytes is not None:
            total += int(parent_bytes)
    param_bytes = record_member(op, _TRAFFIC_PARAM_FIELD, None)
    if param_bytes is not None:
        total += int(param_bytes)
    return total
