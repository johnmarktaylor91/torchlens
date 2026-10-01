"""``groupNodeAttributes`` rows and the ``""`` provenance block (memo D7).

The ``""`` (root) row renders in Model Explorer's side panel when nothing is
selected -- the visible TorchLens provenance-and-disclosure block (versions,
capture kind, honesty facts, privacy/id profiles, every fidelity downgrade,
every omission count). Per-namespace rows carry address/class, direct and
descendant op counts, deduplicated params, activation bytes, call count, and
honest measured duration. Missing values are absent, never zero. Rows are
emitted for ALL namespaces regardless of viewer-side single-child pruning
(memo D9: emission is asserted always; UI reachability only under the
faithful profile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...errors._base import TorchLensError
from ...utils.display import human_readable_size
from ._namespace import NamespaceLevel

__tl_layer__ = "L8"


@dataclass
class _NamespaceStats:
    """Accumulated per-namespace facts consumed by one group row."""

    address: str = ""
    call_indices: set[int] = field(default_factory=set)
    direct_ops: int = 0
    descendant_ops: int = 0
    activation_bytes: int = 0
    has_activation_bytes: bool = False


def accumulate_namespace_stats(
    entries: list[Any],
    namespaces: list[str],
    levels_per_entry: list[list[NamespaceLevel]],
) -> dict[str, _NamespaceStats]:
    """Fold per-entry namespace levels into per-namespace group-row stats."""

    stats: dict[str, _NamespaceStats] = {}
    for entry, namespace, levels in zip(entries, namespaces, levels_per_entry, strict=True):
        activation_memory = getattr(entry, "activation_memory", None)
        for level in levels:
            row = stats.setdefault(level.namespace, _NamespaceStats(address=level.address))
            row.descendant_ops += 1
            if level.call_index is not None:
                row.call_indices.add(level.call_index)
            if activation_memory is not None:
                row.activation_bytes += int(activation_memory)
                row.has_activation_bytes = True
            if level.namespace == namespace:
                row.direct_ops += 1
    return stats


def build_group_rows(
    trace: Any,
    stats: dict[str, _NamespaceStats],
    root_facts: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Build the full ``groupNodeAttributes`` mapping including the root row."""

    rows: dict[str, dict[str, str]] = {"": dict(root_facts)}
    for namespace in sorted(stats):
        rows[namespace] = _namespace_row(trace, stats[namespace])
    return rows


def _namespace_row(trace: Any, row_stats: _NamespaceStats) -> dict[str, str]:
    """Format one per-namespace group row from its accumulated stats."""

    row: dict[str, str] = {"address": row_stats.address}
    module = _module_record(trace, row_stats.address)
    if module is not None:
        class_name = str(getattr(module, "class_name", "") or "")
        if class_name:
            row["class"] = class_name
        num_params = int(getattr(module, "num_params", 0) or 0)
        if num_params > 0:
            row["params"] = str(num_params)
    row["ops"] = str(row_stats.descendant_ops)
    if row_stats.direct_ops:
        row["direct_ops"] = str(row_stats.direct_ops)
    if row_stats.call_indices:
        row["calls"] = str(len(row_stats.call_indices))
    if row_stats.has_activation_bytes:
        row["act_bytes"] = human_readable_size(row_stats.activation_bytes)
    duration = _module_duration(trace, row_stats)
    if duration is not None:
        row["time"] = f"{duration * 1000.0:.3g} ms"
    return row


def _module_record(trace: Any, address: str) -> Any | None:
    """Return the trace's Module record for one address, if present."""

    modules = getattr(trace, "modules", None)
    if modules is None or not address:
        return None
    try:
        return modules[address]
    except (LookupError, TorchLensError):
        return None


def _module_duration(trace: Any, row_stats: _NamespaceStats) -> float | None:
    """Sum honestly measured forward durations across this row's calls."""

    module_calls = getattr(trace, "module_calls", None)
    if module_calls is None or not row_stats.call_indices:
        return None
    total = 0.0
    measured = False
    for call_index in sorted(row_stats.call_indices):
        call = _module_call(module_calls, f"{row_stats.address}:{call_index}")
        if call is None:
            continue
        duration = getattr(call, "forward_duration", None)
        if duration is not None and float(duration) > 0.0:
            total += float(duration)
            measured = True
    return total if measured else None


def _module_call(module_calls: Any, key: str) -> Any | None:
    """Return one ModuleCall record, or ``None`` when the call is absent."""

    try:
        return module_calls[key]
    except (LookupError, TorchLensError):
        return None
