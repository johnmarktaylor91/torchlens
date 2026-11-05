"""Pre-hook provenance of one module call, with raw producer labels finalized.

A module call's forward pre-hooks record before and after input snapshots and
their ordered effects under raw operation labels; these helpers map the
snapshots' producer labels to final operation labels once Step 8 has built the
raw-to-final map.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .._data_classes.prehook import ModuleInputSnapshot, PreHookEffect

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _pre_hook_provenance_for_call(
    trace: Trace, call_label: str
) -> tuple[ModuleInputSnapshot | None, ModuleInputSnapshot | None, tuple[PreHookEffect, ...]]:
    """Return provenance for one call with raw producer labels finalized.

    Parameters
    ----------
    trace:
        Trace whose Step-8 raw-to-final operation map is available.
    call_label:
        Pass-qualified module call label.

    Returns
    -------
    tuple[ModuleInputSnapshot | None, ModuleInputSnapshot | None, tuple[PreHookEffect, ...]]
        Before snapshot, after snapshot, and ordered effects.
    """

    values = trace._module_capture_ws.module_build_data.get("module_pre_hook_provenance", {}).get(
        call_label
    )
    if values is None:
        return None, None, ()
    before, after, effects = values
    return (
        _finalize_snapshot_source_labels(trace, before),
        _finalize_snapshot_source_labels(trace, after),
        tuple(effects),
    )


def _finalize_snapshot_source_labels(
    trace: Trace, snapshot: ModuleInputSnapshot | None
) -> ModuleInputSnapshot | None:
    """Replace raw tensor producer labels in one immutable snapshot."""

    if snapshot is None:
        return None
    mapping = trace._raw_to_final_op_labels
    observations = tuple(
        replace(
            observation,
            source_op=(
                mapping.get(observation.source_op, observation.source_op)
                if observation.source_op is not None
                else None
            ),
        )
        for observation in snapshot.tensor_observations
    )
    return replace(snapshot, tensor_observations=observations)
