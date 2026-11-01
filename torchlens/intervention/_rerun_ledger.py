"""Ledger-record assembly for rerun captures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .hooks import NormalizedHookEntry


def _build_ledger_record(
    log: Trace,
    *,
    started_at: float,
    old_hash: str | None,
    new_hash: str | None,
    old_raw_hash: str | None,
    new_raw_hash: str | None,
    hook_plan: list[NormalizedHookEntry],
    strict: bool,
    divergence_count: int,
    fast_refresh: bool,
    hook_fire_count: int,
    unfired_hook_count: int,
) -> dict[str, Any]:
    """Create the append-only operation history record for a rerun.

    Parameters
    ----------
    log:
        Trace after state replacement.
    started_at:
        Monotonic start time for the rerun.
    old_hash:
        Graph-shape hash before rerun.
    new_hash:
        Candidate graph-shape hash from the fresh capture.
    old_raw_hash:
        Raw-event shape hash before rerun.
    new_raw_hash:
        Candidate raw-event shape hash from the fresh capture.
    hook_plan:
        Normalized hook entries active during rerun.
    strict:
        Whether strict divergence handling was requested.
    divergence_count:
        Number of divergence events detected.
    fast_refresh:
        Whether the rerun refreshed existing graph containers in place.
    hook_fire_count:
        Number of live hook firings observed on the candidate capture.
    unfired_hook_count:
        Number of planned hook entries that fired nowhere.

    Returns
    -------
    dict[str, Any]
        Operation-history entry.
    """

    return {
        "op": "rerun",
        "engine": "rerun",
        "started_at": started_at,
        "strict": strict,
        "append": False,
        "hook_count": len(hook_plan),
        "hook_fire_count": hook_fire_count,
        "unfired_hook_count": unfired_hook_count,
        "divergence_count": divergence_count,
        "fast_refresh": fast_refresh,
        "old_graph_shape_hash": old_hash,
        "new_graph_shape_hash": new_hash,
        "old_raw_event_shape_hash": old_raw_hash,
        "new_raw_event_shape_hash": new_raw_hash,
    }
