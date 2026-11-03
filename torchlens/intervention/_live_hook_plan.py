"""Hook-plan bookkeeping for the live engines: accounting ids and mid-run handle removal.

A rerun, an append rerun or a backward pass fires from a hook plan built at
the run's start and published as ``_state._active_hook_plan``. Removing a
sticky hook through its ``HookHandle`` takes effect at once, like a torch hook
handle: ``retire_live_hook_handles`` rebinds the published plan without the
removed entries, and ``_drop_handles_removed_mid_run`` makes the rerun's fire
audit and ledger count the recipe as it stands after the run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .. import _state

if TYPE_CHECKING:
    from .hooks import NormalizedHookEntry


def _assign_unique_plan_ids(hook_plan: list[NormalizedHookEntry]) -> list[NormalizedHookEntry]:
    """Give every planned entry a unique, stable accounting identifier.

    The fallback identifier ladder (plan id -> hook id -> helper name ->
    callable qualname) can COLLIDE across entries with different targets, and
    the fire audit compares ``Counter`` values keyed by that string: two
    fires of one entry hid the other entry's total miss (``fired=2``,
    ``unfired=()``), so a partially-applied plan claimed every entry fired
    (incomplete f9f5b140). Colliding identifiers get a stable occurrence
    suffix stamped into ``metadata["plan_id"]``, which live execution writes
    into each ``FireResult``, so the audit is per-entry; unique identifiers
    are preserved verbatim.
    """

    import dataclasses as _dataclasses

    counts = Counter(_hook_plan_identifier(entry) for entry in hook_plan)
    seen: Counter[str] = Counter()
    unique_plan: list[NormalizedHookEntry] = []
    for entry in hook_plan:
        base = _hook_plan_identifier(entry)
        if counts[base] > 1:
            metadata = dict(entry.metadata)
            metadata["plan_id"] = f"{base}#occ{seen[base]}"
            metadata["plan_id_base"] = base
            entry = _dataclasses.replace(entry, metadata=metadata)
        seen[base] += 1
        unique_plan.append(entry)
    return unique_plan


def _hook_plan_identifier(entry: NormalizedHookEntry) -> str:
    """Return the identifier written into a live ``FireResult``.

    Parameters
    ----------
    entry:
        Planned normalized hook entry.

    Returns
    -------
    str
        Plan id using the same fallback order as live execution.
    """

    if "plan_id" in entry.metadata:
        return str(entry.metadata["plan_id"])
    if "hook_id" in entry.metadata:
        return str(entry.metadata["hook_id"])
    if entry.helper_spec is not None:
        return str(entry.helper_spec.name)
    return str(getattr(entry.normalized_callable, "__qualname__", "user_hook"))


def sticky_handle_ids(spec: Any) -> set[str]:
    """Return the removable handle ids held by a recipe's sticky hook specs."""

    return {
        hook_spec.handle
        for hook_spec in getattr(spec, "hook_specs", None) or ()
        if hook_spec.handle is not None
    }


def retire_live_hook_handles(handle_ids: Iterable[str]) -> int:
    """Drop sticky hooks removed through their handles from the live plan.

    The next hook site the running engine reaches reads a plan without the
    entry. The plan global is rebound, never mutated in place, so the site
    that is firing when ``remove()`` is called finishes the list it is
    iterating, the way a torch module call fires its per-call snapshot of
    hooks. This runs only at removal time, so a run that removes nothing pays
    nothing per site.

    Parameters
    ----------
    handle_ids:
        Handle ids whose sticky specs were just removed from a recipe.

    Returns
    -------
    int
        Number of live plan entries retired (``0`` when no run is live or no
        entry carried one of the handles).
    """

    plan = _state._active_hook_plan
    retired = set(handle_ids)
    if not plan or not retired:
        return 0
    kept = [entry for entry in plan if getattr(entry, "handle", None) not in retired]
    removed = len(plan) - len(kept)
    if removed:
        _state._active_hook_plan = kept
    return removed


def _drop_handles_removed_mid_run(
    hook_plan: list[NormalizedHookEntry], spec: Any
) -> list[NormalizedHookEntry]:
    """Return the plan without entries whose handle left the recipe during the run.

    The fire audit and the run ledger then count what the recipe holds after
    the run, the same as a removal between reruns. When nothing was removed
    the same list object comes back, so a plain rerun is unchanged.
    """

    live_handles = sticky_handle_ids(spec)
    if all(entry.handle is None or entry.handle in live_handles for entry in hook_plan):
        return hook_plan
    return [entry for entry in hook_plan if entry.handle is None or entry.handle in live_handles]
