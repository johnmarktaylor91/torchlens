"""Make a sticky hook removal reach the rerun or backward pass in progress.

The engines fire from a hook plan built at the run's start and published as
the ``_state._active_hook_plan`` global. A removal through ``HookHandle``
(``Trace.clear_edits`` / ``Trace.clear_hooks``) used to edit only the recipe,
so the removed hook kept firing until that run ended. These helpers retire
the removed entries from the live plan at removal time and let the run's
fire audit and ledger count the recipe as it stands after the run, so a
removal mid-run ends like a removal between runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .. import _state

if TYPE_CHECKING:
    from .hooks import NormalizedHookEntry


def sticky_handle_ids(spec: Any) -> set[str]:
    """Return the removable handle ids held by a recipe's sticky hook specs.

    Parameters
    ----------
    spec:
        Intervention recipe (``InterventionSpec``) or ``None``.

    Returns
    -------
    set[str]
        Handle ids of the sticky hook specs that carry one.
    """

    return {
        hook_spec.handle
        for hook_spec in getattr(spec, "hook_specs", None) or ()
        if hook_spec.handle is not None
    }


def retire_live_hook_handles(handle_ids: Iterable[str]) -> int:
    """Drop sticky hooks removed through their handles from the live plan.

    ``HookHandle.remove()`` takes effect at once, like torch's own hook
    handles: the next hook site the running engine (a rerun, an append rerun
    or a backward pass) reaches reads a plan without the entry. The plan
    global is rebound, never mutated in place, so the site that is firing
    when ``remove()`` is called finishes the list it is iterating, the way a
    torch module call fires its per-call snapshot of hooks. This runs only at
    removal time, so a run that removes nothing pays nothing per site.

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
    if not plan:
        return 0
    retired = set(handle_ids)
    if not retired:
        return 0
    kept = [entry for entry in plan if getattr(entry, "handle", None) not in retired]
    removed = len(plan) - len(kept)
    if removed:
        _state._active_hook_plan = kept
    return removed


def retire_removed_from_live_plan(handles_before: set[str], spec: Any) -> int:
    """Retire from the live plan every handle that ``spec`` no longer holds.

    Parameters
    ----------
    handles_before:
        The recipe's sticky handle ids before the removal.
    spec:
        The recipe after the removal.

    Returns
    -------
    int
        Number of live plan entries retired.
    """

    return retire_live_hook_handles(handles_before - sticky_handle_ids(spec))


def drop_handles_removed_mid_run(
    hook_plan: list[NormalizedHookEntry], spec: Any
) -> list[NormalizedHookEntry]:
    """Return the plan without entries whose handle left the recipe during the run.

    A sticky hook removed through its ``HookHandle`` while the rerun was in
    progress stopped firing at once (``retire_live_hook_handles``); the fire
    audit and the run ledger then count what the recipe holds after the run,
    the same as a removal between reruns. When nothing was removed the same
    list object comes back, so a plain rerun is unchanged.

    Parameters
    ----------
    hook_plan:
        The plan the run started with.
    spec:
        The recipe as it stands after the run.

    Returns
    -------
    list[NormalizedHookEntry]
        The plan restricted to entries the recipe still holds.
    """

    live_handles = sticky_handle_ids(spec)
    if all(entry.handle is None or entry.handle in live_handles for entry in hook_plan):
        return hook_plan
    return [entry for entry in hook_plan if entry.handle is None or entry.handle in live_handles]


def reset_append_state(trace: Any) -> None:
    """Mark ``trace`` as a fresh, un-appended rerun product.

    Parameters
    ----------
    trace:
        Trace whose run state was just replaced by a full rerun.
    """

    trace._provenance_is_appended = False
    trace._append_sequence_id = 0
    trace._append_history = []
