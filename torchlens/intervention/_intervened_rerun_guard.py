"""Refuse the legacy rerun of a loaded trace whose saved recipe holds a placeholder.

``trace.save()`` writes the staged intervention spec as ``intervention.json``
(the recipe grammar ``save_intervention`` also writes): rules of built-in
edits persist with their arguments, so a loaded intervened trace reruns with
its recorded intervention. A user callable is never saved; it becomes a
``callable_placeholder`` (rule level, or inside a built-in edit's arguments),
and the loaded spec records each one in ``metadata["recipe_placeholders"]``.
Rerunning such a trace would re-execute without that callable (an edit staged
after loading would run alone), so every legacy door (plain, append, chunked,
fork, and ``do(engine="rerun")``) refuses ``run_intervention_spec_not_persisted``
naming the rule position and the callable. A loaded trace whose values were
intervened but which carries no recipe at all refuses the same way.
"""

from __future__ import annotations

from typing import Any

from .._runnable.seam import carries_intervened_values
from .errors import EngineDispatchError

_LOADED_PROVIDERS = frozenset({"loaded_sparse", "loaded_analysis"})


def _artifact_values_intervened(trace: Any) -> bool:
    """Return whether the values the trace was LOADED with were intervened.

    Parameters
    ----------
    trace:
        Loaded trace (or a fork of one; forks copy the runnable state).

    Returns
    -------
    bool
        The snapshot the loader took before any session edit
        (``RunnableTraceState.loaded_values_intervened``). An edit staged in
        this session, such as a ``set_only`` edit on a loaded fork, marks ops
        edited too, so the live evidence cannot tell the artifact's values
        from the session's; the snapshot can. When no snapshot was taken the
        live evidence decides.
    """

    snapshot = getattr(getattr(trace, "_runnable", None), "loaded_values_intervened", None)
    if snapshot is not None:
        return bool(snapshot)
    return carries_intervened_values(trace)


def _session_spec_has_fired(trace: Any) -> bool:
    """Return whether a run in this session applied the trace's staged spec.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    bool
        True when the session spec holds fire records. A loaded spec starts
        empty, so any record means the current values come from a rerun of
        this session, not from the artifact.
    """

    return bool(getattr(trace._intervention_spec, "records", None))


def _loaded_recipe_placeholders(trace: Any) -> list[dict[str, Any]] | None:
    """Return the placeholders of the trace's loaded recipe, or ``None`` with no recipe.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[dict[str, Any]] | None
        ``position``/``qualname`` rows (empty for a recipe of built-in edits),
        or ``None`` when no recipe was loaded with the trace.
    """

    metadata = getattr(trace._intervention_spec, "metadata", None) or {}
    placeholders = metadata.get("recipe_placeholders")
    return None if placeholders is None else list(placeholders)


def refuse_unpersisted_intervention_rerun(trace: Any, provider: Any) -> None:
    """Refuse a legacy rerun of a loaded trace whose recipe cannot be replayed.

    Parameters
    ----------
    trace:
        Trace whose legacy ``run(model, x)`` is about to execute.
    provider:
        The trace's run provider (``RunProvider`` or ``None`` for live traces).

    Raises
    ------
    EngineDispatchError
        ``run_intervention_spec_not_persisted`` when the trace was loaded from
        an artifact whose recorded values were intervened, no rerun of this
        session has replaced those values yet, and its recipe holds a callable
        placeholder (or no recipe was saved). Edits staged after loading do
        not count either way: they would run without the recorded
        intervention, and they never make a pristine artifact look
        intervened. Live traces are never refused: an empty staged spec there
        means the user detached it on purpose.
    """

    if str(getattr(provider, "value", provider)) not in _LOADED_PROVIDERS:
        return
    if _session_spec_has_fired(trace):
        return
    if not _artifact_values_intervened(trace):
        return
    placeholders = _loaded_recipe_placeholders(trace)
    if placeholders == []:
        return
    if placeholders:
        first = placeholders[0]
        raise EngineDispatchError(
            "This trace was loaded from an artifact whose intervention recipe holds a "
            f"user callable that was not saved (rule {first.get('position')}: "
            f"{first.get('qualname')}), so run(model, x) would re-execute without it "
            "(anything staged after loading would run alone) and return numbers that do "
            "not match this trace. Remedy: rebuild that rule with its callable and pass it "
            "as intervene= to a fresh capture with tl.trace(model, x, intervene=...), or "
            "re-save the trace with built-in edits",
            code="run_intervention_spec_not_persisted",
        )
    raise EngineDispatchError(
        "This trace was loaded from an artifact whose recorded values were "
        "intervened, but the artifact carries no intervention recipe, so run(model, x) "
        "would re-execute without that intervention (anything staged after loading "
        "would run alone) and return numbers that do not match this trace. Remedy: "
        "re-capture with tl.trace(model, x, intervene=...), or capture plainly with "
        "tl.trace(model, x) for the un-intervened model",
        code="run_intervention_spec_not_persisted",
    )
