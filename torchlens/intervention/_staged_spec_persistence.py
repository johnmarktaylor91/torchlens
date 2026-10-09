"""Honest persistence of a trace's staged intervention spec.

``Trace._intervention_spec`` is session-only (``FieldPolicy.DROP``): an
ordinary save keeps the INTERVENED activations and their per-op provenance
(``intervention_replaced``, fire records, ``intervention_audit``) but not the
recipe that produced them. The recipe travels separately through
``save_intervention``; the runnable save already refuses intervened captures
(``user_intervention_not_replayable``) and names the analysis save as the door
for them. The one place the split could mislead is the legacy rerun of a loaded
artifact whose recorded values were intervened: it would re-execute without
the recorded intervention (an edit staged after loading would run alone), so it
refuses instead, on every legacy door (plain, append, chunked, fork, and
``do(engine="rerun")``).
"""

from __future__ import annotations

from typing import Any

from .errors import EngineDispatchError

_LOADED_PROVIDERS = frozenset({"loaded_sparse", "loaded_analysis"})
# Audit rows that record an edit which changed no recorded value.
_VALUE_NEUTRAL_AUDIT_STATUSES = frozenset({"no_fire", "error"})


def _carries_intervened_values(trace: Any) -> bool:
    """Return whether persisted evidence says recorded values were intervened.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    bool
        True when an op was replaced or fired, or a ``do()`` audit row records
        a fired edit (attempts that failed or fired nowhere changed nothing).
    """

    for layer in getattr(trace, "layer_list", None) or ():
        if getattr(layer, "intervention_replaced", False) or getattr(layer, "interventions", None):
            return True
    return any(
        isinstance(row, dict) and row.get("status") not in _VALUE_NEUTRAL_AUDIT_STATUSES
        for row in getattr(trace, "intervention_audit", None) or ()
    )


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


def refuse_unpersisted_intervention_rerun(trace: Any, provider: Any) -> None:
    """Refuse a legacy rerun of a loaded trace whose intervention recipe was dropped.

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
        an artifact whose recorded values were intervened and no rerun of this
        session has replaced those values yet. Edits staged after loading do
        not count: they would run without the recorded intervention. Live
        traces are never refused: an empty staged spec there means the user
        detached it on purpose.
    """

    if str(getattr(provider, "value", provider)) not in _LOADED_PROVIDERS:
        return
    if _session_spec_has_fired(trace):
        return
    if not _carries_intervened_values(trace):
        return
    raise EngineDispatchError(
        "This trace was loaded from an artifact whose recorded values were "
        "intervened, but tl.save() does not persist the intervention spec, so "
        "run(model, x) would re-execute without that intervention (anything "
        "staged after loading would run alone) and return numbers that do not "
        "match this trace. Remedy: re-capture with tl.trace(model, x, "
        "intervene=...) (load a recipe saved with save_intervention via "
        "tl.io.load_intervention_spec), or capture plainly with tl.trace(model, x) "
        "for the un-intervened model",
        code="run_intervention_spec_not_persisted",
    )
