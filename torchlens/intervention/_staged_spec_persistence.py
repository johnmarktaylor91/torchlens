"""Honest persistence of a trace's staged intervention spec.

``Trace._intervention_spec`` is session-only (``FieldPolicy.DROP``): an
ordinary save keeps the INTERVENED activations and their per-op provenance
(``intervention_replaced``, fire records, ``intervention_audit``) but not the
recipe that produced them. The recipe travels separately through
``save_intervention``; the runnable save already refuses intervened captures
(``user_intervention_not_replayable``) and names the analysis save as the door
for them. The one place the split could mislead is the legacy rerun of a loaded
artifact whose recorded values were intervened: with no spec staged it would
silently re-execute the un-intervened model, so it refuses instead.
"""

from __future__ import annotations

from typing import Any

from .errors import EngineDispatchError

_LOADED_PROVIDERS = frozenset({"loaded_sparse", "loaded_analysis"})


def _staged_entry_counts(trace: Any) -> tuple[int, int]:
    """Return the staged hook and value-replacement counts of a trace."""

    spec = trace._intervention_spec
    return (
        len(getattr(spec, "hook_specs", None) or ()),
        len(getattr(spec, "target_value_specs", None) or ()),
    )


def _carries_intervened_values(trace: Any) -> bool:
    """Return whether persisted evidence says recorded values were intervened."""

    from .._capture_honesty import intervention_facts

    facts = intervention_facts(trace)
    if facts is None:
        return False
    return bool(facts["replaced_op_count"] or facts["audit_row_count"])


def refuse_unpersisted_intervention_rerun(trace: Any, provider: Any) -> None:
    """Refuse a legacy rerun of a loaded intervened trace with no staged spec.

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
        an artifact whose recorded values were intervened while nothing is
        staged to reproduce that intervention. Live traces are never refused:
        an empty staged spec there means the user detached it on purpose.
    """

    if str(getattr(provider, "value", provider)) not in _LOADED_PROVIDERS:
        return
    if any(_staged_entry_counts(trace)):
        return
    if not _carries_intervened_values(trace):
        return
    raise EngineDispatchError(
        "This trace was loaded from an artifact whose recorded values were "
        "intervened, but tl.save() does not persist the intervention spec, so "
        "run(model, x) would silently return UN-intervened numbers. Remedy: "
        "re-capture with tl.trace(model, x, intervene=...) (load a recipe saved "
        "with save_intervention via tl.io.load_intervention_spec), or capture "
        "plainly with tl.trace(model, x) for the un-intervened model",
        code="run_intervention_spec_not_persisted",
    )
