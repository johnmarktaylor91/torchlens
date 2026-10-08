"""Honest persistence of a trace's staged intervention spec.

``Trace._intervention_spec`` is session-only (``FieldPolicy.DROP``): an
ordinary save keeps the INTERVENED activations and their per-op provenance
(``intervention_replaced``, fire records, ``intervention_audit``) but not the
recipe that produced them. The recipe travels separately through
``save_intervention``. Two guards keep that split honest: the save warns when it
leaves a staged recipe behind, and the legacy rerun of a loaded artifact whose
recorded values were intervened refuses instead of silently re-executing the
un-intervened model.
"""

from __future__ import annotations

import warnings
from typing import Any

from .errors import EngineDispatchError, TorchLensInterventionWarning

_LOADED_PROVIDERS = frozenset({"loaded_sparse", "loaded_analysis"})


def _staged_entry_counts(trace: Any) -> tuple[int, int]:
    """Return the staged hook and value-replacement counts of a trace."""

    spec = getattr(trace, "_intervention_spec", None)
    return (
        len(getattr(spec, "hook_specs", None) or ()),
        len(getattr(spec, "target_value_specs", None) or ()),
    )


def warn_staged_spec_not_persisted(trace: Any) -> None:
    """Warn when a save drops a non-empty staged intervention spec.

    Parameters
    ----------
    trace:
        Trace being saved.

    Returns
    -------
    None
        Emits one ``TorchLensInterventionWarning`` when the staged spec holds
        hooks or value replacements.
    """

    staged_hooks, staged_values = _staged_entry_counts(trace)
    if not staged_hooks and not staged_values:
        return
    warnings.warn(
        "This trace stages an intervention spec "
        f"({staged_hooks} hook(s), {staged_values} value replacement(s)) that "
        "tl.save() does not persist: the artifact keeps the intervened values, "
        "and a loaded copy refuses run(model, x) rather than rerun without the "
        "intervention. Save the recipe too with "
        "trace.save_intervention(path, level=...) and re-apply it with "
        "tl.trace(model, x, intervene=...).",
        TorchLensInterventionWarning,
        stacklevel=4,
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
