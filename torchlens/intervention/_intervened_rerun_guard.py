"""Refuse the legacy rerun of a loaded trace whose intervention cannot be re-applied.

A 3.0 artifact carries the intervention recipe (``intervention.json``): the
reader restores its built-in edits into ``Trace._intervention_spec`` so every
rerun door runs them again. Two loaded shapes cannot be re-applied and refuse
instead, on every legacy door (plain, append, chunked, fork, and
``do(engine="rerun")``):

* the restored recipe holds a callable placeholder (a user callable the writer
  could not persist; the placeholder names its position and qualname), so a
  rerun would execute without that edit;
* the artifact predates the recipe (no recipe restored) while its recorded
  values were intervened (``is_edited``, fire records, ``intervention_audit``),
  so a rerun would execute without the recorded intervention.

A rerun of this session that already applied the staged spec replaces the
recorded values and lifts the refusal. Live traces are never refused.
"""

from __future__ import annotations

from typing import Any

from .errors import EngineDispatchError

_LOADED_PROVIDERS = frozenset({"loaded_sparse", "loaded_analysis"})
# Audit rows that record an edit which changed no recorded value (an armed
# gradient edit changes no forward value; its settle row says what fired).
_VALUE_NEUTRAL_AUDIT_STATUSES = frozenset({"no_fire", "error", "armed"})
_CODE = "run_intervention_spec_not_persisted"


def _carries_intervened_values(trace: Any) -> bool:
    """Return whether persisted evidence says recorded values were intervened."""

    for layer in trace._layer_list or ():
        if getattr(layer, "is_edited", False) or getattr(layer, "edits", None):
            return True
    return any(
        isinstance(row, dict) and row.get("status") not in _VALUE_NEUTRAL_AUDIT_STATUSES
        for row in getattr(trace, "intervention_audit", None) or ()
    )


def _session_spec_has_fired(trace: Any) -> bool:
    """Return whether a run in this session applied the trace's staged spec."""

    return bool(getattr(trace._intervention_spec, "records", None))


def _loaded_recipe(trace: Any) -> dict[str, Any] | None:
    """The recipe facts the 3.0 reader attached to the staged spec, or ``None``."""

    spec = getattr(trace, "_intervention_spec", None)
    recipe = (getattr(spec, "metadata", None) or {}).get("recipe")
    return recipe if isinstance(recipe, dict) else None


def _placeholder_sentence(placeholders: list[Any]) -> str:
    shown = ", ".join(f"rule {position} ({qualname})" for position, qualname in placeholders[:4])
    more = f" and {len(placeholders) - 4} more" if len(placeholders) > 4 else ""
    return f"{shown}{more}"


def refuse_unpersisted_intervention_rerun(trace: Any, provider: Any) -> None:
    """Refuse a legacy rerun of a loaded trace whose intervention cannot be re-applied.

    Parameters
    ----------
    trace:
        Trace whose legacy ``run(model, x)`` is about to execute.
    provider:
        The trace's run provider (``RunProvider`` or ``None`` for live traces).

    Raises
    ------
    EngineDispatchError
        ``run_intervention_spec_not_persisted`` when the restored recipe holds a
        callable placeholder (the message names each placeholder's rule position
        and qualname), or when no recipe was restored for an artifact whose
        recorded values were intervened. A rerun of this session that already
        applied the staged spec lifts the refusal; live traces are never refused.
    """

    if str(getattr(provider, "value", provider)) not in _LOADED_PROVIDERS:
        return
    if _session_spec_has_fired(trace):
        return
    recipe = _loaded_recipe(trace)
    if recipe is not None:
        placeholders = list(recipe.get("placeholders") or ())
        if not placeholders:
            return
        detail = (
            "This trace was loaded from an artifact whose intervention recipe holds a "
            f"callable that was not persisted ({_placeholder_sentence(placeholders)}), "
            "so run(model, x) would re-execute without that edit and return numbers that "
            "do not match this trace. Remedy: re-stage the callable on a fresh capture "
            "(tl.trace(model, x, intervene=...)), or use a built-in edit so the recipe persists"
        )
    elif _carries_intervened_values(trace):
        detail = (
            "This trace was loaded from an artifact whose recorded values were intervened "
            "but which carries no intervention recipe, so run(model, x) would re-execute "
            "without that intervention (anything staged after loading would run alone) and "
            "return numbers that do not match this trace. Remedy: re-capture with "
            "tl.trace(model, x, intervene=...) (load a recipe saved with save_intervention "
            "via tl.io.load_intervention_spec), or capture plainly with tl.trace(model, x) "
            "for the un-intervened model"
        )
    else:
        return
    raise EngineDispatchError(detail, code=_CODE)
