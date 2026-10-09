"""Rerun support for interventions staged through the capture predicate door.

A capture-time ``intervene=`` predicate that is not lowered to module hooks
fires at individual ops, and each fire is staged on the trace as a hook on that
op's FINAL label (the persisted addressing, ``spec_derived=True`` on save).
Final labels exist only after postprocessing, so the live rerun matcher refuses
them. A rerun therefore re-arms the trace's own retained predicate through the
same capture door, hands the capture a copy of the staged spec without those
entries, and then checks that the door re-staged exactly the entries the trace
carries. The staged spec itself is never mutated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from .errors import ControlFlowDivergenceError, LiveModeLabelError

_LABEL_SELECTOR_KINDS = frozenset({"label", "contains", "regex"})


@dataclass(frozen=True)
class RerunSpecPlan:
    """How one rerun capture applies a trace's staged spec.

    Attributes
    ----------
    capture_spec:
        Spec handed to the capture: the staged spec itself, or a copy without
        the predicate-door entries when the predicate is re-armed.
    intervene_predicate:
        The trace's retained capture predicate to re-arm, or ``None``.
    staged_keys:
        Identity keys of the staged predicate-door entries, or ``None``.
    staged_record_count:
        Number of fire records the staged spec held when the plan was built.
    """

    capture_spec: Any
    intervene_predicate: Any | None = None
    staged_keys: Counter[tuple[Any, ...]] | None = None
    staged_record_count: int = 0


def is_predicate_door_hook(hook_spec: Any) -> bool:
    """Return whether a staged hook was recorded by a capture predicate fire.

    Parameters
    ----------
    hook_spec:
        Staged ``HookSpec``.

    Returns
    -------
    bool
        True for the per-op label entries the predicate door records. Module
        selectors lowered to hooks at capture keep their module target and are
        live-matchable, so they are not predicate-door entries.
    """

    metadata = getattr(hook_spec, "metadata", None) or {}
    target = getattr(hook_spec, "site_target", None)
    return (
        metadata.get("created_by") == "intervene_predicate"
        and getattr(target, "selector_kind", None) == "label"
    )


def _hook_key(hook_spec: Any) -> tuple[Any, ...]:
    """Return the identity a re-armed predicate must reproduce for one entry.

    Parameters
    ----------
    hook_spec:
        Predicate-door ``HookSpec``.

    Returns
    -------
    tuple[Any, ...]
        Target, originating rule, direction, timing and helper identity.
    """

    metadata = getattr(hook_spec, "metadata", None) or {}
    helper = getattr(hook_spec, "helper", None)
    hook = getattr(hook_spec, "hook", None)
    hook_identity = (
        getattr(helper, "helper_name", None)
        if helper is not None
        else getattr(hook, "__qualname__", type(hook).__name__)
    )
    return (
        hook_spec.site_target.selector_value,
        metadata.get("spec_rule_id"),
        metadata.get("direction"),
        metadata.get("timing"),
        hook_identity,
    )


def _without_rule_id(keys: Counter[tuple[Any, ...]]) -> Counter[tuple[Any, ...]]:
    """Return ``keys`` with the originating rule identity dropped from each entry."""

    stripped: Counter[tuple[Any, ...]] = Counter()
    for key, count in keys.items():
        stripped[key[:1] + key[2:]] += count
    return stripped


def plan_rerun_spec(log: Any, spec: Any) -> RerunSpecPlan:
    """Decide how a rerun capture applies the staged spec.

    Parameters
    ----------
    log:
        Trace being rerun.
    spec:
        The trace's staged intervention spec, or ``None``.

    Returns
    -------
    RerunSpecPlan
        A plan that re-arms the retained predicate when the staged spec holds
        predicate-door entries and the trace still carries that predicate;
        otherwise a plan that hands the staged spec through unchanged.
    """

    hook_specs = list(getattr(spec, "hook_specs", None) or ())
    door_hooks = [hook_spec for hook_spec in hook_specs if is_predicate_door_hook(hook_spec)]
    # Session-only field, absent on traces that never armed a predicate.
    options = log.__dict__.get("_predicate_save_options")
    predicate = getattr(options, "intervene", None)
    if not door_hooks or predicate is None:
        return RerunSpecPlan(capture_spec=spec)
    door_targets = {hook_spec.site_target.freeze() for hook_spec in door_hooks}
    kept_hooks = [hook_spec for hook_spec in hook_specs if not is_predicate_door_hook(hook_spec)]
    kept_targets = {hook_spec.site_target.freeze() for hook_spec in kept_hooks}
    capture_spec = replace(
        spec,
        # The door re-stages its own targets; keeping the staged final-label
        # copies would only duplicate them in the discarded rerun spec.
        targets=[
            target
            for target in spec.targets
            if target.freeze() not in door_targets or target.freeze() in kept_targets
        ],
        target_value_specs=list(spec.target_value_specs),
        hook_specs=kept_hooks,
        records=list(spec.records),
        metadata=dict(spec.metadata),
    )
    return RerunSpecPlan(
        capture_spec=capture_spec,
        intervene_predicate=predicate,
        staged_keys=Counter(_hook_key(hook_spec) for hook_spec in door_hooks),
        staged_record_count=len(spec.records),
    )


def settle_predicate_rerun(plan: RerunSpecPlan, staged_spec: Any, new_log: Any) -> None:
    """Check a re-armed predicate re-staged exactly the staged entries.

    Parameters
    ----------
    plan:
        Plan the rerun capture ran under.
    staged_spec:
        The trace's staged spec (receives the run's fire records on success).
    new_log:
        Fresh rerun capture, not yet swapped into the trace.

    Raises
    ------
    ControlFlowDivergenceError
        If the re-armed predicate fired at a different set of ops than the
        staged entries name (the input changed what the predicate matches, or
        some staged entries were detached). The trace is left untouched.
    """

    if plan.intervene_predicate is None:
        return
    rerun_spec = new_log._intervention_spec
    rerun_keys = Counter(
        _hook_key(hook_spec)
        for hook_spec in getattr(rerun_spec, "hook_specs", None) or ()
        if is_predicate_door_hook(hook_spec)
    )
    if rerun_keys != plan.staged_keys:
        staged_labels = sorted({str(key[0]) for key in plan.staged_keys or ()})
        rerun_labels = sorted({str(key[0]) for key in rerun_keys})
        if _without_rule_id(rerun_keys) == _without_rule_id(plan.staged_keys or Counter()):
            # Same ops, same helpers: only the rule identity moved. Rule ids
            # hash the rule's content, so the staged steer content (for
            # example a steer tensor edited in place) changed since capture.
            cause = (
                "The re-armed intervene= predicate fired at the staged ops "
                f"{staged_labels}, but its rule content no longer matches the "
                "staged entries: the staged steer (for example a steer tensor "
                "edited in place) changed since capture"
            )
        else:
            cause = (
                "This rerun re-armed the trace's capture-time intervene= predicate, "
                f"which fired at {rerun_labels} while the trace stages it at "
                f"{staged_labels}"
            )
        raise ControlFlowDivergenceError(
            f"{cause}; applying either set would not reproduce the staged "
            "intervention exactly, so the rerun is refused and the trace is unchanged. "
            "Remedy: capture fresh with tl.trace(model, x, intervene=...) for this "
            "input, or re-stage the intervention with a module or function selector "
            "through attach_hooks()",
            code="rerun_predicate_restage_mismatch",
            staged_sites=tuple(staged_labels),
            rerun_sites=tuple(rerun_labels),
        )
    staged_spec.records.extend(plan.capture_spec.records[plan.staged_record_count :])


def refuse_unmatchable_staged_labels(capture_spec: Any) -> None:
    """Refuse a rerun whose staged edits are addressed by finalized labels.

    A live rerun matches selectors while the forward runs, before postprocess
    labels exist, so an entry staged on a final label (``Trace.set(label, ...)``,
    ``attach_hooks(tl.label(...), ...)``, or a predicate-door entry whose
    predicate the trace no longer carries) can never fire. The live matcher
    would refuse it mid-capture with a remedy aimed at authoring selectors; this
    refuses before the forward, names the staged entry, and keeps the trace
    unchanged.

    Parameters
    ----------
    capture_spec:
        Spec the rerun capture would apply (predicate-door entries already
        removed when the predicate is re-armed).

    Raises
    ------
    LiveModeLabelError
        ``rerun_staged_label_unmatchable`` when any staged value or hook entry
        names a finalized label.
    """

    entries = [
        *(getattr(capture_spec, "target_value_specs", None) or ()),
        *(getattr(capture_spec, "hook_specs", None) or ()),
    ]
    from ..ir.selector_eval import looks_like_finalized_label

    for entry in entries:
        target = entry.site_target
        value = target.selector_value
        if target.selector_kind not in _LABEL_SELECTOR_KINDS or not isinstance(value, str):
            continue
        if not looks_like_finalized_label(value):
            continue
        metadata = getattr(entry, "metadata", None) or {}
        origin = (
            "a capture-time intervene= predicate this trace no longer carries"
            if metadata.get("created_by") == "intervene_predicate"
            else "Trace.set() or attach_hooks()"
        )
        raise LiveModeLabelError(
            f"This trace stages an edit on tl.{target.selector_kind}({value!r}), staged by "
            f"{origin}. That is a finalized postprocess label, and a rerun matches "
            "selectors during the forward, before those labels exist, so the edit "
            "could never fire; the rerun is refused and the trace is unchanged. "
            "Remedy: re-stage the edit with a selector a live forward can match, "
            "for example trace.set(tl.module('fc2'), value) or "
            "trace.attach_hooks(tl.func('relu'), helper), or propagate the edit on "
            "the captured input with tl.push",
            code="rerun_staged_label_unmatchable",
            staged_selector=f"{target.selector_kind}:{value}",
        )
