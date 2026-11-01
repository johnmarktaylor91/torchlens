"""Capture-door intervention glue for ``tl.trace(intervene=...)``.

Split out of ``user_funcs.py`` (C03 fix cycle): the backward sticky-spec
builder for backward-only selectors and the capture-door fire-evidence
envelope (ledger memo 3.1) are intervention machinery, not trace-entry
resolution; ``user_funcs`` imports them from here. The live hook-plan
spec builder and its non-mutating merge into an existing spec followed
for the same reason.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .intervention.hooks import gradient_inners, normalize_hook_plan, split_site_branches
from .intervention.predicates import InterventionPredicate
from .intervention.spec import Recipe, Rule, _canonical_repr
from .intervention.types import InterventionDecision, InterventionSpec, TargetSpec

if TYPE_CHECKING:
    from ._data_classes.trace import Trace


def _intervene_rule_pairs(intervene_predicate: Any) -> list[tuple[Any, Any]]:
    """Return ``(selector, decision)`` for every rule of a capture ``intervene=``.

    Parameters
    ----------
    intervene_predicate:
        Predicate or ``Recipe`` supplied to ``trace(intervene=...)``.

    Returns
    -------
    list[tuple[Any, Any]]
        One pair per rule (a bare predicate is one rule).
    """

    if isinstance(intervene_predicate, Recipe):
        # C03 spec door: EVERY rule of a multi-clause spec is read -- a
        # single-selector read would silently drop all but a lone clause.
        return [(rule.target, rule.decision) for rule in intervene_predicate.rules]
    return [
        (
            getattr(intervene_predicate, "selector", None),
            getattr(intervene_predicate, "decision", None),
        )
    ]


def _backward_intervention_spec_from_predicate(
    intervene_predicate: InterventionPredicate | None,
) -> InterventionSpec | None:
    """Build the sticky backward spec for the gradient and node branches of ``intervene=``.

    A rule's site decides its pass (SPEC 3.5): the activation branch fires
    live during the capture's forward, and each ``tl.grad(site)`` or
    autograd-node branch becomes a sticky backward hook that the next
    ``trace.backward(loss)`` fires. The branch is validated here, so an edit
    that reads activations on a gradient site refuses ``edit_signal_mismatch``
    at the capture door, before the capture runs.

    Parameters
    ----------
    intervene_predicate:
        Predicate supplied to ``trace(intervene=...)``.

    Returns
    -------
    InterventionSpec | None
        Spec holding one sticky hook per backward branch, or ``None``.
    """

    if intervene_predicate is None:
        return None
    spec: InterventionSpec | None = None
    for selector, decision in _intervene_rule_pairs(intervene_predicate):
        if selector is None or not isinstance(decision, InterventionDecision):
            continue
        if decision.hook is None:
            continue
        for universe, branch in split_site_branches(selector):
            if universe == "activation":
                continue
            target = _target_spec_for(branch)
            # Normalize the selector itself so refusals name the user's spelling.
            entries = normalize_hook_plan(branch, decision.hook)
            if spec is None:
                spec = InterventionSpec()
            spec.targets.append(target)
            for entry in entries:
                spec.add_hook(
                    target,
                    entry.helper_spec
                    if entry.helper_spec is not None
                    else entry.normalized_callable,
                    helper=entry.helper_spec,
                    metadata={
                        **dict(entry.metadata),
                        "created_by": "intervene_backward_selector",
                    },
                )
    return spec


def activation_only_predicate(intervene_predicate: Any) -> Any:
    """Return the view of a capture ``intervene=`` that the live forward evaluates.

    The live forward predicate matches rules against forward ops. A gradient
    address evaluates as its inner site there (the site plan, not the
    evaluator, decides the universe), so a rule's ``tl.grad(...)`` or node
    branch must not reach the forward predicate: it would edit the
    activation. Those branches arm as sticky backward hooks instead
    (:func:`_backward_intervention_spec_from_predicate`).

    Parameters
    ----------
    intervene_predicate:
        Predicate or ``Recipe`` supplied to ``trace(intervene=...)``.

    Returns
    -------
    Any
        The predicate unchanged when every rule is an activation rule; a
        ``Recipe`` of the activation branches otherwise; ``None`` when no rule
        has an activation branch.
    """

    if intervene_predicate is None:
        return None
    if not isinstance(intervene_predicate, Recipe):
        selector = getattr(intervene_predicate, "selector", None)
        if selector is not None and all(
            universe != "activation" for universe, _branch in split_site_branches(selector)
        ):
            return None
        return intervene_predicate
    rules: list[Any] = []
    changed = False
    for rule in intervene_predicate.rules:
        branches = split_site_branches(rule.target)
        activation = [branch for universe, branch in branches if universe == "activation"]
        if len(activation) == len(branches):
            rules.append(rule)
            continue
        changed = True
        if activation:
            rules.append(Rule(activation[0], rule.edit))
    if not changed:
        return intervene_predicate
    return Recipe(rules=tuple(rules)) if rules else None


def _target_spec_for(selector: Any) -> TargetSpec:
    """Return the stored target spec of one branch selector.

    Parameters
    ----------
    selector:
        Branch selector (a ``tl.grad`` address, a node selector, ...).

    Returns
    -------
    TargetSpec
        Target spec carrying the branch.
    """

    if isinstance(selector, TargetSpec):
        return selector
    if hasattr(selector, "to_target_spec"):
        return selector.to_target_spec()
    return TargetSpec("label", selector)


def _armed_branch_count(intervene_request: Any) -> int:
    """Return how many gradient or node branches a capture ``intervene=`` armed.

    Parameters
    ----------
    intervene_request:
        The ``intervene=`` argument as configured.

    Returns
    -------
    int
        Number of backward branches with an executable hook.
    """

    count = 0
    for selector, decision in _intervene_rule_pairs(intervene_request):
        if selector is None or getattr(decision, "hook", None) is None:
            continue
        count += sum(
            1 for universe, _branch in split_site_branches(selector) if universe != "activation"
        )
    return count


def _fired_capture_records(trace: Trace) -> tuple[list[str], int, list[Any]]:
    """Return the labels, count and records of the capture-time intervention fires.

    Parameters
    ----------
    trace:
        Completed capture.

    Returns
    -------
    tuple[list[str], int, list[Any]]
        Labels of the ops that recorded a fire (in op order), the total fire
        count, and every fire record.
    """

    fired_labels: list[str] = []
    fire_count = 0
    fired_records: list[Any] = []
    for label in trace.op_labels:
        op = trace.ops[label]
        records = op.edits or ()
        if records:
            fired_labels.append(label)
            fire_count += len(records)
            fired_records.extend(records)
    return fired_labels, fire_count, fired_records


def _record_capture_intervention_event(trace: Trace, intervene_request: Any) -> None:
    """Write the capture-door InterventionEvent envelope (C03 fire evidence).

    One envelope per intervened capture attempt -- fired or not. The capture
    door was one of the two measured audit-empty doors (ledger memo 3.1);
    zero matches are DATA (``status="no_fire"``), never silence.

    Parameters
    ----------
    trace:
        Completed capture.
    intervene_request:
        The ``intervene=`` argument as configured, when any.
    """

    if intervene_request is None:
        return
    from .intervention.audit import (
        EventStatus,
        record_intervention_event,
        rules_payload,
        site_keys_for_labels,
    )

    fired_labels, fire_count, fired_records = _fired_capture_records(trace)
    zero_fire_rule_ids: tuple[str, ...] = ()
    if isinstance(intervene_request, Recipe):
        rules = rules_payload(intervene_request)
        edit_names = tuple(str(rule["action"]) for rule in rules)
        selection_repr = " | ".join(str(rule["where"]) for rule in rules)
        fired_action_reprs = {
            _canonical_repr(record.edit) for record in fired_records if record.edit is not None
        }
        zero_fire_rule_ids = tuple(
            str(rule["rule_id"]) for rule in rules if str(rule["action"]) not in fired_action_reprs
        )
    else:
        rules = ()
        decision = getattr(intervene_request, "decision", None)
        hook = getattr(decision, "hook", None) if decision is not None else None
        edit_name = getattr(
            hook, "name", getattr(hook, "__name__", type(intervene_request).__name__)
        )
        edit_names = (str(edit_name),)
        selection_repr = repr(getattr(intervene_request, "selector", intervene_request))
    armed_edits = _armed_branch_count(intervene_request)
    status: EventStatus = "fired" if fire_count else ("armed" if armed_edits else "no_fire")
    record_intervention_event(
        trace,
        lane="capture",
        door="trace",
        edit_names=edit_names,
        selection_repr=selection_repr,
        status=status,
        fire_count=fire_count,
        site_keys=site_keys_for_labels(trace, tuple(fired_labels)),
        rules=rules,
        zero_fire_rule_ids=zero_fire_rule_ids,
        extra={"armed_edits": armed_edits, "awaiting": "backward"} if armed_edits else None,
    )
    if fire_count and zero_fire_rule_ids:
        # Some rules fired and some did not: the fact is on the audit row above;
        # the capture's on_zero_fire policy settles the rest. (A request that fired
        # nowhere at all settles at the zero-match selector disclosure instead.)
        from .intervention._zero_fire import settle_capture_zero_fire

        names = ", ".join(zero_fire_rule_ids)
        settle_capture_zero_fire(
            f"Capture-time intervention rule(s) [{names}] fired at zero sites while "
            "other rules fired; the capture completed with those rules unapplied.",
            zero_fire_rule_ids=list(zero_fire_rule_ids),
        )


def _intervention_spec_from_hook_plan(hook_plan: Any) -> InterventionSpec | None:
    """Build an intervention spec for live hook-plan capture.

    Parameters
    ----------
    hook_plan:
        Normalized live hook entries.

    Returns
    -------
    InterventionSpec | None
        Spec carrying hook entries, or ``None`` when no hook plan exists.
    """

    if not hook_plan:
        return None
    spec = InterventionSpec()
    target_keys: set[Any] | None = set()
    for entry in hook_plan:
        site_target = entry.site_target
        if isinstance(site_target, TargetSpec):
            target = site_target
        elif hasattr(site_target, "to_target_spec"):
            target = site_target.to_target_spec()
        else:
            target = TargetSpec("label", site_target)
        frozen_target = target.freeze()
        if target_keys is not None:
            try:
                target_is_new = frozen_target not in target_keys
            except TypeError:
                target_keys = None
                target_is_new = not any(
                    existing.freeze() == frozen_target for existing in spec.targets
                )
        else:
            target_is_new = not any(existing.freeze() == frozen_target for existing in spec.targets)
        if target_is_new:
            spec.targets.append(target)
            if target_keys is not None:
                target_keys.add(frozen_target)
        spec.add_hook(
            target,
            entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
            helper=entry.helper_spec,
            metadata=dict(entry.metadata),
        )
    return spec


def _merge_intervention_spec_hooks(
    destination: InterventionSpec,
    source: InterventionSpec | None,
) -> InterventionSpec:
    """Return a spec holding an existing spec's entries plus hook-plan entries.

    Parameters
    ----------
    destination:
        Spec whose entries come first. Never mutated: it may be a spec a
        Trace already stores.
    source:
        Spec created from normalized hook entries.

    Returns
    -------
    InterventionSpec
        ``destination`` itself when ``source`` is ``None``, else a new spec
        holding ``destination``'s entries followed by ``source``'s.
    """

    if source is None:
        return destination
    targets = list(destination.targets)
    try:
        target_keys: set[Any] | None = {existing.freeze() for existing in targets}
    except TypeError:
        target_keys = None
    for target in source.targets:
        frozen_target = target.freeze()
        if target_keys is not None:
            try:
                target_is_new = frozen_target not in target_keys
            except TypeError:
                target_keys = None
                target_is_new = not any(existing.freeze() == frozen_target for existing in targets)
        else:
            target_is_new = not any(existing.freeze() == frozen_target for existing in targets)
        if target_is_new:
            targets.append(target)
            if target_keys is not None:
                target_keys.add(frozen_target)
    return replace(
        destination,
        targets=targets,
        target_value_specs=list(destination.target_value_specs),
        hook_specs=[*destination.hook_specs, *source.hook_specs],
        records=list(destination.records),
        metadata=dict(destination.metadata),
    )


def intervene_selector_fired_nowhere(trace: Trace, selector: Any) -> bool:
    """Return whether a capture-time ``intervene=`` selector addressed nothing.

    The site decides the branch. An activation branch counts its forward
    fires. A gradient branch (``tl.grad(site)``) fires only at backward time,
    so its zero-match fact is its forward site resolving to no captured op.
    A node branch defers to the after-backward check and never counts here.

    Parameters
    ----------
    trace:
        Completed capture.
    selector:
        Capture-time ``intervene=`` selector.

    Returns
    -------
    bool
        ``True`` when every branch matched zero sites.
    """

    from .intervention.errors import MultiMatchWarning, SiteResolutionError

    branches = split_site_branches(selector)
    if any(universe == "node" for universe, _branch in branches):
        return False
    if any(universe == "activation" for universe, _branch in branches) and int(
        getattr(trace, "_tl_intervene_selector_fire_count", 0)
    ):
        return False
    fanout = len(trace._layer_list or ()) + 1
    for universe, branch in branches:
        if universe != "gradient":
            continue
        for inner in gradient_inners(branch):
            try:
                with warnings.catch_warnings():
                    # Only the zero-match fact is asked; fan-out is the user's design.
                    warnings.simplefilter("ignore", MultiMatchWarning)
                    if trace._resolve_sites(inner, max_fanout=fanout):
                        return False
            except SiteResolutionError:
                continue
    return True
