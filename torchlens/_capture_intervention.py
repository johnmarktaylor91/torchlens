"""Capture-door intervention glue for ``tl.trace(intervene=...)``.

Split out of ``user_funcs.py`` (C03 fix cycle): the backward sticky-spec
builder for backward-only selectors and the capture-door fire-evidence
envelope (ledger memo 3.1) are intervention machinery, not trace-entry
resolution; ``user_funcs`` imports them from here.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from .intervention.hooks import normalize_hook_plan
from .intervention.predicates import InterventionPredicate
from .intervention.resolver import _selector_resolution_direction
from .intervention.types import InterventionDecision, InterventionSpec, TargetSpec

if TYPE_CHECKING:
    from .data_classes.trace import Trace


def _backward_intervention_spec_from_predicate(
    intervene_predicate: InterventionPredicate | None,
) -> InterventionSpec | None:
    """Build a sticky intervention spec for backward-only ``tl.when`` predicates.

    Parameters
    ----------
    intervene_predicate:
        Predicate supplied to ``trace(intervene=...)``.

    Returns
    -------
    InterventionSpec | None
        Spec containing one backward hook for a backward selector, or ``None``.
    """

    if intervene_predicate is None:
        return None
    from .intervention.spec import InterventionSpec as PublicInterventionSpec

    if isinstance(intervene_predicate, PublicInterventionSpec):
        # C03 spec door: EVERY backward rule of a multi-clause spec builds its
        # sticky backward hook -- the single-selector read below would have
        # silently dropped all but a lone clause (the exact silent-drop class
        # the spec noun exists to kill).
        rule_pairs: list[tuple[Any, Any]] = [
            (rule.where, rule.decision) for rule in intervene_predicate.rules
        ]
    else:
        rule_pairs = [
            (
                getattr(intervene_predicate, "selector", None),
                getattr(intervene_predicate, "decision", None),
            )
        ]
    spec: InterventionSpec | None = None
    for selector, decision in rule_pairs:
        if selector is None or not isinstance(decision, InterventionDecision):
            continue
        selector_direction = _selector_resolution_direction(selector)
        if selector_direction == "backward" and decision.direction not in {"backward", "both"}:
            warnings.warn(
                "Forward intervention helper attached to a backward-only selector will not "
                "fire. Use a gradient helper such as tl.grad_zero(), tl.grad_scale(), or "
                "tl.bwd_hook().",
                UserWarning,
                stacklevel=3,
            )
            continue
        if selector_direction != "backward" or decision.direction not in {"backward", "both"}:
            continue
        if decision.hook is None:
            continue
        target = (
            selector.to_target_spec()
            if hasattr(selector, "to_target_spec")
            else TargetSpec("label", selector)
        )
        if spec is None:
            spec = InterventionSpec()
        spec.targets.append(target)
        entries = normalize_hook_plan(
            target,
            decision.hook,
            direction="backward",
        )
        for entry in entries:
            metadata = {
                **dict(entry.metadata),
                "created_by": "intervene_backward_selector",
                "direction": "backward",
            }
            spec.add_hook(
                target,
                entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
                helper=entry.helper_spec,
                metadata=metadata,
            )
    return spec


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
        record_intervention_event,
        rules_payload,
        site_keys_for_labels,
    )
    from .intervention.spec import InterventionSpec as PublicInterventionSpec

    fired_labels: list[str] = []
    fire_count = 0
    fired_records: list[Any] = []
    for label in trace.op_labels:
        op = trace.ops[label]
        records = getattr(op, "interventions", None) or ()
        if records:
            fired_labels.append(label)
            fire_count += len(records)
            fired_records.extend(records)
    zero_fire_rule_ids: tuple[str, ...] = ()
    if isinstance(intervene_request, PublicInterventionSpec):
        rules = rules_payload(intervene_request)
        edit_names = tuple(str(rule["action"]) for rule in rules)
        selection_repr = " | ".join(str(rule["where"]) for rule in rules)
        from .intervention.spec import _canonical_repr

        fired_action_reprs = {
            _canonical_repr(record.helper) for record in fired_records if record.helper is not None
        }
        zero_fire_rule_ids = tuple(
            str(rule["rule_id"]) for rule in rules if str(rule["action"]) not in fired_action_reprs
        )
    else:
        rules = ()
        decision = getattr(intervene_request, "decision", None)
        hook = getattr(decision, "hook", None) if decision is not None else None
        edit_name = getattr(
            hook, "helper_name", getattr(hook, "__name__", type(intervene_request).__name__)
        )
        edit_names = (str(edit_name),)
        selection_repr = repr(getattr(intervene_request, "selector", intervene_request))
    record_intervention_event(
        trace,
        lane="capture",
        door="trace",
        edit_names=edit_names,
        selection_repr=selection_repr,
        status="fired" if fire_count else "no_fire",
        fire_count=fire_count,
        site_keys=site_keys_for_labels(trace, tuple(fired_labels)),
        rules=rules,
        zero_fire_rule_ids=zero_fire_rule_ids,
    )
