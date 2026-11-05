"""Intervention spec, hook-plan normalization and selector checks for the PREPARE stage.

Moved verbatim from ``torchlens._capture_intervention and torchlens._user_funcs`` by the capture seam;
the door's predicate refusals and selector resolution moved verbatim from ``prepare.door``.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from dataclasses import replace
from typing import Any

from torchlens._data_classes.trace import Trace
from torchlens._errors import ArgumentTypeError, InvalidArgumentError
from torchlens._selection.selectors import BaseSelector
from torchlens.intervention._zero_fire import (
    _unfired_forward_rules,
    non_tensor_reason_fields,
    recorded_non_tensor_details,
    settle_capture_zero_fire,
)
from torchlens.intervention.audit import (
    EventStatus,
    record_intervention_event,
    rule_signal as _rule_signal,
    rules_payload,
    site_keys_for_labels,
)
from torchlens.intervention.hooks import split_site_branches as _split_site_branches

from ..._io._source_privacy import _relativize_path_literals
from ..._trace_selection_shape import (
    _layers_to_save_has_negative_index,
    _layers_to_save_live_subset,
    _layers_to_save_mentions_identity,
    _layers_to_save_mentions_output,
    _layers_to_save_needs_final_resolution,
)
from ..._trace_selector_helpers import (
    _combine_save_predicates,
    _is_selective_label_save,
    _make_layers_to_save_predicate,
)
from ...intervention.errors import MultiMatchWarning, SiteResolutionError
from ...intervention.hooks import gradient_inners, normalize_hook_plan, split_site_branches
from ...intervention.predicates import InterventionPredicate
from ...intervention.spec import Recipe, Rule
from ...intervention.types import InterventionDecision, InterventionSpec, TargetSpec
from .request import _DoorArgs, _DoorState


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


def _record_capture_intervention_event(
    trace: Trace, intervene_request: Any, *, fired_rule_ids: Collection[str]
) -> None:
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
    fired_rule_ids:
        Ids of the public-spec rules that fired during this capture (the
        per-capture fire ledger).
    """

    if intervene_request is None:
        return
    fired_labels, fire_count, _fired_records = _fired_capture_records(trace)
    zero_fire_rule_ids: tuple[str, ...] = ()
    unfired_rules: tuple[Rule, ...] = ()
    if isinstance(intervene_request, Recipe):
        rules = rules_payload(intervene_request)
        edit_names = tuple(str(rule["action"]) for rule in rules)
        selection_repr = " | ".join(str(rule["where"]) for rule in rules)
        unfired_rules = _unfired_forward_rules(intervene_request, fired_rule_ids)
        zero_fire_rule_ids = tuple(rule.rule_id for rule in unfired_rules)
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
    _settle_unfired_spec_rules(
        trace, intervene_request, unfired_rules, any_fired=bool(fired_rule_ids)
    )


def _settle_unfired_spec_rules(
    trace: Trace, intervene_request: Any, unfired_rules: tuple[Rule, ...], *, any_fired: bool
) -> None:
    """Settle a public spec's unfired rules under the capture's ``on_zero_fire``.

    Some rules fired and some did not: the fact is on the audit row; the
    policy settles the rest. A multi-clause spec that fired nowhere at all
    settles here too, naming every unfired rule and adding each to the
    persisted unmatched-selector ledger (it has no single selector, so the
    zero-match selector disclosure never sees it). A one-clause spec that
    fired nowhere settles at that selector disclosure instead, unchanged.

    Parameters
    ----------
    trace:
        Completed capture.
    intervene_request:
        The ``intervene=`` argument as configured.
    unfired_rules:
        The spec's unfired forward rules.
    any_fired:
        Whether any rule of the spec fired during the capture.
    """

    if not unfired_rules or not isinstance(intervene_request, Recipe):
        return
    zero_fire_rule_ids = [rule.rule_id for rule in unfired_rules]
    names = ", ".join(zero_fire_rule_ids)
    if any_fired:
        settle_capture_zero_fire(
            f"Capture-time intervention rule(s) [{names}] fired at zero sites while "
            "other rules fired; the capture completed with those rules unapplied.",
            zero_fire_rule_ids=zero_fire_rule_ids,
        )
        return
    if len(intervene_request.rules) < 2:
        return
    for rule in unfired_rules:
        _record_unmatched_selector(trace, "intervene", rule.target, _rule_signal(rule.target))
    settle_capture_zero_fire(
        f"Capture-time intervention rule(s) [{names}] fired at zero sites; no intervention fired.",
        zero_fire_rule_ids=zero_fire_rule_ids,
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


def _warn_zero_match_capture_selectors(
    trace: Trace,
    *,
    save_selector: Any,
    intervene_selector: Any,
    halt_selector: Any = None,
    layers_to_save_request: Any = None,
) -> None:
    """Warn when a capture-time save/intervention/halt selector matched no sites.

    Parameters
    ----------
    trace:
        Completed trace carrying selector fire counts.
    save_selector:
        Capture-time save selector, if configured.
    intervene_selector:
        Capture-time intervention selector, if configured. Its site branches
        decide the warning timing: an autograd-node branch settles after the
        backward pass.
    halt_selector:
        Capture-time halt predicate, if configured. ``halt=`` was the one
        selector slot outside the zero-match disclosure family: a typo'd label
        selector silently ran the FULL forward (spending the memory and latency
        the halt was meant to avoid) and handed back the model's real outputs
        where the caller expected a frontier. Only ``BaseSelector`` halts are
        judged; an arbitrary value-dependent callable never firing is data, not a typo.
    layers_to_save_request:
        The ORIGINAL selective label-based ``layers_to_save`` request, when
        one was made. ``layers_to_save`` resolves through its own predicate/
        deferred machinery (never a ``BaseSelector``), so it sat outside this
        disclosure family: a typo'd layer name retained only the
        always-retained output tail and disclosed NOTHING, live and in the
        artifact -- the highest-traffic instance of the cc2cabbb class
        (grind-r6 b3 R15, opus MED). The request is re-resolved against the
        FINAL label space here; zero matches record and warn like the
        sibling slots.

    Returns
    -------
    None
        Emits at most one warning for each configured selector slot, and
        appends a matching string-only record to the PERSISTED
        ``trace.annotations["unmatched_capture_selectors"]`` ledger
        (B3R4-R15-1): the warning is the most losable disclosure kind, and
        without a durable record a zero-match ablation sweep read as "this
        layer does not matter" on the returned and saved Trace alike.
    """

    def _record_unmatched(slot: str, selector: Any, signal: str | None = None) -> None:
        """Append one zero-match fact to the persisted trace annotations."""

        _record_unmatched_selector(trace, slot, selector, signal)

    defer_backward_intervention = False
    try:
        if (
            isinstance(save_selector, BaseSelector)
            and int(getattr(trace, "_tl_save_selector_fire_count", 0)) == 0
        ):
            _record_unmatched("save", save_selector)
            warnings.warn(
                f"Capture-time save selector {save_selector!r} matched zero sites; "
                "no activations were selected by it.",
                UserWarning,
                stacklevel=3,
            )
        if isinstance(intervene_selector, BaseSelector):
            # A gradient or node branch fires at backward time: keep its fire
            # counter for the after-backward reconciliation.
            defer_backward_intervention = any(
                universe != "activation"
                for universe, _branch in _split_site_branches(intervene_selector)
            )
        if isinstance(intervene_selector, BaseSelector) and intervene_selector_fired_nowhere(
            trace, intervene_selector
        ):
            _record_unmatched("intervene", intervene_selector, _rule_signal(intervene_selector))
            settle_capture_zero_fire(
                f"Capture-time intervention selector {intervene_selector!r} matched zero sites; "
                "no intervention fired.",
                selector=repr(intervene_selector),
                **_non_tensor_zero_fire_fields(trace, intervene_selector),
            )
        if isinstance(halt_selector, BaseSelector) and not getattr(trace, "is_stopped", False):
            _record_unmatched("halt", halt_selector)
            warnings.warn(
                f"Capture-time halt selector {halt_selector!r} matched zero sites; "
                "the capture ran the full forward and completed without halting, so "
                "the outputs are the model's real outputs, not the intended frontier.",
                UserWarning,
                stacklevel=3,
            )
        if layers_to_save_request is not None:
            from .._input_records import _get_op_nums_from_user_labels

            try:
                matched: Any = _get_op_nums_from_user_labels(trace, layers_to_save_request)
            except InvalidArgumentError:
                # The post-capture lookup is loud for unknown keys; here the
                # capture already completed, so an unknown name IS the
                # zero-match fact to disclose, not grounds to destroy the
                # finished trace at return time.
                matched = []
            if not matched:
                _record_unmatched("layers_to_save", layers_to_save_request)
                warnings.warn(
                    f"layers_to_save={layers_to_save_request!r} matched zero layers; "
                    "only the always-retained output tail was saved. Check the "
                    "layer names against trace.layer_labels.",
                    UserWarning,
                    stacklevel=3,
                )
    finally:
        trace.__dict__.pop("_tl_save_selector_fire_count", None)
        if not defer_backward_intervention:
            trace.__dict__.pop("_tl_intervene_selector_fire_count", None)


def _record_unmatched_selector(
    trace: Any, slot: str, selector: Any, signal: str | None = None
) -> None:
    """Append one zero-match fact to the persisted ``unmatched_capture_selectors`` ledger.

    Parameters
    ----------
    trace:
        Completed capture.
    slot:
        Capture selector slot (``"save"``, ``"intervene"``, ...).
    selector:
        The selector (or rule WHERE term) that matched nothing.
    signal:
        The pass the selector fires in, when known.
    """

    annotations = getattr(trace, "annotations", None)
    if not isinstance(annotations, dict):
        return

    # R62: this ledger persists at every save level, so a selector repr
    # embedding an absolute host path (e.g. a file-payload predicate) must
    # go through the same path-relativization belt as every other
    # persisted repr.
    entry: dict[str, str] = {
        "slot": slot,
        "selector": _relativize_path_literals(repr(selector)),
    }
    if signal is not None:
        entry["signal"] = str(signal)
    annotations.setdefault("unmatched_capture_selectors", []).append(entry)


def _non_tensor_zero_fire_fields(trace: Any, selector: Any) -> dict[str, Any]:
    """Zero-fire ``reason`` fields for targeted modules whose output was not a tensor.

    Only a plain ``tl.module`` union names its modules exactly; any other selector
    keeps the generic zero-fire message.
    """

    from ..predicates import _plain_module_union

    addresses = _plain_module_union(selector)
    if not addresses:
        return {}
    return non_tensor_reason_fields(recorded_non_tensor_details(trace, addresses))


def _refuse_predicate_shapes(args: _DoorArgs) -> None:
    """Refuse non-callable predicates and out-of-range lookback settings."""
    if args.intervene is not None and not callable(args.intervene):
        raise ArgumentTypeError(
            f"intervene received non-callable type {type(args.intervene).__name__}",
            code="intervention_predicate_type_invalid",
            remedy="pass a predicate callable such as tl.when(...) or None",
            argument="intervene",
            received_type=type(args.intervene).__name__,
        )
    if args.halt is not None and not callable(args.halt):
        raise ArgumentTypeError(
            f"halt received non-callable type {type(args.halt).__name__}",
            code="halt_predicate_type_invalid",
            remedy="pass a predicate callable or None",
            argument="halt",
            received_type=type(args.halt).__name__,
        )
    if not isinstance(args.lookback, int) or not 0 <= args.lookback <= 1024:
        raise InvalidArgumentError(
            f"lookback={args.lookback!r} is not an integer in the supported range [0, 1024]",
            code="lookback_invalid",
            remedy="set lookback to an integer from 0 through 1024 inclusive",
            argument="lookback",
        )
    if args.lookback_payload_policy not in {
        "metadata_only",
        "detached_raw",
        "transformed",
        "grad_connected",
        "disk_spilled",
    }:
        raise InvalidArgumentError(
            f"lookback_payload_policy={args.lookback_payload_policy!r} is not supported",
            code="lookback_payload_policy_invalid",
            remedy=(
                "set lookback_payload_policy to 'metadata_only', 'detached_raw', "
                "'transformed', 'grad_connected', or 'disk_spilled'"
            ),
            argument="lookback_payload_policy",
        )


def _resolve_selectors(state: _DoorState) -> _DoorState:
    """Resolve the save and gradient selectors into live predicates and deferred requests."""
    layers_to_save = state.layers_to_save
    grads_to_save_resolved = state.grads_to_save_resolved
    save_predicate = state.save_predicate
    if type(layers_to_save) is str:
        layers_to_save = layers_to_save.lower()
    if type(grads_to_save_resolved) is str:
        grads_to_save_resolved = grads_to_save_resolved.lower()
    requested_layers_to_save = layers_to_save
    uses_deferred_gradients = grads_to_save_resolved not in ["all", "none", None, []]
    uses_deferred_activation = _is_selective_label_save(layers_to_save) and (
        _layers_to_save_mentions_output(layers_to_save)
        or _layers_to_save_has_negative_index(layers_to_save)
        or _layers_to_save_mentions_identity(layers_to_save)
        # Integer ordinals and final-label-shaped strings are defined against
        # FINAL layer numbering, which orphan removal renumbers after capture;
        # they must resolve post-postprocess, never against raw indexes.
        or _layers_to_save_needs_final_resolution(layers_to_save)
    )
    live_layers_to_save = (
        _layers_to_save_live_subset(layers_to_save) if uses_deferred_activation else layers_to_save
    )
    needs_live_output_projection = (
        _is_selective_label_save(layers_to_save)
        and state.should_save_grads
        and not uses_deferred_activation
    )
    uses_selective_layers_to_save = _is_selective_label_save(layers_to_save)
    if uses_selective_layers_to_save and not uses_deferred_activation:
        layers_predicate = _make_layers_to_save_predicate(layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
        layers_to_save = "all"
    elif uses_deferred_activation and live_layers_to_save is not None:
        layers_predicate = _make_layers_to_save_predicate(live_layers_to_save)
        save_predicate = _combine_save_predicates(save_predicate, layers_predicate)
    return replace(
        state,
        layers_to_save=layers_to_save,
        requested_layers_to_save=requested_layers_to_save,
        save_predicate=save_predicate,
        grads_to_save_resolved=grads_to_save_resolved,
        uses_deferred_gradients=uses_deferred_gradients,
        uses_deferred_activation=uses_deferred_activation,
        needs_live_output_projection=needs_live_output_projection,
        uses_selective_layers_to_save=uses_selective_layers_to_save,
    )


def _validate_predicate_capabilities(state: _DoorState) -> None:
    """Refuse followed_by predicates in slots whose capability cannot honor them."""
    save_predicate = state.save_predicate
    intervene = state.args.intervene
    halt = state.halt
    if save_predicate is not None or intervene is not None or halt is not None:
        from ..predicates import validate_followed_by_capability

    if save_predicate is not None:
        validate_followed_by_capability(
            save_predicate,
            api_name="trace(save=...)",
            supports_retroactive=True,
        )
    if intervene is not None:
        validate_followed_by_capability(
            intervene,
            api_name="trace(intervene=...)",
            supports_retroactive=False,
        )
    if halt is not None:
        validate_followed_by_capability(
            halt,
            api_name="trace(stop_after=...)",
            supports_retroactive=False,
        )
