"""Gradient sites at the trace doors: arming, settling and the armed read side.

``Trace.do`` routes a plan whose site addresses the backward pass
(``tl.grad(site)`` or an autograd-node selector) here. The plan is split into
its activation half and its backward half, preflighted whole, and armed: a
rerun arms the backward half on a fork (the result), ``engine="set_only"`` arms
it in place. Each backward pass end writes one settle row per arming
transaction (:func:`settle_armed_gradient_edits`, called from the torch
backward door). The mixin methods in ``_trace_intervention`` delegate here.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .._errors import InvalidArgumentError
from .._trace_state import TraceState
from ..options import InterventionOptions, ReplayOptions

if TYPE_CHECKING:
    from ._trace_intervention import _DoTransaction
    from .trace import Trace

#: HookSpec metadata naming the attach batch that armed a gradient rule.
_ARMING_GROUP_KEY = "arming_group"
#: HookSpec metadata naming the ``do`` arming row (``None`` for attach_hooks).
_ARMING_EVENT_KEY = "arming_event_id"
#: Arming group of the sticky backward hooks a capture-time ``intervene=``
#: staged (marked ``created_by`` by the capture door); their arming row is the
#: capture's own envelope (``door="trace"``).
_CAPTURE_ARMING_GROUP = "capture"
_CAPTURE_HOOK_MARK = "intervene_backward_selector"


def _arming_group(hook_spec: Any) -> str | None:
    """Return the arming group of one sticky hook, or ``None`` for a forward hook."""

    group = hook_spec.metadata.get(_ARMING_GROUP_KEY)
    if group is not None:
        return str(group)
    if hook_spec.metadata.get("created_by") == _CAPTURE_HOOK_MARK:
        return _CAPTURE_ARMING_GROUP
    return None


def _arming_event_id(trace: Any, hook_spec: Any) -> str | None:
    """Return the arming row id of one gradient rule (the capture envelope for capture rules)."""

    event_id = hook_spec.metadata.get(_ARMING_EVENT_KEY)
    if event_id is not None or _arming_group(hook_spec) != _CAPTURE_ARMING_GROUP:
        return None if event_id is None else str(event_id)
    for row in getattr(trace, "state_history", ()):
        if isinstance(row, dict) and row.get("op") == "intervention_event":
            if row.get("door") == "trace":
                return str(row.get("event_id"))
    return None


@dataclass(frozen=True)
class _SignalPlan:
    """A ``do()`` plan split into its activation half and its backward half.

    Each half is a ``(site, edit)`` door input (a list of pairs with ``edit``
    ``None``, or a ``Recipe`` with ``edit`` ``None``), or ``None`` when the
    plan has no site in that pass.
    """

    hooks_or_site: Any
    value_or_hook: Any
    forward: tuple[Any, Any] | None
    backward: tuple[Any, Any] | None

    @property
    def backward_repr(self) -> str:
        """Human rendering of the backward half's sites (for refusals)."""

        assert self.backward is not None
        site = self.backward[0]
        if _is_public_recipe(site):
            return ", ".join(repr(rule.target) for rule in site.rules)
        if isinstance(site, list):
            return ", ".join(repr(pair_site) for pair_site, _edit in site)
        return repr(site)


def _is_public_recipe(value: Any) -> bool:
    """Whether ``value`` is the public immutable ``Recipe`` spec."""

    from ..intervention.spec import Recipe

    return isinstance(value, Recipe)


def _door_pairs(hooks_or_site: Any, value_or_hook: Any) -> list[tuple[Any, Any]] | None:
    """Lower the ``do()`` call shapes that can carry a gradient site to pairs.

    Returns ``None`` for shapes that never address a gradient site
    (selection batches, regions, slices), which keep their own doors.
    """

    if _is_public_recipe(hooks_or_site):
        return [(rule.target, rule.edit) for rule in hooks_or_site.rules]
    if value_or_hook is None and isinstance(hooks_or_site, Mapping):
        return list(hooks_or_site.items())
    if value_or_hook is None and isinstance(hooks_or_site, (list, tuple)):
        if all(isinstance(pair, tuple) and len(pair) == 2 for pair in hooks_or_site):
            return list(hooks_or_site)
        return None
    return [(hooks_or_site, value_or_hook)]


def signal_plan(hooks_or_site: Any, value_or_hook: Any) -> _SignalPlan | None:
    """Split a ``do()`` input into its activation and backward halves.

    Returns ``None`` when no site addresses the backward pass, so plain
    activation edits keep the unchanged ``do`` path.
    """

    from ..intervention.spec import Rule, site_signal, split_site_by_signal

    pairs = _door_pairs(hooks_or_site, value_or_hook)
    if pairs is None or all(site_signal(site) == "forward" for site, _edit in pairs):
        return None
    forward: list[tuple[Any, Any]] = []
    backward: list[tuple[Any, Any]] = []
    for site, edit in pairs:
        forward_site, backward_site = split_site_by_signal(site)
        if forward_site is not None:
            forward.append((forward_site, edit))
        if backward_site is not None:
            backward.append((backward_site, edit))

    def _half(half: list[tuple[Any, Any]]) -> tuple[Any, Any] | None:
        """Rebuild one half in the caller's own door shape."""

        if not half:
            return None
        if _is_public_recipe(hooks_or_site):
            from ..intervention.spec import Recipe

            return Recipe(rules=tuple(Rule(site, edit) for site, edit in half)), None
        if len(half) == 1 and value_or_hook is not None:
            return half[0]
        return list(half), None

    return _SignalPlan(
        hooks_or_site=hooks_or_site,
        value_or_hook=value_or_hook,
        forward=_half(forward),
        backward=_half(backward),
    )


def refuse_invalid_do_loss(loss: Any, plan: _SignalPlan | None) -> None:
    """Refuse a ``loss=`` argument to ``do`` before any mutation."""

    from ..intervention.spec import refuse_invalid_loss

    refuse_invalid_loss(loss, has_gradient_site=plan is not None, door="do")


def is_gradient_entry(entry: Any) -> bool:
    """Whether a normalized hook entry's site addresses the backward pass."""

    from ..intervention.spec import site_signal

    return site_signal(getattr(entry, "site_target", None)) == "backward"


def do_signal_plan(
    trace: Trace,
    plan: _SignalPlan,
    *,
    model: nn.Module | None,
    x: Any,
    intervention_options: InterventionOptions,
    loss: Callable[[Trace], torch.Tensor] | None,
) -> Trace:
    """Run one ``do()`` whose plan has a gradient (or autograd-node) site.

    The engine is chosen and the WHOLE plan (both halves, every edit) is
    normalised and resolved before anything mutates, so a refused
    gradient edit leaves no activation edit behind. ``rerun`` runs the
    activation half live on a fork and arms the gradient half on that
    fork only; ``set_only`` arms both on this trace. The canonical audit
    row says ``"armed"`` until a backward pass settles it.

    Parameters
    ----------
    trace:
        The trace ``do`` was called on (the source of a rerun).
    plan:
        The site plan split into its activation and backward halves.
    model:
        Model for the rerun.
    x:
        Input for the rerun.
    intervention_options:
        Merged ``InterventionOptions``.
    loss:
        Optional callable of the returned trace; runs the backward here.

    Returns
    -------
    Trace
        The armed trace: a new trace after a rerun, this trace under
        ``set_only``.
    """

    from ._trace_intervention import _DoTransaction

    engine = select_signal_plan_engine(
        trace, intervention_options.engine, model=model, x=x, plan=plan
    )
    strict = intervention_options.strict
    preflight_txn = _DoTransaction(
        hooks_or_site=plan.hooks_or_site,
        value_or_hook=plan.value_or_hook,
        engine=engine,
        transaction_start=time.monotonic(),
        audit_rows_before=len(trace.intervention_audit),
    )
    try:
        preflight_signal_plan(trace, plan, strict=strict)
        if engine == "rerun":
            assert model is not None
            trace._validate_supplied_model_matches_capture(model)
    except BaseException as exc:
        trace._record_do_intervention_event(
            preflight_txn, status="error", error=repr(exc), event_row=True
        )
        raise
    if engine == "rerun":
        result = trace._fork_trace(name=None)
    else:
        trace._warn_if_root_mutation(confirm_mutation=intervention_options.confirm_mutation)
        result = trace
    txn = preflight_txn._replace(audit_rows_before=len(result.intervention_audit))
    attached: list[str] = []
    try:
        backward_handles, forced = arm_signal_plan(
            result, plan, engine=engine, strict=strict, model=model, x=x, attached=attached
        )
    except BaseException as exc:
        detach_handles(result, attached)
        trace._record_do_intervention_event(
            preflight_txn._replace(audit_rows_before=len(trace.intervention_audit)),
            status="error",
            error=repr(exc),
            event_row=True,
        )
        raise
    envelope = result._record_do_intervention_event(
        txn,
        status=None if plan.forward is not None else "armed",
        staged_only=engine == "set_only" and plan.forward is not None,
        extra_fields={"armed_edits": len(backward_handles), "awaiting": "backward"},
        event_row=True,
    )
    stamp_arming_event(result, backward_handles, str(envelope["event_id"]))
    if forced:
        result.last_run["backward_ready_forced"] = True
    result._record_operation(
        "do",
        mutation_kind="signal_plan",
        engine=engine,
        requested_engine=intervention_options.engine,
        model_supplied=model is not None,
        x_supplied=x is not None,
        strict=strict,
        armed_edits=len(backward_handles),
    )
    if plan.forward is not None and engine == "rerun":
        from .._capture._episode_coupling import quarantine_episode_after_perturbed_replay

        quarantine_episode_after_perturbed_replay(result)
    if loss is not None:
        run_do_loss(result, loss, attached=tuple(attached), txn=txn, source=trace)
    return result


def select_signal_plan_engine(
    trace: Trace,
    engine: str,
    *,
    model: nn.Module | None,
    x: Any,
    plan: _SignalPlan,
) -> str:
    """Pick the engine for a gradient plan, refusing replay before any mutation.

    Raises
    ------
    InvalidArgumentError
        ``intervention_engine_invalid`` for an unknown engine, and
        ``replay_gradient_edit_unsupported`` for ``engine="replay"`` or an
        ``auto`` engine with neither ``model=`` nor ``x=``.
    EngineDispatchError
        When only one of ``model=`` / ``x=`` is given.
    """

    from ..intervention.replay import gradient_edit_replay_text

    if engine not in {"auto", "replay", "rerun", "set_only"}:
        raise InvalidArgumentError(
            "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'; "
            f"received {engine!r}",
            code="intervention_engine_invalid",
            remedy="pass engine='auto', 'replay', 'rerun', or 'set_only'",
            argument="engine",
        )
    if engine == "set_only":
        return engine
    if engine == "replay" or (engine == "auto" and model is None and x is None):
        message, remedy = gradient_edit_replay_text(plan.backward_repr)
        raise InvalidArgumentError(
            message,
            code="replay_gradient_edit_unsupported",
            remedy=remedy,
            argument="hooks_or_site",
        )
    return trace._select_do_engine(engine, model=model, x=x)


def preflight_signal_plan(trace: Trace, plan: _SignalPlan, *, strict: bool) -> None:
    """Normalise and resolve both halves of a gradient plan without mutating.

    Normalisation runs every edit's mount check (an activation-only edit
    on a gradient site refuses ``edit_signal_mismatch`` there), and every
    site resolves on this trace, so a refusal anywhere leaves the trace
    exactly as it was.
    """

    from ..intervention.hooks import normalize_hook_plan
    from ..intervention.spec import entries_from_spec

    for half in (plan.forward, plan.backward):
        if half is None:
            continue
        site, edit = half
        if _is_public_recipe(site):
            entries = entries_from_spec(site, door="do")
        else:
            entries = normalize_hook_plan(site, edit, allow_replay_only_site_targets=True)
        for entry in entries:
            trace._validate_intervention_site(entry.site_target, strict=strict)


def arm_signal_plan(
    trace: Trace,
    plan: _SignalPlan,
    *,
    engine: str,
    strict: bool,
    model: nn.Module | None,
    x: Any,
    attached: list[str],
) -> tuple[tuple[str, ...], bool]:
    """Apply the activation half, run the rerun, then arm the backward half.

    ``attached`` collects every handle this transaction attaches, so the
    caller can detach them all when any step fails.

    Returns
    -------
    tuple[tuple[str, ...], bool]
        The backward-half handle ids and whether the rerun had to force
        backward readiness.
    """

    forced = False
    if plan.forward is not None:
        forward_site = plan.forward[0]
        if engine == "rerun":
            forward_site = lower_live_labels(trace, forward_site)
        _kind, handles = trace._apply_do_mutation(
            forward_site,
            plan.forward[1],
            engine=engine,
            strict=strict,
            confirm_mutation=True,
        )
        attached.extend(handles)
    if engine == "rerun":
        forced = force_backward_ready(trace)
        drop_inherited_backward(trace)
        trace.run(model, x, replay=ReplayOptions(strict=strict))
    assert plan.backward is not None
    site, edit = plan.backward
    if _is_public_recipe(site):
        handle = trace.attach_hooks(site, strict=strict, confirm_mutation=True)
    else:
        handle = trace.attach_hooks(site, edit, strict=strict, confirm_mutation=True)
    backward_handles = tuple(handle.handle_ids)
    attached.extend(backward_handles)
    return backward_handles, forced


def lower_live_labels(trace: Trace, site: Any) -> Any:
    """Readdress the finalized labels of a mixed plan's activation half for a live rerun.

    ``tl.grad(s) | s`` addresses one op in both passes. The gradient half
    arms on the finished rerun trace, where finalized labels exist; the
    activation half matches during the rerun's live forward, where they do
    not yet. Each exact finalized label (a bare string or ``tl.label``) is
    resolved on this trace, which still holds the source graph, and replaced
    by the raw capture labels of the ops it names, which a live forward
    matches. The rerun's own graph-divergence check covers a rerun whose
    graph no longer lines up with the source.

    Parameters
    ----------
    trace:
        Trace about to rerun (a fork of the source, not yet rerun).
    site:
        The activation half's site: a selector, a string, a ``Recipe`` or a
        list of ``(site, edit)`` pairs.

    Returns
    -------
    Any
        The site with every lowerable finalized label readdressed; the same
        object when nothing needed lowering.
    """

    if _is_public_recipe(site):
        from ..intervention.spec import Recipe, Rule

        rules = tuple(Rule(_lower_label_leaves(trace, r.target), r.edit) for r in site.rules)
        return Recipe(rules=rules)
    if isinstance(site, list):
        return [(_lower_label_leaves(trace, pair_site), edit) for pair_site, edit in site]
    return _lower_label_leaves(trace, site)


def _lower_label_leaves(trace: Trace, site: Any) -> Any:
    """Replace finalized exact-label leaves of one selector tree by raw labels."""

    from .._ir.selector_eval import looks_like_finalized_label
    from ..intervention.selectors import CompositeSelector, NotSelector, label
    from ..intervention.spec import _join_union, _selector_children

    kind = getattr(site, "selector_kind", None)
    if kind in ("and", "or"):
        children = tuple(_lower_label_leaves(trace, child) for child in _selector_children(site))
        return CompositeSelector(kind, children)
    if kind == "not":
        return NotSelector(_lower_label_leaves(trace, _selector_children(site)[0]))
    value = site if isinstance(site, str) else getattr(site, "selector_value", None)
    if (isinstance(site, str) or kind == "label") and isinstance(value, str):
        if looks_like_finalized_label(value):
            fanout = len(trace.layer_list) + 1
            ops = trace._resolve_sites(value, max_fanout=fanout)
            raw = [getattr(op, "_layer_label_raw", None) for op in ops]
            names = [name for name in raw if isinstance(name, str)]
            if names and len(names) == len(raw):
                return _join_union([label(name) for name in dict.fromkeys(names)])
    return site


def drop_inherited_backward(trace: Trace) -> None:
    """Clear the backward passes a rerun fork inherited from its source.

    The rerun result is a new execution: its backward passes, gradients and
    settle rows count only the backwards run on it, so a source that already
    ran ``backward`` must not leave pass 1 (and its ``.grad`` values) on the
    fork. The fork is reset before the rerun's forward, so the rerun capture
    then registers its own graph on it.

    Parameters
    ----------
    trace:
        The rerun fork, before its forward runs.
    """

    if not (trace.has_backward_pass or int(getattr(trace, "num_backward_passes", 0) or 0)):
        return
    from ..intervention.replay import _reset_backward_projection

    _reset_backward_projection(trace)


def force_backward_ready(trace: Trace) -> bool:
    """Make this trace's next rerun graph-connected; return whether it had to.

    A gradient edit needs the autograd graph, so a rerun that arms one is
    backward-ready whatever the source capture chose. The caller discloses
    a forced rerun as ``last_run["backward_ready_forced"] = True``.
    """

    if getattr(trace, "backward_ready", False):
        return False
    trace.backward_ready = True
    return True


def detach_handles(trace: Trace, handle_ids: list[str] | tuple[str, ...]) -> None:
    """Remove exactly the sticky hooks one transaction attached."""

    for handle_id in handle_ids:
        trace.clear_edits(handle=handle_id, confirm_mutation=True)


def stamp_arming_event(trace: Trace, handle_ids: tuple[str, ...], event_id: str) -> None:
    """Record the arming row's id on the gradient rules it armed.

    The settle rows written at each backward pass end name this id as
    their ``parent_event_id`` (registration identity), while each pass
    writes its own row (execution identity).
    """

    wanted = set(handle_ids)
    spec = trace._intervention_spec
    if spec is None:
        return
    for hook_spec in spec.hook_specs:
        if hook_spec.handle in wanted and _ARMING_GROUP_KEY in hook_spec.metadata:
            hook_spec.metadata[_ARMING_EVENT_KEY] = event_id
    trace._mark_intervention_spec_mutated(outs_stale=False)


def run_do_loss(
    trace: Trace,
    loss: Callable[[Trace], torch.Tensor],
    *,
    attached: tuple[str, ...],
    txn: _DoTransaction,
    source: Trace,
) -> None:
    """Run ``trace.backward(loss(trace))`` for ``do(..., loss=)``.

    A loss callable that raises (or a backward that refuses its value,
    such as a non-scalar or detached tensor) writes an ``error`` envelope,
    removes the hooks this ``do`` attached, and re-raises the original
    exception, the same cleanup a failed rerun gets. After a rerun the
    caller never receives the result, so the error row is also written on
    the ``source`` trace the caller holds.
    """

    try:
        trace.backward(loss(trace))
    except BaseException as exc:
        detach_handles(trace, attached)
        for target in (trace,) if source is trace else (trace, source):
            target._record_do_intervention_event(
                txn._replace(audit_rows_before=len(target.intervention_audit)),
                status="error",
                error=repr(exc),
                event_row=True,
            )
        raise


def live_backward_graph_available(trace: Trace) -> bool:
    """Whether a backward on this trace can reach its recorded forward.

    A loaded trace carries no live autograd graph, and a replay result's
    values were recomputed off the graph, so gradient rules attached there
    wait for the next rerun.
    """

    if bool(trace.__dict__.get("_loaded_from_bundle", False)):
        return False
    return bool(trace.state != TraceState.REPLAY_PROPAGATED)


def armed_edits(trace: Trace) -> tuple[Any, ...]:
    """The gradient rules in this trace's recipe, as ``Rule`` clauses.

    Read-only. Each entry is ``Rule(target=tl.grad(...), edit=...)`` with
    ``signal == "backward"``; the rules stay armed (sticky) until
    ``clear_edits(...)`` or their handle's ``remove()``.
    """

    from .._ir.selector_eval import normalize_selector_like
    from ..intervention.spec import Rule

    spec = trace._intervention_spec
    if spec is None:
        return ()
    rules: list[Any] = []
    for hook_spec in spec.hook_specs:
        if _arming_group(hook_spec) is None:
            continue
        edit = hook_spec.helper if hook_spec.helper is not None else hook_spec.hook
        target = normalize_selector_like(hook_spec.site_target, lifecycle="site")
        rules.append(Rule(target, edit))
    return tuple(rules)


def armed_summary_line(trace: Trace) -> str | None:
    """Summary text while any armed gradient rule has not fired here.

    Returns
    -------
    str | None
        ``"N gradient edits armed; call trace.backward(loss)"`` or ``None``.
    """

    if bool(trace.__dict__.get("_loaded_from_bundle", False)):
        annotations = getattr(trace, "annotations", None) or {}
        persisted = annotations.get("armed_gradient_edits")
        if isinstance(persisted, int) and persisted > 0:
            noun = "edit" if persisted == 1 else "edits"
            return f"{persisted} gradient {noun} armed; they cannot run until a rerun"
        return None
    pending = unfired_gradient_rule_count(trace)
    if not pending:
        return None
    noun = "edit" if pending == 1 else "edits"
    return f"{pending} gradient {noun} armed; call trace.backward(loss)"


def unfired_gradient_rule_count(trace: Trace) -> int | None:
    """Count the gradient rules in the recipe that never fired on this trace.

    Returns
    -------
    int | None
        The count, or ``None`` when the recipe holds no gradient rule.
    """

    spec = trace._intervention_spec
    hook_specs = [
        hook_spec
        for hook_spec in (spec.hook_specs if spec is not None else ())
        if _arming_group(hook_spec) is not None
    ]
    if not hook_specs:
        return None
    fired_groups = {
        str(row.get("arming_group"))
        for row in trace.state_history
        if isinstance(row, dict)
        and row.get("op") == "intervention_event"
        and row.get("door") == "backward"
        and row.get("status") == "fired"
    }
    return sum(1 for hook_spec in hook_specs if _arming_group(hook_spec) not in fired_groups)


def refresh_armed_annotation(trace: Trace) -> None:
    """Keep ``annotations["armed_gradient_edits"]`` equal to the unfired count.

    The annotation persists with the trace, so a saved armed result says
    how many gradient edits never ran (the recipe itself travels through
    ``save_intervention``).
    """

    count = unfired_gradient_rule_count(trace)
    annotations = getattr(trace, "annotations", None)
    if count is None:
        if isinstance(annotations, dict):
            annotations.pop("armed_gradient_edits", None)
        return
    if isinstance(annotations, dict):
        annotations["armed_gradient_edits"] = count


def settle_armed_gradient_edits(trace: Any, *, pass_index: int) -> None:
    """Write the settle rows of one finished backward pass.

    Called once at backward pass end. For every arming transaction whose
    gradient rules sit in ``trace``'s recipe, one ``intervention_event_v2``
    envelope records what the pass did: ``door="backward"``,
    ``lane="live_hook"``, ``parent_event_id`` = the arming row (registration
    identity), ``status`` ``"fired"`` or ``"no_fire"`` with this pass's
    ``fire_count``, and ``extra`` ``arming_event_id`` / ``backward_pass_index``
    plus a ``no_fire_reason`` from ``{"gradient_not_reached",
    "pass_filtered"}``. Each pass writes its own row, so a second backward
    never rewrites the first pass's row.

    Parameters
    ----------
    trace:
        Trace whose backward pass just ended.
    pass_index:
        One-based index of that backward pass.
    """

    spec = trace._intervention_spec
    if spec is None:
        return
    groups: dict[str, list[Any]] = {}
    for hook_spec in getattr(spec, "hook_specs", ()):
        group = _arming_group(hook_spec)
        if group is not None:
            groups.setdefault(str(group), []).append(hook_spec)
    if not groups:
        return
    pass_records = [
        record
        for record in getattr(spec, "records", ())
        if getattr(record, "signal", None) == "backward"
        and getattr(record, "backward_pass_index", None) == pass_index
    ]
    for hook_specs in groups.values():
        _settle_arming_group(trace, hook_specs, pass_records, pass_index=pass_index)
    refresh_armed_annotation(trace)


def _settle_arming_group(
    trace: Any,
    hook_specs: list[Any],
    pass_records: list[Any],
    *,
    pass_index: int,
) -> None:
    """Write one settle row for the gradient rules of one arming transaction."""

    from ..intervention.audit import record_intervention_event, site_keys_for_labels

    fires = [record for record in pass_records if _record_belongs_to(record, hook_specs)]
    arming_event_id = _arming_event_id(trace, hook_specs[0])
    extra: dict[str, Any] = {
        "arming_event_id": arming_event_id,
        "arming_group": _arming_group(hook_specs[0]),
        "backward_pass_index": pass_index,
    }
    if arming_event_id is not None:
        extra["parent_event_id"] = arming_event_id
    if not fires:
        extra["no_fire_reason"] = (
            "pass_filtered"
            if all(_pass_filtered(hook_spec.site_target, pass_index) for hook_spec in hook_specs)
            else "gradient_not_reached"
        )
    labels = tuple(dict.fromkeys(str(record.site_label or record.target_label) for record in fires))
    record_intervention_event(
        trace,
        lane="live_hook",
        door="backward",
        edit_names=tuple(_hook_spec_edit_name(hook_spec) for hook_spec in hook_specs),
        selection_repr=" | ".join(repr(hook_spec.site_target) for hook_spec in hook_specs),
        status="fired" if fires else "no_fire",
        fire_count=len(fires),
        site_keys=site_keys_for_labels(trace, labels),
        extra=extra,
        append_audit_row=False,
        audit_event_row=True,
    )


def _record_belongs_to(record: Any, hook_specs: list[Any]) -> bool:
    """Whether one backward FireRecord came from one of these hook specs.

    Built-in edits are matched by object identity (the stored helper is the
    one the fire record carries); callables by their display name.
    """

    helper = getattr(record, "helper", None)
    for hook_spec in hook_specs:
        if hook_spec.helper is not None:
            if helper is hook_spec.helper:
                return True
        elif helper is None and getattr(record, "helper_name", None) == getattr(
            hook_spec.hook, "__name__", None
        ):
            return True
    return False


def _hook_spec_edit_name(hook_spec: Any) -> str:
    """Display name of one hook spec's edit for the settle row."""

    if hook_spec.helper is not None:
        return str(getattr(hook_spec.helper, "helper_name", hook_spec.helper))
    return str(getattr(hook_spec.hook, "__name__", type(hook_spec.hook).__name__))


def _pass_filtered(target: Any, pass_index: int) -> bool:
    """Whether a target's backward-pass qualifiers exclude ``pass_index``.

    Walks the frozen target for ``backward_pass`` qualifiers; a target with
    none never filters a pass.
    """

    stack: list[Any] = [target]
    qualifiers: list[Any] = []
    while stack:
        node = stack.pop()
        if isinstance(node, (tuple, list)):
            stack.extend(node)
            continue
        kind = getattr(node, "selector_kind", None)
        if kind is None:
            continue
        value = getattr(node, "selector_value", None)
        if kind == "backward_pass":
            qualifiers.append(value)
        elif value is not None and not isinstance(value, (str, int, float)):
            stack.append(value)
    return bool(qualifiers) and all(_qualifier_pass(value) != pass_index for value in qualifiers)


def _qualifier_pass(value: Any) -> int | None:
    """Pass index carried by a ``backward_pass`` qualifier value."""

    if isinstance(value, int):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], int):
        return int(value[0])
    pass_index = getattr(value, "pass_index", None)
    return int(pass_index) if isinstance(pass_index, int) else None
