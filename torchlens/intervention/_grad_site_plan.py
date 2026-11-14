"""Gradient-site plans for ``Trace.do``: split, preflight, arm and settle.

``Trace.do`` routes a plan whose site addresses the backward pass
(``tl.grad(site)`` or an autograd-node selector) here. The plan is split into
its activation half and its backward half, preflighted whole, and armed: a
rerun arms the backward half on a fork (the result), ``engine="set_only"`` arms
it in place. Each backward pass end writes one settle row per arming
transaction (:func:`settle_armed_gradient_edits`, called from the torch
backward door). ``torchlens._data_classes._trace_grad_sites`` keeps the
read side and reaches this module lazily.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .._capture._episode_coupling import quarantine_episode_after_perturbed_replay
from .._data_classes._trace_grad_sites import (
    _ARMING_EVENT_KEY,
    _ARMING_GROUP_KEY,
    _arming_event_id,
    _arming_group,
    _pass_filtered,
    _SignalDoRequest,
    refresh_armed_annotation,
    select_signal_plan_engine,
)
from .._data_classes._trace_intervention_records import _DoTransaction
from .._data_classes._trace_missing_attrs import is_live_trace
from .._ir.selector_eval import looks_like_finalized_label, normalize_selector_like
from .._selection.resolver import _resolve_unchecked
from .._selection.selectors import CompositeSelector, NotSelector, label
from ..backends._backend_names import TORCH_BACKEND_NAME
from ..options import ReplayOptions
from ._audit import record_intervention_event, site_keys_for_labels
from ._edit_guards import detach_handles, internal_mutation
from ._grad_site_neutral import do_signal_plan_off_torch
from ._replay import _reset_backward_projection, gradient_edit_replay_text
from ._rerun_predicate import raw_live_labels
from ._runtime import apply_live_gradient_hooks
from ._spec import (
    Recipe,
    Rule,
    _join_union,
    _selector_children,
    entries_from_spec,
    refuse_invalid_loss,
    site_signal,
    split_site_by_signal,
)
from ._trace_run import rerun_in_place
from .hooks import gradient_inners, normalize_hook_plan, split_site_branches

if TYPE_CHECKING:
    import torch

    from .._data_classes.trace import Trace

__all__ = [
    "apply_gradient_site_edits",
    "armed_edits",
    "do_signal_plan",
    "gradient_rule_sites",
    "gradient_edit_replay_text",
    "is_gradient_entry",
    "refuse_invalid_do_loss",
    "settle_armed_gradient_edits",
    "signal_plan",
]


@dataclass(frozen=True)
class _SignalPlan:
    """A ``do()`` plan split into its activation half and its backward half.

    Each half is a ``(site, edit)`` door input (a list of pairs with ``edit``
    ``None``, or a ``Recipe`` with ``edit`` ``None``), or ``None`` when the
    plan has no site in that pass.
    """

    site: Any
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

    return isinstance(value, Recipe)


def _door_pairs(site: Any, value_or_hook: Any) -> list[tuple[Any, Any]] | None:
    """Lower the ``do()`` call shapes that can carry a gradient site to pairs.

    Returns ``None`` for shapes that never address a gradient site
    (selection batches, regions, slices), which keep their own doors.
    """

    if _is_public_recipe(site):
        return [(rule.target, rule.edit) for rule in site.rules]
    if value_or_hook is None and isinstance(site, Mapping):
        return list(site.items())
    if value_or_hook is None and isinstance(site, (list, tuple)):
        if all(isinstance(pair, tuple) and len(pair) == 2 for pair in site):
            return list(site)
        return None
    return [(site, value_or_hook)]


def signal_plan(site: Any, value_or_hook: Any) -> _SignalPlan | None:
    """Split a ``do()`` input into its activation and backward halves.

    Returns ``None`` when no site addresses the backward pass, so plain
    activation edits keep the unchanged ``do`` path.
    """

    pairs = _door_pairs(site, value_or_hook)
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
        if _is_public_recipe(site):
            return Recipe(rules=tuple(Rule(site, edit) for site, edit in half)), None
        if len(half) == 1 and value_or_hook is not None:
            return half[0]
        return list(half), None

    return _SignalPlan(
        site=site,
        value_or_hook=value_or_hook,
        forward=_half(forward),
        backward=_half(backward),
    )


def refuse_invalid_do_loss(loss: Any, plan: _SignalPlan | None) -> None:
    """Refuse a ``loss=`` argument to ``do`` before any mutation."""

    refuse_invalid_loss(loss, has_gradient_site=plan is not None, door="do")


def is_gradient_entry(entry: Any) -> bool:
    """Whether a normalized hook entry's site addresses the backward pass."""

    return site_signal(getattr(entry, "site_target", None)) == "backward"


def do_signal_plan(trace: Trace, plan: _SignalPlan, request: _SignalDoRequest) -> Trace:
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
    request:
        The rerun's model and input, the merged ``InterventionOptions`` and
        the optional loss callable of the returned trace (which runs the
        backward here).

    A non-torch trace goes to its backend's door through
    ``_grad_site_neutral.do_signal_plan_off_torch`` before any torch-only step.

    Returns
    -------
    Trace
        The armed trace: a new trace after a rerun, this trace under
        ``set_only``.
    """

    backend = getattr(trace, "_provenance_backend", None) or TORCH_BACKEND_NAME
    if backend != TORCH_BACKEND_NAME:
        return cast("Trace", do_signal_plan_off_torch(trace, plan, request, backend))
    preflight_txn = _preflight_signal_do(trace, plan, request)
    if preflight_txn.engine == "rerun":
        result = trace._fork_trace(name=None)
    else:
        trace._warn_if_root_mutation()
        result = trace
    txn = preflight_txn._replace(audit_rows_before=len(result.intervention_audit))
    backward_handles, forced, attached = _arm_or_record_error(trace, result, plan, request, txn)
    _record_armed_signal_plan(result, plan, request, txn, (backward_handles, forced))
    if request.loss is not None:
        run_do_loss(result, request.loss, attached=tuple(attached), txn=txn, source=trace)
    return result


def _preflight_signal_do(
    trace: Trace, plan: _SignalPlan, request: _SignalDoRequest
) -> _DoTransaction:
    """Pick the engine and preflight the whole plan; an error row records a refusal."""

    model, x, intervention_options = (request.model, request.x, request.intervention_options)
    engine = select_signal_plan_engine(
        trace, intervention_options.engine, model=model, x=x, plan=plan
    )
    strict = intervention_options.strict
    preflight_txn = _DoTransaction(
        site=plan.site,
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
    return preflight_txn


def _arm_or_record_error(
    trace: Trace,
    result: Trace,
    plan: _SignalPlan,
    request: _SignalDoRequest,
    txn: _DoTransaction,
) -> tuple[tuple[str, ...], bool, list[str]]:
    """Arm the plan on ``result``; a failure detaches it and writes the error row on ``trace``."""

    attached: list[str] = []
    try:
        backward_handles, forced = arm_signal_plan(
            result, plan, request, engine=txn.engine, attached=attached
        )
    except BaseException as exc:
        detach_handles(result, attached)
        trace._record_do_intervention_event(
            txn._replace(audit_rows_before=len(trace.intervention_audit)),
            status="error",
            error=repr(exc),
            event_row=True,
        )
        raise
    return backward_handles, forced, attached


def _record_armed_signal_plan(
    result: Any,
    plan: _SignalPlan,
    request: _SignalDoRequest,
    txn: _DoTransaction,
    armed: tuple[tuple[str, ...], bool],
) -> None:
    """Write the armed envelope and the ``do`` operation row of one armed plan."""

    backward_handles, forced = armed
    engine, intervention_options = txn.engine, request.intervention_options
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
        model_supplied=request.model is not None,
        x_supplied=request.x is not None,
        strict=intervention_options.strict,
        armed_edits=len(backward_handles),
    )
    if plan.forward is not None and engine == "rerun":
        quarantine_episode_after_perturbed_replay(result)


def preflight_signal_plan(trace: Trace, plan: _SignalPlan, *, strict: bool) -> None:
    """Normalise and resolve both halves of a gradient plan without mutating.

    Normalisation runs every edit's mount check (an activation-only edit
    on a gradient site refuses ``edit_signal_mismatch`` there), and every
    site resolves on this trace, so a refusal anywhere leaves the trace
    exactly as it was.
    """

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
    request: _SignalDoRequest,
    *,
    engine: str,
    attached: list[str],
) -> tuple[tuple[str, ...], bool]:
    """Apply the activation half, run the rerun, then arm the backward half.

    ``attached`` collects every handle this transaction attaches, so the
    caller can detach them all when any step fails. ``request`` carries the
    rerun's model and input and the merged options (``strict``).

    Returns
    -------
    tuple[tuple[str, ...], bool]
        The backward-half handle ids and whether the rerun had to force
        backward readiness.
    """

    model, x, strict = request.model, request.x, request.intervention_options.strict
    forced = False
    if plan.forward is not None:
        forward_site = plan.forward[0]
        if engine == "rerun":
            forward_site = lower_live_labels(trace, forward_site)
        with internal_mutation():
            _kind, handles = trace._apply_do_mutation(
                forward_site, plan.forward[1], engine=engine, strict=strict
            )
        attached.extend(handles)
    if engine == "rerun":
        forced = force_backward_ready(trace)
        drop_inherited_backward(trace)
        rerun_in_place(trace, model, x, replay=ReplayOptions(strict=strict))
    assert plan.backward is not None
    site, edit = plan.backward
    with internal_mutation():
        if _is_public_recipe(site):
            backward_handles = trace._attach_hooks(site, strict=strict)
        else:
            backward_handles = trace._attach_hooks(site, edit, strict=strict)
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
        rules = tuple(Rule(_lower_label_leaves(trace, r.target), r.edit) for r in site.rules)
        return Recipe(rules=rules)
    if isinstance(site, list):
        return [(_lower_label_leaves(trace, pair_site), edit) for pair_site, edit in site]
    return _lower_label_leaves(trace, site)


def _lower_label_leaves(trace: Trace, site: Any) -> Any:
    """Replace finalized exact-label leaves of one selector tree by raw labels."""

    kind = getattr(site, "selector_kind", None)
    if kind in ("and", "or"):
        children = tuple(_lower_label_leaves(trace, child) for child in _selector_children(site))
        return CompositeSelector(kind, children)
    if kind == "not":
        return NotSelector(_lower_label_leaves(trace, _selector_children(site)[0]))
    value = site if isinstance(site, str) else getattr(site, "selector_value", None)
    if (
        (isinstance(site, str) or kind == "label")
        and isinstance(value, str)
        and looks_like_finalized_label(value)
    ):
        names = raw_live_labels(trace, value)
        if names:
            return _join_union([label(name) for name in names])
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

    if not (trace._autograd_has_backward_pass or int(trace._num_backward_passes or 0)):
        return
    _reset_backward_projection(trace)


def force_backward_ready(trace: Trace) -> bool:
    """Make this trace's next rerun graph-connected; return whether it had to.

    A gradient edit needs the autograd graph, so a rerun that arms one is
    backward-ready whatever the source capture chose. The caller discloses
    a forced rerun as ``last_run["backward_ready_forced"] = True``.
    """

    if is_live_trace(trace) and trace._options_backward_ready:
        return False
    trace._options_backward_ready = True
    return True


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


def gradient_rule_sites(trace: Any) -> list[Any]:
    """Return the sites a capture-time ``tl.grad(site)`` rule addresses.

    Parameters
    ----------
    trace:
        Trace finishing its capture, carrying the capture-time intervention spec.

    Returns
    -------
    list[Any]
        Resolved sites of every op whose output gradient a gradient rule edits.
    """

    spec = trace._intervention_spec
    ops = tuple(trace._layer_list)
    sites: list[Any] = []
    for hook_spec in getattr(spec, "hook_specs", ()) or ():
        for universe, branch in split_site_branches(hook_spec.site_target):
            if universe != "gradient":
                continue
            for inner in gradient_inners(branch):
                sites.extend(_resolve_unchecked(ops, inner, strict=False))
    return sites


def apply_gradient_site_edits(trace: Trace, grad: torch.Tensor, tensor_label: str) -> torch.Tensor:
    """Run the ``tl.grad(site)`` edits that address this op output's gradient.

    Parameters
    ----------
    trace:
        Trace the gradient is recorded on (the backwarded trace).
    grad:
        Gradient arriving at the op output.
    tensor_label:
        Raw label of the op output.

    Returns
    -------
    torch.Tensor
        The gradient after every matching edit (``grad`` when none fired).
    """

    final_label = trace._raw_to_final_layer_labels.get(tensor_label)
    layer_dict = trace._layer_dict_all_keys if is_live_trace(trace) else {}
    op = layer_dict.get(final_label) if final_label is not None else None
    if op is None:
        return grad
    edited, _records = apply_live_gradient_hooks(grad, trace=trace, op=op)
    return edited


def armed_edits(trace: Trace) -> tuple[Any, ...]:
    """The gradient rules in this trace's recipe, as ``Rule`` clauses.

    Read-only. Each entry is ``Rule(target=tl.grad(...), edit=...)`` with
    ``signal == "backward"``; the rules stay armed (sticky) until
    ``clear_edits(...)`` or their handle's ``remove()``.
    """

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


def settle_armed_gradient_edits(trace: Any, *, pass_index: int) -> None:
    """Write the settle rows of one finished backward pass.

    Called once at backward pass end. For every arming transaction whose
    gradient rules sit in ``trace``'s recipe, one ``intervention_event_v2``
    envelope records what the pass did: ``door="backward"``,
    ``lane="live_hook"``, ``parent_event_id`` = the arming row (registration
    identity), ``status`` ``"fired"`` or ``"no_fire"`` with this pass's
    ``fire_count``, and ``extra`` ``arming_event_id`` / ``backward_pass_number``
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
        and getattr(record, "backward_pass_number", None) == pass_index
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

    fires = [record for record in pass_records if _record_belongs_to(record, hook_specs)]
    arming_event_id = _arming_event_id(trace, hook_specs[0])
    extra: dict[str, Any] = {
        "arming_event_id": arming_event_id,
        "arming_group": _arming_group(hook_specs[0]),
        "backward_pass_number": pass_index,
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

    helper = getattr(record, "edit", None)
    for hook_spec in hook_specs:
        if hook_spec.helper is not None:
            if helper is hook_spec.helper:
                return True
        elif helper is None and getattr(record, "edit_name", None) == getattr(
            hook_spec.hook, "__name__", None
        ):
            return True
    return False


def _hook_spec_edit_name(hook_spec: Any) -> str:
    """Display name of one hook spec's edit for the settle row."""

    if hook_spec.helper is not None:
        return str(getattr(hook_spec.helper, "name", hook_spec.helper))
    return str(getattr(hook_spec.hook, "__name__", type(hook_spec.hook).__name__))
