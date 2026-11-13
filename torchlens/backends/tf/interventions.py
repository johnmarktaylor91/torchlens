"""TensorFlow static-label intervention layer.

TensorFlow op callbacks are read-only in eager execution (spike-verified on
TF 2.21: returned replacement outputs are ignored), so interventions run
through a two-level WRITABLE layer on top of the untouched op-callback
capture spine:

1. Module-boundary sites (``tl.module`` / ``tl.in_module`` conditions)
   substitute Keras/`tf.Module` call outputs through the existing module
   ``__call__`` patch.
2. Op-level sites (``tl.func`` / ``tl.label`` / ``tl.label_contains`` conditions,
   and value-dependent callable conditions) substitute returns of the python
   op wrappers every eager op returns through (``_wrap_registry``).

Op-level sites are evaluated ONCE per op, in the op callback that records it,
on the same ``RecordContext`` torch predicates see (real scalar-bool values
included). A site that fires leaves a pending decision on the op's label; the
wrap layer applies it when the op's output reaches Python.

Capture truth stays with op callbacks: replacement values are produced by
real ops that the callback records, so downstream records consume the
substituted values by construction. Site resolution FAILS CLOSED: an op
matched by a site's selector in the callback stream whose call never passed
through the wrap layer raises a typed unreachable error instead of silently
not intervening.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, cast

from ..._ir.intervention import FireResult
from ..._ir.op_record import amend_module_exit_intervention
from ..._ir.selector_eval import first_selector_kind_outside, selector_contains_kind
from ..._postprocess._selective_save import (
    _STATIC_INTERVENTION_SELECTOR_KINDS,
    reject_selector_outside_kinds,
)
from ..._selection.selectors import BaseSelector
from ...intervention._types import Edit, InterventionDecision
from ...recording._exceptions import PredicateError
from ..registry import BackendUnsupportedError
from ._grad_sites import gradient_replay_intervene, stage_tf_grad_clauses
from ._halt import CallbackAbort, TFHaltRuntime, apply_halt_outcome, chain_output_hooks
from ._intervention_hooks import (
    _producer_label,
    _selector_kinds,
    _tf_callable_hook,
    _tf_helper_hook,
    _TFModuleExitContext,
)
from ._intervention_plan import (
    TFInterventionPlan,
    TFInterventionSite,
    TFInterventionSiteUnreachableError,
)
from ._source_context import TFSourceContext
from ._tensor_leaves import _is_tf_tensor_like, _map_tensor_leaves
from ._wrap_registry import _CURATED_WRAP_ENTRIES, iter_wrap_targets

#: Refusal code for gradient-universe sites (registered in `_protocols.tf`).
_TF_REFUSAL_BACKWARD = "backward_capture_unsupported"

# Alias of the neutral authority table (the ``output`` drop is declared
# there) -- never re-spell the kinds here.
_TF_INTERVENTION_SELECTOR_KINDS = _STATIC_INTERVENTION_SELECTOR_KINDS
_MODULE_ONLY_KINDS = frozenset({"module", "in_module", "and", "or", "not"})
#: Kinds a value-dependent rule may use: the static op kinds plus ``tl.where``.
_TF_VALUE_SELECTOR_KINDS = _TF_INTERVENTION_SELECTOR_KINDS | frozenset({"predicate"})


def normalize_tf_interventions(intervene: Any, tf: Any) -> TFInterventionPlan:
    """Normalize the public ``intervene=`` value into a TensorFlow plan.

    Parameters
    ----------
    intervene
        Public ``intervene=`` value: one ``tl.when`` predicate or a sequence.
    tf
        Imported TensorFlow module.

    Returns
    -------
    TFInterventionPlan
        Normalized plan with sites split by dispatch level.
    """

    predicates = list(intervene) if isinstance(intervene, (list, tuple)) else [intervene]
    sites: list[TFInterventionSite] = []
    for index, predicate in enumerate(predicates):
        plan_id = f"tf_intervene_{index}"
        selector = getattr(predicate, "selector", None)
        decision = getattr(predicate, "decision", None)
        if (
            isinstance(selector, BaseSelector)
            and isinstance(decision, InterventionDecision)
            and not selector_contains_kind(selector, "predicate", unwrap=True)
        ):
            sites.append(_static_site(plan_id, predicate, selector, decision, tf))
        else:
            sites.append(_value_site(plan_id, predicate))
    normalized = tuple(sites)
    return TFInterventionPlan(
        sites=normalized,
        op_sites=tuple(site for site in normalized if site.level == "op"),
        module_sites=tuple(site for site in normalized if site.level == "module"),
    )


def _static_site(
    plan_id: str,
    predicate: Any,
    selector: BaseSelector,
    decision: InterventionDecision,
    tf: Any,
) -> TFInterventionSite:
    """Normalize one static-selector ``tl.when`` clause into a site."""

    _reject_gradient_site(selector)
    reject_selector_outside_kinds(
        selector,
        allowed=_TF_INTERVENTION_SELECTOR_KINDS,
        backend_name="tf",
    )
    level: Literal["module", "op"] = (
        "module" if _selector_kinds(selector) <= _MODULE_ONLY_KINDS else "op"
    )
    return TFInterventionSite(
        plan_id=plan_id,
        predicate=predicate,
        selector=selector,
        decision=decision,
        hook=_resolve_tf_hook(decision, tf),
        level=level,
    )


def _value_site(plan_id: str, predicate: Any) -> TFInterventionSite:
    """Normalize a value-dependent (callable or multi-clause) predicate into an op site.

    The predicate runs on each op's ``RecordContext`` in the op callback (paddle and
    torch semantics); its returned decision resolves into a hook per firing. Clauses
    that only a module boundary can answer refuse typed, since op contexts would
    silently never match them.
    """

    if not callable(predicate):
        raise BackendUnsupportedError(
            "tf backend trace(intervene=...) requires tl.when(condition, action) or a "
            "callable predicate returning an intervention decision; got "
            f"{type(predicate).__name__!s}.",
            code="trace_option_unsupported",
            option="intervene",
        )
    for target in (getattr(rule, "target", None) for rule in getattr(predicate, "rules", ())):
        if not isinstance(target, BaseSelector):
            continue
        _reject_gradient_site(target)
        outside = first_selector_kind_outside(target, allowed=_TF_VALUE_SELECTOR_KINDS)
        if outside is not None or _selector_kinds(target) <= _MODULE_ONLY_KINDS:
            raise BackendUnsupportedError(
                f"tf backend evaluates value-dependent and multi-clause intervene= specs "
                f"on op contexts; the clause {target!r} needs a module boundary or a "
                "selector kind tf does not evaluate. Pass module clauses as their own "
                "tl.when entry in the intervene= list.",
                code="trace_option_unsupported",
                option="intervene",
            )
    return TFInterventionSite(plan_id, predicate, None, None, None, "op")


def _reject_gradient_site(selector: BaseSelector) -> None:
    """Refuse a site mixing a gradient or autograd-node address into an activation clause."""

    if _site_universes(selector) != {"activation"}:
        raise BackendUnsupportedError(
            "the tf backend takes a gradient address only as a whole tl.when(tl.grad(...), "
            f"edit) clause; the site {selector!r} mixes it with activation addresses or "
            "names autograd nodes TensorFlow does not record. Split it into separate clauses.",
            code=_TF_REFUSAL_BACKWARD,
        )


def _site_universes(selector: Any) -> set[str]:
    """The site universes (activation, gradient, node) a clause's selector addresses."""

    from ...intervention.hooks import split_site_branches

    if not isinstance(selector, BaseSelector):
        return {"activation"}
    return {universe for universe, _branch in split_site_branches(selector)}


def split_tf_grad_clauses(intervene: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split whole ``tl.grad`` clauses off ``intervene``: ``(the rest or None, grad clauses)``."""

    clauses = list(intervene) if isinstance(intervene, (list, tuple)) else [intervene]
    grad = [c for c in clauses if _site_universes(getattr(c, "selector", None)) == {"gradient"}]
    rest = [c for c in clauses if all(c is not g for g in grad)]
    return (rest or None) if grad else intervene, tuple(grad)


def tf_gradient_replay_intervene(trace: Any, tf: Any, intervene: Any) -> Any:
    """``intervene`` for the derived-gradient replay, its ``tl.grad`` clauses as tape edits."""

    from ...intervention import when  # deferred: the intervention package is in the import cycle

    return gradient_replay_intervene(trace, tf, *split_tf_grad_clauses(intervene), when)


def _resolve_tf_hook(decision: InterventionDecision, tf: Any) -> Callable[[Any], Any]:
    """Resolve a decision's hook into a TensorFlow replacement callable.

    Returns a callable mapping the original output tensor to its replacement.
    """

    hook = decision.hook
    if isinstance(hook, Edit):
        return _tf_helper_hook(hook, tf)
    if callable(hook):
        return _tf_callable_hook(hook)
    raise BackendUnsupportedError(
        f"tf backend interventions require a helper action or callable; got {type(hook).__name__}."
    )


@contextmanager
def tf_intervention_wrap(tf: Any, plan: TFInterventionPlan, session: Any) -> Iterator[None]:
    """Install the curated functional wrap layer for op-level sites.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    plan
        Normalized intervention plan.
    session
        Active ``TFEagerCaptureSession`` recording the forward.

    Yields
    ------
    None
        Control while the curated entry points are wrapped.
    """

    if not plan.op_sites:
        yield
        return
    originals: list[tuple[Any, str, Any]] = []
    wrapper_by_original: dict[int, Any] = {}

    def _restore_installed() -> None:
        """Restore every curated entry-point patch that actually landed."""

        for owner, attr_name, original in reversed(originals):
            setattr(owner, attr_name, original)

    # R07 (the L4 unwind standard): the install loop mutates process-global
    # ``tf``/``tf.nn``/``tf.math`` attributes BEFORE the try that owns the
    # yield; a BaseException escaping it used to strand every wrapper
    # installed so far.
    try:
        for owner, attr_name, original in iter_wrap_targets(tf, _CURATED_WRAP_ENTRIES):
            wrapper = wrapper_by_original.get(id(original))
            if wrapper is None:
                wrapper = _wrap_entry_point(original, plan, session, tf)
                wrapper_by_original[id(original)] = wrapper
            originals.append((owner, attr_name, original))
            setattr(owner, attr_name, wrapper)
    except BaseException:
        _restore_installed()
        raise
    try:
        yield
    finally:
        _restore_installed()


def _wrap_entry_point(
    original: Callable[..., Any],
    plan: TFInterventionPlan,
    session: Any,
    tf: Any,
) -> Callable[..., Any]:
    """Build the substituting wrapper for one curated entry point.

    Parameters
    ----------
    original
        Unwrapped TensorFlow callable.
    plan
        Normalized intervention plan.
    session
        Active capture session.
    tf
        Imported TensorFlow module.

    Returns
    -------
    Callable[..., Any]
        Wrapper substituting matched op-level site outputs.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Run the original entry point and consult op-level sites."""

        out = original(*args, **kwargs)
        if plan._in_hook:
            return out
        return _map_tensor_leaves(
            out,
            tf,
            lambda tensor: _consider_op_substitution(plan, session, tf, tensor),
        )

    return wrapper


def _consider_op_substitution(
    plan: TFInterventionPlan,
    session: Any,
    tf: Any,
    tensor: Any,
) -> Any:
    """Apply the op-level decisions pending on one wrapped-call output tensor.

    Parameters
    ----------
    plan
        Normalized intervention plan.
    session
        Active capture session.
    tf
        Imported TensorFlow module.
    tensor
        Output tensor produced by the wrapped call.

    Returns
    -------
    Any
        Original or replacement tensor.
    """

    label = _producer_label(session, tensor)
    if label is None or label in plan.hook_generated_labels:
        return tensor
    if label not in session.events.op_event_by_label_raw:
        return tensor
    plan.presented_labels.add(label)
    current = tensor
    # Popped, so an output passing through nested wrappers fires once.
    for site in plan.pending_decisions.pop(label, ()):
        current = _fire_site(plan, session, tf, site, current, label)
    if plan.on_presented is not None:
        plan.on_presented(label, current)
    return current


def observe_tf_op_sites(plan: TFInterventionPlan, tf: Any, recorded: Sequence[Any]) -> None:
    """Evaluate every op-level site once per recorded op output, in its op callback.

    Parameters
    ----------
    plan
        Normalized intervention plan.
    tf
        Imported TensorFlow module.
    recorded
        ``(label_raw, context, tensor)`` per output the callback just recorded.
    """

    if plan._in_hook:
        return
    for label, context, _tensor in recorded:
        due = [site for site in (_site_due(item, context, tf) for item in plan.op_sites) if site]
        if due:
            plan.pending_decisions[label] = due


def _site_due(site: TFInterventionSite, context: Any, tf: Any) -> TFInterventionSite | None:
    """Return the site to fire on one op context (value sites resolved), or ``None``."""

    result = site.predicate(context)
    if site.decision is not None:
        return site if result is not None else None
    # deferred: intervention sits above the backends in the layer map
    from ...intervention._predicates import as_intervention_decision

    try:
        decision = as_intervention_decision(result)
    except TypeError as exc:
        raise PredicateError(
            "intervene predicate must return InterventionDecision, Edit, callable, or "
            "None. Remedy: return one of those values from the intervene predicate.",
            ctx=context,
            result=result,
            code="predicate_return_invalid",
        ) from exc
    if decision is None or decision.hook is None:
        return None
    return replace(site, decision=decision, hook=_resolve_tf_hook(decision, tf))


def apply_tf_module_intervention(
    plan: TFInterventionPlan,
    session: Any,
    tf: Any,
    trace: Any,
    frame: Any,
    module_type: str,
    output: Any,
    module_stack: Sequence[Any],
) -> Any:
    """Evaluate module-boundary sites against one module-call exit.

    Parameters
    ----------
    plan
        Normalized intervention plan.
    session
        Active capture session.
    tf
        Imported TensorFlow module.
    trace
        Owning trace exposed as the selector subject's source trace.
    frame
        Exiting ``ModuleFrame``.
    module_type
        Module class name.
    output
        Module-call return value.
    module_stack
        Active module frames including ``frame``.

    Returns
    -------
    Any
        Original or substituted module output.
    """

    if not plan.module_sites:
        return output
    context = _TFModuleExitContext(
        module=f"{frame.address}:{frame.call_index}",
        address=frame.address,
        module_type=module_type,
        module_stack=tuple(
            {
                "address": item.address,
                "module_type": item.module_type,
                "pass_number": item.call_index,
            }
            for item in module_stack
        ),
        source_trace=trace,
    )
    current = output
    for site in plan.module_sites:
        decision = site.predicate(context)
        if decision is None:
            continue

        def _fire_for_site(tensor: Any, site: TFInterventionSite = site) -> Any:
            """Fire the definition-time-bound site on one matched tensor leaf."""

            return _fire_site(plan, session, tf, site, tensor, _producer_label(session, tensor))

        current = _map_tensor_leaves(current, tf, _fire_for_site)
    return current


def _fire_site(
    plan: TFInterventionPlan,
    session: Any,
    tf: Any,
    site: TFInterventionSite,
    tensor: Any,
    label: str | None,
) -> Any:
    """Run one site's hook on a matched tensor and record the fire.

    Parameters
    ----------
    plan
        Normalized intervention plan.
    session
        Active capture session.
    tf
        Imported TensorFlow module.
    site
        Fired site.
    tensor
        Original tensor value.
    label
        Raw label of the tensor's producing op, when resolvable.

    Returns
    -------
    Any
        Replacement tensor.
    """

    # Static sites carry their hook; a value site fires through its per-op resolved copy.
    hook = cast(Callable[[Any], Any], site.hook)
    events_before = len(session.events.op_events)
    plan._in_hook = True
    try:
        replacement = hook(tensor)
    finally:
        plan._in_hook = False
        for event in session.events.op_events[events_before:]:
            plan.hook_generated_labels.add(event.label_raw)
    if not _is_tf_tensor_like(replacement, tf):
        raise BackendUnsupportedError(
            f"tf intervention site {site.plan_id} returned "
            f"{type(replacement).__name__}; replacements must be TensorFlow tensors."
        )
    original_shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    replacement_shape = tuple(int(dim) for dim in getattr(replacement, "shape", ()))
    if replacement_shape != original_shape or str(replacement.dtype) != str(tensor.dtype):
        raise BackendUnsupportedError(
            f"tf intervention site {site.plan_id} changed shape/dtype "
            f"({original_shape}/{tensor.dtype} -> {replacement_shape}/"
            f"{replacement.dtype}); shape-preserving replacements only."
        )
    fire_result = _fire_result(site, label, session, tensor, replacement)
    if label is not None:
        _mark_intervention_event(session, label, fire_result)
    # Accounting is UNCONDITIONAL: a fire whose producer label could not be
    # resolved still fired, and the zero-match audit must not misreport it.
    plan.fired_site_labels.append((site.plan_id, label or "<unresolved>"))
    return replacement


def _fire_result(
    site: TFInterventionSite, label: str | None, session: Any, tensor: Any, replacement: Any
) -> FireResult:
    """Build the normalized fire record for one site firing."""

    original_shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
    replacement_shape = tuple(int(dim) for dim in getattr(replacement, "shape", ()))
    selector = site.selector if site.selector is not None else site.predicate
    return FireResult(
        plan_id=site.plan_id,
        site_label=label or "<unresolved>",
        fired_at_capture_index=len(session.events.op_events),
        pre_hook_shape=original_shape,
        post_hook_shape=replacement_shape,
        pre_hook_dtype=str(tensor.dtype),
        post_hook_dtype=str(replacement.dtype),
        replaced=replacement is not tensor,
        fire_record={
            "backend": "tf",
            "plan_id": site.plan_id,
            "site_label": label or "<unresolved>",
            "level": site.level,
            "selector": repr(selector),
            "replaced": replacement is not tensor,
        },
    )


def _mark_intervention_event(session: Any, label: str, fire_result: FireResult) -> None:
    """Mark one captured op event as an intervention site.

    Parameters
    ----------
    session
        Active capture session.
    label
        Raw label of the site op event.
    fire_result
        Normalized fire record.

    Returns
    -------
    None
        Appends one typed amendment to the session journal.
    """

    # ``op_event_by_label_raw`` is the live-index view, so repeated firings on
    # one site read the already-amended record and accumulate fire_results.
    event = session.events.op_event_by_label_raw.get(label)
    if event is None:
        return
    # Post-commit knowledge rides the typed amendment lane (P4): the site op
    # was committed by the op_callback before the wrap layer fired. In-place
    # op_events[i] writes are reserved for the ledgered cook-owned-projection
    # carve-out (tests/producer_parity/test_op_lane_inplace_writers.py) and
    # are never sanctioned here.
    session.events.append_amendment(
        amend_module_exit_intervention(
            event.seq,
            label,
            intervention_fired=True,
            is_edited=fire_result.replaced,
            fire_results=(*event.fire_results, fire_result),
        )
    )


def audit_tf_site_reachability(
    plan: TFInterventionPlan, session: Any, exempt: frozenset[str] = frozenset()
) -> None:
    """Fail closed on op-level sites the wrap layer could not reach.

    Parameters
    ----------
    plan
        Normalized intervention plan after the forward.
    session
        Capture session holding the full callback event stream.
    exempt
        Labels a ``stop_after`` halt cut off before Python received them.

    Returns
    -------
    None
        Returns when every selector-matched captured op was reachable.

    Raises
    ------
    TFInterventionSiteUnreachableError
        When a site fired on captured ops that never passed through the wrap layer.
    """

    by_site: dict[str, list[str]] = {}
    for label, due in plan.pending_decisions.items():
        if label in exempt:
            continue
        event = session.events.op_event_by_label_raw.get(label)
        op_type = str(event.function.func_name) if event is not None else "?"
        for site in due:
            by_site.setdefault(site.plan_id, []).append(f"{op_type} ({label})")
    for plan_id, unreachable in by_site.items():
        raise TFInterventionSiteUnreachableError(
            f"tf intervention site {plan_id} matched captured ops that never "
            f"passed through the writable wrap layer: {', '.join(unreachable[:8])}. "
            "TensorFlow op callbacks are read-only, so only ops whose output returns "
            "through a python op wrapper (generated tf ops and the tf.nn/tf.math entry "
            "points) or a module boundary can be substituted; refusing instead of "
            "silently not intervening."
        )
    # Zero-match disclosure: a planned site with zero fires (a selector matching
    # nothing, or a module boundary never reached) warns, as torch and paddle do.
    import warnings as _warnings

    fired_plan_ids = {plan_id for plan_id, _label in plan.fired_site_labels}
    for planned_site in (*plan.op_sites, *plan.module_sites):
        if planned_site.plan_id not in fired_plan_ids:
            _warnings.warn(
                f"tf intervention site {planned_site.plan_id} fired at zero sites during "
                "the forward; the capture is byte-identical to plain capture for that "
                "entry (check the selector's op/module name).",
                UserWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class TFEagerWiring:
    """What one wired eager session carries beyond the session itself.

    Parameters
    ----------
    plan
        Normalized intervention plan, or ``None`` without ``intervene=``.
    halt
        ``stop_after`` runtime, or ``None`` without ``stop_after=``.
    """

    plan: Any
    halt: TFHaltRuntime | None


def wire_eager_session(trace: Any, session: Any, intervene: Any, halt: Any) -> TFEagerWiring:
    """Install intervention, halt and source-context hooks on one eager session.

    Parameters
    ----------
    trace
        Trace being captured.
    session
        ``TFEagerCaptureSession`` about to run; it carries the imported TensorFlow
        module and the discovered module tree (``None`` in function-root mode).
    intervene
        Public ``intervene=`` value, or ``None``.
    halt
        Public ``stop_after=`` value, or ``None``.

    Returns
    -------
    TFEagerWiring
        Plan and halt runtime for :func:`run_wired_session`.
    """

    tf = session.tf
    plan = None
    edit_hook = None
    intervene, grad_clauses = split_tf_grad_clauses(intervene)
    stage_tf_grad_clauses(trace, grad_clauses)
    if intervene is not None:
        plan = normalize_tf_interventions(intervene, tf)
        if plan.module_sites and session.module_tree is None:
            raise BackendUnsupportedError(
                "tf module-boundary interventions (tl.module/tl.in_module "
                "conditions) require object-module attribution; this capture "
                "entry discovered no Keras/tf.Module tree."
            )
        # The session calls the hook as (frame, module_type, output, module_stack).
        session.module_exit_hook = partial(apply_tf_module_intervention, plan, session, tf, trace)
        if plan.op_sites:
            edit_hook = partial(observe_tf_op_sites, plan, tf)
    halt_runtime = (
        TFHaltRuntime(halt, trace, isinstance(halt, BaseSelector)) if halt is not None else None
    )
    if halt_runtime is not None and plan is not None and plan.op_sites:
        halt_runtime.edit_pending = plan.pending_decisions.__contains__
        plan.on_presented = halt_runtime.release
    session.output_hook = chain_output_hooks(
        edit_hook, halt_runtime.observe if halt_runtime is not None else None
    )
    session.code_context_fn = TFSourceContext(
        trace.num_context_lines, bool(trace._options_save_source_context)
    )
    return TFEagerWiring(plan=plan, halt=halt_runtime)


def run_wired_session(tf: Any, session: Any, wiring: TFEagerWiring) -> Any:
    """Run the forward under the writable wrap layer, then audit site reachability.

    Parameters
    ----------
    tf
        Imported TensorFlow module.
    session
        Wired capture session.
    wiring
        Result of :func:`wire_eager_session`.

    Returns
    -------
    TFCaptureResult
        The session result (``stop_signal`` set when ``stop_after`` halted).
    """

    try:
        if wiring.plan is None:
            return session.run()
        with tf_intervention_wrap(tf, wiring.plan, session):
            result = session.run()
    except CallbackAbort as abort:
        # A typed refusal raised by an observer inside the op callback.
        raise abort.error.with_traceback(abort.error.__traceback__) from None
    cut = wiring.halt.cut_labels if wiring.halt is not None else frozenset()
    audit_tf_site_reachability(wiring.plan, session, exempt=cut)
    return result


def finish_halt(trace: Any, wiring: TFEagerWiring, signal: Any) -> None:
    """Write the halt outcome, or settle a ``stop_after`` that never fired.

    Parameters
    ----------
    trace
        Trace being finalized.
    wiring
        Result of :func:`wire_eager_session`.
    signal
        ``StopSignal`` that ended the forward, or ``None``.
    """

    apply_halt_outcome(trace, signal)
    if wiring.halt is not None:
        wiring.halt.settle_unfired(trace)


__all__ = [
    "TFEagerWiring",
    "TFInterventionPlan",
    "TFInterventionSite",
    "TFInterventionSiteUnreachableError",
    "apply_tf_module_intervention",
    "audit_tf_site_reachability",
    "finish_halt",
    "normalize_tf_interventions",
    "observe_tf_op_sites",
    "run_wired_session",
    "split_tf_grad_clauses",
    "tf_gradient_replay_intervene",
    "tf_intervention_wrap",
    "wire_eager_session",
]
