"""Live interventions and stop sites for the MLX backend.

The MLX wrapper sees each concrete lazy output before the caller, and the
derived-grad tap observer already proved mid-forward output substitution, so
``trace(intervene=tl.when(...))`` and ``trace(stop_after=...)`` sites apply at
the wrapper boundary. Static selectors and value-dependent conditions
(``tl.where`` predicates and bare callables) are both admitted; a
value-dependent condition sees torch's live predicate context (value fields
``None``), so nothing is evaluated at the op and every capture stays
realize-once (see "Live site matching" below).

Everything here fails closed. A predicate that is not the ``tl.when`` shape, a
selector kind outside the admitted set, a non-forward or
grad-preserving decision, a helper without an MLX-native application, and a
fired hook that changes shape or dtype all refuse typed instead of silently
not-intervening.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ... import _state
from ..._errors import InvalidArgumentError
from ..._ir.intervention import FireResult
from ..._ir.op_record import amend_module_exit_intervention
from ..._ir.predicate import RecordContext
from ..._ir.selector_eval import (
    first_selector_kind_outside,
    module_address_matches,
    module_union_addresses,
    selector_contains_kind,
)
from ..._postprocess._selective_save import _STATIC_INTERVENTION_SELECTOR_KINDS
from ..._selection import selectors as _selectors
from ...intervention._types import Edit, InterventionDecision
from ...intervention.hooks import EDIT_SIGNAL_MISMATCH_REMEDY, make_hook_context, normalize_hook
from ..registry import BackendUnsupportedError
from ._mlx_helpers import mlx_helper_apply
from ._tap_observers import _replace_mlx_array_leaves

#: Refusal code for gradient-universe sites (registered in `_protocols.mlx`).
_MLX_REFUSAL_BACKWARD = "backward_capture_unsupported"

MLX_STATIC_INTERVENTION_SELECTOR_KINDS = _STATIC_INTERVENTION_SELECTOR_KINDS
"""Selector kinds resolvable from static capture-time labels on MLX.

Alias of the neutral authority table (the ``output`` drop is declared
there) -- never re-spell the kinds here.
"""

#: Kinds admitted on the live edit and stop surfaces: the static-label kinds
#: plus ``predicate`` (``tl.where`` and bare callables), whose value fields
#: ``_live_sites._value_record_context`` fills for the run.
_MLX_LIVE_SELECTOR_KINDS = MLX_STATIC_INTERVENTION_SELECTOR_KINDS | frozenset({"predicate"})

#: Selector kinds that may target the short ``{layer_type}_{type_index}``
#: alias, which is only visible through a second evaluation with the label
#: rewritten (mirrors the torch capture-path alias retry).
_ALIAS_RETRY_KINDS = ("label", "contains", "regex")

#: Helper specs with an MLX-native application (``_mlx_helpers`` holds each
#: one; this literal is the closed table the stochastic-family check reads and
#: must list the same names). The torch factories build hooks over ``torch.*``
#: calls that reject MLX arrays, so anything outside this table refuses typed
#: rather than half-working.
_MLX_SUPPORTED_HELPER_NAMES = (
    "zero_ablate",
    "scale",
    "add",
    "clamp",
    "mean_ablate",
    "replace_with",
    "steer",
    "project_onto",
    "project_out",
    "clip_norm",
)


@dataclass(frozen=True)
class MLXHookApplier:
    """Resolved MLX-native application of one intervention decision.

    Parameters
    ----------
    identity:
        Stable identity string pinned into the emit-time replay inventory.
    apply:
        Callable mapping one matched output array to its replacement.
    """

    identity: str
    apply: Callable[[Any], Any]


@dataclass(frozen=True)
class MLXInterventionPlan:
    """Resolved static-label intervention plan for one MLX capture.

    Parameters
    ----------
    selector:
        Static-label site selector from the ``tl.when`` predicate; for a
        ``tl.grad(...)`` site, the forward site whose output gradient it names.
    applier:
        MLX-native hook application for matched output leaves.
    gradient:
        Whether the site is a ``tl.grad(...)`` address: the forward is left
        unchanged and the action edits the derived gradients.
    grad_site_labels:
        Raw labels the gradient site matched during the forward, filled by
        ``_match_live_sites`` and read by the derived-gradient replay.
    module_exit:
        Whether the site is a ``tl.module(...)`` address, edited at the
        module's exit outside any wrapped MLX call.
    """

    selector: _selectors.BaseSelector
    applier: MLXHookApplier
    gradient: bool = False
    grad_site_labels: list[str] = field(default_factory=list)
    module_exit: bool = False


def selector_matches_capture_context(selector: _selectors.BaseSelector, ctx: Any) -> bool:
    """Evaluate a static selector against one capture-time record context.

    Parameters
    ----------
    selector:
        Static-label selector.
    ctx:
        Capture-time ``RecordContext`` for one output leaf.

    Returns
    -------
    bool
        ``True`` when the selector matches, including through the short
        ``{layer_type}_{type_index}`` alias retry for label-universe kinds.
    """

    if module_union_addresses(selector) is not None:
        return False
    if bool(selector(ctx)):
        return True
    if ctx.layer_type is None or ctx.type_index is None:
        return False
    if not any(selector_contains_kind(selector, kind) for kind in _ALIAS_RETRY_KINDS):
        return False
    alias_ctx = replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}")
    return bool(selector(alias_ctx))


def resolve_mlx_intervention_plan(
    intervene: Any | None,
    halt: Any | None,
    mx: Any,
) -> tuple[MLXInterventionPlan | None, _selectors.BaseSelector | None]:
    """Validate and resolve the MLX intervention/halt surface.

    Parameters
    ----------
    intervene:
        Public ``trace(intervene=...)`` value.
    halt:
        Public ``trace(stop_after=...)`` value.
    mx:
        Imported ``mlx.core`` module used by the resolved appliers.

    Returns
    -------
    tuple[MLXInterventionPlan | None, _selectors.BaseSelector | None]
        Resolved intervention plan and halt selector.

    Raises
    ------
    BackendUnsupportedError
        For any shape outside the supported static-label contract.
    """

    plan = None if intervene is None else _resolve_intervene_plan(intervene, mx)
    halt_selector: _selectors.BaseSelector | None = None
    if halt is not None:
        halt_selector = _as_selector(halt)
        if not isinstance(halt_selector, _selectors.BaseSelector):
            raise BackendUnsupportedError(
                "MLX backend supports trace(stop_after=...) for selectors "
                "(tl.func, tl.label, tl.label_contains, tl.in_module, tl.where and "
                "boolean composites) and callables taking the op context; got "
                f"{type(halt).__name__}."
            )
        _reject_non_static_selector(halt_selector, surface="halt")
    return plan, halt_selector


def _resolve_intervene_plan(intervene: Any, mx: Any) -> MLXInterventionPlan:
    """Resolve the one ``tl.when`` rule of ``trace(intervene=...)`` to a plan.

    Raises
    ------
    BackendUnsupportedError
        For any shape outside the supported static-label contract.
    """

    selector = _as_selector(getattr(intervene, "selector", None))
    decision = getattr(intervene, "decision", None)
    if not isinstance(selector, _selectors.BaseSelector) or not isinstance(
        decision, InterventionDecision
    ):
        raise BackendUnsupportedError(
            "MLX backend supports trace(intervene=...) only as a one-rule "
            "tl.when(condition, action), where the condition is a selector "
            "(tl.func, tl.label, tl.module, tl.in_module, tl.where and their "
            "&, |, ~ composites) or a callable taking the op context. Use "
            "that shape or the PyTorch backend."
        )
    gradient, selector = _gradient_site_view(selector)
    _reject_non_static_selector(selector, surface="intervene")
    _validate_module_selector(selector)
    return MLXInterventionPlan(
        selector=selector,
        applier=_resolve_decision_applier(decision, mx, gradient=gradient),
        gradient=gradient,
        module_exit=module_union_addresses(selector) is not None,
    )


def _as_selector(condition: Any) -> Any:
    """Return a bare callable condition as its ``tl.where`` selector.

    A selector passes through; any other value is returned unchanged for the
    caller's typed refusal.
    """

    if isinstance(condition, (_selectors.BaseSelector, type)):
        return condition
    if callable(condition):
        return _selectors.WhereSelector(condition)
    return condition


def _gradient_site_view(selector: _selectors.BaseSelector) -> tuple[bool, Any]:
    """Return whether the site is a ``tl.grad`` address, and the forward site to match.

    A ``tl.grad(s)`` site (or a union of them) names the gradient arriving at
    the output of each forward op ``s`` matches; on MLX the action edits that
    gradient in the derived-gradient run (``_edit_replay``). A site mixing
    universes, an autograd-node site and a backward-pass qualifier refuse.
    """

    from ...intervention.hooks import gradient_inners, split_site_branches

    universes = {universe for universe, _branch in split_site_branches(selector)}
    if universes == {"activation"}:
        return False, selector
    if universes != {"gradient"} or selector_contains_kind(selector, "backward_pass"):
        raise BackendUnsupportedError(
            f"MLX backend supports tl.grad(...) sites and forward sites, one kind per "
            f"rule; the site {selector!r} mixes them, names an autograd node or "
            "qualifies a backward pass (MLX derives one gradient per capture).",
            code=_MLX_REFUSAL_BACKWARD,
        )
    inners = [
        _selectors.label(inner) if isinstance(inner, str) else inner
        for inner in gradient_inners(selector)
    ]
    forward_site = inners[0]
    for inner in inners[1:]:
        forward_site = forward_site | inner
    if selector_contains_kind(forward_site, "module"):
        raise BackendUnsupportedError(
            "MLX backend tl.grad(...) sites take op-level forward sites (tl.func, "
            "tl.label, tl.in_module, tl.where); a tl.module(...) gradient site is not "
            "supported on MLX.",
            code=_MLX_REFUSAL_BACKWARD,
        )
    return True, forward_site


def _reject_non_static_selector(selector: _selectors.BaseSelector, *, surface: str) -> None:
    """Refuse selector trees containing kinds outside the admitted live set.

    Parameters
    ----------
    selector:
        Candidate selector tree.
    surface:
        Public option name used in diagnostics.
    """

    unsupported = first_selector_kind_outside(selector, allowed=_MLX_LIVE_SELECTOR_KINDS)
    if unsupported is None:
        return
    raise BackendUnsupportedError(
        f"MLX backend supports trace({surface}=...) for the selectors tl.func, "
        "tl.label, tl.label_contains, tl.module, tl.in_module, tl.where and "
        f"boolean composites (&, |, ~) of those; unsupported selector kind "
        f"{unsupported!r}. Use one of those selectors or the PyTorch backend."
    )


def _resolve_decision_applier(
    decision: InterventionDecision, mx: Any, *, gradient: bool = False
) -> MLXHookApplier:
    """Resolve one intervention decision to an MLX-native applier.

    Parameters
    ----------
    decision:
        Normalized predicate-time decision from ``tl.when``.
    mx:
        Imported ``mlx.core`` module.
    gradient:
        Whether the action edits a gradient (its callable hook then gets a
        backward-signal context, as torch's gradient hooks do).

    Returns
    -------
    MLXHookApplier
        Resolved applier.

    Raises
    ------
    BackendUnsupportedError
        For actions, flags, or helpers outside the MLX contract.
    """

    if decision.keep_grad or decision.isolate:
        raise BackendUnsupportedError(
            "MLX backend interventions do not support keep_grad or isolate; "
            "both are torch-autograd semantics with no MLX equivalent."
        )
    if decision.action not in {"add_hook", "transform"}:
        raise BackendUnsupportedError(
            f"MLX backend interventions do not support action {decision.action!r}; "
            "supported actions are hook helpers and callable transforms."
        )
    hook = decision.hook
    if hook is None:
        raise BackendUnsupportedError(
            "MLX backend interventions require an action; tl.when(selector, None) "
            "declares no replacement to apply."
        )
    if isinstance(hook, Edit):
        if gradient:
            _refuse_forward_signal_edit(hook)
        return _resolve_helper_applier(hook, mx)
    return _resolve_callable_applier(hook, signal="backward" if gradient else "forward")


def _refuse_forward_signal_edit(spec: Edit) -> None:
    """Refuse an edit that reads recorded forward state at a ``tl.grad(...)`` site.

    The same check and message torch runs at attach time: a neutral edit
    (``Edit.signal is None``) is a function of one tensor and edits a gradient;
    ``mean_ablate`` and other forward-signal edits have no meaning there.

    Raises
    ------
    InvalidArgumentError
        ``edit_signal_mismatch``.
    """

    if spec.signal != "forward":
        return
    raise InvalidArgumentError(
        f"{spec.name} reads recorded forward activations and cannot edit the gradient "
        f"at a tl.grad(...) site. Nothing was attached. Remedy: {EDIT_SIGNAL_MISMATCH_REMEDY}",
        code="edit_signal_mismatch",
        remedy=EDIT_SIGNAL_MISMATCH_REMEDY,
        argument="value_or_hook",
        verb=spec.name,
    )


def _resolve_helper_applier(spec: Edit, mx: Any) -> MLXHookApplier:
    """Resolve a built-in helper spec to its MLX-native applier.

    Parameters
    ----------
    spec:
        Built-in helper spec.
    mx:
        Imported ``mlx.core`` module.

    Returns
    -------
    MLXHookApplier
        Curated MLX application of the helper (``_mlx_helpers``).
    """

    return MLXHookApplier(f"helper:{spec.name}", mlx_helper_apply(spec, mx))


def _resolve_callable_applier(
    hook: Callable[..., Any], *, signal: Any = "forward"
) -> MLXHookApplier:
    """Resolve a user callable transform to an applier.

    Parameters
    ----------
    hook:
        User callable with the standard ``fn(out, *, ctx)`` hook signature.
    signal:
        Hook-context signal: ``"forward"`` for an activation edit,
        ``"backward"`` for a ``tl.grad`` site.

    Returns
    -------
    MLXHookApplier
        Applier invoking the callable with a minimal hook context.
    """

    normalized = normalize_hook(hook)
    identity = f"callable:{getattr(hook, '__qualname__', type(hook).__name__)}"

    def _apply(out: Any) -> Any:
        """Invoke the user hook with a minimal post-forward hook context."""

        context = make_hook_context(
            name=identity,
            timing="post",
            signal=signal,
        )
        return normalized(out, ctx=context)

    return MLXHookApplier(identity, _apply)


__all__ = [
    "MLX_STATIC_INTERVENTION_SELECTOR_KINDS",
    "MLXHookApplier",
    "MLXInterventionPlan",
    "resolve_mlx_intervention_plan",
    "selector_matches_capture_context",
]


def _validate_module_selector(selector: Any) -> None:
    """Refuse module trees whose meaning needs both dispatch doors."""

    if selector_contains_kind(selector, "module") and module_union_addresses(selector) is None:
        raise BackendUnsupportedError(
            "MLX backend module interventions require tl.module(...) or a union "
            "of module selectors; mixed or negated module selectors are unsupported.",
            code="interventions_unsupported",
        )


def _apply_module_exit(trace: Any, backend: Any, frame: Any, output: Any) -> Any:
    """Replace each distinct output leaf once at a selected module exit."""

    plan = trace._mlx_intervention_plan
    if frame is None or plan is None:
        return output
    addresses = module_union_addresses(plan.selector)
    if not addresses or not any(
        module_address_matches((frame.address, frame.call_index), address) for address in addresses
    ):
        return output
    replacements = {}
    for tensor in backend._iter_arrays(output):
        if id(tensor) not in replacements:
            replacements[id(tensor)] = _emit_module_leaf(trace, backend, plan, tensor)
    return _replace_mlx_array_leaves(backend, output, replacements)


def apply_module_exit_and_record(
    trace: Any,
    backend: Any,
    frame: Any,
    output: Any,
    module_tree: Any,
) -> Any:
    """Apply any module-exit intervention, then record what the module call returns.

    Parameters
    ----------
    trace:
        Active TorchLens trace.
    backend:
        MLX backend receiving wrapper events.
    frame:
        The call's module frame, or ``None`` when the call is not a discovered module.
    output:
        The module's raw return value.
    module_tree:
        Discovered ``MLXModuleTree`` holding the capture's ``module_returns`` ledger.

    Returns
    -------
    Any
        The value the caller receives (``output``, or the tree with intervention-replaced
        leaves). That value is the one recorded, so recorded leaves are the values
        downstream code consumes; intervention behaviour is unchanged.
    """

    returned = _apply_module_exit(trace, backend, frame, output)
    if frame is not None and module_tree is not None:
        module_tree.module_returns.record(
            frame.address,
            frame.call_index,
            returned,
            is_leaf=backend.is_tensor,
            label_of=backend.tensor_store.get_label,
        )
    return returned


def _module_capture_op(trace: Any, frame: Any, op_name: str | None) -> str | None:
    """Expose selected descendants instead of capturing their parent as one opaque op."""

    plan = trace._mlx_intervention_plan
    if frame is None or plan is None:
        return op_name
    addresses = module_union_addresses(plan.selector) or ()
    prefix = frame.address + "." if frame.address not in ("", "self") else ""
    if any(
        address.rsplit(":", 1)[0].startswith(prefix) and address.rsplit(":", 1)[0] != frame.address
        for address in addresses
    ):
        return None
    return op_name


def _copy_leaf(tensor: Any) -> Any:
    """Produce a distinct identity value so aliasing keeps the raw producer intact."""

    return tensor.astype(tensor.dtype)


def _emit_module_leaf(trace: Any, backend: Any, plan: Any, tensor: Any) -> Any:
    """Use the existing op intervention/replay path for the boundary replacement."""

    from ..._selection.selectors import FuncSelector  # deferred: L4 vocabulary

    label = backend.tensor_store.get_label(tensor)
    # A separate identity boundary preserves the original producer's replay
    # payload, including consumers that ran before this module returned it.
    trace._mlx_intervention_plan = replace(plan, selector=FuncSelector("interventionreplacement"))
    trace._mlx_capture_depth += 1
    try:
        with _state.pause_logging():
            raw = _copy_leaf(tensor)
            replacement = backend.emit_mlx_operation(
                trace,
                "interventionreplacement",
                _copy_leaf,
                (tensor,),
                {},
                raw,
                module_stack=tuple(trace._mlx_module_stack),
            )
    finally:
        trace._mlx_capture_depth -= 1
        trace._mlx_intervention_plan = plan
    events = trace.capture_events
    boundary = events.op_event_by_label_raw[backend.tensor_store.get_label(replacement)]
    event = events.op_event_by_label_raw.get(label)
    if event is not None:
        events.append_amendment(
            amend_module_exit_intervention(
                event.seq,
                label,
                intervention_fired=True,
                is_edited=True,
                fire_results=(
                    *event.fire_results,
                    *(
                        replace(
                            fire,
                            site_label=label,
                            fire_record={
                                "backend": "mlx",
                                "plan_id": fire.plan_id,
                                "site_label": label,
                                "level": "module",
                                "selector": repr(plan.selector),
                                "replaced": fire.replaced,
                            },
                        )
                        for fire in boundary.fire_results
                    ),
                ),
            )
        )
    return replacement


# -- Live site matching -------------------------------------------------------
# ``MLXBackend.emit_mlx_operation`` hands each wrapped call's output leaves to
# ``_match_live_sites`` before labeling. A matched edit substitutes the leaf (so
# the recorded output IS the value downstream ops consume) and records the
# applier slot the replay oracle re-applies; a matched stop site names the halt
# frontier.
#
# Value-dependent conditions (``tl.where`` predicates and bare callables) see
# the same predicate context as on torch's live capture path: shape, dtype,
# labels and module fields are real, and the value fields ``is_scalar_bool``
# and ``bool_value`` are ``None`` (torch's ``_build_record_context`` leaves them
# unset during the forward). Nothing is evaluated at the op, so a run with a
# value-dependent condition stays realize-once and keeps MLX's graph fusion.
# ``tensor_requires_grad`` stays deferred: MLX arrays carry no per-array
# gradient flag (gradients come from function transforms), so reading it
# raises ``MLXValueUnavailableError`` instead of guessing.


@dataclass
class _LiveSiteMatches:
    """What the edit and stop selectors matched among one call's output leaves."""

    replacements: dict[int, Any] = field(default_factory=dict)
    fire_results_by_site: dict[int, tuple[FireResult, ...]] = field(default_factory=dict)
    slots: list[tuple[int, str]] = field(default_factory=list)
    appliers: list[tuple[int, Any]] = field(default_factory=list)
    halt_label: str | None = None


def _reads_values(selector: Any) -> bool:
    """Whether a selector tree holds a value-dependent (predicate) condition."""

    return selector is not None and selector_contains_kind(selector, "predicate")


def _value_record_context(ctx: RecordContext) -> RecordContext:
    """Give a predicate context torch's live value fields.

    Parameters
    ----------
    ctx
        Deferred-value context from ``build_record_context``.

    Returns
    -------
    RecordContext
        ``ctx`` with ``is_scalar_bool`` and ``bool_value`` set to ``None``, as on
        torch's live capture path, so a predicate that reads them sees the same
        values on both backends and no array is evaluated.
    """

    return replace(ctx, is_scalar_bool=None, bool_value=None)


def _match_live_sites(
    backend: Any,
    trace: Any,
    outputs: tuple[Any, ...],
    reserved: Any,
    func_event_input: Any,
) -> _LiveSiteMatches:
    """Match the active edit plan and stop selector against one call's leaves.

    Parameters
    ----------
    backend
        Active MLX backend.
    trace
        Trace under capture (holds the plan and the stop selector).
    outputs
        The call's output array leaves.
    reserved
        Reserved labels, one per leaf.
    func_event_input
        The call's event input.

    Returns
    -------
    _LiveSiteMatches
        Replacements keyed by leaf identity, fire results and applier slots by
        leaf index, and the halt frontier label when the stop site matched.
    """

    plan = trace._mlx_intervention_plan
    halt_selector = trace._mlx_halt_selector
    with_values = _reads_values(getattr(plan, "selector", None)) or _reads_values(halt_selector)
    found = _LiveSiteMatches()
    for index, (leaf, entry) in enumerate(zip(outputs, reserved, strict=True)):
        record_ctx = backend.build_record_context(trace, entry, func_event_input, leaf)
        if with_values:
            record_ctx = _value_record_context(record_ctx)
        if plan is not None and selector_matches_capture_context(plan.selector, record_ctx):
            _record_site_match(backend, trace, plan, found, index, leaf, entry)
        if (
            halt_selector is not None
            and found.halt_label is None
            and selector_matches_capture_context(halt_selector, record_ctx)
        ):
            found.halt_label = entry.label_raw
    return found


def _record_site_match(
    backend: Any,
    trace: Any,
    plan: Any,
    found: _LiveSiteMatches,
    index: int,
    leaf: Any,
    entry: Any,
) -> None:
    """Apply the edit at one matched leaf, or note a ``tl.grad`` site for the replay."""

    trace._mlx_intervene_fired = True
    if plan.gradient:
        # A tl.grad site leaves the forward unchanged; the derived-gradient
        # replay (_edit_replay) edits the gradient arriving at this output.
        plan.grad_site_labels.append(entry.label_raw)
        return
    replacement, fire_result = _apply_edit(backend, plan, leaf, entry)
    found.replacements[id(leaf)] = replacement
    found.fire_results_by_site[index] = (fire_result,)
    found.slots.append((index, plan.applier.identity))
    found.appliers.append((index, plan.applier.apply))


def _apply_edit(backend: Any, plan: Any, leaf: Any, entry: Any) -> tuple[Any, FireResult]:
    """Apply the plan to one matched leaf.

    Returns
    -------
    tuple[Any, FireResult]
        The replacement array and the fire result recorded for the leaf's site.
    """

    replacement = backend._apply_mlx_intervention(plan, leaf)
    fire_result = FireResult(
        plan_id="mlx_intervene",
        site_label=entry.label_raw,
        fired_at_capture_index=entry.raw_index,
        pre_hook_shape=backend._shape(leaf),
        post_hook_shape=backend._shape(replacement),
        pre_hook_dtype=backend._dtype(leaf),
        post_hook_dtype=backend._dtype(replacement),
        replaced=True,
        fire_record=None,
    )
    return replacement, fire_result
