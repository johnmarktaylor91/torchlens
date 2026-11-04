"""Static-label live interventions for the technical-preview MLX backend.

The MLX wrapper sees each concrete lazy output before the caller, and the
derived-grad tap observer already proved mid-forward output substitution, so
static-label ``trace(intervene=tl.when(...))`` and ``trace(stop_after=...)`` sites
apply at the wrapper boundary. Value-dependent predicates stay refused typed:
predicate-time values would need per-op ``mx.eval``, which would poison lazy
evaluation.

Everything here fails closed. A predicate that is not the ``tl.when`` static
shape, a selector kind outside the static-label set, a non-forward or
grad-preserving decision, a helper without an MLX-native application, and a
fired hook that changes shape or dtype all refuse typed instead of silently
not-intervening.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ... import _state
from ..._ir.op_record import amend_module_exit_intervention
from ..._ir.selector_eval import (
    first_selector_kind_outside,
    module_address_matches,
    module_union_addresses,
    selector_contains_kind,
)
from ..._postprocess._selective_save import _STATIC_INTERVENTION_SELECTOR_KINDS
from ..._selection.selectors import BaseSelector, FuncSelector
from ...intervention.hooks import make_hook_context, normalize_hook
from ...intervention.types import Edit, InterventionDecision
from ..registry import BackendUnsupportedError
from ._tap_observers import _replace_mlx_array_leaves

#: Refusal code for gradient-universe sites (registered in `_protocols.mlx`).
_MLX_REFUSAL_BACKWARD = "backward_capture_unsupported"

MLX_STATIC_INTERVENTION_SELECTOR_KINDS = _STATIC_INTERVENTION_SELECTOR_KINDS
"""Selector kinds resolvable from static capture-time labels on MLX.

Alias of the neutral authority table (the ``output`` drop is declared
there) -- never re-spell the kinds here.
"""

#: Selector kinds that may target the short ``{layer_type}_{type_index}``
#: alias, which is only visible through a second evaluation with the label
#: rewritten (mirrors the torch capture-path alias retry).
_ALIAS_RETRY_KINDS = ("label", "contains", "regex")

#: Helper specs with an MLX-native application. The torch factories build
#: hooks over ``torch.*`` calls that reject MLX arrays, so the curated
#: appliers below are the genuine mechanism — anything outside this table
#: refuses typed rather than half-working.
_MLX_SUPPORTED_HELPER_NAMES = (
    "zero_ablate",
    "scale",
    "add",
    "clamp",
    "mean_ablate",
    "replace_with",
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
        Static-label site selector from the ``tl.when`` predicate.
    applier:
        MLX-native hook application for matched output leaves.
    """

    selector: BaseSelector
    applier: MLXHookApplier


def selector_matches_capture_context(selector: BaseSelector, ctx: Any) -> bool:
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
) -> tuple[MLXInterventionPlan | None, BaseSelector | None]:
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
    tuple[MLXInterventionPlan | None, BaseSelector | None]
        Resolved intervention plan and halt selector.

    Raises
    ------
    BackendUnsupportedError
        For any shape outside the supported static-label contract.
    """

    plan: MLXInterventionPlan | None = None
    halt_selector: BaseSelector | None = None
    if intervene is not None:
        selector = getattr(intervene, "selector", None)
        decision = getattr(intervene, "decision", None)
        if not isinstance(selector, BaseSelector) or not isinstance(decision, InterventionDecision):
            raise BackendUnsupportedError(
                "MLX backend supports trace(intervene=...) only as "
                "tl.when(static_selector, action). Value-dependent intervention "
                "predicates need concrete activation values at predicate time; MLX "
                "lazy evaluation defers them without per-op mx.eval. Use "
                "tl.when(tl.func(...) | tl.label(...) | tl.in_module(...), action) "
                "or the PyTorch backend."
            )
        _reject_gradient_site(selector)
        _reject_non_static_selector(selector, surface="intervene")
        _validate_module_selector(selector)
        plan = MLXInterventionPlan(
            selector=selector,
            applier=_resolve_decision_applier(decision, mx),
        )
    if halt is not None:
        if not isinstance(halt, BaseSelector):
            raise BackendUnsupportedError(
                "MLX backend supports trace(stop_after=...) only for static-label "
                "selectors (tl.func, tl.label, tl.label_contains, tl.in_module and "
                "boolean composites). Value-dependent halt predicates need "
                "concrete values at predicate time; use the PyTorch backend."
            )
        _reject_non_static_selector(halt, surface="halt")
        halt_selector = halt
    return plan, halt_selector


def _reject_gradient_site(selector: BaseSelector) -> None:
    """Refuse a site outside the activation universe: MLX has no backward capture."""

    from ...intervention.hooks import split_site_branches

    if any(universe != "activation" for universe, _branch in split_site_branches(selector)):
        raise BackendUnsupportedError(
            "MLX backend supports forward interventions only; the site "
            f"{selector!r} addresses a gradient, which needs backward capture "
            "that MLX does not declare. Use the PyTorch backend for gradient edits.",
            code=_MLX_REFUSAL_BACKWARD,
        )


def _reject_non_static_selector(selector: BaseSelector, *, surface: str) -> None:
    """Refuse selector trees containing non-static kinds.

    Parameters
    ----------
    selector:
        Candidate selector tree.
    surface:
        Public option name used in diagnostics.
    """

    unsupported = first_selector_kind_outside(
        selector, allowed=MLX_STATIC_INTERVENTION_SELECTOR_KINDS
    )
    if unsupported is None:
        return
    raise BackendUnsupportedError(
        f"MLX backend supports trace({surface}=...) only for static-label "
        "selectors tl.func, tl.label, tl.label_contains, tl.module, tl.in_module and "
        f"boolean composites (&, |, ~) of those; unsupported selector kind "
        f"{unsupported!r}. Value-dependent predicates need concrete activation "
        "values at predicate time, which MLX lazy evaluation does not expose "
        "without per-op mx.eval; use the PyTorch backend."
    )


def _resolve_decision_applier(decision: InterventionDecision, mx: Any) -> MLXHookApplier:
    """Resolve one intervention decision to an MLX-native applier.

    Parameters
    ----------
    decision:
        Normalized predicate-time decision from ``tl.when``.
    mx:
        Imported ``mlx.core`` module.

    Returns
    -------
    MLXHookApplier
        Resolved applier.

    Raises
    ------
    BackendUnsupportedError
        For actions, flags, or helpers outside the MLX contract.
    """

    # Gradient sites (tl.grad(...)) are outside the static selector kinds the
    # caller admits, so every decision reaching here edits an activation.
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
        return _resolve_helper_applier(hook, mx)
    return _resolve_callable_applier(hook)


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
        Curated MLX application of the helper.
    """

    name = spec.name
    kwargs = dict(spec.kwargs)
    identity = f"helper:{name}"
    if name == "zero_ablate":
        return MLXHookApplier(identity, lambda out: mx.zeros_like(out))
    if name == "scale":
        factor = spec.args[0]
        return MLXHookApplier(identity, lambda out: out * factor)
    if name == "add":
        delta = spec.args[0]
        if not isinstance(delta, (int, float)) and not isinstance(delta, getattr(mx, "array", ())):
            raise BackendUnsupportedError(
                "MLX backend tl.add(...) interventions accept a Python scalar or "
                f"mx.array delta; got {type(delta).__name__}. Torch tensors cannot "
                "be added to MLX arrays."
            )
        return MLXHookApplier(identity, lambda out: out + delta)
    if name == "clamp":
        low = kwargs.get("low")
        high = kwargs.get("high")
        return MLXHookApplier(identity, lambda out: mx.clip(out, low, high))
    if name == "mean_ablate":
        source = spec.args[0] if spec.args else None
        if source is not None or kwargs.get("dim") is not None or "population_identity" in kwargs:
            raise BackendUnsupportedError(
                "MLX backend tl.mean_ablate(...) interventions support the "
                "self form with dim=None only; tensor and population sources and "
                "axis-aware means are torch-side semantics."
            )
        return MLXHookApplier(identity, lambda out: mx.zeros_like(out) + mx.mean(out))
    if name == "replace_with":
        value = spec.args[0]

        def _replace(out: Any) -> Any:
            """Return the replacement value, resolving a zero-arg callable at apply time."""

            del out
            return value() if callable(value) else value

        return MLXHookApplier(identity, _replace)
    raise BackendUnsupportedError(
        f"MLX backend has no native application for intervention helper {name!r}; "
        f"supported helpers are {', '.join(_MLX_SUPPORTED_HELPER_NAMES)}, plus callable "
        "transforms written against mx.array. The torch helper factories build "
        "torch-only hooks that reject MLX arrays, so admitting them would silently "
        "fail; use a callable transform instead."
    )


def _resolve_callable_applier(hook: Callable[..., Any]) -> MLXHookApplier:
    """Resolve a user callable transform to an applier.

    Parameters
    ----------
    hook:
        User callable with the standard ``fn(out, *, ctx)`` hook signature.

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
            signal="forward",
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
