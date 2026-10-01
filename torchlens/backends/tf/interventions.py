"""TensorFlow static-label intervention layer.

TensorFlow op callbacks are read-only in eager execution (spike-verified on
TF 2.21: returned replacement outputs are ignored), so interventions run
through a two-level WRITABLE layer on top of the untouched op-callback
capture spine:

1. Module-boundary sites (``tl.module`` / ``tl.in_module`` conditions)
   substitute Keras/`tf.Module` call outputs through the existing module
   ``__call__`` patch.
2. Op-level sites (``tl.func`` / ``tl.label`` / ``tl.contains`` conditions)
   substitute returns of a curated registry of python entry points that
   Keras-3 eager execution actually flows through.

Capture truth stays with op callbacks: replacement values are produced by
real ops that the callback records, so downstream records consume the
substituted values by construction. Site resolution FAILS CLOSED: an op
matched by a site's selector in the callback stream whose call never passed
through the wrap layer raises a typed unreachable error instead of silently
not intervening.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from ...intervention.selectors import BaseSelector
from ...intervention.types import HelperSpec, InterventionDecision
from ...ir.intervention import FireResult
from ...ir.op_record import amend_module_exit_intervention
from ...postprocess._selective_save import (
    _STATIC_INTERVENTION_SELECTOR_KINDS,
    reject_selector_outside_kinds,
)
from ..registry import BackendUnsupportedError

# Alias of the neutral authority table (the ``output`` drop is declared
# there) -- never re-spell the kinds here.
_TF_INTERVENTION_SELECTOR_KINDS = _STATIC_INTERVENTION_SELECTOR_KINDS
_MODULE_ONLY_KINDS = frozenset({"module", "in_module", "and", "or", "not"})

_CURATED_WRAP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("nn", "relu"),
    ("nn", "relu6"),
    ("nn", "leaky_relu"),
    ("nn", "elu"),
    ("nn", "selu"),
    ("nn", "gelu"),
    ("nn", "silu"),
    ("nn", "swish"),
    ("nn", "sigmoid"),
    ("nn", "softmax"),
    ("nn", "log_softmax"),
    ("nn", "softplus"),
    ("nn", "bias_add"),
    ("nn", "conv2d"),
    ("nn", "convolution"),
    ("nn", "max_pool2d"),
    ("nn", "avg_pool2d"),
    ("", "matmul"),
    ("", "add"),
    ("", "subtract"),
    ("", "multiply"),
    ("", "divide"),
    ("", "tanh"),
    ("", "sigmoid"),
    ("", "exp"),
    ("", "concat"),
    ("", "reduce_sum"),
    ("", "reduce_mean"),
    ("", "reduce_max"),
    ("math", "add"),
    ("math", "multiply"),
    ("math", "tanh"),
    ("math", "sigmoid"),
    ("math", "exp"),
    ("math", "log"),
    ("linalg", "matmul"),
)
"""Curated (submodule, name) python entry points wrapped for op-level sites."""


class TFInterventionSiteUnreachableError(BackendUnsupportedError):
    """A selector matched captured ops the writable wrap layer never saw."""


@dataclass(frozen=True)
class TFInterventionSite:
    """One normalized TensorFlow intervention site.

    Parameters
    ----------
    plan_id
        Stable site identifier within the plan.
    predicate
        ``tl.when``-built predicate returning a decision on match.
    selector
        Static selector used for matching and the reachability audit.
    decision
        Normalized intervention decision.
    hook
        Resolved TensorFlow replacement callable.
    level
        Dispatch level, ``"module"`` or ``"op"``.
    """

    plan_id: str
    predicate: Callable[[Any], InterventionDecision | None]
    selector: BaseSelector
    decision: InterventionDecision
    hook: Callable[[Any], Any]
    level: Literal["module", "op"]


@dataclass
class TFInterventionPlan:
    """Mutable per-capture TensorFlow intervention state.

    Parameters
    ----------
    sites
        All normalized sites.
    op_sites
        Op-level sites dispatched by the functional wrap layer.
    module_sites
        Module-boundary sites dispatched at module exits.
    """

    sites: tuple[TFInterventionSite, ...]
    op_sites: tuple[TFInterventionSite, ...]
    module_sites: tuple[TFInterventionSite, ...]
    presented_labels: set[str] = field(default_factory=set)
    hook_generated_labels: set[str] = field(default_factory=set)
    fired_site_labels: list[tuple[str, str]] = field(default_factory=list)
    _in_hook: bool = False


@dataclass(frozen=True)
class _TFModuleExitContext:
    """Selector subject for one TensorFlow module-boundary exit.

    Parameters
    ----------
    module
        Pass-qualified module-call label such as ``"dense:1"``.
    address
        Module address.
    module_type
        Module class name.
    module_stack
        Active module frames including the exiting module.
    source_trace
        Owning trace, so selector evaluation sees a non-torch backend.
    """

    module: str
    address: str
    module_type: str
    module_stack: tuple[dict[str, Any], ...]
    source_trace: Any
    kind: str = "module_exit"
    label: None = None
    raw_label: None = None


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
        selector = getattr(predicate, "selector", None)
        decision = getattr(predicate, "decision", None)
        if not isinstance(selector, BaseSelector) or not isinstance(decision, InterventionDecision):
            raise BackendUnsupportedError(
                "tf backend supports trace(intervene=...) built from "
                "tl.when(selector, action) with static selectors such as tl.func, "
                "tl.label, tl.module, tl.in_module, tl.contains, and boolean "
                "composites; value-dependent callable conditions need predicate-time "
                "semantics the tf preview does not implement."
            )
        reject_selector_outside_kinds(
            selector,
            allowed=_TF_INTERVENTION_SELECTOR_KINDS,
            backend_name="tf",
        )
        if decision.direction != "forward":
            raise BackendUnsupportedError(
                "tf backend interventions are forward-only; direction="
                f"{decision.direction!r} requires true backward capture."
            )
        hook = _resolve_tf_hook(decision, tf)
        level: Literal["module", "op"] = (
            "module" if _selector_kinds(selector) <= _MODULE_ONLY_KINDS else "op"
        )
        sites.append(
            TFInterventionSite(
                plan_id=f"tf_intervene_{index}",
                predicate=predicate,
                selector=selector,
                decision=decision,
                hook=hook,
                level=level,
            )
        )
    normalized = tuple(sites)
    return TFInterventionPlan(
        sites=normalized,
        op_sites=tuple(site for site in normalized if site.level == "op"),
        module_sites=tuple(site for site in normalized if site.level == "module"),
    )


def _selector_kinds(selector: Any) -> frozenset[str]:
    """Return every selector kind used in a selector tree.

    Parameters
    ----------
    selector
        Static selector or composite.

    Returns
    -------
    frozenset[str]
        Selector kinds present in the tree.
    """

    kinds: set[str] = set()
    stack = [selector]
    while stack:
        node = stack.pop()
        kind = getattr(node, "selector_kind", None)
        if kind is None:
            continue
        kinds.add(str(kind))
        value = getattr(node, "selector_value", None)
        if isinstance(value, (list, tuple)):
            stack.extend(value)
        elif isinstance(value, BaseSelector):
            stack.append(value)
        inner = getattr(node, "inner", None)
        if isinstance(inner, BaseSelector):
            stack.append(inner)
        members = getattr(node, "selectors", None)
        if isinstance(members, (list, tuple)):
            stack.extend(members)
    return frozenset(kinds)


def _resolve_tf_hook(decision: InterventionDecision, tf: Any) -> Callable[[Any], Any]:
    """Resolve a decision's hook into a TensorFlow replacement callable.

    Parameters
    ----------
    decision
        Normalized intervention decision.
    tf
        Imported TensorFlow module.

    Returns
    -------
    Callable[[Any], Any]
        Callable mapping the original output tensor to its replacement.
    """

    hook = decision.hook
    if isinstance(hook, HelperSpec):
        return _tf_helper_hook(hook, tf)
    if callable(hook):
        return _tf_callable_hook(hook)
    raise BackendUnsupportedError(
        f"tf backend interventions require a helper action or callable; got {type(hook).__name__}."
    )


def _tf_helper_hook(spec: HelperSpec, tf: Any) -> Callable[[Any], Any]:
    """Return the TensorFlow implementation for a curated helper spec.

    Parameters
    ----------
    spec
        Portable helper spec built by ``tl.zero_ablate``/``tl.scale``/``tl.add``.
    tf
        Imported TensorFlow module.

    Returns
    -------
    Callable[[Any], Any]
        TensorFlow replacement callable.
    """

    if spec.helper_name == "zero_ablate":
        return lambda out: tf.zeros_like(out)
    if spec.helper_name == "scale":
        factor = spec.args[0]
        return lambda out: out * tf.cast(factor, out.dtype)
    if spec.helper_name == "add":
        delta = spec.args[0]
        return lambda out: out + tf.cast(delta, out.dtype)
    raise BackendUnsupportedError(
        f"tf backend interventions implement the curated helpers zero_ablate, scale, "
        f"and add; helper {spec.helper_name!r} has no TensorFlow implementation. Pass "
        "a callable action for custom replacements."
    )


def _tf_callable_hook(hook: Callable[..., Any]) -> Callable[[Any], Any]:
    """Normalize a user callable into the single-argument hook convention.

    Parameters
    ----------
    hook
        User replacement callable, optionally accepting a ``hook`` keyword.

    Returns
    -------
    Callable[[Any], Any]
        Callable invoked with the original output tensor only.
    """

    try:
        parameter_names = set(inspect.signature(hook).parameters)
    except (TypeError, ValueError):
        parameter_names = set()
    if "hook" in parameter_names:
        return lambda out: hook(out, hook=None)
    return hook


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
        for submodule_name, attr_name in _CURATED_WRAP_ENTRIES:
            owner = tf if submodule_name == "" else getattr(tf, submodule_name, None)
            if owner is None:
                continue
            original = getattr(owner, attr_name, None)
            if original is None or not callable(original):
                continue
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
    """Evaluate op-level sites against one wrapped-call output tensor.

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
    event = session.events.op_event_by_label_raw.get(label)
    context = getattr(event, "record_context", None)
    if event is None or context is None:
        return tensor
    plan.presented_labels.add(label)
    current = tensor
    for site in plan.op_sites:
        decision = site.predicate(context)
        if decision is None:
            continue
        current = _fire_site(plan, session, tf, site, current, label)
    return current


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
                "pass_index": item.call_index,
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

    events_before = len(session.events.op_events)
    plan._in_hook = True
    try:
        replacement = site.hook(tensor)
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
    fire_result = FireResult(
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
            "selector": repr(site.selector),
            "replaced": replacement is not tensor,
        },
    )
    if label is not None:
        _mark_intervention_event(session, label, fire_result)
    # Accounting is UNCONDITIONAL: a fire whose producer label could not be
    # resolved still fired, and the zero-match audit must not misreport it.
    plan.fired_site_labels.append((site.plan_id, label or "<unresolved>"))
    return replacement


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
            intervention_replaced=fire_result.replaced,
            fire_results=(*event.fire_results, fire_result),
        )
    )


def audit_tf_site_reachability(plan: TFInterventionPlan, session: Any) -> None:
    """Fail closed on op-level sites the wrap layer could not reach.

    Parameters
    ----------
    plan
        Normalized intervention plan after the forward.
    session
        Capture session holding the full callback event stream.

    Returns
    -------
    None
        Returns when every selector-matched captured op was reachable.

    Raises
    ------
    TFInterventionSiteUnreachableError
        When a site's selector matched captured ops that never passed through
        the curated wrap layer.
    """

    for site in plan.op_sites:
        unreachable: list[tuple[str, str]] = []
        for event in session.events.op_events:
            if event.kind != "op" or event.label_raw in plan.hook_generated_labels:
                continue
            context = getattr(event, "record_context", None)
            if context is None:
                continue
            try:
                matched = bool(site.selector(context))
            except Exception:
                matched = False
            if matched and event.label_raw not in plan.presented_labels:
                unreachable.append((str(event.function.func_name), event.label_raw))
        if unreachable:
            summary = ", ".join(f"{op_type} ({label})" for op_type, label in unreachable[:8])
            raise TFInterventionSiteUnreachableError(
                f"tf intervention site {site.plan_id} matched captured ops that never "
                f"passed through the writable wrap layer: {summary}. TensorFlow op "
                "callbacks are read-only, so only calls through the curated python "
                "entry points (tf.nn/tf.math core) or module boundaries can be "
                "substituted; refusing instead of silently not intervening."
            )
    # Zero-match disclosure (the preview half of the torch-side fix): the
    # reachability refusal above covers matched-but-unpresented ops only, so a
    # selector matching NOTHING -- or a module site whose boundary the exit
    # hook never reached (module sites get no reachability audit by
    # construction) -- produced a trace byte-identical to plain capture with
    # no signal. ``fired_site_labels`` is the accounting the fire path already
    # writes; a planned site with zero fires warns, mirroring the torch
    # forward/backward and paddle disclosures.
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


def _producer_label(session: Any, tensor: Any) -> str | None:
    """Return the raw label of the op event that produced ``tensor``.

    Parameters
    ----------
    session
        Active capture session.
    tensor
        Candidate TensorFlow tensor.

    Returns
    -------
    str | None
        Producer raw label when the callback recorded the tensor.
    """

    ref = getattr(tensor, "ref", None)
    if not callable(ref):
        return None
    try:
        return session.producer_by_ref.get(ref())
    except TypeError:
        return None


def _map_tensor_leaves(value: Any, tf: Any, fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to every tensor leaf of a wrapped-call return value.

    Parameters
    ----------
    value
        Wrapped-call return container or tensor.
    tf
        Imported TensorFlow module.
    fn
        Leaf transformation.

    Returns
    -------
    Any
        Value with transformed tensor leaves.
    """

    if _is_tf_tensor_like(value, tf):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tensor_leaves(item, tf, fn) for item in value)
    if isinstance(value, list):
        return [_map_tensor_leaves(item, tf, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensor_leaves(item, tf, fn) for key, item in value.items()}
    return value


def _is_tf_tensor_like(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a TensorFlow tensor or variable.

    Parameters
    ----------
    value
        Candidate value.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for tensors and variables.
    """

    tensor_type = getattr(tf, "Tensor", None)
    variable_type = getattr(tf, "Variable", None)
    return bool(
        (tensor_type is not None and isinstance(value, tensor_type))
        or (variable_type is not None and isinstance(value, variable_type))
    )


__all__ = [
    "TFInterventionPlan",
    "TFInterventionSite",
    "TFInterventionSiteUnreachableError",
    "apply_tf_module_intervention",
    "audit_tf_site_reachability",
    "normalize_tf_interventions",
    "tf_intervention_wrap",
]
