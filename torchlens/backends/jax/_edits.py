"""Capture-time ``trace(intervene=...)`` for the JAX backend.

The interpreter runs every primitive eagerly on concrete arrays, so a predicate
sees the same facts torch's capture path gives it: the raw label the event will
carry, the primitive (``tl.func``), the decoded ``named_scope`` module stack
(``tl.in_module``), and concrete output values (shape, dtype, scalar booleans).
A matched equation's output goes through the decision's hook before it is
written back, so every downstream equation computes from the edited value.

Labels are simulated with the same counters :class:`CaptureEvents` uses at
emission (inputs, then closure constants, then one event per capture), so the
label a predicate sees is the label the event receives.

Replay validation stays strict: an edited capture keeps its appliers, the
validator replays the primitive and re-applies them, and the result must equal
the saved (edited) output. Parent sensitivity is checked on the raw replay.
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

from ..._ir.events import InterventionAppliedEvent
from ..._ir.intervention import FireResult
from ..._ir.refs import ReservedLabel
from ..._ir.selector_eval import selector_contains_kind
from ..._selection.selectors import BaseSelector
from ...intervention._predicates import as_intervention_decision
from ...intervention._types import Edit, FireRecord, InterventionDecision, TargetSpec
from ...intervention.hooks import make_hook_context, normalize_hook, split_site_branches
from ...recording._exceptions import PredicateError
from ..registry import BackendUnsupportedError
from ._edit_helpers import checked_replacement, resolve_helper_applier
from ._module_frames import jax_event_module_frames
from ._source_lines import EquationFrames
from .jaxpr import JaxEquationCapture

_OPTION = "trace_option_unsupported"
_ALIAS_RETRY_KINDS = ("label", "contains", "regex")
_REGION_LAYER_TYPES = {"region": "jax_region", "region_output": "jax_region_out"}


def _refuse(message: str, *, option: str = "intervene", code: str = _OPTION) -> Exception:
    """Return a typed JAX intervention refusal."""

    if code != _OPTION:
        return BackendUnsupportedError(message, code=code)
    return BackendUnsupportedError(message, code=code, option=option)


@dataclass(frozen=True)
class JaxEditApplier:
    """One resolved edit: a stable identity and the output-to-replacement function."""

    identity: str
    apply: Callable[[Any], Any]


@dataclass(frozen=True)
class JaxEditedEquationCapture(JaxEquationCapture):
    """A primitive capture whose single output an intervention replaced."""

    edit_appliers: tuple[JaxEditApplier, ...] = ()


@dataclass(frozen=True)
class _Fire:
    """One fired site: its fire result, applier and decision."""

    result: FireResult
    applier: JaxEditApplier
    decision: InterventionDecision


def _validate_intervene(intervene: Any, grad_options: Any) -> None:
    """Refuse ``intervene`` shapes the JAX capture path cannot honor."""

    if not callable(intervene):
        raise _refuse(
            "JAX backend trace(intervene=...) requires a callable predicate such as "
            f"tl.when(condition, action); got {type(intervene).__name__}."
        )
    if grad_options is not None:
        raise _refuse(
            "JAX backend trace(intervene=...) cannot be combined with grad_options: "
            "derived gradients differentiate the unedited jaxpr.",
            code="intervention_with_derived_grads_unsupported",
        )
    selector = getattr(intervene, "selector", None)
    if not isinstance(selector, BaseSelector):
        return
    if any(universe != "activation" for universe, _ in split_site_branches(selector)):
        raise _refuse(
            f"JAX backend supports forward interventions only; the site {selector!r} "
            "addresses a gradient, which needs backward capture. Use the PyTorch backend.",
            code="backward_capture_unsupported",
        )
    if selector_contains_kind(selector, "module"):
        raise _refuse(
            "JAX backend trace(intervene=...) edits equation outputs; tl.module(...) "
            "addresses a module's output boundary, which the jaxpr does not carry. "
            "Use tl.in_module(...) with tl.func(...) or tl.label(...)."
        )
    for kind in ("followed_by", "preceded_by"):
        if selector_contains_kind(selector, kind):
            raise _refuse(
                f"JAX backend trace(intervene=...) does not support {kind} selectors, "
                "which need later or earlier sites at fire time.",
                code="followed_by_unsupported",
            )


class JaxEditRuntime:
    """Predicate evaluation and edit application for one JAX capture."""

    def __init__(self, intervene: Any, grad_options: Any) -> None:
        """Validate and store the public ``intervene`` predicate."""

        _validate_intervene(intervene, grad_options)
        self.intervene = intervene
        self.fires: list[_Fire] = []

    def decide(self, ctx: Any) -> InterventionDecision | None:
        """Return the normalized decision for one site, retrying the short label alias."""

        decision = self._evaluate(ctx)
        selector = getattr(self.intervene, "selector", None)
        if decision is not None or not isinstance(selector, BaseSelector):
            return decision
        if not any(selector_contains_kind(selector, kind) for kind in _ALIAS_RETRY_KINDS):
            return None
        return self._evaluate(replace(ctx, label=f"{ctx.layer_type}_{ctx.type_index}"))

    def _evaluate(self, ctx: Any) -> InterventionDecision | None:
        """Call the predicate once and normalize its result."""

        result = self.intervene(ctx)
        try:
            decision = as_intervention_decision(result)
        except TypeError as exc:
            raise PredicateError(
                "intervene predicate must return InterventionDecision, Edit, callable, "
                "or None. Remedy: return one of those values from the intervene predicate.",
                ctx=ctx,
                result=result,
                code="predicate_return_invalid",
            ) from exc
        if decision is None or decision.hook is None:
            return None
        return decision

    def fire(self, ctx: Any, outputs: tuple[Any, ...]) -> tuple[tuple[Any, ...], _Fire | None]:
        """Apply the matched decision (if any) to one equation's outputs."""

        decision = self.decide(ctx)
        if decision is None:
            return outputs, None
        applier = _resolve_applier(decision)
        if len(outputs) != 1:
            raise _refuse(
                f"JAX backend trace(intervene=...) matched {ctx.label_raw!r}, a "
                f"{len(outputs)}-output equation; only single-output equations are editable."
            )
        original = outputs[0]
        replacement = checked_replacement(ctx.label_raw, original, applier.apply(original))
        fire = _Fire(_fire_result(ctx, decision, original, replacement, self), applier, decision)
        self.fires.append(fire)
        return (replacement,), fire


def _resolve_applier(decision: InterventionDecision) -> JaxEditApplier:
    """Resolve one decision to a JAX-native applier, refusing torch-only semantics."""

    if decision.keep_grad or decision.isolate:
        raise _refuse(
            "JAX backend interventions do not support keep_grad or isolate; both are "
            "torch-autograd semantics with no jaxpr equivalent.",
            option="isolate" if decision.isolate else "keep_grad",
        )
    if decision.action not in {"add_hook", "transform"}:
        raise _refuse(
            f"JAX backend interventions do not support action {decision.action!r}; "
            "supported actions are hook helpers and callable hooks."
        )
    hook = decision.hook
    if isinstance(hook, Edit):
        return JaxEditApplier(f"helper:{hook.name}", resolve_helper_applier(hook))
    if not callable(hook):
        raise _refuse(
            "JAX backend intervention decisions must carry a builtin helper or a "
            f"callable hook; got {type(hook).__name__}."
        )
    normalized = normalize_hook(hook)
    identity = f"callable:{getattr(hook, '__qualname__', type(hook).__name__)}"

    def _apply(out: Any) -> Any:
        """Invoke the user hook with a post-forward hook context."""

        return normalized(out, ctx=make_hook_context(name=identity))

    return JaxEditApplier(identity, _apply)


def _fire_result(
    ctx: Any, decision: InterventionDecision, original: Any, replacement: Any, runtime: Any
) -> FireResult:
    """Build the fire record and fire result of one applied edit."""

    helper = decision.hook if isinstance(decision.hook, Edit) else None
    replaced = replacement is not original
    record = FireRecord(
        target_label=ctx.label_raw,
        call_label=ctx.label_raw,
        func_call_id=ctx.func_call_id,
        container_path=(),
        engine="jax_interpreter",
        edit=helper,
        site_label=ctx.label_raw,
        timing="post",
        signal="forward",
        edit_name=helper.name if helper is not None else None,
        timestamp=time.time(),
        call_index=ctx.raw_index,
        replaced=replaced,
    )
    return FireResult(
        plan_id=f"jax-intervene-{len(runtime.fires) + 1}",
        site_label=ctx.label_raw,
        fired_at_capture_index=ctx.raw_index,
        pre_hook_shape=tuple(original.shape),
        post_hook_shape=tuple(replacement.shape),
        pre_hook_dtype=str(original.dtype),
        post_hook_dtype=str(replacement.dtype),
        replaced=replaced,
        fire_record=record,
    )


class JaxEquationSession:
    """Per-capture equation hook: simulated labels, edits and source frames."""

    def __init__(
        self,
        backend: Any,
        trace: Any,
        runtime: JaxEditRuntime | None,
        frames: EquationFrames | None,
        num_sources: int,
    ) -> None:
        """Start label simulation after ``num_sources`` input and constant events."""

        self.backend, self.trace, self.runtime, self.frames = backend, trace, runtime, frames
        self._raw_counter = num_sources
        self._type_counts: Counter[str] = Counter()
        self._seen = 0
        self.fires_by_index: dict[int, _Fire] = {}
        self.fires_by_label: dict[str, FireResult] = {}
        self.context_by_label: dict[str, tuple[Any, ...]] = {}

    def _reserve(self, layer_type: str) -> ReservedLabel:
        """Advance the simulated counters exactly like ``reserve_label_block``."""

        self._raw_counter += 1
        self._type_counts[layer_type] += 1
        type_index = self._type_counts[layer_type]
        label = f"{layer_type}_{type_index}_{self._raw_counter}_raw"
        return ReservedLabel(label, label, self._raw_counter, type_index, layer_type, label)

    def _catch_up(self, captures: Sequence[Any]) -> None:
        """Reserve labels for captures appended since the last primitive hook."""

        for capture in captures[self._seen :]:
            self._reserve(_REGION_LAYER_TYPES.get(capture.kind, capture.primitive))
        self._seen = len(captures)

    def hook(
        self,
        captures: Sequence[Any],
        eqn: Any,
        outputs: tuple[Any, ...],
        stacks: tuple[tuple[str, ...], tuple[tuple[str, int], ...]],
    ) -> tuple[Any, ...]:
        """Label one primitive equation, record its frames and apply any edit."""

        self._catch_up(captures)
        reserved = self._reserve(eqn.primitive.name)
        self._seen += 1
        if self.frames is not None:
            self.context_by_label[reserved.label_raw] = self.frames.locations(eqn)
        if self.runtime is None:
            return outputs
        ctx = self._context(reserved, outputs, stacks)
        outputs, fire = self.runtime.fire(ctx, outputs)
        if fire is not None:
            self.fires_by_label[reserved.label_raw] = fire.result
            if fire.result.replaced:
                self.fires_by_index[len(captures)] = fire
        return outputs

    def _context(self, reserved: ReservedLabel, outputs: tuple[Any, ...], stacks: Any) -> Any:
        """Build the predicate context with module frames and concrete value facts."""

        output = outputs[0] if len(outputs) == 1 else outputs
        ctx = self.backend._record_context(reserved, output, reserved.layer_type)
        _calls, frames = jax_event_module_frames(*stacks)
        shape = getattr(output, "shape", None)
        is_scalar_bool = shape == () and str(getattr(output, "dtype", "")) == "bool"
        return replace(
            ctx,
            module_stack=frames,
            is_scalar_bool=is_scalar_bool,
            bool_value=bool(output) if is_scalar_bool else None,
        )

    def finish(self, result: Any, output: Any) -> tuple[Any, Any]:
        """Rewrap edited captures, record fired specs and return the edited output."""

        if self.runtime is None:
            return result, output
        if not self.runtime.fires:
            _warn_zero_matches(self.runtime.intervene)
        for fire in self.runtime.fires:
            _record_fired_spec(self.trace, fire.result.site_label, fire.decision)
        if not self.fires_by_index:
            return result, output
        captures = tuple(
            _edited_capture(capture, self.fires_by_index[index].applier)
            if index in self.fires_by_index
            else capture
            for index, capture in enumerate(result.captures)
        )
        equations = tuple(c for c in captures if isinstance(c, JaxEquationCapture))
        import jax

        edited_output = jax.tree.unflatten(jax.tree.structure(output), result.outputs)
        return replace(result, captures=captures, equations=equations), edited_output

    def stamp(self, event: Any) -> Any:
        """Return ``event`` with its fire stamps and source frames."""

        context = self.context_by_label.get(event.label_raw)
        if context is not None:
            event = replace(event, function=replace(event.function, code_context=context))
        fire = self.fires_by_label.get(event.label_raw)
        if fire is None:
            return event
        return replace(
            event, intervention_fired=True, is_edited=fire.replaced, fire_results=(fire,)
        )


def _edited_capture(capture: Any, applier: JaxEditApplier) -> JaxEditedEquationCapture:
    """Return ``capture`` rewrapped with its edit applier."""

    values = {field.name: getattr(capture, field.name) for field in fields(JaxEquationCapture)}
    return JaxEditedEquationCapture(**values, edit_appliers=(applier,))


def _record_fired_spec(trace: Any, target_label: str, decision: InterventionDecision) -> None:
    """Persist a fired decision on the trace-owned intervention spec (torch parity)."""

    spec = trace._ensure_intervention_spec()
    target = TargetSpec("label", target_label)
    frozen = target.freeze()
    if all(existing.freeze() != frozen for existing in spec.targets):
        spec.targets.append(target)
    spec.add_hook(
        target,
        decision.hook,
        helper=decision.hook if isinstance(decision.hook, Edit) else None,
        metadata={"created_by": "intervene_predicate", "signal": "forward"},
    )


def _warn_zero_matches(intervene: Any) -> None:
    """Warn when a selector-conditioned predicate never fired."""

    selector = getattr(intervene, "selector", None)
    if isinstance(selector, BaseSelector):
        warnings.warn(
            f"Capture-time intervention selector {selector!r} matched zero sites; "
            "no intervention fired.",
            UserWarning,
            stacklevel=4,
        )


def open_equation_session(
    backend: Any, trace: Any, intervene: Any, capture: tuple[Any, Any, Any, Any]
) -> JaxEquationSession:
    """Open the equation session of one capture and pin it on ``backend``.

    ``capture`` is ``(args, closure_consts, grad_options, save_code_context)``.
    The session is reachable from ``backend._equation_session`` so the event
    appender can stamp events; it only stamps events of its own ``trace``.
    """

    args, consts, grad_options, save_code_context = capture
    runtime = None if intervene is None else JaxEditRuntime(intervene, grad_options)
    frames = EquationFrames.for_trace(trace, save_code_context)
    num_inputs = sum(
        1
        for _path, value in backend._tree_leaves_with_paths(tuple(args))
        if backend.is_tensor(value)
    )
    session = JaxEquationSession(backend, trace, runtime, frames, num_inputs + len(consts))
    backend._equation_session = session
    return session


def append_session_event(backend: Any, trace: Any, event: Any) -> Any:
    """Stamp ``event`` from the open session, append it, and log a live-fire edit."""

    session = getattr(backend, "_equation_session", None)
    if session is not None and session.trace is trace:
        event = session.stamp(event)
    events = trace.capture_events
    events.append(event)
    if event.is_edited:
        events.append_intervention(
            InterventionAppliedEvent(
                label_raw=event.label_raw,
                kind="replaced",
                origin="live_fire",
                timestamp=time.time(),
                run_token=events.run_nonce,
                target_seq=event.seq,
                target_func_call_id=event.func_call_id,
            )
        )
    return event


def replay_with_edits(capture: Any, op: Any, replayed: Any) -> tuple[Any, Any]:
    """Re-apply an edited capture's appliers to its raw replay.

    Returns ``(outputs to compare with the saved outputs, raw replay or None)``.
    A capture whose edit state disagrees with the op's ``is_edited`` raises,
    which the validator records as a replay mismatch.
    """

    appliers = getattr(capture, "edit_appliers", ())
    op_edited = bool(getattr(op, "is_edited", False)) and bool(getattr(op, "edits", ()))
    if bool(appliers) != op_edited:
        raise ValueError(
            f"JAX capture {capture.index} edit state disagrees with the op's edit records."
        )
    if not appliers:
        return replayed, None
    edited = replayed[0]
    for applier in appliers:
        edited = applier.apply(edited)
    return (edited, *tuple(replayed)[1:]), replayed


__all__ = [
    "JaxEditApplier",
    "JaxEditRuntime",
    "JaxEditedEquationCapture",
    "JaxEquationSession",
    "append_session_event",
    "open_equation_session",
    "replay_with_edits",
]
