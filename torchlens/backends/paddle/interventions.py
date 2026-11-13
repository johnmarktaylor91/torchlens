"""Live intervention runtime for the Paddle backend.

Paddle dygraph is eager: the capture wrapper holds the concrete output of every
wrapped call before the caller sees it, so ``trace(intervene=...)`` and
``trace(stop_after=...)`` apply live with real predicate-time values. The runtime is
deliberately fail-closed: unsupported helper specs, non-forward directions,
isolation requests, and non-tensor replacements refuse typed instead of
silently not intervening.

This module must import without Paddle installed; the Paddle runtime is only
touched through the backend instance handed in at apply time.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ... import _state
from ..._capture._stop_after_policy import settle_unfired_stop_after
from ..._ir.intervention import FireResult
from ..._ir.op_record import amend_module_exit_intervention
from ..._ir.predicate import RecordContext
from ..._ir.selector_eval import (
    module_address_matches,
    module_union_addresses,
    selector_contains_kind,
)
from ..._selection.selectors import BaseSelector
from ...intervention._predicates import as_intervention_decision
from ...intervention._types import Edit, FireRecord, InterventionDecision, TargetSpec
from ...recording._exceptions import PredicateError
from .._protocols.paddle import (
    PADDLE_REFUSAL_BACKWARD,
    PADDLE_REFUSAL_OPTION,
    PADDLE_REFUSAL_PREDICATE,
)
from ..registry import BackendUnsupportedError
from ._edit_helpers import _replace_leaves, resolve_paddle_helper
from ._grad_sites import grad_site_neutral

if TYPE_CHECKING:
    from ...intervention._grad_site_neutral import GradSiteCall

_PADDLE_SUPPORTED_HELPER_NAMES = (
    "add",
    "clamp",
    "mean_ablate",
    "replace_with",
    "scale",
    "zero_ablate",
)
"""Builtin helper specs with a Paddle-native adapter, sorted (``_edit_helpers`` applies them)."""


@dataclass(frozen=True)
class PaddleInterventionCapture:
    """Replay-oracle sidecar facts for one intervened Paddle wrapped call.

    Parameters
    ----------
    intervened_paths
        Output container paths whose tensors were replaced by the hook.
    raw_first_output
        Pre-hook tensor at the capture's first output path when that path was
        replaced, otherwise ``None``. The validation oracle replays the
        original callable against THIS value; the recorded public payload is
        the replacement that actually flowed downstream.
    helper_name
        Builtin helper name when the hook came from a helper spec.
    func_call_id
        Wrapper call ordinal shared with the emitted op events.
    site_labels
        Raw labels of the replaced output sites.
    """

    intervened_paths: tuple[tuple[Any, ...], ...]
    raw_first_output: Any | None
    helper_name: str | None
    func_call_id: int
    site_labels: tuple[str, ...]


class PaddleInterventionRuntime:
    """Session-scoped live ``intervene=`` / ``halt=`` state for one capture."""

    def __init__(
        self,
        backend: Any,
        *,
        intervene: Callable[[Any], Any] | None,
        halt: Callable[[Any], Any] | None,
    ) -> None:
        """Validate and store the public predicates for one Paddle capture.

        Parameters
        ----------
        backend
            Active ``PaddleBackend`` (owns the imported paddle module).
        intervene
            Public ``trace(intervene=...)`` predicate, or ``None``.
        halt
            Public ``trace(stop_after=...)`` predicate, or ``None``.
        """

        if intervene is not None and not callable(intervene):
            raise BackendUnsupportedError(
                "paddle backend trace(intervene=...) requires a callable predicate "
                "such as tl.when(condition, action); got "
                f"{type(intervene).__name__!s}.",
                code=PADDLE_REFUSAL_OPTION,
                option="intervene",
            )
        if halt is not None and not callable(halt):
            raise BackendUnsupportedError(
                "paddle backend trace(stop_after=...) requires a callable predicate "
                f"returning bool; got {type(halt).__name__!s}.",
                code=PADDLE_REFUSAL_OPTION,
                option="halt",
            )
        # tl.grad(site) clauses cannot fire during the forward: they are split
        # off here and armed on the finished trace (``arm_gradient_sites``).
        intervene, self.grad_rules = _split_gradient_rules(intervene)
        _validate_module_selector(getattr(intervene, "selector", None))
        self._backend = backend
        self.intervene = intervene
        self.halt = halt
        self.fire_count = 0
        # The effective edited program, for derived-gradient replay
        # (``_edit_replay``): replacing hooks by wrapped-call ordinal and output
        # path, and the ordinal of the call whose output halted the forward.
        self.fired_edits: dict[int, tuple[tuple[tuple[Any, ...], Callable[[Any], Any]], ...]] = {}
        self.halt_call_id: int | None = None

    def arm_gradient_sites(self, trace: Any) -> None:
        """Arm this capture's ``tl.grad`` clauses on the finished trace (``_grad_sites``).

        Parameters
        ----------
        trace
            Finished Paddle trace with materialized ops.
        """

        if self.grad_rules:
            _arm_paddle_grad_sites(trace, self.grad_rules, self._backend.paddle)

    def evaluate_intervene(self, ctx: RecordContext) -> InterventionDecision | None:
        """Return the normalized intervention decision for one output site.

        Parameters
        ----------
        ctx
            Predicate context built for the candidate output tensor.

        Returns
        -------
        InterventionDecision | None
            Normalized decision when the predicate matched.
        """

        if self.intervene is None or selector_contains_kind(
            getattr(self.intervene, "selector", None), "module"
        ):
            return None
        result = self.intervene(ctx)
        try:
            decision = as_intervention_decision(result)
        except TypeError as exc:
            raise PredicateError(
                "intervene predicate must return InterventionDecision, Edit, "
                "callable, or None. "
                "Remedy: return one of those values from the intervene predicate.",
                ctx=ctx,
                result=result,
                code="predicate_return_invalid",
            ) from exc
        if decision is None or decision.hook is None:
            return None
        return decision

    def apply(
        self,
        ctx: RecordContext,
        tensor: Any,
        decision: InterventionDecision,
        container_path: tuple[Any, ...],
    ) -> tuple[Any, FireResult]:
        """Apply one normalized decision to one concrete output tensor.

        The hook body runs under ``pause_logging()``: intervention internals
        are not part of the captured graph. The intervened op keeps its real
        identity and records the replacement payload plus fire metadata.

        Parameters
        ----------
        ctx
            Predicate context for the site.
        tensor
            Concrete pre-hook Paddle output tensor.
        decision
            Normalized decision returned by :meth:`evaluate_intervene`.
        container_path
            Output container path of ``tensor`` in the wrapped call's output.

        Returns
        -------
        tuple[Any, FireResult]
            Replacement tensor (possibly the original object when the hook
            returned it unchanged) and the normalized fire result.
        """

        self._reject_unsupported_decision(decision)
        helper_spec = decision.hook if isinstance(decision.hook, Edit) else None
        hook_fn = self._resolve_hook(decision)
        with _state.pause_logging():
            replacement = hook_fn(tensor)
        if not self._backend.is_tensor(replacement):
            raise BackendUnsupportedError(
                "paddle backend trace(intervene=...) hooks must return a "
                f"paddle.Tensor; hook for site {ctx.label_raw!r} returned "
                f"{type(replacement).__name__!s}.",
                code=PADDLE_REFUSAL_PREDICATE,
            )
        replaced = replacement is not tensor
        self.fire_count += 1
        if replaced:
            self._record_fired_edit(ctx.func_call_id, container_path, hook_fn)
        fire_record = _fire_record(ctx, container_path, helper_spec, replaced)
        fire_result = FireResult(
            plan_id=f"paddle-intervene-{self.fire_count}",
            site_label=ctx.label_raw,
            fired_at_capture_index=ctx.raw_index if ctx.raw_index is not None else -1,
            pre_hook_shape=self._backend._shape(tensor),
            post_hook_shape=self._backend._shape(replacement),
            pre_hook_dtype=self._backend._dtype(tensor),
            post_hook_dtype=self._backend._dtype(replacement),
            replaced=replaced,
            fire_record=fire_record,
        )
        return replacement, fire_result

    def _record_fired_edit(
        self, call_id: int | None, path: tuple[Any, ...], hook_fn: Callable[[Any], Any]
    ) -> None:
        """Remember a replacing hook by call ordinal and path for derived-gradient replay."""

        if call_id is not None:
            self.fired_edits[call_id] = (*self.fired_edits.get(call_id, ()), (path, hook_fn))

    def evaluate_halt(self, ctx: RecordContext) -> bool:
        """Return whether the halt predicate fired for one output site.

        Parameters
        ----------
        ctx
            Predicate context for the site.

        Returns
        -------
        bool
            True when capture should stop at this frontier.
        """

        if self.halt is None:
            return False
        result = self.halt(ctx)
        if not isinstance(result, bool):
            raise PredicateError(
                "halt predicate must return bool. "
                "Remedy: return True or False from the halt predicate.",
                ctx=ctx,
                result=result,
                code="predicate_return_invalid",
            )
        if result:
            self.halt_call_id = ctx.func_call_id
        return result

    def record_fired_spec(
        self, trace: Any, target_label: str, decision: InterventionDecision
    ) -> None:
        """Persist a fired decision on the trace-owned intervention spec.

        Mirrors the torch predicate path: the armed spec is trace-level
        evidence that a user actually registered interventions, which the
        validation carve-out requires before trusting per-op fire stamps.

        Parameters
        ----------
        trace
            Active capture trace.
        target_label
            Raw label of the fired site.
        decision
            Normalized fired decision.
        """

        spec = trace._ensure_intervention_spec()
        target = TargetSpec("label", target_label)
        frozen_target = target.freeze()
        if all(existing.freeze() != frozen_target for existing in spec.targets):
            spec.targets.append(target)
        helper_spec = decision.hook if isinstance(decision.hook, Edit) else None
        spec.add_hook(
            target,
            decision.hook,
            helper=helper_spec,
            metadata={"created_by": "intervene_predicate", "signal": "forward"},
        )

    def settle_unfired(self, trace: Any) -> None:
        """Disclose an unfired intervene selector; refuse an unfired stop_after like torch."""

        self.warn_if_zero_matches()
        # A selector is the selector-shaped site; any other callable is exploratory.
        settle_unfired_stop_after(
            trace, self.halt, selector_shaped=isinstance(self.halt, BaseSelector)
        )

    def warn_if_zero_matches(self) -> None:
        """Warn when a selector-conditioned intervene predicate never fired."""

        selector = getattr(self.intervene, "selector", None)
        if isinstance(selector, BaseSelector) and self.fire_count == 0:
            warnings.warn(
                f"Capture-time intervention selector {selector!r} matched zero sites; "
                "no intervention fired.",
                UserWarning,
                stacklevel=3,
            )

    def _reject_unsupported_decision(self, decision: InterventionDecision) -> None:
        """Refuse decision shapes the Paddle preview cannot honor.

        Parameters
        ----------
        decision
            Normalized intervention decision.
        """

        if decision.isolate:
            raise BackendUnsupportedError(
                "the paddle backend does not support isolate=True intervention "
                "decisions; backend isolation is a torch capture feature.",
                code=PADDLE_REFUSAL_OPTION,
                option="isolate",
            )

    def _resolve_hook(self, decision: InterventionDecision) -> Callable[[Any], Any]:
        """Return a Paddle-native callable for one decision hook.

        Parameters
        ----------
        decision
            Normalized intervention decision carrying a helper spec or callable.

        Returns
        -------
        Callable[[Any], Any]
            Function mapping the pre-hook tensor to its replacement.
        """

        hook = decision.hook
        if isinstance(hook, Edit):
            return self._resolve_helper_hook(hook)
        if callable(hook):
            return hook
        raise BackendUnsupportedError(
            "paddle backend trace(intervene=...) decisions must carry a builtin "
            f"helper spec or a callable hook; got {type(hook).__name__!s}.",
            code=PADDLE_REFUSAL_OPTION,
            option="intervene",
        )

    def _resolve_helper_hook(self, helper: Edit) -> Callable[[Any], Any]:
        """Return the Paddle adapter for one builtin helper spec (``_edit_helpers``)."""

        return resolve_paddle_helper(helper, self._backend.paddle)


__all__ = [
    "PaddleInterventionCapture",
    "PaddleInterventionRuntime",
]


def _fire_record(
    ctx: RecordContext, container_path: tuple[Any, ...], helper_spec: Edit | None, replaced: bool
) -> FireRecord:
    """Fire metadata of one applied Paddle intervention."""

    return FireRecord(
        target_label=ctx.label_raw,
        call_label=ctx.label_raw,
        func_call_id=ctx.func_call_id,
        container_path=container_path,
        engine="paddle_wrapper",
        edit=helper_spec,
        site_label=ctx.label_raw,
        timing="post",
        signal="forward",
        edit_name=helper_spec.name if helper_spec is not None else None,
        timestamp=time.time(),
        call_index=ctx.raw_index,
        replaced=replaced,
    )


def _validate_module_selector(selector: Any) -> None:
    """Refuse mixed module trees before any forward executes."""

    if selector_contains_kind(selector, "module") and module_union_addresses(selector) is None:
        raise BackendUnsupportedError(
            "paddle backend module interventions require tl.module(...) or a union "
            "of module selectors; mixed or negated module selectors are unsupported.",
            code=PADDLE_REFUSAL_OPTION,
            option="intervene",
        )


def _apply_module_exit(trace: Any, frame: Any, output: Any) -> Any:
    """Apply one selected module call to all its output tensor leaves."""

    runtime = trace._paddle_intervention_runtime
    if runtime is None:
        return output
    predicate = runtime.intervene
    addresses = module_union_addresses(getattr(predicate, "selector", None))
    if not addresses or not any(
        module_address_matches((frame.address, frame.call_index), address) for address in addresses
    ):
        return output
    backend = runtime._backend
    by_id = {}
    replacements = {}
    replay = getattr(trace, "_paddle_edit_replay_tap", None)
    for path, tensor in backend._iter_tensors_with_paths(output):
        if id(tensor) not in by_id:
            # Derived-gradient replay re-runs the boundary call through the replay
            # tap, which applies the edit fired at this call ordinal during capture.
            by_id[id(tensor)] = (
                _emit_module_leaf(trace, runtime, tensor)
                if replay is None
                else replay.call(_copy_leaf, "interventionreplacement", (tensor,), {})
            )
        replacements[path] = by_id[id(tensor)]
    return _replace_leaves(output, replacements)


@dataclass(frozen=True)
class _ReplacementBoundaryPredicate:
    """Fire one module edit's decision at its own replacement boundary op only.

    Equivalent here to ``tl.when(tl.func("interventionreplacement"), decision)``:
    the Paddle intervention path reads only the decision's hook and isolate flag,
    and a func selector matches the captured function name or the layer type.
    """

    decision: InterventionDecision | None

    def __call__(self, ctx: Any) -> InterventionDecision | None:
        """Return the decision for the boundary op and ``None`` for any other op."""

        names = (getattr(ctx, "func_name", None), getattr(ctx, "layer_type", None))
        return self.decision if "interventionreplacement" in names else None


def _copy_leaf(tensor: Any) -> Any:
    """Copy the raw producer value without changing its label or existing consumers."""

    return tensor.clone()


def _emit_module_leaf(trace: Any, runtime: Any, tensor: Any) -> Any:
    """Record a declared replacement through Paddle's existing replay machinery."""

    backend = runtime._backend
    label = backend.tensor_store.get_label(tensor)
    predicate = runtime.intervene
    runtime.intervene = _ReplacementBoundaryPredicate(predicate.decision)
    try:
        with _state.pause_logging():
            raw = _copy_leaf(tensor)
            replacement = backend.emit_paddle_operation(
                trace,
                "interventionreplacement",
                _copy_leaf,
                (tensor,),
                {},
                raw,
                module_stack=tuple(trace._paddle_module_stack),
            )
    finally:
        runtime.intervene = predicate
    events = trace.capture_events
    boundary = events.op_event_by_label_raw[backend.tensor_store.get_label(replacement)]
    event = events.op_event_by_label_raw.get(label)
    if event is not None:
        fires = tuple(
            replace(
                fire,
                site_label=label,
                fire_record=replace(
                    fire.fire_record, target_label=label, func_call_id=event.func_call_id
                ),
            )
            for fire in boundary.fire_results
        )
        events.append_amendment(
            amend_module_exit_intervention(
                event.seq,
                label,
                intervention_fired=True,
                is_edited=boundary.is_edited,
                fire_results=(*event.fire_results, *fires),
            )
        )
    return replacement


GradRule = tuple[Any, Any]
"""One gradient clause: ``(gradient site, edit)``."""


def _backward_refusal(message: str) -> BackendUnsupportedError:
    """Return the typed refusal for a gradient address Paddle cannot arm."""

    return BackendUnsupportedError(message, code=PADDLE_REFUSAL_BACKWARD)


def _intervention_hooks() -> Any:
    """The site-branch helpers of ``torchlens.intervention.hooks`` (imported on first use)."""

    from ...intervention import hooks  # deferred: L4 vocabulary, used only with tl.grad clauses

    return hooks


def _split_rule(rule: Any) -> tuple[list[Any], list[GradRule]]:
    """Split one rule into its live activation rule (if any) and its gradient clauses."""

    live: list[Any] = []
    grads: list[GradRule] = []
    for universe, branch in _intervention_hooks().split_site_branches(rule.target):
        if universe == "gradient":
            grads.append((branch, rule.edit))
        elif universe == "activation":
            live.append(rule if branch is rule.target else replace(rule, target=branch))
        else:
            raise _backward_refusal(
                f"the paddle backend cannot arm {branch!r}: autograd-node selectors "
                "(tl.grad_fn, tl.grad_input, ...) address recorded backward nodes, which "
                "paddle does not record. tl.grad(site) addresses are supported."
            )
    return live, grads


def _split_gradient_rules(intervene: Any) -> tuple[Any, tuple[GradRule, ...]]:
    """Split ``intervene`` into its live forward predicate and its gradient clauses.

    Parameters
    ----------
    intervene
        Public ``trace(intervene=...)`` value: a ``tl.when`` recipe or another
        callable predicate.

    Returns
    -------
    tuple[Any, tuple[GradRule, ...]]
        The predicate that runs live during the forward (``None`` when every
        clause addresses a gradient) and the ``(site, edit)`` gradient clauses.
    """

    rules = getattr(intervene, "rules", None)
    if intervene is None or rules is None:
        return intervene, ()
    live: list[Any] = []
    grads: list[GradRule] = []
    for rule in rules:
        rule_live, rule_grads = _split_rule(rule)
        live.extend(rule_live)
        grads.extend(rule_grads)
    if not grads:
        return intervene, ()
    if not live:
        return None, tuple(grads)
    return replace(intervene, rules=tuple(live)), tuple(grads)


def _gradient_edit_function(edit: Any, paddle: Any) -> Callable[[Any], Any]:
    """Return the Paddle-native function one gradient clause applies to a gradient.

    Parameters
    ----------
    edit
        The clause's action: a built-in helper spec, a callable taking the
        gradient, or an ``InterventionDecision`` carrying either.
    paddle
        Imported ``paddle`` module.

    Returns
    -------
    Callable[[Any], Any]
        Function mapping the gradient arriving at the site to its replacement.
    """

    try:
        decision = as_intervention_decision(edit)
    except TypeError as exc:
        raise BackendUnsupportedError(
            f"paddle tl.grad(...) edits must be a builtin helper or a callable; got "
            f"{type(edit).__name__!s}.",
            code=PADDLE_REFUSAL_OPTION,
            option="intervene",
        ) from exc
    hook = None if decision is None else decision.hook
    if isinstance(hook, Edit):
        return resolve_paddle_helper(hook, paddle)
    if callable(hook):
        return hook
    raise BackendUnsupportedError(
        f"paddle tl.grad(...) edits must be a builtin helper or a callable; got "
        f"{type(hook).__name__!s}.",
        code=PADDLE_REFUSAL_OPTION,
        option="intervene",
    )


def _edit_name(edit: Any) -> str:
    """Display name of a gradient edit for fire records."""

    name = getattr(edit, "name", None)
    if isinstance(name, str):
        return name
    return str(getattr(edit, "__name__", type(edit).__name__))


def _resolve_raw_labels(trace: Any, selector: Any) -> tuple[str, ...]:
    """Raw graph labels of the ops ``selector`` names on a finished Paddle trace.

    Parameters
    ----------
    trace
        Finished Paddle trace.
    selector
        Forward selector inside a ``tl.grad(...)`` address.

    Returns
    -------
    tuple[str, ...]
        Raw (capture-identity) labels in trace order, without repeats.
    """

    sites = trace._resolve_sites(selector, strict=False, max_fanout=len(trace._layer_list) + 1)
    labels = (getattr(site, "_label_raw", None) for site in sites)
    return tuple(dict.fromkeys(label for label in labels if isinstance(label, str)))


def _arm_paddle_grad_sites(trace: Any, grad_rules: Sequence[GradRule], paddle: Any) -> None:
    """Resolve each gradient clause on the finished trace and arm it there.

    Parameters
    ----------
    trace
        Finished Paddle trace (ops materialized).
    grad_rules
        ``(site, edit)`` gradient clauses from :func:`_split_gradient_rules`.
    paddle
        Imported ``paddle`` module.
    """

    neutral = grad_site_neutral()
    for site, edit in grad_rules:
        function = _gradient_edit_function(edit, paddle)

        def _apply(grad: Any, call: GradSiteCall, function: Any = function) -> Any:
            """Apply the clause's Paddle function to one gradient."""

            del call
            return function(grad)

        edits = neutral.build_grad_site_edits(
            site,
            _apply,
            lambda inner: _resolve_raw_labels(trace, inner),
            operands=_intervention_hooks()._target_operands,
            name=_edit_name(edit),
        )
        neutral.arm_grad_sites(trace, edits)
