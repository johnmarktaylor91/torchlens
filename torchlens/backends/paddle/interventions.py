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
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

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
    PADDLE_REFUSAL_FOREIGN,
    PADDLE_REFUSAL_OPTION,
    PADDLE_REFUSAL_PREDICATE,
)
from ..registry import BackendUnsupportedError

_PADDLE_SUPPORTED_HELPER_NAMES = ("add", "replace_with", "scale", "zero_ablate")
"""Builtin helper specs with a Paddle-native adapter, sorted."""


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
        from ...intervention.hooks import split_site_branches

        selector = getattr(intervene, "selector", None)
        _validate_module_selector(selector)
        if selector is not None and any(
            universe != "activation" for universe, _branch in split_site_branches(selector)
        ):
            raise BackendUnsupportedError(
                "the paddle backend supports forward interventions only; the site "
                f"{selector!r} addresses a gradient, which needs true backward capture "
                "that paddle does not declare. Use the PyTorch backend for gradient edits.",
                code=PADDLE_REFUSAL_BACKWARD,
            )
        self._backend = backend
        self.intervene = intervene
        self.halt = halt
        self.fire_count = 0

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
        """Return the Paddle adapter for one builtin helper spec.

        The builtin helper factories build torch hooks, so each supported
        helper is re-expressed with Paddle ops from its portable args/kwargs.
        Unsupported helpers refuse typed rather than silently not intervening.

        Parameters
        ----------
        helper
            Builtin helper spec from ``tl.zero_ablate()`` and friends.

        Returns
        -------
        Callable[[Any], Any]
            Paddle-native replacement function.
        """

        paddle = self._backend.paddle
        name = helper.name
        args = tuple(helper.args)
        if name == "zero_ablate":
            return lambda out: paddle.zeros_like(out)
        if name == "scale":
            factor = args[0]
            return lambda out: out * factor
        if name == "add":
            delta = args[0]
            self._reject_foreign_tensor(delta, helper_name=name)
            return lambda out: out + delta
        if name == "replace_with":
            value = args[0]
            if not callable(value):
                self._reject_foreign_tensor(value, helper_name=name)

            def _replace(out: Any) -> Any:
                """Return the replacement, refusing a foreign tensor and casting to ``out``'s dtype."""

                replacement = value() if callable(value) else value
                self._reject_foreign_tensor(replacement, helper_name=name)
                if not self._backend.is_tensor(replacement):
                    return replacement
                return replacement.astype(out.dtype)

            return _replace
        supported = ", ".join(_PADDLE_SUPPORTED_HELPER_NAMES)
        raise BackendUnsupportedError(
            f"the paddle backend has no adapter for intervention helper "
            f"{name!r}; supported builtin helpers: {supported}. Pass a callable "
            "hook operating on paddle.Tensor values, or use the PyTorch backend.",
            code=PADDLE_REFUSAL_OPTION,
            option="intervene",
        )

    def _reject_foreign_tensor(self, value: Any, *, helper_name: str) -> None:
        """Refuse torch-tensor helper arguments on the Paddle backend.

        Parameters
        ----------
        value
            Helper argument value.
        helper_name
            Helper name for the diagnostic.
        """

        module_root = type(value).__module__.split(".", maxsplit=1)[0]
        if module_root == "torch":
            raise BackendUnsupportedError(
                f"paddle backend intervention helper {helper_name!r} received a "
                "torch.Tensor argument; pass a paddle.Tensor or Python scalar.",
                code=PADDLE_REFUSAL_FOREIGN,
            )


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
    for path, tensor in backend._iter_tensors_with_paths(output):
        if id(tensor) not in by_id:
            by_id[id(tensor)] = _emit_module_leaf(trace, runtime, tensor)
        replacements[path] = by_id[id(tensor)]
    return _replace_leaves(output, replacements)


def _replace_leaves(output: Any, replacements: dict[Any, Any], path: tuple[Any, ...] = ()) -> Any:
    """Rebuild the supported output containers with edited leaves."""

    if path in replacements:
        return replacements[path]
    if isinstance(output, tuple):
        return tuple(
            _replace_leaves(item, replacements, (*path, i)) for i, item in enumerate(output)
        )
    if isinstance(output, list):
        return [_replace_leaves(item, replacements, (*path, i)) for i, item in enumerate(output)]
    if isinstance(output, dict):
        return {
            key: _replace_leaves(item, replacements, (*path, key)) for key, item in output.items()
        }
    return output


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
