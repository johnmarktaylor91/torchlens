"""Predicate-time intervention decisions and sugar."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import torch

from .types import HelperDirection, HelperSpec, InterventionDecision

InterventionPredicateDecision: TypeAlias = (
    InterventionDecision | HelperSpec | Callable[..., Any] | None
)
InterventionPredicate: TypeAlias = Callable[[Any], InterventionPredicateDecision]


def as_intervention_decision(
    action: InterventionPredicateDecision,
    *,
    direction: HelperDirection | None = None,
) -> InterventionDecision | None:
    """Normalize predicate action sugar to an intervention decision.

    Parameters
    ----------
    action:
        Helper spec, callable transform, intervention decision, or ``None``.
    direction:
        Optional signal direction override.

    Returns
    -------
    InterventionDecision | None
        Normalized active intervention decision.
    """

    if action is None:
        return None
    if isinstance(action, InterventionDecision):
        if direction is not None:
            return InterventionDecision(
                action=action.action,
                hook=action.hook,
                template_ref=action.template_ref,
                keep_grad=action.keep_grad,
                isolate=action.isolate,
                direction=direction,
            )
        return action
    if isinstance(action, HelperSpec):
        resolved_direction = direction or action.direction or action.kind
        return InterventionDecision(action="add_hook", hook=action, direction=resolved_direction)
    if callable(action):
        callable_direction = direction or getattr(action, "direction", "forward")
        if callable_direction not in {"forward", "backward", "both"}:
            raise TypeError("intervention direction must be 'forward', 'backward', or 'both'")
        return InterventionDecision(
            action="transform",
            hook=action,
            direction=cast(HelperDirection, callable_direction),
        )
    raise TypeError(
        "intervention action must be InterventionDecision, HelperSpec, callable, or None"
    )


def when(
    condition: Callable[[Any], bool],
    action: InterventionPredicateDecision,
    *,
    direction: HelperDirection | None = None,
) -> InterventionPredicate:
    """Return a predicate that fires ``action`` when ``condition`` matches.

    Parameters
    ----------
    condition:
        Predicate or selector evaluated against a ``RecordContext``.
    action:
        Intervention action sugar to normalize when the condition matches.
    direction:
        Signal direction to intervene on.

    Returns
    -------
    InterventionPredicate
        Predicate returning an ``InterventionDecision`` or ``None``.
    """

    decision = as_intervention_decision(action, direction=direction)

    def _predicate(ctx: Any) -> InterventionDecision | None:
        """Evaluate the conditional intervention predicate."""

        if condition(ctx):
            return decision
        return None

    _predicate.selector = condition  # type: ignore[attr-defined]
    _predicate.decision = decision  # type: ignore[attr-defined]
    return _predicate


def add(delta: torch.Tensor | float | int, *, force_shape_change: bool = False) -> HelperSpec:
    """Create a helper that adds ``delta`` to an out tensor.

    Parameters
    ----------
    delta:
        Scalar or tensor value to add to the current out.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in-compatible forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for additive intervention."""

        def _hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            """Add the configured delta to the out."""

            del hook
            if isinstance(delta, torch.Tensor):
                return out + delta.to(device=out.device, dtype=out.dtype)
            return out + cast(float | int, delta)

        return _hook

    return HelperSpec(
        helper_name="add",
        args=(delta,),
        kwargs=(("force_shape_change", force_shape_change),),
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


def replace_with(
    value: torch.Tensor | Callable[[], torch.Tensor],
    *,
    force_shape_change: bool = False,
) -> HelperSpec:
    """Create a helper that replaces an out with a fixed value.

    Parameters
    ----------
    value:
        Tensor or zero-argument callable returning a tensor.
    force_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    HelperSpec
        Built-in-compatible forward helper spec.
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for fixed replacement."""

        def _hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            """Return the configured replacement tensor."""

            del hook
            replacement = value() if callable(value) else value
            return replacement.to(device=out.device, dtype=out.dtype)

        return _hook

    return HelperSpec(
        helper_name="replace_with",
        args=(value,),
        kwargs=(("force_shape_change", force_shape_change),),
        factory=factory,
        batch_independent=True,
        compatible_with_append=not force_shape_change,
    )


__all__ = [
    "InterventionPredicate",
    "InterventionPredicateDecision",
    "add",
    "as_intervention_decision",
    "replace_with",
    "when",
]
