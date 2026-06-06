"""Predicate-time intervention decisions and sugar."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import torch

from .types import HelperSpec, InterventionDecision

InterventionPredicateDecision: TypeAlias = (
    InterventionDecision | HelperSpec | Callable[..., Any] | None
)
InterventionPredicate: TypeAlias = Callable[[Any], InterventionPredicateDecision]


def as_intervention_decision(action: InterventionPredicateDecision) -> InterventionDecision | None:
    """Normalize predicate action sugar to an intervention decision.

    Parameters
    ----------
    action:
        Helper spec, callable transform, intervention decision, or ``None``.

    Returns
    -------
    InterventionDecision | None
        Normalized active intervention decision.
    """

    if action is None:
        return None
    if isinstance(action, InterventionDecision):
        return action
    if isinstance(action, HelperSpec):
        return InterventionDecision(action="add_hook", hook=action)
    if callable(action):
        return InterventionDecision(action="transform", hook=action)
    raise TypeError(
        "intervention action must be InterventionDecision, HelperSpec, callable, or None"
    )


def when(
    condition: Callable[[Any], bool], action: InterventionPredicateDecision
) -> InterventionPredicate:
    """Return a predicate that fires ``action`` when ``condition`` matches.

    Parameters
    ----------
    condition:
        Predicate or selector evaluated against a ``RecordContext``.
    action:
        Intervention action sugar to normalize when the condition matches.

    Returns
    -------
    InterventionPredicate
        Predicate returning an ``InterventionDecision`` or ``None``.
    """

    decision = as_intervention_decision(action)

    def _predicate(ctx: Any) -> InterventionDecision | None:
        """Evaluate the conditional intervention predicate."""

        if condition(ctx):
            return decision
        return None

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
