"""Predicate-time intervention decisions and sugar."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeAlias, cast

import torch

from .._errors import ArgumentTypeError
from .types import Edit, InterventionDecision

InterventionPredicateDecision: TypeAlias = InterventionDecision | Edit | Callable[..., Any] | None
InterventionPredicate: TypeAlias = Callable[[Any], InterventionPredicateDecision]


def as_intervention_decision(
    action: InterventionPredicateDecision,
) -> InterventionDecision | None:
    """Normalize predicate action sugar to an intervention decision.

    The decision carries no pass: the site it is applied at decides whether
    it edits an activation or a gradient.

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
        if action.keep_grad and action.hook is not None:
            from ._gradient_contract import keep_grad

            # keep_grad=True runs the same connectivity check as keep_grad(edit).
            return dataclasses.replace(action, hook=keep_grad(action.hook))
        return action
    if isinstance(action, Edit):
        return InterventionDecision(action="add_hook", hook=action)
    if callable(action):
        return InterventionDecision(action="transform", hook=action)
    raise ArgumentTypeError(
        f"Intervention action has unsupported type {type(action).__name__}",
        code="intervention_action_type_invalid",
        remedy="pass an InterventionDecision, Edit, callable, or None",
        argument="action",
        received_type=type(action).__name__,
    )


def add(delta: torch.Tensor | float | int, *, allow_shape_change: bool = False) -> Edit:
    """Create a helper that adds ``delta`` to an out tensor.

    Parameters
    ----------
    delta:
        Scalar or tensor value to add to the current out.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in-compatible forward helper spec.

    Examples
    --------
    >>> clean = tl.trace(model, x)
    >>> edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.add(1.0)))
    >>> torch.allclose(edited["relu_1_2"].out, clean["relu_1_2"].out + 1.0)
    True
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for additive intervention."""

        def _hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
            """Add the configured delta to the out."""

            del ctx
            if isinstance(delta, torch.Tensor):
                return out + delta.to(device=out.device, dtype=out.dtype)
            return out + cast(float | int, delta)

        return _hook

    return Edit(
        name="add",
        args=(delta,),
        kwargs=(("allow_shape_change", allow_shape_change),),
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


def replace_with(
    value: torch.Tensor | Callable[[], torch.Tensor],
    *,
    allow_shape_change: bool = False,
) -> Edit:
    """Create a helper that replaces an out with a fixed value.

    Parameters
    ----------
    value:
        Tensor or zero-argument callable returning a tensor.
    allow_shape_change:
        Stored escape-hatch metadata for later execution phases.

    Returns
    -------
    Edit
        Built-in-compatible forward helper spec.

    Examples
    --------
    >>> edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.replace_with(torch.ones(1, 4, 8, 8))))
    >>> bool((edited["relu_1_2"].out == 1).all())
    True
    """

    def factory() -> Callable[..., torch.Tensor]:
        """Return the runtime hook for fixed replacement."""

        def _hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
            """Return the configured replacement tensor."""

            del ctx
            replacement = value() if callable(value) else value
            converted = replacement.to(device=out.device, dtype=out.dtype)
            if converted is replacement:
                # ``.to()`` no-ops to the SAME object when device/dtype already
                # match, so 2+ matched sites would inject ONE shared live
                # tensor: each fire stamps its raw label on that object and the
                # LAST fire steals it -- downstream consumers then hang every
                # child off the last site (wrong parents that validate clean),
                # and chained edits at the orphaned site become silent no-ops.
                # Mint a distinct per-fire object carrying the same value.
                converted = replacement.clone()
            return converted

        return _hook

    return Edit(
        name="replace_with",
        args=(value,),
        kwargs=(("allow_shape_change", allow_shape_change),),
        factory=factory,
        is_batch_independent=True,
        is_compatible_with_append=not allow_shape_change,
    )


__all__ = [
    "InterventionPredicate",
    "InterventionPredicateDecision",
    "add",
    "as_intervention_decision",
    "replace_with",
]
