"""Predicate-time intervention decisions and sugar."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, TypeAlias, cast

import torch

from .._errors import ArgumentTypeError
from .._selection.selectors import BaseSelector
from ._zero_fire import warn_multiclause_unfired
from .hooks import site_universe
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


def _warn_intervene_fired_nowhere(intervene: object, fire_count: int) -> None:
    """Warn when a recording's capture-time ``intervene=`` fired at no site.

    A single activation selector keeps its zero-match UserWarning; a
    multi-clause spec, which has no single ``.selector``, warns per forward
    rule with the coded ``trace_zero_fire`` disclosure.

    Parameters
    ----------
    intervene:
        The recording's ``intervene=`` option.
    fire_count:
        How many times the recording's intervene predicate fired.
    """

    intervene_selector = getattr(intervene, "selector", None)
    if (
        isinstance(intervene_selector, BaseSelector)
        and site_universe(intervene_selector) == "activation"
        and fire_count == 0
    ):
        warnings.warn(
            f"Capture-time intervention selector {intervene_selector!r} matched zero sites; "
            "no intervention fired.",
            UserWarning,
            stacklevel=4,
        )
    warn_multiclause_unfired(intervene, fire_count)


__all__ = [
    "InterventionPredicate",
    "InterventionPredicateDecision",
    "add",
    "as_intervention_decision",
    "replace_with",
]
