"""Input-rung law and the normalized pre-capture call description (quickstart B4).

This module is the BOTTOM of the four-layer quickstart split (quickstart memo
D1): it depends only on torch and the L0 error basis, so the resolver ->
inference -> capture import cycle is impossible by construction. Everything
here is DOCUMENTED-UNSTABLE pending the naming sprint.

The ladder's law (memo D2): a real input XOR ``input_size=`` XOR nothing.
Mixing is a typed refusal before any forward; a failed explicit size never
falls through to inference; a failed inference never falls through to a stock
tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .._errors import ArgumentConflictError

__tl_layer__ = "L2"

#: Closed rung vocabulary (memo D4). Rung 4 (meta) is reserved behind the D8
#: ruling; the resolver refuses it typed rather than guessing.
RUNG_GOLD = 1
RUNG_DECLARED = 2
RUNG_INFERRED = 3

#: Closed origin vocabulary mirrored into provenance records.
ORIGIN_BY_RUNG: dict[int, str] = {
    RUNG_GOLD: "user",
    RUNG_DECLARED: "declared",
    RUNG_INFERRED: "inferred",
}


@dataclass(frozen=True)
class InputPlan:
    """Normalized, concrete pre-capture call description.

    An ``InputPlan`` always holds CONCRETE forward arguments: synthesis (for
    the declared rung) and inference (for the zero-argument rung) happen
    before a plan exists. The concrete-input capture primitive consumes plans
    and nothing else, so it can never invoke zero-input resolution (memo D1).

    Attributes
    ----------
    rung:
        1 (gold), 2 (declared), or 3 (inferred).
    input_args:
        Positional forward arguments, normalized to a tuple. May be empty
        when the call is keyword-only.
    input_kwargs:
        Keyword forward arguments (empty dict when none).
    verified_trace:
        On the inferred rung only: the exact verification ``Trace`` the
        inference search already captured, consumed by the surface that asked
        instead of recapturing (memo D9 in-call reuse). ``None`` elsewhere.
    """

    rung: int
    input_args: tuple[Any, ...]
    input_kwargs: dict[str, Any] = field(default_factory=dict)
    verified_trace: Any | None = None


def _has_real_input(input_args: Any, input_kwargs: dict[str, Any] | None) -> bool:
    """Return whether the caller supplied a real (rung-1) input.

    ``None`` positional args plus empty/absent kwargs means "no real input".
    Any other positional value -- including an empty list, a bare tensor, an
    HF prompt string, or ``0`` -- is caller-authoritative gold input.
    """

    if input_args is not None:
        return True
    return bool(input_kwargs)


def resolve_rung(
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    input_size: Any,
) -> int:
    """Apply the precedence law and return the resolved rung (memo D2).

    Parameters
    ----------
    input_args:
        Positional forward arguments as passed to the verb (``None`` when
        omitted).
    input_kwargs:
        Keyword forward arguments (``None`` or ``{}`` when omitted).
    input_size:
        The declared-shape argument (``None`` when omitted).

    Returns
    -------
    int
        ``RUNG_GOLD``, ``RUNG_DECLARED``, or ``RUNG_INFERRED``.

    Raises
    ------
    torchlens._errors.ArgumentConflictError
        Code ``input_rung_conflict`` when a real input and ``input_size=``
        are both supplied. The rungs are exclusive by law; TorchLens never
        guesses which one the caller meant.
    """

    real = _has_real_input(input_args, input_kwargs)
    if real and input_size is not None:
        raise ArgumentConflictError(
            "Got BOTH a real input and input_size=. The input ladder's rungs are "
            "exclusive: a real input (best), input_size= (your shape, synthesized "
            "values, disclosed), or nothing (inferred shape, disclosed).",
            code="input_rung_conflict",
            remedy=(
                "pass the real input alone (drop input_size=), or drop the real "
                "input and keep input_size="
            ),
        )
    if real:
        return RUNG_GOLD
    if input_size is not None:
        return RUNG_DECLARED
    return RUNG_INFERRED


def normalize_gold_args(input_args: Any, input_kwargs: dict[str, Any] | None) -> InputPlan:
    """Build the gold-rung plan from caller-authoritative arguments.

    Mirrors the historical ``tl.trace`` convention: a single tensor (or any
    non-list/tuple value, e.g. an HF prompt string) is one positional
    argument; a list or tuple is the positional argument pack.

    Parameters
    ----------
    input_args:
        The real input as passed to the verb.
    input_kwargs:
        Keyword forward arguments.

    Returns
    -------
    InputPlan
        Gold-rung plan with concrete arguments.
    """

    kwargs = dict(input_kwargs) if input_kwargs else {}
    if input_args is None:
        return InputPlan(rung=RUNG_GOLD, input_args=(), input_kwargs=kwargs)
    if isinstance(input_args, (list, tuple)):
        return InputPlan(rung=RUNG_GOLD, input_args=tuple(input_args), input_kwargs=kwargs)
    return InputPlan(rung=RUNG_GOLD, input_args=(input_args,), input_kwargs=kwargs)


def tensor_leaves(plan: InputPlan) -> tuple[torch.Tensor, ...]:
    """Return the direct tensor arguments of a plan (shallow walk).

    Provenance disclosure needs per-tensor shape/dtype/device facts; nested
    container structure is disclosed as-is without deep traversal (the
    capture machinery owns the deep walk).
    """

    leaves: list[torch.Tensor] = []
    for value in (*plan.input_args, *plan.input_kwargs.values()):
        if isinstance(value, torch.Tensor):
            leaves.append(value)
    return tuple(leaves)
