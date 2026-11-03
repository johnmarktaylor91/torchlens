"""Per-token reducers for the generation recorder (DOCUMENTED-UNSTABLE spellings).

A reducer turns one recorded site activation into the small value that is kept
per generated token instead of the activation itself: a norm, the top-k keys
each query attends to, or the score of an already-fitted linear probe. The
generation recorder (``tl.record(..., generate_kwargs=..., generation=
GenerationOptions(reducers={site: reducer}))``) calls the reducer once per
forward on the site's payload and retains only its result.

Contract for every reducer, built-in or user-written: it receives the site's
payload for one forward (a decode step holds one position, the prefill holds
the whole prompt) and returns a tensor whose FIRST axis runs over that
forward's positions, so the per-step values concatenate into one series over
the sequence. The built-ins drop the batch axis (the recorder holds one prompt
at a time) and move the position axis to the front.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .._errors import InvalidArgumentError

__all__ = ["Reducer", "norm", "probe", "top_k"]


class Reducer:
    """A named per-token reducer: ``name`` is recorded beside every value it produces.

    Parameters
    ----------
    name:
        Short name recorded in each reduced record's metadata.
    fn:
        The reduction; see the module docstring for its contract.
    """

    __slots__ = ("fn", "name")

    def __init__(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.name = name
        self.fn = fn

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.fn(tensor)

    def __repr__(self) -> str:
        return f"Reducer({self.name!r})"


def _position_major(tensor: torch.Tensor, position_dim: int) -> torch.Tensor:
    """Return ``tensor`` without its batch axis and with positions first."""

    if tensor.dim() < 2:
        raise InvalidArgumentError(
            f"a reducer needs a payload with a batch axis and a position axis; got shape "
            f"{tuple(tensor.shape)}",
            code="record_generation_reducer_invalid",
            remedy="declare a site whose output is a [batch, ..., positions, ...] tensor",
        )
    squeezed = tensor[0]
    axis = position_dim - 1 if position_dim > 0 else position_dim
    return torch.movedim(squeezed, axis, 0)


def norm(ord: float = 2.0, *, dim: int = -1, position_dim: int = 1) -> Reducer:
    """Return a reducer that keeps the vector norm of each position.

    Parameters
    ----------
    ord:
        The norm order, as ``torch.linalg.vector_norm`` takes it.
    dim:
        The feature axis the norm runs over (the last axis by default).
    position_dim:
        The position axis of the site's payload (axis 1 for a ``[batch,
        positions, features]`` site).

    Returns
    -------
    Reducer
        Produces ``[positions]`` per forward.
    """

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        """Return one norm per position."""

        values = torch.linalg.vector_norm(tensor.detach().float(), ord=ord, dim=dim)
        return _position_major(values, position_dim)

    return Reducer(f"norm(ord={ord})", reduce)


def top_k(k: int, *, dim: int = -1, position_dim: int = -2) -> Reducer:
    """Return a reducer that keeps the indices of the ``k`` largest entries per position.

    On an attention-probability site shaped ``[batch, heads, queries, keys]``
    the defaults keep, for each query position and head, the ``k`` keys it
    attends to most.

    Parameters
    ----------
    k:
        How many indices to keep.
    dim:
        The axis the top-k runs over (the key axis by default).
    position_dim:
        The position axis of the site's payload (the query axis by default).

    Returns
    -------
    Reducer
        Produces ``[positions, ..., k]`` integer indices per forward.
    """

    if k < 1:
        raise InvalidArgumentError(
            f"top_k needs k >= 1; got {k}",
            code="record_generation_reducer_invalid",
            remedy="pass a positive k",
        )

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        """Return the ``k`` largest indices per position."""

        size = tensor.shape[dim]
        indices = torch.topk(tensor.detach(), min(k, size), dim=dim).indices
        return _position_major(indices, position_dim)

    return Reducer(f"top_k(k={k})", reduce)


def _linear_terms(fitted: Any) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return ``(weight [out, features], bias [out], description)`` of a fitted linear probe."""

    if isinstance(fitted, nn.Linear):
        weight = fitted.weight.detach()
        bias = fitted.bias.detach() if fitted.bias is not None else torch.zeros(weight.shape[0])
        return weight, bias, "nn.Linear"
    if isinstance(fitted, (tuple, list)) and len(fitted) == 2:
        weight = torch.as_tensor(fitted[0])
        bias = torch.as_tensor(fitted[1]) if fitted[1] is not None else torch.zeros(weight.shape[0])
        return torch.atleast_2d(weight), torch.atleast_1d(bias), "(weight, bias)"
    coef = getattr(fitted, "coef_", None)
    if coef is not None:
        weight = torch.atleast_2d(torch.as_tensor(coef))
        intercept = getattr(fitted, "intercept_", 0.0)
        bias = torch.atleast_1d(torch.as_tensor(intercept)).expand(weight.shape[0])
        return weight, bias, type(fitted).__name__
    raise InvalidArgumentError(
        f"probe() needs a fitted linear probe: an object with coef_ and intercept_ (a fitted "
        f"scikit-learn linear estimator), an nn.Linear, or a (weight, bias) pair; got "
        f"{type(fitted).__name__}",
        code="record_generation_reducer_invalid",
        remedy="fit the probe first, or pass its weight and bias as a pair",
    )


def probe(fitted: Any, *, position_dim: int = 1) -> Reducer:
    """Return a reducer that scores every position with an already-fitted linear probe.

    The score is the probe's decision function, ``x @ weight.T + bias``: for a
    fitted scikit-learn estimator it equals ``decision_function`` on the same
    activations. A probe with one output gives one score per position; a
    multi-class probe gives one score per class.

    Parameters
    ----------
    fitted:
        A fitted scikit-learn linear estimator (anything with ``coef_`` and
        ``intercept_``), an ``nn.Linear``, or a ``(weight, bias)`` pair.
    position_dim:
        The position axis of the site's payload.

    Returns
    -------
    Reducer
        Produces ``[positions]`` (one output) or ``[positions, outputs]`` per forward.
    """

    weight, bias, described = _linear_terms(fitted)
    weight = weight.detach().to(torch.float32)
    bias = bias.detach().to(torch.float32)

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        """Return the probe's decision function per position."""

        x = tensor.detach().to(torch.float32)
        if x.shape[-1] != weight.shape[1]:
            raise InvalidArgumentError(
                f"the probe was fitted on {weight.shape[1]} features but the site has "
                f"{x.shape[-1]}",
                code="record_generation_reducer_invalid",
                remedy="declare the site the probe was fitted on",
            )
        scores = x @ weight.to(x.device).T + bias.to(x.device)
        if scores.shape[-1] == 1:
            scores = scores[..., 0]
        return _position_major(scores, position_dim)

    return Reducer(f"probe({described}, outputs={weight.shape[0]})", reduce)
