"""Pairwise distance metrics for the multi-trace subpackage.

These primitives operate on pairs of out (or grad) tensors and return
a scalar tensor in the [0, ~] range -- larger means more dissimilar. The
`METRIC_REGISTRY` and `resolve_metric` helper let callers pass either a string
name or a callable.

The fallback ``relative_l1`` is used implicitly by SuperOp.diff when the
inputs are 0-d (scalar) tensors -- cosine and pearson are meaningless on a single
value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

from .._errors import ArgumentTypeError, InvalidArgumentError

# Small floor used to keep denominators away from zero. Empirically chosen to be
# negligible compared to typical out magnitudes while preventing 0/0
# explosions on dead/zero tensors.
_EPS = 1e-12


def _as_flat_float(t: torch.Tensor) -> torch.Tensor:
    """Return a 1-D float tensor view of ``t`` for metric arithmetic.

    Promotes integer/bool/half tensors to float32 to avoid integer-division
    pitfalls. Always returns a contiguous flattened view (no copy if already
    flat and float).
    """

    if not isinstance(t, torch.Tensor):
        raise ArgumentTypeError(
            f"Intervention metric input has type {type(t).__name__}, not torch.Tensor",
            code="metric_tensor_type_invalid",
            remedy="pass torch.Tensor operands to the metric",
            argument="metric operand",
            received_type=type(t).__name__,
        )
    if t.is_floating_point():
        flat = t.detach().reshape(-1)
    else:
        flat = t.detach().to(torch.float32).reshape(-1)
    return flat


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``1 - cosine_similarity(a, b)`` as a 0-d tensor.

    Both inputs are flattened first. If either flattened vector has zero norm,
    the cosine similarity is defined as 0 (and thus the distance is 1) unless
    the two flattened tensors are bitwise-equal -- in which case we treat them
    as identical (distance 0). This guards self-comparisons of dead/zero
    outs from showing up as maximally-different.
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    if fa.numel() != fb.numel():
        raise InvalidArgumentError(
            f"cosine_distance received {fa.numel()} and {fb.numel()} elements",
            code="metric_shape_mismatch",
            remedy="pass operands with equal element counts",
            metric="cosine_distance",
        )
    na = torch.linalg.vector_norm(fa)
    nb = torch.linalg.vector_norm(fb)
    denom = na * nb
    if denom.item() < _EPS:
        if torch.equal(fa, fb):
            return torch.tensor(0.0, dtype=fa.dtype)
        return torch.tensor(1.0, dtype=fa.dtype)
    return cast(torch.Tensor, 1.0 - (fa @ fb) / denom)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``norm(a - b) / max(norm(a), eps)`` as a 0-d tensor.

    This is asymmetric on purpose -- the denominator is anchored on ``a`` so
    that a "row" comparison `node.compare(other='trace_x')` reads as relative to
    that trace.
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    if fa.numel() != fb.numel():
        raise InvalidArgumentError(
            f"relative_l2 received {fa.numel()} and {fb.numel()} elements",
            code="metric_shape_mismatch",
            remedy="pass operands with equal element counts",
            metric="relative_l2",
        )
    diff = torch.linalg.vector_norm(fa - fb)
    denom = torch.linalg.vector_norm(fa)
    if denom.item() < _EPS:
        # When the reference tensor is the zero tensor, fall back to absolute
        # L2 distance so we still expose the magnitude of the difference.
        return cast(torch.Tensor, diff)
    return cast(torch.Tensor, diff / denom)


def pearson_correlation_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``1 - pearson_r(a, b)`` as a 0-d tensor.

    If either flattened input has zero variance, Pearson is undefined; we
    fall back to 1.0 (maximally uncorrelated) unless the inputs are
    bitwise-equal (treated as identical, distance 0).
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    if fa.numel() != fb.numel():
        raise InvalidArgumentError(
            "pearson_correlation_distance received operands with different element counts",
            code="metric_shape_mismatch",
            remedy="pass operands with equal element counts",
            metric="pearson_correlation_distance",
            element_counts=(fa.numel(), fb.numel()),
        )
    if fa.numel() < 2:
        if torch.equal(fa, fb):
            return torch.tensor(0.0, dtype=fa.dtype)
        return torch.tensor(1.0, dtype=fa.dtype)
    fa_c = fa - fa.mean()
    fb_c = fb - fb.mean()
    denom = torch.linalg.vector_norm(fa_c) * torch.linalg.vector_norm(fb_c)
    if denom.item() < _EPS:
        if torch.equal(fa, fb):
            return torch.tensor(0.0, dtype=fa.dtype)
        return torch.tensor(1.0, dtype=fa.dtype)
    r = (fa_c @ fb_c) / denom
    return cast(torch.Tensor, 1.0 - r)


def relative_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``|a - b| / max(|a|, eps)`` as a 0-d tensor.

    Used as the implicit fallback for scalar (0-d or 1-element) outputs because
    cosine, relative-L2, and Pearson are all meaningless or near-degenerate on a
    single value.
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    # A scalar fallback is only well-defined when both operands hold the same
    # number of elements. Silently reducing a longer vector to its first element
    # (the previous behavior) discards the rest of the tensor and returns a
    # meaningless distance, so validate matching numel and raise on mismatch --
    # matching the numel guards on cosine_distance/relative_l2/pearson.
    if fa.numel() != fb.numel():
        raise InvalidArgumentError(
            "relative_l1 received operands with different element counts",
            code="metric_shape_mismatch",
            remedy="pass scalar-like operands with equal element counts",
            metric="relative_l1",
            element_counts=(fa.numel(), fb.numel()),
        )
    if fa.numel() == 0:
        return torch.tensor(0.0, dtype=fa.dtype)
    a_val = fa.flatten()[0]
    b_val = fb.flatten()[0]
    diff = torch.abs(a_val - b_val)
    denom = torch.abs(a_val)
    if denom.item() < _EPS:
        return diff
    return diff / denom


METRIC_REGISTRY: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cosine": cosine_distance,
    "relative_l2": relative_l2,
    "pearson": pearson_correlation_distance,
}


def resolve_metric(
    metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Resolve a metric specifier to a callable.

    A string is looked up in ``METRIC_REGISTRY``; a callable ops through
    unchanged. Anything else raises ``TypeError``.
    """

    if isinstance(metric, str):
        if metric not in METRIC_REGISTRY:
            valid = ", ".join(sorted(METRIC_REGISTRY))
            raise InvalidArgumentError(
                f"Intervention metric {metric!r} is unknown",
                code="metric_name_invalid",
                remedy=f"choose {valid}, or pass a callable",
                argument="metric",
            )
        return METRIC_REGISTRY[metric]
    if callable(metric):
        return metric
    raise ArgumentTypeError(
        f"Intervention metric has unsupported type {type(metric).__name__}",
        code="metric_type_invalid",
        remedy="pass a registered metric name or a callable",
        argument="metric",
        received_type=type(metric).__name__,
    )


def is_scalar_like(t: torch.Tensor) -> bool:
    """Whether ``t`` should be treated as scalar for diff fallback purposes.

    Treats 0-d tensors and 1-element 1-d tensors uniformly. Anything with 2+
    elements is treated as a vector.
    """

    if not isinstance(t, torch.Tensor):
        return False
    return t.numel() <= 1
