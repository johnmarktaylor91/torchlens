"""Pairwise distance metrics for the multi-trace subpackage.

These primitives operate on pairs of activation (or gradient) tensors and return
a scalar tensor in the [0, ~] range -- larger means more dissimilar. The
`METRIC_REGISTRY` and `resolve_metric` helper let callers pass either a string
name or a callable.

The fallback ``relative_l1_scalar`` is used implicitly by NodeView.diff when the
inputs are 0-d (scalar) tensors -- cosine and pearson are meaningless on a single
value.
"""

from __future__ import annotations

from typing import Callable, Dict, Union

import torch


# Small floor used to keep denominators away from zero. Empirically chosen to be
# negligible compared to typical activation magnitudes while preventing 0/0
# explosions on dead/zero tensors.
_EPS = 1e-12


def _as_flat_float(t: torch.Tensor) -> torch.Tensor:
    """Return a 1-D float tensor view of ``t`` for metric arithmetic.

    Promotes integer/bool/half tensors to float32 to avoid integer-division
    pitfalls. Always returns a contiguous flattened view (no copy if already
    flat and float).
    """

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
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
    activations from showing up as maximally-different.
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    if fa.numel() != fb.numel():
        raise ValueError(
            f"cosine_distance requires equal element counts, got {fa.numel()} vs {fb.numel()}"
        )
    na = torch.linalg.vector_norm(fa)
    nb = torch.linalg.vector_norm(fb)
    denom = na * nb
    if denom.item() < _EPS:
        if torch.equal(fa, fb):
            return torch.tensor(0.0, dtype=fa.dtype)
        return torch.tensor(1.0, dtype=fa.dtype)
    return 1.0 - (fa @ fb) / denom


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``norm(a - b) / max(norm(a), eps)`` as a 0-d tensor.

    This is asymmetric on purpose -- the denominator is anchored on ``a`` so
    that a "row" comparison `node.diff(other='trace_x')` reads as relative to
    that trace.
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    if fa.numel() != fb.numel():
        raise ValueError(
            f"relative_l2 requires equal element counts, got {fa.numel()} vs {fb.numel()}"
        )
    diff = torch.linalg.vector_norm(fa - fb)
    denom = torch.linalg.vector_norm(fa)
    if denom.item() < _EPS:
        # When the reference tensor is the zero tensor, fall back to absolute
        # L2 distance so we still expose the magnitude of the difference.
        return diff
    return diff / denom


def pearson_correlation_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``1 - pearson_r(a, b)`` as a 0-d tensor.

    If either flattened input has zero variance, Pearson is undefined; we
    fall back to 1.0 (maximally uncorrelated) unless the inputs are
    bitwise-equal (treated as identical, distance 0).
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    if fa.numel() != fb.numel():
        raise ValueError(
            f"pearson_correlation_distance requires equal element counts, got "
            f"{fa.numel()} vs {fb.numel()}"
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
    return 1.0 - r


def relative_l1_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``|a - b| / max(|a|, eps)`` as a 0-d tensor.

    Used as the implicit fallback for scalar (0-d or 1-element) outputs because
    cosine, relative-L2, and Pearson are all meaningless or near-degenerate on a
    single value.
    """

    fa = _as_flat_float(a)
    fb = _as_flat_float(b)
    # A scalar fallback is only well-defined for 1-element comparisons; clamp
    # gracefully if the caller hands us a longer vector by reducing to
    # element-wise absolute difference.
    if fa.numel() == 0 or fb.numel() == 0:
        return torch.tensor(0.0, dtype=fa.dtype)
    a_val = fa.flatten()[0]
    b_val = fb.flatten()[0]
    diff = torch.abs(a_val - b_val)
    denom = torch.abs(a_val)
    if denom.item() < _EPS:
        return diff
    return diff / denom


METRIC_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cosine": cosine_distance,
    "relative_l2": relative_l2,
    "pearson": pearson_correlation_distance,
}


def resolve_metric(
    metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Resolve a metric specifier to a callable.

    A string is looked up in ``METRIC_REGISTRY``; a callable passes through
    unchanged. Anything else raises ``TypeError``.
    """

    if isinstance(metric, str):
        if metric not in METRIC_REGISTRY:
            valid = ", ".join(sorted(METRIC_REGISTRY))
            raise ValueError(
                f"Unknown metric '{metric}'. Valid metrics: {valid}, or pass a callable."
            )
        return METRIC_REGISTRY[metric]
    if callable(metric):
        return metric
    raise TypeError(f"metric must be a string or callable, got {type(metric).__name__}")


def is_scalar_like(t: torch.Tensor) -> bool:
    """Whether ``t`` should be treated as scalar for diff fallback purposes.

    Treats 0-d tensors and 1-element 1-d tensors uniformly. Anything with 2+
    elements is treated as a vector.
    """

    if not isinstance(t, torch.Tensor):
        return False
    return t.numel() <= 1
