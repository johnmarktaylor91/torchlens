"""Shared helpers for the attribution evaluation battery.

The deletion and insertion curves (``_curves.py``) and the randomization test
(``_randomization.py``) both read an attribution as a map over one input
tensor. This module aligns an ``AttributionResult`` (or a raw tensor) with
that input, including ``grad_cam`` maps, which arrive as ``(N, 1, H, W)``
input-resolution maps and broadcast across the input channels, and computes
the similarity measures of the randomization test: Spearman rank correlation
and SSIM.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from torchlens.attribution._result import Attribution, AttributionError

_DOCS_POINTER = "See docs/reference/attribution.md, section Evaluating attributions."

# SSIM constants of Wang et al. (2004) with the 7x7 uniform window and sample
# covariance that scikit-image's ``structural_similarity`` uses by default.
_SSIM_WINDOW = 7
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _attribution_tensor(attribution: Attribution | Any) -> Tensor:
    """Return the single attribution tensor of a result or raw value.

    Parameters
    ----------
    attribution
        ``AttributionResult`` or tensor.

    Returns
    -------
    Tensor
        Detached attribution values.

    Raises
    ------
    AttributionError
        If the values are not one tensor.
    """

    values = attribution.values if isinstance(attribution, Attribution) else attribution
    if not isinstance(values, Tensor):
        raise AttributionError(
            "attribution must be an AttributionResult holding one tensor, or a tensor; "
            f"got {type(values).__name__}. Remedy: attribute one input, e.g. "
            "tl.attribution.saliency(model, x, target=0), and pass that result. "
            f"{_DOCS_POINTER}",
            code="attribution_eval_map_misaligned",
        )
    return values.detach()


def _input_space_map(attribution: Attribution | Any, leaf: Tensor, method: str) -> Tensor:
    """Align an attribution with the input it explains.

    An attribution of the input's own shape is used as is. A map with one
    fewer axis, or a singleton channel axis (``grad_cam`` returns
    ``(N, 1, H, W)`` maps upsampled to the input grid), broadcasts across the
    input's channel axis, which keeps its ranking of positions.

    Parameters
    ----------
    attribution
        ``AttributionResult`` or tensor.
    leaf
        The evaluated input.
    method
        Public name used in the refusal.

    Returns
    -------
    Tensor
        Float64 attribution of ``leaf``'s shape.

    Raises
    ------
    AttributionError
        If the attribution cannot be aligned with the input.
    """

    values = _attribution_tensor(attribution).to(device=leaf.device)
    if values.is_complex():
        values = values.abs()
    shape = tuple(leaf.shape)
    if (
        values.ndim == leaf.ndim - 1
        and leaf.ndim >= 3
        and tuple(values.shape)
        == (
            shape[0],
            *shape[2:],
        )
    ):
        values = values.unsqueeze(1)
    if values.ndim == leaf.ndim and tuple(values.shape) != shape:
        broadcastable = all(
            got == want or (axis == 1 and got == 1)
            for axis, (got, want) in enumerate(zip(values.shape, shape, strict=True))
        )
        if broadcastable:
            values = values.expand(shape)
    if tuple(values.shape) != shape:
        raise AttributionError(
            f"{method} needs an attribution over the evaluated input of shape {shape}, "
            f"but got one of shape {tuple(values.shape)}; layer-space values do not "
            "rank input features. Remedy: evaluate an input-space result such as "
            "tl.attribution.saliency(model, x, target=0), or a grad_cam result, which "
            f"is upsampled to the input grid. {_DOCS_POINTER}",
            code="attribution_eval_map_misaligned",
        )
    return values.to(torch.float64)


def _input_space_maps(
    attribution: Attribution | Any, leaves: tuple[Tensor, ...], method: str
) -> tuple[Tensor, ...]:
    """Align an attribution with every evaluated input leaf.

    Parameters
    ----------
    attribution
        ``AttributionResult`` or value tree; one tensor for one input, or a
        tree whose tensors mirror the floating-point inputs in traversal order
        (``None`` slots skipped).
    leaves
        The evaluated inputs.
    method
        Public name used in the refusal.

    Returns
    -------
    tuple[Tensor, ...]
        Float64 attributions, one per leaf, each of its leaf's shape.

    Raises
    ------
    AttributionError
        If the attribution does not mirror the inputs.
    """

    if len(leaves) == 1:
        return (_input_space_map(attribution, leaves[0], method),)
    values = attribution.values if isinstance(attribution, Attribution) else attribution
    flat: list[Tensor] = []

    def _collect(tree: Any) -> None:
        """Collect tensor leaves in traversal order (``None`` slots skipped)."""

        if isinstance(tree, Tensor):
            flat.append(tree)
        elif isinstance(tree, (tuple, list)):
            for item in tree:
                _collect(item)
        elif isinstance(tree, dict):
            for item in tree.values():
                _collect(item)

    _collect(values)
    if len(flat) != len(leaves):
        raise AttributionError(
            f"{method} evaluates {len(leaves)} floating-point inputs, but the attribution "
            f"holds {len(flat)} tensors. Remedy: pass the result of one attribution call "
            "on the same inputs, e.g. tl.attribution.saliency(model, (x, y), target=0). "
            f"{_DOCS_POINTER}",
            code="attribution_eval_map_misaligned",
        )
    return tuple(
        _input_space_map(value, leaf, method) for value, leaf in zip(flat, leaves, strict=True)
    )


def _average_ranks(values: Tensor) -> Tensor:
    """Return 1-based ranks of a vector, ties sharing their average rank.

    Parameters
    ----------
    values
        One-dimensional tensor.

    Returns
    -------
    Tensor
        Float64 ranks of ``values``.
    """

    flat = values.reshape(-1).to(torch.float64)
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_values, return_inverse=True, return_counts=True
    )
    ends = torch.cumsum(counts, dim=0).to(torch.float64)
    starts = ends - counts.to(torch.float64) + 1.0
    sorted_ranks = ((starts + ends) / 2.0)[inverse]
    ranks = torch.empty_like(flat)
    ranks[order] = sorted_ranks
    return ranks


def _spearman(first: Tensor, second: Tensor) -> float | None:
    """Spearman rank correlation of two equally sized tensors.

    Parameters
    ----------
    first, second
        Tensors with the same number of elements.

    Returns
    -------
    float | None
        The Pearson correlation of the average ranks, or ``None`` when either
        map is constant (the correlation is undefined).
    """

    left = _average_ranks(first)
    right = _average_ranks(second)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.sqrt((left * left).sum() * (right * right).sum())
    if float(denominator) == 0.0:
        return None
    return float((left * right).sum() / denominator)


def _ssim_plane(value: Tensor) -> Tensor | None:
    """Reduce one example's map to a 2-D plane rescaled to ``[0, 1]``.

    Parameters
    ----------
    value
        One example's attribution, ``(H, W)`` or ``(C, H, W)``.

    Returns
    -------
    Tensor | None
        The absolute map summed over channels and min-max rescaled, or
        ``None`` when the map is not spatial or a side is under 7.
    """

    plane = value.abs().to(torch.float64)
    if plane.ndim == 3:
        plane = plane.sum(dim=0)
    if plane.ndim != 2 or min(plane.shape) < _SSIM_WINDOW:
        return None
    low, high = plane.min(), plane.max()
    if float(high - low) == 0.0:
        return torch.zeros_like(plane)
    return (plane - low) / (high - low)


def _ssim(first: Tensor, second: Tensor) -> float:
    """Mean SSIM of two ``[0, 1]`` planes (Wang et al., 2004).

    Uses a 7x7 uniform window, sample covariance, ``data_range=1`` and the
    valid region only, matching scikit-image's ``structural_similarity``
    defaults.

    Parameters
    ----------
    first, second
        Planes of equal shape.

    Returns
    -------
    float
        Mean structural similarity.
    """

    window = _SSIM_WINDOW
    pixels = window * window
    left = first.reshape(1, 1, *first.shape)
    right = second.reshape(1, 1, *second.shape)

    def _mean(value: Tensor) -> Tensor:
        """Mean over every valid 7x7 window."""

        return F.avg_pool2d(value, window, stride=1)

    covariance_scale = pixels / (pixels - 1)
    mean_left, mean_right = _mean(left), _mean(right)
    var_left = covariance_scale * (_mean(left * left) - mean_left * mean_left)
    var_right = covariance_scale * (_mean(right * right) - mean_right * mean_right)
    cov = covariance_scale * (_mean(left * right) - mean_left * mean_right)
    c1 = _SSIM_K1**2
    c2 = _SSIM_K2**2
    numerator = (2 * mean_left * mean_right + c1) * (2 * cov + c2)
    denominator = (mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)
    return float((numerator / denominator).mean())


def _map_similarity(original: Tensor, randomized: Tensor) -> dict[str, float | None]:
    """Similarity measures of two attribution maps, averaged over the batch.

    Parameters
    ----------
    original, randomized
        Attribution tensors of equal shape with a leading batch axis.

    Returns
    -------
    dict[str, float | None]
        ``spearman`` (signed values), ``spearman_abs`` (absolute values) and
        ``ssim`` (``None`` when the maps are not spatial planes of side 7 or
        more); a measure undefined for some example is averaged over the rest,
        and ``None`` when undefined for all.
    """

    measures: dict[str, list[float]] = {"spearman": [], "spearman_abs": [], "ssim": []}
    for first, second in zip(original, randomized, strict=True):
        signed = _spearman(first, second)
        if signed is not None:
            measures["spearman"].append(signed)
        absolute = _spearman(first.abs(), second.abs())
        if absolute is not None:
            measures["spearman_abs"].append(absolute)
        left, right = _ssim_plane(first), _ssim_plane(second)
        if left is not None and right is not None:
            measures["ssim"].append(_ssim(left, right))
    return {name: (sum(found) / len(found) if found else None) for name, found in measures.items()}
