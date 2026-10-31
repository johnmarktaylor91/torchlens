"""ROAD: remove and debias (Rong et al., 2022).

"A Consistent and Efficient Evaluation Strategy for Attribution Methods"
(Rong, Leemann, Borisov, Kasneci and Kasneci, ICML 2022) removes features
by *noisy linear imputation* instead of a constant baseline, so the removed
region does not leak the shape of the mask to the model: each removed pixel
takes the weighted mean of its 8 neighbours (weight 1/6 for the 4 direct
neighbours, 1/12 for the 4 diagonals, renormalised at the border), solved
jointly for all removed pixels of a channel, plus Gaussian noise. The curve
itself is the deletion curve of this package: most relevant first (MoRF) or
least relevant first (LeRF), each step a recorded intervention.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._core import InputKwargs, _AttributionTarget
from torchlens.attribution._curve_settings import CurveFeatures
from torchlens.attribution._curves import (
    _DEFAULT_PATCH,
    _DEFAULT_STEPS,
    PerturbationCurve,
    _curve,
    _CurveRequest,
)
from torchlens.attribution._eval_maps import _DOCS_POINTER
from torchlens.attribution._result import Attribution, AttributionError

RoadRanking: TypeAlias = Literal["most_relevant_first", "least_relevant_first"]

# The noise of the authors' reference NoisyLinearImputer.
_DEFAULT_NOISE_STD = 0.01
_NEIGHBOR_WEIGHTS = (
    (1 / 12, 1 / 6, 1 / 12),
    (1 / 6, 0.0, 1 / 6),
    (1 / 12, 1 / 6, 1 / 12),
)
_CG_TOLERANCE = 1e-24


def _neighbor_sum(planes: Tensor) -> Tensor:
    """Weighted sum of each element's in-bounds 8 neighbours, per plane."""

    kernel = torch.tensor(_NEIGHBOR_WEIGHTS, dtype=planes.dtype, device=planes.device)
    return F.conv2d(planes, kernel.view(1, 1, 3, 3), padding=1)


def _plane_dot(first: Tensor, second: Tensor) -> Tensor:
    """Per-plane inner product, kept broadcastable over ``(P, 1, H, W)``."""

    return (first * second).sum(dim=(1, 2, 3), keepdim=True)


def _linear_imputation(leaf: Tensor, mask: Tensor) -> Tensor:
    """Solve the noiseless linear imputation of the masked elements of ``leaf``.

    Every masked element equals the weighted mean of its in-bounds neighbours
    (known neighbours are constants, masked ones unknowns): the symmetric
    system ``(D - W_uu) x_u = W_uk v_k`` per plane, solved by conjugate
    gradients in float64. A masked region with no known neighbour is
    underdetermined and stays at zero.

    Parameters
    ----------
    leaf
        ``(N, C, H, W)`` input.
    mask
        Boolean mask of the elements to impute, the leaf's shape.

    Returns
    -------
    Tensor
        Float64 values, imputed where ``mask`` and the input elsewhere.
    """

    height, width = int(leaf.shape[-2]), int(leaf.shape[-1])
    values = leaf.detach().to(torch.float64).reshape(-1, 1, height, width)
    unknown = mask.reshape(-1, 1, height, width).to(torch.float64)
    weight = _neighbor_sum(torch.ones_like(values))

    def _apply(planes: Tensor) -> Tensor:
        """The system matrix on every plane at once."""

        return (weight * planes - _neighbor_sum(planes * unknown)) * unknown

    solution = torch.zeros_like(values)
    residual = _neighbor_sum(values * (1 - unknown)) * unknown
    direction = residual.clone()
    norm = _plane_dot(residual, residual)
    for _ in range(int(unknown.sum(dim=(1, 2, 3)).max().item()) + 1):
        if float(norm.max()) <= _CG_TOLERANCE:
            break
        applied = _apply(direction)
        curvature = _plane_dot(direction, applied)
        step = torch.where(curvature > 0, norm / curvature.clamp_min(1e-300), 0.0)
        solution = solution + step * direction
        residual = residual - step * applied
        new_norm = _plane_dot(residual, residual)
        direction = (
            residual + torch.where(norm > 0, new_norm / norm.clamp_min(1e-300), 0.0) * direction
        )
        norm = new_norm
    imputed = torch.where(unknown > 0, solution, values)
    return imputed.reshape(leaf.shape)


def _noisy_linear_imputer(noise_std: float, seed: int) -> Callable[[Tensor, Tensor, int], Tensor]:
    """Return the ROAD imputer: linear imputation plus seeded Gaussian noise.

    The noise of a step is drawn from a generator seeded by ``seed`` and the
    step's feature count, so a step recomputes to the same values (its score,
    ``inputs`` and recorded edit agree).
    """

    def _impute(leaf: Tensor, mask: Tensor, n_features: int) -> Tensor:
        """Impute the masked elements of one leaf for one step."""

        imputed = _linear_imputation(leaf, mask)
        if noise_std > 0:
            generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(n_features))
            draw = torch.randn(imputed.shape, generator=generator, dtype=torch.float64)
            imputed = imputed + noise_std * draw.to(imputed.device) * mask.to(imputed.dtype)
        return imputed.to(leaf.dtype)

    return _impute


def _validated_road_settings(inputs: Any, ranking: Any, noise_std: Any) -> None:
    """Refuse an unknown ranking, a negative noise, or inputs that are not images."""

    remedy = "tl.attribution.road_curve(model, x, target=0, attribution=attribution"
    if ranking not in ("most_relevant_first", "least_relevant_first"):
        raise AttributionError(
            f"road_curve ranking must be 'most_relevant_first' or 'least_relevant_first'; got "
            f"{ranking!r}. Remedy: {remedy}, ranking='least_relevant_first'). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)) or noise_std < 0:
        raise AttributionError(
            f"road_curve noise_std must be a number of at least 0; got {noise_std!r}. "
            f"Remedy: {remedy}, noise_std=0.01). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    leaves = [inputs] if isinstance(inputs, Tensor) else list(inputs)
    floating = [leaf for leaf in leaves if isinstance(leaf, Tensor) and leaf.is_floating_point()]
    if not floating or any(leaf.ndim != 4 for leaf in floating):
        shapes = [tuple(leaf.shape) for leaf in floating]
        raise AttributionError(
            f"road_curve imputes removed pixels from their spatial neighbours, so it needs "
            f"(N, C, H, W) image inputs; got floating-point inputs of shapes {shapes}. Remedy: "
            "use the constant-baseline deletion curve for other inputs, e.g. "
            "tl.attribution.deletion_curve(model, x, target=0, attribution=attribution). "
            f"{_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )


def road_curve(  # noqa: PLR0913 - public door: keyword-only evaluation settings
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    attribution: Attribution | Any,
    ranking: RoadRanking = "most_relevant_first",
    num_steps: int = _DEFAULT_STEPS,
    features: CurveFeatures = "auto",
    patch_size: int | tuple[int, int] = _DEFAULT_PATCH,
    noise_std: float = _DEFAULT_NOISE_STD,
    seed: int = 0,
) -> PerturbationCurve:
    """Remove features by noisy linear imputation and record the target score (ROAD).

    ROAD, "remove and debias" (Rong et al., ICML 2022): the deletion curve
    of :func:`deletion_curve`, but a removed pixel takes the weighted mean
    of its 8 neighbours (1/6 direct, 1/12 diagonal, solved jointly for the
    removed pixels of each channel) plus Gaussian noise of ``noise_std``, so
    the mask's shape does not leak into the score. ``ranking`` removes the
    most relevant features first (MoRF, default) or the least relevant
    first (LeRF). Every step is a recorded intervention
    (``CurveStep.build_interventions``) holding the step's imputed values.

    Parameters
    ----------
    model
        PyTorch module the attribution explains.
    inputs
        ``(N, C, H, W)`` image tensor, or tuple/list of positional model
        arguments whose floating-point tensors are images.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index (scored per example) or callable
        ``output -> scalar tensor``.
    attribution
        ``AttributionResult`` or tensor over the inputs (``grad_cam`` maps
        broadcast across channels).
    ranking
        ``"most_relevant_first"`` (default, MoRF) or
        ``"least_relevant_first"`` (LeRF).
    num_steps
        Number of steps after the unperturbed start (default 20).
    features
        Feature grouping as in :func:`deletion_curve` (default ``"auto"``:
        pixels for images).
    patch_size
        Patch side for ``features="patches"`` (default 8).
    noise_std
        Standard deviation of the noise added to imputed values (default
        0.01, the authors' reference value); ``0`` imputes without noise.
    seed
        Seed of the noise draws (a local generator per step).

    Returns
    -------
    CurveResult
        ``metric="road"``; fractions, scores, area, AOPC and the recorded
        steps.

    Raises
    ------
    AttributionError
        On non-image inputs, an unknown ranking, a negative ``noise_std``, or
        the refusals of :func:`deletion_curve`.
    """

    _validated_road_settings(inputs, ranking, noise_std)
    request = _CurveRequest(
        target,
        attribution,
        "zeros",
        num_steps,
        features,
        patch_size,
        imputer=_noisy_linear_imputer(float(noise_std), seed),
        least_relevant_first=ranking == "least_relevant_first",
    )
    result = _curve("deletion", model, inputs, input_kwargs, request)
    settings = {
        **result.settings,
        "baseline": "noisy linear imputation",
        "ranking": ranking,
        "noise_std": float(noise_std),
        "seed": seed,
    }
    return dataclasses.replace(result, kind="road", settings=settings)
