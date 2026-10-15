"""Settings of the deletion and insertion curves: feature grouping, baselines, steps.

Validates and resolves the keyword-only settings of ``deletion_curve`` and
``insertion_curve`` (``features``, ``patch_size``, ``baseline``, ``n_steps``),
each refusal coded ``attribution_eval_setting_invalid`` with a spelling that
runs.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

from torchlens.attribution._eval_maps import _DOCS_POINTER
from torchlens.attribution._result import AttributionError

CurveFeatures: TypeAlias = Literal["auto", "elements", "pixels", "patches", "tokens"]
CurveBaseline: TypeAlias = Literal["zeros", "blur"] | float | Tensor

_BLUR_KERNEL = 11


def _resolve_features(features: str, leaves: tuple[Tensor, ...], method: str) -> str:
    """Resolve ``features="auto"`` and validate the grouping name.

    Parameters
    ----------
    features
        Requested grouping.
    leaves
        Evaluated inputs.
    method
        Public name used in the refusal.

    Returns
    -------
    str
        ``"elements"``, ``"pixels"``, ``"patches"`` or ``"tokens"``.

    Raises
    ------
    AttributionError
        On an unknown grouping or one the input shape does not support.
    """

    allowed = ("auto", "elements", "pixels", "patches", "tokens")
    if features not in allowed:
        raise AttributionError(
            f"{method} features must be one of {allowed}; got {features!r}. Remedy: "
            f"tl.attribution.{method}(model, x, target=0, attribution=result, "
            f"features='elements'). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    if features == "auto":
        ranks = {leaf.ndim for leaf in leaves}
        if ranks == {4}:
            return "pixels"
        if ranks == {3}:
            return "tokens"
        return "elements"
    needed = {"pixels": 4, "patches": 4, "tokens": 3}.get(features)
    bad = [tuple(leaf.shape) for leaf in leaves if needed is not None and leaf.ndim != needed]
    if bad:
        layout = "(N, T, ...)" if features == "tokens" else "(N, C, H, W)"
        raise AttributionError(
            f"{method} features={features!r} groups inputs laid out {layout}, but got "
            f"inputs of shape {bad}. Remedy: tl.attribution.{method}(model, x, target=0, "
            f"attribution=result, features='elements'). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    return features


def _patch_sides(patch_size: int | tuple[int, int], method: str) -> tuple[int, int]:
    """Validate ``patch_size``.

    Parameters
    ----------
    patch_size
        Square side or ``(height, width)``.
    method
        Public name used in the refusal.

    Returns
    -------
    tuple[int, int]
        Patch height and width.

    Raises
    ------
    AttributionError
        If a side is not a positive integer.
    """

    sides = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
    if len(sides) != 2 or any(
        isinstance(side, bool) or not isinstance(side, int) or side <= 0 for side in sides
    ):
        raise AttributionError(
            f"{method} patch_size must be a positive int or a pair of them; got "
            f"{patch_size!r}. Remedy: tl.attribution.{method}(model, x, target=0, "
            f"attribution=result, features='patches', patch_size=8). {_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    return (int(sides[0]), int(sides[1]))


def _leaf_groups(leaf: Tensor, features: str, patch: tuple[int, int]) -> tuple[Tensor, int]:
    """Map each non-batch element of a leaf to a local feature id.

    Parameters
    ----------
    leaf
        One evaluated input.
    features
        Resolved grouping.
    patch
        Patch height and width (``"patches"`` only).

    Returns
    -------
    tuple[Tensor, int]
        Long tensor of ``leaf.shape[1:]`` holding feature ids, and the
        number of features.
    """

    shape = tuple(leaf.shape[1:])
    if features == "elements":
        count = math.prod(shape)
        return torch.arange(count, device=leaf.device).reshape(shape), count
    if features == "tokens":
        tokens = shape[0]
        ids = torch.arange(tokens, device=leaf.device).reshape(tokens, *([1] * (len(shape) - 1)))
        return ids.expand(shape).contiguous(), tokens
    channels, height, width = shape
    if features == "pixels":
        ids = torch.arange(height * width, device=leaf.device).reshape(1, height, width)
        return ids.expand(shape).contiguous(), height * width
    rows = torch.arange(height, device=leaf.device) // patch[0]
    cols = torch.arange(width, device=leaf.device) // patch[1]
    n_cols = int(cols.max()) + 1
    ids = (rows[:, None] * n_cols + cols[None, :]).reshape(1, height, width)
    return ids.expand(channels, height, width).contiguous(), (int(rows.max()) + 1) * n_cols


def _baseline_for(baseline: Any, leaf: Tensor, method: str) -> Tensor:
    """Resolve the baseline of one leaf.

    Parameters
    ----------
    baseline
        ``"zeros"``, ``"blur"``, a number or a tensor of the leaf's shape.
    leaf
        The evaluated input.
    method
        Public name used in the refusal.

    Returns
    -------
    Tensor
        Baseline of the leaf's shape, dtype and device.

    Raises
    ------
    AttributionError
        If the baseline is not one of the supported forms.
    """

    if isinstance(baseline, str) and baseline == "zeros":
        return torch.zeros_like(leaf)
    if isinstance(baseline, str) and baseline == "blur" and leaf.ndim == 4:
        pad = _BLUR_KERNEL // 2
        padded = F.pad(leaf, (pad, pad, pad, pad), mode="replicate")
        return F.avg_pool2d(padded, _BLUR_KERNEL, stride=1).to(leaf.dtype)
    if isinstance(baseline, (int, float)) and not isinstance(baseline, bool):
        return torch.full_like(leaf, float(baseline))
    if isinstance(baseline, Tensor) and tuple(baseline.shape) == tuple(leaf.shape):
        return baseline.detach().to(device=leaf.device, dtype=leaf.dtype)
    raise AttributionError(
        f"{method} baseline must be 'zeros', 'blur' (inputs laid out (N, C, H, W)), a "
        f"number, or a tensor of the input's shape {tuple(leaf.shape)}; got "
        f"{_baseline_repr(baseline)}. Remedy: tl.attribution.{method}(model, x, target=0, "
        f"attribution=result, baseline='zeros'). {_DOCS_POINTER}",
        code="attribution_eval_setting_invalid",
    )


def _baseline_repr(baseline: Any) -> str:
    """Short description of a baseline for settings and messages."""

    if isinstance(baseline, Tensor):
        return f"Tensor(shape={tuple(baseline.shape)})"
    return repr(baseline)


def _validated_steps(n_steps: int, n_features: int, method: str) -> list[int]:
    """Return the feature count of every step.

    Parameters
    ----------
    n_steps
        Requested number of steps.
    n_features
        Features per example.
    method
        Public name used in the refusal.

    Returns
    -------
    list[int]
        ``n_steps + 1`` non-decreasing counts from 0 to ``n_features``.

    Raises
    ------
    AttributionError
        If ``n_steps`` is not a positive integer.
    """

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise AttributionError(
            f"{method} n_steps must be a positive integer; got {n_steps!r}. Remedy: "
            f"tl.attribution.{method}(model, x, target=0, attribution=result, n_steps=20). "
            f"{_DOCS_POINTER}",
            code="attribution_eval_setting_invalid",
        )
    return [(step * n_features) // n_steps for step in range(n_steps + 1)]
