"""Value proof for an eval-mode batch_norm whose input sits exactly at its running mean.

In eval mode (``training`` False) ``torch.batch_norm(input, weight, bias, running_mean,
running_var, training, momentum, eps, cudnn_enabled)`` computes, per channel ``c``,
``(input[:, c] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]``.
When the saved input equals the saved running mean exactly in every channel whose weight
is non-zero, the numerator is exactly zero, so the output equals ``bias`` for every
running_var with ``running_var + eps`` positive and finite: the running_var edge cannot
reach the output value. Untrained models hit this at validation time when a dead ReLU or a
spiking neuron that never fires feeds a fresh BatchNorm (input all zero, running_mean zero).

Narrow by construction: only a single perturbed parent that occupies ``args[4]``
(running_var) and no other positional or keyword slot is ever exempted, only for an
eval-mode call (``args[5] is False``), only when the saved input, running_mean and output are
finite and the input equals the running mean exactly (no tolerance) in every channel that the
weight does not already annihilate. Perturbing the input or running_mean (which moves the
numerator off zero), weight or bias is never exempted.

Two further preconditions make the claim hold for the values validation itself substitutes:

* running_mean is exactly zero in every channel with a non-zero weight. The CPU eval kernel
  folds the statistics into ``input * alpha + (bias - running_mean * alpha)`` with
  ``alpha = weight / sqrt(running_var + eps)``; the two ``running_mean * alpha`` terms cancel
  exactly only when running_mean is zero, so at a non-zero mean the output moves by a few
  ulps with running_var.
* every running_var validation can substitute keeps ``running_var + eps`` positive and the
  per-channel scale finite. The perturbation of a running_var parent moves its saved minimum
  down by at most ``max(1.0, 0.1 * |min|, 0.1 * max|output|)`` (the near-constant widening in
  ``core._perturb_layer_outs``, which also bounds its scaled expansion and the unit step of
  ``core._directional_step_perturb``). Below that floor a live edge reads NaN while a dead
  one reads unchanged, so the proof refuses there; the tests replay every accepted case
  under each strategy so a wider perturbation fails a test instead of going stale.
"""

import math
from typing import Any

import torch

from ._value_predicates import _saved_output_all_finite

#: Unit step and widening fraction of validation's running_var perturbation (module docstring).
_PERTURB_UNIT_STEP = 1.0
_PERTURB_WIDENING_FRACTION = 0.1


def _perturbed_parent_is_only_running_var(layer: Any, layers_to_perturb: list[str]) -> bool:
    """Return whether the single perturbed parent feeds ``args[4]`` and nothing else."""

    from .exemptions import (
        _perturbed_parent_arg_positions,
        _perturbed_parents_only_occupy_template_slot,
    )

    if _perturbed_parent_arg_positions(layer, layers_to_perturb) != {4}:
        return False
    return _perturbed_parents_only_occupy_template_slot(
        layer, layers_to_perturb, template_arg_roots=(4,), template_kwarg_names=()
    )


def _input_at_running_mean(x: torch.Tensor, mean: torch.Tensor, weight: Any) -> bool:
    """Return whether ``x[:, c] == mean[c]`` exactly wherever ``weight[c]`` is non-zero."""

    if x.dim() < 2 or mean.dim() != 1 or x.shape[1] != mean.numel() or x.numel() == 0:
        return False
    if not (torch.isfinite(x).all() and torch.isfinite(mean).all()):
        return False
    view = [1, mean.numel()] + [1] * (x.dim() - 2)
    at_mean = (x == mean.view(view)).transpose(0, 1).reshape(mean.numel(), -1).all(dim=1)
    if weight is None:
        return bool(at_mean.all())
    if not isinstance(weight, torch.Tensor) or weight.shape != mean.shape:
        return False
    return bool((at_mean | (weight == 0)).all())


def _running_mean_zero_where_weighted(mean: torch.Tensor, weight: Any) -> bool:
    """Return whether ``mean[c] == 0`` exactly wherever ``weight[c]`` is non-zero."""

    live = torch.ones_like(mean, dtype=torch.bool) if weight is None else weight.detach() != 0
    return bool((mean[live] == 0).all())


def _perturbed_running_var_stays_in_domain(
    layer: Any, var: Any, eps: Any, weight: Any, x: torch.Tensor
) -> bool:
    """Return whether every running_var validation substitutes keeps the output exact.

    The substituted minimum is at least ``min(var) + eps - excursion`` (module docstring);
    it must stay positive, and ``max|weight| / sqrt`` of it must stay finite in the input
    dtype so ``0 * alpha`` cannot become ``0 * inf``.
    """

    if not isinstance(var, torch.Tensor) or not var.is_floating_point() or var.numel() == 0:
        return False
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        return False
    if not bool(torch.isfinite(var).all()):
        return False
    lo = float(var.min())
    out = layer.out.detach()
    out_scale = float(out.abs().max()) if out.numel() else 0.0
    excursion = max(
        _PERTURB_UNIT_STEP,
        _PERTURB_WIDENING_FRACTION * abs(lo),
        _PERTURB_WIDENING_FRACTION * out_scale,
    )
    margin = lo + float(eps) - excursion
    resolution = 4 * torch.finfo(var.dtype).eps * max(1.0, abs(lo), excursion)
    if not margin > resolution:
        return False
    w_max = 1.0 if weight is None else float(weight.detach().abs().max())
    return w_max / math.sqrt(margin) < torch.finfo(x.dtype).max


def batch_norm_input_at_running_mean(
    layer: Any,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> bool:
    """Return whether running_var provably cannot reach an eval-mode batch_norm's output.

    Parameters
    ----------
    layer:
        ``batch_norm`` op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer`` (ATen order).

    Returns
    -------
    bool
        True only when every precondition in the module docstring holds.
    """

    if getattr(layer, "func_name", None) != "batch_norm" or len(args) < 8:
        return False
    if args[5] is not False:
        return False
    if not _perturbed_parent_is_only_running_var(layer, layers_to_perturb):
        return False
    x, weight, mean, var, eps = args[0], args[1], args[3], args[4], args[7]
    if not isinstance(x, torch.Tensor) or not isinstance(mean, torch.Tensor):
        return False
    if not x.is_floating_point() or not _saved_output_all_finite(layer):
        return False
    if not _input_at_running_mean(x.detach(), mean.detach(), weight):
        return False
    if not _running_mean_zero_where_weighted(mean.detach(), weight):
        return False
    return _perturbed_running_var_stays_in_domain(layer, var, eps, weight, x)
