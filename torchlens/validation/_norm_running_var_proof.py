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
"""

from typing import Any

import torch

from ._value_predicates import _saved_output_all_finite


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

    if getattr(layer, "func_name", None) != "batch_norm" or len(args) < 6:
        return False
    if args[5] is not False:
        return False
    if not _perturbed_parent_is_only_running_var(layer, layers_to_perturb):
        return False
    x, weight, mean = args[0], args[1], args[3]
    if not isinstance(x, torch.Tensor) or not isinstance(mean, torch.Tensor):
        return False
    if not x.is_floating_point() or not _saved_output_all_finite(layer):
        return False
    return _input_at_running_mean(x.detach(), mean.detach(), weight)
