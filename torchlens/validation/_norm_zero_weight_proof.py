"""Value proof for a normalization input annihilated by an all-zero affine weight.

``group_norm``, ``layer_norm`` and ``rms_norm`` all end in an elementwise
affine ``normalized(input) * weight (+ bias)``. An exactly zero ``weight``
annihilates the whole normalized term, so the output equals ``bias`` (or zero)
for every finite input: the input edge cannot reach the output value. This is
the zero-annihilator proof ``exemptions._check_norm_zero_weight_annihilates``
already gives ``batch_norm``/``instance_norm``, applied to the normalizations
whose weight sits at ``args[2]`` (``F.group_norm(input, num_groups, weight,
bias, eps)``, ``F.layer_norm(input, normalized_shape, weight, bias, eps)``,
``F.rms_norm(input, normalized_shape, weight, eps)``). timm's
``zero_init_last`` zero-initializes the last GroupNorm of every residual block
(``resnet50_gn``, ``regnety_040_sgn``, ``vit_small_r26_s32_224``), so an
untrained instance hits this at validation time. Split from
``validation/exemptions.py`` (R43 file-size ratchet).

Narrow by construction: only a perturbed parent that occupies the input slot
(``args[0]``) and no other slot is ever exempted; perturbing ``weight`` or
``bias`` never is, a missing (non-affine) weight never proves anything, and a
non-finite saved output keeps the check strict (``0 * inf`` is NaN, so the
input would reach the output).
"""

from typing import Any

import torch

from ._value_predicates import _is_all_zero_value, _saved_output_all_finite

# Normalizations of the form ``normalized(args[0]) * weight (+ bias)`` whose
# affine weight is ``args[2]`` (or the ``weight`` keyword).
NORM_ZERO_WEIGHT_FUNC_NAMES = frozenset({"group_norm", "layer_norm", "rms_norm"})
_WEIGHT_POSITION = 2


def _norm_weight_operand(layer: Any, args: tuple[Any, ...]) -> Any:
    """Return the saved affine ``weight`` operand, or ``None`` when absent.

    Parameters
    ----------
    layer:
        Normalization op being classified.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    Any
        ``args[2]`` when present, else the saved ``weight`` keyword, else ``None``.
    """

    if len(args) > _WEIGHT_POSITION:
        return args[_WEIGHT_POSITION]
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    return kwargs.get("weight")


def _perturbed_parent_is_only_the_input(layer: Any, layers_to_perturb: list[str]) -> bool:
    """Return whether the single perturbed parent feeds ``args[0]`` and nothing else.

    Parameters
    ----------
    layer:
        Normalization op being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    bool
        True only for one perturbed label recorded at positional slot 0 alone;
        any other positional or keyword occurrence (``x`` also used as weight)
        keeps the check strict.
    """

    # Reuse exemptions' parent-position readers (deferred: exemptions imports
    # this module) so the proof adds no new ``parent_arg_positions`` read site.
    # The first requires exactly one perturbed label at positional slot 0 and
    # nowhere else among the args; the second (no keyword spellings allowed)
    # rejects any keyword occurrence.
    from .exemptions import (
        _perturbed_parent_arg_positions,
        _perturbed_parents_only_occupy_template_slot,
    )

    if _perturbed_parent_arg_positions(layer, layers_to_perturb) != {0}:
        return False
    return _perturbed_parents_only_occupy_template_slot(
        layer, layers_to_perturb, template_arg_roots=(0,), template_kwarg_names=()
    )


def norm_input_annihilated_by_zero_weight(
    layer: Any,
    layers_to_perturb: list[str],
    args: tuple[Any, ...],
) -> bool:
    """Return whether a normalization's input provably cannot reach its output.

    Parameters
    ----------
    layer:
        ``group_norm``/``layer_norm``/``rms_norm`` op whose unchanged perturbed
        replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    bool
        True when the perturbed parent is only the input, the saved output is
        finite, and the saved affine weight is a non-empty all-zero tensor.
    """

    if getattr(layer, "func_name", None) not in NORM_ZERO_WEIGHT_FUNC_NAMES:
        return False
    if not _perturbed_parent_is_only_the_input(layer, layers_to_perturb):
        return False
    if not _saved_output_all_finite(layer):
        return False
    weight = _norm_weight_operand(layer, args)
    return isinstance(weight, torch.Tensor) and _is_all_zero_value(weight)
