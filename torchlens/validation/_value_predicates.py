"""Saved-value predicates used by the posthoc value-proof decisions.

Whole-tensor facts (all zero, all inf, all NaN, all finite, constant along a
dim) that ``exemptions.py`` reads off saved call arguments and outputs when it
proves a perturbation cannot move an output.

Split from ``validation/exemptions.py`` along this seam (R43 file-size ratchet);
the predicates are unchanged.
"""

from numbers import Number
from typing import Any

import torch

from ..data_classes.op import Op
from ..utils.tensor_utils import tensor_all_nan


def _tensor_or_number_is_constant(value: Any) -> bool:
    """Return whether ``value`` is a scalar or a tensor with one constant value.

    Parameters
    ----------
    value:
        Saved scatter source argument (tensor or Python scalar).

    Returns
    -------
    bool
        True when every element provably equals one constant.
    """

    if isinstance(value, Number):
        return True
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return False
    first = value.reshape(-1)[0]
    return bool(torch.eq(value, first).all().item())


def _tensor_constant_along_dim(tensor: torch.Tensor, dim: int) -> bool:
    """Return whether ``tensor`` holds identical values at every index of ``dim``.

    Parameters
    ----------
    tensor:
        Source tensor being indexed.
    dim:
        Normalized dimension the index selects along.

    Returns
    -------
    bool
        True when swapping any two positions along ``dim`` provably leaves the
        tensor unchanged.
    """

    if tensor.numel() == 0 or tensor.shape[dim] == 0:
        return False
    reference = tensor.select(dim, 0).unsqueeze(dim)
    return bool(torch.eq(tensor, reference).all().item())


def _saved_output_all_finite(layer: Op) -> bool:
    """Return whether the op's saved output is a non-empty, all-finite tensor.

    Annihilator and degenerate-shape proofs claim the output is constant in the
    perturbed parent. That identity holds only for finite operands (``0 * inf``
    and ``inf - inf`` are NaN): a non-finite saved operand makes the ORIGINAL
    output non-finite, so a correctly wired edge WOULD change the output under a
    finite perturbation, and an unchanged output is evidence of a capture bug,
    never something to exempt. A finite saved output proves every operand that
    reached it through the annihilated term was finite on this capture.

    Parameters
    ----------
    layer:
        Op whose saved output is inspected.

    Returns
    -------
    bool
        True when ``layer.out`` is a non-empty tensor with no NaN or Inf.
    """

    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor) or out.numel() == 0:
        return False
    return bool(torch.isfinite(out).all().item())


def _is_all_zero_value(value: Any) -> bool:
    """Return whether ``value`` is provably an all-zero tensor/scalar.

    Parameters
    ----------
    value:
        Candidate scalar or tensor value.

    Returns
    -------
    bool
        True when ``value`` can be inspected and every element is zero.
    """

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    if value.numel() == 0:
        return False
    return bool(torch.all(torch.eq(value, 0)).item())


def _is_all_inf_value(value: Any) -> bool:
    """Return whether ``value`` is provably an all-infinite tensor/scalar.

    Parameters
    ----------
    value:
        Candidate scalar or tensor value.

    Returns
    -------
    bool
        True when ``value`` can be inspected and every element is infinite.
    """

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    if value.numel() == 0:
        return False
    return bool(torch.all(torch.isinf(value)).item())


def _is_all_nan_value(value: Any) -> bool:
    """Return whether ``value`` is provably an all-NaN tensor/scalar.

    Parameters
    ----------
    value:
        Candidate scalar or tensor value.

    Returns
    -------
    bool
        True when ``value`` can be inspected and every element is NaN.
    """

    if not isinstance(value, torch.Tensor):
        try:
            value = torch.tensor(value)
        except (TypeError, ValueError, RuntimeError):
            return False
    if value.numel() == 0:
        return False
    return tensor_all_nan(value)


def _extrema_operand_dominates(
    func_name: str, other: torch.Tensor, perturbed: torch.Tensor
) -> bool:
    """Return whether ``other`` wins the extrema at every element.

    Parameters
    ----------
    func_name:
        ``max``/``maximum`` or ``min``/``minimum``; any other name is not proven.
    other:
        Saved non-perturbed operand.
    perturbed:
        Saved perturbed operand.

    Returns
    -------
    bool
        True when ``other`` is at least (for max) or at most (for min) the
        perturbed operand everywhere; False when unproven or not comparable.
    """

    try:
        if func_name in ("max", "maximum"):
            return bool(torch.all(other >= perturbed).item())
        if func_name in ("min", "minimum"):
            return bool(torch.all(other <= perturbed).item())
    except RuntimeError:
        return False
    return False
