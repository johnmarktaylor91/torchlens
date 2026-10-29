"""Value proofs for ``scaled_dot_product_attention`` operands that cannot reach the output.

``scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)`` computes
``softmax(q @ k^T * scale + mask) @ v``. Both proofs read only saved values, compare with
``==`` (no tolerance), and run only after every perturbed replay left the output unchanged.
Both need every softmax row to be well defined: saved ``q`` and ``k`` finite, ``dropout_p``
zero, ``scale`` absent or finite, and a mask that leaves every query row at least one key
(a bool mask with a True in every row, or a float mask with no NaN or +inf and a finite entry
in every row; a causal mask always keeps key 0). Outside that precondition neither proof
applies.

``sdpa_zero_value_output``: with ``v`` all zero, ``P @ v`` is zero for every finite attention
matrix ``P``, so neither ``q`` nor ``k`` can reach the output. Exempted only for a perturbed
query (slot 0) or key (slot 1), with the saved output all zero.

``sdpa_single_key_identity``: with one key (``S = 1``), softmax over that key is exactly 1 for
every finite score, so the output equals ``v`` whatever the query is. Exempted only for a
perturbed query (slot 0), with the saved output equal to ``v`` broadcast over the query rows.
A perturbed key or value stays strict.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from ._operand_value_proofs import sole_perturbed_slot
from ._value_predicates import _is_all_zero_value

if TYPE_CHECKING:
    from .exemptions import PosthocPerturbDecision

_SDPA_FUNC_NAME = "scaled_dot_product_attention"
#: Positional index and default of each sdpa keyword the proofs read.
_SDPA_PARAMS = {"attn_mask": (3, None), "dropout_p": (4, 0.0), "scale": (6, None)}


def _sdpa_param(layer: Any, args: tuple[Any, ...], name: str) -> Any:
    """Return an sdpa parameter from the saved keywords, else its position, else its default."""

    kwargs = getattr(layer, "saved_kwargs", None) or {}
    index, default = _SDPA_PARAMS[name]
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else default


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` is a real, finite, non-bool number."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_finite_float_tensor(value: Any) -> bool:
    """Return whether ``value`` is a non-empty floating tensor with no NaN or Inf."""

    return (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and value.numel() > 0
        and bool(torch.isfinite(value).all())
    )


def _mask_keeps_a_key_in_every_row(mask: Any) -> bool:
    """Return whether an attention mask leaves every query row at least one key.

    Parameters
    ----------
    mask:
        Saved ``attn_mask``: ``None``, a bool mask (True keeps a key) or an additive float
        mask.

    Returns
    -------
    bool
        True for ``None``, for a bool mask with a True in every row, and for a float mask
        with no NaN or +inf and a finite entry in every row.
    """

    if mask is None:
        return True
    if not isinstance(mask, torch.Tensor) or mask.ndim == 0 or mask.numel() == 0:
        return False
    if mask.dtype == torch.bool:
        return bool(mask.any(dim=-1).all())
    if not mask.is_floating_point():
        return False
    if bool(torch.isnan(mask).any()) or bool(torch.isposinf(mask).any()):
        return False
    return bool(torch.isfinite(mask).any(dim=-1).all())


def sdpa_softmax_rows_well_defined(layer: Any, args: tuple[Any, ...]) -> bool:
    """Return whether every softmax row of a saved sdpa call is well defined.

    Parameters
    ----------
    layer:
        ``scaled_dot_product_attention`` op.
    args:
        Saved positional arguments ``(q, k, v, ...)``.

    Returns
    -------
    bool
        True when q and k are finite, ``dropout_p`` is zero, ``scale`` is absent or finite,
        and the mask keeps a key in every query row.
    """

    if not (_is_finite_float_tensor(args[0]) and _is_finite_float_tensor(args[1])):
        return False
    dropout_p = _sdpa_param(layer, args, "dropout_p")
    if not _is_finite_number(dropout_p) or dropout_p != 0:
        return False
    scale = _sdpa_param(layer, args, "scale")
    if scale is not None and not _is_finite_number(scale):
        return False
    return _mask_keeps_a_key_in_every_row(_sdpa_param(layer, args, "attn_mask"))


def sdpa_zero_value_output(layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]) -> bool:
    """Return whether a perturbed query or key meets an all-zero value operand.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments ``(q, k, v, ...)``.

    Returns
    -------
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) != _SDPA_FUNC_NAME or len(args) < 3:
        return False
    if sole_perturbed_slot(layer, layers_to_perturb) not in (0, 1):
        return False
    if not _is_all_zero_value(args[2]) or not _is_all_zero_value(getattr(layer, "out", None)):
        return False
    return sdpa_softmax_rows_well_defined(layer, args)


def _output_equals_value_rows(out: Any, value: Any) -> bool:
    """Return whether every query row of ``out`` equals the single value row exactly."""

    if not isinstance(out, torch.Tensor) or not isinstance(value, torch.Tensor):
        return False
    if out.numel() == 0 or value.ndim < 2 or value.shape[-2] != 1:
        return False
    try:
        if torch.broadcast_shapes(value.shape, out.shape) != out.shape:
            return False
    except RuntimeError:
        return False
    return bool((out == value).all())


def sdpa_single_key_identity(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> bool:
    """Return whether a perturbed query attends over exactly one key.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments ``(q, k, v, ...)``.

    Returns
    -------
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) != _SDPA_FUNC_NAME or len(args) < 3:
        return False
    if sole_perturbed_slot(layer, layers_to_perturb) != 0:
        return False
    key = args[1]
    if not isinstance(key, torch.Tensor) or key.ndim < 2 or key.shape[-2] != 1:
        return False
    if not sdpa_softmax_rows_well_defined(layer, args):
        return False
    return _output_equals_value_rows(getattr(layer, "out", None), args[2])


def sdpa_value_proof_decision(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Return the decision of the first sdpa value proof that applies.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        Exempt decision for a proved case, otherwise non-exempt.
    """

    from .exemptions import PosthocPerturbDecision

    if sdpa_zero_value_output(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "sdpa_zero_value_output",
            "saved value is all zero and every softmax row is well defined, so the attention "
            "output is zero for any finite query and key",
        )
    if sdpa_single_key_identity(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "sdpa_single_key_identity",
            "attention runs over a single key, so softmax is exactly 1 and the output equals "
            "value for any finite query",
        )
    return PosthocPerturbDecision(False, "no_sdpa_value_proof")
