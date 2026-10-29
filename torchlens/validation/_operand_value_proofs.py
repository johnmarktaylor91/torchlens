"""Value proofs for calls whose perturbed operand provably cannot reach the output.

Each proof reads only the op's saved arguments and saved output, compares with ``==`` (no
tolerance), and is consulted only after every perturbed replay left the output unchanged (the
posthoc tier). Each exempts exactly one perturbed parent that occupies exactly one positional
slot and no keyword slot; a parent that also feeds another slot feeds real values and stays
strict.

``einsum_zero_operand``: every output element of ``torch.einsum`` is a sum of products that
take exactly one element from every operand. When a non-perturbed operand is all zero, every
product is zero, so the output is zero for every finite value of the perturbed operand. The
saved output must be all zero too, which refuses a non-finite perturbed operand (``0 * inf``
is NaN), the same guard the matmul zero-operand proof uses.

``scatter_reduce_destination_overwritten``: ``self.scatter_reduce(dim, index, src, reduce,
include_self=False)`` writes, at every position the index addresses, the reduction of the
scattered ``src`` values alone; positions never addressed keep ``self``. When the index
addresses every destination position, the output does not depend on ``self``.

``grid_sampler_zero_input``: every grid-sampler mode (bilinear, nearest, bicubic) and padding
mode returns, per output element, a combination of input elements with weights that depend
only on the grid. An all-zero input makes every term zero for every finite grid.

``conv_zero_input_output_is_bias``: a convolution of an all-zero input is a sum of products
with a zero factor, so the output is exactly the bias (zero without one) for every finite
weight; a finite saved output equal to the broadcast bias proves it on this capture.

``or_absorbing_true_operand``: ``a | b`` is True wherever ``b`` is True, so a bool operand that
is all True fixes the output to all True whatever the other operand holds.

``pow_unit_base``: ``pow(1, y)`` is exactly 1 for every ``y``, NaN included (IEEE 754), so an
exponent cannot reach the output when the base is exactly 1 everywhere.

``empty_output_no_elements``: an output with no elements has no value a perturbation could
change. This widens the existing empty-parent skip (``core._check_perturbation_exemptions``)
and the empty ``__getitem__`` output proof to any op whose saved output is empty; the forward
replay still checks the empty output itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ._destination_coverage import _scatter_index_fully_overwrites_dim
from ._value_predicates import _is_all_zero_value, _saved_output_all_finite

if TYPE_CHECKING:
    from .exemptions import PosthocPerturbDecision

_GRID_SAMPLER_FUNC_NAMES = frozenset(
    {"grid_sampler", "grid_sample", "grid_sampler_2d", "grid_sampler_3d"}
)
#: Convolution spellings and their number of spatial dims.
_CONV_SPATIAL_DIMS = {"conv1d": 1, "conv2d": 2, "conv3d": 3}
_OR_FUNC_NAMES = frozenset({"__or__", "__ror__", "bitwise_or", "logical_or"})
#: pow spellings -> (base slot, exponent slot); ``x.__rpow__(b)`` computes ``b ** x``.
_POW_SLOTS = {"pow": (0, 1), "__pow__": (0, 1), "__rpow__": (1, 0)}


def sole_perturbed_slot(layer: Any, layers_to_perturb: list[str]) -> int | tuple[int, ...] | None:
    """Return the one positional slot the single perturbed parent occupies.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.

    Returns
    -------
    int | tuple[int, ...] | None
        The slot (an int, or a nested index path such as ``(1, 0)``), or ``None`` when more
        than one parent is perturbed, the parent fills a keyword slot, or it fills zero or
        several positional slots.
    """

    if len(layers_to_perturb) != 1:
        return None
    positions = getattr(layer, "parent_arg_positions", None) or {}
    label = layers_to_perturb[0]
    if any(parent == label for parent in (positions.get("kwargs") or {}).values()):
        return None
    slots = [slot for slot, parent in (positions.get("args") or {}).items() if parent == label]
    if len(slots) != 1:
        return None
    slot = slots[0]
    if isinstance(slot, int) and not isinstance(slot, bool):
        return slot
    if (
        isinstance(slot, tuple)
        and slot
        and all(isinstance(i, int) and not isinstance(i, bool) for i in slot)
    ):
        return slot
    return None


def output_has_no_elements(layer: Any) -> bool:
    """Return whether the op's saved output is a tensor with no elements.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.

    Returns
    -------
    bool
        True when the saved output is a tensor whose ``numel()`` is zero.
    """

    out = getattr(layer, "out", None)
    return isinstance(out, torch.Tensor) and out.numel() == 0


def _einsum_operand_paths(args: tuple[Any, ...]) -> list[tuple[int, ...]] | None:
    """Return nested paths of the tensor operands of an einsum call.

    Parameters
    ----------
    args:
        Saved positional arguments: ``(equation, *operands)`` or ``(equation, operands)``.

    Returns
    -------
    list[tuple[int, ...]] | None
        One path per operand, or ``None`` for any other call shape.
    """

    if len(args) < 2 or not isinstance(args[0], str):
        return None
    nested = len(args) == 2 and isinstance(args[1], (list, tuple))
    operands = list(args[1]) if nested else list(args[1:])
    if len(operands) < 2 or not all(isinstance(item, torch.Tensor) for item in operands):
        return None
    if nested:
        return [(1, i) for i in range(len(operands))]
    return [(1 + i,) for i in range(len(operands))]


def _value_at(args: tuple[Any, ...], path: tuple[int, ...]) -> Any:
    """Return the saved argument at a nested index path."""

    value: Any = args
    for index in path:
        value = value[index]
    return value


def einsum_zero_operand(layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]) -> bool:
    """Return whether a non-perturbed einsum operand is all zero and the output is all zero.

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
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) != "einsum":
        return False
    if not _is_all_zero_value(getattr(layer, "out", None)):
        return False
    paths = _einsum_operand_paths(args)
    slot = sole_perturbed_slot(layer, layers_to_perturb)
    perturbed = (slot,) if isinstance(slot, int) else slot
    if paths is None or perturbed not in paths:
        return False
    return any(_is_all_zero_value(_value_at(args, path)) for path in paths if path != perturbed)


def scatter_reduce_destination_overwritten(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> bool:
    """Return whether ``include_self=False`` and the index addresses every destination slot.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments ``(self, dim, index, src, reduce, include_self)``.

    Returns
    -------
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) not in {"scatter_reduce", "scatter_reduce_"}:
        return False
    if len(args) < 4 or sole_perturbed_slot(layer, layers_to_perturb) != 0:
        return False
    kwargs = getattr(layer, "saved_kwargs", None) or {}
    include_self = kwargs.get("include_self", args[5] if len(args) > 5 else True)
    if include_self is not False:
        return False
    dest, dim, index = args[0], args[1], args[2]
    if not isinstance(dest, torch.Tensor) or not isinstance(index, torch.Tensor):
        return False
    if isinstance(dim, bool) or not isinstance(dim, int):
        return False
    if dim < 0:
        dim += dest.ndim
    if not 0 <= dim < dest.ndim:
        return False
    return _scatter_index_fully_overwrites_dim(dest, dim, index)


def grid_sampler_zero_input(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> bool:
    """Return whether a perturbed grid samples an all-zero input to an all-zero output.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments ``(input, grid, ...)``.

    Returns
    -------
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) not in _GRID_SAMPLER_FUNC_NAMES or len(args) < 2:
        return False
    if sole_perturbed_slot(layer, layers_to_perturb) != 1:
        return False
    grid = args[1]
    if not isinstance(grid, torch.Tensor) or not bool(torch.isfinite(grid).all()):
        return False
    return _is_all_zero_value(args[0]) and _is_all_zero_value(getattr(layer, "out", None))


def _conv_bias_view(layer: Any, args: tuple[Any, ...], out: torch.Tensor) -> torch.Tensor | None:
    """Return the conv bias shaped to broadcast over ``out`` (zeros without a bias)."""

    kwargs = getattr(layer, "saved_kwargs", None) or {}
    bias = kwargs.get("bias", args[2] if len(args) > 2 else None)
    channel_dim = out.ndim - _CONV_SPATIAL_DIMS[layer.func_name] - 1
    if channel_dim < 0:
        return None
    if bias is None:
        return torch.zeros((), dtype=out.dtype)
    if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
        return None
    if bias.numel() != out.shape[channel_dim]:
        return None
    shape = [1] * out.ndim
    shape[channel_dim] = bias.numel()
    return bias.reshape(shape)


def conv_zero_input_output_is_bias(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> bool:
    """Return whether a perturbed conv weight meets an all-zero input (output equals bias).

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments ``(input, weight, bias, ...)``.

    Returns
    -------
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) not in _CONV_SPATIAL_DIMS or len(args) < 2:
        return False
    if sole_perturbed_slot(layer, layers_to_perturb) != 1:
        return False
    if not _is_all_zero_value(args[0]) or not _saved_output_all_finite(layer):
        return False
    out = layer.out
    bias = _conv_bias_view(layer, args, out)
    return bias is not None and bool((out == bias).all())


def _is_all_true(value: Any) -> bool:
    """Return whether ``value`` is ``True`` or a non-empty bool tensor that is all True."""

    if value is True:
        return True
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.bool
        and value.numel() > 0
        and bool(value.all())
    )


def or_absorbing_true_operand(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> bool:
    """Return whether the non-perturbed operand of a bool ``or`` is all True.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments ``(a, b)``.

    Returns
    -------
    bool
        True when the proof applies.
    """

    if getattr(layer, "func_name", None) not in _OR_FUNC_NAMES or len(args) != 2:
        return False
    slot = sole_perturbed_slot(layer, layers_to_perturb)
    if slot not in (0, 1):
        return False
    return _is_all_true(args[1 - slot]) and _is_all_true(getattr(layer, "out", None))


def _is_exactly_one(value: Any) -> bool:
    """Return whether ``value`` is the number 1 or a non-empty non-bool tensor all equal to 1."""

    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 1
    return (
        isinstance(value, torch.Tensor)
        and value.dtype != torch.bool
        and value.numel() > 0
        and bool((value == 1).all())
    )


def pow_unit_base(layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]) -> bool:
    """Return whether a perturbed exponent meets a base that is exactly 1 everywhere.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    layers_to_perturb:
        Parent labels selected for perturbation.
    args:
        Saved positional arguments (``(base, exponent)``, or ``(exponent, base)`` for
        ``__rpow__``).

    Returns
    -------
    bool
        True when the proof applies.
    """

    slots = _POW_SLOTS.get(getattr(layer, "func_name", None) or "")
    if slots is None or len(args) != 2:
        return False
    base_slot, exponent_slot = slots
    if sole_perturbed_slot(layer, layers_to_perturb) != exponent_slot:
        return False
    out = getattr(layer, "out", None)
    return (
        _is_exactly_one(args[base_slot]) and isinstance(out, torch.Tensor) and _is_exactly_one(out)
    )


def operand_value_proof_decision(
    layer: Any, layers_to_perturb: list[str], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Return the decision of the first operand value proof that applies.

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

    if output_has_no_elements(layer):
        return PosthocPerturbDecision(
            True,
            "empty_output_no_elements",
            "the output has no elements, so no perturbation can change its value",
        )
    if einsum_zero_operand(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "einsum_zero_operand",
            "a non-perturbed einsum operand is all zero, so every product term and the output "
            "are zero for any finite value of the perturbed operand",
        )
    if scatter_reduce_destination_overwritten(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "scatter_reduce_destination_overwritten",
            "scatter_reduce with include_self=False addresses every destination position, so "
            "the destination's values cannot reach the output",
        )
    if grid_sampler_zero_input(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "grid_sampler_zero_input",
            "the sampled input is all zero, so every finite grid samples to zero",
        )
    if conv_zero_input_output_is_bias(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "conv_zero_input_output_is_bias",
            "the convolution input is all zero, so the output is the bias for any finite weight",
        )
    if or_absorbing_true_operand(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "or_absorbing_true_operand",
            "the non-perturbed or operand is all True, so the output is all True",
        )
    if pow_unit_base(layer, layers_to_perturb, args):
        return PosthocPerturbDecision(
            True,
            "pow_unit_base",
            "the base is exactly 1 everywhere, so pow is exactly 1 for every exponent",
        )
    return PosthocPerturbDecision(False, "no_operand_value_proof")
