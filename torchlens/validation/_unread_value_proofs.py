"""Posthoc value proofs for writes that cannot change the output and unread arguments.

Each decision runs only after every perturbed replay left an op's output
unchanged, reads the SAVED call arguments and output with no tolerance, and
exempts exactly one perturbed parent in the named slot of the named op:

- ``__setitem__`` empty write: the perturbed parent is an integer index
  component and another component is a bool mask with no True entry, so no
  element is written for any value of the index;
- ``__setitem__`` held value: the perturbed parent is an index component, the
  replacement is one finite value ``c``, and every destination element ANY
  in-domain choice of that component could address already equals ``c``;
- ``clamp``/``clip`` with equal finite scalar bounds: the output is the bound
  for every input, so the input (``args[0]``) cannot reach it;
- ``spmm`` with an empty sparse matrix (no stored entries): the dense operand
  (``args[1]``) only multiplies nothing;
- ``masked_scatter`` with an all-True mask: every output element comes from the
  source, so the destination (``args[0]``) is overwritten everywhere;
- ``embedding_bag`` in sum mode: the ``bag_size`` and ``max_indices`` outputs are
  all-zero placeholders the indices (``args[1]``) never reach;
- ``deform_conv2d`` with ``use_mask`` False: the kernel never reads the mask
  (``args[3]``), the zeros placeholder ``torchvision.ops.deform_conv2d`` builds.

Every decision also checks the saved output against the value the proof
predicts, so a replay that drops the write or a forged output falls through.
Dispatched from ``_saved_value_proofs.saved_value_proof_decision``.
"""

from __future__ import annotations

import math
from numbers import Number
from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member
from .._utils.tensor_utils import tensor_nanequal
from ._posthoc_decision import PosthocPerturbDecision

if TYPE_CHECKING:
    from .._data_classes.op import Op

_NOT_PROVED = PosthocPerturbDecision(False, "not_value_proved")
_INDEX_DTYPES = (torch.int32, torch.int64)
_CLAMP_FUNC_NAMES = frozenset({"clamp", "clamp_", "clip", "clip_"})
_MASKED_SCATTER_FUNC_NAMES = frozenset({"masked_scatter", "masked_scatter_"})
_EMBEDDING_BAG_SUM_MODE = 0
_EMBEDDING_BAG_MODE_SLOT = 4
_EMBEDDING_BAG_PLACEHOLDER_OUTPUTS = frozenset({2, 3})
_DEFORM_USE_MASK_SLOT = 13


def _setitem_component_positions(index: Any, keys: set[Any]) -> list[int] | None:
    """Return the index-component positions the perturbed parent fills, or None.

    ``1`` is a bare index (position 0); ``(1, k)`` is component ``k`` of a tuple
    index. The destination (``0``) and value (``2``) slots are never index keys.
    """

    positions: list[int] = []
    for key in keys:
        if key == 1 and not isinstance(index, tuple):
            positions.append(0)
        elif isinstance(key, tuple) and len(key) == 2 and key[0] == 1:
            positions.append(key[1])
        else:
            return None
    return positions or None


def _setitem_parts(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> tuple[torch.Tensor, tuple[Any, ...], int] | None:
    """Return (destination, components, perturbed position) for one perturbed index slot."""

    if len(args) < 3 or not isinstance(args[0], torch.Tensor):
        return None
    positions = _setitem_component_positions(args[1], keys)
    if positions is None or len(positions) != 1:
        return None
    components = args[1] if isinstance(args[1], tuple) else (args[1],)
    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor) or out.shape != args[0].shape:
        return None
    if not tensor_nanequal(out, args[0].to(out.device), allow_tolerance=False):
        return None
    return args[0], components, positions[0]


def _setitem_empty_write_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt an integer index component when another component is an all-False mask."""

    parts = _setitem_parts(layer, keys, args)
    if parts is None:
        return _NOT_PROVED
    _, components, position = parts
    perturbed = components[position]
    if not isinstance(perturbed, torch.Tensor) or perturbed.dtype not in _INDEX_DTYPES:
        return _NOT_PROVED
    empty_mask = any(
        isinstance(component, torch.Tensor)
        and component.dtype == torch.bool
        and component.numel() > 0
        and not bool(component.any().item())
        for at, component in enumerate(components)
        if at != position
    )
    if not empty_mask:
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "setitem_empty_write",
        "another index component is a bool mask with no True entry, so the write "
        "selects no element for any value of the perturbed integer index",
    )


def _uniform_finite_value(value: Any) -> float | None:
    """Return the one finite value a scalar or uniform tensor holds, else None."""

    if isinstance(value, bool) or not isinstance(value, (Number, torch.Tensor)):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or value.is_complex() or value.is_sparse:
            return None
        flat = value.reshape(-1)
        if not bool(torch.all(torch.eq(flat, flat[0])).item()):
            return None
        value = flat[0].item()
    if isinstance(value, complex):
        return None
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _setitem_held_value_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt an index component when every cell it could address already holds the value.

    The perturbed component is replaced by full slices over the dims it
    addresses (one for an integer component, ``mask.ndim`` for a bool mask) with
    the other components fixed; that region holds every cell any in-domain
    choice can write, and it must equal the written value exactly.
    """

    parts = _setitem_parts(layer, keys, args)
    held = _uniform_finite_value(args[2]) if len(args) > 2 else None
    if parts is None or held is None:
        return _NOT_PROVED
    destination, components, position = parts
    perturbed = components[position]
    if not isinstance(perturbed, torch.Tensor):
        return _NOT_PROVED
    if perturbed.dtype == torch.bool:
        widened: tuple[Any, ...] = (slice(None),) * perturbed.ndim
    elif perturbed.dtype in _INDEX_DTYPES:
        widened = (slice(None),)
    else:
        return _NOT_PROVED
    region_index = (*components[:position], *widened, *components[position + 1 :])
    try:
        region = destination[region_index]
    except (IndexError, RuntimeError, TypeError, ValueError):
        return _NOT_PROVED
    if region.numel() == 0 or not bool(torch.all(torch.eq(region, held)).item()):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "setitem_destination_holds_value",
        "every destination element any in-domain choice of the perturbed index can "
        "address already equals the written value, and the saved output equals the "
        "saved destination",
    )


def _setitem_write_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Try the empty-write proof, then the held-value proof."""

    decision = _setitem_empty_write_decision(layer, keys, args)
    if decision.exempt:
        return decision
    return _setitem_held_value_decision(layer, keys, args)


def _scalar_bound(value: Any) -> float | None:
    """Return a finite scalar or 0-d tensor bound as a float, else None."""

    if isinstance(value, torch.Tensor) and value.ndim != 0:
        return None
    return _uniform_finite_value(value)


def _clamp_equal_bounds_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt the input of a clamp whose saved scalar bounds are equal and finite."""

    if keys != {0} or not isinstance(args[0] if args else None, torch.Tensor):
        return _NOT_PROVED
    kwargs = record_member(layer, "saved_kwargs", None) or {}
    low = args[1] if len(args) > 1 else kwargs.get("min")
    high = args[2] if len(args) > 2 else kwargs.get("max")
    low_value, high_value = _scalar_bound(low), _scalar_bound(high)
    if low_value is None or high_value is None or low_value != high_value:
        return _NOT_PROVED
    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor) or out.numel() == 0:
        return _NOT_PROVED
    if not bool(torch.all(torch.eq(out, high_value)).item()):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "clamp_equal_bounds",
        "the saved clamp bounds are equal finite scalars, so every input maps to the "
        "bound, and the saved output equals it everywhere",
    )


def _spmm_empty_sparse_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt the dense operand of an ``spmm`` whose sparse operand stores no entry."""

    if keys != {1} or len(args) < 2:
        return _NOT_PROVED
    sparse, dense, out = args[0], args[1], getattr(layer, "out", None)
    if not isinstance(sparse, torch.Tensor) or not isinstance(dense, torch.Tensor):
        return _NOT_PROVED
    if not isinstance(out, torch.Tensor):
        return _NOT_PROVED
    if sparse.layout == torch.strided or dense.layout != torch.strided:
        return _NOT_PROVED
    if sparse._nnz() != 0 or out.layout != torch.strided or out.numel() == 0:
        return _NOT_PROVED
    if not bool(torch.all(torch.eq(out, 0)).item()):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "spmm_empty_sparse_operand",
        "the saved sparse operand stores no entry, so the product is zero for every "
        "dense operand, and the saved output is all zero",
    )


def _masked_scatter_all_true_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt the destination of a ``masked_scatter`` whose saved mask is True everywhere."""

    if keys != {0} or len(args) < 3:
        return _NOT_PROVED
    mask, source, out = args[1], args[2], getattr(layer, "out", None)
    if not isinstance(mask, torch.Tensor) or not isinstance(source, torch.Tensor):
        return _NOT_PROVED
    if not isinstance(out, torch.Tensor):
        return _NOT_PROVED
    if mask.dtype != torch.bool or out.numel() == 0 or source.numel() < out.numel():
        return _NOT_PROVED
    try:
        full_mask = torch.broadcast_to(mask, out.shape)
    except RuntimeError:
        return _NOT_PROVED
    if not bool(full_mask.all().item()):
        return _NOT_PROVED
    expected = source.reshape(-1)[: out.numel()].reshape(out.shape).to(out.device)
    if expected.dtype != out.dtype or not tensor_nanequal(out, expected, allow_tolerance=False):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "masked_scatter_all_true_mask",
        "the saved mask is True at every output element, so every element comes from "
        "the source, and the saved output equals the source in order",
    )


def _embedding_bag_sum_placeholder_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt the indices of a sum-mode ``embedding_bag``'s zero placeholder outputs."""

    if keys != {1} or len(args) <= _EMBEDDING_BAG_MODE_SLOT:
        return _NOT_PROVED
    if getattr(layer, "multi_output_index", None) not in _EMBEDDING_BAG_PLACEHOLDER_OUTPUTS:
        return _NOT_PROVED
    mode = args[_EMBEDDING_BAG_MODE_SLOT]
    if isinstance(mode, bool) or not isinstance(mode, int) or mode != _EMBEDDING_BAG_SUM_MODE:
        return _NOT_PROVED
    out = getattr(layer, "out", None)
    if not isinstance(out, torch.Tensor) or out.numel() == 0:
        return _NOT_PROVED
    if not bool(torch.all(torch.eq(out, 0)).item()):
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "embedding_bag_sum_placeholder_output",
        "in sum mode embedding_bag returns bag_size and max_indices as all-zero "
        "placeholders the index values never reach, and the saved output is all zero",
    )


def _deform_conv2d_unused_mask_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Exempt the mask of a ``deform_conv2d`` call whose saved ``use_mask`` is False."""

    if keys != {3} or len(args) <= _DEFORM_USE_MASK_SLOT:
        return _NOT_PROVED
    if args[_DEFORM_USE_MASK_SLOT] is not False:
        return _NOT_PROVED
    return PosthocPerturbDecision(
        True,
        "deform_conv2d_unused_mask",
        "the saved use_mask flag is False, so the kernel never reads the mask argument",
    )


_DECISIONS_BY_FAMILY = (
    (frozenset({"__setitem__"}), _setitem_write_decision),
    (_CLAMP_FUNC_NAMES, _clamp_equal_bounds_decision),
    (frozenset({"spmm"}), _spmm_empty_sparse_decision),
    (_MASKED_SCATTER_FUNC_NAMES, _masked_scatter_all_true_decision),
    (frozenset({"embedding_bag"}), _embedding_bag_sum_placeholder_decision),
    (frozenset({"deform_conv2d"}), _deform_conv2d_unused_mask_decision),
)


def unread_value_proof_decision(
    layer: Op, keys: set[Any], args: tuple[Any, ...]
) -> PosthocPerturbDecision:
    """Dispatch to the unread-value proof for ``layer``'s op family.

    Parameters
    ----------
    layer:
        Op whose unchanged perturbed replay is being classified.
    keys:
        Positional keys the single perturbed parent occupies.
    args:
        Saved positional arguments for ``layer``.

    Returns
    -------
    PosthocPerturbDecision
        The family's decision, or non-exempt when no family matches.
    """

    for func_names, decide in _DECISIONS_BY_FAMILY:
        if getattr(layer, "func_name", None) in func_names:
            return decide(layer, keys, args)
    return _NOT_PROVED
