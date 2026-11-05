"""Pre-perturbation checks for argument slots whose value is pinned by structure.

Each check is keyed on one op family and one argument slot, and excuses only a
parent whose every alternative value is invalid for the call (so a perturbed
replay can only raise), never a parent whose value feeds the output:

- packed ``gru`` / ``rnn_tanh`` / ``rnn_relu``: the ``batch_sizes`` slot
  (``args[1]`` of the packed overload), the packed sequence's layout metadata,
  as ``_check_lstm_exempt`` already excuses for ``lstm``;
- ``expand``: a size at a non-singleton input dim, which must equal the input's
  extent there (the saved size is checked to be exactly that extent);
- ``__setitem__``: a bool mask at any slot of the index tuple (Case 2 of
  ``_check_setitem_exempt`` reads only the first slot).

Registered in ``exemptions.CUSTOM_EXEMPTION_CHECKS`` (Menagerie rung 4, round 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

PACKED_RNN_FUNC_NAMES = ("gru", "rnn_tanh", "rnn_relu")


def _arg_keys_of(layer: Op, layers_to_perturb: list[str]) -> list[Any] | None:
    """Return the positional keys of the single perturbed parent, or None to fail closed."""

    if len(layers_to_perturb) != 1:
        return None
    positions = record_member(layer, "parent_arg_positions", None) or {}
    label = layers_to_perturb[0]
    if label in (positions.get("kwargs", {}) or {}).values():
        return None
    keys = [key for key, parent in (positions.get("args", {}) or {}).items() if parent == label]
    return keys or None


def check_packed_rnn_batch_sizes_exempt(
    self: Trace, layer: Op, layers_to_perturb: list[str]
) -> bool:
    """Exempt only the ``batch_sizes`` slot of a packed-sequence RNN call."""

    del self
    keys = _arg_keys_of(layer, layers_to_perturb)
    args = tuple(record_member(layer, "saved_args", ()) or ())
    if keys != [1] or len(args) < 4:
        return False
    data, batch_sizes, params = args[0], args[1], args[3]
    return (
        isinstance(data, torch.Tensor)
        and data.ndim == 2
        and isinstance(batch_sizes, torch.Tensor)
        and batch_sizes.dtype == torch.int64
        and batch_sizes.ndim == 1
        and isinstance(params, (list, tuple))
    )


def _expand_size_slot(args: tuple[Any, ...], key: Any) -> tuple[Any, int, int] | None:
    """Return ``(size value, output dim, number of sizes)`` for an ``expand`` size key."""

    if isinstance(key, int) and key >= 1:
        sizes: tuple[Any, ...] = args[1:]
        dim = key - 1
    elif isinstance(key, tuple) and len(key) == 2 and key[0] == 1:
        if len(args) < 2 or not isinstance(args[1], (list, tuple)):
            return None
        sizes = tuple(args[1])
        dim = key[1]
    else:
        return None
    if not isinstance(dim, int) or not 0 <= dim < len(sizes):
        return None
    return sizes[dim], dim, len(sizes)


def check_expand_pinned_size_exempt(self: Trace, layer: Op, layers_to_perturb: list[str]) -> bool:
    """Exempt an ``expand`` size at a non-singleton input dim equal to that dim's extent."""

    del self
    keys = _arg_keys_of(layer, layers_to_perturb)
    args = tuple(record_member(layer, "saved_args", ()) or ())
    if keys is None or len(keys) != 1 or not args or not isinstance(args[0], torch.Tensor):
        return False
    slot = _expand_size_slot(args, keys[0])
    if slot is None:
        return False
    size, dim, n_sizes = slot
    input_dim = dim - (n_sizes - args[0].ndim)
    if input_dim < 0:
        return False
    extent = int(args[0].shape[input_dim])
    if extent == 1:
        return False
    if isinstance(size, torch.Tensor):
        if size.numel() != 1 or size.dtype.is_floating_point or size.dtype.is_complex:
            return False
        size = int(size.item())
    return type(size) is int and size == extent


def setitem_bool_mask_at_index_slot(layer: Op, layers_to_perturb: list[str]) -> bool:
    """Return whether the perturbed parent is only a bool mask in the index tuple whose
    True-count the value tensor's shape pins.

    A mask is a real value input (a different mask writes different elements), so the
    slot is structural only when the call would REJECT a mask with another True-count:
    the value must broadcast to the selection of the saved mask and not to the selection
    of that mask with one element flipped. A scalar or broadcast value fits every mask
    and keeps the mask edge under perturbation.
    """

    keys = _arg_keys_of(layer, layers_to_perturb)
    args = tuple(record_member(layer, "saved_args", ()) or ())
    if keys is None or len(args) < 3 or not isinstance(args[1], tuple):
        return False
    destination, index, value = args[0], args[1], args[2]
    if not isinstance(destination, torch.Tensor) or not isinstance(value, torch.Tensor):
        return False
    slots: list[int] = []
    for key in keys:
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 1):
            return False
        slot = key[1]
        if not isinstance(slot, int) or not 0 <= slot < len(index):
            return False
        mask = index[slot]
        if not (isinstance(mask, torch.Tensor) and mask.dtype == torch.bool):
            return False
        slots.append(slot)
    return all(_value_shape_pins_mask(destination, index, slot, value) for slot in slots)


def _value_shape_pins_mask(
    destination: torch.Tensor, index: tuple[Any, ...], slot: int, value: torch.Tensor
) -> bool:
    """Return whether ``value`` fits ``index``'s selection and not the selection of the
    same index with one element of the mask at ``slot`` flipped.

    The probe is a zero-strided expansion of a scalar, so indexing it allocates only the
    selection's shape, never a copy of the destination.
    """

    mask = index[slot]
    if mask.numel() == 0:
        return False
    flipped = mask.clone()
    flipped.view(-1)[0] = ~flipped.view(-1)[0]
    alternative = index[:slot] + (flipped,) + index[slot + 1 :]
    probe = torch.zeros((), dtype=torch.bool, device=destination.device).expand(destination.shape)
    try:
        selected_shape = tuple(probe[index].shape)
        alternative_shape = tuple(probe[alternative].shape)
        if tuple(torch.broadcast_shapes(value.shape, selected_shape)) != selected_shape:
            return False
    except (IndexError, RuntimeError, TypeError, ValueError):
        return False
    try:
        fits_alternative = (
            tuple(torch.broadcast_shapes(value.shape, alternative_shape)) == alternative_shape
        )
    except RuntimeError:
        # The flipped mask rejects the value: its True-count is pinned.
        return True
    return not fits_alternative
