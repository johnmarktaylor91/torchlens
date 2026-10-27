"""Pre-perturbation checks for argument slots whose value is pinned by structure.

Each check is keyed on one op family and one argument slot, and excuses only a
parent whose every alternative value is invalid for the call (so a perturbed
replay can only raise), never a parent whose value feeds the output:

- packed ``gru`` / ``rnn_tanh`` / ``rnn_relu``: the ``batch_sizes`` slot
  (``args[1]`` of the packed overload), the packed sequence's layout metadata,
  as ``_check_lstm_exempt`` already excuses for ``lstm``;
- ``expand``: a size at a non-singleton input dim, which must equal the input's
  extent there (the saved size is checked to be exactly that extent).

Registered in ``exemptions.CUSTOM_EXEMPTION_CHECKS`` (Menagerie rung 4, round 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace

PACKED_RNN_FUNC_NAMES = ("gru", "rnn_tanh", "rnn_relu")


def _arg_keys_of(layer: Op, layers_to_perturb: list[str]) -> list[Any] | None:
    """Return the positional keys of the single perturbed parent, or None to fail closed."""

    if len(layers_to_perturb) != 1:
        return None
    positions = getattr(layer, "parent_arg_positions", None) or {}
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
    args = tuple(getattr(layer, "saved_args", ()) or ())
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
    args = tuple(getattr(layer, "saved_args", ()) or ())
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
