"""In-domain perturbations for parents bound by an op's structural contract.

Some integer parents feed an argument whose valid values form a narrow
structure rather than a range: ``embedding_bag`` offsets must start at 0 and
stay non-decreasing within the index count, and a traced ``view``/``reshape``
size must divide the element count left to a ``-1`` wildcard. Every generic
draw and every uniform +1/-1 step breaks that structure, so the perturbed
replay raised and the edge ended ``perturbation_execution_exception``
(unverified) even though the op plainly reads the value. The moves here keep
the call valid while changing what the op computes, so the replay result then
goes through the same sensitivity comparison as any other perturbation: a real
edge is validated, a spurious one still fails. No check is skipped; ``None``
means no valid alternate exists and the generic perturbation runs as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op

#: ``torch.embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode,
#: sparse, per_sample_weights, include_last_offset, padding_idx)`` slots.
_EMBEDDING_BAG_INDICES_SLOT = 1
_EMBEDDING_BAG_OFFSETS_SLOT = 2
_EMBEDDING_BAG_INCLUDE_LAST_SLOT = 7

#: Ops whose size arguments may be traced 0-d tensors next to a ``-1`` wildcard.
_RESHAPE_FUNCS = frozenset({"view", "reshape"})

_INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8})


def structural_domain_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return a valid alternate value for a structurally constrained parent slot.

    Parameters
    ----------
    layer:
        Child op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent value at that slot.

    Returns
    -------
    torch.Tensor or None
        The in-domain alternate, or ``None`` when the slot has no structural
        contract here or no valid alternate exists.
    """

    if parent_values.dtype not in _INTEGER_DTYPES:
        return None
    if layer.func_name == "embedding_bag" and _occupies(
        layer, parent_label, _EMBEDDING_BAG_OFFSETS_SLOT, "offsets"
    ):
        return embedding_bag_offsets_values(layer, parent_values)
    if layer.func_name in _RESHAPE_FUNCS:
        return reshape_size_values(layer, parent_label, parent_values)
    return None


def reshape_size_values(layer: Op, parent_label: str, size: torch.Tensor) -> torch.Tensor | None:
    """Move a traced ``view``/``reshape`` size to the nearest value the wildcard absorbs.

    With the other explicit sizes fixed at product ``P``, a ``-1`` wildcard
    elsewhere admits exactly the sizes that divide ``numel / P``; the nearest
    such size other than the saved one changes the output's shape. A size list
    without a wildcard (or with the wildcard at this slot) admits only the saved
    value, so there is no alternate.

    Parameters
    ----------
    layer:
        The ``view``/``reshape`` op.
    parent_label:
        Parent filling one size slot.
    size:
        Saved 0-d (or one-element) size value.

    Returns
    -------
    torch.Tensor or None
        The alternate size, or ``None`` when no valid alternate exists.
    """

    located = _reshape_sizes(layer, parent_label)
    if located is None or size.numel() != 1:
        return None
    numel, sizes, slot = located
    if sizes.count(-1) != 1 or sizes[slot] == -1 or numel == 0:
        return None
    others = 1
    for position, extent in enumerate(sizes):
        if position != slot and extent != -1:
            others *= extent
    if others <= 0 or numel % others:
        return None
    room, saved = numel // others, sizes[slot]
    candidates = [value for value in _divisors(room) if value != saved]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: (abs(value - saved), value))
    return torch.full_like(size, nearest)


def _divisors(number: int) -> list[int]:
    """Return the positive divisors of ``number`` (trial division to its square root)."""

    found: set[int] = set()
    factor = 1
    while factor * factor <= number:
        if number % factor == 0:
            found.update((factor, number // factor))
        factor += 1
    return sorted(found)


def _reshape_sizes(layer: Op, parent_label: str) -> tuple[int, list[int], int] | None:
    """Return the data numel, the integer size list and ``parent_label``'s slot in it."""

    args: Sequence[Any] = layer.call.saved_args or ()
    positions = (record_member(layer, "parent_arg_positions", None) or {}).get("args", {}) or {}
    keys = [key for key, label in positions.items() if label == parent_label]
    if len(keys) != 1 or not args or not isinstance(args[0], torch.Tensor):
        return None
    key = keys[0]
    if isinstance(key, int) and key >= 1:
        raw, slot = list(args[1:]), key - 1
    elif isinstance(key, tuple) and len(key) == 2 and key[0] == 1 and len(args) == 2:
        raw, slot = list(args[1]), key[1]
    else:
        return None
    sizes = [_as_int(value) for value in raw]
    if any(value is None for value in sizes) or not 0 <= slot < len(sizes):
        return None
    return int(args[0].numel()), [int(value) for value in sizes if value is not None], slot


def _as_int(value: object) -> int | None:
    """Return an int size from a Python int or a one-element integer tensor."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.Tensor) and value.numel() == 1 and value.dtype in _INTEGER_DTYPES:
        return int(value.item())
    return None


def embedding_bag_offsets_values(layer: Op, offsets: torch.Tensor) -> torch.Tensor | None:
    """Move one bag boundary by one index, keeping the offsets contract.

    ``offsets[0]`` stays 0, the offsets stay non-decreasing and at most the
    index count, and with ``include_last_offset`` the closing offset stays put.
    The first interior boundary that can move takes one index from the bag
    after it (or gives one to it), so the pooled output and every bag count
    the edge is real for change.

    Parameters
    ----------
    layer:
        The ``embedding_bag`` op.
    offsets:
        Saved 1-D offsets.

    Returns
    -------
    torch.Tensor or None
        Offsets with one boundary moved, or ``None`` for a single bag (whose
        only valid offsets are ``[0]``) or a malformed capture.
    """

    count = _saved_numel(layer, _EMBEDDING_BAG_INDICES_SLOT)
    if offsets.ndim != 1 or count is None:
        return None
    values = [int(value) for value in offsets.tolist()]
    if not values or values[0] != 0:
        return None
    movable_end = len(values) - (1 if _include_last_offset(layer) else 0)
    for position in range(1, movable_end):
        upper = values[position + 1] if position + 1 < len(values) else count
        if values[position] - 1 >= values[position - 1]:
            moved = values[position] - 1
        elif values[position] + 1 <= upper:
            moved = values[position] + 1
        else:
            continue
        result = offsets.detach().clone()
        result[position] = moved
        return result
    return None


def _include_last_offset(layer: Op) -> bool:
    """Return the captured ``include_last_offset`` flag (``False`` when absent)."""

    kwargs = layer.call.saved_kwargs or {}
    if "include_last_offset" in kwargs:
        return bool(kwargs["include_last_offset"])
    args = layer.call.saved_args or ()
    if len(args) > _EMBEDDING_BAG_INCLUDE_LAST_SLOT:
        return bool(args[_EMBEDDING_BAG_INCLUDE_LAST_SLOT])
    return False


def _saved_numel(layer: Op, position: int) -> int | None:
    """Return the element count of the saved tensor argument at ``position``."""

    args: Sequence[Any] = layer.call.saved_args or ()
    if len(args) <= position or not isinstance(args[position], torch.Tensor):
        return None
    return int(args[position].numel())


def _occupies(layer: Op, parent_label: str, position: int, keyword: str) -> bool:
    """Return whether ``parent_label`` fills positional ``position`` or ``keyword``."""

    positions = record_member(layer, "parent_arg_positions", None) or {}
    if (positions.get("args", {}) or {}).get(position) == parent_label:
        return True
    return (positions.get("kwargs", {}) or {}).get(keyword) == parent_label
