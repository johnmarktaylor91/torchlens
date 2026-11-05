"""Count-preserving mask perturbation for ``masked_scatter``.

``masked_scatter(input, mask, source)`` copies the first ``k`` elements of
``source`` into the positions where ``mask`` is true, where ``k`` is the number
of true entries of the broadcast mask, and raises when ``source`` holds fewer
than ``k`` elements. The generic boolean draw changes ``k``, so a perturbed
replay of a real mask edge usually raised and the edge stayed unverified (a
zuko full-covariance mixture writes its Cholesky factor with
``diag.masked_scatter(tril_mask, off_diag)``). The values built here move the
true entries without raising ``k``; the perturbation check itself is unchanged
and still requires the replayed output to differ from the saved one, so a
recorded mask parent that does not feed the op still fails.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .._data_classes.op import Op

MASKED_SCATTER_FUNCS = frozenset({"masked_scatter", "masked_scatter_", "maskedscatter"})
"""Spellings of ``masked_scatter`` whose mask sits at ``args[1]`` or ``mask=``."""

_MASK_POSITION = 1
_SOURCE_POSITION = 2


def masked_scatter_mask_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return a mask that differs from the saved one and keeps the call valid.

    Parameters
    ----------
    layer:
        The ``masked_scatter`` op being replayed.
    parent_label:
        Label of the parent selected for perturbation.
    parent_values:
        The saved value of that parent.

    Returns
    -------
    torch.Tensor or None
        For the mask operand: the mask with its flattened entries rolled by one
        position, which keeps the number of true entries; when every entry is
        equal (the roll is the identity), an all-true mask with its first entry
        cleared, or an all-false mask with its first entry set when ``source``
        holds enough elements. ``None`` for any other operand, a non-boolean or
        empty mask, or an all-false mask whose ``source`` is too short; the
        caller then uses the generic perturbation.
    """

    if not _is_mask_operand(layer, parent_label):
        return None
    if parent_values.dtype != torch.bool or parent_values.numel() == 0:
        return None
    flat = parent_values.reshape(-1)
    rolled = flat.roll(1)
    if not torch.equal(rolled, flat):
        return rolled.reshape(parent_values.shape)
    toggled = flat.clone()
    toggled[0] = not bool(flat[0])
    if bool(flat[0]) or _source_covers_one_more_entry(layer, parent_values):
        return toggled.reshape(parent_values.shape)
    return None


def _is_mask_operand(layer: Op, parent_label: str) -> bool:
    """Return whether ``parent_label`` is registered at the mask argument."""

    positions = getattr(layer, "parent_arg_positions", None) or {}
    if (positions.get("args", {}) or {}).get(_MASK_POSITION) == parent_label:
        return True
    return (positions.get("kwargs", {}) or {}).get("mask") == parent_label


def _source_covers_one_more_entry(layer: Op, mask: torch.Tensor) -> bool:
    """Return whether ``source`` can fill one more true mask entry after broadcasting.

    One mask entry stands for ``broadcast numel / mask numel`` positions of the
    broadcast mask, so setting it raises the count of true positions by that
    factor.
    """

    args = tuple(getattr(layer, "saved_args", None) or ())
    kwargs = dict(getattr(layer, "saved_kwargs", None) or {})
    source = args[_SOURCE_POSITION] if len(args) > _SOURCE_POSITION else kwargs.get("source")
    target = args[0] if args else kwargs.get("input")
    if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
        return False
    try:
        broadcast = torch.broadcast_shapes(tuple(target.shape), tuple(mask.shape))
    except RuntimeError:
        return False
    per_entry = max(1, int(torch.Size(broadcast).numel()) // max(1, mask.numel()))
    return source.numel() >= per_entry
