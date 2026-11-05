"""Sum-preserving length perturbation for ``ctc_loss``.

``ctc_loss(log_probs, targets, input_lengths, target_lengths, ...)`` reads two
integer length vectors with constraints between arguments: with concatenated
1-D targets, ``target_lengths`` must sum to the number of targets, and every
input length must fit the time axis of ``log_probs``. The generic integer draw
breaks those constraints, so the perturbed replay raised and a real length edge
stayed unverified (an ESPnet CTC head). The values built here rotate the
lengths across the batch, which keeps their sum and their set of values, so the
call stays valid; the perturbation check itself is unchanged and still requires
the replayed output to differ from the saved one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.op import Op

CTC_LOSS_FUNCS = frozenset({"ctc_loss"})
"""Spellings of ``ctc_loss`` whose length vectors sit at ``args[2]`` and ``args[3]``."""

_LENGTH_SLOTS = {2: "input_lengths", 3: "target_lengths"}


def ctc_length_rotation_values(
    layer: Op, parent_label: str, parent_values: torch.Tensor
) -> torch.Tensor | None:
    """Return the lengths rotated by one batch position when that changes them.

    Parameters
    ----------
    layer:
        The ``ctc_loss`` op being replayed.
    parent_label:
        Label of the parent selected for perturbation.
    parent_values:
        The saved value of that parent.

    Returns
    -------
    torch.Tensor or None
        For an ``input_lengths`` or ``target_lengths`` operand: a 1-D integer
        tensor rolled by one position, which keeps the sum and the values.
        ``None`` for any other operand, a non-integer or non-1-D tensor, or
        lengths a roll leaves unchanged (one sample, or equal lengths); the
        caller then uses the generic perturbation.
    """

    if not _is_length_operand(layer, parent_label):
        return None
    if parent_values.dtype.is_floating_point or parent_values.dtype in (
        torch.bool,
        torch.complex64,
        torch.complex128,
    ):
        return None
    if parent_values.dim() != 1:
        return None
    rolled = parent_values.roll(1)
    if torch.equal(rolled, parent_values):
        return None
    return rolled


def _is_length_operand(layer: Op, parent_label: str) -> bool:
    """Return whether ``parent_label`` is registered at a length argument."""

    positions = record_member(layer, "parent_arg_positions", None) or {}
    args_map = positions.get("args", {}) or {}
    kwargs_map = positions.get("kwargs", {}) or {}
    return any(
        args_map.get(position) == parent_label or kwargs_map.get(name) == parent_label
        for position, name in _LENGTH_SLOTS.items()
    )
