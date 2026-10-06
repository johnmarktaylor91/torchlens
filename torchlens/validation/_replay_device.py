"""Device alignment for replay when saved payloads live off the op's device.

A validator capture with ``output_device`` other than ``"same"`` keeps every
saved activation on that device, while saved function arguments
(``save_arg_values``) stay on the device the op consumed them on. Replay must
still run each op where it originally ran, so a parent payload swapped into a
replay argument slot moves to the device of the op's own saved argument at
that slot, and the recomputed output moves to the saved output's device for
the comparison. Both moves are exact copies: no value, tolerance, or check
changes. On the default ``output_device="same"`` path every device already
matches and both helpers return their input unchanged.
"""

from __future__ import annotations

from typing import Any

import torch


def _slot_value(input_args: dict[str, Any], arg_type: str, key: Any) -> Any:
    """Return the original saved-argument value at a replay slot, or ``None``.

    Parameters
    ----------
    input_args:
        Replay argument dict with ``"args"`` and ``"kwargs"`` entries, still
        holding the op's own saved-argument copies.
    arg_type:
        ``"args"`` or ``"kwargs"``.
    key:
        Parent-argument position key, possibly a nested tuple path.

    Returns
    -------
    Any
        The slot value, or ``None`` when the slot cannot be read.
    """

    path = key if isinstance(key, tuple) else (key,)
    value: Any = input_args[arg_type]
    try:
        for step in path:
            value = value[step]
    except (IndexError, KeyError, TypeError):
        return None
    return value


def align_parent_to_slot_device(
    input_args: dict[str, Any],
    arg_type: str,
    key: Any,
    parent_value: Any,
) -> Any:
    """Move a swapped-in parent payload to the device its slot was consumed on.

    Parameters
    ----------
    input_args:
        Replay argument dict before the parent value is written.
    arg_type:
        ``"args"`` or ``"kwargs"``.
    key:
        Parent-argument position key.
    parent_value:
        Saved (possibly perturbed) parent payload about to be written.

    Returns
    -------
    Any
        ``parent_value`` on the slot's original device, or unchanged when the
        devices already match or the slot holds no tensor.
    """

    if not isinstance(parent_value, torch.Tensor):
        return parent_value
    slot = _slot_value(input_args, arg_type, key)
    if not isinstance(slot, torch.Tensor) or slot.device == parent_value.device:
        return parent_value
    if slot.device.type == "meta":
        # Offload-hooked slots hold meta placeholders; they place inputs themselves.
        return parent_value
    return parent_value.to(slot.device)


def align_output_to_saved_device(recomputed: Any, saved: Any) -> Any:
    """Move a recomputed replay output to the saved output's device.

    Parameters
    ----------
    recomputed:
        Output of the replayed op.
    saved:
        The op's saved output payload.

    Returns
    -------
    Any
        ``recomputed`` on ``saved``'s device, or unchanged when either is not
        a tensor or the devices already match.
    """

    if not isinstance(recomputed, torch.Tensor) or not isinstance(saved, torch.Tensor):
        return recomputed
    if recomputed.device == saved.device:
        return recomputed
    return recomputed.to(saved.device)
