"""Device alignment for replay when saved payloads live off the op's device.

A validator capture with ``output_device`` other than ``"same"`` keeps every
saved activation on that device. Forward validation's own capture also keeps
its saved function arguments (``save_arg_values``) there, apart from parameter
aliases, each copy tagged with the device the op consumed it on. Replay must
still run each op where it originally ran, so while one layer is checked its
ops' tagged saved arguments are held on their recorded devices, a parent
payload swapped into a replay argument slot moves to the device of the op's
own saved argument at that slot, and the recomputed output moves to the saved
output's device for the comparison. Every move is an exact copy: no value,
tolerance, or check changes. On the default ``output_device="same"`` path every
device already matches, no argument is tagged, and the helpers change nothing.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch

from .._validation_argument_references import CONSUMED_DEVICE_ATTR


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


def _on_consumed_device(value: Any) -> Any:
    """Return ``value`` with each tagged tensor copied exactly to its recorded device.

    Parameters
    ----------
    value:
        A saved argument: a tensor, a plain list or tuple of values, or anything
        else (returned unchanged).

    Returns
    -------
    Any
        ``value`` itself when nothing in it is tagged, else a copy holding the
        moved tensors.
    """

    if isinstance(value, torch.Tensor):
        device = getattr(value, CONSUMED_DEVICE_ATTR, None)
        if device is None:
            return value
        # Grad mode follows the snapshot so the copy keeps its requires_grad.
        with torch.set_grad_enabled(value.requires_grad):
            return value.to(device, copy=True)
    if type(value) in (list, tuple):
        items = [_on_consumed_device(item) for item in value]
        if all(item is original for item, original in zip(items, value, strict=True)):
            return value
        return type(value)(items)
    return value


def _hold_on_consumed_devices(container: Any, swapped: list[tuple[Any, Any, Any, Any]]) -> None:
    """Swap each tagged entry of a saved-args list or saved-kwargs dict in place.

    Parameters
    ----------
    container:
        ``Op.saved_args`` or ``Op.saved_kwargs``; anything else is skipped.
    swapped:
        Receives ``(container, key, host value, moved value)`` for every swap.
    """

    if isinstance(container, list):
        keys: list[Any] = list(range(len(container)))
    elif isinstance(container, dict):
        keys = list(container)
    else:
        return
    for key in keys:
        host_value = container[key]
        moved = _on_consumed_device(host_value)
        if moved is not host_value:
            container[key] = moved
            swapped.append((container, key, host_value, moved))


@contextlib.contextmanager
def saved_args_on_consumed_devices(trace: Any, label: str) -> Iterator[None]:
    """Hold one layer entry's offloaded saved arguments on their recorded devices.

    Every saved-argument reader that checks this entry (replay, argument
    logging, probes, live-parameter restore) then sees each value on the device
    the op consumed it on, as when arguments were never offloaded. The host
    copies are put back on every exit, an exception included.

    Parameters
    ----------
    trace:
        Trace being validated.
    label:
        Validation queue label of the entry about to be checked.

    Yields
    ------
    None
        While the entry is checked.
    """

    from .core import _op_for_validation_label, _validation_ops_for_entry

    entry = trace.layer_logs.get(label, _op_for_validation_label(trace, label))
    swapped: list[tuple[Any, Any, Any, Any]] = []
    try:
        for op in _validation_ops_for_entry(entry):
            _hold_on_consumed_devices(getattr(op, "saved_args", None), swapped)
            _hold_on_consumed_devices(getattr(op, "saved_kwargs", None), swapped)
        yield
    finally:
        for container, key, host_value, moved in reversed(swapped):
            if container[key] is moved:
                container[key] = host_value
