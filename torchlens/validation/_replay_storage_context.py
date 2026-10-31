"""Replay a storage-geometry receiver on a clone of its saved storage context.

``as_strided_`` and an in-storage ``resize_`` / ``resize_as_`` read bytes of the
receiver's storage outside the receiver's own view, so capture saves that
receiver with its whole storage (:func:`torchlens._storage_bindings.copy_with_storage_context`).
Replay rebuilds the receiver from a private clone of that context and writes the
swapped-in parent values (saved or perturbed) into the view, so the call reads
the bytes the captured call read around the values under test.
"""

from __future__ import annotations

from typing import Any

import torch

from .._storage_bindings import (
    STORAGE_CONTEXT_RECEIVER_FUNCS,
    copy_with_storage_context,
    views_part_of_its_storage,
)
from ._replay_grad_fidelity import _mirror_captured_requires_grad, _snapshot_grad_flags


def restore_receiver_storage_context(layer: Any, input_args: dict[str, Any]) -> None:
    """Put the receiver's saved storage context under the replay arguments.

    Parameters
    ----------
    layer:
        Op being replayed.
    input_args:
        Replay arguments already deep-cloned from the saved snapshot; the
        receiver clone at ``args[0]`` is compact and is replaced here.
    """

    if getattr(layer, "func_name", None) not in STORAGE_CONTEXT_RECEIVER_FUNCS:
        return
    saved_args = getattr(layer, "saved_args", None)
    args = input_args.get("args")
    if not saved_args or not args:
        return
    saved = saved_args[0]
    if not isinstance(saved, torch.Tensor) or not views_part_of_its_storage(saved):
        return
    copy = copy_with_storage_context(saved.detach())
    args[0] = _mirror_captured_requires_grad(copy, *_snapshot_grad_flags(saved))


def write_into_receiver_storage_context(
    layer: Any, input_args: dict[str, Any], arg_type: str, key: Any, values: Any
) -> bool:
    """Write swapped-in parent values into the receiver's storage-context view.

    Returns ``False`` when the slot is not such a receiver (the caller replaces
    the slot as usual), so no other replay path changes.
    """

    if getattr(layer, "func_name", None) not in STORAGE_CONTEXT_RECEIVER_FUNCS:
        return False
    if arg_type != "args" or key != 0:
        return False
    current = input_args["args"][0]
    if not isinstance(current, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    if not views_part_of_its_storage(current) or current.shape != values.shape:
        return False
    if current.requires_grad or values.requires_grad or current.dtype != values.dtype:
        return False
    with torch.no_grad():
        current.copy_(values)
    return True
