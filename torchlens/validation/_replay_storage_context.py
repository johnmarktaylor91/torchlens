"""Replay a storage-geometry or view-function receiver on a clone of its saved context.

``as_strided_`` and an in-storage ``resize_`` / ``resize_as_`` read bytes of the
receiver's storage outside the receiver's own view, so capture saves that
receiver with its whole storage (:func:`torchlens._storage_bindings.copy_with_storage_context`).
The view-function methods (``_view_func`` and kin) read the receiver's view chain,
so capture saves that receiver rebuilt on a private copy of its base
(:func:`torchlens._storage_bindings.copy_with_view_context`).
Replay rebuilds the receiver from a private clone of that context and writes the
swapped-in parent values (saved or perturbed) into the view, so the call reads
the bytes and view chain the captured call read around the values under test.
"""

from __future__ import annotations

from typing import Any

import torch

from .._storage_bindings import (
    STORAGE_CONTEXT_RECEIVER_FUNCS,
    VIEW_CONTEXT_RECEIVER_FUNCS,
    copy_with_storage_context,
    copy_with_view_context,
    has_view_geometry_of,
    views_part_of_its_storage,
)
from .._utils.collections import assign_to_sequence_or_dict
from ._grad_flags import _mirror_captured_requires_grad, _snapshot_grad_flags


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

    if getattr(layer, "func_name", None) in VIEW_CONTEXT_RECEIVER_FUNCS:
        _restore_receiver_view_context(layer, input_args)
        return
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


def _restore_receiver_view_context(layer: Any, input_args: dict[str, Any]) -> None:
    """Rebuild a view-function receiver on a fresh copy of its saved base.

    A saved receiver that is not a view (no view context was saved) stays the
    compact clone; :func:`check_view_context_receiver` refuses it before the call.
    """

    saved_args = getattr(layer, "saved_args", None)
    args = input_args.get("args")
    if not saved_args or not args:
        return
    saved = saved_args[0]
    if isinstance(saved, torch.Tensor) and saved._is_view():
        args[0] = copy_with_view_context(saved)


def check_view_context_receiver(layer: Any, input_args: dict[str, Any]) -> None:
    """Refuse to replay a view-function method on a receiver that is not its recorded view.

    Parameters
    ----------
    layer:
        Op being replayed.
    input_args:
        Prepared replay arguments.

    Raises
    ------
    ValueError
        When the replay receiver is not a view with the saved receiver's shape,
        strides and storage offset (torch would return ``None`` or replay a
        different view chain).
    """

    if getattr(layer, "func_name", None) not in VIEW_CONTEXT_RECEIVER_FUNCS:
        return
    saved_args = getattr(layer, "saved_args", None) or ()
    args = input_args.get("args") or ()
    recorded = saved_args[0] if saved_args else None
    receiver = args[0] if args else None
    if not (
        isinstance(recorded, torch.Tensor)
        and isinstance(receiver, torch.Tensor)
        and has_view_geometry_of(receiver, recorded)
    ):
        raise ValueError(
            f"Replay of {layer.func_name} needs its receiver rebuilt as the recorded view "
            "(shape, strides, storage offset and view relation); the saved receiver has no "
            "view context or the rebuilt receiver differs from it."
        )


def _write_into_receiver_view_context(current: Any, values: Any) -> bool:
    """Copy swapped-in receiver values into a rebuilt view-function receiver."""

    if not isinstance(current, torch.Tensor) or not isinstance(values, torch.Tensor):
        return False
    if not current._is_view() or current.shape != values.shape or current.dtype != values.dtype:
        return False
    with torch.no_grad():
        current.copy_(values.detach())
    return True


def write_into_receiver_storage_context(
    layer: Any, input_args: dict[str, Any], arg_type: str, key: Any, values: Any
) -> bool:
    """Write swapped-in parent values into the receiver's storage-context or view-context view.

    Returns ``False`` when the slot is not such a receiver (the caller replaces
    the slot as usual), so no other replay path changes.
    """

    if getattr(layer, "func_name", None) in VIEW_CONTEXT_RECEIVER_FUNCS:
        if arg_type != "args" or key != 0:
            return False
        return _write_into_receiver_view_context(input_args["args"][0], values)
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


def _write_nested_replay_arg_value(
    container: Any,
    key_path: tuple[Any, ...],
    value: torch.Tensor,
) -> Any:
    """Write a nested replay argument by key path.

    Parameters
    ----------
    container:
        Nested list, tuple, or dict containing the value to replace.
    key_path:
        One or more nested keys beneath the replay arg root.
    value:
        Replacement tensor.

    Returns
    -------
    Any
        Updated container, rebuilding tuple levels as needed.
    """

    if len(key_path) == 1:
        return assign_to_sequence_or_dict(container, key_path[0], value)
    child = _write_nested_replay_arg_value(container[key_path[0]], key_path[1:], value)
    return assign_to_sequence_or_dict(container, key_path[0], child)


def write_replay_slot(
    layer: Any, input_args: dict[str, Any], arg_type: str, key: Any, value: Any
) -> None:
    """Write one parent's replay value into its slot of ``input_args``.

    A storage-geometry receiver slot is written through its saved storage
    context; a nested key path rebuilds the containers above the slot.
    """
    if not isinstance(key, tuple):
        if not write_into_receiver_storage_context(layer, input_args, arg_type, key, value):
            input_args[arg_type][key] = value
    else:
        input_args[arg_type][key[0]] = _write_nested_replay_arg_value(
            input_args[arg_type][key[0]], key[1:], value
        )
