"""Saved-argument copies for ``save_args`` captures, and validation's by-reference aliases.

Every capture with ``save_args=True`` copies each call's arguments twice: a
pre-call snapshot (in case the call mutates them) and the retained
``saved_args``/``saved_kwargs``. Inside forward validation's own capture some of
those copies are replaced by aliases; see
:mod:`torchlens._validation_argument_references` for the guard that decides.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from ..._state import pause_logging
from ..._storage_bindings import (
    VIEW_CONTEXT_RECEIVER_FUNCS,
    copy_with_storage_context,
    copy_with_view_context,
    storage_context_arg_index,
)
from ..._utils._torch_compat import get_force_original_view_tracking, get_view_func_replay
from ..._utils.arg_handling import copy_arg_tree
from ..._validation_argument_references import active_argument_references

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


__all__ = (
    "_copy_saved_arg_tree",
    "_copy_saved_call_args",
    "_is_flat_sequence",
    "_saved_parent_values",
)


def _saved_parent_values(
    trace: "Trace", fields_dict: dict[str, Any]
) -> dict[Any, dict[Any, tuple[str, Any]]] | None:
    """Map each argument slot of one call to its recorded parent's saved output.

    Only inside validation's capture with saved-value aliasing on; ``None``
    everywhere else, so no other capture reads a parent's output here. Keys
    follow :func:`_copy_saved_call_args`: the outer key is the position or
    keyword, the inner key ``None`` for a bare tensor or the item index of a
    flat list or tuple. Slots whose parent kept no saved tensor are left out.
    """

    references = active_argument_references()
    if references is None or not references.aliases_saved_values:
        return None
    by_label = trace.capture_events.live_index.by_raw_label
    positions = fields_dict.get("parent_arg_positions") or {}
    values: dict[Any, dict[Any, tuple[str, Any]]] = {}
    for domain in ("args", "kwargs"):
        for location, label in (positions.get(domain) or {}).items():
            output = getattr(by_label.get(label), "output", None)
            payload = getattr(getattr(output, "tensor", None), "payload", None)
            if not getattr(output, "has_saved_activation", False):
                continue
            if not isinstance(payload, torch.Tensor):
                continue
            path = location if isinstance(location, tuple) else (location,)
            if len(path) == 1:
                values.setdefault(path[0], {})[None] = (label, payload)
            elif len(path) == 2:
                values.setdefault(path[0], {})[path[1]] = (label, payload)
    return values


def _copy_saved_call_args(
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    retained: bool = False,
    parent_values: dict[Any, dict[Any, tuple[str, Any]]] | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Copy one call's arguments for ``save_arg_values``, storage and view context included.

    The positional argument named by
    :func:`~torchlens._storage_bindings.storage_context_arg_index` keeps its whole
    storage, and the receiver (positional argument 0) of a
    :data:`~torchlens._storage_bindings.VIEW_CONTEXT_RECEIVER_FUNCS` call keeps its
    view relation; every other argument is copied as :func:`_copy_saved_arg_tree` does.
    ``parent_values`` maps a position or keyword to its recorded parents' saved outputs.
    """

    copiers = _context_copiers(func_name)
    values = parent_values or {}
    arg_copies = tuple(
        _copy_saved_arg_tree(
            arg, retained, receiver_copy=copiers.get(index), parent_values=values.get(index)
        )
        for index, arg in enumerate(args)
    )
    kwarg_copies = {
        key: _copy_saved_arg_tree(value, retained, parent_values=values.get(key))
        for key, value in kwargs.items()
    }
    return arg_copies, kwarg_copies


def _context_copiers(func_name: str) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Map each positional argument of ``func_name`` saved with context to its copier."""

    copiers: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    if func_name in VIEW_CONTEXT_RECEIVER_FUNCS:
        copiers[0] = _copy_receiver_with_view_context
    index = storage_context_arg_index(func_name)
    if index is not None:
        copiers[index] = copy_with_storage_context
    return copiers


def _copy_receiver_with_view_context(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a view-function receiver with its view context, under original-view tracking."""
    return copy_with_view_context(
        tensor, get_force_original_view_tracking(), get_view_func_replay()
    )


def _copy_saved_arg_tree(
    arg: Any,
    retained: bool = False,
    *,
    receiver_copy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    parent_values: dict[Any, tuple[str, Any]] | None = None,
) -> Any:
    """Copy one ``save_arg_values`` argument tree, as the pre-call and saved snapshots do.

    Outside forward validation's own capture this is exactly
    :func:`~torchlens._utils.arg_handling.copy_arg_tree`. Inside it, a plain
    parameter argument (or a tensor on a parameter's storage), bare or in a
    flat list or tuple, is replaced by an alias of the same storage and version
    counter instead of a copy; see :mod:`torchlens._validation_argument_references`
    for the guard that reruns validation on this copy path if such a value changes.
    For the ``retained`` snapshot (``saved_args``/``saved_kwargs``), a validation
    ``output_device`` other than ``"same"`` also keeps every other such tensor's
    copy on that device, tagged with the device the op consumed it on. A ``retained``
    tensor whose recorded parent saved the identical value (``parent_values``, keyed
    ``None`` for the bare tensor and by index for a flat item) may alias that value.
    """

    references = active_argument_references()
    if references is None:
        if receiver_copy is not None and isinstance(arg, torch.Tensor):
            return _copy_receiver_with_context(receiver_copy, arg)
        return copy_arg_tree(arg)
    if isinstance(arg, torch.Tensor) and receiver_copy is _copy_receiver_with_view_context:
        # A parameter alias is not a view, so a view-function receiver is always copied.
        return _copy_receiver_with_context(receiver_copy, arg)
    parents = parent_values or {}
    if isinstance(arg, torch.Tensor):
        return _parameter_alias(
            references, arg, retained, receiver_copy=receiver_copy, parent=parents.get(None)
        )
    if _is_flat_sequence(arg):
        return type(arg)(
            _parameter_alias(references, item, retained, parent=parents.get(index))
            if isinstance(item, torch.Tensor)
            else item
            for index, item in enumerate(arg)
        )
    return copy_arg_tree(arg)


def _is_flat_sequence(arg: Any) -> bool:
    """Whether ``arg`` is a plain list or tuple holding no nested container."""

    return type(arg) in (list, tuple) and not any(
        isinstance(item, (list, tuple, dict)) for item in arg
    )


def _parameter_alias(
    references: Any,
    tensor: torch.Tensor,
    retained: bool = False,
    *,
    receiver_copy: Callable[[torch.Tensor], torch.Tensor] | None = None,
    parent: tuple[str, Any] | None = None,
) -> torch.Tensor:
    """Alias a parameter-storage tensor (recording its version), else copy it as before.

    The guard decides and builds the alias (see
    ``ParameterArgumentReferences.alias``) and, for a ``retained`` snapshot,
    the offloaded copy (``ParameterArgumentReferences.offload``); anything it
    declines is copied exactly as on the copy path (through ``receiver_copy``
    when it is set; an alias already shares the storage). A ``retained`` argument with
    no storage or view context may instead alias its ``parent``'s identical saved
    output (``ParameterArgumentReferences.value_alias``).
    """

    # Paused: the guard's storage and version reads, its alias and its copy are
    # TorchLens's own calls, never captured ops (and never re-enter this copy).
    with pause_logging():
        alias = references.alias(tensor)
        if alias is None and retained and parent is not None and receiver_copy is None:
            alias = references.value_alias(tensor, parent[1], parent[0])
        if alias is None and retained and references.offloads:
            alias = references.offload(tensor)
    if alias is not None:
        return alias
    if receiver_copy is not None:
        return _copy_receiver_with_context(receiver_copy, tensor)
    return cast(torch.Tensor, copy_arg_tree(tensor))


def _copy_receiver_with_context(
    receiver_copy: Callable[[torch.Tensor], torch.Tensor], tensor: torch.Tensor
) -> torch.Tensor:
    """Copy a receiver with its storage or view context, as TorchLens's own work."""

    with pause_logging():
        return receiver_copy(tensor)
