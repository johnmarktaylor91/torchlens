"""Replay a storage-geometry or view-function receiver on a clone of its saved context.

``as_strided_`` and an in-storage ``resize_`` / ``resize_as_`` read bytes of the
receiver's storage outside the receiver's own view, and ``_rev_view_func_unsafe``
reads bytes of its new view's storage the same way, so capture saves that argument
with its whole storage (:func:`torchlens._storage_bindings.copy_with_storage_context`).
The view-function methods (``_view_func`` and kin) read the receiver's view chain,
so capture saves that receiver rebuilt on a private copy of its base
(:func:`torchlens._storage_bindings.copy_with_view_context`).
Replay rebuilds the receiver from a private clone of that context and writes the
swapped-in parent values (saved or perturbed) into the view, so the call reads
the bytes and view chain the captured call read around the values under test.

Every other parent slot is written in the memory layout its argument had at the
call (:func:`relay_to_captured_layout`): a saved out is a compact copy, and a kernel
can round differently on a strided operand than on a contiguous one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from .._storage_bindings import (
    VIEW_CONTEXT_RECEIVER_FUNCS,
    copy_with_storage_context,
    copy_with_view_context,
    has_view_geometry_of,
    storage_context_arg_index,
    views_part_of_its_storage,
)
from .._utils._record_group_table import record_member
from .._utils._tensor_layout import CaptureLayout, layout_of, relay_to_layout
from .._utils._torch_compat import get_force_original_view_tracking, get_view_func_replay
from .._utils.collections import assign_to_sequence_or_dict
from ._grad_flags import _mirror_captured_requires_grad, _snapshot_grad_flags


def restore_receiver_storage_context(layer: Any, input_args: dict[str, Any]) -> None:
    """Put the saved storage or view context under the replay arguments.

    Parameters
    ----------
    layer:
        Op being replayed.
    input_args:
        Replay arguments already deep-cloned from the saved snapshot; the
        compact clone of each context-saved argument is replaced here.
    """

    func_name = getattr(layer, "func_name", None)
    if func_name in VIEW_CONTEXT_RECEIVER_FUNCS:
        _restore_receiver_view_context(layer, input_args)
    index = storage_context_arg_index(func_name)
    saved_args = record_member(layer, "saved_args", None) or ()
    args = input_args.get("args")
    if index is None or not args or len(saved_args) <= index or len(args) <= index:
        return
    saved = saved_args[index]
    if not isinstance(saved, torch.Tensor) or not views_part_of_its_storage(saved):
        return
    copy = copy_with_storage_context(saved.detach())
    args[index] = _mirror_captured_requires_grad(copy, *_snapshot_grad_flags(saved))


def _restore_receiver_view_context(layer: Any, input_args: dict[str, Any]) -> None:
    """Rebuild a view-function receiver on a fresh copy of its saved base.

    A saved receiver that is not a view (no view context was saved) stays the
    compact clone; :func:`check_view_context_receiver` refuses it before the call.
    """

    saved_args = record_member(layer, "saved_args", None)
    args = input_args.get("args")
    if not saved_args or not args:
        return
    saved = saved_args[0]
    if isinstance(saved, torch.Tensor) and saved._is_view():
        args[0] = copy_with_view_context(
            saved, get_force_original_view_tracking(), get_view_func_replay()
        )


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
    saved_args = record_member(layer, "saved_args", None) or ()
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
    """Write swapped-in parent values into an argument's storage-context or view-context view.

    Returns ``False`` when the slot is not such a receiver (the caller replaces
    the slot as usual), so no other replay path changes.
    """

    func_name = getattr(layer, "func_name", None)
    if arg_type != "args":
        return False
    if func_name in VIEW_CONTEXT_RECEIVER_FUNCS and key == 0:
        return _write_into_receiver_view_context(input_args["args"][0], values)
    if key != storage_context_arg_index(func_name):
        return False
    return _write_into_storage_context(input_args["args"][key], values)


def _write_into_storage_context(current: Any, values: Any) -> bool:
    """Copy swapped-in values into an argument restored with its whole storage."""

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
            input_args[arg_type][key] = relay_to_captured_layout(layer, arg_type, key, value)
    else:
        input_args[arg_type][key[0]] = _write_nested_replay_arg_value(
            input_args[arg_type][key[0]],
            key[1:],
            relay_to_captured_layout(layer, arg_type, key, value),
        )


def relay_to_captured_layout(layer: Any, arg_type: str, key: Any, value: Any) -> Any:
    """Return a replay value placed in the memory layout its slot had at the captured call.

    Saved outs are compact copies, so the value written into a parent slot is
    row-major even when the captured call received a strided view (one half of a
    ``chunk``, a column slice, a transposed non-dense view). Kernels pick their
    path by layout: a float32 ``gelu`` takes the oneDNN kernel only for a
    contiguous input, and BLAS routines sum in a different order for a strided
    operand, so a replay from the compact copy can miss the saved out by a
    rounding step. Capture records each parent argument's non-contiguous layout on
    the call's argument template (``ParentRef._capture_layout``); this copies the
    value into that layout, keeping its values and its grad flags. Values are
    unchanged, so the comparison is exactly as strict as before.

    Parameters
    ----------
    layer:
        Op being replayed.
    arg_type:
        Either ``"args"`` or ``"kwargs"``.
    key:
        Slot key, possibly a nested key path.
    value:
        Replay value headed for the slot (saved, perturbed or probed).

    Returns
    -------
    Any
        ``value`` itself when the slot has no recorded layout or ``value`` cannot
        take it, else a copy laid out as recorded.
    """

    if not isinstance(value, torch.Tensor):
        return value
    layout = slot_capture_layout(layer, arg_type, key)
    if layout is None:
        return value
    if not value.requires_grad:
        return relay_to_layout(value, layout)
    detached = value.detach()
    relayed = relay_to_layout(detached, layout)
    if relayed is detached or not isinstance(relayed, torch.Tensor):
        return value
    with torch.enable_grad():
        grad_leaf = relayed.detach().requires_grad_(True)
        if value.is_leaf:
            return grad_leaf
        # A non-leaf stays a non-leaf that is not an autograd view, as
        # ``_mirror_captured_requires_grad`` builds it, now in the recorded layout.
        return relay_to_layout(grad_leaf.contiguous(), layout)


def slot_capture_layout(layer: Any, arg_type: str, key: Any) -> CaptureLayout | None:
    """Return the layout recorded for one parent slot on the op's argument template."""

    # Read by attribute: importing the template types here would join this module
    # to the intervention package's import cycle.
    template = record_member(layer, "args_template", None)
    if arg_type not in ("args", "kwargs"):
        return None
    component: Any = getattr(template, arg_type, None)
    path = key if isinstance(key, tuple) else (key,)
    for part in path:
        component = _template_child(component, part)
        if component is None:
            return None
    return cast("CaptureLayout | None", getattr(component, "_capture_layout", None))


def replayed_storage_alias(
    written: torch.Tensor,
    geometry: tuple[tuple[int, ...], CaptureLayout | None],
    lineage: list[torch.Tensor | None],
    others: list[torch.Tensor],
) -> torch.Tensor | None:
    """Return the replayed tensor on ``written``'s storage that a slot read.

    Parameters
    ----------
    written:
        The replayed output of an in-place op that wrote through a view.
    geometry:
        The read's shape and its recorded layout (``None`` for a row-major
        contiguous read, whose storage offset is not recorded).
    lineage:
        Replayed values along the in-place op's receiver chain (the base and the
        views it wrote through), searched first.
    others:
        Every other replayed differentiable value, searched when the lineage holds
        no match. Within either pool the matches must agree on one geometry: two
        contiguous aliases of one shape at different offsets are ambiguous.

    Returns
    -------
    torch.Tensor | None
        The alias, or ``None`` when the replay holds none.

    Raises
    ------
    ValueError
        When the matching candidates of the pool searched are ambiguous.
    """

    shape, layout = geometry

    def matches(candidate: torch.Tensor | None) -> bool:
        """Return whether ``candidate`` is on ``written``'s storage with the read geometry."""

        if candidate is None or tuple(candidate.shape) != shape:
            return False
        same_storage = (
            candidate.device == written.device
            and candidate.untyped_storage().data_ptr() == written.untyped_storage().data_ptr()
        )
        if not same_storage:
            return False
        return candidate.is_contiguous() if layout is None else layout_of(candidate) == layout

    for pool in (lineage, others):
        candidates = [candidate for candidate in pool if matches(candidate)]
        if len({cast(torch.Tensor, candidate).storage_offset() for candidate in candidates}) > 1:
            raise ValueError("the storage alias a view write reached is ambiguous in the replay")
        if candidates:
            return candidates[0]
    return None


def storage_key(tensor: torch.Tensor) -> tuple[torch.device, int]:
    """Return the device and storage address identifying ``tensor``'s storage."""

    return tensor.device, tensor.untyped_storage().data_ptr()


def alias_provenance_holds(
    found: torch.Tensor, written: torch.Tensor, geometry: tuple[tuple[int, ...], Any]
) -> bool:
    """Return whether an alias provably reads the recorded region of the written storage.

    Used where values cannot be compared (a perturbed replay): the alias must sit on
    the very storage the in-place op's replayed output wrote, with the slot's shape,
    and with its recorded layout (stride and storage offset) whenever one was recorded.
    """

    shape, layout = geometry
    if storage_key(found) != storage_key(written) or tuple(found.shape) != shape:
        return False
    return found.is_contiguous() if layout is None else layout_of(found) == layout


def writes_through_view(
    slot: tuple[str, Any], receiver: torch.Tensor, enclosing: frozenset[tuple[torch.device, int]]
) -> bool:
    """Return whether an in-place op's slot is a write through a view to replay in place.

    Only the receiver (first positional argument) of an in-place op that is a
    non-leaf view of a replay-owned base is written in place, as capture wrote it,
    so the write reaches the replayed base. ``enclosing`` holds the storages of an
    enclosing replay's tensors: a nested replay never writes into them.
    """

    if slot != ("args", 0) or receiver.is_leaf or receiver._base is None:
        return False
    return storage_key(receiver) not in enclosing


def _template_child(component: Any, part: Any) -> Any:
    """Step one key into a template level: an index of a sequence, a key of a mapping."""

    if not isinstance(component, tuple):
        return None
    if isinstance(part, int):
        return component[part] if -len(component) <= part < len(component) else None
    pairs = (item for item in component if isinstance(item, tuple) and len(item) == 2)
    return next((item[1] for item in pairs if item[0] == part), None)


_LINEAGE_BOUND = 64
"""Longest receiver chain searched for the storage alias a view write reached."""


def receiver_lineage(op: Any, resolve: Callable[[str], Any]) -> list[str]:
    """Return the concrete labels along ``op``'s receiver (first argument) chain."""

    labels: list[str] = []
    current = op
    while current is not None and len(labels) < _LINEAGE_BOUND:
        labels.append(current.label)
        positions = (record_member(current, "parent_arg_positions", {}) or {}).get("args", {})
        receiver = positions.get(0)
        current = resolve(receiver) if receiver is not None else None
    return labels


def rebuilt_alias(
    written: torch.Tensor, layout: CaptureLayout | None, written_layout: CaptureLayout | None
) -> torch.Tensor | None:
    """Rebuild, on the written storage, an alias the replay never built.

    Only when the read's full layout was recorded and the written view sits where
    capture's receiver did (``written_layout``), so the recorded layout addresses
    the same elements. The view is taken on the root base: ``as_strided``
    differentiates only over the elements of the tensor it is called on, and the
    written view covers fewer.
    """

    written_at = layout_of(written)
    if layout is None or written_at is None or written_at != written_layout:
        return None
    base = written._base if written._base is not None else written
    return base.as_strided(*layout)
