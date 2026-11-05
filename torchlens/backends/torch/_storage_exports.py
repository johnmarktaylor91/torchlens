"""Copy out pending payload aliases before a tensor's storage is handed to the host zero-copy.

A plain capture saves an op's out as a clone-on-write alias of the live storage and
copies it out only before a wrapped call that can write that storage. A raw pointer
(``data_ptr``), a storage handle, a DLPack capsule or a NumPy view lets code outside
torch dispatch write the bytes later (a warp or numba kernel, ctypes, an ndarray
assignment) with no interception and no version bump, so the saved out would show the
later write. The wrapper of each export method therefore copies out the pending
aliases of the receiver's storage before handing it over, for calls the model makes
while its forward is being logged (TorchLens's own bookkeeping reads run paused or
under the internal-read marker). After the capture returns, TorchLens's postprocess
reads storage pointers unmarked, so exports there are not intercepted: a foreign
write into a live tensor after capture can still reach a saved out that aliases it,
as can a write through a pointer the model took before the payload was deferred.
Exports inside a live ``autograd.Function`` forward are left to the Function boundary
(``_function_boundary``): a forward that leaves torch is collapsed into one boundary
op recorded after the kernel's write, and an ordinary Function's interior keeps the
census tripwire, so a foreign write there still reaches the interior's saved outs.

The same export also records the receiver's storage for the capture
(:func:`note_storage_export`). A later call that reads a tensor on that storage with a
value other than its producer's saved out stores a per-child version snapshot like any
changed parent; the write that changed it ran outside torch and is in no recorded op.
:func:`note_foreign_written_snapshot` marks that snapshot with the exact argument slots
the parent was bound to, and forward validation accepts a snapshot it cannot otherwise
explain only for those slots (``validation/_snapshot_binding.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ... import _state
from ..._utils._internal_read import _internal_read_state
from ..._utils.collections import index_nested
from ..._utils.tensor_utils import _DEFER_PENDING, materialize_deferred_for_call

__all__ = (
    "EXPORTED_STORAGES_ATTR",
    "FOREIGN_WRITTEN_SLOTS_ATTR",
    "STORAGE_EXPORT_FUNC_NAMES",
    "materialize_deferred_for_export",
    "note_foreign_written_snapshot",
    "note_storage_export",
)

EXPORTED_STORAGES_ATTR = "_exported_storages"
"""Session-time Trace attribute: exported storage key to the exported tensor (popped in step 3)."""

FOREIGN_WRITTEN_SLOTS_ATTR = "_foreign_written_slots"
"""Trace attribute: ``(parent raw label, child raw label)`` to the parent's bound slots."""

STORAGE_EXPORT_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "data_ptr",
        "untyped_storage",
        "storage",
        "_typed_storage",
        "__dlpack__",
        "numpy",
        "__array__",
        "__cuda_array_interface__",
    }
)
"""Tensor methods that hand the host a zero-copy view of the receiver's storage."""


def materialize_deferred_for_export(args: Sequence[Any]) -> None:
    """Copy out pending payload aliases of the receiver's storage before an export.

    Parameters
    ----------
    args:
        Positional arguments of the export call; the receiver is ``args[0]``. The
        caller has already checked that the owner thread's forward is being logged
        and that no ``Function.apply`` forward is live.
    """

    if not _DEFER_PENDING or not args:
        return
    if getattr(_internal_read_state, "depth", 0):
        return
    materialize_deferred_for_call(args[:1])


def _storage_key(tensor: Any) -> tuple[str, int] | None:
    """Return ``(device, storage data pointer)`` of a tensor, or ``None`` when unreadable."""

    if not isinstance(tensor, torch.Tensor):
        return None
    # Bookkeeping, not a user exposure: read paused so the completeness witness never
    # counts TorchLens's own pointer read as a raw-pointer escape.
    with _state.pause_logging():
        try:
            storage = tensor.untyped_storage()
            key = (str(storage.device), int(storage.data_ptr()))
        except (RuntimeError, TypeError, AttributeError, NotImplementedError):
            return None
    return None if key[1] == 0 else key


def _slot_value(container: Any, position: Any) -> Any:
    """Return the live argument at ``position``, or ``None`` when it cannot be indexed."""

    try:
        return index_nested(container, position)
    except (IndexError, KeyError, TypeError):
        return None


def note_storage_export(trace: Any, args: Sequence[Any]) -> None:
    """Copy out pending aliases, then record the exported storage for the capture.

    Parameters
    ----------
    trace:
        Live capture trace; the owner thread's forward is being logged and no
        ``Function.apply`` forward is live.
    args:
        Positional arguments of the export call; the receiver is ``args[0]``.
    """

    materialize_deferred_for_export(args)
    if not args or getattr(_internal_read_state, "depth", 0):
        return
    key = _storage_key(args[0])
    if key is not None:
        # The tensor is held so its storage pointer cannot be reused by a later
        # allocation and taken for an exported one.
        trace.__dict__.setdefault(EXPORTED_STORAGES_ATTR, {})[key] = args[0]


def note_foreign_written_snapshot(
    trace: Any,
    parent_label: str,
    child_label: str,
    parent_arg_positions: Mapping[str, Mapping[Any, str]],
    live_args: tuple[Sequence[Any], Mapping[str, Any]],
) -> None:
    """Mark a parent's version snapshot whose live slot sits on an exported storage.

    Parameters
    ----------
    trace:
        Live capture trace.
    parent_label, child_label:
        Raw labels of the parent and of the child the snapshot was stored for.
    parent_arg_positions:
        The child's ``{"args": {position: label}, "kwargs": {name: label}}`` binding.
    live_args:
        The child's live positional and keyword arguments.
    """

    exported = trace.__dict__.get(EXPORTED_STORAGES_ATTR)
    if not exported:
        return
    slots = [
        (kind, position)
        for kind in ("args", "kwargs")
        for position, label in (parent_arg_positions.get(kind) or {}).items()
        if label == parent_label
    ]
    sources = dict(zip(("args", "kwargs"), live_args, strict=True))
    if any(_storage_key(_slot_value(sources[kind], pos)) in exported for kind, pos in slots):
        marks = trace.__dict__.setdefault(FOREIGN_WRITTEN_SLOTS_ATTR, {})
        marks[(parent_label, child_label)] = frozenset(slots)
