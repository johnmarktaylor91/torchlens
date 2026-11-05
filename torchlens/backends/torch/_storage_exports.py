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
(:func:`note_storage_export`, in the capture's ``ForeignJournal``). A later call that reads a tensor on that storage with a
value other than its producer's saved out stores a per-child version snapshot like any
changed parent; the write that changed it ran outside torch and is in no recorded op.
:func:`note_foreign_written_snapshot` marks that snapshot with the exact argument slots
the parent was bound to, only when each of those slots holds the parent's own labeled
tensor on the very storage that was exported, and forward validation accepts a snapshot
it cannot otherwise explain only for those slots (``validation/_snapshot_binding.py``).
The journal keeps a weak reference per exported storage, so it keeps nothing alive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from ... import _state
from ..._utils._internal_read import _internal_read_state
from ..._utils._tensor_label import (
    get_tensor_meta,
    session_label_storage_intact,
    session_meta_is_anchored,
)
from ..._utils.collections import index_nested
from ..._utils.tensor_utils import _DEFER_PENDING, materialize_deferred_for_call
from ._foreign_data import FOREIGN_DATA_DEPARTURES_ATTR, ForeignJournal, foreign_journal

__all__ = (
    "STORAGE_EXPORT_FUNC_NAMES",
    "materialize_deferred_for_export",
    "note_foreign_written_snapshot",
    "note_storage_export",
)

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


def _storage_of(tensor: Any) -> tuple[tuple[str, int], Any] | None:
    """Return ``((device, storage data pointer), untyped storage)``, or ``None`` when unreadable."""

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
    return None if key[1] == 0 else (key, storage)


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
    if not args:
        return
    if getattr(_internal_read_state, "depth", 0):
        return
    found = _storage_of(args[0])
    if found is not None:
        # A weak reference to the storage itself, not the tensor: nothing exported is
        # kept alive, and an expired reference never lets a reused pointer pass for an
        # exported storage.
        key, storage = found
        foreign_journal(trace).exported[key] = StorageWeakRef(storage)


def _on_exported_parent_storage(exported: Mapping[Any, Any], value: Any, parent_label: str) -> bool:
    """Return whether ``value`` is the parent's own labeled tensor on a still-exported storage.

    A write outside torch dispatch changes the parent's value only through the
    parent's own storage, so the slot must hold the object stamped with the parent's
    label in this session, on the storage it was stamped with (``label_storage``), and
    that very storage (same storage object, alive) must have been exported.
    """

    meta = get_tensor_meta(value) if isinstance(value, torch.Tensor) else None
    if meta is None or meta.label_raw != parent_label or not session_meta_is_anchored(meta):
        return False
    if not session_label_storage_intact(meta, value):
        return False
    found = _storage_of(value)
    if found is None:
        return False
    ref = exported.get(found[0])
    return ref is not None and not ref.expired() and ref == StorageWeakRef(found[1])


def note_foreign_written_snapshot(
    trace: Any,
    parent_label: str,
    child_label: str,
    parent_arg_positions: Mapping[str, Mapping[Any, str]],
    live_args: Sequence[Any],
    live_kwargs: Mapping[str, Any],
) -> None:
    """Mark a parent's version snapshot whose live slot sits on an exported storage.

    The caller passes only snapshots taken because the slot's value differed from the
    parent's saved out, never ones the child's own write or alias contract asked for:
    the live slot is read after the call, so it shows the pre-call storage only when
    the child neither wrote nor rebound it.

    Parameters
    ----------
    trace:
        Live capture trace.
    parent_label, child_label:
        Raw labels of the parent and of the child the snapshot was stored for.
    parent_arg_positions:
        The child's ``{"args": {position: label}, "kwargs": {name: label}}`` binding.
    live_args, live_kwargs:
        The child's live positional and keyword arguments.

    The mark is set only when every slot bound to the parent holds the parent's own
    labeled tensor on its stamp-time storage and that storage was exported
    (:func:`_on_exported_parent_storage`): a slot holding any other tensor, on an
    exported storage or not, is a binding no foreign write explains.
    """

    journal = trace.__dict__.get(FOREIGN_DATA_DEPARTURES_ATTR)
    if not isinstance(journal, ForeignJournal) or not journal.exported:
        return
    slots = [
        (kind, position)
        for kind in ("args", "kwargs")
        for position, label in (parent_arg_positions.get(kind) or {}).items()
        if label == parent_label
    ]
    sources = {"args": live_args, "kwargs": live_kwargs}
    if slots and all(
        _on_exported_parent_storage(journal.exported, _slot_value(sources[kind], pos), parent_label)
        for kind, pos in slots
    ):
        journal.written_slots[(parent_label, child_label)] = frozenset(slots)
