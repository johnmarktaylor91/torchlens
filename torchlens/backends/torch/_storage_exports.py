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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..._utils._internal_read import _internal_read_state
from ..._utils.tensor_utils import _DEFER_PENDING, materialize_deferred_for_call

__all__ = ("STORAGE_EXPORT_FUNC_NAMES", "materialize_deferred_for_export")

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
