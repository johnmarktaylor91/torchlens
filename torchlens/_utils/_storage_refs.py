"""Whether a payload may safely carry a weak reference on this torch."""

from __future__ import annotations

import torch

from . import _torch_compat


def payload_weakref_is_safe(payload: object) -> bool:
    """Return whether a weak reference to ``payload`` is safe to hold.

    Parameters
    ----------
    payload:
        Tensor or storage object a caller wants to watch through a weak reference.

    Returns
    -------
    bool
        True for tensors, and for storages on a torch that keeps one storage
        wrapper per storage (``HAS_CACHED_UNTYPED_STORAGE_WRAPPER``). False for a
        storage on torch 2.1, where a weak reference to the per-call wrapper can
        leave freed memory behind and crash CPython later.
    """

    return isinstance(payload, torch.Tensor) or _torch_compat.HAS_CACHED_UNTYPED_STORAGE_WRAPPER
