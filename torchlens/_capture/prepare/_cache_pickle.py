"""Pickle reductions that keep a capture-cache entry loadable on every supported torch."""

from __future__ import annotations

import copyreg
from typing import Any


def _untyped_storage_from_bytes(byte_view: Any) -> Any:
    """Rebuild a bare ``torch.UntypedStorage`` from the ``uint8`` tensor it was pickled as.

    Parameters
    ----------
    byte_view
        A ``uint8`` tensor spanning every byte of the storage.

    Returns
    -------
    torch.UntypedStorage
        The tensor's storage.
    """

    return byte_view.untyped_storage()


def _reduce_untyped_storage(storage: Any) -> tuple[Any, tuple[Any]]:
    """Pickle a bare ``torch.UntypedStorage`` as a ``uint8`` tensor over its bytes.

    Torch pickles a bare untyped storage through its legacy serializer, whose
    loader reads ``storage_type.dtype`` and raises ``AttributeError`` for
    ``UntypedStorage`` (torch 2.1 through 2.13). An op that takes a storage
    (``Tensor.set_(storage)``) keeps one in its recorded and saved arguments, so
    the entry was written and could never be loaded. A tensor pickles through the
    typed path, which loads on every supported torch.

    Parameters
    ----------
    storage
        The storage to pickle.

    Returns
    -------
    tuple[Any, tuple[Any]]
        The rebuild function and its single argument, the byte view.
    """

    import torch

    byte_view = torch.empty(0, dtype=torch.uint8, device=storage.device)
    byte_view.set_(storage)
    return _untyped_storage_from_bytes, (byte_view,)


def capture_cache_dispatch_table() -> dict[Any, Any]:
    """Return the pickler dispatch table the capture cache writes entries with.

    Returns
    -------
    dict[Any, Any]
        The process-wide ``copyreg`` table plus the bare-storage reduction.
    """

    import torch

    return {**copyreg.dispatch_table, torch.UntypedStorage: _reduce_untyped_storage}
