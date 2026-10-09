"""Per-tensor metadata caches that stay correct when a tensor's ``id`` is reused.

``BufferWriteTracker`` caches storage keys and byte ranges between operation
boundaries. ``id()`` alone is not an identity across a window: a tensor freed
mid-window hands its id to the next allocation, and serving the newcomer the
dead tensor's storage identity mis-roots (or misses) a buffer alias. Each entry
therefore carries a weak reference and counts as a hit only while that reference
still resolves to the very object being asked about (split out of
``buffer_writes.py`` under the R43 file-size ratchet).
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TypeVar

import torch

_CachedT = TypeVar("_CachedT")

#: ``(id, version counter) -> (weak reference to the tensor, cached value)``.
IdentityCache = dict[tuple[int, int | None], tuple["weakref.ReferenceType[torch.Tensor]", _CachedT]]


def identity_cached(
    cache: IdentityCache[_CachedT],
    tensor: torch.Tensor,
    version: int | None,
    compute: Callable[[torch.Tensor], _CachedT],
) -> _CachedT:
    """Return ``compute(tensor)`` through an id-keyed cache that is safe under id reuse.

    Parameters
    ----------
    cache:
        Table owned by the caller and cleared at its own boundaries.
    tensor:
        Tensor whose metadata is requested.
    version:
        The tensor's version counter (part of the key, so an in-place write
        invalidates the entry).
    compute:
        Uncached metadata reader.

    Returns
    -------
    _CachedT
        The cached value when the entry was recorded for this very object,
        otherwise a freshly computed one (cached when the tensor accepts a weak
        reference).
    """

    cache_key = (id(tensor), version)
    entry = cache.get(cache_key)
    if entry is not None and entry[0]() is tensor:
        return entry[1]
    value = compute(tensor)
    try:
        cache[cache_key] = (weakref.ref(tensor), value)
    except TypeError:
        cache.pop(cache_key, None)
    return value
