"""The code-context cache lookup for one selection of kept frames.

:func:`~torchlens._utils.introspection._get_code_context` names its selected
frames as ``(code, lineno, lasti)`` triples and keys the per-capture
code-context cache on them plus its options. Module entry's incremental lookup
(``backends/torch/_module_entry_site``) builds the same name without the walk
and must look it up exactly as the walk does, so both use this function.
Module entry also needs the name the walk selected when it does run the walk:
:func:`record_selections` makes this thread's lookups append their names to a
list for the length of one walk.
"""

from __future__ import annotations

import threading
from types import CodeType
from typing import Any

__all__ = ("context_cache_lookup", "record_selections")

#: One kept frame as the cache key names it.
Named = tuple[CodeType, int, int]

#: Per thread: the list this thread's lookups append their names to, or ``None``.
_selections = threading.local()


def record_selections(out: list[Any] | None) -> list[Any] | None:
    """Make this thread's lookups append their names to ``out`` (``None`` stops); return the previous list."""

    previous: list[Any] | None = getattr(_selections, "out", None)
    _selections.out = out
    return previous


def context_cache_lookup(
    named: tuple[Named, ...], options: tuple[int, bool, bool], context_cache: dict[Any, Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...] | None]:
    """Return ``(cache_key, cached)`` for the selection ``named`` under ``options``.

    ``cached`` is the tuple of locations a capture already built for this key, or
    ``None`` (build them and store them under ``cache_key``). Value-equal code
    objects from two files share the first key; a cached entry whose files
    differ from ``named``'s moves the key to one that also names the files.
    """

    out = getattr(_selections, "out", None)
    if out is not None:
        out.append(named)
    cache_key: tuple[Any, ...] = (*options, named)
    cached = context_cache.get(cache_key)
    if cached is None:
        return cache_key, None
    for loc, (code, _lineno, _lasti) in zip(cached, named, strict=True):
        if loc.file != code.co_filename:  # a value-equal code from another file
            cache_key = (*cache_key, tuple(c.co_filename for c, _l, _a in named))
            return cache_key, context_cache.get(cache_key)
    return cache_key, cached
