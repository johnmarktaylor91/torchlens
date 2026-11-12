"""An identity-keyed weak registry whose reads skip ``WeakIdRef``'s Python methods.

``torch.utils.weak.WeakIdKeyDictionary.get`` builds a ``WeakIdRef`` for the key
(a Python ``__init__``), hashes it (a Python ``__hash__``) and, on a hit,
compares it with the stored key (a Python ``__eq__`` calling the Python
``__call__`` of both refs, each probing the referent for ``_fix_weakref``; on an
``nn.Module`` that probe runs ``Module.__getattr__`` and raises). The module and
parameter registries are read several times per module call and per op.

:class:`IdKeyRegistry` keeps those semantics and adds a read cache keyed by
``id(key)``: each entry is ``(weakref to the key, value)``, and a read is served
from it only when the weakref still returns the very object asked about, so a
dead key or a reused id falls through to the inherited lookup. Every write path
(set, delete, pop, setdefault, update, popitem, clear) drops the key's cache
entry before it changes the mapping, and a cache entry's weakref callback drops
it when its key dies, so the cache never answers with a value the mapping no
longer holds for that object. The only difference is that a cache hit does not
call the key's ``_fix_weakref`` hook (a no-op for an object the caller holds a
strong reference to, which every caller does while asking about it).
"""

from __future__ import annotations

import weakref
from typing import Any

from torch.utils.weak import WeakIdKeyDictionary

__all__ = ("IdKeyRegistry",)

_MISSING = object()


class IdKeyRegistry(WeakIdKeyDictionary):
    """``WeakIdKeyDictionary`` with an identity-checked read cache (see the module docstring)."""

    def __init__(self) -> None:
        super().__init__()
        self._reads: dict[int, tuple[weakref.ref[Any], Any]] = {}

    def _remember(self, key: Any, value: Any) -> None:
        key_id = id(key)
        reads = self._reads

        def forget(ref: weakref.ref[Any], key_id: int = key_id) -> None:
            entry = reads.get(key_id)
            if entry is not None and entry[0] is ref:
                del reads[key_id]

        reads[key_id] = (weakref.ref(key, forget), value)

    def _forget(self, key: Any) -> None:
        self._reads.pop(id(key), None)

    def get(self, key: Any, default: Any = None) -> Any:
        entry = self._reads.get(id(key))
        if entry is not None and entry[0]() is key:
            return entry[1]
        value = super().get(key, _MISSING)
        if value is _MISSING:
            return default
        self._remember(key, value)
        return value

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key: Any) -> bool:
        entry = self._reads.get(id(key))
        if entry is not None and entry[0]() is key:
            return True
        return super().__contains__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self._forget(key)
        super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        self._forget(key)
        super().__delitem__(key)

    def pop(self, key: Any, *args: Any) -> Any:
        self._forget(key)
        return super().pop(key, *args)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        self._forget(key)
        return super().setdefault(key, default)

    def update(self, dict: Any = None, **kwargs: Any) -> None:  # noqa: A002 - the base's name
        self._reads.clear()
        super().update(dict, **kwargs)

    def popitem(self) -> Any:
        self._reads.clear()
        return super().popitem()

    def clear(self) -> None:
        self._reads.clear()
        super().clear()
