"""The explicit marker for TorchLens-internal scalar reads during capture.

A leaf by design: the tensor-label helpers (``_tl``) and the shared tensor
utilities wrap their own bookkeeping reads in :func:`internal_scalar_read`,
so this marker must import nothing from the completeness package that those
modules feed.

At import it registers the wrapper pass-through predicate and the unwrap ledger
with :func:`torchlens._utils._torch_symbols.register_internal_reads`, so the
low-level routed tensor reads can reach the saved originals without importing
this layer.

It also holds :class:`IdKeyRegistry`, the identity-keyed weak mapping the
module and parameter metadata registries (``_tensor_label``) use, because those
registries' reads are TorchLens-internal bookkeeping reads too and the class is
a leaf in the same sense (weakref and torch only). Its reads skip
``WeakIdRef``'s Python methods, which
``torch.utils.weak.WeakIdKeyDictionary.get`` runs on every read: it builds a ``WeakIdRef`` for the key
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

import threading
import weakref
from typing import Any

from torch.utils.weak import WeakIdKeyDictionary

from .. import _state
from ._torch_symbols import register_internal_reads

__all__ = ("internal_scalar_read", "_internal_read_state", "IdKeyRegistry")

_internal_read_state = threading.local()
"""Per-thread depth counter for the explicit TorchLens internal-scalar-read marker."""


class _InternalScalarRead:
    """The context manager :func:`internal_scalar_read` returns.

    A slotted class rather than a ``@contextmanager`` generator: capture enters this
    marker tens of thousands of times per trace, and the generator protocol costs a
    generator, a helper object and a ``StopIteration`` per entry. Same order and
    restores as the generator it replaced: depth up, then the fingerprint hold;
    on exit (also on an exception, never suppressed) the hold is released, then the
    depth restored.
    """

    __slots__ = ("_depth", "_counted")

    def __enter__(self) -> None:
        depth = getattr(_internal_read_state, "depth", 0)
        self._depth = depth
        _internal_read_state.depth = depth + 1
        # An internal read is not a model call: keep it out of the ordered call
        # fingerprint under the same owner-thread rule as ``pause_logging()``.
        self._counted = _state.hold_fingerprint()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            _state.release_fingerprint(self._counted)
        finally:
            _internal_read_state.depth = self._depth


def internal_scalar_read() -> _InternalScalarRead:
    """Mark a region as a genuine TorchLens capture-internal scalar/comparison read.

    TorchLens itself reads a scalar/bool from a tensor during capture -- extracting a
    scalar-``bool`` op-output value (``ops._log_output_tensor_info``), comparing a
    pre-call input copy against its post-call value for mutation/alias detection
    (``tensor_nanequal`` via ``detect_torch_alias_contract`` and the child-version
    snapshot), and similar bookkeeping reads. Those reads lower to
    ``aten._local_scalar_dense`` (or, for content comparisons, ``aten.equal`` /
    ``aten.allclose`` returning a Python ``bool``) and would otherwise be misrecorded as
    USER host escapes and falsely trip the fail-closed INCOMPLETE gates. This context
    manager marks them EXPLICITLY so the internal-vs-user classifier is an
    allowlist-BY-CONSTRUCTION: an escape is "internal" iff this marker is active, never
    because a stack frame's filename happens to resolve inside the ``torchlens`` package
    (which is spoofable by an ``exec``-compiled user helper carrying a torchlens
    ``co_filename``, and fails for a frameless C-callable). A genuine user escape runs
    with the marker inactive and is always recorded.
    """

    return _InternalScalarRead()


_MISSING = object()


class IdKeyRegistry(WeakIdKeyDictionary):
    """``WeakIdKeyDictionary`` with an identity-checked read cache (see the module docstring)."""

    def __init__(self) -> None:
        super().__init__()
        self._reads: dict[int, tuple[weakref.ref[Any], Any]] = {}

    def _remember(self, key: Any, value: Any) -> None:
        """Cache ``value`` for ``key``, dropping the entry when ``key`` is collected."""
        key_id = id(key)
        reads = self._reads

        def forget(ref: weakref.ref[Any], key_id: int = key_id) -> None:
            """Drop the cached read for ``key_id`` if it still belongs to ``ref``."""
            entry = reads.get(key_id)
            if entry is not None and entry[0] is ref:
                del reads[key_id]

        reads[key_id] = (weakref.ref(key, forget), value)

    def _forget(self, key: Any) -> None:
        """Drop any cached read for ``key``."""
        self._reads.pop(id(key), None)

    def get(self, key: Any, default: Any = None) -> Any:
        """Return the value for ``key``, or ``default``; serve repeat reads from the cache."""
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
        """Remove ``key`` and return its value, invalidating its cached read."""
        self._forget(key)
        return super().pop(key, *args)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Insert ``default`` for a missing ``key`` and return the stored value."""
        self._forget(key)
        return super().setdefault(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update from a mapping or pairs, clearing the read cache."""
        self._reads.clear()
        super().update(*args, **kwargs)

    def popitem(self) -> Any:
        """Remove and return an arbitrary ``(key, value)`` pair, clearing the read cache."""
        self._reads.clear()
        return super().popitem()

    def clear(self) -> None:
        """Remove every entry and clear the read cache."""
        self._reads.clear()
        super().clear()


register_internal_reads(_state.internal_reads_pass_through, _state.wrap_epoch_ledgers()[0])
