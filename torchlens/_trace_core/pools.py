"""Per-trace exact-type intern pools.

The pool key contract is absorbed VERBATIM from the shipped Op metadata
compaction (``op.py`` ``_pool_key``): keys carry the exact type, so ``True``,
``1``, ``1.0``, and ``Bytes(1)`` can never collapse into one entry. Pools are
per ``TraceCore`` — NEVER process-global — and only immutable value objects
enter them; payload tensors, callables, live handles, and mutable user
objects never do.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any


class InternPool:
    """Interning table mapping values to dense int ids and back."""

    __slots__ = ("_by_key", "_values", "tokens")

    def __init__(self) -> None:
        """Create an empty pool."""

        self._by_key: dict[Hashable, int] = {}
        self._values: list[Any] = []
        #: The trace's write-time container tokens (:class:`ContainerTokens`),
        #: handed over when the capture workspaces drop and taken by the
        #: freeze's compaction; ``None`` before and after.
        self.tokens: ContainerTokens | None = None

    def __len__(self) -> int:
        """Return the number of distinct interned values."""

        return len(self._values)

    @staticmethod
    def key_for(value: Any) -> Hashable:
        """Return the exact-type injective pool key for a value."""

        return (type(value), value)

    def intern(self, value: Any) -> int:
        """Return the pool id for ``value``, interning it if new."""

        key = self.key_for(value)
        pool_id = self._by_key.get(key)
        if pool_id is None:
            pool_id = len(self._values)
            self._by_key[key] = pool_id
            self._values.append(value)
        return pool_id

    def value(self, pool_id: int) -> Any:
        """Return the interned value for a pool id."""

        return self._values[pool_id]


class ClosurePool:
    """Interned ancestor-closure pool: one entry per distinct closure.

    The column contract is a pool id; the encoding (frozenset here; a dense
    bitset over a trace-local label table is an equivalent internal choice)
    is invisible to consumers, which always receive a fresh mutable ``set``
    on materialization.
    """

    __slots__ = ("_by_closure", "_closures")

    def __init__(self) -> None:
        """Create an empty closure pool."""

        self._by_closure: dict[frozenset, int] = {}
        self._closures: list[frozenset] = []

    def __len__(self) -> int:
        """Return the number of distinct closures."""

        return len(self._closures)

    def intern(self, members: Any) -> int:
        """Intern one closure (any iterable of labels) and return its id."""

        closure = frozenset(members)
        closure_id = self._by_closure.get(closure)
        if closure_id is None:
            closure_id = len(self._closures)
            self._by_closure[closure] = closure_id
            self._closures.append(closure)
        return closure_id

    def materialize(self, closure_id: int) -> set:
        """Return a FRESH mutable set for one closure id."""

        return set(self._closures[closure_id])


class ContainerTokens:
    """Write-time content tokens of fresh container cells, owned by one trace.

    A writer that builds a fresh exact ``dict`` from a few scalar reads it
    already makes registers it here with a token: an immutable value equal
    for two containers exactly when their contents are equal, and fixed from
    the write to the freeze (nothing mutates a registered container). The
    freeze's container pooling then keys a registered cell by its token
    instead of walking its content (:mod:`torchlens._data_classes._container_key`):
    the first cell of each token computes the real content key once and the
    rest reuse it, so every cell ends with the same key, and the same pooled
    cell, as a full walk would give it. The alias census still runs, so a
    registered container that is shared with another cell is still never
    pooled, and the containers nested in a registered one are checked against
    the census as the full walk checks them.

    Registered containers hold only members the immutable pool can never
    replace (``bool``, ``None``, ``torch.dtype`` singletons and nested dicts of
    those), so the member sweep skips them as well.

    Lifetime: the writers fill one plain table per trace's runtime workspace
    (``WrapperRuntimeWorkspace.container_tokens``); when the workspaces drop it
    moves, wrapped here, to that trace's ``TraceCore.pool.tokens``, and the
    freeze's compaction takes it from there and drops it. It is never shared
    between traces or held by the process.
    """

    __slots__ = ("by_id", "keys")

    def __init__(self, by_id: dict[int, tuple[Any, Hashable]] | None = None) -> None:
        """Wrap the writers' registration table (a fresh empty one by default)."""

        #: ``id(container) -> (container, token)``, filled by the
        #: writers as a plain dict; holding the container keeps its id from
        #: being reused while it is registered.
        self.by_id: dict[int, tuple[Any, Hashable]] = {} if by_id is None else by_id
        #: ``token -> content key`` cache filled at compaction.
        self.keys: dict[Hashable, Any] = {}

    def __len__(self) -> int:
        """Return the number of registered containers."""

        return len(self.by_id)
