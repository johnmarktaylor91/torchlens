"""The flag the torch wrappers read before a write, instead of the pending-alias registry.

When Dynamo's trace-rule tables predate the first wrap, Dynamo inlines the TorchLens
wrappers and guards every object they read. Reading the registry dict there guarded its
storage keys: every capture forced a recompile, and a weakref callback that dropped a
guarded key during a compile raised ``KeyError`` while Dynamo built its guards. The gate
gives Dynamo one boolean to guard, which changes only when the registry fills or empties.

A leaf: it imports nothing from TorchLens, so the shared tensor utilities and the
wrappers can both bind it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def _materialize_unset(tensors: Iterable[Any]) -> None:
    """Fail loudly if a wrapper writes before the tensor utilities bind the copy-out."""

    raise RuntimeError("torchlens deferred-payload gate used before its materializer was bound")


class _DeferGate:
    """Whether pending payload aliases may exist, and the copy-out to run before a write.

    ``armed`` turns True at every pending-alias registration and False only when a
    materialize call finds the registry empty, so it is never False while an alias is
    pending. ``materialize`` is ``materialize_deferred_for_call``, or its
    ``torch.compiler.disable`` form once Dynamo can inline the wrappers, so a compiled
    write breaks the graph and copies out at run time instead of being traced.
    """

    def __init__(self) -> None:
        self.armed = False
        self.materialize: Callable[[Iterable[Any]], None] = _materialize_unset


_DEFER_GATE = _DeferGate()
