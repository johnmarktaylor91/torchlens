"""Identity memo for the frame-reachable inventory walk.

The deep inventory walk (:mod:`.deep_inventory`) remembers every object it has
visited for the whole RNG-monitor window, so a capture that keeps referencing
the same namespaces pays only lookups. A memo keyed by address alone is unsound:
an object the walk visited can be freed inside the window and a new object can
take its address, which a bare address set then skips. A generator behind that
new object is never digested, and the window can report a false VERIFIED.

Every entry here can therefore prove the identity it was keyed by:

* a weakly referenceable object is kept through a weak reference, so the memo
  never extends its life; once it dies its entry stops matching and a new
  object at the same address is walked;
* any other object (``dict``, ``list``, ``tuple`` and the like) is held
  strongly, which keeps its address from being reused until the memo is
  released.

The monitor drops its memo at teardown, which every exit of the window runs,
and the walk releases it when the node budget is exhausted, after which the
walk never runs again in that window.
"""

from __future__ import annotations

import weakref as _weakref_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .monitor import host_nondeterminism_monitor

#: Base ``type`` getset, so a metaclass attribute of the same name never runs.
_weakref_offset = type.__dict__["__weakrefoffset__"].__get__
_MISSING = object()


class HeldWalkMemo:
    """Visited objects of one monitor window, keyed by address, checked by identity."""

    __slots__ = ("_strong", "_weak")

    def __init__(self) -> None:
        self._strong: dict[int, Any] = {}
        self._weak: dict[int, _weakref_module.ref[Any]] = {}

    def __len__(self) -> int:
        """Return the number of entries, live or dead."""

        return len(self._strong) + len(self._weak)

    def holds(self, value: Any) -> bool:
        """Return whether ``value`` itself, not just its address, was visited."""

        key = id(value)
        if self._strong.get(key, _MISSING) is value:
            return True
        ref = self._weak.get(key)
        return ref is not None and ref() is value

    def claim(self, value: Any) -> bool:
        """Record ``value`` as visited; return False when it already was."""

        if self.holds(value):
            return False
        key = id(value)
        if _weakref_offset(type(value)):
            try:
                self._weak[key] = _weakref_module.ref(value)
            except TypeError:
                pass
            else:
                return True
        # A dead weak entry at this address is stale; the strong entry replaces it.
        self._weak.pop(key, None)
        self._strong[key] = value
        return True

    def counts(self) -> tuple[int, int]:
        """Return ``(strongly held, weakly held)`` entry counts."""

        return len(self._strong), len(self._weak)

    def release(self) -> None:
        """Drop every entry, releasing the objects held strongly."""

        self._strong.clear()
        self._weak.clear()


def exhaust_deep_inventory(monitor: host_nondeterminism_monitor) -> None:
    """Flag the walk's budget as exhausted and release its memo.

    The walk never runs again in this window once the flag is set, so nothing
    it held has a reason to stay alive until teardown.
    """

    monitor._deep_inventory_exhausted = True
    monitor._flag_uncertain("deep_inventory_budget_exhausted")
    monitor._deep_walk_seen_ids.release()
