"""Process-wide slots of the host-nondeterminism monitor: the active window and patch stacks.

Kept apart from the monitor class so the stream helpers (:mod:`.streams`, :mod:`.seeding`)
can bracket their own torch RNG touches under :func:`_suppress_active_monitor_marks`
without importing the monitor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only cycle
    from .monitor import host_nondeterminism_monitor

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_ACTIVE_MONITOR: host_nondeterminism_monitor | None = None
"""The capture-scoped monitor currently installed, or ``None`` (r41 hon2_1).

Published as the LAST statement of ``__enter__`` and restored to the PREVIOUS occupant
FIRST in the teardown, so readers never observe a partially-installed window and a
nested window's exit cannot null the slot mid-outer-window. Captures do not nest
(``active_logging`` rejects nested captures), so a single slot is sufficient.
"""


class _PatchStackEntry:
    """One live monitor patch on ``(holder, name)``, spliceable out of order.

    ``holder`` is retained STRONGLY so its ``id()`` cannot be reused by another object
    while the entry is stacked (the stack is keyed by ``(id(holder), name)``).
    """

    __slots__ = ("holder", "name", "original", "wrapper")

    def __init__(self, holder: Any, name: str, original: Any, wrapper: Any) -> None:
        self.holder = holder
        self.name = name
        self.original = original
        self.wrapper = wrapper


_PATCH_STACKS: dict[tuple[int, str], list[_PatchStackEntry]] = {}
"""Live monitor patches per ``(id(holder), name)``, innermost last.

Restoration is SPLICE-aware. A monitor unwinding out of order (a non-LIFO overlap) used
to hand the true original back and then have the outer window's restore write ITS
snapshot -- which was the inner wrapper -- leaving ``time.time`` / ``os.urandom`` /
``np.random.default_rng`` wrapped for the life of the process. Splicing instead rewrites
the successor entry's recorded original, so whoever restores last always writes the
genuine pre-monitor value.
"""


@contextmanager
def _suppress_active_monitor_marks() -> Iterator[None]:
    """Suppress channel marks for a TorchLens-OWNED RNG bookkeeping bracket (r65 Z).

    TorchLens's own seed / snapshot / restore helpers legitimately touch the torch RNG
    APIs that the r65 registry rows monitor -- per-op state logging and in-forward
    intervention replay run INSIDE the capture window (``set_rng_from_saved_states``
    calls the patched ``torch.random.set_rng_state``). A TorchLens-initiated
    restore is NOT model host nondeterminism, so these helpers bracket themselves with
    the active monitor's ``_mark``-choke-point suppression (surface-complete: wrapper
    marks, held-code call marks, and default-generator receiver marks all funnel
    through the same choke). USER code never executes inside these brackets -- the
    replayed op itself runs OUTSIDE the bracket in
    :func:`execute_with_restored_rng_autocast`. No active monitor -> no-op.
    """

    monitor = _ACTIVE_MONITOR
    if monitor is None:
        yield
        return
    with monitor._monitor_internal_probe():
        yield


def _skip_retired_hooks(candidate: Any, predecessor_attr: str) -> Any:
    """Return the first chain link that is not a torn-down monitor's hook.

    A non-LIFO overlap used to restore an already-retired window's hook into a
    profile slot; the sys slot self-heals on its next event, but the
    ``threading`` registration only seeds NEW threads, so a dead hook parked
    there never fires again on the main thread and taxes (and misclassifies
    into) every later capture. Walking each candidate's owning monitor lets the
    restore skip straight to the newest LIVE link.
    """

    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        owner = getattr(candidate, "_tl_owner", None)
        if owner is None or not getattr(owner, "_hooks_retired", False):
            break
        # Key the predecessor off WHICH of the owner's two hooks this link IS,
        # not off the slot being restored: a thread spawned during a previous
        # window carries that window's THREADING hook even when the slot under
        # restore is the sys slot, and following the slot's attr handed it the
        # dead owner's sys predecessor instead of the threading chain
        # (grind-r5 b8 R57).
        if candidate is getattr(owner, "_threading_hook", None):
            candidate = getattr(owner, "_previous_threading_profile", None)
        elif candidate is getattr(owner, "_sys_hook", None):
            candidate = getattr(owner, "_previous_sys_profile", None)
        else:
            candidate = getattr(owner, predecessor_attr, None)
    return candidate
