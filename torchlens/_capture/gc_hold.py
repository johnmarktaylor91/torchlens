"""Hold off the cyclic garbage collector's automatic runs for the length of one capture.

A capture allocates many container objects (records, rows, dicts) that live
until the trace is dropped. At CPython's default generation-0 threshold
those allocations trigger a young-generation collection every few dozen ops,
and each one traverses everything allocated since the last, tensors
included, while finding almost nothing to free. On a small transformer that
work was a measurable share of the capture forward.

The hold does two things for the outermost capture in the process:

- the generation-0 threshold is raised to :data:`HOLD_GEN0_THRESHOLD` (never
  lowered), so collections during the capture run rarely;
- ``gc.freeze()`` moves every object tracked before the capture into the
  permanent generation, so a full collection that a long capture still
  triggers scans only objects the capture (and the model) made, not the
  whole program.

The collector stays enabled: a model that makes cyclic garbage on every op
still has it collected during the capture, so its memory stays bounded. On
every exit the thresholds return to the exact values seen at entry and what
the hold froze is unfrozen. The hold never forces a collection: the user's
restored thresholds and the collector's own counters decide when the next
one runs, exactly as they would after any allocation burst.

Left alone on purpose: a collector that is already disabled at entry (no
freeze, no threshold change; the caller's choice stands), and a process that
already holds frozen objects (``gc.get_freeze_count() > 0``, a pre-fork
server, for one): the hold then only raises the threshold, because
``gc.unfreeze()`` cannot tell the caller's frozen objects from ours.

Recorded values do not depend on collection timing: identity-cache hits
compare object identity through a live weak reference, and the save budget's
release credit fires when a payload's reference count reaches zero, which
the hold does not delay. Only the release time of cyclic garbage the model
makes changes, and it stays bounded by the raised threshold.

Nested and concurrent captures share one process-wide hold: the first entry
takes it and the last exit restores it.
"""

from __future__ import annotations

import gc
import threading

#: Generation-0 threshold while a capture runs (CPython's default is 700); the
#: same value the stats fast loop uses. A capture makes about 50 tracked
#: objects per op, so a collection runs about every 2000 ops: small captures
#: see none, and a model that makes cyclic garbage on every op still has it
#: collected. Applied only when the collector is enabled and the user's own
#: threshold is lower and nonzero (zero means automatic collection is off).
HOLD_GEN0_THRESHOLD = 100_000


class _HoldState:
    """Process-wide hold bookkeeping, guarded by ``_LOCK``."""

    __slots__ = ("depth", "froze", "thresholds")

    def __init__(self) -> None:
        self.depth = 0
        self.froze = False
        self.thresholds: tuple[int, ...] = ()


_LOCK = threading.Lock()
_STATE = _HoldState()


def hold_collector() -> None:
    """Enter the hold; only the outermost entry changes the collector.

    Every call must be paired with one :func:`release_collector` call, in a
    ``finally`` that runs on every exit path.
    """

    with _LOCK:
        _STATE.depth += 1
        if _STATE.depth > 1:
            return
        thresholds = gc.get_threshold()
        _STATE.thresholds = thresholds
        enabled = gc.isenabled()
        _STATE.froze = enabled and gc.get_freeze_count() == 0
        if _STATE.froze:
            gc.freeze()
        if enabled and 0 < thresholds[0] < HOLD_GEN0_THRESHOLD:
            gc.set_threshold(HOLD_GEN0_THRESHOLD, *thresholds[1:])


def release_collector() -> None:
    """Leave the hold; the last exit restores the thresholds and unfreezes."""

    with _LOCK:
        if _STATE.depth == 0:
            return
        _STATE.depth -= 1
        if _STATE.depth:
            return
        thresholds, froze = _STATE.thresholds, _STATE.froze
        _STATE.thresholds, _STATE.froze = (), False
        try:
            gc.set_threshold(*thresholds)
        finally:
            if froze:
                gc.unfreeze()
