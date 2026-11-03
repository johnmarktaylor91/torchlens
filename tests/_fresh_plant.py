"""A 64 MiB host plant that is a fresh anonymous mapping by construction.

``np.ones(8 * 2**20)`` is a fresh mapping only while glibc's heap holds no free
chunk that large. Earlier tests in the same process can leave one: after a
freed block raises the dynamic mmap threshold, blocks below it come from the
heap, and when they are freed they coalesce into a resident free chunk that
malloc serves a 64 MiB request from before it would map anything, so writing
the "plant" touches no new page and the resident set does not grow at all.
A private anonymous ``mmap`` with every page written always grows the
resident set by its full size, whatever the allocator's history.

Releasing the plant is the measured event, so :meth:`FreshPlant.close` makes it
read the same on every run. Linux keeps the RSS counters per CPU and sets the
``VmHWM`` mark at unmap time from the approximate total, which leaves out each
CPU's unfolded residue (up to one counter batch, ``max(32, 2 * CPUs)`` pages:
1.75 MiB on a 224-CPU host). TorchLens adds back the residue it can read when
it reads the mark, but the unmap itself folds the residue of the CPU it runs
on, so that CPU's share at the moment of the unmap is gone for good and the
mark reads short by a varying amount below one batch. ``close`` pins the
thread to one CPU, folds that CPU's anonymous-page counter (writing fresh
pages until the kernel moves the residue into the total), unmaps the plant,
then folds again, so that CPU holds no residue at the unmap or at the read.
Hosts without ``sched_setaffinity`` or a readable ``/proc/self/stat`` just unmap.
"""

from __future__ import annotations

import mmap
import os

import numpy as np

PLANT_BYTES = 64 * 2**20
_PAGE = mmap.PAGESIZE
#: Pages written after the unmap; kept mapped until the next plant is released
#: so their own unmap never lands between a bracket's peak and its read.
_HELD: list[mmap.mmap] = []


def _lag_pages() -> int | None:
    """Exact RSS (``statm``) minus the kernel's approximate RSS (``/proc/self/stat``), in pages."""

    try:
        with open("/proc/self/statm") as handle:
            exact = int(handle.read().split()[1])
        with open("/proc/self/stat") as handle:
            text = handle.read()
        return exact - int(text[text.rindex(")") + 2 :].split()[21])
    except (OSError, ValueError, IndexError):
        return None


def _fold_this_cpu(keep: list[mmap.mmap]) -> None:
    """Write fresh anonymous pages on this (pinned) CPU until its counter folds.

    Each written page adds one to this CPU's share, which the kernel moves into
    the total once it reaches the batch; that move is the first fall in the
    lag. A share below zero needs up to two batches, hence the region size.
    """

    batch = max(32, 2 * (os.cpu_count() or 1))
    pages = 2 * batch + 64
    region = mmap.mmap(-1, pages * _PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    keep.append(region)
    if hasattr(mmap, "MADV_NOHUGEPAGE"):
        region.madvise(mmap.MADV_NOHUGEPAGE)  # one page per fault, never 512
    previous = _lag_pages()
    if previous is None:
        return
    for page in range(pages):
        region[page * _PAGE] = 1
        lag = _lag_pages()
        if lag is None or lag < previous:
            return
        previous = lag


class FreshPlant:
    """The resident 64 MiB region; :meth:`close` unmaps it with this CPU's counter folded."""

    def __init__(self, region: mmap.mmap) -> None:
        self._region = region

    def close(self) -> None:
        """Unmap the plant pinned to one CPU whose anonymous-page share is folded before and after."""

        if not hasattr(os, "sched_setaffinity") or _lag_pages() is None:
            self._region.close()
            return
        home = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(home)})
        try:
            before: list[mmap.mmap] = []
            _fold_this_cpu(before)
            self._region.close()
            for region in [*_HELD, *before]:
                region.close()
            _HELD.clear()
            _fold_this_cpu(_HELD)
        finally:
            os.sched_setaffinity(0, home)


def touch_fresh_plant() -> FreshPlant:
    """Map 64 MiB of private anonymous memory and write one byte on every page.

    Returns
    -------
    FreshPlant
        The resident region; ``close()`` unmaps it.
    """

    region = mmap.mmap(-1, PLANT_BYTES, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    np.frombuffer(region, dtype=np.uint8)[:: mmap.PAGESIZE] = 1
    return FreshPlant(region)
