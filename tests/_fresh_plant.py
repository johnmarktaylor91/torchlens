"""A 64 MiB host plant that is a fresh anonymous mapping by construction.

``np.ones(8 * 2**20)`` is a fresh mapping only while glibc's heap holds no free
chunk that large. Earlier tests in the same process can leave one: after a
freed block raises the dynamic mmap threshold, blocks below it come from the
heap, and when they are freed they coalesce into a resident free chunk that
malloc serves a 64 MiB request from before it would map anything, so writing
the "plant" touches no new page and the resident set does not grow at all.
A private anonymous ``mmap`` with every page written always grows the
resident set by its full size, whatever the allocator's history.
"""

from __future__ import annotations

import mmap

import numpy as np

PLANT_BYTES = 64 * 2**20


def touch_fresh_plant() -> mmap.mmap:
    """Map 64 MiB of private anonymous memory and write one byte on every page.

    Returns
    -------
    mmap.mmap
        The resident region; ``close()`` unmaps it.
    """

    region = mmap.mmap(-1, PLANT_BYTES, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    np.frombuffer(region, dtype=np.uint8)[:: mmap.PAGESIZE] = 1
    return region
