"""Host and device memory headroom probes for the save budget.

Split from ``_save_budget`` so the budget module stays within its size row;
both probes answer ``None`` when a surface cannot be measured, never a guess.
"""

from __future__ import annotations

import os

import torch

from ._optional_psutil import optional_psutil
from ._utils._mps_memory import mps_headroom_bytes


def _available_host_bytes() -> int | None:
    """Return available host RAM in bytes, or ``None`` when unmeasurable.

    Returns
    -------
    int | None
        Available bytes. Prefers ``MemAvailable`` from ``/proc/meminfo`` because
        it accounts for reclaimable cache, then the POSIX available-pages count,
        then ``psutil`` when installed. macOS has neither procfs nor
        ``SC_AVPHYS_PAGES``, so without the ``psutil`` fallback every default
        ``save_budget="auto"`` capture there warned that CPU budgeting was off.
    """

    try:
        # One unbuffered binary read: a text-mode line loop built a decoder and a str per
        # line on every capture (the ledger asks once per capture, CPU device).
        with open("/proc/meminfo", "rb", buffering=0) as meminfo:
            text = meminfo.read()
        for line in text.split(b"\n"):
            if line.startswith(b"MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except Exception:
        pass
    psutil = optional_psutil()
    if psutil is None:
        return None
    try:
        available = int(psutil.virtual_memory().available)
    except Exception:
        return None
    return available if available > 0 else None


def _available_mps_bytes() -> int | None:
    """Return the MPS headroom in bytes, or ``None`` when torch cannot report it.

    Returns
    -------
    int | None
        ``recommended_max_memory() - driver_allocated_memory()`` (torch >= 2.5,
        gated by ``HAS_MPS_MEMORY_QUERY``), clamped at zero and bounded by the
        host's available RAM because Apple silicon shares one unified memory.
        ``None`` on a torch without the queries or when a query raises.
    """

    headroom = mps_headroom_bytes()
    if headroom is None:
        return None
    host = _available_host_bytes()
    return headroom if host is None else min(headroom, host)


def available_device_bytes(device: torch.device) -> int | None:
    """Return the memory headroom for one device, or ``None`` when unmeasurable.

    Parameters
    ----------
    device:
        Device whose free memory is queried.

    Returns
    -------
    int | None
        Free bytes, or ``None`` when this device type exposes no headroom query.
        ``None`` means *unbudgeted and reported as such*, never "assume
        infinite silently".
    """

    device_type = device.type
    if device_type == "cpu":
        return _available_host_bytes()
    if device_type == "cuda":
        try:
            # Pin the query to THIS device explicitly: mem_get_info
            # resolves an index-less device to the CURRENT device, so a
            # capture saving to cuda:1 while cuda:0 is current read the wrong
            # card's headroom.
            with torch.cuda.device(device):
                free_bytes, _total = torch.cuda.mem_get_info(device)
            return int(free_bytes)
        except Exception:
            return None
    if device_type == "mps":
        return _available_mps_bytes()
    if device_type == "meta":
        # Meta tensors have no storage, so nothing is ever really committed.
        return None
    return None
