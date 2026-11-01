"""Host and device memory headroom probes for the save budget.

Split from ``_save_budget`` so the budget module stays within its size row;
both probes answer ``None`` when a surface cannot be measured, never a guess.
"""

from __future__ import annotations

import os

import torch


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
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
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
    try:
        import psutil
    except ImportError:
        return None
    try:
        available = int(psutil.virtual_memory().available)
    except Exception:
        return None
    return available if available > 0 else None


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
            # Pin the query to THIS device explicitly (R36-7b): mem_get_info
            # resolves an index-less device to the CURRENT device, so a
            # capture saving to cuda:1 while cuda:0 is current read the wrong
            # card's headroom.
            with torch.cuda.device(device):
                free_bytes, _total = torch.cuda.mem_get_info(device)
            return int(free_bytes)
        except Exception:
            return None
    if device_type == "meta":
        # Meta tensors have no storage, so nothing is ever really committed.
        return None
    return None
