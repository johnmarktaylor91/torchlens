"""Device and host memory readings that bracket a captured backward pass."""

from __future__ import annotations

import torch

from ..._capture.peak_memory import peak_rss_bytes, process_rss_bytes, reset_peak_rss


def _memory_snapshot(device: torch.device) -> tuple[str, int]:
    """Return backend name and current allocated memory for a device.

    Parameters
    ----------
    device:
        Device associated with the backward loss tensor.

    Returns
    -------
    tuple[str, int]
        Backend label and memory snapshot in bytes.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        return "cuda", int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps"):
        return "mps", int(torch.mps.current_allocated_memory())
    return "cpu", max(process_rss_bytes(), peak_rss_bytes())  # VmHWM: a freed buffer counts


def _peak_memory_baseline(device: torch.device, measure_host: bool) -> tuple[str, int]:
    """Return the pre-pass memory baseline WITHOUT resetting peak tracking.

    this used to call ``torch.cuda.reset_peak_memory_stats`` on every
    ``backward``, clobbering the caller's process-wide high-water counter.
    The baseline is now the pre-existing ``max_memory_allocated`` snapshot:
    a backward that pushes a new device peak reports the exact excess over
    the prior high-water mark, and one that stays under it honestly reads
    ``0``. Under ``CaptureOptions(measure_host_peak_memory=True)`` CPU first
    resets the process ``VmHWM`` to the current RSS (Linux; see
    ``capture.peak_memory.reset_peak_rss`` for the process-wide effect), so its
    baseline is the pass-start RSS and its end reading the pass peak. Without
    the opt-in nothing is reset and the CPU peak is recorded as ``None``.

    Parameters
    ----------
    device:
        Device associated with the backward loss tensor.
    measure_host:
        Whether the trace opted in to the CPU peak (and so to the reset).

    Returns
    -------
    tuple[str, int]
        Backend label and starting memory snapshot in bytes.
    """
    if measure_host and device.type not in {"cuda", "mps"}:
        reset_peak_rss()
    return _memory_snapshot(device)
