"""Host peak-memory instrumentation helpers (F20, brainpipe memo D-7).

Split out of ``capture/trace.py`` under the R43 file-size ratchet: the
process RSS readers, the per-capture resident high-water scoping, and the
``Trace.forward_peak_memory_pair`` reader live here; the forward bracket in
``capture/trace.py`` stays the only writer (through :func:`record_forward_peak`).
"""

from __future__ import annotations

from typing import Any


def process_rss_bytes() -> int:
    """Return the current process resident-set size in bytes, or 0 if unavailable.

    The baseline of the CPU forward and backward peak brackets. psutil is an
    optional dependency; without it Linux reads ``/proc/self/statm`` (resident
    pages times the page size), and other platforms degrade to 0 rather than
    raising.

    Returns
    -------
    int
        Resident-set size in bytes, or 0 when neither psutil nor procfs is usable.
    """

    try:
        import psutil
    except ImportError:
        try:
            import os

            with open("/proc/self/statm") as handle:
                return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return 0
    return int(psutil.Process().memory_info().rss)


#: The one switch for the process-wide ``VmHWM`` reset: every reset (the forward
#: bracket and each CPU backward pass) goes through :func:`reset_peak_rss`.
RESET_HOST_PEAK = True


def reset_peak_rss() -> bool:
    """Best-effort reset of the process resident high-water mark (Linux).

    Writing ``5`` to ``/proc/self/clear_refs`` resets ``VmHWM`` to the
    current RSS, making a per-capture resident PEAK measurable instead of
    the process-lifetime maximum (which reads 0 for every capture after the
    first -- the brainpipe memo's sweep-scale instrument defect). Only the
    peak-RSS counter is touched; no page state relevant to correctness
    changes. The reset is process-wide and cannot be undone: after a capture
    (forward bracket) or a CPU ``log_backward`` (each pass), a caller's own
    ``VmHWM`` and ``ru_maxrss`` reads report the high-water since that bracket
    began, not since process start. Nothing in TorchLens reads it across
    brackets. With :data:`RESET_HOST_PEAK` off nothing is written and the
    peaks fall back to the unscoped (process-lifetime) basis.

    Returns
    -------
    bool
        Whether the reset succeeded (Linux with a writable procfs).
    """

    if not RESET_HOST_PEAK:
        return False
    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def peak_rss_bytes() -> int:
    """Return the process resident high-water mark in bytes, or 0.

    Reads ``VmHWM`` from ``/proc/self/status`` on Linux; falls back to
    ``ru_maxrss`` elsewhere. Both are process-lifetime maxima unless
    :func:`reset_peak_rss` succeeded beforehand.

    Returns
    -------
    int
        Peak resident-set size in bytes, or 0 when unreadable.
    """

    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, OSError, ValueError, AttributeError):
        return 0


def read_peak_pair(trace: Any) -> dict[str, Any] | None:
    """Return a trace's (live, resident) forward peak pair, when measured.

    F20 brainpipe D-7 (spelling DOCUMENTED-UNSTABLE): live allocation and
    resident high-water are different physical quantities that can differ
    by more than an order of magnitude on shipped modes, so the measured
    peak is a PAIR with its backend named, never one number. Keys:
    ``live`` (bytes or ``None`` when unmeasured -- on CPU it exists only
    under the ``measure_python_peak_memory`` opt-in), ``resident`` (bytes
    or ``None``), ``backend`` (``"cuda:allocated+reserved"`` /
    ``"cpu:maxlive+rss"`` / ``"mps:allocated+rss"``), and
    ``resident_basis`` (``"per_capture"`` when the host high-water mark was
    reset for this bracket and a baseline was measured, ``"process_lifetime"``
    when measured but the reset was unavailable, ``"unavailable"`` when
    ``resident`` could not be computed at all (e.g. psutil is not installed,
    so the pre-forward RSS baseline reads 0 regardless of the reset outcome
    -- a typed disclosure, never a bare unexplained ``None``), or
    ``"prior_high_water_delta"`` on CUDA). Session-time only: ``None`` on
    loaded artifacts, which never re-measure.

    Parameters
    ----------
    trace:
        Trace whose forward bracket may have written the pair.

    Returns
    -------
    dict[str, Any] | None
        Read-only copy of the measured pair, or ``None``.
    """

    try:
        pair = trace._forward_peak_memory_pair
    except AttributeError:
        pair = None
    return dict(pair) if pair is not None else None


#: The recorded reason for a CPU forward peak of ``None``: the host
#: RSS delta read 0, which says only that the forward fit in already-resident
#: heap headroom, never that it allocated nothing.
CPU_ZERO_PEAK_REASON = (
    "not measured: the CPU RSS delta read 0; pass "
    "CaptureOptions(measure_python_peak_memory=True) for the Python-allocation peak"
)

#: The reason when the opt-in tracemalloc peak ALSO read 0.
CPU_ZERO_PEAK_WITH_PYTHON_REASON = (
    "not measured: the CPU RSS delta and the Python-allocation peak "
    "(measure_python_peak_memory=True) both read 0"
)


def forward_peak_value(
    backend: str, measured: int, *, python_peak_measured: bool
) -> tuple[int | None, str | None]:
    """Return the forward peak to record and its reason.

    Parameters
    ----------
    backend:
        The measurement backend (``"cpu"``, ``"mps"``, ...).
    measured:
        The measured host peak in bytes (:func:`host_forward_peak`).
    python_peak_measured:
        Whether the opt-in tracemalloc probe ran.

    Returns
    -------
    tuple[int | None, str | None]
        ``(measured, None)`` for every positive value and every non-CPU
        backend (MPS and CUDA semantics are unchanged); ``(None, reason)`` for
        a CPU zero, the reason naming ``measure_python_peak_memory=True``.
    """

    if backend != "cpu" or measured > 0:
        return measured, None
    if python_peak_measured:
        return None, CPU_ZERO_PEAK_WITH_PYTHON_REASON
    return None, CPU_ZERO_PEAK_REASON


def host_forward_peak(
    backend: str, rss_delta: int, traced_peak: int, resident_growth: int | None, basis: str
) -> int:
    """Return the host forward peak in bytes before the CPU-zero rule.

    CPU uses the largest of the end-of-forward RSS delta, the opt-in
    tracemalloc peak and, when the ``VmHWM`` reset scoped it to this
    bracket (``basis == "per_capture"``), the resident high-water growth: a
    buffer the forward allocates and frees before it returns is part of its
    peak. MPS keeps the allocator delta (the resident pair is host memory).

    Parameters
    ----------
    backend:
        ``"cpu"`` or ``"mps"``.
    rss_delta:
        End-of-forward RSS (or MPS allocator) minus the bracket baseline.
    traced_peak:
        Opt-in tracemalloc peak over its baseline, else 0.
    resident_growth:
        ``VmHWM`` after the forward minus the baseline RSS, when computable.
    basis:
        The pair's ``resident_basis``.

    Returns
    -------
    int
        The measured peak in bytes.
    """

    scoped = resident_growth if backend == "cpu" and basis == "per_capture" else None
    return max(rss_delta, traced_peak, scoped or 0)


def record_forward_peak(trace: Any, backend: str, measured: int, python_probe: Any) -> None:
    """Write ``forward_peak_memory`` and its reason onto ``trace``.

    Parameters
    ----------
    trace:
        The trace the forward bracket measured.
    backend:
        The measurement backend label.
    measured:
        The measured host peak in bytes (:func:`host_forward_peak`).
    python_probe:
        The tracemalloc module when the opt-in probe ran, else ``None``.
    """

    from ..quantities import Bytes

    peak, trace.forward_peak_memory_reason = forward_peak_value(
        backend, measured, python_peak_measured=python_probe is not None
    )
    trace.forward_peak_memory = None if peak is None else Bytes(peak)
