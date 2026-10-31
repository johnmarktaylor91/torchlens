"""Host peak-memory instrumentation helpers (F20, brainpipe memo D-7).

Split out of ``capture/trace.py`` under the R43 file-size ratchet: the
process RSS readers, the per-capture resident high-water scoping, and the
``Trace.cost.forward_peak_memory_pair`` reader live here; the forward bracket in
``capture/trace.py`` stays the only writer (through :func:`record_forward_peak`).

CPU peaks are opt-in: measuring one resets the
process-wide resident high-water mark, which also resets the caller's own
``VmHWM`` and ``ru_maxrss`` readings and cannot be undone. Only a capture with
``CaptureOptions(measure_host_peak_memory=True)`` calls :func:`reset_peak_rss`
(the forward bracket, and each CPU backward pass); by default nothing is written,
and the CPU forward and backward peaks read ``None`` with
:data:`CPU_PEAK_NOT_REQUESTED_REASON`. CUDA and the MPS allocator delta never
touch the host mark and are unchanged.
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
    brackets. Called only under ``CaptureOptions(measure_host_peak_memory=True)``.

    Returns
    -------
    bool
        Whether the reset succeeded (Linux with a writable procfs).
    """

    try:
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def peak_rss_bytes() -> int:
    """Return the process resident high-water mark in bytes, or 0.

    Reads ``VmHWM`` from ``/proc/self/status`` on Linux, corrected by
    :func:`_kernel_mark_lag_bytes` when the mark is above the current RSS;
    falls back to ``ru_maxrss`` elsewhere, which macOS reports in bytes and
    Linux and the BSDs in kibibytes. Both are process-lifetime maxima unless
    :func:`reset_peak_rss` succeeded beforehand.

    Returns
    -------
    int
        Peak resident-set size in bytes, or 0 when unreadable.
    """

    try:
        status = _proc_status_kib(("VmHWM", "VmRSS"))
    except (OSError, ValueError, IndexError):
        status = {}
    hwm_kib = status.get("VmHWM")
    if hwm_kib is not None:
        rss_kib = status.get("VmRSS", hwm_kib)
        lag = 0
        if hwm_kib > rss_kib:
            lag = _kernel_mark_lag_bytes(rss_kib * 1024, _approximate_rss_bytes())
        return hwm_kib * 1024 + lag
    try:
        import resource
        import sys

        scale = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
    except (ImportError, OSError, ValueError, AttributeError):
        return 0


def _proc_status_kib(keys: tuple[str, ...]) -> dict[str, int]:
    """Return the named ``/proc/self/status`` fields, in kibibytes, from one read."""

    found: dict[str, int] = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            key = line.split(":", 1)[0]
            if key in keys:
                found[key] = int(line.split()[1])
    return found


def _kernel_mark_lag_bytes(exact_rss: int, approximate_rss: int | None) -> int:
    """Return how far the kernel's high-water mark lags the exact RSS, in bytes.

    Linux keeps the RSS counters per CPU. Where ``/proc`` reports the exact
    sum (``VmRSS``, ``statm``; measured on Linux 7.0), the kernel still sets
    ``VmHWM``'s mark from the approximate total (``/proc/self/stat`` field
    24), which leaves out each CPU's unfolded residue: up to about 63 pages
    per CPU, 8 MiB measured on a 32-CPU host. A mark set when a freed buffer
    was unmapped therefore reads below the RSS ``/proc`` showed while the
    buffer was resident. The lag read now stands in for the lag at that
    unmap. On kernels whose ``/proc`` RSS is the approximate total the two
    readings agree and the lag is 0.

    Parameters
    ----------
    exact_rss:
        Current RSS from ``/proc/self/status`` (``VmRSS``), in bytes.
    approximate_rss:
        Current RSS from ``/proc/self/stat``, in bytes, or ``None`` when unreadable.

    Returns
    -------
    int
        The lag in bytes, never negative; 0 when the approximate RSS is unreadable.
    """

    if approximate_rss is None:
        return 0
    return max(0, exact_rss - approximate_rss)


def _approximate_rss_bytes() -> int | None:
    """Return the kernel's approximate RSS (``/proc/self/stat`` field 24) in bytes, or ``None``."""

    try:
        import os

        with open("/proc/self/stat") as handle:
            text = handle.read()
        # Fields after the parenthesised command name start at field 3.
        pages = int(text[text.rindex(")") + 2 :].split()[21])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


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
    when measured but the reset was unavailable, ``"not_requested"`` (with
    ``resident`` ``None``) when the capture did not opt in with
    ``measure_host_peak_memory=True``, so the mark was neither reset nor read,
    ``"unavailable"`` when
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

#: The reason for a CPU forward or backward peak of ``None`` on a capture that
#: did not opt in: measuring resets the process-wide high-water mark.
CPU_PEAK_NOT_REQUESTED_REASON = (
    "not measured: a CPU peak resets the process-wide resident high-water mark "
    "(VmHWM, ru_maxrss); pass CaptureOptions(measure_host_peak_memory=True) to measure it"
)


def resident_growth_and_basis(
    rss_before: int, scoped: bool, requested: bool
) -> tuple[int | None, str]:
    """Return the forward pair's ``resident`` growth and its ``resident_basis``.

    Parameters
    ----------
    rss_before:
        The bracket's baseline RSS in bytes (0 when unreadable).
    scoped:
        Whether :func:`reset_peak_rss` scoped the mark to this bracket.
    requested:
        Whether the capture opted in with ``measure_host_peak_memory=True``.

    Returns
    -------
    tuple[int | None, str]
        ``(None, "not_requested")`` without the opt-in (the mark is not read);
        otherwise the ``VmHWM`` growth over the baseline with ``"per_capture"``
        or ``"process_lifetime"``, or ``(None, "unavailable")`` when unreadable.
    """

    if not requested:
        return None, "not_requested"
    growth: int | None = None
    try:
        peak_rss_after = peak_rss_bytes()
        if peak_rss_after > 0 and rss_before > 0:
            growth = max(0, peak_rss_after - rss_before)
    except Exception:  # noqa: BLE001 - measurement never raises into capture
        growth = None
    if growth is None:
        return None, "unavailable"
    return growth, "per_capture" if scoped else "process_lifetime"


def forward_peak_value(
    backend: str, measured: int, *, python_peak_measured: bool, host_requested: bool = True
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
    host_requested:
        Whether the capture opted in with ``measure_host_peak_memory=True``.

    Returns
    -------
    tuple[int | None, str | None]
        ``(measured, None)`` for every positive value and every non-CPU
        backend (MPS and CUDA semantics are unchanged); on CPU without the
        opt-in ``(None, CPU_PEAK_NOT_REQUESTED_REASON)``; ``(None, reason)``
        for a measured CPU zero, the reason naming ``measure_python_peak_memory=True``.
    """

    if backend == "cpu" and not host_requested:
        return None, CPU_PEAK_NOT_REQUESTED_REASON
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


def record_forward_peak(
    trace: Any, backend: str, measured: int, python_probe: Any, host_requested: bool = True
) -> None:
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
    host_requested:
        Whether the capture opted in with ``measure_host_peak_memory=True``.
    """

    from .._quantities import Bytes

    peak, trace._cost_forward_peak_memory_reason = forward_peak_value(
        backend,
        measured,
        python_peak_measured=python_probe is not None,
        host_requested=host_requested,
    )
    trace._cost_forward_peak_memory = None if peak is None else Bytes(peak)


def record_backward_peak(trace: Any, backend: str, peak_delta: int) -> None:
    """Fold one backward pass's peak into ``trace._cost_backward_peak_memory``.

    CPU passes aggregate by max (the largest peak above a pass's starting
    baseline) and only under ``measure_host_peak_memory=True``; without the
    opt-in the CPU peak is ``None`` with :data:`CPU_PEAK_NOT_REQUESTED_REASON`.
    CUDA sums its per-pass high-water excesses (their total is the growth of the
    device mark); MPS keeps its allocator-delta sum.

    Parameters
    ----------
    trace:
        The trace whose backward pass just finished.
    backend:
        The pass's measurement backend (``"cpu"``, ``"cuda"`` or ``"mps"``).
    peak_delta:
        The pass's peak above its starting baseline, in bytes.
    """

    from .._quantities import Bytes

    if backend == "cpu" and not getattr(trace, "measure_host_peak_memory", False):
        trace._cost_backward_peak_memory = None
        trace.backward_peak_memory_reason = CPU_PEAK_NOT_REQUESTED_REASON
        return
    prior_value = getattr(trace, "_cost_backward_peak_memory", None)
    prior = int(prior_value) if prior_value is not None else 0
    trace._cost_backward_peak_memory = Bytes(
        max(prior, peak_delta) if backend == "cpu" else prior + peak_delta
    )
    trace.backward_peak_memory_reason = None
