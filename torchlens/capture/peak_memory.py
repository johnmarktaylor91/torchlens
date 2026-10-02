"""Host peak-memory instrumentation helpers (F20, brainpipe memo D-7).

Split out of ``capture/trace.py`` under the R43 file-size ratchet: the
process RSS readers, the per-capture resident high-water scoping, and the
``Trace.forward_peak_memory_pair`` reader live here; the forward bracket in
``capture/trace.py`` stays the only writer.
"""

from __future__ import annotations

from typing import Any


def psutil_available() -> bool:
    """Whether ``psutil`` can be imported (capability probe, not torch-specific).

    ``process_rss_bytes`` is the host-resident-set baseline the CPU/MPS forward-peak
    bracket reads BEFORE the forward pass; without psutil that baseline is always 0,
    so the bracket can never compute a resident delta no matter how the VmHWM
    high-water mark itself behaves. Callers (the bracket's ``resident_basis``
    disclosure, this module's test suite) use this probe to distinguish "psutil
    missing" from "measured, zero growth" instead of reporting a misleading basis
    label alongside an unexplained ``None``.

    Returns
    -------
    bool
        ``True`` when ``import psutil`` succeeds.
    """

    try:
        import psutil  # noqa: F401
    except ImportError:
        return False
    return True


def process_rss_bytes() -> int:
    """Return the current process resident-set size in bytes, or 0 if unavailable.

    Used as a coarse host-memory proxy for the CPU forward-pass peak. psutil is an
    optional dependency; absence degrades to 0 rather than raising.

    Returns
    -------
    int
        Resident-set size in bytes, or 0 when psutil is unavailable.
    """

    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


def reset_peak_rss() -> bool:
    """Best-effort reset of the process resident high-water mark (Linux).

    Writing ``5`` to ``/proc/self/clear_refs`` resets ``VmHWM`` to the
    current RSS, making a per-capture resident PEAK measurable instead of
    the process-lifetime maximum (which reads 0 for every capture after the
    first -- the brainpipe memo's sweep-scale instrument defect). Only the
    peak-RSS counter is touched; no page state relevant to correctness
    changes.

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
