"""The host allocation budget the runnable preflight refuses against.

The budget is the smaller of two upper bounds on what one new host allocation
could get: available memory plus free swap, and the process's soft
address-space limit (``RLIMIT_AS``) when one is set. Both over-estimate, so
the preflight still refuses only requests that could never have succeeded.
The limit matters on hosts with far more memory than the process may map: on
a node reporting terabytes available, a request just under that fits the
memory bound but can never be mapped under a limit of a few gigabytes.
"""

from __future__ import annotations

import sys


def _probe_host_memory_budget_bytes() -> int | None:
    """Probe the host allocation budget without memoization.

    Returns
    -------
    int | None
        The smaller of available memory plus free swap and the soft
        address-space limit, either one alone when the other is unknown, or
        ``None`` when neither can be probed.
    """

    available = _available_memory_bytes()
    limit = _address_space_limit_bytes()
    if available is None or limit is None:
        return limit if available is None else available
    return min(available, limit)


def _address_space_limit_bytes() -> int | None:
    """Return the soft ``RLIMIT_AS`` in bytes, or ``None`` when unlimited or unreadable.

    The whole limit, not the headroom above the current mapped size: an
    allocation can reuse freed space inside the existing mappings, so only the
    limit itself is a bound no single allocation can exceed.
    """

    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    if soft == resource.RLIM_INFINITY or soft < 0:
        return None
    return int(soft)


def _available_memory_bytes() -> int | None:
    """Probe available host memory plus free swap without memoization."""

    if sys.platform.startswith("linux"):
        try:
            proc_fields: dict[str, int] = {}
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    name, separator, rest = line.partition(":")
                    if name not in {"MemAvailable", "SwapFree"}:
                        continue
                    parts = rest.split()
                    if separator != ":" or len(parts) != 2 or not parts[0].isdigit():
                        break
                    if parts[1] != "kB":
                        break
                    proc_fields[name] = int(parts[0]) * 1024
                    if len(proc_fields) == 2:
                        return proc_fields["MemAvailable"] + proc_fields["SwapFree"]
        except (OSError, UnicodeError):  # pragma: no cover - missing/malformed Linux procfs
            pass

    try:
        import psutil
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available) + int(psutil.swap_memory().free)
        except Exception:  # pragma: no cover - defensive probe fallback
            pass
    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    fields[name.strip()] = int(parts[0]) * 1024
        if "MemAvailable" in fields:
            return fields["MemAvailable"] + fields.get("SwapFree", 0)
    except OSError:  # pragma: no cover - non-Linux hosts without /proc
        pass
    return None
