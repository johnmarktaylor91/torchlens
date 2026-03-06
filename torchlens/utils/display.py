"""Human-readable formatting, print overrides, and environment detection.

Formatting helpers used by the display/print paths of ModelLog and LayerLog,
plus environment checks (Jupyter detection, parallel-processing guard).
"""

import multiprocessing as mp
from typing import Any, List


def identity(x: Any) -> Any:
    """Return the input unchanged.

    Used as a no-op placeholder where a callable is expected (e.g.
    ``activation_postfunc`` when no user postprocessing is desired).
    """
    return x


def int_list_to_compact_str(int_list: List[int]) -> str:
    """Collapse a list of integers into a compact range string.

    Contiguous runs are collapsed into ``"start-end"`` ranges, separated
    by commas.  Example: ``[1, 2, 3, 7, 8, 10]`` becomes ``"1-3,7-8,10"``.

    Args:
        int_list: List of integers (need not be sorted).

    Returns:
        Compact string representation.
    """
    int_list = sorted(int_list)
    if len(int_list) == 0:
        return ""
    if len(int_list) == 1:
        return str(int_list[0])
    ranges = []
    start = int_list[0]
    end = int_list[0]
    for i in range(1, len(int_list)):
        if int_list[i] == end + 1:
            # Extend the current contiguous run.
            end = int_list[i]
        else:
            # Current run ended — flush it and start a new one.
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = int_list[i]
            end = int_list[i]
    # Flush the final run.
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def human_readable_size(size: float, decimal_places: int = 1) -> str:
    """Convert a byte count into a human-readable string (e.g. ``"1.5 MB"``).

    Args:
        size: Number of bytes.
        decimal_places: Number of decimal places for non-byte units.

    Returns:
        String with human-readable size and unit suffix.
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0 or unit == "PB":
            break
        size /= 1024.0
    if unit == "B":
        size = int(size)  # No fractional bytes.
    else:
        size = round(size, decimal_places)
    return f"{size} {unit}"


def in_notebook() -> bool:
    """Return True if running inside a Jupyter notebook kernel.

    Checks for the IPython kernel app in the running IPython instance's
    config.  Returns False in plain Python, IPython terminal, or when
    IPython is not installed.
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def warn_parallel():
    """Raise ``RuntimeError`` if called from a child process.

    TorchLens is single-threaded by design — its global toggle state and
    ordered tensor counter are not safe for concurrent access.  This guard
    is called early in ``log_forward_pass`` to fail fast rather than
    produce silently corrupted logs.
    """
    if mp.current_process().name != "MainProcess":
        raise RuntimeError(
            "WARNING: It looks like you are using parallel execution; only run "
            "torchlens in the main process, since certain operations "
            "depend on execution order."
        )
