"""Human-readable formatting, print overrides, and environment detection."""

import multiprocessing as mp
from typing import Any, List


def identity(x: Any) -> Any:
    """Return the input unchanged."""
    return x


def int_list_to_compact_str(int_list: List[int]) -> str:
    """Given a list of integers, returns a compact string representation of the list, where
    contiguous stretches of the integers are represented as ranges (e.g., [1 2 3 4] becomes "1-4"),
    and all such ranges are separated by commas.

    Args:
        int_list: List of integers.

    Returns:
        Compact string representation of the list.
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
            end = int_list[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = int_list[i]
            end = int_list[i]
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ",".join(ranges)


def human_readable_size(size: float, decimal_places: int = 1) -> str:
    """Utility function to convert a size in bytes to a human-readable format.

    Args:
        size: Number of bytes.
        decimal_places: Number of decimal places to use.

    Returns:
        String with human-readable size.
    """
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0 or unit == "PB":
            break
        size /= 1024.0
    if unit == "B":
        size = int(size)
    else:
        size = round(size, decimal_places)
    return f"{size} {unit}"


def in_notebook() -> bool:
    """Return True if the code is running inside a Jupyter notebook, False otherwise."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except (ImportError, AttributeError):
        return False
    return True


def warn_parallel():
    """
    Utility function to give raise error if it's being run in parallel processing.
    """
    if mp.current_process().name != "MainProcess":
        raise RuntimeError(
            "WARNING: It looks like you are using parallel execution; only run "
            "torchlens in the main process, since certain operations "
            "depend on execution order."
        )
