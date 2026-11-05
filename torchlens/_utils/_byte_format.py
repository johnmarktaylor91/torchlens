"""Human-readable byte counts for budget and capture messages."""

from __future__ import annotations

_BYTES_UNITS = (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1))


def format_bytes(num_bytes: float) -> str:
    """Render a byte count in the largest unit that keeps it readable.

    Parameters
    ----------
    num_bytes:
        Byte count.

    Returns
    -------
    str
        Human-readable size such as ``"1.44 GB"``.
    """

    for unit, scale in _BYTES_UNITS:
        if abs(num_bytes) >= scale or unit == "B":
            if unit == "B":
                return f"{int(num_bytes)} B"
            return f"{num_bytes / scale:.2f} {unit}"
    return f"{int(num_bytes)} B"
