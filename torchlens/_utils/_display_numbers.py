"""Number and unit spellings shared by every printout (house style v1).

One spelling per fact: significant-digit floats (the precision law), exact
and compact counts, IEC bytes and durations. Pure formatters over Python
numbers; nothing here touches a tensor or a record.
"""

from __future__ import annotations

import math

#: Binary byte units, smallest first (IEC spelling).
_IEC_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")
#: Duration units: (name, seconds per unit), largest first.
_DURATION_UNITS = (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9))


def format_sig(value: float, digits: int = 4) -> str:
    """Format a float with ``digits`` significant digits, trailing zeros kept.

    Scientific notation outside ``[1e-4, 1e4)``; NaN and infinities print as
    Python spells them. ``0.99997`` prints ``1.000``, never ``1``, so columns
    of numbers align and the digit count states the precision.
    """

    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    if value == 0:
        return "0.000"[: digits + 1] if digits > 1 else "0"
    magnitude = abs(value)
    if magnitude < 1e-4 or magnitude >= 1e4:
        text = f"{value:.{max(digits - 1, 0)}e}"
        mantissa, exponent = text.split("e")
        return f"{mantissa}e{int(exponent):+03d}"
    decimals = digits - 1 - int(math.floor(math.log10(magnitude)))
    decimals = max(decimals, 0)
    return f"{value:.{decimals}f}"


def format_count(count: int) -> str:
    """Compact counts with one decimal and unit rollover: ``15.4k``, ``124.4M``."""

    magnitude = abs(count)
    if magnitude < 1000:
        return str(count)
    for suffix, divisor in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if magnitude >= divisor * 0.99995:
            return f"{count / divisor:.1f}{suffix}"
    return str(count)


def format_exact_count(count: int) -> str:
    """Exact count with thousands separators, for cards: ``124,439,808``."""

    return f"{int(count):,}"


def format_iec_bytes(nbytes: int) -> str:
    """IEC bytes, three significant digits kept: ``60.0 KiB``, ``15.0 MiB``."""

    value = float(nbytes)
    for unit in _IEC_UNITS:
        rounded = float(f"{value:.3g}")
        if abs(rounded) < 1000 or unit == _IEC_UNITS[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{format_sig(rounded, 3)} {unit}"
        value /= 1024
    return f"{int(nbytes)} B"


def format_duration(seconds: float) -> str:
    """Duration with 3 significant digits and a spaced unit: ``151 us``, ``4.02 ms``."""

    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return f"{seconds} s"
    if seconds == 0:
        return "0 s"
    magnitude = abs(seconds)
    for unit, scale in _DURATION_UNITS:
        # One adjacent float keeps unit conversion from missing an exact halfway value.
        if magnitude >= math.nextafter(scale * 0.9995, 0.0) or unit == "ns":
            scaled = math.nextafter(seconds / scale, math.copysign(math.inf, seconds))
            rounded = float(f"{scaled:.3g}")
            return f"{format_sig(rounded, 3)} {unit}"
    return f"{seconds:.3g} s"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """``1 call``, ``2 calls``; ``plural_form`` for irregular nouns (``passes``)."""

    noun = singular if count == 1 else (plural_form or f"{singular}s")
    return f"{count} {noun}"
