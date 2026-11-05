"""Precision-law scalar formatting shared by the basis display helpers and the stats renderers."""

from __future__ import annotations

import math

__all__ = ["format_sig"]


def format_sig(value: float, digits: int = 4) -> str:
    """Precision-law scalar formatter (D7).

    4 significant digits with trailing zeros KEPT (``%.4g`` strips them, so
    ``0.99997`` printed ``1`` and no column aligned -- the live bug this
    replaces); scientific notation outside ``[1e-4, 1e4)``; integral floats
    keep the digit budget rather than gaining a spurious wide tail.
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
