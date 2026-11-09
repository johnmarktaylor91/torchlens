"""Torch's op spelling list (see :mod:`torchlens.backends._op_spelling`).

Torch's normalized spelling (``_normalize_func_name``) is the canonical one,
so every torch name is its own canonical spelling through :func:`spelling_key`
and only torch's aliases are listed: two torch spellings of one operation
(``x / y`` logs ``__truediv__``, ``torch.div`` logs ``div``) share one
canonical spelling. Used only for candidate preference and drawing labels;
never for pairing.
"""

from __future__ import annotations

from ..._capture.arg_positions import _normalize_func_name

#: Torch alias (normalized) to the canonical spelling.
SPELLINGS: dict[str, str] = {
    "truediv": "div",
    "truedivide": "div",
    "divide": "div",
    "itruediv": "div",
    "multiply": "mul",
    "imul": "mul",
    "subtract": "sub",
    "isub": "sub",
    "iadd": "add",
    "negative": "neg",
    "absolute": "abs",
    "floordiv": "floordivide",
    "mod": "remainder",
    "clip": "clamp",
    "fix": "trunc",
    "concatenate": "cat",
    "concat": "cat",
    "swapaxes": "transpose",
    "swapdims": "transpose",
    "moveaxis": "movedim",
    "greater": "gt",
    "greaterequal": "ge",
    "less": "lt",
    "lessequal": "le",
    "notequal": "ne",
}

#: Torch names with no canonical spelling beyond their own (none: torch's
#: unlisted names fall back to their normalized name with family ``other``).
UNMAPPED: dict[str, str] = {}


def spelling_key(func_name: str) -> str:
    """Return the normalized torch spelling the list is keyed by."""

    return _normalize_func_name(func_name)
