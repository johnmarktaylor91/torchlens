"""Display keys for ops and layers (house style v1, rule A5).

Records print by the key a user types: the stored ``short_layer_label``
(``tanh_6``) for a single-pass record and ``short_layer_label:pass``
(``tanh_1:2``) when the layer ran more than once. The positional suffix of
the full label (``tanh_6_270``) never prints. Keys come from stored fields,
never from parsing a label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .fail_open import fail_open

#: Most candidates an error message offers.
CANDIDATE_LIMIT = 5


def display_key(record: Any) -> str:
    """The key a record prints as; falls back to its stored label.

    Reads ``short_layer_label``, ``num_passes`` and ``pass_number`` from the
    record's stored fields. A multi-pass Layer
    (no single pass) prints the bare short layer label.
    """

    short = _stored(record, "short_layer_label")
    if not isinstance(short, str) or not short:
        label = (
            _stored(record, "layer_label")
            or _stored(record, "label")
            or _stored(record, "label_raw")
        )
        return str(label) if label is not None else "?"
    num_passes = _stored(record, "num_passes")
    pass_number = _stored(record, "pass_number")
    if isinstance(num_passes, int) and num_passes > 1 and isinstance(pass_number, int):
        return f"{short}:{pass_number}"
    return short


def candidates_text(candidates: Sequence[str], limit: int = CANDIDATE_LIMIT) -> str:
    """``'tanh_1', 'tanh_6'; and 7 more`` for an error's candidate row."""

    shown = ", ".join(repr(candidate) for candidate in candidates[:limit])
    remainder = len(candidates) - limit
    return f"{shown}; and {remainder} more" if remainder > 0 else shown


def _stored(record: Any, name: str) -> Any:
    """Read a stored attribute without triggering a raising property."""

    try:
        return vars(record)[name]
    except (TypeError, KeyError):
        pass
    return fail_open(lambda: getattr(record, name, None), lambda _error: None)
