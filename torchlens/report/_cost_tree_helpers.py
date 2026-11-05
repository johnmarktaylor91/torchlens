"""Pure helpers for the cost tree view (digests and label spellings)."""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple


def _structure_digest(
    row_id: str,
    self_flops: int | None,
    children: dict[str, list[str]],
    digests: dict[str, str],
    extra: str = "",
) -> str:
    """Bottom-up structural digest used by the strict fold signature.

    Two sibling calls fold ONLY when their entire subtrees agree on
    structure AND cost (byte-equal digests) -- the summary memo's strict
    fold doctrine: ``(xN)`` must never hide a sibling that differs.
    """

    child_digests = ",".join(digests[child] for child in children.get(row_id, ()))
    return hashlib.sha256(f"{self_flops}{extra}|{child_digests}".encode()).hexdigest()[:16]


def _label_spellings(op: Any) -> tuple[str, ...]:
    """The label spellings a GradFn's ``op_label`` may use for one op."""

    spellings = []
    for name in ("label", "layer_label", "layer_label_w_pass", "layer_label_no_pass"):
        value = getattr(op, name, None)
        if value:
            spellings.append(str(value))
    return tuple(spellings)


def _call_display(call: Any) -> str:
    """Display label for one module call (address:pass spelling kept)."""

    return str(getattr(call, "label", "?"))


class _RemainderMass(NamedTuple):
    """The conserved mass one REMAINDER row folds in."""

    flops: int
    n_ops: int
    n_unknown: int
    passes: int | None
    forward_mac: int = 0
    backward: int | None = None


def _pct(value: int | None, total: int) -> float | None:
    """Percent of the known partition total; None-safe."""

    if value is None or total <= 0:
        return None
    return 100.0 * value / total
