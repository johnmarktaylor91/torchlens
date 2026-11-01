"""Pure label helpers for fastlog recording records."""

from __future__ import annotations

__tl_layer__ = "L1"

from .._ir.predicate import RecordContext


def _distinct_label_index_keys(label: str, raw_label: str | None) -> tuple[str, ...]:
    """Return the distinct label keys that should index one activation record.

    Parameters
    ----------
    label
        Primary public label for the retained record.
    raw_label
        Optional raw label alias for the same retained record.

    Returns
    -------
    tuple[str, ...]
        Unique label keys that should reference the record exactly once.
    """

    if raw_label is None or raw_label == label:
        return (label,)
    return (label, raw_label)


def _public_fastlog_layer_label(ctx: RecordContext) -> str:
    """Return a compact public label for a predicate-mode operation context."""

    if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
        return f"{ctx.layer_type}_{ctx.type_index}"
    return ctx.label
