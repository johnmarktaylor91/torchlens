"""Trace-label helpers for the NaN and Inf finders: label lookup and unsaved ancestors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import _ordered_ops

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace


def _op_by_any_label(trace: Trace) -> dict[str, Op]:
    """Build a lookup of final and raw labels to trace ops.

    Parameters
    ----------
    trace:
        Trace whose graph labels should be indexed.

    Returns
    -------
    dict[str, Op]
        Label lookup for parent traversal.
    """

    lookup: dict[str, Op] = {}
    for op in _ordered_ops(trace):
        for label in (
            getattr(op, "label", None),
            getattr(op, "layer_label", None),
            getattr(op, "_label_raw", None),
        ):
            if isinstance(label, str):
                lookup[label] = op
    return lookup


def _op_label(op: Op) -> str:
    """Return the canonical public label for an op.

    Routes through the ONE resolver convention (observe item 1): the step-8
    final LOOKUP spelling -- bare ``layer_label`` for single-pass layers,
    pass-qualified ``label`` for multi-pass layers (an unqualified spelling on
    a recurrent block would name every pass at once). A record with no
    finalized label renders ``"unknown"``; a raw spelling is NEVER stripped
    into a public one, because the raw-to-final ordinal offset is not a
    constant and stripping fabricates a name that may belong to a different
    op.

    Parameters
    ----------
    op:
        Operation whose label should be rendered.

    Returns
    -------
    str
        Canonical public label, or ``"unknown"`` when none exists.
    """

    from .._capture._nonfinite_prefix import canonical_public_label

    label = canonical_public_label(op)
    return label if label else "unknown"


def _unsaved_ancestors(trace: Trace, op: Op) -> tuple[str, ...]:
    """Return unsaved computational ancestors of an op in execution order.

    Parameters
    ----------
    trace:
        Trace containing ``op``.
    op:
        First saved non-finite op.

    Returns
    -------
    tuple[str, ...]
        Labels of ancestor ops that do not retain output payloads.
    """

    by_label = _op_by_any_label(trace)
    pending = list(getattr(op, "parents", ()) or ())
    visited: set[str] = set()
    unsaved: set[str] = set()
    while pending:
        parent_label = pending.pop()
        if parent_label in visited:
            continue
        visited.add(parent_label)
        parent = by_label.get(parent_label)
        if parent is None:
            continue
        pending.extend(getattr(parent, "parents", ()) or ())
        if not bool(getattr(parent, "is_input", False)) and not bool(
            getattr(parent, "has_saved_activation", False)
        ):
            unsaved.add(_op_label(parent))
    return tuple(
        _op_label(candidate) for candidate in _ordered_ops(trace) if _op_label(candidate) in unsaved
    )
