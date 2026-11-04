"""Bind a resolved selection to the trace ``do()`` targets, across fork lineage.

A fork shares its parent's recorded sites: the same ``(layer_label,
pass_index)`` addresses, the same structural site keys and the same output
index spaces. A selection resolved on a trace therefore names the same sites
on every fork descended from it, so ``trace.fork().do(selection, edit)`` binds
the selection to the fork by SITE identity instead of refusing it on object
identity. Every row is checked on the fork (its record exists, and for an
activation site the structural site key and output shape agree), so a fork
whose graph has since changed refuses typed rather than guessing.

A selection resolved on a trace outside the target's fork ancestry keeps the
``selection_trace_mismatch`` refusal: cross-run bridging stays the explicit
``Selection.align_to`` door. The parent's selection is never mutated; the
fork receives a new frozen ``Selection`` over copied rows.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ._site_record import bind_entries
from .errors import SelectionError
from .selection import Selection, Site, _site_shape

__all__ = ("bind_to_target",)


def _fork_ancestors(trace: Any) -> list[Any]:
    """Return ``trace``'s fork ancestors, nearest first (cycle-safe)."""

    ancestors: list[Any] = []
    seen = {id(trace)}
    parent = getattr(trace, "parent_trace", None)
    while parent is not None and id(parent) not in seen:
        ancestors.append(parent)
        seen.add(id(parent))
        parent = getattr(parent, "parent_trace", None)
    return ancestors


def _site_drift(entry: Site, record: Any) -> str | None:
    """Return why a fork's record is not the row's site, or ``None`` when it is."""

    if entry.kind != "ACT":
        return None
    expected_key = entry.structural_site_key
    if expected_key is not None and getattr(record, "site_key", None) != expected_key:
        return f"its structural site key is {getattr(record, 'site_key', None)!r}, not {expected_key!r}"
    shape = _site_shape(record)
    if shape != entry.shape:
        return f"its output index space is {shape!r}, not {entry.shape!r}"
    return None


def _fork_row(entry: Site, fork: Any, source: Any) -> Site:
    """Copy one row onto ``fork``, refusing typed when the fork lacks the same site."""

    row = dataclasses.replace(entry)
    bind_entries(fork, (row,))
    try:
        record = row.record
    except SelectionError as error:
        drift: str | None = (
            f"the fork has no such site ({error.fields.get('reason', 'unresolvable')})"
        )
    else:
        drift = _site_drift(entry, record)
    if drift is not None:
        raise SelectionError(
            f"the resolved selection is bound to {getattr(source, 'name', None)!r}, an "
            f"ancestor of this fork, but site {entry.key!r} does not carry over: {drift}. "
            "Resolve the selection on the fork itself.",
            code="selection_trace_mismatch",
            remedy="resolve the selection on the fork itself",
            site=repr(entry.key),
        )
    return row


def bind_to_target(resolved: Selection, trace: Any, message: str) -> Selection:
    """Return ``resolved`` bound to ``trace``, rebinding onto a fork by site identity.

    Parameters
    ----------
    resolved:
        A resolved selection given to ``do()``.
    trace:
        The trace the edit targets.
    message:
        The refusal text for a selection bound to a trace outside the target's
        fork ancestry.

    Returns
    -------
    Selection
        ``resolved`` itself when it is bound to ``trace``; otherwise a new
        selection over the same sites and masks, bound to ``trace``.

    Raises
    ------
    SelectionError
        ``selection_trace_mismatch`` when ``resolved`` is bound to a trace that
        is not ``trace`` or one of its fork ancestors, or when a row's site does
        not carry over to the fork unchanged.
    """

    source = resolved._trace
    if source is trace:
        return resolved
    if not any(ancestor is source for ancestor in _fork_ancestors(trace)):
        raise SelectionError(
            message,
            code="selection_trace_mismatch",
            remedy="resolve the selection on this trace, or bridge with resolved.align_to(trace)",
        )
    rows = tuple(_fork_row(entry, trace, source) for entry in resolved)
    return Selection(trace, resolved.kind, rows, selector=resolved.selector)
