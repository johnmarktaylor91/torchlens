"""Bind a resolved selection to a trace or to a fork descended from its trace.

A fork shares its parent's recorded sites (the same ``(layer_label,
pass_index)`` addresses, structural site keys and output index spaces), so
:func:`bind_to_target` binds a selection resolved on a trace to any fork
descended from it by SITE identity instead of refusing it on object identity.
Every row is checked on the fork (its record exists, and for an activation
site the structural site key and output shape agree), so a fork whose graph
has since changed refuses typed. A selection resolved on a trace outside the
fork's ancestry keeps the ``selection_trace_mismatch`` refusal; cross-run
bridging stays the explicit ``Selection.align_to``. The source selection is
never mutated: the fork receives a new frozen ``Selection`` over copied rows,
each row's ``provenance.source`` disclosing the ancestor it was carried over
from (``[carried over to fork from '<name>']``).

Every door that accepts an already-resolved selection binds through here:
``do()``, ``trace[selection]`` (``Selection._resolve``), and the selection
operands embedded in lazy producers (a value producer's ``within``
population, ``random_like`` operands, graph-producer endpoints). The module
imports nothing from ``selection.py`` so that module can call it.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member
from ._site_record import bind_entries
from .errors import SelectionError

if TYPE_CHECKING:
    from .selection import Selection, Site

#: The refusal text for a selection bound to a trace outside the target's ancestry.
FOREIGN_TRACE_MESSAGE = (
    "the resolved selection is bound to a different trace; resolve against "
    "this trace first, or bridge explicitly with resolved.align_to(trace) "
    "(cross-run alignment on L1 site keys, same-policy captures only)."
)


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
    if expected_key is not None and record_member(record, "site_key", None) != expected_key:
        return f"its structural site key is {record_member(record, 'site_key', None)!r}, not {expected_key!r}"
    shape = getattr(record, "shape", None)
    if shape != entry.shape:
        return f"its output index space is {shape!r}, not {entry.shape!r}"
    return None


def _fork_row(entry: Site, fork: Any, source: Any) -> Site:
    """Copy one row onto ``fork``, refusing typed when the fork lacks the same site.

    The copy's ``provenance.source`` names ``source``, the ancestor the row was
    resolved on, so the rebound row discloses its origin.
    """

    source_label = str(getattr(source, "name", "") or "ancestor trace")
    provenance = dataclasses.replace(
        entry.provenance,
        source=f"{entry.provenance.source} [carried over to fork from {source_label!r}]",
    )
    row = dataclasses.replace(entry, provenance=provenance)
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


def bind_to_target(
    resolved: Selection, trace: Any, message: str = FOREIGN_TRACE_MESSAGE
) -> Selection:
    """Return ``resolved`` bound to ``trace``, rebinding onto a fork by site identity.

    Parameters
    ----------
    resolved:
        A resolved selection given to a door that targets ``trace``.
    trace:
        The trace the selection is used on.
    message:
        The refusal text for a selection bound to a trace outside the target's
        fork ancestry.

    Returns
    -------
    Selection
        ``resolved`` itself when it is bound to ``trace``; otherwise a new
        selection over the same sites and masks, bound to ``trace``, whose
        rows' provenance names the ancestor they were carried over from.

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
    return type(resolved)(trace, resolved.kind, rows, selector=resolved.selector)
