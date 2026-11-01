"""Envelope fields of one ``do()`` intervention event.

``TraceInterventionMixin._record_do_intervention_event`` writes the ONE
InterventionEvent envelope of a ``do()`` transaction; these helpers build the
parts that need no intervention-layer import: the edit names and selection
rendering of a non-spec input, and the ``extra`` disclosures (staged-only,
gradient-plan keys and the selection-batch slots). ``_DoTransaction`` is the
transaction identity both the mixin and the gradient-site plan mint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from .trace import Trace


class _DoTransaction(NamedTuple):
    """Identity of one ``do()`` transaction, minted once at the door.

    ``transaction_start`` is the monotonic bound scoping this transaction's
    FireRecords; ``audit_rows_before`` detects whether the mutation step
    already appended its own audit row.
    """

    hooks_or_site: Any
    value_or_hook: Any
    engine: str
    transaction_start: float
    audit_rows_before: int


def _edit_identity(edit: Any) -> str:
    """String identity of an edit for an audit row: its ``Edit.name`` when that is a string,
    else the callable's ``__name__``, else its type name (a replacement tensor answers
    ``name`` with a non-string)."""

    name = getattr(edit, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(getattr(edit, "__name__", type(edit).__name__))


def do_event_edit_names(
    trace: Trace, hooks_or_site: Any, value_or_hook: Any
) -> tuple[tuple[str, ...], Any]:
    """Return the edit names and selection rendering of a non-spec ``do()`` input.

    Parameters
    ----------
    trace:
        The trace the ``do`` ran on.
    hooks_or_site:
        The transaction's site input (a selection batch or one site).
    value_or_hook:
        The transaction's edit for the ``(site, edit)`` input shape.

    Returns
    -------
    tuple[tuple[str, ...], Any]
        The edit names in order and the selection rendering.
    """

    if trace._is_selection_batch(hooks_or_site):
        # One transactional envelope for the whole batch (D33): per-pair
        # edits named in order, plus the kind-specific staged-store slots
        # disclosure so cross-kind completion needs no schema change.
        edit_names = tuple(_edit_identity(pair_edit) for _sel, pair_edit in hooks_or_site)
        selection_repr = " ; ".join(repr(sel) for sel, _pair_edit in hooks_or_site)
        return edit_names, selection_repr
    edit_name = _edit_identity(value_or_hook)
    selection_repr = trace._history_site_payload(hooks_or_site)
    if not isinstance(selection_repr, str):
        selection_repr = repr(selection_repr)
    return (str(edit_name),), selection_repr


def do_event_extra(
    trace: Trace,
    hooks_or_site: Any,
    *,
    staged_only: bool,
    extra_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the ``extra`` disclosures of one ``do()`` envelope.

    Parameters
    ----------
    trace:
        The trace the ``do`` ran on.
    hooks_or_site:
        The transaction's site input.
    staged_only:
        Whether the edit was staged without any propagation attempt.
    extra_fields:
        Further envelope disclosures (the armed keys of a gradient plan).

    Returns
    -------
    dict[str, Any]
        The disclosures; empty when there are none.
    """

    extra: dict[str, Any] = {}
    if staged_only:
        extra["staged_only"] = True
    if extra_fields:
        extra.update(extra_fields)
    if trace._is_selection_batch(hooks_or_site):
        kinds = [trace._selection_batch_kind(selection) for selection, _edit in hooks_or_site]
        extra["selection_batch"] = {
            "pairs": len(hooks_or_site),
            "kinds": kinds,
            # Kind-specific staged-store slots (D33), one count per door.
            "staged_stores": {
                door: sum(kind == door.upper() for kind in kinds)
                for door in ("act", "param", "edge")
            },
        }
    return extra
