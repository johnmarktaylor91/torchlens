"""Intervention spec state and history records of a ``Trace`` (L1).

``intervention._trace_edits.record_do_intervention_event`` writes the ONE
InterventionEvent envelope of a ``do()`` transaction; the helpers here build the
parts that need no intervention-layer import: the edit names and selection
rendering of a non-spec input, and the ``extra`` disclosures (staged-only,
gradient-plan keys and the selection-batch slots). ``_DoTransaction`` is the
transaction identity both the ``do`` door and the gradient-site plan mint. The
record-side state helpers (the mutable spec, its revision and staleness marks,
the ``state_history`` operation records) live here as functions taking the
trace first; ``Trace`` keeps thin methods for them.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, NamedTuple

from .._trace_state import TraceState
from ._trace_grad_sites import refresh_armed_annotation

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


def _edit_identity(edit: Any, fallback: str | None = None) -> str:
    """String identity of an edit for an audit row or message.

    An ``Edit`` carries its ``name`` and a hook callable carries ``__name__``;
    only non-empty string attributes count (a replacement tensor answers
    ``name`` with ``None``), and the persisted audit contract requires a string
    ``edit`` field. Otherwise the answer is ``fallback``, or the edit's type
    name when no fallback is given.
    """

    for attribute in ("name", "__name__"):
        candidate = getattr(edit, attribute, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return type(edit).__name__ if fallback is None else fallback


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


def history_site_payload(site: Any) -> Any:
    """Return a stable site payload for ``state_history`` records.

    Parameters
    ----------
    site:
        Original selector-like site payload supplied to a mutator.

    Returns
    -------
    Any
        Plain layer labels for direct label targets, otherwise a stable
        repr-style string for human-readable history records.
    """

    if isinstance(site, str):
        return site
    selector_kind = getattr(site, "selector_kind", None)
    selector_value = getattr(site, "selector_value", None)
    if selector_kind == "label" and isinstance(selector_value, str):
        return selector_value
    return repr(site)


def record_operation(trace: Trace, op: str, **payload: Any) -> None:
    """Append a structured operation record to ``state_history``.

    Parameters
    ----------
    op:
        Operation name.
    **payload:
        Operation-specific metadata.

    Returns
    -------
    None
        The history list is mutated in place.
    """

    trace.state_history.append(
        {
            "op": op,
            "spec_revision": trace._spec_revision,
            "timestamp": time.monotonic(),
            **payload,
        }
    )


def recipe_is_clean(trace: Trace) -> bool:
    """Return whether propagated outs match the current spec revision.

    Returns
    -------
    bool
        ``True`` when the current out recipe revision equals the
        mutable intervention spec revision.
    """

    return trace._spec_revision == trace._out_recipe_revision


def mark_intervention_spec_mutated(trace: Trace, *, outs_stale: bool = True) -> None:
    """Invalidate cached frozen views and mark the spec stale.

    ``outs_stale=False`` is for gradient-only recipe changes: they edit
    nothing the forward recorded, so a trace whose outs matched its recipe
    keeps that state and stays current.

    Returns
    -------
    None
        This model log is mutated in place.
    """

    was_clean = recipe_is_clean(trace)
    trace._spec_revision += 1
    trace.__dict__.pop("recipe", None)
    trace.__dict__.pop("_frozen_intervention_spec", None)
    trace.__dict__.pop("_cached_frozen_intervention_spec", None)
    refresh_armed_annotation(trace)
    if not outs_stale and was_clean:
        trace._out_recipe_revision = trace._spec_revision
        return
    trace.state = TraceState.SPEC_STALE
