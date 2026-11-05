"""Live-event field readers behind the capture-time Op property table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .._ir.events import OpEvent


def _grad_fn_handle_from_index(trace: Trace, event: OpEvent) -> Any:
    """Read the live autograd handle from its single owner, the journal index.

    grad_fn single ownership (producer unification P2): the journal's
    ``grad_fn_handles_by_label_raw`` side index is the one handle authority.
    The event-field fallback keeps byte-identity for detached streams until
    the compat ``OpEvent`` field dies with the legacy producer.
    """

    events = getattr(trace, "capture_events", None)
    if events is not None:
        handle = events.grad_fn_handles_by_label_raw.get(event.label_raw)
        if handle is not None:
            return handle
    # Compat OpEvents still carry the handle field; decomposed OpRecords never
    # do (single ownership) and read as None here by strict-protocol default.
    return getattr(event, "grad_fn_handle", None)


def _live_grad_fn_object_id(trace: Trace, event: OpEvent) -> Any:
    """Return the id of the live autograd handle, if one is indexed."""

    handle = _grad_fn_handle_from_index(trace, event)
    return None if handle is None else id(handle)


def _live_internal_source_parents(trace: Trace, event: OpEvent) -> list[str]:
    """Return parent labels whose events carry an internal-source ancestor."""

    return [
        edge.parent_label_raw
        for edge in event.parents
        if trace.capture_events.live_index.require_event(
            edge.parent_label_raw
        ).has_internal_source_ancestor
    ]


def _live_interventions(trace: Trace, event: OpEvent) -> list[Any]:
    """Return fire records minted for this event's interventions."""

    return [result.fire_record for result in event.fire_results if result.fire_record is not None]
