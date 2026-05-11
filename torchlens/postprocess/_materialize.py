"""Step 0 of postprocess: materialize CaptureEvents into raw Trace state.

This is a temporary adapter shim during capture-pipeline unification. M3
introduces it so the existing postprocess steps continue to read
``_raw_layer_dict`` and related raw-order state. M6 will inline this work into
the appropriate steps and drop ``_raw_layer_dict`` from the final Trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchlens.ir import CaptureEvents
from torchlens.ir.events import OpEvent

if TYPE_CHECKING:
    from torchlens.data_classes.model_log import Trace
    from torchlens.data_classes.op_log import OpLog


def register_materialized_event(trace: "Trace", event: OpEvent) -> None:
    """Append an event and expose its materialized log to in-flight hooks.

    Parameters
    ----------
    trace
        Active trace receiving the event.
    event
        Operation event with a temporary M3 ``materialized_log`` bridge.

    Returns
    -------
    None
        Mutates ``trace.capture_events`` and the raw capture indexes.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        events = CaptureEvents()
        trace.capture_events = events
    events.append(event)
    _register_raw_log(trace, event)


def materialize_from_events(trace: "Trace", events: CaptureEvents) -> None:
    """Drain capture events into ``trace._raw_layer_dict`` and raw order lists.

    Parameters
    ----------
    trace
        Trace whose raw postprocess state should be rebuilt.
    events
        Mutable event accumulator owned by the active capture session.

    Returns
    -------
    None
        Destructively consumes ``events.op_events``.
    """

    if not events.op_events:
        return

    trace._raw_layer_dict.clear()
    trace._raw_layer_labels_list.clear()
    while events.op_events:
        event = events.op_events.pop(0)
        _register_raw_log(trace, event)
    events.module_events.clear()
    events.conditional_events.clear()


def _register_raw_log(trace: "Trace", event: OpEvent) -> None:
    """Register one materialized event in Trace raw lookup structures.

    Parameters
    ----------
    trace
        Trace receiving the raw log.
    event
        Operation event to materialize.

    Returns
    -------
    None
        Mutates ``trace._raw_layer_dict`` and ``trace._raw_layer_labels_list``.
    """

    op_log = event.materialized_log
    if op_log is None:
        raise ValueError(f"OpEvent {event.label_raw!r} has no materialized OpLog")
    typed_log = op_log  # keep the runtime object intact for parity.
    trace._raw_layer_dict[event.label_raw] = typed_log
    trace._raw_layer_labels_list.append(event.label_raw)
