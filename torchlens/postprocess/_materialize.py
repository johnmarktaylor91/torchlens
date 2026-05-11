"""Step 0 of postprocess: drain CaptureEvents after live raw state is registered."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchlens.ir import CaptureEvents
from torchlens.ir.events import OpEvent

if TYPE_CHECKING:
    from torchlens.data_classes.model_log import Trace
    from torchlens.data_classes.op_log import OpLog


def register_materialized_event(trace: "Trace", event: OpEvent, op_log: "OpLog") -> None:
    """Append an event and expose its live log to in-flight hooks.

    Parameters
    ----------
    trace
        Active trace receiving the event.
    event
        Operation event emitted for the new log.
    op_log
        Live operation log registered in the transient build state.

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
    _register_raw_log(trace, event, op_log)


def materialize_from_events(trace: "Trace", events: CaptureEvents) -> None:
    """Clear capture event payloads after live build-state registration.

    Parameters
    ----------
    trace
        Trace whose transient build state was populated during capture.
    events
        Mutable event accumulator owned by the active capture session.

    Returns
    -------
    None
        Destructively consumes event lists after postprocess no longer needs them.
    """

    events.op_events.clear()
    events.module_events.clear()
    events.conditional_events.clear()


def _register_raw_log(trace: "Trace", event: OpEvent, op_log: "OpLog") -> None:
    """Register one live log in transient raw lookup structures.

    Parameters
    ----------
    trace
        Trace receiving the raw log.
    event
        Operation event corresponding to ``op_log``.
    op_log
        Live operation log to register.

    Returns
    -------
    None
        Mutates ``trace._raw_layer_dict`` and ``trace._raw_layer_labels_list``.
    """

    trace._raw_layer_dict[event.label_raw] = op_log
    trace._raw_layer_labels_list.append(event.label_raw)
