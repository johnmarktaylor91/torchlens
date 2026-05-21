"""Step 0 of postprocess: drain CaptureEvents after live raw state is registered."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchlens.ir import CaptureEvents, LiveOpRecord
from torchlens.ir.events import OpEvent

if TYPE_CHECKING:
    from torchlens.data_classes.model_log import Trace
    from torchlens.data_classes.op_log import Op


def materialize_log_from_fields(fields_dict: dict[str, object]) -> "Op":
    """Construct the live log object for one captured operation.

    Parameters
    ----------
    fields_dict
        Raw field mapping populated by the backend hot path.

    Returns
    -------
    Op
        Materialized operation or buffer log.
    """

    from torchlens.data_classes.buffer_log import Buffer
    from torchlens.data_classes.op_log import Op

    pending_blob_ids = _pop_pending_blob_ids(fields_dict)
    if fields_dict.get("is_buffer"):
        return Buffer(fields_dict)  # type: ignore[return-value]
    op_log = Op(fields_dict)  # type: ignore[arg-type]
    for field_name, blob_id in pending_blob_ids.items():
        setattr(op_log, field_name, blob_id)
    return op_log


def _pop_pending_blob_ids(fields_dict: dict[str, object]) -> dict[str, object]:
    """Remove streaming-only pending blob ids before Op construction.

    Parameters
    ----------
    fields_dict
        Raw field mapping populated by the backend hot path.

    Returns
    -------
    dict[str, object]
        Pending blob-id values keyed by Op attribute name.
    """

    pending_fields = (
        "_pending_blob_id",
        "_pending_transformed_out_blob_id",
        "_pending_grad_blob_id",
        "_pending_transformed_grad_blob_id",
    )
    return {
        field_name: fields_dict.pop(field_name)
        for field_name in pending_fields
        if field_name in fields_dict
    }


def register_materialized_event(
    trace: "Trace",
    event: OpEvent,
    op_log: "Op",
    live_record: LiveOpRecord | None = None,
) -> None:
    """Append an event and expose its live log to in-flight hooks.

    Parameters
    ----------
    trace
        Active trace receiving the event.
    event
        Operation event emitted for the new log.
    op_log
        Live operation log registered in the transient build state.
    live_record
        Optional mutable capture-time projection for this operation.

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
    if live_record is not None:
        live_record.event = event
        events.live_by_raw_label[event.label_raw] = live_record
    _register_raw_log(trace, event, op_log)


def materialize_from_events(trace: "Trace", events: CaptureEvents) -> None:
    """Materialize capture events into raw build-state logs.

    Parameters
    ----------
    trace
        Trace whose transient build state was populated during capture.
    events
        Mutable event accumulator owned by the active capture session.

    Returns
    -------
    None
        Populates raw trace lookup structures and destructively consumes event lists.
    """

    for event in events.op_events:
        live_record = events.live_by_raw_label.get(event.label_raw)
        if live_record is None:
            continue
        op_log = materialize_log_from_fields(live_record.fields)
        _register_raw_log(trace, event, op_log)

    events.op_events.clear()
    events.module_events.clear()
    events.conditional_events.clear()
    events.live_by_raw_label.clear()
    events.op_event_by_label_raw.clear()


def _register_raw_log(trace: "Trace", event: OpEvent, op_log: "Op") -> None:
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
