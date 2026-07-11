"""Sibling projections over a sealed capture run core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable

from ..ir.events import OpEvent
from .session import CapturedRunCore

if TYPE_CHECKING:
    from ..fastlog.types import ActivationRecord


def _event_from_core(core: CapturedRunCore, fact_index: int) -> OpEvent:
    """Resolve one event fact with its authoritative stable-id sidecars.

    Parameters
    ----------
    core
        Sealed source for the projection.
    fact_index
        Producer-order index of the event fact.

    Returns
    -------
    OpEvent
        Immutable event view with decision and payload fields sourced from the
        ledgers keyed by the fact's stable event identity.
    """

    fact = core.event_facts[fact_index]
    event = fact.event
    decision = core.decisions.get(fact.event_id)
    payload = core.payloads.get(fact.event_id)
    if decision is not None:
        event = replace(
            event,
            predicate_matched=decision.predicate_matched,
            intervention_fired=decision.intervention_fired,
            intervention_replaced=decision.intervention_replaced,
            fire_results=decision.fire_results,
        )
    if payload is not None:
        event = replace(event, output=payload.output)
    return event


@dataclass(frozen=True, slots=True)
class TraceProjector:
    """Read Trace Step-0 operation events from a sealed run core."""

    core: CapturedRunCore

    def events(self) -> tuple[OpEvent, ...]:
        """Return operation events in producer order.

        Returns
        -------
        tuple[OpEvent, ...]
            Repeatedly readable operation facts resolved through stable-id
            decision and payload ledgers.
        """

        return tuple(
            _event_from_core(self.core, index) for index in range(len(self.core.event_facts))
        )


@dataclass(frozen=True, slots=True)
class RecordingProjection:
    """Sparse activation records and lookup indexes projected from sealed facts."""

    records: tuple["ActivationRecord", ...]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]
    events: tuple[OpEvent, ...]
    capture_events: object | None
    output_tensors: tuple[object, ...]
    output_tensor_addresses: tuple[str, ...]
    output_labels: tuple[str | None, ...]
    trace_facts: dict[str, object]
    buffer_layers: tuple[str, ...]
    internal_source_ops: tuple[str, ...]

    def prepare_trace(self, trace: Any) -> None:
        """Apply sealed run facts required before Trace postprocessing.

        Parameters
        ----------
        trace
            Fresh Trace projection shell to initialize from core facts.
        """

        trace.buffer_layers = list(self.buffer_layers)
        trace.internal_source_ops = list(self.internal_source_ops)
        trace.capture_start_time = self.trace_facts["capture_start_time"]
        trace.setup_duration = self.trace_facts["setup_duration"]
        trace.forward_duration = self.trace_facts["forward_duration"]
        trace.forward_peak_memory = self.trace_facts["forward_peak_memory"]
        trace.forward_memory_backend = self.trace_facts["forward_memory_backend"]
        trace._source_model_ref = self.trace_facts["source_model_ref"]
        trace.random_seed = self.trace_facts["random_seed"]
        trace._layer_counter = int(self.trace_facts["layer_counter"])

        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        for tensor, label in zip(self.output_tensors, self.output_labels):
            if label is not None and get_tensor_label(tensor) is None:
                set_tensor_label(tensor, label)

    def bind_halt_frontier(self, tensor: object, label: str) -> None:
        """Restore a core-attributed halt-frontier label before projection.

        Parameters
        ----------
        tensor
            Retained halt-frontier tensor selected from projected records.
        label
            Authoritative raw event label for that tensor.
        """

        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        if get_tensor_label(tensor) is None:
            set_tensor_label(tensor, label)


class RecordingProjector:
    """Build Recording activation indexes directly from sealed run cores."""

    def project(self, cores: Iterable[CapturedRunCore]) -> RecordingProjection:
        """Build the sparse activation projection without first building a Trace.

        Parameters
        ----------
        cores
            Sealed cores in Recorder pass order.

        Returns
        -------
        RecordingProjection
            Retained records and the current public lookup indexes.
        """

        from .projections import activation_record_from_event

        records: list[ActivationRecord] = []
        by_pass: dict[int, list[int]] = {}
        by_label: dict[str, list[tuple[int, int]]] = {}
        by_address: dict[str, list[int]] = {}
        all_events: list[OpEvent] = []
        captured_cores = tuple(cores)
        for core in captured_cores:
            core_events = TraceProjector(core).events()
            all_events.extend(core_events)
            stored_records = core.projection_facts.get("records", ())
            if stored_records:
                projected_records = stored_records
            else:
                projected_records = tuple(
                    record
                    for event in core_events
                    if (record := activation_record_from_event(event)) is not None
                )
            for record in projected_records:
                index = len(records)
                records.append(record)
                by_pass.setdefault(record.ctx.pass_index, []).append(index)
                by_label.setdefault(record.ctx.label, []).append((record.ctx.pass_index, index))
                if record.ctx.raw_label is not None:
                    by_label.setdefault(record.ctx.raw_label, []).append(
                        (record.ctx.pass_index, index)
                    )
                if record.ctx.address is not None:
                    by_address.setdefault(record.ctx.address, []).append(index)
        last_facts = captured_cores[-1].projection_facts if captured_cores else {}
        capture_events = next(
            (
                core.projection_facts.get("capture_events")
                for core in captured_cores
                if core.projection_facts.get("capture_events") is not None
            ),
            None,
        )
        if capture_events is not None:
            capture_events = capture_events.copy_for_replay()
            capture_events.op_events = list(all_events)
            capture_events.op_event_by_label_raw = {event.label_raw: event for event in all_events}
            capture_events.op_event_index_by_label_raw = {
                event.label_raw: index for index, event in enumerate(all_events)
            }
            capture_events.raw_layer_counter = max(
                (event.raw_index for event in all_events), default=0
            )
        return RecordingProjection(
            tuple(records),
            by_pass,
            by_label,
            by_address,
            tuple(all_events),
            capture_events,
            tuple(last_facts.get("output_tensors", ())),
            tuple(last_facts.get("output_tensor_addresses", ())),
            tuple(last_facts.get("output_labels", ())),
            dict(last_facts),
            tuple(event.label_raw for event in all_events if event.layer_type == "buffer"),
            tuple(
                event.label_raw
                for event in all_events
                if event.layer_type != "input" and not event.parents
            ),
        )
