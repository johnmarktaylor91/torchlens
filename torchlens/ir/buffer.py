"""Mutable capture event accumulator for one forward pass."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
import weakref

from .events import (
    ConditionalEvent,
    ModuleEnterEvent,
    ModuleEvent,
    ModuleExitEvent,
    ModulePrepEvent,
    OpEvent,
)
from .predicate import RecordContext
from .refs import ParamRef, ReservedLabel

if TYPE_CHECKING:
    import torch

    from .intervention import FireResult


@dataclass(slots=False)
class CaptureEvents:
    """Mutable event buffer allocated once per capture."""

    op_events: list[OpEvent] = field(default_factory=list)
    module_events: list[ModuleEvent] = field(default_factory=list)
    module_prep_events: list[ModulePrepEvent] = field(default_factory=list)
    module_enter_events: list[ModuleEnterEvent] = field(default_factory=list)
    module_exit_events: list[ModuleExitEvent] = field(default_factory=list)
    conditional_events: list[ConditionalEvent] = field(default_factory=list)
    param_refs: dict[str, ParamRef] = field(default_factory=dict)
    raw_layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=dict)
    func_call_id_counter: int = 0
    recent_events: deque[RecordContext] = field(default_factory=deque)
    backend_session: object | None = None
    live_by_raw_label: dict[str, "LiveOpRecord"] = field(default_factory=dict)
    op_event_by_label_raw: dict[str, OpEvent] = field(default_factory=dict)
    parent_op_label_raws: dict[str, list[str]] = field(default_factory=dict)
    child_op_label_raws: dict[str, list[str]] = field(default_factory=dict)
    parent_param_label_raws: dict[str, list[str]] = field(default_factory=dict)
    output_variations_by_label_raw: dict[str, list[tuple[Any, ...]]] = field(default_factory=dict)
    replacement_template_by_label_raw: dict[str, str] = field(default_factory=dict)
    module_stack_by_label_raw: dict[str, tuple[str, ...]] = field(default_factory=dict)
    grad_fn_handles_by_label_raw: dict[str, Any] = field(default_factory=dict)

    def append(self, event: OpEvent) -> None:
        """Append a single operation event."""
        self.op_events.append(event)
        self.op_event_by_label_raw[event.label_raw] = event

    def extend(self, events: tuple[OpEvent, ...] | list[OpEvent]) -> None:
        """Append multiple operation events in order."""
        for event in events:
            self.append(event)

    def reserve_label(self, layer_type: str) -> ReservedLabel:
        """Reserve the next raw label for a single output site."""
        return self.reserve_label_block(layer_type, 1)[0]

    def reserve_label_block(self, layer_type: str, n: int) -> tuple[ReservedLabel, ...]:
        """Reserve a contiguous block of raw labels for output sites."""
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return ()

        type_counter = self.raw_layer_type_counter.get(layer_type, 0)
        labels: list[ReservedLabel] = []
        for _ in range(n):
            self.raw_layer_counter += 1
            type_counter += 1
            label_raw = f"{layer_type}_{type_counter}_{self.raw_layer_counter}_raw"
            labels.append(
                ReservedLabel(
                    label=label_raw,
                    label_raw=label_raw,
                    raw_index=self.raw_layer_counter,
                    type_index=type_counter,
                    layer_type=layer_type,
                    site=label_raw,
                )
            )
        self.raw_layer_type_counter[layer_type] = type_counter
        return tuple(labels)


@dataclass(slots=True)
class LiveOpRecord:
    """Mutable capture-time projection for one raw op label.

    Parameters
    ----------
    event
        Capture event for this operation, if emitted.
    fields
        Mutable pre-postprocess field mapping used by live capture consumers.
    tensor_ref
        Weak reference to the live output tensor, when weak-referenceable.
    t_args
        Positional call arguments used for activation saving.
    t_kwargs
        Keyword call arguments used for activation saving.
    fire_results
        Intervention hook results recorded for this operation.
    """

    event: OpEvent | None
    fields: dict[str, Any]
    tensor_ref: "weakref.ReferenceType[torch.Tensor] | None"
    t_args: tuple[Any, ...]
    t_kwargs: dict[str, Any]
    fire_results: "tuple[FireResult, ...]" = ()


def register_live_event(trace: Any, event: OpEvent, live_record: LiveOpRecord) -> None:
    """Register an event and its live projection on a trace.

    Parameters
    ----------
    trace
        Active trace receiving capture events.
    event
        Operation event emitted for the new raw label.
    live_record
        Mutable live projection for capture-time consumers.

    Returns
    -------
    None
        Mutates ``trace.capture_events``.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        events = CaptureEvents()
        trace.capture_events = events
    live_record.event = event
    events.append(event)
    events.live_by_raw_label[event.label_raw] = live_record
    if event.grad_fn_handle is not None:
        events.grad_fn_handles_by_label_raw[event.label_raw] = event.grad_fn_handle


def live_record_for_label(trace: Any, label_raw: str) -> LiveOpRecord:
    """Return the live capture projection for a raw label.

    Parameters
    ----------
    trace
        Active trace.
    label_raw
        Raw operation label.

    Returns
    -------
    LiveOpRecord
        Live projection for ``label_raw``.
    """

    events = getattr(trace, "capture_events", None)
    if events is not None and label_raw in events.live_by_raw_label:
        return events.live_by_raw_label[label_raw]
    legacy_log = trace._raw_layer_dict[label_raw]
    return LiveOpRecord(
        event=getattr(events, "op_event_by_label_raw", {}).get(label_raw),
        fields=legacy_log.__dict__,
        tensor_ref=None,
        t_args=(),
        t_kwargs={},
        fire_results=(),
    )
