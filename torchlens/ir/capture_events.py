"""Mutable capture event accumulator for one forward pass."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
import weakref

from .events import (
    BackwardPassEnd,
    BackwardPassStart,
    ConditionalEvent,
    GradFnDiscovered,
    GradFnFired,
    ModuleEnterEvent,
    ModuleEvent,
    ModuleExitEvent,
    ModulePrepEvent,
    OpGradObserved,
    OpEvent,
    OutputVersionEvent,
    PreHookProvenanceEvent,
)
from .live_index import LiveIndex
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
    pre_hook_events: list[PreHookProvenanceEvent] = field(default_factory=list)
    conditional_events: list[ConditionalEvent] = field(default_factory=list)
    output_version_events: list[OutputVersionEvent] = field(default_factory=list)
    backward_events: list[
        BackwardPassStart | OpGradObserved | BackwardPassEnd | GradFnDiscovered | GradFnFired
    ] = field(default_factory=list)
    param_refs: dict[str, ParamRef] = field(default_factory=dict)
    raw_layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=dict)
    func_call_id_counter: int = 0
    recent_events: deque[RecordContext] = field(default_factory=deque)
    backend_session: object | None = None
    live_by_raw_label: dict[str, "LiveOpRecord"] = field(default_factory=dict)
    op_event_by_label_raw: dict[str, OpEvent] = field(default_factory=dict)
    op_event_index_by_label_raw: dict[str, int] = field(default_factory=dict)
    live_index: LiveIndex = field(default_factory=LiveIndex)
    parent_op_label_raws: dict[str, list[str]] = field(default_factory=dict)
    child_op_label_raws: dict[str, list[str]] = field(default_factory=dict)
    parent_param_label_raws: dict[str, list[str]] = field(default_factory=dict)
    output_variations_by_label_raw: dict[str, list[tuple[Any, ...]]] = field(default_factory=dict)
    replacement_template_by_label_raw: dict[str, str] = field(default_factory=dict)
    module_stack_by_label_raw: dict[str, tuple[str, ...]] = field(default_factory=dict)
    grad_fn_handles_by_label_raw: dict[str, Any] = field(default_factory=dict)
    backward_event_seq: int = 0

    def copy_for_replay(self) -> "CaptureEvents":
        """Return a structural copy safe to drain during materialization.

        ``_postprocess`` (``postprocess/_materialize.py``) destructively drains
        the event containers it is handed -- ``op_events.clear()``,
        ``module_events.clear()``, ``live_index.clear()``, and so on -- and
        ``postprocess/graph_traversal.py`` replaces entries in ``op_events`` /
        ``op_event_by_label_raw`` in place. When a long-lived, frozen
        ``Recording`` cooks itself into a ``Trace`` via ``Recording.to_trace()``
        it must NOT alias its own ``_capture_events`` into the new ``Trace``, or
        that single materialization pass silently empties the Recording's own
        read-only event stream: a second ``to_trace()`` then crashes and the
        lazy ``recording_trace`` / ``records`` accessors memoize empty/wrong
        answers. Hand ``_postprocess`` a copy instead.

        Every mutable container is duplicated into a fresh object (nested list
        values included where they are rebuilt in place); the frozen ``OpEvent``
        objects and any tensor payloads are shared by reference, so the copy is
        cheap and does not clone activations. Scalars and the opaque
        ``backend_session`` are copied by value / reference.

        Returns
        -------
        CaptureEvents
            Independent event buffer over the same underlying events.
        """

        return CaptureEvents(
            op_events=list(self.op_events),
            module_events=list(self.module_events),
            module_prep_events=list(self.module_prep_events),
            module_enter_events=list(self.module_enter_events),
            module_exit_events=list(self.module_exit_events),
            pre_hook_events=list(self.pre_hook_events),
            conditional_events=list(self.conditional_events),
            output_version_events=list(self.output_version_events),
            backward_events=list(self.backward_events),
            param_refs=dict(self.param_refs),
            raw_layer_counter=self.raw_layer_counter,
            raw_layer_type_counter=dict(self.raw_layer_type_counter),
            func_call_id_counter=self.func_call_id_counter,
            recent_events=deque(self.recent_events),
            backend_session=self.backend_session,
            live_by_raw_label=dict(self.live_by_raw_label),
            op_event_by_label_raw=dict(self.op_event_by_label_raw),
            op_event_index_by_label_raw=dict(self.op_event_index_by_label_raw),
            live_index=self.live_index.copy(),
            parent_op_label_raws={
                key: list(value) for key, value in self.parent_op_label_raws.items()
            },
            child_op_label_raws={
                key: list(value) for key, value in self.child_op_label_raws.items()
            },
            parent_param_label_raws={
                key: list(value) for key, value in self.parent_param_label_raws.items()
            },
            output_variations_by_label_raw={
                key: list(value) for key, value in self.output_variations_by_label_raw.items()
            },
            replacement_template_by_label_raw=dict(self.replacement_template_by_label_raw),
            module_stack_by_label_raw=dict(self.module_stack_by_label_raw),
            grad_fn_handles_by_label_raw=dict(self.grad_fn_handles_by_label_raw),
            backward_event_seq=self.backward_event_seq,
        )

    def next_backward_seq(self) -> int:
        """Return the next monotonic backward event sequence number."""

        self.backward_event_seq += 1
        return self.backward_event_seq

    def append(self, event: OpEvent) -> None:
        """Append a single operation event."""
        self.op_event_index_by_label_raw[event.label_raw] = len(self.op_events)
        self.op_events.append(event)
        self.op_event_by_label_raw[event.label_raw] = event
        self.live_index.append(event)
        from ..capture.session import capture_session_for_events

        session = capture_session_for_events(self)
        if session is not None:
            session.observe_event(event)

    def append_backward(
        self,
        event: BackwardPassStart
        | OpGradObserved
        | BackwardPassEnd
        | GradFnDiscovered
        | GradFnFired,
    ) -> None:
        """Append a backward sidecar event."""

        self.backward_events.append(event)

    def extend(self, events: tuple[OpEvent, ...] | list[OpEvent]) -> None:
        """Append multiple operation events in order."""
        for event in events:
            self.append(event)

    def append_output_version(self, event: OutputVersionEvent) -> None:
        """Append a parent output-version sibling event."""
        self.output_version_events.append(event)

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
    events.append(event)
    if event.grad_fn_handle is not None:
        events.grad_fn_handles_by_label_raw[event.label_raw] = event.grad_fn_handle


def replace_op_event(trace: Any, label_raw: str, **updates: Any) -> OpEvent | None:
    """Replace one emitted operation event with updated field values.

    Parameters
    ----------
    trace
        Active trace carrying the capture event buffer.
    label_raw
        Raw label identifying the operation event.
    **updates
        Dataclass field updates to apply to the frozen event.

    Returns
    -------
    OpEvent | None
        Updated event when found, otherwise ``None``.
    """

    events = getattr(trace, "capture_events", None)
    if events is None:
        return None
    event = events.op_event_by_label_raw.get(label_raw)
    if event is None:
        return None
    updated_event = replace(event, **updates)
    events.op_event_by_label_raw[label_raw] = updated_event
    index = events.op_event_index_by_label_raw.get(label_raw)
    if index is None:
        index = next(
            (
                candidate_index
                for candidate_index, candidate in enumerate(events.op_events)
                if candidate.label_raw == label_raw
            ),
            None,
        )
        if index is None:
            return updated_event
        events.op_event_index_by_label_raw[label_raw] = index
    events.op_events[index] = updated_event
    events.live_index.replace(updated_event)
    from ..capture.session import capture_session_for_events

    session = capture_session_for_events(events)
    if session is not None:
        session.replace_event(updated_event)
    return updated_event


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

    raise KeyError(
        f"{label_raw!r} has no mutable live record; use CaptureEvents.live_index instead."
    )
