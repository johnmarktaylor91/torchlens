"""Mutable capture event accumulator for one forward pass."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .events import ConditionalEvent, ModuleEvent, OpEvent
from .predicate import RecordContext
from .refs import ParamRef, ReservedLabel


@dataclass(slots=False)
class CaptureEvents:
    """Mutable event buffer allocated once per capture."""

    op_events: list[OpEvent] = field(default_factory=list)
    module_events: list[ModuleEvent] = field(default_factory=list)
    conditional_events: list[ConditionalEvent] = field(default_factory=list)
    param_refs: dict[str, ParamRef] = field(default_factory=dict)
    raw_layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=dict)
    func_call_id_counter: int = 0
    recent_events: deque[RecordContext] = field(default_factory=deque)
    backend_session: object | None = None

    def append(self, event: OpEvent) -> None:
        """Append a single operation event."""
        self.op_events.append(event)

    def extend(self, events: tuple[OpEvent, ...] | list[OpEvent]) -> None:
        """Append multiple operation events in order."""
        self.op_events.extend(events)

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
                    capture_index=self.raw_layer_counter,
                    type_index=type_counter,
                    layer_type=layer_type,
                    site=label_raw,
                )
            )
        self.raw_layer_type_counter[layer_type] = type_counter
        return tuple(labels)
