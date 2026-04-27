"""Session-local state for fastlog predicate recording."""

from __future__ import annotations

import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from .options import RecordingOptions
from .types import (
    ActivationRecord,
    ModuleStackFrame,
    PredicateFailure,
    RecordContext,
    Recording,
    StorageIntent,
)

_active_recording_state: "RecordingState | None" = None


def _resolve_storage_intent(options: RecordingOptions) -> StorageIntent:
    """Resolve storage destinations from StreamingOptions."""

    if options.streaming is None or options.streaming.bundle_path is None:
        return StorageIntent(in_ram=True, on_disk=False)
    return StorageIntent(
        in_ram=options.streaming.retain_in_memory,
        on_disk=True,
    )


@dataclass(slots=True)
class RecordingState:
    """Mutable state for one active predicate recording pass."""

    options: RecordingOptions
    recording: Recording
    history: deque[RecordContext] = field(default_factory=deque)
    op_counts: dict[str, int] = field(default_factory=dict)
    module_stack: list[ModuleStackFrame] = field(default_factory=list)
    predicate_failures: list[PredicateFailure] = field(default_factory=list)
    predicate_failure_overflow_count: int = 0
    error_slot: BaseException | None = None
    sample_id: str | int | None = None
    pass_index: int = 0
    event_index: int = 0
    op_index: int = 0
    storage_intent: StorageIntent = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived storage policy."""

        self.storage_intent = _resolve_storage_intent(self.options)

    def append_context(self, ctx: RecordContext) -> None:
        """Append an event context to the bounded sliding window."""

        if self.options.history_size == 0:
            return
        self.history.append(ctx)
        while len(self.history) > self.options.history_size:
            self.history.popleft()

    def add_record(self, record: ActivationRecord) -> None:
        """Append a retained activation record and update indexes."""

        index = len(self.recording.records)
        self.recording.records.append(record)
        self.recording.by_pass.setdefault(record.ctx.pass_index, []).append(index)
        self.recording.by_label.setdefault(record.ctx.label, []).append(
            (record.ctx.pass_index, index)
        )
        if record.ctx.raw_label is not None:
            self.recording.by_label.setdefault(record.ctx.raw_label, []).append(
                (record.ctx.pass_index, index)
            )
        if record.ctx.module_address is not None:
            self.recording.by_module_address.setdefault(record.ctx.module_address, []).append(index)
        object.__setattr__(self.recording, "n_records", len(self.recording.records))

    def add_predicate_failure(self, ctx: RecordContext, exc: BaseException) -> None:
        """Record a predicate failure subject to the configured cap."""

        failure = PredicateFailure(
            event_index=ctx.event_index,
            kind=ctx.kind,
            label=ctx.label,
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
        if len(self.predicate_failures) < self.options.max_predicate_failures:
            self.predicate_failures.append(failure)
        else:
            self.predicate_failure_overflow_count += 1
        object.__setattr__(self.recording, "predicate_failures", list(self.predicate_failures))
        object.__setattr__(
            self.recording,
            "predicate_failure_overflow_count",
            self.predicate_failure_overflow_count,
        )


def get_active_recording_state() -> RecordingState:
    """Return the active recording state or fail clearly."""

    if _active_recording_state is None:
        raise RuntimeError("fastlog predicate state is not active")
    return _active_recording_state


@contextmanager
def active_recording_state(state: RecordingState) -> Iterator[RecordingState]:
    """Install fastlog recording state for one active logging scope."""

    global _active_recording_state
    previous = _active_recording_state
    _active_recording_state = state
    try:
        yield state
    finally:
        _active_recording_state = previous
