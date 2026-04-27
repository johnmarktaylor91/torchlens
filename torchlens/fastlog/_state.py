"""Session-local state for fastlog predicate recording."""

from __future__ import annotations

import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import torch

from .options import RecordingOptions
from .types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    PredicateFailure,
    RecordContext,
    Recording,
    StorageIntent,
)
from .exceptions import PredicateError

_active_recording_state: "RecordingState | None" = None


class _StorageBackend(Protocol):
    """Protocol implemented by fastlog storage backends."""

    def append(self, record: ActivationRecord) -> None:
        """Append one retained record."""

    def resolve_payloads(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        intent: StorageIntent,
        *,
        options: RecordingOptions,
        ctx: RecordContext | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads for one selected tensor."""

    def finalize(self) -> None:
        """Finalize storage."""

    def abort(self, reason: str) -> None:
        """Abort storage."""


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
    no_tensor_capture: bool = False
    all_contexts: list[RecordContext] = field(default_factory=list)
    storage_intent: StorageIntent = field(init=False)
    storage_backend: _StorageBackend = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived storage policy."""

        self.storage_intent = _resolve_storage_intent(self.options)
        if self.storage_intent.on_disk:
            from .storage_disk import DiskStorageBackend

            self.storage_backend = DiskStorageBackend(self.options, self.recording)
        else:
            from .storage_ram import RamStorageBackend

            self.storage_backend = RamStorageBackend(self.recording)

    def append_context(self, ctx: RecordContext) -> None:
        """Append an event context to the bounded sliding window."""

        self.all_contexts.append(ctx)
        if self.options.history_size == 0:
            return
        self.history.append(ctx)
        while len(self.history) > self.options.history_size:
            self.history.popleft()

    def add_record(self, record: ActivationRecord) -> None:
        """Append a retained activation record and update indexes."""

        self.storage_backend.append(record)
        object.__setattr__(self.recording, "n_records", len(self.recording.records))

    def resolve_storage(
        self,
        tensor: torch.Tensor,
        spec: CaptureSpec,
        *,
        ctx: RecordContext | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Resolve payloads through the active storage backend.

        Returns
        -------
        tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
            ``(ram_payload, disk_payload, transformed_ram_payload,
            transformed_disk_payload)``. Any element may be ``None``.
        """

        if self.no_tensor_capture:
            return None, None, None, None
        return self.storage_backend.resolve_payloads(
            tensor,
            spec,
            self.storage_intent,
            options=self.options,
            ctx=ctx,
        )

    def finalize_storage(self) -> None:
        """Finalize the active storage backend."""

        self.storage_backend.finalize()

    def abort_storage(self, reason: str) -> None:
        """Abort the active storage backend after a failed pass."""

        self.storage_backend.abort(reason)

    def effective_predicate_error_mode(self) -> str:
        """Return the concrete predicate exception policy for this session."""

        if self.options.on_predicate_error != "auto":
            return self.options.on_predicate_error
        return "fail-fast" if self.storage_intent.on_disk else "accumulate"

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

    def handle_predicate_exception(self, ctx: RecordContext, exc: BaseException) -> None:
        """Apply the configured predicate exception policy."""

        self.add_predicate_failure(ctx, exc)
        if self.effective_predicate_error_mode() == "fail-fast":
            raise exc

    def raise_accumulated_predicate_error(self) -> None:
        """Raise a final PredicateError when accumulated failures exist."""

        if self.effective_predicate_error_mode() != "accumulate":
            return
        if not self.predicate_failures and self.predicate_failure_overflow_count == 0:
            return
        details = ""
        if self.predicate_failures:
            details = f": {self.predicate_failures[0].traceback.strip().splitlines()[-1]}"
        raise PredicateError(
            f"fastlog predicate failed during recording{details}",
            failures=list(self.predicate_failures),
            total_count=len(self.predicate_failures) + self.predicate_failure_overflow_count,
            overflow=self.predicate_failure_overflow_count,
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
