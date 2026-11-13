"""The recorder ``tl.record`` runs for a backend that binds a record producer.

``BackendRecorder`` keeps the torch ``Recorder``'s lifecycle: a with-block,
repeated passes that reuse the cached model discovery and the compiled
selectors (never an output or a taken branch), ``on_forward_error`` partials,
the accumulate and fail-fast predicate modes, halted passes, and the same
``Recording`` with the same record indexes. Each pass drives the producer
against a fresh :class:`~torchlens.recording._backend_events.RecordSink`, so
only the selected ops' values are copied.
"""

from __future__ import annotations

import time
import traceback as traceback_module
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import Any

from .._capture._trace_capture_error import failed_forward_error
from .._capture.outcome import (
    CaptureOutcome,
    CaptureStatus,
    classify_failure_origin,
    safe_exception_repr,
    safe_exception_str,
    stamp_recording_outcome,
)
from .._capture.predicates import validate_followed_by_capability
from .._capture.projections import RecordingState, _empty_recording
from .._output_slot import (
    forward_failed as output_forward_failed,
    returned as output_returned,
    stopped as output_stopped,
)
from ._backend_events import (
    BackendRecordProducer,
    RecordSink,
    _PassFacts,
    refuse_backend_record_state,
)
from ._backend_options import RecordDoors, resolve_backend_record_config
from ._halt import StopSignal, _mark_recording_halted
from ._options import _RecordingConfig
from ._recorder import _settle_recording_outcome
from ._types import Recording
from ._validation import _warn_zero_match_capture_selectors


class BackendRecorder:
    """Context manager recording a non-torch model through its backend's producer."""

    def __init__(
        self, model: Any, *, producer: BackendRecordProducer, doors: RecordDoors | None = None
    ) -> None:
        """Resolve and validate the options, then discover the model once."""

        self.producer = producer
        self.options: _RecordingConfig = resolve_backend_record_config(
            producer, doors if doors is not None else RecordDoors()
        )
        validate_followed_by_capability(
            self.options.keep_op, api_name="record(save=...)", supports_retroactive=False
        )
        self.model = model
        self._prepared = producer.prepare(model)
        self._state: RecordingState | None = None
        self._recording: Recording | None = None
        self._entered = False
        self._exited = False
        self._failed = False
        self._next_pass_index = 1

    def __enter__(self) -> BackendRecorder:
        """Enter the recorder resource scope."""

        if self._entered or self._exited:
            refuse_backend_record_state("Recorder cannot be re-entered")
        recording = _empty_recording(self.options)
        self._state = RecordingState(options=self.options, recording=recording)
        object.__setattr__(recording, "_recording_state", self._state)
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finalize or abort the recorder resource scope."""

        _ = exc_type, traceback
        if not self._entered or self._exited or self._state is None:
            refuse_backend_record_state("Recorder is not active")
        self._entered = False
        self._exited = True
        if self._failed:
            return
        if exc_value is not None:
            self._state.abort_storage(safe_exception_str(exc_value))
            return
        self._state.finalize_storage()
        self._recording = _lean_stream_recording(self._state)
        _settle_recording_outcome(self._recording, self._state)
        _warn_zero_match_capture_selectors(self._state)
        self._state.raise_accumulated_predicate_error()

    def __call__(
        self,
        input_args: Any,
        input_kwargs: dict[str, Any] | None = None,
        *,
        sample_id: str | int | None = None,
    ) -> Any:
        """Record one forward pass and return the model output."""

        state = self._state
        if not self._entered or self._exited or state is None:
            refuse_backend_record_state("calling a Recorder requires an active with-block")
        if self._failed:
            refuse_backend_record_state(
                "Recorder is in a failed state after a forward exception; "
                "create a new Recorder for further captures."
            )
        self._reset_state_for_pass(state, sample_id)
        sink = RecordSink(
            state=state,
            codec=self.producer.codec,
            backend=str(self.producer.backend),
            pause_capture=getattr(self.producer, "pause_capture", None) or _no_pause,
        )
        try:
            output = self._run_pass(state, sink, input_args, input_kwargs)
        finally:
            state.op_context_by_raw_label.clear()
        self._next_pass_index += 1
        return output

    def _run_pass(
        self, state: RecordingState, sink: RecordSink, input_args: Any, input_kwargs: Any
    ) -> Any:
        """Drive the producer for one pass and settle its output, stop or failure."""

        state.recording.start_times.append(time.time())
        try:
            output = self.producer.run_pass(self._prepared, input_args, input_kwargs or {}, sink)
            sink.finish_pass()
        except StopSignal as stop:
            sink.stop = stop
            output = None
        except Exception as exc:
            return self._fail_pass(state, sink, exc)
        finally:
            state.recording.end_times.append(time.time())
        recording = state.recording
        object.__setattr__(recording, "num_ops", max(recording.num_ops, self._next_pass_index))
        if sink.stop is not None:
            _mark_recording_halted(recording, self._next_pass_index, sink.stop.reason)
            object.__setattr__(recording, "_output", output_stopped(sink.stop.reason))
            return None
        object.__setattr__(recording, "_output", output_returned(output))
        return output

    def _fail_pass(self, state: RecordingState, sink: RecordSink, exc: Exception) -> None:
        """Apply ``on_forward_error`` to a pass whose producer or predicate raised."""

        error = failed_forward_error(sink, exc, sink.facts.num_op_events)
        disposition = self.options.on_forward_error
        state.abort_storage(safe_exception_str(exc))
        if disposition == "raise":
            _raise(error, exc)
        partial = _lean_stream_recording(state)
        stamp_failed_backend_recording(partial, sink.facts, exc)
        self._failed = True
        self._recording = partial
        if disposition == "attach_partial":
            error.partial_recording = partial  # type: ignore[attr-defined]
            _raise(error, exc)
        return None

    def _reset_state_for_pass(self, state: RecordingState, sample_id: str | int | None) -> None:
        """Reset per-pass predicate state while keeping the retained records."""

        state.history.clear()
        state.op_counts.clear()
        state.module_stack.clear()
        state.clear_pass_indexes()
        state.sample_id = sample_id
        state.pass_index = self._next_pass_index
        state.event_index = 0
        state.step_index = 0

    @property
    def recording(self) -> Recording:
        """Return the finalized recording after context-manager exit."""

        if self._recording is None:
            refuse_backend_record_state("Recorder.recording is only available after __exit__")
        return self._recording


def backend_recorder_factory(producer: BackendRecordProducer) -> Callable[..., BackendRecorder]:
    """Return the ``tl.record`` recorder factory for one producer.

    The factory takes the torch ``Recorder``'s call shape (``model`` plus the
    keyword doors); an unknown door is a ``TypeError`` from :class:`RecordDoors`.
    """

    def build(model: Any, **doors: Any) -> BackendRecorder:
        return BackendRecorder(model, producer=producer, doors=RecordDoors(**doors))

    return build


def _lean_stream_recording(state: RecordingState) -> Recording:
    """Bind the session's recording: built records, no journal, no run cores.

    The producer stream keeps only the selected ops, so ``to_trace()`` and the
    journal reads refuse typed instead of cooking a partial graph.
    """

    recording = state.recording
    object.__setattr__(recording, "_capture_events", None)
    object.__setattr__(recording, "_captured_run_cores", ())
    object.__setattr__(recording, "_records_built", True)
    object.__setattr__(recording, "_recording_trace", None)
    object.__setattr__(recording, "_recording_state", state)
    return recording


def stamp_failed_backend_recording(
    recording: Recording, facts: _PassFacts, exc: BaseException
) -> None:
    """Stamp the torch recorder's failed-partial fields onto a backend recording.

    ``num_committed_ops`` counts the op events the failing pass emitted before
    the exception (the failing op emitted nothing); ``last_event_*`` describe the
    last emitted event, best effort, as on torch.
    """

    last_ctx = facts.last_ctx
    last_event = facts.last_event
    values: dict[str, Any] = {
        "status": "partial_error",
        "is_failed": True,
        "error_repr": safe_exception_repr(exc),
        "error_traceback": "".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "num_committed_ops": facts.num_op_events,
        "last_successful_op_label": facts.last_op_label,
        "last_event_label": None if last_ctx is None else last_ctx.label,
        "last_event_func": None if last_event is None else last_event.func_name,
        "last_event_source_line": None,
        "last_event_input_meta": None,
        "_output": output_forward_failed(safe_exception_str(exc)),
    }
    for name, value in values.items():
        object.__setattr__(recording, name, value)
    stamp_recording_outcome(
        recording,
        CaptureOutcome(
            status=CaptureStatus.FAILED,
            origin=classify_failure_origin(exc),
            reason=safe_exception_str(exc),
            error_type=type(exc).__name__,
            num_committed_ops=facts.num_op_events,
        ),
    )


def _raise(error: BaseException, cause: BaseException) -> None:
    """Raise the door's error, chained from the forward's own exception."""

    if error is cause:
        raise cause
    raise error from cause


def _no_pause() -> AbstractContextManager[None]:
    """Return a do-nothing context for producers without a capture pause."""

    return nullcontext()
