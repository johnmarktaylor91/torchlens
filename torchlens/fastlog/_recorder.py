"""Public Recorder context manager for fastlog sessions."""

from __future__ import annotations

from types import TracebackType
from typing import Any

from torch import nn

from .._deprecations import MISSING, MissingType
from ..decoration.model_prep import _ensure_model_prepared
from ..options import StreamingOptions
from ._orchestrator import _empty_recording, _run_predicate_pass
from ._state import RecordingState
from ._validation import validate_recording_options
from .exceptions import RecorderStateError
from .options import PredicateErrorMode, PredicateFn, RecordingOptions, merge_recording_options
from .types import CaptureSpec, Recording


class Recorder:
    """Context manager for explicitly captured fastlog forwards."""

    def __init__(
        self,
        model: nn.Module,
        *,
        keep_op: PredicateFn | None | MissingType = MISSING,
        keep_module: PredicateFn | None | MissingType = MISSING,
        default_op: bool | CaptureSpec | MissingType = MISSING,
        default_module: bool | CaptureSpec | MissingType = MISSING,
        history_size: int | MissingType = MISSING,
        include_source_events: bool | MissingType = MISSING,
        max_predicate_failures: int | MissingType = MISSING,
        on_predicate_error: PredicateErrorMode | MissingType = MISSING,
        streaming: StreamingOptions | None | MissingType = MISSING,
        random_seed: int | None | MissingType = MISSING,
    ) -> None:
        """Initialize a recorder and perform construction-time validation.

        Parameters
        ----------
        model:
            PyTorch module to record.
        keep_op, keep_module, default_op, default_module, history_size,
        include_source_events, max_predicate_failures, on_predicate_error, streaming,
        random_seed:
            Fastlog recording options.
        """

        self.model = model
        self.options = merge_recording_options(
            recording=None,
            keep_op=keep_op,
            keep_module=keep_module,
            default_op=default_op,
            default_module=default_module,
            history_size=history_size,
            include_source_events=include_source_events,
            max_predicate_failures=max_predicate_failures,
            on_predicate_error=on_predicate_error,
            streaming=streaming,
            random_seed=random_seed,
        )
        validate_recording_options(self.options)
        _ensure_model_prepared(self.model)
        self._state: RecordingState | None = None
        self._recording: Recording | None = None
        self._entered = False
        self._exited = False
        self._next_pass_index = 1

    def __enter__(self) -> "Recorder":
        """Enter the recorder resource scope."""

        if self._entered or self._exited:
            raise RecorderStateError("Recorder cannot be re-entered")
        recording = _empty_recording(self.options)
        self._state = RecordingState(options=self.options, recording=recording)
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
            raise RecorderStateError("Recorder is not active")
        if exc_value is None:
            self._state.finalize_storage()
            object.__setattr__(self._state.recording, "n_records", len(self._state.recording))
            self._recording = self._state.recording
        else:
            self._state.abort_storage(str(exc_value))
        self._entered = False
        self._exited = True
        if exc_value is None:
            self._state.raise_accumulated_predicate_error()

    def log(
        self,
        input_args: Any,
        input_kwargs: dict[str, Any] | None = None,
        *,
        sample_id: str | int | None = None,
    ) -> Any:
        """Capture one forward pass and return the model output.

        Parameters
        ----------
        input_args:
            Tensor, tuple, or list of positional model inputs.
        input_kwargs:
            Optional keyword arguments for the model call.
        sample_id:
            Optional caller-provided sample identifier stored on event contexts.

        Returns
        -------
        Any
            Model forward output.
        """

        if not self._entered or self._exited or self._state is None:
            raise RecorderStateError("Recorder.log() requires an active with-block")
        output, _ = _run_predicate_pass(
            self.model,
            input_args,
            input_kwargs,
            self.options,
            state=self._state,
            pass_index=self._next_pass_index,
            sample_id=sample_id,
            finalize_storage=False,
        )
        self._next_pass_index += 1
        return output

    @property
    def recording(self) -> Recording:
        """Return the finalized recording after context-manager exit."""

        if self._recording is None:
            raise RecorderStateError("Recorder.recording is only available after __exit__")
        return self._recording
