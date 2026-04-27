"""Public Recorder context manager for fastlog sessions."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any, cast

import torch
from torch import nn

from .._deprecations import MISSING, MissingType
from .._training_validation import TrainingModeConfigError, reject_compiled_model
from ..decoration.model_prep import _ensure_model_prepared
from ..options import StreamingOptions
from ..types import ActivationPostfunc
from ._orchestrator import _empty_recording, _run_predicate_pass
from ._state import RecordingState
from ._validation import validate_recording_options
from .exceptions import RecorderStateError
from .options import PredicateErrorMode, PredicateFn, RecordingOptions, merge_recording_options
from .types import CaptureSpec, Recording


def _rank_prefixed_streaming_options(
    streaming: StreamingOptions | None | MissingType,
) -> StreamingOptions | None | MissingType:
    """Return streaming options with a rank-local directory prefix.

    Parameters
    ----------
    streaming:
        Caller-supplied streaming options.

    Returns
    -------
    StreamingOptions | None | MissingType
        Options with ``bundle_path`` rewritten to include ``rank_NN`` when a
        bundle path is configured.
    """

    if isinstance(streaming, MissingType) or streaming is None or streaming.bundle_path is None:
        return streaming
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    bundle_path = Path(streaming.bundle_path)
    return StreamingOptions(
        bundle_path=bundle_path.parent / f"rank_{rank:02d}" / bundle_path.name,
        retain_in_memory=streaming.retain_in_memory,
        activation_callback=streaming.activation_callback,
    )


def _unwrap_ddp_for_fastlog(
    model: nn.Module,
    streaming: StreamingOptions | None | MissingType,
) -> tuple[nn.Module, StreamingOptions | None | MissingType]:
    """Unwrap DDP/DataParallel wrappers for rank-local fastlog capture.

    Parameters
    ----------
    model:
        Candidate model supplied to fastlog.
    streaming:
        Caller-supplied streaming options.

    Returns
    -------
    tuple[nn.Module, StreamingOptions | None | MissingType]
        The model to execute and possibly rewritten streaming options.
    """

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        pass
    else:
        if isinstance(model, FullyShardedDataParallel):
            raise RuntimeError(
                "torchlens.fastlog does not support FullyShardedDataParallel (FSDP): "
                "parameters are sharded across ranks and there is no unsharded module to log."
            )

    try:
        from torch.nn.parallel import DistributedDataParallel
    except ImportError:
        distributed_data_parallel: type[nn.Module] | None = None
    else:
        distributed_data_parallel = DistributedDataParallel

    if distributed_data_parallel is not None and isinstance(model, distributed_data_parallel):
        return cast(nn.Module, model.module), _rank_prefixed_streaming_options(streaming)
    if isinstance(model, nn.DataParallel):
        return cast(nn.Module, model.module), _rank_prefixed_streaming_options(streaming)
    return model, streaming


def _resolve_train_mode_default(
    *,
    field_name: str,
    value: bool | CaptureSpec | MissingType,
    train_mode: bool,
) -> bool | CaptureSpec | MissingType:
    """Resolve one default capture option for train-mode sugar."""

    if train_mode and value is MISSING:
        return CaptureSpec(keep_grad=True, save_activation=True, save_metadata=True)
    if not train_mode or value is MISSING or value is False:
        return value
    if value is True:
        raise TrainingModeConfigError(
            f"train_mode=True conflicts with {field_name}=True because True uses "
            "keep_grad=False; use CaptureSpec(keep_grad=True) or omit the default"
        )
    if isinstance(value, CaptureSpec) and not value.keep_grad:
        raise TrainingModeConfigError(
            f"train_mode=True conflicts with {field_name}=CaptureSpec(keep_grad=False)"
        )
    return value


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
        activation_postfunc: ActivationPostfunc | None | MissingType = MISSING,
        save_raw_activation: bool | MissingType = MISSING,
        train_mode: bool = False,
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
        activation_postfunc:
            Optional callable applied to each retained activation copy after
            dtype/device transforms. The callable runs under ``pause_logging``
            and must return a ``torch.Tensor``. Errors are wrapped in
            :class:`torchlens.TorchLensPostfuncError`.
        save_raw_activation:
            When ``False`` and ``activation_postfunc`` is set, only the
            transformed payload is retained on the record. Defaults to
            ``True`` to mirror the slow path.
        train_mode:
            If True, omitted defaults are promoted to ``CaptureSpec(keep_grad=True)``.
        """

        reject_compiled_model(model, api_name="torchlens.fastlog.Recorder")
        unwrapped_model, streaming = _unwrap_ddp_for_fastlog(model, streaming)
        default_op = _resolve_train_mode_default(
            field_name="default_op",
            value=default_op,
            train_mode=train_mode,
        )
        default_module = _resolve_train_mode_default(
            field_name="default_module",
            value=default_module,
            train_mode=train_mode,
        )
        self.model = unwrapped_model
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
            activation_postfunc=activation_postfunc,
            save_raw_activation=save_raw_activation,
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
