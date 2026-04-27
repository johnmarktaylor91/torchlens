"""Lightweight predicate-recording namespace for TorchLens."""

from typing import Any

from .._deprecations import MISSING, MissingType
from ..options import StreamingOptions
from ._orchestrator import _run_predicate_pass
from .cleanup import cleanup_partial
from .exceptions import (
    BundleNotFinalizedError,
    PredicateError,
    RecorderStateError,
    RecordingConfigError,
    RecordContextFieldError,
    RecoveryError,
)
from .options import PredicateErrorMode, PredicateFn, RecordingOptions, merge_recording_options
from .recover import load, recover
from .types import (
    ActivationRecord,
    CaptureSpec,
    ModuleStackFrame,
    PredicateFailure,
    Recording,
    RecordingTrace,
    RecordContext,
    StorageIntent,
)


def record(
    model: Any,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
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
    return_output: bool = False,
    random_seed: int | None | MissingType = MISSING,
) -> Recording | tuple[Any, Recording]:
    """Record one model forward pass with fastlog predicates.

    Parameters
    ----------
    model:
        PyTorch module to execute.
    input_args:
        Tensor, list, or tuple of positional model inputs.
    input_kwargs:
        Optional keyword arguments for the model call.
    keep_op, keep_module, default_op, default_module, history_size,
    include_source_events, max_predicate_failures, on_predicate_error, streaming,
    random_seed:
        Fastlog recording options.
    return_output:
        Whether to return ``(model_output, recording)``.

    Returns
    -------
    Recording | tuple[Any, Recording]
        Fastlog recording, optionally with the model output.
    """

    options = merge_recording_options(
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
    _validate_non_empty_capture(options)
    output, recording = _run_predicate_pass(model, input_args, input_kwargs, options)
    if return_output:
        return output, recording
    return recording


def _validate_non_empty_capture(options: RecordingOptions) -> None:
    """Reject statically empty fastlog configurations.

    Parameters
    ----------
    options:
        Fully merged fastlog recording options.

    Raises
    ------
    RecordingConfigError
        If no predicate or default can keep any event.
    """

    if (
        options.keep_op is None
        and options.keep_module is None
        and options.default_op is False
        and options.default_module is False
    ):
        raise RecordingConfigError("fastlog requires a predicate or a true default capture spec")


__all__ = [
    "ActivationRecord",
    "BundleNotFinalizedError",
    "CaptureSpec",
    "cleanup_partial",
    "load",
    "ModuleStackFrame",
    "PredicateError",
    "PredicateFailure",
    "RecorderStateError",
    "Recording",
    "RecordingConfigError",
    "RecordingOptions",
    "RecordingTrace",
    "RecordContext",
    "RecordContextFieldError",
    "RecoveryError",
    "recover",
    "record",
    "StorageIntent",
    "merge_recording_options",
]
