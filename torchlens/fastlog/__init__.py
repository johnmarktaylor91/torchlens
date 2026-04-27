"""Lightweight predicate-recording namespace for TorchLens."""

from .exceptions import (
    BundleNotFinalizedError,
    PredicateError,
    RecorderStateError,
    RecordingConfigError,
    RecordContextFieldError,
    RecoveryError,
)
from .options import RecordingOptions, merge_recording_options
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

__all__ = [
    "ActivationRecord",
    "BundleNotFinalizedError",
    "CaptureSpec",
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
    "StorageIntent",
    "merge_recording_options",
]
