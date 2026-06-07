"""Lightweight predicate-recording namespace for TorchLens."""

from ._halt import HaltSignal, halt
from ._record_one_shot import record
from ._recorder import Recorder
from .cleanup import cleanup_partial
from .dry_run import dry_run
from .exceptions import (
    BundleNotFinalizedError,
    InvalidStorageError,
    PredicateError,
    RecorderStateError,
    RecordingConfigError,
    RecordContextFieldError,
    RecoveryError,
)
from .options import RecordingOptions
from .recover import load, recover
from .types import (
    ActivationRecord,
    CaptureSpec,
    GradientRecord,
    GradRecordContext,
    ModuleStackFrame,
    RecordContext,
    Recording,
    RecordingTrace,
)
from ..ir.predicate import MLXValueUnavailableError, _DEFERRED_VALUE
from ..visualization.fastlog_preview import preview_fastlog as preview

__all__ = [
    "ActivationRecord",
    "BundleNotFinalizedError",
    "CaptureSpec",
    "GradientRecord",
    "GradRecordContext",
    "HaltSignal",
    "InvalidStorageError",
    "MLXValueUnavailableError",
    "ModuleStackFrame",
    "PredicateError",
    "Recorder",
    "RecorderStateError",
    "Recording",
    "RecordingConfigError",
    "RecordingOptions",
    "RecordingTrace",
    "RecordContext",
    "RecordContextFieldError",
    "RecoveryError",
    "_DEFERRED_VALUE",
    "cleanup_partial",
    "dry_run",
    "halt",
    "load",
    "preview",
    "recover",
    "record",
]
