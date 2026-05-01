"""Lightweight predicate-recording namespace for TorchLens."""

from ._record_one_shot import record
from ._recorder import Recorder
from .cleanup import cleanup_partial
from .dry_run import dry_run
from .exceptions import (
    BundleNotFinalizedError,
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
    ModuleStackFrame,
    RecordContext,
    Recording,
    RecordingTrace,
)
from ..visualization.fastlog_preview import preview_fastlog as preview

__all__ = [
    "ActivationRecord",
    "BundleNotFinalizedError",
    "CaptureSpec",
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
    "cleanup_partial",
    "dry_run",
    "load",
    "preview",
    "recover",
    "record",
]
