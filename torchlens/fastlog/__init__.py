"""Lightweight predicate-recording namespace for TorchLens."""

from __future__ import annotations

import importlib as _importlib
from typing import Any


_LAZY_ATTRS = {
    "ActivationRecord": ("torchlens.fastlog.types", "ActivationRecord"),
    "BundleNotFinalizedError": ("torchlens.fastlog.exceptions", "BundleNotFinalizedError"),
    "CaptureSpec": ("torchlens.fastlog.types", "CaptureSpec"),
    "GradientRecord": ("torchlens.fastlog.types", "GradientRecord"),
    "GradRecordContext": ("torchlens.fastlog.types", "GradRecordContext"),
    "HaltSignal": ("torchlens.fastlog._halt", "HaltSignal"),
    "InvalidStorageError": ("torchlens.fastlog.exceptions", "InvalidStorageError"),
    "MLXValueUnavailableError": ("torchlens.ir.predicate", "MLXValueUnavailableError"),
    "ModuleStackFrame": ("torchlens.fastlog.types", "ModuleStackFrame"),
    "PredicateError": ("torchlens.fastlog.exceptions", "PredicateError"),
    "Recorder": ("torchlens.fastlog._recorder", "Recorder"),
    "RecorderStateError": ("torchlens.fastlog.exceptions", "RecorderStateError"),
    "Recording": ("torchlens.fastlog.types", "Recording"),
    "RecordingConfigError": ("torchlens.fastlog.exceptions", "RecordingConfigError"),
    "RecordingOptions": ("torchlens.fastlog.options", "RecordingOptions"),
    "RecordingTrace": ("torchlens.fastlog.types", "RecordingTrace"),
    "RecordContext": ("torchlens.fastlog.types", "RecordContext"),
    "RecordContextFieldError": ("torchlens.fastlog.exceptions", "RecordContextFieldError"),
    "RecoveryError": ("torchlens.fastlog.exceptions", "RecoveryError"),
    "_DEFERRED_VALUE": ("torchlens.ir.predicate", "_DEFERRED_VALUE"),
    "cleanup_partial": ("torchlens.fastlog.cleanup", "cleanup_partial"),
    "dry_run": ("torchlens.fastlog.dry_run", "dry_run"),
    "halt": ("torchlens.fastlog._halt", "halt"),
    "load": ("torchlens.fastlog.recover", "load"),
    "recover": ("torchlens.fastlog.recover", "recover"),
    "record": ("torchlens.fastlog._record_one_shot", "record"),
}

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


def __getattr__(name: str) -> Any:
    """Resolve public fastlog attributes only when they are requested.

    Parameters
    ----------
    name:
        Public fastlog attribute to resolve.

    Returns
    -------
    Any
        Resolved public object.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the fastlog public surface.
    """

    if name == "preview":
        from ..visualization.fastlog_preview import preview_fastlog

        return preview_fastlog
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        value = getattr(_importlib.import_module(module_path), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
