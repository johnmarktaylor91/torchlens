"""Lightweight predicate-recording namespace for TorchLens."""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any

_LAZY_ATTRS = {
    "ActivationRecord": ("torchlens.recording.types", "ActivationRecord"),
    "CaptureSpec": ("torchlens.recording.types", "CaptureSpec"),
    "GradRecord": ("torchlens.recording.types", "GradRecord"),
    "GradRecordContext": ("torchlens.recording.types", "GradRecordContext"),
    "StopSignal": ("torchlens.recording._halt", "StopSignal"),
    "ModuleStackFrame": ("torchlens.recording.types", "ModuleStackFrame"),
    "Recorder": ("torchlens.recording._recorder", "Recorder"),
    "DryRun": ("torchlens.recording.types", "DryRun"),
    "RecordContext": ("torchlens.recording.types", "RecordContext"),
    "_DEFERRED_VALUE": ("torchlens._ir.predicate", "_DEFERRED_VALUE"),
    "cleanup_partial": ("torchlens.recording.cleanup", "cleanup_partial"),
    "dry_run": ("torchlens.recording.dry_run", "dry_run"),
    "stop": ("torchlens.recording._halt", "stop"),
    "recover": ("torchlens.recording.recover", "recover"),
}

__all__ = [
    "ActivationRecord",
    "CaptureSpec",
    "GradRecord",
    "GradRecordContext",
    "StopSignal",
    "ModuleStackFrame",
    "Recorder",
    "DryRun",
    "RecordContext",
    "_DEFERRED_VALUE",
    "cleanup_partial",
    "dry_run",
    "stop",
    "recover",
]


class _FastlogFacadeModule(_ModuleType):
    """Keep lazy public exports ahead of equally named child modules.

    Importing ``torchlens.recording.recover`` installs the child module as
    ``torchlens.recording.recover`` before this facade can resolve the exported
    ``recover`` function.  Eager package initialization imported the function
    afterwards, so the function won.  Restore that ordering whenever an
    attribute is read from the lazy facade.
    """

    def __getattribute__(self, name: str) -> Any:
        """Return lazy public exports instead of shadowing child modules.

        Parameters
        ----------
        name:
            Attribute requested from the fastlog package.

        Returns
        -------
        Any
            The requested package attribute.
        """

        lazy_attrs = super().__getattribute__("_LAZY_ATTRS")
        if name in lazy_attrs:
            module_path, attr_name = lazy_attrs[name]
            value = vars(self).get(name)
            if attr_name is not None and isinstance(value, _ModuleType):
                value = getattr(_importlib.import_module(module_path), attr_name)
                vars(self)[name] = value
                return value
        return super().__getattribute__(name)


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

    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        value = getattr(_importlib.import_module(module_path), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_sys.modules[__name__].__class__ = _FastlogFacadeModule
