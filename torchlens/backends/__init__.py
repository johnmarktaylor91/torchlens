"""Backend Protocol and public registry exports for TorchLens adapters."""

from __future__ import annotations

from ._protocol import CaptureBackend
from .registry import (
    BackendAmbiguityError,
    BackendCapabilities,
    BackendMismatchError,
    BackendName,
    BackendPayloadUnsupportedError,
    BackendRegistryError,
    BackendRuntimeCompatibilityError,
    BackendSpec,
    BackendUnsupportedError,
    SerializationPolicy,
    UnknownBackendError,
    get_backend_spec,
    registered_backend_specs,
    register_backend_spec,
    resolve_backend_spec,
    unregister_backend_spec,
)
from . import default_specs as _default_specs

__all__ = [
    "BackendAmbiguityError",
    "BackendCapabilities",
    "BackendMismatchError",
    "BackendName",
    "BackendPayloadUnsupportedError",
    "BackendRegistryError",
    "BackendRuntimeCompatibilityError",
    "BackendSpec",
    "BackendUnsupportedError",
    "CaptureBackend",
    "SerializationPolicy",
    "UnknownBackendError",
    "get_backend_spec",
    "registered_backend_specs",
    "register_backend_spec",
    "resolve_backend_spec",
    "unregister_backend_spec",
]
