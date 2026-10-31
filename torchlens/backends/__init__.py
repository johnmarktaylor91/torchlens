"""Backend Protocol and public registry exports for TorchLens adapters."""

from __future__ import annotations

# Import for backend registration side effects.
from . import default_specs as _default_specs  # noqa: F401
from ._protocol import CaptureBackend
from .registry import (
    TINYGRAD_BACKEND_NAME as TINYGRAD_BACKEND_NAME,
    TORCH_BACKEND_NAME as TORCH_BACKEND_NAME,
    BackendAmbiguityError,
    BackendCapabilities,
    BackendCapabilityConformanceError,
    BackendMismatchError,
    BackendName as BackendName,
    BackendPayloadUnsupportedError,
    BackendRegistryError,
    BackendRuntimeCompatibilityError,
    BackendSpec,
    BackendUnsupportedError,
    SerializationPolicy,
    UnknownBackendError,
    get_backend_spec as spec,
    register_backend_spec as register,
    registered_backend_specs as specs,
    require_capability_implementation,
    resolve_backend_spec as resolve,
    unregister_backend_spec as unregister,
)

__all__ = [
    "BackendAmbiguityError",
    "BackendCapabilities",
    "BackendCapabilityConformanceError",
    "BackendMismatchError",
    "BackendPayloadUnsupportedError",
    "BackendRegistryError",
    "BackendRuntimeCompatibilityError",
    "BackendSpec",
    "BackendUnsupportedError",
    "CaptureBackend",
    "SerializationPolicy",
    "UnknownBackendError",
    "spec",
    "specs",
    "register",
    "require_capability_implementation",
    "resolve",
    "unregister",
]
