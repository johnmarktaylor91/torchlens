"""Backend Protocol and public registry exports for TorchLens adapters."""

from __future__ import annotations

# Import for backend registration side effects.
from . import default_specs as _default_specs  # noqa: F401
from ._backend_protocol import (
    REFUSAL_CODES,
    VALIDATION_REASONS,
    BackendProtocol,
    FrameworkRequirement,
)
from ._protocol import CaptureBackend
from .registry import (
    TINYGRAD_BACKEND_NAME as TINYGRAD_BACKEND_NAME,
    TORCH_BACKEND_NAME as TORCH_BACKEND_NAME,
    BackendCapabilities,
    BackendName as BackendName,
    BackendSpec,
    get_backend_spec as spec,
    register_backend_spec as register,
    registered_backend_specs as specs,
    require_capability_implementation,
    resolve_backend_spec as resolve,
    unregister_backend_spec as unregister,
)

__all__ = [
    "BackendCapabilities",
    "BackendProtocol",
    "BackendSpec",
    "CaptureBackend",
    "FrameworkRequirement",
    "REFUSAL_CODES",
    "register",
    "require_capability_implementation",
    "resolve",
    "spec",
    "specs",
    "unregister",
    "VALIDATION_REASONS",
]
