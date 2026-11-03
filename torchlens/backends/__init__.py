"""Backend Protocol and public registry exports for TorchLens adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchlens._utils._private_submodules import (
    refuse_private_submodule as _refuse_private_submodule,
)

# Import for backend registration side effects.
from . import _default_specs as _default_specs  # noqa: F401
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


# Runtime only: under TYPE_CHECKING a module ``__getattr__`` would make every
# name on this module type-check as Any.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Refuse a submodule path that 3.0 made private, typed; miss plainly otherwise.

        Parameters
        ----------
        name:
            Attribute normal lookup did not find.

        Raises
        ------
        AttributeError
            Always; ``FacadeTeachingError`` (``facade_redirect``) for a private
            submodule's old public spelling.
        """

        _refuse_private_submodule(__name__, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
