"""Backend Protocol and public registry exports for TorchLens adapters."""

from __future__ import annotations

from typing import Any

from .._utils.facade import resolve_facade_attr

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

# The per-framework packages (``tl.backends.jax.GradOptions`` and friends) resolve on
# first attribute access, so ``import torchlens`` imports no framework and the
# documented spelling works in a fresh process before any capture.
_BACKEND_PACKAGES = frozenset({"jax", "mlx", "paddle", "tf", "tinygrad", "torch"})


def __getattr__(name: str) -> Any:
    """Import one backend package on first access (``tl.backends.jax``).

    Parameters
    ----------
    name:
        Attribute requested from ``torchlens.backends``.

    Returns
    -------
    Any
        The backend package module.

    Raises
    ------
    AttributeError
        Plain for unknown names; typed when the backend package fails to import.

    Examples
    --------
    >>> import torchlens as tl
    >>> hasattr(tl.backends, "no_such_backend")
    False
    """

    return resolve_facade_attr(
        owner=__name__,
        name=name,
        module_globals=globals(),
        submodules=_BACKEND_PACKAGES,
    )
