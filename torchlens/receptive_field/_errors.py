"""Exception types for receptive-field queries and validation."""

from __future__ import annotations

from ..backends.registry import BackendUnsupportedError
from ..errors import TorchLensError


class ReceptiveFieldError(TorchLensError):
    """Base class for receptive-field failures."""


class AmbiguousInputError(ReceptiveFieldError, ValueError):
    """Raised when a query has more than one reachable model input."""


class AmbiguousPassError(ReceptiveFieldError, ValueError):
    """Raised when a layer-level query has multiple executed passes."""


class AmbiguousCallError(ReceptiveFieldError, ValueError):
    """Raised when a module-level query has multiple executed calls."""


class ReceptiveFieldUnavailableError(ReceptiveFieldError, RuntimeError):
    """Raised when a receptive-field result requires unavailable capture state."""


class ReceptiveFieldValidationError(ReceptiveFieldError, AssertionError):
    """Raised when receptive-field cross-validation detects a violation."""


__all__ = [
    "AmbiguousCallError",
    "AmbiguousInputError",
    "AmbiguousPassError",
    "BackendUnsupportedError",
    "ReceptiveFieldError",
    "ReceptiveFieldUnavailableError",
    "ReceptiveFieldValidationError",
]
