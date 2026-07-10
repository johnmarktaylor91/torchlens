"""Shared TorchLens exception types."""

from .errors._base import CaptureError
from .errors._base import ConfigurationError
from .errors._base import TorchLensWarning


class TorchLensCaptureGapError(CaptureError, RuntimeError):
    """Reserved enforcement error for an unrepresented torch invocation."""


class TorchLensCaptureGapWarning(TorchLensWarning):
    """Shadow-mode report for a possible unrepresented torch invocation."""


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when activation_transform or grad_transform raises."""


class MutatedReferenceError(CaptureError, RuntimeError):
    """Raised when a reference-mode saved tensor changed before it was read."""


class PostTraceParamUnavailable(CaptureError, RuntimeError):
    """Raised when a released Param cannot re-fetch its live model parameter."""


class AmbiguousOpLookupError(ValueError):
    """Raised when a bare Op lookup matches multiple pass-qualified Ops."""


class ShapeInferenceError(ConfigurationError, RuntimeError):
    """Raised when debug input-shape inference cannot produce a valid input."""
