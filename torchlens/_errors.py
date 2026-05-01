"""Shared TorchLens exception types."""

from .errors._base import CaptureError


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when activation_postfunc or gradient_postfunc raises."""
