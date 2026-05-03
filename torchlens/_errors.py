"""Shared TorchLens exception types."""

from .errors._base import CaptureError


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when out_postfunc or grad_transform raises."""
