"""Shared TorchLens exception types."""

from .errors._base import CaptureError


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when activation_transform or grad_transform raises."""
