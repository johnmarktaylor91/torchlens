"""Shared TorchLens exception types."""

from .errors._base import CaptureError


class TorchLensPostfuncError(CaptureError, RuntimeError):
    """Raised when activation_transform or grad_transform raises."""


class MutatedReferenceError(CaptureError, RuntimeError):
    """Raised when a reference-mode saved tensor changed before it was read."""


class PostTraceParamUnavailable(CaptureError, RuntimeError):
    """Raised when a released Param cannot re-fetch its live model parameter."""
