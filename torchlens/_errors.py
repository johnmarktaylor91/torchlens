"""Shared TorchLens exception types."""


class TorchLensPostfuncError(RuntimeError):
    """Raised when activation_postfunc or gradient_postfunc raises."""
