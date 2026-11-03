"""Validation error classes shared by the engine and the validation tripwire."""

from __future__ import annotations

from .._errors import _ActionableErrorMixin
from ._base import ValidationError

__all__ = ["MetadataInvariantError"]


class MetadataInvariantError(_ActionableErrorMixin, ValidationError, ValueError):
    """Raised when a metadata invariant check fails.

    Embeds the check name (e.g., ``"graph_topology"``) in the message prefix
    and stores it as an attribute for programmatic inspection in tests.
    ``_ActionableErrorMixin`` supplies ``__reduce__`` so the strict two-argument
    constructor survives pickle/deepcopy/process boundaries (R64-F1) instead of
    degrading to a bare ``TypeError`` when ``cls(*args)`` is replayed.
    """

    def __init__(self, check_name: str, message: str, *, code: str | None = None) -> None:
        """Initialize a metadata invariant failure.

        Parameters
        ----------
        check_name:
            Invariant check group name.
        message:
            Human-readable failure detail.
        code:
            Optional stable refusal code from the error-refusal contract,
            carried on ``fields["code"]``.
        """
        text = f"[{check_name}] {message}"
        if code is None:
            super().__init__(text)
        else:
            super().__init__(text, code=code)
        self.check_name = check_name
