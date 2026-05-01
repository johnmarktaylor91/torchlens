"""Exception types for fastlog predicate recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors._base import CaptureError, ConfigurationError

if TYPE_CHECKING:
    from .types import PredicateFailure, RecordContext


class RecordingConfigError(ConfigurationError, ValueError):
    """Raised when fastlog options are internally inconsistent."""


class RecorderStateError(CaptureError, RuntimeError):
    """Raised when a recorder is used outside its valid lifecycle."""


class RecoveryError(CaptureError, RuntimeError):
    """Raised when a partial fastlog bundle cannot be recovered."""


class BundleNotFinalizedError(CaptureError, RuntimeError):
    """Raised when loading a bundle that was not finalized cleanly."""


class RecordContextFieldError(ConfigurationError, AttributeError):
    """Raised when a predicate asks for a field outside the RecordContext schema."""

    def __init__(self, field: str) -> None:
        """Initialize a missing RecordContext field error.

        Parameters
        ----------
        field:
            Missing field name.
        """

        super().__init__(field)
        self.field = field

    def __str__(self) -> str:
        """Return a concise field-access error message."""

        return f"RecordContext has no field {self.field!r}"


class PredicateError(CaptureError, RuntimeError):
    """Raised when a predicate returns an invalid value or fails during evaluation."""

    def __init__(
        self,
        message: str,
        *,
        ctx: "RecordContext | None" = None,
        result: Any = None,
        failures: list["PredicateFailure"] | None = None,
        total_count: int | None = None,
        overflow: int = 0,
    ) -> None:
        """Initialize a predicate error with optional event context.

        Parameters
        ----------
        message:
            Human-readable failure reason.
        ctx:
            Predicate event context, when available.
        result:
            Invalid predicate return value, when applicable.
        failures:
            Accumulated predicate failures for deferred reporting.
        total_count:
            Total number of predicate failures observed.
        overflow:
            Number of failures omitted because the failure list was capped.
        """

        super().__init__(message)
        self.ctx = ctx
        self.result = result
        self.failures = failures or []
        self.total_count = total_count if total_count is not None else len(self.failures)
        self.overflow = overflow
