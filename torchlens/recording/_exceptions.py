"""Exception types for fastlog predicate recording."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, NoReturn

from ..errors._base import CaptureError, ConfigurationError

if TYPE_CHECKING:
    from ._types import PredicateFailure, RecordContext


class RecordingConfigError(ConfigurationError, ValueError):
    """Raised when fastlog options are internally inconsistent.

    Examples
    --------
    >>> issubclass(RecordingConfigError, tl.errors.ConfigurationError)
    True
    >>> issubclass(RecordingConfigError, tl.errors.TorchLensError)
    True
    """


class InvalidStorageError(RecordingConfigError):
    """Raised when a selected fastlog payload cannot use the configured storage.

    Examples
    --------
    >>> issubclass(InvalidStorageError, tl.errors.RecordingConfigError)
    True
    >>> issubclass(InvalidStorageError, tl.errors.TorchLensError)
    True
    """


class RecorderStateError(CaptureError, RuntimeError):
    """Raised when a recorder is used outside its valid lifecycle.

    Examples
    --------
    >>> issubclass(RecorderStateError, tl.errors.CaptureError)
    True
    >>> issubclass(RecorderStateError, tl.errors.TorchLensError)
    True
    """


def refuse_recorder_uninitialized(problem: str) -> NoReturn:
    """Raise ``recorder_state_uninitialized`` for a recorder read before it was entered."""

    remedy = (
        "enter the recorder context (with tl.Recorder(...) as recorder) before reading it; "
        "if dry_run built the recorder, report it as a TorchLens bug"
    )
    raise RecorderStateError(
        f"{problem}. Remedy: {remedy}.", code="recorder_state_uninitialized", remedy=remedy
    )


class RecoveryError(CaptureError, RuntimeError):
    """Raised when a partial fastlog bundle cannot be recovered.

    Examples
    --------
    >>> issubclass(RecoveryError, tl.errors.CaptureError)
    True
    >>> issubclass(RecoveryError, tl.errors.TorchLensError)
    True
    """


class BundleNotFinalizedError(CaptureError, RuntimeError):
    """Raised when loading a bundle that was not finalized cleanly.

    Examples
    --------
    >>> issubclass(BundleNotFinalizedError, tl.errors.CaptureError)
    True
    >>> issubclass(BundleNotFinalizedError, tl.errors.TorchLensError)
    True
    """


class RecordContextFieldError(ConfigurationError, AttributeError):
    """Raised when a predicate asks for a field outside the RecordContext schema.

    Examples
    --------
    >>> issubclass(RecordContextFieldError, tl.errors.ConfigurationError)
    True
    >>> issubclass(RecordContextFieldError, tl.errors.TorchLensError)
    True
    """

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
    """Raised when a predicate returns an invalid value or fails during evaluation.

    Examples
    --------
    >>> issubclass(PredicateError, tl.errors.CaptureError)
    True
    >>> issubclass(PredicateError, tl.errors.TorchLensError)
    True
    """

    def __init__(
        self,
        message: str,
        *,
        ctx: RecordContext | None = None,
        result: Any = None,
        failures: list[PredicateFailure] | None = None,
        num_failures: int | None = None,
        overflow: int = 0,
        **payload: Any,
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
        num_failures:
            Total number of predicate failures observed.
        overflow:
            Number of failures omitted because the failure list was capped.
        **payload:
            Structured diagnostic payload forwarded to ``TorchLensError``
            (``code=``, ``remedy=``, ...). R65a: the historical bare
            ``super().__init__(message)`` swallowed this channel, leaving
            ``exc.fields`` empty at every raise site.
        """

        super().__init__(message, **payload)
        self.ctx = ctx
        self.result = result
        self.failures = failures or []
        self.total_count = num_failures if num_failures is not None else len(self.failures)
        self.overflow = overflow


def warn_followed_by_parent_outside_window(parent: str, *, stacklevel: int) -> None:
    """Warn that a ``tl.followed_by`` successor's parent left the lookback window.

    Torch's live predicate and the post-capture evaluator for preview backends
    share this one warning, so both report the same text.

    Parameters
    ----------
    parent:
        Raw label of the parent outside the window.
    stacklevel:
        Caller's stack level; this helper adds its own frame.
    """

    warnings.warn(
        f"followed_by parent {parent!r} is outside the lookback window; "
        "increase lookback to make this dependency queryable.",
        RuntimeWarning,
        stacklevel=stacklevel + 1,
    )
