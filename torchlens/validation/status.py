"""Replay-validation status objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValidationReplayState = Literal["available", "passed", "failed", "unavailable"]
ValidationReplaySource = Literal["live", "loaded", "unknown"]


@dataclass(frozen=True)
class ValidationReplayStatus:
    """Machine-readable replay-validation availability or result.

    Parameters
    ----------
    state:
        Replay-validation state.
    available:
        Whether replay validation can run for this trace.
    backend:
        Trace backend name.
    source:
        Whether the trace is live, loaded, or unknown.
    reason:
        Stable reason code for unavailable statuses.
    message:
        Human-readable status explanation.
    payload_load_status:
        Optional payload materialization status attached by ``tl.load``.
    """

    state: ValidationReplayState
    available: bool
    backend: str
    source: ValidationReplaySource
    reason: str | None
    message: str
    payload_load_status: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether replay validation ran and passed.

        Returns
        -------
        bool
            True only for a completed passing replay-validation result.
        """

        return self.state == "passed"

    @classmethod
    def available_live(cls, *, backend: str) -> "ValidationReplayStatus":
        """Build the default status for a live trace with validation support.

        Parameters
        ----------
        backend:
            Trace backend name.

        Returns
        -------
        ValidationReplayStatus
            Available live replay-validation status.
        """

        return cls(
            state="available",
            available=True,
            backend=backend,
            source="live",
            reason=None,
            message="Replay validation is available for this live trace.",
        )

    @classmethod
    def result(
        cls,
        *,
        passed: bool,
        backend: str,
        source: ValidationReplaySource,
        payload_load_status: str | None = None,
    ) -> "ValidationReplayStatus":
        """Build a completed replay-validation result.

        Parameters
        ----------
        passed:
            Whether replay validation passed.
        backend:
            Trace backend name.
        source:
            Trace source.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.

        Returns
        -------
        ValidationReplayStatus
            Completed pass/fail status.
        """

        state: Literal["passed", "failed"] = "passed" if passed else "failed"
        return cls(
            state=state,
            available=True,
            backend=backend,
            source=source,
            reason=None,
            message=f"Replay validation {state}.",
            payload_load_status=payload_load_status,
        )

    @classmethod
    def unavailable_loaded_runtime_stripped(
        cls,
        *,
        backend: str,
        payload_load_status: str | None = None,
    ) -> "ValidationReplayStatus":
        """Build the loaded non-torch unavailable status.

        Parameters
        ----------
        backend:
            Trace backend name.
        payload_load_status:
            Optional payload materialization status attached by ``tl.load``.

        Returns
        -------
        ValidationReplayStatus
            Explicit unavailable status for loaded traces whose runtime replay
            captures were stripped during save.
        """

        return cls(
            state="unavailable",
            available=False,
            backend=backend,
            source="loaded",
            reason="loaded_trace_runtime_capture_stripped",
            message=(
                "Replay validation is unavailable for this loaded non-torch trace; "
                "runtime capture data was stripped during portable save."
            ),
            payload_load_status=payload_load_status,
        )

    @classmethod
    def unavailable_unsupported(cls, *, backend: str) -> "ValidationReplayStatus":
        """Build an unavailable status for a backend without replay validation.

        Parameters
        ----------
        backend:
            Trace backend name.

        Returns
        -------
        ValidationReplayStatus
            Unsupported-backend status.
        """

        return cls(
            state="unavailable",
            available=False,
            backend=backend,
            source="unknown",
            reason="backend_validation_replay_unsupported",
            message=f"Backend {backend!r} does not support replay validation.",
        )

    def __bool__(self) -> bool:
        """Return bool only for completed pass/fail results.

        Returns
        -------
        bool
            True for ``state='passed'`` and False for ``state='failed'``.

        Raises
        ------
        TypeError
            If the status is availability-only or unavailable.
        """

        if self.state in {"passed", "failed"}:
            return self.passed
        raise TypeError(
            "ValidationReplayStatus is not a boolean pass/fail result unless "
            "replay validation has actually run."
        )


__all__ = ["ValidationReplayStatus", "ValidationReplayState", "ValidationReplaySource"]
