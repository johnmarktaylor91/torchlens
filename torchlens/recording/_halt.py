"""Fastlog early-abort control signal and the halted-state writer for a frozen Recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from .types import Recording


class StopSignal(BaseException):
    """Signal that the active fastlog recording should stop early.

    Parameters
    ----------
    reason:
        Optional user-facing reason stored on the resulting recording.

    Examples
    --------
    >>> issubclass(StopSignal, BaseException), issubclass(StopSignal, Exception)
    (True, False)
    >>> StopSignal("enough").reason
    'enough'
    """

    def __init__(
        self,
        reason: str = "",
        frontier_output: Any | None = None,
        *,
        boundary_kind: str | None = None,
        boundary_label: str | None = None,
    ) -> None:
        """Initialize the halt signal.

        Parameters
        ----------
        reason:
            User-facing halt reason.
        frontier_output:
            Optional live output object used by ``trace(stop_after=...)`` to finalize
            a partial graph at the halt boundary.
        boundary_kind:
            Event kind at the halt boundary (settlement metadata); ``None``
            for imperative halts, which carry no event context.
        boundary_label:
            Authoritative op label at the halt boundary (raw where available).
        """

        super().__init__(reason)
        self.reason = reason
        self.frontier_output = frontier_output
        self.boundary_kind = boundary_kind
        self.boundary_label = boundary_label


def stop(reason: str = "") -> NoReturn:
    """Halt the current fastlog recording at this predicate call site.

    Parameters
    ----------
    reason:
        Optional reason stored on ``Recording.stop_reason`` and
        ``Recording.stops_by_pass``.

    Raises
    ------
    HaltSignal
        Always raised to unwind to the recorder boundary.

    Examples
    --------
    >>> def keep(ctx):
    ...     if ctx.kind == "op":
    ...         tl.recording.stop("enough")
    ...     return True
    >>> recording = tl.record(model, x, save=keep)
    >>> recording.is_stopped, recording.stop_reason
    (True, 'enough')
    """

    # F6 stop-request latch: the direct-raise public spelling must be as
    # swallow-proof as the predicate spellings. With no active capture the
    # latch no-ops and the signal propagates to the user exactly as before.
    from .. import _state

    active_trace = _state._active_trace
    if active_trace is not None:
        from .._capture.outcome import StopRequest

        active_trace.__dict__["_stop_requested"] = StopRequest(kind="stop", reason=reason)
    raise StopSignal(reason)


def _mark_recording_halted(recording: Recording, pass_index: int, reason: str) -> None:
    """Set halt state on a frozen ``Recording`` (mutated via ``object.__setattr__``); an empty
    ``reason`` means none was given.
    """

    recording.stops_by_pass.setdefault(pass_index, reason)
    if recording.is_stopped:
        return
    object.__setattr__(recording, "is_stopped", True)
    object.__setattr__(recording, "status", "halted")
    object.__setattr__(recording, "stop_reason", reason)
