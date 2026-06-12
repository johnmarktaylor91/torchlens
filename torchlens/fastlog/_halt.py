"""Fastlog early-abort control signal."""

from __future__ import annotations

from typing import Any, NoReturn


class HaltSignal(BaseException):
    """Signal that the active fastlog recording should stop early.

    Parameters
    ----------
    reason:
        Optional user-facing reason stored on the resulting recording.
    """

    def __init__(self, reason: str = "", frontier_output: Any | None = None) -> None:
        """Initialize the halt signal.

        Parameters
        ----------
        reason:
            User-facing halt reason.
        frontier_output:
            Optional live output object used by ``trace(halt=...)`` to finalize
            a partial graph at the halt boundary.
        """

        super().__init__(reason)
        self.reason = reason
        self.frontier_output = frontier_output


def halt(reason: str = "") -> NoReturn:
    """Halt the current fastlog recording at this predicate call site.

    Parameters
    ----------
    reason:
        Optional reason stored on ``Recording.halt_reason`` and
        ``Recording.halts_by_pass``.

    Raises
    ------
    HaltSignal
        Always raised to unwind to the recorder boundary.
    """

    raise HaltSignal(reason)
