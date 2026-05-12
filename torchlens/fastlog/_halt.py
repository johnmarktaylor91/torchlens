"""Fastlog early-abort control signal."""

from __future__ import annotations

from typing import NoReturn


class HaltSignal(BaseException):
    """Signal that the active fastlog recording should stop early.

    Parameters
    ----------
    reason:
        Optional user-facing reason stored on the resulting recording.
    """

    def __init__(self, reason: str = "") -> None:
        """Initialize the halt signal."""

        super().__init__(reason)
        self.reason = reason


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
