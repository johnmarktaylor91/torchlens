"""Run-state ownership for future TorchLens intervention execution.

This module intentionally stays outside ``torchlens.intervention`` so
``torchlens._state`` can import ``RunState`` later without creating a cycle.
"""

from enum import Enum


class RunState(Enum):
    """Future user-visible operational state machine for intervention runs."""

    IDLE = "idle"
    CAPTURING = "capturing"
    REPLAYING = "replaying"
    RERUNNING = "rerunning"


__all__ = ["RunState"]
