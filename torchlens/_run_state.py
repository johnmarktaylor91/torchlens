"""Run-state ownership for TorchLens intervention execution.

This module intentionally stays outside ``torchlens.intervention`` so
``torchlens._state`` can import ``RunState`` later without creating a cycle.
"""

from enum import Enum


class RunState(Enum):
    """User-visible operational state machine for intervention runs."""

    PRISTINE = "pristine"
    SPEC_STALE = "spec_stale"
    REPLAY_PROPAGATED = "replay_propagated"
    RERUN_PROPAGATED = "rerun_propagated"
    LIVE_CAPTURED = "live_captured"
    DIRECT_WRITE_DIRTY = "direct_write_dirty"
    APPENDED = "appended"


__all__ = ["RunState"]
