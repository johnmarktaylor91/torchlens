"""Typed refusal classes for the tracker family (workstream F26).

Every raise site in ``torchlens.trackers`` carries a stable
``fields["code"]`` plus a non-empty ``fields["remedy"]`` and is ledgered in
``docs/reference/error_refusal_contract.md``. Callers branch on the code,
never on message text. The loud-failure contract (trackers memo 3.13) lives
in these classes: nothing in this package ever becomes an empty dashboard
panel without a typed explanation.
"""

from __future__ import annotations

from ..errors._base import CaptureError, ConfigurationError

__tl_layer__ = "L8"


class TrackersError(ConfigurationError):
    """Base for tracker-family configuration refusals."""


class SinkProtocolError(TrackersError):
    """Raised for sink capability and sink-object contract refusals."""


class SinkDeliveryError(TrackersError):
    """Raised when a sink write/flush/close fails; the sink latches failed."""


class TagGrammarError(TrackersError):
    """Raised for tag-safety and namespace-grammar refusals (D4 majority)."""


class WatchConfigError(TrackersError):
    """Raised for ``watch()`` attach-time validation refusals."""


class WatchRuntimeError(CaptureError):
    """Raised for watch step-law and lifecycle misuse at run time."""


__all__ = [
    "SinkDeliveryError",
    "SinkProtocolError",
    "TagGrammarError",
    "TrackersError",
    "WatchConfigError",
    "WatchRuntimeError",
]
