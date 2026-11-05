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
    """Base for tracker-family configuration refusals.

    Examples
    --------
    >>> issubclass(tl.errors.SinkProtocolError, TrackersError)
    True
    >>> issubclass(TrackersError, tl.errors.ConfigurationError)
    True
    """


class SinkProtocolError(TrackersError):
    """Raised for sink capability and sink-object contract refusals.

    Examples
    --------
    >>> issubclass(SinkProtocolError, tl.errors.TrackersError)
    True
    >>> issubclass(SinkProtocolError, tl.errors.TorchLensError)
    True
    """


class SinkDeliveryError(TrackersError):
    """Raised when a sink write/flush/close fails; the sink latches failed.

    Examples
    --------
    >>> issubclass(SinkDeliveryError, tl.errors.TrackersError)
    True
    >>> issubclass(SinkDeliveryError, tl.errors.TorchLensError)
    True
    """


class TagGrammarError(TrackersError):
    """Raised for tag-safety and namespace-grammar refusals (D4 majority).

    Examples
    --------
    >>> issubclass(TagGrammarError, tl.errors.TrackersError)
    True
    >>> issubclass(TagGrammarError, tl.errors.TorchLensError)
    True
    """


class WatchConfigError(TrackersError):
    """Raised for ``watch()`` attach-time validation refusals.

    Examples
    --------
    >>> issubclass(WatchConfigError, tl.errors.TrackersError)
    True
    >>> issubclass(WatchConfigError, tl.errors.TorchLensError)
    True
    """


class WatchRuntimeError(CaptureError):
    """Raised for watch step-law and lifecycle misuse at run time.

    Examples
    --------
    >>> issubclass(WatchRuntimeError, tl.errors.CaptureError)
    True
    >>> issubclass(WatchRuntimeError, tl.errors.TorchLensError)
    True
    """


__all__ = [
    "SinkDeliveryError",
    "SinkProtocolError",
    "TagGrammarError",
    "TrackersError",
    "WatchConfigError",
    "WatchRuntimeError",
]
