"""Typed refusal classes for the observability substrates (lane C06).

Every raise site in ``torchlens._observability`` carries a stable
``fields["code"]`` plus a non-empty ``fields["remedy"]`` and is ledgered in
``docs/reference/error_refusal_contract.md``. Callers branch on the code,
never on message text.
"""

from __future__ import annotations

from ..errors._base import CaptureError, ConfigurationError

__tl_layer__ = "L5"


class ObservabilityError(ConfigurationError):
    """Base for observability-substrate configuration refusals.

    Examples
    --------
    >>> issubclass(ObservabilityError, tl.errors.ConfigurationError)
    True
    >>> issubclass(ObservabilityError, tl.errors.TorchLensError)
    True
    """


class StatKernelError(ObservabilityError):
    """Raised for StreamingStat kernel input/merge refusals.

    Examples
    --------
    >>> issubclass(StatKernelError, tl.errors.ObservabilityError)
    True
    >>> issubclass(StatKernelError, tl.errors.TorchLensError)
    True
    """


class HistorySchemaError(ObservabilityError):
    """Raised for per-step history record and merge refusals.

    Examples
    --------
    >>> issubclass(HistorySchemaError, tl.errors.ObservabilityError)
    True
    >>> issubclass(HistorySchemaError, tl.errors.TorchLensError)
    True
    """


class HistoryArtifactError(ObservabilityError):
    """Raised for history artifact write/read/recovery refusals.

    Examples
    --------
    >>> issubclass(HistoryArtifactError, tl.errors.ObservabilityError)
    True
    >>> issubclass(HistoryArtifactError, tl.errors.TorchLensError)
    True
    """


class WatchPlanError(ObservabilityError):
    """Raised for module-tier collector plan and budget refusals.

    Examples
    --------
    >>> issubclass(WatchPlanError, tl.errors.ObservabilityError)
    True
    >>> issubclass(WatchPlanError, tl.errors.TorchLensError)
    True
    """


class WatchLifecycleError(CaptureError):
    """Raised for collector attach/step lifecycle misuse.

    Examples
    --------
    >>> issubclass(WatchLifecycleError, tl.errors.CaptureError)
    True
    >>> issubclass(WatchLifecycleError, tl.errors.TorchLensError)
    True
    """


class ObserverEventError(ObservabilityError):
    """Raised for chassis observer-event schema violations.

    Examples
    --------
    >>> issubclass(ObserverEventError, tl.errors.ObservabilityError)
    True
    >>> issubclass(ObserverEventError, tl.errors.TorchLensError)
    True
    """


class SpanError(ObservabilityError):
    """Raised for span registry, region, and label refusals.

    Examples
    --------
    >>> issubclass(SpanError, tl.errors.ObservabilityError)
    True
    >>> issubclass(SpanError, tl.errors.TorchLensError)
    True
    """


class ProfilerSessionError(ObservabilityError):
    """Raised for profiler session engine lifecycle refusals.

    Examples
    --------
    >>> issubclass(ProfilerSessionError, tl.errors.ObservabilityError)
    True
    >>> issubclass(ProfilerSessionError, tl.errors.TorchLensError)
    True
    """


__all__ = [
    "HistoryArtifactError",
    "HistorySchemaError",
    "ObservabilityError",
    "ObserverEventError",
    "ProfilerSessionError",
    "SpanError",
    "StatKernelError",
    "WatchLifecycleError",
    "WatchPlanError",
]
