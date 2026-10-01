"""Typed refusal classes for the observability substrates (lane C06).

Every raise site in ``torchlens.observability`` carries a stable
``fields["code"]`` plus a non-empty ``fields["remedy"]`` and is ledgered in
``docs/reference/error_refusal_contract.md``. Callers branch on the code,
never on message text.
"""

from __future__ import annotations

from ..errors._base import CaptureError, ConfigurationError

__tl_layer__ = "L5"


class ObservabilityError(ConfigurationError):
    """Base for observability-substrate configuration refusals."""


class StatKernelError(ObservabilityError):
    """Raised for StreamingStat kernel input/merge refusals."""


class HistorySchemaError(ObservabilityError):
    """Raised for per-step history record and merge refusals."""


class HistoryArtifactError(ObservabilityError):
    """Raised for history artifact write/read/recovery refusals."""


class WatchPlanError(ObservabilityError):
    """Raised for module-tier collector plan and budget refusals."""


class WatchLifecycleError(CaptureError):
    """Raised for collector attach/step lifecycle misuse."""


class ObserverEventError(ObservabilityError):
    """Raised for chassis observer-event schema violations."""


class SpanError(ObservabilityError):
    """Raised for span registry, region, and label refusals."""


class ProfilerSessionError(ObservabilityError):
    """Raised for profiler session engine lifecycle refusals."""


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
