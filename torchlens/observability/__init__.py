"""Observability substrates (megasprint lane C06).

The shared substrate the observe / checks / trackers / explorer / snoop /
torchnative feature families build on:

- ONE per-step history artifact schema (``_schema`` / ``_artifact``): four
  normalized layers (Run / Site / StepBlock / Observation), a fixed signed
  log2 sketch grid, exact integer merges, disk-first atomic append.
- Spine + Histogram ``StreamingStat`` kernels (``_kernels``), re-exported
  through ``torchlens.stats``.
- The module-tier collector, Route A (``_collector``): discovery forward,
  persistent observers, staged per-phase transfers, both step spellings.
- The observer/check slot (``_chassis``): one bounded immutable event
  stream with the shared event schema checks and trackers both consume.
- The profiler span registry, ``region()``, and the ONE profiler session
  engine (``_spans`` / ``_region`` / ``_session``).

Access spelling is ``import torchlens.observability`` for now; root-facade
routing (``tl.region`` etc.) is an F35 registration fragment. Every spelling
here is DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from ._artifact import (
    DEFAULT_RING_CAPACITY,
    RAM_POLICIES,
    CommittedBlock,
    HistoryReader,
    HistoryWriter,
    RamRing,
    coarsen_pair,
)
from ._chassis import EventStream, ObserverEvent
from ._collector import HistoryCollector, StepTruth, WatchPlan, WatchSettings
from ._errors import (
    HistoryArtifactError,
    HistorySchemaError,
    ObservabilityError,
    ObserverEventError,
    ProfilerSessionError,
    SpanError,
    StatKernelError,
    WatchLifecycleError,
    WatchPlanError,
)
from ._kernels import (
    DEFAULT_DESCRIPTOR,
    Histogram,
    HistogramDescriptor,
    HistogramResult,
    Spine,
    SpineResult,
)
from ._region import RegionRecord, region
from ._schema import (
    GRAD_SCALE_PROVENANCE,
    HISTORY_SCHEMA_VERSION,
    OPTIMIZER_STATUS,
    PHASES,
    PRESENCE,
    STEP_PROVENANCE,
    STREAMS,
    ObservationRecord,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
    merge_histogram_results,
    merge_observations,
    merge_spine_results,
    validate_step_order,
)
from ._session import ProfilerSession, SessionResult, session
from ._spans import SpanRecord, SpanRegistry

__tl_layer__ = "L5"

__all__ = [
    "DEFAULT_DESCRIPTOR",
    "DEFAULT_RING_CAPACITY",
    "GRAD_SCALE_PROVENANCE",
    "HISTORY_SCHEMA_VERSION",
    "OPTIMIZER_STATUS",
    "PHASES",
    "PRESENCE",
    "RAM_POLICIES",
    "STEP_PROVENANCE",
    "STREAMS",
    "CommittedBlock",
    "EventStream",
    "Histogram",
    "HistogramDescriptor",
    "HistogramResult",
    "HistoryArtifactError",
    "HistoryCollector",
    "HistoryReader",
    "HistorySchemaError",
    "HistoryWriter",
    "ObservabilityError",
    "ObservationRecord",
    "ObserverEvent",
    "ObserverEventError",
    "ProfilerSession",
    "ProfilerSessionError",
    "RamRing",
    "RegionRecord",
    "RunRecord",
    "SessionResult",
    "SiteRecord",
    "SpanError",
    "SpanRecord",
    "SpanRegistry",
    "Spine",
    "SpineResult",
    "StatKernelError",
    "StepBlockRecord",
    "StepTruth",
    "WatchLifecycleError",
    "WatchPlan",
    "WatchPlanError",
    "WatchSettings",
    "coarsen_pair",
    "merge_histogram_results",
    "merge_observations",
    "merge_spine_results",
    "region",
    "session",
    "validate_step_order",
]
