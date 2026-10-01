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

from ..ir.summary_role import SummaryTransform, is_summary_transform, summary
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
from ._join import (
    JoinCoverage,
    KinetoJoinResult,
    LaunchRow,
    MarkerSpan,
    join_events,
)
from ._kernels import (
    DEFAULT_DESCRIPTOR,
    Histogram,
    HistogramDescriptor,
    HistogramResult,
    Spine,
    SpineResult,
    spine_vector_fused,
)
from ._kineto import ExtractionResult, NormalizedEvent, extract_events
from ._memory_parity import (
    EMITTED_CATEGORIES,
    NEVER_EMITTED_CATEGORIES,
    categorized_key_counts,
    category_vocabulary_report,
    pinned_parity_scenario,
)
from ._memory_totals import (
    total_transformed_activation_memory,
    total_transformed_gradient_memory,
)
from ._native_profile import (
    NativeProfileResult,
    join_result_for,
    join_session,
    native_profile,
    register_join_result,
)
from ._op_tier import FacetSplitter, OpTierCollector, split_heads
from ._overhead import ArmSpec, OverheadMeasurement, RefusalPredicate, measure_overhead
from ._payloads import histogram_payload, observation_row
from ._perf_harness import (
    MIN_PUBLISHABLE_REPEATS,
    ABMeasurement,
    format_markdown,
    measure_ab,
    measure_noise_band,
)
from ._quantiles import (
    DerivedQuantile,
    WatchRenderError,
    derived_quantile,
    derived_quantiles,
)
from ._region import RegionRecord, region
from ._render import (
    RANKING_SCORE,
    HistoryView,
    SeriesPoint,
    color_by_watch,
    contact_sheet_pages,
    rank_sites,
    render_contact_sheet,
    render_detail,
    render_fan,
    render_waterfall,
)
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
    "EMITTED_CATEGORIES",
    "NEVER_EMITTED_CATEGORIES",
    "DEFAULT_RING_CAPACITY",
    "GRAD_SCALE_PROVENANCE",
    "HISTORY_SCHEMA_VERSION",
    "MIN_PUBLISHABLE_REPEATS",
    "OPTIMIZER_STATUS",
    "PHASES",
    "PRESENCE",
    "RAM_POLICIES",
    "RANKING_SCORE",
    "STEP_PROVENANCE",
    "STREAMS",
    "ABMeasurement",
    "ArmSpec",
    "CommittedBlock",
    "DerivedQuantile",
    "EventStream",
    "ExtractionResult",
    "Histogram",
    "HistogramDescriptor",
    "HistogramResult",
    "HistoryArtifactError",
    "HistoryCollector",
    "HistoryReader",
    "HistorySchemaError",
    "HistoryView",
    "HistoryWriter",
    "FacetSplitter",
    "JoinCoverage",
    "KinetoJoinResult",
    "LaunchRow",
    "MarkerSpan",
    "NativeProfileResult",
    "NormalizedEvent",
    "ObservabilityError",
    "ObservationRecord",
    "OverheadMeasurement",
    "ObserverEvent",
    "ObserverEventError",
    "OpTierCollector",
    "ProfilerSession",
    "ProfilerSessionError",
    "RamRing",
    "RefusalPredicate",
    "RegionRecord",
    "RunRecord",
    "SeriesPoint",
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
    "SummaryTransform",
    "WatchLifecycleError",
    "WatchPlan",
    "WatchPlanError",
    "WatchRenderError",
    "WatchSettings",
    "categorized_key_counts",
    "category_vocabulary_report",
    "coarsen_pair",
    "color_by_watch",
    "contact_sheet_pages",
    "derived_quantile",
    "derived_quantiles",
    "extract_events",
    "format_markdown",
    "histogram_payload",
    "is_summary_transform",
    "join_events",
    "join_result_for",
    "join_session",
    "measure_ab",
    "measure_noise_band",
    "merge_histogram_results",
    "merge_observations",
    "measure_overhead",
    "merge_spine_results",
    "native_profile",
    "observation_row",
    "pinned_parity_scenario",
    "rank_sites",
    "region",
    "register_join_result",
    "render_contact_sheet",
    "render_detail",
    "render_fan",
    "render_waterfall",
    "session",
    "spine_vector_fused",
    "split_heads",
    "summary",
    "total_transformed_activation_memory",
    "total_transformed_gradient_memory",
    "validate_step_order",
]
