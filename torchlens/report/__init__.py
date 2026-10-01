"""Reporting helpers for TorchLens observer metadata."""

from __future__ import annotations

# ``log_value``'s canonical home is ``torchlens.observers`` (capture-time
# annotation WRITERS live with the observers; reporting surfaces are the
# READERS). This import is the kept compatibility alias for the historical
# ``tl.report.log_value`` spelling.
from ..observers import log_value
from ._compute_truth import ComputeAggregation, ComputeRow, RatioFact, compute_aggregation
from ._explain import explain
from ._factcore import (
    FACTCORE_SCHEMA_VERSION,
    GRAINS,
    CountsRecord,
    FactCore,
    IdentityIndex,
    MemoryFacts,
    ParamFacts,
    capture_fingerprint,
    factcore,
)
from ._health import (
    HEALTH_FACTS_ANNOTATIONS_KEY,
    HEALTH_FACTS_SCHEMA_VERSION,
    HEALTH_VERDICTS,
    HealthFacts,
    health_facts,
    nonfinite_verdict,
    normalize_health_facts,
)
from ._profile import TraceProfile, build_profile
from ._stats_table import STATS_ROW_STATES, StatsTable, StatsTableRow, build_stats_table
from ._summary_report import (
    EVIDENCE_VALUES,
    CaptureFacts,
    SummaryReport,
    SummaryRow,
    SummaryTotals,
    build_summary_report,
)

__all__ = [
    "FACTCORE_SCHEMA_VERSION",
    "GRAINS",
    "HEALTH_FACTS_ANNOTATIONS_KEY",
    "HEALTH_FACTS_SCHEMA_VERSION",
    "HEALTH_VERDICTS",
    "EVIDENCE_VALUES",
    "CaptureFacts",
    "ComputeAggregation",
    "ComputeRow",
    "CountsRecord",
    "FactCore",
    "HealthFacts",
    "IdentityIndex",
    "MemoryFacts",
    "ParamFacts",
    "RatioFact",
    "STATS_ROW_STATES",
    "StatsTable",
    "StatsTableRow",
    "SummaryReport",
    "SummaryRow",
    "SummaryTotals",
    "TraceProfile",
    "build_profile",
    "build_stats_table",
    "build_summary_report",
    "capture_fingerprint",
    "compute_aggregation",
    "explain",
    "factcore",
    "health_facts",
    "log_value",
    "nonfinite_verdict",
    "normalize_health_facts",
]
