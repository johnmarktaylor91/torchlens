"""Reporting helpers for TorchLens observer metadata."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._compute_truth import ComputeAggregation, ComputeRow, RatioFact, compute_aggregation
from ._cost_backward import (
    BACKWARD_STATES,
    BackwardEstimate,
    BackwardStatus,
    backward_estimate,
    backward_status,
    grad_enabled_at_capture,
)
from ._cost_perf import (
    CACHE_RESIDENT_BYTES_DEFAULT,
    COST_REPORT_TIERS,
    MFU_DENOMINATOR,
    PEAK_SOURCES,
    ROOFLINE_REASONS,
    CleanStepTime,
    CostReportResult,
    CostTierMeasurement,
    DevicePeaks,
    InstrumentedRateRow,
    InstrumentedRateTable,
    MachineBalance,
    MfuProvenance,
    MfuResult,
    PeakRow,
    RateCalibration,
    RooflineResult,
    RooflineRow,
    attributed_kernel_utilization,
    cost_report,
    device_peaks,
    instrumented_rate,
    instrumented_rate_calibration,
    machine_balance,
    mfu,
    roofline,
    time_clean_step,
)
from ._cost_scaling import (
    POSITION_SCALING_CLASSES,
    PaddingWaste,
    padding_waste,
    position_scaling_class,
)
from ._cost_tree import (
    COST_COLUMN_ADDITIVITY,
    COST_ROW_ROLES,
    CostTree,
    CostTreeRow,
    build_cost_tree as cost_tree,
)
from ._explain import explain
from ._factcore import (
    FACTCORE_SCHEMA_VERSION,
    GRAINS,
    CountsRecord,
    FactCore,
    IdentityIndex,
    MemoryFacts,
    ParamFacts,
    PrunedParamFacts,
    capture_fingerprint,
    factcore,
)
from ._flops_report import FlopsReport, SixNDFacts, flops_report
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
from ._registry import (
    COST_CLASSES,
    INVARIANTS,
    REGISTERS,
    SUBJECTS,
    SURFACE_REGISTRY,
    CapabilityCard,
    CapabilityRow,
    NextStep,
    SurfaceEntry,
    capability_card,
    registry_entry,
    which_do_i_use,
)
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
    "ValidationStatus",
    "ValidationFailure",
    "ValidationDiagnostic",
    "InterventionValidationReport",
    "kernel_profile",
    "KernelProfileReport",
    "MergeReport",
    "CompatWindow",
    "diff",
    "DiffReport",
    "param_health",
    "ParamHealthReport",
    "BACKWARD_STATES",
    "CACHE_RESIDENT_BYTES_DEFAULT",
    "COST_CLASSES",
    "COST_COLUMN_ADDITIVITY",
    "COST_REPORT_TIERS",
    "COST_ROW_ROLES",
    "INVARIANTS",
    "MFU_DENOMINATOR",
    "PEAK_SOURCES",
    "REGISTERS",
    "ROOFLINE_REASONS",
    "SUBJECTS",
    "SURFACE_REGISTRY",
    "FACTCORE_SCHEMA_VERSION",
    "GRAINS",
    "HEALTH_FACTS_ANNOTATIONS_KEY",
    "HEALTH_FACTS_SCHEMA_VERSION",
    "HEALTH_VERDICTS",
    "EVIDENCE_VALUES",
    "POSITION_SCALING_CLASSES",
    "BackwardEstimate",
    "BackwardStatus",
    "CapabilityCard",
    "CapabilityRow",
    "CaptureFacts",
    "CleanStepTime",
    "ComputeAggregation",
    "ComputeRow",
    "CostReportResult",
    "CostTierMeasurement",
    "CostTree",
    "CostTreeRow",
    "CountsRecord",
    "DevicePeaks",
    "FactCore",
    "FlopsReport",
    "HealthFacts",
    "IdentityIndex",
    "InstrumentedRateRow",
    "InstrumentedRateTable",
    "MachineBalance",
    "MemoryFacts",
    "MfuProvenance",
    "MfuResult",
    "NextStep",
    "PaddingWaste",
    "ParamFacts",
    "PeakRow",
    "PrunedParamFacts",
    "RateCalibration",
    "RatioFact",
    "RooflineResult",
    "RooflineRow",
    "SurfaceEntry",
    "STATS_ROW_STATES",
    "SixNDFacts",
    "StatsTable",
    "StatsTableRow",
    "SummaryReport",
    "SummaryRow",
    "SummaryTotals",
    "TraceProfile",
    "attributed_kernel_utilization",
    "backward_estimate",
    "backward_status",
    "build_profile",
    "capability_card",
    "build_stats_table",
    "build_summary_report",
    "capture_fingerprint",
    "compute_aggregation",
    "cost_report",
    "cost_tree",
    "device_peaks",
    "explain",
    "factcore",
    "flops_report",
    "grad_enabled_at_capture",
    "health_facts",
    "instrumented_rate",
    "instrumented_rate_calibration",
    "machine_balance",
    "mfu",
    "nonfinite_verdict",
    "normalize_health_facts",
    "padding_waste",
    "position_scaling_class",
    "registry_entry",
    "roofline",
    "time_clean_step",
    "which_do_i_use",
]


if TYPE_CHECKING:
    from torchlens.checks._audit import ParamHealthReport, param_health
    from torchlens.differential import DiffReport, diff
    from torchlens.ecosystem.compat import CompatWindow
    from torchlens.merged._presenter import MergeReport
    from torchlens.observability._native_profile import KernelProfileReport, kernel_profile
    from torchlens.validation.consolidated import InterventionValidationReport
    from torchlens.validation.diagnostics import ValidationDiagnostic, ValidationFailure
    from torchlens.validation.status import ValidationStatus

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ValidationStatus": ("torchlens.validation.status", "ValidationStatus"),
    "ValidationFailure": ("torchlens.validation.diagnostics", "ValidationFailure"),
    "ValidationDiagnostic": ("torchlens.validation.diagnostics", "ValidationDiagnostic"),
    "InterventionValidationReport": (
        "torchlens.validation.consolidated",
        "InterventionValidationReport",
    ),
    "kernel_profile": ("torchlens.observability._native_profile", "kernel_profile"),
    "KernelProfileReport": ("torchlens.observability._native_profile", "KernelProfileReport"),
    "MergeReport": ("torchlens.merged._presenter", "MergeReport"),
    "CompatWindow": ("torchlens.ecosystem.compat", "CompatWindow"),
    "diff": ("torchlens.differential", "diff"),
    "DiffReport": ("torchlens.differential", "DiffReport"),
    "param_health": ("torchlens.checks._audit", "param_health"),
    "ParamHealthReport": ("torchlens.checks._audit", "ParamHealthReport"),
}


def __getattr__(name: str) -> Any:
    """Resolve exports whose defining modules would import-cycle if loaded eagerly.

    Parameters
    ----------
    name:
        Attribute requested from this namespace.

    Returns
    -------
    Any
        The exported object.

    Raises
    ------
    AttributeError
        If ``name`` is not an export of this namespace.
    """

    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
