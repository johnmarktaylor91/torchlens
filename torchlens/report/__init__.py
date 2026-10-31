"""Reports that describe a completed trace: summary, cost, profile, FLOPs and health.

The debugging tools (first NaN or Inf, precision loss, lineage, input-shape guesses,
comparisons, determinism, gradient and parameter health) live in ``tl.debug``.
"""

from __future__ import annotations

from ..debug._audit import HealthReport
from ..debug._flops_vs_dispatch import FlopsCrossCheck
from ._cost_backward import (
    BackwardEstimate,
    BackwardStatus,
    backward_estimate,
    backward_status,
)
from ._cost_perf import (
    CleanStepTime,
    CostTierMeasurement,
    DevicePeaks,
    MachineBalance,
    MfuProvenance,
    MfuReport,
    OverheadReport,
    PeakRow,
    RooflineReport,
    RooflineRow,
    attributed_kernel_utilization,
    device_peaks,
    measure_clean_step,
    mfu,
    overhead,
    roofline,
)
from ._cost_scaling import (
    PaddingWaste,
    padding_waste,
)
from ._cost_tree import (
    CostTree,
    CostTreeRow,
    build_cost_tree as cost_tree,
)
from ._explain import ExplainReport
from ._factcore import (
    MemoryFacts,
    ParamFacts,
    PrunedParamFacts,
)
from ._flops_report import FlopsReport, SixNDFacts, flops
from ._health import (
    HealthFacts,
)
from ._profile import ProfileReport, profile
from ._registry import (
    CapabilityCard,
    CapabilityRow,
)
from ._stats_table import StatsTable, StatsTableRow
from ._summary_report import (
    CaptureFacts,
    SummaryReport,
    SummaryRow,
    SummaryTotals,
    summary,
)

__all__ = [
    "FlopsCrossCheck",
    "HealthReport",
    "BackwardEstimate",
    "BackwardStatus",
    "CapabilityCard",
    "CapabilityRow",
    "CaptureFacts",
    "CleanStepTime",
    "OverheadReport",
    "CostTierMeasurement",
    "CostTree",
    "CostTreeRow",
    "DevicePeaks",
    "ExplainReport",
    "FlopsReport",
    "HealthFacts",
    "MachineBalance",
    "MemoryFacts",
    "MfuProvenance",
    "MfuReport",
    "PaddingWaste",
    "ParamFacts",
    "PeakRow",
    "PrunedParamFacts",
    "RooflineReport",
    "RooflineRow",
    "SixNDFacts",
    "StatsTable",
    "StatsTableRow",
    "SummaryReport",
    "SummaryRow",
    "SummaryTotals",
    "ProfileReport",
    "attributed_kernel_utilization",
    "backward_estimate",
    "backward_status",
    "profile",
    "summary",
    "overhead",
    "cost_tree",
    "device_peaks",
    "flops",
    "mfu",
    "padding_waste",
    "roofline",
    "measure_clean_step",
]
