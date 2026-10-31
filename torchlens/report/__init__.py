"""Reports and diagnostics for completed TorchLens traces.

Debugging numerical problems
----------------------------
Each tool below answers one symptom; all of them read a completed trace (or rerun the
model once) and return a typed report.

- First NaN or Inf in the forward pass: ``tl.report.bisect_nan(trace)`` finds the first
  non-finite op of a trace, ``tl.report.find_nan(model, inputs)`` captures and finds it,
  and ``tl.report.nonfinite_origin(trace)`` names where it started and why.
- First NaN or Inf in the backward pass: ``tl.report.bisect_nan_backward(...)``.
- Precision loss (fp16, bf16, TF32): ``tl.report.bisect_precision(...)`` finds the first
  op whose low-precision output drifts; ``tl.report.dtype_range(trace)`` flags overflow
  headroom and subnormal outputs.
- Where a tensor came from: ``tl.report.lineage(trace, ...)``.
- Guessing the input shape a model expects: ``tl.report.guess_input_shape(model)``.
- Two traces or two models differ: ``tl.report.compare(...)``,
  ``tl.report.first_divergence(...)`` and ``tl.report.compare_params(...)``.
- Runs that do not repeat: ``tl.report.check_determinism(model, inputs)``.
- Dead units, vanishing or exploding gradients, bad parameters:
  ``tl.report.dead_neurons(trace)``, ``tl.report.grad_health(trace)`` and
  ``tl.report.param_health(model)``.
- Everything a trace can check about itself at once: ``trace.health``.
"""

from __future__ import annotations

from .._diagnostics import (
    DeterminismReport,
    DTypeRangeReport,
    FlopsCrossCheck,
    HealthReport,
    NonfiniteReport,
    ParamHealthReport,
    Report,
    check_determinism,
    dtype_range,
    grad_health,
    nonfinite_origin,
    param_health,
)
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
from ._explain import explain
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
    "DTypeRangeReport",
    "DeterminismReport",
    "FlopsCrossCheck",
    "HealthReport",
    "NonfiniteReport",
    "ParamHealthReport",
    "Report",
    "check_determinism",
    "dtype_range",
    "grad_health",
    "nonfinite_origin",
    "param_health",
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
    "explain",
    "flops",
    "mfu",
    "padding_waste",
    "roofline",
    "measure_clean_step",
]
