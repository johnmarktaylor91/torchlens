"""Reports that describe a completed trace: summary, cost, profile, FLOPs and health.

The debugging tools (first NaN or Inf, precision loss, lineage, input-shape guesses,
comparisons, determinism, gradient and parameter health) live in ``tl.debug``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ..debug._audit import HealthReport
from ..debug._compile_counter import count_compiles
from ..debug._cost import hot_path, hot_path_rows
from ..debug._first_bad import FirstBadThing, amp_scaled_grads_hint
from ..debug._flops_vs_dispatch import FlopsCrossCheck, flops_vs_dispatch
from ..debug._grad_fn_walk import GradFnNode, GradFnSketch, sketch_grad_fn, walk_grad_fn
from ..debug._graph import LineageReport, compare, first_divergence, lineage
from ..debug._graph_breaks import graph_breaks
from ..debug._infer_input_shape import GuessInputShapeReport, guess_input_shape
from ..debug._nan import BisectNanReport, FindNanReport, bisect_nan, find_nan
from ..debug._nan_backward import BackwardNanReport, BackwardTransition, bisect_nan_backward
from ..debug._params import compare_params
from ..debug._precision import BisectPrecisionReport, bisect_precision
from ..debug._recompute import recompute_candidates
from ..intervention._topology.topology import compare_topology
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
    "isolated_capture",
    "FindNanReport",
    "compare",
    "compare_params",
    "compare_topology",
    "find_nan",
    "FlopsCrossCheck",
    "HealthReport",
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
    "amp_scaled_grads_hint",
    "attributed_kernel_utilization",
    "backward_estimate",
    "backward_status",
    "profile",
    "summary",
    "overhead",
    "cost_tree",
    "device_peaks",
    "flops_vs_dispatch",
    "flops",
    "mfu",
    "padding_waste",
    "roofline",
    "measure_clean_step",
    "BackwardNanReport",
    "BackwardTransition",
    "bisect_nan",
    "bisect_nan_backward",
    "bisect_precision",
    "BisectNanReport",
    "BisectPrecisionReport",
    "count_compiles",
    "first_divergence",
    "FirstBadThing",
    "GradFnNode",
    "GradFnSketch",
    "graph_breaks",
    "hot_path",
    "hot_path_rows",
    "guess_input_shape",
    "GuessInputShapeReport",
    "lineage",
    "LineageReport",
    "recompute_candidates",
    "sketch_grad_fn",
    "walk_grad_fn",
]


if TYPE_CHECKING:
    from torchlens._ecosystem.compat import CompatWindow
    from torchlens._merged._presenter import MergeReport
    from torchlens._observability._native_profile import KernelProfileReport, kernel_profile
    from torchlens.differential import DiffReport, diff
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
    "kernel_profile": ("torchlens._observability._native_profile", "kernel_profile"),
    "KernelProfileReport": ("torchlens._observability._native_profile", "KernelProfileReport"),
    "MergeReport": ("torchlens._merged._presenter", "MergeReport"),
    "CompatWindow": ("torchlens._ecosystem.compat", "CompatWindow"),
    "diff": ("torchlens.differential", "diff"),
    "DiffReport": ("torchlens.differential", "DiffReport"),
    "isolated_capture": ("torchlens.debug._rerun", "isolated_capture"),
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
