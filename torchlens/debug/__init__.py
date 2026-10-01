"""Power-user debugging helpers for completed TorchLens traces."""

from __future__ import annotations

from ._audit import AuditFinding as AuditFinding, TraceAudit, audit_trace
from ._compile_counter import (
    CompileCounts as CompileCounts,
    CompileCountsUnavailableError,
    count_compiles,
)
from ._cost import hot_path, hot_path_rows
from ._determinism import DeterminismReport, check_determinism
from ._dtype_range import DTypeRangeAudit, dtype_range_audit
from ._first_bad import FirstBadThing, amp_scaled_gradients_hint
from ._flops_vs_dispatch import FlopsCrossCheck, flops_vs_dispatch
from ._grad_fn_walk import (
    GradFnNode,
    GradFnSketch,
    GradFnWalkError,
    sketch_grad_fn,
    walk_grad_fn,
)
from ._gradients import gradient_flow_audit, gradient_flow_audit_rows
from ._graph import (
    LineageResult,
    compare,
    compare_rows,
    dead_neurons,
    dead_neurons_rows,
    first_divergence,
    lineage,
)
from ._graph_breaks import (
    GraphBreak as GraphBreak,
    GraphBreakReport as GraphBreakReport,
    GraphBreaksNormalizationError,
    GraphBreaksUnavailableError,
    graph_breaks,
)
from ._infer_input_shape import InferInputShapeResult, infer_input_shape
from ._nan import BisectNanResult, FindNanResult, NanReport, bisect_nan, find_nan, nan_report
from ._nan_backward import BackwardNanResult, BackwardTransition, bisect_nan_backward
from ._params import compare_params
from ._params_audit import ParamAudit, audit_params
from ._precision import BisectPrecisionResult, PrecisionRow as PrecisionRow, bisect_precision
from ._recompute import recompute_candidates
from ._rerun import clone_input_tree, isolated_capture, preserved_rng_state

__all__ = [
    "BackwardNanResult",
    "BackwardTransition",
    "BisectNanResult",
    "BisectPrecisionResult",
    "DeterminismReport",
    "FirstBadThing",
    "CompileCountsUnavailableError",
    "DTypeRangeAudit",
    "FindNanResult",
    "FlopsCrossCheck",
    "GradFnNode",
    "GradFnSketch",
    "GradFnWalkError",
    "GraphBreaksNormalizationError",
    "GraphBreaksUnavailableError",
    "InferInputShapeResult",
    "LineageResult",
    "NanReport",
    "ParamAudit",
    "TraceAudit",
    "amp_scaled_gradients_hint",
    "audit_params",
    "audit_trace",
    "bisect_nan",
    "bisect_nan_backward",
    "bisect_precision",
    "check_determinism",
    "find_nan",
    "compare",
    "compare_params",
    "compare_rows",
    "clone_input_tree",
    "count_compiles",
    "dead_neurons",
    "dead_neurons_rows",
    "dtype_range_audit",
    "first_divergence",
    "flops_vs_dispatch",
    "isolated_capture",
    "preserved_rng_state",
    "walk_grad_fn",
    "gradient_flow_audit",
    "gradient_flow_audit_rows",
    "graph_breaks",
    "hot_path",
    "hot_path_rows",
    "infer_input_shape",
    "lineage",
    "nan_report",
    "recompute_candidates",
    "sketch_grad_fn",
]
