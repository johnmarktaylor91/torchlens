"""Power-user debugging helpers for completed TorchLens traces."""

from __future__ import annotations

from ._audit import AuditFinding, TraceAudit, audit_trace
from ._compile_counter import (
    CompileCounts,
    CompileCountsUnavailableError,
    count_compiles,
)
from ._cost import hot_path, hot_path_rows
from ._dtype_range import DTypeRangeAudit, dtype_range_audit
from ._gradients import gradient_flow_audit, gradient_flow_audit_rows
from ._graph import LineageResult, compare, compare_rows, dead_neurons, dead_neurons_rows, lineage
from ._graph_breaks import (
    GraphBreak,
    GraphBreakReport,
    GraphBreaksNormalizationError,
    GraphBreaksUnavailableError,
    graph_breaks,
)
from ._infer_input_shape import InferInputShapeResult, infer_input_shape
from ._nan import BisectNanResult, FindNanResult, bisect_nan, find_nan
from ._params import compare_params
from ._precision import BisectPrecisionResult, PrecisionRow, bisect_precision
from ._recompute import recompute_candidates

__all__ = [
    "BisectNanResult",
    "BisectPrecisionResult",
    "PrecisionRow",
    "AuditFinding",
    "CompileCounts",
    "CompileCountsUnavailableError",
    "DTypeRangeAudit",
    "FindNanResult",
    "GraphBreak",
    "GraphBreakReport",
    "GraphBreaksNormalizationError",
    "GraphBreaksUnavailableError",
    "InferInputShapeResult",
    "LineageResult",
    "TraceAudit",
    "audit_trace",
    "bisect_nan",
    "bisect_precision",
    "find_nan",
    "compare",
    "compare_params",
    "compare_rows",
    "count_compiles",
    "dead_neurons",
    "dead_neurons_rows",
    "dtype_range_audit",
    "gradient_flow_audit",
    "gradient_flow_audit_rows",
    "graph_breaks",
    "hot_path",
    "hot_path_rows",
    "infer_input_shape",
    "lineage",
    "recompute_candidates",
]
