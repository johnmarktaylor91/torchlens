"""Power-user debugging helpers for completed TorchLens traces."""

from __future__ import annotations

from ._cost import hot_path
from ._audit import AuditFinding, TraceAudit, audit_trace
from ._graph import LineageResult, compare, dead_neurons, lineage
from ._gradients import gradient_flow_audit
from ._infer_input_shape import InferInputShapeResult, infer_input_shape
from ._nan import BisectNanResult, FindNanResult, bisect_nan, find_nan
from ._recompute import recompute_candidates

__all__ = [
    "BisectNanResult",
    "AuditFinding",
    "FindNanResult",
    "InferInputShapeResult",
    "LineageResult",
    "TraceAudit",
    "audit_trace",
    "bisect_nan",
    "find_nan",
    "compare",
    "dead_neurons",
    "gradient_flow_audit",
    "hot_path",
    "infer_input_shape",
    "lineage",
    "recompute_candidates",
]
