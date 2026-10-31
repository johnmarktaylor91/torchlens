"""Power-user debugging helpers for completed TorchLens traces.

Every report-shaped result is a :class:`torchlens.report.Report` (DOCUMENTED-UNSTABLE):
``report_type``/``subject``/``status``, ``to_dict()``, ``to_pandas()`` and a bounded
printed card.
"""

from __future__ import annotations

from ._audit import AuditFinding as AuditFinding
from ._compile_counter import CompileCounts as CompileCounts, count_compiles
from ._cost import hot_path, hot_path_rows
from ._first_bad import FirstBadThing, amp_scaled_gradients_hint
from ._flops_vs_dispatch import flops_vs_dispatch
from ._grad_fn_walk import (
    GradFnNode,
    GradFnSketch,
    sketch_grad_fn,
    walk_grad_fn,
)
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
    graph_breaks,
)
from ._infer_input_shape import InferInputShapeResult, infer_input_shape
from ._nan import BisectNanResult, FindNanResult, bisect_nan, find_nan
from ._nan_backward import BackwardNanResult, BackwardTransition, bisect_nan_backward
from ._params import compare_params
from ._precision import BisectPrecisionResult, PrecisionRow as PrecisionRow, bisect_precision
from ._recompute import recompute_candidates
from ._rerun import clone_input_tree, isolated_capture, preserved_rng_state

__all__ = [
    "BackwardNanResult",
    "BackwardTransition",
    "BisectNanResult",
    "BisectPrecisionResult",
    "FirstBadThing",
    "FindNanResult",
    "GradFnNode",
    "GradFnSketch",
    "InferInputShapeResult",
    "LineageResult",
    "amp_scaled_gradients_hint",
    "bisect_nan",
    "bisect_nan_backward",
    "bisect_precision",
    "find_nan",
    "compare",
    "compare_params",
    "compare_rows",
    "clone_input_tree",
    "count_compiles",
    "dead_neurons",
    "dead_neurons_rows",
    "first_divergence",
    "flops_vs_dispatch",
    "isolated_capture",
    "preserved_rng_state",
    "walk_grad_fn",
    "graph_breaks",
    "hot_path",
    "hot_path_rows",
    "infer_input_shape",
    "lineage",
    "recompute_candidates",
    "sketch_grad_fn",
]
