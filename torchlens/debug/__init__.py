"""Debugging tools for completed TorchLens traces, grouped by the symptom they answer.

- First NaN or Inf in the forward pass: ``tl.debug.bisect_nan(trace)`` finds the first
  non-finite op of a trace, ``tl.debug.find_nan(model, inputs)`` captures and finds it,
  and ``tl.debug.nonfinite_origin(trace)`` names where it started and why.
- First NaN or Inf in the backward pass: ``tl.debug.bisect_nan_backward(trace)``.
- Precision loss (fp16, bf16, TF32): ``tl.debug.bisect_precision(model, inputs)`` finds
  the first op whose low-precision output drifts; ``tl.debug.dtype_range(trace)`` flags
  overflow headroom and subnormal outputs.
- Where a tensor came from: ``tl.debug.lineage(trace, ...)``.
- The input shape a model expects: ``tl.debug.guess_input_shape(model)``.
- Two runs or two models differ: ``tl.debug.compare(...)``,
  ``tl.debug.first_divergence(...)`` and ``tl.debug.compare_params(...)``.
- Runs that do not repeat: ``tl.debug.check_determinism(model, inputs)``.
- Dead neurons, gradient flow and parameter health: ``tl.debug.dead_neurons(trace)``,
  ``tl.debug.grad_health(trace)`` and ``tl.debug.param_health(model)``; mixed-precision
  gradients that go non-finite under a grad scaler: ``tl.debug.amp_scaled_grads_hint()``.
- Autograd graph structure: ``tl.debug.walk_grad_fn(...)`` and
  ``tl.debug.sketch_grad_fn(...)``.
- Compilation and speed: ``tl.debug.count_compiles(...)``, ``tl.debug.graph_breaks(...)``,
  ``tl.debug.hot_path(trace)`` and ``tl.debug.recompute_candidates(trace)``.
- Re-running a capture in isolation: ``tl.debug.isolated_capture(...)``.

Whole-trace health is ``trace.health``; the FLOPs cross-check is
``tl.report.flops_vs_dispatch``. Every report-shaped result is a
:class:`tl.report.Report` (DOCUMENTED-UNSTABLE): ``report_type``/``subject``/``status``,
``to_dict()``, ``to_pandas()`` and a bounded printed card.
"""

from __future__ import annotations

from .._diagnostics import (
    DeterminismReport,
    DTypeRangeReport,
    NonfiniteReport,
    ParamHealthReport,
    check_determinism,
    dtype_range,
    grad_health,
    nonfinite_origin,
    param_health,
)
from ._audit import AuditFinding as AuditFinding
from ._compile_counter import CompileCounts as CompileCounts, count_compiles
from ._cost import hot_path, hot_path_rows
from ._first_bad import FirstBadThing, amp_scaled_grads_hint
from ._grad_fn_walk import (
    GradFnNode,
    GradFnSketch,
    sketch_grad_fn,
    walk_grad_fn,
)
from ._graph import (
    LineageReport,
    compare,
    dead_neurons,
    first_divergence,
    lineage,
)
from ._graph_breaks import (
    GraphBreak as GraphBreak,
    GraphBreakReport as GraphBreakReport,
    graph_breaks,
)
from ._infer_input_shape import GuessInputShapeReport, guess_input_shape
from ._nan import BisectNanReport, FindNanReport, bisect_nan, find_nan
from ._nan_backward import BackwardNanReport, BackwardTransition, bisect_nan_backward
from ._params import compare_params
from ._precision import BisectPrecisionReport, PrecisionRow as PrecisionRow, bisect_precision
from ._recompute import recompute_candidates
from ._rerun import isolated_capture

__all__ = [
    "DTypeRangeReport",
    "DeterminismReport",
    "NonfiniteReport",
    "ParamHealthReport",
    "check_determinism",
    "dtype_range",
    "grad_health",
    "nonfinite_origin",
    "param_health",
    "BackwardNanReport",
    "BackwardTransition",
    "BisectNanReport",
    "BisectPrecisionReport",
    "FirstBadThing",
    "GradFnNode",
    "GradFnSketch",
    "GuessInputShapeReport",
    "LineageReport",
    "FindNanReport",
    "amp_scaled_grads_hint",
    "bisect_nan",
    "bisect_nan_backward",
    "bisect_precision",
    "compare",
    "compare_params",
    "count_compiles",
    "dead_neurons",
    "find_nan",
    "first_divergence",
    "walk_grad_fn",
    "graph_breaks",
    "hot_path",
    "hot_path_rows",
    "guess_input_shape",
    "isolated_capture",
    "lineage",
    "recompute_candidates",
    "sketch_grad_fn",
]
