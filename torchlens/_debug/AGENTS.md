# debug/ - Implementation Guide

Power-user diagnostics for completed traces. Imported lazily as `tl._debug` and
deliberately NOT in the top-level `__all__`. Everything here is read-only analysis;
nothing mutates a trace or the capture pipeline.

## The report object (U42.2; DOCUMENTED-UNSTABLE)
- Every report-shaped result is a `tl._debug.DebugReport` (defined in
  `torchlens/_debug_report.py`, at the package root so `checks.ParamAudit` can
  subclass it without importing this package): `report_type`, `subject`,
  `status`, `to_dict()`, `to_pandas()`, and `str()` = a bounded card (lead line,
  at most five findings, exact remainder, exits). Results with a house-voice
  `__repr__` (`TraceAudit`, `DTypeRangeAudit`, `FindNanResult`, `ParamAudit`) keep
  printing it (`_prints_repr = True`; the card is `report.card()`).
- Subclasses state the lead through `_report_subject` / `_report_status` /
  `_report_findings` (optionally `_report_coverage`, `_report_severity_counts`,
  `_report_rows`). Do not add dataclass fields named `report_type`, `subject` or
  `status` (they are base properties).
- The tabular helpers (`hot_path`, `dead_neurons`, `compare`,
  `recompute_candidates`, `gradient_flow_audit`, `compare_params`) return a
  tabular `DebugReport` built by `table_report(...)`; `.to_pandas()` is the frame
  (attrs included). It never forwards DataFrame methods. The `*_rows` helpers stay
  the pandas-free core.
- Not reports: the context managers (`isolated_capture`, `preserved_rng_state`,
  `count_compiles`), `clone_input_tree`, `amp_scaled_gradients_hint` (a string),
  `first_divergence` (a dict or None), and `_grad_fn_walk.py` (VIZ-R).

## _nan.py
- `find_nan(model, x, **trace_kwargs)` captures fresh and localizes the first
  nonfinite value; `find_nan_in_trace(trace)` / `bisect_nan(trace)` work on an
  existing trace.
- Results are `FindNanResult` / `BisectNanResult`; unsaved ancestors become an
  explicit `uncertainty_zone`, never a silent guess.

## _precision.py
- `bisect_precision(model, input_args, input_kwargs=None, *, reference_dtype=fp64,
  rtol=None, atol=None, seed=0)` runs the same seeded forward at native precision and
  at the reference dtype (deep copies, forked RNG -- caller model/RNG untouched) and
  returns `BisectPrecisionResult`: first divergent op + full `PrecisionRow` table +
  disclosed skips. Per-op default tolerances derive from the NATIVE dtype
  (`rtol=eps**0.5`, `atol=eps*10`); stochastic ops (dropout et al.) are flagged, since
  RNG consumption is not comparable across dtypes. fp64-on-MPS refuses typed.
  DOCUMENTED-UNSTABLE spellings.

## _cost.py
- `hot_path(trace, by="flops")` ranks source lines by cost metric (a tabular
  `DebugReport`; `.to_pandas()` is the frame).
- `theoretical_op_bytes(op)` estimates input/parameter traffic from metadata only.

## _compile_counter.py
- `count_compiles()` is a context manager yielding `CompileCounts`; raises
  `CompileCountsUnavailableError` when Dynamo counters are not available.

## _graph_breaks.py
- `graph_breaks(model, x, **trace_kwargs)` returns a `GraphBreakReport` of
  `GraphBreak` records correlated back to trace op locations.
- Snapshots and restores model state around its probe run; typed failures are
  `GraphBreaksUnavailableError` / `GraphBreaksNormalizationError`.

## _audit.py
- `audit_trace(trace)` returns a `TraceAudit` of `AuditFinding`s; accepts a
  `PartialTrace` and routes it through `_audit_partial_trace()`.

## _graph.py
- `lineage(...)` -> `LineageResult`; `compare(...)` diffs two traces;
  `dead_neurons(trace, dim=..., threshold=...)` scans saved activations.

## _dtype_range.py
- `dtype_range_audit(...)` -> `DTypeRangeAudit`: overflow headroom, subnormal
  fractions, and narrowing casts per op.

## _gradients.py
- `gradient_flow_audit(...)` builds a gradient-magnitude table (a tabular
  `DebugReport`); its frame is an annotated empty frame with a message rather than
  raising when grads are absent.

## _infer_input_shape.py
- `infer_input_shape(model, **kwargs)` -> `InferInputShapeResult`; iterative
  probe-and-refine search driven by shape-error parsing and module priors.

## _recompute.py
- `recompute_candidates(trace, budget_gb=...)` ranks ops worth recomputing vs saving.

## _common.py
- Shared private helpers (`_ordered_ops`, `_resolve_op`, `_require_pandas`, ...).

## Gotchas
- pandas is required lazily per call (`_require_pandas()`), not at import.
- Several helpers (`find_nan`, `graph_breaks`, `infer_input_shape`) run fresh
  forwards/captures; they are not safe on models with destructive side effects.
- Keep result objects typed dataclasses that subclass `DebugReport`; callers branch
  on fields, not messages.
