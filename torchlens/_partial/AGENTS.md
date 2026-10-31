# partial/ - Implementation Guide

Failed-capture recovery (`tl._partial`). A capture that fails mid-forward
raises with `exc.partial_log` attached; `from_failed_capture(exc)` wraps that
state as a `PartialTrace` for post-mortem inspection. `__all__` is exactly
`["PartialTrace", "from_failed_capture"]`.

## Surface

- `from_failed_capture(exception)` -> `PartialTrace` (also
  `PartialTrace.from_trace(trace, exception)`).
- `PartialTrace` is a THIN wrapper, not a `Trace` subclass: it holds the
  partially populated `trace`, the `original_exception`, and exposes
  `raw_layers` (tuple of raw `Op` records), `outcome` (the settled
  `CaptureOutcome`, see `docs/reference/capture_outcomes.md`),
  `first_nonfinite()`, `audit()` (-> `HealthReport`, same as
  `tl.debug.audit_trace`), `draw()`, and `show(method="graph"|"repr"|"html")`.

## Gotchas

- `partial.explain()` has a dedicated partial path (text +
  `capture_status="partial"` JSON) built on recorded evidence only; fields
  without evidence report `"unknown"`, never inferred.
- Failed partials are analysis-only: conversion/replay surfaces refuse
  through the capture-outcome gates (N-gates), not here.
- Exception text is folded through `safe_exception_str` — keep new fields
  string-only.

## Tests

`tests/test_capture_failure_reporting.py`, `tests/test_capture_outcome_gates.py`
(partial/outcome surfaces), `tests/test_report_explain.py` (partial explain).
