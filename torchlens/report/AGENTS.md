# report/ - Implementation Guide

Reporting helpers over finished captures and observer metadata (`tl.report`).

## Files

- `_explain.py` — `explain(log, ...)`: the prose explainer for a Trace / partial
  trace (used as `tl.report.explain(log)` in the root docs' Common Patterns).
  `max_tokens=N` (DOCUMENTED-UNSTABLE) budget-prunes whole sections
  low-value-first with a disclosed `Truncation` section; the `Capture status`
  honesty facts and partial failure evidence are never dropped.
- `_agent_json.py` — `build_agent_json(log, max_ops=None)`: the
  `torchlens.agent_trace.v1` self-describing dump behind
  `Trace.to_agent_json()` (DOCUMENTED-UNSTABLE). Carries the same
  capture-verification facts as `explain()`; payloads never inlined;
  truncation disclosed, never silent.
- `_profile.py` — `TraceProfile` + `build_profile(...)`: tabular resource
  profile at `level="op" | "module" | "call"` (durations, FLOPs, params,
  honesty rows, capture-verification banner, call tree). `forward_peak_memory`
  is NOT a profile column — it lives on the Trace itself. pandas is an optional
  dependency resolved lazily via `_require_pandas()` — never import it at module top.
- `__init__.py` — public surface (`explain`, `TraceProfile`, `build_profile`,
  `log_value`); `log_value(name, value)` records observer values through `_state`.

## Gotchas

- Honesty rows must reflect capture verification state faithfully — a rescued or
  ceilinged capture (`capture_verified=False`) must stay visible in profile/explain
  output; never present an unverified capture as clean.
- `forward_peak_memory` is a runtime measurement that legitimately reads `0` on the
  default CPU path — never present it as a portable fact (root `AGENTS.md` rule).
