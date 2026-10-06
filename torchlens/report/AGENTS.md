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
- `_factcore.py` / `_compute_truth.py` / `_health.py` — the C02 numbers
  substrate: FactCore (params/compute/memory/counts/identity), the canonical
  compute aggregation (the ONE reader of raw per-op compute fields), and the
  three-state health facts. Every summary/profile number is a projection.
- `_summary_report.py` — `SummaryReport(str)`: the detached typed result
  `summary()` returns (rows/totals/capture + the F08 result API:
  `render`/`print`/`details`/`to_pandas`/`to_markdown`/`to_html` + scalar
  raw fields). The str payload is canonical byte-stable ASCII.
- `_summary_ladder.py` — the F08 auto view ladder (coalesced hybrid ->
  strictly folded module tree -> descending depth -> protected elision) under
  the derived 48-body-row budget, with the identity-partition invariant
  (`owned_ops`/`owned_params` disjoint-and-total across rows + root remainder).
- `_summary_config.py` — the summary option grammar: column registry, view
  bundles, the removed-spelling tables (`REMOVED_SUMMARY_OPTIONS`,
  `REMOVED_SUMMARY_LEVELS`: each legacy spelling refuses typed naming its
  successor), typed
  `summary_option_conflict`/`summary_option_invalid`/`summary_level_invalid`.
- `_summary_charset.py` — the charset contract: 1:1 glyph table, byte-exact
  `degrade()`, fail-toward-ASCII `detect_style()` ladder
  (`TORCHLENS_SUMMARY_STYLE` env rung registered in `utils/env_flags.py`).
- `_summary_render.py` — the hairline-table renderer (pure over typed data;
  five-subject footer; honesty banner consumed from `_capture_honesty`'s ONE
  chokepoint; no ANSI/OSC-8 byte can appear in any returned string).
- `_summary_html.py` — dependency-free escaped HTML fragment (sticky header,
  dark-mode-safe, zero JS, `data-*` interactivity plumbing).
- `_summary_result.py` — the rebuilt-summary assembler both entry doors share
  (`build_rebuilt_summary`), fact collection, filtering + coverage
  disclosure, and the projection implementations.

## Gotchas

- Honesty rows must reflect capture verification state faithfully — a rescued or
  ceilinged capture (`capture_verified=False`) must stay visible in profile/explain
  output; never present an unverified capture as clean.
- `forward_peak_memory` is a runtime measurement that legitimately reads `0` on the
  default CPU path — never present it as a portable fact (root rule, `docs/agent-reference/known-gotchas.md`).
