# Which report surface do I use?

<!-- GENERATED from torchlens/report/_registry.py::which_do_i_use();
     edit the registry, then regenerate. The lockstep test
     tests/test_report_family_registry.py pins this file to the
     generator output. -->

Every row is one question. Every spelling below executes in CI against a
real fixture in its declared state.

| surface | one question it answers | subject | register | cost |
|---|---|---|---|---|
| `trace.summary()` | Orient me: what model ran, its layers, parameters, and totals. | MODEL | SHAPE | metadata_only |
| `tl.report.explain(trace)` | Narrate this capture: what ran, its health states, and anything unusual. | CAPTURE | NARRATIVE | metadata_only |
| `trace.profile(sort_by=..., top_k=...)` | Rank me: which ops/modules cost the most time, FLOPs, or memory. | RUN-RESOURCES | RANK | metadata_only |
| `tl.report.cost_tree(trace)` | Show compute as the module tree with exact self/subtree conservation. | RUN-RESOURCES | SHAPE | metadata_only |
| `tl.report.flops_report(trace_or_model, ...)` | The one-call paper number: analytic forward FLOPs with coverage and params. | RUN-RESOURCES | SHAPE | metadata_only |
| `trace.stats_table()` | Observations of ONE captured batch per site: mean/std/zero/NaN fractions. | RUN-VALUES | SHAPE | payload_scan |
| `trace.audit()` | Judge me: findings with severities, checks run, and skips disclosed. | RUN-HEALTH | VERDICT | payload_scan |
| `tl.report.health_facts(trace)` | The three-state nonfinite record (found / clean / not-checked) with basis. | RUN-HEALTH | SHAPE | payload_scan |
| `trace.bill_of_materials()` | What does THIS object retain: payload bytes now, counts, annotations. | CAPTURE | SHAPE | metadata_only |
| `trace.to_agent_json()` | The machine navigation map: op rows, edges, hierarchy, guide. | CAPTURE | ADDRESS | metadata_only |
| `tl.report.backward_status(trace)` | The three-state executed-backward answer (0 / unknown / observed). | RUN-RESOURCES | SHAPE | metadata_only |
| `tl.report.backward_estimate(trace)` | The HYPOTHETICAL training-backward counterfactual behind its named door. | RUN-RESOURCES | SHAPE | metadata_only |
| `tl.report.roofline(trace, ridge_intensity=...)` | Theoretical intensity/work map; bound verdicts are hypotheses. | RUN-RESOURCES | SHAPE | metadata_only |
| `tl.report.instrumented_rate(trace)` | The boxed CPU-only FLOPs-per-instrumented-second triage diagnostic. | RUN-RESOURCES | RANK | metadata_only |
| `tl.report.cost_report(model, x)` | Measured capture cost on YOUR model and host (never an estimate). | RUN-RESOURCES | SHAPE | new_forward |
| `tl.utils.doctor()` | Is my environment healthy: versions, capabilities, degradations. | ENVIRONMENT | VERDICT | env_probe |

## Refusal conditions

- `summary`: refuses when the capture failed before finalization (use explain on the partial).
- `explain`: refuses when never (degrades to a partial-capture diagnosis).
- `profile`: refuses when the capture failed before finalization.
- `cost_tree`: refuses when the capture failed before finalization.
- `flops_report`: refuses when the trace door is passed model-door inputs.
- `stats_table`: refuses when payloads were not retained (typed row states, never fabrication).
- `audit`: refuses when never (partial captures get the degraded audit).
- `health_facts`: refuses when never (an underivable basis is the NOT-CHECKED shape).
- `bill_of_materials`: refuses when the capture failed before finalization.
- `agent_json`: refuses when the capture failed before finalization.
- `backward_status`: refuses when never.
- `backward_estimate`: refuses when never (always labeled hypothetical; never an actual-cost slot).
- `roofline`: refuses when never (uncoverable ops land in the by-reason split).
- `instrumented_rate`: refuses when every timed op executed on CUDA (host time is not a CUDA rate).
- `cost_report`: refuses when an unknown tier is requested.
- `doctor`: refuses when never.

## Honesty invariants per surface

- `summary`: conserves_totals, mints_no_facts
- `explain`: mints_no_facts, truncation_disclosed
- `profile`: truncation_disclosed, instrumented_time_labeled, conserves_totals
- `cost_tree`: conserves_totals, truncation_disclosed
- `flops_report`: conserves_totals, mints_no_facts
- `stats_table`: mints_no_facts, truncation_disclosed
- `audit`: checks_and_skips_complete
- `health_facts`: mints_no_facts
- `bill_of_materials`: mints_no_facts
- `agent_json`: mints_no_facts, stable_addressing, truncation_disclosed
- `backward_status`: mints_no_facts
- `backward_estimate`: mints_no_facts
- `roofline`: mints_no_facts, truncation_disclosed
- `instrumented_rate`: instrumented_time_labeled
- `cost_report`: mints_no_facts
- `doctor`: checks_and_skips_complete
