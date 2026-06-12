# jax-tinygrad-build — autonomous build loop state

run_name: jax-tinygrad-build
state: RUNNING
started: 2026-06-12 ~20:20 (overnight autonomous mandate, JMT)
plan: .research/jax-tinygrad-sprint_PLAN.md (v13 BUILD-READY; round-13 critic checklist = build checklist)
branch: capture-unification
max_rounds: 40 (backstop; anti-flail 3-same-issue governs)

## AUTONOMOUS DECISION LOG (JMT reviews later — append EVERY gate call here)
| # | When | Decision | Rationale |
|---|---|---|---|
| D1 | 20:20 | Backend sprint overlaps P6 measurement (benchmark ~10h; serializing blows the morning deadline). Overlapping rounds FORBID pytest runs; perf wrap-up waits for benchmark end. | JMT morning goal > measurement purity; measurement conditions already documented as loaded-host. |

## Stop criterion
Committed phases (M0.1a artifacts, S0.J-slim, S0.G) + gate-issued phases through
M1 J4 green per the plan's round-13 checklist — OR honest defer decisions logged.
At ~07:30 local: HARD CHECKPOINT — whatever is green: README edits (Fable), merge
capture-unification -> main, VERSION GUARD (no major bump: grep BREAKING/! in
commits), push, summary + iMessage JMT.

## Wake-up case routing
| Signal | Action |
|---|---|
| codex alive | yield |
| CODEX_DONE | verify (git log, report tail, gates per round type), append round row, dispatch next per plan sequence |
| CODEX_FAILED quota | fallback: Agent(general-purpose, opus) with adapted spec; if both blocked: state BLOCKED + iMessage + ScheduleWakeup |
| verification FAIL | focused fixup spec, ONE retry per failure class, then RESIDUAL/defer |
| same issue 3 rounds | RESIDUAL; log; continue or defer phase |
| perf measurement (pid 1133032) done | verify gate JSON; dispatch perf WRAP-UP round (tests allowed then); on its completion mark perf-build_STATE DONE + iMessage headline numbers |
| 07:30 checkpoint | execute stop-criterion endgame regardless of remaining rounds |

## Phase sequence (from plan §5)
M0.1a artifacts 1-5 (docs-only rounds) || S0.G discovery (parallel lane) ->
S0.J-slim (2-3) -> IMPACT GATE (Fable decides under D-mandate, logs D#) ->
M0.1b parity gates -> M0.2 substrate -> M0.3 -> M1 J1-J4 -> [M2 if S0.G go].
Per-round gates: predicate suite + smoke (SUSPENDED during benchmark overlap —
substituted with ruff/mypy + targeted-non-CPU checks; full gates re-run at first
post-benchmark round) + torch parity goldens once built.

## M0.1a artifact checklist
- [x] 1. Invariant contract: `.research/backend-substrate/artifact-1-invariant-contract.md`
- [x] 2. Public-surface + kwarg + backward-surface matrices: `.research/backend-substrate/artifact-2-public-surface-kwarg-backward-matrices.md`
- [x] 3. Serialization contract with three version axes: `.research/backend-substrate/artifact-3-serialization-contract.md`
- [x] 4. BackendSpec registry contract as executable migration map: `.research/backend-substrate/artifact-4-backendspec-registry-migration-map.md`
- [x] 5. Docs/glossary change list: `.research/backend-substrate/artifact-5-docs-glossary-change-list.md`

## Iteration log
| Round | Lane | Phase | Commit | Result | Next |
|---|---|---|---|---|---|
| 1 | tinygrad | S0.G discovery | this commit | PASS: pinned `tinygrad==0.13.0`; added runnable probes for interception/realization, mutation identity, autograd lifecycle, TinyJit/GC, and initial payload non-interference. | Continue S0.G with deeper scheduler/JIT capture, grad-payload, stale-lineage, and device-copy probes. |
| 1 | M0.1a | artifact 1 - invariant contract | this commit | Wrote docs-only backend invariant contract; no tests/ruff per benchmark constraint. | artifact 2 - public-surface + kwarg + backward-surface matrices |
| 2 | M0.1a | artifact 2 - public-surface + kwarg + backward-surface matrices | this commit | Wrote docs-only public API, trace kwarg, and backward-surface executable matrices; no tests/ruff per benchmark constraint. | artifact 3 - serialization contract with three version axes |
| 3 | M0.1a | artifact 3 - serialization contract with three version axes | this commit | Wrote docs-only serialization contract separating on-disk family, manifest schema, and pickled object-state axes; no tests/ruff per benchmark constraint. | artifact 4 - BackendSpec registry executable migration map |
| 2 | tinygrad | S0.G discovery | this commit | PASS: added scheduler/JIT, explicit-realize, alias-mutation, explicit-gradient, and realized payload snapshot probes under `tinygrad==0.13.0`; spike ruff passed. | Continue S0.G with TinyJit captured-execution payload reads, safe device variation, and final source-of-truth/payload decisions. |
| 4 | M0.1a | artifact 4 - BackendSpec registry executable migration map | this commit | Wrote docs-only BackendSpec registry contract and migration map for backend literals, MLX branches, validation dispatch, serialization, and public accessors; no tests/ruff per benchmark constraint. | artifact 5 - docs/glossary change list |
| 3 | tinygrad | S0.G discovery | this commit | PASS: added captured TinyJit runner/payload probes under `tinygrad==0.13.0`; finalized UOp pre-realization source-of-truth lean and backend-conditional payload capability decision; spike ruff passed. | Optional final S0.G round only for safe non-default device coverage and import-time capability probe text. |
| 1 | jax | S0.J-slim feasibility | this commit | PASS: pinned CPU `jax==0.10.1`/`jaxlib==0.10.1`; added asserted closed-jaxpr interpreter, replay/perturbation validation, nested/effect rejection probes, 16-case corpus measurement at 13/16 accepted, and toy/MLP overhead measurement. | Continue S0.J with random/CNN nested-call decision, corpus expansion toward 20-25 cases, and import-time compatibility probe sketch. |
| 2 | jax | S0.J-slim safe inlining | this commit | PASS: added safe pure-call inlining prototype for allowlisted library `jit` helpers and library `custom_jvp_call` forward paths; conv/ReLU and explicit-key random now capture; expanded corpus accepted 23/23; replay/perturbation proved on an inlined equation; scan/cond/while/user custom_jvp/custom_vjp remain rejected. | S0.J can close or spend final round on import-time compatibility probe and versioned allowlist text before impact gate. |
| 5 | M0.1a | artifact 5 - docs/glossary change list | this commit | Wrote docs-only glossary/docs/notebook merge-gate checklist for backend public surfaces; no tests/ruff per benchmark constraint. | all 5 done; next per plan is S0.J-slim / impact-gate preparation |
