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

## M0.2 substrate checklist
- [x] 1. BackendSpec registry/API slice: typed registry, torch+MLX default specs, explicit `backend=` resolution, mismatch/ambiguity tests, fake backend public trace/validate harness.
- [x] 2. Full registry migration-map cutover: torch capture body fully owned by `TorchSpec.capture_trace`, generated hard-coded-branch audit, MLX canonical unsupported conversion complete.
- [x] 3. Neutral data/accessor fields: `dtype_ref`, `device_ref`, `backend_address`, resolver status, `Trace.param_source`, and object-state/default-fill decisions.
- [x] 4. Serialization schema v2: backend/runtime manifest fields, nullable torch fields, v1 fixture compatibility, audit-only non-torch payload policy.
- [x] 5. Fake backend end-to-end acceptance: trace object, save/load round-trip, accessors, invariant registry split with non-torch corruption fixtures.
- [x] 6. Docs/glossary/notebook lockstep for each newly reachable public surface beyond the registry/API slice.

## M0 status
M0 COMPLETE: M0.1a artifacts, M0.1b parity gates, M0.2 substrate items, and M0.3
docs/glossary/notebook lockstep are complete.

## JAX M1 status
JAX M1 COMPLETE: J1 jaxpr-first capture, J2 validation tripwire, J3 derived
gradients, and J4 hardening/docs/spec amendment are complete. JAX remains a
functional preview with true backward capture and T1 per-op gradients reserved for
research.

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
| D2 | 19:36 | S0.G discovery CLOSED at 3 rounds: M2 lean accepted = GO with backend-conditional payload gate (full-save only where sanctioned copies proven; audit-only fallback). Final M2 funding decision at impact gate. | Probes passed all 5 deliverables; TinyJit observable via engine.jit.run_linear; pin tinygrad==0.13.0. |
| 1 | jax | S0.J-slim feasibility | this commit | PASS: pinned CPU `jax==0.10.1`/`jaxlib==0.10.1`; added asserted closed-jaxpr interpreter, replay/perturbation validation, nested/effect rejection probes, 16-case corpus measurement at 13/16 accepted, and toy/MLP overhead measurement. | Continue S0.J with random/CNN nested-call decision, corpus expansion toward 20-25 cases, and import-time compatibility probe sketch. |
| 2 | jax | S0.J-slim safe inlining | this commit | PASS: added safe pure-call inlining prototype for allowlisted library `jit` helpers and library `custom_jvp_call` forward paths; conv/ReLU and explicit-key random now capture; expanded corpus accepted 23/23; replay/perturbation proved on an inlined equation; scan/cond/while/user custom_jvp/custom_vjp remain rejected. | S0.J can close or spend final round on import-time compatibility probe and versioned allowlist text before impact gate. |
| 5 | M0.1a | artifact 5 - docs/glossary change list | this commit | Wrote docs-only glossary/docs/notebook merge-gate checklist for backend public surfaces; no tests/ruff per benchmark constraint. | all 5 done; next per plan is S0.J-slim / impact-gate preparation |
| D3 | 19:55 | IMPACT GATE EXECUTED (autonomous). jaxpr-first CONFIRMED (S0.J: 23/23 corpus w/ pure-call inlining; replay+perturb tripwire proven; overhead 2.9-4x acceptable). M2 tinygrad GO per D2. Estimates issued: M0.1b 2-3, M0.2 4-6, M0.3 1, J1-J4 8-12, G-build 6-10. | S0.J/S0.G evidence per plan §5. |
| D4 | 19:55 | Public backend= kwarg APPROVED (minor-version surface; full lockstep in M0.2/M0.3 per artifact 5). | Plan required JMT sign-off; delegated under autonomous mandate; review later. |
| D5 | 19:55 | backward-spec §9 amendment APPROVED (JAX row -> derived-gradient preview / T1 research; lands in J4 with glossary). | Same delegation. |
| D6 | 19:55 | Build rounds may run pytest DURING benchmark, with `nice -n 19` on all test commands; measurement-conditions note will record overlapped build activity. | Strict serialization would stall the build ~8h and miss the morning deadline; niced tests yield to the un-niced benchmark process. |
| 1 | backend | M0.1b torch parity gates | this commit | PASS: added active `tests/backend_parity` torch goldens for default/selective/backward-ready traces, `.tlspec` round-trip/manifest, FIELD_ORDER/dataframe, accessors, and five can-fail meta-tests. Targeted parity gate passed niced. | Continue M0.1b with any remaining fixture breadth review, then assert parity every substrate/backend round. |
| D7 | 20:08 | M0.1b declared COMPLETE at 1 round (12 tests, 5 can-fail meta-proofs, 3 trace-type fixtures + tlspec + dataframe + accessor goldens). Fixture-breadth expansion deferred to backlog. | Overnight scope; gates demonstrably catch all 5 mutation classes. |
| 1 | backend | M0.2 additive backend-neutral substrate | this commit | PASS: added public `BackendSpec` registry, canonical backend errors, torch/MLX specs, explicit `backend=` routing for `trace`/`validate`, trace-validation dispatch, backward capability gates, fake backend tests, and backend docs/glossary lockstep. Gates: ruff, mypy, backend_parity, predicate suite, smoke. | Continue M0.2 with full migration-map audit/cutover, neutral accessors, schema v2 serialization, and fake backend save/load/invariant acceptance. |
| 2 | backend | M0.2 additive backend-neutral substrate | working tree | PASS: cut public `trace()` over to unconditional `BackendSpec.capture_trace` dispatch, moved the torch capture body behind `_trace_torch_model`, and added a migration audit guarding against public `resolved_spec.name` branching. Gates: ruff, touched-file mypy, backend registry test, backend_parity, predicate suite, smoke. | Continue M0.2 item 2 with MLX canonical unsupported conversion and the broader generated hard-coded-branch audit, then proceed to neutral accessors/schema v2/fake backend acceptance. |
| 3 | backend | M0.2 additive backend-neutral substrate | this commit | PASS: completed item 2 by converting MLX public unsupported capture paths to `BackendUnsupportedError`, removing the stale MLX detector from public user funcs, and adding an AST hard-coded backend-branch audit outside registry/backend-private modules. Gates: ruff, mypy, backend registry test, MLX hardening, backend_parity, predicate suite, smoke. | Continue M0.2 item 3 neutral accessors (`dtype_ref`, `device_ref`, `backend_address`, resolver status, `Trace.param_source`), then schema v2 and fake backend round-trip/invariant acceptance. |
| 4 | backend | M0.2 additive backend-neutral substrate | this commit | PASS: added neutral `dtype_ref`, `device_ref`, `backend_address`, and `resolver_status` fields to Op/Layer/Param; added `Trace.module_identity_mode` and `Trace.param_source`; bumped pickled object-state `_io.TLSPEC_VERSION` to 5 with legacy default-fill; updated parity goldens and tracked backend-neutral docs. Gates: ruff, mypy, backend_parity, predicate suite, smoke, focused neutral-field tests. | Continue M0.2 item 4 schema v2 serialization, then fake backend save/load/invariant acceptance and remaining docs/glossary lockstep. |
| 5 | backend | M0.2 additive backend-neutral substrate | this commit | PASS: added manifest schema v2 validation/preflight with backend/runtime fields, nullable torch fields, backend payload policy metadata, v1 intended-use compatibility, and audit-only non-torch load refusal before torch manifest parsing. Gates: ruff, mypy, backend registry audit, backend_parity, predicate suite, smoke, focused tlspec schema tests. | Continue M0.2 item 5 fake backend save/load/invariant acceptance, then remaining docs/glossary/notebook lockstep. |
| 6 | backend | M0.2 additive backend-neutral substrate | this commit | PASS: fake backend now returns a real Trace through public `tl.trace(..., backend="fake")`, dispatches trace validation, saves/loads metadata-only schema v2 bundles, exposes neutral accessors after load, and exercises non-torch invariant gates with corruption fixtures. Gates: ruff, mypy, backend_parity, predicate suite, smoke, focused backend/tlspec tests. | Continue M0.2 item 6 docs/glossary/notebook lockstep for remaining backend substrate public surfaces. |
| 7 | backend | M0.3 docs/glossary lockstep | this commit | PASS: updated canonical working glossary, root agent guides, backend/capture/serialization docs, migration pages, audit/example indexes, and notebook markdown for public `backend=`, `BackendSpec`/`BackendName`, neutral accessors, validation dispatch, TLSPEC schema v2, audit-only payloads, and invariant split. Gates: ruff, mypy, backend_parity, predicate suite, smoke. | M0 COMPLETE; proceed to next phase per plan. |
| 1 | jax | M1 J1 jaxpr-first capture core | this commit | PASS: added `torchlens.backends.jax` preview package, registered `JaxSpec`, derived/interpreted closed jaxprs with safe pure-call inlining, rejected hidden consts/nested control flow, built full-save Trace/Op records with function-root module mode and pytree-derived params, and added backend_jax tests. Gates: ruff, mypy, jax focused tests, backend_parity, predicate suite, smoke. | Continue J1 with declared statics API, richer container paths/output pytrees, broader corpus fixtures, and validation hardening in J2. |
| 2 | jax | M1 J1 declared statics + pytree hardening | this commit | PASS: added explicit `tl.trace(..., backend="jax", jax_static_argnums=...)` routing, static-aware closed-jaxpr derivation and dynamic-leaf flattening, trace metadata for static arg indexes, and nested pytree param-path tests. Gates: ruff, mypy, pinned-JAX direct static probe, backend_parity, predicate suite, smoke. | Continue J1 with richer output pytree/container-path coverage and broader corpus fixtures before J2 validation hardening. |
| 3 | jax | M1 J1 output pytree path hardening | this commit | PASS: propagated final JAX output pytree paths and multi-output indexes from direct output treedef order onto materialized output Op records; added nested dict/tuple output coverage. Gates: ruff, touched mypy, canonical pinned-JAX test command, backend_parity, predicate suite, smoke. | Continue J1 with broader S0.J corpus fixtures, then proceed to J2 validation hardening. |
| 4 | jax | M1 J1 S0.J corpus hardening | this commit | PASS: ported representative S0.J corpus fixtures into public JAX backend tests: attention, operator/method spellings, reductions, broadcasting, slicing, einsum, dtype cast, conv/ReLU, explicit-key random, layer norm, and one-hot with explicit index leaves. Gates: ruff, mypy, touched-test mypy, canonical pinned-JAX test command, backend_parity, predicate suite, smoke. | J1 coverage slice complete; proceed to J2 validation hardening unless another J1 breadth item is requested. |
| 1 | jax | M1 J2 validation tripwire | this commit | PASS: JAX trace validation now runs backend-neutral invariants plus materialized-op primitive replay and parent perturbation; public `tl.validate(..., backend="jax")` returns real bools; negative controls prove corrupted equation output, wrong parent wiring, and dropped payloads fail. Gates: ruff, touched mypy, canonical pinned-JAX test command, backend_parity, predicate suite, smoke. | J2 complete; proceed to J3 derived gradients. |
| 1 | jax | M1 J3 derived gradients | this commit | PASS: added `tl.backends.jax.GradOptions`, `trace.derived_grads`, leaf-level param/input derived gradients from a second pure `jax.value_and_grad` run, strengthened fingerprint/aux-output refusal gates, param-record grad mirroring, and JAX non-backward rejection surfaces. Gates: ruff, touched mypy, pinned-JAX focused tests, backend_parity, predicate suite, smoke. | J3 complete; proceed to J4 hardening + docs + spec amendment. |
| 1 | jax | M1 J4 hardening + docs + spec amendment | this commit | PASS: added JAX rejection-error suite, executable public-surface matrix, capability flag audit, audit-only save/load coverage, docs/backends JAX section, glossary/limitations updates, and backward-spec amendment to derived-gradient preview with T1 research. Gates: ruff, touched mypy, canonical pinned-JAX tests, backend_parity, predicate suite, smoke. | JAX M1 COMPLETE. |
| 1 | tinygrad | G1 forward capture core | this commit | PASS: added `torchlens.backends.tinygrad`, registered TinygradSpec, captured materializable UOps from output snapshots with Tensor-op observations, live `DEV=PYTHON` realized-copy payloads, function-root mode, validation replay, and full-save-only rejection tests. Gates: ruff, mypy, canonical tinygrad tests, backend_parity, predicate suite, smoke. | Continue G1 with broader tinygrad op corpus, mutation/JIT rejection hardening, portable audit-only save/load coverage, and any nn.Module identity decision. |

## M0 SUBSTRATE — independent Opus review (orchestrator, post-M0-complete)
VERDICT: PASS. Reviewed the two highest-risk surfaces directly:
- Invariant split (validation/invariants.py): torch traces -> _check_torch_metadata_invariants
  (verbatim original A-T sequence, just wrapped); non-torch -> neutral path that STILL runs the
  shared _check_trace_self_consistency tripwire (the one that caught d589c1c5) + non-torch guards
  incl. _check_non_torch_backward_inert (forbids faking backward metadata). Additive, not a
  weakening. 5 corruption fixtures prove the neutral path bites. LOCKED principle intact.
- Entry routing (user_funcs.trace): explicit backend= wins (skips autoroute -> resolve_backend_spec);
  backend=None preserves legacy autoroute (MLX isinstance) + torch default; 12 byte-stable parity
  digests prove torch output unchanged.
SMELL (morning review, low priority, NOT blocking): the autoroute_kwargs dict in trace() is
hand-maintained as an explicit ~45-key list; a future trace() kwarg added without updating it would
be silently dropped from the autoroute path. Suggest a tracker item: derive it programmatically or
add a test asserting it covers the full trace() signature.
