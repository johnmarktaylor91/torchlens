# TorchLens Capture Spine Contract v1

**Status: internal normative contract, semi-permanently locked. Not public API.**
This document governs the capture journal and its producers/consumers. Code conforms to
this contract; a disagreement between the two means the code is the bug. Public names,
signatures, labels, field values, error taxonomy, `.tlspec` manifests, and rendering
inputs are FROZEN independently of anything here.

The per-clause implementation status appendix at the end is honest bookkeeping for the
phased migration, not part of the contract.

## Clauses

**S1 — ONE RUN, ONE OWNER.** A capture run is owned by exactly one session, reachable at
exactly one place (`Trace._capture_session` while live; the trace-owned
`_capture_events` stream afterwards). No weak side registry may be an ownership head. A
released run raises a typed error (`BackwardStreamUnavailableError`); it MUST NEVER
fabricate a substitute buffer.

**S2 — ONE JOURNAL, ONE ORDER.** Every canonical fact receives a unique, strictly
increasing `seq` from one run-monotonic counter, spanning all kinds and all phases
(forward ops, module structure, pre-hook provenance, output versions, buffer writes,
intervention edits, and the whole backward family). `seq` order IS observation order.
Timestamps are diagnostics and MUST NOT decide order. Physical storage layout (per-kind
lanes plus derived indexes) is unspecified and MUST NOT appear in the contract.

**S3 — APPEND-ONLY.** A committed record is never mutated or removed. Later knowledge is
a typed record keyed to an earlier identity. Failed transactions remain as evidence: a
failed backward walk keeps its `BackwardPassStart` and gains a terminal
`BackwardPassEnd(status="error")`; a start record is never removed.

**S4 — IDENTITY SEPARATION.** Journal identity is `seq`. Operation identity is
`(raw_index, label_raw)`, reserved through the journal's label reservation. A native
pointer or Python `id()` MUST NOT be a durable identity; `id()`-keyed lookups are
runtime acceleration only and must be safe against recycling (today: object pinning for
the armed window guarantees stability; any earlier release must first introduce a
session-owned node uid).

**S5 — CORE/FACET.** An operation record is a required core plus typed optional detail.
A backend MUST NOT fabricate detail it cannot observe: absent detail != empty detail. A
consumer treats absence as typed absence — degrade explicitly or raise, never silently
default. (The physical `OpCore + FacetSet` decomposition is gated on the F1
microbenchmark, which has been run and reports DECOMPOSE; see status appendix.)

**S6 — HANDLE-FREE DURABLE RECORDS.** No durable record holds a live backend object,
payload, callable, or Trace back-reference. Torch op events are trace-backref-free from
birth (`source_trace=None`); natives are owned by run-scoped stores with declared
release boundaries (the escrow/spill machinery IS the payload store). A closed snapshot
must be mechanically scannable for native objects.

**S7 — ONE COMMIT TRANSACTION.** Per observed operation the stages (reserve identity ->
intervention at the characterized point -> selector/lookback exactly once -> demanded
enrichment -> payload disposition exactly once -> atomic append -> index update ->
halt/nonfinite at their characterized points) occur in that order, at most once each.
Exhaustive and predicate capture are two enrichment policies over this one transaction.
A failed stage emits a typed failure or a detectable reserved gap, never a half-mutated
public structure.

**S8 — EDITS ARE RECORDS.** An intervention, replacement, or any capture-time
modification of an observed value is an `InterventionAppliedEvent` referencing its
target label — never an op kind and never a mutation.
`"intervention_replacement"` is not a legal operation kind. Corollary: a functionless op
appearing during PLAIN capture is a capture bug and validation MUST fail it; the only
functionless carve-out is an op corroborated by a journal edit record appended by a site
that directly observed the edit (2026-06-02 lesson, kept armed by
`tests/test_validation_hardening.py`).

**S9 — MERGE LAW.** Every journal lane declares a merge policy
(`torchlens.ir.capture_events.LANE_MERGE_POLICIES`, total over all lanes);
`CaptureEvents.concat` is the ONLY way to combine streams (multi-pass recording,
failed-partial recovery). Ad-hoc lane splicing is forbidden. Merging preserves the
source stream's cross-lane chronological order, appends CLONES (the sealed source is
never mutated), and re-stamps them into the target journal's sequence domain by the
single writer. Every merging (non-`run_local`) lane MUST have a registered appender;
a declared-but-unwired lane fails closed (`LaneMergePolicyError`) before any event
moves, and the totality test asserts appender coverage.

**S10 — SINGLE WRITER.** Only the `CaptureEvents` append methods allocate `seq` or
append. Backends, projectors, postprocess, indexes, and public objects never mutate a
lane list directly, with ONE ledgered carve-out: the fastlog ancestry-closure backfill
(`fastlog/types.py::_backfill_cooked_ancestry`) replaces op-lane cells in place on the
`copy_for_replay` projection a cook owns (never a sealed stream) and invalidates
`_amended_fold_cache`. The closed set of in-place op-lane writers is enforced by the
package-wide AST ledger in `tests/producer_parity/test_op_lane_inplace_writers.py`.
Derived indexes (`LiveIndex`) are rebuildable caches, never authorities.

**S11 — HONEST LIFECYCLE.** Forward completion does not imply capture closure: backward
MAY append while armed. Detached streams (pickle restore, forks) record their
projection baseline (`pass_index_base` + cumulative counters) so rebuilds preserve
pre-detach facts instead of erasing them.

**S12 — BACKWARD BRACKETING AND COVERAGE.** Every ATTEMPTED backward pass has exactly
one start and exactly one terminal record in `seq` order (BaseException-safe), with all
its pass-scoped facts strictly between them; brackets are disjoint or properly nested.
Coverage gaps are typed `BackwardCoverageGap` records, never silent skips; only proven
framework-contract exclusions may preserve a complete-coverage claim, and validation
fails closed on every other reason (internal richness may exceed the frozen public
boolean, which stays true only for a full pass).

**S13 — NO FALSE BACKWARD.** Derived-gradient facts are a separately named family and
MUST NOT set true-backward capability or populate true-backward projections. Backward
capture is torch-only until a backend honestly implements it.

**S14 — PURE PROJECTION.** Public views (`grad_fn_logs`, `GradFn.calls`, per-op and
per-param gradient records, pass records, totals, `has_gradients`) are deterministic
functions of the journal plus frozen compatibility policy. Hooks and walkers emit
records and coordinator-local state only; a projection value without a backing record is
a validation failure (multiset reconciliation). One gradient OWNER per raw label:
aliased hook registrations emit exactly one observation per logical output per pass.

**S15 — CAPABILITY HONESTY.** Emission conformance is required of every backend;
optional drivers map 1:1 to declared capability flags, which are the only capability
truth — never inferred from method presence. Preview streams join the single-writer and
journal-seq disciplines in the ports phase; until then the journal-seq invariant is
enforced for the torch backend.

**S16 — TWO ORACLES, NEITHER WEAKENED.** The PUBLIC capture oracle is defined over
public observable facts (labels, field orders, graph order, payload results, error
taxonomy, TLSPEC manifests, rendering inputs) and is NEVER re-baselined. Internal spine
shape versions only by explicit schema change. No check, invariant, or tolerance may be
relaxed, exempted, or broadened for a migration. A check made trivially true by a
stronger recorded fact is REWRITTEN to assert the exact fact in the same commit, with a
mutation test proving it still fails on a planted violation. Redundant witnesses must
agree; disagreement is a failure, not a fallback.

**Non-contracts (explicit).** Single-threaded by design: the journal has no concurrency
contract. Physical journal storage, private class layout, object identity, and timing
are not compatibility surfaces. Events do not serialize into `.tlspec` under this
contract.

## Implementation status (2026-08-13, backend phases 0-6 + producer unification P0-P7)

| Clause | Status |
|---|---|
| S1 | Shipped (phases 2-3: registries deleted, `_EVENT_STREAMS` deleted, typed refusal). |
| S2 | Shipped for the torch backend (one writer-stamped seq across every lane; journal-seq invariant + mutation tests). Preview lanes: ports phase. |
| S3 | Shipped for backward (failed walks keep evidence) AND forward: the typed `OpAmendment` lane (nine exact-set families, lane-local seq domain, single writer `append_amendment`, one reducer `amended_op_records()`) landed with the producer decomposition (P4); `replace_op_event` deleted. |
| S4 | Shipped as stated; node-uid substitution deferred while dossier-#2 conservative pinning holds. |
| S5 | SHIPPED: decomposed `OpRecord` (`OpCore` + typed facets) is the ONE torch producer (dual-path transition P3-P6; legacy producer, `_op_event_from_log`, `_event_from_record`, and the `TORCHLENS_CAPTURE_PRODUCER` switch deleted in P7). Previews adapt at the one ingest boundary until S15. |
| S6 | Shipped: records trace-backref-free from birth; grad-fn handles live ONLY in the journal side index (`grad_fn_handles_by_label_raw`) — records never carry a handle attribute (single ownership, P2/P3). |
| S7 | Shipped for torch: ONE commit tail (`commit_op`: freeze -> atomic append) at every exhaustive and sparse site, stage matrix conformance-tested; preview conformance rides the ports phase. |
| S8 | Shipped (`InterventionAppliedEvent`, kind vocabulary cleaned, side ledger deleted, carve-out journal-backed). |
| S9 | Shipped (`concat` + total `LANE_MERGE_POLICIES` + total `_LANE_APPENDERS`, fail-closed on unwired lanes; chronological clone merge; recorder and failed-fastlog snapshot splices routed through `concat`). |
| S10 | Shipped (writer methods; direct lane appends routed, including the failed-fastlog snapshot; invariant enforces counter consistency). |
| S11 | Shipped (detached-stream baselines, phases 2-3 + fix rounds). |
| S12 | Shipped (exact bracketing, nesting, typed coverage gaps, fail-closed validation). |
| S13 | Standing (previews inert; `_check_non_torch_backward_inert`). |
| S14 | Shipped (single-write projections, multiset reconciliation, one-owner-per-label). |
| S15 | Standing declaration; ports/conformance suite is the parity-sprint deliverable. |
| S16 | Standing doctrine; eligibility classifier replaced the 0.80 coverage tolerance. |
