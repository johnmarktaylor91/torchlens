# Sparse runnable model

This document is the short normative model for TorchLens sparse runnable `.tlspec`
artifacts. It explains the ownership boundary and the invariants a maintainer needs in
order to change the subsystem safely. The exhaustive, frozen contract remains
[`runnable_tlspec_contract.md`](runnable_tlspec_contract.md). If this model and the
exhaustive contract disagree, the exhaustive contract and the behavior-free types in
`torchlens.runnable` are authoritative; the disagreement is a release-blocking defect.

## Scope

A sparse runnable artifact records one observed execution path without embedding a model
object or tensor-valued intermediates in its sparse core. It can be loaded for analysis and,
when readiness succeeds, replayed through the transactional `Trace.run()` API.

Runnable saving requires replay-template metadata from the original capture. Opt in with
`capture=tl.options.CaptureOptions(intervention_ready=True)` (or the equivalent flat
`intervention_ready=True` argument) before calling
`tl.save(trace, path, level="runnable", ...)`. A normal capture remains valid for analysis
saves, but runnable preflight rejects it with `RunnablePreflightError` and the structured
`MISSING_CALLABLE_REF` finding; saving cannot reconstruct the missing call templates later.

The subsystem has four ownership regions:

| Owner | Responsibility |
|---|---|
| `torchlens.runnable` | Frozen public schemas, enums, reports, and error vocabulary |
| Runnable internals | Descriptor semantics, validation, resolution, state preparation, execution, and attestation |
| `torchlens._io` | Manifest and blob transport, syntax-level encoding/decoding, integrity, and filesystem transactions |
| `Trace` | User-facing properties and provider selection through one private runnable-state seam |

The transport layer must not decide witness meaning, readiness, path faithfulness, or
attestation. Runnable code must not own filesystem transactions or manifest/blob integrity.
The narrow coordinator vocabulary is:

- `produce`: derive the semantic sparse descriptor and payload declarations from a live Trace;
- `decode`: parse transport data into typed runnable records without executing it;
- `prepare`: validate semantic consistency, resolve capabilities, bind payloads, and report
  non-executing readiness;
- `execute`: run the selected provider transactionally and settle one result.

`bundle.py` orchestrates transport through this boundary. It must not import or invoke private
witness-builder helpers directly. `torchlens.runnable` remains behavior-free and must not import
I/O, capture, state preparation, or execution modules.

No optional package extra is warranted. Sparse runnable execution requires no dependency beyond
TorchLens's existing torch requirement, so an extra would not describe a real capability boundary.

## Artifact families

The runnable artifact consists of one required sparse program and up to three external tensor
payload families. The sparse core itself is always tensor-value-free.

| Family | Presence | Purpose |
|---|---|---|
| `sparse_recorded_taken_path_v2` | Required | Taken-path call recipe, slots, contexts, state bindings, output contract, and honesty obligations |
| `state_dict_v1` | Optional | Full capture-time `state_dict`: named parameters plus persistent buffers |
| `runnable_nonpersistent_buffer_v1` | Required when used | Capture-time values of used non-persistent buffers in the declared state model |
| `selected_activation_v2` | Optional | Capture-selected `out`/`transformed_out` payloads for inspection and eligible byte attestation |

Weights and archived activations are independent opt-ins. Non-persistent-buffer payloads are not
an opt-in: when the taken path consumes one, its value is required state and the save discloses
that fact. Archived activations never seed execution. The complete payload rules are in
[contract section 5](runnable_tlspec_contract.md#5-producer-preflight-and-no-payload-invariant)
and the public option spelling is in
[section 12](runnable_tlspec_contract.md#12-optional-payload-api-spelling-and-docs-lockstep).

Every v2 call carries an explicit `CallExecutionContext`, and the descriptor carries an explicit
`AmbientExecutionContext`. Missing context records do not mean defaults; they identify a legacy
artifact that is analysis-only. The frozen call recipe is
`non_tensor_args_tensor_slots_context_and_obligations_v3`. These are internal schema version
strings, not a TorchLens major-version statement.

## Oracle and declared state

`verified` means faithful reproduction against oracle 1: a separate, fresh live-model execution
of the descriptor's program from the declared state, under the recorded execution context, on the
given inputs. It does not mean reproduction of a later call on the same mutated Python object.

The declared state is:

1. the capture-time `state_dict` universe of named parameters and persistent buffers;
2. the capture-time values of used non-persistent buffers; and
3. the recorded taken-path DAG and its explicit contexts and obligations.

Run preflight chooses state in this order: explicitly staged user state, embedded capture state,
then the frozen `torchlens_role_init_v2` random initializer. State staging is strict and atomic;
it never reconstructs a model object and never writes tensor payloads into the sparse descriptor.
The full boundary is specified in
[contract section 11](runnable_tlspec_contract.md#11-honesty-divergence-poison-and-exactness),
especially its declared-state subsection.

## Verdict lattice

Readiness is non-executing and uses exactly:

- `ready`;
- `unavailable`.

A completed path settles to exactly one `PathFaithfulness` value:

- `verified`: all applicable structural checks pass and no evidence ceiling prevents the claim;
- `diverged`: replay disagrees with the captured path or a required runtime fact;
- `unverifiable`: execution may complete, but evidence is insufficient to claim faithfulness.

Numeric attestation is downstream of path settlement:

- `attested`: eligible archived raw slots match exactly;
- `numeric_attestation_failed`: an eligible comparison mismatched and the transaction fails;
- `not_applicable`: the run is ineligible, including any path verdict other than `verified`;
- `not_present`: no selected-activation payload exists.

Attestation never upgrades path faithfulness. `attested` implies `verified` and an unpoisoned
result. A `diverged` result is returned only under the explicit `return_diverged` policy and is
monotonically poisoned. Incomplete witness coverage can settle only as `unverifiable`, never as
`verified`. The result and readiness shapes are frozen in
[contract section 8](runnable_tlspec_contract.md#8-readiness-and-result-shapes), and settlement
semantics are frozen in [section 11](runnable_tlspec_contract.md#11-honesty-divergence-poison-and-exactness).

## Witness obligations

The runnable proof is obligation-driven. Every replay-structure fact that can affect a verdict
creates a typed obligation on the record that owns it. Each obligation must be discharged by
exactly one matching witness or represented by a typed `WitnessCoverageGap`.

`WITNESS_FAMILY_REGISTRY` version `witness_family_registry_v2` is the closed inventory of every
verdict-steering witness family. It covers direct control witnesses, shape/structure fact
families, and claim-only families. The required-witness inventory is a redundant mirror of that
registry, not a separate authority.

`witness_completeness` is derived from the coverage-gap ledger. A persisted summary cannot
override that derivation. Deleting records or witnesses cannot improve a verdict: deletion either
leaves a still-complete exact discharge or creates a gap/refusal. The detailed owner fields,
anchors, XOR rule, and validation order are in
[contract section 4](runnable_tlspec_contract.md#4-authoritative-descriptor).

## Transaction boundary

`Trace.run(inputs=..., seed=...)` selects a live or loaded-sparse provider and returns
`RunResult(output, trace, report)`. The source Trace is unchanged for ordinary runs. State,
context, RNG, input binding, sparse calls, output reconstruction, honesty checks, and optional
numeric attestation settle inside one transaction.

The default divergence policy raises and rolls back. `return_diverged` is the sole opt-in that
returns a poisoned diagnostic Trace. Both providers use one settlement authority so they cannot
silently evolve different poison, exception, or report rules. Internal sparse calls run with
TorchLens logging paused.

`fast=True` is an explicit stateful static-loop mode, not a relaxation of the proof contract.
Loaded traces must first complete an ordinary verified run. Required input, path, output, and
control-witness guards remain active, and divergence always raises.

For dataset-style activation collection, use that ordinary run as a verify-once gate, then call
`run(..., fast=True)` for subsequent batches. Fast mode reuses staged state and compiled binders;
it does not remove the static-path and control-witness guards.

## Threat-model boundary

The proof covers faithful replay of the program described by the artifact. Coherent reauthoring is
out of scope: an author can construct a different, internally consistent descriptor for a weaker
program, and that descriptor may verify against that weaker program's own oracle 1. This is a
scope statement, not an undetected gap in witness coverage.

Within an artifact, record stripping must fail closed as a typed refusal, divergence, or
unverifiable verdict. The exact reauthoring boundary and residuals are frozen in
[the threat-model subsection of contract section 11](runnable_tlspec_contract.md#threat-model-scope-coherent-reauthoring-is-out-of-contract-scope-r71).

## Change discipline

The runnable vocabulary, witness meaning, verdict lattice, payload eligibility, and attestation
semantics are frozen. Changes require an explicit versioned contract amendment and synchronized
updates to typed schemas, serialization, validation, tests, and both runnable documents.

Refactors that do not amend the contract must preserve serialized bytes, error codes, readiness,
verdicts, poisoning, and attestation behavior. The exhaustive runnable, `.tlspec`, I/O, security,
and provider-parity suites are mandatory gates for changes at this boundary.

## Extension points (the runnable seam contract)

This section is the surface downstream consumers may bind to when extending the runnable
subsystem (new run modes, run-time state interventions, segment or backward replay
products). Everything else inside the runnable machinery is internal and may change
without notice.

**Binding rule.** `torchlens/_runnable_execution.py` is a namespace-aggregator facade over
its slice modules; slice modules have no importers besides the aggregator. Every extension
point below is named on ONE of `torchlens/runnable.py`, `torchlens/_runnable_seam.py`,
`torchlens/_runnable_execution.py`, or `torchlens/_runnable_state.py` (the coordinator
verbs additionally have their transport-side homes in `torchlens/_io/runnable.py` and
`torchlens/_io/runnable_load.py`, reached through the coordinator boundary). Consumers
never import a slice module directly; moving a name off its contract module is a contract
change.

**E1 -- Descriptor.** `SparseRunDescriptor` plus the closed enums and dataclasses in
`torchlens.runnable`, mirrored 1:1 by the exhaustive contract document. The registries a
consumer extends by adding a row are `WITNESS_FAMILY_REGISTRY` (the only dispatch
authority for witness producer/parser/mutation handling), `WITNESS_GAP_REGISTRY`, and
`CANONICAL_INITIALIZER_BY_ROLE`. New descriptor content is a versioned contract amendment
plus a registered persisted-field family (pre-release fields ship DROP-gated); no consumer
adds a descriptor field directly.

**E2 -- Provider.** `RunProvider` is a closed enum. The four-verb coordinator boundary is
`RunnableCoordinator` (`torchlens/_runnable_seam.py`): produce
(`build_sparse_run_descriptor`), decode (`parse_sparse_run_descriptor`), prepare
(`attach_sparse_run_readiness` plus preflight), execute (`run_loaded_sparse_trace` /
`run_live_trace`). Provider dispatch is the one hardcoded ladder inside `Trace.run`; a new
run mode extends that ladder, never adds a second dispatch site. The settlement spine is
single: every provider finalizes through `_finalize_provider_run`, and
`mark_trace_path_status` / `refuse_poisoned_trace` are the shared verdict authorities. The
Trace-side public surface is the closed member set `RUNNABLE_TRACE_PUBLIC_MEMBERS`;
session state lives only in `RunnableTraceState` under `trace.__dict__["_runnable"]`.

**E3 -- State.** The staged-state lifecycle: `load_trace_state_dict` strict atomic
staging, the ordered run-time precedence cascade (staged user state, then embedded
capture state, then random init), and the declared state model boundary of contract
section 11. Any consumer that binds, mutates, or substitutes state during a run goes
through the staging surface and the transaction; direct writes to live model state inside
runnable code are forbidden, and defensive materialization routes through the byte-guard
chokepoints.

**E4 -- Transaction.** `run_live_trace` and the loaded-sparse executor share the
call-cone scheduler, all-checks-before-exposure, slot-container clearing plus fork
unregistration plus host-RNG restore on every escape path, and RNG fork/restore in
`finally`. New run modes execute inside this transaction shape; early-stopping run
variants are implemented as scheduler-level cuts, never as post-hoc filters over a full
run.

**E5 -- Fast run.** `run_fast_loaded_trace` / `run_fast_live_trace`: verify-once then
guarded loop. The guard and detection-stage namespace (`fast_verify_once`,
`fast_state_static_guard`, `fast_seed_guard`, `fast_execution_context_guard`,
`fast_live_function_plan`, `fast_live_module_plan`, `fast_until_site_unsaved`, `fast_until_prefix`)
is contract surface. Fast mode
composes with new run keywords only by explicit matrix entry; silence means typed
refusal, never undefined behavior.

**Public entry.** The one public verb is `Trace.run`, whose dispatch ladder and typed
keyword-conflict matrix are contract surface: every new keyword lands with its
conflict-matrix row, and consumers bind totality tests to the CODE list, not a count.

**Protocol extension strategy.** `RunnableTraceProtocol.run` is the contract-pinned
stable typed minimum: its existing four parameters (`inputs`, `seed`, `fast`,
`on_divergence`) never change, reorder, or disappear. New keywords land as additive
keyword-only parameters with defaults on the concrete `Trace.run` while they are
documented-unstable; a ratified spelling joins the protocol as an additive
keyword-only-with-default revision in the same change as its documentation.

**Invariants consumers may rely on (and may not weaken).**

1. Single settlement: one finalizer, one report constructor, monotonic poison; a poisoned
   trace never un-poisons.
2. Closed vocabularies: `RunnableErrorCode`, `PathFaithfulness`, `CaptureStatus`, the
   capability-gate table. Additions only via contract amendment; never an in-code
   carve-out.
3. Deletion or omission cannot improve a verdict (contract section 4), extended to
   run-time omission: no truncated or partial run ever settles a positive claim it did
   not earn.
4. Transactionality: no partial state, RNG, or mode leakage on any failure path; restore
   runs in `finally`.
5. The live provider re-drives the full native forward except under an explicit,
   disclosed run-level truncation, whose result is blocked from full-forward export.
6. Meta-tested refactor constraints: no `RunResult` construction or report call outside
   `_finalize_provider_run`; no verdict derivation from the persisted completeness
   summary; no local alias-engine reimplementation; no bare state-path clones outside
   the byte-guard core; no direct legacy runnable Trace-attribute readers;
   `torchlens.runnable` import purity.
