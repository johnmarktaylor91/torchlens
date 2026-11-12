## Critical Invariants

1. `_state.py` imports no torchlens modules EXCEPT the one sanctioned cycle-safe leaf
   import of `CaptureError` from `.errors._base` (documented in `_state.py` itself).
2. `pause_logging()` must wrap internal torch ops during logging (`safe_copy`,
   `activation_transform`). `get_memory_amount()` deliberately does NOT toggle it:
   it resolves the unwrapped size methods once instead (hot-path perf, `08dca260`).
   The pause also keeps an internal call out of `Trace._raw_call_fingerprint`, the
   ordered call/module-entry fingerprint the guarded fast re-run compares against a
   native forward (`torchlens/_call_fingerprint.py`): an unpaused internal torch call,
   even one that logs no op (`register_hook`, `element_size`), breaks that equality,
   and `tests/test_call_fingerprint.py` pins it. The same rule covers TorchLens's own
   metadata reads on op outputs: `internal_scalar_read()` pauses the fingerprint, unwrapped
   properties (`ndim`) replace wrapped methods (`dim()`), the synthetic `identity` op never
   notes a token, and a cached fast session's module token hooks are inert outside its own
   run (a Qwen3 capture otherwise counted 7 extra tokens, or every module twice).
3. Wrappers are persistent after lazy installation; `_logging_enabled` gates behavior.
4. FIELD_ORDER constants and class definitions must stay in sync.
5. Module suffixes are appended to `equivalence_class` at op creation before loop detection.
6. RNG state capture/restore must happen before `active_logging()` context.
7. There is no standalone `postprocess_fast()` orchestrator. Refresh captures run the full
   `postprocess()` entry point against the established Trace state; Step 0 reads the sealed
   `CapturedRunCore.events` snapshot and `RefreshProjector` applies refreshed payloads onto
   the existing graph.
8. `_tracing_finished` is set once at finalization and mirrored onto every retained op
   (`backends/_finalize.py`); nothing resets it per pass.
9. Portable `.tlspec` public schema is manifest-only; executable callables are not portable.
9a. Rehydration floor: this release reads only its own format. An artifact stamped
    `tlspec_version` < 11 was saved by torchlens 2.x and refuses to load with the one typed
    `tl.errors.ArtifactTooOldError` (code `artifact_version_below_floor`); no alias list,
    retired-key pop or legacy spelling adapter remains on any record class. The 2.16
    intervention spec format also refuses at this floor. Save writes typed JSON with no
    pickle: a value with no typed encoding is stringified with a manifest disclosure, a value
    with no 3.0 form at all refuses typed (`metadata_value_unportable`), and the reload check
    parses and decodes the written records before publish; load validates graph structure,
    edge carriers, and TorchLens-owned annotation families.
9b. tlspec v9 (coordinated schema write): the audit grammar admits per-site
    `source`, PARAM rows/recipes, and the EVENT envelope row kind (+ optional
    `seq`/`prev_event_digest` hash-chain slots); `"sidecar"` persists plainly with
    envelope-shape load validation; `"health_facts"`/`"capture_advisories"` are reserved
    annotation families; entry-dark declared slots `Op.call.injection_provenance`,
    `Trace.source.snapshots`, `Trace.graph.structure_evidence` validate fail-closed.
    Contract of record `torchlens/schemas/writer_contract_v10.json`; the field-intent
    census is maintained privately.
9c. The amendment rides the same v9 window (TLSPEC_VERSION stays 9): bundle relation
    grammar v2 (required/optional split; `successor_of` optional evidence envelope +
    `carry_mode` (closed `none`/`verbatim`/`summarized`)/`state_source`/`seams`; item origin
    `class` (seven kinds, required under reserved schemas); closed claim grades + contracted
    unchecked reasons; row grades DERIVED by `Bundle.relation_grades` (`broken` on exactly the
    bulk-origin and verbatim-carry premises; seam-local divergences never escalate);
    1 MiB per-row evidence budget; direction pin `from` = the LATER member), the
    preserve-and-disclose loader doctrine (unknown namespaced relation kinds ->
    `OpaqueRelationRow` under `tl.load(unknown_relations=)`; unknown evidence schema ids
    opaque; namespaced bundle.json sections preserve via `Bundle.preserved_sections`,
    bare-unknown refuse `bundle_section_unknown`), reserved IDs (`version_boundary_v1`/
    `turn_boundary_v1`; sidecar families `torchlens.input_origin`/`input_digest`/
    `boundary_facts`), episode ledger grammar v2 (`episode_ledger_version=2`; declared
    `step_output_kind` with the declaration-driven derivation — `step_output_from`
    source disclosure, tail-aligned `step_axis`, minted `capture_digest`; generic
    `step_output`; `cache_len` DELETED for the channel-keyed
    `entry_state_digest`/`exit_state_digest` carried-state witness slots; v1/foreign
    versions quarantine), and entry-dark slots `Trace.provenance.root_entry_point` (written unconditionally),
    `Op.episode_step`, `Op.call.tl_authored_root` with fail-closed validation.
9d. tlspec v10 (batch-2 coordinated schema write): one bump for every new saved
    field of the window, each admitted by the v10 field registry
    (`tests/test_schema_v10_contract_and_goldens.py::V10_FIELD_ADMISSIONS`) before its writer
    lands, so no further bump follows; the writer contract states the floor release as
    `3.0.0`, the first 3.0 writer (v9, v10 and v11 keep their frozen texts);
    `reserved_vocabularies.step_scores` reserves the
    generation step-score names, credited to inseq. Contract of record
    `torchlens/schemas/writer_contract_v10.json`; the v9 contract stays frozen beside it.
9e. tlspec v11 (persisted-field rename write): stored keys follow the decided public
    names (for example `Op.is_edited`, `Trace.is_replayable`, `Trace.name`), with no alias and
    no old-key reader, so a v10 file refuses rather than loading under new names. Contract of
    record `torchlens/schemas/writer_contract_v11.json`; the v10 contract stays frozen beside it
    (`tests/test_schema_v10_contract_and_goldens.py::V10_CONTRACT_SHA256`).
9f. tlspec v12 (3.0 columnar records; current): `Op.is_constant` (stamped by the shared
    constants rule) and `Op.source.arg_expressions` persist on the op record, admitted by
    `tests/test_schema_v12_contract.py::V12_FIELD_ADMISSIONS`; the reader floor moves to 12 with
    the contract. Contract of record `torchlens/schemas/writer_contract_v12.json`; the v11
    contract stays frozen beside it.
10. `backward_ready=True` rejects contradictory detaching/disk-save settings and preserves user
    `requires_grad` choices.
10a. `inference_only=True` wraps forward capture in `torch.no_grad()` for forward-only analysis
     and is mutually exclusive with backward-related capture because it discards autograd history.
11. Sibling ordering is forward/unrolled/dot-only; collapsed, rolled, backward, focused,
    conditional, and large graphs must conservatively no-op. Predicate-based smart collapse
    keeps sibling ordering enabled when endpoints survive as rendered nodes.
12. Predicate `save=` is the ONLY selective-capture spelling; the old
    `record(keep_op=...)` / `record(keep_module=...)` alias kwargs are removed and raise
    TypeError (`dry_run` likewise takes `save=` only). Module-boundary event recording is
    gated by `default_module=`, which records ALL module enter/exit events uniformly —
    predicate-gated module-event selection has no public spelling
    (`docs/reference/deprecations.md` has the honest capability statement).
13. `torch.func` / functorch transforms are captured as boundary ops; do not expect their
    per-element internal eager operations to appear unless a future expand-inside mode exists.
    `torch.autograd.grad` called inside forward is an `autogradgrad` boundary op
    (`transform_kind="autograd.grad"`, one op per returned gradient, selected by
    `tl.func_transform("autograd.grad")`); validation re-derives its gradients by replaying
    the recorded subgraph from its `inputs` to its `outputs`, never by exemption.
    `loss.backward()` / `torch.autograd.backward` called inside forward is an
    `autogradbackward` boundary op (`transform_kind="autograd.backward"`, one op per leaf
    gradient the pass writes, parented on the roots and on the leaf's previous grad when the
    pass accumulates into one, so the second pass through a retained graph is an in-place op
    and a later `param.grad` read has a producer; selected by
    `tl.func_transform("autograd.backward")`); validation re-derives each written gradient by
    replaying the recorded subgraph from the leaf to the roots and adding the previous grad.
    The engine's interior dispatches (including the accumulation `add_`) belong to that op,
    so the dispatch census balances without any exemption. A custom
    `autograd.Function` whose forward LEAVES torch (a `numpy` / `__array__` / `__dlpack__` /
    `to_dlpack` export, or a `from_numpy` / `from_dlpack` / `frombuffer` or parentless
    `as_tensor` / `asarray` op, inside its `apply`), or that is a known foreign kernel
    (`backends/torch/_foreign_data.py`: every Function of `flash_attn.cute.interface`,
    FlashAttention 4, hands its tensors to a CuTe kernel through TVM FFI, which no export patch
    sees; gated by `HAS_FLASH_ATTENTION_4_INTERFACE`, a filesystem probe that never imports the
    package), is one
    `autogradfunction` boundary op (`transform_kind="autograd.Function"`,
    `backends/torch/_function_boundary.py`): its interior ops are dropped in postprocess step
    3, every dispatch inside it is owned by the audited `autograd_function:foreign` token, and
    validation replays it by calling the Function's `apply` again; when a known kernel returns
    several tensors, the kernel table declares per output which positional arguments it depends
    on (ruling: "(b) Declare per-output dependence per kernel in a table."; FlashAttention 4's
    log-sum-exp reads q and k, not v) and the perturbation check excuses exactly the
    declared-independent edges, every undeclared kernel, output and position staying strict
    (`validation/_foreign_kernel_dependence.py`). The same table declares native custom ops
    recorded as sibling ops with no boundary (FlashAttention 2's
    `flash_attn::_flash_attn_forward`, keyed on the operator its replay callable calls plus
    its `func_name`: the log-sum-exp and dropout mask read q and k, the rng state none of q,
    k, v; the attention output has no row). A `vmap` boundary over a known transformers mask
    builder is checked against a declared mask-builder table (ruling
    d-aad71aaad45648ba93beaf7e05f2a1bb): `head_idx`, position 1, is excused only when the
    mapped callable is `and_mask`, `causal_mask_function` or the packed-sequence, padding or
    sliding-window inner mask of `transformers.masking_utils` by identity, an `and_mask`
    composes only table builders, and no leaf's code references `head_idx`; everything else
    stays strict (`validation/_mask_builder_dependence.py`). An ordinary custom Function
    (interior in torch) is captured op by op exactly as before. A round trip outside any
    Function is never collapsed into a boundary that hides ops: the export is journaled as a
    departure (`backends/torch/_foreign_data.py`; its own dispatches owned by the audited
    `foreign_data:export` token) and the host constructor that brings data back is one
    `foreigndata` op (`transform_kind="foreign_data"`) whose logged arguments are the departed
    tensors. A zero-copy or copy round trip (`resolution` `view` / `copy`) replays exactly from
    the departed tensor plus logged literals through the portable
    `torchlens._utils._foreign_data_replay.foreigndata` (modes `strided`, `gather`, `copy`),
    which a runnable artifact resolves from the closed synthetic table
    (`_io/_runnable_synthetic.py`); the departure's storage-range read runs with logging
    paused so it never poisons the completeness witness; a re-entry after host computation TorchLens did not see
    (`resolution` `unresolved`) is linked to every compatible departure and validation records
    the typed failure `foreign_data_roundtrip_unreplayable` for it, never a constant; only a
    `from_dlpack` of a consumed capsule in a capture with NO departure replays its recorded
    value (`resolution` `constant`). A `torch.jit.ScriptFunction` called from an
    eager forward is one `jitscript` boundary op (`transform_kind="jit.script"`,
    `backends/torch/_script_boundary.py`, same core): its dispatches are owned by the audited
    `jit_script:boundary` token and replay calls the function again; a call that writes into a
    tensor argument in place is never collapsed and stays a census gap.
14. Public backend-neutral state (`Trace.provenance.backend`, `module_identity_mode`, `param_source`,
    `dtype_ref`, `device_ref`, `backend_address`, `resolver_status`) must stay in docs,
    glossary, FIELD_ORDER, and serialization compatibility gates together.
15. TensorFlow `backend="tf"` / `backend="tensorflow"` targets Keras 3 on TF>=2.16 with
    `keras.backend.backend() == "tensorflow"`. Eager `op_callbacks` capture is the shipped primary
    path; the graph-only FuncGraph static importer is implemented for compiled/SavedModel entries
    (opaque regions stay unverified); derived gradients (leaf + exact T1 intermediates) ship
    for eager entries via `tl.backends.tf.GradOptions` with divergence refusal, and graph-only
    captures refuse `grad_options` typed; static-label `intervene=` ships for eager entries
    through the two-level writable layer (module-boundary + curated functional wrap) with
    fail-closed site reachability; `stop_after=`, `recipes=`, and true backward capture are deferred.
    All four eager non-PyTorch backends (tf/mlx/tinygrad/paddle) group recurrent calls into multi-pass
    layers through the neutral grouper; `recurrence_detection` stores the EFFECTIVE value
    (the TF static FuncGraph path stays ungrouped at `False`), validation sidecars stay keyed
    to raw capture identities, and tamper tests prove stale-label oracles fail closed.
16. Smart-collapse metadata is computed, not serialized: `Module.collapse_score`,
    `Trace.module_collapse_order`, and the per-mode ranking `Drawing.collapse_plan` carries
    (the public per-mode order accessor and its documented-inert `weights=` are gone, collapse
    memo) must stay
    out of `*_FIELD_ORDER` schemas until the policy is intentionally stabilized.
17. `Trace.load_state_dict(sd)` on a loaded sparse runnable Trace validates and stages state only;
   it never executes the DAG or writes values into the sparse core. User staging overrides the
   optional embedded capture state, and N1-a random fallback must name every initialized slot.
17a. `include_weights=True` is runnable-save-only and bundles the full capture-time `state_dict`
    (named parameters plus persistent buffers) in a separate `state_dict_v1` blob family. The
    sparse core remains tensor-value-free; load uses the ordinary strict binder and reports
    `embedded_capture_state`, never a reconstructed model. Used NON-persistent buffers are always
    shipped in the REQUIRED `runnable_nonpersistent_buffer_v1` family (not gated on either include
    flag) and are part of the declared state model -- a default runnable save of such a model
    carries those tensor values (disclosed via the manifest and a one-time save warning).
17b. `include_activations=True` is runnable-save-only and archives exactly the capture-time
    `save=`-selected `out`/`transformed_out` payloads in a separate `selected_activation_v2`
    family, whose eligibility metadata includes physical `InputAttestationFingerprint` records.
    `Trace.archived_activations` is inspection/attestation-only and never seeds execution.
    Original-input, capture-equivalent real-state runs must byte-attest saved raw slots before
    exposure; mismatch raises `replay_value_mismatch` with rollback, while changed-input
    (logical OR physical), random/non-equivalent-state, and nondeterministic-capture-context runs
    report `not_applicable`. `attested` implies `verified` and unpoisoned, always.
17c. Runnable descriptors are `sparse_recorded_taken_path_v2` (call recipe
    `non_tensor_args_tensor_slots_context_and_obligations_v3`): every call carries a REQUIRED
    explicit `CallExecutionContext` (autocast with affirmative disabled state + grad/inference
    mode) and the descriptor carries one `AmbientExecutionContext`, both restored at replay or
    refused typed. Absent context records only ever mean a legacy v1 artifact, which loads
    analysis-only. The runnable contract closes the witness-strip class with a structural obligation/discharge
    invariant: `WITNESS_FAMILY_REGISTRY` v2 (`witness_family_registry_v2`) covers EVERY
    verdict-steering witness family -- the four direct control kinds plus the shape families plus
    two claim-only families -- each row naming its independent replay-structural anchor. Every
    obligation is stamped on its owning replay record (`control_obligations` /
    `control_dependencies` on calls, `host_escape` / `inert_sink` on slots,
    `captured_requires_grad` / `captured_grad_fn` / `host_escape_disposition` on state bindings,
    the REQUIRED `input_boundary` record) and discharged by an exact witness XOR a typed
    `WitnessCoverageGap`; `witness_completeness` is DERIVED from the gap ledger (the summary is a
    redundant assertion, never authority), and the required-witness inventory is a redundant
    mirror. No record deletion can improve a verdict; the ONE out-of-scope boundary is coherent
    reauthoring (an honest capture of a weaker program), documented in the contract's threat-model
    subsection.
18. `Trace.run_transaction(inputs=..., seed=...)` is transactional for live and loaded sparse providers, runs
    internal sparse calls under `pause_logging()`, and returns `RunResult(output, trace, report)`.
    Stage 6 enforces input/state, per-call/output, and control-witness honesty before exposure;
    default divergence raises with rollback, while `on_divergence="return"` is the sole monotonic poisoned
    opt-in. Incomplete witness coverage is `unverifiable`; sparse-only and ineligible activation
    runs report numeric attestation as `not_applicable`.
18a. `Trace.run_transaction(inputs=..., fast=True)` is an explicit stateful static-loop mode: loaded sparse
    traces must first settle an ordinary run as `verified`, then may reuse staged state, compiled
    binders, and one result Trace; live traces use native forward plus targeted module hooks and
    functional collection only for the functional ops the capture saved (an explicit functional
    `save=` predicate or the save-all default), each module call collected by its call number.
    Per-call input, path, output structure/shape/dtype, and control-witness guards remain
    mandatory; divergence always raises. `until=` on a live fast run cuts the native forward
    after the saved site: the cut is a `live_stop_after` truncation, later sites read
    `None` for that run, and the verdict is `unverifiable` (no whole-forward fingerprint), which
    marks the live trace monotonically like every truncated run product. `fast=False`
    preserves the full transaction and attestation contract. The live session also fires a
    module-targeted staged intervention spec at module exit (the steered rerun engine behind
    `trace.run(model, x)`, which falls back to capture on a typed refusal and records it in
    `last_run["fast_refused"]`); a size-only input change is admitted only when the capture
    sealed a call fingerprint and the rerun's fingerprint matches, and the op metadata the run
    did not refresh is reset to `None`, never left at capture-time values. The session handle
    `Trace._fast_run_session` is a session-time `FieldPolicy.DROP` field (ordered under a private
    name, never persisted), ledgered in
    `tests/test_schema_lockstep.py::PRIVATE_ORDERED_DROP_FIELDS`.
19. Runnable vocabulary is frozen in the private `torchlens._runnable` (a user reads the report enums as `tl.types.*`): readiness is `ready|unavailable`,
    faithfulness is `verified|diverged|unverifiable`, state source is
    `live_model_state|embedded_capture_state|user_state_dict|random_initialization|not_applicable`,
    and divergence policy is `raise|return`. Error handling uses `RunnableErrorCode`, never
    exception-message matching. The frozen codes include `state_alias_topology_unsupported`
    (save-time refusal of distinct-object overlapping/unprovable bound-state alias topology; tied
    live-identity state stages as ONE alias-group allocation instead) and `context_field_invalid`
    (parse-time refusal of persisted execution-context values outside their closed vocabulary).
    The complete taxonomy and payload rules live in
    `docs/reference/runnable_tlspec_contract.md`.
19a. Runnable honesty boundaries: zero-tensor-leaf model outputs and namedtuple/mapping/registered
    containers carrying extra per-instance state refuse at runnable save
    (`output_container_contract_missing`; one per-kind capability table --
    `CONTAINER_KIND_CAPABILITIES` -- governs capture proof, save refusal, and forged-flag-proof
    load recompute; `tl.types.register_container` gains an explicit `state_complete=` declaration).
    Host nondeterminism outside the two replayable global engines (RNG instances incl.
    outside-held NumPy Generators via a chained `sys`/`threading.setprofile` classifier + a cheap
    model-attribute state digest (no process-wide gc scan), bare
    `_random.Random`, unseeded-construction `randbits` entropy, `SystemRandom`/`secrets`, OS
    entropy, `uuid4`, `default_rng`, the full clock family incl. `datetime.now`/`localtime`/
    `os.times`/`getrusage`) permanently ceilings replay at `unverifiable` + `not_applicable`;
    monitor uncertainty (install/chain/restore/inventory failure) downgrades completeness to
    INCOMPLETE; a realistic pre-existing-thread persistent-generator draw is witnessed by the
    setprofile classifier / model digest, and only an externally-held generator drawn on a
    pre-existing non-hooked thread is a documented residual (a benign background thread never
    ceilings a capture). Loaded-sparse and live
    providers settle through ONE finalizer
    (identical verdict class): a live opaque-container output is `unverifiable`+poisoned, a
    parse-refused descriptor degrades every payload family to analysis-only, and an inexecutable
    divergent input raises `PathDivergenceError`; structseq trust keys on the resolution authority
    (`spec.type_module`), never the spoofable `__module__`. Alias proofs run on absolute device-scoped byte
    intervals (storage-pointer equality never decides); the recorded default device enters as a
    scoped `with torch.device(...)`; CUDA state stages lazily and atomically at run preparation
    with a no-allocation readiness capability gate.
- ATen recording is opt-in and never silent: with `CaptureOptions(aten_ops=False)` (the default)
  no dispatch mode is installed and `trace.aten_ops` / `op.aten_ops` / `trace.aten_gaps` raise
  `AtenOpsUnavailableError`; with the switch on, a loaded profile's foreign keys are hostile input
  and still validate, and `mode_paused_interior` gaps are disclosed through `aten_gaps`, never
  filled with synthetic rows.
- ATen values (`CaptureOptions(aten_ops="values")`) are graph-detached snapshots in `AtenOp.out`
  kept only for calls whose `Op` saved its output: `aten_values_consistent` fails a kept value whose
  leaf count, shapes or dtypes disagree with `output_tensor_facts`, a forward row with a value whose
  owner saved nothing, and (under `save_mode="copy"`, no disclosed gap) a saved op none of whose
  rows reproduces its `out`. With `aten_ops=True` or off, no value is ever copied (tripwire in
  `tests/test_aten_values.py`).
