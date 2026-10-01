## Critical Invariants

1. `_state.py` imports no torchlens modules EXCEPT the one sanctioned cycle-safe leaf
   import of `CaptureError` from `.errors._base` (documented in `_state.py` itself).
2. `pause_logging()` must wrap internal torch ops during logging (`safe_copy`,
   `activation_transform`). `get_memory_amount()` deliberately does NOT toggle it:
   it resolves the unwrapped size methods once instead (hot-path perf, `08dca260`).
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
9a. Rehydration floor: artifacts stamped `tlspec_version` < 6 refuse to load with the typed
    `tl.errors.ArtifactVersionBelowFloorError`; legacy field-alias resurrection ladders are
    deleted. The first tlspec-6 writer was released torchlens 2.31.0, so 2.31.0 / 2.32.4
    artifacts load (the floor is the stamp, never a release name). Legacy 2.16 intervention
    specs remain loadable (floor covers Trace rehydration only). Save dry-runs `metadata.pkl`
    through the loader's default-deny unpickler and refuses unportable values typed
    (`annotation_value_unportable` / `metadata_value_unportable`); load anchors manifest facts
    to the pickled state (`bundle_manifest_metadata_mismatch`) and validates graph structure,
    edge carriers, and TorchLens-owned annotation families.
9b. tlspec v9 (C07 coordinated schema write; current): the audit grammar admits per-site
    `source`, PARAM rows/recipes, and the EVENT envelope row kind (+ optional
    `seq`/`prev_event_digest` hash-chain slots); `"sidecar"` persists plainly with
    envelope-shape load validation; `"health_facts"`/`"capture_advisories"` are reserved
    annotation families; entry-dark declared slots `Op.injection_provenance` (F01 writes),
    `Trace.source_snapshots` (F30), `Trace.structure_evidence` (F33) validate fail-closed.
    Contract of record `torchlens/schemas/writer_contract_v9.json`; the field-intent
    census is kept with the internal sprint records.
9c. The C07X amendment rides the same v9 window (TLSPEC_VERSION stays 9): bundle relation
    grammar v2 (required/optional split; `successor_of` optional evidence envelope +
    `carry_mode`/`state_source`; closed claim grades + contracted unchecked reasons;
    1 MiB per-row evidence budget; direction pin `from` = the LATER member), the
    preserve-and-disclose loader doctrine (unknown namespaced relation kinds ->
    `OpaqueRelationRow` under `tl.load(unknown_relations=)`; unknown evidence schema ids
    opaque; namespaced bundle.json sections preserve via `Bundle.preserved_sections`,
    bare-unknown refuse `bundle_section_unknown`), reserved IDs (`version_boundary_v1`/
    `turn_boundary_v1`; sidecar families `torchlens.input_origin`/`input_digest`/
    `boundary_facts`), episode ledger grammar v2 (`episode_ledger_version=2`; declared
    `step_output_kind` with the F40b declaration-driven derivation — `step_output_from`
    source disclosure, tail-aligned `step_axis`, minted `capture_digest`; generic
    `step_output`; `cache_len` DELETED for the channel-keyed
    `entry_state_digest`/`exit_state_digest` carried-state witness slots; v1/foreign
    versions quarantine), and entry-dark slots `Trace.root_entry_point` (written unconditionally),
    `Op.episode_step`, `Op.tl_authored_root` with fail-closed validation.
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
14. Public backend-neutral state (`Trace.backend`, `module_identity_mode`, `param_source`,
    `dtype_ref`, `device_ref`, `backend_address`, `resolver_status`) must stay in docs,
    glossary, FIELD_ORDER, and serialization compatibility gates together.
15. TensorFlow `backend="tf"` / `backend="tensorflow"` targets Keras 3 on TF>=2.16 with
    `keras.backend.backend() == "tensorflow"`. Eager `op_callbacks` capture is the shipped primary
    path; the graph-only FuncGraph static importer is implemented for compiled/SavedModel entries
    (opaque regions stay unverified); derived gradients (leaf + exact T1 intermediates) ship
    for eager entries via `tl.backends.tf.GradOptions` with divergence refusal, and graph-only
    captures refuse `grad_options` typed; static-label `intervene=` ships for eager entries
    through the two-level writable layer (module-boundary + curated functional wrap) with
    fail-closed site reachability; `halt=`, `recipes=`, and true backward capture are deferred.
    All four eager previews (tf/mlx/tinygrad/paddle) group recurrent calls into multi-pass
    layers through the neutral grouper; `recurrence_detection` stores the EFFECTIVE value
    (the TF static FuncGraph path stays ungrouped at `False`), validation sidecars stay keyed
    to raw capture identities, and tamper tests prove stale-label oracles fail closed.
16. Smart-collapse metadata is computed, not serialized: `Module.collapse_score`,
    `Trace.module_collapse_order`, and `Trace.collapse_order(mode=...)` (documented-inert
    `weights=` removed, collapse memo D9) must stay
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
    exposure; mismatch raises `numeric_attestation_failed` with rollback, while changed-input
    (logical OR physical), random/non-equivalent-state, and nondeterministic-capture-context runs
    report `not_applicable`. `attested` implies `verified` and unpoisoned, always.
17c. Runnable descriptors are `sparse_recorded_taken_path_v2` (call recipe
    `non_tensor_args_tensor_slots_context_and_obligations_v3`): every call carries a REQUIRED
    explicit `CallExecutionContext` (autocast with affirmative disabled state + grad/inference
    mode) and the descriptor carries one `AmbientExecutionContext`, both restored at replay or
    refused typed. Absent context records only ever mean a legacy v1 artifact, which loads
    analysis-only. r71 A closes the witness-strip class with a structural obligation/discharge
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
18. `Trace.run(inputs=..., seed=...)` is transactional for live and loaded sparse providers, runs
    internal sparse calls under `pause_logging()`, and returns `RunResult(output, trace, report)`.
    Stage 6 enforces input/state, per-call/output, and control-witness honesty before exposure;
    default divergence raises with rollback, while `return_diverged` is the sole monotonic poisoned
    opt-in. Incomplete witness coverage is `unverifiable`; sparse-only and ineligible activation
    runs report numeric attestation as `not_applicable`.
18a. `Trace.run(inputs=..., fast=True)` is an explicit stateful static-loop mode: loaded sparse
    traces must first settle an ordinary run as `verified`, then may reuse staged state, compiled
    binders, and one result Trace; live traces use native forward plus targeted module hooks and
    only explicitly requested functional collection. Per-call input, path, output structure/shape/
    dtype, and control-witness guards remain mandatory; divergence always raises. `fast=False`
    preserves the full transaction and attestation contract. The session handle
    `Trace._fast_run_session` is a session-time `FieldPolicy.DROP` field (ordered under a private
    name, never persisted), ledgered in
    `tests/test_schema_lockstep.py::PRIVATE_ORDERED_DROP_FIELDS`.
19. Runnable public vocabulary is frozen in `torchlens.runnable`: readiness is `ready|unavailable`,
    faithfulness is `verified|diverged|unverifiable`, state source is
    `live_model_state|embedded_capture_state|user_state_dict|random_initialization|not_applicable`,
    and divergence policy is `raise|return_diverged`. Error handling uses `RunnableErrorCode`, never
    exception-message matching. r37 adds the frozen codes `state_alias_topology_unsupported`
    (save-time refusal of distinct-object overlapping/unprovable bound-state alias topology; tied
    live-identity state stages as ONE alias-group allocation instead) and `context_field_invalid`
    (parse-time refusal of persisted execution-context values outside their closed vocabulary).
    The complete taxonomy and payload rules live in
    `docs/reference/runnable_tlspec_contract.md`.
19a. r37 honesty boundaries: zero-tensor-leaf model outputs and namedtuple/mapping/registered
    containers carrying extra per-instance state refuse at runnable save
    (`missing_output_container_contract`; one per-kind capability table --
    `CONTAINER_KIND_CAPABILITIES` -- governs capture proof, save refusal, and forged-flag-proof
    load recompute; `tl.register_container` gains an explicit `state_complete=` declaration).
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
