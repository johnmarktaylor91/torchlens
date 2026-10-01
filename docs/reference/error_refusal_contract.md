# Error refusal contract

User-reachable TorchLens refusals are subclasses of `torchlens.errors.TorchLensError` and retain
their historical built-in exception compatibility where applicable. Callers should branch on
`exc.fields["code"]`, never on message text. Each refusal covered by this contract also carries a
non-empty `exc.fields["remedy"]`, and its human-readable message ends with that remedy.

The argument and capture-context classes are resolved lazily from `torchlens.errors`; they do not
add names to the top-level `torchlens` namespace:

- `InvalidArgumentError(ConfigurationError, ValueError)`
- `ArgumentTypeError(ConfigurationError, TypeError)`
- `ArgumentConflictError(ConfigurationError, ValueError)` — VALUE-combination
  conflicts (well-typed options whose values are mutually exclusive, e.g.
  `streaming.bundle_path` + `streaming.out_callback`) were historically raw
  `ValueError`s; the typed refusal preserves that catchability.
- `KeywordConflictError(ConfigurationError, TypeError)` — KEYWORD/call-surface
  conflicts (deprecated kwarg + replacement, grouped option + flat field, two
  exclusive call surfaces) were historically raw `TypeError`s (Python's
  duplicate-keyword convention); the typed refusal preserves that catchability.
  The split is site-by-site per git history, never a blanket base.
- `CaptureContextError(CaptureError, RuntimeError)`
- `DiagnosticSeverityError(ConfigurationError, ValueError)`
- `PayloadUnavailableError(CaptureError, ValueError)`
- `RecordBindingError(CaptureError, RuntimeError)`

## Stable codes

| Code | Refusal | Remedy class |
|---|---|---|
| `activation_not_saved` | Requested op activation was not retained by `save=` | Re-run the capture with `save=` covering the op |
| `annotation_backend_unsupported` | Tensor annotation on a non-torch trace | Pass JSON-serializable data instead |
| `annotation_namespace_invalid` | `annotations["user"]` is no longer a dict | Restore the user namespace to a dict |
| `annotation_not_json_serializable` | Annotation data is neither JSON nor a tensor | Convert arrays to `torch.Tensor` |
| `annotation_payload_missing` | `annotate()` received neither data nor image | Pass `data=`, `image=`, or both |
| `annotation_tensor_not_portable` | Annotation tensor fails the payload codec | Pass a dense, codec-supported tensor |
| `artifact_kind_mismatch` | Specialized loader received another artifact kind | Use the matching loader or generic `io.load` |
| `artifact_checkpoint_witness_invalid` | Loaded `Trace.checkpoint_invocation_witness` violates its closed token schema, D1-D6 flag vocabulary, verdict coherence, count geometry, or retained-site relations | Re-run backward capture and re-save; do not hand-edit checkpoint evidence |
| `artifact_distributed_scope_invalid` | Loaded `Trace.distributed_scope` is outside the closed `None` / `rank_local_shard` vocabulary | Re-save the source trace; do not invent distributed-scope markers |
| `artifact_capture_advisories_invalid` | Loaded `Trace.annotations["capture_advisories"]` rows violate their closed `{kind, count, first_location, message}` shape (tlspec v9) | Re-capture and re-save; do not hand-edit capture advisories |
| `artifact_grad_fn_timing_provenance_invalid` | Loaded `Trace.grad_fn_timing_provenance` names no supported timing source | Re-run backward capture and re-save without editing the clock source |
| `artifact_injection_provenance_invalid` | Loaded `Op.injection_provenance` violates the closed injected-op identity record (host_site_key, spec_rule_id, host_pass, firing_index, nesting_path, local_op_ordinal, output_slot; tlspec v9 entry-dark slot) | Re-capture with log_injections and re-save; do not hand-edit injected-op identity records |
| `artifact_intervention_audit_invalid` | Loaded `Trace.intervention_audit` or `HelperSpec.selection_recipe` violates its closed row schema or their resolve-digest relation | Re-apply the intervention and re-save; do not hand-edit audit or recipe rows |
| `artifact_kernel_telemetry_invalid` | Loaded kernel telemetry violates its closed launch schema or cites a nonexistent primitive sequence / launch index | Re-profile and re-save the trace; do not hand-edit launch or relation rows |
| `artifact_save_level_invalid` | `.tlspec` save level is unknown | Choose a documented save level |
| `artifact_save_level_unsupported` | Artifact kind cannot provide the requested save level | Choose a level supported by that kind |
| `artifact_sidecar_invalid` | Loaded `Trace.annotations["sidecar"]` namespace or an envelope violates its closed `{schema_id, version, owner, payload}` shape (tlspec v9; payload semantics stay with the owning family's read-time validator) | Re-attach through `torchlens.io.attach_sidecar` and re-save; do not hand-edit sidecar envelopes |
| `artifact_site_key_invalid` | Loaded `Op.site_key` is partial, malformed, or differs from the byte-exact `site_key_v1` recomputation over persisted op structure | Re-capture and re-save with one current TorchLens version; artifacts at the tlspec v6/v7 boundary may be wholly keyless |
| `artifact_source_snapshots_invalid` | Loaded `Trace.source_snapshots` violates the closed per-(path, digest) row schema (tlspec v9 entry-dark slot) | Re-capture and re-save; do not hand-edit the source-snapshot table |
| `artifact_structure_evidence_invalid` | Loaded `Trace.structure_evidence` envelope violates its closed key set / claim vocabularies, or decorates a capture whose `structure_only` marker is False (tlspec v9 entry-dark slot) | Re-capture structure-only and re-save; do not hand-edit the evidence envelope |
| `artifact_structure_only_incoherent` | Loaded structure-only marker is non-bool, co-occurs with a retained value payload (M-C2), or claims `capture_verified=True` (M-C3) | Re-capture and re-save with one current TorchLens version; do not hand-edit the marker, payloads, or the verification verdict |
| `artifact_version_above_runtime` | Bundle `tlspec_version` is newer than this runtime supports | Upgrade torchlens to the release that wrote the artifact (or newer) |
| `artifact_version_below_floor` | Artifact predates the rehydration floor (`tlspec_version` < 6 / torchlens < 2.33) | Load and re-save it with a torchlens release that still reads it |
| `history_artifact_corrupt` | A committed history chunk is missing or fails its recorded sha256; recovery never silently drops committed data (`HistoryArtifactError`) | Restore the chunk from backup or drop the artifact |
| `history_artifact_invalid` | History artifact directory misuse: existing manifest at write, unknown format/newer schema at read, or an uncataloged site in a chunk (`HistoryArtifactError`) | Write to a fresh directory, read a `torchlens.history.v1` artifact, or `add_site` first |
| `history_merge_incompatible` | Observation merge across different identities, conflicting scale stamps or presences, or mismatched sketch coverage (`HistorySchemaError`) | Merge only observations sharing (step, site, stream, phase) with consistent stamps |
| `history_ram_budget_exceeded` | The RAM-only history ring is full under the default `refuse` policy; nothing was dropped or degraded (`HistoryArtifactError`) | Attach a disk archive, raise the ring capacity or cadence, or opt into `drop_oldest`/`coarsen` explicitly |
| `history_step_regression` | Duplicate or decreasing global step within one segment (`HistorySchemaError`) | Pass strictly increasing steps, or declare a new segment for a resume |
| `history_vocab_invalid` | A history record field is outside its closed vocabulary, or payload/presence coherence is violated (`HistorySchemaError`, `HistoryArtifactError`) | Use the closed tokens; a non-observed presence carries no payloads |
| `observer_event_invalid` | An ObserverEvent violates the event schema: unknown kind/axis, missing kind-specific fields, or a gradient scalar without stage + scale_provenance (`ObserverEventError`) | Fix the kind-specific required fields; `unavailable` needs a reason and `'unknown'` is a legal scale |
| `observer_subscriber_disabled` | WARNING code (`TorchLensWarning`, contract row in `docs/reference/warning_contract.md`): an event-stream subscriber was disabled after consecutive failures reached the limit; collection was never interrupted | Fix the subscriber and re-subscribe it; `failure_counts()` carries the per-subscriber tallies |
| `observer_subscriber_failed` | WARNING code (`TorchLensWarning`, contract row in `docs/reference/warning_contract.md`): an event-stream subscriber raised during publish; collection continues and the subscriber will be disabled after consecutive failures reach the limit | Fix or unsubscribe the failing subscriber; subscriber bugs never interrupt collection |
| `profiler_session_invalid` | Profiler session argument misuse: borrowed mode without a profiler, owned mode with one, or an unknown mode/availability token (`ProfilerSessionError`) | Fix the session() arguments |
| `profiler_session_nested` | A profiler session is already active; there is ONE session engine and ONE activation knob (`ProfilerSessionError`) | Close the active session first, or add markers inside it via region() |
| `region_metadata_invalid` | Span or region metadata carries a non-scalar value (`SpanError`) | Pass bool, int, float, or str metadata values only |
| `sketch_descriptor_invalid` | Histogram grid descriptor with unsupported base, geometry, or encoding (`StatKernelError`) | Use base=2, hi_exp > lo_exp, bins_per_octave >= 1, encoding='dense' |
| `sketch_descriptor_mismatch` | Histogram merge across unequal grid descriptors; there is no rebinning path (`StatKernelError`) | Re-accumulate both populations on one shared descriptor |
| `span_altitude_invalid` | Span altitude outside the closed op / grad_fn_fire / aten set (`SpanError`) | Use one of the three altitudes |
| `span_not_open` | Span close named an id that is not open in the registry (`SpanError`) | Close each span exactly once, from the code path that opened it |
| `span_owner_invalid` | Span owner outside the closed owner classification incl. the mandatory torchlens_internal bucket (`SpanError`) | Use a closed owner kind |
| `stat_dtype_unsupported` | Spine/Histogram reduction over a complex dtype, which has no signed magnitude ordering (`StatKernelError`) | Reduce an explicit real view (`value.abs()` or `value.real`) and label it |
| `stat_merge_incompatible` | Spine merge with a non-Spine accumulator (`StatKernelError`) | Merge Spine accumulators with Spine accumulators only |
| `unknown_persisted_field` | Incoming persisted state carries field names outside the reader's declared record contract (fields + defaults + declared aliases); refuses in both writer directions before any object mutation | Load under the release that wrote the artifact; inspect inertly with `torchlens.io.inspect_state_contract`; a reader that dropped a persisted field must declare an alias |
| `ambiguous_op_lookup` | Accessor key matches multiple pass-qualified objects | Use a full address, pass label, or call index |
| `auto_environment_unsupported` | `TORCHLENS_AUTO=1` requested implicit capture | Unset it and call `auto_capture()` |
| `backward_capture_conflict` | `save_grads` conflicts with `backward_ready=False` | Enable or omit `backward_ready` |
| `backward_ready_conflict` | `backward_ready=True` conflicts with another capture option or ambient state — disk saves, explicit detaching, active inference mode, or a `keep_grad=False` default (`TrainingModeConfigError`, `ValueError` lineage) | Drop the conflicting option or drop `backward_ready=True` |
| `backward_graph_unavailable` | Backward drawing without a captured backward graph | Call `log_backward(loss)` first |
| `backward_pass_filter_invalid` | Backward pass filter is not a positive one-based int | Pass a positive pass number |
| `batch_items_invalid` | Export batch-items count is negative | Pass a non-negative integer |
| `batchnorm_train_stats_mutated` | WARNING (S-18 contract, not a refusal): tracing ran the real forward and train-mode norm layers advanced their running statistics in place; once per process | `model.eval()` before observational captures, or snapshot `state_dict()`; training-through-trace users ignore |
| `failed_capture_release_incomplete` | WARNING (S-18 contract, not a refusal): TorchLens could not fully remove its instrumentation after a failed capture; the model may keep instance forwards and fail to pickle | Call `tl.release_model(model)` once the underlying condition is resolved |
| `batch_render_invalid` | Batch render policy is unknown or malformed | Choose a documented batch_render policy |
| `backend_ambiguity` | Auto-resolution found multiple equal backend matches | Pass `backend=` explicitly |
| `backend_capability_conformance` | Advertised backend capability has no implementation (`BackendCapabilityConformanceError`, dual `ValueError` + `NotImplementedError` lineage) | Disable it or register its implementation |
| `backend_error` | Base-class default of the backend registry family — never raised directly; every live registry refusal carries one of the specific `backend_*`/`unknown_backend` codes below | Branch on the specific backend codes; this row exists only so an unmigrated future subclass is still documented |
| `backend_mismatch` | Explicit backend cannot handle the model or inputs | Select the owning backend |
| `bundle_load_failed` | Bundle load failed on torch/codec drift or a missing dependency | Inspect the chained cause; restore the missing dependency or re-save |
| `bundle_metadata_integrity_refused` | Bundle metadata pickle is denylisted, corrupt, or truncated | Treat as tamper/corruption; re-save from the source capture |
| `bundle_producer_unverifiable` | Current-schema bundle's recorded `torchlens_version` does not parse under PEP 440 | Re-save the artifact with a released torchlens |
| `bundle_torch_incompatible` | Bundle's recorded torch version is major-incompatible with (or unparseable against) the runtime torch | Load under a torch runtime with the recorded major version |
| `bundle_save_failed` | Bundle save failed; the staging dir was marked PARTIAL and any pre-overwrite bundle restored | Fix the chained cause named in the message and re-save |
| `bundle_member_has_relations` | A Bundle mutator (remove/clear/eviction) would orphan member-relation rows — silent orphaning is forbidden (S6 R5) | Pass `cascade_relations=True` to drop the rows explicitly, or remove the relations first |
| `bundle_relation_member_missing` | A member-relation row names a member absent from the Bundle (S6 R1: no dangling edges, ever) | Add the named member or drop the row |
| `bundle_relation_schema_invalid` | A member-relation row is outside the closed S6 schema (unknown kind, wrong row shape for the kind, or undeclared params) | Use the documented relation kinds and their declared params |
| `backend_payload_unsupported` | Backend payload has no supported codec (`BackendPayloadUnsupportedError`, dual `ValueError` + `NotImplementedError` lineage) | Save metadata only or use another backend |
| `backend_runtime_compatibility` | Runtime cannot materialize serialized backend data | Install a compatible runtime or analyze only |
| `backend_unsupported` | Backend does not implement the requested capability (`BackendUnsupportedError`, dual `ValueError` + `NotImplementedError` lineage; the TF site-reachability subclass shares it) | Omit it or use another backend |
| `buffer_visibility_invalid` | Unsupported `show_buffers` value | Choose a documented visibility policy |
| `bundle_member_payload_missing` | Bundle member retained no tensor at this node | Query a node with stored tensors |
| `bundle_member_unknown` | Bundle member name is not in the bundle | Pass a known member name |
| `bundle_shape_mismatch` | Bundle members have incompatible shapes | Query members with matching shapes |
| `bundle_stack_incomplete` | Not every bundle member retained a tensor | Stack where every member has a tensor |
| `bundle_diff_layout_invalid` | Bundle diff layout is unsupported | Pass `layout='paired'` |
| `bundle_diff_members_invalid` | Bundle diff sides are missing or identical | Name two distinct members |
| `bundle_statistic_invalid` | Bundle statistic is unknown | Choose `mean`, `std`, `var`, or `norm` |
| `code_panel_callable_return_invalid` | Callable code panel returned a non-string at render time (`ArgumentTypeError`) | Return the panel text as a string |
| `code_panel_model_collected` | Callable code panel needs the live model | Use a built-in code_panel mode |
| `code_panel_option_invalid` | Code panel mode literal is unknown (`InvalidArgumentError`) | Pass a documented mode or a callable |
| `code_panel_side_invalid` | Code panel side is unknown | Pass `side='right'` or `'left'` |
| `comparison_name_required` | Bundle comparison metric is a callable, which has no derivable stable storage name (`InvalidArgumentError`) | Pass an explicit `name=` for callable metrics |
| `comparison_unknown` | No Bundle comparison was stored under the requested name (`InvalidArgumentError`; the message lists the stored names) | Store the comparison first or query a stored name |
| `custom_callable_import_path_missing` | Custom function registry key lacks its `import_path` reference (`InvalidArgumentError`) | Supply `import_path='module:qualname'` on the registry key entry |
| `custom_callable_module_denied` | Custom callable resolves from a dangerous or stdlib/builtin module; denied even under trust (`UntrustedCallableError`) | Ship the recipe in a user module; dangerous modules never resolve |
| `custom_callable_module_not_allowlisted` | Custom callable's module is not in `allowed_custom_callable_modules` (`UntrustedCallableError`) | Add the named module to the allowlist if trusted |
| `custom_callable_not_pure` | Bundle-supplied callable is not a pure forward/tensor op (`UntrustedCallableError`) | Use pure forward/tensor ops in portable specs |
| `custom_callable_private_first_party` | Torchlens-owned callable is private or side-effecting; only vetted-inert public helpers auto-trust (`UntrustedCallableError`) | Reference a public facet recipe/transform/intervention helper |
| `custom_callable_untrusted` | Foreign custom callable resolution was not trusted (`UntrustedCallableError`) | Pass `allowed_custom_callable_modules={<named module>}` for a trusted spec |
| `dagua_renderer_not_opted_in` | Experimental dagua renderer used without opt-in | Import `torchlens.experimental.dagua` first |
| `capture_context_required` | Capture-only helper called outside `trace()` | Call it from the captured forward |
| `cleanup_during_active_capture` | `Trace.cleanup()` on the trace a live capture window is writing into | Let the capture or backward projection finish first |
| `child_process_capture_unsupported` | Capture attempted from a non-deliberate child process | Capture from the owning process, or an initialized SPMD rank |
| `unwrap_during_active_capture` | `unwrap_torch()` called while a capture is running | Finish or abort the capture before unwrapping |
| `pristine_ledger_poisoned` | The `_decorated_to_orig` unwrap ledger records a TorchLens wrapper as an original callable, so the pristine-torch restoration cannot be trusted | Restart the process; if it recurs, a wrap-machinery defect is recording wrapped callables as originals — report it |
| `collapse_level_invalid` | Float collapse level is outside `[0, 1]` | Choose an in-range level |
| `collapse_mode_invalid` | Collapse mode is unsupported — PER-SURFACE DOMAINS: rendering `collapse=` accepts `none`/`auto`/`max`/float, `collapse_plan(mode=)` accepts `auto`/`max`/float, `collapse_order(mode=)` accepts only `auto`/`max` | Choose a mode documented for that surface; the raised remedy names the exact set |
| `collapse_plan_unavailable` | Collapse optimizer declined the render context | Use a supported render context and mode |
| `container_leaf_not_saved` | Container leaf value was not retained | Re-run with `save=` covering the leaves |
| `container_not_reconstructable` | Container spec or backend support is absent | Capture with `capture_container_structure=True` |
| `container_selector_requires_registry` | Snapshot selector on a non-registry view | Call `reconstruct()` without site/role |
| `container_selector_unresolved` | Container selector matched zero or many records | Pass a more specific `site=`/`role=` |
| `container_spec_inadmissible` | Recorded output-container spec is corrupt, tampered, or names an inadmissible type (`ContainerReconstructionError`, `ValueError` lineage; default-deny security tripwire, also raised by public `Op.multi_output_type`) | Re-save the artifact from a trusted capture |
| `container_value_source_invalid` | Container value source is unknown | Pass `values='out'` or `'transformed'` |
| `context_field_invalid` | Persisted execution-context field fails its closed-vocabulary parse | Re-export the artifact; do not hand-edit descriptor context fields |
| `decoded_output_not_classification` | Decoded output is not a batch top-k table | Capture with classification output decoding |
| `decoded_output_unavailable` | Logits were not retained for re-decoding | Capture with retained logits or lower `top_n` |
| `derived_field_assignment_invalid` | Assignment to a derived compatibility field | Do not assign derived fields |
| `compiled_callable_unsupported` | Compiled plain callable has no module capture surface | Pass the original eager module |
| `compile_counts_unavailable` | `tl.debug.count_compiles()` found no Dynamo compile counters in this torch runtime (`CompileCountsUnavailableError`, `RuntimeError` lineage) | Upgrade torch or skip the verification on this runtime |
| `diagnostic_severity_invalid` | Diagnostic severity is outside the closed vocabulary | Choose a documented severity |
| `digest_hash_invalid` | `merkle_digest` got a hash name outside the closed `blake2b`/`sha256` vocabulary; the name is recorded in the manifest as a recomputation contract (spelling DOCUMENTED-UNSTABLE) | Pass `hash_name='blake2b'` (default) or `'sha256'` |
| `distributed_payload_witness_unsupported` | Payload witnesses are reserved | Use digest witnesses |
| `distributed_witness_invalid` | Distributed witness mode is unknown | Choose `none` or `digest` |
| `encoding_callable_error` | An encoding-channel user callable (`color_by=fn`) raised while resolving a node's value; the original exception is chained. UNSTABLE code (pre-ratification) | Fix the callable; read `Layer.ops` for per-pass truth instead of per-pass attributes on rolled aggregates |
| `encoding_requires_dot_layout` | Explicit `layout="rank"` with an active encoding channel (`color_by`); v1 channels are dot-layout-only. UNSTABLE code (pre-ratification) | Pass `layout="dot"` or `layout="auto"`, or drop the channel |
| `encoding_source_invalid` | An encoding-channel source names no known record field or scalar builtin, or (on a rolled multi-pass node) a field with no declared rolled-aggregate semantics row. UNSTABLE code (pre-ratification) | Pass a Layer/Op field name, a scalar builtin, or a callable; unroll the graph for per-pass sources |
| `encoding_scale_invalid` | `scale=` is outside the closed size-channel vocabulary (`"sqrt"`/`"linear"`; log rejected by design). UNSTABLE code (pre-ratification) | Pass `scale="sqrt"` (default) or `scale="linear"` |
| `encoding_value_invalid` | An encoding-channel source produced a value the channel cannot encode (bool, non-scalar tensor, shape from a callable, or other non-numeric). UNSTABLE code (pre-ratification) | Encode a numeric source, or convert the value inside a callable |
| `edge_intervention_engine_unsupported` | `do()` over an EDGE selection on the rerun/set_only engines; edge substitution ships on the replay/push engine only (L6 stage 3; spelling DOCUMENTED-UNSTABLE) | Run the edge intervention on the replay/push engine |
| `edge_intervention_save_unsupported` | Save of a trace carrying tier-(ii) edge-substitution entries under a schema whose active policy would drop them (schema-regression tripwire; the tlspec v8 bump persists the carriers, so ordinary saves proceed; spelling DOCUMENTED-UNSTABLE) | Save the un-edited source trace, or restore the persisting carrier policy |
| `edge_provenance_unavailable` | `trace.edges` / EDGE-kind resolution on a capture without edge provenance (`intervention_ready` not armed; spelling DOCUMENTED-UNSTABLE) | Re-capture with `CaptureOptions(intervention_ready=True)` |
| `env_flag_invalid` | A TorchLens boolean environment variable is set to an unrecognized value | Use `1`/`true`/`yes`/`on` or `0`/`false`/`no`/`off`, or unset the variable |
| `factcore_grain_invalid` | `FactCore` counts were asked for a grain outside the closed menu (`op` / `layer` / `site` / `module` / `call`) (`InvalidArgumentError`, `ValueError` lineage; spelling DOCUMENTED-UNSTABLE) | Pass one of the grain-menu tokens; the bare word "operations" has no grain |
| `factcore_identity_unknown` | An `IdentityIndex` join received a label or site key the identity spine does not carry — a silent empty join would fabricate absence (`InvalidArgumentError`, `ValueError` lineage; spelling DOCUMENTED-UNSTABLE) | Use a label from `identity.op_labels` / `identity.layer_labels` or a key from `identity.op_sites` |
| `flop_convention_invalid` | An FMA convention outside `{1, 2}` was requested | Pass fma=2 (stored convention, one MAC = 2 FLOPs) or fma=1 |
| `flop_convention_unavailable` | Explicit fma=1 request over ops with no derivable MAC split | Report under fma=2, or register a cost rule declaring the split (`register_op_rule`) |
| `episode_declaration_invalid` | The `episode=` declaration is unusable: the stepped module is not a proper submodule of the traced root, the token axis/output contract is unmet, the forced-token feed is malformed, or the declaration combines with chunking, `cache=True`, or a value-free save policy (the structure-only combination refuses with `structure_only_episode_unsupported`) | Fix the declaration per the message; wrap the loop in an `nn.Module` and declare its stepped submodule |
| `episode_ledger_incoherent` | Episode ledger geometry violates the monotone prefix law, a coherence arm, the declared step count, or the value-mode token presence rule; on load the ledger quarantines and the outcome derivation degrades fail-closed | Re-capture the episode; a hand-edited ledger never loads as claims |
| `episode_ledger_payload_in_structure_only` | A structure-only episode ledger carries token payloads (S7 presence rule) | Remove the token payloads or drop the structure-only marker |
| `episode_ledger_without_declaration` | An episode ledger is attached to a capture that carries no episode declaration — illegal per the marker-combination table | Capture with `episode=EpisodeSpec(...)` instead of hand-attaching a ledger |
| `episode_state_unsnapshotable` | Declared episode-carried state has no snapshot/restore support inside the declared checkpoint scope (E-A4); refused at declaration time, before execution | Declare only snapshotable state, or make the item deep-copyable |
| `error_constructor_args_conflict` | Diagnostic constructor got message args and fields | Pass a message or named fields, not both |
| `extraction_left_padding_unsupported` | An extraction batch's attention-mask pad geometry is not right-aligned and correct `position_ids` cannot be derived for this model (no declared `position_ids` forward parameter); absolute-position models read wrong activations under left padding (measured rel 25.6% BERT / 41.6% GPT-2; spelling DOCUMENTED-UNSTABLE) | Right-pad the batch (`padding_side="right"`), extract with `batch_size=1`, or include correct `position_ids` in each batch |
| `extraction_manifest_invalid` | A dataset-extraction artifact's `manifest.json` is unparseable, carries the wrong schema, is still `in_progress`, is missing ledgered shard files, or lacks a requested output key (spelling DOCUMENTED-UNSTABLE) | Finish the run with `resume=True`, request recorded output keys, or delete the directory and re-extract |
| `export_target_not_callable` | `register_export_target()` received a non-callable exporter (C01 export door; spelling DOCUMENTED-UNSTABLE) | Pass the exporter function itself |
| `export_target_tier_invalid` | Export-target registration declares a tier outside the closed present/bridge vocabulary (spelling DOCUMENTED-UNSTABLE) | Declare `tier='present'` or `tier='bridge'` |
| `extraction_ledger_invalid` | An artifact-v2 `ledger.jsonl` holds an unparseable MIDDLE line; later rows depend on the cumulative row ranges before it, so the trusted prefix ends typed (a torn FINAL line is crash debris — dropped, never trusted, never fatal; spelling DOCUMENTED-UNSTABLE) | Delete the output directory and re-extract, or restore the ledger from a backup |
| `extraction_ledger_prefix_broken` | A LEDGERED shard file is missing or byte-size-mismatched; under the append-only v2 ledger a committed row can never be rewritten, so both resume and load refuse instead of silently truncating (spelling DOCUMENTED-UNSTABLE) | Restore the missing shard file from a backup, or delete the output directory and re-extract |
| `extraction_model_identity_invalid` | `model_identity=` is outside its closed vocabulary (`"measured"` / `"none"` / an assertion Mapping — `"sampled"` is not offered: a sampled fingerprint admits exact algebraic collisions), or `asserted` was requested without an assertion (spelling DOCUMENTED-UNSTABLE) | Pass `"measured"` (default), `"none"`, or the assertion mapping itself |
| `extraction_resume_model_identity_mismatch` | The resuming run's MODEL IDENTITY differs from the artifact's record (different measured digest, different assertion, cross-level comparison, or an `unavailable` record) — continuing would silently mix the artifact's activations with a different model's, the T-MODELSWAP hazard a random-init resume used to complete with (spelling DOCUMENTED-UNSTABLE) | Resume with the exact model state that produced the artifact, or extract into a fresh directory |
| `extraction_resume_opaque_transform` | Resuming an interrupted artifact would run forwards through an OPAQUE transform step; an opaque callable's identity is a disclosure, not a proof, so the continuation cannot be verified to produce the prefix's numbers (transforms memo decision 14; spelling DOCUMENTED-UNSTABLE) | Register the transform under a versioned name (`torchlens.transforms.register_transform`) and pass the registered spelling on both runs, or extract into a fresh directory |
| `extraction_resume_requires_output_dir` | `extract_dataset(resume=True)` without `output_dir`; only disk mode leaves a shard ledger to resume from (spelling DOCUMENTED-UNSTABLE) | Pass `output_dir=` or drop `resume=True` |
| `extraction_resume_signature_mismatch` | A resumed extraction's semantic run parameters differ from the artifact's recorded signature, compared FIELD BY FIELD against the known-fields list (layers, batch size, transform chain, stimulus descriptor, stimulus-id digest, mode/pad policies, integrity policy, ...); the refusal names the exact fields (spelling DOCUMENTED-UNSTABLE) | Re-run with the artifact's original configuration, or delete the output directory to start fresh |
| `extraction_resume_unmanifested_dir` | `resume=True` on a directory holding batch shards but no manifest; completed work cannot be verified (spelling DOCUMENTED-UNSTABLE) | Delete the output directory (or use a fresh one) and re-run |
| `extraction_resume_v1_in_progress` | `resume=True` on an IN-PROGRESS v1-layout artifact: v1 recorded neither model identity, model mode, grad state, nor padding policy, so the prefix cannot be proven compatible with any resuming run (completed v1 artifacts migrate to v2 without a forward; spelling DOCUMENTED-UNSTABLE) | Read the trusted prefix with `load_extraction()` and re-extract into a fresh directory to finish the dataset |
| `extraction_stimulus_ids_cardinality` | `stimulus_ids` disagrees with the stimulus count — a sized input validates upfront, an unsized iterable refuses before the first under-covered shard commit; a mis-lengthed id list would mislabel every later row (spelling DOCUMENTED-UNSTABLE) | Pass exactly one identifier per stimulus, in order |
| `extraction_stimulus_ids_in_memory_unsupported` | `extract_dataset(stimulus_ids=...)` without `output_dir`: the in-memory result is a bare tensor mapping that can neither carry nor be affected by validated identifiers -- a false affordance (spelling DOCUMENTED-UNSTABLE) | Pass `output_dir=` to record stimulus identity in the manifest, or drop `stimulus_ids=` |
| `extraction_stimulus_ids_sidecar_invalid` | The write-once ordered `stimulus_ids.json` sidecar is unreadable or already records a DIFFERENT id list; relabeling is a separate audited verb, never a resume side effect (spelling DOCUMENTED-UNSTABLE) | Resume with the artifact's original `stimulus_ids`, or extract into a fresh directory |
| `facade_dependency_missing` | A real lazy-facade name's declared optional dependency is not installed, or its resolution import failed (`MissingDependencyError`, `AttributeError` lineage so `hasattr` answers `False`; the missing package and install command ride the message and `exc.fields["dependency"]`/`exc.fields["install"]` — CPython's instance-layout conflict forbids dual `ImportError` lineage; spelling DOCUMENTED-UNSTABLE) | Run the install command named in the message (`exc.fields["install"]`) |
| `facade_redirect` | A lazy-facade attribute access hit a redirect row: the name is not active here and the message names the canonical spelling (`FacadeTeachingError`, `AttributeError` lineage; spelling DOCUMENTED-UNSTABLE) | Use the canonical spelling named in the message |
| `facade_refusal` | A lazy-facade attribute access hit a refusal row: the name is deliberately not provided and the message names the owning tool or reason (`FacadeTeachingError`, `AttributeError` lineage; spelling DOCUMENTED-UNSTABLE) | Follow the pointer in the message |
| `fold_repeats_invalid` | Repeat-fold policy is invalid | Choose `None`, `True`, or `False` |
| `followed_by_unsupported` | `tl.followed_by(...)` predicate shape or retroactive capture is unsupported on this surface (`PredicateError`, `RuntimeError` lineage) | Compose `candidate & tl.followed_by(successor)` and capture with `tl.trace(save=...)` |
| `fsdp_capture_unsupported` | `record()` received an FSDP-wrapped model | Record the unsharded module |
| `import_path_invalid` | Custom-callable import reference is malformed | Use the `module:qualname` form |
| `intervening_cluster_invalid` | Intervening-cluster policy is unknown | Choose `upstream`, `outside`, `downstream`, or `own` |
| `intervention_tensor_unsupported` | Intervention save tensor fails the codec | Use dense, codec-supported tensors |
| `layers_not_logged` | Rendering requires a fully-logged trace | Capture with full layer logging |
| `lazy_uninitialized` | Capture entry on a model carrying un-materialized lazy BUFFERS (`LazyBatchNorm*` running stats): a pending buffer has no physical storage for the capture-boundary buffer-write tracker to index, so entry refuses with the pending modules/parameters/buffers named on `exc.fields` (`LazyStateUnsupportedError`; spelling DOCUMENTED-UNSTABLE). Pending lazy PARAMETERS alone never refuse: executed lazy modules (`LazyLinear`, `LazyConv2d`, ...) materialize during the ONE captured forward and never-run ones stay at zero geometry in the inventory. The same code names the zero-input shape-inference refusal (a lazy module accepts any width, so probing has no error signal) | Materialize once outside capture (`with torch.no_grad(): model(x)`), then retry the capture |
| `history_size_invalid` | Recorder history size is out of range | Pass an integer in `[0, 1024]` |
| `gradient_not_saved` | Requested gradient payload was not retained | Capture with gradient saving enabled |
| `graph_breaks_normalization_failed` | `tl.debug.graph_breaks()` got a Dynamo explain result of unrecognized shape (`GraphBreaksNormalizationError`, `RuntimeError` lineage) | Report the shape to TorchLens or pin a recognized torch version |
| `graph_breaks_unavailable` | `tl.debug.graph_breaks()` found no `torch._dynamo.explain` in this torch runtime (`GraphBreaksUnavailableError`, `RuntimeError` lineage) | Upgrade torch or skip the correlation on this runtime |
| `graphviz_binary_unavailable` | The Graphviz executable is not on PATH, so no render subprocess can start (`GraphvizUnavailableError`, `RuntimeError` lineage) | Install the Graphviz system package (`apt install graphviz` / `brew install graphviz`) |
| `graphviz_render_failed` | Graphviz did not produce a usable rendered artifact (`GraphvizRenderError`, `RuntimeError` lineage) | Lower dpi, render direct SVG, or cap the graph size |
| `gradient_pass_ambiguous` | Gradient query spans multiple backward passes | Pick one pass or record positionally |
| `grad_scale_invalid` | `gradient_flow_audit(grad_scale=...)` got a non-positive or non-finite loss scale (`InvalidArgumentError`) | Pass `scaler.get_scale()` from the GradScaler the captured backward ran under, or omit `grad_scale` |
| `grad_fn_fire_timing_unavailable` | `trace.grad_fn_fire_timings` read on a trace without its runtime capture event stream (loaded artifact or cleaned trace); the event-stream accessor is live-trace-only (spelling provisional pending the naming session / S2 routing) | Read on the live capturing trace; loaded artifacts carry the persisted per-fire `GradFnCall` timing fields (tlspec v8, discriminated by `Trace.grad_fn_timing_provenance`) |
| `grouping_invalid` | `grouping=` value is outside the closed vocabulary | Choose a documented grouping policy value |
| `grouping_policy_unavailable` | `grouping=` value is legal vocabulary but not entry-legal for this capture kind/wave (spelling provisional pending the S2 vocabulary amendment) | Use the default `grouping='structural'` |
| `halt_predicate_type_invalid` | `tl.trace` `halt` is not callable (`ArgumentTypeError`, `TypeError` lineage; the `tl.record` twin is `recording_halt_predicate_type_invalid`) | Pass a predicate or `None` |
| `hash_content_type_unsupported` | `tl.hash.content` value cannot be deterministically encoded (`ArgumentTypeError`, `TypeError` lineage) | Pass tensors, arrays, builtin scalars/containers, or `__dict__`-inspectable objects |
| `hash_expected_type_invalid` | `tl.assert_unchanged` pin is neither a string nor `None` (`ArgumentTypeError`, `TypeError` lineage) | Pass the pinned hash string, or `None` to bootstrap a pin |
| `inference_only_conflict` | `inference_only=True` combined with backward-related capture flags that need the discarded autograd graph (`TrainingModeConfigError`, `ValueError` lineage) | Drop `inference_only` or drop the backward flag |
| `input_namedtuple_schema_not_total` | Model-input tuple subclass declares a namedtuple `_fields` schema that does not account for the physical tuple — malformed non-tuple-of-str `_fields`, or declared arity differing from physical arity (`InvalidArgumentError`) | Fix the `_fields` declaration (one str per positional element) or pass a plain tuple/list |
| `input_tree_cycle` | Model-input tree contains a self-referential container (`InvalidArgumentError`) | Remove the container reference cycle from the model input |
| `input_tree_depth_exceeded` | Model-input tree nesting exceeds the input-boundary depth ceiling (`InvalidArgumentError`) | Flatten the nested input containers before tracing |
| `input_tree_stack_exhausted` | Walking the model-input tree exhausted the Python stack budget before the depth ceiling — capture was entered with most of the interpreter stack already consumed (`InvalidArgumentError`) | Enter capture from a shallower call stack or raise `sys.setrecursionlimit()` |
| `input_kwargs_type_invalid` | `tl.trace` `input_kwargs` is not a Mapping — usually multiple positional inputs passed as separate arguments (`ArgumentTypeError`, `TypeError` lineage) | Pass keyword args as a dict, or bundle positional inputs into one tuple |
| `intervention_replacement_invalid` | Intervention replacement payload has the wrong shape/type at the matched site (`HookValueError`) | Fix the replacement tensor passed to the `intervene=` clause |
| `layers_to_save_type_invalid` | Deprecated positional `layers_to_save` slot received a `torch.Tensor` — almost always a fourth positional model input (`ArgumentTypeError`, `TypeError` lineage) | Bundle positional inputs into one tuple; use `save=` for selection |
| `intervention_predicate_type_invalid` | `tl.trace` `intervene` is not callable (`ArgumentTypeError`, `TypeError` lineage; the `tl.record` twin is `recording_intervention_predicate_type_invalid`) | Pass `tl.when(...)`, another predicate, or `None` |
| `intervention_action_direction_invalid` | Predicate-side intervention action names an unknown direction (`ArgumentTypeError`; historically `TypeError`, so the live capture path converts it to `PredicateError`) | Choose `forward`, `backward`, or `both` |
| `intervention_action_type_invalid` | Intervention action has an unsupported type | Pass a decision, helper, callable, or `None` |
| `intervention_direction_invalid` | Trace-side intervention direction is unknown (`InvalidArgumentError`; historically `ValueError`) | Choose `forward`, `backward`, or `both` |
| `intervention_rule_type_invalid` | `InterventionSpec` clause is not an `InterventionRule` (PROVISIONAL spelling, documented-unstable; C03 spec noun) | Build clauses with `tl.when(...)` and merge specs |
| `intervention_spec_type_invalid` | `InterventionSpec.merge()` operand is not an `InterventionSpec` (PROVISIONAL spelling, documented-unstable) | Build each clause with `tl.when(...)` before merging |
| `intervention_where_invalid` | `tl.when(...)` WHERE term is not a callable selector or predicate (PROVISIONAL spelling, documented-unstable) | Pass a selector (`tl.func`, `tl.in_module`, `tl.site`, `tl.label`) or predicate callable |
| `intervention_engine_invalid` | `do(..., intervention=InterventionOptions(engine=...))` value is unknown | Choose `auto`, `replay`, `rerun`, or `set_only` |
| `extra_positional_input_invalid` | A tensor landed in `trace`'s fourth positional slot (`grad_transform`) -- almost always an extra positional model input | Bundle positional inputs into one tuple |
| `intervention_helper_unknown` | Built-in helper name is unknown | Choose a registered helper |
| `dense_subspace_full_axis_edit` | WARNING code (`TorchLensWarning`, contract row in `docs/reference/warning_contract.md`): `do()` on a dense-direction subspace selection performs a full-axis edit under set semantics, never a projection along the direction | Use a sparse direction (or raise `tol=`), or accept the full-axis edit knowingly |
| `intervention_over_invalid` | `mean_ablate(over=)` token is outside the closed vocabulary (`InvalidArgumentError`; fail-closed -- an accepted-but-ignored token would record a policy the hook does not compute, and a typo used to silently flip `batch_independent` open) | Pass `over='self'` for the global fire-time mean, or `source=<tensor>` for an external-source mean |
| `intervention_source_conflict` | `scramble_elements` received BOTH `source=` and `from_=` (one slot, two spellings; choosing one silently would hide a caller mistake) | Pass exactly one of `source=` or `from_=` |
| `jax_control_flow_invalid` | JAX control-flow mode is unknown | Choose `reject`, `unroll`, or `region` |
| `layer_pass_ambiguous` | Per-pass field read on a multi-pass layer | Access the field on one pass via `.ops[k]` |
| `layer_site_ambiguous` | `Layer.site_key` read on a layer spanning multiple structural sites (spelling provisional pending the S2 vocabulary amendment) | Read the per-pass key via `.ops[k].site_key` |
| `link_format_invalid` | Source-link format is unknown | Choose `terminal`, `html`, or `text` |
| `jax_unroll_range_invalid` | JAX unroll limit is below one | Pass a positive integer |
| `jax_unroll_type_invalid` | JAX unroll limit is not an integer | Pass a positive integer |
| `load_path_symlink_rejected` | A load path (bundle, manifest, metadata, or blobs) is a symlink | Pass the resolved real path |
| `logit_lens_head_unavailable` | No module in the trace exposes the `unembed_weight` facet, so the model's own lens cannot be reconstructed (`LogitLensError`, `RuntimeError` lineage; spelling DOCUMENTED-UNSTABLE) | Register a facet recipe producing the `language_model_head` facet names for the architecture (`tl.facets.register`), or pass `lens=` explicitly |
| `logit_lens_k_invalid` | `logit_lens_predictions(k=...)` received a non-positive top-k width (spelling DOCUMENTED-UNSTABLE) | Pass `k >= 1` |
| `logit_lens_position_invalid` | A `positions=` entry normalizes outside the full sequence length (spelling DOCUMENTED-UNSTABLE) | Pass positions inside `[-length, length)` |
| `logit_lens_projection_rank_invalid` | A layer's projected logits have rank below 2, so per-position prediction extraction is impossible (spelling DOCUMENTED-UNSTABLE) | Pass a `lens=` returning at-least-rank-2 logits for that layer |
| `logit_lens_token_id_invalid` | A `tokens=` id lies outside the projected vocabulary (spelling DOCUMENTED-UNSTABLE) | Pass token ids inside `[0, vocab)` |
| `lookback_invalid` | Lookback is not an integer in `[0, 1024]` | Pass an in-range integer |
| `lookback_payload_policy_invalid` | Lookback payload policy is unknown | Choose a documented payload policy |
| `lookback_payload_policy_conflict` | `tl.followed_by(...)` under `lookback_payload_policy='metadata_only'`, which retains no candidate payloads (`PredicateError`, `RuntimeError` lineage) | Pass a payload-retaining lookback policy such as `'detached_raw'` |
| `fastlog_index_too_large` | Fastlog recovery index exceeds the byte ceiling | Treat as a hostile/implausible bundle; re-record |
| `manifest_missing` | Bundle directory has no `manifest.json` | Pass the bundle directory produced by `tl.save()` |
| `manifest_not_json_object` | Manifest root parses but is not a JSON object | Re-save the artifact; do not hand-edit the manifest |
| `manifest_schema_invalid` | Manifest parses as a JSON object but violates the bundle schema (missing/mistyped field, forged entry, count mismatch) | Re-save the artifact with `tl.save()`; do not hand-edit the manifest |
| `manifest_unreadable` | Manifest cannot be read or does not parse within bounds | Check permissions/integrity; re-save if truncated |
| `manifest_write_failed` | `manifest.json` could not be written during save | Check disk space and directory permissions, then re-save |
| `meta_kernel_unavailable` | An op with no meta kernel died inside torch dispatch during structure-only capture (`MetaKernelUnavailableError`; original chained) | Run a real capture, or upgrade torch for broader meta-kernel coverage |
| `metadata_object_count_exceeded` | `metadata.pkl` opcode count exceeds the allocation ceiling | Treat as a hostile/implausible artifact; re-save from source |
| `metadata_payload_not_a_mapping` | `metadata.pkl` payload is not a metadata mapping | The artifact is corrupt or hand-edited; re-save with `tl.save()` |
| `model_type_unsupported` | Torch capture model is not an `nn.Module` | Pass a module or select its backend |
| `model_wrapper_uninstrumentable` | Model is a transformer_lens `TransformerBridge`, whose `__setattr__` redirects assignment so capture instrumentation cannot land (`UninstrumentableModelWrapperError`) | Trace a pristine reloaded HF model or a HookedTransformer instead of the bridge |
| `max_pairs_invalid` | Bundle diff pair budget is below one | Pass `max_pairs >= 1` or None |
| `max_predicate_failures_invalid` | Predicate failure budget is not a non-negative int | Pass a non-negative integer |
| `module_focus_empty` | Focused module contains no rendered layers | Focus a module with layers |
| `module_focus_invalid` | Module focus value has the wrong kind or owner | Pass an owned Module or address string |
| `module_focus_not_found` | Module focus address is not in the trace | Pass an existing module address |
| `metric_name_invalid` | Intervention metric name is unknown | Choose a registered metric or callable |
| `metric_shape_mismatch` | Metric operands have different element counts | Pass equal-size operands |
| `metric_tensor_type_invalid` | Metric operand is not a tensor | Pass tensor operands |
| `metric_type_invalid` | Metric selector is neither a name nor callable | Pass a registered name or callable |
| `module_call_ambiguous` | Single-call accessor on a multi-call module | Access one call via `module.calls[N]` |
| `multipass_bare_label_ambiguous` | A bare layer label (exact string, `tl.label`, or `do`/`attach_hooks`/`push_from` string target) addressed a multi-pass (recurrence-grouped) layer: every pass is a distinct op, and a single-op consumer must never guess a pass (`SiteAmbiguityError`; the teaching message names the layer, its pass count, and every pass-qualified spelling; spelling DOCUMENTED-UNSTABLE). `tl.units` refuses the same case as `selection_unresolvable` / `multipass_bare_label` | Address one pass with a pass-qualified label (`label:pass`), or select every pass explicitly with the Layer selection (`log[label].__selection__()`) |
| `node_label_field_invalid` | Node label field name is unknown | Pass documented label field names |
| `node_overlay_invalid` | Node overlay name is unknown | Choose a supported overlay |
| `op_lookup_index_out_of_range` | Integer layer index is out of range | Pass an in-range index |
| `op_lookup_not_found` | Lookup key matches no layer, op, or module | Use a valid label, index, or address |
| `op_lookup_pass_out_of_range` | Pass qualifier exceeds the recorded pass count | Specify a lower pass number |
| `op_lookup_pass_required` | Bare label names a multi-pass layer | Append a pass qualifier such as `:2` |
| `on_forward_error_invalid` | Forward-error policy is unknown | Choose `raise`, `attach_partial`, or `return_partial` |
| `on_predicate_error_invalid` | Predicate-error policy is unknown | Choose `auto`, `accumulate`, or `fail-fast` |
| `predicate_default_invalid` | Default capture decision is neither bool nor `CaptureSpec` (`PredicateError`, `RuntimeError` lineage) | Pass `True`, `False`, or a `CaptureSpec` as the default |
| `predicate_name_conflict` | `register_predicate(name)` collides with an already-registered predicate name (S4 registry; spelling DOCUMENTED-UNSTABLE) | Pick an unused name, or unregister the existing entry first |
| `predicate_unregistered` | A string predicate name passed to a capture-lifecycle slot is not in the S4 registry (spelling DOCUMENTED-UNSTABLE) | Register the predicate first, or pass the callable directly |
| `predicate_evaluation_failed` | Accumulated predicate exceptions surfaced at the end of a recording (`PredicateError`, `RuntimeError` lineage; `exc.failures` carries the tracebacks) | Fix the predicate using the accumulated failure tracebacks |
| `predicate_return_invalid` | A save, intervene, or halt predicate returned a value outside its declared contract (`PredicateError`, `RuntimeError` lineage) | Return a documented decision value from the predicate |
| `predicate_storage_conflict` | `keep_grad=True` conflicts with disk-only storage or an integer/bool payload dtype (`PredicateError`, `RuntimeError` lineage; the grad-slot twin raises `InvalidStorageError`) | Keep the payload in RAM with a floating dtype, or drop `keep_grad=True` |
| `param_substitution_engine_unsupported` | `do()` over a PARAM selection on the rerun/set_only engines; parameter substitution ships on the replay/push engine only — the value each op consumes is substituted "as if" the parameter were changed, and the live parameter is never written (spelling DOCUMENTED-UNSTABLE) | Run the parameter substitution on the replay/push engine (the default when no model/x is passed) |
| `param_substitution_occurrence_underivable` | A PARAM-selection `do()` could not address every consumption of the parameter: no matching top-level template component, a nested container position, a released/legacy capture without identity or barcode, a bare (pass-ambiguous) multi-pass consumer spelling, or a consumer inventory omitting a pass of a multi-pass layer — a partial substitution would be a silent wrong "as if". Recurrently reused params (tied weights, multi-pass consumers) ARE substitutable when every consumption is recorded pass-qualified (spelling DOCUMENTED-UNSTABLE) | Target a parameter whose consumptions are top-level arguments, or re-capture live with `intervention_ready=True` |
| `patch_donor_pass_ambiguous` | `patch_from` reached a donor site that is multi-pass on the source trace while the fire context carries no pass: a donor pass is never guessed (the historical bare lookup silently returned the LAST pass's value); the teaching message names every pass-qualified donor spelling (`HookValueError`; spelling DOCUMENTED-UNSTABLE) | Address one donor pass explicitly with a pass-qualified label (`label:pass`) |
| `patch_ineffective` | An activation-patching rerun left an empty positive fire ledger — the hook never fired, or every fire was refused (`replaced=False`) — so the metric row would silently equal the corrupted baseline (`PatchApplicationError`, `RuntimeError` lineage; spelling DOCUMENTED-UNSTABLE) | Inspect `tl.facets.facet_coverage(trace)`, choose a facet whose home is a real computation site, or patch an explicit op site via `fork.do()` |
| `postprocess_audit_asserts_stripped` | `TORCHLENS_POSTPROCESS_ASSERTIONS` is armed under `-O`/`-OO`, so every contract check would be stripped and the audit would report clean without verifying anything | Re-run without `-O`, or unset the variable |
| `postprocess_audit_env_invalid` | A `TORCHLENS_POSTPROCESS_*` audit knob holds an unrecognized value (a typo must never silently rearm or disarm an audit) | Use a documented value for the knob, or unset it |
| `option_group_conflict` | Grouped and flat options set the same field on a merge entrypoint (`ArgumentConflictError`; historically `ValueError`) | Use one option style |
| `option_group_keyword_conflict` | Flat draw kwarg and `VisualizationOptions` field set the same option (`KeywordConflictError`; historically `TypeError`) | Use one option style |
| `option_group_type_invalid` | Grouped option has the wrong object type | Pass the documented options class |
| `output_attribution_failed` | A model output tensor could not be attributed to any traced op — an opaque execution boundary or a pre-bound torch function that escaped wrapping (`OutputAttributionError`) | Use ordinary torch module attributes during forward, or bind/import torch functions after TorchLens has wrapped torch |
| `output_unsupported_tensor_variant` | A model output is a NESTED tensor constructed inside `forward()`, an unsupported tensor variant TorchLens cannot log (`OutputAttributionError`) | Build the nested tensor outside the traced region, or pad to a dense tensor before the ops you want captured |
| `output_device_invalid` | Output device policy is unknown | Choose `same`, `cpu`, or `cuda` |
| `output_tree_cycle` | Model-output tree contains a self-referential container, so its occurrence-weighted tensor sum is undefined in `validate_backward_pass` (`InvalidArgumentError`) | Remove the container reference cycle from the model output |
| `output_tree_depth_exceeded` | Model-output tree nesting exceeds the output-boundary depth ceiling in `validate_backward_pass` (`InvalidArgumentError`) | Flatten the nested output containers before validating |
| `payload_unavailable` | The scoped non-attaching payload reader found neither a resident payload nor a lazy blob ref on the op — nothing to read (`PayloadUnavailableError`, `ValueError` lineage; spelling DOCUMENTED-UNSTABLE) | Re-capture with `save=` covering the op, or save the artifact with `include_outs=True` |
| `recipes_unknown_provider` | WARNING (S-18 contract, not a refusal): `activate_entrypoint_recipes(names=...)` was asked for entry-point names that are not installed; the unknown names were skipped and the installed provider names listed | Activate an installed provider name from `installed_recipe_providers()`, or install the provider distribution first |
| `record_not_bound` | Record's owning Trace reference is gone | Keep the owning Trace alive |
| `recording_events_not_retained` | `to_trace()` on a disk-recovered Recording | Convert the in-session Recording |
| `recording_backward_halted` | `log_backward()` on a halted Recording (the sparse frontier retained no complete output to root the backward walk) | Re-record without `halt=` or use `trace(halt=...)` for prefix backward |
| `recording_event_stream_unavailable` | Raw per-event metadata read on a Recording that no longer holds its capture event stream — explicitly cleaned, or restored from a payload-only projection (`RecorderStateError`) | Read events on the in-session Recording before cleanup |
| `recording_failed_not_convertible` | `to_trace()` on a failed partial Recording | Fix the forward and re-record |
| `recording_halt_frontier_missing` | Halted Recording retained no frontier payload | Save the halt frontier or use `trace(halt=...)` |
| `recording_halt_predicate_type_invalid` | `tl.record` `halt` is not callable (`InvalidArgumentError`, `ValueError` lineage; the `tl.trace` twin is `halt_predicate_type_invalid`) | Pass a halt predicate or `None` |
| `recording_intervention_predicate_type_invalid` | `tl.record` `intervene` is not callable (`InvalidArgumentError`, `ValueError` lineage; the `tl.trace` twin is `intervention_predicate_type_invalid`) | Pass `tl.when(...)`, another predicate, or `None` |
| `recording_multipass_not_convertible` | `to_trace()` on a multi-pass Recording | Record one pass per Recording |
| `reentrant_trace` | `tl.trace` was started while another capture was active (`ReentrantTraceError`, `RuntimeError` lineage) | Finish the outer capture before starting another |
| `release_during_active_capture` | `tl.release_model()` while a capture is still active | Let the capture finish before releasing the model |
| `recording_option_duplicate` | Recording option was specified twice | Pass each option exactly once |
| `recording_option_type_invalid` | Recording option has an unsupported type | Pass the documented type for that option |
| `registry_capabilities_missing` | A registration reached a typed registry door without capability rows (`RegistryError`; registration is refused, never accepted blind) | Declare `capabilities={...}` stating what the provider can and cannot do |
| `registry_capability_key_invalid` | A capability row key is not a non-empty string (`RegistryError`) | Use non-empty string capability names |
| `registry_capability_value_invalid` | A capability row value is not a portable scalar (bool/int/float/str or tuple-of-str) (`RegistryError`) | Use portable scalar capability values |
| `registry_domain_duplicate` | A second kernel registry was created under an existing domain name (`RegistryError`) | Import the owning door module instead of re-creating its registry |
| `registry_entry_duplicate` | A registration collides with an existing entry id; collisions refuse by default (`RegistryError`) | Pass `replace=True` explicitly, or register under a different id |
| `registry_entry_id_invalid` | A registry entry id is not a non-empty string (`RegistryError`) | Pass a stable non-empty string entry id |
| `registry_entry_unknown` | A lookup named an entry id absent from the registry; the refusal lists the registered ids (`RegistryError`) | Register through the domain's public door first, or use a registered id |
| `registry_name_invalid` | A kernel registry was created with an invalid domain name (`RegistryError`) | Pass a non-empty string registry name |
| `registry_provider_invalid` | A registration carries no stable provider identity (`RegistryError`) | Pass `provider=ProviderInfo(provider_id=...)`; builtins use `TORCHLENS_PROVIDER` |
| `collapse_floor_fallback` | WARNING code (`TorchLensWarning`): `collapse="auto"`/`"max"` fell back to the conservative floor plan; typed half on `OptimizerResult` (collapse memo D4) | Reduce the rendered graph with `module=` focus, `vis_call_depth`, or rolled mode |
| `collapse_near_uncollapsed` | WARNING code (`TorchLensWarning`): an auto/max plan left >=90% of rendered units visible (collapse memo D4 release invariant) | Reduce the rendered graph with `module=` focus, `vis_call_depth`, or rolled mode |
| `layout_engine_stderr` | WARNING code (`TorchLensWarning`): a zero-exit Graphviz layout wrote to stderr (the cairo clamp class; vizmech D24) | Read `trace._last_render_geometry`; on a size clamp render direct SVG or reduce the graph |
| `lens_member_unknown` | A lens preset declared a member key that is not a draw() parameter name (`InvalidArgumentError`; themes registry, spelling DOCUMENTED-UNSTABLE) | Use draw() parameter names as lens member keys |
| `lens_name_taken` | A lens registration reused an existing (surface, name) key (`InvalidArgumentError`; spelling DOCUMENTED-UNSTABLE) | Pick an unregistered lens name or surface |
| `lens_secondary_undeclared` | A lens marked SECONDARY members it does not set (`InvalidArgumentError`; spelling DOCUMENTED-UNSTABLE) | Declare every SECONDARY member in the lens members mapping |
| `lens_unknown` | A lens lookup named an unregistered row; the refusal lists the roster (`InvalidArgumentError`; spelling DOCUMENTED-UNSTABLE) | Pick a registered lens name |
| `lens_user_kwarg_unknown` | Lens resolution received an explicit kwarg that is not a draw() parameter (`InvalidArgumentError`; spelling DOCUMENTED-UNSTABLE) | Pass draw() parameter names only |
| `rank_render_endpoint_undeclared` | Rank-path DOT referenced an undeclared or unpositioned edge endpoint — a guaranteed `neato -n` hard error, asserted at emit time instead (`RankRenderEndpointError`, `GraphvizRenderError` lineage; vizmech D20) | TorchLens render-pipeline bug: report the model and draw() arguments; render with `vis_node_placement='dot'` meanwhile |
| `relation_assignment_type_invalid` | Finished relation field assigned a non-container | Assign list/set/tuple/frozenset or None |
| `report_subject_unsupported` | `tl.report.explain()` / `Trace.to_agent_json()` received a presenter (`MergedTrace`, `TraceSlice`, `Bundle`, `Recording`) rather than a Trace — a hollow report would answer the wrong question (`InvalidArgumentError`, `ValueError` lineage; spelling DOCUMENTED-UNSTABLE) | Report on the underlying Trace (e.g. a merge member, the slice's parent trace, a bundle member, or `Recording.to_trace()`) |
| `renderer_capability_unsupported` | RenderIR requires a capability its renderer lacks (`UnsupportedRendererCapabilityError`, `RuntimeError` lineage) | Use the graphviz renderer or drop the option needing the capability |
| `run_carry_state_requires_live_model` | `carry_state=True` on a loaded provider, which mutates staged clones and has no live model for state to carry into (PROVISIONAL spelling, documented-unstable) | Drop `carry_state=` on loaded traces, or run the live model |
| `run_fast_carry_state_unsupported` | `carry_state=True` with `fast=True`; fast mode's cached-oracle contract forbids declared-state mutation (PROVISIONAL spelling, documented-unstable) | Drop `carry_state=` or drop `fast=` |
| `run_fast_divergence_policy_invalid` | `fast=True` with a non-raise divergence policy | Use `on_divergence='raise'` or drop `fast=` |
| `run_fast_until_unsupported` | `until=` with `fast=True`; the fast tier compiles the full recorded path (PROVISIONAL spelling, documented-unstable) | Drop `until=` or drop `fast=` |
| `run_until_form_invalid` | `until=`/run-time `save=` run-window selection invalid: a non-string non-predicate form, an empty or unresolvable selection, sites with no producing recorded call, or a `save=` site outside the `until=` executed window (PROVISIONAL spelling, documented-unstable; predicate/selector forms refuse separately via `run_capability_unavailable` at stage `predicate_surface_pending` until the S4 merge) | Pass static layer labels, module addresses, or `'saved'`, inside the executed window |
| `run_input_missing` | Legacy rerun received no forward input | Pass the input as `log.run(model, x)` |
| `run_fast_requires_inputs` | `fast=True` on the legacy run surface | Call `trace.run(inputs=..., fast=True)` |
| `run_legacy_arguments_conflict` | Unified and legacy run arguments were mixed | Pass one input form only |
| `run_legacy_options_conflict` | Legacy and unified run options were mixed in either direction (legacy rerun options on the unified surface, or the unified-only `carry_state=` on the legacy surface) | Drop the mismatched options |
| `run_source_model_collected` | Live model reference is no longer retained | Pass the model to `trace.run(model, input)` |
| `run_staged_spec_unapplied` | `run(inputs=...)` on a trace carrying a USER-staged intervention spec (sticky hooks attached via `attach_hooks()`/the spec door, or staged `set()` value replacements): the unified provider is a fresh verified execution that never installs the staged spec, so running would silently return an un-intervened result. Engine-owned sticky hooks minted by a selection `do()` plan are excluded — that edit was already applied to the trace's saved values, so a new-input run gets the `PendingValueEditsWarning` disclosure instead (D1, never a refusal) (PROVISIONAL spelling, documented-unstable; C03 honesty gate, settled under both OP1 branches) | Apply the spec via the legacy intervened rerun `run(model, x)` or a fresh `tl.trace(..., intervene=...)`, or detach the staged hooks (`clear_hooks()` / `detach_hooks()`) before `run(inputs=...)` |
| `run_state_snapshot_unsupported` | The default live run() could not snapshot declared state before executing (enumeration unavailable, unprovable/overlapping alias topology, or clone/allocation failure); fail-before-execute, no forward ran (`StateBindingError`, `ValueError` lineage; PROVISIONAL spelling, documented-unstable) | Pass `carry_state=True` if you accept declared-state mutation persisting, or fix the named state entry |
| `run_state_restore_failed` | A declared-state restore failed AFTER execution (the live model's state is now unknown), or a later live/fast run() was attempted on a trace carrying that session-scoped state-compromised latch (`StateBindingError`, `ValueError` lineage; PROVISIONAL spelling, documented-unstable) | Reload known-good weights (or re-capture), then run again |
| `output_sink_conflict` | Disk storage and callback sink were both configured | Choose one sink |
| `save_budget_invalid` | `save_budget` is not `'auto'`, a float in `(0, 1]`, an int byte cap, or `None` (`InvalidArgumentError`, `ValueError` lineage) | Pass one of the documented spellings |
| `save_mode_invalid` | Activation save mode is unknown | Choose a documented save mode |
| `state_baseline_unavailable` | An intervention-ready capture needs a byte baseline of the model's state, but pending (un-materialized) lazy parameters or buffers have no bytes at the capture boundary, so no replayable baseline can exist; refuses before any mutation with the pending slots named (`CaptureContextError`; spelling DOCUMENTED-UNSTABLE) | Materialize once outside capture (`with torch.no_grad(): model(x)`), then retry the armed capture |
| `save_predicate_type_invalid` | `save=` is neither SaveOptions, predicate, selector, nor `None` (`save='all'` lands here) | Pass a predicate or SaveOptions; use `layers_to_save='all'` for exhaustive saves |
| `save_payload_level_conflict` | Optional payload family requires runnable level | Use runnable level or omit that family |
| `scale_requires_size_by` | `scale=` was supplied without `size_by`; the scale transform applies to the size channel only. UNSTABLE code (pre-ratification) | Pass `size_by=` alongside `scale=`, or drop `scale=` |
| `selection_alignment_invalid` | `ResolvedSelection.align_to(target)` cannot bridge this selection cross-run: `exc.fields["reason"]` carries the closed reason (`kind_unsupported` / `grouping_stamp_degraded` / `grouping_stamp_mismatch` / `site_key_unavailable` / `site_not_in_target` / `index_space_mismatch`) (L6 stage 4a; same-policy captures only per the L1 cross-stamp rule; spelling DOCUMENTED-UNSTABLE) | Align live same-policy captures of one architecture; re-resolve the query Selection on the target trace otherwise |
| `selection_apply_invalid` | `do(selection, edit)` got an edit outside the closed edit surface, or an edit/selection combination the engine cannot apply (L6; spelling DOCUMENTED-UNSTABLE) | Pass a `tl.Edit` helper (`zero_ablate`, `scale`, `patch_from`, ...) legal for the selection's kind |
| `selection_bool_ambiguous` | `bool()` on an unresolved Selection QUERY; emptiness is an element-level property of the RESOLVED selection (L6 two-level denotation; spelling DOCUMENTED-UNSTABLE) | Call `selection.resolve(trace)` and test the resolved selection |
| `selection_kind_incompatible` | Composition or application mixes the closed selection kinds (ACT/PARAM/EDGE) (L6; spelling DOCUMENTED-UNSTABLE) | Compose selections of one kind; resolve and apply each kind separately |
| `selection_trace_mismatch` | A ResolvedSelection is used against a different trace than the one it was resolved on (session-only binding; spelling DOCUMENTED-UNSTABLE) | Re-resolve the query Selection on the target trace |
| `sidecar_absent` | `read_sidecar()` on a trace that carries no envelope for that family (C01 sidecar seam; spelling DOCUMENTED-UNSTABLE) | Attach the sidecar first, or read a present family |
| `sidecar_budget_exceeded` | Sidecar payload is over the family's declared size budget (spelling DOCUMENTED-UNSTABLE) | Store digests/references, or register a larger `size_budget_bytes` |
| `sidecar_envelope_invalid` | Persisted sidecar envelope is not a self-describing mapping with a payload (spelling DOCUMENTED-UNSTABLE) | Re-attach through `attach_sidecar()`; never hand-edit envelopes |
| `sidecar_family_id_invalid` | Sidecar family id is not namespaced `<owner_ns>.<name>` (spelling DOCUMENTED-UNSTABLE) | Use an id like `myorg.saliency` |
| `sidecar_family_type_invalid` | `register_sidecar_family()` received something other than a `SidecarFamily` (spelling DOCUMENTED-UNSTABLE) | Construct `torchlens.io.SidecarFamily(...)` |
| `sidecar_payload_invalid` | Sidecar payload is not JSON-portable (spelling DOCUMENTED-UNSTABLE) | Pass JSON-serializable payload data |
| `sidecar_schema_invalid` | Sidecar family declares an invalid schema id, version, or size budget (spelling DOCUMENTED-UNSTABLE) | Declare a non-empty `schema_id`, `version >= 1`, positive budget |
| `sidecar_trace_invalid` | Sidecar attach/read target has no annotations mapping (spelling DOCUMENTED-UNSTABLE) | Pass a captured or loaded torchlens Trace |
| `sidecar_version_unsupported` | Sidecar envelope declares a version newer than the registered family; the artifact is valid, the provider too old (spelling DOCUMENTED-UNSTABLE) | Upgrade the provider that owns the family |
| `shard_local_persistence_unsupported` | Ordinary save of a trace carrying the shard-local capture marker (`distributed_scope == "rank_local_shard"`) while the active schema does not persist the disclosure — the PERMANENT erasure-prevention invariant at the bundle-save chokepoint: a shard-local trace must never reload as a plain trace (L8/F6; spelling DOCUMENTED-UNSTABLE). `Trace.distributed_scope` persists as of the tlspec v8 bump, so ordinary saves proceed; the refusal survives as a schema-regression tripwire | Under a regressed schema, analyze in-session; on v8+ the save proceeds |
| `selection_unresolvable` | `selection.resolve(trace)` cannot denote the query on this trace; `exc.fields["reason"]` carries the closed reason (L6; spelling DOCUMENTED-UNSTABLE) | Branch on the structured reason: re-capture with the required evidence or narrow the query |
| `selector_function_pattern_type_invalid` | `func()` pattern is not a string | Pass a function-name string |
| `selector_subtraction_operand_invalid` | `selector - other` with a non-selector right operand; subtraction desugars to `and(a, not(b))` over selectors only (L6; spelling DOCUMENTED-UNSTABLE) | Subtract a selector, or build the composition from predicate terms |
| `skip_fn_boundary_invalid` | `skip_fn` tried to skip an input or output layer | Return False for boundary layers |
| `slice_save_unsupported` | `tl.save()` received a `TraceSlice`: a slice is a presenter over one trace's sub-DAG with declared dangling boundary edges, never a self-contained capture, so persisting it would imply replay/validation capabilities it cannot honour (L6 graph slice; spelling DOCUMENTED-UNSTABLE) | Save the underlying trace (`tl.save(slice.source_trace, path)`) and re-derive the view after loading |
| `site_selector_empty` | `tl.site()` received neither a rendered key nor any component filter (PROVISIONAL spelling, documented-unstable; C03 structural selector) | Pass `op.site_key`, or components such as `tl.site(module_path=..., op_type=...)` |
| `site_selector_key_conflict` | `tl.site(key)` combined with component filters; the rendered key already fixes every component (PROVISIONAL spelling, documented-unstable) | Pass either the rendered key or components, not both |
| `site_selector_key_invalid` | `tl.site(key)` received a malformed `site_key_v1` string (PROVISIONAL spelling, documented-unstable) | Pass a rendered `op.site_key` string |
| `spec_door_extra_arguments` | An intervention door received an `InterventionSpec` PLUS extra hook/edit/direction arguments; the spec already carries site, action, and direction (PROVISIONAL spelling, documented-unstable; C03 spec doors) | Pass only the spec, or use the `(site, hook)` call shape without a spec |
| `spec_format_version_unsupported` | Intervention `.tlspec` format version is unknown | Use a supported format version |
| `summary_execution_mode_invalid` | One-call `tl.summary` `execution_mode` outside `'eval'`/`'train'`/`'same'` | Use `'eval'` (safe default; modes restored), `'train'` (explicit opt-in), or `'same'` |
| `spec_rule_unreplayable` | A recorded-graph door (`do`/`attach_hooks`) received a spec containing a runtime-only predicate rule; value-dependent tests are valid only where something really runs, and no lane silently drops a rule (PROVISIONAL spelling, documented-unstable; refused rules are named) | Re-express the WHERE term as a structural or recorded selector, or run the spec via `tl.trace(model, x, intervene=spec)` |
| `spec_rules_duplicate` | Two `InterventionSpec` clauses carry the same WHERE, action, and direction (PROVISIONAL spelling, documented-unstable) | Drop the duplicate clause |
| `spec_rules_overlap` | More than one `InterventionSpec` rule matched one op at fire time; a spec never implicitly chains two edits at one site (PROVISIONAL spelling, documented-unstable) | Compose the actions explicitly with `tl.compose(...)`, or narrow the WHERE terms |
| `summary_fields_invalid` | Summary field names are unknown | Pass documented summary fields |
| `summary_grad_mode_invalid` | One-call `tl.summary` `grad_mode` outside `'off'`/`'same'` | Use `'off'` (default; forward runs under `torch.no_grad()`) or `'same'` |
| `summary_level_invalid` | Summary level is unknown | Pass a documented summary level |
| `summary_option_conflict` | Aliased summary options disagree | Pass one alias, or equal values |
| `site_key_unavailable` | Site accessor read on a trace without site keys (legacy artifact or detached layer; spelling provisional pending the S2 vocabulary amendment) | Re-capture with a current TorchLens to mint site keys |
| `size_by_rolled_varying` | The `size_by` source cannot be certified single-valued on a rolled multi-pass node (marker-varying reconciled field, unreconciled per-call/first-pass projection, or the varying output shape under `"dims"`); size REFUSES where color degrades — an unencoded box is indistinguishable from an encoded small box. UNSTABLE code (pre-ratification) | Unroll the graph to size each pass by its own value, choose a cross-pass total (`total_*`), or pass a callable asserting your own aggregate semantics |
| `stack_by_auto_underivable` | `stack_by=True/"auto"` on a trace with no multi-pass layers, or whose multi-pass execution order is not globally monotone (chained loops, late-resumption skips, non-monotone nested tallies) — no derivable ground truth for "same column = same execution window". UNSTABLE code (pre-ratification) | Pass an explicit annotation (`stack_by=<field>` or a callable) naming your own stacking semantics |
| `stack_by_requires_unrolled` | `stack_by` on a rolled graph: one aggregate node per layer leaves no per-pass nodes to stack. UNSTABLE code (pre-ratification) | Pass `vis_mode='unrolled'` (the default), or drop `stack_by` |
| `stack_ordinals_duplicate` | Stacked ops share an execution ordinal | Narrow the selector to distinct ops |
| `stack_ordinals_unavailable` | Matched ops lack recorded execution ordinals | Select ops with recorded ordinals |
| `stack_output_not_tensor` | Stacked op's saved primary out is not one tensor | Select single-tensor-output ops |
| `stack_selector_no_match` | `trace.stack` selector matched no sites | Pass a selector matching saved ops |
| `stack_shape_mismatch` | Stacked outputs have different shapes | Select ops with identical output shapes |
| `module_filter_zero_saved` | WARNING (S-18 contract, not a refusal): the `module_filter` third save gate suppressed every payload the save selection picked; the trace has metadata but zero saved activations | Write the filter against op-record fields (its argument is an op-record namespace, never an `nn.Module`), or drop `module_filter` |
| `stop_after_chunked_conflict` | `stop_after=` with `chunk_size`: the chunk fan-out runs several captures and one stop frontier is ambiguous across them | Drop `chunk_size` or drop `stop_after` |
| `stop_after_halt_conflict` | `stop_after=` and `halt=` were both configured; they compile into one stop-directive slot | Pass one stop directive: keep `halt=` (compose predicates with `&`/`|`) or keep `stop_after=` |
| `stop_after_never_fired` | A selector-shaped `stop_after=` site never fired: the capture ran the full forward, so the result is not the requested frontier | Check the site against the model's module addresses (`model.named_modules()`) or use `tl.func(...)`/`tl.module(...)`; exploratory predicates that may never fire belong in a callable |
| `stop_after_never_fired_callable` | WARNING (S-18 contract, not a refusal): a callable or ambient `stop_after` site never fired; the capture ran the full forward | Check the stop_after site against the executed model, or drop stop_after |
| `stop_after_site_not_live` | `stop_after=` received a finalized postprocess label; stop_after runs live during capture, before finalized labels exist | Pass a module address string, a live selector (`tl.func`/`tl.module`), or a callable predicate |
| `stop_after_type_invalid` | `stop_after=` received an unsupported type | Pass a module address string, a `tl.*` selector, or a callable predicate |
| `storage_argument_conflict` | `storage` and `streaming` were both supplied | Prefer `storage`, or remove it |
| `structural_hash_mismatch` | Model structural hash differs from the pinned value (`StructuralHashMismatchError`, `AssertionError` lineage) | Inspect the captured traces for the divergence, or re-pin if intentional |
| `structure_only_backward_unsupported` | Backward/gradient capture is refused on a structure-only trace (v1) | Run a real capture for backward surfaces |
| `structure_only_discharge_precondition` | `discharge_against` preconditions unmet (not a structure-only trace, oracle not ordinary, or oracle not COMPLETE) | Discharge a structure-only trace against a settled COMPLETE real capture |
| `structure_only_episode_unsupported` | Episode capture does not compose with structure-only (v1; S2-amendment candidate) | Run the episode capture without `structure_only` |
| `structure_only_option_conflict` | `structure_only=True` combined with an option that needs tensor values (`raise_on_nan`, `intervention_ready`, a not-provably-value-free `halt=`) | Drop the conflicting option or run a real capture |
| `structure_only_refuted_hypothesis` | A registered real-run discharge REFUTED this structure-only trace's hypotheses; hypothesis consumers refuse | Re-capture after fixing the divergence, or consume the discharge record directly |
| `structure_only_replay_unsupported` | Replay/run requires tensor values a structure-only trace never records (v1 floor; the declared late-bind posture is the row's named flip event) | Run the real model, or corroborate via `Trace.discharge_against(real_trace)` |
| `structure_only_runnable_unsupported` | Runnable save is refused on a structure-only trace (v1 floor; declared late-bind slots arrive with the L7b S2 StateSource amendment) | Run a real capture with `intervention_ready=True` for runnable artifacts |
| `structure_only_type_invalid` | Capture option `structure_only` is not a bool | Pass `structure_only=True` or `structure_only=False` |
| `structure_only_validation_unsupported` | Validation entry has nothing to compare against on a structure-only trace; discharge is the verification story | Use `Trace.discharge_against(real_trace)` instead |
| `structure_only_values_unsupported` | A value-payload request (save selection, gradients, streaming sinks, raw input/output, output decode) cannot be honored under `structure_only=True` | Drop the payload-requesting option or run a real capture |
| `sweep_intervention_conflict` | `sweep()` received a second intervention | Express the target through `at` |
| `sweep_names_length_mismatch` | Sweep names and values have different lengths | Pass one name per value |
| `sweep_site_missing` | Sweep site target is missing | Pass `at` |
| `sweep_spec_at_conflict` | `sweep(at=...)` combined with `InterventionSpec` values; each spec already names its own WHERE and action (PROVISIONAL spelling, documented-unstable; C03 spec door) | Drop `at=` when sweeping over specs |
| `sweep_spec_values_mixed` | Sweep values mixed `InterventionSpec` and plain replacement values (PROVISIONAL spelling, documented-unstable) | Pass all-spec values without `at=`, or all-plain values with `at=` |
| `sweep_site_type_invalid` | Sweep site target has an unsupported type | Pass a label, selector, or predicate |
| `sweep_values_empty` | Sweep values iterable is empty | Pass at least one value |
| `sweep_values_missing` | Sweep values iterable is missing | Pass a non-empty iterable |
| `tensor_connection_labels_missing` | Manual edge endpoint lacks a capture label | Use tensors already captured in the active trace |
| `tlspec_format_markers_incoherent` | Artifact manifest carries `kind` without `tlspec_version` yet has no `spec.json` — format markers are incoherent (`TorchLensIOError`) | Restore the manifest's `tlspec_version` key or re-save the artifact from its source trace |
| `top_n_invalid` | Requested `top_n` is below one | Pass a positive integer |
| `trace_cleaned_up` | Public read on a Trace that `cleanup()` husked (`TraceCleanedUpError`, `AttributeError` lineage) | Re-capture with `tl.trace(...)`; cleanup permanently empties a Trace |
| `trace_not_finished` | Export requested before the forward pass finished | Wait until `trace(...)` has returned |
| `trace_reference_collected` | Owning Trace was garbage-collected | Keep the Trace alive while reading records |
| `transform_axis_roles_unavailable` | A semantic role (e.g. `"token"`) has no recorded axis-role declaration at the tensor's rank; TorchLens never guesses axis semantics from rank (transforms memo decision 11; spelling DOCUMENTED-UNSTABLE) | Declare axis roles (e.g. `TransformContext(roles=tl.transforms.BTD)`) or use the explicit-axis primitive |
| `transform_builtin_shadowed` | A registration names a TorchLens builtin transform; builtin names are a CLOSED SET and non-replaceable (spelling DOCUMENTED-UNSTABLE) | Pick a distinct name for the custom transform |
| `transform_coercion_invalid` | The transform slot got an uncoercible value: not `None`/callable/registered name/spec/ordered sequence, an UNORDERED container (a set has no left or right, T-C8), a Mapping at the single-chain door, or a non-callable at `with_context` (spelling DOCUMENTED-UNSTABLE) | Pass a callable, a `tl.transforms` constructor result, a registered name, or an ordered list of them |
| `transform_mapping_key_unknown` | A per-site transform Mapping names an output key the run does not have; a typo'd site name silently untransformed is a fails-open bug (spelling DOCUMENTED-UNSTABLE) | Key the Mapping by the run's output keys (plus `tl.transforms.DEFAULT` for the fallback chain) |
| `transform_name_not_preset` | A bare string named a registered transform that takes parameters; strings resolve only to zero-parameter deterministic presets, never a parameter mini-language (spelling DOCUMENTED-UNSTABLE) | Call the constructor instead (e.g. `tl.transforms.cast(...)`) |
| `transform_name_unknown` | A string in the transform slot names no registered transform; strings are never import paths (spelling DOCUMENTED-UNSTABLE) | Use a builtin name, register the transform, or pass the callable itself |
| `transform_not_differentiable` | An out/grad/activation transform returned a non-tensor, non-grad dtype, or graph-disconnected value while `backward_ready=True`/`keep_grad=True` (`TrainingModeConfigError`, `ValueError` lineage) | Return a differentiable floating-dtype tensor that stays on the autograd graph |
| `transform_opaque_unresolvable` | An opaque transform step rehydrated from an artifact record was applied; identification is a disclosure, not code — artifact loading never imports or executes user code (spelling DOCUMENTED-UNSTABLE) | Register the transform under a versioned name and rebuild the chain from names, or pass the live callable |
| `transform_output_invalid` | A transform step returned a non-tensor at the engine boundary (spelling DOCUMENTED-UNSTABLE) | Return a `torch.Tensor` from every transform step |
| `transform_params_invalid` | Transform params fail their kernel's validation: non-canonical-JSON values, a non-float cast target (T-C7), an op outside the closed reduce vocabulary, a literal stimulus-axis (`0`) address (T-C2), an invalid seed/seed_source pairing (T-C4), or smuggled unvalidated params on a hand-built spec (spelling DOCUMENTED-UNSTABLE) | Fix the named parameter per the message |
| `transform_pipeline_record_invalid` | A persisted `tl_transform_pipeline_v1` record is malformed (wrong schema id, missing steps array, non-mapping rows, unknown step kind) (spelling DOCUMENTED-UNSTABLE) | Pass the manifest's recorded transform_pipeline block unmodified |
| `transform_plan_invalid` | A frozen transform step cannot plan against the concrete input: axis out of range or resolving to the stimulus axis, rank too low, or an index out of the declared bound (T-C6; spelling DOCUMENTED-UNSTABLE) | Address a per-stimulus axis/index that exists at the input's rank |
| `transform_plan_violated` | A shard's observed post-transform output contradicts the batch-zero FROZEN plan (shape or dtype); the shard is refused BEFORE publication — ragged shape is validated per shard, never inferred forever from batch one (T-C6; spelling DOCUMENTED-UNSTABLE) | Keep per-stimulus shapes fixed across batches (pad or pool before the chain), or re-extract into a fresh directory |
| `transform_role_declaration_invalid` | An axis-role declaration is empty or holds non-string roles (spelling DOCUMENTED-UNSTABLE) | Pass one role name per tensor axis, e.g. `('batch', 'token', 'feature')` |
| `transform_role_evidence_invalid` | Axis-role evidence is outside the closed `declared` / `op_contract` / `unknown` vocabulary; `inferred` is RESERVED — rank is never evidence for axis semantics (spelling DOCUMENTED-UNSTABLE) | Declare roles with `evidence='declared'` or leave roles undeclared |
| `transform_row_axis_violated` | A transform step changed the stimulus axis (row count/order); T-C2 is enforced at the engine for built-ins AND raw callables, and the ledger row count derives from the input batch (spelling DOCUMENTED-UNSTABLE) | Reduce over non-batch axes only (e.g. `tl.transforms.reduce` with `axis>=1`), or drop the offending step |
| `transform_version_mismatch` | A frozen spec's algorithm version differs from the installed registration; a version change is numerics-visible and never silently substituted (T-C13; spelling DOCUMENTED-UNSTABLE) | Install the matching transform version, or rebuild the spec against the installed one |
| `unknown_backend` | Explicit backend name is not registered | Choose a registered backend |
| `unsupported_tensor_variant` | Model/input carries meta, fake, functional, or sparse tensor variants (`UnsupportedTensorVariantError`) | Materialize dense, strided tensors with concrete shapes on a real device |
| `value_dependent_branch_unsupported` | A tensor-value escape reached user code during structure-only capture (`ValueDependentBranchError`; device-neutral, exact source line) | Run a real capture, or restructure the branch to be shape-derived |
| `visualization_bool_option_invalid` | A bool-only draw/visualization option received a non-bool (strings such as `'no'` were silently truthy) | Pass True or False |
| `visualization_show_containers_invalid` | `show_containers` is outside its closed vocabulary | Pass False or one of `labels`, `cluster`, `collapsed`, `auto`, `nodes` |
| `visualization_intervention_mode_invalid` | Intervention rendering mode is unknown | Choose `node_mark` or `as_node` |
| `visualization_layout_invalid` | Visualization layout is unknown | Choose `auto`, `dot`, or `rank` |
| `visualization_direction_invalid` | Render direction is unknown | Choose `bottomup`, `topdown`, or `leftright` |
| `visualization_renderer_invalid` | Visualization renderer is unknown | Choose `graphviz` or `dagua` |
| `visualization_theme_invalid` | Visualization theme is unknown | Choose a supported theme |
| `visualization_mode_invalid` | Backend visualization mode is unsupported | Choose a backend-supported mode |
| `visualization_node_style_invalid` | Node style is unknown | Choose a documented style |
| `watch_lifecycle_invalid` | Collector lifecycle misuse: plan/attach before discover, or double attach (`WatchLifecycleError`) | Follow discover -> attach -> step -> detach |
| `watch_plan_empty` | The watch site selection matched nothing; zero-match selectors refuse at plan time (`WatchPlanError`) | Pass sites= matching module paths or parameter names, or sites=None for leaf modules |
| `watch_plan_invalid` | Un-collectable watch request: a stream outside Route A, an unknown cadence stream, or param streams without the optimizer boundary (`WatchPlanError`) | Use Route-A streams and pass attach(optimizer=...) for parameter streams |
| `watch_step_conflict` | A step transaction is already open; steps never nest (`WatchLifecycleError`) | Close the open step before starting the next |
| `wrappers_removed_before_capture` | A concurrent `unwrap_torch()` removed the torch wrappers between model preparation and capture admission | Do not call `unwrap_torch()` concurrently with capture entry; re-run `tl.trace` to re-install the wrappers |

## Constant-spelled refusal kinds

These refusals identify themselves on `exc.fields["kind"]` (one
entry per structured finding) rather than `exc.fields["code"]`, and their identifier
strings are spelled as module-level constants rather than inline `code="..."`
literals. They are part of the same stable public vocabulary: branch on the kind
string, never on message text. The lockstep gate enrolls each constant explicitly
(`tests/test_error_contract_lockstep.py`), so renaming the constant or drifting its
string value fails the gate exactly like an inline code.

| Kind | Refusal | Remedy class |
|---|---|---|
| `ambiguous_group_lifetime` | A collective used a process group whose pre-arming lifetime cannot be proven | Call `tl.distributed.arm()` at process start, before any group is created |
| `uncaptured_collective_op` | Arm-time recognizer set-inequality or dispatcher schema scan found a collective the wraps would not capture | Upgrade TorchLens to a build whose recognizer covers the installed torch, or avoid the unrecognized collective in the traced forward |
| `wildcard_recv_unsupported` | A point-to-point receive from `ANY_SOURCE` cannot be attributed to a sender | Pass an explicit source rank to `recv`/`irecv` |
| `intervention_fire_results_unrecordable` | An intervention changed execution but its tensor accepts neither transient metadata nor storage-backed fire-result evidence | Intervene on ordinary tensor outputs, or drop the intervention for this op |
| `intervention_fire_results_cleanup_failed` | Intervention fire metadata could not be cleared after consumption; refusing prevents stale evidence entering a later capture | Re-run the capture; report the tensor type if it recurs |
| `rerun_zero_fire` | WARNING (S-18 contract, not a refusal): a rerun hook plan entry fired at zero sites -- the rerun completed but those interventions were silent no-ops | Resolve the target sites against the rerun trace (`trace.resolve_sites`) before re-applying, or route the edit through the push engine (`fork().do(...)`), which validates sites at plan time |
| `runnable_random_init_run` | WARNING (S-18 contract, not a refusal): a weight-free runnable artifact is executing on RANDOM role-init state -- outputs come from a freshly initialized model, not the captured one | Re-save with `tl.save(trace, path, level='runnable', include_weights=True)`, or bind real weights with `trace.load_state_dict(state_dict)` before `run()` |
| `patch_campaign_all_identical` | WARNING (S-18 contract, not a refusal): an activation-patching campaign's every fire replaced the site value with an identical tensor, so the table is guaranteed to equal the corrupted baseline | Inspect `tl.facets.facet_coverage(trace)` and re-anchor the facet before publishing the table as a null result |
| `extraction_position_ids_derived` | WARNING (S-18 contract, not a refusal): `extract_dataset` derived `position_ids` from a non-right-aligned attention mask and passed them to the model's forward (disclosed once per run; exact for absolute-position models, a no-op for rotary models) | Right-pad the batch or pass explicit `position_ids` to silence the derivation |
| `annotation_sweep_sites_skipped` | WARNING (S-18 contract, not a refusal): a default `*_evolution` sweep skipped saved sites that are not stimulus-indexed (buffer overwrites, leading-axis mismatches) instead of fabricating stimulus-space results from them | Select stimulus-indexed sites explicitly; an explicit `save=` selection of a skipped site raises the full teaching refusal |
| `option_receipt_not_options` | `option_receipt()` was passed something other than a constructed TorchLens options object | Pass a constructed options dataclass such as `tl.options.CaptureOptions(...)` |
| `option_receipt_unknown_field` | An `option_receipt()` adjustment named a field the options class does not have | Use public field names of the passed options class as adjustment keys |
| `option_receipt_reason_invalid` | An `option_receipt()` adjustment carried a reason outside the closed vocabulary | Adjustment reasons are `forced`, `normalized`, or `refused`; `explicit` and `default` are derived, never supplied |

The related `group_lifetime_evidence_conflict` kind is governed by the merged-trace
contract (`docs/reference/merged_trace_contract.md`), where it is also a
`MergedErrorCode` member.

Adding or renaming a code is a public vocabulary change and must update this table and the
corresponding typed-door test in the same change. Constant-spelled kinds must additionally
update the enrollment table in `tests/test_error_contract_lockstep.py`.
