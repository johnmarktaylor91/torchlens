# TorchLens glossary

This glossary fixes the public meaning of TorchLens terms. It describes the current 2.x API;
removed spellings are listed separately in [Deprecations](deprecations.md).

## Capture products

**Trace**
: The finalized record of one executed path: operations, dataflow edges, module context, selected
  payloads, and capture metadata. `tl.trace(...)` returns a `Trace`.

**Trace.provenance / Trace.options / Trace.cost / Trace.source / Trace.health / Trace.graph / Trace.autograd**
: Read-only concern groups of a `Trace`; every persisted capture fact lives in exactly one group and
  is stored privately as `_<group>_<leaf>` (tlspec v11). `trace.provenance` says how, when and from
  what the capture happened (`backend`, `backend_version`, `backend_config`, `kind`, `start_time`,
  `end_time`, `duration`, `setup_duration`, `cleanup_duration`, `verified`, `verification_reason`,
  `cache_hit`, `cache_key`, `cache_path`, `recording_kept`, `fingerprint`, `input_signature_hash`,
  `input_structure`, `input_object_id`, `model_object_id`, `param_hash_quick`, `param_hash_full`,
  `param_source`, `root_entry_point`, `owner_thread_id`, `thread_count_start`, `thread_count_end`,
  `thread_activity_detected`, `distributed_scope`, `observer_spans`, `is_appended`, ...) and
  the capture-provenance text block keeps an interim private spelling
  (`trace.provenance._summary()`) until its public name is decided; `trace.options` is the
  stored snapshot of the capture options (`mode`, `save`, `save_grads`, `save_mode`, `save_args`,
  `save_raw_input`, `save_transformed_output`, `save_rng`, `save_source_context`,
  `source_context_lines`, `backward_ready`, `inference_only`, `structure_only`, `grouping`,
  `grouping_policy`, `seed`, `on_nonfinite`, `input_transform`, `activation_transform`,
  `grad_transform`, `output_device`, `verbose`, `chunk_size`, ...); `trace.cost` holds compute,
  memory and timing totals (`total_flops`, `total_macs`, `macs_by_kind`, `forward_duration`,
  `backward_durations`, `func_calls_duration`, `overhead_duration`, `forward_peak_memory`,
  `total_activation_memory`, `total_param_memory`, `total_grad_memory`, `saved_grad_memory`, ...);
  `trace.source` holds the model source facts (`class_name`, `class_qualname`, `class_file`,
  `class_line`, `class_docstring`, `init_signature`, `forward_signature`, `frames`, `snapshots`,
  `class_location`, `forward_location`, `init_location`, `cls`); `trace.health` holds the honesty
  diagnostics (`completeness_verified`, `completeness_mode`, `num_witness_events`,
  `num_accounted_witness_events`, `num_unaccounted_witness_events`, `escape_detector_verified`,
  `escape_detector_mode`, `num_escape_detector_events`, ...); `trace.graph` holds graph-shape
  facts (`edges`, `is_recurrent`, `is_dynamic`, `conditionals`, `conditional_records`,
  `stop_frontier`, `op_equivalence_classes`, `relationship_evidence`, `structure_evidence`,
  `manual_tensor_connections`, `orphans`, `tied_param_groups`, ...); `trace.autograd` holds the
  backward records (`backward_passes`, `grad_fns`, `grad_fn_calls`, `grad_fn_order`,
  `has_backward_pass`, `has_grads`, `derived_grads`, `backward_root_grad_fn_object_ids`, ...).

**Trace.params filters and totals**
: `trace.params.trainable`, `.frozen`, `.executed` and `.unexecuted` are parameter collections;
  `len()` counts their Param records and `.numel` adds up their elements (ties counted once);
  `trace.params.numel_by_path` counts ties once per module path; `.labels` gives the addresses
  (`trace.params.unexecuted.labels`).

**Layer filters**
: `trace.layers.compute`, `.recurrent`, `.saved_grad`, `.internal_source` and `.internal_sink`
  are Layer collections; on any Layer collection they keep the members that pass the Trace-level
  filter, so `len(trace.layers.compute)` counts the compute Layers.

**Op filters**
: `trace.ops.compute` and `trace.ops.nonfinite` are Op collections; on any Op collection they
  keep the members that pass the Trace-level filter (`Trace.compute_ops`, `Trace.nonfinite_ops`),
  so `len(trace.ops.compute)` counts the compute ops and `trace.saved_ops.nonfinite` the saved
  ops whose output held NaN or Inf.

**Module filters**
: `trace.modules.uncalled` lists the addresses `Trace.uncalled_modules` records; an uncalled
  module has no Module record, so the answer is addresses, not records.

**Collection labels (`.labels`)**
: Every collection serves `.labels`, the tuple of its lookup labels in collection order:
  `trace.layers.labels` the Layer labels, `trace.ops.labels` the counted Op labels,
  `trace.modules.labels` the module addresses, `trace.params.labels` the parameter addresses, and
  the same on any filtered collection (`trace.layers.compute.labels`). The per-noun Trace
  projections (`Trace.layer_labels`) stay.

**Recording**
: A sparse, torch-only event stream returned by `tl.record(model, inputs, save=...)`.
  `Recording.to_trace()` materializes full graph structure; reading a payload that was not selected
  by `save=` fails explicitly. `tl.record` installs gradient hooks only when `backward_ready=True`,
  `save_grads=` or `default_grad=` is given; without one, `Recording.backward()` refuses with
  code `recording_backward_not_requested`. `tl.record` and `Recorder` build it; `Recording` has no
  public constructor. `tl.record` always returns one object.

**Trace.output / Recording.output**
: The model's return value exactly as the model returned it (tensor, tuple, dict, Hugging Face
  output object, distribution); for a multi-pass `Recorder`, the last pass's. With
  `generate_kwargs=`, `GenerationRecording.output` (and its `recording.output`) is what
  `generate()` returned. `Trace.output` is the exact return from a live `tl.trace` forward,
  independent of `save=` and output transforms; it is neither transformed `raw_output` nor
  the saved-payload rebuild from `reconstruct_output()`. Traces cooked from recordings,
  forked, or merged from chunks have no live forward return. The trace or recording keeps
  that object alive for as long as it exists (for a Hugging Face model this includes
  `past_key_values`); the object is released when the trace or recording is dropped. Held in
  memory only: a trace or recording loaded from disk or unpickled refuses
  with `output_not_stored`, a failed partial recording (`on_forward_error="return_partial"`)
  with `output_forward_failed`, and a capture that `stop_after=` ended before the model
  returned with `output_capture_stopped` (all `PayloadUnavailableError`); it never returns
  `None` for a missing output.

**Recording.options / RecordingSettings**
: `Recording.options` returns a `tl.types.RecordingSettings`, the read-only record of the settings
  the recording was made with, across every door that configured it: the capture knobs from
  `capture=CaptureOptions(...)` (`RecordingSettings.save_grads`,
  `RecordingSettings.activation_transform`, `RecordingSettings.grad_transform`,
  `RecordingSettings.save_raw_activations`, `RecordingSettings.save_raw_grads`,
  `RecordingSettings.detach_saved_activations`, `RecordingSettings.lookback`,
  `RecordingSettings.lookback_payload_policy`, `RecordingSettings.seed`), the recorder knobs from
  `recording=RecordingOptions(...)` (`RecordingSettings.default_op`,
  `RecordingSettings.default_module`, `RecordingSettings.default_grad`,
  `RecordingSettings.history_size`, `RecordingSettings.include_source_events`,
  `RecordingSettings.max_predicate_failures`, `RecordingSettings.on_predicate_error`,
  `RecordingSettings.on_forward_error`, `RecordingSettings.postprocess`) and the door keywords
  (`RecordingSettings.keep_op` from `save=`, `RecordingSettings.intervene` from `intervene=`,
  `RecordingSettings.stop_after`, `RecordingSettings.storage`, `RecordingSettings.narrate`). Values are the live objects the
  recorder used, never repr strings; there is no public constructor. A recording recovered from
  disk keeps the plain values (bools, numbers, strings); callables and objects are not stored with
  saved recordings and read `None` there.

**Saved-activation autograd history** *(knob spellings documented-unstable)*
: `CaptureOptions(detach_saved_activations=False)`, the `tl.trace` default, keeps every saved
  payload graph-connected on every save path: the save-all default, `tl.module` and
  `tl.in_module` selectors, op-label `save=`, the legacy `trace.run(model, x)` rerun,
  `trace.run(inputs=...)` and `trace.run(inputs=..., fast=True)`; `True` detaches on every one
  of them. `tl.extract(..., detach_saved_activations=False)` and
  `tl.record(..., capture=CaptureOptions(detach_saved_activations=False))` (also
  `tl.recording.Recorder`) attach on request; both keep their detached
  default, and `tl.record` refuses an explicit `True` with `backward_ready=True`
  (`backward_ready_conflict`). A per-site `CaptureSpec(save_grad=True)` stays attached either
  way, and disk payloads are always detached. An escrowed deferred-selector payload spilled to
  temporary disk comes back detached, with a `RuntimeWarning` naming the count.
  `Trace.cost.pinned_autograd_memory` discloses what the attached payloads keep alive: the bytes of
  the tensors autograd saved for every op upstream of a saved site, once per storage,
  excluding Parameters and views of them. It reads `Bytes(0)` when every payload is detached
  or the forward ran without grad, and `None` on a loaded trace (session-time, never saved).

**PartialTrace**
: A recoverable prefix of a failed `tl.trace(...)` capture. A raised capture exception may expose
  `exc.partial_trace`: a forward that raises during capture raises `TraceCaptureError` carrying
  it, and TorchLens's own errors carry it too. A partial trace is not silently treated as
  complete.

**Failed partial Recording**
: A `Recording` with `status="partial_error"` and `is_failed` `True`. Use
  `on_forward_error="attach_partial"` to attach it as `exc.partial_recording`, or
  `on_forward_error="return_partial"` to return it. It carries string-only error metadata,
  `num_committed_ops`, and best-effort `last_event_*` fields. It cannot be passed to
  `Recording.to_trace()` or `Recording.backward()`.

**Generation recording** *(spellings documented-unstable)*
: `tl.record(model, input_ids, save=..., generate_kwargs={...})` runs the model's own Hugging Face
  `generate(*inputs, **input_kwargs, **generate_kwargs)` once, with its KV cache, and records the
  selected sites at every forward it runs, so each decode step records only its new position and
  the cost grows linearly with the number of generated tokens. It returns a
  `GenerationRecording`: `.recording` (the multi-pass `Recording`, one pass per forward),
  `.steps` (one `GenerationStep` per forward: `step`, 0 for the prefill; `pass_index`; `kind`;
  `positions`, the absolute positions it computed; `n_ops`, `n_records`, `saved_bytes`),
  `.sequences`, `.prompt_length`, `.settings` (the settings that decide the tokens, the torch seed
  and an RNG-state digest), `.not_recorded` (cache tensors, the sampling between forwards, the
  last token's own position), `.n_steps`, `.records_at(step)` and `.step_of_position(position)`
  (both refuse a missing step with `record_generation_step_missing`). `seed=` seeds once
  for the whole generation. Beam search, more than one return sequence, batched prompts, assisted
  decoding, encoder-decoder models, `use_cache=False`, a generate that runs decode steps without a
  cache, `intervene=`, `stop_after=` and partial forward-error modes refuse with
  `record_generation_unsupported`, whose `reason` field names what is unsupported.

**Capture outcome**
: The frozen `Trace.outcome`, `Recording.outcome`, or `PartialTrace.outcome`. Its status is one of
  `COMPLETE`, `STOPPED`, `ABORTED_NONFINITE`, `FAILED`, `UNATTESTED`, or `UNKNOWN`; failed outcomes
  also identify the phase. See [Capture outcomes](capture_outcomes.md).

**Inference-only capture**
: A capture whose forward ran under `torch.no_grad()`, so it built no autograd graph. Request it
  with `CaptureOptions(inference_only=True)`; a `tl.trace` or `tl.record` entered under
  `torch.no_grad()` or `torch.inference_mode()` implies it unless `backward_ready`, `save_grads`
  (or `default_grad` on `tl.record`, `replayable` on `tl.trace`) asks for a backward, and
  an explicit `inference_only=False` keeps it off. The implied flag holds only when the forward
  really discarded its graph: a forward that turns grad back on (`with torch.enable_grad():`)
  and returns an output with autograd history is not inference-only, and backward stays
  available. An explicit `inference_only=True` is never changed. `Trace.options.inference_only` and
  `outcome.inference_only` report the effective value; a later `backward()` refuses and names
  the cause (`recording_backward_inference_only` on a `Recording`).

**carry_state (CaptureOptions) / pure-observer capture** — *unstable — no deprecation shim owed*
: A default `tl.trace` is a pure observer: it snapshots buffers (persistent and non-persistent),
  every module's `training` flag and the global random states (torch, live CUDA devices, Python,
  NumPy) before the forward and restores them afterwards, on success and on failure, so a
  train-mode BatchNorm's running statistics do not advance. A buffer the forward changed is written
  back only where its bytes changed, and trace payloads that alias it keep their forward-time
  values. Every plain module attribute is restored too: a rebound or removed attribute gets its
  original object back, a tensor or builtin `list`/`dict`/`set` changed in place is restored in
  place, and attributes the forward added are removed; what the forward wrote is read from the
  trace (`Module.attribute_writes`). Builtin containers are snapshotted at any size (a tensor inside
  one is cloned whatever its size); custom objects are tracked by identity: a rebinding is undone,
  a change inside them is neither undone nor recorded. A container nested more than 256 levels
  deep is tracked by identity below that level and named in a `capture_state_restore_incomplete`
  warning. Lazy modules are left as their materializing forward leaves them.
  `CaptureOptions(carry_state=True)` opts into forward-equivalent state: buffers, modes and
  plain attributes advance exactly as one direct `model(x)` would, and the attribute writes are
  recorded the same way, the same spelling as `trace.run(inputs=..., carry_state=True)`. No capture advances the caller's random streams
  under either setting (each capture draws from its own seed and restores the global engines). A capture-cache hit runs no forward, so nothing advances. An item that
  cannot be restored is named in one `capture_state_restore_incomplete` warning. Transient: an
  option only, never persisted.

**Module.attribute_writes / AttributeWrite** — *unstable — no deprecation shim owed*
: The plain instance attributes one module's forward wrote during the capture, as a read-only
  mapping `{attribute name: AttributeWrite}`, for example
  `trace.modules["encoder.attn"].attribute_writes["last_weights"].value`.
  `torchlens.types.AttributeWrite` is a frozen record with `kind`
  (`"added"`, `"replaced"` for a rebinding or an in-place change, `"removed"`) and `value`, what
  the forward left: the object itself when the forward bound a new object (the trace keeps it
  alive, so a large stashed tensor costs its full memory for the trace's lifetime), a copy when
  the forward changed the original object in place (builtin `list`/`tuple`/`dict`/`set`
  containers are copied recursively, a tensor that existed before the forward is cloned whatever
  its size, a tensor the forward created is kept by reference), `None` when removed. `{}` when the forward wrote
  nothing on the module; `None` when no record exists (a loaded trace, a lazy module, a
  non-torch backend). Recorded by default and under `CaptureOptions(carry_state=True)`; a change
  inside an identity-tracked object is not recorded; a capture that re-ran the forward (escape
  rescue) records the net effect of both runs. Session-time only: never persisted, excluded
  from `to_pandas()`.

**Bound-method root**
: The ruled `tl.trace` root contract: an `nn.Module` OR a bound method of one, e.g.
  `tl.trace(model.generate, ids, ...)` (also accepted by `tl.validate`). The owner resolves via
  `method.__self__` and registers as the `owner` submodule of a TorchLens-authored wrapper root
  whose forward calls the exact bound method exactly once; the capture's identity reads
  `type(owner).__name__`, `Trace.provenance.root_entry_point` records
  `"bound_method:<owner_qualname>.<method>"`, and the root op record (the output-boundary ops)
  carries `tl_authored_root=True` — both persisted with fail-closed load validation. On
  bound-method episode captures `EpisodeOptions.stepped_module` defaults to the owner. Bound-method
  captures refuse rerun/append at the identity gate (`rerun_entry_point_unsupported`; an absent
  fact on a legacy artifact refuses `root_entry_point_unavailable`); replay engines are
  unaffected. Other callables (closures, bare functions) refuse `model_type_unsupported`. All
  spellings are provisional.

**Episode capture**
: One wrapped multi-step generation run captured as a single product
  (`capture_kind=episode`): `tl.trace(episode_root, x, episode=EpisodeOptions(stepped_module=...))`
  stamps the declaration and lands a per-step status ledger (grammar v2,
  `episode_ledger_version=2`: header with the declared `step_output_kind`
  (`tokens`/`digest`/`none`), the `step_output_from` source disclosure, the
  `step_axis` with the disclosed `step_output_positions` each row was read
  from, the minted `capture_digest` binding, the measured
  `step_join` envelope grading every cross-step join —
  continuous/forced/transformed/declared/exogenous/unchecked, with episode-dependent
  claims refused across a measured break — and, on intervened (COUPLED)
  captures, the deterministic `intervention_digest`; rows with
  `complete`/`interrupted`/`absent`
  statuses, the generic `step_output`, the channel-keyed carried-state witness
  slots `entry_state_digest`/`exit_state_digest` (`None` = not measured; the
  arithmetic `cache_len` is deleted), the measured per-step `fire_count` on
  coupled captures (zero and multiple fires are first-class), and the
  managed-RNG entry
  seed) at `trace.annotations["episode"]`. `episode=` x `intervene=` runs
  COUPLED (attested coupling): `trace.episode_coupling` recomputes and
  compares the capture digest (the positive ledger-to-product binding) and
  serves fire counts plus per-segment facts that never span a measured break;
  `torchlens.at_step(*steps)` is the step-qualified selector
  (live and post hoc via the persisted `Op.episode_step` stamps); replay
  derives a fresh ledger or refuses (`episode_coupled_replay_underivable` on
  both `run()` engines; `do()` edits quarantine inherited episode evidence). A diagnostic-tier product for tens of steps
  (cost is superlinear in step count); the ledger is a disclosure, never a settlement
  authority, and it persists plainly as of the tlspec v8 coordinated bump (loads validate
  fail-closed). All episode
  spellings are provisional (no deprecation shim owed). See
  [Episode capture](episode_capture.md).

## Graph records

**Op**
: One executed callable invocation. Ops are the dataflow graph's nodes and have unique full labels
  such as `relu_1_2`.

**Layer**
: The user-facing grouping of Ops that run one place in a module's code. Running that place
  again (a loop, a recursion, a comprehension) is another pass, addressed with a one-based
  suffix such as `linear_1:2`; `op.pass_number` is the pass and `num_passes` the count. A module
  object called again is another module call (`m:1`, `m:2`), a separate concept: each op inside it
  runs once per module call, so its passes match the module's calls. Two different places are two layers even with the same weights, so a
  loop written out by hand is N layers. The exact rule (scope, call site, operation, output slot)
  and the case table are in `docs/guides/grouping.md`.

**Module**
: A captured framework module or backend-neutral module-like object. A module contains Ops but is
  not itself an Op.

**Parameter / Buffer**
: Registered model state. Parameters are trainable state; buffers are non-parameter state and may
  have multiple captured versions. Every journaled write (a reassignment, a `.data =` or `set_`
  swap, a tracked in-place write) is its own version whatever value it wrote, so a capture records
  the same buffer structure whatever values the buffers start from. A training-mode `BatchNorm2d` collapse label can say
  “5 layers total” because it counts the parameter leaves `weight` and `bias` plus the buffer
  leaves `running_mean`, `running_var`, and `num_batches_tracked`; it is not claiming five
  BatchNorm calls. An in-place op on a Parameter inside `forward`
  (`with torch.no_grad(): self.temp.clamp_(lo, hi)`) is captured as an op whose `params` holds
  that Parameter; every later read of the Parameter in the same pass binds to that op as a graph
  parent, not as a parameter edge, so it is absent from that read's `params`. `draw()` shows
  such a Parameter as a grey cylinder labeled `parameter <name>` in its owning module's box, with
  an edge to each op that reads its pre-mutation value; Parameters never mutated get no node.
  BatchNorm calls. A plain tensor a module already holds when the forward starts (a tensor
  attribute, a list/tuple item, or a dict value such as a warm-filled attention-bias cache) is
  captured as a buffer too, addressed `<module>.<attr>`, `<module>.<attr>.<index>`, or
  `<module>.<attr>.<key>`.
  parent, not as a parameter edge, so it is absent from that read's `params`.
  The receiver may be given by keyword, as the `input` keyword of `torch.clamp_`. A frozen Parameter written
  in place with a grad-requiring operand ends the pass a non-leaf, exactly as in eager. An `out=`
  write into a Parameter (`torch.add(a, b, out=p)`) is not captured yet; validation reports it as
  a completeness failure.
  an edge to each op that reads its pre-mutation value, under both layout engines (`dot` and
  `rank`). The node is drawn when the first in-place op on the Parameter is drawn as its own
  node: a collapsed module box or `skip_fn` that hides that op hides the node with it, and a
  `module=` focus draws it only when the focused module is the owner or contains it. Parameters
  never mutated get no node. The legend row's swatch is the trainable-parameter grey.
  attribute, a list/tuple item, or a dict value such as a warm-filled attention-bias cache,
  nested up to four container levels) is captured as a buffer too, addressed
  `<module>.<attr>`, `<module>.<attr>.<index>`, or `<module>.<attr>[<key>]` with one
  `[<index>]`/`[<key>]` per nested level, e.g. `attn.attention_bias_cache['cpu']`. The key
  is its `repr` with `.`, `:` and `%` percent-escaped, so the last `.` always separates the
  owning module; keys that print alike get `#2`, `#3`, ... A public tensor attribute of a
  plain (non-Module) object the module holds, such as a `torch.distributions` prior or a
  diffusers scheduler built in `__init__`, is a buffer too, addressed
  `<module>.<attr>[<name>]` (e.g. `vae.prior[scale]`); only the object's own attributes are
  read, not containers inside it.

**GradFn**
: A first-class live backward-autograd node captured during backward logging. Portable artifacts
  do not contain PyTorch's live autograd graph.

**Facet**
: A named semantic view into one captured value, such as an attention projection or an LSTM state.

### Documented-unstable ATen profile INDEX

The ATen execution profile is recorded by the public switch `CaptureOptions(aten_ops=True)`
(value-free rows) or `CaptureOptions(aten_ops="values")` (rows plus each call's output tensors in
`AtenOp.out`) and
served by the stable accessors `Trace.aten_ops`, `Trace.has_aten_ops`, `Trace.aten_gaps` and
`Op.aten_ops` (entries below this index); an absent profile refuses typed with
`AtenOpsUnavailableError` (`aten_ops_unavailable`). As of the tlspec v8 coordinated bump the
profile's fields persist (`FieldPolicy.KEEP`) and loaded profiles validate their foreign keys
whenever present; older v6/v7 artifacts contain no primitive profile. The names below are the
row-level surface the profile exposes. Each is documented unstable and may be renamed or removed
without a compatibility alias. No validation or honesty tripwire may be weakened.

<!-- ATEN-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Facades, record kind, disclosure, and provenance | `AtenOp`, `OpRef`, `SuperAtenOp`, `primitive_op`, `mode_paused_interior`, `exact_via_aten`, `heuristic` | unstable -- no deprecation shim owed |
| Field-order contract | `PRIMITIVE_OP_FIELD_ORDER` | unstable -- no deprecation shim owed |
| Primitive fields | `label`, `sequence`, `capture_phase`, `forward_pass_index`, `backward_epoch_index`, `owner_func_call_id`, `parent_op_refs`, `parent_grad_fn_call_ref`, `owner_status`, `decomposition_slot`, `namespace`, `operator`, `overload`, `schema`, `schema_fingerprint`, `module_call_stack`, `input_tensor_facts`, `output_tensor_facts`, `mutation_kind`, `view_copy_kind`, `autocast_context`, `dispatch_key_context`, `grad_fn_ref`, `grad_fn_link_status`, `grad_fn_link_provenance`, `algorithmic_flops`, `flop_status`, `flop_formula_source`, `flop_formula_version`, `outcome`, `exception_type`, `execution_context`, `out` | unstable -- no deprecation shim owed |
| Observer-gap fields | `kind`, `capture_phase`, `sequence_before`, `sequence_after`, `owner_func_call_id`, `parent_op_refs`, `reason` | unstable -- no deprecation shim owed |
| Redundant Op-reference fields | `op_row_index`, `op_label`, `func_call_id` | unstable -- no deprecation shim owed |
| Tensor-fact fields | `container_path`, `tensor_impl_capability`, `logical_version`, `storage_alias_group`, `shape`, `stride`, `dtype`, `device`, `layout`, `requires_grad` | unstable -- no deprecation shim owed |
| Execution-context fields | `pytorch_version`, `backend`, `device_model`, `device_capability`, `grad_mode`, `inference_mode`, `module_training_summary`, `autocast`, `deterministic_algorithms`, `tf32_matmul_policy`, `sdpa_policy`, `compile_stance`, `owner_thread_coverage`, `completeness_witness_mode` | unstable -- no deprecation shim owed |
| Super comparison fields | `comparison_status`, `has_observation_gap` | unstable -- no deprecation shim owed |
| Invariant contracts | `primitive_op_invariants`, `non_torch_primitive_op_inert`, `aten_values_consistent` | unstable -- no deprecation shim owed |
| Switch-active load failures | `primitive_op_schema_invalid`, `primitive_op_fk_invalid` | unstable -- no deprecation shim owed |
| Capture phases | `forward`, `backward`, `setup` | unstable -- no deprecation shim owed |
| Mutation classes | `none`, `in_place`, `out_variant`, `metadata_only`, `unknown` | unstable -- no deprecation shim owed |
| View/copy classes | `view`, `copy`, `alias`, `unknown` | unstable -- no deprecation shim owed |
| Owner classes | `forward_op`, `backward_grad_fn_call`, `orphan`, `unresolved` | unstable -- no deprecation shim owed |
| Grad-link classes | `linked`, `unlinked`, `conflict`, `not_applicable` | unstable -- no deprecation shim owed |
| Dispatcher outcomes | `returned`, `raised` | unstable -- no deprecation shim owed |
| FLOP evidence classes | `formula_exact`, `estimated`, `unsupported` | unstable -- no deprecation shim owed |
| Super alignment classes | `all_present_same_schema`, `all_present_different_schema`, `sparse`, `coverage_indeterminate` | unstable -- no deprecation shim owed |
| Execution and disclosure tokens | `forced_eager`, `strict_subclass_constructor` | unstable -- no deprecation shim owed |
| Temporary label grammar | `aten_<sequence>` | unstable -- no deprecation shim owed |

<!-- ATEN-UNSTABLE-INDEX:END -->

**aten_ops (CaptureOptions)**
: Opt-in (default `False`): the torch capture records the ATen operations each logged torch call
  dispatched to, as `AtenOp` rows, through a dispatch mode installed for the forward and, with
  `backward_ready=True`, for `Trace.backward()`. `True` keeps the rows
  value-free; `"values"` additionally keeps what each call returned in `AtenOp.out`, under the
  capture's save policy (the `save=` selection, `save_mode`, `output_device`, `save_budget`); any
  other value refuses typed (`InvalidArgumentError`, `aten_ops_invalid`). The default installs
  no dispatch mode and adds no per-op work. A session-time knob (not persisted; load restores
  `False`) that `run()` refreshes inherit; the recorded rows themselves persist (tlspec v8+), so a
  loaded trace keeps `has_aten_ops`. `run()` results, fast or not, inherit the recording of the
  trace they refresh; non-torch backends record nothing. GPU kernel recording is out of scope. Spelling: velocity walk builds.2 (1 Nov 2026); `"values"` literal: aten-values lane (3 Nov 2026).

**Trace.has_aten_ops**
: `True` when the trace carries recorded ATen operations: after a capture with
  `CaptureOptions(aten_ops=True)` (a present-but-empty profile counts) and after loading an
  artifact saved from one; `False` otherwise. The one accessor that never raises. Spelling:
  velocity walk builds.3 (1 Nov 2026; booleans take `is_`/`has_`).

**Trace.aten_ops**
: Every recorded `AtenOp` in dispatch order, forward rows then backward rows, as a tuple. Raises
  `AtenOpsUnavailableError` (`aten_ops_unavailable`) when the capture recorded none, so an empty
  answer never stands in for "recording was off".

**Trace.aten_gaps**
: The disclosed interiors where ATen rows are missing by construction (the profile's
  `mode_paused_interior` records): each names the capture phase, the dispatch sequence it spans
  and the owning op references of a stretch where the dispatch mode was paused, for example inside
  a user-injected intervention. An empty tuple means the recorded rows are complete. Raises
  `AtenOpsUnavailableError` when the capture recorded none.

**Op.aten_ops**
: The ATen operations this op's torch call dispatched to, in dispatch order: the rows whose
  resolved owner references carry the op's label. An op whose call dispatched nothing observable
  yields an empty tuple while the trace still `has_aten_ops`. Raises `AtenOpsUnavailableError`
  when the owning trace recorded none.

**AtenOp.out** -- *unstable -- no deprecation shim owed*
: What the ATen call returned, mirroring `Op.out`: a tensor for a single-tensor result, a tuple of
  tensors (one per `output_tensor_facts` entry, in the same order) for a container result, `None`
  when no value was kept. Recorded only under `CaptureOptions(aten_ops="values")`; otherwise every
  row reads `None`. Each value is a snapshot copied below autograd at the moment the call returned:
  it never carries a `grad_fn` (`detach_saved_activations=False` attaches `Op.out`, not these), an
  in-place call's value is the mutated tensor as returned, a view's value is the view's content, and
  aliasing between rows is not preserved (`view_copy_kind` and `storage_alias_group` describe it).
  Which rows keep values follows the owning torch call: a forward row keeps its value when an `Op`
  of its call saved its output (`has_saved_activation`), so `save=` restricts ATen values the way
  it restricts `Op.out`; a dropped owner frees its values during the forward. Backward rows (with
  `backward_ready=True`) and orphan rows keep values unconditionally. Values honour `save_mode`
  (`"reference"` becomes a copy for in-place calls), `output_device` and the `save_budget` (each
  value is admitted like an activation; the budget error names `aten:<operator>.<overload>`). Not
  retained when activations stream to disk or an out sink is active (every row reads `None`; a
  typed gap, see limitations). Persist and load with `Op.out` (`BLOB_RECURSIVE`;
  `include_outs=False` drops them while the facts stay). The `aten_values_consistent` invariant
  checks every kept value against its facts and, under `save_mode="copy"`, that some row beneath
  each saved op equals the op's `out`.
  A `run()` refresh re-records the profile against the new input, so values follow `Op.out`; the
  fast tier (`run(inputs=..., fast=True)`, and the guarded fast rerun behind `run(model, x)` with
  staged hooks) never re-records ATen values, so it refuses a values capture with
  `run_capability_unavailable` at stage `fast_run_aten_values_unsupported` and the legacy door
  falls back to the capture engine.

**AtenOpsUnavailableError**
: `torchlens.errors.AtenOpsUnavailableError` (`CaptureError`, `RuntimeError` lineage), code
  `aten_ops_unavailable` in `exc.fields["code"]`: `Trace.aten_ops`, `Trace.aten_gaps` or
  `Op.aten_ops` was read on a capture that recorded no ATen operations (the switch was off, the
  artifact was saved without it, or the trace is a `run(fast=True)` result). The message names the
  remedy, `CaptureOptions(aten_ops=True)`; check `Trace.has_aten_ops` first to avoid it.

`AtenOp` is one dispatcher call measured during a concrete capture (value-free unless
`aten_ops="values"` filled `out`). Its label is opaque,
capture-local, and intentionally excluded from universal Trace string lookup. `OpRef` is a redundant
dense foreign key whose row index, Op label, and function-call witness must all agree. A
`mode_paused_interior` entry says only that TorchLens paused its owned dispatch observer around a
strict Tensor-subclass constructor; recorded rows and counts on that parent are lower bounds, and no
synthetic primitive row is created for the unseen interior.

`SuperAtenOp` aligns observed rows positionally by `decomposition_slot`. Its `comparison_status`
distinguishes equal-schema coverage, different-schema coverage, proven sparse membership, and
coverage that is indeterminate because at least one member has an observation gap. Positional
alignment is evidence, not semantic equivalence.

### Documented-unstable kernel telemetry INDEX

The optional CUDA/CUPTI adapter is the detachable trailing lane of the ATen execution profile. Its
rows persist as of the tlspec v8 coordinated bump (`FieldPolicy.KEEP`), and it is not imported by
the capture core. Every spelling below is documented unstable
and may be renamed or removed without a compatibility alias.

<!-- KERNEL-TELEMETRY-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Facade and computed views | `KernelLaunch`, `AtenOp.gpu_kernels`, `Op.gpu_kernels` | unstable -- no deprecation shim owed |
| Launch fields | `launch_name`, `device`, `stream`, `duration`, `runtime_correlation`, `attribution_status` | unstable -- no deprecation shim owed |
| Attribution classes | `attributed`, `unavailable`, `ambiguous`, `unattributed` | unstable -- no deprecation shim owed |

<!-- KERNEL-TELEMETRY-UNSTABLE-INDEX:END -->

`KernelLaunch` is measured profiler evidence, not graph identity or replay authority. The adapter
places a unique marker around each redispatched ATen call and joins CUDA runtime calls to kernels
and memory copies through Kineto correlation identifiers; it never joins by operator or launch-name
substring. `AtenOp.gpu_kernels` exposes the relation for one primitive row, while
`Op.gpu_kernels` is the deduplicated union across that Op's observed primitive rows.

An unavailable profiler session returns one fact-free row with
`attribution_status="unavailable"`; it never reports zero kernels. When a parent has a
`mode_paused_interior` disclosure, every launch set and any count derived from it is a lower bound:
the paused region is not profiled synthetically and no hidden launch is fabricated. See
[Kernel telemetry](kernel_telemetry.md).

### Documented-unstable attribution kit INDEX

The L6 stage-4b attribution kit is a detachable orchestration layer over shipped gradient,
Selection, intervention, and receptive-field visualization primitives. Every spelling below is
documented unstable and may be renamed or removed without a compatibility alias.

<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Integrated Gradients method and controls | `integrated_gradients`, `target`, `num_steps`, `baseline` | unstable -- no deprecation shim owed |
| Completeness evidence | `attribution_sum`, `target_delta`, `completeness_residual` | unstable -- no deprecation shim owed |
| Occlusion method and controls | `occlusion`, `selection`, `score`, `blur_kernel_size` | unstable -- no deprecation shim owed |
| Occlusion baseline policies | `zeros`, `mean`, `blur` | unstable -- no deprecation shim owed |
| Occlusion evidence | `original_score`, `occluded_score`, `selection_digest` | unstable -- no deprecation shim owed |
| CAM method and controls | `grad_cam`, `layer`, `reshape`, `relu`, `overlay`, `image`, `alpha`, `cmap` | unstable -- no deprecation shim owed |
| CAM token grid and layer-choice evidence | `grid_route`, `feature_map`, `derived`, `prefix_tokens`, `grid_evidence`, `layer_source`, `caller`, `trace`, `layer_reason` | unstable -- no deprecation shim owed |
| CAM resolution evidence | `native_map_resolution`, `rendered_map_resolution`, `upsampling`, `bilinear_display_only` | unstable -- no deprecation shim owed |
| Graph overlay bridge and controls | `overlay`, `source`, `reduce`, `abs_sum`, `abs_mean`, `sum`, `max` | unstable -- no deprecation shim owed |
| F06 wrapping contract | `noise_tunnel`, `attribute`, `method`, `method_kwargs` | unstable -- no deprecation shim owed |
| F06 noise tunnel controls and evidence | `num_samples`, `stdevs`, `reduce`, `noise_bank`, `mean_square`, `variance`, `stdevs_resolved`, `residual_of_means` | unstable -- no deprecation shim owed |
| F06 GradientShap controls | `gradient_shap`, `baselines`, `draw_bank`, `store_draws` | unstable -- no deprecation shim owed |
| F06 guided methods and evidence | `guided_backprop`, `deconvolution`, `sites`, `absolute`, `site_census` | unstable -- no deprecation shim owed |
| F06 occlusion map controls | `occlusion_map`, `window`, `strides`, `baseline_value`, `occlude_leaf`, `max_passes`, `average` | unstable -- no deprecation shim owed |
| F06 IG step runner controls and evidence | `step_batch_size`, `step_audit`, `step_audit_seed`, `per_call`, `per_chunk`, `path_evaluations_logical`, `physical_forward_calls` | unstable -- no deprecation shim owed |
| F06 completeness disclosure | `residual_rel`, `target_delta_abs`, `completeness_caveat` | unstable -- no deprecation shim owed |
| F06 metrics | `infidelity`, `sensitivity`, `AttributionMetric`, `perturb`, `noise_std`, `radius`, `gaussian`, `square_removal` | unstable -- no deprecation shim owed |
| F06 token attribution | `text`, `TokenAttribution`, `TokenAttributionPayload`, `keep_special_tokens`, `steps_per_batch`, `max_length`, `pad_token`, `auto`, `converged`, `show` | unstable -- no deprecation shim owed |
| F06 LRP mechanism | `SiteStash`, `stash`, `fetch`, `mark_firing`, `leftovers`, `register_site`, `label_of` | unstable -- no deprecation shim owed |
| Site targets | `SiteTarget`, `unit`, `call` | unstable -- no deprecation shim owed |
| Integer-input evidence | `integer_input_layers` | unstable -- no deprecation shim owed |
| Evaluation curves | `deletion_curve`, `insertion_curve`, `PerturbationCurve`, `CurveStep`, `fractions`, `scores`, `area`, `aopc`, `steps`, `features`, `patch_size`, `auto`, `elements`, `pixels`, `patches`, `tokens`, `num_features`, `index`, `fraction`, `inputs`, `input_kwargs`, `build_baseline_mask`, `intervention` | unstable -- no deprecation shim owed |
| Randomization test | `randomization_test`, `RandomizationTest`, `RandomizationStep`, `randomization`, `cascading`, `independent`, `layers`, `seed`, `keep_values`, `original_attribution`, `randomized_module_addresses`, `similarity`, `similarities`, `spearman`, `spearman_abs`, `ssim`, `values` | unstable -- no deprecation shim owed |
| ROAD curve | `road_curve`, `ranking`, `most_relevant_first`, `least_relevant_first`, `noise_std`, `seed`, `road` | unstable -- no deprecation shim owed |
| Causal audit | `causal_audit`, `CausalAudit`, `CausalAuditUnit`, `CausalAuditSet`, `claim`, `k`, `edit`, `zero`, `mean`, `controls`, `top`, `bottom`, `random`, `num_random_draws`, `per_unit`, `layer`, `site`, `original_score`, `units`, `sets`, `rank_agreement`, `set_of`, `unit`, `claimed_effect`, `measured_effect`, `kind`, `draw`, `intervention`, `component`, `claim_kind`, `constant_row`, `rows_not_audited`, `corrupted_input`, `metric`, `edges`, `edge_table`, `edge_patch`, `example_reduction`, `logits_site`, `sets_not_measured`, `reason`, `trace` | unstable -- no deprecation shim owed |
| Attribution graphs | `build_graph`, `AttributionGraph`, `GraphNode`, `transcoders`, `embedding_module`, `target`, `max_num_logits`, `logit_probability_coverage`, `demeaned_logits`, `logits_at`, `frozen`, `nodes`, `adjacency_matrix`, `node_values`, `constants`, `encoder_bias`, `decoder_bias`, `other`, `active_feature_indices`, `feature_activations`, `transcoder_sites`, `error_sites`, `error_vectors`, `input_token_ids`, `logit_token_ids`, `logit_probabilities`, `capture_status`, `disclosures`, `node_counts`, `node_indices`, `kind`, `site`, `position`, `feature_index`, `token`, `logit`, `error` | unstable -- no deprecation shim owed |
| Attribution graph export | `to_neuronpedia`, `slug`, `model_id`, `tokenizer`, `prompt`, `node_influence_coverage`, `edge_influence_coverage`, `path` | unstable -- no deprecation shim owed |
| Attribution graph overlay and upload | `influence`, `activation`, `upload_to_neuronpedia`, `api_key` | unstable -- no deprecation shim owed |

<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:END -->

Integrated Gradients always reports its signed completeness residual against the exact endpoint
target delta. Occlusion always names its replacement baseline. CAM overlays visibly state the
native measured map resolution, while their larger display grid is labeled as interpolation.
`build_graph` builds an attribution graph from one prompt's capture and a set of transcoders: typed feature, error, token and logit nodes whose edges are direct linear effects along a frozen graph, with every feature and logit node accounted for by its incoming edges plus named constant terms (`AttributionGraph`). A `SiteTarget` passed as `target=` attributes a hidden layer's output (or one unit of it) instead
of the model output, and with `sae=` one latent of that SAE's encoding of the output (unstable, no
deprecation shim owed); layer methods given only integer inputs start their input path at the
embedding output. `deletion_curve` and `insertion_curve` score an attribution by removing or
revealing its most-attributed input features in steps (each step a recorded intervention), and
`randomization_test` checks that a method's maps change when the model's layers are
re-initialized. `road_curve` is ROAD, the deletion curve with removed pixels filled by noisy
linear imputation from their neighbours. `causal_audit` grades an internal attribution (a layer attribution, a direct-logit-attribution
table or a one-backward read table) by ablating the units it ranks, through forks of an
intervention-ready trace, and comparing the measured effects with the claim; each row names its
`site` and its `component`, and a direct-logit table's `constant_row` is reported apart. An EDGE table
from `torchlens.mechinterp.edge_scores` is audited by patching each edge alone with the `corrupted_input` inputs and
`metric` it was scored with (`edges` names a subset); every set is patched in one call, into one
receiver or several, and `sets_not_measured` stays empty. The callable/index/site annotation used inside `torchlens.attribution` is
privately named `_AttributionTarget`; `TargetSpec` refers only to the intervention selector record and is not an
attribution-target alias. See the [attribution reference](attribution.md).

### Documented-unstable mechinterp kit additions INDEX

Batch-2 additions to `torchlens.mechinterp` (the transformer analysis kit). Every spelling below
is documented unstable and may be renamed or removed without a compatibility alias.

<!-- MECHINTERP-KIT-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Factored low-rank algebra (vendored from TransformerLens, MIT) | `FactoredMatrix`, `A`, `B`, `AB`, `BA`, `T`, `svd`, `U`, `S`, `V`, `eigenvalues`, `norm`, `make_even`, `corner`, `collapse_l`, `collapse_r`, `unsqueeze`, `pair`, `ldim`, `mdim`, `rdim`, `ndim` | unstable -- no deprecation shim owed |
| QK and OV circuits on verified head weight views | `qk_circuit`, `ov_circuit` | unstable -- no deprecation shim owed |
| Per-neuron MLP rows under the identity gate | `mlp_neuron_contributions`, `layers`, `neurons`, `onto`, `positions` | unstable -- no deprecation shim owed |
| Alias table generation value | `generation="4.x"` (TransformerBridge canonical hook names) | unstable -- no deprecation shim owed |
| Alias table generation value | `generation="nnterp"` (nnterp `StandardizedTransformer` accessor names: `token_embeddings`, `embed_tokens.output`, `layers_input[L]`, `layers_output[L]`, `attentions_input[L]`, `attentions_output[L]`, `attention_probabilities[L]`, `mlps_input[L]`, `mlps_output[L]`, `ln_final.input`, `lm_head.output`, `logits`); `alias_report(trace, generation="nnterp")` lists the trace's sites under these names | unstable -- no deprecation shim owed |
| Appliance on the `trace.using` door | `trace.using("mi")`, `head_scores`, `qk_circuit`, `ov_circuit` | unstable -- no deprecation shim owed |
| Per-query-head patching under grouped-query attention | `patch_query_heads_grid`, `site`, `heads`, `effect_site` | unstable -- no deprecation shim owed |
| Component edge graph (heads and MLPs as nodes, q/k/v split receivers) | `component_graph`, `ComponentGraph`, `ComponentSender`, `ComponentReceiver`, `ComponentEdge`, `nodes`, `receivers`, `edges`, `layer_index`, `head`, `senders`, `contribution`, `identity_receipt`, `input_label`, `norm_address`, `module_address`, `patch_edges`, `corrupted_graph` | unstable -- no deprecation shim owed |
| Edge attribution patching over the component graph | `edge_scores`, `corrupted_input`, `metric`, `method`, `eap`, `eap_ig`, `steps`, `example_reduction`, `mean`, `sum` | unstable -- no deprecation shim owed |
| Circuit faithfulness areas | `circuit_performance_ratio`, `circuit_model_difference`, `faithfulness`, `fractions`, `log_scale` | unstable -- no deprecation shim owed |
| Fixed-period induction score | `FixedPeriodInduction`, `head_scores(..., kind=FixedPeriodInduction(period=n))`, `period` | unstable -- no deprecation shim owed |
<!-- MECHINTERP-KIT-UNSTABLE-INDEX:END -->

`component_graph(trace)` returns a `ComponentGraph`: read-only `nodes`, `receivers` and `edges`
keyed by MIB/EAP-IG vocabulary name (`input`, `a{L}.h{H}`, `b{L}`, `m{L}`; receivers
`a{L}.h{H}<q|k|v>`, `m{L}`, `logits`; edges `"<sender>-><receiver>"`). Each `ComponentReceiver`
holds the captured pre-norm input (`value`, from the op `input_label` entering the norm at
`norm_address`) and its `senders`. Nodes and receivers carry `layer_index` (transformer block
index) and `head` (head index), each `None` where it does not apply. `ComponentSender.module_address`
is the owning module (embedding, attention or MLP; the attention module for head and bias nodes)
and `ComponentReceiver.module_address` the reading module (attention, MLP, or the unembedding
head). Each `ComponentEdge` exposes
the sender's output as its `contribution`. `identity_receipt` records the in-API check that every receiver input equals the
sum of its contributions. Each attention output bias is its own node `b{L}`. A `ComponentEdge` is
also a selection (`__selection__`): an EDGE selection of the pre-norm input edge for `m{L}` and
`logits` receivers, an ACT selection of one head's q, k or v projection slice otherwise.
`patch_edges(trace, edges, *, corrupted_graph)` returns a fork of `trace` in which every receiver the
edges enter reads `input - sum(s) + sum(s_corrupt)` over its patched edges (`corrupted_graph` is the
corrupted input's `ComponentGraph`); `input` and `s` are the patched forward's own values (the
clean values when one receiver is patched), so a circuit (everything outside a kept set patched)
is one call; the trace needs `replayable=True`.

`edge_scores(model, inputs, *, corrupted_input, metric, method="eap_ig", steps=None,
example_reduction="mean")` returns a one-backward `ReadTable` of `EDGE` rows, one per
component-graph edge, addressed `(edge name,)`: the edge attribution patching score
`(s_corrupt - s_clean) . dM/d(input of t)`, summed over positions and reduced over examples
(`example_reduction`, `"mean"` or `"sum"`). `metric` maps the unembedding output to one value per
example. `method="eap"` takes the gradient at the clean input; `method="eap_ig"` averages it over
`steps` points (default 5) from the corrupted to the clean `input` node (left Riemann sum).
`circuit_performance_ratio(faithfulness, *, fractions=None, log_scale=False)` (CPR) and
`circuit_model_difference(...)` (CMD) are the trapezoidal areas under `f` and under `|1 - f|` of a
faithfulness curve over kept edge fractions (`fractions` default: the ten MIB fractions 0.001 to
1; `log_scale` integrates over the log fraction).

`qk_circuit(trace, attention)` and `ov_circuit(trace, attention)` return a `FactoredMatrix` of
shape `[num_q_heads, d_model, d_model]` built from `head_weight_views` (TransformerLens's
`model.QK` / `model.OV` convention; biases excluded; under grouped-query attention each query head
pairs with the KV head that serves it). `mlp_neuron_contributions` returns a `ComponentStack` with
one row per hidden neuron (`hidden[n] * W_out[n]`, read from the output projection's captured
input), a verified bias constant, and the D7 grading; unselected neurons under `neurons=` fold into
one exact `other_neurons` row.

`patch_query_heads_grid(model, clean_input, corrupted_input, metric, site="v", heads=None,
budget=None, trace_kwargs=None)` returns a `PatchGrid` with axes `("layer", "head")` whose head
coordinates are QUERY heads, on grouped-query attention. Each cell patches one query head's copy of
its key/value group (`site="v"` for the value copy, `"k"` for the key copy) at the op that expands
the grouped tensor to the query heads, so the other heads of the group are untouched; a k/v-site
`grid` can only patch the whole group. Receipts add `effect_site` (the interceptor wording),
`site`, `num_q_heads`, `n_kv_heads` and `expansion_ops`, and every cell fires exactly once. As for
every `PatchGrid`, `values` are patched-run metrics and the receipts carry the unpatched baselines
`clean_metric` and `corrupted_metric`. Fused
kernels with no expansion op refuse `mi_kv_expansion_unavailable`; several expansion ops refuse
`mi_kv_expansion_ambiguous`; equal query and key/value head counts refuse
`mi_kv_heads_not_grouped`.

## Selection and storage

**Predicate**
: A site selector built with `tl.func`, `tl.module`, `tl.in_module`, `tl.label`, and boolean or
  temporal composition. `save=` is the only selective sparse-capture spelling; removed
  `keep_op=` and `keep_module=` arguments raise `TypeError`. On `tl.trace`, `save=` also takes
  the list form (layer names, pass-qualified labels, module addresses, integer indexes,
  substrings, `'all'`, `'none'`), with final-label selection resolved after capture. `save="stats"` is a third level beside `'all'` and `'none'`:
  reduce and discard, one `tl.stats.RunningStats` per op and no payload (see Streaming
  statistics).

**Capture options**
: `CaptureOptions` configures capture mechanics. Select payloads with the `save=`
  argument on `tl.trace` and `tl.record`; payload selection is not a public field
  of `CaptureOptions`. Preview saving is not a capture option. `module_filter`
  remains an additional gate on the selected payloads.

**Lookback**
: A bounded window of earlier metadata or detached payloads retained so a later temporal predicate
  can select its predecessors.

**Save mode**
: The activation-retention policy: `copy` isolates the retained value, `reference` preserves the
  live autograd-capable object, and `view` intentionally leaves aliasing visible.

**Streaming storage**
: A `storage=tl.to_disk(...)` capture that writes selected payloads while capture proceeds. It is
  distinct from saving a completed in-memory Trace. `trace` blob writes overlap the forward on a
  bounded single-worker pipeline by default (`to_disk(..., async_writes=...)`, a
  DOCUMENTED-UNSTABLE spelling): payloads are snapshotted at submission so later in-place
  mutation never reaches the artifact, writes land in submission order, `max_pending_bytes`
  (default 256 MiB) blocks capture when the disk falls behind, a failed write raises a typed
  `ArtifactError` and marks the temp bundle `PARTIAL`, and finalization waits for every
  pending write before the bundle publishes. `tl.record` streaming stays synchronous and
  refuses an explicit `async_writes=True`. The model output the forward returns stays in
  memory (the output op's payload may alias it). A streamed trace does not run: `trace.run(...)`
  on it refuses `trace_option_unsupported` (every run door, `inputs=` and `fast=True` included),
  because a run would re-capture every payload into RAM while the bundle kept the first
  capture; capture again with `to_disk` to a new path, or without `to_disk`, and run that trace
  (a trace loaded from the bundle runs through its loaded providers). A `path` that already
  exists refuses with `bundle_path_exists`, as does `Trace.save` or `Bundle.save` to an existing
  path without `overwrite=True`. A `path` whose final component is a symlink refuses (the same
  final-component rule `Trace.save` and `tl.load` apply; a symlinked parent passes).

**Capture cache**
: The opt-in `cache=True` content-hash store of finished captures. A hit requires the model's
  tensor content (including device and `requires_grad`), training flags, module tree and
  `forward` code, plain instance attributes, user-registered module hooks, inputs, and capture
  configuration to all match. Entries are HMAC-authenticated, bounded by entry count and bytes,
  and `tl.clear_cache()` empties the cache while preserving its secret.

## Replay and intervention

**Intervention**
: A value or gradient edit performed at a selected live site. `tl.when(predicate, helper)` binds a
  selector to helpers such as `tl.zero_ablate()` or `tl.scale(...)`. `spec.bind(model)` returns a
  bound executor that applies a spec to live calls. A synchronous generator or iterator returned
  by a bound call (for example `bound.generate(...)`) stays steered for every step, and its report
  settles on `last_report` when the iteration is exhausted, closed or raises; an async lazy output
  (async generator or iterator, coroutine, awaitable) refuses `bind_lazy_output`. A
  module-boundary rule (`tl.module(address)`) edits the tensor leaves of a module output that
  is a tensor or a supported container (tuple, list, dict, namedtuple, dataclass, HF model
  output); an output of another type (a distribution object) holds no addressable tensor
  site, so the rule records zero fire there at both doors, bound executor and capture.

**Zero-fire policy (`on_zero_fire`)** *(spelling documented-unstable)*
: One knob on every intervention door: `spec.bind`, `tl.trace(intervene=)` and `Trace.run`. An intervention that fires at zero sites is always
  recorded on the door's own record (the bind report's `zero_fire_rule_ids`, the capture's
  `annotations["unmatched_capture_selectors"]` and intervention audit, the rerun's
  `last_run["hooks_unfired"]`). The knob then decides the same way everywhere: `"warn"`
  (default) emits the door's coded warning (`bind_zero_fire`, `trace_zero_fire`,
  `rerun_zero_fire`), `"error"` raises the door's typed refusal with that code, and
  `"disclose"` records only. Any other value refuses `zero_fire_policy_invalid`. A rerun edit
  that cannot fire live (`rerun_selector_unmatchable_live`) is refused before the policy runs,
  whatever the knob says. When the rule targeted a module (a plain `tl.module` union at
  `tl.trace(intervene=)`, a module entry at a rerun, a module rule under `spec.bind`) whose
  output is not a tensor, the warning, the refusal and the fast rerun's
  `fast_rerun_graph_unsteered` refusal carry
  `fields["reason"] == "module_output_not_tensor"` and a `reason_detail` sentence naming the
  output type and, for a `torch.distributions` output, the parameter output paths to target
  instead (`module 'policy' output is a Normal, not a tensor; target its parameter ops instead
  (output paths .loc, .scale)`).

**Bound intervention executor** *(spelling documented-unstable)*
: `spec.bind(model)` pairs a `tl.Recipe` with a base model as a capture-free callable and
  returns a `tl.intervention.BoundModel`; its read-only `BoundModel.recipe` is the bound
  `tl.Recipe` and `BoundModel.model` the base model.
  It runs the model with the spec's edits held live, returns the model's own output, records
  only a firing report (`.last_report`), and supports HF `generate` with the KV cache on or off.
  It is not an `nn.Module` and bindings never nest; a rule that never fires is recorded on
  `.last_report` and settles by `on_zero_fire` (see **Zero-fire policy**). Decode steps: the binding counts root forwards, so
  `torchlens.at_step(k)` on a bound call or `generate` names the k-th forward
  (step 0 is the prefill), on op-level rules and as `tl.module(address) & at_step(...)`
  boundary rules, capture-free; outside a binding or an episode capture the qualifier still
  refuses `episode_step_selector_without_episode`. Gradient sites (`tl.grad(...)`) refuse
  `bind_rule_unsupported`: the lane replaces forward values only; edit gradients with
  `trace.do(tl.grad(site), edit, model=model, inputs=x, loss=...)`, or capture with
  `backward_ready=True, intervene=tl.when(tl.grad(site), edit)` and call `trace.backward(loss)`.

**Recipe and Rule** *(spelling documented-unstable)*
: `tl.Recipe` is the immutable multi-clause intervention spec that `tl.when(...)` builds and every
  intervention door accepts. `Recipe.rules` is the tuple of its clauses, each a
  `tl.intervention.Rule` with `target` (the selector or predicate) and `edit` (the helper or
  callable). `Rule.signal` is a read-only property derived from the target, never a field and
  never saved: `"backward"` for a gradient site (`tl.grad(...)`) or a node selector, `"forward"`
  for an activation site, `"both"` for a union of the two. The site decides the pass: `Rule`
  has no `direction` field, and `Rule(..., direction=...)` (or the old third positional
  value) refuses `intervention_direction_invalid`. `Recipe.merge(*others)`
  (also `recipe_a & recipe_b`) returns one `tl.Recipe` joining the clauses; a duplicate clause
  refuses `spec_rules_duplicate`.

**Trace.recipe and Trace.clear_edits**
: `Trace.recipe` is a frozen snapshot of the trace's current intervention recipe.
  `Trace.clear_edits(site=None, handle=None, *, strict=False, confirm_mutation=False)` removes the
  sticky edits staged with `do` by site or by handle id, marks the recipe stale and
  returns the trace; with `strict=True` it refuses a call that names neither or matches nothing.
  Removal (also through the handle's `remove()`) takes effect at once, like a torch hook handle: a
  rerun or backward pass in progress skips the removed hooks from the next hook site it reaches,
  while the site firing at that moment finishes its queued hooks; afterwards the recipe and the
  handle are the same as after a removal between runs.

**Region instances** *(spellings documented-unstable)*
: A region edit's admissible region is partitioned into pass instances. `RegionTarget.labels`
  lists the region's pass-qualified member op labels in execution order; each `RegionInstance`
  carries `RegionInstance.labels` (that instance's member labels) and
  `RegionInstance.exit_labels` (its exit op labels, in the order the replacement function must
  return its tensors; the `labels=` and `exit_labels=` constructor fields). A replacement
  returning several exits returns a tuple of
  `len(exit_labels)` tensors.

**allow_changed_params** (`InterventionOptions`, `ReplayOptions`) *(spelling documented-unstable)*
: A replay template holds the live `nn.Parameter` each call consumed, and capture records its
  version counter, storage pointer and shape. If a parameter changed since capture (an optimizer
  step, `load_state_dict`, an in-place op under `torch.no_grad()`, `param.data = t`), every replay
  door refuses `replay_param_changed_since_capture` (`ReplayPreconditionError`, naming the
  parameter and the op) rather than silently compute with the new value: `do()` on the replay
  engine, `push`, `push_from`, differentiable replay, and later replays of every fork (a rerun
  fork checks against the weights its rerun ran with). `allow_changed_params=True` (default
  `False`) replays with the current values and records the parameter names in
  `trace.last_run["replayed_with_changed_params"]`. Not detected: an in-place write through
  `param.data` or through a numpy or raw-storage alias. A replaced parameter object
  (`module.weight = nn.Parameter(t)`) is not used by replay at all: the template keeps the
  captured object. For an as-if weight edit use `fork.do(tl.param(name), edit)`. The rerun
  engine runs the live model and ignores the option.

**torchlens.intervention.subspace_ablate(basis, *, mode="zero", dim=-1, mean=None, seed=None)** — *documented-unstable*
: Edit acting on the component of a site value inside the span of a `[k, d]` basis (rows need
  not be orthonormal; a rank-deficient or non-finite basis refuses `subspace_basis_invalid`).
  With `P` the orthogonal projector onto the span: `mode="zero"` gives `x - P x` (the
  k-dimensional `project_out`), `mode="mean"` gives `x - P x + P mean`, and `mode="resample"`
  gives `x - P x + P x[perm]` for a seeded batch permutation (`seed=` required). Inconsistent
  mode arguments refuse `subspace_mode_invalid`; a basis over the wrong width refuses
  `subspace_basis_dim_mismatch` at fire time. Zero mode on the span of class-mean differences
  is the LEACE mean-equalizing erasure.

**torchlens.intervention.set_affine_coordinates(weight, *, bias=None)** — *documented-unstable*
: Source verb for a subspace edit: under `do(tl.subspace(site, R.T, origin=...), set_affine_coordinates(W,
  bias=b))` the new coordinates are `W x + b` of the whole site value `x`, so the site gets
  `x + Q^T (W x + b - Q x)` with `Q` the basis rows (LoReFT, `h + R (W h + b - R^T h)`). `weight` is
  a finite real `[k, d]` tensor shaped like the basis rows, `bias` `None` or a finite real `[k]`
  tensor; anything else refuses `affine_source_invalid`. Outside a subspace (a site label,
  `tl.units`, a capture or replay hook) it refuses `affine_source_needs_subspace`. Builtin: it
  saves inside the `subspace_edit` payload, `weight` and `bias` as tensor sidecars. TorchLens
  applies a LoReFT edit trained elsewhere and never fits one.

**torchlens.intervention.subspace_interchange(basis, source, *, dim=-1)** — *documented-unstable*
: k-dimensional basis interchange: `x - P x + P s`, with `s` the source trace's value at the
  same site (same-site donor resolution as `tl.patch_from`); applies a learned rotation the
  user trained elsewhere.

**torchlens.intervention.keep_grad(edit)** — *documented-unstable*
: Returns the same forward edit (an `Edit` or a hook callable) with the gradient-connectivity
  check: at fire time it raises `InvalidArgumentError` / `edit_detached_from_graph` (with the
  site label, a `reason` of `edit_detached`, `site_has_no_graph` or `grad_disabled`, and a
  spelling that keeps the graph) when the edit writes values carrying no autograd graph, when
  the site value has none, or when gradients are disabled. `keep_grad(tl.scale(0.0))` is the
  graph-keeping zero; `keep_grad(tl.intervention.compose(tl.scale(0.0), tl.add(value)))` the
  graph-keeping fixed value. A wrapped `Edit` carries `("keep_grad", True)` in its metadata and
  `portability="opaque_audit"`. A backward edit or a non-edit refuses `keep_grad_edit_invalid`.
  Under differentiable replay (`push(replay=ReplayOptions(differentiable=True))`) a passing
  check warns once per session with code `keep_grad_replay_frontier`: parameters upstream of the
  replay frontier get no gradient; the live door `tl.when(site, keep_grad(edit)).bind(model)(x)`
  reaches them. Which verbs keep the graph under which door: docs/gradient_contract.md.

**torchlens.positions(edit, positions, *, mask=None, assume_no_padding=False, axis=1)** — *documented-unstable*
: Restricts a same-shape edit to chosen token positions of a `(batch, tokens, ...)` site value
  (token-position editing): `positions` is an int, a list of ints, a slice, a per-row
  `(batch,)` index tensor, or `"first"` / `"last"`, the first or last REAL token of each row
  resolved from `mask=` under the token-pooling mask policy (no mask and no
  `assume_no_padding=True` refuses `transform_mask_missing`; a mask that does not describe the
  value refuses `transform_mask_incompatible`). Unsupported spellings refuse
  `position_edit_invalid`; positions past the sequence refuse `position_edit_out_of_range`.

**tl.intervention.fit_direction(pos_traces, neg_traces, site, method="mean_diff", *, read_token_index=-1)** — *documented-unstable*
: Builds one contrastive steering direction from saved activations. `pos_traces` and
  `neg_traces` are a trace or a sequence of traces with one prompt per batch row; row `i` of the
  negatives pairs with row `i` of the positives. One token per prompt is read at `site`
  (`-1` is the last real token when the trace saved its `attention_mask` input; a sequence gives
  one position per prompt; `None` reads `[n, hidden]` outs unsliced). With `D = pos - neg`:
  `"mean_diff"` is `D.mean(0)` on the activation scale; `"pca"` is the first uncentered
  principal axis of `D`, unit norm, signed toward the positives; `"logistic"` is L2 logistic
  regression (`C = 1`, no intercept) of `D / 2` against `-D / 2`, solved with torch only, unit
  norm. Also `torchlens.intervention.fit_direction`. Returns a `Direction`. Refusals use
  the `direction_*` codes of the error contract.

**Direction (`torchlens.intervention.Direction`)** — *documented-unstable*
: The frozen result of `fit_direction`: `vector` (1-D, the rows' dtype), `site`,
  `method`, `num_pairs`, `read_token_index`, `model_type` (the traced model's
  `config.model_type`, or `None`) and `provenance`, a
  `tl.types.BasisProvenance` whose origin records the site, method, pair
  count, read position and a sha256 digest of the rows, and whose `sha256` is the digest of
  the canonical float64 vector. `direction.steer(m)` is `tl.steer(vector, m, dim=-1)`;
  place it with `tl.when(tl.module(direction.site), ...)`. `direction.to_steering_vectors()`,
  `direction.to_repeng()` and `direction.to_dialz()` convert it to a real
  `steering_vectors.SteeringVector`, `repeng.ControlVector` and `dialz.SteeringVector` without
  recomputing it (thin converters `tl.bridge.steering_vectors.from_direction`,
  `tl.bridge.repeng.from_direction` and `tl.bridge.dialz.from_direction`), keyed by
  `direction.library_layer(layer=None)`: the site label's trailing index unless `layer=` is
  given; a different explicit index refuses `direction_layer_mismatch`. Applied with the same
  multiplier, each library object gives the native steer's logits.

**`Direction.upload_to_neuronpedia` (publish a vector)** — *documented-unstable*
: `direction.upload_to_neuronpedia(*, api_key, model_id, name, layer=None,
  hook_type="resid_post", default_steering_strength=1.0)` publishes the vector on neuronpedia.org under
  the account that owns `api_key` through Neuronpedia's "Create Vector" endpoint and returns
  the vector page's URL. `model_id` is Neuronpedia's model id, `name` the name it shows,
  `layer` defaults to `direction.library_layer()`, `hook_type` is `"resid_post"` or
  `"resid_pre"`, `default_steering_strength` the default steering strength (-100 to 100). Every field
  is checked before the request (`neuronpedia_upload_argument_invalid`); the key never
  enters a message.

**Bundle**
: A named collection of aligned Traces, constructed with `tl.Bundle(...)` or `tl.Bundle(...)`, for
  cross-run comparison. `tl.Bundle()` with no members builds an empty bundle to fill later with
  `add`; an empty bundle saves and loads like any other.

**Bundle.relations** *(spelling documented-unstable)*
: The bundle's typed, directed relation rows between members, read as an immutable tuple of
  `MemberRelationRow` (plus `OpaqueRelationRow` for unknown namespaced kinds loaded from an
  artifact). Pair kinds: `alternative_of` (peers; still comparable), `forked_from`,
  `successor_of`, `escalates`, `fed_by`, `selected_by`; member kind: `episode_member`. Each kind's
  params are a
  closed set; rows persist in `bundle.json` and every load re-checks them. On a pair row `from` is
  always the LATER member. Rows are ordering and topology claims: members joined by a directed
  path of ordering kinds are not comparison operands.

**Bundle.relation_grades** *(spelling documented-unstable)*
: `b.relation_grades()` returns one `RelationGrade` per relation row, in table order: the row's
  grade DERIVED at read from its evidence items and declared params, never stored. The ladder,
  weakest to strongest: `unchecked` (no check ran; the contracted reasons are listed),
  `disclosed` (declared only: a bare row, a `carry_mode="none"` fresh start, disclosed items),
  `consistent` (every check that ran agrees, not proven), `verified` (every item verified). Any
  divergent item grades the row `divergent`. `broken` fires on exactly two premises, named in
  `broken_premises`: `bulk_origin_failure` (more than half of the checked member-origin items
  diverge) and `verbatim_carry_failure` (`carry_mode="verbatim"` with a divergent
  `declared_carry` item). A divergence at a declared seam (the item's `seam` is in the row's
  `seams`) is counted in `seam_local_divergences`, grades `divergent` on record and never
  escalates. `OpaqueRelationRow` grades `unchecked` with reason `kind_unregistered`.

**RelationGrade** *(`from torchlens.bundle import RelationGrade`; spelling documented-unstable)*
: Frozen record of one row's derived grade: `kind`, `members`, `grade`, `broken_premises`,
  `unchecked_reasons`, `seam_local_divergences`, `exogenous_items` (items whose class lies
  outside the member-measured kinds; derived, never written to an item), `items_by_class`,
  `items_by_grade`.

**successor_of.evidence item class** *(evidence item key `class`; spelling documented-unstable)*
: The ground-truth origin of one evidence item, from the closed vocabulary of seven kinds:
  member-measured `member_input`, `member_output`, `member_state`, `member_params`, and
  exogenous `declared_carry`, `declared_transform`, `harness_record`. Required on every item of a
  reserved schema (`version_boundary_v1` admits the four member kinds plus `declared_transform`;
  `turn_boundary_v1` admits `member_input`, `member_output`, `member_state`, `declared_carry`,
  `declared_transform`, `harness_record`); optional under a user schema, but load-validated
  against the seven when present (`bundle_relation_schema_invalid`). "Exogenous" is a rollup
  read from the class, never a persisted value.

**successor_of.carry_mode / seams / item seam** *(relation params; spelling documented-unstable)*
: `carry_mode` is the closed vocabulary `none` (a declared fresh start: nothing carried across
  the boundary), `verbatim` (state carried unchanged) and `summarized` (state carried through a
  declared transform); any other value refuses at load. `seams` is a list of distinct names for
  the boundaries where the chain is EXPECTED to change (a chat-template revision, a tokenizer
  swap); an evidence item may name one of them under `seam`, and a divergence there never
  escalates the row to `broken`. An item `seam` that the row does not declare refuses at load.

**Bundle.relate** *(spelling documented-unstable)*
: Appends relation rows and returns the bundle. Positional form, read as the sentence it records:
  `b.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 3, 7]], label="critique")` is
  "critique fed_by draft" (kind, the later member, the earlier member, then the kind's params; a
  member kind names one member). Row form: `b.relate(MemberRelationRow(...))` or payload mappings.
  Refusals carry stable codes (`bundle_relation_schema_invalid`, `bundle_relation_member_missing`,
  `bundle_relation_auto_spans_unavailable`, `bundle_relation_auto_spans_ambiguous`,
  `bundle_relation_auto_spans_unmatched`, `bundle_relation_span_out_of_bounds`,
  `bundle_relation_span_unverified`); a refusal leaves the table unchanged.

**fed_by** *(relation kind; spelling documented-unstable)*
: "B fed_by A": the later member B's prompt contains the earlier member A's output. Content
  provenance only: a router whose decision selects a call without its output entering that call's
  prompt is not `fed_by`; record that decision with `selected_by`. Optional params: `token_spans`
  (rows of four ints, half-open: source start, source end, destination start, destination end),
  `basis` (`declared`, `verified`, `matched`), `coverage` (a fraction in [0, 1]) and `label` (free
  human text such as `"critique"` or `"judge"`; nothing in TorchLens interprets it).
  `token_spans="auto"` asks TorchLens to find the spans: the source's eligible output (its bound
  completion range minus `excluded_ids`) is searched as exact runs of token ids in the reader's
  prompt, trimming up to two boundary tokens per side when a run is not found whole, and the row
  is written with `basis="matched"` and a computed `coverage`. Several placements (a repeated
  quote, or a tie between two trims) refuse `bundle_relation_auto_spans_ambiguous`; no placement
  refuses `bundle_relation_auto_spans_unmatched`; missing evidence (no source completion range,
  ids not retained) refuses `bundle_relation_auto_spans_unavailable`. `fed_by` is an ordering kind
  for the comparison gate.

**selected_by** *(relation kind; spelling documented-unstable)*
: "B selected_by A": the earlier member A's decision (a router's route, a judge's verdict)
  selected the later member B, deciding whether B ran or which input it got. Control only: A's
  text need not appear in B's prompt (text that does is `fed_by`), and the row never says A
  caused B's content. `b.relate("selected_by", "writer", "router", chosen=1, candidates=[0, 1])`
  persists `from="writer"`, `to="router"`, like every ordering kind. Optional params, a closed set:
  `chosen`, `candidates` and `evidence`. Malformed params refuse
  `bundle_relation_schema_invalid`. An ordering kind: `upstream`/`downstream` walk it and the
  comparison gate refuses the selected call and its decider as operands. Admitted from tlspec 10;
  an artifact stamped earlier that carries the kind refuses.

**selected_by.chosen** *(relation param; spelling documented-unstable)*
: The decision, as one JSON scalar: a bool, an int, a finite float or a non-empty string of at
  most 256 characters (route `1`, candidate `"c2"`). Compared by type and value, so `1` and `True`
  are different decisions. When `candidates` is also given, `chosen` must be one of them.

**selected_by.candidates** *(relation param; spelling documented-unstable)*
: The options the decider chose among: a non-empty list of distinct JSON scalars (same scalar
  rule as `chosen`). They may be member names, routes or tool names; TorchLens never resolves
  them as members.

**selected_by.evidence** *(relation param; spelling documented-unstable)*
: Why the decider chose, as the graded evidence envelope `successor_of` admits
  (`{schema, items, facts_digest}`; each item a graded claim with a nullable `basis`), for
  example the router's scores. User-written measurement grades (`verified`, `consistent`,
  `divergent`) are stored as `disclosed` with the original under `declared_grade`, and the
  envelope keeps the 1 MiB per-row budget (`bundle_relation_evidence_over_budget`).

**Token-stream binding** *(`token_boundary_v1`; spelling documented-unstable)*
: What makes `fed_by` spans checkable: a per-member descriptor in the reserved
  `"torchlens.boundary_facts"` sidecar family naming the captured stream the coordinates index
  (`"input"`, the first integer-dtype input; or `"output"`), the `batch_row`, the measured
  `length` and `ids_digest` of its token ids (filled in by `relate` when left out), the member's
  own `completion` range, the `excluded_ids` outside the eligible set (padding, special tokens),
  and `tokenizer` / `template` identity strings. Attach one with `torchlens.io.attach_sidecar`
  before relating; with none, `relate` derives an input-stream binding for a batch-of-one capture
  whose ids are retained. A row reads `basis="verified"` only when both members are bound, each
  digest matches the retained ids, the source binding names its completion range, every source
  span lies inside it, and the token ids agree span by span; anything less is `declared`. A
  claimed `verified`/`matched` row re-checks at every door (relate, constructor, load): missing
  evidence makes it `declared`, contradicting evidence refuses `bundle_relation_span_unverified`.
  A `matched` row also re-runs the matcher on the members and keeps `matched` only when it finds
  the same spans; spans that check out but are not what the matcher finds are stored as
  `verified`. `coverage` on a verified or matched row is
  computed: covered eligible source tokens over all eligible ones (the bound completion range
  minus `excluded_ids`).

**Bundle.upstream / Bundle.downstream** *(spelling documented-unstable)*
: `b.upstream(name)` lists every member `name` transitively came after (what fed it), and
  `b.downstream(name)` every member that came after it, nearest first, over the ordering kinds
  (`fed_by`, `selected_by`, `successor_of`, `forked_from`, `escalates`). Bundle methods, like
  `compare`.

**Bundle.token_pairs** *(spelling documented-unstable)*
: `b.token_pairs(reader, source)` returns the verified one-to-one token positions of "reader
  fed_by source", arguments in `relate`'s order (the later member, then the earlier one). The
  result's `pairs` are `(source_pos, reader_pos)` tuples, each position at most once, with
  `source_positions` / `reader_positions` lists for indexing activations; `coverage` is the paired
  share of the source's eligible tokens (`None` without a bound completion range); `excluded`
  lists what was not paired, each entry a dict with `side`, `start`, `end`, `reason` and
  `row_index`. Only spans of `verified` or `matched` rows that re-check at call time pair.
  Declared spans (retokenized many-to-many text, a token mixing a template delimiter with copied
  text), excluded ids, positions claimed twice, streams whose bindings name different tokenizers,
  and eligible source tokens no verified span covers are reported, never paired. Equal lengths are
  never evidence of matching meaning. A verified row whose evidence now contradicts it refuses
  `bundle_relation_span_unverified`. A Bundle method; any reduction over a
  many-to-one group is the caller's own descriptive comparison.

**Bundle.token_divergence** *(spelling documented-unstable)*
: `b.token_divergence(original, rerun)` returns a `TokenDivergence`: where two members' prompts and
  completions part, by exact token ids: `original=` is the member from the original run (the
  reference), `rerun=` the member from the re-run compared against it. Each member needs a
  token-stream binding that names its
  `completion` range, with its token ids retained; the prompt is the bound stream before the
  completion. Typical use is a call and its copy from a steered re-run
  (`b.token_divergence("critique", "cf_critique")`), but any two bound members compare. Refuses
  `bundle_token_divergence_unavailable` (fields `member`, `reason`) when a binding, its completion
  range or the ids are missing; an unknown name raises `KeyError`. A Bundle method.

**TokenDivergence** *(`from torchlens.bundle import TokenDivergence`; spelling documented-unstable)*
: The frozen result of `Bundle.token_divergence`. `original`, `rerun`: the member names.
  `prompt_divergence`: the first prompt position whose ids differ (the shorter prompt's length when
  one extends the other; `None` when equal). `completion_divergence`: the same for the completions,
  counted from the first generated token. `lengths`: `((prompt, completion), (prompt,
  completion))`. `shared`: the stretches both runs share as `(orig_start, orig_end, rerun_start,
  rerun_end)` half-open stream positions, the `token_spans` grammar: the common prompt prefix, the
  common prompt tail after a divergence, the common completion prefix. `intervened`: whether each
  capture ran under an intervention. Properties `prompt_same` and `completion_same`.

**Bundle.add_alternative** *(spelling documented-unstable)*
: `b.add_alternative(second_run, steered=("draft",), prefix="cf_")` merges a second run of your
  own harness, made with the `steered` calls captured under their specs, into `b` and returns an
  `AlternativeReport`. Calls pair by member name, never by order. Before anything changes it
  refuses, with these codes: `alternative_target_missing` (a steered call absent from either run;
  field `side`), `alternative_target_unsteered` (a declared call captured without a spec that
  fired), `alternative_steer_undeclared` (another second-run call captured under a spec),
  `alternative_capture_position_dependent` (the spec edits chosen positions or decode steps:
  `at_positions`, `at_step`), `alternative_member_name_taken` (a copy would reuse a name; pass
  another `prefix`), `bundle_token_divergence_unavailable` (a paired call lacks a bound completion
  range) and `alternative_unchanged_call_diverged` (an unsteered call saw the same prompt but wrote
  different text: noise TorchLens cannot separate from the steer). Then every second-run member is
  added as `prefix + name`, each paired call gets an `alternative_of` row from its copy, the second
  run's own relation rows are copied onto the copies, and one `add` operation records the statuses.
  A failure while merging restores the bundle. TorchLens never runs your harness. A Bundle
  helper-table method.

**AlternativeReport** *(`from torchlens.bundle import AlternativeReport`; spelling documented-unstable)*
: The frozen result of `Bundle.add_alternative`: `prefix`, `steered` (the declared calls), `calls`
  (one `AlternativeCall` per call of either run, the original's order first), `uncopied_rows` (the
  second run's rows of unknown namespaced kinds, not copied) and `table()`, a fixed-width text
  table with one line per call.

**AlternativeCall** *(`from torchlens.bundle import AlternativeCall`; spelling documented-unstable)*
: One call of an `AlternativeReport`: `call` (the original member name), `member` (its copy,
  `None` when dropped), `status`, and `prompt_divergence` / `completion_divergence` as on
  `TokenDivergence` (`None` for unpaired calls). Statuses: `steered` (declared in `steered=`),
  `unchanged` (prompt and output the same), `absorbed` (the prompt changed, the output did not; the
  common case), `changed` (both changed), `added` (only the second run made the call) and
  `dropped` (only the original made it).

**Bundle.at**
: `b.at(site)` returns the cross-member `Super*` view of one site. A string is a label from any
  Bundle accessor family; any other selector-like site (`tl.func("relu")`, `tl.module(address)`)
  resolves to one op in every member after the model-axis floor (`same_param_shapes`) passes for
  every member pair (a view read, so different inputs are allowed). `b.at(site).compare()` is the
  member-by-member matrix at that site and first runs the comparison gate over the members it
  reads. To read a subset of members, build a bundle of them:
  `tl.Bundle({name: b[name] for name in names}).at(site)`, so one foreign member does not disable
  the view for the rest.

**Bundle.draw** *(`nodes=` spelling documented-unstable)*
: `b.draw(path=None, *, graph="forward", nodes="ops", rolled=False, direction=None, theme=None,
  show_legend=None, file_format=None, open_viewer=False, graph_overrides=None,
  node_overrides=None, edge_overrides=None, members=None, metric="relative_l2", on="out")` draws
  the bundle as one picture and returns a `tl.viz.Drawing` (`str(drawing)` is the DOT source,
  `drawing.path` the written file). It follows `Trace.draw`'s rules (JMT decision, Sun 1 Nov 2026,
  3:00 PM PT): the path comes first and its extension picks the format (`b.draw("diff.svg")`
  writes SVG); `file_format=` applies only to a path with no extension (`<path>.<file_format>`,
  `"pdf"` by default); a `file_format` that disagrees with the extension refuses with
  `ArgumentConflictError` (`draw_format_mismatch`). Outside a notebook a bare `b.draw()` writes a
  collision-safe `bundle_modelgraph.<format>` and prints its path; in a Jupyter kernel a bare call
  writes no file and returns the drawing for inline display. No viewer opens unless
  `open_viewer=True`. `graph="forward"` (default), `"backward"` (each member's captured grad_fn
  graph) or `"combined"` draw the bundle's union op graph for that pass, with a member list on
  each node and arrow that not every member runs; `rolled=True` draws each repeated module once;
  `direction=None` is `"bottomup"`; `theme` takes the `Trace.draw` skin names (`None` is
  `"torchlens"`). `node_overrides=` (Bundle-only) keys Graphviz node attributes by op label
  (member name under `nodes="members"`), `edge_overrides=` by `(source, target)`. `nodes="ops"`
  (default) draws the ops; `nodes="members"` draws one node per member and one edge per relation
  row, from the earlier member to the later one, labelled with the kind, the row's `label`, on
  `fed_by` its basis and coverage, and on `selected_by` its `chosen` decision; solid edges carry
  token spans TorchLens checked, every other row is dashed, and `graph` must stay `"forward"`.
  `graph="diff"` draws two members side by side, one node per op, coloured by `metric` distance
  on the `on` field; each pass of a layer that runs more than once is its own node with its own
  distance, the per-pass rows of `Bundle.compare`: `members=(left, right)` names them (default the baseline and the
  next member); its defaults are `theme="paper"`, `"svg"` and a `bundle_diff` file, and it runs
  the comparison gate on the rendered pair only. Any other keyword (a `Trace.draw` keyword the
  bundle renderer does not support yet, such as `depth=`) refuses with
  `ArgumentTypeError` (`bundle_draw_keyword_unsupported`) naming the supported list; each is added
  when the renderer supports it (JMT decision, Sun 1 Nov 2026, 3:00 PM PT). Replaces the removed
  `show_bundle_graph` and its `signal=` keyword (now `graph=`; no alias).

**Bundle.show**
: `bundle.show(kind=None, **kwargs)` is the one bundle display door. With no kind (or
  `kind="graph"`) it is `bundle.draw(**kwargs)` and returns that `Drawing`, the merged graph of
  where the members differ. `kind="strip"` draws each member's own graph with
  `Trace.draw(**kwargs)` and returns a dict of member name to `Drawing` (a `path` gets
  `_<member>` appended per member); `kind="repr"` and `kind="html"` return each member's `repr` /
  HTML text by member name. Any other kind refuses with `InvalidArgumentError`
  (`bundle_show_kind_invalid`).

**Bundle.compare**
: `bundle.compare(a, b, *, metric="cosine", on="out")` returns the per-site differences between
  two members as `(site_label, distance)` rows over the sites both members share, paired through
  the structural alignment; `metric` is a metric name or callable (scalar-like sites always use
  `relative_l1`) and `on` reads `"out"` or `"grad"`. The receiver sets the axis: `bundle.at(site)
  .compare()` is the member-by-member matrix at one site. The pair goes through the comparison
  gate first and refuses typed when the members cannot be aligned; its grade is
  `bundle.comparability(a, b)`, a `tl.types.Relationship`.

**Bundle.store_comparison / Bundle.stored_comparison / Bundle.stored_comparison_names**
: `b.store_comparison(metric="relative_l2", *, name=None, baseline=None, on="out")` computes the
  per-node distances of every member from the baseline once (the configured baseline, else the
  first member) and stamps them on the supergraph under the returned name (the metric's name
  unless `name` is given; a callable metric needs an explicit name); `b.stored_comparison(name)`
  reads one back without recomputing (`comparison_unknown` for a name never stored) and
  `b.stored_comparison_names()` lists the stored names. A layer that runs more than once has
  one value per member: the metric over its passes' tensors joined in pass order. Session-level
  state on the built supergraph, never persisted. Every (baseline, member) pair goes through the comparison gate
  first, the same predicates as `bundle.at(site).compare()`: no ordering path between them, a
  shared graph, and sites that line up (proven-equal input values, or for runs on different inputs
  or checkpoints, equal graph-shape hashes); otherwise the read refuses with the gate's
  `bundle_gate_*` code.

**Bundle.episode_status** *(spelling documented-unstable)*
: `b.episode_status(episode_id, *, episode_ledger=None)` folds the outcomes of the members named by
  `episode_member` rows into a derived episode status (`EpisodeFoldResult`). A derivation, never
  a stored Bundle outcome; without a ledger the ledger-only arms degrade to `episode_unknown`.

**Bundle.why / Bundle.provenance** *(spelling documented-unstable)*
: `b.why(member, relative_to=None)` returns a `WhyReport`: the exact-or-refused ancestry of a
  member against the baseline (or `relative_to`), with lineage status, payload fidelity, value
  residual and comparability reported as separate axes. `b.provenance.summary()` is the per-member list of
  the same facts. Both are derived views; nothing is stored. Reference:
  `docs/reference/experiments.md`.

**Bundle.vary** *(spelling documented-unstable)*
: `b.vary(mapping, *, unmentioned=None)` applies one explicit intervention spec (or `None`, the
  identity) per named member, refusing incomplete coverage before any member changes; a mid-apply
  failure refuses `vary_partial_failure` and discloses what completed (no rollback). Broadcast
  `do()` is unchanged.

**Parameter value basis** *(spelling documented-unstable)*
: The derived, read-time, persisted-nowhere disclosure `Param.value_basis` of where a parameter
  value comes from: `live_ref` (the live-model handle resolves; the value may have moved since
  capture) or `absent(not_persisted)` (deserialized; parameter bytes were never persisted —
  replacing the historical untyped bare `None`). `snapshot` arrives only with capture-time
  parameter snapshots (R8(b)). `Param.value` keeps its documented live-handle meaning; the basis
  is a disclosure beside the live read, never a change to it.

**Checkpoint live-ref guard** *(spelling documented-unstable)*
: TorchLens records which parameter a run used, not its bytes, so cross-member parameter
  value/difference/trajectory reads on a Bundle (`weight_norm_diff`, `compare`, `aggregate`,
  `out`/`grad` on `SuperParam` views with two or more members) refuse BEFORE tensor lookup with
  the stable code `checkpoint_series_live_params` unless every member carries immutable
  capture-time parameter evidence. The guard is keyed on the claim, never on Python object
  identity, and survives save/load; a version-axis relation row still orders members.

**Runnable artifact**
: A `.tlspec` saved at `level="runnable"` with a sparse taken-path descriptor. Its source capture
  must use `replayable=True`: the flag records the runnable apparatus (callable registry keys,
  state snapshots, alias topology, escape records), which no default capture keeps; the per-op
  argument templates that `trace.do(site, edit)` replays are recorded on every capture.

**Readiness**
: The non-executing runnable state `ready` or `unavailable`.

**Path faithfulness**
: A runnable result verdict: `verified`, `diverged`, or `unverifiable`. `unverifiable` means the run
  completed without enough evidence for the stronger claim; it is not a pass.

**Numeric attestation**
: Byte-level comparison of eligible archived selected activations. Its status is `attested`,
  `numeric_attestation_failed`, `not_applicable`, or `not_present`; attestation never upgrades path
  faithfulness.

## Capture honesty and visualization

**Capture verification reason**
: The machine-readable `Trace.provenance.verification_reason` explaining why a live capture is not
  positively verified, for example `mode_rescue_rerun`, `escape_rescue_unrecovered`,
  `owner_thread_tripwire_changed`, or `dynamo_region_not_logged`.

**Pristine (`Trace.is_pristine`)** *(spelling documented-unstable)*
: Whether nothing ever intervened on the trace, read from the evidence it carries: `True`,
  `False`, or `None`. It is derived at read time from records that persist with the trace (the
  intervention audit, the operation history, mutation states, direct writes and op intervention
  marks), never stored as a flag, so it survives `Trace.save`/`tl.load`, pickle and `fork()`, and no
  reset makes a touched trace read `True` again. `False` means some touch was recorded: a
  capture-time `intervene=` (including one that fired nowhere), a staged hook or value
  replacement (even after `clear_edits(site)` removes it), `do()`, `set()`, a region edit, a
  hooked rerun or run, a direct write. Plain reruns, plain runs, forks and `clear_edits(site)` on
  an untouched trace are not touches. `None` means no evidence: the artifact predates tlspec 9 or the backend is not
  torch, and it is never a pristine reading. Bundle's pristine label and baseline
  auto-detection count a member only when it reads `True`. It is a reading, not a verdict: it
  never changes validation or a run report. Not `Trace.state`: `TraceState.PRISTINE` is an
  operational state that an assignment can set and that a func-targeted `intervene=` capture
  keeps. Not validation's pristine-torch oracle, which means running on unwrapped torch
  functions. How the reading relates to the runnable clean-run promise (a hooked run
  settles `unverifiable` and marks the trace non-pristine) is stated in the maintainer page
  `runnable_model.md`, "Clean run, hooked run, and disclosure".

**Rescue re-run**
: A single disclosed second forward under a `TorchFunctionMode` net that redirects detected stale
  pre-wrap torch references to their wrappers. The capture-honesty facts record `rescue_rerun` and reason
  `mode_rescue_rerun`.

**Collapse**
: A rendering-only condensation of module detail. `collapse="none"`, `"auto"`, `"max"`, or a
  float in `[0, 1]` never changes the underlying Trace. Admission gates on the rendered
  universe U against the defensive constant `COLLAPSE_OPTIMIZER_MAX_OPS` (2000) plus a
  measured (U, W) work estimator: over-budget requests degrade to a deterministic compact
  fallback plan (coded `collapse_budget_fallback`), never an uncollapsed wall; only
  pathological inputs (raw ops above 20x the constant) decline outright
  (`collapse_pathological_skip`; `Drawing.collapse_plan` refuses typed
  (`collapse_plan_unavailable`); `Drawing.collapse_schedule` degrades to its single
  full-graph step).

**Module.collapse_score / Trace.module_collapse_order** — *unstable — no deprecation shim owed*
: A module's smart-collapse priority: its hidden ops (distinct subtree ops minus one) over the
  largest submodule's, rounded to 6 places; 0.0 for the root, a module of at most one Layer, or the
  only child of a non-root module that hides the same ops (the parent, the name the user wrote,
  ranks instead; in a chain of such wrappers the outermost ranks). It ranks candidates and never reads the selected plan
  (`Drawing.collapse_plan` may skip a high-priority module); `module_collapse_order` lists every
  module's priority, highest first. Computed at access, never persisted.

**Run folding**
: `fold_repeats=True` replaces eligible runs of distinct same-class sibling modules with a
  representative and an honest `+N more` label. It differs from recurrence, where `(xN)` means the
  same parameters executed repeatedly.

**Receptive field / Projective field**
: Influence geometry through the captured DAG toward inputs / toward outputs. Entity-level
  `op.receptive_field` and `op.projective_field` pair with Trace-level
  `Trace.receptive_fields()` and `Trace.projective_fields()` tables. A field view's
  `.at(unit)` answers for one unit; `.of(units, within=None, method="geometric",
  probe_budget=256)` answers for many (the units `tl.units` takes: a bool mask over the
  owner's full output or full-shape coordinate tuples) and returns a trace-bound
  `tl.Selection`, one region per `within=` site (default: model outputs for projective,
  model inputs for receptive), usable in `trace.do`, `trace.subgraph` and `& | - ~`
  *(unstable — no deprecation shim owed)*.

**Encoding channel** *(unstable — no deprecation shim owed)*
: A declarative value source (record field name, scalar builtin, or callable `node -> value`)
  mapped to a visual channel on `draw()`. v1 ships the color channel; size and rank channels
  follow. Channels are presentation-only: they never change the Trace or the collapse plan, and
  they require the Graphviz dot layout (`layout="auto"` forces dot when a channel is active;
  explicit `layout="rank"` refuses with `encoding_requires_dot_layout`).

**color_by** *(unstable — no deprecation shim owed; keyword-only)*
: `draw(color_by=...)` fills eligible operation nodes from a colorblind-safe sequential ramp,
  normalized linear min-max over the visible nodes (the legend states the transform). Missing,
  non-finite, or rolled-ambiguous values leave nodes unencoded with a legend note. On rolled
  multi-pass layers, field sources resolve through a name-keyed rolled-aggregate allowlist:
  per-pass-varying and first-pass-only sources are never painted as if uniform, exact cross-pass
  totals encode with a mandatory aggregation legend line, and unclassified sources refuse
  (`encoding_source_invalid`). Wrong-typed values refuse (`encoding_value_invalid`); a raising
  callable refuses with the original exception chained (`encoding_callable_error`). On a
  collapsed draw, module boxes and segments take the exact sum of their member ops for additive
  sources (`"time"`, `"bytes"`, `"flops"`, `"device_time"`, additive fields) with an aggregation
  legend line; other sources leave them unencoded with a legend note.

**size_by** *(unstable — no deprecation shim owed; keyword-only)*
: `draw(size_by=...)` sizes eligible operation nodes from a scalar field, the `color_by`
  builtin tokens (`"time"`, `"bytes"`, `"flops"`, ... with the same meaning and units), the
  closed `"dims"` shape token (numel of the non-batch output shape), or a callable
  `node -> scalar`. Collapsed boxes and segments size from the same exact member-op sums as
  `color_by`. Sizes are label-aware: the median label box among encoded nodes (or a node's
  own label box when larger) is the smallest-value size and the largest value gets 4x that area. Emitted sizes are width/height
  MINIMUMS under `fixedsize=false`: a label can never be truncated by an encoding and fonts never
  scale. Strictly opt-in — plain `draw()` keeps
  uniform boxes. On rolled multi-pass layers size REFUSES where color degrades
  (`size_by_rolled_varying`): a size source that cannot be certified single-valued
  (marker-varying, first-pass projection, varying shape under `"dims"`) has no honest "n/a"
  rendering. Exact cross-pass totals (`total_*`) encode with a mandatory aggregation legend line;
  callables bypass the rolled table with a legend disclosure.

**scale (size channel)** *(unstable — no deprecation shim owed; keyword-only)*
: The size-channel scale transform: `"sqrt"` (default; compresses dynamic range) or `"linear"`
  (the literal area motif). Log is rejected by design (flattens 512-vs-4096). Supplied without
  `size_by` it refuses (`scale_requires_size_by`); an unknown token refuses
  (`encoding_scale_invalid`). Every legend drawn states the active scale.

**stack_by** *(unstable — no deprecation shim owed; keyword-only)*
: `draw(stack_by=...)` pins nodes sharing an annotation value to one Graphviz rank (the classic
  unrolled-RNN timestep diagram; strictly opt-in). `True`/`"auto"` derives the annotation
  (`pass_number` on multi-pass ops only) under the LOCKSTEP LICENSE: granted iff the pass_number
  sequence over all multi-pass ops in raw execution order is globally non-decreasing — then
  "same column = same execution window" is exactly what the figure claims. Non-monotone traces
  refuse (`stack_by_auto_underivable`); an explicit field/callable bypasses the license with the
  caption disclosing what was used; rolled graphs refuse (`stack_by_requires_unrolled`).
  Rank groups ride `RenderIR.stack_rank_groups` and emit as `rank=same` subgraphs under
  `newrank=true`; the sibling-ordering post-pass no-ops while stacking is active.

**show_redundant_args** *(unstable — no deprecation shim owed; keyword-only)*
: Checked suppression of redundant constructor-arg label rows is DEFAULT-ON: `draw()` omits a
  module constructor arg exactly when the check licenses it — the arg value provably equals the
  captured shape dimension it claims to duplicate on THIS trace (a closed torch nn module-family
  candidate table; kernel_size/stride/padding/groups/num_embeddings/num_heads are never
  candidates). A mismatch or unavailable shape keeps the arg VISIBLE — the rule can only reveal
  more, never hide a discrepancy. Rolled varying aggregates keep args visible while their
  unrolled per-pass nodes suppress (deliberate divergence); detached records render all args.
  `draw(show_redundant_args=True)` shows every captured arg. Reference:
  `docs/reference/encoding.md`.

**show_legend tri-state**
: `show_legend` accepts `None` (default, AUTO: no legend unless an encoding channel is active,
  then a channel-only disclosure legend; a `module=` focused draw also gets the key and a reading
  key), `True` (the key: one compact row per drawn mark, each drawn with its real shape and fill --
  input, output, trainable and frozen weights, module frame, module run as one op, collapsed
  module, hidden buffers, buffer, true/false test, mutated parameter or intervention only when the
  picture has one -- plus
  channel rows when active), and `False` (no legend, honored even with channels active — the
  encoding is then undisclosed). On `Trace.draw` the key sits in the top-left block under the
  header and above the graph, so a top crop keeps both. The `None` value is *(unstable — no deprecation shim owed)* pending ratification.

**show_containers / container_max_inline (Trace.draw kwargs)**
: `show_containers` accepts `False` (default: the plain graph, container paths on the node
  labels), `"labels"` (the key or index on each edge into a container leaf), `"collapsed"`
  (`"labels"` plus: an unrolled container with more than `container_max_inline` leaves of
  identical shape folds into one dashed summary box) or `"nodes"` (`"collapsed"` plus one dashed
  record node per container, grey keyed source and sink edges, dashed member-of ties for mid-graph
  containers, and a dotted box around an op-returned tuple whose leaves share one module). Any
  string setting forces the `dot` engine. `container_max_inline` (default 12) is read only under
  `"collapsed"` and `"nodes"`. The former `"auto"` (drew `"collapsed"`) and `"cluster"` (drew
  `"labels"` plus the tuple box) spellings refuse `visualization_show_containers_invalid` with the
  replacement named. Guide: `docs/containers.md`.

## Backend-neutral identity

**Backend**
: `Trace.provenance.backend`, such as `torch`, `jax`, `mlx`, `tinygrad`, `paddle`, or `tf`.

**Module identity mode**
: `Trace.options.module_identity_mode`, describing how the backend identifies module-like ownership.

**Backend address / resolver status**
: Portable origin and resolution fields for backend records. They travel with `param_source`,
  `dtype_ref`, and `device_ref`; none should be inferred from a torch-only object when a
  backend-neutral field exists.

## Site selectors

**Label selectors**
: `tl.label` (exact final label), `tl.label_contains` (label substring), and `tl.label_matches` (label
  regex pattern) select ops by their public labels. `tl.where` filters a table with a
  predicate callable.

**Pass tokens (`label:N`, `label:last`, `label:all`)**
: Passes are 1-based: `relu_1_1:2` is the second pass of a recurrent layer, and there are no
  negative pass indices. `relu_1_1:last` names the final pass and `relu_1_1:all` every pass,
  wherever a label is accepted: `trace["relu_1_1:last"]` and `trace.ops["relu_1_1:last"]`
  return that Op, `trace["relu_1_1:all"]` and `trace[tl.label("relu_1_1:all")]` return the
  Selection of every pass (an Op collection refuses `:all` with a `KeyError` naming those
  doors), and `trace.layers["relu_1_1:last"]` is the aggregate Layer like any pass spelling.
  A bare multi-pass label through `trace[...]` refuses (`multipass_bare_label_ambiguous`, the
  `tl.label` refusal) naming these spellings and `trace.layers[label]`; `trace[...]` never
  returns a Layer.

**tl.all_sites** — *unstable — no deprecation shim owed*
: `tl.all_sites()` is the universal selector: `trace[tl.all_sites()]` selects every op of the
  trace as a whole site, the same result as `trace[tl.label_matches(".*")]` (it stores that
  kind and pattern, so it matches, binds and persists the same way; a saved spec reloads as
  `tl.label_matches(".*")`). It composes with `|`, `&`, `-` and `~` like any selector.
  `tl.all_sites() - region` is the complement of a region in the whole trace; the graph
  producers and `trace.subgraph` read it as the sites outside `region`, because a site whose
  element mask is empty seeds nothing. Resolution refuses typed (`non_tensor_site` /
  `no_index_space`) when some op has no output index space, rather than dropping it.

**Structural selectors**
: `tl.func` / `tl.module` / `tl.in_module(address)` select by callable or module context
  (`in_module` only builds a selector; whether one record lies in a module is read on the
  record, `layer.module == address` or `address in layer.modules`);
  a module registered under several names (`self.dec = self.enc`, all recorded in
  `Module.all_addresses`) is selected by its canonical `named_modules()` name, which fires at
  every call site of it (pass labels such as `"enc:2"` count calls of that one object); an
  alias spelling (`"dec"`, `"dec:2"`) refuses with `bind_static_anchor_unresolved` and one
  message at every door (capture, record, post hoc, `spec.bind`), since no hook can tell which
  name a forward called the object through;
  `tl.head` selects one attention head; `tl.facet` selects a semantic facet view; an unscoped
  `tl.head(i)` writes head `i` of every matching facet in every module and says how many sites
  it writes (a `TorchLensWarning` coded `facet_selector_fanout`, with `fields["site_count"]`;
  36 on a 12-layer GPT-2: q, k and v in each layer); `.in_module(address)` scopes it to one
  module, and so does composition: `tl.module(address) & tl.head(i)` (either order) is that
  same scoped facet selector, accepted wherever a selector is (`trace[sel]`, `tl.when`, `do`);
  a facet already scoped to another module refuses
  `selector_module_facet_scope_conflict`;
  `tl.func_transform` selects `torch.func` and `torch.autograd` transform boundary ops
  (including `autogradgrad`, an in-forward `torch.autograd.grad` call, with
  `transform_kind="autograd.grad"`; `autogradbackward`, an in-forward `loss.backward()` or
  `torch.autograd.backward` call, with `transform_kind="autograd.backward"`, one op per leaf
  gradient the pass writes, parented on the roots and, when the leaf already held a grad, on
  that previous grad as its in-place receiver, so a later `param.grad` read has a producer;
  `autogradfunction`, a custom `autograd.Function`
  whose forward left torch through a numpy or dlpack round trip or a known foreign kernel
  (FlashAttention 4's `flash_attn.cute` Functions, exit `flash_attn.cute`), with
  `transform_kind="autograd.Function"`, `foreigndata`, a host constructor (`from_numpy`,
  `from_dlpack`, `frombuffer`, or a tensorless `tensor` / `as_tensor` / `asarray`) that brings
  data back into torch after a plain-code `numpy` / `__array__` / `__dlpack__` / `to_dlpack` /
  `tolist` export outside any Function, with `transform_kind="foreign_data"` and
  `transform_config["resolution"]` of `view`, `copy`, `constant` or `unresolved`, and `jitscript`, one call to a
  `torch.jit.ScriptFunction` from an eager forward, with `transform_kind="jit.script"`);
  `tl.model_output` selects model outputs, and
  `tl.model_input` selects a nested model-input path.

**Temporal composition**
: `tl.followed_by` (retroactive successor) and `tl.preceded_by` (lookback predecessor)
  compose with boolean operators to form temporal predicates over the op stream.

**Backward selectors**
: `tl.grad_fn` (backward grad_fn; `label=` matches one exact grad_fn label),
  `tl.grad_input` / `tl.grad_output` (backward event tensors), `tl.in_backward_pass`
  (one backward pass number), and `tl.without_op` (grad_fns without a paired forward op;
  the former `intervening` alias is removed). These are NODE selectors: they address an
  autograd node and its `grad_input` tuple, resolved against `trace.autograd.grad_fns` after
  `backward()`. The gradient of a forward site's output is addressed with `tl.grad(site)`
  instead (see **tl.grad**).

**tl.grad(site)** *(spelling documented-unstable)*
: Builds a `GradSelector` (frozen, hashable, selector kind `"grad"`; also
  `torchlens._selection.selectors.grad`) that names the gradient of the loss with respect to the
  site's OUTPUT, observed at the TorchLens boundary before it enters the op's backward function:
  the same tensor `trace[site].grad` stores. It is an address and never calls autograd.
  `repr(tl.grad("relu_3"))` is `tl.grad('relu_3')`. It accepts a label (pass-qualified
  `"relu_3:2"` on multi-pass layers), any activation selector (`tl.label`, `tl.func`,
  `tl.module`, `tl.in_module`, `tl.label_contains`, `tl.label_matches`, `tl.where`,
  `tl.model_output`, `tl.model_input`, `tl.followed_by`, `tl.preceded_by`, `tl.at_step`, ... and
  `&`/`|`/`~` compositions of these), an `Op` or `Layer` record, or a `find_sites()` table. An op with
  several outputs has one gradient per output (`tuple_index` names the slot). It refuses, with
  `SelectorCompositionError` carrying `code` and `remedy`: a nested `tl.grad`
  (`grad_selector_nested`), a node selector or a pass qualifier inside
  (`grad_selector_inner_not_forward`: write `tl.grad(s) & tl.in_backward_pass(2)`), a facet or
  head selector (`grad_facet_unsupported`), a `Selection` (`grad_selection_unsupported`), and a
  list or mapping (`grad_selector_inner_invalid`). No site door takes a pass keyword; the site
  decides. Read side is unchanged in spelling: `trace[site].grad`, `.grads` and `.has_grad`
  (`trace[tl.grad(s)]` is not an indexing form).

**Activation, gradient and node universes**
: Every site belongs to one universe. A bare selector or label is an ACTIVATION site (forward out
  hook, payload the activation); `tl.grad(inner)` is a GRADIENT site (the output gradient of each
  forward op `inner` matches); the backward selectors are NODE sites (a grad_fn post-hook per
  `grad_input` slot). A bare label never reaches the backward graph unless wrapped in `tl.grad`.
  `tl.grad(a) | tl.grad(b)` is one gradient branch, `tl.grad(s) | s` is two branches (activation
  and gradient, each resolved, mounted and recorded separately; union keeps both because
  identity is typed, never the label alone), `~tl.grad(s)` normalises to `tl.grad(~s)`, and
  `tl.grad(s) & tl.in_backward_pass(n)` restricts the gradient branch to pass `n`. `&` across
  universes refuses `selector_cross_universe_and` (`tl.grad(s) & s`, `s & tl.grad_fn(...)`,
  `tl.grad(s) & tl.grad_fn(...)`), `~` over a union that spans universes refuses
  `selector_cross_universe_not` (negate inside each branch: `tl.grad(~a) | ~a`), a pass
  qualifier on a bare activation selector refuses `selector_signal_mismatch`, and
  `tl.module(a) & tl.head(i).in_module(b)` with `a != b` refuses
  `selector_module_facet_scope_conflict`. A gradient site resolves its inner selector exactly as the
  activation door does (zero matches refuse); a matched op the loss never reaches is a
  `no_fire`, never a resolution error and never a zero tensor. `Trace.find_sites(tl.grad(s))`
  returns the matched forward sites with `signal="backward"`; its `.to_dataframe()` table
  carries that value in a `signal` column (`"forward"` for activation sites, `"backward"` for node sites).

**signal (record field)** *(spelling documented-unstable)*
: The one field that says which pass a record or event belongs to: `"forward"` or `"backward"`
  (plan and span summaries may say `"both"`). It is spelled `signal` on `HookContext`,
  `FireRecord`, `TapRecord`, `Edit`, `NarrationEvent` and `Rule` (derived), on the audit payload,
  the coupling record and the span record. Sites carry no `signal=` keyword; the only door that
  does is `tl.observers.span(name, signal="both")`. `HookContext` also carries `ctx.state` (the
  per-(site, hook) scratch dict) and `ctx.run_state` (the run-scoped dict); a callable attached
  to `tl.grad(s) | s` runs in both passes and reads `ctx.signal`.

**`Accessor.count_by(*keys)`**
: Every accessor (`trace.layers`, `trace.ops`, `trace.modules`, `trace.params`, ...) counts
  its records by one or more record fields: one key gives `{value: count}`, two or more give
  `{(v1, v2, ...): count}`, in first-seen order; `len(accessor)` is the total. Layers and ops
  count by `"module"` (containing module address) and `"kind"` (the op family token,
  `conv2d`, `relu`); other accessors count by any string or int field their records carry
  (`trace.modules.count_by("class_name")`, `trace.params.count_by("dtype")`). No key refuses
  `accessor_count_by_key_required`; an unknown key refuses `accessor_count_by_key_unknown`
  naming the countable fields. Filtering (the labels for one value) is a selector lookup:
  `trace[tl.in_module(address)].labels`.

## Intervention helpers

**Value helpers**
: `tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`, `tl.add_noise` (Gaussian noise),
  `tl.mean_ablate` (fill with a mean; its own entry below), `torchlens.intervention.scramble_elements`
  (elementwise iid scramble from a flattened source -- the honest rename of
  the removed `resample_ablate` spelling, which a saved spec may not name either: it refuses
  `intervention_helper_renamed`; for coherent donor patching use
  `tl.patch_from`), `tl.replace_with` (fixed value), `tl.steer` (add a scaled steering direction), `tl.project_onto` /
  `tl.project_out` (keep or remove the component along a direction), and
  `tl.splice_module` (call a module as a black-box forward splice), and `tl.clip_norm` (norm clip).
  Availability outside
  the torch backend is narrower; see the per-backend rosters in the backends guide.
  The value helpers take the keyword-only `allow_shape_change=` (default `False`), which
  allows the edit to change the site's output shape; by default the edit must preserve it. `tl.steer`
  also takes `strength=` (its second argument, default `1.0`), the multiplier applied to the
  steering direction.
  Tensor-carrying helpers (`tl.steer`, `tl.mean_ablate`, `tl.project_onto`, `tl.project_out`)
  keep the caller's tensor by reference. Each staged entry records a
  full-content digest of those tensors when it is staged; a rerun and `save_intervention`
  refuse `helper_tensor_changed_since_capture` once the tensor changed (in place or through
  `.data`). Re-stage the edit to use the new value. A bound executor (`spec.bind(model)`) reads
  the helper's tensor at each call; in-place changes apply to the next call, and
  `BindReport.helper_tensor_versions` (on `.last_report`) records each tensor's version counter,
  keyed by the rule id the call started with (a rule id follows its tensor's content).

**`tl.mean_ablate(source=None, *, dim, allow_shape_change=False)`**
: The one mean door. `dim` is required: `None` averages every axis of the source to one
  scalar; an int or tuple of ints averages those axes (`keepdim`, broadcast back to the site).
  `source` is what is averaged: the site's own value (the default; it reads the whole batch, so
  the spec derives `is_batch_independent=False`, the append gate refuses and a chunked capture
  refuses `chunked_replay_batch_coupling`), a tensor or Op-like object (averaged instead of the
  site; `is_batch_independent=True`), or a `tl.intervention.Population` (its members are averaged
  elementwise first; `dim=None` fills with that member-shaped mean, axes reduce it further; the
  spec is `opaque_audit` and the fire record says `stochastic=False`). A mean is a reduction,
  never a draw: no seed. A semantic axis name (`"batch"`, `"position"`, `"feature"`) refuses
  `axis_semantics_unknown` until axis-role evidence exists; any other `dim` refuses
  `intervention_over_invalid` at construction. The removed `mean_fill(dim=...)` is
  `mean_ablate(dim=...)`, the removed `mean_from(population)` is `mean_ablate(population,
  dim=None)`, and a saved spec naming `mean_fill` refuses `intervention_helper_renamed`.

**Edits on gradient sites**
: A tensor edit is a function of one tensor and the site decides which tensor, so the neutral
  edits apply to activation and gradient sites alike: `tl.scale`, `tl.zero_ablate`, `tl.clamp`,
  `tl.add_noise`, `torchlens.intervention.scramble_elements`, `tl.steer`, `tl.project_onto`,
  `tl.project_out`, `tl.replace_with`, `swap_with` with a fixed tensor donor, `tl.clip_norm`,
  and any callable `fn(tensor, *, ctx)`; their `Edit.signal` is `None`. Edits that read recorded
  forward state (`tl.patch_from`, `tl.mean_ablate`, `tl.splice_module`, `swap_with` with a
  site or trace donor; `Edit.signal == "forward"`) refuse on a gradient site with
  `edit_signal_mismatch` at attach time, before any mutation. There are no gradient-only helpers:
  `trace.do(tl.grad("relu_3"), tl.zero_ablate(), model=model, inputs=x)` zeroes a gradient.

**tl.clip_norm(max_norm, norm_type=2.0)** *(spelling documented-unstable)*
: Neutral, portable edit that rescales its tensor so its `norm_type` norm is at most `max_norm`,
  per tensor. On `tl.grad(site)` it is gradient-norm clipping at that site.

**Armed gradient edits and Trace.armed_edits** *(spellings documented-unstable)*
: `trace.do(tl.grad(s), edit, model=model, inputs=x)` reruns the forward with the activation half of
  the plan applied live and the gradient half carried into the result's recipe as sticky backward
  hooks of the returned execution only (the source trace is not armed). The rerun is made
  graph-connected whatever the source's `backward_ready` was and says so in
  `result.last_run["backward_ready_forced"]`. Its `intervention_event_v2` envelope has status
  `"armed"` (beside `"fired"`, `"no_fire"`, `"error"`), `fire_count=0` and
  `extra={"armed_edits": n, "awaiting": "backward"}` (a mixed plan keeps the activation half's
  `fired`/`no_fire` status and adds the same keys). `Trace.armed_edits` is the read-only tuple of
  the gradient rules in the trace's recipe, and the trace summary says "N gradient edits armed;
  call trace.backward(loss)" until they fire. Each `trace.backward(loss)` then writes one settle
  envelope per arming transaction (`door="backward"`, `lane="live_hook"`, `parent_event_id` and
  `extra["arming_event_id"]` the arming row, `backward_pass_number`, status `fired` or `no_fire`
  with `no_fire_reason` `"gradient_not_reached"` or `"pass_filtered"`); a second backward never
  rewrites the first pass's row. `loss=` on `Trace.do` (and `tl.sweep`) is a callable of the
  returned trace returning a scalar with a `grad_fn`; the door runs `result.backward(loss(result))`
  in the same call. A non-callable refuses `do_loss_not_callable`, a plan with no gradient site
  refuses `do_loss_without_gradient_site`, and a callable that raises writes an `error` envelope,
  detaches the hooks this call attached, and re-raises. `engine="set_only"` arms the gradient rules
  on the trace itself for its own `trace.backward(loss)`. Staging the recipe
  `do(tl.when(tl.grad(s), edit), intervention=InterventionOptions(engine="set_only"))`
  registers the sticky rule without a rerun; when the trace holds no live backward graph its
  staging history row carries `armed_for="next_rerun"` (the rule waits for the next rerun's
  backward). A saved trace keeps the armed rules and records their
  count in `annotations["armed_gradient_edits"]`.

**Replay verbs**
: `Trace.do` applies a one-shot intervention to a captured log; `Trace.push` pushes an edit
  downstream through the recorded graph (DAG replay) and `Trace.push_from` pushes from a
  pre-mutated site; `Trace.run` performs a full-forward run with the log's active
  intervention spec; `tl.sweep` captures one intervened trace per swept replacement
  value. (The former `replay`/`replay_from`/`rerun` aliases are removed.) Replay re-executes
  forward ops only, so a gradient site on `do` without `model=`/`inputs=`, on `engine="replay"`, or in
  `push` refuses `replay_gradient_edit_unsupported` before any mutation ("Pass model= and inputs= to
  rerun with the edit armed, then call result.backward(loss) (or pass loss=), or stage it with
  engine='set_only' and call trace.backward(loss) on this trace"); differentiable replay of
  gradient edits is out of scope this release. The replay engine operates on pass-qualified op labels
  (`label:pass`): cone traversal, the replay overlay, hook targets, and commits are
  keyed per pass, so edits on multi-pass (recurrence-grouped) layers touch exactly the
  addressed pass and recompute every downstream pass; a bare label naming a multi-pass
  layer refuses `multipass_bare_label_ambiguous` (single-pass bare labels stay
  accepted). Replay disclosures (`last_run` origins/cone, `_replay_frontier` keys) spell
  multi-pass ops pass-qualified and keep bare labels for single-pass layers.
  Engines treat the trace's staged spec as read-only: a rerun (plain, append, or
  chunked), a failed or interrupted rerun, and a fork leave it exactly as staged.
  A capture-time `intervene=` predicate that is not lowered to module hooks stages
  one entry per fired op on its final label; `Trace.run` re-arms the trace's retained
  predicate through the capture door instead, and refuses
  `rerun_predicate_restage_mismatch` (trace unchanged) when the predicate fires at
  different ops than the staged entries name, or at the same ops with changed rule content
  (a staged steer edited since capture). A staged edit addressed by a finalized exact label
  (`do(label, value)` with the `set_only` engine, `do(tl.label(...), edit)`) is translated before the forward to
  the raw capture labels of the ops it names, the translation a mixed gradient plan's
  activation half already uses, so it fires on the rerun; each pass of a multi-pass op keeps
  its own raw label, so a pass-qualified label edits that pass alone (JMT, Sun 1 Nov 2026).
  A substring or regex selector on a finalized label still refuses through the live matcher
  (`LiveModeLabelError`, with its authoring remedy). A staged edit whose selector matches sites on the trace but fires at none
  during a rerun of the same graph on the same inputs refuses with
  `rerun_selector_unmatchable_live` (trace unchanged), and so does, on any inputs, a
  `tl.where` predicate that matches sites on the rerun's own trace but fired at none during
  its forward (a predicate on `p.layer_label`, which exists only after postprocess). A
  predicate that matches nothing on the new inputs warns `rerun_zero_fire`. The rerun divergence check (`ControlFlowDivergenceWarning`, or
  `ControlFlowDivergenceError` with `strict=True`) compares raw op order, edges and shapes with
  each value-only edit node folded into the op it replaces, so a staged edit is never
  reported as control-flow divergence while a changed op, edge, shape or batch size still is.

**Guarded fast rerun** *(spellings documented-unstable)*
: The engine a steered rerun takes when the trace's staged intervention spec targets only
  plain module selectors (`tl.module(...)` / module-address hooks, no `set()` value
  replacements): a native forward with the staged hooks fired at each targeted module's exit
  and forward hooks refreshing the saved sites, so the rerun costs close to a plain hooked
  forward instead of a full capture. Both doors use it: the legacy `trace.run(model, x)` on a
  trace with a staged intervention tries it first (a trace with nothing staged keeps the capture
  rerun) and falls back to the capture engine on any typed refusal, and
  `trace.run(inputs=..., fast=True)` on a live or loaded-activation trace applies the staged
  spec through it (so the `run_staged_spec_unapplied` gate does not fire there). `last_run`
  discloses the outcome: `engine` is `guarded_fast` when it ran and `rerun` after a fallback,
  where `fast_refused` names the refusing guard as `<code>:<stage>` (for example
  `run_capability_unavailable:fast_rerun_target_unsupported` for a value replacement or a
  non-module target, `run_capability_unavailable:fast_rerun_graph_unsteered` when a staged
  entry has never fired in the trace's recorded graph, as after a sticky hook staged by `do` with
  `engine="set_only"` on a plain capture, or when the graph records a fire no staged entry
  accounts for, as after `clear_edits(site)` on a steered capture, so the capture engine reruns
  once to record the current plan, or
  `<divergence code>:fast_live_call_fingerprint` for a structural change); `hooks_fired` / `hooks_unfired` count the staged plans that fired, and a plan that
  never fired settles by `on_zero_fire` (default: warns `rerun_zero_fire`). Through
  `run(inputs=..., fast=True)` a run that applies staged edits reports `unverifiable`, never
  `verified` (diagnostic `user_intervention_unsupported`, stage `hooked_run_unverified`):
  it is an edited forward, and no hook-aware check exists yet. On a live trace the
  `fast=True` door returns a new result Trace carrying that `last_run` and leaves the source
  Trace untouched (JMT, Sun 1 Nov 2026); the legacy rerun door (`run(model, inputs)` with
  `engine="rerun"`) still refreshes the source in place. Output leaves (the model's and each saved module's) are paired
  with the captured output ops by container path through capture's own output walker, so a
  stock Hugging Face forward returning a `DynamicCache` of per-layer key/value tensors reruns
  fast; a changed leaf count or an unmatched path still refuses
  (`output_structure_mismatch:fast_live_model_output_structure`). Input sizes may differ from the capture (generation)
  when rank, dtype, device and input tree match and the capture sealed a call fingerprint
  (`trace._raw_call_fingerprint`, an ordered rolling hash of every wrapped torch call and
  module entry, compared against `last_run["call_fingerprint"]`); such a run sets
  `last_run["shape_varied"]` and resets `shape`, `transformed_out_shape`, `activation_memory`
  and `transformed_activation_memory` to `None` on every op the run did not refresh, so no
  capture-time number is presented as current. The save scope never widens: unsaved ops stay
  unsaved, `transformed_output` follows the capture's output-transform rule, and the stored spec is
  unchanged. A real structural divergence still refuses (`PathDivergenceError` from the
  explicit door; the legacy door falls back and records why). A staged helper tensor changed
  since it was staged refuses `helper_tensor_changed_since_capture` on both doors, as the
  capture rerun does, and the explicit door's cached session rebuilds whenever the staged spec
  changes, so it always applies the spec staged at call time.

## Extraction, observers, and admin

**Extraction helpers**
: `tl.pluck` returns the saved out for one layer, `tl.extract` for many layers
  (detached by default; `detach_saved_activations=False` returns graph-connected outs), and
  `tl.extract_dataset` extracts outs from an iterable dataset in batches. (The former
  `peek` and `batched_extract` aliases are removed.) Disk-mode `extract_dataset`
  (``path=``) writes a SELF-DESCRIBING artifact: atomic ``batch_XXXXX.pt`` shards
  plus a ``manifest.json`` recording the run signature, per-site identity (layer label
  and structural site key where derivable), stimulus ordering/provenance
  (``stimulus_ids=`` optional), axis semantics, dtypes, devices, and the TorchLens
  version. ``resume=True`` (DOCUMENTED-UNSTABLE, like ``stimulus_ids=`` and the loader)
  continues an interrupted disk run from its last completed shard after a signature
  check; `torchlens.load(path)` reads the artifact back with
  its metadata as a `LoadedExtraction`.

**Module and stage names** *(rule documented-unstable)*
: A module name resolves to that module's output at every selector door: `tl.extract`,
  `tl.pluck`, `trace[...]`, `tl.trace(save=[...])` and `trace.run(until=...)`. Three spellings
  name a module: its address (`"fc"`, `"layer1.0"`); a name relative to a wrapped inner model
  (timm `FeatureGetterNet` reports `"blocks.9"` for the module at `"model.blocks.9"`); and a
  dotted name whose module was registered with dots flattened to underscores (timm
  `FeatureListNet` registers stage `"stages.0"` as `"stages_0"`). A module whose output is also
  a model output resolves to its own output op, not the model-output alias. A relative name
  that ends several module addresses refuses `ambiguous_op_lookup` and lists them; a
  `":<call>"` suffix selects one call of a reused module.
**Intervened extraction** *(spellings documented-unstable)*
: `tl.extract_dataset(..., intervene=spec)` takes a `tl.when(selector, helper)` spec and
  captures every batch with it armed, so the stored rows are the intervened activations.
  Batch zero freezes the structural sites the spec fired at; a later batch that fires
  nowhere refuses `extraction_intervention_no_fire` and one that fires elsewhere refuses
  `extraction_intervention_site_drift`. Module (`tl.module`, `tl.in_module`) and op-type
  (`tl.func`) addressed specs run; finer address classes refuse
  `harvest_address_class_ungated`, and every helper must declare `is_batch_independent=True`.
  Disk mode records the counterfactual in the manifest's `tl_counterfactual_harvest_v1`
  block (policy, engine `live_hook`, address class, site keys, fire receipt digest),
  ledgers each shard's fire receipt, and puts the intervention identity in the resume
  signature; a clean run declares `policy="none"`.

**Token store** *(spellings documented-unstable)*
: `tl.extract_dataset(..., store_input_ids=True)` stores each batch's token ids as the
  ordinary `input_ids` key, trimmed by the batch mask under `ragged="trim"` like the
  token-axis activations. `ExtractionReader.token_batch(shard_index, keys=None)` and
  `ExtractionReader.token_rows(keys=None, *, max_bytes=None)` read stored keys one row per
  token as a `torchlens.dataset_extraction.TokenRows` (`values` per key, `stimulus_row`,
  `position`, `num_tokens`); padding is never a token row.
  `ExtractionReader.shuffled_batches(batch_size, *, seed=0,
  keys=None, buffer_shards=1, tokens=False)` shuffles across `buffer_shards` shards at
  once (the cross-shard mixing buffer) and yields token rows when `tokens=True`.

**Observers**
: `tl.observers.tap` creates a tap observer for a site (`tap(tl.grad(s))` records gradients with
  `signal="backward"` and `grad_kind="grad_output"`; `tap(tl.grad(s) | s)` records both);
  `tl.observers.span(name, signal="both")` records a named observer span around captures or hook
  execution, `signal=` (`"forward"`, `"backward"` or `"both"`) scoping which records carry the span
  name (the former `record_span` alias is removed);
  `tl.observers.register_tensor_connection` registers a manual parent-child tensor edge during
  capture.

**Validation entry**
: `tl.validate(model, x, scope=...)` validates a model/input pair for a requested scope
  (for example `"saved"` or `"receptive_field"`), capturing what it needs itself. Like
  `tl.trace`, it seeds the global RNG streams (Python `random`, NumPy, torch CPU and CUDA) for
  its own runs and restores the caller's streams on return and on error; the lower-level
  `torchlens.validation.validate_forward_pass` and `validate_backward_pass` make the same
  promise, as does `Trace.validate_forward_pass`, which replays each op under its recorded
  RNG state.
  Under `scope="backward"`, when torch reports a backward through a graph an earlier backward
  freed and a module attribute holds a tensor with autograd history, the model is refused with
  `ValidationError` code `backward_validation_carried_graph`, naming those attributes; without
  that evidence torch's own error propagates unchanged.
  Keyword-only `output_device` and `save_budget` set the validator capture's save options,
  with the same values and defaults as `CaptureOptions.output_device` (`"same"`) and
  `CaptureOptions.save_budget` (`"auto"`): `output_device="cpu"` lets a GPU model validate
  with its saved activations in host memory (replay still runs each op on its original
  device), and an exceeded budget raises `SaveBudgetExceededError` as the capture does.
  Under `output_device="cpu"` saved function arguments (the replay inputs) other than
  parameters are kept in host memory too and moved back to each op's device for its
  check, so the GPU keeps only the parameters and one forward's transients. On every
  `output_device`, parameter arguments the validator capture keeps as references to its own
  copy of the model are not charged to its `save_budget`. Both keywords
  apply to the `"forward"`, `"saved"`, and `"intervention"` scopes; a non-default value on
  `"backward"` or `"receptive_field"` raises `TypeError`. The lower-level
  `torchlens.validation.validate_forward_pass` accepts the same two keywords.

**Session admin**
: `tl.release_model` evicts TorchLens's module metadata and preparation bookkeeping for a
  traced model and normalizes plain attributes holding torch functions from the other wrap
  epoch, so whole-model pickle / `torch.save` survive later wrap-state changes (a capture
  itself leaves no TorchLens callable on the model: submodule `forward` wrappers are
  session-scoped); `tl.clear_cache`
  empties the capture cache; `tl.io.list_traces` / `tl.io.reset_naming_counter` manage log
  bookkeeping.

## Persistence, containers, and namespaces

**Save / load**
: `Trace.save` persists a `Trace` into a portable `.tlspec` directory bundle at a chosen
  level; `tl.load` loads a `.tlspec` object with eager tensor materialization.
  3.0 format: the maintainer contract `docs/maintainers/saved_format_3.md` (not on the site).
  Both doors check the final path component only (`path.is_symlink()` on the bundle path and,
  at load, on its manifest, metadata and blobs entries): a symlink there refuses
  (`load_path_symlink_rejected` at load), a symlinked parent directory passes on both sides,
  and save and load use the same rule; `storage=tl.to_disk(...)` and `save_intervention`
  apply it to their own targets (documented as is, JMT, Sun 1 Nov 2026).
  The staged intervention spec is session-only: `Trace.save` keeps the intervened
  values and their per-op provenance, and every legacy rerun door of a loaded intervened trace
  (`run(model, x)` plain, append or chunked, a fork's `run`, `do(engine="rerun")`, and an edit
  staged after loading) refuses `run_intervention_spec_not_persisted` rather than rerun without
  the recorded intervention (the recipe travels through `save_intervention`); the replay and
  `run(inputs=...)` doors cannot run an analysis artifact at all.
  A rerun replays the capture's recorded `save=` request on its own graph, so it saves the same
  sites a fresh capture would even after a staged edit inserts ops. A trace restored from pickle
  carries no recorded request and re-saves by the capture's raw op indices, which is exact only
  when the rerun graph matches op for op; otherwise the rerun refuses `rerun_resave_ops_shifted`
  (trace unchanged).
  `tl.io.PayloadLoadHints` carries backend-specific payload materialization hints
  (`tl.io.JaxPayloadLoadHint` is the JAX-specific form).

**Options**
: `tl.options` groups the public option dataclasses: `CaptureOptions`,
  `VisualizationOptions`, `ReplayOptions`, `InterventionOptions`, `StorageOptions`,
  `RecordingOptions`. The doors take only the model and the inputs by position;
  every capture knob rides `capture=CaptureOptions(...)` (`activation_transform`,
  `grad_transform`, `save_raw_activations`, `save_raw_grads`, `save_mode`,
  `reconstruction_ready`, `lookback`, `lookback_payload_policy`, `grouping`,
  `jax_static_argnums`, `grad_options`, `chunk_size` included), and `tl.record`'s
  recorder knobs ride `recording=RecordingOptions(...)` (`default_op`,
  `default_module`, `history_size`, `include_source_events`,
  `max_predicate_failures`, `on_predicate_error`, `on_forward_error`,
  `default_grad`, `postprocess`), matching `tl.trace`'s `capture=`. The door keywords are
  `save=`, `intervene=`, `recording=` (on `tl.record`),
  `stop_after=` (a site or a predicate), `storage=`,
  `narrate=`, `episode=`, `facets=`, `chunk_paths=`, `backend=`, `input_shape=`
  and `on_zero_fire=`.

**on_nonfinite (CaptureOptions)**
: The one NaN/Inf policy, replacing `raise_on_nan` and `track_nonfinite`:
  `'ignore'` (the default; neither checks nor records, today's default
  behaviour), `'track'` (per-op finiteness verdicts, served by
  `Trace.nonfinite_ops` / `Trace.nonfinite_coverage`), `'warn'` (tracks and
  warns `capture_nonfinite_detected` once per capture when any op produced a
  non-finite output) and `'raise'` (stop at the first non-finite tensor and
  raise). Any other value refuses `on_nonfinite_invalid`.

**Structural hash**
: `tl.fingerprint` is the provisional structural-hash namespace; `tl.fingerprint.assert_unchanged` asserts
  a model still matches a pinned address-free structural hash.

**Container registration**
: `tl.types.register_container` registers a custom container type (flatten/unflatten pair) for
  capture and reconstruction. `tl.types.Container` is the computed view over a captured Python
  output container. The container surface is stable: `tl.types.Container`, `Op.container`,
  the `input_at` and `output_at` selectors, `tl.types.register_container`, `Container.reconstruct`,
  `Trace.reconstruct_output`, `show_containers` (including `"nodes"`) and the
  `capture_container_structure` option (see [Containers](../containers.md)).

**Capture-product bases**
: `tl.types.TraceLike` is the shared base for uncooked and cooked capture projections.

**Namespaces**
: `tl.recording` is the sparse predicate-recording namespace behind `tl.record`;
  `tl.facets` is the semantic facet namespace (canonical home `torchlens._semantic`);
  `tl.inputs` registers input preprocessors (`tl.inputs.register`, `tl.inputs.unregister`)
  that turn a model's raw input into its forward arguments for every later `tl.trace`;
  `tl.outputs` registers output decoders (`tl.outputs.register`, `tl.outputs.unregister`)
  that label a classifier's logits at capture time and for `decode_output(style=...)`;
  `tl.export` holds static export helpers; `tl.Bundle(...)` / `tl.Bundle` build aligned
  Trace collections, and `Bundle.draw` (below) draws a bundle's merged graph. `tl.load(path, lazy=True)` on a
  saved bundle passes `lazy` (with `map_location` and `materialize_nested`) to every member.

**Appliance door (`Trace.using`)** *(spelling documented-unstable)*
: `trace.using(name)` returns the appliance registered as `name`, bound to that trace (for
  example `trace.using("mi").head_scores(...)`; TorchLens registers `"mi"`, the mechinterp
  kit, whose methods return exactly what the matching `torchlens.mechinterp` functions return).
  Appliances hang off this one door instead of adding methods to `Trace`; TorchLens's own and
  outside packages register the same way, with
  `torchlens.appliances.register(name, factory, summary="", replace=False)`, where
  `factory` is a callable taking the trace or a lazily imported `"module:attribute"` path.
  `torchlens.appliances.names()` lists the names with their summaries. An
  unknown name refuses `appliance_unknown` and lists the registered names; re-registering a
  name without `replace=True` refuses `appliance_name_taken`; a malformed name or factory
  refuses `appliance_registration_invalid`. The word clashes with `torchlens.facets.adding(*facets)`
  (a recipe scope); the naming review settles both.
**Safetensors Trace export** *(spellings documented-unstable)*
: `tl.export.safetensors(trace, path)` writes one pickle-free `.safetensors` file: every
  tensor-valued op payload (`out`, `transformed_out`, `grad`, `transformed_grad`) as a
  tensor entry, and a JSON header document (schema `tl_trace_safetensors_v1`) holding the
  tlspec version, the writer-contract digest, the capture-honesty facts, and, for the
  `trace` record and every `op` record, each writer-contract KEEP field with an exact JSON
  encoding (the rest listed under `omitted` with their value types).
  `tl.io.read_safetensors(path)` reads it back as a read-only record (`header`,
  `trace_fields`, `op_fields`, `tensors`, `omitted`); payloads are bit-equal, complex
  included.

**Streaming statistics**
: `tl.stats` is the streaming-statistics namespace: constant-memory accumulators
  (`tl.stats.Mean`, `tl.stats.Norm`, `tl.stats.Covariance`, `tl.stats.CrossCovariance`,
  `tl.stats.Quantile`, `tl.stats.TopK`, `tl.stats.PCA`, `tl.stats.CKA` -- exact
  full-data linear CKA, never a minibatch approximation -- `tl.stats.Histogram`,
  `tl.stats.RunningStats`) built on the `tl.stats.StreamingStat` protocol, plus one
  persistence path (`tl.stats.save_fitted` / `tl.stats.load_fitted`) for fitted PCA and,
  DOCUMENTED-UNSTABLE, for any accumulator's whole state (`tl_stats_state_v1`; a reload
  keeps folding as if never interrupted; `load_fitted(path, encoder=...)` takes back an
  `encoder=` callable, which is not stored). `tl.stats.aggregate(model,
  dataloader, metrics)` streams activations or gradients through those accumulators, keyed
  by site: `metrics={"relu_3": tl.stats.Mean(), tl.grad("relu_3"): tl.stats.Mean()}` (a plain
  string key is an activation; a gradient key needs `loss_fn=`, else
  `aggregate_loss_fn_required`; result and cache keys use the key's canonical repr) across a whole
  dataloader without retaining per-batch tensors; `cache=True` (with `cache_dir=`, as
  for `tl.trace`; DOCUMENTED-UNSTABLE) keys the run by value (model contents and code,
  every batch, each accumulator's starting state and encoder, the metric keys, `loss_fn=`,
  versions) so a rerun restores the saved states and skips the loop, and refuses a run
  it cannot key (`stats_aggregate_cache_unkeyable`). Per-unit statistics
  (DOCUMENTED-UNSTABLE): the keyword-only `unit_axis=` on `tl.stats.Quantile` and
  `tl.stats.TopK` keeps one reservoir or top-k list per unit (neuron, channel or SAE
  latent); per-unit `TopK` returns a `tl.stats.UnitTopK` (`values`, `example_ids`,
  `positions`, `unit_axis`), the one top-k-over-a-dataset tool for neurons and SAE
  latents, with ties broken toward the lowest example id, then position;
  `encoder=` maps each batch (for example through an SAE encoder) before it is
  folded; `TopK.update(..., example_ids=...)` names examples outside
  `tl.stats.aggregate`. Reduce and discard (DOCUMENTED-UNSTABLE): `tl.trace(model, x,
  save="stats")` folds every op's out into a `tl.stats.RunningStats` during the forward and
  keeps no payload (one 16-slot float64 vector per op); `tl.stats.site_stats(trace)`
  returns `{op label: RunningStats}`, payload reads refuse `activation_reduced_to_stats`, and
  replay or validation on the trace refuses `stats_capture_not_replayable`. Trajectory
  statistics (DOCUMENTED-UNSTABLE): `tl.mechinterp.measure_trajectory(model, data, *,
  statistics=None, facet="resid_post", layers=None, lens=None, mask_input=False,
  verify=True)` traces each batch and folds the logit-lens trajectory's `entropy`,
  `kl_to_final`, `max_prob` and, for `(input_ids, target_ids)` batches,
  `cross_entropy` and `target_rank` into a `tl.mechinterp.TrajectoryStats`
  (`position_means[statistic]`, a float64 `[layers, positions]` tensor;
  `running_stats[statistic][layer]`; `summary(statistic)`; `examples`, the number of examples
  (batch rows) folded; `validated`, whether the default lens was numerically validated;
  `tie_convention`, how `target_rank` breaks ties). Doc of record:
  [docs/reference/stats.md](stats.md).

## Quantities and typed errors

**Quantities**
: `tl.types.Quantity` is the marker base for numeric quantity wrappers with unit-aware
  display: `tl.types.Bytes` (memory), `tl.types.Duration` (seconds), `tl.types.Flops` (floating-point
  operations), `tl.types.Macs` (multiply-accumulates).

**FLOP convention (`flop_convention=`)**
: `"fma2"` (default; one multiply-accumulate = 2 FLOPs) or `"fma1"` (recounted
  from each op's two-term compute record; refuses typed with
  `flop_convention_unavailable` when an op has no MAC split). Accepted by
  `summary(...)`; the removed `count_fma_as_two=`
  spelling refuses typed naming it.

**Autograd-saved bytes**
: `Op.autograd_memory` / `Op.num_autograd_tensors` count the tensors the captured forward's
  autograd saved for that op's backward (each storage once within the op; a view counts in full),
  and `Layer.total_autograd_memory`, `ModuleCall.autograd_memory`,
  `Module.total_autograd_memory` and `Trace.cost.total_autograd_memory` sum those per-op bands, so a
  tensor two ops save counts twice. Capture never changes a Parameter's `requires_grad`, so on
  frozen Parameters these and `Trace.cost.pinned_autograd_memory` match what an eager forward saves.

**Module and call counts**
: `len(ModuleCall.ops)` counts the call's pass-qualified Ops (the root call included: a Layer it
  runs twice counts twice) and `ModuleCall.num_layers` the distinct Layers among them.
  `len(Module.layer_labels)` counts a Module's distinct Layers and `len(Module.params)` the
  parameter tensors it owns.
  `Trace.num_buffer_layers` counts distinct buffer Layers, not their passes. `Param.num_calls` is
  the number of Ops that consumed the parameter (0 for a never-called module's parameters).
  `Trace.num_uncalled_modules` is recorded when the capture finishes, so it survives the source
  model being collected; it reads `None` when it cannot be known (a loaded or unpickled trace
  whose model is gone), never 0.

**Conditional depth**
: `Op.conditional_depth` and `Layer.conditional_depth` count the distinct conditionals the record
  takes a direct role in, as listed in `in_conditionals`: the `evaluation` role marks the bool an
  arm's test consumed, and the `body` role marks an op an arm's entry edge feeds (an op whose
  input was computed outside that arm), not every op inside the arm. An op nested in two `if`
  bodies takes a role only where an entry edge reaches it. Nesting depth is
  `len(conditional_branch_stack)`.

**Lookup and reentrancy errors**
: `tl.errors.AmbiguousOpLookupError` is raised when a bare Op lookup matches multiple
  pass-qualified Ops; `tl.errors.ReentrantTraceError` when a trace is started while another
  trace is active.

**Report disclosure helpers** *(spellings documented-unstable)*
: `tl.debug.amp_scaled_grads_hint(*, all_nonfinite=False)` returns the shared text that names a
  possible `torch.amp.GradScaler` loss scale behind large or non-finite gradients and the
  `grad_scale=` remedy; it edits no rows or verdicts. `tl.report.flops_vs_dispatch(model,
  inputs=None, input_kwargs=None, *, sdpa_backend=None, **trace_kwargs)` runs the detached
  two-way FLOP cross-check on one model and input and returns a `FlopsCrossCheck` holding both
  totals, the per-overload native ledger and the coverage notes.

**Output-difference metrics**
: `tl.stats.cosine_distance(a, b)` is `1 - cosine_similarity` of the two flattened tensors;
  `tl.stats.relative_l2(a, b)` is `norm(a - b) / max(norm(a), eps)`; `tl.stats.relative_l1(a, b)`
  is `|a - b| / max(|a|, eps)`, the scalar fallback. Each returns a 0-d tensor; `relative_l1` and
  `relative_l2` are anchored on `a`, and mismatched element counts refuse typed.

## Distributed capture and cross-rank merging

**Distributed arming**
: `tl.distributed.arm()` opts a process into first-class capture of explicit
  `torch.distributed` collectives (required at process start for MPMD / spawn-rank
  programs; SPMD processes may arm lazily at capture entry). Sharded state (DTensor,
  tensor/pipeline parallel) still refuses typed at capture entry.

**Collective boundary op**
: A captured op whose portable `annotations["collective"]` carries the
  `collective_boundary_v1` payload: correlation key, role-indexed dual geometry, event
  disclosures, witness fields, and lifetime evidence.

**merge_ranks / MergedTrace**
: `tl.merge_ranks([trace_or_path, ...])` stitches N rank-local captures into a
  `MergedTrace` presenter (never a `Trace` subclass) at their explicit collective
  boundaries; loads rerun the derivation and refuse tampered artifacts typed.

**merge_report**
: `tl.merge_report(...)` is the graph-free merge diagnostic that derives alignment and
  consistency verdicts without constructing a merged graph, and never raises on
  conflicts.

**Merge vocabularies**
: `MergeAlignment` and `MergeValueStatus` are frozen vocabularies in `tl.types`,
  release-gated against
  [the merged-trace contract](merged_trace_contract.md).

**Trace.provenance.distributed_scope** — *unstable — no deprecation shim owed*
: The persisted shard-local capture marker (L8/F6): `None` for ordinary captures,
  `"rank_local_shard"` when any parameter or input the capture saw was one rank's local
  shard — the trace is a rank-local recording, never whole-model truth. Persists as of
  tlspec v8 with closed-vocabulary load validation; the bundle-save chokepoint refuses
  `shard_local_persistence_unsupported` under any schema that would silently drop the
  disclosure (see the [error and refusal contract](error_refusal_contract.md)).

## Backends: the backend protocol, maturity and conformance

Every backend, torch included, fills one contract, `tl.backends.BackendProtocol`. Its capability
table is declared in `BackendCapabilities` and measured by the shared conformance suite; the
measured result ships as a scorecard record per backend and shows in `tl.report.doctor()`. The
backends guide (`docs/backends.md`) carries the maturity table and links one page per backend
(`docs/backends/<name>.md`: a tutorial, a can and cannot table and the framework requirement).

**tl.backends.BackendProtocol**
: The one trace-producing contract every backend fills (`typing.Protocol`, runtime-checkable).
  Members: `name` (the registered name: `"torch"`, `"tf"`, `"jax"`, `"mlx"`, `"paddle"`,
  `"tinygrad"`), `maturity`, `framework` (a `FrameworkRequirement`), `materialization`,
  `capabilities()` (the declared `BackendCapabilities`), `runtime_available()` (`None`, or the
  refusal code `runtime_missing` / `runtime_version_unsupported`; never raises),
  `can_handle(model, args, kwargs)`, `capture(model, args, kwargs, **trace_options)` (returns a
  `Trace`; an unsupported option or capability refuses typed before any framework call),
  `reference_output(model, args, kwargs)` (the framework's own forward with no TorchLens wrapper
  installed: the oracle's independent arm), `validate(trace, ground_truth)` (returns a
  `ValidationStatus`, never a bare bool; a failure names the first offending op and a reason from
  `tl.backends.VALIDATION_REASONS`), `is_tensor(value)`, `host_copy(value)` (a detached NumPy copy
  with shape and dtype kept; 0-d stays 0-d), `realize_once(values)`, `trace_options()` and
  `refusal_codes()` (a subset of `tl.backends.REFUSAL_CODES`). Reached through
  `BackendSpec.protocol()`; registration checks the returned object against the member list. The
  older `CaptureBackend` orchestration adapter stays torch's internal capture seam; it is not the
  backend contract.

**BackendSpec.protocol** and **BackendSpec.maturity**
: `BackendSpec.protocol` is the lazy factory (`Callable[[], BackendProtocol] | None`) that returns
  the backend's protocol object. `BackendSpec.maturity` is the backend's launch label, the single
  source `tl.report.doctor()` and the docs read.

**maturity** (`"preview"` | `"stable"`)
: A backend's launch label, on `BackendSpec.maturity` and `BackendProtocol.maturity`. Torch is
  `"stable"`; tf, jax, mlx, paddle and tinygrad are `"preview"`. Only a recorded maintainer
  decision changes it, as a one-line change to the backend's registration plus the docs table;
  no code computes it, and `graduation_report` is advisory. A backend registered `"stable"` must
  have every graduation item met in its shipped scorecard (`tests/test_backend_maturity.py`).

**materialization** (`"eager"` | `"lazy"`)
: Whether a backend's op outputs are concrete values when the wrapped call returns (`"eager"`:
  torch, tf, jax, paddle) or deferred until an evaluation (`"lazy"`: mlx, tinygrad). Lazy backends
  follow the realize-once rule: the traced forward runs once to its outputs, then
  `realize_once(values)` evaluates every recorded value in one framework evaluation before any
  payload is read, so random-number consumption equals one eager forward. Random state created
  before the forward (a default-initialized weight) is realized in that same evaluation and
  recorded as the value it produced; random numbers drawn inside the traced forward refuse with
  `rng_state_in_graph`, because replay would draw them again. On eager backends `realize_once` is
  a no-op.

**tl.backends.FrameworkRequirement**
: What a backend needs from its framework, stated once (frozen dataclass): `distributions`
  (installed distribution names that version the framework, in lookup order), `minimum`
  (inclusive floor), `below` (exclusive ceiling, or `None` when none is declared), `tested_through`
  (the highest version the backend's suite has run on; a preview sets `below` or this), `pinned`
  (the exact version the per-change CI leg installs, or `None` while the leg is unpinned) and
  `extra` (the pyproject extra that installs the framework; `""` for a base dependency). The
  supported range is `[minimum, below)`. `pinned` lies inside it, the extra's specifier equals it
  and the CI leg installs `pinned` (`tests/test_backend_framework_pins.py`).
  `tl.report.doctor()` shows the installed version, the pin and the range per backend.

**tl.backends.REFUSAL_CODES** (backend refusal codes)
: The closed registry of codes a backend may raise. A refusal is a `BackendUnsupportedError`
  whose `exc.code` is a member (the code is the contract; message text is not pinned). It refuses
  before any framework call with side effects and is never a silent wrong result. A raise with a
  code outside the registry fails the backend's `c0_refusal_codes_registered` cell.

  | Code | Raised when |
  |---|---|
  | `backward_capture_unsupported`, `fastlog_unsupported`, `interventions_unsupported`, `rng_replay_unsupported`, `streaming_unsupported`, `structure_only_capture_unsupported`, `validation_replay_unsupported`, `payload_materialization_unsupported`, `intermediate_derived_grads_unsupported` | the matching `BackendCapabilities` flag is `False` and the capability was requested; one code per flag |
  | `trace_option_unsupported` | a trace option outside the backend's `trace_options()` |
  | `module_identity_mode_unsupported`, `save_level_unsupported` | the requested mode or level is outside the backend's declared set |
  | `explicit_seed_unsupported` | `seed=` on a preview backend |
  | `runtime_missing`, `runtime_version_unsupported` | the framework is not importable, or its version is outside the supported range |
  | `foreign_tensor_input` | an input leaf belongs to another framework |
  | `constant_unrealizable` | a closure or literal constant has no readable buffer; the message names the constant op |
  | `rng_state_in_graph` | the traced forward draws random numbers itself (`Tensor.rand`, dropout) on a lazy backend, so replay would draw different values; random state created before the forward is captured |
  | `unwrapped_op_in_capture` | a wrapper backend met an output or intermediate with no captured producer and the caller asked for a refusal; without that request the trace is marked `capture_verified=False` with this code as `provenance.verification_reason` and a warning |
  | `runtime_mode_unsupported` | Paddle runs outside dygraph mode, or in PIR mode |
  | `inplace_op_unsupported` | an in-place or global-state Paddle op runs inside the forward |
  | `input_mutation_unsupported` | the traced tinygrad callable mutates an input (`Tensor.assign`, `Tensor.replace`, setitem) |
  | `tensor_output_required` | the traced tinygrad callable returns no tinygrad Tensor |
  | `tinygrad_input_required` | no positional input of the tinygrad call is a tinygrad Tensor |
  | `live_payloads_required` | tinygrad replay validation or derived gradients were asked of an audit-only trace, which has no live realized-copy payloads |
  | `mid_capture_execution_unsupported` | the traced tinygrad callable realizes a tensor or runs a `TinyJit` function during capture |
  | `traced_transform_entry_unsupported` | the MLX capture entry is an `mx.compile`, `mx.grad` or `mx.vmap` wrapper rather than the eager module or callable |
  | `intervention_with_derived_grads_unsupported` | MLX `grad_options=` combined with `intervene=` or `stop_after=` in one call |
  | `intervention_result_invalid` | an MLX intervention returned something other than an `mx.array`, or changed the output's shape or dtype |

  Codes the backends already raised before the registry (`value_dependent_branch_unsupported`,
  `followed_by_unsupported`, `torchscript_boundary` and the rest) are members too; the frozenset is
  the full list.

**tl.backends.VALIDATION_REASONS** (validation decision reasons)
: The closed list of per-op decision reasons a backend's `validate` uses in
  `ValidationStatus.decisions[*]["reason"]` and the status reason counts. One word per failure: a
  backend whose cause matches a definition uses that name, and a different cause has its own entry.
  Shared: `replay_mismatch` (re-executing the op from its saved inputs did not reproduce its saved
  output), `parent_payload_missing` (a payload the replay needs is absent: the op's own saved output
  or a parent's), `no_replayable_function` (no callable can re-execute the op: none was recorded for
  a computational op, or a preview backend could not rebuild or re-invoke the saved call),
  `constant_producer_leaf`, `constant_parent_payload`, `user_intervention_replacement`,
  `effectful_region`, `perturbation_insensitive`, `dtype_tolerance_exceeded`, `op_unclassified`,
  `capture_inconsistent`. Torch's own: `arg_logging_mismatch` (the logged parent-argument positions
  disagree with the data parents, or a parent's saved output disagrees with the argument snapshot at
  its logged position), `missing_saved_args` (the argument snapshots replay needs were not
  captured), `dispatch_op_count_mismatch` (replay saw a different number of dispatcher-level ops
  than capture recorded), `bfs_incomplete` (the walk from outputs back to inputs did not reach every
  op), `metadata_invariant_exception` (a metadata invariant check raised),
  `replay_execution_exception` (the recorded function raised on re-execution),
  `unattributed_tensor_arg` (a tensor argument matches no recorded parent and no known source),
  `dropped_parent_edge_witness` (the completeness witness shows a parent edge capture dropped).
  MLX's own: `declared_parents_inconsistent` (an op's declared parents disagree with the arrays its
  saved call reads), `replay_coverage_incomplete` (the replay did not cover every captured op).
  `op_unclassified`: an op the checker cannot classify (outside every
  classification table, or an initializer that contaminated the forward), recorded as a failed
  `metadata`-phase decision on that op. `capture_inconsistent`: capture records that disagree with
  each other (a callback self-consistency failure, an equation or op count mismatch, a replay kind
  that differs from the captured kind, a region boundary check), recorded as a failed
  `metadata`-phase decision on the op or, for a trace-wide disagreement, on `"<trace>"`.
  `constant_producer_leaf`: replay reads a
  constant's saved payload because there is no function to re-execute it from (counted as
  exempted; a parentless op that does have a replayable function is still replayed and a mismatch
  still fails). `constant_parent_payload`: perturbation skips a constant parent, and the consumer
  is still perturbed on its other parents. The suite checks membership, so the list may be a
  superset of what one backend emits.

**`constant` op (closure constants and literal producers)**
: A tensor the forward used that is neither an input, a parameter or buffer of the model, nor the
  output of a recorded op (a closure tensor built outside the traced call) is recorded as a source
  op with `func_name="constant"`, no parents and a saved payload; the op that read it lists it as
  a parent. Labels follow the backend's label convention (`constant_<n>` on the previews; torch
  numbers it like its other source ops). An op with no tensor parents whose arguments are all
  literals (a `fill` or `ones` inside the forward) keeps its own `func_name`, has zero parents and
  a saved payload. Constant payloads count toward the payload budget and `save=` applies.
  Validation treats a constant as a leaf (reasons `constant_producer_leaf` and
  `constant_parent_payload`); a constant with no readable buffer refuses `constant_unrealizable`.

**`backend_scorecard:<name>` (doctor row)**
: `tl.report.doctor()` carries one row per registered backend, key `backend_scorecard:<name>`,
  label `"<name> backend scorecard"`, built from the scorecard record shipped in the backend's
  package (no framework is imported and nothing executes). Status `pass` when the record's highest
  tier is the durable tier with no failed cell, `known_broken` when any cell failed, `scope` when the framework
  is not installed here, `not_tested` when no record ships. Details read
  `"<maturity>; <passed>/<denominator> cells passed, <unsupported> unsupported, <failed> failed;
  pinned <pinned>, installed <version or absent>, supported [<minimum>, <below>)"`; the
  suggestion names the install extra when the framework is absent, otherwise the first failed
  cell.

**`python -m tests.conformance.scorecard --backend <name>`**
: Computes the backend's scorecard. `--write` writes the shipped record
  (`torchlens/backends/<name>/scorecard.json`, in the wheel), `--check` recomputes and exits
  nonzero when the shipped record differs (the backend's pinned CI leg runs it, so the record is a
  tested fact, never hand-edited) and `--graduation` prints the graduation report from the shipped
  record.

**Capability summary** (`supported` | `unsupported` | `untested`)
: The scorecard's per-flag reading of `BackendCapabilities`: `supported` when the flag is `True`
  and the suite consumed it, `unsupported` when it is `False` and its `c1_refusal_<flag>` cell
  passed (the request refuses with the contracted code), `untested` when it is `False` and the
  refusal cell failed or did not run. Each backend page's can and cannot table equals this
  summary and renders `untested` as "not yet proven" (`scripts/checks/backend_docs_lockstep.py`).

**Conformance case ids added by the protocol**
: Tier 0, the provider API (no model execution): `c0_protocol_complete`, `c0_framework_requirement_consistent`,
  `c0_refusal_codes_registered`, `c0_capability_table_complete`. Tier 1 (capture):
  `c1_graph_integrity`, `c1_intermediates_match`, `c1_shape_metadata_matches_payload`,
  `c1_validation_passes`, `c1_validation_names_reason`, `c1_constant_parents_recorded`,
  `c1_rng_consumed_once`, `c1_no_silent_gap`, one `c1_refusal_<flag>` per capability flag the
  backend sets `False` and one `c1_refusal_trace_option_<name>` per public trace option outside
  its `trace_options()`. Tier 2 (durable): `c2_payload_values_equal`, `c2_validation_after_load`. A
  typed refusal inside any other cell makes it `unsupported`; an untyped exception, a wrong value
  or a trace marked `capture_verified=False` makes it `failed`.

**Per-backend namespaces**
: `tl.backends.torch` is the stable PyTorch eager backend (`docs/backends/torch.md`).
  `tl.backends.tf` (`GradOptions`, `TFBackend`; `docs/backends/tf.md`), `tl.backends.jax`
  (`GradOptions`, `JAXBackend`; `docs/backends/jax.md`), `tl.backends.mlx` (`GradOptions`,
  `MLXBackend`; `docs/backends/mlx.md`), `tl.backends.paddle` (`GradOptions`, `PaddleBackend`;
  `docs/backends/paddle.md`) and `tl.backends.tinygrad` (`GradOptions`, `TinygradBackend`;
  `docs/backends/tinygrad.md`) are the five preview backends. Each `GradOptions` requests the
  backend's derived gradients (`trace.autograd.derived_grads`, opt-in intermediate derived gradients); each
  `<Name>Backend` is the backend's capture object behind `tl.trace(..., backend="<name>")`.

## Unstable surfaces (documented-unstable; no deprecation shim owed)

Spellings below shipped ahead of their naming-session/S2 ratification under
the provisional-name protocol: they may rename WITHOUT deprecation
shims, by declared contract. Each carries the same tag at its definition.

**splice / SpliceResult (`tl.bridge.sae.splice(model_or_log, inputs, *, site, sae, latents_edit=None, tokens=None, exclude_bos=False, keep_latents=True, log_latents=False)` returns `tl.bridge.sae.SpliceResult`)** — *unstable — no deprecation shim owed*
: The SAE splice experiment: a duck-typed SAE's reconstruction replaces one
  site and the fork replays downstream. Every metric is float64 and follows
  the SAELens evals and the Gemma Scope paper. `fraction_variance_explained_pooled`
  is the standard reading (each feature dimension, the last axis, centred on
  its own mean over tokens); `fraction_variance_explained_by_dim` gives it per
  dimension; `fraction_variance_explained` centres on one global mean. A zero
  denominator gives `nan`, `zero_variance=True` and a reason in
  `unavailable_metrics`, never 0.0. `latents` is the encoder output exactly as
  `sae.encode` returned it and `edited_latents` the latents after
  `latents_edit`, both kept by default (`keep_latents=False` drops them;
  `sae_splice_latents_large` warns once past 256 MiB). With `tokens=` (the
  integer ids the model ran on) the first output shaped `tokens.shape +
  (vocab,)` is read as logits: `ce_loss_clean`, `ce_loss_spliced`,
  `ce_loss_zero_ablated`, `ce_loss_recovered` = `(CE_zero - CE_spliced) /
  (CE_zero - CE_clean)` and `kl_divergence` = mean `KL(P_clean || P_spliced)`;
  without it they are `None` with a reason. `exclude_bos=True` keeps sequence
  position 0 clean and out of every metric. `log_latents=True` (off by
  default; live-model door only) makes `spliced` a fresh capture with the SAE
  run inside the model as ordinary ops, so the encoder ops, the latents node
  and the decoder ops are trace nodes between the site and its consumers;
  `latents_site` names the latents node (`None` with logging off). The spliced
  outputs are the same with logging on and off. Recipe:
  `docs/field-guide/pretrained_saes.md`.

**register / unregister (`tl.bridge.sae.register(name, sae, *, site)`, `tl.bridge.sae.unregister(name)`)** — *unstable — no deprecation shim owed*
: Make (or stop making) an SAE's latents the facet family `name` at `site`.
  `tl.facet(name).latent(k)` then addresses latent `k` at the doors a
  `tl.head` selector serves (`Trace.do`,
  `fork().do`; and, as a bare selector, `trace[selector]`, which returns the
  `Facet` whose `.value` is
  the read, and `tl.trace(save=selector)`, which keeps the ops that read
  needs), and `trace[site].facets[name]` (every latent),
  `facets[f"{name}[{k}]"]` or `facets[f"{name}[{j},{k}]"]` read the encoder
  activation. A write edits the latents and adds the decoder-space change
  `decode(z') - decode(z)` to the site (for an affine decoder, `delta *
  W_dec[k]`), so the reconstruction error is untouched. A string site must
  name one op of a layer that runs once (`sae_latent_site_ambiguous`
  otherwise); a selector site is every match, and an edit there fans out and
  reports its count (`facet_selector_fanout`). Registration is process-wide.

**FacetSelector.latent (`tl.facet(name).latent(index)`)** — *unstable — no deprecation shim owed*
: Latent `index` (an int, or a list of ints edited together in one write) of
  the SAE registered as `name`; refuses `sae_latent_family_unknown`,
  `sae_latent_index_invalid` or `sae_latent_index_unknown`.

**reconstruct (`tl.bridge.sae_lens.reconstruct(trace, site, sae)`)** — *unstable — no deprecation shim owed*
: The SAE's reconstruction of a saved site activation,
  `sae.decode(sae.encode(out))`; a callable SAE without that pair is called on
  the activation (an SAE Lens `SAE`'s forward is the same encode then decode).
  Replaces the removed `decode(log, site, sae)`, which passed the activation
  to `sae.decode` as if it were latents. `tl.bridge.sae_lens.encode(trace, site,
  sae)` gives the latents.

**write_delphi (`tl.bridge.sae.write_delphi(latents, tokens, *, site, model_name, n_splits=5, directory)`)** — *unstable — no deprecation shim owed*
: Writes SAE latents as the latent cache delphi's `LatentDataset` loads
  (eai-delphi 0.1.3): `directory/site/{start}_{end}.safetensors` splits with
  `locations` (example, position, latent), `float16` `activations` and the
  `tokens`, plus a `config.json` naming `model_name`. `latents` is dense
  (`SpliceResult.latents`, every activation above delphi's 1e-5) or the
  `UnitTopK` of `tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` over a
  dataset (only the retained top entries). Recipe:
  `docs/field-guide/sae_latent_export.md`.

**write_sae_dashboard (`tl.bridge.sae.write_sae_dashboard(latents, tokens, *, latent_ids=None, k=20, path)`)** — *unstable — no deprecation shim owed*
: Writes SAE latents and each latent's top examples as the JSON SAEDashboard's
  `SaeVisData.load_json` loads (sae-dashboard 0.8.0): one "TOP ACTIVATIONS"
  group of whole sequences per latent, ranked by activation, then lowest
  example, then lowest position (the `tl.stats.TopK` order). Dense latents
  carry every token's activation and give `feature_stats` `max` and
  `frac_nonzero`; a dataset `UnitTopK` carries the ranked token's value only.
  `latent_ids` defaults to every latent that fired; `sae_export_dashboard_large`
  warns once past 256 MiB. Recipe: `docs/field-guide/sae_latent_export.md`.

**similarity_repository_id (`rdm_evolution(...).similarity_repository_id`, `tl.stats.CKA.similarity_repository_id`, the `SimilarityScore` that `tl.stats.cka` and `tl.stats.CKA.compute()` return, `ScoreInterval.similarity_repository_id`, the `neuro.rdms()` source-mode descriptor)** — *unstable — no deprecation shim owed*
: The similarity-repository catalogue identifier of the exact measure variant a
  TorchLens result computed, with `*` for the implementing repository (for
  example `measure/*/cka-kernel=linear-hsic=gretton-score` for linear CKA,
  `rdm/*/correlation` for a correlation RDM). On `rdm_evolution` results it is
  `None` when the catalogue has no exact match (the `"manhattan"` metric); the
  `neuro.rdms()` descriptor is then absent. The full table is in
  [the brain-benchmarking seam](brain_benchmarking_seam.md).

**SimilarityScore (`tl.stats.SimilarityScore`; returned by `tl.stats.cka` and `tl.stats.CKA.compute()`)** — *unstable — no deprecation shim owed*
: A `float` subclass: the score itself (arithmetic, comparison, `repr` and JSON
  encoding are a float's; arithmetic returns a plain `float`) plus
  `similarity_repository_id`, the exact measure variant it computes
  (`measure/*/cka-kernel=linear-hsic=gretton-score` for linear CKA). `to_dict()`
  is the serialized form that keeps the identifier
  (`{"score": ..., "similarity_repository_id": ...}`); pickling keeps it too.

**ScoreInterval (`tl.repgeom.ScoreInterval`; returned by `tl.repgeom.compare_rdms_interval` and `tl.repgeom.cka_interval`)** — *unstable — no deprecation shim owed*
: A TorchLens score with a descriptive uncertainty interval beside it:
  `score` (the unchanged estimator on all stimuli), `low` and `high` (a
  percentile bootstrap over stimuli), `confidence`, `num_resamples`, `num_valid_resamples`
  (resamples with a finite score), `seed`, `basis` (how the interval was
  made, in words) and `similarity_repository_id` (the measure variant `score`
  computes: the linear CKA identifier from `cka_interval`, `None` from
  `compare_rdms_interval`, whose precomputed RDMs do not name their metric),
  plus `width` and `to_dict()` (every field). `rdm_compare_interval(rdm_a,
  rdm_b, method, num_resamples=1000, seed=0, confidence=0.95)` resamples
  stimuli and recomputes `compare_rdms`, dropping self-pairs; `cka_interval(a,
  b, ...)` does the same for linear CKA. Descriptive, not a test; inference
  stays with rsatoolbox.

**ScoreParts (`tl.bridge.brain_score.ScoreParts`; returned by `tl.bridge.brain_score.score_parts` and, per layer, `tl.bridge.brain_score.per_layer`)** — *unstable — no deprecation shim owed*
: A benchmark score split into named parts so a ceiling-normalised value
  never stands in for the raw one: `raw`, `ceiling_normalised`, `ceiling`,
  `error`, `note`. A Brain-Score result that carries a ceiling gives its
  value as `ceiling_normalised` and its raw attribute as `raw`; a result
  without one is raw and `ceiling_normalised` is `None`. TorchLens reads
  these numbers; it computes no ceiling.

**metric_disagreement (`tl.repgeom.metric_disagreement`; returns `tl.repgeom.MetricDisagreement` with `tl.repgeom.DisagreeingPair` rows)** — *unstable — no deprecation shim owed*
: The rank order of sites (layers or models) under each comparison metric on
  one extraction, with every pair of sites that two metrics order oppositely
  flagged. Input is `scores[metric][site]` (a float or a `ScoreInterval`);
  higher scores rank first, ties share their average rank, missing scores
  take no side. Each flagged pair names the site each metric ranks higher
  and, when the scores carry intervals, whether the opposite orders are
  separated by them (`firm`). `MetricDisagreement.metrics` lists the metric
  names in input order and `MetricDisagreement.ranks` holds
  `ranks[metric][site]` (1 is the highest score; ties share their average
  rank; a NaN score gets a NaN rank). Each `DisagreeingPair` names its two
  sites as `site_a` and `site_b` (in the report's site order) and, in
  `higher`, the site each metric scores higher (`None` for a tie or a
  missing score). `table()` gives the plain-text report and
  `to_dict()` the data form.

**neuro dependency conflict (refusal code `neuro_dependency_conflict`)** — *unstable — no deprecation shim owed*
: When rsatoolbox is installed but cannot import because another installed package
  pins scikit-learn below what it needs (the `torchlens[neuro,brainscore]` install),
  `tl.neuro` leaves `rdms` and `datasets` out of `__all__` and `dir()`, and touching
  them raises `MissingDependencyError` with this code. The diagnosis reads package
  metadata only. `exc.fields` carries `dependency_version`, `conflicting`,
  `conflicting_version` and `pinned_by` (`None` when no pinning package is found).

**neuro.compare (`tl.neuro.compare`; needs no extra)** — *unstable — no deprecation shim owed*
: One call that scores every site of one extraction against one target with
  several descriptive analyses: `tl.neuro.compare(source, target,
  analyses=("rsa", "ridge", "cka"))`. Each site's activations are read once
  from what `source` (a Trace with saved activations, an extraction, or a
  `{site: array}` mapping) already holds; the model is never re-run. `"rsa"`
  is `compare_rdms` over `rdm(site)` and `rdm(target)` (`rdm_metric`,
  `rsa_method`); `"ridge"` is a cross-validated ridge score (a seeded
  `num_folds` split, the penalty per fold chosen from `ridge_alphas` by exact
  leave-one-out on the training fold, then the mean over target columns of
  the Pearson r between out-of-fold predictions and the target); `"cka"` is
  linear CKA. `num_resamples=` adds a `ScoreInterval` per score. The returned
  report has `scores`, `intervals`, `settings`, `ridge_alphas_chosen`,
  `similarity_repository_ids`, `table()`, `disagreement()` (a
  `MetricDisagreement`) and `to_dict()`. Descriptive only: inference,
  variance partitioning, voxel maps and stacked fits stay with rsatoolbox,
  himalaya, scikit-learn and Net2Brain.

**site_key (`Op.site_key`)** — *unstable — no deprecation shim owed*
: The portable structural-position identity of one retained op
  (`site_key_v1`): a `"s1|"`-prefixed string of the pass-free module address
  stack, the normalized op type, the output slot, and a 1-based occurrence
  ordinal within one pass-qualified innermost module call instance, minted
  from raw records at grouping time on every backend. Policy-independent
  (identical whether grouping ran, degraded, or was off), process-portable
  (no barcodes, ids, or arg hashes), and a BRIDGING relation: two captures of
  the same program agree on site keys even when their layer labels disagree.
  It proves POSITION, never source identity — cross-capture joins carry a
  per-call-instance cardinality guard, a source-location witness, and a
  three-tier verdict (corroborated / positional / refused). Editing the
  model's `forward()` changes downstream sites: keys bridge captures of the
  SAME program, never a program diff. The `"s1"` prefix makes any future
  re-keying a visible schema event. Persists as of tlspec v8 with
  byte-exact recomputation at load; older artifacts are wholly keyless.

**Layer.site_peers** — *unstable — no deprecation shim owed*
: Layers sharing any of this layer's site keys within the trace (the
  reused-relu cohort surface), computed live per call and never persisted.
  Refuses typed `site_key_unavailable` on keyless layers — a legacy artifact
  never collapses into a `None`-key peer-of-everything.

**Layer.shape_summary** — *unstable — no deprecation shim owed*
: Derived (never persisted) data string summarizing output shapes ACROSS
  PASSES of one layer: `None` for single-pass and shape-uniform layers; one
  varying axis renders `"A->B"` (monotone) or `"A-B"` (min-max); multi-axis
  or rank-varying layers render first-to-last full shapes
  (`"2x64x8x8->2x512x4x4"`). Distinct from the internal module-run fold
  summary (`ModuleRepeatFold.shape_summary`), which summarizes across a
  repeated MODULE RUN. The string legitimately contains `->`; renderers must
  HTML-escape it (escape-at-render, never assert-absence).

**Op.has_params** / **Layer.has_params** — *unstable — no deprecation shim owed*
: Whether the operation (or any pass of the layer) consumed a parameter tensor.
  Derived from the captured parameter barcodes; read-only.

**Op.has_saved_grad** — *unstable — no deprecation shim owed*
: Whether a gradient was captured and saved for this operation's output.

**Buffer.overwrite_index** — *unstable — no deprecation shim owed*
: The buffer pass of the buffer's most recent graph version (how many times the
  buffer was overwritten when that version was read); `None` when the buffer has
  no graph version.

**Buffer.writer_label** — *unstable — no deprecation shim owed*
: The final label of the op that produced the buffer's current value (the most
  recent write), or `None` when nothing in the forward pass wrote it. Always a
  live lookup key into the trace.

**Buffer.write_ops** — *unstable — no deprecation shim owed*
: The buffer's graph versions produced by writes, excluding the static initial
  read.

**Buffer.grad_memory** / **Param.grad_memory** — *unstable — no deprecation shim owed*
: Size in bytes of the captured gradient: the final observed gradient of the
  buffer, or the parameter's `.grad` tensor.

**Trace.nonfinite_ops** — *unstable — no deprecation shim owed*
: The queryable per-op NaN/Inf record: a tuple of pass-qualified op labels
  (`Op.label`, each a valid `trace[label]` key) whose output held at least one
  NaN or Inf. On default captures it derives from the memoized saved-payload
  scan already backing `print(trace)` — zero capture-time cost, one scan paid
  at first query. When the capture ran with
  `CaptureOptions(on_nonfinite='track')` it serves the capture-time verdicts
  instead, which also cover ops that retained no payload. An empty tuple is
  only as strong as its coverage; read `Trace.nonfinite_coverage` first.

**Trace.nonfinite_coverage** — *unstable — no deprecation shim owed*
: The frozen evidence disclosure behind `nonfinite_ops`: `basis`
  (`"capture"` vs `"saved_payloads"`), `checked`, `nonfinite`, `unchecked`
  (dtypes with no runnable finiteness kernel), `unexamined` (ops the scan
  could not look at), and `unmapped` (capture-basis events whose op did not
  survive postprocessing). The programmatic twin of the prose coverage-gap
  note in `first_nonfinite()`.

**on_nonfinite='track' (CaptureOptions)** — *unstable — no deprecation shim owed*
: Session-time opt-in: torch capture records a per-op finiteness verdict for
  every committed op output. Never changes control flow (`on_nonfinite='raise'` stays
  the stop-and-throw and is independent); device flags are drained in one
  batch after the forward, never a per-op CUDA synchronization.
  `structure_only=True` refuses the combination typed. Load restores the
  default `False`; loaded traces serve the saved-payload basis. The default
  stays `False` (no overhead); the `print(trace)` NOT-CHECKED health line
  names this option.

**grouping= (trace kwarg) / trace.options.grouping** — *unstable — no deprecation
shim owed; S2-gated vocabulary*
: Closed-vocabulary grouping-policy knob: `"structural"` (default — call-site
  grouping, `docs/guides/grouping.md`), `"strict_shapes"` (reserved; refuses typed until its
  own reviewed design lands), `"fold_sites"` (the D1 within-capture folding
  axis; refuses typed on plain captures until an affirmative D1 ruling).
  Unknown values refuse `grouping_invalid`; legal-but-not-entry-legal values
  refuse `grouping_policy_unavailable`. The mirror field `trace.options.grouping`
  records the requested value. Distinct from the display-only `fold_repeats`
  viz knob, which folds repeated module runs at render time and never
  changes grouping.

**trace.options.grouping_policy** — *unstable — no deprecation shim owed; S2-gated
vocabulary*
: The persisted, load-validated `grouping_policy_v1` stamp recording HOW the
  trace was grouped: `policy` (the step-7 grouping that actually ran —
  `structural` / `params_only` / later `fold_sites`), `requested` (the knob
  mirror), `folded_sites` and `site_join` (two distinct site-granular axes:
  step-7 folds vs product-layer joins), `detector` (`callsite_v1` for
  call-site grouping, `grouper_v1` for the structural grouper that serves
  captures without call-site evidence, `unknown` when grouping did not run),
  `effective`, and
  `note`. Loads validate against the exact writer key set, closed
  vocabularies, and coherence rules C1–C8; parse failure or incoherence
  warns once and settles to THE canonical degraded representation
  (`policy="unknown"`, `note="grouping_stamp_<reason>"`), which
  round-trips byte-stable and stays degraded — verdicts only worsen across
  persistence. Legacy pre-stamp artifacts settle silently to
  `grouping_stamp_legacy`; degraded stamps refuse stamp-consuming
  operations typed. Persists as of tlspec v8 with C1-C8 load validation.

**L1 grouping refusal codes** — *unstable — no deprecation shim owed;
S2-gated*
: `layer_site_ambiguous`, `site_key_unavailable`, `grouping_invalid`,
  `grouping_policy_unavailable`.

**structure_only (capture kwarg) / trace.options.structure_only** — *unstable — no
deprecation shim owed*
: `tl.trace(model, x, capture=CaptureOptions(structure_only=True))` runs the
  capture under the structure-only contract: the op graph, module hierarchy,
  parameter geometry, and per-op shape/dtype are recorded with every
  value-bearing claim a HYPOTHESIS; value payloads are never retained,
  value-requiring consumers refuse typed through
  `torchlens._capture.structure_only.require_structure_only_capability`, and
  value-dependent branches refuse with the user's source line
  (device-neutral). The mirror field `trace.options.structure_only` declares the
  mode. Capability contract:
  [structure_only_capabilities.md](structure_only_capabilities.md).

**Trace.discharge_against(real_trace)** — *unstable — no deprecation shim owed*
: Discharges a structure-only trace's hypotheses against an ordinary settled
  COMPLETE capture of the same graph. Returns a frozen `StructureDischarge`
  (per-claim table + overall corroborated/refuted verdict, positional join
  licensed by graph-shape digest equality); a REFUTED discharge flips
  hypothesis consumers to typed refusals. Neither trace is mutated.

**StructureClaimStatus (hypothesis / corroborated / refuted)** — *unstable —
no deprecation shim owed*
: The tri-state evidence class of a structure-only trace's value-bearing
  claims; never silently promoted.

**Structure-only refusal codes** — *unstable — no deprecation shim owed;
S2-gated*
: `structure_only_option_conflict`, `structure_only_values_unsupported`,
  `value_dependent_branch_unsupported`, `meta_kernel_unavailable`,
  `structure_only_{save,runnable,replay,validation,backward,episode}_unsupported`,
  `structure_only_refuted_hypothesis`, `structure_only_discharge_precondition`,
  `structure_only_type_invalid`.

**Selector / Selection / `__selection__`** — *unstable — no
deprecation shim owed (do/Selection spellings slate-ratified subject to D7)*
: `tl.Selector` is the composable selection QUERY — a frozen AST over leaf
  terms (selector / receptive-field box / gradient-RF / facet / param / unit)
  and boolean combinators, trace-independent; `trace[selector]` (the one
  resolution door) returns a `tl.Selection` — frozen, trace-bound, an ordered
  tuple of `tl.types.Site(key, mask, provenance)` rows, never capped (a row's
  `.record` is the `Op`, `Param` or `EdgeUseRecord` it names, read through a weak
  link to the trace; `selection_trace_released` once the trace is gone); a
  spelling that names one thing (an exact label, a target spec, a facet
  selector) must match exactly one site and returns that record
  (`site_spelling_not_unique` otherwise). `Selection` is a container: `len`,
  iteration, `[i]` (a `Site`) and slices (a `Selection`), `.labels` (property,
  pass-qualified op labels), `.selector` (the selector that produced it, or
  `None`), `.kind`, `.is_empty`, `.digest`, `.align_to(other)` and
  `.to_pandas()`; its repr is `Selection[ACT]` followed by the counts,
  `(N sites, M elements)`. The former `SiteTable` (`resolve_sites`, `.first`, `.where`) folded into it:
  `selection[0]` and `trace[selector & other]` spell those. TWO-LEVEL DENOTATION:
  (touched-site family, selected-element set); zero-mask entries are retained
  first-class (`len()` and iteration count them; the graph producers and
  `trace.subgraph` do not seed from them); `.is_empty`/`__bool__` are ELEMENT-level; `bool()` on the QUERY
  type refuses typed (`selection_bool_ambiguous`). Anything region-shaped
  implements `__selection__`; `Site.mask` returns a FRESH materialization
  (mutation cannot alter the selection). Masks are exact AS SETS; producer
  inexactness rides the closed `provenance.relation` lattice
  (`exact | upper_bound | lower_bound | unknown`). `tl.Selection` is
  session-time only, never persisted. Selection kinds `ACT | PARAM | EDGE`
  are a closed vocabulary; mixed kinds refuse `selection_kind_incompatible`.
  See also: [Completeness of the selection algebra](selection_completeness.md).

**Selection operators `| & - ~` (+ reflected)** — *ratified set (slate 5.4);
semantics unstable-documented*
: Same-site operands compose as masks; different-site yields a MULTI-SITE
  selection; `-` never un-touches sites (`fam(A-B) = fam(A)`); `~` is the
  touched-site mask complement (never predicate negation and never
  model-universe). NO `__xor__`: `(a - b) | (b - a)` spells it.
  `BaseSelector` keeps its shipped composite semantics; `selector - selector`
  desugars to `and(a, not(b))`; a selector composed with a region producer
  defers to the Selection algebra.

**tl.units / tl.param / tl.random_like** — *unstable — no deprecation
shim owed*
: Stage-1 producer constructors: `units(site, indices)` (explicit site +
  index set), `params(name, mask=None)` (named-parameter element region),
  `random_selection(like=, within=, seed=)` (seeded size-matched control
  sampled without replacement inside `within`; too-small populations refuse
  `selection_unresolvable` / `population_too_small`).

**Value producers: tl.top_k / tl.top_fraction / tl.threshold / tl.sign** —
*unstable — no deprecation shim owed*
: Select by WHAT VALUES ARE on the resolution trace's retained activations:
  `top_k(within, k, by='value'|'abs', largest=)` (global ranking across the
  population with a deterministic stable tie-break: canonical site order,
  then flat index; too-few rankable elements refuse `population_too_small`),
  `top_fraction(within, fraction)` (`k = ceil(fraction * population)`),
  `threshold(within, above=, below=, by=)` (strict open bounds; both = band),
  `sign(within, 'positive'|'negative'|'zero'|'nonzero', atol=)` (`'zero'` is
  the sparsity mask and the single-capture "didn't fire on this input"
  spelling). `within` is `None` (default population: every retained tensor
  site — value claims exist only about retained values), a site label
  string, or any ACT selection; PARAM/EDGE populations refuse
  `selection_kind_incompatible`. Masks are computed at resolve time and are
  exact as sets about THIS capture (`provenance.relation="exact"`); unsaved
  payloads refuse `value_not_saved`; NaN elements never satisfy a criterion;
  complex payloads refuse ordered comparisons
  (`value_criterion_invalid` — use `by='abs'`).

**Statistical producers: tl.dead / tl.saturated / tl.low_variance** —
*unstable — no deprecation shim owed*
: Inherently MULTI-SAMPLE selections over an explicit evidence set:
  `samples=` takes an iterable of Traces (a `Bundle` iterates its members),
  at least TWO — the single-capture form is deliberately a different
  spelling (`sign(site, 'zero')`), so one name never means two things. The
  resolution trace supplies geometry and population only; include it in
  `samples` to count it as evidence. `dead(samples, atol=)` selects elements
  with `|v| <= atol` in EVERY sample; `saturated(samples, low=, high=, atol=)`
  selects elements pinned within `atol` of the same declared bound in every
  sample (a bound is required — saturation is relative to the
  nonlinearity's range); both make dispositional claims from finite
  evidence and declare `provenance.relation="upper_bound"` (observed-silent
  is a superset of truly-dead) with the sample count disclosed in
  `provenance.source`. `low_variance(samples, threshold=)` names the sample
  statistic itself (elementwise unbiased variance in float64) and declares
  `exact`; its touched-family complement `~low_variance(...)` is the
  high-variance selection. Missing sample sites, unsaved sample payloads,
  and cross-sample shape drift refuse typed (`site_not_in_trace` /
  `value_not_saved` / `mask_shape_mismatch` with the offending sample
  named), never silently shrink the evidence set.

**Graph-structural producers: tl.neighborhood / tl.between** — *unstable —
no deprecation shim owed*
: Select by WHERE OPS SIT IN THE EXECUTED DAG.
  `neighborhood(selector, hops=, direction=)` selects every op within N recorded
  dataflow hops of a seed region (`hops=0` is the seed family itself;
  `direction` is `'both' | 'upstream' | 'downstream'`).
  `between(sources, sinks)` selects the executed sub-DAG carrying influence
  from the source region to the sink region — exactly the ops on at least
  one directed source-to-sink path, endpoints included; no directed path
  resolves to the EMPTY selection (emptiness is disclosure, never an
  error). Operands are site label strings (a bare layer label on a
  multi-pass layer is the all-passes Layer spelling), `Op`/`Layer` handles,
  or any ACT selection; `between` endpoints also accept lists of regions.
  A seed or endpoint site that carries at least one selected element counts
  as its whole node (a partial mask never shrinks a graph region). A site
  whose element mask is EMPTY seeds nothing: `x - y` keeps every site `x`
  touched, with an empty mask where `y` removed all of it, and those sites
  are not part of the region the closure grows from, so
  `tl.neighborhood(tl.all_sites() - region, ...)` grows from the sites
  outside `region`. `trace.subgraph(selection)` takes the same member set.
  Membership is a structural fact about THIS
  capture, so entries are whole-site masks with
  `provenance.relation="exact"`. PARAM/EDGE operands refuse
  `selection_kind_incompatible`; unknown sites refuse `site_not_in_trace`;
  member sites with no output index space refuse `non_tensor_site` /
  `no_index_space`. Both producers are pure functions over the ONE
  executed-DAG graph substrate (`torchlens._selection.graph._ExecutedDag`,
  shared with the influence-geometry path machinery), the seam a future
  graph-MOTIF producer plugs into. See also: [Completeness of the selection
  algebra](selection_completeness.md) (why `tl.neighborhood` and `tl.between` are necessary).

**TraceSlice / trace.between / trace.subgraph** — *unstable — no
deprecation shim owed*
: `trace.between(sources, sinks)` returns the influence region as a
  sub-DAG VIEW: the same member set the `tl.between` producer selects (one
  idea, two binding modes), presented as a `TraceSlice` — a frozen
  presenter (composition, never a `Trace`/`Bundle` subclass, exactly like
  `MergedTrace`) exposing the member `Op` records in execution order, the
  region's internal dataflow edges, and an EXPLICIT BOUNDARY: every edge
  crossing into the region (`boundary_in_edges` — the external
  dependencies) or out of it (`boundary_out_edges`) is declared, never
  silently dropped, plus the region's entry/exit ops
  (`source_ops`/`sink_ops`). A slice deliberately offers NO
  save/replay/validate (a region with dangling external inputs cannot
  honour them); `Trace.save` refuses it typed (`slice_save_unsupported`).
  `__selection__` lifts the member family back into the algebra, so a
  slice composes under `| & - ~` and feeds `do()`.
  `trace.subgraph(selection)` is the general door: any ACT region — an
  n-hop neighborhood, explicit units, a future motif matcher's hits —
  presents as the same view via its touched-site family. Session-time
  only; never persisted.
**Differential producers: tl.changed / tl.top_changed** — *unstable — no
deprecation shim owed*
: Select by HOW VALUES DIFFER between two runs. The SUBJECT is the resolution
  trace (masks land on its sites); the REFERENCE is one explicit Trace
  argument (a fork after `do()`, a capture on another input — deliberately
  pairwise: multi-sample dispersion is `low_variance(samples=...)`). The
  delta is directional, `subject - reference`, elementwise in float64;
  `by='abs'` (default) compares `|delta|`, `by='signed'` the signed delta
  (increased/decreased). `changed(reference, within=None, above=, below=)`
  applies strict bounds (neither given defaults to `above=0.0` — bare
  `changed(reference)` is the "every element that moved" intervention-effect
  mask); `top_changed(reference, within=None, k= | fraction=, by=,
  largest=)` globally ranks deltas with the deterministic stable tie-break
  (exactly one of `k`/`fraction`; `largest=False` selects the least-moved
  control; too-few rankable refuse `population_too_small`). STRUCTURE
  HONESTY: the reference must retain a shape-identical payload at every
  population site's pass-qualified address — missing sites
  (`site_not_in_trace`), unsaved payloads (`value_not_saved`), shape drift
  (`mask_shape_mismatch`, all naming the reference), and structural-site-key
  disagreement (label coincidence across different architectures) refuse
  typed; structures are NEVER silently intersected, and resolving against
  the reference itself refuses (vacuous by construction). NaN deltas never
  satisfy or rank; complex deltas refuse `by='signed'`. Relation is `exact`
  (the pairwise delta between THESE two captures is complete evidence).
  PARAM populations refuse `selection_kind_incompatible`: Param records hold
  a LIVE parameter reference, never capture-time payloads, so
  "weights that shifted between checkpoints" cannot be claimed honestly from
  Trace records (compare runnable-save `state_dict_v1` blobs; a
  capture-time weight differential is a named possibility, not a promise).

**Cross-pass producers: tl.stable_across_passes / tl.pass_variance** —
*unstable — no deprecation shim owed*
: Select by behaviour ACROSS a recurrent layer's passes, pass-qualified
  throughout (one capture is the evidence; population entries group per
  layer). `stable_across_passes(within=None, atol=, passes=)` selects
  elements whose range (max - min, float64) across the pass window is
  `<= atol`; `pass_variance(within=None, above=, below=, passes=)` applies
  strict bounds on the unbiased (n-1) cross-pass variance (at least one
  bound; "variance explodes late" = `above=` on a late window).
  `passes=None` uses every population pass; an explicit window is an
  iterable of >= 2 distinct 1-based pass indices (each must be in the
  population, else `site_not_in_trace`, pass-qualified). HONESTY FLOOR: a
  layer contributing fewer than two window passes refuses
  `population_too_small` with a teaching message (a single-pass cross-pass
  claim is vacuous — the default population on a feedforward model refuses
  loudly rather than returning an everything-mask). Cross-pass shape drift
  refuses `mask_shape_mismatch`; complex payloads refuse
  `value_criterion_invalid`; NaN at any window pass excludes the element.
  The element population is the INTERSECTION of the window entries' masks
  (complete evidence per element), and the mask lands on EVERY window
  pass-site — the claim is about the unit across the window, so `do()`
  edits every window pass. Relation is `exact` (the statistic of this
  capture's passes; the `low_variance` precedent).

**Subspace producer: tl.subspace** — *unstable — no deprecation shim owed*
: Select the elements a DIRECTION (or small subspace) in activation space
  lives on — probe directions, steering vectors, PCA components, SAE
  decoder rows. `subspace(within, basis, *, origin=, method=None, dim=-1,
  atol=0.0)` takes one direction `[d]` or a stack `[k, d]` (canonicalized to
  float64). Under `do()` the edit acts on the COMPONENT along the span of the
  basis rows: with `Q` the rows, on the coordinates `c = Q x`, written back as
  `x + Q^T (edit(c) - c)`; the orthogonal complement is unchanged. Coordinate
  verbs (`zero_ablate`, `scale`, `clamp`, `add_noise`, `mean_ablate(dim=None)`, hook
  callables, ...) act on `c`; value verbs (`patch_from`, the interchange;
  `replace_with`, `steer`, ...) write a value's component; other verbs refuse
  `subspace_edit_unsupported`. An orthonormal basis is used as given; any
  other full-rank basis is orthonormalised (Gram-Schmidt in row order, signs
  kept). A subspace combined with set operators, or resolved, refuses in
  `do()` (`subspace_projection_composed`); `within=` restricts positions
  (a partial-axis mask refuses `subspace_projection_partial_axis`).
  `resolve()` reports the basis's SUPPORT SET: every element whose coordinate
  on the bound axis carries weight `|w| > atol` in at least one basis vector,
  expanded across the site's other axes (a dense direction supports the WHOLE
  axis) (docs/intervention_explainers.md, "Subspace edits: the component along
  a basis"). BASIS PROVENANCE IS MANDATORY: `origin=` is
  a required non-empty account of where the basis came from; origin,
  optional `method=`, geometry, and the basis's sha256 content digest are
  stamped into every resolved entry's `provenance.source` (and therefore
  `do()` audit records); the frozen
  `tl.types.BasisProvenance` record is the programmatic
  face. DIMENSION HONESTY: `dim=` names the bound site axis (default `-1`,
  the trailing feature axis; conv channel directions are `dim=1`); a site
  whose extent there differs from `d` refuses `selection_unresolvable` /
  `basis_dim_mismatch` — never a silent broadcast, truncation, or padding.
  `within=` is REQUIRED (a direction is minted for one representation
  space). Non-finite bases and rows entirely at-or-below `atol` refuse at
  construction (vacuous direction). Resolution is geometry-only (no payload
  reads — unsaved sites resolve); relation is `exact`; PARAM/EDGE
  populations refuse `selection_kind_incompatible` (a parameter-space
  direction is a named possibility, not a promise). PORTABLE ROTATIONS: with a
  builtin verb the staged per-site edit is the builtin `subspace_edit`, whose
  stored arguments are the verb, the float64 orthonormal rows it applies (a
  tensor sidecar), the bound `axis`, the rows' `basis_sha256` and the basis
  provenance (`origin`, `method`, `origin_sha256`, `source_dtype`), so
  `tl.io.save_intervention(trace, path, level="portable")` reloads it as the
  same projection, never as the plain verb over the whole site (the target path
  follows the save/load symlink rule: its final component may not be a
  symlink, a symlinked parent passes). A rotation
  learned elsewhere (DAS, ReFT) is applied as `tl.subspace(site, R[:k],
  origin=...)`; TorchLens applies it and does not fit it. At load a malformed
  field refuses `subspace_spec_invalid`, rows changed after saving refuse
  `subspace_spec_basis_tampered`, non-orthonormal rows refuse
  `subspace_spec_basis_not_orthonormal`. A verb that is not builtin
  (`patch_from`, population verbs, hook callables) keeps the edit
  `opaque_audit`. `BasisProvenance.source_dtype` is the dtype of the caller's
  basis before the float64 canonicalisation. A subspace restricted by a
  `within=` element mask (one position, as LoReFT edits) stores the dense
  `mask` and its `mask_sha256` too, so it still saves portably; a malformed
  stored mask refuses `subspace_spec_invalid` and one changed after saving
  refuses `subspace_spec_mask_tampered`; any other masked edit keeps its
  portability. A stored keyword the loader does not know refuses
  `subspace_spec_invalid`.

**SelectionError / selection refusal codes** — *unstable — no deprecation
shim owed; S2-gated*
: One carrier class (`torchlens.errors.SelectionError`, catalogued in the
  intervention error catalog) for the closed codes
  `selection_trace_mismatch`, `selection_bool_ambiguous`,
  `selection_kind_incompatible`, `selection_unresolvable` (closed reason set
  `site_not_in_trace | value_not_saved | non_tensor_site | no_index_space |
  mask_shape_mismatch | facet_write_mask_unavailable | population_too_small |
  multipass_bare_label | value_criterion_invalid | basis_dim_mismatch`),
  `selection_apply_invalid` (stage 2), `selection_trace_released` (`Site.record`
  after the trace was released), and `slice_save_unsupported`
  (`Trace.save` on a `TraceSlice` presenter). The `multipass_bare_label` reason
  is the `tl.units` face of the multi-pass bare-label ambiguity refusal: a
  bare layer label addresses only single-pass layers, and each pass of a
  recurrence-grouped layer must be named pass-qualified (`label:pass`, `label:last`, or
  `label:all` for every pass); the
  string/`tl.label` addressing face (`trace[label]` included) is
  `multipass_bare_label_ambiguous` on `AmbiguousSiteError` (both teaching refusals name
  every pass-qualified spelling, `:last`, `:all` and `trace.layers[label]`; spellings
  DOCUMENTED-UNSTABLE).

**tl.Edit / do(selection, edit)** — *Edit ratified (slate 5.5, subject to D7
default-keep); mask-application semantics documented-unstable*
: `tl.Edit` is the public edit-object type (the old class
  spelling is removed, U00 clean break). `repr(edit)` prints the
  public spelling: a builtin edit prints as the public call that builds it
  (`tl.scale(2.0)`, default-valued keywords omitted; evaluating it
  rebuilds an equal edit), any other edit as `tl.Edit('name', signal=...,
  portability=...)`. An `Edit` is a frozen record of its identity:
  `name`, positional `args`, keyword `kwargs`
  (a tuple of name and value pairs), `signal`, `portability`, `metadata`
  (name and value pairs, such as `("keep_grad", True)` or the selection
  disclosure a masked `do()` adds), `is_batch_independent`,
  `is_compatible_with_append` (an edit that may change the output shape is
  refused on an append rerun unless this is true) and `selection_recipe`
  (the selection recipe a selection-targeted `do()` attaches; `None`
  otherwise). The hook `factory` is session-side. `Edit._PORTABLE_STATE_SPEC`
  is the class-level table of how a saved spec keeps each field: `factory`
  is dropped, `selection_recipe` is saved as a recursive blob, every other
  field is kept. `trace.do(selection, edit)`
  applies an edit to a resolved selection under the NORMATIVE
  MASK-APPLICATION CONTRACT: the edit hook computes its full replacement
  exactly as today (helpers stay mask-oblivious), then the ENGINE applies
  `torch.where(mask, edited, original)` on a FRESH tensor — never in-place
  on, never a view aliasing, the stored capture value. Whole-site masks
  short-circuit the scatter (exactly today's behavior). No broadcasting in
  v1; shape/dtype/device/broadcast mismatches and ineligible sites refuse
  `selection_apply_invalid` (closed reason set
  `shape | dtype | device | broadcast | not_maskable`). Learned-parameter
  edits route through PARAMETER SUBSTITUTION on the replay engine (see the
  entry below; the 2026-08-17 decision supersedes the D3 typed-refusal
  default there — rerun/set_only keep refusing typed). Each
  Selection-targeted do() appends an audit record (query repr + resolve
  digest + per-site relations) to `trace.intervention_audit` (persisted as
  of tlspec v8 with its load-validated digest relation; the tlspec v9 write
  admits the shipped per-site `source` disclosure, the PARAM row kind and
  PARAM-shaped recipes, and the EVENT transaction-envelope row kind with
  its optional hash-chain extension -- pre-v9 readers refused a saved
  selection- or param-intervened artifact's own load).

**tl.patch_from(source_trace, *, source_site=None, positions=None, position_axis=1)** — *unstable — no deprecation shim owed*
: Edit factory patching targeted sites from another trace's recorded
  post-capture values (activation patching); with a Selection target only
  the selected elements are patched. With a facet or head target
  (`tl.head(h, "v")`, `tl.facet(name)`) the donor is the same facet view of
  the source's home-op value, so only that slice is copied (a single head of
  GPT-2's fused `c_attn`, not the whole output); `positions=` then indexes
  the facet slice's axes. Source-to-target mapping (Patchscopes,
  SelfIE): `source_site=` reads the donor at another site of the source trace
  (a label, pass-qualified on multi-pass layers, or a selector naming exactly
  one op; else `patch_source_site_unresolved`), and `positions=` (an int, a
  `{source_position: target_position}` dict, or a list of pairs; negatives
  count from the end) writes those source positions into those target
  positions along `position_axis`, leaving the other target positions
  unchanged; the prompts may differ in length and a donor batch of one
  broadcasts (`patch_positions_invalid`, `patch_positions_out_of_range`,
  `patch_positions_shape_mismatch`). Portability `opaque_audit`: persisted
  args carry source-trace IDENTITY plus the mapping (never the Trace, never
  tensors); values bind at do() time session-side; no executable-save path
  in v1.

**trace.graph.edges / edge substitution** — *unstable — no deprecation shim owed;
S2/S3-gated*
: `trace.graph.edges` returns the dataflow edge family (one `EdgeUseRecord` per
  parent→child occurrence; requires an `replayable` capture, else
  `edge_provenance_unavailable`). The canonical occurrence address is
  `(child_func_call_id, arg_kind, arg_path)`. Edge records lift as EDGE-kind
  selections (whole-edge granularity; `~` complements within the trace's
  edge family). `do(edge_selection, edit)` replaces the value CONSUMED on
  the edge — only the child's consumption changes; `parent.out` stays
  producer truth. Ships on the replay/push engine ONLY
  (`edge_intervention_engine_unsupported` otherwise; the rerun-engine design
  is an escalated named future). Storage fork: the substituted value lives
  in the `Op.edge_substitutions` store, persisted as of the tlspec v8 bump (+
  `Op.edge_replacement_stamps`, `FireRecord.edge_address`); capture truth
  (`saved_args`, `out_versions_by_child`, `parent.out`) is retained
  unmodified — the pre-edit snapshot that makes divergence decidable.
  Validation: every tier-(ii) entry must be corroborated (FireRecord +
  stamp, and the stored value must digest to the stamp's `value_digest` —
  `edge_substitution_stamp_mismatch` otherwise) else FAIL; corroborated
  children are RE-EXECUTED with the substituted value spliced at the
  address and must match (verdict `edge_intervention_boundary` — a
  different check, never no check). Commits are PASS-QUALIFIED: an edge
  into pass k of a multi-pass child edits exactly pass k (the FireRecord,
  store, and pushed cone all land on `child.label`, never the bare
  `layer_label` that names the last pass). Edits at one occurrence
  COMPOSE like replay hooks (a second edit derives from the substituted
  value; `scale(0.5)` twice is a quarter, every fire is kept), and every
  tier-(ii) entry — edge, param, region — is RE-SPLICED whenever the cone
  recomputes the child, so a second edge edit on another arg, an upstream
  act edit, or a param edit reaching the child never silently reverts it.
  FULL VALIDATION OF EDITED FORKS: the replay engine records the
  capture-time content digest of every site's out the first time it
  overwrites it (session-time, in the replay run context), and the
  argument-logging check compares each child's retained `saved_args`
  snapshot against that digest for replay-recomputed parents — the same
  exact bidirectional check, never skipped — so non-identity act, edge,
  and param forks validate against their hand-computed ground truth and
  the boundary check is reached in a passing validation; a recomputed
  parent with no recordable digest reads `unverified`
  (`replay_recomputed_parent_unattested`), never validated.
  PERSISTENCE: the edge carriers persist as of tlspec v8, so ordinary
  saves proceed. The `edge_intervention_save_unsupported` refusal survives
  as a schema-regression tripwire (re-firing at ALL four save levels,
  preceding `artifact_save_level_unsupported`, if the carrier policy ever
  drops again).

**parameter substitution (do over PARAM selections)** — *unstable — no
deprecation shim owed*
: `fork.do(tl.param(name, mask=None), edit)` applies the edit "as if" the
  parameter were changed, FOR REPLAY ONLY: the value each consuming op sees
  is substituted at its derived occurrence address, and the live
  `nn.Parameter` object is NEVER written (bit-identical before and after —
  pinned). Parameters are not in the dataflow edge family (they classify as
  `LiteralTensor` template components), so the occurrence addresses are
  DERIVED — `Param.used_by_ops` + template-component identity/barcode
  matching, FAIL-CLOSED (`param_substitution_occurrence_underivable` when
  any consumption cannot be addressed: nested container positions,
  released legacy captures, a bare pass-ambiguous multi-pass consumer
  spelling, or a consumer inventory omitting a pass). Recurrently reused
  parameters (tied weights, one module called at N passes of a
  recurrence-grouped layer) ARE substitutable: consumers are recorded and
  staged pass-qualified (`label:pass`), the edit lands at EVERY
  consumption (a parameter has one identity across passes), and the
  pass-qualified replay engine recomputes each pass faithfully. The
  substitution then drives the shipped edge-substitution
  engine: tier-(ii) `Op.edge_substitutions` entries marked
  `substitution_kind="param"`, edit-then-scatter masking over the param
  index space, one replay pass over all consumer origins (cone
  recomputation RE-SPLICES every tier-(ii) entry — param, region, AND
  edge kinds — so later pushes never silently revert any edit; a chained
  `do(tl.param(p), scale(0.5))` COMPOSES on the staged "as if" value,
  0.25, keeping both FireRecords), and the same validation boundary
  (`edge_intervention_boundary` — a different check, never no check;
  uncorroborated entries FAIL). Replay/push engine ONLY:
  rerun/set_only refuse `param_substitution_engine_unsupported`. The audit
  record (kind `PARAM`) discloses "substituted at consumption … live
  parameters unchanged" — the product is DERIVED-class, never a blessed
  VERIFIED reproduction of the original weights. BUFFER OPERANDS
  (documented-unstable): `do(tl.param("<buffer address>", mask), edit)` on a
  registered buffer (BatchNorm running stats, a rotary cache, recurrent state)
  substitutes the buffer's INITIAL version node, the value its consumers read,
  and replays the cone through the leaf-site path (needs an
  `replayable=True` capture); in-forward writes thread the edited
  value; the live buffer is never written.

**TapObserver.values(masked=True)** — *unstable — no deprecation shim owed*
: `tap(resolved_selection)` stores each firing site's mask on the
  `TapRecord`; `values(masked=True)` returns fresh masked copies (selected
  elements). `values()` stays exactly the shipped full-snapshot behavior.

**color_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The v1 encoding-channel value source on `Trace.draw` (L5 channel core). See
  the "color_by" entry above for semantics.

**size_by / scale (draw kwargs)** — *unstable — no deprecation shim owed; keyword-only*
: The wave-1 size encoding channel on `Trace.draw` (D4 default-applied: sqrt +
  conservative area-only mapping + typed refusal on rolled varying sources).
  See the "size_by" and "scale (size channel)" entries above for semantics.

**highlight (draw kwarg)** — *keyword-only*
: `Trace.draw(highlight=...)` draws a named highlight over the graph; `None` (the
  default) draws the plain graph. The one value today is `"interventions"`: the surgery
  lens marks every recorded intervention on its node (solid border = recorded fact, dashed
  = heuristic attribution), the caption carries the per-lane census, and a written figure
  gets the full census text beside it as `<path>.census.txt` (beside the collision-safe
  file a bare draw chose). A trace with no recorded intervention refuses
  `surgery_evidence_missing`; any other value refuses `draw_highlight_invalid`, naming the
  allowed values.

**ContactSheet (contact-sheet page)** — *documented-unstable*
: What `tl.viz.contact_sheet(view, page=...)` returns: the page's SVG text (a `str`, so
  every string use keeps working) carrying `page` (zero-based index) and `page_count`
  (pages in the full sheet at this `tiles_per_page`, at least 1; the "N of M" footer
  prints the same number).

**stack_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The wave-1 rank encoding channel on `Trace.draw` (stacking split (a):
  explicit + licensed auto on plain traces). The `True`/`"auto"` request form
  is itself an unstable spelling. See the "stack_by" entry above.

**show_redundant_args (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: Opt-out for the default-on checked suppression of redundant constructor-arg
  label rows (slate 8.7). See the "show_redundant_args" entry above.

**direction="rightleft" (draw value)** — *unstable — no deprecation shim owed*
: The fourth layout direction on `Trace.draw(direction=)` for the forward, backward and combined graphs: edges flow right to left (Graphviz `rankdir=RL`), beside
  `"bottomup"`, `"topdown"` and `"leftright"`. An unknown value refuses
  `visualization_direction_invalid` naming all four.

**path=None (draw default)** — *unstable — no deprecation shim owed*
: Outside a notebook a bare `Trace.draw()` writes a collision-safe file in the current directory --
  `modelgraph.<format>`, then `modelgraph-2.<format>`, `modelgraph-3.<format>` -- and always prints
  `TorchLens graph written to <path>`, so a second bare draw never overwrites the first picture.
  In a Jupyter kernel (the same detection as `tl.draw`) a bare `trace.draw()` writes no file and
  returns the `Drawing` with its rendered bytes (SVG unless `file_format=` says otherwise), which
  the notebook displays inline (JMT decision, Sun 1 Nov 2026, 3:00 PM PT). An explicit `path=` is
  always written exactly (overwriting), in a notebook too, and printed only on a verbose trace.
  `Bundle.draw` follows the same rules.

**file_format (draw kwarg)** — *unstable — no deprecation shim owed*
: The path's extension picks the format: `trace.draw("graph.png")` writes PNG, and
  `b.draw("graph.svg")` writes SVG. `file_format=` applies only to a path with no extension
  (`trace.draw("graph", file_format="svg")` writes `graph.svg`) and to a bare call; its default is
  `"pdf"` (`"svg"` for a notebook bare call and for `Bundle.draw(graph="diff")`). A `file_format`
  that disagrees with the path's extension refuses with `ArgumentConflictError`
  (`draw_format_mismatch`). Both draw doors share this rule (JMT decision, Sun 1 Nov 2026,
  3:00 PM PT); `VisualizationOptions.file_format` reaches `Trace.draw` only when set explicitly.

**show_module_paths / show_grad_values (draw kwargs, `graph="backward"`)** — *unstable — no deprecation shim owed; keyword-only*
: Opt-in label rows on `Trace.draw(graph="backward")`: `show_module_paths=True` adds `module <address>` to
  each backward node a module owns (`param <address>` on AccumulateGrad nodes); `show_grad_values=True`
  adds `grad norm <x>, absmax <y>` from the node's captured output gradient (`grad values not
  captured` when the capture kept none). Both default to `False`, leaving the default render
  unchanged.

**theme (draw kwarg, `graph="backward"`)** — *unstable — no deprecation shim owed; keyword-only*
: The skin preset on `Trace.draw(graph="backward")`, the same names the forward picture takes
  (`"torchlens"` default, `"paper"`, `"dark"`, `"colorblind"`, `"high_contrast"`): it sets the page
  colour, text ink, typography and the backward key's ink; backward node fills and gradient arrows
  keep their colours. On `Trace.draw` the skin also sets the role colours: `torchlens` and
  `colorblind` render the Okabe-Ito set (green input, vermillion output, blue weights), `paper`
  and `high_contrast` the legacy set (light green input, light red output, grey weights). The functional form with grouped options reads it from
  `VisualizationOptions.theme`.

**theme / show_legend (Bundle.draw kwargs)** — *unstable — no deprecation shim owed*
: `bundle.draw(path, theme="dark", show_legend=True)` draws the bundle picture like
  `Trace.draw`: the skin's page, ink and Helvetica face, `draw`'s node shapes and fills (oval for
  one function call, box for a leaf module, cylinder for a buffer; the weight fills for weights), and with
  `show_legend=True` the key under the header. Member lists appear only on nodes and arrows that
  not every member runs.

**Bundled render font** — *unstable — no deprecation shim owed*
: Every Graphviz render resolves the pinned `Helvetica` family to the bundled Liberation Sans 2.1.5
  (SIL Open Font License 1.1, shipped unmodified under `torchlens/_visualization/fonts/` with its
  license) through a fontconfig file layered over the caller's own (it maps `Helvetica` and
  `Arial`, the family Graphviz's PostScript alias table asks for), so text metrics no longer
  depend on which face a machine has installed. The emitted DOT keeps `fontname="Helvetica"`.

**Expansion manifest** — *unstable — no deprecation shim owed*
: The JSON file `<outpath>.expansion.json` (schema `torchlens_expansion_manifest_v1`) a `draw`
  render writes beside an over-ceiling overview: a floor-fallback plan still above the readable
  band, or the bounded top-level module overview served past the pathological pre-gate. One entry
  per collapsed unit: a stable `expansion_id`, `kind`, `module`, `members`, and the `draw()`
  keywords (`expand`) that open it. Disclosed by the `collapse_expansion_manifest` warning.

**render_vector_oversize (warning)** — *unstable — no deprecation shim owed*
: A forward SVG or PDF page past the 14,400 pt per-side vector page ceiling is written exactly as
  requested and warned with its size and the remedies (`collapse='auto'`, `module=`,
  `depth=`): warn and obey, never a refusal.

**"shape_summary" node_label_fields token** — *unstable — no deprecation shim owed*
: Selector token rendering `Layer.shape_summary` (the L1 across-pass shape
  summary) as a label row; skipped when the field is unset. The default label
  renders the summary automatically on rolled varying multi-pass nodes,
  directly after the title row.

**"dtype" node_label_fields token** — *unstable — no deprecation shim owed*
: Selector token rendering the node's output dtype as a label row with the
  framework namespace dropped (`"float16"`, `"bfloat16"`; Keras `show_dtype`
  style), for spotting where a mixed-precision graph changes dtype. Skipped
  when the node has no dtype.

**torchlens.preprocessing.imagenet_default()** — *unstable — no deprecation shim owed*
: The explicit opt-in to the TorchLens-authored ImageNet recipe (resize 256,
  center-crop 224, ImageNet mean/std), passed as
  `CaptureOptions(input_transform=torchlens.preprocessing.imagenet_default())`. A PIL image
  whose model resolves no preprocessing authority (torchvision weights
  handle, timm data config, HF image processor) refuses typed
  `preprocessing_unresolved` instead of receiving this recipe silently. The
  opted-in capture's `input_preprocessor` keeps `source="imagenet_default"`,
  `verified=False`, status `unverified_fallback`: never an authority.

**torchlens.preprocessing.attach_authority(model, authority)** — *unstable — no deprecation shim owed*
: Records a model's preprocessing authority on the model so discovery finds
  it: a torchvision model built with `weights=` keeps no handle to its
  weights, and the preset cannot be inferred from the class. Afterwards
  `resolve(model=model)`, PIL-input autorouting and `check_card` resolve the
  attached authority as model metadata. Accepts any recognized authority;
  an opaque callable refuses `preprocessing_attach_opaque`. Returns the model.

**torchlens.preprocessing.check_card(trace, *, model=None, authority=None, offline=None)** — *unstable — no deprecation shim owed*
: The on-request preprocessing verifier: `trace` is a `Trace` (its
  `input_preprocessor` is read) or a `ResolvedPreprocessing` record. It compares the
  preprocessing a capture recorded (`Trace.provenance.input_preprocessor`) with the model's known card (HF image
  processor, torchvision weights, timm or open_clip config, or a named
  `authority=`), returns the field-level `PreprocessingComparison`, and warns
  `preprocessing_card_mismatch` when a field disagrees. Nothing runs it
  automatically. An opaque record compares as unknown, never as a match.

**torchlens.bridge.pyreft.read(directory, *, repo=None, revision=None)** — *documented-unstable*
: Read a pyreft LoReFT artifact (the directory `ReftModel.save` writes: `config.json` plus
  `intkey_<key>.bin`, under the key's dotted or underscored name) into a
  `torchlens.bridge.pyreft.LoreftArtifact`; never imports pyreft or pyvene and never fits
  anything. Apply it with `fork.do(artifact.subspace(site, positions=mask), artifact.edit)`, which
  is `tl.subspace(tl.units(site, ...), R.T, origin=..., method="loreft")` with
  `tl.intervention.set_affine_coordinates(W, bias=b)`. pyreft trains `R` in float32 (orthonormal to about
  `1e-7`), so the subspace orthonormalises it once in float64 and the saved edit records
  `origin_sha256` (R as read), `basis_sha256` (rows applied), `orthonormalised=True` and
  `source_dtype="float32"`. Refuses `pyreft_artifact_invalid` (missing or malformed file) and
  `pyreft_artifact_unsupported` (not exactly one `LoreftIntervention`).

**torchlens.bridge.pyreft.LoreftArtifact** — *documented-unstable*
: Frozen record of one LoReFT intervention as read: `rotation` (`R`, `[d, k]`), `weight` (`W`,
  `[k, d]`), `bias` (`b`, `[k]`) in their saved dtypes; `layer`, `component`, `unit` and
  `units_per_example` as pyreft names the site (for example layer 8, `"block_output"`, `"pos"`,
  1); `rank`; `origin` (the subspace `origin=` text); `provenance` (repo, revision, each file's
  sha256, the weight file name, each tensor's source dtype, the float64 sha256 of `R`, `W`, `b`,
  and `orthonormality_max_abs`). `subspace(site, *, positions=None)` returns the selection
  (`site` a site label or the site from a trace, `trace[label]`; `positions` a bool mask over the
  site's axes except the trailing feature axis; `None` is every position; a non-bool mask refuses
  `pyreft_positions_invalid`). Given the site from a trace, its shape is checked at once: a
  trailing axis that is not the artifact's width (or a feature axis that is not last) refuses
  `pyreft_site_invalid`, and a mask that does not cover the other axes refuses
  `pyreft_positions_invalid`; a label is checked at `do()`. `edit` is the float64 `set_affine_coordinates`.

**torchlens.bridge.huggingface_hub.from_hub(repo_id, *, path_in_repo, revision, token, api)** — *unstable — no deprecation shim owed*
: Downloads the gzipped `.tlspec` tar that `push_to_hub` uploads, checks
  every tar member (no links, absolute or parent-escaping names, one
  top-level bundle), and loads it eagerly onto the CPU through `tl.load`'s default-deny
  loader. `push_to_hub` accepts only a `Trace` or `Bundle` and never uploads
  a pickle (`hub_push_artifact_unsupported`); its default `path_in_repo` is
  `torchlens_artifact.tar.gz`.

**call_kwargs (torchlens.bridge.pytorch_grad_cam.cam)** — *unstable — no deprecation shim owed*
: Mapping forwarded unchanged to the pytorch-grad-cam call (`aug_smooth`,
  `eigen_smooth`); an option given both here and as a keyword refuses
  `bridge_gradcam_call_option_conflict`.

**SummaryCallback (torchlens.callbacks.lightning)** — *unstable — no deprecation shim owed*
: Lightning callback printing `tl.summary` once per fit. Sample: the
  module's `example_input_array` (at fit start) else the first training batch
  (before its training step), whose forward input is `input_getter(batch)`,
  else `batch["x"]`, else `batch[0]`, else the batch (a mapping as keyword
  arguments). One extra forward in eval mode under no_grad; every module's
  training flag and the RNG state are restored. Rank zero only; the report
  stays on `.report`; a rejected sample warns `lightning_summary_skipped`.

**LayerProfilerCallback run_id / disclosure (torchlens.callbacks.lightning)** — *unstable — no deprecation shim owed*
: Every JSONL record carries `run_id` (fresh at each Lightning `setup`, or
  fixed by `LayerProfilerCallback(..., run_id=...)`) and `disclosure`
  (`PROFILE_DISCLOSURE`: one extra forward per profiled batch, eval mode,
  no_grad, training flags restored). A batch the forward cannot take is
  skipped with a `lightning_profile_skipped` warning; training continues.

**MLflowSink, CometSink (torchlens.trackers)** — *unstable — no deprecation shim owed*
: Forward-watch sinks over a run or experiment the user created (never
  started or ended by the sink). `MLflowSink(client=None, *, run_id=None)`
  takes the fluent module, an `MlflowClient` + `run_id`, or nothing; keeps the
  caller's step; raw histograms are exact JSON artifacts. `CometSink(experiment)`
  takes an `Experiment` or `OfflineExperiment`; scalars and text only (Comet
  re-bins histograms, so `raw_histogram` is not advertised). Both implement
  the `torchlens.trackers` sink protocol: `capabilities()` returns what the sink
  accepts (`MLflowSink`: `scalar`, `raw_histogram`, `text_manifest`,
  `run_metadata`; `CometSink` the same without `raw_histogram`);
  `emit_scalar(point)` logs one metric at the caller's step;
  `emit_text(point)` stores one text payload (an MLflow text artifact, a
  Comet text log tagged in its metadata); `emit_histogram(point)` stores
  exact counts and edges as one MLflow JSON artifact, while `CometSink`
  refuses it with `tracker_sink_capability_missing`; `flush()` has nothing
  to force (MLflow writes synchronously, Comet writes at `experiment.end()`);
  `close()` is idempotent and never ends the user's run or experiment.

**tl.export.mlflow / tl.export.aim parity keys** — *unstable — no deprecation shim owed*
: Both exporters log the per-layer table, the report and every honesty fact as
  a string tag (wandb-export parity) and return `table`, `report`, `tags`
  (and `artifacts` for MLflow) beside the summary metrics and `capture_honesty`.

**tl.export.xarray(trace)** — *unstable — no deprecation shim owed*
: Returns an `xarray.Dataset` (CHANGED, batch-2 buffer ruling, option c: it returned
  a bare `DataArray`; read today's array as `tl.export.xarray(trace)["out"]`).
  `out` is the NeuroidAssembly `DataArray` of saved non-buffer outs,
  `presentation` by `neuroid` with `layer` / `layer_label` / `neuroid_index`
  coords. Each saved buffer layer is its own variable named by its layer label:
  own dims `<label>_dim_<i>`, no presentation axis, values unchanged, attrs
  `torchlens_kind="module_state"`, `layer_label`, `buffer_address`; the Dataset's
  `attrs["module_state_variables"]` lists them. Save-all and partial traces both
  export buffers (the former buffer-layer refusal is removed);
  unsaved layers are listed in `attrs["skipped_unsaved_layers"]`. Leading-size
  disagreement refuses `xarray_presentation_mismatch`.

**watch(grain="op", allow_forward_wrap=True) (torchlens.trackers)** — *unstable — no deprecation shim owed*
: Op-grain (tier O) watching: per-op activations from the Route-B op-tier
  collector during ordinary training. Needs `signals=("activations",)`,
  `select=` (module path prefixes or a selector-algebra object) and
  `inputs=`; refuses `watch_forward_wrap_consent_required` without
  `allow_forward_wrap=True`, the consent to wrap the root module's forward.
  Attach calibrates the wrap (plain vs wrapped forward, RNG and buffers
  restored) into `session.collector.calibration` and discloses it in
  `plan_report()`, the `torchlens/meta/op_grain_calibration` and
  `torchlens/run/op_grain_overhead_median_x` rows and one
  `watch_op_grain_cost_disclosed` warning. `facets={op_label: splitter}`
  adds per-facet series under `<op label>/<facet>`. Unknown `grain=` refuses
  `watch_grain_invalid`.

**Watch callback environment configuration (TORCHLENS_WATCH_SIGNALS, _SELECT, _EVERY, _HIST_EVERY)** — *unstable — no deprecation shim owed*
: An explicitly constructed `HFTrainerWatchCallback` / `LightningWatchCallback`
  takes settings it was not given from these variables (configuration only;
  the environment never activates instrumentation) and prints each setting's
  source at attach through `print_fn=`: `[argument]`, `[env NAME]`,
  `[derived: logging_steps=N]` or `[default]`. Resolved values and sources stay
  on `callback.settings`; `hist_every=` is a constructor argument too. A
  malformed variable refuses `watch_env_config_invalid`.

**show_legend=None AUTO value** — *unstable — no deprecation shim owed*
: The tri-state AUTO value on the stable `show_legend` kwarg: no legend unless
  an encoding channel is active, then a channel-only disclosure legend; a
  `module=` focused draw also shows the key and a reading key (what `@name`,
  an op name's two counters, the shape and size line and call arguments mean).
  `True`/`False` keep their stable historical meanings.

**Encoding refusal codes** — *unstable — no deprecation shim owed*
: `encoding_source_invalid`, `encoding_value_invalid`,
  `encoding_callable_error`, `encoding_requires_dot_layout`,
  `size_by_rolled_varying`, `scale_requires_size_by`, `encoding_scale_invalid`,
  `stack_by_auto_underivable`, `stack_by_requires_unrolled`.

**trace.autograd.grad_fn_fire_timings** — *unstable — no deprecation shim owed*
: Live-trace-only per-fire backward timing spans, keyed like
  `trace.autograd.grad_fn_calls` (`"<grad_fn_label>:<call_index>"`). Both stamps of
  every span come from `time.perf_counter()`, paired at capture time by a
  per-node keyed LIFO (stale entries are discarded, never paired); untimed
  fires read `None`, never a false zero. Served from the runtime
  `GradFnFired` event stream, which never persists: loaded or cleaned traces
  refuse typed `grad_fn_fire_timing_unavailable`. As of the tlspec v8 bump
  the persisted `GradFnCall` timing fields carry the per-fire
  `perf_counter` pair, discriminated by `Trace.grad_fn_timing_provenance`.

**Trace.grad_fn_timing_provenance** — *unstable — no deprecation shim owed*
: The per-fire timing clock-provenance marker: `"unmeasured"` until a timing
  prehook arms, `"perf_counter"` on the universal path
  (`"perf_counter_grad_saved_only"` reserved for the D15 fallback mode).
  Persists as of tlspec v8, where the persisted `GradFnCall` stamps carry
  the per-fire `perf_counter` pair this marker discriminates; an untimed
  fire persists `(None, None)` and reads a `None` duration.

**Trace.checkpoint_invocation_witness** — *unstable — no deprecation shim owed*
: The projected checkpoint-invocation summary: token count, per-token pack
  counts / unpack window evidence / BACKWARD-DERIVED site-key candidates,
  degrade flags (`classifier_unavailable`, `patch_unavailable`,
  `exotic_subclass`, `unmatched_backward_warn`, `reentrant_node_discovered`,
  `unwitnessed_checkpoint_enter`), and an evidence-scoped completeness
  verdict — any flag withdraws the affirmative no-checkpoint claim. Tokens
  are minted only for classified non-reentrant `_checkpoint_hook` enters on
  the armed owner thread outside any engine invocation; reentrant
  checkpointing is token-free and affirmatively sentinel-flagged.
  Persists as of tlspec v8 with closed-vocabulary load validation. The
  typed checkpoint-ambiguity refusal awaits a pending contract amendment and is not yet
  shipped.

**trace.grad_fn_site_summary** — *unstable — no deprecation shim owed*
: The grouped-backward floor: a read-only per-`site_key` rollup of
  GradFn/GradFnCall facts (labels, fire counts, pass coverage, and live
  per-fire timing totals when evidence exists). Grad-fns without an op FK
  aggregate under the `None` key; op-backed grad-fns on a keyless legacy
  artifact refuse typed `site_key_unavailable`. Accessor-level only — no
  persisted fields.

**BackwardPass.aten_ops / DispatchedOp** — *unstable — no deprecation shim owed*
: The ATen calls one backward pass dispatched, recorded while the pass ran
  through a capture that requested a backward (`backward_ready=True`,
  `save_grads=` or `default_grad=`). Each `DispatchedOp` is value-free:
  `operator` (`"aten.mm.default"`), `input_shapes`/`input_dtypes` (one entry
  per positional argument, `None` for non-tensors), `output_shapes`/
  `output_dtypes`, `grad_fn_call_ref` (`(grad_fn_object_id, call_index,
  pass_index)` of the GradFn call it ran under, or `None`), `family`
  (`"matmul"`, `"convolution"`, `"zero_arithmetic"`, `"uncounted"`) and
  `flops` (fma2 for counted calls, `0` for layout calls, `None` outside the
  counted families). Session-time: `None` on a loaded trace or an implicit
  pass. Calls made inside TorchLens's own GradFn posthooks are not recorded;
  gradient-saving copies from its tensor hooks appear as zero-arithmetic calls.

**BackwardStatus.executed_flops / executed_flops_per_pass / uncounted_op_counts / convention** — *unstable — no deprecation shim owed*
: `tl.report.backward_status(trace)` in the `backward_observed` state counts
  the backward that ran over each pass's `aten_ops`: `executed_flops`
  is exact for the matmul and convolution families at fma2
  (`convention`, FlopCounterMode's convention), or `None` when a
  pass has no record on this object. `executed_flops_per_pass` pairs each
  pass index with its count; a `create_graph=True` gradient and the backward
  through it are separate passes. `uncounted_op_counts` names every recorded op
  kind outside the counted families with its call count, never zeroed or
  estimated. `grad_disabled` still reads `0` and `grad_enabled_unobserved`
  still reads unknown; the hypothetical `backward_estimate` is unchanged.

**BackwardPass.flops_attribution / BackwardFlopsAttribution / BackwardFlopsRow** — *unstable — no deprecation shim owed*
: One backward pass's executed FLOPs charged to the GradFn call each counted
  call ran under and to the forward op that node differentiates. A
  `BackwardFlopsAttribution` holds `rows` (one `BackwardFlopsRow` per GradFn
  call: `pass_index`, `grad_fn_label`, `call_index`, `op_label` or `None` for
  a node with no forward op, `flops`, `recompute_flops`), the
  `unattributed_flops` bucket with every unattributed call named in
  `unattributed_ops` (the seed gradient, nodes no capture walked), so rows
  plus bucket equal `executed_flops` exactly; `uncounted_ops`, `recompute`
  (`"labeled"`, `"none_observed"`, `"unknown"`), `other_trace_ops` (calls on
  another capture's nodes in a shared backward, left to it) and `source`.
  `flops_by_op()` sums rows per forward op. `None` when the pass has no
  census on this object; a pass that raised has `executed_flops` `None`.

**Persisted backward FLOPs record (`BackwardPass`)** — *unstable — no deprecation shim owed*
: The pass's counted executed backward and its attribution as a closed
  record (`convention`, `executed_flops`, `uncounted_ops`, `rows`,
  `unattributed`; ints, `None` and short strings only), written when the
  trace is saved so a loaded trace reads the counted answer through
  `flops_attribution` and `tl.report.backward_status`. The record is the
  private persisted field `_executed_flops_record` (stored key
  `_executed_flops_record` as of tlspec v11), not a public read; it is `None`
  on a live pass. A malformed record refuses at load with
  `artifact_executed_flops_record_invalid`.

**BackwardStatus.flops_attributions / unattributed_flops / recompute_flops** — *unstable — no deprecation shim owed*
: `tl.report.backward_status(trace)` in the `backward_observed` state: one
  `BackwardFlopsAttribution` per counted pass, the counted work that ran
  under no recorded node, and the part that replayed forward kernels under
  activation checkpointing (`None` when the evidence cannot separate replay
  from backward, as in a `create_graph` pass that checkpoints). Empty or
  `None` in the other two states. The record's constructor keyword is
  `flops_attributions=`.

**DispatchedOp.grad_enabled** — *unstable — no deprecation shim owed*
: Whether autograd grad mode was on when the call dispatched during a
  recorded backward. Backward kernels run with grad mode off unless the pass
  builds a graph; checkpoint replays run with it on, which is how a replayed
  forward kernel is labeled recompute.

**CostTreeRow forward/backward/total columns; CostTree.backward_state / total_backward_flops / total_forward_mac_flops / column_conventions** — *unstable — no deprecation shim owed*
: `tl.report.cost_tree(trace)` rows carry `self_forward_mac_flops` (2 x true
  MACs), `self_backward_flops` (executed backward, matmul and convolution
  families, fma2) and `self_forward_backward_flops` (their sum, one convention), with
  non-additive subtree twins `CostTreeRow.subtree_forward_mac_flops`,
  `CostTreeRow.subtree_backward_flops` and `CostTreeRow.subtree_forward_backward_flops`
  (constructor keywords `subtree_forward_mac_flops=`,
  `subtree_backward_flops=`, `subtree_forward_backward_flops=`). Backward work on nodes with no
  forward op and unattributed calls get their own remainder rows, so the self
  backward column sums to `total_backward_flops`. Without a counted backward the
  cells are `None` and `backward_state` names the state; never 0.
  `column_conventions` names each FLOPs column's convention. Siblings fold
  only when their MAC and backward cells agree too.

**until= (run kwarg) / report.truncated / report.stopped_at** — *unstable — no
deprecation shim owed*
: `trace.run(until=...)` executes only the dependency closure (or disclosed
  sequential prefix) needed to reach the named sites. Truncation is a
  RESULT/REPORT term, never a capture outcome: the report's flat surface is
  `report.truncated` (bool) and `report.stopped_at` (the stop-frontier site
  label), with the full `RunTruncation` disclosure on `report.truncation`
  (regime, requested sites, executed/skipped counts, skipped-site digest).
  Skipped sites are "not-run", never "passed". With `fast=True` on a live trace
  the native forward is cut right after the `until=` site, which must be a
  site the capture saved (a module address, `"address:call"`, or a saved op label;
  otherwise `run_capability_unavailable` at stage `fast_until_site_unsaved`): the
  prefix keeps the fast path's per-call guards and comes back graph-connected under
  the default save, the regime is `live_stop_after`, the output and the later saved
  sites read `None` for that run (the next full run refreshes them), and the verdict
  is `unverifiable` because the whole-forward call fingerprint cannot be compared.
  The fast run refreshes its result trace (`trace.run(..., fast=True)` on a live trace
  returns a new Trace and leaves the source untouched; JMT, Sun 1 Nov 2026), so that
  verdict marks the result trace as every truncated run product is marked (monotonic: save, export or validate it before
  training through `until=`).
  A loaded artifact's fast tier still refuses `run_fast_until_unsupported`. Full
  contract: [the runnable tlspec contract](runnable_tlspec_contract.md).

**Trace.sites()** — *unstable — no deprecation shim owed*
: The tabular view of a trace's structural sites: one row per distinct L1
  `site_key` in first-occurrence execution order, aggregating the ops that
  share the site (`module_site`, `layer_type`, `output_slot`,
  `call_ordinal`, `n_ops`, labels, passes, shapes). Requires the `tabular`
  extra (pandas); legacy artifacts without site keys refuse
  `site_key_unavailable`, never a silently empty table.

**Trace.bill_of_materials()** — *unstable — no deprecation shim owed*
: The inventory of what a trace actually contains: a nested, JSON-friendly
  dict of sections (`capture`, `graph`, `parameters`, `buffers`,
  `activations`, `backward`, `annotations`), every figure read from fields
  the trace already carries — the rollup mints no new claims.

**logit_lens (torchlens._semantic)** — *unstable — no deprecation shim owed*
: `torchlens._semantic.logit_lens(trace)` projects each block's
  residual-stream facet through the model's own final norm + unembedding,
  returning a `LogitLens` (per-layer logits, `summary()`,
  `top_tokens()`). The reconstructed lens is validated against the captured
  logits before use and refuses `LogitLensError` when the head cannot be
  represented; pass `lens=` for a tuned lens or `verify=False` to trust
  the reconstruction explicitly. With no language-model head, the default
  lens falls back to a vision head recipe, validated the same way:
  `classification_head` (final norm plus `classifier` over the `[CLS]`
  token; `lens_source="classifier_head"`) or CLIP's
  `contrastive_image_head` (per-layer scores against the captured text
  features of the label prompts; `lens_source="contrastive_head"`). A user
  lens may carry `model_identity` (an extraction manifest's identity
  record): a differing measured identity refuses
  `logit_lens_identity_mismatch` unless `verify=False`; a lens with no
  comparable identity warns `logit_lens_identity_unrecorded` when its
  last-layer decode misses the captured logits. Its streaming sibling
  `logit_lens_predictions(trace, ...)` keeps only per-layer top-k, `logsumexp`
  and requested-token values; its keyword-only `mask_input=True` bans the
  model input's own token ids before those reductions (the trace's first input
  must be saved integer token ids, else `logit_lens_mask_input_unavailable`).
  Trajectory statistics over many batches: `tl.mechinterp.measure_trajectory`. See
  [facets](../facets.md).

**render_logit_lens_heatmap (torchlens.transformer)** — *unstable — no deprecation shim owed*
: `torchlens.viz.logit_lens(result, path)` draws a
  `LogitLens` as a layers-by-positions heatmap: one row per projected
  block, one column per input position, each cell printing that layer's
  top-1 next-token prediction and colored on a fixed 0-to-1 probability
  scale (the top-1 probability, or with `value="target_prob",
  target_token_id=...` a tracked token's). The footer states the lens
  source, whether it was validated, and that every row is projected through
  the final norm and head; only a validated lens lets it say that the last
  row reproduces the model's own output scores. `title=` prefixes the
  figure title with a subject such as the model's name. Lenses
  wider than 48 positions refuse `tv_record_invalid` until `positions=`
  names the columns; nothing is cropped silently. Returns a `SavedFigure`.

**kv-group receipts (torchlens.transformer)** — *unstable — no deprecation shim owed*
: `tviz.receipt_from_patch_grid(grid, ..., kv_groups=tviz.GqaInfo(...))`
  builds a `CausalReceipt` whose cells are grouped-query KV groups, for
  grids over a key or value site. The receipt's `kv_groups` field holds the
  grouping, `unit` reads `"kv group"`, and `sharing_lines()` names each
  group's query heads; `render_receipt_grid` labels the columns by group
  and prints that no cell is a per-query-head effect. `GqaInfo` gains
  `query_heads(group)` and `group_header(group)`. A grid whose head axis
  exceeds the group count refuses `tv_record_invalid`; mixing kv-group and
  query-head rows refuses `tv_receipt_invalid`; `Annotation.from_receipt`
  refuses a kv-group receipt `tv_annotation_invalid`. A grid from
  `mechinterp.patch_query_heads_grid` is an ordinary query-head receipt: its
  `effect_site` receipt becomes the receipt's `effect_source`, printed as
  "effect source: ..." on the grid figure, and passing `kv_groups=` for it
  refuses `tv_receipt_invalid`.

**Receipt cells and patch-grid baselines (torchlens.transformer, torchlens.mechinterp)** — *unstable — no deprecation shim owed*
: A `mechinterp.PatchGrid` from `mechinterp.patch_grid`, its presets or
  `mechinterp.patch_query_heads_grid` keeps `values` as the metric of each
  patched run (`to_pandas` names the column `metric`) and records the
  unpatched baselines as `receipts["clean_metric"]` and
  `receipts["corrupted_metric"]`. `tviz.receipt_from_patch_grid` turns a
  row into effects: patched metric minus `corrupted_metric`, the run every
  cell patches into, so a cell that changes nothing is exactly 0, the same
  quantity as the self-patch negative control. A grid without
  `corrupted_metric` refuses `tv_receipt_invalid`. The `CausalReceipt`
  carries the baseline as `baseline_metric` (`None` when a producer did not
  record one) and names its run in `baseline_run` ("unpatched corrupted
  run" for a patch-grid receipt); its `effects`, `joint_effect` and
  controls are all deltas from that run. The default `intervention` wording is "clean activation of
  one head patched into the corrupted run" ("one kv group" with
  `kv_groups=`); a row's `fires` is its own cell count when the grid's
  ledger shows one fire per cell. `tl.viz.ablation_grid` titles the
  figure "measured head effects on METRIC", labels the colorbar "change in
  METRIC", labels the x axis "query head" (or "kv group"), prints "each
  cell: change in METRIC from the unpatched corrupted run (B)" and says
  single-head effects need not be additive in the footer, refuses rows
  measured from different baselines (`tv_receipt_invalid`), and reserves the footer's height below the matrix
  and its tick labels.

**Summary columns (`summary(columns=...)`)** — *unstable — no deprecation shim owed*
: The column registry of `tl.summary` / `trace.summary`: `name`, `input`,
  `output`, `kernel_size`, `groups`, `params`, `params_pct`,
  `params_trainable`, `flops`, `flops_pct`, `macs`, `train`, `mode`,
  `passes`, `evidence`. `input` lists the shapes of the distinct tensors
  entering a row; `output` lists every member shape of the row's output in
  return order (past 60 characters a run of identical shapes prints once as
  `shape ×N`, and a list still too long ends in `... +K more`; `input` cells
  compact the same way); `kernel_size` / `groups` read the row's own conv or
  pooling op; `params_trainable` is the per-path trainable count (`train`
  stays the yes/no/partial `requires_grad` tri-state); `mode` is
  `module.training` as captured (`train` / `eval` / `mixed`); `flops_pct`
  is the row's share of the known forward FLOPs. A `+name` delta inserts
  at the column's registry position. A FLOP cell whose count includes a
  full-square attention op ends in `^` with a footer note. See
  [summary](summary.md).

**SummaryReport.table (`trace.summary(view="stats").table`)** — *unstable — no deprecation shim owed*
: The typed observation table (`StatsTable`) behind a `view="stats"` report: `rows`
  (`StatsTableRow` records with `state`, `value_stats`, `reason`, `parent_mark`), `ok_rows`,
  `sort_rows(by=..., descending=...)` (sorts on a sampled column are disclosed) and
  `fold_same_as_parent()` (opt-in fold of `= parent` rows with an exact count). The report's
  text is one rendering of it. A report built with any other view carries no table and
  refuses `summary_table_unavailable`.

**Summary Params footer extras** — *unstable — no deprecation shim owed*
: The Params footer line carries `size` (unique parameter storage, dtype
  true), a `pruned:` note with raw and effective nonzero counts on models
  reparametrized by `torch.nn.utils.prune` (`<name>_orig` + `<name>_mask`),
  and `optimizer state N params, X MB` when the capture was given
  `CaptureOptions(optimizer=...)` (read from the optimizer's live state;
  omitted on a loaded trace). The raw values ride `SummaryTotals.params_bytes`,
  `.params_effective`, `.optimizer_state_params`, and
  `.optimizer_state_bytes`: `params_bytes` is the unique parameters'
  storage, `params_effective` is set only on a pruned model (raw total minus
  mask zeros), and `optimizer_state_params` and `optimizer_state_bytes` are
  set only when the capture was given an optimizer; each is `None`
  otherwise.

**ParamFacts.nbytes / ParamFacts.pruned (`tl.report.PrunedParamFacts`)** — *unstable — no deprecation shim owed*
: FactCore parameter facts (schema version 2): `bytes` is the unique
  parameters' storage; `pruned` is `None` unless prune reparametrizations
  exist, else a `PrunedParamFacts(paths, raw, masked)` naming the pruned
  tensors, their raw element count, and the mask zeros (`masked=None` when
  the capture retained no mask values).

**FlopsReport.attention_full_square_ops / ProfileReport.attention_full_square_ops** — *unstable — no deprecation shim owed*
: The count (the `attention_full_square_ops` field) of `scaled_dot_product_attention` ops in the forward FLOP total
  of a `flops` or a `trace.profile()`; when non-zero the report or the
  profile footer prints that attention FLOPs count the full query x key
  square and masks / `is_causal` are not discounted. Every attention FLOP
  line carries the same note: the `tl.summary` FLOP cells (marked, with the
  note as a legend), the `explain` FLOPs line, and graph labels, where an
  attention node's `flops` label field or `flops` overlay row is followed by
  the note.

**`decode_style_unknown`** — *unstable — no deprecation shim owed*
: The typed refusal `tl.errors.DecodeStyleUnknownError` (an `InvalidArgumentError`
  subclass) for a `Trace.decode_output(style=...)` no built-in or registered custom output
  detector resolves, with a did-you-mean over the built-in styles (`classification`,
  `imagenet`, `imagenet1k`, `hf_text`). The same code refuses `style="hf_text"` when no
  rank-2 or rank-3 logits are retained. `style="hf_text"` without a `tokenizer=` is not a
  refusal: the rows carry the argmax `token_ids` and `text` is `None` (the trace was
  captured without a tokenizer). Output decoding is read-time only: the capture runs the
  automatic detector pass with no option, and `CaptureOptions` has no decode field;
  `CaptureOptions(output_style=...)`, `CaptureOptions(output_head=...)` and the same
  keywords on `tl.trace` refuse naming `trace.decode_output(...)`.

**`decode_head_unknown`** — *unstable — no deprecation shim owed*
: `Trace.decode_output(head=...)` named a head that is not one of the trace's output heads
  (the output container paths such as `"logits"`, `"0"`, `"0.logits"`); the refusal lists the
  available heads in `fields["available_heads"]`.

**`decode_head_ambiguous`** — *unstable — no deprecation shim owed*
: `Trace.decode_output(style=...)` found several qualifying output heads and none was named;
  pass `head=` one of the heads the refusal lists.

**tl.bridge.transformers.register_attention(model=None)** — *unstable — no deprecation shim owed*
: Registers TorchLens as a Hugging Face attention backend: one call puts a
  `"torchlens"` attention function in transformers' `AttentionInterface`
  AND its matching mask function in `AttentionMaskInterface` (the boolean
  `sdpa_mask` with the causal skip off), so the model always receives an
  explicit causal and padding mask. Registering the attention function alone
  is the mask trap: transformers then builds no mask and padding leaks into
  attention. Never registers over a built-in key; a foreign `"torchlens"`
  registration refuses `hf_attention_key_taken`. With `model=`, also calls
  `model.set_attn_implementation("torchlens")`. Returns the key, for
  `from_pretrained(..., attn_implementation="torchlens")`. Memory cost: the
  backend materializes the full `(batch, heads, q_len, kv_len)` probability
  matrix in every attention call, unlike fused SDPA or flash kernels.

**tl.bridge.transformers.record_attention(model, *, check_sdpa=False)** — *unstable — no deprecation shim owed*
: Context manager yielding an `AttentionRecording`: every attention call of
  a model on the `"torchlens"` backend records its per-head probabilities,
  keyed by the attention module's qualified name, without a TorchLens trace.
  The recorded tensor is exactly the one the backend multiplied by the
  values (and returns as `output_attentions`). Refuses
  `hf_attention_backend_not_selected` for a model on another backend.
  `check_sdpa=True` also runs `scaled_dot_product_attention` on each call
  and refuses `hf_attention_sdpa_mismatch` beyond the semantic SDPA
  reconstruction tolerance (doubles the attention cost; calls with dropout,
  softcapping or attention sinks are left unchecked and say so). Retained
  memory is `batch * heads * q_len * kv_len * itemsize` bytes per call: for
  a Llama-2-7B shape (32 layers, 32 heads) at 4096 tokens in bfloat16, 1 GiB
  per layer and 32 GiB per sequence.

**AttentionRecording / AttentionCall (`tl.bridge.transformers`)** — *unstable — no deprecation shim owed*
: `AttentionRecording` is a read-only mapping from attention-module name to
  probabilities `(batch, heads, q_len, kv_len)`; `recording[name]` serves a
  module called once and refuses `hf_attention_multiple_calls` otherwise,
  `calls_for(name)` returns every call in order, `.calls` lists all
  `AttentionCall` records (`module_name`, `probs`, `num_key_value_heads`,
  `mask_kind`, `sdpa_checked`), and `.nbytes` is the retained total.

**Report (`tl.report.Report`)** — *unstable — no deprecation shim owed*
: The one typed result of every report-shaped `tl.debug` helper:
  `report_type`, `subject` and `status` (the lead), `to_dict()` (a plain,
  JSON-friendly payload; tensors and graph records summarized, never
  inlined), `to_pandas()` (the table), and `str(report)` (a bounded card:
  lead line, at most five findings, the exact remainder, the exits). The
  typed results (`HealthReport`, `DeterminismReport`, `NonfiniteReport`,
  `FindNanReport`, `ParamHealthReport`, ...) subclass it; those with a house-voice
  `__repr__` (`HealthReport`, `ParamHealthReport`) keep printing it and serve
  the card as `report.card()`. The
  table helpers `compare`, `compare_params`, `dead_neurons`,
  `grad_health`, `hot_path` and `recompute_candidates` return a
  tabular `Report` whose `.to_pandas()` is the DataFrame they used to
  return (`attrs` included); the report has a table and never is one.
  Every finding that names an op names its user source line too: the
  `source_line` field or column holds `"<file>:<line>"` with the full path, and
  the card prints ` at <file>:<line>` relative to the working directory when
  the file is under it.

**Trace.profile(group_by="source_line") / ProfileReport.group_by** — *unstable — no deprecation shim owed*
: `group_by="source_line"` folds the op rows of `trace.profile()` into one
  row per user source line (`"<file>:<line>"`, the op's recorded call site),
  summing activation bytes, FLOPs and instrumented time; boundary rows fold
  into one `<inputs and outputs>` row, so every additive column sums to the
  ungrouped op view. Only with `level="op"`. `ProfileReport.group_by` records
  the key (`None` for one row per op).

**tl.summary("<hub model id>", input_shape=...)** — *unstable — no deprecation shim owed*
: A string model is a Hugging Face Hub id (or local config directory):
  `tl.summary` fetches the config only, builds the declared architecture on
  the meta device (no weights downloaded or allocated, `trust_remote_code`
  never set) and takes the weights-free summary path, so it needs
  `input_shape=` or meta inputs. Requires `transformers`.

**Trace.cost.forward_peak_memory / Trace.cost.backward_peak_memory**
: The largest memory growth above a bracket's starting baseline, including memory freed before the
  bracket ends. CPU measures only under `CaptureOptions(measure_host_peak_memory=True)`: it reads
  the process resident high-water mark, reset at each bracket on Linux (the reset also resets a
  caller's own `VmHWM` and `ru_maxrss` readings). Without the opt-in (the default) a CPU
  peak is `None` and its `*_peak_memory_reason` names the opt-in. CUDA reads the device allocator
  peak; MPS the allocator delta. `backward_peak_memory` is the largest peak above a pass's starting
  baseline over all backward passes (CUDA: the growth of the device high-water mark).

**measure_host_peak_memory (CaptureOptions)** — *unstable — no deprecation shim owed*
: Opt-in (default `False`) for the CPU `Trace.cost.forward_peak_memory` and
  `Trace.cost.backward_peak_memory`. Measuring resets the process-wide resident high-water mark at the
  forward bracket and at each CPU backward pass, which also resets the caller's own `VmHWM` and
  `ru_maxrss` readings and cannot be undone; the default writes nothing and reports CPU peaks as
  `None` with a reason. Mirrored on the trace as `Trace.measure_host_peak_memory`, a session-time
  knob (not persisted; load restores `False`); reruns and refreshes of a trace inherit it. CUDA
  peaks and the MPS allocator delta ignore it.

**Trace.cost.forward_peak_memory_reason** — *unstable — no deprecation shim owed*
: Why `Trace.cost.forward_peak_memory` is `None`. A CPU capture without
  `CaptureOptions(measure_host_peak_memory=True)` records `None` with a reason naming that
  opt-in. A CPU forward that measured zero (resident high-water growth,
  end-of-forward RSS delta and, when opted in, the Python-allocation peak all 0) is recorded as
  `None` with a reason naming `CaptureOptions(measure_python_peak_memory=True)` (forks.13).
  `None` when a value was measured, and on artifacts that predate the field. Persisted
  (KEEP); a loaded reason beside a number refuses
  `artifact_forward_peak_memory_invalid`. CUDA and MPS keep their numeric semantics.

**Trace.backward_peak_memory_reason** — *unstable — no deprecation shim owed*
: Why `Trace.cost.backward_peak_memory` is `None`: a CPU backward on a capture without
  `CaptureOptions(measure_host_peak_memory=True)` records `None` with a reason naming that
  opt-in. `None` when a value was measured, before any backward, and on artifacts
  that predate the field. Persisted (KEEP); a loaded reason that is not a non-empty string, or
  sits beside a number, refuses `artifact_backward_peak_memory_invalid`. CUDA and MPS keep their
  numeric semantics.

**Trace.cost.total_backward_memory**
: Bytes of the gradient payloads the trace retains, counted once per storage: a payload that is a
  view charges its whole base storage, an expanded gradient its smaller storage.

**trace.ops counting (`len(trace.ops)`)** — *unstable — no deprecation shim owed*
: `len`, iteration, integer indexing, `keys()`/`values()`/`items()` and
  `to_pandas()` of `trace.ops` cover the counted ops (no
  input, output or buffer rows; sumfam F1). Label lookup still resolves every
  row: `trace.ops["input_1"]` returns the input row and `"input_1" in
  trace.ops` is true although the label is not in `trace.ops.keys()`. The
  every-row sequence (inputs, outputs and buffers included) is
  `trace.ops`; `len(trace)`, `trace[int]` and `trace[plane]` index it.
  Only `trace.ops` is the counted view: `input_ops`, `output_ops`,
  `saved_ops`, `saved_grad_ops`, `compute_ops` and `ops_with_params` list
  exactly their rows, boundary rows included.

**`inv_rms` facet (Hugging Face `Gemma2RMSNorm`)** — *unstable — no deprecation shim owed*
: The norm's single `rsqrt` op output, `1 / sqrt(mean(x**2) + eps)`, the
  per-position factor that multiplies the input before the `(1 + weight)`
  gain. `tl.facet("inv_rms")` names it in every Gemma-2 norm; an attribution
  graph holds it with `tl.attribution.build_graph(..., frozen=[tl.facet("pattern"),
  tl.facet("inv_rms")])`, which also holds every Gemma-2 attention pattern
  (`Gemma2Attention` serves the attention facets; `pattern` is the eager
  softmax after the score softcap).

**`AttributionGraph.to_neuronpedia` (export in Neuronpedia's attribution-graph schema)** — *unstable — no deprecation shim owed*
: `graph.to_neuronpedia(*, slug, model_id, tokenizer, prompt=None, node_influence_coverage=0.8,
  edge_influence_coverage=0.98, path=None)` returns the graph, pruned by circuit-tracer's
  influence rule, as a dict in Neuronpedia's published attribution-graph schema
  (and writes it to `path` when given). `slug` is the graph's name on
  Neuronpedia, `model_id` Neuronpedia's model id, `tokenizer` anything with
  `decode(ids) -> str`, `prompt` the prompt string (`None` decodes the input
  tokens), `node_influence_coverage` and `edge_influence_coverage` the shares of total influence
  kept. Refusal codes `attribution_graph_export_argument_invalid` and
  `attribution_graph_export_unsupported`.

**`trace.draw(color_by=graph)` with an `AttributionGraph`** — *unstable — no deprecation shim owed*
: `trace.draw(color_by=graph)` paints an
  attribution graph onto the trace it was built from: each site holding feature or error
  nodes gets the `abs_sum` total of its nodes' influence on the logits over every position
  (circuit-tracer's rule); a site-to-value mapping paints any other quantity, feature
  activations included. Refusal code `attribution_overlay_graph_mismatch`.

**`tl.attribution.graph.upload_to_neuronpedia` (publish an exported graph)** — *unstable — no deprecation shim owed*
: `upload_to_neuronpedia(exported, *, api_key)` publishes the dict `to_neuronpedia` returned
  (or the file it wrote) on neuronpedia.org under the account that owns `api_key`, through
  Neuronpedia's documented two-step upload, and returns the graph page's URL. Standard
  library only; the key goes in the `x-api-key` header and never into a message. Checked
  before any request: the export's structure (`attribution_graph_contract_invalid`), the
  `slug` and `model_id` characters and the file size (`neuronpedia_upload_argument_invalid`).
  HTTP failures raise `neuronpedia_upload_http_error`, answers without their fields
  `neuronpedia_upload_response_invalid`.

## Capture door and options: current spellings

**Capture keywords**
: `CaptureOptions` and the capture doors spell the capture knobs `seed=` (`CaptureOptions.seed`),
  `save_args=` (`CaptureOptions.save_args`), `save_source_context=`
  (`CaptureOptions.save_source_context`), `save_rng=` (`CaptureOptions.save_rng`),
  `save_transformed_output=` (`CaptureOptions.save_transformed_output`) and `input_transform=`
  (`CaptureOptions.input_transform`). The saved `Trace` keeps its stored field names
  in this release.

**Recording keywords**
: `tl.record(...)` and `Recorder(...)` take `narrate=` (a
  `tl.options.NarrationOptions`, with `NarrationOptions.as_dict()`,
  `NarrationOptions.from_values()` and `NarrationOptions.is_field_explicit()`);
  `save_raw_grads` is a `CaptureOptions` field (`capture=CaptureOptions(save_raw_grads=...)`
  on `tl.record`). `NarrationOptions.sites` (`sites=`)
  scopes the narration (`True` for every event, `"modules"` for structure lines only, or a
  live selector); `trace.narrate(select=...)` filters post hoc. `NarrationOptions.sink` receives the
  narrated lines, `NarrationOptions.max_lines` caps them, `NarrationOptions.on_error_only`
  narrates only when the forward raises, and `NarrationOptions.tail_on_error` is how many of the
  last selected events the sink receives on a forward exception.

**StorageOptions**
: `tl.options.StorageOptions` (also reachable as `torchlens._user_funcs.StorageOptions`) groups the
  streaming-save knobs: `StorageOptions.path` (`path=`),
  `StorageOptions.retain_in_memory`, `StorageOptions.on_save` (keyword
  `on_save=`),
  `StorageOptions.include_custom_attributes`, `StorageOptions.include_buffer_values`,
  `StorageOptions.as_dict()`, `StorageOptions.from_values()` and
  `StorageOptions.is_field_explicit()`.

**CaptureOutcome flags**
: `CaptureOutcome.is_partial`, `CaptureOutcome.is_derived` (`is_derived=`),
  `CaptureOutcome.is_recovered` (`is_recovered=`), `CaptureOutcome.num_committed_ops` and
  `CaptureOutcome.note`. `CaptureStatus.STOPPED` (value `"stopped"`) is the status of a capture
  stopped by `stop_after=` or `tl.recording.stop()`.

**CaptureOutcome failure and boundary fields**
: `CaptureOutcome.error_type` (qualified exception type name of a failed capture),
  `CaptureOutcome.boundary_kind` and `CaptureOutcome.boundary_label` (the stop-boundary event kind
  and op label), `CaptureOutcome.frontier_labels` (the halt-frontier labels). A failed capture
  names its phase, one of `CapturePhase.FORWARD`, `CapturePhase.FINALIZE`,
  `CapturePhase.POSTPROCESS` and `CapturePhase.TEARDOWN`, and its diagnostic origin, one of
  `FailureOrigin.USER_OP`, `FailureOrigin.TORCHLENS` and `FailureOrigin.INTERRUPT`.

**Recording state**
: `Recording.is_failed`, `Recording.is_stopped`,
  `Recording.stop_reason`, `Recording.stops_by_pass`,
  `Recording.records_in_pass(pass_index)`, `Recording.num_ops`,
  `Recording.num_committed_ops`, `Recording.num_passes`, `Recording.lazy_trace`,
  `Recording.is_recovered` and `Recording.backward(loss)`. `Recorder.__call__` runs one pass
  (`recorder(x)`) and `Recorder.backward(loss)` records its backward.
  `ActivationRecord.recorded_time` and `GradRecord.recorded_time` (`recorded_time=`) stamp when a
  record was taken.

**Recording types**
: `tl.recording.GradRecord` (`GradRecord.ctx`, `GradRecord.ram_payload`,
  `GradRecord.transformed_ram_payload`, `GradRecord.disk_payload`,
  `GradRecord.transformed_disk_payload`), `tl.recording.DryRun` (`DryRun.events`,
  `DryRun.contexts`, `DryRun.decisions`, `DryRun.predicate_failures`, `DryRun.filter(...)`,
  `DryRun.print_tree()`, `DryRun.timeline_html()`, `DryRun.to_pandas()`), `tl.recording.stop()` and `tl.recording.StopSignal`. `GradRecordContext.kind`
  names the record kind and `CaptureSpec.save_grad` keeps a payload attached to autograd.
  `CaptureSpec.save_out` keeps the tensor payload of an event, `CaptureSpec.save_metadata` keeps its
  non-payload metadata and `CaptureSpec.save_mode` picks the retention mode (`copy`, `reference`,
  `view` or `cpu_async`). The predicate input for a backward gradient event, `GradRecordContext`,
  carries `GradRecordContext.grad_fn_class_name`, `GradRecordContext.backward_call_index`,
  `GradRecordContext.grad_kind` (`grad_input` or `grad_output`), `GradRecordContext.grad_input_index`,
  `GradRecordContext.grad_output_index`, `GradRecordContext.module_stack`,
  `GradRecordContext.has_forward_op`, `GradRecordContext.has_op`, `GradRecordContext.order`,
  `GradRecordContext.event_index`, `GradRecordContext.tensor_device` and the read-only
  `GradRecordContext.effective_label` (the joined forward label when there is one, else the
  autograd-node label).
  `tl.types.TraceLike` is the read protocol shared by `Trace` and `Recording`
  (`TraceLike.op_events`, `TraceLike.backward_events`, `TraceLike.event_stream`,
  `TraceLike.sibling_events`, `TraceLike.activation_by_raw_label(raw_label)`,
  `TraceLike.activations_by_pass(pass_index)`, `TraceLike.activations_by_address(address)` and
  `TraceLike.activation_payloads_by_raw_label`).

**Backend registry**
: `tl.backends.spec(name)`, `tl.backends.specs()`, `tl.backends.register(spec)`,
  `tl.backends.unregister(name)` and `tl.backends.resolve(...)`. Each `BackendSpec` carries
  `maturity` and the `protocol` factory (see "Backends: the backend protocol, maturity and
  conformance").
  `BackendCapabilities.has_streaming` is the streaming-save capability flag (flag name
  `has_streaming`). `CaptureBackend.cleanup_stopped_forward_session` tears down a forward
  stopped early.

**Extraction and preprocessing names**
: `BatchEnvelope.num_rows` and `RaggedBatch.num_rows` (`num_rows=`), `TokenRows.num_tokens`,
  `Resolution.preprocessing` and `SiteMatrix.shaping` (`shaping=`).
  `tl.extract_dataset(..., output_transform=...)` and `tl.neuro.plan_extraction(...,
  output_transform=...)` take the output transform;
  `input_transform=` preprocesses inputs.

**Admin and debugging names**
: `tl.clear_cache()` (also `torchlens._user_funcs.clear_cache`) empties the capture cache;
  `tl.observers.register_tensor_connection` registers a tensor connection;
  `tl.debug.isolated_capture` runs a capture in isolation, with `tl.debug.preserved_rng_state()`
  and `tl.debug.clone_input_tree(...)` as its parts; `Trace.capture_backward`
  (a thin `Trace` method over `backends/torch/_trace_backward.py`) captures a backward pass for validation.
### Debugging tools and report results
**DTypeRangeReport (`tl.debug.DTypeRangeReport`)** — *unstable — no deprecation shim owed*
: Coverage-bounded dtype range audit result.
**clone_input_tree (`tl.debug.clone_input_tree`)** — *unstable — no deprecation shim owed*
: Return a detached clone of an input tree for one isolated run: tensors are detached and
  cloned (keeping `requires_grad`), dicts, lists and tuples are rebuilt as new containers, and
  other leaves pass through as the same object.
**dtype_range (`tl.debug.dtype_range`)** — *unstable — no deprecation shim owed*
: Audit saved activations for numeric-range and precision hazards.
**nonfinite_origin (`tl.debug.nonfinite_origin`)** — *unstable — no deprecation shim owed*
: One-call NaN forensics.
**param_health (`tl.debug.param_health`)** — *unstable — no deprecation shim owed*
: Audit parameter/buffer space for numeric-health findings.
**preserved_rng_state (`tl.debug.preserved_rng_state`)** — *unstable — no deprecation shim owed*
: Context manager that saves the Python, NumPy, torch CPU and initialized CUDA RNG states and
  restores them when the block exits, normally or by an exception.
**ExplainReport (`tl.report.ExplainReport`)** — *unstable — no deprecation shim owed*
: The plain-language report trace.explain() returns: a str of multi-section prose.
**FlopsCrossCheck (`tl.report.FlopsCrossCheck`)** — *unstable — no deprecation shim owed*
: Side-by-side dispatch-formula evidence (never an oracle).
**MfuReport (`tl.report.MfuReport`)** — *unstable — no deprecation shim owed*
: The mode-split time-normalized MFU facts.
**OverheadReport (`tl.report.OverheadReport`)** — *unstable — no deprecation shim owed*
: Measured capture costs on the USER'S model and host.
**RooflineReport (`tl.report.RooflineReport`)** — *unstable — no deprecation shim owed*
: The roofline map: rows, sum/sum aggregate, coverage by reason.
**measure_clean_step (`tl.report.measure_clean_step`)** — *unstable — no deprecation shim owed*
: Measure an UNINSTRUMENTED forward wall time (the MFU denominator).
**overhead (`tl.report.overhead`)** — *unstable — no deprecation shim owed*
: Measure requested capture tiers on the user's own model/host.
**profile (`tl.report.profile`)** — *unstable — no deprecation shim owed*
: Build a unified op, module, or invocation resource profile.
**DTypeRangeReport fields** — *unstable — no deprecation shim owed*
: `DTypeRangeReport.findings`, `DTypeRangeReport.num_audited_ops`, `DTypeRangeReport.num_ops`.
**HealthReport fields** — *unstable — no deprecation shim owed*
: `HealthReport.checks_run`, `HealthReport.findings`, `HealthReport.skipped`.
**MfuReport fields** — *unstable — no deprecation shim owed*
: `MfuReport.denominator`, `MfuReport.denominator_scope`, `MfuReport.ideal_seconds`, `MfuReport.mfu`, `MfuReport.mode_terms`, `MfuReport.peaks_source`, `MfuReport.step_seconds`.
**NonfiniteReport fields** — *unstable — no deprecation shim owed*
: `NonfiniteReport.amp_disclosure`, `NonfiniteReport.capture_status`, `NonfiniteReport.cost_tier`, `NonfiniteReport.coverage_basis`, `NonfiniteReport.defer_recipe`, `NonfiniteReport.has_nonfinite`, `NonfiniteReport.nonfinite_labels`, `NonfiniteReport.num_checked_ops`, `NonfiniteReport.num_unchecked_ops`, `NonfiniteReport.num_unexamined_ops`, `NonfiniteReport.origin_label`, `NonfiniteReport.phase`, `NonfiniteReport.scope_note`, `NonfiniteReport.tensor_role`, `NonfiniteReport.uncertainty_zone`.
**OverheadReport fields** — *unstable — no deprecation shim owed*
: `OverheadReport.num_forward_runs`, `OverheadReport.tiers`.
**ParamHealthReport fields** — *unstable — no deprecation shim owed*
: `ParamHealthReport.checks_run`, `ParamHealthReport.findings`, `ParamHealthReport.schema_version`, `ParamHealthReport.skipped`, `ParamHealthReport.to_json`.
**ProfileReport fields** — *unstable — no deprecation shim owed*
: `ProfileReport.capture_status`, `ProfileReport.capture_verification_reason`, `ProfileReport.column_additivity`, `ProfileReport.evidence_labels`, `ProfileReport.frame`, `ProfileReport.is_capture_verified`, `ProfileReport.partition_total`, `ProfileReport.tree`, `ProfileReport.unknown_events`.
**RooflineReport fields** — *unstable — no deprecation shim owed*
: `RooflineReport.aggregate_intensity`, `RooflineReport.coverage_by_reason`, `RooflineReport.excluded_cache_resident`, `RooflineReport.headline_compute_bound`, `RooflineReport.headline_memory_bound`, `RooflineReport.ridge_intensity`.
## Visuals and kits spellings
Spellings introduced by the visuals-and-kits rename (attribution, facets, the `tl.viz` and
`tl.neuro` homes, transformer views, statistics and bridges), one line each. The sections above
describe each surface in depth.
### Namespace names
| Spelling | Meaning |
| --- | --- |
| `tl.attribution.Attribution` | Container for provisional input-attribution results |
| `tl.bridge.transformers.trace_image` | Trace an image model with PIL input and resolved preprocessing |
| `tl.bridge.transformers.trace_multimodal` | Trace a Hugging Face multimodal model with dict-keyed input |
| `tl.bridge.transformers.trace_text` | Trace a Hugging Face language model with raw text input |
| `tl.errors.AttributionError` | Error raised for unsupported or invalid attribution requests |
| `tl.errors.MissingFacetError` | Raised when a declared facet needs a different capture to be available |
| `tl.experiment.GridCell` | One cell of a `SweepGrid`: coordinates, status, value, error and member name |
| `tl.experiment.SweepGrid` | The labeled result of `sweep_grid`: float64 `values` shaped by the axes (NaN for refused or failed cells) beside the Bundle |
| `tl.experiment.sweep_grid` | Sweep the product of named axes (sites, at most one axis of edits); one member per cell, baseline first |
| `tl.facets.FacetFunction` | Registered facet recipe metadata |
| `tl.facets.MissingGrad` | Sentinel for a known facet gradient that is not currently saved |
| `tl.facets.describe` | List registered facet function names and declared facets for a class name |
| `tl.facets.enable_transformer_lens_aliases` | Enable or disable opt-in TransformerLens-style facet aliases |
| `tl.facets.functions` | List registered facet functions, optionally filtered by target scope and class-name glob |
| `tl.mechinterp.HeadScores` | Per-head taxonomy scores |
| `tl.mechinterp.LogitLensPrediction` | Streaming per-layer prediction summary (one row per projected layer) |
| `tl.mechinterp.LogitLensPredictions` | Result of a streaming :func:`logit_lens_predictions` sweep |
| `tl.mechinterp.PatchedRun` | The receipted product of one lowered counterfactual |
| `tl.mechinterp.SavePlan` | A priced, composable retention plan (mikit D15) |
| `tl.mechinterp.patch_head` | Patch one head (or its pattern) on a fused model via exact lowering |
| `tl.mechinterp.patching` | Prebuilt activation and attribution patching helpers for facets |
| `tl.neuro.ExtractionPlan` | Immutable measured plan for one whole-model extraction sweep |
| `tl.neuro.PlanSite` | One requested site's measured plan row |
| `tl.neuro.export_npz` | Export extracted features as npz interchange (D-21) |
| `tl.neuro.plan_extraction` | Build a measured extraction plan from two real probe captures (D-15) |
| `tl.recording.ModuleStackFrame` | One module frame in a predicate record context stack |
| `tl.recording.cleanup_partial` | Remove partial recording bundles and sibling temp directories |
| `tl.recording.dry_run` | Run predicates over one forward pass without retaining tensor payloads |
| `tl.recording.recover` | Recover a finalized or partial recording bundle |
| `tl.stats.RunningStats` | Always-on scalar spine accumulator (StreamingStat; D8) |
| `tl.stats.TensorStats` | Frozen tensor statistics shared by tensor displays and narration; per-family evidence owns every claim |
| `tl.stats.RunningStatsResult` | Finalized spine statistics for one population |
| `tl.viz.Drawing` | The picture every draw returns: DOT source (or animation HTML), format, written path; `save(path, editable=)`, `show()`, `close()` (releases a retained trace; idempotent; the `with` block exit calls it); a forward drawing also carries `collapse_plan` and `collapse_schedule` |
| `Trace.animate` | `trace.animate(site, path=None, *, open_viewer=False)`: the HTML animation of the repeated ops at `site` (a site label, pass-qualified label or record) as a `Drawing`; the one animation door (the former `animate_ops`), not a `draw` keyword |
| `tl.stats.NarrationObserver` | Live stats-line echo mounted on the shipped observer surface |
| `tl.stats.RunningStatsResult` | Finalized spine statistics for one population |
| `tl.stats.narrate` | Create an echo observer (the live line density) for one site |
| `tl.trackers.HistoryView` | Session-side read view over one run's committed history |
| `tl.trackers.SeriesPoint` | One (step-span, block-truth, observation) sample of a series |
| `tl.viz.NodeSpec` | Graphviz node attributes produced by TorchLens before user customization |
| `tl.viz.attention_atlas` | Render the whole-model attention atlas (BertViz model view, matched) |
| `tl.viz.contact_sheet` | Render one page of the ranked overview contact sheet |
| `tl.viz.feature_maps` | Compute and store per-site spatial activation maps for graph nodes |
| `tl.viz.histogram_waterfall` | Render the histogram waterfall (step x signed-log2-bin density) |
| `tl.viz.image_scatter` | Render a bounded 2D scatter with optional PIL thumbnails |
| `tl.viz.intervention_diff` | Diff ``subject`` against ``reference``, site-key-first |
| `tl.viz.lineplot` | Render one or more series as a compact PIL line plot |
| `tl.viz.metric_strip` | Render a loss/entropy strip to paper output |
| `tl.viz.nodes.channel_grid` | Create a grid visualizer for the first ``n`` activation channels |
| `tl.viz.nodes.heatmap` | Create a channel-averaged 2D activation heatmap visualizer |
| `tl.viz.nodes.histogram` | Create a compact value-distribution histogram visualizer |
| `tl.viz.nodes.mds_scatter` | Return a draw-time node callback for stored MDS scatter annotations |
| `tl.viz.nodes.scree` | Return a draw-time node callback for stored scree annotations |
| `tl.viz.nodes.text_table` | Render a text batch as a Graphviz HTML-like table label |
| `tl.viz.prediction_table` | Render the top-k table to paginated paper pages (whole positions) |
| `tl.viz.quantile_fan` | Render the quantile-fan strip for one series (the default view) |
| `tl.viz.site_detail` | Render the per-site detail sheet: every stream on one step axis |
| `tl.viz.timeline` | Render one timeline v2 artifact as deterministic standalone SVG text |
| `tl.viz.token_strip` | Render the paper-format token strip via matplotlib |
### Fields and methods
| Spelling | Meaning |
| --- | --- |
| `AbsenceReason.detail` | Human-readable explanation of the absence |
| `AbsenceReason.remedy` | Optional capture instruction for actionable missing-payload cases |
| `Attribution.target_repr` | Compact representation of the scalarization target |
| `AttributionGraph.capture_status` | The capture's completeness verdict, as the trace reports it |
| `AttributionGraph.frozen_sites` | Labels of the sites held at their captured values |
| `AttributionGraph.has_demeaned_logits` | Whether a logit node's value is its logit minus the mean logit |
| `CausalAuditSet.build_interventions` | The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect`` |
| `CausalAuditSet.draw_index` | Index of the random draw (``0`` for ``"top"`` and ``"bottom"``) |
| `CausalAuditUnit.build_interventions` | The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect`` |
| `ComponentStack.topk` | Return the ``k`` largest rows as ``(label, score)`` pairs |
| `ContributionScores.topk` | Return the ``k`` largest contributors at one (position, direction) |
| `CurveStep.build_inputs` | The perturbed inputs of this step, shaped like the caller's ``inputs`` |
| `CurveStep.build_interventions` | The recorded edits of this step, as ``(selection, edit)`` pairs |
| `Drawing.collapse_plan` | The collapse plan a forward draw with `collapse="auto"`, `"max"` or a float level rendered from, computed against the drawing's resolved render context (property); refuses typed (`collapse_plan_unavailable`) for `collapse="none"`, a backward or combined graph, an animation, or a released trace |
| `Drawing.collapse_schedule` | The monotone float collapse schedule of the drawn forward graph (level 0.0 is the full graph, level 1.0 the maximum collapse) under the drawing's resolved render context (property); refuses typed for non-forward drawings and a released trace |
| `Drawing.file_format` | The rendered file format (``"pdf"``, ``"svg"``, ...) |
| `Drawing.receipt` | Execution receipt from the pinned capture policy |
| `ExtractionPlan.batch_size` | field of `ExtractionPlan` |
| `ExtractionPlan.included_sites` | Return the sites in the run's save set |
| `ExtractionPlan.input_signature` | field of `ExtractionPlan` |
| `ExtractionPlan.memory_budget_bytes` | field of `ExtractionPlan` |
| `ExtractionPlan.model_class` | field of `ExtractionPlan` |
| `ExtractionPlan.num_stimuli` | field of `ExtractionPlan` |
| `ExtractionPlan.peak_basis` | field of `ExtractionPlan` |
| `ExtractionPlan.planned_passes` | Return exact planned forwards: warm-up + two probes (already run) + batches |
| `ExtractionPlan.predicted_peak_bytes` | field of `ExtractionPlan` |
| `ExtractionPlan.probe_batch_sizes` | field of `ExtractionPlan` |
| `ExtractionPlan.probe_peak_pairs` | field of `ExtractionPlan` |
| `ExtractionPlan.retained_bytes_per_batch` | Return fitted retained bytes for one batch across included sites |
| `ExtractionPlan.safety_margin` | field of `ExtractionPlan` |
| `ExtractionPlan.total_output_bytes` | Return fitted final artifact bytes across all stimuli |
| `ExtractionPlan.transform_repr` | field of `ExtractionPlan` |
| `FacetFunction.class_names` | Short class names matched by this recipe |
| `FacetFunction.has_predicate` | Whether this recipe has an arbitrary predicate matcher |
| `FacetFunction.qualnames` | Fully-qualified class names matched by this recipe |
| `FacetFunction.target_scope` | Record scope the recipe targets |
| `FacetRegistrySnapshot.functions` | Immutable active recipe entries |
| `FacetRegistrySnapshot.provenance_id` | Stable digest of the recipe identities and their source tier |
| `FacetRegistrySnapshot.version` | Monotonic global registry version at snapshot construction time |
| `FacetSpec.apply_transforms` | Apply the transform chain to a value |
| `FacetSpec.capability_class` | Weakest capability class of the transform chain |
| `FacetSpec.capability_flags` | Weakest-link capability flags of the transform chain |
| `FacetSpec.conflict_alias_group` | Shared-home conflict and alias group for intervention |
| `FacetSpec.function_id` | Recipe identifier that produced this facet |
| `FacetSpec.function_version` | Recipe version that produced this facet |
| `FacetSpec.home` | Runtime home object; intentionally not portable |
| `FacetSpec.home_address` | Stable home address when available |
| `FacetSpec.home_kind` | Kind of home value |
| `FacetSpec.home_label` | Stable home label when available |
| `FacetSpec.output_path` | Structural output path inside a container output |
| `FacetSpec.scatter_update` | Scatter an edited facet slice back into the full home-op output |
| `FacetSpec.split` | Return specs for equal sections along a dimension |
| `FacetSpec.transpose` | Append a transpose primitive |
| `FacetSpec.value_version` | Which home value version the spec reads |
| `FacetSpec.write_mask` | Return a boolean mask of home positions written by this facet |
| `FacetView.function_source` | Return recipe name provenance for this view |
| `FacetView.get` | Return an available facet value, or a default when it is unavailable now |
| `FacetView.has` | Whether a facet is available in the current trace |
| `FacetView.invalidate` | Clear cached computed facet values |
| `FamilyEvidence.policy` | `exact`, `sampled`, `unavailable`, or nonfinite presence-only `verdict`; unavailable evidence carries a reason |
| `FamilyEvidence.is_sampled` | Whether this family rode the seeded gathered sample |
| `GraphNode.token_id` | Token id: the input token at ``position`` (token nodes) or the target vocabulary id (logit nodes) |
| `HeadScores.eligible_rows` | field of `HeadScores` |
| `HeadScores.module_addresses` | field of `HeadScores` |
| `HeadScores.topk` | Return the ``k`` highest-scoring heads as ``(address.headN, score)`` |
| `HeadWeightViews.num_kv_heads` | field of `HeadWeightViews` |
| `HistoryView.run_record` | field of `HistoryView` |
| `InputProvenance.has_semantic_values` | True only when the values are caller-supplied real data. Gates derived-semantics claims |
| `LogitLens.entries` | Per-layer projections in block execution order |
| `LogitLens.final_logits` | The model's captured real output logits when readable, else ``None`` |
| `LogitLens.is_validated` | Whether the reconstructed lens was numerically validated against the captured last-block ``resid_post`` -> logits pair |
| `LogitLens.stacked` | Return all per-layer logits stacked on a new leading layer axis |
| `LogitLensPrediction.token_ids` | Requested-token ids as given; empty when no tokens were requested |
| `LogitLensPrediction.token_logits` | Requested-token logits, shaped `[batch, n_positions, n_tokens]` |
| `LogitLensPrediction.token_probs` | Requested-token probabilities, shaped `[batch, n_positions, n_tokens]` |
| `LogitLensPrediction.token_ranks` | One-based ranks of the requested tokens as int64; tie convention recorded on the parent result |
| `LogitLensPrediction.top_ids` | Top-k token ids, shaped `[batch, n_positions, k]` |
| `LogitLensPrediction.top_logits` | Top-k logit values, shaped `[batch, n_positions, k]` |
| `LogitLensPredictions.is_validated` | Whether the reconstructed lens passed numeric validation |
| `MissingGrad.raise_error` | Raise the missing-gradient tensor-use error |
| `NarrationObserver.lines` | field of `NarrationObserver` |
| `NarrationObserver.signal` | property of `NarrationObserver`, derived from its `site` (`"backward"` for a `tl.grad(...)` site); not a constructor keyword |
| `NarrationObserver.sink` | field of `NarrationObserver` |
| `NarrationObserver.style` | field of `NarrationObserver` |
| `PatchedRun.metric_value` | field of `PatchedRun` |
| `PatchedRun.read_facet` | Read a facet from the patched rerun, refusing stale virtual reads |
| `PatchedRun.receipt` | field of `PatchedRun` |
| `PlanSite.bytes_at_batch` | Return fitted retained bytes for one batch of ``batch_size`` |
| `PlanSite.bytes_fixed` | Batch-independent byte intercept from the two-point linear fit |
| `PlanSite.bytes_per_stimulus` | Byte slope per stimulus from the two-point fit |
| `PlanSite.exclusion_reason` | Actionable reason when excluded; never silently dropped |
| `PlanSite.is_batch_invariant` | Whether the site's bytes did NOT scale with batch |
| `PlanSite.is_included` | Whether the site is in the run's save set |
| `PlanSite.transform_seconds` | Measured live transform time at the probe batch (D-15: the probe executes the transform chain; post-transform bytes and time are measured, never derived) |
| `PlanSite.transformed_bytes` | Post-transform bytes at the probe batch, when a transform ran |
| `PlanSite.transformed_shape` | Post-transform shape at the probe batch, when a transform ran |
| `ProjectionSource.has_fused_qkv` | Whether the role is a column slice of a fused QKV weight |
| `RandomizationTest.module_addresses` | Randomized modules, top (output side) first |
| `ReadRow.is_detached` | ``True`` when ``value`` is a detached copy (always true in v1 for carried tensors) |
| `ReadRow.is_frozen_requested` | field of `ReadRow` |
| `ReadRow.is_frozen_resolved` | field of `ReadRow` |
| `ReadRow.is_rescorable` | ``False`` on loaded tables (the runtime registry is not serialized) |
| `ReadTable.is_rescorable` | Whether the table can be re-scored against its live trace |
| `RunningStats.compute` | Return the finalized spine statistics (`RunningStatsResult`) |
| `RunningStats.merge` | Fold another spine into this one |
| `RunningStatsResult.num_finite` | field of `RunningStatsResult` |
| `RunningStatsResult.num_nan` | field of `RunningStatsResult` |
| `RunningStatsResult.num_negative` | field of `RunningStatsResult` |
| `RunningStatsResult.num_neginf` | field of `RunningStatsResult` |
| `RunningStatsResult.num_posinf` | field of `RunningStatsResult` |
| `RunningStatsResult.numel` | field of `RunningStatsResult` |
| `RunningStatsResult.num_zero` | field of `RunningStatsResult` |
| `RunningStatsResult.finite_absmax` | field of `RunningStatsResult` |
| `RunningStatsResult.finite_max` | field of `RunningStatsResult` |
| `RunningStatsResult.finite_min` | field of `RunningStatsResult` |
| `RunningStatsResult.m2` | field of `RunningStatsResult` |
| `RunningStatsResult.reduction_dtype` | field of `RunningStatsResult` |
| `RunningStatsResult.sum_abs` | field of `RunningStatsResult` |
| `RunningStatsResult.sum_squares` | field of `RunningStatsResult` |
| `SavePlan.analyses` | field of `SavePlan` |
| `SavePlan.describe` | Return the full per-site table as text |
| `SavePlan.estimated_bytes` | Return the summed payload estimate for the named sites |
| `SavePlan.module_subtrees` | field of `SavePlan` |
| `SeriesPoint.is_coarsened` | field of `SeriesPoint` |
| `TensorStats.mean_evidence` | `FamilyEvidence` with policy, population, sample size, and reason for the mean |
| `TensorStats.nonfinite_evidence` | Exact census evidence, unavailable with a reason, or presence-only `verdict` with counts unknown |
| `TensorStats.finite_min` | Minimum finite value, or `None`; read `extrema_evidence` for its policy |
| `TensorStats.finite_max` | Maximum finite value, or `None`; read `extrema_evidence` for its policy |
| `TensorStats.mean` | Mean, or `None`; read `mean_evidence` for its policy |
| `TensorStats.numel` | Number of elements in the tensor |
| `TensorStats.nbytes` | Tensor payload bytes, or `None` when unknown |
| `TensorStats.has_no_finite_values` | Whether no element is finite (the fully poisoned case) |
| `TensorStats.is_all_false` | bool family: every element false (exact) |
| `TensorStats.is_all_true` | bool family: every element true (exact) |
| `TensorStats.is_all_zero` | Whether every element is exactly zero (exact claim) |
| `TensorStats.is_magnitude_based` | field of `TensorStats` |
| `TensorStats.num_finite` | Elements with finite values, or `None` when any nonfinite count is unknown |
| `TensorStats.num_nan` | Exact count, or `None` when unknown |
| `TensorStats.num_neginf` | Exact count, or `None` when unknown |
| `TensorStats.num_nonfinite` | Total nonfinite elements, or `None` when any count is unknown |
| `TensorStats.num_posinf` | Exact count, or `None` when unknown |
| `TensorStats.num_true` | field of `TensorStats` |
| `TensorStats.num_zero` | field of `TensorStats` |
| `TensorStats.policy` | Read-only family summary: `sampled` if any family is sampled, `reuse` if only reused facts are present, otherwise `exact` |
| `TensorStats.std` | field of `TensorStats` |
| `TensorStats.std_evidence` | field of `TensorStats` |
| `TensorStats.std_se` | field of `TensorStats` |
| `TokenAttribution.completeness` | ``attribution_sum`` / ``target_delta`` / ``completeness_residual`` / ``residual_rel`` / ``target_delta_abs`` |
| `TokenAttribution.convergence` | Grid-pair evidence: requested spelling, evaluated grids, residual and both stability norms per grid, and the stopping statement |
| `TokenAttribution.cost` | Logical/physical call accounting incl. the batching audit evidence |
| `TokenAttribution.display_tokens` | Display strings per raw token (marker cleanup only) |
| `TokenAttribution.footer_lines` | Build the mandatory disclosure footer |
| `TokenAttribution.offsets` | Character offsets per token, or ``None`` when the tokenizer cannot provide them (disclosed) |
| `TokenAttribution.payload` | Build the renderer-neutral payload (the tviz contract) |
| `TokenAttribution.raw_tokens` | Tokenizer tokens, never silently merged |
| `TokenAttribution.special_tokens_mask` | 1 where the token is a tokenizer special, or ``None`` |
| `TokenAttribution.target_repr` | Full target resolution disclosure incl. the decoded token |
| `TokenAttributionPayload.to_markdown` | Render a plain-text fallback table with the footer lines |
| `TrajectoryStats.compute_stats` | Return ``{layer: RunningStatsResult}`` for one statistic (count, min, max, mean, ...) |
| `TrajectoryStats.is_input_masked` | Whether each row's own input tokens were banned before reduction |
| `TrajectoryStats.module_addresses` | Block addresses in depth order (one row per layer) |
| `TrajectoryStats.num_examples` | Number of examples (batch rows) folded |
| `TrajectoryStats.num_positions` | Sequence length every batch shared |
| `VisualizationOptions.size_scale` | UNSTABLE size-channel scale transform (``"sqrt"``/``"linear"``); see ``Trace.draw`` |
| `VisualizationOptions.graph` | Which graph the picture draws: ``"forward"`` (default), ``"backward"`` or ``"combined"`` |
| `VisualizationOptions.rolled` | ``True`` draws each repeated module once (the rolled graph); ``False`` (default) draws every pass |
| `VisualizationOptions.open_viewer` | ``True`` opens the written picture in the system viewer; the default never does |
| `VisualizationOptions.file_format` | Format for a ``container_path`` with no extension (default ``"pdf"``); an extension on the path picks the format, and a disagreeing explicit value refuses ``draw_format_mismatch`` |
| `VisualizationOptions.animate` | A site label, pass-qualified label or record: the picture is the HTML animation of the repeated ops at that site (default ``None``) |

#### `tl.Recording`

| Spelling | Meaning |
| --- | --- |
| `Recording.add_grad_record` | Append one retained gradient record and update the gradient indexes |
| `Recording.by_address` | Map from module address to indexes of its retained records |
| `Recording.by_label` | Map from layer label to `(pass, record)` index pairs of its retained records |
| `Recording.by_pass` | Map from pass index to indexes of the retained records captured in that pass |
| `Recording.end_times` | Per-pass end timestamps of the recording |
| `Recording.enrich` | Return a new recording with the requested incremental enrichments, or all feasible ones |
| `Recording.error_repr` | Repr of the error that ended a failed recording, or `None` |
| `Recording.error_traceback` | Traceback text of the error that ended a failed recording, or `None` |
| `Recording.from_capture_events` | Build a lazy recording projection from a predicate capture session |
| `Recording.grad_by_grad_fn_label` | Map from gradient function label to indexes into `grad_records` |
| `Recording.grad_by_label` | Map from layer label to indexes into `grad_records` |
| `Recording.grad_by_pass` | Map from pass index to indexes into `grad_records` |
| `Recording.grad_records` | Retained gradient records, in capture order |
| `Recording.history_size` | Number of earlier events kept for predicate history at recording time |
| `Recording.last_event_func` | Function name of the last captured event on a failed recording, best effort |
| `Recording.last_event_input_meta` | Input metadata of the last captured event on a failed recording, best effort |
| `Recording.last_event_label` | Label of the last captured event on a failed recording, best effort |
| `Recording.last_event_source_line` | Source line of the last captured event on a failed recording, best effort |
| `Recording.last_successful_op_label` | Label of the last op committed before a failure, or `None` |
| `Recording.orphan_records` | Payload-free rows (raw label, label, payload reference) of the orphan island ops kept by `keep_orphans=True` |
| `Recording.raw_metadata` | Return payload-free raw metadata rows for every captured event, retained or not |
| `Recording.records` | Retained activation records, in capture order |
| `Recording.recovery_warnings` | Warnings collected while recovering this recording from disk |
| `Recording.start_times` | Per-pass start timestamps of the recording |

#### `tl.options.RecordingOptions`

| Spelling | Meaning |
| --- | --- |
| `RecordingOptions.default_grad` | Gradient capture decision for events the predicate does not decide; giving it installs gradient hooks for a later `Recording.backward()` |
| `RecordingOptions.default_module` | Capture decision for module events the predicate does not decide |
| `RecordingOptions.default_op` | Capture decision for operation events the predicate does not decide |
| `RecordingOptions.history_size` | Number of earlier events kept for predicate history; 0 to 1024 |
| `RecordingOptions.include_source_events` | Whether source events appear in the predicate lookback of recent ops |
| `RecordingOptions.max_predicate_failures` | Maximum number of predicate failures stored on the recording; further ones are counted as overflow |
| `RecordingOptions.on_forward_error` | Failed-forward handling: `"raise"` (default), `"attach_partial"` or `"return_partial"` |
| `RecordingOptions.on_predicate_error` | Predicate error handling: `"auto"`, `"accumulate"` or `"fail-fast"` |
| `RecordingOptions.postprocess` | Post-capture enrichment preset applied to the recording; `"none"` by default |

### Keywords
| Spelling | Meaning |
| --- | --- |
| `ce_loss_clean=` | field of `SpliceResult` |
| `ce_loss_spliced=` | field of `SpliceResult` |
| `extra_positional=` | Var-positional catch on `tl.trace` and `tl.record`: only the model and the inputs are positional; any further positional argument refuses typed `extra_positional_input_invalid` (`input_kwargs=` is keyword-only) |
| `axis_roles=` | field of `ReceptiveField` |
| `backward_pass_number=` | field of `FireRecord` |
| `bin_edges=` | field of `HistogramPoint` |
| `can_intervene=` | keyword of `BackendCapabilities` |
| `candidate_num_components=` | keyword of `tl.transforms.srp_fidelity_probe`: the candidate numbers of output components (projection widths k) to probe, one report row each |
| `comet_experiment=` | keyword of `tl.trackers.CometSink`: the Comet experiment to log to |
| `edit=` | field of `FireRecord`; keyword of `tl.attribution.causal_audit` (keyword-only; default `'zero'`): `"zero"` (default), `"mean"` (each unit's mean over the batch), or a number or tensor (the baseline activation, broadcast to the site); an EDGE table is patched, not ablated, and takes only the default; keyword of `tl.experiment.sweep_grid` (keyword-only; default `None`); keyword of `tl.experiment.sweep_sites` (keyword-only; required): REQUIRED. The edit applied at each candidate site (which ablation is sound is a live scientific argument — no default); keyword of `tl.intervention.keep_grad` (required): A forward `Edit` (`tl.scale(2.0)`, `tl.patch_from(source)`, ...) or a hook callable `fn(out, *, ctx)`; keyword of `tl.positions` (required): Any same-shape forward edit (`tl.zero_ablate()`, `tl.steer(...)`, `tl.patch_from(...)`, or a hook callable taking `(out, *, ctx)`) |
| `edit_name=` | field of `FireRecord` |
| `extra_facets=` | keyword of `tl.facets.snapshot` |
| `function_id=` | keyword of `FacetSpec.computed`, `FacetSpec.from_home`, `tl.facets.FacetSpec` |
| `function_version=` | keyword of `FacetSpec.computed`, `FacetSpec.from_home`, `tl.facets.FacetSpec` |
| `functions=` | Immutable active recipe entries |
| `grad_fn_label=` | field of `GradFnCall` |
| `has_fused_qkv=` | keyword of `tl.mechinterp.ProjectionSource` |
| `has_semantic_values=` | True only when the values are caller-supplied real data. Gates derived-semantics claims; constructor keyword of `tl.types.InputProvenance` (required): True only when the values are caller-supplied real data. Gates derived-semantics claims |
| `is_capture_complete=` | keyword of `ModuleInputSnapshot` |
| `is_compatible_with_append=` | field of `Edit` |
| `is_detected=` | field of `DoctorCheck` |
| `kl_divergence=` | field of `SpliceResult` |
| `module_pass_number=` | field of `RecordContext` |
| `num_components=` | keyword of `tl.repgeom.classical_mds`, `tl.stats.PCA` and `tl.transforms.srp`: the number of components (output dimensions) to return; constructor keyword of `tl.transforms.SrpSizing` (required): Number of output components: the sized projection width |
| `num_heads=` | keyword of `FacetSpec.heads`: number of attention heads to split into |
| `rolled=` | keyword of `Trace.draw` and `VisualizationOptions`: ``True`` for the rolled graph, ``False`` (default) for every pass; keyword of `Bundle.draw` (keyword-only; default `False`); keyword of `Trace.to_dagua_graph` (default `False`): Forwarded unchanged to `trace_to_dagua_graph`; keyword of `tl.experimental.dagua.trace_to_dagua_graph` (default `False`) |
| `open_viewer=` | keyword of `Trace.draw`, `tl.draw` and `VisualizationOptions`: ``True`` opens the written picture; the default never does; keyword of `Bundle.draw` (keyword-only; default `False`); keyword of `Trace.animate` (keyword-only; default `False`): `True` opens the animation in the system viewer; keyword of `tl.viz.bundle_diff` (keyword-only; default `False`): `True` opens the written picture in a viewer; the default never does |
| `size_scale=` | keyword of `Trace.draw` and `VisualizationOptions`: size-channel transform ``"sqrt"`` (default) or ``"linear"``; refuses without `size_by=` |
| `where=` | keyword of `Trace.summary`: regex string (searched in the row name, its module address and, for an op row, the module that ran it), TorchLens selector (``tl.in_module(...)``, ``tl.func(...)``; keeps rows owning a matched op) or ``row -> bool`` callable; presentation-only (whole-model totals preserved and coverage disclosed) |
| `graph=` | keyword of `Trace.draw` and `VisualizationOptions`: ``"forward"`` (default), ``"backward"`` or ``"combined"``; keyword of `Bundle.draw` (keyword-only; default `'forward'`); keyword of `tl.export.html` (keyword-only; default `'collapsed'`): `"collapsed"` (default) / `"flat"` / `"none"` |
| `editable=` | keyword of `Drawing.save`: ``True`` writes the Graphviz source instead of a rendered picture |
| `num_stimuli=` | keyword of `tl.neuro.plan_extraction` |
| `pass_number=` | field of `BackwardPass`, `DiffRow`, `GradRecordContext`, `ModuleStackFrame`, `RecordContext`, `SiteRow` |
| `query_slice=` | keyword of `AttentionView.crop_to`: which query positions to keep, as a slice |
| `sampling=` | field of `FireRecord` |
| `value_fn=` | keyword of `FacetSpec.computed` |

## Current spellings: namespaces, members and keywords
Public names whose spelling or home changed, and names new to the public surface, one line each.
The sections above describe each surface in depth.
### Namespace names
| Spelling | Meaning |
| --- | --- |
| `tl.backends.BackendProtocol` | The one contract every backend fills: capability table, capture, validate, reference output, host copy, realize-once |
| `tl.backends.FrameworkRequirement` | A backend's framework distributions, supported range, tested-through version, CI pin and install extra |
| `tl.backends.REFUSAL_CODES` | Closed registry of backend refusal codes (`exc.code` of `BackendUnsupportedError`) |
| `tl.backends.VALIDATION_REASONS` | Closed list of per-op validation decision reasons every backend's `validate` uses |
| `tl.clip_norm` | Neutral edit that rescales a tensor (on `tl.grad(site)`, the gradient) to at most `max_norm` in `norm_type` norm |
| `tl.debug.BackwardNanReport` | Result of `tl.debug.bisect_nan_backward`: the backward pass walked, the earliest provable non-finite birth and the full transition ledger |
| `tl.debug.BisectNanReport` | Result of `tl.debug.bisect_nan` |
| `tl.debug.BisectPrecisionReport` | Result of `tl.debug.bisect_precision` |
| `tl.debug.CheckFinding` | One finding recorded by a training-loop check |
| `tl.debug.CheckReport` | The findings a `tl.debug.ChecksSession` collected, as a report |
| `tl.debug.ChecksSession` | Capture-free training-loop checks bound to a model and optimizer; see `docs/reference/checks_kit.md` |
| `tl.debug.ClipLedger` | Ledger of gradient-clipping events across steps |
| `tl.debug.ClipLedgerSnapshot` | Point-in-time read of a `tl.debug.ClipLedger` |
| `tl.debug.GuessInputShapeReport` | Result of `tl.debug.guess_input_shape` |
| `tl.debug.LineageReport` | Result of `tl.debug.lineage` |
| `tl.debug.LivenessProbe` | Per-site recorder of zero-fraction and running maximum across fires; a statistic, never a verdict |
| `tl.debug.OptimizerFacts` | Adapter-read facts about one optimizer; `is_known` gates every derived band |
| `tl.debug.RangeProbe` | Opt-in live semantic range check per site |
| `tl.debug.ScaleLedger` | Ledger of grad-scaler scale and skipped-step history across steps |
| `tl.debug.ScaleLedgerSnapshot` | Point-in-time read of a `tl.debug.ScaleLedger` |
| `tl.debug.ScanRow` | Health facts for one unique tensor in a parameter scan; `aliases` keeps every other name bound to the same tensor |
| `tl.debug.Watchdog` | Death-spiral watchdog: warns when no optimizer step is ever accepted |
| `tl.debug.WatchdogSnapshot` | Point-in-time read of a `tl.debug.Watchdog` |
| `tl.debug.guess_input_shape` | Infer a synthetic input shape while leaving the caller's global RNG untouched |
| `tl.differential.DiffReport` | The settled two-capture differential: rows plus honest totals |
| `tl.differential.diff` | Settle two captures into the guarded differential projection |
| `tl.distributed.is_vetted_build` | Capability probe: does this torch build match a censused snapshot? |
| `tl.distributed.verify_membership_lineages` | Audit per-membership lineage evidence across rank cores |
| `tl.errors.AmbiguousAxisError` | Raised when a helper cannot infer a feature axis safely |
| `tl.errors.AmbiguousBackendError` | Raised when backend auto-resolution has multiple equal-priority matches |
| `tl.errors.AmbiguousBatchInputError` | Raised when `chunk_size` cannot infer which input leaf is batched |
| `tl.errors.AmbiguousCallError` | Raised when a module-level query has multiple executed calls |
| `tl.errors.AmbiguousLabelError` | Raised when `Bundle.at` finds a label in multiple accessors |
| `tl.errors.AmbiguousPassError` | Raised when a layer-level query has multiple executed passes |
| `tl.errors.AmbiguousTargetError` | Raised when a projective query has more than one reachable target |
| `tl.errors.AppendStreamingUnsupportedError` | Raised when append rerun would need to mutate active streamed activation blobs |
| `tl.errors.ArtifactTooNewError` | Raised when an artifact declares a schema newer than this runtime's ceiling |
| `tl.errors.ArtifactTooOldError` | Raised when an artifact predates the supported rehydration floor |
| `tl.errors.BundleComparisonError` | Raised when bundle members fail a gated operation's comparison gate |
| `tl.errors.CaptureGapError` | Reserved enforcement error for an unrepresented torch invocation |
| `tl.errors.CaptureGapWarning` | Shadow-mode report for a possible unrepresented torch invocation |
| `tl.errors.CaptureNotActiveError` | Raised when a capture-only operation is called outside an active capture |
| `tl.errors.DatasetExtractionResumeError` | Raised when a resumable extraction artifact cannot be safely continued |
| `tl.errors.DirectWriteSaveError` | Raised when executable spec save sees direct out writes |
| `tl.errors.InterventionWarning` | Base class for TorchLens intervention warnings |
| `tl.errors.LiveIndexWindowError` | Raised when a live-index query targets an unavailable raw label |
| `tl.errors.MLXValueUnavailableError` | Raised when MLX lazy evaluation makes a predicate value unavailable |
| `tl.errors.MergeConflictError` | The presented rank cores structurally contradict each other |
| `tl.errors.MergeInputError` | A merge input is unusable: not a rank capture, duplicate rank, missing distributed evidence, or an out-of-scope boundary |
| `tl.errors.MergedArtifactError` | Merged artifact save/load failure: schema, integrity, or tamper |
| `tl.errors.MergedSurfaceUnsupportedError` | A refused surface on a merged trace (run/export/validate/selectors) |
| `tl.errors.MergedTraceError` | Base class for merged-trace errors; `fields["code"]` is frozen |
| `tl.errors.NarrationConfigError` | Raised for `narrate=` configuration refusals at capture entry |
| `tl.errors.NarrationStatsError` | Raised when an armed echo stats rung refuses a value computation |
| `tl.errors.NoInfluencePathError` | Raised when two graph points have no directed influence path |
| `tl.errors.OpaqueCallableError` | Raised when an executable intervention save would require opaque code |
| `tl.errors.ParamEvidenceMissingError` | Cross-member parameter read refused: no immutable capture-time evidence |
| `tl.errors.ParamReleasedError` | Raised when a released Param cannot re-fetch its live model parameter |
| `tl.errors.ReceptiveFieldConfigurationError` | Raised when a receptive-field call is misconfigured by its caller |
| `tl.errors.ReceptiveFieldError` | Base class for receptive-field failures |
| `tl.errors.ReceptiveFieldUnavailableError` | Raised when a receptive-field result requires unavailable capture state |
| `tl.errors.ReceptiveFieldValidationError` | Raised when receptive-field cross-validation detects a violation |
| `tl.errors.RenderError` | Raised when a transformer picture cannot be honestly produced |
| `tl.errors.ReplayValueMismatchError` | A recomputed selected activation differed from its archived bytes |
| `tl.errors.SidecarError` | Raised for sidecar-family registration and payload refusals |
| `tl.errors.TensorSubclassUnsupportedError` | A strict Tensor subclass could not be constructed while a mode was active here |
| `tl.errors.TraceCaptureError` | Raised by every capture door when the forward raises a non-TorchLens exception; a `CaptureError` chained `from` the original (`exc.__cause__`, traceback intact, never mutated). `TraceCaptureError.partial_trace`: the `PartialTrace` captured before the failure (or `None`); `TraceCaptureError.origin`: the outcome's `FailureOrigin`. `fields['code']` is `forward_raised` (origin `user_op` or `unknown`) or `capture_internal_error` (origin `torchlens`); `fields` also carries `cause_type`, `last_captured_op` and `num_captured_ops`. TorchLens's own errors propagate unwrapped and carry `partial_trace` too |
| `tl.errors.TraceMutationWarning` | First root-log mutation; Trace mutators operate in place |
| `tl.errors.TransformContractError` | Raised when a transform spec, chain, or plan violates its contract |
| `tl.errors.UserTransformError` | Raised when activation_transform or grad_transform raises |
| `tl.experiment.sweep_sites` | Ablate every candidate site, one member per candidate, baseline first |
| `tl.export.describe_target` | Return one export target's registration metadata (tier row included) |
| `tl.export.list_targets` | Return the registered export-target names |
| `tl.export.register_target` | Register one export target through the public door |
| `tl.export.resolve_target` | Return one registered exporter callable; unknown names refuse teaching |
| `tl.export.unregister_target` | Remove one export-target registration |
| `tl.fingerprint.StructuralHashMismatchError` | Raised when a model's provisional structural hash differs from its pin |
| `tl.fingerprint.content` | Return a deterministic SHA-256 digest of nested tensor content |
| `tl.inputs` | Input preprocessor registry: teach `tl.trace` to accept a model's raw input; registration affects future traces only |
| `tl.inputs.register` | Decorator registering `fn(model, raw)` that returns the model input built from `raw`, or `None` to decline; keyed by `name=`, ordered by `priority=`, limited by `class_name=` |
| `tl.inputs.unregister` | Remove a registered input preprocessor by name; later traces no longer use it |
| `CheckReport.counters` | Event counters the checks session accumulated |
| `CheckReport.finalized_reason` | Why the report was finalized, such as scope exit |
| `CheckReport.ledgers` | The clip, scale and watchdog snapshots collected into the report |
| `ClipLedger.observe_step` | Record one accepted step's post-clip total norm, and the pre-clip norm when the magnitude tier is armed |
| `LivenessProbe.facts` | Per-site liveness statistics recorded so far |
| `LivenessProbe.hook_plan` | The `hooks=` mapping to pass to `CaptureOptions` |
| `RangeProbe.hook_plan` | The `hooks=` mapping to pass to `CaptureOptions` |
| `RangeProbe.observed_fires` | Per-label fire counts, so a site that never fired is visible |
| `Watchdog.on_accepted` | Reset on an accepted step and re-arm the single warning |
| `Watchdog.on_backward` | Record one backward pass; returns `True` exactly once, when the watchdog trips |
| `tl.grad` | Address the gradient of a site's output: `tl.grad("relu_3")`; builds a `GradSelector` |
| `tl.intervention.BundleComparisonError` | The same class as `tl.errors.BundleComparisonError` |
| `tl.intervention.Population` | A prepared, provenance-stamped donor population: `members`, `datums`, `origin`, `kind`, `content_digest`, `digest_kind`, `identity`, `eligible_indices(...)`, `mean(dim=None, ...)`, `std(dim=None, ...)`; built by `tl.intervention.population` |
| `tl.intervention.SamplingRecord` | One realized draw of a stochastic edit, exported beside `FireRecord` (a `FireRecord.sampling` entry): a frozen record with `verb`, `base_seed`, `derived_seed`, `share_draw`, `logical_firing_coordinate`, `eligible_count`, `population_count`, `population_identity`, `donor_group_id`, `draw_indices` and `realized_draw_digest`; the field rows below define each |
| `tl.intervention.label_contains` | Create a label-substring selector |
| `tl.intervention.label_matches` | Create a label regex-pattern selector |
| `tl.intervention.model_input` | Create a nested model-input path selector |
| `tl.intervention.model_output` | Create an output selector |
| `tl.intervention.population` | Build a `Population` from a tensor, a sequence of tensors or a sequence of Traces, with a required `origin=` provenance string |
| `tl.options.HtmlReportOptions` | Options for emitting the offline HTML report (thumbnail budgets, determinism, share-safe stripping) |
| `tl.outputs` | Output decoder registry: give a classifier's logits human-readable labels; registration affects future traces only |
| `tl.outputs.register` | Decorator registering `fn(outputs)` that returns class labels in logit order (a list, or a dict from class index), or `None` to decline; decodes at capture time and answers `decode_output(style=name)` |
| `tl.outputs.unregister` | Remove a registered output decoder by name; later traces no longer use it |
| `tl.receptive_field.GradReceptiveField` | Empirical receptive-field influence set measured by autograd |
| `tl.report.DiffReport` | The same class as `tl.differential.DiffReport` |
| `tl.report.InterventionValidationReport` | Five-axis intervention validation result |
| `tl.report.KernelProfileReport` | One door call: one execution, one trace, one joined result |
| `tl.report.MergeReport` | Graph-free diagnostic summary of how rank captures were merged |
| `tl.report.ValidationDiagnostic` | Structured non-failing diagnostic emitted by validation |
| `tl.report.ValidationFailure` | A structured description of the first surfaced validation mismatch |
| `tl.report.ValidationStatus` | Machine-readable replay-validation availability or result |
| `tl.report.compare_topology` | Compare two Traces structurally |
| `tl.report.diff` | The same function as `tl.differential.diff` |
| `tl.report.DoctorReport` | Environment and optional model compatibility report; records are constructed by TorchLens |
| `tl.report.DoctorCheck` | One diagnostic row; constructor keywords are outside the public contract |
| `tl.report.doctor` | Inspect the environment always, and a model/input tree when supplied; never executes the supplied model |
| `tl.report.kernel_profile` | Run one save-nothing capture under the owned session and join it |
| `tl.trackers.HistoryReader` | Read one artifact without loading the run or importing a model |
| `tl.trackers.correct_running_stats` | Apply the closed-form unscale correction to one spine record |
| `tl.trackers.corrected_grad_block` | Return `block` with every scaled gradient observation corrected |
| `tl.trackers.running_stats_scalars` | Derive the portable scalar statistics from one spine record |
| `tl.transformer.ANNOTATION_KINDS` | The closed set of transformer annotation kinds: `descriptive_head_score`, `additive_logit_contribution`, `screening_estimate`, `intervention_effect` |
| `tl.transformer.AttentionView` | One attention pattern in canonical coordinates |
| `tl.transformer.CropInfo` | Crop disclosure: cropping never renormalizes |
| `tl.transformer.EpisodeCoordinates` | Episode coordinates (step and position) carried by transformer view records |
| `tl.transformer.MaskInfo` | Mask provenance for an attention view |
| `tl.transformer.PredictionTable` | Top-k tokens per position |
| `tl.transformer.PredictionTrajectory` | Per-layer prediction trajectory record |
| `tl.transformer.ScoreDecomposition` | Term-complete score decomposition for one (head, dst, src) pair |
| `tl.transformer.TokenAxis` | One labeled token axis |
| `tl.transformer.TokenMetrics` | Per-token metric strip record |
| `tl.transformer.TokenScoreRow` | One labeled row of per-token scores (strip row) |
| `tl.transformer.TokenScores` | Colored-token strip record, single- or multi-row |
| `tl.transformer.attention_view` | Extract one layer's attention pattern as a typed view |
| `tl.transformer.attention_views` | Extract every (or the named) attention layer's view, in trace order |
| `tl.transformer.bertviz_tuple` | Return attention in the BertViz and Hugging Face `output_attentions` tuple format |
| `tl.transformer.circuitsvis_attention` | Hand the payload to circuitsvis's attention component (notebook only) |
| `tl.transformer.circuitsvis_payload` | Build the CircuitsVis attention payload from typed views |
| `tl.transformer.metric_strip` | Convert a metric record to a strip record for rendering |
| `tl.transformer.prediction_table` | Build the per-position top-k table from the native lens row |
| `tl.transformer.prediction_table_html` | Emit the zero-dependency HTML top-k table |
| `tl.transformer.prediction_trajectory` | Build the trajectory record from a streaming lens result |
| `tl.transformer.receipt_from_patch_grid` | Build a validated receipt from a mech-interp patch grid row |
| `tl.transformer.render_answer_trajectory` | Render the tracked answer token's probability + rank panels |
| `tl.transformer.render_neuron_card` | Render the selected-pair neuron card |
| `tl.transformer.render_prediction_ribbon` | Render the layer-prediction ribbon |
| `tl.transformer.score_decomposition` | Decompose one attention score into its per-dimension terms |
| `tl.transformer.token_metrics` | Compute the loss and entropy strips from ONE logits source |
| `tl.transformer.token_strip_html` | Emit the zero-dependency, self-contained HTML token strip |
| `tl.types.Accessor` | Generic dict-like accessor for ordered TorchLens log objects |
| `tl.types.ContainerLeafOccurrence` | One tensor occurrence at one container path |
| `tl.types.ContainerSpec` | Portable description of an output container seen during capture |
| `tl.types.ContractCheck` | One contract check recorded in a run report |
| `tl.types.InputSpec` | Explicit per-slot input override for the quickstart input grammar |
| `tl.types.MergedFinding` | One structured merge finding |
| `tl.types.NarrationEvent` | One immutable narration record |
| `tl.types.NumericAttestationStatus` | Result states for the optional saved-activation oracle |
| `tl.types.PathFaithfulness` | Honesty states reported by a completed sparse run |
| `tl.types.ReadinessReport` | Complete non-executing readiness report for a Trace provider |
| `tl.types.ReadinessStatus` | Overall non-executing runnable readiness states |
| `tl.types.Relationship` | The derived relationship between two Bundle members, returned by `Bundle.comparability` and `comparability_table` (persisted bundle data; values unchanged) |
| `tl.types.RunProvider` | Providers selected by the `Trace.run` contract |
| `tl.types.RunReport` | Honesty, state, resolution, and contract metadata for one run |
| `tl.types.RunnableCompatibility` | Producer/runtime versions recorded for readiness diagnostics |
| `tl.types.RunnableDiagnostic` | Structured diagnostic shared by preflight, readiness, and run reports |
| `torchlens._user_funcs.CaptureNotActiveError` | The same class as `tl.errors.CaptureNotActiveError` |
| `torchlens._user_funcs.UserTransformError` | The same class as `tl.errors.UserTransformError` |
### Members
| Spelling | Meaning |
| --- | --- |
| `Accessor.labels` | The lookup label of every record the collection lists, in order |
| `Accessor.count_by` | Count records by one or more record fields |
| `ModuleAccessor.uncalled` | Addresses of registered modules the captured pass never called |
| `Accessor.search` | Return records whose lookup key contains `needle` |
| `TraceOpAccessor.compute` | The members that are not input, output or buffer rows |
| `TraceOpAccessor.nonfinite` | The members whose output held NaN or Inf |
| `AmbiguousBackendError.default_remedy` | The remedy the error names when none is passed |
| `BackendCapabilities.can_intervene` | Whether live intervention capture is supported (was `interventions`) |
| `BackendCapabilities.has_validation_replay` | Whether the backend can perform replay validation |
| `BackendSpec.maturity` | The backend's launch label, `"preview"` or `"stable"` |
| `BackendSpec.protocol` | Lazy factory returning the backend's `BackendProtocol` object |
| `BackwardEstimate.is_freeze_pattern_captured` | Whether the parameter freeze pattern was captured |
| `BackwardEstimate.is_grad_disabled_at_capture` | Whether grad was disabled when the forward was captured |
| `BackwardEstimate.num_frozen_weight_ops` | Ops whose weight gradient is skipped because the weight is frozen |
| `BackwardEstimate.num_multiplied_ops` | Ops whose backward cost the estimate multiplies from the forward cost |
| `BackwardNanReport.amp_hint` | The GradScaler hint when every finding is an `inf` and no `grad_scale` was given; else `None` |
| `BackwardNanReport.backward_pass` | The one-based backward pass walked |
| `BackwardNanReport.birth` | The earliest provable birth transition (finite in, non-finite out), or `None` |
| `BackwardNanReport.checked_calls` | Calls with saved, checkable gradient payloads |
| `BackwardNanReport.first_bad_thing` | Project this result into the shared first-bad-thing vocabulary |
| `BackwardNanReport.found` | Whether any non-finite gradient was observed in the walked pass |
| `BackwardNanReport.healed` | Labels of transitions where a non-finite arrival was erased (non-finite in, finite out) |
| `BackwardNanReport.message` | Human-readable summary headlining the earliest provable birth |
| `BackwardNanReport.transitions` | The full transition ledger in measured fire order |
| `BackwardNanReport.unchecked_calls` | Calls without saved or checkable gradient payloads |
| `BackwardNanReport.verdict_scope` | `"complete"` when every earlier call was checkable, else `"found_first_among_checked"` |
| `BackwardPass.pass_number` | 1-based number of this backward pass |
| `BackwardStatus.num_backward_passes` | Number of backward passes observed |
| `BackwardStatus.num_grad_fn_records` | Number of GradFn records the status read |
| `BackwardStatus.uncounted_op_counts` | Every recorded op kind outside the counted families with its call count, never zeroed or estimated |
| `BasisProvenance.num_vectors` | Number of basis vectors |
| `BindReport.is_grad_enabled` | Whether grad mode was on when the bound run executed |
| `BindReport.is_model_training` | Whether the model was in training mode when the bound run executed |
| `BindReport.num_fires` | How many times the bound intervention fired |
| `BisectNanReport.first_bad_thing` | Project this result into the shared first-bad-thing vocabulary |
| `BisectNanReport.found` | Whether a saved activation containing NaN or Inf was found |
| `BisectNanReport.message` | Actionable human-readable result summary |
| `BisectPrecisionReport.found` | Whether any compared op diverged beyond tolerance |
| `BisectPrecisionReport.max_abs_err` | First divergent op's maximum absolute error, or `None` |
| `BisectPrecisionReport.max_rel_err` | First divergent op's maximum relative error, or `None` |
| `BisectPrecisionReport.message` | Human-readable summary |
| `Bundle.comparability` | Return the derived relationship for a pair of members (`tl.types.Relationship`) |
| `Bundle.comparability_table` | Return pairwise relationships for all member pairs |
| `Bundle.map` | Apply a function independently to each member; released effect-table candidates read `"unmeasured"`. `fn(member)`; the member name is passed only where `fn` takes it: positionally to a required second parameter or `*args` (`lambda log, name: ...`), by keyword to a parameter called `name`; a defaulted second parameter (`lambda log, k=3: ...`) keeps its default |
| `Bundle.compare` | Per-site differences between two members as `(site_label, distance)` rows; `metric=` and `on=`. Row keys: a multi-pass op gives one row per pass, keyed by its pass-qualified label (`tanh_1_1:2`); a single-pass op keeps its bare layer label (JMT, Sun 1 Nov 2026) |
| `Bundle.rank_changes` | Rank sites by their distance from a baseline member; rows keyed like `Bundle.compare` (one row per pass of a multi-pass op) |
| `BundleComparisonError.severity` | The error's severity class |
| `CapabilityRow.is_available` | Whether the capability is available in this environment |
| `CaptureFacts.is_structure_only` | Whether the capture kept structure only, without payloads |
| `CaptureFacts.is_verified` | Whether the capture was verified (`None` when not checked) |
| `CausalReceipt.num_measured_heads` | Return how many displayed heads carry a measured effect |
| `CheckFinding.check_name` | Name of the check that produced the finding |
| `CheckFinding.param_names` | The offending parameter or site names, with every alias of a tied tensor kept |
| `CheckFinding.accepted_step_id` | Identifier of the accepted optimizer step the finding belongs to, or `None` |
| `CheckFinding.attempted_step_id` | Identifier of the attempted step the finding belongs to, or `None` |
| `CheckFinding.backward_id` | Identifier of the backward pass the finding belongs to, or `None` |
| `CheckFinding.collateral_zeroed` | Count of values zeroed as collateral of the problem, or `None` |
| `CheckFinding.follow_up` | The call that localizes the culprit the finding only narrows to a cone |
| `CheckFinding.global_step` | The training step the finding was recorded at, or `None` |
| `CheckFinding.scale_provenance` | Where the grad-scaler scale in the finding came from, or `None` |
| `CheckFinding.step_provenance` | How the step number was known: observed or inferred |
| `CheckFinding.zero_baseline` | Whether the value was compared against a zero baseline, or `None` |
| `ChecksSession.register_nonfinite_grad_check` | Register the live parameter-gradient nonfinite check |
| `ChecksSession.attach` | Install the scheduled gradient, optimizer and step hooks |
| `ChecksSession.detach` | Remove every hook handle; safe to call more than once |
| `ChecksSession.disable` | Suspend all checks until `enable` is called |
| `ChecksSession.enable` | Re-enable checks after `disable` |
| `ChecksSession.estimated_snapshot_bytes` | Clone-tier snapshot budget estimated from element count and dtype size alone |
| `ChecksSession.final_report` | The report finalized at scope exit, when the session was entered as a context manager |
| `ChecksSession.paused` | Context manager that suspends all checks inside its block |
| `ChecksSession.register_change_check` | Register the parameter-change fact check |
| `ChecksSession.register_frozen_check` | Register the declared-frozen-parameter invariant check |
| `ChecksSession.register_magnitude_check` | Arm the pre-clip gradient magnitude check |
| `ChecksSession.register_nonfinite_param_check` | Register the scheduled parameter and buffer nonfinite scan |
| `ChecksSession.register_update_ratio_check` | Register the update-to-weight ratio check |
| `ClipLedgerSnapshot.has_constancy_hint` | Whether the clip ledger saw a constant clipping pattern worth hinting |
| `ClipLedgerSnapshot.accepted_steps_observed` | Number of accepted steps the clip ledger observed |
| `ClipLedgerSnapshot.applied_clip_factors` | The exact applied clip factor per step, when the magnitude tier was armed |
| `ClipLedgerSnapshot.constant_total` | The post-clip total norm when it stayed constant, else `None` |
| `ClipLedgerSnapshot.declared_clip_norm` | The clip norm the caller declared, or `None` |
| `ClipLedgerSnapshot.nonfinite_totals` | Count of steps whose total norm was nonfinite |
| `ClipLedgerSnapshot.saturated_steps` | Count of steps clipped to the declared norm |
| `ClipLedgerSnapshot.saturation_rate` | Fraction of observed steps that saturated the declared clip norm, or `None` |
| `ClipLedgerSnapshot.tier` | Evidence tier of the clip facts: exact factors, declared norm, or constancy hint |
| `CloseReport.is_disabled` | Whether the watcher was disabled |
| `CloseReport.is_final_sample_forced` | Whether close forced a final sample |
| `CostTreeRow.num_unknown_ops` | Directly-owned ops with no cost rule |
| `DeterminismReport.first_problem` | Project the first divergence into the shared vocabulary |
| `DiagnosticFinding.check_name` | Check name (`"nonfinite"`, `"dtype_regime"`, `"layout"` or `"confinement"`) |
| `DiffReport.rows_with_status` | Return the rows carrying one closed-vocabulary status |
| `DiffRow.pass_number` | Pass number of the compared op |
| `DoctorCheck.details` | Diagnostic explanation |
| `DoctorCheck.is_detected` | Detected condition; environment rows use status other than pass |
| `DoctorCheck.key` | Stable row key; environment keys retain probe names except torch_capabilities |
| `DoctorCheck.label` | Readable diagnostic title |
| `DoctorCheck.severity` | ok, info, warning, or error |
| `DoctorCheck.status` | pass, known_broken, scope, or not_tested |
| `DoctorCheck.suggestion` | Remedy text or an empty string |
| `DoctorReport.capability_snapshot` | Full named runtime capability mapping |
| `DoctorReport.check` | Return one diagnostic row by stable key; raises KeyError for an unknown key |
| `DoctorReport.checks` | Ordered tuple of environment and optional model checks |
| `DoctorReport.model_type` | Qualified model type, or None for an environment-only report |
| `DoctorReport.show` | Print and return the fixed-width table; str(report) returns the same text |
| `DoctorReport.to_markdown` | Return the diagnostic table as Markdown |
| `DoctorReport.torch_version` | Installed PyTorch version, always set |
| `DryRun.filter` | Return a new trace with decisions from a new op predicate |
| `Edit.is_compatible_with_append` | Whether the edit keeps the output shape, so an append rerun may stack it with other edits |
| `Edit.name` | The edit's public name (the builder that made it, such as `zero_ablate`) |
| `Edit.signal` | Pass an edit is bound to: `"forward"` for edits that read recorded activations, `None` for neutral edits |
| `EffectsView.rank_changes` | Rank candidates by the size of their change from baseline, descending |
| `EffectsView.to_grid` | Rebuild the `SweepGrid` of a `sweep_grid` operation from its effect table and recorded axes |
| `EffectsView.to_selection` | Convert the view to a selection: the union of its WHERE regions |
| `EpisodeOptions.acknowledge_step_cost` | Accept a step declaration beyond the cost ceiling |
| `EpisodeOptions.crossings` | Declared exogenous crossings: step indices whose inputs come from outside the capture |
| `EpisodeOptions.escalated_from` | Digest of the cheaper capture this capture escalates, when escalating |
| `EpisodeOptions.expected_tokens` | The cheaper capture's per-step tokens, carried so the escalated capture can check fidelity |
| `EpisodeOptions.feed` | Join strictness: `"open"` (default) captures across feed breaks and grades every join; `"closed"` holds joins strict |
| `EpisodeOptions.forced_tokens` | Teacher-forced token sequence fed instead of model emissions |
| `EpisodeOptions.on_feed_break` | What a feed break does: `"disclose"` (default) returns the trace with the break disclosed; `"refuse"` raises |
| `EpisodeOptions.rng` | Seeding discipline; only `"managed"` ships |
| `EpisodeOptions.step_input_from` | Which stepped-call argument carries the step input: a position, a keyword name, or `None` |
| `FindNanReport.bad_tensors` | Tensor positions found non-finite (`("output",)` for an eager finding) |
| `FindNanReport.first_bad_thing` | Project this result into the shared first-bad-thing vocabulary |
| `FindNanReport.found` | Whether a NaN or Inf was found |
| `FindNanReport.label_status` | Whether the reported label is final (`"final"`) or absent (`"none"`) |
| `FindNanReport.message` | Human-readable diagnostic summary |
| `FireRecord.backward_pass_number` | Backward pass the edit fired in, or `None` for a forward fire |
| `FireRecord.edit` | The `Edit` that fired, or `None` for a raw hook |
| `FireRecord.edit_name` | Name of the edit or hook that fired |
| `FireRecord.sampling` | Realized draws of the stochastic verbs during this fire: a tuple of `SamplingRecord`, saved with the trace (`determinism_note` keeps the readable line) |
| `FireRecord.signal` | Pass the fire happened in: `"forward"` or `"backward"` |
| `GqaInfo.num_query_heads` | Number of query heads |
| `GradFnCall.fire_record` | The fire record of the edit that produced this grad-fn call, or `None` |
| `GradFnCall.grad_fn_label` | Label of the grad_fn this call belongs to |
| `GradFnCall.label` | Pass-qualified call label (`<grad_fn_label>:<call_index>`) |
| `GradFnSketch.num_nodes` | Number of walked nodes |
| `GradReceptiveField.batch_support` | Batch indices the influence set reaches, or `None` |
| `GradReceptiveField.cross_batch_influence` | Whether the influence crosses batch elements |
| `GradReceptiveField.io_role` | The far end of the probe, the space `support_mask` covers: the source (model input name or op label) for a receptive probe, the target op label for a projective probe; `source_key` / `target_key` read it direction-stably |
| `GradReceptiveField.num_nonfinite` | Number of NaN or Inf gradient entries; nonzero makes validation indeterminate |
| `GradReceptiveField.rtol` | Relative tolerance for the support threshold |
| `GradReceptiveField.signed_grad` | The signed gradient, kept so a sign-disagreeing adjoint can be checked |
| `GradReceptiveField.source_key` | Return the direction-stable key of the source graph point |
| `GradReceptiveField.spatial_support_mask` | The support mask reduced to the spatial axes, when the layout has them |
| `GradReceptiveField.support_mask` | Boolean mask of input elements above the tolerance threshold |
| `GradReceptiveField.support_ranges` | Per-axis index ranges of the support, or `None` |
| `GradReceptiveField.target_key` | Return the direction-stable key of the target graph point |
| `GradReceptiveField.unit_shape` | Shape of the target unit |
| `GradReceptiveField.warnings` | Warnings raised while measuring |
| `GradRecordContext.pass_number` | Pass number of the op whose gradient is recorded |
| `GridCell.candidate_id` | The cell id: the comma-joined value names in axis order, also the retained member's name |
| `GridCell.coords` | Axis name to value name for this cell |
| `GridCell.error` | The refusal (`grid_cell_unresolved`, `grid_cell_multi_site`) or failure text; None otherwise |
| `GridCell.member_name` | The retained member's name, None when the cell holds no member |
| `GridCell.status` | `completed`, `released`, `refused` or `failed` |
| `GridCell.value` | The metric value, None for a refused or failed cell |
| `GuessInputShapeReport.attempts` | Probe diary as `(shape, outcome)` tuples |
| `GuessInputShapeReport.constraining_module` | Module qualname that constrained the inferred shape, when known |
| `GuessInputShapeReport.constraining_op` | Operation name that constrained the inferred shape, when known |
| `GuessInputShapeReport.example_input` | Ready-to-use input object |
| `GuessInputShapeReport.flexible_dims` | Dimensions that are expected to tolerate other sizes |
| `GuessInputShapeReport.found` | Whether a verified input was found |
| `GuessInputShapeReport.message` | Actionable human-readable summary |
| `GuessInputShapeReport.shapes` | Multi-input shapes, when applicable |
| `GuessInputShapeReport.strategy` | Strategy that produced the winning input |
| `GuessInputShapeReport.value_range` | Synthetic value recipe: `("uniform", 0, 1)` or `("randint", 0, vocab)` |
| `HealthFacts.num_nonfinite_ops` | Nonfinite ops in the identity partition (alias rows excluded) |
| `HealthFacts.num_unmapped_ops` | Ops the health partition could not map |
| `Histogram.compute` | Return the finalized histogram counts (`HistogramResult`) |
| `HistogramPoint.bin_edges` | Histogram bin edges for the point's counts |
| `HistoryReader.num_chunks` | Number of committed chunks |
| `HookContext.run_state` | Run-scoped scratch dict shared by every hook of one run |
| `HookContext.signal` | Pass the hook is firing in: `"forward"` or `"backward"` |
| `HookContext.state` | Per-(site, hook) scratch dict |
| `HtmlReportOptions.deterministic` | Suppress the manifest timestamp for byte-exact regeneration |
| `HtmlReportOptions.max_array_bytes` | Thumbnail byte budget, enforced before payloads are written |
| `HtmlReportOptions.max_array_sites` | Thumbnail site budget, enforced before payloads are written |
| `HtmlReportOptions.share_safe` | Strip every embedded value |
| `InterventionValidationReport.is_complete` | Whether the validation exercised all requested axes |
| `InterventionValidationReport.is_consistent` | Whether repeated validation decisions agree |
| `InterventionValidationReport.is_invariant` | Whether a baseline forward validation succeeds |
| `InterventionValidationReport.is_local` | Whether intervention checks stay local to the requested model/input |
| `InterventionValidationReport.is_specific` | Whether hook targets can be represented without ambiguity |
| `InterventionWarning.severity` | The warning's severity class (`"recoverable"` by default) |
| `KernelProfileReport.availability` | Profiler session availability in five states (missing is `None`, never zero) |
| `KernelProfileReport.device_time_ns` | Exact-attribution device nanoseconds for one op label |
| `KernelProfileReport.device_time_table` | Per-op exact device time, descending; refuses when unjoined |
| `KernelProfileReport.join` | The profiler join result that maps kernels to ops |
| `KernelProfileReport.session_facts` | Facts about the profiler session |
| `Layer.is_edited` | Whether a user edit replaced the output of any pass of this layer |
| `Layer.short_label` | Short pass-qualified label of a single-pass layer |
| `Layer.short_layer_label` | Short layer label (no pass suffix) |
| `Layer.source_trace` | Back-reference to the owning Trace; a detached Layer refuses instead of returning `None` |
| `LineageReport.message` | Human-readable status |
| `LineageReport.start_label` | Resolved starting op label, or the requested label when lookup failed |
| `MachineBalance.peak_bytes_per_s` | Peak memory bandwidth used for the roofline, in bytes per second |
| `MachineBalance.peak_flops_per_s` | Peak compute rate used for the roofline, in FLOPs per second |
| `MemoryFacts.captured_bytes` | Bytes the capture retained when it ran (an immutable capture fact) |
| `MemoryFacts.captured_saved_ops` | Ops whose payload the capture saved |
| `MemoryFacts.retained_bytes` | Bytes this object holds now (present or lazily loadable payloads); zero on a payload-stripped artifact |
| `MemoryFacts.retained_lazy_ops` | Ops whose payload this object can load lazily |
| `MemoryFacts.retained_present_ops` | Ops whose payload is present in memory |
| `MergeReport.degradations` | Environment degradations recorded for the merge |
| `MergeReport.num_joins` | Number of cross-rank joins made |
| `MergedFinding.from_dict` | Rebuild a finding from `to_payload` output |
| `MergedTrace.at` | Return the SPMD fan: the ops matching `label` on every rank |
| `MergedTrace.degradations` | Environment degradations recorded at load (empty on a live merge) |
| `ModuleCall.has_pre_hook_input_change` | Whether a user forward pre-hook changed the call's effective inputs (`None` when evidence is incomplete) |
| `ModuleCall.label` | Pass-qualified module call label (`address:call_index`) |
| `ModuleCall.parent_label` | Label of the parent ModuleCall in the dynamic call tree |
| `ModuleInputSnapshot.is_capture_complete` | Whether all supported mutation evidence was available for the snapshot |
| `ModuleStackFrame.pass_number` | Call number of this module in the stack |
| `NarrationEvent.has_intervention` | Whether an intervention acted at this event |
| `NarrationEvent.value_stats` | `tl.stats.TensorStats` for this tensor, or `None` when no statistics were requested or available |
| `NarrationEvent.tier` | Capture tier the event came from, such as `post_hoc`, or `None` |
| `Op.edits` | The FireRecords of every edit that fired at this op, in fire order (was `interventions`) |
| `Op.is_edited` | Whether a user edit replaced this op's output |
| `Op.pass_number` | 1-based pass of this op within its layer |
| `Op.short_label` | Short pass-qualified op label |
| `Op.short_layer_label` | Short layer label (no pass suffix) |
| `Op.source_trace` | Back-reference to the owning Trace; a detached Op refuses instead of returning `None` |
| `Op.type` | The op's normalized operation type token (for example `"linear"`); `Layer.layer_type` is the Layer's spelling |
| `OptimizerFacts.is_known` | Whether the optimizer is recognised; an unknown optimizer gives `band_reason` instead of a guessed band |
| `OptimizerFacts.decoupled_decay` | Whether the optimizer applies weight decay decoupled from the gradient step |
| `OptimizerFacts.group_fact` | One per-parameter-group hyperparameter fact, or `None` when unknown |
| `OptimizerFacts.param_group_facts` | The per-parameter-group facts read from the optimizer |
| `PreHookEffect.has_changes` | `True`/`False` when the pre-hook's mutation evidence is complete, otherwise `None` |
| `PreprocessingComparison.applied_source` | How the applied side was obtained: `"resolution"` / `"declaration"` / `"parsed_transform"` / `"undeclared"` / `"opaque_transform"` |
| `PreprocessingComparison.authority_status` | The authority's standing (`authoritative` / `unverified_fallback` / `unknown`); a non-authoritative reference caps the verdict below `verified` |
| `PreprocessingComparison.mismatched_fields` | Names of every mismatched field, in field order |
| `PreprocessingComparison.unknown_fields` | Names of every uncomparable field, in field order |
| `PreprocessingComparison.unknown_reasons` | Deduplicated reasons behind every unknown field, in field order |
| `PreprocessingComparison.verdict` | `"verified"` / `"mismatch"` / `"unknown"` |
| `ProfileReport.capture_verification_reason` | Why the capture is unverified when `is_capture_verified` is `False`, else `None` |
| `ProfileReport.has_rescue_rerun` | Whether the profile needed a rescue rerun |
| `ProfileReport.is_structure_only` | Whether the profiled capture kept structure only |
| `ReceptiveField.axis_roles` | Axis layout of the input grid: per-axis kinds, windowed axes and their provenance |
| `ReceptiveFieldValidation.num_violations` | Number of cross-validation violations found |
| `ReceptiveFieldView.axis_roles` | Axis layout of the viewed op's input grid |
| `ReceptiveFieldView.of` | Many-unit influence region as a trace-bound `tl.Selection`, one region per `within=` site |
| `ReceptiveFieldView.probe_grad` | Empirical gradient probe of one unit's receptive field |
| `ReceptiveFieldView.status` | Field status of the only reachable endpoint, or a read-only mapping from each endpoint's IO role to its status when several are reachable (a projective view of a GPT-2 activation reaches the logits and every later block's cached keys and values) |
| `Recipe.address_class` | Most restrictive address class over all rules (worst wins) |
| `Recipe.decision` | The sole rule's normalized decision (`None` on multi-clause specs) |
| `Recipe.door_pairs` | Lower the spec to `(site, action)` pairs for the hook doors |
| `Recipe.match` | Return the single rule matching one op context, or `None` |
| `RecordContext.module_pass_number` | Call number of the innermost module, or `None` |
| `RecordContext.pass_number` | Pass number of the recorded op |
| `Recording.num_predicate_failure_overflows` | Predicate failures beyond the kept list (counted, not stored) |
| `RooflineRow.is_cache_resident_hint` | Whether the row's working set likely fits in cache (a hint, not a measurement) |
| `Rule.action_repr` | Canonical rendering of the action over its current tensor content |
| `Rule.address_class` | How the rule addresses its sites (value-dependent by default) |
| `Rule.decision` | The rule's normalized decision, or `None` |
| `Rule.payload_fidelity` | Audit fidelity of this rule's action payload |
| `Rule.rule_id` | Content-derived rule identity over WHERE and action |
| `Rule.signal` | Pass the rule edits (`"forward"`, `"backward"` or `"both"`), derived from its target |
| `Rule.where_repr` | Canonical rendering of the rule's WHERE clause |
| `RunReport.has_carried_state` | Whether mutations were deliberately left on the live model; the default run restores declared state and reports `False` |
| `RunReport.is_poisoned` | Whether the run left the provider unusable for further runs |
| `RunnableDiagnostic.is_analysis_load_available` | Whether the artifact can still load for analysis only |
| `SamplingRecord.base_seed` | The plan's declared or capture-derived base seed; `None` for an averaging verb |
| `SamplingRecord.derived_seed` | The stateless per-firing seed derived from the base seed, population identity, donor group and coordinate |
| `SamplingRecord.donor_group_id` | Donor-sharing group of the firing (`per_group` plans share one draw across sites) |
| `SamplingRecord.draw_indices` | The full realized index vector: donor member ids, or the row permutation of `permute_batch`; empty when nothing was drawn |
| `SamplingRecord.eligible_count` | Members eligible for the draw after agreement-class filtering |
| `SamplingRecord.logical_firing_coordinate` | Where the draw fired: the logical site, pass and row coordinate the seed derivation used |
| `SamplingRecord.population_count` | Size of the population drawn from |
| `SamplingRecord.population_identity` | Digest identity of the population |
| `SamplingRecord.realized_draw_digest` | Short SHA-256 of the realized index vector |
| `SamplingRecord.share_draw` | Sharing rule of the plan: `per_firing`, `per_rule` or `per_group` |
| `SamplingRecord.verb` | The stochastic verb that drew: `patch_from`, `resample_rows_from`, `permute_batch`, `set_direction_mean` or `mean_ablate` |
| `SavedFigure.disclosure_lines` | Every honesty line the artifact rendered, in order |
| `SavedFigure.num_pages` | Total page count (equals `len(paths)` for file output) |
| `SavedFigure.svg_fonttype` | The matplotlib `svg.fonttype` used (`"path"` by default) |
| `ScaleLedger.is_enabled` | Whether a live, enabled GradScaler is attached |
| `ScaleLedger.observe_accepted_step` | Record one accepted optimizer step |
| `ScaleLedger.observe_backward` | Record one backward's scale reading; count decrements exactly |
| `ScaleLedger.observe_scale` | Fold any unobserved grad-scaler movement into the ledger without counting a backward |
| `ScaleLedgerSnapshot.has_scaler` | Whether a GradScaler was attached |
| `ScaleLedgerSnapshot.is_enabled` | Whether the attached GradScaler was enabled |
| `ScaleLedgerSnapshot.accepted_steps` | Number of optimizer steps the scaler accepted |
| `ScaleLedgerSnapshot.backward_reads` | Number of backward passes observed |
| `ScaleLedgerSnapshot.current_scale` | The grad-scaler scale at snapshot time |
| `ScaleLedgerSnapshot.growth_events` | Number of scale growth events observed |
| `ScaleLedgerSnapshot.inferred_k` | The inferred growth interval, or `None` when it could not be inferred |
| `ScaleLedgerSnapshot.initial_scale` | The grad-scaler scale when observation began |
| `ScaleLedgerSnapshot.preconditions` | The preconditions the ledger checked before inferring scaler behaviour |
| `ScaleLedgerSnapshot.skipped_attempts` | Number of step attempts the scaler skipped |
| `StatsTableRow.value_stats` | `tl.stats.TensorStats` for this op on `state == "ok"` rows, or `None` otherwise (`state` names why the numbers are absent) |
| `ScanRow.is_audited` | False exactly when `skip_reason` says why the row was not audited |
| `ScanRow.is_constant` | Whether the scanned value stayed constant (`None` when unknown) |
| `ScanRow.abs_max` | Largest absolute value of the scanned tensor |
| `ScanRow.bounds_above` | Count of values above the declared upper bound |
| `ScanRow.bounds_below` | Count of values below the declared lower bound |
| `ScanRow.dtype_max` | Largest finite value the tensor dtype can hold |
| `ScanRow.subnormal_fraction` | Fraction of values in the subnormal range |
| `ScanRow.zero_fraction` | Fraction of values equal to zero |
| `Selection.align_to` | Re-bind a resolved selection onto another trace |
| `Selection.labels` | Pass-qualified op label of every row, in row order |
| `Site.record` | The `Op`, `Param` or `EdgeUseRecord` the row names, through a weak trace link |
| `Selection.selector` | The selector that produced the rows, or `None` |
| `Selection.to_pandas` | One row per site: label, kind, key, num_selected, numel, shape, relation, source |
| `Session.to_bundle` | Return a bundle containing all captured invocation logs |
| `Site.num_selected` | Return the number of selected elements at this site |
| `Site.structural_site_key` | The structural site key when present, else `None` |
| `SiteRow.pass_number` | Pass number of the site |
| `SixNDFacts.num_params` | Parameter count used by the 6ND estimate |
| `SpecCompat.has_identical_target_resolution` | Whether the spec resolves to the same targets on both sides |
| `SpliceResult.ce_loss_clean` | Mean next-token cross-entropy of the clean run when `tokens=` is given, else `None` |
| `SpliceResult.ce_loss_spliced` | Mean next-token cross-entropy with the SAE reconstruction spliced in, else `None` |
| `SpliceResult.kl_divergence` | Mean KL divergence from the clean to the spliced next-token distribution, else `None` |
| `SrpSizing.num_components` | Recommended number of projection components |
| `SrpSizing.num_samples` | Number of samples (rows) the projection width was sized for |
| `SummaryTotals.captured_bytes` | Bytes the capture retained when it ran |
| `SummaryTotals.retained_bytes` | Bytes this object holds now |
| `Super.absent_names` | Return bundle member names not represented by this view |
| `Super.is_complete` | Return whether every bundle member is represented |
| `SweepGrid.axes` | Axis name to value names, in declaration order |
| `SweepGrid.baseline_value` | The metric on the un-edited baseline |
| `SweepGrid.bundle` | The Bundle (baseline plus retained members) the grid's effect table lives on |
| `SweepGrid.candidate_id` | The cell id at the given coordinates (every axis by keyword) |
| `SweepGrid.cell` | The `GridCell` at the given coordinates |
| `SweepGrid.cells` | Every cell in product order |
| `SweepGrid.coords` | The coordinates of a cell id (the inverse of `candidate_id`) |
| `SweepGrid.deltas` | `values - baseline_value`; NaN stays NaN |
| `SweepGrid.edit_axis` | The axis holding edits, None when edit= served every cell |
| `SweepGrid.effects` | The `EffectsView` over the grid's effect table |
| `SweepGrid.failed` | Cell ids whose forward or metric raised |
| `SweepGrid.history_id` | The recorded `site_sweep` operation's id |
| `SweepGrid.index` | The integer index tuple into `values` at the given coordinates |
| `SweepGrid.refused` | Cell ids refused at preflight (never run) |
| `SweepGrid.to_pandas` | Long format, one row per cell (needs the `tabular` extra) |
| `SweepGrid.values` | float64 tensor shaped by the axes; NaN for refused or failed cells |
| `TapRecord.signal` | Pass the tapped value belongs to: `"forward"` or `"backward"` |
| `TensorInputObservation.mutation_version` | Captured PyTorch version counter of the input tensor (diagnostic, not portable) |
| `TensorInputObservation.python_object_id` | In-process Python object identity of the input tensor (diagnostic, not portable) |
| `TensorSpec.from_tensor` | Build a spec from a tensor (alternate constructor) |
| `Trace.armed_edits` | Read-only tuple of the gradient rules in the trace's recipe |
| `Trace.input` | The original user input before `input_transform` was applied (was `raw_input`) |
| `Trace.is_replayable` | Whether the capture ran with `replayable=True` (the runnable apparatus: runnable save, edge provenance). Saved-graph replay (`do()` without `model=`, `push`) needs only the per-op argument templates every `tl.trace` capture records (`trace.options.save_arg_templates`), so it does not need this flag |
| `Trace.is_stopped` | Whether the capture stopped early (`until=`/`stop_after=`); a stopped trace is partial (was `halted`) |
| `Trace.options.seed` | The torch seed the capture ran under (`seed=` on the capture door), or `None`; saved with the trace |
| `Trace.transformed_output` | Human-readable model output after `output_transform` (was `raw_output`) |
| `TraceSlice.empty` | Return whether the slice has no members (emptiness is disclosure) |
| `ValidationDiagnostic.check_name` | Stable name for the validation diagnostic source |
| `ValidationFailure.check_name` | Which validation phase fired (always set) |
| `ValidationStatus.available_live` | Build the default status for a live trace with validation support |
| `ValidationStatus.exempted_reason_counts` | Stable reason-code counts for nodes whose replay/perturbation check had a provable exemption |
| `ValidationStatus.from_replay_counts` | Fold replay outcome counts into a trace-level status |
| `ValidationStatus.is_available` | Whether replay validation can run for this trace |
| `ValidationStatus.message` | Human-readable status explanation |
| `ValidationStatus.num_effect_region_ops` | Number of effectful or stateful nodes excluded from pure per-op replay |
| `ValidationStatus.num_failed_ops` | Number of replay-validation failures contributing to a failed result |
| `ValidationStatus.num_pure_unverified_ops` | Number of pure value nodes that classified cleanly but were not replayed, usually because no faithful backend raw-op replay is available |
| `ValidationStatus.num_replayed_ops` | Number of ops or regions replay-validated for a completed result |
| `ValidationStatus.num_unverified_ops` | Aggregate number of nodes or regions excluded from per-op replay |
| `ValidationStatus.passed` | Whether replay validation ran and passed |
| `ValidationStatus.payload_load_status` | Optional payload materialization status attached by `tl.load` |
| `ValidationStatus.unavailable_loaded_runtime_stripped` | Build the loaded non-torch unavailable status |
| `ValidationStatus.unavailable_unsupported` | Build an unavailable status for a backend without replay validation |
| `ValidationStatus.unverified_reason_counts` | Stable reason-code counts for nodes that replay could not verify |
| `WatchdogSnapshot.is_tripped` | Whether the watchdog fired |
| `WatchdogSnapshot.backwards_since_accepted` | Backward passes seen since the last accepted step |
### Members by class
Members of classes whose public home changed, one entry per class.

**`tl.types.Container`**
: Computed view over a captured Python output container. `Container.leaf_occurrences`: every tensor occurrence and its container path; `Container.leaves`: the ops whose outputs are the container's tensor leaves; `Container.reconstruct_at`: rebuild the container from a snapshot selector (`site=`, `role=`, `values=`); `Container.reconstructable`: whether the view can rebuild the original Python object; `Container.root_id`: identity of the container's root; `Container.root_kind`: what kind of root the container hangs from; `Container.supports_reconstruct`: whether reconstruction is supported for this container.

**`tl.types.ContainerSpec`**
: Portable description of an output container seen during capture. `ContainerSpec.aux_data`: auxiliary data needed to rebuild the container type; `ContainerSpec.child_specs`: the child path components and their nested specs; `ContainerSpec.kind`: the container kind (`tuple`, `list`, `dict`, `namedtuple`, `dataclass`, `hf_model_output`, `registered`, `distribution` for a `torch.distributions` object recorded one output per parameter at attribute paths such as `.loc`, `literal`, or `opaque`); `ContainerSpec.literal_value`: the stored value for a non-tensor literal leaf; `ContainerSpec.lossy_reconstruction`: whether rebuilding the container loses information; `ContainerSpec.type_module`: module of the container's type; `ContainerSpec.type_qualname`: qualified name of the container's type.

**`tl.types.ContainerLeafOccurrence`**
: One tensor occurrence at one container path. `ContainerLeafOccurrence.occ_index`: index of this occurrence; `ContainerLeafOccurrence.producer_op_label`: label of the op that produced the tensor, or `None`; `ContainerLeafOccurrence.tensor_identity`: identity of the tensor, or `None`.

**`tl.types.MergeAlignment`**
: Structural merge verdict; value divergence never changes it. `MergeAlignment.ALIGNED`: the rank captures line up structurally; `MergeAlignment.CONFLICTED`: at least one structural conflict was found.

**`tl.types.MergeValueStatus`**
: Merge-level value verdict derived from the join ledger. `MergeValueStatus.ATTESTED_COMPLETE`: every applicable join boundary is attested; `MergeValueStatus.ATTESTED_PARTIAL`: some applicable join boundaries are attested, not all; `MergeValueStatus.DIVERGENT`: at least one join boundary mismatched across ranks; `MergeValueStatus.UNWITNESSED`: no join boundary is attested.

**`tl.types.MergedFinding`**
: One structured merge finding. `MergedFinding.detail`: human-readable explanation (diagnostics may sharpen it but are never authority); `MergedFinding.membership_digest`: the membership the finding concerns, when it is membership-scoped.

**`tl.report.MergeReport`**
: Graph-free diagnostic summary of a merge. `MergeReport.alignment`: the effective structural verdict; `MergeReport.conflicts`: the structural conflict findings, which never join and never become gaps; `MergeReport.divergences`: the value-divergence findings: ranks disagreeing on a witnessed value; `MergeReport.expected_ranks`: the declared world, when one was given; `MergeReport.gaps`: the presence-gap findings: boundaries a rank never recorded; `MergeReport.stored_alignment`: the structural verdict derived at merge time; `MergeReport.value_status`: the merge-level value verdict.

**`tl.types.MergedTrace`**
: N rank-local captures stitched at their collective boundaries. `MergedTrace.alignment`: the effective verdict: the stored one, lowered by load degradations; `MergedTrace.expected_ranks`: the declared world, when one was given (it can only widen expectations); `MergedTrace.gaps`: the typed presence-gap ledger; `MergedTrace.happens_before`: partial-order query over two join keys (`left`, `right`); `MergedTrace.join_ops`: resolve a join's boundary ops on every presenting rank; `MergedTrace.joins`: the cross-rank collective joins, in deterministic key order; `MergedTrace.member_outcomes`: read-only map of each rank to its settled capture-outcome status; `MergedTrace.rank_ids`: the global ranks present in the merge; `MergedTrace.cleanup`: drop every rank-trace handle so the member traces can be reclaimed; `MergedTrace.stored_alignment`: the structural verdict derived at merge time; `MergedTrace.stored_value_status`: the value verdict derived at merge time; `MergedTrace.value_status`: the merge-level value verdict from the join ledger.

**`tl.trackers.HistoryReader`**
: Reads one history artifact without loading the run or importing a model. `HistoryReader.observations`: materialize observations, optionally filtered by `step`, `site_id` and `stream`; verifies checksums; `HistoryReader.step_blocks`: the committed step blocks, oldest first.

**`tl.trackers.HistoryView`**
: Session-side read view over one run's committed history. `HistoryView.from_blocks`: wrap in-memory committed blocks; `HistoryView.from_collector`: read the live collector's in-memory ring; `HistoryView.from_reader`: build a view from a disk artifact (checksums verified); `HistoryView.series`: one site and stream series in step order, disambiguated by phase; `HistoryView.site_streams`: the (stream, phase) pairs observed for one site.

**`tl.trackers.SeriesPoint`**
: One sample of a series. `SeriesPoint.observation`: the observation record; `SeriesPoint.step_hi`: last step of the span the sample covers; `SeriesPoint.step_lo`: first step of the span the sample covers.

**`tl.types.PartialTrace`**
: Thin wrapper around raw capture state from a failed forward pass. `PartialTrace._FIELD_POLICY`: the declared field policy, so field-declaration checks also cover failed partial captures; `PartialTrace.audit`: an evidence-backed health report for the partial capture; `PartialTrace.capability_card`: which family members work on this object; `PartialTrace.from_trace`: build the wrapper from a failed capture's internal state and its exception; `PartialTrace.original_exception`: the exception the failed capture raised; `PartialTrace.output_table`: typed refusal: a failed forward has no outputs; `PartialTrace.raw_layers`: the layer entries captured before the failure; `PartialTrace.to_agent_json`: typed refusal: the machine map needs finalized identities.

**`tl.types.InputProvenance`**
: Persistent provenance for how a capture's input came to exist. `InputProvenance.cache_key_version`: version of the input cache key, or `None`; `InputProvenance.cache_status`: whether the input came from the cache, or `None`; `InputProvenance.call_structure`: normalized call description: positional count, keyword names, and whether non-tensor arguments are present; `InputProvenance.caveats`: the rendered disclosure sentences; `InputProvenance.d8_meta`: metadata for the reserved meta-device rung, or `None`; `InputProvenance.evidence_grade`: `verified_exact`, `empirically_flexible`, `under_determined` or `unsupported`; `InputProvenance.execution_policy`: `"caller_mode"` (the caller's flags are kept) or `"eval_no_grad_restored"` (eval and no-grad are pinned, then restored); `InputProvenance.lazy_transitions`: module addresses whose lazy parameters materialized during the captured forward; `InputProvenance.rung`: the input ladder rung, 1 to 3.

**`tl.types.ContractCheck`**
: One contract check recorded in a run report. `ContractCheck.diagnostic`: the diagnostic for a failed check, or `None`.

**`tl.types.NumericAttestationStatus`**
: Result states for the optional saved-activation check. `NumericAttestationStatus.ATTESTED`: the run's values match the saved activations; `NumericAttestationStatus.NOT_APPLICABLE`: the check does not apply to this run; `NumericAttestationStatus.NOT_PRESENT`: no saved activations were available to check against; `NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED`: the run's values did not match the saved activations.

**`tl.types.PathFaithfulness`**
: Honesty states reported by a completed sparse run. `PathFaithfulness.DIVERGED`: the run left the recorded path; `PathFaithfulness.UNVERIFIABLE`: the path could not be checked; `PathFaithfulness.VERIFIED`: the run followed the recorded path.

**`tl.types.ReadinessReport`**
: Complete non-executing readiness report for a trace provider. `ReadinessReport.capability`: the capability the report is about, or `None`; `ReadinessReport.diagnostics`: the readiness diagnostics; `ReadinessReport.provider`: which provider would serve the run; `ReadinessReport.resolver_records`: how each callable was resolved; `ReadinessReport.state_sources_available`: the state sources the run could use; `ReadinessReport.witness_completeness`: how complete the recorded witnesses are, or `None`.

**`tl.types.ReadinessStatus`**
: Overall non-executing readiness states. `ReadinessStatus.READY`: the trace can run; `ReadinessStatus.UNAVAILABLE`: the trace cannot run.

**`tl.types.RunProvider`**
: Providers selected by `Trace.run`. `RunProvider.LIVE`: run on the live model; `RunProvider.LOADED_ANALYSIS`: run from a loaded analysis trace; `RunProvider.LOADED_SPARSE`: run from a loaded sparse trace.

**`tl.types.RunReport`**
: Honesty, state, resolution and contract metadata for one run. `RunReport.contract_checks`: the contract checks the run recorded; `RunReport.first_mismatch`: the first mismatch diagnostic, or `None`; `RunReport.initializer_policy_version`: version of the initializer policy used to fill slots, or `None`; `RunReport.nondeterministic_sources`: sources of nondeterminism the run met; `RunReport.numeric_attestation`: the saved-activation check result; `RunReport.path_faithfulness`: whether the run followed the recorded path; `RunReport.random_filled_slot_ids`: slots filled with random values; `RunReport.readiness`: the readiness report the run started from; `RunReport.state_source`: where the run's state came from.

**`tl.types.RunnableCompatibility`**
: Producer and runtime versions recorded for readiness diagnostics. `RunnableCompatibility.torchlens_version`: the torchlens version that produced the artifact; `RunnableCompatibility.backend_version`: backend version; `RunnableCompatibility.call_recipe_version`: call recipe version; `RunnableCompatibility.callable_ref_schema_version`: callable reference schema version; `RunnableCompatibility.descriptor_version`: descriptor version; `RunnableCompatibility.initializer_policy_version`: initializer policy version; `RunnableCompatibility.python_version`: Python version.

**`tl.types.RunnableDiagnostic`**
: Structured diagnostic shared by preflight, readiness and run reports. `RunnableDiagnostic.affected_op_labels`: labels of the ops the diagnostic concerns; `RunnableDiagnostic.current_runtime`: the runtime now in use, or `None`; `RunnableDiagnostic.details`: extra (key, value) details; `RunnableDiagnostic.detection_stage`: the stage that found the problem; `RunnableDiagnostic.recorded_runtime`: the runtime recorded at capture, or `None`; `RunnableDiagnostic.registry_id`: the registry entry involved, or `None`; `RunnableDiagnostic.resolver_provenance`: how the callable was resolved, or `None`.

**`tl.types.BasisProvenance`**
: Frozen reproducibility record for one subspace basis. `BasisProvenance.space_dim`: dimension of the space the basis lives in.

**`tl.types.NarrationEvent`**
: One immutable narration record. `NarrationEvent.value_stats` holds `tl.stats.TensorStats` or `None`; sampled narration omits the finiteness family, while a reused positive verdict renders "nonfinite present" without a count. `NarrationEvent.extra_labels`: extra (key, value) labels; `NarrationEvent.late`: ordinal distance disclosed on a buffered release; `NarrationEvent.module_depth`: nesting depth of the module; `NarrationEvent.module_type`: class name of the module, or `None`; `NarrationEvent.ordinal`: position of the event in the narration; `NarrationEvent.output_index`: index of the output, or `None`; `NarrationEvent.source_location`: the source location, or `None`; `NarrationEvent.step_index`: the step index, or `None`.

**`tl.types.TraceSlice`**
: Frozen presenter over one trace's executed sub-DAG. `TraceSlice.as_region`: derive an admissible region that is operated on as one unit; the slice views its trace without copying it.

**`tl.transformer.Annotation`**
: One typed annotation attached to an attention picture. `Annotation.legend`: the kind legend line.

**`tl.transformer.AttentionView`**
: One attention pattern in canonical coordinates. `AttentionView.coordinate`: optional shared site coordinate; `AttentionView.crop`: optional crop disclosure (cropping never renormalizes); `AttentionView.crop_to`: return a cropped view with an exact omitted-mass disclosure; `AttentionView.domain`: `"probability"` (fixed 0 to 1) or `"per_panel"`; `AttentionView.gqa`: optional grouped-query-attention disclosure; `AttentionView.head_view`: a single-head view selected by original head index; `AttentionView.key_tokens`: source-axis tokens, a separate record even for self-attention; `AttentionView.provenance_wording`: the rendered provenance wording for the view; `AttentionView.query_tokens`: destination-axis tokens.

**`tl.transformer.CausalReceipt`**
: Validated measured-intervention evidence for an annotation. `CausalReceipt.cell_sum`: the sum of the measured single-head effects; `CausalReceipt.fingerprints`: content fingerprints for the joined artifacts; `CausalReceipt.negative_control`: measured self-patch metric delta (must be about 0); `CausalReceipt.negative_control_tol`: tolerance for the negative control; `CausalReceipt.pattern_source`: where the attention pattern came from (default `"captured"`); `CausalReceipt.positive_control`: measured large-perturbation metric delta (must move); `CausalReceipt.positive_control_min`: smallest movement the positive control must show.

**`tl.transformer.CropInfo`**
: Crop disclosure; cropping never renormalizes. `CropInfo.key_range`: kept source (column) index range; `CropInfo.omitted_keys`: number of source positions cropped away; `CropInfo.omitted_mass_max`: maximum probability mass omitted from any kept destination row, measured on the uncropped pattern; `CropInfo.query_range`: kept destination (row) index range.

**`tl.transformer.EpisodeCoordinates`**
: Episode coordinates carried by transformer records. `EpisodeCoordinates.role`: `"prefill"` or `"decode"`.

**`tl.transformer.PredictionTable`**
: Top-k tokens per position. `PredictionTable.context_tokens`: the input token at each position; `PredictionTable.top_probs`: full-vocabulary probabilities, one row of k per position.

**`tl.transformer.PredictionTrajectory`**
: Per-layer prediction trajectory record. `PredictionTrajectory.target_probs`: the tracked token's probability at each layer, or `None`; `PredictionTrajectory.target_ranks`: the tracked token's rank at each layer, or `None`; `PredictionTrajectory.target_token`: optional tracked answer token; `PredictionTrajectory.top_probs`: full-vocabulary probabilities, one row of k per layer.

**`tl.transformer.ScoreDecomposition`**
: Term-complete score decomposition for one (head, destination, source) triple. `ScoreDecomposition.additive_terms`: named additive contributions applied after the scaled dot product; `ScoreDecomposition.destination`: destination position; `ScoreDecomposition.key_vector`: the key vector; `ScoreDecomposition.products`: elementwise query-times-key products; `ScoreDecomposition.query_vector`: the query vector; `ScoreDecomposition.reference_score`: the captured score this decomposition claims to equal; `ScoreDecomposition.tolerance`: closure tolerance.

**`tl.transformer.TokenAxis`**
: One labeled token axis. `TokenAxis.role`: `"query"`, `"key"` or `"position"`.

**`tl.transformer.TokenMetrics`**
: Per-token metric strip record (loss, entropy). `TokenMetrics.source_fingerprint`: fingerprint of the one logits tensor both metrics were computed from.

**`tl.transformer.TokenScores`**
: Colored-token strip record. `TokenScores.domain`: `"zero_centered_diverging"`, `"unit_interval"` or `"magnitude_sequential"`; `TokenScores.from_attribution`: build a strip record from an attribution token payload.

### Record class members

Members of the capture record classes that had no entry of their own. Each row names the member on its
public class; every class below is reached through the home shown in its heading. Rows are in
alphabetical order.

#### `tl.Trace`

| Spelling | Meaning |
| --- | --- |
| `Trace.__enter__` | Context-manager entry; returns the Trace itself so `with trace:` binds the same object |
| `Trace.__exit__` | Context-manager exit; calls `Trace.remove`, detaching the most recent single-hook `do` attachment |
| `Trace.annotate` | Attach user-owned annotation data or an image to the graph nodes picked by a selector |
| `Trace.archived_activations` | Mapping of label to inspection-only activation payloads from a runnable archive |
| `Trace.attention_blocks` | Yield the Modules that have an available query-projection facet |
| `Trace.backend_runtime_config` | Backend runtime configuration recorded at capture, or `None` |
| `Trace.backend_runtime_device_summary` | Summary of the backend devices in use at trace time, or `None` |
| `Trace.backend_runtime_version` | Version string of the backend runtime that ran the capture, or `None` |
| `Trace.backward` | Run backward from `loss` and populate this Trace with backward metadata (accepts `save_grads`; other keywords pass to `torch.Tensor.backward`); returns the Trace |
| `Trace.batch_render` | Policy for rendering a batch of raw input in visualizations; defaults to `"auto"` |
| `Trace.buffer_layers` | Labels of the Layers that represent buffers as graph boundary values |
| `Trace.buffer_read_ops` | Labels of buffer Ops that read registered-buffer values into the graph |
| `Trace.buffer_write_ops` | Labels of buffer Ops that record writes to registered buffers |
| `Trace.call_tree` | The root `ModuleCall` of the nested module call tree, as structured data |
| `Trace.capture_end_time` | Wall-clock time (seconds since the epoch) when the capture ended |
| `Trace.capture_guard_passes` | List with one dict per capture guard pass, giving its index, capture mode and owner thread id |
| `Trace.capture_start_time` | Wall-clock time (seconds since the epoch) when the capture started |
| `Trace.capture_tensor_grad_hooks` | Whether forward tensors receive tensor-level backward hooks for implicit backward events and per-op gradients |
| `Trace.check_metadata_invariants` | Run the metadata invariant checks on this completed trace and return whether they pass |
| `Trace.check_plan` | Audit a plan against this trace, resolving its selectors and checking declared shapes, without arming or running anything |
| `Trace.chunked_forward` | Whether the trace was assembled from forward chunks |
| `Trace.cleanup` | Delete log data, break reference cycles and free cached GPU memory |
| `Trace.completeness_diagnostics` | List of reports of ATen dispatch events the completeness witness could not account for; empty when capture was complete |
| `Trace.compute_coverage` | Counts of layer-list rows in each of four compute-coverage categories, as a dict |
| `Trace.decode_output` | Return the decoded output rows captured during tracing, up to `k` ranks per batch item; with `style=`, `head=` or `tokenizer=` decode again on demand from the retained output tensors without touching `decoded_output` (`style="hf_text"` without a tokenizer returns the argmax `token_ids` with `text` `None`); raise if nothing was captured and no output tensor is retained |
| `Trace.escape_diagnostics` | List of reports of raw torch callables that ran outside their registered wrapper during capture, each with file, line and function |
| `Trace.factcore` | The single numbers substrate for this trace: counts, identity spine, parameter totals, compute aggregation and memory figures |
| `Trace.find_nan` | Return the first NaN or Inf found among saved outputs in execution order |
| `Trace.find_sites` | Return a table of the intervention sites that match a selector |
| `Trace.has_frozen_params` | Whether this Trace contains at least one frozen parameter |
| `Trace.has_trainable_params` | Whether this Trace contains at least one trainable parameter |
| `Trace.health_facts` | Normalized, serializable health observations for this trace, including non-finite value findings and coverage counts |
| `Trace.injected_ops` | Tuple of records for computation run inside intervention hooks when `log_injections=True`; empty if none ran |
| `Trace.input_layers` | Labels of the Layers that are model inputs |
| `Trace.internal_sink_ops` | Labels of Ops that end inside the model without reaching an output |
| `Trace.internal_source_ops` | Labels of Ops whose values originate inside the model, such as buffers and tensors created during forward |
| `Trace.internally_terminated_bool_ops` | Labels of boolean tensor ops that end inside the model without reaching an output |
| `Trace.jax_capture_index_to_final_op_label` | Dict mapping each JAX capture index to the final label of its op in the trace |
| `Trace.jax_equation_captures` | Tuple of the per-equation capture records from tracing the JAX function |
| `Trace.jax_outvar_key_to_capture_index` | Dict mapping a JAX output-variable key to the capture index of the equation that produced it |
| `Trace.last_run_records` | Return the fire records from the most recent replay, rerun or live capture |
| `Trace.layer_labels` | Layer labels in execution order, without pass suffixes such as `:2` |
| `Trace.layer_visualizers` | Optional mapping from site selectors to per-layer thumbnail callables used when rendering; not saved with the trace |
| `Trace.layers_with_params` | Dict mapping each parameter-consuming layer label to its parameter records |
| `Trace.logged_values` | Dict of the values recorded through `log_value` during this capture |
| `Trace.macs_unknown_split_ops` | Labels of compute ops whose exact multiply-accumulate (MAC) split cannot be derived |
| `Trace.model_ops` | Tuple of the trace's ordinary executed op records, excluding any ops injected by intervention hooks |
| `Trace.model_profile` | The computed `ModelProfile` semantic input/output descriptor for this trace, not persisted |
| `Trace.module_calls` | Accessor for per-invocation `ModuleCall` records, looked up by call label or index |
| `Trace.modules_with_facet` | Yield the Modules whose semantic facet view currently has the facet `name` available |
| `Trace.nonfinite_verdict` | Three-state health verdict: `found`, `checked_and_clean` or `not_checked` |
| `Trace.num_context_lines` | Number of source lines captured around each call site for `FuncCallLocation` |
| `Trace.num_edges` | Number of distinct edges in the per-pass Op graph, including boundary edges |
| `Trace.num_input_layers` | Number of distinct input-boundary Layers (a multi-pass Layer counts once) |
| `Trace.num_output_layers` | Number of distinct output-boundary Layers (a multi-pass Layer counts once) |
| `Trace.num_params_frozen` | Number of scalar parameters that do not require gradients |
| `Trace.num_params_trainable` | Number of scalar parameters that require gradients |
| `Trace.op_labels` | Op labels in execution order, always pass-qualified (ending in a pass suffix such as `:1`) |
| `Trace.output_id2label` | Dict mapping output class ids to label strings, used to decode model output, or `None` |
| `Trace.output_layers` | Labels of the Layers that are model outputs |
| `Trace.output_num_classes` | Number of output classes used to decode model output, or `None` |
| `Trace.parent_trace` | The immediate parent Trace in a fork or rerun lineage, or `None` |
| `Trace.preview_fastlog` | Render a preview of what a fastlog predicate would keep on this model graph |
| `Trace.profile_enabled` | Whether this capture was run with profiling requested |
| `Trace.reconstruct_container` | Rebuild a captured container selected by site and role from saved Op values |
| `Trace.reconstruct_output` | Rebuild the traced model's final Python output object from saved Op values |
| `Trace.remove` | Detach the hook attachments made by the most recent single-hook `do` call |
| `Trace.replace_state_from` | Atomically replace this trace's run state with that of another `Trace` |
| `Trace.root_module` | The `Module` record for the top-level model, `modules["self"]` |
| `Trace.root_trace` | The ultimate root Trace of this fork or rerun lineage, or `None` |
| `Trace.save_visualizations` | Whether rendered visualizations are persisted in bundles |
| `Trace.saved_grad_module_calls` | Accessor for the ModuleCalls whose outputs include saved gradients |
| `Trace.saved_grad_modules` | Accessor for the Modules with at least one saved-gradient ModuleCall |
| `Trace.saved_layers` | Accessor for the Layers containing at least one Op with a saved activation |
| `Trace.saved_module_calls` | Accessor for the ModuleCalls whose outputs include saved activations |
| `Trace.saved_modules` | Accessor for the Modules with at least one saved-activation ModuleCall |
| `Trace.show_call_tree` | Print this Trace's `ModuleCall` tree as an ASCII tree, with options for depth, atomic calls and call indices |
| `Trace.state_history` | Append-only list of lifecycle events for this trace, such as fork, push, run and direct write |
| `Trace.tlspec_version` | Integer version of the `.tlspec` format, used for save and load compatibility |
| `Trace.to_dagua_graph` | Translate this trace into an experimental Dagua graph |
| `Trace.transforms` | Tuple of the Op records that mark transform boundaries |
| `Trace.uncalled_modules` | Registered module addresses that were not called during the captured pass |
| `Trace.unknown_flop_ops` | Ledger of ops with unknown FLOPs, grouped by op name, with counts and example labels |
| `Trace.validation_replay_status` | Machine-readable status of replay validation: unavailable, unverified, or the last completed result |
| `Trace.walk_calls` | Yield every `ModuleCall` in the call tree in depth-first order |

#### `tl.Op`

| Spelling | Meaning |
| --- | --- |
| `Op.activation_transform` | The callable applied to this Op's output before it is saved, or `None` |
| `Op.arg_expressions` | Source expressions of this Op's call arguments, parsed lazily from `code_context`; an empty list when unavailable |
| `Op.arg_names` | Names of the arguments in this Op's function call |
| `Op.args_summary` | Human-readable summary of this Op's positional arguments |
| `Op.args_template` | Structural template of this Op's positional arguments, used to rebuild the call |
| `Op.arithmetic_intensity` | Theoretical forward FLOPs per byte of ideal memory traffic, or `None` |
| `Op.atomic_module` | The `Module` record for the atomic module this Op is the output of, or `None` |
| `Op.atomic_module_address` | Module address, without the pass suffix, of the atomic module this Op outputs from, or `None` |
| `Op.atomic_module_call` | Stored `ModuleCall` label for an atomic-module output Op, or `None` |
| `Op.atomic_module_call_label` | Stable `ModuleCall` label for an atomic-module output Op, or `None` |
| `Op.backend_address` | The Op's locator in the backend's own graph, such as `jaxpr:<label>` |
| `Op.bool_value` | The Python value of this Op's output when it is a scalar bool |
| `Op.buffer_pass` | Pass number of this Op's buffer, as used in the `address:pass` lookup key |
| `Op.buffer_replay_validated` | Whether replaying this buffer write as an identity from its source matched the recorded value |
| `Op.buffer_sink_ops` | Accessor over the buffer-write Ops that directly consume this Op |
| `Op.buffer_source` | Label of the Op that wrote this buffer value, or `None` for a static buffer |
| `Op.buffer_source_func_name` | Name of the function that produced this buffer value |
| `Op.buffer_source_ops` | Accessor over the buffer-source Ops that feed this Op |
| `Op.buffer_value_changed` | Whether this buffer write changed the buffer's value |
| `Op.buffer_write_kind` | Kind of write that produced this buffer version, or `None` for a static buffer |
| `Op.bytes_delta_at_call` | Change in allocated bytes at the time of this Op's function call |
| `Op.bytes_peak_at_call` | Peak allocated bytes at the time of this Op's function call |
| `Op.bytes_read` | Theoretical ideal read-once memory traffic for this operation, in bytes |
| `Op.bytes_written` | Theoretical ideal write-once memory traffic for this operation, in bytes |
| `Op.children` | Labels of the Ops that directly consume this Op's output |
| `Op.co_parents` | Labels of other Ops that share at least one child with this Op |
| `Op.code_context` | Call-stack frames showing where this Op ran, outermost first: `[0]` is the call site that started the capture and `[-1]` is the frame that ran the Op; `print(op)` shows the innermost frame outside torch itself |
| `Op.compute_record` | Two-term compute record (`fma_macs` and `other_flops`), derived on read |
| `Op.conditional_arm_children` | Dict mapping each conditional's id to its arm kinds (then, elif_N, else) and the labels of this Op's children inside each arm |
| `Op.conditional_branch_depth` | Number of conditional branches this Op is nested inside; 0 when it is not in a conditional body |
| `Op.conditional_context_kind` | Kind of context (`"if"` or `"while"`) of the conditional that owns this terminal bool, or `None` |
| `Op.conditional_elif_children` | Dict mapping each elif arm index to the labels of this Op's children in that arm |
| `Op.conditional_else_children` | Labels of this Op's children that sit in an else arm of a conditional |
| `Op.conditional_entry_children` | Labels of this Op's children whose edge from it enters a conditional arm |
| `Op.conditional_then_children` | Labels of this Op's children that sit in a then arm of a conditional |
| `Op.conditional_wrapper_kind` | Wrapper kind of the conditional that owns this terminal bool, or `None` |
| `Op.container` | Runtime container view for this Op, when available |
| `Op.container_spec` | Rebuildable description of the container that holds this Op's output |
| `Op.containers` | All input and output container views associated with this Op |
| `Op.derived_grad` | Backend-owned intermediate gradient payload for this Op, or `None` |
| `Op.dropped_edge_tensor_args` | Argument positions, such as `"arg1"`, whose tensor inputs did not become graph edges at capture |
| `Op.edge_uses` | Tuple of `EdgeUseRecord` entries, one per parent edge use captured for this Op |
| `Op.equivalence_class` | Key of the equivalence class that loop detection assigned to this Op |
| `Op.equivalent_ops` | Labels of other Ops in the same equivalence class (same function and arguments) |
| `Op.flops_backward` | Approximate backward FLOPs for this Op |
| `Op.flops_forward` | Approximate forward FLOPs for this Op |
| `Op.flops_total` | Approximate total FLOPs for this Op |
| `Op.func_autocast_state` | Autocast state in effect when this Op's function was called |
| `Op.func_config` | Dict of lightweight hyperparameters captured from the call, such as `kernel_size` and `stride` |
| `Op.func_duration` | Time spent in this Op's function call, as a `Duration` |
| `Op.func_id` | Portable key identifying the captured function this Op ran: its namespace, qualified name and dispatch kind |
| `Op.func_non_tensor_args` | Flat list of all non-tensor arguments of this Op's function call |
| `Op.func_qualname` | Qualified name of the function this Op called |
| `Op.func_rng_states` | RNG state snapshots recorded for this Op's function call |
| `Op.fx_call_index` | Call index in the `torch.fx`-style identity of this Op, when applicable |
| `Op.fx_label` | A `torch.fx`-style label for this Op, or `None` when module context is unavailable |
| `Op.fx_qualpath` | Module path in the `torch.fx`-style identity of this Op, when applicable |
| `Op.get_children` | Return the child `Op` objects of this Op |
| `Op.get_parents` | Return the parent `Op` objects of this Op |
| `Op.grad_dtype` | Dtype of this Op's saved output gradient |
| `Op.grad_fn_class_name` | Class name of this Op's autograd node |
| `Op.grad_fn_class_qualname` | Qualified class name of this Op's autograd node |
| `Op.grad_fn_handle` | The live PyTorch autograd node object for this Op; not portable |
| `Op.grad_fn_object_id` | Python `id()` of the autograd node object; valid only within the same session |
| `Op.grad_for` | Return the saved gradient tensor for one backward pass |
| `Op.grad_ref` | Lazy reference to this Op's gradient blob in a saved bundle, or `None`; the gradient loads only when read |
| `Op.grad_shape` | Shape of this Op's saved output gradient |
| `Op.gradient_memory` | Bytes used by this Op's saved output gradient |
| `Op.gradient_transform` | Transform used for this Op's saved gradient, or `None` when unset |
| `Op.grads` | Accessor over the per-backward-pass gradient records saved for this Op |
| `Op.has_children` | Whether this Op has any child Ops |
| `Op.has_co_parents` | Whether this Op shares children with other Ops |
| `Op.has_frozen_params` | Whether this Op consumes at least one frozen parameter |
| `Op.has_grad` | Whether a gradient was saved for this Op's output |
| `Op.has_input_ancestor` | Whether any model input is an ancestor of this Op |
| `Op.has_internal_source_ancestor` | Whether any internally initialized source Op is an ancestor of this Op |
| `Op.has_out_variations` | Whether different children observed different versions of this Op's output |
| `Op.has_output_descendant` | Whether any model output is a descendant of this Op |
| `Op.has_parents` | Whether this Op has any parent Ops |
| `Op.has_saved_activation` | Whether this Op's output activation was saved |
| `Op.has_saved_args` | Whether this Op's argument values were saved |
| `Op.has_siblings` | Whether this Op shares parents with other Ops |
| `Op.has_trainable_params` | Whether this Op consumes at least one trainable parameter |
| `Op.in_multi_output` | Whether this Op's output came from a list, tuple, dict or set returned by a function |
| `Op.in_submodule` | Whether this Op was computed inside a non-root module's forward |
| `Op.injection_provenance` | Record of which intervention injected this Op and where, or `None` for ordinary model Ops |
| `Op.input_activations` | Saved parent activations consumed by this Op, in `parents` order |
| `Op.input_ancestors` | Labels of the model inputs that are ancestors of this Op |
| `Op.input_containers` | Input container views consumed by this Op |
| `Op.input_memory` | Sum of saved activation bytes across this Op's parent Ops |
| `Op.input_to_module_calls` | Labels of the `ModuleCall` records whose input this Op's output fed |
| `Op.input_to_modules` | Module addresses, without the pass suffix, whose input this Op's output fed |
| `Op.input_was_parameter` | Whether this Op is an input boundary whose value was a model parameter |
| `Op.internal_source_ancestors` | Labels of the internally initialized source Ops that are ancestors of this Op |
| `Op.internal_source_parents` | Labels of the internally initialized source Ops that are direct parents of this Op |
| `Op.is_atomic_module` | Whether this Op is the sole operation output of an atomic module call |
| `Op.is_buffer` | Stored flag marking this Op as a buffer boundary; `is_buffer_source` exposes it |
| `Op.is_buffer_source` | Whether this Op represents a buffer boundary (overwrites a buffer) |
| `Op.is_compute_op` | Whether this Op executed a torch function rather than being a boundary sentinel |
| `Op.is_final_output` | Whether this Op is a final model output |
| `Op.is_in_conditional` | Whether this Op participates in any conditional role |
| `Op.is_in_conditional_body` | Whether this Op sits inside the body of a conditional arm |
| `Op.is_in_conditional_evaluation` | Whether this Op computes a conditional arm's condition |
| `Op.is_inplace` | Whether the underlying operation was in-place |
| `Op.is_input` | Whether this Op is a model input |
| `Op.is_internal_sink` | Whether this Op's result ends inside the model without reaching an output |
| `Op.is_internal_source` | Whether this Op was created inside the model, such as by a factory call, rather than fed from an input |
| `Op.is_internally_initialized` | Alias of `is_internal_source`: whether this Op was created inside the model rather than fed from an input |
| `Op.is_module_input` | Whether this Op's output feeds at least one `ModuleCall` as an input |
| `Op.is_module_output` | Whether this Op is the output of a `ModuleCall` |
| `Op.is_orphan` | Whether this Op is disconnected from both the input and output flows |
| `Op.is_output` | Whether this Op is a model output |
| `Op.is_output_parent` | Whether this Op is directly returned from `forward` |
| `Op.is_scalar_bool` | Whether this Op's output is a scalar bool |
| `Op.is_terminal_bool` | Whether this Op is a scalar bool that ends a chain of operations and decides control flow |
| `Op.is_terminal_conditional_bool` | Whether this terminal bool is the test of a recorded conditional |
| `Op.is_transform` | Whether this Op is a function-transform boundary node |
| `Op.kwargs_summary` | Human-readable summary of this Op's keyword arguments |
| `Op.kwargs_template` | Structural template of this Op's keyword arguments, used to rebuild the call |
| `Op.short_label` | Short Op label without the step index, such as `conv2d_1:2` |
| `Op.short_layer_label` | Short layer label without the pass suffix or step index, such as `conv2d_1` |
| `Op.lookup_keys` | Alternate keys that resolve to this Op in lookups |
| `Op.macs_backward` | Backward MACs for this Op; not derivable, so always `None` |
| `Op.macs_forward` | True forward multiply-accumulate count for this Op, or `None` |
| `Op.macs_total` | Total MACs for this Op; not derivable, so always `None` |
| `Op.materialize_grad` | Materialize this Op's saved gradient from a lazy bundle reference |
| `Op.materialize_out` | Materialize this Op's saved output from a lazy bundle reference |
| `Op.max_distance_from_input` | Longest graph distance from any model input to this Op |
| `Op.max_distance_to_output` | Longest graph distance from this Op to any model output |
| `Op.min_distance_from_input` | Shortest graph distance from any model input to this Op |
| `Op.min_distance_to_output` | Shortest graph distance from this Op to any model output |
| `Op.module_call_depth` | Depth of this Op's active `ModuleCall` nesting (the length of `module_call_stack`) |
| `Op.module_entry_arg_keys` | Mapping from `ModuleCall` label to the argument keys this Op filled at module entry |
| `Op.multi_output_index` | 0-based position of this Op's output within its multi-output container |
| `Op.multi_output_name` | Semantic name of this Op's output within its multi-output container, such as a dict key or namedtuple field |
| `Op.multi_output_type` | Python class of the multi-output container this Op came from, or `None` |
| `Op.non_tensor_kwargs` | Non-tensor keyword arguments of this Op's function call |
| `Op.non_tensor_pos_args` | Non-tensor positional arguments of this Op's function call |
| `Op.num_args_total` | Total number of arguments in this Op's function call |
| `Op.num_inputs` | Number of parent Ops of this Op |
| `Op.num_kwargs` | Number of keyword arguments in this Op's function call |
| `Op.num_param_tensors_frozen` | Number of frozen parameter tensors consumed by this Op |
| `Op.num_param_tensors_trainable` | Number of trainable parameter tensors consumed by this Op |
| `Op.num_params_frozen` | Total number of frozen parameter elements consumed by this Op |
| `Op.num_params_trainable` | Total number of trainable parameter elements consumed by this Op |
| `Op.num_pos_args` | Number of positional arguments in this Op's function call |
| `Op.ordinal_index` | 0-based position of this Op in `trace.ops`; `trace[op.ordinal_index]` returns it |
| `Op.out_ref` | Lazy reference to this Op's saved output blob in a saved bundle, or `None`; the tensor loads only when read |
| `Op.output_containers` | Output container views associated with this Op |
| `Op.output_descendants` | Labels of the model outputs that are descendants of this Op |
| `Op.output_of_module_calls` | Labels of the `ModuleCall` records whose output this Op is |
| `Op.output_of_modules` | Module addresses, without the pass suffix, whose output this Op is |
| `Op.param_dtypes` | Dtypes of the parameters consumed by this Op |
| `Op.param_memory` | Bytes used by the parameters consumed by this Op |
| `Op.param_shapes` | Shapes of the parameters consumed by this Op |
| `Op.parent_arg_positions` | Mapping from call argument positions, positional and keyword, to the parent Op label used there |
| `Op.parent_param_ops` | Mapping from each consumed parameter's identifying barcode to its pass number |
| `Op.parent_params` | The `nn.Parameter` objects consumed by this Op |
| `Op.parents` | Labels of the Ops that directly feed this Op |
| `Op.raw_index` | 1-based raw capture-time counter; may have gaps after orphan removal |
| `Op.recurrent_ops` | Labels of the Ops in the same recurrent group: equivalent and at a structurally corresponding position |
| `Op.resolver_status` | Status of the backend's address resolution for this Op, such as `resolved` |
| `Op.root_ancestors` | Labels of the root Ops at the start of the graph that this Op descends from |
| `Op.saved_kwargs` | Keyword argument values saved for this Op's call, or `None` when not saved |
| `Op.siblings` | Labels of other Ops that share at least one parent with this Op |
| `Op.terminal_bool_for` | `(conditional_id, arm_index)` when this Op is the terminal bool of a conditional arm |
| `Op.terminal_conditional_id` | Id of the conditional whose test this terminal bool decides, or `None` |
| `Op.transform_chain` | Tuple of transform tags applied to the function this Op ran |
| `Op.transform_config` | Configuration of the function transform for this transform-boundary Op |
| `Op.transform_fn_name` | Name of the function wrapped by the transform at this boundary Op |
| `Op.transform_fn_qualname` | Qualified name of the function wrapped by the transform at this boundary Op |
| `Op.transform_fn_source` | Source location of the function wrapped by the transform at this boundary Op |
| `Op.transformed_grad_dtype` | Dtype of this Op's saved transformed gradient |
| `Op.transformed_grad_shape` | Shape of this Op's saved transformed gradient |
| `Op.transformed_gradient_memory` | Bytes used by this Op's saved transformed gradient |
| `Op.transformed_out_dtype` | Dtype of this Op's saved transformed output |
| `Op.type_index` | 1-based position of this Op among Ops of the same type |
| `Op.unattributed_tensor_args` | Tensor arguments that could not be attributed to a parent Op |
| `Op.var_names` | Variable names the Op's result is assigned to at its call site; empty when unresolved |
| `Op.visualizer_path` | PNG path used for the node thumbnail in graph renders, or `None` |

#### `tl.Layer`

| Spelling | Meaning |
| --- | --- |
| `Layer.activation_transform` | The callable applied to saved activations before they are stored, or `None` when no transform was set |
| `Layer.arg_names` | Parameter names from the signature of the function this layer calls |
| `Layer.backend_address` | Backend-native address of the op or state this layer stands for, as the capturing backend names it, or `None` |
| `Layer.bool_value` | The Python `bool` value of a scalar boolean layer, or `None` otherwise |
| `Layer.buffer_overwrite_index` | Which overwrite of its buffer this layer represents |
| `Layer.buffer_pass` | Sequential number of this buffer write among the writes to the same buffer address, starting at 1, or `None` for non-writes |
| `Layer.buffer_replay_validated` | Whether replaying this buffer write from its source layer reproduced the saved value, or `None` when not a buffer write |
| `Layer.buffer_source` | The label of the op that wrote this buffer value, or `None` for static buffers |
| `Layer.buffer_source_func_name` | Name of the function that produced this buffer write, or `None` when this layer is not a buffer write |
| `Layer.buffer_value_changed` | Whether this buffer write changed the buffer's value, or `None` when this layer is not a buffer write |
| `Layer.buffer_write_kind` | How the buffer was written, one of `reassign`, `inplace`, `fused` or `data_reassign`, or `None` for a non-write |
| `Layer.call_labels` | Labels of the ops (one per pass) that make up this layer |
| `Layer.child_ops_per_layer` | Mapping from each child layer label to the pass numbers of this layer's ops that connect to it |
| `Layer.children` | Union of child layer labels across all ops of this layer, without pass suffixes |
| `Layer.children_per_pass` | Mapping from pass number to the child layer labels (without pass suffixes) of that pass |
| `Layer.co_parents` | Union across ops of the co-parent layer labels, without pass suffixes |
| `Layer.code_context` | The captured call stack of the user code that ran this layer's function, for single-pass layers, outermost first with the frame that ran it last |
| `Layer.conditional_arm_children` | Dict mapping each conditional's id to its arm kinds (`then`, `elif_N`, `else`) and the labels of this layer's child ops inside each arm |
| `Layer.conditional_branch_stack_ops` | Dict mapping each distinct conditional branch stack to the sorted pass numbers of this layer that ran under it |
| `Layer.conditional_elif_children` | Dict mapping each `elif` arm index to the labels of this layer's children that sit in that arm |
| `Layer.conditional_else_children` | List of labels of this layer's child ops that sit in an `else` arm of a conditional |
| `Layer.conditional_entry_children` | List of labels of this layer's child ops whose edge from it enters a conditional arm |
| `Layer.conditional_role_stacks` | List of the distinct conditional branch stacks, one per different nesting, that this layer's passes ran under |
| `Layer.conditional_then_children` | List of labels of this layer's child ops that sit in a `then` arm of a conditional |
| `Layer.edges_vary_across_ops` | Whether the graph edges of this layer differ between its passes |
| `Layer.equivalence_class` | Fingerprint string that groups ops repeating the same operation, used for loop detection |
| `Layer.equivalent_ops` | Labels of the ops that share this layer's equivalence class |
| `Layer.flops_backward` | Backward FLOPs of this layer's representative (first) pass |
| `Layer.flops_forward` | Forward FLOPs of this layer's representative (first) pass |
| `Layer.flops_total` | Representative total FLOPs for this layer |
| `Layer.func_config` | Non-tensor configuration of the function call, stored as a dict |
| `Layer.func_duration` | Execution time of this layer's function call, for single-pass layers |
| `Layer.func_qualname` | Qualified name of the function this layer calls |
| `Layer.func_rng_states` | RNG states captured for this layer's function call, for single-pass layers |
| `Layer.fx_call_index` | The FX-style call index of a single-op layer |
| `Layer.fx_label` | A `torch.fx`-style label for a single-op layer, or `None` |
| `Layer.fx_qualpath` | The FX-style qualified path of a single-op layer, or `None` |
| `Layer.get_children` | Return the child `Layer` objects of this layer |
| `Layer.get_parents` | Return the parent `Layer` objects of this layer |
| `Layer.grad_fn_class_name` | Class name of the autograd node that produced this layer's output, or `None` |
| `Layer.grad_fn_class_qualname` | Qualified class name of the autograd node that produced this layer's output, or `None` |
| `Layer.grad_fn_handle` | The live autograd node object for this layer's output, not kept in saved files |
| `Layer.grad_fn_object_id` | The Python `id` of the autograd node object for this layer's output |
| `Layer.has_children` | Whether any pass of this layer has child layers |
| `Layer.has_co_parents` | Whether this layer has any co-parent layers besides itself |
| `Layer.has_frozen_params` | Whether this layer uses at least one frozen parameter |
| `Layer.has_grad` | Whether the single pass has a saved gradient |
| `Layer.has_input_ancestor` | Whether any model input is an ancestor of this layer |
| `Layer.has_parents` | Whether any pass of this layer has parent layers |
| `Layer.has_saved_activation` | Whether the single pass has a saved output activation |
| `Layer.has_siblings` | Whether this layer has any sibling layers besides itself |
| `Layer.has_trainable_params` | Whether this layer uses at least one trainable parameter |
| `Layer.in_multi_output` | Whether this layer is one output of a function that returns several outputs |
| `Layer.in_submodule` | Whether this layer was computed inside a submodule |
| `Layer.input_ancestors` | Labels of the model-input layers that are ancestors of this layer |
| `Layer.input_to_module_calls` | Labels of the module calls that this layer's output is fed into |
| `Layer.input_was_parameter` | Whether this model input was an `nn.Parameter` |
| `Layer.internal_source_parents` | Labels of the parent layers whose ancestry includes an internal source, a tensor made inside the model rather than passed in |
| `Layer.is_atomic_module` | Whether this layer's op is the only op of a module call, a single-op leaf module |
| `Layer.is_buffer` | Whether this layer is a registered-buffer version node, the flag `is_buffer_source` reports |
| `Layer.is_buffer_source` | Whether this layer represents a buffer overwrite boundary |
| `Layer.is_compute_layer` | Whether this layer's representative op is a compute op |
| `Layer.is_final_output` | Whether this layer is one of the model's final outputs |
| `Layer.is_in_conditional` | Whether this layer takes part in any conditional role |
| `Layer.is_in_conditional_body` | Whether this layer sits in the body of a conditional arm |
| `Layer.is_in_conditional_evaluation` | Whether this layer computes the condition of a conditional arm |
| `Layer.is_inplace` | Whether this layer's function operates in place |
| `Layer.is_input` | Whether this layer is a model input |
| `Layer.is_internal_sink` | Whether this layer's value ends inside the model without reaching any output |
| `Layer.is_internal_source` | Whether this layer creates a tensor inside the model instead of receiving it from a model input |
| `Layer.is_module_input` | Whether this layer's representative op feeds into at least one `ModuleCall`, for single-op layers |
| `Layer.is_orphan` | Whether any op in this layer is disconnected from the main graph |
| `Layer.is_output` | Whether this layer is a model output |
| `Layer.is_scalar_bool` | Whether this layer's value is a scalar boolean |
| `Layer.is_terminal_bool` | Whether this layer is a scalar boolean that ends inside the model, such as a branch condition |
| `Layer.short_label` | The short label of a single-pass layer with its pass suffix, such as `conv2d_1:1` |
| `Layer.short_layer_label` | The short layer label without a pass suffix, built from the layer type and its type index, such as `conv2d_1` |
| `Layer.leaf_module_ops` | Set of atomic module calls whose single op belongs to this layer |
| `Layer.lookup_keys` | Keys that look up this layer in the trace, for single-pass layers |
| `Layer.macs_backward` | Always `None`, because backward MACs are not derivable |
| `Layer.macs_forward` | True forward multiply-accumulate count of the representative pass, or `None` when unknown |
| `Layer.macs_total` | Always `None`, because total MACs are not derivable |
| `Layer.module_call_depth` | Depth of module nesting around this layer |
| `Layer.multi_output_index` | Position of this layer among the outputs of a multi-output function, or `None` |
| `Layer.multi_output_name` | Name of this output within a multi-output result, derived from its container path, or `None` |
| `Layer.num_args_total` | Total number of positional and keyword arguments passed to this layer's function |
| `Layer.num_kwargs` | Number of keyword arguments passed to this layer's function |
| `Layer.num_param_tensors_frozen` | Number of frozen parameter tensors used by this layer |
| `Layer.num_param_tensors_trainable` | Number of trainable parameter tensors used by this layer |
| `Layer.num_params_frozen` | Number of frozen parameter elements used by this layer |
| `Layer.num_params_trainable` | Number of trainable parameter elements used by this layer |
| `Layer.num_pos_args` | Number of positional arguments passed to this layer's function |
| `Layer.op_labels` | Labels of the ops that belong to this layer |
| `Layer.ordinal_index` | Ordinal index of this layer, read from its first op |
| `Layer.output_descendants` | Labels of the model-output layers reachable downstream of this layer |
| `Layer.output_of_module_calls` | Labels of the module calls whose output this layer is |
| `Layer.output_of_modules` | Addresses of the modules whose output this layer is |
| `Layer.param_dtypes` | Dtypes of the parameters used by this layer |
| `Layer.param_shapes` | Shapes of the parameters used by this layer |
| `Layer.parent_arg_positions` | Map from argument kind and position to the parent layer label, merged across ops |
| `Layer.parent_ops_per_layer` | Mapping from each parent layer label to the pass numbers of this layer's ops that connect from it |
| `Layer.parent_params` | The parameter objects that feed this layer's operation |
| `Layer.parents` | Union of parent layer labels across all ops of this layer, without pass suffixes |
| `Layer.parents_per_pass` | Mapping from pass number to the parent layer labels (without pass suffixes) of that pass |
| `Layer.raw_index` | Position of this layer's first op in raw capture order |
| `Layer.resolver_status` | Whether the backend address resolves to a live object, as a string such as `resolved` or `metadata_only` |
| `Layer.saved_kwargs` | Keyword arguments captured for this layer's function call, for single-pass layers |
| `Layer.siblings` | Union across ops of the sibling layer labels, without pass suffixes |
| `Layer.terminal_bool_for` | The conditional that this terminal boolean decides, or `None` |
| `Layer.total_activation_memory` | Sum of activation memory in bytes across all ops of this layer |
| `Layer.total_flops_backward` | Sum of backward FLOPs across all ops of this layer |
| `Layer.total_flops_forward` | Sum of forward FLOPs across all ops of this layer |
| `Layer.total_flops_total` | Sum of forward and backward FLOPs across all ops of this layer |
| `Layer.total_func_duration` | Sum of function-call durations across all ops of this layer |
| `Layer.total_gradient_memory` | Sum of gradient memory in bytes across all ops of this layer |
| `Layer.total_macs_backward` | Always `None`, because backward MACs are not derivable |
| `Layer.total_macs_forward` | Sum of true forward MACs across all ops of this layer, or `None` when any pass's split is unknown |
| `Layer.total_macs_total` | Always `None`, because total MACs are not derivable |
| `Layer.total_param_memory` | Bytes used by this layer's parameters |
| `Layer.transformed_grad_dtype` | Dtype of the saved transformed gradient, or `None` |
| `Layer.transformed_grad_shape` | Shape of the saved transformed gradient, or `None` |
| `Layer.transformed_gradient_memory` | Bytes used by the saved transformed gradient, or `None` |
| `Layer.transformed_out_dtype` | Dtype of the saved transformed output, or `None` |
| `Layer.type_index` | Index of this layer among layers of the same type, counted from 1 in execution order |
| `Layer.visualizer_path` | Path of the PNG used as this node's thumbnail in graph renders, or `None` |

#### `tl.types.Module`

| Spelling | Meaning |
| --- | --- |
| `Module.address_children` | Addresses of the modules directly nested under this module in the static module tree |
| `Module.address_depth` | Depth of this module's address in the static module tree, with the root at 0 |
| `Module.address_parent` | Address of this module's parent in the static module tree, or `None` for the root |
| `Module.backward_duration` | Backward duration for this module, or `None` when backward timing is not available |
| `Module.backward_hooks` | The backward hooks registered on this module, as a list of `HookInfo` |
| `Module.backward_pre_hooks` | The backward pre-hooks registered on this module, as a list of `HookInfo` |
| `Module.buffer_layers` | Labels of the buffer layers that belong to this module |
| `Module.call_children` | Addresses of the modules this module calls directly in the dynamic call tree |
| `Module.call_depth` | Depth of this module in the dynamic call stack |
| `Module.call_labels` | Labels of this module's calls |
| `Module.call_parent` | Address of the module that calls this one in the dynamic call tree, or `None` at the root |
| `Module.call_tree` | Return this module's single `ModuleCall` as the root of a nested call tree |
| `Module.class_docstring` | The docstring of this module's class, or `None` |
| `Module.class_name` | The class name of this module |
| `Module.class_qualname` | The qualified class name of this module |
| `Module.class_source_file` | Path of the source file that defines this module's class, or `None` |
| `Module.class_source_line` | Line number where this module's class is defined, or `None` |
| `Module.class_source_location` | The class source location as `file:line`, or `None` |
| `Module.custom_attributes` | User-defined attributes of this module beyond those of a stock `nn.Module` |
| `Module.custom_methods` | Names of user-defined methods of this module beyond those of a stock `nn.Module` |
| `Module.forward_args` | Positional arguments captured for the forward call of a single-call module, or `None` |
| `Module.forward_args_summary` | Human-readable summary of the forward positional arguments of a single-call module |
| `Module.forward_args_template` | Representative structural template of the forward positional arguments |
| `Module.forward_docstring` | The docstring of this module's `forward` method, or `None` |
| `Module.forward_duration` | Wall-clock duration of this module's single forward call |
| `Module.forward_hooks` | The forward hooks registered on this module, as a list of `HookInfo` |
| `Module.forward_kwargs` | Keyword arguments captured for the forward call of a single-call module, or `None` |
| `Module.forward_kwargs_summary` | Human-readable summary of the forward keyword arguments of a single-call module |
| `Module.forward_kwargs_template` | Representative structural template of the forward keyword arguments |
| `Module.forward_pre_hooks` | The forward pre-hooks registered on this module, as a list of `HookInfo` |
| `Module.forward_signature` | The signature of this module's `forward` method, or `None` |
| `Module.forward_source_file` | Path of the source file that defines this module's `forward` method, or `None` |
| `Module.forward_source_line` | Line number where this module's `forward` method is defined, or `None` |
| `Module.forward_source_location` | The `forward` source location as `file:line`, or `None` |
| `Module.full_backward_hooks` | The full backward hooks registered on this module, as a list of `HookInfo` |
| `Module.full_backward_pre_hooks` | The full backward pre-hooks registered on this module, as a list of `HookInfo` |
| `Module.func_calls_duration` | Inclusive torch function duration of this module's single call |
| `Module.grads` | Saved output gradients of a single-call module, as a list |
| `Module.has_backward_hooks` | Whether any backward hook list on this module is nonempty |
| `Module.has_forward_hooks` | Whether any forward hook list on this module is nonempty |
| `Module.has_frozen_params` | Whether this module owns at least one frozen parameter |
| `Module.has_multiple_addresses` | Whether this module appears at more than one address |
| `Module.has_trainable_params` | Whether this module owns at least one trainable parameter |
| `Module.init_docstring` | The docstring of this module's `__init__` method, or `None` |
| `Module.init_signature` | The signature of this module's `__init__` method, or `None` |
| `Module.init_source_file` | Path of the source file that defines this module's `__init__` method, or `None` |
| `Module.init_source_line` | Line number where this module's `__init__` method is defined, or `None` |
| `Module.init_source_location` | The `__init__` source location as `file:line`, or `None` |
| `Module.input_layers` | Labels of the layers that carry inputs into this module |
| `Module.internal_param_memory` | Bytes of parameters across this module's address subtree, including descendants |
| `Module.layer_labels` | Labels of the layers computed inside this module, without pass suffixes |
| `Module.max_descendant_depth` | Deepest nesting of module calls beneath this module |
| `Module.num_descendant_calls` | Total number of module calls nested beneath all calls of this module |
| `Module.num_edges` | Total number of internal, input and output edges of this module |
| `Module.num_input_edges` | Number of edges whose child op is inside this module and whose parent op is outside it |
| `Module.num_internal_edges` | Number of edges whose parent and child ops are both inside this module |
| `Module.num_output_edges` | Number of edges whose parent op is inside this module and whose child op is outside it |
| `Module.num_param_tensors_frozen` | Number of frozen parameter tensors owned by this module |
| `Module.num_param_tensors_trainable` | Number of trainable parameter tensors owned by this module |
| `Module.num_params_frozen` | Number of frozen parameter elements owned by this module |
| `Module.num_params_trainable` | Number of trainable parameter elements owned by this module |
| `Module.num_recursive_param_tensors` | Number of parameter tensors across this module and its address descendants |
| `Module.num_recursive_param_tensors_frozen` | Number of frozen parameter tensors across this module and its address descendants |
| `Module.num_recursive_param_tensors_trainable` | Number of trainable parameter tensors across this module and its address descendants |
| `Module.num_recursive_params` | Number of parameter records across this module and its address descendants |
| `Module.num_recursive_params_frozen` | Number of frozen parameter records across this module and its address descendants |
| `Module.num_recursive_params_trainable` | Number of trainable parameter records across this module and its address descendants |
| `Module.out_shape` | Output shape of a single-call, single-output module |
| `Module.out_shapes` | Output shapes of a single-call module, as a list |
| `Module.output_layers` | Labels of the layers that produce this module's outputs |
| `Module.output_structure` | The structure of the single call's return container, or `None` when ambiguous |
| `Module.outs` | Saved output tensors of a single-call module, as a list |
| `Module.param_memory` | Bytes of parameters at this module's own address, the same for every call |
| `Module.recursive_params` | The parameters owned by this module's address and by all of its address descendants |
| `Module.show_call_tree` | Print the call subtrees of this module as ASCII trees |
| `Module.total_backward_duration` | Total backward duration across calls, or `None` when backward timing is not available |
| `Module.total_flops_backward` | Sum of backward FLOPs across every op this module's calls ran |
| `Module.total_flops_forward` | Sum of forward FLOPs across every op this module's calls ran |
| `Module.total_forward_duration` | Sum of wall-clock forward durations across all calls of this module |
| `Module.total_func_calls_duration` | Sum of inclusive torch function durations across all calls of this module |
| `Module.total_internal_activation_memory` | Sum of internal activation bytes across this module's calls |
| `Module.total_internal_gradient_memory` | Sum of internal gradient bytes across this module's calls |
| `Module.total_macs_backward` | Always `None`, because backward MACs are not derivable |
| `Module.total_macs_forward` | True forward MACs across every op this module's calls ran, counting only ops with a known split |
| `Module.total_output_activation_memory` | Sum of output activation bytes across this module's calls |
| `Module.total_output_gradient_memory` | Sum of output gradient bytes across this module's calls |
| `Module.walk_descendants` | Yield every `ModuleCall` nested beneath any call of this module |

#### `tl.types.ModuleCall`

| Spelling | Meaning |
| --- | --- |
| `ModuleCall.backward_duration` | Backward time for this module call; currently always `None` |
| `ModuleCall.call_children` | Labels of the module calls nested directly inside this call |
| `ModuleCall.call_children_labels` | Labels of the child module calls in the dynamic call tree |
| `ModuleCall.label` | Pass-qualified label of this call, usually `address:N` |
| `ModuleCall.call_parent` | Label of the parent module call in the call tree, or `None` at the root |
| `ModuleCall.call_tree` | This call as the root of its nested `ModuleCall` subtree |
| `ModuleCall.class_name` | Class name of the module invoked by this call |
| `ModuleCall.class_qualname` | Qualified class name of the module invoked by this call |
| `ModuleCall.code_context` | List of `FuncCallLocation` frames for the call stack at the entry to `forward()`, outermost first |
| `ModuleCall.forward_arg_names` | Names of the `forward()` parameters, as resolved from its signature |
| `ModuleCall.forward_args` | Positional arguments passed to `forward()`, or `None` when not captured |
| `ModuleCall.forward_args_summary` | Short text summary of the positional `forward()` arguments |
| `ModuleCall.forward_args_template` | Structural template of the positional `forward()` arguments |
| `ModuleCall.forward_duration` | Wall-clock time in seconds spent in this call's `forward()` |
| `ModuleCall.forward_kwargs` | Keyword arguments passed to `forward()`, or `None` when not captured |
| `ModuleCall.forward_kwargs_summary` | Short text summary of the keyword `forward()` arguments |
| `ModuleCall.forward_kwargs_template` | Structural template of the keyword `forward()` arguments |
| `ModuleCall.forward_pre_hook_effects` | Tuple of `PreHookEffect` records, one per user forward pre-hook that ran on this call |
| `ModuleCall.func_calls_duration` | Total time spent in torch function calls inside this call, summed over their `Op` durations |
| `ModuleCall.grads` | Saved output gradients of this call, one per tensor output in container-path order |
| `ModuleCall.has_frozen_params` | Whether the called module owns any frozen parameters |
| `ModuleCall.has_multiple_addresses` | Whether the called module appears at more than one address |
| `ModuleCall.has_saved_forward_args` | Whether the `forward()` arguments were captured for this call |
| `ModuleCall.has_trainable_params` | Whether the called module owns any trainable parameters |
| `ModuleCall.input_layers` | Labels of the layers at the input boundary of this call |
| `ModuleCall.inputs_after_pre_hooks` | Snapshot of the module inputs after user forward pre-hooks ran, or `None` |
| `ModuleCall.inputs_before_pre_hooks` | Snapshot of the module inputs as passed to `Module.__call__`, before any pre-hook ran, or `None` |
| `ModuleCall.internal_activation_memory` | Activation bytes of the non-output `Op`s executed inside this call |
| `ModuleCall.internal_gradient_memory` | Gradient bytes of the non-output `Op`s executed inside this call |
| `ModuleCall.internal_param_memory` | Parameter bytes summed over the called module and all modules beneath it |
| `ModuleCall.max_descendant_depth` | Depth of the deepest nested `ModuleCall` beneath this call |
| `ModuleCall.num_descendant_calls` | Number of `ModuleCall`s nested under this call, counted recursively |
| `ModuleCall.num_edges` | Total number of internal, input and output edges of this call |
| `ModuleCall.num_forward_args_total` | Total number of positional and keyword arguments passed to `forward()` |
| `ModuleCall.num_forward_kwargs` | Number of keyword arguments passed to `forward()` |
| `ModuleCall.num_forward_pos_args` | Number of positional arguments passed to `forward()` |
| `ModuleCall.num_input_edges` | Number of edges whose child `Op` is inside this call and whose parent `Op` is outside |
| `ModuleCall.num_internal_edges` | Number of edges whose parent and child `Op`s are both inside this call |
| `ModuleCall.num_output_edges` | Number of edges whose parent `Op` is inside this call and whose child `Op` is outside |
| `ModuleCall.num_param_tensors` | Number of parameter tensors owned by the called module |
| `ModuleCall.num_param_tensors_frozen` | Number of frozen parameter tensors owned by the called module |
| `ModuleCall.num_param_tensors_trainable` | Number of trainable parameter tensors owned by the called module |
| `ModuleCall.ordinal_index` | 0-based position of this call among all module calls in the trace |
| `ModuleCall.out_dtype` | Output dtype of a single-output call; raises when the call has several outputs |
| `ModuleCall.out_dtypes` | Output dtypes of this call, one per tensor output in container-path order |
| `ModuleCall.out_shape` | Output shape of a single-output call; raises when the call has several outputs |
| `ModuleCall.out_shapes` | Output shapes of this call, one per tensor output in container-path order |
| `ModuleCall.output_activation_memory` | Activation bytes of the `Op`s whose outputs form this call's return value |
| `ModuleCall.output_gradient_memory` | Gradient bytes of the `Op`s whose outputs form this call's return value |
| `ModuleCall.output_layers` | Labels of the layers at the output boundary of this call |
| `ModuleCall.output_paths` | Tuple of container paths locating each tensor output inside this call's return structure |
| `ModuleCall.output_structure` | `ContainerSpec` describing the shape of this call's return container, or `None` |
| `ModuleCall.outs` | Saved output tensors of this call, one per tensor output in container-path order |
| `ModuleCall.param_memory` | Parameter bytes owned directly by the module invoked by this call |
| `ModuleCall.show_call_tree` | Print this call's subtree as an ASCII tree |
| `ModuleCall.walk_descendants` | Yield this call, if `include_self` is true, and all descendant calls in depth-first order |

#### `tl.types.Param`

| Spelling | Meaning |
| --- | --- |
| `Param.all_module_addresses` | Addresses of every module that owns this parameter, more than one when it is shared |
| `Param.backend_address` | The parameter's address as seen by the backend, normally equal to `address` |
| `Param.barcode` | Identity token for the underlying tensor, shared across aliases of the same parameter |
| `Param.co_parent_params` | Addresses of other parameters used by the same operations as this one |
| `Param.grad_dtype` | Dtype of this parameter's cached gradient, or `None` |
| `Param.grad_shape` | Shape of this parameter's cached gradient, or `None` |
| `Param.grad_memory` | Size in bytes of this parameter's cached gradient |
| `Param.grads` | Accessor over the per-backward-pass gradient increments observed for this parameter |
| `Param.has_grad` | Whether this parameter has a cached gradient |
| `Param.has_multiple_addresses` | Whether this parameter is registered at more than one address |
| `Param.has_optimizer` | Whether optimizer state was detected for this parameter, or `None` when unknown |
| `Param.is_quantized` | Whether this parameter uses a quantized dtype such as `qint8` or `quint8` |
| `Param.is_trainable` | Whether this parameter requires gradients |
| `Param.module_cls` | The live class of the owning module, or `None` when unavailable |
| `Param.param_memory` | Memory used by the parameter tensor, in bytes |
| `Param.resolver_status` | How the backend resolved this parameter, such as `"resolved"` or `"metadata_only"` |
| `Param.used_by_layers` | Labels of the `Layer`s that used this parameter |

#### `tl.types.Buffer`

| Spelling | Meaning |
| --- | --- |
| `Buffer.writer_label` | Label of the producer of the most recent write to this buffer, or `None` |
| `Buffer.final_value` | The buffer value at its last observed version, or `None` when there are no versions |
| `Buffer.grad_dtype` | Dtype of the saved gradient at the final observed version |
| `Buffer.grad_shape` | Shape of the saved gradient at the final observed version |
| `Buffer.grad_memory` | Memory used by the saved gradient at the final observed version |
| `Buffer.has_grad` | Whether the final observed version has a saved gradient |
| `Buffer.has_multiple_addresses` | Whether this buffer is registered at more than one module address |
| `Buffer.has_saved_activation` | Whether the final observed version has a saved activation |
| `Buffer.initial_value` | The buffer value at its first observed version, or `None` when there are no versions |
| `Buffer.is_overwritten` | Whether the buffer was written at least once during the forward pass |
| `Buffer.num_overwrites` | Number of write events recorded for this buffer address |
| `Buffer.value_after` | Return the buffer value after the n-th write, counting writes from 1 |
| `Buffer.value_at` | Return the buffer value at version n, counting versions from 1 |
| `Buffer.versions` | List of `Op` nodes, in order, that record each observed value of this buffer |

#### `tl.types.GradFn`

| Spelling | Meaning |
| --- | --- |
| `GradFn.backward_docstring` | Docstring of the custom autograd `backward` method, or `None` |
| `GradFn.backward_duration` | Elapsed backward time for this GradFn when it has a single measured call, or `None` when ambiguous |
| `GradFn.backward_signature` | Signature of the custom autograd `backward` method as a string, or `None` |
| `GradFn.backward_source_file` | Source file defining the custom autograd `backward` method, or `None` |
| `GradFn.backward_source_line` | Source line of the custom autograd `backward` method, or `None` |
| `GradFn.backward_source_location` | Source location of the custom autograd `backward` method as a string, or `None` |
| `GradFn.call_labels` | Pass-qualified labels of this GradFn's calls |
| `GradFn.children` | Labels of the GradFns that fire after this one in the backward pass |
| `GradFn.class_docstring` | Docstring of the grad-fn class, or `None` |
| `GradFn.class_name` | Name of the PyTorch grad-fn class |
| `GradFn.class_qualname` | Qualified name of the PyTorch grad-fn class, including its module |
| `GradFn.class_source_file` | Source file defining the grad-fn class, or `None` |
| `GradFn.class_source_line` | Source line where the grad-fn class is defined, or `None` |
| `GradFn.class_source_location` | Source location of the grad-fn class as a string, or `None` |
| `GradFn.co_parents` | Labels of the GradFns that feed the same child GradFns as this one |
| `GradFn.creator_object_id` | The `id()` of the object that created this grad-fn, or `None` |
| `GradFn.differentiates` | Label of the forward `Op` this GradFn differentiates, or `None` when unknown |
| `GradFn.forward_docstring` | Docstring of the custom autograd `forward` method, or `None` |
| `GradFn.forward_signature` | Signature of the custom autograd `forward` method as a string, or `None` |
| `GradFn.forward_source_file` | Source file defining the custom autograd `forward` method, or `None` |
| `GradFn.forward_source_line` | Source line of the custom autograd `forward` method, or `None` |
| `GradFn.forward_source_location` | Source location of the custom autograd `forward` method as a string, or `None` |
| `GradFn.grad_fn_object_id` | The `id()` of the grad-fn object at capture time, used as its key for the trace lifetime |
| `GradFn.has_children` | Whether this GradFn has children in the backward graph |
| `GradFn.has_co_parents` | Whether this GradFn has co-parents in the backward graph |
| `GradFn.has_op` | Whether a forward `Op` was captured for this GradFn |
| `GradFn.has_parents` | Whether this GradFn has parents in the backward graph |
| `GradFn.has_saved_call` | Whether any call of this GradFn has recorded call data |
| `GradFn.has_siblings` | Whether this GradFn has siblings in the backward graph |
| `GradFn.init_docstring` | Docstring of the grad-fn class `__init__`, or `None` |
| `GradFn.init_signature` | Signature of the grad-fn class `__init__` as a string, or `None` |
| `GradFn.init_source_file` | Source file defining the grad-fn class `__init__`, or `None` |
| `GradFn.init_source_line` | Source line of the grad-fn class `__init__`, or `None` |
| `GradFn.init_source_location` | Source location of the grad-fn class `__init__` as a string, or `None` |
| `GradFn.is_custom` | Whether this GradFn comes from a user-defined autograd `Function` |
| `GradFn.module_membership_source` | How this GradFn was assigned to a module, such as `"inferred"` or `"paired"`, or `None` |
| `GradFn.next_grad_fn_ids` | Object ids of the grad-fn nodes that follow this one, taken from torch's `next_functions` |
| `GradFn.order` | Integer position of this GradFn in the discovered backward traversal, or `None` when unresolved |
| `GradFn.ordinal_index` | 0-based position of this GradFn in the order grad-fns were discovered in the trace |
| `GradFn.origin_backward_pass` | Index of the backward pass in which this GradFn was first encountered, or `None` |
| `GradFn.parents` | Labels of the GradFns that fire before this one in the backward pass |
| `GradFn.siblings` | Labels of the other GradFns that share a parent with this one |
| `GradFn.total_backward_duration` | Total backward time summed over this GradFn's measured calls |
| `GradFn.type_index` | 1-based index of this GradFn among those of the same normalized type |

#### `tl.types.GradFnCall`

| Spelling | Meaning |
| --- | --- |
| `GradFnCall.backward_duration` | Measured backward time for this call, or `None` when not measured |
| `GradFnCall.backward_pass_number` | 1-based number of the backward pass this call belongs to, or `None` |
| `GradFnCall.label` | Pass-qualified label of this call, in the form `label:call_index` |
| `GradFnCall.grad_inputs` | Gradients flowing into this grad-fn, as observed by the backward hook |
| `GradFnCall.grad_outputs` | Gradients flowing out of this grad-fn, as observed by the backward hook |
| `GradFnCall.is_saved` | Whether this call has saved gradient payloads |
| `GradFnCall.ordinal_index` | 0-based position of this call in `trace.autograd.grad_fn_calls` |
| `GradFnCall.timestamp` | Wall-clock time in seconds when the hook fired, or `None` |

#### `tl.types.BackwardPass`

| Spelling | Meaning |
| --- | --- |
| `BackwardPass.backward_call_context` | Source location of the user's `backward` call that opened this pass, or `None` when it was not recorded |
| `BackwardPass.engine_flags` | Keyword flags passed to the autograd backward call, such as `create_graph` and `retain_graph`, or `None` |
| `BackwardPass.implicit` | Whether this pass was inferred from orphan tensor-hook emissions rather than an observed backward call |
| `BackwardPass.inputs_subset` | The `inputs` subset given to the backward call, as a tuple, empty when none was given |
| `BackwardPass.order` | Position of this pass in the discovered backward traversal order, or `None` when it could not be resolved |
| `BackwardPass.order_attribution_coverage` | Fraction of this pass's grad-fn calls whose traversal order was resolved, or `None` when the pass has no calls |
| `BackwardPass.ordinal_index` | 0-based position of this pass among the trace's backward passes |
| `BackwardPass.origin_backward_pass` | Index of the earlier backward pass that created the graph this pass walks, or `None` |
| `BackwardPass.outer_context` | Outer TorchLens trigger context when this pass was nested inside another, or `None` |
| `BackwardPass.peak_memory` | Peak memory observed during this pass in bytes, or `None` when not measured |
| `BackwardPass.root_grad_arguments` | The gradient arguments given to the backward call for its roots, or `None` |
| `BackwardPass.root_grad_fn_ids` | Object ids of the root grad-fn nodes this pass started from |
| `BackwardPass.root_meta` | Tuple of shape, dtype and device descriptions, one per root tensor of the backward call |
| `BackwardPass.save_grads_policy` | Text form of the gradient-saving policy active during this pass, or `None` |
| `BackwardPass.trigger` | The entry point that opened this backward pass, as a string |

#### `tl.types.FuncCallLocation`

| Spelling | Meaning |
| --- | --- |
| `FuncCallLocation.call_line` | The exact source line that ran this frame, or `''` when the text is unavailable. Without `save_source_context=True` the text is read from the file when accessed, so it reflects the file on disk at read time, and only while the file keeps the size and modification time it had at capture |
| `FuncCallLocation.code_context` | The source lines around the call site as a list, or `None` when source is unavailable (read lazily like `call_line`) |
| `FuncCallLocation.code_context_labeled` | The surrounding source context with the call line marked |
| `FuncCallLocation.code_firstlineno` | First source line of the function that contains the call |
| `FuncCallLocation.col_offset` | Column offset of the call within its source line, or `None` when unknown |
| `FuncCallLocation.func_docstring` | Docstring of the called function, or `None` when unavailable |
| `FuncCallLocation.func_qualname` | Qualified name of the called function, or `None` when unknown |
| `FuncCallLocation.func_signature` | Signature of the called function as a string, or `None` when unavailable |
| `FuncCallLocation.line_number` | One-indexed source line of the call |
| `FuncCallLocation.num_context_lines` | Number of source context lines actually loaded around the call |
| `FuncCallLocation.source_context` | The surrounding source context as a single string |
| `FuncCallLocation.source_loading_enabled` | Whether this location holds a source-text snapshot (`True` by default; captures set it from `save_source_context`). When `False` the text is read lazily from the file, and `repr()` names the reason when it cannot be shown (`source changed on disk since capture`, `source file not found`) |
| `FuncCallLocation.to_html_link` | Return a VS Code source-location link for this call, for HTML renderers |

#### `tl.types.PreHookEffect`

| Spelling | Meaning |
| --- | --- |
| `PreHookEffect.after_fingerprint` | Metadata-only fingerprint of the inputs after the hook ran |
| `PreHookEffect.before_fingerprint` | Metadata-only fingerprint of the inputs before the hook ran |
| `PreHookEffect.change_kinds` | Tuple of short strings naming the kinds of change the hook made to the inputs |
| `PreHookEffect.changed_paths` | Tuple of typed input paths that the hook changed |
| `PreHookEffect.exception_message` | Message of the exception that interrupted the hook, or `None` |
| `PreHookEffect.hook` | Descriptive metadata about the hook, as a `HookInfo` |
| `PreHookEffect.incomplete_reasons` | Tuple of stable reason strings for why change evidence or attribution was downgraded |
| `PreHookEffect.registration_id` | PyTorch's registry identifier for the hook, kept for diagnostics |
| `PreHookEffect.with_kwargs` | Whether PyTorch invoked the hook with keyword arguments |

#### `tl.types.ModuleInputSnapshot`

| Spelling | Meaning |
| --- | --- |
| `ModuleInputSnapshot.incomplete_reasons` | Tuple of stable reason strings explaining why the evidence is incomplete |
| `ModuleInputSnapshot.input_structure` | Compact immutable description of the input container topology |
| `ModuleInputSnapshot.tensor_observations` | Ordered `TensorInputObservation`s of the input tensors, keyed by typed input path |

#### `tl.types.TensorInputObservation`

| Spelling | Meaning |
| --- | --- |
| `TensorInputObservation.alias_group` | Snapshot-local identity shared by input paths that alias the same tensor |
| `TensorInputObservation.payload_origin` | Description of how the retained payload was obtained |
| `TensorInputObservation.payload_status` | `"saved"`, or the reason no payload is available |
| `TensorInputObservation.source_op` | Label of the final producer `Op` of this tensor, or `None` when unavailable |

#### `tl.recording.RecordContext`

| Spelling | Meaning |
| --- | --- |
| `RecordContext.backend_requires_isolation` | Whether predicate evaluation must be isolated for backend safety |
| `RecordContext.bool_value` | Scalar boolean value when it can be observed safely, otherwise `None` or a deferred value |
| `RecordContext.event_index` | Chronological index of this event within the active capture |
| `RecordContext.input_output_address` | Structural address of an input or output boundary event, or `None` |
| `RecordContext.is_bottom_level_func` | Whether the event came from a leaf decorated function call, or `None` when not applicable |
| `RecordContext.is_output_parent` | Whether this event is a direct parent of a model output |
| `RecordContext.is_scalar_bool` | Whether the output is a scalar boolean, possibly a deferred value |
| `RecordContext.is_transform` | Whether this event is a captured transform boundary |
| `RecordContext.label_prefix` | Short raw label alias for the event, for example `"relu_1"` |
| `RecordContext.label_raw` | The full raw label of the event, or an empty string when unavailable; never the final label |
| `RecordContext.module_pass_number` | Call index of the nearest active module, or `None` |
| `RecordContext.module_stack` | Active module frames from outermost to innermost |
| `RecordContext.output_of_module_calls` | Labels of the module calls whose output this event is, empty when it is not a module-call output |
| `RecordContext.parent_labels` | Parent labels exposed to the predicate |
| `RecordContext.parent_labels_raw` | Full raw parent labels, when the backend records them separately |
| `RecordContext.raw_index` | One-based operation index used in raw graph labels, or `None` |
| `RecordContext.recent_events` | Bounded chronological lookback of earlier events, as `RecordContext`s |
| `RecordContext.recent_ops` | Operation-only view of the bounded lookback, as `RecordContext`s |
| `RecordContext.sample_id` | Identifier supplied by a batched predicate caller, or `None` |
| `RecordContext.tensor_device` | Backend-neutral reference to the output tensor's device, or `None` |
| `RecordContext.tensor_requires_grad` | Whether the output tensor requires gradients, `None` when unknown, or a deferred value |
| `RecordContext.time_since_pass_start` | Wall-clock seconds elapsed since capture began |
| `RecordContext.type_index` | One-based occurrence index of this event within its `layer_type`, or `None` |
| `RecordContext.window_miss` | Whether a history-dependent selector looked past the retained lookback window |

#### `tl.recording.ModuleStackFrame`

| Spelling | Meaning |
| --- | --- |
| `ModuleStackFrame.module_id` | The `id()` of the module object at this stack frame |

#### `tl.stats.HistogramDescriptor`

| Spelling | Meaning |
| --- | --- |
| `HistogramDescriptor.base` | Logarithm base of the bin grid; only `2` is supported |
| `HistogramDescriptor.bins_per_octave` | Number of bins per factor of two in magnitude |
| `HistogramDescriptor.bins_per_side` | Number of magnitude bins on each sign side |
| `HistogramDescriptor.bucket_edges` | Return the ascending magnitude bin edges for one sign side |
| `HistogramDescriptor.hi_exp` | Upper edge of the grid as a power of the base |
| `HistogramDescriptor.lo_exp` | Lower edge of the grid as a power of the base |

#### `tl.stats.HistogramResult`

| Spelling | Meaning |
| --- | --- |
| `HistogramResult.counts_fit_int32` | Return whether every stored count fits the signed thirty-two-bit integer used by the observation encoding |
| `HistogramResult.neg_counts` | Per-bin counts for the negative side, in ascending magnitude |
| `HistogramResult.pos_counts` | Per-bin counts for the positive side, in ascending magnitude |
| `HistogramResult.specials` | Exact counts of zeros, per-side underflow and overflow, NaN and infinities, keyed by name |

<!-- trace group views: generated by export_groups.py -->
**`tl.types.TraceCost`**
: Compute, memory and timing totals of a Trace; `trace.cost` is the instance a Trace serves. `TraceCost.backward_durations`: seconds per backward pass; `last_backward_duration` / `total_backward_duration` (@properties); `TraceCost.backward_peak_memory`: peak memory during forward / backward. On CPU/MPS this is only the cheap host RSS (or MPS allocator) delta and legitimately reads `0` for small models; `CaptureOptions(measure_python_peak_memory=True)` opts into a `tracemalloc` Python-allocation peak at 1.7x-2.5x capture-time cost (session-time only, never persisted); `TraceCost.flops_by_op_type`: group FLOPs by layer type; `TraceCost.forward_duration`: phase timings (`func_calls_duration` = pure model compute proxy); `TraceCost.forward_peak_memory`: peak memory during forward / backward. On CPU/MPS this is only the cheap host RSS (or MPS allocator) delta and legitimately reads `0` for small models; `CaptureOptions(measure_python_peak_memory=True)` opts into a `tracemalloc` Python-allocation peak at 1.7x-2.5x capture-time cost (session-time only, never persisted); `TraceCost.forward_peak_memory_pair`: session-time (live, resident) forward peak pair with backend named; `TraceCost.forward_peak_memory_reason`: why the forward peak memory is absent or estimated, if it is; `TraceCost.func_calls_duration`: phase timings (`func_calls_duration` = pure model compute proxy); `TraceCost.last_backward_duration`: most recent backward-pass duration, if any; `TraceCost.macs_by_kind`: group TRUE forward MACs by layer type; `TraceCost.overhead_duration`: time spent on TorchLens overhead (total minus function calls); `TraceCost.pinned_autograd_memory`: bytes of autograd memory pinned by retained saved tensors; `TraceCost.saved_activation_memory`: bytes for all / actually-saved Op activations; `TraceCost.saved_grad_memory`: bytes for all / actually-saved output gradients; `TraceCost.total_activation_memory`: bytes for all / actually-saved Op activations; `TraceCost.total_autograd_memory`: bytes autograd retained for backward; `TraceCost.total_backward_duration`: sum of all captured backward-pass durations; `TraceCost.total_backward_memory`: total bytes of backward activations retained; `TraceCost.total_flops`: total FLOPs (forward + backward); `TraceCost.total_flops_backward`: total backward FLOPs across all layers (skipping None/unknown); `TraceCost.total_flops_forward`: total forward FLOPs (the canonical identity-partition aggregation); `TraceCost.total_grad_memory`: bytes for all / actually-saved output gradients; `TraceCost.total_macs`: total MACs are not derivable and always ``None`` (see total_macs_backward); `TraceCost.total_macs_backward`: backward MACs are not derivable and always ``None`` (estimates only); `TraceCost.total_macs_forward`: total TRUE forward MACs over ops with a known MAC split; `TraceCost.total_param_grad_memory`: bytes for parameter tensors / populated parameter gradients; `TraceCost.total_param_memory`: bytes for parameter tensors / populated parameter gradients.

**`tl.types.TraceGraph`**
: Graph-shape facts of a Trace: edges, degrees, branching and conditionals; `trace.graph` is the instance a Trace serves. `TraceGraph.branching_factor`: mean fan-out: children (consumers) per compute Op; `TraceGraph.conditional_arm_entry_edges`: entry edges of each conditional arm, keyed by conditional; `TraceGraph.conditional_branch_edges`: edges that branch on a conditional; `TraceGraph.conditional_edge_call_indices`: call indices of the ops on each conditional edge; `TraceGraph.conditional_records`: raw conditional-flow records in trace order; `TraceGraph.conditionals`: conditionalAccessor (by int ordinal or stable label); `TraceGraph.edges`: return this trace's dataflow edge family (finalized edge views); `TraceGraph.has_conditional_branching`: whether any layer is in a conditional branch; `TraceGraph.has_conditionals`: whether this Trace contains conditional-flow records; `TraceGraph.is_branching`: whether any layer has more than one child; `TraceGraph.is_dynamic`: whether execution depends on runtime tensor values; `TraceGraph.is_recurrent`: whether any layer has more than one pass; `TraceGraph.manual_tensor_connections`: user-declared tensor connections added to the graph; `TraceGraph.max_in_degree`: maximum number of parents over compute Ops; `TraceGraph.max_ops_per_layer`: maximum number of ops for any layer; `TraceGraph.max_out_degree`: maximum number of children over compute Ops; `TraceGraph.num_buffer_edges`: distinct Op graph edges with at least one buffer endpoint; `TraceGraph.num_compute_edges`: distinct Op graph edges whose endpoints are both compute Ops; `TraceGraph.num_layer_edges`: distinct edges in the aggregate Layer graph; `TraceGraph.op_equivalence_classes`: dict of equivalence-class id → set of Op labels; `TraceGraph.orphans`: access retained orphan island operations by raw or final label; `TraceGraph.relationship_evidence`: evidence behind each inferred op relationship; `TraceGraph.stop_frontier`: early-halt state (partial trace); `TraceGraph.structure_evidence`: evidence behind the structure-only reconstruction; `TraceGraph.tied_param_groups`: alias-address groups of parameters registered at more than one path.

**`tl.types.TraceAutograd`**
: Backward-pass records and grad_fn facts of a Trace; `trace.autograd` is the instance a Trace serves. `TraceAutograd.backward_passes`: access backward pass records by 0-based position or named pass number; `TraceAutograd.backward_root_grad_fn_object_ids`: root grad-fn `object_id` per backward pass; `last_backward_root_grad_fn_object_id`; `TraceAutograd.derived_grads`: derivedGradAccessor for leaf-level derived gradients (non-torch backends / explicit derivation); `TraceAutograd.disarm`: detach this Trace from global autograd backward interception so plain backward calls no longer record into it; `TraceAutograd.escape_detector_backward_coverage`: backward coverage reported by the escape detector; `TraceAutograd.grad_fn_calls`: access per-invocation GradFnCall records by qualified label or index; `TraceAutograd.grad_fn_fire_timings`: return live per-fire backward timing spans keyed like ``grad_fn_calls``; `TraceAutograd.grad_fn_order`: grad_fn labels in backward execution order; `TraceAutograd.grad_fns`: access backward grad_fn_handle metadata by label, index, pass label, or substring; `TraceAutograd.has_backward_pass`: backward data / gradient values present; `TraceAutograd.has_grads`: backward data / gradient values present; `TraceAutograd.intermediate_derived_grads`: access exact op-level derived gradient records, when a backend provides them; `TraceAutograd.last_backward_pass`: return the most recent backward pass record, if any; `TraceAutograd.last_backward_root_grad_fn_object_id`: most recent backward root grad_fn_handle object id, if any; `TraceAutograd.num_backward_edges`: distinct edges in the backward GradFn graph, or ``None`` if ungated; `TraceAutograd.saved_grad_fn_calls`: access GradFnCall records with saved gradient inputs or outputs; `TraceAutograd.saved_grad_fns`: access GradFns containing at least one saved GradFnCall.

**`tl.types.TraceSource`**
: Where the traced model's source lives: the class, `__init__` and `forward` locations (file, line, combined location), the forward signature, the three docstrings, the captured frames, and the live class while the model is still alive; `trace.source` is the instance a Trace serves. `TraceSource.class_docstring`: model class location and docs; `TraceSource.class_file`: model class location and docs; `TraceSource.class_line`: model class location and docs; `TraceSource.class_location`: combined model class source location; `TraceSource.class_name`: short Python class name of the source model; `TraceSource.class_qualname`: fully qualified class name; useful for run verification; `TraceSource.cls`: return the live source model class when the model is still alive; `TraceSource.forward_docstring`: `forward()` info; `TraceSource.forward_file`: `forward()` info; `TraceSource.forward_line`: `forward()` info; `TraceSource.forward_location`: combined forward source location; `TraceSource.forward_signature`: `forward()` info; `TraceSource.frames`: python call-stack frames at capture time (`list[FuncCallLocation]`); `TraceSource.init_docstring`: `__init__` info; `TraceSource.init_file`: `__init__` info; `TraceSource.init_line`: `__init__` info; `TraceSource.init_location`: combined model ``__init__`` source location; `TraceSource.init_signature`: `__init__` info; `TraceSource.snapshots`: source snapshots taken for the traced model.

**`tl.types.TraceProvenance`**
: How, when and from what this Trace was captured; `trace.provenance` is the instance a Trace serves. `TraceProvenance.backend`: `Literal["torch","tf","jax","mlx","paddle","tinygrad"]` capture backend tag. Portable; `TraceProvenance.backend_config`: backend runtime snapshot captured at trace time; `TraceProvenance.backend_version`: backend runtime snapshot captured at trace time; `TraceProvenance.backward_memory_backend`: backend used to measure backward memory; `TraceProvenance.cache_hit`: whether this capture was served from the capture cache; `TraceProvenance.cache_key`: key of the capture-cache entry this capture used or wrote; `TraceProvenance.cache_path`: path of the capture-cache entry, when one was used; `TraceProvenance.cleanup_duration`: phase timings (`func_calls_duration` = pure model compute proxy); `TraceProvenance.distributed_scope`: distributed scope of the capture (rank and world facts), if any; `TraceProvenance.duration`: total elapsed. `overhead_duration`: TorchLens overhead (capture minus func calls); `TraceProvenance.end_time`: epoch timestamps; `TraceProvenance.fingerprint`: hash of captured topology + shape-relevant structure; `TraceProvenance.forward_memory_backend`: memory backend that measured the forward peak; `TraceProvenance.input_annotations`: user-attached metadata (Trace-wide / inputs); `TraceProvenance.input_object_id`: python `id()` of the input at capture. Within-session only; `TraceProvenance.input_preprocessor`: descriptor of the input preprocessor applied before capture, if any; `TraceProvenance.input_signature_hash`: hash of input shapes/dtypes/devices; `TraceProvenance.input_structure`: structural descriptor of the model input (shape/dtype/pytree spec); `TraceProvenance.is_appended`: chunked-append provenance; `TraceProvenance.jax_closed_jaxpr`: closed jaxpr of the traced JAX function; `TraceProvenance.jax_inlined_call_primitives`: JAX call primitives inlined during capture; `TraceProvenance.kind`: kind of capture (live, cooked from a Recording, loaded); `TraceProvenance.model_object_id`: python `id()` of the model at capture time. Within-session only; `TraceProvenance.observer_spans`: observer spans recorded during the capture; `TraceProvenance.output_postprocessor`: output-decoding descriptors; `TraceProvenance.owner_thread_id`: thread id of the thread that owned the capture; `TraceProvenance.owner_thread_qualified`: whether the owner thread was fully qualified for the capture; `TraceProvenance.param_hash_full`: parameter-value hashes; `TraceProvenance.param_hash_quick`: parameter-value hashes; `TraceProvenance.param_source`: how params were derived — `"native-module"`, `"pytree-derived"`, `"none"`; `TraceProvenance.recording_kept`: whether the capture's Recording was kept; `TraceProvenance.root_entry_point`: persisted identity fact disclosing a bound-method capture root (`"bound_method:<owner>.<method>"`), absent on ordinary module-root captures; `TraceProvenance.setup_duration`: phase timings (`func_calls_duration` = pure model compute proxy); `TraceProvenance.start_time`: epoch timestamps; `TraceProvenance.summary()`: the capture-provenance block (the relocated summary preamble); `TraceProvenance.thread_activity_detected`: whether other threads were active during the capture; `TraceProvenance.thread_count_end`: thread count when the capture ended; `TraceProvenance.thread_count_start`: thread count when the capture started; `TraceProvenance.verification_reason`: machine-readable reason a live capture is not positively verified; `TraceProvenance.verified`: whether the live capture is positively verified (``None`` when not recorded).

**`tl.types.TraceSettings`**
: The capture options this Trace ran with (the stored snapshot); `trace.options` is the instance a Trace serves. `TraceSettings.activation_transform`: callable applied to saved forward outputs before storage; `TraceSettings.backward_ready`: backward machinery prepared at capture. `inference_only`: captured under `torch.no_grad()`; `TraceSettings.chunk_size`: chunked-append provenance; `TraceSettings.detach_saved_activations`: saved tensors detached from autograd; `TraceSettings.emit_nvtx`: capture toggles; `TraceSettings.grad_transform`: callable applied to saved gradients; `TraceSettings.grouping`: `"structural"` (default), `"strict_shapes"`/`"fold_sites"` (reserved, refuse typed pre-ratification); `TraceSettings.grouping_policy`: stamp of the layer-grouping policy that produced the layers; `TraceSettings.inference_only`: whether the capture ran under inference mode; `TraceSettings.input_transform`: portable `repr()` of the input / gradient transform callables. (`gradient_transform` ruled v12; not yet shipped -- the field itself is still `grad_transform`.); `TraceSettings.jax_static_argnums`: static argument positions passed to the JAX capture; `TraceSettings.keep_orphans`: whether orphan island Ops are retained; `TraceSettings.mark_layer_depths`: capture toggles; `TraceSettings.mode`: `"exhaustive"`, `"fast"`, or `"predicate"` (the capture-mechanism word); `TraceSettings.module_identity_mode`: how modules are identified — `"torch_module"`, `"pytree_module"`, `"function_root"`, `"object_module"`; `TraceSettings.on_nonfinite`: capture toggles; `TraceSettings.output_device`: device where saved outputs are placed. `save_mode`: tensor retention mode (`"copy"`, `"reference"`, `"view"`, `"cpu_async"`); `TraceSettings.recurrence_detection`: loop/recurrence detection on/off; `TraceSettings.save`: the save selector the capture ran with; `TraceSettings.save_arg_templates`: deep-copied Op arguments / structural arg templates; `TraceSettings.save_args`: deep-copied Op arguments / structural arg templates; `TraceSettings.save_grads`: whether operation gradients are retained after backward; `TraceSettings.save_mode`: how saved activations were retained (``copy`` or ``reference``); `TraceSettings.save_raw_activations`: whether raw untransformed tensors are saved; `TraceSettings.save_raw_grads`: whether raw untransformed tensors are saved; `TraceSettings.save_raw_input`: raw-I/O save policy + batch rendering policy for visualization; `TraceSettings.save_rng`: RNG snapshots. `save_code_context`: source-code context; `TraceSettings.save_source_context`: whether source text was snapshotted at capture for each frame (otherwise it is read lazily from disk while the file is unchanged); `TraceSettings.save_transformed_output`: raw-I/O save policy + batch rendering policy for visualization; `TraceSettings.seed`: RNG seed applied for capture, if any; `TraceSettings.source_context_lines`: number of source-context lines saved per op (stored as ``num_context_lines`` in v11); `TraceSettings.structure_only`: see Capture products above; `TraceSettings.verbose`: capture toggles.

**`tl.types.TraceHealth`**
: Honesty diagnostics of the capture: witness and escape-detector facts; `trace.health` is the instance a Trace serves. `TraceHealth.completeness_callback_ns`: time spent in the completeness-witness callback, in nanoseconds; `TraceHealth.completeness_decompositions`: completeness-witness decompositions of the captured op stream; `TraceHealth.completeness_mode`: mode the completeness witness ran in; `TraceHealth.completeness_verified`: whether the completeness witness verified the capture; `TraceHealth.escape_detector_callback_ns`: time spent in the escape-detector callback, in nanoseconds; `TraceHealth.escape_detector_mode`: mode the escape detector ran in; `TraceHealth.escape_detector_verified`: whether the escape detector verified the capture; `TraceHealth.num_accounted_witness_events`: number of witness events accounted for by captured ops; `TraceHealth.num_escape_detector_events`: number of escape-detector events observed; `TraceHealth.num_expected_opaque_witness_events`: number of witness events expected to be opaque; `TraceHealth.num_unaccounted_witness_events`: number of witness events no captured op accounts for; `TraceHealth.num_witness_events`: number of completeness-witness events observed.

<!-- end trace group views -->

### Records built by TorchLens
Records are built by TorchLens: the classes below, and every class of a family row (`tl.errors.*`:
every exception and warning), have no public constructor keywords; a caller reads each value as the
attribute of the same name. No keyword row names a record class (`tests/test_glossary_params.py`
fails on one that does), and the lockstep gate owes no glossary entry for a keyword that only
record constructors take; `PYTHONPATH=tools python -m name_lockstep.glossary_backfill --prune-records`
drops a newly listed record from the keyword rows.

| Spelling | Meaning |
| --- | --- |
| `tl.errors.*` | every exception and warning TorchLens raises; a caller reads `.args` and the typed attributes, never builds one |
| `tl.Edit` | Portable identity and hook factory for helper-built interventions |
| `tl.Layer` | Aggregate per-layer metadata for a logged model operation |
| `tl.Op` | Metadata for a single tensor operation (one pass of one layer) |
| `tl.Recording` | Result of a fastlog recording session, built by `tl.record` and `Recorder` |
| `tl.Selection` | Frozen, concrete, trace-bound selection: an ordered tuple of Site |
| `tl.Trace` | Top-level container for a logged forward pass |
| `tl.attribution.AttributionGraph` | Direct linear effects between features, errors, tokens and logits |
| `tl.attribution.Attribution` | Container for provisional input-attribution results |
| `tl.attribution.CausalAuditSet` | A set of units ablated together: the top-k claim or a control |
| `tl.attribution.CausalAuditUnit` | One unit of the audited site, ablated alone |
| `tl.attribution.CausalAudit` | The outcome of a causal audit of one internal attribution |
| `tl.attribution.CurveStep` | One point of a deletion or insertion curve |
| `tl.attribution.GraphNode` | One node of an attribution graph |
| `tl.attribution.PerturbationCurve` | A deletion or insertion curve, its area and AOPC |
| `tl.attribution.RandomizationStep` | One randomization of the test |
| `tl.attribution.RandomizationTest` | The outcome of a model-parameter randomization test |
| `tl.attribution.ReadRow` | One read result row; immutable; honesty fields are not optional |
| `tl.attribution.ReadTable` | Immutable trace-bound mapping of read rows |
| `tl.attribution.TokenAttributionPayload` | Renderer-neutral token-score payload (the tviz contract) |
| `tl.attribution.TokenAttribution` | Frozen result of the token-attribution two-liner |
| `tl.bridge.brain_score.ScoreParts` | A benchmark score with its raw and ceiling-normalised values named apart |
| `tl.bridge.pyreft.LoreftArtifact` | One pyreft LoReFT intervention, as read from its saved directory |
| `tl.bridge.sae.SpliceResult` | Settled record of one SAE splice experiment |
| `tl.bridge.transformers.AttentionCall` | One recorded attention call |
| `tl.bridge.transformers.AttentionRecording` | Per-head attention probabilities recorded by the TorchLens backend |
| `tl.bridge.treescope.BridgeStatus` | One `status` snapshot |
| `tl.bundle.SuperAtenOp` | Positional, coverage-qualified alignment of one ATen slot across members |
| `tl.bundle.SuperOp` | View of a single site across all bundle members |
| `tl.dataset_extraction.BatchEnvelope` | One collated batch, typed |
| `tl.dataset_extraction.LoadedExtraction` | A dataset-extraction artifact read back with its self-description |
| `tl.dataset_extraction.RaggedBatch` | One shard's trimmed carrier for one output key (the Arrow triplet) |
| `tl.dataset_extraction.TokenRows` | Stored keys flattened to one row per token (DOCUMENTED-UNSTABLE) |
| `tl.debug.BackwardNanReport` | Result of `tl.debug.bisect_nan_backward`: the backward pass walked, the earliest provable non-finite birth and the full transition ledger |
| `tl.debug.BackwardTransition` | One backward node execution, classified |
| `tl.debug.BisectNanReport` | Result of `tl.debug.bisect_nan` |
| `tl.debug.BisectPrecisionReport` | Result of `tl.debug.bisect_precision` |
| `tl.debug.CheckFinding` | One finding recorded by a training-loop check |
| `tl.debug.CheckReport` | The findings a `tl.debug.ChecksSession` collected, as a report |
| `tl.debug.ClipLedgerSnapshot` | Point-in-time read of a `tl.debug.ClipLedger` |
| `tl.debug.ClipLedger` | Ledger of gradient-clipping events across steps |
| `tl.debug.DTypeRangeReport` | Coverage-bounded dtype range audit result |
| `tl.debug.DeterminismReport` | Typed result of `check_determinism` |
| `tl.debug.FindNanReport` | Structured first-non-finite diagnostic returned by `find_nan` |
| `tl.debug.FirstBadThing` | One first-bad-thing finding in the shared diagnostic vocabulary |
| `tl.debug.GradFnNode` | One node of a structure-only `grad_fn` sketch |
| `tl.debug.GradFnSketch` | Structure-only autograd-graph sketch of already-computed tensors |
| `tl.debug.GuessInputShapeReport` | Result of `tl.debug.guess_input_shape` |
| `tl.debug.LineageReport` | Result of `tl.debug.lineage` |
| `tl.debug.NonfiniteReport` | One-call NaN forensics report |
| `tl.debug.OptimizerFacts` | Adapter-read facts about one optimizer; `is_known` gates every derived band |
| `tl.debug.ParamHealthReport` | Notebook-friendly parameter-space audit (mirrors `HealthReport`) |
| `tl.debug.ScaleLedgerSnapshot` | Point-in-time read of a `tl.debug.ScaleLedger` |
| `tl.debug.ScaleLedger` | Ledger of grad-scaler scale and skipped-step history across steps |
| `tl.debug.ScanRow` | Health facts for one unique tensor in a parameter scan; `aliases` keeps every other name bound to the same tensor |
| `tl.debug.WatchdogSnapshot` | Point-in-time read of a `tl.debug.Watchdog` |
| `tl.differential.DiffRow` | One settled position of the differential projection |
| `tl.distributed.ArmingRecord` | What `arm()` established, stamped on every boundary record |
| `tl.distributed.GroupIdentity` | Resolved lifetime identity of one process group on this rank |
| `tl.experiment.EffectsView` | Read view over one persisted MemberEffectTable (item 8) |
| `tl.experiment.GridCell` | One cell of a `SweepGrid`: coordinates, status, value, error and member name |
| `tl.facets.AbsenceReason` | Reason a declared facet is unavailable in the current trace |
| `tl.facets.AttentionHeadView` | Scoped accessor for one attention head within a parent facet view |
| `tl.facets.FacetRegistrySnapshot` | Immutable facet recipe set captured for one trace |
| `tl.facets.FacetView` | Lazy dict-like and attribute-access semantic view on a TorchLens record |
| `tl.facets.MissingFacet` | Sentinel for a known facet whose value is intentionally unavailable |
| `tl.facets.MissingGrad` | Sentinel for a known facet gradient that is not currently saved |
| `tl.features.ShapingRecord` | Provenance of one shaping application (rides sidecars and adapters) |
| `tl.features.SiteMatrix` | One site's feature matrix with its row identity and provenance |
| `tl.intervention.BindReport` | The out-of-band ledger for ONE bound call (surgery 3.3 item 3) |
| `tl.intervention.BoundModel` | The bound intervention executor: one spec, one model, changing neither |
| `tl.intervention.FireRecord` | Runtime record for one intervention firing |
| `tl.intervention.HookContext` | Frozen metadata snapshot passed to hook callables |
| `tl.intervention.RegionInstance` | One pass instance of a region (a weakly-connected member component) |
| `tl.intervention.RegionTarget` | A frozen admissible region: members + typed boundary + instances |
| `tl.intervention.SamplingRecord` | One realized draw of a stochastic edit, exported beside `FireRecord` (a `FireRecord.sampling` entry): a frozen record with `verb`, `base_seed`, `derived_seed`, `share_draw`, `logical_firing_coordinate`, `eligible_count`, `population_count`, `population_identity`, `donor_group_id`, `draw_indices` and `realized_draw_digest`; the field rows below define each |
| `tl.inventory.SiteInventory` | The structured site inventory (tvscope B6) |
| `tl.inventory.SiteRow` | One extractable site |
| `tl.io.AnalysisOnlySidecar` | Unvalidated view of a persisted sidecar whose provider is absent |
| `tl.io.SidecarFamily` | One registered sidecar family's declared contract |
| `tl.mechinterp.AliasResolution` | One resolved alias: native coordinates + honest capability status |
| `tl.mechinterp.ComponentEdge` | One edge `sender->receiver` of the component graph |
| `tl.mechinterp.ComponentGraph` | The component edge graph of one trace (built by `component_graph`) |
| `tl.mechinterp.ComponentReceiver` | One residual reader (a graph target) |
| `tl.mechinterp.ComponentRow` | One component row: a coordinate plus its tensor value |
| `tl.mechinterp.ComponentSender` | One residual writer (a graph source node) |
| `tl.mechinterp.ComponentStack` | An ordered, graded stack of component rows summing to a target |
| `tl.mechinterp.ContributionScores` | Per-component direct-logit contribution rows |
| `tl.mechinterp.HeadScores` | Per-head taxonomy scores |
| `tl.mechinterp.HeadWeightViews` | Head-indexed projection weight views for one attention module |
| `tl.mechinterp.LogitLensPrediction` | Streaming per-layer prediction summary (one row per projected layer) |
| `tl.mechinterp.LogitLensPredictions` | Result of a streaming `logit_lens_predictions` sweep |
| `tl.mechinterp.LogitLens` | Result of a `logit_lens` sweep across transformer blocks |
| `tl.mechinterp.PatchGrid` | A labelled patch grid: values + coordinates + receipts |
| `tl.mechinterp.PatchedRun` | The receipted product of one lowered counterfactual |
| `tl.mechinterp.PromptInspection` | Structured prompt inspection (record first, rendering second) |
| `tl.mechinterp.TrajectoryStats` | Per-layer trajectory statistics folded over every example and position |
| `tl.neuro.ExtractionPlan` | Immutable measured plan for one whole-model extraction sweep |
| `tl.neuro.PlanSite` | One requested site's measured plan row |
| `tl.observers.TapRecord` | One observed tensor value |
| `tl.preprocessing.AuditFinding` | One comparable field's audit row |
| `tl.preprocessing.DiagnosticFinding` | One opt-in tensor check's outcome |
| `tl.preprocessing.InputDiagnostics` | Opt-in tensor findings (record three of three) |
| `tl.preprocessing.PreprocessingComparison` | The field-level audit report (record two of three) |
| `tl.preprocessing.Resolution` | One authority resolution: provenance record + declared fields + transform |
| `tl.receptive_field.AmbiguousInputError` | Raised when a query has more than one reachable model input |
| `tl.receptive_field.EmpiricalAdjointCheck` | One sampled comparison of receptive and projective empirical derivatives |
| `tl.receptive_field.ReceptiveFieldAxis` | Geometric receptive-field description for one model-input axis |
| `tl.receptive_field.ReceptiveFieldBoxAxis` | Concrete receptive-field bounds for one model-input axis |
| `tl.receptive_field.ReceptiveFieldBox` | Concrete per-unit geometric receptive-field answer |
| `tl.receptive_field.ReceptiveFieldProfile` | Tabular receptive-field report at one trace granularity |
| `tl.receptive_field.ReceptiveFieldRuleContext` | Metadata and result constructors available to receptive-field rules |
| `tl.receptive_field.ReceptiveFieldValidation` | Tri-state receptive-field geometric-versus-gradient validation result |
| `tl.receptive_field.ReceptiveFieldVerification` | Combined containment and empirical-adjoint diagnostic report |
| `tl.receptive_field.ReceptiveFieldView` | Lazy query surface for one operation's per-input receptive fields |
| `tl.receptive_field.ReceptiveFieldViolation` | One gradient-support coordinate outside its geometric box |
| `tl.receptive_field.ReceptiveField` | Per-operation, per-input geometric receptive-field descriptor |
| `tl.recording.ActivationRecord` | One retained fastlog event |
| `tl.recording.DryRun` | Predicate dry-run trace without retained tensor payloads |
| `tl.recording.GradRecordContext` | Predicate input schema for one fastlog backward gradient event |
| `tl.recording.GradRecord` | One retained fastlog gradient event |
| `tl.recording.ModuleStackFrame` | One module frame in a predicate record context stack |
| `tl.recording.RecordContext` | Predicate input schema for one chronological capture event |
| `tl.repgeom.DisagreeingPair` | Two sites that two or more metrics order oppositely |
| `tl.repgeom.MetricDisagreement` | Per-metric rank orders of sites and the pairs where metrics disagree |
| `tl.repgeom.ScoreInterval` | A point score with its stimulus-bootstrap interval beside it |
| `tl.report.BackwardEstimate` | The named-door training-backward counterfactual: HYPOTHETICAL |
| `tl.report.BackwardStatus` | The always-visible backward STATUS facts |
| `tl.report.CapabilityCard` | The registry filtered by the object in hand: metadata-only |
| `tl.report.CapabilityRow` | One capability-card row: THIS object's answer for one surface |
| `tl.report.CaptureFacts` | Capture honesty facts the report carries detached |
| `tl.report.CleanStepTime` | One opt-in clean-step measurement: method fully disclosed |
| `tl.report.CostTierMeasurement` | One measured tier: absolute time first, ratio derived |
| `tl.report.CostTreeRow` | One accounting-unit row of the cost tree |
| `tl.report.CostTree` | The cost tree: rows in render order plus the invariant denominators |
| `tl.report.DevicePeaks` | An explicit peaks catalog: constructed, never auto-selected |
| `tl.report.DiffReport` | The same class as `tl.differential.DiffReport` |
| `tl.report.DoctorCheck` | One diagnostic row; constructor keywords are outside the public contract |
| `tl.report.DoctorReport` | Environment and optional model compatibility report |
| `tl.report.FlopsCrossCheck` | Side-by-side dispatch-formula evidence (never an oracle) |
| `tl.report.FlopsReport` | The detached one-call cost report |
| `tl.report.HealthFacts` | Immutable normalized health observations for one capture |
| `tl.report.HealthReport` | Notebook-friendly report from trace-local health diagnostics |
| `tl.report.InterventionValidationReport` | Five-axis intervention validation result |
| `tl.report.KernelProfileReport` | One door call: one execution, one trace, one joined result |
| `tl.report.MachineBalance` | A roofline ceiling pair and its ridge point: basis disclosed |
| `tl.report.MemoryFacts` | Payload byte figures under the payload scope law |
| `tl.report.MergeReport` | Graph-free diagnostic summary of how rank captures were merged |
| `tl.report.MfuReport` | The mode-split time-normalized MFU facts |
| `tl.report.OverheadReport` | Measured capture costs on the USER'S model and host |
| `tl.report.PaddingWaste` | The class-split padding-waste estimate: a DISCLOSURE |
| `tl.report.ParamFacts` | Parameter truth (A07's identity partition, read as facts) |
| `tl.report.ProfileReport` | A printable, tabular resource profile assembled from one trace |
| `tl.report.PrunedParamFacts` | `torch.nn.utils.prune` state read from the capture (`<name>_orig` + `<name>_mask`) |
| `tl.report.RooflineReport` | The roofline map: rows, sum/sum aggregate, coverage by reason |
| `tl.report.RooflineRow` | One op's theoretical intensity/work coordinates |
| `tl.report.SixNDFacts` | The 6ND comparator block: reproduces what HF Trainer logs |
| `tl.report.StatsTableRow` | One per-op observation row |
| `tl.report.StatsTable` | The typed observation table (a FactCore projection, session-only) |
| `tl.report.SummaryReport` | The summary text WITH its typed, detached data (items 10 + 12) |
| `tl.report.SummaryRow` | One typed accounting-unit row (op grain; families coincide there) |
| `tl.report.SummaryTotals` | Headline totals read from FactCore (never re-summed here) |
| `tl.report.ValidationDiagnostic` | Structured non-failing diagnostic emitted by validation |
| `tl.report.ValidationFailure` | A structured description of the first surfaced validation mismatch |
| `tl.report.ValidationStatus` | Machine-readable replay-validation availability or result |
| `tl.stats.FamilyEvidence` | Per-family computation evidence |
| `tl.stats.FittedPCA` | The frozen fitted-PCA payload |
| `tl.stats.HistogramResult` | Finalized signed-log2 histogram counts for one population |
| `tl.stats.RunningStatsResult` | Finalized spine statistics for one population |
| `tl.stats.SimilarityScore` | A similarity score that names the exact measure variant it computes |
| `tl.stats.TensorStats` | Frozen tensor statistics shared by tensor displays and narration; per-family evidence owns every claim |
| `tl.trackers.CloseReport` | The close report: every scheduled observation ends as data, an enumerated presence/skip outcome, or an exception |
| `tl.trackers.GraphIR` | The dep-free executed-DAG graph payload (sink capability `graph`) |
| `tl.trackers.GraphNodeIR` | One executed-op node: name, op kind, inputs, shapes, evidence |
| `tl.trackers.HistogramPoint` | One histogram emission: exact counts on explicit edges |
| `tl.trackers.HistoryView` | Session-side read view over one run's committed history |
| `tl.trackers.ScalarPoint` | One scalar emission: `tag` at `step` with `value` |
| `tl.trackers.SeriesPoint` | One (step-span, block-truth, observation) sample of a series |
| `tl.trackers.StepEmission` | Everything one committed step block emits, in emission order |
| `tl.trackers.TextPoint` | One text emission (manifests, disclosures) |
| `tl.transformer.AttentionView` | One attention pattern in canonical coordinates |
| `tl.transformer.CausalReceipt` | Validated measured-intervention evidence for annotation |
| `tl.transformer.CropInfo` | Crop disclosure: cropping never renormalizes |
| `tl.transformer.EpisodeCoordinates` | Episode coordinates (step and position) carried by transformer view records |
| `tl.transformer.GqaInfo` | Grouped-query-attention disclosure |
| `tl.transformer.MaskInfo` | Mask provenance for an attention view |
| `tl.transformer.PredictionTable` | Top-k tokens per position |
| `tl.transformer.PredictionTrajectory` | Per-layer prediction trajectory record |
| `tl.transformer.SavedFigure` | The result of saving a picture: paths + everything the figure said |
| `tl.transformer.ScoreDecomposition` | Term-complete score decomposition for one (head, dst, src) pair |
| `tl.transformer.TokenMetrics` | Per-token metric strip record |
| `tl.transformer.TokenScoreRow` | One labeled row of per-token scores (strip row) |
| `tl.transformer.TokenScores` | Colored-token strip record, single- or multi-row |
| `tl.transforms.ProbeReport` | Fidelity-probe result: rows per candidate width plus scope fields |
| `tl.transforms.SpecialTokenFacts` | Recorded special-token evidence for semantic token addressing |
| `tl.transforms.TransformContext` | Frozen per-invocation context handed to context-capable transforms |
| `tl.transforms.TransformSpec` | One frozen, declared transform step (the B1 protocol object) |
| `tl.types.Accessor` | Generic dict-like accessor for ordered TorchLens log objects |
| `tl.types.AtenOp` | One value-free ATen dispatcher call observed during capture |
| `tl.types.BackwardPass` | Projected metadata for one backward engine invocation |
| `tl.types.BasisProvenance` | Frozen reproducibility record for one subspace basis |
| `tl.types.Buffer` | Persistent metadata entity for one registered buffer address |
| `tl.types.CaptureOutcome` | Durable, string-only terminal outcome record for one capture product |
| `tl.types.ContainerLeafOccurrence` | One tensor occurrence at one container path |
| `tl.types.ContainerSpec` | Portable description of an output container seen during capture |
| `tl.types.Container` | Computed view over a captured Python output container |
| `tl.types.ContractCheck` | One contract check recorded in a run report |
| `tl.types.FuncCallLocation` | A location in source code where a function call occurred |
| `tl.types.GradFnCall` | Runtime data for one execution of an autograd `grad_fn_handle` node |
| `tl.types.GradFn` | Metadata and runtime ops for one autograd `grad_fn_handle` node |
| `tl.types.InputSpec` | Explicit per-slot input override for the quickstart input grammar |
| `tl.types.MergedFinding` | One structured merge finding |
| `tl.types.MergedTrace` | N rank-local captures stitched at their collective boundaries |
| `tl.types.ModuleCall` | Per-(module, call_index) data for one invocation of a module |
| `tl.types.Module` | Aggregate metadata for one nn.Module across all its invocations |
| `tl.types.NarrationEvent` | One immutable narration record |
| `tl.types.Param` | Metadata about a single model parameter (weight or bias) |
| `tl.types.PartialTrace` | Thin wrapper around raw capture state from a failed forward pass |
| `tl.types.PreHookEffect` | Immutable record of one executed user forward pre-hook transition |
| `tl.types.ReadinessReport` | Complete non-executing readiness report for a Trace provider |
| `tl.types.RunReport` | Honesty, state, resolution, and contract metadata for one run |
| `tl.types.RunnableCompatibility` | Producer/runtime versions recorded for readiness diagnostics |
| `tl.types.RunnableDiagnostic` | Structured diagnostic shared by preflight, readiness, and run reports |
| `tl.types.SpecCompat` | Compatibility result for applying a saved spec to a model log |
| `tl.types.TargetManifestDiff` | Diff between saved target manifest and a new model log |
| `tl.types.TensorInputObservation` | Metadata and optional payload for one tensor in a module input snapshot |
| `tl.types.TraceAutograd` | Backward-pass records and grad_fn facts of a Trace |
| `tl.types.TraceCost` | Compute, memory and timing totals of a Trace |
| `tl.types.TraceGraph` | Graph-shape facts of a Trace: edges, degrees, branching and conditionals |
| `tl.types.TraceHealth` | Honesty diagnostics of the capture: witness and escape-detector facts |
| `tl.types.TraceProvenance` | How, when and from what this Trace was captured |
| `tl.types.TraceSettings` | The capture options this Trace ran with (the stored snapshot) |
| `tl.types.TraceSlice` | Frozen presenter over one trace's executed sub-DAG |
| `tl.types.TraceSource` | Source facts of the traced model class: locations, signatures, docstrings |
| `tl.viz.Drawing` | The picture every draw returns: DOT source (or animation HTML), format, written path; `save(path, editable=)`, `show()`, `close()` (releases a retained trace; idempotent; the `with` block exit calls it); a forward drawing also carries `collapse_plan` and `collapse_schedule` |
| `tl.viz.NodeSpec` | Graphviz node attributes produced by TorchLens before user customization |

### Keywords
Every parameter a caller can pass by name to a public callable has a row here (or in the
keyword table above) naming that callable after "keyword of", with its kind, default and
meaning; `tests/test_glossary_params.py` fails on a public parameter no row names. Constructor
keywords of the records built by TorchLens (the table above) are not public and have no row.

| Spelling | Meaning |
| --- | --- |
| `A=` | constructor keyword of `tl.mechinterp.FactoredMatrix` (required): Left factor, `[..., ldim, mdim]` |
| `B=` | constructor keyword of `tl.mechinterp.FactoredMatrix` (required): Right factor, `[..., mdim, rdim]` |
| `a=` | keyword of `Bundle.comparability`, `Bundle.compare` (required): First member name; keyword of `CKA.update`, `CrossCovariance.update` (required): First tensor-like batch, with rows as observations; keyword of `tl.repgeom.cka_interval` (required): Representations with the same first (stimulus) axis; trailing axes are flattened; keyword of `tl.report.compare_topology`, `tl.stats.cosine_distance`, `tl.stats.relative_l1`, `tl.stats.relative_l2` (required); keyword of `tl.stats.cka` (required): First tensor-like representation |
| `above=` | keyword of `tl.changed`, `tl.pass_variance`, `tl.threshold` (keyword-only; default `None`) |
| `absolute=` | keyword of `tl.attribution.deconvolution`, `tl.attribution.guided_backprop` (keyword-only; default `False`): Whether to absolute-value the result (default `False`) |
| `accumulation_steps=` | constructor keyword of `tl.debug.ChecksSession` (keyword-only; default `None`): Declared static accumulation factor K (watchdog threshold + exact boundary grouping); constructor keyword of `tl.debug.Watchdog` (required) |
| `acknowledge_step_cost=` | constructor keyword of `tl.options.EpisodeOptions` (default `False`): Explicit override for the declared-step cost ceiling (placeholder spelling, [UI-SPRINT]). `True` accepts a declaration beyond the ceiling with the diagnostic-tier cost acknowledged; the default refuses typed so hundreds-of-steps wrapped captures are a decision, never an accident |
| `action=` | keyword of `ChecksSession.register_change_check` (keyword-only; default `'warn'`); keyword of `ChecksSession.register_frozen_check`, `ChecksSession.register_nonfinite_grad_check`, `ChecksSession.register_nonfinite_param_check` (keyword-only; default `'raise'`); keyword of `ChecksSession.register_magnitude_check`, `ChecksSession.register_update_ratio_check` (keyword-only; default `'collect'`); keyword of `tl.when` (required): Helper spec (`tl.scale`, `tl.add_noise`, ...), callable transform, or `InterventionDecision`. The condition decides which tensor it edits: `tl.when(tl.grad("relu_3"), tl.scale(0))` edits the gradient |
| `activation_transform=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `activations=` | keyword of `tl.repgeom.activation_distance_matrix` (required): Activation array or tensor with shape `[N, ...]`; keyword of `tl.repgeom.effective_dimensionality`, `tl.repgeom.scree` (required): Activation representation with leading stimulus dimension, or a square precomputed distance matrix; keyword of `tl.repgeom.rdm` (required): Activation array or tensor with shape `[N, ...]` -- or, with `input_kind="batched"`, `[B, N, ...]` for one RDM per batch element (3-D input already MEANS one flattened stimulus set today, so the batched reading is explicit, never guessed) |
| `activities=` | keyword of `tl.report.kernel_profile` (keyword-only; default `None`): Optional explicit `torch.profiler` activity list; the default asks for CPU plus CUDA when CUDA is available |
| `adapt=` | constructor keyword of `tl.preprocessing.AuthorityAdapter` (required): Normalizer: authority object -> `Resolution`; keyword of `tl.preprocessing.register_authority_adapter` (required): Normalizer returning a `Resolution` |
| `address=` | keyword of `tl.in_module`: the module address whose contained sites the selector matches; keyword of `Trace.activations_by_address` (required): Module or buffer address, e.g. `"layer1.0.conv1"`; keyword of `tl.experimental.attribute_walk` (required): Address such as `"transformer.h[5].mlp.output"`; keyword of `tl.module` (required): Module address or pass label |
| `addresses=` | constructor keyword of `tl.intervention.RegionEdge` (required): The canonical edge occurrence addresses `(child_func_call_id, arg_kind, arg_path)` this dataflow edge resolves to (one value may be consumed at several argument positions) |
| `aggregation=` | keyword of `tl.report.cost_tree` (keyword-only; default `None`): An existing `ComputeAggregation` to project (the tree is a layout over the same rows, never a second traversal); built once from the trace when omitted |
| `agree_on=` | keyword of `Population.eligible_indices` (required): Datum -> hashable key callable (`None` means every member is eligible); constructor keyword of `tl.intervention.SamplingPlan` (required); keyword of `tl.intervention.sample_from` (keyword-only; default `None`): Optional datum -> key callable; donors are drawn only from the agreement class matching the subject's key (causal scrubbing's agreement-conditioned resample) |
| `alias_policy=` | keyword of `FacetSpec.scatter_update` (keyword-only; default `None`): Reserved explicit alias policy for future GQA writes. `None` refuses aliasing selections |
| `aliases=` | constructor keyword of `tl.backends.BackendSpec` (default `()`): Alternate explicit names |
| `aliasing=` | keyword of `FacetSpec.select` (keyword-only; default `False`) |
| `align=` | keyword of `tl.repgeom.mds_evolution` (keyword-only; default `True`): Whether to Procrustes-align each processed embedding to the previous processed embedding |
| `all_nonfinite=` | keyword of `tl.debug.amp_scaled_grads_hint` (keyword-only; default `False`): Whether the caller's finding is non-finite gradients (fp16 scale OVERFLOW wording) rather than merely large norms (scale wording) |
| `allow_changed_params=` | constructor keyword of `tl.options.InterventionOptions` (optional): Let the replay engine use a parameter's current value when it changed since capture. Default `False` refuses with `replay_param_changed_since_capture`; `True` names the parameters used in `trace.last_run["replayed_with_changed_params"]`. The rerun engine always runs the live model and ..; constructor keyword of `tl.options.ReplayOptions` (optional): Replay with a parameter's current value when it changed since capture (an optimizer step, `load_state_dict`, an in-place edit). Default `False` refuses with `replay_param_changed_since_capture`; `True` names the parameters used in `trace.last_run["replayed_with_changed_params"]` |
| `allow_direct_writes=` | keyword of `Trace.save_intervention` (keyword-only; default `False`): Whether executable saves may proceed after direct out writes; keyword of `tl.io.save_intervention` (keyword-only; default `False`): Whether executable saves may proceed when direct out writes were detected |
| `allow_forward_wrap=` | keyword of `tl.trackers.watch` (keyword-only; default `False`): `grain="op"` only: explicit consent to wrap the root module's forward. Attach then calibrates the wrap (plain vs wrapped forward on `inputs`, RNG and buffers restored) and discloses the cost in `plan_report()`, the tracker rows and one warning |
| `allow_missing_phases=` | keyword of `tl.trackers.watch` (keyword-only; default `False`): Demote the missing-optimizer-phase refusal to a named skip (mixed train/eval loops) |
| `allow_shape_change=` | keyword of `tl.add`, `tl.add_noise`, `tl.clamp`, `tl.intervention.scramble_elements`, `tl.project_onto`, `tl.project_out`, `tl.replace_with`, `tl.scale`, `tl.steer`, `tl.zero_ablate` (keyword-only; default `False`): Stored escape-hatch metadata for later execution phases; keyword of `tl.mean_ablate` (keyword-only; default `False`): Stored escape-hatch metadata for execution; keyword of `tl.splice_module` (keyword-only; default `False`): Stored escape hatch allowing output metadata changes |
| `allow_unsupported_runnable_versions=` | keyword of `tl.validation.validate_tlspec` (keyword-only; default `False`): Preserve structural validation while deferring runnable descriptor version ceilings to the structured readiness report. Public callers should retain the default tripwire behavior |
| `allowed_custom_callable_modules=` | keyword of `tl.io.load_intervention_spec`, `tl.load` (keyword-only; default `None`): Optional allowlist of custom callable module names. When supplied, custom imports must be listed even if `trust_custom_callables=True` |
| `alpha=` | keyword of `ReceptiveFieldView.show` (keyword-only; default `0.6`): Overlay opacity; keyword of `tl.attribution.grad_cam` (keyword-only; default `0.6`): Heatmap opacity in `[0, 1]`; keyword of `tl.receptive_field.node_spec` (keyword-only; default `0.6`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested; keyword of `tl.viz.nodes.feature_map` (keyword-only; default `0.75`): Heatmap opacity used for overlays |
| `analyses=` | constructor keyword of `tl.mechinterp.SavePlan` (required); keyword of `tl.mechinterp.retention_plan` (default `('decomposition', 'dla')`): Analysis names among `accumulation` / `decomposition` / `dla` / `heads` / `head_scores` / `prompts`; keyword of `tl.neuro.compare` (keyword-only; default `('rsa', 'ridge', 'cka')`): Subset of `("rsa", "ridge", "cka")`, in the order to report |
| `annotation=` | keyword of `tl.viz.attention` (keyword-only; default `None`): Optional typed annotation; `intervention_effect` requires a valid receipt and renders as printed numbers plus a diverging border -- never recolored attention cells |
| `answer=` | keyword of `tl.mechinterp.direct_logit_contributions` (keyword-only; required): Answer token id(s) -- the logit(s) to decompose; keyword of `tl.mechinterp.inspect_prompt` (keyword-only; default `None`): Optional answer token id(s) to score at `position`; keyword of `tl.mechinterp.test_prompt` (keyword-only; required) |
| `antialias=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Whether resize applies antialiasing: `True`/`False`, the literal string `"warn"` (torchvision's own legacy sentinel -- antialiasing differs by input type and torchvision warns once; declared as-is rather than coerced into a guessed bool), or `None` (undeclared) |
| `api=` | keyword of `tl.bridge.huggingface_hub.from_hub` (keyword-only; default `None`): Optional `HfApi`-compatible object exposing `hf_hub_download`, for tests, mirrors or advanced callers; keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `None`): Optional `HfApi`-compatible object for tests or advanced callers |
| `api_key=` | keyword of `Direction.upload_to_neuronpedia` (keyword-only; required): Your Neuronpedia API key (under Settings on neuronpedia.org) |
| `append=` | constructor keyword of `tl.options.ReplayOptions` (optional): Whether rerun appends a compatible batch chunk |
| `applied=` | keyword of `tl.preprocessing.compare` (default `None`): What actually ran: a `Resolution`, declared fields, a declaration mapping, a Compose-shaped pipeline (parsed), an opaque callable (every field `unknown`, never match), or `None` (nothing declared) |
| `apply_fn=` | constructor keyword of `tl.transforms.TransformDefinition` (required): ` (spec, tensor, ctx) -> tensor` pure application; row count and order preserved |
| `args=` | keyword of `BackendProtocol.can_handle`, `BackendProtocol.capture`, `BackendProtocol.reference_output`, `HFTrainerWatchCallback.on_train_begin`, `HFTrainerWatchCallback.on_train_end` (required) |
| `arguments=` | keyword of `tl.agent.call_tool`, `tl.bridge.mcp.call_tool` (default `None`): JSON arguments matching the tool's declared input schema; keyword of `tl.agent.call_tool_envelope` (default `None`): JSON arguments |
| `arrays=` | keyword of `tl.export.html` (keyword-only; default `'frontier'`): `"frontier"` (default) / `"flagged"` (hard-capped debugging preset) / `"none"` / a `tl.Selector` (the general form) |
| `artifact=` | keyword of `tl.agent.build_envelope` (keyword-only; default `None`): Artifact identity block (`None` for environment-only results); keyword of `tl.viz.timeline` (required): A `"torchlens.memory_timeline.v2"` artifact |
| `as_summary=` | keyword of `ProfileReport.evidence_labels` (keyword-only; default `False`): Return the one-line disclosure string that feeds the report's evidence line instead of the frame |
| `ascending=` | keyword of `Trace.profile` (keyword-only; default `False`): Whether to sort the selected metric ascending; keyword of `Trace.projective_fields` (keyword-only; default `True`); keyword of `Trace.receptive_fields` (keyword-only; default `True`): Whether an explicit `sort_by` sorts ascending; keyword of `tl.report.profile` (keyword-only; default `False`): Whether to reverse the default descending order |
| `assume_no_padding=` | keyword of `tl.positions` (keyword-only; default `False`): Assert that every position is real, licensing physical first/last without a mask (recorded in the edit's identity); keyword of `tl.transforms.cls_token` (default `False`): Recorded caller assertion licensing physical position 0; keyword of `tl.transforms.pool_tokens` (default `False`): Recorded caller assertion that every position is valid; the only licence for mask-free pooling (three-valued policy, decision 12) |
| `async_writes=` | constructor keyword of `tl.options.StorageOptions` (keyword-only; optional): Whether `trace(..., storage=...)` blob writes overlap forward capture through a bounded single-worker pipeline instead of pausing capture for each write (DOCUMENTED-UNSTABLE spelling, naming deferred to the UI sprint). Tri-state: `None` (default) means the consumer default — async for `trace` ..; keyword of `tl.to_disk` (keyword-only; default `None`): Whether `trace(..., storage=...)` blob writes overlap capture on a bounded single-worker pipeline instead of pausing the forward for each write (DOCUMENTED-UNSTABLE spelling, naming deferred to the UI sprint). Tri-state: `None` (default) means async for `trace` captures and synchronous for .. |
| `aten_ops=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `atol=` | keyword of `ReceptiveFieldView.check`, `tl.receptive_field.cross_validate` (keyword-only; default `0.0`): Accepted for compatibility and ignored. Validation always uses exact finite nonzero gradient support; keyword of `ReceptiveFieldView.probe_grad` (keyword-only; default `0.0`): Non-negative gradient support thresholds; keyword of `tl.dead`, `tl.saturated`, `tl.sign`, `tl.stable_across_passes`, `tl.subspace` (keyword-only; default `0.0`); keyword of `tl.debug.bisect_precision` (keyword-only; default `None`): Fixed absolute tolerance; `None` derives per-op from native dtype; keyword of `tl.debug.check_determinism` (keyword-only; default `None`): Optional absolute tolerance; keyword of `tl.debug.compare`, `tl.debug.compare_params`, `tl.debug.first_divergence` (keyword-only; default `1e-08`): Absolute tolerance for `torch.allclose`; constructor keyword of `tl.receptive_field.GradReceptiveField` (required); keyword of `tl.validate` (keyword-only; default `None`): Backward-only absolute tolerance. `None` (default) derives per gradient dtype in `validate_backward_pass`; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `None`): Absolute allclose tolerance. `None` (default) derives the tolerance per gradient dtype (R13); the historical fp32 decimal pair applied to every dtype was ~4.5e11 fp64 ULPs loose and false-failed fp16 grads |
| `attachment=` | keyword of `tl.export.netron` (keyword-only; default `None`): The `<stem>.attachment.json` metrics companion (netron's own sidecar channel; drop it onto the open model). `None` (the default) writes it beside every file export and skips it for the memory-only serve, which has no file for it to sit beside; `True` demands it (and refuses `path=None`); .. |
| `attempts=` | constructor keyword of `tl.types.InputProvenance` (default `None`): Bounded probe diary (rung 3): `(shape-or-None, outcome)` rows. Inference evidence (rung 3), when known |
| `attention=` | keyword of `tl.bridge.lit.model` (keyword-only; default `None`): REFUSED: LIT deleted its attention panel upstream; the type serializes into a void; keyword of `tl.mechinterp.head_weight_views` (required): Attention module record or address (a module exposing the attention facet vocabulary: `q`/`k`/`v`, head geometry); keyword of `tl.mechinterp.ov_circuit`, `tl.mechinterp.qk_circuit` (required): Attention module record or address, or an already-derived `HeadWeightViews` |
| `attribute=` | keyword of `tl.attribution.noise_tunnel` (keyword-only; default `None`): Closed callable `(inputs, input_kwargs) -> AttributionResult`. Exclusive with `method`; `model`/`target`/`method_kwargs` are refused beside it (nothing is forwarded to a closed callable); keyword of `tl.attribution.sensitivity` (keyword-only; default `None`): Closed callable route (exclusive with `method`); subject to the determinism ladder's provenance and behavioral rungs |
| `attribution=` | keyword of `tl.attribution.deletion_curve` (keyword-only; required): `AttributionResult` or tensor over the inputs; `grad_cam` maps (`(N, 1, H, W)` at input resolution) broadcast across channels; keyword of `tl.attribution.infidelity` (keyword-only; required): The attribution to judge: an `AttributionResult` or a value tree mirroring the attributed inputs (input space only; layer-space values refuse with the expansion recipe); keyword of `tl.attribution.insertion_curve` (keyword-only; required): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.road_curve` (keyword-only; required): `AttributionResult` or tensor over the inputs (`grad_cam` maps broadcast across channels) |
| `attribution_model=` | keyword of `tl.bridge.inseq.attribute` (keyword-only; default `None`): Optional pre-built inseq attribution model |
| `audience=` | keyword of `PartialTrace.explain`, `Trace.explain` (keyword-only; default `'auto'`): Report style: `"researcher"` adds graph-pattern detail, `"practitioner"` emphasizes operational status, `"auto"` balances the two |
| `authority=` | keyword of `tl.preprocessing.attach_authority` (required): A recognized authority: torchvision weights preset or the object its `transforms()` returns, HF image processor, timm data config, open_clip preprocess `Compose`, or a declaration mapping; keyword of `tl.preprocessing.check_card` (keyword-only; default `None`): Explicit card instead of the model's own: a weights preset, image processor, data config, preprocess `Compose` or declaration mapping, or an already resolved `Resolution`; keyword of `tl.preprocessing.compare` (required): The reference side: a `Resolution`, a raw authority object (resolved through `resolve`), or a `ResolvedPreprocessing` record is NOT accepted directly -- resolve it first so declared fields are available; keyword of `tl.preprocessing.resolve` (default `None`): Explicit authority object: torchvision `Weights` preset, HF (image) processor, timm data config, torchvision/open_clip `Compose`, or an explicit declaration mapping. Explicit authorities never touch the network. An opaque callable resolves `unknown` (it declares nothing) with the callable .. |
| `ax=` | keyword of `tl.viz.causal_trace_heatmap` (keyword-only; default `None`): Optional matplotlib axes |
| `axes=` | keyword of `tl.experiment.sweep_grid`: axis name -> values (a list or a mapping of value name -> value), in declaration order; keyword of `ReceptiveFieldRuleContext.full` (default `'all_nonmatched'`); constructor keyword of `tl.experiment.SweepGrid` (required); keyword of `tl.mechinterp.patch_grid` (keyword-only; default `('layer', 'position')`): Grid axes from {layer, position, head}; `layer` is always present; constructor keyword of `tl.transforms.RoleDeclaration` (required): One role name per tensor axis, in axis order (axis 0 is the stimulus axis and its role is conventionally `"batch"`) |
| `axis=` | keyword of `tl.intervention.permute_batch` (keyword-only; default `None`); keyword of `tl.intervention.resample_rows_from` (keyword-only; default `None`): The subject value's batch axis, explicit; keyword of `tl.positions` (keyword-only; default `1`): Token axis of the site value (default 1, `(batch, tokens, hidden)`); keyword of `tl.transforms.reduce` (required): Axis or axes to reduce (never resolving to the stimulus axis); keyword of `tl.transforms.take_index` (required): Axis to index (never resolves to the stimulus axis); keyword of `tl.transforms.unit_norm` (default `-1`): Axis to normalize along (never resolves to the stimulus axis) |
| `axis_images=` | keyword of `tl.viz.heatmap` (keyword-only; default `None`): Optional PIL thumbnails aligned with both heatmap axes |
| `axis_kinds=` | constructor keyword of `tl.receptive_field.GridLayout` (required) |
| `axis_labels=` | keyword of `tl.viz.heatmap` (keyword-only; default `None`): Optional text labels applied to BOTH axes (convenience for square symmetric data). Per-axis `row_labels`/`col_labels` take precedence on their axis when also given |
| `b=` | keyword of `Bundle.comparability`, `Bundle.compare` (required): Second member name; keyword of `CKA.update`, `CrossCovariance.update` (required): Second tensor-like batch, with rows as observations; keyword of `tl.repgeom.cka_interval` (required): Representations with the same first (stimulus) axis; trailing axes are flattened; keyword of `tl.report.compare_topology`, `tl.stats.cosine_distance`, `tl.stats.relative_l1`, `tl.stats.relative_l2` (required); keyword of `tl.stats.cka` (required): Second tensor-like representation with the same row count |
| `backend=` | keyword of `ValidationStatus.available_live`, `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unavailable_loaded_runtime_stripped`, `ValidationStatus.unavailable_unsupported`, `ValidationStatus.unverified` (keyword-only; required): Trace backend name; keyword of `tl.backends.resolve` (required): Explicit backend name, or `None` for detector-based resolution; keyword of `tl.record` (keyword-only; default `None`): Optional backend selector. `tl.record` is torch-only in backend v1; non-torch backends raise a canonical unsupported error; keyword of `tl.trace`, `tl.validate`, `tl.validation.validate_forward_pass` (keyword-only; default `None`): Explicit backend name. `None` preserves legacy auto-resolution |
| `background=` | keyword of `tl.bridge.shap.explain` (keyword-only; required): SHAP background (reference) data, required. SHAP values are each input's contribution relative to the background's expected output, so explaining an input against itself gives all zeros; there is no default; keyword of `tl.viz.image_scatter` (keyword-only; default `(255, 255, 255)`): RGB canvas background |
| `backward_capture=` | constructor keyword of `tl.backends.BackendCapabilities` (required): Whether true backward graph capture is supported |
| `backward_node_spec_fn=` | keyword of `Trace.draw` (keyword-only; default `None`): Combined graph: per-node styling callback for the backward nodes |
| `backward_ready=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `base=` | constructor keyword of `tl.stats.HistogramDescriptor` (default `2`) |
| `baseline=` | keyword of `Bundle.rank_changes` (default `None`): Optional baseline override; keyword of `Bundle.store_comparison` (keyword-only; default `None`); constructor keyword of `tl.Bundle` (keyword-only; default `None`): Optional baseline member name or `Trace` reference; keyword of `tl.attribution.deletion_curve` (keyword-only; default `'zeros'`): What a removed feature becomes: `"zeros"` (default), `"blur"` (an 11x11 box blur of the input, `(N, C, H, W)` inputs only), a number, or a tensor of the input's shape; keyword of `tl.attribution.insertion_curve` (keyword-only; default `'zeros'`): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.integrated_gradients`, `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (keyword-only; default `None`): Optional baseline tree matching attributed input leaves. A bare tensor is accepted when there is exactly one attributed leaf; keyword of `tl.attribution.occlusion` (keyword-only; default `'zeros'`): Replacement policy: `"zeros"` (default), `"mean"`, or `"blur"`; keyword of `tl.attribution.text` (keyword-only; default `'auto'`): `"auto"` (task-aware, always resolved to a printed concrete baseline), `"zeros"`, `"pad_token"`, an embeddings tensor, or aligned baseline token ids; keyword of `tl.experiment.head_ablation_candidates` (required): The un-edited capture the sweep will fork from; keyword of `tl.experiment.sweep_grid` (required); keyword of `tl.experiment.sweep_sites` (required): The un-edited capture (`replayable`); becomes the `"baseline"` member and the comparison anchor; keyword of `tl.export.netron` (keyword-only; default `None`): Optional reference for the `"torchlens.baseline"` diff slot (netron #1544). A `Trace` is diffed against `trace` through `torchlens.report.diff` (`log - baseline`): each exported node carries its `"torchlens.diff.*"` facts (status, largest value delta, reference label, join verdict; row ..; keyword of `tl.viz.bundle_diff` (keyword-only; default `None`): Baseline member for the per-node distances. Defaults to `left` |
| `baseline_value=` | keyword of `tl.attribution.occlusion_map` (keyword-only; default `'zeros'`): Replacement policy: `"zeros"` (default), `"mean"`, a scalar, or a broadcastable tensor. Zeros is rarely a neutral input (a normalized image's zero is mean grey, a token embedding's zero is no token), so choose the value whose absence you mean to measure and report it; the policy rides ..; constructor keyword of `tl.experiment.SweepGrid` (required) |
| `baselines=` | keyword of `tl.attribution.gradient_shap` (keyword-only; required): REQUIRED aligned baseline pool: a tensor whose leading axis stacks baseline examples (single attributed leaf), or a tree mirroring the input structure whose leaves share one pool size. Each example draws ONE pool index per sample, shared across every leaf (modality) |
| `basis=` | keyword of `tl.debug.grad_health` (keyword-only; default `'total_norm'`): `"total_norm"` (legacy default, deprecation window announced in `attrs["basis_notice"]`) or `"rms"`; keyword of `tl.intervention.subspace_ablate` (required): `[k, d]` rows spanning the subspace (`[d]` for one direction); keyword of `tl.intervention.subspace_interchange` (required): `[k, d]` rows spanning the subspace; constructor keyword of `tl.mechinterp.SavePlan` (default `...`); keyword of `tl.subspace` (required); keyword of `tl.transforms.project` (required): `(out_extent, in_extent)` float component rows |
| `batch=` | keyword of `tl.preprocessing.diagnose` (required): The exact tensor batch fed (or about to be fed) to the model; keyword of `tl.transformer.bertviz_tuple` (keyword-only; default `True`): Prepend the singleton batch dim (the HF tuple layout) |
| `batch_axis=` | keyword of `tl.features.as_matrix` (keyword-only; default `0`): Which axis indexes stimuli (default 0, the TorchLens convention) |
| `batch_index=` | keyword of `LogitLens.summary`, `LogitLens.top_tokens`, `tl.transformer.prediction_table`, `tl.transformer.prediction_trajectory` (keyword-only; default `0`): Batch element to inspect; keyword of `ReceptiveFieldView.center_unit` (keyword-only; default `None`): Required index for the derived batch-like output axis; keyword of `tl.receptive_field.cross_validate` (keyword-only; default `0`): Explicit seeded batch index used by named presets; keyword of `tl.transformer.attention_view`, `tl.transformer.attention_views` (keyword-only; default `0`): Batch element to extract; keyword of `tl.transformer.score_decomposition` (keyword-only; default `0`): Batch element; keyword of `tl.viz.logit_lens` (keyword-only; default `0`): Batch element to draw |
| `batch_items=` | keyword of `Trace.output_table` (default `None`): `None` returns all captured batch items, an integer returns the first `N` batch items, and a sequence selects explicit batch item indices |
| `batch_render=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `batch_size=` | keyword of `PlanSite.bytes_at_batch` (required); keyword of `tl.debug.guess_input_shape` (keyword-only; default `1`): Batch dimension for synthesized inputs; keyword of `tl.extract_dataset` (default `32`): Number of stimuli per forward pass; keyword of `tl.neuro.plan_extraction` (keyword-only; required): The run's batch size. Belongs to the user: the planner refuses over-budget plans with arithmetic, never shrinks this |
| `batch_support=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `below=` | constructor keyword of `tl.backends.FrameworkRequirement` (required): Exclusive version ceiling, or `None` when none is declared; then `tested_through` states the range's upper evidence; keyword of `tl.changed`, `tl.pass_variance`, `tl.threshold` (keyword-only; default `None`) |
| `benchmark=` | keyword of `tl.bridge.brain_score.per_layer` (required): Callable benchmark accepting `(out, layer=..., **kwargs)` |
| `bias=` | keyword of `tl.intervention.set_affine_coordinates` (keyword-only; default `None`): Optional finite real `[k]` tensor added to the new coordinates |
| `bin_encoding=` | constructor keyword of `tl.stats.HistogramDescriptor` (default `'dense'`): Bin-count encoding of the grid; `'dense'` is the only v1 encoding |
| `bins=` | keyword of `tl.viz.nodes.histogram` (default `30`): Number of histogram bins |
| `bins_per_octave=` | constructor keyword of `tl.stats.HistogramDescriptor` (default `4`) |
| `block=` | keyword of `tl.trackers.corrected_grad_block`, `tl.trackers.emission_from_block` (required) |
| `blocks=` | keyword of `HistoryView.from_blocks` (required) |
| `blur_kernel_size=` | keyword of `tl.attribution.occlusion` (keyword-only; default `3`): Odd spatial averaging width for `baseline="blur"` |
| `boundary_kind=` | constructor keyword of `tl.recording.StopSignal` (keyword-only; default `None`): Event kind at the halt boundary (settlement metadata); `None` for imperative halts, which carry no event context |
| `boundary_label=` | constructor keyword of `tl.recording.StopSignal` (keyword-only; default `None`): Authoritative op label at the halt boundary (raw where available) |
| `bounds=` | keyword of `ChecksSession.register_update_ratio_check` (keyword-only; default `None`); constructor keyword of `tl.debug.RangeProbe` (required): Site target -> `(low, high)` declared semantic bounds (either side `None`). Site targets are anything the public `hooks=` door accepts (labels, selectors); bounds are the USER'S semantics, never a dtype fact -- dtype headroom is `dtype_range`'s different, already-covered question; keyword of `tl.debug.param_health` (keyword-only; default `None`): Exact name -> `(low, high)` declared semantic bounds, either side `None`. Unknown names refuse typed |
| `box_color=` | keyword of `ReceptiveFieldView.show` (keyword-only; default `'#FF3B30'`): Geometric-box overlay color; keyword of `tl.receptive_field.node_spec` (keyword-only; default `'#FF3B30'`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested |
| `breakdown_k=` | keyword of `tl.report.flops`: how many rows the breakdown lists |
| `budget=` | keyword of `tl.mechinterp.patch_grid` (keyword-only; default `None`): Hard cell-count ceiling; exceeding it REFUSES before any rerun; keyword of `tl.mechinterp.patch_query_heads_grid` (keyword-only; default `None`): Hard cell-count ceiling; exceeding it refuses before any rerun |
| `budget_gb=` | keyword of `tl.debug.recompute_candidates` (keyword-only; default `None`): Optional greedy freed-memory target in GiB |
| `budgets=` | keyword of `tl.trackers.watch` (keyword-only; default `None`): Hard caps checked against the plan as preflight FACTS: `max_sites`, `max_bytes_per_step`, `max_elements_per_step` |
| `budgets_in_handler=` | constructor keyword of `tl.agent.AgentToolSpec` (default `False`) |
| `buffers=` | keyword of `Trace.summary` (keyword-only; default `None`): `"summary"` (footer totals; default) \| `"hide"` |
| `bundle=` | keyword of `Bundle.provenance`, `tl.experiment.retained_bytes` (required); keyword of `Bundle.vary` (required): The receiver (mutating spelling; `bundle.fork().vary(...)` preserves the source); keyword of `Bundle.why` (required): The container whose material record is walked; constructor keyword of `tl.experiment.SweepGrid` (required); keyword of `tl.export.chrome_trace_diff` (required): TorchLens `Bundle` with a `supergraph` accessor; keyword of `tl.viz.bundle_diff` (required): Bundle containing at least two traces |
| `bundle_id=` | constructor keyword of `tl.Bundle` (keyword-only; default `None`): Load-side carriage for the persisted container identity. `None` (the user door) mints a fresh random id; the artifact loader passes the saved id verbatim |
| `bwd=` | keyword of `Op.grad_for` (keyword-only; required); keyword of `Trace.draw` (keyword-only; default `None`): Backward and combined graphs: which backward pass or passes to draw; keyword of `tl.debug.bisect_nan_backward` (keyword-only; default `None`): One-based backward pass selector; required when multiple passes were captured (the `grad_health` convention verbatim); keyword of `tl.debug.grad_health` (keyword-only; default `None`): One-based backward pass selector. Required when multiple backward passes are captured |
| `by=` | keyword of `ComponentStack.topk` (keyword-only; default `'norm'`): Scoring reduction: `"norm"` (L2 over the whole row value) is the only v1 option; keyword of `StatsTable.sort_rows` (required): One of the sortable numeric fields (`mean`/`sd`/`min`/`max`/`zero_count`/`nan_count`/ `numel`); keyword of `tl.changed`, `tl.top_changed` (keyword-only; default `'abs'`); keyword of `tl.debug.hot_path` (default `'flops'`): Cost metric: `"flops"`, `"memory"`, or `"duration"`; keyword of `tl.debug.hot_path_rows` (default `'flops'`): Cost metric: `"flops"`, `"memory"`, `"duration"`, or `"device_time"` (joined Kineto device nanoseconds); keyword of `tl.threshold`, `tl.top_fraction`, `tl.top_k` (keyword-only; default `'value'`) |
| `cache=` | constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.stats.aggregate` (keyword-only; default `False`): `True` keys the run (model contents and code, every batch by value, each accumulator's starting state and encoder, the metric keys, `loss_fn=`, the torchlens and torch versions) and, when the same run was cached before, restores each accumulator's saved final state in place and returns its .. |
| `cache_dir=` | keyword of `tl.clear_cache` (default `None`): Cache root accepted by `trace(cache_dir=...)`. `None` uses the configured default; keyword of `tl.examples.load` (keyword-only; default `None`): Reserved for future Hub-backed example downloads. Currently unused; constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.stats.aggregate` (keyword-only; default `None`): Cache directory, as for `tl.trace(cache=True)`: default `TORCHLENS_CACHE_DIR` or `~/.cache/torchlens`; entries go in its `aggregate` subdirectory |
| `cache_key_version=` | constructor keyword of `tl.types.InputProvenance` (default `None`) |
| `cache_resident_bytes=` | keyword of `tl.report.roofline` (keyword-only; default `4194304`): Ops whose ideal traffic fits under this threshold are excluded from headline bound counts (hinted, still listed) |
| `cache_status=` | constructor keyword of `tl.types.InputProvenance` (default `None`) |
| `call=` | constructor keyword of `tl.attribution.SiteTarget` (keyword-only; default `None`): Which call of the layer, counted from 0, when the layer runs more than once in a forward pass. `None` requires that the layer runs once |
| `call_index=` | keyword of `Recording.records_in_pass` (required); keyword of `TapObserver.record_backward` (keyword-only; default `None`): One-based backward call index; constructor keyword of `tl.facets.FacetSpec` (default `None`): Module call index for module homes |
| `call_kwargs=` | keyword of `tl.bridge.pytorch_grad_cam.cam` (keyword-only; default `None`): Options forwarded unchanged to the CAM call, e.g. `{"aug_smooth": True, "eigen_smooth": True}` |
| `call_structure=` | constructor keyword of `tl.types.InputProvenance` (required): Normalized call description: positional count, keyword names, and whether non-tensor arguments are present |
| `cam_class=` | keyword of `tl.bridge.pytorch_grad_cam.cam` (keyword-only; default `None`): Optional CAM class or factory. Defaults to `pytorch_grad_cam.GradCAM` |
| `can_handle=` | constructor keyword of `tl.backends.BackendSpec` (required): Side-effect-light detector used only for `backend=None` or mismatch checks |
| `candidate_id=` | keyword of `SweepGrid.coords` (required) |
| `candidates=` | keyword of `tl.experiment.sweep_sites` (keyword-only; required): Ordered candidate plan: `{candidate_id: site}` or a sequence of sites (ids minted `c0..cN`). A site is a Selection / region producer / scoped facet selector / label / selector. A multi-site Selection is ONE candidate knockout set only when DECLARED via `multi_site=` |
| `canvas_size=` | keyword of `tl.viz.image_scatter` (keyword-only; default `420`): Width and height of the square output image; keyword of `tl.viz.nodes.mds_scatter` (keyword-only; default `420`): Width and height of the rendered scatter image in pixels; keyword of `tl.viz.nodes.rdm` (keyword-only; default `360`): Width and height of the rendered heatmap image in pixels |
| `capabilities=` | constructor keyword of `tl.backends.BackendSpec` (required): Consolidated capability flags; keyword of `tl.export.register_target` (keyword-only; default `None`): Extra capability rows; the door always records the tier; keyword of `tl.io.register_sidecar_family` (keyword-only; default `None`): Optional extra capability rows; the door always declares the payload encoding and budget; constructor keyword of `tl.trackers.MemorySink` (keyword-only; default `None`) |
| `capability_class=` | constructor keyword of `tl.facets.FacetSpec` (default `'bijective_view'`): Weakest capability class of the transform chain |
| `capability_flags=` | constructor keyword of `tl.facets.FacetSpec` (optional): Weakest-link capability flags |
| `capability_implementations=` | constructor keyword of `tl.backends.BackendSpec` (default `None`): Lazy factories for the implementing surface of each `True` gated capability flag (`GATED_CAPABILITY_FLAGS`). Gates open only through `require_capability_implementation`, never through the boolean alone, so a flag flip without a registered implementation refuses typed instead of silently .. |
| `capture=` | keyword of `tl.inventory.list_sites` (keyword-only; default `None`): Optional `CaptureOptions` override for the forward rung, used for capture metadata (e.g. `structure_only=True`; shapes are then labeled hypotheses). Default is a metadata-only capture retaining no activation payloads; keyword of `tl.record` (keyword-only; default `None`): `tl.options.CaptureOptions`. The fields `record` reads: `lookback`, `lookback_payload_policy`, `seed`, `activation_transform`, `save_raw_activations`, `save_grads`, `grad_transform`, `save_raw_grads`, `backward_ready` and `detach_saved_activations` (meanings below); keyword of `tl.trace` (keyword-only; default `None`): Grouped capture options (`CaptureOptions`). The one spelling for every capture knob (`seed`, `replayable`, `verbose`, `structure_only`, `save_mode`, `lookback`, `chunk_size`, `grouping`, ...); the former flat kwargs are removed. `grouping` (UNSTABLE) is the closed-vocabulary ..; constructor keyword of `tl.recording.Recorder` (keyword-only; default `None`): `tl.options.CaptureOptions` carrying the capture-time knobs the recorder reads: `lookback`, `lookback_payload_policy`, `seed`, `activation_transform`, `save_raw_activations`, `save_grads`, `grad_transform`, `save_raw_grads`, `backward_ready` and `detach_saved_activations`. Any .. |
| `capture_backend=` | constructor keyword of `tl.backends.BackendSpec` (default `None`): Optional factory for the lower-level Protocol adapter used by shared capture orchestration |
| `capture_container_structure=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `capture_mode=` | keyword of `CaptureBackend.set_capture_producer_policy` (required) |
| `capture_tensor_grad_hooks=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `capture_trace=` | constructor keyword of `tl.backends.BackendSpec` (required): Public trace entry for this backend |
| `captured_bytes=` | field of `MemoryFacts`, `SummaryTotals` |
| `captured_saved_ops=` | field of `MemoryFacts` |
| `captured_trace=` | keyword of `tl.fingerprint.trace` (required): Completed TorchLens trace to hash |
| `carry_state=` | keyword of `Trace.run` (keyword-only; default `False`): PROVISIONAL spelling (documented-unstable). Default `False`: the live provider snapshot-restores the model's declared state (named parameters, registered buffers, alias topology preserved) around the run, so repeated `run()` calls leave the model bit-identical. `True` skips the restore: ..; constructor keyword of `tl.options.CaptureOptions` (optional) |
| `cascade_relations=` | keyword of `Bundle.clear`, `Bundle.remove_except` (keyword-only; default `False`): Whether relation rows naming any removed member are dropped with it. `False` (default) refuses typed BEFORE any member is removed when a removed member is named in a relation row; keyword of `Bundle.remove` (keyword-only; default `False`): Whether relation rows naming a removed member are dropped with it. `False` (default) refuses typed BEFORE any member is removed when a removed member is named in a relation row |
| `caveats=` | constructor keyword of `tl.types.InputProvenance` (default `()`): Rendered disclosure sentences |
| `cell_size=` | keyword of `tl.viz.nodes.feature_map` (keyword-only; default `72`): Width and height of each rendered map cell in pixels |
| `cells=` | constructor keyword of `tl.experiment.SweepGrid` (required) |
| `center=` | keyword of `tl.transforms.project` (default `None`): Optional `(in_extent,)` vector subtracted before projecting. Uncentered apply leaves Euclidean RDMs exact but damages correlation-distance RDMs (r ~ 0.77 measured centered-vs-not) |
| `channel_dependency=` | keyword of `ReceptiveFieldRuleContext.window` (keyword-only; default `None`) |
| `channel_order=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Channel convention (`"rgb"` / `"bgr"`) |
| `channels=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `None`): Optional channel count override for convolutional inputs; keyword of `tl.viz.feature_maps` (keyword-only; default `None`): Channel selection. `None` stores one channel-aggregate map per stimulus, a sequence stores explicit channels, and `"top"` stores per-stimulus top channels by mean absolute activation |
| `chat_template=` | keyword of `tl.bridge.transformers.trace_text` (keyword-only; default `False`): When True and text is a chat-message list, apply the tokenizer's chat template before tokenization |
| `check_name=` | keyword of `TagGrammar.check` (required) |
| `check_sdpa=` | keyword of `tl.bridge.transformers.record_attention` (keyword-only; default `False`): Also run `scaled_dot_product_attention` on each call's inputs and refuse (`hf_attention_sdpa_mismatch`) when the outputs disagree beyond the semantic SDPA reconstruction tolerance. Calls with dropout, softcapping or attention sinks are not checked (`AttentionCall.sdpa_checked` says which .. |
| `checks=` | keyword of `tl.preprocessing.diagnose` (keyword-only; default `('nonfinite', 'dtype_regime', 'layout')`): Which checks to run. Default is the intrinsic set (`nonfinite` / `dtype_regime` / `layout`); the measured-weak `"confinement"` arm must be requested explicitly |
| `checksums=` | keyword of `tl.extract_dataset` (keyword-only; default `'fast'`): The integrity level: `"fast"` (final-file CRC-32 plus the on-device value reduction; default), `"crypto"` (blake2b file digests), `"none"` (recorded opt-out; verification refuses) |
| `child=` | constructor keyword of `tl.intervention.RegionEdge` (required): Pass-qualified label of the consuming op; keyword of `tl.observers.register_tensor_connection` (required): Child tensor already tagged by TorchLens |
| `chunk_paths=` | keyword of `Trace.run` (keyword-only; default `None`): Optional explicit tensor leaf paths to split; keyword of `tl.trace` (keyword-only; optional): Optional explicit tensor leaf paths to split when multiple batched tensor leaves are present |
| `chunk_size=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional); constructor keyword of `tl.options.ReplayOptions` (optional): Forward chunk size for rerun chunking sugar. Splits positional input along dimension 0 and appends compatible chunks |
| `claim=` | keyword of `tl.attribution.causal_audit` (keyword-only; required): The internal attribution to audit: an `AttributionResult` of a layer method whose values have the site's output shape, a `ContributionScores` table, or a `ReadTable` |
| `class_name=` | keyword of `tl.facets.describe` (required); keyword of `tl.facets.functions`, `tl.io.reset_naming_counter` (default `None`); keyword of `tl.facets.register` (keyword-only; default `None`): Short class name or match-any tuple of names; constructor keyword of `tl.mechinterp.ProjectionSource` (required): Owning module class (the orientation authority); keyword of `tl.inputs.register`, `tl.outputs.register` (keyword-only; default `None`): Model class name, glob pattern or match-any tuple of them; the preprocessor or decoder runs only for models whose class name matches |
| `class_names=` | constructor keyword of `tl.facets.FacetFunction` (required): Short class names matched by this recipe |
| `class_qualname=` | keyword of `tl.facets.register` (keyword-only; default `None`): Fully-qualified class name or match-any tuple of names |
| `clean_input=` | keyword of `tl.mechinterp.patch_grid`, `tl.mechinterp.patch_head`, `tl.mechinterp.patch_heads_grid`, `tl.mechinterp.patch_query_heads_grid`, `tl.mechinterp.patch_residual_grid` (required) |
| `cli_verb=` | constructor keyword of `tl.agent.AgentToolSpec` (required) |
| `client=` | keyword of `tl.export.mlflow` (default `None`): Optional `mlflow` module or `mlflow.MlflowClient` (or an object exposing `log_metric`, `log_table`, `log_text` and `set_tag`); constructor keyword of `tl.trackers.MLflowSink` (default `None`): The fluent `mlflow` module (its active run is used, or `run_id=` when given), an `mlflow.MlflowClient` (needs `run_id=`), or `None` to import `mlflow` and use its active run. This sink never calls `mlflow.start_run` |
| `clip=` | keyword of `ReceptiveFieldView.at` (keyword-only; default `True`): Whether bounds are clipped to captured input extents |
| `clip_norm=` | constructor keyword of `tl.debug.ChecksSession` (keyword-only; default `None`): Declared `clip_grad_norm_` max_norm |
| `clock_assumptions=` | constructor keyword of `tl.report.PeakRow` (required) |
| `cmap=` | keyword of `ReceptiveFieldView.show` (keyword-only; default `'magma'`): Gradient heatmap colormap; keyword of `tl.attribution.grad_cam` (keyword-only; default `'magma'`): Heatmap colormap; keyword of `tl.receptive_field.node_spec` (keyword-only; default `'magma'`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested; keyword of `tl.viz.causal_trace_heatmap` (keyword-only; default `'viridis'`): Matplotlib colormap; keyword of `tl.viz.heatmap` (keyword-only; default `'viridis'`): Colormap name: `"viridis"`, `"magma"`, or `"gray"`; keyword of `tl.viz.nodes.feature_map` (keyword-only; default `'magma'`): Colormap name passed to the node plot colormap helper; keyword of `tl.viz.nodes.rdm` (keyword-only; default `'viridis'`): Colormap passed to `heatmap` |
| `code_panel=` | keyword of `Trace.draw` (keyword-only; default `False`): Source-location panel option (`False`, `True` or a panel spec) |
| `coercible=` | constructor keyword of `tl.backends.BackendSpec` (default `False`): Whether explicit resolution may accept inputs `can_handle` returns false for |
| `col_labels=` | keyword of `tl.viz.heatmap` (keyword-only; default `None`): Optional text labels for the column axis (drawn on top). For an attention picture these are the KEY tokens (columns attend to) |
| `collapse=` | keyword of `Trace.draw` (keyword-only; default `'none'`): Smart module-collapse mode. `"none"` preserves the full graph, `"auto"` uses the readability-targeted engine, `"max"` condenses every eligible module, and a float in `[0.0, 1.0]` selects a deterministic monotone schedule (`0.0` is `"none"`, `1.0` is `"max"`; larger values never ..; constructor keyword of `tl.options.VisualizationOptions` (optional): Smart module-collapse mode: `"none"`, `"auto"`, `"max"`, or a float in `[0.0, 1.0]` on the public monotone schedule |
| `collapse_fn=` | keyword of `Trace.draw` (keyword-only; default `None`): Per-node styling and selection callbacks (see `tl.viz.nodes`); constructor keyword of `tl.options.VisualizationOptions` (optional): Optional module collapse predicate |
| `collapsed_node_spec_fn=` | keyword of `Trace.draw` (keyword-only; default `None`): Per-node styling and selection callbacks (see `tl.viz.nodes`); constructor keyword of `tl.options.VisualizationOptions` (optional): Optional collapsed module-node customization callback |
| `collate=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): Optional collate (DOCUMENTED-UNSTABLE): a callable receiving the batch's item list and returning a `BatchEnvelope`, a Mapping (kwargs), or a Tensor (one positional arg) — bare tuples refuse as ambiguous. `hf_collate` builds the .. |
| `collector=` | keyword of `HistoryView.from_collector` (required); constructor keyword of `tl.trackers.WatchSession` (required) |
| `color_by=` | keyword of `Trace.draw` (keyword-only; default `None`): Fill encoding channel (keyword-only; unstable). A Layer/Op field name (`"flops_forward"`), a scalar builtin (`"time"`, `"flops"`, `"bytes"`, `"magnitude"`, `"grad_norm"`), a callable `node -> value`, a mapping of layer label or module address to value (an overlay; the legend names it ..; constructor keyword of `tl.options.VisualizationOptions` (optional): UNSTABLE encoding-channel value source; see `Trace.draw` |
| `colors=` | keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional RGB colors for each series |
| `column_budget=` | keyword of `tl.viz.timeline` (keyword-only; default `320`): Maximum rendered columns; larger traces bin contiguous ordinals and the caption discloses the bin size |
| `columns=` | keyword of `Trace.summary` (keyword-only; default `None`): Bundle name, exact ordered list, or `+name`/`-name` deltas |
| `commit_message=` | keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `'Add TorchLens artifact'`): Commit message for the upload |
| `compatibility=` | keyword of `tl.neuro.export_npz` (keyword-only; default `None`): `"net2brain"` (with `layout="per_stimulus"`) enforces their FILE-NAMING CONTRACT: their encoding loader globs and sorts LEXICOGRAPHICALLY, so filenames are zero-padded in stimulus order and the sidecar records the name-to-stimulus-id map -- get this wrong and rows silently misalign against .. |
| `components=` | keyword of `tl.mechinterp.apply_norm_scale` (required): Any component stack whose rows live in the norm's input space; keyword of `tl.mechinterp.direct_logit_contributions` (default `None`): Any COMPLETE or CLOSED-CONSTANT `ComponentStack` (default: `residual_decomposition(trace)`). Admissibility is verified by the identity, not trusted; diagnostic stacks refuse |
| `compute_device=` | keyword of `tl.repgeom.rdm` (keyword-only; default `None`): Device for the pairwise kernel (e.g. `"cuda"`). `None` keeps the historical CPU compute |
| `compute_input_output_distances=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `condition=` | keyword of `tl.when` (required): Selector or predicate evaluated against each op context (the WHERE term; see the address law in the module docstring) |
| `cone=` | keyword of `tl.receptive_field.node_spec` (keyword-only; default `True`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested |
| `confidence=` | keyword of `tl.neuro.compare` (keyword-only; default `0.95`): Nominal coverage of the intervals; keyword of `tl.repgeom.cka_interval` (keyword-only; default `0.95`): Bootstrap settings (see `rdm_compare_interval`); keyword of `tl.repgeom.compare_rdms_interval` (keyword-only; default `0.95`): Nominal coverage of the percentile interval |
| `config=` | constructor keyword of `tl.trackers.WatchSession` (required) |
| `confirm_mutation=` | keyword of `Trace.clear_edits` (keyword-only; default `False`): Suppress the once-per-root mutate-in-place warning for callers that intentionally mutate this log; constructor keyword of `tl.options.InterventionOptions` (optional): Whether root-mutation warnings are suppressed for intentional mutation |
| `conflict_alias_group=` | constructor keyword of `tl.facets.FacetSpec` (default `None`): Shared-home conflict/alias group for future intervention |
| `conformance_ref=` | keyword of `tl.export.register_target` (keyword-only; default `None`): Optional pointer to the registration's conformance row |
| `constraining_op=` | constructor keyword of `tl.types.InputProvenance` (default `None`) |
| `construction=` | keyword of `tl.transforms.srp` (default `'very_sparse_fixed'`): `"very_sparse_fixed"` (default; exact per-column counts, cheap, `distance_claim="empirical_only"`) or `"iid_bernoulli"` (the i.i.d. matrix, O(D*k) generation, also `empirical_only` until the literature task resolves); keyword of `tl.transforms.srp_fidelity_probe` (default `'very_sparse_fixed'`): SRP construction to probe |
| `container_max_inline=` | keyword of `Trace.draw` (keyword-only; default `12`): Inline container size ceiling for `show_containers="collapsed"` and `"nodes"` |
| `container_path=` | constructor keyword of `tl.options.VisualizationOptions` (optional): Output path stem for rendered graph files |
| `container_type=` | keyword of `tl.types.register_container` (required): Runtime class to recognize as a container |
| `content_digest=` | constructor keyword of `tl.intervention.Population` (required): sha256 identity: member bytes for tensor members, the address digest for trace members (see `digest_kind`) |
| `context_capable=` | constructor keyword of `tl.transforms.OpaqueStep` (required): Whether the callable DECLARED context capability (`ContextTransform`); never sniffed; constructor keyword of `tl.transforms.PlannedStep` (required): Whether the step declared context capability; constructor keyword of `tl.transforms.TransformDefinition` (default `False`): Whether `apply_fn` consumes the context |
| `context_tokens=` | keyword of `tl.transformer.prediction_table` (keyword-only; default `None`): Optional input tokens per position (defaults to position numbers) |
| `contributions=` | keyword of `tl.mechinterp.inspect_prompt` (keyword-only; default `False`): Append the top DLA contributors for `answer` (needs the residual stream retained; refuses with the plan otherwise); keyword of `tl.mechinterp.test_prompt` (keyword-only; default `False`) |
| `control=` | keyword of `HFTrainerWatchCallback.on_train_begin`, `HFTrainerWatchCallback.on_train_end` (required) |
| `controls=` | keyword of `tl.attribution.causal_audit` (keyword-only; default `('bottom', 'random')`): Control sets beside the top-k set: `"bottom"` and `"random"` (default both) |
| `coords=` | keyword of `tl.viz.image_scatter` (required): `[N, 2]` NumPy-like or Torch tensor coordinates |
| `copy=` | keyword of `Trace.annotate` (keyword-only; default `False`): If `True`, annotate and return an owned fork instead of mutating this trace |
| `corrupted_graph=` | keyword of `tl.mechinterp.patch_edges` (keyword-only; required): The `ComponentGraph` of the corrupted input (same model, same shapes); each edge's corrupted contribution is `corrupt.nodes[edge.sender.name].value` |
| `corrupted_input=` | keyword of `tl.attribution.causal_audit` (keyword-only; default `None`): For an EDGE table only: the corrupted inputs it was scored with (`edge_scores`' `corrupted_input=`), the same shape as `inputs`; keyword of `tl.mechinterp.edge_scores` (keyword-only; required): Keyword-only. The corrupted inputs, the same shape as `inputs`; keyword of `tl.mechinterp.patch_grid`, `tl.mechinterp.patch_head`, `tl.mechinterp.patch_heads_grid`, `tl.mechinterp.patch_query_heads_grid`, `tl.mechinterp.patch_residual_grid` (required) |
| `cotangent=` | keyword of `tl.attribution.seed` (keyword-only; default `None`): Explicit cotangent tensor over the site's output shape |
| `create_graph=` | keyword of `Recorder.backward` (keyword-only; default `False`); keyword of `Recording.backward` (keyword-only; default `False`): Forwarded to `Tensor.backward` |
| `create_repo=` | keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `True`): Whether to create the repo before upload |
| `crop_size=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Center/final crop size (int or `(h, w)`) |
| `cross_batch_influence=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `crossings=` | constructor keyword of `tl.options.EpisodeOptions` (default `()`): DECLARED exogenous crossings: step indices (1-based joins into steps 1+) whose entries are declared to come from outside the capture (the tool-call shape). A declared crossing grades `declared` -- disclosed, never silently continuous; series claims still refuse across it .. |
| `ctx=` | keyword of `OpaqueStep.apply` (required): Optional context; passed only to declared context-capable steps; keyword of `Recipe.match` (required): `RecordContext` (or context-shaped subject) for the candidate op; keyword of `TapObserver.record_backward` (keyword-only; default `None`): Hook context of a gradient-site fire; keyword of `TransformPipeline.apply` (default `None`): Optional context; `None` always legal; keyword of `TransformPipeline.plan` (default `None`): Optional context spec; `None` always legal; keyword of `TransformSpec.apply` (default `None`): Optional context; `None` is always legal; keyword of `TransformSpec.plan` (default `None`): Optional context spec; `None` is always legal; keyword of `tl.transforms.apply_dispatch` (required): Optional context; keyword of `tl.transforms.srp_realized_facts` (default `None`): Optional context (share keying, workspace, device); keyword of `tl.transforms.srp_verify_matrix` (default `None`): Optional context (share keying) |
| `d8_meta=` | constructor keyword of `tl.types.InputProvenance` (default `None`) |
| `d_head=` | keyword of `FacetSpec.heads` (required) |
| `data=` | keyword of `Trace.annotate` (keyword-only; default `None`): JSON-serializable metadata or a portable tensor payload. Tensor payloads are persisted under `_annotation_blobs` and are supported only for torch traces; keyword of `tl.agent.build_envelope` (keyword-only; required): Tool-specific result record; keyword of `tl.intervention.population` (keyword-only; default `None`): Optional per-member agreement datums, aligned with member order; keyword of `tl.mechinterp.measure_trajectory` (required): Iterable of batches, as for `tl.stats.aggregate`: a batch is the model input, or a tuple `(model_input, targets)` whose `targets` is an integer tensor `[batch, positions]` of the token each position should predict. A bare tensor is one batch; keyword of `tl.repgeom.classical_mds` (required): Pairwise distance matrix or row-wise coordinate/feature data; keyword of `tl.viz.heatmap` (required): Two-dimensional array-like values to colorize |
| `dataloader=` | keyword of `tl.stats.aggregate` (required): Iterable of model inputs |
| `datums=` | constructor keyword of `tl.intervention.Population` (required): Optional per-member agreement datums (`None` when absent). Length always equals `len(members)` when present |
| `decisions=` | keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unverified` (keyword-only; default `None`): Optional per-op validation decisions |
| `declared=` | keyword of `tl.preprocessing.diagnose` (keyword-only; default `None`): The declared configuration the findings may contradict; omit for intrinsic-only checks |
| `decomposition=` | keyword of `tl.transformer.render_neuron_card` (required): A validated `ScoreDecomposition` |
| `default=` | keyword of `FacetView.get` (default `None`): Value returned when the facet is unavailable now; keyword of `ReceptiveFieldRuleContext.arg` (default `None`): Value used when the argument was not captured; keyword of `ReceptiveFieldRuleContext.cfg` (default `None`): Value used when the operation did not record `key` |
| `default_grad=` | keyword of `Recorder.backward` (keyword-only; default `None`); keyword of `Recording.backward` (keyword-only; default `None`): Default capture decision when no predicate is configured; constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional) |
| `default_limits=` | constructor keyword of `tl.agent.AgentToolSpec` (optional) |
| `default_module=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional) |
| `default_op=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional) |
| `default_steering_strength=` | keyword of `Direction.upload_to_neuronpedia` (keyword-only; default `1.0`): The default steering strength Neuronpedia offers, from -100 to 100 |
| `definition=` | keyword of `tl.transforms.register_transform` (required): The transform's `TransformDefinition` |
| `degradations=` | field of `MergeReport` |
| `delta=` | keyword of `tl.add` (required): Scalar or tensor value to add to the current out |
| `demeaned_logits=` | keyword of `tl.attribution.build_graph` (keyword-only; default `True`): `True` (default) makes each logit node its logit minus the mean logit; `False` makes it the raw logit |
| `denominator=` | constructor keyword of `tl.report.MfuProvenance` (default `'step_wall_time'`) |
| `denominator_scope=` | constructor keyword of `tl.report.MfuProvenance` (default `('forward',)`) |
| `dense_flops_per_s=` | constructor keyword of `tl.report.PeakRow` (required) |
| `density=` | keyword of `tl.transforms.srp` (default `'auto'`): `"auto"` (-> `1/sqrt(D)`) or an explicit float in `(0, 1]`; the realized integer density is recorded per site |
| `depth=` | keyword of `Trace.draw` (keyword-only; default `1000`): Maximum module nesting depth shown; keyword of `Trace.summary` (keyword-only; default `None`): `"auto"` \| `"all"` \| int module-tree depth; keyword of `Trace.to_dagua_graph` (default `1000`): Forwarded unchanged to `trace_to_dagua_graph`; keyword of `tl.export.netron` (keyword-only; default `1`): Module level shown in the root graph (module granularity only); the interim default 1 is the shallow architecture diagram, recorded in metadata; constructor keyword of `tl.options.HtmlReportOptions` (default `None`); constructor keyword of `tl.options.VisualizationOptions` (optional): Maximum module nesting depth shown in the graph; keyword of `tl.experimental.dagua.trace_to_dagua_graph` (default `1000`) |
| `descending=` | keyword of `StatsTable.sort_rows` (keyword-only; default `True`): Sort direction; rows without numbers keep their relative order after all valued rows |
| `description=` | constructor keyword of `tl.agent.AgentToolSpec` (required); keyword of `tl.bridge.lit.dataset` (keyword-only; default `'TorchLens LIT bridge dataset.'`): Human-readable dataset description |
| `descriptor=` | keyword of `TensorBoardSink.preflight_histograms`, `WandbSink.preflight_histograms` (required); constructor keyword of `tl.stats.Histogram` (default `...`); keyword of `tl.trackers.watch` (keyword-only; default `None`): Histogram grid override. Only when both this and `settings` are None does the first sink offering `default_histogram_descriptor` supply the grid (`WandbSink`: `WANDB_SAFE_DESCRIPTOR`, 497 buckets), and that one grid then applies to every sink; otherwise the caller's grid, or the .. |
| `dest=` | keyword of `tl.dataset_extraction.export_extraction` (required): Destination directory (created; published atomically by rename) |
| `destination=` | keyword of `tl.transformer.score_decomposition` (keyword-only; required): Query (row) position |
| `detach_saved_activations=` | keyword of `tl.extract` (keyword-only; default `True`): DOCUMENTED-UNSTABLE. Defaults to `True`: the outs are detached copies. `False` returns graph-connected outs (`grad_fn` set wherever the model's forward carries autograd history), so a loss built from them backpropagates into the model's parameters; constructor keyword of `tl.options.CaptureOptions` (optional) |
| `deterministic=` | constructor keyword of `tl.options.HtmlReportOptions` (default `False`) |
| `device=` | keyword of `ExtractionPlan.run` (keyword-only; default `None`): Optional device for model and stimuli; keyword of `tl.bridge.brain_score.activations_extractor` (keyword-only; default `None`): Optional device override for model and stimuli; keyword of `tl.bridge.brain_score.get_activations_fn` (keyword-only; default `None`): Optional device override; defaults to the model's first parameter's device (CPU for parameterless models). The model is moved there once; keyword of `tl.debug.guess_input_shape` (keyword-only; default `None`): Optional device override; defaults to the model's first parameter or buffer device; keyword of `tl.extract_dataset` (default `None`): Optional device for model and stimuli; constructor keyword of `tl.recording.CaptureSpec` (default `None`): Optional target device for retained payloads; constructor keyword of `tl.stats.CKA` (keyword-only; default `None`): Compute/state device (e.g. `"cuda"`). `None` keeps the historical CPU placement bit-identical; constructor keyword of `tl.stats.Covariance`, `tl.stats.CrossCovariance` (keyword-only; default `None`): Compute/state device (default: historical CPU); keyword of `tl.stats.cka` (keyword-only; default `None`): Compute device (e.g. `"cuda"` keeps the Gram work on-device). `None` keeps the historical CPU placement |
| `device_identity=` | keyword of `DevicePeaks.for_device` (required); constructor keyword of `tl.report.PeakRow` (required) |
| `devices=` | constructor keyword of `tl.io.JaxPayloadLoadHint` (default `None`): Optional explicit local devices to use for metadata reconstruction |
| `differentiable=` | constructor keyword of `tl.options.ReplayOptions` (optional): Whether replay returns a new differentiable Trace over fresh frontier leaves instead of mutating the existing Trace in place |
| `digest_kind=` | constructor keyword of `tl.intervention.Population` (required): `"content"` (value hash) or `"address"` (identity facts only) |
| `digest_population=` | keyword of `tl.intervention.population` (keyword-only; default `False`): For trace-backed populations, opt into value hashing of every RETAINED payload (measured cost: hashing one gpt2 batch-32 residual is ~310 ms). The default records the ADDRESS identity -- persisted capture facts plus per-op geometry, never object addresses -- which already identifies the draw .. |
| `dilation=` | keyword of `ReceptiveFieldRuleContext.window` (keyword-only; default `1`) |
| `dim=` | keyword of `FacetSpec.select` (required); keyword of `FacetSpec.split` (default `-1`); keyword of `Population.mean` (default `None`): `None` for the elementwise across-member mean, or explicit member-shape axes (int or tuple of ints) to reduce further with `keepdim=True`. Named axis tokens resolve through the axis ladder and refuse `axis_semantics_unknown` until axis-role evidence exists (never guessed from rank; edits memo ..; keyword of `Population.std` (default `None`); keyword of `tl.debug.dead_neurons` (keyword-only; default `None`): Unit axis; `None` picks it per op (last axis of a 2-D or 3-D activation, the channel axis of a convolution, pooling or per-channel norm and of the same-shaped ops it feeds, axis 1 for four or more dimensions) and an int forces that axis on every op; keyword of `tl.intervention.set_direction_mean` (keyword-only; required): Explicit axis of the site value that `direction` lives on; keyword of `tl.intervention.subspace_ablate` (keyword-only; default `-1`): Axis of the site value the basis indexes (default the trailing axis); keyword of `tl.intervention.subspace_interchange` (keyword-only; default `-1`): Axis of the site value the basis indexes; keyword of `tl.mean_ablate` (keyword-only; required): REQUIRED. `None` averages every axis of a tensor or self source to one scalar; an int or tuple of ints averages those axes with `keepdim` broadcast. For a `Population`, `None` is the elementwise across-member mean and axes reduce that mean further. A semantic axis name refuses ..; keyword of `tl.project_onto`, `tl.project_out` (keyword-only; default `None`): Required when aligning a vector with a higher-rank out; keyword of `tl.steer` (keyword-only; default `None`): Required when a vector direction must be aligned with an out axis, avoiding silent batch/position assumptions; keyword of `tl.subspace` (keyword-only; default `-1`) |
| `dim0=` | keyword of `FacetSpec.transpose` (required) |
| `dim1=` | keyword of `FacetSpec.transpose` (required) |
| `dim_others=` | keyword of `tl.receptive_field.node_spec` (keyword-only; default `True`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested |
| `direction=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `ContributionScores.topk` (keyword-only; default `0`): Index into the requested directions axis; keyword of `ReceptiveFieldView.at`, `ReceptiveFieldView.probe_grad`, `ReceptiveFieldView.show` (keyword-only; default `None`): Optional per-call receptive/projective direction override. `None` uses the view's bound direction; keyword of `ReceptiveFieldView.check` (keyword-only; default `None`): Receptive or projective containment direction; keyword of `Trace.draw` (keyword-only; default `None`): Edge flow: `"bottomup"`, `"topdown"`, `"leftright"` or `"rightleft"`. `None` takes the graph's default (forward `"bottomup"`, backward `"topdown"`, combined `"leftright"`); keyword of `Trace.receptive_fields` (keyword-only; default `'receptive'`); keyword of `Trace.to_dagua_graph` (default `'bottomup'`): Forwarded unchanged to `trace_to_dagua_graph`; keyword of `tl.bridge.dialz.from_direction`, `tl.bridge.repeng.from_direction`, `tl.bridge.steering_vectors.from_direction` (required): A TorchLens `Direction`; keyword of `tl.debug.lineage` (keyword-only; default `'ancestors'`): `"ancestors"`, `"descendants"`, or `"both"`; keyword of `tl.experimental.dagua.trace_to_dagua_graph` (default `'bottomup'`); keyword of `tl.intervention.set_direction_mean` (required): 1-D direction vector `v` (finite, nonzero norm); keyword of `tl.neighborhood` (keyword-only; default `'both'`); constructor keyword of `tl.options.VisualizationOptions` (optional): Graph layout direction; keyword of `tl.project_onto` (required): Projection direction; keyword of `tl.project_out` (required): Direction to remove; constructor keyword of `tl.receptive_field.GradReceptiveField` (keyword-only; optional); keyword of `tl.receptive_field.cross_validate` (keyword-only; optional): Receptive or projective containment direction; keyword of `tl.receptive_field.node_spec` (keyword-only; optional); keyword of `tl.steer` (required): Direction tensor |
| `directory=` | keyword of `tl.bridge.pyreft.read` (required): The directory pyreft saved (`config.json` plus `intkey_<key>.bin`), for example a Hugging Face snapshot; keyword of `tl.bridge.sae.write_delphi` (keyword-only; required): Cache root (delphi's `raw_dir`); created if missing |
| `disclosure=` | keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; default `'direct'`): `"direct"` or the output-equivalent wording for lowered engines |
| `disclosures=` | constructor keyword of `tl.transforms.PlannedStep` (default `()`): Plan-time cost/behavior disclosures surfaced to the caller (e.g. the O(D*k) `iid_bernoulli` generation cost, so a multi-second generation inside a harvest cannot surprise anyone). Additive field; empty for most steps |
| `display=` | keyword of `tl.viz.nodes.rdm` (keyword-only; default `'raw'`): Value transform applied before rendering: `"raw"` (default, historical behavior), `"rank"`, or `"percentile"` -- the standard RSA display convention (Nili et al. 2014's percentile-ranked RDMs; thingsvision ships the same rank-scaled display) via `rank_transform_rdm`. .. |
| `distances=` | keyword of `tl.repgeom.rank_transform_rdm` (required): Square symmetric zero-diagonal dissimilarity matrix |
| `distributed_witness=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `distributions=` | constructor keyword of `tl.backends.FrameworkRequirement` (required): Installed distribution names that version the framework, in lookup order (`("tensorflow", "tensorflow-cpu")`) |
| `donor_group_id=` | constructor keyword of `tl.intervention.SamplingPlan` (required) |
| `dpi=` | keyword of `Trace.draw` (keyword-only; default `None`): Typography and raster resolution overrides; constructor keyword of `tl.options.VisualizationOptions` (optional): Optional Graphviz output DPI |
| `draw_bank=` | keyword of `tl.attribution.gradient_shap` (keyword-only; default `None`): Optional stored draws (`alphas`, `pool_indices`, optional `noise`) replayed instead of fresh draws; the stored-draw oracle row pins the estimator convention through this door |
| `dry_run=` | keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `False`): If True, serialize locally and return planned upload metadata without contacting the Hub |
| `dtype=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): Optional save dtype (DOCUMENTED-UNSTABLE): a `torch.dtype`/name or per-key mapping; `None` preserves. Cast after pool/postprocess, before the CPU snapshot; bf16/fp8 store byte-exact in safetensors shards; constructor keyword of `tl.recording.CaptureSpec` (default `None`): Optional target dtype for retained payloads; keyword of `tl.repgeom.rdm` (keyword-only; default `None`): Computation dtype (default `float64`, the historical behavior); constructor keyword of `tl.stats.CKA`, `tl.stats.Covariance`, `tl.stats.CrossCovariance` (keyword-only; default `None`): Floating compute dtype (default: historical `float64`); keyword of `tl.stats.cka` (keyword-only; default `None`): Floating compute dtype (default: historical `float64`); constructor keyword of `tl.transforms.TensorSpec` (required): `str(torch.dtype)` spelling, e.g. `"torch.float32"`; keyword of `tl.transforms.cast` (required): Target float dtype (`float16`/`bfloat16`/`float32`/`float64`) |
| `edge_influence_coverage=` | keyword of `AttributionGraph.to_neuronpedia` (keyword-only; default `0.98`): Shares of total influence kept, between 0 and 1 |
| `edge_overrides=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `Trace.draw` (keyword-only; default `None`): Graphviz attribute overrides for the graph, forward edges, gradient edges and module boxes; constructor keyword of `tl.options.VisualizationOptions` (optional): Forward-edge style overrides |
| `edges=` | keyword of `tl.attribution.causal_audit` (keyword-only; default `None`): For an EDGE table only: the edge names (or `ComponentEdge` objects) of the rows to audit; `None` (default) audits every `ok` row; keyword of `tl.mechinterp.patch_edges` (required): `ComponentEdge` objects, into one receiver or several. A circuit runs as one call: pass every edge outside it, `[e for name, e in graph.edges.items() if name not in kept]`. No edges returns an unpatched fork |
| `edit_axis=` | field and constructor keyword of `SweepGrid`: the axis holding edits, None when edit= served every cell |
| `edited_slice=` | keyword of `FacetSpec.scatter_update` (required): Edited facet slice returned by the user hook |
| `effects=` | constructor keyword of `tl.experiment.SweepGrid` (required) |
| `embedding_module=` | keyword of `tl.attribution.build_graph` (keyword-only; required): Module name whose output rows are the token vectors |
| `emit_nvtx=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `empirical_adjoint_atol=` | keyword of `tl.receptive_field.verify` (keyword-only; default `None`): Non-negative floating-point comparison tolerances used only for the reported equality of two empirical derivative probes. `None` (default) derives the pair per compared-gradient dtype via `layer_grad_tolerances_for_dtype` (fp32 resolves to the legacy layer-grad constants) |
| `empirical_adjoint_rtol=` | keyword of `tl.receptive_field.verify` (keyword-only; default `None`): Non-negative floating-point comparison tolerances used only for the reported equality of two empirical derivative probes. `None` (default) derives the pair per compared-gradient dtype via `layer_grad_tolerances_for_dtype` (fp32 resolves to the legacy layer-grad constants) |
| `enabled=` | keyword of `tl.facets.enable_transformer_lens_aliases` (default `True`): Whether alias keys such as `hook_pattern` should resolve to native TorchLens facets such as `pattern` |
| `encoder=` | constructor keyword of `tl.stats.Quantile` (keyword-only; default `None`): Optional callable applied to each batch before it is folded, for example an SAE's encoder to estimate per-latent quantiles; constructor keyword of `tl.stats.TopK` (keyword-only; default `None`): Optional callable applied to each batch before it is folded, for example an SAE's encoder to rank examples per latent; keyword of `tl.stats.load_fitted` (keyword-only; default `None`): The `encoder=` callable the saved accumulator was built with (it is not stored in the file); required exactly when one was set |
| `engine=` | keyword of `tl.experiment.sweep_grid` (keyword-only; default `'live_hook'`); keyword of `tl.experiment.sweep_sites` (keyword-only; default `'live_hook'`): `"live_hook"` (default; fork + do(site, hook, model=model, inputs=x), a real forward) or `"replay"` (fork + do on saved payloads). The lane is stamped in the envelope and the effect table; keyword of `tl.mechinterp.patch_grid` (keyword-only; default `'rerun'`): `"rerun"` (live-hook route). `"replay"` refuses typed until the known HF replay crash (FIX-A) lands; keyword of `tl.neuro.plan_extraction` (keyword-only; default `'trace'`): `"trace"` only at launch; constructor keyword of `tl.options.InterventionOptions` (optional): Propagation engine: `"auto"`, `"replay"`, `"rerun"`, or `"set_only"` |
| `entries=` | constructor keyword of `tl.intervention.RegionBoundary` (required) |
| `episode=` | keyword of `tl.trace` (keyword-only; default `None`): Torch-only episode capture declaration (`EpisodeOptions`) |
| `episode_id=` | keyword of `Bundle.episode_status` (required): Episode entity named by `episode_member` relation rows |
| `episode_ledger=` | keyword of `Bundle.episode_status` (keyword-only; default `None`): Optional episode ledger; supplies `n_steps_declared` and the driver-halt geometry (fold arms 2 and 4 are ledger-only facts and degrade fail-closed to `episode_unknown` without it) |
| `eps=` | constructor keyword of `tl.transforms.SrpSizing` (required): Requested maximum pairwise-L2 distortion; keyword of `tl.transforms.srp_dims_for` (default `0.1`): Maximum pairwise-L2 distortion target, in `(0, 1)`; keyword of `tl.transforms.unit_norm` (default `1e-12`): Norm floor guarding division by zero |
| `escalated_from=` | constructor keyword of `tl.options.EpisodeOptions` (default `None`): Producer digest of the cheap-tier product this capture escalates (present iff escalation; travels with `reason` — E-A2). Build the whole escalation declaration with `torchlens._capture._episode_ledger.escalation_spec(producer, ...)` |
| `event_stream=` | constructor keyword of `tl.debug.ChecksSession` (keyword-only; default `None`): Optional external `EventStream`; by default the session owns one; constructor keyword of `tl.debug.LivenessProbe` (keyword-only; default `None`); constructor keyword of `tl.debug.RangeProbe` (keyword-only; default `None`): Optional stream to publish per-fire verdict events into; keyword of `tl.trackers.watch` (keyword-only; default `None`): A chassis stream whose check verdicts serialize as `torchlens/check/<name>` rows |
| `every=` | keyword of `ChecksSession.register_nonfinite_param_check` (keyword-only; default `1`); keyword of `tl.experimental.auto_capture` (default `100`): Capture interval; constructor keyword of `tl.trackers.HFTrainerWatchCallback`, `tl.trackers.LightningWatchCallback` (keyword-only; default `None`); keyword of `tl.trackers.watch` (keyword-only; default `50`): Scalar cadence on the caller's step axis (first and last eligible steps always sample) |
| `evidence=` | constructor keyword of `tl.transforms.RoleDeclaration` (default `'declared'`): Member of `ROLE_EVIDENCE`. `"declared"` records a caller assertion; `"op_contract"` is the v1.5 trace-derived producer |
| `evidence_grade=` | constructor keyword of `tl.types.InputProvenance` (default `None`): `verified_exact` / `empirically_flexible` / `under_determined` / `unsupported` -- exact-verification evidence held separately from authoritative-canonical evidence |
| `exact=` | keyword of `ReceptiveFieldRuleContext.full`, `ReceptiveFieldRuleContext.window` (keyword-only; default `True`); keyword of `ReceptiveFieldRuleContext.piecewise` (keyword-only; default `False`); keyword of `ReceptiveFieldRuleContext.window_edges` (keyword-only; required) |
| `example_ids=` | keyword of `TopK.update` (keyword-only; default `None`): Per-unit mode only: one integer id per example (axis 0) of this batch, naming the examples instead of their stream index; constructor keyword of `tl.stats.UnitTopK` (required): `int64 [units, k]`: the example each value came from (its index in the stream of examples, unless `update(example_ids=...)` named them); `-1` in empty slots |
| `example_reduction=` | keyword of `tl.mechinterp.edge_scores` (keyword-only; default `'mean'`): Keyword-only. `"mean"` (default) averages the per-example scores over the batch; `"sum"` adds them |
| `examples=` | keyword of `tl.bridge.lit.dataset` (required): Strings, or dicts carrying `text` (and optionally `label`) |
| `exc=` | keyword of `CaptureBackend.cleanup_failed_forward_session` (required): Exception the capture door is about to raise: a `TraceCaptureError` wrapping the forward's exception, or a TorchLens error propagating as itself; keyword of `tl.agent.build_error_envelope` (required): Raised exception (typed TorchLens refusals carry `fields`) |
| `exception=` | keyword of `PartialTrace.from_trace` (required): Original exception raised during capture |
| `exclude_bos=` | keyword of `tl.bridge.sae.splice` (keyword-only; default `False`): Leave sequence position 0 out: it keeps its clean value in the spliced and zero-ablated runs and is excluded from every metric (the Gemma Scope evaluation convention). Needs a `(batch, sequence, features)` site |
| `excluded=` | constructor keyword of `tl.backends.FrameworkRequirement` (default `()`): Versions inside the range that are refused (broken wheels) |
| `execution_mode=` | keyword of `DevicePeaks.for_device` (required); keyword of `Trace.summary` (keyword-only; default `None`): ONE-CALL-ONLY spellings: valid on `tl.summary(model, ...)`, typed errors here (this method reports an existing capture); constructor keyword of `tl.report.PeakRow` (required) |
| `execution_policy=` | constructor keyword of `tl.types.InputProvenance` (default `'caller_mode'`): `"caller_mode"` (tl.trace keeps the caller's flags) or `"eval_no_grad_restored"` (summary/render pin and restore) |
| `exempted_reason_counts=` | keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unverified` (keyword-only; default `None`): Stable reason-code counts for provable replay exemptions |
| `exits=` | keyword of `TraceSlice.as_region` (keyword-only; default `None`): Optional user-declared exit list, checked for completeness (a leaking list refuses, naming the leaked edges); constructor keyword of `tl.intervention.RegionBoundary` (required) |
| `expected=` | keyword of `tl.fingerprint.assert_unchanged` (required): Previously pinned structural hash, or `None` to bootstrap one |
| `expected_ranks=` | keyword of `tl.merge_ranks` (keyword-only; default `None`): Optional declared world. Widens presence expectations only: a declared rank without a core is a presence gap; declaring fewer ranks than the recorded group memberships never narrows anything; keyword of `tl.merge_report` (keyword-only; default `None`) |
| `expected_tokens=` | constructor keyword of `tl.options.EpisodeOptions` (default `None`): The cheap-tier product's per-step token column (one tuple per step), carried so the escalated capture can discharge the E-A3 fidelity obligation at write time: prefix-equal columns record `fidelity_basis="tokens"`; a mismatch records `"diverged"` — the escalated product is still a valid capture .. |
| `expert=` | constructor keyword of `tl.mechinterp.Coordinate` (default `None`): MoE expert index when the writer is a routed expert; `None` today |
| `explainer_class=` | keyword of `tl.bridge.shap.explain` (keyword-only; default `None`): Optional explainer class or factory. Defaults to `shap.DeepExplainer` |
| `exploding_threshold=` | keyword of `ChecksSession.register_magnitude_check` (keyword-only; default `10000.0`); keyword of `tl.debug.grad_health` (keyword-only; default `10000.0`): Norm above which a finite gradient is flagged exploding |
| `extra=` | constructor keyword of `tl.attribution.AttributionMetric` (required): Additional disclosure (distributions, probes, qualifications); constructor keyword of `tl.backends.FrameworkRequirement` (required): The pyproject extra that installs the framework (`""` for a base dependency); keyword of `tl.trackers.build_manifest` (keyword-only; default `None`) |
| `extra_address=` | keyword of `CaptureBackend.log_source_tensor` (default `None`): Optional backend-specific input or buffer address |
| `extras=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (optional): Adapter-specific disclosures that are NOT compared (kept for the provenance record, e.g. `crop_pct`, processor class names) |
| `facet=` | keyword of `tl.experiment.head_ablation_candidates` (keyword-only; default `'v'`): The write-capable facet lowered to; `"v"` (default) is the equivalence-proven lowering for standard MHA; keyword of `tl.mechinterp.measure_trajectory` (keyword-only; default `'resid_post'`): Hidden-state facet to project, as in `logit_lens` |
| `facets=` | keyword of `tl.facets.register` (keyword-only; default `()`): Internal declared facet names used for cheap `keys()`; keyword of `tl.trace` (keyword-only; optional): Per-trace additive facet recipes captured into the immutable registry snapshot for the returned trace; keyword of `tl.trackers.watch` (keyword-only; default `None`): `grain="op"` only: `{op_label: splitter}` declaring facet slices of op sites (`torchlens._observability.split_heads(n)` slices attention heads); each slice is its own series under `<op label>/<facet>` |
| `factor=` | constructor keyword of `tl.debug.Watchdog` (required); keyword of `tl.scale` (required): Multiplicative factor |
| `factory=` | keyword of `tl.appliances.register` (required): Callable called with the trace on every `trace.using(name)`, or a `"module:attribute"` string naming such a callable (imported lazily) |
| `faithfulness=` | keyword of `tl.mechinterp.circuit_model_difference`, `tl.mechinterp.circuit_performance_ratio` (required): Faithfulness at each swept edge fraction: a sequence (one value per fraction) or a `{fraction: faithfulness}` mapping |
| `family=` | keyword of `TagGrammar.data` (required); keyword of `tl.io.register_sidecar_family` (required): The family contract. `family_id` must be namespaced (`"<owner_ns>.<name>"`), `schema_id` non-empty, `version` a positive int, `size_budget_bytes` positive |
| `family_id=` | keyword of `tl.io.attach_sidecar`, `tl.io.read_sidecar`, `tl.io.unregister_sidecar_family` (required) |
| `fast=` | keyword of `Trace.run` (keyword-only; default `False`): Explicit stateful static-loop mode. The live provider runs native `forward` with targeted module/function collection and a per-call path/shape guard. The loaded provider performs one ordinary verified run, then reuses staged state and compiled argument binders. Both keep the source Trace .. |
| `fastlog=` | constructor keyword of `tl.backends.BackendCapabilities` (required): Whether sparse `tl.record` capture is supported |
| `features=` | keyword of `tl.attribution.deletion_curve` (keyword-only; default `'auto'`): Feature grouping: `"auto"` (default: `"pixels"` for `(N, C, H, W)` inputs, `"tokens"` for `(N, T, D)` inputs, else `"elements"`), `"elements"` (every scalar), `"pixels"` (all channels at one position), `"patches"` (square patches of `patch_size`) or `"tokens"` (all values of ..; keyword of `tl.attribution.insertion_curve` (keyword-only; default `'auto'`): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.road_curve` (keyword-only; default `'auto'`): Feature grouping as in `deletion_curve` (default `"auto"`: pixels for images); keyword of `tl.transforms.srp_fidelity_probe` (required): Pilot activations, `(n, ...)` with the stimulus axis leading (flattened per row); `n >= 4` so the RDMs have enough pairs |
| `feed=` | constructor keyword of `tl.options.EpisodeOptions` (default `'open'`): Join-arm strictness (lane F40c). `"open"` (default) captures across cross-step feed breaks and grades every join in the measured `step_join` envelope; episode-dependent claims refuse across a broken join, everything else stays usable. `"closed"` is the opt-in STRICT arm: the feed is declared .. |
| `fetch=` | keyword of `tl.preprocessing.unknown_resolution` (keyword-only; default `None`): Optional fetch disclosure block (attempted / performed / policy / error) from a network-gated adapter |
| `field_name=` | keyword of `NarrationOptions.is_field_explicit`, `ReplayOptions.is_field_explicit`, `StorageOptions.is_field_explicit`, `VisualizationOptions.is_field_explicit` (required) |
| `file=` | keyword of `Module.show_call_tree`, `ModuleCall.show_call_tree`, `Trace.show_call_tree` (default `None`): Optional output stream. `None` prints to stdout; keyword of `SummaryReport.print` (default `None`) |
| `file_format=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `Trace.draw` (keyword-only; default `None`): Format for a path with no extension, or a bare call (`"pdf"`, or `"svg"` inline); a different format than the path's extension refuses; keyword of `tl.dataset_extraction.export_extraction` (keyword-only; default `'npy'`): `"npy"` (base install) \| `"hdf5"` \| `"mat"` (both behind the named optional extra; lazy imports); keyword of `tl.debug.sketch_grad_fn` (keyword-only; default `'png'`): Graphviz output format (`png`, `svg`, `pdf`); constructor keyword of `tl.options.VisualizationOptions` (optional): Graph output file format; keyword of `tl.viz.bundle_diff` (keyword-only; default `'svg'`): Output format. Any Graphviz-supported format (e.g., `"svg"`, `"pdf"`, `"png"`). Accessibility metadata is only embedded for `"svg"` |
| `finalized_reason=` | keyword of `ChecksSession.report` (keyword-only; default `'on_demand'`) |
| `fire_record=` | field of `GradFnCall` |
| `fit_scope=` | keyword of `PCA.fitted` (default `'unspecified'`): Recorded disclosure of what the fit consumed (fitting across held-out stimuli leaks analysis information; say so here) |
| `fitted=` | keyword of `tl.stats.save_fitted` (required): The fitted payload, or the accumulator to save; keyword of `tl.transforms.pca_apply` (required): A `FittedPCA` payload (from `tl.stats.PCA(...).fitted(...)` or `tl.stats.load_fitted`) |
| `flag=` | keyword of `tl.backends.require_capability_implementation` (required): Gated capability flag name from `GATED_CAPABILITY_FLAGS` |
| `flatten=` | keyword of `tl.types.register_container` (required): Callable returning `(children, aux_data)` for an instance |
| `flexible_dims=` | constructor keyword of `tl.types.InputProvenance` (default `None`) |
| `flop_convention=` | keyword of `Trace.summary` (keyword-only; default `None`): `"fma2"` (stored convention; default) \| `"fma1"` |
| `flops_attributions=` | field of `BackwardStatus` |
| `flops_by_mode=` | keyword of `tl.report.mfu` (keyword-only; required) |
| `fn=` | keyword of `Bundle.map` (required): Callable receiving one member log. A callable that accepts a second positional parameter additionally receives the member's NAME (`bundle.map(lambda log, name: ...)`); single-parameter callables keep the historical contract unchanged; keyword of `tl.export.register_target` (required): The exporter callable (`fn(log, ...)`); constructor keyword of `tl.transforms.OpaqueStep` (default `None`): The live callable, or `None` when rehydrated from an artifact record (identification cannot resurrect code; applying an unresolved opaque step refuses typed); keyword of `tl.transforms.with_context` (required): Callable taking `(tensor, ctx)` |
| `fold_patterns=` | keyword of `Trace.draw` (keyword-only; default `None`): Declarative user-named pattern folding (DOCUMENTED-UNSTABLE): `None` = off; `"idiomatic"` selects the curated conv-bn-relu preset; a mapping declares named linear paths such as `{"ConvBnRelu": "conv2d > batch_norm > relu"}`. Combining with automatic collapse refuses typed .. |
| `fold_repeats=` | keyword of `Trace.draw` (keyword-only; default `None`): Repeat-fold policy. `None` keeps the default (off for `collapse="none"`, band-pressure folding otherwise), `True` folds every eligible repeated run, `False` disables folding; keyword of `Trace.summary` (keyword-only; default `None`): `"auto"`/`True` fold repeated sibling runs; `False` restores every member; constructor keyword of `tl.options.VisualizationOptions` (optional): Repeat-fold policy. `None` preserves the collapse mode default, `True` folds every eligible run, and `False` disables run folding; keyword of `tl.report.cost_tree` (keyword-only; default `True`): Fold structurally identical sibling subtrees under a strict byte-equal signature, annotated `(xN)` |
| `font_size=` | keyword of `Trace.draw` (keyword-only; default `None`): Typography and raster resolution overrides; constructor keyword of `tl.options.VisualizationOptions` (optional): Optional Graphviz font size |
| `force=` | keyword of `tl.io.cleanup_tmp` (keyword-only; default `False`): Whether temp dirs without a `PARTIAL` sentinel, and backup dirs that are not provably redundant, should also be removed; keyword of `tl.recording.cleanup_partial` (keyword-only; default `False`): Whether candidates without `PARTIAL` sentinels may be removed |
| `forced_tokens=` | constructor keyword of `tl.options.EpisodeOptions` (default `None`): Optional teacher-forced feed declaration: the token sequence the driver feeds instead of model emissions. Declaring it stamps `token_feed="forced"` and `fidelity_basis="forced"` on the ledger header — an explicitly NON-VERIFYING disclosed mode; token fidelity obligations (E-A3) never verify a .. |
| `formula=` | constructor keyword of `tl.transforms.SrpSizing` (required): Formula id (`"dense_jl_l2_heuristic"`) |
| `fraction=` | keyword of `tl.top_changed` (keyword-only; default `None`); keyword of `tl.top_fraction` (default `0.01`) |
| `fractions=` | keyword of `tl.mechinterp.circuit_model_difference`, `tl.mechinterp.circuit_performance_ratio` (keyword-only; default `None`): Keyword-only. The edge fractions of a sequence; `None` means the ten MIB fractions (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1) |
| `frame=` | keyword of `CaptureBackend.pop_module_frame` (required): Expected top frame to pop; keyword of `CaptureBackend.push_existing_module_frame` (required): Existing backend frame to push |
| `frontier_output=` | constructor keyword of `tl.recording.StopSignal` (default `None`): Optional live output object used by `trace(stop_after=...)` to finalize a partial graph at the halt boundary |
| `frozen=` | keyword of `tl.attribution.build_graph` (keyword-only; optional): Module names whose outputs are held at their captured values, such as attention-pattern and normalisation-scale modules, op labels for values computed inside a module (`"rsqrt_1_5"`), or facet selectors such as `tl.facet("pattern")`, which hold the op behind that facet in every module that ..; keyword of `tl.attribution.read` (keyword-only; default `torchlens.attribution.onebackward.DEFAULT`): Omitted -> the default MLP-output linearization; `None` -> total derivative; explicit ACT selection/site specs -> exactly that set, replacing the default |
| `func=` | keyword of `tl.site` (keyword-only; default `None`): TorchLens layer type (`"linear"`, `"relu"`, ...) |
| `func_call_id=` | constructor keyword of `tl.types.OpRef` (keyword-only; required) |
| `func_event_input=` | keyword of `CaptureBackend.build_record_context`, `CaptureBackend.detect_backend_semantics`, `CaptureBackend.emit_function_outputs` (required) |
| `generalize_batch=` | keyword of `TensorSpec.from_tensor` (keyword-only; default `True`): When true (default), record the batch extent as `None` so the plan holds for every batch size |
| `generate_kwargs=` | keyword of `tl.record` (keyword-only; default `None`): DOCUMENTED-UNSTABLE. When given (`{}` for the model's defaults), run the model's own `generate(*input_args, **input_kwargs, **generate_kwargs)` once instead of one forward, and record the selected sites at every forward it runs (the prefill, then each decode step) while the KV cache is on, so .. |
| `generated_texts=` | keyword of `tl.bridge.inseq.attribute` (keyword-only; default `None`): Optional target text or batch, forwarded as inseq's own `generated_texts=` (forced decoding: the text to attribute instead of generating). The attributed text is checked against it |
| `generation=` | keyword of `tl.mechinterp.alias_report` (keyword-only; default `'2.x'`) |
| `global_step=` | keyword of `ChecksSession.step` (keyword-only; default `None`); keyword of `WatchSession.step` (required) |
| `grad=` | keyword of `ReceptiveFieldView.show` (keyword-only; default `False`): Whether to include empirical gradient magnitude; constructor keyword of `tl.receptive_field.GradReceptiveField` (required); keyword of `tl.receptive_field.node_spec` (keyword-only; default `False`) |
| `grad_edge_overrides=` | keyword of `Trace.draw` (keyword-only; default `None`): Graphviz attribute overrides for the graph, forward edges, gradient edges and module boxes; constructor keyword of `tl.options.VisualizationOptions` (optional): Gradient-edge style overrides |
| `grad_fn_handle=` | keyword of `TapObserver.record_backward` (keyword-only; default `None`): GradFn site whose hook fired |
| `grad_input=` | keyword of `TapObserver.record_backward` (required): Output gradient (gradient site) or the autograd `grad_input` tuple (node site) |
| `grad_mode=` | keyword of `Trace.summary` (keyword-only; default `None`): ONE-CALL-ONLY spellings: valid on `tl.summary(model, ...)`, typed errors here (this method reports an existing capture) |
| `grad_options=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `grad_output=` | keyword of `TapObserver.record_backward` (keyword-only; default `None`): Autograd grad_output tuple at the grad_fn_handle hook, when available |
| `grad_scale=` | keyword of `tl.debug.bisect_nan_backward` (keyword-only; default `None`): Disclosed AMP loss scale (`scaler.get_scale()`); recorded on the result. The transition ledger classifies non-finiteness, which no positive scale changes, so this is DISCLOSURE, not arithmetic; keyword of `tl.debug.grad_health` (keyword-only; default `None`): Positive loss scale the captured backward ran under (AMP: `scaler.get_scale()`); see `gradient_flow_audit_rows` |
| `grad_scales=` | keyword of `tl.debug.grad_health` (keyword-only; default `None`): Per-backward scale stamps `{bwd: scale}`; see `gradient_flow_audit_rows` |
| `grad_transform=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `grain=` | keyword of `tl.trackers.watch` (keyword-only; default `None`): `None` (module grain) or `"op"`: per-op activations through the Route-B op-tier collector, behind `allow_forward_wrap=True`. Only `signals=("activations",)`; `inputs=` and `select=` required; only the first forward of a sampled step is observed |
| `grains=` | keyword of `tl.trackers.build_manifest` (keyword-only; default `None`) |
| `grammar=` | constructor keyword of `tl.trackers.WatchSession` (required); keyword of `tl.trackers.build_manifest`, `tl.trackers.emission_from_block` (required) |
| `granularity=` | keyword of `tl.export.netron` (keyword-only; default `'module'`): `"module"` (default, ruling N2): per-call FunctionProto hierarchy, one-click drill-down. `"op"`: every leaf op flat. `"rolled"`: repeated passes merged by rolled identity, variance-gated types |
| `graph_module=` | keyword of `tl.compat.from_fx` (required): `torch.fx.GraphModule` or graph-module-like object |
| `graph_overrides=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `Trace.draw` (keyword-only; default `None`): Graphviz attribute overrides for the graph, forward edges, gradient edges and module boxes; constructor keyword of `tl.options.VisualizationOptions` (optional): Graphviz graph-level style overrides |
| `grid=` | keyword of `tl.transformer.receipt_from_patch_grid` (required): A `PatchGrid` with a `head` axis. Each effect is the row's patched metric minus the grid's `receipts["corrupted_metric"]`: the grid patches clean values into corrupted reruns, so the unpatched corrupted run is the baseline and a cell that changes nothing reads .. |
| `ground_truth=` | keyword of `BackendProtocol.validate` (required) |
| `ground_truth_output_tensors=` | keyword of `Trace.validate_forward_pass` (required): Ground-truth model outputs. A bare tensor -- what `model(x)` naturally produces for a single-output model -- is normalized to `[tensor]`: iterating it directly counted the tensor's ROWS as expected outputs, so a byte-correct capture reported a validation FAILURE (``"1 logged vs <batch> .. |
| `group=` | keyword of `GqaInfo.group_header`, `GqaInfo.query_heads` (required) |
| `group_by=` | keyword of `Population.mean` (keyword-only; default `None`): Optional agreement key callable over member datums; keyword of `Population.std` (keyword-only; default `None`); keyword of `Trace.profile` (keyword-only; default `None`): `"source_line"` folds op rows into one row per user source line (activation bytes, FLOPs and time summed; the groups sum to the ungrouped view). Only with `level="op"`. DOCUMENTED-UNSTABLE; keyword of `tl.intervention.resample_rows_from` (keyword-only; default `None`): Optional datum -> key callable over population datums; keyword of `tl.report.profile` (keyword-only; default `None`): `"source_line"` folds the op rows into one row per user source line (the op's recorded call site), summing its activation bytes, FLOPs and time; boundary rows fold into one row, so every additive column still sums to the ungrouped op view. Only valid with `level="op"`. `None` (default) keeps .. |
| `group_index=` | keyword of `OptimizerFacts.group_fact` (required) |
| `grouping=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `handle=` | keyword of `Trace.clear_edits` (default `None`): Optional hook handle id (or the tuple of ids one `do` call staged) |
| `handler=` | constructor keyword of `tl.agent.AgentToolSpec` (required) |
| `has_carried_state=` | field of `RunReport` |
| `has_constancy_hint=` | field of `ClipLedgerSnapshot` |
| `has_identical_target_resolution=` | field of `SpecCompat` |
| `has_intervention=` | field of `NarrationEvent` |
| `has_predicate=` | constructor keyword of `tl.facets.FacetFunction` (required): Whether this recipe has an arbitrary predicate matcher |
| `has_rescue_rerun=` | field of `ProfileReport` |
| `has_scaler=` | field of `ScaleLedgerSnapshot` |
| `has_streaming=` | constructor keyword of `tl.backends.BackendCapabilities` (required): Whether streaming save is supported |
| `has_validation_replay=` | field and constructor keyword of `BackendCapabilities` |
| `head=` | keyword of `AttentionView.head_view` (required); keyword of `Trace.decode_output` (keyword-only; default `None`): Output head to decode, as an output container path such as `"logits"`, `"0"` or `"0.logits"`; keyword of `Trace.output_table` (keyword-only; default `None`): Optional output head for a decode on demand (see `decode_output`); keyword of `tl.mechinterp.patch_head`, `tl.transformer.score_decomposition` (keyword-only; required): Query head index |
| `head_index=` | keyword of `FacetView.head` (required) |
| `heads=` | keyword of `tl.experiment.head_ablation_candidates` (keyword-only; required): Head indices: an int `n` means `range(n)`; or an explicit sequence of indices; keyword of `tl.mechinterp.patch_grid` (keyword-only; default `None`): Head indices for the head axis (default: all); keyword of `tl.mechinterp.patch_query_heads_grid` (keyword-only; default `None`): Query-head indices (default: all query heads); constructor keyword of `tl.transformer.Annotation` (required): Original head indices, aligned with `values`; keyword of `tl.viz.attention` (keyword-only; default `None`): Optional subset of ORIGINAL head indices to render |
| `height=` | keyword of `tl.viz.heatmap` (keyword-only; default `240`): Output image height in pixels; keyword of `tl.viz.lineplot` (keyword-only; default `180`): Output image height in pixels; keyword of `tl.viz.nodes.histogram` (default `160`): Rendered image height; keyword of `tl.viz.nodes.scree` (keyword-only; default `180`): Height of the rendered scree image in pixels |
| `hi_exp=` | constructor keyword of `tl.stats.HistogramDescriptor` (default `16`) |
| `high=` | keyword of `tl.clamp` (keyword-only; default `None`): Optional upper bound; keyword of `tl.saturated` (keyword-only; default `None`) |
| `highlight=` | keyword of `Trace.draw` (keyword-only; default `None`): Keyword-only. Named highlight drawn over the graph. `None` (the default) draws the plain graph. `"interventions"` draws the surgery lens: every recorded intervention is marked on its node (solid border = recorded fact, dashed = heuristic attribution), the caption carries the per-lane census, .. |
| `hist_every=` | constructor keyword of `tl.trackers.HFTrainerWatchCallback`, `tl.trackers.LightningWatchCallback` (keyword-only; optional); keyword of `tl.trackers.watch` (keyword-only; default `None`): Histogram cadence; may run coarser than scalars. `None` disables the sketch tier (scalars remain) |
| `history_id=` | keyword of `Bundle.effects`: which recorded history to read; constructor keyword of `tl.experiment.SweepGrid` (required) |
| `history_size=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional); keyword of `tl.recording.dry_run` (keyword-only; default `8`): Fastlog dry-run options. Predicate exceptions propagate immediately |
| `home=` | keyword of `FacetSpec.from_home` (required): Runtime home object; constructor keyword of `tl.facets.FacetSpec` (default `None`): Runtime home object. This is intentionally non-portable |
| `home_address=` | constructor keyword of `tl.facets.FacetSpec` (default `None`): Stable home address when available |
| `home_kind=` | keyword of `FacetSpec.from_home` (keyword-only; default `'op'`): Kind of home value; constructor keyword of `tl.facets.FacetSpec` (required): Kind of home value |
| `home_label=` | constructor keyword of `tl.facets.FacetSpec` (default `None`): Stable home label when available |
| `home_out=` | keyword of `FacetSpec.scatter_update` (required): Full home-op output tensor passed to the intervention hook |
| `hook_type=` | keyword of `Direction.upload_to_neuronpedia` (keyword-only; default `'resid_post'`): `"resid_post"` (a decoder block's output, the default) or `"resid_pre"` (its input) |
| `hooks=` | constructor keyword of `tl.options.CaptureOptions` (optional); constructor keyword of `tl.options.ReplayOptions` (optional): Optional replay hook mapping |
| `hooks_or_site=` | keyword of `Trace.do` (required): Mapping/list batch input or selector-like site |
| `hops=` | keyword of `tl.neighborhood` (default `1`) |
| `idempotent=` | constructor keyword of `tl.agent.AgentToolSpec` (default `True`) |
| `identifier=` | keyword of `tl.bridge.brain_score.activations_extractor` (keyword-only; default `None`): Brain-Score activations identifier; defaults to the model class name, matching `PytorchWrapper` |
| `ids=` | constructor keyword of `tl.transformer.TokenAxis` (default `None`): Optional tokenizer ids aligned with `tokens` |
| `image=` | keyword of `ReceptiveFieldView.show` (keyword-only; default `None`): Optional source image override; keyword of `Trace.annotate` (keyword-only; default `None`): Optional local image path used by the existing `NodeSpec.image` render hook; keyword of `tl.attribution.grad_cam` (keyword-only; default `None`): Optional source image override for the rendered overlay; keyword of `tl.bridge.transformers.trace_image` (required): PIL image or non-empty list of PIL images |
| `images=` | keyword of `tl.viz.image_scatter` (keyword-only; default `None`): Optional PIL images to paste at scatter positions; keyword of `tl.viz.montage` (required): Input images to render |
| `in_progress=` | keyword of `tl.dataset_extraction.open_extraction` (keyword-only; default `False`): Open a still-writing artifact's trusted committed prefix; `reader.refresh()` rescans |
| `include_activations=` | keyword of `Trace.save` (keyword-only; default `False`): Archive exactly the capture-time `save=`-selected payloads for inspection and eligible byte-exact attestation; those payloads never seed execution |
| `include_atomic=` | keyword of `Module.show_call_tree`, `ModuleCall.show_call_tree`, `Trace.show_call_tree` (default `True`): Whether to include atomic module leaves |
| `include_baseline=` | keyword of `tl.sweep` (keyword-only; default `False`): Whether to ALSO capture one PRISTINE (un-intervened) trace first, named `"baseline"` and set as the Bundle baseline — without it a sweep bundle has no comparison anchor and `most_changed` raises `BaselineUndeterminedError`. The default `False` keeps the legacy .. |
| `include_buffer_values=` | keyword of `Trace.save` (keyword-only; default `None`): Persist buffer values (`None` follows the level default); constructor keyword of `tl.options.StorageOptions` (keyword-only; optional): Whether captured pre-forward buffer values (`Trace._buffer_initial_values`) are persisted in the streamed bundle. The streamed-bundle counterpart of `trace.save(..., include_buffer_values=)` (R62 buffer extension); keyword of `tl.to_disk` (keyword-only; default `True`): Whether captured pre-forward buffer values are persisted in the streamed bundle (the `Trace.save` opt-out, mirrored for streaming) |
| `include_buffers=` | keyword of `ChecksSession.register_nonfinite_param_check` (keyword-only; default `True`); keyword of `tl.debug.compare_params` (keyword-only; default `False`): Also compare registered buffers (rows carry `kind="buffer"`); keyword of `tl.debug.param_health` (keyword-only; default `True`): Whether registered buffers join a module inventory |
| `include_custom_attributes=` | keyword of `Trace.save` (keyword-only; default `True`): Persist user-set custom attributes; constructor keyword of `tl.options.StorageOptions` (keyword-only; optional): Whether harvested module attributes (`Module.custom_attributes`) are persisted verbatim in the streamed bundle. The streamed-bundle counterpart of `trace.save(..., include_custom_attributes=)` (R62: the streaming path had no opt-out and reopened the token-leak class the save-path fix closed); keyword of `tl.to_disk` (keyword-only; default `True`): Whether harvested module attributes are persisted verbatim in the streamed bundle (the `Trace.save` opt-out, mirrored for streaming) |
| `include_decoded_output_summary=` | keyword of `Trace.to_pandas` (default `False`): If `True`, add a gated `decoded_output_summary` column populated only on output-node rows. The default remains schema-stable |
| `include_evidence=` | keyword of `ProfileReport.to_pandas` (keyword-only; default `False`): Interleave per-cell evidence columns (`<column>_evidence`) into the frame. The labels are the same `honesty` projection; the `TOTAL` row carries no evidence labels |
| `include_grad_edges=` | keyword of `Trace.to_dagua_graph` (default `None`): Forwarded unchanged to `trace_to_dagua_graph`; keyword of `tl.experimental.dagua.trace_to_dagua_graph` (default `None`) |
| `include_grads=` | keyword of `Trace.save` (keyword-only; default `None`): Persist gradient payloads (`None` follows the level default) |
| `include_mid=` | keyword of `tl.mechinterp.residual_accumulation` (keyword-only; default `True`): Include intermediate merges (the attention add inside a block). Real merges only; nothing is fabricated for parallel blocks |
| `include_outs=` | keyword of `Trace.save` (keyword-only; default `None`): Persist activation payloads (`None` follows the level default) |
| `include_rng_states=` | keyword of `Trace.save` (keyword-only; default `None`): Persist RNG states (`None` follows the level default) |
| `include_saved_args=` | keyword of `Trace.save` (keyword-only; default `None`): Persist saved call arguments (`None` follows the level default) |
| `include_self=` | keyword of `ModuleCall.walk_descendants` (default `True`): Whether to yield this call before its descendants |
| `include_source=` | keyword of `Trace.save` (keyword-only; default `True`): Embed the captured model source code (default `True`); `False` strips verbatim source, docstrings, and source-file references. Absolute source paths are always relativized to a bare basename |
| `include_source_events=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional); keyword of `tl.recording.dry_run` (keyword-only; default `False`): Fastlog dry-run options. Predicate exceptions propagate immediately |
| `include_timing=` | keyword of `tl.trackers.build_graph_ir` (keyword-only; default `True`) |
| `include_totals=` | keyword of `ProfileReport.to_pandas` (keyword-only; default `False`): Append the explicit `TOTAL` row (`kind="total"`): the additive families carry the whole-capture partition totals -- the sort/top_k view above it never changes these denominators |
| `include_unmatched=` | keyword of `tl.viz.bundle_diff` (keyword-only; default `False`): Whether to add unpaired nodes with gray borders |
| `include_weights=` | keyword of `Trace.save` (keyword-only; default `False`): For runnable saves, bundle the full capture-time `state_dict`: all named parameters and persistent buffers, as state rather than a reconstructed model |
| `incomplete_reasons=` | constructor keyword of `tl.types.ModuleInputSnapshot` (required): Stable reasons explaining incomplete evidence |
| `indent=` | keyword of `CheckReport.to_json`, `ParamHealthReport.to_json` (keyword-only; default `1`) |
| `index=` | keyword of `FacetSpec.select` (required); keyword of `RaggedBatch.row` (required): Row index within this carrier; keyword of `tl.attribution.seed` (keyword-only; default `None`): Output-element coordinates for a one-hot cotangent; keyword of `tl.head` (required): Zero-based attention head index; keyword of `tl.transforms.take_index` (required): Position to select (negative indexing legal) |
| `indices=` | keyword of `tl.units` (required): Either a bool mask over the site's output index space, or an iterable of integer coordinate tuples (ints accepted for 1-d sites) |
| `inference_only=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `input_container_structure=` | constructor keyword of `tl.backends.BackendCapabilities` (default `'none'`): Public input-container structure level. `"none"` means no stable paths/specs, `"paths_only"` means path metadata may exist but is not generally reconstructable, and `"full_spec"` means full portable `ContainerSpec` metadata is supported |
| `input_dict=` | keyword of `tl.bridge.transformers.trace_multimodal` (required): Dict containing modality keys such as `text` and `images` |
| `input_dtype=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `None`): Optional primary input dtype override |
| `input_kind=` | keyword of `tl.repgeom.classical_mds` (keyword-only; default `'auto'`): How to read `data`: `"auto"` detects a distance matrix from structure (and warns when the structure is ambiguous), `"distances"` declares a precomputed pairwise distance matrix, and `"features"` declares row-wise feature data even when it is square and symmetric; keyword of `tl.repgeom.rdm` (keyword-only; default `'activations'`): `"activations"` (default) or `"batched"` for `[B, N, ...]` |
| `input_kwargs=` | keyword of `CaptureBackend.setup_inputs_and_device` (required); keyword of `Session.invoke`, `tl.validate` (default `None`): Keyword model input; keyword of `tl.attribution.causal_audit`, `tl.attribution.deconvolution`, `tl.attribution.deletion_curve`, `tl.attribution.grad_cam`, `tl.attribution.gradient_shap`, `tl.attribution.guided_backprop`, `tl.attribution.infidelity`, `tl.attribution.layer_attribution`, `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients`, `tl.attribution.occlusion_map`, `tl.attribution.randomization_test`, `tl.attribution.road_curve` (default `None`): Optional keyword arguments for `model`; keyword of `tl.attribution.input_x_grad`, `tl.attribution.integrated_gradients`, `tl.attribution.saliency`, `tl.attribution.smoothgrad` (default `None`): Optional keyword arguments for `model`. Floating-point or complex tensor leaves are attributed; keyword of `tl.attribution.insertion_curve` (default `None`): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.noise_tunnel` (default `None`): Optional keyword arguments for the wrapped call; perturbed by the same rule; keyword of `tl.attribution.sensitivity` (default `None`): Optional keyword arguments for the wrapped call; keyword of `tl.backends.resolve` (default `None`): Keyword input mapping supplied by the user; keyword of `tl.debug.bisect_precision`, `tl.debug.check_determinism`, `tl.debug.isolated_capture` (default `None`): Optional forward keyword arguments; keyword of `tl.inventory.list_sites` (keyword-only; default `None`): Optional keyword inputs for the forward rung (kwargs-only models); keyword of `tl.record` (keyword-only; default `None`): Optional keyword arguments for the model call; keyword of `tl.recording.dry_run` (default `None`): Optional keyword arguments for the model call; keyword of `tl.report.flops`, `tl.report.measure_clean_step`, `tl.report.overhead` (default `None`); keyword of `tl.report.flops_vs_dispatch` (default `None`): Forwarded to both executions identically; keyword of `tl.report.kernel_profile` (default `None`): Forwarded to `tl.trace`; keyword of `tl.sweep` (keyword-only; default `None`): Optional keyword inputs passed to `model.forward`; keyword of `tl.trace` (keyword-only; default `None`): Keyword args for `model.forward()`; keyword of `tl.validation.validate_backward_pass` (default `None`): Keyword args for `model.forward()`; keyword of `tl.validation.validate_forward_pass` (default `None`): Keyword arguments for model forward pass |
| `input_op=` | keyword of the receptive-field queries (`Trace.receptive_fields`, `Trace.projective_fields`): the input op to measure against; keyword of `ReceptiveFieldView.at`, `ReceptiveFieldView.center_unit`, `ReceptiveFieldView.check`, `ReceptiveFieldView.probe_grad`, `ReceptiveFieldView.show` (keyword-only; default `None`): Optional exact IO role or model-input operation; keyword of `Trace.projective_fields`, `tl.receptive_field.node_spec` (keyword-only; default `None`); keyword of `Trace.receptive_fields` (keyword-only; default `None`): Optional model-input `Op` handle. String layer names are rejected |
| `input_provenance=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): Provenance-only stamp (tvscope B4, DOCUMENTED-UNSTABLE spelling) for stimuli preprocessed OUTSIDE this run: a `PreprocessingComparison` (its verdict rides the manifest block), a `Resolution`, or a `ResolvedPreprocessing` record (authority-only; verdict stays .. |
| `input_route=` | keyword of `tl.splice_module`: the route name for the spliced input |
| `input_schema=` | constructor keyword of `tl.agent.AgentToolSpec` (required) |
| `input_shape=` | keyword of `Trace.summary` (keyword-only; default `None`); keyword of `tl.trace` (keyword-only; default `None`): Declared input shape(s) (torch backend only): one flat positive-int tuple, a sequence of shape tuples, or a mapping of forward keyword names to shapes (batch included exactly as written). Dtype/value recipes are fail-closed static facts (override: `torchlens.types.InputSpec`); local seed-0 .. |
| `input_spec=` | keyword of `TransformPipeline.plan` (required): Description of the tensor entering the chain; keyword of `TransformSpec.plan` (required): Description of the incoming batch tensor; keyword of `tl.transforms.srp_realized_facts`, `tl.transforms.srp_verify_matrix` (required): Concrete input description (or a tensor to describe) |
| `input_specs=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `None`): Reserved for explicit multi-input specs. Coupled multi-input inference is currently unsupported unless the caller supplies a ready-made tensor/container in future work |
| `input_structure=` | constructor keyword of `tl.types.ModuleInputSnapshot` (required): Immutable compact description of the input container topology |
| `input_transform=` | keyword of `Layer.do` (required): Transform or hook to apply to this layer's output; keyword of `Op.do` (required); keyword of `Trace.run` (keyword-only; optional): Stored-transform sentinel, `False` to bypass, or explicit input transform callable for this run; keyword of `tl.extract_dataset` (keyword-only; default `None`): The unambiguous INPUT-preprocessing path (tvscope B5, DOCUMENTED-UNSTABLE spelling), distinct from `output_transform=` which transforms OUTPUTS before storage. Pass a `Resolution` (the golden path: the authority's own callable and its provenance travel together, the ..; constructor keyword of `tl.options.CaptureOptions` (optional) |
| `inputs=` | keyword of `Extractor.forward` (required): Input passed to the wrapped model; keyword of `Layer.do` (keyword-only; default `None`): Input required when `intervention.engine="rerun"`; keyword of `Op.do`, `tl.experiment.sweep_grid`, `tl.experiment.sweep_sites` (keyword-only; default `None`); keyword of `Trace.do` (keyword-only; default `None`): Input required when `engine="rerun"`; keyword of `Trace.run` (keyword-only; optional): Unified provider input tree. Supplying this keyword returns a transactional `RunResult` and leaves this Trace unchanged. A tuple or list spreads as positional arguments, as `tl.trace` binds `input_args`, on every provider and with `fast=True` (on a live trace a one-parameter forward ..; keyword of `tl.attribution.causal_audit` (required): Bare tensor or tuple/list of positional model arguments (the inputs the claim was computed on); keyword of `tl.attribution.deconvolution`, `tl.attribution.guided_backprop`, `tl.attribution.infidelity`, `tl.attribution.occlusion_map`, `tl.attribution.randomization_test`, `tl.attribution.sensitivity` (required): Bare tensor or tuple/list of positional model arguments; keyword of `tl.attribution.deletion_curve` (required): Bare tensor or tuple/list of positional model arguments; every floating-point tensor is perturbed (integer and boolean tensors pass through unchanged); keyword of `tl.attribution.grad_cam`, `tl.attribution.layer_attribution`, `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (required): Bare tensor for v1 behavior, or tuple/list of positional model arguments. Floating-point or complex tensor leaves are threaded through the model call; keyword of `tl.attribution.gradient_shap` (required): Bare tensor for v1 behavior, or tuple/list of positional model arguments; floating/complex leaves are attributed. The leading axis of every attributed leaf is the EXAMPLE axis; keyword of `tl.attribution.input_x_grad`, `tl.attribution.integrated_gradients`, `tl.attribution.saliency`, `tl.attribution.smoothgrad` (required): Bare tensor for v1 behavior, or tuple/list of positional model arguments. Floating-point or complex tensor leaves are attributed; keyword of `tl.attribution.insertion_curve` (required): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.noise_tunnel` (required): Bare tensor or tuple/list of positional model arguments. Floating and complex tensor leaves are perturbed; every other leaf passes through unchanged; keyword of `tl.attribution.road_curve` (required): `(N, C, H, W)` image tensor, or tuple/list of positional model arguments whose floating-point tensors are images; keyword of `tl.bridge.captum.attribute` (keyword-only; default `None`): Optional explicit Captum input (a tensor or a tuple of tensors). When omitted, every logged forward input is forwarded: the single tensor, or the tuple of positional tensors in `forward` order; keyword of `tl.bridge.depyf.dump` (required): Example input. A tuple is unpacked as positional arguments; any other value is passed as the single argument; keyword of `tl.bridge.inseq.attribute` (required): Source text or batch forwarded as inseq's `input_texts`; keyword of `tl.bridge.pytorch_grad_cam.cam` (keyword-only; default `None`): Optional input tensor. Defaults to the first tensor input saved in `trace`; keyword of `tl.bridge.sae.splice` (default `None`): Model inputs, required when `model_or_log` is a module and refused when it is a trace; keyword of `tl.bridge.shap.explain` (keyword-only; default `None`): Optional inputs to explain. Defaults to the first tensor input saved in `trace`; keyword of `tl.debug.graph_breaks` (required): Model input accepted by `trace`; keyword of `tl.extract`, `tl.pluck` (required): Positional input argument or argument container for `model.forward`; keyword of `tl.fingerprint.assert_unchanged`, `tl.fingerprint.model` (required): Example positional input accepted by `trace`; keyword of `tl.inventory.list_sites` (default `None`): Optional input. Omitted: the free no-input module listing (shapes unknown). Supplied: ONE disclosed-cost metadata-only real forward (no activations retained) reporting shapes, dtypes, pass indices, and functional-vs-module origin; keyword of `tl.mechinterp.edge_scores` (required): The clean inputs (token ids `[batch, pos]`); keyword of `tl.merge_ranks` (required): Live/loaded `Trace` objects and/or rank-core `.tlspec` paths, one per rank, in any order. Every input must have been captured under the distributed opt-in; keyword of `tl.merge_report` (required); keyword of `tl.receptive_field.cross_validate` (keyword-only; default `None`): Optional input selector or selectors; keyword of `tl.report.doctor` (default `None`): Optional input tree inspected only when a model is supplied; keyword of `tl.sweep` (required): Positional input argument or argument container passed to `tl.trace`; keyword of `tl.trackers.watch` (keyword-only; default `None`): Attach-time discovery batch (required for activations); keyword of `Bundle.run` (default `None`): Forward input; keyword of `CaptureBackend.setup_inputs_and_device`, `tl.report.measure_clean_step`, `tl.report.overhead` (required); keyword of `Session.invoke`, `tl.validate` (required): Positional model input; keyword of `tl.backends.resolve` (required): Positional input object supplied by the user; keyword of `tl.debug.bisect_precision`, `tl.debug.isolated_capture` (required): Forward input value or positional-argument list/tuple, exactly as accepted by `tl.trace`; keyword of `tl.debug.check_determinism` (required): Forward input value or positional-argument list/tuple; keyword of `tl.debug.find_nan` (required): Positional model input accepted by `trace`; keyword of `tl.debug.nonfinite_origin` (default `None`): Model input, required for the live form; keyword of `tl.record`, `tl.recording.dry_run` (required): Tensor, list, or tuple of positional model inputs; keyword of `tl.report.flops` (default `None`); keyword of `tl.report.flops_vs_dispatch` (default `None`): Forwarded to both executions identically; keyword of `tl.report.kernel_profile` (default `None`): Forwarded to `tl.trace`; keyword of `tl.trace` (default `None`): Positional args for `model.forward`; a single tensor or list. The input ladder: a real input is the gold rung; omit it and pass `input_shape=`, or omit both to infer -- synthesized values, disclosed, or a teach; mixing refuses `input_rung_conflict`; keyword of `tl.validation.validate_backward_pass` (required): Positional args for `model.forward()`; keyword of `tl.validation.validate_forward_pass` (required): Input for which to validate the saved outs |
| `intermediate_derived_grads=` | constructor keyword of `tl.backends.BackendCapabilities` (default `False`): Whether the backend can derive exact op-level gradients outside true backward capture |
| `interpolation=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Interpolation mode name, lower-cased (`"bilinear"`, `"bicubic"`) |
| `intervene=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): Optional `InterventionSpec` (`tl.when(selector, helper)`; DOCUMENTED-UNSTABLE): every batch is captured with the spec armed, so stored rows are the INTERVENED activations. Batch zero freezes the structural sites the spec fired at; a batch firing nowhere or ..; constructor keyword of `tl.options.RecordingOptions` (optional); keyword of `tl.record` (keyword-only; default `None`): Optional predicate-time intervention slot evaluated on operation contexts; constructor keyword of `tl.recording.Recorder` (keyword-only; optional); keyword of `tl.trace` (keyword-only; default `None`): Optional predicate returning an intervention decision for current-op live mutation |
| `intervening_cluster=` | keyword of `Trace.draw` (keyword-only; default `'upstream'`): Combined graph: placement of backward-only nodes (`"upstream"`, `"outside"`, `"downstream"` or `"own"`) |
| `intervention=` | keyword of `Layer.do` (keyword-only; default `None`): Grouped intervention options (`InterventionOptions`); keyword of `Op.do` (keyword-only; default `None`); keyword of `Trace.do` (keyword-only; default `None`): Grouped intervention options (`InterventionOptions`: `engine` in `"auto"`/`"replay"`/`"rerun"`/`"set_only"`, `confirm_mutation`, `strict`); keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; default `None`): Intervention wording; the default says what a patch grid does ("clean activation of one head patched into the corrupted run", or "... one kv group ..." with `kv_groups`) |
| `intervention_mode=` | keyword of `Trace.draw` (keyword-only; default `'node_mark'`): How intervened sites are marked: `"node_mark"` or `"cone"`; constructor keyword of `tl.options.VisualizationOptions` (optional): Intervention overlay mode |
| `invocations=` | constructor keyword of `tl.experimental.Session` (optional) |
| `io_role=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `ir=` | keyword of `tl.trackers.graph_ir_to_tensorboard` (required) |
| `is_analysis_load_available=` | field of `RunnableDiagnostic` |
| `is_audited=` | field of `ScanRow` |
| `is_available=` | field of `CapabilityRow`, `ValidationStatus` |
| `is_cache_resident_hint=` | field of `RooflineRow` |
| `is_capture_verified=` | field of `ProfileReport` |
| `is_complete=` | field of `InterventionValidationReport` |
| `is_consistent=` | field of `InterventionValidationReport` |
| `is_constant=` | field of `ScanRow` |
| `is_custom=` | keyword of `tl.grad_fn` (keyword-only; default `None`): Optional custom-autograd predicate |
| `is_disabled=` | field of `CloseReport` |
| `is_enabled=` | field of `ScaleLedgerSnapshot` |
| `is_final_sample_forced=` | field of `CloseReport` |
| `is_freeze_pattern_captured=` | field of `BackwardEstimate` |
| `is_grad_disabled_at_capture=` | field of `BackwardEstimate` |
| `is_grad_enabled=` | field of `BindReport` |
| `is_invariant=` | field of `InterventionValidationReport` |
| `is_known=` | field of `OptimizerFacts` |
| `is_local=` | field of `InterventionValidationReport` |
| `is_model_training=` | field of `BindReport` |
| `is_poisoned=` | field of `RunReport` |
| `is_specific=` | field of `InterventionValidationReport` |
| `is_structure_only=` | field of `CaptureFacts`, `ProfileReport` |
| `is_tripped=` | field of `WatchdogSnapshot` |
| `is_verified=` | field of `CaptureFacts` |
| `isolated_output=` | keyword of `CaptureBackend.emit_function_outputs` (required) |
| `jax=` | constructor keyword of `tl.io.PayloadLoadHints` (default `None`): Optional JAX payload hint |
| `jax_control_flow=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `jax_max_control_flow_unroll=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `jax_static_argnums=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `join=` | keyword of `MergedTrace.join_ops` (required) |
| `joint_effect=` | keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; default `None`): Measured joint effect of the same intervention on all the row's heads together, as a delta from the same corrupted baseline; required later by the grid FIGURE; keyword of `tl.viz.ablation_grid` (keyword-only; default `None`): Measured joint effect of the whole displayed set; REQUIRED when more than one row is displayed (a single receipt may carry its own) |
| `k=` | keyword of `Bundle.rank_changes` (keyword-only; default `10`): Maximum rows to RETURN. Every site is scored first; this is a display cap on the sorted result, not a compute bound; keyword of `ComponentStack.topk` (default `5`): Number of rows to return (clamped to the row count); keyword of `ContributionScores.topk` (default `5`): Number of rows to return; keyword of `EffectsView.rank_changes`, `tl.top_changed` (default `None`); keyword of `EffectsView.top`, `FactoredMatrix.unsqueeze` (required); keyword of `FactoredMatrix.corner` (default `3`); keyword of `HeadScores.topk` (default `5`); keyword of `LogitLens.top_tokens` (default `5`): Number of top tokens per layer; keyword of `Trace.decode_output` (default `None`): Optional maximum rank to return per batch item; keyword of `Trace.output_table` (default `5`): Maximum number of class rows per batch item; keyword of `Trace.profile` (keyword-only; default `None`): Number of rows to keep after sorting, or `None` for all rows; keyword of `tl.attribution.causal_audit` (keyword-only; default `5`): Units per ablated set (default 5); keyword of `tl.bridge.lit.model` (keyword-only; default `10`): Next-token candidates per row (causal LM only); keyword of `tl.bridge.sae.write_sae_dashboard` (keyword-only; default `20`): Sequences per latent for dense latents (SAEDashboard's `top_acts_group_size` default, 20); a `UnitTopK` keeps its own `k`; keyword of `tl.mechinterp.inspect_prompt` (keyword-only; default `5`): How many top predictions to include; keyword of `tl.mechinterp.test_prompt` (keyword-only; default `5`); keyword of `tl.report.cost_tree` (keyword-only; default `None`): Keep the `k` largest OWNER/SUBTOTAL rows per parent (by subtree mass); hidden siblings collapse into one deterministic REMAINDER row per parent. Denominators are unaffected; keyword of `tl.report.profile` (keyword-only; default `None`): Number of sorted bottleneck rows to retain. `None` retains all rows. Percent denominators are UNAFFECTED; constructor keyword of `tl.stats.TopK` (default `10`): Number of largest values to retain (per unit with `unit_axis=`); keyword of `tl.top_k` (default `1`); keyword of `tl.viz.feature_maps` (keyword-only; default `4`): Number of top channels to consider for `channels="top"` before the `max_channels` display cap is applied |
| `keep=` | keyword of `Bundle.remove_except` (required): Member name, Trace object, or list of either to retain |
| `keep_latents=` | keyword of `tl.bridge.sae.splice` (keyword-only; default `True`): Keep the latents in the result (default). Past 256 MiB a warning fires once per session; `keep_latents=False` drops them |
| `keep_op=` | keyword of `Trace.preview_fastlog` (default `None`): Predicate callables that receive synthesized fastlog `RecordContext` objects; constructor keyword of `tl.options.RecordingOptions` (optional) |
| `keep_orphans=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `keep_special_tokens=` | keyword of `tl.attribution.text` (keyword-only; default `True`): Whether pad-style baselines keep special tokens at their true embeddings (scaffolding; the only measured converging encoder baseline) |
| `keep_values=` | keyword of `tl.attribution.randomization_test` (keyword-only; default `True`): Keep every randomized map in `steps` (default `True`); past 256 MiB of kept maps the test warns once and names `keep_maps=False` |
| `kernel=` | keyword of `ReceptiveFieldRuleContext.window` (keyword-only; required) |
| `key=` | keyword of `DoctorReport.check` (required): Row key to look up; keyword of `OptimizerFacts.group_fact`, `TagGrammar.run_health` (required); keyword of `ReceptiveFieldRuleContext.cfg` (required): Salient configuration key; constructor keyword of `tl.types.Site` (required); keyword of `tl.site` (default `None`): A rendered `op.site_key` string (exact match) |
| `key_tokens=` | keyword of `tl.transformer.attention_view`, `tl.transformer.attention_views` (keyword-only; default `None`): Source-axis tokens (cross-attention) |
| `keys=` | keyword of `AttentionView.crop_to` (required): Source (column) slice to keep; keyword of `tl.dataset_extraction.export_extraction` (keyword-only; default `None`): Optional output-key subset |
| `kind=` | keyword of `AttributionGraph.node_indices` (required): One of `"feature"`, `"error"`, `"token"` or `"logit"`; keyword of `Bundle.show` (default `None`): `None` (default) or `"graph"`: the merged graph, `draw` with `**kwargs`. `"strip"`: each member's own `Trace.draw` side by side (a given `path` gets `_<member>` appended). `"repr"` / `"html"`: each member's `repr` / HTML text. Any other kind refuses ..; keyword of `CausalAudit.set_of`, `TagGrammar.meta` (required); keyword of `Layer.show`, `Op.show` (default `None`): Picture kind. `None` (the default) or `"auto"` chooses from tensor shape; keyword of `PartialTrace.show` (default `None`): `"graph"` returns DOT, `"repr"` returns `repr(self)`, and `"html"` returns a compact HTML fragment; keyword of `Trace.show` (default `None`): Picture kind. `None` or `"graph"` delegates to `draw`, `"repr"` returns `repr(self)`, and `"html"` returns `_repr_html_()`; constructor keyword of `tl.Selector` (keyword-only; required); constructor keyword of `tl.types.Site` (required); keyword of `tl.func_transform` (default `None`): Optional transform kind to match. Unsanitized and sanitized spellings are both accepted; constructor keyword of `tl.intervention.Population` (required): Closed member-kind vocabulary: `"tensor"` or `"trace"`; constructor keyword of `tl.mechinterp.Coordinate` (default `'writer'`): Row-kind classification (embedding / attention / mlp / writer / constant / unresolved_remainder / residual); keyword of `tl.mechinterp.head_scores` (keyword-only; default `'induction'`): `previous_token` / `duplicate_token` / `induction`, or `FixedPeriodInduction(period=n)` to score the fixed repeat of a sequence with period `n` (destination `i` on source `i - n + 1`, counted when its token equals the token at `i - n`) instead of the token-equality induction mask; keyword of `tl.mechinterp.patch_head` (keyword-only; default `'head'`): `"head"` (z-delta) or `"pattern"` (pattern-delta against the corrupted values). `"q"`/`"k"` refuse: softmax intervenes and no exact lowering exists; constructor keyword of `tl.transformer.Annotation` (required): One of `ANNOTATION_KINDS`; constructor keyword of `tl.types.AttributeWrite` (required): `"added"`, `"replaced"` or `"removed"` |
| `kineto_trace=` | keyword of `tl.bridge.profiler.join` (required): Kineto/Chrome trace JSON path or already-loaded dictionary |
| `kv_groups=` | keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; default `None`): Pass the model's grouping for a grid over a key or value site (`sites="k"` / `"v"`): its head axis counts shared KV groups, not query heads, and the receipt then says which query heads share each group. Under grouped-query attention a per-query-head effect is not measurable at those sites. A .. |
| `kwargs=` | keyword of `BackendProtocol.can_handle`, `BackendProtocol.capture`, `BackendProtocol.reference_output` (required) |
| `label=` | keyword of `CaptureBackend.set_tensor_label` (required); keyword of `MergedTrace.at` (required): A rank-core op/layer label (unqualified); keyword of `tl.grad_fn` (keyword-only; default `None`): Substring to match against the grad_fn label; constructor keyword of `tl.mechinterp.Coordinate` (required): Human-facing component label -- a module address in eval mode (`transformer.h.3.attn.c_proj`), never a raw op name |
| `labels=` | keyword of `tl.bridge.lit.dataset` (keyword-only; default `None`): Classifier vocabulary; adds a `label` CategoryLabel field whose missing values default to the first entry; keyword of `tl.bridge.lit.model` (keyword-only; default `None`): Classifier vocabulary override, validated against `config.id2label` (the textattack SST-2 checkpoint ships useless LABEL_0/1); keyword of `tl.viz.image_scatter` (keyword-only; default `None`): Optional labels for point fallback rendering; keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional series labels for the legend |
| `largest=` | keyword of `tl.top_changed`, `tl.top_fraction`, `tl.top_k` (keyword-only; default `True`) |
| `last=` | keyword of `PartialTrace.narrate`, `Recording.narrate` (default `None`): Keep only the last `last` rows; keyword of `Trace.narrate` (default `None`): Keep only the last `last` rows (the crash-tail idiom) |
| `latent_ids=` | keyword of `tl.bridge.sae.write_sae_dashboard` (keyword-only; default `None`): Latents to write; default every latent that fired (an activation with absolute value above `1e-6`) |
| `latents=` | keyword of `tl.bridge.sae.write_delphi` (required): Dense latents `[examples, sequence, latents]` (or `[examples, latents]`, one position), for example `SpliceResult.latents`: every activation with absolute value above `1e-5` is written, delphi's own threshold. Or the `UnitTopK` of `tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` ..; keyword of `tl.bridge.sae.write_sae_dashboard` (required): Dense latents `[examples, sequence, latents]` (or `[examples, latents]`), for example `SpliceResult.latents`: each sequence carries the latent's activation at every token. Or the `UnitTopK` of `tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` over a dataset: each sequence carries the .. |
| `latents_edit=` | keyword of `tl.bridge.sae.splice` (keyword-only; default `None`): Optional transform applied to the encoded latents before decoding -- the causal-testing knob (ablate or steer individual features) |
| `layer=` | keyword of `Direction.library_layer` (default `None`): Explicit layer index, or `None` to read it from the site label; keyword of `Direction.to_dialz` (keyword-only; default `None`): Decoder layer `SteeringModel` applies it to. Defaults to the trailing index of the site label; keyword of `Direction.to_repeng` (keyword-only; default `None`): Decoder layer `ControlModel` applies it to. Defaults to the trailing index of the site label; keyword of `Direction.to_steering_vectors` (keyword-only; default `None`): Decoder layer the library patches. Defaults to the trailing index of the site label (`"model.layers.3"` is layer 3); keyword of `Direction.upload_to_neuronpedia` (keyword-only; default `None`): Decoder layer (0-based). Defaults to the trailing index of the site label, as `to_repeng` reads it; keyword of `tl.attribution.grad_cam` (keyword-only; default `None`): Name from `dict(model.named_modules())` identifying the target layer. The layer must fire exactly once during the forward pass; Grad-CAM has no defined semantics for a reused layer, so multi-fire raises rather than silently attributing one arbitrary call. `None` chooses the layer from a ..; keyword of `tl.attribution.layer_attribution`, `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (keyword-only; required): Name from `dict(model.named_modules())` identifying the target layer; keyword of `tl.bridge.dialz.from_direction` (keyword-only; default `None`): Decoder layer `SteeringModel` applies it to; defaults to the trailing index of the direction's site label (see `Direction.library_layer`); keyword of `tl.bridge.repeng.from_direction` (keyword-only; default `None`): Decoder layer `ControlModel` applies it to; defaults to the trailing index of the direction's site label (see `Direction.library_layer`); keyword of `tl.bridge.steering_vectors.from_direction` (keyword-only; default `None`): Decoder layer the library patches; defaults to the trailing index of the direction's site label (see `Direction.library_layer`); keyword of `tl.experimental.freeze_module` (required): Module whose parameters should be frozen; keyword of `tl.mechinterp.patch_head` (keyword-only; required): Attention module address (or index into executed attention order); keyword of `tl.pluck` (required): Layer label, module path, pass-qualified label, or unique substring. Resolves strictly to one result; ambiguous lookups refuse typed; keyword of `tl.trackers.node_name_for` (required); keyword of `tl.transformer.attention_view` (required): Module address of the attention module (see `attention_views` for discovery); keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; required): The layer coordinate to extract (module address as recorded in the grid's `layer` coordinates); keyword of `tl.transformer.score_decomposition` (required): Attention module address |
| `layer_grad_atol=` | keyword of `tl.validate` (keyword-only; default `None`): Backward-only layer-gradient absolute tolerance; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `None`): Optional layer-gradient absolute tolerance |
| `layer_grad_rtol=` | keyword of `tl.validate` (keyword-only; default `None`): Backward-only layer-gradient relative tolerance; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `None`): Optional layer-gradient relative tolerance |
| `layer_map=` | keyword of `tl.bridge.brain_score.activations_extractor` (keyword-only; default `None`): Optional mapping from Brain-Score layer names to TorchLens lookups. The special name `"logits"` defaults to the model output; map it here to override; keyword of `tl.bridge.brain_score.get_activations_fn` (keyword-only; default `None`): Optional mapping from Brain-Score layer names to TorchLens lookups. Unmapped names are used verbatim (module dotted paths work as-is, and any TorchLens lookup — including functional-op labels that module hooks cannot address — is accepted) |
| `layer_type=` | keyword of `Direction.to_steering_vectors`, `tl.bridge.steering_vectors.from_direction` (keyword-only; default `'decoder_block'`): steering-vectors layer type (default `"decoder_block"`) |
| `layer_visualizers=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `layers=` | keyword of `tl.attribution.randomization_test` (keyword-only; default `None`): Module names to randomize, top first; default: every module with parameters of its own, in reverse order of first execution; constructor keyword of `tl.compat.Extractor` (required): Layer names accepted by `torchlens.extract`; keyword of `tl.compat.from_fx` (default `None`): Optional explicit layer lookups. Defaults to FX node names; keyword of `tl.compat.from_torchextractor` (default `None`): Optional layers to extract. When omitted, `model.layers` or `model.layer_names` is used; keyword of `tl.extract` (required): Either a list of layer lookups or a mapping of `user_label -> layer_lookup`; keyword of `tl.extract_dataset` (required): List or mapping of layer lookups (`extract` semantics), a capture SELECTOR (`tl.func`/`tl.in_module`/ composed predicates), or a selector-valued mapping. Selector requests resolve against batch zero in execution order, FREEZE (ordered site keys + pass-qualified labels), and ..; keyword of `tl.mechinterp.measure_trajectory` (keyword-only; default `None`): Block addresses to project, as in `logit_lens`; default every block; keyword of `tl.mechinterp.mlp_neuron_contributions` (default `None`): MLP-module selector: `None` (all), executed-order index or address, or a list of them; keyword of `tl.transformer.attention_views` (keyword-only; default `None`): Explicit module addresses; defaults to every pattern-bearing module |
| `layout=` | keyword of `Trace.draw` (keyword-only; default `'auto'`): Node placement: `"auto"`, `"dot"` or `"rank"`; keyword of `tl.neuro.export_npz` (keyword-only; default `'consolidated'`): `"consolidated"` (default): ONE npz with one array per site plus a JSON sidecar manifest so numpy-only consumers get provenance. `"per_stimulus"`: one npz PER STIMULUS containing one array per site -- Net2Brain's loader layout; constructor keyword of `tl.options.VisualizationOptions` (optional): Layout engine selector: `"auto"`, `"dot"`, or `"rank"`; keyword of `tl.viz.bundle_diff` (keyword-only; default `'paired'`): Layout strategy. Only `"paired"` is currently supported |
| `lazy=` | keyword of `tl.load` (keyword-only; default `False`): Whether direct out/grad blobs should remain lazy placeholders. For a saved `Bundle` this, `map_location` and `materialize_nested` reach every member |
| `lazy_transitions=` | constructor keyword of `tl.types.InputProvenance` (default `()`): Module addresses whose lazy parameters materialized during the ONE captured forward (disclosure of the sanctioned permanent mutation) |
| `leaf=` | keyword of `TagGrammar.data` (required) |
| `left=` | keyword of `MergedTrace.happens_before` (required); keyword of `tl.viz.bundle_diff` (keyword-only; default `None`): Left member name or log. Defaults to the first bundle member |
| `legend=` | constructor keyword of `tl.transformer.Annotation` (default `''`): Kind legend line |
| `lens=` | keyword of `tl.mechinterp.measure_trajectory` (keyword-only; default `None`): Optional user lens (a tuned lens), as in `logit_lens` |
| `level=` | keyword of `Bundle.save` (keyword-only; default `'portable'`): Save level: `"audit"`, `"executable_with_callables"`, or `"portable"`; keyword of `Trace.profile` (default `'op'`): Profile granularity: operation, module address, or module call; keyword of `Trace.projective_fields` (default `'op'`); keyword of `Trace.receptive_fields` (default `'op'`): Table granularity: operation, layer, module invocation, or module; keyword of `Trace.save` (keyword-only; default `'portable'`): `"audit"`, `"executable_with_callables"`, `"portable"` or `"runnable"`; keyword of `Trace.save_intervention` (keyword-only; default `'executable_with_callables'`): Save level: `"audit"`, `"executable_with_callables"`, or `"portable"`; keyword of `Trace.summary` (default `None`): Rebuilt row grain: `"auto"` (default) \| `"module"` \| `"op"`. With `view="stats"`, `"op"` (and `"auto"`) is one row per executed op and `"module"` one row per module-output payload; keyword of `tl.io.save_intervention` (keyword-only; default `...`): Save level: `"audit"`, `"executable_with_callables"`, or `"portable"`; keyword of `tl.report.profile` (keyword-only; default `'op'`): `"op"` for one row per operation, `"module"` for one row per module address, or `"call"` for one row per module invocation |
| `like=` | keyword of `tl.random_like` (keyword-only; required) |
| `limits=` | keyword of `tl.agent.build_envelope` (keyword-only; default `None`): Applied limits echo |
| `lines=` | constructor keyword of `tl.stats.NarrationObserver` (optional) |
| `link_format=` | keyword of `Trace.first_nonfinite` (default `'text'`): Source-location link style. `"text"` (the default) emits plain `path:line` text; `"terminal"` emits OSC 8 hyperlinks and `"html"` emits VS Code URI anchors. The styled registers are OPT-IN because both embed the capture machine's RESOLVED ABSOLUTE path inside the link target: returned .. |
| `lo_exp=` | constructor keyword of `tl.stats.HistogramDescriptor` (default `-48`) |
| `local_files_only=` | keyword of `tl.compat.from_huggingface` (keyword-only; default `True`): Whether to restrict loading to the local cache |
| `log_injections=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `log_latents=` | keyword of `tl.bridge.sae.splice` (keyword-only; default `False`): Off by default. On, `spliced` is a fresh capture of the live model (live-model door only) with the SAE run inside it, so its encoder ops, the latents node (`result.latents_site`) and its decoder ops are trace nodes; the spliced outputs equal those with it off |
| `log_or_bundle_or_spec=` | keyword of `tl.bridge.huggingface_hub.push_to_hub` (required): `Trace` or `Bundle` to publish |
| `log_scale=` | keyword of `tl.mechinterp.circuit_model_difference`, `tl.mechinterp.circuit_performance_ratio` (keyword-only; default `False`): Keyword-only. Integrate over `log(fraction)` (MIB's `log_scale`) |
| `logdir=` | keyword of `tl.trackers.graph_ir_to_tensorboard` (required) |
| `logging_steps=` | keyword of `tl.trackers.derived_cadence` (keyword-only; default `None`) |
| `logit_probability_coverage=` | keyword of `tl.attribution.build_graph` (keyword-only; default `0.95`): Keep the most probable logits until their probabilities sum to this |
| `logits=` | keyword of `tl.transformer.token_metrics` (required): `[n_positions, vocab]` captured logits for one sequence |
| `logits_at=` | keyword of `tl.attribution.build_graph` (keyword-only; default `None`): `SiteTarget` naming the module whose output holds the logits; `None` reads the model output |
| `logits_lookup=` | keyword of `tl.bridge.brain_score.get_activations_fn` (keyword-only; default `'output_1'`): TorchLens lookup used for Brain-Score's special `"logits"` name |
| `lookback=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `lookback_payload_policy=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `loss=` | keyword of `Trace.do` and `tl.sweep`: a callable of the returned trace whose scalar result the door runs `backward` on; keyword of `Recorder.backward` (required); keyword of `Recording.backward` (required): Loss tensor whose autograd graph should be walked; keyword of `Trace.backward` (required): Tensor whose `grad_fn_handle` roots the backward graph |
| `loss_fn=` | keyword of `tl.stats.aggregate` (keyword-only; default `None`): Callable building a loss from `(output, *batch_tail)`; required when any key is a gradient site (`aggregate_loss_fn_required`); keyword of `tl.validate` (keyword-only; default `None`): Backward-only loss function; keyword of `tl.validation.validate_backward_pass` (default `None`): Optional callable mapping model outputs to a scalar loss. Defaults to summing all returned tensors |
| `low=` | keyword of `tl.clamp` (keyword-only; default `None`): Optional lower bound; keyword of `tl.saturated` (keyword-only; default `None`) |
| `main_input_numel=` | keyword of `tl.report.flops` (keyword-only; default `None`): Explicit 6ND `D` for the trace door (the comparator's HF applicability cannot be established from a trace alone; the line is OMITTED without it, never guessed) |
| `map_location=` | keyword of `Op.materialize_grad`, `Op.materialize_out` (keyword-only; default `'cpu'`): Target device for the materialized tensor; keyword of `tl.io.materialize_nested` (keyword-only; default `'cpu'`): Target device for the materialized nested tensors; keyword of `tl.load` (keyword-only; default `'cpu'`): Target device for eager tensor materialization |
| `mapping=` | keyword of `Bundle.vary` (required): `{member: InterventionSpec \| (selection, edit) \| None}` |
| `mask=` | keyword of `tl.param` (default `None`); keyword of `tl.positions` (keyword-only; default `None`): `(batch, tokens)` attention mask for `"first"` / `"last"`; keyword of `tl.transformer.attention_view` (keyword-only; default `None`): Explicit user/model mask metadata (source 3 of the hierarchy); bool `[n_destination, n_source]`, True = masked out. Used only when no captured source resolves |
| `mask_input=` | keyword of `tl.mechinterp.measure_trajectory` (keyword-only; default `False`): `True` bans each row's own input token ids (the model input must be an integer tensor) before every reduction, final output included |
| `masked=` | keyword of `TapObserver.values` (default `False`): `False` (default): the FULL snapshots, exactly the shipped behavior. `True`: each record's stored selection mask is applied and fresh masked copies (the selected elements, flat) are returned; records without a stored mask return the full snapshot clone. Callers may not override the stored mask .. |
| `matching=` | keyword of `Population.mean` (keyword-only; default `None`): Subject datum selecting the eligible agreement class; keyword of `Population.std` (keyword-only; default `None`); constructor keyword of `tl.intervention.SamplingPlan` (required); keyword of `tl.intervention.resample_rows_from` (keyword-only; default `None`): Subject datums: `PerRowDatums` (one per batch row) or `OneDatum`; keyword of `tl.intervention.sample_from` (keyword-only; default `None`): The subject's datum: a bare value or `OneDatum` for whole-event donors (`PerRowDatums` belongs to `resample_rows_from`) |
| `matching_value=` | keyword of `Population.eligible_indices` (required): The subject's datum for this event or row (already unwrapped) |
| `materialize_nested=` | keyword of `tl.load` (keyword-only; default `True`): Whether nested blob refs in captured args and RNG states should be materialized when `lazy=True` |
| `maturity=` | constructor keyword of `tl.backends.BackendSpec` (default `'preview'`): Launch label, `"preview"` or `"stable"`; only a recorded maintainer decision changes it |
| `max_activation=` | keyword of `tl.debug.dead_neurons` (keyword-only; default `0.0`): Largest output magnitude a dead unit may reach at any position: absolute, on the layer's output, `0.0` means exactly zero |
| `max_array_bytes=` | constructor keyword of `tl.options.HtmlReportOptions` (default `524288`) |
| `max_array_sites=` | constructor keyword of `tl.options.HtmlReportOptions` (default `8`) |
| `max_axis_items=` | keyword of `tl.viz.heatmap` (keyword-only; default `8`): Maximum number of axis thumbnails or labels to draw before adding a cap marker. The rendered `+N more` count is computed AFTER the final overlap selection, so it always states exactly how many axis items are not shown (tviz FIX-H) |
| `max_bytes=` | keyword of `tl.transformer.circuitsvis_attention` (keyword-only; default `None`): Payload ceiling override; defaults to the tested threshold |
| `max_cells=` | keyword of `tl.experiment.sweep_grid`: guard on the product of the axis lengths (`grid_too_large`) |
| `max_channels=` | keyword of `tl.viz.feature_maps` (keyword-only; default `4`): Maximum number of explicit or top channels to store; keyword of `tl.viz.nodes.feature_map` (keyword-only; default `4`): Maximum number of channel columns to display |
| `max_components=` | keyword of `tl.viz.nodes.scree` (keyword-only; default `24`): Maximum number of leading components to plot |
| `max_depth=` | keyword of `Module.show_call_tree` (default `None`): Maximum descendant depth to print for each call, or `None` for no limit; keyword of `ModuleCall.show_call_tree`, `Trace.show_call_tree` (default `None`): Maximum descendant depth to print, or `None` for no limit; keyword of `tl.debug.lineage` (keyword-only; default `None`): Optional maximum graph distance from the start op; keyword of `tl.report.cost_tree` (keyword-only; default `None`): Fold rows deeper than this depth into their ancestor's REMAINDER row; conservation stays exact |
| `max_fanout=` | keyword of `Trace.annotate` (keyword-only; default `1000000`): Explicit maximum number of selected sites. The default is intentionally large so annotation can fan out across many layers; keyword of `Trace.find_sites` (keyword-only; default `8`): Maximum number of matching sites |
| `max_fraction=` | keyword of `tl.debug.dtype_range` (keyword-only; default `0.9`): Fraction of the dtype maximum at or above which a finite magnitude is flagged. Defaults to `0.9`; keyword of `tl.debug.param_health` (keyword-only; default `0.9`): Fraction of the dtype ceiling above which headroom warns (knob parity with `tl.debug.dtype_range`) |
| `max_items=` | keyword of `tl.viz.image_scatter` (keyword-only; default `16`): Maximum number of coordinates to draw before adding a cap marker; keyword of `tl.viz.montage` (required): Maximum number of images to include; keyword of `tl.viz.nodes.text_table` (required): Maximum number of strings to include before a `+N more` row |
| `max_length=` | keyword of `tl.attribution.text` (keyword-only; default `None`): Optional EXPLICIT truncation; records the original length and the retained span, and warns visibly. Never applied silently: without it, an input longer than the model config's position limit refuses `text_input_exceeds_position_limit` |
| `max_lines=` | constructor keyword of `tl.options.NarrationOptions` (keyword-only; optional): Opt-in live-volume bound. Reaching it emits ONE suppression marker with the exact count, keeps counting (the footer discloses the total), and never silently samples |
| `max_minibatch_size=` | keyword of `tl.bridge.lit.model` (keyword-only; default `8`): LIT minibatch bound |
| `max_new_tokens=` | keyword of `tl.bridge.lit.model` (keyword-only; default `16`): Fixed deterministic-generation length (causal LM only) |
| `max_nodes=` | keyword of `tl.debug.sketch_grad_fn` (keyword-only; default `20000`): Defensive walk ceiling (disclosed on truncation); keyword of `tl.debug.walk_grad_fn` (keyword-only; default `20000`): Defensive walk ceiling; hitting it sets `truncated` and is disclosed on every render |
| `max_norm=` | keyword of `tl.clip_norm` (required): Largest allowed `norm_type` norm |
| `max_num_logits=` | keyword of `tl.attribution.build_graph` (keyword-only; default `10`): Cap on the salient logits |
| `max_ops=` | keyword of `Trace.to_agent_json` (keyword-only; default `None`): Optional cap on emitted op rows (execution order, first `max_ops` kept). Any omission is disclosed in the `truncation` block |
| `max_pairs=` | keyword of `tl.viz.bundle_diff` (keyword-only; default `16`): Maximum aligned pairs to draw. `None` draws every pair |
| `max_passes=` | keyword of `tl.attribution.occlusion_map` (keyword-only; default `4096`): Deterministic pass budget (default 4096). Exceeding it refuses typed AFTER measuring one pass, with the full arithmetic in the message and fields |
| `max_pending_bytes=` | constructor keyword of `tl.options.StorageOptions` (keyword-only; optional): Pending snapshot byte budget for the async pipeline (`None` uses the 256 MiB default). Once pending writes hold this many bytes, capture blocks until the disk catches up, so a slow disk slows capture instead of accumulating unbounded RAM; keyword of `tl.to_disk` (keyword-only; default `None`): Pending snapshot byte budget for the async pipeline (`None` uses the 256 MiB default); once pending writes hold this many bytes, capture blocks until the disk catches up, bounding the RAM the deferred writes may occupy |
| `max_predicate_failures=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional) |
| `max_probes=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `64`): Maximum forward probes |
| `max_rows=` | keyword of `Trace.summary` (keyword-only; default `None`): Body-row budget for the auto ladder (default 48) |
| `max_scan_elements` | option of `Trace.summary(view="stats")`, accepted through its `**options` rather than as a named parameter: the payload-scan element budget; later retained rows become typed `unscanned` rows; refused typed with any other view |
| `max_scan_elements=` | keyword of `Trace.summary` (keyword-only; default `None`): The payload-scan element budget of `view="stats"`; later retained rows become typed `unscanned` rows. Refused typed (`summary_option_invalid`) with any other view |
| `max_size=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `512`): Maximum spatial side considered; keyword of `tl.viz.montage` (default `600`): Maximum montage width or height; keyword of `tl.viz.nodes.channel_grid` (default `300`): Maximum width or height of the rendered grid; keyword of `tl.viz.nodes.heatmap` (default `200`): Maximum width or height of the rendered image |
| `max_steps=` | keyword of `tl.trackers.derived_cadence` (keyword-only; default `None`) |
| `max_stimuli=` | keyword of `tl.viz.feature_maps` (keyword-only; default `4`): Maximum number of stimuli to store; keyword of `tl.viz.nodes.feature_map` (keyword-only; default `4`): Maximum number of stimulus rows to display; keyword of `tl.viz.nodes.rdm` (keyword-only; default `8`): Maximum number of stimulus rows/columns to label before adding a cap marker |
| `max_thumbnails=` | keyword of `tl.viz.nodes.mds_scatter` (keyword-only; default `16`): Maximum number of stimuli to draw before adding a `+K more` indicator |
| `max_tokens=` | keyword of `PartialTrace.explain`, `Trace.explain` (keyword-only; default `None`): Optional token budget. Whole sections drop low-value-first until the report fits, every drop is disclosed, and the capture-status section never drops |
| `max_width=` | keyword of `tl.stats.glance` (keyword-only; default `None`): Optional width bound; degradation follows the retention order (bytes, then `n=`, then the sparkline -- never shape/dtype/extrema/health) |
| `may_alias=` | constructor keyword of `tl.transforms.PlannedStep` (required): Whether the output may share storage with the input — a first-class planning fact; no peak-memory claim follows from shape reduction alone |
| `mean=` | keyword of `tl.intervention.subspace_ablate` (keyword-only; default `None`): Mean activation (broadcastable to the site value) for `mode="mean"`; constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Per-channel normalization means |
| `measure=` | keyword of `RandomizationTest.similarity` (default `'spearman'`): `"spearman"` (default), `"spearman_abs"` or `"ssim"`; keyword of `tl.mechinterp.head_scores` (keyword-only; default `'mul'`): `mul` (attention mass on mask-eligible sources, TLens-compatible) or `abs` (1 - L1/2 distance between the pattern row and the mask row distribution) |
| `measure_host_peak_memory=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `measure_python_peak_memory=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `member=` | keyword of `Bundle.why` (required): Member name (or Trace) to explain |
| `member_name=` | field of `GridCell` |
| `members=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `Super.from_members` (required): Mapping from bundle member name to resolved object; constructor keyword of `tl.Bundle` (default `()`): Mapping of member names to `Trace` objects, sequence of logs, or sequence of `(name, log)` tuples. Empty (the default) builds an empty bundle to fill later with `add`; keyword of `tl.export.model_explorer_diff` (required): Bundle or sequence of logs to compare; constructor keyword of `tl.intervention.Population` (required): Prepared member payloads: tensors (`kind="tensor"`) or captured traces (`kind="trace"`; a member's donor value resolves at the firing site, pass-qualified, at fire time -- an index/read, never a model run) |
| `memory_budget=` | keyword of `tl.neuro.plan_extraction` (keyword-only; default `None`): Optional budget accepted by `parse_bytes` |
| `mesh_axis_names=` | constructor keyword of `tl.io.JaxPayloadLoadHint` (default `None`): Optional expected mesh axis names for metadata reconstruction. When provided, the saved contract must match exactly |
| `message=` | keyword of `ValidationStatus.unverified` (keyword-only; required): Human-readable explanation |
| `method=` | keyword of `ReceptiveFieldView.of`: `"geometric"` (union of per-unit boxes, the default) or `"gradient"` (autograd probes on a backward-ready trace); keyword of `tl.attribution.layer_attribution` (keyword-only; default `'activation_x_grad'`): Layer attribution method. `"activation_x_grad"` returns `activation * gradient`. `"grad"` returns `abs(gradient)`. A layer reused `N` times contributes through every firing, so both methods sum the per-firing terms; keyword of `tl.attribution.noise_tunnel`, `tl.attribution.sensitivity` (keyword-only; default `None`): Kit-contract callable; requires `model` and `target`; keyword of `tl.attribution.randomization_test` (keyword-only; required): Explanation method called as `method(model_copy, inputs, input_kwargs, target=target, **method_kwargs)`, returning an `AttributionResult` or a tensor (`grad_cam` maps are compared at input resolution); keyword of `tl.attribution.read` (keyword-only; default `'activation_x_grad'`): `'activation_x_grad'` (default, signed), `'grad'`, `'activation'`; keyword of `tl.bridge.captum.attribute` (required): Captum attribution object exposing `attribute`; keyword of `tl.bridge.inseq.attribute` (keyword-only; default `'integrated_gradients'`): Attribution method: an inseq identifier (`inseq.list_feature_attribution_methods()`) or the TorchLens spelling of the same method (`"input_x_grad"` maps to inseq's `"input_x_gradient"`). Used only when this call loads the model (`attribution_model` is None); constructor keyword of `tl.intervention.Direction` (required): `"mean_diff"`, `"pca"` or `"logistic"`; keyword of `tl.intervention.fit_direction` (default `'mean_diff'`): `"mean_diff"` (default), `"pca"` or `"logistic"`; see the module docstring for the exact math; keyword of `tl.mechinterp.edge_scores` (keyword-only; default `'eap_ig'`): Keyword-only. `"eap_ig"` (default; the gradient averaged over `steps` points between the corrupted and the clean input embedding, left Riemann sum) or `"eap"` (one gradient, at the clean input); keyword of `tl.repgeom.compare_rdms` (default `'spearman'`): `"spearman"` (default), `"pearson"`, or `"kendall"` (tau-a); keyword of `tl.repgeom.compare_rdms_interval` (default `'spearman'`): `"spearman"` (default), `"pearson"`, or `"kendall"` (tau-a; quadratic in the number of pairs per resample). For kendall, the copies of one stimulus pair that a resample repeats are never compared with each other (they are one observation); every other comparison is weighted by the copies' ..; keyword of `tl.subspace` (keyword-only; default `None`) |
| `method_kwargs=` | keyword of `tl.attribution.noise_tunnel` (keyword-only; default `None`): Explicit child settings mapping, echoed verbatim into `extra["method_kwargs"]`; keyword of `tl.attribution.randomization_test` (keyword-only; default `None`): Extra keyword arguments for `method` (for example `layer=` for `grad_cam`); keyword of `tl.attribution.sensitivity` (keyword-only; default `None`): Explicit child settings mapping. A known-stochastic method WITHOUT a seed or stored bank in here refuses BEFORE any work (ladder rung 1) |
| `metric=` | keyword of `Bundle.compare` (keyword-only; default `'cosine'`): Metric name or callable over `(left, right)` tensors; scalar-like sites always use `relative_l1`; keyword of `Bundle.draw` (keyword-only; default `'relative_l2'`); keyword of `Bundle.rank_changes` (keyword-only; default `'cosine'`): Pairwise tensor metric; keyword of `Bundle.store_comparison` (default `'relative_l2'`); keyword of `SuperOp.compare` (default `'cosine'`): Metric name or callable; constructor keyword of `tl.attribution.AttributionMetric` (required): Metric name (`"infidelity"` or `"sensitivity_max"`); keyword of `tl.attribution.causal_audit` (keyword-only; default `None`): For an EDGE table only: the metric it was scored with (`edge_scores`' `metric=`, `logits -> values`, one value per example or one scalar); keyword of `tl.experiment.sweep_grid` (keyword-only; required); keyword of `tl.experiment.sweep_sites` (keyword-only; required): REQUIRED. `callable(member_trace) -> finite real scalar` — the metric IS the experiment; readouts resolve to SITE TENSORS on the member (`reconstruct_output` is not in this path); keyword of `tl.mechinterp.edge_scores` (keyword-only; required): Keyword-only. `logits -> values`: the metric `M` from the unembedding output, one value per example (`[batch]`) or one scalar; keyword of `tl.mechinterp.patch_grid`, `tl.mechinterp.patch_query_heads_grid` (required): Callable `Trace -> scalar tensor`; keyword of `tl.mechinterp.patch_head` (keyword-only; default `None`): Optional `Trace -> scalar`; evaluated on the patched rerun; keyword of `tl.mechinterp.patch_heads_grid`, `tl.mechinterp.patch_residual_grid` (required); keyword of `tl.repgeom.activation_distance_matrix` (default `'euclidean'`): Dissimilarity metric. Supported values are `"euclidean"`, `"cosine"`, `"correlation"`, and `"gaussian"`; keyword of `tl.repgeom.effective_dimensionality`, `tl.repgeom.scree` (keyword-only; default `'euclidean'`): Distance metric used when activations are not already distances; keyword of `tl.repgeom.mds_evolution`, `tl.repgeom.rdm_evolution`, `tl.repgeom.scree_evolution` (keyword-only; default `'euclidean'`): Activation distance metric passed to `activation_distance_matrix`; keyword of `tl.repgeom.rdm` (default `'euclidean'`): `"euclidean"`, `"manhattan"`, `"cosine"`, `"correlation"`, or `"gaussian"` (dispatch is table-driven; callable metrics are a later table row). `"gaussian"` computes a data-derived bandwidth over the full matrix and runs on the historical numpy path only. The default deliberately differs ..; keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; default `'metric'`): Metric wording printed on the figure; keyword of `tl.viz.bundle_diff` (keyword-only; default `'relative_l2'`): Per-site tensor metric (the baseline comparison `Bundle.compare` runs) |
| `metrics=` | keyword of `tl.stats.aggregate` (required): Mapping from site to streaming statistic. A key names the tensor it reads: a plain string (a layer label, a label substring, or `"output"`) or a selector is an activation; `tl.grad(site)` is the gradient of that site's output, so `{"relu_3": Mean(), tl.grad("relu_3"): Mean()}` measures both; keyword of `tl.transformer.metric_strip` (required); keyword of `tl.viz.metric_strip` (required): The metric record from `token_metrics` |
| `micro_batches=` | keyword of `ChecksSession.step`, `WatchSession.step` (keyword-only; default `None`) |
| `min_distance=` | keyword of `tl.viz.image_scatter` (keyword-only; default `None`): Optional minimum center-to-center distance in pixels. Pass `0` to disable overlap-avoidance and draw every item at its exact projected coordinate; keyword of `tl.viz.nodes.mds_scatter` (keyword-only; default `None`): Minimum center-to-center distance in pixels. Defaults to the thumbnail size for thumbnail scatters and a smaller point spacing for fallbacks |
| `min_size=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `1`): Minimum spatial side considered |
| `min_stimuli=` | keyword of the `tl.repgeom` geometry functions (`classical_mds`, `effective_dimensionality`, `scree`): fewest stimuli accepted; keyword of `tl.repgeom.classical_mds` (keyword-only; default `8`): Minimum number of stimuli required for visualization-oriented MDS; keyword of `tl.repgeom.effective_dimensionality`, `tl.repgeom.scree` (keyword-only; default `3`): Minimum number of stimuli required to compute the scree spectrum; keyword of `tl.repgeom.mds_evolution` (keyword-only; default `8`): Minimum number of stimuli required by `classical_mds`; keyword of `tl.repgeom.rdm_evolution` (keyword-only; default `2`): Minimum number of stimuli required for each RDM. Must be at least 2; keyword of `tl.repgeom.scree_evolution` (keyword-only; default `3`): Minimum number of stimuli required for each scree spectrum |
| `minimum=` | constructor keyword of `tl.backends.FrameworkRequirement` (required): Inclusive version floor |
| `modality=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `'image'`): Input modality tag; adapters are modality-tagged from day one so non-image authorities can land later without a redesign |
| `mode=` | keyword of `FacetSpec.scatter_update` (keyword-only; default `'replace'`): Write mode. `"replace"` writes `edited_slice` directly; `"add"` adds it to the selected slice; keyword of `tl.intervention.subspace_ablate` (keyword-only; default `'zero'`): `"zero"` removes the component; `"mean"` replaces it with the component of `mean`; `"resample"` replaces it with another batch row's component under a seeded permutation; keyword of `tl.transforms.srp` (default `'flatten'`): `"flatten"` (bind ONE flattened extent, refuse drift) or `"features"` (project the last axis) |
| `model=` | keyword of `BackendProtocol.can_handle`, `BackendProtocol.capture`, `BackendProtocol.reference_output`, `CaptureBackend.prepare_model`, `CaptureBackend.prepare_model_once`, `CaptureBackend.prepare_model_session`, `CaptureBackend.setup_inputs_and_device`, `tl.mechinterp.patch_heads_grid`, `tl.mechinterp.patch_residual_grid`, `tl.mechinterp.test_prompt`, `tl.report.measure_clean_step`, `tl.report.overhead` (required); keyword of `Bundle.run` (required): Model forwarded to each member; keyword of `Layer.do` (keyword-only; default `None`): Model required when `intervention.engine="rerun"`; keyword of `Op.do`, `tl.experiment.sweep_grid` (keyword-only; default `None`); keyword of `Recipe.bind` (required): The base `nn.Module` (validated through the one audited model-door funnel; bindings do not nest); keyword of `Trace.do` (keyword-only; default `None`): Model required when `engine="rerun"`; keyword of `Trace.run` (default `None`): Model to execute through TorchLens decorated wrappers. When omitted, the live model captured by this `Trace` is reused if still available -- the trace holds it weakly, so this requires the caller to have kept a strong reference (see Raises); keyword of `tl.attribution.causal_audit` (required): The PyTorch module the claim explains; never modified; keyword of `tl.attribution.deconvolution`, `tl.attribution.grad_cam`, `tl.attribution.gradient_shap`, `tl.attribution.guided_backprop`, `tl.attribution.input_x_grad`, `tl.attribution.integrated_gradients`, `tl.attribution.layer_attribution`, `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients`, `tl.attribution.occlusion_map`, `tl.attribution.saliency`, `tl.attribution.smoothgrad` (required): PyTorch module to attribute; keyword of `tl.attribution.deletion_curve`, `tl.attribution.infidelity`, `tl.attribution.road_curve` (required): PyTorch module the attribution explains; keyword of `tl.attribution.insertion_curve` (required): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.noise_tunnel`, `tl.attribution.sensitivity` (keyword-only; default `None`): Model for the `method=` sugar route; keyword of `tl.attribution.randomization_test` (required): PyTorch module the method explains; never modified; keyword of `tl.attribution.text` (required): HF-style model whose forward accepts `inputs_embeds` and returns logits; keyword of `tl.backends.resolve` (required): Candidate model or callable; keyword of `tl.bridge.brain_score.activations_extractor` (required): PyTorch model to capture; keyword of `tl.bridge.brain_score.get_activations_fn` (required): PyTorch model to capture. Run in `eval()` mode under `no_grad()` for each batch, matching Brain-Score's own `PytorchWrapper`; keyword of `tl.bridge.depyf.dump` (required): Eager `nn.Module` (or callable) to compile and run once; keyword of `tl.bridge.transformers.record_attention` (required): A Hugging Face model switched to the backend (`register_attention` with the model, or `attn_implementation="torchlens"`); keyword of `tl.bridge.transformers.register_attention` (default `None`): Optional Hugging Face model; when given, it is switched to the backend with `model.set_attn_implementation("torchlens")`; keyword of `tl.bridge.transformers.trace_image` (required): PyTorch model to trace; keyword of `tl.bridge.transformers.trace_multimodal` (required): Hugging Face multimodal model with resolvable `AutoProcessor`; keyword of `tl.bridge.transformers.trace_text` (required): Hugging Face model, or any PyTorch model with a resolvable tokenizer; constructor keyword of `tl.compat.Extractor` (required): PyTorch model to execute; keyword of `tl.compat.from_ilg` (required): Source model or `IntermediateLayerGetter`-like object; keyword of `tl.compat.from_sentence_transformers` (required): Sentence Transformers model or model identifier; keyword of `tl.compat.from_torchextractor` (required): PyTorch model or torchextractor extractor-like object; constructor keyword of `tl.debug.ChecksSession` (required): `nn.Module` or name->parameter Mapping supplying the watched inventory. Tied tensors dedup by identity, aliases preserved; keyword of `tl.debug.bisect_precision` (required): Source model. It is deep-copied twice and never mutated; keyword of `tl.debug.check_determinism` (required): Source model; never mutated (deepcopy-impossible models return an `inconclusive` verdict with a `model_factory` remedy); keyword of `tl.debug.find_nan` (required): PyTorch model to execute; keyword of `tl.debug.graph_breaks` (required): Eager PyTorch module to inspect; keyword of `tl.debug.guess_input_shape` (required): PyTorch module to probe; keyword of `tl.debug.isolated_capture` (required): Source model. Never mutated; each call copies it afresh; keyword of `tl.debug.sketch_grad_fn` (keyword-only; default `None`): Optional module providing leaf names via `named_parameters()`; keyword of `tl.debug.walk_grad_fn` (keyword-only; default `None`): Optional module whose `named_parameters()` provide leaf names (composes with `params`; explicit `params` wins on collisions); keyword of `tl.experiment.sweep_sites` (keyword-only; default `None`): REQUIRED on the live_hook lane (the real forward's module and input); unused on replay; constructor keyword of `tl.experimental.Session` (required): Model to invoke repeatedly; keyword of `tl.experimental.attribute_walk` (required): Root PyTorch module; keyword of `tl.experimental.auto_capture` (required): Model whose `forward` method is wrapped temporarily; keyword of `tl.extract`, `tl.extract_dataset`, `tl.pluck` (required): PyTorch model to run; keyword of `tl.fingerprint.assert_unchanged`, `tl.fingerprint.model` (required): Model or callable accepted by `trace`; constructor keyword of `tl.trackers.WatchSession` (required); keyword of `tl.inventory.list_sites` (required): PyTorch model to inventory; keyword of `tl.mechinterp.edge_scores` (required): A pre-norm additive-residual language model (the models `component_graph` accepts; eager attention); keyword of `tl.mechinterp.measure_trajectory` (required): Model whose blocks expose `facet` and whose head the lens is built from (the same facet recipes `logit_lens` reads); keyword of `tl.mechinterp.patch_grid` (required): Model to trace and rerun (state transactionally restored). The two prompts; length mismatch refuses at the facet shapes; keyword of `tl.mechinterp.patch_head` (required): The model (eval mode; SDPA or eager -- the lowering is implementation-independent, which is what makes it verifiable). Equal-length prompts; the delta patches CLEAN behavior into the CORRUPTED run; keyword of `tl.mechinterp.patch_query_heads_grid` (required): Grouped-query attention model to trace and rerun (state restored). The two prompts; equal lengths are required; keyword of `tl.neuro.plan_extraction` (required): Model to plan against (held weakly; keep it alive through run()); keyword of `tl.preprocessing.attach_authority` (required): Model object to annotate (returned unchanged otherwise); keyword of `tl.preprocessing.check_card` (keyword-only; default `None`): Model whose card is the reference, resolved through `resolve` (`model=`). Defaults to the trace's live source model when `authority` is not given; keyword of `tl.preprocessing.resolve` (keyword-only; default `None`): Auto-discovery source, used only when `authority` is `None` and only for metadata the model object genuinely carries; torchvision weights are never inferred from the model class; keyword of `tl.record`, `tl.recording.dry_run` (required): PyTorch module to execute; constructor keyword of `tl.recording.Recorder` (required): PyTorch module to record; keyword of `tl.release_model` (required): Model whose full module tree should be restored. The operation is safe for never-traced and already-released models; keyword of `tl.report.doctor` (default `None`): Optional eager model to inspect; its forward is never executed; keyword of `tl.report.flops_vs_dispatch` (required): The model (any `tl.trace`-able `nn.Module`); keyword of `tl.report.kernel_profile` (required): The model to run (any `tl.trace`-able callable); keyword of `tl.stats.aggregate` (required): Model to capture; keyword of `tl.sweep` (required): PyTorch module to capture; keyword of `tl.trace` (required): PyTorch model, or a bound method of one: `tl.trace (model.generate, ids, ...)` resolves the owner via `method.__self__`, wraps it in a TL-authored synthetic root, and calls the method exactly once; keyword of `tl.trackers.watch` (required): The live module to watch. Never wrapped, never re-traced: tier P installs optimizer hooks only, tier M ordinary module hooks; keyword of `tl.validate` (required): Model to validate; keyword of `tl.validation.validate_backward_pass` (required): PyTorch model; keyword of `tl.validation.validate_forward_pass` (required): Model or callable to validate |
| `model_a=` | keyword of `tl.debug.compare_params` (required): First model |
| `model_b=` | keyword of `tl.debug.compare_params` (required): Second model |
| `model_class_name=` | keyword of `tl.compat.from_timm` (required): timm model name |
| `model_id=` | keyword of `AttributionGraph.to_neuronpedia`, `Direction.upload_to_neuronpedia` (keyword-only; required): Neuronpedia's model id, for example `"gemma-2-2b"` |
| `model_identity=` | keyword of `tl.extract_dataset` (keyword-only; default `'measured'`): Disk mode (DOCUMENTED-UNSTABLE): `"measured"` (default — cryptographic threaded-Merkle state digest), `"none"` (recorded opt-out), or an assertion Mapping for unmeasurable models |
| `model_name=` | keyword of `tl.bridge.sae.write_delphi` (keyword-only; required): Hugging Face id of the model, which delphi uses to load the tokenizer when none is passed |
| `model_object_id=` | keyword of `tl.compat.from_huggingface` (required): Hugging Face model identifier or local model path |
| `model_or_id=` | keyword of `tl.bridge.inseq.attribute` (required): Model object or model identifier accepted by `inseq.load_model` |
| `model_or_log=` | keyword of `tl.bridge.sae.splice` (required): Live `nn.Module` (captured here with a plain `tl.trace`) or an already-captured intervention-ready `Trace` |
| `model_type=` | keyword of `Direction.to_dialz` (keyword-only; default `None`): `SteeringVector.model_type`; defaults to `model_type`; keyword of `Direction.to_repeng` (keyword-only; default `None`): `ControlVector.model_type`; defaults to `model_type`; keyword of `tl.bridge.dialz.from_direction` (keyword-only; default `None`): `SteeringVector.model_type`; defaults to `direction.model_type`; keyword of `tl.bridge.repeng.from_direction` (keyword-only; default `None`): `ControlVector.model_type`; defaults to `direction.model_type`; constructor keyword of `tl.intervention.Direction` (required): `config.model_type` of the traced model, or `None` when it has none |
| `module=` | keyword of `Trace.draw` (keyword-only; default `None`): Focus the picture on one module (a `Module` or its address); keyword of `tl.experiment.head_ablation_candidates` (required): REQUIRED named module address (e.g. `"transformer.h.5.attn"`) — scoping is mandatory; there is no all-layers spelling here; keyword of `tl.splice_module` (required): Module to call under `pause_logging()` in the execution helper; constructor keyword of `tl.transforms.OpaqueStep` (required): `__module__` disclosure of the callable (or its type) |
| `module_address=` | keyword of `PatchedRun.read_facet` (required); constructor keyword of `tl.attribution.SiteTarget` (required); constructor keyword of `tl.mechinterp.Coordinate` (default `None`): Innermost containing module address, when resolved; constructor keyword of `tl.mechinterp.ProjectionSource` (required): Owning module of the underlying parameter; constructor keyword of `tl.attribution.SiteTarget` (required) |
| `module_identity_mode=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `module_identity_modes=` | constructor keyword of `tl.backends.BackendCapabilities` (default `('torch_module',)`): Supported module identity modes |
| `module_overrides=` | keyword of `Trace.draw` (keyword-only; default `None`): Graphviz attribute overrides for the graph, forward edges, gradient edges and module boxes; constructor keyword of `tl.options.VisualizationOptions` (optional): Module cluster style overrides |
| `module_path=` | keyword of `tl.site` (keyword-only; default `None`): Module-site match: the full `/`-joined site axis or any one module address on it |
| `module_stack=` | keyword of `CaptureBackend.pop_module_frame`, `CaptureBackend.push_existing_module_frame` (required): Mutable module-stack list owned by the active recording state |
| `module_subtrees=` | constructor keyword of `tl.mechinterp.SavePlan` (required) |
| `modules=` | keyword of `tl.mechinterp.attention_head_contributions` (default `None`): Attention-module selector: `None` (all), index/address, or a list; keyword of `tl.mechinterp.head_scores` (keyword-only; default `None`): Optional attention-module selector (indices/addresses, as in head contributions) |
| `multi_site=` | keyword of `tl.experiment.sweep_grid` (keyword-only; default `False`); keyword of `tl.experiment.sweep_sites` (keyword-only; default `False`): `True` (every candidate may span sites) or a collection of candidate ids declared multi-site. An undeclared candidate whose resolved site count exceeds one — or cannot be proven one — REFUSES |
| `n=` | keyword of `Accessor.head` (default `5`); keyword of `tl.viz.nodes.channel_grid` (default `16`): Maximum number of channels to render. When the tensor has more channels, the grid shows the FIRST `n` and draws a `+K more` marker for the hidden remainder |
| `n_splits=` | keyword of `tl.bridge.sae.write_delphi` (keyword-only; default `5`): Number of latent splits, delphi's `CacheConfig.n_splits` |
| `name=` | keyword of `AttentionRecording.calls_for`, `Bundle.downstream`, `Bundle.stored_comparison`, `Bundle.upstream`, `FacetView.has`, `PatchedRun.read_facet`, `Trace.using`, `WatchSession.emit_check`, `tl.agent.spec_for`, `tl.bridge.sae.unregister`, `tl.export.describe_target`, `tl.export.resolve_target`, `tl.export.unregister_target`, `tl.mechinterp.stack_facet`, `tl.param` (required); keyword of `Bundle.fork` (default `None`): Optional suffix prefix for forked member names; keyword of `Bundle.store_comparison` (keyword-only; default `None`); keyword of `Direction.upload_to_neuronpedia` (keyword-only; required): The name Neuronpedia shows for the vector; keyword of `FacetView.get` (required): Facet name; keyword of `ReadTable.aggregate_targets` (keyword-only; default `None`): Optional reducer name recorded in the aggregated rows' `reduction` column (defaults to the callable's `__name__`); keyword of `ReadTable.column` (required): A `ReadRow` field name; keyword of `Trace.fork` (default `None`): Optional name for the forked log; keyword of `Trace.modules_with_facet` (required): Facet name to look up using the current capture's available values; constructor keyword of `tl.agent.AgentToolSpec`, `tl.facets.FacetFunction` (required); keyword of `tl.agent.call_tool`, `tl.agent.call_tool_envelope`, `tl.bridge.mcp.call_tool` (required): Registered tool name; keyword of `tl.appliances.register` (required): Appliance name; starts with a letter, then letters, digits, `_`, `.` or `-`; constructor keyword of `tl.backends.BackendSpec` (required): Backend identifier used by `backend=` and `Trace.provenance.backend`; keyword of `tl.backends.spec`, `tl.backends.unregister` (required): Registered backend name or alias; keyword of `tl.bridge.lit.model` (keyword-only; default `None`): Display name; defaults to the model class name; keyword of `tl.bridge.sae.register` (required): Family name: no brackets, commas or whitespace; keyword of `tl.dataset_extraction.register_pure_module` (required): Top-level module name (e.g. `"einops"`); keyword of `tl.examples.load` (required): Example name. `"name"`, `"tiny_linear"`, and `"linear"` are accepted aliases for the built-in smoke artifact; keyword of `tl.export.register_target` (required): Stable target name (the `tl.export.<name>` spelling for builtins); keyword of `tl.export.wandb` (default `'torchlens_trace'`): Logged table key; keyword of `tl.facet` (required): Facet name to target; keyword of `tl.func` (required): Function name to match; keyword of `tl.grad_fn` (default `None`): Autograd class name or normalized grad_fn type to match; keyword of `tl.head` (default `None`): Optional facet name, such as `"q"`, `"k"`, or `"v"`; keyword of `tl.label` (required): TorchLens final, raw, short, or pass-qualified label; keyword of `tl.mechinterp.resolve_alias` (required): Full TLens name (2.x, 3.x-bridge or 4.x canonical), tuple form `(facet, layer)`, or compact `"k6"`; keyword of `tl.observers.log_value` (required): Value name; keyword of `tl.observers.span` (required): Span name; constructor keyword of `tl.options.CaptureOptions` (optional); constructor keyword of `tl.preprocessing.AuthorityAdapter` (required): Stable adapter name (rides the provenance record's config); keyword of `tl.preprocessing.register_authority_adapter` (required): Stable adapter name; constructor keyword of `tl.stats.Aggregator` (keyword-only; default `None`): Optional metric name; constructor keyword of `tl.stats.CKA`, `tl.stats.Covariance`, `tl.stats.CrossCovariance`, `tl.stats.Mean`, `tl.stats.Norm`, `tl.stats.PCA`, `tl.stats.Quantile`, `tl.stats.TopK` (default `None`): Optional metric name; constructor keyword of `tl.stats.Histogram`, `tl.stats.RunningStats` (default `None`); constructor keyword of `tl.trackers.TagGrammar` (keyword-only; default `None`); keyword of `tl.trackers.watch` (keyword-only; default `None`): Multi-model tag component distinguishing sessions that share a sink; constructor keyword of `tl.transforms.PlannedStep` (required): Transform name (registry entry id), or the opaque disclosure name; constructor keyword of `tl.transforms.TransformDefinition` (required): Stable transform name (the registry entry id); keyword of `tl.transforms.lookup_transform` (required): Registered transform name; keyword of `tl.inputs.register`, `tl.outputs.register` (keyword-only; required): Unique preprocessor or decoder name, used by `unregister`, stamped in the trace's provenance and, for a decoder, its on-demand `style=`; keyword of `tl.inputs.unregister`, `tl.outputs.unregister` (required): The name given to `register` |
| `name_hint=` | keyword of `tl.where` (keyword-only; default `None`): Optional human-readable name for diagnostics and saved specs |
| `name_or_pos=` | keyword of `ReceptiveFieldRuleContext.arg` (required): Argument name, resolved against the raw call signature, or a direct index into the captured (tensor-filtered) positional arguments |
| `names=` | keyword of `Bundle.add` (default `None`): Optional member name or names; keyword of `Bundle.remove` (required): Member name, Trace object, or a list of either; keyword of `ChecksSession.register_frozen_check` (required); constructor keyword of `tl.Bundle` (keyword-only; default `None`): Optional names for a sequence of logs; keyword of `tl.sweep` (keyword-only; default `None`): Optional Bundle member names. When omitted, names are derived from `SWEEP_NAME` and the value index |
| `namespace=` | constructor keyword of `tl.trackers.TagGrammar` (keyword-only; default `None`); keyword of `tl.trackers.watch` (keyword-only; default `None`): Tag re-rooting remedy for layout collisions in a shared sink |
| `nan_color=` | keyword of `tl.viz.heatmap` (keyword-only; default `(235, 235, 235)`): RGB color used for non-finite data values |
| `narrate=` | keyword of `tl.record` (keyword-only; default `None`): Live narration scope: `True`, `"modules"`, a live selector, or a grouped `tl.options.NarrationOptions`. Narration and retention are fully independent ("narrate everything, keep nothing" is legal); constructor keyword of `tl.recording.Recorder` (keyword-only; optional); keyword of `tl.trace` (keyword-only; default `None`): Live narration: `True` narrates every completed tensor-output event plus module structure lines; `"modules"` narrates structure only; a live selector scopes the stream; `tl.options.NarrationOptions` groups the full surface (stats rungs, sink, crash tail). .. |
| `native_chrome_path=` | keyword of `tl.report.kernel_profile` (keyword-only; default `None`): When given, the NATIVE chrome trace (Kineto's clock, foreign events and all) is preserved at this path with a TorchLens mapping sidecar beside it (`<path>.torchlens-map.json`) carrying exact marker-name -> op-label rows. Host timestamps are never projected onto the profiler's clock |
| `needed=` | keyword of `tl.trackers.require_capabilities` (required) |
| `needle=` | keyword of `Accessor.search` (required): Substring matched against every string lookup key; keyword of `SiteInventory.resolve` (required): Selector, address, label, or bare leaf-name spelling |
| `neg_traces=` | keyword of `tl.intervention.fit_direction` (required): The negative prompts, laid out the same way: row `i` of the negatives pairs with row `i` of the positives |
| `negative_control=` | keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; required): Measured self-patch metric delta (mandatory, must be ~0) |
| `nested=` | constructor keyword of `tl.intervention.RegionEdge` (required): Whether any occurrence address has a nested (container) argument path -- disclosed because the splice must extend into the container |
| `net=` | keyword of `tl.bridge.lit.model` (required): The LIVE torch model. A finished Trace refuses: LIT calls `predict` on NEW inputs (edited text, counterfactuals), which a historical trace cannot execute |
| `neurons=` | keyword of `tl.mechinterp.mlp_neuron_contributions` (keyword-only; default `None`): Optional neuron indices kept as their own rows in every selected layer; the other neurons fold into one exact `other_neurons` row per layer, so the stack still sums to the MLP output |
| `new_ids=` | keyword of `tl.dataset_extraction.relabel_extraction` (required): Replacement identifiers, one per stimulus row, in row order |
| `new_segment=` | keyword of `WatchSession.step` (keyword-only; default `False`) |
| `node=` | constructor keyword of `tl.Selector` (required) |
| `node_influence_coverage=` | keyword of `AttributionGraph.to_neuronpedia` (keyword-only; default `0.8`): Shares of total influence kept, between 0 and 1 |
| `node_label_fields=` | keyword of `Trace.draw` (keyword-only; default `None`): Explicit list of label rows to show on every node; constructor keyword of `tl.options.VisualizationOptions` (optional): Optional explicit label row fields |
| `node_overlay=` | keyword of `Trace.draw` (keyword-only; default `None`): Per-node label row: a stock overlay name (`"flops"`, `"time"`, `"bytes"`, `"magnitude"`, `"nan"`, `"grad_norm"`, `"intervention"`, `"bundle_delta"`), a mapping of layer label to value, or a callable `node -> value`; constructor keyword of `tl.options.VisualizationOptions` (optional): Built-in overlay name, an external label->value score mapping, or a callable invoked as `fn(node)` to compute a per-node overlay value |
| `node_overrides=` | keyword of `Bundle.draw` (keyword-only; default `None`) |
| `node_spec_fn=` | keyword of `Trace.draw` (keyword-only; default `None`): Per-node styling and selection callbacks (see `tl.viz.nodes`); constructor keyword of `tl.options.VisualizationOptions` (optional): Optional layer-node customization callback |
| `node_style=` | keyword of `Trace.draw` (keyword-only; default `'default'`): Node label style: `"default"` or `"profiling"`; constructor keyword of `tl.options.VisualizationOptions` (optional): Built-in node label/style preset |
| `nodes=` | keyword of `Bundle.draw` (keyword-only; default `'ops'`) |
| `noise_bank=` | keyword of `tl.attribution.noise_tunnel` (keyword-only; default `None`): Optional stored unit-scale draws (`noise_bank[sample][unique_leaf]`) replayed instead of fresh draws; oracle rows use this to feed our runner and a reference implementation the same bank |
| `noise_level=` | keyword of `tl.attribution.smoothgrad` (keyword-only; default `0.1`): Standard deviation of Gaussian noise added to each input copy; must be positive (zero noise refuses `smoothgrad_noise_level_invalid`) |
| `noise_std=` | keyword of `tl.attribution.infidelity` (keyword-only; default `0.003`): Gaussian scale, disclosed in every result (default 0.003); keyword of `tl.attribution.road_curve` (keyword-only; default `0.01`): Standard deviation of the noise added to imputed values (default 0.01, the authors' reference value); `0` imputes without noise |
| `norm=` | keyword of `tl.mechinterp.apply_norm_scale` (keyword-only; default `'final'`): `"final"` (the dataflow-anchored final norm) or a norm module address for intermediate folds |
| `norm_type=` | keyword of `tl.clip_norm` (default `2.0`): Norm order passed to `torch.linalg.vector_norm` |
| `normalize_params=` | constructor keyword of `tl.transforms.TransformDefinition` (required): Callable validating and normalizing raw constructor params into the canonical JSON-portable dict recorded in the spec |
| `note=` | keyword of `ReceptiveFieldRuleContext.axis_map`, `ReceptiveFieldRuleContext.dissolve`, `ReceptiveFieldRuleContext.full`, `ReceptiveFieldRuleContext.passthrough`, `ReceptiveFieldRuleContext.piecewise`, `ReceptiveFieldRuleContext.window`, `ReceptiveFieldRuleContext.window_edges` (keyword-only; default `None`); keyword of `ReceptiveFieldRuleContext.data_dependent`, `ReceptiveFieldRuleContext.unknown`, `ReceptiveFieldRuleContext.unsupported` (required); constructor keyword of `tl.transforms.SrpSizing` (required): The scope label: a DENSE-JL Euclidean heuristic, a poor guide for correlation-distance RDMs |
| `num_audited_ops=` | field of `DTypeRangeReport` |
| `num_backward_passes=` | field of `BackwardStatus`; field of `BackwardStyleInventory` |
| `num_checked_ops=` | field of `HealthFacts`, `NonfiniteReport` |
| `num_effect_region_ops=` | field of `ValidationStatus`; keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unverified` (keyword-only; default `0`): Number of effectful or stateful nodes excluded from pure replay |
| `num_failed_ops=` | field of `ValidationStatus`; keyword of `ValidationStatus.from_replay_counts` (keyword-only; default `0`): Number of replay failures; keyword of `ValidationStatus.result` (keyword-only; default `0`): Number of replay failures. Defaults to `0` for legacy boolean callers that only report aggregate pass/fail; keyword of `ValidationStatus.unverified` (keyword-only; default `0`): Number of replay failures. Must be zero for an unverified result |
| `num_fires=` | field of `BindReport` |
| `num_folds=` | keyword of `tl.neuro.compare` (keyword-only; default `5`): Cross-validation folds for ridge (each training fold needs at least 3 stimuli) |
| `num_forward_runs=` | field of `CleanStepTime`, `OverheadReport` |
| `num_frozen_weight_ops=` | field of `BackwardEstimate` |
| `num_grad_fn_records=` | field of `BackwardStatus` |
| `num_joins=` | field of `MergeReport` |
| `num_multiplied_ops=` | field of `BackwardEstimate` |
| `num_nonfinite=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required): Number of NaN or Inf gradient entries; nonzero makes validation indeterminate |
| `num_pages=` | field of `SavedFigure` |
| `num_pairs=` | constructor keyword of `tl.intervention.Direction` (required) |
| `num_pure_unverified_ops=` | field of `ValidationStatus`; keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unverified` (keyword-only; default `0`): Number of pure value nodes not replayed by a backend raw-op replay |
| `num_query_heads=` | field of `GqaInfo` |
| `num_random_draws=` | keyword of `tl.attribution.causal_audit` (keyword-only; default `10`): Number of seeded random-k sets (default 10) |
| `num_replayed_ops=` | field of `ValidationStatus`; keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.unverified` (keyword-only; required): Number of ops or regions that replay-validated successfully; keyword of `ValidationStatus.result` (keyword-only; default `0`): Number of ops or regions that replay-validated successfully |
| `num_resamples=` | keyword of `tl.neuro.compare` (keyword-only; default `None`): When set, every score gets a stimulus-bootstrap `ScoreInterval` (the ridge interval resamples the evaluation of the out-of-fold predictions; it does not refit); keyword of `tl.repgeom.cka_interval` (keyword-only; default `1000`): Bootstrap settings (see `rdm_compare_interval`); keyword of `tl.repgeom.compare_rdms_interval` (keyword-only; default `1000`): Bootstrap resamples (at least 2) |
| `num_samples=` | keyword of `tl.attribution.gradient_shap` (keyword-only; default `25`): Number of Monte-Carlo samples (default 25, captum parity); keyword of `tl.attribution.infidelity` (keyword-only; default `10`): Number of perturbation samples (default 10, reference parity); keyword of `tl.attribution.noise_tunnel` (keyword-only; default `25`): Number of noised copies (logical child calls); keyword of `tl.attribution.sensitivity` (keyword-only; default `10`): Number of perturbed evaluations (default 10); keyword of `tl.attribution.smoothgrad` (keyword-only; default `25`): Number of noised saliency samples to average; constructor keyword of `tl.transforms.SrpSizing` (required): Number of samples (rows) the bound was sized for; keyword of `tl.transforms.srp_dims_for` (required): Number of rows the projection must preserve pairwise (`>= 2`) |
| `num_steps=` | keyword of `tl.attribution.deletion_curve`, `tl.attribution.road_curve` (keyword-only; default `20`): Number of steps after the unperturbed start (default 20); keyword of `tl.attribution.insertion_curve` (keyword-only; default `20`): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.integrated_gradients` (keyword-only; default `50`): Number of midpoint Riemann samples along the straight path; keyword of `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (keyword-only; default `50`): Number of midpoint Riemann samples along the straight input path (Captum's default rule is Gauss-Legendre, so values at equal `num_steps` differ slightly; see `integrated_gradients`); keyword of `tl.attribution.text` (keyword-only; default `128`): Fixed integer grid (default 128) or `"auto"` for the dual-criterion ladder 64 -> 128 -> 256 -> 512; constructor keyword of `tl.options.EpisodeOptions` (default `None`) |
| `num_unchecked_ops=` | field of `HealthFacts`, `NonfiniteReport` |
| `num_unexamined_ops=` | field of `HealthFacts`, `NonfiniteReport` |
| `num_unknown_ops=` | field of `CostTreeRow` |
| `num_unmapped_ops=` | field of `HealthFacts` |
| `num_unverified_ops=` | field of `ValidationStatus`; keyword of `ValidationStatus.from_replay_counts` (keyword-only; required): Legacy aggregate number of nodes or regions excluded from per-op replay. When separate pure/effect counts are supplied, this is treated as an additional legacy/importer-region bucket; keyword of `ValidationStatus.result` (keyword-only; default `0`): Legacy aggregate number of nodes or regions excluded from per-op replay; keyword of `ValidationStatus.unverified` (keyword-only; required): Legacy aggregate number of nodes or regions excluded from per-op replay |
| `num_vectors=` | field of `BasisProvenance` |
| `num_violations=` | field of `ReceptiveFieldValidation` |
| `obj=` | keyword of `tl.bridge.treescope.consume_repr_suppression`, `tl.bridge.treescope.display`, `tl.bridge.treescope.treescope_repr` (required); keyword of `tl.compat.lovely.lovely`, `tl.compat.lovely.str`, `tl.compat.torchshow.show` (required): Tensor or TorchLens layer log |
| `occlude_leaf=` | keyword of `tl.attribution.occlusion_map` (keyword-only; default `0`): Traversal index of the attributed leaf to occlude; every other input is held fixed |
| `offline=` | keyword of `tl.preprocessing.check_card` (keyword-only; default `None`): Fetch policy for the HF tier of model discovery (see `resolve`); keyword of `tl.preprocessing.resolve` (keyword-only; default `None`): `None` (default) respects `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`; `True` forbids fetches; `False` permits them. Only the HF discovery tier can fetch, and its attempt is disclosed on the record |
| `on=` | keyword of `Bundle.compare` (keyword-only; default `'out'`): Tensor field to compare: `"out"` or `"grad"`; keyword of `Bundle.draw`, `Bundle.store_comparison` (keyword-only; default `'out'`); keyword of `SuperOp.aggregate` (default `'out'`): Tensor field to aggregate; keyword of `SuperOp.compare` (default `'out'`): Tensor field to compare; keyword of `tl.io.suppress_mutate_warnings` (default `True`); keyword of `tl.viz.bundle_diff` (keyword-only; default `'out'`): Tensor field compared at each site |
| `on_divergence=` | keyword of `Trace.run` (keyword-only; default `'raise'`): Strict divergence behavior or the sole poison-return opt-in |
| `on_error_only=` | constructor keyword of `tl.options.NarrationOptions` (keyword-only; optional): Narrate NOTHING in the happy path and speak only on failure -- cheap enough to recommend as a standing training-loop habit |
| `on_failure=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `'return'`): `"return"` for notebook-friendly diagnostics or `"raise"` for `ShapeInferenceError`. With `"return"`, inference failures of any kind (including trace-verification and internal errors) come back as a structured result instead of raising; only invalid arguments raise regardless |
| `on_feed_break=` | constructor keyword of `tl.options.EpisodeOptions` (default `'disclose'`): The arm switch (both arms built; the ruling selects the default's shape). `"disclose"` (default, the panel 2-1): return the one Trace with the break marked and claims refused across it. `"refuse"`: the settlement write raises typed `episode_feed_break_exogenous` carrying the settled .. |
| `on_forward_error=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional): Controls failed-forward handling: `"raise"` raises the capture's `TraceCaptureError`; `"attach_partial"` also hangs the failed partial `Recording` on it as `exc.partial_recording` before re-raising. `"return_partial"` returns `None` from the failed `log()` and exposes the failed .. |
| `on_nonfinite=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `on_predicate_error=` | constructor keyword of `tl.options.RecordingOptions` (optional); constructor keyword of `tl.recording.Recorder` (keyword-only; optional): Fastlog recording options |
| `on_save=` | constructor keyword of `tl.options.StorageOptions` (optional): Callback invoked with `(label, tensor)` for each saved out |
| `on_zero_fire=` | keyword of `Recipe.bind` (keyword-only; default `'warn'`): Zero-fire settlement policy, the same knob at every intervention door. A rule that never fired is always recorded on `.last_report.zero_fire_rule_ids`; `"warn"` (default) also emits a `bind_zero_fire` coded warning, `"error"` raises `bind_zero_fire` after the call, and `"disclose"` ..; keyword of `Trace.run` (keyword-only; optional): What a staged hook entry that fired at zero sites on this run does after the fact is recorded (`last_run["hooks_unfired"]`): `"warn"` (default) emits a `rerun_zero_fire` coded warning, `"error"` raises `rerun_zero_fire`, `"disclose"` records only. The same knob as `spec.bind` and ..; keyword of `tl.trace` (keyword-only; optional): What a capture-time `intervene=` that fired at zero sites does once the fact is recorded (`annotations["unmatched_capture_selectors"]`, intervention audit): `"warn"`, used when not passed, warns `trace_zero_fire`; `"error"` raises it; `"disclose"` records only. Same knob as .. |
| `onto=` | keyword of `tl.mechinterp.attention_head_contributions` (keyword-only; default `None`): Optional direction vector(s) `[d_model]` or `[d_model, n]`: rows are contracted BEFORE any full stack exists (contract-before-expand; the memory number lives in the module docstring). Contracted rows are score tensors, not residual writers; keyword of `tl.mechinterp.mlp_neuron_contributions` (keyword-only; default `None`): Optional direction vector(s) `[d_model]` or `[d_model, n]`; rows are contracted BEFORE any full row exists (contracted rows are scores, not residual writers) |
| `op=` | keyword of `ReceptiveFieldView.projective` (required): Source operation whose output grid owns `unit` coordinates; keyword of `tl.experimental.node_styles.attention_node_mode`, `tl.experimental.node_styles.vision_node_mode` (required): Per-pass Op or aggregate Layer being rendered; keyword of `tl.receptive_field.node_spec` (required): Target operation whose receptive-field ancestry is shown; keyword of `tl.transforms.pool_spatial` (default `'mean'`): `'mean'` (default) or `'max'`; keyword of `tl.transforms.pool_tokens` (required): One of `'mean'`, `'max'`, `'first'`, `'last'`. `first` and `last` gather each row's first/GREATEST VALID index under the mask, never physical endpoints (those need the recorded assertion); keyword of `tl.transforms.reduce` (required): One of `'mean'`, `'sum'`, `'max'`, `'min'` |
| `op_label=` | keyword of `KernelProfileReport.device_time_ns` (required); constructor keyword of `tl.mechinterp.Coordinate` (default `None`): Pass-qualified captured op label the value rode (`add_10_101:2`); constructor keyword of `tl.receptive_field.GradReceptiveField` (required); constructor keyword of `tl.types.OpRef` (keyword-only; required) |
| `op_or_label=` | keyword of `tl.debug.lineage` (required): Start op or lookup key accepted by `trace.__getitem__` |
| `op_row_index=` | constructor keyword of `tl.types.OpRef` (keyword-only; required) |
| `ops=` | keyword of `tl.receptive_field.cross_validate` (keyword-only; default `None`): Explicit operations or labels. `None` selects saved eligible compute ops |
| `optimizer=` | constructor keyword of `tl.debug.ChecksSession` (default `None`): One optimizer or a sequence. Step-family checks refuse at attach without one; overlapping param ownership across several refuses typed; constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.trackers.unscale_stage` (required); keyword of `tl.trackers.watch` (keyword-only; default `None`): The optimizer whose step boundary anchors parameter streams |
| `options=` | keyword of `CaptureBackend.start_session` (required); keyword of `tl.export.html` (keyword-only; default `None`): `tl.options.HtmlReportOptions` emission plumbing: `share_safe` (strip every embedded value including the frontier), `deterministic` (suppress the manifest timestamp for byte-exact regeneration), `vis_call_depth` (graph-depth knob, forwarded and disclosed), and the thumbnail budgets |
| `order_siblings=` | keyword of `Trace.draw` (keyword-only; default `True`): Keep sibling ops in execution order (a post-pass; off while `stack_by` is active); constructor keyword of `tl.options.VisualizationOptions` (optional): Whether Graphviz `dot` renders should verify and apply execution-order placement for true parallel siblings |
| `ordinal=` | keyword of `tl.site` (keyword-only; default `None`): 1-based occurrence ordinal of the position within its pass-qualified innermost call instance |
| `orient=` | keyword of `tl.export.json` (keyword-only; default `'records'`): JSON orientation passed to `DataFrame.to_json` |
| `orientation=` | constructor keyword of `tl.mechinterp.ProjectionSource` (required): `"in_out"` (Conv1D) or `"out_in"` (Linear) |
| `origin=` | constructor keyword of `tl.intervention.Population` (required): REQUIRED non-empty human provenance string (whence and why); keyword of `tl.intervention.population` (keyword-only; required): REQUIRED non-empty provenance string -- whence and why; keyword of `tl.subspace` (keyword-only; required); constructor keyword of `tl.types.InputProvenance` (required): `"user"` (rung 1), `"declared"` (rung 2), or `"inferred"` (rung 3) |
| `original=` | keyword of `Bundle.token_divergence` (required): The member (name or Trace) from the original run, the reference |
| `other=` | keyword of `Histogram.merge`, `RunningStats.merge` (required); keyword of `SuperOp.compare` (default `None`): Optional member name for a one-row comparison |
| `other_keep_op=` | keyword of `DryRun.filter` (default `None`): Predicate for op, input, and buffer events. Module boundary events have no predicate slot and are never re-selected |
| `out_to_parent_axis=` | keyword of `ReceptiveFieldRuleContext.axis_map` (required) |
| `outcome=` | keyword of `WatchSession.emit_check` (required) |
| `outlier_perc=` | keyword of `tl.viz.causal_trace_heatmap` (keyword-only; default `2.0`): Percentage clipped from each tail before rendering |
| `output=` | keyword of `CaptureBackend.build_record_context`, `CaptureBackend.detect_backend_semantics` (required); keyword of `tl.func` (keyword-only; default `None`): Optional output index or semantic role to match; keyword of `tl.repgeom.rank_transform_rdm` (keyword-only; default `'percentile'`): `"percentile"` (default) or `"rank"`; keyword of `tl.repgeom.rdm` (keyword-only; default `'square'`): `"square"` (default) or `"condensed"` (strict upper triangle, row-major -- the rsatoolbox/scipy vector form); constructor keyword of `tl.transforms.PlannedStep` (required): Predicted output `TensorSpec` |
| `output_container_structure=` | constructor keyword of `tl.backends.BackendCapabilities` (default `'none'`): Public output-container structure level. `"none"` means no stable paths/specs, `"paths_only"` means path metadata may exist but is not generally reconstructable, and `"full_spec"` means full portable `ContainerSpec` metadata is supported |
| `output_device=` | constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.repgeom.rdm` (keyword-only; default `None`): Where the result lives. `None`/CPU returns `np.ndarray` (historical); a non-CPU device returns a `torch.Tensor` there; keyword of `tl.validate` (keyword-only; default `'same'`): Where the validator capture keeps its saved activations, with the same values and default as `CaptureOptions.output_device` (`"same"`, `"cpu"`, or `"cuda"`). `"cpu"` lets a GPU model validate with its activations held in host memory; replay still runs each op on its original device. Saved ..; keyword of `tl.validation.validate_forward_pass` (keyword-only; default `'same'`): Where the validator capture keeps saved activations (same values and default as `CaptureOptions.output_device`). Saved function arguments (the replay inputs) other than parameters are kept there too and replay moves each back to its op's device, so under `"cpu"` the GPU keeps only parameters .. |
| `output_path=` | constructor keyword of `tl.facets.FacetSpec` (default `()`): Structural output path inside a container output |
| `output_route=` | keyword of `tl.splice_module`: the route name for the spliced output |
| `output_schema=` | constructor keyword of `tl.agent.AgentToolSpec` (required) |
| `output_sites=` | keyword of `CaptureBackend.emit_function_outputs` (required) |
| `output_transform=` | keyword of `Trace.run` (keyword-only; optional): Stored-transform sentinel, `False` to bypass, or explicit output transform callable for this run; keyword of `tl.extract_dataset` (default `None`): Transform slot, through the ONE ` coerce_transform` door (DOCUMENTED-UNSTABLE). Selector-valued `layers=` accepts `None` or one global chain (per-site mappings need knowable keys); keyword of `tl.neuro.plan_extraction` (keyword-only; default `None`): Optional unary transform executed live at the probe; constructor keyword of `tl.options.CaptureOptions` (optional) |
| `outputs=` | keyword of `CaptureBackend.extract_and_mark_outputs` (required): Raw model output object returned by the captured forward pass; keyword of `tl.debug.sketch_grad_fn` (required): Tensor(s) with a `grad_fn`; see `walk_grad_fn`; keyword of `tl.debug.walk_grad_fn` (required): A tensor with a `grad_fn` (a real loss tensor, a logits tensor, an `autograd.grad(..., create_graph=True)` result) or a sequence of them |
| `overlap=` | keyword of `tl.attribution.occlusion_map` (keyword-only; default `'average'`): `"average"` (default; per-element mean over covering windows) or `"sum"` |
| `overlay=` | keyword of `tl.attribution.grad_cam` (keyword-only; default `True`): Whether to render the CAM over the dependency-proven spatial input; keyword of `tl.receptive_field.node_spec` (keyword-only; default `True`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested; keyword of `tl.viz.nodes.feature_map` (keyword-only; default `True`): Whether to alpha-blend heatmaps over matching raw PIL stimuli when available |
| `overwrite=` | keyword of `Bundle.save`, `Trace.save_intervention` (keyword-only; default `False`): Whether an existing destination may be replaced; keyword of `MergedTrace.save` (keyword-only; default `False`); keyword of `Trace.save` (keyword-only; default `False`): Replace an existing directory at `path`; keyword of `tl.io.save_intervention` (keyword-only; default `False`): Whether an existing target directory may be replaced |
| `overwrite_index=` | keyword of `Buffer.value_after` (required) |
| `p=` | constructor keyword of `tl.stats.Norm` (default `2.0`): Norm order passed to `torch.linalg.vector_norm` |
| `pad_token=` | keyword of `tl.dataset_extraction.hf_collate` (keyword-only; default `None`): `None` (a missing pad token refuses typed) or `"eos"` — the explicit, disclosed opt-in recorded as `pad_token_source` |
| `padded_length=` | keyword of `tl.report.padding_waste` (keyword-only; required): The padded sequence length S every row was padded to |
| `padding=` | keyword of `ReceptiveFieldRuleContext.window` (keyword-only; default `0`) |
| `padding_mask=` | keyword of `tl.transformer.token_metrics` (keyword-only; default `None`): Optional `[n_positions]` mask (1 = real token); padded cells become N/A on BOTH strips |
| `page=` | keyword of `tl.viz.contact_sheet` (keyword-only; default `0`) |
| `param_names=` | field of `CheckFinding` |
| `params=` | keyword of `tl.debug.sketch_grad_fn` (keyword-only; default `None`): Optional `name -> tensor` leaf-naming mapping; keyword of `tl.debug.walk_grad_fn` (keyword-only; default `None`): Optional `name -> tensor` mapping (the torchviz convention, `dict(model.named_parameters())`) used to name `AccumulateGrad` leaves |
| `parent=` | constructor keyword of `tl.intervention.RegionEdge` (required): Pass-qualified label of the producing op; keyword of `tl.observers.register_tensor_connection` (required): Parent tensor already tagged by TorchLens |
| `parent_index=` | keyword of `ReceptiveFieldRuleContext.parent_role` (required): Index into the operation's ordered parent list |
| `pass_index=` | keyword of `Trace.activation_by_raw_label` (keyword-only; default `None`): Optional 1-based model-call pass index; keyword of `Trace.activations_by_pass` (required); constructor keyword of `tl.facets.FacetSpec` (default `None`): Pass index for pass-qualified homes; keyword of `tl.in_backward_pass` (required): One-based backward pass number; constructor keyword of `tl.mechinterp.Coordinate` (default `None`): Pass qualifier of the op (within-forward reuse disambiguation) |
| `passed=` | keyword of `ValidationStatus.result` (keyword-only; required): Whether replay validation passed |
| `passes=` | keyword of `tl.pass_variance`, `tl.stable_across_passes` (keyword-only; default `None`) |
| `patch_size=` | keyword of `tl.attribution.deletion_curve` (keyword-only; default `8`): Patch side, or `(height, width)`, for `features="patches"` (default 8; edge patches may be smaller); keyword of `tl.attribution.insertion_curve` (keyword-only; default `8`): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.road_curve` (keyword-only; default `8`): Patch side for `features="patches"` (default 8) |
| `path=` | keyword of `AttributionGraph.to_neuronpedia` (keyword-only; default `None`): When given, the JSON is also written there; keyword of `Bundle.draw` (default `None`); keyword of `Bundle.save`, `Trace.save_intervention`, `tl.io.save_intervention` (required): Destination `.tlspec` directory path; keyword of `Drawing.save`, `MergedTrace.save`, `tl.bridge.treescope.treescope_repr` (required); keyword of `ExtractionPlan.run` (keyword-only; default `None`): Artifact directory (disk mode) or `None` for in-memory; keyword of `PartialTrace.draw` (default `None`): Accepted for API symmetry; no file is written by this minimal renderer (`Drawing.save(path)` writes one); keyword of `ReadTable.save` (required): Destination file path (created or overwritten); keyword of `Trace.animate` (default `None`): Output path; `.html` is appended when the suffix differs. `None` (the default) keeps the animation in memory; keyword of `Trace.draw` (default `None`): Output path; its extension picks the format (`"g.png"` writes PNG). An explicit path is always written (overwriting), printed only when the trace is verbose. `None` (the default): inside a Jupyter kernel the picture displays inline and no file is written; elsewhere a collision-safe file is ..; keyword of `Trace.save` (required): Destination `.tlspec` directory; keyword of `tl.attribution.load_read_table` (required): Path written by `save`; keyword of `tl.bridge.depyf.dump` (required): Output directory for depyf's dump (created when missing). Required: depyf writes files, and this bridge never picks a location for you; keyword of `tl.bridge.sae.write_sae_dashboard` (keyword-only; required): Output `.json` file; keyword of `tl.dataset_extraction.export_extraction` (required): Source artifact directory; keyword of `tl.dataset_extraction.open_extraction` (required): Artifact directory written by disk-mode `extract_dataset`; keyword of `tl.dataset_extraction.relabel_extraction` (required): Artifact directory; keyword of `tl.debug.sketch_grad_fn` (default `None`): Output path WITHOUT extension (torchviz/graphviz convention: `path.<file_format>` is written). `None` skips file rendering and returns DOT source only; keyword of `tl.export.chrome_trace`, `tl.export.chrome_trace_diff`, `tl.export.json`, `tl.export.memory_timeline`, `tl.export.memory_timeline_v2`, `tl.export.model_explorer`, `tl.export.speedscope` (required): Destination JSON path; keyword of `tl.export.csv` (required): Destination CSV path; keyword of `tl.export.flamegraph` (required): Destination folded-stack text path; keyword of `tl.export.html` (required): Destination HTML path; keyword of `tl.export.model_explorer_diff` (required): Destination directory; keyword of `tl.export.netron` (default `None`): Destination JSON path; must end `.json` for netron's sniffer. `None` serves the artifact from memory (requires `open=True`); keyword of `tl.export.parquet` (required): Destination Parquet path; keyword of `tl.export.safetensors` (required): Destination `.safetensors` path (written atomically); keyword of `tl.extract_dataset` (default `None`): Optional directory. When supplied, each batch is committed as an immutable shard under the v2 protocol and paths are returned; keyword of `tl.io.cleanup_tmp` (required): The **target bundle path itself** (e.g. `"demo_bundle"`, the same path you pass to `save(path, ...)`/`load(path)`) -- NOT its containing directory. Sibling `.tmp.*`/`.bak.*` candidates are found by globbing `f"{Path(path).name}.tmp.*"` / `f"{Path(path).name}.bak.*"` inside ..; keyword of `tl.io.detect_tlspec_format`, `tl.io.inspect_tlspec` (required): Directory path to inspect; keyword of `tl.io.load_intervention_spec` (required): Directory containing an intervention `.tlspec`; keyword of `tl.io.read_safetensors` (required): File written by `safetensors`; keyword of `tl.load`, `tl.validation.validate_tlspec` (required): `.tlspec` directory path; keyword of `tl.neuro.export_npz` (required): Output file (consolidated) or directory (per-stimulus); constructor keyword of `tl.options.StorageOptions` (optional): Portable bundle directory for streamed out saves; keyword of `tl.recording.cleanup_partial` (required): Fastlog bundle path or final path whose `.tmp.*` siblings should be inspected; keyword of `tl.recording.recover` (required): Fastlog bundle directory or streaming temp directory; keyword of `tl.stats.load_fitted` (required): File written by `save_fitted`; keyword of `tl.stats.save_fitted` (required): Destination file path (conventionally `*.safetensors`); keyword of `tl.to_disk` (required): Destination bundle directory; it must not exist yet (`bundle_path_exists`); constructor keyword of `tl.trackers.HistoryReader`, `tl.trackers.JSONLSink` (required); keyword of `tl.transformer.render_answer_trajectory`, `tl.transformer.render_neuron_card`, `tl.transformer.render_prediction_ribbon`, `tl.viz.ablation_grid`, `tl.viz.logit_lens`, `tl.viz.metric_strip`, `tl.viz.token_strip` (required): Output path (PNG/SVG/PDF by suffix); keyword of `tl.viz.attention` (required): Output path; the suffix picks the format. Multi-page output writes `name-pageNN.ext` files; keyword of `tl.viz.attention_atlas`, `tl.viz.prediction_table` (required): Output path; pages write `name-pageNN.ext`; keyword of `tl.viz.bundle_diff` (keyword-only; default `'bundle_diff'`): Output path, with or without a recognised extension; constructor keyword of `tl.trackers.HistoryReader`, `tl.trackers.JSONLSink` (required) |
| `path_in_repo=` | keyword of `tl.bridge.huggingface_hub.from_hub` (keyword-only; default `'torchlens_artifact.tar.gz'`): Artifact filename inside the repository; keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `'torchlens_artifact.tar.gz'`): Destination filename inside the repository |
| `pattern=` | keyword of `tl.label_matches` (required): Regular expression pattern to match against TorchLens labels. The pattern is matched with `search` (partial match) |
| `pattern_source=` | keyword of `tl.mechinterp.head_scores` (keyword-only; default `None`): Optional callable `module -> [b, head, dst, src]` pattern override (the SSM pseudo-pattern socket); default reads the `pattern` facet |
| `payload=` | keyword of `CaptureBackend.tensor_ref`, `JSONLSink.emit_graph`, `MemorySink.emit_graph`, `MergedFinding.from_dict`, `SidecarFamily.validate_payload`, `tl.io.attach_sidecar` (required); keyword of `TokenScores.from_attribution` (required): A `TokenAttributionPayload`-shaped object (`display_tokens`, `scores`, `footer_lines`, `score_domain` attributes); keyword of `tl.agent.canonical_dumps` (required): JSON-safe payload (run through `json_safe` first when the value may contain non-finite floats or tuples); keyword of `tl.export.validate_model_explorer_payload` (required): A `{label, graphs, ...}` collection dict (the exporter's output) |
| `payload_hints=` | keyword of `Op.materialize_grad`, `Op.materialize_out`, `tl.io.materialize_nested` (keyword-only; default `None`): Optional backend payload hints used during materialization; keyword of `tl.load` (keyword-only; default `None`): Optional backend payload hints used during materialization. This is separate from `map_location`; JAX sharding hints must be passed here |
| `payload_load_status=` | keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unavailable_loaded_runtime_stripped`, `ValidationStatus.unverified` (keyword-only; default `None`): Optional payload materialization status attached by `tl.load` |
| `payload_materialization=` | constructor keyword of `tl.backends.BackendCapabilities` (required): Whether loaded payloads can materialize as runtime arrays |
| `payload_policy=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `peak_bytes_per_s=` | field of `MachineBalance` |
| `peak_flops_per_s=` | field of `MachineBalance` |
| `peaks_by_mode=` | keyword of `tl.report.mfu` (keyword-only; required) |
| `peaks_source=` | constructor keyword of `tl.report.MfuProvenance` (default `'user_supplied'`) |
| `per_axis_edges=` | keyword of `ReceptiveFieldRuleContext.window_edges` (required) |
| `per_unit=` | keyword of `tl.attribution.causal_audit` (keyword-only; default `True`): Ablate every unit alone for the per-unit table and rank agreement (default `True`); past 1024 units the audit warns once and names `per_unit=False` (and `edges=` for an EDGE table) |
| `period=` | constructor keyword of `tl.mechinterp.FixedPeriodInduction` (required): The repeated sequence's length, a positive int |
| `perturb=` | keyword of `tl.attribution.infidelity` (keyword-only; default `'gaussian'`): `"gaussian"` (default; `dx ~ N(0, noise_std^2)`), the named `"square_removal"`, or a fully compatible callable `perturb(inputs_tuple) -> (perturbations, perturbed_inputs)` where both returns mirror the attributed-leaf tuple |
| `phase=` | keyword of `HistoryView.series`, `tl.viz.histogram_waterfall`, `tl.viz.quantile_fan` (default `None`) |
| `pinned=` | constructor keyword of `tl.backends.FrameworkRequirement` (required): The exact version the per-change CI leg installs, or `None` while the leg is not pinned yet (a graduation blocker, never an error here) |
| `pipeline_id=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): Caller-asserted pipeline identity: permits resuming past a PARTIAL callable classification or a deliberate complete-class refactor, with strict continuity — the id must already be recorded in the artifact's prefix |
| `pl_module=` | keyword of `LightningWatchCallback.on_train_end`, `LightningWatchCallback.on_train_start` (required) |
| `plan=` | keyword of `Trace.check_plan` (required): Iterable of selector entries or `(selector, replacement)` pairs |
| `plan_fn=` | constructor keyword of `tl.transforms.TransformDefinition` (required): `(spec, input_spec, ctx) -> PlannedStep` shape/dtype predictor; refuses typed instead of guessing |
| `plane=` | keyword of `ReceptiveFieldView.show` (keyword-only; default `None`): Required plane selector for three-dimensional inputs |
| `platform=` | constructor keyword of `tl.io.JaxPayloadLoadHint` (default `None`): Optional JAX device platform to use for metadata reconstruction, for example `"cpu"` |
| `point=` | keyword of `CometSink.emit_histogram`, `CometSink.emit_scalar`, `CometSink.emit_text`, `JSONLSink.emit_histogram`, `JSONLSink.emit_scalar`, `JSONLSink.emit_text`, `MLflowSink.emit_histogram`, `MLflowSink.emit_scalar`, `MLflowSink.emit_text`, `MemorySink.emit_histogram`, `MemorySink.emit_scalar`, `MemorySink.emit_text`, `Sink.emit_histogram`, `Sink.emit_scalar`, `Sink.emit_text`, `TensorBoardSink.emit_histogram`, `TensorBoardSink.emit_scalar`, `TensorBoardSink.emit_text`, `WandbSink.emit_histogram`, `WandbSink.emit_scalar`, `WandbSink.emit_text` (required) |
| `pointwise_coords=` | keyword of `ReceptiveFieldBox.slices` (default `None`): Coordinates required for pointwise axes. Omitted coordinates use full slices, representing all possible same-index locations |
| `policy=` | keyword of `CaptureBackend.safe_copy`, `CaptureBackend.tensor_ref` (required) |
| `pool=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): Optional pool preset (DOCUMENTED-UNSTABLE): one of `flatten` / `spatial_mean` / `spatial_max` / `token_mean` / `token_max` / `token_sum` / `cls` / `last_token` (token presets are MASK-AWARE), a `{"preset": ..., "unmasked": true}` recorded override, or a per-key .. |
| `pooling=` | keyword of `tl.bridge.lit.model` (keyword-only; default `'mean_masked'`): `"mean_masked"` / `"first_token"` / `"last_unmasked"` or a callable ` (tensor, attention_mask) -> [batch, features]` |
| `population=` | constructor keyword of `tl.intervention.SamplingPlan` (required); keyword of `tl.intervention.sample_from` (required): The prepared population noun (`reference`) |
| `pos_traces=` | keyword of `tl.intervention.fit_direction` (required): One `Trace` or a sequence of traces of the positive prompts, one prompt per batch row. Rows are concatenated in order |
| `position=` | keyword of `ContributionScores.topk` (keyword-only; default `-1`): Index into the SELECTED positions axis (default: the last); keyword of `LogitLens.summary`, `LogitLens.top_tokens` (keyword-only; default `-1`): Sequence position to inspect; keyword of `tl.mechinterp.inspect_prompt` (keyword-only; default `-1`): Position whose next-token distribution is inspected (default last); keyword of `tl.transformer.prediction_trajectory` (keyword-only; default `-1`): Sequence position to inspect (negative = from the end) |
| `position_axis=` | keyword of `tl.patch_from` (keyword-only; default `1`): Axis the positions index, on both the donor and the target value |
| `positions=` | keyword of `LoreftArtifact.subspace` (keyword-only; default `None`): Optional bool mask over the site's axes except the trailing feature axis (`[batch, sequence]` for a transformer block), True where pyreft edits (LoReFT's `unit_locations`). `None` edits every position; keyword of `tl.mechinterp.attention_head_contributions`, `tl.mechinterp.mlp_neuron_contributions` (keyword-only; default `None`): Optional position indices; rows are sliced on axis 1 after the remainder proof runs on full tensors; keyword of `tl.mechinterp.direct_logit_contributions` (keyword-only; default `-1`): Position index or indices (default: the last position); keyword of `tl.mechinterp.full_decomposition` (keyword-only; default `None`); keyword of `tl.mechinterp.patch_grid` (keyword-only; default `None`): Position indices for the position axis (default: all); keyword of `tl.mechinterp.residual_accumulation` (keyword-only; default `None`): Optional position indices; rows are views sliced on axis 1; keyword of `tl.mechinterp.residual_decomposition` (keyword-only; default `None`): Optional position indices; row VIEWS are sliced on axis 1 after the identity gate runs on the full tensors; keyword of `tl.patch_from` (keyword-only; default `None`): Source-to-target position map; `None` patches the whole value; keyword of `tl.positions` (required): An int, a non-empty list of ints, a slice, a per-row `(batch,)` integer tensor, or `"first"` / `"last"` (the first / last REAL token of each row, resolved from `mask`); constructor keyword of `tl.stats.UnitTopK` (required): `int64 [units, k, position_axes]`: the index along each position axis (the axes other than the example and unit axes, in order); `position_axes` is 0 when none remain; `-1` in empty slots; keyword of `tl.viz.logit_lens` (keyword-only; default `None`): Column positions to draw (negative counts from the end); default all, refused above `MAX_HEATMAP_POSITIONS` rather than cropped |
| `positive_control=` | keyword of `tl.transformer.receipt_from_patch_grid` (keyword-only; required): Measured large-perturbation metric delta (mandatory, must move -- the only check that catches the silent no-fire class) |
| `post_clip_total=` | keyword of `ClipLedger.observe_step` (required) |
| `postprocess=` | constructor keyword of `tl.options.RecordingOptions` (optional) |
| `pre_clip_total=` | keyword of `ClipLedger.observe_step` (required) |
| `predicate=` | keyword of `Trace.preview_fastlog` (default `None`): Predicate callables that receive synthesized fastlog `RecordContext` objects; keyword of `tl.facets.register` (keyword-only; default `None`): Arbitrary predicate on the target record; keyword of `tl.where` (required): Callable that receives a layer pass record |
| `predictions=` | keyword of `tl.transformer.prediction_table` (required): A `LogitLensPredictions` whose final row is the native-output row (the default extraction includes it); keyword of `tl.transformer.prediction_trajectory` (required): A `LogitLensPredictions` |
| `preferred_sizes=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `(224, 256, 384, 299, 128, 96, 64, 32, 28)`): Spatial candidates to try before measured fallback search |
| `prefix=` | keyword of `Bundle.add_alternative` (keyword-only; default `'cf_'`): Prefix naming each second-run member's copy (default `"cf_"`); keyword of `tl.export.aim` (default `'torchlens'`): Metric, param and tag name prefix; keyword of `tl.export.mlflow` (default `'torchlens'`): Metric, tag and artifact-folder name prefix; keyword of `tl.export.tensorboard` (keyword-only; default `'torchlens'`): Metric name prefix |
| `prepare=` | keyword of `tl.debug.isolated_capture` (keyword-only; default `None`): Optional callable applied to the fresh copy AFTER release and BEFORE tracing (e.g. `lambda m: m.to(torch.float64)` for a reference-dtype run). May return the model or `None` for in-place preparation |
| `prepared_model=` | keyword of `CaptureBackend.cleanup_failed_forward_session`, `CaptureBackend.cleanup_stopped_forward_session` (required): Backend-prepared model object, or a backend-specific cleanup tuple; keyword of `CaptureBackend.cleanup_model_session` (required) |
| `preprocessing=` | keyword of `tl.bridge.brain_score.activations_extractor` (required): Brain-Score preprocessing callable mapping stimulus paths to model inputs (for images, typically `load_preprocess_images`); `None` passes stimuli through unchanged; constructor keyword of `tl.types.InputProvenance` (default `None`): Nested preprocessing provenance (e.g. the HF tokenizer record when a prompt string was the gold input). Reserved slots (memo section 7). Unused in wave 1 |
| `preserved_sections=` | constructor keyword of `tl.Bundle` (keyword-only; default `None`): Load-side carriage for unknown NAMESPACED top-level `bundle.json` sections (loader doctrine leg (c), the C07X amendment): preserved verbatim, disclosed through `preserved_sections`, never executed, and re-emitted on save. User construction normally leaves this `None` |
| `pretrained=` | keyword of `tl.compat.from_timm` (keyword-only; default `False`): Whether timm should load pretrained weights |
| `print_fn=` | constructor keyword of `tl.trackers.HFTrainerWatchCallback`, `tl.trackers.LightningWatchCallback` (keyword-only; optional) |
| `priority=` | constructor keyword of `tl.backends.BackendSpec` (default `0`): Auto-resolution priority. Equal-priority matches are ambiguous; keyword of `tl.inputs.register`, `tl.outputs.register` (keyword-only; default `100`): Sort key; lower values run first, and the built-in routes sit at 10 to 30 |
| `private=` | keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `None`): Optional repository privacy flag used when creating the repo |
| `probe=` | constructor keyword of `tl.preprocessing.AuthorityAdapter` (required): Cheap predicate: does this object look like this authority kind?; keyword of `tl.preprocessing.register_authority_adapter` (required): Cheap predicate over candidate authority objects |
| `probe_budget=` | keyword of `ReceptiveFieldView.of`: the most autograd probes `method="gradient"` may run before refusing (`influence_probe_budget`) |
| `probe_inputs=` | keyword of `tl.neuro.plan_extraction` (required): One REAL probe batch (leading stimulus dimension >= 1). The plan is valid only for this input signature |
| `progress=` | keyword of `ExtractionPlan.run` (keyword-only; default `True`): Whether to show batch progress; keyword of `tl.extract_dataset` (default `True`): Whether to wrap batch iteration with `tqdm` |
| `prompt=` | keyword of `AttributionGraph.to_neuronpedia` (keyword-only; default `None`): The prompt string; `None` decodes the input tokens together; keyword of `tl.compat.from_sentence_transformers` (keyword-only; default `None`): Optional prompt name retained for migration notes; keyword of `tl.mechinterp.test_prompt` (required) |
| `protocol=` | constructor keyword of `tl.backends.BackendSpec` (default `None`): Lazy factory for this backend's `BackendProtocol` object (the one protocol every backend fills; `_backend_protocol.py`). Validated at registration like `capture_backend` |
| `provenance=` | constructor keyword of `tl.receptive_field.GridLayout`, `tl.types.Site` (required); constructor keyword of `tl.intervention.Direction` (required): Origin, method, geometry and content digest of the vector; constructor keyword of `tl.mechinterp.Coordinate` (default `'captured'`): How the value was obtained (`"captured"`, `"computed"`, `"verified_constant"`, `"injected"`), extension-open by design; keyword of `tl.report.mfu` (keyword-only; default `None`) |
| `provider=` | keyword of `tl.appliances.register` (keyword-only; default `None`): Optional provider identity for out-of-tree packages; defaults to an anonymous user provider row; keyword of `tl.export.register_target` (keyword-only; default `None`): Stable provider identity; defaults to the TorchLens builtin row; keyword of `tl.io.register_sidecar_family` (keyword-only; default `None`): Stable provider identity; defaults to the TorchLens builtin row (an out-of-tree provider passes its own); keyword of `tl.transforms.register_transform` (keyword-only; default `None`): Optional provider identity for out-of-tree distributions; defaults to an anonymous user provider row |
| `qualname=` | constructor keyword of `tl.transforms.OpaqueStep` (required): `__qualname__` disclosure of the callable (or its type) |
| `qualnames=` | constructor keyword of `tl.facets.FacetFunction` (required): Fully-qualified class names matched by this recipe |
| `quantiles=` | constructor keyword of `tl.stats.Quantile` (default `(0.5, 0.95, 0.99)`): Quantiles in `[0, 1]` to estimate |
| `query_head=` | keyword of `GqaInfo.group`, `GqaInfo.header` (required) |
| `query_tokens=` | keyword of `tl.transformer.attention_view` (keyword-only; default `None`): Destination-axis tokens (cross-attention; T5 is rectangular); keyword of `tl.transformer.attention_views` (keyword-only; default `None`): Destination-axis tokens (cross-attention) |
| `radius=` | keyword of `tl.attribution.sensitivity` (keyword-only; default `0.02`): L-infinity perturbation radius (default 0.02) |
| `ragged=` | keyword of `tl.extract_dataset` (keyword-only; default `'refuse'`): `"refuse"` (default) \| `"trim"` \| `"as_captured"`. `refuse` fails typed BEFORE committing the first width-drifted shard; `trim` stores true per-stimulus extents as the packed values/offsets/shapes carrier (mask-derived ` (start, extent)` slicing — left padding is why extent .. |
| `raise_on_failure=` | keyword of `tl.receptive_field.cross_validate` (keyword-only; default `False`): Raise on `FAIL` only |
| `randomization=` | keyword of `tl.attribution.randomization_test` (keyword-only; default `'cascading'`): `"cascading"` (default) or `"independent"` |
| `rank=` | keyword of `PromptInspection.assert_rank` (required); keyword of `tl.transforms.axis_for_role` (required): Rank of the tensor being addressed |
| `rank_install_epochs=` | keyword of `tl.distributed.verify_membership_lineages` (default `None`): Optional explicit install epochs per rank. When omitted, each membership uses the epoch stamped on the rank's ledger events for that membership (the normal, self-describing case) |
| `rank_ledgers=` | keyword of `tl.distributed.verify_membership_lineages` (required): Mapping from global rank to that rank's group-lifecycle ledger |
| `ranking=` | keyword of `tl.attribution.road_curve` (keyword-only; default `'most_relevant_first'`): `"most_relevant_first"` (default, MoRF) or `"least_relevant_first"` (LeRF) |
| `raw_label=` | keyword of `Trace.activation_by_raw_label` (required): Raw operation label emitted during capture |
| `rdm_a=` | keyword of `tl.repgeom.compare_rdms` (required): Square RDM (`N x N`) or condensed upper-triangle vector; keyword of `tl.repgeom.compare_rdms_interval` (required): Square RDMs (or condensed upper triangles) over the same stimuli in the same order |
| `rdm_b=` | keyword of `tl.repgeom.compare_rdms` (required): Square RDM or condensed vector over the SAME stimuli in the same order; keyword of `tl.repgeom.compare_rdms_interval` (required): Square RDMs (or condensed upper triangles) over the same stimuli in the same order |
| `rdm_metric=` | keyword of `tl.neuro.compare` (keyword-only; default `'correlation'`): RDM metric for RSA (`tl.repgeom.rdm`'s names) |
| `read_only=` | constructor keyword of `tl.agent.AgentToolSpec` (default `True`) |
| `read_token_index=` | constructor keyword of `tl.intervention.Direction` (required): Token position(s) read per prompt, or `None` for unsliced outs; keyword of `tl.intervention.fit_direction` (keyword-only; default `-1`): Token position read from every prompt (default `-1`, the last). When a trace saved its `attention_mask` input, positions count within each prompt's real tokens, so `-1` is the last real token of a padded prompt. A sequence gives one position per prompt across all traces; `None` reads the .. |
| `reader=` | keyword of `Bundle.token_pairs` (required): The later member (name or Trace) whose prompt holds the fed text; keyword of `HistoryView.from_reader` (required) |
| `real_lengths=` | keyword of `tl.report.padding_waste` (keyword-only; required): Real (unpadded) token counts per batch row |
| `real_trace=` | keyword of `Trace.discharge_against` (required) |
| `reason=` | keyword of `ValidationStatus.unverified` (keyword-only; required): Stable reason code for the unverified region classification; constructor keyword of `tl.options.EpisodeOptions` (default `None`): Escalation reason, closed vocabulary `{"step_failed", "divergence", "requested"}`; keyword of `tl.preprocessing.unknown_resolution` (required): Human-readable one-line reason nothing resolved; constructor keyword of `tl.recording.StopSignal` (default `''`): Optional user-facing reason stored on the resulting recording; keyword of `tl.recording.stop` (default `''`): Optional reason stored on `Recording.stop_reason` and `Recording.stops_by_pass` |
| `receipt=` | keyword of `Annotation.from_receipt` (required); constructor keyword of `tl.transformer.Annotation` (default `None`): The validating receipt; REQUIRED for `intervention_effect` |
| `receipts=` | keyword of `tl.viz.ablation_grid` (required): One validated receipt per layer row (or a single receipt) |
| `recipes=` | constructor keyword of `tl.types.InputProvenance` (default `None`): Synthesis recipe rows (rungs 2-3): recipe kind, low, high, dtype, seed. `None` on the gold rung |
| `reconstruct_sharding=` | constructor keyword of `tl.io.JaxPayloadLoadHint` (default `False`): Whether to reconstruct a saved `NamedSharding` from portable `codec_metadata` |
| `reconstruction_ready=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `record=` | keyword of `Recording.add_grad_record` (required); keyword of `tl.transformer.token_strip_html`, `tl.viz.token_strip` (required): The token-scores record (single- or multi-row) |
| `recording=` | keyword of `tl.record` (keyword-only; default `None`): Grouped recorder options (`tl.options.RecordingOptions`). The fields `record` reads: `default_op`, `default_module`, `default_grad`, `history_size`, `include_source_events`, `max_predicate_failures`, `on_predicate_error`, `on_forward_error` and `postprocess` (see .. |
| `records=` | constructor keyword of `tl.observers.TapObserver` (optional) |
| `recurrence_detection=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `reduce=` | keyword of `tl.attribution.noise_tunnel` (keyword-only; default `'mean'`): `"mean"`, `"mean_square"`, or population `"variance"`; keyword of `tl.attribution.read` (keyword-only; default `'sum'`): Named scalar reduction (site grain) or `None` (element grain); keyword of `tl.viz.feature_maps` (keyword-only; default `'mean'`): Channel aggregation reduction for `channels=None`: `"mean"` or `"max"` |
| `reducer=` | keyword of `ReadTable.aggregate_targets` (required): Callable folding the per-target score tuple to one float |
| `reference=` | keyword of `tl.changed`, `tl.top_changed`, `tl.viz.intervention_diff` (required); keyword of `tl.intervention.set_direction_mean` (required): Tensor-membered population supplying the coefficient evidence; keyword of `tl.report.diff` (required): The one explicit reference capture |
| `reference_dtype=` | keyword of `tl.debug.bisect_precision` (keyword-only; default `torch.float64`): High-precision reference dtype for the side-by-side run |
| `relations=` | constructor keyword of `tl.Bundle` (keyword-only; default `None`) |
| `relative_to=` | keyword of `Bundle.why` (default `None`): Reference member; defaults to the bundle baseline |
| `relu=` | keyword of `tl.attribution.grad_cam` (keyword-only; default `True`): Whether to apply ReLU to the channel-reduced CAM |
| `renderer=` | keyword of `Trace.draw` (keyword-only; default `'graphviz'`): `"graphviz"` (default) or the experimental `"dagua"` after opt-in; constructor keyword of `tl.options.VisualizationOptions` (optional): Renderer backend selector |
| `repeats=` | keyword of `tl.report.measure_clean_step` (keyword-only; default `5`); keyword of `tl.report.overhead` (keyword-only; default `3`) |
| `replace=` | keyword of `tl.appliances.register` (keyword-only; default `False`): Allow replacing an existing registration under the same name; keyword of `tl.backends.register` (keyword-only; default `False`): Whether to replace an existing spec with the same name or alias; keyword of `tl.export.register_target` (keyword-only; default `False`): Explicit replacement opt-in (collision refusal otherwise); keyword of `tl.io.register_sidecar_family` (keyword-only; default `False`): Explicit replacement opt-in (kernel collision refusal otherwise); keyword of `tl.receptive_field.register_rf_rule` (keyword-only; default `False`): Whether existing registrations may be replaced |
| `replay=` | keyword of `Trace.push` (default `None`): Grouped replay options (`ReplayOptions`: `strict`, `hooks`, `differentiable`); keyword of `Trace.push_from` (default `None`): Grouped replay options (`ReplayOptions`); keyword of `Trace.run` (keyword-only; default `None`) |
| `replayable=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `replicates=` | keyword of `tl.experiment.sweep_grid` (keyword-only; default `False`); keyword of `tl.experiment.sweep_sites` (keyword-only; default `False`): Whether duplicate resolved candidates are deliberate replicates |
| `repo=` | keyword of `tl.bridge.pyreft.read` (keyword-only; default `None`): Optional names of where the directory came from (a Hub repo id and commit); recorded in `provenance` and the subspace `origin` |
| `repo_id=` | keyword of `tl.bridge.huggingface_hub.from_hub` (required): Source Hugging Face repository ID; keyword of `tl.bridge.huggingface_hub.push_to_hub` (required): Target Hugging Face repository ID |
| `request=` | keyword of `tl.agent.build_envelope` (keyword-only; default `None`): Normalized request echo |
| `rerun=` | keyword of `Bundle.token_divergence` (required): The member (name or Trace) from the re-run compared against it, typically its copy from a steered second run (`"cf_critique"`) |
| `reserved=` | keyword of `CaptureBackend.build_record_context` (required) |
| `reserved_block=` | keyword of `CaptureBackend.emit_function_outputs` (required) |
| `reservoir_size=` | constructor keyword of `tl.stats.Quantile` (default `8192`): Maximum sampled values retained in memory (per unit with `unit_axis=`) |
| `reshape=` | keyword of `tl.attribution.grad_cam` (keyword-only; default `None`): Optional callable mapping the layer output to `N, C, H, W`. It must be a pure rearrangement (slicing, transposing, reshaping), because it is applied to the activation and to its gradient alike. Without it, a token-sequence output `N, T, C` is laid on the grid that the trace proves (the .. |
| `resize_size=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Shorter-side int or explicit `(h, w)` resize target |
| `result=` | keyword of `tl.viz.logit_lens` (required): A `LogitLensResult` whose per-layer logits are `[batch, positions, vocab]` |
| `result_byte_budget=` | keyword of `tl.attribution.read` (keyword-only; default `None`): Required for multi-target element-grain reads; refused with an estimate when exceeded |
| `resume=` | keyword of `ExtractionPlan.run` (keyword-only; default `False`): Continue an interrupted artifact (disk mode); keyword of `tl.extract_dataset` (keyword-only; default `False`): Disk mode only (DOCUMENTED-UNSTABLE): continue an interrupted run from its trusted ledger prefix. Every semantic signature field is compared individually; the skipped prefix is replayed at zero forward cost and verified against the ledgered per-shard `input_digest` where recorded |
| `retain=` | keyword of `tl.experiment.sweep_grid` (keyword-only; required); keyword of `tl.experiment.sweep_sites` (keyword-only; required): REQUIRED. `"all"` \| `"none"` \| `top_k(k)` — retention decides what evidence is DESTROYED. Pair any policy with `retain_ceiling_bytes=` for the typed BEFORE-the-first-candidate byte preflight |
| `retain_ceiling_bytes=` | keyword of `tl.experiment.sweep_grid` (keyword-only; default `None`); keyword of `tl.experiment.sweep_sites` (keyword-only; default `None`): Optional byte ceiling for the retention preflight and rolling release (lower-bound basis) |
| `retain_graph=` | keyword of `ReceptiveFieldView.check` (keyword-only; default `False`): Keep the armed autograd graph alive after the check. The default `False` FREES the graph, exactly like `gradient()`'s default: call with `retain_graph=True` (or re-capture) when a later `gradient(..., retain_graph=True)` or second check must run on the same armed capture; keyword of `ReceptiveFieldView.probe_grad` (keyword-only; default `False`): Whether autograd should retain saved graph buffers; keyword of `ReceptiveFieldView.show` (keyword-only; default `False`): Whether the empirical gradient probe retains autograd buffers, so repeated `show(grad=True)` or later `gradient()` calls over the same captured graph stay possible; keyword of `Recorder.backward` (keyword-only; default `None`); keyword of `Recording.backward` (keyword-only; default `None`): Forwarded to `Tensor.backward` |
| `retain_in_memory=` | constructor keyword of `tl.options.StorageOptions` (optional): Whether streamed outs remain in memory; keyword of `tl.to_disk` (keyword-only; default `False`): Whether streamed payloads also remain as RAM copies. The model output the forward returns stays in memory either way (the output op's payload may alias it). A streamed trace does not run: `trace.run(...)` refuses `trace_option_unsupported` (a run would re-capture into RAM while the bundle at .. |
| `retained_bytes=` | field of `MemoryFacts`, `SummaryTotals` |
| `retained_lazy_ops=` | field of `MemoryFacts` |
| `retained_present_ops=` | field of `MemoryFacts` |
| `return_layers=` | keyword of `tl.compat.from_ilg` (default `None`): Optional torchvision `return_layers` mapping |
| `revision=` | keyword of `tl.bridge.huggingface_hub.from_hub` (keyword-only; default `None`): Optional branch, tag or commit to download from; keyword of `tl.bridge.pyreft.read` (keyword-only; default `None`): Optional names of where the directory came from (a Hub repo id and commit); recorded in `provenance` and the subspace `origin` |
| `ridge_alphas=` | keyword of `tl.neuro.compare` (keyword-only; default `...`): Penalty grid (absolute, on centred features) |
| `ridge_intensity=` | keyword of `tl.report.roofline` (keyword-only; default `None`): Machine balance point in FLOPs/byte (from `machine_balance`); without it no bound hypotheses render |
| `right=` | keyword of `MergedTrace.happens_before` (required); keyword of `tl.viz.bundle_diff` (keyword-only; default `None`): Right member name or log. Defaults to the second bundle member |
| `rng=` | constructor keyword of `tl.options.EpisodeOptions` (default `'managed'`): Seeding discipline. Only `"managed"` ships: the capture's effective `random_seed` is drawn-or-passed as today and recorded as the ledger header's `entry_seed` |
| `rng_replay=` | constructor keyword of `tl.backends.BackendCapabilities` (required): Whether operation-level RNG replay is supported |
| `rng_state=` | keyword of `CaptureBackend.restore_rng` (required) |
| `role=` | keyword of `Container.reconstruct_at` (keyword-only; default `None`): Optional boundary role; keyword of `Trace.reconstruct_container` (keyword-only; default `None`): Optional boundary role selector; constructor keyword of `tl.intervention.RegionEdge` (required): `"entry"` (value flows into the region) or `"exit"` (value flows out of it); constructor keyword of `tl.mechinterp.ProjectionSource` (required): `"q"` / `"k"` / `"v"` / `"o"`; keyword of `tl.stats.tensor_stats` (keyword-only; default `None`): Graph-proven semantic role passthrough (`embedding_index` etc.); the kernel never invents one; constructor keyword of `tl.transformer.TokenAxis` (required): `"query"`, `"key"`, or `"position"`; keyword of `tl.transforms.axis_for_role` (required): Semantic role to resolve (e.g. `"token"`) |
| `roles=` | keyword of `tl.transforms.axis_for_role` (required): The declaration attached to the tensor, or `None` when nothing was declared (semantic presets must then refuse, never guess) |
| `row_chunk_size=` | keyword of `tl.repgeom.rdm` (keyword-only; default `None`): Row-block size for memory-bounded computation; each finished block lands on `output_device` before the next is computed |
| `row_ids=` | keyword of `tl.features.as_matrix` (keyword-only; default `None`): Optional per-row identifiers; supplied, their cardinality must equal the stimulus-axis extent or the shaping refuses typed |
| `row_labels=` | keyword of `tl.viz.heatmap` (keyword-only; default `None`): Optional text labels for the row axis (drawn on the left). For an attention picture these are the QUERY tokens (rows attend from) |
| `rows=` | keyword of `tl.report.device_peaks` (required) |
| `rsa_method=` | keyword of `tl.neuro.compare` (keyword-only; default `'spearman'`): `"spearman"`, `"pearson"` or `"kendall"` (tau-a) |
| `rtol=` | keyword of `ReceptiveFieldView.check`, `tl.receptive_field.cross_validate` (keyword-only; default `0.0`): Accepted for compatibility and ignored. Validation always uses exact finite nonzero gradient support; keyword of `ReceptiveFieldView.probe_grad` (keyword-only; default `0.0`): Non-negative gradient support thresholds; keyword of `tl.debug.bisect_precision` (keyword-only; default `None`): Fixed relative tolerance; `None` derives per-op from native dtype; keyword of `tl.debug.check_determinism` (keyword-only; default `None`): Optional relative tolerance; supplying either tolerance changes the claim from exact to NUMERICAL repeatability; keyword of `tl.debug.compare`, `tl.debug.compare_params`, `tl.debug.first_divergence` (keyword-only; default `1e-05`): Relative tolerance for `torch.allclose`; constructor keyword of `tl.receptive_field.GradReceptiveField` (required); keyword of `tl.validate` (keyword-only; default `None`): Backward-only relative tolerance. `None` derives per dtype likewise; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `None`): Relative allclose tolerance. `None` (default) derives per gradient dtype, matching `torchlens.validation.backward` |
| `rules=` | constructor keyword of `tl.Recipe` (default `()`) |
| `run=` | keyword of `HistoryView.from_blocks` (required) |
| `run_ctx=` | keyword of `TapObserver.record_backward` (keyword-only; default `None`): Shared hook run context. Accepted for hook API compatibility |
| `run_id=` | keyword of `tl.export.mlflow` (keyword-only; default `None`): MLflow run to log into. Required with an `MlflowClient`; with the fluent module it is forwarded as `run_id=`; constructor keyword of `tl.trackers.MLflowSink` (keyword-only; default `None`): The run to log into |
| `run_state=` | field of `HookContext` |
| `rung=` | constructor keyword of `tl.types.InputProvenance` (required): The ladder rung, 1-3 |
| `running_stats=` | keyword of `tl.trackers.correct_running_stats`, `tl.trackers.running_stats_scalars` (required) |
| `runs=` | keyword of `tl.debug.check_determinism` (keyword-only; default `2`): Number of same-seed isolated executions (>= 2) |
| `sae=` | constructor keyword of `tl.attribution.SiteTarget` (keyword-only; default `None`): An SAE (any object with `encode` and `decode`) trained on the layer's output. Given, the site is the SAE's latents, `sae.encode(output)`, and `unit` selects among them: `SiteTarget("fc1", sae=sae, unit=3442)` attributes latent 3442. DOCUMENTED-UNSTABLE; keyword of `tl.bridge.sae.register` (required): Duck-typed SAE exposing `encode` and `decode`; keyword of `tl.bridge.sae.splice` (keyword-only; required): Duck-typed SAE exposing `encode` and `decode`; keyword of `tl.bridge.sae_lens.encode` (required): SAE object exposing `encode` or a callable SAE; keyword of `tl.bridge.sae_lens.reconstruct` (required): SAE object exposing `encode` and `decode`, or a callable SAE |
| `safety_margin=` | keyword of `tl.neuro.plan_extraction` (keyword-only; default `0.15`): Fraction of the budget held back (default 0.15) |
| `salience=` | keyword of `tl.bridge.lit.model` (keyword-only; default `False`): Opt-in model-provided `TokenSalience` from the torchlens attribution kit (`autorun=False` is fixed cost semantics) |
| `samples=` | keyword of `tl.dead`, `tl.low_variance`, `tl.saturated` (required) |
| `save=` | keyword of `Trace.run` (keyword-only; default `None`); keyword of `tl.record` (keyword-only; default `None`): The one predicate spelling (matches `trace(save=...)`); constructor keyword of `tl.recording.Recorder` (keyword-only; optional); keyword of `tl.recording.dry_run` (keyword-only; default `None`): Fastlog dry-run options. Predicate exceptions propagate immediately; keyword of `tl.repgeom.mds_evolution`, `tl.repgeom.rdm_evolution`, `tl.repgeom.scree_evolution` (default `None`): Optional selector limiting which layers or pass-qualified ops to process. When omitted, all layers with saved activations are used; keyword of `tl.trace` (keyword-only; optional): Which layers to save outs for; the canonical selection kwarg (see **Layer selection** above). Accepts `'all'`, `'none'`/`None`/`[]`, label/module/index/substring lists, predicate selectors (`tl.func`, `tl.in_module`, `tl.followed_by`, `tl.when` combinators) or `"stats"` (per-op ..; keyword of `tl.viz.feature_maps` (default `None`): Optional selector limiting which layers or pass-qualified ops to process |
| `save_args=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `save_budget=` | constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.validate` (keyword-only; default `'auto'`): Per-device ceiling on the validator capture's retained bytes, with the same values and default as `CaptureOptions.save_budget` (`"auto"`, a float fraction in `(0, 1]`, an int byte cap, or `None` to disable). Exceeding it raises `SaveBudgetExceededError` exactly as ..; keyword of `tl.validation.validate_forward_pass` (keyword-only; default `'auto'`): Per-device retained-bytes ceiling for the validator capture (same values and default as `CaptureOptions.save_budget`); parameters kept by reference are not charged |
| `save_grad=` | constructor keyword of `tl.recording.CaptureSpec` (default `False`): Whether the in-RAM tensor clone should stay attached to autograd |
| `save_grads=` | keyword of `Recorder.backward` (keyword-only; default `None`); keyword of `Recording.backward` (keyword-only; default `None`): Optional per-gradient predicate overriding the recording default; constructor keyword of `tl.options.CaptureOptions` (optional) |
| `save_level=` | keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `'portable'`): Public `.tlspec` save level of the uploaded bundle (`"audit"`, `"executable_with_callables"`, or `"portable"`) |
| `save_levels=` | constructor keyword of `tl.backends.BackendCapabilities` (default `('audit', 'executable_with_callables', 'portable')`): Supported portable save levels |
| `save_metadata=` | constructor keyword of `tl.recording.CaptureSpec` (default `True`): Whether non-payload metadata should be retained for this event |
| `save_mode=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional); constructor keyword of `tl.recording.CaptureSpec` (default `'copy'`): Tensor retention mode for saved payloads |
| `save_out=` | constructor keyword of `tl.recording.CaptureSpec` (default `True`): Whether tensor payloads should be retained for this event |
| `save_raw_activations=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `save_raw_grads=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `save_raw_input=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `save_rng=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `save_source_context=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `save_transformed_output=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `save_visualizations=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `scale=` | keyword of `tl.mechinterp.apply_norm_scale` (keyword-only; default `'frozen'`): `"frozen"` (default; TLens's own default, verified from source) or `"component"` -- the per-component RECOMPUTE mode, which is NOT additive: the row-sum identity is skipped, the receipt says so, and DLA refuses such stacks. TLens `apply_ln_to_stack`; keyword of `tl.trackers.correct_histogram`, `tl.trackers.correct_running_stats`, `tl.trackers.corrected_grad_block` (required) |
| `scaler=` | keyword of `WatchSession.step` (keyword-only; default `None`); constructor keyword of `tl.debug.ChecksSession` (keyword-only; default `None`): The loop's `GradScaler` handle, or `None` as the DECLARATION that this loop runs no scaler; keyword of `tl.trackers.unscale_stage` (required) |
| `schema=` | keyword of `tl.agent.build_envelope` (keyword-only; required): Versioned payload schema id (`torchlens.agent.<name>.v1`) |
| `scope=` | keyword of `SummaryReport.to_pandas` (default `'display'`); keyword of `tl.validate` (keyword-only; required): Validation scope: `"forward"`, `"backward"`, `"saved"`, `"intervention"`, or `"receptive_field"`. `"receptive_field"` captures with `save_mode="reference"`, so a model whose forward writes its own buffers in place (a train-mode BatchNorm's running statistics) raises .. |
| `score=` | keyword of `tl.attribution.occlusion` (keyword-only; required): Callable receiving a Trace and returning one finite real scalar. It is invoked once on the original and once on the occluded fork; keyword of `tl.bridge.brain_score.score_parts` (required): A Brain-Score `Score` (any object with `values` and `attrs`) or a plain number |
| `scores=` | keyword of `tl.repgeom.metric_disagreement` (required): `scores[metric][site]`: a float or a `ScoreInterval` per site. Sites missing under a metric count as NaN there; keyword of `tl.viz.causal_trace_heatmap` (required): 2D array-like patching scores |
| `sdpa_backend=` | keyword of `tl.report.flops_vs_dispatch` (keyword-only; default `None`): Provenance disclosure of the SELECTED SDPA backend when the caller forced one (e.g. `"math"` under `sdpa_kernel(SDPBackend.MATH)`, which changes the EXECUTED PROGRAM: formula validation only, never a timing comparison). `None` means default selection |
| `second_run=` | keyword of `Bundle.add_alternative` (required): The bundle the same harness built with the `steered` calls captured under their specs |
| `sections=` | keyword of `FacetSpec.split` (required) |
| `seed=` | keyword of `CaptureBackend.seed_rng` (required); keyword of `Trace.run` (keyword-only; default `None`): Optional deterministic live refresh, random-state, and runtime RNG seed; keyword of `tl.add_noise` (keyword-only; default `None`): Optional hook-local seed. Seeded helpers use a private generator and do not consume global RNG; keyword of `tl.attribution.causal_audit` (keyword-only; default `0`): Seed of the random draws (a local generator; the global random state is restored); keyword of `tl.attribution.gradient_shap` (keyword-only; default `None`): Optional deterministic seed for alphas, pool draws, and noise; keyword of `tl.attribution.infidelity` (keyword-only; default `None`): Optional deterministic seed; keyword of `tl.attribution.noise_tunnel` (keyword-only; default `None`): Optional deterministic seed (per-device generators, seeded on first use). For a stochastic child with no explicit child seed, independent per-sample child substreams are derived from this seed and disclosed; keyword of `tl.attribution.randomization_test` (keyword-only; default `0`): Seed of the re-initialization draws (default 0), drawn inside a forked random state that is restored afterwards; keyword of `tl.attribution.road_curve` (keyword-only; default `0`): Seed of the noise draws (a local generator per step); keyword of `tl.attribution.sensitivity` (keyword-only; default `None`): Optional deterministic seed for the perturbation draws; keyword of `tl.attribution.smoothgrad` (keyword-only; default `None`): Optional random seed for deterministic noise samples without mutating global torch RNG state; keyword of `tl.debug.bisect_precision` (keyword-only; default `0`): Seed applied to both runs inside the RNG fork; keyword of `tl.debug.check_determinism` (keyword-only; default `0`): Seed for every compared run; keyword of `tl.debug.guess_input_shape` (keyword-only; default `0`): RNG seed used for deterministic probes; keyword of `tl.debug.isolated_capture` (keyword-only; required): Seed applied to Python, NumPy, and torch generators inside the preserved-RNG scope before the forward runs. It is also routed to `CaptureOptions(seed=...)` (unless the caller's `capture` options carry an explicit `random_seed`), because unseeded captures pick a FRESH private seed per capture ..; constructor keyword of `tl.intervention.SamplingPlan` (required); keyword of `tl.intervention.permute_batch`, `tl.random_like` (keyword-only; required); keyword of `tl.intervention.resample_rows_from`, `tl.intervention.sample_from` (keyword-only; required): REQUIRED int or `"auto"`; keyword of `tl.intervention.scramble_elements` (keyword-only; default `None`): Optional hook-local RNG seed; keyword of `tl.intervention.subspace_ablate` (keyword-only; default `None`): Permutation seed for `mode="resample"` (required: a resample edit must be reproducible); keyword of `tl.neuro.compare` (keyword-only; default `0`): Seed for the fold split and the bootstrap; constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.recording.dry_run` (keyword-only; default `None`): Fastlog dry-run options. Predicate exceptions propagate immediately; keyword of `tl.repgeom.cka_interval` (keyword-only; default `0`): Bootstrap settings (see `rdm_compare_interval`); keyword of `tl.repgeom.compare_rdms_interval` (keyword-only; default `0`): Seed for `numpy.random.default_rng`; the same seed gives the same interval; constructor keyword of `tl.stats.Quantile` (default `0`): Seed for the private sampling stream. The default `0` makes estimates reproducible run-to-run; pass `None` for a fresh-entropy stream (still decoupled from the global engine); keyword of `tl.transforms.srp` (default `None`): Base seed. Omitting it records `seed=0` with `seed_source="library_default"`; passing it (even `0`) records `"explicit"`. NOTE: under `share="by_extent"` five runs with the SAME seed share ONE projection — independence needs DISTINCT seeds under either policy; keyword of `tl.transforms.srp_fidelity_probe` (default `0`): Base seed for the probe projections; keyword of `tl.validate` (keyword-only; default `None`): Optional random seed for forward-like validation. Like `tl.trace`, the call restores the caller's global RNG streams on return and on error; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `None`): Fixed RNG seed for stock and candidate passes. Like `tl.trace`, the call seeds the process-global RNG engines for its runs and restores the caller's streams on return and on error; keyword of `tl.validation.validate_forward_pass` (default `None`): Fixed RNG seed; like `tl.trace`, the caller's streams are restored on return; constructor keyword of `tl.intervention.SamplingPlan` (required) |
| `seed_sensitivity=` | keyword of `tl.debug.check_determinism` (keyword-only; default `False`): Opt into the SECOND named verdict from one extra differently-seeded run |
| `segments=` | keyword of `ReceptiveFieldRuleContext.piecewise` (required) |
| `select=` | keyword of `PartialTrace.narrate`, `Recording.narrate`, `Trace.narrate` (keyword-only; default `None`): Optional filter: a substring, or a callable over `NarrationEvent` rows; constructor keyword of `tl.trackers.HFTrainerWatchCallback`, `tl.trackers.LightningWatchCallback` (keyword-only; default `None`); keyword of `tl.trackers.watch` (keyword-only; default `None`): Site selection (module path prefixes / parameter names). REQUIRED for activations; optional narrowing for parameter streams. At `grain="op"` strings are module path prefixes (OR-ed `tl.in_module`) and any selector-algebra object (`tl.func(...)`, composed predicates) passes through as the .. |
| `selection=` | keyword of `Trace.subgraph` (required): A selection-shaped ACT region (Selection, ResolvedSelection, Op/Layer, receptive-field region, or site label string). PARAM/EDGE selections refuse `selection_kind_incompatible`; keyword of `tl.attribution.occlusion` (required): Selection query, resolved selection, or region producer to occlude |
| `selector=` | keyword of `Super.from_members` (required): Original user query; keyword of `Trace.annotate`, `Trace.find_sites` (required): Selector, target spec, frozen target spec, or non-strict bare string; keyword of `Trace.stack` (required): TorchLens selector accepted by `find_sites`; keyword of `tl.neighborhood` (required) |
| `seq_len=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `None`): Optional sequence length override for token or recurrent inputs |
| `serialization_policy=` | constructor keyword of `tl.backends.BackendSpec` (optional): Backend-owned serialization policy |
| `series=` | keyword of `tl.viz.lineplot` (required): One-dimensional `[K]` or two-dimensional `[S, K]` numeric values |
| `serve=` | keyword of `tl.export.netron`: open the exported graph in a local viewer |
| `session=` | keyword of `CaptureBackend.active_logging`, `CaptureBackend.apply_live_hooks`, `CaptureBackend.build_record_context`, `CaptureBackend.cleanup_forward_memory`, `CaptureBackend.cleanup_model_session`, `CaptureBackend.detect_backend_semantics`, `CaptureBackend.emit_function_outputs`, `CaptureBackend.inference_context`, `CaptureBackend.pause_logging`, `CaptureBackend.prepare_model`, `CaptureBackend.prepare_model_session`, `CaptureBackend.restore_rng`, `CaptureBackend.safe_copy`, `CaptureBackend.seed_rng`, `CaptureBackend.set_capture_producer_policy`, `CaptureBackend.set_tensor_label`, `CaptureBackend.setup_inputs_and_device`, `CaptureBackend.snapshot_rng`, `CaptureBackend.tensor_ref` (required); keyword of `CaptureBackend.cleanup_failed_forward_session`, `CaptureBackend.cleanup_stopped_forward_session`, `CaptureBackend.extract_and_mark_outputs`, `CaptureBackend.finalize_forward_session`, `CaptureBackend.log_source_tensor`, `CaptureBackend.pop_module_frame`, `CaptureBackend.push_existing_module_frame` (required): Active backend capture session; keyword of `Recording.from_capture_events` (required): Trace-like session exposing `capture_events` and `_fastlog_recording` metadata |
| `settings=` | constructor keyword of `tl.attribution.AttributionMetric` (required): The RESOLVED evaluation policy: every value that must match for a cross-result comparison to mean anything; keyword of `tl.trackers.watch` (keyword-only; default `None`): Advanced collector plumbing (ring capacity, output_dir, ...) |
| `shape=` | constructor keyword of `tl.transforms.TensorSpec` (required): Full shape INCLUDING the leading stimulus axis; `None` marks an unknown extent |
| `shard_format=` | keyword of `tl.extract_dataset` (keyword-only; default `None`): `None` (default: safetensors on fresh runs, the recorded format on resume) \| `"safetensors"` \| `"pt"` |
| `sharding=` | constructor keyword of `tl.io.JaxPayloadLoadHint` (default `None`): Explicit JAX `Sharding` target. When supplied, materialized JAX payloads are placed with this sharding and reconstruction metadata is not consulted |
| `share=` | keyword of `tl.transforms.srp` (default `'by_extent'`): `"by_extent"` (default; equal-width sites share one matrix) or `"by_site"` (adds the stable site identity to the key) |
| `share_draw=` | constructor keyword of `tl.intervention.SamplingPlan` (required); keyword of `tl.intervention.sample_from` (keyword-only; default `'per_firing'`): Recorded draw-sharing policy: `"per_firing"` (default), `"per_rule"`, or `"per_group"` |
| `share_safe=` | constructor keyword of `tl.options.HtmlReportOptions` (default `False`) |
| `show_axes=` | keyword of `tl.viz.image_scatter` (keyword-only; default `True`): Whether to draw faint central guide axes |
| `show_axis_thumbnails=` | keyword of `tl.viz.nodes.rdm` (keyword-only; default `True`): Whether to use matching raw PIL image stimuli as symmetric axis thumbnails when available |
| `show_buffers=` | keyword of `Trace.draw` (keyword-only; default `'meaningful'`): Buffer visibility: `"never"`, `"meaningful"` or `"always"`; keyword of `Trace.to_dagua_graph` (default `False`): Forwarded unchanged to `trace_to_dagua_graph`; keyword of `tl.experimental.dagua.trace_to_dagua_graph` (default `False`); keyword of `tl.export.netron` (keyword-only; default `'meaningful'`): The existing tri-state buffer policy (`never\|meaningful\|always`); `meaningful` also drops counter-update chains and disclosed hidden counts ride the owning function docString and the model metadata; constructor keyword of `tl.options.VisualizationOptions` (optional): Buffer visibility policy |
| `show_call_index=` | keyword of `Module.show_call_tree`, `ModuleCall.show_call_tree`, `Trace.show_call_tree` (default `True`): Whether to include the `:N` call suffix in labels |
| `show_cone=` | keyword of `Trace.draw` (keyword-only; default `False`): Shade the downstream cone of every intervention; constructor keyword of `tl.options.VisualizationOptions` (optional): Whether the nodes and arrows an intervention changed downstream get the translucent purple highlighter (off by default; the edited node is always marked) |
| `show_containers=` | keyword of `Trace.draw` (keyword-only; default `False`): Container rendering: `False` (default: the plain graph), `"labels"` (the key or index on each edge into a container leaf), `"collapsed"` (`"labels"` plus the fold of an unrolled same-shape container over `container_max_inline` leaves into one dashed box) or `"nodes"` (`"collapsed"` .. |
| `show_grad_values=` | keyword of `Trace.draw` (keyword-only; default `False`): Backward graph: opt-in rows naming the owning module of each backward node and summarizing its captured output gradient |
| `show_input_transform_summary=` | keyword of `Trace.draw` (keyword-only; default `False`): Add the input-transform summary row |
| `show_legend=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `Trace.draw` (keyword-only; default `None`): Tri-state legend visibility. `None` (default) draws a legend only when an encoding channel is active, and the key with a reading key on a `module=` focused draw; `True` draws the full theme legend, `False` disables the legend even with channels active; constructor keyword of `tl.options.VisualizationOptions` (optional): Tri-state legend visibility (`None` = auto: legend only when an encoding channel is active, plus the key on a `module=` focused draw); see `Trace.draw`; keyword of `tl.viz.lineplot` (keyword-only; default `True`): Whether to draw a legend when labels are supplied |
| `show_module_paths=` | keyword of `Trace.draw` (keyword-only; default `False`): Backward graph: opt-in rows naming the owning module of each backward node and summarizing its captured output gradient |
| `show_orphans=` | keyword of `Trace.draw` (keyword-only; default `False`): Render orphan (island) ops, captured but unreachable from inputs and outputs, as a dashed greyed cluster (capture with `keep_orphans=True`) |
| `show_redundant_args=` | keyword of `Trace.draw` (keyword-only; default `False`): Show constructor-arg label rows that provably duplicate a captured shape dimension (suppressed by default; the rule only ever reveals more); constructor keyword of `tl.options.VisualizationOptions` (optional): UNSTABLE: show constructor args the checked-suppression equality check proved redundant (default `False`); see `Trace.draw` |
| `show_saved_for_backward=` | keyword of `Trace.draw` (keyword-only; default `False`): Add a `saved for backward: N tensors, X MB` row on every op whose grad_fn measurably retained tensors |
| `signal=` | keyword of `tl.observers.span` (`"forward"`, `"backward"` or `"both"`); the removed `show_bundle_graph` keyword is `Bundle.draw(graph=...)`; constructor keyword of `tl.Selector` (keyword-only; default `'forward'`); constructor keyword of `tl.observers.TapObserver` (default `'both'`): Signal on which this tap records; `"both"` records every fire; keyword of `tl.observers.tap` (keyword-only; default `'both'`): Pass filter on top of the site: `"forward"` records forward fires only, `"backward"` backward fires only, `"both"` (default) every pass the site fires on |
| `signals=` | constructor keyword of `tl.trackers.HFTrainerWatchCallback`, `tl.trackers.LightningWatchCallback` (keyword-only; default `None`); keyword of `tl.trackers.watch` (keyword-only; default `('gradients',)`): Subset of `SIGNALS`. Default watches parameter gradients |
| `signed=` | constructor keyword of `tl.stats.HistogramDescriptor` (default `True`) |
| `signed_grad=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (keyword-only; default `None`) |
| `signs=` | keyword of `tl.viz.causal_trace_heatmap` (keyword-only; default `'all'`): Captum-style sign selector: `"positive"`, `"negative"`, `"absolute_value"`, or `"all"` |
| `sink=` | constructor keyword of `tl.options.NarrationOptions` (optional): Narration destination: `None` (stderr; stdout stays clean), a path (per-line flush -- bytes on disk survive a hard process death), an open text stream, or `callable(str)` (the sole seam to trackers/loggers); constructor keyword of `tl.stats.NarrationObserver` (default `None`); keyword of `tl.stats.narrate` (keyword-only; default `None`): Optional per-line consumer (e.g. `print` or a logger method); lines always also accumulate on `observer.lines`; keyword of `tl.trackers.require_capabilities` (required) |
| `sinks=` | keyword of `Trace.between` (required): Same forms as `sources`; keyword of `tl.between` (required); constructor keyword of `tl.trackers.WatchSession` (required) |
| `site=` | keyword of `Bundle.at` (required): Label from any Bundle accessor family, or a selector-like site; keyword of `CaptureBackend.apply_live_hooks` (required); keyword of `Container.reconstruct_at` (keyword-only; default `None`): Optional boundary site. Site aliases are considered matches; keyword of `LoreftArtifact.subspace` (required): The TorchLens site of the value pyreft edits (for `block_output` of layer `n`, the output of decoder layer `n`): its label, or the site itself from a trace (`trace[label]`). Given the site, its shape is checked here: the trailing axis must be the artifact's feature axis, `d` wide, and ..; keyword of `Trace.animate` (required): A site label, pass-qualified label or record (anything with a `layer_label`). The picture is the HTML animation of the repeated ops at that site: `str(drawing)` is the HTML, `drawing.show()` plays it and `drawing.save(path)` writes it; keyword of `Trace.clear_edits` (default `None`): Optional selector-like target. When provided, all sticky hooks for that target are removed; keyword of `Trace.push_from` (required): Layer pass or selector resolving to one origin. The origin's current out is preserved and used as the override; keyword of `Trace.reconstruct_container` (keyword-only; default `None`): Optional boundary site selector; keyword of `tl.attribution.seed` (required): Site address (pass-qualified label, Op, or Layer); keyword of `tl.bridge.captum.layer` (required): Module address, module pass label, layer selector, or layer object; keyword of `tl.bridge.pytorch_grad_cam.cam` (required): Module address, module pass label, op label, layer selector, or layer object; resolved by `layer`; keyword of `tl.bridge.pytorch_grad_cam.layer` (required): Site or module lookup; keyword of `tl.bridge.rsatoolbox.dataset` (default `None`): Site selector: Trace lookup (qualified module address, pass-qualified label) or extraction output key. `None` selects the final tensor output (the historical behavior; Trace route only); keyword of `tl.bridge.sae.register` (keyword-only; required): Where the SAE reads: a layer label or module address (must name one op of each trace), or a selector (every match is a home); keyword of `tl.bridge.sae.splice` (keyword-only; required): Layer label, selector, or layer object naming the splice site; keyword of `tl.bridge.sae.write_delphi` (keyword-only; required): The site the latents were read at; it names the cache folder and is the module name delphi lists (`modules=[site]`); keyword of `tl.bridge.sae_lens.encode` (required): Layer label, selector, or layer object to encode; keyword of `tl.bridge.sae_lens.reconstruct` (required): Layer label, selector, or layer object to reconstruct; keyword of `tl.bridge.xarray.data_array` (required): Site selector (qualified module address, pass-qualified label) or extraction output key; keyword of `tl.bridge.xarray.neuroid_assembly` (required): Site selector or extraction output key; keyword of `tl.experimental.stop_after` (required): Module-address or function-name string, live selector (`tl.func`/`tl.module`), or callable halt predicate; keyword of `tl.features.site_matrix` (required): Site selector: a Trace lookup (module address, label) or an extraction output key; constructor keyword of `tl.intervention.Direction` (required): Label of the site the rows were read at; keyword of `tl.intervention.fit_direction` (required): Site label (or a layer record, whose label is resolved in every trace) whose saved out holds `[n_prompts, n_tokens, hidden]` activations, or `[n_prompts, hidden]` with `read_token_index=None`; keyword of `tl.mechinterp.patch_query_heads_grid` (keyword-only; default `'v'`): `"v"` (default) patches the expanded value copy, `"k"` the expanded key copy; constructor keyword of `tl.observers.TapObserver` (required): Selector-like site where this tap should be registered; keyword of `tl.observers.tap` (required): Selector-like site to observe. `tl.grad(site)` observes the gradient of the site's output on each backward pass; `tl.grad(site) \| site` observes both passes; constructor keyword of `tl.stats.NarrationObserver` (default `None`); keyword of `tl.stats.narrate` (required): Selector-like site to observe (any `tap()`-accepted selector); `tl.grad(site)` narrates the gradient of the site's output; keyword of `tl.sweep` (optional): Intervention site target. Strings match either an exact TorchLens label or a function name; selectors and predicate callables are used directly; keyword of `tl.trackers.assert_tag_safe` (keyword-only; default `None`); keyword of `tl.units` (required): Site spelling: a pass-qualified op label (`'relu_1_2:1'`), a bare layer label (single-pass layers only — a bare label naming a multi-pass layer refuses `selection_unresolvable` / `multipass_bare_label`), or an input io role |
| `site_id=` | keyword of `HistoryReader.observations` (keyword-only; default `None`); keyword of `HistoryView.series`, `HistoryView.site_streams`, `tl.viz.histogram_waterfall`, `tl.viz.quantile_fan`, `tl.viz.site_detail` (required) |
| `site_key=` | constructor keyword of `tl.mechinterp.Coordinate` (default `None`): Live structural site key for the op (recapture-stable identity) |
| `sites=` | keyword of `HistoryView.from_blocks`, `tl.trackers.architecture_fingerprint`, `tl.trackers.build_manifest`, `tl.trackers.emission_from_block` (required); keyword of `tl.attribution.deconvolution` (keyword-only; default `'all'`): `"all"` (op-level coverage, default) or `"module"` (`nn.ReLU`-dispatched firings only); keyword of `tl.attribution.guided_backprop` (keyword-only; default `'all'`): `"all"` (op-level coverage, default) or `"module"` (`nn.ReLU`-dispatched firings only -- the module-hook coverage class for reference-implementation parity); keyword of `tl.bridge.brain_score.per_layer` (keyword-only; default `None`): Optional iterable of sites to score: layer labels, selectors, or module dotted paths (the spelling Brain-Score users already write). Defaults to every STIMULUS-INDEXED saved tensor layer except input placeholders, through the core eligibility gate -- a partially saved trace scores its saved sites ..; keyword of `tl.bridge.lit.model` (keyword-only; default `None`): Which activations to expose as LIT `Embeddings` fields: the gate-passed `"blocks"` preset, explicit module addresses (optionally pass-qualified `"address:N"`), or `` for none. Required -- the bridge refuses rather than guesses ([UI-SPRINT] item 4); constructor keyword of `tl.debug.LivenessProbe`, `tl.mechinterp.SavePlan` (required); keyword of `tl.mechinterp.patch_grid` (keyword-only; default `'resid_pre'`): Facet to patch (`resid_pre` / `attn_out` / `z` / ...); keyword of `tl.neuro.compare` (keyword-only; default `None`): Sites to score (default: every stimulus-indexed saved site, or every mapping key); keyword of `tl.neuro.plan_extraction` (keyword-only; default `'all'`): `"all"` (every saved-activation site on the observed path) or an iterable of layer labels to include; constructor keyword of `tl.options.NarrationOptions` (optional): Narration scope: `True` (every completed tensor-output event plus module structure lines), `"modules"` (structure lines only -- a cheap flight recorder), or any LIVE-evaluable selector/callable (`tl.func`, `tl.in_module`, boolean combinations, bounded `followed_by`). Selectors needing .. |
| `size_by=` | keyword of `Trace.draw` (keyword-only; default `None`): Size encoding channel (keyword-only; unstable): a Layer/Op field name, the `"dims"` shape token, or a callable `node -> scalar`. Encoded nodes get width/height minimums (labels are never truncated), clamped to 4x the default area. A rolled multi-pass layer whose source cannot be certified ..; constructor keyword of `tl.options.VisualizationOptions` (optional): UNSTABLE size-channel value source (field, `"dims"`, or callable); see `Trace.draw` |
| `sketch=` | keyword of `tl.trackers.correct_histogram`, `tl.trackers.histogram_points` (required) |
| `skip_fn=` | keyword of `Trace.draw` (keyword-only; default `None`): Per-node styling and selection callbacks (see `tl.viz.nodes`); constructor keyword of `tl.options.VisualizationOptions` (optional): Optional layer skip predicate |
| `slice_cols=` | constructor keyword of `tl.mechinterp.ProjectionSource` (required): `(start, stop)` column slice into the math-oriented fused weight, `None` for unfused roles |
| `slot=` | keyword of `tl.site` (keyword-only; default `None`): Multi-output slot index (`None` = single-output ops) |
| `slug=` | keyword of `AttributionGraph.to_neuronpedia` (keyword-only; required): The graph's identifier on Neuronpedia, for example `"capital-of-texas"` |
| `snapshot_kind=` | keyword of `ChecksSession.register_frozen_check` (keyword-only; default `'clone'`): `'clone'` keeps exact detached dtype-preserving snapshots and can show the offending delta; `'digest'` is the labeled content-digest opt-in whose findings disclose probabilistic evidence |
| `sort_by=` | keyword of `Trace.profile` (keyword-only; default `'time'`): Resource column used for sorting. Time is descending by default; keyword of `Trace.projective_fields` (keyword-only; default `None`); keyword of `Trace.receptive_fields` (keyword-only; default `None`): Optional table column used for stable sorting; keyword of `tl.report.profile` (keyword-only; default `'time'`): Column used for sorting. `"time"` sorts descending by default |
| `source=` | keyword of `Bundle.token_pairs` (required): The earlier member (name or Trace) that produced that text; keyword of `CaptureBackend.log_source_tensor` (required): Source role, such as `"input"` or `"buffer"`; keyword of `ReceptiveFieldView.at` (keyword-only; default `None`): Optional ancestor graph point. When given, the returned box is in the SOURCE operation's output-grid coordinate space (not this operation's); keyword of `ReceptiveFieldView.check` (keyword-only; default `None`): Optional ancestor graph point for layer-to-layer receptive validation; keyword of `ReceptiveFieldView.probe_grad` (keyword-only; default `None`): Reserved ancestor graph point for layer-to-layer receptive probes; keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unverified` (keyword-only; required): Trace source; keyword of `tl.bridge.rsatoolbox.dataset` (required): A TorchLens `Trace` with saved activations (in-memory route), a `LoadedExtraction`, or an extraction-artifact directory path (file route); keyword of `tl.bridge.xarray.data_array`, `tl.bridge.xarray.neuroid_assembly` (required): A TorchLens `Trace` with saved activations, a `LoadedExtraction`, or an extraction-artifact directory path; constructor keyword of `tl.receptive_field.GridLayout` (required); keyword of `tl.export.model_explorer_serve` (required): Log or exported payload path to serve; constructor keyword of `tl.facets.FacetFunction` (default `'user'`); keyword of `tl.features.site_matrix` (required): A TorchLens `Trace` (in-memory route), a `LoadedExtraction`, or an extraction-artifact directory path (file route); keyword of `tl.intervention.population` (required): The population payload: a tensor (members = axis-0 slices, the noun's documented stacking contract), a sequence of same-shape tensors, or a sequence of captured Traces (donor values resolve per-site at fire time, pass-qualified); keyword of `tl.intervention.resample_rows_from` (required): The population noun; members are DONOR ROWS (build it from a stacked tensor -- members are axis-0 slices by the noun's documented contract); keyword of `tl.intervention.scramble_elements` (default `None`): Source tensor or Op-like object; keyword of `tl.intervention.subspace_interchange` (required): Source trace supplying the donor value at each edited site; keyword of `tl.mean_ablate` (default `None`): What is averaged. `None` averages the site's own value at hook fire time (a value that couples every batch row, so the spec derives `is_batch_independent=False` and the append gate refuses). A tensor or an Op-like object exposing `out` is averaged instead of the site, and never reads the ..; keyword of `tl.neuro.compare` (required): A Trace with saved activations, a `LoadedExtraction` or extraction directory, or a mapping `{site: [n_stimuli, ...]}`; keyword of `tl.neuro.export_npz` (required): In-memory extraction mapping (`{site_label: tensor}`) as returned by the in-memory runner, or an extraction artifact directory; keyword of `tl.patch_from` (required): Source trace (or a `SamplingPlan`, which takes no mapping); keyword of `tl.receptive_field.cross_validate` (keyword-only; default `None`): Optional receptive-direction ancestor selector or selectors; constructor keyword of `tl.report.PeakRow` (default `'user_supplied'`); keyword of `tl.trackers.observed_grad_scale` (required): A `torch.amp.GradScaler` (or duck-typed equivalent exposing `is_enabled()` / `get_scale()`), or `None`; constructor keyword of `tl.transformer.Annotation` (required): Producer wording rendered in the legend; keyword of `tl.transformer.score_decomposition` (keyword-only; required): Key (column) position |
| `source_2d=` | keyword of `tl.repgeom.procrustes_align` (required): Source coordinates with shape `[N, 2]` |
| `source_citation=` | constructor keyword of `tl.report.PeakRow` (required) |
| `source_context_lines=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `source_date=` | constructor keyword of `tl.report.PeakRow` (required) |
| `source_site=` | keyword of `tl.patch_from` (keyword-only; default `None`): Donor site on the source trace; `None` reads the edited site |
| `sources=` | keyword of `Trace.between` (required): One region or a list of regions: site label strings, `Op`/`Layer` handles, or any ACT selection; keyword of `tl.between` (required) |
| `sparse_policy=` | constructor keyword of `tl.report.PeakRow` (default `'dense_only'`) |
| `spatial_rank=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `'auto'`): Spatial rank override, or `"auto"` from the first convolution |
| `spatial_support_mask=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `spec=` | keyword of `tl.backends.register` (required): Backend spec to register; keyword of `tl.backends.require_capability_implementation` (required): Backend spec whose gate is being opened; keyword of `tl.experimental.node_styles.attention_node_mode`, `tl.experimental.node_styles.vision_node_mode` (required): Current node spec; constructor keyword of `tl.facets.Facet` (required): Facet specification to evaluate lazily; keyword of `tl.intervention.check_spec_compat` (required): Loaded or in-memory intervention spec; keyword of `tl.transforms.srp_realized_facts`, `tl.transforms.srp_verify_matrix` (required): The srp spec |
| `specified_fields=` | keyword of `NarrationOptions.from_values`, `ReplayOptions.from_values`, `StorageOptions.from_values`, `VisualizationOptions.from_values` (required) |
| `square=` | keyword of `tl.debug.guess_input_shape` (keyword-only; default `True`): Whether spatial search should use equal side lengths. Non-square exact inference is intentionally not attempted without a future aspect-ratio hint |
| `stack_by=` | keyword of `Trace.draw` (keyword-only; default `None`): Rank encoding channel (keyword-only; unstable): nodes sharing an annotation value pin to one Graphviz rank, the unrolled-RNN timestep diagram. `True` derives the pass index (unrolled multi-pass ops only); rolled graphs refuse (`stack_by_requires_unrolled`); constructor keyword of `tl.options.VisualizationOptions` (optional): UNSTABLE rank-channel annotation source (`True`/`"auto"`, field, or callable); see `Trace.draw` |
| `start_axis=` | keyword of `tl.transforms.flatten` (default `1`): First axis to fold; must be `>= 1` (the stimulus axis never folds) |
| `state=` | keyword of `HFTrainerWatchCallback.on_train_begin`, `HFTrainerWatchCallback.on_train_end` (required); constructor keyword of `tl.options.EpisodeOptions` (default `()`): Declared episode-carried state items beyond the built-in scope (token prefix, KV cache, RNG streams). EVERY declared item is preflighted at DECLARATION time, unconditionally: an item without snapshot/restore support refuses typed (`episode_state_unsnapshotable`) before execution (E-A4) |
| `state_complete=` | keyword of `tl.types.register_container` (keyword-only; default `False`): Trusted declaration that the hook pair round-trips ALL per-instance state; without it, runnable saves refuse instances carrying extra `__dict__` state instead of dropping it silently on replay |
| `state_dict=` | keyword of `Trace.load_state_dict`: the sparse-run state to stage |
| `statistic=` | keyword of `SuperOp.aggregate` (default `'mean'`): Reduction to apply; keyword of `TagGrammar.data`, `TrajectoryStats.compute_stats` (required) |
| `statistics=` | keyword of `tl.mechinterp.measure_trajectory` (keyword-only; default `None`): Names to measure among `"entropy"`, `"kl_to_final"` (KL from the final output distribution to the layer's), `"max_prob"`, `"cross_entropy"` and `"target_rank"` (the last two need targets). Default: every statistic the batches support |
| `stats_names=` | keyword of `GraphIR.join_oracle` (default `None`) |
| `status=` | keyword of `DiffReport.rows_with_status` (required); keyword of `tl.agent.build_envelope` (keyword-only; default `'ok'`): Result status token (`"ok"` unless a floor degradation applies) |
| `statuses=` | keyword of `Trace.projective_fields` (keyword-only; default `None`); keyword of `Trace.receptive_fields` (keyword-only; default `None`): Optional typed receptive-field-status filter |
| `std=` | keyword of `tl.add_noise` (required): Noise standard deviation; constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Per-channel normalization stds |
| `stdevs=` | keyword of `tl.attribution.gradient_shap` (keyword-only; default `0.0`): ABSOLUTE Gaussian input-noise scale (default 0.0 -- zero noise); keyword of `tl.attribution.noise_tunnel` (keyword-only; required): ABSOLUTE Gaussian noise scale: one nonnegative float for every attributed leaf, or a sequence aligned with the leaves in traversal order. Complex leaves draw a standard complex normal (real and imaginary components each carry variance 1/2). Required -- a hidden range-relative default would be an .. |
| `steered=` | keyword of `Bundle.add_alternative` (keyword-only; required): The steered call name, or a sequence of names |
| `step=` | keyword of `HistoryReader.observations` (keyword-only; default `None`); keyword of `WatchSession.emit_check` (required); keyword of `tl.export.tensorboard` (keyword-only; required): Global step for emitted summaries. REQUIRED: the historical `step=0` default was exactly the axis footgun the tracker design bans (every emission path carries a caller step); keyword of `tl.trackers.watch` (keyword-only; default `None`): Optional zero-argument callable returning the caller's global step (the unchanged-loop spelling). The explicit `watch.step(n)` scope is authoritative and REQUIRED for resume; keyword of `tl.transforms.apply_dispatch` (required): A raw unary callable or a declared `ContextTransform` |
| `step_audit=` | keyword of `tl.attribution.integrated_gradients` (keyword-only; default `None`): Audit-ladder rung when batching is on: `"per_call"` (default; one randomized `(chunk, row)` recomputed sequentially), `"per_chunk"` (an independent random row per chunk), or `"off"` (explicit expert choice, disclosed). The audit is a sampled test, never a proof; keyword of `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (keyword-only; default `None`): Audit-ladder rung when batching is on (`"per_call"` default, `"per_chunk"`, or the explicit expert `"off"`) |
| `step_audit_seed=` | keyword of `tl.attribution.integrated_gradients` (keyword-only; default `None`): Optional deterministic seed for the audit's own (chunk, row) draw; `None` draws a fresh disclosed seed. The seed always rides `extra["step_audit"]["seed"]`; keyword of `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (keyword-only; default `None`): Optional deterministic seed for the audit's own (chunk, row) draw; `None` draws a fresh disclosed seed riding `extra["step_audit"]` |
| `step_axis=` | constructor keyword of `tl.options.EpisodeOptions` (default `-1`): Axis of the step-output source along which per-step evidence lies. Positions are TAIL-ALIGNED: the LAST `n_steps` positions along this axis are the per-step emissions (one appended position per step), so a root returning what real `generate()` returns — prompt+completion — keeps its prompt .. |
| `step_batch_size=` | keyword of `tl.attribution.integrated_gradients` (keyword-only; default `None`): Optional number of path points stacked on the ordinary batch axis per forward/backward. `None` (the default) and `1` run sequentially -- batching is strictly opt-in and is a THROUGHPUT feature, not a memory feature. Batched runs are guarded by the randomized audit; keyword of `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients` (keyword-only; default `None`): Optional number of path points stacked per forward/backward; `None` (default) and `1` run sequentially. Strictly opt-in throughput |
| `step_input_from=` | constructor keyword of `tl.options.EpisodeOptions` (default `None`): Which stepped-call argument is the carried step input the join measures: a positional index (int) or a keyword name (str). None selects by the disclosed preference rule (input_ids and friends first, auxiliary masks/positions/timesteps last). A declaration naming no tensor argument is disclosed, .. |
| `step_output_from=` | constructor keyword of `tl.options.EpisodeOptions` (default `None`): Declared step-output SOURCE: which root-output slot the per-step evidence derives from, as a dot-separated container path over the root output structure (dict keys and namedtuple/`ModelOutput` field names by name, tuple positions by index — `"sequences"`, `"0"`, `"0.logits"`). `None` .. |
| `step_output_kind=` | constructor keyword of `tl.options.EpisodeOptions` (default `'tokens'`): Declared per-step evidence kind: `"tokens"` (default) reads integer token ids from the source; `"digest"` reads per-step content digests (any dtype — the float/hidden-state shape); `"none"` derives no evidence at all (a status-only ledger; the shape for roots with no per-step .. |
| `step_seconds=` | keyword of `tl.report.mfu` (keyword-only; required) |
| `stepped_module=` | constructor keyword of `tl.options.EpisodeOptions` (default `None`): The stepped model: the `nn.Module` whose successive top-level calls define step boundaries (call 1 is the prefill, ledger row 0). Must be a submodule of the traced episode root; refused typed (`episode_declaration_invalid`) otherwise. Step tallying is FLAT: each top-level call of this module is .. |
| `steps=` | keyword of `Recording.enrich` (required): Enrichment names, or `"all-feasible"` for all currently computable enrichments; keyword of `tl.mechinterp.edge_scores` (keyword-only; default `None`): Keyword-only. EAP-IG's number of evaluation points `m`; `None` means 5 (the MIB reference default). Refused with `method="eap"`; constructor keyword of `tl.transforms.TransformPipeline` (required): Ordered steps: frozen `TransformSpec` rows and/or `OpaqueStep` disclosures. Applied left to right |
| `steps_per_batch=` | keyword of `tl.attribution.text` (keyword-only; default `8`): Path points stacked per forward (default 8, gated by the randomized batching audit); `None`/`1` for sequential |
| `stimuli=` | keyword of `ExtractionPlan.run` (required): Tensor with leading stimulus dimension, or iterable of items; keyword of `tl.extract_dataset` (required): Tensor with a leading batch dimension, or an iterable of stimulus items (tensors, containers, or TEXT strings — text routes through the once-per-run tokenizer with padding and the full mapping forwarded); keyword of `tl.viz.feature_maps` (keyword-only; default `None`): Stimulus selection. `None` selects the first `max_stimuli` items, an integer selects the first that many items, and a sequence selects explicit indices |
| `stimulus_ids=` | keyword of `ExtractionPlan.run` (keyword-only; default `None`): Optional per-stimulus identifiers (disk mode); keyword of `tl.extract_dataset` (keyword-only; default `None`): Optional per-stimulus identifiers (DOCUMENTED-UNSTABLE): non-empty strings, duplicates legal (recorded as `ids_unique`); validated against cardinality BEFORE any forward for sized inputs, in lockstep for unsized iterables, and with exact equality before completion. Disk mode only; keyword of `tl.neuro.export_npz` (keyword-only; default `None`): Optional per-stimulus identifiers recorded in the sidecar |
| `stochastic=` | constructor keyword of `tl.transforms.TransformDefinition` (default `False`): Whether the transform draws randomness; stochastic transforms must record an explicit realized seed + seed_source |
| `stop_after=` | keyword of `tl.pluck` (default `None`): Stop-early site or predicate, passed to the capture as `trace(stop_after=...)`: the forward halts inclusively after the site, and an unsupported spelling refuses typed. `None` reads the ambient `tl.experimental.stop_after` site, if one is active; keyword of `tl.record` (keyword-only; default `None`): Optional predicate evaluated after each event's save decision. Returning `True` stops the active forward pass and marks the recording stopped. `activation_transform` is applied to each retained out copy after dtype/device transforms (errors propagate as `UserTransformError`); ..; keyword of `tl.trace` (keyword-only; default `None`): One stop directive, a site or a predicate: a module address string (`"encoder.layer.4"`), a function name (`"relu"`), a live selector (`tl.func("relu")`, `tl.module(...)`) or a callable predicate taking the live op context. Capture halts inclusively after the first match and returns a ..; constructor keyword of `tl.recording.Recorder` (keyword-only; optional): Optional predicate evaluated after each event's save decision. Returning `True` stops the active forward pass and marks the recording stopped |
| `storage=` | keyword of `tl.record` (keyword-only; default `None`): Grouped storage options (`tl.options.StorageOptions`); constructor keyword of `tl.recording.Recorder` (keyword-only; optional): Fastlog recording options; keyword of `tl.trace` (keyword-only; default `None`): Shared storage routing option. `storage=tl.to_disk(path)` streams predicate-selected saves to a disk bundle during the forward pass. `None` preserves the existing in-RAM behavior |
| `store_draws=` | keyword of `tl.attribution.gradient_shap` (keyword-only; default `False`): Whether to place the full draw record in `extra["draws"]` (off by default; ordinary results keep only digests and distributions) |
| `store_input_ids=` | keyword of `tl.extract_dataset` (keyword-only; default `False`): Store each batch's token ids as the ordinary `input_ids` key (DOCUMENTED-UNSTABLE, token store), trimmed by the batch mask under `ragged="trim"` like the token-axis activations; read token rows back with `open_extraction(...).token_rows()` |
| `strategy=` | constructor keyword of `tl.types.InputProvenance` (default `None`): Inference strategy that produced the winning input (rung 3) |
| `stream=` | keyword of `HistoryReader.observations` (keyword-only; default `None`); keyword of `HistoryView.series`, `tl.viz.contact_sheet`, `tl.viz.histogram_waterfall`, `tl.viz.quantile_fan` (default `'activation'`) |
| `stream_safe=` | constructor keyword of `tl.transforms.PlannedStep` (required): Whether the step can run at emission time; constructor keyword of `tl.transforms.TransformDefinition` (default `True`): Whether the transform can run at emission time |
| `strength=` | keyword of `Direction.steer`, `tl.steer` (default `1.0`): Scalar multiplier |
| `strict=` | keyword of `Trace.clear_edits` (keyword-only; default `False`): Whether no-op detach requests should raise; keyword of `Trace.find_sites` (keyword-only; default `False`): Whether to reject non-portable query forms; keyword of `Trace.save` (keyword-only; default `True`): Refuse typed on any field the level cannot persist; keyword of `tl.export.validate_model_explorer_payload` (keyword-only; default `False`): Raise the typed refusal on the first failure instead of reporting; constructor keyword of `tl.intervention.SamplingPlan` (required); keyword of `tl.intervention.resample_rows_from` (keyword-only; default `False`): Refuse agreement classes of size 1; keyword of `tl.intervention.sample_from` (keyword-only; default `False`): Refuse (rather than disclose) agreement classes of size 1; keyword of `tl.mechinterp.residual_decomposition` (keyword-only; default `True`): Enforce the execution-order bitwise identity (default). A failing strict walk refuses OPEN; `strict=False` instead appends an `unresolved_remainder` row, stamps the stack `diagnostic_only`, and is rejected by DLA, gallery, and launch claims; constructor keyword of `tl.options.InterventionOptions` (optional): Whether selector and propagation checks raise instead of warning; constructor keyword of `tl.options.ReplayOptions` (optional): Whether divergence warnings should raise; keyword of `tl.preprocessing.compare` (keyword-only; default `False`): When True, refuse `mismatch` and `unknown` verdicts with DISTINCT typed codes (`preprocessing_audit_mismatch` / `preprocessing_audit_unknown`). Default is non-raising (the report is the product) |
| `stride=` | keyword of `ReceptiveFieldRuleContext.window` (keyword-only; default `1`) |
| `strides=` | keyword of `tl.attribution.occlusion_map` (keyword-only; default `None`): Strides over the same axes; defaults to the window (non-overlapping) |
| `strings=` | keyword of `tl.viz.nodes.text_table` (required): Text items to render |
| `structural_site_key=` | constructor keyword of `tl.types.Site` (default `None`) |
| `structure_only=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `structure_only_capture=` | constructor keyword of `tl.backends.BackendCapabilities` (default `False`): Whether structure-only capture (`structure_only=True`) is supported. DOCUMENTED-UNSTABLE surface pending naming-session/ ratification |
| `style=` | keyword of `SummaryReport.print` (default `'auto'`); keyword of `SummaryReport.render` (default `'ascii'`); keyword of `Trace.decode_output` (keyword-only; default `None`): Decode style: `"classification"`, `"imagenet"`, `"imagenet1k"`, `"hf_text"` or a registered custom style. `None` lets the registered detectors pick one; keyword of `Trace.output_table` (keyword-only; default `None`): Optional decode style for a decode on demand (see `decode_output`); keyword of `Trace.summary` (keyword-only; default `None`): `"auto"` \| `"ascii"` \| `"unicode"` for display helpers; `str(result)` is ALWAYS the canonical ASCII payload; constructor keyword of `tl.stats.NarrationObserver` (default `'ascii'`); keyword of `tl.stats.glance` (keyword-only; default `'ascii'`): `"ascii"` (canonical) or `"unicode"` -- the two byte-checked renderings of identical content (`ascii == degrade(unicode)`); keyword of `tl.stats.narrate` (keyword-only; default `'ascii'`): Rendering style for the echoed cores |
| `subject=` | keyword of `tl.debug.nonfinite_origin` (required): A completed `Trace` (post-hoc, free) or a model (live tripwire); keyword of `tl.report.diff` (required): The capture whose perspective the delta takes (`subject - reference`); keyword of `tl.report.flops` (required): A finished `Trace` (the retroactive door), or an `nn.Module` (the model door -- runs ONE capture under the summary execution contract: eval mode, `torch.no_grad()`, module training flags and RNG state restored bit-identically afterwards). Model-door inputs; ..; keyword of `tl.viz.intervention_diff` (required) |
| `subnormal_fraction_threshold=` | keyword of `tl.debug.dtype_range` (keyword-only; default `0.1`): Minimum fraction of all tensor elements in the subnormal interval to flag. Defaults to `0.1`; keyword of `tl.debug.param_health` (keyword-only; default `0.1`): Minimum subnormal fraction that warns |
| `substring=` | keyword of `tl.label_contains` (required): Substring to match against labels |
| `subtree_renderer=` | keyword of `tl.bridge.treescope.treescope_repr` (required) |
| `summary=` | keyword of `tl.appliances.register` (keyword-only; default `''`): Optional one-line description; constructor keyword of `tl.types.ModuleInputSnapshot` (required): Bounded summary that does not materialize lazy payloads |
| `support_mask=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `support_ranges=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `svg_fonttype=` | keyword of `tl.transformer.render_answer_trajectory`, `tl.transformer.render_neuron_card`, `tl.transformer.render_prediction_ribbon`, `tl.viz.ablation_grid`, `tl.viz.attention_atlas`, `tl.viz.logit_lens`, `tl.viz.metric_strip`, `tl.viz.prediction_table`, `tl.viz.token_strip` (keyword-only; default `'path'`): `'path'` (default) or `'none'`; keyword of `tl.viz.attention` (keyword-only; default `'path'`): `'path'` (default) or `'none'` for editable text |
| `table=` | keyword of `tl.transformer.prediction_table_html` (required); keyword of `tl.viz.prediction_table` (required): The prediction-table record |
| `tag=` | keyword of `TagGrammar.is_data_tag`, `tl.trackers.assert_tag_safe` (required) |
| `tail_on_error=` | constructor keyword of `tl.options.NarrationOptions` (keyword-only; optional): Bound of the always-riding crash deque (metadata-only rendered lines). On a forward exception the sink synchronously receives the last `tail_on_error` selected events, oldest first. `max_lines` never caps this deque |
| `target=` | keyword of `ReceptiveFieldView.at`, `ReceptiveFieldView.probe_grad` (keyword-only; default `None`): Optional descendant graph point selecting the far endpoint of a projective query. Not permitted together with `source`; keyword of `ReceptiveFieldView.check` (keyword-only; default `None`): Optional descendant graph point for projective validation; keyword of `ReceptiveFieldView.show` (keyword-only; default `None`): Optional descendant graph point selecting the far endpoint of a projective overlay; keyword of `Selection.align_to` (required); keyword of `tl.attribution.build_graph` (keyword-only; default `None`): `None` picks the salient logits (`max_num_logits` and `logit_probability_coverage`); a list of token ids names them; keyword of `tl.attribution.causal_audit` (keyword-only; default `None`): The target the claim was computed for: for a layer claim an integer class index, a callable `output -> scalar`, or a `SiteTarget`; for a read table the `tl.attribution.seed(...)` it was read for. Left out for a direct-logit table, which fixes its own logits, and for an EDGE table, whose ..; keyword of `tl.attribution.deconvolution`, `tl.attribution.gradient_shap`, `tl.attribution.guided_backprop` (keyword-only; required): Integer class index, callable `output -> scalar tensor`, or a `SiteTarget` naming a hidden layer's output to attribute; keyword of `tl.attribution.deletion_curve` (keyword-only; required): Integer class index (scored per example) or callable `output -> scalar tensor`; the target the attribution explains; keyword of `tl.attribution.grad_cam`, `tl.attribution.input_x_grad`, `tl.attribution.integrated_gradients`, `tl.attribution.layer_attribution`, `tl.attribution.layer_conductance`, `tl.attribution.layer_integrated_gradients`, `tl.attribution.saliency`, `tl.attribution.smoothgrad` (keyword-only; required): Integer class index selecting `output[..., target]` and summing the selected values, a callable `output -> scalar tensor`, or a `SiteTarget` naming a hidden layer's output (and unit) to attribute instead of the model output; keyword of `tl.attribution.infidelity` (keyword-only; required): Integer class index or callable `output -> scalar tensor`; must be the SAME target the attribution was computed for; keyword of `tl.attribution.insertion_curve` (keyword-only; required): As in `deletion_curve`; `baseline` is the starting input; keyword of `tl.attribution.noise_tunnel` (keyword-only; default `None`): Target for the `method=` sugar route; keyword of `tl.attribution.occlusion_map` (keyword-only; required): Integer class index (per-example deltas) or callable `output -> scalar tensor` (batch-aggregate deltas, disclosed); keyword of `tl.attribution.randomization_test` (keyword-only; required): Target passed to `method`; keyword of `tl.attribution.read` (keyword-only; default `None`): Keyword-only: a `seed(site, index=\|cotangent=)` edge-seeded target (primary), a graph-connected scalar Tensor, a pure `Trace -> Tensor` callable (1-D result = target batch), or a list of these. Order preserved. Required for gradient-bearing methods; forbidden for `method='activation'`; keyword of `tl.attribution.road_curve` (keyword-only; required): Integer class index (scored per example) or callable `output -> scalar tensor`; keyword of `tl.attribution.sensitivity` (keyword-only; default `None`): Target for the sugar route; keyword of `tl.attribution.text` (keyword-only; default `None`): `None` (the model's own argmax, disclosed), an int (vocab id at the final non-padding position for causal LMs -- the position-ambiguity band WARNS), a `(position, vocab)` tuple, a single-token string, a contrastive `{"target": ..., "foil": ...}` pair, or a callable `logits -> scalar`; keyword of `tl.bridge.captum.attribute` (default `None`): Captum target, forwarded as `target=` only when given: the activation and neuron classes (`LayerActivation`, `NeuronGradient`, ...) take no `target` and reject one; keyword of `tl.debug.param_health` (required): `nn.Module` or any name->tensor `Mapping` (state_dict, flattened optimizer state, EMA shadow, loaded checkpoint); keyword of `tl.mechinterp.full_decomposition` (default `'final'`); keyword of `tl.mechinterp.residual_accumulation` (default `'final'`): `"final"` (the tensor the final norm consumes, anchored by dataflow), an op label, or an op record; keyword of `tl.mechinterp.residual_decomposition` (default `'final'`): `"final"`, an op label, or an op record (see `residual_accumulation`); keyword of `tl.model_output` (required): Output index or semantic role; keyword of `tl.neuro.compare` (required): `[n_stimuli, ...]` responses (brain recordings, another model's features) for the same stimuli in the same order; keyword of `tl.receptive_field.cross_validate` (keyword-only; default `None`): Optional projective-direction descendant selector or selectors; keyword of `tl.receptive_field.node_spec` (keyword-only; default `None`) |
| `target_2d=` | keyword of `tl.repgeom.procrustes_align` (required): Target coordinates with shape `[N, 2]` |
| `target_batch_size=` | keyword of `tl.attribution.read` (keyword-only; default `'auto'`): `'auto'` (plan disclosed; CUDA default stays sequential pending C-READ) or an explicit positive int |
| `target_id=` | keyword of `ReadTable.target` (required): One of `target_ids` |
| `target_scope=` | constructor keyword of `tl.facets.FacetFunction` (required): Record scope the recipe targets; keyword of `tl.facets.functions` (default `None`): Only facet functions targeting this record scope; keyword of `tl.facets.register` (keyword-only; default `'any'`): Internal scope marker used for discoverability filtering |
| `target_token_id=` | keyword of `tl.transformer.prediction_trajectory` (keyword-only; default `None`): Optional tracked answer token; must be one of the ids the extractor retained (`token_ids`); keyword of `tl.viz.logit_lens` (keyword-only; default `None`): Tracked token id; required with `value="target_prob"` |
| `targets=` | keyword of `tl.bridge.pytorch_grad_cam.cam` (keyword-only; default `None`): Optional pytorch-grad-cam targets |
| `task=` | keyword of `tl.bridge.lit.model` (keyword-only; default `'classification'`): `"classification"` or `"causal_lm"` |
| `tensor=` | keyword of `CaptureBackend.log_source_tensor` (required): Backend tensor value to log as a source; keyword of `OpaqueStep.apply`, `TransformPipeline.apply`, `TransformSpec.apply` (required): Batch tensor with the stimulus axis leading; keyword of `TensorSpec.from_tensor` (required): Tensor with the stimulus axis leading; keyword of `tl.features.as_matrix` (required): Activation tensor with a stimulus axis; keyword of `tl.stats.glance` (required): Tensor to summarize. Never mutated; autograd/RNG state untouched; the kernel is allocation-free and single-sync per the contract; keyword of `tl.stats.tensor_stats` (required): Payload to summarize. Never mutated; autograd state untouched; keyword of `tl.transforms.apply_dispatch` (required): Batch tensor |
| `tensor_observations=` | constructor keyword of `tl.types.ModuleInputSnapshot` (required): Ordered tensor observations keyed by typed input paths |
| `tensors=` | constructor keyword of `tl.types.InputProvenance` (required): Per-tensor shape/dtype/device facts (direct arguments; nested containers are the capture record's job) |
| `tested_through=` | constructor keyword of `tl.backends.FrameworkRequirement` (required): Highest version the backend's suite has run on |
| `text=` | keyword of `tl.attribution.text` (required): Input text; keyword of `tl.bridge.transformers.trace_text` (required): String, list of strings, or list of message dictionaries when `chat_template=True`; keyword of `tl.report.summary` (required) |
| `theme=` | keyword of `Bundle.draw` (keyword-only; default `None`); keyword of `Trace.draw` (keyword-only; default `'torchlens'`): Skin preset: `"torchlens"` (default), `"paper"`, `"dark"`, `"colorblind"` or `"high_contrast"`; constructor keyword of `tl.options.VisualizationOptions` (optional): Theme name; keyword of `tl.viz.bundle_diff` (keyword-only; default `'paper'`): Visualization theme preset |
| `threshold=` | keyword of `tl.low_variance` (keyword-only; required) |
| `thumbnail_size=` | keyword of `tl.viz.image_scatter` (keyword-only; default `36`): Maximum width and height for pasted thumbnails; keyword of `tl.viz.nodes.feature_map` (keyword-only; default `72`): Required minimum size for raw stimulus thumbnails; keyword of `tl.viz.nodes.mds_scatter` (keyword-only; default `36`): Maximum width and height for each pasted thumbnail; keyword of `tl.viz.nodes.rdm` (keyword-only; default `24`): Maximum width and height for axis thumbnails |
| `tier=` | keyword of `tl.export.register_target` (keyword-only; required): Per-member tier row: `"present"` (native emitter) or `"bridge"` (foreign-peer-shaped writer) |
| `tiers=` | keyword of `tl.report.overhead` (keyword-only; default `('raw_forward', 'trace')`) |
| `tiles_per_page=` | keyword of `tl.viz.contact_sheet` (keyword-only; default `12`) |
| `title=` | keyword of `tl.viz.logit_lens` (keyword-only; default `None`): Optional subject (for example the model's name) printed before the fixed figure title; the lens result carries no model identity |
| `to=` | keyword of `tl.export.aim` (default `None`): Optional object exposing `track`, `add_tag` and item assignment; keyword of `tl.export.wandb` (default `None`): Optional existing W&B run object. If omitted, `wandb.run` is used when present, but a new run is not created; constructor keyword of `tl.trackers.HFTrainerWatchCallback`, `tl.trackers.LightningWatchCallback` (keyword-only; required); constructor keyword of `tl.trackers.WandbSink` (required); keyword of `tl.trackers.build_manifest` (required); keyword of `tl.trackers.watch` (keyword-only; required): One sink or a tuple of sinks implementing the documented protocol (`TensorBoardSink`, `WandbSink`, `JSONLSink`, or your own) |
| `token=` | keyword of `tl.bridge.huggingface_hub.from_hub` (keyword-only; default `None`): Optional Hub token (used only when `api` is not given); keyword of `tl.bridge.huggingface_hub.push_to_hub` (keyword-only; default `None`): Optional Hub token |
| `token_id=` | keyword of `PromptInspection.assert_rank`, `PromptInspection.assert_top` (required) |
| `token_ids=` | keyword of `tl.transformer.token_metrics` (required): `[n_positions]` input token ids aligned with the logits positions |
| `tokenizer=` | keyword of `AttributionGraph.to_neuronpedia` (keyword-only; required): Anything with `decode(list_of_ids) -> str`, such as the model's Hugging Face tokenizer; it turns input and target token ids into the schema's token strings; keyword of `LogitLens.summary` (keyword-only; default `None`): Optional tokenizer used to decode token ids; keyword of `LogitLens.top_tokens` (keyword-only; default `None`): Optional tokenizer with `convert_ids_to_tokens` or `decode`; when omitted, raw token ids are returned; keyword of `Trace.decode_output` (keyword-only; default `None`): Tokenizer with a `decode` method, used by `"hf_text"`; without one the rows carry the argmax `token_ids` and `text` is `None`; keyword of `tl.attribution.text` (required): Matching tokenizer; keyword of `tl.bridge.lit.model` (default `None`): The checkpoint's tokenizer. Never mutated; batches are padded manually with `pad_token_id` (falling back to `eos_token_id`); keyword of `tl.bridge.transformers.trace_text` (keyword-only; default `None`): Optional explicit tokenizer. Defaults to auto-resolving from `model.name_or_path` or `model.config.name_or_path`; keyword of `tl.dataset_extraction.hf_collate` (default `None`): Caller-supplied tokenizer (first in the resolution order); `None` defers to the attached-then-hub doors; keyword of `tl.mechinterp.inspect_prompt` (keyword-only; default `None`): Optional tokenizer for piece rendering + the BOS disclosure; keyword of `tl.mechinterp.test_prompt` (keyword-only; default `None`); keyword of `tl.transformer.prediction_table`, `tl.transformer.prediction_trajectory` (keyword-only; default `None`): Optional tokenizer for display strings; keyword of `tl.viz.logit_lens` (keyword-only; default `None`): Optional tokenizer used to decode the predicted token ids |
| `tokens=` | keyword of `tl.bridge.sae.splice` (keyword-only; default `None`): Integer token ids the model ran on. The first model output shaped `tokens.shape + (vocab,)` is read as next-token logits; one more replay with the site zero-ablated gives the CE baseline. Without it the CE and KL fields are `None` with a reason; keyword of `tl.bridge.sae.write_delphi` (required): Integer token ids `[examples, sequence]`; row `i` is example `i` (the example ids a `UnitTopK` names); keyword of `tl.bridge.sae.write_sae_dashboard` (required): Integer token ids `[examples, sequence]`; row `i` is example `i`; keyword of `tl.mechinterp.token_directions` (required): Token id or sequence of token ids; constructor keyword of `tl.transformer.TokenAxis` (required): Display strings, one per axis position, never merged; keyword of `tl.transformer.attention_view` (keyword-only; default `None`): Token strings for BOTH axes (self-attention convenience); keyword of `tl.transformer.attention_views` (keyword-only; default `None`): Token strings for BOTH axes (see `attention_view`); keyword of `tl.transformer.token_metrics` (required): Display strings per position; keyword of `tl.viz.logit_lens` (keyword-only; default `None`): Optional input-token labels, one per captured position |
| `tolerance=` | keyword of `tl.transformer.score_decomposition` (keyword-only; default `0.001`): Relative closure tolerance for the sum-to-score invariant |
| `tool_class=` | constructor keyword of `tl.agent.AgentToolSpec` (required) |
| `trace=` | keyword of `BackendProtocol.validate`, `tl.experimental.dagua.trace_to_dagua_graph`, `tl.io.attach_sidecar`, `tl.io.read_sidecar`, `tl.mechinterp.alias_report`, `tl.mechinterp.full_decomposition`, `tl.mechinterp.stack_facet`, `tl.report.backward_estimate`, `tl.report.backward_status`, `tl.report.summary`, `tl.trackers.build_graph_ir`, `tl.validation.get_validation_failure` (required); keyword of `PartialTrace.from_trace` (required): `Trace` instance active when capture failed; keyword of `Trace.replace_state_from` (required): Fully postprocessed fresh log whose graph, layer containers, accessors, output metadata, shape/hash fields, and per-pass entries should replace this log's current run-state; keyword of `tl.attribution.build_graph` (required): A finished capture of one sequence, `tl.trace(model, tokens[None])`, with activations saved; keyword of `tl.attribution.occlusion` (required): Live captured Trace with saved values; keyword of `tl.attribution.read` (required): Live finished torch trace; keyword of `tl.bridge.brain_score.per_layer`, `tl.bridge.sae_lens.encode`, `tl.bridge.sae_lens.reconstruct` (required): TorchLens `Trace` containing saved outs; keyword of `tl.bridge.captum.attribute` (required): TorchLens `Trace` from the model being attributed; keyword of `tl.bridge.captum.layer`, `tl.bridge.pytorch_grad_cam.cam`, `tl.bridge.shap.explain` (required): TorchLens `Trace` with a live source model reference; keyword of `tl.bridge.profiler.join`, `tl.bridge.pytorch_grad_cam.layer`, `tl.export.to_model_explorer_dict` (required): TorchLens `Trace`; keyword of `tl.debug.bisect_nan` (required): Completed TorchLens trace with saved activations; keyword of `tl.debug.bisect_nan_backward` (required): Completed TorchLens trace with a captured backward; keyword of `tl.debug.dead_neurons`, `tl.debug.dtype_range`, `tl.debug.grad_health`, `tl.debug.hot_path`, `tl.debug.hot_path_rows`, `tl.debug.lineage`, `tl.debug.recompute_candidates`, `tl.facets.coverage` (required): Completed TorchLens trace; keyword of `tl.export.aim`, `tl.export.chrome_trace`, `tl.export.csv`, `tl.export.flamegraph`, `tl.export.json`, `tl.export.memory_timeline`, `tl.export.memory_timeline_v2`, `tl.export.mlflow`, `tl.export.model_explorer`, `tl.export.netron`, `tl.export.parquet`, `tl.export.safetensors`, `tl.export.speedscope`, `tl.export.wandb` (required): TorchLens `Trace` to export; keyword of `tl.export.html` (required): TorchLens `Trace` or `PartialTrace` to export; keyword of `tl.export.tensorboard` (required): TorchLens `Trace` to summarize; keyword of `tl.export.xarray` (required): TorchLens `Trace` whose saved tensor outs should be exported; keyword of `tl.intervention.check_spec_compat` (required): Model log to check; keyword of `tl.inventory.sites_of_trace` (required): Any TorchLens `Trace` (a metadata-only capture suffices; a `structure_only` capture serves shapes labeled as hypotheses); keyword of `tl.io.materialize_nested` (required): Model log loaded from a portable bundle; keyword of `tl.io.save_intervention` (required): Trace-like object whose `_intervention_spec` should be persisted; keyword of `tl.mechinterp.apply_norm_scale` (required): The trace whose captured norm supplies the operating point; keyword of `tl.mechinterp.attention_head_contributions`, `tl.mechinterp.head_weight_views`, `tl.mechinterp.ov_circuit`, `tl.mechinterp.qk_circuit`, `tl.mechinterp.residual_decomposition`, `tl.mechinterp.resolve_alias` (required): A finished torchlens trace; keyword of `tl.mechinterp.direct_logit_contributions`, `tl.mechinterp.token_directions` (required): A finished torchlens trace with an unembedding head; keyword of `tl.mechinterp.head_scores` (required): A finished torchlens trace with attention patterns readable; keyword of `tl.mechinterp.inspect_prompt` (required): A finished torchlens trace (post-hoc from a saved trace works; retention: logits only); keyword of `tl.mechinterp.mlp_neuron_contributions` (required): A finished torchlens trace with the MLP interior saved; keyword of `tl.mechinterp.residual_accumulation` (required): A finished torchlens trace of the model forward; keyword of `tl.mechinterp.retention_plan` (required): Any finished trace of the model -- a cheap metadata-first capture works; the plan is structural (the two-forward discovery route: capture metadata, plan, recapture selectively with `save=plan.predicate`); keyword of `tl.preprocessing.check_card` (required): A `Trace` (its `input_preprocessor` is read) or a `ResolvedPreprocessing` record; keyword of `tl.receptive_field.cross_validate` (required): Backward-ready captured trace; keyword of `tl.receptive_field.verify` (required): Backward-ready trace to inspect; keyword of `tl.repgeom.mds_evolution`, `tl.repgeom.rdm_evolution`, `tl.repgeom.scree_evolution`, `tl.viz.feature_maps` (required): Captured TorchLens trace with saved activation payloads; keyword of `tl.report.cost_tree`, `tl.report.roofline` (required): Finished trace; keyword of `tl.report.padding_waste` (required): Finished trace of a PADDED batch forward; keyword of `tl.report.profile` (required): Completed trace whose existing metadata is reported; keyword of `tl.stats.site_stats` (required): A trace captured with `tl.trace(model, x, save="stats")`; keyword of `tl.transformer.attention_view` (required): A finished TorchLens trace of a transformer forward; keyword of `tl.transformer.attention_views`, `tl.transformer.score_decomposition` (required): A finished TorchLens trace; keyword of `tl.validation.check_metadata_invariants` (required): Postprocessed model log to validate; keyword of `tl.validation.get_validation_diagnostics` (required): Trace potentially carrying validation diagnostics; keyword of `tl.mechinterp.component_graph` (required): A finished trace of a pre-norm additive-residual language model, captured with every op saved (`save='all'`) and the builtin attention recipes' head views (eager attention); keyword of `tl.mechinterp.patch_edges` (required): The clean trace the edges' graph was built from, captured with `save='all'` and `replayable=True` |
| `trace_a=` | keyword of `tl.debug.compare`, `tl.debug.first_divergence` (required): First completed TorchLens trace |
| `trace_b=` | keyword of `tl.debug.compare`, `tl.debug.first_divergence` (required): Second completed TorchLens trace |
| `trace_kwargs=` | keyword of `tl.debug.check_determinism` (keyword-only; default `None`): Extra `tl.trace` keyword arguments for every run; keyword of `tl.mechinterp.patch_grid`, `tl.mechinterp.patch_query_heads_grid` (keyword-only; default `None`): Extra kwargs forwarded to `tl.trace`; keyword of `tl.mechinterp.patch_head` (keyword-only; default `None`): Extra kwargs for the baseline traces |
| `trace_options=` | constructor keyword of `tl.backends.BackendCapabilities` (default `()`): Backend-specific public `trace` keyword options accepted by this backend in addition to the backend-neutral options |
| `trace_state=` | keyword of `CaptureBackend.finalize_forward_session` (required): The session trace's raw-graph workspace, passed as the explicit ownership token |
| `traces=` | keyword of `Bundle.add` (required): Model log or logs to add; constructor keyword of `tl.experimental.Session` (optional) |
| `track_device_memory=` | constructor keyword of `tl.options.CaptureOptions` (keyword-only; optional) |
| `trainer=` | keyword of `LightningWatchCallback.on_train_end`, `LightningWatchCallback.on_train_start` (required) |
| `trajectory=` | keyword of `tl.transformer.render_answer_trajectory` (required): The trajectory record; must carry a tracked target token; keyword of `tl.transformer.render_prediction_ribbon` (required): The trajectory record |
| `transcoders=` | keyword of `tl.attribution.build_graph` (required): Mapping from read site to transcoder. A read site is the MLP module name (the transcoder reads the module's call input) or an op label (it reads that op's output, for a read point inside a module, such as a norm's value before its gain); an op-label key needs `W_dec` and `b_dec` as mappings .. |
| `transforms=` | constructor keyword of `tl.facets.FacetSpec` (default `()`): Primitive transform chain |
| `truncation=` | keyword of `tl.agent.build_envelope` (keyword-only; default `None`): Truncation disclosure, or `None` when nothing was dropped |
| `trust_custom_callables=` | keyword of `tl.io.load_intervention_spec` (keyword-only; default `False`): Explicit permission to import custom callables when no allowlist is supplied. Enable only for specs from a trusted source; keyword of `tl.load` (keyword-only; default `False`): Explicit permission to import custom callables from an intervention spec when no allowlist is supplied. Enable only for trusted specs |
| `uncounted_op_counts=` | field of `BackwardStatus` |
| `unflatten=` | keyword of `tl.types.register_container` (required): Callable accepting `(aux_data, children)` and returning an instance |
| `unit=` | keyword of `ReceptiveFieldView.at` (required): Windowed-axis coordinates, or `"center"` for their midpoint. Coordinates must be non-negative and in range; unlike `.probe_grad()` and `.check()`, negative indices are rejected rather than wrapped; keyword of `ReceptiveFieldView.check`, `ReceptiveFieldView.probe_grad` (required): Complete target output-element index; keyword of `ReceptiveFieldView.show` (default `None`): Optional complete target output-element index; constructor keyword of `tl.attribution.SiteTarget` (keyword-only; default `None`): `None` (the default) attributes the whole output, summed. An int selects `output[..., unit]` (the last axis, as an integer `target=` does on the model output). A tuple indexes the axes after the batch axis, so `(3,)` is channel 3 of a convolution output and `(3, 1, 2)` one position of it; constructor keyword of `tl.receptive_field.GradReceptiveField` (required); keyword of `tl.receptive_field.node_spec` (keyword-only; default `None`): Draw-time RF options. `unit` is forwarded only when an input-node overlay is requested |
| `unit_axis=` | constructor keyword of `tl.stats.Quantile`, `tl.stats.TopK` (keyword-only; default `None`): Axis of each batch that indexes units (axis 0 is the example axis); `None` pools every value; constructor keyword of `tl.stats.UnitTopK` (required): The unit axis the accumulator was built with |
| `unit_shape=` | constructor keyword of `tl.receptive_field.GradReceptiveField` (keyword-only; default `()`) |
| `units=` | keyword of `ReceptiveFieldView.of` (required): What `tl.units(site, indices)` takes as `indices`, over this operation's FULL output shape (batch and channel included, unlike `.at()`'s windowed-axis coordinates): a bool mask, or an iterable of coordinate tuples; keyword of `Trace.summary` (keyword-only; default `None`): `"human"` (K/M/G) \| `"raw"` (full digits) in rendered text; keyword of `tl.receptive_field.cross_validate` (keyword-only; default `'center'`): `"center"`, `"corners"`, one complete index, or complete indices |
| `unknown_relations=` | keyword of `tl.load` (keyword-only; default `'opaque'`): Bundle loader-doctrine policy for member-relation rows of unknown NAMESPACED kinds (the C07X preserve-and-disclose amendment): `"opaque"` (default) loads them preserved, disclosed, unchecked, and never executed; `"refuse"` restores the strict typed refusal. Bare unknown kinds refuse under both .. |
| `unmentioned=` | keyword of `Bundle.vary` (keyword-only; default `None`): `None` (default) demands COMPLETE coverage — a member absent from the mapping refuses typed before any mutation. The explicit `"unchanged"` opts a subset call in; unmentioned members are recorded as unmentioned (distinguishable from explicit `None` identity after load) |
| `until=` | keyword of `Trace.run` (keyword-only; default `None`) |
| `unverified_reason_counts=` | keyword of `ValidationStatus.from_replay_counts`, `ValidationStatus.result`, `ValidationStatus.unverified` (keyword-only; default `None`): Stable reason-code counts for unverified replay decisions |
| `unwinding=` | keyword of `WatchSession.close` (keyword-only; default `False`) |
| `unwrap_when_done=` | constructor keyword of `tl.options.CaptureOptions` (optional) |
| `validate_entry=` | constructor keyword of `tl.backends.BackendSpec` (required): Public validation entry for model/input validation |
| `validate_layer_grads=` | keyword of `tl.validate` (keyword-only; default `None`): Backward-only layer-gradient validation flag. Omission enables honest captured-gradient validation by default for backward scope. The root module call is never compared, so a model whose forward calls no child module returns False with a `RuntimeWarning`; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `True`): If True (default), also validate captured per-module-output gradients. The root module call is never compared, so a model whose forward calls no child module returns False with a coded `TorchLensWarning` (`backward_validation_no_module_output_evidence`) |
| `validate_metadata=` | keyword of `Trace.validate_forward_pass` (default `True`): Forwarded unchanged to `validate_saved_outs`; keyword of `tl.validate` (keyword-only; default `True`): Whether metadata invariant checks should run for forward-like scopes; keyword of `tl.validation.validate_backward_pass` (keyword-only; default `True`): If True, run metadata invariant checks on the captured backward trace; keyword of `tl.validation.validate_forward_pass` (default `True`): If True, also run metadata invariant checks |
| `validate_trace=` | constructor keyword of `tl.backends.BackendSpec` (required): Validation entry for an already-built trace |
| `value=` | keyword of `Aggregator.update`, `Mean.update`, `Quantile.update`, `TopK.update` (required): Tensor-like batch value; keyword of `BackendProtocol.host_copy`, `BackendProtocol.is_tensor`, `CaptureBackend.apply_live_hooks`, `CaptureBackend.is_parameter`, `CaptureBackend.is_tensor`, `CaptureBackend.is_wrapped`, `CaptureBackend.safe_copy`, `CaptureBackend.set_tensor_label`, `CaptureBackend.tensor_ref`, `CaptureBackend.unwrap`, `CaptureBackend.wrap`, `FacetSpec.apply_transforms`, `Histogram.update`, `Norm.update`, `RunningStats.update`, `StreamingStat.update` (required); keyword of `Covariance.update` (required): Tensor-like batch. One-dimensional inputs are treated as one row; keyword of `PCA.update` (required): Tensor-like batch; keyword of `tl.agent.json_safe` (required): Arbitrary JSON-adjacent value; constructor keyword of `tl.attribution.AttributionMetric` (required): The primary scalar (reference-parity convention); keyword of `tl.debug.clone_input_tree` (required): Input value, container tree, or non-tensor leaf; keyword of `tl.fingerprint.content` (required): Tensor or nested tensor-containing value to hash; keyword of `tl.observers.log_value` (required): Scalar or JSON-like value to record; keyword of `tl.replace_with` (required): Tensor or zero-argument callable returning a tensor; constructor keyword of `tl.types.AttributeWrite` (required): What the forward left in the attribute: the object itself when the forward bound a new object (the record keeps it alive as long as the trace, so a large stashed tensor costs its full memory), a copy when the forward changed the original object in place (builtin `list`/`tuple`/`dict`/`set` ..; keyword of `tl.viz.logit_lens` (keyword-only; default `'top_prob'`): `"top_prob"` (color = the top-1 token's probability) or `"target_prob"` (color = `target_token_id`'s probability) |
| `value_or_hook=` | keyword of `Trace.do` (default `None`): Optional hook for the `(site, hook)` input shape |
| `value_range=` | constructor keyword of `tl.preprocessing.DeclaredPreprocessing` (default `None`): Declared pre-normalization value range, usually `(0.0, 1.0)` |
| `value_stats=` | constructor keyword of `tl.options.NarrationOptions` (optional): Value-stats rung: `"off"` (default; metadata only, zero value reads, zero device syncs), `"reuse"` (only facts another armed feature already paid for), `"sampled"` (bounded seeded-sample moments, `~`-marked, NEVER a finiteness claim), or `"exact"` (full exact stats; typed refusal above .. |
| `value_version=` | constructor keyword of `tl.facets.FacetSpec` (default `'raw_out'`): Which home value version the spec reads |
| `values=` | keyword of `BackendProtocol.realize_once`, `NarrationOptions.from_values`, `ReplayOptions.from_values`, `StorageOptions.from_values`, `VisualizationOptions.from_values` (required); keyword of `Container.reconstruct` (default `'out'`): Leaf value source. `"out"` uses saved outputs; `"transformed"` uses transformed outputs; keyword of `Container.reconstruct_at` (keyword-only; default `'out'`): Leaf value source. `"out"` uses saved outputs; `"transformed"` uses transformed outputs; keyword of `ReadTable.to_pandas` (default `'omit'`): `'omit'` drops dense values (scalar rows only; the default), `'object'` carries detached tensors as object cells, `'explode'` emits one row per element with a flat index column; keyword of `Trace.reconstruct_container` (keyword-only; default `'out'`): Leaf value source: `"out"` or `"transformed"`; keyword of `Trace.reconstruct_output` (default `'out'`): Leaf value source: `"out"` or `"transformed"`; constructor keyword of `tl.experiment.SweepGrid` (required); constructor keyword of `tl.stats.UnitTopK` (required): `float64 [units, k]`: each unit's largest values, descending. A unit that saw fewer than `k` non-NaN values has NaN in its empty slots; keyword of `tl.sweep` (optional): Replacement values to sweep at `site`, or edits (`tl.scale(0.5)`, a callable) applied there. With `site=tl.grad(s)` each member's edit changes the gradient of `s`'s output; constructor keyword of `tl.transformer.Annotation` (required): Per-head numbers; `None` = unmeasured (blank, never zero) |
| `vanishing_threshold=` | keyword of `ChecksSession.register_magnitude_check` (keyword-only; default `1e-07`); keyword of `tl.debug.grad_health` (keyword-only; default `1e-07`): Norm below which a nonzero finite gradient is flagged vanishing |
| `variance_threshold=` | keyword of `tl.repgeom.effective_dimensionality` (keyword-only; default `0.9`): Cumulative variance fraction used for the component-count callout; keyword of `tl.repgeom.scree_evolution` (keyword-only; default `0.9`): Validated threshold for consistency with render-time callouts. The returned and stored payloads remain eigenvalue vectors only; keyword of `tl.viz.nodes.scree` (keyword-only; default `0.9`): Cumulative variance fraction used for the threshold line and callout |
| `vector=` | constructor keyword of `tl.intervention.Direction` (required): The direction, a 1-D tensor over the site's trailing feature axis, in the dtype of the rows it was fitted on |
| `verbose=` | keyword of `Trace.validate_forward_pass` (default `False`): Forwarded unchanged to `validate_saved_outs`; constructor keyword of `tl.options.CaptureOptions` (optional); keyword of `tl.validate` (keyword-only; default `False`): Whether validators should emit diagnostics; keyword of `tl.validation.validate_forward_pass` (default `False`): If True, print detailed error messages on validation failure |
| `verification=` | constructor keyword of `tl.mechinterp.ProjectionSource` (required): `"payload_verified"` or `"unavailable"` (payloads not saved) |
| `verify=` | keyword of `tl.dataset_extraction.open_extraction` (keyword-only; default `'structural'`): `"structural"` (default: manifest/ledger well-formed, files present at their exact ledgered byte sizes) or `"first_access"` (each shard's CRC verified once per handle lifetime); keyword of `tl.mechinterp.measure_trajectory` (keyword-only; default `True`): As in `logit_lens`; the lens is checked on the first batch |
| `version=` | constructor keyword of `tl.transforms.PlannedStep` (required): Numerics-visible algorithm version of the step; constructor keyword of `tl.transforms.SrpSizing` (required): Formula version; constructor keyword of `tl.transforms.TransformDefinition` (required): Numerics-visible algorithm version; any numeric change bumps it |
| `version_index=` | keyword of `Buffer.value_at` (required) |
| `view=` | keyword of `Trace.summary` (keyword-only; default `None`): `"overview"` (default) \| `"compute"` \| `"stats"` (per-site payload statistics; see docs/reference/summary.md, "The stats view"); keyword of `tl.debug.sketch_grad_fn` (keyword-only; default `False`): Open the rendered file in the managed viewer (never a raw Popen); keyword of `tl.viz.attention` (required): The attention view record; keyword of `tl.viz.contact_sheet`, `tl.viz.histogram_waterfall`, `tl.viz.quantile_fan`, `tl.viz.site_detail` (required) |
| `views=` | keyword of `tl.transformer.bertviz_tuple` (required): Per-layer attention views in layer order; keyword of `tl.transformer.circuitsvis_attention`, `tl.transformer.circuitsvis_payload` (required): Per-layer attention views sharing one token axis; keyword of `tl.viz.attention_atlas` (required): One attention view per layer, in execution order |
| `vmax=` | keyword of `tl.viz.heatmap` (keyword-only; default `None`): Optional upper normalization bound |
| `vmin=` | keyword of `tl.viz.heatmap` (keyword-only; default `None`): Optional lower normalization bound |
| `vs=` | keyword of `tl.mechinterp.direct_logit_contributions` (keyword-only; default `None`): Optional comparison token id(s); when given, rows decompose the LOGIT DIFF `answer - vs` (the recommended, norm-robust form); keyword of `tl.mechinterp.inspect_prompt` (keyword-only; default `None`): Optional single comparator token id (logit-diff row); keyword of `tl.mechinterp.test_prompt` (keyword-only; default `None`) |
| `warmup=` | keyword of `tl.report.measure_clean_step` (keyword-only; default `2`) |
| `warnings=` | keyword of `tl.agent.build_envelope` (keyword-only; default `None`): Human-readable warnings (never load-bearing); constructor keyword of `tl.receptive_field.GradReceptiveField` (required) |
| `weight=` | keyword of `tl.intervention.set_affine_coordinates` (required): Finite real `[k, d]` tensor: row `j` maps the site value to the new coordinate `j` |
| `which=` | keyword of `tl.sign` (default `'positive'`) |
| `widen_unsupported=` | keyword of `tl.dataset_extraction.export_extraction` (keyword-only; default `True`): Widen bf16/fp8 members to fp32 with disclosure (default) or refuse typed |
| `width=` | keyword of `tl.viz.contact_sheet` (keyword-only; default `900`); keyword of `tl.viz.heatmap` (keyword-only; default `240`): Output image width in pixels; keyword of `tl.viz.histogram_waterfall`, `tl.viz.quantile_fan` (keyword-only; default `640`); keyword of `tl.viz.lineplot` (keyword-only; default `320`): Output image width in pixels; keyword of `tl.viz.nodes.histogram` (default `240`): Rendered image width; keyword of `tl.viz.nodes.scree` (keyword-only; default `320`): Width of the rendered scree image in pixels; keyword of `tl.viz.site_detail` (keyword-only; default `720`) |
| `window=` | keyword of `ChecksSession.register_change_check` (keyword-only; default `None`); keyword of `tl.attribution.occlusion_map` (keyword-only; required): Window extents over the trailing swept axes of the occluded leaf (axes between the batch axis and the swept tail are fully occluded by every window -- a `(15, 15)` window on an image occludes all channels of each patch) |
| `windowed_axes=` | constructor keyword of `tl.receptive_field.GridLayout` (required) |
| `within=` | keyword of `ReceptiveFieldView.of`: the region or list of regions the influence answer lands in (label, module address, `Op`, `Layer`, `Module`, `ModuleCall` or selector); `None` means the model outputs (projective) or inputs (receptive); keyword of `ChecksSession.register_change_check`, `ChecksSession.register_update_ratio_check`, `tl.dead`, `tl.low_variance`, `tl.saturated` (keyword-only; default `None`); keyword of `tl.attribution.read` (keyword-only; default `None`): Population scoping (same seam as every value producer): `None` = implicit eligible population; a site label / Selection = explicit contract population; keyword of `tl.changed`, `tl.pass_variance`, `tl.sign`, `tl.stable_across_passes`, `tl.threshold`, `tl.top_changed`, `tl.top_fraction`, `tl.top_k` (default `None`); keyword of `tl.debug.param_health` (keyword-only; default `None`): Optional iterable of EXACT qualified names to audit (day-1 spelling; selector sugar is a UI-sprint handoff). Unknown names refuse typed; keyword of `tl.random_like` (keyword-only; required); keyword of `tl.subspace` (required) |
| `writer=` | keyword of `tl.export.tensorboard` (required): Existing writer object, for example `SummaryWriter` |
| `writer_or_logdir=` | constructor keyword of `tl.trackers.TensorBoardSink` (required) |
| `x=` | keyword of `Trace.run` (default `None`): Forward input. If `model` is omitted, the first positional argument is treated as the new user input |
| `x_label=` | keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional x-axis label |
| `x_values=` | keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional `[K]` x coordinates. Defaults to `0..K-1` |
| `y_label=` | keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional y-axis label |
| `y_max=` | keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional upper y-axis bound |
| `y_min=` | keyword of `tl.viz.lineplot` (keyword-only; default `None`): Optional lower y-axis bound |
| `zero_param_preset=` | constructor keyword of `tl.transforms.TransformDefinition` (default `False`): Whether a BARE STRING may coerce to this transform with all-default params (memo decision 2: strings are never import paths or a parameter mini-language) |
