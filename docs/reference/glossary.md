# TorchLens glossary

This glossary fixes the public meaning of TorchLens terms. It describes the current 2.x API;
removed spellings are listed separately in [Deprecations](deprecations.md).

## Capture products

**Trace**
: The finalized record of one executed path: operations, dataflow edges, module context, selected
  payloads, and capture metadata. `tl.trace(...)` returns a `Trace`.

**Recording**
: A sparse, torch-only event stream returned by `tl.record(model, inputs, save=...)`.
  `Recording.to_trace()` materializes full graph structure; reading a payload that was not selected
  by `save=` fails explicitly. `tl.record` installs gradient hooks only when `backward_ready=True`,
  `save_grads=` or `default_grad=` is given; without one, `Recording.log_backward()` refuses with
  code `recording_backward_not_requested`.

**Saved-activation autograd history** *(knob spellings documented-unstable)*
: `CaptureOptions(detach_saved_activations=False)`, the `tl.trace` default, keeps every saved
  payload graph-connected on every save path: the save-all default, `tl.module` and
  `tl.in_module` selectors, op-label `layers_to_save`, the legacy `trace.run(model, x)` rerun,
  `trace.run(inputs=...)` and `trace.run(inputs=..., fast=True)`; `True` detaches on every one
  of them. `tl.extract(..., detach_saved_activations=False)` and
  `tl.record(..., detach_saved_activations=False)` (also `tl.fastlog.Recorder` and
  `RecordingOptions.detach_saved_activations`) attach on request; both keep their detached
  default, and `tl.record` refuses an explicit `True` with `backward_ready=True`
  (`backward_ready_conflict`). A per-site `CaptureSpec(keep_grad=True)` stays attached either
  way, and disk payloads are always detached. An escrowed deferred-selector payload spilled to
  temporary disk comes back detached, with a `RuntimeWarning` naming the count.
  `Trace.pinned_autograd_memory` discloses what the attached payloads keep alive: the bytes of
  the tensors autograd saved for every op upstream of a saved site, once per storage,
  excluding Parameters and views of them. It reads `Bytes(0)` when every payload is detached
  or the forward ran without grad, and `None` on a loaded trace (session-time, never saved).

**PartialTrace**
: A recoverable prefix of a failed `tl.trace(...)` capture. A raised capture exception may expose
  `exc.partial_log`; recover it with `tl.partial.from_failed_capture(exc)`. A partial trace is not
  silently treated as complete.

**Failed partial Recording**
: A `Recording` with `status="partial_error"` and `failed=True`. Use
  `on_forward_error="attach_partial"` to attach it as `exc.partial_recording`, or
  `on_forward_error="return_partial"` to return it. It carries string-only error metadata,
  `n_ops_completed`, and best-effort `last_event_*` fields. It cannot be passed to
  `Recording.to_trace()` or `Recording.log_backward()`.

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
  (both refuse a missing step with `record_generation_step_missing`). `random_seed=` seeds once
  for the whole generation. Beam search, more than one return sequence, batched prompts, assisted
  decoding, encoder-decoder models, `use_cache=False`, a generate that runs decode steps without a
  cache, `intervene=`, `halt=` and partial forward-error modes refuse with
  `record_generation_unsupported`, whose `reason` field names what is unsupported.

**Capture outcome**
: The frozen `Trace.outcome`, `Recording.outcome`, or `PartialTrace.outcome`. Its status is one of
  `COMPLETE`, `HALTED`, `ABORTED_NONFINITE`, `FAILED`, `UNATTESTED`, or `UNKNOWN`; failed outcomes
  also identify the phase. See [Capture outcomes](capture_outcomes.md).

**Inference-only capture**
: A capture whose forward ran under `torch.no_grad()`, so it built no autograd graph. Request it
  with `CaptureOptions(inference_only=True)`; a `tl.trace` or `tl.record` entered under
  `torch.no_grad()` or `torch.inference_mode()` implies it unless `backward_ready`, `save_grads`
  (or `default_grad` on `tl.record`, `intervention_ready` on `tl.trace`) asks for a backward, and
  an explicit `inference_only=False` keeps it off. The implied flag holds only when the forward
  really discarded its graph: a forward that turns grad back on (`with torch.enable_grad():`)
  and returns an output with autograd history is not inference-only, and backward stays
  available. An explicit `inference_only=True` is never changed. `Trace.inference_only` and
  `outcome.inference_only` report the effective value; a later `log_backward()` refuses and names
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
  `type(owner).__name__`, `Trace.root_entry_point` records
  `"bound_method:<owner_qualname>.<method>"`, and the root op record (the output-boundary ops)
  carries `tl_authored_root=True` — both persisted with fail-closed load validation. On
  bound-method episode captures `EpisodeSpec.stepped_module` defaults to the owner. Bound-method
  captures refuse rerun/append at the identity gate (`rerun_entry_point_unsupported`; an absent
  fact on a legacy artifact refuses `root_entry_point_unavailable`); replay engines are
  unaffected. Other callables (closures, bare functions) refuse `model_type_unsupported`. All
  spellings are provisional.

**Episode capture**
: One wrapped multi-step generation run captured as a single product
  (`capture_kind=episode`): `tl.trace(episode_root, x, episode=EpisodeSpec(stepped_module=...))`
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
  `torchlens.intervention.at_step(*steps)` is the step-qualified selector
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
: The user-facing grouping of equivalent or recurrent Ops. A recurrent layer may have multiple
  passes, addressed with a one-based suffix such as `linear_1:2`.

**Module**
: A captured framework module or backend-neutral module-like object. A module contains Ops but is
  not itself an Op.

**Parameter / Buffer**
: Registered model state. Parameters are trainable state; buffers are non-parameter state and may
  have multiple captured versions. A training-mode `BatchNorm2d` collapse label can say
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
  The receiver may be given by keyword (`torch.clamp_(input=p, ...)`). A frozen Parameter written
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
  owning module; keys that print alike get `#2`, `#3`, ...

**GradFn**
: A first-class live backward-autograd node captured during backward logging. Portable artifacts
  do not contain PyTorch's live autograd graph.

**Facet**
: A named semantic view into one captured value, such as an attention projection or an LSTM state.

### Documented-unstable ATen profile INDEX

The wave-0 ATen execution-profile core is gated off for ordinary callers. As of the tlspec v8
coordinated bump its fields persist (`FieldPolicy.KEEP`) and loaded profiles validate their
foreign keys whenever present; older v6/v7 artifacts contain no primitive profile. The public recorder, entity accessors, and absent-profile
refusal remain unavailable until the S2-owned capability amendment lands. The names below are the
exact surface already introduced by the gated core. Each is documented unstable and may be renamed
or removed without a compatibility alias. No validation or honesty tripwire may be weakened.

<!-- ATEN-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Facades, record kind, disclosure, and provenance | `AtenOp`, `OpRef`, `SuperAtenOp`, `primitive_op`, `mode_paused_interior`, `exact_via_aten`, `heuristic` | unstable -- no deprecation shim owed |
| Field-order contract | `PRIMITIVE_OP_FIELD_ORDER` | unstable -- no deprecation shim owed |
| Primitive fields | `label`, `sequence`, `capture_phase`, `forward_pass_index`, `backward_epoch_index`, `owner_func_call_id`, `parent_op_refs`, `parent_grad_fn_call_ref`, `owner_status`, `decomposition_slot`, `namespace`, `operator`, `overload`, `schema`, `schema_fingerprint`, `module_call_stack`, `input_tensor_facts`, `output_tensor_facts`, `mutation_kind`, `view_copy_kind`, `autocast_context`, `dispatch_key_context`, `grad_fn_ref`, `grad_fn_link_status`, `grad_fn_link_provenance`, `algorithmic_flops`, `flop_status`, `flop_formula_source`, `flop_formula_version`, `outcome`, `exception_type`, `execution_context` | unstable -- no deprecation shim owed |
| Observer-gap fields | `kind`, `capture_phase`, `sequence_before`, `sequence_after`, `owner_func_call_id`, `parent_op_refs`, `reason` | unstable -- no deprecation shim owed |
| Redundant Op-reference fields | `op_row_index`, `op_label`, `func_call_id` | unstable -- no deprecation shim owed |
| Tensor-fact fields | `container_path`, `tensor_impl_capability`, `logical_version`, `storage_alias_group`, `shape`, `stride`, `dtype`, `device`, `layout`, `requires_grad` | unstable -- no deprecation shim owed |
| Execution-context fields | `pytorch_version`, `backend`, `device_model`, `device_capability`, `grad_mode`, `inference_mode`, `module_training_summary`, `autocast`, `deterministic_algorithms`, `tf32_matmul_policy`, `sdpa_policy`, `compile_stance`, `owner_thread_coverage`, `completeness_witness_mode` | unstable -- no deprecation shim owed |
| Super comparison fields | `comparison_status`, `has_observation_gap` | unstable -- no deprecation shim owed |
| Invariant contracts | `primitive_op_invariants`, `non_torch_primitive_op_inert` | unstable -- no deprecation shim owed |
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

`AtenOp` is one value-free dispatcher call measured during a concrete capture. Its label is opaque,
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
| Integrated Gradients method and controls | `integrated_gradients`, `target`, `n_steps`, `baseline` | unstable -- no deprecation shim owed |
| Completeness evidence | `attribution_sum`, `target_delta`, `completeness_residual` | unstable -- no deprecation shim owed |
| Occlusion method and controls | `occlusion`, `selection`, `score`, `blur_kernel_size` | unstable -- no deprecation shim owed |
| Occlusion baseline policies | `zeros`, `mean`, `blur` | unstable -- no deprecation shim owed |
| Occlusion evidence | `original_score`, `occluded_score`, `selection_digest` | unstable -- no deprecation shim owed |
| CAM method and controls | `grad_cam`, `layer`, `reshape`, `relu`, `overlay`, `image`, `alpha`, `cmap` | unstable -- no deprecation shim owed |
| CAM token grid and layer-choice evidence | `grid_route`, `feature_map`, `derived`, `prefix_tokens`, `grid_evidence`, `layer_source`, `caller`, `trace`, `layer_reason` | unstable -- no deprecation shim owed |
| CAM resolution evidence | `native_map_resolution`, `rendered_map_resolution`, `upsampling`, `bilinear_display_only` | unstable -- no deprecation shim owed |
| Graph overlay bridge and controls | `overlay`, `source`, `reduce`, `abs_sum`, `abs_mean`, `sum`, `max` | unstable -- no deprecation shim owed |

| F06 wrapping contract | `noise_tunnel`, `attribute`, `method`, `method_kwargs` | unstable -- no deprecation shim owed |
| F06 noise tunnel controls and evidence | `n_samples`, `stdevs`, `aggregation`, `noise_bank`, `mean_square`, `variance`, `stdevs_resolved`, `residual_of_means` | unstable -- no deprecation shim owed |
| F06 GradientShap controls | `gradient_shap`, `baselines`, `draw_bank`, `store_draws` | unstable -- no deprecation shim owed |
| F06 guided methods and evidence | `guided_backprop`, `deconvolution`, `sites`, `absolute`, `site_census` | unstable -- no deprecation shim owed |
| F06 occlusion map controls | `occlusion_map`, `window`, `strides`, `baseline_value`, `occlude_leaf`, `max_passes`, `average` | unstable -- no deprecation shim owed |
| F06 IG step runner controls and evidence | `step_batch_size`, `step_audit`, `step_audit_seed`, `per_call`, `per_chunk`, `path_evaluations_logical`, `physical_forward_calls` | unstable -- no deprecation shim owed |
| F06 completeness disclosure | `residual_rel`, `target_delta_abs`, `completeness_caveat` | unstable -- no deprecation shim owed |
| F06 metrics | `infidelity`, `sensitivity`, `MetricResult`, `perturb`, `noise_std`, `radius`, `gaussian`, `square_removal` | unstable -- no deprecation shim owed |
| F06 token attribution | `text`, `TokenAttributionResult`, `TokenAttributionPayload`, `keep_special_tokens`, `steps_per_batch`, `max_length`, `pad_token`, `auto`, `converged`, `show` | unstable -- no deprecation shim owed |
| F06 LRP mechanism | `SiteStash`, `stash`, `fetch`, `mark_firing`, `leftovers`, `register_site`, `label_of` | unstable -- no deprecation shim owed |
| Site targets | `SiteTarget`, `unit`, `call` | unstable -- no deprecation shim owed |
| Integer-input evidence | `integer_input_layers` | unstable -- no deprecation shim owed |
| Evaluation curves | `deletion_curve`, `insertion_curve`, `CurveResult`, `CurveStep`, `fractions`, `scores`, `area`, `aopc`, `steps`, `features`, `patch_size`, `auto`, `elements`, `pixels`, `patches`, `tokens`, `n_features`, `index`, `fraction`, `inputs`, `input_kwargs`, `baseline_mask`, `intervention` | unstable -- no deprecation shim owed |
| Randomization test | `randomization_test`, `RandomizationResult`, `RandomizationStep`, `randomization`, `cascading`, `independent`, `layers`, `seed`, `keep_maps`, `original_attribution`, `randomized`, `similarity`, `similarities`, `spearman`, `spearman_abs`, `ssim`, `values` | unstable -- no deprecation shim owed |
| ROAD curve | `road_curve`, `ranking`, `most_relevant_first`, `least_relevant_first`, `noise_std`, `seed`, `road` | unstable -- no deprecation shim owed |
| Causal audit | `causal_audit`, `CausalAuditResult`, `CausalAuditUnit`, `CausalAuditSet`, `claim`, `k`, `ablation`, `zero`, `mean`, `controls`, `top`, `bottom`, `random`, `random_draws`, `per_unit`, `layer`, `site`, `original_score`, `units`, `sets`, `rank_agreement`, `set_of`, `unit`, `claimed_effect`, `measured_effect`, `kind`, `draw`, `intervention`, `component`, `claim_kind`, `constant_row`, `rows_not_audited`, `corrupted`, `metric`, `edges`, `edge_table`, `edge_patch`, `example_reduction`, `logits_site`, `sets_not_measured`, `reason`, `trace` | unstable -- no deprecation shim owed |
| Attribution graphs | `build_graph`, `AttributionGraph`, `GraphNode`, `transcoders`, `embedding`, `target`, `max_n_logits`, `desired_logit_prob`, `demeaned_logits`, `logits_at`, `frozen`, `nodes`, `adjacency_matrix`, `node_values`, `constants`, `encoder_bias`, `decoder_bias`, `other`, `active_features`, `activation_values`, `transcoder_sites`, `error_sites`, `error_vectors`, `input_tokens`, `logit_tokens`, `logit_probabilities`, `capture_outcome`, `disclosures`, `node_counts`, `node_indices`, `kind`, `site`, `position`, `feature`, `token`, `logit`, `error` | unstable -- no deprecation shim owed |
| Attribution graph export | `to_neuronpedia`, `slug`, `scan`, `tokenizer`, `prompt`, `node_threshold`, `edge_threshold`, `path` | unstable -- no deprecation shim owed |
| Attribution graph overlay and upload | `graph_value`, `influence`, `activation`, `upload_to_neuronpedia`, `api_key` | unstable -- no deprecation shim owed |

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
from `tl.mechinterp.edge_scores` is audited by patching each edge alone with the `corrupted` inputs and
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
| Factored low-rank algebra (vendored from TransformerLens, MIT) | `FactoredMatrix`, `A`, `B`, `AB`, `BA`, `T`, `svd`, `U`, `S`, `V`, `eigenvalues`, `norm`, `make_even`, `get_corner`, `collapse_l`, `collapse_r`, `unsqueeze`, `pair`, `ldim`, `mdim`, `rdim`, `ndim` | unstable -- no deprecation shim owed |
| QK and OV circuits on verified head weight views | `qk_circuit`, `ov_circuit` | unstable -- no deprecation shim owed |
| Per-neuron MLP rows under the identity gate | `mlp_neuron_contributions`, `layers`, `neurons`, `onto`, `positions` | unstable -- no deprecation shim owed |
| Alias table generation value | `generation="4.x"` (TransformerBridge canonical hook names) | unstable -- no deprecation shim owed |
| Appliance on the `trace.using` door | `trace.using("mi")`, `head_scores`, `qk_circuit`, `ov_circuit` | unstable -- no deprecation shim owed |
| Per-query-head patching under grouped-query attention | `patch_query_heads_grid`, `site`, `heads`, `effect_site` | unstable -- no deprecation shim owed |
| Component edge graph (heads and MLPs as nodes, q/k/v split receivers) | `component_graph`, `ComponentGraph`, `ComponentNode`, `ComponentReceiver`, `ComponentEdge`, `nodes`, `receivers`, `edges`, `layer_index`, `head`, `senders`, `contribution`, `identity_receipt`, `input_label`, `norm_address`, `module_address`, `patch_edges`, `corrupt` | unstable -- no deprecation shim owed |

| Edge attribution patching over the component graph | `edge_scores`, `corrupted`, `metric`, `method`, `eap`, `eap_ig`, `steps`, `example_reduction`, `mean`, `sum` | unstable -- no deprecation shim owed |
| Circuit faithfulness areas | `circuit_performance_ratio`, `circuit_model_difference`, `faithfulness`, `fractions`, `log_scale` | unstable -- no deprecation shim owed |
| Fixed-period induction score | `FixedPeriodInduction`, `head_scores(..., kind=FixedPeriodInduction(period=n))`, `period` | unstable -- no deprecation shim owed |
<!-- MECHINTERP-KIT-UNSTABLE-INDEX:END -->

`component_graph(trace)` returns a `ComponentGraph`: read-only `nodes`, `receivers` and `edges`
keyed by MIB/EAP-IG vocabulary name (`input`, `a{L}.h{H}`, `b{L}`, `m{L}`; receivers
`a{L}.h{H}<q|k|v>`, `m{L}`, `logits`; edges `"<sender>-><receiver>"`). Each `ComponentReceiver`
holds the captured pre-norm input (`value`, from the op `input_label` entering the norm at
`norm_address`) and its `senders`. Nodes and receivers carry `layer_index` (transformer block
index) and `head` (head index), each `None` where it does not apply. `ComponentNode.module_address`
is the owning module (embedding, attention or MLP; the attention module for head and bias nodes)
and `ComponentReceiver.module_address` the reading module (attention, MLP, or the unembedding
head). Each `ComponentEdge` exposes
the sender's output as its `contribution`. `identity_receipt` records the in-API check that every receiver input equals the
sum of its contributions. Each attention output bias is its own node `b{L}`. A `ComponentEdge` is
also a selection (`__selection__`): an EDGE selection of the pre-norm input edge for `m{L}` and
`logits` receivers, an ACT selection of one head's q, k or v projection slice otherwise.
`patch_edges(trace, edges, *, corrupt)` returns a fork of `trace` in which every receiver the
edges enter reads `input - sum(s) + sum(s_corrupt)` over its patched edges (`corrupt` is the
corrupted input's `ComponentGraph`); `input` and `s` are the patched forward's own values (the
clean values when one receiver is patched), so a circuit (everything outside a kept set patched)
is one call; the trace needs `intervention_ready=True`.

`edge_scores(model, inputs, *, corrupted, metric, method="eap_ig", steps=None,
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
shape `[n_q_heads, d_model, d_model]` built from `head_weight_views` (TransformerLens's
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
`site`, `n_q_heads`, `n_kv_heads` and `expansion_ops`, and every cell fires exactly once. As for
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
  substrings, `'all'`, `'none'`), which selects exactly what
  `capture=CaptureOptions(layers_to_save=[...])` selects; passing both refuses
  `save_selection_conflict`. `save="stats"` is a third level beside `'all'` and `'none'`:
  reduce and discard, one `tl.stats.Spine` per op and no payload (see Streaming
  statistics).

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
  `TorchLensIOError` and marks the temp bundle `PARTIAL`, and finalization waits for every
  pending write before the bundle publishes. `tl.record` streaming stays synchronous and
  refuses an explicit `async_writes=True`.

**Capture cache**
: The opt-in `cache=True` content-hash store of finished captures. A hit requires the model's
  tensor content (including device and `requires_grad`), training flags, module tree and
  `forward` code, plain instance attributes, user-registered module hooks, inputs, and capture
  configuration to all match. Entries are HMAC-authenticated, bounded by entry count and bytes,
  and `tl.clear_capture_cache()` empties the cache while preserving its secret.

## Replay and intervention

**Intervention**
: A value or gradient edit performed at a selected live site. `tl.when(predicate, helper)` binds a
  selector to helpers such as `tl.zero_ablate()` or `tl.scale(...)`. `spec.bind(model)` returns a
  bound executor that applies a spec to live calls. A synchronous generator or iterator returned
  by a bound call (for example `bound.generate(...)`) stays steered for every step, and its report
  settles on `last_report` when the iteration is exhausted, closed or raises; an async lazy output
  (async generator or iterator, coroutine, awaitable) refuses `bind_lazy_output`, as does a lazy
  `generate` output under `torchlens.intervention.steer_generate`, whose report must settle
  before it returns.

**Zero-fire policy (`on_zero_fire`)** *(spelling documented-unstable)*
: One knob on every intervention door: `spec.bind`, `torchlens.intervention.steer_generate`,
  `tl.trace(intervene=)` and `Trace.run`. An intervention that fires at zero sites is always
  recorded on the door's own record (the bind report's `zero_fire_rule_ids`, the capture's
  `annotations["unmatched_capture_selectors"]` and intervention audit, the rerun's
  `last_run["hooks_unfired"]`). The knob then decides the same way everywhere: `"warn"`
  (default) emits the door's coded warning (`bind_zero_fire`, `trace_zero_fire`,
  `rerun_zero_fire`), `"error"` raises the door's typed refusal with that code, and
  `"disclose"` records only. Any other value refuses `zero_fire_policy_invalid`. A rerun edit
  that cannot fire live (`rerun_selector_unmatchable_live`) is refused before the policy runs,
  whatever the knob says.

**Bound intervention executor** *(spelling documented-unstable)*
: `spec.bind(model)` pairs an `InterventionSpec` with a base model as a capture-free callable.
  It runs the model with the spec's edits held live, returns the model's own output, records
  only a firing report (`.last_report`), and supports HF `generate` with the KV cache on or off.
  It is not an `nn.Module` and bindings never nest; a rule that never fires is recorded on
  `.last_report` and settles by `on_zero_fire` (see **Zero-fire policy**). `torchlens.intervention.steer_generate` is one-call sugar for
  `spec.bind(model).generate(...)`. Decode steps: the binding counts root forwards, so
  `torchlens.intervention.at_step(k)` on a bound call or `generate` names the k-th forward
  (step 0 is the prefill), on op-level rules and as `tl.module(address) & at_step(...)`
  boundary rules, capture-free; outside a binding or an episode capture the qualifier still
  refuses `episode_step_selector_without_episode`. Gradient edits (`tl.grad_zero`,
  `tl.grad_scale`, `tl.bwd_hook`, ...) refuse `bind_rule_unsupported`: the lane replaces forward
  values only; edit gradients on a `backward_ready` trace and call `trace.backward(loss)`.

**torchlens.intervention.subspace_ablate(basis, *, mode="zero", feature_axis=-1, mean=None, seed=None)** — *documented-unstable*
: Edit acting on the component of a site value inside the span of a `[k, d]` basis (rows need
  not be orthonormal; a rank-deficient or non-finite basis refuses `subspace_basis_invalid`).
  With `P` the orthogonal projector onto the span: `mode="zero"` gives `x - P x` (the
  k-dimensional `project_off`), `mode="mean"` gives `x - P x + P mean`, and `mode="resample"`
  gives `x - P x + P x[perm]` for a seeded batch permutation (`seed=` required). Inconsistent
  mode arguments refuse `subspace_mode_invalid`; a basis over the wrong width refuses
  `subspace_basis_dim_mismatch` at fire time. Zero mode on the span of class-mean differences
  is the LEACE mean-equalizing erasure.

**torchlens.intervention.affine_source(weight, *, bias=None)** — *documented-unstable*
: Source verb for a subspace edit: under `do(tl.subspace(site, R.T, origin=...), affine_source(W,
  bias=b))` the new coordinates are `W x + b` of the whole site value `x`, so the site gets
  `x + Q^T (W x + b - Q x)` with `Q` the basis rows (LoReFT, `h + R (W h + b - R^T h)`). `weight` is
  a finite real `[k, d]` tensor shaped like the basis rows, `bias` `None` or a finite real `[k]`
  tensor; anything else refuses `affine_source_invalid`. Outside a subspace (a site label,
  `tl.units`, a capture or replay hook) it refuses `affine_source_needs_subspace`. Builtin: it
  saves inside the `subspace_edit` payload, `weight` and `bias` as tensor sidecars. TorchLens
  applies a LoReFT edit trained elsewhere and never fits one.

**torchlens.intervention.subspace_interchange(basis, source, *, feature_axis=-1)** — *documented-unstable*
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

**InterventionDecision.keep_grad** — *documented-unstable*
: `True` runs the same check as `torchlens.intervention.keep_grad(edit)` on the decision's hook
  (`edit_detached_from_graph` at fire time when the written values carry no graph). `False`, the
  default, checks nothing. MLX refuses `keep_grad=True`.

**torchlens.intervention.at_positions(edit, positions, *, mask=None, assume_no_padding=False, axis=1)** — *documented-unstable*
: Restricts a same-shape edit to chosen token positions of a `(batch, tokens, ...)` site value
  (token-position editing): `positions` is an int, a list of ints, a slice, a per-row
  `(batch,)` index tensor, or `"first"` / `"last"`, the first or last REAL token of each row
  resolved from `mask=` under the token-pooling mask policy (no mask and no
  `assume_no_padding=True` refuses `transform_mask_missing`; a mask that does not describe the
  value refuses `transform_mask_incompatible`). Unsupported spellings refuse
  `position_edit_invalid`; positions past the sequence refuse `position_edit_out_of_range`.

**torchlens.intervention.direction_from_pairs(pos_traces, neg_traces, site, method="mean_diff", *, read_token_index=-1)** — *documented-unstable*
: Builds one contrastive steering direction from saved activations. `pos_traces` and
  `neg_traces` are a trace or a sequence of traces with one prompt per batch row; row `i` of the
  negatives pairs with row `i` of the positives. One token per prompt is read at `site`
  (`-1` is the last real token when the trace saved its `attention_mask` input; a sequence gives
  one position per prompt; `None` reads `[n, hidden]` outs unsliced). With `D = pos - neg`:
  `"mean_diff"` is `D.mean(0)` on the activation scale; `"pca"` is the first uncentered
  principal axis of `D`, unit norm, signed toward the positives; `"logistic"` is L2 logistic
  regression (`C = 1`, no intercept) of `D / 2` against `-D / 2`, solved with torch only, unit
  norm. Returns a `Direction`. Refusals use the `direction_*` codes of the error contract.

**Direction (`torchlens.intervention.Direction`)** — *documented-unstable*
: The frozen result of `direction_from_pairs`: `vector` (1-D, the rows' dtype), `site`,
  `method`, `n_pairs`, `read_token_index`, `model_type` (the traced model's
  `config.model_type`, or `None`) and `provenance`, a
  `torchlens.selection_subspace.BasisProvenance` whose origin records the site, method, pair
  count, read position and a sha256 digest of the rows, and whose `sha256` is the digest of
  the canonical float64 vector. `direction.steer(m)` is `tl.steer(vector, m, feature_axis=-1)`;
  place it with `tl.when(tl.module(direction.site), ...)`. `direction.to_steering_vectors()`,
  `direction.to_repeng()` and `direction.to_dialz()` convert it to a real
  `steering_vectors.SteeringVector`, `repeng.ControlVector` and `dialz.SteeringVector` without
  recomputing it (thin converters `tl.bridge.steering_vectors.from_direction`,
  `tl.bridge.repeng.from_direction` and `tl.bridge.dialz.from_direction`), keyed by
  `direction.library_layer(layer=None)`: the site label's trailing index unless `layer=` is
  given; a different explicit index refuses `direction_layer_mismatch`. Applied with the same
  multiplier, each library object gives the native steer's logits.

**`Direction.upload_to_neuronpedia` (publish a vector)** — *documented-unstable*
: `direction.upload_to_neuronpedia(*, api_key, scan, vector_label, layer=None,
  hook_type="resid_post", steer_strength=1.0)` publishes the vector on neuronpedia.org under
  the account that owns `api_key` through Neuronpedia's "Create Vector" endpoint and returns
  the vector page's URL. `scan` is Neuronpedia's model id, `vector_label` the name it shows,
  `layer` defaults to `direction.library_layer()`, `hook_type` is `"resid_post"` or
  `"resid_pre"`, `steer_strength` the default steering strength (-100 to 100). Every field
  is checked before the request (`neuronpedia_upload_argument_invalid`); the key never
  enters a message.

**Bundle**
: A named collection of aligned Traces, constructed with `tl.bundle(...)` or `tl.Bundle(...)`, for
  cross-run comparison. `tl.Bundle()` with no members builds an empty bundle to fill later with
  `add`; an empty bundle saves and loads like any other.

**Bundle.member_relations** *(spelling documented-unstable)*
: The bundle's typed, directed relation rows between members, read as an immutable tuple of
  `MemberRelationRow` (plus `OpaqueRelationRow` for unknown namespaced kinds loaded from an
  artifact). Pair kinds: `alternative_of` (peers; still comparable), `forked_from`,
  `successor_of`, `escalates`, `fed_by`; member kind: `episode_member`. Each kind's params are a
  closed set; rows persist in `bundle.json` and every load re-checks them. On a pair row `from` is
  always the LATER member. Rows are ordering and topology claims: members joined by a directed
  path of ordering kinds are not comparison operands.

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
  prompt is not `fed_by`; record that order with `successor_of` and keep the route decision in
  your own metadata (TorchLens has no control-dependency kind). Optional params: `token_spans`
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
  (`fed_by`, `successor_of`, `forked_from`, `escalates`). Bundle helper-table methods, like
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
  `bundle_relation_span_unverified`. A Bundle helper-table method; any reduction over a
  many-to-one group is the caller's own descriptive comparison.

**Bundle.node** *(members= spelling documented-unstable)*
: `b.node(site, members=None)` returns a `SuperOp` view of one site across the members, or across
  the named subset, gated pairwise inside that subset (`same_param_shapes` floor), so one foreign
  member no longer disables the view for the rest.

**Bundle delta family** *(`delta_map`, `norm_delta`, `output_delta`, `compare`)*
: Per-node and per-output distances from a baseline member (the configured baseline, else the
  first member). Every (baseline, member) pair goes through the comparison gate first, the same
  predicates as `compare_at`: no ordering path between them, a shared graph, and proven-equal
  input values; otherwise the read refuses with the gate's `bundle_gate_*` code.

**Bundle.derive_episode_status** *(spelling documented-unstable)*
: `b.derive_episode_status(episode_id, *, ledger=None)` folds the outcomes of the members named by
  `episode_member` rows into a derived episode status (`EpisodeFoldResult`). A derivation, never
  a stored Bundle outcome; without a ledger the ledger-only arms degrade to `episode_unknown`.

**Bundle.why / Bundle.provenance** *(spelling documented-unstable)*
: `b.why(member, relative_to=None)` returns a `WhyReport`: the exact-or-refused ancestry of a
  member against the baseline (or `relative_to`), with lineage status, payload fidelity, value
  residual and comparability reported as separate axes. `b.provenance()` is the per-member list of
  the same facts. Both are derived views; nothing is stored. Reference:
  `docs/reference/experiment_ledger.md`.

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
  value/difference/trajectory reads on a Bundle (`weight_norm_diff`, `diff_pair`, `aggregate`,
  `out`/`grad` on `SuperParam` views with two or more members) refuse BEFORE tensor lookup with
  the stable code `checkpoint_series_live_params` unless every member carries immutable
  capture-time parameter evidence. The guard is keyed on the claim, never on Python object
  identity, and survives save/load; a version-axis relation row still orders members.

**Runnable artifact**
: A `.tlspec` saved at `level="runnable"` with a sparse taken-path descriptor. Its source capture
  must use `intervention_ready=True`; this flag supplies replay templates even when no intervention
  is planned.

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
: The machine-readable `Trace.capture_verification_reason` explaining why a live capture is not
  positively verified, for example `mode_rescue_rerun`, `escape_rescue_unrecovered`,
  `owner_thread_tripwire_changed`, or `dynamo_region_not_logged`.

**Pristine (`Trace.pristine`)** *(spelling documented-unstable)*
: Whether nothing ever intervened on the trace, read from the evidence it carries: `True`,
  `False`, or `None`. It is derived at read time from records that persist with the trace (the
  intervention audit, the operation history, mutation states, direct writes and op intervention
  marks), never stored as a flag, so it survives `tl.save`/`tl.load`, pickle and `fork()`, and no
  reset makes a touched trace read `True` again. `False` means some touch was recorded: a
  capture-time `intervene=` (including one that fired nowhere), a staged hook or value
  replacement (even after `clear_hooks()`), `do()`, `set()`, a region edit, a hooked rerun or run,
  a direct write. Plain reruns, plain runs, forks and `clear_hooks()` on an untouched trace are
  not touches. `None` means no evidence: the artifact predates tlspec 9 or the backend is not
  torch, and it is never a pristine reading. Bundle's pristine label and baseline
  auto-detection count a member only when it reads `True`. It is a reading, not a verdict: it
  never changes validation or a run report. Not `Trace.state`: `TraceState.PRISTINE` is an
  operational state that an assignment can set and that a func-targeted `intervene=` capture
  keeps. Not validation's pristine-torch oracle, which means running on unwrapped torch
  functions.

**Rescue re-run**
: A single disclosed second forward under a `TorchFunctionMode` net that redirects detected stale
  pre-wrap torch references to their wrappers. The trace records `trace.rescue_rerun` and reason
  `mode_rescue_rerun`.

**Collapse**
: A rendering-only condensation of module detail. `collapse="none"`, `"auto"`, `"max"`, or a
  float in `[0, 1]` never changes the underlying Trace. Admission gates on the rendered
  universe U against the defensive constant `COLLAPSE_OPTIMIZER_MAX_OPS` (2000) plus a
  measured (U, W) work estimator: over-budget requests degrade to a deterministic compact
  fallback plan (coded `collapse_budget_fallback`), never an uncollapsed wall; only
  pathological inputs (raw ops above 20x the constant) decline outright
  (`collapse_pathological_skip`; `Trace.collapse_plan()` refuses typed
  (`collapse_plan_unavailable`); `Trace.collapse_schedule()` degrades to its single
  full-graph step).

**Module.collapse_score / Trace.module_collapse_order** — *unstable — no deprecation shim owed*
: A module's smart-collapse priority: its hidden ops (distinct subtree ops minus one) over the
  largest submodule's, rounded to 6 places; 0.0 for the root, a module of at most one Layer, or one
  whose single child hides the same ops. It ranks candidates and never reads the selected plan
  (`Trace.collapse_plan()` may skip a high-priority module); `module_collapse_order` lists every
  module's priority, highest first. Computed at access, never persisted.

**Run folding**
: `fold_repeats=True` replaces eligible runs of distinct same-class sibling modules with a
  representative and an honest `+N more` label. It differs from recurrence, where `(xN)` means the
  same parameters executed repeatedly.

**Receptive field / Projective field**
: Influence geometry through the captured DAG toward inputs / toward outputs. Entity-level
  `op.receptive_field` and `op.projective_field` pair with Trace-level
  `Trace.receptive_fields()` and `Trace.projective_fields()` tables.

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
  callable refuses with the original exception chained (`encoding_callable_error`).

**size_by** *(unstable — no deprecation shim owed; keyword-only)*
: `draw(size_by=...)` sizes eligible operation nodes from a scalar field, the closed `"dims"`
  shape token (numel of the non-batch output shape — the D4 default mapping, applied because D4
  is unruled), or a callable `node -> scalar`. Emitted sizes are width/height MINIMUMS under
  `fixedsize=false`: a label can never be truncated by an encoding and fonts never scale; the
  encoded area is clamped to 4x the default node area. Strictly opt-in — plain `draw()` keeps
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
  (`pass_index` on multi-pass ops only) under the LOCKSTEP LICENSE: granted iff the pass_index
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
  then a channel-only disclosure legend), `True` (full theme legend, plus channel rows when
  active; its `mutated parameter` row appears only when the render draws one), and `False` (no legend, honored even with channels active — the encoding is then
  undisclosed). The `None` value is *(unstable — no deprecation shim owed)* pending ratification.

## Backend-neutral identity

**Backend**
: `Trace.backend`, such as `torch`, `jax`, `mlx`, `tinygrad`, `paddle`, or `tf`.

**Module identity mode**
: `Trace.module_identity_mode`, describing how the backend identifies module-like ownership.

**Backend address / resolver status**
: Portable origin and resolution fields for backend records. They travel with `param_source`,
  `dtype_ref`, and `device_ref`; none should be inferred from a torch-only object when a
  backend-neutral field exists.

## Site selectors

**Label selectors**
: `tl.label` (exact final label), `tl.contains` (label substring), and `tl.regex` (label
  regex pattern) select ops by their public labels. `tl.where` filters a table with a
  predicate callable.

**Structural selectors**
: `tl.func` / `tl.module` / `tl.in_module` select by callable or module context;
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
  module;
  `tl.func_transform` selects `torch.func` and `torch.autograd` transform boundary ops
  (including `autogradgrad`, an in-forward `torch.autograd.grad` call, with
  `transform_kind="autograd.grad"`); `tl.output` selects model outputs, and
  `tl.output_at` / `tl.input_at` select a nested output or model-input path.

**Temporal composition**
: `tl.followed_by` (retroactive successor) and `tl.preceded_by` (lookback predecessor)
  compose with boolean operators to form temporal predicates over the op stream.

**Backward selectors**
: `tl.grad_fn` (backward grad_fn), `tl.grad_fn_label` (exact grad_fn label),
  `tl.grad_input` / `tl.grad_output` (backward event tensors), `tl.in_backward_pass`
  (one backward pass number), and `tl.without_op` (grad_fns without a paired forward op;
  the former `intervening` alias is removed).

## Intervention helpers

**Value helpers**
: `tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`, `tl.noise` (Gaussian noise),
  `tl.mean_ablate` (replace with a source mean), `torchlens.intervention.scramble_elements`
  (elementwise iid scramble from a flattened source -- the honest rename of
  the removed `resample_ablate` spelling, which a saved spec may not name either: it refuses
  `intervention_helper_renamed`; for coherent donor patching use
  `tl.patch_from`), `tl.replace_with` (fixed value), `tl.swap_with` (another
  site's tensor), `tl.steer` (add a scaled steering direction), `tl.project_onto` /
  `tl.project_off` (keep or remove the component along a direction), and
  `tl.splice_module` (call a module as a black-box forward splice). Availability outside
  the torch backend is narrower; see the per-backend rosters in the backends guide.
  Tensor-carrying helpers (`tl.steer`, `tl.mean_ablate`, `tl.project_onto`, `tl.project_off`,
  `tl.swap_with`) keep the caller's tensor by reference. Each staged entry records a
  full-content digest of those tensors when it is staged; a rerun and `save_intervention`
  refuse `helper_tensor_changed_since_capture` once the tensor changed (in place or through
  `.data`). Re-stage the edit to use the new value. A bound executor (`spec.bind(model)`) reads
  the helper's tensor at each call; in-place changes apply to the next call, and
  `BindReport.helper_tensor_versions` (on `.last_report`) records each tensor's version counter,
  keyed by the rule id the call started with (a rule id follows its tensor's content).

**Backward helpers**
: `tl.bwd_hook` builds a live/rerun-only backward hook; `tl.grad_zero`, `tl.grad_scale`,
  `tl.grad_clamp`, `tl.grad_clip`, and `tl.grad_noise` edit gradient tensors during the
  backward pass.

**Replay verbs**
: `tl.do` applies a one-shot intervention to a captured log; `tl.push` pushes an edit
  downstream through the recorded graph (DAG replay) and `tl.push_from` pushes from a
  pre-mutated site; `tl.run` performs a full-forward run with the log's active
  intervention spec; `tl.sweep` captures one intervened trace per swept replacement
  value. (The former `replay`/`replay_from`/`rerun` aliases are removed.) The replay engine operates on pass-qualified op labels
  (`label:pass`): cone traversal, the replay overlay, hook targets, and commits are
  keyed per pass, so edits on multi-pass (recurrence-grouped) layers touch exactly the
  addressed pass and recompute every downstream pass; a bare label naming a multi-pass
  layer refuses `multipass_bare_label_ambiguous` (single-pass bare labels stay
  accepted). Replay disclosures (`last_run` origins/cone, `replay_frontier` keys) spell
  multi-pass ops pass-qualified and keep bare labels for single-pass layers.
  Engines treat the trace's staged spec as read-only: a rerun (plain, append, or
  chunked), a failed or interrupted rerun, and a fork leave it exactly as staged.
  A capture-time `intervene=` predicate that is not lowered to module hooks stages
  one entry per fired op on its final label; `tl.run` re-arms the trace's retained
  predicate through the capture door instead, and refuses
  `rerun_predicate_restage_mismatch` (trace unchanged) when the predicate fires at
  different ops than the staged entries name, or at the same ops with changed rule content
  (a staged steer edited since capture). A staged edit addressed by a finalized label
  (`Trace.set(label, ...)`, `attach_hooks(tl.label(...))`) can never fire during a live
  forward, so a rerun refuses it up front with `rerun_staged_label_unmatchable` (trace
  unchanged). A staged edit whose selector matches sites on the trace but fires at none
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
  entry has never fired in the trace's recorded graph, as after `attach_hooks()` on a plain
  capture, or when the graph records a fire no staged entry accounts for, as after
  `clear_hooks()` on a steered capture, so the capture engine reruns once to record the current
  plan, or
  `<divergence code>:fast_live_call_fingerprint` for a structural change); `hooks_fired` / `hooks_unfired` count the staged plans that fired, and a plan that
  never fired settles by `on_zero_fire` (default: warns `rerun_zero_fire`). Through
  `run(inputs=..., fast=True)` a run that applies staged edits reports `unverifiable`, never
  `verified` (diagnostic `user_intervention_not_replayable`, stage `hooked_run_unverified`):
  it is an edited forward, and no hook-aware check exists yet. Output leaves (the model's and each saved module's) are paired
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
  unsaved, `raw_output` follows the capture's output-transform rule, and the stored spec is
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
  (``output_dir=``) writes a SELF-DESCRIBING artifact: atomic ``batch_XXXXX.pt`` shards
  plus a ``manifest.json`` recording the run signature, per-site identity (layer label
  and structural site key where derivable), stimulus ordering/provenance
  (``stimulus_ids=`` optional), axis semantics, dtypes, devices, and the TorchLens
  version. ``resume=True`` (DOCUMENTED-UNSTABLE, like ``stimulus_ids=`` and the loader)
  continues an interrupted disk run from its last completed shard after a signature
  check; `torchlens.dataset_extraction.load_extraction` reads the artifact back with
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
  `harvest_address_class_ungated`, and every helper must declare `batch_independent=True`.
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
  `position`, `n_tokens`); padding is never a token row.
  `torchlens.dataset_extraction.shuffled_batches(reader, batch_size, *, seed=0,
  keys=None, buffer_shards=1, tokens=False)` shuffles across `buffer_shards` shards at
  once (the cross-shard mixing buffer) and yields token rows when `tokens=True`.

**Observers**
: `tl.tap` creates a tap observer for a site; `tl.span` records a named observer span
  around captures or hook execution (the former `record_span` alias is removed);
  `tl.record_kpi_in_graph` records a user KPI on the active capture graph;
  `tl.register_tensor_connection` registers a manual parent-child tensor edge during
  capture; `tl.decide_recording_of_batch` retroactively keeps or discards a captured
  batch log.

**Validation entry**
: `tl.validate(model, x, scope=...)` validates a model/input pair for a requested scope
  (for example `"saved"` or `"receptive_field"`), capturing what it needs itself.
  Keyword-only `output_device` and `save_budget` set the validator capture's save options,
  with the same values and defaults as `CaptureOptions.output_device` (`"same"`) and
  `CaptureOptions.save_budget` (`"auto"`): `output_device="cpu"` lets a GPU model validate
  with its saved activations in host memory (replay still runs each op on its original
  device), and an exceeded budget raises `SaveBudgetExceededError` as the capture does.
  Under `output_device="cpu"` saved function arguments (the replay inputs) other than
  parameters are kept in host memory too and moved back to each op's device for its
  check, so the GPU keeps only the parameters and one forward's transients. They
  apply to the `"forward"`, `"saved"`, and `"intervention"` scopes; a non-default value on
  `"backward"` or `"receptive_field"` raises `TypeError`. The lower-level
  `torchlens.validation.validate_forward_pass` accepts the same two keywords.

**Session admin**
: `tl.release_model` evicts TorchLens's module metadata and preparation bookkeeping for a
  traced model and normalizes plain attributes holding torch functions from the other wrap
  epoch, so whole-model pickle / `torch.save` survive later wrap-state changes (a capture
  itself leaves no TorchLens callable on the model: submodule `forward` wrappers are
  session-scoped); `tl.clear_capture_cache`
  empties the capture cache; `tl.io.list_logs` / `tl.io.reset_naming_counter` manage log
  bookkeeping.

## Persistence, containers, and namespaces

**Save / load**
: `tl.save` persists a `Trace` into a portable `.tlspec` directory bundle at a chosen
  level; `tl.load` loads a `.tlspec` object with eager tensor materialization.
  The staged intervention spec is session-only: `tl.save` keeps the intervened
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
  `tl.PayloadLoadHints` carries backend-specific payload materialization hints
  (`tl.JaxPayloadLoadHint` is the JAX-specific form).

**Options**
: `tl.options` groups the public option dataclasses: `CaptureOptions`, `SaveOptions`,
  `VisualizationOptions`, `ReplayOptions`, `InterventionOptions`, `StreamingOptions`.

**Structural hash**
: `tl.hash` is the provisional structural-hash namespace; `tl.assert_unchanged` asserts
  a model still matches a pinned address-free structural hash.

**Container registration**
: `tl.register_container` registers a custom container type (flatten/unflatten pair) for
  capture and reconstruction. `tl.Container` is the computed view over a captured Python
  output container.

**Capture-product bases**
: `tl.CapturedRun` is the shared base for uncooked and cooked capture projections;
  `tl.ActivationLookup` is the protocol for raw-label/pass/address activation lookup
  consumers.

**Namespaces**
: `tl.fastlog` is the sparse predicate-recording namespace behind `tl.record`;
  `tl.facets` is the semantic facet namespace (canonical home `torchlens.semantic`);
  `tl.export` holds static export helpers; `tl.bundle(...)` / `tl.Bundle` build aligned
  Trace collections and `tl.show_bundle_graph` renders a bundle's graph:
  `nodes="ops"` (default) the union op graph, `nodes="members"` (spelling documented-unstable)
  one node per member and one edge per relation row, from the earlier member to the later one,
  labelled with the kind, the row's `label` and, on `fed_by`, its basis and coverage; solid edges
  carry token spans TorchLens checked, every other row is dashed. `tl.load(path, lazy=True)` on a
  saved bundle passes `lazy` (with `map_location` and `materialize_nested`) to every member.

**Appliance door (`Trace.using`)** *(spelling documented-unstable)*
: `trace.using(name)` returns the appliance registered as `name`, bound to that trace (for
  example `trace.using("mi").head_scores(...)`; TorchLens registers `"mi"`, the mechinterp
  kit, whose methods return exactly what the matching `torchlens.mechinterp` functions return).
  Appliances hang off this one door instead of adding methods to `Trace`; TorchLens's own and
  outside packages register the same way, with
  `torchlens.accessors.register_appliance(name, factory, summary="", replace=False)`, where
  `factory` is a callable taking the trace or a lazily imported `"module:attribute"` path.
  `torchlens.accessors.registered_appliances()` lists the names with their summaries. An
  unknown name refuses `appliance_unknown` and lists the registered names; re-registering a
  name without `replace=True` refuses `appliance_name_taken`; a malformed name or factory
  refuses `appliance_registration_invalid`. The word clashes with `tl.semantic.using(*recipes)`
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
  `tl.stats.Spine`) built on the `tl.stats.StreamingStat` protocol, plus one
  persistence path (`tl.stats.save_fitted` / `tl.stats.load_fitted`) for fitted PCA and,
  DOCUMENTED-UNSTABLE, for any accumulator's whole state (`tl_stats_state_v1`; a reload
  keeps folding as if never interrupted; `load_fitted(path, encoder=...)` takes back an
  `encoder=` callable, which is not stored). `tl.aggregate(model,
  dataloader, metrics)` streams activations (`target="out"`) or gradients
  (`target="grad"`, with `loss_fn=`) through those accumulators across a whole
  dataloader without retaining per-batch tensors; `cache=True` (with `cache_dir=`, as
  for `tl.trace`; DOCUMENTED-UNSTABLE) keys the run by value (model contents and code,
  every batch, each accumulator's starting state and encoder, `target=`, `loss_fn=`,
  versions) so a rerun restores the saved states and skips the loop, and refuses a run
  it cannot key (`stats_aggregate_cache_unkeyable`). Per-unit statistics
  (DOCUMENTED-UNSTABLE): the keyword-only `unit_axis=` on `tl.stats.Quantile` and
  `tl.stats.TopK` keeps one reservoir or top-k list per unit (neuron, channel or SAE
  latent); per-unit `TopK` returns a `tl.stats.UnitTopK` (`values`, `example_ids`,
  `positions`, `unit_axis`), the one top-k-over-a-dataset tool for neurons and SAE
  latents, with ties broken toward the lowest example id, then position;
  `encoder=` maps each batch (for example through an SAE encoder) before it is
  folded; `TopK.update(..., example_ids=...)` names examples outside
  `tl.aggregate`. Reduce and discard (DOCUMENTED-UNSTABLE): `tl.trace(model, x,
  save="stats")` folds every op's out into a `tl.stats.Spine` during the forward and
  keeps no payload (one 16-slot float64 vector per op); `tl.stats.site_spines(trace)`
  returns `{op label: Spine}`, payload reads refuse `activation_reduced_to_stats`, and
  replay or validation on the trace refuses `stats_capture_not_replayable`. Trajectory
  statistics (DOCUMENTED-UNSTABLE): `tl.stats.measure_trajectory(model, data, *,
  statistics=None, facet="resid_post", layers=None, lens=None, mask_input=False,
  validate=True)` traces each batch and folds the logit-lens trajectory's `entropy`,
  `kl_to_final`, `max_prob` and, for `(input_ids, target_ids)` batches,
  `cross_entropy` and `target_rank` into a `tl.stats.TrajectoryStats`
  (`position_means[statistic]`, a float64 `[layers, positions]` tensor;
  `spines[statistic][layer]`; `summary(statistic)`; `examples`, the number of examples
  (batch rows) folded; `validated`, whether the default lens was numerically validated;
  `tie_convention`, how `target_rank` breaks ties). Doc of record:
  [docs/reference/stats.md](stats.md).

## Quantities and typed errors

**Quantities**
: `tl.Quantity` is the marker base for numeric quantity wrappers with unit-aware
  display: `tl.Bytes` (memory), `tl.Duration` (seconds), `tl.Flops` (floating-point
  operations), `tl.Macs` (multiply-accumulates).

**FLOP convention (`flop_convention=`)**
: `"fma2"` (default; one multiply-accumulate = 2 FLOPs) or `"fma1"` (recounted
  from each op's two-term compute record; refuses typed with
  `flop_convention_unavailable` when an op has no MAC split). Accepted by
  `summary(...)` and `tl.utils.flop_count(...)`; the removed `count_fma_as_two=`
  spelling refuses typed naming it.

**Autograd-saved bytes**
: `Op.autograd_memory` / `Op.num_autograd_tensors` count the tensors the captured forward's
  autograd saved for that op's backward (each storage once within the op; a view counts in full),
  and `Layer.total_autograd_memory`, `ModuleCall.autograd_memory`,
  `Module.total_autograd_memory` and `Trace.total_autograd_memory` sum those per-op bands, so a
  tensor two ops save counts twice. Capture runs floating Parameters with `requires_grad=True`
  unless `backward_ready=True`, so on frozen Parameters these and `Trace.pinned_autograd_memory`
  exceed what an eager forward saves; with `backward_ready=True` they match eager.

**Module and call counts**
: `ModuleCall.num_ops` counts the call's pass-qualified Ops (the root call included: a Layer it
  runs twice counts twice) and `ModuleCall.num_layers` the distinct Layers among them.
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
: `tl.AmbiguousOpLookupError` is raised when a bare Op lookup matches multiple
  pass-qualified Ops; `tl.ReentrantTraceError` when a trace is started while another
  trace is active.

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
: `MergeAlignment`, `BoundaryConsistency`, `MergeValueStatus`, and `MergedErrorCode` are
  frozen vocabularies in `torchlens.merged`, release-gated against
  [the merged-trace contract](merged_trace_contract.md).

**Trace.distributed_scope** — *unstable — no deprecation shim owed*
: The persisted shard-local capture marker (L8/F6): `None` for ordinary captures,
  `"rank_local_shard"` when any parameter or input the capture saw was one rank's local
  shard — the trace is a rank-local recording, never whole-model truth. Persists as of
  tlspec v8 with closed-vocabulary load validation; the bundle-save chokepoint refuses
  `shard_local_persistence_unsupported` under any schema that would silently drop the
  disclosure (see the [error and refusal contract](error_refusal_contract.md)).

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
  `docs/recipes/pretrained_saes.md`.

**register / unregister (`tl.bridge.sae.register(name, sae, *, site)`, `tl.bridge.sae.unregister(name)`)** — *unstable — no deprecation shim owed*
: Make (or stop making) an SAE's latents the facet family `name` at `site`.
  `tl.facet(name).latent(k)` then addresses latent `k` at the doors a
  `tl.head` selector serves (`Trace.set`, `attach_hooks`, `Trace.do`,
  `fork().do`, `tl.do`; and, as a bare selector, `trace[selector]` and
  `trace.resolve_sites(selector)`, which return the `Facet` whose `.value` is
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

**reconstruct (`tl.bridge.sae_lens.reconstruct(log, site, sae)`)** — *unstable — no deprecation shim owed*
: The SAE's reconstruction of a saved site activation,
  `sae.decode(sae.encode(out))`; a callable SAE without that pair is called on
  the activation (an SAE Lens `SAE`'s forward is the same encode then decode).
  Replaces the removed `decode(log, site, sae)`, which passed the activation
  to `sae.decode` as if it were latents. `tl.bridge.sae_lens.encode(log, site,
  sae)` gives the latents.

**write_delphi (`tl.bridge.sae.write_delphi(latents, tokens, *, site, model_name, n_splits=5, directory)`)** — *unstable — no deprecation shim owed*
: Writes SAE latents as the latent cache delphi's `LatentDataset` loads
  (eai-delphi 0.1.3): `directory/site/{start}_{end}.safetensors` splits with
  `locations` (example, position, latent), `float16` `activations` and the
  `tokens`, plus a `config.json` naming `model_name`. `latents` is dense
  (`SpliceResult.latents`, every activation above delphi's 1e-5) or the
  `UnitTopK` of `tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` over a
  dataset (only the retained top entries). Recipe:
  `docs/recipes/sae_latent_export.md`.

**write_sae_dashboard (`tl.bridge.sae.write_sae_dashboard(latents, tokens, *, latent_ids=None, k=20, path)`)** — *unstable — no deprecation shim owed*
: Writes SAE latents and each latent's top examples as the JSON SAEDashboard's
  `SaeVisData.load_json` loads (sae-dashboard 0.8.0): one "TOP ACTIVATIONS"
  group of whole sequences per latent, ranked by activation, then lowest
  example, then lowest position (the `tl.stats.TopK` order). Dense latents
  carry every token's activation and give `feature_stats` `max` and
  `frac_nonzero`; a dataset `UnitTopK` carries the ranked token's value only.
  `latent_ids` defaults to every latent that fired; `sae_export_dashboard_large`
  warns once past 256 MiB. Recipe: `docs/recipes/sae_latent_export.md`.

**similarity_repository_id (`rdm_evolution(...).similarity_repository_id`, `tl.stats.CKA.similarity_repository_id`, the `SimilarityScore` that `tl.stats.cka` and `tl.stats.CKA.result()` return, `ScoreInterval.similarity_repository_id`, the `neuro.rdms()` source-mode descriptor)** — *unstable — no deprecation shim owed*
: The similarity-repository catalogue identifier of the exact measure variant a
  TorchLens result computed, with `*` for the implementing repository (for
  example `measure/*/cka-kernel=linear-hsic=gretton-score` for linear CKA,
  `rdm/*/correlation` for a correlation RDM). On `rdm_evolution` results it is
  `None` when the catalogue has no exact match (the `"manhattan"` metric); the
  `neuro.rdms()` descriptor is then absent. The full table is in
  [the brain-benchmarking seam](brain_benchmarking_seam.md).

**SimilarityScore (`tl.stats.SimilarityScore`; returned by `tl.stats.cka` and `tl.stats.CKA.result()`)** — *unstable — no deprecation shim owed*
: A `float` subclass: the score itself (arithmetic, comparison, `repr` and JSON
  encoding are a float's; arithmetic returns a plain `float`) plus
  `similarity_repository_id`, the exact measure variant it computes
  (`measure/*/cka-kernel=linear-hsic=gretton-score` for linear CKA). `to_dict()`
  is the serialized form that keeps the identifier
  (`{"score": ..., "similarity_repository_id": ...}`); pickling keeps it too.

**ScoreInterval (`tl.repgeom.ScoreInterval`; returned by `tl.repgeom.rdm_compare_interval` and `tl.repgeom.cka_interval`)** — *unstable — no deprecation shim owed*
: A TorchLens score with a descriptive uncertainty interval beside it:
  `score` (the unchanged estimator on all stimuli), `low` and `high` (a
  percentile bootstrap over stimuli), `confidence`, `n_resamples`, `n_valid`
  (resamples with a finite score), `seed`, `basis` (how the interval was
  made, in words) and `similarity_repository_id` (the measure variant `score`
  computes: the linear CKA identifier from `cka_interval`, `None` from
  `rdm_compare_interval`, whose precomputed RDMs do not name their metric),
  plus `width` and `to_dict()` (every field). `rdm_compare_interval(rdm_a,
  rdm_b, method, n_resamples=1000, seed=0, confidence=0.95)` resamples
  stimuli and recomputes `rdm_compare`, dropping self-pairs; `cka_interval(a,
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

**neuro.compare (`tl.neuro.compare`; needs no extra)** — *unstable — no deprecation shim owed*
: One call that scores every site of one extraction against one target with
  several descriptive analyses: `tl.neuro.compare(source, target,
  analyses=("rsa", "ridge", "cka"))`. Each site's activations are read once
  from what `source` (a Trace with saved activations, an extraction, or a
  `{site: array}` mapping) already holds; the model is never re-run. `"rsa"`
  is `rdm_compare` over `rdm(site)` and `rdm(target)` (`rdm_metric`,
  `rsa_method`); `"ridge"` is a cross-validated ridge score (a seeded
  `n_folds` split, the penalty per fold chosen from `ridge_alphas` by exact
  leave-one-out on the training fold, then the mean over target columns of
  the Pearson r between out-of-fold predictions and the target); `"cka"` is
  linear CKA. `n_resamples=` adds a `ScoreInterval` per score. The returned
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

**Layer.site_key** — *unstable — no deprecation shim owed*
: The layer's single site key iff every op in the layer shares exactly one.
  A site-SPANNING layer (within-call-instance recurrence such as transformer
  residual-add pairs, or root-context loops) refuses typed
  `layer_site_ambiguous` — read per-pass keys via `.ops[k].site_key`.
  Site-uniformity and pass-uniformity are DIFFERENT axes: a reused-module
  multi-pass layer has one key. Legacy pre-site-key artifacts refuse typed
  `site_key_unavailable`, never a `None` read.

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

**Trace.nonfinite_ops** — *unstable — no deprecation shim owed*
: The queryable per-op NaN/Inf record: a tuple of pass-qualified op labels
  (`Op.label`, each a valid `trace[label]` key) whose output held at least one
  NaN or Inf. On default captures it derives from the memoized saved-payload
  scan already backing `print(trace)` — zero capture-time cost, one scan paid
  at first query. When the capture ran with
  `CaptureOptions(track_nonfinite=True)` it serves the capture-time verdicts
  instead, which also cover ops that retained no payload. An empty tuple is
  only as strong as its coverage; read `Trace.nonfinite_coverage` first.

**Trace.nonfinite_coverage** — *unstable — no deprecation shim owed*
: The frozen evidence disclosure behind `nonfinite_ops`: `basis`
  (`"capture"` vs `"saved_payloads"`), `checked`, `nonfinite`, `unchecked`
  (dtypes with no runnable finiteness kernel), `unexamined` (ops the scan
  could not look at), and `unmapped` (capture-basis events whose op did not
  survive postprocessing). The programmatic twin of the prose coverage-gap
  note in `first_nonfinite()`.

**track_nonfinite (CaptureOptions)** — *unstable — no deprecation shim owed*
: Session-time opt-in: torch capture records a per-op finiteness verdict for
  every committed op output. Never changes control flow (`raise_on_nan` stays
  the stop-and-throw and is independent); device flags are drained in one
  batch after the forward, never a per-op CUDA synchronization.
  `structure_only=True` refuses the combination typed. Load restores the
  default `False`; loaded traces serve the saved-payload basis. The default
  stays `False` (no overhead); the `print(trace)` NOT-CHECKED health line
  names this option.

**grouping= (trace kwarg) / trace.grouping** — *unstable — no deprecation
shim owed; S2-gated vocabulary*
: Closed-vocabulary grouping-policy knob: `"structural"` (default — today's
  recurrence grouping), `"strict_shapes"` (reserved; refuses typed until its
  own reviewed design lands), `"fold_sites"` (the D1 within-capture folding
  axis; refuses typed on plain captures until an affirmative D1 ruling).
  Unknown values refuse `grouping_invalid`; legal-but-not-entry-legal values
  refuse `grouping_policy_unavailable`. The mirror field `trace.grouping`
  records the requested value. Distinct from the display-only `fold_repeats`
  viz knob, which folds repeated module runs at render time and never
  changes grouping.

**trace.grouping_policy** — *unstable — no deprecation shim owed; S2-gated
vocabulary*
: The persisted, load-validated `grouping_policy_v1` stamp recording HOW the
  trace was grouped: `policy` (the step-7 grouping that actually ran —
  `structural` / `params_only` / later `fold_sites`), `requested` (the knob
  mirror), `folded_sites` and `site_join` (two distinct site-granular axes:
  step-7 folds vs product-layer joins), `detector`, `effective`, and
  `settlement_note`. Loads validate against the exact writer key set, closed
  vocabularies, and coherence rules C1–C8; parse failure or incoherence
  warns once and settles to THE canonical degraded representation
  (`policy="unknown"`, `settlement_note="grouping_stamp_<reason>"`), which
  round-trips byte-stable and stays degraded — verdicts only worsen across
  persistence. Legacy pre-stamp artifacts settle silently to
  `grouping_stamp_legacy`; degraded stamps refuse stamp-consuming
  operations typed. Persists as of tlspec v8 with C1-C8 load validation.

**L1 grouping refusal codes** — *unstable — no deprecation shim owed;
S2-gated*
: `layer_site_ambiguous`, `site_key_unavailable`, `grouping_invalid`,
  `grouping_policy_unavailable`.

**structure_only (capture kwarg) / trace.structure_only** — *unstable — no
deprecation shim owed*
: `tl.trace(model, x, capture=CaptureOptions(structure_only=True))` runs the
  capture under the structure-only contract: the op graph, module hierarchy,
  parameter geometry, and per-op shape/dtype are recorded with every
  value-bearing claim a HYPOTHESIS; value payloads are never retained,
  value-requiring consumers refuse typed through
  `torchlens.capture.structure_only.require_structure_only_capability`, and
  value-dependent branches refuse with the user's source line
  (device-neutral). The mirror field `trace.structure_only` declares the
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

**tl.ir.ModuleCaptureWorkspace.root_forward_args** — *unstable — no deprecation shim owed*
: The root call's `(args, kwargs)`, kept on a capture's module workspace while the capture
  runs: the root forward is never decorated, so the per-module forward-argument table has no
  entry for it. Set at capture entry, read when finalization builds the root module pass, then
  cleared with the other temporary build state (`None` outside a capture). The
  `ModuleCaptureWorkspace(..., root_forward_args=...)` keyword sets the same field.

**PredicateProtocol** — *unstable — no deprecation shim owed*
: `torchlens.ir.predicate_registry.PredicateProtocol` — the frozen callable
  signature for the capture-lifecycle `save`/`halt`/`until` predicate slots:
  one positional concrete `RecordContext`, returning a normalized decision.
  The S4 seam contract; normative page:
  [predicate_runtime.md](predicate_runtime.md). The `intervene=` and grad
  slots are outside the protocol.

**coerce_predicate(value, \*, slot)** — *unstable — no deprecation shim owed*
: The single documented coercion door for predicate consumers
  (`torchlens.ir.predicate_registry`). Closed value domain {raw callable,
  registered-name str}; raw callables (including `BaseSelector` instances and
  `followed_by` composites) return BY IDENTITY, registered names return a
  slot-aware enforcing wrapper carrying
  `__torchlens_cache_key__ = ("registered", name, version)`. Slot vocabulary
  `save | halt | until` is closed (S2-owned); unknown slots refuse typed.

**Selection / ResolvedSelection / `__selection__`** — *unstable — no
deprecation shim owed (do/Selection spellings slate-ratified subject to D7)*
: `tl.Selection` is the composable selection QUERY — a frozen AST over leaf
  terms (selector / receptive-field box / gradient-RF / facet / param / unit)
  and boolean combinators, trace-independent; `selection.resolve(trace)`
  returns a `tl.ResolvedSelection` — frozen, trace-bound, an ordered tuple of
  `SiteEntry(site_key, mask, provenance)` rows. TWO-LEVEL DENOTATION:
  (touched-site family, selected-element set); zero-mask entries are retained
  first-class; `.empty`/`__bool__` are ELEMENT-level; `bool()` on the QUERY
  type refuses typed (`selection_bool_ambiguous`). Anything region-shaped
  implements `__selection__`; `SiteEntry.mask` returns a FRESH materialization
  (mutation cannot alter the selection). Masks are exact AS SETS; producer
  inexactness rides the closed `provenance.relation` lattice
  (`exact | upper_bound | lower_bound | unknown`). `ResolvedSelection` is
  session-time only, never persisted. Selection kinds `ACT | PARAM | EDGE`
  are a closed vocabulary; mixed kinds refuse `selection_kind_incompatible`.

**Selection operators `| & - ~` (+ reflected)** — *ratified set (slate 5.4);
semantics unstable-documented*
: Same-site operands compose as masks; different-site yields a MULTI-SITE
  selection; `-` never un-touches sites (`fam(A-B) = fam(A)`); `~` is the
  touched-site mask complement (never predicate negation and never
  model-universe). NO `__xor__`: `(a - b) | (b - a)` spells it.
  `BaseSelector` keeps its shipped composite semantics; `selector - selector`
  desugars to `and(a, not(b))`; a selector composed with a region producer
  defers to the Selection algebra.

**tl.units / tl.params / tl.random_selection** — *unstable — no deprecation
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
  `sign(within, 'positive'|'negative'|'zero'|'nonzero', tol=)` (`'zero'` is
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
  `samples` to count it as evidence. `dead(samples, tol=)` selects elements
  with `|v| <= tol` in EVERY sample; `saturated(samples, low=, high=, tol=)`
  selects elements pinned within `tol` of the same declared bound in every
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
  `neighborhood(of, hops=, direction=)` selects every op within N recorded
  dataflow hops of a seed region (`hops=0` is the seed family itself;
  `direction` is `'both' | 'upstream' | 'downstream'`).
  `between(sources, sinks)` selects the executed sub-DAG carrying influence
  from the source region to the sink region — exactly the ops on at least
  one directed source-to-sink path, endpoints included; no directed path
  resolves to the EMPTY selection (emptiness is disclosure, never an
  error). Operands are site label strings (a bare layer label on a
  multi-pass layer is the all-passes Layer spelling), `Op`/`Layer` handles,
  or any ACT selection; `between` endpoints also accept lists of regions.
  Element masks never shrink a graph region (a touched site is a touched
  node — family semantics). Membership is a structural fact about THIS
  capture, so entries are whole-site masks with
  `provenance.relation="exact"`. PARAM/EDGE operands refuse
  `selection_kind_incompatible`; unknown sites refuse `site_not_in_trace`;
  member sites with no output index space refuse `non_tensor_site` /
  `no_index_space`. Both producers are pure functions over the ONE
  executed-DAG graph substrate (`torchlens.selection_graph._TraceGraph`,
  shared with the influence-geometry path machinery), the seam a future
  graph-MOTIF producer plugs into.

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
  honour them); `tl.save` refuses it typed (`slice_save_unsupported`).
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
  `changed(ref)` is the "every element that moved" intervention-effect
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
  layer). `stable_across_passes(within=None, tol=, passes=)` selects
  elements whose range (max - min, float64) across the pass window is
  `<= tol`; `pass_variance(within=None, above=, below=, passes=)` applies
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
  tol=0.0)` takes one direction `[d]` or a stack `[k, d]` (canonicalized to
  float64). Under `do()` the edit acts on the COMPONENT along the span of the
  basis rows: with `Q` the rows, on the coordinates `c = Q x`, written back as
  `x + Q^T (edit(c) - c)`; the orthogonal complement is unchanged. Coordinate
  verbs (`zero_ablate`, `scale`, `clamp`, `noise`, `mean_ablate()`, hook
  callables, ...) act on `c`; value verbs (`patch_from`, the interchange;
  `replace_with`, `steer`, ...) write a value's component; other verbs refuse
  `subspace_edit_unsupported`. An orthonormal basis is used as given; any
  other full-rank basis is orthonormalised (Gram-Schmidt in row order, signs
  kept). A subspace combined with set operators, or resolved, refuses in
  `do()` (`subspace_projection_composed`); `within=` restricts positions
  (a partial-axis mask refuses `subspace_projection_partial_axis`).
  `resolve()` reports the basis's SUPPORT SET: every element whose coordinate
  on the bound axis carries weight `|w| > tol` in at least one basis vector,
  expanded across the site's other axes (a dense direction supports the WHOLE
  axis) (docs/intervention_explainers.md, "Subspace edits: the component along
  a basis"). BASIS PROVENANCE IS MANDATORY: `origin=` is
  a required non-empty account of where the basis came from; origin,
  optional `method=`, geometry, and the basis's sha256 content digest are
  stamped into every resolved entry's `provenance.source` (and therefore
  `do()` audit records); the frozen
  `torchlens.selection_subspace.BasisProvenance` record is the programmatic
  face. DIMENSION HONESTY: `dim=` names the bound site axis (default `-1`,
  the trailing feature axis; conv channel directions are `dim=1`); a site
  whose extent there differs from `d` refuses `selection_unresolvable` /
  `basis_dim_mismatch` — never a silent broadcast, truncation, or padding.
  `within=` is REQUIRED (a direction is minted for one representation
  space). Non-finite bases and rows entirely at-or-below `tol` refuse at
  construction (vacuous direction). Resolution is geometry-only (no payload
  reads — unsaved sites resolve); relation is `exact`; PARAM/EDGE
  populations refuse `selection_kind_incompatible` (a parameter-space
  direction is a named possibility, not a promise). PORTABLE ROTATIONS: with a
  builtin verb the staged per-site edit is the builtin `subspace_edit`, whose
  stored arguments are the verb, the float64 orthonormal rows it applies (a
  tensor sidecar), the bound `axis`, the rows' `basis_sha256` and the basis
  provenance (`origin`, `method`, `origin_sha256`, `source_dtype`), so
  `tl.io.save_intervention(trace, path, level="portable")` reloads it as the
  same projection, never as the plain verb over the whole site. A rotation
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
: One carrier class (`torchlens.selection.SelectionError`, catalogued in the
  intervention error catalog) for the closed codes
  `selection_trace_mismatch`, `selection_bool_ambiguous`,
  `selection_kind_incompatible`, `selection_unresolvable` (closed reason set
  `site_not_in_trace | value_not_saved | non_tensor_site | no_index_space |
  mask_shape_mismatch | facet_write_mask_unavailable | population_too_small |
  multipass_bare_label | value_criterion_invalid | basis_dim_mismatch`),
  `selection_apply_invalid` (stage 2), and `slice_save_unsupported`
  (`tl.save` on a `TraceSlice` presenter). The `multipass_bare_label` reason
  is the `tl.units` face of the multi-pass bare-label ambiguity refusal: a
  bare layer label addresses only single-pass layers, and each pass of a
  recurrence-grouped layer must be named pass-qualified (`label:pass`); the
  string/`tl.label` addressing face is `multipass_bare_label_ambiguous` on
  `SiteAmbiguityError` (both teaching refusals name every pass-qualified
  spelling; spellings DOCUMENTED-UNSTABLE).

**tl.Edit / do(selection, edit)** — *Edit ratified (slate 5.5, subject to D7
default-keep); mask-application semantics documented-unstable*
: `tl.Edit` is the public edit-object type (the old class
  spelling is removed, U00 clean break). `repr(edit)` prints the
  public spelling: a builtin edit prints as the public call that builds it
  (`tl.grad_scale(2.0)`, default-valued keywords omitted; evaluating it
  rebuilds an equal edit), any other edit as `tl.Edit('name', kind=...,
  portability=...)`. An `Edit` is a frozen record of its identity:
  `helper_name` (also read as `name`), positional `args`, keyword `kwargs`
  (a tuple of name and value pairs), `kind`, `portability`, `metadata`
  (name and value pairs, such as `("keep_grad", True)` or the selection
  disclosure a masked `do()` adds), `direction`, `batch_independent`,
  `compatible_with_append` (an edit that may change the output shape is
  refused on an append rerun unless this is true) and `selection_recipe`
  (the selection recipe a selection-targeted `do()` attaches; `None`
  otherwise). The hook `factory` is session-side. `Edit.PORTABLE_STATE_SPEC`
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
  the selected elements are patched. Source-to-target mapping (Patchscopes,
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

**trace.edges / edge substitution** — *unstable — no deprecation shim owed;
S2/S3-gated*
: `trace.edges` returns the dataflow edge family (one `EdgeUseRecord` per
  parent→child occurrence; requires an `intervention_ready` capture, else
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
: `fork.do(tl.params(name, mask=None), edit)` applies the edit "as if" the
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
  `do(tl.params(p), scale(0.5))` COMPOSES on the staged "as if" value,
  0.25, keeping both FireRecords), and the same validation boundary
  (`edge_intervention_boundary` — a different check, never no check;
  uncorroborated entries FAIL). Replay/push engine ONLY:
  rerun/set_only refuse `param_substitution_engine_unsupported`. The audit
  record (kind `PARAM`) discloses "substituted at consumption … live
  parameters unchanged" — the product is DERIVED-class, never a blessed
  VERIFIED reproduction of the original weights. BUFFER OPERANDS
  (documented-unstable): `do(tl.params("<buffer address>", mask), edit)` on a
  registered buffer (BatchNorm running stats, a rotary cache, recurrent state)
  substitutes the buffer's INITIAL version node, the value its consumers read,
  and replays the cone through the leaf-site path (needs an
  `intervention_ready=True` capture); in-forward writes thread the edited
  value; the live buffer is never written.

**TapObserver.values(masked=True)** — *unstable — no deprecation shim owed*
: `tap(resolved_selection)` stores each firing site's mask on the
  `TapRecord`; `values(masked=True)` returns fresh masked copies (selected
  elements). `values()` stays exactly the shipped full-snapshot behavior.

**register_predicate(name, \*, replace=False)** — *unstable — no deprecation
shim owed*
: Registers a plain predicate callable under a name for later
  `coerce_predicate` acceptance; returns the function truly unchanged (no
  attribute stamped, nothing the restricted loader consults). Duplicate user
  names refuse `predicate_name_conflict` without `replace=True`; builtin
  names are never replaceable; name misses at coercion refuse
  `predicate_unregistered`. The registry is INERT until consuming surfaces
  adopt name acceptance.

**color_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The v1 encoding-channel value source on `Trace.draw` (L5 channel core). See
  the "color_by" entry above for semantics.

**size_by / scale (draw kwargs)** — *unstable — no deprecation shim owed; keyword-only*
: The wave-1 size encoding channel on `Trace.draw` (D4 default-applied: sqrt +
  conservative area-only mapping + typed refusal on rolled varying sources).
  See the "size_by" and "scale (size channel)" entries above for semantics.

**stack_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The wave-1 rank encoding channel on `Trace.draw` (stacking split (a):
  explicit + licensed auto on plain traces). The `True`/`"auto"` request form
  is itself an unstable spelling. See the "stack_by" entry above.

**show_redundant_args (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: Opt-out for the default-on checked suppression of redundant constructor-arg
  label rows (slate 8.7). See the "show_redundant_args" entry above.

**direction="rightleft" (draw value)** — *unstable — no deprecation shim owed*
: The fourth layout direction on `Trace.draw(direction=)`, `draw_backward(vis_direction=)` and
  `draw_combined(vis_direction=)`: edges flow right to left (Graphviz `rankdir=RL`), beside
  `"bottomup"`, `"topdown"` and `"leftright"`. An unknown value refuses
  `visualization_direction_invalid` naming all four.

**vis_outpath=None (draw default)** — *unstable — no deprecation shim owed*
: A bare `Trace.draw()` writes a collision-safe file in the current directory --
  `modelgraph.<format>`, then `modelgraph-2.<format>`, `modelgraph-3.<format>` -- and always prints
  `TorchLens graph written to <path>`, so a second bare draw never overwrites the first picture.
  An explicit `vis_outpath=` is written exactly (overwriting) and printed only on a verbose trace.

**show_module_paths / show_grad_values (draw_backward kwargs)** — *unstable — no deprecation shim owed; keyword-only*
: Opt-in label rows on `Trace.draw_backward`: `show_module_paths=True` adds `module <address>` to
  each backward node a module owns (`param <address>` on AccumulateGrad nodes); `show_grad_values=True`
  adds `grad norm <x>, absmax <y>` from the node's captured output gradient (`grad values not
  captured` when the capture kept none). Both default to `False`, leaving the default render
  unchanged.

**Bundled render font** — *unstable — no deprecation shim owed*
: Every Graphviz render resolves the pinned `Helvetica` family to the bundled Liberation Sans 2.1.5
  (SIL Open Font License 1.1, shipped unmodified under `torchlens/visualization/fonts/` with its
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
  `vis_call_depth`): warn and obey, never a refusal.

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
  `CaptureOptions(transform=tl.preprocessing.imagenet_default())`. A PIL image
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

**torchlens.preprocessing.check_card(trace_or_record, *, model=None, authority=None, offline=None)** — *unstable — no deprecation shim owed*
: The on-request preprocessing verifier: `trace_or_record` is a `Trace` (its
  `input_preprocessor` is read) or a `ResolvedPreprocessing` record. It compares the
  preprocessing a capture recorded (`Trace.input_preprocessor`) with the model's known card (HF image
  processor, torchvision weights, timm or open_clip config, or a named
  `authority=`), returns the field-level `PreprocessingAudit`, and warns
  `preprocessing_card_mismatch` when a field disagrees. Nothing runs it
  automatically. An opaque record compares as unknown, never as a match.

**torchlens.bridge.pyreft.read(directory, *, repo=None, revision=None)** — *documented-unstable*
: Read a pyreft LoReFT artifact (the directory `ReftModel.save` writes: `config.json` plus
  `intkey_<key>.bin`, under the key's dotted or underscored name) into a
  `torchlens.bridge.pyreft.LoreftArtifact`; never imports pyreft or pyvene and never fits
  anything. Apply it with `fork.do(artifact.subspace(site, positions=mask), artifact.edit)`, which
  is `tl.subspace(tl.units(site, ...), R.T, origin=..., method="loreft")` with
  `tl.intervention.affine_source(W, bias=b)`. pyreft trains `R` in float32 (orthonormal to about
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
  `pyreft_positions_invalid`; a label is checked at `do()`. `edit` is the float64 `affine_source`.

**torchlens.bridge.huggingface.from_hub(repo_id, *, path_in_repo, revision, token, api)** — *unstable — no deprecation shim owed*
: Downloads the gzipped `.tlspec` tar that `push_to_hub` uploads, checks
  every tar member (no links, absolute or parent-escaping names, one
  top-level bundle), and loads it eagerly onto the CPU through `tl.load`'s default-deny
  loader. `push_to_hub` accepts only a `Trace` or `Bundle` and never uploads
  a pickle (`hub_push_artifact_unsupported`); its default `path_in_repo` is
  `torchlens_artifact.tar.gz`.

**call_kwargs (torchlens.bridge.gradcam.cam)** — *unstable — no deprecation shim owed*
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
  the `tl.trackers` sink protocol: `capabilities()` returns what the sink
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

**tl.export.xarray(log)** — *unstable — no deprecation shim owed*
: Returns an `xarray.Dataset` (CHANGED, batch-2 buffer ruling, option c: it returned
  a bare `DataArray`; read today's array as `tl.export.xarray(log)["out"]`).
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
  `example_input=`; refuses `watch_forward_wrap_consent_required` without
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

**StepBlockRecord.trace_ref (torchlens.observability)** — *unstable — no deprecation shim owed*
: Nullable pointer (path or URI) from a history step block to a retained
  `Trace` for that step; there is no raw-tensor snapshot tier. History schema
  version 2 (`HISTORY_SCHEMA_VERSION`); version-1 artifacts load with
  `trace_ref=None`; a non-string or empty pointer refuses
  `history_trace_ref_invalid`. `OpTierCollector.settle_open_block(...)`
  sets it (and the optimizer truth) for the open op-tier block;
  `OpTierCollector.instrumented_forward(...)` runs the per-step pass without
  committing a block.

**WatchSettings(sketch_embeddings=, artifact_budget_bytes=) (torchlens.observability)** — *unstable — no deprecation shim owed*
: `sketch_embeddings=False` (default) reduces embedding-table parameters with
  at least `EMBEDDING_SKETCH_CAP_NUMEL` (2**20) elements spine-only and says
  so in the plan table; `True` sketches them. `artifact_budget_bytes` (default
  256 MiB, `None` disables) caps the disk history artifact: the block that
  would cross it refuses `history_artifact_budget_exceeded` before any of its
  bytes are written.

**show_legend=None AUTO value** — *unstable — no deprecation shim owed*
: The tri-state AUTO value on the stable `show_legend` kwarg: no legend unless
  an encoding channel is active, then a channel-only disclosure legend.
  `True`/`False` keep their stable historical meanings.

**Encoding refusal codes** — *unstable — no deprecation shim owed*
: `encoding_source_invalid`, `encoding_value_invalid`,
  `encoding_callable_error`, `encoding_requires_dot_layout`,
  `size_by_rolled_varying`, `scale_requires_size_by`, `encoding_scale_invalid`,
  `stack_by_auto_underivable`, `stack_by_requires_unrolled`.

**trace.grad_fn_fire_timings** — *unstable — no deprecation shim owed*
: Live-trace-only per-fire backward timing spans, keyed like
  `trace.grad_fn_calls` (`"<grad_fn_label>:<call_index>"`). Both stamps of
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

**BackwardPass.dispatched_ops / DispatchedOp** — *unstable — no deprecation shim owed*
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

**BackwardStatus.executed_flops / executed_flops_per_pass / uncounted_ops / counting_convention** — *unstable — no deprecation shim owed*
: `tl.report.backward_status(trace)` in the `backward_observed` state counts
  the backward that ran over each pass's `dispatched_ops`: `executed_flops`
  is exact for the matmul and convolution families at fma2
  (`counting_convention`, FlopCounterMode's convention), or `None` when a
  pass has no record on this object. `executed_flops_per_pass` pairs each
  pass index with its count; a `create_graph=True` gradient and the backward
  through it are separate passes. `uncounted_ops` names every recorded op
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

**BackwardPass.executed_flops_record** — *unstable — no deprecation shim owed*
: The pass's counted executed backward and its attribution as a closed
  record (`convention`, `executed_flops`, `uncounted_ops`, `rows`,
  `unattributed`; ints, `None` and short strings only), written when the
  trace is saved (tlspec v10) so a loaded trace reads the counted answer
  through `flops_attribution` and `tl.report.backward_status`. `None` on a
  live pass and on artifacts saved before the field, which read unknown
  (record keyword `executed_flops_record=`). A
  malformed record refuses at load with
  `artifact_executed_flops_record_invalid`.

**BackwardStatus.attributions / unattributed_flops / recompute_flops** — *unstable — no deprecation shim owed*
: `tl.report.backward_status(trace)` in the `backward_observed` state: one
  `BackwardFlopsAttribution` per counted pass, the counted work that ran
  under no recorded node, and the part that replayed forward kernels under
  activation checkpointing (`None` when the evidence cannot separate replay
  from backward, as in a `create_graph` pass that checkpoints). Empty or
  `None` in the other two states. The record's constructor keyword is
  `attributions=`.

**DispatchedOp.grad_enabled** — *unstable — no deprecation shim owed*
: Whether autograd grad mode was on when the call dispatched during a
  recorded backward. Backward kernels run with grad mode off unless the pass
  builds a graph; checkpoint replays run with it on, which is how a replayed
  forward kernel is labeled recompute.

**CostTreeRow forward/backward/total columns; CostTree.backward_state / backward_total / forward_mac_total / column_conventions** — *unstable — no deprecation shim owed*
: `tl.report.cost_tree(trace)` rows carry `self_forward_mac_flops` (2 x true
  MACs), `self_backward_flops` (executed backward, matmul and convolution
  families, fma2) and `self_total_flops` (their sum, one convention), with
  non-additive subtree twins `CostTreeRow.subtree_forward_mac_flops`,
  `CostTreeRow.subtree_backward_flops` and `CostTreeRow.subtree_total_flops`
  (constructor keywords `subtree_forward_mac_flops=`,
  `subtree_backward_flops=`, `subtree_total_flops=`). Backward work on nodes with no
  forward op and unattributed calls get their own remainder rows, so the self
  backward column sums to `backward_total`. Without a counted backward the
  cells are `None` and `backward_state` names the state; never 0.
  `column_conventions` names each FLOPs column's convention. Siblings fold
  only when their MAC and backward cells agree too.

**BackwardPassEnd.close_path** — *unstable — no deprecation shim owed*
: Sidecar-event-only implicit-pass close-path disclosure: `"engine_drain"`
  when the queued final callback journaled the close, `"sync_point"` for
  every backstop path, `None` on explicit passes. The projected
  `BackwardPass` record field waits for the wave-3 bump.

**until= (run kwarg) / report.truncated / report.stopped_at** — *unstable — no
deprecation shim owed*
: `trace.run(until=...)` executes only the dependency closure (or disclosed
  sequential prefix) needed to reach the named sites. Truncation is a
  RESULT/REPORT term, never a capture outcome: the report's flat surface is
  `report.truncated` (bool) and `report.stopped_at` (the stop-frontier site
  label), with the full `RunTruncation` disclosure on `report.truncation`
  (regime, requested sites, executed/skipped counts, skipped-site digest).
  Skipped sites are "not-run", never "passed". With `fast=True` on a live trace
  (U26.3) the native forward is cut right after the `until=` site, which must be a
  site the capture saved (a module address, `"address:call"`, or a saved op label;
  otherwise `run_capability_unavailable` at stage `fast_until_site_unsaved`): the
  prefix keeps the fast path's per-call guards and comes back graph-connected under
  the default save, the regime is `live_stop_after`, the output and the later saved
  sites read `None` for that run (the next full run refreshes them), and the verdict
  is `unverifiable` because the whole-forward call fingerprint cannot be compared.
  The fast run refreshes the live trace in place, so that verdict marks the trace as
  every truncated run product is marked (monotonic: save, export or validate it before
  training through `until=`).
  A loaded artifact's fast tier still refuses `run_fast_until_unsupported`. Full
  contract: [the runnable tlspec contract](runnable_tlspec_contract.md).

**Trace.sites_table()** — *unstable — no deprecation shim owed*
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

**logit_lens (torchlens.semantic)** — *unstable — no deprecation shim owed*
: `torchlens.semantic.logit_lens(trace)` projects each block's
  residual-stream facet through the model's own final norm + unembedding,
  returning a `LogitLensResult` (per-layer logits, `summary()`,
  `top_tokens()`). The reconstructed lens is validated against the captured
  logits before use and refuses `LogitLensError` when the head cannot be
  represented; pass `lens=` for a tuned lens or `validate=False` to trust
  the reconstruction explicitly. With no language-model head, the default
  lens falls back to a vision head recipe, validated the same way:
  `classification_head` (final norm plus `classifier` over the `[CLS]`
  token; `lens_source="classifier_head"`) or CLIP's
  `contrastive_image_head` (per-layer scores against the captured text
  features of the label prompts; `lens_source="contrastive_head"`). A user
  lens may carry `model_identity` (an extraction manifest's identity
  record): a differing measured identity refuses
  `logit_lens_identity_mismatch` unless `validate=False`; a lens with no
  comparable identity warns `logit_lens_identity_unrecorded` when its
  last-layer decode misses the captured logits. Its streaming sibling
  `logit_lens_predictions(trace, ...)` keeps only per-layer top-k, `logsumexp`
  and requested-token values; its keyword-only `mask_input=True` bans the
  model input's own token ids before those reductions (the trace's first input
  must be saved integer token ids, else `logit_lens_mask_input_unavailable`).
  Trajectory statistics over many batches: `tl.stats.measure_trajectory`. See
  [facets](../facets.md).

**render_logit_lens_heatmap (torchlens.tviz)** — *unstable — no deprecation shim owed*
: `torchlens.tviz.render_logit_lens_heatmap(result, path)` draws a
  `LogitLensResult` as a layers-by-positions heatmap: one row per projected
  block, one column per input position, each cell printing that layer's
  top-1 next-token prediction and colored on a fixed 0-to-1 probability
  scale (the top-1 probability, or with `value="target_prob",
  target_token_id=...` a tracked token's). The footer states the lens
  source, whether it was validated, and that every row is projected through
  the final norm and head; only a validated lens lets it say that the last
  row reproduces the model's own output scores. `title=` prefixes the
  figure title with a subject such as the model's name. Lenses
  wider than 48 positions refuse `tv_record_invalid` until `positions=`
  names the columns; nothing is cropped silently. Returns an `Artifact`.

**kv-group receipts (torchlens.tviz)** — *unstable — no deprecation shim owed*
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

**Receipt cells and patch-grid baselines (torchlens.tviz, torchlens.mechinterp)** — *unstable — no deprecation shim owed*
: A `mechinterp.PatchGrid` from `mechinterp.grid`, its presets or
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
  ledger shows one fire per cell. `tviz.render_receipt_grid` titles the
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
  return order; `kernel_size` / `groups` read the row's own conv or
  pooling op; `params_trainable` is the per-path trainable count (`train`
  stays the yes/no/partial `requires_grad` tri-state); `mode` is
  `module.training` as captured (`train` / `eval` / `mixed`); `flops_pct`
  is the row's share of the known forward FLOPs. A `+name` delta inserts
  at the column's registry position. A FLOP cell whose count includes a
  full-square attention op ends in `^` with a footer note. See
  [summary](summary.md).

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

**ParamFacts.bytes / ParamFacts.pruned (`tl.report.PrunedParamFacts`)** — *unstable — no deprecation shim owed*
: FactCore parameter facts (schema version 2): `bytes` is the unique
  parameters' storage; `pruned` is `None` unless prune reparametrizations
  exist, else a `PrunedParamFacts(paths, raw, masked)` naming the pruned
  tensors, their raw element count, and the mask zeros (`masked=None` when
  the capture retained no mask values).

**FlopsReport.attention_full_square_ops / TraceProfile.attention_full_square_ops** — *unstable — no deprecation shim owed*
: The count (the `attention_full_square_ops` field) of `scaled_dot_product_attention` ops in the forward FLOP total
  of a `flops_report` or a `trace.profile()`; when non-zero the report or the
  profile footer prints that attention FLOPs count the full query x key
  square and masks / `is_causal` are not discounted. Every attention FLOP
  line carries the same note: the `tl.summary` FLOP cells (marked, with the
  note as a legend), the `explain` FLOPs line, and graph labels, where an
  attention node's `flops` label field or `flops` overlay row is followed by
  the note.

**`output_style_unknown`** — *unstable — no deprecation shim owed*
: The typed refusal `tl.errors.OutputStyleUnknownError` (an `InvalidArgumentError`
  subclass) for a `CaptureOptions(output_style=...)` no detector can resolve, with a did-you-mean over the built-in styles
  (`classification`, `imagenet`, `imagenet1k`, `hf_text`). It refuses before
  the forward when no custom output detector is registered, and after the
  capture when registered custom detectors all declined the style.

**tl.bridge.hf.register_attention(model=None)** — *unstable — no deprecation shim owed*
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

**tl.bridge.hf.record_attention(model, *, check_sdpa=False)** — *unstable — no deprecation shim owed*
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

**AttentionRecording / AttentionCall (`tl.bridge.hf`)** — *unstable — no deprecation shim owed*
: `AttentionRecording` is a read-only mapping from attention-module name to
  probabilities `(batch, heads, q_len, kv_len)`; `recording[name]` serves a
  module called once and refuses `hf_attention_multiple_calls` otherwise,
  `calls_for(name)` returns every call in order, `.calls` lists all
  `AttentionCall` records (`module_name`, `probs`, `num_key_value_heads`,
  `mask_kind`, `sdpa_checked`), and `.nbytes` is the retained total.

**DebugReport (`tl.debug.DebugReport`)** — *unstable — no deprecation shim owed*
: The one typed result of every report-shaped `tl.debug` helper (U42.2):
  `report_type`, `subject` and `status` (the lead), `to_dict()` (a plain,
  JSON-friendly payload; tensors and graph records summarized, never
  inlined), `to_pandas()` (the table), and `str(report)` (a bounded card:
  lead line, at most five findings, the exact remainder, the exits). The
  typed results (`TraceAudit`, `DeterminismReport`, `NanReport`,
  `FindNanResult`, `ParamAudit`, ...) subclass it; those with a house-voice
  `__repr__` keep printing it and serve the card as `report.card()`. The
  table helpers `compare`, `compare_params`, `dead_neurons`,
  `gradient_flow_audit`, `hot_path` and `recompute_candidates` return a
  tabular `DebugReport` whose `.to_pandas()` is the DataFrame they used to
  return (`attrs` included); the report has a table and never is one.

**Trace.profile(group_by="source_line") / TraceProfile.group_by** — *unstable — no deprecation shim owed*
: `group_by="source_line"` folds the op rows of `trace.profile()` into one
  row per user source line (`"<file>:<line>"`, the op's recorded call site),
  summing activation bytes, FLOPs and instrumented time; boundary rows fold
  into one `<inputs and outputs>` row, so every additive column sums to the
  ungrouped op view. Only with `level="op"`. `TraceProfile.group_by` records
  the key (`None` for one row per op).

**tl.summary("<hub model id>", input_size=...)** — *unstable — no deprecation shim owed*
: A string model is a Hugging Face Hub id (or local config directory):
  `tl.summary` fetches the config only, builds the declared architecture on
  the meta device (no weights downloaded or allocated, `trust_remote_code`
  never set) and takes the weights-free summary path, so it needs
  `input_size=` or meta inputs. Requires `transformers`.

**Trace.forward_peak_memory / Trace.backward_peak_memory**
: The largest memory growth above a bracket's starting baseline, including memory freed before the
  bracket ends. CPU reads the process resident high-water mark, reset at each bracket on Linux (the
  reset also resets a caller's own `VmHWM` and `ru_maxrss` readings); CUDA reads the device
  allocator peak. `backward_peak_memory` is the largest peak above a pass's starting baseline over
  all backward passes (CUDA: the growth of the device high-water mark).

**Trace.forward_peak_memory_reason** — *unstable — no deprecation shim owed*
: Why `Trace.forward_peak_memory` is `None`: a CPU forward that measured zero
  (resident high-water growth, end-of-forward RSS delta and, when opted in, the
  Python-allocation peak all 0) is recorded as `None` with this reason, which names
  `CaptureOptions(measure_python_peak_memory=True)` (forks.13). `None` when a
  value was measured, and on artifacts that predate the field. Persisted
  (KEEP); a loaded reason beside a number refuses
  `artifact_forward_peak_memory_invalid`. CUDA, MPS and the backward peak keep
  their numeric semantics.

**Trace.total_backward_memory**
: Bytes of the gradient payloads the trace retains, counted once per storage: a payload that is a
  view charges its whole base storage, an expanded gradient its smaller storage.
**torchlens.conformance.preview_scorecard(backend, *, workdir=None)** — *unstable — no deprecation shim owed*
: Runs the full C0-C2 conformance packs for one preview backend (`jax`, `tf`,
  `mlx`, `tinygrad`, `paddle`; `torch` scores the reference as a baseline) on
  that backend's own config-built roster (a function root and a module or
  object root) and returns a `PreviewScorecard`. Every case the runner defines
  for the roster is one cell in exactly one state: `passed` (executed, its
  independent oracle matched: output parity against a direct run compared as
  arrays, a repeat with the same op count, a distinct equal reload), `failed`,
  `unsupported` (a typed `BackendUnsupportedError`) or `not_run` (framework not
  importable, with the install remedy, or not reached after an earlier failure
  for the same model). All cells stay in the denominator. A scorecard is never
  a claim: toy inputs earn nothing, and it carries no claim string. The
  backends `run_conformance` and `resolve_adapter` accept are the same set
  (`torchlens.conformance.PREVIEW_BACKENDS` plus `torch` and `fake`); an absent
  framework yields named skips, never an error. A roster row that declares
  `RosterModel.expected_op_kinds` (`RosterModel(..., expected_op_kinds=...)`; its independent op inventory: every
  `Op.func_name` one forward performs, in order, derived from the model
  definition) also runs `c1_op_inventory_complete`, so output parity can never
  pass with an op missing or extra; every preview roster row declares one.

**torchlens.conformance.PreviewScorecard** — *unstable — no deprecation shim owed*
: The frozen canonical scorecard: `scorecard_version`, `backend`,
  `framework_version` (`""` when absent), `suite_version`,
  `torchlens_version`, `cells` (`ScorecardCell` rows of `tier`, `case_id`,
  `family`, `state`, `detail`), `counts` (cells per state plus `denominator`),
  `highest_tier` (highest tier whose every cell passed, cumulative from C0, or
  `""`) and `sha256` over the canonical bytes with that field excluded.
  `to_canonical_json()` renders the attested bytes (sorted keys, compact,
  ASCII).

**trace.ops counting (`len(trace.ops) == trace.num_ops`)** — *unstable — no deprecation shim owed*
: `len`, iteration, integer indexing, `keys()`/`values()`/`items()` and
  `to_pandas()` of `trace.ops` cover the ops `trace.num_ops` counts (no
  input, output or buffer rows; sumfam F1). Label lookup still resolves every
  row: `trace.ops["input_1"]` returns the input row and `"input_1" in
  trace.ops` is true although the label is not in `trace.ops.keys()`. The
  every-row sequence (inputs, outputs and buffers included) is
  `trace.layer_list`; `len(trace)`, `trace[int]` and `trace[slice]` index it.
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
: `graph.to_neuronpedia(*, slug, scan, tokenizer, prompt=None, node_threshold=0.8,
  edge_threshold=0.98, path=None)` returns the graph, pruned by circuit-tracer's
  influence rule, as a dict in Neuronpedia's published attribution-graph schema
  (and writes it to `path` when given). `slug` is the graph's name on
  Neuronpedia, `scan` Neuronpedia's model id, `tokenizer` anything with
  `decode(ids) -> str`, `prompt` the prompt string (`None` decodes the input
  tokens), `node_threshold` and `edge_threshold` the shares of total influence
  kept. Refusal codes `attribution_graph_export_argument_invalid` and
  `attribution_graph_export_unsupported`.

**`tl.attribution.overlay` with an `AttributionGraph` (`graph_value`)** — *unstable — no deprecation shim owed*
: `tl.attribution.overlay(trace, graph, *, reduce="abs_sum", graph_value=None)` paints an
  attribution graph onto the trace it was built from: each site holding feature or error
  nodes gets the `reduce` total of its nodes' values over every position. `graph_value`
  (graph sources only) is `"influence"` (what `None` means: each node's influence on the
  logits, circuit-tracer's rule) or `"activation"` (feature activations). Refusal codes
  `attribution_overlay_argument_invalid` and `attribution_overlay_graph_mismatch`.

**`tl.attribution.graph.upload_to_neuronpedia` (publish an exported graph)** — *unstable — no deprecation shim owed*
: `upload_to_neuronpedia(exported, *, api_key)` publishes the dict `to_neuronpedia` returned
  (or the file it wrote) on neuronpedia.org under the account that owns `api_key`, through
  Neuronpedia's documented two-step upload, and returns the graph page's URL. Standard
  library only; the key goes in the `x-api-key` header and never into a message. Checked
  before any request: the export's structure (`attribution_graph_contract_invalid`), the
  `slug` and `scan` characters and the file size (`neuronpedia_upload_argument_invalid`).
  HTTP failures raise `neuronpedia_upload_http_error`, answers without their fields
  `neuronpedia_upload_response_invalid`.
