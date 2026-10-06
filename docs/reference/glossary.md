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
  by `save=` fails explicitly.

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

**Capture outcome**
: The frozen `Trace.outcome`, `Recording.outcome`, or `PartialTrace.outcome`. Its status is one of
  `COMPLETE`, `HALTED`, `ABORTED_NONFINITE`, `FAILED`, `UNATTESTED`, or `UNKNOWN`; failed outcomes
  also identify the phase. See [Capture outcomes](capture_outcomes.md).

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
| CAM method and controls | `grad_cam`, `layer`, `relu`, `overlay`, `image`, `alpha`, `cmap` | unstable -- no deprecation shim owed |
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

<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:END -->

Integrated Gradients always reports its signed completeness residual against the exact endpoint
target delta. Occlusion always names its replacement baseline. CAM overlays visibly state the
native measured map resolution, while their larger display grid is labeled as interpolation.
The callable/index annotation used inside `torchlens.attribution` is privately named
`_AttributionTarget`; `TargetSpec` refers only to the intervention selector record and is not an
attribution-target alias. See the [attribution reference](attribution.md).

## Selection and storage

**Predicate**
: A site selector built with `tl.func`, `tl.module`, `tl.in_module`, `tl.label`, and boolean or
  temporal composition. `save=` is the only selective sparse-capture spelling; removed
  `keep_op=` and `keep_module=` arguments raise `TypeError`.

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
  selector to helpers such as `tl.zero_ablate()` or `tl.scale(...)`.

**Bundle**
: A named collection of aligned Traces, constructed with `tl.bundle(...)` or `tl.Bundle(...)`, for
  cross-run comparison.

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
  `tl.head` selects one attention head; `tl.facet` selects a semantic facet view;
  `tl.func_transform` selects `torch.func` transform boundary ops; `tl.output` selects
  model outputs, and `tl.output_at` / `tl.input_at` select a nested output or model-input
  path.

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
  `tl.resample_ablate`, which still resolves; for coherent donor patching use
  `tl.patch_from`), `tl.replace_with` (fixed value), `tl.swap_with` (another
  site's tensor), `tl.steer` (add a scaled steering direction), `tl.project_onto` /
  `tl.project_off` (keep or remove the component along a direction), and
  `tl.splice_module` (call a module as a black-box forward splice). Availability outside
  the torch backend is narrower; see the per-backend rosters in the backends guide.

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

## Extraction, observers, and admin

**Extraction helpers**
: `tl.pluck` returns the saved out for one layer, `tl.extract` for many layers, and
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
  Saved function arguments (the replay inputs) stay on each op's device under
  `output_device="cpu"`, so the GPU footprint roughly halves rather than disappearing; if
  the GPU budget is still exceeded, pass `save_budget=None` or a larger value. They
  apply to the `"forward"`, `"saved"`, and `"intervention"` scopes; a non-default value on
  `"backward"` or `"receptive_field"` raises `TypeError`. The lower-level
  `torchlens.validation.validate_forward_pass` accepts the same two keywords.

**Session admin**
: `tl.release_model` releases a traced model from persistent TorchLens preparation
  (restoring whole-model pickle / `torch.save` serializability); `tl.clear_capture_cache`
  empties the capture cache; `tl.io.list_logs` / `tl.io.reset_naming_counter` manage log
  bookkeeping.

## Persistence, containers, and namespaces

**Save / load**
: `tl.save` persists a `Trace` into a portable `.tlspec` directory bundle at a chosen
  level; `tl.load` loads a `.tlspec` object with eager tensor materialization.
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
  Trace collections and `tl.show_bundle_graph` renders a bundle's graph.

**Streaming statistics**
: `tl.stats` is the streaming-statistics namespace: constant-memory accumulators
  (`tl.stats.Mean`, `tl.stats.Norm`, `tl.stats.Covariance`, `tl.stats.CrossCovariance`,
  `tl.stats.Quantile`, `tl.stats.TopK`, `tl.stats.PCA`, `tl.stats.CKA` -- exact
  full-data linear CKA, never a minibatch approximation -- `tl.stats.Histogram`,
  `tl.stats.Spine`) built on the `tl.stats.StreamingStat` protocol, plus fitted-PCA
  persistence (`tl.stats.save_fitted` / `tl.stats.load_fitted`). `tl.aggregate(model,
  dataloader, metrics)` streams activations (`target="out"`) or gradients
  (`target="grad"`, with `loss_fn=`) through those accumulators across a whole
  dataloader without retaining per-batch tensors. Doc of record:
  [docs/reference/stats.md](stats.md).

## Quantities and typed errors

**Quantities**
: `tl.Quantity` is the marker base for numeric quantity wrappers with unit-aware
  display: `tl.Bytes` (memory), `tl.Duration` (seconds), `tl.Flops` (floating-point
  operations), `tl.Macs` (multiply-accumulates).

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
  default `False`; loaded traces serve the saved-payload basis.

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
  float64) and resolves to the basis's SUPPORT SET: every element whose
  coordinate on the bound axis carries weight `|w| > tol` in at least one
  basis vector, expanded across the site's other axes. SET, NOT PROJECTION
  (the normative boundary): a Selection denotes sets, masks are exact as
  sets, and `do()` is elementwise — the weights are disclosed but never
  carried, a dense direction supports the WHOLE axis, and
  `do(subspace(...), edit)` edits every supported element, never "the
  component along the direction" (projection-valued selections are a named
  design fork, not a promise). BASIS PROVENANCE IS MANDATORY: `origin=` is
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
  direction is a named possibility, not a promise).

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
: `tl.Edit` is the public edit-object type; `HelperSpec` is its deprecated
  alias (stable surface, no removal scheduled). `trace.do(selection, edit)`
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

**tl.patch_from(source_trace)** — *unstable — no deprecation shim owed*
: Edit factory patching targeted sites from another trace's recorded
  post-capture values (activation patching); with a Selection target only
  the selected elements are patched. Portability `opaque_audit`: persisted
  args carry source-trace IDENTITY only (never the Trace, never tensors);
  values bind at do() time session-side; no executable-save path in v1.

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
  VERIFIED reproduction of the original weights.

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

**"shape_summary" node_label_fields token** — *unstable — no deprecation shim owed*
: Selector token rendering `Layer.shape_summary` (the L1 across-pass shape
  summary) as a label row; skipped when the field is unset. The default label
  renders the summary automatically on rolled varying multi-pass nodes,
  directly after the title row.

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
  Skipped sites are "not-run", never "passed". Full contract:
  [the runnable tlspec contract](runnable_tlspec_contract.md).

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
  the reconstruction explicitly. See [facets](../facets.md).
