# TorchLens glossary

Short definitions of the names TorchLens uses: what a `Trace`, `Op` or `Layer` is, what its fields
and methods give you, and which keywords each call takes. Names marked experimental may change in
any release. Names removed in earlier releases are listed in [Deprecations](deprecations.md).

## Contents

- [Capture products](#capture-products)
- [Graph records](#graph-records)
- [Selection and storage](#selection-and-storage)
- [Replay and intervention](#replay-and-intervention)
- [Capture accuracy and visualization](#capture-accuracy-and-visualization)
- [Backend-neutral identity](#backend-neutral-identity)
- [Site selectors](#site-selectors)
- [Intervention helpers](#intervention-helpers)
- [Extraction, observers, and admin](#extraction-observers-and-admin)
- [Persistence, containers, and namespaces](#persistence-containers-and-namespaces)
- [Quantities and typed errors](#quantities-and-typed-errors)
- [Distributed capture and cross-rank merging](#distributed-capture-and-cross-rank-merging)
- [Backends](#backends)
- [Experimental parts](#experimental-parts)

## Capture products

**Trace**: The finalized record of one executed path: operations, dataflow edges, module context, selected
payloads, and capture metadata. `tl.trace(...)` returns a `Trace`.

**Trace.provenance / Trace.options / Trace.cost / Trace.source / Trace.health / Trace.graph / Trace.autograd**: Read-only groups that sort a `Trace`'s capture facts by concern. Every capture fact saved with
the trace lives in exactly one group.

- `trace.provenance`: how, when and from what the capture happened. Run fields: `backend`,
  `backend_version`, `backend_config`, `kind`, `start_time`, `end_time`, `duration`,
  `setup_duration`, `cleanup_duration`, `verified`, `verification_reason`, `cache_hit`,
  `cache_key`, `cache_path`, `recording_kept`.
- Provenance identity fields: `fingerprint`, `input_signature_hash`, `input_structure`,
  `input_object_id`, `model_object_id`, `param_hash_quick`, `param_hash_full`, `param_source`,
  `root_entry_point`.
- Provenance thread and scope fields: `owner_thread_id`, `thread_count_start`,
  `thread_count_end`, `thread_activity_detected`, `distributed_scope`, `observer_spans`,
  `is_appended`, and more. `trace.provenance.summary()` returns the capture-provenance text
  block.
- `trace.options`: the stored snapshot of the capture options. Save fields: `mode`, `save`,
  `save_grads`, `save_mode`, `save_args`, `save_raw_input`, `save_transformed_output`,
  `save_rng`, `save_source_context`, `source_context_lines`.
- Other option fields: `backward_ready`, `inference_only`, `structure_only`, `grouping`,
  `grouping_policy`, `seed`, `on_nonfinite`, `input_transform`, `activation_transform`,
  `grad_transform`, `output_device`, `verbose`, `chunk_size`, and more.
- `trace.cost`: compute, memory and timing totals. Fields: `total_flops`, `total_macs`,
  `macs_by_kind`, `forward_duration`, `backward_durations`, `func_calls_duration`,
  `overhead_duration`, `forward_peak_memory`, `total_activation_memory`, `total_param_memory`,
  `total_grad_memory`, `saved_grad_memory`, and more.
- `trace.source`: facts about the model's source code. Fields: `class_name`, `class_qualname`,
  `class_file`, `class_line`, `class_docstring`, `init_signature`, `forward_signature`,
  `frames`, `snapshots`, `class_location`, `forward_location`, `init_location`, `cls`.
- `trace.health`: checks of how complete the capture really was. Fields: `completeness_verified`,
  `completeness_mode`, `num_witness_events`, `num_accounted_witness_events`,
  `num_unaccounted_witness_events`, `escape_detector_verified`, `escape_detector_mode`,
  `num_escape_detector_events`, and more.
- `trace.graph`: facts about the graph's shape. Fields: `edges`, `is_recurrent`, `is_dynamic`,
  `conditionals`, `conditional_records`, `stop_frontier`, `op_equivalence_classes`,
  `relationship_evidence`, `structure_evidence`, `manual_tensor_connections`, `orphans`,
  `tied_param_groups`, and more.
- `trace.autograd`: the backward records. Fields: `backward_passes`, `grad_fns`,
  `grad_fn_calls`, `grad_fn_order`, `has_backward_pass`, `has_grads`, `derived_grads`,
  `backward_root_grad_fn_object_ids`, and more.

**Trace.params filters and totals**: `trace.params.trainable`, `.frozen`, `.executed` and `.unexecuted` are parameter collections;
`len()` counts their Param records and `.numel` adds up their elements (ties counted once);
`trace.params.numel_by_path` counts ties once per module path; `.labels` gives the addresses
(`trace.params.unexecuted.labels`).

**Layer filters**: `trace.layers.compute`, `.recurrent`, `.saved_grad`, `.internal_source` and `.internal_sink`
are Layer collections; on any Layer collection they keep the members that pass the Trace-level
filter, so `len(trace.layers.compute)` counts the compute Layers.

**Op filters**: `trace.ops.compute` and `trace.ops.nonfinite` are Op collections; on any Op collection they
keep the members that pass the Trace-level filter (`Trace.compute_ops`, `Trace.nonfinite_ops`),
so `len(trace.ops.compute)` counts the compute ops and `trace.saved_ops.nonfinite` the saved
ops whose output held NaN or Inf.

**Module filters**: `trace.modules.uncalled` lists the addresses `Trace.uncalled_modules` records; an uncalled
module has no Module record, so the answer is addresses, not records.

**Collection labels (`.labels`)**: Every collection serves `.labels`, the tuple of its lookup labels in collection order.

- `trace.layers.labels`: the Layer labels. `trace.ops.labels`: the counted Op labels.
- `trace.modules.labels`: the module addresses. `trace.params.labels`: the parameter
  addresses.
- The same works on any filtered collection (`trace.layers.compute.labels`). The per-noun
  Trace projections (`Trace.layer_labels`) stay.

**Recording**: A sparse, torch-only event stream returned by `tl.record(model, inputs, save=...)`.
`tl.record` and `Recorder` build it; `Recording` has no public constructor. `tl.record` always
returns one object.

- Converting: `Recording.to_trace()` builds the full graph structure. Recorded ops carry the
  capture-time `equivalence_class` and parameter identity exhaustive capture gives them, so the
  resulting `Trace` groups a reused layer into passes the way `tl.trace` does.
- Payloads: reading a payload that was not selected by `save=` fails explicitly.
- Drawing: a `Recording` has no `draw()` and no facets; `to_trace()` unlocks every
  `Trace.draw` option and `modules[...].facets`.
- `tl.extract` takes only `model`, `inputs`, `layers` and `detach_saved_activations`; every
  other capture option is a `tl.trace` keyword.
- Backward: `tl.record` installs gradient hooks only when `backward_ready=True`, `save_grads=`
  or `default_grad=` is given; without one, `Recording.backward()` raises an error with code
  `recording_backward_not_requested`.
- Trace-parity verbs: `Recording.to_agent_json()`, `Recording.save(path)`, `Recording.fork()`
  and `Recording.do(site, edit)` convert through `to_trace()` and delegate to the Trace verb, as
  do `tl.export.xarray(recording)` and the bundle writer behind `Trace.save` when handed a
  Recording. A typed error from `to_trace` passes through unchanged, and a failed recording
  raises a capture-outcome error (code `N1`) instead of being converted.
- Episodes: `tl.record(..., episode=)` raises a typed error (`record_episode_unsupported`);
  episode capture runs on `tl.trace(model, x, episode=...)`.

**Trace.output / Recording.output**: The model's return value exactly as the model returned it (tensor, tuple, dict, Hugging Face
output object, distribution); for a multi-pass `Recorder`, the last pass's.

- Generation: with `generate_kwargs=`, `GenerationRecording.output` (and its
  `recording.output`) is what `generate()` returned.
- `Trace.output` is the exact return from a live `tl.trace` forward, independent of `save=`
  and output transforms. It is neither transformed `raw_output` nor the saved-payload rebuild
  from `reconstruct_output()`. Traces converted from recordings, forked, or merged from chunks
  have no live forward return.
- Memory: the trace or recording keeps that object alive for as long as it exists (for a
  Hugging Face model this includes `past_key_values`); the object is released when the trace
  or recording is dropped.
- Held in memory only. It never returns `None` for a missing output; it raises a
  `PayloadUnavailableError` instead, with one of these codes:
  - `output_not_stored`: a trace or recording loaded from disk or unpickled.
  - `output_forward_failed`: a failed partial recording (`on_forward_error="return_partial"`).
  - `output_capture_stopped`: a capture that `stop_after=` ended before the model returned.

**Trace.decode_output / Trace.output_table**: The model's output read as predictions. `trace.decode_output(k)` returns
decoded rows, up to `k` per batch item; `trace.output_table(k=5)` returns them as a pandas
dataframe, `k` rows per batch item.

- Classifier (a classification decode): the top `k` classes, with columns `batch_item`, `rank`,
  `label` and `prob`; `decode_output` rows also carry `class_index`.
- Language model (an `hf_text` decode): the top `k` next tokens at the last position, with columns
  `batch_item`, `rank`, `token_id`, `text` and `prob`. `token_id` is the vocabulary index of the
  token, `text` its decoded string (`None` when no tokenizer was available) and `prob` its
  probability under the softmax over the vocabulary at the last position.
- `style=`, `head=` and `tokenizer=` decode again at read time from the retained output tensors
  (see [Semantic I/O](../semantic_io.md)).

**Recording.options / RecordingSettings**: `Recording.options` returns a `tl.types.RecordingSettings`, the read-only record of the settings
the recording was made with, from every argument that configured it. Values are the live objects
the recorder used, never repr strings; there is no public constructor.

- Capture knobs from `capture=CaptureOptions(...)`: `RecordingSettings.save_grads`,
  `RecordingSettings.activation_transform`, `RecordingSettings.grad_transform`,
  `RecordingSettings.save_raw_activations`, `RecordingSettings.save_raw_grads`,
  `RecordingSettings.detach_saved_activations`, `RecordingSettings.lookback`,
  `RecordingSettings.lookback_payload_policy`, `RecordingSettings.seed`.
- Recorder knobs from `recording=RecordingOptions(...)`: `RecordingSettings.default_op`,
  `RecordingSettings.default_module`, `RecordingSettings.default_grad`,
  `RecordingSettings.history_size`, `RecordingSettings.include_source_events`,
  `RecordingSettings.lean` (name may change), `RecordingSettings.max_predicate_failures`,
  `RecordingSettings.on_predicate_error`, `RecordingSettings.on_forward_error`,
  `RecordingSettings.postprocess`.
- From `tl.record` keywords: `RecordingSettings.keep_op` from `save=`, `RecordingSettings.intervene` from
  `intervene=`, `RecordingSettings.stop_after`, `RecordingSettings.storage`,
  `RecordingSettings.narrate`.
- Saved recordings: a recording recovered from disk keeps the plain values (bools, numbers,
  strings); callables and objects are not stored with saved recordings and read `None` there.

**Saved-activation autograd history**: Whether saved payloads stay connected to the autograd graph. The option names may change.
`CaptureOptions(detach_saved_activations=False)`, the `tl.trace` default, keeps every saved
payload graph-connected on every save path; `True` detaches on every one of them.

- The save paths: the save-all default, `tl.module` and `tl.in_module` selectors, op-label
  `save=`, the legacy `trace.run(model, x)` rerun, `trace.run_transaction(inputs=...)` and
  `trace.run_transaction(inputs=..., fast=True)`.
- `tl.extract(..., detach_saved_activations=False)` and
  `tl.record(..., capture=CaptureOptions(detach_saved_activations=False))` (also
  `tl.recording.Recorder`) attach on request; both keep their detached default. `tl.record`
  raises an error for an explicit `True` with `backward_ready=True` (`backward_ready_conflict`).
- A per-site `CaptureSpec(save_grad=True)` stays attached either way, and disk payloads are
  always detached. A payload held back for a deferred selector and spilled to temporary disk
  comes back detached, with a `RuntimeWarning` naming the count.
- Memory: `Trace.cost.pinned_autograd_memory` reports what the attached payloads keep alive:
  the bytes of the tensors autograd saved for every op upstream of a saved site, once per
  storage, excluding Parameters and views of them. It reads `Bytes(0)` when every payload is
  detached or the forward ran without grad, and `None` on a loaded trace (it is never saved).

**PartialTrace**: A recoverable prefix of a failed `tl.trace(...)` capture. A raised capture exception may expose
`exc.partial_trace`: a forward that raises during capture raises `TraceCaptureError` carrying
it, and TorchLens's own errors carry it too. A partial trace is not silently treated as
complete.

**Failed partial Recording**: A `Recording` with `status="partial_error"` and `is_failed` `True`. Use
`on_forward_error="attach_partial"` to attach it as `exc.partial_recording`, or
`on_forward_error="return_partial"` to return it. It carries string-only error metadata,
`num_committed_ops`, and best-effort `last_event_*` fields. It cannot be passed to
`Recording.to_trace()` or `Recording.backward()`.

**Generation recording**: `tl.record(model, input_ids, save=..., generate_kwargs={...})` runs the model's own Hugging Face
`generate(*inputs, **input_kwargs, **generate_kwargs)` once, with its KV cache, and records the
selected sites at every forward it runs. Each decode step records only its new position, so the
cost grows linearly with the number of generated tokens. `seed=` seeds once for the whole
generation. The names may change.

- It returns a `GenerationRecording`. Fields: `.recording` (the multi-pass `Recording`, one
  pass per forward), `.steps`, `.sequences`, `.prompt_length`, `.settings` (the settings that
  decide the tokens, the torch seed and an RNG-state digest), `.not_recorded` (cache tensors,
  the sampling between forwards, the last token's own position), `.n_steps`.
- `.steps` holds one `GenerationStep` per forward. Fields: `step` (0 for the prefill),
  `pass_number` (the recorder pass holding its records; formerly `pass_index`), `kind`,
  `positions` (the absolute positions it computed), `n_ops`, `n_records`, `saved_bytes`.
- Methods: `.records_at(step)` and `.step_of_position(position)`; both raise an error for a missing step
  with `record_generation_step_missing`.
- Not supported: beam search, more than one return sequence, batched prompts, assisted decoding,
  encoder-decoder models, `use_cache=False`, a generate that runs decode steps without a cache,
  `intervene=`, `stop_after=` and partial forward-error modes raise
  `record_generation_unsupported`, whose `reason` field names what is unsupported.

**Streaming generation recording**: `tl.record(model, input_ids, generate_kwargs={...}, generation=GenerationOptions(...))` widens
the generation recorder from "keep the selected sites at every token" to a per-token stream
that stays small at thousands of tokens. `torchlens.recording.GenerationOptions` groups its
options. The names may change.

- `reducers=` maps a site address to a callable applied to the site's payload after every
  forward. Its result (first axis the forward's positions) replaces the payload in the record
  and is recorded in the record's `metadata` (`reducer`, `reduced_from_shape`). The declared
  sites are recorded even without `save=`.
- `torchlens.recording.reducers` ships three reducers, each a named `Reducer` (calling it,
  `Reducer.__call__`, applies the wrapped callable `Reducer.fn`):
  `norm(ord=2.0, dim=-1, position_dim=1)` (one norm per position),
  `top_k(k, dim=-1, position_dim=-2)` (the `k` largest indices per position, the attended keys
  on an attention-probability site) and `probe(fitted, position_dim=1)` (the decision function
  of an already-fitted linear probe: a scikit-learn linear estimator with
  `coef_`/`intercept_`, an `nn.Linear` or a `(weight, bias)` pair).
- `kv_sites=` names decoder layer indexes (or `True` for all) whose new KV-cache slices are
  kept per token as series `kv.<layer>.key` and `kv.<layer>.value`
  (`[positions, kv_heads, head_dim]`); `kv_sites=` alone needs no `save=`.
- `keep_last=N` is a ring buffer: the payloads of older forwards are written to `spill=` (one
  `torch.save` file per step) or dropped, and the record keeps a metadata stub (`payload`,
  `spill_file`).
- `max_saved_bytes=` (512 MiB by default, `None` off) raises
  `record_generation_budget_exceeded` when the retained bytes exceed the budget, projected for
  the whole generation right after the first decode step.
- Memory: what the options do not keep is released as each forward is reduced (a pass's model
  inputs, the fed-back KV cache among them, and the logits of every pass but the newest), so
  memory grows with the token count, not its square.
- The `GenerationRecording` adds `.options`, `.series_names`, `.series(name)`,
  `.peak_saved_bytes` and `.evicted_steps` (step to `"spilled"` or `"dropped"`).
- `.series(name)` returns a `TokenSeries`: `name`, `positions` (a `range` of absolute
  positions), `values` (one row per position), and `len(series)` (`TokenSeries.__len__`), the
  number of positions. Values are read back from the spill when evicted. It raises an error for an
  unknown or fully dropped series with `record_generation_step_missing`.
- `record_generation_reducer_invalid`: a reducer that raises, returns a non-tensor or a value
  whose first axis is not the forward's positions, a site a forward did not record, a duplicate
  site per forward, or a `kv_sites` layer the cache lacks.
- `record_generation_options_invalid`: a malformed `GenerationOptions`, or `generation=`
  without `generate_kwargs=`.
- The cost benchmark is `benchmarks/generation_stream_cost.py`.

**Capture outcome**: The frozen `Trace.outcome`, `Recording.outcome`, or `PartialTrace.outcome`. Its status is one of
`COMPLETE`, `STOPPED`, `ABORTED_NONFINITE`, `FAILED`, `UNATTESTED`, or `UNKNOWN`; failed outcomes
also identify the phase. See [Capture outcomes](capture_outcomes.md).

**Inference-only capture**: A capture whose forward ran under `torch.no_grad()`, so it built no autograd graph. Request it
with `CaptureOptions(inference_only=True)`; an explicit `inference_only=True` is never changed.

- Implied: a `tl.trace` or `tl.record` entered under `torch.no_grad()` or
  `torch.inference_mode()` implies it, unless `backward_ready`, `save_grads` (or `default_grad`
  on `tl.record`, `replayable` in `capture=tl.options.CaptureOptions(...)` on `tl.trace`) asks
  for a backward. An explicit `inference_only=False` keeps it off.
- The implied flag holds only when the forward really discarded its graph: a forward that turns
  grad back on (`with torch.enable_grad():`) and returns an output with autograd history is
  not inference-only, and backward stays available.
- Reporting: `Trace.options.inference_only` and `outcome.inference_only` report the effective
  value; a later `backward()` refuses and names the cause (`recording_backward_inference_only`
  on a `Recording`).

**carry_state (CaptureOptions) / pure-observer capture**: A default `tl.trace` is a pure observer: it restores the model's state after the forward, on
success and on failure. `CaptureOptions(carry_state=True)` opts into forward-equivalent state
instead. It is a capture option only and is never saved with the trace.

- Default restore: buffers (persistent and non-persistent), every module's `training` flag and
  the global random states (torch, live CUDA devices, Python, NumPy) are snapshotted before
  the forward and restored afterwards, so a train-mode BatchNorm's running statistics do not
  advance. A buffer the forward changed is written back only where its bytes changed, and
  trace payloads that alias it keep their forward-time values.
- Plain module attributes are restored too: a rebound or removed attribute gets its original
  object back, a tensor or builtin `list`/`dict`/`set` changed in place is restored in place,
  and attributes the forward added are removed. What the forward wrote is read from the trace
  (`Module.attribute_writes`).
- Containers: builtin containers are snapshotted at any size (a tensor inside one is cloned
  whatever its size). Custom objects are tracked by identity: a rebinding is undone, a change
  inside them is neither undone nor recorded. A container nested more than 256 levels deep is
  tracked by identity below that level and named in a `capture_state_restore_incomplete`
  warning. Lazy modules are left as their materializing forward leaves them.
- `CaptureOptions(carry_state=True)`: buffers, modes and plain attributes advance exactly as
  one direct `model(x)` would, and the attribute writes are recorded the same way. It is the
  same spelling as `trace.run_transaction(inputs=..., carry_state=True)`.
- Random streams: no capture advances the caller's random streams under either setting (each
  capture draws from its own seed and restores the global engines). A capture-cache hit runs
  no forward, so nothing advances.
- An item that cannot be restored is named in one `capture_state_restore_incomplete` warning.

**Module.attribute_writes / AttributeWrite**: The plain instance attributes one module's forward wrote during the capture, as a read-only
mapping `{attribute name: AttributeWrite}`, for example
`trace.modules["encoder.attn"].attribute_writes["last_weights"].value`.

- `torchlens.types.AttributeWrite` is a frozen record. Fields: `kind` (`"added"`, `"replaced"`
  for a rebinding or an in-place change, `"removed"`) and `value`, what the forward left.
- `value` is the object itself when the forward bound a new object (the trace keeps it alive,
  so a large stashed tensor costs its full memory for the trace's lifetime); a copy when the
  forward changed the original object in place (builtin `list`/`tuple`/`dict`/`set`
  containers are copied recursively, a tensor that existed before the forward is cloned
  whatever its size, a tensor the forward created is kept by reference); `None` when removed.
- The mapping is `{}` when the forward wrote nothing on the module, and `None` when no record
  exists (a loaded trace, a lazy module, a non-torch backend).
- Recorded by default and under `CaptureOptions(carry_state=True)`. A change inside an
  identity-tracked object is not recorded; a capture that re-ran the forward (escape rescue)
  records the net effect of both runs.
- Kept in memory only: never saved, and left out of `to_pandas()`.

**Bound-method root**: The roots `tl.trace` accepts: an `nn.Module` or a bound method of one, e.g.
`tl.trace(model.generate, ids, ...)` (also accepted by `tl.validate`). All spellings are
provisional.

- Owner: resolved via `method.__self__`. It is the root module `self` of a TorchLens-authored
  wrapper root whose forward calls the exact bound method exactly once.
- Names read as on a plain `tl.trace(model, ...)`: module addresses, layer labels, module
  paths, drawings, pandas exports and saved files (`transformer.h.5.mlp`, no wrapper prefix).
  The owner's own calls inside the method are the root's calls (`self:1` .. `self:N`,
  `trace.modules["self"].num_calls == N`).
- Identity: the capture's identity reads `type(owner).__name__`,
  `Trace.provenance.root_entry_point` records `"bound_method:<owner_qualname>.<method>"`, and
  the root op record (the output-boundary ops) carries `tl_authored_root=True`. Both are saved
  with the trace and checked on load; a failed check raises an error.
- Episodes: on bound-method episode captures `EpisodeOptions.stepped_module` defaults to the
  owner.
- Errors: rerunning or appending to a bound-method capture raises
  `rerun_entry_point_unsupported` (an older saved trace without the entry-point fact raises
  `root_entry_point_unavailable`); replay is unaffected. Other callables (closures, bare
  functions) raise `model_type_unsupported`.

**Episode capture**: One multi-step generation run, wrapped and captured as a single product (`capture_kind=episode`).
`tl.trace(episode_root, x, episode=EpisodeOptions(stepped_module=...))` records the declaration
and writes a per-step status record at `trace.annotations["episode"]`. Meant for diagnosing runs
of tens of steps (cost grows faster than the step count). All episode names may change. See
[Episode capture](episode_capture.md).

- Record header (format version 2, `episode_ledger_version=2`): the declared `step_output_kind`
  (`tokens`/`digest`/`none`), where the step output was read from (`step_output_from`), the
  `step_axis` and the `step_output_positions` each row was read from, and the `capture_digest`
  that ties the record to this capture.
- Joins between steps: `step_join` grades each measured step-to-step join
  (continuous/forced/transformed/declared/exogenous/unchecked), and claims that depend on the
  episode raise an error across a measured break. On captures with interventions (COUPLED) the
  header also holds the deterministic `intervention_digest`.
- Record rows: a status of `complete`, `interrupted` or `absent`; the generic `step_output`;
  per-channel digests of the carried state on entry and exit, `entry_state_digest` and
  `exit_state_digest` (`None` means not measured; they replace `cache_len`, which is removed);
  the measured per-step `fire_count` on coupled captures (zero or several fires are normal
  results); and the managed random seed at step entry.
- Coupling: `episode=` together with `intervene=` runs COUPLED. `trace.episode_coupling`
  recomputes the capture digest and compares it with the record, and serves fire counts plus
  per-segment facts that never span a measured break.
- Step selector: `torchlens.at_step(*steps)` selects by step, live or after the capture, using
  the saved `Op.episode_step` stamps.
- Replay builds a fresh record or raises `episode_coupled_replay_underivable` (on both `run()`
  engines); `do()` edits set aside the episode evidence they inherit.
- The record is a report, not a final authority. It is saved with the trace and checked on load;
  a failed check raises an error.

## Graph records

**Op**: One executed callable invocation. Ops are the dataflow graph's nodes and have unique full labels
such as `relu_1_2`.

**Layer**: The user-facing grouping of Ops that run one place in the code. Running that place
again (a loop, a recursion, a comprehension) is another pass, addressed with a one-based
suffix such as `linear_1:2`.

- Passes: `op.pass_number` is the pass and `num_passes` the count. `graph.is_recurrent` is
  true when some layer runs more than one pass.
- Module calls: a module object called again is another module call (`m:1`, `m:2`), a
  separate concept, recorded for every module. For a module that owns state (a parameter or
  buffer, its own or a descendant's) each op inside it runs once per module call, so its
  passes match the module's calls. A module with no state (`nn.ReLU`, `nn.Dropout`) is read
  as the function it is: its ops group by where the stateful code around it called it, so
  one `nn.ReLU` called on two lines is two layers and one called from a loop line is one
  layer with a pass per iteration.
- Two different places are two layers even with the same weights, so a loop written out by
  hand is N layers.
- Related: the exact rule (scope, call site, operation, output slot) and the case table are in
  `docs/guides/grouping.md`.

**Module**: A captured framework module or backend-neutral module-like object. A module contains Ops but is
not itself an Op.

**Parameter / Buffer**: Registered model state. Parameters are trainable state; buffers are non-parameter state and may
have multiple captured versions.

- Versions: every recorded write (a reassignment, a `.data =` or `set_` swap, a tracked
  in-place write) is its own version whatever value it wrote, so a capture records the same
  buffer structure whatever values the buffers start from.
- Collapse labels: a training-mode `BatchNorm2d` collapse label can say "5 layers total"
  because it counts the parameter leaves `weight` and `bias` plus the buffer leaves
  `running_mean`, `running_var`, and `num_batches_tracked`; it is not claiming five BatchNorm
  calls.
- In-place writes to a Parameter: an in-place op on a Parameter inside `forward`
  (`with torch.no_grad(): self.temp.clamp_(lo, hi)`) is captured as an op whose `params`
  holds that Parameter. Every later read of the Parameter in the same pass binds to that op
  as a graph parent, not as a parameter edge, so it is absent from that read's `params`. The
  receiver may be given by keyword, as the `input` keyword of `torch.clamp_`. A frozen
  Parameter written in place with a grad-requiring operand ends the pass a non-leaf, exactly
  as in eager. An `out=` write into a Parameter (`torch.add(a, b, out=p)`) is not captured
  yet; validation reports it as a completeness failure.
- Drawing: `draw()` shows such a Parameter as a grey cylinder labeled `parameter <name>` in
  its owning module's box, with an edge to each op that reads its pre-mutation value, under
  both layout engines (`dot` and `rank`). The node is drawn when the first in-place op on the
  Parameter is drawn as its own node: a collapsed module box or `skip_fn` that hides that op
  hides the node with it, and a `module=` focus draws it only when the focused module is the
  owner or contains it. Parameters never mutated get no node. The legend row's swatch is the
  trainable-parameter grey.
- Module-held tensors: a plain tensor a module already holds when the forward starts (a
  tensor attribute, a list/tuple item, or a dict value such as a warm-filled attention-bias
  cache, nested up to four container levels) is captured as a buffer too. It is addressed
  `<module>.<attr>`, `<module>.<attr>.<index>`, or `<module>.<attr>[<key>]` (not
  `<module>.<attr>.<key>`), with one `[<index>]`/`[<key>]` per nested level, e.g.
  `attn.attention_bias_cache['cpu']`.
- Keys: a key is rendered as its `repr` with `.`, `:` and `%` percent-escaped, so the last `.`
  always separates the owning module; keys that print alike get `#2`, `#3`, ...
- Plain objects: a public tensor attribute of a plain (non-Module) object the module holds,
  such as a `torch.distributions` prior or a diffusers scheduler built in `__init__`, is a
  buffer too, addressed `<module>.<attr>[<name>]` (e.g. `vae.prior[scale]`). Only the
  object's own attributes are read, not containers inside it.

**GradFn**: A live backward-autograd node captured while logging the backward pass. Portable artifacts
do not contain PyTorch's live autograd graph.

**Facet**: A named semantic view into one captured value, such as an attention projection or an LSTM state.

### ATen operation profile (experimental)

With `CaptureOptions(aten_ops=True)`, a capture also records the lower-level ATen operations each
PyTorch call ran, as `AtenOp` rows; `aten_ops="values"` keeps each call's output tensors too. Read
them with `Trace.aten_ops`, `Trace.has_aten_ops`, `Trace.aten_gaps` and `Op.aten_ops`. Reading
them on a capture that did not record them raises `AtenOpsUnavailableError`. The recorded rows are
saved with the trace. The row fields may change in any release.

**aten_ops (CaptureOptions)**: Opt-in switch (default `False`) that makes the torch capture record the ATen operations each
logged torch call dispatched to, as `AtenOp` rows. It works through a dispatch mode installed
for the forward and, with `backward_ready=True`, for `Trace.backward()`.

- Values: `True` keeps the rows value-free; `"values"` also keeps what each call returned in
  `AtenOp.out`, under the capture's save policy (the `save=` selection, `save_mode`,
  `output_device`, `save_budget`). Any other value raises `InvalidArgumentError`
  (`aten_ops_invalid`).
- Cost: the default installs no dispatch mode and adds no per-op work.
- Saving: the switch is a session-time setting that is not saved (load restores `False`); the
  recorded rows themselves are saved, so a loaded trace keeps `has_aten_ops`.
- Reruns: `run()` refreshes inherit the switch, and `run()` results, fast or not, inherit the
  recording of the trace they refresh.
- Scope: non-torch backends record nothing. GPU kernel recording is out of scope.

**Trace.has_aten_ops**: `True` when the trace carries recorded ATen operations: after a capture with
`CaptureOptions(aten_ops=True)` (a present-but-empty profile counts) and after loading an
artifact saved from one; `False` otherwise. The one accessor that never raises. Like every
boolean it is named with `is_`/`has_`.

**Trace.aten_ops**: Every recorded `AtenOp` in dispatch order, forward rows then backward rows, as a tuple. Raises
`AtenOpsUnavailableError` (`aten_ops_unavailable`) when the capture recorded none, so an empty
answer never stands in for "recording was off".

**Trace.aten_gaps**: The stretches where ATen rows are missing by design (the profile's
`mode_paused_interior` records): each names the capture phase, the dispatch sequence it spans
and the owning op references of a stretch where the dispatch mode was paused, for example inside
a user-injected intervention. An empty tuple means the recorded rows are complete. Raises
`AtenOpsUnavailableError` when the capture recorded none.

**Op.aten_ops**: The ATen operations this op's torch call dispatched to, in dispatch order: the rows whose
resolved owner references carry the op's label. An op whose call dispatched nothing observable
yields an empty tuple while the trace still `has_aten_ops`. Raises `AtenOpsUnavailableError`
when the owning trace recorded none.

**AtenOp.out**: What the ATen call returned (experimental), mirroring `Op.out`: a tensor for a single-tensor result, a tuple of
tensors (one per `output_tensor_facts` entry, in the same order) for a container result, `None`
when no value was kept. Recorded only under `CaptureOptions(aten_ops="values")`; otherwise every
row reads `None`.

- Snapshots: each value is copied below autograd at the moment the call returned. It never
  carries a `grad_fn` (`detach_saved_activations=False` attaches `Op.out`, not these); an
  in-place call's value is the mutated tensor as returned; a view's value is the view's
  content; aliasing between rows is not preserved (`view_copy_kind` and `storage_alias_group`
  describe it).
- Which rows keep values: this follows the owning torch call. A forward row keeps its value
  when an `Op` of its call saved its output (`has_saved_activation`), so `save=` restricts
  ATen values the way it restricts `Op.out`; a dropped owner frees its values during the
  forward. Backward rows (with `backward_ready=True`) and orphan rows keep values
  unconditionally.
- Save policy: values honour `save_mode` (`"reference"` becomes a copy for in-place calls),
  `output_device` and the `save_budget` (each value is admitted like an activation; the budget
  error names `aten:<operator>.<overload>`).
- Streaming: values are not retained when activations stream to disk or an out sink is active;
  every row reads `None` (a typed gap, see limitations).
- Saving and loading: values persist and load with `Op.out` (`BLOB_RECURSIVE`);
  `include_outs=False` drops them while the facts stay.
- Checks: the `aten_values_consistent` invariant checks every kept value against its facts
  and, under `save_mode="copy"`, that some row beneath each saved op equals the op's `out`.
- Reruns: a `run()` refresh re-records the profile against the new input, so values follow
  `Op.out`. The fast tier (`run_transaction(inputs=..., fast=True)`, and the guarded fast
  rerun behind `run(model, x)` on a selective-save trace) never re-records ATen values, so it
  raises
  `run_capability_unavailable` (stage `fast_run_aten_values_unsupported`) for a values capture,
  and the older `run(model, x)` path falls back to a full capture.

**AtenOpsUnavailableError**: `torchlens.errors.AtenOpsUnavailableError` (`CaptureError`, `RuntimeError` lineage), code
`aten_ops_unavailable` in `exc.fields["code"]`. Raised when `Trace.aten_ops`, `Trace.aten_gaps`
or `Op.aten_ops` is read on a capture that recorded no ATen operations.

- Causes: the switch was off, the artifact was saved without it, or the trace is a
  `run(fast=True)` result.
- Remedy: the message names `CaptureOptions(aten_ops=True)`; check `Trace.has_aten_ops` first
  to avoid the error.

`AtenOp` is one dispatcher call measured during a concrete capture (value-free unless
`aten_ops="values"` filled `out`). Its label is internal to one capture, so string lookup on a `Trace` does not find it. `OpRef`
points from an ATen row back to its `Op` by row index, label and call id, and all three must agree.
A `mode_paused_interior` entry says TorchLens paused its ATen recording around a strict
Tensor-subclass constructor; rows and counts for that op are then lower bounds, and TorchLens does
not invent rows for the part it could not see.

`SuperAtenOp` aligns observed rows positionally by `decomposition_slot`. Its `comparison_status`
distinguishes equal-schema coverage, different-schema coverage, proven sparse membership, and
coverage that cannot be decided because at least one member has a recording gap. Lining rows
up by position does not prove they compute the same thing.

### Kernel telemetry (experimental)

An optional CUDA profiler adapter records the GPU kernel launches behind each recorded ATen call,
as `KernelLaunch` rows saved with the trace. Its names may change in any release.

`KernelLaunch` is a measurement from the profiler; it is not part of the graph and is not used
for replay. The adapter
places a unique marker around each redispatched ATen call and joins CUDA runtime calls to kernels
and memory copies through Kineto correlation identifiers; it never joins by operator or launch-name
substring. The relation persists in the artifact's kernel annotation, but this release has no
public reader for it: the former per-op and per-call kernel views are removed (see
[deprecations](deprecations.md)) and raise the typed `facade_redirect` error. A public
kernel-recording switch is planned for a later release.

An unavailable profiler session returns one fact-free row with
`attribution_status="unavailable"`; it never reports zero kernels. When an op has a
`mode_paused_interior` entry, every launch set and any count derived from it is a lower bound:
TorchLens does not invent launches for the paused region. See
[Kernel telemetry](kernel_telemetry.md).

### Attribution kit (experimental)

`torchlens.attribution` builds input-attribution methods (Integrated Gradients, occlusion, CAM
overlays, attribution graphs) on top of TorchLens's gradients, selections and interventions. Its
names may change in any release.

Integrated Gradients always reports its signed completeness residual against the exact endpoint
target delta. Occlusion always names its replacement baseline. CAM overlays visibly state the
native measured map resolution, while their larger display grid is labeled as interpolation.

`build_graph` builds an attribution graph from one prompt's capture and a set of transcoders:
typed feature, error, token and logit nodes whose edges are direct linear effects along a frozen
graph, with every feature and logit node accounted for by its incoming edges plus named constant
terms (`AttributionGraph`).

A `SiteTarget` passed as `target=` attributes a hidden layer's output (or one unit of it) instead
of the model output, and with `sae=` one latent of that SAE's encoding of the output (experimental). Layer methods given only integer inputs start their input path at the
embedding output.

`deletion_curve` and `insertion_curve` score an attribution by removing or revealing its
most-attributed input features in steps (each step a recorded intervention), and
`randomization_test` checks that a method's maps change when the model's layers are
re-initialized. `road_curve` is ROAD, the deletion curve with removed pixels filled by noisy
linear imputation from their neighbours.

`causal_audit` grades an internal attribution (a layer attribution, a direct-logit-attribution
table or a one-backward read table) by ablating the units it ranks, through forks of an
intervention-ready trace, and comparing the measured effects with the claim. Each row names its
`site` and its `component`, and a direct-logit table's `constant_row` is reported apart. An EDGE
table from `torchlens.mechinterp.edge_scores` is audited by patching each edge alone with the
`corrupted_input` inputs and `metric` it was scored with (`edges` names a subset); every set is
patched in one call, into one receiver or several, and `sets_not_measured` stays empty.

`TargetSpec` refers only to the intervention selector record; it is not an alias for the
callable/index/site attribution target used inside `torchlens.attribution`. See the
[attribution reference](attribution.md).

### Mechanistic interpretability kit additions (experimental)

Additions to `torchlens.mechinterp`, the transformer analysis kit: factored matrices, QK and OV
circuits, per-neuron MLP contributions, component graphs, edge attribution patching, circuit
faithfulness, ablation controls and feature visualization. Its names may change in any release.

`component_graph(trace)` returns a `ComponentGraph`: read-only `nodes`, `receivers` and `edges`
keyed by MIB/EAP-IG vocabulary name (`input`, `a{L}.h{H}`, `b{L}`, `m{L}`; receivers
`a{L}.h{H}<q|k|v>`, `m{L}`, `logits`; edges `"<sender>-><receiver>"`). Each `ComponentReceiver`
holds the captured pre-norm input (`value`, from the op `input_label` entering the norm at
`norm_address`) and its `senders`. Nodes and receivers carry `layer_index` (transformer block
index) and `head` (head index), each `None` where it does not apply.
`ComponentSender.module_address` is the owning module (embedding, attention or MLP; the
attention module for head and bias nodes) and `ComponentReceiver.module_address` the reading
module (attention, MLP, or the unembedding head). Each `ComponentEdge` exposes the sender's
output as its `contribution`. `identity_receipt` records the in-API check that every receiver
input equals the sum of its contributions. Each attention output bias is its own node `b{L}`. A
`ComponentEdge` is also a selection (`__selection__`): an EDGE selection of the pre-norm input
edge for `m{L}` and `logits` receivers, an ACT selection of one head's q, k or v projection slice
otherwise.

`patch_edges(trace, edges, *, corrupted_graph)` returns a fork of `trace` in which every receiver
the edges enter reads `input - sum(s) + sum(s_corrupt)` over its patched edges (`corrupted_graph`
is the corrupted input's `ComponentGraph`). `input` and `s` are the patched forward's own values
(the clean values when one receiver is patched), so a circuit (everything outside a kept set
patched) is one call. The trace needs `capture=tl.options.CaptureOptions(replayable=True)`.

`edge_scores(model, inputs, *, corrupted_input, metric, method="eap_ig", steps=None, example_reduction="mean")`
returns a one-backward `ReadTable` of `EDGE` rows, one per component-graph edge, addressed
`(edge name,)`: the edge attribution patching score `(s_corrupt - s_clean) . dM/d(input of t)`,
summed over positions and reduced over examples (`example_reduction`, `"mean"` or `"sum"`).
`metric` maps the unembedding output to one value per example. `method="eap"` takes the gradient
at the clean input; `method="eap_ig"` averages it over `steps` points (default 5) from the
corrupted to the clean `input` node (left Riemann sum).

`circuit_performance_ratio(faithfulness, *, fractions=None, log_scale=False)` (CPR) and
`circuit_model_difference(...)` (CMD) are the trapezoidal areas under `f` and under `|1 - f|` of a
faithfulness curve over kept edge fractions (`fractions` default: the ten MIB fractions 0.001 to
1; `log_scale` integrates over the log fraction).

`qk_circuit(trace, attention)` and `ov_circuit(trace, attention)` return a `FactoredMatrix` of
shape `[num_q_heads, d_model, d_model]` built from `head_weight_views` (TransformerLens's
`model.QK` / `model.OV` convention; biases excluded; under grouped-query attention each query head
pairs with the KV head that serves it). `mlp_neuron_contributions` returns a `ComponentStack` with
one row per hidden neuron (`hidden[n] * W_out[n]`, read from the output projection's captured
input), a verified bias constant, and the stack grading; unselected neurons under `neurons=` fold
into one exact `other_neurons` row.

`patch_query_heads_grid(model, clean_input, corrupted_input, metric, site="v", heads=None, budget=None, trace_kwargs=None)`
returns a `PatchGrid` with axes `("layer", "head")` whose head coordinates are QUERY heads, on
grouped-query attention. Each cell patches one query head's copy of its key/value group
(`site="v"` for the value copy, `"k"` for the key copy) at the op that expands the grouped tensor
to the query heads, so the other heads of the group are untouched; a k/v-site `grid` can only
patch the whole group. Receipts add `effect_site` (the interceptor wording), `site`,
`num_q_heads`, `n_kv_heads` and `expansion_ops`, and every cell fires exactly once. As for every
`PatchGrid`, `values` are patched-run metrics and the receipts carry the unpatched baselines
`clean_metric` and `corrupted_metric`. Errors: fused kernels with no expansion op raise
`mi_kv_expansion_unavailable`; several expansion ops raise `mi_kv_expansion_ambiguous`; equal
query and key/value head counts raise `mi_kv_heads_not_grouped`.

## Selection and storage

**Predicate**: A site selector built with `tl.func`, `tl.module`, `tl.in_module`, `tl.label`, and boolean or
temporal composition. `save=` is the only argument that chooses what to save.

- Removed arguments: `keep_op=` and `keep_module=` raise `TypeError`.
- List form: on `tl.trace`, `save=` also takes a list (layer names, pass-qualified labels,
  module addresses, integer indexes, substrings, `'all'`, `'none'`), with final-label
  selection resolved after capture.
- Statistics: `save="stats"` is a third level beside `'all'` and `'none'`: reduce and discard,
  one `tl.stats.RunningStats` per op and no payload (see Streaming statistics).

**Capture options**: `CaptureOptions` configures capture mechanics. Select payloads with the `save=`
argument on `tl.trace` and `tl.record`; payload selection is not a public field
of `CaptureOptions`. Preview saving is not a capture option. `module_filter`
remains an additional gate on the selected payloads.

**Lookback**: A bounded window of earlier metadata or detached payloads retained so a later temporal predicate
can select its predecessors.

**Save mode**: The activation-retention policy: `copy` isolates the retained value, `reference` preserves the
live autograd-capable object, and `view` intentionally leaves aliasing visible.

**Streaming storage**: A `storage=tl.to_disk(...)` capture that writes selected payloads while capture proceeds. It is
distinct from saving a completed in-memory Trace.

- Async writes: `trace` payload writes run in the background during the forward, on one writer with a
  bounded queue, by default (`to_disk(..., async_writes=...)`; the name may change). Payloads are
  copied when submitted, so a later in-place change never reaches the saved file, and writes
  land in submission order. `max_pending_bytes` (default 256 MiB) blocks capture when the disk
  falls behind. A failed write raises a typed `ArtifactError` and marks the temp bundle
  `PARTIAL`. Finalization waits for every pending write before the bundle publishes.
- `tl.record` streaming stays synchronous and raises an error for an explicit `async_writes=True`.
- Model output: the output the forward returns stays in memory (the output op's payload may
  alias it).
- No reruns: a streamed trace does not run. `trace.run(...)` on it raises
  `trace_option_unsupported` (every way to run, `inputs=` and `fast=True` included), because a
  run would re-capture every payload into RAM while the bundle kept the first capture. Capture
  again with `to_disk` to a new path, or without `to_disk`, and run that trace (a trace loaded
  from the bundle runs through its loaded providers).
- Paths: a `path` that already exists raises `bundle_path_exists`, as does `Trace.save`
  or `Bundle.save` to an existing path without `overwrite=True`. A `path` whose final component
  is a symlink raises an error (the same final-component rule `Trace.save` and `tl.load` apply; a
  symlinked parent passes).

**Capture cache**: The opt-in `cache=True` content-hash store of finished captures.

- A hit requires all of these to match: the model's tensor content (including device and
  `requires_grad`), training flags, module tree and `forward` code, plain instance attributes,
  user-registered module hooks, inputs, and capture configuration.
- Entries are HMAC-authenticated with a per-user secret kept outside every cache directory
  (`$XDG_STATE_HOME/torchlens`, default `~/.local/state/torchlens`), so a copied-in or
  restored cache directory cannot forge entries.
- Entries are bounded by entry count and bytes.
- `tl.clear_cache()` empties the cache and leaves the secret alone.

## Replay and intervention

**Intervention**: A value or gradient edit applied at a selected live site.

- Syntax: `tl.when(predicate, helper)` binds a selector to helpers such as `tl.zero_ablate()` or
  `tl.scale(...)`. `spec.bind(model)` returns a bound executor that applies a spec to live calls.
- Lazy outputs: a synchronous generator or iterator returned by a bound call (for example
  `bound.generate(...)`) stays steered for every step, and its report settles on `last_report`
  when the iteration is exhausted, closed or raises. An async lazy output (async generator or
  iterator, coroutine, awaitable) raises `bind_lazy_output`.
- Module-boundary rules: `tl.module(address)` edits the tensor leaves of a module output that is a
  tensor or a supported container (tuple, list, dict, namedtuple, dataclass, HF model output). An
  output of another type (a distribution object) holds no addressable tensor site, so the rule
  records that it fired zero times there, both under the bound executor and during capture.

**Zero-fire policy (`on_zero_fire`)**: One setting on every intervention entry point (`spec.bind`, `tl.trace(intervene=)` and `Trace.run`) for an
intervention that fires at zero sites (experimental). The zero fire is always recorded on that entry point's own record;
the setting then decides what else happens, the same way everywhere.

- Where it is recorded: the bind report's `zero_fire_rule_ids`, the capture's
  `annotations["unmatched_capture_selectors"]` and intervention audit, the rerun's
  `last_run["hooks_unfired"]`.
- Values: `"warn"` (default) emits the entry point's warning (`bind_zero_fire`, `trace_zero_fire`,
  `rerun_zero_fire`); `"error"` raises the entry point's typed error with that code; `"disclose"`
  records only. Any other value raises `zero_fire_policy_invalid`.
- Unmatchable rerun edits: a rerun edit that cannot fire live (`rerun_selector_unmatchable_live`)
  raises an error before the policy applies, whatever the setting says.
- Non-tensor module outputs: when the rule targeted a module (a plain `tl.module` union at
  `tl.trace(intervene=)`, a module entry at a rerun, a module rule under `spec.bind`) whose output
  is not a tensor, the warning, the error and the fast rerun's `fast_rerun_graph_unsteered`
  error carry `fields["reason"] == "module_output_not_tensor"` and a `reason_detail` sentence.
  The sentence names the output type and, for a `torch.distributions` output, the parameter output
  paths to target instead, for example:
  `module 'policy' output is a Normal, not a tensor; target its parameter ops instead (output paths .loc, .scale)`

**Bound intervention executor**: `spec.bind(model)` pairs a `tl.Recipe` with a base model as a capture-free callable and returns a
`tl.intervention.BoundModel` (experimental). It runs the model with the spec's edits held live and returns the
model's own output.

- Attributes: the read-only `BoundModel.recipe` is the bound `tl.Recipe`, and `BoundModel.model`
  is the base model.
- Behaviour: it records only a firing report (`.last_report`) and supports HF `generate` with the
  KV cache on or off. It is not an `nn.Module`, and bindings never nest. A rule that never fires
  is recorded on `.last_report` and handled by `on_zero_fire` (see **Zero-fire policy**).
- Decode steps: the binding counts root forwards, so `torchlens.at_step(k)` on a bound call or
  `generate` names the k-th forward (step 0 is the prefill). This works capture-free on op-level
  rules and as `tl.module(address) & at_step(...)` boundary rules. Outside a binding or an episode
  capture the qualifier still raises `episode_step_selector_without_episode`.
- Gradient sites: `tl.grad(...)` raises `bind_rule_unsupported`, because the bound executor
  replaces forward values only. Edit gradients with
  `trace.do(tl.grad(site), edit, model=model, inputs=x, loss=...)`, or capture with
  `backward_ready=True, intervene=tl.when(tl.grad(site), edit)` and call `trace.backward(loss)`.

**Recipe and Rule**: `tl.Recipe` is the immutable multi-clause intervention spec that `tl.when(...)` builds and every
intervention entry point accepts (experimental). Each clause is a `tl.intervention.Rule`.

- `Recipe.rules`: the tuple of clauses. Each `tl.intervention.Rule` has `target` (the selector or
  predicate) and `edit` (the helper or callable).
- `Rule.signal`: a read-only property derived from the target, never a field and never saved. It
  is `"backward"` for a gradient site (`tl.grad(...)`) or a node selector, `"forward"` for an
  activation site, `"both"` for a union of the two.
- No direction field: the site decides the pass. `Rule` has no `direction` field, and
  `Rule(..., direction=...)` (or the old third positional value) raises
  `intervention_direction_invalid`.
- `Recipe.merge(*others)` (also `recipe_a & recipe_b`): returns one `tl.Recipe` joining the
  clauses; a duplicate clause raises `spec_rules_duplicate`.

**Recipe.save and tl.load of a recipe**: `Recipe.save(path, level="portable", overwrite=False)` writes the authored rules alone as a
`.tlspec` directory, and `tl.load(path)` returns them as a `tl.Recipe` (experimental).

- Contents: the manifest carries `kind: "recipe"` (no trace, model or fire records); tensor
  arguments travel in `blobs/recipe-<nnnn>.safetensors`.
- Round trip: the loaded recipe's rule identities (`Recipe.digest`) equal the saved ones.
- Errors: a predicate WHERE term or an opaque callable edit has no portable form, so saving it
  below `"audit"` raises `OpaqueCallableError`. An audit record stores it as a callable placeholder,
  and `tl.load` of that record raises `recipe_audit_only`. A hand-edited or foreign record raises
  `recipe_record_malformed`. `level="runnable"` raises `artifact_save_level_unsupported`: a
  recipe carries no model to run.
- Related: the saved intervention artifact (`Trace.save_intervention`, `kind: "intervention"`) is
  the form tied to a capture and still loads as an `InterventionSpec`.

**Trace.recipe and Trace.clear_edits**: `Trace.recipe` is a frozen snapshot of the trace's current intervention recipe. Clearing edits,
below, is the one way to detach an edit: `do` returns the trace, not a removable handle.

- `Trace.clear_edits(site=None, handle=None, *, strict=False)`: removes the sticky edits staged
  with `do` by site or by handle id, marks the recipe stale and returns the trace. With
  `strict=True` it raises an error for a call that names neither or matches nothing.
- Timing: removal takes effect at once, like a torch hook handle. A rerun or backward pass in
  progress skips the removed hooks from the next hook site it reaches, while the site firing at
  that moment finishes its queued hooks; afterwards the recipe is the same as after a removal
  between runs.
- Mutation warning: editing a root trace warns `TraceMutationWarning` once per root trace. A
  `fork()` never warns, and `torchlens.errors.quiet()` quiets the session. There is no per-call
  opt-out keyword: the `confirm_mutation` option is gone from every entry point and from
  `InterventionOptions`.

**Region instances**: A region edit's eligible region is split into one instance per pass, each a `RegionInstance` (experimental). A
replacement returning several exits returns a tuple of `len(exit_labels)` tensors.

- `RegionTarget.labels`: the region's pass-qualified member op labels, in execution order.
- `RegionInstance.labels`: that instance's member labels (constructor field `labels=`).
- `RegionInstance.exit_labels`: its exit op labels, in the order the replacement function must
  return its tensors (constructor field `exit_labels=`).

**allow_changed_params** (`InterventionOptions`, `ReplayOptions`): Lets replay run with parameters that changed since capture (experimental). Without it (default `False`), every
replay path raises an error rather than silently computing with the new value.

- Detection: a replay template holds the live `nn.Parameter` each call consumed, and capture
  records its version counter, storage pointer and shape. Detected changes include an optimizer
  step, `load_state_dict`, an in-place op under `torch.no_grad()` and `param.data = t`.
- Error: `replay_param_changed_since_capture` (`ReplayPreconditionError`, naming the parameter
  and the op).
- Where it is checked: `do()` on the replay engine (a single site, a mapping of several edits, or a
  resolved selection), `do(..., intervention=InterventionOptions(differentiable=True))`, and later
  replays of every fork (a rerun fork checks against the weights its rerun ran with).
- `allow_changed_params=True`: replays with the current values and records the parameter names in
  `trace.last_run["replayed_with_changed_params"]`.
- Not detected: an in-place write through `param.data` or through a numpy or raw-storage alias.
- Replaced parameters: a replaced parameter object (`module.weight = nn.Parameter(t)`) is not used
  by replay at all, because the template keeps the captured object. For an as-if weight edit use
  `fork.do(tl.param(name), edit)`.
- The rerun engine runs the live model and ignores the option.

**differentiable** (`InterventionOptions`): `InterventionOptions(differentiable=True)` (default `False`) on `Trace.do` makes the replay's
outputs stay differentiable (experimental).

- Only the replay engine reads it; rerun and set-only ignore it. `ReplayOptions` carries no
  `differentiable` field.
- The replay lands on a new fork whose outputs stay differentiable (fresh frontier leaves), and
  `do` returns that fork instead of replaying the trace it was called on in place.
- Related: which edits keep the graph on which path is in docs/gradient_contract.md.

**torchlens.intervention.subspace_ablate(basis, *, mode="zero", dim=-1, mean=None, seed=None)**: Edit acting on the component of a site value inside the span of a `[k, d]` basis (experimental). The rows need
not be orthonormal.

- Modes, with `P` the orthogonal projector onto the span: `mode="zero"` gives `x - P x` (the
  k-dimensional `project_out`); `mode="mean"` gives `x - P x + P mean`; `mode="resample"` gives
  `x - P x + P x[perm]` for a seeded batch permutation (`seed=` required).
- Errors: a rank-deficient or non-finite basis raises `subspace_basis_invalid`; inconsistent
  mode arguments raise `subspace_mode_invalid`; a basis over the wrong width raises
  `subspace_basis_dim_mismatch` at fire time.
- Zero mode on the span of class-mean differences is the LEACE mean-equalizing erasure.

**torchlens.intervention.set_affine_coordinates(weight, *, bias=None)**: Source verb for a subspace edit: it sets the subspace coordinates to an affine function of the
whole site value (experimental). TorchLens applies a LoReFT edit trained elsewhere and never fits one.

- Effect: under `do(tl.subspace(site, R.T, origin=...), set_affine_coordinates(W, bias=b))` the
  new coordinates are `W x + b` of the whole site value `x`, so the site gets
  `x + Q^T (W x + b - Q x)` with `Q` the basis rows (LoReFT, `h + R (W h + b - R^T h)`).
- Arguments: `weight` is a finite real `[k, d]` tensor shaped like the basis rows; `bias` is
  `None` or a finite real `[k]` tensor. Anything else raises `affine_source_invalid`.
- Outside a subspace (a site label, `tl.units`, a capture or replay hook) it raises
  `affine_source_needs_subspace`.
- Saving: it is builtin and saves inside the `subspace_edit` payload, with `weight` and `bias` as
  tensor sidecars.

**torchlens.intervention.subspace_interchange(basis, source, *, dim=-1)**: k-dimensional basis interchange: `x - P x + P s`, with `s` the source trace's value at the same
site (same-site donor resolution as `tl.patch_from`; experimental). It applies a learned rotation the user
trained elsewhere.

**torchlens.intervention.keep_grad(edit)**: Returns the same forward edit (an `Edit` or a hook callable) with a gradient-connectivity check
added (experimental). `keep_grad(tl.scale(0.0))` is the graph-keeping zero;
`keep_grad(tl.intervention.compose(tl.scale(0.0), tl.add(value)))` is the graph-keeping fixed
value.

- Check: at fire time it raises `InvalidArgumentError` / `edit_detached_from_graph` when the edit
  writes values carrying no autograd graph, when the site value has none, or when gradients are
  disabled. The error carries the site label, a `reason` of `edit_detached`, `site_has_no_graph`
  or `grad_disabled`, and a suggested form that keeps the graph.
- Metadata: a wrapped `Edit` carries `("keep_grad", True)` in its metadata and
  `portability="opaque_audit"`.
- Errors: a backward edit or a non-edit raises `keep_grad_edit_invalid`.
- Differentiable replay: under
  `do(site, edit, intervention=InterventionOptions(differentiable=True))` a passing check warns
  once per session with code `keep_grad_replay_frontier`, because parameters upstream of the
  replay frontier get no gradient. The live path `tl.when(site, keep_grad(edit)).bind(model)(x)`
  reaches them.
- Related: which verbs keep the graph on which path is in docs/gradient_contract.md.

**torchlens.positions(edit, positions, *, mask=None, assume_no_padding=False, axis=1)**: Restricts a same-shape edit to chosen token positions of a `(batch, tokens, ...)` site value
(token-position editing; experimental).

- `positions`: an int, a list of ints, a slice, a per-row `(batch,)` index tensor, or `"first"` /
  `"last"`. The last two pick the first or last REAL token of each row, resolved from `mask=`
  under the token-pooling mask policy.
- Errors: no mask and no `assume_no_padding=True` raises `transform_mask_missing`; a mask that
  does not describe the value raises `transform_mask_incompatible`; unsupported forms raise
  `position_edit_invalid`; positions past the sequence raise `position_edit_out_of_range`.

**tl.intervention.fit_direction(pos_traces, neg_traces, site, method="mean_diff", *, read_token_index=-1)**: Builds one contrastive steering direction from saved activations and returns a `Direction` (experimental). Also
spelled `torchlens.intervention.fit_direction`.

- Inputs: `pos_traces` and `neg_traces` are a trace or a sequence of traces with one prompt per
  batch row; row `i` of the negatives pairs with row `i` of the positives.
- Read position: one token per prompt is read at `site`. `-1` is the last real token when the
  trace saved its `attention_mask` input; a sequence gives one position per prompt; `None` reads
  `[n, hidden]` outs unsliced.
- Methods, with `D = pos - neg`: `"mean_diff"` is `D.mean(0)` on the activation scale; `"pca"` is
  the first uncentered principal axis of `D`, unit norm, signed toward the positives; `"logistic"`
  is L2 logistic regression (`C = 1`, no intercept) of `D / 2` against `-D / 2`, solved with torch
  only, unit norm.
- Errors use the `direction_*` codes.

**Direction (`torchlens.intervention.Direction`)**: The frozen result of `fit_direction`: a steering vector plus a record of how it was fitted (experimental). It
steers natively and converts to three steering libraries.

- Fields: `vector` (1-D, the rows' dtype), `site`, `method`, `num_pairs`, `read_token_index`,
  `model_type` (the traced model's `config.model_type`, or `None`) and `provenance`.
- `provenance`: a `tl.types.BasisProvenance` whose origin records the site, method, pair count,
  read position and a sha256 digest of the rows, and whose `sha256` is the digest of the canonical
  float64 vector.
- Steering: `direction.steer(m)` is `tl.steer(vector, m, dim=-1)`; place it with
  `tl.when(tl.module(direction.site), ...)`.
- Converters: `direction.to_steering_vectors()`, `direction.to_repeng()` and
  `direction.to_dialz()` convert it to a real `steering_vectors.SteeringVector`,
  `repeng.ControlVector` and `dialz.SteeringVector` without recomputing it (thin converters
  `tl.bridge.steering_vectors.from_direction`, `tl.bridge.repeng.from_direction` and
  `tl.bridge.dialz.from_direction`). Applied with the same multiplier, each library object gives
  the native steer's logits.
- Layer key: the converters are keyed by `direction.library_layer(layer=None)`, the site label's
  trailing index unless `layer=` is given; a different explicit index raises
  `direction_layer_mismatch`.

**`Direction.upload_to_neuronpedia` (publish a vector)**: Publishes the vector on neuronpedia.org under the account that owns `api_key`, through
Neuronpedia's "Create Vector" endpoint, and returns the vector page's URL (experimental).

- Signature:
  `direction.upload_to_neuronpedia(*, api_key, model_id, name, layer=None, hook_type="resid_post", default_steering_strength=1.0)`
- Arguments: `model_id` is Neuronpedia's model id; `name` is the name it shows; `layer` defaults
  to `direction.library_layer()`; `hook_type` is `"resid_post"` or `"resid_pre"`;
  `default_steering_strength` is the default steering strength (-100 to 100).
- Every field is checked before the request (`neuronpedia_upload_argument_invalid`); the key never
  enters a message.

**Bundle**: A named collection of aligned Traces, constructed with `tl.Bundle(...)`, for cross-run comparison.
`tl.Bundle()` with no members builds an empty bundle to fill later with `add`; an empty bundle
saves and loads like any other.

**Bundle.relations**: The bundle's typed, directed relation rows between members, read as an immutable tuple of
`MemberRelationRow` (plus `OpaqueRelationRow` for unknown namespaced kinds loaded from an
artifact) (experimental). Rows record order and connection: members joined by a directed path of ordering
kinds cannot be compared with each other.

- Pair kinds: `alternative_of` (peers; still comparable), `forked_from`, `successor_of`,
  `escalates`, `fed_by`, `selected_by`. On a pair row `from` is always the LATER member.
- Member kind: `episode_member`.
- Params: each kind accepts a fixed set of params.
- Saving: rows are saved in the bundle file (`relations.json` in saved format 3), and every load
  re-checks them.

**Bundle.relation_grades**: `b.relation_grades()` returns one `RelationGrade` per relation row, in table order (experimental). Each grade is
computed when read, from the row's evidence items and declared params, and never stored.

- Ladder, weakest to strongest: `unchecked` (no check ran; the reasons are listed),
  `disclosed` (declared only: a bare row, a `carry_mode="none"` fresh start, items graded `disclosed`),
  `consistent` (every check that ran agrees, not proven), `verified` (every item verified).
- `divergent`: any divergent item grades the row `divergent`.
- `broken`: applies in exactly two cases, named in `broken_premises`: `bulk_origin_failure` (more
  than half of the checked member-origin items diverge) and `verbatim_carry_failure`
  (`carry_mode="verbatim"` with a divergent `declared_carry` item).
- Seams: a divergence at a declared seam (the item's `seam` is in the row's `seams`) is counted in
  `seam_local_divergences`, graded `divergent` and never raised to `broken`.
- `OpaqueRelationRow` grades `unchecked` with reason `kind_unregistered`.

**RelationGrade**: Frozen record of one row's derived grade (experimental). Import it with `from torchlens.bundle import RelationGrade`.

- Fields: `kind`, `members`, `grade`, `broken_premises`, `unchecked_reasons`,
  `seam_local_divergences`, `exogenous_items` (items whose class lies outside the member-measured
  kinds; derived, never written to an item), `items_by_class`, `items_by_grade`.

**successor_of.evidence item class**: The ground-truth origin of one evidence item, from a fixed list of seven kinds (experimental). It is stored under the evidence item key `class`. "Exogenous"
is a grouping read from the class, never a stored value.

- Member-measured: `member_input`, `member_output`, `member_state`, `member_params`.
- Exogenous: `declared_carry`, `declared_transform`, `harness_record`.
- Reserved schemas require it on every item: `version_boundary_v1` accepts the four member kinds
  plus `declared_transform`; `turn_boundary_v1` accepts `member_input`, `member_output`,
  `member_state`, `declared_carry`, `declared_transform`, `harness_record`.
- User schemas: optional, but checked against the seven on load when present
  (`bundle_relation_schema_invalid`).

**successor_of.carry_mode / seams / item seam**: Relation params that say what crosses a boundary and where the chain is expected to change (experimental).

- `carry_mode`: one of `none` (a declared fresh start: nothing carried across the
  boundary), `verbatim` (state carried unchanged) and `summarized` (state carried through a
  declared transform). Any other value raises an error on load.
- `seams`: a list of distinct names for the boundaries where the chain is EXPECTED to change (a
  chat-template revision, a tokenizer swap).
- Item `seam`: an evidence item may name one of the row's seams, and a divergence there never
  escalates the row to `broken`. An item `seam` that the row does not declare raises an error on load.

**Bundle.relate**: Appends relation rows and returns the bundle (experimental). If it raises an error, the table is unchanged.

- Positional form, read as the sentence it records:
  `b.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 3, 7]], label="critique")` is
  "critique fed_by draft". The arguments are the kind, the later member, the earlier member, then
  the kind's params; a member kind names one member.
- Row form: `b.relate(MemberRelationRow(...))` or payload mappings.
- Error codes: `bundle_relation_schema_invalid`, `bundle_relation_member_missing`,
  `bundle_relation_auto_spans_unavailable`, `bundle_relation_auto_spans_ambiguous`,
  `bundle_relation_auto_spans_unmatched`, `bundle_relation_span_out_of_bounds`,
  `bundle_relation_span_unverified`.

**fed_by**: "B fed_by A": the later member B's prompt contains the earlier member A's output (experimental). It
records content flow only: a router whose decision selects a call without its output entering that call's
prompt is not `fed_by`; record that decision with `selected_by`. `fed_by` is an ordering kind for
the comparison gate.

- Optional params: `token_spans` (rows of four ints, half-open: source start, source end,
  destination start, destination end), `basis` (`declared`, `verified`, `matched`), `coverage` (a
  fraction in [0, 1]) and `label` (free human text such as `"critique"` or `"judge"`; nothing in
  TorchLens interprets it).
- `token_spans="auto"`: TorchLens finds the spans. The source's eligible output (its bound
  completion range minus `excluded_ids`) is searched as exact runs of token ids in the reader's
  prompt, trimming up to two boundary tokens per side when a run is not found whole. The row is
  written with `basis="matched"` and a computed `coverage`.
- Auto-matching errors: several placements (a repeated quote, or a tie between two trims) raise
  `bundle_relation_auto_spans_ambiguous`; no placement raises
  `bundle_relation_auto_spans_unmatched`; missing evidence (no source completion range, ids not
  retained) raises `bundle_relation_auto_spans_unavailable`.

**selected_by**: "B selected_by A": the earlier member A's decision (a router's route, a judge's verdict) selected
the later member B, deciding whether B ran or which input it got (experimental). Control only: A's text need not
appear in B's prompt (text that does is `fed_by`), and the row never says A caused B's content.

- Example: `b.relate("selected_by", "writer", "router", chosen=1, candidates=[0, 1])` stores
  `from="writer"`, `to="router"`, like every ordering kind.
- Optional params, and no others: `chosen`, `candidates` and `evidence`. Malformed params raise
  `bundle_relation_schema_invalid`.
- Ordering: `upstream`/`downstream` follow it, and the comparison gate will not compare the
  selected call with its decider.
- Saved files accept the kind from tlspec format 10 on; an older file that carries it raises an
  error on load.

**selected_by.chosen**: The decision, as one JSON scalar: a bool, an int, a finite float or a non-empty string of at most
256 characters (route `1`, candidate `"c2"`) (experimental). Compared by type and value, so `1` and `True` are
different decisions. When `candidates` is also given, `chosen` must be one of them.

**selected_by.candidates**: The options the decider chose among: a non-empty list of distinct JSON scalars (same scalar rule
as `chosen`) (experimental). They may be member names, routes or tool names; TorchLens never resolves them as
members.

**selected_by.evidence**: Why the decider chose, as the same graded evidence structure `successor_of` accepts
(`{schema, items, facts_digest}`; each item a graded claim with a nullable `basis`), for example
the router's scores (experimental).

- User-written measurement grades (`verified`, `consistent`, `divergent`) are stored as
  `disclosed`, with the original under `declared_grade`.
- The evidence is limited to 1 MiB per row (`bundle_relation_evidence_over_budget`).

**Token-stream binding**: What makes `fed_by` spans checkable: a per-member descriptor in the reserved
`"torchlens.boundary_facts"` sidecar family that names the captured token stream the span
coordinates index (experimental). Its schema name is `token_boundary_v1`. Attach one with `torchlens.io.attach_sidecar` before relating; with none,
`relate` derives an input-stream binding for a batch-of-one capture whose ids are retained.

- Fields: the stream (`"input"`, the first integer-dtype input; or `"output"`), the `batch_row`,
  the measured `length` and `ids_digest` of its token ids (filled in by `relate` when left out),
  the member's own `completion` range, the `excluded_ids` outside the eligible set (padding,
  special tokens), and `tokenizer` / `template` identity strings.
- Verified basis: a row reads `basis="verified"` only when both members are bound, each digest
  matches the retained ids, the source binding names its completion range, every source span lies
  inside it, and the token ids agree span by span. Anything less is `declared`.
- Re-checks: a claimed `verified`/`matched` row is re-checked every time (relate, constructor,
  load). Missing evidence makes it `declared`; contradicting evidence raises
  `bundle_relation_span_unverified`.
- Matched rows: a `matched` row also re-runs the matcher on the members and keeps `matched` only
  when it finds the same spans; spans that check out but are not what the matcher finds are stored
  as `verified`.
- Coverage: `coverage` on a verified or matched row is computed as covered eligible source tokens
  over all eligible ones (the bound completion range minus `excluded_ids`).

**Bundle.upstream / Bundle.downstream**: `b.upstream(name)` lists every member `name` transitively came after (what fed it), and
`b.downstream(name)` every member that came after it, nearest first, over the ordering kinds
(`fed_by`, `selected_by`, `successor_of`, `forked_from`, `escalates`) (experimental). Bundle methods, like
`compare`.

**Bundle.token_pairs**: `b.token_pairs(reader, source)` returns the verified one-to-one token positions of "reader fed_by
source", arguments in `relate`'s order (the later member, then the earlier one) (experimental). Equal lengths are
never evidence of matching meaning. A Bundle method; any reduction over a many-to-one group is the
caller's own descriptive comparison.

- Result: `pairs` are `(source_pos, reader_pos)` tuples, each position at most once, with
  `source_positions` / `reader_positions` lists for indexing activations. `coverage` is the paired
  share of the source's eligible tokens (`None` without a bound completion range). `excluded`
  lists what was not paired, each entry a dict with `side`, `start`, `end`, `reason` and
  `row_index`.
- What pairs: only spans of `verified` or `matched` rows that re-check at call time.
- Reported, never paired: declared spans (retokenized many-to-many text, a token mixing a template
  delimiter with copied text), excluded ids, positions claimed twice, streams whose bindings name
  different tokenizers, and eligible source tokens no verified span covers.
- Error: a verified row whose evidence now contradicts it raises
  `bundle_relation_span_unverified`.

**Bundle.token_divergence**: `b.token_divergence(original, rerun)` returns a `TokenDivergence`: where two members' prompts and
completions part, by exact token ids (experimental). A Bundle method.

- Arguments: `original=` is the member from the original run (the reference); `rerun=` is the
  member from the re-run compared against it. Typical use is a call and its copy from a steered
  re-run (`b.token_divergence("critique", "cf_critique")`), but any two bound members compare.
- Requirements: each member needs a token-stream binding that names its `completion` range, with
  its token ids retained; the prompt is the bound stream before the completion.
- Errors: `bundle_token_divergence_unavailable` (fields `member`, `reason`) when a binding, its
  completion range or the ids are missing; an unknown name raises `KeyError`.

**TokenDivergence**: The frozen result of `Bundle.token_divergence` (experimental). Import it with `from torchlens.bundle import TokenDivergence`.

- `original`, `rerun`: the member names.
- `prompt_divergence`: the first prompt position whose ids differ (the shorter prompt's length
  when one extends the other; `None` when equal).
- `completion_divergence`: the same for the completions, counted from the first generated token.
- `lengths`: `((prompt, completion), (prompt, completion))`.
- `shared`: the stretches both runs share as `(orig_start, orig_end, rerun_start, rerun_end)`
  half-open stream positions (the `token_spans` grammar): the common prompt prefix, the common
  prompt tail after a divergence, the common completion prefix.
- `intervened`: whether each capture ran under an intervention.
- Properties: `prompt_same` and `completion_same`.

**Bundle.add_alternative**: `b.add_alternative(second_run, steered=("draft",), prefix="cf_")` merges a second run of your own
harness, made with the `steered` calls captured under their specs, into `b` and returns an
`AlternativeReport` (experimental). Calls pair by member name, never by order. TorchLens never runs your harness.
A Bundle method.
Before changing anything it raises one of these codes:

- `alternative_target_missing`: a steered call is absent from either run (field `side`).
- `alternative_target_unsteered`: a declared call was captured without a spec that fired.
- `alternative_steer_undeclared`: another second-run call was captured under a spec.
- `alternative_capture_position_dependent`: the spec edits chosen positions or decode steps
  (`at_positions`, `at_step`).
- `alternative_member_name_taken`: a copy would reuse a name; pass another `prefix`.
- `bundle_token_divergence_unavailable`: a paired call lacks a bound completion range.
- `alternative_unchanged_call_diverged`: an unsteered call saw the same prompt but wrote different
  text, which is noise TorchLens cannot separate from the steer.
- Merge: every second-run member is added as `prefix + name`, each paired call gets an
  `alternative_of` row from its copy, the second run's own relation rows are copied onto the
  copies, and one `add` operation records the statuses. A failure while merging restores the
  bundle.

**AlternativeReport**: The frozen result of `Bundle.add_alternative` (experimental). Import it with `from torchlens.bundle import AlternativeReport`.

- Fields: `prefix`, `steered` (the declared calls), `calls` (one `AlternativeCall` per call of
  either run, the original's order first), `uncopied_rows` (the second run's rows of unknown
  namespaced kinds, not copied).
- Methods: `table()`, a fixed-width text table with one line per call.

**AlternativeCall**: One call of an `AlternativeReport` (experimental). Import it with `from torchlens.bundle import AlternativeCall`.

- Fields: `call` (the original member name), `member` (its copy, `None` when dropped), `status`,
  and `prompt_divergence` / `completion_divergence` as on `TokenDivergence` (`None` for unpaired
  calls).
- Statuses: `steered` (declared in `steered=`), `unchanged` (prompt and output the same),
  `absorbed` (the prompt changed, the output did not; the common case), `changed` (both changed),
  `added` (only the second run made the call) and `dropped` (only the original made it).

**Bundle.at**: `b.at(site)` returns the cross-member `Super*` view of one site.

- Sites: a string is a label from any Bundle accessor family. Any other selector-like site
  (`tl.func("relu")`, `tl.module(address)`) resolves to one op in every member after the
  model-axis floor (`same_param_shapes`) passes for every member pair (a view read, so different
  inputs are allowed).
- Comparing: `b.at(site).compare()` is the member-by-member matrix at that site and first runs the
  comparison gate over the members it reads.
- Subsets: to read a subset of members, build a bundle of them,
  `tl.Bundle({name: b[name] for name in names}).at(site)`, so one foreign member does not disable
  the view for the rest.

**Bundle.draw**: Draws the bundle as one picture and returns a `tl.viz.Drawing` (`str(drawing)` is the DOT source,
`drawing.path` the written file). It follows `Trace.draw`'s rules.

- Signature:
  `b.draw(path=None, *, graph="forward", nodes="ops", rolled=False, direction=None, theme=None, show_legend=None, file_format=None, open_viewer=False, graph_overrides=None, node_overrides=None, edge_overrides=None, members=None, metric="relative_l2", on="out", depth=None)`
- Path and format: the path comes first and its extension picks the format (`b.draw("diff.svg")`
  writes SVG). `file_format=` applies only to a path with no extension (`<path>.<file_format>`,
  `"pdf"` by default); a `file_format` that disagrees with the extension raises
  `ArgumentConflictError` (`draw_format_mismatch`).
- Bare call: outside a notebook a bare `b.draw()` writes a collision-safe
  `bundle_modelgraph.<format>` and prints its path; in a Jupyter kernel it writes no file and
  returns the drawing for inline display. No viewer opens unless `open_viewer=True`.
- Passes: `graph="forward"` (default), `"backward"` (each member's captured grad_fn graph) or
  `"combined"` draw the bundle's union op graph for that pass, with a member list on each node and
  arrow that not every member runs.
- Layout: `rolled=True` draws each repeated module once; `direction=None` is `"bottomup"`; `theme`
  takes the `Trace.draw` skin names (`None` is `"torchlens"`).
- Overrides: `node_overrides=` (Bundle-only) keys Graphviz node attributes by op label (member
  name under `nodes="members"`), `edge_overrides=` by `(source, target)`.
- Node kinds (the `nodes=` names may change): `nodes="ops"` (default) draws the ops. `nodes="members"` draws one node per member
  and one edge per relation row, from the earlier member to the later one, labelled with the kind,
  the row's `label`, on `fed_by` its basis and coverage, and on `selected_by` its `chosen`
  decision. Solid edges carry token spans TorchLens checked, every other row is dashed, and
  `graph` must stay `"forward"`.
- Diff: `graph="diff"` draws two members side by side, one node per op, coloured by `metric`
  distance on the `on` field. Each pass of a layer that runs more than once is its own node with
  its own distance (the per-pass rows of `Bundle.compare`). `members=(left, right)` names the two
  (default the baseline and the next member). Its defaults are `theme="paper"`, `"svg"` and a
  `bundle_diff` file, and it runs the comparison gate on the rendered pair only. `direction=`
  turns the picture (`None` keeps the two members as side-by-side columns), the three
  `*_overrides` apply as on the union graph (nodes and edges keyed by op label, in both
  members), and `show_legend=False` drops the colour key; `rolled=True` and `nodes="members"`
  have no diff meaning and refuse with `ArgumentTypeError` (`bundle_draw_keyword_unsupported`)
  naming the keyword.
- Folded diff: `graph="diff", depth=N` draws the compared member (the second of `members`; the
  first is the baseline) as its own model graph, exactly as `Trace.draw(depth=N)` folds it, with
  every op and every folded module box filled by its `metric` change on `on` from the baseline. A
  box takes its most changed op (the fold `rank_changes(by="module")` uses), and every op and box
  the drawing shows gets its value (no top-k cap). Edit marks stay on the edited op or box. Path,
  format, theme (`"torchlens"`), legend and `direction` follow `Trace.draw`; `graph_overrides`
  pass through, while the op-keyed `node_overrides`/`edge_overrides` and `rolled=True` refuse with
  `depth`. `depth` with `graph="forward"`, `"backward"` or `"combined"` refuses (the union graph
  has no module folding). Without `depth` the diff is the two-column picture above
  (`runs.draw(graph="diff", depth=2)` on GPT-2 draws its 12 blocks).
- Unsupported keywords: any other keyword (a `Trace.draw` keyword the bundle renderer does not
  support yet, such as `collapse=`) raises `ArgumentTypeError`
  (`bundle_draw_keyword_unsupported`) naming the supported list; each is added when the renderer
  supports it.
- Replaces the removed `show_bundle_graph` and its `signal=` keyword (now `graph=`; no alias).

**Bundle.show**: `bundle.show(kind=None, **kwargs)` is the single way to display a bundle.

- No kind (or `kind="graph"`): it is `bundle.draw(**kwargs)` and returns that `Drawing`, the
  merged graph of where the members differ.
- `kind="strip"`: draws each member's own graph with `Trace.draw(**kwargs)` and returns a dict of
  member name to `Drawing` (a `path` gets `_<member>` appended per member).
- `kind="repr"` and `kind="html"`: return each member's `repr` / HTML text by member name.
- Any other kind raises `InvalidArgumentError` (`bundle_show_kind_invalid`).

**Bundle.compare**: `bundle.compare(a, b, *, metric="cosine", on="out")` returns the per-site differences between two
members as `(site_label, distance)` rows over the sites both members share, paired through the
structural alignment.

- Arguments: `metric` is a metric name or callable (scalar-like sites always use `relative_l1`);
  `on` reads `"out"` or `"grad"`.
- Axis: the receiver sets it. `bundle.at(site).compare()` is the member-by-member matrix at one
  site.
- Gate: the pair goes through the comparison gate first and raises a typed error when the members cannot
  be aligned; its grade is `bundle.comparability(a, b)`, a `tl.types.Relationship`.

**Bundle.store_comparison / Bundle.stored_comparison / Bundle.stored_comparison_names**: Computes the per-node distances of every member from the baseline once and stamps them on the
supergraph under a name, so later reads do not recompute. Held in memory on the built
supergraph, never saved.

- `b.store_comparison(metric="relative_l2", *, name=None, baseline=None, on="out")`: computes and
  stores; the baseline is the configured baseline, else the first member. It returns the stored
  name: the metric's name unless `name` is given (a callable metric needs an explicit name).
- `b.stored_comparison(name)`: reads one back without recomputing (`comparison_unknown` for a name
  never stored).
- `b.stored_comparison_names()`: lists the stored names.
- Repeated layers: a layer that runs more than once has one value per member, the metric over its
  passes' tensors joined in pass order.
- Gate: every (baseline, member) pair goes through the comparison gate first, with the same
  predicates as `bundle.at(site).compare()`: no ordering path between them, a shared graph, and
  sites that line up (proven-equal input values, or for runs on different inputs or checkpoints,
  equal graph-shape hashes). Otherwise the read raises the gate's `bundle_gate_*` code.

**Bundle.episode_status**: `b.episode_status(episode_id, *, episode_ledger=None)` folds the outcomes of the members named by
`episode_member` rows into a derived episode status (`EpisodeFoldResult`) (experimental). It is computed on demand,
never stored as a Bundle outcome; without an episode record, statuses that need one fall back to
`episode_unknown`.

**Bundle.why / Bundle.provenance**: `b.why(member, relative_to=None)` returns a `WhyReport`: the exact ancestry of a member (or an error when it cannot be exact)
against the baseline (or `relative_to`), with lineage status, payload fidelity, value residual and
comparability reported as separate axes (experimental). `b.provenance.summary()` is the per-member list of the
same facts. Both are computed on demand; nothing is stored. Reference: `docs/reference/experiments.md`.

**Bundle.vary**: `b.vary(mapping, *, unmentioned=None)` applies one explicit intervention spec (or `None`, no
change) to each named member (experimental). It raises an error before any member changes if the
mapping does not cover every member; a failure partway through raises `vary_partial_failure` and
reports what completed (nothing is rolled back). Broadcast `do()` is unchanged.

**Parameter value basis**: `Param.value_basis` says where a parameter value comes from (experimental). It is computed when
read and never saved. It sits beside the live read and never changes it: `Param.value` keeps its
documented live-handle meaning.

- Values: `live_ref` (the live-model handle resolves; the value may have moved since capture) or
  `absent(not_persisted)` (loaded from disk; parameter bytes were never saved). `snapshot` arrives
  only with capture-time parameter snapshots.

**Checkpoint live-ref guard**: TorchLens records which parameter a run used, not its bytes, so cross-member parameter value,
difference and trajectory reads on a Bundle raise an error unless every member carries immutable
capture-time parameter evidence (experimental).

- Guarded reads: `weight_norm_diff`, `compare`, `aggregate`, and `out`/`grad` on `SuperParam`
  views with two or more members.
- Error: the stable code `checkpoint_series_live_params`, raised before any tensor is looked up.
- The guard is keyed on the claim, never on Python object identity, and survives save/load; a
  version-axis relation row still orders members.

**Runnable artifact**: A `.tlspec` saved at `level="runnable"` with a sparse taken-path descriptor. No callable is ever
pickled.

- Source capture: it must use `capture=tl.options.CaptureOptions(replayable=True)`. The flag
  records what a runnable save needs (callable registry keys, state snapshots, alias topology, escape
  records), which no default capture keeps. The per-op argument templates that
  `trace.do(site, edit)` replays are recorded on every capture.
- On disk (saved format 3): the descriptor is `runnable/descriptor.json` and the state is
  safetensors files beside it, listed by the manifest's `runnable` object.
- Maintainer reference: `docs/maintainers/saved_format_3.md`, section 7.

**Readiness**: The non-executing runnable state `ready` or `unavailable`.

**Path faithfulness**: A runnable result verdict: `verified`, `diverged`, or `unverifiable`. `unverifiable` means the run
completed without enough evidence for the stronger claim; it is not a pass.

**Numeric attestation**: Byte-level comparison of eligible archived selected activations. Its status is `attested`,
`numeric_attestation_failed`, `not_applicable`, or `not_present`; attestation never upgrades path
faithfulness.

## Capture accuracy and visualization

**Capture verification reason**: The machine-readable `Trace.provenance.verification_reason` explaining why a live capture is not
positively verified, for example `mode_rescue_rerun`, `escape_rescue_unrecovered`,
`owner_thread_tripwire_changed`, or `dynamo_region_not_logged`.

**Pristine (`Trace.is_pristine`)**: Whether nothing ever intervened on the trace: `True`, `False`, or `None` (experimental). It is a reading taken
from the evidence the trace carries, not a verdict: it never changes validation or a run report.

- Evidence: derived at read time from records saved with the trace (the intervention audit, the
  operation history, mutation states, direct writes and op intervention marks), never stored as
  a flag. It survives `Trace.save`/`tl.load`, pickle and `fork()`, and no reset makes a touched
  trace read `True` again.
- `False`: some touch was recorded: a capture-time `intervene=` (including one that fired
  nowhere), a staged hook or value replacement (even after `clear_edits(site)` removes it),
  `do()`, `set()`, a region edit, a hooked rerun or run, or a direct write. Plain reruns, plain
  runs, forks and `clear_edits(site)` on an untouched trace are not touches.
- `None`: no evidence, because the artifact was saved by an older format that does not record
  it, or the backend is not torch. It is never a pristine reading.
- Bundle's pristine label and baseline auto-detection count a member only when it reads `True`.
- Clean runs: a hooked run settles `unverifiable` and marks the trace non-pristine. The
  maintainer page `runnable_model.md` relates this reading to the runnable clean-run promise.
- Not `Trace.state`: `TraceState.PRISTINE` is an operational state that an assignment can set
  and that a func-targeted `intervene=` capture keeps.
- Not validation's pristine-torch oracle, which means running on unwrapped torch functions.

**Rescue re-run**: A single, reported second forward pass under a `TorchFunctionMode` that redirects stale torch
references (taken before wrapping) to their wrappers. The capture-accuracy facts record `rescue_rerun` and reason
`mode_rescue_rerun`.

**Collapse**: A rendering-only condensation of module detail: `collapse="none"`, `"auto"`, `"max"`, or a
float in `[0, 1]`. It never changes the underlying Trace.

- Budget: the size of the drawn graph (U) is checked against the safety limit
  `COLLAPSE_OPTIMIZER_MAX_OPS` (2000) plus a measured estimate of the work (U, W).
- Over budget: the request degrades to a deterministic compact fallback plan (coded
  `collapse_budget_fallback`), never an uncollapsed wall.
- Very large inputs (raw ops above 20x the limit) skip collapsing entirely
  (`collapse_pathological_skip`): `Drawing.collapse_plan` raises a typed error
  (`collapse_plan_unavailable`) and `Drawing.collapse_schedule` degrades to its single
  full-graph step.

**Module.collapse_score / Trace.module_collapse_order**: A module's smart-collapse priority, used to rank collapse candidates (experimental). Computed when read, never
saved.

- Value: the module's hidden ops (distinct subtree ops minus one) over the largest submodule's,
  rounded to 6 places.
- 0.0 for the root, for a module of at most one Layer, and for the only child of a non-root
  module that hides the same ops: the parent, the name the user wrote, ranks instead (in a chain
  of such wrappers the outermost ranks).
- It ranks candidates and never reads the selected plan, so `Drawing.collapse_plan` may skip a
  high-priority module.
- `module_collapse_order` lists every module's priority, highest first.

**Run folding**: `fold_repeats=True` replaces eligible runs of distinct same-class sibling modules with a
representative and a `+N more` label. It differs from recurrence, where `(xN)` means the
same parameters executed repeatedly.

**Receptive field / Projective field**: Influence geometry through the captured DAG: toward inputs (receptive) or toward outputs
(projective).

- Per op: `op.receptive_field` and `op.projective_field`. Per trace: the
  `Trace.receptive_fields()` and `Trace.projective_fields()` tables.
- `.at(unit)` on a field view answers for one unit.
- `.of(units, within=None, method="geometric", probe_budget=256)` answers for many units (the
  units `tl.units` takes: a bool mask over the owner's full output, or full-shape coordinate
  tuples). It returns a trace-bound `tl.Selection` with one region per `within=` site
  (default: model outputs for projective, model inputs for receptive), usable in `trace.do`,
  `trace.subgraph` and the `& | - ~` set operations (experimental).

**Encoding channel**: A declarative value source (record field name, scalar builtin, or callable `node -> value`)
mapped to a visual channel on `draw()` (experimental). The channels are color (`color_by`), size (`size_by`) and rank (`stack_by`).

- Channels are presentation-only: they never change the Trace or the collapse plan.
- They require the Graphviz dot layout: `layout="auto"` forces dot when a channel is active,
  and explicit `layout="rank"` raises `encoding_requires_dot_layout`.

**color_by**: `draw(color_by=...)` fills eligible operation nodes from a colorblind-safe sequential ramp,
normalized linear min-max over the visible nodes (experimental).

- The drawn key is a colour bar labelled with the quantity and its units at both ends, and
  nothing else (no transform, coverage or notes).
- The ramp is a stop list blended in OKLab at even perceptual speed (default viridis, dark =
  high on a light page).
- Missing, non-finite, or rolled-ambiguous values leave nodes unencoded.
- Rolled multi-pass layers: field sources are looked up in a fixed, name-keyed list of rolled
  aggregates. Per-pass-varying and first-pass-only sources are never painted as if uniform;
  exact cross-pass totals encode (the aggregation is kept on the encoding state, not drawn);
  sources not on the list raise `encoding_source_invalid`.
- Collapsed draws: module boxes and segments take the exact sum of their member ops for
  additive sources (`"time"`, `"bytes"`, `"flops"`, `"device_time"`, additive fields), with an
  aggregation line on the encoding state; other sources leave them unencoded with a note on
  the encoding state.
- Error codes: `encoding_value_invalid` for wrong-typed values; `encoding_callable_error`
  for a raising callable, with the original exception chained.

**size_by**: `draw(size_by=...)` sizes eligible operation nodes from a value source (experimental). Strictly opt-in: plain
`draw()` keeps uniform boxes.

- Sources: a scalar field, the `color_by` builtin tokens (`"time"`, `"bytes"`, `"flops"`, ...
  with the same meaning and units), the `"dims"` shape token (numel of the non-batch
  output shape), or a callable `node -> scalar`.
- Collapsed boxes and segments size from the same exact member-op sums as `color_by`.
- Sizes: the smallest value draws at the drawing's typical node size (the median size the
  nodes' labels need, widened for ellipses and other non-box shapes) and the largest at 2x
  that area. A node whose text needs more room draws at what its text needs, so one long
  label never inflates the other nodes.
- Emitted sizes are width/height minimums under `fixedsize=false`, so an encoding can never
  truncate a label and fonts never scale.
- Rolled multi-pass layers: size raises an error where color would leave the node unencoded
  (`size_by_rolled_varying`), because a size source that cannot be shown to have one value
  (marker-varying, first-pass projection, varying shape under `"dims"`) has no accurate "n/a"
  rendering. Exact cross-pass totals
  (`total_*`) encode with a mandatory aggregation line on the encoding state; callables bypass
  the rolled table with a note on the encoding state.
- The drawn key is one line, for example `size: output bytes, 15 KB ... 60 KB`.

**size_scale (size channel)**: The size-channel scale transform: `"sqrt"` (default; compresses dynamic range) or `"linear"`
(area in direct proportion to the value) (experimental). Log is rejected by design (it flattens 512-vs-4096).

- Error codes: `scale_requires_size_by` when supplied without `size_by`;
  `encoding_scale_invalid` for an unknown token.

**stack_by**: `draw(stack_by=...)` pins nodes sharing an annotation value to one Graphviz rank, as in the
classic unrolled-RNN timestep diagram (experimental). Strictly opt-in.

- `True`/`"auto"` derives the annotation (`pass_number` on multi-pass ops only) only
  when the pass_number sequence over all multi-pass ops in
  raw execution order is globally non-decreasing. Then "same column = same execution window"
  is exactly what the figure claims.
- An explicit field or callable skips that check, and the caption says what was used.
- Error codes: `stack_by_auto_underivable` for non-monotone traces;
  `stack_by_requires_unrolled` for rolled graphs.
- Rendering: rank groups are carried in `RenderIR.stack_rank_groups` and emit as `rank=same` subgraphs
  under `newrank=true`; the sibling-ordering post-pass does nothing while stacking is active.

**show_redundant_args**: By default `draw()` hides constructor-arg label rows that only repeat a captured shape;
`draw(show_redundant_args=True)` turns that off and shows every captured arg (experimental).

- The check: `draw()` omits a module constructor arg only when the arg value provably equals
  the captured shape dimension it claims to duplicate on this trace. Candidates come from a
  fixed table of torch nn module families; kernel_size/stride/padding/groups/num_embeddings/
  num_heads are never candidates.
- A mismatch or unavailable shape keeps the arg visible: the rule can only reveal more, never
  hide a discrepancy.
- Rolled varying aggregates keep args visible while their unrolled per-pass nodes suppress
  (on purpose); detached records render all args.
- Reference: `docs/reference/encoding.md`.

**show_legend tri-state**: `show_legend` accepts three values. On `Trace.draw` the key sits in the top-left block under
the header and above the graph, so a top crop keeps both.

- `None` (default, AUTO): no legend unless an encoding channel is active, then the channel key
  alone; a `module=` focused draw also gets the key. The `None` behavior is
  experimental.
- `True` (the key): one compact row per drawn mark, each drawn with its real shape and fill
  (input, output, trainable and frozen weights, module frame, module run as one op, collapsed
  module, hidden buffers, buffer, true/false test, and mutated parameter or intervention only
  when the picture has one), plus the channel key when active.
- `False`: no legend, honored even with channels active (the key is then not drawn).

**show_containers / container_max_inline (Trace.draw kwargs)**: `show_containers` sets how `Trace.draw` shows container structure; `container_max_inline` sets
when a container folds. Any string setting of `show_containers` forces the `dot` engine.
Guide: `docs/containers.md`.

- `False` (default): the plain graph, with container paths on the node labels.
- `"labels"`: the key or index on each edge into a container leaf.
- `"collapsed"`: `"labels"` plus: an unrolled container with more than `container_max_inline`
  leaves of identical shape folds into one dashed summary box.
- `"nodes"`: `"collapsed"` plus one dashed record node per container, grey keyed source and
  sink edges, dashed member-of ties for mid-graph containers, and a dotted box around an
  op-returned tuple whose leaves share one module.
- `container_max_inline` (default 12) is read only under `"collapsed"` and `"nodes"`.
- Removed values: the former `"auto"` (drew `"collapsed"`) and `"cluster"` (drew `"labels"`
  plus the tuple box) raise `visualization_show_containers_invalid` with the replacement
  named.

## Backend-neutral identity

**Backend**: `Trace.provenance.backend`, such as `torch`, `jax`, `mlx`, `tinygrad`, `paddle`, or `tf`.

**Module identity mode**: `Trace.options.module_identity_mode`, describing how the backend identifies module-like ownership.

**Backend address / resolver status**: Portable origin and resolution fields for backend records. They travel with `param_source`,
`dtype_ref`, and `device_ref`; none should be inferred from a torch-only object when a
backend-neutral field exists.

## Site selectors

**Label selectors**: `tl.label` (exact final label), `tl.label_contains` (label substring), and `tl.label_matches` (label
regex pattern) select ops by their public labels. `tl.where` filters a table with a
predicate callable.

**Pass tokens (`label:N`, `label:last`, `label:all`)**: Spellings that name one pass, the final pass, or every pass of a recurrent layer. They work
wherever a label is accepted.

- `label:N`: passes are 1-based, so `relu_1_1:2` is the second pass of a recurrent layer. There
  are no negative pass indices.
- `label:last`: `relu_1_1:last` names the final pass. `trace["relu_1_1:last"]` and
  `trace.ops["relu_1_1:last"]` return that Op; `trace.layers["relu_1_1:last"]` is the
  aggregate Layer like any pass spelling.
- `label:all`: `relu_1_1:all` names every pass. `trace["relu_1_1:all"]` and
  `trace[tl.label("relu_1_1:all")]` return the Selection of every pass; an Op collection
  raises `:all` with a `KeyError` naming those entry points.
- Bare labels: a bare layer label at `trace[...]` is the Layer, single- or multi-pass. A
  single-pass Layer passes single-op reads (`.out`, `.shape`) through to its Op; on a
  multi-pass Layer they refuse (`layer_pass_ambiguous`) naming `label:1` and `label:last`.
- A entry point that must give one Op refuses a bare multi-pass label: `tl.label(label)` with
  `multipass_bare_label_ambiguous` naming these spellings, and `trace[label]`,
  `trace.ops[label]` with `ambiguous_op_lookup`.

**Indexing coherence (every entry point returns the kind it names)**: The plain entry point `trace[key]` returns the kind the key names. A typed entry point always returns its own
kind and converts a key that names exactly one such object.

- Plain entry point: a layer label gives its `Layer` and `label:N` its Op (*Pass tokens*, above); a
  module address gives its `Module` whether called once or many times, and `address:N` its
  `ModuleCall`; a parameter address gives its `Param`, a buffer address its `Buffer`
  (`buffer_address:N` the version Op), and a grad_fn label its `GradFn`.
- Selections at the plain entry point: a selector gives the `tl.Selection` of its matches, and an
  already-resolved `Selection` gives itself on its own trace. On a fork descended from that trace
  it gives a copy bound to the fork, matched by site; one from an unrelated trace raises
  `selection_trace_mismatch` (bridge with `resolved.align_to(trace)`). `do()` and the selection
  operands of lazy producers (the `within=` population of the value producers, `random_like`'s
  `like` and `within`, `tl.neighborhood` and `tl.between` regions) bind a resolved selection the
  same way, so `t.fork().do(tl.top_fraction(t[sel], 0.1), edit)` works.
- Typed entry points: `trace.ops[address]` and `trace.ops["address:N"]` give the call's output Op,
  `trace.layers[address]` its Layer, `trace.modules[op_label]` (also a parameter or buffer
  address) the Module it belongs to, `trace.module_calls[op_label]` the call that ran it,
  `trace.params[address_or_op_label]` and `trace.buffers[...]` the single record a module or
  op owns, and `trace.autograd.grad_fns[op_label]` the op's grad_fn.
- Ambiguity: a key that names several objects of the entry point's kind raises
  `ambiguous_op_lookup` listing the `:N` spellings (a module called twice at `trace.ops`, a
  module with weight and bias at `trace.params`, a substring matching several grad_fns).
- Inside a module the same rule holds: `module[address]` is a descendant Module,
  `module["address:N"]` its call, `module[label]` what `trace[label]` gives, and
  `module.calls[op_label]` the call that ran it. It also holds on `layer.ops["address:N"]` and
  on every Bundle entry point, which converts through each member's typed entry point and propagates its
  refusal.
- Filtered collections (`trace.params.frozen()`) convert only to their own members.
- Wrapper-relative module spellings (*Module and stage names*) are plain-entry point and selector
  affordances.

**tl.all_sites** (experimental): `tl.all_sites()` is the universal selector: `trace[tl.all_sites()]` selects every op of the
trace as a whole site.

- It gives the same result as `trace[tl.label_matches(".*")]`: it stores that kind and
  pattern, so it matches, binds and persists the same way, and a saved spec reloads as
  `tl.label_matches(".*")`.
- It composes with `|`, `&`, `-` and `~` like any selector. `tl.all_sites() - region` is the
  complement of a region in the whole trace; the graph producers and `trace.subgraph` read it
  as the sites outside `region`, because a site whose element mask is empty seeds nothing.
- Resolution raises a typed error (`non_tensor_site` / `no_index_space`) when some op has no output
  index space, rather than dropping it.

**Structural selectors**: Selectors that pick sites by callable, module context, attention head, facet, transform
boundary, or model input and output.

- `tl.func` / `tl.module` / `tl.in_module(address)` select by callable or module context.
  `in_module` only builds a selector; whether one record lies in a module is read on the
  record, `layer.module == address` or `address in layer.modules`.
- Aliased modules: a module registered under several names (`self.dec = self.enc`, all
  recorded in `Module.all_addresses`) is selected by its canonical `named_modules()` name,
  which fires at every call site of it (pass labels such as `"enc:2"` count calls of that one
  object). An alias spelling (`"dec"`, `"dec:2"`) refuses with `bind_static_anchor_unresolved`
  and one message at every entry point (capture, record, post hoc, `spec.bind`), since no hook can
  tell which name a forward called the object through.
- `tl.head` selects one attention head; `tl.facet` selects a semantic facet view.
- Unscoped heads: `tl.head(i)` writes head `i` of every matching facet in every module and
  says how many sites it writes (a `TorchLensWarning` coded `facet_selector_fanout`, with
  `fields["site_count"]`; 36 on a 12-layer GPT-2: q, k and v in each layer).
- Scoping: `.in_module(address)` scopes it to one module, and so does composition:
  `tl.module(address) & tl.head(i)` (either order) is that same scoped facet selector,
  accepted wherever a selector is (`trace[sel]`, `tl.when`, `do`). A scoped head is one
  module's head, so `do` writes its `q`, `k` and `v` (one write each) silently; the fan-out
  warning is the unscoped broadcast's. A facet already scoped to
  another module raises `selector_module_facet_scope_conflict`.
- Facet unions: in `do`, a `|` union of facet selectors
  (`tl.module(a) & tl.head(i) | tl.module(b) & tl.head(j)`, or
  `(tl.module(a) | tl.module(b)) & tl.head(i)`) edits every branch in one call, exactly as
  chained `do` calls would. `find_sites` and any other composite holding a facet raise
  `facet_selector_unsupported`.
- `tl.func_transform` selects `torch.func` and `torch.autograd` transform boundary ops,
  including the five kinds in the next bullets.
- `autogradgrad`: an in-forward `torch.autograd.grad` call, with
  `transform_kind="autograd.grad"`.
- `autogradbackward`: an in-forward `loss.backward()` or `torch.autograd.backward` call, with
  `transform_kind="autograd.backward"`. There is one op per leaf gradient the pass writes,
  parented on the roots and, when the leaf already held a grad, on that previous grad as its
  in-place receiver, so a later `param.grad` read has a producer.
- `autogradfunction`: a custom `autograd.Function` whose forward left torch through a numpy or
  dlpack round trip or a known foreign kernel (FlashAttention 4's `flash_attn.cute`
  Functions, exit `flash_attn.cute`), with `transform_kind="autograd.Function"`.
- `foreigndata`: a host constructor (`from_numpy`, `from_dlpack`, `frombuffer`, or a
  tensorless `tensor` / `as_tensor` / `asarray`) that brings data back into torch after a
  plain-code `numpy` / `__array__` / `__dlpack__` / `to_dlpack` / `tolist` export outside any
  Function, with `transform_kind="foreign_data"` and `transform_config["resolution"]` of
  `view`, `copy`, `constant` or `unresolved`.
- `jitscript`: one call to a `torch.jit.ScriptFunction` from an eager forward, with
  `transform_kind="jit.script"`.
- `tl.model_output` selects model outputs, and `tl.model_input` selects a nested model-input
  path.

**Temporal composition**: `tl.followed_by` (retroactive successor) and `tl.preceded_by` (lookback predecessor)
compose with boolean operators to form temporal predicates over the op stream.

**Backward selectors**: NODE selectors: they address an autograd node and its `grad_input` tuple, resolved against
`trace.autograd.grad_fns` after `backward()`.

- `tl.grad_fn`: a backward grad_fn; `label=` matches one exact grad_fn label.
- `tl.grad_input` / `tl.grad_output`: backward event tensors.
- `tl.in_backward_pass`: one backward pass number.
- `tl.without_op`: grad_fns without a paired forward op (the former `intervening` alias is
  removed).
- The gradient of a forward site's output is addressed with `tl.grad(site)` instead (see
  **tl.grad**).

**tl.grad(site)** (experimental): Builds a `GradSelector` that names the gradient of the loss with respect to the site's OUTPUT,
observed at the TorchLens boundary before it enters the op's backward function: the same
tensor `trace[site].grad` stores. It is an address and never calls autograd.

- `GradSelector`: frozen, hashable, selector kind `"grad"`; also
  `torchlens._selection.selectors.grad`. `repr(tl.grad("relu_3"))` is `tl.grad('relu_3')`.
- Accepts: a label (pass-qualified `"relu_3:2"` on multi-pass layers); any activation selector
  (`tl.label`, `tl.func`, `tl.module`, `tl.in_module`, `tl.label_contains`,
  `tl.label_matches`, `tl.where`, `tl.model_output`, `tl.model_input`, `tl.followed_by`,
  `tl.preceded_by`, `tl.at_step`, ... and `&`/`|`/`~` compositions of these); an `Op` or
  `Layer` record; or a `find_sites()` table.
- An op with several outputs has one gradient per output (`tuple_index` names the slot).
- Refusals raise `SelectorCompositionError` carrying `code` and `remedy`: a nested `tl.grad`
  (`grad_selector_nested`); a node selector or a pass qualifier inside
  (`grad_selector_inner_not_forward`: write `tl.grad(s) & tl.in_backward_pass(2)`); a facet or
  head selector (`grad_facet_unsupported`); a `Selection` (`grad_selection_unsupported`); a
  list or mapping (`grad_selector_inner_invalid`).
- No site entry point takes a pass keyword; the site decides.
- Reading is unchanged in spelling: `trace[site].grad`, `.grads` and `.has_grad`
  (`trace[tl.grad(s)]` is not an indexing form).

**Activation, gradient and node universes**: Every site belongs to one universe: activation, gradient or node. A bare label never reaches
the backward graph unless wrapped in `tl.grad`.

- Activation site: a bare selector or label (forward out hook, payload the activation).
- Gradient site: `tl.grad(inner)`, the output gradient of each forward op `inner` matches.
- Node site: the backward selectors (a grad_fn post-hook per `grad_input` slot).
- Composition: `tl.grad(a) | tl.grad(b)` is one gradient branch. `tl.grad(s) | s` is two
  branches (activation and gradient, each resolved, mounted and recorded separately; union
  keeps both because identity is typed, never the label alone). `~tl.grad(s)` normalises to
  `tl.grad(~s)`, and `tl.grad(s) & tl.in_backward_pass(n)` restricts the gradient branch to
  pass `n`.
- Refusals: `&` across universes raises `selector_cross_universe_and` (`tl.grad(s) & s`,
  `s & tl.grad_fn(...)`, `tl.grad(s) & tl.grad_fn(...)`); `~` over a union that spans
  universes raises `selector_cross_universe_not` (negate inside each branch:
  `tl.grad(~a) | ~a`); a pass qualifier on a bare activation selector raises
  `selector_signal_mismatch`; `tl.module(a) & tl.head(i).in_module(b)` with `a != b` raises
  `selector_module_facet_scope_conflict`.
- Resolution: a gradient site resolves its inner selector exactly as the activation entry point does
  (zero matches refuse). A matched op the loss never reaches is a `no_fire`, never a
  resolution error and never a zero tensor.
- `Trace.find_sites(tl.grad(s))` returns the matched forward sites with `signal="backward"`;
  its `.to_dataframe()` table carries that value in a `signal` column (`"forward"` for
  activation sites, `"backward"` for node sites).

**signal (record field)** (experimental): The one field that says which pass a record or event belongs to: `"forward"` or `"backward"`
(plan and span summaries may say `"both"`).

- Spelled `signal` on `HookContext`, `FireRecord`, `TapRecord`, `Edit`, `NarrationEvent` and
  `Rule` (derived), on the audit payload, the coupling record and the span record.
- Sites carry no `signal=` keyword; the only entry point that does is
  `tl.observers.span(name, signal="both")`.
- `HookContext` also carries `ctx.state` (the per-(site, hook) scratch dict) and
  `ctx.run_state` (the run-scoped dict). A callable attached to `tl.grad(s) | s` runs in both
  passes and reads `ctx.signal`.

**`Accessor.count_by(*keys)`**: Every accessor (`trace.layers`, `trace.ops`, `trace.modules`, `trace.params`, ...) counts its
records by one or more record fields. `len(accessor)` is the total.

- Result: one key gives `{value: count}`, two or more give `{(v1, v2, ...): count}`, in
  first-seen order.
- Keys: layers and ops count by `"module"` (containing module address) and `"kind"` (the op
  family token, `conv2d`, `relu`); other accessors count by any string or int field their
  records carry (`trace.modules.count_by("class_name")`, `trace.params.count_by("dtype")`).
- Refusal codes: `accessor_count_by_key_required` when no key is given;
  `accessor_count_by_key_unknown` for an unknown key, naming the countable fields.
- Filtering (the labels for one value) is a selector lookup:
  `trace[tl.in_module(address)].labels`.

## Intervention helpers

**Value helpers**: Ready-made edits that change the value at a site. Availability outside the torch backend is
narrower; see the per-backend rosters in the backends guide.

- Helpers: `tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`, `tl.add_noise` (Gaussian
  noise), `tl.mean_ablate` (fill with a mean; its own entry below), `tl.replace_with` (fixed
  value), `tl.steer` (add a scaled steering direction), `tl.project_onto` / `tl.project_out`
  (keep or remove the component along a direction), `tl.splice_module` (call a module as a
  black-box forward splice), and `tl.clip_norm` (norm clip).
- `torchlens.intervention.scramble_elements`: elementwise iid scramble from a flattened
  source. It replaces the removed `resample_ablate` spelling, which a saved spec may not name
  either (it raises `intervention_helper_renamed`). For coherent donor patching use
  `tl.patch_from`.
- Shape: the value helpers take the keyword-only `allow_shape_change=` (default `False`). By
  default the edit must preserve the site's output shape; setting it allows a change.
- `tl.steer` also takes `strength=` (its second argument, default `1.0`), the multiplier
  applied to the steering direction, and `dim=`, the out axis a vector direction lies along.
  When `dim=` is omitted it is inferred only if exactly one out axis has the direction's
  length (`tl.steer(direction, 4.0)` with a `[d_model]` direction on a
  `[batch, seq, d_model]` site); two or more matching axes, or none, raise
  `tl.errors.AmbiguousAxisError` naming the matching dims. Pass `dim=` whenever another axis
  can take the same size during a run (a generated sequence growing to `d_model` tokens).
- Tensor-carrying helpers (`tl.steer`, `tl.mean_ablate`, `tl.project_onto`,
  `tl.project_out`) keep the caller's tensor by reference. Each staged entry records a
  full-content digest of those tensors when it is staged; once the tensor changes (in place
  or through `.data`), a rerun and `save_intervention` raise
  `helper_tensor_changed_since_capture`. Re-stage the edit to use the new value.
- A bound executor (`spec.bind(model)`) instead reads the helper's tensor at each call, so
  in-place changes apply to the next call. `BindReport.helper_tensor_versions` (on
  `.last_report`) records each tensor's version counter, keyed by the rule id the call
  started with (a rule id follows its tensor's content).

**`tl.mean_ablate(source=None, *, dim, allow_shape_change=False)`**: The one mean entry point: it fills the site with a mean. A mean is a reduction, never a draw, so it
takes no seed.

- `dim` (required) picks the axes: `None` averages every axis of the source to one scalar,
  whatever the source; an int or tuple of ints averages those axes (`keepdim`, broadcast back
  to the site); `"members"` averages over a `Population`'s members and raises
  `intervention_over_invalid` for any other source.
- A semantic axis name (`"batch"`, `"position"`, `"feature"`) raises
  `axis_semantics_unknown` until axis-role evidence exists; any other `dim` raises
  `intervention_over_invalid` at construction.
- `source` is what is averaged. The default is the site's own value; it reads the whole
  batch, so the spec derives `is_batch_independent=False`, the append gate refuses and a
  chunked capture raises `chunked_replay_batch_coupling`.
- A tensor or Op-like `source` is averaged instead of the site (`is_batch_independent=True`).
- A `tl.intervention.Population` source has its members averaged elementwise first:
  `dim="members"` fills with that member-shaped mean, `dim=None` reduces it to one scalar,
  and axes reduce it with `keepdim`. The spec is `opaque_audit` and the fire record says
  `stochastic=False`.
- Removed spellings: `mean_fill(dim=...)` is now `mean_ablate(dim=...)`, and
  `mean_from(population)` is now `mean_ablate(population, dim="members")`. A saved spec
  naming `mean_fill` raises `intervention_helper_renamed`.

**Edits on gradient sites**: A tensor edit is a function of one tensor, and the site decides which tensor, so the neutral
edits apply to activation and gradient sites alike. There are no gradient-only helpers:
`trace.do(tl.grad("relu_3"), tl.zero_ablate(), model=model, inputs=x)` zeroes a gradient.

- Neutral edits (their `Edit.signal` is `None`): `tl.scale`, `tl.zero_ablate`, `tl.clamp`,
  `tl.add_noise`, `torchlens.intervention.scramble_elements`, `tl.steer`, `tl.project_onto`,
  `tl.project_out`, `tl.replace_with`, `swap_with` with a fixed tensor donor, `tl.clip_norm`,
  and any callable `fn(tensor, *, ctx)`.
- Edits that read recorded forward state (`Edit.signal == "forward"`): `tl.patch_from`,
  `tl.mean_ablate`, `tl.splice_module`, and `swap_with` with a site or trace donor. On a
  gradient site they raise `edit_signal_mismatch` at attach time, before any mutation.

**tl.clip_norm(max_norm, norm_type=2.0)** (experimental): Neutral, portable edit that rescales its tensor so its `norm_type` norm is at most `max_norm`,
per tensor. On `tl.grad(site)` it is gradient-norm clipping at that site.

**Armed gradient edits and Trace.armed_edits** (experimental): An edit on a gradient site is armed: it waits as a backward hook until a backward pass runs.
`Trace.armed_edits` lists the armed rules.

- Rerun: `trace.do(tl.grad(s), edit, model=model, inputs=x)` reruns the forward with the
  activation half of the plan applied live, and carries the gradient half into the result's
  recipe as sticky backward hooks of the returned execution only (the source trace is not
  armed). The rerun is made graph-connected whatever the source's `backward_ready` was, and
  says so in `result.last_run["backward_ready_forced"]`.
- Arming record: its `intervention_event_v2` envelope has status `"armed"` (beside `"fired"`,
  `"no_fire"`, `"error"`), `fire_count=0` and
  `extra={"armed_edits": n, "awaiting": "backward"}`. A mixed plan keeps the activation half's
  `fired`/`no_fire` status and adds the same keys.
- `Trace.armed_edits`: the read-only tuple of the gradient rules in the trace's recipe. The
  trace summary says "N gradient edits armed; call trace.backward(loss)" until they fire.
- Settling: each `trace.backward(loss)` writes one settle envelope per arming transaction
  (`door="backward"`, `lane="live_hook"`, `parent_event_id` and `extra["arming_event_id"]`
  the arming row, `backward_pass_number`, status `fired` or `no_fire` with `no_fire_reason`
  `"gradient_not_reached"` or `"pass_filtered"`). A second backward never rewrites the first
  pass's row.
- `loss=` on `Trace.do` (and `tl.sweep`): a callable of the returned trace returning a scalar
  with a `grad_fn`; the entry point runs `result.backward(loss(result))` in the same call. A
  non-callable raises `do_loss_not_callable`, a plan with no gradient site raises
  `do_loss_without_gradient_site`, and a callable that raises writes an `error` envelope,
  detaches the hooks this call attached, and re-raises.
- `engine="set_only"` arms the gradient rules on the trace itself, for its own
  `trace.backward(loss)`. Staging the recipe
  `do(tl.when(tl.grad(s), edit), intervention=InterventionOptions(engine="set_only"))`
  registers the sticky rule without a rerun. When the trace holds no live backward graph, its
  staging history row carries `armed_for="next_rerun"` (the rule waits for the next rerun's
  backward).
- Saving: a saved trace keeps the armed rules and records their count in
  `annotations["armed_gradient_edits"]`.

**Replay verbs**: `Trace.do` is the one editing verb on a captured log, and it propagates what it stages.
`Trace.run` performs a full-forward run with the log's active intervention spec, and
`tl.sweep` captures one intervened trace per swept replacement value.

- What `do` accepts: a non-callable value sets the site; a hook or helper attaches as a
  sticky rule; a mapping of sites to edits (`trace.do({a: h1, b: h2})`) applies several edits
  in one propagation; an `InterventionSpec` or `Recipe` applies its rules, including a spec
  loaded back with `tl.io.load_intervention_spec`.
- Signatures: in `Trace.do(site, value_or_hook, /, *, ...)` the first parameter is `site`,
  positional-only, on `Trace`, `Layer`, `Op` and `Bundle` alike. In
  `Trace.run(model, x, /, *, inputs=..., ...)` the model and input are positional-only;
  positional inputs return a new `Trace`.
- Engine options: on the replay engine the edit is pushed downstream through the recorded
  graph (DAG replay of the affected cone). `InterventionOptions(engine="set_only")` stages the
  edit without propagating. `InterventionOptions(differentiable=True)` replays onto a new fork
  whose outputs stay differentiable and returns it.
- Undoing: `Trace.clear_edits` detaches staged edits, and the scoped form
  `with trace.fork() as scoped: scoped.do(...)` detaches the last `do`'s hooks on exit.
- Removed: the former `replay`/`replay_from`/`rerun` aliases, and the separate stage,
  propagate and detach verbs that `do` folded in; docs/reference/deprecations.md maps each.
- Gradient sites: replay re-executes forward ops only, so a gradient site on `do` without
  `model=`/`inputs=` or on `engine="replay"` raises `replay_gradient_edit_unsupported`
  before any mutation ("Pass model= and inputs= to rerun with the edit armed, then call
  result.backward(loss) (or pass loss=), or stage it with engine='set_only' and call
  trace.backward(loss) on this trace"). Differentiable replay of gradient edits is out of
  scope this release.
- Multi-pass layers: the replay engine works on pass-qualified op labels (`label:pass`).
  Cone traversal, the replay overlay, hook targets and commits are keyed per pass, so edits
  on multi-pass (recurrence-grouped) layers touch exactly the addressed pass and recompute
  every downstream pass.
- Bare labels: an edit or selector that names a multi-pass layer by its bare label (a `do`
  string target, `tl.label(label)`, so `trace[tl.label(label)]`) raises
  `multipass_bare_label_ambiguous`, and `trace.ops[label]` raises `ambiguous_op_lookup`;
  single-pass bare labels stay accepted. Reading is not refused: `trace[label]` returns the
  aggregate Layer, and `trace["label:N"]` or `trace["label:last"]` gives one pass's Op.
  Replay records (`last_run` origins/cone, `_replay_frontier` keys) spell multi-pass ops
  pass-qualified and keep bare labels for single-pass layers.
- Staged spec is read-only: a rerun (plain, append, or chunked), a failed or interrupted
  rerun, and a fork leave the trace's staged spec exactly as staged.
- Capture-time predicates: an `intervene=` predicate that is not lowered to module hooks
  stages one entry per fired op on its final label. `Trace.run` re-arms the trace's retained
  predicate through the capture entry point instead, and raises `rerun_predicate_restage_mismatch`
  (trace unchanged) when the predicate fires at different ops than the staged entries name,
  or at the same ops with changed rule content (a staged steer edited since capture).
- Exact labels: a staged edit addressed by a finalized exact label (`do(label, value)` with
  the `set_only` engine, `do(tl.label(...), edit)`) is translated before the forward to the
  raw capture labels of the ops it names, so it fires on the rerun. A mixed gradient plan's
  activation half already uses the same translation. Each pass of a multi-pass op keeps its
  own raw label, so a pass-qualified label edits that pass alone. A substring or regex
  selector on a finalized label still refuses through the live matcher
  (`LiveModeLabelError`, with its authoring remedy).
- Unmatched selectors: a staged edit whose selector matches sites on the trace but fires at
  none during a rerun of the same graph on the same inputs raises
  `rerun_selector_unmatchable_live` (trace unchanged). So does, on any inputs, a `tl.where`
  predicate that matches sites on the rerun's own trace but fired at none during its forward
  (a predicate on `p.layer_label`, which exists only after postprocess). A predicate that
  matches nothing on the new inputs warns `rerun_zero_fire`.
- Divergence check: the rerun divergence check (`ControlFlowDivergenceWarning`, or
  `ControlFlowDivergenceError` with `strict=True`) compares raw op order, edges and shapes,
  with each value-only edit node folded into the op it replaces. A staged edit is never
  reported as control-flow divergence, while a changed op, edge, shape or batch size still is.

**Guarded fast rerun** (experimental): The fast engine a rerun takes on a trace that saved a selection of sites, when nothing is
staged or the staged intervention spec targets only plain module selectors (`tl.module(...)` /
module-address hooks, no `set()` value replacements). It runs a native forward with the staged
hooks, if any, fired at each targeted module's exit and forward hooks refreshing the saved
sites, so the rerun costs close to a plain hooked forward instead of a full capture.

- Legacy entry point: `trace.run(model, x)` tries it first on a trace that saved a selection (edited
  or not) or stages an intervention, and falls back to the capture engine on any typed
  refusal. A trace that saved every activation with nothing staged keeps the capture rerun
  (`engine` `rerun`, `fast_refused` None): its value is every op's record, which only a
  capture produces.
- Explicit entry point: `trace.run_transaction(inputs=..., fast=True)` on a live or loaded-activation
  trace applies the staged spec through it.
- Without `fast=True`: the plain `trace.run_transaction(inputs=...)` on a live trace applies
  the staged spec too, inside its refresh capture (`engine` `live`, the fork adopts the
  captured graph and `last_run["unfired_plan_ids"]` names the staged plans that fired
  nowhere). Only a loaded artifact refuses with `run_staged_spec_unapplied`.
- Outcome: `last_run` reports `engine` as `guarded_fast` when the fast engine ran and `rerun`
  after a fallback, where `fast_refused` names the refusing guard as `<code>:<stage>`.
  Examples follow.
- `run_capability_unavailable:fast_rerun_target_unsupported`: a value replacement or a
  non-module target.
- `run_capability_unavailable:fast_rerun_graph_unsteered`: a staged entry has never fired in
  the trace's recorded graph (as after a sticky hook staged by `do` with `engine="set_only"`
  on a plain capture), or the graph records a fire no staged entry accounts for (as after
  `clear_edits(site)` on a steered capture), so the capture engine reruns once to record the
  current plan.
- `<divergence code>:fast_live_call_fingerprint`: a structural change.
- Fire counts: `hooks_fired` / `hooks_unfired` count the staged plans that fired, and a plan
  that never fired settles by `on_zero_fire` (default: warns `rerun_zero_fire`).
- Verification: through `run_transaction(inputs=...)`, fast or not, a run that applies staged
  edits reports `unverifiable`, never `verified` (diagnostic `user_intervention_unsupported`,
  stage `hooked_run_unverified`): it is an edited forward, and no hook-aware check exists yet.
- Source trace: on a live trace the `fast=True` entry point returns a new result Trace carrying that
  `last_run` and leaves the source Trace untouched; the legacy rerun entry point (`run(model, x)`)
  also returns a new Trace and leaves the source untouched.
- Outputs: output leaves (the model's and each saved module's) are paired with the captured
  output ops by container path through capture's own output walker, so a stock Hugging Face
  forward returning a `DynamicCache` of per-layer key/value tensors reruns fast. A changed
  leaf count or an unmatched path still raises
  (`output_structure_mismatch:fast_live_model_output_structure`).
- Input sizes may differ from the capture (generation) when rank, dtype, device and input
  tree match and the capture sealed a call hash (`trace._raw_call_fingerprint`, an
  ordered rolling hash of every wrapped torch call and module entry, compared against
  `last_run["call_fingerprint"]`). Such a run sets `last_run["shape_varied"]` and resets
  `shape`, `transformed_out_shape`, `activation_memory` and `transformed_activation_memory`
  to `None` on every op the run did not refresh, so no capture-time number is presented as
  current.
- Save scope never widens: unsaved ops stay unsaved, `transformed_output` follows the
  capture's output-transform rule, and the stored spec is unchanged.
- Refusals: a real structural divergence still refuses (`PathDivergenceError` from the
  explicit entry point; the legacy entry point falls back and records why). A staged helper tensor changed
  since it was staged raises `helper_tensor_changed_since_capture` on both entry points, as the
  capture rerun does. The explicit entry point's cached session rebuilds whenever the staged spec
  changes, so it always applies the spec staged at call time.

## Extraction, observers, and admin

**Extraction helpers**: Helpers that return saved outs for chosen layers, or for a whole dataset in batches. (The
former `peek` and `batched_extract` aliases are removed.)

- `tl.pluck` returns the saved out for one layer, `tl.extract` for many layers (detached by
  default; `detach_saved_activations=False` returns graph-connected outs).
- Both skip the capture when every requested name is an exact module address, the model is in
  eval mode and the input is one tensor or text on the model's device: one plain forward with a
  hook on each module returns the same tensors. A `tl.pluck` `stop_after=` that is also an exact
  module address ends that forward right after the module's first call. Any other request
  runs the full capture (see [Fast paths](../guides/fast_paths.md)).
- `tl.extract_dataset` extracts outs from an iterable dataset in batches.
- Disk-mode `extract_dataset`
  (``path=``) writes a SELF-DESCRIBING artifact: atomic ``batch_XXXXX.pt`` shards plus a
  ``manifest.json`` recording the run signature, per-site identity (layer label and
  structural site key where derivable), stimulus ordering/provenance
  (``stimulus_ids=`` optional), axis semantics, dtypes, devices, and the TorchLens version.
- ``resume=True`` (experimental, like ``stimulus_ids=`` and the loader) continues an
  interrupted disk run from its last completed shard after a signature check.
  `torchlens.load(path)` reads the artifact back with its metadata as a `LoadedExtraction`.

**unwrap_when_done (CaptureOptions)**: `CaptureOptions(unwrap_when_done=True)` (default `False`) removes TorchLens's torch
function wrappers when the `tl.trace` call returns, so `torch.cos`, `torch.Tensor.__add__` and the
rest are PyTorch's originals again. By default the wrappers that the first capture installs stay
in place until `torchlens.backends.torch.wrappers.unwrap_torch()` is called; the next capture
installs them again.

**torchlens.backends.torch.wrappers.wrapped()**: A context manager that installs TorchLens's torch function wrappers on entry and
removes them on exit, so torch is clean again after the block: `with wrapped(): trace =
tl.trace(model, x)` leaves `torch.cos` and the rest as PyTorch's originals once the block ends.

- `escape_detector=` (`"off"` or the diagnostic `"shadow"`) and `completeness_witness=` (a
  bool, `"off"` or `"shadow"`) are passed to `wrap_torch()`.

**Module and stage names** (experimental): A module name resolves to that module's output at every selector entry point: `tl.extract`,
`tl.pluck`, `tl.trace(save=[...])`, `trace.run_transaction(inputs=..., until=...)` and the
typed op entry points (`trace.ops[name]`, `trace.layers[name]`). At the plain entry point `trace[name]` it
is the Module itself (see *Indexing coherence*).

- Three spellings name a module: its address (`"fc"`, `"layer1.0"`); a name relative to a
  wrapped inner model (timm `FeatureGetterNet` reports `"blocks.9"` for the module at
  `"model.blocks.9"`); and a dotted name whose module was registered with dots flattened to
  underscores (timm `FeatureListNet` registers stage `"stages.0"` as `"stages_0"`).
- A module whose output is also a model output resolves to its own output op, not the
  model-output alias.
- A relative name that ends several module addresses raises `ambiguous_op_lookup` and lists
  them; a `":<call>"` suffix selects one call of a reused module.

**Intervened extraction** (experimental): `tl.extract_dataset(..., intervene=spec)` takes a `tl.when(selector, helper)` spec and
captures every batch with it armed, so the stored rows are the intervened activations.

- Site freeze: batch zero freezes the structural sites the spec fired at. A later batch that
  fires nowhere raises `extraction_intervention_no_fire`, and one that fires elsewhere
  raises `extraction_intervention_site_drift`.
- Supported addresses: module (`tl.module`, `tl.in_module`) and op-type (`tl.func`)
  addressed specs run; finer address classes raise `harvest_address_class_ungated`. Every
  helper must declare `is_batch_independent=True`.
- Disk mode records the counterfactual in the manifest's `tl_counterfactual_harvest_v1` block
  (policy, engine `live_hook`, address class, site keys, fire receipt digest), records each
  shard's fire receipt, and puts the intervention identity in the resume signature. A clean
  run declares `policy="none"`.

**Token store** (experimental): `tl.extract_dataset(..., store_input_ids=True)` stores each batch's token ids as the ordinary
`input_ids` key, trimmed by the batch mask under `ragged="trim"` like the token-axis
activations. Padding is never a token row.

- Reading tokens: `ExtractionReader.token_batch(shard_index, keys=None)` and
  `ExtractionReader.token_rows(keys=None, *, max_bytes=None)` read stored keys one row per
  token as a `torchlens.dataset_extraction.TokenRows`. Fields: `values` per key,
  `stimulus_row`, `position`, `num_tokens`.
- Shuffling:
  `ExtractionReader.shuffled_batches(batch_size, *, seed=0, keys=None, buffer_shards=1, tokens=False)`
  shuffles across `buffer_shards` shards at once (the cross-shard mixing buffer) and yields
  token rows when `tokens=True`.

**Observers**: Observers record extra information at sites during a capture.

- `tl.observers.tap` creates a tap observer for a site. `tap(tl.grad(s))` records gradients
  with `signal="backward"` and `grad_kind="grad_output"`; `tap(tl.grad(s) | s)` records both.
- `tl.observers.span(name, signal="both")` records a named observer span around captures or
  hook execution; `signal=` (`"forward"`, `"backward"` or `"both"`) scopes which records carry
  the span name. The former `record_span` alias is removed.
- `tl.observers.register_tensor_connection` registers a manual parent-child tensor edge
  during capture.

**Validation entry**: `tl.validate(model, x, scope=...)` validates a model/input pair for a requested scope,
capturing what it needs itself. The scope is `"forward"` when omitted, so
`tl.validate(model, x)` runs the forward replay check; others include `"saved"` and
`"receptive_field"`.

- Validation trace: on the forward and saved scopes (torch backend), `trace_observer=` is
  called exactly once with the validation `Trace` after replay and before cleanup, or with
  `None` when a precondition refusal returns `False` before any capture. It reads public
  fields (`len(trace.ops)`, `trace.provenance.fingerprint`,
  `trace.validation_replay_status.state` / `.reason` / `.num_failed_ops`) and keeps values,
  not the trace. `num_threads=` pins torch intra-op threads for every validation forward and
  restores them on return and on error. Neither changes a check or the `bool` verdict.
- RNG: like `tl.trace`, it seeds the global RNG streams (Python `random`, NumPy, torch CPU and
  CUDA) for its own runs and restores the caller's streams on return and on error. The
  lower-level `torchlens.validation.validate_forward_pass` and `validate_backward_pass` make
  the same promise, as does `Trace.validate_forward_pass`, which replays each op under its
  recorded RNG state.
- Carried graphs: under `scope="backward"`, when torch reports a backward through a graph an
  earlier backward freed and a module attribute holds a tensor with autograd history, the
  model is refused with `ValidationError` code `backward_validation_carried_graph`, naming
  those attributes. Without that evidence torch's own error propagates unchanged.
- Keywords: keyword-only `output_device` and `save_budget` set the validator capture's save
  options, with the same values and defaults as `CaptureOptions.output_device` (`"same"`) and
  `CaptureOptions.save_budget` (`"auto"`). Both apply to the `"forward"`, `"saved"`, and
  `"intervention"` scopes; a non-default value on `"backward"` or `"receptive_field"` raises
  `TypeError`. The lower-level `torchlens.validation.validate_forward_pass` accepts the same
  two keywords.
- `output_device="cpu"` lets a GPU model validate with its saved activations in host memory
  (replay still runs each op on its original device). Saved function arguments (the replay
  inputs) other than parameters are kept in host memory too and moved back to each op's
  device for its check, so the GPU keeps only the parameters and one forward's transients.
- Budget: an exceeded budget raises `SaveBudgetExceededError` as the capture does. On every
  `output_device`, parameter arguments the validator capture keeps as references to its own
  copy of the model are not charged to its `save_budget`.

**Session admin**: Housekeeping calls for traced models and the TorchLens session.

- `tl.release_model` evicts TorchLens's module metadata and preparation bookkeeping for a
  traced model, and normalizes plain attributes holding torch functions from the other wrap
  epoch, so whole-model pickle / `torch.save` survive later wrap-state changes. A capture
  itself leaves no TorchLens callable on the model: submodule `forward` wrappers are
  session-scoped.
- `tl.clear_cache` empties the capture cache.
- `tl.io.list_traces` / `tl.io.reset_naming_counter` manage log bookkeeping.

## Persistence, containers, and namespaces

**`tl.io.detect_tlspec_format(path)`**: Inspects a saved-artifact directory without loading its payloads. Returns `"v3"` when the
root manifest names saved format 3, `"pre_3"` for save markers of torchlens 2.37 and earlier,
or `"unknown"` when no TorchLens save markers are present. Symlinked metadata paths refuse.
Detection does not validate the artifact; use `tl.validation.validate_tlspec(path)` for that.

**`tl.types.UnencodableValue`**: The stand-in a loaded saved format 3 artifact carries where the save met a value outside the
saved format's vocabulary (a user object in a module attribute, a custom metadata value).

- Fields (frozen): `type_name` (the value's `module.qualname`), `repr_text` (its repr text,
  bounded) and `at` (the record, row and column it sat in).
- Advisory: when the artifact's manifest records such values, the load attaches one
  advisory, `unencodable_values_stringified`, to the trace's capture advisories.
- Refusals: a record cell that fails its typed schema raises `record_schema_invalid`.


**Save / load**: `Trace.save` writes a `Trace` into a portable `.tlspec` directory bundle at a chosen level;
`tl.load` loads a `.tlspec` object with eager tensor materialization. Saved format 3
(torchlens 2.38) is specified in `docs/maintainers/saved_format_3.md`, a maintainer document
that is not on the site.

- Symlinks: both entry points check the final path component only (`path.is_symlink()` on the bundle
  path and, at load, on its manifest, metadata and blobs entries). A symlink there raises
  (`load_path_symlink_rejected` at load); a symlinked parent directory passes on both sides.
  Save and load use the same rule, and `storage=tl.to_disk(...)` and `save_intervention`
  apply it to their own targets.
- Failed or unfinished saves: every save stages into `<name>.tmp.<hex>` beside the path and
  renames it into place. A failed overwrite puts the previous copy back and leaves that
  staging folder marked `PARTIAL`, with a `REASON.txt`; a killed save leaves it unmarked.
  While such a folder is at least as new as the copy on disk, `tl.load` raises
  `newer_save_unfinished` instead of serving the older copy as current. Re-save, or run
  `tl.io.cleanup_tmp(path)` to accept the older copy (`force=True` for an unmarked folder).
  A leftover folder older than the copy only warns. A failed first save leaves nothing at
  the path, and loading it fails as for any missing path.
- Values with no typed encoding: written as text and counted in the manifest's
  `disclosures`. They load back as `tl.types.UnencodableValue`, a frozen stand-in carrying
  `type_name` (the original type's dotted name), `repr_text` (its `repr`, cut at 256
  characters) and `at` (where it was in the artifact). A saved cell that fails its type check
  on load raises `record_schema_invalid`.
- Interventions: `Trace.save` writes the staged intervention spec as the typed recipe
  `intervention.json` beside the intervened values and their per-op provenance
  (`save_intervention` writes the same recipe as an artifact of its own). A loaded trace whose
  recipe holds only built-in edits reruns it through every legacy rerun entry point (`run(model, x)`
  plain, append or chunked, a fork's `run`, `do(..., intervention=InterventionOptions(engine="rerun"))`, and an edit staged after
  loading).
- Custom callables: saved as a callable placeholder (its position and qualified name, never
  its code). A recipe holding one raises `run_intervention_spec_not_persisted`, naming the
  rule, rather than rerun without the recorded intervention. The replay and
  `run_transaction(inputs=...)` entry points cannot run an analysis artifact at all.
- Re-saving on rerun: a rerun replays the capture's recorded `save=` request on its own graph,
  so it saves the same sites a fresh capture would, even after a staged edit inserts ops. A
  trace restored from pickle carries no recorded request and re-saves by the capture's raw op
  indices, which is exact only when the rerun graph matches op for op; otherwise the rerun
  raises `rerun_resave_ops_shifted` (trace unchanged).
- Load hints: `tl.io.PayloadLoadHints` carries backend-specific payload materialization hints
  (`tl.io.JaxPayloadLoadHint` is the JAX-specific form).

**Options**: `tl.options` groups the public option dataclasses. The entry points take only the model and the
inputs by position; every other setting rides an options object or an entry point keyword.

- Dataclasses: `CaptureOptions`, `VisualizationOptions`, `ReplayOptions`,
  `InterventionOptions`, `StorageOptions`, `RecordingOptions`.
- Capture settings ride `capture=CaptureOptions(...)`, including `activation_transform`,
  `grad_transform`, `save_raw_activations`, `save_raw_grads`, `save_mode`,
  `reconstruction_ready`, `lookback`, `lookback_payload_policy`, `grouping`,
  `jax_static_argnums`, `grad_options`, `chunk_size`.
- Recorder settings of `tl.record` ride `recording=RecordingOptions(...)`, matching `tl.trace`'s
  `capture=`: `default_op`, `default_module`, `history_size`, `include_source_events`,
  `max_predicate_failures`, `on_predicate_error`, `on_forward_error`, `default_grad`,
  `postprocess`, and `lean`, a PROVISIONAL SPELLING (see
  **lean (tl.record; PROVISIONAL SPELLING)**).
- Entry point keywords: `save=`, `intervene=`, `recording=` (on `tl.record`), `stop_after=` (a site
  or a predicate), `storage=`, `narrate=`, `episode=`, `facets=`, `chunk_paths=`, `backend=`,
  `input_shape=` and `on_zero_fire=`.

**lean (tl.record; PROVISIONAL SPELLING)**: `tl.record(..., lean=True)` (also `RecordingOptions(lean=True)` and `Recorder(lean=True)`)
journals only the events the selection keeps. An op or module boundary the save predicate does
not keep gets no shell journal event, no frozen record and no replay clone, which removes most
of the per-op bookkeeping of a one-module read. Default `False`: a default recording's contents
are unchanged.

- Spelling: the behaviour is settled, but the public spelling `lean` is provisional and may
  change, so no other name derives from it.
- What was omitted: `Recording.lean_omissions` names the omitted content
  (`shell_records_of_unkept_ops`, `module_boundary_events_of_unkept_modules`,
  `event_indexes_of_unkept_events`, `op_counts`; `()` on a default recording).
- Reads that need the full journal refuse with `recording_lean_omitted` instead of answering
  from a partial journal: `Recording.event_stream`, `Recording.op_events`,
  `Recording.lazy_trace`, `Recording.raw_metadata()`, `Recording.to_trace()`.
- Complete reads: `Recording.records`, `Recording.output`, `summary()`, `to_pandas()` and the
  other retained-record reads.
- Refusals: gradient capture (`backward_ready=True`, `save_grads=`, `default_grad=`),
  `intervene=`, `narrate=` and the partial forward-error modes refuse at construction with
  `record_lean_unsupported`.
- Related: `RecordingSettings.lean` keeps the flag with the recording's settings.

**emit_nvtx (CaptureOptions)**: When `True`, the torch capture marks each logged operation with an NVIDIA NVTX range named
`torchlens::<op>#<call_id>`, so Nsight Systems and Nsight Compute timelines show the model's work
op by op; TorchLens's own work appears under `torchlens::internal::`. Default `False`; needs CUDA.

**on_nonfinite (CaptureOptions)**: The one NaN/Inf policy, replacing `raise_on_nan` and `track_nonfinite`. Any other value
raises `on_nonfinite_invalid`.

- `'ignore'`: the default; neither checks nor records (today's default behaviour).
- `'track'`: per-op finiteness verdicts, served by `Trace.nonfinite_ops` /
  `Trace.nonfinite_coverage`.
- `'warn'`: tracks, and warns `capture_nonfinite_detected` once per capture when any op
  produced a non-finite output.
- `'raise'`: stops at the first non-finite tensor and raises.

**Structural hash**: `tl.fingerprint` is the provisional structural-hash namespace; `tl.fingerprint.assert_unchanged`
asserts a model still matches a pinned address-free structural hash.

**Container registration**: `tl.types.register_container` registers a custom container type (a flatten/unflatten pair) for
capture and reconstruction. `tl.types.Container` is the computed view over a captured Python
output container.

- Stable surface: `tl.types.Container`, `Op.container`, the `input_at` and `output_at`
  selectors, `tl.types.register_container`, `Container.reconstruct`,
  `Trace.reconstruct_output`, `show_containers` (including `"nodes"`) and the
  `capture_container_structure` option (see [Containers](../containers.md)).

**Capture-product bases**: `tl.types.TraceLike` is the shared base for uncooked and cooked capture projections.

**Namespaces**: Submodules that group related public names.

- `tl.recording`: the sparse predicate-recording namespace behind `tl.record`.
- `tl.facets`: the semantic facet namespace (canonical home `torchlens._semantic`).
- `tl.export`: static export helpers.
- `tl.Bundle(...)` / `tl.Bundle`: build aligned Trace collections; `Bundle.draw` (below)
  draws a bundle's merged graph. `tl.load(path, lazy=True)` on a saved bundle passes `lazy`
  (with `map_location` and `materialize_nested`) to every member.

**Appliance entry point (`Trace.using`)** (experimental): `trace.using(name)` returns the appliance registered as `name`, bound to that trace, for
example `trace.using("mechinterp").logit_lens()`. Appliances hang off this one entry point instead of
adding methods to `Trace`.

- Built in: TorchLens registers `"mechinterp"`, the mechinterp kit, whose methods return
  exactly what the matching `torchlens.mechinterp` functions return.
- Registering: TorchLens's own and outside packages register the same way, with
  `torchlens.appliances.register(name, factory, summary="", replace=False)`, where `factory`
  is a callable taking the trace or a lazily imported `"module:attribute"` path.
  `torchlens.appliances.names()` lists the names with their summaries.
- Refusals: `appliance_unknown` for an unknown name (it lists the registered names);
  `appliance_name_taken` for re-registering a name without `replace=True`;
  `appliance_registration_invalid` for a malformed name or factory.
- Naming: the word clashes with `torchlens.facets.adding(*facets)` (a recipe scope), so either
  spelling may still change.

**Safetensors Trace export** (experimental): `tl.export.safetensors(trace, path)` writes one pickle-free `.safetensors` file;
`tl.io.read_safetensors(path)` reads it back. Payloads are bit-equal, complex included.

- Tensor entries: every tensor-valued op payload (`out`, `transformed_out`, `grad`,
  `transformed_grad`).
- JSON header (schema `tl_trace_safetensors_v1`): the saved-format version, the
  saved-format digest, the capture accuracy facts, and, for the `trace` record and every
  `op` record, each saved (KEEP) field with an exact JSON encoding. The rest are
  listed under `omitted` with their value types.
- Reading back: a read-only record with `header`, `trace_fields`, `op_fields`, `tensors`,
  `omitted`.

**Bridges (`tl.bridge.*`)**: Optional-extra adapters to third-party tools (reference: `docs/bridges.md`).

- Captum: `tl.bridge.captum.layer(trace, site, *, pass_index=None)` gives Captum's layer
  methods a module for any site: the live module when a single-call module returns the site's
  tensor, otherwise a site probe that `tl.bridge.captum.attribute` splices into the forward at
  the op's `site_key`. `pass_index` (1-based) picks one call of a repeated module, the same as
  the `"address:N"` spelling, or one pass of a label naming several.
- Hugging Face Hub: published traces come back through
  `torchlens.bridge.huggingface_hub.from_hub` (its own entry), the reader for `push_to_hub`.

**Streaming statistics**: `tl.stats` is the streaming-statistics namespace: constant-memory accumulators built on the
`tl.stats.StreamingStat` protocol, a driver that streams a whole dataloader through them, and
one persistence path. Doc of record: [docs/reference/stats.md](stats.md).

- Accumulators: `tl.stats.Mean`, `tl.stats.Norm`, `tl.stats.Covariance`,
  `tl.stats.CrossCovariance`, `tl.stats.Quantile`, `tl.stats.TopK`, `tl.stats.PCA`,
  `tl.stats.CKA` (exact full-data linear CKA, never a minibatch approximation),
  `tl.stats.Histogram`, `tl.stats.RunningStats`.
- Persistence: `tl.stats.save_fitted` / `tl.stats.load_fitted` for fitted PCA and,
  experimental, for any accumulator's whole state (`tl_stats_state_v1`). A reload keeps
  folding as if never interrupted; `load_fitted(path, encoder=...)` takes back an `encoder=`
  callable, which is not stored.
- Aggregating: `tl.stats.aggregate(model, dataloader, metrics)` streams activations or
  gradients through those accumulators, keyed by site, across a whole dataloader without
  retaining per-batch tensors. Example:
  `metrics={"relu_3": tl.stats.Mean(), tl.grad("relu_3"): tl.stats.Mean()}`. A plain string
  key is an activation; a gradient key needs `loss_fn=`, else `aggregate_loss_fn_required`.
  Result and cache keys use the key's canonical repr.
- Caching (experimental): `cache=True` (with `cache_dir=`, as for `tl.trace`) keys the
  run by value (model contents and code, every batch, each accumulator's starting state and
  encoder, the metric keys, `loss_fn=`, versions), so a rerun restores the saved states and
  skips the loop. A run it cannot key raises `stats_aggregate_cache_unkeyable`.
- Per-unit statistics (experimental): the keyword-only `unit_axis=` on
  `tl.stats.Quantile` and `tl.stats.TopK` keeps one reservoir or top-k list per unit (neuron,
  channel or SAE latent). Per-unit `TopK` returns a `tl.stats.UnitTopK` (`values`,
  `example_ids`, `positions`, `unit_axis`), the one top-k-over-a-dataset tool for neurons and
  SAE latents, with ties broken toward the lowest example id, then position. `encoder=` maps
  each batch (for example through an SAE encoder) before it is folded;
  `TopK.update(..., example_ids=...)` names examples outside `tl.stats.aggregate`.
- Reduce and discard (experimental): `tl.trace(model, x, save="stats")` folds every
  op's out into a `tl.stats.RunningStats` during the forward and keeps no payload (one 16-slot
  float64 vector per op). `tl.stats.site_stats(trace)` returns `{op label: RunningStats}`;
  payload reads raise `activation_reduced_to_stats`, and replay or validation on the trace
  raises `stats_capture_not_replayable`.
- Trajectory statistics (experimental):
  `tl.mechinterp.measure_trajectory(model, data, *, statistics=None, facet="resid_post", layers=None, lens=None, mask_input=False, verify=True)`
  traces each batch and folds the logit-lens trajectory's `entropy`,
  `kl_to_final`, `max_prob` and, for `(input_ids, target_ids)` batches, `cross_entropy` and
  `target_rank` into a `tl.mechinterp.TrajectoryStats`.
- `tl.mechinterp.TrajectoryStats` fields: `position_means[statistic]` (a float64
  `[layers, positions]` tensor), `running_stats[statistic][layer]`, `summary(statistic)`,
  `examples` (the number of examples, or batch rows, folded), `validated` (whether the default
  lens was numerically validated) and `tie_convention` (how `target_rank` breaks ties).

## Quantities and typed errors

`moe_routing_activation_missing` is the `PayloadUnavailableError` code for a
routing-table read without saved router weights or expert indices. Its
`activation` field names the missing op; capture with `save='all'` to retain it.

**Quantities**: `tl.types.Quantity` is the marker base for numeric quantity wrappers with unit-aware
display: `tl.types.Bytes` (memory), `tl.types.Duration` (seconds), `tl.types.Flops`
(floating-point operations), `tl.types.Macs` (multiply-accumulates).

**FLOP convention (`flop_convention=`)**: How FLOPs are counted. Accepted by `summary(...)`; the removed `count_fma_as_two=` spelling
raises a typed error, naming it.

- `"fma2"` (default): one multiply-accumulate = 2 FLOPs.
- `"fma1"`: recounted from each op's two-term compute record; raises a typed error with
  `flop_convention_unavailable` when an op has no MAC split.

**Autograd-saved bytes**: `Op.cost.autograd_memory` / `Op.autograd.num_autograd_tensors` count the tensors the captured
forward's autograd saved for that op's backward (each storage once within the op; a view
counts in full).

- Totals: `Layer.cost.total_autograd_memory`, `ModuleCall.autograd_memory`,
  `Module.total_autograd_memory` and `Trace.cost.total_autograd_memory` sum those per-op
  bands, so a tensor two ops save counts twice.
- Frozen Parameters: capture never changes a Parameter's `requires_grad`, so on frozen
  Parameters these and `Trace.cost.pinned_autograd_memory` match what an eager forward saves.

**Module and call counts**: How to count the parts of a module, its calls and its parameters.

- `len(ModuleCall.ops)` counts the call's pass-qualified Ops (the root call included: a Layer
  it runs twice counts twice); `ModuleCall.num_layers` counts the distinct Layers among them.
- `len(Module.layer_labels)` counts a Module's distinct Layers and `len(Module.params)` the
  parameter tensors it owns.
- `Trace.num_buffer_layers` counts distinct buffer Layers, not their passes.
- `Param.num_calls` is the number of Ops that consumed the parameter (0 for a never-called
  module's parameters).
- `Trace.num_uncalled_modules` is recorded when the capture finishes, so it survives the
  source model being collected. It reads `None` when it cannot be known (a loaded or unpickled
  trace whose model is gone), never 0.

**Conditional depth**: `Op.control_flow.depth` and `Layer.control_flow.depth` count the distinct conditionals the
record takes a direct role in, as listed in `in_conditionals`. Nesting depth is
`len(conditional_branch_stack)`.

- The `evaluation` role marks the bool an arm's test consumed.
- The `body` role marks an op an arm's entry edge feeds (an op whose input was computed
  outside that arm), not every op inside the arm. An op nested in two `if` bodies takes a role
  only where an entry edge reaches it.

**Lookup and reentrancy errors**: Two typed errors for lookups and nested captures.

- `tl.errors.AmbiguousOpLookupError` (`ambiguous_op_lookup`): raised when a bare Op lookup
  matches multiple pass-qualified Ops, and when a typed entry point's key names several objects of
  its kind (`trace.ops["shared"]` for a module called twice lists `shared:1`, `shared:2`).
- `tl.errors.ReentrantTraceError`: raised when a trace is started while another trace is
  active.

**Report text helpers** (experimental): Helpers that produce shared report text and cross-checks.

- `tl.debug.amp_scaled_grads_hint(*, all_nonfinite=False)` returns the shared text that names
  a possible `torch.amp.GradScaler` loss scale behind large or non-finite gradients and the
  `grad_scale=` remedy; it edits no rows or verdicts.
- `tl.report.flops_vs_dispatch(model, inputs=None, input_kwargs=None, *, sdpa_backend=None, **trace_kwargs)`
  runs the detached
  two-way FLOP cross-check on one model and input and returns a `FlopsCrossCheck` holding both
  totals, the per-overload native counts and the coverage notes.

**Output-difference metrics**: Scalar distances between two outputs. Each returns a 0-d tensor; `relative_l1` and
`relative_l2` are anchored on `a`, and mismatched element counts raise a typed error.

- `tl.stats.cosine_distance(a, b)`: `1 - cosine_similarity` of the two flattened tensors.
- `tl.stats.relative_l2(a, b)`: `norm(a - b) / max(norm(a), eps)`.
- `tl.stats.relative_l1(a, b)`: `|a - b| / max(|a|, eps)`, the scalar fallback.

## Distributed capture and cross-rank merging

**Distributed arming**: `tl.distributed.arm()` opts a process into first-class capture of explicit
`torch.distributed` collectives (required at process start for MPMD / spawn-rank
programs; SPMD processes may arm lazily at capture entry). Sharded state (DTensor,
tensor/pipeline parallel) still raises a typed error at capture entry.

**tl.distributed.ArmingRecord**: What `arm()` established, stamped on every boundary record. Fields:
`ArmingRecord.install_epoch` (the `InstallEpoch` the arming installed under),
`ArmingRecord.recognizer_snapshot` (the recognizer snapshot id the arming derived) and
`ArmingRecord.source` (how arming happened: explicit or lazy at capture entry).

**tl.distributed.GroupIdentity**: The resolved lifetime identity of one process group on this rank. Fields:
`GroupIdentity.membership_digest`, `GroupIdentity.lifetime_ordinal` (which lifetime of
that membership this group is), `GroupIdentity.ordinal_source` (where the ordinal came
from), `GroupIdentity.global_ranks` (the member ranks) and `GroupIdentity.backend`; the
`GroupIdentity.group_uid` property is the exactly-two-field correlation group identity
(membership digest, lifetime ordinal).

**Collective boundary op**: A captured op whose portable `annotations["collective"]` carries the
`collective_boundary_v1` payload: correlation key, role-indexed dual geometry, event
notes, the fields that prove each match, and lifetime evidence.

**merge_ranks / MergedTrace**: `tl.merge_ranks([trace_or_path, ...])` stitches N rank-local captures into a
`MergedTrace` presenter (never a `Trace` subclass) at their explicit collective
boundaries; loads rerun the derivation and refuse tampered artifacts typed.

**merge_report**: `tl.merge_report(...)` is the graph-free merge diagnostic that derives alignment and
consistency verdicts without constructing a merged graph, and never raises on
conflicts.

**Merge vocabularies**: `MergeAlignment` and `MergeValueStatus` are frozen vocabularies in `tl.types`,
release-gated against
[merged traces](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/reference/merged_trace_contract.md).

**Trace.provenance.distributed_scope** (experimental): The shard-local capture marker, saved with the trace: `None` for ordinary captures,
`"rank_local_shard"` when any parameter or input the capture saw was one rank's local shard.
Such a trace is a rank-local recording, never whole-model truth. This name is
experimental and may change in any release.

- Load: the value is checked against its closed vocabulary.
- Refusals: saving a bundle raises `shard_local_persistence_unsupported` under any schema
  that would silently drop the marker (see the
  [error codes reference](error_refusal_contract.md)).

## Backends

Every backend, torch included, implements one interface, `tl.backends.BackendProtocol`. Its capability
table is declared in `BackendCapabilities` and measured by the shared conformance suite; the
measured result ships as a scorecard record per backend and shows in `tl.report.doctor()`. The
backends guide (`docs/backends.md`) carries the maturity table and links one page per backend
(`docs/backends/<name>.md`: a tutorial, a can and cannot table and the framework requirement).

**tl.backends.BackendProtocol**: The one interface every backend implements to produce traces (`typing.Protocol`, runtime-checkable).
It is reached through `BackendSpec.protocol()`; registration checks the returned object against
the member list. The older `CaptureBackend` orchestration adapter stays torch's internal
capture seam; it is not the backend interface.

- Identity members: `name` (the registered name: `"torch"`, `"tf"`, `"jax"`, `"mlx"`,
  `"paddle"`, `"tinygrad"`), `maturity`, `framework` (a `FrameworkRequirement`),
  `materialization`, `capabilities()` (the declared `BackendCapabilities`),
  `runtime_available()` (`None`, or the refusal code `runtime_missing` /
  `runtime_version_unsupported`; never raises).
- Capture members: `can_handle(model, args, kwargs)`,
  `capture(model, args, kwargs, **trace_options)` (returns a `Trace`; an unsupported option or
  capability raises a typed error before any framework call), `reference_output(model, args, kwargs)`
  (the framework's own forward with no TorchLens wrapper installed: the oracle's independent
  arm).
- Validation: `validate(trace, ground_truth)` returns a `ValidationStatus`, never a bare bool;
  a failure names the first offending op and a reason from `tl.backends.VALIDATION_REASONS`.
- Value members: `is_tensor(value)`, `host_copy(value)` (a detached NumPy copy with shape and
  dtype kept; 0-d stays 0-d), `realize_once(values)`.
- Declarations: `trace_options()` and `refusal_codes()` (a subset of
  `tl.backends.REFUSAL_CODES`).

**BackendSpec.protocol** and **BackendSpec.maturity**: `BackendSpec.protocol` is the lazy factory (`Callable[[], BackendProtocol] | None`) that returns
the backend's protocol object. `BackendSpec.maturity` is the backend's launch label, the single
source `tl.report.doctor()` and the docs read.

**maturity** (`"preview"` | `"stable"`): A backend's launch label, on `BackendSpec.maturity` and `BackendProtocol.maturity`. Torch is
`"stable"`; tf, jax, mlx, paddle and tinygrad are `"preview"`.

- Changing it: only a recorded maintainer decision changes it, as a one-line change to the
  backend's registration plus the docs table. No code computes it, and `graduation_report` is
  advisory.
- Rule: a backend registered `"stable"` must have every graduation item met in its shipped
  scorecard; `tests/test_backend_maturity.py` checks this.

**materialization** (`"eager"` | `"lazy"`): Whether a backend's op outputs are concrete values when the wrapped call returns (`"eager"`:
torch, tf, jax, paddle) or deferred until an evaluation (`"lazy"`: mlx, tinygrad).

- Run-once rule (lazy backends): the traced forward runs once to its outputs, then
  `realize_once(values)` evaluates every recorded value in one framework evaluation before any
  payload is read, so random-number consumption equals one eager forward.
- Random state: state created before the forward (a default-initialized weight) is realized in
  that same evaluation and recorded as the value it produced. Random numbers drawn inside the
  traced forward refuse with `rng_state_in_graph`, because replay would draw them again.
- On eager backends `realize_once` is a no-op.

**tl.backends.FrameworkRequirement**: What a backend needs from its framework, stated once (frozen dataclass). The supported range is
`[minimum, below)`. `tl.report.doctor()` shows the installed version, the pin and the range per
backend.

- `distributions`: installed distribution names that version the framework, in lookup order.
- `minimum`: the inclusive floor.
- `below`: the exclusive ceiling, or `None` when none is declared.
- `tested_through`: the highest version the backend's suite has run on; a preview sets `below`
  or this.
- `pinned`: the exact version the per-change CI leg installs, or `None` while the leg is
  unpinned.
- `extra`: the pyproject extra that installs the framework; `""` for a base dependency.
- Consistency: `pinned` lies inside the range, the extra's specifier equals it and the CI leg
  installs `pinned`; `tests/test_backend_framework_pins.py` checks this.

**tl.backends.REFUSAL_CODES** (backend refusal codes): The fixed list of codes a backend may raise. A refusal is a `BackendUnsupportedError`
whose `exc.code` is a member (match on the code, not on the message text). It raises
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
| `unwrapped_op_in_capture` | a wrapper backend met an output or intermediate with no captured producer and the caller asked for a refusal; without that request the trace is marked `provenance.verified=False` with this code as `provenance.verification_reason` and a warning |
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

**tl.backends.VALIDATION_REASONS** (validation decision reasons): The closed list of per-op decision reasons a backend's `validate` uses in
`ValidationStatus.decisions[*]["reason"]` and the status reason counts. One word per failure: a
backend whose cause matches a definition uses that name, and a different cause has its own
entry. The suite checks membership, so the list may be a superset of what one backend emits.

- Shared, with meanings: `replay_mismatch` (re-executing the op from its saved inputs did not
  reproduce its saved output), `parent_payload_missing` (a payload the replay needs is absent:
  the op's own saved output or a parent's), `no_replayable_function` (no callable can
  re-execute the op: none was recorded for a computational op, or a preview backend could not
  rebuild or re-invoke the saved call).
- Shared, explained below: `constant_producer_leaf`, `constant_parent_payload`,
  `user_intervention_replacement`, `effectful_region`, `perturbation_insensitive`,
  `dtype_tolerance_exceeded`, `op_unclassified`, `capture_inconsistent`.
- Torch's own: `arg_logging_mismatch` (the logged parent-argument positions disagree with the
  data parents, or a parent's saved output disagrees with the argument snapshot at its logged
  position), `missing_saved_args` (the argument snapshots replay needs were not captured),
  `dispatch_op_count_mismatch` (replay saw a different number of dispatcher-level ops than
  capture recorded), `bfs_incomplete` (the walk from outputs back to inputs did not reach every
  op), `metadata_invariant_exception` (a metadata invariant check raised).
- Torch's own, continued: `replay_execution_exception` (the recorded function raised on
  re-execution), `unattributed_tensor_arg` (a tensor argument matches no recorded parent and no
  known source), `dropped_parent_edge_witness` (the completeness check shows a parent edge
  capture dropped).
- MLX's own: `declared_parents_inconsistent` (an op's declared parents disagree with the arrays
  its saved call reads), `replay_coverage_incomplete` (the replay did not cover every captured
  op).
- `op_unclassified`: an op the checker cannot classify (outside every classification table, or
  an initializer that contaminated the forward), recorded as a failed `metadata`-phase decision
  on that op.
- `capture_inconsistent`: capture records that disagree with each other (a callback
  self-consistency failure, an equation or op count mismatch, a replay kind that differs from
  the captured kind, a region boundary check), recorded as a failed `metadata`-phase decision
  on the op or, for a trace-wide disagreement, on `"<trace>"`.
- `constant_producer_leaf`: replay reads a constant's saved payload because there is no
  function to re-execute it from (counted as exempted; a parentless op that does have a
  replayable function is still replayed and a mismatch still fails).
- `constant_parent_payload`: perturbation skips a constant parent, and the consumer is still
  perturbed on its other parents.

**`constant` op (closure constants and literal producers)**: A tensor the forward used that is neither an input, a parameter or buffer of the model, nor the
output of a recorded op (a closure tensor built outside the traced call) is recorded as a
source op with `func_name="constant"`, no parents and a saved payload; the op that read it lists
it as a parent.

- Labels follow the backend's label convention (`constant_<n>` on the previews; torch numbers
  it like its other source ops).
- Literal producers: an op with no tensor parents whose arguments are all literals (a `fill` or
  `ones` inside the forward) keeps its own `func_name`, has zero parents and a saved payload.
- Payloads: constant payloads count toward the payload budget and `save=` applies.
- Validation treats a constant as a leaf (reasons `constant_producer_leaf` and
  `constant_parent_payload`); a constant with no readable buffer raises
  `constant_unrealizable`.

**`backend_scorecard:<name>` (doctor row)**: `tl.report.doctor()` carries one row per registered backend, key `backend_scorecard:<name>`,
label `"<name> backend scorecard"`, built from the scorecard record shipped in the backend's
package (no framework is imported and nothing executes).

- Statuses: `pass` when the record's highest tier is the durable tier with no failed cell,
  `known_broken` when any cell failed, `scope` when the framework is not installed here,
  `not_tested` when no record ships.
- Details read:
  `"<maturity>; <passed>/<denominator> cells passed, <unsupported> unsupported, <failed> failed; pinned <pinned>, installed <version or absent>, supported [<minimum>, <below>)"`.
- Suggestion: names the install extra when the framework is absent, otherwise the first failed
  cell.

**`HAS_MODULE_CALL_DISPATCH_FRAME` (torch capability flag)**: True when TorchLens can tell a direct `module.forward(x)` call from a normal `module(x)` call on
this torch, measured by a behavioural probe (never a version parse). Shown in the torch
capability snapshot that `tl.report.doctor()` reports. When it is False no call is classed
direct, so the fast run keeps refusing a hook-invisible module call as
`fast_live_module_hook_unobserved`.

**`python -m tests.conformance.scorecard --backend <name>`**: Computes the backend's scorecard.

- `--write` writes the shipped record (`torchlens/backends/<name>/scorecard.json`, in the
  wheel).
- `--check` recomputes and exits nonzero when the shipped record differs (the backend's pinned
  CI leg runs it, so the record is a tested fact, never hand-edited).
- `--graduation` prints the graduation report from the shipped record.

**Capability summary** (`supported` | `unsupported` | `untested`): The scorecard's per-flag reading of `BackendCapabilities`. Each backend page's can and cannot
table equals this summary and renders `untested` as "not yet proven"; the check is
`scripts/checks/backend_docs_lockstep.py`.

- `supported`: the flag is `True` and the suite consumed it.
- `unsupported`: the flag is `False` and its `c1_refusal_<flag>` cell passed (the request
  raises its documented error code).
- `untested`: the flag is `False` and the refusal cell failed or did not run.

**Conformance case ids added by the protocol**: The conformance cells the backend protocol adds, by tier. A typed refusal inside any other cell
makes it `unsupported`; an untyped exception, a wrong value or a trace marked
`provenance.verified=False` makes it `failed`.

- Tier 0, the provider API (no model execution): `c0_protocol_complete`,
  `c0_framework_requirement_consistent`, `c0_refusal_codes_registered`,
  `c0_capability_table_complete`.
- Tier 1 (capture): `c1_graph_integrity`, `c1_intermediates_match`,
  `c1_shape_metadata_matches_payload`, `c1_validation_passes`, `c1_validation_names_reason`,
  `c1_constant_parents_recorded`, `c1_rng_consumed_once`, `c1_no_silent_gap`.
- Tier 1, per backend: one `c1_refusal_<flag>` per capability flag the backend sets `False`
  and one `c1_refusal_trace_option_<name>` per public trace option outside its
  `trace_options()`.
- Tier 2 (durable): `c2_payload_values_equal`, `c2_validation_after_load`.

**Per-backend namespaces**: `tl.backends.torch` is the stable PyTorch eager backend (`docs/backends/torch.md`); the other
five are preview backends.

- `tl.backends.tf` (`GradOptions`, `TFBackend`; `docs/backends/tf.md`).
- `tl.backends.jax` (`GradOptions`, `JAXBackend`; `docs/backends/jax.md`).
- `tl.backends.mlx` (`GradOptions`, `MLXBackend`; `docs/backends/mlx.md`).
- `tl.backends.paddle` (`GradOptions`, `PaddleBackend`; `docs/backends/paddle.md`).
- `tl.backends.tinygrad` (`GradOptions`, `TinygradBackend`; `docs/backends/tinygrad.md`).
- Each `GradOptions` requests the backend's derived gradients
  (`trace.autograd.derived_grads`, opt-in intermediate derived gradients); each
  `<Name>Backend` is the backend's capture object behind `tl.trace(..., backend="<name>")`.

## Experimental parts

These parts of TorchLens are experimental: they work and are tested, but their names and outputs
may change in any release without notice.

| What | Names |
| --- | --- |
| Selections: pick units, ops or sites by value, statistics, graph position, change across runs or passes, or a subspace | `tl.top_k`, `tl.top_fraction`, `tl.threshold`, `tl.sign`, `tl.dead`, `tl.saturated`, `tl.low_variance`, `tl.neighborhood`, `tl.between`, `tl.changed`, `tl.top_changed`, `tl.stable_across_passes`, `tl.pass_variance`, `tl.subspace`, `tl.units`, `tl.param`, `tl.random_like`, `tl.Selector`, `tl.Selection`, `trace.between`, `trace.subgraph` |
| Edits over selections, patching from another trace, and edge edits | `tl.Edit`, `tl.patch_from`, `trace.graph.edges` |
| Structure-only capture, grouping options and non-finite tracking | `structure_only=`, `grouping=`, `on_nonfinite="track"`, `Trace.discharge_against`, `Trace.nonfinite_ops`, `Trace.nonfinite_coverage` |
| Drawing options | `color_by=`, `size_by=`, `size_scale=`, `view=`, `stack_by=`, `show_redundant_args=`, `show_arg_labels=`, `file_format=`, `theme=`, `show_legend=`, `tl.viz.contact_sheet` |
| Sparse autoencoders | `tl.bridge.sae.splice`, `tl.bridge.sae.SpliceResult`, `tl.bridge.sae.register`, `tl.bridge.sae.unregister`, `tl.bridge.sae.write_delphi`, `tl.bridge.sae.write_sae_dashboard`, `tl.bridge.sae_lens.reconstruct` |
| Representation similarity and brain scores | `tl.stats.SimilarityScore`, `tl.repgeom.ScoreInterval`, `tl.repgeom.metric_disagreement`, `tl.bridge.brain_score.ScoreParts`, `tl.neuro.compare` |
| Input preprocessing and model cards | `torchlens.preprocessing`, `torchlens.postprocessing` |
| Integrations | `torchlens.bridge.pyreft.LoreftArtifact`, `torchlens.trackers`, `tl.export.xarray`, `tl.bridge.transformers` |
| Backward timing and backward FLOPs | `trace.autograd.grad_fn_fire_timings`, `BackwardPass.aten_ops`, `BackwardPass.flops_attribution` |
| Reports and summaries | `Trace.sites()`, `Trace.bill_of_materials()`, `summary(columns=...)`, `tl.report.Report`, `ProfileReport.evidence_labels`, `measure_host_peak_memory` |
| Attribution graphs | `tl.attribution.build_graph`, `tl.attribution.upload_to_neuronpedia`, `AttributionGraph.to_neuronpedia` |
| ATen operations and GPU kernels | `aten_ops=`, `AtenOp`, `KernelLaunch` |
