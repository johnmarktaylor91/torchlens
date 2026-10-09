## Current 2.x Surface

- Top-level `torchlens.__all__` has 115 names: capture, save/load, intervention,
  selectors, helper transforms, observers, validation, and the three main log classes.
- Fast paths (doc of record [docs/guides/fast_paths.md](../guides/fast_paths.md)):
  `spec.bind(model)` (`tl.when(site, action).bind(model)`) is the capture-free bound executor,
  about 1.0 to 1.2x a plain forward hook and exact, also through HF `generate()` with the KV
  cache (`bound.generate(...)`, or `torchlens.intervention.steer_generate(model, ids, spec, ...)`).
  `tl.record(..., intervene=spec, return_output=True)` is the lighter evidence path (about
  1 ms per op against about 3 ms for `tl.trace` on the tested CPU decoders). A pure
  `tl.module(...)` save selector (or a `|` union of them) is live-retained at about the cost of
  saving everything; mixing it with other terms escrows every op and reports deferred in
  `torchlens.capture.preflight.address_preflight`. Trace-then-rerun on a module-targeted staged
  spec takes the guarded fast rerun (about 2x a plain hook per generation step, exact, including
  a stock HF forward that returns its KV cache; `last_run["fast_refused"]` names a fallback's
  reason).
- Relation accessors on FINISHED traces return IMMUTABLE views (authorized public type
  break, decided 2026-08-12): label sequences (`op.parents`, `op.children`, `op.modules`,
  `op.module_call_stack`, conditional child lists, `Layer.parents`/`Layer.children`, ...)
  are `tuple`; label sets (`input_ancestors`, `output_descendants`, `root_ancestors`,
  `internal_source_ancestors`) are `frozenset`. Reads are identity-stable, in-place
  mutation (`op.children.append(...)`) raises, direct assignment still works on `Op`
  records (a raw `list`/`set` normalizes to the view type) — but NOT on `Layer`:
  `Layer.parents`/`Layer.children` are read-only properties and assignment raises
  `AttributeError` (and `log[label]` returns a `Layer`). Equal views may be shared across
  records. Dict-shaped relation metadata (`parent_arg_positions`,
  `conditional_elif_children`, `module_entry_arg_keys`, ...) keeps its mutable dict
  type. Legacy saves load with the same immutable surface. `equivalent_ops`/
  `recurrent_ops` are LIVE group-membership views (`frozenset`/`tuple`): every member
  of a group reads THE one cached immutable view (O(1), identity-stable), removal
  scrub rebinds the group row once for all members, and caller mutation is
  impossible — this supersedes the historical fresh-mutable-copy-per-read barrier.
- Steering many forwards or a generation loop: `spec.bind(model)` (capture-free, about 1x a
  plain hook, works with HF `generate()` and the KV cache), `tl.record(..., intervene=spec,
  return_output=True)` per forward when activations are needed as evidence, and one full
  `tl.trace(..., intervene=spec)` as the correctness oracle. Trace-then-rerun
  (`trace.run(model, x)` on a module-targeted staged spec) is a fast path too: the guarded
  fast engine, about 1.5x a plain hook, with the trace's saved sites refreshed. Recipe:
  [Common Patterns](common-patterns.md), "Steering many forwards / generation".
- `tl.record(..., save=...)` is the sparse predicate recorder; it returns `Recording`.
  `Recording.to_trace()` cooks the event stream into a full-structure `Trace`, with unsaved
  payload reads rejected explicitly. `tl.record()`/fastlog is torch-only in the backend-v1
  registry. The old `keep_op=`/`keep_module=` alias kwargs are removed; `save=` is the only
  predicate spelling. Failed forwards default to
  the historical `on_forward_error="raise"` behavior; opt into
  `on_forward_error="attach_partial"` to attach `exc.partial_recording` and re-raise, or
  `on_forward_error="return_partial"` to return a failed partial `Recording`. Failed partials
  set `status="partial_error"`, `failed=True`, string-only error metadata, `n_ops_completed`,
  and best-effort `last_event_*` fields. user-op failures exclude the failing call; TL-side
  capture failures may include a skipped/partial current-call event. Failed partials cannot be
  converted with `Recording.to_trace()` or used with `Recording.log_backward()`.
  Trace-side failed captures expose `exc.partial_log`, recoverable with
  `tl.partial.from_failed_capture(exc)`.
- **Every capture product carries ONE settled typed outcome** (early-stopping
  unification; doc of record `docs/reference/capture_outcomes.md`).
  `Trace.outcome` / `Recording.outcome` / `PartialTrace.outcome` return a frozen
  `CaptureOutcome` (`tl.types.{CaptureOutcome, CaptureStatus, CapturePhase,
  FailureOrigin}`): status one of COMPLETE / HALTED / ABORTED_NONFINITE / FAILED
  (with FORWARD/FINALIZE/POSTPROCESS/TEARDOWN phase) / UNATTESTED (legacy
  finished artifacts without attestation — never blessed COMPLETE) / UNKNOWN
  (unprovable, most restrictive). The outcome PERSISTS (`_capture_outcome`,
  tlspec v7) as a string-only payload validated at load against closed
  vocabularies and a status coherence matrix; incoherent/forged attestations
  degrade to UNKNOWN with a warning. Capability gates run through one
  chokepoint with stable codes on `tl.errors.CaptureOutcomeError.fields["code"]`:
  N1 (failed/aborted/unknown exports refuse), N2 (validation ENTRY refusal only
  — tripwire bodies untouched, UNATTESTED enters), N3 (replay/backward), N4
  (halted runnable save, `RunnableErrorCode.HALTED_CAPTURE_NOT_RUNNABLE`), N5
  (halted LIVE-provider replay refuses — loaded-sparse `run()` stays allowed).
  A swallowed halt/nonfinite signal (broad `except:` in user code) raises
  `tl.errors.StopSignalSwallowedError` at the capture boundary and settles
  FAILED, never COMPLETE. Refresh re-arms `raise_on_nan`. Halted analysis
  `tl.save` works (the transient-leak refusal is fixed); halted `log_backward`
  and loaded-sparse `run()` remain allowed.
- QUERYABLE NONFINITE RECORD (spellings DOCUMENTED-UNSTABLE): `trace.nonfinite_ops`
  returns the pass-qualified labels of ops whose output held NaN/Inf and
  `trace.nonfinite_coverage` the frozen evidence disclosure (basis + checked/
  unchecked/unexamined counts — read it before trusting an empty answer). On
  default captures the record derives from the memoized saved-payload scan at
  ZERO capture cost; `CaptureOptions(track_nonfinite=True)` opts into
  capture-time per-op checks that also cover unsaved ops (selective-save
  captures), costing ~3-15% of capture time on CPU and deferring device flag
  reads to ONE batch at the finalize seam (never a per-op CUDA sync).
  `raise_on_nan` stop-and-throw is unchanged and independent; `structure_only`
  refuses the combination typed. Session-time knob (`FieldPolicy.DROP`): load
  restores the default and loaded traces serve the saved-payload basis. Doc:
  `docs/reference/capture_outcomes.md` ("The queryable nonfinite record").
- QUICKSTART INPUT LADDER (F17; every quickstart spelling DOCUMENTED-UNSTABLE pending the
  naming sprint): `tl.trace`/`tl.summary`/the render facade share ONE input resolver --
  a real input (gold rung) XOR `input_size=` (declared shape, seed-0 local synthesis,
  fail-closed static dtype facts with `torchlens.quickstart.InputSpec` as the override;
  flat tuple / sequence of tuples / forward-keyword-to-shape mapping, which reaches
  multi-input models like CLIP today) XOR nothing (inference verifies once and the
  surface consumes the EXACT verified trace; non-default capture kwargs recapture with
  disclosure). Mixing refuses `input_rung_conflict`. Synthesized rungs persist an
  `InputProvenance` payload inside the existing `Trace.input_preprocessor` KEEP field
  (`torchlens.quickstart.trace_input_provenance` re-hydrates it; absent payload reads as
  gold); derived-semantics claims (`decode_output`/`output_table`) refuse
  `nongold_semantics_unavailable` on synthesized values and the first raw `Layer.out`
  read warns once per trace (`nongold_raw_value_read`, internal readers ack via
  `torchlens.quickstart.internal_read()`). `render(model, ...)` (home
  `torchlens.user_funcs.render`; `tl.render` root spelling awaits the F35 registration
  sweep) is a curated facade over the resolver + `Trace.draw`: metadata-only pinned
  eval/no-grad capture with flags/RNG/norm-buffers restored and state-dict-hash
  verified, `collapse="auto"` as its default (by design: THERE ONLY -- `Trace.draw`
  keeps `"none"`), detached `RenderResult` (owns DOT/bytes/path/provenance/receipt,
  `save()` re-renders without the trace), never auto-opens a viewer, collision-safe
  `<ModelClass>-graph.<fmt>` default filename in scripts (fork F1 branch A constant).
  Lazy completion: gold/declared rungs materialize executed lazy modules during the ONE
  captured forward with truthful totals (the metadata-invariant flip signal passes);
  zero-arg refuses before probing; armed captures refuse `state_baseline_unavailable`.
  Doc of record: `docs/quickstart.md`.
- BOUND-METHOD ROOTS (F41, by design; spellings DOCUMENTED-UNSTABLE): `tl.trace`
  (and `tl.validate`) accept an `nn.Module` OR a bound method of one — `tl.trace(model.generate,
  ids, ...)`. The owner resolves via `method.__self__` and registers as the `owner` submodule of
  a TL-authored wrapper root (`torchlens/backends/torch/bound_root.py`) whose forward calls the
  exact bound method EXACTLY ONCE; module addresses read `owner.*`. The synthetic root reads
  `type(owner).__name__` (never "method") for `model_class_name`/`model_class_qualname`, writes
  `root_entry_point="bound_method:<owner_qualname>.<method>"`, and marks the root op record
  (output-boundary ops) `tl_authored_root=True` (both persisted, fail-closed load validation).
  On bound-method episode captures `EpisodeSpec.stepped_module` DEFAULTS to the owner (`None` on
  a module root refuses typed). The entry-point fact is joined to the ONE rerun/append identity
  gate FAIL-CLOSED: an absent fact refuses `root_entry_point_unavailable` (legacy artifacts) and
  a non-`module_call` root refuses `rerun_entry_point_unsupported` — interim posture per foldA
  D11 (supplying `owner` for a capture of `owner.generate` would run `__call__`, a DIFFERENT
  entry point, under a fidelity-claiming report), never widened past the ruling; replay engines
  are unaffected. Closures/bare functions refuse `model_type_unsupported` teaching the ruled
  spelling.
- EPISODE CAPTURE (torch-only; every spelling DOCUMENTED-UNSTABLE): one wrapped multi-step
  generation run is ONE product — `tl.trace(episode_root, x,
  episode=tl.options.EpisodeSpec(stepped_module=model, n_steps=N))` stamps
  `capture_kind=episode` and lands the per-step status ledger (GRAMMAR v2 as of the
  C07X amendment, `episode_ledger_version=2` family-local: header carries the declared
  `step_output_kind` tokens|digest|none + the `step_output_from` source disclosure /
  `step_axis`, the minted `capture_digest` binding (F40b), the MEASURED `step_join`
  envelope (F40c), and on coupled captures the `intervention_digest` (F42); rows carry
  complete/interrupted/absent status, the generic `step_output` from the root output,
  the measured carried-state witness slots `entry_state_digest`/`exit_state_digest`
  (channel-keyed, None = NOT MEASURED; the arithmetic `cache_len` is DELETED), and the
  measured `fire_count` on coupled captures (a first-class 0 on zero-fire started
  rows; None = uncoupled/never-started); managed-RNG entry_seed) at `trace.annotations["episode"]`
  after settlement. EVIDENCE DERIVATION IS DECLARATION-DRIVEN (F40b, foldA D8):
  tokens/digest/none per the declared kind, `step_output_from` resolves
  dict/ModelOutput/tuple root slots by container path, and per-step positions are
  TAIL-ALIGNED along `step_axis` (last n_steps positions) — real `generate()`
  prompt+completion shapes and float roots (under digest/none) capture instead of
  refusing; declaration mismatches refuse teaching (`episode_declaration_invalid`);
  `step_output_kind="none"` admits value-free save policies. The family version is
  CONSUMED at load: a grammar v1 payload (`tokens`/`cache_len` rows) or any foreign
  version quarantines typed, never normalizes. CROSS-STEP JOINS ARE MEASURED (F40c,
  live boundary hooks under pause_logging + exact settlement re-grade): per-row grades
  continuous/forced/transformed/declared/exogenous/unchecked in the `episode_step_join_v1`
  header envelope; the DEFAULT arm returns the one Trace with the break marked and
  refuses episode-dependent claims across a measured break (step-series reads via
  `EpisodeLedger.step_output_series()`, whole-episode `run()`, the blessing fold,
  escalation — `episode_feed_break_exogenous`/`episode_join_declared_crossing`; ops/
  values/graph/per-segment reads unaffected); `EpisodeSpec(on_feed_break="refuse")` is
  the built FORK-4 arm (b) (settlement raises typed with the product on
  `exc.partial_log`); `EpisodeSpec(feed="closed")` is the strict arm (undeclared
  crossing halts at the NEXT step entry — one-step detection latency —
  `episode_feed_closed_violation`; declared crossing stops before entry,
  `episode_declared_crossing_stop`); `EpisodeSpec(crossings=(k,...))` declares
  tool-call-shaped exogenous entries (grade `declared`, disclosed, still
  chain-shaped). The claim is PER-ARTIFACT: an envelope-less ledger reads
  step_join=unmeasured everywhere and series claims refuse `episode_join_unmeasured`
  — the build switch never upgrades an artifact's own evidence. The ledger is a
  DISCLOSURE, never a settlement authority (outcome vocabulary and N1-N5 unchanged);
  loads validate fail-closed (illegal attachment refuses `episode_ledger_without_declaration`,
  geometry violations quarantine `episode_ledger_incoherent`); the annotations key AND the
  Bundle `member_relations` key persist plainly as of the tlspec v8 coordinated bump.
  DIAGNOSTIC-TIER cost, superlinear (gpt2-124M CPU: N=20 79 s / N=100 657 s, 947 MB, 5.4 GB
  RSS) — tens of steps, never hundreds; guarded-fast (`trace.run(fast=True)`, which needs a
  functional `save=tl.func(...)` on the capture -- the default capture re-runs through
  `trace.run(inputs=...)`) is the engine for episode-scale re-runs and must reproduce wrapped
  tokens bit-exactly (pinned). The same engine serves a STEERED rerun: `trace.run(model, x)`
  on a module-targeted staged spec fires the staged hooks at module exit inside a native
  forward (`last_run["engine"] == "guarded_fast"`), admits a changed input length under the
  sealed ordered call fingerprint (`last_run["shape_varied"]`, unrefreshed shape/memory
  metadata reset to `None`), and falls back to the capture engine with
  `last_run["fast_refused"] == "<code>:<stage>"` on any typed refusal (value replacements,
  non-module targets, structural divergence); `fast=True` on a live or loaded-activation
  trace applies the staged spec the same way. Glossary: "Guarded fast rerun". ATTESTED COUPLING
  (F42, the foldA D5 flip; spellings DOCUMENTED-UNSTABLE): `episode=` x `intervene=`
  runs COUPLED — a fire-attribution session attributes every live FireRecord to its
  step (root-loop fires bucket outside-step, disclosed in the digest, never guessed
  into a row), settlement writes per-row `fire_count` + the deterministic header
  `intervention_digest` (`episode_intervention_digest_v1`; identical re-runs mint
  identical digests — compare digests, never output equality alone) +
  `fidelity_basis="perturbed"` when a fired edit replaced a value (outranks
  forced/escalation bases). `torchlens.intervention.at_step(*steps)` is the
  step-qualified selector (live via the armed join session; post hoc via the persisted
  `Op.episode_step` stamps written at settlement; composes with pass-qualified labels;
  no-steps-to-qualify doors refuse `episode_step_selector_without_episode` /
  `episode_step_unstamped` / `episode_step_selector_invalid`).
  `trace.episode_coupling` CONSUMES the capture digest (recompute-and-compare; mismatch
  refuses `episode_coupling_unbound`, pre-binding artifacts
  `episode_coupling_unmintable`) and serves per-segment facts that never span a
  measured break join. Replay derives a fresh ledger or refuses: `run()` on a coupled
  product refuses `episode_coupled_replay_underivable` on BOTH engines (re-capture is
  the fresh-ledger arm), and `do()` edits quarantine inherited episode evidence
  (`episode_evidence_dropped_perturbed_replay`). Teacher forcing
  (`forced_tokens=`) is a disclosed NON-VERIFYING mode; escalation re-runs the WHOLE episode
  wrapped with `escalated_from`/`reason`/`fidelity_basis` disclosed (E-A3: mismatch records
  `diverged`, never a settlement input); declared unsnapshotable state refuses at declaration
  time (`episode_state_unsnapshotable`). Bundles gain the optional S6 member-relation table
  (`member_relations=`, `Bundle.relate`, `Bundle.derive_episode_status` — a derived fold,
  never Bundle-level settlement; mutators cascade explicitly or refuse typed). Doc of record:
  `docs/reference/episode_capture.md`.
- EXPERIMENTS REMEMBER (F03; every spelling DOCUMENTED-UNSTABLE pending
  naming-session ratification; doc of record `docs/reference/experiment_ledger.md`):
  the Bundle is the MATERIAL record — persisted random `bundle_id` (never a
  content hash), per-member `member_construction` origin anchors (closed
  constructed/added/forked/varied/swept/loaded vocabulary), and the
  hash-chained `operations` chronology (`bundle_operation_v1`) all ride
  `bundle.json` fail-closed (`bundle_lineage_invalid`); `Bundle.fork()`
  carries relations + preserved sections and anchors the source id on BOTH
  containers. `bundle.why(member)` is the provenance join — a pure derived
  ordered event-id walk with four honesty axes (lineage exact/partial/
  diverged/unrelated/unattested with the evidence basis disclosed; payload
  fidelity declared/opaque; value residual where identical recorded chains
  with differing outputs read `unexplained`; comparability) — additive
  wording only on an empty reference suffix; `bundle.provenance()` is the
  per-member view. `bundle.vary(mapping)` = one explicit spec or None
  identity PER member (do broadcasts; vary names), complete coverage by
  default, whole-mapping donor normalization, no-rollback partial outcomes
  (`vary_partial_failure` + `material_action_completed`).
  `tl.sweep(include_baseline=True)` mints the PRISTINE baseline member (the
  legacy default discloses `sweep_baseline_absent` at construction).
  `torchlens.experiment.site_sweep(baseline, candidates=, edit=, metric=,
  retain=, engine="live_hook", model=, x=)` is the ONE candidate engine
  (serial, extract-before-cleanup, rolling `top_k` retention, byte
  preflight via `retention_projection_over_budget`); candidates execute in
  site-key coordinates and undeclared multi-site candidates (incl. the
  unscoped `tl.head(i)` broadcast) become REFUSED effect rows; the complete
  per-candidate `member_effect_table_v1` (released/refused/failed included)
  persists and `bundle.effects()`/`measure_members()` serve it (member-axis
  `most_changed`, explicit session-only tier-c `.selection()`).
  `head_ablation_candidates` is the module-scoped v-facet sugar (GQA/MQA
  refuses typed; hand-written-hook oracle beside it). The experiment LEDGER
  (`torchlens.experiment.ledger(path, hypothesis=, metric=)`) is the opt-in
  SEMANTIC record: ContextVar-armed, single-writer, append-only hash-chained
  JSONL with per-event fsync (kill -9 recovers exactly k events, torn tail
  disclosed, interior tamper refuses), three-layer honesty (actor-stamped
  interpretive fields, `declared_at_seq`, verdict basis or "asserted; no
  recorded basis", closed supports/refutes/inconclusive/not_assessed
  vocabulary, lint-never-block), ONE material step per top-level operation,
  quarantine-not-raise on post-material write failure (the material result
  returns unharmed; `on_record_error="raise"` opts out), and read-only
  serving via `ledger_overview`/`ledger_entry`/`ledger_evidence` (MCP:
  `torchlens_ledger_*`) over the mid-experiment-fresh artifact.
- CHECKPOINT LIVE-REF GUARD (A-CKPT; spellings DOCUMENTED-UNSTABLE): parameter reads resolve
  through LIVE model handles (or nothing after deserialization), never capture-time bytes, so
  EVERY cross-member parameter value/difference/trajectory read on a Bundle (`SuperParam` views
  with >= 2 members: `weight_norm_diff`, `diff_pair`, `aggregate`, `out`/`grad`) refuses BEFORE
  tensor lookup with stable code `checkpoint_series_live_params` — keyed on the CLAIM, never
  Python object identity (save/load manufactures relationship-rank upgrades); fires at the ONE
  `_TensorBearing._tensor_dict` funnel and survives save/load, replacing the historical silent
  degradation (false-zero `weight_norm_diff`, "identical" `diff_pair`, all-zero `aggregate`
  live; NaN/empty/None post-load). Beside it, `Param.value_basis` is a derived read-time
  persisted-nowhere disclosure: `live_ref` / `absent(not_persisted)` (typed replacement for the
  bare post-load `None`); `snapshot` arrives only with R8(b) capture-time snapshots and is the
  only basis the guard passes. `Param.value` keeps its documented live-handle contract;
  version-axis relation rows still ORDER members; single-member views keep the live read.
- `tl.trace(..., backend=None)` routes through `BackendSpec`; explicit backend mismatches,
  unknown names, unsupported capabilities, and audit-only payload reads raise typed backend
  errors. Public backend-neutral metadata lives on `Trace.backend`, `Trace.module_identity_mode`,
  `Trace.param_source`, and record fields such as `dtype_ref`, `device_ref`,
  `backend_address`, and `resolver_status`.
- TensorFlow is available as `backend="tf"` / `backend="tensorflow"` for the Keras-3 / TF>=2.16
  preview when `keras.backend.backend() == "tensorflow"`. The shipped path is eager live capture
  with `op_callbacks` as the primary mechanism: real values, real taken-branch control flow,
  op-level records, and Keras/`tf.Module` module stacks. The graph-only FuncGraph static path is
  implemented for compiled/SavedModel entries (opaque regions stay honestly unverified).
  Derived gradients (leaf + exact T1 intermediates) ship for eager entries via
  `tl.backends.tf.GradOptions` — one GradientTape replay with divergence refusal; graph-only
  captures refuse `grad_options` typed. Static-label `intervene=` ships for eager entries
  through a two-level writable layer (module-boundary substitution + curated tf.nn/tf.math
  functional wrap) with FAIL-CLOSED site reachability — a selector matching callback-captured
  ops the wrap layer never saw refuses typed. `halt=`/`recipes=`, true backward capture, and
  value-dependent predicates remain deferred like sibling preview gaps.
- The Paddle preview supports live forward `trace(intervene=...)` and `trace(halt=...)` (eager
  dygraph: the wrapper holds each concrete output before the caller sees it). The intervened op
  keeps its real identity with the replacement payload plus hook-minted `FireRecord`s; the replay
  oracle uses a narrow corroborated user-intervention carve-out, so a replacement value presented
  as captured-native FAILS validation. Builtin helper adapters: `zero_ablate`, `scale`, `add`,
  `replace_with` (others refuse typed); decisions are forward-only; intervention/halt predicates
  may be value-dependent (real `tensor_requires_grad`/`is_scalar_bool`/`bool_value` in the ctx);
  `recipes=` attaches facet recipes; `grad_options` cannot combine with `intervene=`/`halt=`.
  Value-dependent `save=`, streaming, fastlog, and rng_replay stay refused on paddle.
- All four eager previews (tf/mlx/tinygrad/paddle) group recurrent calls into multi-pass layers
  through the same neutral grouper torch and JAX use (`layer_label:pass` labels, `pass_index`/
  `num_passes`, `recurrent_ops`; `recurrence_detection=False` opts back into the historical
  single-pass layout). The stored `recurrence_detection` flag is the EFFECTIVE value — the TF
  static FuncGraph path stays ungrouped and keeps `False`. Validation sidecars stay keyed to raw
  capture identities and oracles compare in raw-label space; per-backend tamper tests prove a
  stale-label sidecar FAILS validation rather than silently passing.
- `Trace.draw(order_siblings=True)` is the default Graphviz sibling-ordering pass for
  forward unrolled graphs; set it to `False` to render the raw dot layout.
- `Trace.draw(color_by=...)` (UNSTABLE spelling, keyword-only, no deprecation shim owed until
  the naming session ratifies it) is the v1 encoding channel: a record field name, scalar
  builtin (`time`/`flops`/`bytes`/`magnitude`/`grad_norm`), or callable `node -> value` fills
  eligible op nodes from a colorblind-safe sequential ramp (linear min-max, legend-disclosed).
  Channels are dot-layout-only (AUTO forces dot with a notice; explicit `layout="rank"` refuses
  `encoding_requires_dot_layout`) and presentation-only (collapse plan and Trace untouched).
  On rolled multi-pass layers, field sources resolve through the name-keyed rolled-aggregate
  allowlist in `torchlens/visualization/_encoding.py`: marker-varying and mirrored
  first-pass-only numerics (`raw_index`, `step_index`, `ordinal_index`, `grad_fn_object_id`,
  `buffer_pass`, `transformed_gradient_memory`, `conditional_depth`) stay UNENCODED with a
  legend note — an encoding must never imply uniformity it cannot prove (tripwire class);
  exact cross-pass totals (`total_*`, the autograd trio) encode with a mandatory aggregation
  legend line; unclassified sources refuse `encoding_source_invalid`. `show_legend` is now
  tri-state: `None` (default, AUTO) draws a channel-only disclosure legend iff a channel is
  active; `True`/`False` keep their historical meanings, and explicit `False` is honored even
  with channels active. Typed refusals: `encoding_source_invalid`, `encoding_value_invalid`
  (bools and non-scalar tensors refuse — a bool is not a magnitude), `encoding_callable_error`
  (chains the user exception), `encoding_requires_dot_layout`.
- `Trace.draw(size_by=..., scale="sqrt"|"linear")` (UNSTABLE spellings, keyword-only) is the
  wave-1 SIZE channel, shipped with the D4 DEFAULT APPLIED (D4 unruled at merge: sqrt scale +
  conservative area-only mapping + typed refusal on rolled varying sources). Sources: a scalar
  record field, the closed `"dims"` shape token (numel of the NON-BATCH output shape; the only
  shape-valued source — callables must return scalars), or a callable. Emitted sizes are
  width/height MINIMUMS under `fixedsize=false` (labels never truncate, fonts never scale),
  encoded area clamped to `SIZE_BY_MAX_AREA_MULT` (4.0) x the default node area; strictly
  opt-in (plain `draw()` keeps uniform boxes). On rolled multi-pass nodes size REFUSES where
  color degrades: `size_by_rolled_varying` fires for any source that cannot be certified
  single-valued (marker-varying reconciled fields, mirrored/per-pass projections, varying
  shape under `"dims"` — the refusal fires BEFORE dims resolution, so the honest-aggregate
  range strings are unreachable). Summed-family sources (`total_*`) encode with a mandatory
  aggregation legend line. The spec funnel DROPS NodeSpec width/height when `spec.image` is
  set (an image node's size is pixel-derived); `extra_attrs` stays the power-valve override
  and wins on key conflicts by merge order. `scale=` without `size_by` refuses
  `scale_requires_size_by`; unknown scale tokens refuse `encoding_scale_invalid`.
- `Trace.draw(stack_by=...)` (UNSTABLE spelling, keyword-only) is the wave-1 RANK channel
  (stacking split (a)): nodes sharing an annotation value pin to one Graphviz rank — the
  classic unrolled-RNN timestep diagram; STRICTLY OPT-IN. `True`/`"auto"` derives
  `pass_index` on multi-pass ops only, granted ONLY under the LOCKSTEP LICENSE (the
  pass_index sequence over all multi-pass ops in raw order must be globally non-decreasing;
  then "same column = same execution window" — a layer absent from window k has no node in
  that column, disclosed in the caption). Non-monotone traces (chained loops,
  late-resumption skips, non-monotone nested tallies) refuse `stack_by_auto_underivable`;
  explicit field/callable sources BYPASS the license (caption discloses the source); rolled
  graphs refuse `stack_by_requires_unrolled`. Rank groups resolve at the prepass, travel as
  `RenderIR.stack_rank_groups`, and emit as top-level `rank=same` subgraphs under
  `newrank=true` (cross-cluster constraints are silently ignored without it). While stacking
  is active the sibling-ordering post-pass NO-OPS (two rank-constraint systems would fight);
  collapsed boxes and fold reps stay un-annotated in v1; v1 has exactly ONE cohort.
- CHECKED SUPPRESSION (UNSTABLE spelling `show_redundant_args`, DEFAULT-ON): `draw()` node
  labels omit a module constructor arg exactly when the equality check licenses it — the arg
  value provably equals the captured shape dimension it duplicates on THIS trace (closed torch
  module-family table in `torchlens/visualization/_arg_suppression.py`; the check runs at the
  trace-bearing prepass and is data equality on records, never a render-back loop).
  kernel_size/stride/padding/dilation/groups/num_embeddings/num_heads are NEVER candidates; a
  mismatch or unavailable shape keeps the arg VISIBLE (self-honest — the mismatch case is the
  interesting one); rolled varying aggregates keep args visible while unrolled per-pass nodes
  suppress (deliberate divergence, pinned both modes); detached records render all args.
  `show_redundant_args=True` shows everything. Doc: `docs/reference/encoding.md`.
- `Trace.draw(collapse="none"|"auto"|"max"|t, fold_repeats=None|True|False)` controls v2 smart
  collapse for rolled and unrolled graphs, where float `t` in `[0.0, 1.0]` follows the public
  monotone schedule (`0.0 == "none"`, `1.0 == "max"`). `auto` IS the first schedule point whose
  visible count enters the readable band (unfrozen, F11 memo D8: auto, float levels, and the
  public schedule all read one typed event ladder; interior t maps geometrically in realized
  count; band misses serve the disclosed strongest point).
  `None` preserves defaults (`"none"` has no run folding; `"auto"`/`"max"` use band-pressure
  folding), `True` folds eligible repeated runs even with `collapse="none"`, and `False` disables
  run folding. `collapse="max"` may emit segment boxes; `(xN)`, ellipsis, and segment labels must
  `Trace.collapse_plan(mode=...)` returns the diagnostic
  plan, and `Trace.collapse_schedule()` returns the float schedule metadata. Smart-collapse
  admission (F11, memo D5) gates on U -- the rendered universe of the full plan -- tiered by
  a measured (U, W) work estimator, with `COLLAPSE_OPTIMIZER_MAX_OPS` (2000,
  `torchlens.visualization.collapse_optimizer`) as the defensive constant: over-budget
  requests DEGRADE to a deterministic compact fallback plan (coded
  `collapse_budget_fallback`, `planner="linear_fallback"`), never an uncollapsed wall;
  only pathological inputs (raw ops above 20x the constant) still decline outright
  (`collapse_pathological_skip`; `collapse_plan()` refuses `collapse_plan_unavailable`;
  the schedule degrades to one full-graph step). Context reductions (`module=` focus,
  `vis_call_depth`, rolled mode) genuinely re-admit the quality planner.
- Smart-collapse metadata is computed at access time: `Module.collapse_score`,
  `Trace.module_collapse_order`, and `Trace.collapse_order(mode=...)` (the documented-inert
  `weights=` parameter was removed -- collapse memo D9, clean-v2). These are
  not portable fields and must not be added to `*_FIELD_ORDER` without an explicit schema change.
- `Trace.forward_peak_memory` is a real runtime measurement, never a portable fact. CUDA
  reports the device peak; CPU/MPS report only the cheap host RSS (or MPS allocator) delta,
  which legitimately reads `0` when a forward fits in already-resident heap headroom.
  `CaptureOptions(measure_python_peak_memory=True)` additionally folds in a `tracemalloc`
  Python-allocation peak, which stays positive for tiny models. It is OFF by default because
  the allocator hook costs 1.7x-2.5x total capture time on real CNNs/ViTs. The flag is a
  session-time knob (`FieldPolicy.DROP`, not in `MODEL_LOG_FIELD_ORDER`) and does not survive
  save/load. Never assert `forward_peak_memory > 0` on the default path.
- Sharded distributed state is REFUSED, not silently mis-captured. `tl.compat.report` carries
  `dtensor`, `device_mesh`, `tensor_parallel`, and `pipeline_parallel` rows, and capture entry
  raises `tl.errors.DistributedCaptureUnsupportedError` (structured findings on
  `exc.fields["findings"]`; branch on `finding.kind`, never message text). DTensor/ShardedTensor
  work below the `__torch_function__` layer, so an unguarded capture reported 0 modules and 0
  params. `dtensor`, active `tensor_parallel` hooks/styles, and `pipeline_parallel` refuse;
  `PrepareModuleInput` can omit redistribution/collectives even with dense parameters. Only a bare
  inert `DeviceMesh` remains informational. The bounded scan covers inspectable inputs/plain attrs
  and direct TP-namespace hook registries; slots/descriptor-only holders, opaque user-wrapped hooks,
  over-bound state, and tensors created inside `forward` remain disclosed residuals.
  Detection lives in `torchlens/_distributed.py` and is shared verbatim by both surfaces; the
  `HAS_DTENSOR` / `HAS_DEVICE_MESH` / `HAS_PIPELINING` capability flags gate the exact-`isinstance`
  path and fall back to structural namespace matching. The `dtensor` finding carries per-site
  dual geometry on `finding.geometry` (logical shape, placements, shard offset, local elements),
  so refused sharded state is precisely identified rather than reported as zero parameters.
- EXPLICIT `torch.distributed` python collectives in a traced forward are captured as first-class
  boundary nodes (merge-ranks tier b). The opt-in is `tl.distributed.arm()` at process start
  (REQUIRED for MPMD / spawn-rank programs; installs group-lifecycle wraps, verifies the
  five-namespace collective recognizer fail-closed, stamps the install-epoch record) or lazy
  arming at capture entry for already-initialized SPMD processes. Each boundary op's portable
  `annotations["collective"]` carries the `collective_boundary_v1` payload -- correlation key
  `(membership_digest, lifetime_ordinal, channel, seq)` with issue-ticked per-`(uid, channel)`
  counters, role-indexed dual-geometry entries, event disclosures (async completions are
  `completion_binding="unobserved"` + `read_of_inflight_destination`; never guessed), witness
  fields, and `lifetime_evidence` -- and the trace serializes its group-lifecycle ledger under
  `trace.annotations["distributed"]`. `CaptureOptions(distributed_witness="digest")` is the
  session-time witness knob (`FieldPolicy.DROP`; `"payload"` reserved for C1). Typed refusals:
  `ambiguous_group_lifetime` (unprovable pre-arming group lifetime),
  `uncaptured_collective_op` (arm-time recognizer set-inequality or dispatcher schema-scan hit),
  `wildcard_recv_unsupported`, and `collective_boundary_runnable_unsupported` (runnable save +
  forward-replay validation refuse on collective-crossing traces; metadata invariants run in
  full). The pre-join membership-lineage audit over per-rank ledgers lives in
  `torchlens.distributed.audit_membership_lineages` and is run by the C1 merge engine. Arming
  relaxes NO tier-(a) refusal: DTensor/TP/FSDP2/PP capture stays refused until the C2/C3 census
  proves fidelity. Distributed rank processes (initialized process group, non-daemonic) are the
  one sanctioned exception to the no-child-process capture guard.
- CROSS-RANK MERGING (rung C1): `tl.merge_ranks([trace_or_path, ...])` stitches N rank cores into
  a `MergedTrace` presenter (never a `Trace`/`Bundle` subclass) at their explicit collective
  boundaries; `tl.merge_report(...)` is the graph-free diagnostic that never raises on conflicts.
  The one pure derivation (audit-first: conflicted memberships never join and never become
  presence gaps; seq-DELTA alignment from each rank's first recorded key, absolute bases never
  compared; demote-only digest witnesses with the totalized `BoundaryConsistency` derivation)
  runs at merge time and again verbatim at load rederivation. Frozen vocabularies
  (`MergeAlignment` stored-vs-effective, `BoundaryConsistency`, `MergeValueStatus`,
  `MergedErrorCode`, finding kinds) live in `torchlens.merged` and are release-gated against
  `docs/reference/merged_trace_contract.md`. `merged.save(path)` writes the `merged-directory`
  artifact (canonical-JSON descriptor CACHE + per-member tree hashes + byte-identical rank
  cores); every load reruns the derivation and requires EXACT cache equality (tamper refuses
  typed, never degrades to a gap); an unparseable member enters `load_degradations` and caps the
  effective alignment at `partial`. Refused typed in C1: p2p/pipeline boundaries (C3), DTensor
  dual geometry (C2), merged-level selectors, merged replay/runnable export/validate.
  `distributed_witness="payload"` remains a typed construction refusal (digest witnesses only).
- `CaptureOptions(save_budget=...)` bounds retained activation bytes per device, defaulting to
  `"auto"` (half of each device's available memory measured at its first save). Crossing it raises
  `tl.errors.SaveBudgetExceededError` naming the committed footprint, the tripping op, and the
  remedies. The primary retained copy is admitted before allocation and physical storage is
  alias-aware, but this is not a general OOM guarantee: the forward, transform-only deltas, and
  cross-device temporaries can allocate first. A float sets another fraction, an int an absolute
  cap, and `None` disables it. Predicate-selected disk-only saves are exempt; exhaustive
  `capture=tl.options.CaptureOptions(layers_to_save="all")` plus `to_disk(...)` remains budgeted
  until postprocess eviction (the former bare flat
  `layers_to_save=` kwarg is removed). Unmeasurable auto
  devices warn on first charge and require an absolute budget for enforcement. The reported figure
  is an explicitly-labelled lower bound.
  Like `measure_python_peak_memory` it is a session-time knob (`FieldPolicy.DROP`, not in
  `MODEL_LOG_FIELD_ORDER`) and load restores the default.
- STREAMED DISK WRITES ARE ASYNC BY DEFAULT for `trace(storage=tl.to_disk(...))` (spellings
  DOCUMENTED-UNSTABLE): blob serialize+write+sha256 overlap the forward on ONE FIFO worker
  (manifest order preserved), payloads are snapshotted at submission (value-at-call-time survives
  later in-place mutation; the worker never runs a wrapped torch op),
  `to_disk(max_pending_bytes=)` (default 256 MiB) BLOCKS capture when the disk falls behind
  (bounded RAM, measured exact), a failed write latches and raises typed `TorchLensIOError`
  marking the temp bundle PARTIAL, and finalize drains every pending write before publish —
  async and sync bundles are byte-identical. `to_disk(async_writes=False)` restores synchronous
  writes; `tl.record` streaming stays synchronous and refuses an explicit `True`.
- `torch.compile` coexists with capture as a boundary UPGRADE on torch >= 2.6 and a graceful
  boundary below. Behind the feature-detected `HAS_SET_STANCE` flag, every capture holds the public
  `torch.compiler.set_stance("force_eager")` scoped to the forward (skipped when Dynamo was never
  imported), so compiled plain attributes and free functions run their ORIGINAL eager Python: the
  interior is fully logged with FULL verified semantics (no ceiling), interior interventions work,
  zero graph breaks / zero new compiles happen during capture (fresh shapes included), compiled
  caches stay intact with the warm artifact bitwise-reproduced afterward, and TorchLens wrapper
  install/uninstall costs at most ONE bounded recompile on the next compiled call (verify with
  `tl.debug.count_compiles()`; correlate breaks with `tl.debug.graph_breaks()`). Compiled child
  `nn.Module`s are still unwrapped to their eager source in both regimes. On torch < 2.6 (or a
  stance that fails to engage) the exact historical fallback holds: a Dynamo-traced region reached
  mid-capture is bypassed with a one-per-forward warning and the returned `Trace` honestly contains
  only what ran outside it, marked `capture_verified=False` /
  `capture_verification_reason="dynamo_region_not_logged"` (top precedence, since Dynamo's compile
  threads and unaccounted aten dispatches are symptoms of that same region), and compiled plain
  attributes are inventoried before forward and invoked with logging paused so cold/warm honesty
  does not depend on `is_compiling()` timing (conservatively ceilings even an unused attribute; hot
  global/free callables remain a disclosed residual there). `FakeTensor` /
  `FunctionalTensor` on inspectable inputs or
  parameters refuse at capture entry with `UnsupportedTensorVariantError`, alongside meta and sparse;
  exact-type `_to_functional_tensor` values are covered, while slots-only holders and values created
  inside `forward` remain disclosed by the compat row.
  Gated by `HAS_SET_STANCE` / `HAS_DYNAMO_IS_COMPILING` / `HAS_TRACING_TENSOR_TYPES`.
- Stale pre-wrap torch references (safety net, stage 2): the sys.modules crawler is DELETED —
  TorchLens never crawls `__main__`, reads module sources, or mutates user objects during capture
  (the one sanctioned, user-invoked exception: `tl.release_model` normalizes held torch-function
  attributes on the released model, and wrap-state flips re-normalize registered released
  models). A capture with
  an escape signal (provenance warning / detector diagnostic / output-attribution failure) is
  re-run ONCE with a `TorchFunctionMode` net that redirects stale calls to their exact wrappers
  (`backends/torch/rescue.py`); the result is disclosed (`capture_verified=False`, reason
  `"mode_rescue_rerun"`, session-time `trace.rescue_rerun`). Primary captures are NEVER mode-armed
  (fused-path observer effect). Protocol-invisible constructors that no mode can see (derived
  per build: `from_numpy`, `from_dlpack`, `frombuffer`, `Tensor.as_subclass`, `Tensor._make_subclass`) keep targeted module-attr patching
  (`backends/torch/belt.py`). Residuals declared, typed, never silent: worker-thread stale refs
  (modes are thread-local) and de-moded `handle_torch_function` composite interiors disclose
  `"escape_rescue_unrecovered"`; an authoritative witness/detector/dynamo verdict stays in place,
  and a witness-VERIFIED capture suppresses the rescue. The former
  `wrap_torch(patch_policy=, patch_modules=)` no-op kwargs are removed. Full contract:
  `docs/migration/scoped_detached_patching.md`.
- `torchlens._io` and `torchlens.io` own portable `.tlspec` save/load helpers. Manifest
  schema v2 is backend-aware; non-torch preview bundles may be audit-only or metadata-only.
  Rehydration floor: artifacts stamped `tlspec_version` < 6 refuse to load with the typed
  `tl.errors.ArtifactVersionBelowFloorError` (drop-not-resurrect; the legacy field-alias
  ladders are deleted). The first tlspec-6 writer was released torchlens 2.31.0, so 2.31.0 /
  2.32.4 artifacts LOAD (the compat ledger's governed producer windows are the authority;
  the floor is the STAMP, never a release name). Legacy 2.16 intervention specs remain
  loadable — the floor covers Trace rehydration only. WRITE/READ SYMMETRY (W051-IO): the
  save path dry-runs the exact `metadata.pkl` bytes through the loader's default-deny
  unpickler before writing — an unportable value refuses typed (`annotation_value_unportable`
  naming the key path / `metadata_value_unportable`), tensors under annotations (incl.
  `log_value` tensors) are coerced to plain detached tensors on the scrubbed copy, and the
  loader anchors manifest facts to the pickled state (`bundle_manifest_metadata_mismatch`),
  validates op graph coherence (`artifact_graph_structure_invalid`), corroborates tier-(ii)
  edge carriers against the audit (`artifact_edge_substitutions_invalid`), and checks the
  TorchLens-owned annotation families (`artifact_annotations_invalid`); a v9+ artifact with
  no `root_entry_point` refuses `artifact_root_entry_point_invalid`.
- TLSPEC V9 (the C07 coordinated schema write, 2026-08-27; current version): the
  `intervention_audit` grammar admits the shipped per-site `source` disclosure, PARAM
  rows/recipes (pre-v9 readers refused a saved selection- or param-intervened artifact's OWN
  load), and the EVENT (`intervention_event_v2`) row kind with its optional hash-chain
  extension (`seq`/`prev_event_digest`, the F03 ledger's slots); the `"sidecar"` annotations
  namespace persists plainly (envelope-shape load validation, provider-absent reads stay
  analysis-only); `"health_facts"` and `"capture_advisories"` are reserved annotation
  families with their validation rows; and three ENTRY-DARK slots are declared with
  fail-closed load validation so their Phase-3 writers need no further bump:
  `Op.injection_provenance` (F01 log_injections), `Trace.source_snapshots` (F30),
  `Trace.structure_evidence` (F33). Contract of record:
  `torchlens/schemas/writer_contract_v9.json` (v8 pinned under `tests/release_goldens/`);
  the C07-adjudicated field-intent census is maintained privately. The C07X
  AMENDMENT rides the SAME v9 window (TLSPEC_VERSION stays 9; JF-9 ratified): bundle
  relation grammar v2 (required/optional split; `successor_of` admits the optional
  evidence envelope `{schema, items[], facts_digest}` + `carry_mode`/`state_source`;
  graded claims over the closed verified/consistent/disclosed/unchecked/divergent
  vocabulary with the contracted unchecked-reason menu incl. `no_param_snapshot` --
  user-authored MEASUREMENT grades (verified/consistent/divergent) are clamped to
  `disclosed` at row construction, relate-time and load-time alike, with the declared
  grade preserved under `declared_grade` (users never mint trust; M8); a
  1 MiB per-row evidence budget refuses `bundle_relation_evidence_over_budget`; the
  successor_of direction pin: `from` = the LATER member), the three-leg
  preserve-and-disclose loader doctrine (unknown NAMESPACED relation kinds load opaque
  as `OpaqueRelationRow` under `tl.load(unknown_relations="opaque"|"refuse")`; unknown
  evidence schema ids load opaque; unknown namespaced bundle.json sections preserve
  through `Bundle.preserved_sections` while bare-unknown sections refuse
  `bundle_section_unknown`), reserved ID registrations (evidence schema ids
  `version_boundary_v1`/`turn_boundary_v1`; sidecar family ids
  `torchlens.input_origin`/`input_digest`/`boundary_facts` behind a namespace squat
  guard), episode ledger grammar v2 (the EPISODE CAPTURE bullet above), and three more
  entry-dark slots with fail-closed validation: `Trace.root_entry_point` (the closed
  `module_call|bound_method|function_call` root descriptor, written UNCONDITIONALLY on
  every capture; joined to the rerun identity gate FAIL-CLOSED — the F41 bullet above),
  `Op.episode_step` (F-EPISODE writes), `Op.tl_authored_root` (F41 writes it on
  bound-method captures, live). The Bundle-side ledger rows
  join (foldB s4.4 item 12) SPLIT OFF per the review proviso: the C07 ledger shape stays
  unresolved pre-F03 and must not hold the amendment past V9-FREEZE.
- SITE KEYS + GROUPING SURFACE (L1 wave 0; every spelling DOCUMENTED-UNSTABLE
  pending naming-session/S2 ratification): every retained op carries
  `op.site_key` (`site_key_v1`) — a portable, policy-independent
  STRUCTURAL-POSITION identity minted at grouping time on every backend
  (`"s1|" + module-site/type/slot/ordinal`, percent-escaped; ordinals restart
  per pass-qualified innermost call instance, so reused-module calls share
  keys across instances). It is a BRIDGING relation (cross-capture joins on
  position, never proven source identity: per-call-instance cardinality guard
  + source-location witness + corroborated/positional/refused verdict tiers,
  internal until S2). `Layer.site_key` returns the single shared key or
  refuses typed `layer_site_ambiguous` (within-call recurrence groups span
  sites); `Layer.site_peers` indexes same-site layers live; keyless legacy
  artifacts refuse `site_key_unavailable`. `Layer.shape_summary` is the
  derived across-pass shape string ("2->4" monotone / "2-4" min-max /
  first->last full shapes; contains `->`, escape at render). The
  `grouping=` trace kwarg is closed-vocabulary ("structural" default;
  "strict_shapes"/"fold_sites" refuse typed until their designs/D1 rule);
  `trace.grouping` mirrors the request and `trace.grouping_policy` is the
  load-validated `grouping_policy_v1` stamp (coherence rules C1-C8;
  invalid/legacy stamps settle to the canonical degraded representation and
  refuse stamp-consuming operations typed). All persisted rows are
  persisted as of the tlspec v8 coordinated bump (load-validated). Invariants
  I-S1/I-S2/I-S3' are live tripwires; folding stays OFF everywhere except
  future episode products (D1 default) — the tier-(a) fold closure ships
  entry-dark as a pure function.
- STRUCTURE-ONLY CAPTURE (L7a wave 0, D8-DEFAULT branch; every spelling
  DOCUMENTED-UNSTABLE pending naming-session/S2 ratification):
  `tl.trace(model, x, capture=CaptureOptions(structure_only=True))` records the
  op graph, module hierarchy, parameter geometry, and per-op shape/dtype with
  every value-bearing claim a HYPOTHESIS — value payloads are never retained,
  value-requiring consumers (save/replay/validate/backward) refuse typed
  through the ONE chokepoint in `torchlens.capture.structure_only`
  (capability contract: `docs/reference/structure_only_capabilities.md`), and
  value-dependent branches refuse DEVICE-NEUTRALLY with the user's exact
  source line (`value_dependent_branch_unsupported`; missing meta kernels
  refuse `meta_kernel_unavailable`). Option conflicts (`raise_on_nan`,
  `intervention_ready`, not-provably-value-free `halt=`) refuse
  `structure_only_option_conflict` at entry. `trace.structure_only` mirrors
  the flag (persisted as of the tlspec v8 bump with M-C2/M-C3 load-validation
  rows); `trace.discharge_against(real_trace)` corroborates or refutes the
  hypotheses against a real capture, and a REFUTED discharge flips hypothesis
  consumers to typed refusals. WEIGHTS-FREE ADMISSION IS LIVE (D8 granted
  2026-08-26, F33): meta-materialized models are admitted IFF structure-only
  is in force AND the substrate is uniform (all-meta inputs + state; mixed
  cells refuse `structure_only_substrate_mismatch` in both directions, and a
  real tensor minted mid-forward refuses through the same family at the
  user's line). The parity gate is the acceptance authority (real digest ==
  meta digest AND `discharge_against` CORROBORATED — distilgpt2 291/291
  records + 873 claims, BERT 307/307 + 921, resnet50 389/389 with all 106
  declared BN buffer writes, Llama-2-7B geometry 6,738,415,616 params +
  225 linears exact). The discharge has a comparable-twins preflight
  (`structure_only_discharge_incomparable`; refuse is not refute — build
  BOTH twins before the first capture), the report names deltas
  structurally, and every capture carries the persisted evidence envelope
  `Trace.structure_evidence` (tlspec v9 slot, fail-closed load validation)
  with the HYPOTHESIS/CORROBORATED/REFUTED ladder on every surface (repr,
  summary, slices, draw captions, exports, agent JSON, MCP). Measurement
  exports (chrome_trace/speedscope/flamegraph/memory_timeline) refuse
  `structure_only_measurements_unsupported`; `tl.summary(meta_model,
  input_size=...)` is the rung-4 auto-selecting facade (bare summary with
  no input evidence refuses); `Trace.check_plan(plan)` is the audit-only
  nnsight-scan-parity checker (`executable=False` always). Doc of record:
  `docs/reference/weightsfree_capture.md`.
- SELECTION ALGEBRA (L6 stage 1; Selection/ResolvedSelection/resolve/
  `__selection__`/operators slate-ratified subject to D7, producer
  constructors + members DOCUMENTED-UNSTABLE): `tl.Selection` is the
  composable trace-independent query AST; `selection.resolve(trace)` returns
  the frozen trace-bound `tl.ResolvedSelection` (ordered `SiteEntry(site_key,
  mask, provenance)` tuple; SESSION-ONLY, never persisted). TWO-LEVEL
  denotation: (touched-site family, selected-element set) — zero-mask entries
  stay first-class, `.empty`/`__bool__` are element-level, and `bool()` on
  the QUERY refuses typed. Operators `| & - ~` + reflected forms, NO
  `__xor__`; `-` never un-touches, `~` is touched-site mask complement (never
  predicate negation: `~lift(s) != lift(~s)`). Every region-shaped producer
  implements `__selection__` (BaseSelector, ReceptiveFieldBox,
  GradientReceptiveField, FacetSpec, Op, Layer) and carries the operator
  mixin, so `u1.receptive_field.at(p) | u2.receptive_field.at(q)` IS a
  Selection; `selector OP selector` keeps shipped CompositeSelector semantics
  and `selector - selector` desugars to `and(a, not(b))`. Masks are exact AS
  SETS with producer inexactness on the closed `provenance.relation` lattice
  (`exact|upper_bound|lower_bound|unknown`; JOIN/FLIP/DIFFERENCE tables are
  normative). Kinds `ACT|PARAM|EDGE` are closed; mixed kinds refuse
  `selection_kind_incompatible`; resolution refusals ride
  `SelectionError` with `selection_unresolvable` + a closed reason set.
  Producers: `tl.units(site, indices)`, `tl.params(name, mask=None)`,
  `tl.random_selection(like=, within=, seed=)` (seeded size-matched control).
  VALUE + STATISTICAL PRODUCERS (producer wave; DOCUMENTED-UNSTABLE):
  `tl.top_k(within, k, by=, largest=)` / `tl.top_fraction(within, fraction)`
  (global deterministic ranking over the population; too-few rankable
  elements refuse `population_too_small`), `tl.threshold(within, above=,
  below=, by=)`, `tl.sign(within, 'positive'|'negative'|'zero'|'nonzero',
  tol=)` (`'zero'` = the sparsity mask / single-capture "didn't fire") read
  the RESOLUTION trace's retained activations at resolve time — exact-as-set
  claims about THIS capture (`relation="exact"`); `within=None` means every
  retained tensor site, PARAM/EDGE populations refuse
  `selection_kind_incompatible`, unsaved payloads refuse `value_not_saved`,
  NaN never satisfies a criterion, complex ordered comparisons refuse
  `value_criterion_invalid`. `tl.dead(samples, tol=)` / `tl.saturated(
  samples, low=, high=, tol=)` / `tl.low_variance(samples, threshold=)` are
  explicitly MULTI-SAMPLE (`samples=` iterable of >= 2 Traces; Bundle
  iterates; the single-capture form is deliberately the different spelling
  `sign(site,'zero')`): the resolution trace supplies geometry/population
  only, dispositional claims (dead/saturated) declare
  `relation="upper_bound"`, the sample statistic (low_variance) declares
  `exact`, and missing/unsaved/shape-drifted sample evidence refuses typed
  with the offending sample named.
  COMPARATIVE PRODUCERS (comparative wave; DOCUMENTED-UNSTABLE, interface
  flagged for the UI-sprint review): `tl.changed(reference, within=None,
  above=, below=, by='abs'|'signed')` / `tl.top_changed(reference,
  within=None, k=|fraction=, by=, largest=)` select by HOW VALUES DIFFER
  between two runs — the SUBJECT is the resolution trace, the REFERENCE one
  explicit Trace (pairwise, directional `subject - reference` in float64;
  bare `changed(ref)` = the "every element that moved" intervention-effect
  mask; `relation="exact"`). Structures NEVER silently intersect: reference
  missing-site/unsaved/shape-drift refusals name the reference, a
  structural-site-key disagreement refuses (label coincidence across
  architectures is caught), and self-comparison refuses (vacuous). PARAM
  populations refuse — Param records hold live refs, never capture-time
  payloads, so checkpoint weight diffs are not claimable from Traces.
  `tl.stable_across_passes(within=None, tol=, passes=)` /
  `tl.pass_variance(within=None, above=, below=, passes=)` select by
  behaviour ACROSS a recurrent layer's passes (range-within-tol / variance
  bounds, float64, pass-qualified throughout): windows are explicit >= 2
  distinct 1-based passes or all population passes, a layer contributing
  < 2 window passes refuses `population_too_small` (vacuous single-pass
  claim, teaching message), cross-pass shape drift refuses, the element
  population is the INTERSECTION of window masks, and the mask lands on
  EVERY window pass-site (`do()` edits every window pass);
  `relation="exact"`.
  SUBSPACE PRODUCER (subspace wave; DOCUMENTED-UNSTABLE):
  `tl.subspace(within, basis, *, origin=, method=None, dim=-1, tol=0.0)`
  selects the elements a DIRECTION in activation space lives on (probe
  directions, steering vectors, PCA components, SAE decoder rows; basis
  `[d]` or `[k, d]`, canonicalized float64). SET, NOT PROJECTION: it
  resolves to the basis SUPPORT SET (`|w| > tol` on the bound axis, union
  over rows, expanded across other axes; a dense direction supports the
  WHOLE axis and `do()` edits every supported element, never "the
  component along the direction" — projection-valued selections are a
  named design fork, not a promise). BASIS PROVENANCE MANDATORY:
  `origin=` required non-empty; origin/method/geometry/sha256 content
  digest ride `provenance.source` and `do()` audit records
  (`selection_subspace.BasisProvenance` is the programmatic face).
  DIMENSION HONESTY: extent mismatch on the bound `dim=` (default `-1`;
  conv channels `dim=1`) refuses `basis_dim_mismatch`, never
  broadcast/truncate; `within=` REQUIRED; non-finite or sub-tol-row bases
  refuse at construction; resolution is geometry-only (unsaved sites
  resolve); relation `exact`; PARAM/EDGE refuse
  `selection_kind_incompatible`. New extension seam:
  `torchlens.selection.register_term_resolver` (producer modules register
  frozen AST terms at import; `torchlens/selection_values.py`,
  `torchlens/selection_graph.py`, `torchlens/selection_subspace.py`).
  GRAPH-STRUCTURAL PRODUCERS + SLICE VIEW (graph wave; DOCUMENTED-UNSTABLE):
  `tl.neighborhood(of, hops=, direction='both'|'upstream'|'downstream')`
  (every op within N recorded dataflow hops of the seed region; `hops=0` =
  the seed family) and `tl.between(sources, sinks)` (the executed sub-DAG on
  at least one directed source-to-sink path, endpoints included; operands
  accept lists; no path = EMPTY selection, disclosure not error) select by
  STRUCTURAL POSITION — whole-site masks, `relation="exact"`, element masks
  never shrink a graph region (family semantics), PARAM/EDGE operands refuse
  `selection_kind_incompatible`. Both are pure functions over the ONE
  executed-DAG substrate (`selection_graph._TraceGraph`, adjacency shared
  verbatim with the influence-geometry path machinery) — a future graph-
  MOTIF producer is one more pure function over the same object, not a
  rewrite. `trace.between(sources, sinks)` returns the SAME region as a
  `TraceSlice` — a frozen PRESENTER (composition, never a Trace subclass,
  the MergedTrace precedent): member ops in execution order, internal
  dataflow edges, and an EXPLICIT boundary (`boundary_in_edges` /
  `boundary_out_edges` name every crossing edge; `source_ops`/`sink_ops`
  the entry/exit ops). A slice offers NO save/replay/validate; `tl.save`
  refuses `slice_save_unsupported`; `__selection__` lifts the member family
  back into the algebra (slices compose and feed `do()`).
  `trace.subgraph(selection)` is the general door presenting ANY ACT region
  as the same view. Session-time only; never persisted.
  `torchlens/selection_compare.py`).
  CROSS-RUN (L6 stage 4a; DOCUMENTED-UNSTABLE): `resolved.align_to(target)`
  re-binds an ACT selection onto another trace keyed on the L1 structural
  site keys each `SiteEntry` records (`structural_site_key`, now live data),
  SAME-POLICY captures only per the L1 cross-stamp rule (healthy agreeing
  grouping stamps both sides); refusals ride `selection_alignment_invalid`
  with a closed six-reason set, `do()` keeps refusing foreign resolved
  selections (`selection_trace_mismatch` -- alignment is the one explicit
  door), and `align_to` + `tl.patch_from(source)` is the cross-run patching
  spelling the acceptance gallery pins.
- EDGE SUBSTITUTION (L6 stage 3; DOCUMENTED-UNSTABLE): `trace.edges` is the
  dataflow edge family (EdgeUseRecords; intervention_ready-gated, refusal
  `edge_provenance_unavailable`); canonical occurrence address
  `(child_func_call_id, arg_kind, arg_path)`. `do(edge_selection, edit)`
  replaces the value CONSUMED on the edge on the replay/push engine ONLY
  (rerun/set_only refuse `edge_intervention_engine_unsupported`; rerun-side
  design is an escalated named future) — the child re-executes from the
  substituted input (node-level `intervention_replaced` never fires for
  edges), the substituted value rides the tier-(ii) store
  `Op.edge_substitutions` (+ `edge_replacement_stamps`,
  `FireRecord.edge_address`; all persisted as of tlspec v8, so ordinary
  saves of edge-intervened traces proceed), and capture truth
  (saved_args / out_versions_by_child / parent.out) is retained unmodified
  (parity-pinned). Validation: uncorroborated tier-(ii) entries FAIL;
  corroborated children re-execute from the spliced value and must match
  (distinct verdict `edge_intervention_boundary`). SCHEMA-REGRESSION
  TRIPWIRE at the `_io/bundle.py` save entry (fires IFF tier-(ii) entries
  are present AND the active schema would drop the carrier): ALL four
  levels refuse `edge_intervention_save_unsupported`, PRECEDING
  `artifact_save_level_unsupported`; save-entry refusal order is
  MergedTrace -> N1 outcome -> L7a structure-only -> L6 edge boundary (do
  not silently reorder another owner's refusal). `tap(resolved_selection)`
  stores per-site masks on TapRecords; `values(masked=True)` returns fresh
  masked copies.
- PARAMETER SUBSTITUTION (param-operand, decided 2026-08-17, supersedes the
  D3 typed-refusal default on the replay path; DOCUMENTED-UNSTABLE):
  `fork.do(tl.params(name, mask=None), edit)` applies the edit "AS IF" the
  parameter were changed, for replay only — the value each consuming op sees
  is substituted at its derived occurrence address and the live
  `nn.Parameter` is NEVER written (bit-identical pinned). Parameters are NOT
  in the edge family (LiteralTensor template components, not EdgeUseRecords),
  so `intervention/param_substitution.py` DERIVES the addresses
  (`Param.used_by_ops` + template identity/barcode match, FAIL-CLOSED:
  nested positions, released legacy captures, bare pass-ambiguous
  multi-pass consumer spellings, and consumer inventories omitting a pass
  refuse `param_substitution_occurrence_underivable`). Recurrently reused
  params (tied weights, multi-pass consumers) ARE substitutable: consumers
  stage pass-qualified and the edit lands at EVERY consumption (a
  parameter has one identity across passes; contrast the bare-LAYER-label
  ambiguity, which stays a refusal). Derivation drives the SAME tier-(ii)
  engine: entries marked
  `substitution_kind="param"`, edit-then-scatter masking over the param
  space, ONE replay pass over all consumer origins whose cone recomputation
  RE-SPLICES every tier-(ii) entry (param, region, and edge kinds), so later
  pushes never silently revert any edit; chained param edits compose), and the SAME
  validation boundary (`edge_intervention_boundary`; uncorroborated FAIL).
  Replay/push engine ONLY: rerun/set_only refuse
  `param_substitution_engine_unsupported`. The audit record (kind `PARAM`)
  discloses "substituted at consumption ... live parameters unchanged".
- PASS-QUALIFIED REPLAY (decided 2026-08-17; refusal spelling
  DOCUMENTED-UNSTABLE): the replay/push engine operates on pass-qualified
  op labels (`Op.label`, the `label:pass` spelling — single-pass ops carry
  `:1`). Cone traversal, the replay overlay, hook targets, origin sets, and
  pending commits are all keyed per pass, so `fork.do(...)` on multi-pass
  (recurrence-grouped) layers edits exactly the addressed pass, recomputes
  every downstream pass, and commits every pass's record (previously bare
  layer_label keys silently truncated the cone, fell back to the LAST
  pass's captured out, fired one pass's hook at every pass, and committed
  only the last pass — wrong values, nothing raised). The parent-edge
  divergence check compares in replay-key space (no more spurious
  `ControlFlowDivergenceWarning` on multi-pass replays; `strict=True`
  multi-pass replay works). BOUNDARY: a bare layer label naming a
  multi-pass layer refuses typed — `multipass_bare_label_ambiguous`
  (`SiteAmbiguityError`) on string/`tl.label` addressing
  (do/attach_hooks/push_from/resolve_sites), `selection_unresolvable` /
  `multipass_bare_label` on `tl.units` — with a teaching message naming
  every pass-qualified spelling; bare labels on single-pass layers stay
  accepted, predicate selectors keep fan-out, and the explicit Layer
  selection (`log[label].__selection__()`) is the all-passes spelling.
  Post-hoc `contains`/string addressing accepts pass-qualified needles
  (a needle containing `:` also matches `Op.label`), and the hook-context
  `layer_log` snapshot carries `label` + `pass_index`. Replay disclosures
  (`last_run` origins/cone, `replay_frontier` keys) spell multi-pass ops
  pass-qualified and keep bare labels for single-pass layers. The
  param-substitution multi-pass refusal is NARROWED on this engine (tied /
  recurrently-reused params substitute at every consumption, staged
  pass-qualified); `param_substitution_occurrence_underivable` still fires
  fail-closed on bare pass-ambiguous consumer spellings and
  pass-incomplete consumer inventories.
- BACKWARD RESIDUALS (L9; every spelling DOCUMENTED-UNSTABLE pending
  naming-session routing): PER-FIRE TIMING -- every hooked grad_fn
  gets a timing prehook; ONE clock (`perf_counter`) paired at capture by a
  per-node keyed LIFO (stale entries discarded, untimed fires `(None,
  None)`, never a cross-clock pair); stamps ride the runtime `GradFnFired`
  event only, served by live-trace-only `trace.grad_fn_fire_timings`
  (loaded/cleaned traces refuse `grad_fn_fire_timing_unavailable`); as of
  the tlspec v8 coordinated bump the persisted `GradFnCall` timing fields
  carry the per-fire `perf_counter` semantics, discriminated by the
  persisted `Trace.grad_fn_timing_provenance`; timing
  registration failure degrades to untimed, never a coverage gap (D15 A/B
  measured ~4-5%, under the 10% gate; universal path shipped). CHECKPOINT
  TOKENS -- classified non-reentrant `_checkpoint_hook` enters mint one
  per-trace ordinal token (armed owner thread, outside engine invocations;
  fail-closed one-way); pack evidence count-only (forward slot->op binding
  NOT claimed), unpack evidence backward-derived (fire bracket -> user-op
  pairing -> L1 site keys); persisted `Trace.checkpoint_invocation_witness`
  carries counts/candidates/degrade-flags D1-D6/evidence-scoped verdict; the
  ambiguity REFUSAL awaits a pending contract amendment -- identity-read accessors
  are NOT shipped until it lands. IMPLICIT-BOUNDARY --
  `_close_implicit_backward_pass_if_open` is a journal/scavenge/finalize
  split with the finalize guard IN-ROUTINE (D2H fence + projection never run
  inside an engine invocation; mid-engine reads journal without
  materializing); implicit opens enqueue an identity-checked engine-drain
  final callback with the sync-point backstop always armed; the
  `BackwardPassEnd.close_path` disclosure is sidecar-event-only in wave 2.
  GROUPED FLOOR -- `trace.grad_fn_site_summary` rolls backward facts up per
  L1 site_key (read-only; keyless legacy refuses `site_key_unavailable`).
- PREDICATE RUNTIME EXTENSION POINT (S4 seam; every spelling
  DOCUMENTED-UNSTABLE pending naming-session ratification):
  `torchlens.ir.predicate_registry` is the ONE documented door through which
  predicate consumers accept user predicates for the capture-lifecycle
  `save`/`halt`/`until` slots (`PredicateProtocol` — one positional concrete
  `RecordContext`; `coerce_predicate(value, slot=...)` — raw callables incl.
  `BaseSelector` instances returned BY IDENTITY, registered names via a
  slot-aware enforcing wrapper; `register_predicate(name)` — mutates nothing
  on the user's object, stamps no loader-consulted attribute). The registry
  is INERT until consumers adopt name acceptance. `intervene=`/grad slots are
  outside the contract. Contract: `docs/reference/predicate_runtime.md`.
- NETRON EXPORT v2 (F14; every kwarg spelling DOCUMENTED-UNSTABLE pending the
  naming sprint): `tl.export.netron(log, path=None, *, granularity="module"|"op"|"rolled",
  depth=1, show_buffers="never"|"meaningful"|"always", attachment=False, open=False,
  baseline=None)` emits schema-v2 ONNX protobuf-JSON (irVersion 10, custom domains
  `ai.torchlens.lossy`/`ai.torchlens.module` — NEVER re-domained for colour, ruling N1)
  with typed values + native graph I/O, per-call module FunctionProtos (module is the FILE
  default per ruling N2; interim fixed depth 1, single-op modules inlined with their
  FunctionProto DELETED, DAG belt fail-softs to op with coded warning), tri-state buffer
  policy + counter-chain rule with disclosures, variance-gated rolled projection (feedback
  disclosed via the `recurrence` attr, never drawn — a cycle fails the checker), curated
  wording-contract panel attrs (missing is ABSENT never zero; the module-path attr is not
  named `module`), the `netron:attachment` companion (four fail-closed vendor guards,
  `attachment=True`), serve/widget one-liner (`open=True`, extra `torchlens[netron]`), and
  an extent-budget WARN past ~27 ranks. Every artifact stays green under strict
  `onnx.ModelProto` parse + `check_model(full_check=True)`; netron 9.2.2's own parser
  executes over it in CI (tests/test_netron_export_*.py). Doc:
  `docs/reference/netron_export.md`.
- SUMMARY REBUILD (F08; every spelling DOCUMENTED-UNSTABLE pending
  naming-session ratification): bare `trace.summary()` / `tl.summary(model, x)`
  render the AUTO VIEW LADDER (coalesced hybrid -> strictly folded module tree
  -> descending depth -> protected totals-conserving elision) under a derived
  48-body-row budget, returning a `SummaryReport` (str subclass) whose text is
  canonical byte-stable ASCII (unicode only at display boundaries through a 1:1
  glyph table; `ascii == degrade(unicode)` CI-pinned; detection fails toward
  ASCII, `TORCHLENS_SUMMARY_STYLE` overrides). The identity partition is law:
  every param identity and executed op event owned by exactly one row at every
  depth/fold/filter/elision; alias rows never enter the ladder. Grammar:
  level/view/depth/columns/filter/buffers/fold_repeats/max_rows/
  flop_convention/units/style -- the legacy spellings (`level="graph"`,
  `preset=`, `fields=`, `show_ops=`, `count_fma_as_two=`, ...) are removed and
  refuse typed naming their successor (see MIGRATIONS.md); nothing is
  accepted-and-ignored (`fma1` refuses
  `flop_convention_unavailable` when underivable). One-call door: input
  precedence args XOR `input_size=` XOR zero-input (reuses
  `infer_input_shape`'s verified trace; synthesis disclosed);
  `execution_mode`/`grad_mode`/`input_size`
  are one-call-only (`summary_one_call_only` on `trace.summary()`). Result
  API: `render`/`print`/`details`/`to_pandas`/`to_markdown`/`to_html` +
  scalar raw ints (the raw-numbers pin); the report survives model/Trace
  teardown; `trace.provenance()` serves the relocated preamble byte-exact;
  `Trace._repr_html_` delegates to the summary HTML table. Pins re-derive via
  `tools/derive_summary_pins.py`; perf ratio published at
  `docs/benchmarks/summary_performance.md`. Docs: `docs/reference/summary.md`,
  `docs/migration/from_torchinfo.md`.
- AGENT SURFACE (both spellings DOCUMENTED-UNSTABLE pending naming ratification):
  `Trace.to_agent_json(max_ops=None)` emits the self-describing JSON-serializable
  `torchlens.agent_trace.v1` dump (capture honesty facts, counts, pass-qualified op rows
  with graph edges, module hierarchy, embedded navigation guide pointing back at the live
  public surface; payloads never inlined; `max_ops` truncation disclosed, never silent).
  `tl.report.explain(trace, max_tokens=N)` budget-prunes the text report by whole
  sections low-value-first with a disclosed `Truncation` section; capture-status honesty
  facts and partial-capture failure evidence never drop, and `max_tokens` refuses typed
  with `format="json"`. Doc: `docs/for-ai-agents.md`.
- TRANSFORMER PICTURES (`torchlens.tviz`; every spelling DOCUMENTED-UNSTABLE pending the
  naming session): the CircuitsVis/BertViz/Ecco/inspectus picture families on typed
  session-only display records -- `attention_views(trace, tokens=)` reads the semantic
  `pattern` facet into canonical `[query_head, destination, source]` `AttentionView`s
  (provenance captured/reconstructed/user_supplied; mask provenance is the D6
  three-source hierarchy: recorded SDPA call args, the captured eager additive-mask
  operand keyed on `finfo(dtype).min` never `-inf`, explicit user metadata -- absent all
  three the picture renders "zero and masked positions are not distinguished"; zeros
  inference is banned). Renderers save paper-ready PNG/SVG/PDF via matplotlib resolved
  at CALL time (`tv_matplotlib_missing` prints `pip install "torchlens[viz]"`; the
  zero-dependency SVG/HTML token-strip emitters work on a bare install): rect meshes
  never `imshow`, zero `<image>` elements in numeric SVG (vector colorbars included),
  `svg.fonttype='path'` default, pagination that never silently drops. Pictures:
  single-head/grid/atlas attention (`render_attention`, `render_attention_atlas`),
  token strips (`render_token_strip` + `token_strip_html/svg`, multi-row NMF form),
  lens pictures over the streaming `logit_lens_predictions` (`prediction_trajectory`,
  `render_prediction_ribbon`, `render_answer_trajectory`, `prediction_table`),
  loss/entropy strips from ONE fingerprinted logits source (`token_metrics`), the
  term-complete score decomposition with the HARD sum-to-score invariant
  (`score_decomposition` refuses `tv_decomposition_unclosed` rather than
  mis-decomposing; RoPE families refuse per-family), and CAUSAL RECEIPT grids
  (`CausalReceipt` validates fires + BOTH controls at construction;
  `render_receipt_grid` refuses without the measured joint effect -- single-head
  effects are not additive -- and prints "measured N of M"; the closed four-kind
  `Annotation` grammar lets only a valid receipt mint `intervention_effect`, and
  effects never recolor attention cells). Bridges: `circuitsvis_attention`
  (payload-bounded, dormancy disclosed) and the `bertviz_tuple` parity oracle.
  FIX-H shipped in `tl.viz.render_heatmap`: nearest-neighbor discrete cells, separate
  `row_labels=`/`col_labels=`, omission marker computed AFTER final overlap selection.
  Records are session-only, never persisted. Doc: `docs/reference/tviz.md`.
- MECHINTERP KIT (F05; every spelling DOCUMENTED-UNSTABLE pending the naming
  session): `torchlens.mechinterp` is the transformer analysis kit validated
  against TransformerLens oracles -- `residual_accumulation` /
  `residual_decomposition` / `full_decomposition` (bitwise-closed),
  `direct_logit_contributions` (DLA), `attention_head_contributions`,
  `head_scores`, `patch_heads_grid` / `patch_residual_grid` (turnkey
  `PatchGrid` results), `lowered_counterfactual` (patch one head on a FUSED
  SDPA model via exact implementation-independent lowering),
  `resolve_alias` / `translation_table` (TLens spellings resolve against a
  trace; tuple and compact `"k6"` forms), `test_prompt`, `apply_norm_scale`
  (cached-scale LayerNorm linearization, TLens-parity bit-exact), retention
  planning (`retention_plan`). ~34 names; `MechInterpError` family.
- TRAINING-TIME WATCH (F26; DOCUMENTED-UNSTABLE): `torchlens.trackers` is the
  attach-once watch engine over the C06 records -- `watch`/`WatchSession`,
  sinks (TensorBoardSink incl. `add_histogram_raw` precomputed buckets,
  WandbSink, JSONLSink, MemorySink), `HFTrainerWatchCallback` /
  `LightningWatchCallback`, TB GraphDef with real StepStats. CHANGED:
  `tl.export.tensorboard` requires keyword `step` (default removed).
- OBSERVABILITY HOME (C06 + F24/F25/F27; DOCUMENTED-UNSTABLE):
  `torchlens.observability` (~100 names) carries the L5 record schema,
  summary transforms, `HistoryView` (bounded exactly-mergeable training
  history), contact sheets, `measure_ab`, pass peaks, the categorized
  memory-timeline v2, and the Kineto/native-profile join (`native_profile`;
  `hot_path(by="device_time")`; `tl.region` correlation). `torchlens.observe`
  is the user-facing verbs door.
- ECOSYSTEM RUNTIME (F32; DOCUMENTED-UNSTABLE): `torchlens.ecosystem` --
  `compat_window(...)` (dated support-window ledger; would have caught the
  v2.33.0-cannot-load-v2.34.1 class), `migrate` (tl.migrate v1,
  `MigrationReport`/`MigrationStep`), and `plugins` (entry-point discovery
  that NEVER executes third-party code at discovery; kill switch).
  `torchlens.conformance` holds the C0-C2 packs.
- LIVE NARRATION (F28; DOCUMENTED-UNSTABLE): `echo=` on `tl.trace`/`tl.record`
  streams compact per-op narration (the torchsnooper niche on the capture
  substrate, 3-4x cheaper than settrace); `torchlens.snoop` owns
  `NarrationEvent`/`EchoSession`/sinks and post-hoc `narrate_trace` /
  `narrate_recording` / `narrate_partial`; `Trace.narrate` replays a finished
  capture as narration.
- `torchlens.debug` owns power-user diagnostics such as `bisect_nan` and `hot_path`;
  the submodule is imported as `tl.debug` and is deliberately not in `__all__`.
- `tl.receptive_field` is a lazy power-user submodule. `Op`, `Layer`, `ModuleCall`, and
  `Module` expose `receptive_field` and `projective_field` views; `Trace` exposes the matching
  `receptive_fields()` and `projective_fields()` tables.
- `torchlens.bridge` contains optional adapters for Captum, HF, SHAP, SAE Lens, LIT,
  profiler, and related tools. `bridge.sae.splice(model_or_log, inputs, site=, sae=,
  latents_edit=)` (DOCUMENTED-UNSTABLE) is the packaged SAE splice experiment: fork +
  `tl.splice_module` + push swaps a duck-typed SAE's reconstruction (`encode`/`decode`
  pair, no SAE package required) in at one site and reports reconstruction fidelity plus
  output-level causal effect; `latents_edit=` is the per-feature causal knob (demo:
  `notebooks/sae_splice_tutorial.ipynb`). `bridge.brain_score` additionally serves Brain-Score's
  `ActivationsExtractorHelper` seam: `get_activations_fn(model)` is the offline
  per-batch callable (module dotted paths or any TorchLens lookup; `"logits"` = model
  output) and `activations_extractor(...)` wires it into the real helper (requires
  `brainscore_vision`, Python >= 3.11; LIVE-GATE VERIFIED 2026-08-29 against a running
  brainscore-vision 2.3.22 install on py3.12/CPU -- resnet18 through a real helper +
  StimulusSet with value/order/coordinate/logits/functional-op/short-batch parity and
  default sites on a partially saved trace; also exposed as `tl.neuro.activations_extractor`
  / `get_activations_fn`, gated on brainscore_vision alone). `torchlens.bridge.mcp` (extra `torchlens[mcp]`,
  mcp>=2.0; DOCUMENTED-UNSTABLE) is the Model Context Protocol stdio server (`python -m
  torchlens.bridge.mcp`): read-only tools over saved `.tlspec` artifacts + environment
  (doctor / api_map / load_overview / agent_dump / explain), wrapping the same public
  surface -- no user-code execution, no mutation.
- DATASET EXTRACTION (D7/V5; `resume=`, `stimulus_ids=`, and the loader
  DOCUMENTED-UNSTABLE): disk-mode `tl.extract_dataset(..., output_dir=)` writes atomic
  shards plus a self-describing `manifest.json` (run signature, per-site identity with
  L1 site keys where derivable, stimulus ordering/provenance, axis semantics, dtypes,
  devices, transform disclosure, TorchLens version), ledgered per batch. `resume=True`
  continues an interrupted run from its last completed shard after a signature check
  (typed refusals `extraction_resume_*` / `extraction_manifest_invalid`);
  `torchlens.dataset_extraction.load_extraction` reads the artifact back with metadata.
  Crash-safety is pinned by a hard-process-death test.
- BRAINPIPE (F20; every spelling DOCUMENTED-UNSTABLE): `torchlens.brainpipe` is the
  memory-planned whole-model extraction planner over the extraction artifact --
  `extraction_plan(model, probe_inputs, sites=, n_stimuli=, batch_size=, transform=,
  memory_budget=, engine="trace")` runs one unmeasured warm-up + TWO measured probes
  (batch b and 2b), fits per-site bytes linearly (batch-invariant sites detected, never
  total-scaled), executes the transform chain LIVE at the probe, and keys the plan to
  the full input signature. `plan.table()` prints one row per REQUESTED site (excluded
  rows carry reasons, never dropped) plus exact planned passes, the probes' peak PAIRS,
  the predicted run peak with basis disclosed, and budget arithmetic; `plan.run()` is
  the single-pass door onto `extract_dataset` (drifted signatures refuse
  `extraction_plan_signature_mismatch`; over-budget plans refuse
  `extraction_plan_over_budget` naming the batch-size lever; `engine!="trace"` refuses
  `extraction_engine_unsupported`). `export_npz` writes consolidated or per-stimulus
  npz (`compatibility="net2brain"` enforces their lexicographic glob+sort naming
  contract with a name-to-stimulus-id sidecar; `allow_pickle=False`, object arrays
  refuse `npz_export_invalid`); `parse_bytes("8 GiB")` is the human-unit budget
  reader. `Trace.forward_peak_memory_pair` is the session-time (live, resident) peak
  pair with the backend named -- per-capture resident scoping on Linux, so multi-capture
  sweeps read real peaks. `tl.repgeom.rdm` gains keyword-only `metric="manhattan"`,
  `compute_device=`, `row_chunk_size=`, `output_device=`, `dtype=`,
  `output="condensed"`, `input_kind="batched"` (the no-new-keyword call is
  bit-identical); `tl.repgeom.rdm_compare(a, b, method=)` is the DESCRIPTIVE
  Pearson/Spearman/Kendall tau-a over aligned strict upper triangles (diagonal
  excluded; inference points at rsatoolbox); geometry evolution verbs read TRANSFORMED
  payloads when raw is dropped and disclose `result.payload_basis`; `tl.stats.cka`/
  `CKA` accept `device=`/`dtype=`. The nine-method seam ledger is
  `docs/reference/brain_benchmarking_seam.md`. Lazy-BUFFER models (`LazyBatchNorm*`)
  now capture without pre-materialization (the entry refusal flipped off with the
  buffer-side completion).
- PREPROCESSING PROVENANCE + SITE INVENTORY (tvscope F21; every spelling
  DOCUMENTED-UNSTABLE pending the naming sprint): `torchlens.preprocessing` is the
  verified-capture story for the NeuroAI feature-extraction audience -- `resolve()`
  reads the preprocessing authority the user's OWN loader ships (torchvision Weights
  preset, HF image processor, timm data config, open_clip Compose, explicit mapping;
  `register_authority_adapter` extends the kinds; no-network tiers first, HF fetch
  disclosed, network failure -> unknown NEVER a lower tier's guess, and TorchLens
  hosts no recipe table); `audit()` is the field-level configuration comparison
  (match/mismatch/unknown-with-reason per field over resize/crop/interpolation/
  antialias/channel-order/value-range/mean/std; strict mode refuses mismatch AND
  unknown with distinct codes `preprocessing_audit_mismatch`/`_unknown`; opaque
  callables and partial metadata never become match; the demoted `imagenet_default`
  fallback never anchors `verified`); `diagnose()` is the opt-in tensor arm whose
  outcome vocabulary HAS NO match member. `ResolvedPreprocessing.status` derives the
  closed authority standing (authoritative/unverified_fallback/unknown); plain
  `tl.trace(transform=)` captures stamp honest `user_transform` provenance.
  `torchlens.inventory.list_sites` is the two-rung site inventory (free module
  listing; one disclosed-cost metadata-only forward) whose emitted selectors are
  CERTIFIED against the engine's matcher at emission -- every selector extracts keyed
  by the requested string or `resolve()` teaches a working spelling
  (`site_selector_ambiguous`/`_unknown`). `torchlens.features.as_matrix` is the ONE
  recorded stimuli-x-features shaping op (`flatten_features_v1`) shared by
  `bridge.rsatoolbox.dataset(source, site=)` (now PER-SITE) and the new
  `bridge.xarray.data_array`/`neuroid_assembly` adapters -- file and in-memory routes
  numerically identical. `tl.extract_dataset` gains the unambiguous INPUT path
  (`input_transform=`, distinct from the OUTPUT `transform=`; Resolution-backed runs
  stamp verified-by-construction and join the resume signature as a value-level
  `transform_pipeline` extension; bare callables disclose opaque and refuse resume
  continuation) plus `input_provenance=` (disk-only stamp), and every new manifest
  carries the `tl_input_preprocessing_v1` block (authority + audit + verdict +
  unknown reasons + versions; legacy artifacts read as unknown via
  `input_preprocessing_of`/`LoadedExtraction.input_preprocessing`; resume never
  grafts). Docs: `docs/neuroai/` (journey/loaders/BYO-alignment/demand ledger) +
  the rewritten `docs/migration/from_thingsvision.md`.
- Appliance packages `notebook` and `neuro` reserve extras boundaries and enforce
  import gating for their optional dependencies.
- NEURO TREATY DESK (F22; every spelling DOCUMENTED-UNSTABLE pending the naming
  sprint): `torchlens.neuro` is a thin landing zone, never a second home for the
  geometry/stats vocabulary (no re-exports). `tl.neuro.datasets(source, sites=None,
  pool=None, obs=None, stimulus_ids=None)` converts every STIMULUS-INDEXED saved site
  of a Trace / LoadedExtraction / extraction directory into per-site rsatoolbox
  Datasets through the core eligibility gate (buffers skipped with one summarized
  disclosure; explicit ineligible requests refuse; ledger on `.ledger`), with the full
  descriptor identity story (site/site_key/layer/pool ALWAYS incl. "flatten"/versions/
  dtype/casts -- f16/bf16 widen to f32, recorded) and the ALWAYS-written
  `tl_presentation_index` obs descriptor (rsatoolbox `calc_rdm(descriptor=)` SORTS
  rows; the index is measured recoverable at 0.1.5 AND 0.3.2). `tl.neuro.rdms` is ONE
  name, two type-distinct modes: source mode computes via tl.repgeom (default
  `metric="correlation"` -- deliberate divergence from repgeom's euclidean,
  cross-referenced; `measure_source="computed"` + `measure_convention` formula;
  anti-divergence-pinned to `rdm_evolution`) and matrix mode converts precomputed
  square matrices with `dissimilarity_measure=` REQUIRED, full validation, and never
  invents pattern identity; modes reject mixed args; `chunked=` reserved-refused.
  Legacy `bridge.rsatoolbox.dataset` is a delegation to the same code path ("neuroid"
  retired for `feature_index`; legacy integer `presentation` column kept). The
  teaching surface: redirect table (rdm/cka/mds/scree/pca/effective_dimensionality/
  procrustes_align/extract_dataset/load_extraction -> canonical spellings, no extra
  needed) + refuse-and-point table (noise ceilings undefined-not-unimplemented,
  crossnobis/mahalanobis with the silent-euclidean CAUTION, searchlight, kendall
  tau-a/tau-b trap, encoding models, CCA/SVCCA/PWCCA, non-metric MDS, model zoos,
  brain data), all AttributeError-lineage typed; `__all__`/`dir()` advertise only
  dependency-present names. Brain-Score alias branch LIVE (gate passed):
  `tl.neuro.activations_extractor`/`get_activations_fn` gated on brainscore_vision
  alone (extras split D17: neuro = rsatoolbox-only; new `brainscore` extra, filed).
  repgeom riders: `rdm(metric="gaussian")` (thingsvision bandwidth convention,
  credited), `rank_transform_rdm` + `rdm_node_spec(display="rank"|"percentile")`.
  `tl.stats.cka` contract published (linear, biased, CPU float64, matched rows, NaN
  on zero variance; pinned literals). brain_score bridge fixes: default `sites=`
  survives partially saved traces via the core stimulus-indexed filter (buffer rows
  never scored), documented ValueError contract live, `sites=` accepts module dotted
  paths. Doc of record: `docs/reference/neuro.md`.
