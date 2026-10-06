## Architecture

See `architecture.md` in `.project-context` for the older full map and
`state_of_torchlens.md` in `.project-context` for the current 2.x map. See subpackage
`AGENTS.md` files for per-module details.

Key entry points:
- Main capture: `torchlens/user_funcs.py` - `trace()`, `show_model_graph()`,
  `draw_backward()`, `validate_forward_pass()`
- Backend registry: `torchlens/backends/registry.py` - `BackendSpec`, `BackendName`,
  backend resolution, validation dispatch, and canonical backend errors.
- Site keys + grouping surface (L1 wave 0, DOCUMENTED-UNSTABLE): every
  retained op carries the portable structural-position key `op.site_key`
  (`site_key_v1`, minted at grouping time on every backend,
  policy-independent, the cross-capture bridging relation);
  `Layer.site_key` / `Layer.site_peers` / `Layer.shape_summary` are the
  Layer surface (typed refusals `layer_site_ambiguous` /
  `site_key_unavailable`); `grouping=` is the closed-vocabulary knob
  ("structural" default; others refuse typed pre-D1/S2) mirrored on
  `trace.grouping`, with the load-validated `grouping_policy_v1` stamp on
  `trace.grouping_policy` (C1-C8 coherence; degrade-settlement monotonic).
  Persisted rows are live as of the tlspec v8 bump (load-validated); join/fold
  machinery in `torchlens/postprocess/_site_key.py` / `_site_join.py` /
  `_grouping_stamp.py`.
- Quickstart input ladder (F17; spellings DOCUMENTED-UNSTABLE): trace/summary/
  render share one resolver -- real input XOR `input_size=` (flat tuple /
  sequence / keyword-to-shape mapping; seed-0 local synthesis; fail-closed
  dtype facts, `torchlens.quickstart.InputSpec` override) XOR zero-arg
  inference (the exact verified trace is consumed, never recaptured).
  Synthesized rungs persist `InputProvenance` inside the existing
  `Trace.input_preprocessor` KEEP field; `decode_output`/`output_table`
  refuse `nongold_semantics_unavailable` on synthesized values; first raw
  read warns once (`nongold_raw_value_read`). `torchlens.user_funcs.render`
  is the one-call facade (metadata-only pinned eval/no-grad, state-hash
  verified restore, `collapse="auto"` there only, detached `RenderResult`,
  never auto-opens). Lazy: gold/declared rungs materialize during the one
  captured forward with truthful totals; zero-arg refuses before probing.
  Doc: `docs/quickstart.md`.
- Summary rebuild (F08; spellings DOCUMENTED-UNSTABLE pending naming
  ratification): bare `trace.summary()` / `tl.summary(model, x)` render the
  auto view ladder (coalesced hybrid -> strictly folded module tree ->
  descending depth -> protected totals-conserving elision) under a derived
  48-body-row budget, returning a `SummaryReport` (str subclass) whose text
  is canonical byte-stable ASCII; unicode only at display boundaries through
  a 1:1 glyph table (`ascii == degrade(unicode)` CI-pinned; detection fails
  toward ASCII; `TORCHLENS_SUMMARY_STYLE` overrides). Identity partition is
  law (every param identity / op event owned by exactly one row at every
  depth/fold/filter/elision). The legacy spellings are removed: each refuses
  typed naming its successor (`REMOVED_SUMMARY_OPTIONS`/`REMOVED_SUMMARY_LEVELS`
  in `report/_summary_config.py`); `fma1` refuses `flop_convention_unavailable`
  when underivable. One-call input precedence: args XOR `input_size=` XOR
  zero-input (reuses `infer_input_shape`'s verified trace, synthesis
  disclosed). Result API: `render`/`print`/`details`/`to_pandas`/
  `to_markdown`/`to_html` + scalar raw ints; survives model/Trace teardown;
  `trace.provenance()` serves the relocated preamble byte-exact;
  `Trace._repr_html_` delegates to the summary HTML. Pins:
  `tools/derive_summary_pins.py`; ratio page:
  `docs/benchmarks/summary_performance.md`; docs: `docs/reference/summary.md`,
  `docs/migration/from_torchinfo.md`.
- Agent surface (both spellings DOCUMENTED-UNSTABLE pending naming
  ratification): `Trace.to_agent_json(max_ops=None)` emits the self-describing
  JSON-serializable `torchlens.agent_trace.v1` dump (capture honesty facts,
  counts, pass-qualified op rows with graph edges, module hierarchy, embedded
  navigation guide; payloads never inlined; `max_ops` truncation disclosed).
  `tl.report.explain(trace, max_tokens=N)` budget-prunes the text report by
  whole sections low-value-first with a disclosed `Truncation` section;
  capture-status honesty facts and partial-capture failure evidence never
  drop; refuses typed with `format="json"`. `torchlens.bridge.mcp` (extra
  `torchlens[mcp]`, mcp>=2.0) serves read-only MCP stdio tools over saved
  `.tlspec` artifacts + environment (doctor / api_map / load_overview /
  agent_dump / explain); no user-code execution, no mutation.
  Doc: `docs/for-ai-agents.md`.
- Neuro treaty desk (F22; every spelling DOCUMENTED-UNSTABLE pending the
  naming sprint): `torchlens.neuro` is a thin landing zone with NO
  repgeom/stats re-exports. `tl.neuro.datasets(...)` converts every
  stimulus-indexed saved site (Trace / LoadedExtraction / extraction dir)
  into per-site rsatoolbox Datasets through the core eligibility gate, full
  descriptor identity story (pool ALWAYS recorded, "flatten" default;
  f16/bf16 widen to f32 with the cast recorded) and the ALWAYS-written
  `tl_presentation_index` obs descriptor (rsatoolbox `calc_rdm(descriptor=)`
  SORTS rows; recoverability measured at 0.1.5 AND 0.3.2); sweep ledger on
  `.ledger`. `tl.neuro.rdms` = ONE name, two modes: source mode computes via
  tl.repgeom (default correlation; `measure_source="computed"` +
  `measure_convention`; anti-divergence-pinned to rdm_evolution), matrix
  mode requires `dissimilarity_measure=` and never invents pattern identity;
  mixed args refuse. Legacy `bridge.rsatoolbox.dataset` delegates to the
  same path ("neuroid" retired). Teaching surface: redirect + refusal
  tables, all AttributeError-lineage; `__all__`/`dir()` advertise only
  dependency-present names. Brain-Score alias branch LIVE (4.3 gate passed
  2026-08-29 vs brainscore-vision 2.3.22, py3.12/CPU):
  `tl.neuro.activations_extractor`/`get_activations_fn` gated on
  brainscore_vision alone. repgeom riders: `rdm(metric="gaussian")`
  (thingsvision convention, credited), `rank_transform_rdm`,
  `rdm_node_spec(display=)`. cka contract published (linear/biased/CPU-f64,
  pinned literals). brain_score bridge: default `sites=` survives partially
  saved traces, buffers never scored, module dotted paths accepted. Doc:
  `docs/reference/neuro.md`.
- Transformer pictures (`torchlens.tviz`; every spelling DOCUMENTED-UNSTABLE
  pending the naming session): typed session-only display records
  (`AttentionView`/`TokenScores`/`PredictionTrajectory`/`PredictionTable`/
  `TokenMetrics`/`ScoreDecomposition`/`CausalReceipt`/`Annotation`/`Artifact`)
  under matplotlib renderers resolved at CALL time (`tv_matplotlib_missing`
  prints the `torchlens[viz]` install command; zero-dependency SVG/HTML token
  emitters keep bare installs alive). Attention views read the semantic
  `pattern` facet with D6 mask provenance (SDPA call args / eager additive
  operand keyed on `finfo.min` / user metadata; zeros inference banned);
  renderers are rect meshes never `imshow`, zero `<image>` elements in SVG,
  pagination never silently drops. Causal receipt grids validate fires + both
  controls at construction and refuse without the measured joint effect;
  only a valid `CausalReceipt` mints the `intervention_effect` annotation
  kind, and effects never recolor attention cells. `score_decomposition`
  enforces the hard sum-to-score invariant (`tv_decomposition_unclosed`).
  Bridges: payload-bounded `circuitsvis_attention` + the `bertviz_tuple`
  parity oracle. FIX-H shipped in `tl.viz.render_heatmap` (nearest-neighbor
  cells, separate `row_labels=`/`col_labels=`, post-selection omission
  marker). Doc: `docs/reference/tviz.md`.
- Structure-only capture (DOCUMENTED-UNSTABLE; D8 GRANTED — weights-free
  admission live, F33): `tl.trace(model, x,
  capture=CaptureOptions(structure_only=True))` records structure +
  shape/dtype HYPOTHESES, never values; META-BUILT models are admitted
  under this contract with a uniform substrate (mixed cells refuse
  `structure_only_substrate_mismatch`); value consumers refuse typed
  through `torchlens.capture.structure_only` (contract:
  `docs/reference/structure_only_capabilities.md`); value-dependent
  branches refuse device-neutrally at the user's source line;
  `trace.discharge_against(real_trace)` corroborates/refutes hypotheses
  behind a comparable-twins preflight; `Trace.structure_evidence` is the
  persisted envelope; `tl.summary(meta_model, input_size=...)` and
  `Trace.check_plan(plan)` are the facade + audit verbs. Doc:
  `docs/reference/weightsfree_capture.md`.
- Selection algebra (L6; core spellings slate-ratified subject to D7, rest
  DOCUMENTED-UNSTABLE): `tl.Selection` (composable query AST) /
  `selection.resolve(trace)` -> `tl.ResolvedSelection` (frozen, trace-bound,
  session-only). Operators `| & - ~` + reflected, no `__xor__`; two-level
  denotation (family + elements, zero-mask entries retained); region
  producers (BaseSelector, RF box/gradient, FacetSpec, Op, Layer) implement
  `__selection__`; kinds ACT|PARAM|EDGE closed; refusals ride
  `SelectionError` (`selection_*` codes). Producers `tl.units`/`tl.params`/
  `tl.random_selection`; value producers `tl.top_k`/`tl.top_fraction`/
  `tl.threshold`/`tl.sign` (read the resolution trace's retained
  activations at resolve time, exact-as-set, `within=None` = every retained
  tensor site, unsaved payloads refuse `value_not_saved`, complex ordered
  comparisons refuse `value_criterion_invalid`); statistical producers
  `tl.dead`/`tl.saturated`/`tl.low_variance` (explicitly multi-sample:
  `samples=` iterable of >= 2 Traces, Bundle iterates; dead/saturated are
  dispositional `upper_bound` claims, low_variance is the `exact` sample
  statistic; the single-capture form is `sign(site,'zero')`); graph
  producers `tl.neighborhood(of, hops=, direction=)` / `tl.between(sources,
  sinks)` (structural position on the executed DAG: n-hop region and the
  source-to-sink influence sub-DAG; whole-site exact masks, family
  semantics, empty = disclosure; pure functions over the one
  `selection_graph._TraceGraph` substrate a future motif producer extends).
  `trace.between(sources, sinks)` presents the same region as a
  `TraceSlice` (frozen presenter, never a Trace: member ops, internal
  edges, EXPLICIT `boundary_in_edges`/`boundary_out_edges`, no
  save/replay/validate — `tl.save` refuses `slice_save_unsupported`;
  `__selection__` lifts it back into the algebra);
  `trace.subgraph(selection)` is the general slice door for any ACT region.
  statistic; the single-capture form is `sign(site,'zero')`); comparative
  producers `tl.changed`/`tl.top_changed` (subject-vs-ONE-reference
  directional delta, exact-as-set; structure mismatches and self-comparison
  refuse typed, never silently intersect; PARAM refuses — no capture-time
  weight payloads) and `tl.stable_across_passes`/`tl.pass_variance`
  (cross-pass range/variance on recurrent layers, pass-qualified, >= 2
  window passes per layer or `population_too_small`, masks land on every
  window pass-site); subspace producer `tl.subspace(within, basis, *,
  origin=, method=, dim=, tol=)` (direction/subspace SUPPORT-SET selection —
  set, not projection; mandatory basis provenance with sha256 digest riding
  `provenance.source` and do() audits; extent mismatch on the bound axis
  refuses `basis_dim_mismatch`, never broadcast/truncate; geometry-only
  resolution, unsaved sites resolve).
  Cross-run (stage 4a): `resolved.align_to(target)`
  re-binds ACT selections across runs on L1 site keys, same-policy captures
  only (`selection_alignment_invalid`, closed six-reason set); `do()` still
  refuses foreign resolved selections typed. Parameter substitution
  (DOCUMENTED-UNSTABLE): `fork.do(tl.params(name, mask=None), edit)` applies
  the edit "as if" the parameter were changed, replay engine only — the value
  each consumer sees is substituted at its derived occurrence address via the
  tier-(ii) edge-substitution store (`substitution_kind="param"`; cone
  recomputation RE-SPLICES every tier-(ii) entry (param, region, and edge kinds),
  so later pushes never silently revert any edit; chained param edits compose),
  the live `nn.Parameter` is never written, and
  rerun/set_only refuse `param_substitution_engine_unsupported`. Recurrently
  reused params (tied weights, multi-pass consumers) substitute at EVERY
  consumption via pass-qualified staging; what still refuses
  `param_substitution_occurrence_underivable` (fail-closed): nested container
  positions, released legacy captures, bare pass-ambiguous consumer
  spellings, and consumer inventories omitting a pass.
- Pass-qualified replay (decided 2026-08-17; refusal spelling
  DOCUMENTED-UNSTABLE): the replay/push engine keys cone traversal, the
  overlay, hook targets, and commits by pass-qualified op labels
  (`Op.label`, `label:pass`), so multi-pass edits touch exactly the
  addressed pass, recompute downstream passes, and commit every pass's
  record; strict multi-pass replay works and the spurious multi-pass
  `ControlFlowDivergenceWarning` is gone. A bare layer label naming a
  multi-pass layer refuses typed with a teaching message naming every
  pass-qualified spelling (`multipass_bare_label_ambiguous` on
  string/`tl.label` addressing; `selection_unresolvable` /
  `multipass_bare_label` on `tl.units`); single-pass bare labels stay
  accepted and `log[label].__selection__()` is the all-passes spelling.
- Edge substitution (L6 stage 3; DOCUMENTED-UNSTABLE): `trace.edges` is the
  dataflow edge family (EdgeUseRecords; requires an `intervention_ready`
  capture, refusal `edge_provenance_unavailable`; canonical occurrence
  address `(child_func_call_id, arg_kind, arg_path)`).
  `fork.do(edge_selection, edit)` replaces the value CONSUMED on the edge on
  the replay/push engine only (`edge_intervention_engine_unsupported`
  otherwise); the substituted value rides `Op.edge_substitutions` (+
  `edge_replacement_stamps`, `FireRecord.edge_address`; persisted as of
  tlspec v8) while capture truth stays unmodified, and uncorroborated
  entries FAIL validation (`edge_intervention_boundary`).
- Backward residuals (L9; DOCUMENTED-UNSTABLE): per-fire timing -- one clock
  (`perf_counter`), per-node keyed-LIFO pairing, stamps on the runtime
  `GradFnFired` event only; live-only `trace.grad_fn_fire_timings` (loaded
  traces refuse `grad_fn_fire_timing_unavailable`); as of the tlspec v8
  coordinated bump the persisted GradFnCall timing fields carry the per-fire
  `perf_counter` semantics, discriminated by the persisted
  `Trace.grad_fn_timing_provenance`. Checkpoint invocation
  tokens: classified non-reentrant `_checkpoint_hook` enters mint per-trace
  ordinal tokens; pack evidence count-only, unpack evidence backward-derived
  to L1 site keys; persisted `Trace.checkpoint_invocation_witness` with
  degrade flags D1-D6; the ambiguity refusal awaits a pending contract amendment and its
  identity-read accessors are unshipped until the amendment lands.
  Implicit-boundary: journal/scavenge/finalize split with the finalize guard
  in-routine (never inside an engine invocation), identity-checked
  engine-drain close callback, sync-point backstop always armed,
  `BackwardPassEnd.close_path` sidecar-only disclosure. Grouped floor:
  `trace.grad_fn_site_summary` per-site backward rollups (read-only L1
  consumption).
- Predicate runtime extension point (DOCUMENTED-UNSTABLE, S4 seam):
  `torchlens.ir.predicate_registry` — `PredicateProtocol` (one positional
  concrete `RecordContext`), `coerce_predicate(value, slot="save"|"halt"|"until")`
  (raw callables incl. `BaseSelector` returned BY IDENTITY; registered names
  via a slot-aware enforcing wrapper), `register_predicate(name)` (no
  user-object mutation, no loader-consulted attribute). Registry INERT until
  consumers adopt names. Contract: `docs/reference/predicate_runtime.md`.
- Sparse capture: `tl.record(model, x, save=...)` is torch-only in backend v1; it returns
  `Recording`, and `Recording.to_trace()` materializes full graph structure with explicit
  errors for unsaved payload reads. Forward exceptions default to
  `on_forward_error="raise"`; `on_forward_error="attach_partial"` attaches
  `exc.partial_recording` and re-raises, while `on_forward_error="return_partial"` returns a
  failed partial `Recording`. Failed partials set `status="partial_error"`, `failed=True`,
  string-only error metadata, `n_ops_completed`, and best-effort `last_event_*` fields.
  user-op failures exclude the failing call; TL-side capture failures may include a
  skipped/partial current-call event. Trace failed captures separately expose
  `exc.partial_log`, recoverable with `tl.partial.from_failed_capture(exc)`.
- Every capture product carries one settled typed outcome: `Trace.outcome` /
  `Recording.outcome` / `PartialTrace.outcome` return a frozen `CaptureOutcome`
  (COMPLETE / HALTED / ABORTED_NONFINITE / FAILED+phase / UNATTESTED / UNKNOWN),
  persisted as `_capture_outcome` (tlspec v7) and validated fail-closed at load.
  Capability gates N1-N5 branch on `tl.errors.CaptureOutcomeError.fields["code"]`;
  swallowed halt/nonfinite signals raise `tl.errors.StopSignalSwallowedError`.
  Doc of record: `docs/reference/capture_outcomes.md`.
- Queryable nonfinite record (spellings DOCUMENTED-UNSTABLE): `trace.nonfinite_ops`
  (pass-qualified labels of ops whose output held NaN/Inf) +
  `trace.nonfinite_coverage` (evidence basis and coverage counts). Default
  captures serve it from the memoized saved-payload scan at zero capture cost;
  `CaptureOptions(track_nonfinite=True)` opts into capture-time per-op checks
  covering unsaved ops, with device flags drained in one batch at the finalize
  seam (never a per-op CUDA sync). `raise_on_nan` is independent and unchanged.
- Attribution kit (DOCUMENTED-UNSTABLE): `torchlens.attribution` ships
  `integrated_gradients`, `occlusion`, `grad_cam`, and the display-only
  `overlay` renderer, all operating on Traces. Doc of record:
  `docs/reference/attribution.md`; glossary carries the unstable index.
- Kernel telemetry (optional CUDA/CUPTI adapter, DOCUMENTED-UNSTABLE):
  importing `torchlens.kernel_telemetry` installs `gpu_kernels` views over
  the ATen execution profile; rows persist as of tlspec v8 (`FieldPolicy.KEEP`)
  and counts are lower bounds wherever `mode_paused_interior` is non-empty.
  Doc of record: `docs/reference/kernel_telemetry.md`.
- Lazy decoration: `torchlens/backends/torch/model_prep.py:_ensure_model_prepared()` calls
  `wrap_torch()` and the belt/rescue stale-reference machinery
- Forward-pass orchestration: `torchlens/capture/trace.py`
- Postprocess: `torchlens/postprocess/__init__.py` current 26-step pipeline (declared
  contract keys `0`..`20` plus fractional inserts `11.5`/`11.75`/`15.5`/`16.5`/`17.5` in
  `postprocess/_contracts.py::POSTPROCESS_STEP_CONTRACTS`)
- Portable I/O: `torchlens/_io/bundle.py`, `torchlens/_io/tlspec.py`, `torchlens/io/__init__.py`
- Intervention: `torchlens/intervention/` plus top-level selector/helper aliases. Live
  `trace(intervene=...)` runs on torch, on the eager Paddle preview
  (`torchlens/backends/paddle/interventions.py`; forward-only, builtin helper adapters
  `zero_ablate`/`scale`/`add`/`replace_with`, corroborated validation carve-out), and on the
  eager TF preview (static-label, two-level writable layer, fail-closed site reachability;
  see invariant 15). `trace(halt=...)` runs on torch and Paddle; the remaining previews
  refuse typed.
- Visualization encoding channel (UNSTABLE naming, keyword-only): `Trace.draw(color_by=...)`
  fills op nodes from a sequential ramp (field name / scalar builtin / callable), dot-layout-only
  (AUTO forces dot; explicit rank refuses `encoding_requires_dot_layout`), legend-disclosed via
  the tri-state `show_legend` (`None`=AUTO channel-only legend, `True`/`False` historical; explicit
  `False` honored). Rolled multi-pass field sources resolve through the name-keyed allowlist in
  `torchlens/visualization/_encoding.py`; varying/first-pass-only sources stay unencoded with a
  legend note (honest-visuals tripwire), and unclassified sources refuse `encoding_source_invalid`.
  Wave-1 size channel `Trace.draw(size_by=..., scale=...)` (UNSTABLE, D4 default-applied):
  scalar field / `"dims"` (non-batch numel) / callable mapped to width/height MINIMUMS
  (`fixedsize=false`, area clamped 4x default, fonts never scale, strictly opt-in). Rolled
  multi-pass sources that cannot be certified single-valued refuse `size_by_rolled_varying`
  (size refuses where color degrades); `total_*` sums encode + aggregation legend line; the
  funnel drops NodeSpec width/height on image nodes; `scale=` without `size_by` refuses
  `scale_requires_size_by`. Wave-1 rank channel `Trace.draw(stack_by=...)` (UNSTABLE,
  strictly opt-in): annotation -> `rank=same` groups (`newrank=true`); `True`/`"auto"` is
  licensed by global pass_index monotonicity (else `stack_by_auto_underivable`), explicit
  field/callable bypasses with caption disclosure, rolled refuses
  `stack_by_requires_unrolled`, sibling ordering no-ops while stacking. Checked suppression
  (UNSTABLE `show_redundant_args`, DEFAULT-ON): labels omit constructor args PROVEN equal to
  captured shape dims (closed torch-family table; mismatch/unavailable stays visible;
  `show_redundant_args=True` shows all).
- Visualization: `Trace.draw(order_siblings=True)` applies a Graphviz-only verified
  sibling-ordering post-pass for forward unrolled graphs under the node cap.
  `Trace.draw(collapse="none"|"auto"|"max"|t, fold_repeats=None|True|False)` controls v2 smart
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

Common unified capture patterns:

```python
import torchlens as tl

torch_trace = tl.trace(
    model,
    x,
    backend="torch",
    capture=tl.options.CaptureOptions(intervention_ready=True),
)
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
relu_trace = tl.trace(model, x, save=tl.func("relu"))
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("streamed.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
# D18: eval-mode BatchNorm is runnable on the default live path (no value-changing
# buffer writes); train-mode buffer writers refuse with the typed BufferSinkRoutingError
# (RunnableErrorCode.BUFFER_SINK_ROUTING_MUTABLE, provisional/documented-unstable).
run_result = torch_trace.run(inputs=x, seed=42)
runnable_path = "architecture.tlspec"
tl.save(torch_trace, runnable_path, level="runnable", include_weights=True)
loaded_trace = tl.load(runnable_path)
verified = loaded_trace.run(inputs=x, seed=42, on_divergence="raise")
overview_svg = torch_trace.draw(collapse="auto", vis_fileformat="svg", vis_save_only=True)
module_scores = torch_trace.module_collapse_order

# Influence geometry follows the real captured DAG in both directions.
op = torch_trace["relu_1_2"]
rf_box = op.receptive_field.at((10, 10))
unit = op.receptive_field.center_unit(batch_index=0)
rf_check = op.receptive_field.check(unit)
outgoing_box = op.projective_field.at((10, 10))
layer_to_layer = op.receptive_field.at((10, 10), source=torch_trace.input_ops[0])
rf_table = torch_trace.receptive_fields(level="layer")
pf_table = torch_trace.projective_fields(level="layer")
# Gradient verification needs requires_grad inputs, backward_ready=True, and
# save_mode="reference"; verify().verdict is PASS / FAIL / INDETERMINATE.
armed_trace = tl.trace(model, x.requires_grad_(True),
                       capture=tl.options.CaptureOptions(backward_ready=True),
                       save_mode="reference")
armed_op = armed_trace["relu_1_2"]
armed_unit = armed_op.receptive_field.center_unit(batch_index=0)
rf_gradient = armed_op.receptive_field.gradient(armed_unit, retain_graph=True)
rf_results = tl.receptive_field.verify(armed_trace, units="center")
# show(gradient=True) recomputes the gradient WITHOUT retain_graph and frees the
# autograd graph -- call it last (or re-capture) if later backward passes are needed.
rf_image = armed_op.receptive_field.show(armed_unit, gradient=True)
# tl.validate(model, x, scope="receptive_field") captures an armed trace itself.
```
