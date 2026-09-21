# Torchlens project instructions

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

TorchLens logs backend-resolved execution into a `Trace`. The stable default is PyTorch
eager capture: run a normal forward pass, record operation metadata and activations, then
inspect the result. Torch function wrapping is lazy in 2.x: `import torchlens` keeps torch
clean, and the first torch capture calls `wrap_torch()` through model preparation. The
wrappers then stay installed until an explicit
`torchlens.backends.torch.wrappers.unwrap_torch()`.

## Install

```bash
pip install torchlens
pip install -e ".[test]"  # local development with test extras
```

Graphviz rendering needs Graphviz (`apt install graphviz` on Debian/Ubuntu). Optional
extras gate appliance and bridge namespaces; see `pyproject.toml` for the current list.

## Torch Version Compatibility

TorchLens supports torch 2.1 -> 2.12+ for eager torch capture. The declared floor stays
`torch>=2.1`; torch 2.0 may work best-effort through guarded fallbacks, but it is not a
declared support floor.

Every fragile torch-private-API probe or cross-version torch signature must route through
`torchlens/utils/_torch_compat.py`. Feature-detect the runtime capability; do not parse
`torch.__version__` for behavioral branching. Every graceful degradation must flip a named
`HAS_*` capability flag and be visible through the torch capability snapshot in
`torchlens.utils.doctor()` / `torchlens.compat.report()`.

## Model Menagerie (`menagerie/`)

`menagerie/` is a browsable atlas of 8,500+ catalog entries across ~3,600 neural-net architecture
families captured with TorchLens (8,533 rows / 3,637 families measured in this checkout,
2026-08-16; the full corpus incl. locally-validated additions lives on the menagerie machine):
a queryable catalog (`python -m menagerie.catalog stats|query|recipe`), 2,700+ hand-built historical
"classics" with no prior PyTorch implementation (`menagerie/classics/`, each trace-verified), and a
disk-safe graph renderer (`python -m menagerie.generate_menagerie`).

**To DISCOVER new families** — periodically, after each conference cycle, or **whenever a more
capable model becomes available** (a smarter auditor finds more) — use the canonical durable prompt at
**`menagerie/DISCOVER_MODELS.md`**. It is the reusable, adversarial "hunt exhaustively for architecture
families we missed" sweep: hostile framing, every-axis + non-English + newly-published coverage,
strict family-not-variant discipline, and exact instructions for folding finds into the catalog or
`classics/`. Dispatch cross-lab adversarial sub-hunters with it; seed candidates with the starter
`python -m menagerie.discover_crawler` (recent-arXiv harvester, meant to be extended).

### To ADD / BUILD found models into the roster (LOCKED — READ THE METHODOLOGY, DO NOT REINVENT)

**BEFORE adding ANY model, READ and FOLLOW `menagerie/METHODOLOGY.md` + `menagerie/UPDATE_RECIPE.md` +
`menagerie/HARVEST_SOURCES.md`.** The catalog's 8,400+ rows were built by ONE established process; do not
re-derive it. The build-bridge is: harvest the model's **REAL constructor** into a 9-column source row
(`name, zoo, constructor_call, input_shape, input_dtype, family, domain, era, notes`), run it through
`python -m menagerie.tools.tsv_to_jsonl` → typed JSONL record in `menagerie/data/master_catalog.jsonl`
(or `deferred.jsonl`), then `python -m menagerie.catalog build` and `python -m menagerie.validate_menagerie`
(renders/validates random-init in **grouped/fat envs** — the renderer amortizes dependency installs; use a
few fat pixi env-islands via `menagerie/envs.py`, NOT one env per model).

**IF SOURCE CODE EXISTS FOR A MODEL, USE THE REAL SOURCE — never write a from-scratch "approximation".**
That is SLOP and is forbidden (2026-07-01 incident: ~1029 such reimpls deleted, huge token/$ waste). The ladder
per candidate: (1) real class from an installed base lib IF the arch is unmodified; (2) the real repo code, run
it in a (fat) env / vendor its actual model file; (3) **faithful PORT** transcribed from the real repo code, only
if it genuinely can't be made to run; (4) **faithful REIMPLEMENT from a DETAILED description** (paper/thesis/etc.)
only when NO usable code exists at all — the triage's REIMPLEMENT class, still faithful, not a gist; (5) skip +
document ONLY if not even a detailed description exists (triage UNAVAILABLE) or it is not a real trainable NN.
`classics/` is ONLY for no-prior-code models (faithful ports + rung-4 reimpls). The triage's
SOURCE_AVAILABLE / ENV_SETUP / REIMPLEMENT / UNAVAILABLE / NOT_TRACEABLE class IS the signal for which rung — honor it.


`menagerie/` is a browsable catalog of 8,500+ entries across ~3,600 neural-net architecture
families captured with TorchLens (8,533 rows / 3,637 families measured in this checkout,
2026-08-16; the full corpus incl. locally-validated additions lives on the menagerie machine):
a queryable catalog (`python -m menagerie.catalog stats|query|recipe`), 2,700+ trace-verified
hand-built
historical "classics" (`menagerie/classics/`), and a disk-safe renderer (`menagerie.generate_menagerie`).
To expand/update the roster — periodically, after a conference cycle, or whenever a stronger model ships
— use the canonical durable adversarial prompt at **`menagerie/DISCOVER_MODELS.md`** (hostile framing,
every-axis + non-English + newly-published coverage, family-not-variant discipline, and how to add finds
to the catalog/`classics/`). Seed it with `python -m menagerie.discover_crawler`.

## Common Patterns

```python
import torchlens as tl

log = tl.trace(model, x, save=tl.func("relu"))
activation = log["relu_1_2"].out
print(log.summary())
print(tl.report.explain(log))
log.draw(order_siblings=True)  # default: verified sibling ordering for dot/unrolled graphs
log.draw(collapse="auto", show_containers=False)  # readability-targeted module overview
print(log.module_collapse_order[:10])
tl.release_model(model)  # restore whole-model pickle / torch.save serializability

# Influence geometry is lazy: the first property access solves the captured DAG.
op = log["relu_1_2"]
rf = op.receptive_field
box = rf.at((10, 10))
unit = rf.center_unit(batch_index=0)
check = rf.check(unit)
projective = op.projective_field.at((10, 10))
layer_to_layer = op.receptive_field.at((10, 10), source=log.input_ops[0])
table = log.receptive_fields(level="layer")
outgoing = log.projective_fields(level="layer")
# Arming the gradient tripwire needs requires_grad inputs, backward_ready=True,
# and save_mode="reference"; verify().verdict is PASS / FAIL / INDETERMINATE.
armed = tl.trace(model, x.requires_grad_(True),
                 capture=tl.options.CaptureOptions(backward_ready=True),
                 save_mode="reference")
armed_op = armed["relu_1_2"]
armed_unit = armed_op.receptive_field.center_unit(batch_index=0)
gradient = armed_op.receptive_field.gradient(armed_unit, retain_graph=True)
validated = tl.receptive_field.verify(armed, units="center")
# show(gradient=True) recomputes the gradient WITHOUT retain_graph and frees the
# autograd graph -- call it last (or re-capture) if later backward passes are needed.
overlay = armed_op.receptive_field.show(armed_unit, gradient=True)
# tl.validate(model, x, scope="receptive_field") captures an armed trace itself.
```

Use the unified predicate surface for selective capture, windowed saves, interventions, and
storage:

```python
conv_before_relu = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
log = tl.trace(
    model,
    x,
    save=conv_before_relu,
    lookback=4,
    lookback_payload_policy="detached_raw",
)

ablated = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)

disk_log = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
full_structure = recording.to_trace()

# Selection algebra (L6): compose regions, resolve explicitly, edit with do().
log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
u1, u2 = log["relu_1_2"], log["conv2d_2_3"]
inter = u1.receptive_field.at((3, 3)) & u2.receptive_field.at((5, 5))  # a Selection
resolved = inter.resolve(log)              # frozen, trace-bound, session-only
fork = log.fork()
fork.do(inter, tl.zero_ablate())           # edit-then-scatter: only masked elements
fork2 = log.fork()
fork2.do(tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(fork2), tl.patch_from(log))
fork3 = log.fork()
fork3.do(tl.params("head.weight"), tl.scale(0.5))  # "as if" the weight changed, replay-only:
# every consumption is substituted; the live nn.Parameter is NEVER written.
print(fork.intervention_audit[-1])         # query repr + resolve digest + relations
```

Use `backend=` only when the backend is intentionally part of the test or example:

```python
torch_trace = tl.trace(model, x, backend="torch")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
assert torch_trace.backend == "torch"
```

Before debugging wrapper-specific failures, run:

```python
print(tl.compat.report(model, x).to_markdown())
```

## Current 2.x Surface

- Top-level `torchlens.__all__` has 119 names: capture, save/load, intervention,
  selectors, helper transforms, observers, validation, and the three main log classes.
- Relation accessors on FINISHED traces return IMMUTABLE views (authorized public type
  break, JMT 2026-08-12): label sequences (`op.parents`, `op.children`, `op.modules`,
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
- EPISODE CAPTURE (torch-only; every spelling DOCUMENTED-UNSTABLE): one wrapped multi-step
  generation run is ONE product — `tl.trace(episode_root, x,
  episode=tl.options.EpisodeSpec(stepped_module=model, n_steps=N))` stamps
  `capture_kind=episode` and lands the per-step status ledger (header + rows:
  complete/interrupted/absent, emitted tokens from the root output, managed-RNG
  entry_seed) at `trace.annotations["episode"]` after settlement. The ledger is a
  DISCLOSURE, never a settlement authority (outcome vocabulary and N1-N5 unchanged);
  loads validate fail-closed (illegal attachment refuses `episode_ledger_without_declaration`,
  geometry violations quarantine `episode_ledger_incoherent`); the annotations key AND the
  Bundle `member_relations` key persist plainly as of the tlspec v8 coordinated bump.
  DIAGNOSTIC-TIER cost, superlinear (gpt2-124M CPU: N=20 79 s / N=100 657 s, 947 MB, 5.4 GB
  RSS) — tens of steps, never hundreds; guarded-fast (`trace.run(fast=True)`) is the default
  engine and must reproduce wrapped tokens bit-exactly (pinned). Teacher forcing
  (`forced_tokens=`) is a disclosed NON-VERIFYING mode; escalation re-runs the WHOLE episode
  wrapped with `escalated_from`/`reason`/`fidelity_basis` disclosed (E-A3: mismatch records
  `diverged`, never a settlement input); declared unsnapshotable state refuses at declaration
  time (`episode_state_unsnapshotable`). Bundles gain the optional S6 member-relation table
  (`member_relations=`, `Bundle.relate`, `Bundle.derive_episode_status` — a derived fold,
  never Bundle-level settlement; mutators cascade explicitly or refuse typed). Doc of record:
  `docs/reference/episode_capture.md`.
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
  monotone schedule (`0.0 == "none"`, `1.0 == "max"`). `auto` is the first schedule point whose
  visible count enters the readable band, but its implementation remains frozen for compatibility.
  `None` preserves defaults (`"none"` has no run folding; `"auto"`/`"max"` use band-pressure
  folding), `True` folds eligible repeated runs even with `collapse="none"`, and `False` disables
  run folding. `collapse="max"` may emit segment boxes; `(xN)`, ellipsis, and segment labels must
  stay honest about hidden calls or ranges. `Trace.collapse_plan(mode=...)` returns the diagnostic
  plan, and `Trace.collapse_schedule()` returns the float schedule metadata. Smart collapse
  has a preflight compute ceiling `COLLAPSE_OPTIMIZER_MAX_OPS` (2000 ops,
  `torchlens.visualization.collapse_optimizer`): above it the optimizer DECLINES with a
  `TorchLensWarning` -- `draw(collapse="auto"|"max")` renders uncollapsed,
  `Trace.collapse_plan()` refuses typed (`collapse_plan_unavailable`), and
  `collapse_schedule()` degrades to its single full-graph step; reduce the rendered graph first (`module=` focus,
  `vis_call_depth`, rolled mode).
- Smart-collapse metadata is computed at access time: `Module.collapse_score`,
  `Trace.module_collapse_order`, and `Trace.collapse_order(weights=..., mode=...)`. These are
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
  until postprocess eviction (the bare flat `layers_to_save=` kwarg is a deprecated alias and
  warns). Unmeasurable auto
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
  and a witness-VERIFIED capture suppresses the rescue. `wrap_torch(patch_policy=, patch_modules=)`
  are deprecated no-ops. Full contract: `docs/migration/scoped_detached_patching.md`.
- `torchlens._io` and `torchlens.io` own portable `.tlspec` save/load helpers. Manifest
  schema v2 is backend-aware; non-torch preview bundles may be audit-only or metadata-only.
  Rehydration floor: artifacts older than torchlens 2.33 (`tlspec_version` 6) refuse to load
  with the typed `tl.errors.ArtifactVersionBelowFloorError` (drop-not-resurrect; the legacy
  field-alias ladders are deleted). Legacy 2.16 intervention specs remain loadable — the
  floor covers Trace rehydration only.
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
  consumers to typed refusals. Meta-materialized models (HF
  `device_map='meta'`) still refuse at the entry gate — admission is decision
  point D8, unruled. Human surfaces (summary/profile/explain) carry the
  structure-only hypothesis banner.
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
- PARAMETER SUBSTITUTION (param-operand, JMT-ruled 2026-08-17, supersedes the
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
  RE-SPLICES param-kind entries (edge-kind entries keep shipped no-re-splice
  semantics; later pushes never silently revert the edit), and the SAME
  validation boundary (`edge_intervention_boundary`; uncorroborated FAIL).
  Replay/push engine ONLY: rerun/set_only refuse
  `param_substitution_engine_unsupported`. The audit record (kind `PARAM`)
  discloses "substituted at consumption ... live parameters unchanged".
- PASS-QUALIFIED REPLAY (JMT-ruled 2026-08-17; refusal spelling
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
  naming-session/E-L9-4 routing): PER-FIRE TIMING -- every hooked grad_fn
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
  ambiguity REFUSAL is S2-authored (R-L9-1 filed) -- identity-read accessors
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
- AGENT SURFACE (both spellings DOCUMENTED-UNSTABLE pending naming ratification):
  `Trace.to_agent_json(max_ops=None)` emits the self-describing JSON-serializable
  `torchlens.agent_trace.v1` dump (capture honesty facts, counts, pass-qualified op rows
  with graph edges, module hierarchy, embedded navigation guide pointing back at the live
  public surface; payloads never inlined; `max_ops` truncation disclosed, never silent).
  `tl.report.explain(trace, max_tokens=N)` budget-prunes the text report by whole
  sections low-value-first with a disclosed `Truncation` section; capture-status honesty
  facts and partial-capture failure evidence never drop, and `max_tokens` refuses typed
  with `format="json"`. Doc: `docs/for-ai-agents.md`.
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
  `brainscore_vision`, Python >= 3.11; pinned to the 2.3.22 wheel source, stub-tested,
  UNVERIFIED against a running install). `torchlens.bridge.mcp` (extra `torchlens[mcp]`,
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
- Appliance packages `notebook` and `neuro` reserve extras boundaries and enforce
  import gating for their optional dependencies.

## Anti-Patterns

- Do not log `torch.compile`, TorchScript, or `torch.export` artifacts; log the eager
  source module.
- Do not expect `torch.func` / functorch transforms to expose per-element internal ops;
  TorchLens captures transform calls as boundary nodes with provenance edges.
- Do not run captures concurrently across Python threads or worker processes.
- Do not expect fused kernels to expose hidden internal tensors.
- Do not put opaque callables in portable artifacts unless audit-only behavior is
  acceptable.
- Do not add new top-level API names casually; use submodules and deprecation shims.

## Validation Integrity (LOCKED PRINCIPLE — never violate)

The `validation/` pipeline (forward replay, backward checks, metadata invariants) is a
**TRIPWIRE, not a formality.** Its entire purpose is to CATCH capture bugs — ops that
weren't traced, wrong replay inputs, broken metadata, silent corruption.

**NEVER weaken, loosen, exempt, broaden a tolerance, or skip a validation check / invariant
to make a test pass.** A validation failure is the system *working*: ROOT-CAUSE it and fix
the actual bug. Silencing a failing check defeats the entire point and lets exactly the kind
of silent breakage validation exists to prevent ship undetected.

The ONLY legitimate exemption is behavior that is **correct by design and provably outside the
check's contract** (e.g. a user-injected intervention tensor genuinely has no traceable
function to replay). Even then the carve-out must be NARROW (only the intended case) and must
NOT mask the unintended case — e.g. an auto-synthesized placeholder op appearing during PLAIN
capture is a capture bug, and validation must STILL fail on it.

**Incident (2026-06-02):** `test_mistral` / `test_audio_vits` emitted functionless
`interventionreplacement` placeholder ops during plain tracing — a real capture gap (ops
TorchLens failed to wrap). An exemption was added to the metadata invariant to pass them. That
was backwards: it disarmed the tripwire. The correct fix is to make capture actually trace those
ops so no placeholder is synthesized during plain capture; any replacement-op exemption must be
scoped to GENUINE user interventions only.

## Keep the glossary + docs in lockstep with code (LOCKED)

The glossary is the **canonical** API spec (vault `brain/projects/torchlens/reports/<date>-glossary-vN/torchlens_glossary.md`); code conforms to it (spec-drives-code). A rename is not *done* until the docs match too:

- **Rename / add / remove any PUBLIC name** (dataclass field, `@property`, method, top-level `tl.*` name, kwarg) → in the SAME change, update: (1) the **glossary** entry (canonical), (2) this `AGENTS.md` examples, (3) the audit notebooks (`notebooks/audit/`) and `examples/` that use it.
- A change that touches code but leaves the glossary/docs stale is **INCOMPLETE.** This is exactly how the v7 `memory → activation_memory` gap and the stale `log_forward_pass`/`vis_opt` examples slipped through.
- After a rename/conformance sprint: re-file the updated glossary to the vault (it supersedes the prior dated version), and confirm a `grep` of every old name is clean across `torchlens/`, `tests/`, `examples/`, `notebooks/`, AND the glossary itself.

### Trusted custom callable imports

Intervention-spec loads tolerate foreign `custom` callable keys for safe structural and metadata
analysis without importing their modules. Resolution for execution denies those foreign imports by
default because module imports execute top-level code. Trusted execution may opt in with
`trust_custom_callables=True`; prefer the narrower
`allowed_custom_callable_modules={"my_trusted_module"}`, which remains enforced even alongside broad
trust. TorchLens-owned `torchlens.*` custom callables and the fixed `torch`, `torch.Tensor`,
`torch.nn.functional`, and `operator` namespaces always resolve.

### Sparse runnable state binding

Loaded sparse runnable traces accept `trace.load_state_dict(sd)` to strictly validate and atomically
stage canonically named parameter and persistent-buffer tensors. The method never executes the DAG
or writes tensor payloads into the sparse descriptor. Run preflight selects explicitly staged user
state, then optional embedded capture state, then the versioned
`torchlens_role_init_v2` fallback (degenerate-total: empty slots consume zero RNG); random reports
name every initialized slot.

`tl.save(trace, path, level="runnable", include_weights=True)` bundles the full capture-time
`state_dict` (all named parameters plus persistent buffers) as the separate, schema-versioned
`state_dict_v1` blob family. The default is `include_weights=False`, so the sparse core stays
tensor-value-free. Load validates embedded state through the same strict binder; run reports
`embedded_capture_state`, never a reconstructed model, and a later `load_state_dict()` overrides it.
Used NON-persistent buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family
(declared state, not gated on either include flag; the save discloses it). State ALIAS topology is
declared (r37): repeated live object identity (tied weights, double-registered buffers) becomes a
shared alias group staged as ONE allocation; distinct-object overlapping or unprovable state
topology refuses at save with `state_alias_topology_unsupported`. Payload blobs keep their
`map_location` transport placement; readiness capability-checks recorded slot devices without
allocating, and one atomic run-preparation staging pass moves all state families to their recorded
devices (a CUDA artifact on a CPU-only host loads for analysis and refuses `.run()` typed).

`tl.save(trace, path, level="runnable", include_activations=True)` independently archives exactly
the activations already retained by the capture-time `save=` decision, including retained raw and
transformed outputs, as `selected_activation_v2` with physical `InputAttestationFingerprint`
eligibility records. Load exposes them through
`trace.archived_activations` for inspection. They never seed the sparse DAG. On original-input runs
with embedded or capture-equivalent staged state, recomputed saved raw slots are compared by exact
bytes and report `attested`; the first mismatch raises `numeric_attestation_failed` and rolls back.
Changed-input (logical or physical), random-state, non-equivalent-state, and
nondeterministic-capture-context runs report `not_applicable`; `attested` always implies
`verified`.

### Sparse runnable execution

`trace.run(inputs=x, seed=...)` is the provider-neutral execution spelling. A live Trace delegates
on a fork to the existing `save_new_outs` fast capture path. The live refresh projector's
buffer-sink refusal is TRAINING-MODE AWARE (D18, explicit JMT ruling): eval-mode BatchNorm
(every buffer sink carries derived write evidence `buffer_value_changed=False` with agreeing
mode claims) is refresh-eligible and runnable on the DEFAULT path, while any value-changing
buffer write (train-mode running stats, counters), unproven (`None`) evidence, a mode claim
contradicting the evidence, or a value-changing write in the refreshed rerun's own journal
refuses with the typed `BufferSinkRoutingError` carrying
`RunnableErrorCode.BUFFER_SINK_ROUTING_MUTABLE` (provisional spelling, documented-unstable;
pinned "computational graph changed" message term preserved). A loaded sparse Trace binds cloned
input leaves plus staged/random state and executes its resolved taken-path DAG under
`pause_logging()`. Both return `RunResult(output, trace, report)` and leave the source Trace
unchanged. Analysis-only loads raise typed `run_capability_unavailable`. Stage 5 populates
`report.path_faithfulness`; Stage 6 enforces the three-layer honesty transaction. Divergence raises
and rolls back by default; `return_diverged` is the sole opt-in and returns a monotonic poisoned
Trace refused by validation, export, faithful comparison, and path-assuming intervention chaining.
Incomplete witness coverage is `unverifiable`, never `verified`; numeric attestation is
`not_applicable` for sparse-only or ineligible activation-payload runs. Model outputs with ZERO
tensor leaves remain subject to the output-container rules below. `trace.run(inputs=x, fast=True)`
is the explicit guarded static-loop exception to source immutability: live traces execute native
`forward()` with targeted module hooks (and scoped functional collection only for an explicit
functional `save=` predicate); loaded traces perform one ordinary verified run, then reuse staged
state, compiled binders, and one result Trace. Every fast iteration retains input, output-shape/
dtype, call-path, and control-witness guards and always raises on divergence. Default `fast=False`
keeps the full transactional validation/attestation path. The fast-mode handle
`Trace._fast_run_session` is a session-time `FieldPolicy.DROP` field: it is ordered in
`MODEL_LOG_FIELD_ORDER` under a private name but never survives save/load, so it joins
`measure_python_peak_memory`, `save_budget`, and `distributed_witness` in the session-time class
(the private-named ordered DROP fields are ledgered with reasons in
`tests/test_schema_lockstep.py::PRIVATE_ORDERED_DROP_FIELDS`). Model outputs with ZERO
tensor leaves (all-literal trees, literal roots, empty containers) and namedtuple/mapping/
registered-container outputs carrying extra per-instance state refuse at save
(`missing_output_container_contract`; one per-kind capability table governs capture proof, save
refusal, and load-time recompute). Host nondeterminism beyond the two replayable global engines
-- RNG instances (incl. outside-held NumPy Generators, witnessed by a chained
`sys`/`threading.setprofile` receiver classifier + a cheap model-attribute state digest -- NO
process-wide gc scan; bare `_random.Random`; unseeded-construction entropy via the
`randbits` alias), `SystemRandom`/`secrets`, OS entropy, `uuid4`, the `default_rng` factory, and
the full clock family (`time.*` counters, `localtime`/`strftime`/`datetime.now`/`date.today`,
`os.times`/`getrusage`) -- ceilings every replay permanently (`unverifiable` + `not_applicable`);
monitor uncertainty (install/chain/restore/inventory failure) downgrades completeness, never reads
as no-consumption. A realistic pre-existing-thread draw from a persistent numpy generator is
witnessed by the setprofile classifier / model digest; only an externally-held generator drawn on
a pre-existing (non-hooked) thread is a documented residual (a benign background thread never
ceilings a capture). The loaded-sparse and live-refresh
providers settle through ONE finalizer (identical verdict class): a live opaque-container output
is `unverifiable` + poisoned (never a wrongly-blessed bare tensor), a parse-refused descriptor
degrades EVERY payload family to analysis-only with its typed diagnostic intact, and an
inexecutable divergent input raises `PathDivergenceError` (not `RuntimeSignatureDriftError`). Structseq
reconstruction trust keys on the RESOLUTION AUTHORITY (`spec.type_module == "torch.return_types"` +
identity re-resolution), never the spoofable `__module__` attribute; a namedtuple TYPE that can
carry instance state refuses at save even with an empty instance. Persisted execution-context values validate at parse
time against closed vocabularies (`context_field_invalid`); the recorded default device is entered
as a scoped `with torch.device(...)` context, never via `set_default_device`.

Runnable descriptors are `sparse_recorded_taken_path_v2` (call recipe
`non_tensor_args_tensor_slots_context_and_obligations_v3`): per-call `CallExecutionContext` and the
capture-scoped `AmbientExecutionContext` are REQUIRED and EXPLICIT, restored at replay or refused
typed; a legacy v1 artifact loads analysis-only with a typed readiness refusal (absent context is
never defaulted).

The witness-strip class is closed structurally (r71 A): `WITNESS_FAMILY_REGISTRY` v2
(`witness_family_registry_v2`) covers EVERY verdict-steering witness family (the four direct
control kinds + shape families + two claim-only families), each row naming its independent
replay-structural anchor. Every replay item that steers a verdict is a typed obligation stamped
on its OWNING replay record (`control_obligations`/`control_dependencies` on calls,
`host_escape`/`inert_sink` on slots, `captured_requires_grad`/`captured_grad_fn`/
`host_escape_disposition` on state bindings, the REQUIRED `input_boundary` record), discharged by
an exact witness XOR a typed `WitnessCoverageGap`; `witness_completeness` is DERIVED from the gap
ledger (redundant assertion, never authority) and the required-witness inventory is a redundant
mirror. No record deletion can improve a verdict; the ONE out-of-scope boundary is coherent
reauthoring -- byte-for-byte an honest capture of a weaker program, whose VERIFIED is TRUE against
that program's oracle 1 -- a documented threat-model scope statement, not an open residual.
Cluster C: instance-state inspection is fail-closed (`inspect_instance_state`; a
property/descriptor-shadowed `__dict__` or a custom `__getattribute__`/`__getattr__` on a
declared-schema container refuses `instance_state_uninspectable` without running the hook).
Cluster B: slice/composite-literal components are classifier-first (a semantic scalar can never
launder through a `slice` component). Cluster D: reserved input-path sentinels (the whole `\x00`
namespace) are escaped by the key codec so a real dict key equal to a marker round-trips.

The declared state model is the capture-time `state_dict` (named parameters plus persistent buffers)
PLUS the capture-time values of used non-persistent buffers (the required
`runnable_nonpersistent_buffer_v1` family), and the taken-path DAG. `verified` is faithfulness
against a *fresh* live-model run from that state on
the given inputs (oracle 1) — NOT reproduction of a specific already-run instance's later, differently
branched forwards. Hidden non-`state_dict` Python state mutated *across* forwards (an arbitrary
attribute, or a retained activation-derived handle — a kept `numpy()`/`untyped_storage()` view or a
detached tensor) is out of scope and stays `verified`; the "divergence" exists only against re-running
the same mutated instance, never oracle 1. In scope and witnessed identically for activations,
parameters, and buffers: a host write *within* the captured forward into captured storage — caught by
whole-storage byte comparison + per-consumption sampling, with the raw `data_ptr()` surface fail-closed
to `unverifiable`, and a read-only exposure staying `verified`. Full boundary in
`docs/reference/runnable_tlspec_contract.md` section 11.

The frozen runnable enums live in `torchlens.runnable`: `ReadinessStatus`, `RunProvider`,
`StateSource`, `PathFaithfulness`, `DivergencePolicy`, `NumericAttestationStatus`, and
`RunnableErrorCode`. Public code branches on these values or the structured report, not exception
text. The exhaustive error vocabulary and release threshold are maintained in
`docs/reference/runnable_tlspec_contract.md`.

## Internal notes stay PRIVATE (LOCKED — this repo is PUBLIC)

`johnmarktaylor91/torchlens` is a **public** GitHub repo. Internal planning, riffing, sprint
specs, adversarial reviews, STATE/SUMMARY files, and the working task tracker are **JMT's eyes
only** and must NEVER be committed.

- **Private (gitignored, never commit):** all of `.research/`, and `.project-context` EXCEPT the
  two whitelisted curated docs. The agent task tracker `todos.md` in `.project-context` and the
  agent-facing `torchlens_glossary.md` in `.project-context` (canonical lives in the vault) are private.
- **Public (the only tracked `.project-context` files):** `architecture.md`,
  `state_of_torchlens.md`. The user-facing glossary is `docs/reference/glossary.md` (shipped)
  — a separate, curated artifact, NOT the agent copy.
- **Enforcement:** `.gitignore` excludes them and a `no-internal-notes` pre-commit hook
  (`.pre-commit-config.yaml`) HARD-FAILS any commit that stages a private path. Never `git add -f`
  to bypass it; never `git rm` the local files (they are your working notes). Long-form
  human-readable reports go to the Obsidian vault, not the repo.

## Testing Tiers

```bash
ruff check . --fix
mypy torchlens/
pytest tests/<files for the code you touched> -x --tb=short     # per-step gate: targeted suites (seconds-minutes)
pytest tests/ -m smoke -x --tb=short                            # commit-level gate (~20 min; measured 2026-08-13)
pytest tests/ -m "not rare and not slow and not heavy" -x --tb=short  # mid backstop (heavy = 5-20s tests)
pytest tests/ -m "not rare and not slow" -x --tb=short  # phase-boundary backstop; public API/boundaries
```

Tiers by cost: `smoke` selects ~4.9k tests (4,920/12,840 collect-only, measured 2026-08-16).
The census tip was adb3d450 (the fixwave-5 settle; tri-lab b2 probe).
The last instrumented `--durations=0` smoke wall measurement (measured 2026-08-13, 4-core dev
box under parallel sprint load) took 1194s (~20 min) against the then-selected ~3.2k tests
(~500s on a quieter box earlier the same sprint); budget at least that at today's ~40%
larger selection. Smoke is NOT
sub-minute and NOT a per-step gate — per-step verification is the targeted test files for
the code touched; smoke is the commit-level gate, `not rare and not slow and not heavy`
the mid backstop, and `not slow` the phase-boundary backstop. Partition: `smoke` tests
must each run <5s measured, `heavy` carries the 5-20s tests, `slow` the >20s ones.
`tests/test_marker_lint.py` enforces it: combining `smoke` with `heavy`/`slow`/`serial`/`rare`
fails (markers are additive — the test would still run under `-m smoke`), and the runtime
tripwire holds smoke/unmarked tests to budget 5s and heavy 20s (load-scaled 1x-4x plus a 2s
boundary-noise grace, charged on min(wall, cpu)) — an offender fails the session it ran in. `pytest -n auto` requires the
optional `pytest-xdist` plugin, which is not installed by TorchLens's declared test extra.
When xdist is installed separately, measure before relying on it: torch intra-op threads can
oversubscribe workers, and fixture/import setup may dominate.

Use `pytest.importorskip()` for optional migration dependencies. Keep tests
deterministic and run documentation examples when they are meant to be executable.

## Project Overview

TorchLens extracts activations and computational graph metadata from backend-resolved
captures into `Trace`, `Layer`, and `Op` objects. The stable default is PyTorch eager
capture: it lazily wraps PyTorch functions with toggle-gated wrappers on first torch
capture, runs forward passes with the logging toggle enabled, and records operations.

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
- Structure-only capture (DOCUMENTED-UNSTABLE, D8-default):
  `tl.trace(model, x, capture=CaptureOptions(structure_only=True))` records
  structure + shape/dtype HYPOTHESES, never values; value consumers refuse
  typed through `torchlens.capture.structure_only` (contract:
  `docs/reference/structure_only_capabilities.md`); value-dependent branches
  refuse device-neutrally at the user's source line;
  `trace.discharge_against(real_trace)` corroborates/refutes hypotheses.
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
  tier-(ii) edge-substitution store (`substitution_kind="param"`, re-spliced
  on cone recomputation), the live `nn.Parameter` is never written, and
  rerun/set_only refuse `param_substitution_engine_unsupported`. Recurrently
  reused params (tied weights, multi-pass consumers) substitute at EVERY
  consumption via pass-qualified staging; what still refuses
  `param_substitution_occurrence_underivable` (fail-closed): nested container
  positions, released legacy captures, bare pass-ambiguous consumer
  spellings, and consumer inventories omitting a pass.
- Pass-qualified replay (JMT-ruled 2026-08-17; refusal spelling
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
  degrade flags D1-D6; the ambiguity refusal is S2-authored (R-L9-1) and its
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
  monotone schedule (`0.0 == "none"`, `1.0 == "max"`). `auto` is the first schedule point whose
  visible count enters the readable band, but its implementation remains frozen for compatibility.
  `None` preserves defaults (`"none"` has no run folding; `"auto"`/`"max"` use band-pressure
  folding), `True` folds eligible repeated runs even with `collapse="none"`, and `False` disables
  run folding. `collapse="max"` may emit segment boxes; `(xN)`, ellipsis, and segment labels must
  stay honest about hidden calls or ranges. `Trace.collapse_plan(mode=...)` returns the diagnostic
  plan, and `Trace.collapse_schedule()` returns the float schedule metadata. Smart collapse
  has a preflight compute ceiling `COLLAPSE_OPTIMIZER_MAX_OPS` (2000 ops,
  `torchlens.visualization.collapse_optimizer`): above it the optimizer DECLINES with a
  `TorchLensWarning` -- `draw(collapse="auto"|"max")` renders uncollapsed,
  `Trace.collapse_plan()` refuses typed (`collapse_plan_unavailable`), and
  `collapse_schedule()` degrades to its single full-graph step; reduce the rendered graph first (`module=` focus,
  `vis_call_depth`, rolled mode).

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
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
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

## Conventions

- **This repo is PUBLIC; internal notes stay private.** Never commit `.research/` or
  `.project-context` (except `architecture.md` / `state_of_torchlens.md`) — they are
  gitignored and a `no-internal-notes` pre-commit hook hard-fails on them. Never `git add -f`
  to bypass. See AGENTS.md "Internal notes stay PRIVATE (LOCKED)".
- Conventional commits: prefer `docs(scope):`, `chore(scope):`, `test(scope):` for
  non-release changes; never use major-bump markers casually.
- TorchLens host-object metadata lives under `obj._tl`; sub-fields are snake_case and
  new metadata should extend a `TorchLensMeta` subclass rather than adding `tl_*` attrs.
- `_raw_` prefix for pre-postprocessing state; `_final_` for post-processed state
- FIELD_ORDER constants in `constants.py` define canonical field sets; update both class
  fields and constants when adding fields
- NumPy-format docstrings on all functions
- Type hints on all functions
- Import order: stdlib -> third-party -> local (enforced by ruff)
- Line length: 100
- `tl.receptive_field` is lazy; entity-level `receptive_field` / `projective_field` siblings
  pair with `Trace.receptive_fields()` / `Trace.projective_fields()` tables.
- EPISODE CAPTURE (torch-only, spellings DOCUMENTED-UNSTABLE): `tl.trace(episode_root, x,
  episode=tl.options.EpisodeSpec(stepped_module=model, n_steps=N))` captures one wrapped
  multi-step generation run as ONE product with a per-step status ledger at
  `trace.annotations["episode"]` (disclosure, never a settlement authority; persists
  plainly as of the tlspec v8 coordinated bump, load-validated fail-closed). DIAGNOSTIC-TIER: cost is superlinear
  in step count — tens of steps, never hundreds. Bundles carry the optional S6
  member-relation table (`member_relations=`, `Bundle.relate`,
  `Bundle.derive_episode_status`). Doc of record: `docs/reference/episode_capture.md`;
  refusal codes in `docs/reference/error_refusal_contract.md`.

## Quality Gates

Every task must pass before completion unless the task explicitly narrows verification:

```bash
ruff format .
ruff check . --fix
mypy torchlens/
pytest tests/ -m smoke -x --tb=short
```

(CI lint runs `ruff format --check` plus `ruff check` over `torchlens tests scripts tools
benchmarks examples notebooks`; run `ruff format` locally or the format-check leg fails.)

For changes touching module boundaries or public API, also run:

```bash
pytest tests/ -m "not rare and not slow" -x --tb=short
```

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
9a. Rehydration floor: artifacts older than torchlens 2.33 (`tlspec_version` 6) refuse to load
    with the typed `tl.errors.ArtifactVersionBelowFloorError`; legacy field-alias resurrection
    ladders are deleted. Legacy 2.16 intervention specs remain loadable (floor covers Trace
    rehydration only).
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
    `Trace.module_collapse_order`, and `Trace.collapse_order(weights=..., mode=...)` must stay
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

## Known Gotchas

- Intervention-spec loads tolerate foreign `custom` callable keys for safe analysis without importing
  their modules. Resolving one for execution denies by default because imports execute top-level code;
  trusted execution may use `trust_custom_callables=True`, while
  `allowed_custom_callable_modules={"my_trusted_module"}` is narrower and remains restrictive even if
  the boolean is also true. TorchLens-owned `torchlens.*` custom callables and the fixed trusted
  namespaces resolve without an opt-in.
- `Trace.forward_peak_memory` on CPU/MPS is only the cheap host RSS (or MPS allocator) delta and
  legitimately reads `0` for small models. The `tracemalloc` Python-allocation peak is opt-in via
  `CaptureOptions(measure_python_peak_memory=True)` because the allocator hook costs 1.7x-2.5x
  total capture time. Never assert `forward_peak_memory > 0` on the default path.
- Distributed/sharded state is detected in `torchlens/_distributed.py` and refused at capture entry
  with `DistributedCaptureUnsupportedError`; the same detection feeds the `dtensor` / `device_mesh` /
  `tensor_parallel` / `pipeline_parallel` rows of `tl.compat.report`, so the two cannot drift.
  `dtensor`, active TP hooks/styles, and `pipeline_parallel` refuse; dense parameters do not make a
  `PrepareModuleInput` redistribution safe. Only a bare inert mesh is informational. Detection is
  capability-probed (`HAS_DTENSOR`, `HAS_DEVICE_MESH`, `HAS_PIPELINING`), never version-parsed, and
  bounded to inspectable instance state (12 levels / 4096 objects). Slots/descriptor-only holders,
  opaque user-wrapped TP hooks, over-bound state, and tensors created inside `forward` remain
  disclosed residuals. The `dtensor` finding identifies refused state precisely via per-site
  dual geometry on `finding.geometry`.
- Explicit `torch.distributed` python collectives in a traced forward become first-class boundary
  nodes under the distributed opt-in (`tl.distributed.arm()` at process start, REQUIRED for
  MPMD/spawn ranks; lazy arming covers initialized SPMD first-captures). The portable payload is
  `op.annotations["collective"]` (`collective_boundary_v1`: correlation key, role entries, event
  and witness disclosures) plus the trace-level group-lifecycle ledger in
  `trace.annotations["distributed"]`. `CaptureOptions(distributed_witness="digest")` opts into
  byte-exact contribution/destination digests. Async completions record
  `completion_binding="unobserved"`; wildcard recv refuses typed; collective-crossing traces
  refuse runnable save and forward-replay validation
  (`collective_boundary_runnable_unsupported`) while metadata invariants run in full. Arming
  relaxes no tier-(a) refusal (DTensor/TP/FSDP2/PP still refuse). Distributed rank processes
  (initialized process group, non-daemonic) are the sanctioned exception to the
  main-process-only capture guard.
- `tl.merge_ranks([trace_or_path, ...])` merges N rank cores into a `MergedTrace` presenter at
  their explicit collective boundaries (rung C1); `tl.merge_report(...)` diagnoses without
  constructing. The derivation is audit-first (a conflicted membership never joins and never
  becomes a presence gap), aligns seq counters as deltas from each rank's first recorded key,
  and treats witness digests as demote-only evidence. Artifacts save as `merged-directory`
  bundles whose descriptor is a cache: loads rederive from the rank cores and refuse typed on
  any inequality. Frozen enums + error/finding codes: `torchlens.merged` +
  `docs/reference/merged_trace_contract.md` (ordered-equality gated). p2p/pipeline (C3) and
  DTensor topologies (C2) refuse typed; merged replay does not exist.
- `CaptureOptions(save_budget=...)` is a per-device ceiling on retained activation bytes, default
  `"auto"` = half of measurable available memory. Exhaustive, predicate, and deferred
  `Op.save_activation()` paths pre-admit the primary source-sized RAM copy before allocation, then
  reconcile alias-aware physical storage. Transform
  deltas and cross-device temporaries cannot always be known pre-allocation, so this is not a general
  OOM guarantee. Predicate disk-only saves are exempt; exhaustive
  `capture=tl.options.CaptureOptions(layers_to_save="all")` plus `to_disk(...)`
  stays budgeted until postprocess eviction (the bare flat `layers_to_save=` kwarg is a deprecated
  alias and warns). Unmeasurable auto devices warn on first charge.
- Streamed disk writes are ASYNC BY DEFAULT for `trace(storage=tl.to_disk(...))` (spellings
  DOCUMENTED-UNSTABLE): one FIFO worker overlaps blob serialize+write+sha256 with the forward,
  payloads snapshot at submission, `to_disk(max_pending_bytes=)` (256 MiB default) blocks capture
  when the disk falls behind, a failed write raises typed `TorchLensIOError` + PARTIAL, finalize
  drains before publish, and bundles are byte-identical to sync. `async_writes=False` opts out;
  `tl.record` streaming stays synchronous and refuses an explicit `True`.
- On torch >= 2.6 (`HAS_SET_STANCE`), every capture holds
  `torch.compiler.set_stance("force_eager")` scoped to the forward (entered inside
  `prepare_compiled_capture`; skipped when `torch._dynamo` was never imported), so compiled plain
  attributes and free functions run their ORIGINAL eager Python: interiors are fully logged with
  ordinary verified semantics, zero compiles happen during capture, compiled caches survive with at
  most ONE bounded recompile on the next compiled call afterward (wrapper install/uninstall guard
  invalidation), and the plain-attribute pause-logging bypass is NOT installed. The paragraph below
  is the torch < 2.6 / no-stance fallback, pinned byte-for-byte by the `_no_stance` tamper tests in
  `test_dynamo_fake_guard.py`.
- A Dynamo-traced region reached during capture is bypassed in the wrapper (see
  `_is_inside_dynamo_compilation`), warning once per forward and setting
  `trace._raw_transform_escape_detected` (which licenses the unattributable-output tolerance,
  exactly like the functorch guard beside it) plus `trace._raw_dynamo_region_detected`, which is
  the top-precedence `capture_verification_reason` -- `"dynamo_region_not_logged"` -- at BOTH
  verdict sites (`completeness_witness._finalize_census` and the `escape_detection` capture-scope
  `finally`, the last writer). Without the dedicated flag the Trace blamed
  `owner_thread_tripwire_changed`, since compiling spawns threads. Plain compiled-callable
  attributes are also inventoried before forward and invoked with logging paused, conservatively
  arming the same flags on cold/warm runs where `is_compiling()` may never fire; global/free hot
  callables remain disclosed. Compiled child `nn.Module`s are still unwrapped to eager BEFORE
  capture, so their interiors stay logged -- there
  is a test asserting the bypass did not regress that into a silent gap. Fake/functional tensors on
  inputs or params refuse at capture entry in `_robustness.py`; never let one reach the metadata
  path, where `data_ptr()` on a FakeTensor is a torch-flagged bug.
- `__wrapped__` is removed from built-in function wrappers to avoid `inspect.unwrap`
  failures.
- Fast-path module decoration skips `_record_module_entry_metadata`; alignment state must be
  replicated manually.
- `get_memory_amount()` deliberately avoids `pause_logging()`: it resolves the
  UNWRAPPED `nelement()`/`element_size()` methods without toggling global logging
  state per tensor (hot-path perf commit `08dca260`); re-adding the toggle is a
  regression, not a fix.
- If a `@property` raises `AttributeError`, Python falls through to `__getattr__`; use
  `ValueError` for TorchLens multi-pass access errors.
- `copy()` on `Op` deep-copies graph metadata and shares tensor payloads/callables; see data_classes/AGENTS.md.
- `torchlens.__version__` and `pyproject.toml` are release-pipeline state; do not update them
  in feature/docs tasks unless release work explicitly asks for it.

## Build & Test

```bash
pip install -e ".[dev]"
pip install -e ".[test]"
pip install build && python -m build
pytest tests/ -m smoke
pytest tests/ -m "not rare and not slow"
pytest tests/
ruff format && ruff check --fix
```
