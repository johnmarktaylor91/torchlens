# Task & Bug Tracker

## Active Tasks

### Backward validation per-layer-grad oracle (DEFERRED from P6 via AD-32)

P6 intentionally shipped parameter-gradient parity hardening only. The
per-layer gradient oracle remains deferred because the round 5/6/7/8 design
iterations each exposed correctness risks in the proposed mechanism:
retain-grad-on-clone mismatch, same-run side-channel coupling, autograd
version-counter hazards, and zero-grad accumulation hazards.

Follow-up sprint should redesign the oracle from scratch. Candidate directions:
`torch.fx` symbolic trace, hook-install-via-wrappers stock run, or a
disposable-trace probe followed by state restoration. Do not reuse the deleted
P6 helper names (`_collect_stock_grads`, `_compare_layer_grads`,
`CoverageDiagnostic`, etc.) without a fresh design review.

### Intervention API naming sprint leftovers (raised cycle 2 round 7, 2026-04-29)

Naming sprint 2026-05-11. Items 1-7 and 10 from the original v1
implementation-spec maintenance list were resolved in `da15b5f`.

8. **`confirm_mutation` vs `suppress_mutate_warnings` naming consistency.**
   Per-call kwarg uses "confirm"; session config uses "suppress." Naming
   pass should align. Goes to JMT's separate naming workstream.

9. **`tl.suppress_mutate_warnings` as context manager option.** Currently
   accepts bool only. Consider context manager form:
   ```python
   with tl.suppress_mutate_warnings():
       log.do(...)
   ```
   Naming pass.

## Bugs

## Improvements (Nice-to-Have)

### Tensor-id-keyed metadata (move tl_* attrs off tensors) (raised 2026-05-10)

**SUPERSEDED 2026-05-10:** superseded by the `_tl` namespace refactor
(`0e4509d`). That refactor moved TorchLens host-object metadata under
`obj._tl` and removed the motivating `tl_*` attribute pollution without
requiring a tensor-id side table. Original note retained below for history.

**Status:** superseded; do not schedule as active work.

**Hypothesis:** replace per-tensor `tl_*` attribute decoration with a
`state.tensor_metadata: dict[int, TensorMetadata]` keyed on `id(tensor)`,
using `weakref.finalize(tensor, callback)` to clean up entries when tensors
are GC'd. The module-containment refactor proves the "metadata in side
state, not on tensors" pattern at module scope; this generalizes it to all
tensor-attached metadata (`tl__label_raw`, `tl_module_address`,
`tl_module_type`, `tl_buffer_address`, `tl_buffer_parent`, the whole
`tl_*` family).

**Wins:**
- Tensors stay clean -- `tensor.__dict__` and `print(tensor)` no longer
  surface TorchLens internals. Captum / fvcore / debugging tools stop
  seeing TorchLens noise.
- In-place semantics fall out for free. `id()` is stable across in-place
  mutation, so the ~80 lines of safe-copy + back-propagation in
  `decoration/torch_funcs.py:480-533` becomes a no-op.
- Cross-capture pollution genuinely solved -- each Trace owns its
  `state.tensor_metadata`; tensors that survive across captures don't
  carry stale metadata.
- Save/load gets simpler -- saved tensors don't need `tl_*` attribute
  scrubbing.
- Less surface area for "did I forget to strip an attr in cleanup?".

**Complications:**
- Tensor cloning paths (`.clone()`, `.detach()`, `.to()`, `.contiguous()`)
  produce new tensors with new ids. Need explicit "inherit metadata" rules
  at each captured op. Today's attribute approach inherits implicitly.
- `id()` reuse after GC: solved by `weakref.finalize(tensor, lambda: ...)`
  registered at write time.
- Tensor subclasses without `__slots__ = ()` need to support `__weakref__`.
  `torch.Tensor` does; verify for `DTensor` etc.
- Storage-shared views: same as today, new tensor = new metadata at
  op-capture time. No difference.
- Performance: C-level `getattr` slightly faster than dict lookup, but
  both are tens of nanoseconds. Net likely faster because in-place
  propagation logic disappears.

**Order:** AFTER module-containment-refactor. The module work proves the
pattern at module scope; tensor-id-keyed extends it. Doing simultaneously
would make both reviews harder.

**Estimated effort:** 3-5 days. Net code reduction.

### Capture-path unification: log_forward_pass internally as Recording (raised 2026-04-29)

**Status:** future architectural cleanup. NOT for the intervention API push.

Hypothesis: `log_forward_pass` could internally use fastlog's capture
machinery (Recording-shaped events) and promote to ModelLog during
postprocessing. Single capture path, divergent endpoints (Recording vs
ModelLog). Discussed 2026-04-29.

**Wins (real):**
- Eliminate capture-path drift between log_forward_pass and fastlog —
  one event model, one set of edge cases.
- Streaming / disk-backed ModelLog becomes possible (today: log_forward_pass
  is RAM-only; fastlog has disk backing but loses the graph).
- Intervention sweeps with lazy promotion (1000 Recordings, promote on
  demand to ModelLog) become natural.

**Costs (real):**
- Performance regression on the hot path — 5-30% slower estimated by
  going through predicate machinery even with "keep-all" predicate.
- Field-model reconciliation is significant work — `RecordContext`
  (slots=True frozen) vs `LayerPassLog` (mutable, 85+ fields with
  @property accessors). Whichever direction you reconcile, real engineering.
- Postprocessing assumes graph completeness; fastlog's event model is
  sparse-by-design. "Unified" means fastlog gains a "keep everything +
  full metadata" mode that's basically log_forward_pass internally —
  illusion of unification, two paths under one entry point.

**Phased path forward:**

- **Phase 1 (this push):** ship the BRIDGE — `recording.to_modellog()`
  adapter when fastlog runs with `full_metadata=True`. Low risk, adds
  optionality, doesn't touch log_forward_pass internals.
- **Phase 2 (after intervention API ships):** measure. Benchmark
  to_modellog promotion. Verify Recording can losslessly carry
  LayerPassLog's field set. If yes, proceed; if no, document the gap.
- **Phase 3 (possibly never):** internal refactor — log_forward_pass
  becomes "fastlog with full_metadata + keep_all + postprocess."
  User-facing API unchanged. Worth doing only if Phase 2 shows the
  benefits outweigh the regression.

**Drift-prevention discipline (do now regardless):**
- When adding `to_modellog()` adapter, write a real round-trip parity
  test — Recording → ModelLog and verify which fields survive. Document
  any fastlog drops.
- When LayerPassLog gains new fields, audit whether RecordContext
  should grow them too. Don't let them silently diverge.
- When fastlog adds capture options, mirror in `log_forward_pass` where
  it makes sense.

This is the cheap maintenance that keeps the unification door open
without paying for it now.

---

### Cross-model layer alignment for RSA-style ops (raised 2026-04-29)

**Status:** deferred, future Bundle feature.

When a TraceBundle contains members from genuinely different
architectures (e.g., ResNet-50 vs VGG-16, base model vs quantized,
teacher vs student), the bundle's auto-detected relationship taxonomy
flags pairs as `DIFF_MODEL` or `SHARED_ARCHITECTURE`. Most bundle ops
that compare per-node activation across members can't operate on
`DIFF_MODEL` pairs by default — there's no canonical site mapping.

For RSA-style cross-architecture comparison and similar ops, expose a
user-supplied alignment mechanism:
```python
bundle.aligned_pairs(
    {"resnet": ["block1.conv1", "block2.conv1", ...],
     "vgg":    ["features.0", "features.5", ...]}
)
# Or via callable:
bundle.aligned_pairs(lambda member_a, member_b: [...])
```

Operations that work on aligned pairs accept them as an extra parameter.
Bundle remains permissive (anything goes in); ops that need alignment
require it explicitly when relationships aren't structurally compatible.

**When to ship:** when a real user asks. Defer until then. Bundle's
auto-detection plus the existing per-node ops (which work for
`SHARED_GRAPH_*` relationships) covers the common cases.

---

### Naming polish pass before TorchLens 2.0 marketing push (raised 2026-04-28)

JMT's instinct: the public API surface should "sing" before the 2.0
marketing push. The `tl.do(...)` idea (do-operator from causal inference)
is the trigger — short, loaded with the right literature meaning, plants
a flag for both mech-interp and causal-inference audiences. That made
him want a holistic pass over names, not just the intervention API.

**Scope:** review every public-facing name in `tl.*` and key methods on
ModelLog / TraceBundle / etc. for:

1. **Brevity.** Compare `tl.apply_intervention` vs `tl.do`. Which lands in
   tutorials and tweets?
2. **Literature resonance.** Does the name align with the term of art in
   the literature we want to reach? (do-operator → causal inference;
   "trace" → existing interp idiom; etc.)
3. **Distinctiveness.** Is this a name nobody else has claimed? Or are we
   colliding with TransformerLens / NNsight / Pyvene's naming? Both wins
   and losses possible — sometimes parity is good (familiar to migrators);
   sometimes our own term is right.
4. **Two-audience reach.** Some names (like `do`) speak to two communities
   at once. Find more of these.
5. **Read-as-English.** `tl.do(model_log, hooks={"site": tl.zero_ablate()})`
   reads as "do this." Test every public name against the read-aloud check.
6. **Avoid cute-dated risk.** Decades-old canonical terms (do-operator,
   trace, hook) age well. Pop-slang or in-joke names age badly.

**Specific candidates surfaced 2026-04-28:**
- `tl.do(...)` to replace `tl.apply_intervention(...)` (CC + JMT lean yes,
  pending overnight thought).
- Imperative one-shot — currently planned `tl.intervene(...)`. Reconsider
  whether to keep it, fold into `tl.do(...)` overload, or use a different
  verb. Also reconsider whether `model_log.fork(...)` is the right verb
  vs `model_log.branch(...)`, `model_log.checkpoint(...)`, etc.
- The Fork D registry (when Fork D resolves): `model_log.runs[name]`
  vs `model_log.interventions[name]` vs `model_log.derived[name]`
  vs `model_log.children[name]`. Different connotations.
- Helper names — `zero_ablate` / `mean_ablate` / `resample_ablate` are
  established interp terminology; probably keep. Less-canonical helpers
  like `swap_with` / `splice_module` worth re-examining.
- `train_mode=True` flag — fine but generic. Consider alternatives.
- `tl.where(predicate)` — clean, Pandas-resonant, probably keep.
- The whole `tl.adapters.*` submodule — adapter names will be locked then.

**Process:** before MVP code-freeze for the 2.0 marketing push, dedicate
one focused session to walk every public name with this checklist. Likely
a small team brainstorm rather than a unilateral CC pass — naming
benefits from multiple perspectives.

**Hard constraint:** lock names BEFORE marketing push (see
`project_torchlens_2_marketing.md`). After that point, renames cost a
major version bump and break user code.

---

### Interactive Jupyter widget for intervention (raised 2026-04-28, tabled)

Vision: click-to-intervene compute graph in a Jupyter widget. Pan/zoom,
hover tooltips, click a node to inspect activation, drag a slider to
perturb, watch the cone of effect re-render live. Penzai-treescope-quality
interactive surface for TorchLens compute graphs + interventions.

**Status:** TABLED. Modular enough that we can build later without
changing core decisions made during the intervention API design.

**Constraints on intervention-API work to KEEP THIS DOOR OPEN:**
- InterventionSpec must be JSON-serializable (locked in Fork K).
- `model_log.runs[name]` (or whatever Fork D resolves to) must be
  queryable by name programmatically.
- `node_spec_fn` callback stays public so widget can hijack rendering.
- Site grammar (Fork C) must remain structured / introspectable so the
  widget can offer autocomplete and predicate-builder UI.

**Implementation sketch (when picked up):**
- Probably a sister repo: `torchlens-notebook` or similar.
- Built on anywidget or D3.js; exports to standalone HTML.
- Reads model_log + runs over an ipywidgets bridge.
- Intervention authoring surface UI builds an InterventionSpec via the
  same dict-spec format `apply_intervention` accepts.

Estimate: multi-month, separate developer. Not on the critical path for
TorchLens 2.0 marketing push.

---

### Gradient intervention deferred items (raised 2026-04-28 during Fork G)

The intervention API ships Tier 1 (`tensor.register_hook`) in MVP and
Tier 2 (grad_fn-level) in v1. The following are explicitly deferred:

- **Training-time gradient routing (Tier 3).** Cloud et al. 2024 style:
  selectively mask which params receive gradients per data shard / batch.
  Used for unlearning, capability isolation, modular training. Big new
  abstraction (routing config + per-batch mask compositing). Couples to
  optimizer / training loop, not just forward/backward. Treat as its
  own multi-month sub-project, possibly its own paper. Don't entangle
  with the intervention API roadmap.
- **Higher-order gradients (grad-of-grad).** Required for some
  attribution methods. Nontrivial to wire through the replay engine.
  Defer until a user asks.
- **Automatic jacobian / vjp tracing.** Captum-style linearization
  utilities. Captum already owns this — likely better as a Captum
  integration helper than reimplemented.
- **Backward replay** — re-running backward with a different forward
  intervention applied. The forward replay engine handles this naturally
  if the user runs forward replay first then standard backward, but
  decoupled "replay only the backward with a tweaked forward overlay"
  is a separate design problem.

---

### Retire ELK from visualization (raised 2026-04-28)

Policy decision: visualization is **graphviz now, dagua later** — no more
ELK work. `torchlens/visualization/elk_layout.py` (~1276 lines) and any
graphviz-side ELK plumbing should be deleted in a dedicated cleanup PR.
Audit and remove:
- `elk_layout.py` and any tests targeting it.
- ELK-related kwargs / config knobs in `rendering.py`.
- Any docs or examples referencing ELK output.

Time the deletion to not collide with the intervention API work — neither
work touches ELK by intent, but a cleanup pass makes the codebase smaller
before the dagua bridge fully takes over.

---

### Stacked multi-pass single-ModelLog (raised 2026-04-28, separate from intervention API)

Workflow gap: user has 1000 prompts, model fits batch=4 in memory. Today
they run 250 forward passes and manually concatenate activations in their
own scripts. TorchLens should offer a first-class API:

```python
# Returns a SINGLE ModelLog whose per-layer activations are stacked along
# batch dim across all passes; graph metadata is union/intersection.
model_log = tl.log_forward_pass_streaming(
    model,
    inputs=dataloader,        # iterable yielding sub-batches
    layers_to_save=[...],
    stack_dim=0,
)
# model_log["relu_4"].activation has shape [1000, ...] not [4, ...]
```

Distinct from:
- **TraceBundle** — N independent ModelLogs side-by-side, not concatenated.
- **Streaming aggregate** (Multi-trace V2 todo) — computes running stats,
  discards raw activations.
- **save_new_activations** — REPLACES on a single fixed-batch model_log.

Open design questions:
- Topology divergence across passes: refuse if traces don't match? (Probably
  yes — if topology differs use TraceBundle.)
- Memory: stream activations to disk via fastlog-style backing? Probably
  required at scale.
- Graph metadata: pass-1 metadata wins (assumes shared topology). Document.
- Validation/replay implications: replay needs to know how to slice the
  stacked tensor to align with a particular pass's RNG state.

Estimate: 2-3 weeks. Related but not coupled to intervention API. Worth
its own design pass.

---

### Multi-output module handling (raised 2026-04-28 during intervention API design)

PyTorch modules routinely return multiple tensors: `nn.LSTM` returns
`(output, (h_n, c_n))`, `nn.MultiheadAttention` returns
`(attn_output, attn_output_weights)`, `nn.GRU` returns `(output, h_n)`,
many HuggingFace modules return dicts/dataclasses. Audit current TorchLens
behavior across all module-touching surfaces and fix gaps:

- **Module output addressing.** When a user references `"features.0"` (a
  module that returns a tuple), what does `model_log["features.0"]`
  return? Today this is one-of-the-tensors (probably the first); should be
  a structured handle that exposes `.outputs[0]`, `.outputs[1]`, etc., or
  named fields if the module returns a dict.
- **Saved activations.** `save_new_activations` and the layer save path
  need to handle tuple/dict outputs without dropping the non-primary
  tensors. Verify each output tensor's `LayerPassLog` is captured.
- **Module hooks (intervention API).** When a hook is registered at a
  module address that returns a tuple, the post-hook receives the tuple
  (mirroring `nn.Module.register_forward_hook`). Hook can return a
  modified tuple. Pre-hooks similar. Document the contract.
- **Visualization.** Multi-output modules should render with multiple
  output edges, one per output tensor, optionally labeled.
- **Bundle compatibility.** TraceBundle / Supergraph need to handle
  multi-output modules as part of topology — each output is its own
  graph node, all attributed to the same module.

This is a torchlens-wide audit, not just an intervention concern.
Surfaces affected: `data_classes/module_log.py`, `capture/output_tensors.py`,
`postprocess/`, `visualization/`, `_lookup_keys.py`. Estimate: 1-2 weeks
including tests against LSTM/GRU/MultiheadAttention/HF model fixtures.

### Multi-trace V2 (deferred from 2026-04-27 sprint design)

These were scoped OUT of the multi-trace MVP (TraceBundle + TraceOverlay)
but are natural follow-ons. Pick up after MVP ships.

- **Branch-divergence detection.** Soft version of "counterfactual
  tracing." Given a TraceOverlay, surface "branch-divergence nodes"
  where execution paths split, expose as a queryable list. NOT true
  forced-branch enumeration -- just detection of where forward paths
  diverged across the bundled inputs.

- **True counterfactual branch enumeration** (forced-branch execution).
  Programmatically force both arms of every Python `if` in a dynamic
  network. Hard problem -- requires either symbolic tracing
  (incompatible with TorchLens's runtime model), bytecode hacking
  (fragile), or an intervention API where the user explicitly drives
  forced inputs to coax branches. Defer indefinitely; the
  branch-divergence detection above is the practical 80% solution.

- **Streaming aggregate over a dataloader.** `tl.aggregate(model,
  dataloader, metrics=[...])` -- consumes traces from a generator,
  computes running per-node statistics (mean, var, RDM,
  dimensionality), discards raw activations. For "10K images through
  ResNet50" workflows where holding traces is impractical. Function,
  not a class.

- **Interactive viewer for bundles/overlays.** D3.js or anywidget-based
  Jupyter widget. Pan/zoom, hover tooltips, click-to-expand node
  visualizations, path highlighting, export to standalone HTML. After
  graphviz MVP is stable.

- **Custom node visualizations.** Pluggable per-node display: PCA-RGB
  for conv layers, MDS scatter for linear layers, histograms for
  activations, dimensionality estimates. Bundle/Overlay would accept
  `node_display={'conv.*': 'pca_rgb', ...}` mappings.

- **Convenience constructors with intervention APIs.** `tl.zero(node)`,
  `tl.steer(node, vector)`, `tl.compare(model, input, {...})`,
  `tl.sweep(model, input, {...})`. Interventions produce a new
  ModelLog (never mix clean and intervened activations in a Bundle).
  Out of scope for MVP; design once Bundle/Overlay are stable.

- **Rename ModelLog -> Trace** (one-way door). Clean public-API
  migration -- `tl.trace()` constructor, `Trace` class. Major
  breaking change to a published-and-cited package. Do as a separate
  deliberate migration, never bundled with feature work.

- **`vis_opt='rolled'` mode for `show_bundle_graph`.** Phase 2 plumbed
  the kwarg through with arg validation but the bundle renderer
  currently treats every supergraph node as a single rendered node
  regardless. The supergraph already operates on `LayerLog` (rolled-
  equivalent) entries, so cluster titles already carry `(xN)` count
  suffixes for multi-pass modules. A true `rolled` mode would also
  collapse loop-detected layer chains into single rendered nodes the
  way `rendering.py` does for `vis_mode='rolled'`. Defer until a real
  use case appears; the current `unrolled` behaviour covers the
  visualization story for shared-topology and divergent bundles.

- **`direction='backward'` for `show_bundle_graph`.** Currently raises
  `ValueError` on anything except `'forward'`. Backward graph topology
  depends on forward execution path, so multi-trace backward
  visualization needs a design decision -- per-trace separate backward
  subgraphs, or a unified backward supergraph keyed on grad_fn? Defer
  to the multi-trace V2 batch alongside the other Phase 2 follow-ons.

- **Module-type second line in bundle cluster labels.** `show_model_graph`
  cluster labels include a `(<ModuleType>)` line under the module name;
  `show_bundle_graph` omits it because `Supergraph` doesn't preserve
  module class on its nodes. Recoverable in two small edits: add a
  `module_type: str | None` field to `SupergraphNode` populated during
  `build_supergraph` from the canonical trace's `LayerLog.containing_module_type`
  chain, then teach the bundle cluster builder in
  `torchlens/multi_trace/_bundle_clusters.py` to emit the second line
  when present. Defer until aesthetic parity matters more than a clean
  diff.

- **Public `TraceBundle.supergraph` accessor.** The bundle renderer
  reads `bundle._supergraph` privately. If we want users to introspect
  the supergraph directly (e.g. to enumerate selective nodes by their
  edge frequencies, or build custom visualizations), promote to a
  public property. One-line change; defer until a real user asks.

- **Per-node / per-edge styling primitives in `_render_utils.py`.** Phase 2
  polish extracted cluster styling, file-format dispatch, HTML escape,
  and direction translation into shared `_render_utils.py`. Per-node
  styling (`_build_layer_node`, `_build_collapsed_module_node`) and
  per-edge styling were left in `rendering.py` because the shapes
  differ substantially between ModelLog (LayerLog/NodeSpec coupled) and
  bundle (mode-driven coloring) -- agent's call was a parameterised
  version would be thin pass-through or parameter explosion. Revisit if
  a third visualization consumer appears or if the bundle/ModelLog
  styling logic genuinely converges; otherwise the current seam is the
  right one.

### Other improvements

- Replace input-tensor-derived module attribution with thread-local
  call-stack snapshot (raised 2026-05-09). Vast simplification.

  **Status 2026-05-10:** Step 6 elimination is DONE in `8dd33a6`.
  `_fix_modules_for_internal_tensors` was deleted; the module-stack suffix
  is appended to `equivalence_class` at op creation, with a six-model
  equivalence-class parity check against the pre-refactor pipeline. The
  numbered postprocess pipeline was updated accordingly. Historical design
  notes below are retained for context.

  TODAY (three mechanisms, one concept):
  1. `_get_input_module_info` (capture/source_tensors.py) walks an
     op's input tensors, finds the most-deeply-nested parent's
     module stack, applies the parent's `_module_boundary_thread_output`
     transitions to get current state, returns that as `op.modules`.
  2. Per-tensor `_module_boundary_thread_inputs` /
     `_module_boundary_thread_output` track entry/exit events on
     each tensor as it crosses module boundaries.
  3. Postprocess Step 6 (`_fix_modules_for_internal_tensors`) walks
     the graph propagating module info backward to internal tensors
     that have no input parents (factory ops like `torch.arange`),
     replaying entry/exit threads in reverse.

  All three exist because module info is currently graph-derived
  (inferred from tensor inputs), not call-stack-derived. Factory ops
  and pure-internal subgraphs need the fix-up because they have no
  input tensor to inherit from.

  PROPOSED (one mechanism):
  - Thread-local module call stack pushed/popped by entry/exit hooks.
    The hooks already fire today; we just don't read them at op-time.
  - At op capture: `fields_dict["modules"] = list(_state.module_call_stack)`.
    Done. One line.
  - Drop `_get_input_module_info`. Drop Step 6 (`_fix_modules_for_internal_tensors`)
    entirely. Drop or relegate `_module_boundary_thread_*` fields
    (kept only if rendering still needs per-tensor entry/exit
    annotation; investigate).

  Net effect:
  - Three mechanisms collapse to one.
  - Factory / orphan / internally-generated ops get correct module
    attribution for free, no fix-up needed.
  - The `keep_orphans=True` todo above becomes simpler — orphan
    attribution is automatic via the call stack.
  - Step 6 of postprocess deleted; pipeline shrinks to 19 steps.

  Edge cases to validate before shipping:
  - **Direct `forward()` calls bypassing `__call__`.** Hooks don't
    fire if user does `submodule.forward(x)` directly. Today: input
    derivation might recover module info from earlier-properly-called
    parents. Call-stack: would miss the bypassed submodule. Check
    test coverage; document as anti-pattern if relevant.
  - **`output_of_modules` / `output_of_module_calls` derivation.**
    These currently use `_module_boundary_thread_output` to identify
    when a tensor exits a module. If we drop per-tensor threads, we
    need a different derivation (probably: track module exit events
    in the global thread + record which op was the "last op inside"
    each module call as it pops).
  - **Cross-module tensor flow semantics.** Today's input-derived
    approach answers "where did this op's data originate?"; call-
    stack answers "where was this op called from?". The call-stack
    answer is what users almost certainly mean by `op.modules` /
    `op.containing_modules` — confirm via the existing test suite +
    audit notebook outputs.
  - **Recurrent / loop modules.** Both approaches handle pass
    indexing identically; no difference expected. Verify via the
    existing loop-detection test suite.
  - **Module-prep-time tensors (params/buffers).** Their attribution
    comes from a separate prep-time channel
    (`tl_module_address`/`tl_module_type`); unchanged by this
    refactor.

  Implementation cost: 1-2 days.
  1. Add thread-local module call stack to `_state.py`. Push on
     entry hook, pop on exit hook. Already-fired hooks just write
     to the new stack as well as the existing per-tensor thread.
  2. Replace `_get_input_module_info` call site at
     `output_tensors.py:1129` with a stack snapshot.
  3. Audit `output_of_modules` / `output_of_module_calls` derivation
     and re-implement on the global thread if needed.
  4. Delete `_fix_modules_for_internal_tensors` (Step 6) once the
     above lands and tier-2 stays green.
  5. Update `postprocess/CLAUDE.md` step table.
  6. Decide on `_module_boundary_thread_*` fate: keep (for rendering)
     vs drop (if no consumer remains).

  Worth doing as a follow-on to the super-family-sprint or as a
  small standalone refactor sprint. The simplification is genuine
  and the data-model story gets cleaner: one channel for "where was
  this op called from," with model-prep-time channel for static
  param/buffer attribution. The graph-walk fix-up disappears.

- `keep_orphans=True` flag to retain island ops for tracking
  (raised 2026-05-09). Today `_remove_orphan_nodes` (postprocess
  Step 3) flood-fills bidirectionally from input AND output and
  removes any node unreachable from both. The surviving "orphans"
  it purges are NOT input-connected dead ends (those are kept) —
  they are TRUE islands: internally-generated subgraphs with no
  input ancestry AND no output descendancy. Examples:
  `_ = torch.arange(10).mean()` inside forward, scratch
  `torch.eye(3) @ torch.eye(3)` arithmetic, discarded `torch.zeros`.
  Almost always computational lint, not user-meaningful.

  Niche debugging value: a user who expected an internal tensor
  (e.g., `pos_emb = torch.arange(seq_len)` they forgot to add to
  embeddings) to flow somewhere should be able to discover that
  it ran but went nowhere. Today they get `trace.orphan_ops`
  (label list); we can do better.

  Proposed flag:

  ```
  trace = tl.trace(model, x, keep_orphans=True)
  # or via the canonical capture options:
  trace = tl.trace(model, x, capture=tl.CaptureOptions(keep_orphans=True))
  ```

  Default `False` (historical behavior; orphans are usually noise).
  Opt-in `True` keeps orphans in the data with `is_orphan=True` flag
  and a proper `trace.orphans` accessor (now trivial post-Phase-1
  Accessor[T] base — ~10 lines).

  Module attribution for orphans (the user's specific concern):
  TorchLens has TWO attribution channels and only ONE is graph-
  dependent.
  - **At-execution-time** (`op.containing_modules`): captured when
    the op fires, from the live `nn.Module` call stack. Doesn't
    care about graph topology. Works for orphans created inside
    a module's `forward()` -- attribution is correct.
  - **Postprocess graph-walk fix-up** (`_fix_modules_for_internal_tensors`,
    Step 6): infers module by walking descendants. Doesn't work for
    islands. Skip explicitly when `keep_orphans=True`.

  The honest gap: orphans created OUTSIDE any module forward (e.g.,
  during model setup) have empty `containing_modules` because the
  call stack was empty at execution time -- accurately reflecting
  "this op wasn't called from any module."

  Implementation shape (~half day):
  1. `capture.keep_orphans: bool = False` in CaptureOptions.
  2. `_remove_orphan_nodes` branches on flag: when True, set
     `is_orphan=True` on raw nodes; keep them in `_raw_layer_dict`.
  3. `_fix_modules_for_internal_tensors` skips orphans.
  4. Downstream postprocess steps (loop detection, labeling,
     finalization) explicitly exclude orphans from main-graph
     operations -- they don't participate in loop grouping or
     final-label renames.
  5. Promote `trace.orphans` from label list to `Accessor[OpLog]`
     (post-Phase-1 base makes this trivial).
  6. Visualization: orphans hidden by default; `vis_show_orphans=True`
     renders in distinct gray-dashed style so they're visually marked
     as not-part-of-the-real-computation.
  7. Validation: skip orphans by default; `validate(..., include_orphans=True)`
     opts in.
  8. Save/load: `is_orphan` round-trips via PORTABLE_STATE_SPEC.
     Orphan tensors obey the same `layers_to_save` / streaming rules
     as anything else; if memory becomes a concern, a future
     `save_orphans=False` knob can split storage from existence.

  Slot in after Phase 8 of the super-family-sprint (post-hygiene)
  or as a small standalone PR. Bounded; uses Phase 1's Accessor[T]
  base; no architectural risk.

- Comprehensive naming pass before TorchLens 2.0 marketing push
  (raised 2026-05-09). Pinning the rule from the morning naming
  riff: every public name has to pass the read-aloud test. Anything
  new that fails it gets pushed back; existing names that fail get
  reviewed in this sprint. Already-penciled changes:
  - `peek` -> `pluck` (charm + honesty about cost; matches
    `purrr::pluck` semantics; pairs with `attach` as `attach`/`pluck`
    symmetric verb pair on the trace; reinforces the dotted-path
    walking we designed for the recipe/container `op.outs` work).
    Singular extraction verb only — discrete one-thing action.
  - `extract` stays as the batch verb — close enough to a key
    function that it should be unmistakable and clear; `harvest`
    was considered (better metaphor pair with `pluck`) but rejected
    for now to keep charm budget focused.

  Other candidates to revisit during the sprint:
  - **Selectors** — workmanlike (`tl.func`, `tl.module`, `tl.label`).
    Could ship a poetic library: `tl.starts_with`, `tl.ends_with`,
    `tl.contains`, `tl.matches`, `tl.shaped_like((1, 4, 8, 8))`,
    `tl.between("conv2d_1", "conv2d_5")`, `tl.where(predicate)`,
    `tl.everything()`. Compose with existing func/module selectors.
    Single biggest tidyverse-flavored move available.
  - **`OpLog` / `LayerLog` / `ModuleLog` etc.** — `-Log` suffix is a
    tell that we're naming from the implementation outward (records
    in a log). User thinks "the layer," "the op." We renamed
    `ModelLog` -> `Trace`; consider whether `OpLog` -> `Op`,
    `LayerLog` -> `Layer` follows. Tradeoff: `Module` collides with
    `nn.Module`; `Op` and `Layer` are general English/PyTorch words.
    The qualifier may be load-bearing for clarity. Decide deliberately.
  - **Comparison/aggregation verbs** — `most_changed`, `compare_at`,
    `relationship`, `aligned_pairs`, `delta_map` reads as a
    grab-bag. Tidyverse-shaped would be a single `compare()` verb
    returning a tibble-shaped result with kwargs for direction and
    metric. Worth a coherence pass.
  - **Symmetric pairs** — audit for missing inverses. `set`/`unset`,
    `attach`/`detach`, `fork`/`merge`, `do`/`undo` (don't have this;
    should we?), `replay`/(nothing). When a verb has no inverse,
    document it as out-of-scope or add the inverse.

  Sprint hygiene:
  - One canonical name per concept; deprecation alias for one minor
    cycle on each rename. Drop deprecated aliases at the next major
    bump.
  - Family cohesion across subpackages — `tl.intervention.*`,
    `tl.viz.*`, `tl.bridge.*`, `tl.transforms.*` should all sound
    like one family. Lowercase verbs, return data, chain.
  - Anti-name list: if anything ends in `-er` / `-tion` / `-Manager`
    / `-Orchestrator` / `-Context` and isn't a borrowed domain word,
    it's suspect.
  - This is naming-sprint v3 (after the v1 + v2 we already shipped
    on `naming-sprint-impl`). Do NOT bundle with feature work — pure
    rename pass, all-tests-green, no behavior changes.

  Drop `-Log` suffix from log classes (raised 2026-05-09; bundle
  with naming-sprint v3, NOT a separate sprint):

  Following the precedent of `ModelLog -> Trace`: the `-Log` suffix
  is naming-from-the-implementation-outward (the class IS a record-
  of-a-thing, but the user thinks of it as "the layer," "the op").
  Class names mostly surface in repr / type hints / introspection —
  users access via `trace.layers["..."]` accessors and almost never
  import the class directly. So the rename is mostly cosmetic from
  the user's perspective but makes repr / docstrings cleaner.

  **Principled rule**: drop `-Log` UNLESS the bare name collides with
  a high-frequency PyTorch concept.

  | Class | Bare name | Collision check | Verdict |
  |---|---|---|---|
  | `OpLog` | `Op` | weak | drop |
  | `LayerLog` | `Layer` | weak | drop |
  | `BufferLog` | `Buffer` | weak | drop |
  | `ParamLog` | `Param` | weak (PyTorch uses `Parameter`) | drop |
  | `GradFnLog` | `GradFn` | weak | drop |
  | `ModuleLog` | `Module` | **strong (nn.Module is ubiquitous)** | **keep** |
  | `ModuleCallLog` | compound noun, awkward either way | defer |
  | `GradFnCallLog` | compound noun, awkward either way | defer |

  The slight inconsistency (most lose `-Log`, `ModuleLog` keeps it)
  is principled, not arbitrary. Document in the rename PR:
  "ModuleLog keeps the qualifier because Module collides with
  nn.Module."

  Compound-noun cases (`ModuleCallLog`, `GradFnCallLog`) are
  awkward either way; defer to a separate underlying-noun renaming
  pass that may produce better names like `ModulePass` (matching the
  existing `LayerPassLog -> OpLog` rename precedent).

  Each rename ships with a deprecation alias for one minor cycle.
  Coordinated batch pass — users update once, not five times.

- Generic `transform=` kwarg on `tl.trace` + raw-input rendering
  (raised 2026-05-07). Refines the text-input idea below into a
  cleaner architecture: instead of putting tokenization in a bridge,
  add a generic preprocessing primitive to core that ANY domain can
  use, store the original input alongside the trace, and render it
  in the graph's input node. The HF text-input bridge entry below
  becomes a ~5-line wrapper around this primitive.

  Core primitive:
  ```
  trace = tl.trace(model, raw_input, transform=callable)
  ```
  - `transform` is `user_input -> model_input`. Runs once before
    the trace starts.
  - `Trace.raw_input` (or `Trace.user_input`) holds the original;
    `Trace.input_args` holds what the model received (current
    behavior, unchanged).
  - Multi-arg models: transform should handle (a) single value ->
    tensor, (b) tuple -> tuple, (c) dict -> dict-of-kwargs (matching
    `model.forward` signature). HF tokenizers naturally return
    dict-shaped output — auto-`**unpack` when the transform returns
    a Mapping.
  - Composition is the user's job. Single callable, not a list.
    `transform=lambda x: tokenize(preprocess(x))` is fine.

  Save / load policy:
  - Default: save a small representation of the raw input (thumbnail
    for images, truncated string for text — say, first ~10 KB).
    Don't round-trip the original — privacy-sensitive prompts and
    large image bytes don't belong in a portable artifact.
  - `tl.trace(..., save_raw_input='small'|True|False)` overrides.
  - Transformed model input we already save (it's the input node's
    `op.out`).

  Visualization — the genuinely fun part. Graphviz has `image=` and
  HTML-table labels; this is afternoon-scale, not month-scale.
  - **Text inputs:** HTML-table label with the prompt as content,
    aggressive truncation (~80 chars + ellipsis). For HF
    `trace_text` callers we additionally render token boundaries as
    `|`-separated chunks with token IDs aligned underneath in a
    second row.
  - **Image inputs:** save a 200x200 thumbnail to the output dir,
    set `image=` + `imagescale=true` + `shape=none` on the input
    node. Graphviz embeds it. Done.
  - **Audio inputs:** matplotlib waveform sketch -> PNG -> graphviz
    `image=`. Same pattern.
  - **Multimodal:** input becomes a cluster, one sub-node per
    modality. We already use clusters for modules; reuse the
    primitive.
  - **Hover / clickable.** `tooltip=` for full untruncated metadata
    (file path, dimensions, byte size); `URL=` for click-out to the
    original (`file:///...` for image paths). Free; pure attribute
    plumbing.

  Renderer dispatch:
  ```
  def _render_raw_input(value):
      if isinstance(value, str): ...
      if isinstance(value, PIL.Image.Image): ...
      if isinstance(value, np.ndarray) and value.ndim == 3: ...
      if torch.is_tensor(value) and value.ndim == 4 and value.shape[1] in (1,3): ...
      if isinstance(value, dict): ...
      return None  # fall back to existing tensor-shape display
  ```
  ~30 lines + per-modality helpers (~15 lines each). User extension
  via `tl.register_input_renderer(MyType, fn)` if anyone asks; not
  required for v1.

  No `view='raw_input'` toggle — just always render when we know how.
  Unknown type falls back to the existing default, no regression.

  Batched inputs. The input node represents the whole batch; render it
  as a summary. Shared sampling policy across modalities:

  - `batch == 1` (or no batch dim): full single-item render.
  - `batch <= 4`: render all items.
  - `batch > 4`: render first 4 + `+N more` badge.

  Per-modality summarization:

  - **Images.** PIL montage — grid of thumbnails on a fixed canvas
    (~600x600 cap so the SVG doesn't bloat). `cols = ceil(sqrt(n))`,
    thumbnails sized to fit. Save as one PNG, set `image=` on the
    node. ~15 lines.
  - **Text.** HTML-label table — one row per prompt, truncated to
    ~60 chars per row, max 4 rows + `+N more` footer. Pure
    HTML-label markup, no image file.
  - **Audio.** Stack mini-waveforms vertically up to 4; above that,
    first + count. matplotlib for the waveform; same `image=` pattern
    as single-audio.
  - **Heterogeneous / multimodal batches.** Defer — fall back to the
    existing tensor-shape display. Add later if anyone asks.

  Override via `tl.trace(..., batch_render='all' | 'first' |
  'first_n:N' | 'shape_only')`. Defaults are good for ~95% of cases.

  Important: the batch-summarize helpers should be a reusable building
  block, not entangled with the input node renderer. Same
  `_montage` / `_text_table` / `_waveform_stack` helpers should be
  available wherever batched tensors get visualized (future
  activation-image rendering, bundle visuals, etc.). Lay them out as
  a small `tl.viz.batch_summary` module from day one.

  Bonus side-benefit: better error messages. If `transform` produced
  a tensor of wrong shape for the model, we can surface
  `"You passed text='...'; transform produced tensor (1, 5);
  model expected (B, 1024)"`. Currently we can only show the tensor
  side.

  Layer visualizers — the symmetric extension to intermediate nodes.
  Same plugin dispatch, same `image=` mechanism, same shared helpers.
  Each layer (or any matched subset) gets an inline thumbnail
  decorating its rendered node body, replacing the boring
  "shape (KB)" line with something semantic.

  ```
  trace = tl.trace(
      model, x,
      layer_visualizers={
          tl.func("conv2d"): tl.viz.channel_grid(n=16),
          tl.func("relu"):   tl.viz.heatmap(),
          "transformer.blocks.*.attn": tl.viz.attention_heatmap(),
          "embedding_1": tl.viz.tsne(n=200),
      },
  )
  ```

  Contract: `visualizer(tensor, *, layer_label=None) -> PIL.Image | str`.
  PIL.Image gets embedded as a graphviz node image; string is used
  as an HTML-label body. One function, one return type, one contract.

  Selectors reuse the existing intervention selector vocabulary
  (`tl.func`, `tl.module`, label lists, glob patterns). No new
  selector machinery; same matching rules as `find_sites`,
  `attach_hooks`, `do`.

  Default: nothing rendered per-layer (current behavior). Selective
  opt-in is critical — ResNet-50 with 50 layers is fine; LLaMA-7B
  with hundreds is unrenderably noisy. Force the user to choose
  what's worth showing.

  Built-in library (all afternoon-scale):
  - `tl.viz.heatmap()` — channel-averaged 2D heatmap with colormap.
  - `tl.viz.channel_grid(n=16)` — mosaic of first-N channels for
    conv activations.
  - `tl.viz.histogram()` — value distribution.
  - `tl.viz.attention_heatmap()` — attention matrix render
    (auto-detects shape).
  - `tl.viz.tsne()` / `tl.viz.pca_2d()` / `tl.viz.umap()` /
    `tl.viz.mds()` — 2D projection for embedding-shaped tensors
    (rows as points).
  - `tl.viz.violin()` — per-channel value distributions.

  Rendering timing: **eager at trace time**, not lazy at draw time.
  Tensors are in memory during capture; render and save thumbnails
  to the trace's tmp dir as each matched layer fires. Lazy rendering
  means re-traversing tensors on every `.draw()` call and risks them
  having been evicted under streaming/eviction mode. Eager + cached
  PNG paths makes draw-time cheap and idempotent.

  Save/load: visualizers are code (drop on save). Rendered PNG
  thumbnails CAN ride along in the portable bundle (small,
  self-contained, useful for offline viewing) — opt in via
  `tl.trace(..., save_visualizations=True)`. On load without code:
  you get the cached thumbnails. With code: re-render against
  loaded tensors.

  Shares infra with input/output rendering and the `tl.viz.batch_summary`
  helpers from the batched-input section above. Same montage code
  works for an activation grid as for a batch-of-images input node.

  Compound payoff: input image (cat) → conv1 channel grid → relu
  heatmap → attention pattern → top-5 output labels w/ confidences.
  Whole graph reads as a visual tour through the model's processing.
  Combined with the intervention API: fork, ablate, rerun, and
  visualizations change across the bundle — publication-quality
  figure with no extra code.

  Symmetric output_transform. Mirrors the input side: a callable that
  maps `model_output -> user_form` (logits -> ImageNet label, vocab
  logits -> decoded text, detection output -> labeled boxes). Stored
  on the trace alongside the input transform; runs once after the
  forward; populates `Trace.raw_output` for rendering and downstream
  use. The output node in the rendered graph shows the human-readable
  form (e.g. "Egyptian cat (45.2%)") in addition to or instead of the
  tensor shape. Same plugin dispatch as input rendering — built-in
  transforms include `tl.transforms.imagenet_topk(k=5)`,
  `tl.transforms.hf_decode(tokenizer, top_k=1)`,
  `tl.transforms.softmax_topk(k, labels=...)`, etc. Serialization
  policy matches input transform: code isn't pickled; only the
  computed user-form is saved (with size cap). `trace.rerun(...)`
  re-applies both transforms — preprocess the new input, decode the
  new output — so the rerun loop reads end-to-end. Killer compound
  case: text-in / text-out language models render the prompt on the
  input node, the predicted continuation on the output node, the
  whole graph in between. Trace as narrative.

  Trace stores transform; rerun reuses it. The `Trace` object holds
  a strong ref to the `transform` callable for its in-memory life so
  `trace.rerun(new_user_input)` re-applies it automatically:

  ```
  trace = tl.trace(model, "the cat sat", transform=tokenize)
  trace.rerun("the dog sat")          # auto-applies tokenize
  trace.rerun(["batched", "prompts"])  # same
  trace.rerun(tensor, transform=False) # bypass; raw model input
  ```

  Resolution rule: if the trace has a stored transform and
  `transform=False` isn't passed, apply it. Right default — users
  almost always want "feed me another prompt," rarely "I built
  tokens manually." Composes naturally with `fork` + `do` + `bundle`:
  one-line prompt swap inside an intervention experiment instead of
  the user keeping a tokenizer reference around.

  Loaded traces: transform isn't serialized (code with potential
  closures, not safe to pickle into a tlspec). `tl.load(path).rerun("...")`
  raises with a clear message: "This trace was loaded from disk and
  has no associated transform; pass `transform=callable` explicitly
  or rerun with a pre-transformed tensor." Don't try to be clever.

  Built-in transforms for common cases. Once the primitive ships,
  add a small library of pre-built transforms so users don't reach
  for `tokenizer(text, return_tensors="pt")` boilerplate. Same
  appliance pattern as recipes: small core registry, long tail in
  bridges.

  Sketch:
  - `tl.transforms.hf_tokenize(tokenizer)` -> callable. Wraps a HF
    tokenizer; returns a dict suitable for `**`-unpack into
    `model.forward`. Auto-applies chat template if the tokenizer has
    one and input looks like a message list.
  - `tl.transforms.hf_tokenize_for(model)` -> callable. As above but
    auto-resolves the tokenizer from the model. Lives in
    `tl.bridge.hf` since it depends on `transformers`.
  - `tl.transforms.image_preprocess(size=224, mean=..., std=...)` ->
    callable. Standard ImageNet-style preprocessing on a PIL Image
    or path string. Lives in `tl.bridge.vision` (later).
  - `tl.transforms.audio_resample(target_sr=16000)` -> callable.
    Audio resample to model's expected sample rate. Lives in
    `tl.bridge.audio` (much later, only if demand).

  Once these exist, the HF text-input ergonomic shrinks to:
  ```
  trace = tl.trace(model, "hello", transform=tl.transforms.hf_tokenize_for(model))
  ```
  And `tl.bridge.hf.trace_text(model, "hello")` is a 3-line
  convenience wrapper that fills in the transform default.

  Out of scope: animation / autoregressive-generation visualization
  (different feature; needs multi-trace alignment + SVG animation
  primitives). Defer.

  Total scope: ~100 lines for primitive + dispatch + per-modality
  renderers + rerun integration. Built-in transform library adds
  another ~50-100 lines per ecosystem (HF / vision / audio). Genuinely
  an afternoon for the primitive; another afternoon-or-two for the
  built-ins.

- HuggingFace bridge: text-input ergonomics for language models
  (raised 2026-05-07). TransformerLens lets users feed raw text to a
  language model and auto-applies tokenization + embedding lookup.
  Pleasant UX. We can match the ergonomics WITHOUT TransformerLens's
  architecture-reimplementation cost — we just preprocess the input
  and then run the genuine HF forward.

  Architectural placement: this lives in `tl.bridge.hf`, NOT in core
  `tl.trace`. The trap is making `tl.trace(model, x)` accept strings
  for some models and tensors for others — that couples core to the
  text-model domain, then vision wants `trace_image`, audio wants
  `trace_audio`, multimodal wants all-of-the-above, and core ends up
  knowing about every model domain. Keep core's contract clean:
  `tl.trace(model, x)` runs `model(x)`, no domain magic. The HF bridge
  is where HF-specific conveniences live, alongside the auto-detect
  recipe registry from the slicing-recipes todo.

  API surface:
  ```
  trace = tl.bridge.hf.trace_text(model, "Once upon a time")
  trace = tl.bridge.hf.trace_text(model, ["batch", "of", "prompts"])
  trace = tl.bridge.hf.trace_text(
      model,
      [{"role": "user", "content": "hi"}],
      chat_template=True,
  )
  ```

  What it does (~30 lines of HF-specific glue):
  1. Find the tokenizer. Use `tokenizer=` kwarg if passed; else look
     for `model.config.tokenizer_class` and `AutoTokenizer.from_pretrained`
     against the model's name-or-path. Fail loud if neither resolves —
     don't guess.
  2. Detect chat template (`tokenizer.chat_template is not None`) and
     apply when the user passes a string + `chat_template=True` or a
     list-of-message-dicts.
  3. Tokenize -> tensor (`return_tensors="pt"`).
  4. Move encoded inputs to the model's device.
  5. Call `tl.trace(model, **encoded_inputs)` — the trace is of the
     real HF forward pass, not a TorchLens reimplementation.

  Pairs with the slicing-recipes auto-detect: the same model-class
  identification machinery that fires `gpt2_combined_qkv@v1` recipe
  also picks the right tokenizer. Single registry, two payloads.

  Vision analogue (later, separate todo): `tl.bridge.vision.trace_image`
  could do PIL load + resize + normalize. Same pattern, different
  domain. Don't ship until there's demand — keeps the bridge surface
  scoped.

  What we DON'T do (deliberately different from TransformerLens):
  - No `HookedTransformer`-style reimplementation of architectures.
    TransformerLens's text-input magic is intertwined with their
    custom forward implementations; ours is preprocessing only. The
    genuine HF model runs; we just feed it tokens.
  - No global "register a text input" hook on the model. The bridge
    function is a one-shot wrapper, not a state-changing
    registration. Same reproducibility argument as recipes.

  Skip / non-goals: chat-history management, generation streaming
  (use HF's `model.generate(...)` and trace inside if needed), prompt
  templating beyond what `tokenizer.apply_chat_template` already
  provides. We're a tracing tool that happens to know how to feed
  text to a transformer; we're not an inference framework.

- Tensor-slicing recipes for semantic sub-addressing (raised 2026-05-07).
  Modern attention models pack Q/K/V into one tensor (`qkv_proj` output
  is `(batch, seq, 3 * num_heads * head_dim)`), then split + reshape
  per-head. Interpretability users want to address a specific head's
  query/key/value directly — TransformerLens does this by inserting
  named hook points into a custom attention impl. TorchLens shouldn't
  modify the model; instead, let users hand us a *recipe* that says
  "when you see this op, here's how to name and slice its output."

  Architecturally **layered on top of the `op.outs` API** owned by the
  tensor-container entry above. Recipes don't introduce a new dict;
  they add entries to the same `op.outs` populated by container
  introspection. Same lookup, same renderer story, same save/load
  contract. The provenance flag on `op.outs` is what distinguishes
  recipe-derived entries (views of a packed tensor; mutating one
  mutates the parent) from container-derived entries (independent
  tensors; safe to mutate in isolation). Ship containers first; recipes
  is incremental once that infrastructure exists.

  Lookup target after recipe application:
  ```
  trace["transformer.blocks.0.attn.qkv_proj.query"]
  trace["transformer.blocks.0.attn.qkv_proj.query.head_3"]
  ```

  User-facing API surface — three orthogonal supply mechanisms that compose:

  1. **Auto-detect (default-on).** `tl.trace(hf_model, x)` recognizes
     common architectures and applies built-in recipes automatically.
     Most users never write a recipe. Strict class-identity matching;
     no fuzzy fallback.
  2. **Explicit at trace time.** `tl.trace(model, x, slicing_recipes=[...])`
     accepts patterns (glob/regex over layer label or module path) plus
     slicing specs. Stacks with auto-detect by default; pass
     `auto_detect=False` to disable detection while keeping explicit
     recipes. For custom models or non-HF architectures.
  3. **Post-hoc on the trace.** `trace.apply_recipes([...])` on an
     already-captured (or `tl.load`-ed) trace. Recipes slice already-
     captured tensors — no re-execution needed. Two payoffs:
     (a) **iteration loop** — developing a recipe, you tweak the spec
     and re-apply without re-running the model (huge for big models).
     (b) **loaded traces** — apply recipes to a `.tlspec` from disk;
     the saved tensors are enough, recipe code re-runs locally.

  Disable everything: `slicing_recipes=False, auto_detect=False`.
  Useful for users who want raw `op.out` tensors only.

  Conflict resolution: when explicit + auto both define a key like
  `query` on the same op, **user-explicit wins**. Emit a
  `MultiMatchWarning`-style notification (mirror the existing pattern
  used for selector matches) so the user sees that an auto-detected
  recipe was shadowed.

  Inspect what fired: `trace.applied_recipes` returns
  `{"transformer.h.0.attn": "gpt2_combined_qkv@v1", ...}` so users can
  verify detection was correct. Critical for debugging silent
  mis-slicing — wrong slicing produces plausible-looking nonsense.

  Save / load behavior:
  - Recipes are NOT serialized into the portable bundle. Recipe code
    can carry closures / refs to user state; not safe to pickle into
    a tlspec.
  - `trace.applied_recipes` (just names + version tags) IS serialized
    so a loaded trace can re-resolve from the registry by name.
  - Custom user recipes that aren't in the registry: don't survive
    save/load. User re-applies on the loaded trace via
    `loaded.apply_recipes([...])`. Document this clearly.

  Out-of-scope mechanisms (deliberately rejected):
  - **Global registry** (`tl.recipes.register(...)`). Action-at-a-
    distance — same `tl.trace(...)` call gives different results
    depending on hidden global state. Hurts reproducibility. Pass them
    in explicitly.
  - **Mutating the model** (`tl.attach_recipes(model, [...])`).
    TorchLens stays a passive observer; we don't write recipe metadata
    onto `nn.Module` instances. Recipes live with the trace, not the
    model.

  Slicing spec options to consider:
  - Index tuples (general, verbose).
  - Reshape + named-axes (richer, matches einops' vocabulary).
  - Helper builders: `tl.slice(dim=-1, parts={"q": (0,h), "k": (h,2*h), "v": (2*h,3*h)})`,
    `tl.heads(num_heads=12, head_dim=64)`.

  Composition: `qkv_proj` -> `{q,k,v}` -> `{head_0..head_N}`. Keys in
  `op.outs` are dotted strings; `query.head_3` is a single flat key,
  not a nested object. Recipes register slices with dotted names
  directly. Slices are tensor views (no compute, no extra memory)
  where possible; materialize only on access.

  Auto-detection design:
  - Pattern-match strictly on module class identity
    (`type(submodule).__module__.startswith("transformers.")` plus the
    class name). False positives are catastrophic — wrong slicing gives
    plausible-looking nonsense activations. Better to fall back to
    "no recipe applied" than to mis-slice.
  - Built-in registry kept intentionally small in core: ship recipes
    for ~5 dominant architecture patterns (BERT-style, GPT-2 combined
    QKV, GPT-NeoX/LLaMA separate QKV, GQA/MQA, T5 with relative pos).
  - Long tail goes to a `torchlens-bridges-hf` extension package
    (appliance pattern, like the existing `bridge.captum` etc.).

  Recipe versioning + stability framing:
  - Recipes are explicitly a **convenience feature**. Core capture API
    gets strict semver; recipe behavior evolves more freely. Document
    this so users opt into looser stability promises by depending on
    them.
  - Every recipe gets a version tag (`gpt2_combined_qkv@v1`) recorded
    in `trace.applied_recipes`. If we ever fix a recipe, ship `@v2` and
    let users pin via `slicing_recipes={"@v1"}` or similar.
  - Convenience features tend to creep into being load-bearing — once
    `query.head_3` is in a published paper, you can't remove it. The
    versioning + provenance plumbing exists specifically to mitigate
    this.

  Hard constraint: recipes name SLICES of existing captured tensors —
  they must NEVER synthesize new ops. The moment a recipe says "now
  compute attention scores from these Q/K," it becomes a
  model-modification tool, not a metadata layer. That's the trap
  TransformerLens is in (their reimplementation of attention diverges
  from HF over time). Stay strict: recipes name slices, nothing else.

  Intervention surface:
  - `trace.do("...qkv_proj.query.head_3", tl.zero_ablate())` writes
    into the slice of the larger tensor. Semantics are well-defined
    (slice is a view, parent tensor is what flows downstream) but
    worth a doc-level safety note: mutating a slice propagates to the
    parent tensor's downstream consumers. That's the intended behavior
    for ablation, but a footgun if the user thinks the slice is
    independent. The provenance flag on `op.outs` lets the intervention
    API surface a clear "this is a view; mutation propagates" warning
    when needed.

  Visualization:
  - Default off — fan-out per head explodes node count.
  - `view='expand_outs'` (or `view='attention_heads'`) expands the
    sub-fields into rendered sub-nodes. Same renderer code path as the
    container fan-out from the entry above.

  Pairs naturally with the eventual HuggingFace bridge — ship pre-built
  recipes for common HF architectures so users don't hand-build for
  every model.

- Accept FX-style qualpath as a documented secondary lookup key
  (raised 2026-05-07). Coordinates with the metadata-field todo below
  (`OpLog.fx_qualpath`); ship the field first, then make it queryable.
  Goal: ease translation for users coming from `torch.fx.symbolic_trace`
  / TorchDynamo printouts who already think in
  `encoder.block_2.attention.relu` form. Lookup canonically still routes
  through `trace["relu_1_2"]`; this is a secondary form, not a replacement.

  Concrete shape:
  - Accept `trace["encoder.block_2.attention.relu"]` (structural path)
    and `trace["encoder.block_2.attention.relu_0"]` (path + FX-style
    disambiguator) and resolve to the corresponding OpLog.
  - Pick ONE FX flavor (recommendation: `torch.fx.symbolic_trace`'s
    convention, since it's the most canonical and most stable).
    Document the exact disambiguation rule WE commit to, independent of
    upstream FX evolution. If FX's rule diverges from ours later, ours
    stays put — we're a separate contract.
  - Add to the resolver's "did you mean…" suggestions when a near-miss
    is detected.

  Mandatory caveats in the docs:
  - PyTorch-only. Cross-framework callers (MLX, eventual tinygrad) will
    not have this lookup form.
  - Refactor-fragile, same as our existing `module_path` lookup
    (`trace["encoder.layer.0"]`) — renaming a wrapper or inserting a
    `Sequential` shifts the path. Not a regression vs current state.
  - Names will not always byte-equal what `torch.fx.symbolic_trace` /
    Dynamo / `torch.export` print, especially for re-invocations and
    around graph-break boundaries. We commit to a stable convention; we
    do not commit to tracking upstream FX naming changes.

  Don't add as a hidden / undocumented form. The right shape is
  "documented, caveated, mentioned" or nothing — see discussion in the
  2026-05-07 thread.

- Surface FX-style call-site qualpath as a metadata field (raised 2026-05-07).
  We deliberately don't accept `encoder.block_2.attention.relu` as a TorchLens
  lookup key (canonical lookup stays `trace["relu_1_2"]`), but the
  symbolic-trace-flavored string is useful for cross-tool reference and for
  users coming from FX / `torch.fx`. File as a non-key metadata field.
  Concrete shape:
  - `OpLog.fx_qualpath` (e.g. `encoder.block_2.attention.relu`) — the
    structural module-call path, stable across re-invocations of the same
    module method in one forward pass.
  - `OpLog.fx_call_index` (0/1/2/...) — disambiguator that mirrors FX's
    `relu_0` / `relu_1` suffixing when a method is called multiple times.
    Together these reproduce FX's `relu_0` form on demand.
  - `LayerLog.fx_qualpath` when all member ops share one path; otherwise
    expose as a list since a single LayerLog can span recurrent re-uses.
  - `ModuleLog` already has `module_address` (user-defined PyTorch path);
    the new field is specifically the call-site path, which can differ
    when the same module is invoked from multiple parents.
  Cost: negligible. We already track `op.containing_modules`; the qualpath
  is just a derived join. Naming MUST telegraph "metadata, not key" — pick
  `fx_qualpath` / `pytorch_qualpath` / `module_call_path`, NOT
  `module_address` (which would invite the lookup-pattern misuse we're
  avoiding). Document one caveat: FX/Dynamo's exact disambiguation rules
  (Sequential indexing, anonymous-module handling, `_modules` ordering)
  aren't a public spec and shift between PyTorch versions; promise
  consistent-with-our-rules output, not byte-equal-to-FX-forever output.

- Better support for "tensor container" data structures + the unified
  `op.outs` API (raised 2026-05-07). Many real models pass and return
  tensors via containers, not bare `torch.Tensor`s: HuggingFace
  `ModelOutput` dataclasses, `NamedTuple`s, Detectron2 `Instances`, dict
  / list / tuple of tensors. Today TorchLens handles a few cases ad-hoc
  (see `_move_nested_to_device`, `_collate_batch` in
  `torchlens/__init__.py`); there's no general pattern.

  This entry owns the **`op.outs` API design**. Both this feature and
  the recipe-slicing entry below populate the same dict; ship containers
  first because the auto-detection story is simpler (pytree introspection
  on the returned object, no user config). Once `op.outs` exists,
  recipes layer on top by adding more entries to it at log-time.

  Capture / log surface. When a forward function returns a container,
  recursively descend with `torch.utils._pytree` (PyTorch's own answer
  to this — see thread-local discussion) and populate `op.outs` with
  `{field_path: tensor}`. Field path becomes part of the lookup key
  (`trace["myblock_1.logits"]`) and the alignment key for Bundles.

  Unified `op.outs` API:
  - `op.out` — primary output tensor (current behavior, unchanged).
    For container outputs, this returns the container as-is so users
    can pass it through downstream code that expects the original
    structure.
  - `op.outs: dict[str, Tensor]` — flat dict of named tensor leaves.
    Keys are dotted paths for nested structure (`past_key_values.0`,
    `attentions.3`). Mirrors PyTree's `tree_flatten` semantics.
  - `op.outs_provenance` (or per-key metadata, TBD): a small flag
    indicating whether each entry is independent ("container") or a
    view into the parent ("recipe"). Drives mutation-safety semantics
    in the intervention API — see the recipe entry below for why this
    matters.

  Visualization. Container outputs render as a single node whose label
  lists the field names + per-field shapes (compact "struct" view). For
  power users, a `view='unpack_containers'` mode (or a unified
  `view='expand_outs'`) splits into one child node per field with a
  dotted edge from container -> field. Default to compact; opt in for
  fan-out. Same renderer hook serves recipe-derived outs.

  Save / load. `op.outs` for containers is reconstructable from the
  pytree-flattened leaves + the `treespec`. Save the leaves + spec in
  the portable bundle; reconstruct the dict on load. Recipe-derived
  entries are dropped on save and re-applied on load by replaying the
  recipe.

  Skip: full custom-class introspection (arbitrary user classes with
  hidden tensor attrs). Stick to PyTree-registered types + the common
  shipped containers (NamedTuple, dataclass, dict, list, tuple). Users
  can register their custom classes via the standard PyTree API.

- Multi-arm conditional traversal / "all paths" view (raised 2026-05-07).
  Today TorchLens captures only the executed arm; unfired arms appear in
  `event.branch_ranges` (AST) and `Conditional.arms` (with `fired=False`)
  but have no recorded ops, no tensors, and don't show up in the graph.
  Worth surfacing the full source-static control flow, not just the
  taken path. Two tiers, ship in order:

  Tier 1 — Static AST sketch. From `event.branch_ranges` + the source
  file, extract per-arm source spans and render placeholder nodes for
  each unfired arm under the bool node, labeled `THEN/ELIF_N/ELSE`,
  styled distinctively (dashed border, gray fill) to mark "not executed
  this run". No tensor metadata, no FLOPs. Cheap, safe, opt in with
  `view_unfired_arms=True` (or via a `view='all_paths'` knob). Fixes
  the "I see THEN but no ELSE in the graph" surprise users hit on
  asymmetric conditionals.

  Tier 2 — Multi-run traversal Bundle. User supplies N inputs that
  collectively trigger every arm; TorchLens traces each, then unions
  them into a single rendered view that shares the common prefix and
  forks at each `Conditional`. Implementation sketch: build a
  `Bundle({'arm=then': trace_pos, 'arm=else': trace_neg, ...})` and add
  a `bundle.show_paths(...)` renderer that detects which Conditional
  fired in each member and aligns arms accordingly. Reuses the existing
  Bundle alignment + supergraph plumbing. Naming TBD —
  `bundle.show_paths()` or `trace.show_with_alternatives(others)`.
  Bonus: a `trace.suggest_alt_inputs()` helper that, given one trace,
  uses the bool's source location + recorded inputs to suggest a
  perturbation that flips the arm — quick smoke way to enumerate
  paths without hand-crafting inputs.

  Forcing-by-override (monkey-patch the bool to return False when it
  evaluated True) is tempting but unsafe: branches often depend on
  tensor shape / dtype / NaN, and forcing the "wrong" arm can crash
  or produce malformed tensors that propagate. Skip for v1; let the
  user supply real inputs.

- Combined forward+backward graph rendering (raised 2026-05-07). Today
  `trace.draw()` shows forward only and `trace.draw_backward()` shows
  the autograd graph only. Niche but valuable: a single SVG that places
  forward op nodes alongside their corresponding `grad_fn` nodes, with
  a distinctive edge style linking each forward op to its backward
  partner (e.g. dashed light-purple, matching the existing
  `GRADIENT_ARROW_COLOR = "#9197F6"`). Useful for: (a) teaching how
  autograd actually wires the graph, (b) debugging which forward op
  produced a specific grad_fn (the `[i]` "intervening" nodes are not
  obvious), (c) per-op autograd-saved-bytes attribution at a glance.
  Default off; opt in with `view='both'` (or similar) when
  `trace.has_backward_pass`. Watch for visual density on real models
  (ResNet-18 already has ~70 forward nodes; doubling makes it noisy)
  — recommend pairing with the existing `module=` focus and skip/collapse
  knobs so users can scope the combined view to a region of interest.
  Adjacent to the `backward_pass_sprint.md` parking lot.

- Some Trace-level intervention/inspection verbs should ALSO be available
  on the resolved log object (raised 2026-05-07). Today the user writes
  `trace.do('conv2d_2_1', tl.mean_ablate())`, which is fine but somewhat
  awkward when you already have a handle on the layer/op. A user with
  `op = trace['conv2d_2_1']` (or `trace.layers[...]` / `trace.ops[...]`)
  would naturally expect `op.do(tl.mean_ablate())` to apply the same
  intervention scoped to that op, no label repetition. Same pattern for
  other selector-first verbs (`set`, `attach_hooks`, `replay`, `find_sites`
  scoped to this op as default, etc.). Implementation note: keep Trace as
  the source of truth for the intervention spec; LayerLog/OpLog methods
  should resolve back to the owning Trace and call its method with the
  pre-resolved label/site. Watch for the case where a LayerLog is detached
  from its source Trace (e.g. after a load) — surface a clear error rather
  than silently no-op. Surface during a UX-focused naming/ergonomics pass
  before the 2.0 marketing push.

- Bundle diff color scale is semantically wrong (raised 2026-05-07).
  `torchlens/visualization/bundle_diff.py:_delta_color` uses a
  blue → white → red gradient over `[0, 0.5*max, max]`. Diverging
  blue-white-red is conventionally signed: white = neutral/zero,
  blue = negative, red = positive. The L2 delta we render is
  unsigned and bounded `[0, max]` — zero IS the neutral point.
  Fix: switch to a sequential single-hue colormap (e.g. white → red,
  or pale-blue → dark-blue) so the visual rests on white-ish at zero
  delta and saturates at max, with no implication of "directionality".
  Update legend labels accordingly (`zero` / `mid` / `max` rather than
  `low` / `mid` / `high`). Sites: `_delta_color`, `_add_legend`, and
  the snapshot/demo SVGs. Surfaced during the 2026-05-07 demo notebook
  push.

- Rethink the parameter name `activation_postfunc` itself. Current name is
  awkward (`-postfunc` suffix) and the semantic now reads as a "transform"
  hook, not a "post-processing function" (after the raw-vs-transformed
  split landed in PR #166). Candidates: `activation_transform`,
  `activation_hook`, `transform_activation`. Keep `activation_postfunc` as
  a deprecated alias for at least one minor release. Defer to a
  UX-focused naming pass.

- Estimated autograd_saved_bytes via static formula (no graph required).
  Companion to the introspection-based `autograd_saved_bytes` field
  shipped in PR #165: a per-op lookup table keyed on forward function
  name + input/output tensor shapes that returns the expected bytes
  autograd WOULD save if `requires_grad` were on. Useful for what-if
  estimation in `inference_mode` / `no_grad` workflows. Maintenance cost:
  needs PyTorch version pinning + tests for table accuracy across
  releases. Defer until a user actually asks; introspection covers the
  90% case.

- Auto-published model menagerie (replace manual Google Drive). Design
  notes: `.project-context/research/menagerie_revamp.md`. Hybrid CI
  (smoke gallery on PR + full on release) -> GitHub Pages, PDFs as
  release assets, generalize `build_torchlens_theme_gallery.py` as
  template.

- Per-grad_fn auto-computed memory cost. Once GradFnLog has
  saved_for_backward refs from the backward-pass sprint, memory cost per
  grad_fn = sum of saved tensor sizes + output gradient shapes plus
  type-specific contributions. Distinct from the per-op
  `autograd_saved_bytes` shipped in PR #165 -- this is a per-grad_fn
  view (backward-graph node accounting). Currently using explicit
  peak-memory capture per backward sweep. Design notes:
  `.project-context/research/backward_pass_sprint.md` (parking lot).

- Fastlog gradient support (PR C). Predicate-selected gradient capture
  in fastlog. As of 2026-04-27, slow-path backward IS settled (PRs
  #161-163, #165), so this is unblocked architecturally -- gated only
  on JMT direction to dispatch (a research-and-spec sprint). Design
  context: `.project-context/research/backward_pass_sprint.md`. Once
  shipped, fastlog `gradient_postfunc` parity (mirroring slow path
  behavior added in PR #166) becomes the natural followup; do NOT add
  `gradient_postfunc` to fastlog BEFORE gradient capture lands per
  research conclusion in
  `.project-context/research/fastlog_postfunc_parity_2026-04-27.md`.

- Document activation_postfunc / gradient_postfunc portable-save
  persistence story. PR #166 made the callable-drop / repr-keep
  behavior explicit in code, but the user-facing rationale is not
  surfaced in the public docstring or README. Add a short note
  explaining that `torchlens.save` strips the callable for portability
  and retains only `activation_postfunc_repr` /
  `gradient_postfunc_repr` strings as a record. Source: postfunc
  review Finding #7 (`.project-context/research/postfunc_review_2026-04-27.md`).

- Reduce first-call cost of `patch_detached_references()`. Profiling
  audit measured 16.7s cumulative / 8.64s self time on a small smoke
  model -- 70% of total runtime. The cost is one-time (caches across
  calls) but cold-start UX is rough. Options: idempotent re-entry
  guard, narrower `sys.modules` crawl, or an opt-out for environments
  that don't need detached-import patching. Source: profiling audit
  Finding #5 (`.project-context/research/profiling_audit_2026-04-27.md`).

- Bound the AST/file cache for long-running services. Caches in
  `torchlens/postprocess/ast_branches.py` persist by filename and
  aren't bounded. The 50-iteration leak loop confirmed flat behavior
  in normal use (no leak), but daemon-style processes that touch many
  unique source files could accumulate parsed ASTs over time. Add an
  LRU cap and/or a public cache-clear API. Source: profiling audit
  Finding #6.

## Completed (recent)

### 2026-05-10 draw + postprocess + intervention cleanup sprint

- `1c31b6b` **Trace.draw() ergonomics.** Deleted the
  `Trace.render_graph` shim and updated stale code/test/doc references to
  `draw`. Kept `view=True` behavior for interactive use, but headless Linux
  / non-forwarded SSH now renders the file and skips viewer launch with one
  concise stderr line.
- `8dd33a6` **Postprocess Step 6 elimination.** Deleted
  `_fix_modules_for_internal_tensors`; module-stack suffixes are appended to
  `equivalence_class` at op creation. Verified six-model
  equivalence-class parity against the old postprocess suffix path.
- `da15b5f` **Intervention API v1 maintenance sweep.** Resolved original
  items 1-7 and 10:
  1. §4.2 hook positional-arg-name clarification was already present.
  2. §12.16 attribution-patching formula sign convention documented.
  3. `FrozenTargetSpec` concrete frozen dataclass was already present.
  4. `intervention_spec` cached-property invalidation test was already
     present; smoke marker added.
  5. `_construction_done` / `object.__setattr__` guard pattern and tests
     were already present.
  6. Rerun state replacement was already a single `__dict__.update(...)`;
     added an explicit KeyboardInterrupt-before-swap regression test.
  7. `tl.list_logs()` / `_state.list_logs()` already returned tuple
     snapshots; concurrent smoke coverage marker added.
  10. TransformerLens `act_patch` -> TorchLens `tl.bwd_hook + rerun`
      migration row added.
  Items 8 and 9 remain active for the naming sprint on 2026-05-11.
- Tensor-id-keyed metadata was marked SUPERSEDED by `_tl` namespace refactor
  commit `0e4509d`; original text retained for history.

### 2026-04-27 multi-trace sprint (Phase 1 + Phase 2)

Versions shipped: 2.14.0 (auto-released after Phase 1 merge); Phase 2
release pending semantic-release run on Phase 2 merge (expected 2.15.0).

- PR #170 -> 2.14.0: Phase 1 data layer. New `torchlens/multi_trace/`
  subpackage with `TraceBundle`, `NodeView`, `Supergraph`,
  `compare_topology`, diff metrics (cosine / relative_l2 / pearson +
  scalar L1 fallback). Holds N `ModelLog` instances by reference;
  unified handling of shared-topology and divergent-topology bundles
  (degenerate Overlay model -- one class, not two). 28 tests; ruff +
  mypy + smoke clean.
- PR #171 -> CLOSED (superseded by #172). Initial Phase 2 visualization
  pass shipped working bundle rendering but with judicious duplication
  vs `rendering.py` and simpler module clusters than `show_model_graph`.
  Architect rejected pre-merge.
- PR #172 -> Phase 2 polish + visualization. Three commits on top of
  the closed-#171 state:
  - `refactor(visualization): extract reusable rendering primitives`
    -- pulled cluster styling, file-format dispatch, direction
    translation, HTML escape into `torchlens/visualization/_render_utils.py`;
    adopted in BOTH `rendering.py` AND `elk_layout.py` (genuinely shared,
    not bundle-private).
  - `feat(multi-trace): bundle renderer refactor + module cluster
    aesthetic parity` -- bundle renderer 779 -> 352 LOC (-55%); module
    cluster penwidth/pass-suffix/nesting parity with `show_model_graph`;
    bonus fix for divergence-mode crash on multi-pass canonical nodes.
  - `test(multi-trace): cluster parity tests + defer rolled/backward
    bundle modes` -- 4 cluster-parity regression tests + the two
    deferred-mode todos in this file.
  Single-trace regression tests pass byte-equivalent
  (`test_output_aesthetics`, `test_visualization`,
  `test_backward_visualization`, `test_dagua_theme` all green).

Sprint notes:
- Codex hit daily quota mid-Phase-1 dispatch; pivoted to Claude
  general-purpose subagents for all three dispatches per the
  fallback procedure in `~/.claude/CLAUDE.md`.
- PR #171 was closed (not merged) after the architect flagged
  insufficient refactoring + missing cluster aesthetic parity. Same
  branch carried forward into PR #172 -- no force-push, three
  commits on top, clean history.

### 2026-04-27 grab-bag + activation_postfunc + perf sprint

Versions shipped: 2.10.0 -> 2.13.0 (six PRs).

- PR #164 -> 2.10.0: `extra_data` + `input_metadata` plumbing on
  LayerLog / LayerPassLog / ModelLog (open-ended user dicts).
- PR #165 -> 2.11.0: `autograd_saved_bytes` per-op introspection
  (distinct from existing `saved_activation_memory` -- autograd's
  saves vs torchlens's own).
- PR #166 -> 2.12.0: `activation_postfunc` raw-vs-transformed split,
  `train_mode` hardening, `ActivationPostfunc` / `GradientPostfunc`
  type aliases, `TorchLensPostfuncError`, README + docstring expansion.
- PR #167 -> 2.12.1: two-pass mode in-place module fix (regression
  source: commit 326b8a90, fast-mode pass-through detector mistakenly
  treating in-place modules as pass-throughs because input labels were
  read AFTER `orig_forward()` already mutated them).
- PR #168 -> 2.12.2: perf bundle, 3-of-3 fixes (bytecode
  column-offset cache, Step 5 branch-attribution fast-skip, CUDA probe
  guard for CPU runs). Shipped via Claude general-purpose agent after
  codex hit daily quota.
- PR #169 -> 2.13.0: fastlog `activation_postfunc` parity. Intentional
  architectural divergence from slow path per
  `.project-context/research/fastlog_postfunc_parity_2026-04-27.md`
  (parallel `transformed_*_payload` fields on `ActivationRecord`,
  postfunc runs in `_storage_resolver` post-predicate, predicates
  still see raw metadata, no `gradient_postfunc` until fastlog gets
  gradient capture). Also via Claude agent.

Research reports committed in `.project-context/research/`:
- `postfunc_review_2026-04-27.md`
- `fastlog_postfunc_parity_2026-04-27.md`
- `profiling_audit_2026-04-27.md`
- `two_pass_diagnostic_2026-04-27.md`

Other artifacts:
- `~/.claude/CLAUDE.md` hardened with a codex-quota-exhaustion
  fallback procedure (added 2026-04-27 incident section under Codex
  Dispatch).
- `.gitignore` extended to ignore `/modelgraph.*` and
  `/backward_modelgraph.*` (default `vis_outpath` filenames that were
  polluting the repo root).

### Intervention API v1 implementation spec hardening (raised 2026-04-29)

**Status:** pre-implementation hardening; not architecture/UI work.

After 4-round adversarial review of the intervention API architecture/UI plan,
codex reviewers identified pre-implementation contract gaps that belong in
the v1 implementation spec (NOT the architecture plan):

1. `_is_fork=False` default-fill for old loaded logs (FIELD_ORDER exact-key
   check on rehydrate).
2. `_derivatives.clear()` cascade semantics — unregister children only, or
   recursively call `child.cleanup()`?
3. §11.5 `tl.check_spec_compat` Plan B equivalence predicate — exact rules
   for parent-label / module-address / shape / func-name / pass-context
   comparison.
4. §13.8 typed exception hierarchy — base class + payload contract:
   `site`, `selector`, `matched_sites`, `structured_diff`, `original_error`.

Address these when MVP code lands.

### Intervention API DECISIONS-FOR-JMT (surfaced 2026-04-29)

10 decisions adopted as conservative defaults during the 4-round review.
JMT can override before implementation. Top four:
- `tl.intervene` in MVP (kept; vs N+1 entry-points concern).
- `tl.skip_module()` in MVP (kept; vs shape-contract complexity).
- `gradient_zero/scale` in MVP framing (kept as observation-only labeled).
- Auto-promote-with-warning for capture flags (kept explicit error).

Plus 6 more in `<vault>/2026-04-29-intervention-plan/ADVERSARIAL_REVIEW.md`.

### Intervention API: holistic naming pass before TorchLens 2.0 marketing push (already in todos)

Plan uses placeholder names throughout. JMT runs naming pass before public
API lock. Strong candidate: `tl.do(...)` for declarative R2 entry (Pearl's
do-operator).

### Intervention API — gradient interventions in v1 (raised 2026-04-29 mid-redraft)

**Status:** intentionally deferred; restore in v1.

MVP ships with FORWARD-ONLY intervention (replay path detaches saved
tensors). `tl.gradient_zero()` and `tl.gradient_scale()` were dropped
from the MVP helper list per JMT decision — observation-only framing
was confusing users into thinking trainable interventions worked, when
in fact they need v1's differentiable replay regime.

For v1, restore as:
- `tl.gradient_zero()` and `tl.gradient_scale()` helpers (clean signature
  matching forward helpers).
- Tier 1 backward via `tensor.register_hook` (~100 LOC) gated by
  `train_mode=True`.
- Tier 2 grad_fn-level via existing `_make_grad_fn_hook` infra in
  `capture/backward.py`.
- The differentiable_replay regime (separate from gradient helpers; this
  is the bigger architectural piece) is the v1 milestone that unlocks
  attribution patching, DAS, ReFT.

Tier 3 (training-time gradient routing, Cloud et al. 2024 style) remains
deferred indefinitely — out of scope for any version of this API; users
needing it should use specialized gradient-routing libraries.


### Intervention API — chunked-batching append nuances (raised 2026-04-29)

**Status:** design locked at high level; per-call state fields nuance to revisit at implementation time.

`log.rerun(model, x_new, append=True)` cats activations along batch dim
(dim 0). Use case: chunked batching when full stimulus set doesn't fit in
one forward. Result is indistinguishable from running everything in one
big batch, just spread across multiple physical forwards.

Constraints raised at append-call time:
- Same model object
- Same recipe (intervention_spec unchanged)
- Identical non-batch shape, dtype, device
- Same graph topology

**Per-call state fields nuance** (to revisit during implementation):

A few LayerPassLog fields are per-physical-call, not per-stimulus:
`func_rng_states`, `func_time`, `func_autocast_state`. These don't
naturally cat along batch dim.

MVP plan (option 1 — simplest):
- Keep LAST chunk's value for these fields
- Mark log with `is_appended=True` flag
- Validation surface knows: skip RNG-replay validation across chunks,
  only validate within the last chunk

Alternative options to consider during implementation:
- Option 2: Store list per chunk with `appended_chunk_boundaries: [4, 10, 15]`. Cleaner but adds plumbing.
- Option 3: Disable RNG-replay validation on appended logs entirely.

Lean option 1 for MVP; revisit if real users hit replay-validation needs
on chunked logs.


### Intervention API — revisit `tl.skip_module()` (raised 2026-04-29 mid-redraft)

**Status:** dropped from MVP per JMT decision; can be added back later.

`tl.skip_module()` was originally a helper for the "skip this module entirely"
case. With the simplified ontology, `tl.splice_module(nn.Identity())`
covers the same ergonomic. Dropped to keep MVP helper count tight at 11.

Add back in v1+ if real users hit ergonomic friction with the
`splice_module(Identity())` pattern. The contract for `skip_module` is
non-trivial:
- Multi-input modules require `input_index=` to specify which input
  becomes the output
- Shape compat pre-flight (output shape must match input shape at the
  selected index)
- Output structure handling for tuple/dict-returning modules

If revisited, address the contract first.

### Intervention API — revisit relationship enum after MVP usage (raised 2026-04-29)

**Status:** locked verbose-but-honest names for MVP; revisit naming
after real users have given feedback.

Current enum: `SAME_MODEL_OBJECT_AT_CAPTURE`,
`SAME_PARAM_SHAPES_AT_CAPTURE`, `SAME_WEIGHTS_AT_CAPTURE`,
`SHARED_GRAPH_SHAPE`, `LINEAGE`, `SAME_INPUT_OBJECT_AT_CAPTURE`,
`SAME_INPUT_SHAPE`, `DIFF_MODEL`, `UNKNOWN`.

These are evidence-typed (e.g. `SAME_MODEL_OBJECT_AT_CAPTURE` means
"we observed `id(model)` matched at capture time" NOT "weights are
still equal"). Verbose but precise.

After MVP ships, observe:
- Are users actually surprised by the verbose names?
- Is the evidence-typing distinction earning its keep?
- Could simpler names (e.g. `SAME_MODEL` instead of `SAME_MODEL_OBJECT_AT_CAPTURE`)
  work without losing meaningful precision?

Revisit with real usage data.


### Intervention API — REDRAFT REQUIRED post-session 2026-04-29

**Status:** PLAN.md v4.1 in vault is OBSOLETE post-this-session; major
ontology shift via JMT pushback session. Redraft to v5 needed before
implementation.

Key shifts (from v4.1 → v5):
- Single `Bundle` type (drop `DerivativesBundle`); `bundle[str]` → ModelLog
  by name; `bundle.node(s)` → NodeView (lazy supergraph; relationship-gated)
- Drop `log.derivatives` field entirely; users hold their own results
- Drop `apply_intervention` (replaced by canonical `do` / fork+attach+replay
  flow)
- Drop `tl.intervene` (covered by `do` polymorphism)
- Drop `_is_fork` marker (mutation is fine; fork to preserve)
- Drop `tl.fork` top-level (method-only; `tl.do` kept top-level for Pearl pitch)
- ModelLogs auto-named at `log_forward_pass` time: lowercase + strip HF
  class suffixes (`LMHeadModel`, `ForCausalLM`, `ForSequenceClassification`,
  `ForTokenClassification`, `Model`)
- ModelLog.name in PORTABLE_STATE_SPEC as KEEP (round-trip preserved)
- Mutate-in-place ontology — operations modify log directly; fork to preserve
- Two propagation engines: `replay` (saved-DAG, cheap, doesn't detect graph
  changes) vs `rerun` (wrapper-based, full forward, detects graph changes)
- Five foundational verbs: `set`, `attach_hooks`, `do`, `replay`/`rerun`, `fork`
- `set(site, value_or_fn)` accepts tensor (one-shot value) OR callable
  (one-shot apply, fn discarded — distinct from sticky `attach_hooks`)
- `do(site, value_or_hook, model=None, x=None)` — polymorphic; presence of
  `model=` switches replay→rerun
- `replay_from(site)` verb suffix (Python `from` is reserved word)
- `attach_hooks` composition: same-site hooks stack in attach order
- Direct attribute writes allowed but non-canonical (no spec tracking)
- 11 MVP helpers (dropped `skip_module`, `gradient_zero`, `gradient_scale`)
- Visualization defaults: bold magenta border for intervention site,
  subtler magenta border for cone; explicit color picked at prototype time
- Cone shading: opt-in via `vis_show_cone=True`; default just marks
  intervention sites
- `find_sites(query)` is method-only on ModelLog; returns custom
  SiteTable class with rich repr + `.labels()` / `.where(...)` / `.first()`
- Save/load: keep three levels (audit/executable_with_callables/portable);
  audit is useful and cheap to support
- `tl.adapters.*` deferred to v1; MVP uses `tl.splice_module` directly
- `ControlFlowDivergenceWarning` by default for replay; users reach for
  `rerun(model)` for the safety check
- `rerun(model, x_new, append=True)` cats along batch dim for chunked
  batching; constraints raise on mismatch (model identity, recipe,
  shape, dtype, device, topology)
- Site fan-out behavior: typed selectors fan out silently; bare strings
  emit `MultiMatchWarning` once when N>1; `max_fanout=N` cap; saved
  specs (strict mode) require typed selectors only
- Top-level partition: `tl.*` for STATIC (helpers, types, constructors);
  log methods for OPERATIONS

Redraft PLAN.md to v5 once design has stabilized further. Until redraft:
this todo + prior synthesis docs are the source of truth.

### Intervention API — detailed user manual (raised 2026-04-29)

**Status:** todo for after MVP code lands.

The PLAN.md is architecture/UI design intent. Users need a separate
detailed user manual:
- Quickstart / first 5 minutes
- Cohort migration guides (TL, NNsight, Pyvene, baukit) with
  side-by-side translation
- Method reference per verb with examples
- "When to use replay vs rerun" guide (the safety/cost tradeoff)
- "When to fork" guide (preservation patterns)
- Visualization gallery
- Recipes: activation patching, paired-prompt, IOI heatmap,
  steering, ablations, custom hook patterns
- Save/load workflow guide
- Error catalog with copy-paste fixes
- Limits and visibility classes (when does the moat end?)

Deliver alongside MVP. Sphinx or mkdocs site.


### Intervention API — SESSION_FINAL_DECISIONS doc is now CANONICAL

**Status:** post-context-reset, this is the source of truth.

The full record of the post-round-4 pushback session lives at:
`/home/jtaylor/Documents/Second Brain/brain/projects/torchlens/reports/2026-04-29-intervention-plan/SESSION_FINAL_DECISIONS.md`

That document contains:
- 59 numbered locked decisions from the session
- Comprehensive ontology rewrite (mutate-in-place, single Bundle type,
  five foundational verbs, two propagation engines)
- Updated API surface (top-level vs methods)
- Test plan / quality bars
- Open todos and naming workstream items

Read SESSION_FINAL_DECISIONS.md FIRST before referencing PLAN.md v4.1
which is now obsolete in many places.


### Consider Marimo for `notebooks/total_audit/` (deferred — file 2026-05-01)

**Status:** idea filed, not scheduled.

The 24 maintainer-facing audit notebooks may be a good fit for Marimo
(reactive .py-based notebooks) instead of `.ipynb`. User-facing galleries
in `examples/` should stay on Jupyter (audience expects it).

**Why Marimo could win for `total_audit/` specifically:**
- Single-author audience (JMT) — no friction from forcing a new tool on users.
- Reactive dataflow eliminates stale-state bugs ("worked when I ran it!")
  which were the #1 source of friction during the sprint.
- Plain `.py` files diff cleanly in PRs; `.ipynb` JSON diffs are hostile.
- Notebooks run as scripts (`python notebooks/total_audit/27_*.py`), so
  the audit refresh contract simplifies — no papermill needed for that gallery.
- Editing a public-API check no longer requires manual cell re-execute.

**Cost estimate:** ~1 day port + ~30 lines added to
`scripts/generate_audit_coverage_manifest.py` and
`scripts/check_audit_coverage.py` for `.py`-format detection.

**Trigger to revisit:** when the audit notebooks have seen real maintenance
use and stale-state friction is felt, OR when audit infra is being touched
for another reason. Until then: defer.


### Borrow circuit-tracer UI primitives (filed 2026-05-01)

Source: `brain/projects/torchlens/reports/2026-04-28-intervention-api-research/03_other_tools_landscape.md` §9 (circuit-tracer / Anthropic Fellows, May 2025). Three primitives we flagged but didn't ship in 2.x. All target the deferred interactive HTML viewer (`torchlens/viewer/` appliance stub).

#### A. User-driven supernode grouping in the viz (HIGH INTEREST)

JMT note 2026-05-01: "Rules for merging nodes in the visual is quite interesting eh."

Today TorchLens has THREE built-in grouping mechanisms in the viz:
- **Module nesting** — `containing_modules` defines a tree of clusters.
- **Loop iso-groups** — loop detection collapses repeated iterations.
- **`vis_nesting_depth`** — coarse fixed-depth collapse.

Missing: **user-driven, ad-hoc, persistent grouping** ("supernodes"). User clicks N nodes, hits "group", names the group, optionally annotates it. The grouping persists in a sidecar file (`.tlspec` extension or separate `.tlview` file). On reload, viz applies the user's grouping on top of the structural grouping.

Design questions to surface when we get to it:
- **Predicate-based vs explicit-set-based** grouping? Explicit set is what circuit-tracer does. Predicate-based would let "all conv layers in encoder" be one supernode programmatically. Probably support both.
- **Composition rules** — can supernodes nest? Can they overlap? (circuit-tracer: supernodes nest but don't overlap.)
- **Edge handling** — when N nodes collapse to one supernode, edges in/out of the group merge how? (Bundle, sum, max-weight?) Affects FLOPs/memory aggregates shown on the supernode label.
- **Persistence format** — supernodes are a viz overlay, but if they're stored alongside the `.tlspec`, they travel with the model log. JSON sidecar or first-class field in the manifest?
- **NodeSpec interaction** — `NodeSpecFn` already lets users customize per-node rendering. Supernode grouping is the structural layer above that.

This is a fork-tag for the eventual interactive viewer push: implement grouping via the API first (programmatic supernodes via `VisualizationOptions.groups=[...]`), THEN add UI affordances on top (click → group → name → save).

Cost estimate: medium (1-2 weeks for the API + render-side; UI affordances depend on viewer choice).

Trigger to revisit: when interactive viewer work begins, OR when JMT requests it for a specific demo where grouping by hand would help.

#### B. One-click "Steer" button from viz nodes

circuit-tracer / Neuronpedia: every graph node has a Steer button — click → type a value → re-run forward → see perturbed output. Intervention is one click away from the visualization.

TorchLens today: intervention is API-driven (`log.find_sites(...).attach_hooks(tl.zero_ablate())` then `rerun(...)`). All the machinery is there; the missing layer is the UI affordance that maps "click a node + pick a verb (zero / scale / set / steer) + set a value + run" to that API call.

Cost estimate: cheap once the interactive viewer exists. Hard part is the viewer, not the steer wiring.

Trigger to revisit: ditto interactive viewer.

#### C. Attribution-style portable graph file with prune-then-render pipeline

circuit-tracer's three-stage architecture: attribute → prune to graph file → render+intervene. The graph file is a portable JSON artifact; you can share it across machines, version it, regenerate viz without re-running attribution.

TorchLens already has `.tlspec` as the portable bundle (covers the "graph file" + activations role). What we DO NOT have:
- A **pruning step** that takes a `ModelLog` and reduces it to "high-effect nodes only" by some attribution-like score (gradient-attribution, integrated gradients, layer-importance, FLOPs/memory threshold, user-selected metric).
- An **attribution primitive** itself — TorchLens captures activations, not attributions. Adding attribution would mean either: (a) shipping a small attribution helper that runs on top of `ModelLog`, or (b) consuming attributions from Captum / circuit-tracer and rendering them.

The interesting cross-cut: TorchLens could ingest a circuit-tracer attribution graph and render it in the same viz layer the user already knows. That positions TorchLens as the universal visualization substrate; circuit-tracer (and others) become attribution producers.

Cost estimate: small if we just ingest external attribution graphs; medium if we ship our own attribution helper.

Trigger to revisit: when attribution becomes a felt need, OR when we want a circuit-tracer interop story for the 2.0 marketing push.
