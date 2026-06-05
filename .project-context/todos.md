# Task & Bug Tracker

## Architectural endpoint (JMT 2026-05-15)

> "Configurable capture + postprocess + output, all on one substrate. Pay only for what you use."

The destination architecture is a single capture path with orthogonal configuration axes (capture predicates, retention level, postprocess stages, output type). Every TL capability collapses into a position on these axes; no parallel implementations.

**Orthogonal axes:**
- **Capture**: which ops fire wrappers (always all), what predicate retains them, what ctx fields get built (lazy)
- **Retention**: tensor save mode (none / reference / copy), metadata depth (event-only / event+graph-position / full)
- **Postprocess**: which stages run (none / graph build / labels / intervention DAG / viz prep / all)
- **Output**: Recording (raw events) / PartialTrace (some stages) / Trace (all stages)

**Why this is the elegant endpoint, not a local maximum:**
1. Factoring is orthogonal — new capabilities are knobs on existing axes, not new code paths
2. "Pay only for what you use" is provably stable — future optimizations preserve it; future features work with it
3. Survives feature accretion without tangling — architectural property, not a discipline

**Architectural story for docs:**
> TorchLens captures eager-mode tensor execution through a single configurable substrate. Every capability — sparse capture, full Trace, halt, intervention, validation — is a configuration of one underlying machine, not a parallel implementation. Backends (PyTorch today, MLX today, JAX and others future) plug in via a small adapter protocol; everything above the adapter is framework-agnostic. Feature additions do not slow the core path: there is only one core path to protect, and it works across any tensor framework that wraps ops.

This frames the future of the project. Every refactor / sprint should ask: does this move us toward or away from the single configurable substrate + pluggable backend?

**Backend-agnostic layering (JMT 2026-05-15):**

```
User API (tl.trace) → Capture orchestrator → Backend protocol → {Torch, MLX, future JAX}
                                                              → Event stream (framework-agnostic)
                                                              → Postprocess pipeline (framework-agnostic)
                                                              → Output (Recording / Trace, framework-agnostic)
```

Backend protocol is small: `install_wrappers()`, `prime_model(model)`, `emit_event(kind, ctx)`, `teardown()`. Each backend is ~200 LOC of adapter rather than a parallel capture implementation.

**The differentiator claim this earns:** every peer interp tool is PyTorch-only. TorchLens via the adapter protocol becomes "eager-mode capture across any tensor framework." JAX adoption among interp researchers is real; "TorchLens for JAX" without a new library is a story no peer can tell.

**Combined sprint estimate (unification + backend-adapter refactor):** ~2-3 weeks. The two pieces are tightly coupled — when refactoring to unify trace+fastlog, you're already touching the layers that need to become backend-pluggable. Worth doing as one cohesive architectural reset post-launch.

## Performance north star

JMT 2026-05-14: **"Any op you want, as fast as you need — even faster than raw forward when you halt early."**

Two beats:
1. **Substrate breadth claim**: any op (not just module outputs), barely slower than raw forward for steady-state capture. Earned by lazy fastlog ctx + smart wrapper paths.
2. **Early-halt claim**: `tl.fastlog.halt()` lets you abort the forward pass after grabbing what you wanted. For deep models with target activation in early layers, TL can be **fractional-x raw forward** — strictly faster than running the whole model. No peer ships this.

This is the acceptance criterion for the perf sprints below. It's also the docs comparison soundbite that flips the apparent perf weakness ("TL captures more, looks slower") into the actual selling point ("TL captures more, can be faster when you opt to halt").

Every perf-relevant PR should answer: "does this move us closer to that north star?"

Current state (perf benchmark 2026-05-14, commit `3409c25`):
- fastlog_zero / fastlog_module-only (steady-state, no halt): target 2-5x raw; currently ~15-30x on GPU transformers
- Trace capture full-save (steady-state): ~100-300x raw (acceptable; user is opting into data work)
- rerun (new inputs): currently ~same as full capture; target ~5-10x raw
- **fastlog + halt at early layer: NOT YET BENCHMARKED.** Expected fractional-x raw on deep models. This is the strongest single perf story TL has.

### Comprehensive functorch / `torch.func` transform coverage — vmap, grad, jac, functional_call (filed 2026-06-02) **[CAPTURE COVERAGE GAP]**

**The gap.** TorchLens captures by intercepting eager `torch.*` calls via its wrappers. Operations that run *inside* a functorch / `torch.func` transform — `torch.vmap`, `torch.func.grad` / `jacfwd` / `jacrev` / `vjp` / `jvp`, `functional_call` (and `torch.compile`/fx, separately) — execute under a **different dispatch path** (the functorch interpreter / `BatchedTensor` stack) that **bypasses the eager wrappers entirely**. So every op inside such a region is **invisible** to TorchLens, and the tensors they produce surface **untraced**. This is a whole class of operations, not one op.

**How it surfaced (2026-06-02, Mistral/Audio-VITS).** Modern HF `transformers` builds the causal/sliding-window attention mask with `torch.vmap` over a mask function (`masking_utils.create_causal_mask`). Tracing those models produced an untraced `(1,1,16,16)` mask tensor that entered the decoder layers unlabeled and got a functionless synthetic `interventionreplacement` placeholder. The metadata invariant **correctly caught it** — but the first attempt wrongly **exempted** the placeholder (band-aid; see CLAUDE.md "Validation Integrity"). The shipped narrow fix captures/source-tags the vmap-region output + arms a regression test so the tripwire stays live. **This TODO is the comprehensive version of that narrow fix.**

**What a real solution needs.**
- **Detection:** know when capture is inside, or consuming the output of, a functorch transform region — hook the functorch interpreter/dispatch, or wrap the `torch.vmap` / `torch.func.*` / `functional_call` entry points, or (floor) reliably tag the *output* tensors of such regions as first-class nodes with provenance ("produced inside `torch.vmap(fn)`"), never silent placeholders.
- **Semantics — PENCILED (2026-06-02, leaning; to EXPLORE later, NOT a final decision): single opaque op looks like the right model.** Treat the whole transformed function as one op with typed input(s)/output that stores the transform callable (`lambda *xs: torch.vmap(fn)(*xs)`, etc.) as its replayable `func`, so validation re-runs it on the saved inputs and passes **legitimately** — no exemption (this is the band-aid lesson: validation must stay honest). Do NOT trace *inside* by default — the functorch `__torch_dispatch__` layer is fragile to hook AND the per-element shapes inside vmap don't match the real batched tensors, so an inside-trace produces a misleading graph. "Expand inside" is an **opt-in** mode only, never the default. Single-op also preserves the input→output edge and matches how TL already treats fused kernels + the user's mental model of a transform as one vectorized op. **Tradeoff:** the stored callable is opaque → portable `.tlspec` save is **audit-only** for these ops (same as the existing opaque-callable anti-pattern); in-session replay/validation is unaffected.
- **Note on the shipped narrow stopgap (2026-06-02):** the pre-launch fix (`fix(capture): log untraceable vmap-built tensors as internal sources`, model_prep.py) tags the vmap output as a labeled *source/leaf* so it's no longer a functionless placeholder + the tripwire is armed — but a source has no inputs, so it **drops the input→output edge**. The penciled single-op direction above is the current lean for the comprehensive fix (to explore, not locked).
- **Uniformity:** apply to `vmap`, `grad`/`jacfwd`/`jacrev`, `vjp`/`jvp`, `functional_call` — they share the identical blind spot. (Coordinate with the existing `torch.compile`/TorchScript/`torch.export` anti-pattern — same "alternate dispatch" family.)
- **Tests:** a matrix with a tiny model exercising each transform; assert (a) no functionless placeholder ops appear in plain capture, and (b) `validate_forward_pass` passes legitimately.
- **Docs/anti-patterns:** add a functorch-transform note alongside the existing "do not expect fused kernels to expose hidden internal tensors"; remove it once real coverage lands.

**Design notes — what a single-node vmap op can/can't capture (penciled 2026-06-02; vmap deserves its own careful treatment):**
- **Info you GET for free** (intercepting at the `torch.vmap(fn)(*inputs)` boundary): the real input tensor(s) — shape/dtype/memory + their graph producers (so `parents` are real); the real **output** tensor + its activation (extractable / interventable / validatable) + its consumers (real `children`, so the input→output edge is preserved); the callable as a **replayable `func`** (+ `fn.__name__`/qualname/source if introspectable); the `torch.vmap(...)` **call-site** (`code_context`); timing / RNG / autocast / module-containment as usual; and **vmap config** — `in_dims`/`out_dims`, the **batch size** (vmapped dim length), `randomness` mode.
- **What you LOSE:** the per-element internal ops (`full`/`triu`/comparisons/`masked_fill`/…) and their intermediate activations — identical to how TL already treats a fused SDPA kernel (one node, typed I/O, no internal exposure).
- **Internals are RECONSTRUCTABLE on demand (NOT captured live):** because the node saves `fn` + the inputs + `in_dims`, you can slice one element `x[i]` and call `fn(x[i])` directly (NO vmap) → normal eager dispatch → TL traces the full per-element subgraph. Faithful because `vmap(fn)` IS `fn` applied per element, so the un-vmapped run does exactly the same ops. **Caveats:** (a) it's a SECOND execution — exact only if `fn` is pure/deterministic (true for masks + most vmapped fns; randomness/side-effects need RNG restore / `randomness=` handling); (b) it's PER-ELEMENT (pick a slice / loop a few), NOT all batched intermediates at once (that needs true inside-tracing); (c) inspection-oriented, a reconstruction not a replay of history.
- **Intervention angle:** by default patch the node's **OUTPUT** (e.g. the mask). "Patch INSIDE" is harder + OPEN — either patch within the reconstructed single-slice run (inspection-grade, not live) or intervene in the live functorch execution (hard). Decide later.
- **Opt-in "expand inside" mode:** surface the single-slice reconstruction as a user-facing expansion, clearly labeled "representative single-element view" (not the batched reality).

**Open questions for the careful treatment:** node identity/labeling for transform ops; how `out_dims` / multi-output transforms map to TL outputs; nested transforms (vmap-of-grad) → nested nodes vs flattened; `functional_call` (params-as-inputs) semantics; portable-save (the callable is audit-only); whether to auto-expand small batches; and how all of this rides on the single-substrate / backend-adapter refactor.

**Why it matters / urgency.** `torch.func`/vmap adoption is rising fast in mainstream libraries — transformers' mask construction is now vmap-based **by default** — so this gap will silently hit **more and more** models over time, precisely in the spot where validation must stay armed. Connects to the **single-substrate architectural endpoint** (above): alternate-dispatch coverage is a capture-breadth axis the unified substrate should own.

**Sizing.** Pre-2.0-launch: the narrow mask fix + armed tripwire (shipped) is sufficient. Comprehensive transform coverage is its own **post-launch sprint** (est. multi-day; touches `decoration/` + the dispatch layer; needs the semantics decision first). Best folded into the capture-path-unification / backend-adapter refactor since that already reworks the dispatch layer.

### Follow-up benchmark: `fastlog_halt_early`

Add early-halt rows to the perf suite once the in-flight no-save addendum lands. For each model, benchmark fastlog capture + halt at three depths: ~25% / ~50% / ~75% through the model. Expected: 25%-depth halt should run in ~1/4 of raw forward time, demonstrating sub-baseline TL performance.

Models where this matters most: deep ones (ResNet-18, GPT-2, GPT-2 HookedTransformer). LSTM less interesting (recurrent, all layers fire per timestep).

Headline number for docs: "TL fastlog + halt at layer N runs in X ms vs Y ms for raw forward — N/M of the time." That's a flag-planting comparison row peer tools cannot match.

Filed 2026-05-14 from JMT spotting the halt-implies-sub-baseline-perf insight during benchmark review.

## Active Tasks

### Capture + surface input preprocessing pipeline (filed 2026-05-28)

[PARTIAL -- core shipped (`tl.trace` has live `transform=`/`save_raw_input`/`output_transform` kwargs; `Trace.raw_input`/`raw_output` instance fields exist + are KEEP in portable spec; `tl.autoroute.input`/`output` registries + HF `trace_text`/`trace_image`/`trace_multimodal` bridge tracers shipped; `_render_raw_input` + image=/imagescale in rendering.py). STILL OPEN: a structured `trace.input_preprocessing` record (tokenizer/processor descriptor for portable serialization) -- only the small raw-input thumbnail/repr is saved today, not a serialized transform descriptor.]

**Status:** filed during the `tl.trace(hf_model, "text")` auto-routing sprint. When the input to `tl.trace` isn't already a tensor (e.g. raw text, PIL image, audio waveform), there's a preprocessing step that converts it to the actual tensor the model consumes. Today that preprocessing is opaque to the trace — once tokenization runs, only the resulting token-ID tensor is visible. Users lose the link back to the source text / image / audio.

**Idea:** explicitly capture the preprocessing step and surface it on the Trace:

- `trace.raw_input` — the original non-tensor input (already exists on Trace as `save_raw_input` opt-in; verify and confirm).
- `trace.input_preprocessing` — a record of the transform applied. Could be:
  - The tokenizer / processor object (runtime-only, with the qualname for portable .tlspec).
  - A serialized description of the transform (e.g., for tokenizers: vocab path, model max length, padding strategy).
  - Optionally the intermediate forms (raw text → tokens → tensors → model input).
- `trace.draw()` could optionally show a "preprocessing" node at the input boundary, separate from the actual graph nodes.
- For multi-input cases (vision + text), capture each branch's preprocessing distinctly.

**Why:** debugging, reproducibility, and self-documenting traces. If you save a trace and reopen it months later, you should be able to see "this came from the string 'Hello world!' tokenized with distilbert-base-uncased tokenizer" not just the final token ID tensor.

**Scope:** medium. Touches `Trace` schema, the auto-routing in `tl.trace`, the visualization to render the preprocessing boundary, and `.tlspec` save/load for portable serialization of preprocessing descriptors. Probably 1-2 days.

**When:** post-2.0, after the auto-routing feature has been used for a while and we know what users actually want exposed.

### Add `show_param_memory` viz kwarg (node-label-format sprint follow-up, 2026-05-28)

**Status:** filed during the node-label-format cleanup sprint. JMT and Claude agreed param-byte annotation makes the params line denser by default but is genuinely useful in the Embedding-dominates-memory case (e.g. weight (30522, 768) → 94 MB while the OUTPUT is only 39 KB — three orders of magnitude visible only with the annotation).

**Decision:** skip in the default node label. Add `Trace.draw(show_param_memory=True)` as an opt-in viz kwarg.

**When `show_param_memory=True`, the param entry becomes:**
```
params: weight (3072, 768) 9.0 MB · bias (3072,) 12.0 KB
```
(Single space between shape tuple and memory string. Same `tl.Bytes`/`format_memory` formatting as the tensor-info line.)

**Default (False) keeps the current locked form:**
```
params: weight (3072, 768) · bias (3072,)
```

**Scope:** ~30 LOC in the param-list formatter to thread the flag through; ~5 LOC plumbing in `Trace.draw()` signature. Add to the standard set of viz kwargs (probably alongside `vis_buffers`, `vis_direction`, etc.). Update the docstring on `Trace.draw()`. One small viz-diff golden if any tests cover the WITH variant. Maybe half a day.

**Where it shines:** Embedding layers (massive vocab × dim), big linears in LM heads, expert FFN blocks in MoE models. Anywhere the param footprint dwarfs the activation footprint.

### Revisit facets framework follow-ups (filed 2026-05-27 after sprint landed)

**Status:** facets framework shipped on `main` as `3ada85d` + `9406863` (LOCKED entry in `glossary_walkthrough_deltas.md`, glossary v8 in vault, HF notebook with examples). Tier-2 GREEN. End-to-end verified against DistilBERT and GPT-2 in the notebook. But several rough edges surfaced that JMT should weigh in on before 2.0 launch.

**Open items, in priority order:**

1. **`keys()` vs accessibility mismatch.** Recipes declare facet names via `facets=(...)` at registration time so `view.keys()` is cheap (doesn't invoke the recipe). But the recipe at compute time may legitimately produce only a subset of the declared keys (e.g., when `param.value` is None, `add_if_present` skips the entry). Result: `view.keys()` lists `gamma` but `view.gamma` can raise `AttributeError`. Two possible fixes: (a) make `keys()` reflect actually-computed contents (requires invoking the recipe on first `keys()` call -- loses the cheap-list property), or (b) have access fall back to `None` instead of raising. JMT to decide which contract is right. Currently the notebook works around this with `.get('gamma')` and `.has('gamma')` checks -- workable but not elegant.

2. **`Module.cls` stores the class, not the instance.** This tripped Codex during the sprint (LayerNorm recipe used `cls.weight` -- always None) and tripped Claude during the notebook example (user recipe used `mod.modules[...]` -- doesn't exist on Module). Recipes have to know to read scalar config from `module.custom_attributes` and submodule outputs from the trace via `_source_trace`. Worth either: (a) renaming `Module.cls` to something more honest like `Module.module_class` to surface the "class not instance" distinction, or (b) adding sugar like `Module.config_attr(name)` and `Module.child(name)` that wraps the common access patterns. Either would prevent future recipes from hitting the same trap.

3. **LayerNorm `gamma` / `beta` need eagerly-loaded `param.value`.** In default capture mode `param.value` may be None, so the norm recipe degrades to omitting those facets. Either: (a) add a capture flag that keeps param tensor refs in-memory through trace lifecycle, (b) have `param.value` lazy-resolve from the live module at access time (with a clear error when the model is no longer reachable), or (c) accept the degradation and document it. Same concern applies to any future recipe that wants raw weight tensors (MLP weights, attention projection weights, etc.).

4. **Recipe-self-recursion.** A user recipe that accesses `mod.facets.X` from inside itself recurses infinitely (the FacetView re-iterates ALL matching recipes including itself). The notebook example sidesteps this by computing from `mod.calls[0].out` directly. Either: (a) detect recursion and raise an informative error, (b) document the limitation in the spec and notebook clearly, or (c) restructure FacetView to support inter-recipe dependencies (more invasive).

5. **Glossary should call out the `del record.facets` vs `record.facets.invalidate()` distinction more clearly.** `invalidate()` clears the value cache but keeps the matched-recipe set. `del record.facets` (or `del record._facets_cache`) drops the entire FacetView and re-matches on next access -- needed to pick up newly-registered recipes mid-session. Notebook now uses `del attn0.facets` for the user-recipe example; spec text should make this distinction explicit.

6. **Notebook gotcha worth documenting:** users registering recipes for module classes that already have a built-in recipe (e.g., adding extra facets to `DistilBertSdpaAttention`) will see `recipe_source` become a tuple. That's correct multi-recipe-merge behavior, but worth showing in the docs.

7. **[DESIGN, raised 2026-06-02] Integrated treatment of slices / facets / outs / multi-output.** JMT takeaway: these are currently three+ disjoint concepts that overlap at the edges and should be presented as ONE coherent addressing model rather than bolted-on siblings. Verified state as of 2026-06-02:
   - `facets` (v8, SHIPPED) already covers **semantic, recipe-driven** named views, and the built-in attention recipes ALREADY slice fused QKV internally (`recipes/attention.py:134` GPT-2: `c_attn_out.split(...)` -> `q`/`k`/`v` reshaped into heads; GQA handled). So `module.facets.q/k/v` is the 80% mech-interp read case and it's DONE.
   - `multi_output_*` / `container_path` (SHIPPED) covers ops returning a **container of separate tensors** (split, max->values+indices, RNN out+hidden). Structural.
   - The proposed `op.outs` (2.1, NOT BUILT) was partly justified by "named QKV read access" -- **that justification is now redundant with facets.** What `op.outs` genuinely adds beyond facets: (a) recipe-free **raw/positional tensor-slice addressing** (`op.outs[:, :, 768:1536]`) that works with no registered recipe, and (b) **slice-level INTERVENTION write-back** -- patch a sub-region of a fused tensor and have it flow forward. Facets are a read-only derived view and structurally cannot do (b).
   - **Action:** before building 2.1, design a UNIFIED slice/facet/outs/multi-output addressing surface (one mental model: "how do I name and access a sub-piece of what an op/module produced, for both read AND intervention"), rather than shipping `op.outs` as a fourth parallel concept. Re-scope 2.1 down to the genuinely net-new parts (raw-slice addressing + slice intervention).

**Recommendation:** items 1 and 2 are worth resolving before 2.0 launch since they affect contract semantics. Items 3-6 are documentation / polish that can ride a later cleanup. Estimated scope: 1 day for 1+2 (small surface, clear options), half day for 3-6 (mostly docs + a few small ergonomic additions).

### Harmonize scoped accessors to 0-based integer indexing + short/long label form support

Locked 2026-05-21 during rename pass. Currently `Layer.ops[N]`, `Module.calls[N]`, `GradFn.calls[N]` use 1-based pass/call-index keying. Should be 0-based positional (Python list-like). The 1-based semantic indices stay on the record fields (`op.pass_index`, etc.) and in label formats.

Also: label-based lookup should accept BOTH short (`conv2d_2:1`) and long (`conv2d_2_3:1`) Layer-label forms, resolving to the same Op when unambiguous.

**Files to update:**
- `torchlens/data_classes/_accessor_base.py` — base Accessor `__getitem__` to use 0-based positional integer indexing (currently uses direct dict lookup; need to convert int → sorted-by-key positional)
- `torchlens/data_classes/layer_log.py:71-79` (`OpAccessor` scoped) — change integer handling to 0-based positional; update docstring
- `torchlens/data_classes/module_log.py:69` (`ModuleCallAccessor` scoped) — same fix
- `torchlens/data_classes/grad_fn_log.py:46` (`GradFnCallAccessor` scoped) — same fix
- `torchlens/data_classes/_lookup_keys.py` — add short-form label resolution alongside long-form

**Tests required:**
- `layer.ops[0]` returns first pass (was `layer.ops[1]`)
- `layer.ops[-1]` returns last pass
- `len(layer.ops)` matches `layer.num_passes`
- Iteration order matches pass order
- Both `layer.ops["conv2d_2:1"]` and `layer.ops["conv2d_2_3:1"]` return same Op
- Same checks on `Module.calls`, `GradFn.calls`

**Migration note:** Existing user code doing `layer.ops[1]` for first pass breaks. Pre-launch hard-break is acceptable; will surface in docs/changelog.

Filed 2026-05-21. Implementation lands with the rename sprint.

### Unify buffer layer-vs-op treatment with other boundary types

Currently buffer-side has asymmetric coverage compared to other boundary types (input, output, internal source/sink):

| Boundary type | Layer accessor | Op accessor | Layer count | Op count |
|---|---|---|---|---|
| Input | ✓ | ✓ | ✓ | ✓ |
| Output | ✓ | ✓ | ✓ | ✓ |
| Internal source | ✓ | ✓ | ✓ | ✓ |
| Internal sink | ✓ | ✓ | ✓ | ✓ |
| **Buffer** | ✓ (`buffer_layers`) | ✗ (deferred) | ✓ (`num_buffer_layers`) | ✗ (deferred) |

The Op-side accessors for buffers are deferred to Buffer Option B (the persistent-entity refactor). But there's a deeper conceptual issue: currently `Buffer(Op)` IS an Op subclass that mixes layer and op semantics — there's no clean separation between "this is a buffer position in the equivalence-class graph" (layer-level) and "this is one buffer-read/buffer-overwrite event" (op-level).

The Buffer Option B proposal at `.project-context/buffer_refactor_proposal.md` addresses this by:
- Promoting `Buffer` to a persistent entity (sibling of Module/Param)
- Buffer-source and buffer-sink as Op SUBTYPES (via flags, not subclass)
- Then `buffer_source_ops`, `buffer_sink_ops` accessors fall out naturally
- And `Trace.num_buffer_source_ops`, `Trace.num_buffer_sink_ops` paralleling internal_source/sink

This todo is the specific cross-reference: when Buffer Option B work happens, make sure the layer-vs-op axis lands cleanly with parity to the other boundary types' surfaces. Don't ship Buffer Option B without that parity check.

Filed 2026-05-21 during count parity completion lock. Cross-reference: see `buffer_refactor_proposal.md` and the existing "Buffer revamp" todo entry for the full proposal.

### Principled exposure of runtime PyTorch handles across TL records

Current state: only `Op.grad_fn_handle` exposes a runtime PyTorch instance publicly. Module / Param / Buffer have internal weakrefs (`_source_module_ref`, `_param_ref`, etc.) but no public handle. Justified by access asymmetry — users have the runtime model object for most cases; only grad_fns aren't easily reachable.

**Decide post-launch:** should we expose runtime handles uniformly?

| Hypothetical | What |
|---|---|
| `Module.handle` | Live `nn.Module` subclass instance (weakref) |
| `Param.handle` | Live `nn.Parameter` instance |
| `Buffer.handle` | Live buffer tensor |

**Naming convention is already established:** `_handle` suffix = exposed runtime instance. Adding would slot in cleanly.

**Use case for adding:** portable-loaded captures (no user model object), API-uniform debugging surfaces, certain low-level introspection workflows.

**Cost:** weakref management complexity per class; API surface bloat for what's currently a rare use case.

**Subtlety to address: `Param.grad` semantic.** Currently `Op.grad` is the captured SNAPSHOT (immutable post-capture); `Param.grad` is the LIVE `nn.Parameter.grad` tensor (mutable; changes with each backward call). Same field name, different semantics. The naming doesn't telegraph the live-vs-snapshot distinction. Options:
- Rename `Param.grad` → `Param.live_grad` for explicit live indicator
- Stay with `Param.grad` and rely on docstring/docs to explain
- Adopt `_live` suffix as a convention for runtime-mutable queries (parallel to `_handle` for runtime instance refs)
- Possibly extend to other locations where TL exposes live runtime state vs snapshots

The `_live` convention (if adopted) would parallel `_handle` — `_live` for runtime-mutable tensor queries, `_handle` for runtime instance refs, bare names for TL snapshots / records. Worth designing during the same pass that decides whether to expose runtime handles broadly.

Filed 2026-05-21 during glossary v3 rename pass. Post-launch design decision; not blocking rename sprint.

### Auto-suppress grad_fn creation when backward_ready=False (part of backward-pass sprint)

JMT's "charm": only spawn grad_fns if wanted. Currently `backward_ready=False` means TL doesn't HOOK grad_fns, but PyTorch may still CREATE them (based on input `requires_grad` state). Wasteful for inference-only capture.

**Proposed:** when `backward_ready=False` AND user hasn't requested any backward-related capture (`save_gradients=False`, no intervention spec with grad hooks, etc.), wrap the user's forward in `torch.no_grad()` internally. PyTorch then skips grad_fn creation entirely. Saves:
- The C++ autograd-machinery cost during forward
- Autograd memory (no `saved_tensors` from forward ops)
- Cleanup overhead

**Implementation:** small — capture machinery already wraps `model.forward(x)`; adding a conditional `torch.no_grad()` context is a one-liner change.

**Flag options to consider:**
- Implicit: when `backward_ready=False`, auto-suppress. Opinionated default.
- Explicit: `suppress_autograd=True` kwarg. User opts in.
- Implicit with override: auto-suppress when `backward_ready=False`; `suppress_autograd=False` lets user keep grad_fns for some other purpose (e.g., custom hooks unrelated to TL's backward path).

I'd lean implicit-with-override — match user intent in the common inference case, escape hatch available.

**Validation matrix to consider:** what other capture-config flags interact with this? `intervention_ready`, `save_gradients`, etc. — should they auto-disable suppression?

Filed 2026-05-21 during glossary v3 rename pass. Part of unified backward-pass sprint. Small implementation but design discussion needed on the flag interaction matrix.

### Higher-order gradient support (part of backward-pass sprint)

TorchLens currently has PARTIAL support for higher-order gradients (Hessian-style, MAML, adversarial via backward-of-backward, etc.). The data structures can accommodate them; the capture machinery does NOT auto-instrument them.

**What works today:**
- Standard backward (first-order grads) — fully captured
- Gradient accumulation across multiple `loss.backward()` calls — fully captured (multiple GradFnCalls per GradFn)
- `loss.backward(create_graph=True)` itself runs fine — PyTorch handles it, TL captures the first-order firings

**What doesn't work:**
- Autograd nodes CREATED by the first backward (the gradient-computation graph when `create_graph=True`) are NOT hooked by TL
- Second-order backward firings on those new nodes go uncaptured
- `torch.autograd.grad()` calls that create new autograd nodes are similarly not instrumented
- Hessian / Jacobian computations, MAML, second-order optimization — partially captured

**Root cause:**
TL registers hooks on grad_fns AT FORWARD CAPTURE TIME (when each Op is recorded). For higher-order, the first backward creates NEW grad_fns that TL never sees at forward-capture. Second backward fires those new hooks without TL knowing.

**What would need to change:**
1. After first backward (when `create_graph=True`), traverse the resulting gradient chain to find the newly-created autograd nodes
2. Register hooks on those new grad_fns
3. Capture their firings as GradFnCall records when the second backward triggers them

Non-trivial — needs post-first-backward graph walking + hook installation. Plus design questions:
- How do second-order GradFns relate to first-order GradFns in the record graph? (Are they linked? Separate cluster? Children of the first-order grad_fns?)
- How do they label? (`addmm_back_back_1_5`? Or `addmm_back^2_1_5`? Some convention for "backward of backward.")
- Do they get their own `step_index` axis, or share with first-order?
- Do users typically want full Hessian-level capture, or just sampled second-order? (Storage cost matters.)

**Data structures already cover most of it.** GradFn class is general; could hold higher-order nodes. GradFnCall is general; could hold higher-order firings. The label / record-relationship design is the open question.

**Priority:** post-2.0; advanced users; not blocking the rename sprint. Folds into the unified backward-pass sprint (alongside backward-pass deep-dive, BackwardPass first-class records, recurrent-loop semantics).

Filed 2026-05-21 during glossary v3 rename pass.

### Backward call-site code context (part of backward-pass sprint)

For tracking "from where in user code was `loss.backward()` triggered?" — capture Python call stack at backward invocation as part of the per-event entry in `state_history` (one entry per `loss.backward()` call). Should NOT be a per-GradFnCall field (backward hooks fire from PyTorch's autograd engine; the stack at hook-fire time is uniformly uninformative).

Proposed: `state_history[N].backward_call_context` (Python call stack at the moment `trace.backward(loss)` was invoked, as `FuncCallLocation`). Only present for backward-engine entries.

Filed 2026-05-21 during glossary v3 rename pass. Folds into the unified backward-pass-untangling sprint (below). Not blocking the rename sprint.

### Backward labeling + recurrent-loop semantics (part of the backward-pass unified sprint)

**Filed 2026-05-21.** Surfaced during glossary v3 rename pass. Genuinely thorny; needs design work; coordinate with the broader backward-pass deep-dive (below).

**Three intertwined concerns:**

**1. Recurrent forward loops produce distinct grad_fn Python objects — TorchLens currently appears to MERGE them.**

PyTorch creates a new grad_fn instance for each forward pass through a recurrent layer. A 3-pass recurrent `conv2d_1_5` produces 3 distinct grad_fn Python objects. The current label-construction code uses `layer.trace_index` (per-Layer, not per-Op) for `total_num`, suggesting TorchLens MERGES recurrent grad_fns into one GradFn per Layer equivalence class.

JMT's preferred direction: **treat each grad_fn object as its own GradFn record** (3 separate records for 3 recurrent forward passes). Because they ARE genuinely distinct Python objects in PyTorch; merging obscures the underlying autograd structure.

Consequences if we go this direction:
- GradFn labels need to be unique per forward pass. Could use Op-label style with `:N` for forward pass index: `addmm_back_1_5:1`, `:2`, `:3` — one per pass
- GradFnCall labels then get `:M` appended for backward firings: `addmm_back_1_5:1:1` (1st pass's grad_fn, 1st firing) — verbose but unambiguous
- OR keep label structure but disambiguate differently (e.g., embed pass index in the type/step suffix)

Hard because: confuses users who expected GradFn ↔ Layer to be a simpler 1-to-1 in recurrent cases. Need clear docs.

**2. Two type_index construction schemes inside one class.**

Currently `_grad_fn_label_parts` (`backward.py:100-135`):
- `has_op=True` → mirror forward Layer's `type_index` and `step_index` (visual correspondence with forward labels)
- `has_op=False` → separate per-type counter + discovery order

The visual correspondence is convenient but creates two code paths and state-dependent index semantics. Cleaner alternative: always per-type counter + discovery order; forward correspondence retrievable via `grad_fn.op.label` (one-hop resolver).

Discussion needed on whether visual correspondence is worth the complexity.

**3. Function name introspection robustness for grad_fn classes.**

`grad_fn.__class__.__name__` is the source. Generally robust:
- ✓ PyTorch built-ins (always have class names; normalize for version-suffix stability)
- ✓ Custom autograd.Function subclasses
- ✗ Lambda Functions (rare; `__name__ = "<lambda>"`)
- Edge: decorator-wrapped Functions (mostly OK; `__name__` from wrapper class)

Worth verifying normalization layer handles edge cases gracefully (doesn't crash on `<lambda>`).

**Recommendation: address all three in one unified backward-pass-untangling sprint.**

Coordinate with:
- "Backward pass handling deep-dive — does loss.backward() spawn GradFn logs?" (below)
- "Promote backward passes to first-class `BackwardPass` records" (post-2.0 follow-on)

These three areas interact heavily. The backward subsystem deserves a dedicated sprint, not piecemeal fixes.

### Backward pass handling deep-dive — does loss.backward() spawn GradFn logs?

Open architectural question: when a user calls `trace.backward(loss)`:
- Are GradFn records created at that point (from the live autograd graph)?
- Or were they already created during capture (forward time) and `backward()` just populates them?
- How are multiple backward calls handled (re-entrant `backward(loss)` after first)?
- What's the relationship between `GradFn` records and the live `tensor.grad_fn` chain?

Related questions for the same review:
- Where do GradFnCall records get instantiated? At hook-firing time? Pre-allocated?
- How do `has_op` (paired forward Op) and orphan grad_fns (no forward op) interact with backward triggering?
- Multi-backward case: should each backward call get its own root + per-call event records? (Earlier filed: `BackwardPass` first-class records — this is the design needed.)
- Backward pass lifecycle on the `Trace.state` enum — what transitions happen?

Filed 2026-05-21. Needs design review of capture/intervention/backward.py logic. Post-launch; not blocking rename sprint.

### Full treatment of container classes (multi-output return shapes)

ContainerSpec, container_path, multi_output_type, multi_output_name, TupleIndex, DictKey, NamedField, DataclassField, HFKey — the multi-output container surface is rich and currently spread across multiple records.

Items to address in a focused review:
- Are the type-classes (TupleIndex, DictKey, NamedField, etc.) public API? Naming consistency.
- Is `ContainerSpec` documented as a public type? Should it have a stable schema?
- How do nested containers (`tuple` of `dict` of `list`) get represented? Worked examples needed.
- Are HF-specific keys (`HFKey`) appropriately scoped, or generalize to "object-with-dict-style-access"?
- How do containers interact with intervention API — can you swap a tensor inside a tuple output?
- Saved arg templates for module forward args — do they use the same container machinery?

Filed 2026-05-21. Post-launch design pass; medium scope. The multi-output naming surface (`out` / `outs` / `multi_output_*` / `container_path`) was touched in this rename sprint; the next pass is the structural review.

### Review intervention records handling

The intervention surface in TorchLens (selectors, sites, hooks, helpers, intervention_spec, fork/replay/rerun) was largely deferred from the rename sprint per the existing deltas ("Intervention method names are intentionally not promoted as final here; the audit deferred them to the integrated `Site` concept review").

Items to address:
- `Op.interventions` — what does this store? Per-Op intervention records?
- `Trace.intervention_spec` — is this a stable public type?
- `Site` concept review — site labels, site resolution, site dispatch
- `attach_hooks`, `do`, `set` method naming
- Intervention storage save/load semantics
- Bundle-level intervention coordination
- Whether `Op.interventions` should be label-list or resolved-records (per the locked filter-list convention)

Filed 2026-05-21. Post-launch design pass. The intervention API is a big surface and deserves dedicated review separate from the names-only sprint.

### Audit gradient memory accounting — Op-side vs GradFnCall-side overlap

`Trace.total_gradient_memory` currently counts ONLY Op-level gradients (`Layer.grad_memory` summed across layers — see `backends/torch/tensor_tracking.py:107-108`). GradFnCall-side gradient storage (`grad_inputs`, `grad_outputs` on each GradFnCall record) is NOT included in this total — invisible to the Trace-level "how much memory does gradient capture cost?" answer.

**Three potential issues:**

1. **Hidden memory.** GradFnCall storage can be substantial (per-call snapshots; many calls; both inputs and outputs). Not visible in `total_gradient_memory`.

2. **Redundant storage when has_op=True.** For grad_fns with a corresponding forward Op, `GradFnCall.grad_outputs` is typically the same tensor data as `Op.grad`. Saving both is redundant. (Differences: multi-backward case where `Op.grad` accumulates while GradFnCall snapshots per-call; has_op=False case where GradFnCall is the only record.)

3. **Mismatch between predicate and reality.** `has_saved_gradient` on Op covers Op-side; `GradFnCall.is_saved` covers GradFnCall-side. No unified predicate for "is there any gradient saved for this position."

**Three fix options:**

- **A. Documentation only:** clarify `total_gradient_memory` as Op-level only; add `total_grad_fn_call_memory` field for the other side. Visible but not deduplicated.
- **B. Unified `total_backward_memory`:** sum Op-level + GradFnCall-level (with sub-totals). Most accurate "cost of backward" answer.
- **C. Audit + dedupe at capture time:** add `save_grad_fn_call_grads` flag (default off when has_op=True for single-backward; on when needed). Avoids redundant storage. Then totals account cleanly without double-counting.

**Recommendation:** Option C is architecturally cleanest (eliminate the redundancy at source); Option A is the minimum (make the accounting visible). Probably do BOTH — A first for clarity, then C for actual storage reduction.

Filed 2026-05-21 during glossary v3 rename pass. Audit + design call needed before rename sprint touches this area.

### Post-2.0 TorchLens-2.0 follow-on items (filed 2026-05-21)

These came up during the rename-sprint walkthrough; all post-rename-sprint and post-2.0-launch. Listed together to keep the launch surface bounded; revisit individually after rollout + docs sprint lands.

**a. Buffer revamp (Option B). [SHIPPED 2026-06-05 — local main commits 91e1645..6e3a291 + fix 430357a]**
Originally filed at `.project-context/buffer_refactor_proposal.md`. DONE: `Buffer` is now a
first-class persistent entity (Module/Param-sibling); graph version nodes are plain `Op` +
`is_buffer` flag (subclass retired). Write capture lands all three kinds (reassignment via
scoped class `__setattr__`, in-place via storage snapshot, fused/native via post-op value
snapshot), each a validatable version node; `validate_forward_pass` green across the stress
battery + real models. Gradient flow through reassignment verified (hooks observational).
Docs: `docs/buffers.md` + glossary lockstep. Loop-detection crash on the RNN-cell reassignment
pattern (dangling per-op `equivalent_ops` after buffer merge) root-caused + fixed generally
(`_scrub_per_op_equivalence_lists`). Build spec: `.research/buffer-sprint/PLAN_v5_BUILD.md`.
Residual edges (documented, by design, NOT bugs): `.data = tensor` reassignment unsupported
(reconciliation diagnostic raises); a dead intermediate fused write that is never read then
overwritten is not displayable (computationally inert); non-registered Python-attr state is
out of scope. Follow-on (still open): the layer-vs-op parity check below (item "Unify buffer
layer-vs-op treatment") — op-side buffer accessors (`buffer_source_ops`/`buffer_sink_ops`).

**b. Attribute-bloat cleanup — consider more sub-config classes.**
Decided NOT to add Op/Layer sub-config objects pre-launch (only ~5 fields per scope; below the bloat threshold). Trace.capture_config namespace migration is already deferred in the v3 deltas. Post-launch: re-audit ALL classes for "is this attribute count a problem?" — Op has ~70 fields; some natural clusters (function-call cluster, output-tensor cluster, output-gradient cluster, etc.) might earn sub-config dataclasses. Decide per-cluster only when a clear sub-object naming + boundary is evident.

**c. Convenience fields for Op/Layer INPUTS (currently output-focused).**
[DONE verified 2026-06-01 -- the full locked cluster `op.input_ops`/`input_activations`/`input_shapes`/`input_dtypes`/`input_memory`/`num_inputs` is present on a real Op instance (verified via `tl.trace` of a toy model).]
**STATUS 2026-05-23: NAMES LOCKED (see deltas log `Op input-side @property cluster`). Implementation pending.** Final cluster: `op.input_ops`, `op.input_activations`, `op.input_shapes`, `op.input_dtypes`, `op.input_memory`, `op.num_inputs`. Graph-parents only; Param/Buffer flows deferred to follow-on. Notes below preserved for archaeology.

Op fields default to OUTPUT (locked convention). But users often want input-side info too — input shapes, input dtypes, input memory aggregate. Currently accessible via `op.parents` (label list) + per-parent Op lookup. Convenience fields like `Op.input_shapes`, `Op.input_dtypes`, `Op.input_memory_total` (`@property` derivations) would be ergonomic for inspection. Discussion topic: which inputs do users actually want, what's the right level of derivation, do these conflict with the OUTPUT convention. Defer until use cases mature.

**d. First-class BackwardPass records.**
Already filed earlier in this todos file as "Promote backward passes to first-class BackwardPass records." Keep filed; this is the dedicated-classes story for backward passes.

**e. Dedicated While-loop / data-dependent iteration handling.**
TorchLens currently doesn't have a first-class concept for data-dependent loops (while loops with run-time-determined iteration counts). Different from recurrence-detection (which handles repeated SAME-INPUT-SHAPED iterations). Need to scope: what does dedicated handling look like? A `Loop` record class? An iteration-counter field? Defer for design exploration.

**f. Fastlog refactor / unification with Trace.**
Already filed as "Unified fastlog + main-trace save-selection treatment" earlier in this todos file. Post-launch: do the actual unification work — fastlog vs trace API parity story, predicate API consolidation, `fastlog` naming decision, internal structural sharing where possible.

**g. Revisit `cleanup` treatment.**
`Trace.cleanup()` exists ("Clear circular references and runtime-only heavyweight objects"). Naming and surface unclear post-rename — does it parallel `release_param_ref()` / `release_buffer_ref()`? Should it be `release_runtime_refs()`? Are there per-class cleanup methods? Revisit holistically.

**i. Variable-name introspection for saved tensors.**
For a saved Op output, capture the variable name used in the source code (e.g., a tensor saved as `t = relu(x)` would have `var_name = "t"`). Already in deltas as "Variable name introspection (`Op.var_names`): deferred (possible future feature)." Re-evaluate viability post-launch — implementation likely uses AST inspection of the forward() source; non-trivial; possibly not viable for all callsites.

**j. Dynamic-branch organization (conditionals tree).**
Beyond the current `Conditional` / `ConditionalArm` machinery — a higher-level "dynamic branches tree" of all data-dependent paths in a model. Would let users ask "what conditionals fired across this Trace?" / "what's the branching topology?" Could be a `Trace.branch_tree` accessor returning a structured tree. Defer for design.

**k. Op-as-multiple-args-of-child (e.g., `y = x + x`).**
When an Op's output is passed as MULTIPLE args to a child Op (e.g., `add(x, x)`), the data model probably needs to handle this cleanly — current parents list might dedupe, losing the "passed twice" info. Visualization should mark this (double edge? multi-arrow notation?). Audit current behavior, decide on the correct data + visual representation.

**l. Visualization for multi-version outputs (when `has_output_variations` is True).**
When an Op's output was modified in-place between children, the children see different tensor versions. Currently captured in data (`out_versions_by_child`) but unclear if visualization marks this. Should be a visual cue (color, annotation, branching) so users notice. Defer to vis sprint.

**m. `num_modules` on Trace — total submodule count for the model.**
[DONE verified 2026-06-01 -- `Trace.num_modules` present on a real Trace instance.]
**STATUS 2026-05-23: NAMED LOCKED as `Trace.num_modules` (bare).** Implementation pending. See deltas log `Trace.num_modules`.

**n. Module descendant call count + call-depth-from-beneath.**
[DONE verified 2026-06-01 -- `Module.num_descendant_calls`/`max_descendant_depth` AND `ModuleCall.num_descendant_calls`/`max_descendant_depth` all present (hasattr on the classes).]
**STATUS 2026-05-23: NAMES LOCKED.** Final: `Module.num_descendant_calls`, `Module.max_descendant_depth` (and same on ModuleCall, per-call). Folded into the call-tree accessor lock — see deltas log `Module / ModuleCall call-tree accessor + display`.

**o. Op vs Module parity for `args_summary` / `kwargs_summary` / `args_template` / `kwargs_template`.**
[DONE verified 2026-06-01 -- `ModuleCall.forward_args_summary`/`forward_kwargs_summary` (instance attrs, set in __init__ + FieldPolicy.KEEP in module.py) and `forward_args_template`/`forward_kwargs_template` (class attrs) all present; `Op.args_summary`/`kwargs_summary` present too. Full quartet shipped.]
**STATUS 2026-05-23: NAMES LOCKED.** ModuleCall gets the full quartet: `forward_args_summary` + `forward_kwargs_summary` + `forward_args_template` + `forward_kwargs_template`. See deltas log `Module / ModuleCall args/kwargs template parity`.

**p. Unified handles/references sprint — Param handles + all other runtime PyTorch object handles.**
Several Trace records currently store `_object_id` Python ids for ground-truth identity but lack the actual handle. Spans: Param, Buffer, Module, the underlying torch tensor on the source side, GradFn objects (the `torch.autograd.Function` `grad_fn` instance), hook handles, optimizer references. Today this is sprinkled ad hoc. Sprint should define the principled exposure: which records expose `.handle` (or `.torch_obj`) and how lifetime/staleness is handled. Already partially filed as "Principled exposure of runtime PyTorch handles" earlier in this file; this addendum specifically adds Param handles to that sprint's scope.

**q. Cross-class consistency sprint — labels vs object references everywhere.**
The Reference Form Convention (4 principles, locked 2026-05-21) should be enforced in one comprehensive pass post-rename. Sweep ALL inter-record fields across ALL classes (Trace, Op, Layer, Module, ModuleCall, Param, Buffer, GradFn, GradFnCall, Bundle, Super[T], ConditionalRecord, ConditionalArm) and verify each follows the convention: (1) portability dictates storage form, (2) tensors/callables/handles as objects, (3) frequent-ref bare-label + `_label` + resolver property, (4) bare label lists for plurals (with the paired-plural-resolver exception for genuinely-different entities). Output: an audit table marking each field as compliant or non-compliant. Likely a few stragglers — the rename sprint catches the obvious ones, but a dedicated sweep catches the long tail. Should run after rename sprint settles.

### Supergraph naming / structure cleanup (post-launch)

`bundle.supergraph` is the cross-trace alignment engine — internal-ish. `Supergraph` and `SupergraphNode` classes live in `torchlens/intervention/_topology/` (underscore-prefixed = internal), are NOT in `torchlens.__init__`, NOT imported outside `_topology/`. The field `bundle.supergraph` is public but advanced/diagnostic use only.

Can defer to post-launch. The high-traffic Bundle surface (Super[T] family, comparison helpers) is what matters for the rollout/docs sprint.

Items to revisit later:
- Should `Supergraph` get a `__getitem__` for label lookup of nodes?
- Should `SupergraphNode.layer_refs` follow Principle 3 with `_label` + @property resolver?
- Should `Bundle.supergraph` be `@property` (lazy build) or stored field?
- Should `Supergraph` ever be public? (Probably stay internal; users go through Bundle methods.)
- Naming sanity check on `SupergraphNode.fingerprint` (tuple of `module_path`, `func_name`) — clear enough?
- Alignment with the v3 naming conventions (drop `_log` etc.) — `SupergraphNode.layer_refs` value type is currently `LayerLog`; sync to renamed `Layer` type post-rename-sprint

Filed 2026-05-21 during glossary v3 rename pass. Park until after rename sprint + 2.0 launch.

### Promote backward passes to first-class `BackwardPass` records (parallels ModuleCall / GradFnCall)

Filed 2026-05-21 as Level 3 follow-up to the `backward_root_grad_fn_id` → `backward_root_grad_fn_ids` rename (locked the lightweight plural fix in v3; this is the heavier architectural improvement).

**Trigger to revisit:** when users start wanting richer per-backward-pass metadata (loss value, duration, status, grad outputs summary, etc.). If they just want the root grad-fn id, the current plural-list approach suffices.

**Proposed surface:**

New `BackwardPass` record class (sibling of ModuleCall / GradFnCall):
- `BackwardPass.root_grad_fn_id`: runtime `id()` of the root grad-fn
- `BackwardPass.loss_value`: the loss tensor passed (optional capture, gated by flag)
- `BackwardPass.timestamp`: when the backward was invoked
- `BackwardPass.duration_s`: how long the backward took
- `BackwardPass.call_index`: 1-based invocation index (parallels ModuleCall.call_index)
- `BackwardPass.ordinal_index`: 0-based position in `trace.backward_passes`
- `BackwardPass.engine`: backward engine identifier
- `BackwardPass.status`: success / failure / partial
- `BackwardPass.grad_fn_calls`: ordered list of GradFnCall labels fired during this backward

On Trace:
- `Trace.backward_passes`: Accessor for BackwardPass records (parallels `module_calls`, `grad_fn_calls`)
- `Trace.num_backward_passes`: already exists
- `Trace.last_backward_pass` (`@property`): convenience for the most recent

On rename / migration:
- `Trace.backward_root_grad_fn_ids` (Level 1 lock) becomes derivable: `[bp.root_grad_fn_id for bp in trace.backward_passes]`
- `Trace.last_backward_root_grad_fn_id` becomes `trace.last_backward_pass.root_grad_fn_id`
- Could remove the list fields once BackwardPass lands, or keep as deprecation shims

**Cost:** new record class, accessor, ordinal indexing cascade. Moderate sprint scope. Coordinate with whatever backward-API maturation work is in flight at that point.

Not blocking the rename sprint; post-2.0 architectural improvement.

### Rename `replace_state_from` / `append_state_from` — current names confusing

JMT flagged 2026-05-21: `Trace.replace_state_from(new_log)` and `Trace.append_state_from(new_log)` read awkwardly. "State from" parses ambiguously (English: "replace state, originating from X"; method: "replace MY state using X as source"). The word "state" is also overloaded — could mean `Trace.state` (the enum) or "the Trace's overall captured state."

Current entries:
- `Trace.replace_state_from(new_log)`: Atomically replace this Trace's run state from a freshly-built Trace.
- `Trace.append_state_from(new_log)`: Merge compatible chunk outs from `new_log` into this Trace.

Candidate cleaner names (pick during rename-sprint revisit):
- `replace_from(other)`, `append_from(other)` — drop "state"; simpler
- `swap_state(other)`, `extend_state(other)` — direct verbs
- `commit_from(other, mode="replace"|"append")` — single method, mode arg; "commit" implies atomicity
- `adopt_from(other)`, `extend_with(other)` — adoption metaphor

Also worth resolving: are these methods user-facing or internal-only? If internal, prefix with `_` and they vanish from the public API surface. If user-facing, the name needs to be obvious.

Filed 2026-05-21 during glossary v3 rename pass. Low priority; current names work, just confusing.

### Unified fastlog + main-trace save-selection treatment (consolidate predicate API)

Capture-config has three save-selection fields (`layers_to_save`, `gradients_to_save`, `module_filter`) plus a parallel predicate-first API in `tl.fastlog.record()` (`keep_op`, `keep_module`, `default_op`, `default_module`). Both surfaces select what to save, with different APIs and different semantics around "selector list vs predicate callable."

Open items to resolve together:
1. **`module_filter` → `save_predicate` rename** (already deferred in v3 deltas; verify and lock during this treatment)
2. **Unify `layers_to_save` accepted forms.** Today probably accepts `"all"`, list of labels, selectors, callable. Document the full surface and decide if all forms stay or if some are deprecated.
3. **Decide fastlog↔trace API parity story.** Should main `tl.trace()` accept `keep_op` / `keep_module` predicates directly (currently only fastlog)? Or should fastlog adopt `layers_to_save` style (currently only trace)? Pick one canonical form, or document the case-by-case rationale.
4. **`fastlog` naming.** Deltas already note this rename is deferred. Pick during the treatment.

Filed 2026-05-21 during glossary v3 rename pass. Park until post-rename. The v3 surface for capture-config save-selection fields stays as-is in the meantime.

### Recursive params accessor on Module (PyTorch parity gap)

[DONE verified 2026-06-01 -- `Module.recursive_params`, `Module.num_recursive_params`, `Module.recursive_param_addresses` all present (hasattr on Module). The locked companion-count family shipped.]
**STATUS 2026-05-23: NAMES LOCKED (see deltas log `Recursive params accessor on Module`). Implementation pending.**

Currently `Module.params` returns only directly-owned Params (this Module's address only). PyTorch's `nn.Module.parameters()` defaults to RECURSIVE (this module + all address-based sub-modules). TorchLens lacks the recursive equivalent.

**Proposed addition:**
```python
module.params              # existing — directly-owned Params (this Module's address only)
module.recursive_params    # NEW — Params recursively through this Module + all address-based sub-Modules
```

Plus companion count fields for parity:
- `module.num_recursive_params`
- `module.num_recursive_params_trainable`
- `module.num_recursive_params_frozen`
- `module.num_recursive_param_tensors`
- `module.num_recursive_param_tensors_trainable`
- `module.num_recursive_param_tensors_frozen`
- `module.recursive_param_addresses` (label list)

**Scope semantic clarification:**
- "Recursive" means **address-based** (static `nn.Module` tree) — matching PyTorch's `parameters(recurse=True)` semantic
- Different from "dynamic call-tree recursive" (params used by ModuleCalls beneath this one in the dynamic call tree) — defer unless demand surfaces

**Why useful:**
- PyTorch parity — common workflow is "give me all params under this module"
- Avoids the verbose `[p for child in module.address_descendants for p in trace.modules[child].params]` comprehension
- Filtering/inspection workflows ("which params are trainable under this transformer block?")

**Naming alternatives weighed:**
- `recursive_params` — explicit; matches PyTorch's `recurse` kwarg vocabulary
- `descendant_params` — clear; "descendant" matches the address-tree framing
- `all_params` — concise but ambiguous (all WHICH params?)
- `params_subtree` — implies subtree but verbose

Lean `recursive_params` for PyTorch idiom alignment.

Filed 2026-05-21 during glossary v3 rename pass. Post-documentation-sprint addition; not blocking the rename or 2.0 launch. Small implementation work (address-tree walk + recursive collection).

### Call-tree fetching/display on Trace/Module/ModuleCall

[PARTIAL -- the DISPLAY method `show_call_tree(max_depth, include_atomic, show_call_index, file)` is shipped on Trace, Module, and ModuleCall (verified present on instances; matches glossary v9 lines 694/1206/1325). STILL OPEN: the structured DATA accessor `call_tree` / `CallTreeNode` (the `node.call` + `node.children` object form) -- `call_tree` is absent on Trace, Module, and ModuleCall, and is NOT in glossary v9. Only the ASCII display landed.]
**STATUS 2026-05-23: NAMES LOCKED (see deltas log `Module / ModuleCall call-tree accessor + display`). API + companion fields fully locked. Display method body pending implementation.**

Currently the call tree is accessible only one-hop at a time (`ModuleCall.call_parent`, `call_children`). Reconstructing the full subtree requires manual recursion. Should be a first-class accessor + display method.

**Proposed programmatic accessor (structured data):**
```python
trace.call_tree              # full call tree from root → leaves; nested CallTreeNode object
module.call_tree             # subtree rooted at this Module (across all its calls; aggregate view)
module_call.call_tree        # subtree rooted at this specific call (per-invocation view)
```

Data structure options:
- A. Recursive `CallTreeNode` dataclass: `node.call` (ModuleCall) + `node.children` (list of CallTreeNode)
- B. Flat `dict[call_label, list[child_labels]]` — simpler but loses tree-walking convenience
- C. Networkx-style DiGraph — heavyweight; integrates with viz tools

Lean (A) — clean Python iteration; minimal external dependencies.

**Proposed display method:**
```python
trace.show_call_tree()       # prints indented tree to stdout (or file via kwarg)
module.show_call_tree()      # subtree rendered
module_call.show_call_tree() # subtree rooted at this call

# Output style:
# encoder.block.0:1
# ├── encoder.block.0.attention:1
# │   ├── encoder.block.0.attention.qkv_proj:1
# │   ├── encoder.block.0.attention.softmax:1
# │   └── encoder.block.0.attention.out_proj:1
# ├── encoder.block.0.layer_norm:1
# └── encoder.block.0.feedforward:1
```

Tree-drawing characters via standard tree-rendering (matches `tree` command idiom).

**Configuration:**
- `max_depth` kwarg — limit display depth
- `include_atomic` kwarg — show/hide atomic-module leaves
- `show_call_index` — toggle `:N` suffix display
- Output-to-file via `file` kwarg (paralleling `print()`)

**Use cases:**
- Debugging: "what got called inside this module during forward?"
- Inspection: quick model-structure overview from REPL/notebook
- Documentation: generate call-tree text for issue reports

**Why valuable:**
- Tree-structured data should have tree-walking + tree-display methods (standard library design)
- Bridge between `trace.draw()` (heavy visual graph) and per-call inspection
- One-line REPL convenience for `print(trace.show_call_tree())`-style debugging

**Naming alternatives weighed:**
- `call_tree` (data) + `show_call_tree()` (display) — proposed
- `dynamic_call_tree` (data) — disambiguates from static address tree; verbose
- `module_call_tree` — same; verbose
- `print_call_tree()` — alternative to `show_call_tree()`; both work

Lean `call_tree` + `show_call_tree()` — bare names; receiver class disambiguates from address tree (`Module.address_children` is the static-tree analog).

Filed 2026-05-21 during glossary v3 rename pass. Post-documentation-sprint addition; not blocking. Medium-small implementation work (recursive tree-walking + tree-rendering display).

### Module-scope memory aggregate naming — three quantities need disambiguating field family

[DONE verified 2026-06-01 -- the Option-3 principled refactor shipped: `ModuleCall.output_activation_memory`/`internal_activation_memory`/`output_gradient_memory`/`internal_gradient_memory` (single-call quadrants) and `Module.total_output_activation_memory`/`total_internal_activation_memory` (cross-call) all present on instances; the legacy bare `activation_memory` is GONE from both ModuleCall and Module. The `output_`/`internal_` prefix family + `total_` cross-call convention landed.]
**STATUS 2026-05-23: NAMES LOCKED (see deltas log `Module / ModuleCall internal-memory cluster + output_/internal_ prefixes`). Implementation pending.** Final convention: `output_` and `internal_` prefixes at ModuleCall (single-call quadrants) and Module (cross-call `total_` versions only — drill into ModuleCall for per-call values). Notes below preserved for archaeology.

Module/ModuleCall have THREE distinct legitimate "memory" quantities that current v3 naming doesn't cleanly disambiguate:

| Quantity | Meaning | Example value |
|---|---|---|
| A. Output memory | Bytes of tensor(s) returned by `forward()` | 4MB for one transformer block output |
| B. Internal aggregate | Sum of memory across all internal ops in the call | 60MB for the same block (all intermediates) |
| C. Cross-call aggregate | Sum of A or B across multiple calls of the same Module | varies |

**Current v3 state:**
- `Module.activation_memory` (and `ModuleCall.activation_memory`) means Quantity A (output) — lives in Output Passthroughs section
- No Quantity B field at Module scope (users compute manually via comprehension)
- Naming a hypothetical `Module.total_activation_memory` is ambiguous — could plausibly mean B or C

**Three naming proposals (deferred decision):**

- **Option 1:** Rename output → `out_memory`; use `total_activation_memory` for internal aggregate. Breaks Layer parallel where `Layer.activation_memory` IS output.
- **Option 2 (minimum disruption):** Keep `activation_memory` as output; add `total_internal_activation_memory` for B. Verbose name; "total_internal" reads clunky.
- **Option 3 (principled refactor):** TWO prefix families — `out_*` for output cluster (`out_memory`, `out_gradient_memory`, etc.), `internal_*` for cost cluster (`internal_activation_memory`, `internal_gradient_memory`, etc.). Plus `total_X` for cross-call. Renames `activation_memory` → `out_memory` etc. throughout.

**My lean: Option 3 long-term (cleanest semantic split), Option 2 minimal-disruption near-term.**

This affects related fields too:
- `gradient_memory` — same three-quantity problem
- `autograd_memory` — only Quantity B makes sense (autograd is internal); but field naming convention needs to be consistent
- `transformed_activation_memory` — also has output-vs-internal axis

Use case for Quantity B (internal): profiling — "how much memory did this transformer block use total?" Currently requires comprehension; should be one-shot field.

Filed 2026-05-21 during glossary v3 rename pass. Post-launch refactor; significant cascade. Coordinate with `Trace.capture_config` namespace migration (also deferred — both touch the broader memory/capture surface).

### Comprehensive review of Trace.capture_config surface

The capture-config field cluster on Trace has grown to 20+ flags and accumulated tech debt. Needs a holistic review separate from individual field renames.

**Current capture-config-style fields on Trace (in v3):**

Save-related:
- `save_raw_activations`, `save_raw_gradients`, `save_gradients`, `save_arg_values`, `save_arg_templates`, `save_rng_states`, `save_code_context`

Selection-related (locked as "unified fastlog treatment" todo):
- `layers_to_save`, `gradients_to_save`, `module_filter` (rename to `save_predicate` pending)

Mode/state:
- `backward_ready`, `intervention_ready`, `intervention_spec`, `capture_mode`, `detach_saved_activations`, `recurrence_detection`

Behavioral flags:
- `emit_nvtx`, `mark_layer_depths`, `raise_on_nan`, `verbose`

Output handling:
- `output_device`

Transforms (callables):
- `activation_transform`, `gradient_transform`

**Items to address:**

1. **`Trace.capture_config` namespace migration** (already deferred in v3 deltas) — move all 20+ flags into a `Trace.capture_config` dataclass. `trace.capture_config.save_raw_activations` instead of `trace.save_raw_activations`. Cleans up Trace's top-level surface dramatically.

2. **Categorization within capture_config** — does the dataclass have sub-clusters (save, selection, mode, behavior)? Or flat?

3. **Naming consistency audit across all flags** — e.g., why `save_raw_activations` vs `save_arg_values` (different verb style)? Why `recurrence_detection` (bare noun) vs `save_gradients` (verb-noun)?

4. **`emit_nvtx` documentation tweak** — add explanatory note ("NVTX = NVIDIA Tools Extension; ranges show up in Nsight profilers"). Matches PyTorch's `torch.autograd.profiler.emit_nvtx` API; don't rename, but make docs self-contained.

5. **`capture_mode` enum values** — what are the possible values? Make sure they're documented and named consistently.

6. **`module_filter` → `save_predicate` rename** (deferred in v3 deltas; unify with fastlog predicate API per separate todo).

7. **Defaults audit** — for each flag, what's the default? Is the default the "least surprising" or "most useful"? Documented?

8. **save_arg_values vs save_arg_templates** — these are paired (heavy vs light). Could they be one field with a tri-state (`off | template | full`)?

9. **Per-Op overrides** — some Op fields are config carry-overs (output_device, activation_transform, gradient_transform). Are they consistently documented as such?

10. **Validation interactions** — capture_config flags interact with backward, intervention, save policies. Is the validation matrix documented anywhere?

Filed 2026-05-21 during glossary v3 rename pass. Post-launch design pass. Coordinate with the fastlog unification todo (predicate API consolidation) and the `Trace.capture_config` namespace migration (already deferred).

### Audit function-argument-name introspection (Op.arg_names + companions)

Op fields `arg_names`, `num_args_total`, `num_pos_args`, `num_kwargs`, plus the saved-args templates (`args_template`, `kwargs_template`) and the new `args_summary` / `kwargs_summary`, all rely on TorchLens being able to introspect the captured function's signature and map positional args to names.

**Open question: how bulletproof is this?**

Sources of fragility:
1. **C++-implemented torch functions** (most of `torch.*`) — `inspect.signature(func)` often returns empty/incomplete signatures for C++ ops. TorchLens probably has hardcoded fallback specs (the `FUNC_ARG_SPECS` table referenced in earlier bug fixes — `ARG-KWARGS-MISSING`). What's its coverage today? Which torch ops fall through?

2. **Custom autograd Functions** — `torch.autograd.Function` subclasses have user-defined `forward()` signatures. Should be introspectable but might have edge cases.

3. **Lambdas / partial functions** — `functools.partial`, lambdas, and similar callable wrappers may not expose argument names cleanly.

4. **Variadic args** — `*args` / `**kwargs` patterns; how does TorchLens record these in `arg_names` / `non_tensor_pos_args` etc.?

5. **Method-vs-function** — bound methods, `self` handling, instance methods on tensors vs free functions.

6. **`functorch` / `torch.func` transforms** — `vmap`, `grad`, etc. wrap functions; do they preserve introspection?

7. **MLX backend** — MLX functions have their own signature characteristics; does the introspection machinery handle MLX-side correctly?

**Items to audit:**
- Coverage of `FUNC_ARG_SPECS` static table — what's in, what's out, how exhaustive?
- Failure modes — when introspection fails, what does Op.arg_names contain? Empty list? None? Best-effort guesses?
- Are the failures user-visible? Silent? Logged?
- Cross-reference to known bug `ARG-KWARGS-MISSING` (per the Phase 14 cleanup) — is that fully resolved or are there lingering edge cases?
- Test coverage — are there tests for argument-name introspection across major torch op families?

**Knock-on effects of fragile introspection:**
- `args_summary` quality degrades when arg names are missing (falls back to positional indices?)
- `args_template` / `kwargs_template` storage depends on slot-mapping by arg name — failures here could break replay
- `module_entry_arg_keys` similarly relies on arg names

**Filed 2026-05-21.** Stability/completeness audit. Not blocking the rename sprint (names stay the same regardless of introspection quality), but a real area to harden before the 2.0 launch and the documentation push.

### Call-site source-name capture (caller's variable names, per-frame on FuncCallLocation)

**Filed 2026-05-24 (riffed with Claude).** Distinct from the function-signature introspection above (#645) — that captures parameter names FROM the called function's signature (`relu(x)` → param `input`). This captures the CALLER'S VARIABLE name at the call site (`relu(x)` → source expression `x`). Complementary, not redundant.

**Mechanism: `executing` library**

Alex Hall's `executing` (also powers IceCream, snoop) finds the exact AST node being executed in a given frame by matching bytecode offsets to parsed source. Mature, ~99% reliable in normal Python.

```python
import executing, ast
node = executing.Source.executing(frame).node   # ast.Call node
if node is not None:
    arg_exprs = tuple(ast.unparse(a) for a in node.args)        # ("x",)
    kwarg_exprs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
    call_expr = ast.unparse(node)                                # "self.attn(hidden)"
```

**Wrapper-frame skip pattern**

Naive "caller's frame" is wrong because TorchLens wraps every torch function:
```
[your script]   model.layer1(x)            <-- frame we want
[nn.Module]     def __call__(...)          <-- torch internal, skip
[torchlens]     def wrapped_func(...)      <-- our wrapper, skip
[torchlens]     out = original_func(...)   <-- where we are
```

Walk frames past `torch.*` and `torchlens.*` (or anything in `site-packages` for stricter heuristic) until hitting user code. Standard `code_context` capture already does this; same skip-list applies.

**Per-frame capture: walk every user frame, not just innermost**

Since `code_context` is already a LIST of FuncCallLocation (the full user-code stack), capture at each frame. Across multiple frames you get the call chain — same op might be known as `inputs` at the script level, `x` at the pipeline wrapper, `hidden` at the forward method. Caveat: the SAME variable name across frames doesn't guarantee the SAME tensor (intermediate ops can rename it). So this is "names of immediate args at each call site that led to this op," not strictly a tensor lineage. Still useful.

**FuncCallLocation field additions**

Package directly into `FuncCallLocation` since it's already the per-frame record. New fields:

```python
@dataclass
class FuncCallLocation:
    file: str
    line: int
    function_name: str
    source_line: str | None              # (existing)
    # NEW:
    call_expr_source: str | None         # full unparsed call expression at this line, e.g. "self.attn(hidden)"
    arg_source_names: tuple[str | None, ...] | None    # positional arg expressions, e.g. ("hidden",)
    kwarg_source_names: dict[str, str] | None          # kwarg expressions, e.g. {"dim": "0"}
```

Asymmetry note: FuncCallLocation is ALSO used for definition locations (`Module.forward_source_location`, `HookInfo.source_location`) where these fields don't apply. Set to None for those uses. Same pattern as existing optional-on-some-uses fields (e.g. `Op.module_call_stack`).

**Op-level convenience fields (the "outermost / innermost" sugar JMT asked about)**

Walking the full code_context is the exhaustive surface. For the common case ("just give me the user's variable name"), add convenience properties on Op:

```python
op.outermost_call_args   # @property -> arg_source_names of code_context[0]  (script-level call site)
op.innermost_call_args   # @property -> arg_source_names of code_context[-1] (closest to the actual op)
op.outermost_call_expr   # @property -> call_expr_source of code_context[0]
op.innermost_call_expr   # @property -> call_expr_source of code_context[-1]
```

"Outermost" = highest frame in user code stack (script entry point). "Innermost" = deepest user code frame (immediate caller of the torch op). Avoiding "top"/"bottom" since Python traceback convention reverses them.

These all return None when capture is disabled or when the source isn't available.

Same pattern can apply to ModuleCall (uses the same code_context list at module-entry time).

**Opt-in capture flag**

```python
trace = tl.log_forward_pass(model, x, capture_arg_names=True)
```

Off by default to avoid the per-op overhead. When False, FuncCallLocation's three new fields are None across every frame. When True, populated. Same pattern as `save_gradients` / `save_buffer_history`.

**Performance estimate**

- Per-frame overhead: ~µs after first lookup per source file (executing caches per-file AST).
- For ResNet-50 forward (~150 ops × ~5 user frames each): ~750 lookups → ~1-3% wall time.
- For LLM forward (thousands of ops): more like 5-10%. Hence opt-in.
- Zero overhead when flag is False (early return before frame walk).

**Failure modes (all graceful, all -> None)**

- REPL / exec'd source — no source file to parse.
- Bytecode optimized away (rare with `executing`).
- Jupyter cells — `linecache` handles these.
- C-extension callers — not an issue for torch (always Python-side dispatch).
- Comprehensions / generator expressions — captures the loop variable name (technically correct, less informative).

Worst case: field is None. No crashes.

**Use cases this unlocks**

- **Lookup by source name**: `trace.find_ops_where(arg_name="attention_mask")` becomes possible — walk every op's code_context for matching arg_source_names.
- **Source-code-aligned viz**: node labels can include the leaf variable name (`relu(hidden)`), tooltips can show full lineage across frames.
- **Better repr**: `Op(linear[2:1], called as 'self.q_proj(hidden)')` reads more naturally than the current bare label.
- **Lineage-style queries**: `op.outermost_call_args` answers "what's the highest-level identity of this tensor in the user's code?"
- **Debugging**: stack traces showing "this NaN appeared in an op consuming `attention_mask`" beats "consuming arg[0]".

**Sprint scope**

- `executing` library dep (transitive dep of common Python tooling; small)
- 3 fields on FuncCallLocation
- 4 convenience properties on Op (and parallel on ModuleCall if wanted)
- 1 capture flag on `log_forward_pass`
- Frame-walk + wrapper-skip in the existing code_context capture path (~30 lines)
- Tests for nested forwards, missing source fallback, REPL fallback, comprehension capture
- Glossary entry under FuncCallLocation section + Op section

Estimate: 1-2 days. Small enough to ship pre-2.0 as standalone polish OR park as a post-2.0 nice-to-have. Cost-benefit favors the latter unless someone wants the magic-moment "wait it knows what I called my variable?" experience for the 2.0 launch demo.

**Open questions to resolve at lock-time:**
1. Field names — `arg_source_names` vs `arg_expressions` vs `call_args_source`?
2. Single-frame capture (innermost only) as a lighter alternative — half the overhead, 80% of the value. Could be the v1 surface with full per-frame as a follow-up.
3. Should the `executing` dep be required or optional (fall back gracefully if unavailable)?
4. Name of the capture flag — `capture_arg_names`, `capture_source_names`, `capture_call_sites`?

### Revisit emit_nvtx documentation + naming

Minor follow-up: `Trace.emit_nvtx` is currently a bare flag. The naming matches `torch.autograd.profiler.emit_nvtx` (PyTorch idiom) — recommended to keep. But the docstring needs expansion so users who don't know NVTX can read the v3 glossary entry and understand:

> "NVTX = NVIDIA Tools Extension — a CUDA profiling annotation library. When True, TorchLens wraps captured ops with NVTX range markers so they show up labeled in NVIDIA profilers (Nsight Systems, Nsight Compute). Default OFF; enable when profiling on CUDA. Adds small per-op overhead."

Filed 2026-05-21 during glossary v3 rename pass. Tiny tweak; can land in same pass as the capture_config review or earlier.

### Revisit `ordinal_index` as universal trace-level index field name

Currently `ordinal_index` is the universal field name for "0-based position in the trace-level accessor" — appears on Op, Layer, Module, ModuleCall, Param, Buffer, GradFn, GradFnCall, Conditional (etc.). JMT noted some lingering doubt about whether this is the best name; flagging for post-launch revisit.

**Concerns / alternative names to weigh:**
- `ordinal_index` (current) — "ordinal" is mathematically loaded (often "ordinal numbers" in math/logic), might not telegraph "position in trace-level accessor" clearly
- `trace_position` — explicit about scope (trace-level) and intent (position)
- `trace_index` — was the name on GradFn before we renamed to `step_index` for symmetry with Op; using "trace_index" universally would conflict with GradFn.step_index semantic
- `global_index` — explicit about global vs scoped, less mathy
- `position` (bare) — bare scope-derived; relies on context, might be ambiguous
- `index_in_trace` — verbose but unambiguous

**Use cases the field serves:**
- Round-trip lookup: `trace.ops[op.ordinal_index] is op`
- Iteration in canonical order
- Per-record stable position for cross-references

**What's load-bearing:**
- 0-based (Pythonic indexing)
- Uniform across all per-record classes (single name family)
- "Position in the trace-level accessor" semantic

**Post-launch decision:** Audit which alternative reads most naturally for users. The current "ordinal_index" is functional but JMT's hesitation suggests it could be clearer. Possibly rename to `trace_position` or similar; would cascade to ~8 classes.

Filed 2026-05-21 during rename sprint planning. Cosmetic rename only; behavioral semantic stays the same. Post-launch.

### Examine state / state_history / last_run holistically — lifecycle surface review

Three closely-related fields make up the Trace lifecycle surface, but their design is fragmented and may have redundancy / inconsistency. Audit them together to ensure they form a coherent system.

**The three fields:**

1. **`Trace.state`** — Enum-like `TraceState` value: `PRISTINE`, `SPEC_STALE`, `REPLAY_PROPAGATED`, `RERUN_PROPAGATED` (likely needs rename to `FORWARD_PROPAGATED` per the pencil notes), `LIVE_CAPTURED`, `DIRECT_WRITE_DIRTY`, `APPENDED`. Single snapshot of the Trace's current lifecycle position.

2. **`Trace.state_history`** — Append-only log of lifecycle state changes (fork, replay, rerun, direct-write). Each entry: `(timestamp, transition_type, payload)`.

3. **`Trace.last_run`** — Dict summary of the most recent replay/rerun/append operation (engine, timing, intervention spec revision, flags, divergence info, hash before/after). Renamed from `last_run_ctx`.

**Open questions:**

- **Redundancy between `last_run` and `state_history[-1]`:** are these the same data, or do they differ in fields? If equivalent, `last_run` could become a `@property` returning `state_history[-1]`. If they differ, both stay but the difference needs documenting. Original earlier-filed todo "Verify `Trace.last_run` is redundant with `Trace.state_history[-1]`; possibly collapse" is now subsumed by this broader audit.

- **Consistency of state enum values:** `RERUN_PROPAGATED` likely needs renaming to `FORWARD_PROPAGATED` if the broader `rerun` → `forward` rename lands.

- **Coverage of state transitions:** does every operation that modifies a Trace push an entry to `state_history`? Are there silent modifications that bypass the log?

- **state_history schema:** is it a stable public schema (users can query it programmatically)? Or is it diagnostic-only?

- **Timing fields in state_history vs Trace-level aggregates:** state_history has per-event timestamps; `backward_durations` etc. (locked this session) are aggregates. Is the per-event data the source of truth, and aggregates are derived? Or both stored separately?

- **Naming of `last_run`:** is "run" the right word? An entry in state_history represents one "operation" or "transition" — not all are "runs" (e.g., direct-write isn't a run). Maybe `last_operation` / `last_transition` / `last_state_change`?

- **Public-vs-internal:** are `state` / `state_history` / `last_run` all part of the user-facing public API, or are some diagnostic-only?

- **Cleanup behavior:** does `Trace.cleanup()` clear state_history? Should it?

- **Multi-backward case:** with `backward_durations` (list, recently locked) and `BackwardPass` records (future todo), how does the backward-pass-lifecycle interact with state/state_history?

**Recommendation when revisiting:** treat state / state_history / last_run as ONE design problem. Don't fix one in isolation — the relationships are load-bearing.

Filed 2026-05-21 during rename sprint planning. Replaces the earlier narrower "Verify last_run redundancy" todo. Post-launch design pass.

### Universal accessor behavior audit — ALL accessors across the package

This consolidates the harmonized accessor rules. Audit every Accessor class and ensure ALL of them follow the uniform behavior locked during the rename sprint.

**Universal accessor rules (locked):**

1. **Type-strict.** Every accessor ALWAYS returns its own class type. No "for convenience" overlaps (e.g., `trace.layers[op_label]` returns Layer, not Op — strips `:N` to find parent).

2. **0-based positional integer indexing.** `accessor[N]` is Python-list-like. `accessor[-1]` works. `len(accessor)`, iteration in canonical order, etc.

3. **Short and long label forms accepted — FULLY INTERCHANGEABLE.** For Op-like labels, ALL FOUR forms must resolve to the same Op (when unambiguous):
   - `conv2d_2` — short, bare (no step_index, no pass_index) — resolves to Layer; or to Op for single-pass case
   - `conv2d_2_3` — long, bare (includes step_index) — resolves to Layer; or to Op for single-pass case
   - `conv2d_2:1` — short, pass-qualified — resolves to Op
   - `conv2d_2_3:1` — long, pass-qualified — resolves to Op

   This interchangeability is CRITICAL for users. The locked convention is that users should not have to remember which form to type — both short and long, both pass-qualified and bare, all work as inputs to ANY label-accepting accessor. The accessor implementation must:
   - Build alternate-form lookup keys for every Op/Layer (alternate short and long forms automatically registered)
   - Resolve any of the four forms to the same target object
   - Same applies to ModuleCall (`encoder.block` ↔ `encoder.block:1` for single-call)
   - Same applies to GradFn (`addmm_back_1_5` ↔ `addmm_back_1_5:1` for single-call)

   **Tests required:** every accessor verified to accept all four forms; same object returned regardless of form; alias-form lookup is part of CI.

4. **Bidirectional bare/qualified resolution:**
   - **Aggregate accessors** (`trace.layers`, `trace.modules`, `trace.grad_fns`): accept bare aggregate identifier OR per-event-qualified label (strips `:N` to find aggregate).
   - **Per-event accessors** (`trace.ops`, `trace.module_calls`, `trace.grad_fn_calls`): accept per-event-qualified label directly. For SINGLE-X aggregates, also accept bare aggregate identifier → unique per-event record. For multi-X, raise with guidance.

5. **Scoped accessors** (`layer.ops`, `module.calls`, `grad_fn.calls`): bare parent identifier accepted for single-X case (e.g., `layer.ops["conv2d_1_2"]` returns unique Op for single-pass Layer).

6. **Super[X] member accessors** (`super_layer.members`): supports `[str]` (trace name) AND `[int]` (0-based positional in Bundle member order).

7. **Bundle accessors** (`bundle.ops`, `bundle.layers`, etc. returning Super[X]): same bidirectional rule as Trace-level — return Super[X] versions instead of single-class records.

8. **Bundle universal accessor** (`bundle.at[X]` AND `bundle.at(X)`): square OR round brackets accepted; universal Super[X] resolution order.

**Accessors to audit + verify:**

Trace-level:
- `trace.layers`, `trace.ops`, `trace.modules`, `trace.module_calls`, `trace.params`, `trace.buffers`, `trace.grad_fns`, `trace.grad_fn_calls`
- Filter Accessors: `compute_ops`, `compute_layers`, `saved_*` (8 fields), `orphans`, `input_layers`, `output_layers`, `buffer_layers`, `internal_source_*`, `internal_sink_*`, `layers_with_params`
- `trace.conditionals` (ConditionalAccessor)

Scoped:
- `Layer.ops`, `Module.calls`, `GradFn.calls`
- `Module.params`, `Layer.params`, `Op.params`

Bundle:
- `bundle.traces` (TraceAccessor)
- `bundle.ops`, `bundle.layers`, `bundle.modules`, `bundle.module_calls`, `bundle.params`, `bundle.buffers`, `bundle.grad_fns`, `bundle.grad_fn_calls`
- `bundle.at[X]` universal

Super[X]:
- `super_X.members` (on every Super[X] class)

Universal:
- `trace[X]` (universal accessor)
- `bundle[X]` (member lookup; or universal? — verify which behavior locked)

**Test surface required:**
For each accessor: int positional (including negative), str label (all accepted forms — bare, qualified, short, long), wrong-type returns clear KeyError with disambiguation suggestions, fuzzy-match suggestions via existing `_lookup_keys.py` machinery extended uniformly.

**Implementation files likely affected:**
- `torchlens/data_classes/_accessor_base.py` (base Accessor)
- `torchlens/data_classes/_lookup_keys.py` (fuzzy match)
- Per-class accessor implementations in `layer_log.py`, `module_log.py`, `grad_fn_log.py`, `param_log.py`, `buffer_log.py`
- `torchlens/intervention/_super/_accessor_base.py` (Super[X] accessor base)
- `torchlens/intervention/bundle.py` (Bundle accessor + at)
- `torchlens/data_classes/model_log.py` (Trace `__getitem__`)

Filed 2026-05-21 during rename sprint planning. Lands with the strict-type-accessor implementation work. Critical for users — once locked, must apply uniformly or surprise behavior emerges.

### Strict-type explicit accessors on Trace (`trace.layers` always Layer, `trace.ops` always Op)

Universal `trace[X]` accessor locked 2026-05-21 covers cross-type convenience lookup. As a corollary, explicit class accessors should be type-strict — no "Layer accessor returns Op for convenience" surprises. Specifically:

- `trace.layers[X]` ALWAYS returns Layer. If `X` is a pass-qualified Op label (`conv2d_1_2:1`), strip the `:N` and return the Layer at `conv2d_1_2`. (Equivalent to using the bare Layer label.)
- `trace.ops[X]` ALWAYS returns Op. If `X` is a bare Layer label (`conv2d_1_2`) and the Layer has ONE Op, return that Op (single-pass passthrough). If the Layer has multiple Ops and `X` is bare, raise `AmbiguousOpLookupError` with guidance ("Layer has N passes; use pass-qualified label like `conv2d_1_2:1`").

Apply same rule to `trace.modules`, `trace.module_calls`, `trace.params`, `trace.buffers`, `trace.grad_fns`, `trace.grad_fn_calls` — each returns its own class type only. Cross-class lookup is the universal accessor's job.

Rename-sprint locks the SPEC; this todo tracks the implementation:
1. Remove the "for convenience" fallback from `LayerAccessor.__getitem__`
2. Add single-pass passthrough to `OpAccessor.__getitem__` (bare Layer label → Op if single-pass)
3. Add ambiguous-lookup error class
4. Tests: type assertions on returns + ambiguity raise

Filed 2026-05-21 during glossary v3 rename pass. Lands with the universal-accessor implementation work.

### Kill `streaming_pass_logs` / `num_streamed_ops` / `streamed_trace()` (redundant with Bundle)

[DONE verified 2026-06-01 -- grep across torchlens/ finds ZERO non-pycache hits for `streaming_pass_logs` / `num_streamed_ops` / `num_streamed_passes` / `streamed_trace`; the entire streaming-pass surface is removed. (Separate "activation streaming during capture" surface intentionally KEPT per the note below.)]

Pre-launch removal. The streaming-pass surface is a pre-Bundle hack that survived: a utility iterates inputs, captures one Trace per input, stashes the list on the first Trace as `streaming_pass_logs: List[Trace]` with companion count `num_streamed_ops`. Bundle handles this use case cleanly (named members, Super* alignment, proper container management).

**Remove:**
- `Trace.streaming_pass_logs` field
- `Trace.num_streamed_ops` field (also referenced as `num_streamed_passes` in glossary — both names mismatched between code and docs)
- `streamed_trace()` utility (`torchlens/utils/__init__.py:645-669`)
- All field-policy entries referencing `streaming_pass_logs` (constants.py, model_log.py)

**Migration for users (hard break, pre-launch acceptable):**
```python
# Was:
result = streamed_trace(model, inputs_iter)
for tr in result.streaming_pass_logs:
    ...

# Becomes:
bundle = tl.bundle(*[tl.trace(model, x) for x in inputs_iter])
for tr in bundle.traces.values():
    ...
```

Optional thin helper to consider as part of removal: `tl.bundle_inputs(model, inputs_iter, **kwargs)` returns a Bundle directly. Three-line implementation; can add later if a user asks.

**KEEP separately: activation streaming during capture.** Different concept — saves activations to disk/callback as they're produced (for memory-constrained capture of big models). Surfaces: `bundle_path` streaming, `out_callback` sinks, `_is_streaming_append_active` check in rerun.py. **JMT flagged 2026-05-21 to REVIEW the activation-streaming surface separately later — important and stays.** Don't conflate with this removal.

Filed 2026-05-21 during glossary v3 rename pass. Add to rename-sprint "Removed" list in deltas when locked.

### Benchmark addendum: zero-retention TL variant (apples-to-apples wrapper overhead)

Current perf suite (commit `3409c25`) compares raw forward vs TL Trace, which conflates wrapper-dispatch overhead with tensor-save cost. The fair "TL tax" measurement is wrappers-on + zero retention. Add a new operation `fastlog_zero` to `benchmarks/perf_runner.py`:

```python
return lambda: tl.fastlog.record(
    model, x,
    keep_op=lambda ctx: False,
    keep_module=lambda ctx: False,
    default_op=False,
    default_module=False,
)
```

Wrappers fire, predicates return False, nothing stored. This is the pure-dispatch-tax floor. Run across the same matrix (TinyNet/ResNet-18/GPT-2-HF/GPT-2-HookedTransformer/SmallLSTM x CPU+CUDA), 50 samples + memory pass. ~30 min run.

**Expected docs payoff:** the wrapper tax should be much lower than the full-Trace number — maybe 1.5-5x over baseline rather than 100-300x. That flips the comparison story from "TL is 300x slower" to "TL adds ~3x wrapper overhead; every additional cost is data you asked to capture." Much more honest and flattering frame for power-user docs.

JMT filed 2026-05-14 after reviewing initial benchmark results. Not blocking; small follow-up to the perf sprint.

### Perf sprint: leaner `Trace.rerun` (purpose = capture on new inputs)

**Correction 2026-05-14:** `Trace.rerun(model, x)` is for re-capturing on BRAND-NEW inputs `x`, not for re-applying interventions on the same input. The fast-replay-on-cached-input path is `Trace.replay()`. Earlier sketch confused the two.

Current `rerun` calls `_capture_with_active_spec` at `torchlens/intervention/rerun.py:84` — a full fresh capture pipeline. Re-allocates Trace metadata fields, re-builds graph-shape structure, re-saves every layer's activation.

Since `x` is genuinely new, the forward math MUST execute. But several things shouldn't have to:

1. **Topology-hash short-circuit:** if the new run produces the same `graph_shape_hash` as the prior Trace, skip metadata-tree reconstruction. Just refresh activation tensors in place. (Hash already computed for divergence detection at `torchlens/intervention/rerun.py:80`.)
2. **Skip re-saving for layers user opted out of:** if the original Trace was captured with `layers_to_save=[...]` (subset), rerun should respect the same scope rather than saving everything.
3. **Reuse prepared-model state:** `_prepare_model_once` is already cached, but the per-call setup inside the capture pipeline could be lighter when the model has been seen before.
4. **Append-mode rerun is already supported** (`append=True`); standard rerun is what needs the lean path.

**Expected gain:** rerun ≈ baseline_forward + wrapper_overhead + (user-requested save cost). For a no-save scope, that's roughly the zero-retention fastlog floor. For a partial-save scope, intermediate. Should land 5-50x over baseline depending on save scope rather than the current ~200x.

**Estimate:** 1-2 days. Impl in `torchlens/intervention/rerun.py` + new helpers for topology-hash fast path. Tests for: same-topology fast path, divergent-topology falls back to full rebuild, append-mode unchanged, `layers_to_save` scope respected.

Filed 2026-05-14 from JMT review of perf benchmark results. Real perf opportunity surfaced by the new suite.

### Docs sprint addendum: `docs/performance.md` page for "I want fast activation pulling"

The 2026-05-14 perf benchmarks (commits `3409c25` + `30dc452`) surface concrete numbers that power users will want when deciding HOW to use TorchLens. The docs plan v36 currently puts perf claims in the README hero + comparison pages but has no dedicated "performance guide" page. Add one.

**Audience:** power users who have read the comparison table, seen TL is slower per-op than vanilla hooks, and want to know "how do I get the speed I need for my workflow."

**Target page:** `docs/performance.md` (or `docs/guides/performance.md`). ~300-500 lines. Linked from:
- README power-user routing section (alongside cookbook/tutorials links)
- Every `docs/comparisons/vs_*.md` page footer ("worried about perf? see docs/performance.md")
- Cookbook recipes that pull activations
- `docs/for-ai-agents.md` (the agent-coder doc) — agents need this decision tree fast

**Content outline:**

1. **Decision tree** ("which TL mode should I use?")
   - "I want one activation from one layer, fast as possible" -> `fastlog.record(...) + tl.fastlog.halt()`. Sub-1x raw forward on deep models.
   - "I want all module outputs, peer-tool equivalent" -> `fastlog.record(default_module=True, default_op=False)`. ~5x raw forward.
   - "I want every-op metadata, no tensors saved" -> `log_forward_pass(layers_to_save=[], vis_opt='none')`. Wrapper overhead floor.
   - "I want everything, full inspect+intervene+viz" -> `log_forward_pass(intervention_ready=True)`. Substrate mode; pay the data cost.
   - "I want to capture once, swap inputs N times" -> capture + `Trace.rerun(model, x_2)` per input. Same cost as fresh capture per call but writes into existing Trace.
   - "I want to apply an intervention without recomputing" -> `Trace.replay(...)`. ~20x raw forward (cached state).

2. **The benchmark table** (from `benchmarks/perf_results_2026-05-14.md`). Headline rows only, per-model summarized. Link to full report.

3. **The "halt-early" superpower section.** Highlighted callout: TL can be FASTER than raw forward when you only need an early activation. Worked example with code + measured timing.

4. **Knob reference:**
   - `intervention_ready` (cost vs functionality)
   - `vis_opt` (cost vs viz output)
   - `layers_to_save` (scope control)
   - `module_filter`
   - `fastlog` predicates
   - Future: `save_mode="reference"` once that ships from the perf sprint (link to changelog)

5. **When NOT to reach for TorchLens:**
   - "I only ever need the final logits" -> just run the model
   - "I work exclusively on HuggingFace transformer families and want named hooks" -> TransformerLens is more ergonomic
   - "I need attribution methods" -> captum
   - Honest concession-style framing per the comparison-page convention.

6. **Architectural note** explaining WHY TL has the overhead it has (wrapper-per-op as the substrate trade) so engineers understand the design, not just the numbers.

**Estimate:** ~3-4h to author once the benchmark numbers are committed. Should land at Phase 1.1 (alongside README) since power users hit it from day one, not Phase 4. Update plan Phase 1 task list to include this page.

Filed 2026-05-14 from JMT pointing out that power-user perf guidance needs a concentrated docs surface.

### Ergonomic: string input to `log_forward_pass` (auto-tokenize)

[DONE verified 2026-06-01 -- delivered via the input auto-routing system rather than the duck-typed dispatch sketched here: `tl.autoroute.input` is a priority registry and `tl.trace` dispatches non-tensor inputs (text/image/multimodal) to the HF bridge tracers (`trace_text`/`trace_image`/`trace_multimodal`). String-input auto-tokenize is shipped (architecture differs from this entry's sketch but the user-facing ergonomic is delivered).]

Allow `tl.log_forward_pass(model, "Hello world")` to auto-tokenize when the model exposes a tokenizer. Matches TransformerLens UX; lowers cognitive friction for the dominant interp workflow; doesn't leak HF into core.

**Duck-typed dispatch (no HF dependency in core):**

```python
def log_forward_pass(model, x, **kwargs):
    if isinstance(x, str):
        if hasattr(model, "to_tokens"):
            x = model.to_tokens(x)              # TransformerLens
        elif hasattr(model, "tokenizer"):
            x = model.tokenizer(x, return_tensors="pt").input_ids
        else:
            raise TypeError(
                "String input requires model.to_tokens(...) or model.tokenizer(...). "
                "For HuggingFace models, attach a tokenizer: "
                "model.tokenizer = AutoTokenizer.from_pretrained(...)"
            )
    return _existing_log_forward_pass(model, x, **kwargs)
```

Handle: single string, list[str] (batch passthrough to same path), helpful error otherwise.

**Apply to all entry points that take an input:** `log_forward_pass`, `trace` (if separate), `fastlog.record`, `Trace.rerun` (since rerun is for new inputs, string should work there too).

**Scope:** ~50 LOC + tests covering HookedTransformer, HF model with attached tokenizer, plain nn.Module raises helpful error, batched list[str], existing tensor input still works unchanged.

**Symmetric extension (optional v2):** vision models via duck-typed `model.processor` for PIL/numpy image inputs. Lower ROI; defer unless asked.

**Docs payoff:** README quickstart + flex notebook + Phase 4 comparison pages all become significantly cleaner. The "before/after" of demo code is dramatic.

**Priority:** Pre-launch. Should land alongside docs Phase 1 since it makes README and flex notebook code substantially shorter. Easy enough to slot in during Phase 0.

Filed 2026-05-15 from JMT proposing this as a TransformerLens-style ergonomic.

### Investigate: `rerun_no_save` slower than `rerun` (benchmark anomaly)

Perf benchmark addendum (commit `30dc452`) shows `Trace.rerun(model, x), no saved outs` runs slower than the with-save baseline across every cell:
- GPT-2-HF CPU: rerun 2984ms vs rerun_no_save 5725ms (~1.9x worse)
- ResNet-18 CPU: rerun 841ms vs rerun_no_save 2053ms (~2.4x worse)
- GPT-2-Hooked CUDA: rerun 3286ms vs rerun_no_save 6611ms (~2.0x worse)
- ResNet-18 CUDA: rerun 913ms vs rerun_no_save 860ms (~0.94x, only cell where no-save is faster)

Expected: rerun_no_save should be cheaper than rerun-with-save (fewer tensor copies). Observed: consistently slower.

Likely causes:
- Codex impl quirk in how `rerun_no_save` sets up the originating Trace (different code path triggers extra setup per call)
- `layers_to_save=[]` semantics may force a no-op-save path that's actually more expensive than the default-save path (counter-intuitive but possible if the empty-list case lacks an early-exit optimization)
- Memory-pass artifact: the 10 untimed runs in memory pass might be hitting a slow path that's measured

Action: read `benchmarks/perf_runner.py` `rerun_no_save` impl; check whether the originating Trace setup differs from the with-save case in unexpected ways. Possibly an actual TL perf bug in the no-save Trace setup path worth fixing.

Filed 2026-05-14 from perf addendum results. Not urgent but interesting.

### Perf sprint: zero-copy `save_mode="reference"` for activations

Current Trace save path does `.detach().clone()` (or `.detach().cpu()`) per saved tensor. For models with large activations (GPT-2 hidden states), the clone is the dominant fraction of save cost.

**Proposal:** add `save_mode` option (or `clone_saved_tensors: bool` flag) to `log_forward_pass` / `fastlog.record`:
- `save_mode="copy"` (default, current behavior; safe)
- `save_mode="reference"` (new): `.detach()` only, store tensor as-is, NO clone
- `save_mode="view"` (new, optional): keep autograd graph alive too — enables backprop through saved Trace tensors as if they were live model intermediates; useful for attribution-method wrappers
- `save_mode="cpu_async"` (future): pinned-memory async copy; overlaps with next GPU op

**Contract for "reference" mode:**
- Saved tensors are LIVE references to model intermediates
- Mutating them in-place invalidates the Trace
- Loud warning on first use per session
- Documented anti-patterns: don't run more forward passes with the same Trace held; don't use in-place ops on retrieved activations

**Risks to audit:**
1. Torch internal in-place ops (fused BN, fused activations in HF transformers) — verify which intermediates are mutated; document or restrict
2. GPU memory retention — references hold allocator pool; for deep models on small GPUs, may push past limits where copy mode would have been fine
3. Autograd graph survival — `.detach()` should break grad-fn, but verify no surprise retention

**Where it pays off:**
Read-only interpretability workflows (saliency, SAE feature extraction, activation inspection) never mutate saved tensors. These are the dominant use cases. Reference mode is the right default for them.

**Acceptance:** new benchmark row `trace_save_reference` runs 30-60% faster than `trace_save_copy` on GPT-2-HF CUDA; reference-mode Trace activations are bit-identical to copy-mode results after capture; in-place mutation triggers warning; no autograd memory leak in tests.

**Estimate:** 1-2 days. Touches `torchlens/backends/torch/...` save path + a clear API surface + tests + docs.

Filed 2026-05-14 from JMT spotting that the clone cost is independent of the wrapper-dispatch cost and is independently optimizable.

### Perf sprint: lazy fastlog ctx (only compute what predicate needs)

Current fastlog wrapper EAGERLY constructs the full ctx on every op:
- func_name, kind (cheap, always eager)
- source_location (expensive: `inspect.stack()` / frame walk, ~10-50us)
- module_address (~5us module-stack walk)
- args/kwargs refs (~1us dict/list construction)
- counter bookkeeping (~100ns)

If the user's predicate only touches `ctx.func_name`, the wrapper has done 50-100x the work it needed to.

**Fix: lazy ctx attributes.** Make `source_location`, `module_address`, `args_summary` into `@cached_property` or equivalent. Predicates that only read `func_name`/`kind` trigger zero expensive computation. Predicates that need more, or storage paths that need more, force materialization only on access.

**Constraint:** per-type counters (`relu_1_3` numbering) MUST still tick on every op — TL invariant for deterministic labeling. Counter cost is negligible.

**Expected gain:** ~50-70% reduction in per-op wrapper cost for the common predicate-on-func_name case. Translates to fastlog_zero landing at 2-5x raw forward instead of 15-30x on GPU transformer models (per benchmark surface).

**Estimate:** half-day to a day. Touches `torchlens/fastlog/_record_one_shot.py` ctx construction + `Recorder` consumers + all in-tree predicate examples.

**Risk:** medium. Need test coverage that lazy fields produce identical results when forced; need to verify all storage-path consumers tolerate lazy access.

**Acceptance:** new benchmark row "fastlog_zero, predicate touches only func_name" runs <5x raw forward on GPT-2-HF CUDA; existing fastlog tests pass; recorded events bit-identical to eager-ctx baseline.

Filed 2026-05-14 from JMT spotting the architectural slack during perf review.

### Sharpen comparison-page concessions before Phase 4 docs sprint

Concession paragraphs in `docs/comparisons/vs_*.md` were drafted too modestly in the v36 frozen plan. Specific corrections to apply when Phase 4 writes the pages:

- **vs_transformerlens.md**: TL "readable hook names" advantage shrinks dramatically because TorchLens has QKV tags + module-indexed access (`log["attention_1.q"]`). The real concession is just "if you already type `blocks.5.attn.hook_pattern` from muscle memory, TL's literal string match is what you want" — onboarding-style preference, not a capability gap. Tighten the concession sentence accordingly.
- **vs_captum.md**: "captum has attribution methods we don't" is a packaging difference, not a capability one. TorchLens already exposes gradients + activations + modules — all the substrate needed for IG/LRP/GradCAM/DeepLift. Could ship `torchlens.attribution.*` wrappers later. Honest concession is "captum has 6 years of tuned implementations + paper-validated hyperparameters", not "they can do something we can't." Consider adding attribution wrappers as a separate sprint after docs.
- **vs_torchexplorer.md**: TorchLens `Trace.draw(format="svg")` (verify this works) gives browser-native `<title>` hover. torchexplorer's real edge narrows to **full pan/zoom + clickable-node→tensor-inspector in one canvas**, not "any hover at all." Rewrite concession to that scope.

Filed 2026-05-13 from JMT pushback during plan freeze. Not blocking Phase 4 dispatch — corrections happen inside the comparison-page authoring tasks themselves.

### Power-user docs section opener: "universal interpretability substrate"

When Phase 4.3 / Phase 2 dispatches the power-user cookbook + reference section, lead with the substrate framing — this reframes the project from "another viz tool" into "the layer everything else builds on" and converts perceived overengineering (36-round-reviewed docs, parameter-grad parity validation, etc.) from "excessive" to "load-bearing for what builds on top."

**Draft opener (workshop at authoring time):**

> TorchLens is a universal interpretability substrate. Wrap every PyTorch op once at session start; the wrappers don't care what architecture sits on top. New architecture lands tomorrow — Mamba, RWKV, RetNet, the next thing — TorchLens captures it day one, no library update needed. That's what "universal" buys you, and it's why the validation infrastructure is the way it is: a substrate has to be load-bearing for whatever gets built on it.

**Two beats this opener carries that the rest of the section can lean on:**

1. **Architecture-scaling claim.** TL hooks *ops*, not *patterns*. Every peer tool has had to chase architectures (TransformerLens shipping `HookedMamba`, `HookedRWKV`, etc. as explicit work). TL shipped once and kept working. This is genuinely under-marketed and is a strong differentiator for the "I work on novel models" crowd — exactly the Anthropic-adjacent interp researcher persona the docs target.

2. **Validation-as-feature reframing.** Once you call it a substrate, the parameter-grad parity oracle, module-output gradient validation, `tl.validate(scope=...)`, `tl.compat.report` — all of that stops looking like perfectionism and starts looking like appropriate engineering for "the layer everyone trusts." Substrate credibility = validation infrastructure. Lean into it.

**Tone:** confident, not breathless. Anchor every claim on a concrete primitive (`tl.validate`, `tl.compat.report`, etc.) so it doesn't read as marketing.

Filed 2026-05-13 from JMT.


### Docs: emphasize "exhaustive metadata as queryable data" as a top differentiator

Sibling to the "universal interpretability substrate" opener above. Same docs sprint, different angle. The substrate framing is about SCOPE (works on any architecture). This framing is about DEPTH (every Op carries a comprehensive record). Both reinforce each other; the docs should hit both.

**Core claim:** TorchLens is the only PyTorch introspection tool that captures every operation as a record with full provenance — source location, graph topology, autograd metadata, gradients, memory accounting, conditional branches, args/kwargs templates — as queryable data that survives save/load. Peers each have a slice (nnsight = live module-output access; baukit = stored activations; TransformerLens = rich per-component but transformer-only; torch.profile = perf only). Nobody has the full column.

**Audit before authoring:** Confirm existing docs (overview, the existing "universal substrate" opener if landed, comparison pages, README) already hit this beat. If they do, sharpen and add concrete query examples. If they don't, add a section.

**Concrete query examples that demonstrate the breadth (each should be 5-10 lines max):**

1. **Source-line attribution of NaNs:** "Find every Op that produced NaN, grouped by source file/line." Joins `op.has_saved_activation` + activation check + `op.code_context[0].source_file`. One-liner-ish in pandas after `trace.to_dataframe()` (if that exists; if not, this argues for it).

2. **Replay one op with a perturbed weight:** Uses `args_template` + intervention API. Demonstrates that capture is full enough for selective replay, not just bulk re-run.

3. **Cross-checkpoint layer comparison:** Bundle two traces, walk `bundle.super_layers["attention.softmax"]` to compare outputs across versions. Works because Layer abstraction is stable across runs.

4. **FLOPs by source file:** `[(op.flops_forward, op.code_context[0].source_file) for op in trace.ops]` → group by file → sort. Identifies which user-code module is the hot path. Doesn't exist as a question elsewhere.

5. **Branch coverage across a dataset:** Run a bundle over N inputs, query `trace.conditional_records` for each, tally which arms fired. Branch coverage as a query.

6. **Gradient flow audit:** Per-Layer `gradient_memory` walks let you find dead branches (zero-gradient cliffs). Layer-wise rather than module-wise because Layer is the equivalence class of recurrent ops.

**Sound bites for inline use:**

- "Capture once, query forever." (Contrast with nnsight's "live access during trace.")
- "Every op is a row in a database you can join, filter, and group by."
- "`op.code_context` tells you which line of user code produced this tensor." (This detail surprises people every time — casual win that demonstrates breadth.)
- "TorchLens isn't an interpretability tool. It's the substrate other interpretability tools should be built on." (Strongest framing; positions TL as infrastructure.)

**Pitch arc (honest concession + payoff):**

> nnsight is the right tool when you need live module-output access during a trace, especially on NDIF-hosted models. TorchLens is the right tool when you want to capture an execution, save the whole record, and then ask arbitrary structured questions about it — across runs, across models, with full provenance — without re-running capture each time.

The "across runs, across models, with full provenance" part is the piece peers can't match. Lean into it.

**Section recommendation:** Either a dedicated `docs/the-substrate.md` (or whatever name fits the existing tree) that walks through the 5-6 representative queries above, OR a "Why exhaustive metadata?" subsection inside the overview / first-pass docs. The "ah, this is different" moment for new users comes when they SEE the breadth composed in real queries, not when they read a feature list.

**Coordinates with:** the existing "Power-user docs section opener: universal interpretability substrate" (above) — these two together form the differentiator pitch. Substrate = scope, exhaustive metadata = depth. Authoring them as a pair is cleaner than alone.

Filed 2026-05-23 from JMT.


### Comprehensive `__str__` / `__repr__` audit across all public classes (**PRE-2.0 PRIORITY**)

**Priority upgraded 2026-05-23:** moved from docs-sprint nice-to-have to **pre-2.0 commitment**. Rationale: `__repr__` is effectively a public API contract for data-heavy libraries. Users will parse it, bake it into test goldens, reference it in tutorials, and AI agents will pattern-match on it. Post-launch changes are sticky in spirit even if Python doesn't formally enforce. Better to lock the format BEFORE 2.0 ships than to break users later.

Audit and sharpen the string-formatting surface on EVERY public class TorchLens exposes — records, accessors, containers, value-types. Goal: every `print(x)` or `repr(x)` on a TL object produces immediately useful output without manual formatting. Default REPL ergonomics matter for adoption.

**Framing — repr as documentation:** For data-heavy libraries (TorchLens, pandas, polars), `__repr__` is a first-class documentation surface, not cosmetic polish. The repr documents object IDENTITY and CURRENT STATE (what is this thing I'm holding? what's in it?). Docstrings document the CLASS (purpose, contract, usage). Both layers are needed; the repr layer is the one users hit first in REPL/Jupyter/agent workflows.

**Why repr matters for TL specifically:**
- Jupyter renders last-expression repr automatically. `trace.ops` should render informative output immediately, not `<torchlens.OpAccessor at 0x...>`.
- AI-agent workflows can't easily read external docs — informative reprs ARE the discoverable API surface.
- TL's value-add is the data, not the methods. Users spend orders of magnitude more time inspecting records than calling library functions.

### Visual-language design principle (consistent across all classes)

Lock ONE pattern and apply it uniformly:

```
ClassName(<identifying field>, <scope or shape>, <one headline stat or key=value summary>)
```

Examples (illustrative):

```
Op(conv2d_2:1, layer=conv2d_2, shape=(1,64,28,28))
Layer(conv2d_2, passes=3, shape=(1,64,28,28))
Module(encoder.block.0, TransformerBlock, calls=12, params=3.4MB)
ModuleCall(encoder.block.0:1, TransformerBlock, 4.2ms)
Param(encoder.block.0.weight, (768,768), float32, trainable)
GradFn(AddmmBackward0, calls=12, has_op=True)
ParamAccessor(N=12, scope=Module<encoder>, total_memory=3.4MB)
OpAccessor(N=4328, scope=trace)
OpAccessor(N=12, scope=Layer<attention>)
Bundle(my_sweep, members=5)
SuperOpAccessor(conv2d_2:1, traces={A, B, C})
```

Reading rules:
- Field 1: the one thing that uniquely identifies this instance to the user (label, address, class_name).
- Field 2: scope or shape — orienting context (which Layer? which Module? what's the data shape?).
- Field 3+: one or two headline stats (size, memory, duration) — what the user most wants to know at a glance.

Consistent across record / accessor / container classes. Same parse pattern: `Name(id, scope, *summary)`. Extensible (more fields can be added at end without breaking the leading pattern). Parseable (downstream tools can rely on the form).

**Scope — every public class:**

#### Record classes (the "data" surface)

| Class | Proposed one-line repr |
|---|---|
| `Op` | `Op(conv2d_2:1, layer=conv2d_2, shape=(1,64,28,28))` |
| `Layer` | `Layer(conv2d_2, passes=3, shape=(1,64,28,28))` |
| `Module` | `Module(encoder.block.0, TransformerBlock, calls=12, params=3.4MB)` |
| `ModuleCall` | `ModuleCall(encoder.block.0:1, TransformerBlock, 4.2ms)` |
| `Param` | `Param(encoder.block.0.weight, (768,768), float32, trainable)` |
| `Buffer` | `Buffer(encoder.block.0.norm.running_mean, (768,), float32)` |
| `GradFn` | `GradFn(AddmmBackward0, calls=12, has_op=True)` |
| `GradFnCall` | `GradFnCall(AddmmBackward0:1, 1.8ms)` |
| `Trace` | Multi-line summary block by default (this is the entry point — users always print it first). Includes: model class, num_ops, num_layers, num_modules, num_params, total_activation_memory, capture_duration. Report-card format. ALSO offers a `repr_one_line()` for embedding in collections. |
| `Bundle` | `Bundle(my_sweep, members=5)` |
| `Super[T]` family (SuperOp, SuperLayer, etc.) | `SuperOp(conv2d_2:1, traces={A, B, C})` |
| `ConditionalEvent`, `ConditionalArm`, `Conditional` | `Conditional(cond_id, fired=if_branch)` / `ConditionalArm(name, ops=15)` |
| `HookInfo` | `HookInfo(forward, fn=my_hook)` |
| `FuncCallLocation` | `FuncCallLocation(model.py:142, MyModel.forward)` |

#### Accessor classes (the "navigation" surface) — equally important

This is where JMT specifically pushed for audit scope (2026-05-23). Accessors are what users see when they print `trace.ops` or `module.calls` — and the default Python `repr` for these is currently useless.

| Class | Proposed short-form repr |
|---|---|
| `OpAccessor` (trace-scope) | `OpAccessor(N=4328, scope=trace)` |
| `OpAccessor` (Layer-scope) | `OpAccessor(N=3, scope=Layer<conv2d_2>)` |
| `LayerAccessor` | `LayerAccessor(N=87, scope=trace)` |
| `ModuleAccessor` | `ModuleAccessor(N=12)` |
| `ModuleCallAccessor` | `ModuleCallAccessor(N=42, scope=Module<encoder.block>)` |
| `ParamAccessor` | `ParamAccessor(N=12, total_memory=3.4MB)` |
| `BufferAccessor` | `BufferAccessor(N=6)` |
| `GradFnAccessor`, `GradFnCallAccessor` | Same pattern |
| `Super[T]Accessor` family | `SuperOpAccessor(N=4328, traces={A, B, C})` |
| `TraceAccessor` | `TraceAccessor(trace=my_run, ops=4328, layers=87, modules=12)` |

**Long-form `__repr__` for accessors:** tabular preview. Show first ~5 members in a column-aligned table with the most useful field for each member type (label + shape for Op/Layer, label + class_name for Module, etc.). For accessors with >10 members, render `[ first_5 ] ... [ last_2 ]` style. Inspired by pandas DataFrame repr — users intuit "this is a collection I can iterate / index" from the visual.

#### Value-type classes

| Class | Treatment |
|---|---|
| `CaptureOptions` (and similar config dataclasses) | One-line summary of NON-DEFAULT values only: `CaptureOptions(save_gradients=True, backward_ready=True)`. Don't dump every default. |
| `ContainerSpec` | One-line: shape of the container structure. |
| `Shape` (if there's a wrapper type) | `Shape(1, 3, 224, 224)` (mimic tuple repr but tagged with class). |
| Any other public dataclass | Audit case-by-case. |

### `.locator` property — stable opt-in formatted surface

Separate from `__str__` (which has soft stability obligations). The `.locator` property is the EXPLICIT, contract-stable formatted-string surface. Users opt into it for diagnostic / logging / report-generation; `__str__` can evolve more freely.

```python
op.locator         # "conv2d_2:1 (model.py:142, MyModel.forward)"
layer.locator      # "conv2d_2 (passes=3)"
module.locator     # "encoder.block.0 (TransformerBlock, calls=12)"
module_call.locator  # "encoder.block.0:1 (TransformerBlock @ 4.2ms)"
```

Diagnostic functions can `return op` and the user does `print(f"NaN at {op.locator}")`. Stable contract for formatted output, independent of `__str__` evolution.

### Trace-level summary methods (richer than `__repr__`)

Beyond `__str__` / `__repr__`, decide if there should be richer summary methods:

- `trace.summary()` — multi-line report card (probably already exists or planned; verify and align).
- `module.summary()` — per-module summary including descendant calls, param footprint, hook count.
- `bundle.summary()` — cross-trace summary table.

These are richer than default `__repr__` — meant for explicit human-friendly inspection, not REPL default.

### Stability concerns

- `__repr__` output may be load-bearing for tests / golden output. Audit whether any current tests assert on exact repr strings before changing them. If yes, update goldens in the same commit.
- The `.tlspec` serialization format is independent of `__str__` / `__repr__`, so save/load is not affected.
- Long-form accessor reprs (the tabular preview) need a max-width / max-rows constraint so they don't break terminal display.

### Implications for users / downstream tooling

- Document the visual-language pattern in the API reference so users know what to expect.
- Provide a `repr_v=2` or similar versioning if the format ever needs to change — users can opt into newer formats.
- AI-agent-friendly: agents pattern-matching on `Name(id, ...)` style reprs benefit from consistency.

### When to do it

**Pre-2.0 ship.** No longer deferrable to docs sprint. The repr surface IS public API.

Sequencing suggestion:
1. Lock the visual-language pattern (`Name(id, scope, *stats)`).
2. Implement per-class one-line reprs across all 30+ classes.
3. Implement long-form tabular previews for accessors.
4. Implement `.locator` properties on records.
5. Audit existing tests for golden-repr dependencies; update goldens in the locking commit.
6. Document the pattern in API reference + AI-agent docs.

### Audit deliverable

A doc page listing every public class, its current `__str__` / `__repr__` output, the proposed output following the visual-language pattern, the rationale, and any breaking-change risk for existing test goldens. ~30 classes total. Self-contained pre-2.0 task.

### If we later need a `Locus` class

The forcing function is a diagnostic that returns a heterogeneous "where" (e.g., gradient flow audit returning op-to-op edges with magnitudes — see the creative-ideas section). At that point, revisit the Locus question. Until then, the per-class `__str__` + `.locator` covers the need.

Filed 2026-05-23 from JMT (initial filing); scope expanded same day to include all Accessor classes; priority upgraded same day to pre-2.0 commitment with visual-language design principle.

**Context:** filed during a riff (2026-05-23) about whether to introduce a dedicated `Locus` / `Site` class for diagnostic return types. Decided that's excessive — Op already carries the location info. The lighter and broader move is to give every TL class better string formatting, so most diagnostic functions can just return records and `print(f"NaN at {op}")` works, AND interactive REPL inspection of any TL surface is informative by default.

**Scope — every public class:**

#### Record classes (the "data" surface)

| Class | What `__str__` / `__repr__` should show |
|---|---|
| `Op` | One-line: pass-qualified label + source `file:line` if available. Example: `Op(conv2d_2:1 @ model.py:142)`. |
| `Layer` | One-line: layer label + num_passes + representative shape. Example: `Layer(conv2d_2, passes=3, shape=(1,64,28,28))`. |
| `Module` | One-line: address + class_name + num_calls + recursive_param_memory. Example: `Module(encoder.block.0, TransformerBlock, calls=12, params=3.4MB)`. |
| `ModuleCall` | One-line: call_label + class_name + forward_duration. Example: `ModuleCall(encoder.block.0:1, TransformerBlock, 4.2ms)`. |
| `Param` | One-line: address + shape + dtype + trainable flag. Example: `Param(encoder.block.0.weight, (768,768), float32, trainable)`. |
| `Buffer` | One-line: address + shape + dtype. Example: `Buffer(encoder.block.0.norm.running_mean, (768,), float32)`. |
| `GradFn` | One-line: class_name + num_calls + has_op flag. Example: `GradFn(AddmmBackward0, calls=12, has_op=True)`. |
| `GradFnCall` | One-line: call label + backward_duration. |
| `Trace` | Multi-line summary block by default (this is the entry point — users always print it first). Include: model class, num_ops, num_layers, num_modules, num_params, total_activation_memory, capture_duration. Format like a small report card. |
| `Bundle` | One-line: bundle label + N member traces. |
| `Super[T]` family (SuperOp, SuperLayer, etc.) | One-line: aggregate label + N traces present. |
| `ConditionalEvent`, `ConditionalArm`, `Conditional` | One-line: conditional_id + which arm fired (for `Conditional`) / arm name (for `ConditionalArm`). |
| `HookInfo` | One-line: hook type + handle status. |
| `FuncCallLocation` | One-line: `file:line` (function_name). |

#### Accessor classes (the "navigation" surface) — equally important

This is where JMT specifically wants the audit pushed (2026-05-23). Accessors are what users see when they print `trace.ops` or `module.calls` — and the default Python `repr` for these is currently useless.

| Class | What `__str__` / `__repr__` should show |
|---|---|
| `OpAccessor` (trace-scope and Layer-scope variants) | One-line: `OpAccessor(N ops, scope=trace)` or `OpAccessor(N ops, scope=Layer<conv2d_2>)`. Tabular preview on long-form repr (e.g., first 5 + last 2 with "..." in the middle). |
| `LayerAccessor` | Same pattern: `LayerAccessor(N layers, scope=trace)`. |
| `ModuleAccessor` | `ModuleAccessor(N modules)`. |
| `ModuleCallAccessor` | `ModuleCallAccessor(N calls, scope=trace)` or `ModuleCallAccessor(N calls, scope=Module<encoder.block>)`. |
| `ParamAccessor` | `ParamAccessor(N params, total_memory=...)`. |
| `BufferAccessor` | `BufferAccessor(N buffers)`. |
| `GradFnAccessor`, `GradFnCallAccessor` | Same pattern. |
| `Super[T]Accessor` family | `SuperOpAccessor(N ops, traces={A,B,C})` — bundle scope visible. |
| `TraceAccessor` (the universal lookup on Trace) | `TraceAccessor(trace=<label>, ops=N, layers=M, modules=K)`. |

**Long-form repr (`__repr__`) suggestions for accessors:** tabular preview. Show first ~5 members in a column-aligned table with the most useful field for each member type (label + shape for Op/Layer, label + class_name for Module, etc.). For accessors with >10 members, render `[ first_5 ] ... [ last_2 ]` style.

Default REPL behavior matters: if a user types `>>> trace.ops` they should see something informative immediately, not `<torchlens.OpAccessor object at 0x...>`. This is a 30-second UX win that pays compounding interest.

#### Value-type classes

| Class | Treatment |
|---|---|
| `CaptureOptions` (and similar config dataclasses) | One-line summary of non-default values: `CaptureOptions(save_gradients=True, backward_ready=True)`. Don't dump every default. |
| `ContainerSpec` | One-line: shape of the container structure. |
| `Shape` (if there's a wrapper type) | Mimic tuple repr but tagged: `Shape(1, 3, 224, 224)`. |
| Any other public dataclass | Audit case-by-case. |

#### Trace-level summary methods

Beyond `__str__` / `__repr__`, decide if there should be richer summary methods:

- `trace.summary()` — multi-line report card (probably already exists or planned; verify).
- `module.summary()` — per-module summary including descendant calls, param footprint, hook count.
- `bundle.summary()` — cross-trace summary table.

These are richer than default `__repr__` — meant for explicit human-friendly inspection, not REPL default.

**Stability concerns:**

- `__repr__` output is sometimes load-bearing for tests / golden output. Audit whether any current tests assert on exact repr strings before changing them. If yes, update goldens in the same commit.
- The `.tlspec` serialization format is independent of `__str__` / `__repr__`, so save/load is not affected.

**Recommendation: introduce `.locator` property on records as a separate, stable surface:**

```python
op.locator        # "conv2d_2:1 (model.py:142, MyModel.forward)"
layer.locator     # "conv2d_2 (passes=3)"
module.locator    # "encoder.block.0 (TransformerBlock, calls=12)"
```

Lets users opt into the formatted string without depending on `__str__` (which has stability obligations). Diagnostic functions can `return op` and the user does `print(f"NaN at {op.locator}")`. Stable contract for formatted output.

**When to do it:** docs sprint Phase 4 or whenever the public-API surface gets its docstring polish pass. Not blocking 2.0; not blocking impl sprints. Quality-of-life UX work tied to "what does the user see in the REPL / log output."

**Audit deliverable:** a table or doc page listing every public class, its current `__str__` / `__repr__` output, the proposed output, and the rationale. ~30 classes total. Could be a self-contained Phase-4 task.

**If we later need a `Locus` class:** the forcing function is a diagnostic that returns a heterogeneous "where" (e.g., gradient flow audit returning op-to-op edges with magnitudes — see the creative-ideas section). At that point, revisit the Locus question. Until then, the per-class `__str__` + `.locator` covers the need.

Filed 2026-05-23 from JMT (initial filing); scope expanded same day to include all Accessor classes.


### Ship `torchlens.attribution.*` (post-docs sprint)

Attribution methods compute per-input-feature relevance scores from activations + gradients. TorchLens already has every primitive; this sprint just packages the standard recipes as thin wrappers. Flips the vs_captum comparison row from "use captum for IG/LRP" to "TorchLens ships native attribution; bridge to captum for long-tail methods."

**Tier 1 (easy, ~1h each, ~50 lines):**
- `torchlens.attribution.saliency(model, x, target)` — `grad(out[target], x)`, optionally × input.
- `torchlens.attribution.integrated_gradients(model, x, target, baseline=None, n_steps=50)` — trapezoid-sum of gradients along baseline→x path; uses Bundle for the batched interpolation.
- `torchlens.attribution.smooth_grad(model, x, target, n_samples=50, noise_std=0.1)` — mean of grads over noisy copies; uses replay.
- `torchlens.attribution.grad_cam(model, x, target, target_layer)` — ReLU(Σ α_k · A_k) at a chosen conv layer; uses selectors.

**Tier 2 (medium, ~half-day each):**
- DeepLift — careful chain-rule handling vs a reference baseline.

**Tier 3 (real work, defer or punt):**
- LRP — per-layer decomposition rules. Punt to `tl.bridge.captum` until/unless someone asks for it natively.

**API shape:** every method returns input-shaped relevance tensor + optional `viz` kwarg producing a saliency-map overlay (matplotlib backend, reuses `torchlens.viz` figure conventions).

**Test plan:** smoke against TinyNet + a 1-layer MLP per method; correctness validated against captum on the same input (within float tolerance) for IG/GradCAM where captum is the ground truth.

**Documentation:** new tutorial `docs/tutorials/13_attribution.md` (or extend tutorial 12); add cookbook recipe per method; vs_captum.md concession rewritten.

Estimate: ~1-2 days of focused work for Tier 1 + DeepLift. Filed 2026-05-13 from JMT pushback on captum concession.

### Sharpen the 50-word elevator pitch ("should sing")

Current pitch at plan §1.2 lines 743-747:
> TorchLens runs your PyTorch model normally and records every tensor-producing operation -- forward and backward -- into a structured `Trace`. The same selector DSL drives inspection, visualization, save/load, validation, multi-run comparison, and interventions. One model, one line, one rich queryable object.

JMT's note 2026-05-13: "I want it to really sing. The elevator should sound like an answer to prayers." Current version is competent-informational; doesn't telegraph the pain it solves before the reader has read the API.

**Target reader's prior pain:**
- 50 `register_forward_hook` calls to grab activations, manual cleanup, no persistence
- Manually tracking which intermediate tensor corresponds to which source line
- Forward = one API surface; backward = a different one
- Comparing two model variants = saving tensors to dicts by hand
- Can't snapshot a run and come back to it tomorrow

**The "answer to prayers" beat is the selector DSL itself.** It's the unifying primitive: one vocabulary for inspection, intervention, viz, comparison, validation. That's what no peer tool gives you. Lead with that, not with the structural description.

**Draft variants to workshop when the README gets written (Phase 1.1):**

Variant A (pain-first):
> Stop writing hooks. TorchLens runs your model and hands you a queryable `Trace` of every tensor it touched -- forward and backward, named and visualizable. The same selector vocabulary inspects, intervenes, compares, saves, validates. One model, one line, one DSL for everything.

Variant B (capability-led):
> One selector DSL. Inspection, intervention, visualization, save/load, validation, multi-run diff -- the same vocabulary drives all of them, on forward AND backward. Run your model normally. TorchLens captures every op. You query and manipulate from there.

Variant C (concrete):
> `log = tl.log_forward_pass(model, x)`. That's it. Now every tensor your model touched is named, queryable, visualizable, intervenable, comparable across runs, savable to disk, and validated against autograd -- through one selector vocabulary. Forward and backward in the same object. Skip the hook-writing tax.

Whichever lands: the pitch MUST lead with "you've been writing hooks, here's what replaces that" rather than "TorchLens has a data structure." Decision deferred to Phase 1.1 when the README author has the README hero figure in front of them.

Filed 2026-05-13 from JMT.

### Promote `docs/glossary.md` from excerpt to exhaustive reference

Plan §10 Phase 5.3 currently ships `docs/glossary.md` as a 150-line "user-facing excerpt of canonical glossary." The canonical glossary at `.project-context/torchlens_glossary.md` is already 1,135 lines and refreshed 2026-05-13, covering every class, attribute, function, argument, kwarg, return type, helper, observer, and convention.

**Action when Phase 5.3 dispatches:** instead of authoring a thin excerpt from scratch, *promote* the canonical glossary to `docs/reference/glossary.md` with a pruning pass. Strip agent-facing-only content (anything referencing `.project-context/`, agent dispatch protocols, retro notes), redact not-yet-shipped target names (e.g. the `Trace.backward` rename target stays out until renamed), and add a short "how to read this" header. Probably lands at ~600-800 lines post-pruning.

This becomes the single searchable exhaustive reference downstream tool builders reach for ("what does `Layer.passes` return?" "how does the Layer-vs-Op accessor rule work?" "what fields does `BufferLog` carry?"). Bumps Phase 5.3 from S to M sizing. Drop the 150-line `file_lines_max` budget on `docs/glossary.md`; replace with the tutorial/cookbook-class 800-line budget.

Filed 2026-05-13 from JMT.

### Ship `docs/for-ai-agents.md` (or `AGENTS.md` for downstream consumers)

Increasingly real audience: LLM coding agents (Claude Code, Codex, Cursor agents, Aider, etc.) that have been asked "use torchlens to do X" in someone else's codebase and need maximum signal-to-noise in minimum tokens. This is the inverse of the repo's existing `CLAUDE.md` (which guides agents working *on* TorchLens); this new doc guides agents working *with* TorchLens.

**Target content (~300-500 lines, high density):**
- One-paragraph mental model (selectors + Trace + replay/rerun)
- API quick reference table: all 47 public names, one line each, what they do, signature shape
- Common pattern templates as complete <30-line snippets: inspect activations / intervene mid-forward / compare two runs / save+load / backward inspection / fastlog sparse capture
- Anti-patterns table: don't log `torch.compile` / TorchScript / `torch.export`; don't run concurrent captures; don't store opaque callables in `.tlspec`; don't depend on hidden internal tensors of fused kernels
- Validation recipe: how to use `tl.compat.report(model, x)` + `tl.validate(...)` to confirm what was captured matches what was asked for
- Decision tree: full Trace vs fastlog Recording; forward-only vs forward+backward; eager vs MLX backend
- Error-recovery patterns: "if you see X, likely cause is Y" (e.g. "tl_module_address missing on tensor mid-forward = user injected a fresh tensor without going through the intervention API")
- Verification block at end: a 5-line snippet the agent can run to confirm a TorchLens-based change works

**Format conventions:**
- Plain prose with code fences, NO clever formatting — agents tokenize markdown literally
- Every snippet self-contained (imports + model + call + assert) so the agent can copy-paste
- Cross-link to `docs/glossary.md` for definitions; this doc is the action surface

**Verification:** add the file to `scripts/check_docs_api_truth.py`'s execution sweep so every code block in it runs on every PR — anything that drifts gets caught immediately.

Estimate: ~4-6h for a first draft including snippet verification. Probably authored at Phase 2.6 or as a new Phase 2.7. Could be a notable Anthropic-evaluator signal (the project takes AI-coding-agent usage seriously as a first-class audience).

Filed 2026-05-13 from JMT.

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

### Hot-path IR-only OpEvent construction (PARTIAL @4afa79c — full rewrite still pending)

**Status update (2026-05-11):** Mini-Sprint α item 3 moved `OpLog`
construction OUT of `backends/torch/ops.py` (zero `OpLog(` hits there
now) but capture still materializes a live OpLog because mid-forward
lookup + backward-hook behavior depend on it. The original Trojan-horse
pattern from M3 is slightly cleaner but persists in spirit.

Remaining work to fully realize the architectural intent:
- Refactor mid-forward lookups (`Trace._raw_layer_dict`-style reads
  during capture) to use the event stream or a lighter projection
- Refactor backward-hook attachment so it doesn't need a live OpLog
  mid-capture
- Move OpLog construction ENTIRELY into `postprocess/_materialize.py:
  materialize_into_build_state()`
- Parity gate must stay green byte-for-byte

Estimated remaining: ~4-8h codex once mid-forward consumers are
audited. Worth scheduling as its own focused sprint rather than
bundling with other cleanups.

### Mypy accessor typing errors — DONE (Mini-Sprint α item 1 @0f54ba5)

Resolved in commit `0f54ba5`. `mypy torchlens/data_classes/` is now
clean (16 source files, 0 errors). Codex fixed broader leakage too.

### Golden parity pickle size — DONE (Mini-Sprint α item 2 @0947830)

Resolved in commit `0947830`. Both resnet50 goldens now 131KB (was 765KB).
Solution: gzip-compressed pickle storage with transparent read/write
support in the parity test (input shape stayed similar; metadata
dominates pickle size more than input HW does, so compression was the
practical fix).

### Fastlog vs bare-forward perf benchmark gap (raised 2026-05-11)

There's no direct benchmark of `tl.fastlog.record(...)` overhead vs bare
`torch` forward in the repo. Only `benchmarks/intervention_overhead_results.md`
gives a TinyMLP baseline (bare 57 µs; `log_forward_pass(intervention_ready=False)`
32.3 ms = 564× slowdown — worst case because the model is tiny).

What we want to know:
- Per-op overhead for fastlog with all-False predicate (topology-only path)
- Same on realistic models: TinyMLP, ResNet-50, GPT-2 small, ViT
- How does the overhead scale with op count? With predicate complexity?
- Memory ratio (RAM footprint) under sparse-only vs full capture

Belongs in: the capture-pipeline-unification M3/M5 perf gates if not already
covered, OR a follow-up fastlog perf sprint. Plan §11 currently sets perf
bars for full capture (ResNet-50 ≤+5%, GPT-2 ≤+7.5%) but doesn't state a
bar for the fastlog/predicate=False path. Add: "predicate=False on ResNet-50
must be within Xx of bare forward" (X TBD — start with 2x, tighten with data).

Likely deliverable: add a `bench_fastlog_overhead.py` next to
`bench_log_forward_pass.py` with the same model set + RAM/disk/RAM-mirror
modes + variance reporting. Output to a new `benchmarks/fastlog_overhead_results.md`.

### Fastlog API ergonomics review (raised 2026-05-11, post capture-pipeline-unification)

JMT hasn't used the fastlog API yet. After the capture-pipeline-unification
sprint lands (M1-M8 complete + merged), do a hands-on review of `tl.fastlog`:

- `tl.fastlog.record(model, x, keep_op=...)` ergonomics
- `Recording` / `RecordingTrace` / `ActivationRecord` access patterns
- `RecordingTrace.draw()` graph visualization quality
- `RecordingTrace.timeline_html()` interactive view
- `RecordingTrace.repredicate(...)` — re-evaluate a different predicate against the stored event stream (probably the killer feature; verify it's discoverable in docs)
- Predicate composition with intervention selectors (`tl.func`, `tl.module`, `tl.label`, `tl.contains`, `tl.where`, `tl.in_module`)
- Storage modes (RAM, disk, RAM+disk mirror) — when does each make sense from a user POV?
- Disk bundle recovery via `tl.fastlog.recover()` — does it work cleanly?
- `dry_run()` workflow — predicate iteration without payload retention

After hands-on: file friction-points as separate items, possibly bundle
into a fastlog ergonomics sprint. Particularly relevant once the unified
pipeline ships (Recording becomes a lazy projection over CaptureEvents
per AD-1; verify user experience is identical post-refactor).

Decision deferred under AD-1: should fastlog disk format unify into
`.tlspec` v4 or stay separate? Try the API first; that question may
become obvious one way or the other after real usage.

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

### [FIXED 2026-06-01, commit ea5dd69] distilbert_ffn facet recipe targets a stale transformers class name (filed 2026-06-01)

The built-in MLP facet recipe at `torchlens/semantic/recipes/mlp.py:58` registered against
`class_name="DistilBertFFN"`, but transformers >= 4.57 renamed that module class to `FFN`.
On current transformers the recipe matched NOTHING, so DistilBERT FFN facets silently failed to
populate. FIXED: changed to `class_name=("FFN", "DistilBertFFN")` (exact-match tuple, mirrors the
existing `distilbert_attention` convention) so both current and legacy names resolve. Verified on
a real DistilBERT (facets populate non-empty); added `tests/semantic/test_mlp_recipes.py` (current
class + legacy class + real-model regression guards). Surfaced 2026-06-01 during HF notebook finalization.

## Creative utility ideas from exhaustive metadata (filed 2026-05-23)

Ideas surfaced during a riff session with JMT about what becomes possible once every Op carries full provenance + topology + memory + autograd + source-code metadata. Organized by ambition tier so near-term and strategic items are visible. Each entry names the TL primitives it builds on so the gap-to-ship is obvious.

### [SMALL, raised 2026-06-02] Trace should expose an edge count (`num_edges`)
Trace has `num_ops` / `num_layers` / `num_layers_with_params` etc. but NO edge count. Edge count is a genuinely useful quantity (graph density, layout-cost estimation — DOT's cost tracks edges far more than nodes, complexity reporting). Add a `num_edges` field/`@property` on Trace (and likely the equivalent on Module / Layer where it's meaningful). Decide: count graph edges (op->op adjacency) vs something richer. Glossary entry + `constants.py` FIELD_ORDER if it's a stored field. Cheap to derive.

### Tier 1 — could ship in a week, ~50-200 lines each

#### `tl.bisect_nan(trace)` — first-NaN-op locator
Binary-search the topo-sorted graph for the first op that produced NaN. Returns the source-line that introduced it. Builds on `op.has_saved_activation` + `op.parents` topo + `op.code_context`. "Why does my model NaN" becomes a one-liner.

#### `tl.hot_path(trace, by="flops" | "memory" | "duration")`
Rank ops by cost, group by `code_context[0].source_file:line`, return a sorted DataFrame. Output is "75% of forward FLOPs come from these 3 source lines." Identifies micro-opt targets in 30 seconds. Builds on `op.flops_forward`, `op.activation_memory`, `op.func_duration`, `op.code_context`.

#### `tl.recompute_candidates(trace, budget_gb)`
Find ops with high `activation_memory` but low `flops_forward` (cheap to recompute, expensive to store). Returns a ranked list with the suggested gradient-checkpointing boundary. Replaces the "guess where to put `torch.utils.checkpoint`" UX with a data-driven recommendation.

#### `tl.dead_neurons(trace, dataset=...)`
Bundle N traces, per-Layer compute activation stats (always-zero / always-saturate), return a table. Existing primitives: Bundle + Super[T] + per-Op `activation`. Useful for pruning, training diagnostics, model surgery.

#### `tl.gradient_flow_audit(trace)`
Walk the backward graph (GradFn records), find Layers where `total_gradient_memory` drops by >Nx vs upstream. Surfaces vanishing-gradient cliffs with source-line attribution. "Why is my model not learning" → one query.

#### `tl.to_dataframe(trace)` — the killer power-user feature
Emit a flat pandas/polars DataFrame of all Ops with all queryable fields as columns. Suddenly users do SQL/pandas on neural-net execution. This is the "huh, that's different" feature — the metadata IS the data. Probably the single highest-leverage Tier 1 item.

#### `op.lineage()` — single-tensor ancestor walk
Given any Op, walk its ancestors back to inputs. Returns the chain of ops that produced this tensor, plus per-step source line and shape transformations. "Where did this tensor come from?" as a method, not a manual walk.

#### `tl.compare(trace_a, trace_b, by="layer")` — diff engine
Bundle two traces, walk Super[T], return a per-Layer diff: shape changed? activation diverged by >X? Layer label only in one? Foundational engine — the use cases (pre/post fine-tune, FP32 vs FP16, architecture A/B, regression detection) all reduce to this primitive.

### Tier 2 — couple weeks, more design

#### `tl.subgraph_search(trace, pattern)`
Pattern-match subgraphs in the captured graph. Find all "attention blocks" defined as `matmul → softmax → matmul` on the same tensor lineage. Generalizes to any structural motif. Foundation for higher-order abstractions.

#### Module-call SQL
`trace.module_calls.query("class_name == 'TransformerBlock' and forward_duration > 1e-3")`. Pandas-style or DuckDB-backed. Joins naturally with `to_dataframe()`. Lets users compose queries without learning the TL API.

#### `tl.replay(op, args=...)` — selective replay
Replay a single op with modified inputs, using `args_template` to reconstruct the call. Selective sensitivity analysis without a full forward pass. The intervention API does whole-pass replays; this is the surgical-strike variant.

#### `tl.narrate(trace)` — auto-generated walkthrough
Walk the call tree, render natural-language description. "First, an embedding layer maps 50257 tokens to 768-dim vectors. Then 12 transformer blocks each apply self-attention (8 heads, 768 dim) followed by MLP (3072 dim hidden) with residual connections..." Builds on `class_name`, `forward_args_template`, `num_params`, call tree. Educational + AI-agent-friendly output.

#### `tl.audit(trace, against=baseline_tlspec)` — CI-grade regression detector
Diff structure + numerics against a saved baseline tlspec. Returns: new ops, removed ops, shape changes, numerical drift > tolerance, FLOPs change. Save a tlspec on green; trip the build if the next capture differs unexpectedly.

#### `tl.influence_map(trace, output_op, input_tokens=...)`
Given a model output, walk the autograd graph backward weighted by gradients to surface which inputs influenced it most. Already exists in spirit via existing attribution libs, but TL's version operates on captured records, not live tensors, so it's offline + reproducible + supports any architecture without per-arch adaptation.

#### `Bundle.activation_atlas(layer_label, projection="umap")`
Capture activations of one Layer across a dataset, project to 2D, return a structure that visualizes clusters (semantic structure of what the layer "sees"). Anthropic's Activation Atlas paper, but as a built-in TL primitive on top of Bundle + Super[T].

### Tier 3 — strategic, months but high-leverage

#### Architecture-pattern extractor
Given N captured traces (different models), extract the empirically-recurring structural motifs. "These 8 models all have a 'transformer block' defined as `[norm, attention, residual, norm, mlp, residual]`." Bottom-up architecture taxonomy from execution data, not from human-curated class names.

#### Auto-decomposition tool
Given a Module (any depth), propose a sensible decomposition into named functional sub-blocks based on call-tree structure + recurring patterns. The "find the natural seams" tool. Useful for understanding novel architectures.

#### Cross-framework graph export
tlspec → JAX/ONNX/MLIR with full metadata preserved. The portability claim becomes load-bearing: "captured in PyTorch, analyzed offline, replayed in JAX." Massive leverage with the backend protocol that's already on the roadmap.

#### Differentiable graph rewriting
Pick an Op pattern in the trace (e.g., `matmul → add → activation`), substitute with an optimized fused version (e.g., FlashAttention, fused linear+bias+relu). Output: a rewritten model. Bridges interpretability tool → compiler-style optimization. Niche but real demand.

#### Live training dashboard
Periodic lightweight traces during training, dashboard the per-Layer stats over time. Vanishing gradient detection, dead neuron emergence, capacity utilization. The "what's my model actually doing as it trains" tool. Existing tools (W&B) track scalars; this would track structural+numerical state.

### Curiosity / "wow factor" ideas

#### Computational receipt — print at end of `tl.trace()`
```
forward pass complete.
  17.3 GFLOPs across 4,328 ops
  2.1 GB peak activation memory
  3 source files, 12 modules touched
  ~$0.00012 on AWS p4d.24xlarge
```
Costs nothing (you have all the data) and is delightful. Doubles as a sanity check.

#### `tl.heredity(tensor_label).draw()` — single-tensor family tree
Render the family tree of one specific tensor: every op that contributed to it, backward to inputs, with shape annotations. Single-tensor focused viz instead of whole-model.

#### Source-file heat map — IDE plugin material
Overlay FLOPs / memory / gradient norm on the user's source code as colored highlights. "Show me my model.py with the hot path in red."

#### Architecture fingerprint
Hash the structural graph (post-equivalence-class folding) to get a model fingerprint. Two models with the "same architecture but different weights" share a fingerprint. "Is this model architecturally novel?" → one-line check. Useful for paper-review automation, ensemble dedup, model-zoo dedup.

#### Conditional coverage report
Given a model with data-dependent control flow and a dataset, report which branches fire on what % of inputs. "Code coverage but for neural-net branches." Conditional records are already first-class data; this is just a one-screen summary.

### The pattern across these (worth keeping in mind during docs sprint)

The metadata enables three categories of tooling that don't exist elsewhere:

1. **"Database queries on neural-net execution"** — `to_dataframe`, hot_path, dead_neurons, compare, audit. Everything reduces to "query this column / join with that one." Metadata is the schema.
2. **"Source-code as a first-class dimension"** — heat maps, NaN bisection by source line, FLOPs by file, narration. `code_context` is the bridge between captured execution and user-readable code. Nobody else has this bridge.
3. **"Cross-trace algebra"** — compare, audit, influence map, activation atlas, architecture fingerprint. Bundle + Super[T] is the type system for "two captures of the same shape." Once you have it, comparison primitives compose freely.

Each category alone would be a differentiator. TL has all three because the metadata is exhaustive enough that the operations are meaningful.

**Killer demos for the docs:** #1 from each tier roughly — Tier 1 `to_dataframe()` (power-user gateway drug), Tier 1 computational receipt (first-impressions delight), Tier 2 audit/compare (production-engineering pitch), Tier 3 architecture-pattern extractor (strategic-moat play).

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

### Capture-path unification: log_forward_pass internally as Recording (raised 2026-04-29; reaffirmed 2026-05-15)

**Status:** post-launch wave 2 destination. The arc-of-history endpoint, not a maybe.

JMT 2026-05-15 reaffirmed: "the arc of history trends towards unifying here. Just cleaner." Architectural conviction is now explicit — this is where TorchLens is going, the question is when not whether.

**Why it's the destination (vs the original "possibly never" framing):**

1. **Maintenance compounds against you with two paths.** Every fix lands twice; asymmetric fixes are where bugs hide after a few years of capability stacking.
2. **New capabilities land naturally on both sides post-unification.** `tl.halt()` for Trace, streaming/disk-backed Trace, lazy-promoted Recordings — each becomes a tiny feature instead of a dual-impl project.
3. **Cleaner mental model for users and downstream tool builders.** "TorchLens captures op events; you choose retention scope (full vs predicate) and output shape (Recording vs Trace)." Single sentence.
4. **The 2026-05-14 perf benchmark weakened the original 5-30% regression objection.** Dominant costs (wrapper firing, tensor saves) are shared on the unified path. Postprocessing as a flag rather than inline could be a slight win — it lets the postprocess pass batch-process the event stream rather than build structure incrementally.

Hypothesis: `log_forward_pass` becomes "fastlog + keep_all + full_metadata + postprocess." Single capture path, divergent endpoints (Recording vs ModelLog). User-facing API unchanged.

**Design (JMT 2026-05-15, 4-point spec + 2 inherited checks):**

1. **One capture path** — wrapper firing logic is shared; today's fastlog hot path becomes the only hot path.
2. **Control how much info we capture** — `keep_op`, `keep_module`, `default_op`, `default_module` as the canonical knobs (today's fastlog API). `intervention_ready` and `save_tensors` (or `save_mode`) layer on top.
3. **Control whether we postprocess** — `postprocess=True` returns a Trace; `postprocess=False` returns a Recording. Same capture; different output shape.
4. **Capture-time vs postprocess-time filtering distinction.** Capture-time predicates work on op-level info (`ctx.func_name`, args, module address). Postprocess-time filters work on graph-derived labels (`linear_1_1`, layer paths) — the latter requires postprocessing because canonical labels don't exist until the postprocess pass assigns them.
5. (Inherited) **Backward threads through** — Recording carries backward op events tagged `kind="backward_op"`; postprocess folds them into the same Trace as forward ops.
6. (Inherited) **Halt + postprocess play nicely** — halt produces a partial-graph Trace through postprocess; no new mechanism.

**Realistic timeline (revised down from original "wave 2 multi-week"):**

| Step | Effort |
|---|---|
| Field-model reconciliation (RecordContext ↔ LayerPassLog) | 1 day |
| Core refactor: log_forward_pass → unified capture + postprocess flag | 1-2 days |
| Layer-label filter on postprocess pass | ½ day |
| Backward verification through unified path | ½-1 day |
| Bundle + Super* + intervention API verification (existing tests should mostly pass) | ½ day |
| Full test suite + perf benchmark verification (no regression on existing rows) | 1 day |
| **Total** | **~1 week of focused work** |

Post-launch placement still correct (after docs ship + planted flag); but the engineering is a single contained sprint, not a multi-week wave.

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

[PARTIAL -- the top-level `torchlens/visualization/elk_layout.py` file is GONE. But ELK is NOT fully retired: an `_elk_internal/layout.py` backend still exists and is wired into `rendering.py` as an escape-hatch engine (`vis_node_placement='elk'`, `render_elk_direct`, `render_with_sfdp` all import from `._elk_internal.layout`). The dedicated "delete all ELK" cleanup PR has NOT happened -- only the file was relocated/renamed.]

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

**[2026-06-02 code-verified findings + strengthened removal case]** Inspected
`_elk_internal/layout.py`. Key facts that sharpen the decision:
- ELK's PRIMARY path uses `elk.algorithm = "layered"` (`elk.direction="UP"`,
  `edgeRouting="ORTHOGONAL"`) — i.e. Sugiyama-style **directed** layered layout,
  the SAME family as graphviz `dot`. So "ELK is undirected" is NOT the right
  critique; layered is actually appropriate for directed graphs.
- BUT there's a `stress` sub-path (`_seed_stress_positions`) that IS
  undirected-family (force/MDS) and has to FAKE directionality by seeding
  positions, because "stress doesn't natively support elk.direction." And the
  no-Node.js fallback is graphviz `sfdp` (force-directed / undirected). So the
  degraded paths are the clunky ones.
- The real problem is architectural, not algorithmic: ELK is a **heavyweight,
  JS-dependent (Node.js + elkjs npm), one-off escape hatch** (subprocess
  fork+exec, RLIMIT_STACK hacks to avoid COW memory doubling, 120s timeouts,
  temp-file marshalling) that only fires above `_ELK_NODE_THRESHOLD = 3500`
  nodes — a regime where NO static single-image layout is actually readable.
- There is ALREADY a pure-Python, dependency-free, O(n+m) directed
  topological-rank layout (Kahn's algorithm) used for the `>100k` node tier
  (`_ELK_STRESS_LIMIT = 100_000`, above which ELK's two O(n^2) stress distance
  matrices blow up — "16 TB at 1M nodes").
- **Recommended plan (JMT hot take 2026-06-02): rip ELK out entirely.** Promote
  the existing pure-Python Kahn topological-rank layout UP the ladder to cover
  the whole 3500–100k band (it's directed, zero-dep, already written). Collapse
  four layout paths (dot / elk-layered / elk-stress / sfdp-fallback + python-kahn)
  down to TWO: graphviz `dot` for small graphs, python-kahn rank layout for all
  large graphs. Delete the entire `_elk_internal` ELK pipeline + the Node.js
  script + elkjs dependency. Defer the *actually good* huge-graph experience
  (interactive / zoomable) to dagua, which is the only real answer for that
  regime. Counter-arg acknowledged: ELK-layered produces prettier crossing-min +
  orthogonal routing than naive Kahn ranks in the narrow 3500–~20k band — but on
  an already-marginal-readability graph that delta isn't worth a whole JS runtime
  dependency and 4 code paths, and dagua subsumes it anyway.

---

### More built-in visualization themes + revisit "custom visuals" interface (raised 2026-06-02)

Two related viz-UX asks from JMT:

1. **Ship more built-in visualization themes/presets.** Today the mode presets
   are limited. Add purpose-built themes, e.g.:
   - A **debugging-focused** theme — surface the metadata you actually want when
     hunting a capture/shape bug: shapes, dtypes, device, grad_fn, in/out
     mismatches, NaN/inf flags, untraced-op markers, highlighted boundaries.
   - Plausible others: a **minimal/publication** theme (clean, few labels, good
     for papers), a **memory/perf** theme (activation_memory + FLOPs/MACs heat),
     an **intervention** theme (highlight hook sites / edited tensors / forks),
     a **module-structure** theme (collapse to nn.Module tree, de-emphasize ops).
   - These should be named presets selectable via the visualization options, not
     hand-assembled NodeSpec callbacks each time.

2. **Revisit the "custom visuals" interface more generally.** Audit the
   NodeSpec / callback / mode-preset surface end to end: is it discoverable, is
   it composable (theme + user overrides), is the API ergonomic, are the
   built-in themes just well-chosen sets of the same primitives a user could
   reach for? Goal: a clean layered model where built-in themes and user custom
   visuals are the SAME mechanism, not two systems. Tie-in with the slice/facet
   integrated-treatment thinking (one coherent customization model).

---

### Revisit DenseNet visualization (ugly layout + sibling-ordering inert) (raised 2026-06-04)

Two related DenseNet rendering issues to revisit later:

1. **DenseNet renders ugly** (JMT recollection, 2026-06-04). Dense connectivity
   (every layer feeds every later layer in a block) produces a visually messy graph
   under the current Graphviz layout. Worth a dedicated pass on how to render dense
   connectivity readably (module collapse defaults? edge bundling? a DenseNet-aware
   preset?). Not yet diagnosed in detail.

2. **The sibling-ordering feature (2026-06-04 sprint) is conservatively INERT on
   DenseNet.** Measured during the adversarial design review: DenseNet121 = 0/58 fanouts
   ordered (4/593 siblings kept), because the sole-parent guard skips any sibling with
   more than one rendered parent, and dense connectivity means almost every consumer has
   multiple parents. This is CORRECT (skipped = today's layout, never wrong) but means
   the ordering feature does nothing for DenseNet. ResNet18 was 3/8 fanouts, transformer
   2/6, GoogLeNet 9/9 (the target case). If DenseNet ordering is later deemed valuable,
   it needs a different (non-sole-parent) safety story — revisit alongside (1).

See the sibling-ordering design docs (`/tmp/ordering_design_v3.md` + findings; should be
preserved to vault/.research) for the full guard rationale.

---

### Rolled view: reconcile loop-rolling vs module-rolling (raised 2026-06-04)

GENERAL issue (applies to MODULES and BUFFERS alike — NOT buffer-specific). The rolled
view has TWO independent collapse axes that can BOTH apply to the same node:
- **loop-rolling**: recurrence groups repeated graph positions into a Layer (passes `:N`).
- **module-rolling**: a module called multiple times groups into ModuleCall instances
  (`module_address:N`).

The collision is generic: e.g. a single-op module (a reused `ReLU`) that is BOTH called
multiple times AND sits inside a recurrent loop carries colons along two axes at once
(its Layer pass index AND its module-call index). What does the rolled node show, and
which axis collapses first / how do they compose? This predates buffers — buffers just
make it vivid.

Buffer instance of the same problem: a buffer rewritten in MULTIPLE loops spans multiple
buffer-Layers (positions) AND has a flat version index (`address:N`); in the rolled view,
two loop-positions of the SAME buffer both collapse to nodes labeled by the same address,
distinguished only by graph position.

Need a coherent, GENERAL story for: (a) which axis wins / how they compose when both
apply; (b) how the rolled node is LABELED when both axes apply (address + a position
hint?); (c) consistency across op / module / buffer rolling. Genuinely hard. RARE in
practice (a node that is both multiply-module-called AND loop-recurrent), so it doesn't
block the buffer sprint — decide during or just after it. The buffer sprint's "How
buffers work" doc should at least NAME this case even if the rolled-view answer lands
later.

---

### [BIG, important] TorchLens does not capture buffer WRITES/OVERWRITES (raised 2026-06-04) — [DONE 2026-06-05, local main commits 91e1645..6e3a291 + 430357a]

SHIPPED. All three write kinds now captured as validatable version nodes (reassignment via
scoped class `__setattr__`; in-place via storage snapshot; fused/native via post-op value
snapshot). `num_overwrites` correct for BatchNorm train; `copy_` source edge restored;
mid-forward reassignment caught at any nesting level. Persistent `Buffer` entity shipped;
`Buffer(Op)` retired. Gradient flow verified; tripwire non-vacuous; `docs/buffers.md` + glossary
lockstep. RNN-cell loop-detection crash root-caused + fixed (`_scrub_per_op_equivalence_lists`).
See `.research/buffer-build_SUMMARY.md`. Original analysis below (archival):

Discovered during the buffer-sprint adversarial review (Codex + Claude both proved it
empirically). TorchLens currently captures only buffer READS as graph nodes; the WRITE
side is largely invisible:
- **BatchNorm train**: running-stat update is inside the fused `batch_norm` kernel ->
  only the initial (pre-update) read is captured; `num_overwrites=0`.
- **In-place (`mul_`/`add_`/`copy_`)**: captured as a plain compute op with
  `is_buffer=False`, NO buffer linkage, no new buffer version; `copy_` drops its source edge.
- **Mid-forward reassignment (`self.b = y+1`)**: invisible — `_tag_untagged_buffers`
  (`backends/torch/model_prep.py:724`) only tags at module-entry, so an assigned-then-read
  buffer isn't even a buffer node.
- Two-loop overwrite collapses into ONE Layer (`equivalence_class="buffer_<address>"`
  ignores loop position).
- `trace.buffers` ALREADY LIES for overwritten buffers (`num_overwrites` returns 0;
  collapsed-per-address accessor sees 1 of N ops; shared aliases undiscovered).

This is the proposal's flagged "hard part" (detecting overwrites), and it is NOT built.
Making buffer VERSIONING real (the dual-label, `value_after`, BatchNorm/in-place history)
requires a **capture-pipeline sub-project** ("Option B" in
`.research/buffer-sprint/SCOPE_NOTE.md`): intercept `nn.Module.__setattr__`/`register_buffer`;
post-call buffer snapshot/diff for fused ops; detect in-place ops mutating buffer-tagged
receivers and synthesize the write-version + rewire reads; fix `copy_` source edge; give
buffers loop-position context. WEEKS, risky (wrapper hot path + loop detection + validation).
Separate from the data-model refactor. See `.research/buffer-sprint/` for full empirical
findings.

**UPDATE 2026-06-05 (overnight effort):** The PROMISE this threatened (perfect replay) is
INTACT — the recurrent "failure" was a validator GT-aliasing false-negative, now FIXED + merged
(`39a5029`). Replay survives buffer mutation via `out_versions_by_child`. So the write-version
model is a data-model ENRICHMENT, not a replay requirement. Two dual-lab adversarial rounds
demolished the cheap `_version`-diff approach (systematic false-negative on fused BatchNorm;
alias/view/`.data` evasion; global `__setattr__` leak; new-identity-node double-modeling). The
**VALIDATED, lower-risk design** (both labs converged): build the version chain as a VIEW over
the existing `out_versions_by_child` value-diff (catches BatchNorm via value compare, under
`save_arg_values=True` precondition); the existing mutating op IS the version producer (no new
nodes); module-EXIT re-scan for reassignment (no global monkeypatch); keep "node only if read",
unread-write history on the entity. TWO DECISIONS for JMT first: (a) `save_arg_values=True`
precondition acceptable (memory)? (b) re-measure the dual-label/two-loop split once writes are
real. Full design + review evidence in `.research/buffer-sprint/PLAN_PHASE2.md` +
`PLAN2_REVIEW_*.md` + `.research/buffer-overnight_SUMMARY.md`.

---

### Static buffer as a model's SOLE output -> MetadataInvariantError (raised 2026-06-05)

A model whose ONLY output is a static buffer (`def forward(s,x): return s.b`, no other
consumer) captures no output layer -> `tl.trace` "succeeds" but has no `output_1`, and
`validate_forward_pass` raises `MetadataInvariantError: No output layers found`
(`validation/invariants.py:414`). Real bug (a valid model errors), but the fix is in
load-bearing postprocess (output-layer creation `graph_traversal.py` Step 1 / orphan pruning
Step 3) -> needs care, not an overnight rush. Repro: `/tmp/orphan_diag2.py`. Found during the
buffer overnight effort.

---

### Validation can't reset non-registered mutable state (raised 2026-06-05)

`validate_forward_pass` snapshots/restores `state_dict`, which only covers registered params/
buffers. A model using a plain attribute/list as mutable state (`self.state=[tensor]` mutated
in-place across the run) isn't in `state_dict`, so the trace run starts from the GT-run's
already-mutated state -> false-negative. Narrow/non-idiomatic (real models use
`register_buffer`, which validates correctly). Fix option: deep-copy the model (or all
TorchLens-tagged buffer-like state) for the GT run, or document as unsupported. Repro:
`/tmp/listinplace_diag.py`. Found during the buffer overnight effort.

---

### Edges as first-class objects (`Edge` / `trace.edges`) (raised 2026-06-05, DEFERRED)

Filed during the num_edges mini-sprint (counts shipped; see `.research/edges-proposal.md`).
Idea: promote edges from implicit `Op.parents`/`children` label lists to a first-class `Edge`
view + `trace.edges` accessor (sibling of ops/layers/modules/params/buffers), carrying endpoints,
`arg_position`, direction, the activation (`== source.out`), `crosses_module_boundary` +
modules entered/exited, `kind` (compute/buffer_read/buffer_write/conditional), `is_recurrent`,
`span` (topological jump -> skip/residual detection).

DEFERRED — convenience, not capability. Two reasons it's low-priority:
1. **Interventions don't need it (JMT 2026-06-05).** An edge edit is always reducible to a
   pre-hook on the CHILD targeting a specific arg slot (TL already stores `parent_arg_positions`),
   or a post-hook on the parent when it has a single child. Even fan-out-selective editing (edit
   A's contribution to B but not to C) reduces to (child node, arg_position) because merges happen
   AT a node whose input args keep parents separable. We already have predicates/selectors. So
   edges would be addressing sugar ("ablate edge attn->mlp" reads nicer than "pre-hook mlp arg 0"),
   never a new ability.
2. **Mostly projection + scale risk.** An Edge's data derives from its endpoints; edges >> nodes,
   so it MUST be lazy/on-demand views (never stored/serialized) or it blows up memory + `.tlspec`.

Gate building this on a concrete driver where edge-addressing genuinely beats node+arg addressing
(e.g. a viz/analysis workflow or an edge-targeted intervention API people actually ask for). If
ever built: lazy views only; consider `SuperEdge` for Bundle parity; do NOT add to locked-2.0 surface.

### Consider a broader Op-subclassing refactor (raised 2026-06-04, for completeness)

During the buffer design we chose plain `Op` + `is_buffer` flag over a `BufferOp`
subclass, because EVERY op-role in TorchLens is currently a flag (`is_input`,
`is_output`, `is_internal_source`, `is_internal_sink`, `is_terminal_bool`, `is_buffer`),
not a subclass — and subclassing buffers alone would be the inconsistent odd-one-out and
multiply buffer-related nouns. BUT if we ever want cleaner per-role types (`BufferOp`,
`InputOp`, `OutputOp`, …) with role-specific fields lifted off the base `Op`, that should
be ONE coherent refactor subclassing ALL op-roles, not a buffer-only special case. Filed
for completeness; NOT planned. It'd be a big, breaking, cross-cutting change (data model +
accessors + isinstance handling everywhere). Only worth it if the flat fat-`Op` record
becomes a real pain point. The lighter-weight fix for the only concrete complaint (field
clutter) is a role-gated repr, which the buffer sprint already does.

---

### Revisit loop-based indexing / recurrence semantics (raised 2026-06-04)

JMT feels good about the current loop-finding system (recurrence -> Layers, pass indices,
equivalence-class grouping), but flagged a leaner ALTERNATIVE worth riffing exhaustively
later: treat recurrence WITHOUT loop-finding — flat, un-grouped ops/versions, with
rolling/grouping made opt-in on demand rather than always-computed. Surfaced during the
buffer design (where version<->pass and the loop/module rolling collision live). Open
question: is the equivalence-class loop-finding machinery worth its complexity, or is a
leaner flat model + opt-in grouping cleaner and easier to reason about? Riff exhaustively
later. Closely related to the rolled-view loop-vs-module collision todo above.

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
  - **`OpLog` / `LayerLog` / `ModuleLog` etc. — DONE (shipped).** The
    `-Log` suffix was dropped from EVERY data class: `OpLog`->`Op`,
    `LayerLog`->`Layer`, `ModelLog`->`Trace`, plus `Module`,
    `ModuleCall`, `Param`, `Buffer`, `GradFn`, `GradFnCall`. The
    `nn.Module` collision was resolved by going with `Module` anyway
    (it reads as "the module" and is namespaced under `tl.`).
    `TensorLog = Op` is kept as a back-compat alias. The internal
    record module FILES were also de-`_log`'d this session
    (`op_log.py`->`op.py`, ..., `model_log.py`->`trace.py`). `tl.do`
    (do-operator) also already exists. Remaining naming work is the
    other bullets in this item (peek->pluck, poetic selectors,
    comparison-verb coherence, symmetric-pair audit).
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

- [PARTIAL -- the core primitive shipped: `tl.trace` has live `transform=` (user_input->model_input), `save_raw_input`, `output_transform`, `save_raw_output` kwargs; `Trace.raw_input`/`raw_output` are real instance fields (KEEP in portable spec, default save mode "small"); `rendering.py` has `_render_raw_input` + `image=`/`imagescale=true` for raw-input node rendering + a per-layer `visualizer_path` (image=) path. STILL OPEN: a dedicated `tl.transforms.*` built-in library (`hf_tokenize`/`image_preprocess`/etc.) does NOT exist (no `torchlens.transforms` module); the full per-modality batch-summary helpers (`tl.viz.batch_summary`, montage/text-table/waveform-stack) and `tl.register_input_renderer` aren't surfaced. Primitive + basic rendering done; the transforms-library + viz-polish tail is open.] Generic `transform=` kwarg on `tl.trace` + raw-input rendering
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

- [DONE verified 2026-06-01 -- `torchlens.bridge.hf.trace_text` (and `trace_image` / `trace_multimodal`) all exist and import cleanly; `tl.autoroute.input` is the priority registry `tl.trace` dispatches through by input type. Glossary v9 documents the shipped auto-routing + HF bridge tracers. The text-input ergonomic landed in the bridge as designed (core `tl.trace` stays domain-clean).] HuggingFace bridge: text-input ergonomics for language models
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

- [DONE verified 2026-06-01 -- the canonical kwarg on `tl.trace` / `SaveOptions` is now `activation_transform` (and `grad_transform`); `activation_postfunc` is NOT a live kwarg anywhere in torchlens/ (only the `ActivationPostfunc` type-alias NAME survives). The rename landed.] Rethink the parameter name `activation_postfunc` itself. Current name is
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

### 2026-06-01 glossary-code conformance + record-filename cleanup (commits bdf4f23..4f1b34f)

DONE this session on LOCAL main (not pushed). Code now FULLY conforms to the canonical
glossary (vault v9). tier-2 = 2312 passed / 0 failed.

- `bdf4f23` **Glossary conformance — canonical public names implemented.** Added all
  lock-backed public names to the record classes so code matches the v9 glossary. The
  spec-drives-code conformance sprint (Phases A/B) is RESOLVED — any earlier open items for
  "implement missing glossary APIs" / residual rename locks are RESOLVED this session.
- `01ac105` **Clean-break shim removal.** Dropped the conformance back-compat shims on record
  classes (hard break, pre-launch acceptable). RESOLVED this session.
- `362d52f` **`_log` -> de-suffixed record module filenames.** Record module files renamed to
  drop the `_log` suffix. RESOLVED this session. (NB: the broader `-Log` *class*-name rename
  — `OpLog`->`Op` etc. — is a SEPARATE naming-sprint-v3 item still OPEN; see Improvements.)
- `3858711` **Param `is_trainable` export fix.** Param export schema now uses `is_trainable`.
  RESOLVED this session.
- `32021bb` / `6c328bf` / `4f1b34f` — glossary lock-set marked resolved + agent-guide examples
  and HF tutorial updated to the current trace API.

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
