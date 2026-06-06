# Facets / outs / semantic-access — sprint proposal

## 0. Reminder: how it works TODAY (your "muddy in my brain")

Three DIFFERENT things exist right now, and they are genuinely separate mechanisms:

| Thing | What it is | Access | Source |
|---|---|---|---|
| `op.out` | the ONE output tensor of an op | `trace["linear_1"].out` | captured, automatic |
| `outs` (multi-output) | when one op/module returns MULTIPLE tensors (LSTM -> output,h_n,c_n; torch.max -> values,indices) — the raw structural list | `module_call.outs[1]`, `trace["lstm_1:2"].out` | captured, automatic (container_spec) |
| `facets` | SEMANTIC decomposition of an activation into named interpretable pieces (q/k/v, heads, MLP neurons) computed by a per-architecture **recipe** | `module.facets.q`, `module.facets.head(3).q` | user/library **recipes** (`tl.facets.register`), global registry |

So today's answer to "why are facets and outs separate?": **`outs` = raw structural multiplicity** (the model literally returned N tensors); **`facets` = derived semantic interpretation** (compute the Q sub-projection for you). They were built by different sprints, live in different files, and never touch each other. `facets` is READ-ONLY: it does NOT connect to intervention or gradients yet.

**The honest critique you're sensing is right:** from a USER's point of view, both answer the same question — *"give me a named/indexed sub-piece of this record's output."* The split is an implementation artifact, not a user-meaningful distinction. That's the unification opportunity (Fork B).

## 1. The competitive bar (TransformerLens 3.0 + nnsight 0.7)

- **TransformerLens**: semantic, model-independent hook names (`blocks.0.attn.hook_q` is `[batch,pos,head,d_head]` — head axis BUILT IN), a prebuilt `patching` module, head ablation = a one-line slice. BUT needs a per-architecture **adapter** (3.0's TransformerBridge widened this to ~9k models, still adapter-gated); clunky gradients.
- **nnsight**: wraps ANY `nn.Module` by attribute path (universal, zero porting), **first-class gradients** (`with loss.backward(): tensor.grad`, even in-flight grad editing), remote exec. BUT no head axis (you `einops.rearrange` a fused `d_model` by hand), raw module-path naming (no semantic "residual stream"), no built-in patching helpers.

**TorchLens's natural position = "nnsight's universality + TransformerLens's semantics + per-component gradients neither has cleanly."** TL already traces arbitrary eager models (universal) and has `find_sites`/`attach_hooks`/`fork`/`rerun`. Facets should add TL's semantic per-head layer, deliver it on ANY architecture, AND expose per-head gradients — the trifecta no rival offers.

Target capability checklist (the bar):
1. Per-head Q/K/V/Z/pattern access with an explicit head axis (match TL, beat nnsight's reshape tax).
2. Per-head residual contribution (TL `hook_result`).
3. **Per-head / per-facet GRADIENTS** (beat both — the flagship differentiator).
4. One-call ablation/patching of a head/component + prebuilt patching helpers (match TL `patching`, beat nnsight).
5. Universal architecture support — facets on models TL has no adapter for (beat TL).
6. Semantic names derived from the captured graph, with module-path fallback (synthesize both).
7. Attention pattern/scores as named facets without a source-trace escape hatch.

## 2. Proposed core model — facets as the ONE semantic-access layer

**Unify at the ACCESS layer, keep capture mechanisms distinct underneath.** A `facet` becomes the universal "named sub-view of a record's activation," and the current `outs` multiplicity is exposed AS default facets. Concretely:

- **ALL outputs become auto-facets (JMT-locked rule, revised 2026-06-05).** Every output is a facet through the one `.facets` door:
  - **named output -> its name** (dict keys; dataclass / NamedTuple / `torch.return_types` structseq fields, e.g. `torch.max` -> `values`/`indices`; HF `BaseModelOutputWithPast(last_hidden_state=, attentions=, ...)` -> `model.facets.attentions` FREE on any HF model).
  - **positional output -> `out{i}`** (honest positional addressing — transparently "output #0", not a fake semantic name). Single output stays just `.out` (no `out0`). Nested returns flatten with dotted index (`out1.0`, `out1.1`).
  - **recipe can ALIAS positional -> semantic** (`out0`->`output`, `out1`->`h_n`) so both resolve to the same tensor.
  - Principle: positional names are honest/transparent; semantic names come from the model (named outputs) or a recipe. JMT accepted `out0/out1` (2026-06-05) — completes the uniformity (no "named vs positional" special case).
  - DEPENDENCY: capture must PRESERVE output names — extend `container_spec` to keep dict keys / `NamedTuple._fields` / dataclass fields / structseq fields (today tracks tuple/dict/list shape, not the names). Small, load-bearing.
  - `outs` stays internally (capture truth); `module.outs` becomes a thin alias. The single `.facets` door covers structural + semantic facets.
- **Recipe facets** (q/k/v/heads/neurons) layer on top, same `.facets` access.
- **Every facet knows its provenance**: which record it lives on, and HOW it's derived from the raw activation (a slice spec / reshape / child-op reference). That provenance is what unlocks intervention + gradients (below).

**Core data model (JMT-locked 2026-06-05): facet = (home op reference, structural transform).**
Not an opaque tensor, not an arbitrary callable — a reference to a REAL captured op's output plus a
structural (reshape/slice/split/index) transform. Read / grad / intervene are THREE VIEWS of this one spec:
- **read** = `transform(home.out)`
- **grad** = `transform(home.grad)` — correct because a structural transform is a linear selection, so the
  gradient of the slice equals the same slice of the gradient (forward transform applied to the grad tensor).
- **intervene** = `write_back(home.out, edited_slice)` — needs the INVERSE direction (slice -> full tensor),
  because the forward transform is lossy (you can't run the model on a bare slice). The home op is a real
  intervention site; hook it and write the edited slice back.
A plain forward callable (`tensor -> slice`) suffices for read + grad; INTERVENTION additionally needs the
write-back. Two ways to provide it:
  - **(a) declarative transform** (REC for structural facets — ~all of them): author declares reshape/slice/
    split/index via primitives we provide; we auto-derive BOTH directions + know it's gradient-safe. No inverse written.
  - **(b) fn-pair**: author supplies forward + write-back callables (flexible; inverse-correctness on author).

**RESOLVED (JMT riff 2026-06-05) — callable vs DSL is NOT two ways to do one job; it's two capability tiers,
forced by CORRECTNESS:**
- A raw opaque callable is FUNDAMENTALLY read-only. We can't safely apply it to `op.grad` (correct only for
  pure-selection transforms; we can't introspect a lambda to verify) -> auto-grad through a callable = silently
  WRONG gradients = exactly the silent-wrong-data the validation tripwire forbids. And it can't be inverted for
  write-back. So callable => read-only, structured spec => read+grad+intervene. Not substitutes.
- **Decision: ONE authoring path = an anchored spec object** (normal `__getitem__` slicing + ~5 torch-shaped
  helpers: `.heads(n,d)`, `.split(n,dim)`, `.reshape`, `.transpose`, `.select`). Reads like light torch (fx/
  einops/nnsight-proxy pattern); near-zero learning; self-documenting. A raw callable is the read-only escape hatch.
- **The user never CHOOSES a mode.** Capability = whatever the transform allows: structural -> full power;
  non-structural -> read-only (flagged). The DSL is just the spec object's methods; stepping outside them lands
  you in read-only by construction. Power of "both", simplicity of one path, zero decisions.
- **Elegant invariant: the spec's expressiveness boundary == the safety boundary.** The spec can express
  exactly the structural (selection-Jacobian) transforms — precisely the ones where grad+intervene are valid.
  So an unsafe full-featured facet is UNAUTHORABLE; no silent-wrong-grad footgun.
- Cost: a recording spec is real but bounded engineering (~6-8 structural ops). It's the single foundation that
  makes read/grad/intervene one coherent system. 99% of users never author a recipe (built-ins cover supported
  models); this authoring cost lands on us + rare new-arch contributors, NOT facet users.
Facets that DON'T anchor to a single real op (a pure computation of several tensors not itself captured)
gracefully DEGRADE to read-only (compute+display, no grad/intervene). Almost everything wanted (heads, KQV,
neurons, named outputs, residual stream) IS a real op output + a slice, so the flagship features cover them.

Result: ONE mental model — `record.facets[name]` (and `.facets.head(i)`) — covering raw multi-outputs, semantic heads/KQV, and neurons. Three syntaxes collapse to one.

## 3. The three integration pillars (the actual sprint value)

### Pillar A — facets on ops, modules, AND grad_fns (universal + semantic)
- Keep recipe-based semantic facets (q/k/v/pattern/neurons), expand coverage, ensure they work on ANY model (fallback: module-path access when no recipe classifies a component — nnsight-style universality as the floor, semantic facets as the win).
- Add **residual-stream + per-head-output facets** reconstructed from the captured W_O/op graph (TL `hook_result` parity) WITHOUT a memory-blowup flag.
- grad_fns: a facet that maps to a tensor slice can also project onto the backward graph (Pillar C).

### Pillar B — facet-level INTERVENTION (today: impossible)
- New selector(s): `tl.facet("q")`, `tl.head(i)` (and `tl.facet("q").head(i)`), resolving to a **slice-aware hook on the parent op/module** (the low-friction path the intervention audit confirmed). Ablating/patching a head edits just that slice and reconstructs the parent tensor.
- One-call ergonomics consistent with existing `attach_hooks`/`zero_ablate`/`fork`/`rerun`: `edited.attach_hooks(tl.head(3), tl.zero_ablate())`, or sugar `module.facets.head(3).ablate()`.
- (Stretch) a prebuilt **patching** helper returning `[layer, head]`-shaped results (TL `patching` parity).

### Pillar C — facet-level GRADIENTS (the flagship; beats both rivals)
- A facet IS a slice/reshape of an activation; its gradient is the SAME slice of the parent op's saved gradient. **Lazy, cheap, no capture-time blowup** (compute on access from already-saved gradients; requires `backward_ready`/grad capture).
- TWO complementary lazy sources (JMT 2026-06-05):
  - **`op.grad`** (primary): `module.facets.head(3).q.grad` = slice the recorded `op.grad` by the facet's provenance spec. The usual output-side gradient.
  - **paired `grad_fn`** (where `GradFn.has_op`): `GradFnCall.grad_inputs`/`grad_outputs` — `grad_output` ~ `op.grad` (output side), but the grad_fn ALSO gives the **input-side** gradient (w.r.t. the op's inputs) `op.grad` lacks, plus a fallback when `op.grad` wasn't saved but the backward hook fired. Same slice spec applied to the right tuple element (trivial for single-output ops; index bookkeeping for multi-in/out).
- This is the unique combo: per-head + per-head-gradient + universal + semantic — offered by neither TL nor nnsight.
- **Facet gradient EDITING (in-flight grad steering, nnsight-style) is DEFERRED to the backward-pass megasprint** — read grads now, edit grads later. Facet *activation* intervention stays in Pillar B (this umbrella).
- (Stretch) attribution-patching helper (`grad * (clean-corrupted)` per head) using these facet grads.

## 3c. Attention fusion strategy (fused-kernel internals) [LOCKED 2026-06-05]

Fused attention (`F.scaled_dot_product_attention` / FlashAttention) computes `softmax(QK^T*scale+mask)@V`
in ONE kernel; the **`pattern` (post-softmax) / `scores` (pre-softmax) / `z`** intermediates are NEVER
materialized as tensors. Q/K/V and the attn `output` ARE visible (eager projections). So fusion limits us
ONLY on pattern/scores/z.

Key asset: TorchLens captured the fused op's INPUTS (Q,K,V,mask,scale as upstream saved activations).
- **READ (default): validated reconstruction.** Recompute `scores=Q@Kᵀ*scale+mask`, `pattern=softmax(scores)`,
  `z=pattern@V` from the fused op's OWN recorded inputs (exact tensors the kernel used — no pre/post-RoPE
  ambiguity), then **replay-validate** (`pattern@V (+ output proj) == captured attn output`). Serve as
  "reconstructed (validated)", read-only tier (it's a nonlinear recompute, outside the model's autograd graph).
- **GRAD / INTERVENE require a REAL tensor -> eager.** A fused-internal facet has no tensor to hook, no
  backward node to read a grad from. The principled (NOT finnicky) lever: **eager attention materializes them
  as real ops**, after which normal facet machinery works (slice-aware hook on the real `pattern` op; slice of
  its `op.grad`). Mental model: *look at patterns -> free (reconstruction); edit/differentiate -> capture
  `attention="eager"`*.
  - **Per-module eager (LOCKED granularity):** force eager only on the attention block(s) actually being
    edited (don't pay the decomposed-kernel cost model-wide). `attach_hooks(tl.head(3).pattern, ...)` may
    AUTO-trigger eager for just that module on the rerun.
  - HF: flip `attn_implementation="eager"` for the pass, restore after. Custom `F.sdpa` models: a TorchLens
    un-fuse shim (Python decomposition) for the pass. (Forcing torch `SDPBackend.MATH` does NOT help — still
    one Python-level call; the lever must be at the module/Python level.)
- **Honest `MissingFacet`** with an actionable message when neither applies (never silent None).
- **Module-recompute shim** (intercept module, recompute pattern->output eagerly WITH the edit, substitute the
  module output) is a LATER optimization (needs the recipe to supply the forward continuation pattern->@V->W_O),
  NOT v1. v1 = "eager mode for the modules you edit".

Honest design statement: a fused-internal facet is **read-only (validated reconstruction)** until you give it a
real tensor via eager capture, at which point it's **full read/grad/intervene** — and eager scopes to the
modules you touch. We do NOT reimplement models (TL's approach) — that forfeits universality + faithfulness;
reconstruction gets TL's visibility from the REAL model with a replay guarantee, and eager-only-when-editing
beats TL paying the eager cost on every pass and nnsight needing source-tracing.

## 4. Extensibility + defaults

- Keep the global recipe registry (zero-config defaults, like TL), expand built-in coverage (more attention/MLP/norm/embedding families; SSM/Mamba; MoE; conv blocks).
- Recipes derive everything from the captured graph + module class — so adding an architecture = one small recipe fn (`@tl.facets.register(class_name=...)`).
- Defaults cadence: ship core recipes IN-REPO (versioned + tested, updated each release) and keep the registry clean enough that a community/separate `torchlens-recipes` package can add more. (Avoid auto-fetching recipes at runtime — reproducibility + security.)

## 5. GENUINE FORKS (decide these)

**Fork A — registration scope. [LOCKED 2026-06-05]**
Layered model (zero-config global convenience + reproducible-per-trace + safe permanent extension):
- **Built-in defaults:** always on, zero-config.
- **`@tl.facets.register(...)`:** ADDITIVE to the process registry. Non-matching recipes are INERT (match by
  class name), so extra recipes do no harm -> NO clean-slate toggle (dropped `use_default_recipes`). Name/class
  collision -> USER WINS (override built-ins; sub-decision b).
- **Permanent/durable registration:** entry-point plugin package (`torchlens.recipes` group; installed = always
  loaded, reproducible, like pytest plugins / matplotlib backend packages). Recipes are CODE not data, so NO
  auto-exec'd `~/.torchlens/recipes/` dir by default; an opt-in local-dir auto-load is gated behind an explicit
  setting (off by default) with a reproducibility warning.
- **Per-trace / per-context overrides:** `tl.trace(..., recipes=[...])` (ADDITIVE; sub-decision a) and
  `with tl.facets.using(r): ...` (contextvar-backed) for isolation/experiments.
- **Reproducibility:** each `Trace` SNAPSHOTS the active recipe set at capture time (immune to later global
  mutation; lazy facets stay stable), and every facet records WHICH recipe produced it (provenance,
  `view.recipe_source`).
- **Conflict resolution by SPECIFICITY** (qualname > class_name > predicate; user > built-in), NOT registration
  order. Warn only on true same-tier ambiguity. (Defangs current last-wins fragility.)
- **Hygiene:** `tl.facets.reset()` (back to built-ins), `tl.facets.list()`, `tl.facets.info()`.

**Fork B — unify facets + outs, or keep orthogonal?**
(a) keep separate (status quo; two addressing schemes); (b) **unify at access layer** — outs surfaced as default facets, `.facets` is the one door (my §2 proposal); (c) full merge (collapse the capture too — NOT recommended; structural vs derived are genuinely different to capture).
REC: **(b)** — one user-facing door, distinct capture underneath. This is the direct answer to "can't facets just be one thing?"

**Fork C — facet intervention site model.**
(a) slice-aware hook on the parent op (low friction, no graph change); (b) first-class virtual facet NODES in the graph (heavy; rolls into the deferred Edge/Op-subclass discussions);
REC: **(a)** — a facet is a slice, not a node; hook the parent and edit the slice. Matches the intervention audit's low-friction recommendation.

**Fork D — per-facet gradients: lazy-slice vs capture-time.**
(a) **lazy**: slice `op.grad` on access via the facet's provenance (cheap, no extra memory, needs grad capture on); (b) capture-time per-facet grad saving (expensive, must pre-declare facets).
REC: **(a)** — lazy slicing of the already-saved op gradient. Zero added capture cost (consistent with the edge-counts philosophy you liked).

**Fork E — scope/phasing: how much in sprint 1?**
The full vision (universal+semantic facets + intervention + gradients + patching helpers + expanded recipes) is big. Phase options:
- **P1 (foundation):** unify outs->facets (Fork B), facet provenance spec, per-facet gradients (Pillar C, lazy). Ships the flagship grad capability + the unification.
- **P2:** facet-level intervention selectors + slice-aware hooks (Pillar B).
- **P3:** residual-stream/per-head-output facets + expanded architecture recipes + module-path universal fallback (Pillar A completion).
- **P4 (stretch):** prebuilt patching/attribution-patching helpers (TL `patching` parity).
REC: P1 first (highest value, self-contained, proves the model), then P2, then P3/P4. Each is its own validated mini-sprint.

**Fork F — naming: semantic role-names vs module-path.**
(a) semantic facet names only (q/k/v/pattern/residual_stream — TL-style); (b) module-path only (nnsight-style); (c) **both** — semantic names primary, module-path as universal fallback for unclassified components.
REC: **(c)** — semantic where a recipe classifies, module-path everywhere else, so you're never worse than nnsight and usually as good as TL.

**Fork G — grad_fns as a first-class facet surface?**
You asked "should we consider grad_fns too?" Two readings: (i) per-facet gradients (Pillar C — yes, via op.grad slicing, no grad_fn changes needed); (ii) facets ON the backward graph itself (a GradFn-level facet view).
REC: do (i) now (covers "gradients associated with heads"); defer (ii) unless a concrete need appears — the op.grad slice already gives per-head gradients without touching GradFn internals.

## 5b. Documentation deliverable (REQUIRED — JMT-flagged 2026-06-05)

This is a big, multi-concept surface (recipes, the (home op, structural transform) spec model, registration
scopes, read/grad/intervene as three views, extending to new architectures). It MUST ship with solid docs,
not just glossary entries. Required:
- A standalone **`docs/facets.md`** guide (narrative, like `docs/buffers.md` / `docs/intervention_api.md`):
  what a facet is; `record.facets[...]` access; named vs positional outputs; the spec/derivation model;
  read vs grad vs intervene; per-head/KQV examples; how to read gradients (`facet.grad`); how to ablate/patch
  a head; the limitations (read-only degradation for non-anchored facets).
- A **"writing a recipe" / extending-to-new-architectures** section: the spec object + tiny helper vocabulary
  (`__getitem__`, `.heads`, `.split`, `.reshape`, `.transpose`), the read-only callable escape hatch, and the
  registration scopes (global `@register`, per-trace `recipes=`, context `using()`, entry-point plugin for
  permanent install).
- A **TransformerLens/nnsight migration cheat-sheet** (their syntax -> TorchLens facet syntax) for the
  interp crowd's muscle memory.
- Glossary + CLAUDE/AGENTS + example notebooks lockstep (per the LOCKED keep-glossary-in-lockstep principle).
- Each phase's PR updates the docs in the SAME change (a phase that ships code but leaves docs stale is INCOMPLETE).

## 6. Open questions for JMT
- Q1: Accept Fork B (unify outs under facets)? It changes `outs` from a peer concept to a facet provider (mild deprecation-shim on `module.outs`).
- Q2: Phase order — P1 (gradients+unify) first, as recommended? Or do you want intervention (P2) first?
- Q3: How aggressive on recipe coverage in this sprint vs a follow-on "recipe expansion" pass?
- Q4: Patching-helpers module (Fork E P4) — in scope for the umbrella, or a separate sprint?
- Q5: Should the semantic facet NAMES adopt TL's vocabulary where it exists (`hook_pattern`->`pattern`, `resid_pre`->`residual`) for muscle-memory parity, or keep TorchLens-native names?
