# Facets sprint — build plan v2 (post dual-lab review round 1)

Design north-star: `.research/facets-proposal.md`. Round-1 findings:
`.research/facets_plan_review_{claude,codex}.md`. This v2 reconciles BOTH (they converged).

## Round-1 reconciled DECISIONS (both labs)
1. **facet.grad is a CAPABILITY, not default.** `save_gradients=False`/`backward_ready=False` by default ->
   `op.grad is None`. facet.grad available ONLY when the home op has a saved gradient (grad capture on, or a
   post-hoc `log_backward()` with the home raw-out still saved). Else return a typed **`MissingGradient`** with
   the exact recapture instruction. Never advertise default availability.
2. **DROP paired-grad_fn input-side gradients from P1.** Capture discards the `next_functions` input index
   (`backward.py:138-140`), so the mapping can't be built reliably. facet.grad = `transform(op.grad)` ONLY
   (output-side, the home activation's gradient). Paired-grad_fn -> deferred to the backward megasprint.
3. **Transform CAPABILITY CLASSES (replaces "structural = full power").** Each FacetSpec transform chain is
   classified; the system FAILS CLOSED (read-only) unless provably in a grad/write-capable class:
   - `bijective_view` (reshape/transpose/permute, exact inverse): read + grad + write.
   - `selection` (slice/select/split): read + grad + write via scatter-back + conflict policy.
   - `aliasing_selection` (GQA-repeat/broadcast/expand): read + grad ONLY (write only with an explicit alias policy).
   - `computed` (callable / multi-home / nonlinear): READ-ONLY unless anchored to a captured op.
4. **Write-back = scatter-back with conflict/composition policy.** Each primitive defines `apply(x)`,
   `project_grad(g)`, `scatter_update(home, edited, mode)`, shape/dtype/device checks, and same-home edit
   composition (GPT-2 c_attn q/k/v cumulative edits) + GQA alias semantics. P1 defines this ABI; P2 uses it.
   (Don't call lossy selection "invertible" — call it "scatter-back capable.")
5. **DROP the LSTM-fix + container-name P1 tasks (already shipped).** Replace with: VERIFY + lock regression
   coverage, and EXPOSE existing `ContainerSpec` names through `.facets` with a reversible naming scheme +
   collision policy (the genuinely-missing piece).
6. **Registration = trace-owned IMMUTABLE snapshot.** Build `facet_registry_snapshot` (version + provenance);
   `FacetView` reads ONLY the snapshot; `using()`/`recipes=` affect capture-time snapshot construction, not
   lazy post-capture access; specificity ordering (qualname>class_name>predicate; user>built-in) + diagnostics.
7. **Attention reconstruction MOVED to P3** (it has the most prerequisites: a "reconstruction-ready" capture
   mode + anchor-to-actual-SDPA-input). Not in P1.
8. **Capability inventory of every built-in facet** (op_structural / parameter / module_input / module_output /
   computed_read_only / missing). Parameter facets (norm gamma/beta, embedding weight) = `parameter` home
   (read-only via param; not op.grad). Computed (MLP `intermediate`) = anchor to the captured multiply op or
   degrade read-only. Fail closed: only `op_structural` may claim grad/write.
9. **FacetSpec carries value-VERSION** (raw out / `out_versions_by_child`); in-place/view-derived facets marked
   NOT intervention-safe until replay-validated.
10. **Naming: canonical TYPED item access** (`facets["out1.0"]`, path objects for dict-key/dotted/colliding
    names); attribute access (`facets.q`) is best-effort sugar for valid identifiers not colliding with
    FacetView methods (`keys`/`get`/`head`/`recipe_source`).
11. **FacetSpec PORTABILITY:** structural chains are portable (home label + primitive chain + capability flags +
    recipe id/version); raw callables NEVER serialized as executable (non-exec provenance; defined missing-on-load).

## STATUS: dual-lab review SATISFIED (both labs, round 2) — PLAN LOCKED 2026-06-05
Round-2 verdicts: `.research/facets_confirm_{codex,claude}.md` — both SATISFIED, P1 buildable, zero remaining
blockers. Folded round-2 polish:
- `facet.grad` RETURNS a `MissingGradient` sentinel (does NOT raise); `get`/`in`/iteration uniform across
  `MissingFacet` + `MissingGradient`; only tensor-USE of the sentinel raises (with recapture instruction). Pin in P1b + tests.
- P1c capability inventory records each recipe's actual anchored-vs-read-only outcome AS DATA (non-vacuous gate),
  e.g. MLP `intermediate` -> anchored-to-multiply-op or read-only.
- Multi-pass home op: `module.facets` on a multi-CALL module must raise/select a pass, never silently return
  pass 0 (covered by the LSTM/recurrent test row).

## Global constraints (all phases)
- Stay 2.x.x (minor bump per phase, NO major). No backward-compat (facets unused). Validation tripwire SACRED.
- Per-phase: branch -> validated -> independent review -> merge to LOCAL main. NO stacked PRs. No AI attribution.
- Each phase ships tests + `docs/facets.md` + glossary IN THE SAME change. Demo notebook at the end.

## Core ABI: FacetSpec (built FIRST in P1, before any recipe migration)
Fields: home_kind (op | module_output | module_input | parameter | computed); home_label/address + pass/call
index; output_path (for structural multi-outputs); transform primitive chain (each with pre/post shape
assertions); capability_class (per #3) -> capability flags (read/grad/write/portable/reconstructed);
value_version (#9); conflict/alias group (#4); recipe id + version (#11).
Primitives: `__getitem__`(slice/index), `.heads(n,d)`, `.split(n,dim)`, `.reshape`, `.transpose`, `.select`.
Each declares its capability flags {read, grad, write}. **Chain capability = the INTERSECTION of its primitives'
flags** (precise "weakest link"): `computed` anywhere -> read-only; `aliasing_selection` anywhere -> drops
`write`; `write` requires ALL primitives write-capable (bijective_view exact-inverse or selection scatter-back);
`read` always. Read-only callable = `computed` escape hatch.
**`MissingGradient` surface (Codex polish):** `facet.grad` RETURNS a `MissingGradient` sentinel (parallel to
`MissingFacet`); using it as a tensor RAISES an informative error with the exact recapture instruction
(`backward_ready=True` + `gradients_to_save`). Consistent with the existing `MissingFacet`-raises-on-access pattern.

## Revised phasing (de-risked per both labs' "reduce P1 scope")

### P1 — Foundation (spec ABI + registry + read/grad), NO reconstruction, NO intervention, NO paired-grad_fn
- P1a: trace-owned immutable registry snapshot + specificity ordering + provenance + reset/list/info; expose
  existing structural-output names via `.facets` (reversible naming + collision policy). VERIFY + lock the LSTM/
  container regression (do NOT rewrite container capture).
- P1b: FacetSpec ABI + transform primitives with capability classes; op-anchored READ + GRAD only; the
  `MissingGradient` capability contract (#1).
- P1c: migrate built-in recipes to FacetSpec + the capability inventory (#8), fail-closed.
- GATES: every stress model `validate_forward_pass` green; facet read correct; facet.grad matches a manual
  slice WHEN grad-captured + typed-missing otherwise (test matrix below); ruff/mypy/smoke/not-slow; `docs/facets.md`
  (model + read + grad capability) + glossary for all new public names.

### P2 — Intervention (scatter-back write-back)
- Scatter-back per primitive (#4) + conflict/composition policy (shared home) + GQA alias policy; `tl.facet`/
  `tl.head` selectors -> slice-aware home-op write-back; fork/rerun/attach_hooks integration; in-place/view
  not-intervention-safe guard (#9).
- GATES: edited rerun produces expected change AND `validate_forward_pass` holds on the edited trace; conflicting
  same-home writes detected; docs intervention section; glossary.

### P3 — Attention reconstruction + coverage + residual/head-output + fallback + aliases
- "Reconstruction-ready" capture mode (require fused-op args: Q/K/V parents [default] + mask/scale/is_causal/
  dropout via `save_arg_values`, + RNG for nonzero dropout). Anchor Q/K to the ACTUAL SDPA inputs (post-RoPE,
  found via the graph), not projection outputs. Per-facet validation TARGET (#10: `z` vs SDPA op output;
  `pattern/scores` via recompute-z; `attn_out/result` vs proj). Per-dtype tolerances (fp32 softmax upcast).
  GQA expansion convention. `MissingFacet` names the missing prerequisite.
- Recipe expansion (broad but bounded: HF attention/MLP/norm/embedding variants, MoE, ViT, Mamba/SSM if
  structure cooperates); residual-stream (`resid_pre/mid/post`) + per-head output (`result`) from W_O/op graph;
  module-path universal fallback; opt-in TL alias layer; entry-point plugin permanence.
- GATES: reconstruction replay-validated (non-vacuous, correct target); facets on a spread of real models;
  docs recipe-authoring + TL/nnsight migration cheat-sheet; glossary.

### P4 — Patching / attribution helpers
- Activation-patching (residual/attn-out/per-head, clean-vs-corrupted) -> `[layer,head]` results; attribution-
  patching (`grad*(clean-corrupted)`) using facet grads. GATES: validated vs hand-computed small cases; docs.

## P1 non-negotiable test matrix (Codex D)
- default trace: `facet.grad` is typed-missing with instructions; `backward_ready=True`+`log_backward`: facet
  grad == manual slice; selective `gradients_to_save`: missing on unselected home; GPT-2 fused QKV q/k/v share
  one home (read works; conflicting write specs detected — P2); GQA K/V alias semantics explicit; LSTM/GRU/RNN
  multi-output roles/names/indices/`num_passes`/`recurrent_ops`; NamedTuple/dataclass/HF-ModelOutput/dict-key
  collisions; (P3) SDPA reconstruction with mask/`is_causal`/non-default scale/dropout-zero/dropout-nonzero/
  bf16-fp16 tolerance.

## Risks / watch-items
- FacetSpec recording proxy + capability classes = the load-bearing NEW engineering; build + test first.
- Registry snapshot must be trace-owned/immutable or facet names aren't reproducible (#6).
- In-place/view home-op version ambiguity (#9) — carry value_version; gate intervention.
- Reconstruction (P3) has real prereqs (arg capture, post-RoPE anchor, validation target, dtype/dropout) — keep it OUT of P1.

## Dispatch model
Each phase = a codex build (`codex-bg.sh` + Monitor), validation-gated, independent review before merge to local
main. Sequential: P2/P3/P4 depend on the P1 ABI. P1 itself sequenced P1a->P1b->P1c.
