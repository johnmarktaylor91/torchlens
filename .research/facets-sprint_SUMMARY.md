# facets-sprint — SUMMARY (2026-06-05)

**Result: SHIPPED to local main (P1-P4 + demo notebook). Not pushed (82 commits ahead of origin); no version bump.**
Design: `.research/facets-proposal.md`. Plan (v2, dual-lab-hardened): `.research/facets-sprint-plan.md`.

## What shipped
A structured semantic-facets system: `facet = (home op, structural transform chain)`, with read / grad / intervene
as three views of one spec. Universal (any nn.Module) + semantic (TransformerLens-style names) + per-component
gradients — the combination neither TransformerLens nor nnsight offers.

Commits (local main, c9ef746..de29cb7):
- P1 `77b63c2`/`7df3213`/`79581fc` + glossary `c9ef746`: FacetSpec ABI, transform capability classes
  (bijective_view/selection/aliasing_selection/computed, fail-closed), op-anchored READ + GRAD, `MissingGradient`
  contract, trace-owned immutable registry snapshot + specificity ordering + `recipes=`/`using()`/`reset()`,
  structural outputs as facets (named/positional/typed-item-access), recipe migration + capability inventory.
- P2 `9221e30`: facet INTERVENTION — scatter-back write-back reusing the existing hook machinery, `tl.facet`/
  `tl.head` selectors, shared-c_attn composition + conflict detection, GQA/computed/in-place fail-closed.
- P3 `445e20a`: fused-attention RECONSTRUCTION (pattern/scores/z) anchored to real post-RoPE SDPA inputs +
  non-vacuous replay-validation; residual `resid_pre/mid/post` + per-head `result`; module-path universal fallback;
  `enable_transformerlens_aliases()`; entry-point plugin loading; recipe expansion. `tl.trace(reconstruction_ready=True)`.
- P4 `d4d37e6`: patching helpers `tl.facets.patching.*` (activation-patch residual/attn/per-head/mlp ->
  [layer,head]; attribution-patch via facet grads) — TransformerLens `patching` parity.
- Notebook `3db2d05`/`de29cb7`: `notebooks/facets_tutorial.ipynb` (10 feature areas; executes clean end-to-end).

## Process (per JMT directive: iterate with adversarial Claude+Codex till satisfied, then build per phase)
- STAGE 1: dual-lab adversarial PLAN review (Claude Opus + Codex), 2 rounds. Round 1 found convergent blockers
  (grads/args not default; LSTM+container already shipped; recipes-run-not-record; registry snapshot; RoPE anchor;
  GQA/shared-home write-back; drop paired-grad_fn). Plan revised v2 (11 reconciled decisions, capability classes,
  de-risked P1). Round 2: BOTH labs SATISFIED, 0 blockers. This caught real holes BEFORE building.
- STAGES 2-5: each phase = codex build -> INDEPENDENT review (ran gates myself, hand-checked behavior, confirmed
  tripwire intact + non-vacuous tests) -> merge to local main -> branch swept -> iMessage.
- STAGE 6: demo notebook, executed end-to-end independently (exit 0, 12/12 cells).

## Verification (independent, not trusting codex)
- Tripwire SACRED + intact: P1/P3/P4 touched NO validation/invariant files. P2's only validation change is a NARROW
  carve-out (skip func-replay of genuinely `intervention_replaced` ops) — provably INERT on plain capture (0 flagged
  ops), downstream still validated, aligned with the 2026-06-02 narrow-exemption discipline.
- Flagship grad contract verified: default trace -> `MissingGradient`; grad-captured -> correct sliced tensor.
- Intervention non-vacuous: head ablation changes output AND edited trace validates.
- Reconstruction honest: validates when correct, FAILS on corruption (non-vacuous), `MissingFacet` names prereq.
- Patching non-vacuous: important head large effect; attribution matches activation by sign+argmax+allclose.
- Final gates: full facets suite 48 passed; smoke 223; not-slow 2378 (P4); ruff+mypy clean.

## Residuals (documented, by design / fast-follow)
1. **Auto-scoped-eager-on-edit NOT wired** (P3): editing a fused-internal `pattern` fail-closes + names the eager
   prerequisite. Capability EXISTS via manual eager capture (`attn_implementation="eager"`) + P2; the auto-trigger
   convenience is a fast-follow. (Honest fail-closed, not silent-wrong.)
2. Custom (non-HF) recipes must source head counts themselves (built-ins use HF `config_value`); the Module RECORD
   doesn't expose live-module attrs. Minor authoring note for `docs/facets.md`.
3. Notebook committed un-executed (no nbstripout hook; codex's choice) — runs clean on execute. Could ship with
   outputs if preferred.

## State: NOT pushed. Local main only, 82 ahead of origin. Push when JMT approves.
