# Facets sprint plan v2 — adversarial CONFIRM (Claude, round 2)

Scope: verify round-1 BLOCKING findings are resolved in `facets-sprint-plan.md` v2, and that the
de-risked P1 (P1a/P1b/P1c) is unambiguously buildable. Spot-checked code against v2's claims.

## VERDICT: SATISFIED (P1 buildable)

Every round-1 BLOCKING finding is resolved by an explicit v2 decision, each confirmed against
current code. The de-risked P1 is internally consistent and implementable. Two non-blocking
spec tightenings remain (see Residual notes); neither stops the build.

### B1 (grad not saved by default) — RESOLVED.
Decision #1 + P1b make `facet.grad` a CAPABILITY gated on a saved home-op gradient, with a typed
`MissingGradient` carrying the recapture instruction when absent. Code confirms the premise was
real and the fix is net-new, not vapor: `options.py:626 save_gradients=False`, `:640 backward_ready=False`;
`semantic/facets.py` has only `MissingFacet` (line 56), no `MissingGradient` yet. The non-vacuous
grad gate ("grad == manual slice WHEN grad-captured + typed-missing otherwise", P1 test matrix)
directly answers my "trivially vacuous gate" objection.

### B2 (reconstruction args not saved) — RESOLVED by deferral.
Decision #7 moves attention reconstruction (and its `save_arg_values`/scale/mask/RNG prerequisites)
entirely to P3, behind an explicit "reconstruction-ready" capture mode. P1 ships NO reconstruction.
Confirmed `options.py:625 save_arg_values=False`. The plan no longer implies free validated
reconstruction on a default trace. Correct.

### B3 (RoPE anchor) — RESOLVED by deferral + explicit post-RoPE anchor target.
P3 ("Anchor Q/K to the ACTUAL SDPA inputs (post-RoPE, found via the graph), not projection
outputs") names the exact fix and per-facet validation target. P1 ships no reconstruction, so the
pre-RoPE anchoring in the shipped recipes (confirmed: `recipes/attention.py:171-173` source
`q_proj`/`k_proj`/`v_proj` child outputs, pre-RoPE) is read-only structural extraction only — no
correctness claim to violate. The "no pre/post-RoPE ambiguity" overclaim is gone from the P1 path.

### B4 (GQA / shared-home write-back) — RESOLVED at the ABI level.
Decisions #3+#4 introduce capability classes and a scatter-back ABI: `aliasing_selection`
(GQA-repeat/expand) is read+grad ONLY, write only under an explicit alias policy; `selection`
writes via scatter-back + conflict policy; same-home (GPT-2 c_attn q/k/v) cumulative edits compose.
Critically, all WRITE semantics are deferred to P2 — P1 only DEFINES the ABI and ships read+grad.
Confirmed the read-only many-to-one slice still lives at `semantic/facets.py:118-131`
(`head_index // group_size`), and GPT-2 fuses q/k/v into one `c_attn` home (`recipes/attention.py:140-144`).
The "undefined inverse = silent-wrong edit" tripwire risk is contained: fail-closed (decision #3,
#8) means an unclassified/aliasing transform cannot silently write. This is the right shape.

### B5/B6 (already-shipped tasks) — RESOLVED by reframing, not deletion.
Decision #5 drops the rewrite framing and replaces both with VERIFY + lock regression coverage,
plus the genuinely-open work: expose existing `ContainerSpec` names through `.facets` with a
reversible naming + collision scheme (decision #10), and add structseq coverage (test matrix).
Explicitly does NOT rewrite container capture or touch loop detection without a red repro. Matches
my B5/B6 resolutions.

Bonus confirmations beyond the five blockers:
- **M1 (paired grad_fn index alignment)** is resolved structurally: decision #2 DROPS input-side
  paired-grad_fn gradients from P1. Confirmed the premise is real — `backbackends/torch/backward.py:138`
  iterates `next_functions` but discards `_input_num`, so the input-index map genuinely can't be
  rebuilt. `facet.grad = transform(op.grad)` output-side only is the sound subset.
- **M5 (snapshot vs lazy)** is resolved by decision #6 (trace-owned immutable snapshot; `FacetView`
  reads only the snapshot). Confirmed the gap is real: today `_REGISTRY` is a module global
  (`facets.py:258`), `_matching_recipes` reads it live (`:393-396`), `_compute` is last-wins+warn
  (`:247-253`). P1a's specificity ordering + snapshot is correctly scoped net-new work.

## Remaining blockers (if any) — each with a concrete resolution

NONE. No finding rises to "P1 cannot be built." The points below are spec tightenings, not blockers.

## Residual notes (non-blocking)

1. **R5 raise-vs-return for `MissingGradient`/`MissingFacet` is still not pinned for the widened
   surface.** Code today is inconsistent: `FacetView.__getitem__` RAISES on a `MissingFacet`
   (`facets.py:216-217`) but `get()` only swallows `KeyError`, not the raised `RuntimeError`
   (`:198-201`) — so `view.get("pattern")` on a fused model raises instead of returning the default.
   v2 introduces a SECOND sentinel (`MissingGradient`) and a wider auto-facet surface, multiplying the
   surfaces this inconsistency touches. Decision #10 fixes typed item access/naming but does not state
   whether `facet.grad` on a no-grad trace RETURNS a `MissingGradient` sentinel or RAISES, nor make
   `get()`/membership/iteration uniform. Non-blocking for build, but P1b should add one sentence:
   "`facet.grad` returns a truthy `MissingGradient` sentinel (never raises on access); `get()`/`in`/
   iteration treat both sentinels as present-but-unavailable." Pin it in the test matrix so P1 locks
   the contract rather than discovering it in P2.

2. **`computed` MLP-intermediate degradation is asserted but the anchor rule is one level shy of
   mechanical.** Decision #8 says computed facets (MLP `intermediate`) "anchor to the captured
   multiply op or degrade read-only." Whether a given activation-fn'd intermediate HAS a single
   captured op to anchor to is recipe-by-recipe (e.g. a GELU applied inside the recipe body is not an
   op the recipe currently references). This is correctly fail-closed (worst case = read-only, no
   silent-wrong grad), so it's non-blocking — but P1c's capability inventory should record the actual
   per-recipe outcome (anchored vs read-only) as data, not leave it to build-time discovery, so the
   inventory gate is non-vacuous.

3. **Multi-pass / recurrent home op (my R6) is acknowledged in the ABI (decision #9 / FacetSpec
   "pass/call index") but the P1 default behavior is unstated.** Recipes today hardcode `calls[0]`
   (first pass). The ABI carrying a pass index is sufficient for P1; just ensure `module.facets` on a
   multi-call module either selects a pass explicitly or raises like the existing
   `_single_pass_or_error` pattern rather than silently returning pass 0. Belongs in the P1 test
   matrix's LSTM/recurrent row, which already exists — so this is covered, just confirm the assertion
   is "raises/selects", not "returns pass 0".

None of these block P1. The capability-class taxonomy (bijective_view / selection / aliasing_selection
/ computed) is tight enough to implement read+grad unambiguously, fail-closed by construction, with
all write semantics correctly held back to P2 and all reconstruction to P3.
