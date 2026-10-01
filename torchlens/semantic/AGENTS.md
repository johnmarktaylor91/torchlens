# semantic/ - Implementation Guide

## __init__.py
- Re-exports the facet registry surface from `facets.py` plus the `patching` and
  `recipes` submodules; `__all__` is the public facet vocabulary.

## facets.py
- Registry API: `register()` (decorator, `class_name=`/`predicate=` matching),
  `reset()`, `using()` (contextvar-scoped extra recipes), `list()`, `info()`,
  `snapshot()`, `mark_current_registry_as_builtins()`.
- View machinery: `FacetView` (a `Mapping` built per record), `Facet`,
  `AttentionHeadView`, `FacetSpec`, `TransformPrimitive`, `FacetRecipe`,
  `FacetMenuItem`, `FacetRegistrySnapshot`, `FacetCapabilityFlags`.
- Absence is typed, never silent: `MissingFacet`, `MissingFacetError`,
  `MissingGradient`, `AbsenceReason`.
- TransformerLens-style names: `enable_transformerlens_aliases()` /
  `transformer_lens_aliases_enabled()` toggle `_TRANSFORMERLENS_ALIAS_TO_NATIVE`.
- Registry state is module-global (`_REGISTRY`, `_REGISTRY_VERSION`); `using()`
  layers recipes through the `_CONTEXT_RECIPES` contextvar.

## logit_lens.py
- `logit_lens()` projects per-block residual facets through the model's own
  final norm + unembedding; generic math over the `language_model_head` facet
  vocabulary (architecture knowledge stays in recipes). The reconstructed lens
  is VALIDATED against the captured last-block `resid_post` -> logits pair and
  refuses (`LogitLensError`) on mismatch; `lens=` supplies a user lens,
  `validate=False` is the explicit opt-out. Results: `LogitLensResult` /
  `LogitLensEntry` (`stacked()`, `top_tokens()`, `summary()`). All spellings
  DOCUMENTED-UNSTABLE.

## coverage.py
- `facet_coverage(trace)` -> `FacetCoverageReport` (`ModuleCoverageRow` per
  module): recipes matched, readable facets, typed absences, structural-only
  candidates, and disclosed `unresolved` rows for multi-call facet refusals.
  Input to `tools/facet_maintenance/` (proposals only; recipes are NEVER
  auto-merged). DOCUMENTED-UNSTABLE spellings.

## patching.py
- Prebuilt counterfactual helpers: `activation_patch_residual_stream()`,
  `activation_patch_attention_output()`, `activation_patch_attention_heads()`,
  `activation_patch_mlp_output()`, `attribution_patch_attention_heads()`.
- Built on `trace()` plus the `facet` selector from `intervention/selectors.py`;
  `_CounterfactualStateGuard` snapshots/restores RNG and stateful tensors so
  clean/corrupted runs stay comparable.

## reconstruction.py
- Fused-SDPA facet reconstruction: `SDPAReconstruction`,
  `sdpa_reconstruction_spec()`, `find_sdpa_op()`.
- Reconstruction is checked, not trusted: `_reconstruct_checked()` recomputes and
  `_allclose_sdpa()` compares against the captured output; failure yields
  `MissingFacet`, never a wrong tensor.

## _hops.py (internal; DOCUMENTED-UNSTABLE)
- The hop rule (mikit D3/D4): a closed op-kind GRAMMAR proposes
  value-transporting hops (identity/view kinds, dtype casts, parseable
  basic-index subsets, inert eval-mode dropout), and endpoint PAYLOAD IDENTITY
  (bitwise, replaying recorded casts/index subsets) verifies before any tensor
  claim rides the walk. `walk_upstream(trace, start, is_anchor, grammar=,
  structure_only=)` returns `HopWalk` (pass-qualified labels + site keys on
  every `HopRecord`, plus the `IndexMapRecord` position disclosure) or a
  reason-bearing `HopRefusal`. `BRANCH_GRAMMAR` (dropout regardless of
  training mode) is ONLY for structural branch identification where the
  returned tensor never crosses the hops. A shape-changing op never hops
  silently: it parses into the index map or the walk stops at it, named.

## recipes/ subpackage
- Builtin recipes live in `attention.py`, `embedding.py`, `lm_head.py`,
  `mlp.py`, `norm.py`, `residual.py`; each function is registered with a
  `@register(...)` decorator at import time (e.g. `gpt2_attention`,
  `gated_mlp`, `layer_norm`, `transformer_residuals`, `language_model_head`).
- Attention FUSEDNESS is decided from the CAPTURED GRAPH (a real SDPA op ->
  fused reconstructions; a real matmul->softmax->matmul score path -> eager
  op-anchored scores/pattern/z + a validated computed per-head `result`),
  corroborated by the captured config's `attn_implementation` for the
  no-graph-evidence case (external kernels teach both remedies). Class names
  route records to extraction bodies ONLY -- never fusedness (the 4.x-name
  fused gate reported wrong menus on transformers 5.x). Head geometry falls
  back to `custom_attributes["config"]` (modern HF modules keep head counts
  only on `self.config`). A produced facet value beats an equal-tier absence
  in the merge, and multi-output modules resolve `attn_out` to the PRIMARY
  container leaf via per-op `multi_output_name` (never positional pairing
  with `output_paths`). Served per-head layouts: `scores`/`pattern`
  `[b, head, dst, src]`, `z` `[b, head, pos, d_head]`, `result`
  `[b, pos, head, d_model]` (pre-bias, refuses on sum-check mismatch);
  `AttentionHeadView` slices each accordingly, mapping GQA query heads onto
  KV groups for `k`/`v`.
- `residual.py` anchors `resid_pre` by DATAFLOW + SHAPE: the unique floating
  block input whose recorded shape equals the block's single output shape and
  which feeds the block output (zero or several matches refuse typed -- never
  input-op order, which served token-ids/mask-shaped tensors on real HF
  blocks). Bare multi-pass input labels disambiguate by consumption (the pass
  with a child inside the block), never a last-pass default. `resid_mid` is
  the non-output add consuming the attention branch, found through the
  BRANCH-grammar hop walk (OPT routes attention through dropout before the
  add). Feed-forward evidence accepts the unfused `fc1`+`fc2` direct-child
  pair (OPT/whisper) alongside a wrapped MLP child.
- `lm_head.py` anchors unembedding facets (`logits`, `unembed_weight`/`_bias`,
  `final_norm_kind`/`_eps`/`_gamma`/`_beta`/`_input`, `logits_position_map`);
  the final norm is anchored by a VALUE-grammar hop walk upstream from the
  head's input op (transformers-5.x reshape + `logits_to_keep` subset), with
  tensor-bearing norm facets gated on the walk's payload-identity verification
  and the recorded index subset disclosed as the `logits_position_map`
  `IndexMapRecord`. The norm's own input resolves from OP-LEVEL parent edges
  (pass-qualified), never the call-level bare label. The broad `*RMSNorm`
  class suffix match is safe only because logit_lens numerically validates
  before trusting the reconstruction.
- `recipes/__init__.py` declares `BUILTIN_FACET_CAPABILITY_INVENTORY`, calls
  `mark_current_registry_as_builtins()`, then `_load_entrypoint_recipes()` loads
  `torchlens.recipes` entry points fail-safely (only callables flagged
  `_torchlens_recipe_autoload` are invoked; broken plugins warn, never raise).
- `_helpers.py` holds spec builders shared by recipes: `child_output_spec()`,
  `first_input_spec()`, `module_input_op_spec()`, `module_output_spec()`,
  `parameter_spec()`, `reshape_heads()`, `fused_sdpa_facet()`, `config_value()`.

## Local Invariants / Gotchas
- Dual home: top-level `torchlens/facets.py` is a self-replacing stub that swaps
  itself in `sys.modules` for `torchlens.semantic.facets`, so the two module
  objects are identity-equal; the `tl.facets` attribute resolves lazily via
  `_LAZY_ATTRS` in `torchlens/__init__.py`. Edit `semantic/facets.py` only.
- Builtin recipes register when `torchlens.semantic` is first imported (run
  time), not at `import torchlens`; tests must not assume collection-time
  registration.
- Records expose facets through the `facets` property on `Op`, `ModuleCall`,
  and `Module` (each caches one `FacetView`).
- `list` shadows the builtin in `facets.py`; internal code must use
  `builtins.list`.

## _norm_reconstruction.py (mikit item 3; DOCUMENTED-UNSTABLE)
- ONE shared `NormReconstruction` record for every norm-folding consumer
  (norm facets, `logit_lens`, mechinterp DLA): frozen per-(batch, position)
  scale from the CAPTURED norm input (bit-exact vs TLens `hook_scale` on
  real gpt2), VALIDATED against the captured norm output before anything is
  handed out (cancellation-aware magnitude: |x| + |mean| over scale -- the
  refused-correct-reconstructions class), three-way validated kind
  (`layernorm_affine` / `rmsnorm_affine` / `normalize_only` -- evidenced,
  never defaulted; LayerNormPre lands in the third). Missing eps and
  unmatched conventions (Gemma-style `(1 + weight)`) refuse typed
  (`norm_*` contract rows).
- lm_head.py classifies `LayerNormPre`/`*RMSNormPre` (TLens folded models)
  and accepts the `unembed` head child; the hop grammar's identity set
  includes `identity` ops (nn.Identity / TLens HookPoints) -- payload
  verification still gates every tensor claim.
