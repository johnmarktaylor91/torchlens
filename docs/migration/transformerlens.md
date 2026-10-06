# TransformerLens to TorchLens v2 Migration

| TransformerLens operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Cache activations with `run_with_cache` | `log = tl.trace(model, x, save=tl.in_module("block"))` | TorchLens records a broader PyTorch op DAG, not only transformer hook points. |
| Inspect hook names | `log.find_sites(tl.contains("..."))` or `log.summary()` | Equivalent discovery workflow, different names. |
| Target a hook point by TLens name | `torchlens.mechinterp.resolve_alias(log, "blocks.5.attn.hook_pattern")` (also tuple and compact `"k6"` forms), or discover once and use `tl.label(label)` | TLens spellings resolve against the trace; exact labels remain the reproducible form. |
| `run_with_hooks` | `tl.trace(model, x, intervene=tl.when(site, hook), save=site)` | Capture-time `intervene=` is the direct live execution equivalent when the site is known. |
| Patch from clean cache into corrupted run | Capture clean and corrupted logs, then `corrupted.fork().set(site, clean_site.out).push()` | Equivalent for graph-stable patching. |
| `act_patch` / patching heatmaps | `torchlens.mechinterp.patch_heads_grid` / `patch_residual_grid` (turnkey `PatchGrid` results); gradient-based attribution via `torchlens.attribution` | Turnkey grids ship; validated against TransformerLens oracles. |
| ActivationCache analysis verbs (accumulated residual, decomposition, head stacking, logit attributions) | `torchlens.mechinterp.residual_accumulation` / `residual_decomposition` / `direct_logit_contributions` / `attention_head_contributions` / `head_scores` / `test_prompt` | The analysis kit follows the ActivationCache design, with credit; the LayerNorm linearization is validated bit-exactly against TLens's cached-scale convention. |
| Activation ablation helpers | `tl.zero_ablate`, `tl.mean_ablate`, `torchlens.intervention.scramble_elements` (the honest rename of the removed `resample_ablate`) | Built-in helpers cover common cases. |
| Residual stream steering | `tl.steer(direction, magnitude=...)` at the discovered site | Equivalent if the relevant residual op is visible. |
| Attention-head Q/K/V internals under fused SDPA | Attention facets (`scores`/`pattern`/`z`/`result` menus, head-sliceable views) read head-level semantics with provenance disclosed (captured vs reconstructed); `torchlens.mechinterp.lowered_counterfactual` patches one head on a fused model via exact, implementation-independent lowering | Head-level addressing works on fused models; raw fused kernel intermediates are still not op-DAG sites, and the facet provenance says which basis you got. |
| Accumulate many prompt variants | `tl.bundle({...})` plus `metric`, `joint_metric`, `node` | Equivalent comparison container; v1.x TorchLens had TraceBundle, v2 uses Bundle. |
| Transformer pictures (attention heads, logit lens, token strips) | `torchlens.tviz` (`attention_views`, `render_attention`, `prediction_trajectory`, causal receipt grids) | Ships; CircuitsVis bridge included. |

Every `torchlens.mechinterp` and facet spelling above is DOCUMENTED-UNSTABLE
pending the naming session; the workflows are stable, the exact names may be
ratified differently.

## The honest boundary, updated

Earlier versions of this page conceded that TorchLens lacked a
transformer-native component vocabulary and turnkey patching helpers. That
concession is stale: `torchlens.mechinterp` ships the ActivationCache-style
analysis kit (validated against TransformerLens oracles), TLens-name alias
resolution, and turnkey patch grids, and attention facets give head-level
addressing even under fused SDPA.

What remains true: TransformerLens is still more ergonomic when your entire
workflow lives inside its supported model families and hook-point vocabulary
-- everything is pre-named, and its community corpus (ARENA notebooks,
tutorials) speaks those names natively. Choose TorchLens when you need the
real model (stock HF checkpoints, any architecture), the broader PyTorch op
DAG, verified capture, or architecture-agnostic tooling; choose
TransformerLens when its curated model families and pedagogy corpus are the
point. See `from_transformerlens.md` for the 3.x TransformerBridge refusal
(the bridge mutates the HF model it wraps and cannot be instrumented).
