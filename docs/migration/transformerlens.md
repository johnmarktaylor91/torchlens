# TransformerLens to TorchLens v2 Migration

| TransformerLens operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Cache activations with `run_with_cache` | `log = tl.trace(model, x, save=tl.in_module("block"))` | TorchLens records a broader PyTorch op DAG, not only transformer hook points. |
| Inspect hook names | `log.find_sites(tl.contains("..."))` or `log.summary()` | Equivalent discovery workflow, different names. |
| Target a hook point by name | Discover once, then use `tl.label(label)` | TorchLens labels are graph-derived; exact labels are preferred for reproducibility. |
| `run_with_hooks` | `tl.trace(model, x, intervene=tl.when(site, hook), save=site)` | Capture-time `intervene=` is the direct live execution equivalent when the site is known. |
| Patch from clean cache into corrupted run | Capture clean and corrupted logs, then `corrupted.fork().set(site, clean_site.out).replay()` | Equivalent for graph-stable patching. |
| `act_patch` attribution patching | Use `tl.bwd_hook(...)` during a corrupted `rerun`, then compute `grad * (clean - corrupt)` for the selected site. | Building blocks are present; turnkey heatmap helpers are deferred. |
| Activation ablation helpers | `tl.zero_ablate`, `tl.mean_ablate`, `tl.resample_ablate` | Built-in helpers cover common cases. |
| Residual stream steering | `tl.steer(direction, magnitude=...)` at the discovered site | Equivalent if the relevant residual op is visible. |
| Attention-head Q/K/V internals | Use a manual attention implementation with visible PyTorch ops | Fused SDPA/FlashAttention internals are not visible. |
| Accumulate many prompt variants | `tl.bundle({...})` plus `metric`, `joint_metric`, `node` | Equivalent comparison container; v1.x TorchLens had TraceBundle, v2 uses Bundle. |
| Built-in transformer component naming | No exact equivalent | Deferred to v2.x naming polish; use discovery and exact labels today. |

## Honest concession

TransformerLens remains more ergonomic when your workflow already thinks in names such as
`blocks.5.attn.hook_pattern`. TorchLens can discover and target Q/K/V-like PyTorch sites, and exact
labels are reproducible after discovery, but it does not ship TransformerLens's transformer-native
component vocabulary. Choose TorchLens when you need the broader PyTorch op DAG or architecture-
agnostic capture; choose TransformerLens when its model family and hook vocabulary are the point.
