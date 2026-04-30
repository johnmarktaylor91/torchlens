# TransformerLens to TorchLens v2 Migration

| TransformerLens operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Cache activations with `run_with_cache` | `log = tl.log_forward_pass(model, x, intervention_ready=True)` | TorchLens records a broader PyTorch op DAG, not only transformer hook points. |
| Inspect hook names | `log.find_sites(tl.contains("..."))` or `log.summary()` | Equivalent discovery workflow, different names. |
| Target a hook point by name | Discover once, then use `tl.label(label)` | TorchLens labels are graph-derived; exact labels are preferred for reproducibility. |
| `run_with_hooks` | `log.fork().attach_hooks(site, hook).rerun(model, x)` | Rerun is the closest live execution equivalent. |
| Patch from clean cache into corrupted run | Capture clean and corrupted logs, then `corrupted.fork().set(site, clean_site.activation).replay()` | Equivalent for graph-stable patching. |
| Activation ablation helpers | `tl.zero_ablate`, `tl.mean_ablate`, `tl.resample_ablate` | Built-in helpers cover common cases. |
| Residual stream steering | `tl.steer(direction, magnitude=...)` at the discovered site | Equivalent if the relevant residual op is visible. |
| Attention-head Q/K/V internals | Use a manual attention implementation with visible PyTorch ops | Fused SDPA/FlashAttention internals are not visible. |
| Accumulate many prompt variants | `tl.bundle({...})` plus `metric`, `joint_metric`, `node` | Equivalent comparison container; v1.x TorchLens had TraceBundle, v2 uses Bundle. |
| Built-in transformer component naming | No exact equivalent | Deferred to v2.x naming polish; use discovery and exact labels today. |
