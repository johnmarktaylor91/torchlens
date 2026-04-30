# baukit to TorchLens v2 Migration

| baukit operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| `Trace` a module output | `tl.log_forward_pass(...); log.find_sites(tl.module("module.path"))` | Similar for module boundaries. |
| `TraceDict` over many layers | `log.find_sites(tl.in_module(...), max_fanout=...)` and iterate results | Equivalent discovery, different object model. |
| Edit an activation in a trace | `log.fork().set(site, value).replay()` | Equivalent post-hoc mutation for stable graphs. |
| Retain input/output tensors | TorchLens records activations and selected argument metadata in `LayerPassLog` | TorchLens is broader but heavier. |
| Stop or replace module output inline | Use live `hooks=...` during capture or `rerun` hooks | Similar for visible module/op outputs. |
| Patch generated-token traces | Capture the generation loop and replay/rerun target sites | Equivalent when the Python loop is visible. |
| Lightweight context manager only | No exact equivalent | TorchLens creates a persistent `ModelLog`; lightweight tracing is deferred to v2.x. |
| Causal tracing utilities | Build with `Bundle`, helpers, and metrics | Turnkey causal-tracing dashboards are deferred to v2.x. |
| Save edited trace recipe | `save_intervention(..., level=...)` | TorchLens has first-class recipe persistence. |
| Fused attention internals | Manual unfused implementation | Not visible inside opaque kernels. |
