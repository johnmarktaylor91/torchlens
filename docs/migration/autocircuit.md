# AutoCircuit to TorchLens v2 Migration

| AutoCircuit operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Identify circuit nodes | `log.find_sites(...)`, graph metadata, and visualization | TorchLens exposes raw graph sites; automated circuit search is separate. |
| Patch edges/nodes | `set`, `attach_hooks`, `replay`, and `rerun` at selected sites | Node patching equivalent; edge-specific APIs are deferred. |
| Run path patching sweeps | Fork logs or loop over selectors with `Bundle.metric` | Manual but supported. |
| Compare clean/corrupted datasets | Capture logs and group them with `Bundle` | Equivalent container workflow. |
| Automatic circuit discovery | No direct equivalent | Deferred to v2.x or external AutoCircuit tooling. |
| Prune/score circuit components | Use explicit metrics over intervention variants | Partial; no built-in pruning optimizer. |
| Save circuit intervention | `save_intervention(level="portable")` for supported recipes | Equivalent for recipe publication. |
| Visualize graph with intervention marks | `log.show(..., vis_intervention_mode="node_mark")` | Equivalent graph-level visibility. |
| Append evaluation batches | `rerun(..., append=True)` when graph/hash constraints match | Equivalent for compatible chunks. |
| Fused attention internals | Manual unfused implementation | Hidden internals are not visible. |
