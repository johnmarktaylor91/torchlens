# Pyvene to TorchLens v2 Migration

| Pyvene operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Define intervention locations | Use selectors such as `tl.label`, `tl.func`, `tl.module`, `tl.in_module` | Equivalent targeting concept with TorchLens site labels. |
| Run a configured intervention | `log.attach_hooks(...).rerun(model, x)` or `tl.do(...)` | Equivalent for local PyTorch execution. |
| Representation swap/patch | `log.set(site, source_activation).replay()` | Equivalent for graph-stable activation patching. |
| Zero/mean ablations | `tl.zero_ablate()`, `tl.mean_ablate(...)` | Equivalent common helpers. |
| Directional intervention | `tl.steer(direction, magnitude=...)` | Equivalent tensor steering helper. |
| Multi-source causal mediation templates | Compose captures, `Bundle`, and explicit metrics | Higher-level templates are deferred to v2.x. |
| Interchange interventions over token positions | Use tensor-shaped replacements or `tl.steer` with position axes | Equivalent when the target activation shape is visible. |
| Persist experiment config | `.tlspec/` via `save_intervention` | TorchLens persists recipes, not full experiment notebooks. |
| Dataset-level batched evaluation | `rerun(..., append=True)` when constraints hold | Equivalent for graph-compatible chunks. |
| Model-family-specific abstractions | No exact equivalent | Deferred to v2.x; TorchLens stays PyTorch-op first. |
