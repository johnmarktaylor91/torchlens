# RepE to TorchLens v2 Migration

| RepE operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Apply a representation direction | `tl.steer(direction, magnitude=...)` | Equivalent for visible activation sites. |
| Remove a direction | `tl.project_off(direction)` | Equivalent helper. |
| Project onto a direction | `tl.project_onto(direction)` | Equivalent helper. |
| Choose layer/token positions | Discover the activation site, then use tensor-shaped directions or custom hooks | Equivalent when shape is visible; no named token abstraction. |
| Run controlled generations | Capture/rerun the generation loop under hooks | Partial; high-level generation wrappers are external. |
| Compare control strengths | Fork logs or use `Bundle.metric` across steered variants | Equivalent analysis pattern. |
| Save steering recipe | `.tlspec/` portable save when direction tensors and built-ins are used | Equivalent publication path. |
| Train/read representation directions | No direct equivalent | Deferred to v2.x or external RepE pipeline. |
| Batch evaluation | `rerun(..., append=True)` when append constraints hold | Equivalent for compatible chunks. |
| Intervene inside fused attention | Manual unfused attention | Opaque fused internals are hidden. |
