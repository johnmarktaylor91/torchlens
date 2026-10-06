# RepE to TorchLens v2 Migration

| RepE operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Apply a representation direction | `tl.steer(direction, magnitude=...)` | Equivalent for visible activation sites. |
| Remove a direction | `tl.project_off(direction)` | Equivalent helper. |
| Project onto a direction | `tl.project_onto(direction)` | Equivalent helper. |
| Choose layer/token positions | Discover the activation site, then use tensor-shaped directions or custom hooks | Equivalent when shape is visible; no named token abstraction. |
| Run controlled generations | Capture/rerun the generation loop under hooks | Partial; high-level generation wrappers are external. |
| Compare control strengths | Fork logs or use `Bundle.joint_metric` across steered variants | Equivalent analysis pattern. |
| Save steering recipe | `.tlspec/` portable save when direction tensors and built-ins are used | Equivalent publication path. |
| Train/read representation directions | `tl.bridge.repeng.control_vector(log_pos, "model.layers.<i>", negative_log=log_neg, layer=i)` returns a real `repeng.ControlVector` from saved last-token activations (`tl.bridge.dialz.vector` and `tl.bridge.steering_vectors.vector` do the same for dialz and steering-vectors) | Equivalent: bit-identical to `ControlVector.train` with `batch_size=1` (trace with `use_cache=False`; the last layer reads `"model.norm"`). |
| Batch evaluation | `rerun(..., append=True)` when append constraints hold | Equivalent for compatible chunks. |
| Intervene inside fused attention | Manual unfused attention | Opaque fused internals are hidden. |
