# Penzai to TorchLens v2 Migration

| Penzai operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Name and select model subtrees | `tl.in_module`, `tl.module`, and exact labels after discovery | Similar goal; TorchLens uses PyTorch module addresses and op labels. |
| Functional model surgery | `fork`, `set`, `attach_hooks`, `replay`, `rerun` | TorchLens mutates logs, not model definitions. |
| Inspect intermediate values | `log.find_sites(...).first().activation` | Equivalent for captured tensors. |
| Patch or replace a value | `set(site, tensor)` or `attach_hooks(site, helper)` | Equivalent for visible activations. |
| Declarative named axes | Use explicit tensor shapes and `feature_axis` where helpers require it | No named-axis system; deferred to v2.x. |
| Transformer-specific layer abstractions | Discover PyTorch module/op sites | No built-in Penzai transformer abstraction parity. |
| Save a modified computation | `.tlspec/` saves intervention recipes | TorchLens saves recipes, not rewritten model trees. |
| Compare several variants | `tl.bundle({...})` | Equivalent comparison container. |
| Intervene on fused kernel internals | Manual unfused PyTorch code | Opaque fused internals are hidden. |
| Pure JAX workflows | No equivalent | TorchLens is PyTorch-only. |
