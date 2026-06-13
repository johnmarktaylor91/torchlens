# Penzai to TorchLens v2 Migration

| Penzai operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Name and select model subtrees | `tl.in_module`, `tl.module`, and exact labels after discovery | Similar goal; TorchLens uses PyTorch module addresses and op labels. |
| Functional model surgery | `fork`, `set`, `attach_hooks`, `replay`, `rerun` | TorchLens mutates logs, not model definitions. |
| Inspect intermediate values | `log.find_sites(...).first().out` | Equivalent for captured tensors. |
| Patch or replace a value | `set(site, tensor)` or `attach_hooks(site, helper)` | Equivalent for visible activations. |
| Declarative named axes | Use explicit tensor shapes and `feature_axis` where helpers require it | No named-axis system; deferred to v2.x. |
| Transformer-specific layer abstractions | Discover PyTorch module/op sites | No built-in Penzai transformer abstraction parity. |
| Save a modified computation | `.tlspec/` saves intervention recipes | TorchLens saves recipes, not rewritten model trees. |
| Compare several variants | `tl.bundle({...})` | Equivalent comparison container. |
| Intervene on fused kernel internals | Manual unfused PyTorch code | Opaque fused internals are hidden. |
| Pure JAX workflows | Future explicit `backend="jax"` functional preview | Not shipped in this checkout; Penzai remains better for JAX model surgery. |

## Honest concession

Penzai is the better fit for functional JAX model surgery, named axes, and workflows that want the
model tree itself to be the edited artifact. TorchLens' shipped capture surface is PyTorch eager
plus a technical-preview MLX backend; the planned JAX preview captures declared functional calls
as jaxpr-derived traces. That still does not make TorchLens a JAX tree rewriting system.
