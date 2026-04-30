# Captum to TorchLens v2 Migration

| Captum operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Forward activation capture | `tl.log_forward_pass(model, x, intervention_ready=True)` | TorchLens captures graph metadata and activations. |
| Layer attribution target | Discover with `tl.module`, `tl.in_module`, or `tl.func`, then use exact labels | Similar target-selection step. |
| Feature ablation | `tl.zero_ablate`, `tl.mean_ablate`, `tl.resample_ablate` plus metrics | Equivalent building blocks, not identical attribution API. |
| Occlusion-style perturbation | Use `set`/helpers over chosen sites and compare outputs | Manual loop required. |
| Integrated gradients | No direct equivalent | Deferred to v2.x; use Captum for gradient-integration algorithms. |
| Saliency / gradient attribution | Backward logging plus Tier-1 backward hooks where appropriate | Partial; full Captum method parity is deferred. |
| Neuron conductance | No direct equivalent | Deferred to v2.x. |
| Compare attribution metrics across runs | `Bundle.metric` or `Bundle.joint_metric` | Equivalent container-level computation. |
| Persist attribution setup | `.tlspec/` for intervention recipe, separate code for metric | Partial; metrics are not fully serialized. |
| Fused attention internals | Manual unfused implementation | TorchLens cannot see hidden fused intermediates. |
