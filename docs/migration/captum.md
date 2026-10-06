# Captum to TorchLens v2 Migration

| Captum operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Forward activation capture | `tl.trace(model, x, save=tl.func("relu"))` | TorchLens captures graph metadata and selected activations in one pass. |
| Layer attribution target | Discover with `tl.module`, `tl.in_module`, or `tl.func`, then use exact labels | Similar target-selection step. |
| Feature ablation | `tl.trace(..., intervene=tl.when(site, tl.zero_ablate()), save=site)` plus metrics | Equivalent building blocks, not identical attribution API. |
| Occlusion-style perturbation | `tl.attribution.occlusion_map(model, x, target=..., window=...)` for input occlusion; `tl.attribution.occlusion(trace, selection, score=...)` over captured sites | Shipped. |
| Integrated gradients | `tl.attribution.integrated_gradients(model, x, target=...)`; `tl.attribution.layer_integrated_gradients` for a layer | Shipped; see `docs/reference/attribution.md` for baselines and convergence reporting. |
| Saliency / gradient attribution | `tl.attribution.saliency`, `input_x_grad`, `smoothgrad`, `noise_tunnel`, `gradient_shap`, `guided_backprop`, `deconvolution`, `grad_cam` | Shipped; DeepLift, LIME and Kernel SHAP are not shipped, use Captum for them. |
| Neuron conductance | `tl.attribution.layer_conductance` (layer level) | Layer conductance only; neuron-level conductance stays in Captum. |
| Compare attribution metrics across runs | `Bundle.joint_metric` or an explicit metric loop | Equivalent container-level computation. |
| Persist attribution setup | `.tlspec/` for intervention recipe, separate code for metric | Partial; metrics are not fully serialized. |
| Fused attention internals | Manual unfused implementation | TorchLens cannot see hidden fused intermediates. |
