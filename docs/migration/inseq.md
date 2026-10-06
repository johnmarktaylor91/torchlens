# Inseq to TorchLens v2 Migration

| Inseq operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attribute generation outputs | `tl.attribution.text(model, tokenizer, text, target=...)` attributes one target token to the prompt tokens (integrated gradients over input embeddings); a whole generation run can be captured with episode capture (`episode=tl.options.EpisodeSpec(...)`, documented-unstable) | Partial; per-step attribution across every generated token, with step scores, stays in Inseq. |
| Step-wise hidden-state access | `log.find_sites(...)` over repeated loop sites | Equivalent when generation loop is visible. |
| Contrastive inputs | Capture clean/corrupted logs and compare with `Bundle` | Equivalent comparison pattern. |
| Attribution method selection | `tl.attribution` ships saliency, input x gradient, integrated gradients, SmoothGrad, gradient SHAP, occlusion, layer attribution methods and more; `tl.attribution.text` uses integrated gradients | Partial; Inseq's generation-aware method catalog (including DeepLift, LIME and attention-weight methods) stays in Inseq. |
| Intervention during generation | Live `hooks=...` or `rerun(model, x)` with sticky hooks | Equivalent for local PyTorch generation code. |
| Token-position patching | Tensor-shaped replacements or custom hooks | Equivalent if target tensors expose position dimensions. |
| Save analysis setup | `.tlspec/` saves intervention recipe; metric code remains external | Partial. |
| Visual attribution reports | The `tl.attribution.text` result renders per-token scores (`.show()`, `.to_html()`, `.to_text()`) | Per-token report for one target; no per-step generation heatmap. |
| Fused attention internals | Manual unfused implementation | Hidden internals are not visible. |
| HuggingFace convenience wrappers | `torchlens.bridge.hf.trace_text(model, text)` traces a Hugging Face language model from raw text; `tl.attribution.text` takes the model and its tokenizer | Shipped for capture and single-target token attribution. |

## Honest concession

Inseq is the better fit when the deliverable is per-step attribution across a generated sequence,
with its step scores, method catalog and generation heatmaps. TorchLens attributes a single target
token to the prompt (`tl.attribution.text`), ships general attribution methods under
`tl.attribution`, captures visible PyTorch generation loops, compares clean and corrupted runs, and
exposes gradients and activations for custom analysis.
