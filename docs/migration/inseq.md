# Inseq to TorchLens v2 Migration

| Inseq operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attribute generation outputs | `tl.attribution.text(model, tokenizer, text, target=...)` attributes one target token to the prompt tokens (integrated gradients over input embeddings); a whole generation run can be captured with episode capture (`episode=tl.options.EpisodeSpec(...)`, documented-unstable) | Partial; per-step attribution across every generated token, with step scores, stays in Inseq. |
| Step-wise hidden-state access | `log.find_sites(...)` over repeated loop sites | Equivalent when generation loop is visible. |
| Contrastive inputs | Capture clean/corrupted logs and compare with `Bundle` | Equivalent comparison pattern. |
| Attribution method selection | `tl.attribution` ships saliency, input x gradient, integrated gradients, SmoothGrad, gradient SHAP, occlusion, layer attribution methods and more; `tl.attribution.text` uses integrated gradients | Partial; Inseq's generation-aware method catalog (including DeepLift, LIME and attention-weight methods) stays in Inseq. |
| Intervention during generation | `torchlens.intervention.steer_generate(model, ids, spec)`, or `tl.when(site, action).bind(model)` for many forwards | Equivalent for local PyTorch generation code. |
| Token-position patching | Tensor-shaped replacements or custom hooks | Equivalent if target tensors expose position dimensions. |
| Save analysis setup | `.tlspec/` saves intervention recipe; metric code remains external | Partial. |
| Visual attribution reports | The `tl.attribution.text` result renders per-token scores (`.show()`, `.to_html()`, `.to_text()`) | Per-token report for one target; no per-step generation heatmap. |
| Fused attention internals | Eager-attention load: the softmax and score ops are captured ops you read directly. Default fused SDPA load: `tl.trace(model, x, reconstruction_ready=True)` serves read-only reconstructed attention facets (`scores`, `pattern`, `z`, checked against the fused output; see `docs/facets.md`) | Head-level semantics are readable on fused models; the raw kernel intermediates are not captured ops, and editing them needs an eager recapture. |
| Run inseq itself from a TorchLens session | `tl.bridge.inseq.attribute(model_or_id, text, method=..., generated_texts=...)` (extra `torchlens[inseq]`, tested on inseq 0.7.1) loads inseq's attribution model and returns inseq's own result (its `.show()` renderer works). `method` takes inseq's names or the TorchLens spelling (`input_x_grad`). Forced decoding is checked: a result that attributes any text other than `generated_texts` refuses `bridge_inseq_forced_decoding_ignored` instead of returning free-generation attribution. | Equivalent for one call; the per-step product stays in inseq. |
| HuggingFace convenience wrappers | `torchlens.bridge.transformers.trace_text(model, text)` traces a Hugging Face language model from raw text; `tl.attribution.text` takes the model and its tokenizer | Shipped for capture and single-target token attribution. |

## Honest concession

Inseq is the better fit when the deliverable is per-step attribution across a generated sequence,
with its step scores, method catalog and generation heatmaps. TorchLens attributes a single target
token to the prompt (`tl.attribution.text`), ships general attribution methods under
`tl.attribution`, captures visible PyTorch generation loops, compares clean and corrupted runs, and
exposes gradients and activations for custom analysis.
