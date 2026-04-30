# Inseq to TorchLens v2 Migration

| Inseq operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attribute generation outputs | Capture generation-style Python loop and compute metrics over `ModelLog`/`Bundle` | Partial; Inseq attribution methods are not reimplemented. |
| Step-wise hidden-state access | `log.find_sites(...)` over repeated loop sites | Equivalent when generation loop is visible. |
| Contrastive inputs | Capture clean/corrupted logs and compare with `Bundle` | Equivalent comparison pattern. |
| Attribution method selection | No direct equivalent | Deferred to v2.x; use Inseq for built-in attribution algorithms. |
| Intervention during generation | Live `hooks=...` or `rerun(model, x)` with sticky hooks | Equivalent for local PyTorch generation code. |
| Token-position patching | Tensor-shaped replacements or custom hooks | Equivalent if target tensors expose position dimensions. |
| Save analysis setup | `.tlspec/` saves intervention recipe; metric code remains external | Partial. |
| Visual attribution reports | TorchLens graph visualization, not token attribution reports | Different output. |
| Fused attention internals | Manual unfused implementation | Hidden internals are not visible. |
| HuggingFace convenience wrappers | No direct public wrapper | Deferred to v2.x. |
