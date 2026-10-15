# NNsight to TorchLens v2 Migration

| NNsight operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Trace a model invocation | `tl.trace(model, x, save=tl.func("relu"))` | TorchLens stores a completed `Trace` after execution. |
| Save a node value | Use `save=selector`, then access `log.find_sites(selector).first().out` | Equivalent for captured activations. |
| Assign into a traced value | `log.fork().set(site, value).push()` (replay the recorded graph) or `.run(model, x)` (run the model again) | TorchLens recipes are explicit and auditable. |
| Invoke with intervention in one block | `tl.trace(model, x, intervene=tl.when(site, helper_or_value), save=site)` | Capture-time intervention is the direct one-forward path. |
| Scan modules by path | `tl.in_module("path")`, `tl.module("path")`, `log.find_sites(...)` | Similar workflow; labels differ. |
| Remote execution (NDIF), vLLM serving, tensor parallelism, quantized models | No equivalent | TorchLens captures a local eager PyTorch forward; for those back ends NNsight is the tool. |
| Cross-prompt patching | Capture separate logs and patch with `set` or `attach_hooks` | Equivalent for local PyTorch models. |
| Repeated interventions from one base | `fork()` first, then mutate each branch | TorchLens emphasizes fork-first branch isolation. |
| Save intervention for sharing | `log.save_intervention(path, level="portable")` | TorchLens has `.tlspec/` recipes. |
| Intervene inside fused attention internals | Default fused SDPA load: by-head and by-pattern patching through `torchlens.mechinterp.lowered_counterfactual` (no eager reload); eager-attention load: the softmax and score ops are ordinary intervention sites | Head-level and pattern-level edits work on fused models; other edits inside the fused kernel need an eager recapture. |

## NNsight 0.8

These rows were checked against nnsight 0.8 (0.8.0rc1, a release candidate published on
2026-09-09; the latest stable release is 0.7.0). 0.8 adds vLLM and tensor-parallel
back ends, quantized models, diffusers pipelines, `scan`, and `.source`, which reaches
operations inside a module's forward that no submodule owns. TorchLens records every
executed operation by default, so it needs no `.source` step; what it adds is verified
capture (the recorded graph is replayed and checked against the real forward), saved
and diffable traces, and typed refusals. Worked examples: `from_nnsight.md`.
