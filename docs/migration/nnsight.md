# NNsight to TorchLens v2 Migration

| NNsight operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Trace a model invocation | `tl.trace(model, x, save=tl.func("relu"))` | TorchLens stores a completed `Trace` after execution. |
| Save a node value | Use `save=selector`, then access `log.find_sites(selector).first().out` | Equivalent for captured activations. |
| Assign into a traced value | `log.fork().set(site, value).replay()` or `.rerun(model, x)` | TorchLens recipes are explicit and auditable. |
| Invoke with intervention in one block | `tl.trace(model, x, intervene=tl.when(site, helper_or_value), save=site)` | Capture-time intervention is the direct one-forward path. |
| Scan modules by path | `tl.in_module("path")`, `tl.module("path")`, `log.find_sites(...)` | Similar workflow; labels differ. |
| Remote model execution/session | No built-in equivalent | Deferred to v2.x or external orchestration. |
| Cross-prompt patching | Capture separate logs and patch with `set` or `attach_hooks` | Equivalent for local PyTorch models. |
| Repeated interventions from one base | `fork()` first, then mutate each branch | TorchLens emphasizes fork-first branch isolation. |
| Save intervention for sharing | `log.save_intervention(path, level="portable")` | TorchLens has `.tlspec/` recipes. |
| Intervene inside fused attention internals | Manual unfused attention implementation | Opaque fused kernels hide internal sites from TorchLens. |
