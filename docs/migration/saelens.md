# SAELens to TorchLens v2 Migration

| SAELens operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attach an SAE to a hook point | `tl.splice_module(sae_like_module)` at a discovered site | Equivalent for shape-preserving PyTorch modules. |
| Encode/decode activation features | Implement the SAE as an `nn.Module` or hook callable | TorchLens does not ship SAE training/inference utilities. |
| Ablate SAE features | Hook callable or `splice_module` that edits reconstructed activations | Equivalent building block. |
| Feature steering | `tl.steer(direction, magnitude=...)` or custom hook | Equivalent for tensor directions. |
| Read feature activations | Hook side-channel through `hook.run_ctx` | Equivalent for local analysis. |
| TransformerLens hook-name alignment | Discover TorchLens labels, then save exact labels | Different naming system. |
| SAE dashboards / feature stores | No built-in equivalent | Deferred to v2.x or external SAELens tooling. |
| Publish intervention recipe | `save_intervention(level="portable")` when using built-ins/tensors | Portable if no opaque SAE module is required. |
| Execute opaque SAE module recipe elsewhere | `executable_with_callables` in same code environment | Not portable. |
| Fused attention internals | Manual unfused attention | Hidden internals are not TorchLens sites. |
