# SAELens to TorchLens v2 Migration

| SAELens operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attach an SAE to a hook point | `torchlens.bridge.sae.splice(model_or_log, inputs, site=..., sae=...)` -- the packaged splice experiment (fork + `tl.splice_module` + replay); reports reconstruction fidelity plus output-level causal effect | Ships. Duck-typed SAE (`encode`/`decode` pair); no SAE package required. Tutorial: `notebooks/sae_splice_tutorial.ipynb`. |
| Ablate / edit SAE features | `splice(..., latents_edit=fn)` -- the per-feature causal knob (a callable over the latent tensor) | Ships as part of the splice experiment. |
| Manual attach at any site | `tl.splice_module(sae_like_module)` at a discovered site | Equivalent for shape-preserving PyTorch modules. |
| Train an SAE | Bring your own optimizer over activations from `tl.extract_dataset` / `tl.trace` | TorchLens deliberately ships no trainers; it feeds external training machinery and then measures the trained artifact causally. |
| Feature steering | `tl.steer(direction, magnitude=...)` or a custom hook | Equivalent for tensor directions. |
| Read feature activations | `SpliceResult` carries the latents; or a hook side-channel through `hook.run_ctx` | Equivalent for local analysis. |
| TransformerLens hook-name alignment | `torchlens.mechinterp.resolve_alias(log, name)`, or discover TorchLens labels and save exact labels | TLens spellings resolve against the trace. |
| SAE dashboards / feature stores | No built-in equivalent | Out of scope by design; use SAELens/Neuronpedia tooling on activations TorchLens extracts. |
| Publish intervention recipe | `save_intervention(level="portable")` when using built-ins/tensors | Portable if no opaque SAE module is required. |
| Execute opaque SAE module recipe elsewhere | `executable_with_callables` in same code environment | Not portable. |
| Fused attention internals | Attention facets (`pattern`/`scores`/`z`/`result`) with disclosed provenance; `torchlens.mechinterp.lowered_counterfactual` for head-level patching on fused models | Head-level semantics are addressable; raw fused kernel intermediates are still not op-DAG sites. |

`bridge.sae.splice` and the facet spellings are DOCUMENTED-UNSTABLE pending
the naming session. The boundary statement, kept honest: TorchLens measures
and intervenes on SAEs (reconstruction fidelity, causal effect, feature-level
edits) but ships no SAE training, storage, or dashboard machinery -- that
remains SAELens's home turf, credited in `docs/acknowledgments.md`.
