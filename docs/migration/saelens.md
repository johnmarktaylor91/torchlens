# Coming from SAELens

This page is for people who load published sparse autoencoders with SAELens (the registry and
loader, v6.54.14 released 2026-11-02) and want to run, edit and measure them against a model
in TorchLens. SAELens loads the SAE; TorchLens runs the model and splices it in. TorchLens
does not pin SAELens in its test environment, so the SAELens calls are described in prose and
the splice example uses a stand-in SAE that needs no package. Checked 2026-11-03.

## The same task in both

Attach an SAE at one site of a model and ablate one of its features, then read what changed.

### In SAELens

SAELens loads a published SAE and attaches it to a hook point of a TransformerLens-style
model. You load one with `SAE.from_pretrained(release, sae_id)`, for example the release
`"gemma-scope-2b-pt-res-canonical"` and the id `"layer_12/width_16k/canonical"`; then
`sae.encode(activations)` turns the residual stream at the hook point into latents and
`sae.decode(latents)` reconstructs it. The SAE downloads from the Hugging Face Hub, so this is
described here rather than run; the [pretrained-SAE recipe](../field-guide/pretrained_saes.md)
runs the same calls on an SAE built in memory.

### In TorchLens

`splice` takes any object with an `encode` and a `decode` method; no SAE package is required.
The identity SAE below lets you check the effect by hand. For a pretrained SAE, load it with
SAELens and pass it in, as the [pretrained-SAE recipe](../field-guide/pretrained_saes.md) shows.

```python
# migration-test: tool=torchlens expected=[True, True, True]
import torch
from torch import nn
import torchlens as tl
from torchlens.bridge.sae import splice

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(3, 4)


class IdentitySAE(nn.Module):
    def encode(self, h):
        return h

    def decode(self, z):
        return z


def drop_feature_0(z):
    z = z.clone()
    z[..., 0] = 0.0
    return z


plain = splice(model, x, site=tl.module("1"), sae=IdentitySAE())
edited = splice(model, x, site=tl.module("1"), sae=IdentitySAE(), latents_edit=drop_feature_0)
RESULT = [
    plain.reconstruction_mse == 0.0,
    plain.output_delta_max == 0.0,
    edited.output_delta_max > 0.0,
]
```

## Translation table

| SAELens operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Attach an SAE to a hook point | `torchlens.bridge.sae.splice(model_or_log, inputs, site=..., sae=...)` -- the packaged splice experiment (fork + `tl.splice_module` + replay); reports reconstruction fidelity (MSE, fraction of variance explained pooled over per-dimension variance as SAELens reports it, and per dimension) plus output-level causal effect, and with `tokens=` next-token CE loss recovered and KL divergence. Pretrained SAEs: the [pretrained-SAE recipe](../field-guide/pretrained_saes.md) (SAELens loads, TorchLens runs) | Ships. Duck-typed SAE (`encode`/`decode` pair); no SAE package required. Tutorial: `notebooks/sae_splice_tutorial.ipynb`. |
| Ablate / edit SAE features | `splice(..., latents_edit=fn)` -- the per-feature causal knob (a callable over the latent tensor) | Ships as part of the splice experiment. |
| Manual attach at any site | `tl.splice_module(sae_like_module)` at a discovered site | Equivalent for shape-preserving PyTorch modules. |
| Train an SAE | Bring your own optimizer over activations from `tl.extract_dataset` / `tl.trace`; the [SAE training hand-off](../sae_training_handoff.md) walks it end to end | TorchLens deliberately ships no trainers; it feeds external training machinery and then measures the trained artifact causally. |
| Feature steering | `tl.steer(direction, strength=...)` or a custom hook | Equivalent for tensor directions. |
| Read feature activations | `SpliceResult.latents` is the SAE encoder output for the clean site activation, exactly as `sae.encode` returned it, and `SpliceResult.edited_latents` the latents after `latents_edit=` (what the decoder consumed). Both are kept by default; `splice(..., keep_latents=False)` drops them, and a warning fires once per session when they pass 256 MiB | Equivalent for local analysis. |
| TransformerLens hook-name alignment | `torchlens.mechinterp.resolve_alias(log, name)`, or discover TorchLens labels and save exact labels | TLens spellings resolve against the trace. |
| SAE dashboards / feature stores | No built-in equivalent | Out of scope by design; use SAELens/Neuronpedia tooling on activations TorchLens extracts. |
| Publish intervention recipe | `save_intervention(level="portable")` when using built-ins/tensors | Portable if no opaque SAE module is required. |
| Execute opaque SAE module recipe elsewhere | `executable_with_callables` in same code environment | Not portable. |
| Fused attention internals | Attention facets (`pattern`/`scores`/`z`/`result`) with disclosed provenance; `torchlens.mechinterp.patch_head` for head-level patching on fused models | Head-level semantics are addressable; raw fused kernel intermediates are still not op-DAG sites. |

`bridge.sae.splice` and the facet spellings are DOCUMENTED-UNSTABLE pending
the naming session. The boundary statement, kept honest: TorchLens measures
and intervenes on SAEs (reconstruction fidelity, causal effect, feature-level
edits) but ships no SAE training, storage, or dashboard machinery -- that
remains SAELens's home turf, credited in `docs/acknowledgments.md`.

## What TorchLens adds

- A packaged splice experiment, `torchlens.bridge.sae.splice`: the SAE's reconstruction goes
  in at the site, and the result reports reconstruction fidelity (MSE and fraction of variance
  explained, per dimension too) and output-level causal effect, with next-token loss recovered
  and KL divergence when you pass `tokens=`.
- Feature-level edits through `latents_edit=`, a callable over the latent tensor.
- Every executed op as a site, not only the hook points a wrapper names: `tl.trace(model, x)`
  and the selectors `tl.module`, `tl.in_module`.
- TransformerLens-style names resolved against the trace of a stock Hugging Face model
  (`torchlens.mechinterp.resolve_alias`).
- Activations written with provenance for training or probing outside TorchLens:
  `tl.extract_dataset`, and the [SAE training hand-off](../sae_training_handoff.md).

## What SAELens does that TorchLens does not

- SAELens keeps the registry of published SAEs and loads them (`SAE.from_pretrained`).
  TorchLens ships no SAE loader.
- SAELens trains SAEs. TorchLens deliberately ships no trainer; it feeds external training
  machinery and then measures the trained artifact causally.
- SAE dashboards and feature stores (including Neuronpedia's) are out of scope by design; use
  SAELens or Neuronpedia tooling on activations TorchLens extracts.
- For other tools, see [comparable tools](landscape.md) and
  [where to go instead](where_to_go_instead.md).

## Using both

### SAELens feeds TorchLens

Load the SAE with SAELens and hand it to `torchlens.bridge.sae.splice` (it needs `encode` and
`decode`), as the [pretrained-SAE recipe](../field-guide/pretrained_saes.md) walks through. A
second route, `torchlens.bridge.sae_lens`, has `encode(trace, site, sae)` and
`reconstruct(trace, site, sae)` for reading a SAELens SAE's latents or reconstruction at a
site of a trace.

### TorchLens feeds SAELens

Extracted activations are plain tensors and arrays: `tl.extract_dataset` writes them, and
the [SAE training hand-off](../sae_training_handoff.md) walks one end to end into a trainer.
`SpliceResult.latents` holds the encoder output for a site, which you can hand to any
dashboard code. There is no TorchLens-to-SAELens object conversion beyond that.

## Semantic differences

- **Duck-typed SAE.** `splice` needs only `encode` and `decode`; SAELens's `SAE` object
  qualifies, but nothing in the splice reads SAELens-specific configuration (hook name,
  normalization settings). Match the site to the one the SAE was trained on yourself, or use
  the [pretrained-SAE recipe](../field-guide/pretrained_saes.md).
- **Latents are kept.** `SpliceResult.latents` and `SpliceResult.edited_latents` are kept by
  default and can be large; see the "Read feature activations" row.
- **Stable-spelling status.** `bridge.sae.splice` and the facet spellings are
  documented-unstable.
- **Discovery versus monitoring.** SAEs are now used mainly as a discovery tool (finding
  candidate features), while monitoring a deployed model is done with trained linear
  probes on captured activations. TorchLens serves both by extracting activations
  (`tl.trace`, `tl.extract_dataset`); it ships no probe trainer, so fit probes with
  scikit-learn or your own optimizer on the extracted arrays.
