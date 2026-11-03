# Coming from Captum

This page is for people who compute attributions or layer activations with Captum and want the
TorchLens equivalents. Captum ships many attribution methods; TorchLens ships a subset under
`tl.attribution`. The examples were written against the Captum release installed in the weekly
bridge environment and the TorchLens version on this branch. Checked 2026-11-03.

TorchLens's own methods are saliency, input x gradient, integrated gradients, SmoothGrad, noise
tunnel, gradient SHAP, guided backprop, deconvolution, occlusion, Grad-CAM, layer attribution,
layer conductance, layer integrated gradients, per-token text attribution, and the infidelity and
sensitivity metrics; see [the attribution reference](../reference/attribution.md).

## The same task in both

The task: read the activation of one layer for one input.

### In Captum

Needs captum; the fence is skipped where it is not installed.

```python
# migration-test: tool=captum expected=[[2.5, 2.5]]
import torch
from torch import nn
from captum.attr import LayerActivation


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


model = Tiny()
x = torch.tensor([[2.0, 3.0]])
RESULT = LayerActivation(model, model.proj).attribute(x).detach().tolist()
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=[[2.5, 2.5]]
import torch
from torch import nn
import torchlens as tl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


log = tl.trace(Tiny(), torch.tensor([[2.0, 3.0]]))
RESULT = log["linear_1_1"].out.detach().tolist()
```

## Translation table

| Captum construct | TorchLens equivalent | Parity |
| --- | --- | --- |
| `LayerActivation(model, layer).attribute(x)` | `tl.trace(model, x)` and read the layer activation | Equivalent for captured activations. |
| Forward activation capture | `tl.trace(model, x, save=tl.func("relu"))` | TorchLens captures graph metadata and selected activations in one pass. |
| Layer attribution target | Discover with `tl.module`, `tl.in_module`, or `tl.func`, then use exact labels | Similar target-selection step. |
| Feature ablation | `tl.trace(..., intervene=tl.when(site, tl.zero_ablate()), save=site)` plus metrics | Equivalent building blocks, not identical attribution API. |
| Occlusion-style perturbation | `tl.attribution.occlusion_map(model, x, target=..., window=...)` for input occlusion; `tl.attribution.occlusion(trace, selection, score=...)` over captured sites | Shipped. |
| Integrated gradients | `tl.attribution.integrated_gradients(model, x, target=...)`; `tl.attribution.layer_integrated_gradients` for a layer | Shipped; see `docs/reference/attribution.md` for baselines and convergence reporting. |
| Saliency / gradient attribution | `tl.attribution.saliency`, `input_x_grad`, `smoothgrad`, `noise_tunnel`, `gradient_shap`, `guided_backprop`, `deconvolution`, `grad_cam` | Shipped; DeepLift, LIME and Kernel SHAP are not shipped, use Captum for them. |
| Neuron conductance | `tl.attribution.layer_conductance` (layer level) | Layer conductance only; neuron-level conductance stays in Captum. |
| Compare attribution metrics across runs | A metric called on the `tl.Bundle`, or `bundle.map(fn)` per member | Equivalent container-level computation. |
| Persist attribution setup | `.tlspec/` for intervention recipe, separate code for metric | Partial; metrics are not fully serialized. |
| Fused attention internals | Eager-attention load: the softmax and score ops are captured ops you read directly. Default fused SDPA load: `tl.trace(model, x, reconstruction_ready=True)` serves read-only reconstructed attention facets (`scores`, `pattern`, `z`, checked against the fused output; see `docs/facets.md`) | Head-level semantics are readable on fused models; the raw kernel intermediates are not captured ops, and editing them needs an eager recapture. |

### SHAP

SHAP is a composition partner, not something TorchLens replaces. For deep networks, the
native path is `tl.attribution.gradient_shap(model, inputs, target=..., baselines=...)`:
pass `seed=` and the run is reproducible: the result records the seed and digests of its
baseline and sample draws, and `target=` can be a class index or any function from the
model's output to a scalar.
For SHAP's own explainers, `torchlens.bridge.shap.explain(trace, background=...)` is a
thin handoff (the `shap` extra): it passes the traced model and inputs to
`shap.DeepExplainer` (or the `explainer_class` you name) and returns SHAP's values with
the explainer.

## What TorchLens adds

- One capture serves attribution and inspection: the same `Trace` holds the graph, the saved
  activations and the attribution inputs, and `tl.attribution` methods take the model and the input.
- Reproducibility records: `gradient_shap(..., seed=...)` records the seed and digests of its
  baseline and sample draws; `integrated_gradients` reports baselines and convergence (see the
  attribution reference).
- Capture-time interventions (`tl.when(site, tl.zero_ablate())`) and replay checks next to the
  attribution call, and `tl.Bundle` for comparing metrics across runs.

## What Captum does that TorchLens does not

Keep Captum when you need a method TorchLens does not ship: DeepLift, LIME, Kernel SHAP, feature
ablation as an attribution method, neuron-level conductance, its tuned defaults, or its
paper-validated implementation details. Captum is also where TCAV, influence functions and
adversarial attacks live, and Captum and inseq are the tools for text DeepLift and LIME; see
[inseq.md](inseq.md) and [landscape.md](landscape.md).

## Using both

### Captum feeds TorchLens

Captum results are ordinary tensors; there is no bridge from a Captum attribution into a
`Trace`. Compute with Captum and compare the result with a TorchLens `tl.attribution` method on the
same model and input.

### TorchLens feeds Captum

`torchlens.bridge.captum` hands a trace to Captum: `tl.bridge.captum.attribute(trace, method,
target=..., inputs=...)` forwards the trace's saved inputs to a Captum method object, and
`tl.bridge.captum.layer(trace, site)` resolves a TorchLens site to the live module a Captum layer
method expects. It needs the `captum` extra, and a trace that saved its inputs (the bridge's own test
captures with `save="all"`).

```python
# migration-test: tool=captum expected=True
import torch
from torch import nn
from captum.attr import Saliency

import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
x = torch.randn(2, 4)
trace = tl.trace(model, x, save="all")
bridged = tl.bridge.captum.attribute(trace, Saliency(model), target=0)
direct = Saliency(model).attribute(x, target=0)
RESULT = bool(torch.equal(bridged, direct))
trace.cleanup()
```

## Semantic differences

- Captum targets a layer by passing a module object; TorchLens targets a site by selector or
  label, and a layer that runs more than once needs a pass-qualified label (the bridge refuses a
  module that runs more than once).
- A TorchLens attribution method is selected by name from `tl.attribution` and returns a result
  object that records its settings; Captum returns a bare tensor.
- Feature ablation in TorchLens is an intervention (`tl.zero_ablate`) plus your own metric:
  equivalent building blocks, not an identical attribution API.
- Persisting an attribution setup: `.tlspec/` stores the intervention recipe, and the metric code
  is separate (partial; metrics are not fully serialized).
