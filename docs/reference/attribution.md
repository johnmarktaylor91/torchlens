# `tl.attribution` reference

`tl.attribution` provides input and intermediate-layer attribution methods directly on
PyTorch modules. Each call returns `Attribution(method, values, target_repr, extra)`.
`target` is an output-class index, a callable that maps model output to one scalar tensor, or a
[`SiteTarget`](#site-targets-sitetarget) naming a hidden layer's output; the layer methods name a
module from `dict(model.named_modules())` and also accept [integer inputs](#integer-inputs-token-ids)
such as token ids.

The stage-4b compute-kit spellings (Integrated Gradients completeness fields,
`occlusion`, and Grad-CAM overlay fields) are **DOCUMENTED-UNSTABLE**: they are
provisional pending the naming session and may change without a deprecation shim. The exact
inventory is locked in the [glossary](glossary.md#documented-unstable-attribution-kit-index).

The examples use a deterministic small classifier. Input methods return a tensor shaped like
the attributed input; layer methods return the named layer's activation shape. For convolutional
maps, [`grad_cam`](#grad_cam) returns an input-resolution `N, 1, H, W` map.

Passing the same tensor object to several input slots preserves that identity inside the
attribution forward (`a is b` control flow runs exactly as in your own call), and every
occurrence slot reports the shared tensor's full accumulated gradient. Path methods require
repeated references to carry identical baselines. A module fired several times contributes
through every firing: `layer_attribution`, `layer_integrated_gradients`, and
`layer_conductance` total the per-firing terms, while `grad_cam` requires the target layer to
fire exactly once and raises `AttributionError` otherwise.

## `saliency`

`tl.attribution.saliency(model, inputs, input_kwargs=None, *, target=...)` returns absolute
input gradients for the selected scalar target.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.saliency(model, torch.tensor([[1.0, 2.0]]), target=0))
```

Output:

```text
AttributionResult(method='saliency', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=[])
```

## `input_x_grad`

`tl.attribution.input_x_grad(model, inputs, input_kwargs=None, *, target=...)` returns the
gradient times each attributed input value. Compare it with [`saliency`](#saliency) when the
input magnitude should be part of the score.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.input_x_grad(model, torch.tensor([[1.0, 2.0]]), target=0))
```

Output:

```text
AttributionResult(method='input_x_grad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=[])
```

## `integrated_gradients`

`tl.attribution.integrated_gradients(model, inputs, input_kwargs=None, *, target=...,
num_steps=50, baseline=None)` integrates input gradients along a straight path from `baseline`
(zeros by default). Baseline choice changes the interpretation. Every result computes the
endpoint `target_delta = f(input) - f(baseline)` and reports `attribution_sum` plus the signed
`completeness_residual = attribution_sum - target_delta`; callers can therefore inspect the IG
axiom directly instead of trusting an attribution array without its convergence evidence.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.integrated_gradients(model, torch.tensor([[1.0, 2.0]]), target=0, num_steps=4))
```

Output:

```text
AttributionResult(method='integrated_gradients', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['attribution_sum', 'baseline', 'completeness_residual', 'n_steps', 'target_delta'])
```

## `occlusion`

`tl.attribution.occlusion(trace, selection, *, score=..., baseline="zeros",
blur_kernel_size=3)` resolves an ACT Selection, applies the occlusion through a temporary
`Trace.fork().do(...)`, and returns `score(original) - score(occluded)`. The scorer receives the
Trace, so the output projection is explicit (for example,
`score=lambda run: run.output_ops[0].out[..., 0].sum()`).

The baseline is always named and disclosed. `"zeros"` uses `tl.zero_ablate()` and is the
documented default; `"mean"` uses `tl.mean_ablate()`; `"blur"` uses a same-shaped spatial mean
blur before the Selection engine scatters the chosen region. These counterfactuals are not
interchangeable, and the result records the endpoint scores, Selection digest, and chosen policy.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(
    nn.ReLU(),
    torch.tensor([[-1.0, 2.0]]),
    capture=tl.options.CaptureOptions(intervention_ready=True),
)
region = tl.units(trace.input_ops[0].label, [(0, 1)])
result = tl.attribution.occlusion(
    trace,
    region,
    score=lambda run: run.output_ops[0].out.sum(),
    baseline="zeros",
)
print(result)
trace.cleanup()
```

## `smoothgrad`

`tl.attribution.smoothgrad(model, inputs, input_kwargs=None, *, target=..., num_samples=25,
noise_level=0.1, seed=None)` averages saliency across Gaussian-noised inputs. Supplying `seed`
makes the noise deterministic without changing the global torch RNG state.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.smoothgrad(model, torch.tensor([[1.0, 2.0]]), target=0, num_samples=3, seed=0))
```

Output:

```text
AttributionResult(method='smoothgrad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['n_samples', 'noise_level', 'seed'])
```

## `grad_cam`

`tl.attribution.grad_cam(model, inputs, input_kwargs=None, *, target=..., layer=None,
reshape=None, relu=True, overlay=True, image=None, alpha=0.6, cmap="magma")`
forms a Grad-CAM map from a 4D `N, C, H, W` convolution-style layer and upsamples it to the
spatial size of the input that actually feeds the target layer (proven through the autograd
graph, not taken from argument order). Use a layer name from `model.named_modules()`. If no
spatial input feeds the layer, or several feed it with different grids, the call raises
`AttributionError` instead of guessing a coordinate system.

The values remain the input-resolution interpolated tensor for compatibility, but
`native_map_resolution` and `rendered_map_resolution` are separate metadata. The default rendered
overlay uses the same heatmap blending path as `receptive_field.show()` and carries a visible
footer such as `CAM native map: 7x7; display interpolated`. Interpolation is presentation, not new
measured precision.

```python
import torch
from torch import nn
import torchlens as tl

class ImageModel(nn.Module):
    def __init__(self):
        super().__init__(); self.conv = nn.Conv2d(1, 2, 1); self.head = nn.Linear(8, 2)
    def forward(self, x): return self.head(torch.relu(self.conv(x)).flatten(1))

print(tl.attribution.grad_cam(ImageModel(), torch.ones(1, 1, 2, 2), target=0, layer="conv"))
```

Output:

```text
AttributionResult(method='grad_cam', values=Tensor(shape=(1, 1, 2, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['grid_evidence', 'grid_route', 'integer_input_layers', 'layer', 'layer_reason', 'layer_source', 'native_map_resolution', 'overlay', 'prefix_tokens', 'relu', 'rendered_map_resolution', 'upsampling'])
```

### Vision transformers: token grids

A transformer layer emits tokens `N, T, C`, not a feature map. `grad_cam` lays them on the grid
that a metadata-only trace of the same call proves, read from the graph's shapes and edges and
never from a list of model names. The patch embedding that flattens a rank-4 map into tokens
gives the grid, and a token concatenation whose last piece holds those patch tokens gives the
leading tokens to drop: one class token for ViT and CLIP, class and distillation tokens for
distilled DeiT, the class token plus the register tokens for DINOv2 with registers. A Swin
stage's own grid comes from the view that lays its tokens out as `N, H, W, C`, so each stage after
patch merging gets its own, smaller grid. `extra["grid_route"]` is `"derived"`,
`extra["prefix_tokens"]` the leading-token count dropped, and `extra["grid_evidence"]` the trace
ops that proved the grid.

When the trace proves no grid (tokens cut by `F.unfold`, for example), or proves several, the
call raises `AttributionError` with code `grad_cam_token_grid_unprovable` and a spelling to
run instead. That spelling uses `reshape=`, the escape hatch: a callable that maps the layer
output to `N, C, H, W`. It must only rearrange (slice, transpose, reshape), because it is
applied to the activation and to its gradient alike; `extra["grid_route"]` is then
`"reshape"`.

```python
import torch
from torch import nn
import torchlens as tl

class Patches(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Conv2d(1, 4, 2, stride=2); self.cls = nn.Parameter(torch.zeros(1, 1, 4))
        self.block = nn.Linear(4, 4); self.head = nn.Linear(4, 2)
    def forward(self, x):
        tokens = self.embed(x).flatten(2).transpose(1, 2)
        tokens = self.block(torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1))
        return self.head(tokens.mean(dim=1))

model, x = Patches(), torch.ones(1, 1, 4, 4)
derived = tl.attribution.grad_cam(model, x, target=0, layer="block", overlay=False)
by_hand = tl.attribution.grad_cam(
    model, x, target=0, layer="block", overlay=False,
    reshape=lambda a: a[:, 1:].transpose(1, 2).reshape(a.shape[0], a.shape[2], 2, 2),
)
print(derived.extra["grid_route"], derived.extra["native_map_resolution"], derived.extra["prefix_tokens"])
print(torch.equal(derived.values, by_hand.values))
```

Output:

```text
derived (2, 2) 1
True
```

### Choosing the layer automatically

Called without `layer`, `grad_cam` chooses it from the trace. The candidates are named modules
that fire once and output a spatial map or a provable token grid. The chosen one lies on every
path from the inputs to the outputs, is the deepest such module, and carries a nonzero target
gradient on its grid; several modules that return the same tensor resolve to the outermost.
On a torchvision ResNet this is `layer4`. On a ViT classifier that reads only the class token,
the last block's patch tokens carry a zero gradient, so the choice steps back to the residual
stream entering the last block. `extra["layer_source"]` is `"trace"` and
`extra["layer_reason"]` says why; with an explicit `layer` they are `"caller"` and `None`.
When maps exist but none lies on every input-to-output path (two branches joined after
pooling), the call refuses with `grad_cam_layer_choice_ambiguous` and names the candidates; a
model with no spatial map and no token grid refuses with `grad_cam_layer_unchosen`. The
automatic call computes the same map as the explicit call on the chosen layer.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2))
result = tl.attribution.grad_cam(model, torch.ones(1, 1, 4, 4), target=0, overlay=False)
print(result.extra["layer"], result.extra["layer_source"])
```

Output:

```text
1 trace
```

### torch-cam as an alternative

[TorchCAM](https://github.com/frgfm/torch-cam) (`pip install torchcam`) is a separate library
with a wider set of class activation maps (Grad-CAM++, Smooth Grad-CAM++, Score-CAM, Layer-CAM
and others) and its own automatic target-layer choice. TorchLens ships native Grad-CAM only and
no torch-cam extra; use torch-cam directly when you need another CAM variant. The
[torch-cam recipe](../recipes/torchcam.md) runs both on one model and checks that their Grad-CAM
maps agree.

## `layer_integrated_gradients`

`tl.attribution.layer_integrated_gradients(model, inputs, input_kwargs=None, *, target=...,
layer=..., baseline=None, num_steps=50)` applies the integrated-gradients path rule to a named
intermediate activation. See [`integrated_gradients`](#integrated_gradients) for input-level IG.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_integrated_gradients(model, torch.ones(1, 2), target=0, layer="0", num_steps=4))
```

Output:

```text
AttributionResult(method='layer_integrated_gradients', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['attribution_sum', 'completeness_caveat', 'completeness_residual', 'integer_input_layers', 'layer', 'n_steps', 'path_evaluations_logical', 'physical_forward_calls', 'residual_rel', 'step_audit', 'step_batch_size', 'target_delta', 'target_delta_abs'])
```

## `layer_conductance`

`tl.attribution.layer_conductance(model, inputs, input_kwargs=None, *, target=..., layer=...,
baseline=None, num_steps=50)` decomposes input integrated gradients onto the selected hidden
units along the input path. It uses the same baseline and step controls as
[`layer_integrated_gradients`](#layer_integrated_gradients).

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_conductance(model, torch.ones(1, 2), target=0, layer="0", num_steps=4))
```

Output:

```text
AttributionResult(method='layer_conductance', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['attribution_sum', 'completeness_caveat', 'completeness_residual', 'integer_input_layers', 'layer', 'n_steps', 'path_evaluations_logical', 'physical_forward_calls', 'residual_rel', 'step_audit', 'step_batch_size', 'target_delta', 'target_delta_abs'])
```

## `overlay` — drawing attribution on the graph

`tl.attribution.overlay(trace, source, *, reduce="abs_sum")` bridges attribution numbers onto
the graph: it returns a callable for `Trace.draw(color_by=...)` that paints each attributed
layer's reduced magnitude onto its module-output node and leaves every other node unencoded
(the legend carries the `n/a = unencoded` note). `source` is one layer-scoped
`Attribution` (its `extra["layer"]` anchors it), an iterable of them, or an explicit
mapping from `model.named_modules()` name or trace layer label to a result, tensor, or real
scalar. `reduce` is closed vocabulary: `"abs_sum"` (default), `"abs_mean"`, `"sum"`, `"max"`.
Unknown keys and unknown reductions raise `AttributionError` naming the fix. A module fired
several times paints its per-layer total on every one of its nodes; per-pass splits are not
claimed. `source` may also be an `AttributionGraph`; see [Drawing a graph on
its trace](#drawing-a-graph-on-its-trace-overlay).

```python
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
inputs = torch.ones(1, 2)
results = [
    tl.attribution.layer_attribution(model, inputs, target=0, layer=name)
    for name in ("0", "2")
]
trace = tl.trace(model, inputs)
color_by = tl.attribution.overlay(trace, results)
print(round(color_by(trace["linear_1_1"]), 6) == round(results[0].values.abs().sum().item(), 6))
print(color_by(trace["relu_1_2"]))
```

Output:

```text
True
None
```

To render, pass the callable to the ordinary encoding channel — the legend discloses the
callable source and the min/mid/max ramp:

```text
trace.draw(color_by=tl.attribution.overlay(trace, results), path="attribution_graph")
```

## `layer_attribution`

`tl.attribution.layer_attribution(model, inputs, input_kwargs=None, *, target=..., layer=...,
method="activation_x_grad")` returns either activation-times-gradient or absolute gradient for a
named layer. It is the one-step layer counterpart to the path methods above.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_attribution(model, torch.ones(1, 2), target=0, layer="0"))
```

Output:

```text
AttributionResult(method='layer_activation_x_grad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['integer_input_layers', 'layer'])
```

## Integer inputs (token ids)

The layer methods (`layer_attribution`, `layer_integrated_gradients`, `layer_conductance` and
`grad_cam`) accept inputs that hold no floating-point tensor, such as token ids, because what they
attribute is a layer's output. Integer values have no gradient, so the input path of the path
methods starts where the integers become floats: at the outputs of the innermost modules that take
only integer tensors from your inputs and return a floating-point tensor (an `nn.Embedding`, in
practice). Those outputs stand in for a floating-point input, so each result equals the same
method run on a model that takes the embedding output as its input. The default baseline is a
zero embedding; `baseline=` may instead hold reference ids of the same shape and dtype (a padding
token, say), whose embeddings start the path. `extra["integer_input_layers"]` names the modules the
path started at, and is `()` for floating-point inputs.

```python
import torch
from torch import nn
import torchlens as tl


class TokenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.mix = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, ids):
        return self.head(torch.tanh(self.mix(self.embed(ids))).mean(dim=1))


torch.manual_seed(0)
model = TokenClassifier()
ids = torch.tensor([[1, 4, 2, 7]])
result = tl.attribution.layer_integrated_gradients(model, ids, target=1, layer="mix", num_steps=16)
print(tuple(result.values.shape), result.extra["integer_input_layers"])
```

Output:

```text
(1, 4, 4) ('embed',)
```

The input methods (`saliency`, `input_x_grad`, `integrated_gradients`, `smoothgrad`,
`gradient_shap`, `guided_backprop`, `deconvolution`) attribute the input itself and refuse integer
inputs with `attribution_input_not_float`; the message spells a layer-method call on the embedding
that runs. A path method whose integer inputs reach no embedding (the model casts them with
`.float()`, say) refuses with `layer_path_no_embedding` and spells the path-free
`layer_attribution` call. For `grad_cam`, integer inputs laid out `N, H, W` or `N, C, H, W` give
the map its grid; others refuse with `grad_cam_integer_grid_invalid`.

## Site targets (`SiteTarget`)

`tl.attribution.SiteTarget(layer, *, unit=None, call=None, sae=None)` passed as `target=` attributes a
hidden site instead of the model output: the output of the module named `layer`, optionally one
unit of it, summed over the batch (the summing rule of an integer `target=`). This is the
neuron-attribution family: `saliency` toward a hidden unit gives that unit's input gradients, and
a layer method toward a deeper unit says how an earlier layer drives it.

- `unit=None` attributes the whole output, summed.
- An int `unit` selects `output[..., unit]`, the last axis, as an integer `target=` does.
- A tuple `unit` indexes the axes after the batch axis: `(3,)` is channel 3 of a convolution
  output, `(3, 1, 2)` one position of it.
- `call` picks which call of a module that runs more than once, counted from 0.
- `sae=` (an SAE, any object with `encode` and `decode`, trained on that output) makes the site
  the SAE's latents, `sae.encode(output)`, and `unit` selects among them:
  `SiteTarget("fc1", sae=sae, unit=3442)` attributes SAE latent 3442. The same SAE registered
  with `tl.bridge.sae.register` is the facet family `tl.facet(name).latent(k)` reads and edits.

```python
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
x = torch.randn(2, 3)
result = tl.attribution.saliency(mlp, x, target=tl.attribution.SiteTarget("1", unit=2))
print(result.target_repr, tuple(result.values.shape))
```

Output:

```text
SiteTarget('1', unit=2) (2, 3)
```

Refusals name the fix: a layer the model lacks (`attribution_site_not_found`, with the closest
module name), a unit outside the output (`attribution_site_unit_invalid`), a module that ran more
than once without `call=` or did not run (`attribution_site_firing_invalid`), a site computed
without the attributed inputs, whose attributions would all be zero
(`attribution_site_independent_of_input`), an `sae=` object without an encoder and decoder
(`attribution_site_sae_invalid`), and a method that scores only the model output
(`infidelity`, for example: `attribution_site_target_unsupported`).

## Evaluating attributions

Two checks reviewers ask of an attribution method, built on TorchLens's own pieces. Both take
any input-space result, including [`grad_cam`](#grad_cam) maps (CNN feature maps and ViT token
grids alike; the input-resolution `N, 1, H, W` map is spread across the input channels). The
two existing metrics, [`infidelity` and `sensitivity`](#infidelity-and-sensitivity), sit
beside them. [ROAD](#road-remove-and-debias-road_curve) is the deletion curve with noisy linear
imputation in place of a constant baseline. A further check, the [causal audit](#auditing-internal-attributions-causal_audit),
grades internal attributions: it ablates the hidden units a layer attribution ranks and
compares the measured effects with the claim.

### Deletion and insertion curves (`deletion_curve`, `insertion_curve`)

`tl.attribution.deletion_curve(model, inputs, input_kwargs=None, *, target, attribution,
baseline="zeros", num_steps=20, features="auto", patch_size=8)` ranks the input features by
attribution (highest first, per example) and replaces the top `floor(k * F / num_steps)` of them
with the baseline at step `k`, recording the target score after each step: the
most-relevant-first curve of Samek et al. (2017) and the deletion metric of Petsiuk et al.
(2018). `insertion_curve` (Petsiuk et al., 2018) starts from the baseline and reveals the top
features of the input instead. The `PerturbationCurve` holds `fractions`, `scores`
(`(num_steps + 1, B)`: one column per example for an integer target, one column for a callable
target), `area` (trapezoid rule over the fractions; lower is better for deletion, higher for
insertion) and `aopc`, the mean score change from the start (`mean_k(s_0 - s_k)` for deletion,
`mean_k(s_k - s_0)` for insertion; higher is better for both). Per-example values are in
`extra`, the resolved policy in `settings`.

- `baseline`: `"zeros"`, `"blur"` (an 11x11 box blur of the input, image inputs only), a number,
  or a tensor of the input's shape.
- `features`: `"pixels"` (all channels at one position), `"patches"` (`patch_size` squares;
  edge patches may be smaller), `"tokens"` (all values at one sequence position) or
  `"elements"` (every scalar). `"auto"` picks pixels for `N, C, H, W` inputs, tokens for
  `N, T, D` inputs, elements otherwise. Several floating-point inputs share one ranking.

Each step is a recorded intervention. `curve.steps[k].build_interventions` holds the step's edits as
`(selection, edit)` pairs, one per evaluated input that holds any baseline value:
`tl.units(<input layer>, <baseline mask>)` with `tl.replace_with(<baseline>)`. A curve point
therefore comes with the exact edit that produced it, and it replays through the intervention
engine on a replay-ready trace of the original input with `trace.fork().do(selection, edit)`.
The scores themselves come from direct forward passes, which the replay matches; the first
access to `intervention` traces the model once to name its input layers.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(4, 1)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[4.0, -1.0, 2.0, 0.5]]))
    model.bias.zero_()
x = torch.ones(1, 4)
result = tl.attribution.input_x_grad(model, x, target=0)
curve = tl.attribution.deletion_curve(model, x, target=0, attribution=result, num_steps=4)
print(curve.scores[:, 0].tolist(), curve.aopc)

trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
fork = trace.fork()
for selection, edit in curve.steps[1].build_interventions:
    fork.do(selection, edit)
print(fork[fork.output_layers[0]].out.item(), curve.steps[1].build_inputs.tolist())
```

Output:

```text
[5.5, 1.5, -0.5, -1.0, 0.0] 4.4
1.5 [[0.0, 1.0, 1.0, 1.0]]
```

`step.build_inputs` and `step.build_baseline_mask` show the perturbed input and which elements hold the
baseline. To compare against chance, evaluate a random ranking the same way
(`attribution=torch.rand_like(x)`). Curves score the model output, so a `SiteTarget` refuses
(`attribution_site_target_unsupported`); an attribution that does not cover the input (layer
space) refuses `attribution_eval_map_misaligned`, and a bad baseline, step count or grouping
refuses `attribution_eval_setting_invalid`, each naming a spelling that works.

### The randomization test (`randomization_test`)

`tl.attribution.randomization_test(model, inputs, input_kwargs=None, *, target, method,
method_kwargs=None, randomization="cascading", layers=None, seed=0, keep_values=True)` is the
model-parameter randomization test of Adebayo et al. (2018), "Sanity checks for saliency
maps": it re-initializes layers from the output side (`"cascading"`: cumulatively;
`"independent"`: one at a time), re-runs `method` and compares each map with the original.
An explanation that stays the same once the parameters are destroyed does not explain the
model. The measures are `spearman` (rank correlation of the signed maps), `spearman_abs`
(of their absolute values) and `ssim` (Wang et al., 2004: 7x7 uniform window, on the absolute
maps summed over channels and rescaled to `[0, 1]`; `None` unless the maps are spatial planes
of side 7 or more).

```python
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
x = torch.randn(3, 4)
check = tl.attribution.randomization_test(mlp, x, target=1, method=tl.attribution.saliency)
print(check.module_addresses, [step.randomized_module_addresses for step in check.steps])
print(len(check.similarity("spearman")), check.similarity("ssim"))
```

Output:

```text
('2', '0') [('2',), ('2', '0')]
2 (None, None)
```

`layers=` names the modules to randomize, top first (default: every module with parameters of
its own, in reverse order of first execution). A module is re-initialized by its own
`reset_parameters()` when it has one, else each of its parameters is drawn from
`N(0, 0.02^2)`; `seed=` fixes the draws. Everything runs on deep copies inside a forked random
state, so the caller's model (parameters and gradients) and the global random state are
unchanged. `method` is called as `method(model_copy, inputs, input_kwargs, target=target,
**method_kwargs)`; pass `method_kwargs={"layer": ...}` to `grad_cam` so every map uses the
same layer. Each `RandomizationStep` keeps its map in `values`; past 256 MiB of kept maps the
test warns once (`attribution_randomization_maps_large`) and names `keep_values=False`. Unknown
layers refuse `attribution_randomization_layer_invalid`, and so does a method under which no
module with parameters runs (the test would otherwise randomize nothing and pass).

### ROAD: remove and debias (`road_curve`)

`tl.attribution.road_curve(model, inputs, input_kwargs=None, *, target, attribution,
ranking="most_relevant_first", num_steps=20, features="auto", patch_size=8, noise_std=0.01,
seed=0)` is ROAD (Rong et al., 2022, "A Consistent and Efficient Evaluation Strategy for
Attribution Methods"). It is the deletion curve above with one change: a removed pixel does not
take a constant baseline, whose shape would tell the model where the mask is, but the weighted
mean of its 8 neighbours (1/6 for the direct neighbours, 1/12 for the diagonals, renormalised at
the border), solved jointly for all removed pixels of a channel, plus Gaussian noise of
`noise_std` (0.01, the authors' reference value). `ranking="most_relevant_first"` (MoRF) removes
the most attributed features first, `"least_relevant_first"` (LeRF) the least attributed. The
result is a `PerturbationCurve` with `metric="road"`; each step's `intervention` holds
`tl.units(<input layer>, <mask>)` with `tl.replace_with(<that step's imputed values>)`, so every
point replays like any curve step. The noise of a step comes from a local generator seeded by
`seed` and the step, so a step recomputes to the same values and the global random state is
untouched. A removed region with no kept neighbour stays at zero (plus noise). ROAD needs
`(N, C, H, W)` image inputs; other inputs refuse `attribution_eval_setting_invalid` and name
`deletion_curve`. Quantus also provides ROAD; this one shares the curves' recorded edits.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Flatten(), nn.Linear(9, 1))
with torch.no_grad():
    model[1].weight.copy_(torch.arange(1.0, 10.0).reshape(1, 9))
    model[1].bias.zero_()
x = torch.ones(1, 1, 3, 3)
x[0, 0, 1, 1] = 4.0
result = tl.attribution.input_x_grad(model, x, target=0)
road = tl.attribution.road_curve(model, x, target=0, attribution=result, num_steps=9, noise_std=0.0)
print(road.kind, [round(value, 4) for value in road.scores[:2, 0].tolist()])
print(round(road.steps[1].build_inputs[0, 0, 1, 1].item(), 4), road.settings["ranking"])
```

Output:

```text
road [60.0, 45.0]
1.0 most_relevant_first
```

The center pixel (attribution 20) goes first; with every neighbour equal to 1 its imputed value
is 1, so the score falls from 60 to `60 - 5 * 4 + 5 * 1 = 45`.

### Auditing internal attributions (`causal_audit`)

`tl.attribution.causal_audit(model, inputs, input_kwargs=None, *, claim, target=None,
corrupted_input=None, metric=None, edges=None, k=5, edit="zero", controls=("bottom", "random"),
num_random_draws=10, per_unit=True, seed=0)` tests an internal attribution causally. The `claim` is
one of four kinds: a layer attribution (`layer_attribution`, `layer_integrated_gradients`,
`layer_conductance`), for a model output target or for a [`SiteTarget`](#site-targets-sitetarget),
whose values score every unit of one hidden site; a direct-logit-attribution table from
`torchlens.mechinterp.direct_logit_contributions` ([below](#direct-logit-attribution-tables)); a
one-backward read table from `tl.attribution.read` ([below](#one-backward-read-tables)); or an
EDGE table of edge attribution patching scores from `torchlens.mechinterp.edge_scores`
([below](#edge-tables-edge-attribution-patching)). A layer claim needs the `target=` it was
computed for; an EDGE table needs the `corrupted_input=` and `metric=` it was scored with.
The audit ablates each unit alone (unit ablation, Meyes et al., 2019) and measures the target's
change, `score(original) - score(ablated)`, the same sign as an activation-times-gradient claim.
It then ablates the `k` units with the largest claimed magnitude together, beside two controls:
the `k` units with the smallest claimed magnitude (`"bottom"`) and `num_random_draws` seeded random
sets of `k` (`"random"`). `rank_agreement` is the Spearman correlation of the claimed and
measured per-unit effects; each set reports its claimed sum next to its measured joint effect.

Every ablation runs through the intervention engine on an intervention-ready trace. The rows
run as one `torchlens.experiment.site_sweep` with one candidate per unit,
`tl.units(<site>, <mask>)` (the replay engine, nothing retained), wherever one edit serves every
unit: every ablation of a layer claim, and zero ablation of a table. Each row's value equals
ablating that unit alone on its own `trace.fork().do(...)`, exactly. Sets, and a table's `"mean"`
or baseline rows (each site's edit is that site's own value), run one `trace.fork().do(...)`
each. Each row (`CausalAuditUnit`) and set (`CausalAuditSet`) carries its edit in
`intervention` as `(tl.units(<site>, <mask>), edit)` pairs, the same form as a curve step, so
any point replays exactly.

```python
import torch
from torch import nn
import torchlens as tl


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x)) + x.sum(dim=1, keepdim=True)


model = Residual()
with torch.no_grad():
    model.fc1.weight.copy_(torch.tensor([[1.0, 0, 0], [0, 2, 0], [0, 0, 1], [1, 1, 1]]))
    model.fc2.weight.copy_(torch.tensor([[3.0, -1.0, 0.5, 2.0]]))
    model.fc1.bias.zero_()
    model.fc2.bias.zero_()
x = torch.tensor([[1.0, 2.0, 3.0]])
claim = tl.attribution.layer_attribution(model, x, target=0, layer="fc1")
audit = tl.attribution.causal_audit(model, x, claim=claim, target=0, k=2, num_random_draws=3)
print([(row.index, row.claimed_effect, row.measured_effect) for row in audit.indices])
print(audit.rank_agreement, [(item.kind, item.indices, item.measured_effect) for item in audit.sets[:2]])

trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
fork = trace.fork()
for selection, edit in audit.set_of("top")[0].build_interventions:
    fork.do(selection, edit)
print(audit.original_score - fork[fork.output_layers[0]].out.item())
```

Output:

```text
[((0,), 3.0, 3.0), ((1,), -4.0, -4.0), ((2,), 1.5, 1.5), ((3,), 12.0, 12.0)]
1.0 [('top', ((3,), (1,)), 8.0), ('bottom', ((0,), (2,)), 4.5)]
8.0
```

On this linear model every claim is exact, so agreement is 1.0 and the sets add up; on a
nonlinear model the gap between a set's claimed sum and its measured effect is the finding.

- `edit`: `"zero"` (default; `tl.zero_ablate()`), `"mean"` (each unit's mean over the
  batch, through `tl.replace_with`), or a number or tensor broadcast to the site (a baseline
  activation). Mean ablation over a batch of one changes nothing.
- Units are the site's elements after the batch axis; a unit is ablated in every example, and
  its claimed score is the claim summed over the batch, the same summing rule as an integer
  `target=`. A `SiteTarget` target is read from its own site on every fork.
- `per_unit=False` skips the single-unit table (`units` is empty and `rank_agreement` is
  `None`) and measures only the sets; past 1024 units the audit warns once
  (`attribution_audit_units_many`) and names it.
- The model is traced once in eval mode without gradients, every edit lands on forks of that
  trace, and the random draws use a local generator inside a forked random state, so the
  caller's model and the global random state are unchanged.

Each row also names its `site` (the op label the edit selects) and its `component` (the
claim's own name for it: the layer here, a row label of a direct-logit table, a read row's site,
an EDGE row's edge).

A claim the audit cannot ground refuses `attribution_audit_claim_unsupported`: a plain tensor, an
input-level result, a site whose module runs more than once, a direct-logit row with no captured
site, and the read tables described below. A claim whose values do not have the site's output
shape on these inputs, that was computed for another target, or whose logits are not this run's,
refuses `attribution_audit_claim_misaligned`; bad `k`, `edit`, `controls` or `num_random_draws`
refuse `attribution_eval_setting_invalid`, and so do `corrupted_input=`, `metric=` and `edges=` given
for a claim that is not an EDGE table. Each refusal names a spelling that works.

#### Direct-logit-attribution tables

A `ContributionScores` table from `torchlens.mechinterp.direct_logit_contributions` splits the model's
logit, or logit difference, at chosen positions into one row per component (the embedding and
every residual writer), plus one constant row (the final norm's shift and the unembedding bias).
Each row claims how much logit its component contributes directly. The audit ablates each
component's whole site (`Coordinate.op_label`) and reads the same logit quantity from the
unembedding head's output on the fork, summed over the batch, the table's positions and its
answer and `vs` tokens; the row's claimed effect is its values summed the same way. Leave
`target=` out: the table fixes its own target.

- The native-logit check: before anything is ablated, the head's logits on the audit's trace must
  reproduce the table's `native` logits within the table's own identity tolerance. A table
  computed on other inputs, other weights or in another mode refuses
  `attribution_audit_claim_misaligned`, naming the `direct_logit_contributions` call that
  recomputes it.
- The constant row has no site, so it is not ablated: `extra["constant_row"]` reports its claimed
  sum, and `extra["native_score"]` the table's native sum (rows plus constant).
- Under zero ablation the per-row ablations run as one `torchlens.experiment.site_sweep` (one
  candidate per component site, the replay engine, nothing retained); other ablations and the
  sets use one fork each. A unit is `(row,)`, the row's index in the table.
- Direct logit attribution reads a component's direct path only; the ablation also removes what
  later components read from it, and the final norm rescales. The gap between the two is what
  the audit reports.

```python
import torch
from torch import nn
import torchlens as tl
import torchlens.mechinterp


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 4)
        self.writers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
        self.ln_f = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 5)

    def forward(self, ids):
        hidden = self.embed(ids)
        for writer in self.writers:
            hidden = hidden + writer(hidden)
        return self.lm_head(self.ln_f(hidden))


torch.manual_seed(0)
model = TinyLM().eval()
ids = torch.tensor([[1, 2, 3]])
claim = torchlens.mechinterp.direct_logit_contributions(tl.trace(model, ids), answer=1, vs=2)
audit = tl.attribution.causal_audit(model, ids, claim=claim, k=1, num_random_draws=2)
for row in audit.indices:
    print(row.component, round(row.claimed_effect, 3), round(row.measured_effect, 3))
print(round(audit.extra["constant_row"], 3), round(audit.rank_agreement, 3))
```

Output:

```text
embed 0.739 -0.562
writers.0 -0.239 -0.967
writers.1 0.111 0.32
0.495 0.5
```

Here each writer reads the residual stream and the final norm rescales, so ablating a component
moves the logit by more (or less) than its direct share: the embedding's direct share is positive
and its measured effect negative, and rank agreement is 0.5. The constant row's 0.495 is the part
of the logit difference no component can remove.

#### One-backward read tables

A `ReadTable` from `tl.attribution.read` with `method="activation_x_grad"` scores every site
(`reduce="sum"`) or every element of it (`reduce=None`) for one seeded target. Summed over a site,
activation times gradient is the first-order estimate of what zero-ablating the site does to the
target (attribution patching), and it is exact when the target is linear in the site. The audit
ablates each `ACT` row's site, or each element of it, and measures the target the read was
taken for; pass that target as `target=`, the same `tl.attribution.seed(...)`. A unit is
`(row,)` at site grain or `(row, *element)` at element grain, the row being its index in
`table.rows()` and the element indexed after the batch axis. Under zero ablation every unit,
each element included, is one candidate of the audit's `site_sweep` (an element as
`tl.units(<site>, <mask>)`), so the sweep's effect table holds one member per unit.

- One target per audit: a table read for several targets refuses and names
  `claim=claim.target("t0")`. A `target=` that is not the table's recorded seed, or a table
  whose site shapes differ from these inputs, refuses `attribution_audit_claim_misaligned`.
- Rows whose status is not `ok` carry no score; they are listed in `extra["rows_not_audited"]` as
  `(site, status)` and not ablated.
- `grad` and `activation` reads, and site reductions other than `"sum"`, refuse
  `attribution_audit_claim_unsupported`: they are not estimates of an ablation effect.
- A table of `EDGE` rows is audited by patching each edge ([below](#edge-tables-edge-attribution-patching));
  a table that mixes `EDGE` rows into site reads refuses `attribution_audit_claim_unsupported`.

```python
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 3)
target = tl.attribution.seed("output_1", index=(0, 1))
table = tl.attribution.read(tl.trace(model, x), target=target, frozen=None)
audit = tl.attribution.causal_audit(model, x, claim=table, target=target, k=1, num_random_draws=2)
for row in audit.indices:
    print(row.site, round(row.claimed_effect, 4), round(row.measured_effect, 4))
print(audit.rank_agreement)
```

Output:

```text
linear_1_1:1 0.0531 0.0531
relu_1_2:1 0.0531 0.0531
linear_2_3:1 0.106 0.106
output_1:1 0.106 0.106
1.0
```

Zero-ablating a site of this ReLU network moves the target by exactly its activation times
gradient (a ReLU passes a zeroed input through as zero), so every row agrees; on a model with
smooth nonlinearities, or a read with a frozen linearization, the gap is the finding.

#### EDGE tables (edge attribution patching)

A `ReadTable` from `torchlens.mechinterp.edge_scores(model, inputs, *, corrupted_input, metric)` holds one
`EDGE` row per edge `s->t` of the component graph, addressed by the edge's name
(`"a0.h1->m1"`). Its score, `(s_corrupt - s_clean) . dM/d(input of t)` summed over positions and
reduced over examples (EAP, or EAP-IG along the input path), estimates what patching the edge
does to the metric: receiver `t` reads `clean input - s_clean + s_corrupt`. The audit does that
patch. Each `ok` row's edge is patched alone with `torchlens.mechinterp.patch_edges` on a fork of the
audit's clean trace, with the sender's value from the corrupted run, and the metric is read from
the unembedding output of the fork. The measured effect is `M(patched) - M(clean)`, reduced over
examples as the table was (`extra["example_reduction"]`: `"mean"` or `"sum"`), the same sign as
the score. A unit is `(row,)`, the row's index in `table.rows()`; its `component` is the edge
name and its `site` the op whose value the patch writes (the receiver's input, or for a q, k or v
receiver one head's slice of the projection output).

- Alignment: an EDGE table records neither its corrupted inputs nor its metric (its rows cannot
  be re-scored), so pass the `corrupted_input=` and `metric=` that `edge_scores` was given; they keep
  `edge_scores`' meanings. The audit rebuilds the component graph on these inputs and checks what
  the table does record: its example count, its method and reduction, every `ok` row's edge name
  and sender shape. A missing `corrupted_input=` or `metric=`, corrupted inputs of another shape, a
  metric that does not return one value per example or one scalar, a table scored on other inputs,
  an edge name the graph does not have, or a `target=` refuses `attribution_audit_claim_misaligned`.
  The audit cannot check that `metric=` is the very function the table was scored with; a wrong
  one shows up as a gap between claims and effects.
- Rows whose status is not `ok` (a receiver the metric does not reach) carry no score; they are
  listed in `extra["rows_not_audited"]` as `(edge, status)` and not patched. Bias-sender rows
  (`b{L}->...`) score exactly zero and are audited like any other row: their patch changes
  nothing, so they measure zero up to rounding, and they enter rank agreement.
- Sets: a top-k, bottom-k or random-k set is patched in one `torchlens.mechinterp.patch_edges` call,
  never measured as a sum of single-edge effects. When its edges enter several receivers, each
  later receiver reads the forward as patched so far (the MIB circuit-track convention that
  `patch_edges` follows), so a patched receiver upstream changes what a downstream sender carries.
  `extra["sets_not_measured"]` lists sets the audit could not patch; it is empty, since every set
  is patched.
- `edges=` audits only the named rows (edge names or `ComponentEdge` objects; default every `ok`
  row). A GPT-2 small table has 34,957 rows, and one patch there costs 0.5 to 8 s on 4 CPU cores,
  so past 1024 rows the audit warns once (`attribution_audit_units_many`) and names `edges=` with
  the largest rows, or `per_unit=False`. `edit=` does not apply: an edge is patched, not
  ablated (`settings["edit"]` is `"edge_patch"`).
- Each row's and set's `intervention` holds the `(selection, value)` pairs `patch_edges` wrote,
  one per receiver in the order it wrote them, read from the fork that measured the row or set (the
  EDGE selection of the receiver's input edge with the value substituted there, or the ACT
  selection of one head's projection slice). An EDGE selection addresses its edge by the call ids of one
  capture, so the pairs replay on forks of the audit's own clean trace, `extra["trace"]`:
  `fork = audit.extra["trace"].fork()`, then `fork.do(*pair)` for each pair in order, reproduces
  the measured effect exactly.

```python
import torch
import transformers
import torchlens as tl
import torchlens.mechinterp

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=64, n_positions=32)
model = transformers.GPT2LMHeadModel(config).eval()
model.set_attn_implementation("eager")
clean, corrupted = torch.randint(0, 64, (3, 6)), torch.randint(0, 64, (3, 6))


def metric(logits):
    return logits[:, -1, 1] - logits[:, -1, 2]


claim = torchlens.mechinterp.edge_scores(model, clean, corrupted_input=corrupted, metric=metric)
largest = sorted(claim.rows(), key=lambda row: -abs(row.score))[:6]
audit = tl.attribution.causal_audit(
    model,
    clean,
    claim=claim,
    corrupted_input=corrupted,
    metric=metric,
    edges=[row.address[0] for row in largest],
    k=2,
    num_random_draws=1,
)
for row in audit.indices:
    print(row.component, round(row.claimed_effect, 4), round(row.measured_effect, 4))
for item in audit.sets:
    print(item.kind, round(item.claimed_effect, 4), round(item.measured_effect, 4))
print(round(audit.rank_agreement, 3), len(audit.extra["sets_not_measured"]))
```

Output:

```text
input->a0.h1<v> 0.0009 0.0014
input->m0 -0.0025 0.0012
input->logits -0.0284 -0.0142
a0.h0->logits -0.0006 0.0005
a0.h1->logits 0.001 0.0014
m0->logits -0.0029 0.0019
top -0.0313 -0.0196
bottom 0.0003 0.0019
random -0.0313 -0.0196
0.314 0
```

Here the audit takes the six edges with the largest scores of a random one-layer model. The
largest, `input->logits`, is claimed at -0.0284 and moves the metric by -0.0142, half as much;
three of the six (`input->m0`, `a0.h0->logits`, `m0->logits`) move it the opposite way to their
claim, and rank agreement over the six is 0.314. All three sets are measured (the seeded random
draw happens to pick the top-2 edges again). The top-2 set, `input->logits` and `m0->logits`,
enters one receiver; the bottom-2 set, `a0.h0->logits` and `input->a0.h1<v>`, enters two and is
patched in one call.

Measured on two larger checks. On the deterministic nonlinear fixture of the edge-scoring tests
(two linear MLP senders that read the embedding, a tanh metric), every measured effect equals its
patch effect computed by hand, exactly in float64. There EAP-IG with 5 steps claims -0.066 for
both edges through the first MLP while patching either one moves the metric by +0.150, so the
claim has the wrong sign, and rank agreement over the five edges is 0.65. On GPT-2 small (16
indirect-object prompts, float32, CPU), 200 edges (the 100 with the largest EAP-IG scores and 100
drawn at random) measured by this audit match an independent plain-PyTorch brute-force patcher
within 1.2e-5 per edge, and the EAP-IG claims reach rank agreement 0.886 with the measured
effects. Those 200 patches took 792 s on 2 CPU cores, about 4 s each. The audit's 12 sets of five
edges (top-5, bottom-5 and ten random draws) enter three receivers (the top set) or five (the other
eleven), and each is patched in one call: every set matches the plain-PyTorch patcher run on the same
edges within 1e-5, the claimed set sums reach rank agreement 0.993 with the measured set effects,
and the 12 set patches took about 30 s each on 4 CPU cores.

## Reading attribution results honestly

An attribution map answers a narrower question than it appears to. Five habits keep a reading
honest:

- **Know whether the method is signed.** `saliency` and `smoothgrad` return absolute gradients
  and have no `signed=` switch, so they say how strongly the target responds to a feature, never
  in which direction. `input_x_grad`, `integrated_gradients`, the layer methods and
  `guided_backprop`/`deconvolution` (with the default `absolute=False`) are signed.
- **Name the baseline.** `input_x_grad` measures against an implicit all-zeros input,
  `integrated_gradients` defaults to a zeros baseline, and `occlusion` and `occlusion_map` default
  to filling with zeros. Zero is often not a neutral input (a normalized image's zero is mean
  grey, a zero token embedding is no token), so a large score can measure the model's reaction to
  an off-distribution value. Pass the baseline you mean, prefer `"mean"` or a donor patch
  (`trace.fork().do(selection, tl.patch_from(donor))`, or the population edits on the
  [stochastic and population edits](../stochastic_edits.md) page) over zeros, and report the
  baseline with the figure: it rides `extra["baseline"]` (`extra["replacement_policy"]` for
  `occlusion_map`).
- **Check the quadrature.** Path methods approximate an integral with the midpoint Riemann rule
  at `n_steps` points. Captum's default rule is Gauss-Legendre, so the two libraries give close
  but not identical values at the same `n_steps`. Compare them, and choose `n_steps`, through the
  completeness fields: `extra["residual_rel"]` is the gap between the attributions' sum and the
  output change, relative to `extra["target_delta_abs"]`. A small ratio over a near-zero
  `target_delta_abs` certifies nothing.
- **Sanity-check guided maps.** Guided backpropagation and deconvolution give sharp maps that
  change little when the model's weights are randomized (Adebayo et al., 2018, "Sanity Checks
  for Saliency Maps"). Re-run the method on a copy with re-initialized weights; if the map does
  not change, it shows the input, not the model.
- **Read the logit lens by depth.** Projecting early layers through the final norm and
  unembedding reads them in a basis trained for the last layer, so early-layer top tokens are
  weak evidence; see [the logit lens](../facets.md#unembedding-head-and-logit-lens).

The completeness residual shrinks as `n_steps` grows:

```python
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2)).eval()
x = torch.randn(1, 3)
coarse = tl.attribution.integrated_gradients(model, x, target=0, num_steps=2)
fine = tl.attribution.integrated_gradients(model, x, target=0, num_steps=64)
assert fine.extra["residual_rel"] < coarse.extra["residual_rel"]
assert coarse.extra["baseline"].abs().sum() == 0  # the default zeros baseline, disclosed
```

## One-backward reads (`read`, `seed`, `ReadTable`, `load_read_table`)

`tl.attribution.read(trace, target=..., within=..., frozen=..., method=...,
reduce=..., target_batch_size=..., result_byte_budget=...)` is the
one-capture one-backward attribution read: one suppressed `autograd.grad`
over every addressable site, returning an immutable
`tl.attribution.ReadTable` with per-row honesty columns.
`tl.attribution.seed(site, index=...)` / `seed(site, cotangent=...)` build
the primary edge-seeded targets (no payload retention needed), and
`tl.attribution.load_read_table(path)` loads a persisted scalar
`read_table_v1` artifact back (fail-closed; loaded tables are
`rescorable=False`). Every spelling is DOCUMENTED-UNSTABLE pending the
naming sprint; the doc of record is
[`onebackward_reads.md`](onebackward_reads.md).

## Attribution graphs (`build_graph`, `AttributionGraph`)

`tl.attribution.build_graph(trace, transcoders, *, embedding, target=None, max_num_logits=10,
logit_probability_coverage=0.95, demeaned_logits=True, logits_at=None, frozen)` builds the attribution
graph of one prompt's capture: the direct linear effects between transcoder features, error
nodes, token embeddings and target logits, with transcoder features in place of MLP outputs and
attention patterns and normalisation scales held at their captured values. Documented-unstable.

- `transcoders` maps each MLP module name to a transcoder: an object with `encode(x)`, `W_enc`
  `(d_in, num_features)`, `b_enc`, `W_dec` `(num_features, d_out)` and `b_dec`, the object
  `tl.splice_module(transcoder, input="in")` splices in for that MLP. A cross-layer transcoder
  gives `W_dec` and `b_dec` as mappings from MLP module names to tensors, one per MLP output its
  features write.
- `frozen` names the modules (or op labels) held at their captured values: every attention
  pattern and normalisation scale, so each edge is a direct linear effect. It is required;
  `frozen=()` is the spelling for a model with neither. Every MLP output a transcoder writes is
  always held, and its feature and error nodes explain it.
- `target=None` picks the most probable logits at the last position until their probabilities
  reach `logit_probability_coverage`, at most `max_num_logits`; a list of token ids names them. With
  `demeaned_logits=True` a logit node is its logit minus the mean logit. `logits_at` takes a
  `SiteTarget` naming the module whose output holds the logits (for example the logits before a
  final softcap).

The result, `tl.attribution.AttributionGraph`, has one `GraphNode` (`kind`, `site`, `position`,
`feature_index`, `token`) per row and column of `adjacency_matrix`, in the order features, errors,
tokens, logits; rows are targets and columns sources. `node_values` holds each feature's
pre-activation and each logit node's value. Every feature and logit node is accounted for: its
row of `adjacency_matrix` plus `constants["encoder_bias"]`, `constants["decoder_bias"]` and
`constants["other"]` equals its value. `other` is what is not a node (the model's own biases,
position embeddings); a large `other` on a node means a path to it bypasses the graph, for
example an attention pattern left unfrozen. `error_vectors` holds each MLP output minus its
reconstruction, `active_feature_indices` and `feature_activations` the feature nodes, and
`capture_outcome` and `disclosures` what the capture's completeness verdict says.

```python
import json
from pathlib import Path

import torch
from torch import nn

import torchlens as tl


class Tiny(nn.Module):
    """Embedding, one residual MLP, unembedding."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(6, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        self.unembed = nn.Linear(4, 6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        return self.unembed(x + self.mlp(x))


class Transcoder:
    """A transcoder equal to the MLP, in the SAE Lens orientation."""

    def __init__(self, mlp: nn.Sequential) -> None:
        self.W_enc = mlp[0].weight.detach().T
        self.b_enc = mlp[0].bias.detach()
        self.W_dec = mlp[2].weight.detach().T
        self.b_dec = mlp[2].bias.detach()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)


torch.manual_seed(0)
model = Tiny().eval()
trace = tl.trace(model, torch.tensor([[1, 4, 2]]))
graph = tl.attribution.build_graph(
    trace, {"mlp": Transcoder(model.mlp)}, embedding_module="embed", frozen=(), target=[3]
)
print(graph.node_counts)
logit = graph.node_indices("logit")[0]
strongest = int(graph.adjacency_matrix[logit].abs().argmax())
print(graph.nodes[strongest])
assert graph.error_vectors.abs().max() < 1e-5  # the transcoder is the MLP itself


class Letters:
    """A stand-in tokenizer: token id i decodes to the i-th letter."""

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(ord("a") + i) for i in ids)


path = Path(globals().get("DOCS_TMPDIR", "/tmp")) / "tiny-graph.json"
exported = graph.to_neuronpedia(slug="tiny-graph", model_id="tiny", tokenizer=Letters(), path=path)
print(exported["metadata"]["prompt_tokens"], len(exported["nodes"]), len(exported["links"]))
assert json.loads(path.read_text()) == exported

color_by = tl.attribution.overlay(trace, graph)  # each node's influence, totalled per site
painted = [label for label in trace.layer_labels if color_by(trace[label]) is not None]
print(painted)
```

### Exporting to Neuronpedia (`AttributionGraph.to_neuronpedia`)

`graph.to_neuronpedia(*, slug, scan, tokenizer, prompt=None, node_influence_coverage=0.8,
edge_influence_coverage=0.98, path=None)` returns the graph, pruned, as a JSON-ready dict in
Neuronpedia's published attribution-graph schema, and writes it to `path` when one is given.
Documented-unstable. The schema is `apps/webapp/app/api/graph/graph-schema.json` in
[Neuronpedia's repository](https://github.com/hijohnnylin/neuronpedia) at commit
`2f0bb10893b9c63c409dbc49db61f557883e0a2c` (JSON Schema draft-07, "Anthropic Attribution
Graph" 1.0.0), the same schema as Neuronpedia's [graph validator](https://www.neuronpedia.org/graph/validator).
Node ids, node fields and the direction of links follow circuit-tracer's own writer, so the file
opens in Neuronpedia's graph viewer the way a circuit-tracer graph does.

- `slug` is the graph's name on Neuronpedia and `model_id` is Neuronpedia's model id (for example
  `"gemma-2-2b"`); the graph stores neither.
- `tokenizer` is anything with `decode(ids) -> str`, usually the model's Hugging Face
  tokenizer: it turns each input token and each target logit into the strings the viewer shows.
  `prompt=None` decodes the input tokens together.
- Pruning is circuit-tracer's rule. A node's influence is its total effect on the logits,
  weighted by their probabilities, through the row-normalised absolute adjacency. Nodes are kept
  in order of influence until `node_influence_coverage` of the total is covered, edges likewise until
  `edge_influence_coverage`; token and logit nodes are always kept, and feature and error nodes left
  without kept edges are dropped. A node's `influence` field is the share of the total covered
  up to and including it. Nodes at every position are exported, the first (often BOS) included.
- A feature's `layer` is its transcoder's index in `transcoder_sites`, an error node's is its
  site's index in `error_sites`, and logit nodes sit one layer past the last. A graph with a
  different number of transcoders than MLP outputs is refused
  (`attribution_graph_export_unsupported`), as are graphs without token ids, with NaN or
  infinite values, or whose logits nothing influences; keyword values the schema cannot carry
  raise `attribution_graph_export_argument_invalid`.

### Drawing a graph on its trace (`overlay`)

`tl.attribution.overlay(trace, graph, *, reduce="abs_sum", graph_value=None)` paints an
attribution graph onto the trace it was built from, for `trace.draw(color_by=...)`.
Documented-unstable. Each site holding feature or error nodes (the MLP a feature's transcoder
reads, the MLP output an error node completes) gets the `reduce` total of its nodes' values
over every position, on that module's output nodes; every other node stays unencoded.

- `graph_value=None` (the same as `"influence"`) paints each node's influence on the logits,
  the quantity `to_neuronpedia` prunes by; `graph_value="activation"` paints the feature
  activations, so sites holding only error nodes stay unencoded. Token and logit nodes paint
  nothing.
- A graph built from another trace is refused (`attribution_overlay_graph_mismatch`), as is a
  `graph_value` that is unknown or given with a source that is not a graph
  (`attribution_overlay_argument_invalid`).

The example above ends with the overlay; `trace.draw(color_by=color_by, path="graph")`
renders it.

### Uploading to Neuronpedia (`upload_to_neuronpedia`)

`tl.attribution.graph.upload_to_neuronpedia(exported, *, api_key)` publishes an exported graph
on neuronpedia.org under the account that owns `api_key`, and returns the URL of its page.
Documented-unstable. `exported` is the dict `to_neuronpedia` returned or the path of the file
it wrote. It follows Neuronpedia's documented two-step upload
([API reference](https://www.neuronpedia.org/api-doc), "Upload Graph 1/2" and "Upload Graph
2/2"): `POST /api/graph/signed-put` returns a signed upload address, the JSON is sent there,
and `POST /api/graph/save-to-db` saves the graph under its `model_id` and `slug`. It uses the
standard library only.

- The key is sent in the `x-api-key` header (Neuronpedia shows it under Settings) and is never
  written into a message, `repr` or log.
- Before any request the graph is checked against the export's structure
  (`attribution_graph_contract_invalid`), its `slug` and `model_id` against the characters the
  server takes (letters, digits, `_`, `.`, `-`), and its size against the server's limits of
  1,024 to 104,857,600 bytes (`neuronpedia_upload_argument_invalid`, also raised for a missing
  key or an unreadable file).
- An HTTP error or an unreachable host raises `neuronpedia_upload_http_error`, naming the step
  and the server's answer; an answer without its documented fields raises
  `neuronpedia_upload_response_invalid`.

`direction.upload_to_neuronpedia(*, api_key, scan, vector_label, layer=None,
hook_type="resid_post", default_steering_strength=1.0)` publishes a `tl.intervention.Direction` the same
way through Neuronpedia's "Create Vector" endpoint (`POST /api/vector/new`). `layer` defaults to
the trailing index of the direction's site label, as `to_repeng` reads it; `hook_type` is
`"resid_post"` (a decoder block's output) or `"resid_pre"` (its input); `default_steering_strength` is the
default steering strength Neuronpedia offers, from -100 to 100. Documented-unstable.

```text
# Publishes the graph on neuronpedia.org under your account; continues the example above.
import os

url = tl.attribution.graph.upload_to_neuronpedia(path, api_key=os.environ["NEURONPEDIA_API_KEY"])
print(url)
```

## The F06 attribution kit (DOCUMENTED-UNSTABLE)

The nine-method completion of the kit (attrib panel memo, 2026-08-26). Every spelling below is
documented-unstable pending the naming session; the inventory is locked in the
[glossary index](glossary.md#documented-unstable-attribution-kit-index). Disclosure is the
product: every path-method result carries `attribution_sum`, `target_delta`,
`completeness_residual`, `residual_rel`, and `target_delta_abs` (a relative certificate over a
near-zero output change certifies nothing, so the denominator always rides beside the ratio).

### The wrapping contract: `noise_tunnel` and the two routes

`tl.attribution.noise_tunnel(inputs, input_kwargs=None, *, ...)` runs ANY kit method on noised
copies of the input and aggregates (`"mean"`, `"mean_square"`, population `"variance"`). Two
routes, no registry: the PRIMITIVE `attribute=` (a closed callable
`(inputs, input_kwargs) -> Attribution`; bind settings with `functools.partial`) and the
SUGAR `method=` (any kit-contract callable, plus `model=`, `target=`, and ONE explicit
`method_kwargs=dict(...)` mapping echoed verbatim into the result). The keyword splat is dead:
a wrapper and its child can both own `num_samples` and `seed`, and only an explicit mapping can
say which is which. `stdevs` is ABSOLUTE and required; integer ids, masks, booleans, and
strings pass through unnoised; repeated tensor references share ONE noised object.
`noise_tunnel(method=smoothgrad)` refuses (SmoothGrad IS the tunnel over `saliency` with
per-sample absolute values -- `smoothgrad` itself is now the thin alias);
`noise_tunnel(method=gradient_shap)` works with derived per-sample child seeds and both sample
counts disclosed; trace-bound methods refuse (a static Trace leaves nothing to perturb).

### `gradient_shap`

`tl.attribution.gradient_shap(model, inputs, input_kwargs=None, *, target, baselines, ...)`
estimates expected gradients against a REQUIRED baseline pool (leading pool axis; one pool
index drawn per example per sample, shared across every attributed leaf). The estimator is
pinned to the reference `InputBaselineXGradient` convention -- gradient at the interpolated
point times (noised input minus drawn baseline) -- by a stored-draw oracle. The residual
disclosure is labeled Monte-Carlo DIAGNOSTICS, never a completeness guarantee. Defaults:
`num_samples=25`, `stdevs=0.0`; full draws stored only on request (`store_draws=True`).

### `guided_backprop` and `deconvolution`

Strict exact-ReLU only, at OP level: module-dispatched, functional, in-place, and reused
firings are all rewritten (`sites="all"`, the default); `sites="module"` restricts to
`nn.ReLU`-dispatched firings (the module-hook coverage class, reference-parity-pinned).
Results are SIGNED by default (`absolute=False`). Zero matched sites refuse naming the
model's observed activation kinds; GELU/SiLU/LeakyReLU/ReLU6/Hardtanh/softmax are excluded by
definition. In-place ReLU spellings are served out-of-place with a bit-exact forward-fidelity
tripwire. On torchvision's DenseNet-121, restricted coverage reproduces the module-hook
reference exactly while op-level coverage differs -- the difference is the one functional
`F.relu` in torchvision's own `forward` (the `site_census` in every result carries the
counts). There is deliberately NO spelling that mounts these rules on `integrated_gradients`,
`layer_integrated_gradients`, `layer_conductance`, or `gradient_shap`.

### `occlusion_map`

`tl.attribution.occlusion_map(model, inputs, input_kwargs=None, *, target, window, ...)` --
the direct engine: eval/no-grad forwards, one per window, sweeping one attributed leaf
(`occlude_leaf=`) with every other input held fixed. Clipped-edge full-coverage window
enumeration (origins at `k * stride`, final window clips; reference-grid identical),
`overlap="average"` (default) or `"sum"`, replacement policies `"zeros"`/`"mean"`/scalar/
tensor. The pass budget is DETERMINISTIC (`max_passes=4096` direct tier; the trace-sweep tier
is 512), never wall-clock: the refusal carries the computed pass count, one measured pass
time, and the projected total. Int targets produce per-example maps; callable targets
aggregate (disclosed).

### `infidelity` and `sensitivity`

`tl.attribution.infidelity(model, inputs, input_kwargs=None, *, target, attribution, ...)`
measures `E[(sum(attr * dx) - (f(x) - f(x - dx)))^2]` with a USABLE named default
(`perturb="gaussian"`, `noise_std=0.003`, `num_samples=10`), the named `"square_removal"`, and a
fully compatible callable escape. Both unnormalized (primary) and normalized values return in
one frozen `AttributionMetric`; out-of-range perturbations WARN (coded); layer-space/CAM-shaped
values refuse with the input-space expansion recipe. Infidelity judges the perturbation as
much as the attribution. "Lower is better" only under identical target / baseline /
perturbation / radius / norm / normalization / sample bank -- the qualification rides in every
result. `tl.attribution.sensitivity(...)` (max relative attribution change under uniform
L-infinity input noise, `radius=0.02`) enforces the determinism ladder: a known-stochastic
method without a seed refuses BEFORE work; an opaque callable disclosing unseeded stochastic
provenance refuses; a fully opaque callable is probed twice on the unperturbed inputs
(relative tolerance 1e-6 -- bit-equality is banned as the comparator) and equality records
`determinism_probe="passed_not_proven"`.

### `text` -- the token-attribution two-liner

```text
result = tl.attribution.text(model, tokenizer, "The Eiffel Tower is in", target=" Paris")
result.show()
```

To save the rendering to a file, write the payload's HTML:
`open("tokens.html", "w").write(result.payload().to_html())`. Per-step attribution over
generated text is a recipe: [per-step attribution for generated text](../recipes/generation_step_attribution.md).

Input Integrated Gradients through HF `inputs_embeds` with every integer input held fixed;
signed per-token scores sum-reduce the embedding width so completeness accounting survives.
`baseline="auto"` is TASK-AWARE and always resolves to a printed concrete baseline: decoders
zero all prompt-token embeddings; encoders with a reliable pad token and special-token mask
pad content tokens while scaffolding specials at their true embeddings
(`keep_special_tokens=True`); anything unreliable falls back to zeros-all with a NAMED
disclosure (a missing pad token is never guessed). `n_steps=128` fixed default;
`num_steps="auto"` runs the 64-128-256-512 ladder stopping ONLY on the dual criterion
(residual AND max(L1, L2) successive-grid stability, both <= 1%; rank stability is banned as
a signal). `converged=False` is a first-class outcome with a footer line the user cannot
miss. A bare int target is the vocab id at the final non-padding position, warning in the one
band where it is also a valid position index. `steps_per_batch=8` rides the randomized
batching audit. Rendering: the typed `TokenAttributionPayload` (zero-centered diverging score
domain, mandatory footer lines) feeds the future tviz renderer; until then `show()` returns
the escaped-table fallback. `TokenAttribution` freezes text, ids, raw + display tokens
(wordpieces are never silently merged), offsets, masks, scores, the full `(L, D)` values, the
exact baseline, provenance, truncation (explicit only, warned), steps, audit evidence, cost,
and the convergence record.

### `SiteStash` and the epsilon-LRP litmus

`tl.attribution.SiteStash` is the ENTIRE permanent LRP-adjacent surface: a forward/backward
pairing store (`stash`/`fetch`, LIFO pairing proven on reused modules, leftovers disclosed).
There is no rule registry, no composites, no canonizers -- the cleanroom epsilon-LRP recipe
with its per-site conservation ledger lives in
[`docs/recipes/lrp_epsilon_litmus.md`](../recipes/lrp_epsilon_litmus.md).

### IG step batching (`step_batch_size`, `step_audit`)

`integrated_gradients`, `layer_integrated_gradients`, and `layer_conductance` accept
`step_batch_size=` (path points stacked on the ordinary batch axis -- a THROUGHPUT feature,
not a memory feature; strictly opt-in, default sequential) guarded by the randomized, seeded,
disclosed audit ladder `step_audit=` (`"per_call"` default under batching / `"per_chunk"` /
explicit `"off"`). The audit is a sampled test, never a proof -- the only proof is sequential
execution; a fixed-index audit is defeatable by construction, so the audited (chunk, row) is
randomized (`step_audit_seed=` pins it). Every result disclosed the logical path-evaluation
count, the physical forward-call count, and the audit record. Captum-comparison note: our
path methods are midpoint Riemann; the reference default Gauss-Legendre differs (measured
22.5% at n=16), so cross-tool comparisons must request `method="riemann_middle"` there.
