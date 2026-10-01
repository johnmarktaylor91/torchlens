# `tl.attribution` reference

`tl.attribution` provides input and intermediate-layer attribution methods directly on
PyTorch modules. Each call returns `AttributionResult(method, values, target_repr, extra)`.
`target` is an output-class index or a callable that maps model output to one scalar tensor;
the layer methods name a module from `dict(model.named_modules())`.

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
n_steps=50, baseline=None)` integrates input gradients along a straight path from `baseline`
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
print(tl.attribution.integrated_gradients(model, torch.tensor([[1.0, 2.0]]), target=0, n_steps=4))
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

`tl.attribution.smoothgrad(model, inputs, input_kwargs=None, *, target=..., n_samples=25,
noise_level=0.1, seed=None)` averages saliency across Gaussian-noised inputs. Supplying `seed`
makes the noise deterministic without changing the global torch RNG state.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Linear(2, 2, bias=False)
with torch.no_grad(): model.weight.copy_(torch.eye(2))
print(tl.attribution.smoothgrad(model, torch.tensor([[1.0, 2.0]]), target=0, n_samples=3, seed=0))
```

Output:

```text
AttributionResult(method='smoothgrad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['n_samples', 'noise_level', 'seed'])
```

## `grad_cam`

`tl.attribution.grad_cam(model, inputs, input_kwargs=None, *, target=..., layer=..., relu=True,
overlay=True, image=None, alpha=0.6, cmap="magma")`
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
AttributionResult(method='grad_cam', values=Tensor(shape=(1, 1, 2, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer', 'native_map_resolution', 'overlay', 'relu', 'rendered_map_resolution', 'upsampling'])
```

## `layer_integrated_gradients`

`tl.attribution.layer_integrated_gradients(model, inputs, input_kwargs=None, *, target=...,
layer=..., baseline=None, n_steps=50)` applies the integrated-gradients path rule to a named
intermediate activation. See [`integrated_gradients`](#integrated_gradients) for input-level IG.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_integrated_gradients(model, torch.ones(1, 2), target=0, layer="0", n_steps=4))
```

Output:

```text
AttributionResult(method='layer_integrated_gradients', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer', 'n_steps'])
```

## `layer_conductance`

`tl.attribution.layer_conductance(model, inputs, input_kwargs=None, *, target=..., layer=...,
baseline=None, n_steps=50)` decomposes input integrated gradients onto the selected hidden
units along the input path. It uses the same baseline and step controls as
[`layer_integrated_gradients`](#layer_integrated_gradients).

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
print(tl.attribution.layer_conductance(model, torch.ones(1, 2), target=0, layer="0", n_steps=4))
```

Output:

```text
AttributionResult(method='layer_conductance', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer', 'n_steps'])
```

## `overlay` — drawing attribution on the graph

`tl.attribution.overlay(trace, source, *, reduce="abs_sum")` bridges attribution numbers onto
the graph: it returns a callable for `Trace.draw(color_by=...)` that paints each attributed
layer's reduced magnitude onto its module-output node and leaves every other node unencoded
(the legend carries the `n/a = unencoded` note). `source` is one layer-scoped
`AttributionResult` (its `extra["layer"]` anchors it), an iterable of them, or an explicit
mapping from `model.named_modules()` name or trace layer label to a result, tensor, or real
scalar. `reduce` is closed vocabulary: `"abs_sum"` (default), `"abs_mean"`, `"sum"`, `"max"`.
Unknown keys and unknown reductions raise `AttributionError` naming the fix. A module fired
several times paints its per-layer total on every one of its nodes; per-pass splits are not
claimed.

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
trace.draw(color_by=tl.attribution.overlay(trace, results), vis_outpath="attribution_graph")
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
AttributionResult(method='layer_activation_x_grad', values=Tensor(shape=(1, 2), dtype=torch.float32, device='cpu'), target_repr='index=0', extra_keys=['layer'])
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
`(inputs, input_kwargs) -> AttributionResult`; bind settings with `functools.partial`) and the
SUGAR `method=` (any kit-contract callable, plus `model=`, `target=`, and ONE explicit
`method_kwargs=dict(...)` mapping echoed verbatim into the result). The keyword splat is dead:
a wrapper and its child can both own `n_samples` and `seed`, and only an explicit mapping can
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
`n_samples=25`, `stdevs=0.0`; full draws stored only on request (`store_draws=True`).

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
(`perturb="gaussian"`, `noise_std=0.003`, `n_samples=10`), the named `"square_removal"`, and a
fully compatible callable escape. Both unnormalized (primary) and normalized values return in
one frozen `MetricResult`; out-of-range perturbations WARN (coded); layer-space/CAM-shaped
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

```python
result = tl.attribution.text(model, tokenizer, "The Eiffel Tower is in", target=" Paris")
result.show()
```

Input Integrated Gradients through HF `inputs_embeds` with every integer input held fixed;
signed per-token scores sum-reduce the embedding width so completeness accounting survives.
`baseline="auto"` is TASK-AWARE and always resolves to a printed concrete baseline: decoders
zero all prompt-token embeddings; encoders with a reliable pad token and special-token mask
pad content tokens while scaffolding specials at their true embeddings
(`keep_special_tokens=True`); anything unreliable falls back to zeros-all with a NAMED
disclosure (a missing pad token is never guessed). `n_steps=128` fixed default;
`n_steps="auto"` runs the 64-128-256-512 ladder stopping ONLY on the dual criterion
(residual AND max(L1, L2) successive-grid stability, both <= 1%; rank stability is banned as
a signal). `converged=False` is a first-class outcome with a footer line the user cannot
miss. A bare int target is the vocab id at the final non-padding position, warning in the one
band where it is also a valid position index. `steps_per_batch=8` rides the randomized
batching audit. Rendering: the typed `TokenAttributionPayload` (zero-centered diverging score
domain, mandatory footer lines) feeds the future tviz renderer; until then `show()` returns
the escaped-table fallback. `TokenAttributionResult` freezes text, ids, raw + display tokens
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
