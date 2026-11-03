# Coming from torchvision's `IntermediateLayerGetter`

This page is for people who use `IntermediateLayerGetter` (in `torchvision.models._utils`,
the helper behind torchvision's detection and segmentation backbones) to return the outputs
of some of a model's direct children. The examples were written against the torchvision range
TorchLens's test environment installs (`torchvision>=0.18,<1`). Checked 2026-11-03.

## The same task in both

Return `layer2`, a module in the middle of ResNet-18. The getter's truncation matters here:
it never runs `layer3`, `layer4` or the classifier, while TorchLens runs the whole model and
returns the same `layer2` tensor.

### In torchvision

```python
# migration-test: tool=torchvision expected=[1, 128, 4, 4]
import torch
from torchvision.models import resnet18
from torchvision.models._utils import IntermediateLayerGetter

torch.manual_seed(0)
model = resnet18(weights=None).eval()
getter = IntermediateLayerGetter(model, return_layers={"layer2": "out"})
with torch.no_grad():
    features = getter(torch.randn(1, 3, 32, 32))
RESULT = list(features["out"].shape)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=([1, 128, 4, 4], True)
import torch
from torchvision.models import resnet18
import torchlens as tl

torch.manual_seed(0)
model = resnet18(weights=None).eval()
x = torch.randn(1, 3, 32, 32)
features = tl.extract(model, x, {"out": "layer2"})

# The same tensor a forward hook on model.layer2 sees.
seen = {}
handle = model.layer2.register_forward_hook(lambda m, i, out: seen.setdefault("out", out))
with torch.no_grad():
    model(x)
handle.remove()
RESULT = (list(features["out"].shape), torch.equal(features["out"], seen["out"]))
```

## Translation table

| `IntermediateLayerGetter` | TorchLens | Notes |
| --- | --- | --- |
| `IntermediateLayerGetter(model, return_layers={"layer2": "out"})` | `tl.extract(model, x, {"out": "layer2"})` | Key first, then the address: the mapping runs the other way round. |
| Only direct children of the model can be returned | Any module address (`"layer2.1.conv2"`) or op label | Nested modules and functional ops are lookups too. |
| Builds a new module that drops every child after the last requested one | Runs the whole, unmodified model | Your model is never rebuilt; ask `trace.run(inputs=x, until=...)` for a truncated run. |
| Children must be called in registration order, once each | Any forward, including reused modules and loops | Repeated calls of one module get one pass each. |

## What TorchLens adds

- Nested modules and functional ops are reachable by address or op label, where the getter
  stops at direct children (`tl.trace(model, x).modules` and `.layers` list them).
- Models whose `forward` is not a plain sequence of children work: the getter assumes the
  children run in registration order, once each; TorchLens records the forward as it ran.
- Several lookups can come from one call, and the same call works on a model with reused
  modules or loops, where each call is its own pass.
- `tl.extract_dataset` extends the lookup over a dataloader and writes the activations to
  disk; see [Multi-GPU extraction](multi_gpu_extraction.md).

## What `IntermediateLayerGetter` does that TorchLens does not

- It returns a new `nn.Module` with the later children removed, which you can train, script
  or export as a smaller network. `tl.extract` returns tensors for one call and leaves the
  model whole.
- Because it drops the later children, the work after the last requested one is never done.
  TorchLens runs everything unless you stop at a site with `trace.run(inputs=x, until=...)`.
- It needs no input to build: it works from the module's children.

See [where to go instead](where_to_go_instead.md) for tools outside TorchLens's scope.

## Using both

### torchvision feeds TorchLens

The model you pass to the getter is the model you pass to `tl.extract`; there is nothing to
convert. TorchLens has no bridge for the getter module itself (it is a rebuilt module); trace
the original model.

### TorchLens feeds torchvision

There is no bridge. Read the module addresses from `trace.modules` and use the ones that are
direct children as `return_layers` keys; deeper addresses are outside what the getter can
return.

## Semantic differences

- **Mapping direction.** The getter's `return_layers` maps child name to output key;
  `tl.extract` maps output key to lookup.
- **Whole model runs.** The getter's output comes from a truncated copy of the forward;
  TorchLens's comes from the full run of your own model. The two tensors are equal, as the
  second example checks.
- **Detached by default.** `tl.extract` returns detached copies; pass
  `detach_saved_activations=False` to backpropagate through them.
- **Passes.** A module called several times has one pass per call in a TorchLens trace; the
  getter returns one output per child.
