# Coming from torchvision's feature extractor

This page is for people who use torchvision's `create_feature_extractor` to read intermediate
activations and want the same activations from TorchLens. The examples were written against
the torchvision range TorchLens's test environment installs (`torchvision>=0.18,<1`).
Checked 2026-11-03.

## The same task in both

Read the output of `layer1` of a ResNet-18 as one named feature.

### In torchvision

```python
# migration-test: tool=torchvision expected=[1, 64, 8, 8]
import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

torch.manual_seed(0)
model = resnet18(weights=None).eval()
extractor = create_feature_extractor(model, return_nodes={"layer1": "feat1"})
with torch.no_grad():
    features = extractor(torch.randn(1, 3, 32, 32))
RESULT = list(features["feat1"].shape)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=([1, 64, 8, 8], True)
import torch
from torchvision.models import resnet18
import torchlens as tl

torch.manual_seed(0)
model = resnet18(weights=None).eval()
x = torch.randn(1, 3, 32, 32)
features = tl.extract(model, x, {"feat1": "layer1"})

# The same tensor a forward hook on model.layer1 sees.
seen = {}
handle = model.layer1.register_forward_hook(lambda m, i, out: seen.setdefault("out", out))
with torch.no_grad():
    model(x)
handle.remove()
RESULT = (list(features["feat1"].shape), torch.equal(features["feat1"], seen["out"]))
```

## Translation table

| torchvision | TorchLens | Notes |
| --- | --- | --- |
| `create_feature_extractor(model, return_nodes=["layer1"])` | `tl.extract(model, x, ["layer1"])` | Module addresses are accepted as they are. |
| `return_nodes={"layer1": "feat1"}` (node name to output key) | `tl.extract(model, x, {"feat1": "layer1"})` (output key to lookup) | The mapping runs the other way round: key first, then the address. |
| `get_graph_node_names(model)` | `tl.trace(model, x).layers` and `.modules` | TorchLens lists every executed op and module call, not a symbolic graph. |
| FX node names such as `"layer1.0.relu_1"` | Op labels read from a trace (`trace.layers`) | Any op, including functional calls such as `torch.flatten`, is a lookup. |
| Rebuilt `GraphModule` you call repeatedly | One call per batch, or `tl.extract_dataset` for many batches | Nothing is rewritten: the model you pass is the model that runs. |
| Fails on data-dependent control flow (FX symbolic tracing) | Runs the eager forward as written | Branches, loops and Python logic run normally. |

`tl.extract` returns detached tensors by default. Pass `detach_saved_activations=False` when
you want to backpropagate through the returned activations.

## What TorchLens adds

- Any executed op is a lookup, not only the nodes FX names: `tl.trace(model, x).layers`
  lists them, including functional calls and the arithmetic inside blocks.
- Control flow stays as written. Data-dependent branches and loops run in the eager forward,
  and repeated calls of one module keep one pass each.
- The model you pass is the model that runs: no rebuilt `GraphModule`, so its parameters,
  hooks and `forward` stay yours.
- `tl.extract_dataset` runs the same lookup over a dataloader and writes the activations to
  disk with provenance; see [Multi-GPU extraction](multi_gpu_extraction.md) for sharded runs.
- `trace.run(inputs=x, until=...)` stops a run at a site when you do not want the rest of
  the model to execute.

## What torchvision's feature extractor does that TorchLens does not

- It returns a `torch.nn.Module` (a `GraphModule`) that you can call, fine-tune with a
  training loop, script or export as a truncated network. `tl.extract` returns tensors for
  one call; it does not return a smaller model.
- It drops the nodes after the last requested one, so later layers are not computed.
  `tl.extract` runs the whole model unless you stop it with `trace.run(inputs=x, until=...)`.
- It works from the symbolic graph without any input, which is what `get_graph_node_names`
  uses. TorchLens needs a real input to list what ran.

For tools that live outside TorchLens's scope, see [where to go instead](where_to_go_instead.md).

## Using both

### torchvision feeds TorchLens

A torchvision model is an ordinary `nn.Module`, so there is nothing to convert: pass
`resnet18(weights=...)` to `tl.extract` or `tl.trace` as the first example does. TorchLens has
no bridge for the extractor's `GraphModule` itself; give it the original model.

### TorchLens feeds torchvision

There is no bridge. The module addresses a trace lists (`trace.modules`) are the same strings
`return_nodes` takes for module outputs, so you can find a name with TorchLens and pass it to
`create_feature_extractor`. Functional ops have FX node names that TorchLens labels do not
reproduce.

## Semantic differences

- **Mapping direction.** torchvision's `return_nodes` maps node name to output key;
  `tl.extract` maps output key to lookup.
- **Detached by default.** `tl.extract` returns detached copies. torchvision's extractor
  returns tensors that stay attached to autograd when it runs with gradients enabled.
- **Whole model runs.** TorchLens runs the unmodified model; the tensor at `layer1` is the
  tensor a forward hook on `layer1` sees, as the second example checks.
- **Eager, not symbolic.** TorchLens records what a real input executed; it does not need
  the model to be FX-traceable.
