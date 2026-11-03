# Coming from surgeon-pytorch

This page is for people who use surgeon-pytorch's `Inspect` and `Extract` wrappers to read
intermediate activations or run part of a model. TorchLens's test environment does not pin
surgeon-pytorch; its example skips where surgeon-pytorch is not installed. Checked 2026-11-03.

## The same task in both

Read the output of `layer1` of a ResNet-18 and backpropagate a loss built from it. This is the
case that needs the gradient caveat below: surgeon-pytorch hands back activations that are still
attached to the autograd graph, so a loss on them backpropagates into the model.

### In surgeon-pytorch

```python
# migration-test: tool=surgeon_pytorch expected=([1, 64, 8, 8], True)
import torch
from torchvision.models import resnet18
from surgeon_pytorch import Inspect

torch.manual_seed(0)
model = resnet18(weights=None).eval()
wrapped = Inspect(model, layer="layer1")
y, act = wrapped(torch.randn(1, 3, 32, 32))
act.sum().backward()
RESULT = (list(act.shape), model.conv1.weight.grad is not None)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=([1, 64, 8, 8], True, True)
import torch
from torchvision.models import resnet18
import torchlens as tl

torch.manual_seed(0)
model = resnet18(weights=None).eval()
x = torch.randn(1, 3, 32, 32)
act = tl.extract(model, x, ["layer1"], detach_saved_activations=False)["layer1"]
act.sum().backward()
RESULT = (list(act.shape), act.grad_fn is not None, model.conv1.weight.grad is not None)
```

## Translation table

| surgeon-pytorch | TorchLens | Notes |
| --- | --- | --- |
| `Inspect(model, layer="layer1")`, then `y, act = wrapped(x)` | `tl.extract(model, x, ["layer1"])` | Pass `detach_saved_activations=False` to keep gradients (see below). |
| `Inspect(model, layer=["layer1", "layer2"])` | `tl.extract(model, x, ["layer1", "layer2"])` | Returns a dict keyed by the lookups you passed. |
| `get_layers(model)` | `[m.address for m in tl.trace(model, x).modules]` or `model.named_modules()` | |
| `Extract(model, node_out="...")` and `get_nodes(model)` (FX graph nodes) | `tl.trace(model, x)` then any op label from `trace.layers` | Functional ops are lookups without FX tracing. |
| `Extract(model, node_in=..., node_out=...)` (run a sub-graph) | `trace.run(inputs=x, until=...)` stops at a site | TorchLens never rebuilds the model; the full sub-graph rewrite stays in surgeon-pytorch. |

### The gradient caveat

`tl.extract` returns detached copies by default. To get the surgeon-pytorch behavior, pass
`detach_saved_activations=False`: the returned tensors then carry `grad_fn` and a loss built
from them reaches the model's parameters. The TorchLens example above checks exactly that.

## What TorchLens adds

- Op-level lookups with no FX tracing: any op label in `tl.trace(model, x).layers`, functional
  calls included, where `Extract` works on FX graph nodes.
- Control flow runs as written, since TorchLens records the eager forward instead of a
  symbolic graph.
- `tl.extract_dataset` runs the same lookups over a dataloader and writes the activations to
  disk; see [Multi-GPU extraction](multi_gpu_extraction.md).
- `trace.run(inputs=x, until=...)` stops at a site, without building a new module.

## What surgeon-pytorch does that TorchLens does not

- `Extract` returns a new module that runs only the sub-graph between two FX nodes
  (`node_in` to `node_out`), which you can call on your own intermediate input. TorchLens
  never rebuilds the model; to feed a tensor in mid-network you edit a site in a fork of a
  trace (see [Intervention explainers](../intervention_explainers.md)).
- `Inspect` returns the output and the activations together from one wrapped call. TorchLens
  returns the activations; read the model's output from the trace or from your own call.

See [where to go instead](where_to_go_instead.md) for tools outside TorchLens's scope.

## Using both

### surgeon-pytorch feeds TorchLens

The model you wrap with `Inspect` is an `nn.Module`; pass the original to `tl.extract` or
`tl.trace`. TorchLens has no bridge for surgeon-pytorch's wrapper modules.

### TorchLens feeds surgeon-pytorch

There is no bridge. Module addresses from `trace.modules` are the strings `Inspect` takes as
`layer`, so TorchLens can help you find them; FX node names for `Extract` come from
`get_nodes(model)`.

## Semantic differences

- **Gradients.** surgeon-pytorch's activations stay attached to autograd; `tl.extract`'s are
  detached unless you pass `detach_saved_activations=False`.
- **Result shape.** `Inspect` returns a tuple `(output, activations)`; `tl.extract` returns a
  dict of activations keyed by the lookups you passed.
- **Whole model runs.** `tl.extract` runs the unmodified model; `Extract` runs a rebuilt
  sub-graph.
