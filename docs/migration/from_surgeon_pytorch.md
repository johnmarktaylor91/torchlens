# Coming from surgeon-pytorch

This page maps surgeon-pytorch's `Inspect` and `Extract` wrappers to TorchLens: their
spelling on the left, the TorchLens spelling on the right.

| surgeon-pytorch | TorchLens | Notes |
| --- | --- | --- |
| `Inspect(model, layer="layer1")`, then `y, act = wrapped(x)` | `tl.extract(model, x, ["layer1"])` | Pass `detach_saved_activations=False` to keep gradients (see below). |
| `Inspect(model, layer=["layer1", "layer2"])` | `tl.extract(model, x, ["layer1", "layer2"])` | Returns a dict keyed by the lookups you passed. |
| `get_layers(model)` | `[m.address for m in tl.trace(model, x).modules]` or `model.named_modules()` | |
| `Extract(model, node_out="...")` and `get_nodes(model)` (FX graph nodes) | `tl.trace(model, x)` then any op label from `trace.layers` | Functional ops are lookups without FX tracing. |
| `Extract(model, node_in=..., node_out=...)` (run a sub-graph) | `trace.run(inputs=x, until=...)` stops at a site | TorchLens never rebuilds the model; the full sub-graph rewrite stays in surgeon-pytorch. |

## The gradient caveat

surgeon-pytorch hands back activations that are still attached to the autograd graph, so
a loss on them backpropagates into the model. `tl.extract` returns detached copies by
default. To get the surgeon-pytorch behavior, pass `detach_saved_activations=False`:
the returned tensors then carry `grad_fn` and a loss built from them reaches the model's
parameters. The TorchLens example below checks exactly that.

Their construct:

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

TorchLens equivalent:

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

See also [Multi-GPU extraction](multi_gpu_extraction.md).
