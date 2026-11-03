# Coming from torchvision's feature extractor

This page maps torchvision's `create_feature_extractor` to TorchLens: their spelling on
the left, the TorchLens spelling on the right.

| torchvision | TorchLens | Notes |
| --- | --- | --- |
| `create_feature_extractor(model, return_nodes=["layer1"])` | `tl.extract(model, x, ["layer1"])` | Module addresses are accepted as they are. |
| `return_nodes={"layer1": "feat1"}` (node name to output key) | `tl.extract(model, x, {"feat1": "layer1"})` (output key to lookup) | The mapping runs the other way round: key first, then the address. |
| `get_graph_node_names(model)` | `tl.trace(model, x).layers` and `.modules` | TorchLens lists every executed op and module call, not a symbolic graph. |
| FX node names such as `"layer1.0.relu_1"` | Op labels read from a trace (`trace.layers`) | Any op, including functional calls such as `torch.flatten`, is a lookup. |
| Rebuilt `GraphModule` you call repeatedly | One call per batch, or `tl.extract_dataset` for many batches | Nothing is rewritten: the model you pass is the model that runs. |
| Fails on data-dependent control flow (FX symbolic tracing) | Runs the eager forward as written | Branches, loops and Python logic run normally. |

`tl.extract` returns detached tensors by default. Pass
`detach_saved_activations=False` when you want to backpropagate through the returned
activations.

Their construct:

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

TorchLens equivalent:

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

For many batches, `tl.extract_dataset` runs the same lookup over a dataloader and writes
the activations to disk. See also [Multi-GPU extraction](multi_gpu_extraction.md).
