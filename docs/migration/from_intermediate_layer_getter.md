# Coming from torchvision's `IntermediateLayerGetter`

This page maps torchvision's `IntermediateLayerGetter` (in `torchvision.models._utils`,
used by its detection and segmentation backbones) to TorchLens: their spelling on the
left, the TorchLens spelling on the right.

| `IntermediateLayerGetter` | TorchLens | Notes |
| --- | --- | --- |
| `IntermediateLayerGetter(model, return_layers={"layer2": "out"})` | `tl.extract(model, x, {"out": "layer2"})` | Key first, then the address: the mapping runs the other way round. |
| Only direct children of the model can be returned | Any module address (`"layer2.1.conv2"`) or op label | Nested modules and functional ops are lookups too. |
| Builds a new module that drops every child after the last requested one | Runs the whole, unmodified model | Your model is never rebuilt; ask `trace.run(inputs=x, until=...)` for a truncated run. |
| Children must be called in registration order, once each | Any forward, including reused modules and loops | Repeated calls of one module get one pass each. |

The example below asks for `layer2`, a module in the middle of the network, so the
getter's truncation matters: it never runs `layer3`, `layer4` or the classifier, while
TorchLens runs the whole model and returns the same `layer2` tensor.

Their construct:

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

TorchLens equivalent:

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

See also [Multi-GPU extraction](multi_gpu_extraction.md).
