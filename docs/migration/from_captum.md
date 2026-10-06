# Migrating from Captum

Functional migration pattern: Captum layer-activation collection maps to a TorchLens capture and
module/label lookup. TorchLens also ships its own attribution methods under `tl.attribution`
(saliency, input x gradient, integrated gradients, SmoothGrad, noise tunnel, gradient SHAP,
guided backprop, deconvolution, occlusion, Grad-CAM, layer attribution, layer conductance, layer
integrated gradients, per-token text attribution, and the infidelity and sensitivity metrics); see
[the attribution reference](../reference/attribution.md). Keep Captum when you need a method
TorchLens does not ship (for example DeepLift, LIME, Kernel SHAP or feature ablation as an
attribution method), its tuned defaults, or its paper-validated implementation details.

| Captum construct | TorchLens equivalent |
| --- | --- |
| `LayerActivation(model, layer).attribute(x)`. | `trace(model, x)` and read the layer activation. |

Their construct:

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

TorchLens equivalent:

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
