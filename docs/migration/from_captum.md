# Migrating from Captum

Functional migration pattern: Captum layer-activation collection maps to a TorchLens capture and
module/label lookup. Keep Captum for attribution algorithms such as Integrated Gradients.

| Captum construct | TorchLens equivalent |
| --- | --- |
| `LayerActivation(model, layer).attribute(x)`. | `log_forward_pass(model, x)` and read the layer activation. |

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


log = tl.log_forward_pass(Tiny(), torch.tensor([[2.0, 3.0]]), vis_opt="none")
RESULT = log["linear_1_1"].activation.detach().tolist()
```
