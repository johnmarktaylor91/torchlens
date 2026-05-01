# Migrating from NNsight

Functional migration pattern: an NNsight trace block that saves one module value maps to a
TorchLens capture followed by label/module lookup on the completed log.

| NNsight construct | TorchLens equivalent |
| --- | --- |
| Save an intermediate value during a trace. | Capture the eager forward, then read the saved activation. |

Their construct:

```python
# migration-test: tool=nnsight expected=[[2.5, 2.5]]
import torch
from torch import nn
from nnsight import LanguageModel


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


model = LanguageModel(Tiny(), dispatch=True)
x = torch.tensor([[2.0, 3.0]])
with model.trace(x):
    saved = model.proj.output.save()

RESULT = saved.value.detach().tolist()
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


model = Tiny()
x = torch.tensor([[2.0, 3.0]])
log = tl.log_forward_pass(model, x, vis_opt="none")
RESULT = log["linear_1_1"].activation.detach().tolist()
```
