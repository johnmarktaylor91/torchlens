# Migrating from TorchExtractor

Functional migration pattern: TorchExtractor's module-output dictionary maps to a TorchLens
completed log with layer labels and saved activations.

| TorchExtractor construct | TorchLens equivalent |
| --- | --- |
| Wrap a model and return selected module outputs. | Capture the forward and read selected layer activations. |

Their construct:

```python
# migration-test: tool=torchextractor expected=[[2.5, 2.5]]
import torch
from torch import nn
import torchextractor as tx


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


model = tx.Extractor(Tiny(), ["proj"])
_, features = model(torch.tensor([[2.0, 3.0]]))
RESULT = features["proj"].detach().tolist()
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
