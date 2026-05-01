# Migrating from torch.fx

Functional migration pattern: FX symbolic tracing produces a static `GraphModule`; TorchLens runs the
eager model and records what actually executed for one concrete input.

| torch.fx construct | TorchLens equivalent |
| --- | --- |
| `symbolic_trace(model)` followed by `graph_module(x)`. | `log_forward_pass(model, x)` and inspect the eager-operation log. |

Their construct:

```python
# migration-test: tool=fx expected=[[2.5, 2.5]]
import torch
from torch import nn
from torch.fx import symbolic_trace


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


gm = symbolic_trace(Tiny())
RESULT = gm(torch.tensor([[2.0, 3.0]])).detach().tolist()
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
