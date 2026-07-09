# Migrating from NNsight

Functional migration pattern: an NNsight trace block that saves one module value maps to a
TorchLens capture followed by label/module lookup on the completed log.

| NNsight construct | TorchLens equivalent |
| --- | --- |
| Save an intermediate value during a trace. | Capture the eager forward, then read the saved activation. |

Their construct (pseudocode: NNsight requires a compatible language model wrapper and
token/input contract; the local `Tiny` module below is not such a wrapper):

```python
# illustrative only; adapt the model and input to the installed NNsight version
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


model = LanguageModel("your-supported-hugging-face-model", dispatch=True)
x = torch.tensor([[2.0, 3.0]])
with model.trace(input_ids=x):
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
log = tl.trace(model, x)
RESULT = log["linear_1_1"].out.detach().tolist()
```
