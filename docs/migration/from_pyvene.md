# Migrating from Pyvene

Functional migration pattern: a Pyvene intervention target maps to a TorchLens selector plus an
explicit replay/rerun recipe.

| Pyvene construct | TorchLens equivalent |
| --- | --- |
| Configure an intervention on a representation site. | Capture an intervention-ready log, select a site, apply a helper, and rerun or replay. |

Their construct:

```python
# migration-test: tool=pyvene expected=[[0.0, 0.0]]
import torch
from pyvene import IntervenableConfig, IntervenableModel, ZeroIntervention


config = IntervenableConfig(representations=[{"layer": 0, "component": "block_output"}])
model = IntervenableModel(config, model=None)
_, counterfactual = model(base={"input_ids": torch.ones(1, 2)}, unit_locations=None)
RESULT = counterfactual.detach().tolist()
```

TorchLens equivalent:

```python
# migration-test: tool=torchlens expected=[[0.0, 0.0]]
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
log = tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)
edited = log.fork("zero_linear")
edited.attach_hooks(tl.func("linear"), tl.zero_ablate())
edited.rerun(model, x)
RESULT = edited["linear_1_1"].activation.detach().tolist()
```
