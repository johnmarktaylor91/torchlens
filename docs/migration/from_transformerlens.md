# Migrating from TransformerLens

Functional migration pattern: `run_with_cache` hook-name access maps to a TorchLens capture plus
site discovery or exact graph labels. TorchLens labels are derived from eager PyTorch execution, not
TransformerLens hook-point names.

| TransformerLens construct | TorchLens equivalent |
| --- | --- |
| Read one cached activation from `run_with_cache`. | Capture the forward and read the matching saved activation. |

Their construct:

```python
# migration-test: tool=transformer_lens expected=[[2.5, 2.5]]
import torch
from transformer_lens import HookedTransformer


model = HookedTransformer.from_pretrained("tiny-stories-1M")
tokens = model.to_tokens("hello")
_, cache = model.run_with_cache(tokens)
RESULT = cache["hook_embed"][0, 0, :2].detach().reshape(1, 2).tolist()
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
