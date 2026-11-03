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
# Activation values depend on the downloaded checkpoint.
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


log = tl.trace(Tiny(), torch.tensor([[2.0, 3.0]]))
RESULT = log["linear_1_1"].out.detach().tolist()
```

## A TransformerBridge cannot be traced (and it mutates your HF model)

TransformerLens 4.0 (September 2026) makes `TransformerBridge` its main model class and
starts deprecating `HookedTransformer`. TorchLens refuses to capture a
`transformer_lens.TransformerBridge` (3.x or 4.x) with a typed `CompatibilityError`
naming the remedy. The bridge redirects attribute assignment to the HF modules it wraps,
so TorchLens's forward instrumentation can never land on it; without the refusal,
capture died partway through with an internal `AttributeError`. Trace a pristine model
instead:

- reload the HF model fresh (`AutoModelForCausalLM.from_pretrained(...)`) and trace
  that: this is the path that keeps working as `HookedTransformer` is retired, or
- trace a `HookedTransformer` directly while you still use one: it captures fully and
  is the same-object comparison subject TorchLens's own oracle suite uses.

Your TransformerLens hook names still work on the traced HF model:
`torchlens.mechinterp.resolve_alias(trace, name)` accepts 2.x names, 3.x bridge-cache
names and 4.x canonical names (for example `blocks.3.attn.o.hook_in`), and reports
honestly when a name has no counterpart in the model in front of you.

Separately, be aware that the bridge/boot machinery (`boot_transformers`, bridge
construction) MUTATES the HF model it wraps in place. A model instance that has already
passed through the bridge is no longer the pristine model you loaded; reload it fresh
before tracing rather than reusing it.
