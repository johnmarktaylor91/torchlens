# Penzai to TorchLens v2 Migration

Status: the penzai repository is archived (read-only since its last push on
2025-06-22), and treescope, the pretty-printer that grew out of it, has had no commits
since August 2026. Existing penzai code keeps working; new work needs another home.

| Penzai operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Name and select model subtrees | `tl.in_module("path")`, `tl.module("path")`, and exact labels after discovery | Similar goal; TorchLens uses PyTorch module addresses and op labels. |
| Functional model surgery | Capture with `capture=tl.options.CaptureOptions(intervention_ready=True)`; `trace.fork()`, then `.set(site, value)` or `.attach_hooks(site, helper)`, then `.push()` (replay the recorded graph) or `.run()` (run the model again) | TorchLens edits a recorded run, not the model definition. |
| Inspect intermediate values | `trace.find_sites(selector).first().out`, or `trace[label].out` | Equivalent for captured tensors. |
| Patch or replace a value | `.set(site, tensor)` or `.attach_hooks(site, helper)` on a forked trace | Equivalent for visible activations. |
| Declarative named axes | Explicit tensor shapes, plus `dim=` where a helper needs it | No named-axis system. |
| Transformer-specific layer abstractions | `torchlens.mechinterp` (TransformerLens-style names through `resolve_alias`) and attention facets | Different vocabulary; no penzai layer classes. |
| Save a modified computation | `trace.save_intervention(path, level="portable")` writes a `.tlspec/` recipe | TorchLens saves recipes, not rewritten model trees. |
| Compare several variants | `tl.Bundle({...})` | Equivalent comparison container. |
| Intervene on fused kernel internals | Attention facets and `torchlens.mechinterp.lowered_counterfactual` for heads on fused attention; manual unfused code otherwise | Raw fused-kernel intermediates are not captured ops. |
| Pure JAX workflows | `tl.trace(fn, (params, x), backend="jax")`: a preview backend that captures a functional call `fn(params, *inputs)` from its jaxpr; Equinox and Flax NNX modules are supported roots | Preview: `jax.jit`, `lax.cond`, `lax.scan`, `lax.while_loop` and custom-VJP regions refuse in strict mode; gradients are derived, not captured backward passes (see `docs/backends.md`). |
| Pretty-print model trees and arrays (treescope) | The `torchlens.bridge.treescope` bridge (`docs/guides/treescope_cards.md`), or the bounded `repr`/`str` cards on every trace object | |

The penzai idiom "select a subtree, swap a value, look at the result" in TorchLens,
executed:

```python
# migration-test: tool=torchlens expected=([[2.5, 2.5]], [[0.0, 0.0]])
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


trace = tl.trace(
    Tiny(),
    torch.tensor([[2.0, 3.0]]),
    capture=tl.options.CaptureOptions(intervention_ready=True),
)
edited = trace.fork("zeroed").set(tl.module("proj"), torch.zeros(1, 2), confirm_mutation=True)
edited.push()
RESULT = (
    trace[trace.output_layers[0]].out.tolist(),  # the original run is untouched
    edited[edited.output_layers[0]].out.tolist(),
)
```

## Honest concession

Penzai was the better fit for functional JAX model surgery, named axes, and workflows that
want the model tree itself to be the edited artifact, and nothing in TorchLens replaces
that: TorchLens records and edits runs, it does not rewrite model trees. For PyTorch
models, TorchLens's capture surface is PyTorch eager; for JAX, the preview backend
captures declared functional calls as jaxpr-derived traces, which is enough to read and
compare activations but is not a JAX tree-rewriting system.
