# Coming from forward hooks

<!-- torchnative W0.5: honest in BOTH directions. Hooks are free, in-tree,
     and right for always-on training instrumentation; TorchLens earns its
     cost on the asks hooks structurally cannot serve. -->

`register_forward_hook` is free, ships with torch, adds nothing to your
dependency tree, and is the right tool for always-on training
instrumentation — a scalar norm per module per step costs almost nothing
and never needs a capture. If that is your whole ask, keep your hooks.

TorchLens earns its capture cost on two asks hooks **structurally cannot
serve**, plus the graph they imply:

1. **Functional ops have no hookable module.** `F.scaled_dot_product_attention`,
   `torch.matmul`, `x + y`, `.softmax()` — anything that is not an
   `nn.Module` boundary is invisible to module hooks. TorchLens records
   every wrapped tensor operation, module-shaped or not.
2. **Which call of a reused module?** A weight-tied block or a loop-invoked
   cell fires one hook per call with no identity attached. TorchLens
   addresses each call site pass-qualified (`attn_1_1:2` is the second
   pass), with the containment stack per call.

And what the records compose into: the executed dataflow DAG (who consumed
whose output), source lines per op, selective capture predicates,
interventions, replay, and validation — none of which a hook dictionary
grows into.

## The mapping

| Hook idiom | TorchLens spelling |
|---|---|
| `module.register_forward_hook(save_output)` | `log = tl.trace(model, x, save=tl.in_module("encoder"))` then `log["encoder"].out` |
| Hook on every ReLU | `tl.trace(model, x, save=tl.func("relu"))` |
| Hook dict keyed by module name | `log[label].out`; labels are stable, pass-qualified, and listable |
| `register_full_backward_hook` | `capture=CaptureOptions(backward_ready=True)` + `log.log_backward(loss)`; per-fire grad records with node identity |
| Mutating outputs inside a hook | `intervene=tl.when(tl.func("relu"), tl.zero_ablate())` — declared, audited, replayable |
| Hook that only computes a running scalar | KEEP THE HOOK — cheaper, and correct for always-on telemetry |

```python
import torch
import torch.nn as nn

import torchlens as tl

model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)

log = tl.trace(model, x, save=tl.func("relu"))
relu_out = log["relu_1_2"].out  # functional op: no module to hook
log.cleanup()
```

## Honest cost note

A TorchLens capture is instrumented execution: overhead depends on model,
shape, and save tier, and is published only as a measured matrix (see
[native-torch.md](../native-torch.md)). Hooks cost near zero. Choose by the
question: *always-on scalar telemetry* → hooks; *what actually ran, at op
resolution, with identity* → capture.
