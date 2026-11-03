# Coming from forward hooks

This page is for people who save activations with `register_forward_hook` and want to know when
a TorchLens capture is the better fit. The examples were written against PyTorch 2.x and the
TorchLens version on this branch. Checked 2026-11-03.

`register_forward_hook` ships with torch, adds nothing to your dependency tree, and is the right
tool for always-on training instrumentation: a scalar norm per module per step costs almost
nothing and never needs a capture. If that is your whole ask, keep your hooks.

## The same task in both

The task: save the output of the ReLU in a small network.

### In forward hooks

```python
# migration-test: tool=hooks expected=(2, 8)
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)

saved = {}


def save_output(module, args, output):
    saved["relu"] = output.detach()


handle = model[1].register_forward_hook(save_output)
model(x)
handle.remove()
RESULT = tuple(saved["relu"].shape)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=(2, 8)
import torch
import torch.nn as nn

import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)

log = tl.trace(model, x, save=tl.func("relu"))
relu_out = log["relu_1_2"].out
RESULT = tuple(relu_out.shape)
log.cleanup()
```

## Translation table

| Hook idiom | TorchLens spelling |
|---|---|
| `module.register_forward_hook(save_output)` | `log = tl.trace(model, x, save=tl.in_module("encoder"))` then `log["encoder"].out` |
| Hook on every ReLU | `tl.trace(model, x, save=tl.func("relu"))` |
| Hook dict keyed by module name | `log[label].out`; labels are stable, pass-qualified, and listable |
| `register_full_backward_hook` | `capture=tl.options.CaptureOptions(backward_ready=True)` then `trace.backward(loss)`; per-fire grad records with node identity (see [backward.md](../backward.md)) |
| Mutating outputs inside a hook | `intervene=tl.when(tl.func("relu"), tl.zero_ablate())`: declared, audited, replayable |
| Hook that only computes a running scalar | Keep the hook: it is cheaper, and correct for always-on telemetry |

## What TorchLens adds

- Functional operations. `F.scaled_dot_product_attention`, `torch.matmul`, `x + y` and
  `.softmax()` are not `nn.Module` boundaries, so module hooks never see them. TorchLens records
  every wrapped tensor operation (`tl.func("relu")` selects by operation name).
- Per-call identity. A weight-tied block or a loop-invoked cell fires one hook per call with no
  identity attached. TorchLens labels each call pass-qualified (`attn_1_1:2` is the second pass),
  with the containing-module stack per call.
- The executed dataflow graph (which operation consumed which), source lines per operation,
  selective capture predicates, interventions, replay and validation (`tl.validate`). A hook
  dictionary does not grow into these.

## What forward hooks do that TorchLens does not

- Hooks cost close to nothing per step. A TorchLens capture is instrumented execution whose
  overhead depends on model, shape and save tier, and is published only as a measured matrix
  (see [native-torch.md](../native-torch.md)). For always-on scalar telemetry, use hooks.
- Hooks can change the live forward pass in training with no capture step, and they compose with
  any other PyTorch tooling that registers hooks. TorchLens interventions are declared and
  replayable, which is a different contract.
- For tools that attach to training loops or profile kernels, see
  [landscape.md](landscape.md).

## Using both

### Forward hooks feed TorchLens

There is no bridge. A tensor a hook saved is an ordinary tensor: compare it with
`log[label].out` from a capture of the same model and input, or keep both in one script.

### TorchLens feeds forward hooks

There is no bridge. Tensors from `log[label].out` are plain `torch.Tensor` objects, so any code
that consumed a hook's saved dictionary can take them directly.

## Semantic differences

- A hook fires on `nn.Module` calls only. A functional operation has no module to hook, so the
  same ReLU written as `torch.relu(x)` inside `forward` is invisible to a module hook and is
  recorded by TorchLens:

```python
# migration-test: tool=torchlens expected=True
import torch
import torch.nn as nn

import torchlens as tl


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.fc(x))


torch.manual_seed(0)
model = Net().eval()
x = torch.randn(2, 8)
log = tl.trace(model, x, save=tl.func("relu"))
RESULT = log["relu_1_2"].out.shape == (2, 8)
log.cleanup()
```

- Choose by the question. Always-on scalar telemetry: hooks. What actually ran, at operation
  resolution, with identity: a capture.
