# Coming from torch.fx

This page is for people who use `torch.fx.symbolic_trace` to get a graph of a model's operations
and want to know what a TorchLens trace gives instead. The examples were written against the
PyTorch 2.x `torch.fx` that ships with torch, and the TorchLens version on this branch. Checked
2026-11-03.

FX symbolic tracing produces a static `GraphModule` by running the model on proxy values;
TorchLens runs the eager model on a concrete input and records what actually executed.

## The same task in both

The task: run a small model through the traced graph and read its output.

### In torch.fx

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

### In TorchLens

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

## Translation table

| torch.fx construct | TorchLens equivalent |
| --- | --- |
| `symbolic_trace(model)` followed by `graph_module(x)` | `tl.trace(model, x)` and inspect the eager-operation log (`trace[label].out`) |
| `graph_module.graph.nodes` (node names, targets) | `trace.layers` (labels such as `linear_1_1`, one row per executed operation); `tl.compat.from_fx(graph_module)` lists FX node names as suggested layer lookups |
| Graph rewriting (`graph.node_copy`, `replace_all_uses_with`) | no equivalent; TorchLens does not rewrite graphs. Edit values with `tl.when(site, action)` or `trace.fork().do(site, action)` |

## What TorchLens adds

- Concrete values: each recorded operation carries its output tensor for the input you ran, not
  a symbolic node.
- Data-dependent control flow: the branch the input took is what is recorded (see the fence
  below), where symbolic tracing has no value to branch on.
- Replay verification (`tl.validate`), source lines and containing-module stack per operation,
  loop and pass identity.
- Value edits on the recorded graph (`trace.fork().do(...)`) and capture-time interventions
  (`tl.trace(..., intervene=...)`).

## What torch.fx does that TorchLens does not

- A static graph independent of any input. An FX `GraphModule` is a reusable program: it can be
  transformed, saved as code and recompiled, which a TorchLens trace (one concrete run) is not.
- Graph transformations and passes (rewriting, fusion, quantization and export pipelines build
  on FX). TorchLens records and edits values; it does not produce a transformed module.
- For other destinations see [where to go instead](where_to_go_instead.md).

## Using both

### torch.fx feeds TorchLens

A `GraphModule` is an `nn.Module`, so `tl.trace(graph_module, x)` is the plain call. The migration
helper `tl.compat.from_fx(graph_module)` returns a payload (`schema`, `model`, `layers`) whose
`layers` default to the FX node names; it does not run a capture.

```python
# migration-test: tool=fx expected=('torchlens.fx_migration.v1', True)
import torch
from torch import nn
from torch.fx import symbolic_trace

import torchlens as tl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def forward(self, x):
        return torch.relu(self.proj(x))


payload = tl.compat.from_fx(symbolic_trace(Tiny()))
RESULT = (payload["schema"], "proj" in payload["layers"])
```

### TorchLens feeds torch.fx

There is no bridge. Tensors from `trace[label].out` are plain `torch.Tensor` objects. TorchLens
does not emit an FX graph; to get one, run `symbolic_trace` on the source model.

## Semantic differences

- A symbolic trace cannot branch on a tensor value; TorchLens records the branch the input took.
  The same model, traced both ways:

```python
# migration-test: tool=fx expected=('TraceError', [1.0, 1.0])
import torch
from torch import nn
from torch.fx import symbolic_trace

import torchlens as tl


class Branchy(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return torch.relu(x)
        return -x


try:
    symbolic_trace(Branchy())
    raised = None
except Exception as exc:
    raised = type(exc).__name__

trace = tl.trace(Branchy(), torch.ones(2))
RESULT = (raised, trace[trace.output_layers[0]].out.tolist())
```

- An FX graph is the same for every input; a TorchLens trace is one input's execution, so a model
  with input-dependent control flow gives a different trace for a different input.
