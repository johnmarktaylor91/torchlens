# Coming from baukit

This page is for people who use baukit's `Trace` and `TraceDict` context managers to read or
edit a module's output and want the TorchLens spelling. baukit is not pinned in TorchLens's
test environment; the baukit example skips where it is not installed. Checked 2026-11-03.

## The same task in both

Read the output of the first linear layer of a small network.

### In baukit

```python
# migration-test: tool=baukit expected=[2, 4]
import torch
from torch import nn
from baukit import Trace


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


torch.manual_seed(0)
model = Net().eval()
x = torch.randn(2, 4)
with Trace(model, "fc1") as ret:
    model(x)
RESULT = list(ret.output.shape)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=([2, 4], True)
import torch
from torch import nn
import torchlens as tl


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


torch.manual_seed(0)
model = Net().eval()
x = torch.randn(2, 4)
features = tl.extract(model, x, {"out": "fc1"})

# The same tensor a forward hook on model.fc1 sees.
seen = {}
handle = model.fc1.register_forward_hook(lambda m, i, out: seen.setdefault("out", out))
with torch.no_grad():
    model(x)
handle.remove()
RESULT = (list(features["out"].shape), torch.equal(features["out"], seen["out"]))
```

## Translation table

| baukit operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| `Trace` a module output | `tl.trace(...); log.find_sites(tl.module("module.path"))` | Similar for module boundaries. |
| `TraceDict` over many layers | `log.find_sites(tl.in_module(...), max_fanout=...)` and iterate results | Equivalent discovery, different object model. |
| Edit an activation in a trace | `log.fork().do(site, edit)` | Equivalent post-hoc mutation for stable graphs. |
| Retain input/output tensors | TorchLens records activations and selected argument metadata in `Op` records | TorchLens is broader but heavier. |
| Stop or replace module output inline | `tl.trace(model, x, intervene=tl.when(site, helper))` during capture, or `trace.run(inputs=x, until=site)` to stop early | Similar for visible module/op outputs. |
| Patch generated-token traces | Capture the generation loop, then `.do(site, edit)` (replay) or `.do(site, edit, model=model, inputs=x)` (run) at the target sites | Equivalent when the Python loop is visible. |
| Lightweight context manager only | No exact equivalent | TorchLens creates a persistent `Trace`; lightweight tracing is deferred to v2.x. |
| Causal tracing utilities | Build with `Bundle`, helpers, and metrics | Turnkey causal-tracing dashboards are deferred to v2.x. |
| Save edited trace recipe | `save_intervention(..., level=...)` | TorchLens has first-class recipe persistence. |
| Fused attention internals | Eager-attention load: the softmax and score ops are captured ops you read directly. Default fused SDPA load: `tl.trace(model, x, reconstruction_ready=True)` serves read-only reconstructed attention facets (`scores`, `pattern`, `z`, checked against the fused output; see `docs/facets.md`) | Head-level semantics are readable on fused models; the raw kernel intermediates are not captured ops, and editing them needs an eager recapture. |

Editing an activation after capture, the TorchLens spelling of baukit's `edit_output`: capture
with `replayable=True`, fork, apply an edit at the site, and read the downstream output.

```python
# migration-test: tool=torchlens expected=(True, True)
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
edited = trace.fork("zeroed")
edited.do(tl.module("1"), tl.zero_ablate())  # the ReLU output becomes all zeros
head_out = edited[edited.output_layers[0]].out
head = model[2]
expected = head(torch.zeros(2, 4))  # the head sees an all-zero ReLU output
RESULT = (
    torch.allclose(head_out, expected, atol=1e-6),
    not torch.allclose(head_out, trace[trace.output_layers[0]].out, atol=1e-6),
)
```

## What TorchLens adds

- A persistent `Trace` of the whole run: every executed op with its output, shape and module
  address, so the read of one module is a lookup into a record that also holds the rest.
- Post-hoc edits on a fork (`.do(site, edit)`) that leave the original run untouched, and
  `trace.run(inputs=x, until=site)` to stop early.
- Intervention during capture: `tl.trace(model, x, intervene=tl.when(site, helper))`.
- Recipes saved as `.tlspec/` directories (`save_intervention(..., level=...)`).
- Eager-attention captured ops and, on fused attention, reconstructed read-only facets
  (`tl.trace(model, x, reconstruction_ready=True)`, `docs/facets.md`).

## What baukit does that TorchLens does not

- baukit's `Trace` is a lightweight context manager around one module: no persistent record
  is created. TorchLens creates a `Trace` of the run; lightweight tracing is deferred to v2.x.
- baukit offers turnkey causal-tracing utilities and a `TraceDict` over many named layers
  with inline edits. In TorchLens you build causal tracing with `Bundle`, helpers and
  metrics; turnkey causal-tracing dashboards are deferred to v2.x.
- For other tools, see [where to go instead](where_to_go_instead.md).

## Using both

### baukit feeds TorchLens

There is no bridge. baukit works on an `nn.Module`; the same module goes to `tl.trace` or
`tl.extract`. Tensors you read with `Trace` can be compared against TorchLens saved
activations as ordinary tensors.

### TorchLens feeds baukit

There is no bridge. Module addresses from `trace.modules` are the names `Trace(model, name)`
takes, so TorchLens can help you find a layer name; the activations themselves are plain
tensors.

## Semantic differences

- **Object model.** `TraceDict` over many layers becomes `log.find_sites(tl.in_module(...),
  max_fanout=...)` and an iteration over the results; the trace is one object for the run.
- **Generation.** To patch generated-token traces, capture the generation loop, then `.do(...)`
  at the target sites; this is equivalent when the Python loop is visible.
- **Heavier record.** TorchLens records activations and selected argument metadata in `Op`
  records, which is broader than baukit's retained input and output and heavier.
- **Fused attention.** Head-level semantics are readable on fused models through
  reconstructed facets; the raw kernel intermediates are not captured ops, and editing them
  needs an eager recapture.
