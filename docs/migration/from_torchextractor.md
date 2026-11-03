# Coming from torchextractor

This page is for people who wrap a model in `torchextractor.Extractor` to get named module
outputs. The examples were written against torchextractor's `Extractor(model, layer_names)` API
and the TorchLens version on this branch. Checked 2026-11-03.

TorchExtractor's module-output dictionary maps to a TorchLens completed trace with layer labels
and saved activations, or to `tl.extract(model, x, layers)` when you only want the dictionary.

## The same task in both

The task: read the output of one named module for one input.

### In torchextractor

Needs torchextractor; the fence is skipped where it is not installed.

```python
# migration-test: tool=torchextractor expected=[[2.5, 2.5]]
import torch
from torch import nn
import torchextractor as tx


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


model = tx.Extractor(Tiny(), ["proj"])
_, features = model(torch.tensor([[2.0, 3.0]]))
RESULT = features["proj"].detach().tolist()
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

| torchextractor construct | TorchLens equivalent |
| --- | --- |
| `tx.Extractor(model, ["proj"])` returning `(output, features)` | `tl.extract(model, x, ["proj"])` returning a dictionary of the named outputs; or `tl.trace(model, x)` and `trace[label].out` |
| Layer names as module paths | The same qualified module addresses, plus operation labels (`linear_1_1`) and pass-qualified spellings for repeated modules |
| Feature dictionary keyed by layer name | The dictionary `tl.extract` returns, keyed by the names you passed |
| A migration shim | `tl.compat.from_torchextractor(model, layers)` returns a callable extractor backed by `tl.extract` |

## What TorchLens adds

- Operations that are not modules: a functional op has no module path for torchextractor to hook,
  while a trace records every executed operation.
- Per-call identity for modules that run more than once, and the executed dataflow graph.
- Replay verification (`tl.validate`), source lines per operation, and disk-backed extraction for
  datasets (`tl.extract_dataset`).
- Selective capture predicates (`tl.in_module`, `tl.func`) and interventions.

## What torchextractor does that TorchLens does not

- A small, single-purpose wrapper: `Extractor(model, layers)` is a drop-in that returns
  `(output, features)` from one call, and nothing else to learn. A full trace records more than
  you may want if the dictionary is all you need; `tl.extract` is the closest equivalent.
- For other extraction tools see [landscape.md](landscape.md).

## Using both

### torchextractor feeds TorchLens

`tl.compat.from_torchextractor(model, layers)` takes a model (or a torchextractor extractor-like
object) and returns a TorchLens-backed extractor that is called with one input and returns the
dictionary of named outputs. It needs the `compat-shims` extra.

```python
# migration-test: tool=torchextractor expected=[[2.5, 2.5]]
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


extractor = tl.compat.from_torchextractor(Tiny(), ["proj"])
features = extractor(torch.tensor([[2.0, 3.0]]))
RESULT = features["proj"].detach().tolist()
```

### TorchLens feeds torchextractor

There is no bridge. Features from `tl.extract` are plain tensors in a dictionary, the same shape
torchextractor returns as its second value, so downstream code written for the dictionary takes
them unchanged.

## Semantic differences

- torchextractor's `Extractor` returns `(output, features)` from one call; `tl.extract` and
  `tl.trace` run the model unchanged and return the features separately.
- `tl.extract` returns only the dictionary; the model output is available as the trace's output
  layer when you use `tl.trace`.
