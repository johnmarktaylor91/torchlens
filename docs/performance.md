# Performance guide

TorchLens is fastest when it captures only the payloads you plan to inspect. A full
`tl.trace(model, x)` records a complete operation graph and saves activations for the selected
sites; `tl.record(model, x, save=...)` is the lighter path for tight loops where you only need
selected records and can materialize a `Trace` later.

## Decision tree

| Need | Use | Notes |
| --- | --- | --- |
| Complete graph metadata and a few activations | `tl.trace(model, x, save=predicate)` | Best default for debugging and one-off analysis. |
| Repeated activation pulls in a loop | `tl.record(model, x, save=predicate)` | Lower overhead; call `Recording.to_trace()` only when you need graph structure. |
| A local window around a later op | `tl.trace(..., save=tl.followed_by(...), lookback=K)` | Retains bounded recent metadata, and optionally bounded recent payloads. |
| Disk-backed selected payloads | `tl.trace(..., storage=tl.to_disk(path))` | Keeps selected payloads portable without retaining them all in RAM. |
| Intervention during the forward pass | `tl.trace(..., intervene=tl.when(...), save=...)` | Live edits cost more than passive capture; use only when the model must execute edited values. |
| Final logits only | plain `model(x)` | TorchLens adds wrapper dispatch and metadata work; skip it when no intermediate data is needed. |

## Fast activation pull

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.options import CaptureOptions


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

trace = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    capture=CaptureOptions(save_code_context=False),
)
relu_site = trace.find_sites(tl.func("relu")).first()
relu_activation = relu_site.out

assert relu_activation.shape == (2, 4)
```

Use `save=tl.func(...)`, `save=tl.in_module(...)`, or composed predicates such as
`tl.func("conv2d") & tl.followed_by(tl.func("relu"))` instead of saving every payload. The graph
metadata remains available, but unsaved payloads are intentionally absent.

## Sparse recording loop

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

recording = tl.record(model, x, save=tl.func("gelu"))
trace = recording.to_trace()
gelu_site = trace.find_sites(tl.func("gelu")).first()

assert gelu_site.out.shape == (2, 4)
```

`tl.record(save=...)` is the canonical spelling. `record(keep_op=...)` and
`record(keep_module=...)` remain deprecated aliases for older code.

## Speed knobs

| Knob | Faster setting | Tradeoff |
| --- | --- | --- |
| Payload selection | `save=tl.func(...)` or `save=tl.in_module(...)` | Unsaved activations cannot be read later. |
| Source text | `capture=CaptureOptions(save_code_context=False)` | File/line identity remains, but source text is not loaded. |
| Window payloads | `lookback_payload_policy="metadata_only"` | `tl.followed_by(...)` can select metadata without retaining raw tensors. |
| Retroactive payloads | `lookback_payload_policy="detached_raw"` | Enables payload recovery for recent matched ops at bounded memory cost. |
| Disk storage | `storage=tl.to_disk(path)` | Reduces RAM pressure; disk I/O becomes part of capture cost. |
| Gradients | `save_grads=False` unless needed | Backward-ready captures preserve more state and hooks. |
| Visualization | Call `trace.draw()` after capture, not during hot loops | Rendering is separate from activation collection. |

## Windowed and disk-backed capture

```python
from pathlib import Path

import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU()).eval()
x = torch.randn(1, 1, 5, 5)
path = Path(DOCS_TMPDIR) / "windowed.tlspec"

predicate = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
trace = tl.trace(
    model,
    x,
    save=predicate,
    lookback=4,
    lookback_payload_policy="detached_raw",
    storage=tl.to_disk(path),
)

assert trace.find_sites(tl.func("conv2d")).first().out.shape == (1, 2, 3, 3)
```

Use disk-backed storage for selected payloads that are too large or numerous to keep in memory.
Portable `.tlspec/` bundles store manifest data plus tensor sidecars; executable Python callables
are not portable.

## Intervention cost

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

patched = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)

assert torch.count_nonzero(patched.find_sites(tl.func("relu")).first().out) == 0
```

Use `intervene=` when the edited value must affect downstream execution. For post-hoc experiments
on an existing trace, prefer `trace.fork()`, `set(...)` or `attach_hooks(...)`, and `replay()`.

## When another tool is faster

Use raw PyTorch when you only need the final output. Use `torch.profiler` when the question is
kernel timing rather than activation provenance. Use TransformerLens when your workflow is entirely
inside its supported transformer families and you already want its named hook points. Use Captum or
Inseq when you want mature attribution algorithms out of the box rather than the lower-level
activation and gradient substrate.
