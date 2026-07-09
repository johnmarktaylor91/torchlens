# `tl.hash` reference

`tl.hash` provides a provisional, address-free structural hash for a completed trace or for a
model captured with metadata-only options. It describes operation order, function kinds, parent
topology, output-container structure, and input/output/buffer boundaries. It does not include
parameter values, tensor shapes, dtypes, or module addresses, and is not a security hash.

The names in this namespace are provisional and may become fingerprint-style names after API
review. Hashes are deterministic within a TorchLens version for equivalent capture structure;
pin them in CI to catch unintended architecture changes.

## `trace`

`tl.hash.trace(captured_trace)` returns the established address-free graph-shape hash for an
existing completed trace.

```python
import torch
from torch import nn
import torchlens as tl

captured = tl.trace(nn.Sequential(nn.Linear(2, 2), nn.ReLU()), torch.ones(1, 2))
digest = tl.hash.trace(captured)
print(len(digest), digest == tl.hash.trace(captured))
```

Output:

```text
64 True
```

## `model` and `assert_unchanged`

`tl.hash.model(model, example_input)` performs an inference-only metadata capture with
`layers_to_save=None`, then hashes it. This is the same capture policy used by
`tl.assert_unchanged(model, example_input, expected)`.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
example = torch.ones(1, 2)
pinned = tl.hash.model(model, example)
print(tl.assert_unchanged(model, example, pinned) == pinned)
```

Output:

```text
True
```

To bootstrap a pin, pass `expected=None`. The helper prints and returns the current hash:

```python
pinned = tl.assert_unchanged(model, example, None)
```

On a mismatch it raises `tl.hash.StructuralHashMismatchError`, an `AssertionError` subclass,
whose message includes both the expected and actual hashes. Use the same `layers_to_save=None`
metadata-only setting when producing a trace that you plan to compare directly with `tl.hash.model`.
