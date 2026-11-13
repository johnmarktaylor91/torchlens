# PyTorch backend

PyTorch eager capture is TorchLens's reference backend and its only `stable` backend. Every
other backend is measured against it: the shared conformance suite runs torch first, and its
oracles (tolerances, op inventories, graph integrity) are accepted only when torch passes them.
Torch fills the one backend protocol, `tl.backends.BackendProtocol`, through an adapter over its
public entries, so registering the protocol changed nothing about torch capture.

## Tutorial

Capture a forward pass, check it with replay validation, and drive the same model through the
backend protocol the conformance suite uses.

```python
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.registry import get_backend_spec

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
x = torch.randn(2, 4)

trace = tl.trace(model, x)
print([op.func_name for op in trace])
assert tl.validate(model, x, scope="forward")

protocol = get_backend_spec("torch").protocol()
print(protocol.name, protocol.materialization)
reference = protocol.reference_output(model, x, None)
captured = protocol.capture(model, x, None)
status = protocol.validate(captured, reference)
assert status.state == "passed", status
```

`protocol.reference_output` runs the model on pristine torch (the TorchLens wrappers removed for
the call), so it is an independent oracle arm. `protocol.validate` returns the replay status
object, never a bare boolean.

## Can and cannot

| Capability | State | Notes |
|---|---|---|
| `backward_capture` | supported | True backward capture through autograd hooks. |
| `fastlog` | supported | Sparse recording. |
| `can_intervene` | supported | Live `intervene=` and `stop_after=`. |
| `rng_replay` | supported | Replay restores the captured RNG state. |
| `has_streaming` | supported | Streaming save. |
| `structure_only_capture` | supported | `structure_only=True`. |
| `has_validation_replay` | supported | Ground-truth rerun plus per-op replay and perturbation. |
| `payload_materialization` | supported | Loaded payloads materialize as tensors. |
| `intermediate_derived_grads` | unsupported | Refuses with `intermediate_derived_grads_unsupported`: torch records true gradients through `backward_capture`; the derived-gradient path belongs to the other backends and torch does not implement it. |

Torch cannot log `torch.compile`, TorchScript or `torch.export` artifacts (log the eager source
module), and fused kernels expose no hidden internal tensors.

## Framework requirement

Torch is a base dependency: supported range `torch>=2.1` with no declared ceiling, tested through
2.13.0, and the byte-golden smoke leg pins 2.13.0. `tl.report.doctor()` shows the installed
version against this range.
