# Multi-GPU extraction

TorchLens captures one forward pass in one process. This page says what happens when the
model you pass is wrapped for multi-GPU training, and how to extract activations at
scale anyway.

| Wrapper or state | What `tl.trace` and `tl.extract` do | What to do |
| --- | --- | --- |
| `nn.DataParallel(model)` | Unwraps it and captures `.module`, the single underlying model, on one device and one thread. The batch is not split across replicas. | Pass the wrapper or `.module`; the result is the same. Put the model and the input on the device you want. |
| `DistributedDataParallel(model)` | Unwraps it and captures the rank-local `.module`. | Capture on each rank you need; rank-local traces can be merged with `tl.merge_ranks([...])`. |
| `FullyShardedDataParallel(model)` | Refuses with a `RuntimeError`: the parameters are sharded across ranks, so there is no whole module to capture. | Capture a rank-local copy of the model from before FSDP wrapping. |
| DTensor, tensor-parallel or pipeline-parallel state | Refuses with `DistributedCaptureUnsupportedError`; the error lists what it found. | Capture an unsharded copy, or one stage at a time. |
| Several worker processes or threads capturing at once | Not supported: capture state is process-global. | Capture in one process, one forward at a time. |

To extract over a large dataset on several GPUs, run one independent extraction per GPU,
each on its own shard of the stimuli, for example one `tl.extract_dataset(...)` call per
process with its own `output_dir`. TorchLens does not split a batch across devices for
you.

The example below wraps a model in `nn.DataParallel` and checks that the extracted
activation is exactly the one the unwrapped model produces (on a machine without GPUs,
`DataParallel` simply calls the module):

```python
# migration-test: tool=torchlens expected=([2, 8], True)
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
x = torch.randn(2, 4)
wrapped = nn.DataParallel(model)

from_wrapper = tl.extract(wrapped, x, ["1"])["1"]
from_module = tl.extract(model, x, ["1"])["1"]
RESULT = (list(from_wrapper.shape), torch.equal(from_wrapper, from_module))
```
