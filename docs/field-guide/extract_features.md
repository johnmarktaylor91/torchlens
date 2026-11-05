# Extract features for a stimulus set

**Result in one line:** per-layer activations for every stimulus, in stimulus order, written
as a directory that records which model, which layers and which preprocessing produced them.

<!-- JMT review: plain factual prose, written for review. -->

This is the NeuroAI task guide: the features you would hand to an RSA, encoding-model or
Brain-Score-style analysis. The analysis itself happens in those tools; TorchLens produces the
features and says where they came from.

**Requirements:** `pip install torchlens torchvision`.

## Minimal example on the running model

This page uses [ResNet-18 on one image](running_examples.md#resnet-18-on-one-image).

```python
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.rand(1, 3, 224, 224)

trace = tl.trace(model, x)
```

The trace lists every addressable site. Module names resolve to that module's output; any
operation, including the residual adds that no module returns, has a label of its own.

```python
print([op.label for op in trace.ops if "add" in op.func_name][:3])
features = tl.extract(model, x, ["layer3", "avgpool"])
print({name: tuple(value.shape) for name, value in features.items()})
```

## Extract over a stimulus set

`tl.extract_dataset` runs the model in batches and keeps row `i` of every stored tensor tied
to stimulus `i`. Without `path=` it returns tensors; with `path=` it writes one shard per
batch plus a manifest.

```python
stimuli = torch.rand(8, 3, 224, 224, generator=torch.Generator().manual_seed(1))
stimulus_ids = [f"stimulus_{index:02d}" for index in range(8)]

in_memory = tl.extract_dataset(
    model, stimuli, ["layer3", "avgpool"], batch_size=4, progress=False
)
paths = tl.extract_dataset(
    model,
    stimuli,
    ["layer3", "avgpool"],
    batch_size=4,
    path="features_run",
    stimulus_ids=stimulus_ids,
    progress=False,
)
loaded = tl.load("features_run")
print(len(paths), "shards")
print(loaded.manifest["signature"]["model_identity"]["digest"][:16])
```

## Output

`in_memory["avgpool"]` has shape `(8, 512, 1, 1)`: one row per stimulus. The directory holds
the shards, a stimulus-id sidecar and the manifest with the model's identity digest.

## A check

Row 3 of the dataset equals the features of stimulus 3 extracted on its own, so batching and
ordering did not move anything.

```python
alone = tl.extract(model, stimuli[3:4], ["avgpool"])["avgpool"]
assert in_memory["avgpool"].shape == (8, 512, 1, 1)
assert torch.allclose(in_memory["avgpool"][3:4], alone, atol=1e-5)
```

## Vary it

- Other layers: any name from `model.named_modules()`, or a selector such as
  `tl.func("relu") & tl.in_module("layer4")` for operations that are not module outputs.
- Your preprocessing: pass `input_transform=` so the transform is applied and recorded in the
  manifest (see [the NeuroAI journey](../neuroai/journey.md#4-extract-at-scale)).
- Pooling and dtype: `pool=` and `dtype=` shrink what is stored per site.
- An interrupted run: `resume=True` continues from the shard ledger and refuses if the model,
  batch size or transform changed.

## Hand the features on

rsatoolbox and Brain-Score are where the analysis runs; TorchLens provides the features and
their provenance.

```text
from torchlens.bridge import rsatoolbox as tl_rsa
dataset = tl_rsa.dataset("features_run", site="avgpool")   # an rsatoolbox Dataset

from torchlens.bridge import xarray as tl_xr
assembly = tl_xr.data_array("features_run", site="avgpool")  # presentation x neuroid
```

## Performance note

Each batch is one forward with capture limited to the requested layers. Memory is one batch of
activations for those layers, not the whole set; storage is the shards on disk.

## Limits

- Row order is the iteration order of `stimuli`; pass `stimulus_ids=` so a reordered set is
  detected rather than silently misaligned.
- Feature correctness is not a neuroscience result: comparing to neural data is the analysis
  tool's job.

## Next task

[Intervene on a site](intervene.md), or the full [NeuroAI feature-extraction journey](../neuroai/journey.md).
