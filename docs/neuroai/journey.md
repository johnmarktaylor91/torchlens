# The NeuroAI feature-extraction journey

## For neuroAI and computational neuroscience

Comparing a model with brain or behaviour data needs per-layer features for a fixed stimulus set, extracted with the model's own preprocessing, in stimulus order, at a scale of thousands of images, and then handed to the analysis package (RSA, encoding models, shape metrics) that owns the brain side. TorchLens does the network side of that: it loads with your ecosystem's loader, verifies the input pipeline, addresses any layer precisely, extracts in batches to a self-describing artifact and connects to rsatoolbox, Brain-Score, himalaya and netrep. The links here are the shortest routes in; the numbered sections below walk the whole path once.

- [Tutorial notebooks](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/learn/): twelve short notebooks on laptop-sized models. Notebook 02 reads an activation out, 04 keeps memory in check, 09 extracts over thousands of batches and 10 has a neuroscience branch.
- [Model RDM notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/cog_neuro_rdm.ipynb): a compact representational dissimilarity matrix from one model in a few lines.
- [Extract features for a stimulus set](../field-guide/extract_features.md): per-layer activations for every stimulus, written with the model's identity and preprocessing.
- [Loading models for feature extraction](loaders.md): two lines per ecosystem (torchvision, timm, Hugging Face, CLIP and others), each with its matched preprocessing.
- [Extract a dataset notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/quick/extract_dataset.ipynb): named-layer activations over a whole dataset in batches, in memory or on disk, and [seeded subsampling](../field-guide/seeded_subsample.md) for a reproducible subset.
- [Cognitive neuroscience extraction notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/workflows/cog_neuro_extraction.ipynb): per-layer extraction and an offline Brain-Score-style evaluation, end to end.
- [CKA and RSA between two models notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/examples/methods/cka_rsa_two_models_recipe.ipynb) and the [streaming CKA notebook](https://github.com/johnmarktaylor91/torchlens/blob/main/notebooks/streaming_cka.ipynb): layer-by-layer similarity heatmaps, in constant memory on streamed data.
- [torchlens.neuro](../reference/neuro.md) and [the brain-benchmarking seam](../reference/brain_benchmarking_seam.md): what TorchLens owns (the network) and what rsatoolbox and Brain-Score own (the brain and the hypothesis).
- [TorchLens to himalaya](himalaya.md): banded ridge encoding models over a multi-site feature bank.
- [TorchLens to netrep](netrep.md): shape metrics and the CCA family between representations.
- [Bring your own alignment](byo_alignment.md): linear probes, Procrustes and learned transforms over exported features.
- [Receptive and projective fields](../receptive_projective_fields.md): which input positions can affect a unit, and which outputs a unit can affect, on the graph that ran.
- [Multi-GPU extraction](../migration/multi_gpu_extraction.md) and [streaming statistics](../reference/stats.md): extraction and dataset-level statistics when the data does not fit in memory.
- [Connectors to other tools](../bridges.md): rsatoolbox, Brain-Score, xarray, Hugging Face and timm loaders, one entry each.
- Coming from [thingsvision](../migration/from_thingsvision.md), [torchvision feature extraction](../migration/from_torchvision_feature_extraction.md), a [model zoo wrapper](../migration/from_model_zoos.md) or plain [forward hooks](../migration/from_hooks.md).

The sections below run in order as one script. The preprocessing audit, the extraction `resume` and `stimulus_ids` options, the extraction loader and structural fingerprints are provisional; their spellings may change.

## 1. Load

Two lines per ecosystem, using the loader's own preprocessing authority (see [loading models](loaders.md)). TorchLens hosts no zoo and no recipe table; it reads the recipe your loader ships and verifies against it.

```python
from torchvision.models import ResNet50_Weights, resnet50

import torchlens as tl
import torchlens.preprocessing as pp

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()
pre = pp.resolve(weights)
```

## 2. Verify the input side

Wrong preprocessing does not error; it produces finite, right-shaped, wrong features, and how much it changes an analysis depends on your stimuli and your analysis family. Compare the pipeline you apply with the one the model declares before you extract:

```python
report = pp.compare(pre, applied={"mean": [0.5] * 3, "std": [0.5] * 3})
print(report.verdict)                  # mismatch
print(report.mismatched_fields)        # ('mean', 'std')
```

Strict mode turns the mismatch into an error:

```python
try:
    pp.compare(pre, applied={"mean": [0.5] * 3, "std": [0.5] * 3}, strict=True)
except tl.errors.PreprocessingAuditError as err:
    print(err)   # strict preprocessing audit found mismatched fields: mean: authority=(0.485, 0.456, 0.406) applied=(0.5, 0.5, 0.5); ...
```

The audit compares configurations against your declared authority. It never guesses from tensor statistics, never auto-fixes, and reports `unknown` when the authority declares nothing (torchvision detection presets). Opt-in tensor diagnostics (`pp.diagnose`) catch intrinsic failures such as non-finite values or integer batches under a float declaration; they can only contradict a declaration, never confirm one.

## 3. Address the layer

The stimuli below are flat colour images standing in for your stimulus set; the mechanics are the same for real images.

```python
import torch
from PIL import Image

images = [Image.new("RGB", (256, 256), color=c) for c in ("red", "green", "blue", "white")]
image_ids = ["red", "green", "blue", "white"]
example_batch = torch.stack([pre.input_transform(image) for image in images])
```

```python
import torchlens.inventory as inv

sites = inv.list_sites(model, example_batch)   # runs one real forward to list the sites
row = sites.resolve("avgpool")                 # bare leaf names resolve
print(row.selector, row.shape, row.origin)     # avgpool (4, 2048, 1, 1) module_output
```

Every selector in the inventory's first column round-trips: `tl.extract` returns a dict keyed by the exact string you requested, or the resolver raises a typed error whose remedy shows a working spelling. Recurrent sites are pass-qualified: `"cell:2"` is pass 2 of a reused module.

Two transform slots exist: `CaptureOptions(input_transform=)` preprocesses inputs, and `tl.extract_dataset(output_transform=)` post-processes saved activations before storage. For preprocessing at extraction scale, use `input_transform=` (below).

## 4. Extract at scale

```python
paths = tl.extract_dataset(
    model,
    images,                       # any iterable: PIL images, tensors, a Dataset
    ["avgpool", "layer3"],
    batch_size=2,
    path="features_run",
    input_transform=pre,          # the authority's own transform, applied and recorded
    stimulus_ids=image_ids,       # row order, digested into the manifest
    resume=True,                  # continue an interrupted run from its ledger
)
```

The output directory is self-describing: a shard ledger, a stimulus-id sidecar, per-site identity and axis semantics, the resume signature, and the input-preprocessing provenance block (authority, audit, verdict), which says `unknown` for artifacts written before the block existed.

## 5. Take the features anywhere

```python
loaded = tl.load("features_run")
print(loaded.input_preprocessing["verdict"])

from torchlens.bridge import rsatoolbox as tl_rsa
dataset = tl_rsa.dataset("features_run", site="avgpool")     # per-site, layer-wise RSA

from torchlens.bridge import xarray as tl_xr
assembly = tl_xr.data_array("features_run", site="avgpool")  # presentation x neuroid
```

Both adapters and the exporters share one recorded shaping operation, so the file route and the in-memory route are numerically identical and row order follows your stimulus ids. For learned alignment transforms (including gLocal), export and apply your own: [bring your own alignment](byo_alignment.md).

## 6. Prove it

```python
pinned_hash = tl.fingerprint.model(model, example_batch)               # pin this the first time
print(loaded.manifest["signature"]["model_identity"]["digest"][:16])  # who computed this
print(loaded.input_preprocessing["verdict"])                          # on what input regime
tl.fingerprint.assert_unchanged(model, example_batch, pinned_hash)    # structure still matches
```

A resumed run raises a typed error on any signature mismatch (model state, batch size, or the input transform), and never adds new provenance to rows it did not compute.

## Example: recurrent time-courses (CORnet-S)

Pass-qualified sites give the representational geometry of each timestep of a reused block from one extraction. CORnet ships no `torch.hub` entry point, so install a pinned checkout and import it directly:

```text
pip install "git+https://github.com/dicarlolab/CORnet@d0cc17d4b34ad44dedb01683b70eafd15515adad"
```

`cornet.cornet_s()` returns the network inside `nn.DataParallel` (its published checkpoint is keyed that way), so load first and trace `.module`. `pretrained=True` downloads the published weights (about 200 MB); the example below runs the same mechanics on the random initialization.

```python
# recipe-test: tool=cornet expected=(4, 4)
import cornet
import torch

import torchlens as tl
import torchlens.inventory as inv

torch.manual_seed(0)
model = cornet.cornet_s(pretrained=False).module.eval()  # pretrained=True: published weights
batch = torch.randn(6, 3, 64, 64)
sites = inv.list_sites(model, batch[:1])
v4_passes = [r.selector for r in sites if r.module_address == "V4.output"]
# ['V4.output:1', 'V4.output:2', 'V4.output:3', 'V4.output:4']: one per timestep

log = tl.trace(model, batch, save=tl.module("V4.output"))
rdms = tl.repgeom.rdm_evolution(log, save=tl.module("V4.output"))  # one RDM per timestep
RESULT = (len(v4_passes), len(rdms))
```

The same pass addressing goes further on a recurrent network trained on a task:
[task RNN circuits](task_rnn_circuits.md) edits windows of timesteps, finds the
network's fixed points and shows that they hold its memory.

## Example: whole-network geometry in one pass

```python
log = tl.trace(model, batch)                  # save everything (small batches!)
tl.repgeom.rdm_evolution(log)                 # one RDM per layer, one forward
```
