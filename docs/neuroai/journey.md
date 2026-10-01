# The NeuroAI feature-extraction journey

Extracting features from models you did not train, to compare against brain
or behaviour data, has one failure mode worse than any crash: plausible,
finite, right-shaped features from a silently wrong pipeline. This journey is
the verified path: load with your ecosystem's own loader, verify the input
side, address the layer precisely, extract at scale into a self-describing
artifact, take the features anywhere, and keep the proof.

Every API spelling on this page is documented-unstable (a naming pass
precedes 3.0); the workflow is the stable part.

## 1. Load

Two lines per ecosystem, using the loader's own preprocessing authority --
see [loading models](loaders.md). TorchLens hosts no zoo and no recipe table;
it reads the recipe your loader ships and verifies against it.

```python
from torchvision.models import ResNet50_Weights, resnet50

import torchlens as tl
import torchlens.preprocessing as pp

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()
pre = pp.resolve(weights)
```

## 2. Verify the input side

Wrong preprocessing does not error -- it produces materially wrong science.
Measured on a seeded 128-image sample of diverse natural photographs (COCO
test2017): skipping normalization on ResNet-50 moved the RDM's Spearman
correlation against the correct pipeline to 0.70 (Pearson 0.76) and top-1
agreement to 0.52; wrong geometry (128x128) was comparable (Spearman 0.69).
Second-order analyses can also be forgiving where first-order ones are not:
wrong-FAMILY constants (CLIP's constants on an ImageNet CNN) left the RDM
nearly intact (Pearson 0.99) while CLIP under ImageNet constants dropped
nearest-neighbour retrieval agreement to 0.55. Severity depends on your
stimuli's structure and your analysis family -- which is exactly why the
verifier reports per-field findings with consequences instead of one alarm
bell.

```python
report = pp.audit(pre, applied={"mean": [0.5] * 3, "std": [0.5] * 3})
print(report.verdict)                  # "mismatch"
print(report.mismatched_fields)        # ("mean", "std")

pp.audit(pre, applied={"mean": [0.5] * 3, "std": [0.5] * 3}, strict=True)
# PreprocessingAuditError [preprocessing_audit_mismatch]: strict preprocessing
# audit found mismatched fields -- mean: authority=(0.485, 0.456, 0.406) ...
```

The audit compares configurations against YOUR authority -- it never guesses
from tensor statistics (measured, those guesses false-alarm on legitimate
pipelines such as plain fp16 casts while missing genuinely wrong ones), it
never auto-fixes, and when the authority declares nothing (torchvision
detection presets) it says `unknown` rather than inventing a reference.
Opt-in tensor diagnostics (`pp.diagnose`) exist for intrinsic failures
(non-finite values, integer batches under a float declaration) and can only
ever contradict -- they have no "match" verdict by construction.

## 3. Address the layer

```python
import torchlens.inventory as inv

sites = inv.list_sites(model, example_batch)   # one disclosed real forward
row = sites.resolve("avgpool")                 # bare leaf names resolve
print(row.selector, row.shape, row.origin)     # avgpool (2, 2048, 1, 1) module_output
```

Every selector in the inventory's first column round-trips: `tl.extract`
returns a dict keyed by the exact string you requested, or the resolver
raises a typed error whose remedy shows a working spelling. Recurrent sites
are pass-qualified (`"cell:2"` is pass 2 of a reused module) -- pass indices
are first-class, never collapsed.

Two transform slots exist and they are not the same: `tl.trace(transform=)`
preprocesses INPUTS; `tl.extract_dataset(transform=)` post-processes OUTPUT
activations before storage. For input preprocessing at extraction scale, use
`input_transform=` (below).

## 4. Extract at scale

```python
paths = tl.extract_dataset(
    model,
    images,                       # any iterable: PIL images, tensors, a Dataset
    ["avgpool", "layer3"],
    batch_size=32,
    output_dir="features_run",
    input_transform=pre,          # the authority's OWN transform, applied + recorded
    stimulus_ids=image_ids,       # row-order authority, digested into the manifest
    resume=True,                  # continue an interrupted run from its ledger
)
```

The output directory is a self-describing artifact: an append-only fsynced
shard ledger, a write-once stimulus-id sidecar, per-site identity and axis
semantics, the resume signature compared field-by-field, and the
input-preprocessing provenance block (authority + audit + verdict) -- which
is allowed to say `unknown`, and says exactly that for artifacts written
before the block existed.

## 5. Take the features anywhere

```python
loaded = tl.load_extraction("features_run")
print(loaded.input_preprocessing["verdict"])

from torchlens.bridge import rsatoolbox as tl_rsa
dataset = tl_rsa.dataset("features_run", site="avgpool")     # per-site, layer-wise RSA

from torchlens.bridge import xarray as tl_xr
assembly = tl_xr.data_array("features_run", site="avgpool")  # presentation x neuroid
```

Both adapters (and the exporters) share ONE recorded shaping operation, so
the file route and the in-memory route are numerically identical, row order
is tied to your stimulus ids, and nobody hand-writes a reshape. For learned
alignment transforms (including gLocal), export and apply your own -- see
[bring your own alignment](byo_alignment.md).

## 6. Prove it

```python
print(loaded.manifest["signature"]["model_identity"]["digest"][:16])  # who computed this
print(loaded.input_preprocessing["verdict"])                          # on what input regime
tl.assert_unchanged(model)                                            # model untouched
```

A resumed run refuses typed on ANY semantic signature mismatch (different
model state, batch size, transform identity -- including the input
transform), and resume never grafts new provenance onto rows it did not
compute.

## Showpiece: recurrent time-courses (CORnet-S)

Pass-qualified sites make recurrent dynamics one extraction, not a custom
hook harness -- the representational geometry of each timestep of a reused
block:

```python
model = torch.hub.load("dicarlolab/CORnet", "cornet_s", pretrained=True).eval()
sites = inv.list_sites(model, example_batch)
v4_passes = [r.selector for r in sites if r.module_address == "module.V4.output"]
# ['module.V4.output:1', 'module.V4.output:2', ..] -- one per timestep

log = tl.trace(model, batch, save=tl.in_module("module.V4"))
tl.repgeom.rdm_evolution(log, save=tl.in_module("module.V4"))
```

## Showpiece: whole-network geometry in one pass

```python
log = tl.trace(model, batch)                  # save everything (small batches!)
tl.repgeom.rdm_evolution(log)                 # one RDM per layer, one forward
```
