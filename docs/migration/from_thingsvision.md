# Coming from thingsvision

This page is for people who extract features from a model zoo with thingsvision and want the same
numbers from a model they load themselves. The examples were written against thingsvision 1.4.x
APIs (see the environment note below) and the TorchLens version on this branch. Checked 2026-11-03.

[thingsvision](https://github.com/ViCCo-Group/thingsvision) (ViCCo-Group; Muttenthaler & Hebart)
made preprocessing correctness a first-class concern in NeuroAI feature extraction, and its
journey-shaped docs and `get_transformations` / `show_model()` ergonomics inspired several TorchLens
surfaces. TorchLens covers a different slice: it verifies, addresses, documents, and transports
features from any PyTorch model you bring, and omits model loading, stimulus management, and
analysis pipelines (see the [NeuroAI journey](../neuroai/journey.md) for the end-to-end workflow and
the [loader interop page](../neuroai/loaders.md) for two-line model loading per ecosystem).

## The same task in both

The task: the average-pool features of a ResNet-50, no pretrained weights (shapes are asserted,
which do not depend on weights).

### In thingsvision

Needs thingsvision, in its own environment (see the environment note below); the fence is skipped
where it is not installed.

```python
# migration-test: tool=thingsvision expected=[['avgpool'], [2, 2048, 1, 1]]
# Feature SHAPES are asserted (weights-independent).
import torch
from thingsvision import get_extractor

extractor = get_extractor(
    model_name="resnet50",
    source="torchvision",
    device="cpu",
    pretrained=False,
)
features = extractor.extract_features(
    batches=[torch.zeros(2, 3, 64, 64)],
    module_names=["avgpool"],
    flatten_acts=False,
)
RESULT = [sorted(features), list(features["avgpool"].shape)]
```

### In TorchLens

Same architecture, same site, no zoo: construct the model with torchvision directly.

```python
# migration-test: tool=torchlens expected=[['avgpool'], [2, 2048, 1, 1]]
import torch
from torchvision.models import resnet50

import torchlens as tl

model = resnet50(weights=None).eval()
features = tl.extract(model, torch.zeros(2, 3, 64, 64), ["avgpool"])
RESULT = [sorted(features), list(features["avgpool"].shape)]
```

With real weights, load them the torchvision way and resolve the matched preprocessing from the SAME
weights object; TorchLens verifies it instead of bundling its own recipe table. This sketch needs
a weights download and an image iterable, so it is shown, not executed:

```text
from torchvision.models import ResNet50_Weights, resnet50

import torchlens as tl
import torchlens.preprocessing as pp

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()
pre = pp.resolve(weights)               # the loader's own declared recipe
print(pre.preprocessing.description)           # resize=232 crop=224 (V2 -- not 256/224!)

features = tl.extract_dataset(
    model,
    images,                             # any iterable of PIL images
    ["avgpool"],
    path="resnet50_features",
    input_transform=pre,                # applied AND recorded, verdict "verified"
    stimulus_ids=image_ids,
)
```

## Translation table

| thingsvision construct | TorchLens equivalent |
| --- | --- |
| `get_extractor(model_name, source, ...)` | Load the model with its own library (torchvision, timm, transformers, open_clip); TorchLens has no zoo. |
| bundled matched preprocessing | Resolve YOUR loader's own preprocessing and verify it: `torchlens.preprocessing.resolve(...)` / `.audit(...)`. |
| `extractor.show_model()` | `torchlens.inventory.list_sites(model, x)`: structured rows with copy-paste-safe selectors, shapes, and pass indices. |
| `extract_features(batches, module_names=[...])` | `tl.extract(model, x, [...])` (one batch) or `tl.extract_dataset(...)` (datasets, disk artifacts, resume). |
| `save_features(features, out_path, file_format=...)` | disk-mode `tl.extract_dataset(..., output_dir=...)` plus `tl.load(...)`; per-site adapters in `torchlens.bridge`. |
| CLIP through the custom source | Load with `transformers`, extract `vision_model.post_layernorm`, apply `visual_projection` yourself (below). |

### The CLIP image embedding

thingsvision exposes CLIP through its custom source; with TorchLens you load the model with
`transformers` and take the projected embedding explicitly: one visible projection line, so you
know which representation you are taking (the 768-d `post_layernorm` output is not the 512-d
embedding). Shown, not executed (needs a download):

```text
import torch
from transformers import AutoImageProcessor, CLIPModel

import torchlens as tl
import torchlens.preprocessing as pp

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
pre = pp.resolve(processor)

pixel_values = pre.input_transform(images)["pixel_values"]
out = tl.extract(model, [input_ids, pixel_values], ["vision_model.post_layernorm"])
with torch.no_grad():
    image_embeds = model.visual_projection(out["vision_model.post_layernorm"])
```

On transformers 5.x, `CLIPModel.get_image_features` no longer returns the embedding tensor (it
returns the full vision output object), and the `image_embeds` on the forward output are
L2-normalized; the manual projection above matches them exactly after normalization.

### Export and analysis

thingsvision's `save_features(..., file_format=...)` maps to the disk-mode extraction artifact plus
per-site adapters. The artifact's manifest is the row-order authority and carries the
input-preprocessing provenance block. Shown, not executed:

```text
loaded = tl.load("resnet50_features")
print(loaded.input_preprocessing["verdict"])    # "verified" / "mismatch" / "unknown"

from torchlens.bridge import rsatoolbox as tl_rsa
dataset = tl_rsa.dataset("resnet50_features", site="avgpool")  # layer-wise RSA
```

To keep large layers small during extraction, pass
`transform={"layer4": tl.transforms.srp(n_components, seed=0)}` to `tl.extract_dataset`: sparse
random projection, the speed trick DeepJuice uses for brain benchmarking (worked example in
[model zoos](from_model_zoos.md)).

## What TorchLens adds

- Preprocessing verification: `torchlens.preprocessing.resolve(...)` reads the loader's own
  declared recipe, and `extract_dataset(..., input_transform=pre)` applies and records it with a
  verdict (`verified`, `mismatch` or `unknown`) stored in the artifact.
- Any PyTorch model: sites are addressed by module path, operation label or pass-qualified label,
  not by a fixed model list.
- Structured site listing (`torchlens.inventory.list_sites`) with shapes and pass indices.
- Disk artifacts with resume, a manifest that fixes row order, and `tl.load` to read them back.
- Per-site adapters in `torchlens.bridge` (for example `torchlens.bridge.rsatoolbox.dataset`, a
  destination for representational analysis; TorchLens makes no RSA parity claim).

## What thingsvision does that TorchLens does not

- Model loading: `get_extractor(model_name, source, ...)` covers torchvision, timm, Hugging Face,
  open_clip and other sources by name. TorchLens has no zoo; load with the source's own library.
- Stimulus management and its analysis pipeline. For gLocal and other learned alignment
  transforms, keep using thingsvision's own implementations over the exported matrices; see
  [bring your own alignment](../neuroai/byo_alignment.md).
- Anything beyond feature extraction that the thingsvision documentation describes; see
  [landscape.md](landscape.md) for destinations.

## Using both

### thingsvision feeds TorchLens

There is no bridge. thingsvision and TorchLens cannot share one environment (below). Run the
thingsvision side on its own and save features, for example `.npy` files, then compare or combine
them with TorchLens outputs as arrays.

### TorchLens feeds thingsvision

There is no bridge. Export the extracted matrices (`tl.load(...)` artifacts, or the arrays
`tl.extract` returns) and hand them to thingsvision's analysis functions as NumPy arrays.

## Semantic differences

- Layer names translate. thingsvision's `module_names` are torch module addresses (`"features.10"`,
  `"avgpool"`); TorchLens accepts those same qualified addresses, plus operation labels
  (`"conv2d_2_5"`) and pass-qualified spellings for recurrent sites (`"linear_1_1:2"`). Run
  `torchlens.inventory.list_sites(model, x)` once and copy selectors from the first column.
- The `transform=` trap. On `tl.trace`, `transform=` preprocesses inputs. On `tl.extract_dataset`,
  `transform=` transforms outputs before storage (thingsvision's `apply_transforms` habit). For
  input preprocessing in `extract_dataset`, use `input_transform=`: carrying the trace habit into
  `extract_dataset(output_transform=...)` would silently z-score your features instead of your
  pixels.
- thingsvision and TorchLens cannot share one environment. Checked 2026-10-05 (Python 3.11, Linux
  x86_64): the then-current thingsvision release (2.7.3) pins `torchvision==0.15.2`, which requires
  torch 2.0.1, below TorchLens's `torch>=2.1` floor, so the two cannot be resolved into one
  environment. uv then falls back to thingsvision 1.4.4, which has no top-level `get_extractor`, so
  the thingsvision half of the comparison above does not run there either. The thingsvision fence
  therefore runs only in an environment that has a compatible thingsvision.
