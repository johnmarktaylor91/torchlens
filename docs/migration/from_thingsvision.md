# Migrating from thingsvision

[thingsvision](https://github.com/ViCCo-Group/thingsvision) (ViCCo-Group;
Muttenthaler & Hebart) made preprocessing correctness a first-class concern in
NeuroAI feature extraction, and its journey-shaped docs and
`get_transformations` / `show_model()` ergonomics inspired several TorchLens
surfaces. TorchLens deliberately covers a different slice: it verifies,
addresses, documents, and transports features from any PyTorch model you
bring, and intentionally omits model loading, stimulus management, and
analysis pipelines (see the [NeuroAI journey](../neuroai/journey.md) for the
end-to-end workflow and the
[loader interop page](../neuroai/loaders.md) for two-line model loading per
ecosystem).

The shape of the migration:

| thingsvision construct | TorchLens equivalent |
| --- | --- |
| `get_extractor(model_name, source, ...)` | Load the model with its own library (torchvision, timm, transformers, open_clip); TorchLens has no zoo. |
| bundled matched preprocessing | Resolve YOUR loader's own preprocessing and verify it: `torchlens.preprocessing.resolve(...)` / `.audit(...)`. |
| `extractor.show_model()` | `torchlens.inventory.list_sites(model, x)` -- structured rows with copy-paste-safe selectors, shapes, and pass indices. |
| `extract_features(batches, module_names=[...])` | `tl.extract(model, x, [...])` (one batch) or `tl.extract_dataset(...)` (datasets, disk artifacts, resume). |
| `save_features(features, out_path, file_format=...)` | disk-mode `tl.extract_dataset(..., output_dir=...)` + `tl.load_extraction(...)`; per-site adapters in `torchlens.bridge`. |

Two things to watch when carrying habits over:

- **Layer names translate.** thingsvision's `module_names` are torch module
  addresses (`"features.10"`, `"avgpool"`); TorchLens accepts those SAME
  qualified addresses, plus operation labels (`"conv2d_2_5"`) and
  pass-qualified spellings for recurrent sites (`"linear_1_1:2"`). Run
  `torchlens.inventory.list_sites(model, x)` once and copy selectors from the
  first column.
- **The `transform=` trap.** On `tl.trace`, `transform=` preprocesses INPUTS.
  On `tl.extract_dataset`, `transform=` transforms OUTPUTS before storage
  (thingsvision's `apply_transforms` habit). For input preprocessing in
  `extract_dataset`, use `input_transform=` -- carrying the trace habit into
  `extract_dataset(transform=...)` would silently z-score your FEATURES
  instead of your PIXELS.

## The same task on both sides: ResNet-50 penultimate features

thingsvision (current `module_names` list API):

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

TorchLens equivalent (same architecture, same site, no zoo -- construct the
model with torchvision directly):

```python
# migration-test: tool=torchlens expected=[['avgpool'], [2, 2048, 1, 1]]
import torch
from torchvision.models import resnet50

import torchlens as tl

model = resnet50(weights=None).eval()
features = tl.extract(model, torch.zeros(2, 3, 64, 64), ["avgpool"])
RESULT = [sorted(features), list(features["avgpool"].shape)]
```

With real weights, load them the torchvision way and resolve the matched
preprocessing from the SAME weights object -- TorchLens verifies it instead of
bundling its own recipe table:

```python
from torchvision.models import ResNet50_Weights, resnet50

import torchlens as tl
import torchlens.preprocessing as pp

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights).eval()
pre = pp.resolve(weights)               # the loader's own declared recipe
print(pre.record.description)           # resize=232 crop=224 (V2 -- not 256/224!)

features = tl.extract_dataset(
    model,
    images,                             # any iterable of PIL images
    ["avgpool"],
    output_dir="resnet50_features",
    input_transform=pre,                # applied AND recorded, verdict "verified"
    stimulus_ids=image_ids,
)
```

## The CLIP image embedding, both sides

thingsvision exposes CLIP through its custom source; with TorchLens you load
the model with `transformers` and take the projected embedding explicitly --
one visible projection line, so you always know WHICH representation you are
taking (the 768-d `post_layernorm` output is NOT the 512-d embedding):

```python
import torch
from transformers import AutoImageProcessor, CLIPModel

import torchlens as tl
import torchlens.preprocessing as pp

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
pre = pp.resolve(processor)

pixel_values = pre.transform(images)["pixel_values"]
out = tl.extract(model, [input_ids, pixel_values], ["vision_model.post_layernorm"])
with torch.no_grad():
    image_embeds = model.visual_projection(out["vision_model.post_layernorm"])
```

On transformers 5.x, `CLIPModel.get_image_features` no longer returns the
embedding tensor (it returns the full vision output object), and the
`image_embeds` on the forward output are L2-normalized -- the manual
projection above matches them exactly after normalization.

## Export and analysis

thingsvision's `save_features(..., file_format=...)` maps to the disk-mode
extraction artifact plus per-site adapters -- the artifact's manifest is the
row-order authority and carries the input-preprocessing provenance block:

```python
loaded = tl.load_extraction("resnet50_features")
print(loaded.input_preprocessing["verdict"])    # "verified" / "mismatch" / "unknown"

from torchlens.bridge import rsatoolbox as tl_rsa
dataset = tl_rsa.dataset("resnet50_features", site="avgpool")  # layer-wise RSA
```

For gLocal and other learned alignment transforms, keep using thingsvision's
own implementations over the exported matrices -- see
[bring your own alignment](../neuroai/byo_alignment.md).
