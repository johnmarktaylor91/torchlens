# Loading models for feature extraction (two lines per ecosystem)

TorchLens has no model zoo, and that is a feature: every loader below already
ships the model AND its matched preprocessing, maintained by the people who
trained it. Load with the ecosystem's own two lines, then hand the SAME
authority object to `torchlens.preprocessing.resolve` -- the resolution's
`record.description` prints the declared recipe, and the audit can verify
what you actually ran against it.

Model names are case-sensitive in every ecosystem below; copy them exactly.

**torchvision** (the weights enum IS the authority):

```python
from torchvision.models import ResNet50_Weights, resnet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()

pre = torchlens.preprocessing.resolve(ResNet50_Weights.IMAGENET1K_V2)
print(pre.record.description)   # torchvision preset: resize=232 crop=224
```

Note the V2 recipe resizes to 232, not the folk-memory 256 -- exactly the
kind of drift a resolved authority catches and a hand-typed constant table
misses.

**Hugging Face** (the image processor is the authority; fetching it can touch
the network -- the resolution discloses whether a fetch was attempted):

```python
from transformers import AutoImageProcessor, CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
pre = torchlens.preprocessing.resolve(processor)
print(pre.record.description)
```

**timm** (the model carries its own data config; no network needed):

```python
import timm
model = timm.create_model("resnet18.a1_in1k", pretrained=True).eval()

pre = torchlens.preprocessing.resolve(model=model)
print(pre.record.description)
```

**open_clip** (the returned preprocess pipeline is the authority):

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

pre = torchlens.preprocessing.resolve(preprocess)
print(pre.record.description)
```

**torch.hub** (models declare no standard preprocessing metadata; declare
what you use explicitly and the audit compares against it):

```python
model = torch.hub.load("pytorch/vision", "resnet18", weights=None).eval()

pre = torchlens.preprocessing.resolve({"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
print(pre.record.description)
```

## When the authority declares nothing

Some real, shipped model families declare NO preprocessing constants because
none apply. torchvision's detection presets are the canonical case: a
`FasterRCNN_ResNet50_FPN` normalizes internally, a plain `[0, 1]` float batch
is the CORRECT input, and the preset exposes no mean/std/size:

```python
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
pre = torchlens.preprocessing.resolve(FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
print(pre.declared)      # None -- the authority declares nothing
```

The honest audit verdict for such a family is `unknown`, never a guess and
never a false alarm on the correct input. A verifier can only be as
declarative as the authority; `unknown` is the one answer a bundled recipe
table could never give you.

## Version notes

- On transformers 5.x, `CLIPModel.get_image_features` returns the full vision
  output object rather than the embedding tensor; take the projected
  embedding explicitly (`model.visual_projection(...)` over the
  `vision_model.post_layernorm` site) -- see the
  [migration page](../migration/from_thingsvision.md).
- Packaging: TorchLens installs into YOUR environment and gates optional
  integrations behind extras, so the loaders above stay at whatever versions
  you already run. (thingsvision, by comparison, currently pins
  `tensorflow<2.16`, `torchvision==0.15.2`, `transformers==4.40.1`, and
  `numpy<2` as mandatory dependencies -- a factual packaging difference,
  checkable in both projects' metadata.)
