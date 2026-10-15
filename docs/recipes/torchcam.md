# Grad-CAM with torch-cam beside TorchLens

[TorchCAM](https://github.com/frgfm/torch-cam) computes class activation maps:
Grad-CAM, Grad-CAM++, Smooth Grad-CAM++, Score-CAM, Layer-CAM and others.
TorchLens ships one CAM natively, `tl.attribution.grad_cam`, and no torch-cam
extra. Use torch-cam when you need another CAM variant; use the native
`grad_cam` when you want the layer chosen from a TorchLens trace, token grids
for vision transformers, and the result's provenance in `extra`. On the same
model and layer the two Grad-CAM maps agree, which this page checks.

```text
pip install torchlens torchvision matplotlib
pip install --no-deps torchcam==0.4.0
```

`torchcam` 0.4.0 declares `numpy<2`, which would downgrade numpy under current
torch. Its other dependencies (`torch`, Pillow, matplotlib) come with the
first line, and it runs on numpy 2, so `--no-deps` is enough.

## Both libraries choose the same layer

torch-cam locates its target layer from a dummy forward pass when you name
none. TorchLens chooses from a metadata-only trace of your own call: the
deepest once-firing module that outputs a spatial map, lies on every path from
the input to the output, and carries a nonzero target gradient. On a
torchvision ResNet both choose `layer4`.

```python
# recipe-test: tool=torchcam expected=('layer4', 'layer4', 'trace')
import torch
import torchvision
from torchcam.methods import GradCAM

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.randn(1, 3, 64, 64)

extractor = GradCAM(model)
scores = model(x)
torchcam_map = extractor(3, scores, normalized=False)[0]
extractor.remove_hooks()

native = tl.attribution.grad_cam(model, x, target=3, overlay=False)

RESULT = (extractor.target_names[0], native.extra["layer"], native.extra["layer_source"])
```

## The maps agree

torch-cam returns the map at the layer's own resolution; TorchLens upsamples
it bilinearly to the input grid and reports the native size in
`extra["native_map_resolution"]`. Upsampling the torch-cam map the same way
gives the TorchLens values.

```python
# recipe-test: tool=torchcam expected=((2, 2), True)
import torch.nn.functional as F

upsampled = F.interpolate(
    torchcam_map.unsqueeze(1), size=x.shape[-2:], mode="bilinear", align_corners=False
)
RESULT = (
    native.extra["native_map_resolution"],
    torch.allclose(upsampled, native.values, atol=1e-5),
)
```

## When to use which

- Another CAM variant (Grad-CAM++, Score-CAM, Layer-CAM): torch-cam.
- A vision transformer layer: `tl.attribution.grad_cam`, which lays the tokens
  on the grid the trace proves and drops the class, distillation or register
  tokens it finds; see the [attribution reference](../reference/attribution.md).
- pytorch-grad-cam instead: the `tl.bridge.gradcam` bridge.
