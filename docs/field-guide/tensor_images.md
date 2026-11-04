# Look at a tensor as an image

Two functions draw any tensor as a picture, with no capture needed:
`tl.viz.rgb(t)` draws a three-channel image, and `tl.viz.chans(t)` draws the
leading channels as a grid of grayscale tiles. Both return a PIL image, which a
notebook shows inline and `.save(path)` writes. They are functions on purpose:
importing TorchLens never adds attributes to `torch.Tensor`.

`tl.viz.chans` is `tl.viz.nodes.channel_grid` applied to one tensor, so each
tile is min-max scaled on its own and tiles share no scale. `tl.viz.rgb` scales
the whole image together, the same way `show(kind="rgb")` does, unless
`normalize=` says otherwise.

## Minimal example on the running model

This page uses [ResNet-18 on one image](running_examples.md#resnet-18-on-one-image).
The first block's output is 64 channels of 56 by 56; `chans` draws the first 16.

```python
# recipe-test: tool=torchvision expected=True
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.rand(1, 3, 224, 224)

trace = tl.trace(model, x)

grid = tl.viz.chans(trace["layer1"].out)
RESULT = grid.mode == "RGB" and max(grid.size) <= 300
```

## 1. The input image

`rgb` takes `(3, H, W)` or a batch `(B, 3, H, W)`, whose first element it draws.
A small image grows by a whole-number factor so each pixel stays a square block;
a large one shrinks to `max_size` on its longest side.

```python
# recipe-test: tool=torchvision expected=(224, 224)
picture = tl.viz.rgb(x)
RESULT = picture.size
```

## 2. Normalized inputs and channels-last arrays

`normalize=` picks the scaling: `"minmax"` (the default) stretches the image's
range to fill black to white, `"clip"` keeps values already in `[0, 1]` as they
are, and `"imagenet"` undoes the ImageNet mean and standard deviation that
torchvision's `Normalize` applied. Pass `channels_last=True` for an
`(H, W, 3)` tensor.

```python
# recipe-test: tool=torchvision expected=True
import numpy as np

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
normalized = (x - mean) / std
undone = tl.viz.rgb(normalized, normalize="imagenet")
as_is = tl.viz.rgb(x, normalize="clip")
hwc = tl.viz.rgb(x[0].permute(1, 2, 0), channels_last=True, normalize="clip")
undone_px, as_is_px = np.asarray(undone, dtype=int), np.asarray(as_is, dtype=int)
RESULT = int(abs(undone_px - as_is_px).max()) <= 1 and bool((as_is_px == np.asarray(hwc)).all())
```

## 3. Save

Both return PIL images; `.save` writes any format PIL knows from the suffix.

```python
# recipe-test: tool=torchvision expected=True
import pathlib

tl.viz.chans(trace["layer1"].out, n=8).save("layer1.png")
tl.viz.rgb(x).save("input.png")
RESULT = pathlib.Path("layer1.png").exists() and pathlib.Path("input.png").exists()
```

## What it refuses

`rgb` needs exactly three channels on the channel axis and names the fix
otherwise (`chans` for another count, `channels_last=` for the other axis).
`chans` needs `(C, H, W)` or `(B, C, H, W)`.

```python
# recipe-test: tool=torchvision expected=True
try:
    tl.viz.rgb(trace["layer1"].out)
except ValueError as exc:
    RESULT = "chans()" in str(exc)
```
