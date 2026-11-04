# CRAFT: concepts by factorizing activations, ranked by importance

CRAFT (Fel et al., 2023) explains a class without a list of human concepts. It cuts images of
the class into crops, takes one layer's non-negative activations on the crops, and factorizes
them with non-negative matrix factorization (NMF), `A = U W` approximately, where each row of `W` is a
concept (a direction in activation space) and `U` holds how much of each concept every crop
contains. It then ranks the concepts by how much the class score depends on them, with total
Sobol indices computed by masking the concept coefficients of whole images and re-running the
rest of the network. Where [TCAV](tcav.md) tests a concept you name, CRAFT proposes the
concepts; the two meet at the same place, a direction at one site whose effect on the output
you can test.

Paper: Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien
Colin, Rémi Cadène and Thomas Serre, "CRAFT: Concept Recursive Activation FacTorization for
Explainability", CVPR 2023, https://arxiv.org/abs/2211.10154. The authors' code is
https://github.com/deel-ai/Craft, and Xplique packages CRAFT for PyTorch and TensorFlow
(https://github.com/deel-ai/xplique). TorchLens supplies the activations and the site edit
that re-runs the network from changed concept coefficients; the factorization comes from
scikit-learn and the importance estimate is a few lines here. This page leaves out the
recursive step (factorizing again at an earlier layer) and the concept attribution maps.

## Minimal example on the running model

This page uses [ResNet-18 on one
image](running_examples.md#resnet-18-on-one-image). NMF needs non-negative activations;
`tl.extract` returns the pooled features after the last ReLU, which are. The recipe below
builds its own model for the method.

```python
# recipe-test: tool=sklearn expected=True
import torch
import torchvision

import torchlens as tl

torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None).eval()
x = torch.rand(1, 3, 224, 224)

trace = tl.trace(model, x)

pooled = tl.extract(model, x, ["avgpool"])["avgpool"]
RESULT = tuple(pooled.shape) == (1, 512, 1, 1) and bool((pooled >= 0).all())
```

## Setup: a small CNN whose class depends on one part

Each 24 by 24 image holds one 8 by 8 stripe patch on faint noise: horizontal stripes for
class 1, vertical for class 0. Half of all images, of either class, also hold a bright square,
which says nothing about the class. A good explanation of class 1 finds the horizontal
stripes and the square as separate concepts and ranks the stripes first.

```python
# recipe-test: tool=sklearn expected=True
import torch
from torch import nn

import torchlens as tl

SIZE, PATCH = 24, 8


def make_images(n, generator):
    """Images ``[n, 1, 24, 24]``, labels, and masks of the stripe patch and the square."""
    images = 0.1 * torch.rand(n, 1, SIZE, SIZE, generator=generator)
    labels = torch.randint(0, 2, (n,), generator=generator)
    stripe_mask = torch.zeros(n, SIZE, SIZE, dtype=torch.bool)
    square_mask = torch.zeros(n, SIZE, SIZE, dtype=torch.bool)
    rows = torch.arange(PATCH)
    for i in range(n):
        top, left = torch.randint(0, SIZE - PATCH + 1, (2,), generator=generator).tolist()
        lines = (rows % 2 == 0).float()
        patch = lines[:, None].expand(PATCH, PATCH) if labels[i] else lines[None, :].expand(PATCH, PATCH)
        images[i, 0, top : top + PATCH, left : left + PATCH] = patch
        stripe_mask[i, top : top + PATCH, left : left + PATCH] = True
        if torch.rand(1, generator=generator) < 0.5:
            while True:
                r, c = torch.randint(0, SIZE - 5, (2,), generator=generator).tolist()
                if not stripe_mask[i, max(r - 1, 0) : r + 6, max(c - 1, 0) : c + 6].any():
                    break
            images[i, 0, r : r + 5, c : c + 5] = 1.0
            square_mask[i, r : r + 5, c : c + 5] = True
    return images, labels, stripe_mask, square_mask


torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train = torch.Generator().manual_seed(1)
for _ in range(150):
    images, labels, _, _ = make_images(64, train)
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(images), labels).backward()
    optimizer.step()
model.eval()

test_images, test_labels, test_stripes, test_squares = make_images(400, torch.Generator().manual_seed(2))
with torch.no_grad():
    accuracy = (model(test_images).argmax(1) == test_labels).float().mean().item()
RESULT = accuracy > 0.95
```

## 1. Crops of the class and their activations

CRAFT works on crops so that one concept fills much of each input. The crops here are 12 by
12 tiles at a stride of 6 over the class-1 test images; the model ends in adaptive pooling, so
it takes them at that size. The site is the pooled feature vector after the last ReLU, `"6"`
(the `Flatten` output), which is non-negative. Each crop is tagged by the part it holds, for
the check in the next step: at least half of the stripe patch, at least half of the square, or
neither.

```python
# recipe-test: tool=sklearn expected=True
SITE, CROP, STRIDE = "6", 12, 6

class_images = test_images[test_labels == 1]
crops, tags = [], []
for image, stripes, square in zip(
    class_images, test_stripes[test_labels == 1], test_squares[test_labels == 1]
):
    for top in range(0, SIZE - CROP + 1, STRIDE):
        for left in range(0, SIZE - CROP + 1, STRIDE):
            window = (slice(top, top + CROP), slice(left, left + CROP))
            crops.append(image[:, window[0], window[1]])
            has_stripes = stripes[window].sum() >= stripes.sum() / 2
            has_square = square.any() and square[window].sum() >= square.sum() / 2
            tags.append("stripes" if has_stripes else "square" if has_square else "neither")
crops = torch.stack(crops)
crop_acts = tl.extract(model, crops, [SITE])[SITE]
RESULT = crop_acts.shape == (len(crops), 32) and bool((crop_acts >= 0).all())
```

## 2. Factorize: concepts are the rows of `W`

NMF with three components on the crop activations. Each crop's coefficients `U` say how much
of each concept it holds. The check reads the tags: the concept that is strongest on stripe
crops and the concept that is strongest on square crops are different rows of `W`.

```python
# recipe-test: tool=sklearn expected=True
import numpy as np
from sklearn.decomposition import NMF

nmf = NMF(n_components=3, init="nndsvda", max_iter=1000, random_state=0)
U = nmf.fit_transform(crop_acts.numpy())
W = torch.from_numpy(nmf.components_).float()  # [3 concepts, 32 features]
tags_array = np.array(tags)
mean_coefficients = {tag: U[tags_array == tag].mean(0) for tag in ("stripes", "square", "neither")}
stripe_concept = int(np.argmax(mean_coefficients["stripes"] - mean_coefficients["neither"]))
square_concept = int(np.argmax(mean_coefficients["square"] - mean_coefficients["neither"]))
error = np.linalg.norm(crop_acts.numpy() - U @ nmf.components_) / np.linalg.norm(crop_acts.numpy())
for tag, coefficients in mean_coefficients.items():
    print(f"{tag:>8} crops: mean concept coefficients {np.round(coefficients, 2)}")
print(f"relative reconstruction error {error:.2f}")
RESULT = stripe_concept != square_concept and error < 0.3
```

## 3. Concept importance with total Sobol indices

For whole class-1 images, `nmf.transform` gives each image's concept coefficients `u`. A mask
`m` in `[0, 1]^3` scales them, `(u * m) W` is written into the site in place of its activation,
and the rest of the network runs to the class-1 logit. The total Sobol index of a concept is
the share of the logit's variance over random masks that involves that concept, alone or
with others; the estimate below is Jansen's, with plain uniform masks (CRAFT uses a
quasi-Monte Carlo sequence). The site edit is a bound hook, so no capture runs per mask.

```python
# recipe-test: tool=sklearn expected=True
image_acts = tl.extract(model, class_images, [SITE])[SITE]
image_u = torch.from_numpy(nmf.transform(image_acts.numpy())).float()
edit = {}
bound = tl.when(tl.module(SITE), lambda activation, *, ctx: edit["value"]).bind(model)


def class_logit(u, masks, image):
    """Class-1 logit of ``image`` with its concept coefficients ``u`` scaled by each mask."""
    edit["value"] = (u * masks) @ W
    with torch.no_grad():
        return bound(image.expand(len(masks), -1, -1, -1))[:, 1]


def total_sobol(u, image, n=128, generator=None):
    a = torch.rand(n, 3, generator=generator)
    b = torch.rand(n, 3, generator=generator)
    f_a = class_logit(u, a, image)
    indices = []
    for concept in range(3):
        mixed = a.clone()
        mixed[:, concept] = b[:, concept]
        indices.append(((f_a - class_logit(u, mixed, image)) ** 2).mean() / (2 * f_a.var()))
    return torch.stack(indices)


masks = torch.Generator().manual_seed(3)
importance = torch.stack(
    [total_sobol(u, image[None], generator=masks) for u, image in zip(image_u[:32], class_images[:32])]
).mean(0)
print("total Sobol index per concept", [round(v, 2) for v in importance.tolist()])
RESULT = int(importance.argmax()) == stripe_concept and importance[stripe_concept] > 2 * importance[square_concept]
```

The stripe concept carries the class-1 logit; the square, which appears in both classes, is
found as a concept but matters little to the class.

## Limits

- The number of concepts is a choice. Too few merge parts, too many split one part over
  several rows; check the reconstruction error and look at the top crops of each concept.
- NMF needs non-negative activations: a site after a ReLU, or after pooling of one.
- Sobol indices estimated from random masks are noisy. Report them with the number of masks
  and images, and repeat with another seed before reading small differences.
- CRAFT explains a class through one layer's directions. To test one of those directions as
  a cause, steer along a row of `W` and read the class score, as the [TCAV causal
  check](tcav.md#3-the-causal-check) does.
