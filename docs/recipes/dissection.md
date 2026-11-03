# Network dissection: label units with CLIP-Dissect

CLIP-Dissect (Oikarinen and Weng, 2023, "CLIP-Dissect: Automatic Description of
Neuron Representations in Deep Vision Networks", ICLR,
https://arxiv.org/abs/2204.10965) names what a unit responds to without
labelled data. Run a probe set of images through the network and record each
unit's mean activation per image; ask CLIP how well each image matches each
word of a concept set; then score every pair of unit and concept by how
strongly the unit's top images match the concept, compared with how strongly
every unit's top images do. The unit's label is its best-scoring concept.

TorchLens does the network side: `tl.extract_dataset` runs the probe set and
pools each unit over space, and `tl.stats.TopK` ranks each unit's top images
with their example ids. CLIP's side is any CLIP model; this page builds a
stand-in from the same concepts the probe set contains, so the labels can be
checked. The score is the paper's soft-WPMI, below as plain PyTorch.

```text
pip install torchlens
pip install "open_clip_torch==3.3.0"  # for real use only; the stand-in needs neither
```

## A probe set with planted concepts

Ninety 16 by 16 images: each is tinted red, green or blue, and carries
horizontal stripes, vertical stripes or no texture, with a little noise. The
concept set adds two words no image shows, "dots" and "grey". The network's
first convolution has five planted units, one per concept present: three
colour-opponent units and two edge units.

```python
# recipe-test: tool=torchlens expected=(90, ['input_1', 'conv2d_1_1', 'relu_1_2', 'mean_1_3', 'linear_1_4', 'output_1'])
import torch
from torch import nn

import torchlens as tl

CONCEPTS = ["red", "green", "blue", "horizontal stripes", "vertical stripes", "dots", "grey"]
PLANTED = ["red", "green", "blue", "horizontal stripes", "vertical stripes"]
TEXTURES = ["horizontal stripes", "vertical stripes", None]

generator = torch.Generator().manual_seed(0)
bands = (torch.arange(16) % 4 < 2).float()
images, present = [], torch.zeros(90, len(CONCEPTS))
for index in range(90):
    tint, texture = index % 3, TEXTURES[(index // 3) % 3]
    image = 0.1 * torch.rand(3, 16, 16, generator=generator)
    image[tint] += 0.6
    present[index, tint] = 1.0
    if texture is not None:
        image += 0.3 * (bands[:, None] if texture == "horizontal stripes" else bands[None, :])
        present[index, CONCEPTS.index(texture)] = 1.0
    images.append(image)
probe = torch.stack(images)


class Probed(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, 3)
        self.head = nn.Linear(5, 10)
        weight = torch.zeros(5, 3, 3, 3)
        for colour in range(3):  # tint minus the other two channels
            weight[colour, :, 1, 1] = -0.5
            weight[colour, colour, 1, 1] = 1.0
        weight[3, :, 0, 1], weight[3, :, 2, 1] = 1.0, -1.0  # a change down the rows
        weight[4, :, 1, 0], weight[4, :, 1, 2] = 1.0, -1.0  # a change across the columns
        with torch.no_grad():
            self.conv.weight.copy_(weight)
            self.conv.bias.zero_()

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))


torch.manual_seed(0)
model = Probed().eval()
labels = tl.trace(model, probe[:1]).layer_labels
RESULT = (len(probe), labels)
```

## Extract and rank

The units are the outputs of `relu_1_2`, a functional op. `pool="spatial_mean"`
averages each unit over space, CLIP-Dissect's default `pool_mode="avg"`.
`TopK(k=10, unit_axis=1)` keeps each unit's ten most active probe images,
highest first; the paper uses the top 100 of a larger probe set.

```python
# recipe-test: tool=torchlens expected=((90, 5), (5, 10))
activations = tl.extract_dataset(
    model, probe, ["relu_1_2"], batch_size=32, pool="spatial_mean", progress=False
)["relu_1_2"]
top = tl.stats.TopK(k=10, unit_axis=1)
top.update(activations)
ranked = top.compute()
print(ranked)
for unit in range(5):
    print(f"unit {unit}: top images {ranked.example_ids[unit, :5].tolist()}")
RESULT = (tuple(activations.shape), tuple(ranked.example_ids.shape))
```

```text
UnitTopK(units=5, k=10, position_axes=0, unit_axis=1)
unit 0: top images [57, 33, 48, 9, 69]
unit 1: top images [43, 76, 34, 79, 55]
unit 2: top images [71, 65, 86, 68, 41]
unit 3: top images [28, 74, 54, 55, 2]
unit 4: top images [57, 39, 68, 84, 77]
```

## CLIP's side: image and concept similarity

CLIP-Dissect needs one matrix: the cosine similarity of every probe image's
CLIP embedding with every concept's text embedding. The stand-in embeds each
concept as one of seven orthonormal directions; an image's embedding is the sum
of the directions of the concepts it shows plus noise, and a concept's text
embedding is its direction plus a little noise.

```python
# recipe-test: tool=torchlens expected=(90, 7)
def concept_similarity(image_features, text_features):
    """Cosine similarity of every image with every concept: [images, concepts]."""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return image_features @ text_features.T


clip_generator = torch.Generator().manual_seed(2)
directions = torch.linalg.qr(torch.randn(32, len(CONCEPTS), generator=clip_generator))[0].T
image_features = present @ directions + 0.5 * torch.randn(90, 32, generator=clip_generator)
text_features = directions + 0.1 * torch.randn(len(CONCEPTS), 32, generator=clip_generator)
similarity = concept_similarity(image_features, text_features)
RESULT = tuple(similarity.shape)
```

## Score with soft-WPMI and label

The paper's soft-WPMI, as its reference code computes it
(https://github.com/Trustworthy-ML-Lab/CLIP-dissect, `similarity.py`,
`soft_wpmi`). For unit `k` and concept `m`: the softmax over concepts of `a`
times the similarity gives `P(m | x)` for each image; an image at rank `r` of
the unit's top `K` belongs to the unit's set with probability falling linearly
from `p_start = 0.998` to `p_end = 0.97`; the log of the soft-set probability is
`log P(m | k) = sum over the top images of log(1 + p_r (P(m | x_r) - 1) + 1e-7)`;
the concept's prior `log P(m)` is the log of the mean of `P(m | k)` over every
unit of the layer; and the score is `log P(m | k) - lambda log P(m)`, with
`a = 10` and `lambda = 1`:

```python
# recipe-test: tool=torchlens expected=True
import math


def soft_wpmi(similarity, top_ids, *, a=10.0, lam=1.0, p_start=0.998, p_end=0.97, min_prob=1e-7):
    """CLIP-Dissect's soft-WPMI of every unit and concept: [units, concepts].

    similarity: [images, concepts] cosine similarities; top_ids: [units, K] each
    unit's top images, most active first (UnitTopK.example_ids).
    """
    p_concept = torch.softmax(a * similarity.double(), dim=1)
    k = top_ids.shape[1]
    p_in_set = p_start - torch.arange(k, dtype=torch.float64) / k * (p_start - p_end)
    soft = 1 + p_in_set[None, :, None] * (p_concept[top_ids] - 1)  # [units, K, concepts]
    log_given_unit = torch.log(soft + min_prob).sum(dim=1)
    log_prior = torch.logsumexp(log_given_unit, dim=0) - math.log(log_given_unit.shape[0])
    return log_given_unit - lam * log_prior


scores = soft_wpmi(similarity, ranked.example_ids)
best = scores.argmax(dim=1)
labels = [CONCEPTS[index] for index in best.tolist()]
for unit, label in enumerate(labels):
    print(f"unit {unit}: {label} ({float(scores[unit].max()):.2f})")
RESULT = labels == PLANTED
```

```text
unit 0: red (1.61)
unit 1: green (1.61)
unit 2: blue (1.61)
unit 3: horizontal stripes (1.61)
unit 4: vertical stripes (1.54)
```

Each planted unit gets its planted concept, and neither absent concept labels a
unit. The prior is a mean over the layer's units, so a concept that many units'
top images share scores lower for each of them: score a whole layer at once, as
the reference code does, and read each label with its score beside it.

## Real use: a torchvision model with OpenCLIP

Not run here: the weights and the probe set download. Any probe set works; the
paper uses Broden or the ImageNet validation set, and the concept set is a word
list (the paper's 20,000 most common English words, or a dataset's class
names). The CLIP side is OpenCLIP's ViT-B/32; the network is a torchvision
ResNet-18, dissected at `layer4`:

```text
import open_clip
import torch
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18

import torchlens as tl

clip, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
weights = ResNet18_Weights.IMAGENET1K_V1
net = resnet18(weights=weights).eval()

pictures = [image for image, _ in CIFAR10("data", train=False, download=True)][:1000]
concepts = [line.strip() for line in open("concepts.txt") if line.strip()]

with torch.no_grad():
    image_features = clip.encode_image(torch.stack([clip_preprocess(p) for p in pictures]))
    text_features = clip.encode_text(tokenizer(concepts))
similarity = concept_similarity(image_features.float(), text_features.float())

activations = tl.extract_dataset(
    net, torch.stack([weights.transforms()(p) for p in pictures]), ["layer4"],
    batch_size=64, pool="spatial_mean", progress=False,
)["layer4"]
top = tl.stats.TopK(k=100, unit_axis=1)
top.update(activations)
scores = soft_wpmi(similarity, top.compute().example_ids)
labels = [concepts[index] for index in scores.argmax(dim=1).tolist()]
```

`top.compute().example_ids[unit]` are the probe images to look at beside each
label. The soft-WPMI step needs only each unit's top image ids and the similarity
rows of those images, so the activations can be ranked in batches with repeated
`top.update(...)` calls instead of held all at once.
