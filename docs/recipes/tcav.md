# TCAV with a causal check

TCAV (Kim et al., 2018) asks whether a human concept matters to a class: learn
a concept activation vector (CAV) at one layer, then count how often the class
logit increases along it. A TCAV score alone is correlational. This recipe adds
the causal check: steer the layer along the CAV and measure the change in the
class score. TorchLens supplies the activations, the gradients and the
steering; TCAV itself stays a recipe. For a packaged TCAV with its statistics,
use Captum's `captum.concept.TCAV`.

## Setup: a model whose class depends on a known direction

The toy task makes the answer known: class 1 means the input points along a
direction `d`, so "points along `d`" is a concept the model must use.

```python
# recipe-test: tool=torchlens expected=True
import torch
from torch import nn

import torchlens as tl

torch.manual_seed(0)
d = torch.randn(8)
d = d / d.norm()
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_x = torch.randn(512, 8)
train_y = (train_x @ d > 0).long()
for _ in range(300):
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(train_x), train_y).backward()
    optimizer.step()
model.eval()

test_x = torch.randn(256, 8)
accuracy = (model(test_x).argmax(1) == (test_x @ d > 0).long()).float().mean().item()
RESULT = accuracy > 0.9
```

## 1. Learn the CAV at a layer

Concept examples carry the direction; random examples do not. The CAV is the
unit normal of a linear classifier separating their activations at site `"1"`
(the ReLU output).

```python
# recipe-test: tool=torchlens expected=True
SITE = "1"


def site_activations(x):
    return tl.extract(model, x, [SITE])[SITE]


def learn_cav(positive, negative, steps=300):
    """Unit normal of a logistic-regression boundary between two activation sets."""

    feats = torch.cat([positive, negative])
    labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
    w = torch.zeros(feats.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(feats @ w + b, labels).backward()
        opt.step()
    return (w / w.norm()).detach()


concept_x = 0.5 * torch.randn(100, 8) + 2.5 * d
random_x = torch.randn(100, 8)
cav = learn_cav(site_activations(concept_x), site_activations(random_x))
RESULT = cav.shape == (16,)
```

## 2. The TCAV score

The score is the fraction of class-1 inputs whose class-1 logit has a positive
directional derivative along the CAV at the site. TorchLens saves the site's
gradient during one backward pass.

```python
# recipe-test: tool=torchlens expected=True
def tcav_score(direction, x, target=1):
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
    logits = trace[trace.output_layers[0]].out
    trace.log_backward(logits[:, target].sum())  # rows are independent: per-example grads
    grads = trace[SITE].grad
    return (grads @ direction > 0).float().mean().item()


class_x = test_x[test_x @ d > 0]
score = tcav_score(cav, class_x)

# Random-vs-random CAVs give the null distribution the score must stand out from.
null_scores = [
    tcav_score(learn_cav(site_activations(torch.randn(100, 8)), site_activations(random_x)), class_x)
    for _ in range(5)
]
RESULT = score > 0.9 and score > max(null_scores)
```

## 3. The causal check

Steer the site along the CAV and read the class margin (class-1 logit minus
class-0 logit). If the concept matters, adding the CAV raises the margin and
subtracting it lowers it; a random direction of the same norm moves it less.

```python
# recipe-test: tool=torchlens expected=True
def mean_margin(direction, magnitude, x):
    spec = tl.when(tl.module(SITE), tl.steer(direction, magnitude, feature_axis=1))
    with torch.no_grad():
        logits = spec.bind(model)(x)
    return (logits[:, 1] - logits[:, 0]).mean().item()


margins = {m: mean_margin(cav, m, class_x) for m in (-2.0, 0.0, 2.0)}
random_direction = torch.randn(16)
random_direction = random_direction / random_direction.norm()
random_shift = abs(mean_margin(random_direction, 2.0, class_x) - margins[0.0])
cav_shift = margins[2.0] - margins[0.0]
RESULT = margins[-2.0] < margins[0.0] < margins[2.0] and cav_shift > random_shift
```

## Limits

- A high TCAV score with a failed causal check means the CAV tracks something
  the class correlates with but does not use at that layer; report both.
- The CAV depends on the concept and random sets. Repeat with fresh random
  sets and report the spread, as the null scores above do once.
- Steering adds the direction at every row of the batch. For token models,
  pick the positions explicitly.
