# Train a truncated backbone through `trace.run`

Feature-extractor libraries let you cut a pretrained network at a stage and
fine-tune the prefix. TorchLens does the same from one trace: trace the model
once, then call `trace.run(inputs=x, until=site, fast=True)` inside the
training loop. The native forward stops right after `site`, nothing
downstream runs, and the site's output comes back graph-connected, so the
loss backpropagates into every parameter upstream of the cut.

## The recipe

```python
# recipe-test: tool=torchvision expected=True
import torch
import torchvision
from torch import nn

import torchlens as tl

torch.manual_seed(0)
backbone = torchvision.models.resnet18(weights=None).eval()  # or weights="DEFAULT" (download)
head = nn.Linear(128, 10)  # layer2 has 128 channels
images = torch.randn(8, 3, 32, 32)
labels = torch.randint(0, 10, (8,))

trace = tl.trace(backbone, images[:1], save=tl.module("layer2"))  # trace once
optimizer = torch.optim.SGD([*backbone.parameters(), *head.parameters()], lr=0.05)

losses = []
for step in range(15):
    optimizer.zero_grad()
    result = trace.run(inputs=images, until="layer2", fast=True)
    features = result.trace["layer2"].out.mean(dim=(2, 3))  # global average pool
    loss = nn.functional.cross_entropy(head(features), labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

RESULT = losses[-1] < losses[0]
```

## What the cut guarantees

The run report says it stopped, and the layers after the cut never ran, so
their parameters received no gradient:

```python
# recipe-test: tool=torchvision expected=(True, 'live_stop_after', True, True)
upstream = [p.grad for n, p in backbone.named_parameters() if n.split(".")[0] in {"conv1", "layer2"}]
downstream = [p.grad for n, p in backbone.named_parameters() if n.split(".")[0] in {"layer3", "fc"}]
RESULT = (
    result.report.truncated,
    result.report.truncation.regime,
    all(g is not None for g in upstream),
    all(g is None for g in downstream),
)
```

## Limits

- The `until=` site must be one the trace saved (here `save=tl.module("layer2")`);
  another site refuses with `run_capability_unavailable` at stage
  `fast_until_site_unsaved`.
- For the cut run, the model output and later saved sites read `None`; the
  next full run refreshes them.
- The fast run refreshes the live trace in place and marks it truncated, with
  an `unverifiable` verdict. Save, export or validate the trace before
  training through it if you need it as evidence.
- The backbone stays in eval mode here, so BatchNorm running statistics do not
  update. A loaded artifact's fast tier refuses `run_fast_until_unsupported`;
  train from a live trace.
