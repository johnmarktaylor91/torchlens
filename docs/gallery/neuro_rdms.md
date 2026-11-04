# Neuro extraction with an order-error control

Representational similarity analysis compares stimuli by the dissimilarity of a layer's
activations (an RDM). This entry extracts per-layer RDMs with TorchLens on ResNet-18 and ViT-B/16
and checks them against independent plain-hook extraction and against rsatoolbox 0.3.2. A
control shuffles the stimulus order on one side: the RDMs must stop agreeing and the stamp must
flag the mismatch, so an order bug cannot pass silently. The entry passes on both architectures.
The layer-ranking target is synthetic, and no neuroscience benchmark result is claimed.

| Item | Recorded setup |
| --- | --- |
| Claim | Layer-wise and residual-add RDMs agree with independent extraction and rsatoolbox. |
| Script and run line | `python examples/gallery/neuro_rdms.py --out outputs/neuro_rdms/resnet18` (add `--arch vit_b_16` for ViT-B/16) |
| Pinned inputs | ResNet-18 and ViT-B/16, `IMAGENET1K_V1`, eval, float32; first 24 Fashion-MNIST test images in split order; RGB, antialiased bilinear 224x224, ImageNet normalization; seed 17. |
| Data hash | Compressed-file SHA256 `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073`. |
| Verdict | Passed on both; Linux x86_64 CPU, 3 November 2026, TorchLens 2.36.3, commit `01497c6d`. |
| Cost | ResNet-18: 19.8 s analysis, 31.1 s process wall, 1.08 GiB peak memory. ViT-B/16: 48.8 s analysis, 60.8 s process wall, 2.64 GiB peak memory. Two CPU threads each. |

The [Fashion-MNIST source](https://github.com/zalandoresearch/fashion-mnist) supplies images
under its MIT license. The hash is independently listed by
[TensorFlow Datasets](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/fashion_mnist.txt).
The script verifies the hash before decompression and preserves identifiers
`fashion-mnist:test:00000` onward. It does not redistribute the dataset.

## Oracles

The rules are the ones registered for this entry before any run.

| Rule | ResNet-18 | ViT-B/16 |
| --- | --- | --- |
| Every per-layer RDM from TorchLens has Pearson >= 0.999 with the RDM from independent plain-hook extraction | minimum `0.9999999999999988`, pass | minimum `0.9999999999988609`, pass |
| Every RDM from `tl.neuro.rdms` matches rsatoolbox's own `calc_rdm` pipeline on the same TorchLens activations to max abs 1e-6 | maximum `4.97e-13`, pass | maximum `6.86e-14`, pass |
| Same best layer on the synthetic final-stage target | `avgpool` for both, pass | `encoder.ln` for both, pass |

ResNet includes five stages and all eight pre-ReLU residual additions. ViT includes three
stages, the positional sum and all 24 encoder residual additions; its fifth encoder layer's
output is also one of the residual sums, so 28 requested sites resolve to 27 unique ones. The
independent path uses plain forward hooks and recomputes each residual sum itself.

The first run of this script compared TorchLens RDMs with rsatoolbox RDMs built from the
plain-hook activations instead, at the same 1e-6 bar. On ResNet-18 that difference is the same
`4.97e-13`. On ViT-B/16 it is `1.065e-6`, at the fifth encoder layer. The reason is that TorchLens
switches off PyTorch's fused attention fast path while it captures, so that every operation
inside attention is visible, and the fused kernel rounds differently. With the fast path also
switched off for the plain hooks (`torch.backends.mha.set_fastpath_enabled(False)`), a separate
check on random input found the plain-hook and TorchLens activations bitwise equal at all twelve
encoder layers and the final norm.

## Controls

Shuffle one side with NumPy seed 123: every correlation must fall below 0.5. ResNet-18's range
was `[-0.0679, -0.0272]` and ViT-B/16's `[-0.0851, 0.0005]`: both pass. Comparing the stored
stimulus identifiers flags the mismatch on both, and both stamps record
`stimulus_order_mismatch=true`.

## Artifact and replay

Each run saves a portable `neuro_rdms.tlspec` holding the activations, addresses, stimulus IDs,
preprocessing, shuffled IDs and RDMs. After `tl.load`, the script recomputes the first RDM from
the reloaded activations: both replay errors are exactly 0, and the stored order mismatch stays
flagged.

| Run | Bytes | sha256 |
| --- | --- | --- |
| ResNet-18 | 109,342,780 | `4a6c85af9284bfaa`... |
| ViT-B/16 | 394,121,159 | `47ce4b55200b1e72`... |

## Basis-grid row

| Verbs | TorchLens lines | Plain lines | Architectures |
| --- | --- | --- | --- |
| capture, select, read, compare, save | 252 | 56 | ResNet-18, ViT-B/16 |

Counts are nonblank, noncomment, non-docstring lines. The TorchLens region includes setup, the
command line, plotting, reporting and helpers; the plain region covers independent extraction,
residual recomputation and rsatoolbox. Lines of the original thingsvision reference were not
measured.

## The stamp

ResNet-18:

```text
{
  "commit": "01497c6dcea4b23f5fe5ee3401fc3678fa757123",
  "date_utc": "2026-11-03T15:25:47Z",
  "device": "cpu",
  "libraries": {
    "numpy": "2.2.6",
    "rsatoolbox": "0.3.2",
    "torchvision": "0.28.0+cpu",
    "transformers": "5.18.0"
  },
  "machine": "x86_64",
  "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31",
  "python": "3.10.19",
  "stimulus_order_mismatch": true,
  "torch": "2.13.0+cpu",
  "torch_threads": 2,
  "torchlens": "2.36.3",
  "wall_seconds": 19.8
}
```

ViT-B/16:

```text
{
  "commit": "01497c6dcea4b23f5fe5ee3401fc3678fa757123",
  "date_utc": "2026-11-03T15:26:45Z",
  "device": "cpu",
  "libraries": {
    "numpy": "2.2.6",
    "rsatoolbox": "0.3.2",
    "torchvision": "0.28.0+cpu",
    "transformers": "5.18.0"
  },
  "machine": "x86_64",
  "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31",
  "python": "3.10.19",
  "stimulus_order_mismatch": true,
  "torch": "2.13.0+cpu",
  "torch_threads": 2,
  "torchlens": "2.36.3",
  "wall_seconds": 48.8
}
```

## Vary it

Use `--arch vit_b_16` to repeat with ViT-B/16, `--stimuli 48` to extend the ordered prefix,
or `--quick` for eight stimuli. These small examples avoid downloads and optional neuro
packages; they exercise extraction and the core correlation-RDM path used by the neuro handoff.

```python
# recipe-test: tool=torchvision expected=True
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
import torchlens as tl

torch.manual_seed(17)
model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), BasicBlock(4, 4)).eval()
x = torch.randn(8, 3, 8, 8)
features = tl.extract(model, x, {"residual_block": "1"})["residual_block"]
with torch.no_grad():
    independent = model(x)
RESULT = bool(torch.equal(features, independent))
```

```python
# recipe-test: tool=torchvision expected=True
import numpy as np

rdm = tl.repgeom.rdm(features, metric="correlation")
reference = tl.repgeom.rdm(independent, metric="correlation")
ids = [f"stimulus-{i}" for i in range(8)]
permutation = torch.tensor([3, 0, 6, 1, 7, 4, 2, 5])
wrong_ids = [ids[i] for i in permutation.tolist()]
RESULT = bool(np.array_equal(rdm, reference)) and ids != wrong_ids
```
