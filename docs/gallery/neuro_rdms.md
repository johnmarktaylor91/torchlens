# Neuro extraction with an order-error control

This entry checks extraction against independent PyTorch hooks and rsatoolbox 0.3.2.
The 24-stimulus ResNet-18 run passed; the 8-stimulus ViT-B/16 check exceeded the fixed RDM
error limit, so the entry did not pass and moves to the next batch. The layer-ranking target
is synthetic, and no neuroscience benchmark result is claimed.

| Item | Recorded setup |
| --- | --- |
| Claim | Layer-wise and residual-add RDMs agree with independent extraction and rsatoolbox. |
| Script and run line | `python examples/gallery/neuro_rdms.py --out outputs/neuro_rdms/resnet18` |
| Pinned inputs | ResNet-18 `IMAGENET1K_V1`, eval, float32; first 24 Fashion-MNIST test images in split order; RGB, antialiased bilinear 224x224, ImageNet normalization; seed 17. ViT-B/16 uses the same setup with eight images under `--quick`. |
| Data hash | Compressed-file SHA256 `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073`. |
| Verdict | Did not pass overall. ResNet-18 passed, 3 November 2026 at 6:57 AM PT, commit `30390061a`; ViT-B/16 did not pass, 7:01 AM PT, commit `d827db741`. Both used TorchLens 2.36.3 on Linux x86_64 CPU. |
| Cost | ResNet: 32.2 s analysis, 45.98 s process wall, 1,129,956 KiB peak RSS. ViT: 22.2 s analysis, 35.32 s process wall, 1,685,592 KiB peak RSS. Both used two CPU threads. |

The [Fashion-MNIST source](https://github.com/zalandoresearch/fashion-mnist) supplies images
under its MIT license. The hash is independently listed by
[TensorFlow Datasets](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/fashion_mnist.txt).
The script verifies the hash before decompression and preserves identifiers
`fashion-mnist:test:00000` onward. It does not redistribute the dataset.

## Oracles

| Fixed rule | ResNet-18, 24 images | ViT-B/16, 8 images |
| --- | --- | --- |
| Every RDM Pearson >= 0.999 | Minimum `0.9999999999999988`, pass | Minimum `0.9999999999981969`, pass |
| Every max absolute RDM error <= 1e-6 | Maximum `4.973799150320701e-13`, pass | Maximum `1.065007076750391e-06`, fail |
| Same best layer on the synthetic final-stage target | `avgpool` for both, pass | `encoder.ln` for both, pass |

ResNet includes five stages and all eight pre-ReLU residual additions. ViT includes three
stages, the positional sum and all 24 encoder residual additions. Its encoder-layer output
also names one residual sum, so 28 requested measurements resolve to 27 unique sites.
The independent path uses plain forward hooks and functional recomputation of each sum.
The failed ViT error limit remains unchanged; no full ViT rerun followed that failure.

## Controls

Shuffle one side with NumPy seed 123: every correlation must fall below 0.5. ResNet's range
was `[-0.06790722566, -0.02719278523]`; ViT's was `[-0.1558196402, 0.3135794525]`: both pass. Comparing stored stimulus identifiers
must flag the mismatch: both passed, and both JSON stamps record
`stimulus_order_mismatch=true`.

## Artifact and replay

Both runs save a portable `neuro_rdms.tlspec`, retaining activations, addresses, stimulus IDs,
preprocessing, shuffled IDs and RDMs. Use separate output directories for the two runs.
After `tl.load`, the script recomputes the first RDM; the fixed replay error limit is 1e-6.
Both replay errors were exactly zero and both stored order mismatches remained flagged.

| Run | Bytes | Sorted-path/content SHA256 |
| --- | --- | --- |
| ResNet-18 | 109,342,780 | `1fc37e4de867ffc4f05fed509e1c671d10b629e3801a4554a23b3de11f1f6f36` |
| ViT-B/16 | 132,667,341 | `acca03a29e5158a4cad089c8cdcf65a412e8df6951d96d6db5f8c24c3730d2b1` |

## Basis-grid row

Verbs: capture, select, read, compare, save. Both architectures were executed, with the
separate verdicts above. Nonblank, noncomment, non-docstring source counts were TorchLens
245 / plain 56 for the ResNet run, and TorchLens 247 / plain 56 after the pass-aware alias
lookup correction used by ViT. TorchLens counts include setup, CLI, plotting, reporting and
helpers; plain counts cover independent extraction, residual recomputation and rsatoolbox.
Original thingsvision reference lines were not measured.

## The stamp

ResNet-18 headline:

```text
{
  "commit": "30390061a0c9d81dd9cf347666c50f8c21bd6bfd",
  "date_utc": "2026-11-03T14:57:35Z",
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
  "wall_seconds": 32.2
}
```

ViT-B/16 quick check:

```text
{
  "commit": "d827db741045922c46dbcc3393c651e48539d308",
  "date_utc": "2026-11-03T15:01:19Z",
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
  "wall_seconds": 22.2
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
