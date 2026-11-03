# Neuro extraction with an order-error control

This entry compares layer and residual-add activations against independent PyTorch extraction
and rsatoolbox 0.3.2 correlation RDMs. The 24-stimulus ResNet-18 run passed all 13 sites. It checks extraction
correctness; the target used to compare layer rankings is synthetic.

| Item | Recorded setup |
| --- | --- |
| Claim | Layer-wise RDMs agree with independent extraction and rsatoolbox 0.3.2. |
| Script and run line | `python examples/gallery/neuro_rdms.py --out outputs/neuro_rdms` |
| Pinned inputs | ResNet-18 `IMAGENET1K_V1`, eval, float32; first 24 Fashion-MNIST test images in split order; RGB, antialiased bilinear 224x224, ImageNet normalization; seed 17. |
| Data hash | SHA256 `346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073` of the compressed image file. |
| Verdict | Passed on ResNet-18; Linux x86_64 CPU, 3 November 2026 at 6:57 AM PT; TorchLens 2.36.3, commit `30390061a`. |
| Cost | 32.2 s analysis; 45.98 s process wall, two CPU threads; 1,129,956 KiB peak RSS. |

The [Fashion-MNIST source](https://github.com/zalandoresearch/fashion-mnist) supplies the test
images under its MIT license. The input hash is independently listed by
[TensorFlow Datasets](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/fashion_mnist.txt).
The script verifies the hash before decompressing, then uses identifiers `fashion-mnist:test:00000`
through `fashion-mnist:test:00023`. It does not redistribute the dataset.

## Oracles

Every stage and residual-add RDM must correlate at Pearson >= 0.999 with the independent
extraction. Correlation distances from TorchLens must agree with rsatoolbox's own pipeline
to maximum absolute error <= 1e-6. Both pipelines must select the same best layer against
the final selected stage's independent RDM. This target is a software check, not a neural
recording. Across five stages and eight residual adds, minimum Pearson was `0.9999999999999988`, maximum absolute RDM error was `4.973799150320701e-13`, and both paths selected `avgpool`: all pass.

The independent path uses forward hooks on module outputs and explicitly recomputes each
pre-ReLU residual sum from the main branch and identity/projection branch. ViT's independent
path recomputes the positional sum and both residual sums per encoder block.

## Controls

Shuffle one side with NumPy's generator at seed 123. Every per-layer RDM correlation must
fall below 0.5, and comparing stored stimulus identifiers must flag the order mismatch. The
JSON stamp records `stimulus_order_mismatch=true`. Shuffled correlations ranged from `-0.0679072257` to `-0.0271927852`, all below 0.5: pass.

## Artifact and replay

The portable `neuro_rdms.tlspec` stores selected activations, layer addresses, stimulus identifiers,
preprocessing, order-error identifiers and the RDMs. After `tl.load`, the script recomputes one
RDM through `tl.neuro.rdms` and checks maximum absolute difference <= 1e-6. The activations can
also be handed to `tl.neuro.datasets` for rsatoolbox. The saved directory has 109,342,780 bytes and sorted-path/content SHA256 `1fc37e4de867ffc4f05fed509e1c671d10b629e3801a4554a23b3de11f1f6f36`. Reloaded RDM max error was 0.0: pass; the stored order mismatch remained flagged.

## Basis-grid row

Verbs: capture, select, read, compare, save. Measured source lines: TorchLens 245; plain PyTorch 56;
original reference not measured. The TorchLens count includes CLI setup, loading, reporting,
plotting and helpers. The plain count covers the independent hook extraction, functional
residual recomputation and rsatoolbox calculation functions. Original thingsvision source lines are not measured.
The recorded passing architecture here is ResNet-18; the second-architecture run is reported separately when complete.

## The stamp

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
