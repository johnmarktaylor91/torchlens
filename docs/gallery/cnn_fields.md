# CNN fields as an intervention site

A CNN unit's receptive field is the input region that can influence it. This entry computes the
field of one ResNet-18 unit with TorchLens, checks it against an independent analytic
computation and against autograd gradient support, then uses it as an intervention site:
zeroing everything outside the field must leave the unit exactly unchanged, and zeroing inside
must change it. The entry passes: the box matches the analytic box exactly, both support IoUs
are 1.0, and the outside edit changes the unit by exactly 0. The geometric background is
[Araujo, Norris and Sim (Distill, 2019)](https://distill.pub/2019/computing-receptive-fields/);
gradient support is an additional empirical oracle, not a general equality theorem.

| Item | Recorded setup |
| --- | --- |
| Claim | Geometric CNN influence supports input interventions and predicts projective support. |
| Script and run line | `python examples/gallery/cnn_fields.py --out outputs/cnn_fields` |
| Pinned inputs | torchvision ResNet-18, `IMAGENET1K_V1`, eval, float32, seed 17, one 224x224 random input; layer3, channel 0, centre unit. |
| Verdict | Passed; Linux x86_64 CPU, 3 November 2026, TorchLens 2.36.3, commit `01497c6d`. |
| Cost | 13.9 s measured analysis; 21.1 s process wall time on two CPU threads, 743 MiB peak memory. |

## Oracles

The rules are the ones registered for this entry before any run.

| Oracle | Rule | Measured | Outcome |
| --- | --- | --- | --- |
| Analytic box | Exact match of the clipped box with an independent analytic computation | Expected and measured `[7,218)` on both axes | Pass |
| Gradient support | IoU >= 0.95 between the field and the input positions with nonzero gradient | 1.0 | Pass |
| Outside ablation | Nonempty outside mask; float32 unit delta exactly 0 | 5,655 spatial pixels zeroed; delta 0 | Pass |
| Inside ablation | Unit delta > 0 | 0.1923354864 | Pass |
| Projective support | IoU >= 0.95 against the units a plain run shows changed | 1.0 | Pass |

The first run of this script also required the field's relation label to be `exact`. TorchLens
labels this field `upper_bound`, which is its documented label for a strided hull: the hull can
contain positions that influence nothing, so it is reported as a superset (see
[selection completeness](../reference/selection_completeness.md), Lemma R). The coordinates are
exact; the label is reported here, not graded.

## Controls

| Control | Rule | Measured | Outcome |
| --- | --- | --- | --- |
| Same-size outside patch | A 4x4 corner patch lies outside the field and changes the unit by exactly 0 | outside; delta 0 | Pass |
| Serial module chain | Treating the registered leaves as one chain (ignoring the branches) must give a different box | `[207,224)` against the branched `[7,218)` | Pass (failed as declared) |

## Artifact and replay

The full capture is saved as a portable `cnn_fields.tlspec` (34,539,747 bytes, sha256
`5854ea7ee5a0a391`...). The script reloads it, zeroes the outside field of the reloaded input,
reruns the model and compares the unit with the reloaded value: the delta is exactly 0, as in
memory.

## Basis-grid row

| Verbs | TorchLens lines | Plain-PyTorch lines | Architecture |
| --- | --- | --- | --- |
| capture, select, read, do, compare, save | 233 | 37 | ResNet-18 |

Counts are nonblank, noncomment, non-docstring lines. The TorchLens region includes setup, the
command line, reporting, plotting and helpers; the plain region covers the independent geometry
and the hook and gradient extraction. No count against an original reference implementation was
measured.

## The stamp

```text
{
  "commit": "01497c6dcea4b23f5fe5ee3401fc3678fa757123",
  "date_utc": "2026-11-03T15:27:18Z",
  "device": "cpu",
  "libraries": {
    "numpy": "2.2.6",
    "torchvision": "0.28.0+cpu",
    "transformers": "5.18.0"
  },
  "machine": "x86_64",
  "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31",
  "python": "3.10.19",
  "torch": "2.13.0+cpu",
  "torch_threads": 2,
  "torchlens": "2.36.3",
  "wall_seconds": 13.9
}
```

## Vary it

Use `--layer layer2`, `--channel 1` or the explicit `--arch resnet18` selector. `--quick` runs the
same headline input without extra architecture runs. VGG-16 and ConvNeXt are not claimed here.
These small examples use a randomly initialized torchvision residual block without downloads.

```python
# recipe-test: tool=torchvision expected=True
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock
import torchlens as tl

torch.manual_seed(17)
model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), BasicBlock(4, 4)).eval()
x = torch.randn(1, 3, 16, 16)
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
box = trace["1"].receptive_field.at((8, 8))
RESULT = box.exact and all(a.clipped_stop - a.clipped_start == 7 for a in box.axes[-2:])
```

```python
# recipe-test: tool=torchvision expected=True
mask = torch.zeros_like(x, dtype=torch.bool)
mask[:, :, 0:2, 0:2] = True
fork = trace.fork()
fork.do(tl.units(trace.input_ops[0].label, mask), tl.zero_ablate())
RESULT = bool(torch.equal(fork["1"].out[:, :, 8, 8], trace["1"].out[:, :, 8, 8]))
```
