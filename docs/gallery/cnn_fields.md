# CNN fields as an intervention site

This entry measures geometric influence on a pretrained ResNet-18 against an independent
analytic calculation, gradient support and input interventions. The recorded verdict is pending.
The geometric background is [Araujo, Norris and Sim (Distill, 2019)](https://distill.pub/2019/computing-receptive-fields/);
gradient support is an additional empirical oracle, not a general equality theorem.

| Item | Recorded setup |
| --- | --- |
| Claim | Geometric CNN influence supports input interventions and predicts projective support. |
| Script and run line | `python examples/gallery/cnn_fields.py --out outputs/cnn_fields` |
| Pinned inputs | torchvision ResNet-18, `IMAGENET1K_V1`, eval, float32, seed 17, one 224x224 random input; layer3, channel 0, centre unit. |
| Verdict | Pending the recorded worker run. |
| Cost | Pending measurement; two CPU threads. |

## Oracles

The analytic clipped box must agree exactly and have exact status. The spatial gradient-support
IoU must be at least 0.95. Ablating the nonempty outside mask must change the unit by exactly
zero in float32; inside ablation must change it. Projective-field support must have IoU at
least 0.95 with the spatial units changed by a 4x4 input-patch ablation. Measurements are pending.

## Controls

A deliberately naive registered-leaf chain treats projection shortcuts as sequential layers;
its clipped box must differ from the independently branched analytic calculation. This control
does not assert that every main-path-only formula fails on ResNet. A 4x4 patch in the image
corner, the same size as the projective-field probe, must lie outside the receptive field and
leave the target unchanged. Measurements are pending.

## Artifact and replay

The script saves `cnn_fields.tlspec` as a portable trace, reloads it with `tl.load`, then
repeats the outside-field zero ablation on a fork. The replay requires exactly zero unit change.
Artifact size, hash and replay result are pending.

## Basis-grid row

Verbs: capture, select, read, do, compare, save. The script measures nonblank, noncomment,
non-docstring lines in the marked regions. The TorchLens region includes CLI setup,
reporting, plotting and helpers; the plain region counts the independent geometry and hook
oracle functions. These are stated scopes, not an original-reference code comparison. The original TensorFlow reference implementation's lines are not measured.
Only architectures actually executed are reported.

## The stamp

```text
Pending measurement. Run the script to produce the software, commit and hardware stamp.
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
trace = tl.trace(model, x)
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
