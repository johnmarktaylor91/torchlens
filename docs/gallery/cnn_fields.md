# CNN fields as an intervention site

This entry measures geometric influence on a pretrained ResNet-18 against an independent
analytic calculation, gradient support and input interventions. The recorded run did not pass: its portable artifact could not replay the intervention.
The geometric background is [Araujo, Norris and Sim (Distill, 2019)](https://distill.pub/2019/computing-receptive-fields/);
gradient support is an additional empirical oracle, not a general equality theorem.

| Item | Recorded setup |
| --- | --- |
| Claim | Geometric CNN influence supports input interventions and predicts projective support. |
| Script and run line | `python examples/gallery/cnn_fields.py --out outputs/cnn_fields` |
| Pinned inputs | torchvision ResNet-18, `IMAGENET1K_V1`, eval, float32, seed 17, one 224x224 random input; layer3, channel 0, centre unit. |
| Verdict | Did not pass; Linux x86_64 CPU, 3 November 2026 at 6:48 AM PT; TorchLens 2.36.3, commit `51a5477ce`. |
| Cost | 13.3 s measured analysis; 21.26 s process wall time, two CPU threads, 747,500 KiB peak RSS. |

## Oracles

The analytic clipped box must agree exactly and have exact status. The spatial gradient-support
IoU must be at least 0.95. Ablating the nonempty outside mask must change the unit by exactly
zero in float32; inside ablation must change it. Projective-field support must have IoU at
least 0.95 with the spatial units changed by a 4x4 input-patch ablation. The saved diagnostic reports gradient-support IoU 1.0000 and projective-support IoU 1.0000. The final JSON contains the replay failure rather than the intermediate checks, so the other numeric outcomes are not claimed.

## Controls

A deliberately naive registered-leaf chain treats projection shortcuts as sequential layers;
its clipped box must differ from the independently branched analytic calculation. This control
does not assert that every main-path-only formula fails on ResNet. A 4x4 patch in the image
corner, the same size as the projective-field probe, must lie outside the receptive field and
leave the target unchanged. These controls were reached before artifact replay, but their numeric outcomes were not retained in the final failure record; no pass is claimed.

## Artifact and replay

The script saves `cnn_fields.tlspec` as a portable trace, reloads it with `tl.load`, then
repeats the outside-field zero ablation on a fork. The replay requires exactly zero unit change.
The saved directory is 34,539,749 bytes; its sorted-path/content SHA256 is `31d9a8b7facbcdaadb790a69080a67df9ba518924d2d45e364d21f982cee09cb`. Replay failed with `ReplayPreconditionError`: portable saves omit capture-time callables and argument templates. The requested loaded-trace edit is unsupported. This entry stops here; it does not substitute an in-process replay or claim a complete reproduction.

## Basis-grid row

The attempted path uses capture, select, read, do, compare and save, and ran ResNet-18 only.
The failure JSON has empty verb/architecture lists because replay aborted entry construction.
Measured source counts: TorchLens 195; plain PyTorch 37; original reference not measured.
The script measures nonblank, noncomment,
non-docstring lines in the marked regions. The TorchLens region includes CLI setup,
reporting, plotting and helpers; the plain region counts the independent geometry and hook
oracle functions. These are stated scopes, not an original-reference code comparison. The original TensorFlow reference implementation's lines are not measured.
Only architectures actually executed are reported.

## The stamp

```text
{
  "commit": "51a5477ce54b9736f15e7ca33e078165e75c16da",
  "date_utc": "2026-11-03T14:48:24Z",
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
  "wall_seconds": 13.3
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
