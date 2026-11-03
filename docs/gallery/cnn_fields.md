# CNN fields as an intervention site

This entry measured ResNet-18 geometric influence and input interventions and did not pass:
the coordinates matched the analytic box, but the field status was `upper_bound` rather than exact.
The geometric background is [Araujo, Norris and Sim (Distill, 2019)](https://distill.pub/2019/computing-receptive-fields/);
gradient support is an additional empirical oracle, not a general equality theorem.

| Item | Recorded setup |
| --- | --- |
| Claim | Geometric CNN influence supports input interventions and predicts projective support. |
| Script and run line | `python examples/gallery/cnn_fields.py --out outputs/cnn_fields` |
| Pinned inputs | torchvision ResNet-18, `IMAGENET1K_V1`, eval, float32, seed 17, one 224x224 random input; layer3, channel 0, centre unit. |
| Verdict | Did not pass; Linux x86_64 CPU, 3 November 2026 at 6:54 AM PT; TorchLens 2.36.3, commit `9a5e9678e`. |
| Cost | 18.6 s measured analysis; 26.31 s process wall time, two CPU threads, 722,336 KiB peak RSS. |

## Oracles

| Oracle | Fixed rule | Measured | Outcome |
| --- | --- | --- | --- |
| Analytic box | Exact coordinates and exact status | Expected and measured `[7,218)` on both axes; status `upper_bound` | Fail |
| Gradient support | IoU >= 0.95 | 1.0 | Pass |
| Outside ablation | Nonempty mask; float32 unit delta exactly 0 | 5,655 spatial pixels; delta 0 | Pass |
| Inside ablation | Unit delta > 0 | 0.1923354864 | Pass |
| Projective support | IoU >= 0.95 against changed spatial units | 1.0 | Pass |

The exact-status rule stays unchanged. Matching coordinates alone do not establish that rule.
The script saves each check immediately so later replay errors retain the measurements.

## Controls

A deliberately naive registered-leaf chain treats projection shortcuts as sequential layers;
its clipped box `[207,224)` differs from the branched analytic box `[7,218)`: pass. This control
does not assert that every main-path-only formula fails on ResNet. A 4x4 corner patch lies
outside the receptive field and has unit delta 0: pass.

## Artifact and replay

The portable `cnn_fields.tlspec` directory contains 213 files and 34,539,747 bytes;
sorted-path/content SHA256 `0dddadabe4430728ce019319974becec787c0f738ea475d06d863620e006ed15`. The script reloads it, applies the input mask
with `InterventionOptions(engine="set_only")`, then calls `run(model, edited_input)`.
The rerun refused with `EngineDispatchError`: the edited loaded values no longer have a
persisted intervention recipe. No replay match is claimed. The failed exact-status oracle
already moves this entry to the next batch; neither rule nor tolerance was changed.

## Basis-grid row

Verbs: capture, select, read, do, compare, save. Executed architecture: ResNet-18.
Nonblank, noncomment, non-docstring source lines: TorchLens 234; plain PyTorch 37;
original reference not measured. The TorchLens region includes setup, CLI, reporting,
plotting and helpers. The plain region covers independent geometry and hook/gradient
extraction. These scoped counts do not compare against original-reference code.

## The stamp

```text
{
  "commit": "9a5e9678e7d642f1ba3bc219c662137b01089b62",
  "date_utc": "2026-11-03T14:54:17Z",
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
  "wall_seconds": 18.6
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
