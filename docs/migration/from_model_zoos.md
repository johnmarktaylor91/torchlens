# Coming from model zoos

This page is for people who get models and features through a zoo wrapper (for example
thingsvision's `get_extractor`, or a library's named-model loader) and want to know what
TorchLens takes over. TorchLens is not a model zoo. It does not keep a catalogue of named
vision or language models, and it does not ship gLocal-style aligned transforms. Load the model
with the library that owns it (torchvision, timm, open_clip, Hugging Face transformers, a
research repo's own loader), then hand the `nn.Module` to TorchLens. The examples build
torchvision's ResNet-18 without weights, so they need no download. Checked 2026-11-03.

## The same task in both

Get the `layer4` activations of ResNet-18 for a batch of six stimuli.

### In a model zoo

The zoo's job is loading the model; the extraction step it adds is a forward hook, shown here
on torchvision directly (thingsvision's own `extract_features` is described on
[its page](from_thingsvision.md)):

```python
# migration-test: tool=torchlens expected=(6, 512, 1, 1)
import torch
from torchvision.models import resnet18

torch.manual_seed(0)
model = resnet18(weights=None).eval()  # load it with weights from your library in practice
stimuli = torch.randn(6, 3, 32, 32)
seen = []
handle = model.layer4.register_forward_hook(lambda m, i, out: seen.append(out.detach()))
with torch.no_grad():
    model(stimuli)
handle.remove()
RESULT = tuple(seen[0].shape)
```

### In TorchLens

Sparse random projection: `tl.transforms.srp(n_components, seed=0)` projects each stimulus's
activations to `n_components` dimensions while they are extracted. This is the speed trick
DeepJuice uses for brain benchmarking: it keeps large layers small enough to score, and the
projection's seed and construction are recorded with the output. See
`docs/reference/transforms.md` for when it preserves your RDMs.

```python
# migration-test: tool=torchlens expected=((6, 16), True)
import tempfile

import torch
from torchvision.models import resnet18
import torchlens as tl
from torchlens import transforms as T

torch.manual_seed(0)
model = resnet18(weights=None).eval()  # load it with weights from your library in practice
stimuli = torch.randn(6, 3, 32, 32)
output_dir = tempfile.mkdtemp()
tl.extract_dataset(
    model,
    stimuli,
    ["layer4"],
    batch_size=3,
    path=output_dir,
    output_transform={"layer4": T.chain(T.flatten(), T.srp(16, seed=0))},
    progress=False,
)
loaded = tl.load(output_dir)
projected = torch.as_tensor(loaded.activations["layer4"])
RESULT = (tuple(projected.shape), bool(torch.isfinite(projected).all()))
```

## Translation table

| Model-zoo workflow (for example thingsvision's `get_extractor`) | TorchLens | Notes |
| --- | --- | --- |
| `get_extractor(model_name="resnet18", source="torchvision", pretrained=True)` | `model = torchvision.models.resnet18(weights=...)` | Any loader that returns an `nn.Module` works. |
| `extractor.show_model()` | `tl.trace(model, x)` then `trace.modules` and `trace.layers` | Lists module addresses and executed ops. |
| `extractor.extract_features(batches, module_name="layer4")` | `tl.extract(model, x, ["layer4"])`, or `tl.extract_dataset(model, stimuli, ["layer4"], path=...)` for a dataset | |
| Built-in preprocessing for each zoo model | Use the library's own transforms (`weights.transforms()`, the timm data config, the HF processor) | TorchLens records the preprocessing it recognizes, such as torchvision weight presets and HF processors. |
| gLocal (aligned) transforms | Fit and apply them yourself after extraction | See `docs/neuroai/byo_alignment.md`. |
| Reduce dimensions while extracting | `output_transform=` on `tl.extract_dataset`, for example `tl.transforms.srp(...)` | See the example above. |

## What TorchLens adds

- Extraction from any `nn.Module` a loader returns, by module address or op label
  (`tl.extract`).
- `tl.extract_dataset` writes activations to disk batch by batch and `tl.load(path)` reads
  them back, with provenance recorded with the output.
- Preprocessing recorded where TorchLens recognizes it (torchvision weight presets and HF
  processors), so a saved extraction says how its inputs were prepared.
- Dimension reduction during extraction (`tl.transforms.srp`, `T.chain`), with the seed
  and construction recorded.

## What model zoos do that TorchLens does not

- A zoo keeps the catalogue: named models, their sources and pretrained weights, behind one
  loader. TorchLens does not; use the zoo's loader and pass the model on.
- Zoos ship per-model preprocessing and, in some cases, aligned transforms (gLocal). TorchLens
  leaves alignment to your own fit; see `docs/neuroai/byo_alignment.md`.
- Brain benchmarking and RDM scoring are destinations, not features of this page: TorchLens
  hands activations to them (see [thingsvision](from_thingsvision.md) and
  [where to go instead](where_to_go_instead.md)).

## Using both

### The zoo feeds TorchLens

Load the model with the zoo, then pass the `nn.Module` to `tl.extract`, `tl.extract_dataset`
or `tl.trace`, as the examples do. There is no bridge; the `nn.Module` is the interface.

### TorchLens feeds the zoo

There is no bridge. `tl.extract_dataset` writes activations and `tl.load` returns them as
arrays, which any scoring or benchmarking code can read; the zoo's own extractors are not
fed by TorchLens.

## Semantic differences

- **Loading and extracting are separate.** A zoo's `get_extractor` returns an object that
  loads, preprocesses and extracts. With TorchLens the loader, the preprocessing and the
  extraction are three calls you control.
- **Projection is recorded.** `srp` keeps layers small enough to score, and its seed and
  construction are stored with the output; a projection is not an identity, so check that it
  preserves your RDMs (`docs/reference/transforms.md`).
- **Preprocessing is yours.** TorchLens records the preprocessing it recognizes but does not
  apply a zoo's per-model transform for you.
