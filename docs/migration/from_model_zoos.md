# Model zoos and aligned models: load with your library, extract with TorchLens

TorchLens is not a model zoo. It does not keep a catalogue of named vision or language
models, and it does not ship gLocal-style aligned transforms. Load the model with the
library that owns it (torchvision, timm, open_clip, Hugging Face transformers, a
research repo's own loader), then hand the `nn.Module` to TorchLens.

| Model-zoo workflow (for example thingsvision's `get_extractor`) | TorchLens | Notes |
| --- | --- | --- |
| `get_extractor(model_name="resnet18", source="torchvision", pretrained=True)` | `model = torchvision.models.resnet18(weights=...)` | Any loader that returns an `nn.Module` works. |
| `extractor.show_model()` | `tl.trace(model, x)` then `trace.modules` and `trace.layers` | Lists module addresses and executed ops. |
| `extractor.extract_features(batches, module_name="layer4")` | `tl.extract(model, x, ["layer4"])`, or `tl.extract_dataset(model, stimuli, ["layer4"], output_dir=...)` for a dataset | |
| Built-in preprocessing for each zoo model | Use the library's own transforms (`weights.transforms()`, the timm data config, the HF processor) | TorchLens records the preprocessing it recognizes, such as torchvision weight presets and HF processors. |
| gLocal (aligned) transforms | Fit and apply them yourself after extraction | See `docs/neuroai/byo_alignment.md`. |
| Reduce dimensions while extracting | `transform=` on `tl.extract_dataset`, for example `tl.transforms.srp(...)` | See below. |

Sparse random projection: `tl.transforms.srp(n_components, seed=0)` projects each
stimulus's activations to `n_components` dimensions while they are extracted. This is
the speed trick DeepJuice uses for brain benchmarking: it keeps large layers small enough
to score, and the projection's seed and construction are recorded with the output. See
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
    transform={"layer4": T.chain(T.flatten(), T.srp(16, seed=0))},
    progress=False,
)
loaded = tl.load_extraction(output_dir)
projected = torch.as_tensor(loaded.activations["layer4"])
RESULT = (tuple(projected.shape), bool(torch.isfinite(projected).all()))
```
