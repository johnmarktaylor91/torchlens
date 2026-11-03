# Coming from timm `features_only`

This page maps timm's `features_only=True` backbones to TorchLens: their spelling on the
left, the TorchLens spelling on the right.

| timm | TorchLens | Notes |
| --- | --- | --- |
| `timm.create_model(name, features_only=True)` then `model(x)` | `tl.extract(model, x, model.feature_info.module_name())` | Every timm stage name resolves as a TorchLens lookup. |
| `out_indices=(1, 2)` | Pass only the stage names you want | `model.feature_info.module_name()` lists them in order. |
| `feature_info.channels()` / `.reduction()` | Read `.shape` on the returned tensors, or `trace.modules[...]` for module facts | |
| Stages are fixed by the model author | Any module address or op label, inside a stage or between stages | Use `tl.trace(model, x)` and `trace.layers` to find others. |
| A plain classification model has no `features_only` view | `tl.extract` works on the plain model too | The model is never rebuilt. |

Executing this page needs timm (it is in the `hf` and `test` extras). Without timm
installed, the test that runs this page skips both examples.

Their construct:

```python
# migration-test: tool=timm expected=[[1, 64, 8, 8], [1, 128, 4, 4]]
import timm
import torch

torch.manual_seed(0)
model = timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=(1, 2))
model.eval()
with torch.no_grad():
    features = model(torch.randn(1, 3, 32, 32))
RESULT = [list(f.shape) for f in features]
```

TorchLens equivalent:

```python
# migration-test: tool=timm_torchlens expected=([[1, 64, 8, 8], [1, 128, 4, 4]], True)
import timm
import torch
import torchlens as tl

torch.manual_seed(0)
model = timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=(1, 2))
model.eval()
x = torch.randn(1, 3, 32, 32)
names = model.feature_info.module_name()
features = tl.extract(model, x, names)

with torch.no_grad():
    native = model(x)
RESULT = (
    [list(features[name].shape) for name in names],
    all(torch.equal(features[name], out) for name, out in zip(names, native)),
)
```

See also [Multi-GPU extraction](multi_gpu_extraction.md).
