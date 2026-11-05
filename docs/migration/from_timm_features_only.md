# Coming from timm `features_only`

This page is for people who build timm backbones with `features_only=True` to get
multi-scale feature maps and want the same maps, or others from the same model, from
TorchLens. The examples were written against the timm range TorchLens's test environment
installs (`timm~=1.0`, in the `hf` and `test` extras). Checked 2026-11-03.

## The same task in both

Read the stage outputs of a ResNet-18 backbone at `out_indices=(1, 2)`. Without timm installed,
the test that runs this page skips both examples.

### In timm

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

### In TorchLens

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

## Translation table

| timm | TorchLens | Notes |
| --- | --- | --- |
| `timm.create_model(name, features_only=True)` then `model(x)` | `tl.extract(model, x, model.feature_info.module_name())` | Every timm stage name resolves as a TorchLens lookup. |
| `out_indices=(1, 2)` | Pass only the stage names you want | `model.feature_info.module_name()` lists them in order. |
| `feature_info.channels()` / `.reduction()` | Read `.shape` on the returned tensors, or `trace.modules[...]` for module facts | |
| Stages are fixed by the model author | Any module address or op label, inside a stage or between stages | Use `tl.trace(model, x)` and `trace.layers` to find others. |
| A plain classification model has no `features_only` view | `tl.extract` works on the plain model too | The model is never rebuilt. |

## What TorchLens adds

- Lookups are not limited to the stages the model author exposed: any module address or op
  label works (`tl.trace(model, x).modules` and `.layers`).
- The plain `timm.create_model(name)` classifier works as it is; no second, rebuilt model is
  needed to read an intermediate stage.
- `tl.extract_dataset` writes the same lookups over a dataloader to disk; see
  [Multi-GPU extraction](multi_gpu_extraction.md).
- A trace records every executed op between stages, so a feature map and the arithmetic that
  produced it come from one call.

## What timm `features_only` does that TorchLens does not

- It returns a model whose `forward` is the list of feature maps, ready to plug into a
  detector or segmentation head as a backbone you can train. `tl.extract` returns tensors
  for one call.
- It carries `feature_info`, the author-curated stage names, channel counts and reduction
  factors. TorchLens reports shapes of what ran but not the author's choice of which stages
  count as the pyramid.
- It handles architectures timm rebuilds for feature output (for example by dropping the
  classifier). TorchLens leaves the model whole.

See [where to go instead](where_to_go_instead.md) for tools outside TorchLens's scope.

## Using both

### timm feeds TorchLens

A timm model, with or without `features_only`, is an `nn.Module`: pass it to `tl.extract` or
`tl.trace` (the example above does). Use `model.feature_info.module_name()` to get the stage
lookups. There is no timm-specific bridge.

### TorchLens feeds timm

There is no bridge. Module addresses read from `trace.modules` are the same strings timm uses
for stages when they are modules, so you can find a name with TorchLens and check it against
`feature_info`.

## Semantic differences

- **Output shape.** timm returns a list indexed by `out_indices`; `tl.extract` returns a
  dict keyed by the lookups you passed.
- **Detached by default.** `tl.extract` returns detached copies; pass
  `detach_saved_activations=False` to keep gradients, as a timm backbone's outputs have.
- **Same tensors.** On a `features_only` model the TorchLens lookups equal the model's own
  outputs, as the second example checks.
