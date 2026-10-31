# Weight spectra with WeightWatcher

WeightWatcher fits power laws to the eigenvalue spectra of a model's weight
matrices and reports per-layer quality metrics such as `alpha`. TorchLens has
no weight-spectrum analysis of its own; what it adds is the parameter record
of a traced run: `trace.params` lists every parameter the model declared, with
its address, owning module, shape, and whether a captured op used it. This
page hands those parameters to WeightWatcher and joins its table back to the
TorchLens records.

```text
pip install torchlens weightwatcher==0.7.7
```

## Hand the traced parameters to WeightWatcher

WeightWatcher accepts a plain dict of tensors keyed like a PyTorch
`state_dict` (`"<module>.weight"`, `"<module>.bias"`), so no module needs to
be rebuilt: build the dict from `trace.params`. `Param.address` is the full
parameter address and `Param.value` is the live parameter of the traced model.
WeightWatcher analyzes each `.weight` matrix and names the row by the module
address.

```python
# recipe-test: tool=weightwatcher expected=True
import torch
from torch import nn

import torchlens as tl
import weightwatcher as ww

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(32, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 16),
).eval()
trace = tl.trace(model, torch.randn(4, 32))

weights = {param.address: param.value.detach().clone() for param in trace.params}
details = ww.WeightWatcher(model=weights).analyze()

weight_modules = {param.module_address for param in trace.params if param.name == "weight"}
RESULT = set(details["name"]) == weight_modules
```

## Join the table back to the parameter records

Each WeightWatcher row carries the matrix dimensions (`N` the larger, `M` the
smaller), so every row can be checked against the shape TorchLens recorded for
the same parameter. Keep only parameters a captured op actually used
(`param.used_by_ops`) if the model declares weights its forward never touches.

```python
# recipe-test: tool=weightwatcher expected=True
rows = details.set_index("name")
RESULT = all(
    sorted(trace.params[f"{name}.weight"].shape) == sorted([int(row["M"]), int(row["N"])])
    for name, row in rows.iterrows()
)
```

`details` also holds WeightWatcher's own metrics (`alpha`, `log_norm`,
`stable_rank`, and so on) per row; read their meaning in the WeightWatcher
documentation.

## Limits

- The metrics are WeightWatcher's; TorchLens only supplies the tensors and the
  parameter identities.
- WeightWatcher skips matrices whose smaller dimension is below its
  `min_evals` default (8), 1-D parameters, and normalization layers, so its
  table can have fewer rows than `trace.params` has weights.
- The dict route treats every tensor as a dense or conv weight by its rank; it
  does not know about tied weights. `trace.graph.tied_param_groups` lists ties so
  you can drop duplicates first.
- Power-law fits on very small random matrices are not meaningful; the
  example checks structure only.
