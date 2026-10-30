# Scoring TorchLens attributions with Quantus

Quantus evaluates explanation methods: faithfulness, robustness, complexity,
and more. TorchLens computes input attributions (`torchlens.attribution`:
`saliency`, `input_x_grad`, `integrated_gradients`, and others) but does not
score them, and it ships no Quantus extra. The join is Quantus's
`explain_func` argument: wrap a TorchLens attribution function so it takes
and returns numpy arrays the way Quantus calls it.

```text
pip install torchlens quantus==0.6.0
```

## Wrap an attribution function as `explain_func`

Quantus calls `explain_func(model=..., inputs=..., targets=..., **kwargs)`
with a numpy batch and one integer target per row, and expects a numpy array
with the batch size first and the same number of dimensions as the inputs.
TorchLens attribution functions take one `target`: an integer class index, or
a callable from the model output to a scalar. A callable that sums each row's own target
logit gives every row its own gradient, because the rows of an eval-mode
batch do not interact.

```python
# recipe-test: tool=quantus expected=(4, 1, 8, 8)
import numpy as np
import torch
from torch import nn

import quantus
from torchlens.attribution import input_x_grad, saliency

torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1), nn.ReLU(),
    nn.Flatten(), nn.Linear(4 * 8 * 8, 3),
).eval()
x_batch = np.random.default_rng(0).standard_normal((4, 1, 8, 8)).astype(np.float32)
with torch.no_grad():
    y_batch = model(torch.from_numpy(x_batch)).argmax(dim=1).numpy()


def torchlens_explain(model, inputs, targets, method=saliency, **kwargs):
    """Quantus explain_func backed by a TorchLens attribution function."""
    x = torch.as_tensor(inputs, dtype=torch.float32)
    rows = torch.arange(x.shape[0])
    labels = torch.as_tensor(targets, dtype=torch.long)
    result = method(model, x, target=lambda out: out[rows, labels].sum())
    return result.values.detach().cpu().numpy()


RESULT = torchlens_explain(model, x_batch, y_batch).shape
```

## Run a Quantus metric on it

`MaxSensitivity` perturbs each input with small uniform noise, recomputes the
explanation through `explain_func`, and reports the largest relative change
(lower is more robust). It draws its noise from numpy's global generator, so
seed that for a repeatable score. Extra `explain_func_kwargs` reach the
wrapper, which is how one wrapper serves several TorchLens methods.

```python
# recipe-test: tool=quantus expected={'input_x_grad': ((4,), True), 'saliency': ((4,), True)}
metric = quantus.MaxSensitivity(
    nr_samples=5, lower_bound=0.1, disable_warnings=True, display_progressbar=False
)
scores = {}
for name, method in [("saliency", saliency), ("input_x_grad", input_x_grad)]:
    np.random.seed(0)
    values = np.asarray(
        metric(
            model=model, x_batch=x_batch, y_batch=y_batch, a_batch=None,
            channel_first=True, explain_func=torchlens_explain,
            explain_func_kwargs={"method": method}, device="cpu",
        )
    )
    scores[name] = (values.shape, bool(np.isfinite(values).all() and (values >= 0).all()))
RESULT = scores
```

## Limits

- The scores are Quantus's; TorchLens only produces the attributions. Read
  each metric's assumptions (baselines, noise scale, normalization) in the
  Quantus documentation before comparing methods.
- The per-row callable target relies on rows not interacting. A model with
  batch-dependent layers in the mode you run it (batch norm in training mode,
  for example) needs one attribution call per row instead.
- Quantus metrics call `explain_func` many times; slow attribution methods
  such as `integrated_gradients` with many steps make metrics slow.
- Quantus's own torch and tensorflow extras are not needed for this route.
