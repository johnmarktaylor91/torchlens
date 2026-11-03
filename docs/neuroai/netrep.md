# TorchLens to netrep: shape metrics and the CCA family

netrep (Williams et al., "Generalized shape metrics on neural
representations") computes distances between representations that are true
metrics: Procrustes distance, CCA distance, and the family between them, plus
an angular form of linear CKA. TorchLens supplies the representations: it
extracts activations at named sites on a fixed stimulus set, in stimulus
order, and this page flattens them to the `(num_stimuli, num_features)` matrices
netrep expects. TorchLens does not implement these metrics.

```text
pip install torchlens "netrep @ git+https://github.com/ahwillia/netrep@0186b8a77ec1ebaf541cc8f7173cb2556df8a8f0"
```

netrep is not published on PyPI; the line above pins a commit of its
repository. It needs `numpy`, `scipy`, `scikit-learn`, and `tqdm`.

## Extract two sites on the same stimuli

`tl.extract_dataset` runs the model over the stimuli in batches and returns
one tensor per requested site, row `i` belonging to stimulus `i`. Converting
to float64 keeps netrep's whitening and angle computations precise.

```python
# recipe-test: tool=netrep expected=((100, 16), (100, 16))
from collections import OrderedDict

import torch
from torch import nn

import torchlens as tl
from netrep.metrics import LinearCKA, LinearMetric
from netrep.utils import rand_orth

torch.manual_seed(0)
model = nn.Sequential(OrderedDict(
    fc1=nn.Linear(40, 16), act1=nn.ReLU(),
    fc2=nn.Linear(16, 16), act2=nn.ReLU(),
    head=nn.Linear(16, 4),
)).eval()
stimuli = torch.randn(100, 40, generator=torch.Generator().manual_seed(1))

features = tl.extract_dataset(
    model, stimuli, {"early": "fc1", "late": "fc2"}, batch_size=25, progress=False
)
X = features["early"].reshape(len(stimuli), -1).double().numpy()
Y = features["late"].reshape(len(stimuli), -1).double().numpy()
RESULT = (X.shape, Y.shape)
```

## Procrustes and CCA distances

`LinearMetric(alpha=...)` spans the family: `alpha=1` allows only rotations
(Procrustes), `alpha=0` allows any invertible linear map (CCA), and values in
between partially whiten. Both sites here are 16 wide; for sites of
different widths, `zero_pad=True` (the default) pads the narrower one with
zeros, and netrep then needs `alpha > 0`. A representation and a rotated
copy of itself are at distance zero under every member; two different sites
are not.

```python
# recipe-test: tool=netrep expected={'cca': (True, True), 'procrustes': (True, True)}
X_rotated = X @ rand_orth(X.shape[1], random_state=0)

distances = {}
for name, alpha in [("procrustes", 1.0), ("cca", 0.0)]:
    same = LinearMetric(alpha=alpha).fit_score(X, X_rotated)
    different = LinearMetric(alpha=alpha).fit_score(X, Y)
    distances[name] = (bool(same < 1e-5), bool(different > 1e-2))
RESULT = distances
```

## Angular CKA

netrep's `LinearCKA` returns the angle between the two stimulus-by-stimulus
similarity matrices, which makes linear CKA a metric. It needs no fit.

```python
# recipe-test: tool=netrep expected=(True, True)
cka = LinearCKA()
RESULT = (bool(cka.score(X, X_rotated) < 1e-5), bool(cka.score(X, Y) > 1e-2))
```

## Limits

- The metrics and their statistical behavior are netrep's; read its
  documentation for sample-size requirements. CCA (`alpha=0`) needs more
  stimuli than features to be meaningful; with fewer, every pair of
  representations looks identical.
- Flattening discards spatial or token structure. Pool first (for example
  `pool="spatial_mean"` on `tl.extract_dataset`) when the layout should not
  count as features.
- For a stored artifact with stimulus ids, write the extraction to disk with
  `path=` and `stimulus_ids=`, and join on the ids before comparing
  models run on separately ordered stimuli.
