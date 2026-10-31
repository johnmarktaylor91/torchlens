# TorchLens to himalaya: banded ridge over a multi-site feature bank

Encoding models predict brain responses from model features. When the
features come from several sites (layers, branches, modalities), banded ridge
fits one regularization strength per feature space, so each site's share of
the prediction is estimated rather than fixed. himalaya (Dupre la Tour et al.)
implements banded ridge and multiple-kernel ridge. TorchLens builds the
feature bank: one block of columns per site, rows in stimulus order. It does
not fit encoding models.

```text
pip install torchlens himalaya==0.4.11
```

## Build a feature bank from three sites

The model below has three parallel branches, so its three sites carry
largely separate information, as distinct layers or modalities often do.
`tl.extract_dataset` returns one `(num_stimuli, num_features)` tensor per site;
stacking them side by side gives the feature bank, and the column ranges name
the bands.

```python
# recipe-test: tool=himalaya expected=(80, 24)
import numpy as np
import torch
from torch import nn

import torchlens as tl
from himalaya.backend import set_backend
from himalaya.ridge import BandedRidgeCV, ColumnTransformerNoStack
from himalaya.scoring import r2_score_split
from sklearn.preprocessing import StandardScaler


class ThreeBranch(nn.Module):
    """Three parallel branches read the same input."""

    def __init__(self) -> None:
        super().__init__()
        self.branch_a = nn.Linear(64, 8)
        self.branch_b = nn.Linear(64, 8)
        self.branch_c = nn.Linear(64, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch_a, self.branch_b, self.branch_c)
        parts = [torch.tanh(branch(x)) for branch in branches]
        return torch.cat(parts, dim=-1).sum(dim=-1)


torch.manual_seed(0)
model = ThreeBranch().eval()
stimuli = torch.randn(80, 64, generator=torch.Generator().manual_seed(1))
sites = ["branch_a", "branch_b", "branch_c"]

features = tl.extract_dataset(
    model, stimuli, {s: s for s in sites}, batch_size=40, progress=False
)
blocks = [features[s].reshape(len(stimuli), -1).double().numpy() for s in sites]
bank = np.concatenate(blocks, axis=1)
bands = {s: slice(8 * i, 8 * (i + 1)) for i, s in enumerate(sites)}
RESULT = bank.shape
```

## Synthetic responses from one band

Stand-in "brain" responses: ten targets that are a random linear readout of
the `branch_b` band plus noise. A real study loads recorded responses here,
one row per stimulus in the same order.

```python
# recipe-test: tool=himalaya expected=(80, 10)
rng = np.random.default_rng(0)
band_b = bank[:, bands["branch_b"]]
band_b = (band_b - band_b.mean(0)) / band_b.std(0)
responses = band_b @ rng.standard_normal((8, 10)) + rng.standard_normal((80, 10))
RESULT = responses.shape
```

## Fit banded ridge and read each band's share

`ColumnTransformerNoStack` scales each band and hands the bands to
`BandedRidgeCV` as separate feature spaces (`groups="input"`). The fit learns
one log-weight per band and target (`deltas_`); `predict(..., split=True)`
returns each band's part of the prediction, and `r2_score_split` divides the
held-out R2 among the bands. Both favor the band that generated the targets.

```python
# recipe-test: tool=himalaya expected=('branch_b', 'branch_b', True)
set_backend("numpy")
train, test = slice(0, 60), slice(60, 80)

column_transformer = ColumnTransformerNoStack(
    [(name, StandardScaler(), columns) for name, columns in bands.items()]
)
model_ridge = BandedRidgeCV(
    groups="input",
    solver_params=dict(n_iter=20, alphas=np.logspace(-2, 4, 7), progress_bar=False),
    random_state=0,
)
model_ridge.fit(column_transformer.fit_transform(bank[train]), responses[train])

parts = model_ridge.predict(column_transformer.transform(bank[test]), split=True)
held_out = responses[test] - responses[test].mean(0)
band_scores = np.asarray(r2_score_split(held_out, parts)).mean(axis=1)
band_weights = np.asarray(model_ridge.deltas_).mean(axis=1)

RESULT = (
    sites[int(np.argmax(band_weights))],
    sites[int(np.argmax(band_scores))],
    bool(band_scores[1] > 0.5 and max(band_scores[0], band_scores[2]) < 0.2),
)
```

## Limits

- The estimator, its solvers, and its cross-validation are himalaya's; read
  its documentation for solver choice and hyperparameter search.
- Sites from one sequential network are strongly correlated with each other,
  which makes band shares harder to separate than in this example.
- For many targets (voxels, electrodes), use himalaya's GPU backends
  (`set_backend("torch_cuda")`) and its batching options; the numpy backend
  here is for small examples.
- For large stimulus sets, write the extraction to disk (`path=`,
  `stimulus_ids=`) and build the bank from the stored artifact, so the rows
  stay tied to stimulus ids.
