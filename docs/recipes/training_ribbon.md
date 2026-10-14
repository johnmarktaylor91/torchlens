# A percentile-band training ribbon from watch histograms

A ribbon plot shows how one site's distribution moves over training: the
median as a line, with shaded bands between percentiles at every step. The
histograms that `torchlens.trackers.watch` already writes are enough to draw
it with matplotlib; TorchLens adds no plotting call for it.

## 1. Record histograms while training

`hist_every=1` writes one histogram per watched tensor per step. The JSONL
sink writes one row per emission; histogram rows carry exact `counts` on
explicit `edges` (`len(edges) == len(counts) + 1`, bin `i` covers
`[edges[i], edges[i+1])`).

```python
# recipe-test: tool=matplotlib expected=True
import json

import numpy as np
import torch
from torch import nn

import torchlens.trackers as trk

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
inputs, targets = torch.randn(64, 16), torch.randn(64, 1)

session = trk.watch(
    model,
    to=trk.JSONLSink("run.jsonl"),
    signals=("gradients",),
    optimizer=optimizer,
    every=1,
    hist_every=1,
)
with session:
    for step in range(20):
        with session.step(step):
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(inputs), targets).backward()
            optimizer.step()

rows = [json.loads(line) for line in open("run.jsonl")]
histograms = [row for row in rows if row.get("kind") == "histogram"]
tags = sorted({row["tag"] for row in histograms})
RESULT = len(histograms) == 20 * len(tags) and len(tags) > 0
```

## 2. Percentiles from counts, then the ribbon

Percentiles come from the cumulative counts, interpolated linearly inside the
bin that holds each one. They are estimates at the histogram's resolution;
say so on the figure.

```python
# recipe-test: tool=matplotlib expected=True
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def percentiles(counts, edges, qs):
    """Estimate quantiles ``qs`` (in [0, 1]) from one histogram."""

    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    finite = np.isfinite(edges)
    edges = np.interp(np.arange(len(edges)), np.flatnonzero(finite), edges[finite])
    cdf = np.concatenate([[0.0], np.cumsum(counts)]) / counts.sum()
    return np.interp(qs, cdf, edges)


tag = next(t for t in tags if "0.weight" in t)  # first Linear layer's weight gradient
series = sorted((row for row in histograms if row["tag"] == tag), key=lambda r: r["step"])
steps = np.array([row["step"] for row in series])
qs = (0.05, 0.25, 0.5, 0.75, 0.95)
bands = np.array([percentiles(row["counts"], row["edges"], qs) for row in series])

fig, ax = plt.subplots(figsize=(6, 3))
ax.fill_between(steps, bands[:, 0], bands[:, 4], alpha=0.2, label="p5-p95")
ax.fill_between(steps, bands[:, 1], bands[:, 3], alpha=0.4, label="p25-p75")
ax.plot(steps, bands[:, 2], label="median")
ax.set_xlabel("step")
ax.set_ylabel("gradient value")
ax.set_title(f"{tag} (percentiles estimated from histogram bins)", fontsize=8)
ax.legend(fontsize=7)
fig.savefig("ribbon.png", dpi=120, bbox_inches="tight")
plt.close(fig)
RESULT = bool(np.all(np.diff(bands, axis=1) >= 0)) and bands.shape == (20, 5)
```

## Limits

- The bands are only as fine as the histogram grid. The default grid is
  signed log2 bins, so the estimate is coarser for values far from zero.
- Non-finite counts are not in the bins; check the summary fields of each
  row if a step may have produced NaN or inf.
- For the built-in per-site fan and waterfall views, see
  [monitor training](../monitor_training.md).
