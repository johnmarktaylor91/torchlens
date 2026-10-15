# Streaming statistics: `tl.stats` and `tl.aggregate`

`tl.stats` computes dataset-level statistics of activations or gradients in
constant memory: each accumulator folds one batch at a time and never retains
per-batch tensors. `tl.aggregate` is the one-call door that drives those
accumulators across a whole dataloader. Both are shipped, tested surface --
promoted into `torchlens.__all__` by the completeness work (they were
previously reachable but undeclared). Spellings are DOCUMENTED-UNSTABLE
pending naming-session ratification.

## The accumulators

Every accumulator implements the `tl.stats.StreamingStat` protocol --
`update(value)` folds one batch, `result()` finalizes:

| Accumulator | Result | Notes |
|---|---|---|
| `tl.stats.Mean()` | float | Count-weighted elementwise mean |
| `tl.stats.Norm(p=2.0)` | float | Streaming p-norm over all elements |
| `tl.stats.Quantile(quantiles=...)` | dict | Reservoir-backed quantile estimates; per unit with `unit_axis=` |
| `tl.stats.TopK(k=10)` | list, or `UnitTopK` per unit | Largest k elements seen; per unit with example ids under `unit_axis=` |
| `tl.stats.Covariance()` | Tensor | Feature-space covariance (rows = observations) |
| `tl.stats.CrossCovariance()` | Tensor | Paired `update(a, b)` |
| `tl.stats.CKA()` | `SimilarityScore` (a float) | Paired `update(a, b)`; exact full-data linear CKA; the result names its variant in `similarity_repository_id` |
| `tl.stats.PCA(n_components)` | dict | Streaming PCA; `.fitted()` freezes a transform |
| `tl.stats.Histogram(...)` | histogram | Fixed signed-log2 bins (observability kernel) |
| `tl.stats.Spine(...)` | scalar spine | Always-on scalar summary (observability kernel) |
| `tl.stats.Aggregator(*stats)` | dict | Fans one stream out to several accumulators |

`tl.stats.CKA` is exact: it retains only feature-sized covariance terms, so
the finalized value equals the full-data linear CKA of Kornblith et al.
(2019) computed in one shot -- it is not a minibatch approximation that
depends on batch boundaries. `tl.stats.cka(a, b)` is the one-shot form.

## `tl.aggregate`: statistics across a dataloader

Map layer selectors (any `log[...]` spelling, or `"output"` for the model
output) to accumulators; `tl.aggregate` runs the captures and streams every
batch through them:

```python
# `model` is an application-supplied nn.Module (here: conv -> relu -> conv).
import torch
import torchlens as tl

results = tl.aggregate(
    model,
    [torch.rand(2, 3, 16, 16) for _ in range(4)],   # any iterable of inputs
    metrics={
        "relu_1_2": tl.stats.Aggregator(tl.stats.Mean(), tl.stats.Norm()),
        "output": tl.stats.Quantile(),
    },
)
print(results["relu_1_2"]["Mean"], sorted(results["output"]))
```

Gradient statistics stream the same way: pass `target="grad"` and a
`loss_fn=` used to build the backward pass from `(output, *batch_tail)`:

```python
grad_results = tl.aggregate(
    model,
    [(torch.rand(2, 3, 16, 16),) for _ in range(2)],
    metrics={"relu_1_2": tl.stats.Norm()},
    target="grad",
    loss_fn=lambda output: output.square().mean(),
)
print(grad_results["relu_1_2"] >= 0.0)
```

## Per-unit statistics: neurons, channels and SAE latents

`tl.stats.Quantile` and `tl.stats.TopK` pool every value by default. Name the
unit axis with the keyword-only `unit_axis=` and they keep one reservoir, or
one top-k list, per unit. Axis 0 of every batch is the example axis; every
axis other than the example and unit axes is a position axis (a token
position, a spatial location).

- `Quantile(..., unit_axis=-1).result()` maps each quantile to a `float64
  [units]` tensor: a quantile vector per unit. It is exact while each unit has
  seen at most `reservoir_size` values; beyond that the rank-space error is
  about `sqrt(q * (1 - q) / reservoir_size)`, from a private seeded stream, so
  the same seed gives the same result.
- `TopK(k, unit_axis=-1).result()` is a `tl.stats.UnitTopK`: `values`
  (`float64 [units, k]`, descending), `example_ids` (`int64 [units, k]`) and
  `positions` (`int64 [units, k, position_axes]`, one column per position
  axis). Under `tl.aggregate` an example id is the example's index in the
  dataloader's iteration order, which is the dataset index for an unshuffled
  loader; outside it, `update(batch, example_ids=...)` names the examples.
  Ties break toward the lowest example id, then the lowest position, so the
  answer does not depend on batch size. NaN values are ignored, as in the
  pooled tracker; a unit with fewer than `k` values has NaN values and `-1`
  ids in its empty slots.
- `encoder=` applies a callable to each batch before it is folded, so SAE
  latents use the same tool as neurons: pass the SAE's encoder and the site's
  activations become latents inside the loop. `tl.stats.Aggregator` fans one
  site out to a neuron tracker and a latent tracker.

Memory: per-unit state is `8 * reservoir_size` bytes per unit for `Quantile`
(64 KiB at the default 8192) and `k * (16 + 8 * position_axes)` bytes per unit
for `TopK` (160 B at the default `k=10` with no position axis, 240 B with one).
Past 256 MiB of state, a `stats_per_unit_state_large` warning fires once per
session and names the cheaper spelling: a smaller `reservoir_size=` or `k=`,
or fewer units through `encoder=lambda v: v.narrow(axis, 0, n)`. No unit is
ever dropped.

```python
import torch
import torchlens as tl
from torch import nn

torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
sae_encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU()).eval()  # a tiny SAE encoder
stimuli = [torch.randn(3, 4) for _ in range(3)]  # 9 examples, ragged batches welcome

relu_label = next(
    op.layer_label for op in tl.trace(mlp, stimuli[0]).layer_list if op.layer_type == "relu"
)
per_unit = tl.aggregate(
    mlp,
    stimuli,
    metrics={
        relu_label: tl.stats.Aggregator(
            tl.stats.TopK(k=3, unit_axis=-1, name="neurons"),
            tl.stats.TopK(k=3, unit_axis=-1, encoder=sae_encoder, name="latents"),
            tl.stats.Quantile((0.5, 0.9), unit_axis=-1, name="neuron_quantiles"),
        )
    },
)[relu_label]
neurons, latents = per_unit["neurons"], per_unit["latents"]
print(neurons.values.shape, latents.values.shape)  # (8, 3) and (16, 3)
print("top stimuli of neuron 0:", neurons.example_ids[0].tolist())
print("top stimuli of latent 0:", latents.example_ids[0].tolist())
print(per_unit["neuron_quantiles"][0.5].shape)  # one median per neuron
```

## Paired statistics (CKA between two layers)

Paired accumulators take `update(a, b)`, so drive them from one capture per
batch:

```python
similarity = tl.stats.CKA()
for batch in [torch.rand(2, 3, 16, 16) for _ in range(3)]:
    log = tl.trace(model, batch)
    similarity.update(
        log["conv2d_1_1"].out.flatten(1),
        log["relu_1_2"].out.flatten(1),
    )
print(0.0 <= similarity.result() <= 1.0)
```

## Fitted PCA as a persisted transform

`tl.stats.PCA(...).fitted(...)` freezes a fit into a `FittedPCA` payload;
`tl.stats.save_fitted` / `tl.stats.load_fitted` persist it, and the
activation-transform kernel `pca_apply(fitted)` consumes it at extraction
time (see [docs/reference/transforms.md](transforms.md)). Fit-during-write is
refused -- early and late shards would get different coordinates.

## Where it sits

- Install both: lovely-tensors for any tensor at a REPL, TorchLens for the
  numbers of a whole model's run (this page across a dataset, the stats line on
  each op of a trace).
- `tl.aggregate` captures with the ordinary verified `tl.trace` engine; for
  tight training loops that only need a few sites, `tl.fastlog.Recorder`
  (built once, `.log()` per step) is the cheaper repeated-capture door.
- The observability kernels (`Spine`, `Histogram`) have one implementation
  home shared with the `torchlens.trackers.watch` tiers; see
  [docs/reference/observability_substrate.md](observability_substrate.md).
