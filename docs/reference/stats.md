# Streaming statistics: `tl.stats` and `tl.aggregate`

`tl.stats` computes dataset-level statistics of activations or gradients in
constant memory: each accumulator folds one batch at a time and never retains
per-batch tensors. `tl.aggregate` is the one-call door that drives those
accumulators across a whole dataloader. Both are shipped, tested surface --
promoted into `torchlens.__all__` by the completeness megasprint (they were
previously reachable but undeclared). Spellings are DOCUMENTED-UNSTABLE
pending naming-session ratification.

## The accumulators

Every accumulator implements the `tl.stats.StreamingStat` protocol --
`update(value)` folds one batch, `result()` finalizes:

| Accumulator | Result | Notes |
|---|---|---|
| `tl.stats.Mean()` | float | Count-weighted elementwise mean |
| `tl.stats.Norm(p=2.0)` | float | Streaming p-norm over all elements |
| `tl.stats.Quantile(quantiles=...)` | dict | Reservoir-backed quantile estimates |
| `tl.stats.TopK(k=10)` | list | Largest k elements seen |
| `tl.stats.Covariance()` | Tensor | Feature-space covariance (rows = observations) |
| `tl.stats.CrossCovariance()` | Tensor | Paired `update(a, b)` |
| `tl.stats.CKA()` | float | Paired `update(a, b)`; exact full-data linear CKA |
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

- `tl.aggregate` captures with the ordinary verified `tl.trace` engine; for
  tight training loops that only need a few sites, `tl.fastlog.Recorder`
  (built once, `.log()` per step) is the cheaper repeated-capture door.
- The observability kernels (`Spine`, `Histogram`) have one implementation
  home shared with the `torchlens.trackers.watch` tiers; see
  [docs/reference/observability_substrate.md](observability_substrate.md).
