# stats/ - Implementation Guide

Streaming statistics over activations and dataloaders (`tl.stats`, lazy; one
module, 12-name `__all__`).

## Surface

- `StreamingStat` (Protocol) — the pluggable accumulator contract.
- Built-in accumulators: `Mean`, `Norm`, `Quantile`, `TopK`, `Covariance`,
  `CrossCovariance`, `PCA`, `CKA`. `Quantile` and `TopK` live in
  `_per_unit.py`: pooled by default, per unit (neuron, channel, SAE latent)
  with `unit_axis=`; per-unit `TopK` returns `UnitTopK` with example ids and
  positions, and `encoder=` maps each batch (an SAE encoder) before folding.
- `Aggregator` — drives a set of named `StreamingStat`s over repeated
  captures.
- `aggregate(...)` — the public one-call spelling: iterate a dataloader,
  capture each batch, fold the selected layers' activations into the
  requested stats, return the aggregated results.
- `cka(a, b)` — direct two-matrix CKA convenience.

## Gotchas

- Accumulators are STREAMING by contract: constant memory in the number of
  batches; keep new stats one-pass (Welford-style) — never buffer the whole
  activation history. Per-unit state is units x reservoir (or k) by design;
  its layout is documented in `_per_unit.py` and past `STATE_WARN_BYTES` it
  warns once, never drops units.
- `aggregate` owns the capture loop; it should compose with `save=`
  predicates rather than exhaustive saves for memory sanity.

## Tests

`pytest -k "stats or aggregate or cka"` over `tests/`.
