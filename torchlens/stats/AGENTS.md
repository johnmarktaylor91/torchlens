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
- `save_fitted` / `load_fitted` — the one persistence path: fitted PCA
  (`_fitted.py`) and every accumulator's whole state (`_state_io.py`,
  `tl_stats_state_v1`: JSON description plus safetensors tensors, digest
  checked, class allow-list, never pickle). A new accumulator class joins
  `_state_io._classes()`; its attributes must be tensors, numbers, strings,
  random streams or other listed classes.
- `aggregate(..., cache=True)` — `_aggregate_cache.py` keys the run by value
  and restores saved states in place on a hit; anything it cannot key
  refuses (`stats_aggregate_cache_unkeyable`), never a guessed key.

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
