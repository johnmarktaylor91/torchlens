# Migration: the stats line and value reprs (F10)

TorchLens value displays converged on one grammar (see
`docs/reference/lovely_reprs.md`). What changed for existing users:

## Behavior changes

- `repr(op)` / `repr(layer)` are now ONE line (envelope + core stats);
  `str(...)` is a bounded card whose first line is the repr. The
  historical 21-line Op dump (8x8 preview, full relation lists, eleven
  lookup keys) moved behind the card's `More:` exits (`.out`,
  `.parents`, `.children`, `.func_config`, `.lookup_keys`).
- `print(trace)` is bounded: honesty header first, buffer rows folded to
  one line, module hierarchy and layer roster elided past their rungs
  with EXACT remainders (`... N more layers (see .layer_list)`).
- Collection reprs (`trace.layers`, `trace.params`, `trace.modules`,
  `trace.buffers`) are one-line composition cards with `.head()` /
  `.find()` / `.to_pandas()` exits -- never member dumps.
- `tensor_stats_summary` (the legacy helper line) DROPPED the `neg=`
  field from its default output: on real float activations it read ~50%
  nearly always, and the exact count cost a full extra traversal for a
  fact the sparkline shows. The record and table column keep the exact
  zero/sign story.
- Resolved selections iterate in GRAPH (execution) order, not
  lexicographic site-key order.

## Population sd (divergence from lovely-tensors, on purpose)

TorchLens prints the POPULATION standard deviation (`unbiased=False`):
a line describing this tensor describes a population, not an estimate of
another one. lovely-tensors prints the SAMPLE sd. On `torch.arange(8)`
that is **2.291 (TorchLens) vs 2.449 (lovely)** -- neither tool is
broken; the denominators differ (N vs N-1).

## Exactness model

Extremes, NaN/Inf counts, zero/true counts: exact at every size. Mean:
exact on every supported default path. Only sd and histogram may sample
(above 2^26 / 2^22 elements respectively), from a seeded gathered sample,
marked `~` with disclosed precision. Gate constants are CPU-derived
display policy pending the CUDA re-derivation.
