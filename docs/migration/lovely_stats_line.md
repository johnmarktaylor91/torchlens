# Coming from lovely-tensors

This page is for people who use lovely-tensors to print a one-line summary of a tensor and want
to know what TorchLens's stats line is and how the two differ. Install both: lovely-tensors
for any tensor at a REPL, TorchLens for the numbers of a whole model's run. TorchLens's stats
line complements lovely-tensors and does not replace it. The examples were written against the
lovely-tensors range TorchLens's test environment installs (`lovely-tensors~=0.1.18`, in the
`viz` extra); the lovely-tensors example skips where it is not installed. Checked 2026-11-03.

## The same task in both

Print the summary line of `torch.arange(8)` as a float tensor and read its standard deviation.

### In lovely-tensors

```python
# migration-test: tool=lovely_tensors expected=True
import lovely_tensors as lt
import torch

values = torch.arange(8, dtype=torch.float32)
line = str(lt.lovely(values))
RESULT = "2.449" in line
```

### In TorchLens

TorchLens renders the same kind of line with a function, never a monkey-patch:
`torchlens.stats.glance(t)` for any dense tensor, and `repr(op)` for a captured op.

```python
# migration-test: tool=torchlens expected=True
import torch
from torchlens.stats import glance

values = torch.arange(8, dtype=torch.float32)
line = glance(values)
RESULT = "sd=2.291" in line
```

## Translation table

TorchLens value displays converged on one grammar (see `docs/reference/lovely_reprs.md`).

| lovely-tensors | TorchLens | Notes |
| --- | --- | --- |
| `lt.lovely(t)` for one tensor | `torchlens.stats.glance(t)` | A documented strict subset of the full line: no envelope, no deviation marks, no graph-proven role. |
| `lt.monkey_patch()` so `repr(tensor)` is lovely | Not provided | TorchLens does not change how a `torch.Tensor` prints; its `repr(op)` and `repr(layer)` are the one-line stats for captured values. |
| The summary of every intermediate of a forward | `repr(op)` for one captured value, `print(trace)` for a bounded overview, `trace.layers` for the roster | The op line carries the provenance envelope. |
| Image and channel views (`lt.rgb`, `lt.chans`) | Not provided | These stay in lovely-tensors. |
| Sample standard deviation (`sd` is N-1) | Population standard deviation (N) | On purpose; see below. |

## What TorchLens adds

- The line for every op of a model's run, with the op's place in the graph:
  `repr(op)` is one line (envelope plus core stats), `str(op)` a bounded card whose first line
  is the repr, and `print(trace)` a bounded overview.
- Exactness stated per number: extremes, NaN and Inf counts, and zero and true counts are exact
  at every size; sampled quantities carry a `~` mark (see "Exactness model" below).
- Honest markers for values that are not there: `(not saved)`, `(freed)`, `(edited)`,
  `(disk-backed: metadata only until read)`.
- A live per-firing stream of the line during capture, `torchlens.stats.narrate(site,
  sink=print)` (`docs/reference/lovely_reprs.md`).

## What lovely-tensors does that TorchLens does not

- It summarizes any tensor at a REPL with one import and, with `monkey_patch()`, changes how
  every tensor prints. TorchLens's `glance` is a function you call.
- It draws images and channel grids from tensors (`lt.rgb`, `lt.chans`).
  TorchLens's stats line carries a sparkline, not an image.
- It is the right tool when you have a tensor and no model run to describe. See
  [where to go instead](where_to_go_instead.md).

## Using both

### lovely-tensors feeds TorchLens

There is no bridge. The two do not conflict: a TorchLens trace holds ordinary tensors
(`trace[label].out`), which lovely-tensors prints like any other tensor.

### TorchLens feeds lovely-tensors

There is no bridge. Pass any saved activation to `lt.lovely(...)` to see its lovely line
beside TorchLens's. The sample standard deviation lovely-tensors prints will differ from
TorchLens's population one, as below.

## Semantic differences

### Behavior changes

What changed for existing TorchLens users:

- `repr(op)` / `repr(layer)` are now ONE line (envelope + core stats);
  `str(...)` is a bounded card whose first line is the repr. The
  historical 21-line Op dump (8x8 preview, full relation lists, eleven
  lookup keys) moved behind the card's `More:` exits (`.out`,
  `.parents`, `.children`, `.func_config`, `.lookup_keys`).
- `print(trace)` is bounded: cached health in the identity line and at most
  twelve rows of stored metadata. The module hierarchy lives in `.summary()`.
- Collection reprs (`trace.layers`, `trace.params`, `trace.modules`,
  `trace.buffers`) are one-line count and composition cards. Their `str()` shows
  at most five members, an exact remainder and `.head()` / `.search()` /
  `.to_pandas()` exits.
- `tensor_stats_summary` (the legacy helper line) DROPPED the `neg=`
  field from its default output: on real float activations it read ~50%
  nearly always, and the exact count cost a full extra traversal for a
  fact the sparkline shows. The record and table column keep the exact
  zero/sign story.
- Resolved selections iterate in GRAPH (execution) order, not
  lexicographic site-key order.

### Population sd (divergence from lovely-tensors, on purpose)

TorchLens prints the POPULATION standard deviation (`unbiased=False`):
a line describing this tensor describes a population, not an estimate of
another one. lovely-tensors prints the SAMPLE sd. On `torch.arange(8)`
that is **2.291 (TorchLens) vs 2.449 (lovely)** -- neither tool is
broken; the denominators differ (N vs N-1).

### Exactness model

Extremes, NaN/Inf counts, zero/true counts: exact at every size. Mean:
exact on every supported default path. Only sd and histogram may sample
(above 2^26 / 2^22 elements respectively), from a seeded gathered sample,
marked `~` with disclosed precision. Gate constants are CPU-derived
display policy pending the CUDA re-derivation.
