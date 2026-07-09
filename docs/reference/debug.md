# `tl.debug` reference

`tl.debug` inspects a completed trace when the graph itself is the diagnostic: it can
locate non-finite saved tensors, walk dependencies, compare runs, rank cost and
recomputation candidates, audit saved gradients, and infer a runnable input shape. Most
table-returning helpers require pandas. Capture with the normal `tl.trace(...)` API, then
pass the resulting trace to a helper.

## `bisect_nan`

`tl.debug.bisect_nan(trace)` returns the first saved operation with a NaN or Inf output.
It returns a `BisectNanResult`; it does not raise when every saved output is finite. See
[`lineage`](#lineage) to inspect the offending operation's neighborhood.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Identity(), torch.tensor([float("nan")]))
result = tl.debug.bisect_nan(trace)
print(result.found, result.kind)
```

Output:

```text
True nan
```

## `compare`

`tl.debug.compare(trace_a, trace_b, *, rtol=1e-5, atol=1e-8)` compares saved dense
floating activations with matching operation labels and returns a pandas DataFrame. Its
`attrs` contain aggregate counts.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.ReLU()
left = tl.trace(model, torch.tensor([[-1.0, 2.0]]))
right = tl.trace(model, torch.tensor([[-1.0, 3.0]]))
comparison = tl.debug.compare(left, right)
print(comparison.shape, comparison.attrs["value_diverged"])
```

Output:

```text
(2, 8) 2
```

## `dead_neurons`

`tl.debug.dead_neurons(trace, *, dim=1, threshold=0.0)` reports units whose maximum
activation is at most `threshold` or whose value has zero variance in this trace. One
trace is only one sample; aggregate runs before making a dataset-level claim. Related:
[`gradient_flow_audit`](#gradient_flow_audit).

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.ReLU(), torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]))
report = tl.debug.dead_neurons(trace)
print(report.columns.tolist())
```

Output:

```text
['op', 'total_units', 'dead_count', 'dead_frac', 'sample_dead_idx', 'reason']
```

## `gradient_flow_audit`

`tl.debug.gradient_flow_audit(trace, *, bwd=None, vanishing_threshold=1e-7,
exploding_threshold=1e4)` ranks saved gradients and flags zero, vanishing, and exploding
norms. Capture with `save_grads=True`, then call `trace.log_backward(...)` first.

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.options import CaptureOptions

trace = tl.trace(nn.Linear(2, 1), torch.ones(1, 2), capture=CaptureOptions(save_grads=True))
trace.log_backward(trace[trace.output_layers[0]].out.sum())
report = tl.debug.gradient_flow_audit(trace)
print(report.shape, report.attrs["bwd"])
```

Output:

```text
(2, 7) 1
```

## `hot_path`

`tl.debug.hot_path(trace, by="flops")` aggregates forward FLOPs, activation memory, or
duration by source line. `by` is one of `"flops"`, `"memory"`, or `"duration"`; use
[`recompute_candidates`](#recompute_candidates) when the question is activation-memory
tradeoffs instead.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
report = tl.debug.hot_path(trace, by="flops")
print(report.columns.tolist(), report.attrs["metric"])
```

Output:

```text
['source_file:line', 'op_count', 'total_cost', 'pct_total'] flops
```

## `infer_input_shape`

`tl.debug.infer_input_shape(model, *, batch_size=1, input_dtype=None, channels=None,
spatial_rank="auto", seq_len=None, square=True, min_size=1, max_size=512,
preferred_sizes=(224, 256, 384, 299, 128, 96, 64, 32, 28), max_probes=64, device=None,
seed=0, return_trace=False, on_failure="return", input_specs=None)` probes a module and
returns an `InferInputShapeResult` with a verified synthetic shape or a diagnostic failure.

```python
import torch
from torch import nn
import torchlens as tl

result = tl.debug.infer_input_shape(nn.Linear(3, 2), spatial_rank=0, max_probes=4)
print(result.found, result.shape)
```

Output:

```text
True (1, 3)
```

## `lineage`

`tl.debug.lineage(trace, op_or_label, *, direction="ancestors", max_depth=None)` walks
parents, children, or both from an operation accepted by trace indexing. It returns a
`LineageResult` with labels, depths, source locations, shapes, and dtypes. Pair it with
[`bisect_nan`](#bisect_nan) after locating a non-finite output.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Sequential(nn.Linear(2, 2), nn.ReLU()), torch.ones(1, 2))
result = tl.debug.lineage(trace, trace.output_layers[0], max_depth=1)
print(result.message, len(result.nodes))
```

Output:

```text
2 node(s) 2
```

## `recompute_candidates`

`tl.debug.recompute_candidates(trace, *, budget_gb=None)` ranks operations by activation
memory per forward FLOP and optionally marks a greedy set that reaches an activation-memory
budget. See [`hot_path`](#hot_path) for source-line cost ranking.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
report = tl.debug.recompute_candidates(trace)
print(report.columns.tolist())
```

Output:

```text
['op', 'activation_memory', 'flops_forward', 'mem_per_flop', 'suggested']
```

<!-- TODO: document tl.debug.find_nan after the parallel implementation ships. -->
