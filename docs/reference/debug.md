# `tl.debug` reference

`tl.debug` inspects a completed trace when the graph itself is the diagnostic: it can
locate non-finite saved tensors, walk dependencies, compare runs, rank cost and
recomputation candidates, audit saved gradients, and infer a runnable input shape. Most
table-returning helpers require pandas. Capture with the normal `tl.trace(...)` API, then
pass the resulting trace to a helper.

For *environment* problems (rather than trace problems), start with
[`tl.utils.doctor()`](#tlutilsdoctor) below.

## `tl.utils.doctor`

`tl.utils.doctor()` runs a TorchLens startup health check and returns a structured
`DoctorReport` — use it first when captures fail before a trace even exists (import
errors, missing renderers, stale wrapper installs). It lives in `tl.utils`, not
`tl.debug`, because it needs no trace.

The report is a tuple of `DoctorCheck(name, status, detail)` rows, one per probe, with
`status` one of `"PASS"`, `"FAIL"`, `"SKIP"`, or `"WARN"`:

- `pytorch` — the installed torch version.
- `runtime capabilities` — the feature-detected capability snapshot (the same named
  `HAS_*` flags surfaced by `tl.compat.report()`; graceful degradations flip these).
  The row shows the grouped absences-first summary; the full `flag -> available`
  mapping lives on the detail accessors below.
- `torch wrapper bindings` — warning-only detector for stale torch namespace
  attributes after wrap/unwrap cycles (cannot see closure-bound local aliases).
- `cuda` — device availability and count (`SKIP` on CPU-only hosts).
- `graphviz` / `safetensors` / `extras` — optional-dependency probes for rendering
  and serialization paths.
- `model fingerprint` — a tiny end-to-end capture probe.

```python
import torchlens as tl

report = tl.utils.doctor()
print(report.show())          # text table
bad = [c for c in report.checks if c.status == "FAIL"]
```

A `FAIL` row names the missing dependency or broken probe in `detail`; `WARN` rows are
degraded-but-usable states. For per-model compatibility questions (unsupported tensor
variants, distributed state, wrapper coverage), use `tl.compat.report(model, x)`
instead — `doctor()` checks the environment, `compat.report()` checks one model.

### Capability snapshot accessors

`tl.utils.capability_snapshot()` (DOCUMENTED-UNSTABLE spelling) returns the full
runtime capability snapshot as a `flag -> available` mapping — every feature-detected
`HAS_*` flag, torch plus the optional preview backends. The same mapping is served as
a detail accessor on both report objects: `DoctorReport.capability_snapshot()` and
`CompatReport.capability_snapshot()`. Report rows and reprs deliberately show only the
grouped summary (absences are the signal; the full `name=value` dump was a
~1,900-character line), rendered by the one shared formatter
`tl.utils.format_capability_summary(snapshot)` (DOCUMENTED-UNSTABLE spelling), which
returns `present/total capabilities present` plus the sorted absent flag names.

```python
import torchlens as tl

snapshot = tl.utils.capability_snapshot()
print(all(isinstance(v, bool) for v in snapshot.values()))
print(tl.utils.format_capability_summary({"HAS_A": True, "HAS_B": False}))
```

Output:

```text
True
1/2 capabilities present; absent: HAS_B
```

## `audit_trace`

`tl.debug.audit_trace(trace)` runs every trace-local health diagnostic one capture
supports and returns a `TraceAudit`: findings from `find_nan`, `bisect_nan`
(full-coverage traces), `dtype_range_audit`, and `gradient_flow_audit` (exactly one
captured backward pass). Diagnostics that need more than the trace itself (a second
trace, a fresh execution, a start op, a `bwd=` selection) are listed in `skipped`
with a reason and are never counted as checks that ran. Accepts a completed `Trace`
or a failed `PartialTrace`.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.ReLU(), torch.tensor([[-1.0, 2.0]]))
report = tl.debug.audit_trace(trace)
print(report.checks_run, len(report.skipped) > 0)
```

Output:

```text
('find_nan', 'bisect_nan', 'dtype_range_audit') True
```

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

## `bisect_precision`

`tl.debug.bisect_precision(model, input_args)` (DOCUMENTED-UNSTABLE spelling) runs the
same seeded forward twice -- once at native precision, once with floating state and
inputs cast to `reference_dtype` (fp64 by default) -- and returns a
`BisectPrecisionResult` naming the first op whose native output separates from the
high-precision reference beyond tolerance, plus the full per-op error table in
`result.rows` and disclosed skips in `result.skipped`. Both runs happen on deep copies
under a forked RNG, so the caller's model and global RNG are untouched. Default
tolerances derive from each op's native dtype (`rtol = eps ** 0.5`, `atol = eps * 10`),
so "diverged" means "lost meaningfully more precision than the dtype itself explains".
A first divergence at a stochastic op (dropout et al.) is flagged: random kernels may
consume RNG differently across dtypes, so that usually means mask mismatch, not
precision loss -- bisect in eval mode.

```python
import torch
from torch import nn
import torchlens as tl


class Cancelling(nn.Module):
    """Model that destroys fp32 mantissa bits at a known op."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(8, 8)

    def forward(self, x):
        return (self.pre(x) + 1e7) - 1e7


torch.manual_seed(0)
result = tl.debug.bisect_precision(Cancelling().eval(), torch.randn(2, 8))
print(result.found, result.func_name)
```

Output:

```text
True __sub__
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

## `compare_params`

`tl.debug.compare_params(model_a, model_b, *, rtol=1e-5, atol=1e-8, include_buffers=False)`
(DOCUMENTED-UNSTABLE spelling) is the weight-space counterpart of `compare`: it aligns two
models' parameters on their qualified names and returns a pandas DataFrame with one row per
name and aggregate counts in `attrs`. Tensors on different devices are compared on CPU;
shape/dtype mismatches and meta tensors skip the value comparison with the reason recorded.
`include_buffers=True` adds registered buffers (rows carry `kind="buffer"`).

```python
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
left = nn.Linear(4, 2)
right = nn.Linear(4, 2)
right.load_state_dict(left.state_dict())
with torch.no_grad():
    right.bias.add_(1.0)
report = tl.debug.compare_params(left, right)
print(report.attrs["matched"], report.attrs["value_diverged"])
print(report.loc[report["allclose"] == False, "name"].tolist())  # noqa: E712
```

Output:

```text
1 1
['bias']
```

## `count_compiles`

`tl.debug.count_compiles()` measures Dynamo compilation events (including recompiles) across a
block, through the feature-detected `HAS_DYNAMO_COMPILE_COUNTERS` capability. Use it to verify the
torch.compile coexistence contract on your own model: zero compiles while a capture holds the
`force_eager` stance (torch >= 2.6), and at most one bounded recompile on the next compiled call
after capture. It raises `CompileCountsUnavailableError` when the runtime exposes no Dynamo
counters.

```python
import warnings

import torch
from torch import nn
import torchlens as tl


class Compiled(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act = torch.compile(lambda t: torch.relu(t))

    def forward(self, x):
        return self.act(self.fc(x))


model = Compiled()
x = torch.randn(2, 4)
model(x)  # warm the compile cache

with warnings.catch_warnings(record=True) as capture_notes:
    warnings.simplefilter("always")
    with tl.debug.count_compiles() as during:
        tl.trace(model, x)
assert any("force_eager" in str(note.message) for note in capture_notes)
with tl.debug.count_compiles() as after:
    model(x)
print(during.frames_compiled, after.frames_compiled <= 1)
```

Output:

```text
0 True
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

## `dtype_range_audit`

`tl.debug.dtype_range_audit(trace, *, max_fraction=0.9, subnormal_fraction_threshold=0.1)`
audits saved activations for numeric-range and precision hazards: non-finite values
(same classifier as [`find_nan`](#find_nan)), finite magnitudes near the saved dtype's
finite maximum, subnormal-heavy tensors, and downcasts from recorded wider input dtypes.
Returns a `DTypeRangeAudit` with structured findings and audited/total coverage
(`n_ops_audited` / `n_ops_total`) — unsaved and non-tensor payloads are never
inspected, and the coverage says so.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.ReLU(), torch.tensor([[-1.0, 2.0]]))
audit = tl.debug.dtype_range_audit(trace)
print(len(audit.findings), audit.n_ops_audited <= audit.n_ops_total)
```

Output:

```text
0 True
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

## Pandas-free row cores (`*_rows`)

`tl.debug.compare_rows(trace_a, trace_b, *, rtol=1e-5, atol=1e-8)`,
`tl.debug.dead_neurons_rows(trace, *, dim=1, threshold=0.0)`,
`tl.debug.gradient_flow_audit_rows(trace, *, bwd=None, vanishing_threshold=1e-7,
exploding_threshold=1e4)`, and `tl.debug.hot_path_rows(trace, by="flops")`
(DOCUMENTED-UNSTABLE spellings) are the pandas-free cores of
[`compare`](#compare), [`dead_neurons`](#dead_neurons),
[`gradient_flow_audit`](#gradient_flow_audit), and [`hot_path`](#hot_path).
Each returns `(rows, attrs)` — a list of plain per-row dicts plus the same
`attrs` mapping the DataFrame view carries (aggregate counts, parameters,
capture-honesty facts) — importable and runnable without pandas. The
DataFrame helpers are thin views over these cores, so both spellings always
report identical facts; agent/MCP consumers use the row cores directly.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.ReLU()
left = tl.trace(model, torch.tensor([[-1.0, 2.0]]))
right = tl.trace(model, torch.tensor([[-1.0, 3.0]]))
rows, attrs = tl.debug.compare_rows(left, right)
print(len(rows), attrs["value_diverged"])
```

Output:

```text
2 2
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

result = tl.debug.infer_input_shape(nn.Linear(3, 2), max_probes=4)
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

## `find_nan`

`tl.debug.find_nan(model, x, **trace_kwargs)` runs a memory-light live capture and stops at
the first non-finite operation output. `trace.find_nan()` instead scans the payloads saved in an
existing trace; on a selective-save trace, its result explicitly identifies the uncertainty zone
created by unsaved operations.

```python
import torch
from torch import nn
import torchlens as tl


class NonFinite(nn.Module):
    def forward(self, x):
        zeros = x - x
        return zeros / zeros


model = NonFinite()
x = torch.ones(1, 2)
live = tl.debug.find_nan(model, x)
saved = tl.trace(model, x).find_nan()
print(live.found, live.kind)
print(saved.found, saved.kind)
```

Output:

```text
True nan
True nan
```
