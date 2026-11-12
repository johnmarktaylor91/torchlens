# `tl.debug` reference

`tl.debug` inspects a completed trace when the graph itself is the diagnostic: it can
locate non-finite saved tensors, walk dependencies, compare runs, rank cost and
recomputation candidates, audit saved gradients, and infer a runnable input shape. Capture
with the normal `tl.trace(...)` API, then pass the resulting trace to a helper. A helper that
takes `inputs` itself (`bisect_precision`, `check_determinism`, `find_nan`, `graph_breaks`,
`isolated_capture`, `nonfinite_origin`, `tl.report.flops_vs_dispatch`) takes every input form
`tl.trace` takes: a tensor, a positional list or tuple, and for a Hugging Face model a string, PIL
image or modality dict, resolved as the trace resolves it.

## The report object (`Report`)

Every report-shaped `tl.debug` result is a `tl.report.Report` (experimental):
`report_type`, `subject` and `status` name what was examined and the verdict,
`to_dict()` returns a plain JSON-friendly payload, `to_pandas()` returns the table, and
`print(report)` shows a bounded card (a lead line, at most five findings, the exact
remainder, and the exits). The typed results (`HealthReport`, `DeterminismReport`,
`NonfiniteReport`, `FindNanReport`, `ParamHealthReport`, ...) are `Report` subclasses; results
that already print a house-voice summary keep it, and `report.card()` gives the card.
The table helpers (`compare`, `compare_params`, `dead_neurons`, `grad_health`,
`hot_path`, `recompute_candidates`) return a tabular `Report`: the report has a
table and never is one, so read the DataFrame with `.to_pandas()` (it needs pandas; the
`*_rows` cores below do not).

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
report = tl.debug.hot_path(trace, by="flops")
print(isinstance(report, tl.report.Report), report.report_type)
print(report.to_pandas().columns.tolist())
```

Output:

```text
True hot_path
['source_file:line', 'op_count', 'total_cost', 'pct_total']
```

For *environment* problems (rather than trace problems), start with
[`tl.report.doctor()`](#tlreportdoctor) below.

## `tl.report.doctor`

`tl.report.doctor(model=None, inputs=None)` returns a `tl.report.DoctorReport`.
It always checks the environment; supply a model and optional inputs to add compatibility
findings. Inputs are inspected and the supplied model is never executed.

`DoctorReport.checks` is a tuple of `tl.report.DoctorCheck` rows. Each row exposes
`key`, `label`, `status`, `severity`, `is_detected`, `details`, and `suggestion`.
Statuses are `pass`, `known_broken`, `scope`, and `not_tested`; severities are
`ok`, `info`, `warning`, and `error`. Records are built by TorchLens; constructor
keywords are outside the public contract.

Environment rows cover PyTorch, `torch_capabilities`, torch wrapper bindings, mechanical
belt coverage, CUDA, Graphviz, safetensors, optional extras, and model fingerprinting.
Missing optional extras are disclosed without implying a broken installation.
Model rows inspect supported wrappers, tensor variants, distributed state, and capture
scope. Stable keys let you retrieve one finding with `report.check(key)`.

```python
import torchlens as tl

report = tl.report.doctor()
report.show()                # prints the fixed-width table
text = str(report)           # the same table as text
bad = [c for c in report.checks if c.status == "known_broken"]
# Add a model and input tree: tl.report.doctor(model, inputs)
```

`DoctorReport.model_type` is `None` without a model and the qualified model type otherwise;
`torch_version` is always set. `to_markdown()` returns a Markdown table. The bounded repr
summarizes findings.

### Capability snapshot accessor

`report.capability_snapshot()` returns the full `flag -> available` mapping of named
`HAS_*` capabilities, including the optional non-PyTorch backends. The `torch_capabilities` row
shows a grouped summary and names missing, unprobed, and optional capabilities; graceful
degradations remain visible. Optional absences and lazy probes that have not run do not
imply a failure.

## `param_health`

`tl.debug.param_health(target, *, include_buffers=True, within=None,
bounds=None, max_fraction=0.9, subnormal_fraction_threshold=0.1)` is the
one-shot parameter/buffer-space scan (the checks-kit door; canonical home
`torchlens.debug.param_health`, experimental). `target` is an
`nn.Module` or ANY name->tensor `Mapping` — a `state_dict`, flattened
optimizer state, EMA shadow weights, a loaded checkpoint: same door, same
batched scan kernel. Returns a `ParamHealthReport` (findings / rows / checks_run /
skipped-with-reason / coverage) with signed nonfinite counts, declared-bounds,
dtype-headroom, subnormal, and all-same/all-zero findings. See
[the checks kit](checks_kit.md) for the recipes.

```python
import torch
from torch import nn
import torchlens as tl

param_audit = tl.debug.param_health(nn.Linear(2, 1))
print(len(param_audit.rows), len(param_audit.findings))
```

Output:

```text
2 0
```

## Trace health check (`trace.audit()`)

`trace.audit()` runs every trace-local health diagnostic one capture
supports and returns a `HealthReport`: findings from `find_nan`, `bisect_nan`
(full-coverage traces), `dtype_range`, and `grad_health` (exactly one
captured backward pass). Diagnostics that need more than the trace itself (a second
trace, a fresh execution, a start op, a `bwd=` selection) are listed in `skipped`
with a reason and are never counted as checks that ran. Accepts a completed `Trace`
or a failed `PartialTrace`.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.ReLU(), torch.tensor([[-1.0, 2.0]]))
report = trace.audit()
print(report.checks_run, len(report.skipped) > 0)
```

Output:

```text
('find_nan', 'bisect_nan', 'dtype_range_audit') True
```

## `bisect_nan`

`tl.debug.bisect_nan(trace)` returns the first saved operation with a NaN or Inf output.
It returns a `BisectNanReport`; it does not raise when every saved output is finite. See
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

`tl.debug.bisect_precision(model, inputs)` (experimental) runs the
same seeded forward twice -- once at native precision, once with floating state and
inputs cast to `reference_dtype` (fp64 by default) -- and returns a
`BisectPrecisionReport` naming the first op whose native output separates from the
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

`tl.debug.compare(trace_a, trace_b, *, rtol=None, atol=None, align="auto", module_map=None)`
compares the saved activations of paired operations and returns a tabular `Report` whose
`.to_pandas()` DataFrame carries aggregate counts in `attrs`. On one backend it pairs
operations (default tolerance `rtol=1e-5, atol=1e-8`); across backends, or with
`align="structure"`, it pairs module calls (see
[Comparing runs from different backends](#comparing-runs-from-different-backends)).

How operations pair under `align="auto"` on one backend: when the two traces have the same
graph fingerprint (`trace.provenance.fingerprint`), operations pair by label. When the graphs
differ, for example one run has an extra `x * 1.0`, every later label shifts, so operations
pair by graph structure instead: same module, same function, already-paired inputs, in
execution order, and never by value. An operation with no structural partner is an `only-a`
or `only-b` row, so an inserted operation is reported as itself and every other operation is
still compared. When operations with the same structure differ in number between the two
runs (two identical `relu` calls on one input against one), TorchLens cannot tell which
correspond, so none of them pairs and their rows say `ambiguous`. A row paired by
structure under a different label notes `paired by graph structure` in `reason`.
`align="label"` always pairs by label, even when the fingerprints differ. Runs on different devices
are compared on the CPU and different dtypes in the wider dtype; each row's `reason`
and the card's coverage line name the alignment. Integer and boolean activations
(indices, masks) must match exactly.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.ReLU()
left = tl.trace(model, torch.tensor([[-1.0, 2.0]]))
right = tl.trace(model, torch.tensor([[-1.0, 3.0]]))
comparison = tl.debug.compare(left, right).to_pandas()
print(comparison.shape, comparison.attrs["value_diverged"])
```

Output:

```text
(2, 9) 2
```

## `compare_params`

`tl.debug.compare_params(model_a, model_b, *, rtol=1e-5, atol=1e-8, include_buffers=False)`
(experimental) is the weight-space counterpart of `compare`: it aligns two
models' parameters on their qualified names and returns a tabular `Report` whose
`.to_pandas()` DataFrame has one row per name and aggregate counts in `attrs`. Tensors on different devices are compared on CPU;
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
report = tl.debug.compare_params(left, right).to_pandas()
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
`force_eager` stance (torch >= 2.6), and a bounded burst of recompilations on the first
compiled call after each capture, then steady state (measured: none on `resnet18`, six to seven frames on a
tiny GPT-2). It raises `CompileCountsUnavailableError` when the runtime exposes no Dynamo
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

`tl.debug.dead_neurons(trace, *, dim=None, max_activation=0.0)` reports, for every layer, the
units whose output is identically zero at every position in the batch: every element of
the unit's slice has a magnitude at or below `max_activation` (absolute, on the layer's output;
`0.0` means exactly zero). The units are a layer's output units, so there is one row per
op a module call returned or the model's `forward` returned; the ops inside a composite
activation (the `tanh` of a tanh-approximated GELU, the `sigmoid` of SiLU) get no row,
because a saturated intermediate is not a dead output unit. One trace is only one sample;
aggregate runs before making a dataset-level claim. With `dim=None` the unit axis is
picked per op: the last axis of a 2-D or 3-D activation (a linear layer's
`(batch, features)`, a transformer block's `(batch, tokens, features)`), the channel axis
of a convolution, pooling or per-channel norm and of the same-shaped ops it feeds, and
axis 1 for 4-D and wider. An int `dim` forces one axis on every op.
Related: [`grad_health`](#grad_health).

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.ReLU(), torch.tensor([[-1.0, 2.0], [-3.0, 4.0]]))
report = tl.debug.dead_neurons(trace).to_pandas()
print(report.columns.tolist())
```

Output:

```text
['op', 'module', 'total_units', 'dead_count', 'dead_frac', 'sample_dead_idx', 'reason', 'source_line']
```

## `dtype_range`

`tl.debug.dtype_range(trace, *, max_fraction=0.9, subnormal_fraction_threshold=0.1)`
audits saved activations for numeric-range and precision hazards: non-finite values
(same classifier as [`find_nan`](#find_nan)), finite magnitudes near the saved dtype's
finite maximum, subnormal-heavy tensors, and downcasts from recorded wider input dtypes.
Returns a `DTypeRangeReport` with structured findings and audited/total coverage
(`num_audited_ops` / `num_ops`) — unsaved and non-tensor payloads are never
inspected, and the coverage says so.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.ReLU(), torch.tensor([[-1.0, 2.0]]))
audit = tl.debug.dtype_range(trace)
print(len(audit.findings), audit.num_audited_ops <= audit.num_ops)
```

Output:

```text
0 True
```

## `grad_health`

`tl.debug.grad_health(trace, *, bwd=None, vanishing_threshold=1e-7,
exploding_threshold=1e4, grad_scale=None, grad_scales=None, basis="total_norm")`
ranks saved gradients and flags zero, vanishing, and exploding norms. Capture
with `save_grads=True`, then call `trace.backward(...)` first.

The fix pack added (all additive): `module` + `is_frontier` columns (a
dead layer zeroes the gradient of everything upstream, so the frontier — a
flagged op none of whose children share the flag — names the culprit),
`grad_norm_raw` / `rms` / `numel` columns (`rms = norm/sqrt(numel)` kills the
~128x per-element numel artifact of the absolute threshold), one batched host
sync per device (replacing the historical per-op `.item()`), `stage` +
`scale_provenance` on every row, and per-backward `grad_scales={bwd: scale}`
stamps. Under AMP the saved gradients are SCALED: pass `grad_scale=` (one
factor) or `grad_scales=` (per-backward mapping); unknown provenance yields
`unknown` magnitude verdicts in the `verdict` column, never a silent
pass-as-1.0. `basis="total_norm"` (the default) classifies on each gradient's
total norm; `basis="rms"` classifies on `norm/sqrt(numel)`.

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.options import CaptureOptions

trace = tl.trace(nn.Linear(2, 1), torch.ones(1, 2), capture=CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
report = tl.debug.grad_health(trace).to_pandas()
print(report.shape, report.attrs["bwd"])
```

Output:

```text
(2, 15) 1
```

## `nonfinite_origin`

`tl.debug.nonfinite_origin(trace)` is the one-call NaN forensics door (torchnative
6.5): it consumes the existing capture's queryable nonfinite record FIRST
(free), names the ORIGIN op with module address and file:line, scopes clean
results to CHECKED values, discloses AMP limits, and prints the
`torch.autograd.set_detect_anomaly` recipe verbatim for the cases it does
not cover (backward-born NaNs, custom-autograd exceptions). The live form
`tl.debug.nonfinite_origin(model, inputs)` runs one memory-light tripwire forward. It
NEVER auto-runs anomaly mode — a second execution can mutate state and
follow another stochastic path.

## `flops_vs_dispatch`

`tl.report.flops_vs_dispatch(model, x)` is the PyTorch dispatch-formula
cross-check (never "measured FLOPs"): one detached TorchLens capture, a
state/RNG restore, one raw run under
`torch.utils.flop_counter.FlopCounterMode(display=False)`, an output
witness that refuses on divergence
(`flops_crosscheck_witness_failed`), and a side-by-side report — signed
delta, per-overload native ledger, TorchLens unknown-op ledger. Two
registries disagreeing loudly is the feature; native values never fill
TorchLens unknown cells, and a native ZERO can be a registry-coverage fact
(CPU-default fused attention) rather than a truth verdict.

## `hot_path`

`tl.debug.hot_path(trace, by="flops")` aggregates forward FLOPs, activation memory, or
duration by source line. `by` is one of `"flops"`, `"memory"`, `"duration"`, or
`"device_time"` (joined Kineto device nanoseconds; refuses typed
`device_time_unavailable` without a joined observability
session); use
[`recompute_candidates`](#recompute_candidates) when the question is activation-memory
tradeoffs instead.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
report = tl.debug.hot_path(trace, by="flops").to_pandas()
print(report.columns.tolist(), report.attrs["metric"])
```

Output:

```text
['source_file:line', 'op_count', 'total_cost', 'pct_total'] flops
```

## Pandas-free row core (`hot_path_rows`)

`tl.debug.hot_path_rows(trace, by="flops")` (experimental) is the
pandas-free core of [`hot_path`](#hot_path). It returns `(rows, attrs)`: a list of plain
per-row dicts plus the same `attrs` mapping the report's `.to_pandas()` frame carries
(excluded count, metric, capture-honesty facts), importable and runnable without
pandas. `hot_path` is a thin view over this core, so both spellings always report
identical facts; agent/MCP consumers use the row core directly.

```python
import torch
from torch import nn
import torchlens as tl

trace = tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
rows, attrs = tl.debug.hot_path_rows(trace, by="flops")
print(sorted(rows[0]), attrs["metric"])
```

Output:

```text
['op_count', 'pct_total', 'source_file:line', 'total_cost'] flops
```

## `guess_input_shape`

`tl.debug.guess_input_shape(model, *, batch_size=1, input_dtype=None, channels=None,
spatial_rank="auto", seq_len=None, square=True, min_size=1, max_size=512,
preferred_sizes=(224, 256, 384, 299, 128, 96, 64, 32, 28), max_probes=64, device=None,
seed=0, on_failure="return", input_specs=None)` probes a module and
returns a `GuessInputShapeReport` with a verified synthetic shape or a diagnostic failure.

```python
import torch
from torch import nn
import torchlens as tl

result = tl.debug.guess_input_shape(nn.Linear(3, 2), max_probes=4)
print(result.found, result.shape)
```

Output:

```text
True (1, 3)
```

## `lineage`

`tl.debug.lineage(trace, op_or_label, *, direction="ancestors", max_depth=None)` walks
parents, children, or both from an operation accepted by trace indexing. It returns a
`LineageReport` with labels, depths, source locations, shapes, and dtypes. Pair it with
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
report = tl.debug.recompute_candidates(trace).to_pandas()
print(report.columns.tolist())
```

Output:

```text
['op', 'activation_memory', 'flops_forward', 'mem_per_flop', 'suggested']
```

## `find_nan`

`tl.debug.find_nan(model, inputs, **trace_kwargs)` runs a memory-light live capture and stops at
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

## `bisect_nan_backward`

`tl.debug.bisect_nan_backward(trace, *, bwd=None, grad_scale=None)` (experimental)
walks one captured backward pass's grad-fn fires in MEASURED fire order and
classifies every transition: `clean` / `birth` / `propagated` / `healed` / `root_seed`
(plus `unchecked` for fires `save_grads` did not cover). Healing is measured fact -- a
finite loss gradient does not prove a clean backward -- so the result carries the FULL
transition ledger with the earliest provable birth headlined, joined to the paired
forward op and source line. Multi-backward traces require `bwd=N` (passes never
collapse); `clean` requires complete checkable coverage; an inf-kind first bad thing
without a disclosed `grad_scale` carries the AMP/GradScaler hint. Requires
`CaptureOptions(backward_ready=True, save_grads=...)` plus a logged backward; post-hoc,
no rerun. `result.first_bad_thing` projects into the shared `FirstBadThing` vocabulary.

## `check_determinism`

`tl.debug.check_determinism(model, inputs, *, runs=2, seed=0, rtol=None, atol=None,
seed_sensitivity=False)` (experimental) answers ONE controlled question:
do N isolated same-seed runs agree? Every run executes on a fresh released deep copy
with cloned inputs under preserved, identically-seeded RNG. The verdict is three-state
(`repeatable_under_test` / `nondeterminism_observed` / `inconclusive`) and never
generalizes to "deterministic". The report carries environment witnesses (deterministic
algorithms, cuDNN, `CUBLAS_WORKSPACE_CONFIG`, thread counts), the first divergence
in EXECUTION order, RNG-consuming ops attributed to the CONSUMING op, and host-entropy
facts quoted from the capture. `seed_sensitivity=True` adds exactly ONE differently
seeded run (total cost disclosed as runs+1) landing in a SECOND named verdict slot.
Deepcopy-refusing models return `inconclusive` with a `model_factory` remedy.

## `first_divergence`

`tl.debug.first_divergence(trace_a, trace_b, *, rtol=None, atol=None, align="auto", module_map=None)`
returns the first EXECUTION-ORDER row where two traces separate (value, shape, or structural), or `None`.
It is the direct spelling of the first-diverging-op recipe over [`compare`](#compare),
whose rows follow execution order -- never alphabetical -- and pairs operations the same
way. Operations found only in `trace_b` come after every paired row, so with one inserted
operation and a real value change later, the changed operation is returned; with the
inserted operation alone, the inserted operation is.

## Comparing runs from different backends

[`compare`](#compare) and [`first_divergence`](#first_divergence) choose how to pair two runs with
`align=`. On one backend the default (`align="auto"`) pairs operations by label or graph structure, as above. Across
backends (`trace.provenance.backend` differs, for example a PyTorch trace and an MLX trace of the
same model) operation labels and order do not correspond, so `"auto"` pairs **module calls**
instead (`align="structure"` asks for the same on any pair). `align="label"` across backends is
refused with `tl.errors.BackendUnsupportedError` (code `backend_unsupported`) before any value is
read.

**How module calls pair.** By address and call index, never by value: the same address on both
sides; the same address after dropping a container that is never called (an MLX list `layers.0`
meets a torch `Sequential` child `0`, a torch `ModuleList` `model.layers.0` meets MLX
`model.layers.0`); repeated calls in observed order under the same parent call. A port that
renamed modules passes `module_map={"blocks": "layers"}` (side-a prefix to side-b prefix); those
pairs are marked `asserted`. A call with no partner (a torch-only `lm_head` whose MLX twin is
computed inside the root, an MLX-only `rope`) is a coverage gap, never a divergence.

**What each row says.** Module rows add `module_b` (the partner call), `alignment` (`supported`,
`asserted`, `ambiguous`, `absent`), `outcome` (`equal`, `different`, `shape-incompatible`,
`unavailable`, `not-evaluated`), the effective `rtol` and `atol`, any `transform` applied, and
`inputs_agree`. `different` appears only on a `supported` or `asserted` row. The frame's `attrs`
add `coverage` (per side: `paired`, `unpaired`, `boundary_unrecorded`, `unresolved`), `align`
(the pairing mode that ran, `"structure"`) and `backends` (for example `("torch", "mlx")`);
op-pairing frames on one backend carry neither `align` nor `backends`. A module that returns several
values pairs them by return position, never by the order its ops ran. One value against several
(an eager attention returning its output and its weights, against a port returning only the output)
meets the first returned value, and `reason` says how many more were not compared; when a side's
return order is not recorded, or both sides return several values in trees of different sizes, the
row is `unavailable`. Rows follow side a's
module-exit order, so an inner module comes before the module that returns its value, and a
module whose output agrees again later never hides an earlier difference.

**Tolerance.** Values compare at the wider dtype. Across backends a module output agrees when its
relative RMS error is within tolerance: `rms(a - b) <= atol + rtol * rms(a)`, with `trace_a` as the
reference. The default comes from the narrower dtype of each pair: float32 `1e-4`, float16
`5e-2`, bfloat16 `0.75`, and `atol` is 0. Only when a module's reference output is all zero, where
the relative bound would demand exact zeros, `atol` becomes `c * eps(narrower dtype)` times the
smallest rms among that module's finite, non-zero floating inputs on side a (1.0 when there is
none), never the output being judged; taking the smallest keeps a causal mask from widening it.
`c` is per dtype (float32 `500`, float16 `10`, bfloat16 `20`; float64 `500`, uncalibrated),
measured on one small model with three seeds: a correct float32 LayerNorm over constant rows
needs `c` of at least `163`, a smaller LayerNorm eps would need more, and in bfloat16 no `c`
separates small bugs on all-zero outputs. The `rtol` constants were measured on SmolLM2 (mlx-lm
against the Hugging Face PyTorch model): in float32 they catch one-percent-sized bugs (an attention scale, a RoPE base, a
swapped activation) at their own module. Half precision catches only gross bugs; rows in float16
or bfloat16 say so, and the advice is to compare float32 runs of both sides. Passing `rtol=` or
`atol=` replaces the constant in the same form. Two runs on one backend keep `torch.allclose` with
`rtol=1e-5, atol=1e-8`.

**Shapes.** A leading size-one axis present on one side only (a batch of one against no batch
axis) is dropped and shown in `transform`; any other difference in the model inputs' shapes
refuses. Convolution, pooling and channel-normalization modules may be channels-last on one side;
that transform applies only when it is the one listed transform that fits the shapes. When equal
channel and spatial sizes make more than one fit, the row is `not-evaluated` with
`layout_unresolved`, never resolved by picking whichever transform makes the values agree.

**The answer.** `first_divergence` returns the first `different` row. Its `reason` starts with
"localized inside this module" when the module's inputs agree, or "first observed differing
module output" when they already differ, followed by the computation skipped since the last
agreeing module. It returns `None` only when every paired module agrees, and refuses with
`InvalidArgumentError` (code `module_comparison_inconclusive`) when a side records no module
boundaries, the model inputs differ, nothing could be compared, or a paired module could not be
decided. `compare` returns every row; its coverage line counts module calls on both sides
(paired, unpaired, boundary unrecorded, unresolved) and says when either run trained or a random
producer ran. A saved and reloaded pair gives the same answer with no live model.

```text
>>> row = tl.debug.first_divergence(torch_trace, mlx_trace)
>>> row["op"], row["module_b"], row["alignment"], row["outcome"]
('model.layers.12.self_attn:1', 'model.layers.12.self_attn:1', 'supported', 'different')
>>> row["reason"]
'localized inside this module (inputs agree, output differs)'
>>> print(tl.debug.compare(torch_trace, mlx_trace, module_map={"blocks": "layers"}))
```

**Which backends.** A backend joins module pairing once its recorded module outputs pass
TorchLens's conformance checks; PyTorch, MLX, Paddle, TensorFlow and JAX (Equinox and Flax NNX
modules) are admitted. Any other backend, tinygrad included, is refused with
`backend_unsupported`, and the message names the admitted backends. JAX module calls made inside
`jax.vmap` or another transform are not recorded, so compare an unbatched JAX call. A module call that returns
its input unchanged (an identity, or dropout in eval mode) is compared through the returned
value: PyTorch logs an identity op inside the call, the other admitted backends record the
returned input's op, and both read as that call's output. A call that returns two or more of its inputs unchanged reads as
`boundary unrecorded`, because the order it returned them in is not recorded. Comparing works on the machine that captured both traces;
nothing here promises reading one backend's saved trace on a machine without that backend.

## `isolated_capture` / `preserved_rng_state` / `clone_input_tree`

The shared isolated-rerun harness (experimental) behind
`bisect_precision` and `check_determinism`: `isolated_capture(model, inputs, *,
seed, prepare=None, **trace_kwargs)` runs one seeded capture of a fresh deep copy
that is RELEASED from inherited TorchLens preparation first (the deepcopy-after-trace
`KeyError` repair), with cloned inputs, under `preserved_rng_state()` (Python, NumPy,
torch CPU, initialized CUDA generators saved and restored). `clone_input_tree` is the
input-cloning primitive. The caller's model, inputs, and RNG streams are untouched.

## `walk_grad_fn` / `sketch_grad_fn`

The post-hoc autograd-graph walker (experimental): draw the
`grad_fn` graph of a tensor you ALREADY HAVE — computed in an earlier notebook cell or
inside someone else's training loop — with no re-execution and no capture context (the
`make_dot(y)`-class torchviz capability). Both are explicitly STRUCTURE-ONLY: the walk
reads `tensor.grad_fn` / `next_functions` only (no values, no timing, no verification)
and every render carries that legend.

`tl.debug.walk_grad_fn(outputs, *, params=None, model=None, max_nodes=...)` walks the
graph into a frozen `GradFnSketch` (nodes, forward-direction edges, truncation flag).
`outputs` is a tensor with a `grad_fn` or a sequence of them; `params` is the torchviz
`name -> tensor` leaf-naming convention and `model` provides leaf names via
`named_parameters()` (explicit `params` wins on collisions). The defensive
`max_nodes` ceiling (20,000) sets `truncated` and is disclosed on renders; a tensor
with no walkable graph raises the typed `GradFnWalkError`.

`tl.debug.sketch_grad_fn(outputs, path=None, *, file_format="png", params=None,
model=None, max_nodes=..., view=False)` renders the same sketch through the bounded
TorchLens graphviz runner (a wedged `dot` cannot hang the caller; renders are
time-bounded and atomically published) and returns the DOT source. `path` follows the
torchviz/graphviz convention (`path.<file_format>` is written; `None` skips file
rendering), and `view=True` opens the managed viewer, never a raw `Popen`.

## `amp_scaled_grads_hint`

`tl.debug.amp_scaled_grads_hint(all_nonfinite=False)` returns the ONE shared
AMP/GradScaler disclosure text used by the backward diagnostics: gradients captured
under `torch.amp.GradScaler` carry the loss scale (~2**16 at the default `init_scale`),
so magnitude verdicts accept `grad_scale=scaler.get_scale()` and non-finite verdicts
name fp16 scale overflow as a possibility instead of false-positive certainty.
