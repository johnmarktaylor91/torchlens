# Intervention Explainers

## Which engine when

TorchLens can apply the same edit (`tl.when(site, action)`) in three ways. They answer different
questions, so pick by what you need back, not by habit.

| You want to | Use | Runs `model.forward`? | What you get back |
| --- | --- | --- | --- |
| See every op of one edited run, with a record of where the edit fired | `tl.trace(model, x, intervene=spec)` | yes, once | a `Trace` of the edited run |
| Ask "what if this saved value had been different" on an input you already traced | `trace.fork().do(site, action)` | no: the edit is pushed down the recorded graph from saved values | the fork, with its saved values updated |
| The same question when the edit may change control flow, randomness or module state | `trace.fork().do(site, action, model=model, x=x)`, or `trace.run(model, x)` | yes | the fork, re-executed under the edit |
| Apply one edit to many forwards or to HF `generate()`, with no trace | `spec.bind(model)`, or `torchlens.intervention.steer_generate` | yes, every call | the model's own outputs, plus `last_report` fire counts |
| Apply an edit and keep a few activations on every forward | `tl.record(model, x, save=..., intervene=spec, return_output=True)` | yes, every call | the output and a sparse `Recording` |

`do()` picks its engine from its arguments: with neither `model=` nor `x=` it pushes over the saved
graph; with both it re-executes the model. Pass
`intervention=tl.options.InterventionOptions(engine=...)` to name it explicitly. What each path
costs, and what each one refuses, is in
[Performance: choosing a fast path](guides/fast_paths.md).

The four engines agree on a graph-stable edit:

```python
# docs-test: expected=[True, True, True, True]
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
x = torch.randn(3, 4)
site = tl.module("1")  # the ReLU's output
spec = tl.when(site, tl.zero_ablate())
head = tl.module("2")

# intervene=: edit during a fresh capture; the trace records the edited run.
live = tl.trace(model, x, save=head, intervene=spec)
live_out = live.find_sites(head).first().out

# do() with no model: edit a saved value and push it down the recorded graph.
saved = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
pushed = saved.fork("pushed")
pushed.do(site, tl.zero_ablate())
pushed_out = pushed.find_sites(head).first().out

# do() with model and input: re-execute the model under the edit.
executed = saved.fork("executed")
executed.do(site, tl.zero_ablate(), model=model, x=x)
executed_out = executed.find_sites(head).first().out

# bind: no trace at all; the edit runs as a hook on every call.
bound_out = spec.bind(model)(x)

head_layer = model[2]
expected = head_layer(torch.zeros(3, 8))  # the head sees an all-zero ReLU output
RESULT = [
    torch.equal(live_out, expected),
    torch.equal(pushed_out, expected),
    torch.equal(executed_out, expected),
    torch.equal(bound_out, expected),
]
```

They stop agreeing when the edit changes what the forward does. A push recomputes the downstream
ops exactly as they were recorded, so a branch the edit would have flipped, a dropout mask, or a
module that reads its own state takes the recorded path, not the new one. Use the re-executing
form whenever that can happen.

## Push vs run

`trace.push()` walks the saved TorchLens graph and recomputes the affected downstream cone without
calling `model.forward` again. It is fast and right for graph-stable interventions, where the saved
op metadata is enough to recompute the children.

`trace.run(model, x)` re-executes the model under the trace's active intervention spec. Use it when
the intervention may affect Python control flow, module behavior, randomness, or anything else the
saved graph cannot reconstruct. A module-targeted spec reruns through a guarded fast engine and
falls back to the capture engine when a guard refuses; `trace.last_run` says which engine ran.

The older `replay()` and `rerun()` spellings are removed; `push()` and `run()` replace them.

## Mutate In Place and Fork First

`set`, `attach_hooks`, and `do` mutate the `Trace` they are called on. A root trace warns once
before this kind of mutation. For branched experiments, fork first:

```python
# docs-test: expected=True
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)
clean = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
before = clean.find_sites(tl.module("2")).first().out.clone()

candidate = clean.fork("candidate")
candidate.attach_hooks(tl.func("relu"), tl.zero_ablate())
candidate.push()

# The fork changed; the trace it came from did not.
RESULT = torch.equal(clean.find_sites(tl.module("2")).first().out, before)
```

Forked traces have independent intervention specs and intervention logs while sharing immutable
captured evidence where safe.

## Direct-Write Dirty

Writing directly to a saved value, for example `trace[label].out = value`, bypasses the
intervention recipe. TorchLens marks the owning trace `DIRECT_WRITE_DIRTY` and warns once. A later
push overlays recipe state; it does not treat the direct write as a portable, auditable
intervention. Use `set`, `attach_hooks` or `do` when the edit should be part of the recipe.

## Set, not projection

A `tl.subspace(...)` selection is a set of elements: `do(subspace(...), edit)` edits every element
the direction has weight on, and a dense direction covers the whole axis. It never edits "the
component along the direction". When you mean the component, use a helper that projects:

- `tl.project_off(direction, feature_axis=...)` removes the component along `direction`.
- `tl.project_onto(direction, feature_axis=...)` keeps only that component.
- `torchlens.intervention.set_direction_mean(direction, population, feature_axis=...)` replaces
  the component with the population's mean coefficient, keeping the orthogonal part.

```python
# docs-test: expected=[True, True]
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
x = torch.randn(2, 4)
direction = torch.tensor([1.0, 1.0, 0.0, 0.0])
unit = direction / direction.norm()
site = tl.module("0")

clean = tl.trace(model, x, save=site).find_sites(site).first().out
off = tl.trace(
    model, x, save=site, intervene=tl.when(site, tl.project_off(direction, feature_axis=-1))
).find_sites(site).first().out

# project_off leaves no component along the direction and keeps the rest.
RESULT = [
    torch.allclose(off @ unit, torch.zeros(2), atol=1e-6),
    torch.allclose(off, clean - (clean @ unit)[:, None] * unit, atol=1e-6),
]
```

The population-based edits, `set_direction_mean` among them, are on the
[stochastic and population edits](stochastic_edits.md) page.

## Save Levels

`save_intervention(path, level=...)` writes a `.tlspec/` directory.

| Level | Intended use | Executable? |
| --- | --- | --- |
| `audit` | Record what was attempted, including non-portable pieces. | Not guaranteed. |
| `executable_with_callables` | Re-run in the same code environment when importable callables resolve. | Yes, unless opaque callables block execution. |
| `portable` | Share built-in-helper recipes and tensor sidecars with no opaque local callable dependency. | Yes for supported built-ins. |

Portable saves reject opaque local callables. Use built-in helpers such as `zero_ablate`,
`steer`, `scale` or `project_off` for reproducible publication.

## `.tlspec/` Directory

A `.tlspec/` directory contains:

- `spec.json`: serialized selector, hook, helper, function-registry, append, and target-manifest
  metadata.
- `manifest.json`: tensor sidecar manifest with hashes and storage metadata.
- `tensors/`: tensor payloads stored separately from JSON.
- `README.md`: a generated human-readable summary.

Load with `torchlens.io.load_intervention_spec(path)` and check a fresh capture with
`torchlens.validation.check_spec_compat(spec, new_trace)`.

## Append Constraints

Append mode is for memory-constrained evaluation over compatible chunks:

```text
trace.run(model, first_chunk)
trace.run(model, next_chunk, replay=tl.options.ReplayOptions(append=True))
```

The appended run must match the original graph shape, labels, dtypes, and non-batch dimensions.
Helpers must be batch-independent and append-compatible. Batch-dependent helpers such as
resampling from the current batch are rejected. Training-mode batch normalization warns because
cross-batch statistics can make chunked evaluation semantically different from one full batch.

## Tier-1 Backward Hook

Backward helpers (`bwd_hook`, `grad_zero`, `grad_scale`) are live/run-only in this release. They
are not portable `.tlspec/` publication units and are not pushed post hoc over a saved forward
graph. Use them when the experiment explicitly re-executes the model and backward pass.

## Attribution-Patching Note

TorchLens exposes the ingredients for attribution-style patching: capture clean and corrupted
traces, choose a target site, attach a hook, and compute a metric over a `Bundle`. Turnkey
attribution-patching row and formula helpers are not shipped; keep those formulas in analysis code
and record the exact selectors and metric definitions next to saved specs.

Two sign conventions are in use. The TransformerLens presentation is the first-order estimate
`grad * (clean - corrupt)`, with `grad` taken on the corrupted run. The
[intervention API reference](intervention_api.md#attribution-patching) examples use
`(corrupt - clean) * patched`. Both are valid; name the convention next to the metric so higher
and lower scores stay clear.
