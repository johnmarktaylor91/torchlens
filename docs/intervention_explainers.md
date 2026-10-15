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

## Subspace edits: the component along a basis

`do(tl.subspace(site, basis, origin=...), edit)` edits the component of the site's value inside
the span of the basis rows and leaves the orthogonal complement unchanged. With `Q` the basis
rows, the edit acts on the coordinates `c = Q x` (one per row) and the site gets
`x + Q^T (edit(c) - c)`. This is the addressing distributed alignment search uses.

- Coordinate verbs act on `c`: `tl.zero_ablate()`, `tl.scale(...)`, `tl.clamp(...)`,
  `tl.noise(...)`, `tl.mean_ablate()`, `mean_fill(over=...)` without `ref=`,
  `scramble_elements(seed=...)` without `source=`, `tl.add(scalar)`,
  `splice_module(module, input="out")`, and any hook callable `fn(c, *, hook)`, which receives
  the coordinates with the bound axis resized to `k`.
- Value verbs compute a value in activation space and write its component: `tl.patch_from(source)`
  is the interchange (the component becomes the source run's), and likewise `swap_with`,
  `replace_with`, `tl.add(tensor)`, `steer`, `permute_batch`, `resample_rows_from`, `mean_from` and
  `mean_fill(ref, ...)`.
- A source verb maps the whole site value to new coordinates: `tl.intervention.affine_source(W,
  bias=b)` sets them to `W x + b` (LoReFT; see "A learned affine source" below).
- Every other verb refuses `subspace_edit_unsupported` and names the spelling that works: the
  projection edits themselves (`project_off`, `project_onto`, `set_direction_mean`,
  `subspace_ablate`, `subspace_interchange`), `compose` (apply the edits one at a time),
  `at_positions` (restrict positions through `within=`), a `source=` that is an activation value,
  `splice_module(input="in")` and the backward verbs.

An orthonormal basis is used exactly as given, so the coordinates are yours. Any other full-rank
basis is orthonormalised by Gram-Schmidt in row order with each row's sign kept, so one direction
`v` edits the component along `v / |v|`; the audit record says when that happened. A
rank-deficient basis refuses `subspace_basis_invalid`. Weights at or below `tol=` are dropped
first.

`resolve()` still answers a set question: the basis's support set, the elements a nonzero weight
touches, which is where the edit can land (a dense direction supports the whole axis). Restrict
positions with `within=`, for example `tl.subspace(tl.units(site, mask), basis, origin=...)` with
`mask` True on every feature of the chosen positions; a mask that covers only part of the support
at a position refuses `subspace_projection_partial_axis`. A subspace combined with `| & - ~`, or
already resolved, refuses `subspace_projection_composed`; to edit the support set itself, pass
`tl.units(site, resolved[0].mask)`. Every subspace `do()` stamps the projection, the basis origin
and its sha256 into the audit record.

```python
# docs-test: expected=[True, True, True]
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
options = tl.options.CaptureOptions(intervention_ready=True)
base = tl.trace(model, torch.randn(3, 4), capture=options)
source = tl.trace(model, torch.randn(3, 4), capture=options)
site = "linear_1_1"
basis = torch.tensor([[1.0, 1.0, 0.0, 0.0]]) / 2**0.5  # one unit direction

fork = base.fork()
fork.do(tl.subspace(site, basis, origin="hand-picked direction"), tl.patch_from(source))
x, s, edited = base[site].out, source[site].out, fork[site].out
projector = basis.T @ basis

RESULT = [
    # The component along the direction is the source run's (the interchange) ...
    torch.allclose(edited @ projector, s @ projector, atol=1e-6),
    # ... and the orthogonal complement is the base run's.
    torch.allclose(edited - edited @ projector, x - x @ projector, atol=1e-6),
    "projection" in fork.intervention_audit[-1]["sites"][0]["source"],
]
```

For a fixed direction without a selection, `tl.project_off(direction, feature_axis=...)` removes
the component and `tl.project_onto(direction, feature_axis=...)` keeps only it:

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

### A rotation learned elsewhere

TorchLens applies a rotation you trained elsewhere (distributed alignment search, ReFT); it does
not fit one. The first `k` rows of the learned rotation `R` are the basis, so
`do(tl.subspace(site, R[:k], origin=...), edit)` edits the learned subspace. With a builtin verb
the staged edit is the builtin `subspace_edit`, which stores the verb, the float64 orthonormal rows
it applies, the bound axis, the rows' sha256 and the basis provenance (`origin`, `method`, the
original basis digest and its dtype). `tl.io.save_intervention(trace, path, level="portable")`
saves it, and the loaded spec rebuilds the same projection, never the plain verb over the whole
site. A stored basis that is malformed refuses `subspace_spec_invalid` at load, rows changed after
saving refuse `subspace_spec_basis_tampered`, and rows that are not orthonormal refuse
`subspace_spec_basis_not_orthonormal`. The portable interchange is `tl.replace_with(value)` with
the source run's value at the site; `tl.patch_from(source)` and hook callables are session-time
and save only at the audit level.

```python
# docs-test: expected=[True, True, True]
import tempfile
from pathlib import Path

import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
options = tl.options.CaptureOptions(intervention_ready=True)
x, x_source = torch.randn(3, 4), torch.randn(3, 4)
site = "linear_1_1"
rotation = torch.linalg.qr(torch.randn(4, 4))[0].T  # stands in for a learned rotation
source_value = tl.trace(model, x_source, capture=options)[site].out

fork = tl.trace(model, x, capture=options).fork()
fork.do(
    tl.subspace(site, rotation[:2], origin="DAS rotation trained elsewhere", method="das"),
    tl.replace_with(source_value),
)
path = Path(tempfile.mkdtemp()) / "rotation.tlspec"
tl.io.save_intervention(fork, path, level="portable")

spec = tl.io.load_intervention_spec(path)
fresh = tl.trace(model, x, capture=options).fork()
for hook_spec in spec.hook_specs:
    fresh.attach_hooks(hook_spec.site_target, hook_spec.helper, confirm_mutation=True)
reloaded = fresh.push()
helper = spec.hook_specs[0].helper

RESULT = [
    torch.equal(reloaded[site].out, fork[site].out),  # the same interchange, bit for bit
    helper.helper_name == "subspace_edit",
    dict(helper.kwargs)["origin"] == "DAS rotation trained elsewhere",
]
```

### A learned affine source: LoReFT

LoReFT (pyreft) learns a rotation `R` (`[d, k]`, orthonormal columns) and an affine map `W h + b`
(`W` is `[k, d]`), and writes `h + R (W h + b - R^T h)`: the coordinates along `R` become an affine
function of the whole site value `h`. That is a source verb:
`tl.intervention.affine_source(W, bias=b)` under `tl.subspace(site, R.T, origin=...)`. The
projection hands it the site value and it returns the new `k` coordinates, so the site gets
`x + Q^T (W x + b - Q x)` with `Q = R^T`. The basis rows and `W` must have the same shape `[k, d]`
(else `affine_source_invalid`, as for a non-finite or non-float argument). Outside a subspace a
`[k, d]` map has no meaning, so the verb refuses `affine_source_needs_subspace` with this spelling.

```python
# docs-test: expected=[True, True]
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).double().eval()
options = tl.options.CaptureOptions(intervention_ready=True)
x = torch.randn(3, 4, dtype=torch.float64)
site = "linear_1_1"
rotation = torch.linalg.qr(torch.randn(4, 2, dtype=torch.float64))[0]  # R, [d, k]
weight, bias = torch.randn(2, 4, dtype=torch.float64), torch.randn(2, dtype=torch.float64)

base = tl.trace(model, x, capture=options)
fork = base.fork()
fork.do(
    tl.subspace(site, rotation.T, origin="LoReFT trained elsewhere", method="loreft"),
    tl.intervention.affine_source(weight, bias=bias),
)
h = base[site].out
loreft = h + (h @ weight.T + bias - h @ rotation) @ rotation.T

RESULT = [
    torch.allclose(fork.push()[site].out, loreft, rtol=0.0, atol=1e-12),  # LoReFT, by hand
    fork.intervention_audit[-1]["edit"] == "affine_source",
]
```

LoReFT edits one position (pyreft's `unit_locations`, usually the last prompt token). Restrict the
subspace with `within=tl.units(site, mask)`, `mask` True on every feature of that position. A
subspace edit with a builtin verb stores its mask and the mask's sha256, so a one-position edit
still saves at `level="portable"` and reloads as the same edit; a stored mask that is malformed
refuses `subspace_spec_invalid` at load and one changed after saving refuses
`subspace_spec_mask_tampered`. Other masked edits keep their own portability.
`tl.bridge.pyreft.read(directory)` reads a saved pyreft artifact into these two pieces
(`artifact.subspace(site, positions=...)` and `artifact.edit`); see
[from pyvene](migration/from_pyvene.md#reading-a-pyreft-loreft-artifact).

```python
# docs-test: expected=[True, True, True]
import tempfile
from pathlib import Path

import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).double().eval()
options = tl.options.CaptureOptions(intervention_ready=True)
tokens = torch.randn(1, 5, 4, dtype=torch.float64)  # [batch, positions, features]
site = "linear_1_1"
rotation = torch.linalg.qr(torch.randn(4, 2, dtype=torch.float64))[0]
weight, bias = torch.randn(2, 4, dtype=torch.float64), torch.randn(2, dtype=torch.float64)
last = torch.zeros(1, 5, 4, dtype=torch.bool)
last[:, -1, :] = True  # every feature of the last position

base = tl.trace(model, tokens, capture=options)
fork = base.fork()
fork.do(
    tl.subspace(tl.units(site, last), rotation.T, origin="LoReFT trained elsewhere"),
    tl.intervention.affine_source(weight, bias=bias),
)
edited = fork.push()
path = Path(tempfile.mkdtemp()) / "loreft.tlspec"
tl.io.save_intervention(fork, path, level="portable")

spec = tl.io.load_intervention_spec(path)
fresh = tl.trace(model, tokens, capture=options).fork()
for hook_spec in spec.hook_specs:
    fresh.attach_hooks(hook_spec.site_target, hook_spec.helper, confirm_mutation=True)

RESULT = [
    torch.equal(fresh.push()[site].out, edited[site].out),  # the same edit, bit for bit
    torch.equal(edited[site].out[:, :-1], base[site].out[:, :-1]),  # other positions untouched
    spec.hook_specs[0].helper.portability == "builtin",
]
```

Which verbs keep the autograd graph, under which door, is the
[gradient contract](gradient_contract.md).

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
