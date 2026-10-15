# Gradient contract

Every edit either keeps the autograd graph through the values it writes or cuts it. An edit that
keeps the graph lets the gradient of a loss on the output reach the parameters upstream of the
edited site; an edit that cuts it gives those parameters no gradient through the edited elements.
Cutting is correct for analysis (a patched value is evidence from another run, not a function of
this one) and silent for training, so a caller who needs the gradient says so with
`torchlens.intervention.keep_grad(edit)`.

## The rules

- Verbs whose result is a function of the site value keep the graph: `scale`, `clamp`, `noise`,
  `steer`, `add`, `project_onto`, `project_off`, `mean_ablate()` and `mean_fill(over=...)` (the
  gradient flows through the mean), `scramble_elements(seed=...)` without `source=`,
  `splice_module`, `permute_batch`, `set_direction_mean` (through the orthogonal part), the
  `subspace_*` edits (through the complement), `compose` and `at_positions` of keeping verbs, and
  hook callables that compute from their input.
- Verbs that write a value computed elsewhere cut it: `zero_ablate`, `replace_with`, `swap_with`,
  `patch_from` (the donor is detached by design), `resample_rows_from`, `mean_from`,
  `mean_fill(ref, ...)`, and `mean_ablate` or `scramble_elements` with `source=`.
- A selection door edits only the selected elements, so the rest of the site keeps the graph:
  under `do(tl.units(...))` a cutting verb still lets the gradient through the unselected
  elements, and under `do(tl.subspace(...))` through the orthogonal complement.
- The capture doors, `fork.do` (replay), `do(model=, x=)` (rerun), `attach_hooks` + `run` and
  `spec.bind(model)` keep whatever the verb keeps. `push(replay=ReplayOptions(differentiable=True))`
  re-leafs the replayed values by design, so no gradient reaches the parameters upstream of the
  edit under any verb; the gradient stops at the replay frontier. To train parameters upstream of
  an edit, use a live door (`tl.when(site, edit).bind(model)(x)` or `tl.trace(model, x,
  intervene=...)`).

## Asking for the gradient: `keep_grad`

`torchlens.intervention.keep_grad(edit)` returns the same edit with a fire-time check. When the
values the edit writes carry no autograd graph while the site value had one, it raises
`InvalidArgumentError` with code `edit_detached_from_graph`, the site label and a spelling that
keeps the graph. It also raises when the site value has no graph to keep (`reason`
`site_has_no_graph`) or when gradients are disabled (`grad_disabled`).
`InterventionDecision(keep_grad=True)` runs the same check; `keep_grad=False`, the default,
checks nothing. The check is on the edit, not the door: under differentiable replay a keeping
verb passes it and still delivers no gradient upstream (see the matrix), so there `keep_grad`
warns once per session with code `keep_grad_replay_frontier`: the edit stays connected inside the
replayed computation, parameters upstream of the replay frontier get no gradient, and the live
door `tl.when(site, keep_grad(edit)).bind(model)(x)` reaches them.

The graph-keeping spellings:

- a zero: `keep_grad(tl.scale(0.0))` writes the same zeros as `tl.zero_ablate()` with a zero
  gradient;
- a fixed value: `keep_grad(tl.intervention.compose(tl.scale(0.0), tl.add(value)))` writes `value`
  with a zero gradient;
- an interchange that trains: `do(tl.subspace(site, basis, origin=...), tl.patch_from(source))`
  keeps the gradient through the orthogonal complement, while the written component is the
  source's (detached).

```python
# docs-test: expected=[True, True, True]
import torch
from torch import nn
import torchlens as tl
from torchlens.intervention import keep_grad

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)).eval()
x = torch.randn(3, 4)
site = tl.module("0")

try:
    tl.trace(model, x, intervene=tl.when(site, keep_grad(tl.zero_ablate())))
    refused = False
except ValueError as error:
    refused = error.fields["code"] == "edit_detached_from_graph"

trace = tl.trace(model, x, intervene=tl.when(site, keep_grad(tl.scale(0.0))))
out = trace[trace.output_layers[0]].out
out.sum().backward()

RESULT = [
    refused,
    out.requires_grad,
    bool((model[0].weight.grad == 0).all()),
]
```

## The matrix

One row per door and verb; the two columns are the outcome without and with `keep_grad`. Each
cell is checked by `tests/test_gradient_contract_matrix.py` on a float64 two-layer model with the
edit at the first layer's output: `reaches` means `output.sum().backward()` gives the first
layer's weight a gradient, equal to the gradient of the same edit applied by hand in a plain torch
forward hook; `cut` means it gives none. `raises:<code>` is a typed refusal. Under
`do(tl.subspace(...))` the verbs act on the subspace coordinates or write a value's component (see
[subspace edits](intervention_explainers.md#subspace-edits-the-component-along-a-basis)), and the
projection edits refuse `subspace_edit_unsupported` there. Every `push(differentiable=True)` cell
reads `cut` because the replay frontier re-leafs the edited value; each of its `keep_grad` cells
that does not raise also carries the `keep_grad_replay_frontier` warning, and the test checks it.

| Door | Verb | `keep_grad` off | `keep_grad` on |
| --- | --- | --- | --- |
| capture intervene= | `zero_ablate` | cut | raises:edit_detached_from_graph |
| capture intervene= | `mean_ablate` | reaches | reaches |
| capture intervene= | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| capture intervene= | `scale` | reaches | reaches |
| capture intervene= | `clamp` | reaches | reaches |
| capture intervene= | `noise` | reaches | reaches |
| capture intervene= | `steer` | reaches | reaches |
| capture intervene= | `add(scalar)` | reaches | reaches |
| capture intervene= | `add(tensor)` | reaches | reaches |
| capture intervene= | `replace_with` | cut | raises:edit_detached_from_graph |
| capture intervene= | `swap_with` | cut | raises:edit_detached_from_graph |
| capture intervene= | `project_onto` | reaches | reaches |
| capture intervene= | `project_off` | reaches | reaches |
| capture intervene= | `scramble_elements` | reaches | reaches |
| capture intervene= | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| capture intervene= | `splice_module` | reaches | reaches |
| capture intervene= | `patch_from` | cut | raises:edit_detached_from_graph |
| capture intervene= | `permute_batch` | reaches | reaches |
| capture intervene= | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| capture intervene= | `mean_from` | cut | raises:edit_detached_from_graph |
| capture intervene= | `mean_fill` | reaches | reaches |
| capture intervene= | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| capture intervene= | `set_direction_mean` | reaches | reaches |
| capture intervene= | `subspace_ablate` | reaches | reaches |
| capture intervene= | `subspace_ablate(resample)` | reaches | reaches |
| capture intervene= | `subspace_interchange` | reaches | reaches |
| capture intervene= | `compose` | reaches | reaches |
| capture intervene= | `at_positions` | reaches | reaches |
| capture intervene= | `hook callable` | reaches | reaches |
| capture hooks= | `zero_ablate` | cut | raises:edit_detached_from_graph |
| capture hooks= | `mean_ablate` | reaches | reaches |
| capture hooks= | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| capture hooks= | `scale` | reaches | reaches |
| capture hooks= | `clamp` | reaches | reaches |
| capture hooks= | `noise` | reaches | reaches |
| capture hooks= | `steer` | reaches | reaches |
| capture hooks= | `add(scalar)` | reaches | reaches |
| capture hooks= | `add(tensor)` | reaches | reaches |
| capture hooks= | `replace_with` | cut | raises:edit_detached_from_graph |
| capture hooks= | `swap_with` | cut | raises:edit_detached_from_graph |
| capture hooks= | `project_onto` | reaches | reaches |
| capture hooks= | `project_off` | reaches | reaches |
| capture hooks= | `scramble_elements` | reaches | reaches |
| capture hooks= | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| capture hooks= | `splice_module` | reaches | reaches |
| capture hooks= | `patch_from` | cut | raises:edit_detached_from_graph |
| capture hooks= | `permute_batch` | reaches | reaches |
| capture hooks= | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| capture hooks= | `mean_from` | cut | raises:edit_detached_from_graph |
| capture hooks= | `mean_fill` | reaches | reaches |
| capture hooks= | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| capture hooks= | `set_direction_mean` | reaches | reaches |
| capture hooks= | `subspace_ablate` | reaches | reaches |
| capture hooks= | `subspace_ablate(resample)` | reaches | reaches |
| capture hooks= | `subspace_interchange` | reaches | reaches |
| capture hooks= | `compose` | reaches | reaches |
| capture hooks= | `at_positions` | reaches | reaches |
| capture hooks= | `hook callable` | reaches | reaches |
| capture InterventionDecision | `zero_ablate` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `mean_ablate` | reaches | reaches |
| capture InterventionDecision | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `scale` | reaches | reaches |
| capture InterventionDecision | `clamp` | reaches | reaches |
| capture InterventionDecision | `noise` | reaches | reaches |
| capture InterventionDecision | `steer` | reaches | reaches |
| capture InterventionDecision | `add(scalar)` | reaches | reaches |
| capture InterventionDecision | `add(tensor)` | reaches | reaches |
| capture InterventionDecision | `replace_with` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `swap_with` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `project_onto` | reaches | reaches |
| capture InterventionDecision | `project_off` | reaches | reaches |
| capture InterventionDecision | `scramble_elements` | reaches | reaches |
| capture InterventionDecision | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `splice_module` | reaches | reaches |
| capture InterventionDecision | `patch_from` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `permute_batch` | reaches | reaches |
| capture InterventionDecision | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `mean_from` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `mean_fill` | reaches | reaches |
| capture InterventionDecision | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| capture InterventionDecision | `set_direction_mean` | reaches | reaches |
| capture InterventionDecision | `subspace_ablate` | reaches | reaches |
| capture InterventionDecision | `subspace_ablate(resample)` | reaches | reaches |
| capture InterventionDecision | `subspace_interchange` | reaches | reaches |
| capture InterventionDecision | `compose` | reaches | reaches |
| capture InterventionDecision | `at_positions` | reaches | reaches |
| capture InterventionDecision | `hook callable` | reaches | reaches |
| fork.do (replay) | `zero_ablate` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `mean_ablate` | reaches | reaches |
| fork.do (replay) | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `scale` | reaches | reaches |
| fork.do (replay) | `clamp` | reaches | reaches |
| fork.do (replay) | `noise` | reaches | reaches |
| fork.do (replay) | `steer` | reaches | reaches |
| fork.do (replay) | `add(scalar)` | reaches | reaches |
| fork.do (replay) | `add(tensor)` | reaches | reaches |
| fork.do (replay) | `replace_with` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `swap_with` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `project_onto` | reaches | reaches |
| fork.do (replay) | `project_off` | reaches | reaches |
| fork.do (replay) | `scramble_elements` | reaches | reaches |
| fork.do (replay) | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `splice_module` | reaches | reaches |
| fork.do (replay) | `patch_from` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `permute_batch` | reaches | reaches |
| fork.do (replay) | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `mean_from` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `mean_fill` | reaches | reaches |
| fork.do (replay) | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| fork.do (replay) | `set_direction_mean` | reaches | reaches |
| fork.do (replay) | `subspace_ablate` | reaches | reaches |
| fork.do (replay) | `subspace_ablate(resample)` | reaches | reaches |
| fork.do (replay) | `subspace_interchange` | reaches | reaches |
| fork.do (replay) | `compose` | reaches | reaches |
| fork.do (replay) | `at_positions` | reaches | reaches |
| fork.do (replay) | `hook callable` | reaches | reaches |
| do(model=, x=) (rerun) | `zero_ablate` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `mean_ablate` | reaches | reaches |
| do(model=, x=) (rerun) | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `scale` | reaches | reaches |
| do(model=, x=) (rerun) | `clamp` | reaches | reaches |
| do(model=, x=) (rerun) | `noise` | reaches | reaches |
| do(model=, x=) (rerun) | `steer` | reaches | reaches |
| do(model=, x=) (rerun) | `add(scalar)` | reaches | reaches |
| do(model=, x=) (rerun) | `add(tensor)` | reaches | reaches |
| do(model=, x=) (rerun) | `replace_with` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `swap_with` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `project_onto` | reaches | reaches |
| do(model=, x=) (rerun) | `project_off` | reaches | reaches |
| do(model=, x=) (rerun) | `scramble_elements` | reaches | reaches |
| do(model=, x=) (rerun) | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `splice_module` | reaches | reaches |
| do(model=, x=) (rerun) | `patch_from` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `permute_batch` | reaches | reaches |
| do(model=, x=) (rerun) | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `mean_from` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `mean_fill` | reaches | reaches |
| do(model=, x=) (rerun) | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| do(model=, x=) (rerun) | `set_direction_mean` | reaches | reaches |
| do(model=, x=) (rerun) | `subspace_ablate` | reaches | reaches |
| do(model=, x=) (rerun) | `subspace_ablate(resample)` | reaches | reaches |
| do(model=, x=) (rerun) | `subspace_interchange` | reaches | reaches |
| do(model=, x=) (rerun) | `compose` | reaches | reaches |
| do(model=, x=) (rerun) | `at_positions` | reaches | reaches |
| do(model=, x=) (rerun) | `hook callable` | reaches | reaches |
| attach_hooks + run | `zero_ablate` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `mean_ablate` | reaches | reaches |
| attach_hooks + run | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `scale` | reaches | reaches |
| attach_hooks + run | `clamp` | reaches | reaches |
| attach_hooks + run | `noise` | reaches | reaches |
| attach_hooks + run | `steer` | reaches | reaches |
| attach_hooks + run | `add(scalar)` | reaches | reaches |
| attach_hooks + run | `add(tensor)` | reaches | reaches |
| attach_hooks + run | `replace_with` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `swap_with` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `project_onto` | reaches | reaches |
| attach_hooks + run | `project_off` | reaches | reaches |
| attach_hooks + run | `scramble_elements` | reaches | reaches |
| attach_hooks + run | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `splice_module` | reaches | reaches |
| attach_hooks + run | `patch_from` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `permute_batch` | reaches | reaches |
| attach_hooks + run | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `mean_from` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `mean_fill` | reaches | reaches |
| attach_hooks + run | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| attach_hooks + run | `set_direction_mean` | reaches | reaches |
| attach_hooks + run | `subspace_ablate` | reaches | reaches |
| attach_hooks + run | `subspace_ablate(resample)` | reaches | reaches |
| attach_hooks + run | `subspace_interchange` | reaches | reaches |
| attach_hooks + run | `compose` | reaches | reaches |
| attach_hooks + run | `at_positions` | reaches | reaches |
| attach_hooks + run | `hook callable` | reaches | reaches |
| spec.bind(model) | `zero_ablate` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `mean_ablate` | reaches | reaches |
| spec.bind(model) | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `scale` | reaches | reaches |
| spec.bind(model) | `clamp` | reaches | reaches |
| spec.bind(model) | `noise` | reaches | reaches |
| spec.bind(model) | `steer` | reaches | reaches |
| spec.bind(model) | `add(scalar)` | reaches | reaches |
| spec.bind(model) | `add(tensor)` | reaches | reaches |
| spec.bind(model) | `replace_with` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `swap_with` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `project_onto` | reaches | reaches |
| spec.bind(model) | `project_off` | reaches | reaches |
| spec.bind(model) | `scramble_elements` | reaches | reaches |
| spec.bind(model) | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `splice_module` | reaches | reaches |
| spec.bind(model) | `patch_from` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `permute_batch` | reaches | reaches |
| spec.bind(model) | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `mean_from` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `mean_fill` | reaches | reaches |
| spec.bind(model) | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| spec.bind(model) | `set_direction_mean` | reaches | reaches |
| spec.bind(model) | `subspace_ablate` | reaches | reaches |
| spec.bind(model) | `subspace_ablate(resample)` | reaches | reaches |
| spec.bind(model) | `subspace_interchange` | reaches | reaches |
| spec.bind(model) | `compose` | reaches | reaches |
| spec.bind(model) | `at_positions` | reaches | reaches |
| spec.bind(model) | `hook callable` | reaches | reaches |
| push(differentiable=True) | `zero_ablate` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `mean_ablate` | cut | cut |
| push(differentiable=True) | `mean_ablate(source)` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `scale` | cut | cut |
| push(differentiable=True) | `clamp` | cut | cut |
| push(differentiable=True) | `noise` | cut | cut |
| push(differentiable=True) | `steer` | cut | cut |
| push(differentiable=True) | `add(scalar)` | cut | cut |
| push(differentiable=True) | `add(tensor)` | cut | cut |
| push(differentiable=True) | `replace_with` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `swap_with` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `project_onto` | cut | cut |
| push(differentiable=True) | `project_off` | cut | cut |
| push(differentiable=True) | `scramble_elements` | cut | cut |
| push(differentiable=True) | `scramble_elements(source)` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `splice_module` | cut | cut |
| push(differentiable=True) | `patch_from` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `permute_batch` | cut | cut |
| push(differentiable=True) | `resample_rows_from` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `mean_from` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `mean_fill` | cut | cut |
| push(differentiable=True) | `mean_fill(ref)` | cut | raises:edit_detached_from_graph |
| push(differentiable=True) | `set_direction_mean` | cut | cut |
| push(differentiable=True) | `subspace_ablate` | cut | cut |
| push(differentiable=True) | `subspace_ablate(resample)` | cut | cut |
| push(differentiable=True) | `subspace_interchange` | cut | cut |
| push(differentiable=True) | `compose` | cut | cut |
| push(differentiable=True) | `at_positions` | cut | cut |
| push(differentiable=True) | `hook callable` | cut | cut |
| do(tl.units(...)) | `zero_ablate` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `mean_ablate` | reaches | reaches |
| do(tl.units(...)) | `mean_ablate(source)` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `scale` | reaches | reaches |
| do(tl.units(...)) | `clamp` | reaches | reaches |
| do(tl.units(...)) | `noise` | reaches | reaches |
| do(tl.units(...)) | `steer` | reaches | reaches |
| do(tl.units(...)) | `add(scalar)` | reaches | reaches |
| do(tl.units(...)) | `add(tensor)` | reaches | reaches |
| do(tl.units(...)) | `replace_with` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `swap_with` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `project_onto` | reaches | reaches |
| do(tl.units(...)) | `project_off` | reaches | reaches |
| do(tl.units(...)) | `scramble_elements` | reaches | reaches |
| do(tl.units(...)) | `scramble_elements(source)` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `splice_module` | reaches | reaches |
| do(tl.units(...)) | `patch_from` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `permute_batch` | reaches | reaches |
| do(tl.units(...)) | `resample_rows_from` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `mean_from` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `mean_fill` | reaches | reaches |
| do(tl.units(...)) | `mean_fill(ref)` | reaches | raises:edit_detached_from_graph |
| do(tl.units(...)) | `set_direction_mean` | reaches | reaches |
| do(tl.units(...)) | `subspace_ablate` | reaches | reaches |
| do(tl.units(...)) | `subspace_ablate(resample)` | reaches | reaches |
| do(tl.units(...)) | `subspace_interchange` | reaches | reaches |
| do(tl.units(...)) | `compose` | reaches | reaches |
| do(tl.units(...)) | `at_positions` | reaches | reaches |
| do(tl.units(...)) | `hook callable` | reaches | reaches |
| do(tl.subspace(...)) | `zero_ablate` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `mean_ablate` | reaches | reaches |
| do(tl.subspace(...)) | `mean_ablate(source)` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `scale` | reaches | reaches |
| do(tl.subspace(...)) | `clamp` | reaches | reaches |
| do(tl.subspace(...)) | `noise` | reaches | reaches |
| do(tl.subspace(...)) | `steer` | reaches | reaches |
| do(tl.subspace(...)) | `add(scalar)` | reaches | reaches |
| do(tl.subspace(...)) | `add(tensor)` | reaches | reaches |
| do(tl.subspace(...)) | `replace_with` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `swap_with` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `project_onto` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `project_off` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `scramble_elements` | reaches | reaches |
| do(tl.subspace(...)) | `scramble_elements(source)` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `splice_module` | reaches | reaches |
| do(tl.subspace(...)) | `patch_from` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `permute_batch` | reaches | reaches |
| do(tl.subspace(...)) | `resample_rows_from` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `mean_from` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `mean_fill` | reaches | reaches |
| do(tl.subspace(...)) | `mean_fill(ref)` | reaches | raises:edit_detached_from_graph |
| do(tl.subspace(...)) | `set_direction_mean` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `subspace_ablate` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `subspace_ablate(resample)` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `subspace_interchange` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `compose` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `at_positions` | raises:subspace_edit_unsupported | raises:subspace_edit_unsupported |
| do(tl.subspace(...)) | `hook callable` | reaches | reaches |
