# Stochastic and population edits

Causal scrubbing, resampling ablation and mean ablation replace an activation with values drawn
from, or averaged over, other inputs. TorchLens builds them from one population noun and one
sampling plan, records every draw, and makes each draw reproducible from its seed. Every name on
this page lives in `torchlens.intervention` (not at the `tl.` root), except `tl.mean_ablate` (the root
helper) and `sampling_records`, which imports from `torchlens.intervention.stochastic`. All are
DOCUMENTED-UNSTABLE: the spellings may change without a deprecation shim.

## The pieces

| Name | What it does |
| --- | --- |
| `population(source, origin=...)` | The population: a tensor whose axis-0 slices are the members, a list of same-shape tensors, or a list of captured traces. `origin=` is a required note of where the population came from. |
| `sample_from(population, seed=...)` | The one stochastic primitive: an immutable plan that draws one whole donor per firing. Apply it with `tl.patch_from(plan)`. |
| `resample_rows_from(population, seed=..., axis=...)` | Resampling ablation: each batch row is replaced by one whole donor row, drawn with replacement, independently per row. |
| `permute_batch(seed=..., axis=...)` | Exchanges rows within the current batch (a draw without replacement). |
| `tl.mean_ablate(source=None, *, dim)` | The one mean door. `dim=` is required: `None` averages every axis to one scalar, an int or tuple of ints averages those axes. `source=` is what is averaged: the site's own value (default), a tensor, or a population, whose members are averaged elementwise first (`dim=None` then fills with that member-shaped mean). A mean is a reduction, never a draw, so it takes no seed. |
| `set_direction_mean(direction, population, dim=...)` | Replaces the component along `direction` with the population's mean coefficient and keeps the orthogonal part. |
| `scramble_elements(...)` | An elementwise random scramble. It is not a resampling ablation: each element moves on its own, so rows are no longer coherent inputs. |
| `at_step(*steps)` | A selector qualifier for episode captures: `tl.when(site & at_step(2), edit)` edits only decoding step 2. |
| `sampling_records(trace)` | The recorded draws of every stochastic edit on a trace: seeds, donor ids, permutations. |

One spelling is a false friend. The removed `resample_ablate` was renamed `scramble_elements`
because it never resampled whole inputs; resampling ablation is `resample_rows_from`.

## Seeds and draws

`seed=` is required on every drawing edit: an integer, or `"auto"`, which derives the seed once
per capture from the trace's own random seed and records it. Each firing derives its own seed
from that base and from where it fired, so re-running the same declared experiment reproduces
every draw, and two sites or two passes never silently share one draw. To share a draw on
purpose, pass `share_draw="per_rule"` or `"per_group"` to `sample_from`; the default is
`"per_firing"`.

Axes are never guessed from a tensor's rank: `resample_rows_from` and `permute_batch` take an
explicit `axis=`, and `set_direction_mean` an explicit `dim=`.

## Resampling ablation

Each row of the site is replaced by a whole donor row, and the recorded donor ids reproduce the
result:

```python
# docs-test: expected=[True, True]
import torch
from torch import nn
import torchlens as tl
from torchlens.intervention import population, resample_rows_from
from torchlens.intervention.stochastic import sampling_records


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


torch.manual_seed(0)
trace = tl.trace(
    Toy(), torch.randn(5, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
)
site = tl.label("relu_1_2")
donors = population(torch.randn(3, 4), origin="three donor rows from another dataset")

resampled = trace.fork("resampled")
resampled.do(site, resample_rows_from(donors, seed=2, axis=0))
donor_ids = sampling_records(resampled)[-1]["donor_ids"]
oracle = torch.stack([donors.members[index] for index in donor_ids])

again = trace.fork("again")
again.do(site, resample_rows_from(donors, seed=2, axis=0))
RESULT = [
    torch.equal(resampled["relu_1_2"].out, oracle),
    torch.equal(again["relu_1_2"].out, resampled["relu_1_2"].out),
]
```

`group_by=` and `matching=` restrict the donors to the subject's agreement class, the
agreement-conditioned resample that causal scrubbing needs. A class with one member is disclosed
with a warning, and refused under `strict=True`.

## Means and directional means

```python
# docs-test: expected=[True, True]
import torch
from torch import nn
import torchlens as tl
from torchlens.intervention import population, set_direction_mean


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


torch.manual_seed(0)
trace = tl.trace(
    Toy(), torch.randn(5, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
)
site = tl.label("relu_1_2")
clean = trace["relu_1_2"].out
evidence = population(torch.stack([clean, clean + 2.0]), origin="two evidence runs")

# mean_ablate with a population and dim=None: the elementwise mean over the population.
meaned = trace.fork("meaned")
meaned.do(site, tl.mean_ablate(evidence, dim=None))

# set_direction_mean: only the component along the direction changes.
direction = torch.tensor([1.0, -1.0, 0.0, 0.5])
unit = direction / direction.norm()
directed = trace.fork("directed")
directed.do(site, set_direction_mean(direction, evidence, dim=1))
mean_coefficient = (torch.stack(evidence.members) @ unit).mean()
oracle = clean - (clean @ unit)[:, None] * unit + mean_coefficient * unit

RESULT = [
    torch.allclose(meaned["relu_1_2"].out, clean + 1.0),
    torch.allclose(directed["relu_1_2"].out, oracle, atol=1e-5),
]
```

`set_direction_mean` is the edit to reach for when you mean "the component along a direction".
Under a `tl.subspace(...)` selection, `tl.mean_ablate(population, dim=None)` sets the component in the basis span to the
population mean's component (see
[Subspace edits](intervention_explainers.md#subspace-edits-the-component-along-a-basis)).
`tl.project_out` and `tl.project_onto` are the projection helpers without a population.

## Whole-event donors

`sample_from` builds a plan that draws one whole donor per firing; `tl.patch_from(plan)` applies
it:

```python
# docs-test: expected=True
import torch
from torch import nn
import torchlens as tl
from torchlens.intervention import population, sample_from


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


torch.manual_seed(0)
trace = tl.trace(
    Toy(), torch.randn(5, 4), capture=tl.options.CaptureOptions(intervention_ready=True)
)
donors = population(torch.randn(6, 5, 4), origin="six recorded activations of this site")
patched = trace.fork("patched")
patched.do(tl.label("relu_1_2"), tl.patch_from(sample_from(donors, seed=13)))
RESULT = any(torch.equal(patched["relu_1_2"].out, member) for member in donors.members)
```

A population of captured traces (`population([trace_a, trace_b], origin=...)`) resolves each
donor value at fire time from the matching site of the drawn trace.

## Where these edits run

They run on the torch backend, through `do()` on a trace or fork and through `intervene=` at
capture. When an attribution method's zero fill is not the counterfactual you mean, patch the
site with one of these edits and score the fork yourself (see
[Reading attribution results honestly](reference/attribution.md#reading-attribution-results-honestly)).
