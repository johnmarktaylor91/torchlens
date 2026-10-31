# Pyvene to TorchLens v2 Migration

Pyvene and pyreft are idle upstream. Runnable versions of these rows are on
[Migrating from Pyvene](from_pyvene.md).

| Pyvene operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Define intervention locations | Selectors such as `tl.module`, `tl.func`, `tl.label`, `tl.in_module`; facets such as `tl.head(i, "q")` for one attention head | Equivalent targeting, on the unwrapped model. |
| Run a configured intervention | `tl.trace(model, x, intervene=tl.when(site, action))`, `trace.fork().do(site, action)`, or `tl.when(site, action).bind(model)` | Equivalent for local PyTorch execution; the engines are compared on the [intervention explainers](../intervention_explainers.md#which-engine-when) page. |
| Interchange (representation swap) from a source run | `tl.patch_from(source_trace, source_site=..., positions=...)` | Equivalent, including different prompt lengths through a position map. |
| Zero and mean ablations | `tl.zero_ablate()`, `torchlens.intervention.stochastic.mean_fill(dim=...)`, `torchlens.intervention.stochastic.mean_from(population)` | Equivalent common helpers; see [stochastic and population edits](../stochastic_edits.md). |
| Resampling ablation | `torchlens.intervention.resample_rows_from(population, seed=..., axis=...)` | Whole-row donors from a population, seeded and recorded. |
| Directional intervention | `tl.steer(direction, strength=..., dim=...)` | Equivalent tensor steering helper. |
| Generation under an intervention | `spec.bind(model).generate(ids, ...)` | Runs the model's own `generate()` with the edit on every step, KV cache included. |
| Fixed-direction subspace edits | `tl.project_out(direction, ...)`, `tl.project_onto(direction, ...)`, `torchlens.intervention.set_direction_mean(...)` | Equivalent for a given direction; no learned rotation. |
| SAE feature interventions | `torchlens.bridge.sae.splice(..., latents_edit=...)` | Duck-typed SAE (`encode` and `decode`); no SAE package required. |
| Trainable interventions (DAS, pyreft adapters) | Your own optimizer over parameters a bound spec reads | TorchLens ships no trainer; the training loop is yours. |
| Multi-source causal mediation templates | Several traces in a `Bundle`, with explicit metrics | No packaged template. |
| Interchange over token positions | `positions=` on `tl.patch_from`, or `tl.units(label, indices)` | Equivalent when the target activation shape is visible. |
| Persist experiment config | `trace.save_intervention(path, level=...)` writes a `.tlspec/` recipe | TorchLens persists recipes, not full experiment notebooks. |
| Dataset-level batched evaluation | Call the bound spec per batch, or `trace.run(model, batch, replay=tl.options.ReplayOptions(append=True))` when the graph is the same | Equivalent for graph-compatible chunks. |
| Model-family-specific abstractions | No exact equivalent | TorchLens stays PyTorch-op first. |

```python
# migration-test: tool=torchlens expected=True
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)
spec = tl.when(tl.module("1"), tl.zero_ablate())
trace = tl.trace(model, x, save=tl.module("2"), intervene=spec)
RESULT = torch.equal(trace.find_sites(tl.module("2")).first().out, spec.bind(model)(x))
```
