# Coming from AutoCircuit

This page is for people who find and test circuits with AutoCircuit's patching and pruning and
want to know which parts TorchLens covers. TorchLens gives you the sites, the patch operations
and the metrics container; the automated circuit search is a loop you write (a recipe is
linked below). AutoCircuit is not installed in TorchLens's test environment, so its call is
shown as text and the TorchLens side runs on a tiny model. Checked 2026-11-03.

## The same task in both

Patch one component's activation from a corrupted run into a clean run and read the changed
output.

### In AutoCircuit

AutoCircuit wraps a model as a patchable model and patches edges between its nodes over a
dataloader of clean and corrupted prompts, then prunes or scores edges. Its call is not shown
here because it depends on the version you have; see its documentation.

### In TorchLens

Capture the clean and corrupted runs, fork the clean trace, and patch the site from the
corrupted trace. In this two-layer network the patched hidden activation determines the output,
so the patched run's output equals the corrupted run's.

```python
# migration-test: tool=torchlens expected=(True, True)
import torch
from torch import nn
import torchlens as tl

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
options = tl.options.CaptureOptions(replayable=True)
clean = tl.trace(model, torch.randn(3, 4), capture=options)
corrupt = tl.trace(model, torch.randn(3, 4), capture=options)

patched = clean.fork("patched")
patched.do(tl.module("1"), tl.patch_from(corrupt))
out = patched[patched.output_layers[0]].out
RESULT = (
    torch.allclose(out, corrupt[corrupt.output_layers[0]].out, atol=1e-6),
    not torch.allclose(out, clean[clean.output_layers[0]].out, atol=1e-6),
)
```

## Translation table

| AutoCircuit operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Identify circuit nodes | `log.find_sites(...)`, graph metadata, and visualization | TorchLens exposes raw graph sites; automated circuit search is separate. |
| Patch edges/nodes | `.do(site, edit)` on a forked trace (replays the recorded graph from the edited value; `.do(site, edit, model=model, inputs=x)` runs the model again), or `.attach_hooks(...)` then `.run(...)` | Node patching equivalent; edges patch with `torchlens.mechinterp.patch_edges` (one or several receivers per call). |
| Run path patching sweeps | Fork logs or loop over selectors and compute an explicit metric (or call your metric on the `tl.Bundle`) | Manual but supported. |
| Compare clean/corrupted datasets | Capture logs and group them with `Bundle` | Equivalent container workflow. |
| Automatic circuit discovery | Recipe: `docs/recipes/acdc_ioi.md` (ACDC as a driver loop); edge scores: `torchlens.mechinterp.edge_scores` | Recipe-level; no built-in pruning optimizer. |
| Prune/score circuit components | Use explicit metrics over intervention variants | Partial; no built-in pruning optimizer. |
| Save circuit intervention | `save_intervention(level="portable")` for supported recipes | Equivalent for recipe publication. |
| Visualize graph with intervention marks | `log.show(..., intervention_mode="node_mark")` | Equivalent graph-level visibility. |
| Append evaluation batches | `trace.run(inputs=batch)` per batch, or `tl.when(site, action).bind(model)` for many forwards | Equivalent; see `docs/guides/fast_paths.md`. |
| Fused attention internals | Eager-attention load: the softmax and score ops are captured ops you read directly. Default fused SDPA load: `tl.trace(model, x, reconstruction_ready=True)` serves read-only reconstructed attention facets (`scores`, `pattern`, `z`, checked against the fused output; see `docs/facets.md`) | Head-level semantics are readable on fused models; the raw kernel intermediates are not captured ops, and editing them needs an eager recapture. |

## What TorchLens adds

- Every executed op is a site, with its module address and op label; graph metadata and
  drawings come from the same capture (`log.find_sites(...)`, `trace.draw(...)`).
- Edge patching with `torchlens.mechinterp.patch_edges` (one or several receivers per call),
  and edge scores with `torchlens.mechinterp.edge_scores`.
- An ACDC recipe as a driver loop over those calls (`docs/recipes/acdc_ioi.md`).
- Clean and corrupted runs grouped in a `tl.Bundle`, and recipes saved as `.tlspec/`
  directories (`save_intervention(level="portable")`).
- Intervention marks on the graph picture: `log.show(..., intervention_mode="node_mark")`.

## What AutoCircuit does that TorchLens does not

- AutoCircuit packages a patchable model, ablation and pruning loops over a dataset and its
  circuit scoring. TorchLens has no built-in pruning optimizer; scoring circuit components
  with explicit metrics over intervention variants is yours to write.
- Automatic circuit discovery is a recipe, not a call: `docs/recipes/acdc_ioi.md` shows ACDC
  as a driver loop. See [where to go instead](where_to_go_instead.md).

## Using both

### AutoCircuit feeds TorchLens

There is no bridge. An AutoCircuit patchable model wraps your model, so trace the original
`nn.Module` with `tl.trace`; edge scores or circuits you computed come across as ordinary
Python objects (edge lists, tensors) that you pass to `patch_edges`.

### TorchLens feeds AutoCircuit

There is no bridge. Activations and metrics from TorchLens are plain tensors; hand them to
your own scoring code, or to AutoCircuit's dataloaders as ordinary inputs.

## Semantic differences

- **Nodes versus edges.** Node patching is `.do(site, edit)` on a fork; edges patch through
  `torchlens.mechinterp.patch_edges`.
- **Replay versus rerun.** `.do` on a replayable trace replays the recorded graph from the
  edited value; pass `model=` and `inputs=` to run the model again.
- **Evaluation batches.** Append batches with `trace.run(inputs=batch)` per batch, or
  `tl.when(site, action).bind(model)` for many forwards (see `docs/guides/fast_paths.md`).
- **Fused attention.** Head-level semantics are readable on fused models through
  reconstructed facets (`tl.trace(model, x, reconstruction_ready=True)`, `docs/facets.md`);
  the raw kernel intermediates are not captured ops, and editing them needs an eager
  recapture.
