# Coming from penzai

This page is for people with penzai code (named-axis JAX model trees and functional surgery)
who want to know what carries over to TorchLens. Status: the penzai repository is archived
(read-only since its last push on 2025-06-22), and treescope, the pretty-printer that grew out
of it, is idle (last release v0.1.10 on 2025-08-08, last push 2026-06-18, checked 2026-11-03).
Existing penzai code keeps working; new work needs another home. Other tools' status:
[comparable tools](landscape.md). TorchLens does not run penzai, so the penzai side of the
example is described in prose and the TorchLens side is executed. Checked 2026-11-03.

## The same task in both

Select a subtree, swap its value, and look at the result.

### In penzai

Penzai selects a subtree of a model tree with `pz.select` and rewrites it functionally: you
select the model, narrow the selection to the layer type you want with
`.at_instances_of(pz.nn.Linear)`, and pass your own function from one matched layer to its
replacement to `.apply(...)`, which returns the edited model. Penzai is a JAX library and its
repository is archived, so this is described here rather than run; check the calls against the
penzai version you have.

### In TorchLens

TorchLens edits a recorded run instead of the model definition: capture a trace, fork
it, apply the edit at a site, and read both runs.

```python
# migration-test: tool=torchlens expected=([[2.5, 2.5]], [[0.0, 0.0]])
import torch
from torch import nn
import torchlens as tl


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x):
        return torch.relu(self.proj(x))


trace = tl.trace(Tiny(), torch.tensor([[2.0, 3.0]]))
edited = trace.fork("zeroed")
edited.do(tl.module("proj"), tl.zero_ablate())
RESULT = (
    trace[trace.output_layers[0]].out.tolist(),  # the original run is untouched
    edited[edited.output_layers[0]].out.tolist(),
)
```

## Translation table

| Penzai operation | TorchLens idiom | Parity |
| --- | --- | --- |
| Name and select model subtrees | `tl.in_module("path")`, `tl.module("path")`, and exact labels after discovery | Similar goal; TorchLens uses PyTorch module addresses and op labels. |
| Functional model surgery | Capture with `tl.trace(model, x)`; `trace.fork()`, then `.do(site, edit)` (replays the recorded graph from the edited value) or `.do(site, edit, model=model, inputs=x)` (runs the model again) | TorchLens edits a recorded run, not the model definition. |
| Inspect intermediate values | `trace.find_sites(selector).first().out`, or `trace[label].out` | Equivalent for captured tensors. |
| Patch or replace a value | `.do(site, edit)` on a forked trace | Equivalent for visible activations. |
| Declarative named axes | Explicit tensor shapes, plus `dim=` where a helper needs it | No named-axis system. |
| Transformer-specific layer abstractions | `torchlens.mechinterp` (TransformerLens-style names through `resolve_alias`) and attention facets | Different vocabulary; no penzai layer classes. |
| Save a modified computation | `trace.save_intervention(path, level="portable")` writes a `.tlspec/` recipe | TorchLens saves recipes, not rewritten model trees. |
| Compare several variants | `tl.Bundle({...})` | Equivalent comparison container. |
| Intervene on fused kernel internals | Attention facets and `torchlens.mechinterp.patch_head` for heads on fused attention; manual unfused code otherwise | Raw fused-kernel intermediates are not captured ops. |
| Pure JAX workflows | `tl.trace(fn, (params, x), backend="jax")`: a backend that captures a functional call `fn(params, *inputs)` from its jaxpr; Equinox and Flax NNX modules are supported roots | Limits: `jax.jit`, `lax.cond`, `lax.scan`, `lax.while_loop` and custom-VJP regions refuse in strict mode; gradients are derived, not captured backward passes (see `docs/backends.md`). |
| Pretty-print model trees and arrays (treescope) | The `torchlens.bridge.treescope` bridge (`docs/guides/treescope_cards.md`), or the bounded `repr`/`str` cards on every trace object | |

## What TorchLens adds

- Every executed op is a site, selected by PyTorch module address or op label
  (`tl.module`, `tl.in_module`), without rewriting the model.
- Edits on a fork leave the original run intact, so the clean and edited runs sit side by side
  (`trace.fork()`, `trace.do(...)`), and `tl.Bundle({...})` compares several variants.
- Intervention recipes persist as `.tlspec/` directories (`trace.save_intervention(path,
  level="portable")`).
- Attention facets and head-level patching on fused attention (`docs/facets.md`,
  `torchlens.mechinterp.patch_head`).
- A JAX backend (`tl.trace(fn, (params, x), backend="jax")`), described in
  `docs/backends.md`.

## What penzai does that TorchLens does not

Penzai was the better fit for functional JAX model surgery, named axes, and workflows that
want the model tree itself to be the edited artifact, and nothing in TorchLens replaces that:
TorchLens records and edits runs, it does not rewrite model trees. For PyTorch models,
TorchLens's capture surface is PyTorch eager; for JAX, the JAX backend captures declared
functional calls as jaxpr-derived traces, which is enough to read and compare activations but
is not a JAX tree-rewriting system. Penzai's named axes and layer classes have no TorchLens
counterpart. See [where to go instead](where_to_go_instead.md) and
[comparable tools](landscape.md).

## Using both

### penzai feeds TorchLens

There is no bridge from a penzai model tree to TorchLens. For a pure-JAX workflow, trace the
functional call with `tl.trace(fn, (params, x), backend="jax")` (the JAX backend; Equinox and
Flax NNX modules are supported roots), or port the model to PyTorch.

### TorchLens feeds penzai

There is no bridge. treescope, the pretty-printer that came out of penzai, has an opt-in
upgrade, `torchlens.bridge.treescope` (`docs/guides/treescope_cards.md`); TorchLens's own
`repr`/`str` cards need no treescope. Arrays cross as plain data (for example through NumPy).

## Semantic differences

- **Edits a run, not a tree.** TorchLens saves recipes, not rewritten model trees; the model
  definition is unchanged.
- **Replay versus rerun.** `.do(site, edit)` on a captured trace replays the recorded graph
  from the edited value; passing `model=` and `inputs=` runs the model again, which matters
  when the edit may change control flow or randomness.
- **No named axes.** Tensors stay positional; helpers take `dim=` where they need an axis.
- **JAX backend limits.** `jax.jit`, `lax.cond`, `lax.scan`, `lax.while_loop` and custom-VJP
  regions refuse in strict mode, and gradients are derived, not captured backward passes
  (see `docs/backends.md`).
