# Coming from TransformerLens

This page is for people who read activations with `HookedTransformer.run_with_cache`, name sites
with hook-point strings, or patch with `run_with_hooks`. The examples were written against
TransformerLens 4.x (the version line in [landscape.md](landscape.md)) and the TorchLens version on
this branch. Checked 2026-11-03.

TorchLens labels come from eager PyTorch execution, not from TransformerLens hook-point names, so
the migration is a capture plus site discovery, or a name lookup through
`torchlens.mechinterp.resolve_alias`.

## The same task in both

The task: read the residual stream after the first transformer block of a tiny model.

### In TransformerLens

```python
# migration-test: tool=transformer_lens expected=[1, 4, 16]
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

torch.manual_seed(0)
cfg = HookedTransformerConfig(
    n_layers=1, d_model=16, n_ctx=8, d_head=8, n_heads=2, d_vocab=64, act_fn="relu"
)
model = HookedTransformer(cfg)
tokens = torch.tensor([[1, 2, 3, 4]])
_, cache = model.run_with_cache(tokens)
RESULT = list(cache["blocks.0.hook_resid_post"].shape)
```

### In TorchLens

```python
# migration-test: tool=torchlens expected=[1, 4, 16]
import torch
import transformers

import torchlens as tl

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=64, n_positions=16)
model = transformers.GPT2LMHeadModel(config).eval()
tokens = torch.tensor([[1, 2, 3, 4]])

trace = tl.trace(model, tokens, save=tl.module("transformer.h.0"))
RESULT = list(trace["transformer.h.0"].out.shape)
trace.cleanup()
```

## Translation table

| TransformerLens operation | TorchLens idiom | Parity |
| --- | --- | --- |
| Cache activations with `run_with_cache` | `log = tl.trace(model, x, save=tl.in_module("block"))` | TorchLens records a broader PyTorch op DAG, not only transformer hook points. |
| Inspect hook names | `log.find_sites(tl.label_contains("..."))` or `log.summary()` | Equivalent discovery workflow, different names. |
| Target a hook point by TransformerLens name | `torchlens.mechinterp.resolve_alias(log, "blocks.5.attn.hook_pattern")` (2.x names, 3.x bridge-cache names, 4.x canonical names such as `blocks.5.attn.o.hook_in`, plus tuple and compact `"k6"` forms), or discover once and use `tl.label(label)` | TransformerLens spellings resolve against the trace; exact labels remain the reproducible form. |
| `run_with_hooks` | `tl.trace(model, x, intervene=tl.when(site, hook), save=site)` | Capture-time `intervene=` is the direct live-execution equivalent when the site is known. |
| Patch from a clean cache into a corrupted run | Capture clean and corrupted traces, then `corrupted.fork().do(site, tl.patch_from(clean))` | Equivalent for graph-stable patching. |
| `act_patch` and patching heatmaps | `torchlens.mechinterp.patch_heads_grid` / `patch_residual_grid` (turnkey `PatchGrid` results); gradient-based attribution through `torchlens.attribution` | Turnkey grids ship; checked against TransformerLens as the oracle on the cases the test suite covers. |
| ActivationCache analysis verbs (accumulated residual, decomposition, head stacking, logit attributions) | `torchlens.mechinterp.residual_accumulation` / `residual_decomposition` / `direct_logit_contributions` / `attention_head_contributions` / `head_scores` / `test_prompt` | The analysis kit follows the ActivationCache design, with credit; the LayerNorm linearization is checked against TransformerLens's cached-scale convention. |
| Activation ablation helpers | `tl.zero_ablate`, `tl.mean_ablate`, `torchlens.intervention.scramble_elements` (the rename of the removed `resample_ablate`) | Built-in helpers cover the common cases. |
| Residual stream steering | `tl.steer(direction, strength=...)` at the discovered site | Equivalent if the relevant residual op is visible. |
| Attention-head Q/K/V internals under fused SDPA | Attention facets (`scores`/`pattern`/`z`/`result` menus, head-sliceable views) read head-level semantics with provenance disclosed (captured or reconstructed); `torchlens.mechinterp.patch_head` patches one head on a fused model through an implementation-independent lowering | Head-level addressing works on fused models; raw fused-kernel intermediates are still not op-DAG sites, and the facet provenance says which basis you got. |
| Accumulate many prompt variants | `tl.Bundle({...})` plus `map`, `compare`, `at` | Equivalent comparison container. |
| Transformer pictures (attention heads, logit lens, token strips) | `torchlens.transformer` (`attention_views`, `render_attention`, `prediction_trajectory`, causal receipt grids) | Native renderers; the module also carries a CircuitsVis hand-off. |

Every `torchlens.mechinterp` and facet spelling above is documented-unstable pending the naming
session: the workflows are stable, the exact names may be ratified differently.

## What TorchLens adds

- A trace of the stock model. `tl.trace(model, x)` records every executed tensor operation of any
  PyTorch model, including a stock Hugging Face checkpoint, with no model-family list and no
  re-implementation of the architecture.
- Replay verification: `tl.validate` checks the recording against the model's own output, so a
  mismatch between what ran and what was recorded is reported rather than silent.
- Source lines per operation, loop and pass identity (`attn_1_1:2` is the second call), and
  containment per module.
- Hook-name resolution on the real model: `torchlens.mechinterp.resolve_alias(trace, name)` reports
  whether a name has a counterpart (`real`, `reconstructed_read_only`, `needs_capture`,
  `structurally_absent`, `ambiguous`) and gives the remedy when it does not.
- Interventions as declared, replayable specs (`tl.when`, `trace.fork().do(...)`), and `.tlspec`
  persistence of the recipe.

## What TransformerLens does that TorchLens does not

- TransformerLens is more ergonomic when your whole workflow lives inside its supported model
  families and hook-point vocabulary: everything is pre-named, and its teaching corpus (ARENA
  notebooks, tutorials) speaks those names natively.
- TransformerLens 4.x adds vLLM and Inspect drivers, the backward lens, sparse probing and SVD
  circuit tools, and makes `TransformerBridge` its main model class. TorchLens has none of these
  drivers; for serving-scale capture see [landscape.md](landscape.md).
- Where an earlier version of this page said TorchLens lacked a transformer component vocabulary
  and turnkey patching helpers: `torchlens.mechinterp` now ships the ActivationCache-style
  analysis kit, name-alias resolution and turnkey patch grids, so that concession is withdrawn.
  The remaining gap is the curated pre-named vocabulary, not the analysis verbs.

## Using both

### TransformerLens feeds TorchLens

There is no bridge module for TransformerLens. Two routes work. Trace a `HookedTransformer`
directly (it captures fully and is the subject TorchLens's own oracle tests compare against), or
reload the stock Hugging Face model fresh and trace that, which keeps working as `HookedTransformer`
is retired. TorchLens refuses to trace a `transformer_lens.TransformerBridge` (3.x or 4.x); see
Semantic differences.

### TorchLens feeds TransformerLens

There is no bridge. `trace[label].out` is a plain `torch.Tensor`, so a tensor or a patching value
captured in TorchLens can be returned from a TransformerLens hook function by hand.
`torchlens.mechinterp.resolve_alias` goes the other way for names: it accepts TransformerLens
spellings and returns the native TorchLens coordinates.

## Semantic differences

- A `TransformerBridge` cannot be traced. TransformerLens 4.0 (September 2026) makes
  `TransformerBridge` its main model class and starts deprecating `HookedTransformer`. TorchLens
  refuses to capture a `transformer_lens.TransformerBridge` with a typed `CompatibilityError`
  naming the remedy. The bridge redirects attribute assignment to the HF modules it wraps, so
  TorchLens's forward instrumentation can never land on it; without the refusal, capture died
  partway through with an internal `AttributeError`. Trace a pristine model instead.
- The bridge and boot machinery (`boot_transformers`, bridge construction) mutates the HF model it
  wraps in place. A model instance that has already passed through the bridge is no longer the
  pristine model you loaded; reload it fresh before tracing rather than reusing it.
- TransformerLens hook names still work on the traced Hugging Face model, and an unknown name is
  reported rather than guessed:

```python
# migration-test: tool=torchlens expected=('real', 'transformer.h.1.attn', 'structurally_absent')
import torch
import transformers

import torchlens as tl
import torchlens.mechinterp as mi

torch.manual_seed(0)
config = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
config._attn_implementation = "eager"
model = transformers.GPT2LMHeadModel(config).eval()
tokens = torch.randint(0, 64, (2, 8))

trace = tl.trace(model, tokens, save="all")
RESULT = (
    mi.resolve_alias(trace, "blocks.0.attn.hook_z").status,
    mi.resolve_alias(trace, ("pattern", 1)).module_address,
    mi.resolve_alias(trace, "nonsense.hook_foo").status,
)
trace.cleanup()
```
