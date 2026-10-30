# TransformerLens: agent notes

TransformerLens loads open-weight language models behind a uniform set of named
hook points, caches any internal activation and lets hook functions edit activations
as the model runs. Version 4 has one model system, `TransformerBridge`, wrapping the
Hugging Face model.

## Setup

```python
import torch
from transformer_lens.model_bridge import TransformerBridge
bridge = TransformerBridge.boot_transformers("gpt2", device="cpu")   # fp32, eager attention
bridge.enable_compatibility_mode()      # HookedTransformer-equivalent names and weight processing
prompt = "The Eiffel Tower is in the city of"
tokens = bridge.to_tokens(prompt)       # [batch, pos], BOS prepended
```

Install: `pip install transformer_lens`. Gated models need `HF_TOKEN`. Without
`enable_compatibility_mode()` the bridge runs the raw Hugging Face weights and
exposes Bridge-native hook names only.

## Reading intermediate values

```python
logits, cache = bridge.run_with_cache(prompt)
resid6 = cache["blocks.6.hook_resid_post"]          # [batch, pos, d_model]
attn5  = cache["blocks.5.hook_attn_out"]
z = cache["blocks.5.attn.hook_z"]                   # per-head outputs [batch, pos, head, d_head]
pattern = cache["blocks.5.attn.hook_pattern"]       # [batch, head, q_pos, k_pos]
```

Hook names, canonical (compatibility mode): `hook_embed`, `hook_pos_embed`,
`blocks.{i}.hook_resid_pre`, `.hook_resid_mid`, `.hook_resid_post`, `.hook_attn_out`,
`.hook_mlp_out`, `.attn.hook_q/k/v/z/pattern`, `.mlp.hook_pre`, `.mlp.hook_post`,
`ln_final.hook_normalized`. Bridge-native names are architecture-shaped:
`blocks.{i}.hook_out`, `blocks.{i}.attn.q.hook_out`; the canonical names are aliases
registered at boot. `run_with_cache(x, names_filter=lambda n: n.endswith("resid_post"))`
limits what is stored. Values with no hook point (a sum inside an attention
implementation, for example) are not cached; compute them from neighbouring hooks
or read the component's source.

## Changing intermediate values

A hook function receives `(activation, hook)` and returns the new activation (or
edits it in place and returns None):

```python
def zero_last(act, hook):
    act[:, -1, :] = 0.0
    return act
patched = bridge.run_with_hooks(prompt, fwd_hooks=[("blocks.6.hook_resid_post", zero_last)])

src = cache["blocks.6.hook_resid_post"]               # from another run
def patch_in(act, hook):
    act[:, -1, :] = src[:, -1, :]
    return act
patched = bridge.run_with_hooks(prompt, fwd_hooks=[("blocks.6.hook_resid_post", patch_in)])
```

Persistent hooks: `bridge.add_hook(name, fn)` then `bridge.reset_hooks()`; they also
apply inside `bridge.generate(prompt, max_new_tokens=n, do_sample=False)`, so a
steering hook that adds a direction at `hook_resid_post` runs on every step (index
`[:, -1]` for the newest position once the key-value cache is on). Head-level
patching: edit one head's slice of `attn.hook_z` (`act[:, :, head, :] = ...`).
Mean- or zero-ablation is the same hook with a different right-hand side.

## Gradients and attribution

```python
bridge.requires_grad_(False)
store = {}
def keep(act, hook):
    act.retain_grad(); store[hook.name] = act
    return act
logits = bridge.run_with_hooks(prompt, fwd_hooks=[("blocks.6.hook_resid_post", keep)])
paris = bridge.to_single_token(" Paris")
logits[0, -1, paris].backward()
grad = store["blocks.6.hook_resid_post"].grad
grad_times_act = grad * store["blocks.6.hook_resid_post"].detach()
```

Backward hooks: `run_with_hooks(..., bwd_hooks=[(name, fn)])` receives the gradient
at that hook point. Attribution patching is the forward value difference times the
gradient at the same hook; `torch.autograd.grad` with `grad_outputs` gives a
vector-Jacobian product through a kept activation, and `torch.func.jacrev` on a
function that runs the remaining blocks gives a full Jacobian of a small model.
`transformer_lens.tools.analysis` has single-call analyses over the cache such as
direct logit attribution; `cache.decompose_resid`, `cache.stack_head_results` and
`cache.apply_ln_to_stack` build per-component residual decompositions.

## Batching, timing and determinism

`run_with_cache` on a batch of prompts of equal length caches `[batch, pos, ...]`
tensors; pad with `bridge.to_tokens(list_of_prompts, padding_side="left")` and
index positions from the end. Each hook function is a Python call per forward, so
keep the number of hooked points small in timed loops and use `names_filter`.
Compatibility mode folds layer norms and centres weights (`fold_ln`,
`center_writing_weights`, `center_unembed`); raw-HF comparisons must use the same
load configuration (fp32, eager attention) or turn compatibility mode off.
Determinism: `torch.manual_seed`, `do_sample=False`, CPU float32.

## Pitfalls

1. `to_tokens` prepends BOS by default; position indices from the raw tokenizer are
   off by one. Pass `prepend_bos=False` when comparing with plain `transformers`.
2. Compatibility mode changes the weights (folded layer norm, centred unembed), so
   residual values differ from the raw Hugging Face model even though logits agree.
3. A hook that returns nothing and did not edit in place changes nothing.
4. `hook_resid_mid` and some attention hooks are not present on every
   architecture: probe `bridge.hook_dict` before assuming a name.
5. Hooks added with `add_hook` persist until `reset_hooks()`; stale hooks corrupt
   later runs.
6. `hook_z` is per head before the output projection; multiply by the head's slice
   of `W_O` for the head's contribution to the residual stream.
7. The `HookedTransformer`, `HookedEncoder` and `HookedSAETransformer` classes are
   gone in 4.0; adapt old tutorial code to `TransformerBridge`.

## Where to look

- https://transformerlensorg.github.io/TransformerLens/ (docs, `migrating_to_v4`, `compatibility_mode`)
- https://github.com/TransformerLensOrg/TransformerLens (README, `AGENTS.md`)
- `bridge.hook_dict.keys()` lists every hook point; `print(bridge)` shows the components
- `transformer_lens/hook_points.py` (`HookPoint`, `LensHandle`)
- ARENA tutorials: https://learn.arena.education/chapter1_transformer_interp/
