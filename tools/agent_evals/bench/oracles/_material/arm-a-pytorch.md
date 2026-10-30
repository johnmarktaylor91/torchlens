# Plain PyTorch: agent notes

PyTorch itself gives you every intermediate value of a model's forward and
backward pass through module hooks, autograd and `torch.func`. Nothing else is installed.

## Setup

```python
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0); rng = np.random.default_rng(0)
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).eval()
blocks = model.transformer.h            # GPT-2 blocks; ln_f and lm_head follow them
ids = tok("The Eiffel Tower is in", return_tensors="pt").input_ids
```

## Reading intermediate values

```python
acts = {}
def save(name):
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        acts[name] = t.detach().clone()      # clone: later in-place ops may overwrite it
    return hook
hs = [blocks[i].register_forward_hook(save(f"resid_post.{i}")) for i in range(12)]
hs.append(blocks[5].attn.register_forward_hook(save("attn_out.5")))
hs.append(blocks[5].mlp.register_forward_hook(save("mlp_out.5")))
with torch.no_grad():
    out = model(ids, output_hidden_states=True)   # out.hidden_states[i+1] == block i output
for h in hs: h.remove()
```

Module inputs: `register_forward_pre_hook(lambda mod, args: ...)` sees `args[0]`.
A value computed inside `forward` by a plain function (a residual `x + y`, `F.gelu`,
`F.scaled_dot_product_attention`) has no hook. Options: hook the modules on both
sides and recompute the value from them; call the submodules yourself in the same
order as the module's `forward` (read its source with `inspect.getsource`); or
subclass and override `forward`. Check a recomputation against the model's own
output with `torch.allclose`.

Per-head attention outputs in GPT-2: the input of `blocks[i].attn.c_proj` has shape
`[batch, pos, 768]`; view it as `[batch, pos, 12, 64]` to get head `h` at `[..., h, :]`.

## Changing intermediate values

```python
def replace_last_token(new):            # forward hook that returns a new output
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        t = t.clone(); t[:, -1, :] = new
        return (t, *out[1:]) if isinstance(out, (tuple, list)) else t
    return hook
h = blocks[6].register_forward_hook(replace_last_token(acts["resid_post.6"][0, -1] * 0))
with torch.no_grad(): patched = model(ids).logits
h.remove()
```

Patching from another run: record with the hook above on the source input, then
install a hook that returns the recorded tensor on the target input; make sure both
runs have the same sequence length or index the positions you mean. For a
sub-module input use `register_forward_pre_hook` and return a new `args` tuple.
During generation, hooks stay installed across every step of `model.generate`;
with the key-value cache on, later steps see only the new token's position, so index
`[:, -1]` rather than an absolute position. `model.generate(ids, do_sample=False,
max_new_tokens=n)` is deterministic.

## Gradients and attribution

```python
model.requires_grad_(False)             # no parameter gradients needed
captured = {}
def keep(mod, inp, out):
    t = out[0] if isinstance(out, (tuple, list)) else out
    t.retain_grad(); captured["t"] = t   # keep the graph tensor, do not detach
h = blocks[6].register_forward_hook(keep)
logits = model(ids).logits
metric = logits[0, -1, tok(" Paris").input_ids[0]]
metric.backward()
grad = captured["t"].grad               # same shape as the activation
grad_times_act = grad * captured["t"].detach()
h.remove()
```

`torch.autograd.grad(outputs, inputs, grad_outputs=v)` gives a vector-Jacobian product
without touching `.grad`; pass `retain_graph=True` to reuse one forward for many
`v`. Replicating the input along the batch axis and giving each row its own
`grad_outputs` row computes many VJPs in one backward. For a full Jacobian of a
small function use `torch.func.jacrev(f)(x)` or `torch.func.jvp`; for a function
of an intermediate value, build `f` by running the remaining modules explicitly.
A gradient with respect to a module's *input* needs that input to require grad: use a
pre-hook that calls `args[0].requires_grad_(True)` when parameters are frozen.

## Batching, timing and determinism

Put the variants in the batch dimension and run one forward; hooks see the whole
batch, so index rows inside them. Count hook calls (a module such as a shared
`nn.ReLU` can run several times per forward) and remove every handle you register
(`handle.remove()`; a `try/finally` is cheap). Measure with
`time.perf_counter()` around a warmed-up call; a few hooks cost well under 5 percent.
Determinism on CPU: fixed seeds (`numpy.random.default_rng(seed)` for data,
`torch.manual_seed` for torch), float32, `model.eval()`, `torch.no_grad()` when no
gradient is needed, and the same batch size when comparing against a reference
(convolution kernels can differ by about 1e-5 between batch sizes).

## Pitfalls

1. In-place ops (`nn.ReLU(inplace=True)`, `x += y`) overwrite tensors you hooked
   earlier: clone in the hook.
2. A hook that modifies `out` in place without returning it still changes the run;
   one that builds a new tensor must `return` it.
3. Block outputs may be tuples; take `[0]`. Attention modules return `(output, weights)`.
4. `output_hidden_states` returns the embedding first, and the last entry can be
   after the final norm; check the library's forward before trusting indices.
5. Left padding shifts positions; tokenise one prompt at a time or use attention masks.
6. `retain_grad()` only works on non-leaf tensors that require grad; frozen models
   need a `requires_grad_(True)` on the earliest tensor you differentiate through.
7. Forgetting `handle.remove()` leaves hooks on the model for every later call.
8. Position indices inside hooks differ between the first generation step and later
   cached steps.

## Where to look

- https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html (hooks)
- https://docs.pytorch.org/docs/stable/autograd.html and https://docs.pytorch.org/docs/stable/func.html
- `print(model)` and `inspect.getsource(type(blocks[0]).forward)` for the computation order
- `transformers` model code: `transformers/models/<family>/modeling_<family>.py`
