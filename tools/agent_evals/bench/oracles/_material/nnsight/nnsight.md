# nnsight: agent notes

nnsight reads, replaces and backpropagates through every intermediate value in a
model's forward pass, locally or on NDIF for models too large for your machine.

## Setup

```python
import nnsight
from nnsight import TransformersModel
model = TransformersModel("openai-community/gpt2", dispatch=True, device="cpu")
prompt = "The Eiffel Tower is in the city of"
```

Install: `pip install nnsight` (needs `torch` and `transformers`). Pick a device
with `device="cpu"` / `device="cuda:1"`, not `device_map=`.

## Reading intermediate values

```python
with model.trace(prompt):
    clean = model.output.logits.save()                     # read the output

with model.trace(prompt):
    resid = model.transformer.h[5].output.save()           # read an activation
    attn = model.transformer.h[5].attn.output[0].save()    # attention output is a tuple
    model.transformer.h[8].output[:, -1, :] *= 2           # modify one
    edited = model.output.logits.save()                    # see what it changed
assert tuple(resid.shape) == (1, 10, 768)
```

Collecting many values: save the **container** and append raw values:

```python
with model.trace(prompt):
    per_layer = nnsight.save([])
    for block in model.transformer.h:
        per_layer.append(block.output[0, -1])
assert len(per_layer) == 12
```

`tracer.cache(modules=[...])`, opened first thing in the trace, grabs many modules
at once. `.input` is a module's first positional input; `.inputs` the full call.
Values computed inside a module's forward that are not a submodule output are
reached through `.source` (source tracing): `model.transformer.h[5].attn.source`
lists the operations inside that forward, each with its own `.output`.

## Changing intermediate values

Assign to `.output` (or a slice of it) inside the trace; the model sees the new
value downstream. Patch from another run:

```python
with model.trace(corrupt_prompt):
    src = model.transformer.h[6].output[:, -1, :].save()
with model.trace(prompt):
    model.transformer.h[6].output[:, -1, :] = src
    patched = model.output.logits.save()
```

`model.edit(inplace=True)` makes an intervention permanent for later runs.
Generation: a `tracer.iter` loop must not ask for a step the run does not make;
`max_new_tokens` is only an upper bound, so pass `min_new_tokens=` when the bound
has to hold, and check `len()` of what you collected:

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:3]:
        model.transformer.h[6].output[:, -1, :] += 4.0 * direction
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()
assert len(picks) == 3 and ids.shape[1] == 13
```

## Gradients and attribution

Capture in the forward, read in reverse order inside `with metric.backward():`

```python
paris = model.tokenizer(" Paris").input_ids[0]
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        grad = hidden.grad.clone().save()
    act = hidden.detach().save()
grad_times_act = grad * act
```

Saved activations arrive with `requires_grad=True` and a live `grad_fn` that keeps
the forward graph alive: `.detach()` anything you keep or reuse. For a
vector-Jacobian product, build the cotangent as the metric (`(hidden * v).sum()`)
and read `.grad` the same way; a full Jacobian is a loop of such products or
`torch.func.jacrev` on a function that runs the remaining modules on a saved value.

## Batching, timing and determinism

Batch a sweep into one pass. Loop inside the trace, one `tracer.invoke` per
variant, no input on `trace()`:

```python
with model.trace() as tracer:
    scores = nnsight.save({})                   # keyed: invokes finish in model order
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores[layer] = model.output.logits[0, -1, paris]
assert sorted(scores) == list(range(12))        # 12 ablations, one forward pass
```

Inside an invoke, index as if that input were alone; inputs are left-padded to the
longest, so a full-sequence or absolute index includes the pads: slice
`[:, -len(ids):]`. One trace is one forward pass: N traces to fetch N layers is N
forward passes. Determinism: `do_sample=False` for `generate` and `pipe`, fixed
seeds, float32 on CPU. `model.scan(x)` gives shapes and dtypes with no compute.

| You want | Use | You get back |
|---|---|---|
| one forward pass | `model.trace(x)` | the model's output object |
| generated tokens | `model.generate(x, max_new_tokens=N)` | token ids on `tracer.result` |
| shapes, no compute | `model.scan(x)` | fake tensors, shapes and dtypes only |
| several traces sharing values | `model.session()` | values flow between traces |
| a permanent intervention | `model.edit(inplace=True)` | replayed on every later run |

## Pitfalls

1. `.save()` or it never existed. Assignments inside a trace body do not escape it;
   bind what you save (`x = ....save()`); a bare `....save()` on its own line returns
   nothing, and so does `float(x.save())`. There is no `.value` in 0.8: the saved
   variable *is* the tensor.
2. Access modules in forward-pass order. Reading layer 8 then layer 2 raises
   `OutOfOrderError`. Within a block, `ln_1`, `attn`, `mlp` come before the block's
   own `.output`; an edit at layer 0 goes above a read at layer 11.
3. Don't guess module paths or output types. `model.transformer.h[i]` is GPT-2;
   Llama is `model.model.layers[i]`. A block's `.output` is a plain tensor, but an
   attention submodule's is a tuple, so `.output[0]` copied from an old example
   silently selects batch row 0 on a block.
4. One trace is one forward pass: structure by input, not by activation.
5. Old nnsight code is everywhere and it is wrong here: `.value`, `nnsight.list()`,
   `tracer.next()`, `with tracer.all():`, `LanguageModel`, "proxies" are all pre-0.8.
6. Trace-body code *outside* the invokes runs before any of them; combine across
   invokes through a `nnsight.save({})` written from inside each one.
7. Inside a `tracer.iter` loop a wrong-order read binds to the *next* step instead
   of raising.

## Where to look

- https://nnsight.net (docs) and https://nnsight.net/llms.txt
- https://github.com/ndif-team/skills (maintainers' agent skills: `nnsight`,
  `activation-patching`, `attribution-patching`, `causal-tracing`, `logit-lens`, ...)
- `python scripts/inspect_model.py <repo_id> --prompt "..."` from that skills repo
  prints every block child in execution order and tensor-vs-tuple, without weights
- https://arxiv.org/abs/2407.14561 (paper)
