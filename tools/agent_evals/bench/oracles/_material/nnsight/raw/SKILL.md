---
name: nnsight
description: "Read, modify, and analyze the internals of neural networks with nnsight — tracing activations with model.trace, intervening on modules, batching interventions, gradients, caching, generation, and running on NDIF. Use for any task that touches model internals rather than just model outputs: interpretability experiments, activation extraction, ablation, patching, steering, probing, circuit analysis, or debugging existing nnsight code — and whenever you would otherwise reach for PyTorch forward hooks (register_forward_hook) to read or edit activations. Covers nnsight 0.8 (TransformersModel, tracer.result, no .value) — load this before writing any nnsight code, since idioms from older versions are still widespread and silently wrong."
---

# nnsight

nnsight reads, replaces and backpropagates through every intermediate value in a
model's forward pass, locally or on NDIF for models too large for your machine.
Techniques built on it have their own skills (listed at the bottom).

## Orientation

```python
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    clean = model.output.logits.save()                     # read the output

with model.trace(prompt):
    resid = model.transformer.h[5].output.save()           # read an activation
    model.transformer.h[8].output[:, -1, :] *= 2           # modify one
    edited = model.output.logits.save()                    # see what it changed

assert tuple(resid.shape) == (1, 10, 768)
assert model.tokenizer.decode(clean[0, -1].argmax()) == " Paris"
assert model.tokenizer.decode(edited[0, -1].argmax()) == " London"
```

Install: `pip install nnsight` (needs `torch` and `transformers`). Pick a device
with `device="cpu"` / `device="cuda:1"`, not `device_map=` — `device_map="cpu"`
still lands on the GPU; see the device table in
[references/modules-and-architectures.md](references/modules-and-architectures.md).

## The five things that break agent-written nnsight code

**1. `.save()` or it never existed.** Assignments inside a trace body do not escape
it — the body runs in a different frame. Bind what you save (`x = ....save()`; a
bare `....save()` on its own line returns nothing; neither does `float(x.save())`,
because the call around it still runs in the trace, so bind the saved value and
convert it after the block), and when collecting, save the
**container** and append raw values:

```python
with model.trace("The Eiffel Tower is in the city of"):
    per_layer = nnsight.save([])                  # save the list…
    for block in model.transformer.h:
        per_layer.append(block.output[0, -1])     # …append raw values

assert len(per_layer) == 12
```

There is no `.value` in 0.8 — the saved variable *is* the tensor. Saved
activations arrive with `requires_grad=True` and a live `grad_fn` that keeps the
forward graph alive: `.detach()` anything you keep or reuse (full story: the
`sae-and-dictionary-learning` skill and
[references/gradients.md](references/gradients.md)).

**2. Access modules in forward-pass order.** Reading layer 8 then layer 2 raises
`OutOfOrderError`: your code is a worker that parks until the model produces each
value, and the model has already gone past. Within a block, the submodules
(`ln_1`, `attn`, `mlp`) come before the block's own `.output`. This binds writes
too — an edit at layer 0 goes above a read at layer 11, not below it. Inside a
`tracer.iter` loop a wrong-order read binds to the *next* step instead of raising
— see [references/generation.md](references/generation.md).

**3. Don't guess module paths or output types.** `model.transformer.h[i]` is GPT-2;
Llama is `model.model.layers[i]`; Gemma-3 is `model.model.language_model.layers[i]`.
A block's `.output` is a plain tensor, but an attention submodule's is a tuple —
so `.output[0]` copied from an old example silently selects batch row 0. Run
`python scripts/inspect_model.py <repo_id> --prompt "..."`: without downloading
weights, it prints every block child **in execution order** and whether each
output is a tensor or a tuple.

**4. One trace is one forward pass — structure by input, not by activation.**
N traces to fetch N layers is N forward passes (remotely, N round-trips).

**5. Old nnsight code is everywhere and it is wrong here.** `.value`,
`nnsight.list()`, `tracer.next()`, `with tracer.all():`, `LanguageModel`,
"proxies" — all pre-0.8. If you are adapting code from a tutorial or a paper repo,
convert it first; see the `debugging` skill.

## Picking a run method

| You want | Use | You get back |
|---|---|---|
| one forward pass | `model.trace(x)` | the model's output object |
| generated tokens | `model.generate(x, max_new_tokens=N)` | token ids on `tracer.result` (the checkpoint's `generation_config` decides sampling — pass `do_sample=False`) |
| decoded text / labels | `model.pipe(x, ...)` | pipeline records (often sampled — pass `do_sample=False`) |
| shapes, no compute | `model.scan(x)` | fake tensors: shapes and dtypes only, no values (not on plain `NNsight`) |
| several traces sharing values | `model.session()` | values flow between traces |
| a permanent intervention | `model.edit(inplace=True)` | replayed on every later run |

## Core moves

**Batch a sweep into one pass.** Loop inside the trace, one `tracer.invoke` per
variant, no input on `trace()`:

```python
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris").input_ids[0]

with model.trace() as tracer:
    scores = nnsight.save({})                   # keyed: invokes finish in model order
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores[layer] = model.output.logits[0, -1, paris]

assert sorted(scores) == list(range(12))        # 12 ablations, one forward pass
```

Inside an invoke, index as if that input were alone — `[:, -1, :]` means "this
invoke's rows, last position". That holds for the last position only: inputs are
left-padded to the longest, so a full-sequence or absolute index includes the pads;
slice `[:, -len(ids):]` or see
[references/batching.md](references/batching.md#positions-shift-with-the-batch).
Trace-body code *outside* the invokes runs before
any of them, so it raises `NameError` on a name an invoke binds — combine across
invokes through a `nnsight.save({})` written from inside each one, as above.

**Grab many modules at once** with `tracer.cache(modules=[...])`, opened first thing
in the trace — see [references/caching-and-scan.md](references/caching-and-scan.md).

**Gradients** — capture in the forward, read in reverse order inside
`with metric.backward():`:

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        grad = hidden.grad.clone().save()
```

**Generation** — a `tracer.iter` loop must not ask for a step the run does not
make: one that outruns the run warns, keeps what it saved, and silently drops the
statements after the loop. `max_new_tokens` is only an upper bound, so pass
`min_new_tokens=` when the bound has to hold, and check `len()` of what you collected:

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:3]:
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

assert len(picks) == 3 and ids.shape[1] == 13
```

## Running it somewhere else

The same trace runs on NDIF against a model you can't fit locally — add
`remote=True` and reduce metrics *before* saving, since every `.save()` is a
download:

<!-- test: remote -->
```python
with model.trace(prompt, remote=True):
    last = model.transformer.h[-1].output[:, -1].detach().cpu().save()
```

Anything beyond one trace should be a `model.session(remote=True)` — one job
instead of N round-trips. Load the `remote` skill before writing remote code.

## References

| File | Covers |
|---|---|
| [references/execution-model.md](references/execution-model.md) | how tracing actually works: deferred execution, interleaving, ordering, `.save()` semantics. **Read this once.** |
| [references/access-and-modify.md](references/access-and-modify.md) | `.output` / `.input` / `.inputs`, in-place vs replacement, tuples, heads, positions, devices |
| [references/batching.md](references/batching.md) | `tracer.invoke`, empty invokes, sweeps, cross-invoke values, `tracer.barrier` |
| [references/generation.md](references/generation.md) | `generate` vs `pipe`, `tracer.iter`, per-step interventions, streaming, chat templates |
| [references/gradients.md](references/gradients.md) | `with tensor.backward():`, saliency, input×grad, gradient editing, optimizing through a frozen model |
| [references/caching-and-scan.md](references/caching-and-scan.md) | `tracer.cache()`, `model.scan()`, fake-tensor rules |
| [references/control-flow.md](references/control-flow.md) | `skip`, `tracer.stop`, `session`, `edit`, conditionals |
| [references/source-tracing.md](references/source-tracing.md) | `.source` — attention patterns and other values inside a forward |
| [references/modules-and-architectures.md](references/modules-and-architectures.md) | constructor and device placement, per-family module paths (MoE, SSM, hybrids included), `rename`, wrapping your own module, attaching SAEs/probes, custom served values with `eproperty` / `envoys=` |
| [references/api-reference.md](references/api-reference.md) | model classes and every method, property, config key, and exception in tables |

Scripts (run them, don't read them): `scripts/inspect_model.py <repo_id>
[--prompt P] [--grep attn]` — module paths, execution order, tensor-vs-tuple,
without weights; `scripts/check_env.py [--remote]` — versions, GPUs, NDIF key and
host, deployed models.

## Related skills

Techniques:

- `ablation` — remove a component (zero, mean, resample) to test necessity
- `activation-patching` — copy activations between runs to localize a behavior
- `attribution-patching` — gradient approximation of patching, for screening
- `causal-tracing` — ROME-style corrupt-and-restore to find where a fact lives
- `attention-analysis` — attention patterns, per-head metrics, head types
- `logit-lens` — decode intermediate residual streams through the unembedding
- `probing` — classifiers on activations: what is decodable, and where
- `circuit-discovery` — find and validate the subgraph a task uses
- `model-steering` — add directions to the residual stream at inference
- `model-editing-and-lora` — persistent weight/activation edits, ROME, LoRA
- `sae-and-dictionary-learning` — sparse autoencoders: attach, read, train
- `interp-experiment-design` — metrics, prompt sets, controls, baselines

Infrastructure:

- `remote` — running on NDIF: sessions, request batching, download size
- `debugging` — an error, a hang, an empty result, or pre-0.8 code to port
- `vllm` — nnsight inside the vLLM engine: throughput, serving, taps
- `tensor-parallel` — a model sharded across GPUs with transformers TP
- `quantization` — 4/8-bit loading and what it changes in a trace
- `diffusion-and-multimodal` — VLMs, diffusion pipelines, audio/vision encoders
