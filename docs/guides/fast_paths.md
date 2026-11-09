# Performance: choosing a fast path

TorchLens has several ways to run a model, and they differ in cost by two orders of magnitude.
A full `tl.trace(...)` records every operation and pays a few milliseconds of Python work per
recorded op, so on a large decoder one trace takes tens of seconds. If you only need to steer
or patch the model, you do not need a trace at all. This page says which path to use for which
job, what each one costs, and what each one refuses.
For the general speed knobs and benchmark tables, see the [performance guide](../performance.md).

## Pick a path

| You want to | Use | Cost, measured |
| --- | --- | --- |
| Steer or patch the model over many forwards, including generation | `tl.when(site, action).bind(model)`, and `.bind(model).generate(...)` for HF `generate()` | 1.0 to 1.2x a plain forward hook, exact |
| Steer and also keep a few activations as evidence, every step | `tl.record(model, x, save=..., intervene=spec)`, output in `recording.output` | about 1 ms per op (CPU decoders); 9 s per step on Qwen3.5-9B |
| Steer and refresh a trace's saved activations on each new input, including generation | trace once with `intervene=`, then `trace = trace.run(model, x)` per input (each run returns a new Trace) | about 1.5 to 2x a plain forward hook per step, exact; see [Fast steered rerun](#fast-steered-rerun) |
| Read one module's output, or a few | `tl.pluck(model, x, "block.3")` for one, `tl.extract(model, x, [...])` for several; `stop_after="block.3"` on `pluck` also skips the rest of the model | 0.98x a plain forward, and 0.18x with `stop_after=` at the third of GPT-2's twelve blocks, when every name is an exact module address, the model is in eval mode and the input is one tensor or text on the model's device; otherwise a full trace (11.5x). See [One module's output](#one-modules-output-tlpluck) |
| Inspect the full graph, metadata and activations of one forward | `tl.trace(model, x, save=...)` | about 3 ms per op; 47 s for one Qwen3.5-9B trace |
| Make one trace cheaper | a `tl.func(...)` save selector, `CaptureOptions(inference_only=True)`, and for Qwen3.5-class models flash-linear-attention | see [Cut the cost of a trace](#cut-the-cost-of-a-trace) |
| Final outputs only | the plain model call | no TorchLens cost |

"Exact" means the steered outputs matched a plain `register_forward_hook` doing the same edit:
maximum absolute difference 0.0 and identical greedy tokens. Measurements were taken on Qwen3
and Qwen3.5 decoders (8 to 16 layers, random init, fp32, one CPU thread) and on Qwen3.5-9B
(bf16, eager attention, one H200 GPU). Your numbers will differ with the model and host; the
ratios are what carries over.

## Steering and patching over many forwards: `bind`

`spec.bind(model)` turns an intervention spec into a capture-free executor. Calling the binding
runs the model's own forward with the edits applied and returns exactly what the model returns.
Nothing is recorded, so the cost is close to a hand-written hook: 1.0 to 1.2x a plain forward
hook on every model tested, and 1.02x per greedy step on Qwen3.5-9B (0.22 s against about 64 to
82 s for a fresh `tl.trace` per step). The first call carries a one-time setup of about 0.6 to
0.9 s.

```python
import torch
from torch import nn
import torchlens as tl


torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
direction = torch.randn(8)
spec = tl.when(tl.module("1"), tl.steer(direction, strength=4.0, dim=-1))

# Build the binding once and reuse it for every forward.
bound = spec.bind(model)
inputs = [torch.randn(2, 8) for _ in range(3)]
steered = [bound(x) for x in inputs]


# The same edit as a plain forward hook, for comparison.
def add_direction(module, args, output):
    return output + direction * 4.0


handle = model[1].register_forward_hook(add_direction)
try:
    reference = [model(x) for x in inputs]
finally:
    handle.remove()

assert all(torch.equal(a, b) for a, b in zip(steered, reference))
assert bound.last_report.num_fires == 1  # the ledger of the most recent call
```

What to know about a binding:

- It changes neither the spec nor the model, and it is not an `nn.Module`. Module-style access
  (`bound.parameters()`, `bound.to(...)`) raises a teaching error; call the model itself for
  those.
- Each call settles a `BindReport` on `bound.last_report` (fire counts, resolved targets,
  cleanup verdict). A rule that never fires is recorded there and, by default
  (`on_zero_fire="warn"`), warns `bind_zero_fire` after the call; pass `on_zero_fire="error"` to
  refuse instead or `on_zero_fire="disclose"` to record without warning. The same knob is on
  `tl.trace(intervene=)` and `Trace.run`.
- Plain `tl.module(...)` sites run as real forward hooks; op-level sites such as `tl.func(...)`
  run through a torch function mode. The original computation always runs, and the edited value
  replaces it downstream.
- Calls are serial: do not call one binding from several threads at once.
- The refusal codes are listed under `bind_*` in the
  [error and refusal contract](../reference/error_refusal_contract.md).

### Generation with HF `generate()`

`bound.generate(...)` runs the base model's real `generate()` with the edits held across every
decoding step, KV cache included, and returns the model's own outputs; the fire counts are on
`bound.last_report`. On
Qwen3.5-9B with the KV cache on, steered generation cost 1.07 to 1.10x a plain hook per token
(one run read 1.20x), and every step's scores matched the plain hook exactly.

```python
import torch
import torchlens as tl
from transformers import LlamaConfig, LlamaForCausalLM


torch.manual_seed(0)
config = LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)
lm = LlamaForCausalLM(config).eval()
ids = torch.randint(0, 128, (1, 6))
direction = torch.randn(32)
spec = tl.when(
    tl.module("model.layers.0.mlp"), tl.steer(direction, strength=8.0, dim=-1)
)
gen_kwargs = {"max_new_tokens": 5, "do_sample": False, "use_cache": True, "pad_token_id": 0}

bound = spec.bind(lm)
steered = bound.generate(ids, **gen_kwargs)
assert bound.last_report is not None


def add_direction(module, args, output):
    return output + direction * 8.0


handle = lm.model.layers[0].mlp.register_forward_hook(add_direction)
try:
    reference = lm.generate(ids, **gen_kwargs)
finally:
    handle.remove()

assert torch.equal(steered, reference)
```

If your generation loop is your own code rather than `generate()`, call the binding once per
step (`bound(ids)`); the binding is reused across steps and the input may grow.

To steer only on the steps where a condition holds, with a per-token audit of when the steer was
applied, see the [conditional steering recipe](../field-guide/conditional_steering.md).

A bound `generate` never traces, so it records nothing beyond the fire counts in
`bound.last_report`. To record what generation computed, capture the whole run
as an episode (`tl.trace(..., episode=tl.options.EpisodeOptions(...))`), which is diagnostic tier:
its cost grows faster than the number of steps, so plan for tens of steps, never hundreds
([episode capture](../reference/episode_capture.md)). To edit what the KV cache holds for the
prompt rather than every step, see [the KV-cache recipe](kv_cache.md).

## Steering with evidence: `tl.record`

When each step must also keep a few activations, use `tl.record` with `intervene=`. It applies
the same edits during a recording pass and keeps only the payloads `save=` selects, without
building the full graph a trace builds. `recording.output` is the model output, exactly as the
model returned it. Call `recording.to_trace()` later only if you need graph structure: a
`Recording` has no `draw()` and no facets, and the cooked `Trace` takes every `Trace.draw`
option and groups a reused layer into passes the way `tl.trace` does. Episode capture stays on
`tl.trace(..., episode=...)`; `tl.record(..., episode=...)` refuses with the typed code
`record_episode_unsupported`.

```python
import torch
from torch import nn
import torchlens as tl


torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)
direction = torch.randn(8)
spec = tl.when(tl.module("1"), tl.steer(direction, strength=4.0, dim=-1))

recording = tl.record(model, x, save=tl.func("linear"), intervene=spec)
output = recording.output

assert torch.equal(output, spec.bind(model)(x))  # the same steered output as bind
print(recording.n_records)
```

Cost and caveats:

- On the tested CPU decoders `tl.record(intervene=...)` cost about 1.0 ms of host Python per
  recorded op, against about 3.2 ms per op for a default `tl.trace`. On Qwen3.5-9B it was about
  0.5 ms per op against about 2.6 ms: 9.1 s per greedy step against 64 to 82 s for a fresh trace
  per step, and 1.9 s per step with flash-linear-attention installed.
- It is still per-op work, so it scales with the op count; for steering alone, `bind` is two
  orders of magnitude cheaper.
- `tl.record` is torch-only.
- It attaches backward hooks to output tensors even when nothing asked for gradients. Running
  the call under `torch.no_grad()` made it cheaper in one indicative measurement (1.1 s against
  1.7 s) and stayed exact.

## One module's output: `tl.pluck`

`tl.pluck(model, x, name)` returns one module's output. When `name` is an exact module address
(`"transformer.h.2"`, `"layer1.0"`), the model is in eval mode and the input is one tensor or
text on the model's device, it runs no capture: one plain forward with a forward hook on that
module returns the same tensor a trace would. With `stop_after=` set to an exact module address
as well, a hook ends the forward right after that module's first call, so the layers after it
never run and the call can cost less than a plain forward. `tl.extract` takes the same hooked
path for several module addresses at once.

| `tl.pluck(model, ids, "transformer.h.2")` on GPT-2 small | Median, CPU | vs plain forward |
| --- | ---: | ---: |
| plain forward | 151.5 ms | 1.00x |
| `pluck` | 148.2 ms | 0.98x |
| `pluck` with `stop_after="transformer.h.2"` | 27.6 ms | 0.18x |
| the same request through a full capture | 1,739.9 ms | 11.5x |
| the same request through a full capture, with `stop_after=` | 515.0 ms | 3.4x |

Measured 8 Nov 2026 on pretrained GPT-2 small, 4 sequences of 32 tokens, CPU with 4 threads,
grad mode on, median of 7 calls (3 for the capture rows); the fast results equal the captured
ones bit for bit. Under `torch.no_grad()` the ratios were 0.98x, 0.20x, 12.7x and 3.8x.

```python
import torch
from torch import nn
import torchlens as tl

model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
model = model.eval()
x = torch.randn(4, 8)

hidden = tl.pluck(model, x, "1")  # one forward with a hook on module "1"
early = tl.pluck(model, x, "1", stop_after="1")  # modules "2" to "4" never run
assert torch.equal(hidden, early)
```

Anything else runs the full capture and returns what it always did: an op label such as
`"relu_1_2"`, a pass-qualified or partial name, a model in training mode, a list or dict of
inputs, a module that runs twice or returns a tuple, a `stop_after=` that is a selector, a
callable, or a bare name that could also be a function name (`"relu"`), and an active
`tl.experimental.stop_after` block. The returned tensor keeps its autograd graph when grad mode
is on, as the traced result does.

## Cut the cost of a trace

A trace costs about 3.2 ms of host Python per recorded op with default settings, linear in the
op count with no meaningful fixed cost. The device does not change it: a GPU trace costs the
same as a CPU trace of the same model. So the levers are fewer ops and less work per op.
`len(trace.ops)` tells you how many ops a forward recorded.

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.options import CaptureOptions


model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)

trace = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    capture=CaptureOptions(inference_only=True),
)

print(len(trace.ops))
relu = trace.find_sites(tl.func("relu")).first()
assert relu.out.shape == (2, 8)
```

- **Save selectors.** A function selector such as `save=tl.func("linear")` keeps only matching
  payloads and was the cheapest spelling measured (9.5 s against 10.1 s for saving everything,
  Qwen3.5 16 layers, 128 tokens, CPU).
- **Module save selectors are live-retained.** A `save=tl.module(...)` selector, alone or joined
  with other module paths by `|`, copies only the ops that run inside a matching module pass and
  keeps only that module's outputs, so it costs about what saving everything costs and writes
  nothing to temporary disk (9.0 s against 9.7 s for saving everything, Qwen3.5 16 layers, 128
  tokens, CPU, `inference_only=True`). A module that wraps most of the model still copies most
  ops. A selector that mixes `tl.module(...)` with other terms (`&`, `~`, or `|` with
  `tl.func(...)`) still holds every op output as a candidate and spills to temporary files past
  a 64 MB budget; the address preflight (a private capture helper with no public spelling) reports it as deferred.
- **`CaptureOptions(inference_only=True)`** runs the forward under `torch.no_grad()`. It saved
  about 5% per trace in the measurement above. A capture entered inside `torch.no_grad()` or
  `torch.inference_mode()` implies it unless a backward option (`backward_ready`, `save_grads`)
  asks for one; `trace.options.inference_only` reports the value in force.
- **`tl.record` skips gradient hooks by default.** It installs them only when
  `backward_ready=True`, `save_grads=` or `default_grad=` is given, so pass one of those before a
  `Recording.backward()`, which refuses otherwise.
- **Keep the wrappers installed.** `tl.trace` keeps torch wrapped between calls by default.
  `CaptureOptions(unwrap_when_done=True)` re-wraps on the next call, a fixed 2 to 5% per trace
  on the small models.
- **Stop early.** `stop_after=` ends the forward once the sites you need are captured; see the
  [performance guide](../performance.md).

### Install flash-linear-attention for Qwen3.5-class models

Qwen3.5 and other gated-delta-net models use linear-attention layers. When the
`flash-linear-attention` package is not installed, transformers runs those layers through a
reference PyTorch implementation written as Python loops, and every loop step is ops that
TorchLens records. On Qwen3.5-9B (22 tokens), installing it took one forward from 18,624 ops to
3,096 ops, one trace from about 47 s to about 9 s, and the plain forward from about 0.21 s to
0.04 to 0.07 s.

```bash
pip install flash-linear-attention
```

It provides GPU (Triton) kernels, so it helps on CUDA hosts; CPU-only runs keep the reference
loops. If a trace of such a model records far more ops than you expect, check `len(trace.ops)`
and whether the package imports in your environment. With the KV cache on, each decoding step
processes one token, so the package matters much less for `bind` with `generate()`.

## Trace once, then rerun

Capturing one trace and re-running it for each new input is cheap when the capture saved a
selection of sites (`save=tl.module(...)` or `tl.func(...)`): `trace.run(model, x)` and
`trace.run_transaction(inputs=x, fast=True)` then take the [guarded fast rerun](#fast-steered-rerun),
edited or not, and a staged intervention rides along when it targets plain module selectors
(`tl.module(...)`). A trace that saved every activation recaptures on `trace.run`: its value is
every op's record, and only a capture records ops; a native forward with hooks refreshes saved
sites. The other reuse paths apply a staged intervention through a full capture, refuse a change
in input, or replay over the captured graph.

| Reuse path | What it does | What it refuses or costs |
| --- | --- | --- |
| `trace.run_transaction(inputs=x)` | A fresh verified execution of the recorded program; with a staged intervention, a refresh capture that applies it (`last_run["engine"] == "live"`, fire receipts on `last_run`) | A full capture, not a native forward, and the verdict is `unverifiable` when staged edits applied; refuses `run_staged_spec_unapplied` only on a loaded artifact |
| `trace.run_transaction(inputs=x, fast=True)` | A guarded static-loop refresh: the model's native forward with targeted hooks | Applies a staged module-selector intervention through the guarded fast engine and admits a growing input while the model's call structure is unchanged; any refusal raises (`PathDivergenceError` for a structural change) instead of falling back. Needs a selective `save=` on the capture (`tl.module(...)` or `tl.func(...)`); a save-everything capture refuses. Unsteered and at a fixed shape: 0.41 s on Qwen3.5-9B against 47 s for a trace |
| `trace.run(model, x)` on a trace that saved a selection | The guarded fast rerun: the native forward with the staged hooks, if any, writing the saved sites into a new Trace | About 1.5 to 2x a plain hook per step, exact, and exact on every repeat, with or without a staged edit. When a guard refuses (a changed call structure, a value replacement, a non-module target) it falls back to recapturing the whole forward with the stored intervention: still exact, at the cost of a trace, with the reason in `last_run["fast_refused"]`. A save-everything trace with nothing staged always recaptures (`engine` `rerun`, `fast_refused` None) |
| `trace.fork().do(site, action)` | Replays an edit over the captured graph | Same input only, by design. Works on every `tl.trace` capture (the per-op argument templates it needs are always recorded; `replayable=True` is for runnable saves and costs about 5 to 6x a default trace), and refuses typed (`ReplayPreconditionError`) on some decoder architectures |
| `episode=` capture with `intervene=` | Captures a whole steered generation as one product | Exact, but diagnostic-tier cost (tens of steps, not hundreds), and `run()` refuses on the coupled product by design; see [episode capture](../reference/episode_capture.md) |

For steering alone, across many inputs or through generation, [`bind`](#steering-and-patching-over-many-forwards-bind)
stays the cheapest path. When each step also needs recorded evidence, use
[`tl.record`](#steering-with-evidence-tlrecord) or the fast steered rerun below. Use `tl.trace`
when you want the full record of one forward, then work with that trace.

## Fast steered rerun

On a trace that saved a selection of sites, `trace.run(model, x)` and
`trace.run_transaction(inputs=x, fast=True)` do not recapture: they run the model's native forward
and refresh the saved sites, with the staged hooks fired at their modules when the capture (or a
later `do`) staged an `intervene=` spec that targets plain module selectors (`tl.module(...)`). An
unedited rerun is the same engine with no hooks staged. The input may grow
from step to step, as in generation, as long as the model makes the same sequence of torch calls
and module entries; a structural fingerprint sealed at capture checks that after every run.

```python
import torch
from torch import nn
import torchlens as tl
from transformers import LlamaConfig, LlamaForCausalLM


torch.manual_seed(0)
config = LlamaConfig(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
)
model = LlamaForCausalLM(config).eval()  # the stock forward, KV cache returned
ids = torch.randint(0, 128, (1, 6))
direction = torch.randn(32)
site = tl.module("model.layers.1.mlp")
head = tl.module("lm_head")
spec = tl.when(site, tl.steer(direction, strength=4.0, dim=-1))

trace = tl.trace(model, ids, save=site | head, intervene=spec)
bound = spec.bind(model)
for _ in range(3):
    trace = trace.run(model, ids)  # the steered forward on the longer input, as a new Trace
    assert trace.last_run["engine"] == "guarded_fast"
    assert trace.last_run["fast_refused"] is None
    logits = trace.find_sites(head).first().out[:, -1]
    assert torch.equal(logits, bound(ids).logits[:, -1])
    ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=-1)
```

The traced model can be the stock Hugging Face model, as here: its output is a `ModelOutput`
carrying the KV cache, and the fast engine pairs every output tensor (the logits and each cached
key and value) with the captured output ops by container path. A wrapper that returns a plain
tensor, such as `lm(ids, use_cache=False).logits[:, -1]`, also reruns fast and records fewer
output ops, but it is not required. A changed number of output tensors still refuses
(`output_structure_mismatch:fast_live_model_output_structure`).

- **Cost.** On Qwen3.5-9B, tracing the stock forward with its KV cache returned, a steered
  generation step cost about 2x a plain hook: 0.35 s against 0.17 to 0.18 s, and 0.09 s against
  0.05 s with flash-linear-attention. All 56 measured steps took the fast engine, with greedy
  tokens identical to the hook and a maximum absolute difference of 0.0. On the small CPU
  decoders it cost 1.4 to 1.9x. `bind` stays the cheapest steering path (1.07 to 1.17x on the
  9B); the rerun costs about one more forward and in exchange refreshes the saved activations
  each step. `trace.run_transaction(inputs=x, fast=True)` keeps its session between calls and cost 1.5 to
  1.6x the hook on the 9B without flash-linear-attention, about 20% less than
  `trace.run(model, x)`.
- **Fallback.** When a guard refuses, `trace.run(model, x)` falls back to the capture engine,
  which still gives the right answer, and records why. `last_run["engine"]` is `"guarded_fast"`
  when the fast engine ran and `"rerun"` after a fallback, and `last_run["fast_refused"]` names
  the refusing guard as `"<code>:<stage>"` (it is `None` on a fast run, and `None` on a
  save-everything trace with nothing staged, where the fast engine is not tried). The explicit
  `trace.run_transaction(inputs=x, fast=True)` raises instead of falling back.
- **Early stop.** `trace.run_transaction(inputs=x, fast=True, until=<a saved module>)` ends the
  native forward after that module; `trace.run(model, x)` has no stop argument and runs the
  whole forward.
- **What still takes the capture engine.** Value replacements (`set()`) and non-module targets
  (`fast_rerun_target_unsupported`), hooks attached after a plain capture or cleared after a
  steered one, which recapture once to record the current plan and are eligible afterwards
  (`fast_rerun_graph_unsteered`), and any
  change in the model's call structure (`fast_live_call_fingerprint`).
- **What a fast run does not change.** The save scope never widens, the stored spec is unchanged,
  and op metadata the run did not refresh (shapes and activation memory of unsaved ops) reads
  `None` after a run on a different input size, rather than showing capture-time values.
- **Changed steering tensors refuse.** `tl.steer` keeps your tensor by reference. If you change
  it in place after staging, both doors refuse `helper_tensor_changed_since_capture`, as the
  capture rerun does; re-stage the steer after each update instead.
