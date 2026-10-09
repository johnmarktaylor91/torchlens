# Performance: choosing a fast path

TorchLens has several ways to run a model, and they differ in cost by two orders of magnitude.
A full `tl.trace(...)` records every operation and pays a few milliseconds of Python work per
recorded op, so on a large decoder one trace takes tens of seconds. If you only need to steer
or patch the model, you do not need a trace at all. This page says which path to use for which
job, what each one costs, and what each one refuses. Everything here describes TorchLens 2.36.0,
except the last section, which describes the guarded fast steered rerun that follows it.
For the general speed knobs and benchmark tables, see the [performance guide](../performance.md).

## Pick a path

| You want to | Use | Cost, measured |
| --- | --- | --- |
| Steer or patch the model over many forwards, including generation | `tl.when(site, action).bind(model)`, or `steer_generate` for HF `generate()` | 1.0 to 1.2x a plain forward hook, exact |
| Steer and also keep a few activations as evidence, every step | `tl.record(model, x, save=..., intervene=spec, return_output=True)` | about 1 ms per op (CPU decoders); 9 s per step on Qwen3.5-9B |
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
spec = tl.when(tl.module("1"), tl.steer(direction, magnitude=4.0, feature_axis=-1))

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
assert bound.last_report.fire_count == 1  # the ledger of the most recent call
```

What to know about a binding:

- It changes neither the spec nor the model, and it is not an `nn.Module`. Module-style access
  (`bound.parameters()`, `bound.to(...)`) raises a teaching error; call the model itself for
  those.
- Each call settles a `BindReport` on `bound.last_report` (fire counts, resolved targets,
  cleanup verdict). By default a rule that never fires raises `bind_zero_fire` after the call;
  pass `on_zero_fire="disclose"` to record it in the report instead.
- Plain `tl.module(...)` sites run as real forward hooks; op-level sites such as `tl.func(...)`
  run through a torch function mode. The original computation always runs, and the edited value
  replaces it downstream.
- Calls are serial: do not call one binding from several threads at once.
- The refusal codes are listed under `bind_*` in the
  [error and refusal contract](../reference/error_refusal_contract.md).

### Generation with HF `generate()`

`bound.generate(...)` runs the base model's real `generate()` with the edits held across every
decoding step, KV cache included. `torchlens.intervention.steer_generate(model, ids, spec, ...)`
is the same thing in one call and returns the model's outputs together with the report. On
Qwen3.5-9B with the KV cache on, steered generation cost 1.07 to 1.10x a plain hook per token
(one run read 1.20x), and every step's scores matched the plain hook exactly.

```python
import torch
import torchlens as tl
from torchlens.intervention import steer_generate
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
    tl.module("model.layers.0.mlp"), tl.steer(direction, magnitude=8.0, feature_axis=-1)
)
gen_kwargs = {"max_new_tokens": 5, "do_sample": False, "use_cache": True, "pad_token_id": 0}

steered = spec.bind(lm).generate(ids, **gen_kwargs)
result = steer_generate(lm, ids, spec, **gen_kwargs)
assert torch.equal(result.outputs, steered)


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

## Steering with evidence: `tl.record`

When each step must also keep a few activations, use `tl.record` with `intervene=`. It applies
the same edits during a recording pass and keeps only the payloads `save=` selects, without
building the full graph a trace builds. `return_output=True` returns the model output alongside
the `Recording`. Call `recording.to_trace()` later only if you need graph structure.

```python
import torch
from torch import nn
import torchlens as tl


torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).eval()
x = torch.randn(2, 8)
direction = torch.randn(8)
spec = tl.when(tl.module("1"), tl.steer(direction, magnitude=4.0, feature_axis=-1))

output, recording = tl.record(
    model, x, save=tl.func("linear"), intervene=spec, return_output=True
)

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

## Cut the cost of a trace

A trace costs about 3.2 ms of host Python per recorded op with default settings, linear in the
op count with no meaningful fixed cost. The device does not change it: a GPU trace costs the
same as a CPU trace of the same model. So the levers are fewer ops and less work per op.
`trace.num_ops` tells you how many ops a forward recorded.

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

print(trace.num_ops)
relu = trace.find_sites(tl.func("relu")).first()
assert relu.out.shape == (2, 8)
```

- **Save selectors.** A function selector such as `save=tl.func("linear")` keeps only matching
  payloads and was the cheapest spelling measured (9.5 s against 10.1 s for saving everything,
  Qwen3.5 16 layers, 128 tokens, CPU).
- **Module save selectors cost more in 2.36.0.** A `tl.module(...)` save selector can cost more
  than saving everything: during capture every op output is held as a candidate and, past a
  64 MB in-memory budget, spilled to temporary files. In the same measurement
  `save=tl.module(...)` took 11.6 s and wrote about 0.84 GB to temporary disk to keep one tensor,
  and on Qwen3.5-9B a trace with a module selector took about 48 s against 34 s with a function
  selector. If you need one module's output from a large model, steer and read it with a forward
  hook, or select the op inside the module with `tl.func(...)`.
- **`CaptureOptions(inference_only=True)`** runs the forward under `torch.no_grad()`. It saved
  about 5% per trace in the measurement above.
- **Keep the wrappers installed.** `tl.trace` keeps torch wrapped between calls by default.
  `CaptureOptions(unwrap_when_done=True)` re-wraps on the next call, a fixed 2 to 5% per trace
  on the small models.
- **Stop early.** `halt=` ends the forward once the sites you need are captured; see the
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
loops. If a trace of such a model records far more ops than you expect, check `trace.num_ops`
and whether the package imports in your environment. With the KV cache on, each decoding step
processes one token, so the package matters much less for `bind` with `generate()`.

## Trace once, then rerun: not a fast path for steered generation in 2.36.0

Capturing one trace and re-running it for each new input looks like it should be cheap. In
2.36.0 it is not, for steered generation: every reuse path either refuses a staged intervention,
refuses a change in input length, or recaptures the whole forward.

| Reuse path | What it does | What it refuses or costs |
| --- | --- | --- |
| `trace.run(inputs=x)` | A fresh verified execution of the recorded program | Refuses `run_staged_spec_unapplied` when the trace carries a staged intervention, because it would not apply it |
| `trace.run(inputs=x, fast=True)` | A guarded static-loop refresh: the model's native forward with targeted hooks | Same refusal for a staged intervention. Refuses any change in input shape (`PathDivergenceError`), so a growing sequence fails at the second step. Needs a function selector (`save=tl.func(...)`) on the capture. Unsteered and at a fixed shape it is fast: 0.41 s on Qwen3.5-9B against 47 s for a trace |
| `trace.run(model, x)` on an intervened trace | Recaptures the whole forward with the stored intervention | About 7 to 9x the cost of a fresh trace (483 s for one step on Qwen3.5-9B). Repeated reruns of the same intervened trace also re-apply the staged edit more than once in 2.36.0, so later reruns are not exact. Do not use it for generation |
| `trace.fork().do(site, action)` | Replays an edit over the captured graph | Same input only, by design. Needs a `CaptureOptions(intervention_ready=True)` capture (about 5 to 6x a default trace), and refuses typed (`ReplayPreconditionError`) on some decoder architectures |
| `episode=` capture with `intervene=` | Captures a whole steered generation as one product | Exact, but diagnostic-tier cost (tens of steps, not hundreds), and `run()` refuses on the coupled product by design; see [episode capture](../reference/episode_capture.md) |

For steering across many inputs or through generation, use [`bind`](#steering-and-patching-over-many-forwards-bind).
When each step also needs recorded evidence, use [`tl.record`](#steering-with-evidence-tlrecord).
Use `tl.trace` when you want the full record of one forward, then work with that trace.

## Fast steered rerun

Releases that include the guarded fast steered rerun change two rows of the table above. On a
trace captured with an `intervene=` spec that targets plain module selectors (`tl.module(...)`),
`trace.run(model, x)` and `trace.run(inputs=x, fast=True)` no longer recapture: they run the
model's native forward with the staged hooks and refresh the saved sites. The input may grow
from step to step, as in generation, as long as the model makes the same sequence of torch calls
and module entries; a structural fingerprint sealed at capture checks that after every run.

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
site = tl.module("model.layers.1.mlp")
head = tl.module("lm_head")
spec = tl.when(site, tl.steer(direction, magnitude=4.0, feature_axis=-1))

trace = tl.trace(lm, ids, save=site | head, intervene=spec)
bound = spec.bind(lm)
for _ in range(3):
    trace.run(lm, ids)  # the steered forward on the longer input, saved sites refreshed
    assert trace.last_run["engine"] == "guarded_fast"
    assert trace.last_run["fast_refused"] is None
    logits = trace.find_sites(head).first().out
    assert torch.equal(logits, bound(ids).logits)
    ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], dim=-1)
```

- **Cost.** On Qwen3.5-9B a steered generation step cost about 2x a plain hook: 0.36 s against
  0.18 s, and 0.15 s against 0.075 s with flash-linear-attention, exact at every step with no
  fallback. On the small CPU decoders it cost 1.4 to 1.9x. `bind` stays the cheapest steering
  path (1.07 to 1.17x on the same 9B runs); the rerun costs about one more forward and in
  exchange refreshes the saved activations each step. `trace.run(inputs=x, fast=True)` keeps its
  session between calls and was 15 to 20% cheaper than `trace.run(model, x)`.
- **Fallback.** When a guard refuses, `trace.run(model, x)` falls back to the capture engine,
  which still gives the right answer, and records why. `last_run["engine"]` is `"guarded_fast"`
  when the fast engine ran and `"rerun"` after a fallback, and `last_run["fast_refused"]` names
  the refusing guard as `"<code>:<stage>"` (it is `None` on a fast run). The explicit
  `trace.run(inputs=x, fast=True)` raises instead of falling back.
- **What still takes the capture engine.** Value replacements (`set()`) and non-module targets
  (`fast_rerun_target_unsupported`), hooks attached after a plain capture, which recapture once
  to record their firing and are eligible afterwards (`fast_rerun_graph_unsteered`), and any
  change in the model's call structure (`fast_live_call_fingerprint`).
- **What a fast run does not change.** The save scope never widens, the stored spec is unchanged,
  and op metadata the run did not refresh (shapes and activation memory of unsaved ops) reads
  `None` after a run on a different input size, rather than showing capture-time values.
