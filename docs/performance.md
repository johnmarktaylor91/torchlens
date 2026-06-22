# Performance guide

TorchLens is fastest when it captures only the payloads you plan to inspect. A full
`tl.trace(model, x)` records a complete operation graph and saves activations for the selected
sites; `tl.record(model, x, save=...)` is the lighter path for tight loops where you only need
selected records and can materialize a `Trace` later.

Backend note: these timings and sparse-recording examples are torch-oriented. `tl.record()` /
fastlog is torch-only in the backend-v1 registry. Non-torch preview backends may have different
capture costs; JAX, tinygrad, Paddle, and TensorFlow `.tlspec` saves materialize array payloads but
loaded traces cannot replay-validate stripped runtime captures. Paddle preview capture has its own
dygraph/eager replay and static-inventory audit costs; TensorFlow preview capture runs live eager
`op_callbacks` plus self-consistency and per-op replay/perturbation accounting.

## Decision tree

| Need | Use | Notes |
| --- | --- | --- |
| Complete graph metadata and a few activations | `tl.trace(model, x, save=predicate)` | Best default for debugging and one-off analysis. |
| Repeated activation pulls in a loop | `tl.record(model, x, save=predicate)` | Torch-only lower-overhead path; call `Recording.to_trace()` only when you need graph structure. |
| A local window around a later op | `tl.trace(..., save=tl.followed_by(...), lookback=K)` | Retains bounded recent metadata, and optionally bounded recent payloads. |
| Disk-backed selected payloads | `tl.trace(..., storage=tl.to_disk(path))` | Keeps selected payloads portable without retaining them all in RAM. |
| Intervention during the forward pass | `tl.trace(..., intervene=tl.when(...), save=...)` | Live edits cost more than passive capture; use only when the model must execute edited values. |
| Final logits only | plain `model(x)` | TorchLens adds wrapper dispatch and metadata work; skip it when no intermediate data is needed. |

## Fast activation pull

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.options import CaptureOptions


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

trace = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    capture=CaptureOptions(save_code_context=False),
)
relu_site = trace.find_sites(tl.func("relu")).first()
relu_activation = relu_site.out

assert relu_activation.shape == (2, 4)
```

Use `save=tl.func(...)`, `save=tl.in_module(...)`, or composed predicates such as
`tl.func("conv2d") & tl.followed_by(tl.func("relu"))` instead of saving every payload. The graph
metadata remains available, but unsaved payloads are intentionally absent.

## Sparse recording loop

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

recording = tl.record(model, x, save=tl.func("gelu"))
trace = recording.to_trace()
gelu_site = trace.find_sites(tl.func("gelu")).first()

assert gelu_site.out.shape == (2, 4)
```

`tl.record(save=...)` is the canonical torch sparse-capture spelling. `record(keep_op=...)` and
`record(keep_module=...)` remain deprecated aliases for older code.

When a fastlog forward raises, the default remains `on_forward_error="raise"`. Opt into
`on_forward_error="attach_partial"` to attach `exc.partial_recording` and re-raise, or
`on_forward_error="return_partial"` to return a failed partial `Recording` (`return_output=True`
returns `(None, partial)`). Failed partials set `status="partial_error"`, `failed=True`,
string-only error metadata, `n_ops_completed`, and best-effort `last_event_*` fields. user-op
failures exclude the failing call; TL-side capture failures may include a skipped/partial
current-call event. Failed partials cannot be converted with `Recording.to_trace()` or used with
`Recording.log_backward()`. Full `tl.trace(...)` failures expose `exc.partial_log`, recoverable
with `tl.partial.from_failed_capture(exc)`.

## Speed knobs

| Knob | Faster setting | Tradeoff |
| --- | --- | --- |
| Payload selection | `save=tl.func(...)` or `save=tl.in_module(...)` | Unsaved activations cannot be read later. |
| Source text | `capture=CaptureOptions(save_code_context=False)` | File/line identity remains, but source text is not loaded. |
| Window payloads | `lookback_payload_policy="metadata_only"` | `tl.followed_by(...)` can select metadata without retaining raw tensors. |
| Retroactive payloads | `lookback_payload_policy="detached_raw"` | Enables payload recovery for recent matched ops at bounded memory cost. |
| Disk storage | `storage=tl.to_disk(path)` | Reduces RAM pressure; disk I/O becomes part of capture cost. |
| Gradients | `save_grads=False` unless needed | Backward-ready captures preserve more state and hooks. |
| Forward-only autograd | `inference_only=True` | Runs forward capture under `torch.no_grad()`; incompatible with backward capture. |
| Forward chunking | `chunk_size=N` | Reduces forward-pass peak memory for single-batch tensor inputs; final saved activations are still accumulated in memory. |
| Visualization | Call `trace.draw()` after capture, not during hot loops | Rendering is separate from activation collection. |

### Phase timing buckets

`trace._phase_timings` groups wall-clock timings by stable bucket names. Capture buckets include
`ctx_build:*`, `dispatch:*`, `clone_save:*`, and `object_construction:op`. Postprocess buckets use
`postprocess:Step N: ...`, matching the numbered postprocess pipeline. Graphviz rendering records
`render:graphviz:forward`, `render:graphviz:backward`, or `render:graphviz:combined` when those
render entrypoints run.

## Chunked forward capture

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU()).eval()
x = torch.randn(128, 1024)

trace = tl.trace(model, x, chunk_size=16, save=tl.func("relu"))

assert trace.find_sites(tl.func("relu")).first().out.shape[0] == 128
```

`chunk_size=` is forward-only sugar over a first `trace(...)` followed by
`rerun(..., append=True)` for later chunks. It splits selected positional tensor leaves along
dimension 0, executes one sub-batch at a time, and returns one accumulated in-memory `Trace`.

The v1 limits are intentionally narrow: torch backend only, positional inputs only, no
`backward_ready`, no `save_grads`, no live `hooks=`, no public `intervene=`, and no
`storage=tl.to_disk(...)` or `save_outs_to`. Loaded or live chunked traces also reject
`log_backward()` because they do not retain one full-batch autograd graph.

Auto mode splits only when there is exactly one `ndim > 0` tensor leaf under standard Python
containers (`list`, `tuple`, `dict`, or `namedtuple`). If there are several candidates, pass
explicit paths:

```python
import torch
from torch import nn
import torchlens as tl


class MaskedModel(nn.Module):
    def forward(self, tokens, attention_mask):
        return (tokens * attention_mask).sum(dim=-1)


model = MaskedModel().eval()
tokens = torch.randn(64, 16)
attention_mask = torch.ones(64, 16)

trace = tl.trace(model, (tokens, attention_mask), chunk_size=8, chunk_paths=["0", "1"])
```

Unlisted leaves are passed unchanged to every chunk, which is useful for shared masks or bias
tables. The memory contract is forward-pass peak only: final saved activations are concatenated
and retained according to `save=`, and preprocessing still sees the full batch if `transform=`
materializes it. Disk-backed chunk accumulation is a future item.

For activation extraction without a `Trace`, use `tl.batched_extract(...)`; that path returns
tensors or `.pt` files rather than accumulated graph metadata. `chunk_size=` covers the remaining
"dataloader wrapper" case for stacked multi-pass trace capture.

## Windowed and disk-backed capture

```python
from pathlib import Path

import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.ReLU()).eval()
x = torch.randn(1, 1, 5, 5)
path = Path(DOCS_TMPDIR) / "windowed.tlspec"

predicate = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
trace = tl.trace(
    model,
    x,
    save=predicate,
    lookback=4,
    lookback_payload_policy="detached_raw",
    storage=tl.to_disk(path),
)

assert trace.find_sites(tl.func("conv2d")).first().out.shape == (1, 2, 3, 3)
```

Use disk-backed storage for selected payloads that are too large or numerous to keep in memory.
Portable `.tlspec/` bundles store manifest data plus tensor sidecars when the backend supports
materialized payloads; executable Python callables are not portable. Backend-aware manifest schema
v2 adds `backend`, `backend_runtime`, nullable torch-specific fields, and `payload_policy`.
JAX, tinygrad, Paddle, and TensorFlow preview bundles materialize array payloads; loaded traces
still report replay validation as unavailable because portable save strips runtime replay captures.

## Intervention cost

```python
import torch
from torch import nn
import torchlens as tl


model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
x = torch.randn(2, 4)

patched = tl.trace(
    model,
    x,
    save=tl.func("relu"),
    intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
)

assert torch.count_nonzero(patched.find_sites(tl.func("relu")).first().out) == 0
```

Use `intervene=` when the edited value must affect downstream execution. For post-hoc experiments
on an existing trace, prefer `trace.fork()`, `set(...)` or `attach_hooks(...)`, and `replay()`.

## When another tool is faster

Use raw PyTorch when you only need the final output. Use `torch.profiler` when the question is
kernel timing rather than activation provenance. Use TransformerLens when your workflow is entirely
inside its supported transformer families and you already want its named hook points. Use Captum or
Inseq when you want mature attribution algorithms out of the box rather than the lower-level
activation and gradient substrate.

## Measured numbers (canonical bench host)

Measured on the canonical bench host (`zmachine`, Intel i9-9900X) at SHA `712a4e5` on
`2026-06-16`, from a full non-smoke run (`baseline_status: "canonical"`). The complete 196-row
table (CPU + CUDA, every model and row) lives in
[`docs/_perf_numbers.md`](_perf_numbers.md) and the raw baseline in
`benchmarks/perf_baselines/linux-cpu.json`.

**Headline: with fastlog capture and an early halt at ~25% depth, capture runs _faster than the
raw forward pass itself_ — `fastlog_halt_25` is 0.84x raw forward on ResNet-18 (CPU) and 0.83x on
GPT-2 (HookedTransformer, CPU).** You only pay for the layers you actually reach.

Representative CPU rows ("vs raw" = multiple of the raw forward pass):

| Model | Row | Median ms | vs raw forward |
|---|---|---:|---:|
| resnet18 | raw_forward | 72.2 | 1.00x |
| resnet18 | tl_trace (full capture) | 994.6 | 13.78x |
| resnet18 | fastlog_zero (predicate false) | 157.5 | 2.18x |
| resnet18 | fastlog_halt_25 | 60.8 | **0.84x** |
| gpt2_hf | raw_forward | 130.4 | 1.00x |
| gpt2_hf | tl_trace (full capture) | 1927.0 | 14.77x |
| gpt2_hf | fastlog_zero (predicate false) | 506.0 | 3.88x |
| gpt2_hf | fastlog_halt_25 | 134.7 | 1.03x |
| gpt2_hooked | raw_forward | 343.4 | 1.00x |
| gpt2_hooked | tl_trace (full capture) | 5048.5 | 14.70x |
| gpt2_hooked | fastlog_halt_25 | 283.6 | **0.83x** |

(Full exhaustive capture (`tl_trace`) costs ~14x the forward and amortizes on large models; the
`fastlog_*` rows show selective / early-exit capture, where halting at 25% depth drops below the
raw-forward cost. TinyNet ratios are dominated by fixed per-capture overhead — see the full table.)

## Re-capturing baselines (canonical host)

```bash
# Full suite on the canonical bench host (quiet machine):
python -m benchmarks.perf_suite --rerun --baseline-status canonical \
  --out-json benchmarks/perf_baselines/<host>-<device>.json \
  --out-md benchmarks/perf_results_<date>.md
# Sanity self-compare:
python -m benchmarks.perf_gate --baseline benchmarks/perf_baselines/<host>-<device>.json \
  --current benchmarks/perf_baselines/<host>-<device>.json   # expect "passed": true
# Regenerate the docs table:
python -m benchmarks.generate_perf_numbers benchmarks/perf_baselines/<host>-<device>.json \
  --out docs/_perf_numbers_provisional.md
```

On the canonical host, drop the `-provisional` filename suffix; `--baseline-status canonical`
requires a full non-smoke, non-addendum run and emits generated speed headlines.
