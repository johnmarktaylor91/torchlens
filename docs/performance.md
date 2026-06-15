# Performance guide

TorchLens is fastest when it captures only the payloads you plan to inspect. A full
`tl.trace(model, x)` records a complete operation graph and saves activations for the selected
sites; `tl.record(model, x, save=...)` is the lighter path for tight loops where you only need
selected records and can materialize a `Trace` later.

Backend note: these timings and sparse-recording examples are torch-oriented. `tl.record()` /
fastlog is torch-only in the backend-v1 registry. Non-torch preview backends may have different
capture costs; JAX and tinygrad `.tlspec` saves materialize forward/derived arrays but loaded
traces cannot replay-validate stripped runtime captures.

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
| Visualization | Call `trace.draw()` after capture, not during hot loops | Rendering is separate from activation collection. |

### Phase timing buckets

`trace._phase_timings` groups wall-clock timings by stable bucket names. Capture buckets include
`ctx_build:*`, `dispatch:*`, `clone_save:*`, and `object_construction:op`. Postprocess buckets use
`postprocess:Step N: ...`, matching the numbered postprocess pipeline. Graphviz rendering records
`render:graphviz:forward`, `render:graphviz:backward`, or `render:graphviz:combined` when those
render entrypoints run.

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
JAX and tinygrad preview bundles materialize forward/derived array payloads; loaded traces still
report replay validation as unavailable because portable save strips runtime replay captures.

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

## Provisional measured numbers (dev host)

> **Provisional.** Captured on a non-canonical Linux dev host (Intel i9-9900X, CPU), TinyNet
> smoke subset. These illustrate relative cost on tiny graphs and are **not** headline figures —
> re-capture the full suite on the canonical bench host before quoting any speed claim. The
> committed baseline is `benchmarks/perf_baselines/linux-cpu-provisional.json`
> (`baseline_status: "provisional"`), which suppresses generated speed headlines.

<!-- generated from TorchLens P6 perf gate JSON; do not hand-edit numbers -->

Measured at SHA `a1df040` on `2026-06-14`.

| Model | Device | Row | Median ms | vs raw forward | Status |
|---|---|---|---:|---:|---|
| tinynet | cpu | raw_forward | 0.9 | 1.00x | ok |
| tinynet | cpu | raw_tl_import | 1.0 | 1.04x | ok |
| tinynet | cpu | raw_global_wrapped | 0.9 | 1.01x | ok |
| tinynet | cpu | raw_target_prepared | 0.9 | 1.02x | ok |
| tinynet | cpu | raw_inference_mode | 1.1 | 1.25x | ok |
| tinynet | cpu | global_wrap_dummy | 1660.7 | 1812.43x | ok |
| tinynet | cpu | first_capture_target | 344.0 | 375.45x | ok |
| tinynet | cpu | tl_trace | 38.8 | 42.32x | ok |
| tinynet | cpu | tl_trace_profile | 41.0 | 44.71x | ok |
| tinynet | cpu | tl_rerun | 49.4 | 53.96x | ok |
| tinynet | cpu | fastlog_module | 15.7 | 17.11x | ok |
| tinynet | cpu | aux_save | 29.0 | 31.67x | ok |
| tinynet | cpu | aux_load | 30.2 | 32.95x | ok |

(Ratios on TinyNet are dominated by fixed per-capture overhead, not steady-state cost; large
models amortize this. `global_wrap_dummy`/`first_capture_target` include one-time wrap/prep cost.)

## Re-capturing baselines (canonical host)

```bash
# Full suite on the canonical bench host (quiet machine):
python -m benchmarks.perf_suite --rerun \
  --out-json benchmarks/perf_baselines/<host>-<device>.json \
  --out-md benchmarks/perf_results_<date>.md
# Sanity self-compare:
python -m benchmarks.perf_gate --baseline benchmarks/perf_baselines/<host>-<device>.json \
  --current benchmarks/perf_baselines/<host>-<device>.json   # expect "passed": true
# Regenerate the docs table:
python -m benchmarks.generate_perf_numbers benchmarks/perf_baselines/<host>-<device>.json \
  --out docs/_perf_numbers_provisional.md
```

On the canonical host, drop the `-provisional` filename suffix and remove the `baseline_status`
key so generated speed headlines are emitted.
