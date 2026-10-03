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

`tl.record(save=...)` is the only torch sparse-capture spelling; the old
`keep_op=`/`keep_module=` alias kwargs are removed.

When a fastlog forward raises, the default remains `on_forward_error="raise"`. Opt into
`on_forward_error="attach_partial"` to attach `exc.partial_recording` and re-raise, or
`on_forward_error="return_partial"` to return a failed partial `Recording` (`return_output=True`
returns `(None, partial)`). Failed partials set `status="partial_error"`, `failed=True`,
string-only error metadata, `n_ops_completed`, and best-effort `last_event_*` fields. user-op
failures exclude the failing call; TL-side capture failures may include a skipped/partial
current-call event. Failed partials cannot be converted with `Recording.to_trace()` or used with
`Recording.log_backward()`. `MemoryError` is swallowed like any other forward exception under
`"return_partial"`/`"attach_partial"`, and materializing the partial allocates *more* memory
inside an already-exhausted heap — under genuine memory pressure keep the default
`on_forward_error="raise"`. Full `tl.trace(...)` failures expose `exc.partial_log`, recoverable
with `tl.partial.from_failed_capture(exc)`.

## Speed knobs

| Knob | Faster setting | Tradeoff |
| --- | --- | --- |
| Payload selection | `save=tl.func(...)` or `save=tl.in_module(...)` | Unsaved activations cannot be read later. |
| Source text | `capture=CaptureOptions(save_code_context=False)` | File/line identity remains, but source text is not loaded. |
| Window payloads | `lookback_payload_policy="metadata_only"` | `tl.followed_by(...)` can select metadata without retaining raw tensors. |
| Retroactive payloads | `lookback_payload_policy="detached_raw"` | Enables payload recovery for recent matched ops at bounded memory cost. |
| Disk storage | `storage=tl.to_disk(path)` | Reduces RAM pressure. Blob writes overlap capture on a bounded async pipeline by default; `max_pending_bytes` (256 MiB default) caps the RAM pending writes may hold, and a disk slower than capture blocks the forward instead of accumulating memory. `async_writes=False` restores synchronous per-blob writes. |
| Gradients | `capture=CaptureOptions(save_grads=False)` (the default) unless needed | Backward-ready captures preserve more state and hooks. |
| Forward-only autograd | `inference_only=True` | Runs forward capture under `torch.no_grad()`; incompatible with backward capture. |
| Forward chunking | `chunk_size=N` | Reduces forward-pass peak memory for single-batch tensor inputs; final saved activations are still accumulated in memory. |
| Recurrence detection | `capture=CaptureOptions(recurrence_detection=False)` | Measured 39-42% of capture time off models with thousands of repeated ops (hand-rolled top-level loops, unrolled decodes). Repeated ops stay separate layers instead of rolling into one multi-pass layer, so a 6-iteration loop yields `relu_1_2 ... relu_6_7` (each `num_passes=1`) instead of one `relu_1_2` with `num_passes=6`. `is_recurrent` and `max_layer_op_count` are still reported. It is a TIME knob only: retained activation bytes are unchanged. |
| Visualization | Call `trace.draw()` after capture, not during hot loops | Rendering is separate from activation collection. |

### Escape diagnostics

The sys.modules crawler is deleted (replaced by the rescue re-run + the
mechanical belt), so the default capture pays no crawl cost. The callable
detector is a separate, opt-in diagnostic cost:

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch


model = nn.ReLU()
x = torch.randn(4)
wrap_torch(escape_detector="shadow")
trace = tl.trace(model, x)
print(trace.escape_detector_event_count, trace.escape_detector_callback_ns)
unwrap_torch()
```

The independent aten-level completeness witness is also opt-in and can run alone or alongside the
callable detector. **This is the only route to `capture_verified=True`.** A plain `tl.trace()` never
arms it: every default capture reads `capture_verified=None`, which every honesty surface renders as
"not recorded" -- not a clean bill. Escapes the wrappers themselves observe (worker-thread ops,
`torch.jit.script` regions, direct `torch.ops.aten.*` calls, `.data.copy_()`) still ceiling the
default capture to `False`, and tensor-to-scalar escapes (`.item()`, `int(argmax)`) raise the
`scalar_escape` advisory; but host tensor sources the wrappers cannot see -- `torch.from_numpy(...)`
/ `torch.as_tensor(np_array)` / `torch.from_dlpack(...)` round-trips, an `autograd.Function` whose
forward runs in NumPy, and in-forward storage writes through `.numpy()` / `untyped_storage()`
aliases -- leave NO ceiling and NO advisory on the default capture (the op is visible to a graph
reader as an internal-source op with `parents=()`). Arm the witness (or run `tl.validate`) when that
class of escape matters:

```python
import torch
from torch import nn
import torchlens as tl
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch


model = nn.ReLU()
x = torch.randn(4)
wrap_torch(
    escape_detector="shadow",
    completeness_witness=True,
)
trace = tl.trace(model, x)
print(trace.completeness_witness_verified)
print(trace.completeness_witness_unaccounted_count)
unwrap_torch()
```

The witness attaches each aten dispatch to the live wrapper token, `func_call_id`, and leaf barcode;
it does not correlate by clock time. A captured Python call may own several ordered aten operations,
recorded in `trace.completeness_decompositions`. Dispatches with no owner, or owned by a wrapper whose
`func_call_id` emits no logged op, produce a `TorchLensCaptureGapWarning`, set
`trace.completeness_witness_verified=False`, and append structured rows to
`trace.completeness_diagnostics`. Work under `pause_logging()` is outside the census. The documented
torch.func/functorch transform boundary runs its interior under that paused scope, and fused or opaque
kernel interiors below the dispatcher are not observable; the dispatched boundary operator itself is
still accounted normally.

`completeness_witness_verified=True` makes only this dispatch-census claim: every owner-thread aten
event observed during active logging was correlated to an emitted op or to an exact audited boundary.
It does not prove that every user call route entered active logging. In particular, a transform callable
built before wrapping can escape without a transform boundary op; TorchLens detects that route and leaves
the census result `True` while setting `capture_verified=False` with
`capture_verification_reason="transform_call_route_unverified"`. `escape_detector_verified=True` makes
the separate claim that the callable detector saw no observable raw-call escape. `capture_verified=True`
is the combined result only when the enabled census/detector checks pass, no raw-transform escape was
detected, and the owner-thread qualification remains valid. A bypassed `torch.compile` region reports
the more specific `capture_verification_reason="dynamo_region_not_logged"` in preference to any of the
above, since Dynamo's compile threads and unaccounted aten dispatches are symptoms of that one region.

The honest rollout comparison is **legacy with no guard** versus **scoped with the requested
guard**, not crawl time in isolation. On Python 3.10–3.11, shadow mode uses `sys.setprofile` and can
be expensive for Python-call-heavy models: a representative call-heavy 16-layer MLP measurement on
Python 3.11 was **+371%** versus the unguarded capture. Treat shadow as an expensive, diagnostic-only
soak tool, not a production capture setting. On Python 3.12+ it prefers local Python-start monitoring
plus diagnostic CALL events. Shadow remains default-off.

The dispatcher witness is likewise default-off. Its diagnostic cost is environment- and workload-
dependent; use an alternating on/off measurement on the target host rather than relying on a fixed
cross-hardware percentage. On this worktree's Python 3.11.6/PyTorch 2.8.0 CPU environment, a
16-pair `Linear(64, 64)`/`ReLU` MLP at batch size 16 measured **+17.8%** median overhead: 0.374
s/capture with the witness off and 0.440 s/capture with it on. Ten alternating batches of five
full `tl.trace` calls per mode followed four two-capture warmup batches; the per-batch ranges were
0.341–0.379 s off and 0.415–0.453 s on. The witness-off route does not enter the witness-only
`pause_logging()` contexts used to keep TorchLens bookkeeping dispatches out of the user-op census.

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
`storage=tl.to_disk(...)`/`streaming=`. Loaded or live chunked traces also reject
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

For activation extraction without a `Trace`, use `tl.extract_dataset(...)` (the old
former `tl.batched_extract` alias is removed); that path returns
tensors or `.pt` files rather than accumulated graph metadata. `chunk_size=` covers the remaining
"dataloader wrapper" case for stacked multi-pass trace capture. Disk mode (`output_dir=`) writes
atomic shards plus a self-describing `manifest.json` (site identity, stimulus provenance, axis
semantics, dtypes); an interrupted long extraction continues from its last completed shard with
`resume=True`, and `torchlens.dataset_extraction.load_extraction(...)` reads the artifact back
with its metadata (both spellings DOCUMENTED-UNSTABLE).

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
Trace repr, notebook HTML, and JSON explanation do not materialize lazy disk payloads for their
NaN/Inf summary; those refs are reported as unexamined until you call ``op.materialize_out()``.
This exemption applies to predicate-selected disk-only payloads. Exhaustive saving
(``layers_to_save="all"``, the default) keeps RAM copies until postprocess, so combine a narrower
``save=`` with disk streaming to reduce peak retained memory. Note that ``save=`` itself takes a
predicate, selector, or ``SaveOptions`` — never the string ``"all"``.
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

Measured on the canonical bench host (Intel i9-9900X) at SHA `712a4e5` on
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

The gate judges TorchLens-owned rows on process-CPU statistics (`cpu_median_ms`/`cpu_iqr_ms`);
the wall-clock fallback for pre-CPU-metric payloads is **not authoritative** — a TorchLens row
judged on wall clock fails the gate by default (the committed 2026-06-16 `linux-cpu.json`
baseline predates the CPU metrics, so comparisons against it need either a rebaseline on the
canonical host or the explicit, disclosed `--allow-wall-clock-only` legacy opt-out).
