# Native PyTorch tools: what each is authoritative for

<!-- torchnative W0.4: the authorities page. TorchLens defers BY NAME where
     the native tool is the authority our addressing cannot honestly improve.
     Verified against torch 2.13.0; every "verified" note names the version
     it was checked on. Python fences on this page must compile and their
     torchlens imports must resolve (tests/test_docs_executable_contracts.py). -->

TorchLens adds **addressing** — named ops, module call sites, pass-qualified
recurrence, source lines — to data PyTorch's own instrumentation produces.
Where a native tool already owns a question, TorchLens points at it by name
instead of paraphrasing it. This page is that map.

| Question | Authority | What TorchLens adds / does instead |
|---|---|---|
| Device kernel timing | `torch.profiler` (Kineto/CUPTI) | The correlation-ID join lands kernel time on *your named ops and module calls* (`torchlens.observability.native_profile`); every device number still comes from Kineto's instrumentation |
| Clean micro-benchmarks | `torch.utils.benchmark.Timer` | We locate the hotspot (`tl.debug.hot_path`); torch times it clean — whole-model recipe below |
| OOM autopsy / allocator forensics | `torch.cuda.memory._dump_snapshot` + memory_viz | Tested recipe in [memory_debugging.md](memory_debugging.md); TorchLens never parses the snapshot |
| Backward-born NaNs, custom-autograd exceptions | `torch.autograd.set_detect_anomaly` | `tl.debug.nan_report` names the forward ORIGIN op free on an existing capture, and prints the detect_anomaly recipe verbatim for the cases it does not cover |
| Analytic FLOPs by dispatched overload | `torch.utils.flop_counter.FlopCounterMode` | `tl.debug.flops_vs_dispatch` runs both registries detached and reports them side by side — never blended, never called "measured" |
| Module containment (unordered FQN set) | `torch.utils.module_tracker.ModuleTracker` | Strictly weaker than TorchLens module records (no call order, instance map, or reuse identity), but a free independent CI witness |
| Reference-cycle tensor leaks | `torch.utils.viz._cycles.warn_tensor_cycles` | Named in the memory-debugging recipe page |
| Kernel-level timeline analysis | Perfetto / Chrome tracing / HTA | TorchLens preserves the NATIVE chrome trace (Kineto's clock) with an exact-ID mapping sidecar; our own chrome exports are host-clock semantic views and say so |
| CUDA sanitizer | `torch.cuda._sanitizer` | Named here; no TorchLens wrapper |
| Numerical gradient checks | `torch.autograd.gradcheck` | Named here; no TorchLens wrapper |

## The interposition surface (a docs number, not a coverage claim)

TorchLens is built on `torch.overrides` (`__torch_function__`). On torch
2.13, the wrap inventory intersects the great majority (~88%) of torch's
overridable-function registry; the drift oracle
(`tests/test_torchnative_drift_oracle.py`) fails red the day a torch
upgrade adds overridable functions TorchLens does not see. This number
describes the *interposition surface only* — coverage language requires
executed probes, row by row. What the protocol structurally cannot see
(fused kernel interiors, autograd engine internals, dispatcher-level work)
is documented in the capture-honesty pages.

## Clean whole-model timing (the Timer recipe)

TorchLens per-op wall times are instrumented host times — right for
*locating* cost, wrong for *quoting* it. Quote with torch's authority:

```python
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer

import torchlens as tl

model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).eval()
x = torch.randn(32, 64)

# 1. LOCATE with TorchLens (instrumented; fine for ranking):
log = tl.trace(model, x)
ranked = tl.debug.hot_path(log, by="duration")
log.cleanup()

# 2. QUOTE with torch.utils.benchmark (owns warmup, sync, replicates):
timer = Timer(
    stmt="model(x)",
    globals={"model": model, "x": x},
    num_threads=torch.get_num_threads(),
)
measurement = timer.blocked_autorange(min_run_time=0.2)
# measurement.median is the number you may publish; record the option
# fingerprint (threads, min_run_time, device) beside it.
```

An *isolated-op* snippet emitter is deliberately not offered: a synthesized
snippet can silently benchmark a different program than the one captured
(design of record filed; re-entry gated on a provably faithful replay
callable).

## Overhead numbers

Every TorchLens overhead figure goes through the paired-ratio harness
(`torchlens.observability.measure_overhead`): alternating arm order, a
refusal predicate declared before the run, an execution-scope witness on
both arms, and one-sided floors where point estimates are refused. Capture
overhead ships only as a (model, shape, tier) matrix — measured floors on
one small model spanned 1.9x-8x across two shapes, so any single published
multiplier would be misleading by construction.
