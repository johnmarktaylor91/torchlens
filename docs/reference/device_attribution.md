# Device attribution: what TorchLens can and cannot tell you about GPU time

<!-- F09 (costreport D28): every python code block on this page executes in CI
     (tests/test_report_family_docs.py). The honest limit up front: the
     correlation-ID join SHIPS (F27) but its >= 95% real-GPU acceptance gate
     has not run, so device-time and rate CLAIMS stay gated; the door refuses
     honestly on CPU-only hosts. -->

## The honest limit, first

TorchLens's per-op wall times are **instrumented host times**: they include
TorchLens's own bookkeeping and, on CUDA, they are not the op's device time at
all (kernels launch asynchronously). No TorchLens surface calls an analytic
FLOP count "measured", and no surface prints "achieved" FLOP/s or "throughput"
without joined device time — those words are banned by the cost-reporting
contract until the correlation-ID Kineto join passes its acceptance gate
(>= 95% of device kernel time attributed to named ops, forward and backward
separately, remainder named).

What ships today: the **correlation-ID join door**
(`torchlens.observability.native_profile` — one save-nothing execution under
one owned profiler session; device kernel time lands on your named ops via
runtime correlation IDs, never name matching), **NVTX named ranges for
Nsight** (visual correlation only), the analytic cost surfaces
(`flops_report`, `cost_tree`, `roofline`), the instrumented-host-time
profile, and `device_time` / `kernel_time` / `attribution_status` columns on
every compute row — filled from a joined session, `None` otherwise (missing
is None, never zero). The door shipping BEFORE the gate passes is deliberate:
on an unjoined capture every explicit device-time request refuses typed
(`device_time_unavailable`) with the cause and the remedy.

## NVTX ranges for Nsight (visual correlation)

`emit_nvtx=True` wraps every captured op in an NVTX range whose name carries
CALL IDENTITY — `torchlens::<op>#<call_id>`, so twenty conv2d calls are
twenty distinguishable ranges — and TorchLens's own bookkeeping calls are
separated under `torchlens::internal::` instead of polluting the timeline
(on a real GPT-2 capture, half the ranges used to be our own
`register_hook` installs). Nsight Systems' timeline shows which kernels ran
under which TorchLens-visible op. This is visual correlation only — no
joined table, no FLOPs, no throughput numbers come back.

```python
import torch
import torch.nn as nn

import torchlens as tl

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
x = torch.randn(2, 8)

log = tl.trace(model, x, capture=tl.options.CaptureOptions(emit_nvtx=True))
assert log.emit_nvtx
```

Capture under Nsight Systems with:

```bash
nsys profile -o torchlens_run --trace=cuda,nvtx python your_script.py
```

Open the report in the Nsight GUI: the `torchlens::*` NVTX rows sit above the
CUDA kernel rows, and eyeballing the overlap is the supported workflow today.

Chrome-trace clock basis: if you export a `torch.profiler` chrome trace
alongside, its timestamps are the host monotonic clock, not the CUDA device
clock — do not subtract TorchLens's instrumented times from chrome-trace
device spans; they share no basis.

## Analytic cost: exact rules, named coverage

Forward FLOPs are **actual-path analytic** (the executed path and shapes were
observed; the count comes from per-op cost rules). Coverage is a four-way
classification and unknown-cost ops are a named work queue:

```python
report = tl.report.flops_report(log)
assert type(report.forward_flops) is int
assert "actual-path analytic" in str(report)
```

An op without a cost rule never silently counts as zero — it lands in the
unknown ledger with the exact remedy invocation:

```python
from torchlens.capture.flops import register_op_rule

register_op_rule("my_custom_op", lambda *args, **kwargs: 0)
```

The FLOP convention is explicit: stored counts use fma=2 (one
multiply-accumulate = 2 FLOPs); true MACs are the two-term record's first
term, never `flops // 2`.

## The instrumented profile

```python
profile = log.profile(sort_by="flops")
assert "time (instrumented)" in repr(profile)
```

The time column is labeled instrumented everywhere it appears: it is relative
hotspot guidance under capture instrumentation, never a clean benchmark. For
a clean number, `tl.report.time_clean_step(model, x)` re-runs your forward
uninstrumented and says so.

## Arithmetic intensity (roofline, hypothesis tier)

```python
balance = tl.report.machine_balance(
    compute_peak_flops_per_s=1.0e12,
    memory_peak_bytes_per_s=1.0e11,
    basis="advertised",
)
result = tl.report.roofline(log, ridge_intensity=balance.ridge_intensity)
assert result.aggregate_intensity is not None
```

Traffic is the read-once/write-once ideal — a two-sided estimate — so every
memory-bound/compute-bound verdict is labeled a **hypothesis**, aggregate
intensity is `sum(FLOPs)/sum(bytes)` (never a mean of per-op ratios), and
sub-cache ops are excluded from headline bound counts.

## What refuses, and why

- `tl.report.instrumented_rate(trace)` on an all-CUDA trace refuses typed
  (`instrumented_rate_cuda_unsupported`): host time is not a CUDA rate in
  either direction.
- `tl.report.attributed_kernel_utilization()` refuses typed
  (`kernel_utilization_requires_device_join`) until the correlation-ID join
  lands; that quantity is deliberately NOT called MFU.
- `tl.report.mfu(...)` accepts exactly one denominator — uninstrumented
  end-to-end step wall time — and refuses a kernel-union denominator typed
  (`mfu_denominator_invalid`).
