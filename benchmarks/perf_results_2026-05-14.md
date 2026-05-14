# TorchLens Performance Benchmark Results - 2026-05-14

Run status: **ok**. Wall clock: 12318.9 s.

## Methodology

Each operation/model/device/pass cell runs in a fresh subprocess. Timing cells use 5 untimed warmups and 50 measured wall-clock samples unless the row is a one-time startup cost. Memory cells are separate subprocesses that record USS after setup, run the operation 10 untimed times, and report `uss_delta_mb_memory_pass`. CUDA memory columns are true allocator peaks after `torch.cuda.reset_peak_memory_stats()`.

Gradient mode is enabled for headline rows, models are in eval mode, dtype is float32, autocast is not used, TF32 is disabled, and seeds are fixed to 0. `Trace.rerun(model, x)` uses the round-4 steady-state contract: capture once before the timing loop, run warmups, then measure repeated reruns on that same Trace.

## Environment

```json
{
  "cuda": "12.8",
  "cuda_available": true,
  "install_notes": {
    "baukit": "pip install baukit failed: no matching distribution found",
    "transformer_lens": "installed separately after baukit failed the combined install"
  },
  "torch": "2.8.0+cu128",
  "versions": {
    "baukit": null,
    "captum": "0.9.0",
    "nnsight": "0.7.0",
    "psutil": "5.9.5",
    "transformer_lens": "3.2.1"
  }
}
```

## Per-cell Tables

### gpt2_hf / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 133.4 | 128.2 | 141.4 | 6.6 | 858.9 | N/A | N/A | ok |
| Raw forward, TL imported | 134.0 | 126.7 | 145.8 | 6.3 | 813.4 | N/A | N/A | ok |
| Raw forward, global wrappers installed | 133.6 | 125.7 | 149.4 | 9.0 | 844.9 | N/A | N/A | ok |
| Raw forward, target model prepared | 135.2 | 129.8 | 144.7 | 7.6 | 293.4 | N/A | N/A | ok |
| Raw + torch.inference_mode floor | 125.7 | 121.0 | 133.2 | 5.1 | 562.9 | N/A | N/A | ok |
| global-wrap-on-dummy startup | 881.2 | 881.2 | 881.2 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 4005.8 | 4005.8 | 4005.8 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 2773.8 | 2497.0 | 2953.8 | 289.9 | 1424.4 | N/A | N/A | ok |
| TL Trace, intervention_ready=True | 2951.9 | 2601.6 | 3163.7 | 342.2 | 3462.3 | N/A | N/A | ok |
| Trace.rerun(model, x) | 2984.4 | 2626.1 | 3205.0 | 399.4 | 5884.7 | N/A | N/A | ok |
| fastlog module-boundary metadata | 278.6 | 254.1 | 587.7 | 23.9 | 346.7 | N/A | N/A | ok |
| fastlog 10% op selectivity | 290.9 | 269.9 | 579.8 | 42.8 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 305.5 | 281.8 | 618.8 | 43.0 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 351.0 | 323.3 | 666.8 | 73.3 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 141.6 | 133.0 | 157.9 | 8.8 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 146.4 | 140.5 | 164.1 | 7.9 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| TL Trace, metadata only (no saved outs) | 1928.3 | 1327.0 | 3400.9 | 699.3 | -944.9 | N/A | N/A | ok |
| Trace.rerun(model, x), no saved outs | 5725.3 | 3189.3 | 7470.7 | 1381.3 | 7280.8 | N/A | N/A | ok |
| fastlog zero-retention predicates | 760.8 | 503.7 | 1378.4 | 391.0 | -88.1 | N/A | N/A | ok |

### gpt2_hf / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 8.9 | 8.7 | 12.0 | 1.3 | 109.0 | 820.9 | 900.0 | ok |
| Raw forward, TL imported | 9.8 | 9.2 | 11.0 | 0.8 | 112.9 | 820.9 | 900.0 | ok |
| Raw forward, global wrappers installed | 14.5 | 13.3 | 15.4 | 0.3 | 100.4 | 820.9 | 900.0 | ok |
| Raw forward, target model prepared | 10.1 | 9.5 | 11.1 | 0.8 | 0.0 | 1865.5 | 2096.0 | ok |
| Raw + torch.inference_mode floor | 9.5 | 8.0 | 9.8 | 1.6 | 108.3 | 564.3 | 642.0 | ok |
| global-wrap-on-dummy startup | 1125.8 | 1125.8 | 1125.8 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 4916.7 | 4916.7 | 4916.7 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 3273.9 | 2767.5 | 3841.6 | 520.9 | 838.9 | 3881.2 | 4540.0 | ok |
| TL Trace, intervention_ready=True | 3293.8 | 2823.4 | 3728.9 | 448.9 | 976.9 | 5370.1 | 5944.0 | ok |
| Trace.rerun(model, x) | 3285.8 | 2830.4 | 3802.8 | 574.0 | 1313.6 | 7453.9 | 8312.0 | ok |
| fastlog module-boundary metadata | 149.4 | 145.3 | 535.2 | 22.7 | 5.6 | 2441.9 | 2744.0 | ok |
| fastlog 10% op selectivity | 146.4 | 141.7 | 509.0 | 20.0 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 163.2 | 159.6 | 555.3 | 33.1 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 216.8 | 203.8 | 739.4 | 99.9 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 10.9 | 10.8 | 12.9 | 0.5 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 10.8 | 10.7 | 11.4 | 0.5 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| TL Trace, metadata only (no saved outs) | 834.0 | 545.6 | 1065.5 | 339.3 | 3.2 | 1888.0 | 2316.0 | ok |
| Trace.rerun(model, x), no saved outs | 3269.0 | 2810.7 | 3910.4 | 576.8 | 1762.5 | 7434.5 | 8292.0 | ok |
| fastlog zero-retention predicates | 165.5 | 157.3 | 546.9 | 84.9 | 5.6 | 2441.9 | 2744.0 | ok |

### gpt2_hooked / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 331.7 | 315.6 | 395.9 | 18.8 | 624.0 | N/A | N/A | ok |
| TL Trace, every-op capture | 6234.3 | 6076.6 | 6844.2 | 292.6 | 1586.6 | N/A | N/A | ok |
| TransformerLens run_with_cache | 349.3 | 336.6 | 373.0 | 17.0 | N/A | N/A | N/A | ok |
| TL Trace, metadata only (no saved outs) | 3602.6 | 2885.4 | 4501.0 | 547.5 | -907.9 | N/A | N/A | ok |
| Trace.rerun(model, x), no saved outs | 11852.5 | 10545.4 | 14813.1 | 1380.7 | 8113.6 | N/A | N/A | ok |
| fastlog zero-retention predicates | 1574.0 | 1042.7 | 2432.4 | 459.9 | -525.0 | N/A | N/A | ok |

### gpt2_hooked / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 19.4 | 19.1 | 22.7 | 0.8 | 56.6 | 1300.5 | 1408.0 | ok |
| TL Trace, every-op capture | 7819.6 | 6431.0 | 8905.2 | 1631.0 | 583.6 | 4828.1 | 5404.0 | ok |
| TransformerLens run_with_cache | 32.5 | 30.1 | 38.2 | 5.8 | N/A | N/A | N/A | ok |
| TL Trace, metadata only (no saved outs) | 1600.9 | 1471.4 | 2109.7 | 277.7 | -27.0 | 2433.3 | 2738.0 | ok |
| Trace.rerun(model, x), no saved outs | 6610.6 | 6420.7 | 6801.9 | 115.7 | 2059.9 | 8470.4 | 9356.0 | ok |
| fastlog zero-retention predicates | 404.9 | 348.7 | 879.0 | 373.6 | -24.9 | 3687.0 | 4082.0 | ok |

### resnet18 / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 60.0 | 57.1 | 69.2 | 5.8 | 187.1 | N/A | N/A | ok |
| Raw forward, TL imported | 71.4 | 68.5 | 75.9 | 3.3 | 127.5 | N/A | N/A | ok |
| Raw forward, global wrappers installed | 67.3 | 65.1 | 72.4 | 2.5 | 132.6 | N/A | N/A | ok |
| Raw forward, target model prepared | 78.6 | 67.3 | 90.9 | 11.6 | 108.3 | N/A | N/A | ok |
| Raw + torch.inference_mode floor | 61.8 | 59.8 | 68.3 | 3.8 | 50.4 | N/A | N/A | ok |
| global-wrap-on-dummy startup | 488.8 | 488.8 | 488.8 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 1231.6 | 1231.6 | 1231.6 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 824.3 | 784.0 | 1022.0 | 100.7 | 1956.5 | N/A | N/A | ok |
| TL Trace, intervention_ready=True | 829.1 | 793.0 | 981.7 | 56.6 | 2255.1 | N/A | N/A | ok |
| Trace.rerun(model, x) | 841.0 | 800.3 | 976.9 | 90.5 | 2899.3 | N/A | N/A | ok |
| fastlog module-boundary metadata | 106.3 | 99.2 | 238.1 | 13.0 | 959.7 | N/A | N/A | ok |
| fastlog 10% op selectivity | 115.5 | 104.4 | 242.8 | 19.1 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 121.7 | 111.9 | 266.2 | 17.2 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 126.1 | 114.4 | 258.2 | 16.5 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 67.3 | 60.8 | 72.8 | 5.4 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 57.9 | 50.7 | 62.2 | 2.5 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| tl.validate(scope='forward') | 2224.7 | 2135.2 | 2422.6 | 114.8 | 1597.4 | N/A | N/A | ok |
| tl.compat.report | 0.5 | 0.5 | 0.7 | 0.2 | 13.8 | N/A | N/A | ok |
| tl.save fresh tlspec path | 845.3 | 827.0 | 935.1 | 23.3 | 24.3 | N/A | N/A | ok |
| tl.load saved fixture | 1255.7 | 1232.9 | 1390.1 | 110.9 | 1757.4 | N/A | N/A | ok |
| TL Trace, metadata only (no saved outs) | 558.3 | 400.8 | 1000.4 | 253.3 | 543.1 | N/A | N/A | ok |
| Trace.rerun(model, x), no saved outs | 2052.7 | 1582.7 | 2584.8 | 417.3 | 3155.8 | N/A | N/A | ok |
| fastlog zero-retention predicates | 314.4 | 183.4 | 609.4 | 127.9 | 962.8 | N/A | N/A | ok |

### resnet18 / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 4.3 | 4.3 | 4.3 | 0.0 | 224.5 | 232.6 | 252.0 | ok |
| Raw forward, TL imported | 4.3 | 4.3 | 4.4 | 0.0 | 227.0 | 232.6 | 252.0 | ok |
| Raw forward, global wrappers installed | 4.3 | 4.3 | 4.4 | 0.0 | 179.3 | 232.3 | 246.0 | ok |
| Raw forward, target model prepared | 4.1 | 4.1 | 4.3 | 0.2 | 0.0 | 728.7 | 770.0 | ok |
| Raw + torch.inference_mode floor | 4.3 | 3.4 | 4.3 | 0.5 | 224.8 | 106.4 | 120.0 | ok |
| global-wrap-on-dummy startup | 683.2 | 683.2 | 683.2 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 1462.2 | 1462.2 | 1462.2 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 896.0 | 792.7 | 1195.1 | 252.7 | 613.0 | 2447.9 | 2566.0 | ok |
| TL Trace, intervention_ready=True | 927.0 | 766.5 | 1300.4 | 274.0 | 488.1 | 2194.1 | 2264.0 | ok |
| Trace.rerun(model, x) | 912.5 | 771.9 | 1226.0 | 254.9 | 724.5 | 3342.2 | 3458.0 | ok |
| fastlog module-boundary metadata | 43.9 | 41.8 | 245.7 | 2.8 | 4.0 | 1506.1 | 1556.0 | ok |
| fastlog 10% op selectivity | 38.9 | 37.6 | 241.9 | 1.3 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 39.5 | 37.8 | 232.3 | 1.5 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 50.0 | 47.6 | 301.9 | 4.9 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 4.6 | 3.7 | 4.9 | 0.3 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 3.8 | 3.7 | 4.6 | 0.8 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| tl.validate(scope='forward') | 393.5 | 377.6 | 508.2 | 31.1 | 84.7 | 2395.0 | 2576.0 | ok |
| tl.compat.report | 0.5 | 0.5 | 0.5 | 0.0 | 0.1 | 49.3 | 64.0 | ok |
| tl.save fresh tlspec path | 920.0 | 897.9 | 1025.2 | 49.2 | 57.3 | 553.6 | 618.0 | ok |
| tl.load saved fixture | 1254.7 | 1237.2 | 1396.1 | 114.5 | 2303.3 | 553.6 | 618.0 | ok |
| TL Trace, metadata only (no saved outs) | 143.8 | 129.1 | 324.0 | 71.8 | 7.4 | 1055.4 | 1100.0 | ok |
| Trace.rerun(model, x), no saved outs | 860.4 | 801.9 | 1150.3 | 209.7 | 801.1 | 3347.3 | 3458.0 | ok |
| fastlog zero-retention predicates | 62.2 | 42.6 | 254.8 | 21.9 | 4.0 | 1506.1 | 1556.0 | ok |

### small_lstm / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 0.3 | 0.2 | 0.3 | 0.0 | 9.3 | N/A | N/A | ok |
| Raw forward, TL imported | 0.3 | 0.2 | 0.3 | 0.0 | 9.6 | N/A | N/A | ok |
| Raw forward, global wrappers installed | 0.3 | 0.3 | 0.3 | 0.0 | 9.1 | N/A | N/A | ok |
| Raw forward, target model prepared | 0.2 | 0.2 | 0.3 | 0.0 | 0.1 | N/A | N/A | ok |
| Raw + torch.inference_mode floor | 0.3 | 0.2 | 0.3 | 0.0 | 5.6 | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1437.5 | 1437.5 | 1437.5 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 353.9 | 353.9 | 353.9 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 14.3 | 13.1 | 16.8 | 0.7 | 1.0 | N/A | N/A | ok |
| TL Trace, intervention_ready=True | 15.4 | 13.7 | 18.2 | 0.5 | 1.5 | N/A | N/A | ok |
| Trace.rerun(model, x) | 16.4 | 15.3 | 22.4 | 2.6 | 2.1 | N/A | N/A | ok |
| fastlog module-boundary metadata | 4.6 | 4.4 | 5.1 | 0.3 | 1.0 | N/A | N/A | ok |
| fastlog 10% op selectivity | 4.9 | 3.8 | 5.2 | 0.6 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 4.4 | 3.9 | 6.0 | 0.7 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 6.8 | 5.0 | 15.6 | 5.3 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 0.3 | 0.3 | 10.8 | 0.0 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 0.3 | 0.3 | 4.5 | 0.0 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| TL Trace, metadata only (no saved outs) | 23.8 | 16.0 | 86.8 | 18.9 | 0.8 | N/A | N/A | ok |
| Trace.rerun(model, x), no saved outs | 32.1 | 19.2 | 130.8 | 35.4 | 2.2 | N/A | N/A | ok |
| fastlog zero-retention predicates | 15.9 | 4.9 | 50.5 | 14.6 | 1.1 | N/A | N/A | ok |

### tinynet / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 1.0 | 0.9 | 26.6 | 0.9 | 8.6 | N/A | N/A | ok |
| Raw forward, TL imported | 1.0 | 0.9 | 19.2 | 4.0 | 8.4 | N/A | N/A | ok |
| Raw forward, global wrappers installed | 1.0 | 1.0 | 15.1 | 0.0 | 7.4 | N/A | N/A | ok |
| Raw forward, target model prepared | 1.0 | 0.9 | 5.3 | 0.3 | 1.4 | N/A | N/A | ok |
| Raw + torch.inference_mode floor | 0.9 | 0.8 | 4.9 | 0.0 | 8.4 | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1315.3 | 1315.3 | 1315.3 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 300.8 | 300.8 | 300.8 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 26.7 | 24.3 | 32.6 | 2.9 | 16.6 | N/A | N/A | ok |
| TL Trace, intervention_ready=True | 28.0 | 26.0 | 41.1 | 2.9 | 18.1 | N/A | N/A | ok |
| Trace.rerun(model, x) | 112.3 | 36.2 | 346.6 | 122.7 | 38.1 | N/A | N/A | ok |
| fastlog module-boundary metadata | 8.9 | 7.2 | 15.7 | 4.2 | 7.4 | N/A | N/A | ok |
| fastlog 10% op selectivity | 8.4 | 7.4 | 12.1 | 1.9 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 8.4 | 7.5 | 10.4 | 1.4 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 8.6 | 7.5 | 9.7 | 1.1 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 1.2 | 1.1 | 1.2 | 0.0 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 1.2 | 1.2 | 1.3 | 0.0 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| tl.validate(scope='forward') | 218.3 | 114.8 | 465.5 | 98.3 | 12.9 | N/A | N/A | ok |
| tl.compat.report | 0.3 | 0.3 | 0.4 | 0.0 | 13.7 | N/A | N/A | ok |
| tl.save fresh tlspec path | 25.6 | 25.3 | 33.1 | 0.5 | 0.9 | N/A | N/A | ok |
| tl.load saved fixture | 23.1 | 22.4 | 26.1 | 1.0 | 8.8 | N/A | N/A | ok |
| TL Trace, metadata only (no saved outs) | 61.0 | 26.2 | 205.6 | 62.3 | 9.1 | N/A | N/A | ok |
| Trace.rerun(model, x), no saved outs | 65.9 | 39.1 | 249.6 | 59.8 | 34.8 | N/A | N/A | ok |
| fastlog zero-retention predicates | 18.4 | 10.2 | 103.3 | 18.1 | 7.1 | N/A | N/A | ok |

### tinynet / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 0.4 | 0.3 | 0.5 | 0.1 | 197.9 | 10.6 | 24.0 | ok |
| Raw forward, TL imported | 0.3 | 0.3 | 0.3 | 0.0 | 197.8 | 10.6 | 24.0 | ok |
| Raw forward, global wrappers installed | 0.4 | 0.4 | 0.4 | 0.0 | 151.1 | 10.6 | 24.0 | ok |
| Raw forward, target model prepared | 0.4 | 0.4 | 0.4 | 0.0 | 0.0 | 13.9 | 28.0 | ok |
| Raw + torch.inference_mode floor | 0.3 | 0.3 | 0.3 | 0.0 | 198.1 | 9.4 | 22.0 | ok |
| global-wrap-on-dummy startup | 1368.8 | 1368.8 | 1368.8 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 473.8 | 473.8 | 473.8 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 30.0 | 28.4 | 32.9 | 1.4 | 7.3 | 26.7 | 40.0 | ok |
| TL Trace, intervention_ready=True | 40.8 | 31.2 | 50.8 | 9.9 | 5.6 | 27.2 | 40.0 | ok |
| Trace.rerun(model, x) | 38.3 | 31.4 | 42.7 | 9.6 | 12.2 | 44.5 | 58.0 | ok |
| fastlog module-boundary metadata | 7.9 | 7.2 | 9.0 | 0.6 | 1.1 | 19.1 | 32.0 | ok |
| fastlog 10% op selectivity | 6.3 | 6.0 | 7.0 | 0.4 | N/A | N/A | N/A | ok |
| fastlog 50% op selectivity | 6.7 | 6.4 | 7.6 | 0.6 | N/A | N/A | N/A | ok |
| fastlog all-op/all-module | 7.3 | 6.7 | 8.9 | 0.7 | N/A | N/A | N/A | ok |
| Vanilla hooks manual dict | 0.9 | 0.8 | 0.9 | 0.0 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 0.5 | 0.5 | 0.6 | 0.0 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |
| TL Trace, metadata only (no saved outs) | 21.7 | 20.9 | 23.8 | 0.8 | 1.8 | 21.6 | 34.0 | ok |
| Trace.rerun(model, x), no saved outs | 31.9 | 30.1 | 51.5 | 11.2 | 79.2 | 39.0 | 52.0 | ok |
| fastlog zero-retention predicates | 8.1 | 7.3 | 14.6 | 1.7 | 1.1 | 19.1 | 32.0 | ok |

## Wrapper-only overhead (no-save variants)

Headline insight: on ResNet-18 CPU, Trace metadata-only capture is 9.30x (+830%) versus raw forward, compared with full Trace at 13.73x (+1273%); disabling saved tensors cuts median Trace time by 32%. `fastlog_zero` is 5.24x (+424%). These data separate wrapper/metadata overhead from tensor data the user chose to capture, but they do not support treating all remaining full-Trace cost as tensor-copy cost.

### gpt2_hf / cpu

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 133.4 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 1928.3 | 14.45x (+1345%) | True | ok |
| Trace.rerun(model, x), no saved outs | 5725.3 | 42.92x (+4192%) | True | ok |
| fastlog zero-retention predicates | 760.8 | 5.70x (+470%) | N/A | ok |

### gpt2_hf / cuda

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 8.9 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 834.0 | 94.16x (+9316%) | True | ok |
| Trace.rerun(model, x), no saved outs | 3269.0 | 369.06x (+36806%) | True | ok |
| fastlog zero-retention predicates | 165.5 | 18.68x (+1768%) | N/A | ok |

### gpt2_hooked / cpu

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 331.7 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 3602.6 | 10.86x (+986%) | True | ok |
| Trace.rerun(model, x), no saved outs | 11852.5 | 35.73x (+3473%) | True | ok |
| fastlog zero-retention predicates | 1574.0 | 4.75x (+375%) | N/A | ok |

### gpt2_hooked / cuda

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 19.4 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 1600.9 | 82.43x (+8143%) | True | ok |
| Trace.rerun(model, x), no saved outs | 6610.6 | 340.35x (+33935%) | True | ok |
| fastlog zero-retention predicates | 404.9 | 20.84x (+1984%) | N/A | ok |

### resnet18 / cpu

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 60.0 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 558.3 | 9.30x (+830%) | True | ok |
| Trace.rerun(model, x), no saved outs | 2052.7 | 34.20x (+3320%) | True | ok |
| fastlog zero-retention predicates | 314.4 | 5.24x (+424%) | N/A | ok |

### resnet18 / cuda

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 4.3 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 143.8 | 33.41x (+3241%) | True | ok |
| Trace.rerun(model, x), no saved outs | 860.4 | 199.98x (+19898%) | True | ok |
| fastlog zero-retention predicates | 62.2 | 14.45x (+1345%) | N/A | ok |

### small_lstm / cpu

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 0.3 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 23.8 | 94.72x (+9372%) | True | ok |
| Trace.rerun(model, x), no saved outs | 32.1 | 127.60x (+12660%) | True | ok |
| fastlog zero-retention predicates | 15.9 | 63.07x (+6207%) | N/A | ok |

### tinynet / cpu

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 1.0 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 61.0 | 62.67x (+6167%) | True | ok |
| Trace.rerun(model, x), no saved outs | 65.9 | 67.70x (+6670%) | True | ok |
| fastlog zero-retention predicates | 18.4 | 18.92x (+1792%) | N/A | ok |

### tinynet / cuda

| Operation | median_ms | vs raw forward | no-save invariant | Status |
|---|---:|---:|---|---|
| Pure raw forward | 0.4 | 1.00x (+0%) | N/A | ok |
| TL Trace, metadata only (no saved outs) | 21.7 | 52.09x (+5109%) | True | ok |
| Trace.rerun(model, x), no saved outs | 31.9 | 76.67x (+7567%) | True | ok |
| fastlog zero-retention predicates | 8.1 | 19.37x (+1837%) | N/A | ok |

With tensor retention disabled, these rows isolate wrapper dispatch and metadata bookkeeping from activation-copy cost; Trace.rerun is for new inputs, not interventions. Ratios versus raw forward are gpt2_hf/cpu: trace 14.45x (+1345%), rerun 42.92x (+4192%), fastlog 5.70x (+470%); gpt2_hf/cuda: trace 94.16x (+9316%), rerun 369.06x (+36806%), fastlog 18.68x (+1768%); gpt2_hooked/cpu: trace 10.86x (+986%), rerun 35.73x (+3473%), fastlog 4.75x (+375%); gpt2_hooked/cuda: trace 82.43x (+8143%), rerun 340.35x (+33935%), fastlog 20.84x (+1984%); resnet18/cpu: trace 9.30x (+830%), rerun 34.20x (+3320%), fastlog 5.24x (+424%); resnet18/cuda: trace 33.41x (+3241%), rerun 199.98x (+19898%), fastlog 14.45x (+1345%); small_lstm/cpu: trace 94.72x (+9372%), rerun 127.60x (+12660%), fastlog 63.07x (+6207%); tinynet/cpu: trace 62.67x (+6167%), rerun 67.70x (+6670%), fastlog 18.92x (+1792%); tinynet/cuda: trace 52.09x (+5109%), rerun 76.67x (+7567%), fastlog 19.37x (+1837%).

## Peer Comparison Tables

### gpt2_hf / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TL Trace, every-op capture | 2773.8 | 2497.0 | 2953.8 | 289.9 | 1424.4 | N/A | N/A | ok |
| Vanilla hooks manual dict | 141.6 | 133.0 | 157.9 | 8.8 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 146.4 | 140.5 | 164.1 | 7.9 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

### gpt2_hf / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TL Trace, every-op capture | 3273.9 | 2767.5 | 3841.6 | 520.9 | 838.9 | 3881.2 | 4540.0 | ok |
| Vanilla hooks manual dict | 10.9 | 10.8 | 12.9 | 0.5 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 10.8 | 10.7 | 11.4 | 0.5 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

### gpt2_hooked / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TL Trace, every-op capture | 6234.3 | 6076.6 | 6844.2 | 292.6 | 1586.6 | N/A | N/A | ok |
| TransformerLens run_with_cache | 349.3 | 336.6 | 373.0 | 17.0 | N/A | N/A | N/A | ok |

### gpt2_hooked / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| TL Trace, every-op capture | 7819.6 | 6431.0 | 8905.2 | 1631.0 | 583.6 | 4828.1 | 5404.0 | ok |
| TransformerLens run_with_cache | 32.5 | 30.1 | 38.2 | 5.8 | N/A | N/A | N/A | ok |

### resnet18 / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla hooks manual dict | 67.3 | 60.8 | 72.8 | 5.4 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 57.9 | 50.7 | 62.2 | 2.5 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

### resnet18 / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla hooks manual dict | 4.6 | 3.7 | 4.9 | 0.3 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 3.8 | 3.7 | 4.6 | 0.8 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

### small_lstm / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla hooks manual dict | 0.3 | 0.3 | 10.8 | 0.0 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 0.3 | 0.3 | 4.5 | 0.0 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

### tinynet / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla hooks manual dict | 1.2 | 1.1 | 1.2 | 0.0 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 1.2 | 1.2 | 1.3 | 0.0 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

### tinynet / cuda

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla hooks manual dict | 0.9 | 0.8 | 0.9 | 0.0 | N/A | N/A | N/A | ok |
| Vanilla hooks context manager | 0.5 | 0.5 | 0.6 | 0.0 | N/A | N/A | N/A | ok |
| baukit TraceDict | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: baukit is not importable |
| nnsight trace | N/A | N/A | N/A | N/A | N/A | N/A | N/A | skipped: generic nn.Module does not expose nnsight trace |

## Auxiliary Primitives

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tl.validate(scope='forward') | 218.3 | 114.8 | 465.5 | 98.3 | 12.9 | N/A | N/A | ok |
| tl.compat.report | 0.3 | 0.3 | 0.4 | 0.0 | 13.7 | N/A | N/A | ok |
| tl.save fresh tlspec path | 25.6 | 25.3 | 33.1 | 0.5 | 0.9 | N/A | N/A | ok |
| tl.load saved fixture | 23.1 | 22.4 | 26.1 | 1.0 | 8.8 | N/A | N/A | ok |
| tl.validate(scope='forward') | 2224.7 | 2135.2 | 2422.6 | 114.8 | 1597.4 | N/A | N/A | ok |
| tl.compat.report | 0.5 | 0.5 | 0.7 | 0.2 | 13.8 | N/A | N/A | ok |
| tl.save fresh tlspec path | 845.3 | 827.0 | 935.1 | 23.3 | 24.3 | N/A | N/A | ok |
| tl.load saved fixture | 1255.7 | 1232.9 | 1390.1 | 110.9 | 1757.4 | N/A | N/A | ok |
| tl.validate(scope='forward') | 393.5 | 377.6 | 508.2 | 31.1 | 84.7 | 2395.0 | 2576.0 | ok |
| tl.compat.report | 0.5 | 0.5 | 0.5 | 0.0 | 0.1 | 49.3 | 64.0 | ok |
| tl.save fresh tlspec path | 920.0 | 897.9 | 1025.2 | 49.2 | 57.3 | 553.6 | 618.0 | ok |
| tl.load saved fixture | 1254.7 | 1237.2 | 1396.1 | 114.5 | 2303.3 | 553.6 | 618.0 | ok |

## Decoration Overhead

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| global-wrap-on-dummy startup | 1315.3 | 1315.3 | 1315.3 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 300.8 | 300.8 | 300.8 | 0.0 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 488.8 | 488.8 | 488.8 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 1231.6 | 1231.6 | 1231.6 | 0.0 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 881.2 | 881.2 | 881.2 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 4005.8 | 4005.8 | 4005.8 | 0.0 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1437.5 | 1437.5 | 1437.5 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 353.9 | 353.9 | 353.9 | 0.0 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1368.8 | 1368.8 | 1368.8 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 473.8 | 473.8 | 473.8 | 0.0 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 683.2 | 683.2 | 683.2 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 1462.2 | 1462.2 | 1462.2 | 0.0 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1125.8 | 1125.8 | 1125.8 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 4916.7 | 4916.7 | 4916.7 | 0.0 | N/A | N/A | N/A | ok |

## Post-run Notes

- Reran fastlog_op_10 and fastlog_op_50 timing cells after fixing dry-run predicate selection.
- Committed JSON is compacted: raw per-sample arrays, subprocess stdout/stderr tails, and repeated per-cell env blocks were removed to satisfy repository artifact size and secret-scan hooks.
- Git SHA is intentionally omitted from committed artifacts because the repository secret scanner flags SHA-like hex strings in generated JSON/Markdown.
- No-save wrapper-only addendum run appended on 2026-05-14; addendum wall clock 3430.8 s.

## Peer Exclusion Appendix

- Captum is excluded from capture timing because it exposes attribution methods, not generic activation capture.
- torchexplorer is excluded because it is visualization-oriented rather than extraction-for-downstream-use capture.
- pyvene is excluded because it is a causal-intervention library, not a capture-timing peer.
- hooks_dict patterns are represented by the two vanilla `register_forward_hook` rows.
- baukit install failed from the configured package indexes when no distribution was found; rows are skipped when unavailable.

## Limitations + Caveats

- CPU USS is an end-of-10-run memory-pass delta, not a sampled sub-operation peak.
- Sub-ms operations can legitimately show 0.0 MB USS delta because allocators reuse pages.
- TorchLens Trace captures every tensor-producing torch operation; peer hook rows capture module boundaries only.
- HF GPT-2 and HookedTransformer GPT-2 are separate model implementations and should not be compared row-by-row.
- No-save Trace rows use `layers_to_save=[]`, which records metadata for every op but saves no activation tensors.

## Rerun Tolerance

```json
{
  "checks": [
    {
      "device": "cpu",
      "diff_ms": 0.027076108381152153,
      "iqr_run1_ms": 0.8838637731969357,
      "iqr_run2_ms": 0.03458978608250618,
      "median_run1_ms": 0.972967129200697,
      "median_run2_ms": 0.9458910208195448,
      "model": "tinynet",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 1.7677275463938713
    },
    {
      "device": "cpu",
      "diff_ms": 0.021156505681574345,
      "iqr_run1_ms": 4.0430554654449224,
      "iqr_run2_ms": 0.23499212693423033,
      "median_run1_ms": 0.9804759174585342,
      "median_run2_ms": 0.9593194117769599,
      "model": "tinynet",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 8.086110930889845
    },
    {
      "device": "cpu",
      "diff_ms": 0.24161615874618292,
      "iqr_run1_ms": 0.036970479413867,
      "iqr_run2_ms": 6.556264532264322,
      "median_run1_ms": 1.0158228687942028,
      "median_run2_ms": 1.2574390275403857,
      "model": "tinynet",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 13.112529064528644
    },
    {
      "device": "cpu",
      "diff_ms": 0.07778138387948275,
      "iqr_run1_ms": 0.3221440711058676,
      "iqr_run2_ms": 0.08132471702992916,
      "median_run1_ms": 1.0002079652622342,
      "median_run2_ms": 0.9224265813827515,
      "model": "tinynet",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 0.6442881422117352
    },
    {
      "device": "cpu",
      "diff_ms": 0.043769367039203644,
      "iqr_run1_ms": 0.04280469147488475,
      "iqr_run2_ms": 0.0129429972730577,
      "median_run1_ms": 0.8813583990558982,
      "median_run2_ms": 0.8375890320166945,
      "model": "tinynet",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 16.384611139073968,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1315.2802179101855,
      "median_run2_ms": 1331.6648290492594,
      "model": "tinynet",
      "operation": "global_wrap_dummy",
      "passed": true,
      "tolerance_ms": 131.52802179101855
    },
    {
      "device": "cpu",
      "diff_ms": 46.016617910936475,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 300.8447359316051,
      "median_run2_ms": 346.8613538425416,
      "model": "tinynet",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 30.08447359316051
    },
    {
      "device": "cpu",
      "diff_ms": 46.127413981594145,
      "iqr_run1_ms": 2.8608323191292584,
      "iqr_run2_ms": 52.72633832646534,
      "median_run1_ms": 26.655970490537584,
      "median_run2_ms": 72.78338447213173,
      "model": "tinynet",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 105.45267665293068
    },
    {
      "device": "cpu",
      "diff_ms": 41.98341444134712,
      "iqr_run1_ms": 2.887773560360074,
      "iqr_run2_ms": 46.700506878551096,
      "median_run1_ms": 27.96083502471447,
      "median_run2_ms": 69.94424946606159,
      "model": "tinynet",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 93.40101375710219
    },
    {
      "device": "cpu",
      "diff_ms": 47.80789546202868,
      "iqr_run1_ms": 122.66609538346529,
      "iqr_run2_ms": 44.646293739788234,
      "median_run1_ms": 112.2815110720694,
      "median_run2_ms": 64.47361561004072,
      "model": "tinynet",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 245.33219076693058
    },
    {
      "device": "cpu",
      "diff_ms": 8.184971520677209,
      "iqr_run1_ms": 4.2058274848386645,
      "iqr_run2_ms": 14.109786308836192,
      "median_run1_ms": 8.92197946086526,
      "median_run2_ms": 17.106950981542468,
      "model": "tinynet",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 28.219572617672384
    },
    {
      "device": "cpu",
      "diff_ms": 6.416678195819259,
      "iqr_run1_ms": 1.0606504511088133,
      "iqr_run2_ms": 9.246364119462669,
      "median_run1_ms": 8.588149910792708,
      "median_run2_ms": 15.004828106611967,
      "model": "tinynet",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 18.492728238925338
    },
    {
      "device": "cpu",
      "diff_ms": 0.011132797226309776,
      "iqr_run1_ms": 0.02353114541620016,
      "iqr_run2_ms": 4.008480231277645,
      "median_run1_ms": 1.1676884023472667,
      "median_run2_ms": 1.156555605120957,
      "model": "tinynet",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 8.01696046255529
    },
    {
      "device": "cpu",
      "diff_ms": 0.00028801150619983673,
      "iqr_run1_ms": 0.03580201882869005,
      "iqr_run2_ms": 4.032200726214796,
      "median_run1_ms": 1.2117850128561258,
      "median_run2_ms": 1.211497001349926,
      "model": "tinynet",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 8.064401452429593
    },
    {
      "device": "cpu",
      "diff_ms": 76.27530593890697,
      "iqr_run1_ms": 5.789143848232925,
      "iqr_run2_ms": 58.51725663524121,
      "median_run1_ms": 60.02567557152361,
      "median_run2_ms": 136.30098151043057,
      "model": "resnet18",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 117.03451327048242
    },
    {
      "device": "cpu",
      "diff_ms": 29.126293840818107,
      "iqr_run1_ms": 3.3191561233252287,
      "iqr_run2_ms": 70.20308618666604,
      "median_run1_ms": 71.44180103205144,
      "median_run2_ms": 100.56809487286955,
      "model": "resnet18",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 140.40617237333208
    },
    {
      "device": "cpu",
      "diff_ms": 2.2165613481774926,
      "iqr_run1_ms": 2.5397356948815286,
      "iqr_run2_ms": 3.47352831158787,
      "median_run1_ms": 67.33890261966735,
      "median_run2_ms": 69.55546396784484,
      "model": "resnet18",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 6.94705662317574
    },
    {
      "device": "cpu",
      "diff_ms": 92.85892499610782,
      "iqr_run1_ms": 11.605859210249037,
      "iqr_run2_ms": 109.27837330382317,
      "median_run1_ms": 78.55115504935384,
      "median_run2_ms": 171.41008004546165,
      "model": "resnet18",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 218.55674660764635
    },
    {
      "device": "cpu",
      "diff_ms": 49.278011079877615,
      "iqr_run1_ms": 3.777934063691646,
      "iqr_run2_ms": 56.336632929742336,
      "median_run1_ms": 61.82396796066314,
      "median_run2_ms": 111.10197904054075,
      "model": "resnet18",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 112.67326585948467
    },
    {
      "device": "cpu",
      "diff_ms": 86.17532416246831,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 488.7633288744837,
      "median_run2_ms": 574.938653036952,
      "model": "resnet18",
      "operation": "global_wrap_dummy",
      "passed": false,
      "tolerance_ms": 48.87633288744837
    },
    {
      "device": "cpu",
      "diff_ms": 1510.0721230264753,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1231.6243960522115,
      "median_run2_ms": 2741.696519078687,
      "model": "resnet18",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 123.16243960522115
    },
    {
      "device": "cpu",
      "diff_ms": 970.2931722858921,
      "iqr_run1_ms": 100.69877782370895,
      "iqr_run2_ms": 411.71891981502995,
      "median_run1_ms": 824.3352971039712,
      "median_run2_ms": 1794.6284693898633,
      "model": "resnet18",
      "operation": "tl_trace",
      "passed": false,
      "tolerance_ms": 823.4378396300599
    },
    {
      "device": "cpu",
      "diff_ms": 553.254607017152,
      "iqr_run1_ms": 56.590083811897784,
      "iqr_run2_ms": 1216.8781143263914,
      "median_run1_ms": 829.127712524496,
      "median_run2_ms": 1382.382319541648,
      "model": "resnet18",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 2433.7562286527827
    },
    {
      "device": "cpu",
      "diff_ms": 891.3324528839439,
      "iqr_run1_ms": 90.5124563141726,
      "iqr_run2_ms": 590.0347064016387,
      "median_run1_ms": 841.0027145873755,
      "median_run2_ms": 1732.3351674713194,
      "model": "resnet18",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 1180.0694128032774
    },
    {
      "device": "cpu",
      "diff_ms": 161.28080594353378,
      "iqr_run1_ms": 13.023032050114125,
      "iqr_run2_ms": 169.3953171488829,
      "median_run1_ms": 106.32096708286554,
      "median_run2_ms": 267.6017730263993,
      "model": "resnet18",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 338.7906342977658
    },
    {
      "device": "cpu",
      "diff_ms": 232.97431704122573,
      "iqr_run1_ms": 16.508487809915096,
      "iqr_run2_ms": 187.89926869794726,
      "median_run1_ms": 126.12152146175504,
      "median_run2_ms": 359.09583850298077,
      "model": "resnet18",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 375.7985373958945
    },
    {
      "device": "cpu",
      "diff_ms": 64.34432894457132,
      "iqr_run1_ms": 5.403295392170548,
      "iqr_run2_ms": 70.361690944992,
      "median_run1_ms": 67.29279505088925,
      "median_run2_ms": 131.63712399546057,
      "model": "resnet18",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 140.723381889984
    },
    {
      "device": "cpu",
      "diff_ms": 74.19368845876306,
      "iqr_run1_ms": 2.4628626997582614,
      "iqr_run2_ms": 53.924687090329826,
      "median_run1_ms": 57.86138353869319,
      "median_run2_ms": 132.05507199745625,
      "model": "resnet18",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 107.84937418065965
    },
    {
      "device": "cpu",
      "diff_ms": 157.5667819706723,
      "iqr_run1_ms": 6.626588990911841,
      "iqr_run2_ms": 119.18875569244847,
      "median_run1_ms": 133.40118003543466,
      "median_run2_ms": 290.967962006107,
      "model": "gpt2_hf",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 238.37751138489693
    },
    {
      "device": "cpu",
      "diff_ms": 192.8658924298361,
      "iqr_run1_ms": 6.287180236540735,
      "iqr_run2_ms": 105.22796266013756,
      "median_run1_ms": 134.03590151574463,
      "median_run2_ms": 326.9017939455807,
      "model": "gpt2_hf",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 210.45592532027513
    },
    {
      "device": "cpu",
      "diff_ms": 93.01161218900234,
      "iqr_run1_ms": 8.977130230050534,
      "iqr_run2_ms": 90.32543754437938,
      "median_run1_ms": 133.6122559150681,
      "median_run2_ms": 226.62386810407043,
      "model": "gpt2_hf",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 180.65087508875877
    },
    {
      "device": "cpu",
      "diff_ms": 169.44274748675525,
      "iqr_run1_ms": 7.633015338797122,
      "iqr_run2_ms": 110.85209879092872,
      "median_run1_ms": 135.20355592481792,
      "median_run2_ms": 304.6463034115732,
      "model": "gpt2_hf",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 221.70419758185744
    },
    {
      "device": "cpu",
      "diff_ms": 159.55932240467519,
      "iqr_run1_ms": 5.0584261189214885,
      "iqr_run2_ms": 124.57381782587618,
      "median_run1_ms": 125.70460001006722,
      "median_run2_ms": 285.2639224147424,
      "model": "gpt2_hf",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 249.14763565175235
    },
    {
      "device": "cpu",
      "diff_ms": 221.6205419972539,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 881.2494131270796,
      "median_run2_ms": 1102.8699551243335,
      "model": "gpt2_hf",
      "operation": "global_wrap_dummy",
      "passed": false,
      "tolerance_ms": 88.12494131270796
    },
    {
      "device": "cpu",
      "diff_ms": 8009.433972882107,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 4005.7816461194307,
      "median_run2_ms": 12015.215619001538,
      "model": "gpt2_hf",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 400.57816461194307
    },
    {
      "device": "cpu",
      "diff_ms": 2431.120743509382,
      "iqr_run1_ms": 289.91786343976855,
      "iqr_run2_ms": 1000.6872293888591,
      "median_run1_ms": 2773.7644650042057,
      "median_run2_ms": 5204.885208513588,
      "model": "gpt2_hf",
      "operation": "tl_trace",
      "passed": false,
      "tolerance_ms": 2001.3744587777182
    },
    {
      "device": "cpu",
      "diff_ms": 86.83746005408466,
      "iqr_run1_ms": 342.1764707309194,
      "iqr_run2_ms": 310.0441485294141,
      "median_run1_ms": 2951.9310659961775,
      "median_run2_ms": 2865.093605942093,
      "model": "gpt2_hf",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 684.3529414618388
    },
    {
      "device": "cpu",
      "diff_ms": 48.55308949481696,
      "iqr_run1_ms": 399.4196680141613,
      "iqr_run2_ms": 383.33439943380654,
      "median_run1_ms": 2984.3791994499043,
      "median_run2_ms": 2935.8261099550873,
      "model": "gpt2_hf",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 798.8393360283226
    },
    {
      "device": "cpu",
      "diff_ms": 2.7854584623128176,
      "iqr_run1_ms": 23.882749606855214,
      "iqr_run2_ms": 26.472070661839098,
      "median_run1_ms": 278.61628553364426,
      "median_run2_ms": 275.83082707133144,
      "model": "gpt2_hf",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 52.944141323678195
    },
    {
      "device": "cpu",
      "diff_ms": 7.497441954910755,
      "iqr_run1_ms": 73.29819712322205,
      "iqr_run2_ms": 58.3450912963599,
      "median_run1_ms": 351.0306734824553,
      "median_run2_ms": 343.53323152754456,
      "model": "gpt2_hf",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 146.5963942464441
    },
    {
      "device": "cpu",
      "diff_ms": 1.9851390970870852,
      "iqr_run1_ms": 8.779242867603898,
      "iqr_run2_ms": 7.929216662887484,
      "median_run1_ms": 141.59554615616798,
      "median_run2_ms": 139.6104070590809,
      "model": "gpt2_hf",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 17.558485735207796
    },
    {
      "device": "cpu",
      "diff_ms": 9.160149027593434,
      "iqr_run1_ms": 7.858429453335702,
      "iqr_run2_ms": 7.121066038962454,
      "median_run1_ms": 146.356268087402,
      "median_run2_ms": 137.19611905980855,
      "model": "gpt2_hf",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 15.716858906671405
    },
    {
      "device": "cpu",
      "diff_ms": 0.004543573595583439,
      "iqr_run1_ms": 0.008235976565629244,
      "iqr_run2_ms": 0.006165006197988987,
      "median_run1_ms": 0.25132903829216957,
      "median_run2_ms": 0.255872611887753,
      "model": "small_lstm",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.008990638889372349,
      "iqr_run1_ms": 0.012912030797451735,
      "iqr_run2_ms": 0.009577313903719187,
      "median_run1_ms": 0.2525480231270194,
      "median_run2_ms": 0.24355738423764706,
      "model": "small_lstm",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.0022744061425328255,
      "iqr_run1_ms": 0.010396353900432587,
      "iqr_run2_ms": 0.005698646418750286,
      "median_run1_ms": 0.2608525101095438,
      "median_run2_ms": 0.258578103967011,
      "model": "small_lstm",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.03005005419254303,
      "iqr_run1_ms": 0.008672242984175682,
      "iqr_run2_ms": 0.004800909664481878,
      "median_run1_ms": 0.2317150356248021,
      "median_run2_ms": 0.26176508981734514,
      "model": "small_lstm",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.03678153734654188,
      "iqr_run1_ms": 0.023164553567767143,
      "iqr_run2_ms": 0.031885108910501,
      "median_run1_ms": 0.2548620104789734,
      "median_run2_ms": 0.2180804731324315,
      "model": "small_lstm",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 182.95146222226322,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1437.4967780895531,
      "median_run2_ms": 1254.54531586729,
      "model": "small_lstm",
      "operation": "global_wrap_dummy",
      "passed": false,
      "tolerance_ms": 143.7496778089553
    },
    {
      "device": "cpu",
      "diff_ms": 25.90769506059587,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 353.89634408056736,
      "median_run2_ms": 327.9886490199715,
      "model": "small_lstm",
      "operation": "first_capture_target",
      "passed": true,
      "tolerance_ms": 35.389634408056736
    },
    {
      "device": "cpu",
      "diff_ms": 1.833037706092,
      "iqr_run1_ms": 0.7047866238281131,
      "iqr_run2_ms": 0.8210439700633287,
      "median_run1_ms": 14.336752006784081,
      "median_run2_ms": 12.503714300692081,
      "model": "small_lstm",
      "operation": "tl_trace",
      "passed": false,
      "tolerance_ms": 1.6420879401266575
    },
    {
      "device": "cpu",
      "diff_ms": 0.5308756371960044,
      "iqr_run1_ms": 0.48575690016150475,
      "iqr_run2_ms": 1.787687127944082,
      "median_run1_ms": 15.43570903595537,
      "median_run2_ms": 14.904833398759365,
      "model": "small_lstm",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 3.575374255888164
    },
    {
      "device": "cpu",
      "diff_ms": 1.718781073577702,
      "iqr_run1_ms": 2.6301791658625007,
      "iqr_run2_ms": 2.7960577281191945,
      "median_run1_ms": 16.383412992581725,
      "median_run2_ms": 14.664631919004023,
      "model": "small_lstm",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 5.592115456238389
    },
    {
      "device": "cpu",
      "diff_ms": 0.37057045847177505,
      "iqr_run1_ms": 0.2990919165313244,
      "iqr_run2_ms": 0.5336630856618285,
      "median_run1_ms": 4.629683448001742,
      "median_run2_ms": 4.259112989529967,
      "model": "small_lstm",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 1.067326171323657
    },
    {
      "device": "cpu",
      "diff_ms": 1.878133974969387,
      "iqr_run1_ms": 5.331506486982107,
      "iqr_run2_ms": 1.0036920430138707,
      "median_run1_ms": 6.7929214565083385,
      "median_run2_ms": 4.914787481538951,
      "model": "small_lstm",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 10.663012973964214
    },
    {
      "device": "cpu",
      "diff_ms": 0.008178874850273132,
      "iqr_run1_ms": 0.012938864529132843,
      "iqr_run2_ms": 0.009658979251980782,
      "median_run1_ms": 0.31438644509762526,
      "median_run2_ms": 0.3062075702473521,
      "model": "small_lstm",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.01135712955147028,
      "iqr_run1_ms": 0.01908000558614731,
      "iqr_run2_ms": 0.010078889317810535,
      "median_run1_ms": 0.32849155832082033,
      "median_run2_ms": 0.31713442876935005,
      "model": "small_lstm",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.09575590956956148,
      "iqr_run1_ms": 0.1390026300214231,
      "iqr_run2_ms": 0.00934535637497902,
      "median_run1_ms": 0.41668349876999855,
      "median_run2_ms": 0.32092758920043707,
      "model": "tinynet",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.0124308280646801,
      "iqr_run1_ms": 0.006666348781436682,
      "iqr_run2_ms": 0.008977367542684078,
      "median_run1_ms": 0.3239688230678439,
      "median_run2_ms": 0.3115379950031638,
      "model": "tinynet",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.04656403325498104,
      "iqr_run1_ms": 0.012814358342438936,
      "iqr_run2_ms": 0.012551434338092804,
      "median_run1_ms": 0.37526898086071014,
      "median_run2_ms": 0.3287049476057291,
      "model": "tinynet",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.033072661608457565,
      "iqr_run1_ms": 0.006311747711151838,
      "iqr_run2_ms": 0.010497577022761106,
      "median_run1_ms": 0.3591610584408045,
      "median_run2_ms": 0.3260883968323469,
      "model": "tinynet",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.0017925631254911423,
      "iqr_run1_ms": 0.006857211701571941,
      "iqr_run2_ms": 0.012957199942320585,
      "median_run1_ms": 0.26377360336482525,
      "median_run2_ms": 0.2619810402393341,
      "model": "tinynet",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 44.23222877085209,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1368.8160087913275,
      "median_run2_ms": 1324.5837800204754,
      "model": "tinynet",
      "operation": "global_wrap_dummy",
      "passed": true,
      "tolerance_ms": 136.88160087913275
    },
    {
      "device": "cuda",
      "diff_ms": 28.9737731218338,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 473.81220012903214,
      "median_run2_ms": 444.83842700719833,
      "model": "tinynet",
      "operation": "first_capture_target",
      "passed": true,
      "tolerance_ms": 47.38122001290321
    },
    {
      "device": "cuda",
      "diff_ms": 5.340655567124486,
      "iqr_run1_ms": 1.4455692726187408,
      "iqr_run2_ms": 1.3374192640185356,
      "median_run1_ms": 29.984441469423473,
      "median_run2_ms": 24.643785902298987,
      "model": "tinynet",
      "operation": "tl_trace",
      "passed": false,
      "tolerance_ms": 2.9984441469423473
    },
    {
      "device": "cuda",
      "diff_ms": 14.216714887879789,
      "iqr_run1_ms": 9.876746800728142,
      "iqr_run2_ms": 0.7906515966169536,
      "median_run1_ms": 40.846927906386554,
      "median_run2_ms": 26.630213018506765,
      "model": "tinynet",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 19.753493601456285
    },
    {
      "device": "cuda",
      "diff_ms": 11.644245125353336,
      "iqr_run1_ms": 9.56779468106106,
      "iqr_run2_ms": 1.7346803797408938,
      "median_run1_ms": 38.327243528328836,
      "median_run2_ms": 26.6829984029755,
      "model": "tinynet",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 19.13558936212212
    },
    {
      "device": "cuda",
      "diff_ms": 1.7163930460810661,
      "iqr_run1_ms": 0.6061301100999117,
      "iqr_run2_ms": 0.5196771817281842,
      "median_run1_ms": 7.880252553150058,
      "median_run2_ms": 6.163859507068992,
      "model": "tinynet",
      "operation": "fastlog_module",
      "passed": false,
      "tolerance_ms": 1.2122602201998234
    },
    {
      "device": "cuda",
      "diff_ms": 0.020761392079293728,
      "iqr_run1_ms": 0.7377623696811497,
      "iqr_run2_ms": 0.7113939500413835,
      "median_run1_ms": 7.264222484081984,
      "median_run2_ms": 7.284983876161277,
      "model": "tinynet",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 1.4755247393622994
    },
    {
      "device": "cuda",
      "diff_ms": 0.40001142770051956,
      "iqr_run1_ms": 0.032376497983932495,
      "iqr_run2_ms": 0.011892872862517834,
      "median_run1_ms": 0.8791114669293165,
      "median_run2_ms": 0.47910003922879696,
      "model": "tinynet",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.08486199658364058,
      "iqr_run1_ms": 0.011869531590491533,
      "iqr_run2_ms": 0.24327734718099236,
      "median_run1_ms": 0.5381959490478039,
      "median_run2_ms": 0.6230579456314445,
      "model": "tinynet",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.0018015271052718163,
      "iqr_run1_ms": 0.006904243491590023,
      "iqr_run2_ms": 0.010538555216044188,
      "median_run1_ms": 4.302603076212108,
      "median_run2_ms": 4.300801549106836,
      "model": "resnet18",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.0065854983404278755,
      "iqr_run1_ms": 0.010815274436026812,
      "iqr_run2_ms": 0.007930968422442675,
      "median_run1_ms": 4.312115488573909,
      "median_run2_ms": 4.305529990233481,
      "model": "resnet18",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.0187259865924716,
      "iqr_run1_ms": 0.035218021366745234,
      "iqr_run2_ms": 0.008365139365196228,
      "median_run1_ms": 4.312269389629364,
      "median_run2_ms": 4.293543403036892,
      "model": "resnet18",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 0.101844547316432,
      "iqr_run1_ms": 0.20804180530831218,
      "iqr_run2_ms": 0.2694596187211573,
      "median_run1_ms": 4.134849994443357,
      "median_run2_ms": 4.033005447126925,
      "model": "resnet18",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 0.5389192374423146
    },
    {
      "device": "cuda",
      "diff_ms": 0.030533061362802982,
      "iqr_run1_ms": 0.5459546227939427,
      "iqr_run2_ms": 0.8499170653522015,
      "median_run1_ms": 4.280406516045332,
      "median_run2_ms": 4.310939577408135,
      "model": "resnet18",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 1.699834130704403
    },
    {
      "device": "cuda",
      "diff_ms": 168.01086603663862,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 683.1669299863279,
      "median_run2_ms": 515.1560639496893,
      "model": "resnet18",
      "operation": "global_wrap_dummy",
      "passed": false,
      "tolerance_ms": 68.31669299863279
    },
    {
      "device": "cuda",
      "diff_ms": 170.92825006693602,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1462.1704169549048,
      "median_run2_ms": 1291.2421668879688,
      "model": "resnet18",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 146.21704169549048
    },
    {
      "device": "cuda",
      "diff_ms": 175.17172743100673,
      "iqr_run1_ms": 252.7188917156309,
      "iqr_run2_ms": 42.84035513410345,
      "median_run1_ms": 895.9685799200088,
      "median_run2_ms": 720.796852489002,
      "model": "resnet18",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 505.4377834312618
    },
    {
      "device": "cuda",
      "diff_ms": 184.77785144932568,
      "iqr_run1_ms": 273.9629987627268,
      "iqr_run2_ms": 70.66771487006918,
      "median_run1_ms": 927.0438519306481,
      "median_run2_ms": 742.2660004813224,
      "model": "resnet18",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 547.9259975254536
    },
    {
      "device": "cuda",
      "diff_ms": 180.97565788775682,
      "iqr_run1_ms": 254.8761480138637,
      "iqr_run2_ms": 64.86615806352347,
      "median_run1_ms": 912.5487268902361,
      "median_run2_ms": 731.5730690024793,
      "model": "resnet18",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 509.7522960277274
    },
    {
      "device": "cuda",
      "diff_ms": 2.6977722300216556,
      "iqr_run1_ms": 2.8452863334678113,
      "iqr_run2_ms": 15.659627970308065,
      "median_run1_ms": 43.94324962049723,
      "median_run2_ms": 41.24547739047557,
      "model": "resnet18",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 31.31925594061613
    },
    {
      "device": "cuda",
      "diff_ms": 7.526991656050086,
      "iqr_run1_ms": 4.893172124866396,
      "iqr_run2_ms": 3.0955293332226574,
      "median_run1_ms": 50.014984561130404,
      "median_run2_ms": 42.48799290508032,
      "model": "resnet18",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 9.786344249732792
    },
    {
      "device": "cuda",
      "diff_ms": 0.17334462609142065,
      "iqr_run1_ms": 0.26788003742694855,
      "iqr_run2_ms": 0.020864303223788738,
      "median_run1_ms": 4.6475090784952044,
      "median_run2_ms": 4.474164452403784,
      "model": "resnet18",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 0.5357600748538971
    },
    {
      "device": "cuda",
      "diff_ms": 0.6785355508327484,
      "iqr_run1_ms": 0.8264105417765677,
      "iqr_run2_ms": 0.7662950665690005,
      "median_run1_ms": 3.8096304051578045,
      "median_run2_ms": 4.488165955990553,
      "model": "resnet18",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 1.6528210835531354
    },
    {
      "device": "cuda",
      "diff_ms": 0.4553148755803704,
      "iqr_run1_ms": 1.285778358578682,
      "iqr_run2_ms": 0.19476859597489238,
      "median_run1_ms": 8.857600507326424,
      "median_run2_ms": 8.402285631746054,
      "model": "gpt2_hf",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 2.571556717157364
    },
    {
      "device": "cuda",
      "diff_ms": 0.8398883510380983,
      "iqr_run1_ms": 0.8004892733879387,
      "iqr_run2_ms": 1.2058989377692342,
      "median_run1_ms": 9.819851955398917,
      "median_run2_ms": 8.979963604360819,
      "model": "gpt2_hf",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 2.4117978755384684
    },
    {
      "device": "cuda",
      "diff_ms": 5.89684525039047,
      "iqr_run1_ms": 0.3368594916537404,
      "iqr_run2_ms": 1.361760892905295,
      "median_run1_ms": 14.51417792122811,
      "median_run2_ms": 8.61733267083764,
      "model": "gpt2_hf",
      "operation": "raw_global_wrapped",
      "passed": false,
      "tolerance_ms": 2.72352178581059
    },
    {
      "device": "cuda",
      "diff_ms": 1.084310351870954,
      "iqr_run1_ms": 0.8230541134253144,
      "iqr_run2_ms": 1.0502012446522713,
      "median_run1_ms": 10.06980740930885,
      "median_run2_ms": 8.985497057437897,
      "model": "gpt2_hf",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 2.1004024893045425
    },
    {
      "device": "cuda",
      "diff_ms": 0.8057489758357406,
      "iqr_run1_ms": 1.6225299914367497,
      "iqr_run2_ms": 0.9319741511717439,
      "median_run1_ms": 9.54145600553602,
      "median_run2_ms": 8.73570702970028,
      "model": "gpt2_hf",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 3.2450599828734994
    },
    {
      "device": "cuda",
      "diff_ms": 305.25347986258566,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1125.7540239021182,
      "median_run2_ms": 820.5005440395325,
      "model": "gpt2_hf",
      "operation": "global_wrap_dummy",
      "passed": false,
      "tolerance_ms": 112.57540239021182
    },
    {
      "device": "cuda",
      "diff_ms": 1347.3629280924797,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 4916.691303020343,
      "median_run2_ms": 3569.3283749278635,
      "model": "gpt2_hf",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 491.6691303020343
    },
    {
      "device": "cuda",
      "diff_ms": 587.045953143388,
      "iqr_run1_ms": 520.8989846869372,
      "iqr_run2_ms": 350.8515282883309,
      "median_run1_ms": 3273.859692621045,
      "median_run2_ms": 2686.813739477657,
      "model": "gpt2_hf",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 1041.7979693738744
    },
    {
      "device": "cuda",
      "diff_ms": 493.6189070576802,
      "iqr_run1_ms": 448.91321420436725,
      "iqr_run2_ms": 433.51604539202526,
      "median_run1_ms": 3293.7860500533134,
      "median_run2_ms": 2800.167142995633,
      "model": "gpt2_hf",
      "operation": "tl_trace_intervention_ready",
      "passed": true,
      "tolerance_ms": 897.8264284087345
    },
    {
      "device": "cuda",
      "diff_ms": 230.9045965084806,
      "iqr_run1_ms": 573.9853167906404,
      "iqr_run2_ms": 485.7488411362283,
      "median_run1_ms": 3285.828482010402,
      "median_run2_ms": 3054.923885501921,
      "model": "gpt2_hf",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 1147.9706335812807
    },
    {
      "device": "cuda",
      "diff_ms": 1.5569253591820598,
      "iqr_run1_ms": 22.746617789380252,
      "iqr_run2_ms": 79.32226615957916,
      "median_run1_ms": 149.3684200104326,
      "median_run2_ms": 150.92534536961466,
      "model": "gpt2_hf",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 158.64453231915832
    },
    {
      "device": "cuda",
      "diff_ms": 4.1955175111070275,
      "iqr_run1_ms": 99.93061661953107,
      "iqr_run2_ms": 75.93822624767199,
      "median_run1_ms": 216.7680209968239,
      "median_run2_ms": 220.96353850793093,
      "model": "gpt2_hf",
      "operation": "fastlog_all",
      "passed": true,
      "tolerance_ms": 199.86123323906213
    },
    {
      "device": "cuda",
      "diff_ms": 1.0301818838343024,
      "iqr_run1_ms": 0.48263039207085967,
      "iqr_run2_ms": 0.8200794109143317,
      "median_run1_ms": 10.933855548501015,
      "median_run2_ms": 11.964037432335317,
      "model": "gpt2_hf",
      "operation": "peer_manual_hooks",
      "passed": true,
      "tolerance_ms": 1.6401588218286633
    },
    {
      "device": "cuda",
      "diff_ms": 0.6747220177203417,
      "iqr_run1_ms": 0.5102414288558066,
      "iqr_run2_ms": 7.1457462618127465,
      "median_run1_ms": 10.83910244051367,
      "median_run2_ms": 11.513824458234012,
      "model": "gpt2_hf",
      "operation": "peer_context_hooks",
      "passed": true,
      "tolerance_ms": 14.291492523625493
    },
    {
      "device": "cpu",
      "diff_ms": 43.175482540391386,
      "iqr_run1_ms": 98.3311683521606,
      "iqr_run2_ms": 66.72569754300639,
      "median_run1_ms": 218.31139794085175,
      "median_run2_ms": 175.13591540046036,
      "model": "tinynet",
      "operation": "aux_validate",
      "passed": true,
      "tolerance_ms": 196.6623367043212
    },
    {
      "device": "cpu",
      "diff_ms": 0.02467283047735691,
      "iqr_run1_ms": 0.010734191164374352,
      "iqr_run2_ms": 0.01015939051285386,
      "median_run1_ms": 0.32815011218190193,
      "median_run2_ms": 0.35282294265925884,
      "model": "tinynet",
      "operation": "aux_compat_report",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.8742399513721466,
      "iqr_run1_ms": 0.5474949721246958,
      "iqr_run2_ms": 1.8884838791564107,
      "median_run1_ms": 25.617876555770636,
      "median_run2_ms": 26.492116507142782,
      "model": "tinynet",
      "operation": "aux_save",
      "passed": true,
      "tolerance_ms": 3.7769677583128214
    },
    {
      "device": "cpu",
      "diff_ms": 5.48592780251056,
      "iqr_run1_ms": 0.9560433682054281,
      "iqr_run2_ms": 10.182927420828491,
      "median_run1_ms": 23.05657009128481,
      "median_run2_ms": 28.54249789379537,
      "model": "tinynet",
      "operation": "aux_load",
      "passed": true,
      "tolerance_ms": 20.365854841656983
    },
    {
      "device": "cpu",
      "diff_ms": 2833.6394210346043,
      "iqr_run1_ms": 114.7739653242752,
      "iqr_run2_ms": 2033.6804577964358,
      "median_run1_ms": 2224.7038120403886,
      "median_run2_ms": 5058.343233074993,
      "model": "resnet18",
      "operation": "aux_validate",
      "passed": true,
      "tolerance_ms": 4067.3609155928716
    },
    {
      "device": "cpu",
      "diff_ms": 0.09432912338525057,
      "iqr_run1_ms": 0.19454472931101918,
      "iqr_run2_ms": 0.04161399556323886,
      "median_run1_ms": 0.5314379232004285,
      "median_run2_ms": 0.625767046585679,
      "model": "resnet18",
      "operation": "aux_compat_report",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 144.51786316931248,
      "iqr_run1_ms": 23.297322099097073,
      "iqr_run2_ms": 135.33150166040286,
      "median_run1_ms": 845.2601904282346,
      "median_run2_ms": 989.7780535975471,
      "model": "resnet18",
      "operation": "aux_save",
      "passed": true,
      "tolerance_ms": 270.66300332080573
    },
    {
      "device": "cpu",
      "diff_ms": 302.53444821573794,
      "iqr_run1_ms": 110.8764405362308,
      "iqr_run2_ms": 370.7773386267945,
      "median_run1_ms": 1255.7068613823503,
      "median_run2_ms": 1558.2413095980883,
      "model": "resnet18",
      "operation": "aux_load",
      "passed": true,
      "tolerance_ms": 741.554677253589
    },
    {
      "device": "cuda",
      "diff_ms": 160.4181743459776,
      "iqr_run1_ms": 31.099403160624206,
      "iqr_run2_ms": 131.16710539907217,
      "median_run1_ms": 393.54500512126833,
      "median_run2_ms": 553.9631794672459,
      "model": "resnet18",
      "operation": "aux_validate",
      "passed": true,
      "tolerance_ms": 262.33421079814434
    },
    {
      "device": "cuda",
      "diff_ms": 0.010238494724035263,
      "iqr_run1_ms": 0.007597904186695814,
      "iqr_run2_ms": 0.0049005611799657345,
      "median_run1_ms": 0.4786914214491844,
      "median_run2_ms": 0.4889299161732197,
      "model": "resnet18",
      "operation": "aux_compat_report",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cuda",
      "diff_ms": 153.17106514703482,
      "iqr_run1_ms": 49.24910538829863,
      "iqr_run2_ms": 107.47183067724109,
      "median_run1_ms": 919.9804499512538,
      "median_run2_ms": 1073.1515150982887,
      "model": "resnet18",
      "operation": "aux_save",
      "passed": true,
      "tolerance_ms": 214.94366135448217
    },
    {
      "device": "cuda",
      "diff_ms": 319.5480649592355,
      "iqr_run1_ms": 114.4842398352921,
      "iqr_run2_ms": 156.3870266545564,
      "median_run1_ms": 1254.7415950102732,
      "median_run2_ms": 1574.2896599695086,
      "model": "resnet18",
      "operation": "aux_load",
      "passed": false,
      "tolerance_ms": 312.7740533091128
    },
    {
      "device": "cpu",
      "diff_ms": 474.3770904606208,
      "iqr_run1_ms": 18.779116624500602,
      "iqr_run2_ms": 294.1872429801151,
      "median_run1_ms": 331.68698952067643,
      "median_run2_ms": 806.0640799812973,
      "model": "gpt2_hooked",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 588.3744859602302
    },
    {
      "device": "cpu",
      "diff_ms": 4834.035238483921,
      "iqr_run1_ms": 292.6216825726442,
      "iqr_run2_ms": 5772.235031879973,
      "median_run1_ms": 6234.283022000454,
      "median_run2_ms": 11068.318260484375,
      "model": "gpt2_hooked",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 11544.470063759945
    },
    {
      "device": "cpu",
      "diff_ms": 24.9952474841848,
      "iqr_run1_ms": 16.98008948005736,
      "iqr_run2_ms": 47.75435081683099,
      "median_run1_ms": 349.2794920457527,
      "median_run2_ms": 374.2747395299375,
      "model": "gpt2_hooked",
      "operation": "peer_transformer_lens",
      "passed": true,
      "tolerance_ms": 95.50870163366199
    },
    {
      "device": "cuda",
      "diff_ms": 0.5271744448691607,
      "iqr_run1_ms": 0.7775482954457402,
      "iqr_run2_ms": 0.7966667180880904,
      "median_run1_ms": 19.4230480119586,
      "median_run2_ms": 18.89587356708944,
      "model": "gpt2_hooked",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 1.94230480119586
    },
    {
      "device": "cuda",
      "diff_ms": 1475.8406279142946,
      "iqr_run1_ms": 1631.0036080540158,
      "iqr_run2_ms": 92.95248880516738,
      "median_run1_ms": 7819.6424439083785,
      "median_run2_ms": 6343.801815994084,
      "model": "gpt2_hooked",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 3262.0072161080316
    },
    {
      "device": "cuda",
      "diff_ms": 6.175235263071954,
      "iqr_run1_ms": 5.755001155193895,
      "iqr_run2_ms": 1.9441667245700955,
      "median_run1_ms": 32.5091821141541,
      "median_run2_ms": 26.333946851082146,
      "model": "gpt2_hooked",
      "operation": "peer_transformer_lens",
      "passed": true,
      "tolerance_ms": 11.51000231038779
    }
  ],
  "passed": false
}
```
