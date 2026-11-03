# TorchLens Performance Benchmark Results - 2026-11-03

Run status: **ok**. Wall clock: 239.4 s.

## Methodology

Each operation/model/device/pass cell runs in a fresh subprocess with torch intra-op threads pinned (recorded as `torch_num_threads` in each cell's env block; wall times are not comparable across differing thread pins). Timing cells use 5 untimed warmups and 50 measured samples unless the row is a one-time startup cost; each sample records both wall clock (`perf_counter`) and process CPU time (`process_time`, `cpu_*` stats). Memory cells are separate subprocesses that record USS after setup, run the operation 10 untimed times, and report the end delta `uss_delta_mb_memory_pass` plus the phase-local peaks `uss_peak_delta_mb_memory_pass` and `phase_rss_high_water_delta_mb`. Peaks cover transients INSIDE the measured calls: the RSS high-water mark is reset before the loop where the platform allows (`rss_high_water_phase_local`; elsewhere the delta subtracts the setup-phase peak) and a concurrent sampler thread polls RSS/USS during the runs (`sampled_rss_peak_delta_mb`). CUDA memory columns are true allocator peaks after `torch.cuda.reset_peak_memory_stats()`.

Gradient mode is enabled for headline rows, models are in eval mode, dtype is float32, autocast is not used, TF32 is disabled, and seeds are fixed to 0. `Trace.run(model, x)` uses the round-4 steady-state contract: capture once before the timing loop, run warmups, then measure repeated reruns on that same Trace.

## Environment

```json
{
  "cpu_model": "Intel(R) Xeon(R) CPU @ 2.80GHz",
  "cuda": null,
  "cuda_available": false,
  "hostname": "benchmark-host",
  "install_notes": {
    "baukit": "pip install baukit failed: no matching distribution found",
    "transformer_lens": "installed separately after baukit failed the combined install"
  },
  "os": "Linux 7.0.0-1011-gcp",
  "torch": "2.13.0+cpu",
  "torch_threads_pin": 4,
  "torchlens_git_sha": "f46f78e",
  "versions": {
    "baukit": null,
    "captum": null,
    "nnsight": null,
    "psutil": null,
    "transformer_lens": null
  }
}
```

## Per-cell Tables

### tinynet / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 2.1 | 1.5 | 5.0 | 2.6 | N/A | N/A | N/A | ok |
| Raw forward, TL imported | 1.6 | 1.5 | 1.9 | 0.1 | N/A | N/A | N/A | ok |
| Raw forward, global wrappers installed | 1.7 | 1.6 | 1.9 | 0.1 | N/A | N/A | N/A | ok |
| Raw forward, target model prepared | 1.6 | 1.5 | 1.7 | 0.0 | N/A | N/A | N/A | ok |
| Raw + torch.inference_mode floor | 1.7 | 1.5 | 1.8 | 0.2 | N/A | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1301.3 | 1301.3 | 1301.3 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 1348.8 | 1348.8 | 1348.8 | 0.0 | N/A | N/A | N/A | ok |
| TL Trace, every-op capture | 65.3 | 62.9 | 147.5 | 2.8 | N/A | N/A | N/A | ok |
| TL Trace, phase profile | 68.9 | 64.3 | 135.7 | 3.7 | N/A | N/A | N/A | ok |
| Trace.run(model, x) | 330.8 | 319.5 | 481.5 | 10.6 | N/A | N/A | N/A | ok |
| fastlog module-boundary metadata | 29.7 | 27.4 | 35.5 | 3.3 | N/A | N/A | N/A | ok |
| Trace.save fresh tlspec path | 185.1 | 178.0 | 204.9 | 7.5 | N/A | N/A | N/A | ok |
| tl.load saved fixture | 106.3 | 102.8 | 128.4 | 8.7 | N/A | N/A | N/A | ok |

## Peer Comparison Tables

## Auxiliary Primitives

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Trace.save fresh tlspec path | 185.1 | 178.0 | 204.9 | 7.5 | N/A | N/A | N/A | ok |
| tl.load saved fixture | 106.3 | 102.8 | 128.4 | 8.7 | N/A | N/A | N/A | ok |

## Decoration Overhead

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| global-wrap-on-dummy startup | 1301.3 | 1301.3 | 1301.3 | 0.0 | N/A | N/A | N/A | ok |
| first-capture-of-target-model startup | 1348.8 | 1348.8 | 1348.8 | 0.0 | N/A | N/A | N/A | ok |

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
- No-save Trace rows use `save=lambda _ctx: False`, which records metadata for every op but saves no activation tensors.

## Rerun Tolerance

```json
{
  "checks": [
    {
      "device": "cpu",
      "diff_ms": 3.7180045000000383,
      "iqr_run1_ms": 8.514687249999819,
      "iqr_run2_ms": 3.077369999999968,
      "median_run1_ms": 8.112285000000163,
      "median_run2_ms": 4.394280500000125,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 17.029374499999637
    },
    {
      "device": "cpu",
      "diff_ms": 3.14162249999983,
      "iqr_run1_ms": 3.1770389999999926,
      "iqr_run2_ms": 3.10026450000006,
      "median_run1_ms": 7.57876149999992,
      "median_run2_ms": 4.43713900000009,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 6.354077999999985
    },
    {
      "device": "cpu",
      "diff_ms": 3.3452560000000187,
      "iqr_run1_ms": 2.092780999999988,
      "iqr_run2_ms": 3.042203999999882,
      "median_run1_ms": 7.675252499999896,
      "median_run2_ms": 4.3299964999998775,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 4.185561999999976
    },
    {
      "device": "cpu",
      "diff_ms": 3.2585889999998674,
      "iqr_run1_ms": 3.0592337500000566,
      "iqr_run2_ms": 3.036230749999702,
      "median_run1_ms": 7.570833499999763,
      "median_run2_ms": 4.3122444999998955,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 6.118467500000113
    },
    {
      "device": "cpu",
      "diff_ms": 1.590361500000026,
      "iqr_run1_ms": 3.129365500000204,
      "iqr_run2_ms": 3.0553867499999443,
      "median_run1_ms": 7.595500500000019,
      "median_run2_ms": 6.005138999999993,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 6.258731000000408
    },
    {
      "device": "cpu",
      "diff_ms": 9.09107800000038,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1301.1375340000002,
      "median_run2_ms": 1292.0464559999998,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "global_wrap_dummy",
      "passed": true,
      "tolerance_ms": 130.11375340000004
    },
    {
      "device": "cpu",
      "diff_ms": 199.84509200000002,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1546.1737150000001,
      "median_run2_ms": 1346.328623,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 154.61737150000002
    },
    {
      "device": "cpu",
      "diff_ms": 0.9509560000000761,
      "iqr_run1_ms": 7.188522499999465,
      "iqr_run2_ms": 11.868562000000537,
      "median_run1_ms": 149.72081699999995,
      "median_run2_ms": 150.67177300000003,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 14.972081699999997
    },
    {
      "device": "cpu",
      "diff_ms": 1.3070254999991562,
      "iqr_run1_ms": 11.943315499999073,
      "iqr_run2_ms": 18.482791000000276,
      "median_run1_ms": 158.33966949999922,
      "median_run2_ms": 157.03264400000006,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "tl_trace_profile",
      "passed": true,
      "tolerance_ms": 23.886630999998147
    },
    {
      "device": "cpu",
      "diff_ms": 29.751995999997575,
      "iqr_run1_ms": 14.76836424999999,
      "iqr_run2_ms": 91.6026049999989,
      "median_run1_ms": 523.7459830000014,
      "median_run2_ms": 553.497978999999,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 52.374598300000144
    },
    {
      "device": "cpu",
      "diff_ms": 1.3978149999997953,
      "iqr_run1_ms": 4.979111749999987,
      "iqr_run2_ms": 4.067713249999102,
      "median_run1_ms": 84.57089900000003,
      "median_run2_ms": 83.17308400000023,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 9.958223499999974
    },
    {
      "device": "cpu",
      "diff_ms": 7.1360589999995625,
      "iqr_run1_ms": 7.389215749999721,
      "iqr_run2_ms": 4.0408714999991275,
      "median_run1_ms": 143.09705299999996,
      "median_run2_ms": 135.9609940000004,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "aux_save",
      "passed": true,
      "tolerance_ms": 14.778431499999442
    },
    {
      "device": "cpu",
      "diff_ms": 1.8225495000003775,
      "iqr_run1_ms": 8.724155249999455,
      "iqr_run2_ms": 2.109467999999552,
      "median_run1_ms": 106.3321134999997,
      "median_run2_ms": 104.50956399999933,
      "metric": "cpu_median_ms",
      "model": "tinynet",
      "operation": "aux_load",
      "passed": true,
      "tolerance_ms": 17.44831049999891
    }
  ],
  "passed": false
}
```
