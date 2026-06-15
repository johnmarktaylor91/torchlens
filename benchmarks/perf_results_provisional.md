# TorchLens Performance Benchmark Results - 2026-06-14

Run status: **ok**. Wall clock: 161.1 s.

## Methodology

Each operation/model/device/pass cell runs in a fresh subprocess. Timing cells use 5 untimed warmups and 50 measured wall-clock samples unless the row is a one-time startup cost. Memory cells are separate subprocesses that record USS after setup, run the operation 10 untimed times, and report `uss_delta_mb_memory_pass`. CUDA memory columns are true allocator peaks after `torch.cuda.reset_peak_memory_stats()`.

Gradient mode is enabled for headline rows, models are in eval mode, dtype is float32, autocast is not used, TF32 is disabled, and seeds are fixed to 0. `Trace.rerun(model, x)` uses the round-4 steady-state contract: capture once before the timing loop, run warmups, then measure repeated reruns on that same Trace.

## Environment

```json
{
  "cpu_model": "Intel(R) Core(TM) i9-9900X CPU @ 3.50GHz",
  "cuda": "12.8",
  "cuda_available": true,
  "hostname": "zmachine",
  "install_notes": {
    "baukit": "pip install baukit failed: no matching distribution found",
    "transformer_lens": "installed separately after baukit failed the combined install"
  },
  "os": "Linux 5.15.0-139-generic",
  "torch": "2.8.0+cu128",
  "torchlens_git_sha": "a1df040",
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

### tinynet / cpu

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pure raw forward | 0.9 | 0.9 | 0.9 | 0.0 | 10.4 | N/A | N/A | ok |
| Raw forward, TL imported | 1.0 | 0.9 | 1.2 | 0.3 | 10.2 | N/A | N/A | ok |
| Raw forward, global wrappers installed | 0.9 | 0.8 | 1.6 | 0.1 | 8.6 | N/A | N/A | ok |
| Raw forward, target model prepared | 0.9 | 0.9 | 1.0 | 0.0 | 1.6 | N/A | N/A | ok |
| Raw + torch.inference_mode floor | 1.1 | 1.1 | 5.7 | 0.1 | 9.6 | N/A | N/A | ok |
| global-wrap-on-dummy startup | 1660.7 | 1660.7 | 1660.7 | 0.0 | 119.9 | N/A | N/A | ok |
| first-capture-of-target-model startup | 344.0 | 344.0 | 344.0 | 0.0 | 64.4 | N/A | N/A | ok |
| TL Trace, every-op capture | 38.8 | 36.0 | 121.4 | 23.1 | 27.0 | N/A | N/A | ok |
| TL Trace, phase profile | 41.0 | 36.3 | 175.9 | 11.6 | 27.1 | N/A | N/A | ok |
| Trace.rerun(model, x) | 49.4 | 47.6 | 114.3 | 2.8 | 27.6 | N/A | N/A | ok |
| fastlog module-boundary metadata | 15.7 | 13.3 | 39.0 | 3.1 | 6.9 | N/A | N/A | ok |
| tl.save fresh tlspec path | 29.0 | 27.5 | 39.8 | 5.0 | 1.1 | N/A | N/A | ok |
| tl.load saved fixture | 30.2 | 25.9 | 43.5 | 6.4 | 9.7 | N/A | N/A | ok |

## Peer Comparison Tables

## Auxiliary Primitives

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tl.save fresh tlspec path | 29.0 | 27.5 | 39.8 | 5.0 | 1.1 | N/A | N/A | ok |
| tl.load saved fixture | 30.2 | 25.9 | 43.5 | 6.4 | 9.7 | N/A | N/A | ok |

## Decoration Overhead

| Operation | median_ms | p5_ms | p95_ms | IQR_ms | USS delta MB (10-run memory pass) | max_allocated_mb | max_reserved_mb | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| global-wrap-on-dummy startup | 1660.7 | 1660.7 | 1660.7 | 0.0 | 119.9 | N/A | N/A | ok |
| first-capture-of-target-model startup | 344.0 | 344.0 | 344.0 | 0.0 | 64.4 | N/A | N/A | ok |

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
      "diff_ms": 0.002411194145679474,
      "iqr_run1_ms": 0.021876301616430283,
      "iqr_run2_ms": 0.011071155313402414,
      "median_run1_ms": 0.9162630885839462,
      "median_run2_ms": 0.9138518944382668,
      "model": "tinynet",
      "operation": "raw_forward",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.04889036063104868,
      "iqr_run1_ms": 0.2602535532787442,
      "iqr_run2_ms": 0.12998719466850162,
      "median_run1_ms": 0.9557829471305013,
      "median_run2_ms": 0.9068925864994526,
      "model": "tinynet",
      "operation": "raw_tl_import",
      "passed": true,
      "tolerance_ms": 0.5205071065574884
    },
    {
      "device": "cpu",
      "diff_ms": 0.023446627892553806,
      "iqr_run1_ms": 0.10359403677284718,
      "iqr_run2_ms": 0.021824962459504604,
      "median_run1_ms": 0.923196435905993,
      "median_run2_ms": 0.9466430637985468,
      "model": "tinynet",
      "operation": "raw_global_wrapped",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.02202799078077078,
      "iqr_run1_ms": 0.023970205802470446,
      "iqr_run2_ms": 0.026847352273762226,
      "median_run1_ms": 0.9337475057691336,
      "median_run2_ms": 0.9557754965499043,
      "model": "tinynet",
      "operation": "raw_target_prepared",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 0.29660749714821577,
      "iqr_run1_ms": 0.10881462367251515,
      "iqr_run2_ms": 0.014323682989925146,
      "median_run1_ms": 1.1443599360063672,
      "median_run2_ms": 0.8477524388581514,
      "model": "tinynet",
      "operation": "raw_inference_mode",
      "passed": true,
      "tolerance_ms": 0.5
    },
    {
      "device": "cpu",
      "diff_ms": 168.73727506026626,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 1660.6599220540375,
      "median_run2_ms": 1491.9226469937712,
      "model": "tinynet",
      "operation": "global_wrap_dummy",
      "passed": false,
      "tolerance_ms": 166.06599220540375
    },
    {
      "device": "cpu",
      "diff_ms": 408.1111040432006,
      "iqr_run1_ms": 0.0,
      "iqr_run2_ms": 0.0,
      "median_run1_ms": 344.0105449408293,
      "median_run2_ms": 752.1216489840299,
      "model": "tinynet",
      "operation": "first_capture_target",
      "passed": false,
      "tolerance_ms": 34.40105449408293
    },
    {
      "device": "cpu",
      "diff_ms": 0.07658486720174551,
      "iqr_run1_ms": 23.073158808983862,
      "iqr_run2_ms": 14.69894970068708,
      "median_run1_ms": 38.77976490184665,
      "median_run2_ms": 38.7031800346449,
      "model": "tinynet",
      "operation": "tl_trace",
      "passed": true,
      "tolerance_ms": 46.146317617967725
    },
    {
      "device": "cpu",
      "diff_ms": 2.523454953916371,
      "iqr_run1_ms": 11.618790624197572,
      "iqr_run2_ms": 11.962658376432955,
      "median_run1_ms": 40.96457140985876,
      "median_run2_ms": 38.44111645594239,
      "model": "tinynet",
      "operation": "tl_trace_profile",
      "passed": true,
      "tolerance_ms": 23.92531675286591
    },
    {
      "device": "cpu",
      "diff_ms": 4.597877152264118,
      "iqr_run1_ms": 2.773555868770927,
      "iqr_run2_ms": 15.489956014789641,
      "median_run1_ms": 49.44387834984809,
      "median_run2_ms": 54.04175550211221,
      "model": "tinynet",
      "operation": "tl_rerun",
      "passed": true,
      "tolerance_ms": 30.979912029579282
    },
    {
      "device": "cpu",
      "diff_ms": 0.3771515330299735,
      "iqr_run1_ms": 3.1259426032193005,
      "iqr_run2_ms": 10.022274102084339,
      "median_run1_ms": 15.680309501476586,
      "median_run2_ms": 16.05746103450656,
      "model": "tinynet",
      "operation": "fastlog_module",
      "passed": true,
      "tolerance_ms": 20.044548204168677
    },
    {
      "device": "cpu",
      "diff_ms": 0.8165288018062711,
      "iqr_run1_ms": 4.967630375176668,
      "iqr_run2_ms": 4.769909719470888,
      "median_run1_ms": 29.013828607276082,
      "median_run2_ms": 29.830357409082353,
      "model": "tinynet",
      "operation": "aux_save",
      "passed": true,
      "tolerance_ms": 9.935260750353336
    },
    {
      "device": "cpu",
      "diff_ms": 2.338214428164065,
      "iqr_run1_ms": 6.361376203130931,
      "iqr_run2_ms": 4.509843536652625,
      "median_run1_ms": 30.193276004865766,
      "median_run2_ms": 27.8550615767017,
      "model": "tinynet",
      "operation": "aux_load",
      "passed": true,
      "tolerance_ms": 12.722752406261861
    }
  ],
  "passed": false
}
```
