# TorchLens Benchmarks

This directory contains standalone benchmark scripts and generated benchmark
artifacts.

## Performance benchmark suite

`perf_suite.py` drives the 2026-05-14 performance benchmark matrix described in
`.research/perf-benchmarks_PLAN.md`. It launches `perf_runner.py` in a fresh
subprocess for each operation/model/device/pass cell, writes
`perf_results_2026-05-14.json`, and renders `perf_results_2026-05-14.md`.

Typical commands:

```bash
python benchmarks/perf_suite.py --smoke
python benchmarks/perf_suite.py --rerun
```

Supporting files:

- `perf_models.py` builds the benchmark model/input fixtures without importing
  TorchLens in pure raw-forward subprocesses.
- `perf_peers.py` contains peer-tool hook/capture implementations and structured
  import skips.
- `perf_runner.py` executes one timing or memory pass cell and writes JSON.

## Intervention overhead

`intervention_overhead.py` is the earlier focused benchmark for TorchLens
intervention primitives. Its committed output lives in
`intervention_overhead_results.md`.
