# TorchLens Benchmarks

This directory contains standalone benchmark scripts and generated benchmark
artifacts.

## Performance benchmark suite

`perf_suite.py` drives the 2026-05-14 performance benchmark matrix described in
a private benchmark plan. It launches `perf_runner.py` in a fresh
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
- `host_label.py` names the machine in every record: the `hostname` field holds
  `TORCHLENS_BENCH_HOST_LABEL` (default `benchmark-host`), never the real hostname, and
  captured output spells the checkout as `<repo>`.

## Capture benchmarks

`capture/` holds the matched-contract suites (A: every arm produces the same output;
B: what each TorchLens operation keeps), the operation fixtures, the real-model panel,
the mech-interp workloads and the "what hooks miss" census, all writing one JSON result
format. See `capture/README.md`.

## Intervention overhead

`intervention_overhead.py` is the earlier focused benchmark for TorchLens
intervention primitives. Its committed output lives in
`intervention_overhead_results.md`.

## Pre-hook provenance overhead

`prehook_provenance_overhead.py` measures exhaustive capture on a 181-module
ReLU chain with no user pre-hooks and with one process-global observational
pre-hook. It prints JSON suitable for comparing revisions:

```bash
PYTHONPATH=$PWD python benchmarks/prehook_provenance_overhead.py
```

## Live-index re-seen labels

`live_index_append_counters.py` counts what a re-seen raw label costs the
`LiveIndex` (a multi-pass `Recorder` fold, a failed-partial concatenation, an
amendment replace): full edge rebuilds, child-list operations against the old
plus new parent count, and total Python calls per forward, with a digest of the
index state after every pass. Counters gate; wall time is context only:

```bash
python benchmarks/live_index_append_counters.py --out live_index_counters.json
```

## Save-path counters

`save_path_counters.py` counts the work one `trace.save` of the `tinynet` capture
does on the default portable path (saved format 3): record-table write passes and
encoder calls, verifying re-reads of the record tables, `fsync` calls, Python-level
opens (direct opens, and opens routed through `pathlib` counted apart, so the
split is the same on every Python), and the files and bytes written (split into manifest, metadata and blobs).
Byte counts hold each wall-clock timing float (`TIMING_FLOAT_PATHS`) at a fixed
width, since format 3 writes floats by exact repr; every other byte counts as
written, and the raw size is information only. The pinned values
sit beside the `aux_save` row of `perf_baselines/linux-cpu-provisional.json`
(`metadata.save_path_counters`) and `tests/test_save_path_counters.py` holds them.
Counters gate; wall time and the profiled call count are context only:

```bash
python benchmarks/save_path_counters.py --out save_counters.json
```

## Capture-core cost counters

`capture_core_cost.py` measures what one `tl.trace` costs in deterministic counts on a
fixed model set (`chain1000`, `varied200`, `loopcell100`, `resnet18`, `gpt2_tiny4` and five
small models for the per-trace, per-module and conditional costs): Python and C calls per op by phase
(C1), retained allocation blocks, collector-tracked objects and live record objects (C2,
C4), op-store cells written by capture, ingest and the passes (C3), calls per module call
and per trace (C5, C6), calls for `to_pandas()`, `repr(trace)` and `save` (C7), the share
of op commits on the wrapper fast lane (C8) and the tensor bytes the trace holds during
the forward, split into saved outputs and argument tensors (C9). Each model is measured in
its own interpreter after one warm-up trace; wall time is recorded as context and never
gated.

```bash
python benchmarks/capture_core_cost.py --out run.json     # measure
python benchmarks/capture_core_cost.py --check             # standing bars
python benchmarks/capture_core_cost.py --check --lane W1   # plus one change's own bars
python benchmarks/capture_core_cost.py --compare a.json b.json
```

`perf_baselines/capture_core_bars.json` holds the baseline and the bars, each stored as a
fraction of its baseline counter. It was written once from the baseline (`--write-bars`)
and is not edited by the changes it judges; `--tighten LANE` records a landed change's
measured counters plus 5% as regression gates. `capture_core_counters.py` holds the
instruments and `capture_core_bars.py` the bar arithmetic; `tests/test_capture_core_cost.py`
pins both.
