## TorchLens Profiling Audit (2026-04-27)

### Setup
- Environment: Python 3.11.6, Linux 5.15.0-139-generic x86_64, 20 CPUs, torch 2.8.0+cu128, torchvision 0.23.0+cu128, transformers 5.2.0.
- Models: inline `SmallCNN`; torchvision ResNet50 `weights=None`; Hugging Face `GPT2Model.from_pretrained("gpt2")` loaded successfully.
- Batch sizes used: smoke `4x3x32x32`; ResNet50 `2x3x224x224`; GPT-2 `batch=1`, `seq_len=64`; leak loop `4x3x32x32`.
- Profile artifact paths: all under `/tmp/torchlens_profiling/`; full list in “Raw artifact paths”.

### Time Profile Summary
Smoke baseline, wall time 26.15s:
- Top cumulative: `log_forward_pass` 26.15s; `_run_model_and_save_specified_activations` 26.15s; `_run_and_log_inputs_through_model` 26.06s; `run_and_log_inputs_through_model` 26.05s; `_ensure_model_prepared` 18.38s; `wrap_torch` 18.37s; `patch_detached_references` 16.70s; `_postprocess` 7.35s; `postprocess` 7.28s.
- Top self: `patch_detached_references` 8.64s; `builtins.id` 3.01s; `ast.iter_child_nodes` 2.34s; `warnings._add_filter` 2.03s; `compile` 1.19s; `ast.iter_fields` 1.05s; `_walk_node` 0.60s; `_collect_scopes_from_node` 0.59s; `warnings.simplefilter` 0.57s.
- Surprise >5%: first-call wrapping/patching dominates small models: 70% cumulative in model preparation, 64% in detached-reference patching.

ResNet50, wall time 14.25s:
- Top cumulative: `log_forward_pass` 13.84s; `_run_model_and_save_specified_activations` 13.82s; `_run_and_log_inputs_through_model` 13.76s; `run_and_log_inputs_through_model` 13.75s; PyTorch `Module._call_impl` / ResNet forward 5.06s; `decorated_forward` 5.06s.
- Top self: `patch_detached_references` 1.34s; `torch._C._cuda_getDeviceCount` 1.22s; `dis._get_instructions_bytes` 1.18s; `torch.cuda._raw_device_count_nvml` 0.98s; `dis._unpack_opargs` 0.71s; `_ctypes.dlopen` 0.66s; `compile` 0.60s; `ast.iter_child_nodes` 0.57s.
- Surprise >5%: unconditional CUDA probing/cache clearing costs ~12% cumulative on a CPU run.

GPT-2 small, wall time 33.87s:
- Top cumulative: `log_forward_pass` 33.16s; `_run_model_and_save_specified_activations` 33.15s; `_run_and_log_inputs_through_model` 33.04s; `run_and_log_inputs_through_model` 33.03s; GPT-2 forward / `Module._call_impl` 24.42s; `decorated_forward` 24.34s; `wrapped_func` 24.26s.
- Top self: `dis._get_instructions_bytes` 7.16s; `dis._unpack_opargs` 5.51s; `dis.findlinestarts` 2.54s; `next` 1.93s; `patch_detached_references` 1.72s; `type.__new__` 1.55s; `dis.findlabels` 1.31s; `_get_col_offset` 0.94s; `torch.cuda._raw_device_count_nvml` 0.86s.
- Surprise >5%: bytecode/column-offset inspection is very expensive: `dis.*` self time is ~16.5s, about half of profiled runtime.

Two-pass ResNet50, wall time 13.42s, failed before completing pass 2:
- Top cumulative before failure: `log_forward_pass` 12.98s; `_run_and_log_inputs_through_model` 12.90s; `run_and_log_inputs_through_model` 12.90s; `_run_model_and_save_specified_activations` 12.78s; ResNet forward / `decorated_forward` 4.50s.
- Top self: `patch_detached_references` 1.29s; `torch._C._cuda_getDeviceCount` 1.25s; `dis._get_instructions_bytes` 1.11s; `torch.cuda._raw_device_count_nvml` 0.80s; `_ctypes.dlopen` 0.71s; `dis._unpack_opargs` 0.65s; `compile` 0.62s; `ast.iter_child_nodes` 0.55s.
- Surprise >5%: two-pass sparse capture failed for valid-looking selectors before any fast-pass benefit was observed.

### Memory Profile Summary
- Peak RSS per target:
  - Smoke: 777.03 MiB; after `del model_log; gc.collect()`: 745.35 MiB.
  - ResNet50: 1394.75 MiB; after delete/GC: 1370.94 MiB.
  - GPT-2: 1451.38 MiB; after delete/GC: 1427.70 MiB.
  - Two-pass failed run: 1145.91 MiB; after delete/GC: 1122.37 MiB.
- Allocation hotspots from memray:
  - ResNet50: 8,800,145 allocations, 6.810 GB total allocated, 5.283 GB memray peak. Memray attributed nearly all large allocations to unavailable native stack traces, consistent with PyTorch/C allocator paths.
  - GPT-2: 20,743,290 allocations, 2.436 GB total allocated, 400.014 MB memray peak. Most allocations were small Python allocator churn; largest stack was unavailable.
  - Two-pass failed ResNet50: 8,873,683 allocations, 6.553 GB total allocated, 5.014 GB memray peak.
- Per-target tensor lifetime check: RSS did not return to baseline after deleting `model_log`, but the RSS-only 50-pass control stayed flat from 710.09 MiB to 709.96 MiB with tensor count stable in the instrumented loop. This points more to allocator retention and profiler overhead than an unbounded retained-Tensor leak in the small-model path.

### Leak Loop Results
- Iteration RSS curve summary:
  - Full instrumented run: apparent stepwise growth from 735.93 MiB to 1003.29 MiB, mainly at 10-iteration Pympler/tracemalloc sampling points.
  - RSS-only control: flat/sawtooth, 710.09 MiB at iteration 1 and 709.96 MiB at iteration 50; max 713.62 MiB. No unbounded TorchLens leak reproduced without heavy profilers.
- tracemalloc top-10 allocation diffs, first vs last snapshot:
  - `python3.11/ast.py:50`: +10.60 MB, +139,056 objects.
  - `functools.py:52`: +1.00 MB, +3,917 objects.
  - `linecache.py:137`: +626.6 KB, +6,025 objects.
  - `torchlens/decoration/torch_funcs.py:333`: +396.0 KB, +3,807 objects.
  - `importlib._bootstrap_external:729`: +360.2 KB, +1,942 objects.
  - `torchlens/decoration/torch_funcs.py:332`: +227.9 KB, +1,899 objects.
  - `torchlens/decoration/torch_funcs.py:685`: +182.0 KB, +1,080 objects.
  - `torchlens/postprocess/ast_branches.py:1140`: +155.4 KB, +2,188 objects.
  - `torchlens/decoration/torch_funcs.py:631`: +151.0 KB, +3,776 objects.
  - `torchlens/decoration/torch_funcs.py:641`: +147.5 KB, +1 object.
- gc count delta: control run collected 639, 2556, 2556, 2556, 2556 objects at iterations 10/20/30/40/50 and RSS returned to baseline.
- If unbounded growth: not confirmed. Instrumented growth was profiler-driven; top TorchLens AST object counts stayed constant after first sample: `BranchInterval=808`, `BoolConsumer=695`, `ConditionalRecord=583`, `ScopeEntry=429`.

### Findings (severity-ordered)
1. **Severity:** HIGH
   **Location:** `torchlens/user_funcs.py:665`, `torchlens/capture/output_tensors.py:691` and `:709`
   **Category:** correctness-risk
   **Observed:** ResNet50 two-pass mode failed for `['conv2d_1_1']`, `['conv2d']`, `[1]`, and `[-1]` with “computational graph changed” despite same model/input.
   **Why it matters:** Sparse activation capture is unusable for this common model path; users pay full pass cost then receive no result.
   **Suggested fix direction:** Add diagnostics to report the first mismatching raw label / layer type / parent set, then fix fast-path structural comparison or label reconstruction for ResNet module outputs.

2. **Severity:** MEDIUM
   **Location:** `torchlens/utils/introspection.py:41`, `torchlens/utils/introspection.py:518`
   **Category:** time-perf
   **Observed:** GPT-2 spends ~16.5s self time in `dis.*`; `_get_col_offset()` disassembles code objects per captured frame/op.
   **Why it matters:** On GPT-2 this is roughly half the profiled runtime.
   **Suggested fix direction:** Cache instruction offset-to-column maps per code object, or disable column offsets unless branch attribution needs them.

3. **Severity:** MEDIUM
   **Location:** `torchlens/postprocess/control_flow.py:41`, `torchlens/postprocess/ast_branches.py:294`, `torchlens/postprocess/ast_branches.py:409`
   **Category:** time-perf / memory-perf
   **Observed:** Step 5 branch attribution costs 2.14s on ResNet50 and 2.29s on GPT-2; smoke spends 7.20s due first-call AST indexing.
   **Why it matters:** Even models with no meaningful user conditional branches pay per-op AST attribution cost.
   **Suggested fix direction:** Fast-skip branch attribution when no terminal bools / conditional keys exist, and avoid calling `attribute_op()` for every op unless needed.

4. **Severity:** MEDIUM
   **Location:** `torchlens/postprocess/__init__.py:177`, `torchlens/postprocess/__init__.py:257`, `torchlens/capture/trace.py:532`
   **Category:** time-perf
   **Observed:** CPU-only runs still trigger CUDA availability/device-count paths; ResNet50 shows ~1.64s cumulative in CUDA device count/NVML checks.
   **Why it matters:** Adds ~12% overhead to the profiled CPU ResNet50 run.
   **Suggested fix direction:** Guard `torch.cuda.empty_cache()` behind cached CUDA availability and avoid forcing CUDA lazy init on CPU-only capture.

5. **Severity:** LOW
   **Location:** `torchlens/decoration/torch_funcs.py:894`
   **Category:** time-perf
   **Observed:** `patch_detached_references()` dominates first-call smoke runtime: 16.70s cumulative, 8.64s self.
   **Why it matters:** Small models have very high first-call latency.
   **Suggested fix direction:** Make detached-reference patching idempotent and narrower after first import, or expose an opt-out for environments that do not need deep `sys.modules` crawling.

6. **Severity:** LOW
   **Location:** `torchlens/postprocess/ast_branches.py:31`, `torchlens/postprocess/ast_branches.py:350`
   **Category:** memory-perf
   **Observed:** AST/file caches persist by filename. Leak loop shows constant TorchLens AST object counts after warmup, not unbounded growth.
   **Why it matters:** Long-running processes that touch many unique source files can accumulate parsed ASTs.
   **Suggested fix direction:** Add a bounded cache or public cache-clear guidance for long-lived services.

### Recommendations (prioritized)
- HIGH: Fix/tighten diagnostics for two-pass graph mismatch on ResNet50; this is the only audited issue that blocked a requested workflow.
- MEDIUM: Cache bytecode column-offset lookup in `_get_col_offset()`; this is the clearest large GPT-2 hotspot.
- MEDIUM: Add a no-condition fast path for Step 5 branch attribution.
- MEDIUM: Guard CPU runs from CUDA cache/device probes.
- LOW / NIT: Reduce first-call `patch_detached_references()` cost; consider bounding AST file cache for daemon-style use.

### Raw artifact paths
- `/tmp/torchlens_profiling/environment.json` -- Python, OS, torch, torchvision, transformers, memray, Pympler versions.
- `/tmp/torchlens_profiling/profile_torchlens.py` -- main profiling/probe driver.
- `/tmp/torchlens_profiling/smoke.prof` -- smoke cProfile artifact.
- `/tmp/torchlens_profiling/smoke_pstats.txt` -- smoke pstats text summary.
- `/tmp/torchlens_profiling/smoke_pstats.json` -- smoke top function summary.
- `/tmp/torchlens_profiling/smoke_result.json` -- smoke wall/RSS/profile metadata.
- `/tmp/torchlens_profiling/smoke_tracemalloc_before.snap` -- smoke pre-run tracemalloc snapshot.
- `/tmp/torchlens_profiling/smoke_tracemalloc_after.snap` -- smoke post-run tracemalloc snapshot.
- `/tmp/torchlens_profiling/resnet50.prof` -- ResNet50 cProfile artifact.
- `/tmp/torchlens_profiling/resnet50.memray.bin` -- ResNet50 memray raw allocation trace.
- `/tmp/torchlens_profiling/resnet50_memray_stats.txt` -- ResNet50 memray stats.
- `/tmp/torchlens_profiling/resnet50_pstats.txt` -- ResNet50 pstats text summary.
- `/tmp/torchlens_profiling/resnet50_pstats.json` -- ResNet50 top function summary.
- `/tmp/torchlens_profiling/resnet50_result.json` -- ResNet50 wall/RSS/profile metadata.
- `/tmp/torchlens_profiling/gpt2.prof` -- GPT-2 cProfile artifact.
- `/tmp/torchlens_profiling/gpt2.memray.bin` -- GPT-2 memray raw allocation trace.
- `/tmp/torchlens_profiling/gpt2_memray_stats.txt` -- GPT-2 memray stats.
- `/tmp/torchlens_profiling/gpt2_pstats.txt` -- GPT-2 pstats text summary.
- `/tmp/torchlens_profiling/gpt2_pstats.json` -- GPT-2 top function summary.
- `/tmp/torchlens_profiling/gpt2_result.json` -- GPT-2 wall/RSS/profile metadata.
- `/tmp/torchlens_profiling/two_pass_layer_label.txt` -- requested two-pass selector.
- `/tmp/torchlens_profiling/two_pass_resnet50.prof` -- failed two-pass cProfile artifact.
- `/tmp/torchlens_profiling/two_pass_resnet50.memray.bin` -- failed two-pass memray trace.
- `/tmp/torchlens_profiling/two_pass_resnet50_memray_stats.txt` -- failed two-pass memray stats.
- `/tmp/torchlens_profiling/two_pass_resnet50_pstats.txt` -- failed two-pass pstats text summary.
- `/tmp/torchlens_profiling/two_pass_resnet50_pstats.json` -- failed two-pass top function summary.
- `/tmp/torchlens_profiling/two_pass_resnet50_result.json` -- failed two-pass error/RSS/profile metadata.
- `/tmp/torchlens_profiling/leak_loop_result.json` -- instrumented 50-iteration leak loop results.
- `/tmp/torchlens_profiling/leak_loop_rss.csv` -- instrumented leak-loop RSS by iteration.
- `/tmp/torchlens_profiling/leak_loop_tracemalloc_first.snap` -- leak-loop initial tracemalloc snapshot.
- `/tmp/torchlens_profiling/leak_loop_tracemalloc_last.snap` -- leak-loop final tracemalloc snapshot.
- `/tmp/torchlens_profiling/leak_loop_rss_control.py` -- RSS-only leak control probe.
- `/tmp/torchlens_profiling/leak_loop_rss_control.json` -- RSS-only control results.
- `/tmp/torchlens_profiling/leak_loop_rss_control.csv` -- RSS-only control RSS by iteration.
