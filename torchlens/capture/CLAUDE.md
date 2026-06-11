# capture/ - Real-Time Operation Logging

## What This Does
Captures tensor operations while a model forward pass runs under `active_logging()`.
It supports exhaustive full-graph capture, fast second-pass out capture, backward
graph capture, and fastlog's lightweight `RecordContext` construction.

## Files

| File | Purpose |
|------|---------|
| `trace.py` | Forward-pass orchestration, input normalization, session setup/cleanup, two-pass capture |
| `output_tensors.py` | Core forward logging, live hooks, exhaustive/fast paths, parent links, out saves |
| `source_tensors.py` | Logs model inputs and buffers as source graph nodes |
| `tensor_tracking.py` | Barcode tracking, parent/child labels, arg hashes, backward hook metadata |
| `arg_positions.py` | 3-tier tensor/parameter extraction: static table, dynamic cache, BFS fallback |
| `salient_args.py` | Human-readable function configuration metadata |
| `flops.py` | Forward and backward FLOPs estimates with registry hooks |
| `backward.py` | First-class backward graph capture, grad hooks, streaming grad refs |
| `__init__.py` | Empty package marker |

## How It Connects

Decorated wrappers in `decoration/torch_funcs.py` call this package for every logged
operation. `trace.py` owns the forward session; `output_tensors.py` creates raw
`Op` entries consumed by `postprocess/`. `source_tensors.py` creates roots for
inputs and buffers. `backward.py` runs after a forward log when backward capture is
requested.

`torch.func` / functorch transform entry points are captured as single boundary
ops. The boundary op stores transform metadata, a replay callable, and parent edges
from the transform inputs; the inner transformed function runs with logging paused.
Unattributed tensor-argument markers are collected during arg resolution and warned
once in postprocess.

Fastlog reuses the wrapper hot path but stores `ActivationRecord` data through
`fastlog/_orchestrator.py` instead of building a full `Trace`.

## Key Functions

### trace.py
- `run_and_log_inputs_through_model()` - core runner used by `trace()`.
- `save_new_outs()` - replay-like out refresh on an existing graph.
- `_run_model_and_save_specified_outs()` is called from `user_funcs.py` for two-pass
  selective save behavior.

Ordering matters: capture RNG/autocast state, enter `active_logging()`, run model forward,
cleanup model session, then postprocess.

### output_tensors.py
- `log_function_output_tensors()` - dispatches to exhaustive or fast behavior and applies
  live intervention hooks.
- `log_function_output_tensors_exhaustive()` - builds raw per-output `Op` entries.
- `log_function_output_tensors_fast()` - validates graph alignment and updates selected tensor data.
- `apply_live_hooks_to_outputs()` - applies normalized intervention hooks during capture.

### tensor_tracking.py
- `_get_equivalence_class()` - structural fingerprint used by loop detection.
- Backward hook helpers link tensors to `GradFn` and grad records.

### backward.py
- `log_backward()` - captures autograd graph and grads for a logged forward pass.
- `recording_backward()` - context/helper surface exposed from `Trace.recording_backward`.

## Fast vs Exhaustive
Exhaustive capture owns metadata truth. Fast capture is allowed only when it can align with
the exhaustive pass by operation counter, function name, and parent sets. Any graph
divergence should fail clearly rather than silently saving mismatched outs.

## Training Semantics
Do not introduce bare `.detach()` or `torch.no_grad()` in capture paths. Tensor copy/detach
behavior is controlled by save options and `backward_ready=True`; use `safe_copy()` and existing
storage routing.
