# capture/ - Real-Time Operation Logging

## What This Does
Captures tensor operations while a model forward pass runs under `active_logging()`.
It supports exhaustive full-graph capture, fast second-pass activation capture, backward
graph capture, and fastlog's lightweight `RecordContext` construction.

## Files

| File | Purpose |
|------|---------|
| `trace.py` | Forward-pass orchestration, input normalization, session setup/cleanup, two-pass capture |
| `output_tensors.py` | Core forward logging, live hooks, exhaustive/fast paths, parent links, activation saves |
| `source_tensors.py` | Logs model inputs and buffers as source graph nodes |
| `tensor_tracking.py` | Barcode tracking, parent/child labels, arg hashes, backward hook metadata |
| `arg_positions.py` | 3-tier tensor/parameter extraction: static table, dynamic cache, BFS fallback |
| `salient_args.py` | Human-readable function configuration metadata |
| `flops.py` | Forward and backward FLOPs estimates with registry hooks |
| `backward.py` | First-class backward graph capture, gradient hooks, streaming gradient refs |
| `__init__.py` | Empty package marker |

## How It Connects

Decorated wrappers in `decoration/torch_funcs.py` call this package for every logged
operation. `trace.py` owns the forward session; `output_tensors.py` creates raw
`LayerPassLog` entries consumed by `postprocess/`. `source_tensors.py` creates roots for
inputs and buffers. `backward.py` runs after a forward log when backward capture is
requested.

Fastlog reuses the wrapper hot path but stores `ActivationRecord` data through
`fastlog/_orchestrator.py` instead of building a full `ModelLog`.

## Key Functions

### trace.py
- `run_and_log_inputs_through_model()` - core runner used by `log_forward_pass()`.
- `save_new_activations()` - replay-like activation refresh on an existing graph.
- `_run_model_and_save_specified_activations()` is called from `user_funcs.py` for two-pass
  selective save behavior.

Ordering matters: capture RNG/autocast state, enter `active_logging()`, run model forward,
cleanup model session, then postprocess.

### output_tensors.py
- `log_function_output_tensors()` - dispatches to exhaustive or fast behavior and applies
  live intervention hooks.
- `log_function_output_tensors_exhaustive()` - builds raw per-output `LayerPassLog` entries.
- `log_function_output_tensors_fast()` - validates graph alignment and updates selected tensor data.
- `apply_live_hooks_to_outputs()` - applies normalized intervention hooks during capture.

### tensor_tracking.py
- `_get_operation_equivalence_type()` - structural fingerprint used by loop detection.
- Backward hook helpers link tensors to `GradFnLog` and gradient records.

### backward.py
- `log_backward()` - captures autograd graph and gradients for a logged forward pass.
- `recording_backward()` - context/helper surface exposed from `ModelLog.recording_backward`.

## Fast vs Exhaustive
Exhaustive capture owns metadata truth. Fast capture is allowed only when it can align with
the exhaustive pass by operation counter, function name, and parent sets. Any graph
divergence should fail clearly rather than silently saving mismatched activations.

## Training Semantics
Do not introduce bare `.detach()` or `torch.no_grad()` in capture paths. Tensor copy/detach
behavior is controlled by save options and `train_mode=True`; use `safe_copy()` and existing
storage routing.
