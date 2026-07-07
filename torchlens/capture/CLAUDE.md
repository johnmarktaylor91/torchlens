# capture/ - Real-Time Operation Logging

## What This Does
Captures tensor operations while a model forward pass runs under `active_logging()`.
It supports exhaustive full-graph capture, selective predicate capture, halted/failed
partial diagnostics, and fastlog's lightweight `RecordContext` construction.

## Files

| File | Purpose |
|------|---------|
| `trace.py` | Forward-pass orchestration, input normalization, session setup/cleanup, halt/failure handling |
| `projections.py` | Conversion from backend events into `Trace`/`Recording` projections |
| `predicates.py` | Capture predicate normalization, composition checks, and `followed_by` support |
| `stop.py` | StopDirective policy objects and halt/nonfinite handling |
| `config.py` | Session configuration dataclasses used by backend capture |
| `arg_positions.py` | 3-tier tensor/parameter extraction: static table, dynamic cache, BFS fallback |
| `salient_args.py` | Human-readable function configuration metadata |
| `flops.py` | Forward and backward FLOPs estimates with registry hooks |
| `__init__.py` | Empty package marker |

## How It Connects

Decorated wrappers in `backends/torch/wrappers.py` and `backends/torch/ops.py` emit
backend events for every logged operation. `trace.py` owns the forward session;
backend producers create raw op/input/buffer records consumed by `postprocess/`.
Backward capture is routed through validation/backward and trace methods rather
than a capture-local `backward.py` module.

`torch.func` / functorch transform entry points are captured as single boundary
ops. The boundary op stores transform metadata, a replay callable, and parent edges
from the transform inputs; the inner transformed function runs with logging paused.
Unattributed tensor-argument markers are collected during arg resolution and warned
once in postprocess.

Fastlog reuses the wrapper hot path but stores `ActivationRecord` data through
`fastlog/_recorder.py`, `storage_ram.py`, and `storage_disk.py` instead of building
a full `Trace`.

## Key Functions

### trace.py
- `run_and_log_inputs_through_model()` - core runner used by `trace()`.
- `save_new_outs()` - replay-like out refresh on an existing graph.
- `_run_model_and_save_specified_outs()` is called from `user_funcs.py` for two-pass
  selective save behavior.

Ordering matters: capture RNG/autocast state, enter `active_logging()`, run model forward,
cleanup model session, then postprocess.

### projections.py
- `recording_from_capture_events()` - builds sparse `Recording` projections.
- `trace_from_capture_events()` - materializes a full `Trace` projection.

### predicates.py / stop.py
- Predicate helpers normalize capture decisions and validate `followed_by` support.
- Stop helpers keep halt/nonfinite behavior explicit and typed.

## Fast vs Exhaustive
Exhaustive capture owns metadata truth. Fast capture is allowed only when it can align with
the exhaustive pass by operation counter, function name, and parent sets. Any graph
divergence should fail clearly rather than silently saving mismatched outs.

## Training Semantics
Do not introduce bare `.detach()` or `torch.no_grad()` in capture paths. Tensor copy/detach
behavior is controlled by save options and `backward_ready=True`; use `safe_copy()` and existing
storage routing.
