# torchlens/capture architecture and implementation

Roles are functional: the coordinator owns design and integration, implementers own scoped changes, and reviewers verify evidence. The same rules apply to every harness.

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
| `outcome.py` | SINGLE authority for terminal capture truth: `CaptureOutcome`/`CaptureStatus`/`CapturePhase`/`FailureOrigin` plus the N1-N5 capability chokepoint |
| `session.py` | Capture-session lifecycle state |
| `projectors.py` | Projector CLASSES over a sealed capture core — `RefreshProjector` (consumed by `capture/trace.py` refresh runs) and `RecordingProjection`/`RecordingProjector` (consumed by `fastlog/types.py`), not helper callables for `projections.py` |
| `plan.py` | Capture planning helpers |
| `__init__.py` | Empty package marker |

## How It Connects

Decorated wrappers in `backends/torch/wrappers.py` and `backends/torch/ops.py` emit
backend events for every logged operation. `CaptureEvents` (torchlens/ir) is the ONE
logical journal for a run: its append methods are the single writer and stamp every
event of every kind (op, module prep/enter/exit, pre-hook, output-version, buffer-write,
and the whole backward family) with one run-monotonic `seq`, so cross-kind and
forward/backward ordering is an exact recorded fact. Never append to a lane list
directly. Torch op events are trace-backref-free from birth (`source_trace=None`); the
trace owns its stream through instance attributes -- `_capture_events`, plus the
`capture_events` alias during capture until postprocess drops it (the old
`_EVENT_STREAMS` weak side registry is gone). `trace.py` owns the forward session;
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
- `RecordingState` - live predicate-recording session state (`get_active_recording_state()`,
  `active_recording_state()`).
- `_build_record_context()` / `_record_context_from_event()` - predicate-visible
  `RecordContext` construction for live capture and event replay.
- `append_projected_event()` - sparse `OpEvent` emission for the predicate path
  (`_record_from_record_context()` builds the event payload; the old `_event_from_record()`
  was deleted in P7).
- `sync_recording_grad_records_from_sidecar()` - rebuilds fastlog gradient records from the
  unified backward sidecar.

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

## Label Formats

- Source tensors: `{type}_{num}_raw`, for example `input_1_raw` or `buffer_1_raw`
  (source counters are 1-based; `input_0_raw` never exists).
- Function outputs: `{type}_{num}_{counter}_raw`, for example `conv2d_1_5_raw`.
- Labels are raw during capture and become final labels in `postprocess/labeling.py`.
- Pass-qualified final labels use `{label}:{pass_index}` (the multi-PASS ordinal,
  not the call counter).

## arg_positions.py

- Main entry point: `extract_tensors_and_params(spec, args, kwargs)` — the resolved
  `ArgSpec` (3 fields: `positions`, `sequence_positions`, `tensor_kwargs`) comes
  first, not a function name.
- Lookup order: `FUNC_ARG_SPECS` static table -> `_state._dynamic_arg_specs` dynamic cache
  (uncacheable entries marked with the `DYNAMIC_SPEC_UNCACHEABLE` sentinel) -> BFS fallback.
- `ArgSpec` stores exactly the 3 resolved fields above: tensor arg `positions`,
  tensor `sequence_positions`, and `tensor_kwargs` names (no param-index fields exist).
- Keep keyword handling accurate; stale entries can hide graph parents.

## salient_args.py

- Uses `@_register()` per function/layer family.
- `_build_arg_name_map()` maps positional args to names.
- Extractors are failure-safe and return `{}` on unexpected errors.
- Metadata is display-oriented; never let it affect graph correctness.

## flops.py

- Zero-FLOPs ops, elementwise ops, and specialty handlers feed
  `compute_forward_flops()` and `compute_backward_flops()`.
- `register_op_rule()` is the extension point.
- MAC convention is 2 FLOPs.

## projections.py Gotchas

- Sparse `Recording` projections must preserve pass indexes, raw labels, and payload refs.
- Full `Trace` projections must preserve backend-neutral event metadata until postprocess finalizes
  labels and graph structure.

## predicates.py / stop.py Gotchas

- `followed_by` must stay correct-or-loud for unsupported predicate shapes.
- Halt and nonfinite directives must return partials only through the explicit StopDirective policy.
- Validation for backward lives in `validation/backward.py`.

## Known Risks

- Dynamic `arg_positions` cache is process-local and is not automatically invalidated across
  torch version changes.
- Keyword tensor coverage should be checked whenever adding static specs.
- Predicate-backed selective capture assumes deterministic graph shape for validation-sensitive
  paths; random/control-flow drift must fail clearly when alignment is required.
- `torch.func` / functorch wrappers log transform boundary ops and run the inner callable
  under paused logging. Preserve the boundary parent edge and transform metadata.
- Unlabeled tensor args are provenance markers, not graph parents. Inputs, params, buffers,
  and module tensor attributes should remain known sources; foreign captured tensors should warn.

