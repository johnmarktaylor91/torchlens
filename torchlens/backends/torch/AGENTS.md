# backends/torch/ - Agent Notes

## File Map

Core capture path:
- `wrappers.py` — lazy torch function wrapping for capture-time interception
  (`wrap_torch()` / `unwrap_torch()`, DeviceContext handling).
- `model_prep.py` — prepare `nn.Module` objects for capture sessions (permanent +
  per-session). Release-time held torch-function-ref normalization moved OUT to
  `_held_refs.py` (`normalize_held_torch_function_refs`, called by
  `release_model`); quantized-module FLOPs estimation lives in `_quantized_flops.py`.
- `ops.py` — log tensors produced by decorated torch operations; its implementation is
  split across the `_ops_*.py` family: `_ops_exhaustive.py` (exhaustive emission),
  `_ops_emission.py` (output logging / live-hook dispatch), `_ops_arguments.py`
  (argument templates and provenance), `_ops_activations.py` (activation persistence),
  `_ops_retention.py` (save budgets, lookback), `_ops_capture_records.py` (record
  freezing), `_ops_containers.py` / `_ops_container_base.py` (output containers),
  `_ops_predicates.py` / `_ops_predicate_events.py` / `_ops_finalize.py` (predicate
  path), `_ops_interventions.py` (live interventions), `_ops_autograd.py` (saved-tensor
  stats), `_ops_shared_fields.py` (shared fields).
- `sources.py` — log source tensors; `tensor_tracking.py` — provenance, family links,
  equivalence classes; `module_stack.py` — module-call stack;
  `prehook_provenance.py` — reversible forward-pre-hook input provenance;
  `buffer_writes.py` — registered-buffer write capture; `backward.py` — backward
  execution and autograd graph metadata.

Safety net and honesty:
- `belt.py` — the mechanical BELT: protocol-invisible stale-reference coverage
  (derived per build).
- `rescue.py` — the disclosed mode-rescue re-run recovering escaped ops with a
  `TorchFunctionMode` net.
- `escape_detection.py` — opt-in shadow detection for stale detached callable
  invocations; `identity_shims.py` — keep torch-internal identity checks truthful;
  `aliasing.py` — alias/mutation contract detection.
- `completeness_witness.py` — opt-in aten-dispatch completeness witness, with its
  implementation in the `_completeness_*.py` family (boundaries, cross-thread,
  dispatch census + names, escape state, finalize, metadata, origins, patches,
  storage, shared types).

Backend integration:
- `backend.py` — the torch `CaptureBackend` Protocol implementation;
  `collectives.py` — explicit `torch.distributed` collective boundary capture;
  `_tl.py` — private metadata namespace helpers.

## Wrapper Boundaries
- `wrappers.py` owns persistent torch/function decoration. `_logging_enabled` must stay
  the runtime gate; wrappers remain installed after first capture.
- `torch.func` / functorch transform builders return boundary callables. They should attach
  transform metadata, lazy source locations, and replay callables without tracing inside the
  transformed function.
- Direct-call transform wrappers such as `torch.autograd.functional.jacobian` follow the same
  boundary-node contract.

## Provenance
- `model_prep.py` tags registered buffers and plain module tensor attributes with buffer
  addresses before capture.
- `ops.py` records `unattributed_tensor_args` only for tensor arguments with no TorchLens
  input/op/buffer provenance. This is warn-first diagnostics, not a parent-edge substitute.
- Foreign tensors in output position keep existing output binding behavior and should not
  create provenance warnings.

## Journal Producer (single since producer unification P7)
- Every torch capture freezes decomposed `OpRecord` rows (`OpCore` + typed facets) through
  the ONE commit tail `capture/projections.py::commit_op` (freeze -> atomic append): the
  three exhaustive `_make_layer_log_entry` sites via `ExhaustiveOpDraft` and the sparse
  `append_projected_event` sites via `SparseOpDraft`. The legacy `OpEvent` producer, its
  dual-path env switch, `_op_event_from_log`, and
  `_event_from_record` were deleted in P7.
- Post-commit knowledge never mutates the op lane: it rides the typed `OpAmendment` lane
  (`CaptureEvents.append_amendment`, nine exact-set families) and folds through the ONE
  reducer `amended_op_records()`. grad-fn handles live ONLY in the journal side index
  (`grad_fn_handles_by_label_raw`); records never carry them. The one sanctioned in-place
  carve-out — the fastlog ancestry backfill on a cook-owned `copy_for_replay` projection —
  is pinned (with every other op-lane mutator) by the closed ledger in
  `tests/producer_parity/test_op_lane_inplace_writers.py`.
- Preview backends keep emitting compat `OpEvent`s until S15 and adapt at the one ingest
  boundary (`op_record_from_event`); `OpEvent`, `PATH_TO_FLAT` (amendment fold guard),
  and `_clone_op_event_for_replay` are retained-with-schedule (S15) and guard-tested in
  `tests/producer_parity/test_p7_single_producer.py`.
