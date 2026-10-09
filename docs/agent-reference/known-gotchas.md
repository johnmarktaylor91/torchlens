## Known Gotchas

- Intervention-spec loads tolerate foreign `custom` callable keys for safe analysis without importing
  their modules. Resolving one for execution denies by default because imports execute top-level code;
  trusted execution may use `trust_custom_callables=True`, while
  `allowed_custom_callable_modules={"my_trusted_module"}` is narrower and remains restrictive even if
  the boolean is also true. TorchLens-owned `torchlens.*` custom callables and the fixed trusted
  namespaces resolve without an opt-in.
- `Trace.forward_peak_memory` on CPU/MPS is only the cheap host RSS (or MPS allocator) delta and
  legitimately reads `0` for small models. The `tracemalloc` Python-allocation peak is opt-in via
  `CaptureOptions(measure_python_peak_memory=True)` because the allocator hook costs 1.7x-2.5x
  total capture time. Never assert `forward_peak_memory > 0` on the default path.
- Distributed/sharded state is detected in `torchlens/_distributed.py` and refused at capture entry
  with `DistributedCaptureUnsupportedError`; the same detection feeds the `dtensor` / `device_mesh` /
  `tensor_parallel` / `pipeline_parallel` rows of `tl.compat.report`, so the two cannot drift.
  `dtensor`, active TP hooks/styles, and `pipeline_parallel` refuse; dense parameters do not make a
  `PrepareModuleInput` redistribution safe. Only a bare inert mesh is informational. Detection is
  capability-probed (`HAS_DTENSOR`, `HAS_DEVICE_MESH`, `HAS_PIPELINING`), never version-parsed, and
  bounded to inspectable instance state (12 levels / 4096 objects). Slots/descriptor-only holders,
  opaque user-wrapped TP hooks, over-bound state, and tensors created inside `forward` remain
  disclosed residuals. The `dtensor` finding identifies refused state precisely via per-site
  dual geometry on `finding.geometry`.
- Explicit `torch.distributed` python collectives in a traced forward become first-class boundary
  nodes under the distributed opt-in (`tl.distributed.arm()` at process start, REQUIRED for
  MPMD/spawn ranks; lazy arming covers initialized SPMD first-captures). The portable payload is
  `op.annotations["collective"]` (`collective_boundary_v1`: correlation key, role entries, event
  and witness disclosures) plus the trace-level group-lifecycle ledger in
  `trace.annotations["distributed"]`. `CaptureOptions(distributed_witness="digest")` opts into
  byte-exact contribution/destination digests. Async completions record
  `completion_binding="unobserved"`; wildcard recv refuses typed; collective-crossing traces
  refuse runnable save and forward-replay validation
  (`collective_boundary_runnable_unsupported`) while metadata invariants run in full. Arming
  relaxes no tier-(a) refusal (DTensor/TP/FSDP2/PP still refuse). Distributed rank processes
  (initialized process group, non-daemonic) are the sanctioned exception to the
  main-process-only capture guard.
- `tl.merge_ranks([trace_or_path, ...])` merges N rank cores into a `MergedTrace` presenter at
  their explicit collective boundaries (rung C1); `tl.merge_report(...)` diagnoses without
  constructing. The derivation is audit-first (a conflicted membership never joins and never
  becomes a presence gap), aligns seq counters as deltas from each rank's first recorded key,
  and treats witness digests as demote-only evidence. Artifacts save as `merged-directory`
  bundles whose descriptor is a cache: loads rederive from the rank cores and refuse typed on
  any inequality. Frozen enums + error/finding codes: `torchlens.merged` +
  `docs/reference/merged_trace_contract.md` (ordered-equality gated). p2p/pipeline (C3) and
  DTensor topologies (C2) refuse typed; merged replay does not exist.
- `CaptureOptions(save_budget=...)` is a per-device ceiling on retained activation bytes, default
  `"auto"` = half of measurable available memory. Exhaustive, predicate, and deferred
  `Op.save_activation()` paths pre-admit the primary source-sized RAM copy before allocation, then
  reconcile alias-aware physical storage. Transform
  deltas and cross-device temporaries cannot always be known pre-allocation, so this is not a general
  OOM guarantee. Predicate disk-only saves are exempt; exhaustive
  `capture=tl.options.CaptureOptions(layers_to_save="all")` plus `to_disk(...)`
  stays budgeted until postprocess eviction (the former bare flat `layers_to_save=` kwarg is
  removed). Unmeasurable auto devices warn on first charge.
- Streamed disk writes are ASYNC BY DEFAULT for `trace(storage=tl.to_disk(...))` (spellings
  DOCUMENTED-UNSTABLE): one FIFO worker overlaps blob serialize+write+sha256 with the forward,
  payloads snapshot at submission, `to_disk(max_pending_bytes=)` (256 MiB default) blocks capture
  when the disk falls behind, a failed write raises typed `TorchLensIOError` + PARTIAL, finalize
  drains before publish, and bundles are byte-identical to sync. `async_writes=False` opts out;
  `tl.record` streaming stays synchronous and refuses an explicit `True`.
- On torch >= 2.6 (`HAS_SET_STANCE`), every capture holds
  `torch.compiler.set_stance("force_eager")` scoped to the forward (entered inside
  `prepare_compiled_capture`; skipped when `torch._dynamo` was never imported), so compiled plain
  attributes and free functions run their ORIGINAL eager Python: interiors are fully logged with
  ordinary verified semantics, zero compiles happen during capture, compiled caches survive with at
  most ONE bounded recompile on the next compiled call afterward (wrapper install/uninstall guard
  invalidation), and the plain-attribute pause-logging bypass is NOT installed. The paragraph below
  is the torch < 2.6 / no-stance fallback, pinned byte-for-byte by the `_no_stance` tamper tests in
  `test_dynamo_fake_guard.py`.
- A Dynamo-traced region reached during capture is bypassed in the wrapper (see
  `_is_inside_dynamo_compilation`), warning once per forward and setting
  `trace._raw_transform_escape_detected` (which licenses the unattributable-output tolerance,
  exactly like the functorch guard beside it) plus `trace._raw_dynamo_region_detected`, which is
  the top-precedence `capture_verification_reason` -- `"dynamo_region_not_logged"` -- at BOTH
  verdict sites (`completeness_witness._finalize_census` and the `escape_detection` capture-scope
  `finally`, the last writer). Without the dedicated flag the Trace blamed
  `owner_thread_tripwire_changed`, since compiling spawns threads. Plain compiled-callable
  attributes are also inventoried before forward and invoked with logging paused, conservatively
  arming the same flags on cold/warm runs where `is_compiling()` may never fire; global/free hot
  callables remain disclosed. Compiled child `nn.Module`s are still unwrapped to eager BEFORE
  capture, so their interiors stay logged -- there
  is a test asserting the bypass did not regress that into a silent gap. Fake/functional tensors on
  inputs or params refuse at capture entry in `_robustness.py`; never let one reach the metadata
  path, where `data_ptr()` on a FakeTensor is a torch-flagged bug.
- `__wrapped__` is removed from built-in function wrappers to avoid `inspect.unwrap`
  failures.
- `backends/torch/ops.py` rebinds the split `_ops_*` functions onto its own globals
  (`_rebind_function`), so any new module-level name used inside a rebound function must also be
  bound in `ops.py` (and a test patches such a name on `ops`, not on the split module).
- Fast-path module decoration skips `_record_module_entry_metadata`; alignment state must be
  replicated manually.
- `get_memory_amount()` deliberately avoids `pause_logging()`: it resolves the
  UNWRAPPED `nelement()`/`element_size()` methods without toggling global logging
  state per tensor (hot-path perf commit `08dca260`); re-adding the toggle is a
  regression, not a fix.
- If a `@property` raises `AttributeError`, Python falls through to `__getattr__`; use
  `ValueError` for TorchLens multi-pass access errors.
- `copy()` on `Op` deep-copies graph metadata and shares tensor payloads/callables; see data_classes/AGENTS.md.
- An in-place write to a prepared Parameter inside `forward` is a logged op (receiver positional
  or `input=`); a frozen Parameter written with a grad-requiring operand runs tracked with its own
  `requires_grad=False` and ends a non-leaf, as in eager, so `restore_param_requires_grad` and the
  prep-time forcing skip non-leaf Parameters. Validate a fresh copy, not an already-run model: a
  deepcopy of the run model makes that Parameter a trainable leaf (torch's `Parameter.__deepcopy__`),
  so `tl.validate(..., scope="forward")` raises torch's leaf error, as a second eager forward on
  such a deepcopy does, and `scope="backward"` returns False with a "not autograd leaves" warning.
  Backward validation cuts the history its own stock pass leaves on such a Parameter (`detach_`,
  flag restored) so the captured pass starts from the pre-call model. The mutation op's gradient
  hook sits on that pass's `grad_fn` in the Parameter's history, which later forwards chain onto, so
  that node is a root-matching boundary (a later backward does not open the old trace's bracket
  through it), every hook of that trace records only inside its own managed backward (no implicit
  passes), and `cleanup()` removes the hook. Backward validation compares
  Parameter and module-output grads, not per-op grads; the mutation op's gradient is pinned by tests.
  An `out=` write into a Parameter is still uncaptured:
  `tl.validate` fails it on completeness (pinned in `tests/test_parameter_inplace_mutation.py`).
- `torchlens.__version__` and `pyproject.toml` are release-pipeline state; do not update them
  in feature/docs tasks unless release work explicitly asks for it.
- Legacy constructors (`torch.FloatTensor(...)` and siblings, `Variable(...)`) are captured by
  patching the class's `__new__` in place (`backends/torch/legacy_ctors.py`). While wrapped, a
  warning their C constructor raises (the `volatile=` removal, the `torch.cuda.*Tensor`
  deprecation) is attributed to `utils/_type_new_slot.py`, not the caller, so per-location
  `warnings` filters and `-W error` tracebacks differ from eager; message and category are
  unchanged. A class on which another tool's Python `__new__` is in effect at wrap time (read from
  the MRO dicts, never the slot pointer) is left uncaptured with a `TorchLensWarning` (code
  `legacy_constructor_uncaptured`; `skipped_legacy_constructor_classes()`); once that patch is
  removed, the next wrap re-patches the class from the recorded C constructor.
- A trace's staged intervention spec (`Trace._intervention_spec`) is read-only to every engine:
  reruns, append and chunked reruns, failed or interrupted reruns, and `fork()` leave it exactly
  as staged (`tests/test_state_hygiene_oracles.py`, `tests/test_rerun_hook_staging.py`). Validation
  cannot see a violation (each rerun is self-consistent with the plan it ran), so the oracles
  compare reruns against an independent expectation. A capture-time `intervene=` predicate that is
  not lowered to module hooks stages per-op entries on FINAL labels, which the live matcher refuses;
  reruns re-arm the retained predicate (`_predicate_save_options.intervene`) through the capture
  door and refuse `rerun_predicate_restage_mismatch` if it re-stages a different op set. The spec is
  `FieldPolicy.DROP` (the recipe travels through `save_intervention`), so every legacy rerun door
  of a loaded intervened trace refuses `run_intervention_spec_not_persisted`, including after a
  new edit is staged on it. The rerun divergence hash folds value-only edit nodes
  (`interventionreplacement` with unchanged shape and dtype) into their parent, so a correct
  staged rerun is silent; never filter `ControlFlowDivergenceWarning` in a test of a correct graph.
  Tensor-carrying helpers alias the caller's tensor; staged entries (`HookSpec.metadata`
  `helper_tensor_digests`) fingerprint it, so a loop that updates a steer in place must re-stage
  each step or the next rerun or recipe save refuses `helper_tensor_changed_since_capture`. A
  bound executor is live instead: each call reads the tensor, an in-place change applies to the
  next call, and `BindReport.helper_tensor_versions` shows the version counter moving.
