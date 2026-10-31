## Known Gotchas

- Intervention-spec loads tolerate foreign `custom` callable keys for safe analysis without importing
  their modules. Resolving one for execution denies by default because imports execute top-level code;
  trusted execution may use `trust_custom_callables=True`, while
  `allowed_custom_callable_modules={"my_trusted_module"}` is narrower and remains restrictive even if
  the boolean is also true. TorchLens-owned `torchlens.*` custom callables and the fixed trusted
  namespaces resolve without an opt-in.
- CPU peaks are opt-in: a default capture reads CPU `Trace.forward_peak_memory` and
  `backward_peak_memory` as `None`, with `forward_peak_memory_reason` / `backward_peak_memory_reason`
  naming `CaptureOptions(measure_host_peak_memory=True)`, and never writes the process high-water
  mark. Under the opt-in the CPU forward peak is the host resident high-water growth (`VmHWM` after
  a per-capture reset minus the pre-forward RSS; `/proc/self/statm` when psutil is absent), so a
  buffer freed inside the forward counts; that reset (forward bracket, and each CPU backward pass)
  is process-wide: it also resets the caller's own `VmHWM` and `ru_maxrss` readings, and cannot be
  undone. An opted-in CPU forward that grew nothing (it fit in resident headroom) is recorded as
  `None` with a reason naming `CaptureOptions(measure_python_peak_memory=True)`. MPS is the
  allocator delta and can read `0`. The `tracemalloc` Python-allocation peak is a second opt-in
  because the allocator hook costs 1.7x-2.5x total capture time. Never assert `forward_peak_memory
  > 0` (or not None) on the default path; a test that needs a CPU number opts in.
  `backward_peak_memory` is the largest pass peak on CPU, not a sum.
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
  any inequality. Frozen enums + error/finding codes: `torchlens._merged` +
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
  ordinary verified semantics, zero compiles happen during capture, compiled caches survive with a
  bounded burst of recompilations on the first compiled call after each capture, then
  steady state (wrapper install/uninstall guard invalidation and graph breaks around TorchLens's own
  wrapper; measured by `tests/real_model/test_track_compile_recompile_burst.py`: zero on
  `resnet18`, six to seven frames on a tiny GPT-2, never above Dynamo's recompile limit), and the
  plain-attribute pause-logging bypass is NOT installed. The paragraph below is the torch < 2.6 /
  no-stance fallback, pinned byte-for-byte by the `_no_stance` tamper tests in
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
- A library imported while wrappers are installed keys its `__torch_function__` handler table
  on the WRAPPER (`TD_HANDLED_FUNCTIONS[torch.stack]` in tensordict), but torch's protocol hands
  handlers the ORIGINAL, so every handler declines. When torch reports that every handler declined
  this very function, the wrapper re-offers the same handlers the same arguments once under its own
  identity (`backends/torch/_protocol_redispatch.py`, gated on `HAS_DISABLED_TORCH_FUNCTION_IMPL`);
  a genuine decline still raises torch's own `TypeError`. Unwrapped torch has no wrapper to re-offer
  (validation's pristine ground truth unwraps), so every `unwrap_torch()` first adds the original
  key beside each wrapper key in module- and class-level `dict`/`set` tables of modules imported
  after the redispatch module (torch and TorchLens excluded), and the next wrap removes exactly
  those keys. Every call of the original and every re-raise stays in the wrapper's own frame: the
  escape detector and failure-origin classification read that frame, so never route `func` through
  a helper. Tests: `tests/test_protocol_redispatch.py`.
- Buffer provenance never rests on values: step 6 merges two buffer nodes only when
  capture recorded the same buffer version for both (tensor object, storage and version counter,
  `RawGraphWorkspace.buffer_version_keys`) and the later one is not a journaled write, and a buffer
  version replays as `identity` of its producer only on exact NaN-equal bytes (`tensor_nanequal`).
  Comparing buffer values with IEEE `torch.equal` made a model whose buffers reach NaN from one entry
  state record a different graph, so validation's re-trace called it non-reproducible.
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
- `copy.deepcopy(t)` inside a forward is one op on `t`, recorded as `copy.deepcopy(self)` (the memo
  is not logged: the call fills it with the output). `Tensor.__deepcopy__` copies `t.__dict__`, so
  `TensorMeta.__deepcopy__` drops the session anchor, storage pin and alias flags; without that the
  copy read as `t` itself and its consumers were wired to `t`'s node.
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
- "Pristine" has three senses; never read one as another. `Trace.pristine` (documented-unstable)
  is the derived reading "nothing ever intervened on this trace", computed from the evidence the
  trace carries (`torchlens/_data_classes/_trace_pristine.py`, one function), so it survives save
  and load and stays `False` after `clear_hooks()`, a `Trace.state` reassignment or a fork; `None`
  means no evidence (an artifact older than tlspec 9, or a non-torch backend) and must never be
  treated as `True`. `TraceState.PRISTINE` is only the operational state machine's start value: an
  assignment sets it and a func-targeted `intervene=` capture keeps it. Validation's
  pristine-torch oracle (`torchlens/validation/_pristine.py`) means unwrapped torch functions.
  Bundle's pristine label and baseline auto-detection read `Trace.pristine is True`.
- A capture-time module edit (`intervene=` with a `tl.module(...)` selector) is its own
  `interventionreplacement` op between the module's last op and its consumers. The last op keeps
  its pre-edit value by design (`tl.module(...)` reads the replacement op), and every later op is
  renumbered, so a cross-trace join by literal label drops exactly the ops the edit changed. Bundle
  site reads pair through the supergraph alignment, which folds a replacement op into its parent
  (`torchlens/intervention/_topology/topology.py`, `_structural_parent_labels`;
  `tests/test_bundle_module_edit_alignment.py`).
- Labels inside a binding: `spec.bind(model)` runs the model without a capture, so its ops carry
  only the raw labels minted during the forward (`relu_1_2_raw`). A final, short or pass-qualified
  label (`tl.label("relu_1_2")`, `tl.label("relu_1")`, `"conv2d_2_3:1"`, the same literals in
  `tl.contains` or `tl.regex`) and `tl.followed_by`, `tl.preceded_by` or `tl.input_at` can never
  match there, negated or combined, so `.bind(...)` refuses the rule with `BindingPreflightError`
  code `bind_rule_unsupported` before any forward runs. Select the op structurally instead:
  `tl.func("relu") & tl.in_module("1")` (the label's op type, plus a module address from
  `model.named_modules()`). Labels keep working on a capture (`tl.trace(...)`, `trace.resolve_sites`).
  A `tl.where` predicate that reads post-capture fields (`p.layer_label`) is opaque, so it still
  fails at the first op with `bind_rule_runtime_error`.
- TorchScript and DLPack: a `torch.jit.script` function called from an eager forward runs its
  body in the TorchScript interpreter, below the Python functions TorchLens wraps, so each call
  is recorded as ONE `jitscript` boundary op (`backends/torch/_script_boundary.py`, hook on
  `ScriptFunction.__call__`, gated by `HAS_SCRIPT_FUNCTION_CALL_HOOK`; warning
  `torchscript_boundary`): its interior ops are not in the graph, every dispatch inside it is
  owned by the audited `jit_script:boundary` token, and validation replays it by calling the
  function again (a scripted module is still refused up front). A call that writes into one of
  its tensor arguments in place is NOT recorded: the witness counts its dispatches as
  unaccounted, the capture is `capture_verified=False`, and the advisory
  `scripted_function_not_captured` names the function and the call line. To see the interior,
  set `PYTORCH_JIT=0` in the environment before the model's code is imported
  (`PYTORCH_JIT=0 python your_script.py`) and `torch.jit.script` returns the plain function, so
  every operation inside it is captured. A tensor handed to another library through `to_dlpack`
  runs that library's kernels outside PyTorch's Python layer; `dlpack_call_not_captured` names the
  call line, and no TorchLens setting captures those kernels. Both fire once per session and move
  no verdict; the advisory resolves only names visible in the defining module (module globals, or
  `self.`/`cls.` attributes of its classes), not a scripted function stored on an instance.
- Replay reads live parameters: a replay template holds the `nn.Parameter` itself, so every replay
  door refuses `replay_param_changed_since_capture` once a consumed parameter changed since capture
  (version counter, storage pointer or shape); `allow_changed_params=True` on `InterventionOptions`
  or `ReplayOptions` replays with the current values and records the names in
  `trace.last_run["replayed_with_changed_params"]`. Not detected, so replay silently uses the new
  value: an in-place write through `param.data` (its own version counter on the same storage) or
  through a numpy or raw-storage alias. A replaced parameter object is never read by replay (the
  template keeps the captured one). Pinned in `tests/test_replay_param_changed.py`.
- A model that builds a thread pool lazily in `forward` loses one pool per capture to the capture's
  attribute restore; a later garbage collection that frees it inside another capture ends its idle
  workers there, and the thread check then marks that capture unverified. This errs only one way: a
  false "unverified", never a false "verified".
