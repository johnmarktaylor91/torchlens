## Known Gotchas

- Frame filters identify TorchLens's own frames through `torchlens._utils._package_frames.is_torchlens_code`
  (a memoised normalised-filename comparison), never a raw `co_filename` prefix test. The import
  system spells a filename the way it joined it: a `sys.path` entry of `.` (for example
  `PYTHONPATH=.`) leaves `/./` in every TorchLens filename, and a raw prefix test then kept hundreds
  to thousands of TorchLens frames in the ops' code context. Captured metadata must not depend on how
  the host process imported TorchLens; `tests/test_grouping_host_independence.py` pins it.
- Intervention-spec loads tolerate foreign `custom` callable keys for safe analysis without importing
  their modules. Resolving one for execution denies by default because imports execute top-level code;
  trusted execution may use `trust_custom_callables=True`, while
  `allowed_custom_callable_modules={"my_trusted_module"}` is narrower and remains restrictive even if
  the boolean is also true. TorchLens-owned `torchlens.*` custom callables and the fixed trusted
  namespaces resolve without an opt-in.
- CPU peaks are opt-in: a default capture reads CPU `Trace.cost.forward_peak_memory` and
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
- ATen rows are opt-in: a default capture installs no dispatch mode, `trace.has_aten_ops` is
  `False`, and `trace.aten_ops` / `op.aten_ops` / `trace.aten_gaps` raise
  `AtenOpsUnavailableError` (`aten_ops_unavailable`) rather than returning an empty tuple. Pass
  `capture=tl.options.CaptureOptions(aten_ops=True)`; the switch is a session knob (not persisted,
  inherited by `run()` refreshes, fast or not) while the recorded rows persist.
- ATen values need `aten_ops="values"`, not `True`: with `True` every `row.out` is `None`. Values are
  snapshots copied below autograd, so they never carry `grad_fn` even with
  `detach_saved_activations=False`; they follow the owning call's `save=` decision (a dropped op's
  rows read `None`), count against `save_budget`, and are not retained when activations stream to
  disk or an out sink is active. Values are compared only through `aten_values_consistent` under
  `save_mode="copy"`; a `"reference"` capture still keeps them but skips the owner comparison.
  Both recorder hooks run inside the owning wrapper call under `pause_logging()`
  (`_recorder_paused`): a wrapped tensor method (`stride`, `untyped_storage`, `numel`, the clone)
  reached from the dispatch hook with logging on resets the wrapper's bottom-level barcode (an
  in-place owner's op silently dropped) and counts in the ordered call fingerprint (a rows capture
  then refused the fast tier). `test_values_leave_the_op_graph_unchanged` pins the op list, the
  raw-event hash and the call fingerprint equal across `aten_ops` spellings. A `run()` refresh
  re-records the profile; `run(..., fast=True)` refuses a values capture.
- Distributed/sharded state is detected in `torchlens/_distributed/` and refused at capture entry
  with `DistributedCaptureUnsupportedError`; the same detection feeds the `dtensor` / `device_mesh` /
  `tensor_parallel` / `pipeline_parallel` rows of `tl.report.doctor`, so the two cannot drift.
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
  (`collective_boundary_unsupported`) while metadata invariants run in full. Arming
  relaxes no tier-(a) refusal (DTensor/TP/FSDP2/PP still refuse). Distributed rank processes
  (initialized process group, non-daemonic) are the sanctioned exception to the
  main-process-only capture guard.
- `tl.merge_ranks([trace_or_path, ...])` merges N rank cores into a `MergedTrace` presenter at
  their explicit collective boundaries; `tl.merge_report(...)` diagnoses without
  constructing. The derivation is audit-first (a conflicted membership never joins and never
  becomes a presence gap), aligns seq counters as deltas from each rank's first recorded key,
  and treats witness digests as demote-only evidence. Artifacts save as `merged-directory`
  bundles whose descriptor is a cache: loads rederive from the rank cores and refuse typed on
  any inequality. Frozen enums + error/finding codes: `torchlens._merged` +
  `docs/reference/merged_trace_contract.md` (ordered-equality gated). p2p/pipeline and
  DTensor topologies refuse typed; merged replay does not exist.
- `CaptureOptions(save_budget=...)` is a per-device ceiling on retained activation bytes, default
  `"auto"` = half of measurable available memory. Exhaustive, predicate, and deferred
  `Op._save_activation()` paths pre-admit the primary source-sized RAM copy before allocation, then
  reconcile alias-aware physical storage. Transform
  deltas and cross-device temporaries cannot always be known pre-allocation, so this is not a general
  OOM guarantee. Predicate disk-only saves are exempt; exhaustive
  `save="all"` plus `to_disk(...)`
  stays budgeted until postprocess eviction (the former bare flat layer-selection keyword is
  removed). Unmeasurable auto devices warn on first charge.
- Streamed disk writes are ASYNC BY DEFAULT for `trace(storage=tl.to_disk(...))` (spellings
  DOCUMENTED-UNSTABLE): one FIFO worker overlaps blob serialize+write+sha256 with the forward,
  payloads snapshot at submission, `to_disk(max_pending_bytes=)` (256 MiB default) blocks capture
  when the disk falls behind, a failed write raises typed `ArtifactError` + PARTIAL, finalize
  drains before publish, and bundles are byte-identical to sync. `async_writes=False` opts out;
  `tl.record` streaming stays synchronous and refuses an explicit `True`.
- A failed or killed save leaves `<name>.tmp.<hex>` beside the path (`PARTIAL` plus `REASON.txt`
  after a failed overwrite, unmarked after a kill). `tl.load` then refuses `newer_save_unfinished`
  while that folder is at least as new as the copy on disk (directory mtimes; a rename keeps the
  mtime), because the copy on disk is the older one. `tl.io.cleanup_tmp(path)` accepts the older
  copy (`force=True` for an unmarked folder). Every saver (`Trace.save`, `Bundle.save`,
  `save_intervention`, `Recipe.save`, merged, `to_disk`) stages under that name; tests that
  fail an overwrite and then load must clean up first.
- A streamed (`to_disk`) trace does not run: every `trace.run(...)` door refuses
  `trace_option_unsupported`, because a run would re-capture every payload into RAM while the
  bundle on disk kept the first capture. Capture again with `to_disk` to a new path, or without
  `to_disk`, and run that trace; a trace loaded from the bundle runs through its loaded providers.
  `Trace.save` and `Bundle.save` to an existing path refuse `bundle_path_exists` (typed, no builtin
  `FileExistsError`) unless `overwrite=True`.
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
  verdict sites (`_completeness_finalize._finalize_census` and the `escape_detection` capture-scope
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
- **A peer that re-wraps a torch slot after the first capture is rebased at unwrap, never
  reported as long as the rebase holds.** `nnsight` patches `torch.Tensor.backward` at import
  with `functools.wraps` around whatever is installed; imported after a capture, that is the
  torchlens wrapper, held in `__wrapped__` and in a module global, and `wraps` copies our
  `__dict__` markers onto the foreign function. `unwrap_torch()` leaves the foreign wrapper in
  place, points every such reference at the original (`backends/torch/_foreign_wrappers.py`) and
  strips the copied markers; re-wrap adopts the foreign callable as the slot's new original, so
  later captures still log the call. The burial `TorchLensWarning` names only a reference the
  walk cannot rewrite (a read-only `__wrapped__`), and every teardown report comes AFTER the
  decorated flag flips: raised under warnings-as-errors before the flip, it once left torch
  restored under a flag that said wrapped, and every later capture in the process logged only
  module boundaries. Decide "is this ours" by ledger identity (`id(fn) in
  _state._decorated_to_orig`), never by a `__tl_*` attribute. Tests:
  `tests/test_wrap_state_compat.py` (TestUnwrapSafety), `tests/test_docs_snippets.py` (peer parity).
- Buffer provenance never rests on values: step 6 merges two buffer nodes only when
  capture recorded the same buffer version for both (tensor object, storage and version counter,
  `torchlens._ir.workspaces.RawGraphWorkspace.buffer_version_keys`) and the later one is not a journaled write, and a buffer
  version replays as `identity` of its producer only on exact NaN-equal bytes (`tensor_nanequal`).
  Comparing buffer values with IEEE `torch.equal` made a model whose buffers reach NaN from one entry
  state record a different graph, so validation's re-trace called it non-reproducible.
- `__wrapped__` is removed from built-in function wrappers to avoid `inspect.unwrap`
  failures.
- Split modules import one another and their hub's constants at their bottoms (they form import
  cycles, so the imports run once every definition exists): `backends/torch/_ops_*` with `ops.py`,
  `backends/torch/_completeness_*` with `completeness_witness.py`, `validation/_invariants_*` with
  `invariants.py`, and `_runnable_*` with `_runnable/execution.py`. A new module-level name used in a
  split module is imported there. A test patches a helper on the module that defines it and on every
  module that binds it by import, never on the hub alone: a from-import copy does not follow a patch.
  Module state reassigned after import (`completeness_witness._ACTIVE_WITNESS_STATE`) lives in one
  module, and every other reader and writer uses `owner_module.NAME`, never a from-import copy.
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
  or passed as the torch `input` keyword); a frozen Parameter written with a grad-requiring operand runs tracked with its own
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
  as staged (`tests/test_state_hygiene_oracles.py`, `tests/test_helper_tensor_fingerprint.py`). Validation
  cannot see a violation (each rerun is self-consistent with the plan it ran), so the oracles
  compare reruns against an independent expectation. A capture-time `intervene=` predicate that is
  not lowered to module hooks stages per-op entries on FINAL labels, which the live matcher refuses;
  reruns re-arm the retained predicate (`_predicate_save_options.intervene`) through the capture
  door and refuse `rerun_predicate_restage_mismatch` if it re-stages a different op set. The spec is
  saved as the typed recipe `intervention.json` (`torchlens/_io/format3/recipe.py`), with each
  custom callable replaced by a callable placeholder; every legacy rerun door of a loaded
  intervened trace refuses `run_intervention_spec_not_persisted` only when its recipe holds a
  placeholder (or the artifact carries no recipe), including after a new edit is staged on it
  (`intervention/_intervened_rerun_guard.py` keys on `metadata["recipe_placeholders"]`). The rerun divergence hash folds value-only edit nodes
  (`interventionreplacement` with unchanged shape and dtype) into their parent, so a correct
  staged rerun is silent; never filter `ControlFlowDivergenceWarning` in a test of a correct graph.
  Tensor-carrying helpers alias the caller's tensor; staged entries (`HookSpec.metadata`
  `helper_tensor_digests`) fingerprint it, so a loop that updates a steer in place must re-stage
  each step or the next rerun or recipe save refuses `helper_tensor_changed_since_capture`. A
  bound executor is live instead: each call reads the tensor, an in-place change applies to the
  next call, and `BindReport.helper_tensor_versions` shows the version counter moving.
- "Pristine" has three senses; never read one as another. `Trace.is_pristine` (documented-unstable)
  is the derived reading "nothing ever intervened on this trace", computed from the evidence the
  trace carries (`torchlens/_data_classes/_trace_pristine.py`, one function), so it survives save
  and load and stays `False` after `clear_edits(...)`, a `Trace.state` reassignment or a fork; `None`
  means no evidence (an artifact older than tlspec 9, or a non-torch backend) and must never be
  treated as `True`. `TraceState.PRISTINE` is only the operational state machine's start value: an
  assignment sets it and a func-targeted `intervene=` capture keeps it. Validation's
  pristine-torch oracle (`torchlens/validation/_pristine.py`) means unwrapped torch functions.
  Bundle's pristine label and baseline auto-detection read `Trace.is_pristine is True`.
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
  `tl.label_contains` or `tl.label_matches`) and `tl.followed_by`, `tl.preceded_by` or `tl.model_input` can never
  match there, negated or combined, so `.bind(...)` refuses the rule with `BindingPreflightError`
  code `bind_rule_unsupported` before any forward runs. Select the op structurally instead:
  `tl.func("relu") & tl.in_module("1")` (the label's op type, plus a module address from
  `model.named_modules()`). Labels keep working on a capture (`tl.trace(...)`, `trace[selector]`).
  A `tl.where` predicate that reads post-capture fields (`p.layer_label`) is opaque, so it still
  fails at the first op with `bind_rule_runtime_error`.
- TorchScript and DLPack: a `torch.jit.script` function called from an eager forward runs its
  body in the TorchScript interpreter, below the Python functions TorchLens wraps, so each call
  is recorded as ONE `jitscript` boundary op (`backends/torch/_script_boundary.py`, hook on
  `ScriptFunction.__call__`, gated by `HAS_SCRIPT_FUNCTION_CALL_HOOK`; the sibling
  `Function.apply` hook wraps both slot shapes, the Python classmethod through torch 2.13 and
  the C descriptor from torch 2.14, gated by `HAS_FUNCTION_APPLY_SLOT`; warning
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
- Replay at source records and random sources: a model-input leaf and a buffer record have no
  callable, so a hook aimed at one fires on the captured (or threaded) value and the edit
  propagates from there; a random-source op (`randn_like`, `dropout`, ...) with no recorded RNG
  state (`save_rng_states` off) keeps its captured draw when none of its arguments changed in the
  replay, so an edit at it edits the value the trace holds, not a fresh draw (a random site that
  was itself edited at capture holds a post-edit value, so its replay re-executes the draw, seeded
  only when RNG states were saved). A trace loaded from a
  saved file carries no callables at any level, so every replay edit on it refuses
  `ReplayPreconditionError` naming that condition (remedy: re-capture in-process with
  `tl.trace(model, inputs)`, or `trace.run(model, inputs)`). An in-process capture replays
  without any capture option: every eager capture records the per-op argument templates
  replay needs (`trace.options.save_arg_templates`); a structure-only capture records none. A torch failure inside a replayed
  callable (an edit that drives an index lookup past its table) is re-raised as
  `ReplayPreconditionError` naming the op that raised and the edit origins, the torch error
  chained as `__cause__`, with nothing committed. Pinned in
  `tests/test_replay_source_and_stochastic_origins.py`.
- A model that builds a thread pool lazily in `forward` loses one pool per capture to the capture's
  attribute restore; a later garbage collection that frees it inside another capture ends its idle
  workers there, and the thread check then marks that capture unverified. This errs only one way: a
  false "unverified", never a false "verified".
- Direct `module.forward(x)` calls and the fast run: such a call skips `Module.__call__` and so
  every torch hook, which the fast live run uses to collect sites, apply staged steering and fold
  the call fingerprint. Capture's session `forward` wrapper marks each module it entered that
  way (`ModuleMeta.direct_forward_called` on the live module, kept across captures; the call is
  told apart by its caller frame through `_utils/_module_call_dispatch.py`, gated by
  `HAS_MODULE_CALL_DISPATCH_FRAME`), and the fast run
  routes exactly those through a per-run instance `forward` (`_fast_run.py`), restored by
  identity after each run; every other module stays on hooks. A direct call no capture
  marked (no capture of the model made one, or the flag is False) is still refused as
  `run_capability_unavailable` at stage `fast_live_module_hook_unobserved`.
- **Foreign data round trips link every compatible departure.** After a plain-code
  `.numpy()` / `__array__` / `__dlpack__` / `to_dlpack` / `.tolist()` export, a later host
  constructor with no tensor argument that matches no departure by memory or by value is
  recorded as an `unresolved` `foreigndata` op with every compatible departure as a parent
  (buffer arguments link all departures; list and scalar arguments link `tolist` departures
  only) and validation fails it with `foreign_data_roundtrip_unreplayable`. A genuine constant
  built after such an export therefore fails validation by design: build it before the
  forward or wrap the foreign step in an `autograd.Function`. `.item()`, `int()`, `float()`
  and `bool()` are control-flow scalars under the completeness witness's scalar belt and are
  deliberately not departures.
- torch 2.1 and 2.2 wheels were compiled against NumPy 1: under NumPy 2 they import, but
  `torch.from_numpy`, `Tensor.numpy()` and `np.asarray(tensor)` raise `RuntimeError: Numpy is not
  available` (torch 2.3+ is fine). The lazy `HAS_NUMPY_BRIDGE` flag in `_utils/_torch_compat.py`
  names that state. Product code that needs a tensor's bytes or a host array goes through
  `tensor_payload_bytes` or `copy_host_bytes` (`_utils/tensor_leaves.py`), `digest_byte_view`
  (`_transport.py`), or `tensor_to_numpy`, `tensor_from_numpy` and `host_array`
  (`_utils/_numpy_bridge.py`), never a bare `.numpy()` or `torch.from_numpy`; each falls back to
  a NumPy-free byte copy.
  Pinned in `tests/test_numpy_bridge_compat.py`.
