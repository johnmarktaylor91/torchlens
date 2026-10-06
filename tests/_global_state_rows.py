"""Lifecycle-class row ledger for the global-state inventory census.

Pure data, split out of ``test_global_state_inventory.py`` at the seam its
ratchet ledger row named (2026-08-26): the lifecycle-class frozensets carry
most of the line count and are append-per-new-global, while the census
machinery and the assertions that consume them must be read together and
stay in the test module. Every mutable module global in the package is
classified into EXACTLY ONE lifecycle class below (plus the orthogonal
``_WEAKLY_HELD`` storage-fact ledger); a new global in ``torchlens/`` lands
as a row HERE, with a rationale comment when the classification is not
obvious from the name.
"""

_SCOPED_CAPTURE_STATE = frozenset(
    {
        # torch.ops recorder suppression depth: incremented/decremented in the
        # census redispatch's own try/finally; a survivor would silently stop
        # recording user torch.ops calls.
        ("torchlens/backends/torch/_torch_ops_calls.py", "_SUPPRESS_DEPTH"),
        # Governed-artifact-load window depth: incremented/decremented in the
        # governed_artifact_load() try/finally around .tlspec restore frames;
        # a survivor past the load window would arm the unknown-field refusal
        # for plain session pickles -- exactly the leak class this row catches.
        ("torchlens/_io/state_contract.py", "_GOVERNED_LOAD_DEPTH"),
        # The ONE active profiler-session slot (C06): set on ProfilerSession
        # __enter__ under its lock, restored to None in the finally of BOTH
        # __enter__'s failure path and __exit__ -- a survivor past the context
        # would make every later session refuse profiler_session_nested,
        # exactly the leak class this row catches.
        ("torchlens/observability/_session.py", "_ACTIVE_SESSION"),
        # Held torch-ref rebind journals, keyed by session: registered in
        # per-session model prep and drained at session cleanup -- a survivor
        # past the session would leave a user's held reference pointing at a
        # wrapper, exactly the leak class this row catches.
        ("torchlens/backends/torch/_held_refs_capture.py", "_PENDING_UNDO"),
        # The four scalar control slots below are assigned ONLY through the
        # module object from other modules (no ast.Global anywhere), so the
        # pre-rebind-detector inventory could never classify them
        # (hunt-b2-sol R54).
        ("torchlens/_state.py", "_active_fast_run_collector"),
        ("torchlens/_state.py", "_hook_reentrancy_depth"),
        ("torchlens/_state.py", "_nonowner_belt_armed"),
        ("torchlens/_state.py", "_runnable_ledger_armed"),
        ("torchlens/_state.py", "_active_hook_plan"),
        ("torchlens/_state.py", "_active_intervention_spec"),
        ("torchlens/_state.py", "_active_owner_thread_id"),
        ("torchlens/_state.py", "_aten_recording_armed"),
        # _active_record_spans left this ledger in fixwave-5 (R54): it is now
        # a never-rebound ContextVar holding an immutable tuple, so it is no
        # longer process-global mutable state at all.
        ("torchlens/_state.py", "_active_trace"),
        ("torchlens/_state.py", "_capture_replay_templates"),
        # Pre-admission reservation: claimed before any capture-global side
        # effect, released in run_and_log's outermost finally (refused-loser
        # data-quality fix, hunt-b2 R54). The claim token authenticates
        # same-thread re-entry (grind-r6 b7 R55) and shares the reservation's
        # exact lifetime.
        ("torchlens/_state.py", "_capture_reserved_by"),
        # Continuation token minted with the reservation claim and cleared
        # with its release: same-thread re-entry must present it, so a nested
        # public capture inside the reserved window refuses typed (r7 R55).
        ("torchlens/_state.py", "_capture_reservation_token"),
        ("torchlens/_state.py", "_dynamo_warning_emitted"),
        ("torchlens/_state.py", "_func_call_id_iter"),
        ("torchlens/_state.py", "_function_call_counts"),
        ("torchlens/_state.py", "_function_call_models"),
        ("torchlens/_state.py", "_functorch_warning_emitted"),
        ("torchlens/_state.py", "_logging_enabled"),
        ("torchlens/_state.py", "_relationship_input_id"),
        ("torchlens/_state.py", "_relationship_input_shape_hash"),
        ("torchlens/_state.py", "_relationship_model_class"),
        ("torchlens/_state.py", "_relationship_model_id"),
        ("torchlens/_state.py", "_relationship_weight_fingerprint"),
        ("torchlens/_state.py", "_tagged_buffer_ids"),
        ("torchlens/_trace_core/op_store.py", "_CLONE_SCOPE_DEPTH"),
        ("torchlens/backends/mlx/wrappers.py", "_ACTIVE_TAP_OBSERVER"),
        ("torchlens/backends/paddle/wrappers.py", "_ACTIVE_TAP_OBSERVER"),
        ("torchlens/backends/tinygrad/backend.py", "_ACTIVE_TINYGRAD_MODULE_STACK"),
        ("torchlens/backends/torch/_completeness_finalize.py", "_ACTIVE_WITNESS_STATE"),
        # Plane-W funcol completion session: published by the capture-scoped
        # distributed_recording_session context (armed captures only) and
        # cleared in its finally -- a survivor past the capture is exactly the
        # leaked-interposition class the session's teardown exists to prevent.
        ("torchlens/backends/torch/funcol.py", "_ACTIVE_FUNCOL_SESSION"),
        ("torchlens/backends/torch/_tl.py", "_ACTIVE_LABEL_SESSION"),
        ("torchlens/backends/torch/buffer_writes.py", "_WITNESS_MARKER_STATE"),
        ("torchlens/backends/torch/completeness_witness.py", "_CAPTURED_STORAGE_PTRS"),
        ("torchlens/backends/torch/completeness_witness.py", "_DISPATCH_TENSOR_ORIGINS"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_LEDGER_FACTS"),
        # RF gradient-probe window depth: incremented/decremented in a
        # try/finally around each probe (receptive_field/_gradient.py), read
        # by the hot paths to suppress capture side effects inside the window.
        ("torchlens/_state.py", "_rf_probe_depth"),
        ("torchlens/capture/projections.py", "_active_recording_state"),
        # Episode join/coupling sessions (F40c/F42): published by
        # armed_capture for exactly one capture and cleared in its finally --
        # a survivor past the capture is exactly the leak class this ledger
        # catches (captures are single-threaded by design).
        ("torchlens/capture/_episode_join.py", "_ACTIVE_JOIN_SESSION"),
        ("torchlens/capture/_episode_coupling.py", "_ACTIVE_COUPLING"),
        ("torchlens/capture/trace.py", "_ACTIVE_CAPTURE_BACKEND"),
        ("torchlens/experimental/__init__.py", "_STOP_AFTER_SITE"),
        # Live active_intervention_context publication stack (fixwave-5):
        # entries are pushed at context entry and spliced out on unwind, so a
        # survivor past the last context is exactly the leak class this
        # ledger catches.
        ("torchlens/intervention/runtime.py", "_CONTEXT_ENTRIES"),
        ("torchlens/utils/introspection.py", "_FUNC_CALL_LOCATION"),
        ("torchlens/utils/rng.py", "_ACTIVE_MONITOR"),
        # Live monitor patches per (id(holder), name), spliced back on window
        # unwind -- a survivor past monitor exit is exactly the leak class this
        # row exists to catch (d327e3aa's non-LIFO restore bug).
        ("torchlens/utils/rng.py", "_PATCH_STACKS"),
        # CUDA RNG snapshot retry latch: set by a failed generator read, RE-ARMED
        # at every capture entry by set_random_seed (grind p5, B2P3-16). It was
        # misfiled as an immutable capability memo, which contractually forbade
        # ever recovering from a TRANSIENT read failure (busy device, momentary
        # OOM) -- a capture-fidelity latch, not a fact of the torch build.
        ("torchlens/utils/rng.py", "_cuda_rng_unusable"),
        # Accumulate/drain fence for in-flight cpu_async D2H copies (R36-1):
        # armed per copy on the wrapper hot path, drained at the capture
        # finalize seam and on the failure-scrub arms.
        ("torchlens/utils/tensor_utils.py", "_CPU_ASYNC_PENDING_EVENTS"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_BUSY"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_PENDING"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_STATE_PTRS"),
        ("torchlens/utils/tensor_utils.py", "_DEFER_WINDOW_DEPTH"),
    }
)
"""Per-capture state: set during one capture and cleared or restored on exit.

A leak here is a correctness bug -- the next capture inherits a live owner, an
open session, or a stale window depth. ``test_mid_capture_failure_restores_process_state``
pins the high-risk members against exception and interruption paths.
"""

_INSTALL_STATE_AND_CACHES = frozenset(
    {
        # torch.ops call-class recorders: originals captured at wrap_torch, restored
        # and cleared at unwrap_torch; the per-operator wrapper cache is cleared with
        # them; the shared wrapper code object is an install-time constant.
        ("torchlens/backends/torch/_torch_ops_calls.py", "_DECORATED_BY_OP"),
        ("torchlens/backends/torch/_torch_ops_calls.py", "_ORIGINAL_CALLS"),
        ("torchlens/backends/torch/_torch_ops_calls.py", "_WRAPPED_FUNC_CODE"),
        # S3 pre-release field registrar (wave -0.5). _REGISTRY and
        # _ANNOTATIONS_KEY_REGISTRY are DECLARATION tables populated at import
        # by register_prerelease_field (DROP-only, refuses non-DROP), so they are
        # install-time facts, not caches. _ACTIVE is the TEST-ONLY activation
        # flag: activate_prerelease_fields refuses outside pytest and restores
        # it, and every switch-on write is marker-stamped so a leak cannot pass
        # as a real v7 artifact.
        ("torchlens/_io/prerelease.py", "_ACTIVE"),
        ("torchlens/_io/prerelease.py", "_ANNOTATIONS_KEY_REGISTRY"),
        ("torchlens/_io/prerelease.py", "_REGISTRY"),
        # Selection-AST term dispatch: import-time table, sibling producers register at import.
        ("torchlens/selection.py", "_TERM_RESOLVERS"),
        # F40a annotations travel policy: DECLARATION table populated at import
        # by register_travel_policy (closed drop_mode vocabulary; sub-key owners
        # register at import of their owning modules, the _ANNOTATIONS_KEY_REGISTRY
        # pattern) -- install-time fact, not a cache.
        ("torchlens/capture/_annotations_travel.py", "_TRAVEL_REGISTRY"),
        # Transforms builtin seal (C04): one-way sentinel flipped by
        # _seal_builtins() when _kernels finishes import-time builtin
        # registration; from then on the builtin door refuses
        # transform_builtin_shadowed -- install-time fact, not a cache.
        ("torchlens/transforms/_registry.py", "_BUILTINS_SEALED"),
        # Wrapper-lifecycle slots rebound only through the module object
        # (visible since the cross-module rebind detector, hunt-b2-sol R54).
        # Kernel-telemetry correlation installs by rebinding these two aten-call
        # slots THROUGH THE MODULE OBJECT (kernel_telemetry.py:563-591) and
        # restores the originals on exit -- the same wrapper-lifecycle pattern as
        # the _state.py slots below, not a cache.
        ("torchlens/backends/torch/_aten_capture.py", "_finish_aten_call"),
        ("torchlens/backends/torch/_aten_capture.py", "_prepare_aten_call"),
        ("torchlens/_state.py", "_decorated_identity"),
        ("torchlens/_state.py", "_is_decorated"),
        ("torchlens/_state.py", "_wrap_epoch"),
        ("torchlens/_state.py", "_decorated_func_mapper"),
        ("torchlens/_state.py", "_decorated_to_orig"),
        ("torchlens/_state.py", "_orig_to_decorated"),
        ("torchlens/_state.py", "_prepared_models"),
        ("torchlens/_state.py", "_prepared_root_by_module"),
        ("torchlens/_state.py", "_stale_prepared_roots"),
        ("torchlens/backends/torch/_tl.py", "_RETIRED_LABEL_SESSION"),
        # F3b's lazy public-impl metadata sync (3c7ed93e): a one-way
        # synced-yet? sentinel flipped on first successful wrap, install-class.
        ("torchlens/user_funcs.py", "_public_impl_metadata_synced"),
        # Released-model registry (fixwave-5): wrap_torch()/unwrap_torch()
        # re-normalize held torch-function refs on every registered model, so
        # the registration rides the wrapper install lifecycle (WeakSet -- a
        # registration never pins the released model; row also in
        # _WEAKLY_HELD).
        ("torchlens/backends/torch/_held_refs.py", "_RELEASED_MODELS"),
        ("torchlens/backends/torch/backward.py", "_AUTOGRAD_WRAPPERS_INSTALLED"),
        # Installed-wrapper identity snapshots (fixwave-5, grind-r5 b8 R56):
        # teardown restores a slot only when it still holds OUR wrapper, so
        # the installed identities are wrapper-lifecycle state exactly like
        # their _ORIGINAL_* companions.
        ("torchlens/backends/torch/backward.py", "_INSTALLED_AUTOGRAD_BACKWARD"),
        ("torchlens/backends/torch/backward.py", "_INSTALLED_AUTOGRAD_GRAD"),
        ("torchlens/backends/torch/backward.py", "_INSTALLED_SAVED_TENSORS_HOOKS_ENTER"),
        ("torchlens/backends/torch/backward.py", "_INSTALLED_SAVED_TENSORS_HOOKS_INIT"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_AUTOGRAD_BACKWARD"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_AUTOGRAD_GRAD"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_SAVED_TENSORS_HOOKS_ENTER"),
        ("torchlens/backends/torch/backward.py", "_ORIGINAL_SAVED_TENSORS_HOOKS_INIT"),
        ("torchlens/backends/torch/backward.py", "_SAVED_TENSORS_HOOKS_INIT_PATCHED"),
        # Patched legacy constructor classes (torch.<dtype>Tensor, Variable):
        # installed by wrap_torch, restored and emptied by unwrap_torch.
        ("torchlens/backends/torch/legacy_ctors.py", "_INSTALLED"),
        # Classes the last install skipped over a foreign Python __new__
        # (rebuilt on every install).
        ("torchlens/backends/torch/legacy_ctors.py", "_SKIPPED_FOREIGN_NEW"),
        ("torchlens/backends/torch/belt.py", "_ledger"),
        ("torchlens/backends/torch/belt.py", "_member_map"),
        ("torchlens/backends/torch/belt.py", "_report"),
        # Sweep pre-filter companion to _swept_module_ids, cleared at the same belt
        # reset site. Holds plain ids, so it is NOT a _WEAK_SUBJECT_TABLES row even
        # though correctness rests on weakref death-callback EVICTION rather than on
        # clearing: an evicted id means a recycled id reads as new, never as swept.
        ("torchlens/backends/torch/belt.py", "_swept_ids_live"),
        ("torchlens/backends/torch/belt.py", "_swept_module_ids"),
        # The 8ba75e99 sweep-epoch companions (_swept_modules_dirty,
        # _swept_sys_modules_size) were deleted with the sys.modules crawler
        # in fixwave-5; their rows left this ledger shrink-only.
        ("torchlens/backends/torch/completeness_witness.py", "_AUTHORIZED_INTERNAL_CALLER_CODE"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_AUTHORIZED_INTERNAL_CALLER_CODE_IDS",
        ),
        ("torchlens/backends/torch/escape_detection.py", "_TABLES"),
        # (holder, attribute, original) rows for every installed identity shim;
        # popped by the shim uninstall, so it is install bookkeeping, not capture
        # state.
        # One-way "shim family installed" sentinel alongside _installed; reset
        # by the shim uninstall path.
        ("torchlens/backends/torch/identity_shims.py", "_family_installed"),
        ("torchlens/backends/torch/identity_shims.py", "_installed"),
        # Live shim identity registry (fixwave-6 capture-r7 8ac52828): id ->
        # shim rows written at shim install so handler-presentation can prove
        # a callable is OUR live shim by identity; cleared by the shim
        # uninstall alongside _installed, so it is install bookkeeping, not
        # capture state.
        ("torchlens/backends/torch/identity_shims.py", "_live_shims"),
        # Meta-path finder handle for the lazy causal-bias shim (fix/rescue
        # dcd0ca9c): installed once so a post-wrap `import transformers` still
        # gets the shim, removed by the shim uninstall alongside _installed.
        ("torchlens/backends/torch/identity_shims.py", "_import_hook"),
        # First-read source digests for conditional/source attribution (deep-hunt
        # C4, fix/postproc 7a58a021). Deliberately NOT a _PROCESS_CACHES row:
        # clearing it is not correctness-neutral (the pin is the fail-closed
        # baseline that a rebuilt _file_cache entry must match), so it survives
        # LRU eviction and is cleared only by explicit invalidate_cache().
        ("torchlens/postprocess/ast_branches.py", "_pinned_source_digests"),
        ("torchlens/backends/torch/wrappers.py", "_DEVICE_CONSTRUCTOR_NAMES"),
        ("torchlens/backends/torch/wrappers.py", "_DeviceContext"),
        # One-way "decorate_all_once() ran to COMPLETION" sentinel; deliberately
        # never reset by unwrap_torch() (partial-decoration recovery keys on it).
        ("torchlens/backends/torch/wrappers.py", "_FULL_DECORATION_COMPLETED"),
        # One-way "os.register_at_fork child-hygiene handler registered" latch;
        # fork handlers cannot be unregistered, so the latch never resets.
        ("torchlens/backends/torch/wrappers.py", "_AT_FORK_HYGIENE_INSTALLED"),
        ("torchlens/backends/torch/wrappers.py", "_torchvision_ops_ensured"),
        ("torchlens/capture/arg_positions.py", "_schema_corrections_applied"),
        # Armed distributed state: process-lifetime by design, it survives
        # destroy_process_group so the lifecycle ledger outlives every group.
        # It must not steer a capture with no initialized group: per-capture
        # consumers read it through capture_armed_state(), which is None then
        # (tests/test_distributed_teardown_dormancy.py pins it).
        ("torchlens/distributed/_lifecycle.py", "_STATE"),
    }
)
"""Wrapper install / uninstall bookkeeping and prepared-model registration.

Process-lifetime by design: these hold the torch originals, the decorated-callable
id maps, the belt ledger, and the prepared-model registry. They must survive
between captures and be restored by ``unwrap_torch()`` / ``release_model()``,
not reset per capture.
"""

_WARN_ONCE_STATE = frozenset(
    {
        # Post-T52 census sweep (F27): warn-once ledgers other merged lanes
        # added without classifying -- one disclosure per key, append-only.
        ("torchlens/attribution/onebackward/_frozen.py", "_DISCLOSURE_WARNED"),
        ("torchlens/quickstart/_gate.py", "_WARNED_TRACE_IDS"),
        ("torchlens/_capture_state_helpers.py", "_COMPILED_FORCED_EAGER_WARNED"),
        ("torchlens/_capture_state_helpers.py", "_COMPILED_MODEL_UNWRAP_WARNED"),
        ("torchlens/_capture_state_helpers.py", "_VALIDATION_DEEPCOPY_WARNING_TYPES"),
        ("torchlens/_io/bundle.py", "_NONPERSISTENT_DISCLOSURE_WARNED"),
        ("torchlens/_io/bundle.py", "_UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED"),
        ("torchlens/backends/tf/_tf_compat.py", "_warned_missing_capabilities"),
        ("torchlens/backends/torch/buffer_writes.py", "_PARAM_BYTE_WITNESS_NOT_ARMED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_OBSERVER_FAILED"),
        ("torchlens/backends/torch/legacy_ctors.py", "_WARNED_FOREIGN_NEW"),
        ("torchlens/backends/torch/ops.py", "_UNSUPPORTED_OUTPUT_CONTAINER_WARNED"),
        # BatchNorm train-mode running-stats disclosure fires once per process
        # (user_funcs warn-once flag, C02 lovely tranche).
        ("torchlens/user_funcs.py", "_BATCHNORM_TRAIN_STATS_WARNED"),
        ("torchlens/data_classes/op.py", "_WARNED_REFERENCE_SAVE_MODE"),
        ("torchlens/distributed/_lifecycle.py", "_AUTO_ARM_DEGRADATION"),
        ("torchlens/fastlog/_storage_resolver.py", "_WARNED_REFERENCE_SAVE_MODE"),
        # Once-per-file source-drift disclosure for the C4 pinned-digest refusal
        # (fix/postproc 7a58a021); reset row lives in tests/conftest.py.
        ("torchlens/postprocess/ast_branches.py", "_source_drift_warned"),
        ("torchlens/utils/_torch_compat.py", "_warned_missing_capabilities"),
        ("torchlens/utils/introspection.py", "_col_offset_cache_warned"),
        ("torchlens/validation/_stock_layer_grads.py", "_PASS_INDEX_PARSE_WARNED"),
        ("torchlens/visualization/_render_ordering.py", "_SIBLING_ORDER_WARNING_EMITTED"),
        ("torchlens/visualization/auto_collapse.py", "_COUNT_MISMATCH_WARNING_EMITTED"),
    }
)
"""Once-per-process disclosure sentinels.

Each suppresses a repeat warning. They are ORDER-COUPLING for tests: whichever
test fires the warning first consumes it, so a suite that asserts on the warning
must reset them (the reset fixture is owned by ``tests/conftest.py``). Any new
sentinel landing here without a reset is an order-dependence bug waiting to
happen.
"""

_CAPABILITY_PROBE_STATE = frozenset(
    {
        # Post-T80 census sweep (F42 reconcile): the tl.func-matching wrap
        # universe -- derived once from the installed wrapper layer on first
        # use, never varies while wrapped (F21).
        ("torchlens/intervention/binding.py", "_WRAP_UNIVERSE"),
        # Lazy glibc malloc_trim probe (R33): False = unprobed, None =
        # unavailable, else the resolved libc function. Probed once at the
        # first cleanup(); never varies afterwards.
        ("torchlens/data_classes/cleanup.py", "_MALLOC_TRIM"),
        ("torchlens/utils/_torch_compat.py", "HAS_C10D_ABORT_PG"),
        # F27 Kineto/memory-profile capability flags: probed once at first
        # extraction (a live profiler session is required to observe an
        # event instance), never vary afterwards.
        ("torchlens/utils/_torch_compat.py", "HAS_KINETO_INMEMORY_EVENTS"),
        ("torchlens/utils/_torch_compat.py", "HAS_KINETO_EVENT_SCOPE"),
        ("torchlens/utils/_torch_compat.py", "HAS_MEMORY_PROFILE"),
        ("torchlens/utils/_torch_compat.py", "_KINETO_INMEMORY_EVENTS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_KINETO_EVENT_SCOPE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_MEMORY_PROFILE_PROBED"),
        # F27 wrappers hot-path accessor cache: resolved once on the first
        # wrapped call (import cost off the hot path), never varies after.
        ("torchlens/backends/torch/_op_markers.py", "_active_session_fn"),
        ("torchlens/utils/_torch_compat.py", "HAS_DISABLE_TORCH_FUNCTION"),
        ("torchlens/utils/_torch_compat.py", "HAS_DISPATCH_MODE_STACK_QUERY"),
        ("torchlens/utils/_torch_compat.py", "HAS_DTENSOR_SHARD_GEOMETRY"),
        ("torchlens/utils/_torch_compat.py", "HAS_FAKE_TENSOR_MODE"),
        ("torchlens/utils/_torch_compat.py", "HAS_JIT_SCHEMA_ENUMERATION"),
        ("torchlens/utils/_torch_compat.py", "HAS_TENSORBASE_CLASS"),
        ("torchlens/utils/_torch_compat.py", "HAS_VARIABLE_FUNCTIONS_CLASS"),
        ("torchlens/utils/_torch_compat.py", "HAS_C10D_GROUP_REGISTRY"),
        ("torchlens/utils/_torch_compat.py", "HAS_C10D_GROUP_SEQ"),
        ("torchlens/utils/_torch_compat.py", "HAS_DEVICE_MESH"),
        ("torchlens/utils/_torch_compat.py", "HAS_DTENSOR"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_COMPILE_COUNTERS"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_IS_COMPILING"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_OPTIMIZED_MODULE"),
        ("torchlens/utils/_torch_compat.py", "HAS_DYNAMO_ORIG_CALLABLE_MARKER"),
        ("torchlens/utils/_torch_compat.py", "HAS_FP8_DTYPES"),
        ("torchlens/utils/_torch_compat.py", "HAS_FSDP_WRAPPER"),
        ("torchlens/utils/_torch_compat.py", "HAS_PIPELINING"),
        ("torchlens/utils/_torch_compat.py", "HAS_TRACING_TENSOR_TYPES"),
        ("torchlens/utils/_torch_compat.py", "_C10D_ABORT_PG_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DISABLE_TORCH_FUNCTION_CLS"),
        ("torchlens/utils/_torch_compat.py", "_DISABLE_TORCH_FUNCTION_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DISPATCH_MODE_STACK_FN"),
        ("torchlens/utils/_torch_compat.py", "_DISPATCH_MODE_STACK_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_C10D_GROUP_REGISTRY_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_C10D_GROUP_SEQ_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DEVICE_MESH_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DEVICE_MESH_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_SHARD_GEOMETRY_FN"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_SHARD_GEOMETRY_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DTENSOR_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_COMPILE_COUNTERS"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_COMPILE_COUNTERS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_IS_COMPILING_FN"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_IS_COMPILING_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_OPTIMIZED_MODULE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_OPTIMIZED_MODULE_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_DYNAMO_ORIG_CALLABLE_MARKER_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FAKE_TENSOR_MODE_CLS"),
        ("torchlens/utils/_torch_compat.py", "_FAKE_TENSOR_MODE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FP8_DTYPES"),
        ("torchlens/utils/_torch_compat.py", "_FP8_DTYPES_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FSDP_WRAPPER_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FSDP_WRAPPER_TYPE"),
        ("torchlens/utils/_torch_compat.py", "HAS_FUNCOL_GROUP_RESOLUTION"),
        ("torchlens/utils/_torch_compat.py", "HAS_FUNCOL_WAIT_INTERPOSITION"),
        ("torchlens/utils/_torch_compat.py", "_FUNCOL_GROUP_RESOLVERS"),
        ("torchlens/utils/_torch_compat.py", "_FUNCOL_GROUP_RESOLUTION_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_FUNCOL_WAIT_INTERPOSITION_PROBED"),
        # fix/private-probe-routing: the L8/C2 funcol + L9 backward private
        # touches now resolve through compat, each a lazy HAS_* family (the
        # _tl.py-local _ASYNC_COLLECTIVE_TENSOR_CLASS memo moved here).
        ("torchlens/utils/_torch_compat.py", "_FUNCOL_WAIT_REDISPATCH"),
        ("torchlens/utils/_torch_compat.py", "HAS_FUNCOL_MODULE"),
        ("torchlens/utils/_torch_compat.py", "_FUNCOL_MODULE_OBJ"),
        ("torchlens/utils/_torch_compat.py", "_FUNCOL_MODULE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_ASYNC_COLLECTIVE_TENSOR"),
        ("torchlens/utils/_torch_compat.py", "_ASYNC_COLLECTIVE_TENSOR_TYPE"),
        ("torchlens/utils/_torch_compat.py", "_ASYNC_COLLECTIVE_TENSOR_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_CHECKPOINT_HOOK_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_CHECKPOINT_HOOK_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_CHECKPOINT_HOOK_CLASS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_CHECKPOINT_INTERNAL_HOOK_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_CHECKPOINT_INTERNAL_HOOK_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_CHECKPOINT_INTERNAL_HOOK_CLASS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_AUTOGRAD_ENGINE_QUEUE_CALLBACK"),
        ("torchlens/utils/_torch_compat.py", "_AUTOGRAD_ENGINE_QUEUE_CALLBACK"),
        ("torchlens/utils/_torch_compat.py", "_AUTOGRAD_ENGINE_QUEUE_CALLBACK_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_JIT_SCHEMA_ENUMERATION_FN"),
        ("torchlens/utils/_torch_compat.py", "_JIT_SCHEMA_ENUMERATION_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_PIPELINING_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_PIPELINING_TYPES"),
        ("torchlens/utils/_torch_compat.py", "_TENSORBASE_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_TENSORBASE_CLASS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_TOP_SAVED_TENSORS_DEFAULT_HOOKS_ARGS"),
        ("torchlens/utils/_torch_compat.py", "_TRACING_TENSOR_TYPES"),
        ("torchlens/utils/_torch_compat.py", "_TRACING_TENSOR_TYPES_PROBED"),
        ("torchlens/utils/_torch_compat.py", "_VARIABLE_FUNCTIONS_CLASS"),
        ("torchlens/utils/_torch_compat.py", "_VARIABLE_FUNCTIONS_CLASS_PROBED"),
        # One-shot warm of torch's lazy torch._compile/torch._dynamo cascade,
        # fired by the RNG monitor BEFORE its window arms (hunt-b8 F1).
        ("torchlens/utils/_torch_compat.py", "_LAZY_TORCH_IMPORTS_WARMED"),
        ("torchlens/utils/tensor_utils.py", "_cuda_available"),
        # floor2 fix (2026-10-01): torch 2.1.2-floor capability probes --
        # GradientEdge (one-backward tuple-output signature), the
        # tuple-dim overload of reduce ops, CPU half-precision kernel
        # availability, and the CPU float8 deterministic-fill path -- each a
        # lazy HAS_* probed once on first use, same shape as every other row
        # in this class.
        ("torchlens/utils/_torch_compat.py", "HAS_GRADIENT_EDGE"),
        ("torchlens/utils/_torch_compat.py", "_GRADIENT_EDGE_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_REDUCE_TUPLE_DIM"),
        ("torchlens/utils/_torch_compat.py", "_REDUCE_TUPLE_DIM_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_CPU_HALF_KERNELS"),
        ("torchlens/utils/_torch_compat.py", "_CPU_HALF_KERNELS_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_CPU_FLOAT8_DETERMINISTIC_FILL"),
        ("torchlens/utils/_torch_compat.py", "_CPU_FLOAT8_DETERMINISTIC_FILL_PROBED"),
        ("torchlens/utils/_torch_compat.py", "HAS_META_ITEM_GUARD"),
        ("torchlens/utils/_torch_compat.py", "_META_ITEM_GUARD_PROBED"),
        # ratchet2 ci-fix (2026-10-01): the MHA fastpath-switch probe joined
        # the lazy pattern (was eager at import, tripping the import-hygiene
        # duration budget on the torch>=2.1 floor's py3.10 leg); same shape
        # as every other row in this class.
        ("torchlens/utils/_torch_compat.py", "HAS_MHA_FASTPATH_SWITCH"),
        ("torchlens/utils/_torch_compat.py", "_MHA_FASTPATH_SWITCH_PROBED"),
    }
)
"""Feature-detection memos for the running torch build.

Written once by a lazy probe and then immutable for the process. They are facts
of the runtime, never capture state, and must never be reset to force a
behavioral branch -- ``CLAUDE.md`` forbids version parsing precisely because
these flags are the sanctioned mechanism.
"""

_DIAGNOSTIC_AUDIT_STATE = frozenset(
    {
        # Diagnostic shadow-mode toggles ("off"/"shadow"), flipped only by the
        # detector/witness install surfaces; assigned solely through the
        # module object (visible since the cross-module rebind detector).
        ("torchlens/_state.py", "_completeness_witness_mode"),
        ("torchlens/_state.py", "_escape_detector_mode"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_CLONE_READS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_COLLECTORS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_FINGERPRINTS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_READS"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_ROW_RELEASES"),
        ("torchlens/_trace_core/op_store.py", "_AUDIT_WRITE_EFFECTS"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_CLONE_READS"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_EFFECTIVE_WRITES"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_READS"),
        ("torchlens/postprocess/__init__.py", "RECORDED_STEP_WRITES"),
        # Monotone count of run-fold legality-grammar checks (F11 B2): a pure
        # test instrument read via legality_check_count(); never reset by
        # library code and never steers behavior.
        ("torchlens/visualization/_collapse_runs.py", "_LEGALITY_CHECKS"),
        # Last-run validation readbacks (B8-42 / R33-2): the internal
        # validation trace never escapes tl.validate, so the first failure
        # and the peak observation mirror into these slots, cleared at each
        # run entry. Diagnostics only -- never part of a verdict.
        ("torchlens/validation/diagnostics.py", "_LAST_RUN_FAILURE"),
        ("torchlens/validation/diagnostics.py", "_LAST_RUN_PEAKS"),
    }
)
"""Diagnostic side-channels: audit instrumentation and last-run readbacks.

The audit rows are armed only by an environment variable and stay empty in
production runs (``TORCHLENS_POSTPROCESS_ASSERTIONS`` /
``TORCHLENS_POSTPROCESS_READ_AUDIT`` off); they accumulate within one armed
window and are scoped by the executor's begin/end pass. The validation
last-run slots are overwritten per run and never steer a verdict.
"""

_WEAK_SUBJECT_TABLES = frozenset(
    {
        ("torchlens/_state.py", "_log_registry"),
        # L9 runtime journals are keyed by the owning Trace and contain only
        # timing/token/finalization bookkeeping. They must disappear with
        # that Trace and never become process-lifetime caches.
        ("torchlens/backends/torch/backward.py", "_BACKWARD_TRACE_SLOTS"),
        ("torchlens/backends/torch/backward.py", "_CHECKPOINT_TOKEN_STATE"),
        ("torchlens/backends/torch/_fire_timing.py", "_FIRE_TIMING_STAMPS"),
        # F27 FLIP-2 open-marker LIFO + session-time Kineto join results:
        # keyed weakly by the owning trace, die with it; the marker LIFO is
        # additionally drained at every pass boundary.
        ("torchlens/backends/torch/_gradfn_markers.py", "_GRADFN_MARKER_TOKENS"),
        ("torchlens/observability/_native_profile.py", "_JOIN_TRACES"),
        # Post-T52 census sweep (F27): weak-keyed caches other merged lanes
        # added without classifying -- keyed by the owning trace/bundle
        # member, they die with their subject.
        ("torchlens/attribution/onebackward/_accessor.py", "_INDEX_CACHE"),
        ("torchlens/attribution/onebackward/_edge_plumbing.py", "_PASS_STAMPS"),
        ("torchlens/bundle/_compare_gate.py", "_INPUT_VALUE_DIGEST_CACHE"),
        ("torchlens/backends/torch/backward.py", "_PENDING_BACKWARD_FINALIZE"),
        # Weightsfree (F33) session registries: weak-keyed by the owning Trace
        # (admission record, W1-ORD wrap-generation stamp, live admitted-meta
        # membership) -- the ledger pattern of the completeness-witness tables;
        # entries must vanish with their Trace, never outlive a capture.
        ("torchlens/capture/_weightsfree_admission.py", "_ADMISSIONS"),
        ("torchlens/capture/_weightsfree_admission.py", "_META_ACTIVE"),
        ("torchlens/capture/_weightsfree_admission.py", "_WRAP_GENERATIONS"),
        ("torchlens/backends/torch/completeness_witness.py", "_ALIAS_MUTATION_CANDIDATE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_DATA_ALIAS_MUTATION_TRACES"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_BOOL_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_CROSS_THREAD_CAPTURED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_LABEL_LEAF_ORIGINS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_MUTABLE_WRITEBACK"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_RAW_POINTER"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_SOURCE_LABELS"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_STATE_METADATA_OBSERVATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_METADATA_READS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_NAMES"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_BOOL"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE"),
        ("torchlens/backends/torch/completeness_witness.py", "_INPUT_METADATA_VIEW_READ"),
        ("torchlens/backends/torch/completeness_witness.py", "_LAYOUT_ANCESTRY_CLEAN"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_ALIAS_MUTATION_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_RNG_CONTROL_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_INPUT_STORAGE_SITES"),
        ("torchlens/backends/torch/completeness_witness.py", "_STATE_METADATA_FACTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_STORAGE_REBIND_BARRIER_LABELS"),
        ("torchlens/backends/torch/model_prep.py", "_source_line_cache"),
        # Implicit-backward task ordinals keyed weakly by their owning trace;
        # entries die with the trace.
        ("torchlens/backends/torch/tensor_tracking.py", "_IMPLICIT_BACKWARD_TASK_IDS"),
        # Model-state gradient-hook handles keyed weakly by their owning trace;
        # cleanup() pops and removes them, and entries die with the trace.
        ("torchlens/backends/torch/tensor_tracking.py", "_OWNED_STATE_GRAD_HOOK_HANDLES"),
        ("torchlens/backends/torch/wrappers.py", "_COW_STATE_PTRS_CACHE"),
        ("torchlens/capture/structure_only.py", "_DISCHARGE_REGISTRY"),
        ("torchlens/data_classes/_compaction.py", "_COMPACTED_TRACES"),
        ("torchlens/data_classes/_nonfinite.py", "_MEMOS"),
        # Per-trace grad_fn-call ordinal index (fix/walkers f399c63a, linear
        # ordinal_index): keyed weakly by the owning trace, dies with it.
        ("torchlens/data_classes/grad_fn_call.py", "_ORDINAL_POSITIONS_CACHE"),
        # Per-accessor param ordinal index, keyed weakly by the owning
        # accessor; entries die with it (mirror of _ORDINAL_POSITIONS_CACHE).
        ("torchlens/data_classes/param.py", "_ORDINAL_INDEX_CACHE"),
        # Live telemetry relations are keyed by AtenOp facades and disappear
        # with those rows; they never retain a Trace or profiler session.
        ("torchlens/kernel_telemetry.py", "_ATEN_KERNELS"),
        ("torchlens/partial/__init__.py", "_FAILED_CAPTURE_RESULTS"),
        ("torchlens/visualization/auto_collapse.py", "_ANALYSIS_CACHE"),
        ("torchlens/visualization/auto_collapse.py", "_OP_ADJACENCY_INDEX_CACHE"),
        ("torchlens/visualization/code_panel.py", "_SOURCE_MEMO"),
        ("torchlens/visualization/collapse_optimizer.py", "_BOX_UNITS_CACHE"),
        # Warn-once set for over-ceiling collapse declines (r8 R60-13):
        # declined results no longer enter the revision-keyed result cache,
        # so the one-warning-per-trace dedup rides its own weak set.
        ("torchlens/visualization/collapse_optimizer.py", "_CEILING_WARNED_TRACES"),
        # Warn-once set for the F11 budget-degrade disclosure (memo D5(iv)):
        # auto dedupes per trace; explicit max re-warns every call (N15).
        ("torchlens/visualization/_collapse_disclosures.py", "_BUDGET_WARNED_TRACES"),
        ("torchlens/visualization/collapse_optimizer.py", "_RESULT_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_SCHEDULE_CACHE"),
        # W051-TRACK live session registry (ratchet2 census catch-up): keyed
        # weakly by the watched model via WeakKeyDictionary, so a session's
        # per-grammar entry dies with its model instead of outliving it.
        ("torchlens/trackers/_watch.py", "_ACTIVE_SESSIONS"),
    }
)
"""Side tables keyed WEAKLY by their subject (trace, tensor, model, code).

Entries die with the subject, so these can neither pin memory nor leak state
across captures. The test below verifies mechanically that every member really
is bound to a ``weakref`` container -- a member that silently becomes a strong
dict would otherwise keep its whole subject graph alive.
"""

_PUBLIC_REGISTRATION_STATE = frozenset(
    {
        # Post-T80 census sweep (F42 reconcile): registries merged lanes added
        # without heavy-census classification -- explicit register/list
        # surfaces (extraction pure-module namespaces, ecosystem plugin
        # activation ledger, preprocessing authority adapters).
        ("torchlens/_extraction/callable_identity.py", "_REGISTERED_PURE_MODULES"),
        ("torchlens/ecosystem/plugins.py", "_RECORDS"),
        ("torchlens/preprocessing/_authorities.py", "_REGISTERED_ADAPTERS"),
        # Post-T52 census sweep (F27): registries other merged lanes added
        # without classifying -- explicit register/list surfaces.
        ("torchlens/visualization/theme_registry.py", "_REGISTRY"),
        ("torchlens/attribution/onebackward/_facet_bridge.py", "_FROZEN_POLICIES"),
        ("torchlens/attribution/onebackward/_facet_bridge.py", "FROZEN_POLICY_NAMES"),
        ("torchlens/backends/registry.py", "_REGISTRY"),
        # The C01 registry-kernel inventory: domain doors enroll their one
        # Registry at import (create_registry refuses duplicates); public door
        # registrations mutate the enrolled Registry objects, and the kernel
        # inventory itself is what makes domains COUNTABLE (universe rows).
        ("torchlens/_registry/kernel.py", "_REGISTRIES"),
        # Semantic recipe-provider activation: activate_recipes() is a public
        # trust opt-in that changes which entry-point providers resolve for
        # the rest of the process -- registration state, not a cache.
        ("torchlens/semantic/recipes/__init__.py", "_ACTIVATED_RECIPE_PROVIDERS"),
        ("torchlens/capture/flops.py", "_CUSTOM_OP_RULES"),
        ("torchlens/ir/container.py", "_CONTAINER_REGISTRY"),
        ("torchlens/receptive_field/_rules.py", "_BUILTIN_RF_RULES"),
        ("torchlens/receptive_field/_rules.py", "_RF_RULES"),
        ("torchlens/receptive_field/_rules.py", "_RF_RULES_EPOCH"),
        ("torchlens/semantic/facets.py", "_BUILTIN_REGISTRY"),
        ("torchlens/semantic/facets.py", "_REGISTRY"),
        ("torchlens/semantic/facets.py", "_REGISTRY_VERSION"),
        ("torchlens/semantic/facets.py", "_TRANSFORMERLENS_ALIASES_ENABLED"),
        # L6/S4 predicate runtime extension point: the public registration hook
        # mutates this table, so a leaked entry changes predicate resolution for
        # the rest of the process exactly like a registered facet or RF rule.
        ("torchlens/ir/predicate_registry.py", "_USER_PREDICATES"),
        # Transforms closed builtin name set (C04): populated at import by
        # _kernels through the SAME door custom registrations use, then
        # sealed; membership drives the transform_builtin_shadowed refusal on
        # register_transform(), so it steers public registration behavior.
        ("torchlens/transforms/_registry.py", "_BUILTIN_NAMES"),
    }
)
"""Process state a PUBLIC API mutates: registries, rule tables, feature toggles.

Distinct from a cache because a user call CHANGES capture behavior for the rest
of the process (a registered facet, an RF rule, an enabled alias set). These are
the members whose leakage across a test session can silently change results, so
they carry an epoch/version counter where downstream caches must invalidate.
"""

_PROCESS_CACHES = frozenset(
    {
        # Post-T80 census sweep (F42 reconcile): bounded LRU stores merged
        # lanes added without heavy-census classification -- the projection
        # basis byte-budgeted digest store (+ its size counter) and the SRP
        # per-(instance, site) extent bindings with their test reset door.
        ("torchlens/transforms/_projection.py", "_BASIS_STORE"),
        ("torchlens/transforms/_projection.py", "_STORE_BYTES"),
        ("torchlens/transforms/_srp.py", "_EXTENT_BINDINGS"),
        # Weightsfree (F33) process-lifetime memos: the per-code-object
        # decomposition-frame classification (bounded by loaded code) and the
        # one-per-process D20 meta identity self-test verdict.
        ("torchlens/backends/torch/_weightsfree_transparency.py", "_CODE_CLASSIFICATION"),
        ("torchlens/capture/_weightsfree_admission.py", "_IDENTITY_SELF_TEST_RESULT"),
        # Write-once memo of the CPython tuplegetter descriptor type, probed
        # for the property-shadowed-namedtuple walkers (fix/walkers f399c63a);
        # holds only a builtin type, re-derivable at any time.
        ("torchlens/_input_walk.py", "_TUPLEGETTER_TYPE_CACHE"),
        ("torchlens/_input_walk.py", "_STOCK_NP_SCALAR_CACHE"),
        # C02 numbers-substrate caches: id-keyed, weakref-evicted,
        # _version-invalidated derivation caches over live subjects;
        # re-derivable at any time, never behavior-changing.
        ("torchlens/report/_factcore.py", "_FACTCORE_CACHE"),
        # C04 digest-kernel memo: per-device Horner weight tensors, a pure
        # deterministic derivation (base powers mod 2^64) re-computable at
        # any time; keyed by device string, never behavior-changing.
        ("torchlens/_data_substrate/digests.py", "_WEIGHTS_BY_DEVICE"),
        ("torchlens/stats/_tensor_stats.py", "_STATS_CACHE"),
        ("torchlens/_io/bundle.py", "_NESTED_BLOB_KINDS"),
        ("torchlens/_io/payload_codec.py", "_CODECS"),
        ("torchlens/_io/rehydrate.py", "_REHYDRATE_KINDS"),
        ("torchlens/_io/runnable.py", "_DATACLASS_FIELD_NAMES"),
        ("torchlens/_io/runnable.py", "_SPARSE_CORE_NODE_KINDS"),
        ("torchlens/_io/runnable.py", "_TORCH_SYMBOL_NAMES"),
        ("torchlens/_io/runnable.py", "_TORCH_SYMBOL_NAMESPACE_SIZE"),
        ("torchlens/_io/scrub.py", "_SCRUB_VALUE_KINDS"),
        ("torchlens/_io/state_keys.py", "_CACHE_GENERATION"),
        # Per-class static-attr / shadow-verdict memos (fixwave-5): pure
        # memoization of MRO walks, fingerprint-validated per load generation;
        # weakly keyed by the class (rows also in _WEAKLY_HELD).
        ("torchlens/_io/state_keys.py", "_SHADOW_VERDICT_MEMO"),
        ("torchlens/_io/state_keys.py", "_STATIC_ATTR_MEMO"),
        ("torchlens/_state.py", "_arg_names"),
        ("torchlens/_state.py", "_dir_cache"),
        ("torchlens/_state.py", "_dynamic_arg_specs"),
        ("torchlens/_state.py", "_naming_counters"),
        ("torchlens/_training_validation.py", "_NON_GRAD_DTYPES"),
        ("torchlens/backends/torch/backward.py", "_BACKWARD_GRAD_FN_REGISTRY"),
        # Model-state root-matching boundaries, same lifecycle as the registry
        # above: evicted by the owning trace's slot callback and by cleanup().
        ("torchlens/backends/torch/tensor_tracking.py", "_STATE_BOUNDARY_GRAD_FNS"),
        ("torchlens/backends/torch/completeness_witness.py", "_FRAMEWORK_FILENAME_VERDICTS"),
        ("torchlens/backends/torch/model_prep.py", "_module_class_metadata_cache"),
        ("torchlens/backends/torch/ops.py", "_CAPTURE_PRODUCER_POLICIES"),
        # Agent-surface digest-hint cache (moved from torchlens/bridge/mcp.py
        # in the 2026-10 agent-reference reorg): full stat identity (path,
        # dev, ino, size, mtime_ns) -> content digest, FIFO-evicted at 16. A
        # VERIFIED hint only -- any identity drift re-hashes, so a stale row
        # can never serve a wrong digest; holds only small tuples and hex
        # strings.
        ("torchlens/agent/_artifacts.py", "_DIGEST_HINTS"),
        # Sibling of _DIGEST_HINTS: the same verified-hint pattern over
        # artifact metadata rather than content digests, same stat-identity
        # key and FIFO eviction bound.
        ("torchlens/agent/_artifacts.py", "_METADATA_HINTS"),
        # Agent-surface reload cache for saved .tlspec artifacts (moved from
        # torchlens/bridge/mcp.py): mtime-keyed, FIFO-evicted at 4. Holds
        # traces STRONGLY, so NOT a _WEAK_SUBJECT_TABLES row -- the bound,
        # not weakness, is what keeps it finite; staleness is the mtime's job.
        ("torchlens/agent/_artifacts.py", "_TRACE_CACHE"),
        ("torchlens/capture/arg_positions.py", "FUNC_ARG_SPECS"),
        ("torchlens/capture/projections.py", "_CAPTURE_POLICY_CACHE"),
        ("torchlens/capture/projectors.py", "_REFRESH_SOURCES"),
        ("torchlens/capture/salient_args.py", "_EXTRACTORS"),
        ("torchlens/constants.py", "_TORCHVISION_FUNCS_CACHE"),
        # Lazy once-per-process snapshot of a bare nn.Module()'s instance
        # attributes (the capture-cache attr filter's torch-internal
        # exclusion set, r3 R39-1). Clearing only re-derives from the running
        # torch build; a pure memo, not capability state.
        ("torchlens/_capture_state_helpers.py", "_MODULE_BASELINE_INSTANCE_ATTRS"),
        ("torchlens/data_classes/op.py", "_RELATION_CELL_ENCODINGS"),
        ("torchlens/data_classes/trace.py", "_MODEL_LOG_DEFAULT_FILL"),
        ("torchlens/partial/__init__.py", "_FAILED_CAPTURE_REGISTRY"),
        ("torchlens/postprocess/ast_branches.py", "_file_cache"),
        ("torchlens/receptive_field/_engine.py", "_SCHEMA_OPERAND_SLOTS_CACHE"),
        # Bounded FIFO negative companion of the slots cache (grind-r3
        # T-CACHES): known-miss func names that must not re-enter the C++
        # operator registry per (edge, arg) on every RF solve.
        ("torchlens/receptive_field/_engine.py", "_SCHEMA_OPERAND_MISS_NAMES"),
        ("torchlens/utils/introspection.py", "_COL_OFFSET_CACHE"),
        # Import-time derived ULP tolerance table, lazily extended for dtypes
        # outside _REPLAY_ULP_HEADROOM; clearing only re-derives (pure finfo
        # arithmetic), so it is a memo, not capability state.
        ("torchlens/utils/tensor_utils.py", "_DTYPE_FLOAT_TOLERANCES"),
        # Adaptive defer-registry prune watermark (doubles away from the live
        # population; resets down after a mostly-dead sweep). Clearing it back
        # to the threshold only costs extra sweeps, never correctness.
        ("torchlens/utils/tensor_utils.py", "_defer_prune_watermark"),
        # Companion dead-alias counter (r3 bounds-gap fix): weakref-callback
        # tally of registered alias deaths since the last sweep; crossing the
        # threshold forces a sweep a stuck-high watermark would suppress.
        # Clearing it only delays one sweep, never correctness.
        ("torchlens/utils/tensor_utils.py", "_defer_dead_alias_count"),
        # Fork-inheritance discriminator for warn_parallel (r-b6 R40-3b): the
        # PID that first observed an initialized process group. Clearing it
        # only re-stamps on the next capture entry; it never steers anything
        # but the child-process refusal.
        ("torchlens/utils/display.py", "_DIST_GROUP_OBSERVED_PID"),
        # Live viewer child handles (r-b6 R40-1): retained solely so each
        # launch can reap already-exited viewers. Clearing it costs at most
        # one unreaped zombie per cleared entry until process exit — hygiene,
        # never correctness.
        ("torchlens/visualization/_render_utils.py", "_VIEWER_PROCS"),
    }
)
"""Process-lifetime memos holding strong references.

Correctness-neutral (a cleared cache only costs time) but they are the class
that PINS objects, so each strong key/value must be justified: prefer a weak
table when the key is a user object, and clear the cache in the capture epilogue
when its entries are session-scoped (see ``_module_class_metadata_cache``).
"""


_WEAKLY_HELD = frozenset(
    {
        ("torchlens/_capture_state_helpers.py", "_VALIDATION_DEEPCOPY_WARNING_TYPES"),
        # Weightsfree (F33) session registries (storage fact: WeakKeyDictionary /
        # WeakSet keyed by the owning Trace; lifecycle class is
        # _WEAK_SUBJECT_TABLES).
        ("torchlens/capture/_weightsfree_admission.py", "_ADMISSIONS"),
        ("torchlens/capture/_weightsfree_admission.py", "_META_ACTIVE"),
        ("torchlens/capture/_weightsfree_admission.py", "_WRAP_GENERATIONS"),
        # Kind memos re-keyed weakly by the value TYPE so dynamically created
        # classes stay collectable; lifecycle class stays _PROCESS_CACHES (the
        # ledger is orthogonal: weakness is a storage fact).
        ("torchlens/_io/bundle.py", "_NESTED_BLOB_KINDS"),
        ("torchlens/_io/rehydrate.py", "_REHYDRATE_KINDS"),
        ("torchlens/_io/runnable.py", "_DATACLASS_FIELD_NAMES"),
        ("torchlens/_io/runnable.py", "_SPARSE_CORE_NODE_KINDS"),
        ("torchlens/_io/scrub.py", "_SCRUB_VALUE_KINDS"),
        # Shadow-verdict and static-attr memos re-keyed weakly by TYPE
        # (WeakKeyDictionary) in fixwave-5 concmem so dynamic classes stay
        # collectable; verified weakref.WeakKeyDictionary initializers.
        ("torchlens/_io/state_keys.py", "_SHADOW_VERDICT_MEMO"),
        ("torchlens/_io/state_keys.py", "_STATIC_ATTR_MEMO"),
        ("torchlens/_state.py", "_log_registry"),
        ("torchlens/_state.py", "_prepared_models"),
        ("torchlens/_state.py", "_prepared_root_by_module"),
        ("torchlens/_state.py", "_stale_prepared_roots"),
        # Released-model registry became a WeakSet in fixwave-5 concmem: a
        # release registration must never pin the released model itself.
        ("torchlens/backends/torch/_held_refs.py", "_RELEASED_MODELS"),
        ("torchlens/backends/torch/backward.py", "_BACKWARD_TRACE_SLOTS"),
        ("torchlens/backends/torch/backward.py", "_CHECKPOINT_TOKEN_STATE"),
        ("torchlens/backends/torch/_fire_timing.py", "_FIRE_TIMING_STAMPS"),
        ("torchlens/backends/torch/_gradfn_markers.py", "_GRADFN_MARKER_TOKENS"),
        ("torchlens/observability/_native_profile.py", "_JOIN_TRACES"),
        # Post-T52 census sweep (F27): weak-keyed caches other merged lanes
        # added without classifying -- keyed by the owning trace/bundle
        # member, they die with their subject.
        ("torchlens/attribution/onebackward/_accessor.py", "_INDEX_CACHE"),
        ("torchlens/attribution/onebackward/_edge_plumbing.py", "_PASS_STAMPS"),
        ("torchlens/bundle/_compare_gate.py", "_INPUT_VALUE_DIGEST_CACHE"),
        ("torchlens/backends/torch/backward.py", "_PENDING_BACKWARD_FINALIZE"),
        ("torchlens/backends/torch/buffer_writes.py", "_PARAM_BYTE_WITNESS_NOT_ARMED"),
        ("torchlens/backends/torch/completeness_witness.py", "_ALIAS_MUTATION_CANDIDATE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_CAPTURED_STORAGE_PTRS"),
        ("torchlens/backends/torch/completeness_witness.py", "_DATA_ALIAS_MUTATION_TRACES"),
        ("torchlens/backends/torch/completeness_witness.py", "_DISPATCH_TENSOR_ORIGINS"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_BOOL_CONSUMER_LOCATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_BOOL_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_CROSS_THREAD_CAPTURED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_LABEL_LEAF_ORIGINS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_MUTABLE_WRITEBACK"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_OBSERVER_FAILED"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_RAW_POINTER"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_SOURCE_LABELS"),
        (
            "torchlens/backends/torch/completeness_witness.py",
            "_HOST_ESCAPE_STATE_METADATA_OBSERVATIONS",
        ),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_METADATA_READS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_STATE_SOURCE_NAMES"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_BOOL"),
        ("torchlens/backends/torch/completeness_witness.py", "_HOST_ESCAPE_UNATTRIBUTABLE_OPAQUE"),
        ("torchlens/backends/torch/completeness_witness.py", "_INPUT_METADATA_VIEW_READ"),
        ("torchlens/backends/torch/completeness_witness.py", "_LAYOUT_ANCESTRY_CLEAN"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_ALIAS_MUTATION_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_PRUNED_RNG_CONTROL_LABELS"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_INPUT_STORAGE_SITES"),
        ("torchlens/backends/torch/completeness_witness.py", "_RUNNABLE_LEDGER_FACTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_STATE_METADATA_FACTS"),
        ("torchlens/backends/torch/completeness_witness.py", "_STORAGE_REBIND_BARRIER_LABELS"),
        ("torchlens/backends/torch/model_prep.py", "_source_line_cache"),
        ("torchlens/backends/torch/tensor_tracking.py", "_IMPLICIT_BACKWARD_TASK_IDS"),
        ("torchlens/backends/torch/tensor_tracking.py", "_OWNED_STATE_GRAD_HOOK_HANDLES"),
        ("torchlens/backends/torch/wrappers.py", "_COW_STATE_PTRS_CACHE"),
        ("torchlens/capture/structure_only.py", "_DISCHARGE_REGISTRY"),
        ("torchlens/data_classes/_compaction.py", "_COMPACTED_TRACES"),
        ("torchlens/data_classes/_nonfinite.py", "_MEMOS"),
        ("torchlens/data_classes/grad_fn_call.py", "_ORDINAL_POSITIONS_CACHE"),
        ("torchlens/data_classes/param.py", "_ORDINAL_INDEX_CACHE"),
        ("torchlens/kernel_telemetry.py", "_ATEN_KERNELS"),
        ("torchlens/partial/__init__.py", "_FAILED_CAPTURE_RESULTS"),
        ("torchlens/visualization/auto_collapse.py", "_ANALYSIS_CACHE"),
        ("torchlens/visualization/auto_collapse.py", "_OP_ADJACENCY_INDEX_CACHE"),
        ("torchlens/visualization/code_panel.py", "_SOURCE_MEMO"),
        ("torchlens/visualization/collapse_optimizer.py", "_BOX_UNITS_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_CEILING_WARNED_TRACES"),
        ("torchlens/visualization/_collapse_disclosures.py", "_BUDGET_WARNED_TRACES"),
        ("torchlens/visualization/collapse_optimizer.py", "_RESULT_CACHE"),
        ("torchlens/visualization/collapse_optimizer.py", "_SCHEDULE_CACHE"),
        ("torchlens/trackers/_watch.py", "_ACTIVE_SESSIONS"),
    }
)
"""Every inventory member whose container is bound WEAKLY, across all classes.

Orthogonal to the lifecycle classes on purpose: weakness is a storage fact, and
``_prepared_models`` is both a weak registry AND install state, while
``_VALIDATION_DEEPCOPY_WARNING_TYPES`` is both weak AND a warn-once sentinel.
Freezing the weak set separately means a member that silently changes from
``WeakKeyDictionary()`` to ``{}`` -- and so starts pinning its whole subject
graph -- fails a gate instead of hiding behind a reassuring lifecycle row.
"""


"""Every package module's own ``__all__`` export-list declaration.

``torchlens/neuro/__init__.py`` keeps its ``__all__`` alphabetized with one
``__all__.sort()`` call; the package-wide in-place-mutation detector matches
BY BARE NAME (documented tradeoff in ``_names_mutated_in_place``), so that one
call sweeps every file's own ``__all__`` binding into the observed-state
census, not just neuro's. Each row here is exactly that binding: a static,
hand-written list of public names, never mutated at its own module -- the
single acceptable row to add per the detector's over-inclusion design, done
once for the whole package rather than left as 471 individually-surprising
failures.
"""


_STATIC_EXPORT_LISTS = frozenset(
    {
        ("torchlens/__init__.py", "__all__"),
        ("torchlens/_capture_honesty.py", "__all__"),
        ("torchlens/_data_substrate/__init__.py", "__all__"),
        ("torchlens/_data_substrate/artifact.py", "__all__"),
        ("torchlens/_data_substrate/digests.py", "__all__"),
        ("torchlens/_data_substrate/model_identity.py", "__all__"),
        ("torchlens/_deploy_env.py", "__all__"),
        ("torchlens/_distributed.py", "__all__"),
        ("torchlens/_episode_spec.py", "__all__"),
        ("torchlens/_extraction/__init__.py", "__all__"),
        ("torchlens/_extraction/callable_identity.py", "__all__"),
        ("torchlens/_extraction/context.py", "__all__"),
        ("torchlens/_extraction/dtype_policy.py", "__all__"),
        ("torchlens/_extraction/envelope.py", "__all__"),
        ("torchlens/_extraction/export.py", "__all__"),
        ("torchlens/_extraction/harvest_schema.py", "__all__"),
        ("torchlens/_extraction/pool.py", "__all__"),
        ("torchlens/_extraction/ragged.py", "__all__"),
        ("torchlens/_extraction/reader.py", "__all__"),
        ("torchlens/_extraction/selector_plan.py", "__all__"),
        ("torchlens/_extraction/shards.py", "__all__"),
        ("torchlens/_extraction/views.py", "__all__"),
        ("torchlens/_extraction_provenance.py", "__all__"),
        ("torchlens/_input_walk.py", "__all__"),
        ("torchlens/_io/__init__.py", "__all__"),
        ("torchlens/_io/_artifact_anchors.py", "__all__"),
        ("torchlens/_io/_canonical_pickle.py", "__all__"),
        ("torchlens/_io/_durability.py", "__all__"),
        ("torchlens/_io/_portability_preflight.py", "__all__"),
        ("torchlens/_io/_safe_unpickle.py", "__all__"),
        ("torchlens/_io/_torch_symbols.py", "__all__"),
        ("torchlens/_io/injection_codec.py", "__all__"),
        ("torchlens/_io/payload_reader.py", "__all__"),
        ("torchlens/_io/runnable.py", "__all__"),
        ("torchlens/_io/runnable_load.py", "__all__"),
        ("torchlens/_io/tlspec.py", "__all__"),
        ("torchlens/_model_wrappers.py", "__all__"),
        ("torchlens/_registry/__init__.py", "__all__"),
        ("torchlens/_runnable_execution.py", "__all__"),
        ("torchlens/_runnable_seam.py", "__all__"),
        ("torchlens/_runnable_state.py", "__all__"),
        ("torchlens/_save_budget.py", "__all__"),
        ("torchlens/_selection_align.py", "__all__"),
        ("torchlens/_source_links.py", "__all__"),
        ("torchlens/_trace_core/__init__.py", "__all__"),
        ("torchlens/_trace_state.py", "__all__"),
        ("torchlens/_vocab/trace_state.py", "__all__"),
        ("torchlens/accessors/__init__.py", "__all__"),
        ("torchlens/agent/__init__.py", "__all__"),
        ("torchlens/attribution/__init__.py", "__all__"),
        ("torchlens/attribution/_binder.py", "__all__"),
        ("torchlens/attribution/_core.py", "__all__"),
        ("torchlens/attribution/_gradient_shap.py", "__all__"),
        ("torchlens/attribution/_guided.py", "__all__"),
        ("torchlens/attribution/_metrics.py", "__all__"),
        ("torchlens/attribution/_noise_tunnel.py", "__all__"),
        ("torchlens/attribution/_occlusion.py", "__all__"),
        ("torchlens/attribution/_result.py", "__all__"),
        ("torchlens/attribution/_sampling.py", "__all__"),
        ("torchlens/attribution/_stash.py", "__all__"),
        ("torchlens/attribution/_steps.py", "__all__"),
        ("torchlens/attribution/_text.py", "__all__"),
        ("torchlens/attribution/_viz.py", "__all__"),
        ("torchlens/attribution/onebackward/__init__.py", "__all__"),
        ("torchlens/attribution/onebackward/_accessor.py", "__all__"),
        ("torchlens/attribution/onebackward/_edge_plumbing.py", "__all__"),
        ("torchlens/attribution/onebackward/_engine.py", "__all__"),
        ("torchlens/attribution/onebackward/_errors.py", "__all__"),
        ("torchlens/attribution/onebackward/_facet_bridge.py", "__all__"),
        ("torchlens/attribution/onebackward/_frozen.py", "__all__"),
        ("torchlens/attribution/onebackward/_read.py", "__all__"),
        ("torchlens/attribution/onebackward/_suppress.py", "__all__"),
        ("torchlens/attribution/onebackward/_table.py", "__all__"),
        ("torchlens/attribution/onebackward/_targets.py", "__all__"),
        ("torchlens/autoroute/__init__.py", "__all__"),
        ("torchlens/autoroute/input/__init__.py", "__all__"),
        ("torchlens/autoroute/output/__init__.py", "__all__"),
        ("torchlens/backends/__init__.py", "__all__"),
        ("torchlens/backends/_protocol.py", "__all__"),
        ("torchlens/backends/jax/__init__.py", "__all__"),
        ("torchlens/backends/jax/capabilities.py", "__all__"),
        ("torchlens/backends/mlx/__init__.py", "__all__"),
        ("torchlens/backends/mlx/backend.py", "__all__"),
        ("torchlens/backends/mlx/capabilities.py", "__all__"),
        ("torchlens/backends/mlx/interventions.py", "__all__"),
        ("torchlens/backends/mlx/tensor_store.py", "__all__"),
        ("torchlens/backends/mlx/wrappers.py", "__all__"),
        ("torchlens/backends/paddle/__init__.py", "__all__"),
        ("torchlens/backends/paddle/_param_writes.py", "__all__"),
        ("torchlens/backends/paddle/backend.py", "__all__"),
        ("torchlens/backends/paddle/capabilities.py", "__all__"),
        ("torchlens/backends/paddle/interventions.py", "__all__"),
        ("torchlens/backends/paddle/tensor_store.py", "__all__"),
        ("torchlens/backends/paddle/validation.py", "__all__"),
        ("torchlens/backends/paddle/wrappers.py", "__all__"),
        ("torchlens/backends/tf/__init__.py", "__all__"),
        ("torchlens/backends/tf/_tf_compat.py", "__all__"),
        ("torchlens/backends/tf/capabilities.py", "__all__"),
        ("torchlens/backends/tf/derived_grads.py", "__all__"),
        ("torchlens/backends/tf/interventions.py", "__all__"),
        ("torchlens/backends/tf/validation.py", "__all__"),
        ("torchlens/backends/tinygrad/__init__.py", "__all__"),
        ("torchlens/backends/tinygrad/capabilities.py", "__all__"),
        ("torchlens/backends/torch/__init__.py", "__all__"),
        ("torchlens/backends/torch/_fire_timing.py", "__all__"),
        ("torchlens/backends/torch/_gradfn_markers.py", "__all__"),
        ("torchlens/backends/torch/_held_refs.py", "__all__"),
        ("torchlens/backends/torch/_held_refs_capture.py", "__all__"),
        ("torchlens/backends/torch/_op_markers.py", "__all__"),
        ("torchlens/backends/torch/_tl.py", "__all__"),
        ("torchlens/backends/torch/_weightsfree_ctx.py", "__all__"),
        ("torchlens/backends/torch/_weightsfree_transparency.py", "__all__"),
        ("torchlens/backends/torch/backend.py", "__all__"),
        ("torchlens/backends/torch/belt.py", "__all__"),
        ("torchlens/backends/torch/collectives.py", "__all__"),
        ("torchlens/backends/torch/funcol.py", "__all__"),
        ("torchlens/backends/torch/identity_shims.py", "__all__"),
        ("torchlens/backends/torch/legacy_ctors.py", "__all__"),
        ("torchlens/backends/torch/offload_hooks.py", "__all__"),
        ("torchlens/backends/torch/rescue.py", "__all__"),
        ("torchlens/brainpipe.py", "__all__"),
        ("torchlens/bridge/__init__.py", "__all__"),
        ("torchlens/bridge/_contrastive.py", "__all__"),
        ("torchlens/bridge/_utils.py", "__all__"),
        ("torchlens/bridge/brain_score.py", "__all__"),
        ("torchlens/bridge/captum.py", "__all__"),
        ("torchlens/bridge/depyf.py", "__all__"),
        ("torchlens/bridge/dialz.py", "__all__"),
        ("torchlens/bridge/gradcam.py", "__all__"),
        ("torchlens/bridge/hf.py", "__all__"),
        ("torchlens/bridge/huggingface.py", "__all__"),
        ("torchlens/bridge/inseq.py", "__all__"),
        ("torchlens/bridge/lit/__init__.py", "__all__"),
        ("torchlens/bridge/nnsight.py", "__all__"),
        ("torchlens/bridge/profiler.py", "__all__"),
        ("torchlens/bridge/repeng.py", "__all__"),
        ("torchlens/bridge/rsatoolbox.py", "__all__"),
        ("torchlens/bridge/sae.py", "__all__"),
        ("torchlens/bridge/sae_lens.py", "__all__"),
        ("torchlens/bridge/shap.py", "__all__"),
        ("torchlens/bridge/steering_vectors.py", "__all__"),
        ("torchlens/bridge/treescope.py", "__all__"),
        ("torchlens/bridge/xarray.py", "__all__"),
        ("torchlens/bundle/__init__.py", "__all__"),
        ("torchlens/bundle/_bytes.py", "__all__"),
        ("torchlens/bundle/_lineage.py", "__all__"),
        ("torchlens/bundle/_provenance.py", "__all__"),
        ("torchlens/bundle/_relations.py", "__all__"),
        ("torchlens/bundle/_vary.py", "__all__"),
        ("torchlens/callbacks/__init__.py", "__all__"),
        ("torchlens/callbacks/lightning.py", "__all__"),
        ("torchlens/capture/_annotations_travel.py", "__all__"),
        ("torchlens/capture/_episode_coupling.py", "__all__"),
        ("torchlens/capture/_episode_derivation.py", "__all__"),
        ("torchlens/capture/_episode_failed.py", "__all__"),
        ("torchlens/capture/_episode_fold.py", "__all__"),
        ("torchlens/capture/_episode_join.py", "__all__"),
        ("torchlens/capture/_episode_ledger.py", "__all__"),
        ("torchlens/capture/_episode_ledger_anchors.py", "__all__"),
        ("torchlens/capture/_plan_check.py", "__all__"),
        ("torchlens/capture/_structure_evidence.py", "__all__"),
        ("torchlens/capture/_structure_only_bridge.py", "__all__"),
        ("torchlens/capture/_weightsfree_admission.py", "__all__"),
        ("torchlens/capture/outcome.py", "__all__"),
        ("torchlens/capture/preflight.py", "__all__"),
        ("torchlens/capture/structure_only.py", "__all__"),
        ("torchlens/checks/__init__.py", "__all__"),
        ("torchlens/checks/_adapters.py", "__all__"),
        ("torchlens/checks/_audit.py", "__all__"),
        ("torchlens/checks/_constants.py", "__all__"),
        ("torchlens/checks/_errors.py", "__all__"),
        ("torchlens/checks/_ledgers.py", "__all__"),
        ("torchlens/checks/_param_checks.py", "__all__"),
        ("torchlens/checks/_range.py", "__all__"),
        ("torchlens/checks/_records.py", "__all__"),
        ("torchlens/checks/_scan.py", "__all__"),
        ("torchlens/checks/_session.py", "__all__"),
        ("torchlens/compat/__init__.py", "__all__"),
        ("torchlens/compat/_report.py", "__all__"),
        ("torchlens/compat/lovely.py", "__all__"),
        ("torchlens/compat/torchextractor.py", "__all__"),
        ("torchlens/compat/torchshow.py", "__all__"),
        ("torchlens/conformance/__init__.py", "__all__"),
        ("torchlens/data_classes/__init__.py", "__all__"),
        ("torchlens/data_classes/_trace_intervention.py", "__all__"),
        ("torchlens/data_classes/_trace_stack.py", "__all__"),
        ("torchlens/data_classes/aten_op.py", "__all__"),
        ("torchlens/data_classes/container.py", "__all__"),
        ("torchlens/dataset_extraction.py", "__all__"),
        ("torchlens/debug/__init__.py", "__all__"),
        ("torchlens/debug/_compile_counter.py", "__all__"),
        ("torchlens/debug/_determinism.py", "__all__"),
        ("torchlens/debug/_dtype_range.py", "__all__"),
        ("torchlens/debug/_first_bad.py", "__all__"),
        ("torchlens/debug/_flops_vs_dispatch.py", "__all__"),
        ("torchlens/debug/_grad_fn_walk.py", "__all__"),
        ("torchlens/debug/_graph_breaks.py", "__all__"),
        ("torchlens/debug/_nan_backward.py", "__all__"),
        ("torchlens/debug/_params_audit.py", "__all__"),
        ("torchlens/debug/_rerun.py", "__all__"),
        ("torchlens/differential.py", "__all__"),
        ("torchlens/distributed/__init__.py", "__all__"),
        ("torchlens/distributed/_audit.py", "__all__"),
        ("torchlens/distributed/_dtensor.py", "__all__"),
        ("torchlens/distributed/_ledger.py", "__all__"),
        ("torchlens/distributed/_lifecycle.py", "__all__"),
        ("torchlens/distributed/_recognizer.py", "__all__"),
        ("torchlens/ecosystem/__init__.py", "__all__"),
        ("torchlens/errors/__init__.py", "__all__"),
        ("torchlens/errors/_base.py", "__all__"),
        ("torchlens/errors/episode.py", "__all__"),
        ("torchlens/errors/runnable.py", "__all__"),
        ("torchlens/examples/__init__.py", "__all__"),
        ("torchlens/experiment/__init__.py", "__all__"),
        ("torchlens/experiment/_engine.py", "__all__"),
        ("torchlens/experiment/_head_sugar.py", "__all__"),
        ("torchlens/experiment/_ledger.py", "__all__"),
        ("torchlens/experiment/_mcp.py", "__all__"),
        ("torchlens/experimental/__init__.py", "__all__"),
        ("torchlens/experimental/dagua/__init__.py", "__all__"),
        ("torchlens/experimental/node_styles.py", "__all__"),
        ("torchlens/export/__init__.py", "__all__"),
        ("torchlens/export/_graphs.py", "__all__"),
        ("torchlens/export/_model_explorer/__init__.py", "__all__"),
        ("torchlens/export/_model_explorer/_build.py", "__all__"),
        ("torchlens/export/_netron_emit.py", "__all__"),
        ("torchlens/export/_report.py", "__all__"),
        ("torchlens/fastlog/__init__.py", "__all__"),
        ("torchlens/fastlog/types.py", "__all__"),
        ("torchlens/features.py", "__all__"),
        ("torchlens/hash.py", "__all__"),
        ("torchlens/intervention/__init__.py", "__all__"),
        ("torchlens/intervention/_super/__init__.py", "__all__"),
        ("torchlens/intervention/_super/super_logs.py", "__all__"),
        ("torchlens/intervention/_super/super_op.py", "__all__"),
        ("torchlens/intervention/_topology/__init__.py", "__all__"),
        ("torchlens/intervention/audit.py", "__all__"),
        ("torchlens/intervention/binding.py", "__all__"),
        ("torchlens/intervention/bundle.py", "__all__"),
        ("torchlens/intervention/compose.py", "__all__"),
        ("torchlens/intervention/edge_substitution.py", "__all__"),
        ("torchlens/intervention/errors.py", "__all__"),
        ("torchlens/intervention/handles.py", "__all__"),
        ("torchlens/intervention/helpers.py", "__all__"),
        ("torchlens/intervention/hooks.py", "__all__"),
        ("torchlens/intervention/injection.py", "__all__"),
        ("torchlens/intervention/masked_edit.py", "__all__"),
        ("torchlens/intervention/model_door.py", "__all__"),
        ("torchlens/intervention/param_substitution.py", "__all__"),
        ("torchlens/intervention/population.py", "__all__"),
        ("torchlens/intervention/predicates.py", "__all__"),
        ("torchlens/intervention/regions.py", "__all__"),
        ("torchlens/intervention/replay.py", "__all__"),
        ("torchlens/intervention/rerun.py", "__all__"),
        ("torchlens/intervention/resolver.py", "__all__"),
        ("torchlens/intervention/runtime.py", "__all__"),
        ("torchlens/intervention/save.py", "__all__"),
        ("torchlens/intervention/selectors.py", "__all__"),
        ("torchlens/intervention/site_keys.py", "__all__"),
        ("torchlens/intervention/sites.py", "__all__"),
        ("torchlens/intervention/spec.py", "__all__"),
        ("torchlens/intervention/spec_compat.py", "__all__"),
        ("torchlens/intervention/steering.py", "__all__"),
        ("torchlens/intervention/stochastic.py", "__all__"),
        ("torchlens/intervention/sweep.py", "__all__"),
        ("torchlens/intervention/types.py", "__all__"),
        ("torchlens/inventory.py", "__all__"),
        ("torchlens/io/__init__.py", "__all__"),
        ("torchlens/ir/__init__.py", "__all__"),
        ("torchlens/ir/container.py", "__all__"),
        ("torchlens/ir/predicate_registry.py", "__all__"),
        ("torchlens/ir/selector_eval.py", "__all__"),
        ("torchlens/ir/summary_role.py", "__all__"),
        ("torchlens/kernel_telemetry.py", "__all__"),
        ("torchlens/mechinterp/__init__.py", "__all__"),
        ("torchlens/mechinterp/_aliases.py", "__all__"),
        ("torchlens/mechinterp/_anchors.py", "__all__"),
        ("torchlens/mechinterp/_compose.py", "__all__"),
        ("torchlens/mechinterp/_dla.py", "__all__"),
        ("torchlens/mechinterp/_errors.py", "__all__"),
        ("torchlens/mechinterp/_grids.py", "__all__"),
        ("torchlens/mechinterp/_heads.py", "__all__"),
        ("torchlens/mechinterp/_lowered.py", "__all__"),
        ("torchlens/mechinterp/_params.py", "__all__"),
        ("torchlens/mechinterp/_prompts.py", "__all__"),
        ("torchlens/mechinterp/_records.py", "__all__"),
        ("torchlens/mechinterp/_residual.py", "__all__"),
        ("torchlens/mechinterp/_retention.py", "__all__"),
        ("torchlens/mechinterp/_scores.py", "__all__"),
        ("torchlens/mechinterp/_walk.py", "__all__"),
        ("torchlens/merged/__init__.py", "__all__"),
        ("torchlens/merged/_artifact.py", "__all__"),
        ("torchlens/merged/_engine.py", "__all__"),
        ("torchlens/merged/_enums.py", "__all__"),
        ("torchlens/merged/_errors.py", "__all__"),
        ("torchlens/merged/_evidence.py", "__all__"),
        ("torchlens/merged/_presenter.py", "__all__"),
        ("torchlens/neuro/_datasets.py", "__all__"),
        ("torchlens/neuro/_rdms.py", "__all__"),
        ("torchlens/notebook/__init__.py", "__all__"),
        ("torchlens/notebook/_access.py", "__all__"),
        ("torchlens/notebook/_axis_labels.py", "__all__"),
        ("torchlens/notebook/_grid.py", "__all__"),
        ("torchlens/notebook/_motifs.py", "__all__"),
        ("torchlens/notebook/_truncation.py", "__all__"),
        ("torchlens/notebook/cards.py", "__all__"),
        ("torchlens/notebook/cardtree.py", "__all__"),
        ("torchlens/notebook/frontier.py", "__all__"),
        ("torchlens/observability/__init__.py", "__all__"),
        ("torchlens/observability/_artifact.py", "__all__"),
        ("torchlens/observability/_chassis.py", "__all__"),
        ("torchlens/observability/_collector.py", "__all__"),
        ("torchlens/observability/_errors.py", "__all__"),
        ("torchlens/observability/_join.py", "__all__"),
        ("torchlens/observability/_kernels.py", "__all__"),
        ("torchlens/observability/_kineto.py", "__all__"),
        ("torchlens/observability/_memory_parity.py", "__all__"),
        ("torchlens/observability/_native_profile.py", "__all__"),
        ("torchlens/observability/_op_tier.py", "__all__"),
        ("torchlens/observability/_overhead.py", "__all__"),
        ("torchlens/observability/_payloads.py", "__all__"),
        ("torchlens/observability/_perf_harness.py", "__all__"),
        ("torchlens/observability/_quantiles.py", "__all__"),
        ("torchlens/observability/_region.py", "__all__"),
        ("torchlens/observability/_render.py", "__all__"),
        ("torchlens/observability/_schema.py", "__all__"),
        ("torchlens/observability/_session.py", "__all__"),
        ("torchlens/observability/_spans.py", "__all__"),
        ("torchlens/observe/__init__.py", "__all__"),
        ("torchlens/observe/_device_memory.py", "__all__"),
        ("torchlens/observe/_liveness.py", "__all__"),
        ("torchlens/observe/_peaks.py", "__all__"),
        ("torchlens/observe/_svg.py", "__all__"),
        ("torchlens/observe/_timeline.py", "__all__"),
        ("torchlens/observers.py", "__all__"),
        ("torchlens/options.py", "__all__"),
        ("torchlens/partial/__init__.py", "__all__"),
        ("torchlens/postprocess/__init__.py", "__all__"),
        ("torchlens/postprocess/_primitive_profile.py", "__all__"),
        ("torchlens/postprocess/_site_key.py", "__all__"),
        ("torchlens/postprocess/ast_branches.py", "__all__"),
        ("torchlens/postprocess/loop_grouping_adapter.py", "__all__"),
        ("torchlens/preprocessing/__init__.py", "__all__"),
        ("torchlens/quickstart/__init__.py", "__all__"),
        ("torchlens/receptive_field/__init__.py", "__all__"),
        ("torchlens/receptive_field/_engine.py", "__all__"),
        ("torchlens/receptive_field/_engine_descriptor.py", "__all__"),
        ("torchlens/receptive_field/_engine_forward.py", "__all__"),
        ("torchlens/receptive_field/_engine_geometry.py", "__all__"),
        ("torchlens/receptive_field/_errors.py", "__all__"),
        ("torchlens/receptive_field/_forward_query.py", "__all__"),
        ("torchlens/receptive_field/_gradient.py", "__all__"),
        ("torchlens/receptive_field/_path.py", "__all__"),
        ("torchlens/receptive_field/_query.py", "__all__"),
        ("torchlens/receptive_field/_rules.py", "__all__"),
        ("torchlens/receptive_field/_types.py", "__all__"),
        ("torchlens/receptive_field/_validation.py", "__all__"),
        ("torchlens/receptive_field/_view.py", "__all__"),
        ("torchlens/receptive_field/_viz.py", "__all__"),
        ("torchlens/receptive_field/rules/__init__.py", "__all__"),
        ("torchlens/repgeom/__init__.py", "__all__"),
        ("torchlens/report/__init__.py", "__all__"),
        ("torchlens/runnable.py", "__all__"),
        ("torchlens/selection.py", "__all__"),
        ("torchlens/selection_compare.py", "__all__"),
        ("torchlens/selection_graph.py", "__all__"),
        ("torchlens/selection_subspace.py", "__all__"),
        ("torchlens/selection_values.py", "__all__"),
        ("torchlens/semantic/__init__.py", "__all__"),
        ("torchlens/semantic/_hops.py", "__all__"),
        ("torchlens/semantic/_norm_reconstruction.py", "__all__"),
        ("torchlens/semantic/coverage.py", "__all__"),
        ("torchlens/semantic/facets.py", "__all__"),
        ("torchlens/semantic/logit_lens.py", "__all__"),
        ("torchlens/semantic/recipes/__init__.py", "__all__"),
        ("torchlens/semantic/tolerances.py", "__all__"),
        ("torchlens/snoop/__init__.py", "__all__"),
        ("torchlens/snoop/_errors.py", "__all__"),
        ("torchlens/snoop/_event.py", "__all__"),
        ("torchlens/snoop/_format.py", "__all__"),
        ("torchlens/snoop/_narrate.py", "__all__"),
        ("torchlens/snoop/_normalize.py", "__all__"),
        ("torchlens/snoop/_session.py", "__all__"),
        ("torchlens/snoop/_sink.py", "__all__"),
        ("torchlens/snoop/_stats.py", "__all__"),
        ("torchlens/stats/__init__.py", "__all__"),
        ("torchlens/stats/_fitted.py", "__all__"),
        ("torchlens/trace_slice.py", "__all__"),
        ("torchlens/trackers/__init__.py", "__all__"),
        ("torchlens/trackers/_amp.py", "__all__"),
        ("torchlens/trackers/_callbacks.py", "__all__"),
        ("torchlens/trackers/_errors.py", "__all__"),
        ("torchlens/trackers/_graph.py", "__all__"),
        ("torchlens/trackers/_protocol.py", "__all__"),
        ("torchlens/trackers/_records.py", "__all__"),
        ("torchlens/trackers/_sinks.py", "__all__"),
        ("torchlens/trackers/_tensorboard.py", "__all__"),
        ("torchlens/trackers/_wandb.py", "__all__"),
        ("torchlens/trackers/_watch.py", "__all__"),
        ("torchlens/transforms/__init__.py", "__all__"),
        ("torchlens/transforms/_coerce.py", "__all__"),
        ("torchlens/transforms/_context.py", "__all__"),
        ("torchlens/transforms/_errors.py", "__all__"),
        ("torchlens/transforms/_helpers.py", "__all__"),
        ("torchlens/transforms/_kernels.py", "__all__"),
        ("torchlens/transforms/_pipeline.py", "__all__"),
        ("torchlens/transforms/_pooling.py", "__all__"),
        ("torchlens/transforms/_projection.py", "__all__"),
        ("torchlens/transforms/_registry.py", "__all__"),
        ("torchlens/transforms/_spec.py", "__all__"),
        ("torchlens/transforms/_srp.py", "__all__"),
        ("torchlens/transforms/_srp_hash.py", "__all__"),
        ("torchlens/tviz/__init__.py", "__all__"),
        ("torchlens/tviz/_bridge.py", "__all__"),
        ("torchlens/tviz/_decomp.py", "__all__"),
        ("torchlens/tviz/_errors.py", "__all__"),
        ("torchlens/tviz/_extract.py", "__all__"),
        ("torchlens/tviz/_metrics.py", "__all__"),
        ("torchlens/tviz/_mpl.py", "__all__"),
        ("torchlens/tviz/_predictions.py", "__all__"),
        ("torchlens/tviz/_receipts.py", "__all__"),
        ("torchlens/tviz/_records.py", "__all__"),
        ("torchlens/tviz/_render_attention.py", "__all__"),
        ("torchlens/tviz/_strip.py", "__all__"),
        ("torchlens/tviz/_wording.py", "__all__"),
        ("torchlens/types.py", "__all__"),
        ("torchlens/utils/__init__.py", "__all__"),
        ("torchlens/utils/_multipass_access.py", "__all__"),
        ("torchlens/utils/_torch_compat.py", "__all__"),
        ("torchlens/utils/env_flags.py", "__all__"),
        ("torchlens/utils/facade.py", "__all__"),
        ("torchlens/utils/lazy_state.py", "__all__"),
        ("torchlens/validation/__init__.py", "__all__"),
        ("torchlens/validation/_invariants_primitive_ops.py", "__all__"),
        ("torchlens/validation/consolidated.py", "__all__"),
        ("torchlens/validation/status.py", "__all__"),
        ("torchlens/visualization/__init__.py", "__all__"),
        ("torchlens/visualization/_backward_inventory.py", "__all__"),
        ("torchlens/visualization/_buffer_visibility.py", "__all__"),
        ("torchlens/visualization/_edge_multiplicity.py", "__all__"),
        ("torchlens/visualization/_geometry_audit.py", "__all__"),
        ("torchlens/visualization/_legend.py", "__all__"),
        ("torchlens/visualization/_mutated_params.py", "__all__"),
        ("torchlens/visualization/_render_common.py", "__all__"),
        ("torchlens/visualization/_render_dot.py", "__all__"),
        ("torchlens/visualization/_render_edges.py", "__all__"),
        ("torchlens/visualization/_render_entrypoints.py", "__all__"),
        ("torchlens/visualization/_render_flow.py", "__all__"),
        ("torchlens/visualization/_render_leaf.py", "__all__"),
        ("torchlens/visualization/_render_nodes.py", "__all__"),
        ("torchlens/visualization/_render_ordering.py", "__all__"),
        ("torchlens/visualization/_render_regions.py", "__all__"),
        ("torchlens/visualization/_summary_internal/__init__.py", "__all__"),
        ("torchlens/visualization/_surgery_diff.py", "__all__"),
        ("torchlens/visualization/_svg_compose.py", "__all__"),
        ("torchlens/visualization/_typography.py", "__all__"),
        ("torchlens/visualization/bundle_diff.py", "__all__"),
        ("torchlens/visualization/lenses/__init__.py", "__all__"),
        ("torchlens/visualization/lenses/_budget.py", "__all__"),
        ("torchlens/visualization/lenses/_families.py", "__all__"),
        ("torchlens/visualization/lenses/_filter.py", "__all__"),
        ("torchlens/visualization/lenses/_nonfinite.py", "__all__"),
        ("torchlens/visualization/lenses/_resolve.py", "__all__"),
        ("torchlens/visualization/lenses/_roster.py", "__all__"),
        ("torchlens/visualization/lenses/audit/__init__.py", "__all__"),
        ("torchlens/visualization/lenses/audit/answer_key.py", "__all__"),
        ("torchlens/visualization/lenses/audit/battery.py", "__all__"),
        ("torchlens/visualization/lenses/audit/corpus.py", "__all__"),
        ("torchlens/visualization/lenses/audit/stage0.py", "__all__"),
        ("torchlens/visualization/node_spec.py", "__all__"),
        ("torchlens/visualization/render_execution.py", "__all__"),
        ("torchlens/visualization/renderers/__init__.py", "__all__"),
        ("torchlens/visualization/surgery_visuals.py", "__all__"),
        ("torchlens/visualization/theme_registry.py", "__all__"),
        ("torchlens/viz/__init__.py", "__all__"),
        ("torchlens/viz/feature_maps.py", "__all__"),
    }
)


_LIFECYCLE_CLASSES = (
    _SCOPED_CAPTURE_STATE,
    _INSTALL_STATE_AND_CACHES,
    _WARN_ONCE_STATE,
    _CAPABILITY_PROBE_STATE,
    _DIAGNOSTIC_AUDIT_STATE,
    _WEAK_SUBJECT_TABLES,
    _PUBLIC_REGISTRATION_STATE,
    _PROCESS_CACHES,
    _STATIC_EXPORT_LISTS,
)
"""Every lifecycle class, in declaration order. The union must be exact."""
