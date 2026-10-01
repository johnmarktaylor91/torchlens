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
        # Refreshed-globals seam: user_funcs rebinds the SAME function objects
        # into the private public-impl module on every access (idempotent).
        ("torchlens/_user_public_impls.py", "_run_model_and_save_specified_outs"),
        ("torchlens/_user_public_impls.py", "trace"),
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
        # Theme lens preset registry-as-data (C05 N12): registered at import
        # by torchlens code, mutated only through register_lens. Ledgered by
        # F11 paying the missed C05 governance row (red at bare tip ca622a77).
        ("torchlens/visualization/theme_registry.py", "_REGISTRY"),
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
        ("torchlens/backends/torch/completeness_witness.py", "_FRAMEWORK_FILENAME_VERDICTS"),
        ("torchlens/backends/torch/model_prep.py", "_module_class_metadata_cache"),
        ("torchlens/backends/torch/ops.py", "_CAPTURE_PRODUCER_POLICIES"),
        # MCP digest-hint cache: full stat identity (path, dev, ino, size,
        # mtime_ns) -> content digest, FIFO-evicted at 16. A VERIFIED hint
        # only -- any identity drift re-hashes, so a stale row can never
        # serve a wrong digest; holds only small tuples and hex strings.
        ("torchlens/bridge/mcp.py", "_DIGEST_HINTS"),
        # MCP reload cache for saved .tlspec artifacts: mtime-keyed, FIFO-evicted at
        # 4. Holds traces STRONGLY, so NOT a _WEAK_SUBJECT_TABLES row -- the bound,
        # not weakness, is what keeps it finite; staleness is the mtime's job.
        ("torchlens/bridge/mcp.py", "_TRACE_CACHE"),
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


_LIFECYCLE_CLASSES = (
    _SCOPED_CAPTURE_STATE,
    _INSTALL_STATE_AND_CACHES,
    _WARN_ONCE_STATE,
    _CAPABILITY_PROBE_STATE,
    _DIAGNOSTIC_AUDIT_STATE,
    _WEAK_SUBJECT_TABLES,
    _PUBLIC_REGISTRATION_STATE,
    _PROCESS_CACHES,
)
"""Every lifecycle class, in declaration order. The union must be exact."""
