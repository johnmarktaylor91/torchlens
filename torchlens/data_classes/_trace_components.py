"""Declared Trace field-to-component ownership map (M10 decomposition).

Every field in ``Trace.FIELD_POLICY`` (declared FIELD_ORDER surface plus
portable/runtime internals) is owned by exactly ONE lifecycle component of
the decomposed Trace (docs/reference/trace_core_design.md section 3.7).
A new Trace field must be added here with an owner, or the lockstep test
fails — the ratchet that keeps the god object from regrowing.

Components: header (identity/lifecycle), capture_config (capture-time
options snapshot), graph (the semantic TraceCore plane, incl. backward),
witness (honesty diagnostics), source_metadata (model source facts),
totals (aggregate counts/memory/timing), runnable (RunnableTraceState),
session (non-portable runtime, caches, append/intervention state, build
workspaces).

Second map: ``TRACE_EXTERNAL_WRITE_EXEMPTIONS`` (b5 R50-1a). Backends,
capture, validation, viz, and I/O still ATTACH private attributes to a Trace
from outside ``data_classes/`` without declaring them anywhere -- neither in
``Trace.FIELD_POLICY`` nor (for most of them) in ``_io/scrub.py``'s
runtime-only allowance. Each such attribute is a field of the god object with
no owner, no policy, and no schema row; SF-39 was one instance fixed by hand
and nothing stopped the next. The exemption ledger below makes the class
CLOSED and reason-bearing: ``tests/test_trace_attr_ownership.py`` fails on any
external private write that is neither declared here nor owned above, so a new
undeclared attachment is a reviewed contract diff. The ledger is SHRINK-ONLY;
the real discharge is enrollment (declare the field on ``Trace`` with a
``FieldPolicy`` and an owner row), which lands in the ``data_classes/trace.py``
owner lane, not here.
"""

from __future__ import annotations

TRACE_COMPONENT_VOCABULARY = frozenset(
    {
        "header",
        "capture_config",
        "graph",
        "witness",
        "source_metadata",
        "totals",
        "runnable",
        "session",
    }
)

TRACE_FIELD_OWNERSHIP: dict[str, str] = {
    "trace_label": "header",
    "model_class_name": "header",
    "model_label": "header",
    "tlspec_version": "header",
    "_tracing_finished": "header",
    "backend": "header",
    "backend_runtime_config": "header",
    "backend_runtime_device_summary": "header",
    "backend_runtime_version": "header",
    "module_identity_mode": "header",
    "param_source": "header",
    "derived_grads": "graph",
    "_primitive_op_profile": "graph",
    # Union additions (2026-08-13 grind): fields declared by the G2/F3a/B1-04
    # fixes now carry owners. Semantic-output scratch + predicate keys are
    # session-lifetime; validation side channels are session; bundle-source
    # provenance rides source_metadata.
    "_output_head": "session",
    "_output_style": "session",
    "_output_tokenizer": "session",
    "_semantic_output_metadata": "session",
    "_last_validation_failure": "session",
    "_validation_diagnostics": "session",
    "_retain_layers_to_save_output_parents": "session",
    "_tl_predicate_intervention_spec_keys": "session",
    "_tl_predicate_intervention_target_keys": "session",
    # F01 log_injections stage 0-1: one consolidated session transient
    # (five loose transients consolidated per the size pressure).
    "_tl_injection_state": "session",
    "_source_bundle_manifest_sha256": "source_metadata",
    "_source_bundle_path": "source_metadata",
    "capture_mode": "header",
    # L7a mode marker: witness-owned per the memo (sibling of
    # capture_verified); stamped from the validated flag at entry today, by
    # the settlement witness machinery after S2 ratification (seam labeled in
    # user_funcs.py).
    "structure_only": "witness",
    "intervention_audit": "session",
    "_runnable": "runnable",
    "_fast_run_session": "graph",
    # L8/C2 physical-dispatch evidence is part of capture fidelity, so it
    # belongs with the other witness journals rather than the semantic graph.
    "_distributed_plane_p": "witness",
    "escape_detector_mode": "witness",
    "escape_detector_verified": "witness",
    "escape_diagnostics": "witness",
    "escape_detector_event_count": "witness",
    "escape_detector_callback_ns": "witness",
    "escape_detector_backward_coverage": "witness",
    "completeness_witness_mode": "witness",
    "completeness_witness_verified": "witness",
    "completeness_diagnostics": "witness",
    "completeness_decompositions": "witness",
    "completeness_witness_event_count": "witness",
    "completeness_witness_accounted_count": "witness",
    "completeness_witness_expected_opaque_count": "witness",
    "completeness_witness_unaccounted_count": "witness",
    "completeness_witness_callback_ns": "witness",
    "capture_verified": "witness",
    "capture_verification_reason": "witness",
    "rescue_rerun": "witness",
    "capture_owner_thread_id": "witness",
    "capture_owner_thread_qualified": "witness",
    "capture_thread_count_start": "witness",
    "capture_thread_count_end": "witness",
    "capture_thread_activity_detected": "witness",
    "capture_guard_passes": "witness",
    "halted": "witness",
    "halt_reason": "witness",
    "halt_frontier": "witness",
    "_capture_outcome": "witness",
    "_capture_phase": "session",
    "_settlement_ops_committed": "session",
    "_stop_requested": "session",
    "_layers_logged": "capture_config",
    "_layers_saved": "capture_config",
    "keep_orphans": "capture_config",
    "intervention_ready": "capture_config",
    "save_arg_templates": "capture_config",
    "raw_input": "graph",
    "input_preprocessor": "capture_config",
    "_transform": "capture_config",
    "transform_repr": "capture_config",
    "save_raw_input": "capture_config",
    "batch_render": "capture_config",
    "raw_output": "graph",
    "decoded_output": "graph",
    "output_postprocessor": "capture_config",
    "output_id2label": "capture_config",
    "output_num_classes": "capture_config",
    "_output_transform": "capture_config",
    "save_raw_output": "capture_config",
    "layer_visualizers": "capture_config",
    "save_visualizations": "capture_config",
    "_visualizer_dir": "capture_config",
    "random_seed": "capture_config",
    "detach_saved_activations": "capture_config",
    "output_device": "capture_config",
    "backward_ready": "capture_config",
    "inference_only": "capture_config",
    "chunked_forward": "capture_config",
    "module_filter": "capture_config",
    "emit_nvtx": "capture_config",
    "raise_on_nan": "capture_config",
    "track_nonfinite": "capture_config",
    "track_device_memory": "capture_config",
    "_device_memory_samples": "capture_config",
    "_autograd_saved_bands": "capture_config",
    "_autograd_seen_saved_storages": "capture_config",
    "_nonfinite_frontier_out": "capture_config",
    "_nonfinite_prefix_finalized": "capture_config",
    "_nonfinite_prefix_finalize_error": "capture_config",
    # Whole-field classification only: per-SUB-KEY travel across fresh
    # executions is governed by capture/_annotations_travel.py (foldA D6);
    # the provider settlement finalizer applies it to every run product.
    "annotations": "graph",
    "observer_spans": "graph",
    "manual_tensor_connections": "graph",
    "forward_source_file": "source_metadata",
    "forward_source_line": "source_metadata",
    "class_source_file": "source_metadata",
    "class_source_line": "source_metadata",
    "init_source_file": "source_metadata",
    "init_source_line": "source_metadata",
    "class_docstring": "source_metadata",
    "init_signature": "source_metadata",
    "init_docstring": "source_metadata",
    "forward_signature": "source_metadata",
    "forward_docstring": "source_metadata",
    "code_context": "source_metadata",
    "capture_cache_hit": "graph",
    "capture_cache_key": "graph",
    "capture_cache_path": "graph",
    "facet_registry_snapshot": "graph",
    "recording_kept": "graph",
    "_out_dedup_mode": "graph",
    "_out_identity_cache": "graph",
    "_out_hash_cache": "graph",
    "_code_context_cache": "graph",
    "_replay_arg_version_data_complete": "witness",
    "save_arg_values": "capture_config",
    "num_context_lines": "capture_config",
    "save_grads": "capture_config",
    "capture_tensor_grad_hooks": "capture_config",
    "_grad_op_nums_to_save": "capture_config",
    "grad_transform": "capture_config",
    "grad_transform_repr": "capture_config",
    "save_raw_gradients": "capture_config",
    "save_code_context": "capture_config",
    "save_rng_states": "capture_config",
    "recurrence_detection": "capture_config",
    "grouping": "capture_config",
    "grouping_policy": "capture_config",
    # L8/F6 is a trace-wide semantic-scope marker, not a user capture knob.
    # It belongs to the header beside backend and identity-mode facts.
    "distributed_scope": "header",
    # L9 backward residuals: timing clock provenance + checkpoint witness are
    # capture-evidence disclosures, not graph structure.
    "grad_fn_timing_provenance": "witness",
    "checkpoint_invocation_witness": "witness",
    # tlspec v9 entry-dark Trace slots (C07): the F30 source-snapshot table is
    # a model-source fact beside the other source rows; the F33 structure-only
    # evidence envelope is capture-honesty evidence beside structure_only.
    "source_snapshots": "source_metadata",
    "structure_evidence": "witness",
    "verbose": "capture_config",
    "profile_enabled": "capture_config",
    "has_gradients": "totals",
    "activation_transform": "capture_config",
    "_activation_transform_repr": "capture_config",
    "save_raw_activations": "capture_config",
    "save_mode": "capture_config",
    "input_annotations": "graph",
    "_source_code_blob": "source_metadata",
    "_source_model_ref": "source_metadata",
    "parent_run": "graph",
    "model_object_id": "source_metadata",
    "model_class_qualname": "source_metadata",
    "param_hash_quick": "source_metadata",
    "root_entry_point": "source_metadata",
    "param_hash_full": "source_metadata",
    "input_object_id": "source_metadata",
    "input_signature_hash": "source_metadata",
    "mark_layer_depths": "capture_config",
    "graph_shape_hash": "header",
    "_intervention_spec": "graph",
    "state_history": "graph",
    "last_run": "graph",
    "append_history": "graph",
    "_has_direct_writes": "graph",
    "_warned_direct_write": "graph",
    "_warned_mutate_in_place": "graph",
    "_spec_revision": "graph",
    "_out_recipe_revision": "graph",
    "_append_sequence_id": "graph",
    "_last_hook_handle_ids": "graph",
    "state": "header",
    "is_appended": "header",
    "relationship_evidence": "graph",
    "replay_frontier": "graph",
    "layer_list": "graph",
    "layer_dict_main_keys": "graph",
    "layer_dict_all_keys": "graph",
    "layer_logs": "graph",
    "layer_labels": "graph",
    "op_labels": "graph",
    "layer_num_calls": "graph",
    "by_pass": "graph",
    "_layer_nums_to_save": "capture_config",
    "num_ops": "totals",
    "num_modules": "totals",
    "_raw_to_final_layer_labels": "graph",
    "_raw_to_final_parent_layer_labels": "graph",
    "_raw_to_final_op_labels": "graph",
    "_final_to_raw_layer_labels": "graph",
    "_lookup_keys_to_layer_num_dict": "graph",
    "_layer_num_to_lookup_keys_dict": "graph",
    "_ambiguous_lookup_keys": "graph",
    "input_layers": "graph",
    "output_layers": "graph",
    "input_structure": "graph",
    "_containers": "graph",
    "_annotation_blobs": "graph",
    "buffer_layers": "graph",
    "buffer_num_calls": "graph",
    "_buffer_persistence": "graph",
    "internal_source_ops": "graph",
    "internal_sink_ops": "graph",
    "internally_terminated_bool_ops": "graph",
    "conditional_branch_edges": "graph",
    "conditional_records": "graph",
    "conditional_arm_entry_edges": "graph",
    "conditional_edge_call_indices": "graph",
    "conditionals": "graph",
    "layers_with_params": "graph",
    "op_equivalence_classes": "graph",
    "_orphan_labels": "graph",
    "_orphan_logs": "graph",
    "orphan_records": "graph",
    "total_activation_memory": "totals",
    "total_gradient_memory": "totals",
    "total_backward_memory": "totals",
    "total_autograd_memory": "totals",
    "num_saved_ops": "totals",
    "saved_activation_memory": "totals",
    "saved_gradient_memory": "totals",
    "num_saved_layers": "totals",
    "num_saved_module_calls": "totals",
    "num_saved_grad_fns": "totals",
    "num_saved_grad_fn_calls": "totals",
    "param_logs": "graph",
    "num_param_tensors": "totals",
    "num_layers_with_params": "totals",
    "num_params": "totals",
    "num_params_trainable": "totals",
    "num_params_frozen": "totals",
    "total_param_memory": "totals",
    "total_param_gradient_memory": "totals",
    "forward_peak_memory": "totals",
    "forward_memory_backend": "totals",
    "capture_start_time": "totals",
    "capture_end_time": "totals",
    "_phase_timings": "totals",
    "setup_duration": "totals",
    "forward_duration": "totals",
    "cleanup_duration": "totals",
    "func_calls_duration": "totals",
    "has_backward_pass": "totals",
    "grad_fn_logs": "graph",
    "grad_fn_order": "graph",
    "backward_pass_logs": "graph",
    "_grad_fn_param_refs": "graph",
    "backward_root_grad_fn_object_ids": "graph",
    "backward_durations": "totals",
    "num_backward_passes": "totals",
    "backward_peak_memory": "totals",
    "backward_memory_backend": "totals",
    "_cooked_from": "header",
    "_paddle_capture_depth": "session",
    "_paddle_op_captures": "session",
    "_paddle_alias_annotations": "session",
    "_paddle_capture_gap_markers": "session",
    "_tf_unresolved_producers": "session",
    "_tf_init_op_labels": "session",
    "_tf_op_captures": "session",
    "_tf_validation_result": "session",
    "_tl_save_selector_fire_count": "session",
    "_tl_intervene_selector_fire_count": "session",
    "_module_call_accessor": "graph",
    "_op_accessor_cache": "graph",
    "_layer_accessor_cache": "graph",
    "_receptive_field_solution": "session",
    "_rf_directional_solutions": "session",
    "_optimizer": "session",
    "measure_python_peak_memory": "capture_config",
    "_forward_peak_memory_pair": "totals",
    "distributed_witness": "capture_config",
    "save_budget": "capture_config",
    "_warned_once": "session",
    "_predicate_save_options": "session",
    "_predicate_history_size": "session",
    "_predicate_history": "session",
    "_predicate_lookback": "session",
    "_predicate_lookback_payload_policy": "session",
    "_capture_config": "capture_config",
    "_stop_directive": "session",
    "_halt_returns_partial_trace": "session",
    "_predicate_save_decisions": "session",
    "_predicate_current_contexts": "session",
    "_predicate_lookback_candidates": "session",
    "_postprocessing_active": "session",
    "_raw_transform_escape_detected": "witness",
    "_raw_dynamo_region_detected": "witness",
    "_raw_event_shape_hash": "witness",
    "_output_container_specs_by_raw_label": "graph",
    "_buffer_accessor": "graph",
    "_buffer_write_tracker": "graph",
    "_param_storage_addresses": "graph",
    "_buffer_initial_values": "graph",
    "_saved_grad_labels": "graph",
    "ops_with_params": "graph",
    "_pending_live_fire_records": "session",
    "_module_logs": "graph",
    "_param_logs_by_module": "graph",
    "_raw_graph_ws": "session",
    "_module_capture_ws": "session",
    "_wrapper_runtime_ws": "session",
    "_trace_core": "graph",
    "_pre_forward_rng_states": "session",
    "_buffer_storage_addresses": "graph",
    "_mlx_saved_payloads": "session",
    "_mlx_capture_depth": "session",
    "_out_writer": "session",
    "_save_budget_accountant": "session",
    "_keep_outs_in_memory": "session",
    "_grad_stream_retain_in_memory": "session",
    "_defer_streaming_bundle_finalization": "session",
    "_out_sink": "session",
    "_orphan_pruned_func_call_ids": "graph",
    "_capture_parent_edge_truth": "graph",
    "_nonfinite_capture": "session",
    "_capture_events": "session",
    "_capture_session": "session",
    "_tl_backward_hooked_tensor_keys": "session",
    "_tl_grad_hook_owner_by_label": "session",
    "_tl_rf_probe_active": "session",
    "_active_backward_pass_index": "graph",
    "_backward_roots_by_pass": "graph",
    "_backward_projection_event_count": "graph",
    "_backward_projection_revision": "graph",
    "_backward_projection_fold_state": "graph",
    "_implicit_backward_pass_open": "graph",
    "_tl_backward_triggers_disarmed": "session",
    "_tl_cleaned_up": "session",
    "_grad_fn_param_refs_by_object_id": "graph",
    "_param_log_by_pid": "graph",
    "_session_param_inventory": "session",
    "_session_buffer_inventory": "session",
    "_session_buffer_identity": "session",
    "_backward_gradfn_refs": "graph",
}

#: Private Trace attributes ATTACHED FROM OUTSIDE ``data_classes/`` that carry
#: no declared owner above, each with the reason it is tolerated. Seeded from
#: the b5 census (36 names / 62 write sites at 2026-08-14; only the eight noted
#: as "scrub-declared" have a home in ANY other authority). Exact-equality
#: ledger: a new undeclared attachment fails the gate, and enrolling one
#: requires deleting its row here.
TRACE_EXTERNAL_WRITE_EXEMPTIONS: dict[str, str] = {
    # --- Preview-backend capture scratch (mlx/paddle/tf/jax) ----------------
    # Each is written by its own backend during capture and read back by that
    # same backend; none is portable. The parity lanes own their relocation
    # (into a backend-side session object, not the Trace).
    "_mlx_module_stack": "mlx: module-stack scratch, attached/deleted around the mlx forward",
    "_mlx_op_captures": "mlx: raw op-capture list (scrub-declared runtime-only)",
    "_mlx_replay_inventory": "mlx: replay inventory for validation (scrub-declared runtime-only)",
    "_mlx_halt_selector": "mlx: resolved halt selector for the live forward",
    "_mlx_intervention_plan": "mlx: resolved intervention plan for the live forward",
    "_mlx_intervene_fired": (
        "mlx: fired-intervention flag stamped during the live forward, read "
        "back with a getattr default by the mlx validation epilogue"
    ),
    "_mlx_perturbation_gaps": "mlx: per-op replay perturbation gaps recorded by mlx validation",
    "_paddle_module_stack": "paddle: module-stack scratch, attached/deleted around the forward",
    "_paddle_intervention_runtime": "paddle: live intervention runtime for the dygraph forward",
    "_tf_static_region_labels": "tf: FuncGraph static-path region captures",
    "_tf_static_fallback_error": "tf: FuncGraph static-path fallback diagnostic",
    "_jax_capture_index_to_raw_op_label": (
        "jax: capture-index to raw-label map (scrub-declared runtime-only)"
    ),
    "_selective_save_hidden_payloads": (
        "previews: neutral selective-save side channel (scrub-declared runtime-only)"
    ),
    # --- Cross-backend validation side channels ----------------------------
    # Written by every backend's validation epilogue and by the public
    # validate() impl; the honest fix is a declared validation-result field.
    "_validation_replay_status": (
        "validation: replay status stamped by all five backends + "
        "_user_public_impls "
        "(12 sites; scrub-declared runtime-only, no schema row)"
    ),
    "_validation_pruned_dispatchable_op_count": (
        "validation: counter written under try/except and read with a "
        "getattr default -- a dropped write reads as zero"
    ),
    "_validation_buffer_write_dispatch_op_count": (
        "validation: counter written under try/except and read with a "
        "getattr default -- a dropped write reads as zero"
    ),
    # --- Torch capture-session transients ----------------------------------
    # Live only inside one forward/backward bracket; several are the
    # re-entrancy flags the wrapper hot path reads.
    "_capture_producer_policy": "torch: journal producer policy (scrub-declared runtime-only)",
    "_capture_container_structure": (
        "torch: container-structure scratch, attached and deleted by the "
        "capture entrypoint (scrub-declared runtime-only)"
    ),
    "_active_save_grads_policy": "torch: resolved save_grads policy for the active backward",
    "_tl_active_backward_bracket": "torch: re-entrancy flag for the active backward bracket",
    "_tl_materializing_backward_projection": (
        "torch: re-entrancy flag while a backward projection materializes"
    ),
    "_installing_deferred_gradient_hooks": (
        "capture session: re-entrancy flag while deferred grad hooks install"
    ),
    "_prehook_provenance_ledger": "torch: forward-pre-hook provenance ledger for one capture",
    "_fastlog_recording": "fastlog: live Recording handle bound to the trace for the capture",
    "_echo_session": (
        "snoop (lane F28): the read-only echo narration observer bound at "
        "capture entry; runtime-only by design, never a declared field, "
        "never persisted"
    ),
    # --- Refresh-projection plumbing (user_funcs + capture/projectors) -----
    # A refresh capture carries its resolution decisions on the trace so the
    # projector can rebind them; strictly session-lifetime.
    "_refresh_resolved_layer_nums_to_save": "refresh projection: resolved save set",
    "_refresh_resolved_grad_layer_nums_to_save": "refresh projection: resolved grad save set",
    "_refresh_projection_capture": "refresh projection: marks the trace as a refresh capture",
    "_refresh_projection_target_ref": "refresh projection: weakref back to the projection target",
    "_deferred_retention_selector": "refresh projection: deferred retention selector",
    "_deferred_gradient_selector": "refresh projection: deferred gradient selector",
    # --- Bundle-load provenance (set by _io/bundle via setattr) ------------
    # `_source_bundle_path` / `_source_bundle_manifest_sha256` ARE declared;
    # these siblings were never enrolled with them. The complete family and
    # its save-time strip behavior are declared ONCE in
    # ``torchlens._io.bundle.LOAD_PROVENANCE_TRANSIENT_ATTRS`` (R50-2).
    "_loaded_from_bundle": "bundle load: marks a loaded trace (undeclared sibling of the two owned)",
    "_source_bundle_created_at": "bundle load: manifest created_at (undeclared sibling)",
    "_source_bundle_provenance": "bundle load: manifest provenance (undeclared sibling)",
    "_source_bundle_model_fingerprint": (
        "bundle load: manifest model fingerprint, read back by the runnable "
        "tlspec writer (was the one family member outside every authority, "
        "R50-1/R50-2)"
    ),
    # --- One-shot warning latches -----------------------------------------
    "_warned_direct_write_propagation": "intervention: one-shot warning latch (replay + rerun)",
    "_warned_unknown_append_helper": "intervention: one-shot warning latch (rerun)",
    # --- Visualization scratch --------------------------------------------
    "_last_sibling_ordering_decision": (
        "viz: last sibling-ordering decision for diagnostics (scrub-declared runtime-only)"
    ),
    "_last_encoding_state": (
        "viz: last draw's encoding-channel state (L5 color_by) for diagnostics "
        "(scrub-declared runtime-only)"
    ),
    "_last_render_geometry": (
        "viz: last draw's layout-execution geometry record (vizmech D24) for "
        "diagnostics (scrub-declared runtime-only)"
    ),
    # --- __dict__-spelled transients surfaced by the r6 R45 gate widening --
    # Every row below is popped/consumed within its own window; none can
    # reach tl.save on the exhaustive path (the counter that COULD --
    # _backward_grad_fn_type_counter -- was fixed to rewalk-scoped scratch
    # instead of being ledgered here).
    "_tl_pending_accumulate_grad_fire_records": (
        "torch backward: AccumulateGrad fire records pending projection; popped in the "
        "same backward bracket (both exits)"
    ),
    "_higher_order_grad_fn_terminals": (
        "torch backward: differentiable-pass terminal queue; popped by the rewalk that consumes it"
    ),
    "_module_boundary_adoptions": (
        "torch model prep: module-boundary (entry and exit) adoption queue; popped by its postprocess step "
        "and scrub-declared"
    ),
    "_module_boundary_outside_sources": (
        "torch model prep: module-boundary adoptions of closure/forward-global tensors (no "
        "escape signal); popped by the same postprocess step and scrub-declared"
    ),
    "_fastlog_grad_contexts": (
        "fastlog: predicate-pass grad RecordContexts keyed by public label; lives on the "
        "recorder's session trace only, never on a cooked to_trace() result"
    ),
    # --- FLIP-2 grad_fn marker sink diagnostics (F27) ----------------------
    "_tl_gradfn_marker_leaks": (
        "torch backward: session-time leak counter for markers closed at a pass "
        "boundary rather than their own posthook; read back with a dict.get "
        "default by the test suite, never persisted"
    ),
    "_tl_gradfn_marker_gaps": (
        "torch backward: session-time counter for marker registrations that "
        "degraded to unmarkered; read back with a dict.get default by the test "
        "suite, never persisted"
    ),
    # --- Accelerate offload/dispatch hook shims (F37, R5) ------------------
    "_offload_hook_shims": (
        "torch model prep: session-scoped shim list set only while an "
        "accelerate offload hook exists, deleted at uninstall so no live-trace "
        "attribute ever reaches the save-time portability spec check"
    ),
    "_offload_param_rebinds": (
        "torch model prep: session-scoped weak-valued rebind map set only "
        "while an accelerate offload hook exists, deleted at uninstall for the "
        "same reason as _offload_hook_shims"
    ),
}
