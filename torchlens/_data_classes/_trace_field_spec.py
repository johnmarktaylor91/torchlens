"""The persisted-field policy table of ``Trace`` (``Trace._PORTABLE_STATE_SPEC``).

A pure relocation of the class constant so ``trace.py`` stays inside its file cap; the class
attribute, the generated schema and every reader of ``Trace._PORTABLE_STATE_SPEC`` are unchanged.
"""

from __future__ import annotations

from .._io._field_policy import FieldPolicy

__all__ = ["TRACE_PORTABLE_STATE_SPEC"]


TRACE_PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
    "name": FieldPolicy.KEEP,
    "_source_class_name": FieldPolicy.KEEP,
    "model_name": FieldPolicy.KEEP,
    "_provenance_backend": FieldPolicy.KEEP,
    "_provenance_backend_config": FieldPolicy.KEEP,
    "backend_runtime_device_summary": FieldPolicy.KEEP,
    "_provenance_backend_version": FieldPolicy.KEEP,
    # Provenance marker for traces cooked from a Recording: postprocess
    # runs exhaustive-style (capture_mode stays "exhaustive" for behavior
    # compatibility), but the marker records the true origin so gates and
    # diagnostics never mistake a cooked projection for a live exhaustive
    # capture. Session-only; not a portable fact.
    "_cooked_from": FieldPolicy.DROP,
    # Settlement-authority state (torchlens/_capture/outcome.py). The
    # settled outcome is a PERSISTED portable fact (tlspec v7): saved as
    # its string-only payload, parsed against closed vocabularies and the
    # status coherence matrix at load. The phase marker and stop-request
    # latch are strictly session-transient.
    "_capture_outcome": FieldPolicy.KEEP,
    "_capture_phase": FieldPolicy.DROP,
    "_settlement_ops_committed": FieldPolicy.DROP,
    "_stop_requested": FieldPolicy.DROP,
    "_paddle_capture_depth": FieldPolicy.DROP,
    "_paddle_op_captures": FieldPolicy.DROP,
    "_paddle_alias_annotations": FieldPolicy.DROP,
    "_paddle_capture_gap_markers": FieldPolicy.DROP,
    "_tf_unresolved_producers": FieldPolicy.DROP,
    "_tf_init_op_labels": FieldPolicy.DROP,
    "_tf_op_captures": FieldPolicy.DROP,
    "_tf_validation_result": FieldPolicy.DROP,
    # Live native-backward state a backward_ready capture retains for a later
    # trace.backward(): a tape, scope or pinned captures, never portable. A loaded
    # trace has none, and its backward() refuses.
    "_tf_backward_state": FieldPolicy.DROP,
    "_paddle_backward_state": FieldPolicy.DROP,
    "_tinygrad_backward_state": FieldPolicy.DROP,
    "_jax_backward_captures": FieldPolicy.DROP,
    # TensorFlow tl.grad clauses a capture keeps for a later gradient replay, and
    # the sites already armed from them: live user callables, never portable.
    # A loaded trace has no TensorFlow tape to apply them to.
    "_tf_grad_site_clauses": FieldPolicy.DROP,
    "_tf_grad_sites_armed": FieldPolicy.DROP,
    # JAX capture scratch the v11 groups lane moved behind provenance/options
    # views: written by the JAX backend, read back by it, never portable.
    "_provenance_jax_closed_jaxpr": FieldPolicy.DROP,
    "_provenance_jax_inlined_call_primitives": FieldPolicy.DROP,
    "_options_jax_static_argnums": FieldPolicy.DROP,
    "_tl_save_selector_fire_count": FieldPolicy.DROP,
    "_tl_intervene_selector_fire_count": FieldPolicy.DROP,
    "_module_call_accessor": FieldPolicy.DROP,
    "_op_accessor_cache": FieldPolicy.DROP,
    "_layer_accessor_cache": FieldPolicy.DROP,
    "_receptive_field_solution": FieldPolicy.DROP,
    "_rf_directional_solutions": FieldPolicy.DROP,
    "_options_module_identity_mode": FieldPolicy.KEEP,
    "_provenance_param_source": FieldPolicy.KEEP,
    "_autograd_derived_grads": FieldPolicy.KEEP,
    "num_context_lines": FieldPolicy.KEEP,
    "_optimizer": FieldPolicy.DROP,
    "tlspec_version": FieldPolicy.KEEP,
    "_tracing_finished": FieldPolicy.KEEP,
    "_options_mode": FieldPolicy.KEEP,
    # L7a structure-only mode marker: persists as of tlspec v8 (the
    # coordinated bump) with the marker-coherence load-validation rows in
    # torchlens/_io/forgery_validation.py (the form-(a)
    # stripped-marker case is a documented scope statement there).
    "_options_structure_only": FieldPolicy.KEEP,
    # tlspec v9 entry-dark evidence envelope: None until the writer lands; coherence-validated
    # at load (only a structure-only capture may carry one).
    "_graph_structure_evidence": FieldPolicy.KEEP,
    # L6 resolved-intervention audit record: persists as of tlspec v8
    # (reprs/identities/digests only, never raw values) with its load
    # validation in torchlens/_io/forgery_validation.py.
    "intervention_audit": FieldPolicy.KEEP,
    "_runnable": FieldPolicy.DROP,
    "_fast_run_session": FieldPolicy.DROP,
    "_distributed_plane_p": FieldPolicy.DROP,
    "_health_escape_detector_mode": FieldPolicy.DROP,
    "_health_escape_detector_verified": FieldPolicy.DROP,
    "escape_diagnostics": FieldPolicy.DROP,
    "_health_num_escape_detector_events": FieldPolicy.DROP,
    "_health_escape_detector_callback_ns": FieldPolicy.DROP,
    "_autograd_escape_detector_backward_coverage": FieldPolicy.DROP,
    "_health_completeness_mode": FieldPolicy.DROP,
    "_health_completeness_verified": FieldPolicy.DROP,
    "completeness_diagnostics": FieldPolicy.DROP,
    "_health_completeness_decompositions": FieldPolicy.DROP,
    "_health_num_witness_events": FieldPolicy.DROP,
    "_health_num_accounted_witness_events": FieldPolicy.DROP,
    "_health_num_expected_opaque_witness_events": FieldPolicy.DROP,
    "_health_num_unaccounted_witness_events": FieldPolicy.DROP,
    "_health_completeness_callback_ns": FieldPolicy.DROP,
    # The escape-disclosure VERDICT persists (negative claims
    # only -- __getstate__/__setstate__ degrade anything else to None), so
    # a capture TorchLens refused to bless is no longer byte-
    # indistinguishable from a clean one after save/load. rescue_rerun
    # stays session-time per docs/migration/scoped_detached_patching.md.
    "_provenance_verified": FieldPolicy.KEEP,
    "_provenance_verification_reason": FieldPolicy.KEEP,
    "_rescue_rerun": FieldPolicy.DROP,
    "_provenance_owner_thread_id": FieldPolicy.DROP,
    "_provenance_owner_thread_qualified": FieldPolicy.DROP,
    "_provenance_thread_count_start": FieldPolicy.DROP,
    "_provenance_thread_count_end": FieldPolicy.DROP,
    "_provenance_thread_activity_detected": FieldPolicy.DROP,
    "capture_guard_passes": FieldPolicy.DROP,
    "is_stopped": FieldPolicy.KEEP,
    "stop_reason": FieldPolicy.KEEP,
    "_graph_stop_frontier": FieldPolicy.KEEP,
    "_layers_logged": FieldPolicy.KEEP,
    "_layers_saved": FieldPolicy.KEEP,
    "_options_keep_orphans": FieldPolicy.KEEP,
    "is_replayable": FieldPolicy.KEEP,
    "_options_save_arg_templates": FieldPolicy.KEEP,
    "input": FieldPolicy.KEEP,
    "_provenance_input_preprocessor": FieldPolicy.KEEP,
    "_transform": FieldPolicy.DROP,
    "_options_input_transform": FieldPolicy.KEEP,
    "_options_save_raw_input": FieldPolicy.KEEP,
    "batch_render": FieldPolicy.KEEP,
    "transformed_output": FieldPolicy.KEEP,
    "_decoded_output": FieldPolicy.KEEP,
    "_provenance_output_postprocessor": FieldPolicy.KEEP,
    "output_id2label": FieldPolicy.KEEP,
    "output_num_classes": FieldPolicy.KEEP,
    "_output_transform": FieldPolicy.DROP,
    "_output_slot": FieldPolicy.DROP,
    "_options_save_transformed_output": FieldPolicy.KEEP,
    "layer_visualizers": FieldPolicy.DROP,
    "save_visualizations": FieldPolicy.KEEP,
    "_visualizer_dir": FieldPolicy.DROP,
    "_options_activation_transform": FieldPolicy.DROP,
    "_activation_transform_repr": FieldPolicy.KEEP,
    "_options_save_raw_activations": FieldPolicy.KEEP,
    "_options_save_mode": FieldPolicy.KEEP,
    "_provenance_input_annotations": FieldPolicy.KEEP,
    "_source_code_blob": FieldPolicy.KEEP,
    "_source_model_ref": FieldPolicy.DROP,
    # Session-time: recorded at the module build while the model is live.
    "_uncalled_module_addresses": FieldPolicy.DROP,
    "_parent_run": FieldPolicy.DROP,
    "_provenance_model_object_id": FieldPolicy.KEEP,
    "_source_class_qualname": FieldPolicy.KEEP,
    "_provenance_param_hash_quick": FieldPolicy.KEEP,
    # tlspec v9 root entry-point identity fact (C07X item (iv)): the
    # closed "kind:qualified_identity" descriptor, written
    # unconditionally at capture beside the two facts above; fail-closed
    # grammar validation at load (_io/forgery_validation.py).
    "_provenance_root_entry_point": FieldPolicy.KEEP,
    "_provenance_param_hash_full": FieldPolicy.KEEP,
    "_provenance_input_object_id": FieldPolicy.KEEP,
    "_provenance_input_signature_hash": FieldPolicy.KEEP,
    "_options_seed": FieldPolicy.KEEP,
    "_options_output_device": FieldPolicy.KEEP,
    "_options_detach_saved_activations": FieldPolicy.KEEP,
    "_options_backward_ready": FieldPolicy.DROP,
    "_options_inference_only": FieldPolicy.KEEP,
    "_options_chunk_size": FieldPolicy.KEEP,
    "_options_emit_nvtx": FieldPolicy.KEEP,
    # Session-time measurement knob (like ``backward_ready``): it selects how
    # ``forward_peak_memory`` was measured during this capture and has no
    # meaning for a loaded artifact, which never re-measures. Portable load
    # restores the default ``False``, so it stays out of
    # ``MODEL_LOG_FIELD_ORDER`` and out of the portable schema.
    "measure_python_peak_memory": FieldPolicy.DROP,
    "_forward_peak_memory_pair": FieldPolicy.DROP,  # session pair
    # Session-time witness knob for collective boundary records: it selects
    # what capture PAID FOR (digests or nothing), not what a trace means;
    # the per-boundary witness.policy_resolved field is the portable
    # evidence. Portable load restores the default "none".
    "distributed_witness": FieldPolicy.DROP,
    # Session-time resource ceiling: it bounds what THIS process was willing
    # to retain and has no meaning for a loaded artifact, which retains
    # nothing. Portable load restores the default, so it stays out of
    # ``MODEL_LOG_FIELD_ORDER`` and out of the portable schema.
    "save_budget": FieldPolicy.DROP,
    "_options_on_nonfinite": FieldPolicy.KEEP,
    # Session-time recording knob (same class as measure_python_peak_memory):
    # it selects what capture PAID FOR (per-op finiteness checks), not what a
    # trace means. Portable load restores the default False, so it stays out
    # of MODEL_LOG_FIELD_ORDER and out of the portable schema.
    "track_nonfinite": FieldPolicy.DROP,
    # sampling knob + per-call sample store: allocator counters are
    # process facts of the capturing run; load restores the default.
    "track_device_memory": FieldPolicy.DROP,
    "measure_host_peak_memory": FieldPolicy.DROP,  # session knob
    "_options_aten_ops": FieldPolicy.DROP,  # session knob
    "_device_memory_samples": FieldPolicy.DROP,
    "annotations": FieldPolicy.KEEP,
    "_provenance_observer_spans": FieldPolicy.KEEP,
    "_graph_manual_tensor_connections": FieldPolicy.KEEP,
    "_source_forward_file": FieldPolicy.KEEP,
    "_source_forward_line": FieldPolicy.KEEP,
    "_source_class_file": FieldPolicy.KEEP,
    "_source_class_line": FieldPolicy.KEEP,
    "_source_init_file": FieldPolicy.KEEP,
    "_source_init_line": FieldPolicy.KEEP,
    "_source_class_docstring": FieldPolicy.KEEP,
    "_source_init_signature": FieldPolicy.KEEP,
    "_source_init_docstring": FieldPolicy.KEEP,
    "_source_forward_signature": FieldPolicy.KEEP,
    "_source_forward_docstring": FieldPolicy.KEEP,
    "_source_frames": FieldPolicy.KEEP,
    # tlspec v9 entry-dark source-snapshot table: one row per (path, digest), empty until the writer lands;
    # fail-closed row validation at load.
    "_source_snapshots": FieldPolicy.KEEP,
    "_provenance_cache_hit": FieldPolicy.KEEP,
    "_provenance_cache_key": FieldPolicy.KEEP,
    "_provenance_cache_path": FieldPolicy.KEEP,
    "_provenance_recording_kept": FieldPolicy.KEEP,
    "facet_functions": FieldPolicy.DROP,
    "_out_dedup_mode": FieldPolicy.DROP,
    "_out_identity_cache": FieldPolicy.DROP,
    "_out_hash_cache": FieldPolicy.DROP,
    "_code_context_cache": FieldPolicy.DROP,
    "_options_save_args": FieldPolicy.KEEP,
    "_options_save_grads": FieldPolicy.KEEP,
    "capture_tensor_grad_hooks": FieldPolicy.KEEP,
    "_grad_op_nums_to_save": FieldPolicy.KEEP,
    "_options_grad_transform": FieldPolicy.DROP,
    "_options_save_raw_grads": FieldPolicy.KEEP,
    "_options_save_source_context": FieldPolicy.KEEP,
    "_options_save_rng": FieldPolicy.KEEP,
    "_options_recurrence_detection": FieldPolicy.KEEP,
    # L1 grouping surface: persists as of tlspec v8; the grouping-policy
    # stamp validates at load (_postprocess/_grouping_stamp.py).
    "_options_grouping": FieldPolicy.KEEP,
    "_options_grouping_policy": FieldPolicy.KEEP,
    # Shard-local capture marker (merge-ranks substrate):
    # persists as of tlspec v8 with the closed-vocabulary load validation
    # in torchlens/_io/forgery_validation.py. Value vocabulary
    # ("rank_local_shard") is DOCUMENTED-UNSTABLE pending its amendment.
    "_provenance_distributed_scope": FieldPolicy.KEEP,
    # L9 backward-residuals surface: both persist as of tlspec v8 with
    # closed-vocabulary load validation in _io/forgery_validation.py.
    # Value vocabularies provisional (E-L9-4 routing).
    "grad_fn_timing_provenance": FieldPolicy.KEEP,
    "checkpoint_invocation_witness": FieldPolicy.KEEP,
    "_options_verbose": FieldPolicy.KEEP,
    "profile_enabled": FieldPolicy.KEEP,
    "_autograd_has_grads": FieldPolicy.KEEP,
    "_options_mark_layer_depths": FieldPolicy.KEEP,
    "_provenance_fingerprint": FieldPolicy.KEEP,
    "_intervention_spec": FieldPolicy.DROP,
    "state_history": FieldPolicy.KEEP,
    "last_run": FieldPolicy.DROP,
    "_append_history": FieldPolicy.KEEP,
    "_has_direct_writes": FieldPolicy.KEEP,
    "_warned_direct_write": FieldPolicy.DROP,
    "_warned_mutate_in_place": FieldPolicy.DROP,
    "_warned_once": FieldPolicy.DROP,
    "_spec_revision": FieldPolicy.KEEP,
    "_out_recipe_revision": FieldPolicy.KEEP,
    "_append_sequence_id": FieldPolicy.KEEP,
    "_last_hook_handle_ids": FieldPolicy.DROP,
    "_predicate_save_options": FieldPolicy.DROP,
    "_predicate_history_size": FieldPolicy.DROP,
    "_predicate_history": FieldPolicy.DROP,
    "_predicate_lookback": FieldPolicy.DROP,
    "_predicate_lookback_payload_policy": FieldPolicy.DROP,
    "_capture_config": FieldPolicy.DROP,
    "_stop_directive": FieldPolicy.DROP,
    "_halt_returns_partial_trace": FieldPolicy.DROP,
    "_predicate_save_decisions": FieldPolicy.DROP,
    "_predicate_current_contexts": FieldPolicy.DROP,
    "_predicate_lookback_candidates": FieldPolicy.DROP,
    "_postprocessing_active": FieldPolicy.DROP,
    "_raw_transform_escape_detected": FieldPolicy.DROP,
    "_raw_dynamo_region_detected": FieldPolicy.DROP,
    "_raw_event_shape_hash": FieldPolicy.DROP,
    "_raw_call_fingerprint": FieldPolicy.DROP,
    "_replay_arg_version_data_complete": FieldPolicy.KEEP,
    "state": FieldPolicy.KEEP,
    "_provenance_is_appended": FieldPolicy.KEEP,
    "_graph_relationship_evidence": FieldPolicy.KEEP,
    "_replay_frontier": FieldPolicy.DROP,
    "_output_container_specs_by_raw_label": FieldPolicy.DROP,
    "_layer_list": FieldPolicy.KEEP,
    "_layer_dict_main_keys": FieldPolicy.KEEP,
    "_layer_dict_all_keys": FieldPolicy.KEEP,
    "_layer_logs": FieldPolicy.KEEP,
    "layer_labels": FieldPolicy.KEEP,
    "op_labels": FieldPolicy.KEEP,
    "_layer_num_calls": FieldPolicy.KEEP,
    "_by_pass": FieldPolicy.KEEP,
    "_layer_nums_to_save": FieldPolicy.KEEP,
    "_num_ops": FieldPolicy.KEEP,
    "_raw_to_final_layer_labels": FieldPolicy.KEEP,
    "_raw_to_final_parent_layer_labels": FieldPolicy.KEEP,
    "_raw_to_final_op_labels": FieldPolicy.KEEP,
    "_final_to_raw_layer_labels": FieldPolicy.KEEP,
    "_lookup_keys_to_layer_num_dict": FieldPolicy.KEEP,
    "_layer_num_to_lookup_keys_dict": FieldPolicy.KEEP,
    "_ambiguous_lookup_keys": FieldPolicy.KEEP,
    "input_layers": FieldPolicy.KEEP,
    "output_layers": FieldPolicy.KEEP,
    "_provenance_input_structure": FieldPolicy.BLOB_RECURSIVE,
    "_containers": FieldPolicy.BLOB_RECURSIVE,
    "_annotation_blobs": FieldPolicy.BLOB_RECURSIVE,
    "buffer_layers": FieldPolicy.KEEP,
    "_buffer_num_calls": FieldPolicy.KEEP,
    "_buffer_accessor": FieldPolicy.DROP,
    "_buffer_write_tracker": FieldPolicy.DROP,
    "_param_storage_addresses": FieldPolicy.DROP,
    "_buffer_initial_values": FieldPolicy.BLOB_RECURSIVE,
    "_buffer_persistence": FieldPolicy.KEEP,
    "internal_source_ops": FieldPolicy.KEEP,
    "internal_sink_ops": FieldPolicy.KEEP,
    "internally_terminated_bool_ops": FieldPolicy.KEEP,
    "_graph_conditional_branch_edges": FieldPolicy.KEEP,
    "_graph_conditional_records": FieldPolicy.KEEP,
    "_graph_conditional_arm_entry_edges": FieldPolicy.KEEP,
    "_graph_conditional_edge_call_indices": FieldPolicy.KEEP,
    "_graph_conditionals": FieldPolicy.KEEP,
    "_orphan_labels": FieldPolicy.KEEP,
    "_orphan_logs": FieldPolicy.KEEP,
    "_orphan_records": FieldPolicy.BLOB_RECURSIVE,
    "_saved_grad_labels": FieldPolicy.DROP,
    "layers_with_params": FieldPolicy.KEEP,
    "ops_with_params": FieldPolicy.DROP,
    "_graph_op_equivalence_classes": FieldPolicy.KEEP,
    "_cost_total_activation_memory": FieldPolicy.KEEP,
    "_cost_total_grad_memory": FieldPolicy.KEEP,
    "_cost_total_backward_memory": FieldPolicy.KEEP,
    "_cost_total_autograd_memory": FieldPolicy.KEEP,
    "_cost_pinned_autograd_memory": FieldPolicy.DROP,  # live graph only
    "_num_saved_ops": FieldPolicy.KEEP,
    "_cost_saved_activation_memory": FieldPolicy.KEEP,
    "_cost_saved_grad_memory": FieldPolicy.KEEP,
    "_param_logs": FieldPolicy.KEEP,
    "_num_param_tensors": FieldPolicy.KEEP,
    "_num_layers_with_params": FieldPolicy.KEEP,
    "num_params": FieldPolicy.KEEP,
    "num_params_trainable": FieldPolicy.KEEP,
    "num_params_frozen": FieldPolicy.KEEP,
    "_cost_total_param_memory": FieldPolicy.KEEP,
    "_cost_total_param_grad_memory": FieldPolicy.KEEP,
    "_cost_forward_peak_memory": FieldPolicy.KEEP,
    "_provenance_forward_memory_backend": FieldPolicy.KEEP,
    "_cost_forward_peak_memory_reason": FieldPolicy.KEEP,
    # Live-capture scratch reinstated by ``__setstate__`` alongside
    # ``_tl_backward_hooked_tensor_keys``, but never registered here. Any
    # trace that went through ``__setstate__`` -- which ``cache=True`` makes
    # routine -- therefore tripped the catalog tripwire and failed
    # ``save(level="runnable")`` outright on the SECOND capture. DROP, like
    # its sibling: pending live-fire records are session state and are
    # already reset on rehydrate.
    "_pending_live_fire_records": FieldPolicy.DROP,
    "_module_logs": FieldPolicy.DROP,
    "_param_logs_by_module": FieldPolicy.DROP,
    "_raw_graph_ws": FieldPolicy.DROP,
    "_module_capture_ws": FieldPolicy.DROP,
    "_wrapper_runtime_ws": FieldPolicy.DROP,
    # The per-trace columnar Op row store (torchlens._trace_core). Never
    # portable: plain pickle re-materializes each Op as a detached row
    # from its own state, and .tlspec artifacts stay object-shaped until
    # the direct semantic serialization.
    "_trace_core": FieldPolicy.DROP,
    # L3 primitive-op profile: persists as of tlspec v8; loaded profiles
    # validate through validate_loaded_primitive_profile whenever present.
    "_primitive_op_profile": FieldPolicy.KEEP,
    "_pre_forward_rng_states": FieldPolicy.DROP,
    # Pre-clone per-slot state metadata signatures (producer-side only,
    # never portable) and the buffer storage-pointer attribution index.
    # Capture-time persistent-buffer name universe + geometry
    # (producer-side only, never portable).
    "_buffer_storage_addresses": FieldPolicy.DROP,
    "_mlx_saved_payloads": FieldPolicy.DROP,
    "_mlx_capture_depth": FieldPolicy.DROP,
    "_out_writer": FieldPolicy.DROP,
    # Runtime-only: the live per-device accountant that enforces
    # ``save_budget`` while payloads are being retained. It describes what
    # THIS process was willing to allocate, so it is never portable; a loaded
    # artifact retains nothing and rebuilds it from the restored option.
    "_save_budget_accountant": FieldPolicy.DROP,
    "_keep_outs_in_memory": FieldPolicy.DROP,
    "_grad_stream_retain_in_memory": FieldPolicy.DROP,
    "_defer_streaming_bundle_finalization": FieldPolicy.DROP,
    "_out_sink": FieldPolicy.DROP,
    # Runtime-only: the set of dispatchable op func-call-ids the orphan-removal
    # pass pruned, read by the validation dispatch-count backstop. Never portable.
    "_orphan_pruned_func_call_ids": FieldPolicy.DROP,
    # Runtime-only: the capture-time (slot -> producer) parent-edge
    # truth keyed by raw label, read by the capture_edge_survival metadata
    # invariant. Registered in _io/scrub.py's runtime-only list; declared here
    # so the portable-state cover stays exhaustive. Never portable.
    "_capture_parent_edge_truth": FieldPolicy.DROP,
    # track_nonfinite's runtime store (events/unchecked/pending), set lazily
    # by the torch op-finalize hook; same runtime-bookkeeping class as
    # _capture_parent_edge_truth: never persisted, absent on loaded traces.
    "_nonfinite_capture": FieldPolicy.DROP,
    # session-time stores (saved-band decomposition, first-save set,
    # aborted-nonfinite prefix facts): the _capture_parent_edge_truth
    # class; loaded traces disclose honest absence (field_intent live-only).
    "_autograd_saved_bands": FieldPolicy.DROP,
    "_autograd_seen_saved_storages": FieldPolicy.DROP,
    "_nonfinite_frontier_out": FieldPolicy.DROP,
    "_nonfinite_prefix_finalized": FieldPolicy.DROP,
    "_nonfinite_prefix_finalize_error": FieldPolicy.DROP,
    "_capture_events": FieldPolicy.DROP,
    "_capture_session": FieldPolicy.DROP,
    "_tl_backward_hooked_tensor_keys": FieldPolicy.DROP,
    "_tl_grad_hook_owner_by_label": FieldPolicy.DROP,
    # RF probe suppression flag: declared statically so a Trace serialized
    # BEFORE any receptive-field probe has the same class spec as one
    # serialized after (the probe used to setdefault this at call time).
    "_tl_rf_probe_active": FieldPolicy.DROP,
    "_active_backward_pass_number": FieldPolicy.DROP,
    "_backward_roots_by_pass": FieldPolicy.DROP,
    "_backward_projection_event_count": FieldPolicy.DROP,
    "_backward_projection_revision": FieldPolicy.DROP,
    "_backward_projection_fold_state": FieldPolicy.DROP,
    "_implicit_backward_pass_open": FieldPolicy.DROP,
    "_tl_backward_triggers_disarmed": FieldPolicy.DROP,
    # Idempotent-cleanup sentinel: stamped by cleanup() so a
    # second cleanup() early-returns and every reader of a husked trace
    # funnels into the one typed TraceCleanedUpError. Session-time by
    # construction -- a load rebinds a live trace, never a husk.
    "_tl_cleaned_up": FieldPolicy.DROP,
    "_provenance_start_time": FieldPolicy.KEEP,
    "_provenance_end_time": FieldPolicy.KEEP,
    "_phase_timings": FieldPolicy.KEEP,
    "_provenance_setup_duration": FieldPolicy.KEEP,
    "_cost_forward_duration": FieldPolicy.KEEP,
    "_provenance_cleanup_duration": FieldPolicy.KEEP,
    "_cost_func_calls_duration": FieldPolicy.KEEP,
    "_autograd_has_backward_pass": FieldPolicy.KEEP,
    "_grad_fn_logs": FieldPolicy.KEEP,
    "_autograd_grad_fn_order": FieldPolicy.KEEP,
    "_backward_pass_logs": FieldPolicy.KEEP,
    "_grad_fn_param_refs": FieldPolicy.KEEP,
    "_grad_fn_param_refs_by_object_id": FieldPolicy.DROP,
    "_param_log_by_pid": FieldPolicy.DROP,
    "_session_param_inventory": FieldPolicy.DROP,
    "_session_buffer_inventory": FieldPolicy.DROP,
    "_session_buffer_identity": FieldPolicy.DROP,
    "_autograd_backward_root_grad_fn_object_ids": FieldPolicy.KEEP,
    "_cost_backward_durations": FieldPolicy.KEEP,
    "_num_backward_passes": FieldPolicy.KEEP,
    "_cost_backward_peak_memory": FieldPolicy.KEEP,
    "_provenance_backward_memory_backend": FieldPolicy.KEEP,
    "backward_peak_memory_reason": FieldPolicy.KEEP,
    "_backward_gradfn_refs": FieldPolicy.DROP,
    # the remaining undeclared runtime Trace attrs the widened
    # lockstep gate finds across the 30-axis postprocess matrix. All are
    # session-time by construction; declaring them makes the gate's
    # authority the policy table rather than the pre-spec allowance in
    # `_io/scrub.py`.
    #
    # Predicate-intervention dedup caches (INTERVENED axis). `..._target_keys` pins
    # the live intervention spec, so it is also dropped at the postprocess seam.
    "_tl_predicate_intervention_spec_keys": FieldPolicy.DROP,
    "_tl_predicate_intervention_target_keys": FieldPolicy.DROP,
    "_tl_injection_state": FieldPolicy.DROP,  # session transient
    # Streaming-bundle provenance (STREAMING axes). Already popped and
    # restored around the scrub by `_io/bundle.py` and cleared by
    # `data_classes/cleanup.py`; a load rebinds them fresh, so DROP is the
    # existing behavior made declarative.
    "_source_bundle_path": FieldPolicy.DROP,
    "_source_bundle_manifest_sha256": FieldPolicy.DROP,
    # The capture's save request: the backend reads its two-pass
    # output-parent retention flag (DEFERRED_RETENTION axis) and a legacy
    # rerun replays it.
    "_rerun_save_request": FieldPolicy.DROP,
    # Validation side-channel state. `validate_forward_pass` is
    # a public method on a user-held Trace, and
    # validation ENTRY unconditionally sets `_last_validation_failure`
    # (`reset_validation_failure` writes None on every run), so a plain
    # validate-then-save sequence hit
    # `ArtifactError: Trace._last_validation_failure is missing from
    # _PORTABLE_STATE_SPEC` -- a hard refusal on a completely ordinary
    # workflow. Both attrs are session-time diagnostics carried on the
    # trace as a side channel, exactly the `_fast_run_session` class, and
    # `ValidationFailure`/`ValidationDiagnostic` are not portable records.
    "_last_validation_failure": FieldPolicy.DROP,
    "_validation_diagnostics": FieldPolicy.DROP,
    # Session-time semantic-output scratch. Written at capture
    # entry, consumed only by ``decode_outputs_for_trace``, and dropped on
    # every settlement path (``_capture/trace.py``'s
    # ``_drop_semantic_output_transients`` plus the postprocess seam).
    # Declared here so the runtime-declaration lockstep gate can SEE them
    # on the halted axis and so the portable scrub is policy-driven rather
    # than a hand-maintained name list in ``_io/scrub.py``:
    # ``_output_tokenizer`` holds a LIVE user tokenizer and
    # ``_semantic_output_metadata`` a model-derived key, so a surviving
    # copy leaks vocab/merges into a plain pickle of the Trace.
    "_output_tokenizer": FieldPolicy.DROP,
    "_semantic_output_metadata": FieldPolicy.DROP,
}
