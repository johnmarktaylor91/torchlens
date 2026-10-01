"""Regression tests for final Trace field ownership."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.constants import MODEL_LOG_FIELD_ORDER


def test_trace_field_set_subset_of_user_facing() -> None:
    """Assert final Trace objects carry no capture-only scratch fields.

    Returns
    -------
    None
        Fails if ``Trace.__dict__`` includes post-M6 capture scratch.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))

    allowed_runtime_useful = {
        "_buffer_accessor",
        "_buffer_initial_values",
        "_buffer_write_tracker",
        "_intervention_spec",
        "_layer_nums_to_save",
        "_grad_op_nums_to_save",
        "_source_model_ref",
        "_transform",
        "_output_transform",
        "_grad_transform",
        "_optimizer",
        "_visualizer_dir",
        "_pre_forward_rng_states",
        "_source_code_blob",
        "_module_logs",
        "_param_logs_by_module",
        "_saved_grad_labels",
        "_activation_transform_repr",
        "_out_hash_cache",
        "_has_direct_writes",
        "_warned_direct_write",
        "_warned_mutate_in_place",
        "_spec_revision",
        "_out_recipe_revision",
        "_append_sequence_id",
        "_last_hook_handle_ids",
        "_grad_fn_param_refs",
        "_param_log_by_pid",
        "_capture_events",
        "_tl_backward_hooked_tensor_keys",
        # One-gradient-owner-per-label bookkeeping (FieldPolicy.DROP,
        # trace.py PORTABLE_STATE_SPEC): session-only, stripped from pickle
        # state, reset by refresh rebind and replay resets.
        "_tl_grad_hook_owner_by_label",
        "_backward_gradfn_refs",
        # Backward projection guard trio: session-only fold watermark, all
        # declared FieldPolicy.DROP in Trace.PORTABLE_STATE_SPEC and stripped
        # by __getstate__/__setstate__ (never pickled, never portable).
        "_backward_projection_event_count",
        "_backward_projection_revision",
        "_backward_projection_fold_state",
        "_halt_returns_partial_trace",
        # F20 brainpipe D-7: session-time (live, resident) peak pair
        # (FieldPolicy.DROP; served by the forward_peak_memory_pair property;
        # loaded artifacts never re-measure).
        "_forward_peak_memory_pair",
        "_phase_timings",
        "_postprocessing_active",
        "_raw_event_shape_hash",
        "_replay_arg_version_data_complete",
        "_capture_config",
        "_raw_transform_escape_detected",
        # Honesty/safety phase: the bypassed-torch.compile-region marker that owns
        # the top-precedence capture_verification_reason. Runtime-only and
        # FieldPolicy.DROP for the same reason as the transform-escape flag above.
        "_raw_dynamo_region_detected",
        # Honesty/safety phase: the consolidated warn-once key set (nonfinite
        # check unavailable under raise_on_nan; implicit backward pass).
        "_warned_once",
        "_stop_directive",
        # r65-r81 buffer-rung/RNG-registry sprint: capture-scratch that
        # legitimately survives on a finished Trace (session-scoped
        # param/buffer inventories and storage-address maps used by the
        # buffer-rung witness/completeness machinery). Runnable state now has
        # one declared MODEL_LOG_FIELD_ORDER owner: ``_runnable``.
        "_param_storage_addresses",
        "_buffer_storage_addresses",
        "_orphan_pruned_func_call_ids",
        # r29 F3b: capture-time (slot -> producer) edge truth sealed for the
        # capture_edge_survival metadata invariant; runtime-only, never
        # persisted (registered in _io/scrub.py's runtime-only list).
        "_capture_parent_edge_truth",
        "_session_param_inventory",
        "_session_buffer_inventory",
        "_session_buffer_identity",
        # Session-time knobs and lazily built accessor caches that Trace
        # DECLARES with an explicit ``FieldPolicy.DROP`` (see
        # ``data_classes/trace.py``): deliberately live on a finished Trace and
        # deliberately absent from MODEL_LOG_FIELD_ORDER because they do not
        # survive save/load. ``_module_call_accessor`` is the exact sibling of
        # the allowlisted ``_buffer_accessor``; ``measure_python_peak_memory``
        # is the documented tracemalloc opt-in read back by capture/trace.py;
        # ``save_budget`` and its live accountant are the honesty/safety
        # phase's per-device retained-bytes ceiling.
        "_module_call_accessor",
        "measure_python_peak_memory",
        "distributed_witness",
        "save_budget",
        "_save_budget_accountant",
        # Session-time per-op finiteness recording knob (the exact sibling of
        # ``measure_python_peak_memory``): declared ``FieldPolicy.DROP`` with
        # component owner "capture_config"; read back by the op-finalize hook
        # and never persisted.
        "track_nonfinite",
        # M5 Op seam: the per-trace columnar row store backing every Op
        # facade. Declared ``FieldPolicy.DROP`` with a component owner
        # (``_trace_components.py`` -> "graph"); plain pickle re-materializes
        # each Op as a detached row, so the store never persists.
        "_trace_core",
        # Per-trace op accessor cache (sibling of ``_module_call_accessor``):
        # declared ``FieldPolicy.DROP`` with component owner "graph" and
        # popped in ``__getstate__``, so it never persists.
        "_op_accessor_cache",
        # tlspec v8 bump: three unordered runtime rows made portable
        # (FieldPolicy.KEEP, deliberately absent from MODEL_LOG_FIELD_ORDER --
        # the portable_only_fields ledger in test_field_order_contract.py):
        # the L3 primitive-op profile and the two L9 backward-residual
        # disclosure markers.
        "_primitive_op_profile",
        "checkpoint_invocation_witness",
        "grad_fn_timing_provenance",
        # T85 re-reconcile union: landed session-time FieldPolicy.DROP fields
        # (RUNTIME storage bindings in _schema_bindings.py). The autograd
        # saved-band pair and the device-memory knob are F27 torchnative
        # session scratch; ``_tl_injection_state`` is the F01 log_injections
        # session transient.
        "_autograd_saved_bands",
        "_autograd_seen_saved_storages",
        "track_device_memory",
        "_tl_injection_state",
    }

    actual = set(trace.__dict__.keys())
    canonical = set(MODEL_LOG_FIELD_ORDER) | allowed_runtime_useful
    extra = actual - canonical
    assert not extra, f"Trace carries unexpected fields: {extra}"

    gone_fields = [
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_layer_counter",
        "_raw_layer_type_counter",
        "_current_func_barcode",
        "_mod_entered",
        "_mod_exited",
        "_mod_call_index",
        "_mod_call_labels",
        "_module_build_data",
        "_module_metadata",
        "_module_forward_args",
        "_module_containment_engine",
        "_exhaustive_module_stack",
        "_grad_fn_strong_refs",
        "_in_exhaustive_pass",
        "_pending_live_fire_records",
        "_output_container_specs_by_raw_label",
        "_out_writer",
        "_out_sink",
        "_keep_outs_in_memory",
        "_keep_grads_in_memory",
        "_defer_streaming_bundle_finalization",
    ]
    for field in gone_fields:
        assert field not in actual, f"Capture-only field {field!r} still on Trace"


def test_public_cached_property_read_does_not_poison_save(tmp_path) -> None:
    """Assert reading ``Trace.intervention_spec`` never breaks ``tl.save``.

    ``intervention_spec`` is a ``functools.cached_property``, so a read caches
    its value under its own name in ``Trace.__dict__``. The tlspec scrub walks
    the live ``__dict__`` fail-closed, and before B3R4-R10-1 the undeclared
    cache key made every later ``tl.save`` raise ``TorchLensIOError`` -- a
    read-only public accessor permanently poisoning the artifact path.

    Parameters
    ----------
    tmp_path:
        Pytest-provided temporary directory.

    Returns
    -------
    None
        Fails if the post-read save raises or the artifact does not load.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4))

    spec = trace.intervention_spec
    assert spec is not None
    assert "intervention_spec" in trace.__dict__

    save_path = tmp_path / "after_read.tlspec"
    tl.save(trace, str(save_path))

    loaded = tl.load(str(save_path))
    # The cache is derived state: it must rebuild on the loaded trace, not
    # round-trip as a portable field.
    assert "intervention_spec" not in loaded.__dict__
    assert loaded.intervention_spec is not None
