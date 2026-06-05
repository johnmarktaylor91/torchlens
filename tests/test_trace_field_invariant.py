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
        "_buffer_write_events",
        "_buffer_write_tracker",
        "_intervention_spec",
        "_layer_nums_to_save",
        "_grad_layer_nums_to_save",
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
        "_saved_grads_set",
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
    }

    actual = set(trace.__dict__.keys())
    canonical = set(MODEL_LOG_FIELD_ORDER) | allowed_runtime_useful
    extra = actual - canonical
    assert not extra, f"Trace carries unexpected fields: {extra}"

    gone_fields = [
        "_build_state",
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_layer_counter",
        "_raw_layer_type_counter",
        "_unsaved_layers_lookup_keys",
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
