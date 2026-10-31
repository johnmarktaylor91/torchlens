"""Field-admission gate for the v11 writer contract.

Every field the v11 contract of record carries over v10 (persisted or DROP)
must be registered here under its policy. An unregistered new persisted field
fails this test, so a field never reaches the stored shape by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V10_CONTRACT = SCHEMAS / "writer_contract_v10.json"
V11_CONTRACT = SCHEMAS / "writer_contract_v11.json"

#: The v11 FIELD REGISTRY: record contract key -> {field name -> policy value}
#: for every field added over v10. The v11 renames enter under their new
#: spellings (the old spellings leave with no alias).
V11_FIELD_ADMISSIONS: dict[str, dict[str, str]] = {
    "grad_fn_call": {"fire_record": "keep"},
    "layer": {"is_edited": "keep"},
    "op": {"is_edited": "keep"},
    "trace": {
        "_autograd_backward_root_grad_fn_object_ids": "keep",
        "_autograd_derived_grads": "keep",
        "_autograd_escape_detector_backward_coverage": "drop",
        "_autograd_grad_fn_order": "keep",
        "_autograd_has_backward_pass": "keep",
        "_autograd_has_grads": "keep",
        "_cost_backward_durations": "keep",
        "_cost_backward_peak_memory": "keep",
        "_cost_forward_duration": "keep",
        "_cost_forward_peak_memory": "keep",
        "_cost_forward_peak_memory_reason": "keep",
        "_cost_func_calls_duration": "keep",
        "_cost_pinned_autograd_memory": "drop",
        "_cost_saved_activation_memory": "keep",
        "_cost_saved_grad_memory": "keep",
        "_cost_total_activation_memory": "keep",
        "_cost_total_autograd_memory": "keep",
        "_cost_total_backward_memory": "keep",
        "_cost_total_grad_memory": "keep",
        "_cost_total_param_grad_memory": "keep",
        "_cost_total_param_memory": "keep",
        "_graph_conditional_arm_entry_edges": "keep",
        "_graph_conditional_branch_edges": "keep",
        "_graph_conditional_edge_call_indices": "keep",
        "_graph_conditional_records": "keep",
        "_graph_conditionals": "keep",
        "_graph_manual_tensor_connections": "keep",
        "_graph_op_equivalence_classes": "keep",
        "_graph_relationship_evidence": "keep",
        "_graph_stop_frontier": "keep",
        "_graph_structure_evidence": "keep",
        "_health_completeness_callback_ns": "drop",
        "_health_completeness_decompositions": "drop",
        "_health_completeness_mode": "drop",
        "_health_completeness_verified": "drop",
        "_health_escape_detector_callback_ns": "drop",
        "_health_escape_detector_mode": "drop",
        "_health_escape_detector_verified": "drop",
        "_health_num_accounted_witness_events": "drop",
        "_health_num_escape_detector_events": "drop",
        "_health_num_expected_opaque_witness_events": "drop",
        "_health_num_unaccounted_witness_events": "drop",
        "_health_num_witness_events": "drop",
        "_options_activation_transform": "drop",
        "_options_backward_ready": "drop",
        "_options_chunk_size": "keep",
        "_options_detach_saved_activations": "keep",
        "_options_emit_nvtx": "keep",
        "_options_grad_transform": "drop",
        "_options_grouping": "keep",
        "_options_grouping_policy": "keep",
        "_options_inference_only": "keep",
        "_options_input_transform": "keep",
        "_options_keep_orphans": "keep",
        "_options_mark_layer_depths": "keep",
        "_options_mode": "keep",
        "_options_module_identity_mode": "keep",
        "_options_on_nonfinite": "keep",
        "_options_output_device": "keep",
        "_options_recurrence_detection": "keep",
        "_options_save_arg_templates": "keep",
        "_options_save_args": "keep",
        "_options_save_grads": "keep",
        "_options_save_mode": "keep",
        "_options_save_raw_activations": "keep",
        "_options_save_raw_grads": "keep",
        "_options_save_raw_input": "keep",
        "_options_save_rng": "keep",
        "_options_save_source_context": "keep",
        "_options_save_transformed_output": "keep",
        "_options_seed": "keep",
        "_options_structure_only": "keep",
        "_options_verbose": "keep",
        "_provenance_backend": "keep",
        "_provenance_backend_config": "keep",
        "_provenance_backend_version": "keep",
        "_provenance_backward_memory_backend": "keep",
        "_provenance_cache_hit": "keep",
        "_provenance_cache_key": "keep",
        "_provenance_cache_path": "keep",
        "_provenance_cleanup_duration": "keep",
        "_provenance_distributed_scope": "keep",
        "_provenance_end_time": "keep",
        "_provenance_fingerprint": "keep",
        "_provenance_forward_memory_backend": "keep",
        "_provenance_input_annotations": "keep",
        "_provenance_input_object_id": "keep",
        "_provenance_input_preprocessor": "keep",
        "_provenance_input_signature_hash": "keep",
        "_provenance_input_structure": "blob_recursive",
        "_provenance_is_appended": "keep",
        "_provenance_model_object_id": "keep",
        "_provenance_observer_spans": "keep",
        "_provenance_output_postprocessor": "keep",
        "_provenance_owner_thread_id": "drop",
        "_provenance_owner_thread_qualified": "drop",
        "_provenance_param_hash_full": "keep",
        "_provenance_param_hash_quick": "keep",
        "_provenance_param_source": "keep",
        "_provenance_recording_kept": "keep",
        "_provenance_root_entry_point": "keep",
        "_provenance_setup_duration": "keep",
        "_provenance_start_time": "keep",
        "_provenance_thread_activity_detected": "drop",
        "_provenance_thread_count_end": "drop",
        "_provenance_thread_count_start": "drop",
        "_provenance_verification_reason": "keep",
        "_provenance_verified": "keep",
        "_replay_frontier": "drop",
        "_source_class_docstring": "keep",
        "_source_class_file": "keep",
        "_source_class_line": "keep",
        "_source_class_name": "keep",
        "_source_class_qualname": "keep",
        "_source_forward_docstring": "keep",
        "_source_forward_file": "keep",
        "_source_forward_line": "keep",
        "_source_forward_signature": "keep",
        "_source_frames": "keep",
        "_source_init_docstring": "keep",
        "_source_init_file": "keep",
        "_source_init_line": "keep",
        "_source_init_signature": "keep",
        "_source_snapshots": "keep",
        "facet_functions": "drop",
        "is_replayable": "keep",
        "model_name": "keep",
        "name": "keep",
        "stop_reason": "keep",
    },
}


def _record_contracts(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["contract"]["record_contracts"]


def _field_names(record: dict) -> set[str]:
    return set(record["persisted_field_names"]) | set(record["field_policies"])


def test_v10_to_v11_additions_are_all_registered() -> None:
    """Every field added over v10 (persisted or DROP) is admitted, under its registered policy.

    DROP fields never enter ``persisted_field_names`` but do enter the policy
    table, so the gate reads both.
    """

    old = _record_contracts(V10_CONTRACT)
    new = _record_contracts(V11_CONTRACT)
    assert set(old) == set(new), "no record contract may appear or disappear at v11"
    assert set(V11_FIELD_ADMISSIONS) <= set(new), "admissions name unknown record contracts"
    for record_key in sorted(new):
        added = _field_names(new[record_key]) - _field_names(old[record_key])
        admitted = V11_FIELD_ADMISSIONS.get(record_key, {})
        unregistered = added - set(admitted)
        assert unregistered == set(), (
            f"{record_key}: field(s) {sorted(unregistered)} reach the v11 contract "
            "without a registration; add the admission to V11_FIELD_ADMISSIONS in "
            "the same change"
        )
        policies = new[record_key]["field_policies"]
        for name in sorted(set(admitted) & added):
            assert policies[name] == admitted[name], (
                f"{record_key}.{name}: live policy {policies[name]!r} differs from the "
                f"registered {admitted[name]!r}"
            )


def test_v11_admissions_name_only_real_additions() -> None:
    """A registration names a field that is new in v11, never a stale or v10 field."""

    old = _record_contracts(V10_CONTRACT)
    new = _record_contracts(V11_CONTRACT)
    for record_key, admitted in sorted(V11_FIELD_ADMISSIONS.items()):
        added = _field_names(new[record_key]) - _field_names(old[record_key])
        stale = set(admitted) - added
        assert stale == set(), (
            f"{record_key}: admission(s) {sorted(stale)} are not fields added at v11"
        )
