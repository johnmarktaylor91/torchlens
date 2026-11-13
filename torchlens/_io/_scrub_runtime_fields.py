"""Trace fields a portable save omits as runtime-only interpreter or visualization state.

Relocated from ``scrub._is_runtime_only_trace_field`` so ``scrub.py`` stays inside its file cap;
the predicate and its callers are unchanged.
"""

from __future__ import annotations

from typing import Final

__all__ = ["RUNTIME_ONLY_TRACE_FIELDS"]

RUNTIME_ONLY_TRACE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "_had_unattributed_tensor_args",
        "_module_boundary_adoptions",
        "_module_boundary_outside_sources",
        "_orphan_unattributed_tensor_args",
        "_plain_scalar_escape_labels",
        "_plain_direct_escape_gaps",
        "_foreign_departures",
        "_unrecorded_operator_mutations",
        "_torch_file_loads",
        "_last_sibling_ordering_decision",
        # Sibling render diagnostic (same _render_dot write site); unenrolled, one draw() broke saves.
        "_last_encoding_state",
        # Layout-execution geometry record (vizmech), same write site class.
        "_last_render_geometry",
        "_pending_container_collapse_nodes",
        "_defer_streaming_bundle_finalization",
        "_capture_producer_policy",
        "_capture_config",
        "_stop_directive",
        "_keep_outs_in_memory",
        "_capture_container_structure",
        "_capture_output_structure",
        "_out_sink",
        "_out_writer",
        "_container_ordinals_by_output_op_label",
        "_container_ordinals_by_input_func_call_id",
        "_validation_replay_status",
        "_capture_parent_edge_truth",
        "_orphan_pruned_func_call_ids",
        # `_tl_predicate_intervention_{spec,target}_keys`,
        # `_source_bundle_{path,manifest_sha256}` and the output-parent retention
        # flag (now inside `_rerun_save_request`) moved out of this allowance into
        # declared `Trace._PORTABLE_STATE_SPEC` DROP rows. This allowance is consulted
        # BEFORE the spec, so an entry here makes the declared policy dead code.
        "jax_equation_captures",
        "jax_ordered_captures",
        "jax_region_captures",
        "_jax_capture_index_to_raw_op_label",
        "jax_capture_index_to_final_op_label",
        "jax_outvar_key_to_capture_index",
        "_selective_save_hidden_payloads",
        # the four semantic-output scratch names moved OUT of this
        # allowance and into `Trace._PORTABLE_STATE_SPEC` as declared
        # `FieldPolicy.DROP` rows. This allowance is consulted BEFORE the spec,
        # so an entry here makes the declared policy dead code -- a policy
        # flipped to KEEP would still be silently dropped. Declared rows keep
        # one authority.
        "tinygrad_payload_policy",
        "tinygrad_uop_captures",
        "_mlx_op_captures",
        "_mlx_replay_inventory",
        "_mlx_perturbation_gaps",
        # The TF static FuncGraph path stamps both after capture and never removes them.
        "_tf_static_region_labels",
        "_tf_static_fallback_error",
        # Paddle capture-mode state; the capture pops all three before returning.
        "_paddle_stream_state",
        "_paddle_capture_mode",
        "_paddle_retention_state",
    }
)
