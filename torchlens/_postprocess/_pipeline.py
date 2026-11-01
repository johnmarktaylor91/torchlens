"""Postprocess pipeline: the named ordered step table and ``run_pipeline``.

``PIPELINE`` is one tuple of ``PipelineStep`` records in execution order. A
step is identified by its name, not a rank: its position in the tuple is the
order, and nothing else numbers the steps. Each record names the module that
owns the step body, the trace fields the step reads and the trace fields it
writes, an optional gate (``should_run``, evaluated exactly once per step per
run) and an optional postcondition checked right after the step runs.

Step order is load-bearing (warnings and first-exception order depend on it)
and is verified by the byte-identity oracles. The materialize prologue in
``postprocess`` runs before the table and is deliberately absent from it.

The step adapters, gates and postconditions the table points at live in
``_executor``; the loop below is the only caller of the table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import _executor as ex
from ._executor import StepContext


def _always(ctx: StepContext) -> bool:
    """Default gate: the step always executes."""

    return True


@dataclass(frozen=True)
class PipelineStep:
    """One executable pipeline step.

    Attributes
    ----------
    name:
        The step's name; unique in ``PIPELINE`` and the key of its
        postcondition and timing bucket.
    module:
        Dotted module path (under ``torchlens``) that owns the step body.
        ``_postprocess._executor`` marks a step implemented inline in the executor.
    run:
        The step body.
    reads:
        Trace attributes the step reads (its own writes excluded).
    writes:
        Trace attributes the step assigns, mutates in place or drops. The
        per-step timing bucket in ``_phase_timings`` (written by ``_vtimed``
        for every timed step; its text is the persisted bucket key) is not
        listed.
    should_run:
        Gate evaluated EXACTLY ONCE per step per run, in pipeline order
        (executor invariant, counting-tested).
    postcondition:
        Returns the violation message when the finished step left its
        contract unmet, else ``None``.
    """

    name: str
    module: str
    run: Callable[[StepContext], None]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    should_run: Callable[[StepContext], bool] = _always
    postcondition: Callable[[StepContext], str | None] | None = None


PIPELINE: tuple[PipelineStep, ...] = (
    PipelineStep(
        name="add_output_layers",
        module="_postprocess.graph_traversal",
        run=ex._run_add_output_layers,
        reads=(
            "_out_writer",
            "_raw_graph_ws",
            "_trace_core",
            "activation_transform",
            "capture_events",
            "num_context_lines",
            "op_equivalence_classes",
            "output_layers",
            "save_code_context",
            "save_rng_states",
        ),
        writes=(
            "_raw_graph_ws",
            "_trace_core",
            "layer_list",
            "op_equivalence_classes",
            "output_layers",
        ),
        postcondition=ex._add_output_layers_violation,
    ),
    PipelineStep(
        name="find_output_ancestors",
        module="_postprocess.graph_traversal",
        run=ex._run_find_output_ancestors,
        reads=("_raw_graph_ws", "_tracing_finished", "capture_events"),
        writes=("layer_list",),
    ),
    PipelineStep(
        name="remove_orphan_nodes",
        module="_postprocess.graph_traversal",
        run=ex._run_remove_orphan_nodes,
        reads=(
            "_foreign_kernel_interior_labels",
            "_plain_scalar_escape_labels",
            "_raw_graph_ws",
            "_runnable",
            "_tracing_finished",
            "buffer_layers",
            "capture_events",
            "halted",
            "input_layers",
            "internal_sink_ops",
            "internally_terminated_bool_ops",
            "keep_orphans",
            "layer_logs",
            "output_layers",
        ),
        writes=(
            "_foreign_kernel_interior_labels",
            "_orphan_labels",
            "_orphan_logs",
            "_orphan_pruned_func_call_ids",
            "_orphan_unattributed_tensor_args",
            "_raw_graph_ws",
            "buffer_layers",
            "conditional_arm_entry_edges",
            "conditional_branch_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layer_list",
            "layer_logs",
            "layers_with_params",
            "op_equivalence_classes",
            "orphan_records",
            "output_layers",
        ),
    ),
    PipelineStep(
        name="mark_layer_depths",
        module="_postprocess._layer_depths",
        run=ex._run_mark_layer_depths,
        reads=(
            "_raw_graph_ws",
            "_tracing_finished",
            "capture_events",
            "input_layers",
            "mark_layer_depths",
            "output_layers",
        ),
        writes=("layer_list",),
        should_run=ex._should_run_mark_layer_depths,
    ),
    PipelineStep(
        name="mark_conditional_branches",
        module="_postprocess.control_flow",
        run=ex._run_mark_conditional_branches,
        reads=(
            "_raw_graph_ws",
            "_tracing_finished",
            "capture_events",
            "conditional_arm_entry_edges",
            "conditional_branch_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "internally_terminated_bool_ops",
        ),
        writes=(
            "conditional_arm_entry_edges",
            "conditional_branch_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "internally_terminated_bool_ops",
            "layer_list",
        ),
    ),
    PipelineStep(
        name="fix_buffer_layers",
        module="_postprocess.control_flow",
        run=ex._run_fix_buffer_layers,
        reads=(
            "_raw_graph_ws",
            "_tracing_finished",
            "buffer_layers",
            "capture_events",
            "structure_only",
        ),
        writes=(
            "_raw_graph_ws",
            "buffer_layers",
            "buffer_num_calls",
            "conditional_arm_entry_edges",
            "conditional_branch_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layer_list",
            "layer_logs",
            "layers_with_params",
            "op_equivalence_classes",
            "output_layers",
        ),
    ),
    PipelineStep(
        name="group_recurrent_layers",
        module="_postprocess.loop_detection",
        run=ex._run_group_recurrent_layers,
        reads=(
            "_raw_graph_ws",
            "_tracing_finished",
            "capture_events",
            "grouping",
            "input_layers",
            "internal_source_ops",
            "recurrence_detection",
        ),
        writes=("grouping_policy", "layer_list"),
    ),
    PipelineStep(
        name="map_final_labels",
        module="_postprocess.labeling",
        run=ex._run_map_final_labels,
        reads=("_raw_graph_ws", "_tracing_finished", "capture_events"),
        writes=(
            "_backward_projection_event_count",
            "_backward_projection_fold_state",
            "_backward_projection_revision",
            "_final_to_raw_layer_labels",
            "_raw_to_final_layer_labels",
            "_raw_to_final_op_labels",
            "_raw_to_final_parent_layer_labels",
            "layer_list",
        ),
        postcondition=ex._map_final_labels_violation,
    ),
    PipelineStep(
        name="log_final_layer_info",
        module="_postprocess.labeling",
        run=ex._run_log_final_layer_info,
        reads=(
            "_module_capture_ws",
            "_raw_graph_ws",
            "_raw_to_final_layer_labels",
            "_raw_to_final_op_labels",
            "_raw_to_final_parent_layer_labels",
            "_tracing_finished",
            "capture_events",
            "num_ops",
            "output_layers",
            "param_logs",
            "verbose",
        ),
        writes=(
            "_module_capture_ws",
            "func_calls_duration",
            "layer_list",
            "num_layers_with_params",
            "num_ops",
            "num_param_tensors",
            "num_params",
            "num_params_frozen",
            "num_params_trainable",
            "total_activation_memory",
            "total_param_memory",
        ),
    ),
    PipelineStep(
        name="rename_layer_references",
        module="_postprocess.labeling",
        run=ex._run_rename_layer_references,
        reads=(
            "_intervention_spec",
            "_module_capture_ws",
            "_raw_graph_ws",
            "_raw_to_final_layer_labels",
            "_raw_to_final_op_labels",
            "_raw_to_final_parent_layer_labels",
            "_tracing_finished",
            "buffer_layers",
            "capture_events",
            "conditional_arm_entry_edges",
            "conditional_branch_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "halt_frontier",
            "halt_reason",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layers_with_params",
            "op_equivalence_classes",
            "output_layers",
            "state_history",
        ),
        writes=(
            "_intervention_spec",
            "_module_capture_ws",
            "buffer_layers",
            "conditional_arm_entry_edges",
            "conditional_branch_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "halt_frontier",
            "halt_reason",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layers_with_params",
            "op_equivalence_classes",
            "output_layers",
            "state_history",
        ),
    ),
    PipelineStep(
        name="build_lookup_keys",
        module="_postprocess.labeling",
        run=ex._run_build_lookup_keys,
        reads=(
            "_lookup_keys_to_layer_num_dict",
            "_module_boundary_adoptions",
            "_module_boundary_outside_sources",
            "_module_capture_ws",
            "_orphan_unattributed_tensor_args",
            "_plain_direct_escape_gaps",
            "_plain_scalar_escape_labels",
            "_raw_graph_ws",
            "_tracing_finished",
            "_unrecorded_operator_mutations",
            "annotations",
            "backend",
            "buffer_num_calls",
            "capture_events",
            "capture_mode",
            "layer_dict_all_keys",
            "layer_list",
            "module_calls",
            "num_saved_ops",
        ),
        writes=(
            "_ambiguous_lookup_keys",
            "_had_unattributed_tensor_args",
            "_layer_num_to_lookup_keys_dict",
            "_layers_logged",
            "_layers_saved",
            "_lookup_keys_to_layer_num_dict",
            "_module_boundary_adoptions",
            "_module_boundary_outside_sources",
            "_orphan_unattributed_tensor_args",
            "_plain_direct_escape_gaps",
            "_plain_scalar_escape_labels",
            "_unrecorded_operator_mutations",
            "annotations",
            "layer_dict_all_keys",
            "layer_dict_main_keys",
            "layer_labels",
            "layer_list",
            "layer_num_calls",
            "num_saved_layers",
            "num_saved_module_calls",
            "num_saved_ops",
            "op_labels",
            "pinned_autograd_memory",
            "saved_activation_memory",
        ),
        postcondition=ex._build_lookup_keys_violation,
    ),
    PipelineStep(
        name="populate_source_var_names",
        module="_postprocess.source_provenance",
        run=ex._run_populate_source_var_names,
        reads=("layer_list", "save_code_context", "verbose"),
        writes=("layer_list",),
    ),
    PipelineStep(
        name="resolve_deferred_retention",
        module="_capture.session",
        run=ex._run_resolve_deferred_retention,
        reads=(
            "_deferred_gradient_selector",
            "_deferred_retention_selector",
            "activation_transform",
            "backward_ready",
            "layer_dict_all_keys",
            "layer_list",
            "output_layers",
            "verbose",
        ),
        writes=(
            "_deferred_gradient_selector",
            "_deferred_retention_selector",
            "_grad_op_nums_to_save",
            "_installing_deferred_gradient_hooks",
            "_layer_nums_to_save",
            "_tl_save_selector_fire_count",
            "layer_list",
            "num_saved_layers",
            "num_saved_module_calls",
            "num_saved_ops",
            "pinned_autograd_memory",
            "saved_activation_memory",
        ),
        should_run=ex._should_run_resolve_deferred_retention,
    ),
    PipelineStep(
        name="undecorate_saved_tensors",
        module="_postprocess.finalization",
        run=ex._run_undecorate_saved_tensors,
        reads=("graph", "layer_list"),
        writes=(),
    ),
    PipelineStep(
        name="clear_cuda_cache",
        module="_postprocess._executor",
        run=ex._run_clear_cuda_cache,
        reads=("forward_memory_backend", "layer_list"),
        writes=(),
    ),
    PipelineStep(
        name="log_time_elapsed",
        module="_postprocess.finalization",
        run=ex._run_log_time_elapsed,
        reads=("capture_start_time", "forward_duration", "setup_duration"),
        writes=("capture_end_time", "cleanup_duration"),
    ),
    PipelineStep(
        name="finalize_param_logs",
        module="_postprocess.finalization",
        run=ex._run_finalize_param_logs,
        reads=("_trace_core", "layer_list", "param_logs"),
        writes=("_trace_core", "layer_list", "param_logs"),
    ),
    PipelineStep(
        name="build_layer_logs",
        module="_postprocess.finalization",
        run=ex._run_build_layer_logs,
        reads=(
            "conditional_arm_entry_edges",
            "conditional_edge_call_indices",
            "conditional_records",
            "layer_list",
            "verbose",
        ),
        writes=(
            "by_pass",
            "conditional_edge_call_indices",
            "conditionals",
            "layer_list",
            "layer_logs",
            "total_autograd_memory",
        ),
        postcondition=ex._build_layer_logs_violation,
    ),
    PipelineStep(
        name="build_module_logs",
        module="_postprocess.finalization",
        run=ex._run_build_module_logs,
        reads=(
            "_buffer_initial_values",
            "_module_capture_ws",
            "_module_logs",
            "_param_logs_by_module",
            "_source_model_ref",
            "_trace_core",
            "buffer_layers",
            "code_context",
            "forward_duration",
            "input_layers",
            "layer_dict_all_keys",
            "layer_list",
            "layer_logs",
            "model_class_name",
            "module_calls",
            "module_identity_mode",
            "output_layers",
            "param_logs",
        ),
        writes=(
            "_buffer_accessor",
            "_module_capture_ws",
            "_module_logs",
            "_param_logs_by_module",
            "_trace_core",
            "_uncalled_module_addresses",
            "layer_list",
            "num_saved_module_calls",
        ),
        postcondition=ex._build_module_logs_violation,
    ),
    PipelineStep(
        name="hash_graph_shape",
        module="_utils.hashing",
        run=ex._run_hash_graph_shape,
        reads=("layer_list", "verbose"),
        writes=("graph_shape_hash", "layer_list"),
        postcondition=ex._hash_graph_shape_violation,
    ),
    PipelineStep(
        name="mark_tracing_finished",
        module="_postprocess.finalization",
        run=ex._run_mark_tracing_finished,
        reads=("layer_dict_main_keys",),
        writes=("_tracing_finished", "layer_list"),
        postcondition=ex._mark_tracing_finished_violation,
    ),
    PipelineStep(
        name="adopt_containers_and_drop_workspaces",
        module="_postprocess._executor",
        run=ex._run_adopt_containers_and_drop_workspaces,
        reads=("_wrapper_runtime_ws",),
        writes=(
            "_containers",
            "_module_capture_ws",
            "_output_container_specs_by_raw_label",
            "_raw_graph_ws",
            "_wrapper_runtime_ws",
            "capture_events",
        ),
    ),
    PipelineStep(
        name="finalize_streamed_bundle",
        module="_postprocess.finalization",
        run=ex._run_finalize_streamed_bundle,
        reads=(
            "_defer_streaming_bundle_finalization",
            "_out_writer",
            "layer_list",
            "param_logs",
            "save_arg_values",
            "save_rng_states",
        ),
        writes=(
            "_defer_streaming_bundle_finalization",
            "_out_writer",
            "_source_bundle_manifest_sha256",
            "_source_bundle_path",
            "layer_list",
        ),
        should_run=ex._should_run_finalize_streamed_bundle,
    ),
    PipelineStep(
        name="evict_streamed_outs",
        module="_postprocess.finalization",
        run=ex._run_evict_streamed_outs,
        reads=("_keep_outs_in_memory", "layer_list"),
        writes=("layer_list",),
        should_run=ex._should_run_evict_streamed_outs,
    ),
    PipelineStep(
        name="release_param_refs",
        module="_data_classes._trace_validation",
        run=ex._run_release_param_refs,
        reads=("param_logs",),
        writes=("param_logs",),
    ),
)


def _check_step_postcondition(ctx: StepContext, step: PipelineStep) -> None:
    """Raise if a completed step left its postcondition unmet.

    Runs on every capture, with no env gate, and raises a typed error rather
    than ``assert`` so ``python -O`` cannot strip it. A violation is a
    postprocess defect, never a user error.

    Raises
    ------
    MetadataInvariantError
        With check name and refusal code ``postprocess_step_postcondition``
        and a message naming the step.
    """

    violation = None if step.postcondition is None else step.postcondition(ctx)
    if violation is not None:
        from ..validation.invariants import MetadataInvariantError

        raise MetadataInvariantError(
            "postprocess_step_postcondition",
            f"{violation}. Remedy: report this as a TorchLens bug.",
            code="postprocess_step_postcondition",
        )


def run_pipeline(ctx: StepContext) -> None:
    """Execute ``PIPELINE`` in order, skipping steps whose gate is false.

    After each step that runs, its postcondition (when it declares one) is
    checked by ``_check_step_postcondition``.
    """

    for step in PIPELINE:
        if step.should_run(ctx):
            step.run(ctx)
            _check_step_postcondition(ctx, step)
