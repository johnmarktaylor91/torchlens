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
from time import perf_counter
from typing import Any

from .._utils.display import _vtimed
from ..errors._validation import MetadataInvariantError
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
        per-step timing bucket in ``_phase_timings`` (written by
        ``run_pipeline`` for every timed step; the step name is the persisted
        bucket key) is not listed.
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
            "_capture_events",
            "_graph_op_equivalence_classes",
            "_options_activation_transform",
            "_options_save_rng",
            "_options_save_source_context",
            "_out_writer",
            "_raw_graph_ws",
            "_trace_core",
            "num_context_lines",
            "output_layers",
        ),
        writes=(
            "_graph_op_equivalence_classes",
            "_layer_list",
            "_op_accessor_cache",
            "_raw_graph_ws",
            "_trace_core",
            "output_layers",
        ),
        postcondition=ex._add_output_layers_violation,
    ),
    PipelineStep(
        name="find_output_ancestors",
        module="_postprocess.graph_traversal",
        run=ex._run_find_output_ancestors,
        reads=("_capture_events", "_raw_graph_ws", "_tracing_finished"),
        writes=("_layer_list",),
    ),
    PipelineStep(
        name="remove_orphan_nodes",
        module="_postprocess.graph_traversal",
        run=ex._run_remove_orphan_nodes,
        reads=(
            "_capture_events",
            "_foreign_departures",
            "_foreign_kernel_interior_labels",
            "_layer_logs",
            "_options_keep_orphans",
            "_plain_scalar_escape_labels",
            "_raw_graph_ws",
            "_runnable",
            "_tracing_finished",
            "buffer_layers",
            "input_layers",
            "internal_sink_ops",
            "internally_terminated_bool_ops",
            "is_stopped",
            "output_layers",
        ),
        writes=(
            "_foreign_departures",
            "_foreign_kernel_interior_labels",
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_branch_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_graph_op_equivalence_classes",
            "_layer_list",
            "_layer_logs",
            "_orphan_labels",
            "_orphan_logs",
            "_orphan_pruned_func_call_ids",
            "_orphan_records",
            "_orphan_unattributed_tensor_args",
            "_raw_graph_ws",
            "buffer_layers",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layers_with_params",
            "output_layers",
            "state_history",
        ),
    ),
    PipelineStep(
        name="mark_layer_depths",
        module="_postprocess._layer_depths",
        run=ex._run_mark_layer_depths,
        reads=(
            "_capture_events",
            "_options_mark_layer_depths",
            "_raw_graph_ws",
            "_tracing_finished",
            "input_layers",
            "output_layers",
        ),
        writes=("_layer_list",),
        should_run=ex._should_run_mark_layer_depths,
    ),
    PipelineStep(
        name="mark_conditional_branches",
        module="_postprocess.control_flow",
        run=ex._run_mark_conditional_branches,
        reads=(
            "_capture_events",
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_branch_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_raw_graph_ws",
            "_tracing_finished",
            "internally_terminated_bool_ops",
        ),
        writes=(
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_branch_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_layer_list",
            "internally_terminated_bool_ops",
        ),
    ),
    PipelineStep(
        name="fix_buffer_layers",
        module="_postprocess.control_flow",
        run=ex._run_fix_buffer_layers,
        reads=(
            "_capture_events",
            "_options_structure_only",
            "_raw_graph_ws",
            "_tracing_finished",
            "buffer_layers",
        ),
        writes=(
            "_buffer_num_calls",
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_branch_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_graph_op_equivalence_classes",
            "_layer_list",
            "_layer_logs",
            "_raw_graph_ws",
            "buffer_layers",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layers_with_params",
            "output_layers",
        ),
    ),
    PipelineStep(
        name="group_recurrent_layers",
        module="_postprocess.loop_detection",
        run=ex._run_group_recurrent_layers,
        reads=(
            "_capture_events",
            "_options_grouping",
            "_options_recurrence_detection",
            "_raw_graph_ws",
            "_tracing_finished",
            "input_layers",
            "internal_source_ops",
        ),
        writes=("_layer_list", "_options_grouping_policy"),
    ),
    PipelineStep(
        name="map_final_labels",
        module="_postprocess.labeling",
        run=ex._run_map_final_labels,
        reads=(
            "_capture_events",
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_branch_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_graph_op_equivalence_classes",
            "_graph_stop_frontier",
            "_intervention_spec",
            "_module_capture_ws",
            "_raw_graph_ws",
            "_tracing_finished",
            "buffer_layers",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layers_with_params",
            "state_history",
            "stop_reason",
        ),
        writes=(
            "_backward_projection_event_count",
            "_backward_projection_fold_state",
            "_backward_projection_revision",
            "_final_to_raw_layer_labels",
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_branch_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_graph_op_equivalence_classes",
            "_graph_stop_frontier",
            "_intervention_spec",
            "_layer_list",
            "_module_capture_ws",
            "_raw_to_final_layer_labels",
            "_raw_to_final_op_labels",
            "_raw_to_final_parent_layer_labels",
            "buffer_layers",
            "input_layers",
            "internal_sink_ops",
            "internal_source_ops",
            "internally_terminated_bool_ops",
            "layers_with_params",
            "state_history",
            "stop_reason",
        ),
        postcondition=ex._map_final_labels_violation,
    ),
    PipelineStep(
        name="log_final_layer_info",
        module="_postprocess.labeling",
        run=ex._run_log_final_layer_info,
        reads=(
            "_capture_events",
            "_module_capture_ws",
            "_num_ops",
            "_options_verbose",
            "_param_logs",
            "_raw_graph_ws",
            "_raw_to_final_layer_labels",
            "_raw_to_final_op_labels",
            "_raw_to_final_parent_layer_labels",
            "_tracing_finished",
            "output_layers",
        ),
        writes=(
            "_cost_func_calls_duration",
            "_cost_total_activation_memory",
            "_cost_total_param_memory",
            "_layer_list",
            "_module_capture_ws",
            "_num_layers_with_params",
            "_num_ops",
            "_num_param_tensors",
            "num_params",
            "num_params_frozen",
            "num_params_trainable",
            "output_layers",
        ),
    ),
    PipelineStep(
        name="build_lookup_keys",
        module="_postprocess.labeling",
        run=ex._run_build_lookup_keys,
        reads=(
            "_buffer_num_calls",
            "_capture_events",
            "_layer_dict_all_keys",
            "_layer_list",
            "_lookup_keys_to_layer_num_dict",
            "_module_boundary_adoptions",
            "_module_boundary_outside_sources",
            "_module_capture_ws",
            "_num_saved_ops",
            "_options_mode",
            "_orphan_unattributed_tensor_args",
            "_plain_direct_escape_gaps",
            "_plain_scalar_escape_labels",
            "_provenance_backend",
            "_raw_graph_ws",
            "_torch_file_loads",
            "_tracing_finished",
            "_unrecorded_operator_mutations",
            "annotations",
            "module_calls",
        ),
        writes=(
            "_ambiguous_lookup_keys",
            "_cost_pinned_autograd_memory",
            "_cost_saved_activation_memory",
            "_had_unattributed_tensor_args",
            "_layer_dict_all_keys",
            "_layer_dict_main_keys",
            "_layer_list",
            "_layer_num_calls",
            "_layer_num_to_lookup_keys_dict",
            "_layers_logged",
            "_layers_saved",
            "_lookup_keys_to_layer_num_dict",
            "_module_boundary_adoptions",
            "_module_boundary_outside_sources",
            "_module_call_accessor",
            "_num_saved_ops",
            "_orphan_unattributed_tensor_args",
            "_plain_direct_escape_gaps",
            "_plain_scalar_escape_labels",
            "_torch_file_loads",
            "_unrecorded_operator_mutations",
            "annotations",
            "layer_labels",
            "op_labels",
        ),
        postcondition=ex._build_lookup_keys_violation,
    ),
    PipelineStep(
        name="populate_source_var_names",
        module="_postprocess.source_provenance",
        run=ex._run_populate_source_var_names,
        reads=("_layer_list", "_options_save_source_context", "_options_verbose"),
        writes=("_layer_list",),
    ),
    PipelineStep(
        name="resolve_deferred_retention",
        module="_capture.session",
        run=ex._run_resolve_deferred_retention,
        reads=(
            "_deferred_gradient_selector",
            "_deferred_retention_selector",
            "_layer_dict_all_keys",
            "_layer_list",
            "_options_activation_transform",
            "_options_backward_ready",
            "_options_verbose",
            "output_layers",
        ),
        writes=(
            "_cost_pinned_autograd_memory",
            "_cost_saved_activation_memory",
            "_deferred_gradient_selector",
            "_deferred_retention_selector",
            "_grad_op_nums_to_save",
            "_installing_deferred_gradient_hooks",
            "_layer_list",
            "_layer_nums_to_save",
            "_num_saved_ops",
            "_tl_save_selector_fire_count",
        ),
        should_run=ex._should_run_resolve_deferred_retention,
    ),
    PipelineStep(
        name="undecorate_saved_tensors",
        module="_postprocess.finalization",
        run=ex._run_undecorate_saved_tensors,
        reads=("_layer_list", "graph"),
        writes=(),
    ),
    PipelineStep(
        name="clear_cuda_cache",
        module="_postprocess._executor",
        run=ex._run_clear_cuda_cache,
        reads=("_layer_list", "_provenance_forward_memory_backend"),
        writes=(),
    ),
    PipelineStep(
        name="log_time_elapsed",
        module="_postprocess.finalization",
        run=ex._run_log_time_elapsed,
        reads=("_cost_forward_duration", "_provenance_setup_duration", "_provenance_start_time"),
        writes=("_provenance_cleanup_duration", "_provenance_end_time"),
    ),
    PipelineStep(
        name="finalize_param_logs",
        module="_postprocess.finalization",
        run=ex._run_finalize_param_logs,
        reads=("_layer_list", "_param_logs", "_trace_core"),
        writes=("_layer_list", "_param_logs", "_trace_core"),
    ),
    PipelineStep(
        name="build_layer_logs",
        module="_postprocess.finalization",
        run=ex._run_build_layer_logs,
        reads=(
            "_graph_conditional_arm_entry_edges",
            "_graph_conditional_edge_call_indices",
            "_graph_conditional_records",
            "_layer_list",
            "_options_verbose",
        ),
        writes=(
            "_by_pass",
            "_cost_total_autograd_memory",
            "_graph_conditional_edge_call_indices",
            "_graph_conditionals",
            "_layer_list",
            "_layer_logs",
        ),
        postcondition=ex._build_layer_logs_violation,
    ),
    PipelineStep(
        name="build_module_logs",
        module="_postprocess.finalization",
        run=ex._run_build_module_logs,
        reads=(
            "_buffer_initial_values",
            "_cost_forward_duration",
            "_layer_dict_all_keys",
            "_layer_list",
            "_layer_logs",
            "_module_capture_ws",
            "_module_logs",
            "_options_module_identity_mode",
            "_param_logs",
            "_param_logs_by_module",
            "_source_class_name",
            "_source_frames",
            "_source_model_ref",
            "_trace_core",
            "buffer_layers",
            "input_layers",
            "module_calls",
            "output_layers",
        ),
        writes=(
            "_buffer_accessor",
            "_layer_list",
            "_module_call_accessor",
            "_module_capture_ws",
            "_module_logs",
            "_module_pending",
            "_op_accessor_cache",
            "_param_logs_by_module",
            "_trace_core",
            "_uncalled_module_addresses",
        ),
        postcondition=ex._build_module_logs_violation,
    ),
    PipelineStep(
        name="hash_graph_shape",
        module="_utils.hashing",
        run=ex._run_hash_graph_shape,
        reads=("_layer_list", "_options_verbose"),
        writes=("_layer_list", "_provenance_fingerprint"),
        postcondition=ex._hash_graph_shape_violation,
    ),
    PipelineStep(
        name="adopt_containers_and_drop_workspaces",
        module="_postprocess._executor",
        run=ex._run_adopt_containers_and_drop_workspaces,
        reads=("_wrapper_runtime_ws",),
        writes=(
            "_capture_events",
            "_containers",
            "_module_capture_ws",
            "_output_container_specs_by_raw_label",
            "_raw_graph_ws",
            "_wrapper_runtime_ws",
        ),
    ),
    PipelineStep(
        name="seal_op_store",
        module="_postprocess",
        run=ex._run_seal_op_store,
        reads=("_layer_list", "_trace_core"),
        writes=("_trace_core",),
    ),
    PipelineStep(
        name="mark_tracing_finished",
        module="_postprocess.finalization",
        run=ex._run_mark_tracing_finished,
        reads=("_layer_dict_main_keys",),
        writes=("_layer_list", "_tracing_finished"),
        postcondition=ex._mark_tracing_finished_violation,
    ),
    PipelineStep(
        name="finalize_streamed_bundle",
        module="_postprocess.finalization",
        run=ex._run_finalize_streamed_bundle,
        reads=(
            "_defer_streaming_bundle_finalization",
            "_layer_list",
            "_options_save_args",
            "_options_save_rng",
            "_out_writer",
            "_param_logs",
        ),
        writes=(
            "_defer_streaming_bundle_finalization",
            "_layer_list",
            "_out_writer",
            "_source_bundle_manifest_sha256",
            "_source_bundle_path",
        ),
        should_run=ex._should_run_finalize_streamed_bundle,
    ),
    PipelineStep(
        name="evict_streamed_outs",
        module="_postprocess.finalization",
        run=ex._run_evict_streamed_outs,
        reads=("_keep_outs_in_memory", "_layer_list"),
        writes=("_layer_list",),
        should_run=ex._should_run_evict_streamed_outs,
    ),
    PipelineStep(
        name="release_param_refs",
        module="_data_classes._trace_removal",
        run=ex._run_release_param_refs,
        reads=("_param_logs",),
        writes=("_param_logs",),
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
        raise MetadataInvariantError(
            "postprocess_step_postcondition",
            f"{violation}. Remedy: report this as a TorchLens bug.",
            code="postprocess_step_postcondition",
        )


def _record_step_timing(trace: Any, bucket: str, elapsed_s: float) -> None:
    """Accumulate one step-timing sample (the effect of ``_record_phase_timing``).

    A step bucket gets one sample per capture, so it is looked up with ``get``
    rather than the per-op path's ``try`` that raises a ``KeyError`` on a first sample.
    """

    timings = trace.__dict__.setdefault("_phase_timings", {})
    bucket_stats = timings.get(bucket)
    if bucket_stats is None:
        bucket_stats = timings.setdefault(bucket, {"total_s": 0.0, "count": 0})
    bucket_stats["total_s"] += elapsed_s
    bucket_stats["count"] += 1


def run_pipeline(ctx: StepContext) -> None:
    """Execute ``PIPELINE`` in order, skipping steps whose gate is false.

    A gate other than ``_always`` is evaluated once per step. Each timed step
    (``_executor.STEP_DESCRIPTIONS``) records its elapsed time in the trace's
    ``_phase_timings`` under the step name, also when it raises; in verbose
    mode it prints its progress line through ``_vtimed``. After each step that
    runs, its postcondition (when it declares one) is checked by
    ``_check_step_postcondition``.
    """

    trace = ctx.trace
    verbose = trace._options_verbose
    timed = ex.STEP_DESCRIPTIONS
    for step in PIPELINE:
        if step.should_run is not _always and not step.should_run(ctx):
            continue
        name = step.name
        if name not in timed:
            step.run(ctx)
        elif verbose:
            with _vtimed(trace, ex.step_description(name, ctx), bucket=name):
                step.run(ctx)
        else:
            start = perf_counter()
            try:
                step.run(ctx)
            finally:
                _record_step_timing(trace, name, perf_counter() - start)
        if step.postcondition is not None:
            _check_step_postcondition(ctx, step)
