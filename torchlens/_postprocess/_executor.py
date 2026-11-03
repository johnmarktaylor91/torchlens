"""Postprocess step adapters: ``StepContext`` and one function per pipeline step.

Each ``_run_<step name>`` function here adapts one step body to the pipeline:
it resolves the function through its home module at call time
(``graph_traversal._find_output_ancestors``, never an imported reference), so
a function replaced on its home module is the one that runs, and it wraps the
call in the step's timing bucket. ``_should_run_<step name>`` functions are the
gates of the optional steps and ``_<step name>_violation`` functions are the
postconditions. The ordered table that strings them together is
``_pipeline.PIPELINE``; ``_pipeline.run_pipeline`` is the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .._utils import _cuda_probe, hashing as _hashing, tensor_utils as _tensor_utils
from .._utils.display import _vtimed
from . import (
    _layer_depths,
    control_flow,
    finalization,
    graph_traversal,
    labeling,
    loop_detection,
    saved_summary,
    source_provenance,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


@dataclass
class StepContext:
    """Mutable per-run context threaded through the pipeline."""

    trace: Trace
    output_tensors: list[Any]
    output_tensor_addresses: list[str]
    output_parent_labels: list[Any]
    capture_session: Any | None
    #: Snapshot taken by the ``finalize_streamed_bundle`` gate: that gate IS
    #: the snapshot point. The step clears ``_out_writer``, so a re-evaluated
    #: ``evict_streamed_outs`` predicate would always be false and streamed
    #: outs would never be evicted. Context-writing, never trace-writing.
    finalize_streaming: bool | None = None


def _run_add_output_layers(ctx: StepContext) -> None:
    """Create dedicated output nodes for each attributable output tensor."""

    with _vtimed(ctx.trace, "  Step 1: Add output layers", bucket="add_output_layers"):
        graph_traversal._add_output_layers(
            ctx.trace,
            ctx.output_tensors,
            ctx.output_tensor_addresses,
            ctx.output_parent_labels,
        )


def _run_find_output_ancestors(ctx: StepContext) -> None:
    """Mark every node that is an ancestor of a model output."""

    with _vtimed(ctx.trace, "  Step 2: Trace output ancestors", bucket="find_output_ancestors"):
        graph_traversal._find_output_ancestors(ctx.trace)


def _run_remove_orphan_nodes(ctx: StepContext) -> None:
    """Drop raw nodes that reach no model output."""

    with _vtimed(ctx.trace, "  Step 3: Remove orphan nodes", bucket="remove_orphan_nodes"):
        ctx.trace.__dict__.pop("_foreign_departures", None)  # capture-time journal ends here
        graph_traversal._remove_orphan_nodes(ctx.trace)


def _run_mark_layer_depths(ctx: StepContext) -> None:
    """Write the optional input/output distance metadata."""

    with _vtimed(ctx.trace, "  Step 4: Input/output distances", bucket="mark_layer_depths"):
        _layer_depths._mark_layer_depths(ctx.trace)


def _should_run_mark_layer_depths(ctx: StepContext) -> bool:
    """Gate the depth pass on the opt-in ``Trace.options.mark_layer_depths`` flag."""

    return bool(ctx.trace._options_mark_layer_depths)


def _run_mark_conditional_branches(ctx: StepContext) -> None:
    """Attribute conditional branches (AST, bool, event, and edge evidence)."""

    with _vtimed(
        ctx.trace, "  Step 5: Mark conditional branches", bucket="mark_conditional_branches"
    ):
        control_flow._mark_conditional_branches(ctx.trace)


def _run_fix_buffer_layers(ctx: StepContext) -> None:
    """Deduplicate buffer layers and reconnect their edges."""

    with _vtimed(ctx.trace, "  Step 6: Fix buffer layers", bucket="fix_buffer_layers"):
        control_flow._fix_buffer_layers(ctx.trace)


def _run_group_recurrent_layers(ctx: StepContext) -> None:
    """Group recurrent calls into multi-pass layers.

    With ``recurrence_detection`` off this degrades to shared-parameter
    grouping only, which is the historical single-pass layout.
    """

    trace = ctx.trace
    loop_desc = (
        "  Step 7: Loop detection (full)"
        if trace._options_recurrence_detection
        else "  Step 7: Loop detection (params only)"
    )
    with _vtimed(trace, loop_desc, bucket="group_recurrent_layers"):
        if trace._options_recurrence_detection:
            loop_detection._detect_and_label_loops(trace)
        else:
            loop_detection._group_by_shared_params(trace)


def _run_map_final_labels(ctx: StepContext) -> None:
    """Build the raw-label to final-label map."""

    with _vtimed(ctx.trace, "  Step 8: Map labels", bucket="map_final_labels"):
        labeling._map_raw_labels_to_final_labels(ctx.trace)


def _run_log_final_layer_info(ctx: StepContext) -> None:
    """Write the final per-layer and per-module fields."""

    with _vtimed(ctx.trace, "  Step 9: Log final info", bucket="log_final_layer_info"):
        labeling._log_final_info_for_layers(ctx.trace)


def _run_rename_layer_references(ctx: StepContext) -> None:
    """Rename global label references to their final names."""

    with _vtimed(ctx.trace, "  Step 10: Rename labels", bucket="rename_layer_references"):
        labeling._rename_model_history_layer_names(ctx.trace)


def _run_build_lookup_keys(ctx: StepContext) -> None:
    """Build lookup keys, finalize retained layers, warn on unattributed args.

    The three calls share one step because the saved-summary refresh and the
    unattributed-arg warning both read the retained-layer lists this step
    finalizes.
    """

    with _vtimed(ctx.trace, "  Step 11: Build lookup keys", bucket="build_lookup_keys"):
        labeling._build_lookup_keys_and_finalize_retained_layers(ctx.trace)
        saved_summary._refresh_fast_saved_summary(ctx.trace)
        source_provenance._warn_unattributed_tensor_args(ctx.trace)


def _run_populate_source_var_names(ctx: StepContext) -> None:
    """Resolve source assignment variable names from the AST index."""

    with _vtimed(
        ctx.trace, "  Step 11.5: Populate source var names", bucket="populate_source_var_names"
    ):
        source_provenance._populate_var_names(ctx.trace)


def _run_resolve_deferred_retention(ctx: StepContext) -> None:
    """Let the capture session resolve deferred retention decisions.

    Runs only when a ``CaptureSession`` is attached; ``should_run`` gates on
    that, so the body may assert it.
    """

    capture_session = ctx.capture_session
    assert capture_session is not None  # should_run gates on this
    with _vtimed(
        ctx.trace, "  Step 11.75: Resolve deferred retention", bucket="resolve_deferred_retention"
    ):
        capture_session.resolve_deferred_retention(ctx.trace, list(ctx.output_tensors))


def _should_run_resolve_deferred_retention(ctx: StepContext) -> bool:
    """Gate deferred-retention resolution on an attached capture session."""

    return ctx.capture_session is not None


def _run_undecorate_saved_tensors(ctx: StepContext) -> None:
    """Strip TorchLens ``tl_*`` attributes from every saved tensor."""

    with _vtimed(ctx.trace, "  Step 12: Undecorate tensors", bucket="undecorate_saved_tensors"):
        finalization._undecorate_all_saved_tensors(ctx.trace)


def _run_clear_cuda_cache(ctx: StepContext) -> None:
    """Clear the CUDA caching allocator for CUDA-touching captures."""

    # Untimed (historical). R36-3: additionally gated on the CAPTURE having touched CUDA (the
    # trace-level forward_memory_backend fact) so a CPU-only trace inside a
    # GPU training loop no longer flushes the caller's allocator.
    if _cuda_probe._is_cuda_available() and _tensor_utils.capture_touched_cuda(ctx.trace):
        torch.cuda.empty_cache()


def _run_log_time_elapsed(ctx: StepContext) -> None:
    """Record elapsed capture timing on the trace."""

    with _vtimed(ctx.trace, "  Step 14: Log timing", bucket="log_time_elapsed"):
        finalization._log_time_elapsed(ctx.trace)


def _run_finalize_param_logs(ctx: StepContext) -> None:
    """Build and complete the ``Param`` records."""

    with _vtimed(ctx.trace, "  Step 15: Finalize params", bucket="finalize_param_logs"):
        finalization._finalize_param_logs(ctx.trace)


def _run_build_layer_logs(ctx: StepContext) -> None:
    """Build aggregate ``Layer`` records and the pass-index index.

    ``Trace._by_pass`` maps each pass index to the ``layer_list`` positions of
    the layers recorded at that pass; layers with no ``pass_index`` are absent
    rather than bucketed.
    """

    trace = ctx.trace
    with _vtimed(trace, "  Step 15.5: Build layer logs", bucket="build_layer_logs"):
        finalization._build_layer_logs(trace)
        trace._by_pass = {}
        for index, op in enumerate(trace._layer_list):
            pass_index = getattr(op, "pass_number", None)
            if pass_index is not None:
                trace._by_pass.setdefault(pass_index, []).append(index)


def _run_build_module_logs(ctx: StepContext) -> None:
    """Build ``Module`` records and refresh the saved module-call count."""

    with _vtimed(ctx.trace, "  Step 16: Build module logs", bucket="build_module_logs"):
        finalization._build_module_logs(ctx.trace)
        saved_summary.refresh_saved_module_call_count(ctx.trace)


def _run_hash_graph_shape(ctx: StepContext) -> None:
    """Normalize layer addresses, then hash the graph shape.

    The hash is taken here, before ``mark_tracing_finished`` flips the trace
    to user-facing behavior and changes what the traversal observes.
    """

    trace = ctx.trace
    with _vtimed(trace, "  Step 16.5: Graph shape hash", bucket="hash_graph_shape"):
        _hashing.populate_normalized_layer_addresses(trace)
        trace._provenance_fingerprint = _hashing.compute_graph_shape_hash(trace)


def _run_mark_tracing_finished(ctx: StepContext) -> None:
    """Switch the trace from build-time to user-facing behavior."""

    with _vtimed(ctx.trace, "  Step 17: Mark pass finished", bucket="mark_tracing_finished"):
        finalization._set_tracing_finished(ctx.trace)


def _run_adopt_containers_and_drop_workspaces(ctx: StepContext) -> None:
    """Adopt container records, then drop the per-phase workspaces.

    This is the contracted terminal consume of ``RawGraphWorkspace``,
    ``ModuleCaptureWorkspace``, and ``WrapperRuntimeWorkspace``: nothing after
    this step may read them. The container registry's records are copied onto
    the trace before its live state is cleared. Untimed (historical).
    """

    trace = ctx.trace
    wrapper_ws = trace.__dict__.get("_wrapper_runtime_ws")
    if wrapper_ws is not None:
        registry = getattr(wrapper_ws, "container_registry", None)
        if registry is not None:
            if registry.records:
                trace.__dict__["_containers"] = dict(registry.records)
            registry.clear_live_state()
    for field_name in (
        "_raw_graph_ws",
        "_module_capture_ws",
        "_wrapper_runtime_ws",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ):
        trace.__dict__.pop(field_name, None)


def _should_run_finalize_streamed_bundle(ctx: StepContext) -> bool:
    """THE streaming snapshot point (deliberately context-writing)."""

    trace = ctx.trace
    ctx.finalize_streaming = getattr(trace, "_out_writer", None) is not None and not getattr(
        trace, "_defer_streaming_bundle_finalization", False
    )
    return ctx.finalize_streaming


def _run_finalize_streamed_bundle(ctx: StepContext) -> None:
    """Finalize the streamed out bundle."""

    with _vtimed(
        ctx.trace, "  Step 18: Finalize streamed bundle", bucket="finalize_streamed_bundle"
    ):
        finalization._finalize_streamed_bundle(ctx.trace)


def _should_run_evict_streamed_outs(ctx: StepContext) -> bool:
    """Gate eviction on the bundle having been finalized with in-memory outs not requested."""

    return bool(ctx.finalize_streaming) and not ctx.trace._keep_outs_in_memory


def _run_evict_streamed_outs(ctx: StepContext) -> None:
    """Evict streamed outs from memory."""

    with _vtimed(ctx.trace, "  Step 19: Evict streamed outs", bucket="evict_streamed_outs"):
        finalization._evict_streamed_outs(ctx.trace)


def _run_release_param_refs(ctx: StepContext) -> None:
    """Release live parameter references held for finalization."""

    with _vtimed(ctx.trace, "  Step 20: Release param refs", bucket="release_param_refs"):
        ctx.trace._release_param_refs(allow_iter_rehydrate=True)


def _add_output_layers_violation(ctx: StepContext) -> str | None:
    """Return the output-layer postcondition violation, if any.

    One output layer per ATTRIBUTABLE output: an output whose parent label is
    ``None`` is skipped by design, so all-unattributable outputs leave none.
    """

    expected = sum(1 for label in ctx.output_parent_labels if label is not None)
    registered = len(ctx.trace.output_layers)
    if registered != expected:
        return (
            f"add_output_layers must register one output layer per attributable output: "
            f"registered {registered}, expected {expected}"
        )
    return None


def _map_final_labels_violation(ctx: StepContext) -> str | None:
    """Return the label-map postcondition violation, if any."""

    trace = ctx.trace
    if not trace._raw_to_final_layer_labels:
        return "map_final_labels must build raw-to-final layer labels"
    if not trace._raw_to_final_op_labels:
        return "map_final_labels must build raw-to-final op labels"
    return None


def _build_lookup_keys_violation(ctx: StepContext) -> str | None:
    """Return the first lookup-key postcondition violation, if any."""

    trace = ctx.trace
    for op in trace._layer_list:
        if not op.label:
            return f"build_lookup_keys left {op!r} without a final op label"
        if not op.layer_label:
            return f"build_lookup_keys left {op!r} without a final layer label"
        if not op.lookup_keys:
            return f"build_lookup_keys left {op.label} without lookup keys"
        if trace._layer_dict_all_keys[op.label] is not op:
            return f"build_lookup_keys did not map op label {op.label!r} to its own op"
        # The bare layer label resolves to ONE pass of that layer (the public
        # lookup contract keeps the LAST pass for multi-pass layers), never to
        # a foreign layer's op.
        resolved = trace._layer_dict_all_keys[op.layer_label]
        if resolved.layer_label != op.layer_label:
            return (
                f"build_lookup_keys mapped layer label {op.layer_label!r} "
                f"to a foreign op {resolved.label!r}"
            )
    return None


def _build_layer_logs_violation(ctx: StepContext) -> str | None:
    """Return the aggregate-layer postcondition violation, if any."""

    trace = ctx.trace
    if not trace._layer_logs:
        return "build_layer_logs must build aggregate layer logs"
    if len(trace._layer_logs) != len(trace.layer_labels):
        return (
            f"build_layer_logs built {len(trace._layer_logs)} layer logs for "
            f"{len(trace.layer_labels)} layer labels"
        )
    if not isinstance(trace._by_pass, dict):
        return "build_layer_logs must build by_pass as a dict"
    return None


def _build_module_logs_violation(ctx: StepContext) -> str | None:
    """Return the module-log postcondition violation, if any."""

    if ctx.trace._module_logs is None:
        return "build_module_logs must build module logs"
    return None


def _hash_graph_shape_violation(ctx: StepContext) -> str | None:
    """Return the graph-hash postcondition violation, if any."""

    if ctx.trace._provenance_fingerprint is None:
        return "hash_graph_shape must compute graph_shape_hash"
    return None


def _mark_tracing_finished_violation(ctx: StepContext) -> str | None:
    """Return the tracing-finished postcondition violation, if any."""

    if ctx.trace._tracing_finished is not True:
        return "mark_tracing_finished must mark tracing finished"
    return None


#: The pipeline, in execution order. Position is the order; there is no rank.
