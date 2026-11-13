"""Postprocess step adapters: ``StepContext`` and one function per pipeline step.

Each ``_run_<step name>`` function here adapts one step body to the pipeline:
it resolves the function through its home module at call time
(``graph_traversal._find_output_ancestors``, never an imported reference), so
a function replaced on its home module is the one that runs. Timing is not the
adapters' concern: ``_pipeline.run_pipeline`` times each step listed in
``STEP_DESCRIPTIONS`` under its name. ``_should_run_<step name>`` functions are the
gates of the optional steps and ``_<step name>_violation`` functions are the
postconditions. The ordered table that strings them together is
``_pipeline.PIPELINE``; ``_pipeline.run_pipeline`` is the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from .. import _postprocess as _package
from .._data_classes._compaction import hand_off_container_tokens
from .._utils import _cuda_probe, hashing as _hashing, tensor_utils as _tensor_utils
from . import (
    _layer_depths,
    _step_views,
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

    graph_traversal._add_output_layers(
        ctx.trace,
        ctx.output_tensors,
        ctx.output_tensor_addresses,
        ctx.output_parent_labels,
    )


def _run_find_output_ancestors(ctx: StepContext) -> None:
    """Mark every node that is an ancestor of a model output."""

    graph_traversal._find_output_ancestors(ctx.trace)


def _run_remove_orphan_nodes(ctx: StepContext) -> None:
    """Drop raw nodes that reach no model output."""

    journal = ctx.trace.__dict__.pop("_foreign_departures", None)
    kept = journal.end_capture() if journal is not None else None  # slot marks only
    if kept is not None:
        ctx.trace.__dict__["_foreign_departures"] = kept
    graph_traversal._remove_orphan_nodes(ctx.trace)


def _run_mark_layer_depths(ctx: StepContext) -> None:
    """Write the optional input/output distance metadata."""

    _layer_depths._mark_layer_depths(ctx.trace)


def _should_run_mark_layer_depths(ctx: StepContext) -> bool:
    """Gate the depth pass on the opt-in ``Trace.options.mark_layer_depths`` flag."""

    return bool(ctx.trace._options_mark_layer_depths)


def _run_mark_conditional_branches(ctx: StepContext) -> None:
    """Attribute conditional branches (AST, bool, event, and edge evidence)."""

    control_flow._mark_conditional_branches(ctx.trace)


def _run_fix_buffer_layers(ctx: StepContext) -> None:
    """Deduplicate buffer layers and reconnect their edges."""

    control_flow._fix_buffer_layers(ctx.trace)


def _run_group_recurrent_layers(ctx: StepContext) -> None:
    """Group recurrent calls into multi-pass layers.

    With ``recurrence_detection`` off this degrades to shared-parameter
    grouping only, which is the historical single-pass layout.
    """

    trace = ctx.trace
    if trace._options_recurrence_detection:
        loop_detection._detect_and_label_loops(trace)
    else:
        loop_detection._group_by_shared_params(trace)


def _run_map_final_labels(ctx: StepContext) -> None:
    """Assign final labels by column and remap the trace-level label references."""

    labeling._map_raw_labels_to_final_labels(ctx.trace)


def _run_log_final_layer_info(ctx: StepContext) -> None:
    """Write the final per-layer and per-module fields."""

    labeling._log_final_info_for_layers(ctx.trace)


def _run_build_lookup_keys(ctx: StepContext) -> None:
    """Build lookup keys, finalize retained layers, warn on unattributed args.

    The three calls share one step because the saved-summary refresh and the
    unattributed-arg warning both read the retained-layer lists this step
    finalizes.
    """

    labeling._build_lookup_keys_and_finalize_retained_layers(ctx.trace)
    saved_summary._refresh_fast_saved_summary(ctx.trace)
    source_provenance._warn_unattributed_tensor_args(ctx.trace)


def _run_populate_source_var_names(ctx: StepContext) -> None:
    """Resolve source assignment variable names from the AST index."""

    source_provenance._populate_var_names(ctx.trace)


def _run_resolve_deferred_retention(ctx: StepContext) -> None:
    """Let the capture session resolve deferred retention decisions.

    Runs only when a ``CaptureSession`` is attached; ``should_run`` gates on
    that, so the body may assert it.
    """

    capture_session = ctx.capture_session
    assert capture_session is not None  # should_run gates on this
    capture_session.resolve_deferred_retention(ctx.trace, list(ctx.output_tensors))


def _should_run_resolve_deferred_retention(ctx: StepContext) -> bool:
    """Gate deferred-retention resolution on an attached capture session."""

    return ctx.capture_session is not None


def _run_undecorate_saved_tensors(ctx: StepContext) -> None:
    """Strip TorchLens ``tl_*`` attributes from every saved tensor."""

    finalization._undecorate_all_saved_tensors(ctx.trace)


def _run_clear_cuda_cache(ctx: StepContext) -> None:
    """Clear the CUDA caching allocator for CUDA-touching captures."""

    # Untimed (historical). Additionally gated on the CAPTURE having touched CUDA (the
    # trace-level forward_memory_backend fact) so a CPU-only trace inside a
    # GPU training loop no longer flushes the caller's allocator.
    if _cuda_probe._is_cuda_available() and _tensor_utils.capture_touched_cuda(ctx.trace):
        torch.cuda.empty_cache()


def _run_log_time_elapsed(ctx: StepContext) -> None:
    """Record elapsed capture timing on the trace."""

    finalization._log_time_elapsed(ctx.trace)


def _run_finalize_param_logs(ctx: StepContext) -> None:
    """Build and complete the ``Param`` records."""

    finalization._finalize_param_logs(ctx.trace)


def _run_build_layer_logs(ctx: StepContext) -> None:
    """Group the ops into aggregate ``Layer`` records and build the pass index.

    ``Trace._by_pass`` maps each pass index to the ``layer_list`` positions of
    the layers recorded at that pass; layers with no ``pass_index`` are absent
    rather than bucketed. The step's body builds it in its one scan of the ops.
    """

    trace = ctx.trace
    finalization._build_layer_logs(trace)


def _run_build_module_logs(ctx: StepContext) -> None:
    """Build ``Module`` records and the trace's module-call accessor."""

    finalization._seal_module_logs(ctx.trace)  # the records are built on first lookup


#: Every field ``compute_graph_shape_hash`` reads of a layer on the
#: address-free path (``_record_type`` and ``_record_is_buffer`` included).
_HASH_FIELDS = (
    "layer_label",
    "num_passes",
    "label",
    "module",
    "parents",
    "func_name",
    "container_path",
    "container_spec",
    "is_input",
    "is_output",
    "layer_type",
    "type",
    "_graph_is_buffer",
    "is_buffer",
)


def _run_hash_graph_shape(ctx: StepContext) -> None:
    """Normalize layer addresses, then hash the graph shape.

    The hash is taken here, before ``mark_tracing_finished`` flips the trace
    to user-facing behavior and changes what the traversal observes. The
    fields it reads are read once per field for all ops (``_step_views``),
    and the hash runs over those values: each is what ``getattr(op, name)``
    returns, so the digest is the one the ops themselves give.
    """

    trace = ctx.trace
    ops = trace._layer_list
    bound = _step_views.bind_ops(ops)
    if bound is None:  # a short trace or ops on several stores: read the facade
        _hashing.populate_normalized_layer_addresses(trace)
        trace._provenance_fingerprint = _hashing.compute_graph_shape_hash(
            trace, include_module_address=False
        )
        return
    views = _step_views.row_views(ops, _HASH_FIELDS, bound)
    normalize = _hashing.normalize_address_for_hash
    addresses = [normalize(getattr(view, "module", None)) for view in views]
    _step_views.write_field(ops, "_address_normalized", addresses, bound)
    # The stored fingerprint is the address-free structural hash, the same
    # value ``tl.fingerprint.trace`` returns for this trace.
    trace._provenance_fingerprint = _hashing.compute_graph_shape_hash(
        SimpleNamespace(_layer_list=views), include_module_address=False
    )


def _run_mark_tracing_finished(ctx: StepContext) -> None:
    """Switch the trace from build-time to user-facing behavior."""

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
    hand_off_container_tokens(trace, wrapper_ws)
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


def _run_seal_op_store(ctx: StepContext) -> None:
    """Compact and seal the Op row store, ahead of the finished mark.

    The body is the package-level ``_seal_op_store``, resolved at call time
    (the package imports this module, so the name is read off the package).
    Untimed.
    """

    _package._seal_op_store(ctx.trace)


def _should_run_finalize_streamed_bundle(ctx: StepContext) -> bool:
    """THE streaming snapshot point (deliberately context-writing)."""

    trace = ctx.trace
    ctx.finalize_streaming = getattr(trace, "_out_writer", None) is not None and not getattr(
        trace, "_defer_streaming_bundle_finalization", False
    )
    return ctx.finalize_streaming


def _run_finalize_streamed_bundle(ctx: StepContext) -> None:
    """Finalize the streamed out bundle."""

    finalization._finalize_streamed_bundle(ctx.trace)


def _should_run_evict_streamed_outs(ctx: StepContext) -> bool:
    """Gate eviction on the bundle having been finalized with in-memory outs not requested."""

    return bool(ctx.finalize_streaming) and not ctx.trace._keep_outs_in_memory


def _run_evict_streamed_outs(ctx: StepContext) -> None:
    """Evict streamed outs from memory."""

    finalization._evict_streamed_outs(ctx.trace)


def _run_release_param_refs(ctx: StepContext) -> None:
    """Release live parameter references held for finalization."""

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
    """Return the first lookup-key postcondition violation, if any.

    Reads each op's labels and lookup keys once per field, as columns
    (``_step_views``); the checks and their order are the per-op ones.
    """

    trace = ctx.trace
    ops = trace._layer_list
    views = _step_views.row_views(ops, ("label", "layer_label", "lookup_keys"))
    view_of = {id(op): view for op, view in zip(ops, views, strict=True)}
    all_keys = trace._layer_dict_all_keys
    for op, view in zip(ops, views, strict=True):
        label = view.label
        layer_label = view.layer_label
        if not label:
            return f"build_lookup_keys left {op!r} without a final op label"
        if not layer_label:
            return f"build_lookup_keys left {op!r} without a final layer label"
        if not view.lookup_keys:
            return f"build_lookup_keys left {label} without lookup keys"
        if all_keys[label] is not op:
            return f"build_lookup_keys did not map op label {label!r} to its own op"
        # The bare layer label resolves to ONE pass of that layer (the public
        # lookup contract keeps the LAST pass for multi-pass layers), never to
        # a foreign layer's op.
        resolved = all_keys[layer_label]
        resolved_view = view_of.get(id(resolved))
        resolved_layer_label = (
            resolved.layer_label if resolved_view is None else resolved_view.layer_label
        )
        if resolved_layer_label != layer_label:
            return (
                f"build_lookup_keys mapped layer label {layer_label!r} "
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

    state = ctx.trace.__dict__
    if state.get("_module_pending") is None and state.get("_module_logs") is None:
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


#: Verbose progress line of each timed step; its ``_phase_timings`` bucket is the
#: step name. ``clear_cuda_cache`` and ``adopt_containers_and_drop_workspaces``
#: are untimed and absent. ``run_pipeline`` times the steps; the adapters above
#: carry no timing of their own.
STEP_DESCRIPTIONS: dict[str, str] = {
    "add_output_layers": "  Step 1: Add output layers",
    "find_output_ancestors": "  Step 2: Trace output ancestors",
    "remove_orphan_nodes": "  Step 3: Remove orphan nodes",
    "mark_layer_depths": "  Step 4: Input/output distances",
    "mark_conditional_branches": "  Step 5: Mark conditional branches",
    "fix_buffer_layers": "  Step 6: Fix buffer layers",
    "group_recurrent_layers": "  Step 7: Loop detection (full)",
    "map_final_labels": "  Step 8: Map labels",
    "log_final_layer_info": "  Step 9: Log final info",
    "build_lookup_keys": "  Step 11: Build lookup keys",
    "populate_source_var_names": "  Step 11.5: Populate source var names",
    "resolve_deferred_retention": "  Step 11.75: Resolve deferred retention",
    "undecorate_saved_tensors": "  Step 12: Undecorate tensors",
    "log_time_elapsed": "  Step 14: Log timing",
    "finalize_param_logs": "  Step 15: Finalize params",
    "build_layer_logs": "  Step 15.5: Build layer logs",
    "build_module_logs": "  Step 16: Build module logs",
    "hash_graph_shape": "  Step 16.5: Graph shape hash",
    "mark_tracing_finished": "  Step 17: Mark pass finished",
    "finalize_streamed_bundle": "  Step 18: Finalize streamed bundle",
    "evict_streamed_outs": "  Step 19: Evict streamed outs",
    "release_param_refs": "  Step 20: Release param refs",
}


def step_description(name: str, ctx: StepContext) -> str:
    """Return the verbose progress line of timed step ``name``."""

    if name == "group_recurrent_layers" and not ctx.trace._options_recurrence_detection:
        return "  Step 7: Loop detection (params only)"
    return STEP_DESCRIPTIONS[name]
