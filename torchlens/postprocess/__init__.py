"""Postprocessing pipeline for cleaning up the model log after the forward pass.

After the forward pass captures raw tensor metadata into a Trace, this pipeline
transforms the raw graph into its user-facing form. The full pipeline has 19 steps,
split into thematic submodules:

- graph_traversal (Steps 1-4): Add output nodes, trace ancestry, remove orphans,
  compute input/output distances.
- control_flow (Steps 5-6): Mark conditional branches and deduplicate/merge buffer layers.
- loop_detection (Step 7): Identify repeated operations (loops/recurrence), assign
  same-layer groupings via BFS isomorphic subgraph expansion.
- labeling (Steps 8-11): Generate final human-readable labels, rename all internal
  references, trim/reorder fields, build lookup keys.
- finalization (Steps 12-19): Undecorate saved tensors, log timing, finalize
  ParamLogs, build Layer/Module aggregates, mark pass as finished, then
  finalize any streamed bundle and optionally evict in-memory outs.

Step ordering invariants:
- Steps 1-3 MUST precede Step 5 (conditional branch detection needs orphan-free graph).
- Module suffixes must be present on equivalence_class before Step 7 loop detection.
- Step 7 MUST precede Step 8 (label generation needs recurrent_ops).
- Step 8 MUST precede Step 9 (final info logging uses finalized labels).
- Step 9 MUST precede Step 11 (lookup key generation needs module hierarchy data
  populated in Step 9).
- Step 10 (rename) MUST precede Step 11 (cleanup uses renamed labels).
- Step 15.5 (_build_layer_logs) MUST precede Step 16 (_build_module_logs) because
  Module.layers references Layer keys.

Fast mode skips Steps 1-9 and Step 16, reusing the exhaustive pass's metadata.
It only copies outs from parent to output nodes, trims, renames, builds
LayerLogs, and marks the pass as finished. See postprocess_fast() below.
"""

from typing import TYPE_CHECKING, List

import time
import torch

from ..utils.tensor_utils import _is_cuda_available, safe_copy
from ..utils.hashing import compute_graph_shape_hash

from .control_flow import (
    _fix_buffer_layers,
    _mark_conditional_branches,
)
from .finalization import (
    _build_layer_logs,
    _build_module_logs,
    _evict_streamed_outs,
    _finalize_streamed_bundle,
    _finalize_param_logs,
    _log_time_elapsed,
    _set_tracing_finished,
    _undecorate_all_saved_tensors,
)
from .graph_traversal import (
    _add_output_layers,
    _find_output_ancestors,
    _mark_layer_depths,
    _remove_orphan_nodes,
)
from .labeling import (
    _log_final_info_for_layers,
    _map_raw_labels_to_final_labels,
    _remove_unwanted_entries_and_log_remaining,
    _rename_model_history_layer_names,
    _trim_and_reorder_model_history_fields,
)
from .loop_detection import _detect_and_label_loops, _group_by_shared_params
from ._materialize import materialize_from_events

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

from ..quantities import Bytes
from ..utils.display import _vprint, _vtimed
from ..captured_run import remember_event_stream


def _drop_transient_capture_state(self: "Trace") -> None:
    """Remove capture/session scratch that must not survive on final traces.

    Args:
        self: Trace whose postprocess-local state should be discarded.

    Returns:
        None. Mutates ``self.__dict__``.
    """

    keep_deferred_streaming = bool(
        self.__dict__.get("_defer_streaming_bundle_finalization", False)
        and self.__dict__.get("_out_writer") is not None
    )
    keep_selective_sink = self.__dict__.get("_out_sink") is not None
    field_names = [
        "_build_state",
        "capture_events",
        "_output_container_specs_by_raw_label",
    ]
    if not keep_deferred_streaming and not keep_selective_sink:
        field_names.extend(
            [
                "_out_writer",
                "_out_sink",
                "_keep_outs_in_memory",
                "_keep_grads_in_memory",
                "_defer_streaming_bundle_finalization",
            ]
        )
    elif not keep_deferred_streaming:
        field_names.extend(
            [
                "_out_writer",
                "_keep_outs_in_memory",
                "_keep_grads_in_memory",
                "_defer_streaming_bundle_finalization",
            ]
        )
    for field_name in field_names:
        self.__dict__.pop(field_name, None)


def _refresh_fast_saved_summary(self: "Trace") -> None:
    """Refresh saved-output counters after a fast replay pass.

    Args:
        self: Trace whose final layer entries were updated in fast mode.

    Returns:
        None. Mutates aggregate saved-output fields on ``self``.
    """

    saved_layers = [
        layer_entry
        for layer_entry in self.layer_list
        if getattr(layer_entry, "has_saved_activation", False)
        and not getattr(layer_entry, "is_orphan", False)
    ]
    self.num_saved_ops = len(saved_layers)
    self.saved_activation_memory = Bytes(
        sum(int(getattr(layer_entry, "activation_memory", 0) or 0) for layer_entry in saved_layers)
    )
    self.num_saved_layers = len({layer_entry.layer_label for layer_entry in saved_layers})
    saved_labels = {layer_entry.layer_label for layer_entry in saved_layers}
    self.num_saved_module_calls = sum(
        1
        for module_call in getattr(self, "module_calls", [])
        if any(label in saved_labels for label in getattr(module_call, "layers", []))
    )


def postprocess(
    self: "Trace", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
) -> None:
    """Run the full 18-step postprocessing pipeline (exhaustive mode).

    Transforms the raw Trace captured during the forward pass into its
    final user-facing form. In fast mode, delegates to ``postprocess_fast``
    which skips most steps (graph traversal, loop detection, module annotation)
    and only copies outs, trims fields, and builds LayerLogs.

    Args:
        output_tensors: The actual output tensors returned by the model's forward().
        output_tensor_addresses: Hierarchical address strings for each output
            (e.g., "0.1" for nested tuple outputs).
    """
    if self.capture_mode == "fast":
        postprocess_fast(self)
        return

    capture_events = getattr(self, "capture_events", None)
    if capture_events is not None:
        remember_event_stream(self, capture_events)
        with _vtimed(self, "  Step 0: Materialize capture events"):
            materialize_from_events(self, capture_events)
        delattr(self, "capture_events")

    # Guard: if the model produced no logged layers, skip postprocessing (#153)
    if len(self._raw_layer_labels_list) == 0:
        import warnings

        warnings.warn("No layers were logged during the forward pass; skipping postprocessing.")
        _set_tracing_finished(self)
        _drop_transient_capture_state(self)
        return

    _vprint(
        self,
        f"Postprocessing {len(self._raw_layer_labels_list):,} layers "
        f"({len(self.buffer_layers):,} buffers)...",
    )
    _post_t0 = time.time() if getattr(self, "verbose", False) else 0

    # Step 1: Add dedicated output nodes
    with _vtimed(self, "  Step 1: Add output layers"):
        _add_output_layers(self, output_tensors, output_tensor_addresses)

    # Step 2: Trace which nodes are ancestors of output nodes
    with _vtimed(self, "  Step 2: Trace output ancestors"):
        _find_output_ancestors(self)

    # Step 3: Remove orphan nodes, find nodes that don't terminate in output node
    with _vtimed(self, "  Step 3: Remove orphan nodes"):
        _remove_orphan_nodes(self)

    # Step 4: Find min/max distance from input and output nodes.
    # Conditional: only runs when the user requested distance metadata.
    if self.mark_layer_depths:
        with _vtimed(self, "  Step 4: Input/output distances"):
            _mark_layer_depths(self)

    # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.
    with _vtimed(self, "  Step 5: Mark conditional branches"):
        _mark_conditional_branches(self)

    # Step 6: Fix the buffer ops and parent information.
    with _vtimed(self, "  Step 6: Fix buffer layers"):
        _fix_buffer_layers(self)

    # Step 7: Identify all loops, mark repeated layers.
    loop_desc = (
        "  Step 7: Loop detection (full)"
        if self.recurrence_detection
        else "  Step 7: Loop detection (params only)"
    )
    with _vtimed(self, loop_desc):
        if self.recurrence_detection:
            _detect_and_label_loops(self)
        else:
            _group_by_shared_params(self)

    # Step 8: Go down tensor list, get the mapping from raw tensor names to final tensor names.
    with _vtimed(self, "  Step 8: Map labels"):
        _map_raw_labels_to_final_labels(self)

    # Step 9: Log final info for all layers
    with _vtimed(self, "  Step 9: Log final info"):
        _log_final_info_for_layers(self)

    # Step 10: Rename all raw labels to final labels
    with _vtimed(self, "  Step 10: Rename labels"):
        _rename_model_history_layer_names(self)
        _trim_and_reorder_model_history_fields(self)

    # Step 11: Remove unsaved layers, build lookup key mappings
    with _vtimed(self, "  Step 11: Build lookup keys"):
        _remove_unwanted_entries_and_log_remaining(self)

    # Step 12: Undecorate all saved tensors and remove saved grad_fns.
    with _vtimed(self, "  Step 12: Undecorate tensors"):
        _undecorate_all_saved_tensors(self)

    # Step 13: Clear the cache after any tensor deletions for garbage collection purposes.
    # Gated behind cached cuda.is_available() so CPU-only runs don't pay the
    # CUDA driver / NVML probe cost (per profiling audit 2026-04-27 finding #4).
    if _is_cuda_available():
        torch.cuda.empty_cache()

    # Step 14: Log time elapsed.
    with _vtimed(self, "  Step 14: Log timing"):
        _log_time_elapsed(self)

    # Step 15: Populate Param reverse mappings, linked params, num_calls, and grad metadata.
    with _vtimed(self, "  Step 15: Finalize params"):
        _finalize_param_logs(self)

    # Step 15.5: Build aggregate Layer objects from per-pass Op entries.
    with _vtimed(self, "  Step 15.5: Build layer logs"):
        _build_layer_logs(self)
        self.by_pass = {}
        for index, op in enumerate(self.layer_list):
            pass_index = getattr(op, "pass_index", None)
            if pass_index is not None:
                self.by_pass.setdefault(pass_index, []).append(index)

    # Step 16: Build structured Module objects from raw module_* dicts.
    with _vtimed(self, "  Step 16: Build module logs"):
        _build_module_logs(self)

    # Step 16.5: Compute graph shape hash before _set_tracing_finished changes access behavior.
    with _vtimed(self, "  Step 16.5: Graph shape hash"):
        self.graph_shape_hash = compute_graph_shape_hash(self)

    # Step 17: log the pass as finished, changing the Trace behavior to its user-facing version.
    with _vtimed(self, "  Step 17: Mark pass finished"):
        _set_tracing_finished(self)

    for field_name in ("_build_state", "capture_events", "_output_container_specs_by_raw_label"):
        self.__dict__.pop(field_name, None)

    should_finalize_streaming = getattr(self, "_out_writer", None) is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        with _vtimed(self, "  Step 18: Finalize streamed bundle"):
            _finalize_streamed_bundle(self)

    if should_finalize_streaming and not self._keep_outs_in_memory:
        with _vtimed(self, "  Step 19: Evict streamed outs"):
            _evict_streamed_outs(self)

    if getattr(self, "verbose", False):
        print(f"[torchlens] Postprocessing complete ({time.time() - _post_t0:.2f}s)")
    _drop_transient_capture_state(self)


def postprocess_fast(self: "Trace") -> None:
    """Lightweight postprocessing for fast (second-pass) logging mode.

    The fast pass reuses the exhaustive pass's graph structure, loop groupings,
    and labels. It only needs to:
    1. Copy out data from each output's parent tensor into the output node.
    2. Trim and reorder fields.
    3. Remove unsaved layers and build lookup keys.
    4. Undecorate saved tensors.
    5. Build Layer aggregates.
    6. Mark the pass as finished.

    Skipped steps (already computed by the exhaustive pass):
    - Steps 1-4: Graph structure (output nodes, ancestry, orphans, distances)
    - Steps 5-6: Conditional branches and buffer fixing
    - Step 7: Loop detection
    - Steps 8-9: Label mapping and final info logging
    - Step 16: _build_module_logs — module structure doesn't change between
      ops and _module_build_data isn't repopulated in fast mode (#108).
    """
    _vprint(self, "Fast-pass postprocessing...")
    # Use layer_dict_main_keys to get Op directly (not Layer)
    for output_layer_label in self.output_layers:
        output_layer = self[output_layer_label]
        if not output_layer.parents:
            continue  # Guard for parentless output layers (#152)
        parent_layer = self[output_layer.parents[0]]
        parent_contents = parent_layer.out
        parent_transformed = parent_layer.transformed_out
        output_layer._internal_set(
            "out",
            safe_copy(parent_contents, detach_tensor=self.detach_saved_activations)
            if parent_contents is not None
            else None,
        )
        output_layer._internal_set(
            "transformed_out",
            safe_copy(parent_transformed, detach_tensor=self.detach_saved_activations)
            if isinstance(parent_transformed, torch.Tensor)
            else parent_transformed,
        )
        output_layer.activation_memory = parent_layer.activation_memory
        output_layer.transformed_out_shape = parent_layer.transformed_out_shape
        output_layer.transformed_out_dtype = parent_layer.transformed_out_dtype
        output_layer.transformed_activation_memory = parent_layer.transformed_activation_memory
        output_layer.has_saved_activation = parent_layer.has_saved_activation
        output_layer.has_grad = parent_layer.has_grad
        output_layer._internal_set("grad", parent_layer.grad)
        output_layer._internal_set("transformed_grad", parent_layer.transformed_grad)
        output_layer.transformed_grad_shape = parent_layer.transformed_grad_shape
        output_layer.transformed_grad_dtype = parent_layer.transformed_grad_dtype
        output_layer.transformed_gradient_memory = parent_layer.transformed_gradient_memory
    _refresh_fast_saved_summary(self)
    _trim_and_reorder_model_history_fields(self)
    _undecorate_all_saved_tensors(self)

    # Gated behind cached cuda.is_available() so CPU-only fast-pass runs don't
    # pay the CUDA driver / NVML probe cost.
    if _is_cuda_available():
        torch.cuda.empty_cache()
    _log_time_elapsed(self)
    _build_layer_logs(self)
    # Note: _build_module_logs is NOT called here because module structure
    # doesn't change between ops and _module_build_data isn't repopulated
    # in fast mode (Step 9 is skipped). Existing module logs remain valid. (#108)
    if self.intervention_ready and not getattr(self, "save_arg_templates", False):
        self.intervention_ready = False
    self.graph_shape_hash = compute_graph_shape_hash(self)
    _set_tracing_finished(self)

    should_finalize_streaming = getattr(self, "_out_writer", None) is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        _finalize_streamed_bundle(self)
    if should_finalize_streaming and not self._keep_outs_in_memory:
        _evict_streamed_outs(self)
    _drop_transient_capture_state(self)
