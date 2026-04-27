"""Postprocessing pipeline for cleaning up the model log after the forward pass.

After the forward pass captures raw tensor metadata into a ModelLog, this pipeline
transforms the raw graph into its user-facing form. The full pipeline has 20 steps,
split into thematic submodules:

- graph_traversal (Steps 1-4): Add output nodes, trace ancestry, remove orphans,
  compute input/output distances.
- control_flow (Steps 5-7): Mark conditional branches, fix module containment for
  internally-generated tensors, deduplicate/merge buffer layers.
- loop_detection (Step 8): Identify repeated operations (loops/recurrence), assign
  same-layer groupings via BFS isomorphic subgraph expansion.
- labeling (Steps 9-12): Generate final human-readable labels, rename all internal
  references, trim/reorder fields, build lookup keys.
- finalization (Steps 13-20): Undecorate saved tensors, log timing, finalize
  ParamLogs, build LayerLog/ModuleLog aggregates, mark pass as finished, then
  finalize any streamed bundle and optionally evict in-memory activations.

Step ordering invariants:
- Steps 1-3 MUST precede Step 5 (conditional branch detection needs orphan-free graph).
- Step 6 (module suffix appending) MUST precede Step 8 (loop detection uses
  operation_equivalence_type which includes module suffixes).
- Step 8 MUST precede Step 9 (label generation needs recurrent_group).
- Step 9 MUST precede Step 10 (final info logging uses finalized labels).
- Step 10 MUST precede Step 12 (lookup key generation needs module hierarchy data
  populated in Step 10).
- Step 11 (rename) MUST precede Step 12 (cleanup uses renamed labels).
- Step 16.5 (_build_layer_logs) MUST precede Step 17 (_build_module_logs) because
  ModuleLog.all_layers references LayerLog keys.

Fast mode skips Steps 1-10 and Step 17, reusing the exhaustive pass's metadata.
It only copies activations from parent to output nodes, trims, renames, builds
LayerLogs, and marks the pass as finished. See postprocess_fast() below.
"""

from typing import TYPE_CHECKING, List

import time
import torch

from ..utils.tensor_utils import safe_copy

from .control_flow import (
    _fix_buffer_layers,
    _fix_modules_for_internal_tensors,
    _mark_conditional_branches,
)
from .finalization import (
    _build_layer_logs,
    _build_module_logs,
    _evict_streamed_activations,
    _finalize_streamed_bundle,
    _finalize_param_logs,
    _log_time_elapsed,
    _set_pass_finished,
    _undecorate_all_saved_tensors,
)
from .graph_traversal import (
    _add_output_layers,
    _find_output_ancestors,
    _mark_input_output_distances,
    _remove_orphan_nodes,
)
from .labeling import (
    _log_final_info_for_all_layers,
    _map_raw_labels_to_final_labels,
    _remove_unwanted_entries_and_log_remaining,
    _rename_model_history_layer_names,
    _trim_and_reorder_model_history_fields,
)
from .loop_detection import _detect_and_label_loops, _group_by_shared_params

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog

from ..utils.display import _vprint, _vtimed


def postprocess(
    self: "ModelLog", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
) -> None:
    """Run the full 18-step postprocessing pipeline (exhaustive mode).

    Transforms the raw ModelLog captured during the forward pass into its
    final user-facing form. In fast mode, delegates to ``postprocess_fast``
    which skips most steps (graph traversal, loop detection, module annotation)
    and only copies activations, trims fields, and builds LayerLogs.

    Args:
        output_tensors: The actual output tensors returned by the model's forward().
        output_tensor_addresses: Hierarchical address strings for each output
            (e.g., "0.1" for nested tuple outputs).
    """
    if self.logging_mode == "fast":
        postprocess_fast(self)
        return

    # Guard: if the model produced no logged layers, skip postprocessing (#153)
    if len(self._raw_layer_labels_list) == 0:
        import warnings

        warnings.warn("No layers were logged during the forward pass; skipping postprocessing.")
        _set_pass_finished(self)
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
    if self.mark_input_output_distances:
        with _vtimed(self, "  Step 4: Input/output distances"):
            _mark_input_output_distances(self)

    # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.
    with _vtimed(self, "  Step 5: Mark conditional branches"):
        _mark_conditional_branches(self)

    # Step 6: Annotate the containing modules for all internally-generated tensors.
    with _vtimed(self, "  Step 6: Fix module containment"):
        _fix_modules_for_internal_tensors(self)

    # Step 7: Fix the buffer passes and parent information.
    with _vtimed(self, "  Step 7: Fix buffer layers"):
        _fix_buffer_layers(self)

    # Step 8: Identify all loops, mark repeated layers.
    loop_desc = (
        "  Step 8: Loop detection (full)"
        if self.detect_loops
        else "  Step 8: Loop detection (params only)"
    )
    with _vtimed(self, loop_desc):
        if self.detect_loops:
            _detect_and_label_loops(self)
        else:
            _group_by_shared_params(self)

    # Step 9: Go down tensor list, get the mapping from raw tensor names to final tensor names.
    with _vtimed(self, "  Step 9: Map labels"):
        _map_raw_labels_to_final_labels(self)

    # Step 10: Log final info for all layers
    with _vtimed(self, "  Step 10: Log final info"):
        _log_final_info_for_all_layers(self)

    # Step 11: Rename all raw labels to final labels
    with _vtimed(self, "  Step 11: Rename labels"):
        _rename_model_history_layer_names(self)
        _trim_and_reorder_model_history_fields(self)

    # Step 12: Remove unsaved layers, build lookup key mappings
    with _vtimed(self, "  Step 12: Build lookup keys"):
        _remove_unwanted_entries_and_log_remaining(self)

    # Step 13: Undecorate all saved tensors and remove saved grad_fns.
    with _vtimed(self, "  Step 13: Undecorate tensors"):
        _undecorate_all_saved_tensors(self)

    # Step 14: Clear the cache after any tensor deletions for garbage collection purposes.
    torch.cuda.empty_cache()

    # Step 15: Log time elapsed.
    with _vtimed(self, "  Step 15: Log timing"):
        _log_time_elapsed(self)

    # Step 16: Populate ParamLog reverse mappings, linked params, num_passes, and gradient metadata.
    with _vtimed(self, "  Step 16: Finalize params"):
        _finalize_param_logs(self)

    # Step 16.5: Build aggregate LayerLog objects from per-pass LayerPassLog entries.
    with _vtimed(self, "  Step 16.5: Build layer logs"):
        _build_layer_logs(self)

    # Step 17: Build structured ModuleLog objects from raw module_* dicts.
    with _vtimed(self, "  Step 17: Build module logs"):
        _build_module_logs(self)

    # Step 18: log the pass as finished, changing the ModelLog behavior to its user-facing version.
    with _vtimed(self, "  Step 18: Mark pass finished"):
        _set_pass_finished(self)

    should_finalize_streaming = self._activation_writer is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        with _vtimed(self, "  Step 19: Finalize streamed bundle"):
            _finalize_streamed_bundle(self)

    if should_finalize_streaming and not self._keep_activations_in_memory:
        with _vtimed(self, "  Step 20: Evict streamed activations"):
            _evict_streamed_activations(self)

    if getattr(self, "verbose", False):
        print(f"[torchlens] Postprocessing complete ({time.time() - _post_t0:.2f}s)")


def postprocess_fast(self: "ModelLog") -> None:
    """Lightweight postprocessing for fast (second-pass) logging mode.

    The fast pass reuses the exhaustive pass's graph structure, loop groupings,
    and labels. It only needs to:
    1. Copy activation data from each output's parent tensor into the output node.
    2. Trim and reorder fields.
    3. Remove unsaved layers and build lookup keys.
    4. Undecorate saved tensors.
    5. Build LayerLog aggregates.
    6. Mark the pass as finished.

    Skipped steps (already computed by the exhaustive pass):
    - Steps 1-4: Graph structure (output nodes, ancestry, orphans, distances)
    - Steps 5-7: Conditional branches, module fixing, buffer fixing
    - Step 8: Loop detection
    - Steps 9-10: Label mapping and final info logging
    - Step 17: _build_module_logs — module structure doesn't change between
      passes and _module_build_data isn't repopulated in fast mode (#108).
    """
    _vprint(self, "Fast-pass postprocessing...")
    # Use layer_dict_main_keys to get LayerPassLog directly (not LayerLog)
    for output_layer_label in self.output_layers:
        output_layer = self.layer_dict_main_keys[output_layer_label]
        if not output_layer.parent_layers:
            continue  # Guard for parentless output layers (#152)
        parent_layer = self.layer_dict_main_keys[output_layer.parent_layers[0]]
        parent_contents = parent_layer.activation
        parent_transformed = parent_layer.transformed_activation
        output_layer.activation = (
            safe_copy(parent_contents, detach_tensor=self.detach_saved_tensors)
            if parent_contents is not None
            else None
        )
        output_layer.transformed_activation = (
            safe_copy(parent_transformed, detach_tensor=self.detach_saved_tensors)
            if isinstance(parent_transformed, torch.Tensor)
            else parent_transformed
        )
        output_layer.tensor_memory = parent_layer.tensor_memory
        output_layer.transformed_activation_shape = parent_layer.transformed_activation_shape
        output_layer.transformed_activation_dtype = parent_layer.transformed_activation_dtype
        output_layer.transformed_activation_memory = parent_layer.transformed_activation_memory
        output_layer.has_saved_activations = parent_layer.has_saved_activations
        output_layer.has_gradient = parent_layer.has_gradient
        output_layer.gradient = parent_layer.gradient
        output_layer.transformed_gradient = parent_layer.transformed_gradient
        output_layer.transformed_gradient_shape = parent_layer.transformed_gradient_shape
        output_layer.transformed_gradient_dtype = parent_layer.transformed_gradient_dtype
        output_layer.transformed_gradient_memory = parent_layer.transformed_gradient_memory
        if output_layer.has_saved_activations:
            self.layers_with_saved_activations.append(output_layer_label)
    _trim_and_reorder_model_history_fields(self)
    _remove_unwanted_entries_and_log_remaining(self)
    _undecorate_all_saved_tensors(self)

    torch.cuda.empty_cache()
    _log_time_elapsed(self)
    _build_layer_logs(self)
    # Note: _build_module_logs is NOT called here because module structure
    # doesn't change between passes and _module_build_data isn't repopulated
    # in fast mode (Step 10 is skipped). Existing module logs remain valid. (#108)
    _set_pass_finished(self)

    should_finalize_streaming = self._activation_writer is not None and not getattr(
        self, "_defer_streaming_bundle_finalization", False
    )
    if should_finalize_streaming:
        _finalize_streamed_bundle(self)
    if should_finalize_streaming and not self._keep_activations_in_memory:
        _evict_streamed_activations(self)
