"""Postprocessing pipeline for cleaning up the model log after the forward pass.

Split into thematic modules:
- graph_traversal: Steps 1-4 (output nodes, ancestry, orphans, distances)
- control_flow: Steps 5-7 (conditional branches, module fixes, buffer fixes)
- loop_detection: Step 8 (loop detection, isomorphic subgraph expansion)
- labeling: Steps 9-12 (label mapping, final info, renaming, cleanup)
- finalization: Steps 13-18 (undecoration, timing, params, layer logs, modules, finish)
"""

from typing import TYPE_CHECKING, List

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
from .loop_detection import _detect_and_label_loops

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def postprocess(
    self: "ModelLog", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
) -> None:
    """After the forward pass, cleans up the log into its final form.

    Runs the full 18-step postprocessing pipeline (exhaustive mode) or a
    shortened fast-mode pipeline. Each step is delegated to a thematic
    submodule — see the module docstring for the mapping.
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

    # Step 1: Add dedicated output nodes

    _add_output_layers(self, output_tensors, output_tensor_addresses)

    # Step 2: Trace which nodes are ancestors of output nodes

    _find_output_ancestors(self)

    # Step 3: Remove orphan nodes, find nodes that don't terminate in output node

    _remove_orphan_nodes(self)

    # Step 4: Find mix/max distance from input and output nodes

    if self.mark_input_output_distances:
        _mark_input_output_distances(self)

    # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.

    _mark_conditional_branches(self)

    # Step 6: Annotate the containing modules for all internally-generated tensors (they don't know where
    # they are when they're made; have to trace breadcrumbs from tensors that came from input).

    _fix_modules_for_internal_tensors(self)

    # Step 7: Fix the buffer passes and parent infomration.

    _fix_buffer_layers(self)

    # Step 8: Identify all loops, mark repeated layers.

    _detect_and_label_loops(self)

    # Step 9: Go down tensor list, get the mapping from raw tensor names to final tensor names.

    _map_raw_labels_to_final_labels(self)

    # Step 10: Go through and log information pertaining to all layers:
    _log_final_info_for_all_layers(self)

    # Step 11: Rename the raw tensor entries in the fields of ModelLog:
    _rename_model_history_layer_names(self)
    _trim_and_reorder_model_history_fields(self)

    # Step 12: And one more pass to delete unused layers from the record and do final tidying up:
    _remove_unwanted_entries_and_log_remaining(self)

    # Step 13: Undecorate all saved tensors and remove saved grad_fns.
    _undecorate_all_saved_tensors(self)

    # Step 14: Clear the cache after any tensor deletions for garbage collection purposes:
    torch.cuda.empty_cache()

    # Step 15: Log time elapsed.
    _log_time_elapsed(self)

    # Step 16: Populate ParamLog reverse mappings, linked params, num_passes, and gradient metadata.
    _finalize_param_logs(self)

    # Step 16.5: Build aggregate LayerLog objects from per-pass LayerPassLog entries.
    _build_layer_logs(self)

    # Step 17: Build structured ModuleLog objects from raw module_* dicts.
    _build_module_logs(self)

    # Step 18: log the pass as finished, changing the ModelLog behavior to its user-facing version.

    _set_pass_finished(self)


def postprocess_fast(self: "ModelLog") -> None:
    """Lightweight postprocessing for fast logging mode.

    Copies activation data from each output's parent tensor into the output
    node, then trims, renames, undecorates, and marks the pass as finished.
    Skips graph traversal, loop detection, and module annotation.
    """
    # Use layer_dict_main_keys to get LayerPassLog directly (not LayerLog)
    for output_layer_label in self.output_layers:
        output_layer = self.layer_dict_main_keys[output_layer_label]
        if not output_layer.parent_layers:
            continue  # Guard for parentless output layers (#152)
        parent_layer = self.layer_dict_main_keys[output_layer.parent_layers[0]]
        parent_contents = parent_layer.tensor_contents
        output_layer.tensor_contents = (
            safe_copy(parent_contents, detach_tensor=True) if parent_contents is not None else None
        )
        output_layer.tensor_fsize = parent_layer.tensor_fsize
        output_layer.tensor_fsize_nice = parent_layer.tensor_fsize_nice
        output_layer.has_saved_activations = parent_layer.has_saved_activations
        output_layer.has_saved_grad = parent_layer.has_saved_grad
        output_layer.grad_contents = parent_layer.grad_contents
        if output_layer.has_saved_activations:
            self.layers_with_saved_activations.append(output_layer_label)
    _trim_and_reorder_model_history_fields(self)
    _remove_unwanted_entries_and_log_remaining(self)
    _undecorate_all_saved_tensors(self)

    torch.cuda.empty_cache()
    _log_time_elapsed(self)
    _build_layer_logs(self)
    _set_pass_finished(self)
