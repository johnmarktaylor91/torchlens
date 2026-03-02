"""Postprocessing pipeline for cleaning up the model log after the forward pass.

Split into thematic modules:
- graph_traversal: Steps 1-4 (output nodes, ancestry, orphans, distances)
- control_flow: Steps 5-7 (conditional branches, module fixes, buffer fixes)
- loop_detection: Step 8 (loop detection, isomorphic subgraph expansion)
- labeling: Steps 9-12 (label mapping, final info, renaming, cleanup)
- finalization: Steps 13-18 (undecoration, timing, params, modules, finish, rolling)
"""

from typing import TYPE_CHECKING, List

import torch

from .control_flow import (
    _fix_buffer_layers,
    _fix_modules_for_internal_tensors,
    _mark_conditional_branches,
)
from .finalization import (
    _build_module_logs,
    _finalize_param_logs,
    _log_time_elapsed,
    _roll_graph,
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
    _map_raw_tensor_labels_to_final_tensor_labels,
    _remove_unwanted_entries_and_log_remaining,
    _rename_model_history_layer_names,
    _trim_and_reorder_model_history_fields,
)
from .loop_detection import _detect_and_label_loops

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def postprocess(
    self: "ModelLog", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
):
    """
    After the forward pass, cleans up the log into its final form.
    """
    if self.logging_mode == "fast":
        postprocess_fast(self)
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

    _map_raw_tensor_labels_to_final_tensor_labels(self)

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

    # Step 17: Build structured ModuleLog objects from raw module_* dicts.
    _build_module_logs(self)

    # Step 18: log the pass as finished, changing the ModelLog behavior to its user-facing version.

    _set_pass_finished(self)


def postprocess_fast(self: "ModelLog"):
    for output_layer_label in self.output_layers:
        output_layer = self[output_layer_label]
        output_layer.tensor_contents = self[output_layer.parent_layers[0]].tensor_contents
        output_layer.tensor_fsize = self[output_layer.parent_layers[0]].tensor_fsize
        output_layer.tensor_fsize_nice = self[output_layer.parent_layers[0]].tensor_fsize_nice
        output_layer.has_saved_activations = self[
            output_layer.parent_layers[0]
        ].has_saved_activations
        output_layer.has_saved_grad = self[output_layer.parent_layers[0]].has_saved_grad
        output_layer.grad_contents = self[output_layer.parent_layers[0]].grad_contents
        if output_layer.has_saved_activations:
            self.layers_with_saved_activations.append(output_layer_label)
    _trim_and_reorder_model_history_fields(self)
    _remove_unwanted_entries_and_log_remaining(self)
    _undecorate_all_saved_tensors(self)

    torch.cuda.empty_cache()
    _log_time_elapsed(self)
    _set_pass_finished(self)
