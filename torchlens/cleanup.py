import warnings

import torch

from .constants import MODEL_HISTORY_FIELD_ORDER
from .helper_funcs import remove_entry_from_list
from .tensor_log import TensorLogEntry


def cleanup(self):
    """Deletes all log entries in the model."""
    for tensor_log_entry in self:
        self._remove_log_entry(tensor_log_entry, remove_references=True)
    for attr in MODEL_HISTORY_FIELD_ORDER:
        delattr(self, attr)
    torch.cuda.empty_cache()


def _remove_log_entry(
        self, log_entry: TensorLogEntry, remove_references: bool = True
):
    """Given a TensorLogEntry, destroys it and all references to it.

    Args:
        log_entry: Tensor log entry to remove.
        remove_references: Whether to also remove references to the log entry
    """
    if self._pass_finished:
        tensor_label = log_entry.layer_label
    else:
        tensor_label = log_entry.tensor_label_raw
    for attr in dir(log_entry):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not attr.startswith("_") and not callable(getattr(log_entry, attr)):
                delattr(log_entry, attr)
    del log_entry
    if remove_references:
        _remove_log_entry_references(self, tensor_label)


def _remove_log_entry_references(self, layer_to_remove: str):
    """Removes all references to a given TensorLogEntry in the ModelHistory object.

    Args:
        layer_to_remove: The log entry to remove.
    """
    # Clear any fields in ModelHistory referring to the entry.

    remove_entry_from_list(self.input_layers, layer_to_remove)
    remove_entry_from_list(self.output_layers, layer_to_remove)
    remove_entry_from_list(self.buffer_layers, layer_to_remove)
    remove_entry_from_list(self.internally_initialized_layers, layer_to_remove)
    remove_entry_from_list(self.internally_terminated_layers, layer_to_remove)
    remove_entry_from_list(self.internally_terminated_bool_layers, layer_to_remove)
    remove_entry_from_list(self.layers_with_saved_activations, layer_to_remove)
    remove_entry_from_list(self.layers_with_saved_gradients, layer_to_remove)
    remove_entry_from_list(
        self._layers_where_internal_branches_merge_with_input, layer_to_remove
    )

    self.conditional_branch_edges = [
        tup for tup in self.conditional_branch_edges if layer_to_remove not in tup
    ]

    # Now any nested fields.

    for group_label, group_tensors in self.layers_computed_with_params.items():
        if layer_to_remove in group_tensors:
            group_tensors.remove(layer_to_remove)
    self.layers_computed_with_params = {
        k: v for k, v in self.layers_computed_with_params.items() if len(v) > 0
    }

    for group_label, group_tensors in self.equivalent_operations.items():
        if layer_to_remove in group_tensors:
            group_tensors.remove(layer_to_remove)
    self.equivalent_operations = {
        k: v for k, v in self.equivalent_operations.items() if len(v) > 0
    }

    for group_label, group_tensors in self.same_layer_operations.items():
        if layer_to_remove in group_tensors:
            group_tensors.remove(layer_to_remove)
    self.same_layer_operations = {
        k: v for k, v in self.same_layer_operations.items() if len(v) > 0
    }
