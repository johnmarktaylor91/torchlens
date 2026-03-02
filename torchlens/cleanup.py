import warnings

import torch

from .constants import MODEL_LOG_FIELD_ORDER
from .helper_funcs import remove_entry_from_list
from .data_classes.tensor_log import TensorLog


def cleanup(self):
    """Deletes all log entries in the model."""
    for tensor_log_entry in self:
        self._remove_log_entry(tensor_log_entry, remove_references=True)
    for attr in MODEL_LOG_FIELD_ORDER:
        delattr(self, attr)
    torch.cuda.empty_cache()


def _clear_entry_attributes(log_entry: TensorLog):
    """Clear all instance attributes from a TensorLog entry."""
    for attr in dir(log_entry):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if attr.startswith("__"):
                continue
            if isinstance(getattr(type(log_entry), attr, None), property):
                continue
            if not callable(getattr(log_entry, attr, None)):
                delattr(log_entry, attr)


def _remove_log_entry(self, log_entry: TensorLog, remove_references: bool = True):
    """Given a TensorLog, destroys it and all references to it.

    Args:
        log_entry: Tensor log entry to remove.
        remove_references: Whether to also remove references to the log entry
    """
    if self._pass_finished:
        tensor_label = log_entry.layer_label
    else:
        tensor_label = log_entry.tensor_label_raw
    _clear_entry_attributes(log_entry)
    del log_entry
    if remove_references:
        _remove_log_entry_references(self, tensor_label)


# List fields on ModelLog that hold tensor labels and need filtering during batch removal.
_LIST_FIELDS_TO_CLEAN = [
    "input_layers",
    "output_layers",
    "buffer_layers",
    "internally_initialized_layers",
    "internally_terminated_layers",
    "internally_terminated_bool_layers",
    "layers_with_saved_activations",
    "layers_with_saved_gradients",
    "_layers_where_internal_branches_merge_with_input",
]


def _batch_remove_log_entries(self, entries_to_remove, remove_references: bool = True):
    """Remove multiple TensorLog entries at once, avoiding O(N*M) list.remove() calls.

    Args:
        entries_to_remove: Iterable of TensorLog objects to remove.
        remove_references: Whether to also remove references from ModelLog list/dict fields.
    """
    # Collect labels for O(1) lookup, then clear each entry's attributes.
    labels_to_remove = set()
    for entry in entries_to_remove:
        if self._pass_finished:
            labels_to_remove.add(entry.layer_label)
        else:
            labels_to_remove.add(entry.tensor_label_raw)
        _clear_entry_attributes(entry)

    if not remove_references:
        return

    # Single-pass filter on each list field (replaces N × remove_entry_from_list calls).
    for field in _LIST_FIELDS_TO_CLEAN:
        collection = getattr(self, field)
        collection[:] = [x for x in collection if x not in labels_to_remove]

    self.conditional_branch_edges = [
        tup
        for tup in self.conditional_branch_edges
        if tup[0] not in labels_to_remove and tup[1] not in labels_to_remove
    ]

    # Single-pass filter on dict fields.
    # layers_computed_with_params values are lists.
    for group_label, group_tensors in list(self.layers_computed_with_params.items()):
        self.layers_computed_with_params[group_label] = [
            t for t in group_tensors if t not in labels_to_remove
        ]
    self.layers_computed_with_params = {
        k: v for k, v in self.layers_computed_with_params.items() if len(v) > 0
    }

    # equivalent_operations values are sets.
    for group_label, group_tensors in list(self.equivalent_operations.items()):
        group_tensors -= labels_to_remove
    self.equivalent_operations = {k: v for k, v in self.equivalent_operations.items() if len(v) > 0}


def _remove_log_entry_references(self, layer_to_remove: str):
    """Removes all references to a given TensorLog in the ModelLog object.

    Args:
        layer_to_remove: The log entry to remove.
    """
    # Clear any fields in ModelLog referring to the entry.

    remove_entry_from_list(self.input_layers, layer_to_remove)
    remove_entry_from_list(self.output_layers, layer_to_remove)
    remove_entry_from_list(self.buffer_layers, layer_to_remove)
    remove_entry_from_list(self.internally_initialized_layers, layer_to_remove)
    remove_entry_from_list(self.internally_terminated_layers, layer_to_remove)
    remove_entry_from_list(self.internally_terminated_bool_layers, layer_to_remove)
    remove_entry_from_list(self.layers_with_saved_activations, layer_to_remove)
    remove_entry_from_list(self.layers_with_saved_gradients, layer_to_remove)
    remove_entry_from_list(self._layers_where_internal_branches_merge_with_input, layer_to_remove)

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
    self.equivalent_operations = {k: v for k, v in self.equivalent_operations.items() if len(v) > 0}
