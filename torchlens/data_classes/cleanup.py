"""ModelLog cleanup: removing individual log entries and post-session teardown.

This module provides three levels of cleanup:

1. **cleanup()** — full teardown: deletes all LayerPassLog attributes, then
   deletes all ModelLog attributes (both FIELD_ORDER and internal containers).
   Breaks circular references (ModelLog <-> LayerPassLog.source_model_log,
   LayerLog <-> LayerPassLog.parent_layer_log, ModuleLog <-> _source_model_log).
   Also frees GPU memory via ``torch.cuda.empty_cache()``.

2. **_remove_log_entry()** — removes a single LayerPassLog and all references
   to it from ModelLog's list/dict fields (used by orphan removal).

3. **_batch_remove_log_entries()** — removes multiple entries at once using
   set-based filtering (O(N+M) instead of O(N*M) for N entries in M lists).

Both _remove_log_entry and _batch_remove_log_entries must clean the same
set of fields — if you add a new list/dict field to ModelLog that holds
layer labels, add it to both.
"""

import torch

from ..constants import MODEL_LOG_FIELD_ORDER
from ..utils.collections import remove_entry_from_list
from .layer_pass_log import LayerPassLog


def release_param_refs(self) -> None:
    """Release nn.Parameter references from all ParamLogs.

    After calling this, gradients already cached are still accessible,
    but new gradients won't be detected. The model's parameters can be
    garbage collected independently of the ModelLog.
    """
    for pl in self.param_logs:
        pl.release_param_ref()


def cleanup(self) -> None:
    """Delete all log entries, break circular references, and free GPU memory.

    Called explicitly by the user or automatically at the end of a logging
    session.  After cleanup, the ModelLog is effectively empty and should
    not be used further.
    """
    # GC-1: Release parameter references to allow model GC.
    if hasattr(self, "param_logs"):
        for pl in self.param_logs:
            pl.release_param_ref()
    # First, clear all attributes from each LayerPassLog entry.
    # This breaks the LayerPassLog -> ModelLog circular reference
    # (via source_model_log) without needing per-entry reference removal.
    for tensor_log_entry in self:
        _clear_entry_attributes(tensor_log_entry)
    # Then delete all ModelLog attributes listed in the canonical FIELD_ORDER.
    for attr in MODEL_LOG_FIELD_ORDER:
        if hasattr(self, attr):
            delattr(self, attr)
    # GC-5/GC-12: Also clear internal containers not in MODEL_LOG_FIELD_ORDER.
    # These hold back-references (e.g. _module_logs -> ModuleLog -> _source_model_log)
    # and large data structures (layer_logs, layer_dict_all_keys).
    for attr in [
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_saved_gradients_set",
        "_module_logs",
        "_buffer_accessor",
        "_module_metadata",
        "_module_forward_args",
        "_module_build_data",
        "_param_logs_by_module",
        "layer_logs",
        "layer_dict_all_keys",
        "layer_dict_main_keys",
        "orphan_layers",
        "unlogged_layers",
    ]:
        if hasattr(self, attr):
            delattr(self, attr)
    torch.cuda.empty_cache()


def _clear_entry_attributes(log_entry: LayerPassLog) -> None:
    """Clear all instance attributes from a LayerPassLog entry."""
    for attr in list(log_entry.__dict__):
        delattr(log_entry, attr)


def _remove_log_entry(self, log_entry: LayerPassLog, remove_references: bool = True) -> None:
    """Remove a single LayerPassLog and scrub all references to it.

    Used by orphan removal and other graph-pruning operations.

    Args:
        log_entry: The LayerPassLog to destroy.
        remove_references: If True, also remove the entry's label from
            every list/dict field on the ModelLog.
    """
    # The label used to find references depends on whether postprocessing
    # has run: after postprocessing, layers are keyed by their final
    # human-readable label; during the pass, by the raw internal barcode.
    if self._pass_finished:
        tensor_label = log_entry.layer_label
    else:
        tensor_label = log_entry.tensor_label_raw
    _clear_entry_attributes(log_entry)
    if remove_references:
        _remove_log_entry_references(self, tensor_label)


# List fields on ModelLog that hold tensor labels and need filtering during
# entry removal.  Must stay in sync between _batch_remove_log_entries and
# _remove_log_entry_references — if you add a new label-holding list field
# to ModelLog, add it here AND to _remove_log_entry_references.
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


def _batch_remove_log_entries(self, entries_to_remove, remove_references: bool = True) -> None:
    """Remove multiple LayerPassLog entries at once using set-based filtering.

    More efficient than calling ``_remove_log_entry`` in a loop: builds a
    set of labels to remove, then does a single pass over each list/dict
    field (O(N+M) instead of O(N*M) where N=entries, M=list length).

    Args:
        entries_to_remove: Iterable of LayerPassLog objects to remove.
        remove_references: Whether to also remove references from ModelLog list/dict fields.
    """
    # Build a set of labels for O(1) membership testing, then clear each entry.
    labels_to_remove = set()
    for entry in entries_to_remove:
        if self._pass_finished:
            labels_to_remove.add(entry.layer_label)
        else:
            labels_to_remove.add(entry.tensor_label_raw)
        _clear_entry_attributes(entry)

    if not remove_references:
        return

    # Single-pass filter on each list field (replaces N x remove_entry_from_list calls).
    for field in _LIST_FIELDS_TO_CLEAN:
        collection = getattr(self, field)
        collection[:] = [label for label in collection if label not in labels_to_remove]

    self.conditional_branch_edges = [
        edge
        for edge in self.conditional_branch_edges
        if edge[0] not in labels_to_remove and edge[1] not in labels_to_remove
    ]
    self.conditional_then_edges = [
        edge
        for edge in self.conditional_then_edges
        if edge[0] not in labels_to_remove and edge[1] not in labels_to_remove
    ]

    # Single-pass filter on dict fields:
    # layers_with_params: param_barcode -> [layer_labels]
    for param_group, tensor_labels in list(self.layers_with_params.items()):
        self.layers_with_params[param_group] = [
            label for label in tensor_labels if label not in labels_to_remove
        ]
    self.layers_with_params = {
        param_group: tensor_labels
        for param_group, tensor_labels in self.layers_with_params.items()
        if len(tensor_labels) > 0
    }

    # equivalent_operations: equiv_type -> set(layer_labels)
    for equiv_group, tensor_labels in list(self.equivalent_operations.items()):
        tensor_labels -= labels_to_remove
    self.equivalent_operations = {
        equiv_group: tensor_labels
        for equiv_group, tensor_labels in self.equivalent_operations.items()
        if len(tensor_labels) > 0
    }


def _remove_log_entry_references(self, layer_to_remove: str) -> None:
    """Removes all references to a single LayerPassLog from the ModelLog's list/dict fields.

    This is the single-entry counterpart to the reference-cleaning logic in
    ``_batch_remove_log_entries``. Both must clean the same set of fields —
    if you add a new field to one, update the other as well.

    Args:
        layer_to_remove: The label of the log entry to remove.
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
        edge for edge in self.conditional_branch_edges if layer_to_remove not in edge
    ]
    self.conditional_then_edges = [
        edge for edge in self.conditional_then_edges if layer_to_remove not in edge
    ]

    # Now any nested fields.

    for param_group, tensor_labels in self.layers_with_params.items():
        if layer_to_remove in tensor_labels:
            tensor_labels.remove(layer_to_remove)
    self.layers_with_params = {
        param_group: tensor_labels
        for param_group, tensor_labels in self.layers_with_params.items()
        if len(tensor_labels) > 0
    }

    for equiv_group, tensor_labels in self.equivalent_operations.items():
        if layer_to_remove in tensor_labels:
            tensor_labels.remove(layer_to_remove)
    self.equivalent_operations = {
        equiv_group: tensor_labels
        for equiv_group, tensor_labels in self.equivalent_operations.items()
        if len(tensor_labels) > 0
    }
