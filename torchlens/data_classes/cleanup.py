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

from typing import Dict, Iterable, List, Set, Tuple

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


def _strip_pass_suffix(layer_label: str) -> str:
    """Remove any ``:pass_num`` suffix from a layer label.

    Args:
        layer_label: Layer label, optionally pass-qualified.

    Returns:
        The pass-stripped label.
    """
    return layer_label.split(":", 1)[0]


def _filter_cond_branch_children_by_cond(
    cond_branch_children_by_cond: Dict[int, Dict[str, List[str]]],
    labels_to_remove: Set[str],
) -> Dict[int, Dict[str, List[str]]]:
    """Drop removed labels from ``cond_branch_children_by_cond``.

    Args:
        cond_branch_children_by_cond: ``cond_id -> branch_kind -> child labels``.
        labels_to_remove: Labels that should be removed.

    Returns:
        A new nested dict with removed labels and empty containers pruned.
    """
    filtered_children_by_cond: Dict[int, Dict[str, List[str]]] = {}
    for cond_id, branch_children in cond_branch_children_by_cond.items():
        filtered_branch_children = {
            branch_kind: [
                child_label for child_label in child_labels if child_label not in labels_to_remove
            ]
            for branch_kind, child_labels in branch_children.items()
        }
        filtered_branch_children = {
            branch_kind: child_labels
            for branch_kind, child_labels in filtered_branch_children.items()
            if child_labels
        }
        if filtered_branch_children:
            filtered_children_by_cond[cond_id] = filtered_branch_children
    return filtered_children_by_cond


def _filter_cond_branch_elif_children(
    cond_branch_elif_children: Dict[int, List[str]],
    labels_to_remove: Set[str],
) -> Dict[int, List[str]]:
    """Drop removed labels from ``cond_branch_elif_children``.

    Args:
        cond_branch_elif_children: ``elif_index -> child labels``.
        labels_to_remove: Labels that should be removed.

    Returns:
        A new dict with removed labels and empty lists pruned.
    """
    return {
        elif_ix: [
            child_label for child_label in child_labels if child_label not in labels_to_remove
        ]
        for elif_ix, child_labels in cond_branch_elif_children.items()
        if any(child_label not in labels_to_remove for child_label in child_labels)
    }


def _filter_conditional_arm_edges(
    conditional_arm_edges: Dict[Tuple[int, str], List[Tuple[str, str]]],
    labels_to_remove: Set[str],
) -> Dict[Tuple[int, str], List[Tuple[str, str]]]:
    """Drop removed labels from ``conditional_arm_edges``.

    Args:
        conditional_arm_edges: ``(cond_id, branch_kind) -> [(parent, child)]``.
        labels_to_remove: Labels that should be removed.

    Returns:
        A new dict with empty edge lists pruned.
    """
    filtered_arm_edges: Dict[Tuple[int, str], List[Tuple[str, str]]] = {}
    for key, edge_list in conditional_arm_edges.items():
        filtered_edges = [
            (parent, child)
            for parent, child in edge_list
            if parent not in labels_to_remove and child not in labels_to_remove
        ]
        if filtered_edges:
            filtered_arm_edges[key] = filtered_edges
    return filtered_arm_edges


def _filter_conditional_edge_passes(
    conditional_edge_passes: Dict[Tuple[str, str, int, str], List[int]],
    labels_to_remove_no_pass: Set[str],
) -> Dict[Tuple[str, str, int, str], List[int]]:
    """Drop removed labels from ``conditional_edge_passes`` keys.

    Args:
        conditional_edge_passes: ``(parent_no_pass, child_no_pass, cond_id, branch_kind) -> pass list``.
        labels_to_remove_no_pass: Pass-stripped labels that should be removed.

    Returns:
        A new dict with removed-key entries pruned.
    """
    return {
        key: pass_nums
        for key, pass_nums in conditional_edge_passes.items()
        if key[0] not in labels_to_remove_no_pass and key[1] not in labels_to_remove_no_pass
    }


def _scrub_layer_entry_conditional_fields(
    layer_entry: LayerPassLog,
    labels_to_remove: Set[str],
) -> None:
    """Remove deleted labels from conditional fields on a surviving LayerPassLog.

    Args:
        layer_entry: Surviving layer entry to scrub.
        labels_to_remove: Labels that were removed elsewhere in the log.
    """
    layer_entry.cond_branch_start_children = [
        child_label
        for child_label in layer_entry.cond_branch_start_children
        if child_label not in labels_to_remove
    ]
    layer_entry.cond_branch_children_by_cond = _filter_cond_branch_children_by_cond(
        layer_entry.cond_branch_children_by_cond,
        labels_to_remove,
    )
    layer_entry.cond_branch_then_children = [
        child_label
        for child_label in layer_entry.cond_branch_then_children
        if child_label not in labels_to_remove
    ]
    layer_entry.cond_branch_elif_children = _filter_cond_branch_elif_children(
        layer_entry.cond_branch_elif_children,
        labels_to_remove,
    )
    layer_entry.cond_branch_else_children = [
        child_label
        for child_label in layer_entry.cond_branch_else_children
        if child_label not in labels_to_remove
    ]


def _scrub_layer_log_conditional_fields(self, labels_to_remove_no_pass: Set[str]) -> None:
    """Remove deleted labels from aggregate LayerLog conditional fields.

    Args:
        self: ModelLog owning the LayerLogs.
        labels_to_remove_no_pass: Pass-stripped labels that were removed.
    """
    for layer_log in getattr(self, "layer_logs", {}).values():
        layer_log.cond_branch_start_children = [
            child_label
            for child_label in layer_log.cond_branch_start_children
            if child_label not in labels_to_remove_no_pass
        ]
        layer_log.cond_branch_children_by_cond = _filter_cond_branch_children_by_cond(
            layer_log.cond_branch_children_by_cond,
            labels_to_remove_no_pass,
        )
        layer_log.cond_branch_then_children = [
            child_label
            for child_label in layer_log.cond_branch_then_children
            if child_label not in labels_to_remove_no_pass
        ]
        layer_log.cond_branch_elif_children = _filter_cond_branch_elif_children(
            layer_log.cond_branch_elif_children,
            labels_to_remove_no_pass,
        )
        layer_log.cond_branch_else_children = [
            child_label
            for child_label in layer_log.cond_branch_else_children
            if child_label not in labels_to_remove_no_pass
        ]


def _scrub_conditional_fields_after_removal(
    self,
    labels_to_remove: Set[str],
    surviving_entries: Iterable[LayerPassLog],
) -> None:
    """Scrub conditional references after one or more layer labels are removed.

    Args:
        self: ModelLog being updated.
        labels_to_remove: Removed layer labels using the same qualification as the
            current removal pass.
        surviving_entries: Surviving LayerPassLog entries to scrub in-place.
    """
    labels_to_remove_no_pass = {_strip_pass_suffix(layer_label) for layer_label in labels_to_remove}

    for layer_entry in surviving_entries:
        _scrub_layer_entry_conditional_fields(layer_entry, labels_to_remove)

    _scrub_layer_log_conditional_fields(self, labels_to_remove_no_pass)

    self.conditional_arm_edges = _filter_conditional_arm_edges(
        self.conditional_arm_edges,
        labels_to_remove,
    )
    self.conditional_edge_passes = _filter_conditional_edge_passes(
        self.conditional_edge_passes,
        labels_to_remove_no_pass,
    )
    self.conditional_then_edges = [
        edge
        for edge in self.conditional_then_edges
        if edge[0] not in labels_to_remove and edge[1] not in labels_to_remove
    ]
    self.conditional_elif_edges = [
        edge
        for edge in self.conditional_elif_edges
        if edge[2] not in labels_to_remove and edge[3] not in labels_to_remove
    ]
    self.conditional_else_edges = [
        edge
        for edge in self.conditional_else_edges
        if edge[1] not in labels_to_remove and edge[2] not in labels_to_remove
    ]
    for conditional_event in self.conditional_events:
        conditional_event.bool_layers = [
            layer_label
            for layer_label in conditional_event.bool_layers
            if layer_label not in labels_to_remove
        ]


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
    if remove_references:
        _remove_log_entry_references(self, tensor_label)
    _clear_entry_attributes(log_entry)


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
    entries_to_remove = list(entries_to_remove)
    surviving_entries = [entry for entry in self if entry not in entries_to_remove]

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

    _scrub_conditional_fields_after_removal(self, labels_to_remove, surviving_entries)

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

    _scrub_conditional_fields_after_removal(self, {layer_to_remove}, self)

    self.conditional_branch_edges = [
        edge for edge in self.conditional_branch_edges if layer_to_remove not in edge
    ]
    self.conditional_then_edges = [
        edge for edge in self.conditional_then_edges if layer_to_remove not in edge
    ]
    self.conditional_elif_edges = [
        edge for edge in self.conditional_elif_edges if layer_to_remove not in (edge[2], edge[3])
    ]
    self.conditional_else_edges = [
        edge for edge in self.conditional_else_edges if layer_to_remove not in (edge[1], edge[2])
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
