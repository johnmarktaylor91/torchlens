"""ModelLog cleanup helpers and post-session teardown.

This module provides the helper stack behind ModelLog cleanup operations:

1. **cleanup()** — full teardown: deletes all LayerPassLog attributes, then
   deletes all ModelLog attributes (both FIELD_ORDER and internal containers).
   Breaks circular references (ModelLog <-> LayerPassLog.source_model_log,
   LayerLog <-> LayerPassLog.parent_layer_log, ModuleLog <-> _source_model_log).
   Also frees GPU memory via ``torch.cuda.empty_cache()`` when CUDA is
   available (gated to avoid CUDA driver probe cost on CPU-only runs).

2. **_remove_log_entry_references()** — removes a single layer label from all
   ModelLog list/dict fields that hold graph references.

3. **_scrub_conditional_fields_after_removal()** — repairs conditional metadata
   after one or more labels are removed.

4. **_LIST_FIELDS_TO_CLEAN** — canonical list fields that must stay aligned
   with the removal helpers.
"""

from typing import Dict, Iterable, List, Set, Tuple

import torch

from ..constants import MODEL_LOG_FIELD_ORDER
from ..utils.collections import remove_entry_from_list
from ..utils.tensor_utils import _is_cuda_available
from .layer_pass_log import LayerPassLog


def cleanup(self) -> None:
    """Delete all log entries, break circular references, and free GPU memory.

    Called explicitly by the user or automatically at the end of a logging
    session. After cleanup, the ModelLog is effectively empty and should
    not be used further. No long-lived safetensors handles need to be
    closed here because lazy materialization opens and closes files per call.
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
        "_loaded_from_bundle",
        "_source_bundle_manifest_sha256",
        "_source_bundle_path",
    ]:
        if hasattr(self, attr):
            delattr(self, attr)
    # Gated behind cached cuda.is_available() so CPU-only runs don't pay the
    # CUDA driver / NVML probe cost (per profiling audit 2026-04-27 finding #4).
    if _is_cuda_available():
        torch.cuda.empty_cache()


def _clear_entry_attributes(log_entry: LayerPassLog) -> None:
    """Clear all instance attributes from a LayerPassLog entry."""
    if hasattr(log_entry, "activation_ref"):
        log_entry.activation_ref = None
    if hasattr(log_entry, "gradient_ref"):
        log_entry.gradient_ref = None
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


def _label_for_reference_removal(log_entry: LayerPassLog, pass_finished: bool) -> str:
    """Return the label namespace currently used by graph-level references.

    Parameters
    ----------
    log_entry:
        Entry being removed.
    pass_finished:
        Whether postprocessing has fully completed.

    Returns
    -------
    str
        Final layer label when available, otherwise the raw tensor label.
    """
    if pass_finished:
        return log_entry.layer_label
    if getattr(log_entry, "layer_label", None):
        return log_entry.layer_label
    return log_entry.tensor_label_raw


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
