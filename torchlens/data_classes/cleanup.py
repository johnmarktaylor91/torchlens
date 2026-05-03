"""Trace cleanup helpers and post-session teardown.

This module provides the helper stack behind Trace cleanup operations:

1. **cleanup()** — full teardown: deletes all OpLog attributes, then
   deletes all Trace attributes (both FIELD_ORDER and internal containers).
   Breaks circular references (Trace <-> OpLog.source_trace,
   LayerLog <-> OpLog.parent_layer_log, ModuleLog <-> _source_trace).
   Also frees GPU memory via ``torch.cuda.empty_cache()`` when CUDA is
   available (gated to avoid CUDA driver probe cost on CPU-only runs).

2. **_remove_log_entry_references()** — removes a single layer label from all
   Trace list/dict fields that hold graph references.

3. **_scrub_conditional_fields_after_removal()** — repairs conditional metadata
   after one or more labels are removed.

4. **_LIST_FIELDS_TO_CLEAN** — canonical list fields that must stay aligned
   with the removal helpers.
"""

from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Set, Tuple, cast

import torch

from ..constants import MODEL_LOG_FIELD_ORDER
from ..intervention.types import ParentRef, Unsupported
from ..utils.collections import remove_entry_from_list
from ..utils.tensor_utils import _is_cuda_available
from .op_log import OpLog

if TYPE_CHECKING:
    from .model_log import Trace


def cleanup(self: "Trace") -> None:
    """Delete all log entries, break circular references, and free GPU memory.

    Called explicitly by the user or automatically at the end of a logging
    session. After cleanup, the Trace is effectively empty and should
    not be used further. No long-lived safetensors handles need to be
    closed here because lazy materialization opens and closes files per call.
    """
    # GC-1: Release parameter references to allow model GC.
    if hasattr(self, "param_logs"):
        for pl in self.param_logs:
            pl.release_param_ref()
    # First, clear all attributes from each OpLog entry.
    # This breaks the OpLog -> Trace circular reference
    # (via source_trace) without needing per-entry reference removal.
    for tensor_log_entry in self:
        _clear_entry_attributes(tensor_log_entry)
    # Then delete all Trace attributes listed in the canonical FIELD_ORDER.
    for attr in MODEL_LOG_FIELD_ORDER:
        if hasattr(self, attr):
            delattr(self, attr)
    # GC-5/GC-12: Also clear internal containers not in MODEL_LOG_FIELD_ORDER.
    # These hold back-references (e.g. _module_logs -> ModuleLog -> _source_trace)
    # and large data structures (layer_logs, layer_dict_all_keys).
    for attr in [
        "_raw_layer_dict",
        "_raw_layer_labels_list",
        "_saved_grads_set",
        "_module_logs",
        "_buffer_accessor",
        "_module_metadata",
        "_module_forward_args",
        "_module_build_data",
        "_param_logs_by_module",
        "layer_logs",
        "layer_dict_all_keys",
        "layer_dict_main_keys",
        "orphan_ops",
        "unlogged_ops",
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


def _clear_entry_attributes(log_entry: OpLog) -> None:
    """Clear all instance attributes from a OpLog entry."""
    if hasattr(log_entry, "out_ref"):
        log_entry.out_ref = None
    if hasattr(log_entry, "grad_ref"):
        log_entry.grad_ref = None
    for attr in list(log_entry.__dict__):
        delattr(log_entry, attr)


def _strip_pass_suffix(layer_label: str) -> str:
    """Remove any ``:call_index`` suffix from a layer label.

    Args:
        layer_label: Layer label, optionally pass-qualified.

    Returns:
        The pass-stripped label.
    """
    return layer_label.split(":", 1)[0]


def _label_for_reference_removal(log_entry: OpLog, pass_finished: bool) -> str:
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
        return cast(str, log_entry.layer_label)
    if getattr(log_entry, "layer_label", None):
        return cast(str, log_entry.layer_label)
    return cast(str, log_entry._label_raw)


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


def _filter_conditional_edge_ops(
    conditional_edge_ops: Dict[Tuple[str, str, int, str], List[int]],
    labels_to_remove_no_pass: Set[str],
) -> Dict[Tuple[str, str, int, str], List[int]]:
    """Drop removed labels from ``conditional_edge_ops`` keys.

    Args:
        conditional_edge_ops: ``(parent_no_pass, child_no_pass, cond_id, branch_kind) -> pass list``.
        labels_to_remove_no_pass: Pass-stripped labels that should be removed.

    Returns:
        A new dict with removed-key entries pruned.
    """
    return {
        key: call_indexs
        for key, call_indexs in conditional_edge_ops.items()
        if key[0] not in labels_to_remove_no_pass and key[1] not in labels_to_remove_no_pass
    }


def _scrub_layer_entry_conditional_fields(
    layer_entry: OpLog,
    labels_to_remove: Set[str],
) -> None:
    """Remove deleted labels from conditional fields on a surviving OpLog.

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


def _scrub_layer_log_conditional_fields(self: "Trace", labels_to_remove_no_pass: Set[str]) -> None:
    """Remove deleted labels from aggregate LayerLog conditional fields.

    Args:
        self: Trace owning the LayerLogs.
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
    self: "Trace",
    labels_to_remove: Set[str],
    surviving_entries: Iterable[OpLog],
) -> None:
    """Scrub conditional references after one or more layer labels are removed.

    Args:
        self: Trace being updated.
        labels_to_remove: Removed layer labels using the same qualification as the
            current removal pass.
        surviving_entries: Surviving OpLog entries to scrub in-place.
    """
    labels_to_remove_no_pass = {_strip_pass_suffix(layer_label) for layer_label in labels_to_remove}

    for layer_entry in surviving_entries:
        _scrub_layer_entry_conditional_fields(layer_entry, labels_to_remove)

    _scrub_layer_log_conditional_fields(self, labels_to_remove_no_pass)

    self.conditional_arm_edges = _filter_conditional_arm_edges(
        self.conditional_arm_edges,
        labels_to_remove,
    )
    self.conditional_edge_ops = _filter_conditional_edge_ops(
        self.conditional_edge_ops,
        labels_to_remove_no_pass,
    )
    for conditional_event in self.conditional_events:
        conditional_event.bool_layers = [
            layer_label
            for layer_label in conditional_event.bool_layers
            if layer_label not in labels_to_remove
        ]

    _scrub_intervention_fields_after_removal(self, labels_to_remove, surviving_entries)


def _scrub_intervention_fields_after_removal(
    self: Any,
    labels_to_remove: Set[str],
    surviving_entries: Iterable[OpLog],
) -> None:
    """Scrub replay/intervention metadata that carries layer labels.

    Args:
        self: Trace being updated.
        labels_to_remove: Removed labels in the active label namespace.
        surviving_entries: Surviving entries to scrub in-place.
    """

    for layer_entry in surviving_entries:
        layer_entry.edge_uses = [
            edge
            for edge in getattr(layer_entry, "edge_uses", [])
            if edge.parent_label not in labels_to_remove
            and edge.child_label not in labels_to_remove
        ]
        layer_entry.args_template = _replace_removed_parent_refs(
            getattr(layer_entry, "args_template", None), labels_to_remove
        )
        layer_entry.kwargs_template = _replace_removed_parent_refs(
            getattr(layer_entry, "kwargs_template", None), labels_to_remove
        )
        interventions = [
            record
            for record in getattr(layer_entry, "interventions", [])
            if not _record_mentions_removed_label(record, labels_to_remove)
        ]
        if hasattr(layer_entry, "_internal_set"):
            layer_entry._internal_set("interventions", interventions)
        else:
            layer_entry.interventions = interventions

    self.operation_history = [
        record
        for record in getattr(self, "operation_history", [])
        if not _record_mentions_removed_label(record, labels_to_remove)
    ]
    intervention_spec = getattr(self, "_intervention_spec", None)
    if intervention_spec is not None:
        intervention_spec.records = [
            record
            for record in getattr(intervention_spec, "records", [])
            if not _record_mentions_removed_label(record, labels_to_remove)
        ]


def _replace_removed_parent_refs(value: Any, labels_to_remove: Set[str]) -> Any:
    """Replace template parent refs to removed labels with unsupported leaves.

    Args:
        value: Template or nested component.
        labels_to_remove: Removed layer labels.

    Returns:
        Template value with stale parent refs replaced.
    """

    if isinstance(value, ParentRef) and value.parent_label in labels_to_remove:
        return Unsupported(reason="removed_parent_ref", value_type="ParentRef")
    if isinstance(value, tuple):
        return tuple(_replace_removed_parent_refs(item, labels_to_remove) for item in value)
    if isinstance(value, list):
        return [_replace_removed_parent_refs(item, labels_to_remove) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_removed_parent_refs(item, labels_to_remove) for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _replace_removed_parent_refs(getattr(value, field.name), labels_to_remove)
            for field in fields(value)
            if hasattr(value, field.name)
        }
        return replace(value, **updates)
    return value


def _record_mentions_removed_label(record: Any, labels_to_remove: Set[str]) -> bool:
    """Return whether a record contains a removed label-bearing field.

    Args:
        record: Dataclass record or nested object.
        labels_to_remove: Removed labels.

    Returns:
        Whether the record references a removed label.
    """

    label_fields = {"parent_label", "child_label", "target_label", "call_label", "site_label"}
    if isinstance(record, str):
        return record in labels_to_remove
    if isinstance(record, (list, tuple)):
        return any(_record_mentions_removed_label(item, labels_to_remove) for item in record)
    if isinstance(record, dict):
        return any(
            _record_mentions_removed_label(key, labels_to_remove)
            or _record_mentions_removed_label(value, labels_to_remove)
            for key, value in record.items()
        )
    if is_dataclass(record) and not isinstance(record, type):
        for field in fields(record):
            if not hasattr(record, field.name):
                continue
            field_value = getattr(record, field.name)
            if field.name in label_fields and field_value in labels_to_remove:
                return True
            if field.name not in label_fields and _record_mentions_removed_label(
                field_value, labels_to_remove
            ):
                return True
    return False


# List fields on Trace that hold tensor labels and need filtering during
# entry removal.  Must stay in sync between _batch_remove_log_entries and
# _remove_log_entry_references — if you add a new label-holding list field
# to Trace, add it here AND to _remove_log_entry_references.
_LIST_FIELDS_TO_CLEAN = [
    "input_layers",
    "output_layers",
    "buffer_layers",
    "internally_initialized_ops",
    "internally_terminated_ops",
    "internally_terminated_bool_ops",
    "ops_with_saved_outs",
    "ops_with_saved_grads",
    "_layers_where_internal_branches_merge_with_input",
]


def _remove_log_entry_references(self: "Trace", layer_to_remove: str) -> None:
    """Removes all references to a single OpLog from the Trace's list/dict fields.

    This is the single-entry counterpart to the reference-cleaning logic in
    ``_batch_remove_log_entries``. Both must clean the same set of fields —
    if you add a new field to one, update the other as well.

    Args:
        layer_to_remove: The label of the log entry to remove.
    """
    # Clear any fields in Trace referring to the entry.

    remove_entry_from_list(self.input_layers, layer_to_remove)
    remove_entry_from_list(self.output_layers, layer_to_remove)
    remove_entry_from_list(self.buffer_layers, layer_to_remove)
    remove_entry_from_list(self.internally_initialized_ops, layer_to_remove)
    remove_entry_from_list(self.internally_terminated_ops, layer_to_remove)
    remove_entry_from_list(self.internally_terminated_bool_ops, layer_to_remove)
    remove_entry_from_list(self.ops_with_saved_outs, layer_to_remove)
    remove_entry_from_list(self.ops_with_saved_grads, layer_to_remove)
    remove_entry_from_list(self._layers_where_internal_branches_merge_with_input, layer_to_remove)

    _scrub_conditional_fields_after_removal(self, {layer_to_remove}, self)

    self.conditional_branch_edges = [
        edge for edge in self.conditional_branch_edges if layer_to_remove not in edge
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

    for equiv_group, equiv_tensor_labels in self.equivalent_ops.items():
        if layer_to_remove in equiv_tensor_labels:
            equiv_tensor_labels.remove(layer_to_remove)
    self.equivalent_ops = {
        equiv_group: tensor_labels
        for equiv_group, tensor_labels in self.equivalent_ops.items()
        if len(tensor_labels) > 0
    }
