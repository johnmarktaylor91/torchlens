"""Entry removal and parameter-reference release for ``Trace``.

The bodies behind ``Trace._remove_log_entry``, ``Trace._batch_remove_log_entries`` and
``Trace._release_param_refs``; the graph-reference scrubbing they call stays in ``cleanup``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import OP_STORAGE_ATTRS, record_member
from .cleanup import (
    _LIST_FIELDS_TO_CLEAN,
    _OP_LABEL_FIELDS_TO_CLEAN,
    _clear_entry_attributes,
    _label_for_reference_removal,
    _record_mentions_removed_label,
    _remove_log_entry_references,
    _scrub_conditional_fields_after_removal,
    _scrub_per_op_equivalence_lists,
    _substitute_conditional_branch_edges,
)
from .op import Op

if TYPE_CHECKING:
    from .trace import Trace

__all__ = [
    "batch_remove_log_entries",
    "drop_orphan_entries",
    "release_param_refs",
    "remove_log_entry",
]

_LABEL_ATTRS = tuple(OP_STORAGE_ATTRS.get(name, name) for name in _OP_LABEL_FIELDS_TO_CLEAN)


def _materialize_layer_mirrors_for_removed(
    trace: Trace,
    removed_entries: Iterable[Op],
) -> None:
    """Materialize Layer mirror fields whose representative op is being removed.

    A Layer's mirror descriptors read through to its first-pass op; husking
    that op would otherwise change what a still-held Layer reads. Materializing
    first preserves the dict-era post-removal surface (the copies existed at
    this point in the dict era). Layers whose representative op survives keep
    mirroring — their reads are unaffected by the removal.
    """

    layer_logs = trace.__dict__.get("_layer_logs")
    if not layer_logs:
        return
    from .layer import _layer_rep_op, materialize_layer_mirrors

    removed_ids = {id(entry) for entry in removed_entries}
    for layer_log in layer_logs.values():
        rep = _layer_rep_op(layer_log)
        if rep is not None and id(rep) in removed_ids:
            materialize_layer_mirrors(layer_log)


def _scrub_targets(
    trace: Trace, entries: Iterable[Op], scrub_entries: Iterable[Op] | None
) -> tuple[list[Op], list[Op]]:
    """Return ``(entries to remove, survivors to scrub)``; the survivors default to all."""

    entries_to_remove = list(entries)
    if scrub_entries is not None:
        return entries_to_remove, list(scrub_entries)
    removal_ids = {id(entry) for entry in entries_to_remove}
    return entries_to_remove, [entry for entry in trace if id(entry) not in removal_ids]


def release_param_refs(trace: Trace, *, allow_iter_rehydrate: bool = False) -> None:
    """Release live ``nn.Parameter`` references held by ParamLogs.

    Parameters
    ----------
    trace:
        The owning trace.
    allow_iter_rehydrate:
        If ``True``, iterating ``param_logs`` may lazily restore live
        references from the source model. Public explicit releases leave
        this disabled.

    Returns
    -------
    None
        This method mutates ParamLogs in place.
    """
    if hasattr(trace._param_logs, "_rehydrate_on_iter"):
        trace._param_logs._rehydrate_on_iter = False
    for param_log in trace._param_logs.values():
        param_log._release_param_ref()
    if hasattr(trace._param_logs, "_rehydrate_on_iter"):
        trace._param_logs._rehydrate_on_iter = allow_iter_rehydrate


def remove_log_entry(
    trace: Trace,
    log_entry: Op,
    remove_references: bool = True,
    replacement_labels: dict[str, str] | None = None,
) -> None:
    """Remove a single layer-pass entry and scrub graph references.

    Parameters
    ----------
    trace:
        The owning trace.
    log_entry:
        Entry to remove.
    remove_references:
        Whether to scrub all graph references to the removed entry.
    replacement_labels:
        Removed-label to survivor substitutions for merge-style removals.
    """
    tensor_label = _label_for_reference_removal(log_entry, trace._tracing_finished)
    _materialize_layer_mirrors_for_removed(trace, (log_entry,))
    if remove_references:
        _remove_log_entry_references(trace, tensor_label, replacement_labels)
    _clear_entry_attributes(log_entry)


def batch_remove_log_entries(
    trace: Trace,
    entries_to_remove: Iterable[Op],
    remove_references: bool = True,
    replacement_labels: dict[str, str] | None = None,
    scrub_entries: Iterable[Op] | None = None,
) -> None:
    """Remove multiple layer-pass entries using single-pass filtering.

    Parameters
    ----------
    trace:
        The owning trace.
    entries_to_remove:
        Entries to remove.
    remove_references:
        Whether to scrub all graph references to the removed entries.
    replacement_labels:
        Removed-label to survivor substitutions for merge-style removals.
    scrub_entries:
        Survivors to scrub (default: every survivor); orphan removal passes those naming one.
    """
    entries_to_remove, surviving_entries = _scrub_targets(trace, entries_to_remove, scrub_entries)
    _materialize_layer_mirrors_for_removed(trace, entries_to_remove)
    labels_to_remove = set()
    for entry in entries_to_remove:
        labels_to_remove.add(_label_for_reference_removal(entry, trace._tracing_finished))

    if not remove_references:
        for entry in entries_to_remove:
            _clear_entry_attributes(entry)
        return

    _scrub_conditional_fields_after_removal(
        trace, labels_to_remove, surviving_entries, replacement_labels
    )

    for field_name in _LIST_FIELDS_TO_CLEAN:
        collection = getattr(trace, field_name)
        collection[:] = [label for label in collection if label not in labels_to_remove]

    trace._graph_conditional_branch_edges = _substitute_conditional_branch_edges(
        trace._graph_conditional_branch_edges, labels_to_remove, replacement_labels
    )

    for param_group, tensor_labels in list(trace.layers_with_params.items()):
        trace.layers_with_params[param_group] = [
            label for label in tensor_labels if label not in labels_to_remove
        ]
    trace.layers_with_params = {
        param_group: tensor_labels
        for param_group, tensor_labels in trace.layers_with_params.items()
        if len(tensor_labels) > 0
    }

    for _equiv_group, equivalent_label_set in list(trace._graph_op_equivalence_classes.items()):
        equivalent_label_set -= labels_to_remove
    trace._graph_op_equivalence_classes = {
        equiv_group: equivalent_label_set
        for equiv_group, equivalent_label_set in trace._graph_op_equivalence_classes.items()
        if len(equivalent_label_set) > 0
    }

    _scrub_per_op_equivalence_lists(surviving_entries, labels_to_remove)

    for entry in entries_to_remove:
        _clear_entry_attributes(entry)


def drop_orphan_entries(trace: Trace, orphan_nodes: set[str]) -> None:
    """Remove the orphan rows, scrub the survivors that name them, rebuild the layer dict.

    The general removal scrubs every survivor; here only the survivors with a
    field the scrub reads that names a removed label are scrubbed, with the
    same scrub functions, and the trace-level scrubs run unchanged. Without
    orphans nothing is scrubbed and the layer dict is kept. Call templates are
    not walked: a template's parent references name parents, which are checked.

    Parameters
    ----------
    trace:
        Trace being postprocessed (Step 3, before any label rename).
    orphan_nodes:
        Raw labels the flood did not reach.
    """

    if not orphan_nodes:
        return
    raw_layer_dict = trace._raw_graph_ws.raw_layer_dict
    orphan_entries = [raw_layer_dict[label] for label in orphan_nodes]
    kept = [
        label for label in trace._raw_graph_ws.raw_layer_labels_list if label not in orphan_nodes
    ]
    touched = [
        raw_layer_dict[label]
        for label in kept
        if _names_removed_label(raw_layer_dict[label], orphan_nodes)
    ]
    batch_remove_log_entries(trace, orphan_entries, remove_references=True, scrub_entries=touched)
    trace._raw_graph_ws.raw_layer_labels_list = kept
    trace._raw_graph_ws.raw_layer_dict = OrderedDict(
        (label, raw_layer_dict[label]) for label in kept
    )


def _names_removed_label(op: Op, removed: set[str]) -> bool:
    """Whether a field the removal scrub reads on ``op`` names a removed label."""

    for attr in _LABEL_ATTRS:
        value = getattr(op, attr, None)
        if value and not removed.isdisjoint(value):
            return True
    positions = record_member(op, "parent_arg_positions", None) or {}
    for arg_type in ("args", "kwargs"):
        if not removed.isdisjoint(positions.get(arg_type, {}).values()):
            return True
    if not removed.isdisjoint(getattr(op, "out_versions_by_child", None) or ()):
        return True
    for name in ("conditional_arm_children", "conditional_elif_children"):
        if _nested_names(record_member(op, name, None), removed):
            return True
    for edge in getattr(op, "_edge_uses", None) or ():
        if edge.parent_label in removed or edge.child_label in removed:
            return True
    return any(
        _record_mentions_removed_label(record, removed) for record in getattr(op, "edits", ())
    )


def _nested_names(children_map: Any, removed: set[str]) -> bool:
    """Whether a nested conditional child map (dict of dicts or lists) names a removed label."""

    if not children_map:
        return False
    for value in children_map.values():
        lists: Iterable[Any] = value.values() if isinstance(value, dict) else (value,)
        if any(not removed.isdisjoint(labels) for labels in lists):
            return True
    return False
