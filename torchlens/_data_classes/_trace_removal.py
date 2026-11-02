"""Entry removal and parameter-reference release for ``Trace``.

The bodies behind ``Trace._remove_log_entry``, ``Trace._batch_remove_log_entries`` and
``Trace._release_param_refs``; the graph-reference scrubbing they call stays in ``cleanup``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .cleanup import (
    _LIST_FIELDS_TO_CLEAN,
    _clear_entry_attributes,
    _label_for_reference_removal,
    _remove_log_entry_references,
    _scrub_conditional_fields_after_removal,
    _scrub_per_op_equivalence_lists,
    _substitute_conditional_branch_edges,
)
from .op import Op

if TYPE_CHECKING:
    from .trace import Trace

__all__ = ["batch_remove_log_entries", "release_param_refs", "remove_log_entry"]


def _materialize_layer_mirrors_for_removed(
    trace: Trace,
    removed_entries: Iterable[Op],
) -> None:
    """Materialize Layer mirror fields whose representative op is being removed.

    A Layer's M8 mirror descriptors read through to its first-pass op; husking
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
        Optional removed-label -> survivor substitutions for merge-style
        removals: conditional references repoint to the survivor instead
        of being dropped.
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
        Optional removed-label -> survivor substitutions for merge-style
        removals: conditional references repoint to the survivor instead
        of being dropped.
    """
    entries_to_remove = list(entries_to_remove)
    removal_ids = {id(entry) for entry in entries_to_remove}
    surviving_entries = [entry for entry in trace if id(entry) not in removal_ids]
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
