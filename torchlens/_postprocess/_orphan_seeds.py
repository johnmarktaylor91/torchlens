"""Seeds of the orphan-removal flood (postprocess Step 3).

Split out of ``graph_traversal.py`` (file cap): the initial node stack
``_remove_orphan_nodes`` floods from, and the halted-capture carve-out that
keeps every op that ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member
from ._column_passes_graph import FloodView, flood_kept_siblings

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

_MODULE_BOUNDARY_KINDS = frozenset({"module_enter", "module_exit"})


def _orphan_flood_seeds(trace: Trace) -> list[str]:
    """Return the raw labels the bidirectional orphan flood starts from.

    Seed with inputs, outputs, and written buffer-version nodes. Written
    buffers such as BatchNorm.num_batches_tracked are state transitions even
    when the updated value is not read later in the forward graph.

    A halted capture is an honest prefix of the forward: every op that ran
    stays. A value computed before the halt whose consumer never ran (a
    positional-weight table built from numpy ahead of the embedding it
    scales) is childless AND input-disconnected at the frontier, so the
    bidirectional flood from inputs and the halt output alone would prune
    it as dead computation it is not. Seed every childless op as frontier.
    """

    raw_layer_dict = trace._raw_graph_ws.raw_layer_dict
    written_buffer_layers = [
        label
        for label in trace.buffer_layers
        if getattr(raw_layer_dict[label], "buffer_write_kind", None) is not None
    ]
    node_stack = trace.input_layers + trace.output_layers + written_buffer_layers
    if trace.is_stopped:
        node_stack = node_stack + [
            label
            for label, entry in raw_layer_dict.items()
            if len(entry.children) == 0
            and label not in node_stack
            and not _is_module_boundary_row(trace, label, entry)
        ]
    return node_stack


def _is_module_boundary_row(trace: Trace, label_raw: str, entry: Any) -> bool:
    """Whether a raw row is a module enter/exit marker rather than an operation.

    The sparse recorder journals module boundaries as op-lane rows (kind
    ``module_enter`` / ``module_exit``, no parents, no children), so a cooked
    ``Recording.to_trace()`` carries them into the raw graph, where the orphan
    flood drops them. The halted carve-out above must not resurrect them: a
    boundary marker is not an op that ran, and keeping it would mint fake
    ``module_enter_N`` layers and shift every later label away from the live
    halted capture's. The live exhaustive path keeps boundaries in their own
    event lanes, so this answers False for every row it has.
    """

    capture_events = getattr(trace, "capture_events", None)
    event = None
    if capture_events is not None:
        event = capture_events.op_event_by_label_raw.get(label_raw)
    kind = getattr(event, "kind", None)
    if kind is None:
        kind = getattr(entry, "type", None)
    return kind in _MODULE_BOUNDARY_KINDS


def _expand_seen_nodes_to_complete_func_call_groups(
    self: Trace, nodes_seen: set[str], view: FloodView
) -> set[str]:
    """Add raw-label siblings for any surviving ``func_call_id`` group.

    Parameters
    ----------
    nodes_seen:
        Raw labels reachable from the input/output flood.
    view:
        The step 3 accessors (``_column_passes_graph.flood_view``).

    Returns
    -------
    set[str]
        Reachable raw labels expanded so multi-output wrapper calls are kept
        atomically.

    A sibling kept this way is flooded like any reached node, so an unused
    output of a kept call (the ``sign`` of ``torch.slogdet(weight)`` when only
    ``logabsdet`` is read, on a call with no recorded parents) is logged as an
    internal sink and seeds validation's backward walk. Before, it stayed in
    the graph unflooded: neither a sink nor reachable from one, so the
    completeness check could never reach it.

    Only groups of two or more labels can gain a member: a one-label group
    that meets ``nodes_seen`` is already inside it.
    """

    key, func_call_id = view.key, view.func_call_id
    func_groups: dict[int, set[str]] = {}
    for raw_label in self._raw_graph_ws.raw_layer_labels_list:
        call_id = func_call_id(key(raw_label))
        if call_id is not None:
            group = func_groups.get(call_id)
            if group is None:
                func_groups[call_id] = {raw_label}
            else:
                group.add(raw_label)
    multi = [labels for labels in func_groups.values() if len(labels) > 1]

    expanded_seen = set(nodes_seen)
    changed = bool(multi)
    while changed:
        changed = False
        for raw_labels in multi:
            if expanded_seen.intersection(raw_labels) and not raw_labels.issubset(expanded_seen):
                added = raw_labels - expanded_seen
                expanded_seen.update(raw_labels)
                flood_kept_siblings(self, sorted(added), expanded_seen, view)
                changed = True
    return expanded_seen


def _remember_orphan_rows(self: Trace, orphan_nodes: set[str]) -> None:
    """Retain orphan identities, records and saved payload facts before removal."""

    self._orphan_labels = [
        label for label in self._raw_graph_ws.raw_layer_labels_list if label in orphan_nodes
    ]
    self._orphan_logs = tuple(
        self._raw_graph_ws.raw_layer_dict[label] for label in self._orphan_labels
    )
    self._orphan_records = [
        {
            "raw_label": orphan._label_raw,
            "label": orphan.label,
            "payload_ref": orphan.out_ref
            if record_member(orphan, "out_ref", None) is not None
            else orphan.out,
        }
        for orphan in self._orphan_logs
        if getattr(orphan, "has_saved_activation", False)
    ]
