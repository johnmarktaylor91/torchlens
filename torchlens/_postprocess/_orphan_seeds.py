"""Seeds of the orphan-removal flood (postprocess Step 3).

Split out of ``graph_traversal.py`` (file cap): the initial node stack
``_remove_orphan_nodes`` floods from, and the halted-capture carve-out that
keeps every op that ran.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
