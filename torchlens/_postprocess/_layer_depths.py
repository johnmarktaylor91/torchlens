"""Step 4 of graph traversal: optional layer-depth marking.

``_mark_layer_depths`` runs a forward BFS from the input nodes and a backward BFS
from the output nodes, recording on every op the min/max hop counts from each side.
It is the last of the four graph-traversal steps whose first three live in
``graph_traversal``; it depends on nothing but the trace's raw graph workspace.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _mark_layer_depths(self: Trace) -> None:
    """Step 4: Compute min/max hop distances from inputs and outputs.

    Runs two unidirectional floods: forward from inputs (following children)
    and backward from outputs (following parents). Each flood records
    min_distance_from_{input,output} and max_distance_from_{input,output} on
    every reachable node.

    This step is CONDITIONAL on ``self.mark_layer_depths`` — it is
    skipped when the user doesn't need distance metadata.
    """
    _flood_graph_from_input_or_output_nodes(self, "input")
    _flood_graph_from_input_or_output_nodes(self, "output")


def _flood_graph_from_input_or_output_nodes(self: Trace, mode: str) -> None:
    """Flood the graph from input or output nodes, tracking min/max distance.

    Traverses unidirectionally from starting nodes (input or output), recording
    each node's min and max hop count from the start. Also marks each node's
    ancestry (input_ancestors or output_descendants).

    Unlike the bidirectional flood in Step 3, this flood is UNIDIRECTIONAL:
    from inputs it follows children (forward), from outputs it follows
    parents (backward). This ensures hop counts reflect actual data-flow
    distance, not arbitrary graph traversal paths.

    Nodes are processed once in topological order: forward from inputs, or
    backward from outputs. Each visited node propagates finalized distance and
    lineage state to its data-flow successors.

    Args:
        mode: 'input' to flood forward from inputs, 'output' to flood backward from outputs.
    """
    traversal_order: Iterable[str]
    if mode == "input":
        starting_nodes = self.input_layers[:]
        min_field = "min_distance_from_input"
        max_field = "max_distance_from_input"
        marker_field = "has_input_ancestor"
        layer_logging_field = "input_ancestors"
        forward_field = "children"
        traversal_order = self._raw_graph_ws.raw_layer_labels_list
    elif mode == "output":
        starting_nodes = self.output_layers[:]
        min_field = "min_distance_to_output"
        max_field = "max_distance_to_output"
        marker_field = "has_output_descendant"
        layer_logging_field = "output_descendants"
        forward_field = "parents"
        traversal_order = reversed(self._raw_graph_ws.raw_layer_labels_list)
    else:
        raise ValueError("Mode must be either 'input' or 'output'")

    for starting_node_label in starting_nodes:
        starting_node = self[starting_node_label]
        _update_node_distance_vals(starting_node, min_field, max_field, 0)
        setattr(starting_node, marker_field, True)
        getattr(starting_node, layer_logging_field).add(starting_node_label)

    for current_node_label in traversal_order:
        current_node = self[current_node_label]
        current_min = getattr(current_node, min_field)
        current_max = getattr(current_node, max_field)
        if current_min is None or current_max is None:
            continue

        current_lineage = getattr(current_node, layer_logging_field)
        for next_node_label in getattr(current_node, forward_field):
            next_node = self[next_node_label]
            _update_node_distance_vals(next_node, min_field, max_field, current_min + 1)
            _update_node_distance_vals(next_node, min_field, max_field, current_max + 1)
            setattr(next_node, marker_field, True)
            getattr(next_node, layer_logging_field).update(current_lineage)


def _update_node_distance_vals(
    current_node: Op,
    min_field: str,
    max_field: str,
    nodes_since_start: int,
) -> None:
    """Update a node's min/max distance fields if the current hop count is a new extreme."""
    if getattr(current_node, min_field) is None:
        setattr(current_node, min_field, nodes_since_start)
    else:
        setattr(
            current_node,
            min_field,
            min(nodes_since_start, getattr(current_node, min_field)),
        )

    if getattr(current_node, max_field) is None:
        setattr(current_node, max_field, nodes_since_start)
    else:
        setattr(
            current_node,
            max_field,
            max(nodes_since_start, getattr(current_node, max_field)),
        )
