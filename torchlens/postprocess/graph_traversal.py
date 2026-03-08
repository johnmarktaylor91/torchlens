"""Steps 1-4: Output nodes, ancestry tracing, orphan removal, distance marking.

Step 1 (_add_output_layers): Creates dedicated output LayerPassLog nodes, copying
    metadata from the original output tensors but stripping params and module info.
Step 2 (_find_output_ancestors): DFS backward from outputs marking is_output_ancestor.
Step 3 (_remove_orphan_nodes): Bidirectional flood from inputs AND outputs to find
    connected nodes; any node unreachable from both is removed as an orphan.
Step 4 (_mark_input_output_distances): Optional forward/backward BFS recording
    min/max hop counts from input and output nodes.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, List

import torch

from .._state import pause_logging
from ..utils.display import identity
from ..utils.rng import log_current_rng_states
from ..utils.tensor_utils import safe_copy, safe_to, tensor_nanequal
from ..utils.introspection import _get_func_call_stack
from ..data_classes.layer_pass_log import LayerPassLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _add_output_layers(
    self: "ModelLog", output_tensors: List[torch.Tensor], output_addresses: List[str]
) -> None:
    """Step 1: Add dedicated output nodes to the graph.

    For each tensor in the model's output, creates a new LayerPassLog that acts
    as a terminal "output" node. The new node copies tensor metadata from the
    original output tensor but resets function, parameter, and module information
    to reflect that this is a synthetic bookkeeping node (func_applied=identity,
    no params, no containing module). The original output tensor becomes the
    parent of the new output node.

    Also detects child_tensor_variations: if the actual output tensor differs
    from the parent's saved activation (e.g., due to in-place ops or postfunc),
    the difference is recorded for validation.
    """
    new_output_layers = []
    for i, output_layer_label in enumerate(self.output_layers):
        output_node = self[output_layer_label]
        new_output_node = output_node.copy()
        new_output_node.layer_type = "output"
        new_output_node.is_output_layer = True
        new_output_node.is_input_layer = False
        new_output_node.is_buffer_layer = False
        if i == len(self.output_layers) - 1:
            new_output_node.is_last_output_layer = True
        self._layer_counter += 1
        new_output_node.tensor_label_raw = f"output_{i + 1}_raw"
        new_output_node.layer_label_raw = new_output_node.tensor_label_raw
        new_output_node.realtime_tensor_num = self._layer_counter
        output_address = "output"
        if output_addresses[i] != "":
            output_address += f".{output_addresses[i]}"
        new_output_node.input_output_address = output_address

        # Fix function information:

        new_output_node.func_applied = identity
        new_output_node.func_applied_name = "none"
        new_output_node.func_call_stack = (
            _get_func_call_stack(self.num_context_lines) if self.save_source_context else []
        )
        new_output_node.func_time_elapsed = 0
        new_output_node.func_rng_states = (
            log_current_rng_states(torch_only=True) if self.save_rng_states else {}
        )
        new_output_node.func_argnames = tuple([])
        new_output_node.num_func_args_total = 0
        new_output_node.num_position_args = 0
        new_output_node.num_keyword_args = 0
        new_output_node.func_position_args_non_tensor = []
        new_output_node.func_keyword_args_non_tensor = {}
        new_output_node.func_all_args_non_tensor = []
        new_output_node.gradfunc = None
        new_output_node.creation_args = [output_tensors[i]]
        new_output_node.creation_kwargs = {}

        # Strip any params:

        new_output_node.parent_params = []
        new_output_node.parent_param_barcodes = []
        new_output_node.parent_param_passes = {}
        new_output_node.parent_param_logs = []
        new_output_node.parent_param_shapes = []
        new_output_node.num_params_total = int(0)
        new_output_node.num_params_trainable = 0
        new_output_node.num_params_frozen = 0
        new_output_node.parent_params_fsize = 0

        # Strip module info:

        new_output_node.containing_module_origin = None
        new_output_node.containing_modules_origin_nested = []
        new_output_node.modules_entered = []
        new_output_node.module_passes_entered = []
        new_output_node.modules_exited = [
            mod_pass[0] for mod_pass in output_node.containing_modules_origin_nested
        ]
        new_output_node.module_passes_exited = output_node.containing_modules_origin_nested
        new_output_node.is_submodule_output = False
        new_output_node.is_bottom_level_submodule_output = False
        new_output_node.module_entry_exit_threads_inputs = {}
        new_output_node.module_entry_exit_thread_output = []

        # Fix ancestry information:

        new_output_node.initialized_inside_model = False
        new_output_node.is_output_ancestor = True
        new_output_node.output_descendents = {new_output_node.tensor_label_raw}
        new_output_node.child_layers = []
        new_output_node.has_children = False
        new_output_node.parent_layers = [output_node.tensor_label_raw]
        new_output_node.parent_layer_arg_locs = {
            "args": {0: output_node.tensor_label_raw},
            "kwargs": {},
        }

        # Clear func_config on synthetic output nodes:
        new_output_node.func_config = {}

        # Fix layer equivalence information:
        new_output_node.same_layer_operations = []
        equiv_type = (
            f"output_{'_'.join(tuple(str(s) for s in new_output_node.tensor_shape))}_"
            f"{str(new_output_node.tensor_dtype)}"
        )
        new_output_node.operation_equivalence_type = equiv_type
        self.equivalent_operations[equiv_type].add(new_output_node.tensor_label_raw)

        # Track child tensor variations for output nodes.
        new_output_node.has_child_tensor_variations = False
        new_output_node.children_tensor_versions = {}
        if output_node.has_saved_activations:
            actual_output = safe_copy(output_tensors[i])
            if output_node.output_device not in [str(actual_output.device), "same"]:
                actual_output = safe_to(actual_output, output_node.output_device)
            if self.activation_postfunc is not None:
                with pause_logging():
                    actual_output = self.activation_postfunc(actual_output)
            if not tensor_nanequal(actual_output, output_node.tensor_contents):
                output_node.children_tensor_versions[new_output_node.tensor_label_raw] = (
                    actual_output
                )
                output_node.has_child_tensor_variations = True
                new_output_node.tensor_contents = actual_output

        # Change original output node:

        output_node.child_layers.append(new_output_node.tensor_label_raw)

        self._raw_layer_dict[new_output_node.tensor_label_raw] = new_output_node
        self._raw_layer_labels_list.append(new_output_node.tensor_label_raw)

        new_output_layers.append(new_output_node.tensor_label_raw)

    self.output_layers = new_output_layers


def _find_output_ancestors(self) -> None:
    """Step 2: Mark every node that is an ancestor of an output node.

    Uses a LIFO stack (DFS) starting from output nodes. For each node popped,
    checks its children — if any child is_output_ancestor, this node is too,
    and it inherits the child's output_descendents. Then pushes all unseen parents
    onto the stack.

    Note: A node may be pushed onto the stack multiple times if it's shared by
    sibling paths. The second pop is redundant (nodes_seen prevents re-pushing
    parents) but harmless — it may beneficially propagate output_descendents
    from newly-marked children on the second visit.

    The boolean is_output_ancestor is always correct after this function; the
    output_descendents set may be incomplete for multi-output graphs, but Step 4's
    flood corrects it if distance marking is enabled.
    """
    node_stack = self.output_layers[:]
    nodes_seen = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        nodes_seen.add(node_label)
        node = self[node_label]
        for child_node_label in node.child_layers:
            if self[child_node_label].is_output_ancestor:
                node.is_output_ancestor = True
                node.output_descendents.update(self[child_node_label].output_descendents)
        for parent_node_label in node.parent_layers:
            if parent_node_label not in nodes_seen:
                node_stack.append(parent_node_label)


def _remove_orphan_nodes(self) -> None:
    """Step 3: Remove orphan nodes unreachable from both inputs and outputs.

    Floods BIDIRECTIONALLY from input and output nodes simultaneously. A node is
    reachable if it can be reached by following child_layers OR parent_layers from
    any starting node. This bidirectional approach is necessary because:
    - Forward-only (from inputs) would miss nodes reachable only backward from outputs
      (e.g., internally-initialized tensors that only feed into outputs).
    - Backward-only (from outputs) would miss input-side dead ends.

    Any non-output node with no children is logged as an internally-terminated tensor
    (it produced a value that was never used by downstream computation reaching an output).
    """
    orig_nodes = set(self._raw_layer_labels_list)
    nodes_seen = set()
    # Seed with both input and output nodes for bidirectional reachability.
    node_stack = self.input_layers + self.output_layers
    while len(node_stack) > 0:
        tensor_label = node_stack.pop()
        nodes_seen.add(tensor_label)
        layer_entry = self._raw_layer_dict[tensor_label]
        if (len(layer_entry.child_layers) == 0) and (not layer_entry.is_output_layer):
            _log_internally_terminated_tensor(self, tensor_label)
        # Follow BOTH directions to ensure full bidirectional reachability.
        for next_label in layer_entry.child_layers + layer_entry.parent_layers:
            if next_label not in nodes_seen:
                node_stack.append(next_label)

    orphan_nodes = orig_nodes - nodes_seen
    self.orphan_layers = list(orphan_nodes)

    # Batch-remove orphaned nodes and rebuild the ordered layer dict/list.
    orphan_entries = [self._raw_layer_dict[label] for label in orphan_nodes]
    self._batch_remove_log_entries(orphan_entries, remove_references=True)

    new_layer_dict = OrderedDict()
    new_layer_list = []
    for tensor_label in self._raw_layer_labels_list:
        if tensor_label not in orphan_nodes:
            new_layer_dict[tensor_label] = self._raw_layer_dict[tensor_label]
            new_layer_list.append(tensor_label)
    self._raw_layer_labels_list = new_layer_list
    self._raw_layer_dict = new_layer_dict


def _mark_input_output_distances(self) -> None:
    """Step 4: Compute min/max hop distances from inputs and outputs.

    Runs two unidirectional floods: forward from inputs (following child_layers)
    and backward from outputs (following parent_layers). Each flood records
    min_distance_from_{input,output} and max_distance_from_{input,output} on
    every reachable node.

    This step is CONDITIONAL on ``self.mark_input_output_distances`` — it is
    skipped when the user doesn't need distance metadata, saving time on
    large graphs.
    """
    _flood_graph_from_input_or_output_nodes(self, "input")
    _flood_graph_from_input_or_output_nodes(self, "output")


def _flood_graph_from_input_or_output_nodes(self, mode: str) -> None:
    """Flood the graph from input or output nodes, tracking min/max distance.

    Traverses unidirectionally from starting nodes (input or output), recording
    each node's min and max hop count from the start. Also marks each node's
    ancestry (input_ancestors or output_descendents).

    Unlike the bidirectional flood in Step 3, this flood is UNIDIRECTIONAL:
    from inputs it follows child_layers (forward), from outputs it follows
    parent_layers (backward). This ensures hop counts reflect actual data-flow
    distance, not arbitrary graph traversal paths.

    A node may be revisited if a new path provides a shorter min or longer max
    distance, or adds a new ancestor/descendent — see
    ``_check_whether_to_add_node_to_flood_stack`` for the pruning logic.

    Args:
        mode: 'input' to flood forward from inputs, 'output' to flood backward from outputs.
    """
    if mode == "input":
        starting_nodes = self.input_layers[:]
        min_field = "min_distance_from_input"
        max_field = "max_distance_from_input"
        direction = "forwards"
        marker_field = "has_input_ancestor"
        layer_logging_field = "input_ancestors"
        forward_field = "child_layers"
    elif mode == "output":
        starting_nodes = self.output_layers[:]
        min_field = "min_distance_from_output"
        max_field = "max_distance_from_output"
        direction = "backwards"
        marker_field = "is_output_ancestor"
        layer_logging_field = "output_descendents"
        forward_field = "parent_layers"
    else:
        raise ValueError("Mode but be either 'input' or 'output'")

    nodes_seen = set()

    # Tuples in format node_label, nodes_since_start, traversal_direction
    node_stack = [
        (starting_node_label, starting_node_label, 0, direction)
        for starting_node_label in starting_nodes
    ]
    while len(node_stack) > 0:
        (
            current_node_label,
            orig_node,
            nodes_since_start,
            traversal_direction,
        ) = node_stack.pop()
        nodes_seen.add(current_node_label)
        current_node = self[current_node_label]
        _update_node_distance_vals(current_node, min_field, max_field, nodes_since_start)

        setattr(current_node, marker_field, True)
        getattr(current_node, layer_logging_field).add(orig_node)

        for next_node_label in getattr(current_node, forward_field):
            if _check_whether_to_add_node_to_flood_stack(
                self,
                next_node_label,
                orig_node,
                nodes_since_start,
                min_field,
                max_field,
                layer_logging_field,
                nodes_seen,
            ):
                node_stack.append(
                    (
                        next_node_label,
                        orig_node,
                        nodes_since_start + 1,
                        traversal_direction,
                    )
                )


def _update_node_distance_vals(
    current_node: LayerPassLog,
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


def _check_whether_to_add_node_to_flood_stack(
    self,
    candidate_node_label: str,
    orig_node_label: str,
    nodes_since_start: int,
    min_field: str,
    max_field: str,
    layer_logging_field: str,
    nodes_seen: set,
) -> bool:
    """Decide whether to push a candidate node onto the flood stack.

    Returns True (re-visit needed) if any of:
    - Node has never been visited.
    - The current path provides a new minimum distance.
    - The current path provides a new maximum distance.
    - The originating input/output node is not yet recorded in the candidate's
      ancestor/descendent set (adds new lineage information).

    This pruning prevents redundant BFS expansion while ensuring all
    distance extremes and lineage relationships are captured.
    """
    candidate_node = self[candidate_node_label]

    if candidate_node_label not in nodes_seen:
        return True

    if nodes_since_start + 1 < getattr(candidate_node, min_field):
        return True

    if nodes_since_start + 1 > getattr(candidate_node, max_field):
        return True

    if orig_node_label not in getattr(candidate_node, layer_logging_field):
        return True

    return False


def _log_internally_terminated_tensor(self, tensor_label: str) -> None:
    """Mark a tensor as terminated inside the model (no children reaching an output node)."""
    layer_entry = self[tensor_label]
    layer_entry.terminated_inside_model = True
    if tensor_label not in self.internally_terminated_layers:
        self.internally_terminated_layers.append(tensor_label)
        if layer_entry.is_atomic_bool_layer and (
            tensor_label not in self.internally_terminated_bool_layers
        ):
            self.internally_terminated_bool_layers.append(tensor_label)
            layer_entry.is_terminal_bool_layer = True
