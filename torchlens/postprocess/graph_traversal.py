"""Steps 1-4: Output nodes, ancestry tracing, orphan removal, distance marking.

Step 1 (_add_output_layers): Creates dedicated output OpLog nodes, copying
    metadata from the original output tensors but stripping params and module info.
Step 2 (_find_output_ancestors): DFS backward from outputs marking has_output_descendant.
Step 3 (_remove_orphan_nodes): Bidirectional flood from inputs AND outputs to find
    connected nodes; any node unreachable from both is removed as an orphan.
Step 4 (_mark_layer_depths): Optional forward/backward BFS recording
    min/max hop counts from input and output nodes.
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, cast

import torch

from ..utils.display import identity
from ..utils.rng import log_current_rng_states
from ..utils.tensor_utils import safe_copy, safe_to, tensor_nanequal
from ..utils.introspection import _get_code_context
from ..data_classes.op_log import OpLog

if TYPE_CHECKING:
    from ..data_classes.model_log import Trace


def _add_output_layers(
    self: "Trace", output_tensors: list[torch.Tensor], output_addresses: list[str]
) -> None:
    """Step 1: Add dedicated output nodes to the graph.

    For each tensor in the model's output, creates a new OpLog that acts
    as a terminal "output" node. The new node copies tensor metadata from the
    original output tensor but resets function, parameter, and module information
    to reflect that this is a synthetic bookkeeping node (func=identity,
    no params, no containing module). The original output tensor becomes the
    parent of the new output node.

    Also detects child_tensor_variations: if the actual output tensor differs
    from the parent's saved out (e.g., due to in-place ops or postfunc),
    the difference is recorded for validation.
    """
    new_output_layers = []
    for i, output_layer_label in enumerate(self.output_layers):
        output_node = self[output_layer_label]
        new_output_node = cast(OpLog, output_node.copy())
        new_output_node.layer_type = "output"
        new_output_node.is_output = True
        new_output_node.is_input = False
        new_output_node.is_buffer = False
        if i == len(self.output_layers) - 1:
            new_output_node.is_final_output = True
        self._layer_counter += 1
        new_output_node._label_raw = f"output_{i + 1}_raw"
        new_output_node._layer_label_raw = new_output_node._label_raw
        new_output_node.capture_index = self._layer_counter
        output_address = "output"
        if output_addresses[i] != "":
            output_address += f".{output_addresses[i]}"
        new_output_node.io_role = output_address
        container_path_meta = getattr(self, "_output_container_specs_by_raw_label", {}).get(
            output_node._label_raw
        )
        if container_path_meta is not None:
            new_output_node.container_path = container_path_meta[0]
            new_output_node.container_spec = container_path_meta[1]

        # Fix function information:

        new_output_node.func = identity
        new_output_node.func_name = "none"
        new_output_node.code_context = _get_code_context(
            self.num_context_lines,
            source_loading_enabled=self.save_code_context,
            disable_col_offset=False,
        )
        new_output_node.func_duration = 0
        new_output_node.func_rng_states = (
            log_current_rng_states(torch_only=True) if self.save_rng_states else {}
        )
        new_output_node.arg_names = tuple([])
        new_output_node.num_args_total = 0
        new_output_node.num_pos_args = 0
        new_output_node.num_kwargs = 0
        new_output_node.non_tensor_pos_args = []
        new_output_node.non_tensor_kwargs = {}
        new_output_node.func_non_tensor_args = []
        new_output_node.grad_fn_name = None
        new_output_node.autograd_saved_memory = None
        new_output_node.num_autograd_saved_tensors = None
        new_output_node.bytes_delta_at_call = 0
        new_output_node.bytes_peak_at_call = 0
        new_output_node._internal_set("saved_args", [output_tensors[i]])
        new_output_node._internal_set("saved_kwargs", {})

        # Strip any params:

        new_output_node.parent_params = []
        new_output_node._param_barcodes = []
        new_output_node.parent_param_ops = {}
        new_output_node._param_logs = []
        new_output_node.param_shapes = []
        new_output_node.num_params = int(0)
        new_output_node.num_params_trainable = 0
        new_output_node.num_params_frozen = 0
        new_output_node.param_memory = 0

        # Strip module info:

        new_output_node.module = None
        new_output_node.modules = []
        new_output_node.modules_entered = []
        new_output_node.module_ops_entered = []
        new_output_node.output_of_modules = [mod_pass[0] for mod_pass in output_node.modules]
        new_output_node.output_of_module_calls = output_node.modules
        new_output_node.is_submodule_output = False
        new_output_node.is_atomic_module_op = False

        # Fix ancestry information:

        new_output_node.is_internal_source = False
        new_output_node.has_output_descendant = True
        new_output_node.output_descendants = {new_output_node._label_raw}
        new_output_node.children = []
        new_output_node.has_children = False
        new_output_node.parents = [output_node._label_raw]
        new_output_node.parent_arg_positions = {
            "args": {0: output_node._label_raw},
            "kwargs": {},
        }
        new_output_node.edge_uses = []

        # Clear func_config on synthetic output nodes:
        new_output_node.func_config = {}

        # Fix layer equivalence information:
        new_output_node.recurrent_ops = []
        equiv_type = (
            f"output_{'_'.join(tuple(str(s) for s in new_output_node.shape))}_"
            f"{str(new_output_node.dtype)}"
        )
        new_output_node.equivalence_class = equiv_type
        self.equivalent_ops[equiv_type].add(new_output_node._label_raw)

        # Track child tensor variations for output nodes.
        new_output_node.has_output_variations = False
        new_output_node.output_versions_per_child = {}
        if output_node.has_saved_outs:
            actual_output = safe_copy(output_tensors[i])
            if output_node.output_device not in [str(actual_output.device), "same"]:
                actual_output = safe_to(actual_output, output_node.output_device)
            actual_output_raw = actual_output
            if self.out_postfunc is not None:
                actual_output = output_node._apply_postfunc(
                    actual_output,
                    self.out_postfunc,
                    postfunc_kind="out",
                    streaming_active=False,
                )
            comparison_output = (
                output_node.out if output_node.out is not None else output_node.transformed_out
            )
            if comparison_output is not None and not tensor_nanequal(
                actual_output, comparison_output
            ):
                output_node.output_versions_per_child[new_output_node._label_raw] = actual_output
                output_node.has_output_variations = True
                if output_node.out is None:
                    new_output_node._internal_set("transformed_out", actual_output)
                else:
                    new_output_node._internal_set("out", actual_output_raw)

        # Change original output node:

        output_node.children.append(new_output_node._label_raw)

        self._raw_layer_dict[new_output_node._label_raw] = new_output_node
        self._raw_layer_labels_list.append(new_output_node._label_raw)

        new_output_layers.append(new_output_node._label_raw)

    self.output_layers = new_output_layers


def _find_output_ancestors(self: "Trace") -> None:
    """Step 2: Mark every node that is an ancestor of an output node.

    Uses a LIFO stack (DFS) starting from output nodes. For each node popped,
    checks its children — if any child has_output_descendant, this node is too,
    and it inherits the child's output_descendants. Then pushes all unseen parents
    onto the stack.

    Note: A node may be pushed onto the stack multiple times if it's shared by
    sibling paths. The second pop is redundant (nodes_seen prevents re-pushing
    parents) but harmless — it may beneficially propagate output_descendants
    from newly-marked children on the second visit.

    The boolean has_output_descendant is always correct after this function; the
    output_descendants set may be incomplete for multi-output graphs, but Step 4's
    flood corrects it if distance marking is enabled.
    """
    node_stack = self.output_layers[:]
    nodes_seen = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        nodes_seen.add(node_label)
        node = self[node_label]
        for child_node_label in node.children:
            if self[child_node_label].has_output_descendant:
                node.has_output_descendant = True
                node.output_descendants.update(self[child_node_label].output_descendants)
        for parent_node_label in node.parents:
            if parent_node_label not in nodes_seen:
                node_stack.append(parent_node_label)


def _remove_orphan_nodes(self: "Trace") -> None:
    """Step 3: Remove orphan nodes unreachable from both inputs and outputs.

    Floods BIDIRECTIONALLY from input and output nodes simultaneously. A node is
    reachable if it can be reached by following children OR parents from
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
        if (len(layer_entry.children) == 0) and (not layer_entry.is_output):
            _log_internally_terminated_tensor(self, tensor_label)
        # Follow BOTH directions to ensure full bidirectional reachability.
        for next_label in layer_entry.children + layer_entry.parents:
            if next_label not in nodes_seen:
                node_stack.append(next_label)

    nodes_seen = _expand_seen_nodes_to_complete_func_call_groups(self, nodes_seen)
    orphan_nodes = orig_nodes - nodes_seen
    self.orphan_ops = [label for label in self._raw_layer_labels_list if label in orphan_nodes]
    self.orphan_logs = tuple(self._raw_layer_dict[label] for label in self.orphan_ops)
    if getattr(self, "keep_orphans", False):
        for orphan_label in orphan_nodes:
            self._raw_layer_dict[orphan_label].is_orphan = True
        return

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


def _expand_seen_nodes_to_complete_func_call_groups(
    self: "Trace", nodes_seen: set[str]
) -> set[str]:
    """Add raw-label siblings for any surviving ``func_call_id`` group.

    Parameters
    ----------
    nodes_seen:
        Raw labels reachable from the input/output flood.

    Returns
    -------
    set[str]
        Reachable raw labels expanded so multi-output wrapper calls are kept
        atomically.
    """

    func_groups: dict[int, set[str]] = {}
    for raw_label in self._raw_layer_labels_list:
        func_call_id = getattr(self._raw_layer_dict[raw_label], "func_call_id", None)
        if func_call_id is not None:
            func_groups.setdefault(func_call_id, set()).add(raw_label)

    expanded_seen = set(nodes_seen)
    changed = True
    while changed:
        changed = False
        for raw_labels in func_groups.values():
            if expanded_seen.intersection(raw_labels) and not raw_labels.issubset(expanded_seen):
                expanded_seen.update(raw_labels)
                changed = True
    return expanded_seen


def _mark_layer_depths(self: "Trace") -> None:
    """Step 4: Compute min/max hop distances from inputs and outputs.

    Runs two unidirectional floods: forward from inputs (following children)
    and backward from outputs (following parents). Each flood records
    min_distance_from_{input,output} and max_distance_from_{input,output} on
    every reachable node.

    This step is CONDITIONAL on ``self.mark_layer_depths`` — it is
    skipped when the user doesn't need distance metadata, saving time on
    large graphs.
    """
    _flood_graph_from_input_or_output_nodes(self, "input")
    _flood_graph_from_input_or_output_nodes(self, "output")


def _flood_graph_from_input_or_output_nodes(self: "Trace", mode: str) -> None:
    """Flood the graph from input or output nodes, tracking min/max distance.

    Traverses unidirectionally from starting nodes (input or output), recording
    each node's min and max hop count from the start. Also marks each node's
    ancestry (input_ancestors or output_descendants).

    Unlike the bidirectional flood in Step 3, this flood is UNIDIRECTIONAL:
    from inputs it follows children (forward), from outputs it follows
    parents (backward). This ensures hop counts reflect actual data-flow
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
        forward_field = "children"
    elif mode == "output":
        starting_nodes = self.output_layers[:]
        min_field = "min_distance_from_output"
        max_field = "max_distance_from_output"
        direction = "backwards"
        marker_field = "has_output_descendant"
        layer_logging_field = "output_descendants"
        forward_field = "parents"
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
    current_node: OpLog,
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
    self: "Trace",
    candidate_node_label: str,
    orig_node_label: str,
    nodes_since_start: int,
    min_field: str,
    max_field: str,
    layer_logging_field: str,
    nodes_seen: set[str],
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


def _log_internally_terminated_tensor(self: "Trace", tensor_label: str) -> None:
    """Mark a tensor as terminated inside the model (no children reaching an output node)."""
    layer_entry = self[tensor_label]
    layer_entry.is_internal_sink = True
    if tensor_label not in self.internal_sink_ops:
        self.internal_sink_ops.append(tensor_label)
        if layer_entry.is_scalar_bool and (tensor_label not in self.internally_terminated_bool_ops):
            self.internally_terminated_bool_ops.append(tensor_label)
            layer_entry.is_terminal_bool = True
