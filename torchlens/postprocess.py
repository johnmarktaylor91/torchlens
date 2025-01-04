import itertools as it
import time
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Set, TYPE_CHECKING, Tuple

import torch

from .constants import MODEL_HISTORY_FIELD_ORDER, TENSOR_LOG_ENTRY_FIELD_ORDER
from .helper_funcs import (
    get_vars_of_type_from_obj,
    human_readable_size,
    identity,
    log_current_rng_states,
    safe_copy,
    _get_call_stack_dicts,
)
from .tensor_log import RolledTensorLogEntry, TensorLogEntry

if TYPE_CHECKING:
    from .model_history import ModelHistory


def postprocess(
        self: "ModelHistory", output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
):
    """
    After the forward pass, cleans up the log into its final form.
    """
    if self.logging_mode == "fast":
        postprocess_fast(self)
        return

    # Step 1: Add dedicated output nodes

    _add_output_layers(self, output_tensors, output_tensor_addresses)

    # Step 2: Trace which nodes are ancestors of output nodes

    _find_output_ancestors(self)

    # Step 3: Remove orphan nodes, find nodes that don't terminate in output node

    _remove_orphan_nodes(self)

    # Step 4: Find mix/max distance from input and output nodes

    if self.mark_input_output_distances:
        _mark_input_output_distances(self)

    # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.

    _mark_conditional_branches(self)

    # Step 6: Annotate the containing modules for all internally-generated tensors (they don't know where
    # they are when they're made; have to trace breadcrumbs from tensors that came from input).

    _fix_modules_for_internal_tensors(self)

    # Step 7: Fix the buffer passes and parent infomration.

    _fix_buffer_layers(self)

    # Step 8: Identify all loops, mark repeated layers.

    _assign_corresponding_tensors_to_same_layer(self)

    # Step 9: Go down tensor list, get the mapping from raw tensor names to final tensor names.

    _map_raw_tensor_labels_to_final_tensor_labels(self)

    # Step 10: Go through and log information pertaining to all layers:
    _log_final_info_for_all_layers(self)

    # Step 11: Rename the raw tensor entries in the fields of ModelHistory:
    _rename_model_history_layer_names(self)
    _trim_and_reorder_model_history_fields(self)

    # Step 12: And one more pass to delete unused layers from the record and do final tidying up:
    _remove_unwanted_entries_and_log_remaining(self)

    # Step 13: Undecorate all saved tensors and remove saved grad_fns.
    _undecorate_all_saved_tensors(self)

    # Step 14: Clear the cache after any tensor deletions for garbage collection purposes:
    torch.cuda.empty_cache()

    # Step 15: Log time elapsed.
    _log_time_elapsed(self)

    # Step 16: log the pass as finished, changing the ModelHistory behavior to its user-facing version.

    _set_pass_finished(self)


def postprocess_fast(self: "ModelHistory"):
    _trim_and_reorder_model_history_fields(self)
    _remove_unwanted_entries_and_log_remaining(self)
    _undecorate_all_saved_tensors(self)
    torch.cuda.empty_cache()
    _log_time_elapsed(self)
    _set_pass_finished(self)


def _add_output_layers(
        self: "ModelHistory", output_tensors: List[torch.Tensor], output_addresses: List[str]
):
    """
    Adds dedicated output nodes to the graph.
    """
    new_output_layers = []
    for i, output_layer_label in enumerate(self.output_layers):
        output_node = self[output_layer_label]
        new_output_node = output_node.copy()
        new_output_node.layer_type = "output"
        new_output_node.is_output_layer = True
        if i == len(self.output_layers) - 1:
            new_output_node.is_last_output_layer = True
        self._tensor_counter += 1
        new_output_node.tensor_label_raw = f"output_{i + 1}_raw"
        new_output_node.layer_label_raw = new_output_node.tensor_label_raw
        new_output_node.realtime_tensor_num = self._tensor_counter
        output_address = "output"
        if output_addresses[i] != "":
            output_address += f".{output_addresses[i]}"
        new_output_node.input_output_address = output_address

        # Fix function information:

        new_output_node.func_applied = identity
        new_output_node.func_applied_name = "none"
        new_output_node.func_call_stack = _get_call_stack_dicts()
        new_output_node.func_time_elapsed = 0
        new_output_node.func_rng_states = log_current_rng_states()
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

        new_output_node.computed_with_params = False
        new_output_node.parent_params = []
        new_output_node.parent_param_barcodes = []
        new_output_node.parent_param_passes = {}
        new_output_node.num_param_tensors = 0
        new_output_node.parent_param_shapes = []
        new_output_node.num_params_total = int(0)
        new_output_node.parent_params_fsize = 0
        new_output_node.parent_params_fsize_nice = human_readable_size(0)

        # Strip module info:

        new_output_node.is_computed_inside_submodule = False
        new_output_node.containing_module_origin = None
        new_output_node.containing_modules_origin_nested = []
        new_output_node.modules_entered = []
        new_output_node.module_passes_entered = []
        new_output_node.is_submodule_input = False
        new_output_node.modules_exited = [
            mod_pass[0] for mod_pass in output_node.containing_modules_origin_nested
        ]
        new_output_node.module_passes_exited = (
            output_node.containing_modules_origin_nested
        )
        new_output_node.is_submodule_output = False
        new_output_node.is_bottom_level_submodule_output = False
        new_output_node.module_entry_exit_threads_inputs = {}
        new_output_node.module_entry_exit_thread_output = []

        # Fix ancestry information:

        new_output_node.is_output_ancestor = True
        new_output_node.output_descendents = {new_output_node.tensor_label_raw}
        new_output_node.child_layers = []
        new_output_node.parent_layers = [output_node.tensor_label_raw]
        new_output_node.sibling_layers = []
        new_output_node.has_sibling_tensors = False
        new_output_node.parent_layer_arg_locs = {
            "args": {0: output_node.tensor_label_raw},
            "kwargs": {},
        }

        # Fix layer equivalence information:
        new_output_node.same_layer_operations = []
        equiv_type = (
            f"output_{'_'.join(tuple(str(s) for s in new_output_node.tensor_shape))}_"
            f"{str(new_output_node.tensor_dtype)}"
        )
        new_output_node.operation_equivalence_type = equiv_type
        self.equivalent_operations[equiv_type].add(new_output_node.tensor_label_raw)

        # Fix the getitem stuff:

        new_output_node.was_getitem_applied = False
        new_output_node.children_tensor_versions = {}
        if output_node.was_getitem_applied:
            output_node.children_tensor_versions[
                new_output_node.tensor_label_raw
            ] = safe_copy(output_tensors[i])
            new_output_node.tensor_contents = safe_copy(output_tensors[i])

        # Change original output node:

        output_node.child_layers.append(new_output_node.tensor_label_raw)

        self._raw_tensor_dict[new_output_node.tensor_label_raw] = new_output_node
        self._raw_tensor_labels_list.append(new_output_node.tensor_label_raw)

        new_output_layers.append(new_output_node.tensor_label_raw)

    self.output_layers = new_output_layers


def _find_output_ancestors(self):
    node_stack = self.output_layers[:]
    nodes_seen = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        nodes_seen.add(node_label)
        node = self[node_label]
        for child_node_label in node.child_layers:
            if self[child_node_label].is_output_ancestor:
                node.is_output_ancestor = True
                node.output_descendents.update(
                    self[child_node_label].output_descendents
                )
        for parent_node_label in node.parent_layers:
            if parent_node_label not in nodes_seen:
                node_stack.append(parent_node_label)


def _remove_orphan_nodes(self):
    """
    Removes nodes that are connected to neither the input nor the output by flooding in both directions
    from the input and output nodes.
    """
    orig_nodes = set(self._raw_tensor_labels_list)
    nodes_seen = set()
    node_stack = self.input_layers + self.output_layers
    while len(node_stack) > 0:
        tensor_label = node_stack.pop()
        nodes_seen.add(tensor_label)
        tensor_entry = self._raw_tensor_dict[tensor_label]
        if (len(tensor_entry.child_layers) == 0) and (
                not tensor_entry.is_output_layer
        ):
            _log_internally_terminated_tensor(self, tensor_label)
        for next_label in tensor_entry.child_layers + tensor_entry.parent_layers:
            if next_label not in nodes_seen:
                node_stack.append(next_label)

    orphan_nodes = orig_nodes - nodes_seen
    self.orphan_layers = list(orphan_nodes)

    # Now remove all orphaned nodes.

    new_tensor_dict = OrderedDict()
    new_tensor_list = []
    for tensor_label in self._raw_tensor_labels_list:
        tensor_entry = self[tensor_label]
        if tensor_label not in orphan_nodes:
            new_tensor_dict[tensor_label] = tensor_entry
            new_tensor_list.append(tensor_label)
        else:
            self._remove_log_entry(tensor_entry, remove_references=True)
    self._raw_tensor_labels_list = new_tensor_list
    self._raw_tensor_dict = new_tensor_dict


def _mark_input_output_distances(self):
    """
    Traverses the graph forward and backward, marks the minimum and maximum distances of each
    node from the input and output, and removes any orphan nodes.
    """
    _flood_graph_from_input_or_output_nodes(self, "input")
    _flood_graph_from_input_or_output_nodes(self, "output")


def _flood_graph_from_input_or_output_nodes(self, mode: str):
    """Floods the graph from either the input or output nodes, tracking nodes that aren't seen,
    and the min and max distance from the starting nodes of each node. Traversal is unidirectional
    UNLESS going in the direction of a termin

    Args:
        mode: either 'input' or 'output'

    Returns:
        Set of nodes seen during the traversal
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
        _update_node_distance_vals(
            current_node, min_field, max_field, nodes_since_start
        )

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
        current_node: TensorLogEntry,
        min_field: str,
        max_field: str,
        nodes_since_start: int,
):
    if getattr(current_node, min_field) is None:
        setattr(current_node, min_field, nodes_since_start)
    else:
        setattr(
            current_node,
            min_field,
            min([nodes_since_start, getattr(current_node, min_field)]),
        )

    if getattr(current_node, max_field) is None:
        setattr(current_node, max_field, nodes_since_start)
    else:
        setattr(
            current_node,
            max_field,
            max([nodes_since_start, getattr(current_node, max_field)]),
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
):
    """
    Checker function to trim uninformative nodes when tracing input and output distances:
    trims nodes if they don't exceed the min or max, or don't add an informative new ancestor or descendent.
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


def _log_internally_terminated_tensor(self, tensor_label: str):
    tensor_entry = self[tensor_label]
    tensor_entry.terminated_inside_model = True
    if tensor_label not in self.internally_terminated_layers:
        self.internally_terminated_layers.append(tensor_label)
        if tensor_entry.is_atomic_bool_layer and (
                tensor_label not in self.internally_terminated_bool_layers
        ):
            self.internally_terminated_bool_layers.append(tensor_label)
            tensor_entry.is_terminal_bool_layer = True


def _mark_conditional_branches(self):
    """Starting from any terminal boolean nodes, backtracks until it finds the beginning of any
    conditional branches.
    """
    terminal_bool_nodes = self.internally_terminated_bool_layers[:]

    nodes_seen = set()
    node_stack = terminal_bool_nodes.copy()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        node = self[node_label]
        if node_label in nodes_seen:
            continue
        for next_tensor_label in node.parent_layers + node.child_layers:
            next_node = self[next_tensor_label]
            if (
                    next_node.is_output_ancestor
            ):  # we found the beginning of a conditional branch
                next_node.cond_branch_start_children.append(node_label)
                next_node.in_cond_branch = False
                nodes_seen.add(next_tensor_label)
                self.conditional_branch_edges.append(
                    (next_tensor_label, node_label)
                )
            else:
                if next_tensor_label in nodes_seen:
                    continue
                next_node.in_cond_branch = True
                node_stack.append(next_tensor_label)

        nodes_seen.add(node_label)


def _assign_corresponding_tensors_to_same_layer(self):
    """
    Post-processing function that yokes together operations corresponding to the same layer, based on
    the following rule:
    1) Operations invoking the same parameters are always assigned to the same layer.
    2) Any contiguous operations surrounding repeated parameters are assigned to the same layer
        (e.g., if a ReLU always follows every pass of an FC layer, then all instances of that ReLU
        operation are considered part of the same layer; continue for all such contiguous
        equivalent operations)
    3) Any groups of contiguous operations that "loop" back to back, irrespective of whether
        they include a parameter or not (e.g., in ABCABCABC, then all As count as the same layer, all
        Bs count as the same layer, and all Cs cound as the same layer, but if a D or an F were inserted
        between these triplets, they would no longer be grouped together, since the repeats
        are no longer contiguous)
    It works by starting from root nodes, and starting from the earliest one, going forward one node at a time,
    and checking if there are equivalent operations. If so, it builds forward one node at a time, until
    it no longer finds equivalent operations. If these subgraphs include a parameter node, these nodes
    are then grouped together no matter what. If they don't, they're only grouped together if contiguous.
    To allow for the possibility that a node might have more "equivalent" layers as a subset of some bigger
    subgraph, then while advancing forward, the function checks the number of equivalent layers it has been
    assigned is equal to the number of operations of that type. If so, it's definitely found everything;
    if not, it runs the procedure again to check if more equivalent operations can be found.
    """
    node_stack = self.input_layers + self.internally_initialized_layers
    node_stack = sorted(node_stack, key=lambda x: self[x].realtime_tensor_num)
    operation_equivalence_types_seen = set()
    while len(node_stack) > 0:
        # Grab the earliest node in the stack, add its children in sorted order to the stack in advance.
        node_label = node_stack.pop(0)
        node = self[node_label]
        node_operation_equivalence_type = node.operation_equivalence_type

        # If we've already checked the nodes of this operation equivalence type as starting nodes, continue:
        if node_operation_equivalence_type in operation_equivalence_types_seen:
            continue
        operation_equivalence_types_seen.add(node_operation_equivalence_type)
        for equiv_op in node.equivalent_operations:
            node_stack.extend(self[equiv_op].child_layers)
        node_stack = sorted(node_stack, key=lambda x: self[x].realtime_tensor_num)

        # If no equivalent operations for this node, skip it; it's the only operation for this "layer"
        if len(node.equivalent_operations) == 1:
            node.same_layer_operations = [node_label]
            continue

        # If we've already found the same-layer tensors for this node, and it equals the number of
        # equivalent operations, skip it; the work is already done:
        if len(node.equivalent_operations) == len(node.same_layer_operations):
            continue

        # Else, start from this node and any equivalent operations, and work forward, finding
        # more equivalent operations:
        _find_and_mark_same_layer_operations_starting_from_node(self, node)


def _find_and_mark_same_layer_operations_starting_from_node(
        self, node: TensorLogEntry
):
    """Starting from a given node in the graph, starts from all equivalent operations (e.g., cos, add 5, etc.),
    and crawls forward, finding and marking corresponding operations until there are none left.
    At the end of this, nodes that have the same position with respect to the original node
    are labeled as the same layer either if 1) the subgraph contains a parameter node,
    or 2) the nodes belong to adjacent subgraphs.

    Args:
        node: node to start from
    """
    # Bookkeeping regarding nodes, subgraphs, isomorphic nodes, adjacent subgraphs:
    # Label each subgraph by its starting node label.
    equivalent_operation_starting_labels = sorted(list(node.equivalent_operations))

    # Dictionary specifying isomorphic nodes: key is earliest such node, value is list of isomorphic nodes
    iso_node_groups = OrderedDict(
        {
            equivalent_operation_starting_labels[
                0
            ]: equivalent_operation_starting_labels
        }
    )

    # Reverse dictionary mapping each node to its isomorphism group
    node_to_iso_group_dict = OrderedDict(
        {
            label: equivalent_operation_starting_labels[0]
            for label in equivalent_operation_starting_labels
        }
    )

    # Dictionary of information about each subgraph
    subgraphs_dict = {}
    for starting_label in equivalent_operation_starting_labels:
        subgraphs_dict[starting_label] = {
            "starting_node": starting_label,
            "param_nodes": set(),
            "node_set": {starting_label},
        }
        if node.computed_with_params:
            subgraphs_dict[starting_label]["param_nodes"].add(starting_label)

    # Dictionary mapping each node to the subgraph it is in
    node_to_subgraph_dict = OrderedDict(
        {
            label: subgraphs_dict[label]
            for label in equivalent_operation_starting_labels
        }
    )

    # Dict mapping each subgraph to the set of subgraphs it's adjacent to; initialize each to be self-adjacent
    adjacent_subgraphs = {}

    # The stack will be a list of lists, where each latter list is a list of isomorphic nodes.
    # When adding to the stack, only isomorphic nodes will be added.

    node_stack = [equivalent_operation_starting_labels[:]]

    is_first_node = True  # if the first node, don't look at parents
    while node_stack:
        # Pop a set of isomorphic nodes off of the stack, then add and process the next nodes in the stack.
        isomorphic_nodes = sorted(node_stack.pop(0))
        if len(isomorphic_nodes) == 1:
            continue
        _fetch_and_process_next_isomorphic_nodes(
            self,
            isomorphic_nodes,
            iso_node_groups,
            node_to_iso_group_dict,
            subgraphs_dict,
            node_to_subgraph_dict,
            adjacent_subgraphs,
            is_first_node,
            node_stack,
        )
        is_first_node = False

    _assign_and_log_isomorphic_nodes_to_same_layers(
        self, iso_node_groups, node_to_subgraph_dict, adjacent_subgraphs
    )


def _fetch_and_process_next_isomorphic_nodes(
        self,
        current_iso_nodes: List[str],
        iso_node_groups: Dict[str, List[str]],
        node_to_iso_group_dict: Dict[str, str],
        subgraphs_dict: Dict,
        node_to_subgraph_dict: Dict,
        adjacent_subgraphs: Dict[str, set],
        is_first_node: bool,
        node_stack: List[List[str]],
):
    """Function that takes a set of isomorphic nodes, finds all sets of isomorphic successor nodes,
    then processes them and adds them to the stack.

    Args:
        current_iso_nodes: Current set of isomorphic nodes to get the next nodes from.
        iso_node_groups: Dict mapping each isomorphism node group to the list of nodes in it.
        node_to_iso_group_dict: Reverse dict mapping each node to its isomorphism group.
        subgraphs_dict: Dict of information about each subgraph
        node_to_subgraph_dict: Dict mapping each node to its subgraph
        adjacent_subgraphs: List of sets of adjacent subgraphs
        is_first_node: Whether it's the first node in the subgraph; if so, just do children, not parents to start.
        node_stack: List of lists of isomorphic nodes in the stack.
    """
    # First, get all children and parents of the current nodes, with constraint of not being added
    # to their own subgraph yet to avoid backtracking; if run into another subgraph, mark them
    # adjacent and skip.

    successor_nodes_dict = _log_collisions_and_get_candidate_next_nodes(
        self,
        current_iso_nodes,
        iso_node_groups,
        node_to_iso_group_dict,
        node_to_subgraph_dict,
        adjacent_subgraphs,
        is_first_node,
    )

    # Find sets of isomorphic nodes, process & add to the stack, discard singular nodes, repeat till none left.

    while True:
        # Grab a node and pop it:
        (
            candidate_node_label,
            candidate_node_neighbor_type,
            candidate_node_subgraph,
        ) = _get_next_candidate_node(successor_nodes_dict)
        if candidate_node_label is None:
            break

        new_equivalent_nodes = _get_nodes_isomorphic_to_candidate_node(
            self,
            candidate_node_label,
            candidate_node_neighbor_type,
            candidate_node_subgraph,
            successor_nodes_dict,
        )

        # Now log this new set of isomorphic nodes.

        _log_new_isomorphic_nodes(
            self,
            new_equivalent_nodes,
            iso_node_groups,
            node_to_iso_group_dict,
            subgraphs_dict,
            node_to_subgraph_dict,
            node_stack,
        )


def _log_collisions_and_get_candidate_next_nodes(
        self,
        current_iso_nodes: List[str],
        iso_node_groups: Dict[str, List[str]],
        node_to_iso_group_dict: Dict[str, str],
        node_to_subgraph_dict: Dict,
        adjacent_subgraphs: Dict[str, set],
        is_first_node: bool,
) -> Dict:
    """Helper function that checks all parent and children nodes for overlap with nodes already added
    to subgraphs (either the same subgraph or another one), logs any adjacency among subgraphs,
    and returns a dict with the candidate successor nodes from each subgraph.

    Returns:
        Dict with the candidate next nodes for each subgraph.
    """
    node_type_fields = {"children": "child_layers", "parents": "parent_layers"}
    if is_first_node:
        node_types_to_use = ["children"]
    else:
        node_types_to_use = ["children", "parents"]

    successor_nodes_dict = OrderedDict()
    for node_label in current_iso_nodes:
        node = self[node_label]
        node_subgraph = node_to_subgraph_dict[node_label]
        node_subgraph_label = node_subgraph["starting_node"]
        subgraph_successor_nodes = {"children": [], "parents": []}
        for node_type in node_types_to_use:
            node_type_field = node_type_fields[node_type]
            for neighbor_label in getattr(node, node_type_field):
                if (
                        neighbor_label in node_subgraph["node_set"]
                ):  # skip if backtracking own subgraph
                    continue
                elif (
                        neighbor_label in node_to_subgraph_dict
                ):  # if hit another subgraph, mark them adjacent.
                    _check_and_mark_subgraph_adjacency(
                        node_label,
                        neighbor_label,
                        iso_node_groups,
                        node_to_iso_group_dict,
                        node_to_subgraph_dict,
                        adjacent_subgraphs,
                    )
                else:  # we have a new, non-overlapping node as a possible candiate, add it:
                    subgraph_successor_nodes[node_type].append(neighbor_label)
        successor_nodes_dict[node_subgraph_label] = subgraph_successor_nodes

    return successor_nodes_dict


def _check_and_mark_subgraph_adjacency(
        node_label: str,
        neighbor_label: str,
        iso_node_groups: Dict[str, List[str]],
        node_to_iso_group_dict: Dict[str, str],
        node_to_subgraph_dict: Dict,
        adjacent_subgraphs: Dict[str, set],
):
    """Helper function that updates the adjacency status of two subgraphs"""
    node_subgraph = node_to_subgraph_dict[node_label]
    node_subgraph_label = node_subgraph["starting_node"]
    neighbor_subgraph = node_to_subgraph_dict[neighbor_label]
    neighbor_subgraph_label = neighbor_subgraph["starting_node"]

    # Subgraphs are adjacent if the node in the neighboring subgraph has an
    # isomorphic node in the current subgraph.

    neighbor_iso_group = node_to_iso_group_dict[neighbor_label]
    nodes_isomorphic_to_neighbor_node = iso_node_groups[neighbor_iso_group]
    if (
            len(
                node_subgraph["node_set"].intersection(
                    nodes_isomorphic_to_neighbor_node
                )
            )
            == 0
    ):
        return

    # Update adjacency
    if (node_subgraph_label in adjacent_subgraphs) and (
            neighbor_subgraph_label in adjacent_subgraphs
    ):
        return
    elif (node_subgraph_label in adjacent_subgraphs) and (
            neighbor_subgraph_label not in adjacent_subgraphs
    ):
        adjacent_subgraphs[node_subgraph_label].add(neighbor_subgraph_label)
        adjacent_subgraphs[neighbor_subgraph_label] = adjacent_subgraphs[
            node_subgraph_label
        ]
    elif (node_subgraph_label not in adjacent_subgraphs) and (
            neighbor_subgraph_label in adjacent_subgraphs
    ):
        adjacent_subgraphs[neighbor_subgraph_label].add(node_subgraph_label)
        adjacent_subgraphs[node_subgraph_label] = adjacent_subgraphs[
            neighbor_subgraph_label
        ]
    else:
        new_adj_set = {node_subgraph_label, neighbor_subgraph_label}
        adjacent_subgraphs[neighbor_subgraph_label] = new_adj_set
        adjacent_subgraphs[node_subgraph_label] = new_adj_set


def _get_next_candidate_node(
        successor_nodes_dict: Dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Helper function to grab the next candidate node to consider out of the possible successor nodes.

    Args:
        successor_nodes_dict: Dict of successor nodes from the set of subgraphs being considered

    Returns:

    """
    for subgraph_label, neighbor_type in it.product(
            successor_nodes_dict, ["children", "parents"]
    ):
        subgraph_neighbors = successor_nodes_dict[subgraph_label][neighbor_type]
        if len(subgraph_neighbors) > 0:
            candidate_node_label = subgraph_neighbors.pop(0)
            candidate_node_neighbor_type = neighbor_type
            candidate_node_subgraph = subgraph_label
            return (
                candidate_node_label,
                candidate_node_neighbor_type,
                candidate_node_subgraph,
            )
    return None, None, None


def _get_nodes_isomorphic_to_candidate_node(
        self,
        candidate_node_label: str,
        candidate_node_neighbor_type: str,
        candidate_node_subgraph: str,
        successor_nodes_dict: Dict,
) -> List[Tuple[str, str]]:
    """Finds nodes that are isomorphic with a candidate node.

    Args:
        candidate_node_label: Label of candidate node
        candidate_node_neighbor_type: Whether the candidate node is a child or parent node
        candidate_node_subgraph: Subgraph of the candidate node
        successor_nodes_dict: Dict keeping track of possible successor nodes

    Returns:
        List of nodes isomorphic with the candidate node
    """
    candidate_node = self[candidate_node_label]
    candidate_node_operation_equivalence_type = (
        candidate_node.operation_equivalence_type
    )
    new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
    for subgraph_label in successor_nodes_dict:
        if subgraph_label == candidate_node_subgraph:  # ignore same subgraph
            continue
        other_subgraph_nodes = successor_nodes_dict[subgraph_label][
            candidate_node_neighbor_type
        ]
        for c, comparison_node_label in enumerate(other_subgraph_nodes):
            comparison_node = self[comparison_node_label]
            if (
                    comparison_node.operation_equivalence_type
                    == candidate_node_operation_equivalence_type
            ):
                new_equivalent_nodes.append(
                    (other_subgraph_nodes.pop(c), subgraph_label)
                )
                break  # only add one node per subgraph at most
    new_equivalent_nodes = sorted(new_equivalent_nodes, key=lambda x: x[0])

    # Remove any collisions to the SAME node:
    node_labels = [node[0] for node in new_equivalent_nodes]
    new_equivalent_nodes = [
        node for node in new_equivalent_nodes if node_labels.count(node[0]) == 1
    ]
    return new_equivalent_nodes


def _log_new_isomorphic_nodes(
        self,
        new_isomorphic_nodes: List[Tuple[str, str]],
        iso_node_groups: Dict[str, List[str]],
        node_to_iso_group_dict: Dict[str, str],
        subgraphs_dict: Dict,
        node_to_subgraph_dict: Dict,
        node_stack: List[List[str]],
):
    """Takes a new set of equivalent nodes, and logs them as equivalent, adds them to their subgraphs,
    and adds them to the stack.

    Args:
        new_isomorphic_nodes: Current set of isomorphic nodes to get the next nodes from.
        iso_node_groups: Dict mapping each isomorphism node group to the list of nodes in it.
        node_to_iso_group_dict: Reverse dict mapping each node to its isomorphism group.
        subgraphs_dict: Dict of information about each subgraph
        node_to_subgraph_dict: Dict mapping each node to its subgraph
        node_stack: List of lists of isomorphic nodes in the stack.
    """
    if len(new_isomorphic_nodes) > 0:
        iso_group_label = new_isomorphic_nodes[0][0]
        equivalent_node_labels = [tup[0] for tup in new_isomorphic_nodes]
        iso_node_groups[iso_group_label] = equivalent_node_labels[:]
        for node_label in equivalent_node_labels:
            node_to_iso_group_dict[node_label] = iso_group_label
        for node_label, node_subgraph in new_isomorphic_nodes:
            node = self[node_label]
            subgraphs_dict[node_subgraph]["node_set"].add(node_label)
            if node.computed_with_params:
                subgraphs_dict[node_subgraph]["param_nodes"].add(node_label)
            node_to_subgraph_dict[node_label] = subgraphs_dict[node_subgraph]
        node_stack.append(equivalent_node_labels)


def _assign_and_log_isomorphic_nodes_to_same_layers(
        self,
        iso_node_groups: Dict[str, List],
        node_to_subgraph_dict: Dict,
        adjacent_subgraphs: Dict,
):
    """After extending the subgraphs to maximum size and identifying adjacent subgraphs,
    goes through and labels the layers as corresponding to each other. The rule is that nodes will be
    labeled as corresponding if 1) they are isomorphic with respect to the starting node, and
    2) the subgraphs either contain a param node, or are adjacent.

    Args:
        iso_node_groups: Dict specifying list of isomorphic nodes in each group
        node_to_subgraph_dict: Dict mapping each node to the subgraph its in.
        adjacent_subgraphs: Dict mapping each subgraph to set of adjacent subgraphs.
    """
    # Go through each set of isomorphic nodes, and further partition them into nodes assigned to same layer:
    same_layer_node_groups = _group_isomorphic_nodes_to_same_layers(
        self, iso_node_groups, node_to_subgraph_dict, adjacent_subgraphs
    )

    # Finally, label the nodes corresponding to the same layer.
    for layer_label, layer_nodes in same_layer_node_groups.items():
        # Skip if the new layer asssignment reduces the number of equivalent layers.
        if len(layer_nodes) < max(
                [len(self[layer].same_layer_operations) for layer in layer_nodes]
        ):
            continue
        # convert to list and sort
        layer_nodes = sorted(
            list(layer_nodes), key=lambda layer: self[layer].realtime_tensor_num
        )
        for n, node_label in enumerate(layer_nodes):
            node = self[node_label]
            node.layer_label_raw = layer_label
            node.same_layer_operations = layer_nodes
            node.pass_num = n + 1
            node.layer_passes_total = len(layer_nodes)


def _group_isomorphic_nodes_to_same_layers(
        self,
        iso_node_groups: Dict[str, List],
        node_to_subgraph_dict: Dict,
        adjacent_subgraphs: Dict,
) -> Dict:
    same_layer_node_groups = defaultdict(
        set
    )  # dict of nodes assigned to the same layer
    node_to_layer_group_dict = (
        {}
    )  # reverse mapping: each node to its equivalent layer group

    for iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(iso_nodes_orig)
        for node1_label, node2_label in it.combinations(iso_nodes, 2):
            node1_subgraph = node_to_subgraph_dict[node1_label]
            node2_subgraph = node_to_subgraph_dict[node2_label]
            node1_subgraph_label = node1_subgraph["starting_node"]
            node2_subgraph_label = node2_subgraph["starting_node"]
            node1_param_types = [
                self[pnode].operation_equivalence_type
                for pnode in node1_subgraph["param_nodes"]
            ]
            node2_param_types = [
                self[pnode].operation_equivalence_type
                for pnode in node2_subgraph["param_nodes"]
            ]
            overlapping_param_types = set(node1_param_types).intersection(
                set(node2_param_types)
            )
            subgraphs_are_adjacent = (
                    node1_subgraph_label in adjacent_subgraphs
                    and node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label]
            )
            if (len(overlapping_param_types) > 0) or subgraphs_are_adjacent:
                earlier_node_label = sorted([node1_label, node2_label])[
                    0
                ]  # layer label always the first node
                if earlier_node_label in node_to_layer_group_dict:
                    layer_group = node_to_layer_group_dict[earlier_node_label]
                else:
                    layer_group = earlier_node_label
                same_layer_node_groups[layer_group].update(
                    {node1_label, node2_label}
                )
                node_to_layer_group_dict[node1_label] = layer_group
                node_to_layer_group_dict[node2_label] = layer_group

    return same_layer_node_groups


def _fix_modules_for_internal_tensors(self):
    """
    Since internally initialized tensors don't automatically know what module they're in,
    this function infers this by tracing back from tensors that came from the input.
    """
    # Fetch nodes where internally initialized branches first meet a tensor computed from the input:
    node_stack = self._layers_where_internal_branches_merge_with_input[:]

    # Now go through the stack and work backwards up the internally initialized branch, fixing the
    # module containment labels as we go.

    nodes_seen = set()
    while len(node_stack) > 0:
        node_label = node_stack.pop()
        node = self[node_label]
        # Propagate modules for any parent nodes:
        for parent_label in node.parent_layers:
            parent_node = self[parent_label]
            if (not parent_node.has_input_ancestor) and (
                    parent_label not in nodes_seen
            ):
                _fix_modules_for_single_internal_tensor(
                    node, parent_node, "parent", node_stack, nodes_seen
                )

        # And for any internally generated child nodes:
        for child_label in node.child_layers:
            child_node = self[child_label]
            if any(
                    [
                        node.has_input_ancestor,
                        child_node.has_input_ancestor,
                        child_label in nodes_seen,
                        child_node.is_output_layer,
                    ]
            ):
                continue
            _fix_modules_for_single_internal_tensor(
                node, child_node, "child", node_stack, nodes_seen
            )

    # Now that the module containment is fixed, add this to the operation equivalence types.
    for layer in self:
        module_str = "_".join(
            [
                module_pass[0]
                for module_pass in layer.containing_modules_origin_nested
            ]
        )
        layer.operation_equivalence_type += module_str


def _fix_modules_for_single_internal_tensor(
        starting_node: TensorLogEntry,
        node_to_fix: TensorLogEntry,
        node_type_to_fix: str,
        node_stack: List[str],
        nodes_seen: Set[str],
):
    """Helper function to fix the containing modules for a single internally generated tensor.
    The rule is, start from the child node, and apply in reverse any modules that were entered or exited.

    Args:
        starting_node: Source node that has the correct module information
        node_to_fix: Parent of the source node
        node_type_to_fix: either 'child' or 'parent'
        node_stack: Stack of nodes to consider
        nodes_seen: Nodes seen so far
    """
    node_to_fix_label = node_to_fix.tensor_label_raw
    node_to_fix.containing_modules_origin_nested = (
        starting_node.containing_modules_origin_nested.copy()
    )
    if node_type_to_fix == "parent":
        thread_modules = starting_node.module_entry_exit_threads_inputs[
            node_to_fix.tensor_label_raw
        ]
        step_val = -1
    elif node_type_to_fix == "child":
        thread_modules = node_to_fix.module_entry_exit_threads_inputs[
            starting_node.tensor_label_raw
        ]
        step_val = 1
    else:
        raise ValueError("node_type_to_fix must be 'parent' or 'child'")

    for enter_or_exit, module_address, module_pass in thread_modules[::step_val]:
        module_pass_label = (module_address, module_pass)
        if node_type_to_fix == "parent":
            if (enter_or_exit == "+") and (module_pass_label in node_to_fix.containing_modules_origin_nested):
                node_to_fix.containing_modules_origin_nested.remove(
                    module_pass_label
                )
            elif enter_or_exit == "-":
                node_to_fix.containing_modules_origin_nested.append(
                    module_pass_label
                )
        elif node_type_to_fix == "child":
            if enter_or_exit == "+":
                node_to_fix.containing_modules_origin_nested.append(
                    module_pass_label
                )
            elif enter_or_exit == "-":
                node_to_fix.containing_modules_origin_nested.remove(
                    module_pass_label
                )
    node_stack.append(node_to_fix_label)
    nodes_seen.add(node_to_fix_label)


def _fix_buffer_layers(self):
    """Connect the buffer parents, merge duplicate buffer nodes, and label buffer passes correctly.
    Buffers are duplicates if they happen in the same module, have the same value, and have the same parents.
    """
    buffer_counter = defaultdict(lambda: 1)
    buffer_hash_groups = defaultdict(list)

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        if layer.buffer_parent is not None:
            layer.parent_layers.append(layer.buffer_parent)
            self[layer.buffer_parent].child_layers.append(layer_label)
            layer.func_applied = identity
            layer.func_applied_name = 'identity'
            layer.has_input_ancestor = True
            layer.input_ancestors.update(self[layer.buffer_parent].input_ancestors)
            layer.orig_ancestors.remove(layer.tensor_label_raw)
            layer.orig_ancestors.update(self[layer.buffer_parent].orig_ancestors)
            layer.parent_layer_arg_locs['args'][0] = layer.buffer_parent
            if (self[layer.buffer_parent].tensor_contents is not None) and (layer.creation_args is not None):
                layer.creation_args.append(self[layer.buffer_parent].tensor_contents.detach().clone())

        buffer_hash = str(layer.containing_modules_origin_nested) + str(layer.buffer_parent) + layer.buffer_address
        buffer_hash_groups[buffer_hash].append(layer_label)

    # Now go through and merge any layers with the same hash and the same value.
    for _, buffers_orig in buffer_hash_groups.items():
        buffers = buffers_orig[1:]
        unique_buffers = buffers_orig[0:]
        for b, buffer_label in enumerate(buffers):
            for unique_buffer_label in unique_buffers:
                buffer = self[buffer_label]
                unique_buffer = self[unique_buffer_label]
                if ((buffer.tensor_contents is not None) and (unique_buffer.tensor_contents is not None) and
                        (torch.equal(buffer.tensor_contents, unique_buffer.tensor_contents))):
                    self._merge_buffer_entries(unique_buffer, buffer)
                    break
                unique_buffers.append(buffer)

    # And relabel the buffer passes.

    for layer_label in self.buffer_layers:
        layer = self[layer_label]
        buffer_address = layer.buffer_address
        layer.buffer_pass = buffer_counter[buffer_address]
        self.buffer_num_passes[buffer_address] = buffer_counter[buffer_address]
        buffer_counter[buffer_address] += 1


def _merge_buffer_entries(self, source_buffer: TensorLogEntry,
                          buffer_to_remove: TensorLogEntry):
    """Merges two identical buffer layers.
    """
    for child_layer in buffer_to_remove.child_layers:
        if child_layer not in source_buffer.child_layers:
            source_buffer.child_layers.append(child_layer)
        self[child_layer].parent_layers.remove(buffer_to_remove.tensor_label_raw)
        self[child_layer].parent_layers.append(source_buffer.tensor_label_raw)
        if buffer_to_remove.tensor_label_raw in self[child_layer].internally_initialized_parents:
            self[child_layer].internally_initialized_parents.remove(buffer_to_remove.tensor_label_raw)
            self[child_layer].internally_initialized_parents.append(source_buffer.tensor_label_raw)

        for arg_type in ['args', 'kwargs']:
            for arg_label, arg_val in self[child_layer].parent_layer_arg_locs[arg_type].items():
                if arg_val == buffer_to_remove.tensor_label_raw:
                    self[child_layer].parent_layer_arg_locs[arg_type][arg_label] = source_buffer.tensor_label_raw

    for parent_layer in buffer_to_remove.parent_layers:
        if parent_layer not in source_buffer.parent_layers:
            source_buffer.parent_layers.append(parent_layer)
        self[parent_layer].child_layers.remove(buffer_to_remove.tensor_label_raw)
        self[parent_layer].child_layers.append(source_buffer.tensor_label_raw)

    for parent_layer in buffer_to_remove.internally_initialized_parents:
        if parent_layer not in source_buffer.internally_initialized_parents:
            source_buffer.internally_initialized_parents.append(parent_layer)

    if buffer_to_remove.tensor_label_raw in source_buffer.spouse_layers:
        source_buffer.spouse_layers.remove(buffer_to_remove.tensor_label_raw)

    if buffer_to_remove.tensor_label_raw in source_buffer.sibling_layers:
        source_buffer.sibling_layers.remove(buffer_to_remove.tensor_label_raw)

    for spouse_layer in buffer_to_remove.spouse_layers:
        if buffer_to_remove.tensor_label_raw in self[spouse_layer].spouse_layers:
            self[spouse_layer].spouse_layers.remove(buffer_to_remove.tensor_label_raw)
            self[spouse_layer].spouse_layers.append(source_buffer.tensor_label_raw)

    for sibling_layer in buffer_to_remove.sibling_layers:
        if buffer_to_remove.tensor_label_raw in self[sibling_layer].sibling_layers:
            self[sibling_layer].spouse_layers.remove(buffer_to_remove.tensor_label_raw)
            self[sibling_layer].spouse_layers.append(source_buffer.tensor_label_raw)

    self._raw_tensor_labels_list.remove(buffer_to_remove.tensor_label_raw)
    self._raw_tensor_dict.pop(buffer_to_remove.tensor_label_raw)

    for layer in self:
        if buffer_to_remove.tensor_label_raw in layer.orig_ancestors:
            layer.orig_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.orig_ancestors.add(source_buffer.tensor_label_raw)
        if buffer_to_remove.tensor_label_raw in layer.internally_initialized_ancestors:
            layer.internally_initialized_ancestors.remove(buffer_to_remove.tensor_label_raw)
            layer.internally_initialized_ancestors.add(source_buffer.tensor_label_raw)

    self._remove_log_entry(buffer_to_remove, remove_references=True)


def _map_raw_tensor_labels_to_final_tensor_labels(self):
    """
    Determines the final label for each tensor, and stores this mapping as a dictionary
    in order to then go through and rename everything in the next preprocessing step.
    """
    raw_to_final_layer_labels = {}
    final_to_raw_layer_labels = {}
    layer_type_counter = defaultdict(lambda: 1)
    layer_total_counter = 1
    for tensor_log_entry in self:
        layer_type = tensor_log_entry.layer_type
        pass_num = tensor_log_entry.pass_num
        if pass_num == 1:
            layer_type_num = layer_type_counter[layer_type]
            layer_type_counter[layer_type] += 1
            if layer_type in ["input", "buffer"]:
                layer_total_num = 0
            else:
                layer_total_num = layer_total_counter
                layer_total_counter += 1

        else:  # inherit layer numbers from first pass of the layer
            first_pass_tensor = self[tensor_log_entry.same_layer_operations[0]]
            layer_type_num = first_pass_tensor.layer_type_num
            if layer_type in ["input", "buffer"]:
                layer_total_num = 0
            else:
                layer_total_num = first_pass_tensor.layer_total_num
        tensor_log_entry.layer_type_num = layer_type_num
        tensor_log_entry.layer_total_num = layer_total_num

        if layer_type not in ["input", "output", "buffer"]:
            tensor_log_entry.layer_label_w_pass = (
                f"{layer_type}_{layer_type_num}_{layer_total_num}:{pass_num}"
            )
            tensor_log_entry.layer_label_no_pass = (
                f"{layer_type}_{layer_type_num}_{layer_total_num}"
            )
        else:
            tensor_log_entry.layer_label_w_pass = (
                f"{layer_type}_{layer_type_num}:{pass_num}"
            )
            tensor_log_entry.layer_label_no_pass = f"{layer_type}_{layer_type_num}"

        tensor_log_entry.layer_label_w_pass_short = (
            f"{layer_type}_{layer_type_num}:{pass_num}"
        )
        tensor_log_entry.layer_label_no_pass_short = (
            f"{layer_type}_{layer_type_num}"
        )
        if tensor_log_entry.layer_passes_total == 1:
            tensor_log_entry.layer_label = tensor_log_entry.layer_label_no_pass
            tensor_log_entry.layer_label_short = (
                tensor_log_entry.layer_label_no_pass_short
            )
        else:
            tensor_log_entry.layer_label = tensor_log_entry.layer_label_w_pass
            tensor_log_entry.layer_label_short = (
                tensor_log_entry.layer_label_w_pass_short
            )
        raw_to_final_layer_labels[
            tensor_log_entry.tensor_label_raw
        ] = tensor_log_entry.layer_label
        final_to_raw_layer_labels[
            tensor_log_entry.layer_label
        ] = tensor_log_entry.tensor_label_raw
    self._raw_to_final_layer_labels = raw_to_final_layer_labels
    self._final_to_raw_layer_labels = final_to_raw_layer_labels


def _log_final_info_for_all_layers(self):
    """
    Goes through all layers (before discarding unsaved ones), and logs final info about the model
    and the layers that pertains to all layers (not just saved ones).
    """
    unique_layers_seen = (
        set()
    )  # to avoid double-counting params of recurrent layers
    operation_num = 1
    for t, tensor_entry in enumerate(self):
        if tensor_entry.layer_type in ["input", "buffer"]:
            tensor_entry.operation_num = 0
        elif tensor_entry.layer_type == "output":
            tensor_entry.operation_num = None  # fix later
        else:
            tensor_entry.operation_num = operation_num
            self.num_operations += 1
            operation_num += 1

        # Replace any layer names with their final names:
        _replace_layer_names_for_tensor_entry(self, tensor_entry)

        # Log the module hierarchy information:
        _log_module_hierarchy_info_for_layer(self, tensor_entry)
        if tensor_entry.bottom_level_submodule_pass_exited is not None:
            submodule_pass_nice_name = ":".join(
                [str(i) for i in tensor_entry.bottom_level_submodule_pass_exited]
            )
            tensor_entry.bottom_level_submodule_pass_exited = (
                submodule_pass_nice_name
            )

        # Tally the tensor sizes:
        self.tensor_fsize_total += tensor_entry.tensor_fsize

        # Tally the parameter sizes:
        if (
                tensor_entry.layer_label_no_pass not in unique_layers_seen
        ):  # only count params once
            if tensor_entry.computed_with_params:
                self.total_param_layers += 1
            self.total_params += tensor_entry.num_params_total
            self.total_param_tensors += tensor_entry.num_param_tensors
            self.total_params_fsize += tensor_entry.parent_params_fsize
            # Tally for modules, too.
            for module_name, _ in tensor_entry.containing_modules_origin_nested:
                self.module_nparams[module_name] += tensor_entry.num_params_total

        unique_layers_seen.add(tensor_entry.layer_label_no_pass)

        # Tally elapsed time:

        self.elapsed_time_function_calls += tensor_entry.func_time_elapsed

        # Update model structural information:
        if len(tensor_entry.child_layers) > 1:
            self.model_is_branching = True
        if tensor_entry.layer_passes_total > self.model_max_recurrent_loops:
            self.model_is_recurrent = True
            self.model_max_recurrent_loops = tensor_entry.layer_passes_total
        if tensor_entry.in_cond_branch:
            self.model_has_conditional_branching = True

    for layer in self.output_layers:
        self[layer].operation_num = self.num_operations

    # Extract the module hierarchy information
    for module in self.top_level_module_passes:
        module_no_pass = module.split(":")[0]
        if module_no_pass not in self.top_level_modules:
            self.top_level_modules.append(module_no_pass)

    for module_parent, module_children in self.module_pass_children.items():
        module_parent_nopass = module_parent.split(":")[0]
        for module_child in module_children:
            module_child_nopass = module_child.split(":")[0]
            if (
                    module_child_nopass
                    not in self.module_children[module_parent_nopass]
            ):
                self.module_children[module_parent_nopass].append(
                    module_child_nopass
                )

    self.num_tensors_total = len(self)

    # Save the nice versions of the filesize fields:
    self.tensor_fsize_total_nice = human_readable_size(self.tensor_fsize_total)
    self.total_params_fsize_nice = human_readable_size(self.total_params_fsize)


def _log_time_elapsed(self):
    self.pass_end_time = time.time()
    self.elapsed_time_cleanup = (
            self.pass_end_time
            - self.pass_start_time
            - self.elapsed_time_setup
            - self.elapsed_time_forward_pass
    )
    self.elapsed_time_total = self.pass_end_time - self.pass_start_time
    self.elapsed_time_torchlens_logging = (
            self.elapsed_time_total - self.elapsed_time_function_calls
    )


def _replace_layer_names_for_tensor_entry(self, tensor_entry: TensorLogEntry):
    """
    Replaces all layer names in the fields of a TensorLogEntry with their final
    layer names.

    Args:
        tensor_entry: TensorLogEntry to replace layer names for.
    """
    list_fields_to_rename = [
        "parent_layers",
        "orig_ancestors",
        "child_layers",
        "sibling_layers",
        "spouse_layers",
        "input_ancestors",
        "output_descendents",
        "internally_initialized_parents",
        "internally_initialized_ancestors",
        "cond_branch_start_children",
        "equivalent_operations",
        "same_layer_operations",
    ]
    for field in list_fields_to_rename:
        orig_layer_names = getattr(tensor_entry, field)
        field_type = type(orig_layer_names)
        new_layer_names = field_type(
            [
                self._raw_to_final_layer_labels[raw_name]
                for raw_name in orig_layer_names
            ]
        )
        setattr(tensor_entry, field, new_layer_names)

    # Fix the arg locations field:
    for arg_type in ["args", "kwargs"]:
        for key, value in tensor_entry.parent_layer_arg_locs[arg_type].items():
            tensor_entry.parent_layer_arg_locs[arg_type][
                key
            ] = self._raw_to_final_layer_labels[value]

    # Fix the field names for different children tensor versions:
    new_child_tensor_versions = {}
    for (
            child_label,
            tensor_version,
    ) in tensor_entry.children_tensor_versions.items():
        new_child_tensor_versions[
            self._raw_to_final_layer_labels[child_label]
        ] = tensor_version
    tensor_entry.children_tensor_versions = new_child_tensor_versions


def _log_module_hierarchy_info_for_layer(self, tensor_entry: TensorLogEntry):
    """
    Logs the module hierarchy information for a single layer.

    Args:
        tensor_entry: Log entry to mark the module hierarchy info for.
    """
    containing_module_pass_label = None
    for m, module_pass_label in enumerate(
            tensor_entry.containing_modules_origin_nested
    ):
        module_name, module_pass = module_pass_label
        module_pass_nice_label = f"{module_name}:{module_pass}"
        self.module_num_tensors[module_name] += 1
        self.module_pass_num_tensors[module_pass_nice_label] += 1
        if tensor_entry.layer_label not in self.module_layers[module_name]:
            self.module_layers[module_name].append(tensor_entry.layer_label)
        if (
                tensor_entry.layer_label
                not in self.module_pass_layers[module_pass_nice_label]
        ):
            self.module_pass_layers[module_pass_nice_label].append(
                tensor_entry.layer_label
            )
        if (m == 0) and (
                module_pass_nice_label not in self.top_level_module_passes
        ):
            self.top_level_module_passes.append(module_pass_nice_label)
        else:
            if (containing_module_pass_label is not None) and (
                    module_pass_nice_label
                    not in self.module_pass_children[containing_module_pass_label]
            ):
                self.module_pass_children[containing_module_pass_label].append(
                    module_pass_nice_label
                )
        containing_module_pass_label = module_pass_nice_label
        if self.module_num_passes[module_name] < module_pass:
            self.module_num_passes[module_name] = module_pass
        if module_name not in self.module_addresses:
            self.module_addresses.append(module_name)
        if module_pass_label not in self.module_passes:
            self.module_passes.append(module_pass_nice_label)
    tensor_entry.module_nesting_depth = len(
        tensor_entry.containing_modules_origin_nested
    )


def _remove_unwanted_entries_and_log_remaining(self):
    """Removes entries from ModelHistory that we don't want in the final saved output,
    and logs information about the remaining entries.
    """
    tensors_to_remove = []
    # Quick loop to count how many tensors are saved:
    for tensor_entry in self:
        if tensor_entry.has_saved_activations:
            self.num_tensors_saved += 1

    if self.keep_unsaved_layers:
        num_logged_tensors = len(self)
    else:
        num_logged_tensors = self.num_tensors_saved

    self.layer_list = []
    self.layer_dict_main_keys = {}
    self.layer_labels = []
    self.layer_labels_no_pass = []
    self.layer_labels_w_pass = []
    self.layer_num_passes = {}

    i = 0
    for raw_tensor_label in self._raw_tensor_labels_list:
        tensor_entry = self._raw_tensor_dict[raw_tensor_label]
        # Determine valid lookup keys and relate them to the tensor's realtime operation number:
        if tensor_entry.has_saved_activations or self.keep_unsaved_layers:
            # Add the lookup keys for the layer, to itself and to ModelHistory:
            _add_lookup_keys_for_tensor_entry(
                self, tensor_entry, i, num_logged_tensors
            )

            # Log all information:
            self.layer_list.append(tensor_entry)
            self.layer_dict_main_keys[tensor_entry.layer_label] = tensor_entry
            self.layer_labels.append(tensor_entry.layer_label)
            self.layer_labels_no_pass.append(tensor_entry.layer_label_no_pass)
            self.layer_labels_w_pass.append(tensor_entry.layer_label_w_pass)
            self.layer_num_passes[
                tensor_entry.layer_label
            ] = tensor_entry.layer_passes_total
            if tensor_entry.has_saved_activations:
                self.tensor_fsize_saved += tensor_entry.tensor_fsize
            _trim_and_reorder_tensor_entry_fields(
                tensor_entry
            )  # Final reformatting of fields
            i += 1
        else:
            tensors_to_remove.append(tensor_entry)
            self.unlogged_layers.append(tensor_entry.layer_label)
            self._unsaved_layers_lookup_keys.update(tensor_entry.lookup_keys)

    # Remove unused entries.
    for tensor_entry in tensors_to_remove:
        self._remove_log_entry(tensor_entry, remove_references=False)

    if (self.num_tensors_saved == len(self)) or self.keep_unsaved_layers:
        self._all_layers_logged = True
    else:
        self._all_layers_logged = False

    if self.num_tensors_saved == len(self.layer_list):
        self._all_layers_saved = True
    else:
        self._all_layers_saved = False

    # Make the saved tensor filesize pretty:
    self.tensor_fsize_saved_nice = human_readable_size(self.tensor_fsize_saved)


def _add_lookup_keys_for_tensor_entry(
        self, tensor_entry: TensorLogEntry, tensor_index: int, num_tensors_to_keep: int
):
    """Adds the user-facing lookup keys for a TensorLogEntry, both to itself
    and to the ModelHistory top-level record.

    Args:
        tensor_entry: TensorLogEntry to get the lookup keys for.
    """
    tensor_entry.index_in_saved_log = tensor_index

    # The "default" keys: including the pass if multiple passes, excluding if one pass.
    lookup_keys_for_tensor = [
        tensor_entry.layer_label,
        tensor_entry.layer_label_short,
        tensor_index,
        tensor_index - num_tensors_to_keep,
    ]

    # If just one pass, also allow indexing by pass label.
    if tensor_entry.layer_passes_total == 1:
        lookup_keys_for_tensor.extend(
            [tensor_entry.layer_label_w_pass, tensor_entry.layer_label_w_pass_short]
        )

    # Relabel the module passes if the first pass:
    if self.logging_mode == "exhaustive":
        tensor_entry.module_passes_exited = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in tensor_entry.module_passes_exited
        ]
        tensor_entry.module_passes_entered = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in tensor_entry.module_passes_entered
        ]
        if tensor_entry.containing_module_origin is not None:
            tensor_entry.containing_module_origin = ":".join(
                [str(i) for i in tensor_entry.containing_module_origin]
            )
        tensor_entry.containing_modules_origin_nested = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in tensor_entry.containing_modules_origin_nested
        ]
        if (tensor_entry.containing_module_origin is None) and len(
                tensor_entry.containing_modules_origin_nested) > 0:
            tensor_entry.containing_module_origin = tensor_entry.containing_modules_origin_nested[-1]

    # Allow indexing by modules exited as well:
    for module_pass in tensor_entry.module_passes_exited:
        module_name, _ = module_pass.split(":")
        lookup_keys_for_tensor.append(f"{module_pass}")
        if self.module_num_passes[module_name] == 1:
            lookup_keys_for_tensor.append(f"{module_name}")

    # Allow using buffer/input/output address as key, too:
    if tensor_entry.is_buffer_layer:
        if self.buffer_num_passes[tensor_entry.buffer_address] == 1:
            lookup_keys_for_tensor.append(tensor_entry.buffer_address)
        lookup_keys_for_tensor.append(f"{tensor_entry.buffer_address}:{tensor_entry.buffer_pass}")
    elif tensor_entry.is_input_layer or tensor_entry.is_output_layer:
        lookup_keys_for_tensor.append(tensor_entry.input_output_address)

    lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

    # Log in both the tensor and in the ModelHistory object.
    tensor_entry.lookup_keys = lookup_keys_for_tensor
    for lookup_key in lookup_keys_for_tensor:
        self._lookup_keys_to_tensor_num_dict[
            lookup_key
        ] = tensor_entry.realtime_tensor_num
        self._tensor_num_to_lookup_keys_dict[
            tensor_entry.realtime_tensor_num
        ].append(lookup_key)
        self.layer_dict_all_keys[lookup_key] = tensor_entry


def _trim_and_reorder_tensor_entry_fields(tensor_entry: TensorLogEntry):
    """
    Sorts the fields in TensorLogEntry into their desired order, and trims any
    fields that aren't useful after the pass.
    """
    new_dir_dict = OrderedDict()
    for field in TENSOR_LOG_ENTRY_FIELD_ORDER:
        new_dir_dict[field] = getattr(tensor_entry, field)
    for field in dir(tensor_entry):
        if field.startswith("_"):
            new_dir_dict[field] = getattr(tensor_entry, field)
    tensor_entry.__dict__ = new_dir_dict


def _rename_model_history_layer_names(self):
    """Renames all the metadata fields in ModelHistory with the final layer names, replacing the
    realtime debugging names.
    """
    list_fields_to_rename = [
        "input_layers",
        "output_layers",
        "buffer_layers",
        "internally_initialized_layers",
        "_layers_where_internal_branches_merge_with_input",
        "internally_terminated_layers",
        "internally_terminated_bool_layers",
        "layers_with_saved_gradients",
        "layers_with_saved_activations",
    ]
    for field in list_fields_to_rename:
        tensor_labels = getattr(self, field)
        setattr(
            self,
            field,
            [
                self._raw_to_final_layer_labels[tensor_label]
                for tensor_label in tensor_labels
            ],
        )

    new_param_tensors = {}
    for key, values in self.layers_computed_with_params.items():
        new_key = self[values[0]].layer_label
        new_param_tensors[new_key] = [
            self._raw_to_final_layer_labels[tensor_label] for tensor_label in values
        ]
    self.layers_computed_with_params = new_param_tensors

    new_equiv_operations_tensors = {}
    for key, values in self.equivalent_operations.items():
        new_equiv_operations_tensors[key] = set(
            [
                self._raw_to_final_layer_labels[tensor_label]
                for tensor_label in values
            ]
        )
    self.equivalent_operations = new_equiv_operations_tensors

    new_same_layer_operations = {}
    for key, values in self.same_layer_operations.items():
        new_key = self._raw_to_final_layer_labels[key]
        new_same_layer_operations[new_key] = [
            self._raw_to_final_layer_labels[tensor_label] for tensor_label in values
        ]
    self.same_layer_operations = new_same_layer_operations

    for t, (child, parent) in enumerate(self.conditional_branch_edges):
        new_child, new_parent = (
            self._raw_to_final_layer_labels[child],
            self._raw_to_final_layer_labels[parent],
        )
        self.conditional_branch_edges[t] = (new_child, new_parent)

    for module_pass, arglist in self.module_layer_argnames.items():
        inds_to_remove = []
        for a, arg in enumerate(arglist):
            raw_name = self.module_layer_argnames[module_pass][a][0]
            if raw_name not in self._raw_to_final_layer_labels:
                inds_to_remove.append(a)
                continue
            new_name = self._raw_to_final_layer_labels[raw_name]
            argname = self.module_layer_argnames[module_pass][a][1]
            self.module_layer_argnames[module_pass][a] = (new_name, argname)
        self.module_layer_argnames[module_pass] = [self.module_layer_argnames[module_pass][i]
                                                   for i in range(len(arglist)) if i not in inds_to_remove]


def _trim_and_reorder_model_history_fields(self):
    """
    Sorts the fields in ModelHistory into their desired order, and trims any
    fields that aren't useful after the pass.
    """
    new_dir_dict = OrderedDict()
    for field in MODEL_HISTORY_FIELD_ORDER:
        new_dir_dict[field] = getattr(self, field)
    for field in dir(self):
        if field.startswith("_"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_dir_dict[field] = getattr(self, field)
    self.__dict__ = new_dir_dict


def _undecorate_all_saved_tensors(self):
    """Utility function to undecorate all saved tensors."""
    tensors_to_undecorate = []
    for layer_label in self.layer_labels:
        tensor_entry = self.layer_dict_main_keys[layer_label]
        if tensor_entry.tensor_contents is not None:
            tensors_to_undecorate.append(tensor_entry.tensor_contents)

        tensors_to_undecorate.extend(
            get_vars_of_type_from_obj(
                tensor_entry.creation_args, torch.Tensor, search_depth=2
            )
        )
        tensors_to_undecorate.extend(
            get_vars_of_type_from_obj(
                tensor_entry.creation_kwargs, torch.Tensor, search_depth=2
            )
        )

    for t in tensors_to_undecorate:
        if hasattr(t, "tl_tensor_label_raw"):
            delattr(t, "tl_tensor_label_raw")


def _delete_raw_tensor_entries(self):
    """Deletes the raw tensor entries, leaving only the post-processed entries."""
    for entry_name, tensor_entry in self._raw_tensor_dict.items():
        self._remove_log_entry(tensor_entry)
    self._raw_tensor_dict.clear()


def _set_pass_finished(self):
    """Sets the ModelHistory to "pass finished" status, indicating that the pass is done, so
    the "final" rather than "realtime debugging" mode of certain functions should be used.
    """
    for layer_label in self.layer_dict_main_keys:
        tensor = self.layer_dict_main_keys[layer_label]
        tensor._pass_finished = True
    self._pass_finished = True


def _roll_graph(self):
    """
    Converts the graph to rolled-up format for plotting purposes, such that each node now represents
    all passes of a given layer instead of having separate nodes for each pass.
    """
    for layer_label, node in self.layer_dict_main_keys.items():
        layer_label_no_pass = self[layer_label].layer_label_no_pass
        if (
                layer_label_no_pass in self.layer_dict_rolled
        ):  # If rolled-up layer has already been added, fetch it:
            rolled_node = self.layer_dict_rolled[layer_label_no_pass]
        else:  # If it hasn't been added, make it:
            rolled_node = RolledTensorLogEntry(node)
            self.layer_dict_rolled[node.layer_label_no_pass] = rolled_node
            self.layer_list_rolled.append(rolled_node)
        rolled_node.update_data(node)
        rolled_node.add_pass_info(node)
