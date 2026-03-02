"""Steps 1-4: Output nodes, ancestry tracing, orphan removal, distance marking."""

from collections import OrderedDict
from typing import TYPE_CHECKING, List

import torch

from ..helper_funcs import (
    clean_to,
    human_readable_size,
    identity,
    log_current_rng_states,
    safe_copy,
    tensor_nanequal,
    _get_func_call_stack,
)
from ..data_classes.tensor_log import TensorLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _add_output_layers(
    self: "ModelLog", output_tensors: List[torch.Tensor], output_addresses: List[str]
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
        new_output_node.func_call_stack = _get_func_call_stack(self.num_context_lines)
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
        new_output_node.parent_param_logs = []
        new_output_node.num_param_tensors = 0
        new_output_node.parent_param_shapes = []
        new_output_node.num_params_total = int(0)
        new_output_node.num_params_trainable = 0
        new_output_node.num_params_frozen = 0
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
        new_output_node.module_passes_exited = output_node.containing_modules_origin_nested
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
        new_output_node.has_siblings = False
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

        # Track child tensor variations for output nodes.
        new_output_node.has_child_tensor_variations = False
        new_output_node.children_tensor_versions = {}
        if output_node.has_saved_activations:
            actual_output = safe_copy(output_tensors[i])
            if output_node.output_device not in [str(actual_output.device), "same"]:
                actual_output = clean_to(actual_output, output_node.output_device)
            if self.activation_postfunc is not None:
                self._pause_logging = True
                try:
                    actual_output = self.activation_postfunc(actual_output)
                finally:
                    self._pause_logging = False
            if not tensor_nanequal(actual_output, output_node.tensor_contents):
                output_node.children_tensor_versions[new_output_node.tensor_label_raw] = (
                    actual_output
                )
                output_node.has_child_tensor_variations = True
                new_output_node.tensor_contents = actual_output

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
                node.output_descendents.update(self[child_node_label].output_descendents)
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
        if (len(tensor_entry.child_layers) == 0) and (not tensor_entry.is_output_layer):
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
    current_node: TensorLog,
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
