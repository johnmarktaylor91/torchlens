# This file is for defining the ModelHistory class that stores the representation of the forward pass.

import copy
import itertools as it
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

from torchlens.helper_funcs import get_rng_states, make_var_iterable, \
    get_tensor_memory_amount, human_readable_size, identity, print_override, safe_copy, make_short_barcode_from_input


class TensorLogEntry:
    def __init__(self, t: torch.Tensor):
        """Object that stores information about a single tensor operation in the forward pass,
        including metadata and the tensor itself (if specified).

        Args:
            t: the tensor
        """
        for field in dir(t):
            if not field.startswith('tl_'):  # tl is the keyword for marking relevant fields.
                continue
            field_stripped = field[3:]
            setattr(self, field_stripped, getattr(t, field))
        self.pass_finished = False
        self.tensor_contents = None
        self.creation_args = []
        self.creation_kwargs = {}

    def copy(self):
        """Return a copy of itself.

        Returns:
            Copy of itself.
        """
        copied_entry = copy.copy(self)
        for field in dir(self):
            if field.startswith('_'):
                continue
            setattr(copied_entry, field, getattr(self, field))
        return copied_entry

    def save_tensor_data(self,
                         t: torch.Tensor,
                         t_args: List,
                         t_kwargs: Dict):
        """Saves the tensor data for a given tensor operation.

        Args:
            t: the tensor.
            t_args: tensor positional arguments for the operation
            t_kwargs: tensor keyword arguments for the operation
        """
        # The tensor itself:
        self.tensor_contents = safe_copy(t)

        # Tensor args and kwargs:
        creation_args = []
        for arg in t_args:
            if issubclass(type(arg), (torch.Tensor, torch.nn.Parameter)):
                creation_args.append(safe_copy(arg))
            else:
                creation_args.append(arg)

        creation_kwargs = {}
        for key, value in t_kwargs.items():
            if issubclass(type(value), (torch.Tensor, torch.nn.Parameter)):
                creation_kwargs[key] = safe_copy(value)
            else:
                creation_kwargs[key] = value

        self.creation_args = creation_args
        self.creation_kwargs = creation_kwargs

    def update_tensor_metadata(self, t: torch.Tensor):
        """Updates the logged metadata for a tensor (e.g., if it enters or exits a module)

        Args:
            t: The tensor
        """
        for field in dir(t):
            if not field.startswith('tl_'):  # tl is the keyword for marking relevant fields.
                continue
            field_stripped = field[3:]
            setattr(self, field_stripped, getattr(t, field))

    def _str_during_pass(self):
        s = f"Tensor {self.tensor_label_raw} (layer {self.layer_label_raw}) (PASS NOT FINISHED):"
        s += f"\n\tPass: {self.pass_num}"
        s += f"\n\tTensor info: shape {self.tensor_shape}, dtype {self.tensor_dtype}"
        s += f"\n\tComputed from params: {self.computed_from_params}"
        s += f"\n\tComputed in modules: {self.containing_modules_origin_nested}"
        s += f"\n\tOutput of modules: {self.module_passes_exited}"
        if self.is_bottom_level_submodule_output:
            s += f" (bottom-level submodule output)"
        else:
            s += f" (not bottom-level submodule output)"
        s += f"\n\tFamily info:"
        s += f"\n\t\tParents: {self.parent_tensors}"
        s += f"\n\t\tChildren: {self.child_tensors}"
        s += f"\n\t\tSpouses: {self.spouse_tensors}"
        s += f"\n\t\tSiblings: {self.sibling_tensors}"
        s += f"\n\t\tOriginal Ancestors: {self.orig_ancestors} " \
             f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internally_initialized_ancestors}"
        s += f"\n\t\tOutput Descendents: {self.output_descendents} " \
             f"(min dist {self.min_distance_from_output} nodes, max dist {self.max_distance_from_output} nodes)"
        if self.tensor_contents is not None:
            s += f"\n\tTensor contents: \n{print_override(self.tensor_contents, '__str__')}"
        return s

    def __str__(self):
        if self.pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def __repr__(self):
        return self.__str__()


class RepeatedSubgraph:
    def __init__(self,
                 subgraph_structure: set[Tuple[str, str]],
                 subgraph_hash: str,
                 subgraph_tensors: List[TensorLogEntry]):
        """
        A single repeated subgraph type: a sorted tuple of tuples encoding the types of edges, a compact hash of that
        tuple for lookup, and a list of the instances of that subgraph, where each instance is itself a
        list of the actual nodes in that subgraph. Initialized using an instance of that subgraph type as a template,
        but does not populate the list just yet; the input list of nodes is just a "template".

        Args:
            subgraph_structure: sorted list of tuples, where each tuple is (parent_label, child_label)
            subgraph_hash: Lookup hash of the subgraph structure.
            subgraph_tensors: list of actual nodes for the instance of the subgraph.
        """
        self.subgraph_structure = subgraph_structure
        self.subgraph_hash = subgraph_hash
        self.subgraph_instances = [subgraph_tensors]

    def add_instance(self, node_list: List[TensorLogEntry]):
        """
        Adds an instance of the subgraph to the list of instances.

        Args:
            node_list: List of nodes in the new subgraph.
        """
        self.subgraph_instances.append(node_list)



class RepeatedSubgraphs:
    def __init__(self):
        """
        Class that stores the repeated subgraphs in a model. The overall structure is: there are multiple
        types of repeated subgraphs, each of which has multiple instances, each of which has multiple nodes.
        The types are defined by the structure of the subgraph, which is encoded by a sorted hash of
        the tokens involved in the subgraph.
        """
        self.subgraph_dict = {}


    def add_subgraph_instance(self, node_list: List[TensorLogEntry]):
        """Given a list of nodes constituting a new subgraph instance, checks if the instance matches
        an existing subgraph type; if so, adds it to the list of instances for that type, and if not, makes a new
        type.

        TODO: check if an existing subgraph is a strict subset of the new subgraph being added; if so, replace it.

        Args:
            node_list: List of nodes in the subgraph
        """
        subgraph_structure, subgraph_hash = self._get_subgraph_structure_and_hash(node_list)
        if subgraph_hash in self.subgraph_dict:
            self.subgraph_dict[subgraph_hash].add_instance(node_list)
        else:
            self.subgraph_dict[subgraph_hash] = RepeatedSubgraph(subgraph_structure, subgraph_hash, node_list)

    @staticmethod
    def _get_subgraph_structure_and_hash(node_list: List[TensorLogEntry]):
        """Given a list of nodes constituting a new subgraph instance, returns the sorted hash of the
        subgraph structure.

        Args:
            node_list: List of nodes in the subgraph

        Returns:
            The list of tuples constituting the edges that define the node structure, and the hash
            of that list.
        """
        node_set = set([node.tensor_label_raw for node in node_list])
        node_dict = {node.tensor_label_raw: node for node in node_list}
        subgraph_edges = []
        for node in node_list:
            for child_label in node.child_tensors:
                child_node = node_dict[child_label]
                if child_label in node_set:
                    subgraph_edges.append((node.layer_grouping_token, child_node.layer_grouping_token))

        # Sort the list of tuples:

        subgraph_edges_list = sorted(subgraph_edges, key=lambda x: ''.join(str(i) for i in x))
        subgraph_edges_set = set(subgraph_edges_list)
        subgraph_hash = make_short_barcode_from_input(subgraph_edges)
        return subgraph_edges_set, subgraph_hash



class ModelHistory:
    def __init__(self,
                 model_name: str,
                 random_seed_used: int,
                 tensor_nums_to_save: Union[List[int], str] = 'all'):
        """Object that stores the history of a model's forward pass.
        Both logs the history in real time, and stores a nice
        representation of the full history for the user afterward.

        Args:
            tensor_nums_to_save: the numbers for the tensors to save
            during the forward pass (e.g., the 2nd tensor generated,
            the fifth, etc.). If 'all', saves all tensors.
        """
        # General info
        self.model_name = model_name
        self.pass_finished = False
        self.random_seed_used = random_seed_used

        # Model structure info
        self.model_is_branching = False
        self.model_is_recurrent = False

        # Tensor info
        self.raw_tensor_dict = OrderedDict()
        self.raw_tensor_labels_list = []
        self.tensor_nums_to_save = tensor_nums_to_save
        self.tensor_counter = 1
        self.raw_layer_type_counter = defaultdict(lambda: 1)
        self.input_tensors = []
        self.output_tensors = []
        self.buffer_tensors = []
        self.internally_initialized_tensors = []
        self.internally_terminated_tensors = []
        self.internally_terminated_bool_tensors = []
        self.tensors_computed_with_params = defaultdict(list)
        self.conditional_branch_edges = []

        # Tracking info
        self.track_tensors = True
        self.current_function_call_barcode = None

    def summarize(self):
        """
        Returns an exhaustive summary of the model, including the values of all fields.
        """
        pass

    def to_pandas(self) -> pd.DataFrame:
        """Returns a pandas dataframe with info about each layer.

        Returns:
            Pandas dataframe with info about each layer.
        """
        pass

    #########################################
    ####### Post-Processing Functions #######
    #########################################

    def _add_output_nodes(self):
        """
        Adds dedicated output nodes to the graph.
        """
        new_output_tensors = []
        for i, output_tensor_label in enumerate(self.output_tensors):
            output_node = self[output_tensor_label]
            new_output_node = output_node.copy()
            new_output_node.tensor_label_raw = f"output_{i + 1}_{self.tensor_counter}_raw"
            new_output_node.tensor_num = self.tensor_counter
            self.tensor_counter += 1

            # Fix function information:

            new_output_node.func_applied = None
            new_output_node.func_applied_name = 'none'
            new_output_node.func_time_elapsed = 0
            new_output_node.func_rng_states = get_rng_states()
            new_output_node.num_func_args_total = 0
            new_output_node.num_position_args = 0
            new_output_node.num_keyword_args = 0
            new_output_node.func_position_args_non_tensor = []
            new_output_node.func_keyword_args_non_tensor = {}
            new_output_node.func_all_args_non_tensor = []
            new_output_node.gradfunc = None
            new_output_node.gradfunc_name = 'none'

            # Strip any params:

            new_output_node.computed_from_params = False
            new_output_node.parent_params = []
            new_output_node.parent_param_barcodes = []
            new_output_node.parent_param_passes = {}
            new_output_node.num_param_tensors = 0
            new_output_node.parent_param_shapes = []
            new_output_node.num_params_total = 0
            new_output_node.parent_params_fsize = 0
            new_output_node.parent_params_fsize_nice = human_readable_size(0)

            # Strip module info:

            new_output_node.is_computed_inside_submodule = False
            new_output_node.containing_module_origin = None
            new_output_node.containing_modules_origin_nested = []
            new_output_node.containing_module_final = None
            new_output_node.containing_modules_final_nested = []
            new_output_node.modules_entered = []
            new_output_node.module_passes_entered = []
            new_output_node.is_submodule_input = False
            new_output_node.modules_exited = False
            new_output_node.module_passes_exited = []
            new_output_node.is_submodule_output = False
            new_output_node.is_bottom_level_submodule_output = False
            new_output_node.module_entry_exit_thread = []

            # Fix ancestry information:

            new_output_node.parent_tensors = [output_node.tensor_label_raw]
            new_output_node.sibling_tensors = []
            new_output_node.has_sibling_tensors = False

            # Change original output node:

            output_node.is_output_tensor = False
            output_node.child_tensors = [new_output_node.tensor_label_raw]

            self.raw_tensor_dict[new_output_node.tensor_label_raw] = new_output_node
            self.raw_tensor_labels_list.append(new_output_node.tensor_label_raw)

            new_output_tensors.append(new_output_node.tensor_label_raw)

        self.output_tensors = new_output_tensors

    def _remove_orphan_nodes(self):
        """
        Removes nodes that are connected to neither the input nor the output by flooding in both directions
        from the input and output nodes.
        """
        orig_nodes = set(self.raw_tensor_labels_list)
        nodes_seen = set()
        node_stack = self.input_tensors + self.output_tensors
        while len(node_stack) > 0:
            tensor_label = node_stack.pop()
            nodes_seen.add(tensor_label)
            tensor_entry = self[tensor_label]
            for next_label in tensor_entry.child_tensors + tensor_entry.parent_tensors:
                if next_label not in nodes_seen:
                    node_stack.append(next_label)
        orphan_nodes = orig_nodes - nodes_seen

        # Now remove all orphaned nodes.

        new_tensor_dict = OrderedDict()
        new_tensor_list = []
        for tensor_label in self.raw_tensor_labels_list:
            tensor_entry = self[tensor_label]
            if tensor_label not in orphan_nodes:
                new_tensor_dict[tensor_label] = tensor_entry
                new_tensor_list.append(tensor_label)
            else:
                self.remove_log_entry(tensor_entry)
        self.raw_tensor_labels_list = new_tensor_list
        self.raw_tensor_dict = new_tensor_dict

    def _log_internally_terminated_tensor(self, tensor_label: str):
        tensor_entry = self[tensor_label]
        tensor_entry.terminated_inside_model = True
        if tensor_label not in self.internally_terminated_tensors:
            self.internally_terminated_tensors.append(tensor_label)
            if tensor_entry.is_atomic_bool_tensor and (tensor_label not in self.internally_terminated_bool_tensors):
                self.internally_terminated_bool_tensors.append(tensor_label)
                tensor_entry.is_terminal_bool_tensor = True

    def _flood_graph_from_input_or_output_nodes(self, mode: str):
        """Floods the graph from either the input or output nodes, tracking nodes that aren't seen,
        and the min and max distance from the starting nodes of each node. Traversal is unidirectional
        UNLESS going in the direction of a termin

        Args:
            mode: either 'input' or 'output'

        Returns:
            Set of nodes seen during the traversal
        """
        if mode == 'input':
            starting_nodes = self.input_tensors[:]
            min_field = 'min_distance_from_input'
            max_field = 'max_distance_from_input'
            direction = 'forwards'
            marker_field = 'has_input_ancestor'
            forward_field = 'child_tensors'
        elif mode == 'output':
            starting_nodes = self.output_tensors[:]
            min_field = 'min_distance_from_output'
            max_field = 'max_distance_from_output'
            direction = 'backwards'
            marker_field = 'is_output_ancestor'
            forward_field = 'parent_tensors'
        else:
            raise ValueError("Mode but be either 'input' or 'output'")

        nodes_seen = set()

        # Tuples in format node_label, nodes_since_start, traversal_direction
        node_stack = [(starting_node_label, 0, direction) for starting_node_label in starting_nodes]
        while len(node_stack) > 0:
            current_node_label, nodes_since_start, traversal_direction = node_stack.pop()
            current_node = self[current_node_label]
            nodes_seen.add(current_node_label)
            if getattr(current_node, min_field) is None:
                setattr(current_node, min_field, nodes_since_start)
            else:
                setattr(current_node, min_field, min([nodes_since_start, getattr(current_node, min_field)]))

            if getattr(current_node, max_field) is None:
                setattr(current_node, max_field, nodes_since_start)
            else:
                setattr(current_node, max_field, max([nodes_since_start, getattr(current_node, max_field)]))

            setattr(current_node, marker_field, True)

            if (len(current_node.child_tensors) == 0) and (not current_node.is_output_tensor):
                self._log_internally_terminated_tensor(current_node_label)

            for next_node_label in getattr(current_node, forward_field):
                node_stack.append((next_node_label, nodes_since_start + 1, traversal_direction))

    def _mark_input_output_distances(self):
        """
        Traverses the graph forward and backward, marks the minimum and maximum distances of each
        node from the input and output, and removes any orphan nodes.
        """
        self._flood_graph_from_input_or_output_nodes('input')
        self._flood_graph_from_input_or_output_nodes('output')

    def _mark_conditional_branches(self):
        """Starting from any terminal boolean nodes, backtracks until it finds the beginning of any
        conditional branches.
        """
        terminal_bool_nodes = self.internally_terminated_bool_tensors[:]

        nodes_seen = set()
        node_stack = terminal_bool_nodes.copy()
        while len(node_stack) > 0:
            node_label = node_stack.pop()
            node = self[node_label]
            if node_label in nodes_seen:
                continue
            for next_tensor_label in node.parent_tensors + node.child_tensors:
                next_node = self[next_tensor_label]
                if next_node.is_output_ancestor:  # we found the beginning of a conditional branch
                    next_node.cond_branch_start_children.append(node_label)
                    next_node.in_cond_branch = False
                    nodes_seen.add(next_tensor_label)
                    self.conditional_branch_edges.append((next_tensor_label, node_label))
                else:
                    if next_tensor_label in nodes_seen:
                        continue
                    next_node.in_cond_branch = True
                    node_stack.append(next_tensor_label)

            nodes_seen.add(node_label)

    def _find_repeated_subgraphs_with_params(self):
        """
        Finds all repeated subgraphs in the network that contain params, under the constraints that these
        subgraphs be maximally large. They are allowed to contain multiple params each, they just have
        to be as big as possible to count, such that no subgraph is a subgraph of any other.
        Produces a list of lists of subgraphs, where each list is a list of isomorphic subgraphs, which itself
        is a (consistently ordered) list of the tensors in each subgraph.
        """
        param_subraphs = []
        for param_operation, param_operation_tensors in self.tensors_computed_with_params.items():
            # Crawl backwards from all the tensors either until hitting another parameter operation,
            # or until


        self.param_chunks = []

    def _find_repeated_layers(self):
        """Post-processing function that finds any "repeated" layers. Specifically, it first takes all
        tensors with parameters, and "yokes" them to any contiguous tensors that perform the same operations.
        Then, it looks within, and outside, these "yoked" groups for any loops, and marks these as repeated
        instances as well.
        """
        self._find_repeated_subgraphs_with_params()
        self._find_loops_in_layers_yoked_to_params()
        self._find_loops_in_layers_not_yoked_to_params()

    def postprocess(self):
        """
        After the forward pass, cleans up the log into its final form.
        """
        # Step 1: Add dedicated output nodes

        self._add_output_nodes()

        # Step 2: Remove orphan nodes.

        self._remove_orphan_nodes()

        # Step 3: Find mix/max distance from input and output nodes, find nodes that don't terminate in an output node.

        self._mark_input_output_distances()

        # Step 4: Starting from terminal single boolean tensors, mark the conditional branches.

        self._mark_conditional_branches()

        # Step 5: Identify all loops, mark repeated layers.

        self._find_repeated_layers()

        # Step 6: Annotate the containing modules for all internally-generated tensors.

        # Step 7: Go down tensor list, get the mapping from raw tensor names to final tensor names.

        # Step 8: go down the tensor list, undecorate all tensors, add output nodes, do any tallying/totals/labeling,
        # log the module hierarchy, rename all tensors, get the operation numbers for all layer labels.

    def get_op_nums_from_user_labels(self, which_layers: List[str]) -> List[int]:
        """Given list of user layer labels, returns the original tensor numbers for those labels (i.e.,
        the numbers that were generated on the fly during the forward pass, such that they can be
        saved on a subsequent pass). Raises an error if the user's labels don't correspond to any layers.

        Args:
            which_layers: List of layers to include, using any indexing desired: either the layer label,
            the module label, or the ordinal position of the layer. If a layer has multiple passes and
            none is specified, will return all of them.

        Returns:
            Ordered, unique list of raw tensor numbers associated with the specified layers.
        """
        pass

    def make_tensor_log_entry(self, t: torch.Tensor,
                              t_args: Optional[List] = None,
                              t_kwargs: Optional[Dict] = None):
        """Given a tensor, adds it to the model_history, additionally saving the activations and input
        arguments if specified.

        Args:
            t: tensor to log
            t_args: positional arguments to the function that created the tensor
            t_kwargs: keyword arguments to the function that created the tensor
        """
        if t_args is None:
            t_args = []
        if t_kwargs is None:
            t_kwargs = {}

        new_entry = TensorLogEntry(t)
        if (self.tensor_nums_to_save == 'all') or (t.tl_tensor_num in self.tensor_nums_to_save):
            new_entry.save_tensor_data(t, t_args, t_kwargs)

        self.raw_tensor_dict[new_entry.tensor_label_raw] = new_entry
        self.raw_tensor_labels_list.append(new_entry.tensor_label_raw)

    def update_tensor_log_entry(self, t: torch.Tensor):
        """Given a tensor, updates the log entry for that tensor.

        Args:
            t: tensor for which to update the log entry
        """
        log_entry = self.raw_tensor_dict[t.tl_tensor_label_raw]
        log_entry.update_tensor_metadata(t)

    def add_raw_label_to_tensor(self,
                                t: torch.Tensor,
                                layer_type: str) -> str:
        """Gets the raw label for a layer during the forward pass, and updates relevant counters
        (for the layer type and operation number) to be ready to return the next label;
        'raw' is added to avoid any confusion. Format is {layer_type}_{layer_type_num}_{operation_num}_raw,
        e.g. conv2d_2_4_raw.

        Args:
            t: The raw tensor
            layer_type: Type of layer (e.g., the kind of function, 'input', 'buffer', etc.)

        Returns:
            The layer label.
        """
        layer_type_num = self.raw_layer_type_counter[layer_type]
        operation_num = self.tensor_counter
        tensor_label = f"{layer_type}_{layer_type_num}_{operation_num}_raw"
        t.tl_tensor_label_raw = tensor_label
        t.tl_tensor_num = operation_num

        # Increment the counters to be ready for the next tensor
        self.raw_layer_type_counter[layer_type] += 1
        self.tensor_counter += 1

        return tensor_label

    def log_source_tensor(self,
                          t: torch.Tensor,
                          source: str,
                          buffer_addr: Optional[str] = None):
        """Takes in an input or buffer tensor, marks it in-place with relevant information, and
        adds it to the log.

        Args:
            t: the tensor
            source: either 'input' or 'buffer'
            buffer_addr: Address of the buffer tensor if it's a buffer tensor
        """
        tensor_label = self.add_raw_label_to_tensor(t, source)
        if source == 'input':
            is_input_tensor = True
            has_input_ancestor = True
            is_buffer_tensor = False
            initialized_inside_model = False
            has_internally_initialized_ancestor = False
            input_ancestors = {tensor_label}
            internally_initialized_ancestors = set()
            layer_grouping_token = f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
            self.input_tensors.append(tensor_label)
        elif source == 'buffer':
            is_input_tensor = False
            has_input_ancestor = False
            is_buffer_tensor = True
            initialized_inside_model = True
            has_internally_initialized_ancestor = True
            internally_initialized_ancestors = {tensor_label}
            input_ancestors = set()
            layer_grouping_token = f"buffer_{buffer_addr}"
            self.buffer_tensors.append(tensor_label)
            self.internally_initialized_tensors.append(tensor_label)
        else:
            raise ValueError("source must be either 'input' or 'buffer'")

        # General info
        t.tl_layer_label_raw = t.tl_tensor_label_raw
        t.tl_layer_grouping_token = layer_grouping_token
        t.tl_grouping_chunk = None
        t.tl_pass_num = 1
        t.tl_layer_type = source
        t.tl_source_model_history = self
        t.tl_tensor_shape = tuple(t.shape)
        t.tl_tensor_dtype = t.dtype
        t.tl_tensor_fsize = get_tensor_memory_amount(t)
        t.tl_tensor_fsize_nice = human_readable_size(t.tl_tensor_fsize)

        # Tensor origin info
        t.tl_parent_tensors = []
        t.tl_has_parents = False
        t.tl_orig_ancestors = {tensor_label}
        t.tl_child_tensors = []
        t.tl_has_children = False
        t.tl_parent_tensor_arg_locs = {'args': {}, 'kwargs': {}}
        t.tl_sibling_tensors = []
        t.tl_has_sibling_tensors = False
        t.tl_spouse_tensors = []
        t.tl_has_spouse_tensors = False
        t.tl_is_part_of_iterable_output = False
        t.tl_iterable_output_index = None
        t.tl_is_input_tensor = is_input_tensor
        t.tl_has_input_ancestor = has_input_ancestor
        t.tl_input_ancestors = input_ancestors
        t.tl_min_distance_from_input = None
        t.tl_max_distance_from_input = None
        t.tl_is_output_tensor = False
        t.tl_is_output_ancestor = False
        t.tl_output_descendents = set()
        t.tl_min_distance_from_output = None
        t.tl_max_distance_from_output = None
        t.tl_is_buffer_tensor = is_buffer_tensor
        t.tl_buffer_address = buffer_addr
        t.tl_is_atomic_bool_tensor = False
        t.tl_atomic_bool_val = None
        t.tl_initialized_inside_model = initialized_inside_model
        t.tl_has_internally_initialized_ancestor = has_internally_initialized_ancestor
        t.tl_internally_initialized_parents = []
        t.tl_internally_initialized_ancestors = internally_initialized_ancestors
        t.tl_terminated_inside_model = False
        t.tl_is_terminal_bool_tensor = False
        t.tl_in_cond_branch = False
        t.tl_cond_branch_start_children = []

        # Param info
        t.tl_computed_from_params = False
        t.tl_parent_params = []
        t.tl_parent_param_barcodes = []
        t.tl_parent_param_passes = {}
        t.tl_num_param_tensors = 0
        t.tl_parent_param_shapes = []
        t.tl_num_params_total = 0
        t.tl_parent_params_fsize = 0
        t.tl_parent_params_fsize_nice = human_readable_size(0)

        # Function call info
        t.tl_func_applied = None
        t.tl_func_applied_name = 'none'
        t.tl_func_time_elapsed = 0
        t.tl_func_rng_states = get_rng_states()
        t.tl_num_func_args_total = 0
        t.tl_num_position_args = 0
        t.tl_num_keyword_args = 0
        t.tl_func_position_args_non_tensor = []
        t.tl_func_keyword_args_non_tensor = {}
        t.tl_func_all_args_non_tensor = []
        t.tl_gradfunc = None
        t.tl_gradfunc_name = 'none'

        # Module info

        t.tl_is_computed_inside_submodule = False
        t.tl_containing_module_origin = None
        t.tl_containing_modules_origin_nested = []
        t.tl_containing_module_final = None
        t.tl_containing_modules_final_nested = []
        t.tl_modules_entered = []
        t.tl_module_passes_entered = []
        t.tl_is_submodule_input = False
        t.tl_modules_exited = []
        t.tl_module_passes_exited = []
        t.tl_is_submodule_output = False
        t.tl_is_bottom_level_submodule_output = False
        t.tl_module_entry_exit_thread = []

        self.make_tensor_log_entry(t, t_args=[], t_kwargs={})

    @staticmethod
    def _get_hash_from_untracked_args(args, kwargs):
        """
        Get a hash from the args and kwargs of a function call, excluding any tracked tensors.
        """
        args_to_hash = []
        for arg in list(args) + list(kwargs.values()):
            arg_iter = make_var_iterable(arg)
            for arg_elem in arg_iter:
                if not hasattr(arg, 'tl_tensor_label_raw') and not isinstance(arg_elem, torch.nn.Parameter):
                    args_to_hash.append(arg_elem)

        arg_hash = make_short_barcode_from_input(args_to_hash)
        return arg_hash

    def log_function_output_tensor_func_info(self,
                                             t: torch.Tensor,
                                             args: Tuple,
                                             kwargs: Dict,
                                             func: Callable,
                                             func_name: str,
                                             func_changes_input: bool,
                                             func_time_elapsed: float,
                                             func_rng_states: Dict,
                                             nontensor_args: List,
                                             nontensor_kwargs: Dict,
                                             is_part_of_iterable_output: bool,
                                             iterable_output_index: Optional[int]):
        layer_type = func_name.lower().replace('_', '')
        self.add_raw_label_to_tensor(t, layer_type)

        if func_changes_input:
            grad_fn = t.grad_fn
            grad_fn_name = type(t.grad_fn).__name__
        else:
            grad_fn = identity
            grad_fn_name = 'identity'

        if not is_part_of_iterable_output:
            iterable_output_index = None

        if (t.dtype == torch.bool) and (t.dim()) == 0:
            output_is_single_bool = True
            output_bool_val = t.item()
        else:
            output_is_single_bool = False
            output_bool_val = None

        # General info
        t.tl_layer_type = layer_type
        t.tl_layer_grouping_token = f"{layer_type}_{self._get_hash_from_untracked_args(args, kwargs)}"
        t.tl_grouping_chunk = None
        t.tl_source_model_history = self
        t.tl_tensor_shape = tuple(t.shape)
        t.tl_tensor_dtype = t.dtype
        t.tl_tensor_fsize = get_tensor_memory_amount(t)
        t.tl_tensor_fsize_nice = human_readable_size(t.tl_tensor_fsize)

        # Function call info
        t.tl_func_applied = func
        t.tl_func_applied_name = func_name
        t.tl_func_time_elapsed = func_time_elapsed
        t.tl_func_rng_states = func_rng_states
        t.tl_num_func_args_total = len(args) + len(kwargs)
        t.tl_num_position_args = len(args)
        t.tl_num_keyword_args = len(kwargs)
        t.tl_func_position_args_non_tensor = nontensor_args
        t.tl_func_keyword_args_non_tensor = nontensor_kwargs
        t.tl_func_all_args_non_tensor = nontensor_args + list(nontensor_kwargs.values())
        t.tl_gradfunc = grad_fn
        t.tl_gradfunc_name = grad_fn_name
        t.tl_is_part_of_iterable_output = is_part_of_iterable_output
        t.tl_iterable_output_index = iterable_output_index
        t.tl_is_atomic_bool_tensor = output_is_single_bool
        t.tl_atomic_bool_val = output_bool_val

    def log_function_output_tensor_graph_info(self,
                                              t: torch.Tensor,
                                              parent_tensor_labels: List[str],
                                              parent_tensor_arg_locs: Dict,
                                              input_ancestors: Set[str],
                                              internally_initialized_parents: List[str],
                                              internally_initialized_ancestors: Set[str]):
        """Takes in a tensor that's a function output, marks it in-place with info about its
        connections in the computational graph, and logs it.

        Args:
            t: an input tensor.
            parent_tensor_labels: a list of labels for the parent tensors.
            parent_tensor_arg_locs: a dict mapping parent tensor labels to their argument position
                in the function call.
            input_ancestors: a set of labels for the input ancestors of the tensor
            internally_initialized_parents: a list of labels for the parent tensors that were
                internally initialized.
            internally_initialized_ancestors: a set of labels for the ancestors of the tensor that
                were internally initialized.
        """
        orig_ancestors = input_ancestors.union(internally_initialized_ancestors)
        if len(parent_tensor_labels) > 0:
            has_parents = True
            initialized_inside_model = False
        else:
            has_parents = False
            self.internally_initialized_tensors.append(t.tl_tensor_label_raw)
            initialized_inside_model = True

        if len(input_ancestors) > 0:
            has_input_ancestor = True
        else:
            has_input_ancestor = False

        if len(internally_initialized_ancestors) > 0:
            has_internally_initialized_ancestor = True
        else:
            has_internally_initialized_ancestor = False

        # Tensor origin info
        t.tl_parent_tensors = parent_tensor_labels
        t.tl_parent_tensor_arg_locs = parent_tensor_arg_locs
        t.tl_has_parents = has_parents
        t.tl_orig_ancestors = orig_ancestors
        t.tl_child_tensors = []
        t.tl_has_children = False
        t.tl_sibling_tensors = []
        t.tl_has_sibling_tensors = False
        t.tl_spouse_tensors = []
        t.tl_has_spouse_tensors = False
        t.tl_is_input_tensor = False
        t.tl_has_input_ancestor = has_input_ancestor
        t.tl_input_ancestors = input_ancestors
        t.tl_min_distance_from_input = None
        t.tl_max_distance_from_input = None
        t.tl_is_output_tensor = False
        t.tl_is_output_ancestor = False
        t.tl_output_descendents = set()
        t.tl_min_distance_from_output = None
        t.tl_max_distance_from_output = None
        t.tl_is_buffer_tensor = False
        t.tl_buffer_address = None
        t.tl_initialized_inside_model = initialized_inside_model
        t.tl_has_internally_initialized_ancestor = has_internally_initialized_ancestor
        t.tl_internally_initialized_parents = internally_initialized_parents
        t.tl_internally_initialized_ancestors = internally_initialized_ancestors
        t.tl_terminated_inside_model = False
        t.tl_is_terminal_bool_tensor = False
        t.tl_in_cond_branch = False
        t.tl_cond_branch_start_children = []

        self._update_tensor_family_links(t)

    def log_function_output_tensor_param_info(self,
                                              t: torch.Tensor,
                                              parent_params: List,
                                              parent_param_passes: Dict):
        """Takes in a tensor that's a function output and marks it in-place with parameter info.

        Args:
            t: an input tensor.
            parent_params: list of parameter objects used in the function call
            parent_param_passes: Dict matching param barcodes to how many passes they've had
        """
        layer_type = t.tl_layer_type
        tensor_label = t.tl_tensor_label_raw
        indiv_param_barcodes = list(parent_param_passes.keys())

        if len(parent_param_passes) > 0:
            computed_from_params = True
            layer_label = self._make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
            self.tensors_computed_with_params[layer_label].append(t.tl_tensor_label_raw)
            pass_num = len(self.tensors_computed_with_params[layer_label])
            layer_grouping_token = layer_label[:]
            t.tl_layer_grouping_token = layer_label  # replace with the param label
        else:
            computed_from_params = False
            layer_label = tensor_label
            layer_grouping_token = t.tl_layer_grouping_token  # keep it the same if no params
            pass_num = 1

        # General info
        t.tl_layer_label_raw = layer_label
        t.tl_layer_grouping_token = layer_grouping_token
        t.tl_pass_num = pass_num
        t.tl_computed_from_params = computed_from_params
        t.tl_parent_params = parent_params
        t.tl_parent_param_barcodes = indiv_param_barcodes
        t.tl_parent_param_passes = parent_param_passes
        t.tl_num_param_tensors = len(parent_params)
        t.tl_parent_param_shapes = [tuple(param.shape) for param in parent_params]
        t.tl_num_params_total = np.sum([np.prod(shape) for shape in t.tl_parent_param_shapes])
        t.tl_parent_params_fsize = get_tensor_memory_amount(t)
        t.tl_parent_params_fsize_nice = human_readable_size(t.tl_parent_params_fsize)

    @staticmethod
    def log_function_output_tensor_module_info(t: torch.Tensor,
                                               containing_modules_origin_nested: List[str]):
        """Takes in a tensor that's a function output and marks it in-place with module information.

        Args:
            t: an input tensor.
            containing_modules_origin_nested: a list of module names that the tensor is contained in.
        """
        if len(containing_modules_origin_nested) > 0:
            is_computed_inside_submodule = True
            containing_module_origin = containing_modules_origin_nested[-1]
        else:
            is_computed_inside_submodule = False
            containing_module_origin = None

        t.tl_is_computed_inside_submodule = is_computed_inside_submodule
        t.tl_containing_module_origin = containing_module_origin
        t.tl_containing_modules_origin_nested = containing_modules_origin_nested
        t.tl_modules_entered = []
        t.tl_module_passes_entered = []
        t.tl_is_submodule_input = False
        t.tl_modules_exited = []
        t.tl_module_passes_exited = []
        t.tl_is_submodule_output = False
        t.tl_is_bottom_level_submodule_output = False
        t.tl_module_entry_exit_thread = []

    def log_identity_like_function_output_tensor(self, t: torch.Tensor):
        """Logs a tensor returned by an "identity-like" function that does not change it, but
        that we want to keep track of anyway (e.g., dropout with p = 0; it doesn't
        do anything, but we want to know that it's there).

        Args:
            t:

        Returns:

        """

    def remove_log_entry(self, log_entry: TensorLogEntry):
        """Given a TensorLogEntry, destroys it and all references to it.

        Args:
            log_entry: Tensor log entry to remove.
        """
        tensor_label = log_entry.tensor_label_raw
        for attr in dir(log_entry):
            if not attr.startswith('_') and not callable(getattr(log_entry, attr)):
                delattr(log_entry, attr)
        del log_entry

        # Clear any fields in ModelHistory referring to the entry.
        fields_to_delete = ['input_tensors', 'output_tensors', 'buffer_tensors', 'internally_initialized_tensors',
                            'internally_terminated_tensors', 'internally_terminated_bool_tensors',
                            'tensors_computed_with_params']
        for field in fields_to_delete:
            field_list = getattr(self, field)
            if tensor_label in field_list:
                field_list.remove(tensor_label)

    @staticmethod
    def _make_raw_param_group_barcode(indiv_param_barcodes, layer_type):
        """Given list of param barcodes and layer type, returns the raw barcode for the
        param_group; e.g., conv2d_abcdef_uvwxyz

        Args:
            param_group_list: List of the barcodes for each param in the group.
            layer_type: The layer type.

        Returns:
            Raw barcode for the param group
        """
        param_group_barcode = f"{layer_type}_{'_'.join(sorted(indiv_param_barcodes))}"
        return param_group_barcode

    def _add_sibling_labels_for_new_tensor(self, new_tensor: torch.Tensor, parent_tensor: TensorLogEntry):
        """Given a tensor and specified parent tensor, adds sibling labels to that tensor, and
        adds itself as a sibling to all existing children.

        Args:
            new_tensor: the new tensor
            parent_tensor: the parent tensor
        """
        new_tensor_label = new_tensor.tl_tensor_label_raw
        for sibling_tensor_label in parent_tensor.child_tensors:
            if sibling_tensor_label == new_tensor_label:
                continue
            sibling_tensor = self[sibling_tensor_label]
            sibling_tensor.sibling_tensors.append(new_tensor_label)
            sibling_tensor.has_sibling_tensors = True
            new_tensor.tl_sibling_tensors.append(sibling_tensor_label)
            new_tensor.tl_has_sibling_tensors = True

    def _update_tensor_family_links(self, t):
        """For a given tensor, updates family information for its links to parents, children, siblings, and
        spouses, in both directions (i.e., mutually adding the labels for each family pair).

        Args:
            t: the tensor
        """
        tensor_label = t.tl_tensor_label_raw
        parent_tensor_labels = t.tl_parent_tensors

        # Add the tensor as child to its parents

        for parent_tensor_label in parent_tensor_labels:
            parent_tensor = self[parent_tensor_label]
            if tensor_label not in parent_tensor.child_tensors:
                parent_tensor.child_tensors.append(tensor_label)
                parent_tensor.has_children = True

        # Set the parents of the tensor as spouses to each other

        for spouse1, spouse2 in it.combinations(parent_tensor_labels, 2):
            if spouse1 not in self[spouse2].spouse_tensors:
                self[spouse2].spouse_tensors.append(spouse1)
                self[spouse2].has_spouse_tensors = True
            if spouse2 not in self[spouse1].spouse_tensors:
                self[spouse1].spouse_tensors.append(spouse2)
                self[spouse1].has_spouse_tensors = True

        # Set the children of its parents as siblings to each other.

        for parent_tensor_label in parent_tensor_labels:
            self._add_sibling_labels_for_new_tensor(t, self[parent_tensor_label])

    def _getitem_after_pass(self, ix):
        """Fetches a layer flexibly based on the different lookup options.

        Args:
            ix: A valid index for fetching a layer

        Returns:
            Tensor log entry object with info about specified layer.
        """
        pass

    def _getitem_during_pass(self, ix):
        """Fetches an item when the pass is unfinished, only based on its raw barcode.

        Args:
            ix: layer's barcode

        Returns:
            Tensor log entry object with info about specified layer.
        """
        if ix in self.raw_tensor_dict:
            return self.raw_tensor_dict[ix]
        else:
            raise ValueError(f"{ix} not found in the ModelHistory object.")

    def __getitem__(self, ix):
        """Returns an object logging a model layer given an index. If the pass is finished,
        it'll do this intelligently; if not, it simply queries based on the layer's raw barcode.

        Args:
            ix: desired index

        Returns:
            Tensor log entry object with info about specified layer.
        """
        if self.pass_finished:
            return self._getitem_after_pass(ix)
        else:
            return self._getitem_during_pass(ix)

    def __iter__(self):
        """Loops through all tensors in the log.
        """
        if self.pass_finished:
            return iter(self.tensor_list)
        else:
            return iter(list(self.raw_tensor_dict.values()))

    def __len__(self):
        if self.pass_finished:
            return len(self.tensor_list)
        else:
            return len(self.raw_tensor_dict)

    def _str_after_pass(self) -> str:
        """Readable summary of the model history after the pass is finished.

        Returns:
            String summarizing the model.
        """
        s = f"Log of {self.model_name} forward pass:"
        if self.model_is_branching:
            branch_str = "with branching"
        else:
            branch_str = 'without branching'
        if self.model_is_recurrent:
            s += f"\n\tModel structure: recurrent (at most {self.model_max_recurrent_loops} loops), {branch_str}; " \
                 f"{len(self.module_addresses)} total modules."
        else:
            s += f"\n\tModel structure: purely feedforward, {branch_str}; {len(self.module_addresses)} total modules."
        s += f"\n\t{self.model_num_tensors_total} tensors ({self.model_tensor_fsize_total_nice}) " \
             f"computed in forward pass; {self.model_num_tensors_saved} tensors " \
             f"({self.model_tensor_fsize_saved_nice}) saved."
        s += f"\n\t{self.model_total_param_tensors} parameter operations ({self.model_total_params} params total; " \
             f"{self.model_total_params_fsize_nice})."
        s += f"\n\tRandom seed: {self.random_seed_used}"
        s += f"\n\tTime elapsed: {np.round(self.pass_elapsed_time, 3)}s"

        # Print the module hierarchy.
        s += f"\n\tModule Hierarchy:"
        s += self._module_hierarchy_str()

        # Now print all layers.
        s += f"\n\tLayers:"
        for layer_ind, layer_barcode in enumerate(self.layer_labels):
            pass_num = self.tensor_log[layer_barcode].layer_pass_num
            total_passes = self.tensor_log[layer_barcode].layer_passes_total
            if total_passes > 1:
                pass_str = f" ({pass_num}/{total_passes} passes)"
            else:
                pass_str = ''
            s += f"\n\t\t{layer_ind}: {layer_barcode} {pass_str}"

        return s

    def _str_during_pass(self) -> str:
        """Readable summary of the model history during the pass, as a debugging aid.

        Returns:
            String summarizing the model.
        """
        s = f"Log of {self.model_name} forward pass (pass still ongoing):"
        s += f"\n\tRandom seed: {self.random_seed_used}"
        s += f"\n\tInput tensors: {self.input_tensors}"
        s += f"\n\tOutput tensors: {self.output_tensors}"
        s += f"\n\tInternally initialized tensors: {self.internally_initialized_tensors}"
        s += f"\n\tInternally terminated tensors: {self.internally_terminated_tensors}"
        s += f"\n\tInternally terminated boolean tensors: {self.internally_terminated_bool_tensors}"
        s += f"\n\tBuffer tensors: {self.buffer_tensors}"
        s += f"\n\tRaw layer labels:"
        for layer in self.raw_tensor_labels_list:
            s += f"\n\t\t{layer}"
        return s

    def __str__(self) -> str:
        if self.pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _pretty_print_list_w_line_breaks(lst, indent_chars: str, line_break_every=5) -> str:
        """
        Utility function to pretty print a list with line breaks, adding indent_chars every line.
        """
        pass

    def _get_lookup_help_str(self, layer_label) -> str:
        """Generates a help string to be used in error messages when indexing fails.
        """
        pass

    def _module_hierarchy_str_helper(self, module, level):
        """
        Helper function for _module_hierarchy_str.
        """
        pass

    def _module_hierarchy_str(self) -> str:
        """Helper function for printing the nested module hierarchy.

        Returns:
            String summarizing the model hierarchy.
        """
        pass
