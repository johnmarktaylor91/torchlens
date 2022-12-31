# This file is for defining the ModelHistory class that stores the representation of the forward pass.

import copy
import itertools as it
from collections import OrderedDict, defaultdict, namedtuple
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch

from torchlens.helper_funcs import get_rng_states, get_tensor_memory_amount, human_readable_size, identity, \
    make_short_barcode_from_input, make_var_iterable, print_override, safe_copy


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
        self.has_saved_activations = False
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
        self.has_saved_activations = True

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
        self.tensors_where_internal_branches_merge_with_input = []
        self.internally_terminated_tensors = []
        self.internally_terminated_bool_tensors = []
        self.tensors_computed_with_params = defaultdict(list)
        self.equivalent_operations = defaultdict(set)
        self.same_layer_tensors = defaultdict(list)
        self.conditional_branch_edges = []

        # Tracking info
        self.track_tensors = True
        self.current_function_call_barcode = None

        # Info about decorated functions:
        self.decorated_to_orig_funcs_dict = {}

    ###########################################
    ######### User-Facing Functions ###########
    ###########################################

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

    ########################################
    ########## Built-in Methods ############
    ########################################
    def __len__(self):
        if self.pass_finished:
            return len(self.tensor_list)
        else:
            return len(self.raw_tensor_dict)

    def _getitem_after_pass(self, ix) -> TensorLogEntry:
        """Fetches a layer flexibly based on the different lookup options.

        Args:
            ix: A valid index for fetching a layer

        Returns:
            Tensor log entry object with info about specified layer.
        """
        pass

    def _getitem_during_pass(self, ix) -> TensorLogEntry:
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

    def __getitem__(self, ix) -> TensorLogEntry:
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

    #######################################
    ########### Tensor Logging ############
    #######################################

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
            operation_equivalence_type = f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
            self.input_tensors.append(tensor_label)
        elif source == 'buffer':
            is_input_tensor = False
            has_input_ancestor = False
            is_buffer_tensor = True
            initialized_inside_model = True
            has_internally_initialized_ancestor = True
            internally_initialized_ancestors = {tensor_label}
            input_ancestors = set()
            operation_equivalence_type = f"buffer_{buffer_addr}"
            self.buffer_tensors.append(tensor_label)
            self.internally_initialized_tensors.append(tensor_label)
        else:
            raise ValueError("source must be either 'input' or 'buffer'")

        self.equivalent_operations[operation_equivalence_type].add(t.tl_tensor_label_raw)

        # General info
        t.tl_layer_label_raw = t.tl_tensor_label_raw
        t.tl_operation_equivalence_type = operation_equivalence_type
        t.tl_equivalent_operations = self.equivalent_operations[operation_equivalence_type]
        t.tl_pass_num = 1
        t.tl_layer_type = source
        t.tl_same_layer_tensors = []
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

        if len(args_to_hash) == 0:
            return 'no_args'
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
        operation_equivalence_type = f"{layer_type}_{self._get_hash_from_untracked_args(args, kwargs)}"
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
        t.tl_operation_equivalence_type = operation_equivalence_type
        t.tl_same_layer_tensors = []
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

        if has_input_ancestor and not all([self[parent_label].has_input_ancestor]
                                          for parent_label in parent_tensor_labels):
            self.tensors_where_internal_branches_merge_with_input.append(t.tl_tensor_label_raw)

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

    @staticmethod
    def _make_raw_param_group_barcode(indiv_param_barcodes: List[str], layer_type: str):
        """Given list of param barcodes and layer type, returns the raw barcode for the
        param_group; e.g., conv2d_abcdef_uvwxyz

        Args:
            indiv_param_barcodes: List of barcodes for each individual parameter tensor
            layer_type: The layer type.

        Returns:
            Raw barcode for the param group
        """
        param_group_barcode = f"{layer_type}_{'_'.join(sorted(indiv_param_barcodes))}"
        return param_group_barcode

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
            operation_equivalence_type = layer_label[:]
            t.tl_layer_grouping_token = layer_label  # replace with the param label
        else:
            computed_from_params = False
            layer_label = tensor_label
            operation_equivalence_type = t.tl_operation_equivalence_type  # keep it the same if no params
            pass_num = 1

        self.equivalent_operations[operation_equivalence_type].add(t.tl_tensor_label_raw)

        # General info
        t.tl_layer_label_raw = layer_label
        t.tl_operation_equivalence_type = operation_equivalence_type
        t.tl_equivalent_operations = self.equivalent_operations[operation_equivalence_type]
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

        # Now that parameter stuff done, can tag the

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

    def _remove_log_entry_references(self, tensor_label: TensorLogEntry):
        """Removes all references to a given TensorLogEntry in the ModelHistory object.

        Args:
            tensor_label: The log entry to remove.
        """
        # Clear any fields in ModelHistory referring to the entry.
        fields_to_delete = ['input_tensors', 'output_tensors', 'buffer_tensors', 'internally_initialized_tensors',
                            'internally_terminated_tensors', 'internally_terminated_bool_tensors',
                            'tensors_where_internal_branches_merge_with_input']
        for field in fields_to_delete:
            field_list = getattr(self, field)
            if tensor_label in field_list:
                field_list.remove(tensor_label)

        # Now handle the nested fields:
        nested_fields_to_delete = ['tensor_computed_with_params', 'equivalent_operations']

        for nested_field in nested_fields_to_delete:
            for group_label, group_tensors in getattr(self, nested_field).items():
                if tensor_label in group_tensors:
                    group_tensors.remove(tensor_label)
                if len(group_tensors) == 0:
                    del getattr(self, nested_field)[group_label]

    def remove_log_entry(self,
                         log_entry: TensorLogEntry,
                         remove_references: bool = True):
        """Given a TensorLogEntry, destroys it and all references to it.

        Args:
            log_entry: Tensor log entry to remove.
            remove_references: Whether to also remove references to the log entry
        """
        tensor_label = log_entry.tensor_label_raw
        for attr in dir(log_entry):
            if not attr.startswith('_') and not callable(getattr(log_entry, attr)):
                delattr(log_entry, attr)
        del log_entry
        if remove_references:
            self._remove_log_entry_references(tensor_label)

    ########################################
    ########### Post-Processing ############
    ########################################

    def _add_output_nodes(self):
        """
        Adds dedicated output nodes to the graph.
        """
        new_output_tensors = []
        for i, output_tensor_label in enumerate(self.output_tensors):
            output_node = self[output_tensor_label]
            new_output_node = output_node.copy()
            new_output_node.layer_type = 'output'
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
                self.remove_log_entry(tensor_entry, remove_references=True)
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

    @staticmethod
    def _group_isomorphic_nodes_to_same_layers(iso_node_groups: Dict[str, List],
                                               subgraphs_dict: Dict,
                                               node_to_subgraph_dict: Dict,
                                               adjacent_subgraphs: Dict) -> Dict:
        same_layer_node_groups = defaultdict(set)  # dict of nodes assigned to the same layer
        node_to_layer_group_dict = {}  # reverse mapping: each node to its equivalent layer group

        for iso_group_label, iso_nodes_orig in iso_node_groups.items():
            iso_nodes = sorted(iso_nodes_orig)
            for node1_label, node2_label in it.combinations(iso_nodes, 2):
                node1_subgraph_label = node_to_subgraph_dict[node1_label]
                node2_subgraph_label = node_to_subgraph_dict[node2_label]
                node1_subgraph = subgraphs_dict[node1_subgraph_label]
                node2_subgraph = subgraphs_dict[node2_subgraph_label]
                both_subgraphs_have_params = all([node1_subgraph.has_param_node, node2_subgraph.has_param_node])
                subgraphs_are_adjacent = (node1_subgraph_label in adjacent_subgraphs and
                                          node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label])
                if both_subgraphs_have_params or subgraphs_are_adjacent:
                    earlier_node_label = sorted([node1_label, node2_label])[0]  # layer label always the first node
                    if earlier_node_label in node_to_layer_group_dict:
                        layer_group = node_to_layer_group_dict[earlier_node_label]
                    else:
                        layer_group = earlier_node_label
                    same_layer_node_groups[layer_group].update({node1_label, node2_label})
                    node_to_layer_group_dict[node1_label] = layer_group
                    node_to_layer_group_dict[node2_label] = layer_group

        return same_layer_node_groups

    def _assign_and_log_isomorphic_nodes_to_same_layers(self,
                                                        iso_node_groups: Dict[str, List],
                                                        subgraphs_dict: Dict,
                                                        node_to_subgraph_dict: Dict,
                                                        adjacent_subgraphs: Dict):
        """After extending the subgraphs to maximum size and identifying adjacent subgraphs,
        goes through and labels the layers as corresponding to each other. The rule is that nodes will be
        labeled as corresponding if 1) they are isomorphic with respect to the starting node, and
        2) the subgraphs either contain a param node, or are adjacent.

        Args:
            iso_node_groups: Dict specifying list of isomorphic nodes in each group
            subgraphs_dict: Dict containing entries with information for each subgraph
            node_to_subgraph_dict: Dict mapping each node to the subgraph its in.
            adjacent_subgraphs: Dict mapping each subgraph to set of adjacent subgraphs.
        """
        # Go through each set of isomorphic nodes, and further partition them into nodes assigned to same layer:
        same_layer_node_groups = self._group_isomorphic_nodes_to_same_layers(iso_node_groups,
                                                                             subgraphs_dict,
                                                                             node_to_subgraph_dict,
                                                                             adjacent_subgraphs)

        # Finally, label the nodes corresponding to the same layer.
        for layer_label, layer_nodes in same_layer_node_groups.items():
            layer_nodes = sorted(list(layer_nodes))  # convert to list and sort
            for n, node_label in enumerate(layer_nodes):
                node = self[node_label]
                node.layer_label_raw = layer_label
                node.same_layer_tensors = layer_nodes
                node.pass_num = n + 1
                node.layer_num_passes = len(layer_nodes)

    @staticmethod
    def _update_adjacent_subgraphs(node_subgraph_label: str,
                                   neighbor_subgraph_label: str,
                                   adjacent_subgraphs: Dict[str, set]):
        """Helper function that updates the adjacency status of two subgraphs

        Args:
            node_subgraph_label: Label of the first subgraph
            neighbor_subgraph_label: Label of the second subgraph
            adjacent_subgraphs: Dict mapping each subgraph to set of subgraphs its adjacent to
        """
        if (node_subgraph_label in adjacent_subgraphs) and (
                neighbor_subgraph_label in adjacent_subgraphs):
            return
        elif (node_subgraph_label in adjacent_subgraphs) and (
                neighbor_subgraph_label not in adjacent_subgraphs):
            adjacent_subgraphs[node_subgraph_label].add(neighbor_subgraph_label)
            adjacent_subgraphs[neighbor_subgraph_label] = adjacent_subgraphs[node_subgraph_label]
        elif (node_subgraph_label not in adjacent_subgraphs) and (
                neighbor_subgraph_label in adjacent_subgraphs):
            adjacent_subgraphs[neighbor_subgraph_label].add(node_subgraph_label)
            adjacent_subgraphs[node_subgraph_label] = adjacent_subgraphs[neighbor_subgraph_label]
        else:
            new_adj_set = {node_subgraph_label, neighbor_subgraph_label}
            adjacent_subgraphs[neighbor_subgraph_label] = new_adj_set
            adjacent_subgraphs[node_subgraph_label] = new_adj_set

    def _log_collisions_and_get_candidate_next_nodes(self,
                                                     current_iso_nodes: List[str],
                                                     node_to_subgraph_dict: Dict,
                                                     adjacent_subgraphs: Dict[str, set],
                                                     is_first_node: bool) -> Dict:
        """Helper function that checks all parent and children nodes for overlap with nodes already added
        to subgraphs (either the same subgraph or another one), logs any adjacency among subgraphs,
        and returns a dict with the candidate successor nodes from each subgraph.

        Returns:
            Dict with the candidate next nodes for each subgraph.
        """
        node_type_fields = {'children': 'child_tensors',
                            'parents': 'parent_tensors'}
        if is_first_node:
            node_types_to_use = ['children']
        else:
            node_types_to_use = ['children', 'parents']

        successor_nodes_dict = OrderedDict()
        for node_label in current_iso_nodes:
            node = self[node_label]
            node_subgraph = node_to_subgraph_dict[node_label]
            node_subgraph_label = node_subgraph.starting_node
            subgraph_successor_nodes = {'children': [], 'parents': []}
            for node_type in node_types_to_use:
                node_type_field = node_type_fields[node_type]
                for neighbor_label in getattr(node, node_type_field):
                    neighbor_subgraph = node_to_subgraph_dict[neighbor_label]
                    neighbor_subgraph_label = neighbor_subgraph.starting_node
                    if neighbor_label in node_subgraph.node_set:  # skip if backtracking own subgraph
                        continue
                    elif neighbor_label in node_to_subgraph_dict:  # if hit another subgraph, mark them adjacent & skip
                        self._update_adjacent_subgraphs(node_subgraph_label,
                                                        neighbor_subgraph_label,
                                                        adjacent_subgraphs)
                    else:  # we have a new, non-overlapping node as a possible candiate, add it:
                        subgraph_successor_nodes[node_type].append(neighbor_label)
            successor_nodes_dict[node_subgraph_label] = subgraph_successor_nodes

        return successor_nodes_dict

    @staticmethod
    def _get_next_candidate_node(successor_nodes_dict: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Helper function to grab the next candidate node to consider out of the possible successor nodes.

        Args:
            successor_nodes_dict: Dict of successor nodes from the set of subgraphs being considered

        Returns:

        """
        for subgraph_label, neighbor_type in it.product(successor_nodes_dict, ['children', 'parents']):
            subgraph_neighbors = successor_nodes_dict[subgraph_label][neighbor_type]
            if len(subgraph_neighbors) > 0:
                candidate_node_label = subgraph_neighbors.pop(0)
                candidate_node_neighbor_type = neighbor_type
                candidate_node_subgraph = subgraph_label
                return candidate_node_label, candidate_node_neighbor_type, candidate_node_subgraph
        return None, None, None

    def _get_nodes_isomorphic_to_candidate_node(self,
                                                candidate_node_label: str,
                                                candidate_node_neighbor_type: str,
                                                candidate_node_subgraph: str,
                                                successor_nodes_dict: Dict) -> List[Tuple[str, str]]:
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
        candidate_node_operation_equivalence_type = candidate_node.operation_equivalence_type
        new_equivalent_nodes = [candidate_node_label]
        for subgraph_label in successor_nodes_dict:
            if subgraph_label == candidate_node_subgraph:  # ignore same subgraph
                continue
            other_subgraph_nodes = successor_nodes_dict[subgraph_label][candidate_node_neighbor_type]
            for c, comparison_node_label in enumerate(other_subgraph_nodes):
                comparison_node = self[comparison_node_label]
                if comparison_node.operation_equivalence_type == candidate_node_operation_equivalence_type:
                    new_equivalent_nodes.append((other_subgraph_nodes.pop(c), subgraph_label))
                    break  # only add one node per subgraph at most
        new_equivalent_nodes = sorted(new_equivalent_nodes, key=lambda x: x[0])
        return new_equivalent_nodes

    def _log_new_isomorphic_nodes(self,
                                  new_isomorphic_nodes: List[Tuple[str, str]],
                                  iso_node_groups: Dict[str, List[str]],
                                  node_to_iso_group_dict: Dict[str, str],
                                  subgraphs_dict: Dict,
                                  node_to_subgraph_dict: Dict,
                                  node_stack: List[List[str]]):
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
                subgraphs_dict[node_subgraph].node_set.add(node_label)
                if (not subgraphs_dict[node_subgraph].has_param_node) and (node.computed_with_params):
                    subgraphs_dict[node_subgraph].has_param_node = True
                node_to_subgraph_dict[node_label] = node_subgraph
            node_stack.append(equivalent_node_labels)

    def _fetch_and_process_next_isomorphic_nodes(self,
                                                 current_iso_nodes: List[str],
                                                 iso_node_groups: Dict[str, List[str]],
                                                 node_to_iso_group_dict: Dict[str, str],
                                                 subgraphs_dict: Dict,
                                                 node_to_subgraph_dict: Dict,
                                                 adjacent_subgraphs: Dict[str, set],
                                                 is_first_node: bool,
                                                 node_stack: List[List[str]]):
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

        successor_nodes_dict = self._log_collisions_and_get_candidate_next_nodes(current_iso_nodes,
                                                                                 node_to_subgraph_dict,
                                                                                 adjacent_subgraphs,
                                                                                 is_first_node)

        # Find sets of isomorphic nodes, process & add to the stack, discard singular nodes, repeat till none left.

        while True:
            # Grab a node and pop it:
            candidate_node_label, candidate_node_neighbor_type, candidate_node_subgraph = \
                self._get_next_candidate_node(successor_nodes_dict)
            if candidate_node_label is None:
                break

            new_equivalent_nodes = self._get_nodes_isomorphic_to_candidate_node(candidate_node_label,
                                                                                candidate_node_neighbor_type,
                                                                                candidate_node_subgraph,
                                                                                successor_nodes_dict)

            # Now log this new set of isomorphic nodes.

            self._log_new_isomorphic_nodes(new_equivalent_nodes,
                                           iso_node_groups,
                                           node_to_iso_group_dict,
                                           subgraphs_dict,
                                           node_to_subgraph_dict,
                                           node_stack)

    def _find_and_mark_same_layer_operations_starting_from_node(self, node: TensorLogEntry):
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
        iso_node_groups = OrderedDict({equivalent_operation_starting_labels[0]: equivalent_operation_starting_labels})

        # Reverse dictionary mapping each node to its isomorphism group
        node_to_iso_group_dict = OrderedDict({label: equivalent_operation_starting_labels[0] for label in
                                              equivalent_operation_starting_labels})

        # Dictionary of information about each subgraph
        subgraphs_dict = {}
        subgraph_info = namedtuple('SubgraphInfo', ['starting_node',
                                                    'has_param_node',
                                                    'node_set'])
        for starting_label in equivalent_operation_starting_labels:
            subgraphs_dict[starting_label] = subgraph_info(starting_node=starting_label,
                                                           has_param_node=node.computed_from_params,
                                                           node_set={starting_label})

        # Dictionary mapping each node to the subgraph it is in
        node_to_subgraph_dict = OrderedDict({label: subgraphs_dict[label] for label in
                                             equivalent_operation_starting_labels})

        # Dict mapping each subgraph to the set of subgraphs it's adjacent to; initialize each to be self-adjacent
        adjacent_subgraphs = {}

        # The stack will be a list of lists, where each latter list is a list of isomorphic nodes.
        # When adding to the stack, only isomorphic nodes will be added.

        node_stack = [equivalent_operation_starting_labels[:]]

        is_first_node = True  # if the first node, don't look at parents
        while node_stack:
            # Pop a set of isomorphic nodes off of the stack, then add and process the next nodes in the stack.
            isomorphic_nodes = sorted(node_stack.pop(0))
            self._fetch_and_process_next_isomorphic_nodes(isomorphic_nodes,
                                                          iso_node_groups,
                                                          node_to_iso_group_dict,
                                                          subgraphs_dict,
                                                          node_to_subgraph_dict,
                                                          adjacent_subgraphs,
                                                          is_first_node,
                                                          node_stack)
            is_first_node = False

        self._assign_and_log_isomorphic_nodes_to_same_layers(iso_node_groups,
                                                             subgraphs_dict,
                                                             node_to_subgraph_dict,
                                                             adjacent_subgraphs)

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
        node_stack = self.input_tensors + self.internally_initialized_tensors
        node_stack = sorted(node_stack, key=lambda x: x.tensor_num)
        operation_equivalence_types_seen = set()
        while len(node_stack) > 0:
            # Grab the earliest node in the stack, add its children in sorted order to the stack in advance.
            node_label = node_stack.pop(0)
            node = self[node_label]
            node_operation_equivalence_type = node.operation_equivalence_type
            node_stack.extend(node.child_tensors)
            node_stack = sorted(node_stack, key=lambda x: x.tensor_num)

            # If we've already checked the nodes of this operation equivalence type as starting nodes, continue:
            if node_operation_equivalence_type in operation_equivalence_types_seen:
                continue
            operation_equivalence_types_seen.add(node_operation_equivalence_type)

            # If no equivalent operations for this node, skip it; it's the only operation for this "layer"
            if len(node.equivalent_operations) == 1:
                node.same_layer_tensors = [node_label]
                continue

            # If we've already found the same-layer tensors for this node, and it equals the number of
            # equivalent operations, skip it; the work is already done:
            if len(node.equivalent_operations) == len(node.same_layer_tensors):
                continue

            # Else, start from this node and any equivalent operations, and work forward, finding
            # more equivalent operations:
            self._find_and_mark_same_layer_operations_starting_from_node(node)

    @staticmethod
    def _fix_modules_for_single_internal_tensor(starting_node: TensorLogEntry,
                                                node_to_fix: TensorLogEntry,
                                                node_type_to_fix: str,
                                                node_stack: List[str],
                                                nodes_seen: Set[str]):
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
        node_to_fix.containing_modules_origin_nested = starting_node.containing_modules_origin_nested.copy()
        if node_type_to_fix == 'parent':
            thread_modules = node_to_fix.module_entry_exit_thread
        elif node_type_to_fix == 'child':
            thread_modules = starting_node.module_entry_exit_thread

        for enter_or_exit, module_address, module_pass in thread_modules[::-1]:
            module_pass_label = (module_address, module_pass)
            if node_type_to_fix == 'parent':
                if enter_or_exit == '+':
                    node_to_fix.containing_modules_origin_nested.remove(module_pass_label)
                elif enter_or_exit == '-':
                    node_to_fix.containing_modules_origin_nested.append(module_pass_label)
            elif node_type_to_fix == 'child':
                if enter_or_exit == '+':
                    node_to_fix.containing_modules_origin_nested.append(module_pass_label)
                elif enter_or_exit == '-':
                    node_to_fix.containing_modules_origin_nested.remove(module_pass_label)
        node_stack.append(node_to_fix_label)
        nodes_seen.add(node_to_fix_label)

    def _fix_modules_for_internal_tensors(self):
        """
        Since internally initialized tensors don't automatically know what module they're in,
        this function infers this by tracing back from tensors that came from the input.
        """
        # Fetch nodes where internally initialized branches first meet a tensor computed from the input:
        node_stack = self.tensors_where_internal_branches_merge_with_input[:]

        # Now go through the stack and work backwards up the internally initialized branch, fixing the
        # module containment labels as we go.

        nodes_seen = set()
        while len(node_stack) > 0:
            node_label = node_stack.pop()
            node = self[node_label]
            # Propagate modules for any parent nodes:
            for parent_label in node.parent_tensors:
                parent_node = self[parent_label]
                if (not parent_node.has_input_ancestor) and (parent_label not in nodes_seen):
                    self._fix_modules_for_single_internal_tensor(node,
                                                                 parent_node,
                                                                 'parent',
                                                                 node_stack,
                                                                 nodes_seen)

            # And for any internally generated child nodes:
            for child_label in node.child_tensors:
                child_node = self[child_label]
                if any([node.has_input_ancestor, child_node.has_input_ancestor, child_label in nodes_seen]):
                    continue
                self._fix_modules_for_single_internal_tensor(node,
                                                             child_node,
                                                             'child',
                                                             node_stack,
                                                             nodes_seen)

    def _map_raw_tensor_labels_to_final_tensor_labels(self):
        """
        Determines the final label for each tensor, and stores this mapping as a dictionary
        in order to then go through and rename everything in the next preprocessing step.
        """
        raw_to_final_tensor_labels = {}
        raw_to_final_layer_labels = {}
        layer_type_counter = defaultdict(lambda: 1)
        layer_total_counter = 1
        for tensor_log_entry in self:
            layer_type = tensor_log_entry.layer_type
            pass_num = self.pass_num
            if pass_num == 1:
                layer_type_num = layer_type_counter[layer_type]
                layer_total_num = layer_total_counter
                layer_type_counter[layer_type] += 1
                layer_total_counter += 1
            else:  # inherit layer numbers from first pass of the layer
                first_pass_tensor = self[tensor_log_entry.same_layer_tensors[0]]
                layer_type_num = first_pass_tensor.layer_type_num
                layer_total_num = first_pass_tensor.layer_total_num
            tensor_log_entry.layer_type_num = layer_type_num
            tensor_log_entry.layer_total_num = layer_total_num
            tensor_log_entry.layer_label_w_pass = f"{layer_type}_{layer_type_num}_{layer_total_num}:{pass_num}"
            tensor_log_entry.layer_label_no_pass = f"{layer_type}_{layer_type_num}_{layer_total_num}"
            tensor_log_entry.layer_label_w_pass_short = f"{layer_type}_{layer_type_num}:{pass_num}"
            tensor_log_entry.layer_label_no_pass_short = f"{layer_type}_{layer_type_num}"
            if tensor_log_entry.layer_num_passes == 1:
                tensor_log_entry.layer_label = tensor_log_entry.layer_label_no_pass
                tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_no_pass_short
            else:
                tensor_log_entry.layer_label = tensor_log_entry.layer_label_w_pass
                tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_w_pass_short
            raw_to_final_tensor_labels[tensor_log_entry.tensor_label_raw] = tensor_log_entry.layer_label
            raw_to_final_tensor_labels[tensor_log_entry.layer_label_raw] = tensor_log_entry.layer_label
        self.raw_to_final_tensor_labels = raw_to_final_tensor_labels
        self.raw_to_final_layer_labels = raw_to_final_layer_labels

    def _rename_model_history_layer_names(self):
        """Renames all the metadata fields in ModelHistory with the final layer names, replacing the
        realtime debugging names.
        """
        list_fields_to_rename = ['input_tensors', 'output_tensors', 'buffer_tensors', 'internally_initialized_tensors',
                                 'tensors_where_internal_branches_merge_with_input', 'internally_terminated_tensors',
                                 'internally_terminated_bool_tensors']
        for field in list_fields_to_rename:
            tensor_labels = getattr(self, field)
            setattr(self, field, [self.raw_to_final_tensor_labels[tensor_label] for tensor_label in tensor_labels])

        new_param_tensors = {}
        for key, values in self.tensors_computed_with_params:
            new_key = self.raw_to_final_layer_labels[key]
            new_param_tensors[new_key] = [self.raw_to_final_tensor_labels[tensor_label] for tensor_label in values]
        self.tensors_computed_with_params = new_param_tensors

        new_same_layer_tensors = {}
        for key, values in self.same_layer_tensors:
            new_key = self.raw_to_final_layer_labels[key]
            new_same_layer_tensors[new_key] = [self.raw_to_final_tensor_labels[tensor_label] for tensor_label in values]
        self.same_layer_tensors = new_same_layer_tensors

        for t, (child, parent) in enumerate(self.conditional_branch_edges):
            new_child, new_parent = self.raw_to_final_tensor_labels[child], self.raw_to_final_tensor_labels[parent]
            self.conditional_branch_edges[t] = (new_child, new_parent)

    def _log_module_hierarchy_info_for_layer(self,
                                             tensor_entry: TensorLogEntry):
        """
        Logs the module hierarchy information for a single layer.

        Args:
            tensor_entry: Log entry to mark the module hierarchy info for.
        """
        containing_module_pass_label = None
        for m, module_pass_label in enumerate(tensor_entry.containing_modules_origin_nested):
            module_name, module_pass = module_pass_label
            if (m == 0) and (module_pass_label not in self.top_level_module_passes):
                self.top_level_module_passes.append(module_pass_label)
            else:
                if module_pass_label not in self.module_pass_children[containing_module_pass_label]:
                    self.module_pass_children[containing_module_pass_label].append(module_pass_label)
            containing_module_pass_label = module_pass_label
            if (module_name not in self.module_num_passes) or (self.module_num_passes[module_name] < module_pass):
                self.module_num_passes[module_name] = module_pass

    def _replace_layer_names_for_tensor_entry(self, tensor_entry: TensorLogEntry):
        """
        Replaces all layer names in the fields of a TensorLogEntry with their final
        layer names.

        Args:
            tensor_entry: TensorLogEntry to replace layer names for.
        """
        list_fields_to_rename = ['parent_tensors', 'orig_ancestors', 'child_tensors',
                                 'sibling_tensors', 'spouse_tensors', 'input_ancestors',
                                 'output_descendants', 'internally_initialized_parents',
                                 'internally_initialized_ancestors', 'cond_branch_start_children']
        for field in list_fields_to_rename:
            orig_layer_names = getattr(tensor_entry, field)
            field_type = type(field)
            new_layer_names = field_type([self.raw_to_final_layer_labels[raw_name] for raw_name in orig_layer_names])
            setattr(tensor_entry, field, new_layer_names)

        # Fix the arg locations field:
        for arg_type in ['args', 'kwargs']:
            for key, value in tensor_entry.parent_tensor_arg_locs[arg_type].items():
                tensor_entry.parent_tensor_arg_locs[key] = self.raw_to_final_layer_labels[value]

    def _finalize_tensor_entry_fields(self,
                                      tensor_entry: TensorLogEntry):
        """
        Removes any fields in a TensorLogEntry that were useful in real time, but are not needed by
        the user after the pass is done.
        """
        fields_to_remove = []
        for field in fields_to_remove:
            delattr(tensor_entry, field)

    def _finalize_model_history_fields(self):
        """
        Removes any fields in ModelHistory that were useful in real time, but are not needed by
        the user after the pass is done.
        """
        fields_to_remove = []
        for field in fields_to_remove:
            delattr(self, field)

    def _add_lookup_keys_for_tensor_entry(self,
                                          tensor_entry: TensorLogEntry,
                                          tensor_index: int,
                                          num_tensors_to_keep: int):
        """Adds the user-facing lookup keys for a TensorLogEntry, both to itself
        and to the ModelHistory top-level record.

        Args:
            tensor_entry: TensorLogEntry to get the lookup keys for.
        """
        # The "default" keys: including the pass if multiple passes, excluding if one pass.
        lookup_keys_for_tensor = [tensor_entry.layer_label,
                                  tensor_entry.layer_label_short,
                                  tensor_index,
                                  num_tensors_to_keep - tensor_index]

        # If just one pass, also allow indexing by pass label.
        if tensor_entry.layer_num_passes == 1:
            lookup_keys_for_tensor.extend([tensor_entry.layer_label_w_pass,
                                           tensor_entry.layer_label_w_pass_short])

        # Allow indexing by modules exited as well:
        for module_name, pass_num in tensor_entry.module_passes_exited:
            lookup_keys_for_tensor.append(f"{module_name}:{pass_num}")
            if self.module_num_passes[module_name] == 1:
                lookup_keys_for_tensor.append(f"{module_name}")

        # If buffer tensor, allow using buffer address as a key.
        if tensor_entry.is_buffer_tensor:
            lookup_keys_for_tensor.append(tensor_entry.buffer_address)

        lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

        # Log in both the tensor and in the ModelHistory object.
        tensor_entry.lookup_keys = lookup_keys_for_tensor
        for lookup_key in lookup_keys_for_tensor:
            self.user_lookup_keys_to_tensor_num_dict[lookup_key] = tensor_entry.tensor_num
            self.tensor_mapper_dict[lookup_key] = tensor_entry

    def _final_prettify(self, remove_layers_without_saved_activations: bool = True):
        """Goes through all tensor log entries for the final stages of pre-processing to make the
        user-facing version of ModelHistory.
        """
        total_tensor_size = 0
        total_params = 0
        total_param_tensors = 0
        total_param_layers = 0
        total_param_fsize = 0

        unique_layers_seen = set()  # to avoid double-counting params of recurrent layers
        self.user_lookup_keys_to_tensor_num_dict = {}  # map the user-facing keys to the tensor numbers
        self.tensor_mapper_dict = {}
        self.top_level_module_passes = []
        self.module_pass_children = defaultdict(list)
        self.module_num_passes = OrderedDict()

        num_tensors_with_activations = 0

        # Go through all tensors in ModelHistory for final round of bookkeeping:
        for tensor_entry in self:
            # Replace all raw names with final names:
            self._replace_layer_names_for_tensor_entry(tensor_entry)

            # Log the module hierarchy information:

            self._log_module_hierarchy_info_for_layer(tensor_entry)

            # Tally the tensor and parameter size:

            total_tensor_size += tensor_entry.tensor_fsize
            if tensor_entry.layer_label_no_pass not in unique_layers_seen:  # only count params once
                if tensor_entry.computed_from_params:
                    total_param_layers += 1
                total_params += tensor_entry.num_params_total
                total_param_tensors += tensor_entry.num_param_tensors
                total_param_fsize += tensor_entry.param_fsize
            unique_layers_seen.add(tensor_entry.layer_label_no_pass)

            # Replace any stored functions with undecorated versions:

            tensor_entry.func_applied = self.decorated_to_orig_funcs_dict[tensor_entry.func_applied]

            # Replace field names with the final fields

            # Tally if the tensor has saved activations:
            num_tensors_with_activations += int(tensor_entry.has_saved_activations)

        # And one more pass to delete unused layers from the record and do final tidying up:
        i = 0
        for tensor_entry in self:
            # Determine valid lookup keys and relate them to the tensor's realtime operation number:
            if tensor_entry.has_saved_activations or remove_layers_without_saved_activations:
                self._add_lookup_keys_for_tensor_entry(tensor_entry, i, num_tensors_with_activations)
                self._finalize_tensor_entry_fields(tensor_entry)
                i += 1
            else:
                self.remove_log_entry(tensor_entry, remove_references=False)

        # Add the fields to ModelHistory:
        self.total_tensor_fsize = total_tensor_size
        self.total_tensor_fsize_nice = human_readable_size(total_tensor_size)
        self.total_param_layers = total_param_layers
        self.total_param_tensors = total_param_tensors
        self.total_params = total_params
        self.total_param_fsize = total_param_fsize
        self.total_param_fsize_nice = human_readable_size(total_param_fsize)

        # Rename the raw tensor entries in the fields of ModelHistory:
        self._rename_model_history_layer_names()
        self._finalize_model_history_fields()
        torch.cuda.empty_cache()

    def _set_pass_finished(self):
        """Sets the ModelHistory to "pass finished" status, indicating that the pass is done, so
        the "final" rather than "realtime debugging" mode of certain functions should be used.
        """
        for tensor in self:
            tensor.pass_finished = True
        self.pass_finished = True

    def postprocess(self, only_keep_layers_with_saved_activations: bool = True):
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

        self._assign_corresponding_tensors_to_same_layer()

        # Step 6: Annotate the containing modules for all internally-generated tensors (they don't know where
        # they are when they're made; have to trace breadcrumbs from tensors that came from input).

        self._fix_modules_for_internal_tensors()

        # Step 7: Go down tensor list, get the mapping from raw tensor names to final tensor names.

        self._map_raw_tensor_labels_to_final_tensor_labels()

        # Step 8: Process ModelHistory into its final user-facing version: undecorate all tensors,
        # do any tallying/totals/labeling, log the module hierarchy, rename all tensors,
        # get the operation numbers for all layer labels.

        self._final_prettify(only_keep_layers_with_saved_activations)

        # Step 9: log the pass as finished, changing the ModelHistory behavior to its user-facing version.

        self._set_pass_finished()

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

    ########################################
    ########### Visualization ##############
    ########################################

    ########################################
    ############# Validation ###############
    ########################################
