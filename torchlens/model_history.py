# This file is for defining the ModelHistory class that stores the representation of the forward pass.

from collections import OrderedDict, defaultdict
import itertools as it
import numpy as np
import pandas as pd
import torch
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Set

from torchlens.torch_decorate import safe_copy

from torchlens.helper_funcs import barcode_tensors_in_obj, get_marks_from_tensor_list, get_rng_states, \
    get_tensor_memory_amount, get_tensors_in_obj_with_mark, get_vars_of_type_from_obj, make_barcode, \
    mark_tensors_in_obj, tensor_in_obj_has_mark, human_readable_size, identity




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
        self.pass_done = False
        self.linked_tensor = t  # TODO: make sure this is cleared in the postprocessing step.
        self.tensor_contents = None
        self.creation_args = []
        self.creation_kwargs = {}

    def save_tensor_data(self,
                         t: torch.Tensor,
                         t_args: List,
                         t_kwargs: Dict):
        """Saves the tensor data for a given tensor operation.

        Args:
            t: the tensor.
            t_args: tensor positional arguments for the operation
            t_kwargs: tensor keyword arguments for the operation
            t_params: list of param arguments for the operation; these are stored as references, not copies.
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


class ModelHistory:
    def __init__(self,
                 model_name: str,
                 random_seed_used: int,
                 tensor_nums_to_save: Union[List[int], str] = None):
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
        self.tensors_computed_with_params = {}
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

    def postprocess(self):
        """
        After the forward pass, cleans up the log into its final form.
        """
        pass

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
        log_entry = self.raw_tensor_dict[t.tensor_label_raw]
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
            self.input_tensors.append(tensor_label)
        elif source == 'buffer':
            is_input_tensor = False
            has_input_ancestor = False
            is_buffer_tensor = True
            initialized_inside_model = True
            has_internally_initialized_ancestor = True
            internally_initialized_ancestors = {tensor_label}
            input_ancestors = set()
            self.buffer_tensors.append(tensor_label)
            self.internally_initialized_tensors.append(tensor_label)
        else:
            raise ValueError("source must be either 'input' or 'buffer'")

        # General info
        t.tl_layer_label_raw = t.tl_tensor_label_raw
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
        t.tl_is_output_tensor = False
        t.tl_is_buffer_tensor = is_buffer_tensor
        t.tl_buffer_address = buffer_addr
        t.tl_is_atomic_bool_tensor = False
        t.tl_atomic_bool_val = None
        t.tl_initialized_inside_model = initialized_inside_model
        t.tl_has_internally_initialized_ancestor = has_internally_initialized_ancestor
        t.tl_internally_initialized_parents = []
        t.tl_internally_initialized_ancestors = internally_initialized_ancestors

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

        self.add_raw_label_to_tensor(t, layer_type)

        # General info
        t.tl_layer_type = layer_type
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
        t.tl_is_output_tensor = False
        t.tl_is_buffer_tensor = False
        t.tl_buffer_address = None
        t.tl_initialized_inside_model = initialized_inside_model
        t.tl_has_internally_initialized_ancestor = has_internally_initialized_ancestor
        t.tl_internally_initialized_parents = internally_initialized_parents
        t.tl_internally_initialized_ancestors = internally_initialized_ancestors

        self._update_tensor_family_links(t)

    def log_function_output_tensor_param_info(self,
                                              t: torch.Tensor,
                                              parent_params: List,
                                              parent_param_passes: Dict):
        """Takes in a tensor that's a function output and marks it in-place with parameter info.

        Args:
            t: an input tensor.
        """
        layer_type = t.tl_layer_type
        tensor_label = t.tl_tensor_label_raw
        indiv_param_barcodes = list(parent_param_passes.keys())

        if len(parent_param_passes) > 0:
            computed_from_params = True
            layer_label = self._make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
            self.tensors_computed_with_params[layer_label].append(t.tl_tensor_label_raw)
            pass_num = len(self.tensors_computed_with_params[layer_label])
        else:
            computed_from_params = False
            layer_label = tensor_label
            pass_num = 1

        # General info
        t.tl_layer_label_raw = layer_label
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
    def log_function_output_tensor_module_info(self,
                                               t: torch.Tensor,
                                               containing_modules_origin_nested: List[str]):
        """Takes in a tensor that's a function output and marks it in-place with module information.

        Args:
            t: an input tensor.
            containing_modules_origin_nested: a list of module names that the tensor is contained in.
        """
        if len(containing_modules_origin_nested) > 0:
            is_computed_inside_submodule = True
        else:
            is_computed_inside_submodule = False

        t.tl_is_computed_inside_submodule = is_computed_inside_submodule
        t.tl_containing_module_origin = containing_modules_origin_nested[-1]
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

    def log_model_output_tensor(self, t: torch.Tensor):
        """Takes in a model output tensor, and adds model output nodes.

        Args:
            t: Model output tensor.
        """

    def log_module_entry(self, t: torch.Tensor):
        """Logs a tensor leaving a module.

        Args:
            t: Tensor leaving a module
        """
        pass

    def log_module_exit(self, t: torch.Tensor):
        """Logs a tensor leaving a module.

        Args:
            t: Tensor leaving a module
        """
        pass

    def remove_log_entry(self, log_entry: TensorLogEntry):
        """Given a TensorLogEntry, destroys it and all references to it. #TODO: test whether this cleans up the garbage

        Args:
            log_entry: Tensor log entry to remove.
        """

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
        pass

    def _str_during_pass(self) -> str:
        """Readable summary of the model history during the pass, as a debugging aid.

        Returns:
            String summarizing the model.
        """
        pass

    def __str__(self) -> str:
        if self.pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

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
