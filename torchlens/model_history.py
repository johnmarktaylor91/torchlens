# This file is for defining the ModelHistory class that stores the representation of the forward pass.

from collections import OrderedDict, defaultdict
import pandas as pd
import torch
from typing import Dict, List, Optional, Union

from tensor_tracking import safe_copy

from torchlens.helper_funcs import barcode_tensors_in_obj, get_marks_from_tensor_list, get_rng_states, \
    get_tensor_memory_amount, get_tensors_in_obj_with_mark, get_vars_of_type_from_obj, make_barcode, \
    mark_tensors_in_obj, tensor_in_obj_has_mark, human_readable_size


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

    def _save_tensor_data(self,
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

    def _update_tensor_metadata(self, t: torch.Tensor):
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
        self.internally_generated_tensors = []
        self.internally_terminated_tensors = []
        self.internally_terminated_bool_tensors = []
        self.conditional_branch_edges = []

        # Param info
        self.tensors_computed_with_params = {}

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

    def _postprocess(self):
        """
        After the forward pass, cleans up the log into its final form.
        """
        pass

    def _get_op_nums_from_user_labels(self, which_layers: List[str]) -> List[int]:
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

    def _make_tensor_log_entry(self, t: torch.Tensor,
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
            new_entry._save_tensor_data(t, t_args, t_kwargs)

        self.raw_tensor_dict[new_entry.tensor_label_raw] = new_entry
        self.raw_tensor_labels_list.append(new_entry.tensor_label_raw)

    def _update_tensor_log_entry(self, t: torch.Tensor):
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

    def _log_source_tensor(self,
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
            internally_initialized_ancestors = set()
            self.input_tensors.append(tensor_label)
        elif source == 'buffer':
            is_input_tensor = False
            has_input_ancestor = False
            is_buffer_tensor = True
            initialized_inside_model = True
            has_internally_initialized_ancestor = True
            internally_initialized_ancestors = {tensor_label}
            self.buffer_tensors.append(tensor_label)
            self.internally_generated_tensors.append(tensor_label)
        else:
            raise ValueError("source must be either 'input' or 'buffer'")

        # General info
        t.tl_layer_label_raw = t.tensor_label_raw
        t.tl_layer_type = source
        t.tl_source_model_history = self
        t.tl_tensor_shape = tuple(t.shape)
        t.tl_tensor_dtype = t.dtype
        t.tl_tensor_fsize = get_tensor_memory_amount(t)
        t.tl_tensor_fsize_nice = human_readable_size(t.tl_tensor_fsize)

        # Tensor origin info
        t.tl_parent_tensors = []
        t.tl_has_parents = False
        t.tl_child_tensors = []
        t.tl_has_children = False
        t.tl_parent_tensor_arg_locs = {'args': {}, 'kwargs': {}}
        t.tl_is_part_of_iterable_output = False
        t.tl_iterable_output_position = None
        t.tl_sibling_tensors = []
        t.tl_has_sibling_tensors = False
        t.tl_spouse_tensors = []
        t.tl_has_spouse_tensors = False
        t.tl_is_input_tensor = is_input_tensor
        t.tl_has_input_ancestor = has_input_ancestor
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
        t.tl_num_param_tensors = 0
        t.parent_param_shapes = []
        t.num_params_total = 0
        t.parent_params_fsize = 0
        t.parent_params_fsize_nice = human_readable_size(0)
        t.parent_params = []
        t.parent_param_barcodes = []
        t.parent_param_passes = {}

        # Function call info
        t.func_applied = None
        t.func_applied_name = 'none'
        t.func_time_elapsed = 0
        t.func_rng_states = get_rng_states()
        t.num_func_args_total = 0
        t.func_position_args_non_tensor = []
        t.func_keyword_args_non_tensor = {}
        t.func_all_args_non_tensor = []
        t.gradfunc = None
        t.gradfunc_name = 'none'

        # Module info

        t.is_computed_inside_submodule = False
        t.containing_module_origin = None
        t.containing_modules_origin_nested = []
        t.containing_module_final = None
        t.containing_modules_final_nested = []
        t.modules_entered = []
        t.module_passes_entered = []
        t.is_submodule_input = False
        t.modules_exited = []
        t.module_passes_exited = []
        t.is_submodule_output = False
        t.is_bottom_level_submodule_output = False
        t.module_entry_exit_thread = []

        self._make_tensor_log_entry(t, t_args=[], t_kwargs={})

    def _log_function_output_tensor(self, t: torch.Tensor):
        """Takes in a tensor that's a function output, marks it in-place with relevant information,
        and adds it to the log.

        Args:
            t: an input tensor.
        """
        pass

    def _log_identity_like_function_output_tensor(self, t: torch.Tensor):
        """Logs a tensor returned by an "identity-like" function that does not change it, but
        that we want to keep track of anyway (e.g., dropout with p = 0; it doesn't
        do anything, but we want to know that it's there).

        Args:
            t:

        Returns:

        """

    def _log_model_output_tensor(self, t: torch.Tensor):
        """Takes in a model output tensor, and adds model output nodes.

        Args:
            t: Model output tensor.
        """

    def _log_module_entry(self, t: torch.Tensor):
        """Logs a tensor leaving a module.

        Args:
            t: Tensor leaving a module
        """
        pass

    def _log_module_exit(self, t: torch.Tensor):
        """Logs a tensor leaving a module.

        Args:
            t: Tensor leaving a module
        """
        pass

    def _remove_log_entry(self, log_entry: TensorLogEntry):
        """Given a TensorLogEntry, destroys it and all references to it. #TODO: test whether this cleans up the garbage

        Args:
            log_entry: Tensor log entry to remove.
        """

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
