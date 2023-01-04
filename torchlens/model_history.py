# This file is for defining the ModelHistory class that stores the representation of the forward pass.

import copy
import itertools as it
import random
import time
from collections import OrderedDict, defaultdict, namedtuple
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torchlens.constants import MODEL_HISTORY_FIELD_ORDER, TENSOR_LOG_ENTRY_FIELD_ORDER
from torchlens.helper_funcs import get_attr_values_from_tensor_list, get_tensor_memory_amount, \
    get_vars_of_type_from_obj, human_readable_size, identity, log_current_rng_states, make_random_barcode, \
    make_short_barcode_from_input, make_var_iterable, print_override, remove_entry_from_list, safe_copy


class TensorLogEntry:
    def __init__(self, fields_dict: Dict):
        """Object that stores information about a single tensor operation in the forward pass,
        including metadata and the tensor itself (if specified). Initialized by passing in a dictionary with
        values for all fields.

        Args:
            fields_dict: Dict with values for all fields in TensorLogEntry.
        """
        # Note: this all has to be tediously initialized instead of a for-loop in order for
        # autocomplete features to work well. But, this also serves as a reference for all attributes
        # of a tensor log entry.

        # Check that fields_dict contains all fields for TensorLogEntry:
        field_order_set = set(TENSOR_LOG_ENTRY_FIELD_ORDER)
        fields_dict_key_set = set(fields_dict.keys())
        if fields_dict_key_set != field_order_set:
            error_str = "Error initializing TensorLogEntry:"
            missing_fields = field_order_set - fields_dict_key_set
            extra_fields = fields_dict_key_set - field_order_set
            if len(missing_fields) > 0:
                error_str += f"\n\t- Missing fields {', '.join(missing_fields)}"
            if len(extra_fields) > 0:
                error_str += f"\n\t- Extra fields {', '.join(extra_fields)}"
            raise ValueError(error_str)

        # General info:
        self.tensor_label_raw = fields_dict['tensor_label_raw']
        self.layer_label_raw = fields_dict['layer_label_raw']
        self.operation_num = fields_dict['operation_num']
        self.realtime_tensor_num = fields_dict['realtime_tensor_num']
        self.source_model_history = fields_dict['source_model_history']
        self.pass_finished = fields_dict['pass_finished']

        # Label info:
        self.layer_label = fields_dict['layer_label']
        self.layer_label_short = fields_dict['layer_label_short']
        self.layer_label_w_pass = fields_dict['layer_label_w_pass']
        self.layer_label_w_pass_short = fields_dict['layer_label_w_pass_short']
        self.layer_label_no_pass = fields_dict['layer_label_no_pass']
        self.layer_label_no_pass_short = fields_dict['layer_label_no_pass_short']
        self.layer_type = fields_dict['layer_type']
        self.layer_type_num = fields_dict['layer_type_num']
        self.layer_total_num = fields_dict['layer_total_num']
        self.pass_num = fields_dict['pass_num']
        self.layer_passes_total = fields_dict['layer_passes_total']
        self.lookup_keys = fields_dict['lookup_keys']

        # Saved tensor info:
        self.tensor_contents = fields_dict['tensor_contents']
        self.has_saved_activations = fields_dict['has_saved_activations']
        self.creation_args = fields_dict['creation_args']
        self.creation_kwargs = fields_dict['creation_kwargs']
        self.tensor_shape = fields_dict['tensor_shape']
        self.tensor_dtype = fields_dict['tensor_dtype']
        self.tensor_fsize = fields_dict['tensor_fsize']
        self.tensor_fsize_nice = fields_dict['tensor_fsize_nice']

        # Function call info:
        self.func_applied = fields_dict['func_applied']
        self.func_applied_name = fields_dict['func_applied_name']
        self.func_time_elapsed = fields_dict['func_time_elapsed']
        self.func_rng_states = fields_dict['func_rng_states']
        self.num_func_args_total = fields_dict['num_func_args_total']
        self.num_position_args = fields_dict['num_position_args']
        self.num_keyword_args = fields_dict['num_keyword_args']
        self.func_position_args_non_tensor = fields_dict['func_position_args_non_tensor']
        self.func_keyword_args_non_tensor = fields_dict['func_keyword_args_non_tensor']
        self.func_all_args_non_tensor = fields_dict['func_all_args_non_tensor']
        self.function_is_inplace = fields_dict['function_is_inplace']
        self.gradfunc = fields_dict['gradfunc']
        self.gradfunc_name = fields_dict['gradfunc_name']
        self.is_part_of_iterable_output = fields_dict['is_part_of_iterable_output']
        self.iterable_output_index = fields_dict['iterable_output_index']

        # Param info:
        self.computed_with_params = fields_dict['computed_with_params']
        self.parent_params = fields_dict['parent_params']
        self.parent_param_barcodes = fields_dict['parent_param_barcodes']
        self.parent_param_passes = fields_dict['parent_param_passes']
        self.num_param_tensors = fields_dict['num_param_tensors']
        self.parent_param_shapes = fields_dict['parent_param_shapes']
        self.num_params_total = fields_dict['num_params_total']
        self.parent_params_fsize = fields_dict['parent_params_fsize']
        self.parent_params_fsize_nice = fields_dict['parent_params_fsize_nice']

        # Corresponding layer info:
        self.operation_equivalence_type = fields_dict['operation_equivalence_type']
        self.equivalent_operations = fields_dict['equivalent_operations']
        self.same_layer_tensors = fields_dict['same_layer_tensors']

        # Graph info:
        self.parent_layers = fields_dict['parent_layers']
        self.has_parents = fields_dict['has_parents']
        self.parent_layer_arg_locs = fields_dict['parent_layer_arg_locs']
        self.orig_ancestors = fields_dict['orig_ancestors']
        self.child_layers = fields_dict['child_layers']
        self.has_children = fields_dict['has_children']
        self.sibling_layers = fields_dict['sibling_layers']
        self.has_siblings = fields_dict['has_siblings']
        self.spouse_layers = fields_dict['spouse_layers']
        self.has_spouses = fields_dict['has_spouses']
        self.is_input_layer = fields_dict['is_input_layer']
        self.has_input_ancestor = fields_dict['has_input_ancestor']
        self.input_ancestors = fields_dict['input_ancestors']
        self.min_distance_from_input = fields_dict['min_distance_from_input']
        self.max_distance_from_input = fields_dict['max_distance_from_input']
        self.is_output_layer = fields_dict['is_output_layer']
        self.is_output_ancestor = fields_dict['is_output_ancestor']
        self.output_descendents = fields_dict['output_descendents']
        self.min_distance_from_output = fields_dict['min_distance_from_output']
        self.max_distance_from_output = fields_dict['max_distance_from_output']
        self.is_buffer_layer = fields_dict['is_buffer_layer']
        self.buffer_address = fields_dict['buffer_address']
        self.initialized_inside_model = fields_dict['initialized_inside_model']
        self.has_internally_initialized_ancestor = fields_dict['has_internally_initialized_ancestor']
        self.internally_initialized_parents = fields_dict['internally_initialized_parents']
        self.internally_initialized_ancestors = fields_dict['internally_initialized_ancestors']
        self.terminated_inside_model = fields_dict['terminated_inside_model']

        # Conditional info
        self.is_terminal_bool_layer = fields_dict['is_terminal_bool_layer']
        self.is_atomic_bool_layer = fields_dict['is_atomic_bool_layer']
        self.atomic_bool_val = fields_dict['atomic_bool_val']
        self.in_cond_branch = fields_dict['in_cond_branch']
        self.cond_branch_start_children = fields_dict['cond_branch_start_children']

        # Module info
        self.is_computed_inside_submodule = fields_dict['is_computed_inside_submodule']
        self.containing_module_origin = fields_dict['containing_module_origin']
        self.containing_modules_origin_nested = fields_dict['containing_modules_origin_nested']
        self.containing_module_final = fields_dict['containing_module_final']
        self.containing_modules_final_nested = fields_dict['containing_modules_final_nested']
        self.modules_entered = fields_dict['modules_entered']
        self.module_passes_entered = fields_dict['module_passes_entered']
        self.is_submodule_input = fields_dict['is_submodule_input']
        self.modules_exited = fields_dict['modules_exited']
        self.module_passes_exited = fields_dict['module_passes_exited']
        self.is_submodule_output = fields_dict['is_submodule_output']
        self.is_bottom_level_submodule_output = fields_dict['is_bottom_level_submodule_output']
        self.bottom_level_submodule_exited = fields_dict['bottom_level_submodule_exited']
        self.module_entry_exit_thread = fields_dict['module_entry_exit_thread']

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    def print_all_fields(self):
        """Print all data fields in the layer.
        """
        fields_to_exclude = ['source_model_history', 'func_rng_states']

        for field in dir(self):
            attr = getattr(self, field)
            if not any([field.startswith('_'), field in fields_to_exclude, callable(attr)]):
                print(f"{field}: {attr}")

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

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
                creation_args.append(copy.deepcopy(arg))

        creation_kwargs = {}
        for key, value in t_kwargs.items():
            if issubclass(type(value), (torch.Tensor, torch.nn.Parameter)):
                creation_kwargs[key] = safe_copy(value)
            else:
                creation_kwargs[key] = copy.deepcopy(value)

        self.creation_args = creation_args
        self.creation_kwargs = creation_kwargs

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def get_child_layers(self):
        return [self.source_model_history[child_label] for child_label in self.child_layers]

    def get_parent_layers(self):
        return [self.source_model_history[parent_label] for parent_label in self.parent_layers]

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self):
        if self.pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def _str_during_pass(self):
        s = f"Tensor {self.tensor_label_raw} (layer {self.layer_label_raw}) (PASS NOT FINISHED):"
        s += f"\n\tPass: {self.pass_num}"
        s += f"\n\tTensor info: shape {self.tensor_shape}, dtype {self.tensor_dtype}"
        s += f"\n\tComputed from params: {self.computed_with_params}"
        s += f"\n\tComputed in modules: {self.containing_modules_origin_nested}"
        s += f"\n\tOutput of modules: {self.module_passes_exited}"
        if self.is_bottom_level_submodule_output:
            s += f" (bottom-level submodule output)"
        else:
            s += f" (not bottom-level submodule output)"
        s += f"\n\tFamily info:"
        s += f"\n\t\tParents: {self.parent_layers}"
        s += f"\n\t\tChildren: {self.child_layers}"
        s += f"\n\t\tSpouses: {self.spouse_layers}"
        s += f"\n\t\tSiblings: {self.sibling_layers}"
        s += f"\n\t\tOriginal Ancestors: {self.orig_ancestors} " \
             f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internally_initialized_ancestors}"
        s += f"\n\t\tOutput Descendents: {self.output_descendents} " \
             f"(min dist {self.min_distance_from_output} nodes, max dist {self.max_distance_from_output} nodes)"
        if self.tensor_contents is not None:
            s += f"\n\tTensor contents: \n{print_override(self.tensor_contents, '__str__')}"
        return s

    def _str_after_pass(self):
        if self.layer_passes_total > 1:
            pass_str = f" (pass {self.pass_num}/{self.layer_passes_total}), "
        else:
            pass_str = ", "
        s = f"Layer {self.layer_label_no_pass}" \
            f"{pass_str}operation {self.operation_num + 1}/" \
            f"{self.source_model_history.model_num_tensors_total}:"
        s += f"\n\tOutput tensor: shape={self.tensor_shape}, dype={self.tensor_dtype}, size={self.tensor_fsize_nice}"
        if self.tensor_contents != 'none':
            if len(self.tensor_shape) == 0:
                tensor_slice = self.tensor_contents
            elif len(self.tensor_shape) == 1:
                num_dims = min(5, self.tensor_shape[0])
                tensor_slice = self.tensor_contents[0:num_dims]
            elif len(self.tensor_shape) == 2:
                num_dims = min([5, self.tensor_shape[-2], self.tensor_shape[-1]])
                tensor_slice = self.tensor_contents[0:num_dims, 0:num_dims]
            else:
                num_dims = min([5, self.tensor_shape[-2], self.tensor_shape[-1]])
                tensor_slice = self.tensor_contents.data.clone()
                for _ in range(len(self.tensor_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims]
            tensor_slice = tensor_slice.detach()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if max(self.tensor_shape) > 5:
                s += '...'
        if not self.has_input_ancestor:
            s += f"\n\t(tensor was created de novo inside the model, not computed from input)"
        if not self.is_output_ancestor:
            s += f"\n\t(tensor is not an ancestor of the model output; it terminates within the model)"
        if len(self.parent_param_shapes) > 0:
            params_shapes_str = ', '.join(str(param_shape) for param_shape in self.parent_param_shapes)
            s += f"\n\tParams: Computed from params with shape {params_shapes_str}; " \
                 f"{self.num_params_total} params total ({self.parent_params_fsize_nice})"
        else:
            s += f"\n\tParams: no params used"
        if len(self.parent_layers) > 0:
            parent_layers_str = ', '.join(self.parent_layers)
        else:
            parent_layers_str = "no parent layers"
        s += f"\n\tParent Layers: {parent_layers_str}"
        if len(self.child_layers) > 0:
            child_layers_str = ', '.join(self.child_layers)
        else:
            child_layers_str = "no child layers"
        s += f"\n\tChild Layers: {child_layers_str}"
        if self.containing_module_origin is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.containing_module_origin}"
        if not self.is_input_layer:
            s += f"\n\tFunction: {self.func_applied_name} (gradfunc={self.gradfunc_name}) " \
                 f"{module_str}"
            s += f"\n\tTime elapsed: {self.func_time_elapsed: .3E}s"
        if len(self.modules_exited) > 0:
            modules_exited_str = ', '.join(self.modules_exited)
            s += f"\n\tOutput of modules: {modules_exited_str}"
        else:
            s += f"\n\tOutput of modules: none"
        if self.is_bottom_level_submodule_output:
            s += f"\n\tOutput of bottom-level module: {self.bottom_level_submodule_exited}"
        lookup_keys_str = ', '.join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def __repr__(self):
        return self.__str__()


class RolledTensorLogEntry:
    def __init__(self, source_entry: TensorLogEntry):
        """Stripped-down version TensorLogEntry that only encodes the information needed to plot the model
        in its rolled-up form.

        Args:
            source_entry: The source TensorLogEntry from which the rolled node is constructed
        """
        print(source_entry)

    def __str__(self) -> str:
        s = ''
        for field in dir(self):
            if not field.startswith('_'):
                s += f"{field}: {getattr(self, field)}"
        return s

    def __repr__(self):
        return self.__str__()


class ModelHistory:
    # Visualization constants:
    INPUT_COLOR = "#98FB98"
    OUTPUT_COLOR = "#ff9999"
    PARAMS_NODE_BG_COLOR = "#E6E6E6"
    BUFFER_NODE_COLOR = "#888888"
    DEFAULT_BG_COLOR = 'white'
    CONNECTING_NODE_LINE_COLOR = 'black'
    NONCONNECTING_NODE_LINE_COLOR = '#A0A0A0'
    BOOL_NODE_COLOR = '#F7D460'
    MAX_MODULE_PENWIDTH = 5
    MIN_MODULE_PENWIDTH = 2
    PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH
    COMMUTE_FUNCS = ['add', 'mul', 'cat', 'eq', 'ne']

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
        self.track_tensors = False
        self.current_function_call_barcode = None
        self.random_seed_used = random_seed_used

        # Model structure info
        self.model_is_recurrent = False
        self.model_max_recurrent_loops = 0
        self.model_has_conditional_branching = False
        self.model_is_branching = False

        # Tensor Tracking:
        self.layer_list = []
        self.layer_dict_main_keys = {}
        self.layer_dict_all_keys = {}
        self.layer_labels = []
        self.layer_labels_w_pass = []
        self.layer_labels_no_pass = []
        self.layer_num_passes = OrderedDict()
        self.raw_tensor_dict = OrderedDict()
        self.raw_tensor_labels_list = []
        self.tensor_nums_to_save = tensor_nums_to_save
        self.tensor_counter = 1
        self.raw_layer_type_counter = defaultdict(lambda: 1)
        self.unsaved_layers_lookup_keys = set()

        # Mapping from raw to final layer labels:
        self.raw_to_final_tensor_labels = {}
        self.raw_to_final_layer_labels = {}
        self.lookup_keys_to_tensor_num_dict = {}

        # Special Layers:
        self.input_layers = []
        self.output_layers = []
        self.buffer_layers = []
        self.internally_initialized_layers = []
        self.layers_where_internal_branches_merge_with_input = []
        self.internally_terminated_layers = []
        self.internally_terminated_bool_layers = []
        self.conditional_branch_edges = []
        self.layers_computed_with_params = defaultdict(list)
        self.equivalent_operations = defaultdict(set)
        self.same_layer_tensors = defaultdict(list)

        # Tensor info:
        self.num_tensors_total = 0
        self.tensor_fsize_total = 0
        self.tensor_fsize_total_nice = human_readable_size(0)
        self.num_tensors_saved = 0
        self.tensor_fsize_saved = 0
        self.tensor_fsize_saved_nice = human_readable_size(0)

        # Param info:
        self.total_param_tensors = 0
        self.total_param_layers = 0
        self.total_params = 0
        self.total_params_fsize = 0
        self.total_params_fsize_nice = human_readable_size(0)

        # Info about decorated functions:
        self.decorated_to_orig_funcs_dict = {}

        # Module info:
        self.module_addresses = []
        self.module_passes = []
        self.module_num_passes = OrderedDict()
        self.top_level_module_passes = []
        self.module_pass_children = OrderedDict()

        # Time elapsed:
        self.pass_start_time = 0
        self.pass_end_time = 0
        self.elapsed_time_total = 0
        self.elapsed_time_function_calls = 0
        self.elapsed_time_torchlens_logging = 0

    # ********************************************
    # ********** User-Facing Functions ***********
    # ********************************************

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

    # ********************************************
    # ************ Tensor Logging ****************
    # ********************************************

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
        layer_type = source
        # Fetch counters and increment to be ready for next tensor to be logged
        layer_type_num = self.raw_layer_type_counter[layer_type]
        realtime_tensor_num = self.tensor_counter
        self.raw_layer_type_counter[layer_type] += 1
        self.tensor_counter += 1

        tensor_label = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

        if source == 'input':
            is_input_layer = True
            has_input_ancestor = True
            is_buffer_layer = False
            initialized_inside_model = False
            has_internally_initialized_ancestor = False
            input_ancestors = {tensor_label}
            internally_initialized_ancestors = set()
            operation_equivalence_type = f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
        elif source == 'buffer':
            is_input_layer = False
            has_input_ancestor = False
            is_buffer_layer = True
            initialized_inside_model = True
            has_internally_initialized_ancestor = True
            internally_initialized_ancestors = {tensor_label}
            input_ancestors = set()
            operation_equivalence_type = f"buffer_{buffer_addr}"
        else:
            raise ValueError("source must be either 'input' or 'buffer'")

        fields_dict = {
            # General info:
            'tensor_label_raw': tensor_label,
            'layer_label_raw': tensor_label,
            'realtime_tensor_num': realtime_tensor_num,
            'operation_num': None,
            'source_model_history': self,
            'pass_finished': False,

            # Label Info:
            'layer_label': None,
            'layer_label_short': None,
            'layer_label_w_pass': None,
            'layer_label_w_pass_short': None,
            'layer_label_no_pass': None,
            'layer_label_no_pass_short': None,
            'layer_type': layer_type,
            'layer_type_num': layer_type_num,
            'layer_total_num': None,
            'pass_num': 1,
            'layer_passes_total': 1,
            'lookup_keys': [],

            # Saved tensor info:
            'tensor_contents': None,
            'has_saved_activations': False,
            'creation_args': [],
            'creation_kwargs': {},
            'tensor_shape': tuple(t.shape),
            'tensor_dtype': t.dtype,
            'tensor_fsize': get_tensor_memory_amount(t),
            'tensor_fsize_nice': human_readable_size(get_tensor_memory_amount(t)),

            # Function call info:
            'func_applied': None,
            'func_applied_name': 'none',
            'func_time_elapsed': 0,
            'func_rng_states': log_current_rng_states(),
            'num_func_args_total': 0,
            'num_position_args': 0,
            'num_keyword_args': 0,
            'func_position_args_non_tensor': [],
            'func_keyword_args_non_tensor': {},
            'func_all_args_non_tensor': [],
            'function_is_inplace': False,
            'gradfunc': None,
            'gradfunc_name': 'none',
            'is_part_of_iterable_output': False,
            'iterable_output_index': None,

            # Param info:
            'computed_with_params': False,
            'parent_params': [],
            'parent_param_barcodes': [],
            'parent_param_passes': {},
            'num_param_tensors': 0,
            'parent_param_shapes': [],
            'num_params_total': int(0),
            'parent_params_fsize': 0,
            'parent_params_fsize_nice': human_readable_size(0),

            # Corresponding layer info:
            'operation_equivalence_type': operation_equivalence_type,
            'equivalent_operations': self.equivalent_operations[operation_equivalence_type],
            'same_layer_tensors': [],

            # Graph info:
            'parent_layers': [],
            'has_parents': False,
            'parent_layer_arg_locs': {'args': {}, 'kwargs': {}},
            'orig_ancestors': {tensor_label},
            'child_layers': [],
            'has_children': False,
            'sibling_layers': [],
            'has_siblings': False,
            'spouse_layers': [],
            'has_spouses': False,
            'is_input_layer': is_input_layer,
            'has_input_ancestor': has_input_ancestor,
            'input_ancestors': input_ancestors,
            'min_distance_from_input': None,
            'max_distance_from_input': None,
            'is_output_layer': False,
            'is_output_ancestor': False,
            'output_descendents': set(),
            'min_distance_from_output': None,
            'max_distance_from_output': None,
            'is_buffer_layer': is_buffer_layer,
            'buffer_address': buffer_addr,
            'initialized_inside_model': initialized_inside_model,
            'has_internally_initialized_ancestor': has_internally_initialized_ancestor,
            'internally_initialized_parents': [],
            'internally_initialized_ancestors': internally_initialized_ancestors,
            'terminated_inside_model': False,

            # Conditional info:
            'is_terminal_bool_layer': False,
            'is_atomic_bool_layer': False,
            'atomic_bool_val': None,
            'in_cond_branch': False,
            'cond_branch_start_children': [],

            # Module info:
            'is_computed_inside_submodule': False,
            'containing_module_origin': None,
            'containing_modules_origin_nested': [],
            'containing_module_final': None,
            'containing_modules_final_nested': [],
            'modules_entered': [],
            'module_passes_entered': [],
            'is_submodule_input': False,
            'modules_exited': [],
            'module_passes_exited': [],
            'is_submodule_output': False,
            'is_bottom_level_submodule_output': False,
            'bottom_level_submodule_exited': None,
            'module_entry_exit_thread': []
        }

        self._make_tensor_log_entry(t, fields_dict, (), {})

        # Tag the tensor itself with its label, and with a reference to the model history log.
        t.tl_tensor_label_raw = tensor_label
        t.tl_source_model_history = self

        # Log info to ModelHistory
        self.equivalent_operations[operation_equivalence_type].add(t.tl_tensor_label_raw)
        if source == 'input':
            self.input_layers.append(tensor_label)
        if source == 'buffer':
            self.buffer_layers.append(tensor_label)
            self.internally_initialized_layers.append(tensor_label)

    def log_function_output_tensors(self,
                                    func: Callable,
                                    args: Tuple[Any],
                                    kwargs: Dict[str, Any],
                                    out_orig: Any,
                                    func_time_elapsed: float,
                                    func_rng_states: Dict,
                                    is_bottom_level_func: bool):
        """Logs tensor or set of tensors that were computed from a function call.

        Args:
            func: function that was called
            args: positional arguments to function that was called
            kwargs: keyword arguments to function that was called
            out_orig: original output from function
            func_time_elapsed: time it took for the function to run
            func_rng_states: states of the random number generator when the function is called
            is_bottom_level_func: whether the function is at the bottom-level of function nesting
        """
        # Unpacking and reformatting:
        func_name = func.__name__
        layer_type = func_name.lower().replace('_', '')
        all_args = list(args) + list(kwargs.values())

        fields_dict = {}  # dict storing the arguments for initializing the new log entry

        non_tensor_args = [arg for arg in args if not issubclass(type(arg), torch.Tensor)]
        non_tensor_kwargs = {key: val for key, val in kwargs.items() if not issubclass(type(val), torch.Tensor)}
        arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.Parameter])
        parent_layer_labels = get_attr_values_from_tensor_list(arg_tensors, 'tl_tensor_label_raw')
        parent_layer_entries = [self[label] for label in parent_layer_labels]

        # General info
        fields_dict['layer_type'] = layer_type

        # Function call info
        fields_dict['func_applied'] = func
        fields_dict['func_applied_name'] = func_name
        fields_dict['func_time_elapsed'] = func_time_elapsed
        fields_dict['func_rng_states'] = func_rng_states
        fields_dict['num_func_args_total'] = len(args) + len(kwargs)
        fields_dict['num_position_args'] = len(args)
        fields_dict['num_keyword_args'] = len(kwargs)
        fields_dict['func_position_args_non_tensor'] = non_tensor_args
        fields_dict['func_keyword_args_non_tensor'] = non_tensor_kwargs
        fields_dict['func_all_args_non_tensor'] = non_tensor_args + list(non_tensor_kwargs.values())

        # Graph info
        parent_layer_arg_locs = self._get_parent_tensor_function_call_location(parent_layer_entries, args, kwargs)
        input_ancestors, internally_initialized_ancestors = self._get_ancestors_from_parents(parent_layer_entries)
        internal_parent_layer_labels = [label for label in parent_layer_labels
                                        if self[label].has_internally_initialized_ancestor]

        fields_dict['parent_layers'] = parent_layer_labels
        fields_dict['parent_layer_arg_locs'] = parent_layer_arg_locs
        fields_dict['has_parents'] = (len(fields_dict['parent_layers']) > 0)
        fields_dict['orig_ancestors'] = input_ancestors.union(internally_initialized_ancestors)
        fields_dict['child_layers'] = []
        fields_dict['has_children'] = False
        fields_dict['sibling_layers'] = []
        fields_dict['has_siblings'] = False
        fields_dict['spouse_layers'] = []
        fields_dict['has_spouses'] = False
        fields_dict['is_input_layer'] = False
        fields_dict['has_input_ancestor'] = (len(input_ancestors) > 0)
        fields_dict['input_ancestors'] = input_ancestors
        fields_dict['min_distance_from_input'] = None
        fields_dict['max_distance_from_input'] = None
        fields_dict['is_output_layer'] = False
        fields_dict['is_output_ancestor'] = False
        fields_dict['output_descendents'] = set()
        fields_dict['min_distance_from_output'] = None
        fields_dict['max_distance_from_output'] = None
        fields_dict['is_buffer_layer'] = False
        fields_dict['buffer_address'] = None
        fields_dict['initialized_inside_model'] = (len(parent_layer_labels) == 0)
        fields_dict['has_internally_initialized_ancestor'] = (len(internally_initialized_ancestors) > 0)
        fields_dict['internally_initialized_parents'] = internal_parent_layer_labels
        fields_dict['internally_initialized_ancestors'] = internally_initialized_ancestors
        fields_dict['terminated_inside_model'] = False
        fields_dict['is_terminal_bool_layer'] = False
        fields_dict['in_cond_branch'] = False
        fields_dict['cond_branch_start_children'] = []

        # Param info
        arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
        parent_param_passes = self._process_parent_param_passes(arg_parameters)
        indiv_param_barcodes = list(parent_param_passes.keys())

        fields_dict['computed_with_params'] = (len(parent_param_passes) > 0)
        fields_dict['parent_params'] = arg_parameters
        fields_dict['parent_param_barcodes'] = indiv_param_barcodes
        fields_dict['parent_param_passes'] = parent_param_passes
        fields_dict['num_param_tensors'] = len(arg_parameters)
        fields_dict['parent_param_shapes'] = [tuple(param.shape) for param in arg_parameters]
        fields_dict['num_params_total'] = np.sum([np.prod(shape) for shape in fields_dict['parent_param_shapes']])
        fields_dict['parent_params_fsize'] = int(np.sum([get_tensor_memory_amount(p) for p in arg_parameters]))
        fields_dict['parent_params_fsize_nice'] = human_readable_size(fields_dict['parent_params_fsize'])

        # Module info

        containing_modules_origin_nested = self._get_input_module_info(arg_tensors)
        if len(containing_modules_origin_nested) > 0:
            is_computed_inside_submodule = True
            containing_module_origin = containing_modules_origin_nested[-1]
        else:
            is_computed_inside_submodule = False
            containing_module_origin = None

        fields_dict['is_computed_inside_submodule'] = is_computed_inside_submodule
        fields_dict['containing_module_origin'] = containing_module_origin
        fields_dict['containing_modules_origin_nested'] = containing_modules_origin_nested
        fields_dict['containing_module_final'] = containing_module_origin
        fields_dict['containing_modules_final_nested'] = containing_modules_origin_nested
        fields_dict['modules_entered'] = []
        fields_dict['module_passes_entered'] = []
        fields_dict['is_submodule_input'] = False
        fields_dict['modules_exited'] = []
        fields_dict['module_passes_exited'] = []
        fields_dict['is_submodule_output'] = False
        fields_dict['is_bottom_level_submodule_output'] = False
        fields_dict['bottom_level_submodule_exited'] = None
        fields_dict['module_entry_exit_thread'] = []

        is_part_of_iterable_output = type(out_orig) in [list, tuple, dict, set]
        fields_dict['is_part_of_iterable_output'] = is_part_of_iterable_output
        out_iter = make_var_iterable(out_orig)  # so we can iterate through it

        for i, out in enumerate(out_iter):
            if not self._output_should_be_logged(out, is_bottom_level_func):
                continue

            self._log_info_specific_to_single_function_output_tensor(out, i, args, kwargs,
                                                                     parent_param_passes, fields_dict)
            self._make_tensor_log_entry(out, fields_dict=fields_dict,
                                        t_args=args, t_kwargs=kwargs)
            new_tensor_entry = self[fields_dict['tensor_label_raw']]
            new_tensor_label = new_tensor_entry.tensor_label_raw
            self._update_tensor_family_links(new_tensor_entry)

            # Update relevant fields of ModelHistory
            # Add layer to relevant fields of ModelHistory:
            if fields_dict['initialized_inside_model']:
                self.internally_initialized_layers.append(new_tensor_label)
            if fields_dict['has_input_ancestor'] and any([(self[parent_layer].has_internally_initialized_ancestor and
                                                           not self[parent_layer].has_input_ancestor)
                                                          for parent_layer in fields_dict['parent_layers']]):
                self.layers_where_internal_branches_merge_with_input.append(new_tensor_label)

            # Tag the tensor itself with its label, and with a reference to the model history log.
            out.tl_tensor_label_raw = fields_dict['tensor_label_raw']
            out.tl_source_model_history = self

    def _output_should_be_logged(self,
                                 out: Any,
                                 is_bottom_level_func: bool) -> bool:
        """Function to check whether to log the output of a function.

        Returns:
            True if the output should be logged, False otherwise.
        """
        if type(out) != torch.Tensor:  # only log if it's a tensor
            return False

        has_same_gradfunc = False
        if hasattr(out, 'tl_tensor_label_raw'):
            old_gradfunc = self[getattr(out, 'tl_tensor_label_raw')].gradfunc
            if out.grad_fn == old_gradfunc:
                has_same_gradfunc = True

        if (not hasattr(out, 'tl_tensor_label_raw')) or (not has_same_gradfunc) or is_bottom_level_func:
            return True
        else:
            return False

    def _log_info_specific_to_single_function_output_tensor(self,
                                                            t: torch.Tensor,
                                                            i: int,
                                                            args: Tuple[Any],
                                                            kwargs: Dict[str, Any],
                                                            parent_param_passes: Dict[str, int],
                                                            fields_dict: Dict[str, Any]):
        """Function to log handle the logging of info that's specific to a single output tensor
        (e.g., the shape), and not common to all output tensors.

        Args:
            t: tensor to log
            i: index of the tensor in the function output
            args: positional args to the function that created the tensor
            kwargs: keyword args to the function that created the tensor
            parent_param_passes: Dict mapping barcodes of parent params to how many passes they've seen
            fields_dict: dictionary of fields with which to initialize the new TensorLogEntry
        """
        layer_type = fields_dict['layer_type']
        indiv_param_barcodes = list(parent_param_passes.keys())
        realtime_tensor_num = self.tensor_counter
        layer_type_num = self.raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

        # Increment the counters to be ready for the next tensor
        self.raw_layer_type_counter[layer_type] += 1
        self.tensor_counter += 1

        if len(parent_param_passes) > 0:
            operation_equivalence_type = self._make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
            fields_dict['operation_equivalence_type'] = operation_equivalence_type
            self.layers_computed_with_params[operation_equivalence_type].append(tensor_label_raw)
            fields_dict['pass_num'] = len(self.layers_computed_with_params[operation_equivalence_type])
        else:
            arg_hash = self._get_hash_from_untracked_args(args, kwargs)
            operation_equivalence_type = f"{layer_type}_{arg_hash}"
            if fields_dict['is_part_of_iterable_output']:
                operation_equivalence_type += f'_outindex{i}'
            fields_dict['operation_equivalence_type'] = operation_equivalence_type
            fields_dict['pass_num'] = 1

        self.equivalent_operations[operation_equivalence_type].add(tensor_label_raw)
        fields_dict['equivalent_operations'] = self.equivalent_operations[operation_equivalence_type]

        fields_dict['function_is_inplace'] = hasattr(t, 'tl_tensor_label_raw')
        if fields_dict['function_is_inplace']:
            old_gradfunc = self[getattr(t, 'tl_tensor_label_raw')].gradfunc
            if t.grad_fn == old_gradfunc:
                fields_dict['gradfunc'] = identity
                fields_dict['gradfunc_name'] = 'identity'
            else:
                fields_dict['gradfunc'] = t.grad_fn
                fields_dict['gradfunc_name'] = type(t.grad_fn).__name__
        else:
            fields_dict['gradfunc'] = t.grad_fn
            fields_dict['gradfunc_name'] = type(t.grad_fn).__name__

        if fields_dict['is_part_of_iterable_output']:
            fields_dict['iterable_output_index'] = i
        else:
            fields_dict['iterable_output_index'] = None

        if (t.dtype == torch.bool) and (t.dim()) == 0:
            fields_dict['is_atomic_bool_layer'] = True
            fields_dict['atomic_bool_val'] = t.item()
        else:
            fields_dict['is_atomic_bool_layer'] = False
            fields_dict['atomic_bool_val'] = None

        # General info
        fields_dict['tensor_label_raw'] = tensor_label_raw
        fields_dict['layer_total_num'] = None
        fields_dict['same_layer_tensors'] = []
        fields_dict['realtime_tensor_num'] = realtime_tensor_num
        fields_dict['operation_num'] = None
        fields_dict['source_model_history'] = self
        fields_dict['pass_finished'] = False

        # Other labeling info
        fields_dict['layer_label'] = None
        fields_dict['layer_label_short'] = None
        fields_dict['layer_label_w_pass'] = None
        fields_dict['layer_label_w_pass_short'] = None
        fields_dict['layer_label_no_pass'] = None
        fields_dict['layer_label_no_pass_short'] = None
        fields_dict['layer_type'] = layer_type
        fields_dict['layer_label_raw'] = tensor_label_raw
        fields_dict['layer_type_num'] = layer_type_num
        fields_dict['pass_num'] = 1
        fields_dict['layer_passes_total'] = 1
        fields_dict['lookup_keys'] = []

        # Saved tensor info
        fields_dict['tensor_contents'] = None
        fields_dict['has_saved_activations'] = False
        fields_dict['creation_args'] = []
        fields_dict['creation_kwargs'] = {}
        fields_dict['tensor_shape'] = tuple(t.shape)
        fields_dict['tensor_dtype'] = t.dtype
        fields_dict['tensor_fsize'] = get_tensor_memory_amount(t)
        fields_dict['tensor_fsize_nice'] = human_readable_size(fields_dict['tensor_fsize'])

    def _make_tensor_log_entry(self,
                               t: torch.Tensor,
                               fields_dict: Dict,
                               t_args: Optional[Tuple] = None,
                               t_kwargs: Optional[Dict] = None):
        """
        Given a tensor, adds it to the model_history, additionally saving the activations and input
        arguments if specified. Also tags the tensor itself with its raw tensor label
        and a pointer to ModelHistory.

        Args:
            t: tensor to log
            fields_dict: dictionary of fields to log in TensorLogEntry
            t_args: Positional arguments to the function that created the tensor
            t_kwargs: Keyword arguments to the function that created the tensor
        """
        if t_args is None:
            t_args = []
        if t_kwargs is None:
            t_kwargs = {}

        new_entry = TensorLogEntry(fields_dict)
        if (self.tensor_nums_to_save == 'all') or (new_entry.realtime_tensor_num in self.tensor_nums_to_save):
            new_entry.save_tensor_data(t, t_args, t_kwargs)
        self.raw_tensor_dict[new_entry.tensor_label_raw] = new_entry
        self.raw_tensor_labels_list.append(new_entry.tensor_label_raw)

        return new_entry

    def _get_parent_tensor_function_call_location(self,
                                                  parent_log_entries: List[TensorLogEntry],
                                                  args: Tuple[Any],
                                                  kwargs: Dict[Any, Any]) -> Dict:
        """Utility function that takes in the parent tensors, the args, and kwargs, and returns a dict specifying
        where in the function call the parent tensors were used.

        Args:
            parent_log_entries: List of parent tensors.
            args: Tuple of function args
            kwargs: Dict of function kwargs

        Returns:
            Dict that itself contains two dicts, one specifing which args are associated with parent tensors,
            and another specifing which kwargs are associated with parent tensors.
        """
        tensor_all_arg_positions = {'args': {}, 'kwargs': {}}
        arg_struct_dict = {'args': args, 'kwargs': kwargs}

        for parent_entry in parent_log_entries:
            for arg_type in ['args', 'kwargs']:
                arg_struct = arg_struct_dict[arg_type]
                self._find_arg_positions_for_single_parent(parent_entry,
                                                           arg_type,
                                                           arg_struct,
                                                           tensor_all_arg_positions)

        return tensor_all_arg_positions

    @staticmethod
    def _find_arg_positions_for_single_parent(parent_entry: TensorLogEntry,
                                              arg_type: str,
                                              arg_struct: Union[List, Tuple, Dict],
                                              tensor_all_arg_positions: Dict):
        """Helper function that finds where a single parent tensor is used in either the args or kwargs of a function,
        and updates a dict that tracks this information.

        Args:
            parent_entry: Parent tensor
            arg_type: 'args' or 'kwargs'
            arg_struct: args or kwargs
            tensor_all_arg_positions: dict tracking where the tensors are used
        """
        iterfunc_dict = {'args': enumerate,
                         'kwargs': lambda x: x.items(),
                         list: enumerate,
                         tuple: enumerate,
                         dict: lambda x: x.items()}
        iterfunc = iterfunc_dict[arg_type]

        for arg_key, arg in iterfunc(arg_struct):
            if getattr(arg, 'tensor_label_raw', -1) == parent_entry.tensor_label_raw:
                tensor_all_arg_positions[arg_type][arg_key] = parent_entry.tensor_label_raw
            elif type(arg) in [list, tuple, dict]:
                iterfunc2 = iterfunc_dict[type(arg)]
                for sub_arg_key, sub_arg in iterfunc2(arg):
                    if getattr(sub_arg, 'tensor_label_raw', -1) == parent_entry.tensor_label_raw:
                        tensor_all_arg_positions[arg_type][(arg_key, sub_arg_key)] = parent_entry.tensor_label_raw

    @staticmethod
    def _get_ancestors_from_parents(parent_entries: List[TensorLogEntry]) -> Tuple[set[str], set[str]]:
        """Utility function to get the ancestors of a tensor based on those of its parent tensors.

        Args:
            parent_entries: list of parent entries

        Returns:
            List of input ancestors and internally initialized ancestors.
        """
        input_ancestors = set()
        internally_initialized_ancestors = set()
        for parent_entry in parent_entries:
            input_ancestors.update(parent_entry.input_ancestors)
            internally_initialized_ancestors.update(parent_entry.internally_initialized_ancestors)
        return input_ancestors, internally_initialized_ancestors

    def _update_tensor_family_links(self, entry_to_update: TensorLogEntry):
        """For a given tensor, updates family information for its links to parents, children, siblings, and
        spouses, in both directions (i.e., mutually adding the labels for each family pair).

        Args:
            entry_to_update: dict of information about the TensorLogEntry to be created
        """
        tensor_label = entry_to_update.tensor_label_raw
        parent_tensor_labels = entry_to_update.parent_layers

        # Add the tensor as child to its parents

        for parent_tensor_label in parent_tensor_labels:
            parent_tensor = self[parent_tensor_label]
            if tensor_label not in parent_tensor.child_layers:
                parent_tensor.child_layers.append(tensor_label)
                parent_tensor.has_children = True

        # Set the parents of the tensor as spouses to each other

        for spouse1, spouse2 in it.combinations(parent_tensor_labels, 2):
            if spouse1 not in self[spouse2].spouse_layers:
                self[spouse2].spouse_layers.append(spouse1)
                self[spouse2].has_spouses = True
            if spouse2 not in self[spouse1].spouse_layers:
                self[spouse1].spouse_layers.append(spouse2)
                self[spouse1].has_spouses = True

        # Set the children of its parents as siblings to each other.

        for parent_tensor_label in parent_tensor_labels:
            self._add_sibling_labels_for_new_tensor(entry_to_update, self[parent_tensor_label])

    def _add_sibling_labels_for_new_tensor(self,
                                           entry_to_update: TensorLogEntry,
                                           parent_tensor: TensorLogEntry):
        """Given a tensor and specified parent tensor, adds sibling labels to that tensor, and
        adds itself as a sibling to all existing children.

        Args:
            entry_to_update: the new tensor
            parent_tensor: the parent tensor
        """
        new_tensor_label = entry_to_update.tensor_label_raw
        for sibling_tensor_label in parent_tensor.child_layers:
            if sibling_tensor_label == new_tensor_label:
                continue
            sibling_tensor = self[sibling_tensor_label]
            sibling_tensor.sibling_layers.append(new_tensor_label)
            sibling_tensor.has_sibling_layers = True
            entry_to_update.sibling_layers.append(sibling_tensor_label)
            entry_to_update.has_sibling_layers = True

    @staticmethod
    def _process_parent_param_passes(arg_parameters: List[torch.nn.Parameter]) -> Dict[str, int]:
        """Utility function to mark the parameters with barcodes, and log which pass they're on.

        Args:
            arg_parameters: List of arg parameters

        Returns:

        """
        parent_param_passes = {}
        for param in arg_parameters:
            if not hasattr(param, 'tl_param_barcode'):
                param_barcode = make_random_barcode()
                param.tl_param_barcode = param_barcode
                param.tl_pass_num = 1
            else:
                param_barcode = param.tl_param_barcode
                param.tl_pass_num += 1
            parent_param_passes[param_barcode] = param.tl_pass_num
        return parent_param_passes

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

    @staticmethod
    def _update_tensor_containing_modules(tensor_entry: TensorLogEntry) -> List[str]:
        """Utility function that updates the containing modules of a Tensor by starting from the containing modules
        as of the last function call, then looks at the sequence of module transitions (in or out of a module) as of
        the last module it saw, and updates accordingly.

        Args:
            tensor_entry: Log entry of tensor to check

        Returns:
            List of the updated containing modules.
        """
        containing_modules = tensor_entry.containing_modules_origin_nested[:]
        thread_modules = tensor_entry.module_entry_exit_thread[:]
        for thread_module in thread_modules:
            if thread_module[0] == '+':
                containing_modules.append(thread_module[1:])
            elif (thread_module[0] == '-') and (thread_module[1:] in containing_modules):
                containing_modules.remove(thread_module[1:])
        return containing_modules

    def _get_input_module_info(self,
                               arg_tensors: List[torch.Tensor]):
        """Utility function to extract information about module entry/exit from input tensors.

        Args:
            arg_tensors: List of input tensors

        Returns:
            Variables with module entry/exit information
        """
        max_input_module_nesting = 0
        most_nested_containing_modules = []
        for t in arg_tensors:
            tensor_label = getattr(t, 'tl_tensor_label_raw', -1)
            if tensor_label == -1:
                continue
            tensor_entry = self[tensor_label]
            containing_modules = self._update_tensor_containing_modules(tensor_entry)
            if len(containing_modules) > max_input_module_nesting:
                max_input_module_nesting = len(containing_modules)
                most_nested_containing_modules = containing_modules[:]
        return most_nested_containing_modules

    def _remove_log_entry(self,
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

    def _remove_log_entry_references(self, layer_to_remove: str):
        """Removes all references to a given TensorLogEntry in the ModelHistory object.

        Args:
            layer_to_remove: The log entry to remove.
        """
        # Clear any fields in ModelHistory referring to the entry.

        remove_entry_from_list(self.input_layers, layer_to_remove)
        remove_entry_from_list(self.output_layers, layer_to_remove)
        remove_entry_from_list(self.buffer_layers, layer_to_remove)
        remove_entry_from_list(self.internally_initialized_layers, layer_to_remove)
        remove_entry_from_list(self.internally_terminated_layers, layer_to_remove)
        remove_entry_from_list(self.internally_terminated_bool_layers, layer_to_remove)
        remove_entry_from_list(self.layers_where_internal_branches_merge_with_input, layer_to_remove)

        self.conditional_branch_edges = [tup for tup in self.conditional_branch_edges if layer_to_remove not in tup]

        # Now any nested fields.

        for group_label, group_tensors in self.layers_computed_with_params.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
            if len(group_tensors) == 0:
                del self.layers_computed_with_params[group_label]

        for group_label, group_tensors in self.equivalent_operations.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
            if len(group_tensors) == 0:
                del self.equivalent_operations[group_label]

        for group_label, group_tensors in self.same_layer_tensors.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
            if len(group_tensors) == 0:
                del self.same_layer_tensors[group_label]

    # ********************************************
    # ************* Post-Processing **************
    # ********************************************

    def postprocess(self, keep_layers_without_saved_activations: bool = False):
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

        self._final_prettify(keep_layers_without_saved_activations)

        # Step 9: log the pass as finished, changing the ModelHistory behavior to its user-facing version.

        self._set_pass_finished()

    def _add_output_nodes(self):
        """
        Adds dedicated output nodes to the graph.
        """
        new_output_layers = []
        for i, output_layer_label in enumerate(self.output_layers):
            output_node = self[output_layer_label]
            new_output_node = output_node.copy()
            new_output_node.layer_type = 'output'
            new_output_node.tensor_label_raw = f"output_{i + 1}_{self.tensor_counter}_raw"
            new_output_node.realtime_tensor_num = self.tensor_counter
            self.tensor_counter += 1

            # Fix function information:

            new_output_node.func_applied = None
            new_output_node.func_applied_name = 'none'
            new_output_node.func_time_elapsed = 0
            new_output_node.func_rng_states = log_current_rng_states()
            new_output_node.num_func_args_total = 0
            new_output_node.num_position_args = 0
            new_output_node.num_keyword_args = 0
            new_output_node.func_position_args_non_tensor = []
            new_output_node.func_keyword_args_non_tensor = {}
            new_output_node.func_all_args_non_tensor = []
            new_output_node.gradfunc = None
            new_output_node.gradfunc_name = 'none'

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

            new_output_node.parent_layers = [output_node.tensor_label_raw]
            new_output_node.sibling_layers = []
            new_output_node.has_sibling_tensors = False

            # Change original output node:

            output_node.is_output_layer = False
            output_node.child_layers = [new_output_node.tensor_label_raw]

            self.raw_tensor_dict[new_output_node.tensor_label_raw] = new_output_node
            self.raw_tensor_labels_list.append(new_output_node.tensor_label_raw)

            new_output_layers.append(new_output_node.tensor_label_raw)

        self.output_layers = new_output_layers

    def _remove_orphan_nodes(self):
        """
        Removes nodes that are connected to neither the input nor the output by flooding in both directions
        from the input and output nodes.
        """
        orig_nodes = set(self.raw_tensor_labels_list)
        nodes_seen = set()
        node_stack = self.input_layers + self.output_layers
        while len(node_stack) > 0:
            tensor_label = node_stack.pop()
            nodes_seen.add(tensor_label)
            tensor_entry = self[tensor_label]
            for next_label in tensor_entry.child_layers + tensor_entry.parent_layers:
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
                self._remove_log_entry(tensor_entry, remove_references=True)
        self.raw_tensor_labels_list = new_tensor_list
        self.raw_tensor_dict = new_tensor_dict

    def _mark_input_output_distances(self):
        """
        Traverses the graph forward and backward, marks the minimum and maximum distances of each
        node from the input and output, and removes any orphan nodes.
        """
        self._flood_graph_from_input_or_output_nodes('input')
        self._flood_graph_from_input_or_output_nodes('output')

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
            starting_nodes = self.input_layers[:]
            min_field = 'min_distance_from_input'
            max_field = 'max_distance_from_input'
            direction = 'forwards'
            marker_field = 'has_input_ancestor'
            forward_field = 'child_tensors'
        elif mode == 'output':
            starting_nodes = self.output_layers[:]
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

            if (len(current_node.child_layers) == 0) and (not current_node.is_output_layer):
                self._log_internally_terminated_tensor(current_node_label)

            for next_node_label in getattr(current_node, forward_field):
                node_stack.append((next_node_label, nodes_since_start + 1, traversal_direction))

    def _log_internally_terminated_tensor(self, tensor_label: str):
        tensor_entry = self[tensor_label]
        tensor_entry.terminated_inside_model = True
        if tensor_label not in self.internally_terminated_layers:
            self.internally_terminated_layers.append(tensor_label)
            if tensor_entry.is_atomic_bool_layer and (tensor_label not in self.internally_terminated_bool_layers):
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
        node_stack = sorted(node_stack, key=lambda x: x.realtime_tensor_num)
        operation_equivalence_types_seen = set()
        while len(node_stack) > 0:
            # Grab the earliest node in the stack, add its children in sorted order to the stack in advance.
            node_label = node_stack.pop(0)
            node = self[node_label]
            node_operation_equivalence_type = node.operation_equivalence_type
            node_stack.extend(node.child_layers)
            node_stack = sorted(node_stack, key=lambda x: x.realtime_tensor_num)

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
                                                           has_param_node=node.computed_with_params,
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
        new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
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
        # Remove any collisions to the SAME node:
        node_labels = [node[0] for node in new_equivalent_nodes]
        new_equivalent_nodes = [node for node in new_equivalent_nodes if node_labels.count(node[0]) == 1]
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
                if not subgraphs_dict[node_subgraph].has_param_node and node.computed_with_params:
                    subgraphs_dict[node_subgraph].has_param_node = True
                node_to_subgraph_dict[node_label] = node_subgraph
            node_stack.append(equivalent_node_labels)

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
                node.layer_total_passes = len(layer_nodes)

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

    def _fix_modules_for_internal_tensors(self):
        """
        Since internally initialized tensors don't automatically know what module they're in,
        this function infers this by tracing back from tensors that came from the input.
        """
        # Fetch nodes where internally initialized branches first meet a tensor computed from the input:
        node_stack = self.layers_where_internal_branches_merge_with_input[:]

        # Now go through the stack and work backwards up the internally initialized branch, fixing the
        # module containment labels as we go.

        nodes_seen = set()
        while len(node_stack) > 0:
            node_label = node_stack.pop()
            node = self[node_label]
            # Propagate modules for any parent nodes:
            for parent_label in node.parent_layers:
                parent_node = self[parent_label]
                if (not parent_node.has_input_ancestor) and (parent_label not in nodes_seen):
                    self._fix_modules_for_single_internal_tensor(node,
                                                                 parent_node,
                                                                 'parent',
                                                                 node_stack,
                                                                 nodes_seen)

            # And for any internally generated child nodes:
            for child_label in node.child_layers:
                child_node = self[child_label]
                if any([node.has_input_ancestor, child_node.has_input_ancestor, child_label in nodes_seen]):
                    continue
                self._fix_modules_for_single_internal_tensor(node,
                                                             child_node,
                                                             'child',
                                                             node_stack,
                                                             nodes_seen)

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
        else:
            raise ValueError("node_type_to_fix must be 'parent' or 'child'")

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
            pass_num = tensor_log_entry.pass_num
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
            if tensor_log_entry.layer_total_passes == 1:
                tensor_log_entry.layer_label = tensor_log_entry.layer_label_no_pass
                tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_no_pass_short
            else:
                tensor_log_entry.layer_label = tensor_log_entry.layer_label_w_pass
                tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_w_pass_short
            raw_to_final_tensor_labels[tensor_log_entry.tensor_label_raw] = tensor_log_entry.layer_label
            raw_to_final_tensor_labels[tensor_log_entry.layer_label_raw] = tensor_log_entry.layer_label
        self.raw_to_final_tensor_labels = raw_to_final_tensor_labels
        self.raw_to_final_layer_labels = raw_to_final_layer_labels

    def _final_prettify(self, keep_layers_without_saved_activations: bool = True):
        """
        Goes through all tensor log entries for the final stages of pre-processing to make the
        user-facing version of ModelHistory.
        """
        # Go through and log information pertaining to all layers:
        self._log_final_info_for_all_layers(keep_layers_without_saved_activations)

        # And one more pass to delete unused layers from the record and do final tidying up:
        self._remove_unwanted_entries_and_log_remaining(keep_layers_without_saved_activations)

        # Rename the raw tensor entries in the fields of ModelHistory:
        self._rename_model_history_layer_names()
        self._trim_and_reorder_model_history_fields()

        # Clear the cache after any tensor deletions for garbage collection purposes:
        torch.cuda.empty_cache()

    def _log_final_info_for_all_layers(self, keep_layers_without_saved_activations: bool = True):
        """
        Goes through all layers (before discarding unsaved ones), and logs final info about the model
        and the layers that pertains to all layers (not just saved ones).
        """
        unique_layers_seen = set()  # to avoid double-counting params of recurrent layers
        for t, tensor_entry in enumerate(self):
            tensor_entry.operation_num = t

            # Replace any layer names with their final names:
            self._replace_layer_names_for_tensor_entry(tensor_entry)

            # Log the module hierarchy information:
            self._log_module_hierarchy_info_for_layer(tensor_entry)

            # Tally the tensor sizes:
            self.tensor_fsize_total += tensor_entry.tensor_fsize

            # Tally the parameter sizes:
            if tensor_entry.layer_label_no_pass not in unique_layers_seen:  # only count params once
                if tensor_entry.computed_with_params:
                    self.total_param_layers += 1
                self.total_params += tensor_entry.num_params_total
                self.total_param_tensors += tensor_entry.num_param_tensors
                self.total_params_fsize += tensor_entry.param_fsize
            unique_layers_seen.add(tensor_entry.layer_label_no_pass)

            # Tally elapsed time:

            self.elapsed_time_function_calls += tensor_entry.func_time_elapsed

            # Replace stored function with original undecorated version:
            tensor_entry.func_applied = self.decorated_to_orig_funcs_dict[tensor_entry.func_applied]

            # Update model structural information:
            if len(tensor_entry.child_layers) > 1:
                self.model_is_branching = True
            if tensor_entry.layer_total_passes > self.model_max_recurrent_loops:
                self.model_is_recurrent = True
                self.model_max_recurrent_loops = tensor_entry.layer_total_passes
            if tensor_entry.in_cond_branch:
                self.model_has_conditional_branching = True

        if (self.num_tensors_saved == len(self)) or keep_layers_without_saved_activations:
            self.all_layers_logged = True
        else:
            self.all_layers_logged = False

        # Save the nice versions of the filesize fields:
        self.tensor_fsize_total_nice = human_readable_size(self.tensor_fsize_total)
        self.total_params_fsize_nice = human_readable_size(self.total_params_fsize)

        # Log time elapsed:
        self.pass_end_time = time.time()
        self.elapsed_time_total = self.pass_end_time - self.pass_start_time
        self.elapsed_time_torchlens_logging = self.elapsed_time_total - self.elapsed_time_function_calls

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
            for key, value in tensor_entry.parent_layer_arg_locs[arg_type].items():
                tensor_entry.parent_layer_arg_locs[key] = self.raw_to_final_layer_labels[value]

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
            if module_name not in self.module_addresses:
                self.module_addresses.append(module_name)
            if module_pass_label not in self.module_passes:
                self.module_passes.append(module_pass_label)

    def _remove_unwanted_entries_and_log_remaining(self,
                                                   keep_layers_without_saved_activations: bool = True):
        """Removes entries from ModelHistory that we don't want in the final saved output,
        and logs information about the remaining entries.
        """
        tensors_to_remove = []
        self.unsaved_layers_lookup_keys = set()
        i = 0
        for tensor_entry in self:
            # Determine valid lookup keys and relate them to the tensor's realtime operation number:
            if tensor_entry.has_saved_activations or keep_layers_without_saved_activations:
                # Add the lookup keys for the layer, to itself and to ModelHistory:
                self._add_lookup_keys_for_tensor_entry(tensor_entry, i, self.num_tensors_saved)

                # Log all information:
                self.layer_list.append(tensor_entry)
                self.layer_dict_main_keys[tensor_entry.layer_label] = tensor_entry
                self.layer_labels.append(tensor_entry.layer_label)
                self.layer_labels_no_pass.append(tensor_entry.layer_label_no_pass)
                self.layer_labels_w_pass.append(tensor_entry.layer_label_w_pass)
                self.layer_num_passes[tensor_entry.layer_label] = tensor_entry.layer_total_passes
                if tensor_entry.has_saved_activations:
                    self.num_tensors_saved += 1
                    self.tensor_fsize_saved += tensor_entry.tensor_fsize
                self._trim_and_reorder_tensor_entry_fields(tensor_entry)  # Final reformatting of fields
                i += 1
            else:
                tensors_to_remove.append(tensor_entry)
                self.unsaved_layers_lookup_keys.update(tensor_entry.lookup_keys)

        # Remove unused entries.
        for tensor_entry in tensors_to_remove:
            self._remove_log_entry(tensor_entry, remove_references=False)

        # Make the saved tensor filesize pretty:
        self.tensor_fsize_saved_nice = human_readable_size(self.tensor_fsize_saved)

    def _add_lookup_keys_for_tensor_entry(self,
                                          tensor_entry: TensorLogEntry,
                                          tensor_index: int,
                                          num_tensors_to_keep: int):
        """Adds the user-facing lookup keys for a TensorLogEntry, both to itself
        and to the ModelHistory top-level record.

        Args:
            tensor_entry: TensorLogEntry to get the lookup keys for.
        """
        tensor_entry.index_in_saved_log = tensor_index

        # The "default" keys: including the pass if multiple passes, excluding if one pass.
        lookup_keys_for_tensor = [tensor_entry.layer_label,
                                  tensor_entry.layer_label_short,
                                  tensor_index,
                                  num_tensors_to_keep - tensor_index]

        # If just one pass, also allow indexing by pass label.
        if tensor_entry.layer_passes_total == 1:
            lookup_keys_for_tensor.extend([tensor_entry.layer_label_w_pass,
                                           tensor_entry.layer_label_w_pass_short])

        # Allow indexing by modules exited as well:
        for module_name, pass_num in tensor_entry.module_passes_exited:
            lookup_keys_for_tensor.append(f"{module_name}:{pass_num}")
            if self.module_num_passes[module_name] == 1:
                lookup_keys_for_tensor.append(f"{module_name}")

        # If buffer tensor, allow using buffer address as a key.
        if tensor_entry.is_buffer_layer:
            lookup_keys_for_tensor.append(tensor_entry.buffer_address)

        lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

        # Log in both the tensor and in the ModelHistory object.
        tensor_entry.lookup_keys = lookup_keys_for_tensor
        for lookup_key in lookup_keys_for_tensor:
            self.lookup_keys_to_tensor_num_dict[lookup_key] = tensor_entry.realtime_tensor_num
            self.layer_dict_all_keys[lookup_key] = tensor_entry

    @staticmethod
    def _trim_and_reorder_tensor_entry_fields(tensor_entry: TensorLogEntry):
        """
        Sorts the fields in TensorLogEntry into their desired order, and trims any
        fields that aren't useful after the pass.
        """
        new_dir_dict = OrderedDict()
        for field in TENSOR_LOG_ENTRY_FIELD_ORDER:
            new_dir_dict[field] = getattr(tensor_entry, field)
        for field in dir(tensor_entry):
            if field.startswith('_'):
                new_dir_dict[field] = getattr(tensor_entry, field)
        tensor_entry.__dict__ = new_dir_dict

    def _rename_model_history_layer_names(self):
        """Renames all the metadata fields in ModelHistory with the final layer names, replacing the
        realtime debugging names.
        """
        list_fields_to_rename = ['input_tensors', 'output_tensors', 'buffer_tensors', 'internally_initialized_tensors',
                                 'tensors_where_internal_branches_merge_with_input', 'internally_terminated_tensors',
                                 'internally_terminated_bool_layers']
        for field in list_fields_to_rename:
            tensor_labels = getattr(self, field)
            setattr(self, field, [self.raw_to_final_tensor_labels[tensor_label] for tensor_label in tensor_labels])

        new_param_tensors = {}
        for key, values in self.layers_computed_with_params:
            new_key = self.raw_to_final_layer_labels[key]
            new_param_tensors[new_key] = [self.raw_to_final_tensor_labels[tensor_label] for tensor_label in values]
        self.tensors_computed_with_params = new_param_tensors

        new_equiv_operations_tensors = {}
        for key, values in self.equivalent_operations:
            new_key = self.raw_to_final_layer_labels[key]
            new_equiv_operations_tensors[new_key] = set([self.raw_to_final_tensor_labels[tensor_label]
                                                         for tensor_label in values])
        self.equivalent_operations = new_equiv_operations_tensors

        new_same_layer_tensors = {}
        for key, values in self.same_layer_tensors:
            new_key = self.raw_to_final_layer_labels[key]
            new_same_layer_tensors[new_key] = [self.raw_to_final_tensor_labels[tensor_label] for tensor_label in values]
        self.same_layer_tensors = new_same_layer_tensors

        for t, (child, parent) in enumerate(self.conditional_branch_edges):
            new_child, new_parent = self.raw_to_final_tensor_labels[child], self.raw_to_final_tensor_labels[parent]
            self.conditional_branch_edges[t] = (new_child, new_parent)

    def _trim_and_reorder_model_history_fields(self):
        """
        Sorts the fields in ModelHistory into their desired order, and trims any
        fields that aren't useful after the pass.
        """
        new_dir_dict = OrderedDict()
        for field in MODEL_HISTORY_FIELD_ORDER:
            new_dir_dict[field] = getattr(self, field)
        for field in dir(self):
            if field.startswith('_'):
                new_dir_dict[field] = getattr(self, field)
        self.__dict__ = new_dir_dict

    def _set_pass_finished(self):
        """Sets the ModelHistory to "pass finished" status, indicating that the pass is done, so
        the "final" rather than "realtime debugging" mode of certain functions should be used.
        """
        for tensor in self:
            tensor.pass_finished = True
        self.pass_finished = True

    def get_op_nums_from_user_labels(self, which_layers: List[Union[str, int]]) -> List[int]:
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
        raw_tensor_nums_to_save = set()
        num_layers = len(self.layer_list)
        for layer_key in which_layers:
            if type(layer_key) == int:  # if user specifies ordinal position
                if not -num_layers <= layer_key < num_layers:
                    raise ValueError(f"You specified the {layer_key}th layer, but there are only "
                                     f"{num_layers} layers in the model.")
                raw_tensor_nums_to_save.add(self[layer_key].realtime_tensor_num)
            elif layer_key in self.layer_labels:  # if it's a primary layer key just grab it
                raw_tensor_nums_to_save.add(self[layer_key].realtime_tensor_num)
            elif ':' in layer_key:  # if specific pass given, either add or complain if there aren't that many passes
                label, pass_num = layer_key.split(':')
                if (layer_key in self.layer_labels_w_pass) or (layer_key in self.module_passes):
                    raw_tensor_nums_to_save.add(self[layer_key].realtime_tensor_num)
                elif label in self.layer_labels_no_pass:
                    first_pass_address = f"{label}:1"
                    raise ValueError(f"You specified {label} pass #{pass_num}, but there are only "
                                     f"{self[first_pass_address].layer_passes_total} passes in {label}; "
                                     f"please specify a pass in range 1-{self[first_pass_address].layer_passes_total}.")
                elif label in self.module_addresses:
                    raise ValueError(f"You specified {label} pass #{pass_num}, but there are only "
                                     f"{self.module_num_passes[label]} passes in {label}; "
                                     f"please specify a pass in range 1-{self.module_num_passes[label]}.")
                else:
                    raise ValueError(self._get_lookup_help_str(label))
            elif layer_key in self.layer_labels_no_pass:  # if it's a layer address, add all passes of the layer
                for layer_label_w_pass in self.layer_labels_w_pass:
                    if layer_label_w_pass.startswith(f"{layer_key}:"):
                        raw_tensor_nums_to_save.add(self[layer_label_w_pass].realtime_tensor_num)
            elif layer_key in self.module_addresses:  # if it's a module address, add all passes
                for pass_num in range(1, self.module_num_passes[layer_key] + 1):
                    raw_tensor_nums_to_save.add(self[f"{layer_key}:{pass_num}"].realtime_tensor_num)
            elif type(layer_key) == str:  # as last resort check if any layer labels begin with the provided substring
                found_a_match = False
                for layer in self:
                    if layer_key in layer.layer_label_w_pass:
                        raw_tensor_nums_to_save.add(layer.layer_raw_tensor_num)
                        found_a_match = True
                if not found_a_match:
                    raise ValueError(self._get_lookup_help_str(layer_key))
        raw_tensor_nums_to_save = sorted(list(raw_tensor_nums_to_save))
        return raw_tensor_nums_to_save

    # ********************************************
    # *************** Visualization **************
    # ********************************************

    def render_graph(self):
        """Produces a visualization of the saved graph
        """
        if not self.all_layers_logged:
            raise ValueError("All layers of the network must be logged in order to render the graph; "
                             "either use get_model_structure, or use get_model_activations with all layers saved.")

    # ********************************************
    # *************** Validation *****************
    # ********************************************

    def validate_single_layer(self, layer: TensorLogEntry) -> bool:
        """For a single layer, checks whether computing its value from the saved values of its input
        tensors yields its actually saved value, and whether computing its value from perturbed values of the input
        tensors changes it from the saved value. For the perturbation check, also checks whether the other
        args are all zeros or all ones (e.g., such that changing the input wouldn't matter for addition
        or multiplication respectively); if so, that step is disregarded.

        Args:
            layer: TensorLogEntry for the layer to check.

        Returns:
            True if it passes the tests, False otherwise.
        """
        pass

    def validate_forward_pass(self) -> bool:
        """Starting from inputs and internally generated layers, checks whether plugging the
        saved activations from the input layers into the function for that layer yields the saved
        activaations for that layer, and whether plugging nonense activations also changes
        the saved activations for that layer (unless the other args are "special", e.g. all zeros or all ones),
        and whether following the child operations in this manner eventually yields the output saved activations.
        Fails if a given layer fails this test, or if there is a layer for which the parent layers are not saved.

        Returns:
            True if it passes the tests, False otherwise.
        """
        for layer in tqdm(self, desc="Validating saved layer activations..."):
            if not self.validate_single_layer(layer):
                print(f"Validation failed for layer {layer.layer_label}")
                return False
        return True

    # ********************************************
    # ************ Built-in Methods **************
    # ********************************************
    def __len__(self):
        if self.pass_finished:
            return len(self.layer_list)
        else:
            return len(self.raw_tensor_dict)

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

    def _getitem_after_pass(self, ix):
        """
        Overloaded such that entries can be fetched either by their position in the tensor log, their layer label,
        or their module address.
        #it should say so and tell them which labels are valid.
        """
        if ix in self.layer_dict_all_keys:
            return self.layer_dict_all_keys[ix]
        elif (type(ix) == int) and (ix > len(self.layer_list)):
            raise ValueError(f"You specified the layer with index {ix}, but there are only {len(self.layer_list)} "
                             f"layers; please specify a smaller number.")
        elif ix in self.module_addresses:
            module_num_passes = self.module_num_passes[ix]
            raise ValueError(f"You specified output of module {ix}, but it has {module_num_passes} passes; "
                             f"please specify e.g. {ix}:2 for the second pass of {ix}.")
        elif ix.split(':')[0] in self.module_addresses:
            module, pass_num = ix.split(':')
            module_num_passes = self.module_num_passes[module]
            raise ValueError(f"You specified module {module} pass {pass_num}, but {module} only has "
                             f"{module_num_passes} passes; specify a lower number.")
        elif ix in self.layer_labels_no_pass:
            layer_num_passes = self.layer_num_passes[ix]
            raise ValueError(f"You specified output of layer {ix}, but it has {layer_num_passes} passes; "
                             f"please specify e.g. {ix}:2 for the second pass of {ix}.")
        elif ix.split(':')[0] in self.layer_labels_no_pass:
            layer_label, pass_num = ix.split(':')
            layer_num_passes = self.layer_num_passes[layer_label]
            raise ValueError(f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
                             f"{layer_num_passes} passes. Specify a lower number.")
        else:
            raise ValueError(self._get_lookup_help_str(ix))

    def __iter__(self):
        """Loops through all tensors in the log.
        """
        if self.pass_finished:
            return iter(self.layer_list)
        else:
            return iter(list(self.raw_tensor_dict.values()))

    def __str__(self) -> str:
        if self.pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

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
        s += f"\n\t{self.num_tensors_total} tensors ({self.tensor_fsize_total_nice}) " \
             f"computed in forward pass; {self.num_tensors_saved} tensors " \
             f"({self.tensor_fsize_saved_nice}) saved."
        s += f"\n\t{self.total_param_tensors} parameter operations ({self.total_params} params total; " \
             f"{self.total_params_fsize_nice})."
        s += f"\n\tRandom seed: {self.random_seed_used}"
        s += f"\n\tTime elapsed: {np.round(self.elapsed_time_total, 3)}s"

        # Print the module hierarchy.
        s += f"\n\tModule Hierarchy:"
        s += self._module_hierarchy_str()

        # Now print all layers.
        s += f"\n\tLayers:"
        for layer_ind, layer_barcode in enumerate(self.layer_labels):
            pass_num = self.layer_dict_main_keys[layer_barcode].layer_pass_num
            total_passes = self.layer_dict_main_keys[layer_barcode].layer_passes_total
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
        s += f"\n\tInput tensors: {self.input_layers}"
        s += f"\n\tOutput tensors: {self.output_layers}"
        s += f"\n\tInternally initialized tensors: {self.internally_initialized_layers}"
        s += f"\n\tInternally terminated tensors: {self.internally_terminated_layers}"
        s += f"\n\tInternally terminated boolean tensors: {self.internally_terminated_bool_layers}"
        s += f"\n\tBuffer tensors: {self.buffer_layers}"
        s += f"\n\tRaw layer labels:"
        for layer in self.raw_tensor_labels_list:
            s += f"\n\t\t{layer}"
        return s

    @staticmethod
    def pretty_print_list_w_line_breaks(lst, indent_chars: str, line_break_every=5):
        """
        Utility function to pretty print a list with line breaks, adding indent_chars every line.
        """
        s = f'\n{indent_chars}'
        for i, item in enumerate(lst):
            s += f"{item}"
            if i < len(lst) - 1:
                s += ', '
            if ((i + 1) % line_break_every == 0) and (i < len(lst) - 1):
                s += f'\n{indent_chars}'
        return s

    def _get_lookup_help_str(self, layer_label):
        """Generates a help string to be used in error messages when indexing fails.
        """
        sample_layer1 = random.choice(self.layer_labels_w_pass)
        sample_layer2 = random.choice(self.layer_labels_no_pass)
        if len(self.module_addresses) > 0:
            sample_module1 = random.choice(self.module_addresses)
            sample_module2 = random.choice(self.module_passes)
        else:
            sample_module1 = 'features.3'
            sample_module2 = 'features.3:2'
        module_str = f"(e.g., {sample_module1}, {sample_module2})"
        help_str = (f"Layer {layer_label} not recognized; please specify either \n\t1) an integer giving "
                    f"the ordinal position of the layer, \n\t2) the layer label (e.g., {sample_layer1}, "
                    f"{sample_layer2}), \n\t3) the module address {module_str}"
                    f"\n\t4) A substring of any desired layer labels (e.g., 'pool' will grab all maxpool2d "
                    f"or avgpool2d layers, 'maxpool' with grab all 'maxpool2d' layers, etc.)."
                    f"\n(Label meaning: conv2d_3_4:2 means the second pass of the third convolutional layer,"
                    f"and fourth layer overall in the model.)")
        return help_str

    def _module_hierarchy_str(self):
        """
        Utility function to print the nested module hierarchy.
        """
        s = ''
        for module in self.top_level_module_passes:
            s += f"\n\t\t{module[0]}"
            if len(self.module_pass_children[module]) > 0:
                s += ':'
            s += self._module_hierarchy_str_helper(module, 1)
        return s

    def _module_hierarchy_str_helper(self, module, level):
        """
        Helper function for _module_hierarchy_str.
        """
        s = ''
        any_grandchild_modules = any([len(self.module_pass_children[sub_module]) > 0
                                      for sub_module in self.module_pass_children[module]])
        if any_grandchild_modules or len(self.module_pass_children[module]) == 0:
            for sub_module in self.module_pass_children[module]:
                s += f"\n\t\t{'    ' * level}{sub_module[0]}"
                if len(self.module_pass_children[sub_module]) == 0:
                    s += ':'
                s += self._module_hierarchy_str_helper(sub_module, level + 1)
        else:
            s += self.pretty_print_list_w_line_breaks(
                [module_child[0] for module_child in self.module_pass_children[module]],
                line_break_every=8,
                indent_chars=f"\t\t{'    ' * level}")
        return s

    def __repr__(self):
        return self.__str__()
