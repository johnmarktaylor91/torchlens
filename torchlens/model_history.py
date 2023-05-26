# This file is for defining the ModelHistory class that stores the representation of the forward pass.

import copy
import itertools as it
import os
import random
import time
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import graphviz
import numpy as np
import pandas as pd
import torch
from IPython.core.display import display

from torchlens.constants import MODEL_HISTORY_FIELD_ORDER, TENSOR_LOG_ENTRY_FIELD_ORDER
from torchlens.helper_funcs import get_attr_values_from_tensor_list, get_tensor_memory_amount, \
    get_vars_of_type_from_obj, human_readable_size, identity, in_notebook, int_list_to_compact_str, \
    log_current_rng_states, set_rng_from_saved_states, make_short_barcode_from_input, make_var_iterable, \
    print_override, remove_entry_from_list, safe_copy, make_random_barcode, tuple_tolerant_assign


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
        self.detach_saved_tensor = fields_dict['detach_saved_tensor']
        self.creation_args = fields_dict['creation_args']
        self.creation_kwargs = fields_dict['creation_kwargs']
        self.tensor_shape = fields_dict['tensor_shape']
        self.tensor_dtype = fields_dict['tensor_dtype']
        self.tensor_fsize = fields_dict['tensor_fsize']
        self.tensor_fsize_nice = fields_dict['tensor_fsize_nice']

        # Saved gradient info
        self.grad_contents = fields_dict['grad_contents']
        self.save_gradients = fields_dict['save_gradients']
        self.has_saved_grad = fields_dict['has_saved_grad']
        self.grad_shapes = fields_dict['grad_shapes']
        self.grad_dtypes = fields_dict['grad_dtypes']
        self.grad_fsizes = fields_dict['grad_fsizes']
        self.grad_fsizes_nice = fields_dict['grad_fsizes_nice']

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
        self.same_layer_operations = fields_dict['same_layer_operations']

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
        self.is_last_output_layer = fields_dict['is_last_output_layer']
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
        self.modules_entered = fields_dict['modules_entered']
        self.module_passes_entered = fields_dict['module_passes_entered']
        self.is_submodule_input = fields_dict['is_submodule_input']
        self.modules_exited = fields_dict['modules_exited']
        self.module_passes_exited = fields_dict['module_passes_exited']
        self.is_submodule_output = fields_dict['is_submodule_output']
        self.is_bottom_level_submodule_output = fields_dict['is_bottom_level_submodule_output']
        self.bottom_level_submodule_pass_exited = fields_dict['bottom_level_submodule_pass_exited']
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
        fields_dict = {}
        fields_not_to_deepcopy = ['func_applied', 'gradfunc',
                                  'source_model_history', 'func_rng_states',
                                  'creation_args', 'creation_kwargs', 'parent_params', 'tensor_contents']
        for field in TENSOR_LOG_ENTRY_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = copy.deepcopy(getattr(self, field))
            else:
                fields_dict[field] = getattr(self, field)
        copied_entry = TensorLogEntry(fields_dict)
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
        self.tensor_contents = safe_copy(t, self.detach_saved_tensor)
        self.has_saved_activations = True

        # Tensor args and kwargs:
        creation_args = []
        for arg in t_args:
            creation_args.append(safe_copy(arg))

        creation_kwargs = {}
        for key, value in t_kwargs.items():
            creation_kwargs[key] = safe_copy(value)

        self.creation_args = creation_args
        self.creation_kwargs = creation_kwargs

    def log_tensor_grad(self, grad: torch.Tensor):
        """Logs the gradient for a tensor to the log entry

        Args:
            grad: The gradient to save.
        """
        self.grad_contents = grad
        self.has_saved_grad = True
        self.grad_shapes = [g.shape for g in grad]
        self.grad_dtypes = [g.dtype for g in grad]
        self.grad_fsizes = [get_tensor_memory_amount(g) for g in grad]
        self.grad_fsizes_nice = [human_readable_size(get_tensor_memory_amount(g)) for g in grad]

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
            f"{self.source_model_history.num_tensors_total}:"
        s += f"\n\tOutput tensor: shape={self.tensor_shape}, dype={self.tensor_dtype}, size={self.tensor_fsize_nice}"
        s += self._tensor_contents_str_helper()
        s += self._tensor_family_str_helper()
        if len(self.parent_param_shapes) > 0:
            params_shapes_str = ', '.join(str(param_shape) for param_shape in self.parent_param_shapes)
            s += f"\n\tParams: Computed from params with shape {params_shapes_str}; " \
                 f"{self.num_params_total} params total ({self.parent_params_fsize_nice})"
        else:
            s += f"\n\tParams: no params used"
        if self.containing_module_origin is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.containing_module_origin}"
        if not self.is_input_layer:
            s += f"\n\tFunction: {self.func_applied_name} (grad_fn: {self.gradfunc}) " \
                 f"{module_str}"
            s += f"\n\tTime elapsed: {self.func_time_elapsed: .3E}s"
        if len(self.modules_exited) > 0:
            modules_exited_str = ', '.join(self.modules_exited)
            s += f"\n\tOutput of modules: {modules_exited_str}"
        else:
            s += f"\n\tOutput of modules: none"
        if self.is_bottom_level_submodule_output:
            s += f"\n\tOutput of bottom-level module: {self.bottom_level_submodule_pass_exited}"
        lookup_keys_str = ', '.join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents.
        """
        if self.tensor_contents is None:
            return ""
        s = ''
        tensor_size_shown = 8
        if self.tensor_contents != 'none':
            if len(self.tensor_shape) == 0:
                tensor_slice = self.tensor_contents
            elif len(self.tensor_shape) == 1:
                num_dims = min(tensor_size_shown, self.tensor_shape[0])
                tensor_slice = self.tensor_contents[0:num_dims]
            elif len(self.tensor_shape) == 2:
                num_dims = min([tensor_size_shown, self.tensor_shape[-2], self.tensor_shape[-1]])
                tensor_slice = self.tensor_contents[0:num_dims, 0:num_dims]
            else:
                num_dims = min([tensor_size_shown, self.tensor_shape[-2], self.tensor_shape[-1]])
                tensor_slice = self.tensor_contents.data.clone()
                for _ in range(len(self.tensor_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims]
            tensor_slice = tensor_slice.detach()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if max(self.tensor_shape) > tensor_size_shown:
                s += '...'
        return s

    def _tensor_family_str_helper(self) -> str:
        s = '\n\tRelated Layers:'
        if len(self.parent_layers) > 0:
            s += '\n\t\t- parent layers: ' + ', '.join(self.parent_layers)
        else:
            s += "\n\t\t- no parent layers"

        if len(self.child_layers) > 0:
            s += '\n\t\t- child layers: ' + ', '.join(self.child_layers)
        else:
            s += "\n\t\t- no child layers"

        if len(self.sibling_layers) > 0:
            s += '\n\t\t- shares parents with layers: ' + ', '.join(self.sibling_layers)
        else:
            s += "\n\t\t- shares parents with no other layers"

        if len(self.spouse_layers) > 0:
            s += '\n\t\t- shares children with layers: ' + ', '.join(self.spouse_layers)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:  # todo: put the ancestors here
            s += '\n\t\t- descendent of input layers: ' + ', '.join(self.input_ancestors)
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.is_output_ancestor:
            s += '\n\t\t- ancestor of output layers: ' + ', '.join(self.output_descendents)
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

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
        # Label & general info
        self.layer_label = source_entry.layer_label_no_pass
        self.layer_type = source_entry.layer_type
        self.layer_type_num = source_entry.layer_type_num
        self.layer_total_num = source_entry.layer_total_num
        self.layer_passes_total = source_entry.layer_passes_total
        self.source_model_history = source_entry.source_model_history

        # Saved tensor info
        self.tensor_shape = source_entry.tensor_shape
        self.tensor_fsize_nice = source_entry.tensor_fsize_nice

        # Param info:
        self.computed_with_params = source_entry.computed_with_params
        self.parent_param_shapes = source_entry.parent_param_shapes
        self.num_param_tensors = source_entry.num_param_tensors

        # Graph info
        self.is_input_layer = source_entry.is_input_layer
        self.has_input_ancestor = source_entry.has_input_ancestor
        self.is_output_layer = source_entry.is_output_layer
        self.is_last_output_layer = source_entry.is_last_output_layer
        self.is_buffer_layer = source_entry.is_buffer_layer
        self.buffer_address = source_entry.buffer_address
        self.cond_branch_start_children = source_entry.cond_branch_start_children
        self.is_terminal_bool_layer = source_entry.is_terminal_bool_layer
        self.atomic_bool_val = source_entry.atomic_bool_val
        self.child_layers = set()
        self.parent_layers = set()

        # Module info:
        self.containing_modules_origin_nested = source_entry.containing_modules_origin_nested
        self.modules_exited = source_entry.modules_exited
        self.module_passes_exited = source_entry.module_passes_exited
        self.is_bottom_level_submodule_output = False
        self.bottom_level_submodule_passes_exited = set()

        # Fields specific to rolled node to fill in:
        self.edges_vary_across_passes = False
        self.child_layers_per_pass = defaultdict(set)
        self.child_passes_per_layer = defaultdict(list)
        self.parent_layers_per_pass = defaultdict(set)
        self.parent_passes_per_layer = defaultdict(list)

        # Each one will now be a list of layers, since they can vary across passes.
        self.parent_layer_arg_locs = {'args': defaultdict(set),
                                      'kwargs': defaultdict(set)}

    def add_pass_info(self,
                      source_node: TensorLogEntry):
        """Adds information about another pass of the same layer: namely, mark information about what the
        child and parent layers are for each pass.

        Args:
            source_node: Information for the source pass
        """
        # Label the layers for each pass
        child_layer_labels = [self.source_model_history[child].layer_label_no_pass
                              for child in source_node.child_layers]
        self.child_layers.update(child_layer_labels)
        self.child_layers_per_pass[source_node.pass_num].update(child_layer_labels)
        parent_layer_labels = [self.source_model_history[parent].layer_label_no_pass
                               for parent in source_node.parent_layers]
        self.parent_layers.update(parent_layer_labels)
        self.parent_layers_per_pass[source_node.pass_num].update(parent_layer_labels)

        # Label the passes for each layer, and indicate if any layers vary based on the pass.
        for child_layer in source_node.child_layers:
            child_layer_label = self.source_model_history[child_layer].layer_label_no_pass
            self.child_passes_per_layer[child_layer_label].append(source_node.pass_num)
            if self.child_passes_per_layer[child_layer_label] != list(range(1, source_node.pass_num + 1)):
                self.edges_vary_across_passes = True

        for parent_layer in source_node.parent_layers:
            parent_layer_label = self.source_model_history[parent_layer].layer_label_no_pass
            self.parent_passes_per_layer[parent_layer_label].append(source_node.pass_num)
            if self.parent_passes_per_layer[parent_layer_label] != list(range(1, source_node.pass_num + 1)):
                self.edges_vary_across_passes = True

        # Add submodule info:
        if source_node.is_bottom_level_submodule_output:
            self.is_bottom_level_submodule_output = True
            self.bottom_level_submodule_passes_exited.add(source_node.bottom_level_submodule_pass_exited)

        # For the parent arg locations, have a list of layers rather than single layer, since they can
        # vary across passes.

        for arg_type in ['args', 'kwargs']:
            for arg_key, layer_label in source_node.parent_layer_arg_locs[arg_type].items():
                layer_label_no_pass = self.source_model_history[layer_label].layer_label_no_pass
                self.parent_layer_arg_locs[arg_type][arg_key].add(layer_label_no_pass)

    def __str__(self) -> str:
        fields_not_to_print = ['source_model_history']
        s = ''
        for field in dir(self):
            attr = getattr(self, field)
            if not field.startswith('_') and field not in fields_not_to_print and not (callable(attr)):
                s += f"{field}: {attr}\n"
        return s

    def __repr__(self):
        return self.__str__()


class ModelHistory:
    # Visualization constants:
    INPUT_COLOR = "#98FB98"
    OUTPUT_COLOR = "#ff9999"
    PARAMS_NODE_BG_COLOR = "#E6E6E6"
    BUFFER_NODE_COLOR = "#888888"
    GRADIENT_ARROW_COLOR = "#9197F6"
    DEFAULT_BG_COLOR = 'white'
    BOOL_NODE_COLOR = '#F7D460'
    MAX_MODULE_PENWIDTH = 5
    MIN_MODULE_PENWIDTH = 2
    PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH
    COMMUTE_FUNCS = ['add', 'mul', 'cat', 'eq', 'ne']
    FUNCS_NOT_TO_PERTURB_IN_VALIDATION = ['expand_as', 'new_zeros', 'new_ones', 'zero_']

    def __init__(self,
                 model_name: str,
                 random_seed_used: int,
                 tensor_nums_to_save: Union[List[int], str] = 'all',
                 detach_saved_tensors: bool = False,
                 save_gradients: bool = False):
        """Object that stores the history of a model's forward pass.
        Both logs the history in real time, and stores a nice
        representation of the full history for the user afterward.
        """
        # General info
        self.model_name = model_name
        self.pass_finished = False
        self.track_tensors = False
        self.all_layers_logged = False
        if tensor_nums_to_save in ['none', None, []]:
            tensor_nums_to_save = []
            self.keep_layers_without_saved_activations = True
        else:
            self.keep_layers_without_saved_activations = False
        self.current_function_call_barcode = None
        self.random_seed_used = random_seed_used
        self.detach_saved_tensors = detach_saved_tensors
        self.save_gradients = save_gradients
        self.has_saved_gradients = False

        # Model structure info
        self.model_is_recurrent = False
        self.model_max_recurrent_loops = 1
        self.model_has_conditional_branching = False
        self.model_is_branching = False

        # Tensor Tracking:
        self.layer_list = []
        self.layer_list_rolled = []
        self.layer_dict_main_keys = OrderedDict()
        self.layer_dict_all_keys = OrderedDict()
        self.layer_dict_rolled = OrderedDict()
        self.layer_labels = []
        self.layer_labels_w_pass = []
        self.layer_labels_no_pass = []
        self.layer_num_passes = OrderedDict()
        self.raw_tensor_dict = OrderedDict()
        self.raw_tensor_labels_list = []
        self.tensor_nums_to_save = tensor_nums_to_save
        self.tensor_counter = 0
        self.raw_layer_type_counter = defaultdict(lambda: 0)
        self.unsaved_layers_lookup_keys = set()

        # Mapping from raw to final layer labels:
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
        self.layers_with_saved_gradients = []
        self.layers_computed_with_params = defaultdict(list)
        self.equivalent_operations = defaultdict(set)
        self.same_layer_operations = defaultdict(list)

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

        # Module info:
        self.module_addresses = []
        self.module_types = {}
        self.module_passes = []
        self.module_num_passes = defaultdict(lambda: 1)
        self.top_level_module_passes = []
        self.module_pass_children = defaultdict(list)

        # Time elapsed:
        self.pass_start_time = 0
        self.pass_end_time = 0
        self.elapsed_time_total = 0
        self.elapsed_time_function_calls = 0
        self.elapsed_time_torchlens_logging = 0

    # ********************************************
    # ********** User-Facing Functions ***********
    # ********************************************

    def print_all_fields(self):
        """Print all data fields for ModelHistory.
        """
        fields_to_exclude = ['layer_list', 'layer_dict_main_keys', 'layer_dict_all_keys', 'raw_tensor_dict',
                             'decorated_to_orig_funcs_dict']

        for field in dir(self):
            attr = getattr(self, field)
            if not any([field.startswith('_'), field in fields_to_exclude, callable(attr)]):
                print(f"{field}: {attr}")

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
        model_df = pd.DataFrame()
        fields_for_df = ['layer_label',
                         'layer_label_w_pass',
                         'layer_label_no_pass',
                         'layer_label_short',
                         'layer_label_w_pass_short',
                         'layer_label_no_pass_short',
                         'layer_type',
                         'layer_type_num',
                         'layer_total_num',
                         'layer_passes_total',
                         'pass_num',
                         'operation_num',
                         'tensor_shape',
                         'tensor_dtype',
                         'tensor_fsize',
                         'tensor_fsize_nice',
                         'func_applied_name',
                         'func_time_elapsed',
                         'function_is_inplace',
                         'gradfunc',
                         'is_input_layer',
                         'is_output_layer',
                         'is_buffer_layer',
                         'is_part_of_iterable_output',
                         'iterable_output_index',
                         'parent_layers',
                         'has_parents',
                         'orig_ancestors',
                         'child_layers',
                         'has_children',
                         'output_descendents',
                         'sibling_layers',
                         'has_siblings',
                         'spouse_layers',
                         'has_spouses',
                         'computed_with_params',
                         'num_params_total',
                         'parent_param_shapes',
                         'parent_params_fsize',
                         'parent_params_fsize_nice',
                         'modules_entered',
                         'modules_exited',
                         'is_submodule_input',
                         'is_submodule_output',
                         'containing_module_origin',
                         'containing_modules_origin_nested']

        fields_to_change_type = {'layer_type_num': int,
                                 'layer_total_num': int,
                                 'layer_passes_total': int,
                                 'pass_num': int,
                                 'operation_num': int,
                                 'function_is_inplace': bool,
                                 'is_input_layer': bool,
                                 'is_output_layer': bool,
                                 'is_buffer_layer': bool,
                                 'is_part_of_iterable_output': bool,
                                 'has_parents': bool,
                                 'has_children': bool,
                                 'has_siblings': bool,
                                 'has_spouses': bool,
                                 'computed_with_params': bool,
                                 'num_params_total': int,
                                 'parent_params_fsize': int,
                                 'tensor_fsize': int,
                                 'is_submodule_input': bool,
                                 'is_submodule_output': bool}

        for tensor_entry in self.layer_list:
            tensor_dict = {}
            for field_name in fields_for_df:
                tensor_dict[field_name] = getattr(tensor_entry, field_name)
            model_df = model_df.append(tensor_dict, ignore_index=True)

        for field in fields_to_change_type:
            model_df[field] = model_df[field].astype(fields_to_change_type[field])

        return model_df

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
        for layer_key in which_layers:
            # First check if it matches a lookup key. If so, use that.
            if layer_key in self.lookup_keys_to_tensor_num_dict:
                raw_tensor_nums_to_save.add(self.lookup_keys_to_tensor_num_dict[layer_key])
                continue

            # If not, pull out all layers for which the key is a substring.
            keys_with_substr = [key for key in self.layer_dict_all_keys if layer_key in str(key)]
            if len(keys_with_substr) > 0:
                for key in keys_with_substr:
                    raw_tensor_nums_to_save.add(self.layer_dict_all_keys[key].realtime_tensor_num)
                continue

            # If no luck, try to at least point user in right direction:

            self._give_user_feedback_about_lookup_key(layer_key, 'query_multiple')

        raw_tensor_nums_to_save = sorted(list(raw_tensor_nums_to_save))
        return raw_tensor_nums_to_save

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
        self.tensor_counter += 1
        self.raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self.tensor_counter
        layer_type_num = self.raw_layer_type_counter[layer_type]

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
            'detach_saved_tensor': self.detach_saved_tensors,
            'creation_args': [],
            'creation_kwargs': {},
            'tensor_shape': tuple(t.shape),
            'tensor_dtype': t.dtype,
            'tensor_fsize': get_tensor_memory_amount(t),
            'tensor_fsize_nice': human_readable_size(get_tensor_memory_amount(t)),

            # Grad info:
            'grad_contents': None,
            'save_gradients': self.save_gradients,
            'has_saved_grad': False,
            'grad_shapes': None,
            'grad_dtypes': None,
            'grad_fsizes': 0,
            'grad_fsizes_nice': human_readable_size(0),

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
            'gradfunc': 'none',
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
            'same_layer_operations': [],

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
            'is_last_output_layer': False,
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
            'modules_entered': [],
            'module_passes_entered': [],
            'is_submodule_input': False,
            'modules_exited': [],
            'module_passes_exited': [],
            'is_submodule_output': False,
            'is_bottom_level_submodule_output': False,
            'bottom_level_submodule_pass_exited': None,
            'module_entry_exit_thread': []
        }

        self._make_tensor_log_entry(t, fields_dict, (), {})

        # Tag the tensor itself with its label, and with a reference to the model history log.
        t.tl_tensor_label_raw = tensor_label

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
                                    arg_copies: Tuple[Any],
                                    kwarg_copies: Dict[str, Any],
                                    out_orig: Any,
                                    func_time_elapsed: float,
                                    func_rng_states: Dict,
                                    is_bottom_level_func: bool):
        """Logs tensor or set of tensors that were computed from a function call.

        Args:
            func: function that was called
            args: positional arguments to function that was called
            kwargs: keyword arguments to function that was called
            arg_copies: copies of positional arguments to function that was called
            kwarg_copies: copies of keyword arguments to function that was called
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
        fields_dict['detach_saved_tensor'] = self.detach_saved_tensors

        # Grad info:
        fields_dict['grad_contents'] = None
        fields_dict['save_gradients'] = self.save_gradients
        fields_dict['has_saved_grad'] = False
        fields_dict['grad_shapes'] = None
        fields_dict['grad_dtypes'] = None
        fields_dict['grad_fsizes'] = 0
        fields_dict['grad_fsizes_nice'] = human_readable_size(0)

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
        fields_dict['is_last_output_layer'] = False
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
        fields_dict['num_params_total'] = int(np.sum([np.prod(shape) for shape in fields_dict['parent_param_shapes']]))
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
        fields_dict['modules_entered'] = []
        fields_dict['module_passes_entered'] = []
        fields_dict['is_submodule_input'] = False
        fields_dict['modules_exited'] = []
        fields_dict['module_passes_exited'] = []
        fields_dict['is_submodule_output'] = False
        fields_dict['is_bottom_level_submodule_output'] = False
        fields_dict['bottom_level_submodule_pass_exited'] = None
        fields_dict['module_entry_exit_thread'] = []

        is_part_of_iterable_output = type(out_orig) in [list, tuple, dict, set]
        fields_dict['is_part_of_iterable_output'] = is_part_of_iterable_output
        out_iter = make_var_iterable(out_orig)  # so we can iterate through it

        for i, out in enumerate(out_iter):
            if not self._output_should_be_logged(out, is_bottom_level_func):
                continue

            fields_dict_onetensor = {key: copy.copy(value) for key, value in fields_dict.items()}
            fields_to_deepcopy = ['parent_layer_arg_locs', 'containing_modules_origin_nested', 'parent_param_passes']
            for field in fields_to_deepcopy:
                fields_dict_onetensor[field] = copy.deepcopy(fields_dict[field])
            self._log_info_specific_to_single_function_output_tensor(out, i, args, kwargs,
                                                                     parent_param_passes, fields_dict_onetensor)
            self._make_tensor_log_entry(out, fields_dict=fields_dict_onetensor,
                                        t_args=arg_copies, t_kwargs=kwarg_copies)
            new_tensor_entry = self[fields_dict_onetensor['tensor_label_raw']]
            new_tensor_label = new_tensor_entry.tensor_label_raw
            self._update_tensor_family_links(new_tensor_entry)

            # Update relevant fields of ModelHistory
            # Add layer to relevant fields of ModelHistory:
            if fields_dict['initialized_inside_model']:
                self.internally_initialized_layers.append(new_tensor_label)
            if fields_dict['has_input_ancestor'] and any([(self[parent_layer].has_internally_initialized_ancestor and
                                                           not self[parent_layer].has_input_ancestor)
                                                          for parent_layer in
                                                          fields_dict_onetensor['parent_layers']]):
                self.layers_where_internal_branches_merge_with_input.append(new_tensor_label)

            # Tag the tensor itself with its label, and add a backward hook if saving gradients.
            out.tl_tensor_label_raw = fields_dict_onetensor['tensor_label_raw']
            if self.save_gradients:
                self._add_backward_hook(out, out.tl_tensor_label_raw)

    @staticmethod
    def _output_should_be_logged(out: Any,
                                 is_bottom_level_func: bool) -> bool:
        """Function to check whether to log the output of a function.

        Returns:
            True if the output should be logged, False otherwise.
        """
        if type(out) != torch.Tensor:  # only log if it's a tensor
            return False

        if (not hasattr(out, 'tl_tensor_label_raw')) or is_bottom_level_func:
            return True
        else:
            return False

    def _add_backward_hook(self, t: torch.Tensor, tensor_label):
        """Adds a backward hook to the tensor that saves the gradients to ModelHistory if specified.

        Args:
            t: tensor

        Returns:
            Nothing; it changes the tensor in place.
        """

        # Define the decorator
        def log_grad_to_model_history(g_in, g_out):
            self._log_tensor_grad(g_out, tensor_label)

        t.grad_fn.register_hook(log_grad_to_model_history)

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
        self.tensor_counter += 1
        self.raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self.tensor_counter
        layer_type_num = self.raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

        if len(parent_param_passes) > 0:
            operation_equivalence_type = self._make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
            fields_dict['operation_equivalence_type'] = operation_equivalence_type
            self.layers_computed_with_params[operation_equivalence_type].append(tensor_label_raw)
            fields_dict['pass_num'] = len(self.layers_computed_with_params[operation_equivalence_type])
        else:
            operation_equivalence_type = self._get_operation_equivalence_type(args, kwargs, i,
                                                                              layer_type, fields_dict)
            fields_dict['operation_equivalence_type'] = operation_equivalence_type
            fields_dict['pass_num'] = 1

        self.equivalent_operations[operation_equivalence_type].add(tensor_label_raw)
        fields_dict['equivalent_operations'] = self.equivalent_operations[operation_equivalence_type]

        fields_dict['function_is_inplace'] = hasattr(t, 'tl_tensor_label_raw')
        fields_dict['gradfunc'] = type(t.grad_fn)

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
        fields_dict['same_layer_operations'] = []
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

        # If internally initialized, fix this information:
        if len(fields_dict['parent_layers']) == 0:
            fields_dict['initialized_inside_model'] = True
            fields_dict['has_internally_initialized_ancestor'] = True
            fields_dict['internally_initialized_parents'] = []
            fields_dict['internally_initialized_ancestors'] = {tensor_label_raw}

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

    def _log_tensor_grad(self,
                         grad: torch.Tensor,
                         tensor_label_raw: str):
        """Logs the gradient for a tensor during a backward pass.

        Args:
            grad: the gradient
            tensor_label_raw: the raw tensor label

        Returns:

        """
        self.has_saved_gradients = True
        tensor_label = self.raw_to_final_layer_labels[tensor_label_raw]
        if tensor_label not in self.layers_with_saved_gradients:
            self.layers_with_saved_gradients.append(tensor_label)
            layer_order = {layer: i for i, layer in enumerate(self.layer_labels)}
            self.layers_with_saved_gradients = sorted(self.layers_with_saved_gradients,
                                                      key=lambda x: layer_order[x])
        tensor_log_entry = self[tensor_label]
        tensor_log_entry.log_tensor_grad(grad)

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
            if getattr(arg, 'tl_tensor_label_raw', -1) == parent_entry.tensor_label_raw:
                tensor_all_arg_positions[arg_type][arg_key] = parent_entry.tensor_label_raw
            elif type(arg) in [list, tuple, dict]:
                iterfunc2 = iterfunc_dict[type(arg)]
                for sub_arg_key, sub_arg in iterfunc2(arg):
                    if getattr(sub_arg, 'tl_tensor_label_raw', -1) == parent_entry.tensor_label_raw:
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

    def _get_operation_equivalence_type(self,
                                        args: Tuple,
                                        kwargs: Dict,
                                        i: int,
                                        layer_type: str,
                                        fields_dict: Dict):
        arg_hash = self._get_hash_from_args(args, kwargs)
        operation_equivalence_type = f"{layer_type}_{arg_hash}"
        if fields_dict['is_part_of_iterable_output']:
            operation_equivalence_type += f'_outindex{i}'
        if fields_dict['containing_module_origin'] is not None:
            module_str = fields_dict['containing_module_origin'][0]
            operation_equivalence_type += f'_module{module_str}'
        return operation_equivalence_type

    @staticmethod
    def _get_hash_from_args(args, kwargs):
        """
        Get a hash from the args and kwargs of a function call, excluding any tracked tensors.
        """
        args_to_hash = []
        for a, arg in enumerate(list(args) + list(kwargs.values())):
            if hasattr(arg, 'tl_tensor_label_raw'):
                args_to_hash.append(f"arg{a}{arg.shape}")
            else:
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

    def cleanup(self):
        """Deletes all log entries in the model.
        """
        for tensor_log_entry in self:
            self._remove_log_entry(tensor_log_entry, remove_references=True)
        for attr in MODEL_HISTORY_FIELD_ORDER:
            delattr(self, attr)
        torch.cuda.empty_cache()

    def _remove_log_entry(self,
                          log_entry: TensorLogEntry,
                          remove_references: bool = True):
        """Given a TensorLogEntry, destroys it and all references to it.

        Args:
            log_entry: Tensor log entry to remove.
            remove_references: Whether to also remove references to the log entry
        """
        if self.pass_finished:
            tensor_label = log_entry.layer_label
        else:
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
        self.layers_computed_with_params = {k: v for k, v in self.layers_computed_with_params.items() if len(v) > 0}

        for group_label, group_tensors in self.equivalent_operations.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
        self.equivalent_operations = {k: v for k, v in self.equivalent_operations.items() if len(v) > 0}

        for group_label, group_tensors in self.same_layer_operations.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
        self.same_layer_operations = {k: v for k, v in self.same_layer_operations.items() if len(v) > 0}

    # ********************************************
    # ************* Post-Processing **************
    # ********************************************

    def postprocess(self, decorated_func_mapper: Dict, mark_input_output_distances: bool = True):
        """
        After the forward pass, cleans up the log into its final form.
        """
        # Step 1: Add dedicated output nodes

        self._add_output_layers()

        # Step 2: Remove orphan nodes, find nodes that don't terminate in output node

        self._remove_orphan_nodes()

        # Step 3: Find mix/max distance from input and output nodes

        if mark_input_output_distances:
            self._mark_input_output_distances()

        # Step 4: Starting from terminal single boolean tensors, mark the conditional branches.

        self._mark_conditional_branches()

        # Step 5: Annotate the containing modules for all internally-generated tensors (they don't know where
        # they are when they're made; have to trace breadcrumbs from tensors that came from input).

        self._fix_modules_for_internal_tensors()

        # Step 6: Identify all loops, mark repeated layers.

        self._assign_corresponding_tensors_to_same_layer()

        # Step 7: Go down tensor list, get the mapping from raw tensor names to final tensor names.

        self._map_raw_tensor_labels_to_final_tensor_labels()

        # Step 8: Process ModelHistory into its final user-facing version: undecorate all tensors,
        # do any tallying/totals/labeling, log the module hierarchy, rename all tensors,
        # get the operation numbers for all layer labels.

        self._final_prettify(decorated_func_mapper)

        # Step 9: log the pass as finished, changing the ModelHistory behavior to its user-facing version.

        self._set_pass_finished()

    def _add_output_layers(self):
        """
        Adds dedicated output nodes to the graph.
        """
        new_output_layers = []
        for i, output_layer_label in enumerate(self.output_layers):
            output_node = self[output_layer_label]
            new_output_node = output_node.copy()
            new_output_node.layer_type = 'output'
            if i == len(self.output_layers) - 1:
                new_output_node.is_last_output_layer = True
            self.tensor_counter += 1
            new_output_node.tensor_label_raw = f"output_{i + 1}_{self.tensor_counter}_raw"
            new_output_node.layer_label_raw = new_output_node.tensor_label_raw
            new_output_node.realtime_tensor_num = self.tensor_counter

            # Fix function information:

            new_output_node.func_applied = identity
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
            new_output_node.creation_args = [safe_copy(output_node.tensor_contents)]
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
            new_output_node.modules_exited = []
            new_output_node.module_passes_exited = []
            new_output_node.is_submodule_output = False
            new_output_node.is_bottom_level_submodule_output = False
            new_output_node.module_entry_exit_thread = []

            # Fix ancestry information:

            new_output_node.child_layers = []
            new_output_node.parent_layers = [output_node.tensor_label_raw]
            new_output_node.sibling_layers = []
            new_output_node.has_sibling_tensors = False
            new_output_node.parent_layer_arg_locs = {'args': {0: output_node.tensor_label_raw},
                                                     'kwargs': {}}

            # Fix layer equivalence information:
            new_output_node.same_layer_operations = []
            equiv_type = f"output_{'_'.join(tuple(str(s) for s in new_output_node.tensor_shape))}_" \
                         f"{str(new_output_node.tensor_dtype)}"
            new_output_node.operation_equivalence_type = equiv_type
            self.equivalent_operations[equiv_type].add(new_output_node.tensor_label_raw)

            # Change original output node:

            output_node.is_output_layer = False
            output_node.child_layers.append(new_output_node.tensor_label_raw)

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
            if (len(tensor_entry.child_layers) == 0) and (not tensor_entry.is_output_layer):
                self._log_internally_terminated_tensor(tensor_label)
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
            layer_logging_field = 'input_ancestors'
            forward_field = 'child_layers'
        elif mode == 'output':
            starting_nodes = self.output_layers[:]
            min_field = 'min_distance_from_output'
            max_field = 'max_distance_from_output'
            direction = 'backwards'
            marker_field = 'is_output_ancestor'
            layer_logging_field = 'output_descendents'
            forward_field = 'parent_layers'
        else:
            raise ValueError("Mode but be either 'input' or 'output'")

        nodes_seen = set()

        # Tuples in format node_label, nodes_since_start, traversal_direction
        node_stack = [(starting_node_label, starting_node_label, 0, direction) for starting_node_label in
                      starting_nodes]
        while len(node_stack) > 0:
            current_node_label, orig_node, nodes_since_start, traversal_direction = node_stack.pop()
            nodes_seen.add(current_node_label)
            current_node = self[current_node_label]
            self._update_node_distance_vals(current_node, min_field, max_field, nodes_since_start)

            setattr(current_node, marker_field, True)
            getattr(current_node, layer_logging_field).add(orig_node)

            for next_node_label in getattr(current_node, forward_field):
                if self._check_whether_to_add_node_to_flood_stack(next_node_label, orig_node, nodes_since_start,
                                                                  min_field, max_field, layer_logging_field,
                                                                  nodes_seen):
                    node_stack.append((next_node_label, orig_node, nodes_since_start + 1, traversal_direction))

    @staticmethod
    def _update_node_distance_vals(current_node: TensorLogEntry,
                                   min_field: str,
                                   max_field: str,
                                   nodes_since_start: int):
        if getattr(current_node, min_field) is None:
            setattr(current_node, min_field, nodes_since_start)
        else:
            setattr(current_node, min_field, min([nodes_since_start, getattr(current_node, min_field)]))

        if getattr(current_node, max_field) is None:
            setattr(current_node, max_field, nodes_since_start)
        else:
            setattr(current_node, max_field, max([nodes_since_start, getattr(current_node, max_field)]))

    def _check_whether_to_add_node_to_flood_stack(self,
                                                  candidate_node_label: str,
                                                  orig_node_label: str,
                                                  nodes_since_start: int,
                                                  min_field: str,
                                                  max_field: str,
                                                  layer_logging_field: str,
                                                  nodes_seen: set):
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
            node_stack.extend(node.child_layers)
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
        for starting_label in equivalent_operation_starting_labels:
            subgraphs_dict[starting_label] = {'starting_node': starting_label,
                                              'param_nodes': set(),
                                              'node_set': {starting_label}}
            if node.computed_with_params:
                subgraphs_dict[starting_label]['param_nodes'].add(starting_label)

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
            if len(isomorphic_nodes) == 1:
                continue
            self._fetch_and_process_next_isomorphic_nodes(isomorphic_nodes,
                                                          iso_node_groups,
                                                          node_to_iso_group_dict,
                                                          subgraphs_dict,
                                                          node_to_subgraph_dict,
                                                          adjacent_subgraphs,
                                                          is_first_node,
                                                          node_stack)
            is_first_node = False

        self._assign_and_log_isomorphic_nodes_to_same_layers(iso_node_groups, node_to_subgraph_dict, adjacent_subgraphs)

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
                                                                                 iso_node_groups,
                                                                                 node_to_iso_group_dict,
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
                                                     iso_node_groups: Dict[str, List[str]],
                                                     node_to_iso_group_dict: Dict[str, str],
                                                     node_to_subgraph_dict: Dict,
                                                     adjacent_subgraphs: Dict[str, set],
                                                     is_first_node: bool) -> Dict:
        """Helper function that checks all parent and children nodes for overlap with nodes already added
        to subgraphs (either the same subgraph or another one), logs any adjacency among subgraphs,
        and returns a dict with the candidate successor nodes from each subgraph.

        Returns:
            Dict with the candidate next nodes for each subgraph.
        """
        node_type_fields = {'children': 'child_layers',
                            'parents': 'parent_layers'}
        if is_first_node:
            node_types_to_use = ['children']
        else:
            node_types_to_use = ['children', 'parents']

        successor_nodes_dict = OrderedDict()
        for node_label in current_iso_nodes:
            node = self[node_label]
            node_subgraph = node_to_subgraph_dict[node_label]
            node_subgraph_label = node_subgraph['starting_node']
            subgraph_successor_nodes = {'children': [], 'parents': []}
            for node_type in node_types_to_use:
                node_type_field = node_type_fields[node_type]
                for neighbor_label in getattr(node, node_type_field):
                    if neighbor_label in node_subgraph['node_set']:  # skip if backtracking own subgraph
                        continue
                    elif neighbor_label in node_to_subgraph_dict:  # if hit another subgraph, mark them adjacent & skip
                        self._check_and_mark_subgraph_adjacency(node_label,
                                                                neighbor_label,
                                                                iso_node_groups,
                                                                node_to_iso_group_dict,
                                                                node_to_subgraph_dict,
                                                                adjacent_subgraphs)
                    else:  # we have a new, non-overlapping node as a possible candiate, add it:
                        subgraph_successor_nodes[node_type].append(neighbor_label)
            successor_nodes_dict[node_subgraph_label] = subgraph_successor_nodes

        return successor_nodes_dict

    @staticmethod
    def _check_and_mark_subgraph_adjacency(node_label: str,
                                           neighbor_label: str,
                                           iso_node_groups: Dict[str, List[str]],
                                           node_to_iso_group_dict: Dict[str, str],
                                           node_to_subgraph_dict: Dict,
                                           adjacent_subgraphs: Dict[str, set]):
        """Helper function that updates the adjacency status of two subgraphs
        """
        node_subgraph = node_to_subgraph_dict[node_label]
        node_subgraph_label = node_subgraph['starting_node']
        neighbor_subgraph = node_to_subgraph_dict[neighbor_label]
        neighbor_subgraph_label = neighbor_subgraph['starting_node']

        # Subgraphs are adjacent if the node in the neighboring subgraph has an
        # isomorphic node in the current subgraph.

        neighbor_iso_group = node_to_iso_group_dict[neighbor_label]
        nodes_isomorphic_to_neighbor_node = iso_node_groups[neighbor_iso_group]
        if len(node_subgraph['node_set'].intersection(nodes_isomorphic_to_neighbor_node)) == 0:
            return

        # Update adjacency
        if (node_subgraph_label in adjacent_subgraphs) and (neighbor_subgraph_label in adjacent_subgraphs):
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
                subgraphs_dict[node_subgraph]['node_set'].add(node_label)
                if node.computed_with_params:
                    subgraphs_dict[node_subgraph]['param_nodes'].add(node_label)
                node_to_subgraph_dict[node_label] = subgraphs_dict[node_subgraph]
            node_stack.append(equivalent_node_labels)

    def _assign_and_log_isomorphic_nodes_to_same_layers(self, iso_node_groups: Dict[str, List],
                                                        node_to_subgraph_dict: Dict, adjacent_subgraphs: Dict):
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
        same_layer_node_groups = self._group_isomorphic_nodes_to_same_layers(iso_node_groups, node_to_subgraph_dict,
                                                                             adjacent_subgraphs)

        # Finally, label the nodes corresponding to the same layer.
        for layer_label, layer_nodes in same_layer_node_groups.items():
            # Skip if the new layer asssignment reduces the number of equivalent layers.
            if len(layer_nodes) < len(self[layer_label].same_layer_operations):
                continue
            # convert to list and sort
            layer_nodes = sorted(list(layer_nodes), key=lambda layer: self[layer].realtime_tensor_num)
            for n, node_label in enumerate(layer_nodes):
                node = self[node_label]
                node.layer_label_raw = layer_label
                node.same_layer_operations = layer_nodes
                node.pass_num = n + 1
                node.layer_passes_total = len(layer_nodes)

    def _group_isomorphic_nodes_to_same_layers(self,
                                               iso_node_groups: Dict[str, List],
                                               node_to_subgraph_dict: Dict,
                                               adjacent_subgraphs: Dict) -> Dict:
        same_layer_node_groups = defaultdict(set)  # dict of nodes assigned to the same layer
        node_to_layer_group_dict = {}  # reverse mapping: each node to its equivalent layer group

        for iso_group_label, iso_nodes_orig in iso_node_groups.items():
            iso_nodes = sorted(iso_nodes_orig)
            for node1_label, node2_label in it.combinations(iso_nodes, 2):
                node1_subgraph = node_to_subgraph_dict[node1_label]
                node2_subgraph = node_to_subgraph_dict[node2_label]
                node1_subgraph_label = node1_subgraph['starting_node']
                node2_subgraph_label = node2_subgraph['starting_node']
                node1_param_types = [self[pnode].operation_equivalence_type for pnode in node1_subgraph['param_nodes']]
                node2_param_types = [self[pnode].operation_equivalence_type for pnode in node2_subgraph['param_nodes']]
                overlapping_param_types = set(node1_param_types).intersection(set(node2_param_types))
                subgraphs_are_adjacent = (node1_subgraph_label in adjacent_subgraphs and
                                          node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label])
                if (len(overlapping_param_types) > 0) or subgraphs_are_adjacent:
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
                first_pass_tensor = self[tensor_log_entry.same_layer_operations[0]]
                layer_type_num = first_pass_tensor.layer_type_num
                layer_total_num = first_pass_tensor.layer_total_num
            tensor_log_entry.layer_type_num = layer_type_num
            tensor_log_entry.layer_total_num = layer_total_num
            tensor_log_entry.layer_label_w_pass = f"{layer_type}_{layer_type_num}_{layer_total_num}:{pass_num}"
            tensor_log_entry.layer_label_no_pass = f"{layer_type}_{layer_type_num}_{layer_total_num}"
            tensor_log_entry.layer_label_w_pass_short = f"{layer_type}_{layer_type_num}:{pass_num}"
            tensor_log_entry.layer_label_no_pass_short = f"{layer_type}_{layer_type_num}"
            if tensor_log_entry.layer_passes_total == 1:
                tensor_log_entry.layer_label = tensor_log_entry.layer_label_no_pass
                tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_no_pass_short
            else:
                tensor_log_entry.layer_label = tensor_log_entry.layer_label_w_pass
                tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_w_pass_short
            raw_to_final_layer_labels[tensor_log_entry.tensor_label_raw] = tensor_log_entry.layer_label
        self.raw_to_final_layer_labels = raw_to_final_layer_labels

    def _final_prettify(self, decorated_func_mapper):
        """
        Goes through all tensor log entries for the final stages of pre-processing to make the
        user-facing version of ModelHistory.
        """
        # Go through and log information pertaining to all layers:
        self._log_final_info_for_all_layers()

        # Rename the raw tensor entries in the fields of ModelHistory:
        self._rename_model_history_layer_names()
        self._trim_and_reorder_model_history_fields()

        # And one more pass to delete unused layers from the record and do final tidying up:
        self._remove_unwanted_entries_and_log_remaining()

        # Undecorate all saved tensors and remove saved grad_fns.
        self._undecorate_all_saved_tensors(decorated_func_mapper)

        # Clear the cache after any tensor deletions for garbage collection purposes:
        torch.cuda.empty_cache()

    def _log_final_info_for_all_layers(self):
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
            if tensor_entry.bottom_level_submodule_pass_exited is not None:
                submodule_pass_nice_name = ':'.join([str(i) for i in tensor_entry.bottom_level_submodule_pass_exited])
                tensor_entry.bottom_level_submodule_pass_exited = submodule_pass_nice_name

            # Tally the tensor sizes:
            self.tensor_fsize_total += tensor_entry.tensor_fsize

            # Tally the parameter sizes:
            if tensor_entry.layer_label_no_pass not in unique_layers_seen:  # only count params once
                if tensor_entry.computed_with_params:
                    self.total_param_layers += 1
                self.total_params += tensor_entry.num_params_total
                self.total_param_tensors += tensor_entry.num_param_tensors
                self.total_params_fsize += tensor_entry.parent_params_fsize
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

        self.num_tensors_total = len(self)

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
        list_fields_to_rename = ['parent_layers', 'orig_ancestors', 'child_layers',
                                 'sibling_layers', 'spouse_layers', 'input_ancestors',
                                 'output_descendents', 'internally_initialized_parents',
                                 'internally_initialized_ancestors', 'cond_branch_start_children',
                                 'equivalent_operations', 'same_layer_operations']
        for field in list_fields_to_rename:
            orig_layer_names = getattr(tensor_entry, field)
            field_type = type(orig_layer_names)
            new_layer_names = field_type([self.raw_to_final_layer_labels[raw_name] for raw_name in orig_layer_names])
            setattr(tensor_entry, field, new_layer_names)

        # Fix the arg locations field:
        for arg_type in ['args', 'kwargs']:
            for key, value in tensor_entry.parent_layer_arg_locs[arg_type].items():
                tensor_entry.parent_layer_arg_locs[arg_type][key] = self.raw_to_final_layer_labels[value]

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
            module_pass_nice_label = f"{module_name}:{module_pass}"
            if (m == 0) and (module_pass_nice_label not in self.top_level_module_passes):
                self.top_level_module_passes.append(module_pass_nice_label)
            else:
                if ((containing_module_pass_label is not None) and
                        (module_pass_nice_label not in self.module_pass_children[containing_module_pass_label])):
                    self.module_pass_children[containing_module_pass_label].append(module_pass_nice_label)
            containing_module_pass_label = module_pass_nice_label
            if self.module_num_passes[module_name] < module_pass:
                self.module_num_passes[module_name] = module_pass
            if module_name not in self.module_addresses:
                self.module_addresses.append(module_name)
            if module_pass_label not in self.module_passes:
                self.module_passes.append(module_pass_nice_label)

    def _remove_unwanted_entries_and_log_remaining(self):
        """Removes entries from ModelHistory that we don't want in the final saved output,
        and logs information about the remaining entries.
        """
        tensors_to_remove = []
        self.unsaved_layers_lookup_keys = set()
        # Quick loop to count how many tensors are saved:
        for tensor_entry in self:
            if tensor_entry.has_saved_activations:
                self.num_tensors_saved += 1

        i = 0
        for tensor_entry in self:
            # Determine valid lookup keys and relate them to the tensor's realtime operation number:
            if tensor_entry.has_saved_activations or self.keep_layers_without_saved_activations:
                # Add the lookup keys for the layer, to itself and to ModelHistory:
                self._add_lookup_keys_for_tensor_entry(tensor_entry, i, self.num_tensors_saved)

                # Log all information:
                self.layer_list.append(tensor_entry)
                self.layer_dict_main_keys[tensor_entry.layer_label] = tensor_entry
                self.layer_labels.append(tensor_entry.layer_label)
                self.layer_labels_no_pass.append(tensor_entry.layer_label_no_pass)
                self.layer_labels_w_pass.append(tensor_entry.layer_label_w_pass)
                self.layer_num_passes[tensor_entry.layer_label] = tensor_entry.layer_passes_total
                if tensor_entry.has_saved_activations:
                    self.tensor_fsize_saved += tensor_entry.tensor_fsize
                self._trim_and_reorder_tensor_entry_fields(tensor_entry)  # Final reformatting of fields
                i += 1
            else:
                tensors_to_remove.append(tensor_entry)
                self.unsaved_layers_lookup_keys.update(tensor_entry.lookup_keys)

        if (self.num_tensors_saved == len(self)) or self.keep_layers_without_saved_activations:
            self.all_layers_logged = True
        else:
            self.all_layers_logged = False

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
                                  tensor_index - num_tensors_to_keep]

        # If just one pass, also allow indexing by pass label.
        if tensor_entry.layer_passes_total == 1:
            lookup_keys_for_tensor.extend([tensor_entry.layer_label_w_pass,
                                           tensor_entry.layer_label_w_pass_short])

        # Allow indexing by modules exited as well:
        for module_name, pass_num in tensor_entry.module_passes_exited:
            lookup_keys_for_tensor.append(f"{module_name}:{pass_num}")
            if self.module_num_passes[module_name] == 1:
                lookup_keys_for_tensor.append(f"{module_name}")

        # Now that we don't need the module pass separately, we can relabel the passes:
        tensor_entry.module_passes_exited = [f"{module_name}:{module_pass}" for module_name, module_pass
                                             in tensor_entry.module_passes_exited]
        tensor_entry.module_passes_entered = [f"{module_name}:{module_pass}" for module_name, module_pass
                                              in tensor_entry.module_passes_entered]
        if tensor_entry.containing_module_origin is not None:
            tensor_entry.containing_module_origin = ':'.join([str(i) for i in tensor_entry.containing_module_origin])
        tensor_entry.containing_modules_origin_nested = [f"{module_name}:{module_pass}" for module_name, module_pass
                                                         in tensor_entry.containing_modules_origin_nested]

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
        list_fields_to_rename = ['input_layers', 'output_layers', 'buffer_layers', 'internally_initialized_layers',
                                 'layers_where_internal_branches_merge_with_input', 'internally_terminated_layers',
                                 'internally_terminated_bool_layers']
        for field in list_fields_to_rename:
            tensor_labels = getattr(self, field)
            setattr(self, field, [self.raw_to_final_layer_labels[tensor_label] for tensor_label in tensor_labels])

        new_param_tensors = {}
        for key, values in self.layers_computed_with_params.items():
            new_key = self[values[0]].layer_label
            new_param_tensors[new_key] = [self.raw_to_final_layer_labels[tensor_label] for tensor_label in values]
        self.layers_computed_with_params = new_param_tensors

        new_equiv_operations_tensors = {}
        for key, values in self.equivalent_operations.items():
            new_equiv_operations_tensors[key] = set([self.raw_to_final_layer_labels[tensor_label]
                                                     for tensor_label in values])
        self.equivalent_operations = new_equiv_operations_tensors

        new_same_layer_operations = {}
        for key, values in self.same_layer_operations.items():
            new_key = self.raw_to_final_layer_labels[key]
            new_same_layer_operations[new_key] = [self.raw_to_final_layer_labels[tensor_label] for tensor_label in
                                                  values]
        self.same_layer_operations = new_same_layer_operations

        for t, (child, parent) in enumerate(self.conditional_branch_edges):
            new_child, new_parent = self.raw_to_final_layer_labels[child], self.raw_to_final_layer_labels[parent]
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

    @staticmethod
    def _undecorate_saved_tensor(t, decorated_func_mapper):
        if hasattr(t, 'tl_tensor_label_raw'):
            delattr(t, 'tl_tensor_label_raw')
        for attr_name in dir(t):
            if attr_name.startswith('_') or (attr_name in ['grad']):
                continue
            try:
                attr = getattr(t, attr_name)
            except (AttributeError, TypeError, RuntimeError) as _:
                continue
            if attr in decorated_func_mapper:
                setattr(t, attr_name, decorated_func_mapper[attr])

    def _undecorate_all_saved_tensors(self, decorated_func_mapper: Dict):
        """Utility function to undecorate all saved tensors.

        Args:
            decorated_func_mapper: Maps decorated functions to their original versions.
        """
        tensors_to_undecorate = []
        for layer_label in self.layer_labels:
            tensor_entry = self.layer_dict_main_keys[layer_label]
            if tensor_entry.tensor_contents is not None:
                tensors_to_undecorate.append(tensor_entry.tensor_contents)

            tensors_to_undecorate.extend(get_vars_of_type_from_obj(tensor_entry.creation_args,
                                                                   torch.Tensor,
                                                                   search_depth=4))
            tensors_to_undecorate.extend(get_vars_of_type_from_obj(tensor_entry.creation_kwargs,
                                                                   torch.Tensor,
                                                                   search_depth=4))

        for t in tensors_to_undecorate:
            self._undecorate_saved_tensor(t, decorated_func_mapper)

    def _delete_raw_tensor_entries(self):
        """Deletes the raw tensor entries, leaving only the post-processed entries.
        """
        for entry_name, tensor_entry in self.raw_tensor_dict.items():
            self._remove_log_entry(tensor_entry)
        self.raw_tensor_dict.clear()

    def _set_pass_finished(self):
        """Sets the ModelHistory to "pass finished" status, indicating that the pass is done, so
        the "final" rather than "realtime debugging" mode of certain functions should be used.
        """
        for layer_label in self.layer_dict_main_keys:
            tensor = self.layer_dict_main_keys[layer_label]
            tensor.pass_finished = True
        self.pass_finished = True

    def _roll_graph(self):
        """
        Converts the graph to rolled-up format for plotting purposes, such that each node now represents
        all passes of a given layer instead of having separate nodes for each pass.
        """
        for layer_label, node in self.layer_dict_main_keys.items():
            layer_label_no_pass = self[layer_label].layer_label_no_pass
            if layer_label_no_pass in self.layer_dict_rolled:  # If rolled-up layer has already been added, fetch it:
                rolled_node = self.layer_dict_rolled[layer_label_no_pass]
            else:  # If it hasn't been added, make it:
                rolled_node = RolledTensorLogEntry(node)
                self.layer_dict_rolled[node.layer_label_no_pass] = rolled_node
                self.layer_list_rolled.append(rolled_node)

            rolled_node.add_pass_info(node)

    # ********************************************
    # *************** Visualization **************
    # ********************************************

    def render_graph(self,
                     vis_opt: str = 'unrolled',
                     vis_outpath: str = 'graph.gv',
                     save_only: bool = False,
                     vis_fileformat: str = 'pdf',
                     show_buffer_layers: bool = False,
                     direction: str = 'vertical') -> None:
        """Renders the computational graph for the model.

        Args:
            vis_opt: either 'rolled' or 'unrolled'
            vis_outpath: where to store the rendered graph
            save_only: whether to only save the graph without immediately showing it
            vis_fileformat: file format to use for the rendered graph
            show_buffer_layers: whether to show the buffer layers
            direction: which way the graph should go: either 'vertical' or 'horizontal'

        """
        if not self.all_layers_logged:
            raise ValueError("Must have all layers logged in order to render the graph; either save "
                             "all layers in get_model_activations, or use show_model_graph to "
                             "render the graph without saving any activations.")

        if vis_opt == 'unrolled':
            entries_to_plot = self.layer_dict_main_keys
        elif vis_opt == 'rolled':
            self._roll_graph()
            entries_to_plot = self.layer_dict_rolled
        else:
            raise ValueError("vis_opt must be either 'rolled' or 'unrolled'")

        if direction == 'vertical':
            rankdir = 'BT'
        elif direction == 'horizontal':
            rankdir = 'LR'
        else:
            raise ValueError("direction must be either 'vertical' or 'horizontal'")

        graph_caption = (
            f"<<B>{self.model_name}</B><br align='left'/>{self.num_tensors_total} "
            f"tensors total ({self.tensor_fsize_total_nice})"
            f"<br align='left'/>{self.total_params} params total ({self.total_params_fsize_nice})<br align='left'/>>")

        dot = graphviz.Digraph(name=self.model_name, comment='Computational graph for the feedforward sweep',
                               format=vis_fileformat)
        dot.graph_attr.update({'rankdir': rankdir,
                               'label': graph_caption,
                               'labelloc': 't',
                               'labeljust': 'left',
                               'ordering': 'out'})
        dot.node_attr.update({'shape': 'box', 'ordering': 'out'})

        # list of edges for each subgraph; subgraphs will be created at the end.
        module_cluster_dict = defaultdict(list)

        for node_barcode, node in entries_to_plot.items():
            if node.is_buffer_layer and not show_buffer_layers:
                continue
            self._add_node_to_graphviz(node, dot, module_cluster_dict, vis_opt, show_buffer_layers)

        # Finally, set up the subgraphs.
        self._set_up_subgraphs(dot, module_cluster_dict)

        if in_notebook() and not save_only:
            display(dot)

        dot.render(vis_outpath, view=(not save_only))
        os.remove(vis_outpath)

    def _add_node_to_graphviz(self,
                              node: Union[TensorLogEntry, RolledTensorLogEntry],
                              graphviz_graph,
                              module_edge_dict: Dict,
                              vis_opt: str,
                              show_buffer_layers: bool = False):
        """Addes a node and its relevant edges to the graphviz figure.

        Args:
            node: node to add
            graphviz_graph: The graphviz object to add the node to.
            module_edge_dict: Dictionary of the module clusters.
            vis_opt: Whether to roll the graph or not
            show_buffer_layers: Whether to show the buffer layers
        """

        # Get the address, shape, color, and line style:

        node_address, node_shape, node_color = self._get_node_address_shape_color(node)
        node_bg_color = self._get_node_bg_color(node)

        if node.has_input_ancestor:
            line_style = 'solid'
        else:
            line_style = 'dashed'

        # Get the text for the node label:

        node_label = self._make_node_label(node, node_address, vis_opt)
        graphviz_graph.node(name=node.layer_label.replace(':', 'pass'),
                            label=node_label,
                            fontcolor=node_color,
                            color=node_color,
                            style=f"filled,{line_style}",
                            fillcolor=node_bg_color,
                            shape=node_shape,
                            ordering='out')

        self._add_edges_for_node(node, node_color, module_edge_dict, graphviz_graph, vis_opt, show_buffer_layers)

        # Finally, if it's the final output layer, force it to be on top for visual niceness.

        if node.is_last_output_layer:
            with graphviz_graph.subgraph() as s:
                s.attr(rank='sink')
                s.node(node.layer_label)

    def _get_node_address_shape_color(self,
                                      node: Union[TensorLogEntry, RolledTensorLogEntry]) -> Tuple[str, str, str]:
        """Gets the node shape, address, and color for the graphviz figure.

        Args:
            node: node to add

        Returns:
            node_address: address of the node
            node_shape: shape of the node
            node_color: color of the node
        """
        if node.is_bottom_level_submodule_output:
            if type(node) == TensorLogEntry:
                module_pass_exited = node.bottom_level_submodule_pass_exited
                module, pass_num = module_pass_exited.split(':')
                if self.module_num_passes[module] == 1:
                    node_address = module
                else:
                    node_address = module_pass_exited
            else:
                sample_module_pass = list(node.bottom_level_submodule_passes_exited)[0]
                module = sample_module_pass.split(':')[0]
                if (len(node.bottom_level_submodule_passes_exited) > 1) or self.module_num_passes[module] == 1:
                    node_address = module
                else:
                    node_address = sample_module_pass

            node_address = '<br/>@' + node_address
            node_shape = 'box'
            node_color = 'black'
        elif node.is_buffer_layer:
            node_address = '<br/>@' + node.buffer_address
            node_shape = 'box'
            node_color = self.BUFFER_NODE_COLOR
        else:
            node_address = ''
            node_shape = 'oval'
            node_color = 'black'

        return node_address, node_shape, node_color

    def _get_node_bg_color(self, node: Union[TensorLogEntry, RolledTensorLogEntry]) -> str:
        """Gets the node background color for the graphviz figure.

        Args:
            node: node to add

        Returns:
            node_bg_color: background color of the node
        """
        if node.is_input_layer:
            bg_color = self.INPUT_COLOR
        elif node.is_output_layer:
            bg_color = self.OUTPUT_COLOR
        elif node.is_terminal_bool_layer:
            bg_color = self.BOOL_NODE_COLOR
        elif node.computed_with_params:
            bg_color = self.PARAMS_NODE_BG_COLOR
        else:
            bg_color = self.DEFAULT_BG_COLOR
        return bg_color

    def _make_node_label(self,
                         node: Union[TensorLogEntry, RolledTensorLogEntry],
                         node_address: str,
                         vis_opt: str) -> str:
        """Gets the text for the graphviz node.
        """
        # Pass info:

        if (node.layer_passes_total > 1) and (vis_opt == 'unrolled'):
            pass_label = f":{node.pass_num}"
        elif (node.layer_passes_total > 1) and (vis_opt == 'rolled'):
            pass_label = f' (x{node.layer_passes_total})'
        else:
            pass_label = ''

        # Tensor shape info:

        if len(node.tensor_shape) > 1:
            tensor_shape_str = 'x'.join([str(x) for x in node.tensor_shape])
        elif len(node.tensor_shape) == 1:
            tensor_shape_str = f'x{node.tensor_shape[0]}'
        else:
            tensor_shape_str = 'x1'

        # Layer param info:

        param_label = self._make_param_label(node)

        tensor_fsize = node.tensor_fsize_nice
        node_title = f"<b>{node.layer_type}_{node.layer_type_num}_{node.layer_total_num}{pass_label}</b>"

        if node.is_terminal_bool_layer:
            label_text = str(node.atomic_bool_val).upper()
            bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
        else:
            bool_label = ''

        node_label = (f'<{bool_label}{node_title}<br/>{tensor_shape_str} '
                      f'({tensor_fsize}){param_label}{node_address}>')

        return node_label

    @staticmethod
    def _make_param_label(node: Union[TensorLogEntry, RolledTensorLogEntry]) -> str:
        """Makes the label for parameters of a node.
        """
        if node.num_param_tensors == 0:
            return ''

        each_param_shape = []
        for param_shape in node.parent_param_shapes:
            if len(param_shape) > 1:
                each_param_shape.append('x'.join([str(s) for s in param_shape]))
            else:
                each_param_shape.append('x1')

        param_label = "<br/>params: " + ', '.join([param_shape for param_shape in each_param_shape])
        return param_label

    def _add_edges_for_node(self,
                            parent_node: Union[TensorLogEntry, RolledTensorLogEntry],
                            node_color: str,
                            module_edge_dict: Dict,
                            graphviz_graph,
                            vis_opt: str = 'unrolled',
                            show_buffer_layers: bool = False):
        """Add the rolled-up edges for a node, marking for the edge which passes it happened for.

        Args:
            parent_node: The node to add edges for.
            node_color: Color of the node
            graphviz_graph: The graphviz graph object.
            module_edge_dict: Dictionary mapping each cluster to the edges it contains.
            vis_opt: Either 'unrolled' or 'rolled'
            show_buffer_layers: whether to show the buffer layers
        """
        for child_layer_label in parent_node.child_layers:
            if vis_opt == 'unrolled':
                child_node = self.layer_dict_main_keys[child_layer_label]
            elif vis_opt == 'rolled':
                child_node = self.layer_dict_rolled[child_layer_label]
            else:
                raise ValueError(f"vis_opt must be 'unrolled' or 'rolled', not {vis_opt}")

            if parent_node.has_input_ancestor:
                edge_style = 'solid'
            else:
                edge_style = 'dashed'

            edge_dict = {'tail_name': parent_node.layer_label.replace(':', 'pass'),
                         'head_name': child_layer_label.replace(':', 'pass'),
                         'color': node_color,
                         'fontcolor': node_color,
                         'style': edge_style,
                         'arrowsize': '.7',
                         'labelfontsize': '8'}

            # Mark with "if" in the case edge starts a cond branch
            if child_layer_label in parent_node.cond_branch_start_children:
                edge_dict['label'] = '<<FONT POINT-SIZE="18"><b><u>IF</u></b></FONT>>'

            # Label the arguments to the next node if multiple inputs
            self._label_node_arguments_if_needed(parent_node, child_node, edge_dict, show_buffer_layers)

            # Annotate passes for rolled node edge if it varies across passes
            if vis_opt == 'rolled':
                self._label_rolled_pass_nums(child_node, parent_node, edge_dict)

            # Add it to the appropriate module cluster (most nested one containing both nodes)
            containing_module = self._get_lowest_containing_module_for_two_nodes(parent_node, child_node)
            if containing_module != -1:
                module_edge_dict[containing_module].append(edge_dict)
            else:
                graphviz_graph.edge(**edge_dict)

            # Finally, add a backwards edge if both tensors have stored gradients.
            if vis_opt == 'unrolled':
                self._add_gradient_edge(parent_node, child_node, edge_style, containing_module,
                                        module_edge_dict, graphviz_graph)

    def _label_node_arguments_if_needed(self,
                                        parent_node: Union[TensorLogEntry, RolledTensorLogEntry],
                                        child_node: Union[TensorLogEntry, RolledTensorLogEntry],
                                        edge_dict: Dict,
                                        show_buffer_layers: bool = False):
        """Checks if a node has multiple non-commutative arguments, and if so, adds labels in edge_dict

        Args:
            parent_node: parent node
            child_node: child node
            edge_dict: dict of information about the edge
            show_buffer_layers: whether to show the buffer layers
        """
        if not self._check_whether_to_mark_arguments_on_edge(child_node, show_buffer_layers):
            return

        arg_labels = []
        for arg_type in ['args', 'kwargs']:
            for arg_loc, arg_label in child_node.parent_layer_arg_locs[arg_type].items():
                if (parent_node.layer_label == arg_label) or (parent_node.layer_label in arg_label):
                    arg_labels.append(f"{arg_type[:-1]} {str(arg_loc)}")

        arg_labels = '\n'.join(arg_labels)
        arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_labels}</b></FONT>>"
        if 'label' not in edge_dict:
            edge_dict['label'] = arg_label
        else:
            edge_dict['label'] = edge_dict['label'] + '\n' + arg_label

    def _check_whether_to_mark_arguments_on_edge(self,
                                                 child_node: Union[TensorLogEntry, RolledTensorLogEntry],
                                                 show_buffer_layers: bool = False):
        if child_node.layer_type in self.COMMUTE_FUNCS:
            return False

        if type(child_node) == TensorLogEntry:
            return self._check_whether_to_mark_arguments_on_unrolled_edge(child_node, show_buffer_layers)
        elif type(child_node) == RolledTensorLogEntry:
            return self._check_whether_to_mark_arguments_on_rolled_edge(child_node)

    def _check_whether_to_mark_arguments_on_unrolled_edge(self,
                                                          child_node: TensorLogEntry,
                                                          show_buffer_layers: bool = False):
        num_parents_shown = len(child_node.parent_layers)

        if not show_buffer_layers:
            num_parents_shown -= sum([int(self[parent].is_buffer_layer) for parent in child_node.parent_layers])

        if num_parents_shown > 1:
            return True
        else:
            return False

    def _check_whether_to_mark_arguments_on_rolled_edge(self,
                                                        child_node: RolledTensorLogEntry,
                                                        show_buffer_layers: bool = False):
        for pass_num, pass_parents in child_node.parent_layers_per_pass.items():
            num_parents_shown = len(pass_parents)
            if not show_buffer_layers:
                num_parents_shown -= sum([int(self.layer_dict_rolled[parent].is_buffer_layer)
                                          for parent in pass_parents])
            if num_parents_shown > 1:
                return True

        return False

    @staticmethod
    def _label_rolled_pass_nums(child_node: RolledTensorLogEntry,
                                parent_node: RolledTensorLogEntry,
                                edge_dict: Dict):
        """Adds labels for the pass numbers to the edge dict for rolled nodes.

        Args:
            child_node: child node
            parent_node: parent node
            edge_dict: dictionary of edge information
        """
        parent_pass_nums = parent_node.child_passes_per_layer[child_node.layer_label]
        child_pass_nums = child_node.parent_passes_per_layer[parent_node.layer_label]
        if parent_node.edges_vary_across_passes:
            edge_dict['taillabel'] = f'  Out {int_list_to_compact_str(parent_pass_nums)}  '

        # Mark the head label with the argument if need be:
        if child_node.edges_vary_across_passes and not child_node.is_output_layer:
            edge_dict['headlabel'] = f'  In {int_list_to_compact_str(child_pass_nums)}  '

    @staticmethod
    def _get_lowest_containing_module_for_two_nodes(node1: Union[TensorLogEntry, RolledTensorLogEntry],
                                                    node2: Union[TensorLogEntry, RolledTensorLogEntry]):
        """Utility function to get the lowest-level module that contains two nodes, to know where to put the edge.

        Args:
            node1: The first node.
            node2: The second node.

        Returns:
            Lowest-level module pass containing two nodes.
        """
        node1_modules = node1.containing_modules_origin_nested[:]
        node2_modules = node2.containing_modules_origin_nested[:]

        if (len(node1_modules) == 0) or (len(node2_modules) == 0) or (node1_modules[0] != node2_modules[0]):
            return -1  # no submodule contains them both.

        containing_module = node1_modules[0]
        for m in range(min([len(node1_modules), len(node2_modules)])):
            if node1_modules[m] != node2_modules[m]:
                break
            containing_module = node1_modules[m]
        return containing_module

    def _add_gradient_edge(self,
                           parent_layer,
                           child_layer,
                           edge_style,
                           containing_module,
                           module_edge_dict,
                           graphviz_graph):
        """Adds a backwards edge if both layers have saved gradients, showing the backward pass.
        """
        if parent_layer.has_saved_grad and child_layer.has_saved_grad:
            edge_dict = {'tail_name': child_layer.layer_label.replace(':', 'pass'),
                         'head_name': parent_layer.layer_label.replace(':', 'pass'),
                         'color': self.GRADIENT_ARROW_COLOR,
                         'fontcolor': self.GRADIENT_ARROW_COLOR,
                         'style': edge_style,
                         'arrowsize': '.7',
                         'labelfontsize': '8'}
            if containing_module != -1:
                module_edge_dict[containing_module].append(edge_dict)
            else:
                graphviz_graph.edge(**edge_dict)

    def _set_up_subgraphs(self,
                          graphviz_graph,
                          module_edge_dict: Dict[str, List]):
        """Given a dictionary specifying the edges in each cluster and the graphviz graph object,
        set up the nested subgraphs and the nodes that should go inside each of them. There will be some tricky
        recursive logic to set up the nested context managers.

        Args:
            graphviz_graph: Graphviz graph object.
            module_edge_dict: Dictionary mapping each cluster to the list of edges it contains, with each
                edge specified as a dict with all necessary arguments for creating that edge.
        """
        module_submodule_dict = self.module_pass_children.copy()
        subgraphs = self.top_level_module_passes[:]

        # Get the max module nesting depth:

        max_nesting_depth = self._get_max_nesting_depth(subgraphs,
                                                        module_edge_dict,
                                                        module_submodule_dict)

        subgraph_stack = [[subgraph] for subgraph in subgraphs]
        nesting_depth = 0
        while len(subgraph_stack) > 0:
            parent_graph_list = subgraph_stack.pop(0)
            self._setup_subgraphs_recurse(graphviz_graph,
                                          parent_graph_list,
                                          module_edge_dict,
                                          module_submodule_dict,
                                          subgraph_stack,
                                          nesting_depth,
                                          max_nesting_depth)

    def _setup_subgraphs_recurse(self,
                                 starting_subgraph,
                                 parent_graph_list: List,
                                 module_edge_dict,
                                 module_submodule_dict,
                                 subgraph_stack,
                                 nesting_depth,
                                 max_nesting_depth):
        """Utility function to crawl down several layers deep into nested subgraphs.

        Args:
            starting_subgraph: The subgraph we're starting from.
            parent_graph_list: List of parent graphs.
            module_edge_dict: Dict mapping each cluster to its edges.
            module_submodule_dict: Dict mapping each cluster to its subclusters.
            subgraph_stack: Stack of subgraphs to look at.
            nesting_depth: Nesting depth so far.
            max_nesting_depth: The total depth of the subgraphs.
        """
        subgraph_name_w_pass = parent_graph_list[nesting_depth]
        subgraph_module = subgraph_name_w_pass.split(':')[0]
        cluster_name = f"cluster_{subgraph_module}"
        module_type = self.module_types[subgraph_module]
        if self.module_num_passes[subgraph_module] > 1:
            subgraph_title = subgraph_name_w_pass
        else:
            subgraph_title = subgraph_module

        if nesting_depth < len(parent_graph_list) - 1:  # we haven't gotten to the bottom yet, keep going.
            with starting_subgraph.subgraph(name=cluster_name) as s:
                self._setup_subgraphs_recurse(s, parent_graph_list,
                                              module_edge_dict, module_submodule_dict, subgraph_stack,
                                              nesting_depth + 1, max_nesting_depth)

        else:  # we made it, make the subgraph and add all edges.
            with starting_subgraph.subgraph(name=cluster_name) as s:
                nesting_fraction = (max_nesting_depth - nesting_depth) / max_nesting_depth
                pen_width = self.MIN_MODULE_PENWIDTH + nesting_fraction * self.PENWIDTH_RANGE
                s.attr(label=f"<<B>@{subgraph_title}</B><br align='left'/>({module_type})<br align='left'/>>",
                       labelloc='b',
                       penwidth=str(pen_width))
                subgraph_edges = module_edge_dict[subgraph_name_w_pass]
                for edge_dict in subgraph_edges:
                    s.edge(**edge_dict)
                subgraph_children = module_submodule_dict[subgraph_name_w_pass]
                for subgraph_child in subgraph_children:  # it's weird but have to go in reverse order.
                    subgraph_stack.append(parent_graph_list[:] + [subgraph_child])

    @staticmethod
    def _get_max_nesting_depth(top_modules,
                               module_edge_dict,
                               module_submodule_dict):
        """Utility function to get the max nesting depth of the nested modules in the network; works by
        recursively crawling down the stack of modules till it hits one with no children and at least one edge.

        Args:
            top_modules: modules at highest level of nesting
            module_edge_dict: Edges in each module.
            module_submodule_dict: Mapping from each module to any children.

        Returns:
            Max nesting depth.
        """
        max_nesting_depth = 1
        module_stack = [(graph, 1) for graph in top_modules]

        while len(module_stack) > 0:
            module, module_depth = module_stack.pop()
            module_edges = module_edge_dict[module]
            module_submodules = module_submodule_dict[module]

            if (len(module_edges) == 0) and (len(module_submodules) == 0):  # can ignore if no edges and no children.
                continue
            elif (len(module_edges) > 0) and (len(module_submodules) == 0):
                max_nesting_depth = max([module_depth, max_nesting_depth])
            elif (len(module_edges) == 0) and (len(module_submodules) > 0):
                module_stack.extend([(module_child, module_depth + 1) for module_child in module_submodules])
            else:
                max_nesting_depth = max([module_depth, max_nesting_depth])
                module_stack.extend([(module_child, module_depth + 1) for module_child in module_submodules])
        return max_nesting_depth

    # ********************************************
    # *************** Validation *****************
    # ********************************************

    def validate_saved_activations(self, ground_truth_output_tensors: List[torch.Tensor],
                                   verbose: bool = False) -> bool:
        """Starting from outputs and internally terminated tensors, checks whether computing their values from the saved
        values of their input tensors yields their actually saved values, and whether computing their values from
        their parent tensors yields their saved values.

        Returns:
            True if it passes the tests, False otherwise.
        """
        # First check that the ground truth output tensors are accurate:
        for i, output_layer_label in enumerate(self.output_layers):
            output_layer = self[output_layer_label]
            if not torch.equal(output_layer.tensor_contents, ground_truth_output_tensors[i]):
                print(f"The {i}th output layer, {output_layer_label}, does not match the ground truth output tensor.")
                return False

        # Validate the parents of each validated layer.
        validated_child_edges_for_each_layer = defaultdict(set)
        validated_layers = set(self.output_layers + self.internally_terminated_layers)
        layers_to_validate_parents_for = list(validated_layers)

        while len(layers_to_validate_parents_for) > 0:
            layer_to_validate_parents_for = layers_to_validate_parents_for.pop(0)
            parent_layers_valid = self.validate_parents_of_saved_layer(layer_to_validate_parents_for,
                                                                       validated_layers,
                                                                       validated_child_edges_for_each_layer,
                                                                       layers_to_validate_parents_for,
                                                                       verbose)
            if not parent_layers_valid:
                return False

        if len(validated_layers) < len(self.layer_labels):
            print(f"All saved activations were accurate, but some layers were not reached (check that "
                  f"child args logged accurately): {set(self.layer_labels) - validated_layers}")
            return False

        return True

    def validate_parents_of_saved_layer(self,
                                        layer_to_validate_parents_for_label: str,
                                        validated_layers: Set[str],
                                        validated_child_edges_for_each_layer: Dict[str, Set[str]],
                                        layers_to_validate_parents_for: List[str],
                                        verbose: bool = False) -> bool:
        """Given a layer, checks that 1) all parent tensors appear properly in the saved arguments for that layer,
        2) that executing the function for that layer with the saved parent layer activations yields the
        ground truth activation values for that layer, and 3) that plugging in "perturbed" values for each
        child layer yields values different from the saved activations for that layer.

        Args:
            layer_to_validate_parents_for_label:
            validated_layers:
            validated_child_edges_for_each_layer:
            layers_to_validate_parents_for:
            verbose: whether to print warning messages
        """
        layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]

        # Check that the arguments are logged correctly:
        if not self._check_layer_arguments_logged_correctly(layer_to_validate_parents_for_label):
            print(f"Parent arguments for layer {layer_to_validate_parents_for_label} are not logged properly; "
                  f"either a parent wasn't logged as an argument, or was logged an extra time")
            return False

        # Check that executing the function based on the actual saved values of the parents yields the saved
        # values of the layer itself:

        if not self._check_whether_func_on_saved_parents_yields_saved_tensor(layer_to_validate_parents_for_label,
                                                                             perturb=False):
            return False

        # Check that executing the layer's function on the wrong version of the saved parent tensors
        # yields the wrong tensors, when each saved tensor is perturbed in turn:

        for perturb_layer in layer_to_validate_parents_for.parent_layers:
            if layer_to_validate_parents_for.func_applied_name in self.FUNCS_NOT_TO_PERTURB_IN_VALIDATION:
                continue
            if not self._check_whether_func_on_saved_parents_yields_saved_tensor(layer_to_validate_parents_for_label,
                                                                                 perturb=True,
                                                                                 layers_to_perturb=[perturb_layer],
                                                                                 verbose=verbose):
                return False

        # Log that each parent layer has been validated for this source layer.

        for parent_layer_label in layer_to_validate_parents_for.parent_layers:
            parent_layer = self[parent_layer_label]
            validated_child_edges_for_each_layer[parent_layer_label].add(layer_to_validate_parents_for_label)
            if validated_child_edges_for_each_layer[parent_layer_label] == set(parent_layer.child_layers):
                validated_layers.add(parent_layer_label)
                if not (parent_layer.is_input_layer or parent_layer.is_buffer_layer):
                    layers_to_validate_parents_for.append(parent_layer_label)

        return True

    def _check_layer_arguments_logged_correctly(self,
                                                target_layer_label: str) -> bool:
        """Check whether the activations of the parent layers match the saved arguments of
        the target layer, and that the argument locations have been logged correctly.

        Args:
            target_layer_label: Layer to check

        Returns:
            True if arguments logged accurately, False otherwise
        """
        target_layer = self[target_layer_label]

        # Make sure that all parent layers appear in at least one argument and that no extra layers appear:
        parent_layers_in_args = set()
        for arg_type in ['args', 'kwargs']:
            parent_layers_in_args.update(list(target_layer.parent_layer_arg_locs[arg_type].values()))
        if parent_layers_in_args != set(target_layer.parent_layers):
            return False

        argtype_dict = {'args': (enumerate, 'creation_args'),
                        'kwargs': (lambda x: x.items(), 'creation_kwargs')}

        # Check for each parent layer that it is logged as a saved argument when it matches an argument, and
        # is not logged when it does not match a saved argument.

        for parent_layer_label in target_layer.parent_layers:
            parent_layer = self[parent_layer_label]
            for arg_type in ['args', 'kwargs']:
                iterfunc, argtype_field = argtype_dict[arg_type]
                for key, val in iterfunc(getattr(target_layer, argtype_field)):
                    validation_correct_for_arg_and_layer = self._validate_layer_against_arg(target_layer,
                                                                                            parent_layer,
                                                                                            arg_type,
                                                                                            key,
                                                                                            val)
                    if not validation_correct_for_arg_and_layer:
                        return False
        return True

    def _validate_layer_against_arg(self, target_layer, parent_layer, arg_type, key, val):
        if type(val) in [list, tuple]:
            for v, subval in enumerate(val):
                argloc_key = (key, v)
                validation_correct_for_arg_and_layer = self._check_arglocs_correct_for_arg(target_layer,
                                                                                           parent_layer,
                                                                                           arg_type,
                                                                                           argloc_key,
                                                                                           subval)
                if not validation_correct_for_arg_and_layer:
                    return False

        elif type(val) == dict:
            for subkey, subval in val.items():
                argloc_key = (key, subkey)
                validation_correct_for_arg_and_layer = self._check_arglocs_correct_for_arg(target_layer,
                                                                                           parent_layer,
                                                                                           arg_type,
                                                                                           argloc_key,
                                                                                           subval)
                if not validation_correct_for_arg_and_layer:
                    return False
        else:
            argloc_key = key
            validation_correct_for_arg_and_layer = self._check_arglocs_correct_for_arg(target_layer,
                                                                                       parent_layer,
                                                                                       arg_type,
                                                                                       argloc_key,
                                                                                       val)
            if not validation_correct_for_arg_and_layer:
                return False

        return True

    @staticmethod
    def _check_arglocs_correct_for_arg(target_layer: TensorLogEntry,
                                       parent_layer: TensorLogEntry,
                                       arg_type: str,
                                       argloc_key: Union[str, tuple],
                                       saved_arg_val: Any):
        """For a given layer and an argument to its child layer, checks that it is logged correctly:
        that is, that it's logged as an argument if it matches, and is not logged as an argument if it doesn't match.
        """
        target_layer_label = target_layer.layer_label
        parent_layer_label = parent_layer.layer_label
        parent_activations = parent_layer.tensor_contents

        if type(saved_arg_val) == torch.Tensor:
            parent_layer_matches_arg = torch.equal(saved_arg_val, parent_activations)
        else:
            parent_layer_matches_arg = False
        parent_layer_logged_as_arg = ((argloc_key in target_layer.parent_layer_arg_locs[arg_type]) and
                                      (target_layer.parent_layer_arg_locs[arg_type][argloc_key] == parent_layer_label))

        if parent_layer_matches_arg and not parent_layer_logged_as_arg:
            print(f"Parent {parent_layer_label} of {target_layer_label} has activations that match "
                  f"{arg_type} {argloc_key} for {target_layer_label}, but is not logged as "
                  f"such in parent_layer_arg_locs.")
            return False

        if not parent_layer_matches_arg and parent_layer_logged_as_arg:
            print(f"Parent {parent_layer_label} of {target_layer_label} is logged as {arg_type} {argloc_key} to "
                  f"{target_layer_label}, but its saved activations don't match the saved argument.")
            return False

        return True

    def _check_whether_func_on_saved_parents_yields_saved_tensor(self,
                                                                 layer_to_validate_parents_for_label: str,
                                                                 perturb: bool = False,
                                                                 layers_to_perturb: List[str] = None,
                                                                 verbose: bool = False) -> bool:
        """Checks whether executing the saved function for a layer on the saved value of its parent layers
        in fact yields the saved activations for that layer.

        Args:
            layer_to_validate_parents_for_label: label of the layer to check the saved activations
            perturb: whether to perturb the saved activations
            layers_to_perturb: layers for which to perturb the saved activations

        Returns:
            True if the activations match, False otherwise
        """
        if layers_to_perturb is None:
            layers_to_perturb = []

        layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]

        # Prepare input arguments: keep the ones that should just be kept, perturb those that should be perturbed

        input_args = self._prepare_input_args_for_validating_layer(layer_to_validate_parents_for,
                                                                   layers_to_perturb)

        # set the saved rng value:
        layer_func = layer_to_validate_parents_for.func_applied
        current_rng_states = log_current_rng_states()
        set_rng_from_saved_states(layer_to_validate_parents_for.func_rng_states)
        recomputed_output = layer_func(*input_args['args'], **input_args['kwargs'])
        set_rng_from_saved_states(current_rng_states)

        if layer_func.__name__ in ['__setitem__', 'zero_', '__delitem__']:  # TODO: fix this
            recomputed_output = input_args['args'][0]

        if type(recomputed_output) in [list, tuple]:
            recomputed_output = recomputed_output[layer_to_validate_parents_for.iterable_output_index]

        if not (torch.equal(recomputed_output, layer_to_validate_parents_for.tensor_contents)) and not perturb:
            print(f"Saved activations for layer {layer_to_validate_parents_for_label} do not match the "
                  f"values computed based on the parent layers {layer_to_validate_parents_for.parent_layers}.")
            return False

        if torch.equal(recomputed_output, layer_to_validate_parents_for.tensor_contents) and perturb:
            return self._posthoc_perturb_check(layer_to_validate_parents_for, layers_to_perturb, verbose)

        return True

    def _prepare_input_args_for_validating_layer(self,
                                                 layer_to_validate_parents_for: TensorLogEntry,
                                                 layers_to_perturb: List[str]) -> Dict:
        """Prepares the input arguments for validating the saved activations of a layer.

        Args:
            layer_to_validate_parents_for: Layer being checked.
            layers_to_perturb: Layers for which to perturb the saved activations.

        Returns:
            Dict of input arguments.
        """
        input_args = {'args': list(layer_to_validate_parents_for.creation_args),
                      'kwargs': layer_to_validate_parents_for.creation_kwargs.copy()}

        # Swap in saved parent activations:

        for arg_type in ['args', 'kwargs']:
            for key, parent_layer_arg in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type].items():
                if parent_layer_arg in layers_to_perturb:
                    parent_layer_func_values = self._perturb_layer_activations(self[parent_layer_arg].tensor_contents)
                else:
                    parent_layer_func_values = safe_copy(self[parent_layer_arg].tensor_contents)

                if type(key) != tuple:
                    input_args[arg_type][key] = parent_layer_func_values
                else:
                    input_args[arg_type][key[0]] = tuple_tolerant_assign(input_args[arg_type][key[0]],
                                                                         key[1],
                                                                         parent_layer_func_values)

        return input_args

    @staticmethod
    def _perturb_layer_activations(activations: torch.Tensor) -> torch.Tensor:
        """
        Perturbs the values of a saved tensor.

        Args:
            activations: Tensor of activation values.

        Returns:
            Perturbed version of saved tensor
        """
        device = activations.device
        if activations.dtype in [torch.int, torch.long, torch.short, torch.uint8,
                                 torch.int8, torch.int16, torch.int32, torch.int64]:
            tensor_unique_vals = torch.unique(activations)
            if len(tensor_unique_vals) > 1:
                perturbed_activations = torch.randint(activations.min(), activations.max() + 1,
                                                      size=activations.shape, device=device)
            else:
                perturbed_activations = torch.randint(-10, 11, size=activations.shape, device=device)
        elif activations.dtype == torch.bool:
            perturbed_activations = torch.randint(0, 2, size=activations.shape, device=device).bool()
        else:
            perturbed_activations = torch.randn_like(activations.float(), device=device)

        return perturbed_activations

    def _posthoc_perturb_check(self,
                               layer_to_validate_parents_for: TensorLogEntry,
                               layers_to_perturb: List[str],
                               verbose: bool = False) -> bool:
        """If a layer fails the "perturbation check"--that is, if perturbing the values of parent
        layers doesn't change the values relative to the layer's saved values--checks whether one of the
        remaining arguments is a "special" tensor, such as all-ones or all-zeros, such that perturbing a tensor
        wouldn't necessarily change the output of the layer.

        Args:
            layer_to_validate_parents_for: layer being checked.
            layers_to_perturb: parent layers being perturbed

        Returns:
            True if there's an "excuse" for the perturbation failing, False otherwise.
        """
        arg_type_dict = {'args': (enumerate, 'creation_args'),
                         'kwargs': (lambda x: x.items(), 'creation_kwargs')}

        layer_to_validate_parents_for_label = layer_to_validate_parents_for.layer_label
        for arg_type in ['args', 'kwargs']:
            iterfunc, fieldname = arg_type_dict[arg_type]
            for key, val in iterfunc(getattr(layer_to_validate_parents_for, fieldname)):
                # Skip if it's the argument itself:
                if ((key in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type]) and
                        (layer_to_validate_parents_for.parent_layer_arg_locs[arg_type][key]) in layers_to_perturb):
                    continue
                arg_is_special = self._check_if_arg_is_special_val(val)
                if arg_is_special:
                    if verbose:
                        print(f"Activations for layer {layer_to_validate_parents_for_label} do not change when "
                              f"values for {layers_to_perturb} are changed (out of parent "
                              f"layers {layer_to_validate_parents_for.parent_layers}), but {arg_type[:-1]} {key} is "
                              f"all zeros or all-ones, so validation still succeeds...")
                    return True

        print(f"Activations for layer {layer_to_validate_parents_for_label} do not change when "
              f"values for {layers_to_perturb} are changed (out of parent "
              f"layers {layer_to_validate_parents_for.parent_layers}), and the other "
              f"arguments are not \"special\" (all-ones or all-zeros) tensors.")
        return False

    @staticmethod
    def _check_if_arg_is_special_val(val: Union[torch.Tensor, Any]):
        # If it's one of the other arguments, check if it's all zeros or all ones:
        if type(val) != torch.Tensor:
            val = torch.Tensor(val)
        if torch.all(val == 0) or torch.all(val == 1):
            return True
        else:
            return False

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
        or their module address. It should say so and tell them which labels are valid.
        """
        if ix in self.layer_dict_all_keys:
            return self.layer_dict_all_keys[ix]

        keys_with_substr = [key for key in self.layer_dict_all_keys if ix in str(key)]
        if len(keys_with_substr) == 1:
            return self.layer_dict_all_keys[keys_with_substr[0]]

        self._give_user_feedback_about_lookup_key(ix, 'get_one_item')

    def _give_user_feedback_about_lookup_key(self,
                                             key: Union[int, str],
                                             mode: str):
        """For __getitem__ and get_op_nums_from_user_labels, gives the user feedback about the user key
        they entered if it doesn't yield any matches.

        Args:
            key: Lookup key used by the user.
        """
        if (type(key) == int) and (key >= len(self.layer_list) or key < -len(self.layer_list)):
            raise ValueError(f"You specified the layer with index {key}, but there are only {len(self.layer_list)} "
                             f"layers; please specify an index in the range "
                             f"-{len(self.layer_list)} - {len(self.layer_list) - 1}.")

        if key in self.module_addresses:
            module_num_passes = self.module_num_passes[key]
            raise ValueError(f"You specified output of module {key}, but it has {module_num_passes} passes; "
                             f"please specify e.g. {key}:2 for the second pass of {key}.")

        if key.split(':')[0] in self.module_addresses:
            module, pass_num = key.split(':')
            module_num_passes = self.module_num_passes[module]
            raise ValueError(f"You specified module {module} pass {pass_num}, but {module} only has "
                             f"{module_num_passes} passes; specify a lower number.")

        if key in self.layer_labels_no_pass:
            layer_num_passes = self.layer_num_passes[key]
            raise ValueError(f"You specified output of layer {key}, but it has {layer_num_passes} passes; "
                             f"please specify e.g. {key}:2 for the second pass of {key}.")

        if key.split(':')[0] in self.layer_labels_no_pass:
            layer_label, pass_num = key.split(':')
            layer_num_passes = self.layer_num_passes[layer_label]
            raise ValueError(f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
                             f"{layer_num_passes} passes. Specify a lower number.")

        raise ValueError(self._get_lookup_help_str(key, mode))

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

        # General info

        s += f"\n\tRandom seed: {self.random_seed_used}"
        s += f"\n\tTime elapsed: {np.round(self.elapsed_time_total, 3)}s " \
             f"({np.round(self.elapsed_time_torchlens_logging, 3)}s spent logging)"

        # Overall model structure

        s += "\n\tStructure:"
        if self.model_is_recurrent:
            s += f"\n\t\t- recurrent (at most {self.model_max_recurrent_loops} loops)"
        else:
            s += f"\n\t\t- purely feedforward, no recurrence"

        if self.model_is_branching:
            s += "\n\t\t- with branching"
        else:
            s += "\n\t\t- no branching"

        if self.model_has_conditional_branching:
            s += "\n\t\t- with conditional (if-then) branching"
        else:
            s += "\n\t\t- no conditional (if-then) branching"

        if len(self.buffer_layers) > 0:
            s += f"\n\t\t- contains {len(self.buffer_layers)} buffer layers"

        s += f"\n\t\t- {len(self.module_addresses)} total modules"

        # Model tensors:

        s += "\n\tTensor info:"
        s += f"\n\t\t- {len(self.layer_list)} total tensors ({self.tensor_fsize_total_nice}) computed in forward pass."
        s += f"\n\t\t- {self.num_tensors_saved} tensors ({self.tensor_fsize_saved_nice}) saved in log."

        # Model parameters:

        s += f"\n\tParameters: {self.total_param_layers} parameter operations ({self.total_params} params total; " \
             f"{self.total_params_fsize_nice})"

        # Print the module hierarchy.
        s += f"\n\tModule Hierarchy:"
        s += self._module_hierarchy_str()

        # Now print all layers.
        s += f"\n\tLayers:"
        for layer_ind, layer_barcode in enumerate(self.layer_labels):
            pass_num = self.layer_dict_main_keys[layer_barcode].pass_num
            total_passes = self.layer_dict_main_keys[layer_barcode].layer_passes_total
            if total_passes > 1:
                pass_str = f" ({pass_num}/{total_passes} passes)"
            else:
                pass_str = ''
            s += f"\n\t\t({layer_ind}) {layer_barcode} {pass_str}"

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

    def _get_lookup_help_str(self,
                             layer_label: Union[int, str],
                             mode: str) -> str:
        """Generates a help string to be used in error messages when indexing fails.
        """
        sample_layer1 = random.choice(self.layer_labels_w_pass)
        sample_layer2 = random.choice(self.layer_labels_no_pass)
        if len(self.module_addresses) > 0:
            sample_module1 = random.choice(self.module_addresses)
            sample_module2 = random.choice(self.module_passes)
        else:
            sample_module1 = 'features.3'
            sample_module2 = 'features.4:2'
        module_str = f"(e.g., {sample_module1}, {sample_module2})"
        if mode == 'get_one_item':
            msg = "e.g., 'pool' will grab the maxpool2d or avgpool2d layer, 'maxpool' will grab the 'maxpool2d' " \
                  "layer, etc., but there must be only one such matching layer"
        elif mode == 'query_multiple':
            msg = "e.g., 'pool' will grab all maxpool2d or avgpool2d layers, 'maxpool' will grab all 'maxpool2d' " \
                  "layers, etc."
        else:
            raise ValueError("mode must be either get_one_item or query_multiple")
        help_str = (f"Layer {layer_label} not recognized; please specify either "
                    f"\n\n\t1) an integer giving the ordinal position of the layer "
                    f"(e.g. 2 for 3rd layer, -4 for fourth-to-last), "
                    f"\n\t2) the layer label (e.g., {sample_layer1}, {sample_layer2}), "
                    f"\n\t3) the module address {module_str}"
                    f"\n\t4) A substring of any desired layer label ({msg})."
                    f"\n\n(Label meaning: conv2d_3_4:2 means the second pass of the third convolutional layer, "
                    f"and fourth layer overall in the model.)")
        return help_str

    def _module_hierarchy_str(self):
        """
        Utility function to print the nested module hierarchy.
        """
        s = ''
        for module_pass in self.top_level_module_passes:
            module, pass_num = module_pass.split(':')
            s += f"\n\t\t{module}"
            if self.module_num_passes[module] > 1:
                s += f':{pass_num}'
            s += self._module_hierarchy_str_helper(module_pass, 1)
        return s

    def _module_hierarchy_str_helper(self, module_pass, level):
        """
        Helper function for _module_hierarchy_str.
        """
        s = ''
        any_grandchild_modules = any([len(self.module_pass_children[submodule_pass]) > 0
                                      for submodule_pass in self.module_pass_children[module_pass]])
        if any_grandchild_modules or len(self.module_pass_children[module_pass]) == 0:
            for submodule_pass in self.module_pass_children[module_pass]:
                submodule, pass_num = submodule_pass.split(':')
                s += f"\n\t\t{'    ' * level}{submodule}"
                if self.module_num_passes[submodule] > 1:
                    s += f":{pass_num}"
                s += self._module_hierarchy_str_helper(submodule_pass, level + 1)
        else:
            submodule_list = []
            for submodule_pass in self.module_pass_children[module_pass]:
                submodule, pass_num = submodule_pass.split(':')
                if self.module_num_passes[submodule] == 1:
                    submodule_list.append(submodule)
                else:
                    submodule_list.append(submodule_pass)
            s += self.pretty_print_list_w_line_breaks(submodule_list,
                                                      line_break_every=8,
                                                      indent_chars=f"\t\t{'    ' * level}")
        return s

    def __repr__(self):
        return self.__str__()
