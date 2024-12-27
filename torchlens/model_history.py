# This file is for defining the ModelHistory class that stores the representation of the forward pass.
import copy
from functools import wraps
import inspect
import itertools as it
import os
import random
import time
import types
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

import graphviz
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch import nn

from torchlens.constants import (
    MODEL_HISTORY_FIELD_ORDER,
    ORIG_TORCH_FUNCS,
    TENSOR_LOG_ENTRY_FIELD_ORDER,
)
from torchlens.helper_funcs import (
    get_attr_values_from_tensor_list,
    get_tensor_memory_amount,
    get_vars_of_type_from_obj,
    human_readable_size,
    identity,
    in_notebook,
    int_list_to_compact_str,
    is_iterable,
    log_current_rng_states,
    make_random_barcode,
    make_short_barcode_from_input,
    make_var_iterable,
    nested_getattr,
    nested_assign,
    print_override,
    remove_entry_from_list,
    safe_copy,
    set_random_seed,
    set_rng_from_saved_states,
    tuple_tolerant_assign,
    iter_accessible_attributes,
    remove_attributes_starting_with_str,
    tensor_nanequal,
    tensor_all_nan,
    clean_to,
    index_nested,
)

# todo add saved_layer field, remove the option to only keep saved layers
# Define some constants.
print_funcs = ["__repr__", "__str__", "_str"]
funcs_not_to_log = ["numpy", "__array__", "size", "dim"]


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
        self.tensor_label_raw = fields_dict["tensor_label_raw"]
        self.layer_label_raw = fields_dict["layer_label_raw"]
        self.operation_num = fields_dict["operation_num"]
        self.realtime_tensor_num = fields_dict["realtime_tensor_num"]
        self.source_model_history = fields_dict["source_model_history"]
        self.pass_finished = fields_dict["pass_finished"]

        # Label info:
        self.layer_label = fields_dict["layer_label"]
        self.layer_label_short = fields_dict["layer_label_short"]
        self.layer_label_w_pass = fields_dict["layer_label_w_pass"]
        self.layer_label_w_pass_short = fields_dict["layer_label_w_pass_short"]
        self.layer_label_no_pass = fields_dict["layer_label_no_pass"]
        self.layer_label_no_pass_short = fields_dict["layer_label_no_pass_short"]
        self.layer_type = fields_dict["layer_type"]
        self.layer_type_num = fields_dict["layer_type_num"]
        self.layer_total_num = fields_dict["layer_total_num"]
        self.pass_num = fields_dict["pass_num"]
        self.layer_passes_total = fields_dict["layer_passes_total"]
        self.lookup_keys = fields_dict["lookup_keys"]

        # Saved tensor info:
        self.tensor_contents = fields_dict["tensor_contents"]
        self.has_saved_activations = fields_dict["has_saved_activations"]
        self.output_device = fields_dict["output_device"]
        self.activation_postfunc = fields_dict["activation_postfunc"]
        self.detach_saved_tensor = fields_dict["detach_saved_tensor"]
        self.function_args_saved = fields_dict["function_args_saved"]
        self.creation_args = fields_dict["creation_args"]
        self.creation_kwargs = fields_dict["creation_kwargs"]
        self.tensor_shape = fields_dict["tensor_shape"]
        self.tensor_dtype = fields_dict["tensor_dtype"]
        self.tensor_fsize = fields_dict["tensor_fsize"]
        self.tensor_fsize_nice = fields_dict["tensor_fsize_nice"]

        # Dealing with getitem complexities
        self.was_getitem_applied = fields_dict["was_getitem_applied"]
        self.children_tensor_versions = fields_dict["children_tensor_versions"]

        # Saved gradient info
        self.grad_contents = fields_dict["grad_contents"]
        self.save_gradients = fields_dict["save_gradients"]
        self.has_saved_grad = fields_dict["has_saved_grad"]
        self.grad_shapes = fields_dict["grad_shapes"]
        self.grad_dtypes = fields_dict["grad_dtypes"]
        self.grad_fsizes = fields_dict["grad_fsizes"]
        self.grad_fsizes_nice = fields_dict["grad_fsizes_nice"]

        # Function call info:
        self.func_applied = fields_dict["func_applied"]
        self.func_applied_name = fields_dict["func_applied_name"]
        self.func_call_stack = fields_dict["func_call_stack"]
        self.func_time_elapsed = fields_dict["func_time_elapsed"]
        self.func_rng_states = fields_dict["func_rng_states"]
        self.func_argnames = fields_dict["func_argnames"]
        self.num_func_args_total = fields_dict["num_func_args_total"]
        self.num_position_args = fields_dict["num_position_args"]
        self.num_keyword_args = fields_dict["num_keyword_args"]
        self.func_position_args_non_tensor = fields_dict[
            "func_position_args_non_tensor"
        ]
        self.func_keyword_args_non_tensor = fields_dict["func_keyword_args_non_tensor"]
        self.func_all_args_non_tensor = fields_dict["func_all_args_non_tensor"]
        self.function_is_inplace = fields_dict["function_is_inplace"]
        self.gradfunc = fields_dict["gradfunc"]
        self.is_part_of_iterable_output = fields_dict["is_part_of_iterable_output"]
        self.iterable_output_index = fields_dict["iterable_output_index"]

        # Param info:
        self.computed_with_params = fields_dict["computed_with_params"]
        self.parent_params = fields_dict["parent_params"]
        self.parent_param_barcodes = fields_dict["parent_param_barcodes"]
        self.parent_param_passes = fields_dict["parent_param_passes"]
        self.num_param_tensors = fields_dict["num_param_tensors"]
        self.parent_param_shapes = fields_dict["parent_param_shapes"]
        self.num_params_total = fields_dict["num_params_total"]
        self.parent_params_fsize = fields_dict["parent_params_fsize"]
        self.parent_params_fsize_nice = fields_dict["parent_params_fsize_nice"]

        # Corresponding layer info:
        self.operation_equivalence_type = fields_dict["operation_equivalence_type"]
        self.equivalent_operations = fields_dict["equivalent_operations"]
        self.same_layer_operations = fields_dict["same_layer_operations"]

        # Graph info:
        self.parent_layers = fields_dict["parent_layers"]
        self.has_parents = fields_dict["has_parents"]
        self.parent_layer_arg_locs = fields_dict["parent_layer_arg_locs"]
        self.orig_ancestors = fields_dict["orig_ancestors"]
        self.child_layers = fields_dict["child_layers"]
        self.has_children = fields_dict["has_children"]
        self.sibling_layers = fields_dict["sibling_layers"]
        self.has_siblings = fields_dict["has_siblings"]
        self.spouse_layers = fields_dict["spouse_layers"]
        self.has_spouses = fields_dict["has_spouses"]
        self.is_input_layer = fields_dict["is_input_layer"]
        self.has_input_ancestor = fields_dict["has_input_ancestor"]
        self.input_ancestors = fields_dict["input_ancestors"]
        self.min_distance_from_input = fields_dict["min_distance_from_input"]
        self.max_distance_from_input = fields_dict["max_distance_from_input"]
        self.is_output_layer = fields_dict["is_output_layer"]
        self.is_output_parent = fields_dict["is_output_parent"]
        self.is_last_output_layer = fields_dict["is_last_output_layer"]
        self.is_output_ancestor = fields_dict["is_output_ancestor"]
        self.output_descendents = fields_dict["output_descendents"]
        self.min_distance_from_output = fields_dict["min_distance_from_output"]
        self.max_distance_from_output = fields_dict["max_distance_from_output"]
        self.input_output_address = fields_dict["input_output_address"]
        self.is_buffer_layer = fields_dict["is_buffer_layer"]
        self.buffer_address = fields_dict["buffer_address"]
        self.buffer_pass = fields_dict["buffer_pass"]
        self.buffer_parent = fields_dict["buffer_parent"]
        self.initialized_inside_model = fields_dict["initialized_inside_model"]
        self.has_internally_initialized_ancestor = fields_dict[
            "has_internally_initialized_ancestor"
        ]
        self.internally_initialized_parents = fields_dict[
            "internally_initialized_parents"
        ]
        self.internally_initialized_ancestors = fields_dict[
            "internally_initialized_ancestors"
        ]
        self.terminated_inside_model = fields_dict["terminated_inside_model"]

        # Conditional info
        self.is_terminal_bool_layer = fields_dict["is_terminal_bool_layer"]
        self.is_atomic_bool_layer = fields_dict["is_atomic_bool_layer"]
        self.atomic_bool_val = fields_dict["atomic_bool_val"]
        self.in_cond_branch = fields_dict["in_cond_branch"]
        self.cond_branch_start_children = fields_dict["cond_branch_start_children"]

        # Module info
        self.is_computed_inside_submodule = fields_dict["is_computed_inside_submodule"]
        self.containing_module_origin = fields_dict["containing_module_origin"]
        self.containing_modules_origin_nested = fields_dict[
            "containing_modules_origin_nested"
        ]
        self.module_nesting_depth = fields_dict["module_nesting_depth"]
        self.modules_entered = fields_dict["modules_entered"]
        self.modules_entered_argnames = fields_dict["modules_entered_argnames"]
        self.module_passes_entered = fields_dict["module_passes_entered"]
        self.is_submodule_input = fields_dict["is_submodule_input"]
        self.modules_exited = fields_dict["modules_exited"]
        self.module_passes_exited = fields_dict["module_passes_exited"]
        self.is_submodule_output = fields_dict["is_submodule_output"]
        self.is_bottom_level_submodule_output = fields_dict[
            "is_bottom_level_submodule_output"
        ]
        self.bottom_level_submodule_pass_exited = fields_dict[
            "bottom_level_submodule_pass_exited"
        ]
        self.module_entry_exit_threads_inputs = fields_dict[
            "module_entry_exit_threads_inputs"
        ]
        self.module_entry_exit_thread_output = fields_dict[
            "module_entry_exit_thread_output"
        ]

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    def print_all_fields(self):
        """Print all data fields in the layer."""
        fields_to_exclude = ["source_model_history", "func_rng_states"]

        for field in dir(self):
            attr = getattr(self, field)
            if not any(
                    [field.startswith("_"), field in fields_to_exclude, callable(attr)]
            ):
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
        fields_not_to_deepcopy = [
            "func_applied",
            "gradfunc",
            "source_model_history",
            "func_rng_states",
            "creation_args",
            "creation_kwargs",
            "parent_params",
            "tensor_contents",
            "children_tensor_versions",
        ]
        for field in TENSOR_LOG_ENTRY_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = copy.deepcopy(getattr(self, field))
            else:
                fields_dict[field] = getattr(self, field)
        copied_entry = TensorLogEntry(fields_dict)
        return copied_entry

    def save_tensor_data(
            self,
            t: torch.Tensor,
            t_args: Union[List, Tuple],
            t_kwargs: Dict,
            save_function_args: bool,
            activation_postfunc: Optional[Callable] = None,
    ):
        """Saves the tensor data for a given tensor operation.

        Args:
            t: the tensor.
            t_args: tensor positional arguments for the operation
            t_kwargs: tensor keyword arguments for the operation
            save_function_args: whether to save the arguments to the function
            activation_postfunc: function to apply to activations before saving them
        """
        # The tensor itself:
        self.tensor_contents = safe_copy(t, self.detach_saved_tensor)
        if self.output_device not in [str(self.tensor_contents.device), "same"]:
            self.tensor_contents = clean_to(self.tensor_contents, self.output_device)
        if activation_postfunc is not None:
            self.source_model_history.pause_logging = True
            self.tensor_contents = activation_postfunc(self.tensor_contents)
            self.source_model_history.pause_logging = False

        self.has_saved_activations = True

        # Tensor args and kwargs:
        if save_function_args:
            self.function_args_saved = True
            creation_args = []
            for arg in t_args:
                if type(arg) == list:
                    creation_args.append([safe_copy(a) for a in arg])
                else:
                    creation_args.append(safe_copy(arg))

            creation_kwargs = {}
            for key, value in t_kwargs.items():
                if type(value) == list:
                    creation_kwargs[key] = [safe_copy(v) for v in value]
                else:
                    creation_kwargs[key] = safe_copy(value)
            self.creation_args = creation_args
            self.creation_kwargs = creation_kwargs
        else:
            self.creation_args = None
            self.creation_kwargs = None

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
        self.grad_fsizes_nice = [
            human_readable_size(get_tensor_memory_amount(g)) for g in grad
        ]

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def get_child_layers(self):
        return [
            self.source_model_history[child_label] for child_label in self.child_layers
        ]

    def get_parent_layers(self):
        return [
            self.source_model_history[parent_label]
            for parent_label in self.parent_layers
        ]

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
            s += " (bottom-level submodule output)"
        else:
            s += " (not bottom-level submodule output)"
        s += "\n\tFamily info:"
        s += f"\n\t\tParents: {self.parent_layers}"
        s += f"\n\t\tChildren: {self.child_layers}"
        s += f"\n\t\tSpouses: {self.spouse_layers}"
        s += f"\n\t\tSiblings: {self.sibling_layers}"
        s += (
            f"\n\t\tOriginal Ancestors: {self.orig_ancestors} "
            f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        )
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internally_initialized_ancestors}"
        s += (
            f"\n\t\tOutput Descendents: {self.output_descendents} "
            f"(min dist {self.min_distance_from_output} nodes, max dist {self.max_distance_from_output} nodes)"
        )
        if self.tensor_contents is not None:
            s += f"\n\tTensor contents: \n{print_override(self.tensor_contents, '__str__')}"
        return s

    def _str_after_pass(self):
        if self.layer_passes_total > 1:
            pass_str = f" (pass {self.pass_num}/{self.layer_passes_total}), "
        else:
            pass_str = ", "
        s = (
            f"Layer {self.layer_label_no_pass}"
            f"{pass_str}operation {self.operation_num}/"
            f"{self.source_model_history.num_operations}:"
        )
        s += f"\n\tOutput tensor: shape={self.tensor_shape}, dype={self.tensor_dtype}, size={self.tensor_fsize_nice}"
        if not self.has_saved_activations:
            s += " (not saved)"
        s += self._tensor_contents_str_helper()
        s += self._tensor_family_str_helper()
        if len(self.parent_param_shapes) > 0:
            params_shapes_str = ", ".join(
                str(param_shape) for param_shape in self.parent_param_shapes
            )
            s += (
                f"\n\tParams: Computed from params with shape {params_shapes_str}; "
                f"{self.num_params_total} params total ({self.parent_params_fsize_nice})"
            )
        else:
            s += "\n\tParams: no params used"
        if self.containing_module_origin is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.containing_module_origin}"
        if not self.is_input_layer:
            s += (
                f"\n\tFunction: {self.func_applied_name} (grad_fn: {self.gradfunc}) "
                f"{module_str}"
            )
            s += f"\n\tTime elapsed: {self.func_time_elapsed: .3E}s"
        if len(self.modules_exited) > 0:
            modules_exited_str = ", ".join(self.modules_exited)
            s += f"\n\tOutput of modules: {modules_exited_str}"
        else:
            s += "\n\tOutput of modules: none"
        if self.is_bottom_level_submodule_output:
            s += f"\n\tOutput of bottom-level module: {self.bottom_level_submodule_pass_exited}"
        lookup_keys_str = ", ".join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents."""
        if self.tensor_contents is None:
            return ""
        else:
            s = ""
            tensor_size_shown = 8
            saved_shape = self.tensor_contents.shape
            if len(saved_shape) == 0:
                tensor_slice = self.tensor_contents
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = self.tensor_contents[0:num_dims]
            elif len(saved_shape) == 2:
                num_dims = min([tensor_size_shown, saved_shape[-2], saved_shape[-1]])
                tensor_slice = self.tensor_contents[0:num_dims, 0:num_dims]
            else:
                num_dims = min(
                    [tensor_size_shown, self.tensor_shape[-2], self.tensor_shape[-1]]
                )
                tensor_slice = self.tensor_contents.data.clone()
                for _ in range(len(saved_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims]
            tensor_slice = tensor_slice.detach()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if (len(saved_shape) > 0) and (max(saved_shape) > tensor_size_shown):
                s += "..."
        return s

    def _tensor_family_str_helper(self) -> str:
        s = "\n\tRelated Layers:"
        if len(self.parent_layers) > 0:
            s += "\n\t\t- parent layers: " + ", ".join(self.parent_layers)
        else:
            s += "\n\t\t- no parent layers"

        if len(self.child_layers) > 0:
            s += "\n\t\t- child layers: " + ", ".join(self.child_layers)
        else:
            s += "\n\t\t- no child layers"

        if len(self.sibling_layers) > 0:
            s += "\n\t\t- shares parents with layers: " + ", ".join(self.sibling_layers)
        else:
            s += "\n\t\t- shares parents with no other layers"

        if len(self.spouse_layers) > 0:
            s += "\n\t\t- shares children with layers: " + ", ".join(self.spouse_layers)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:
            s += "\n\t\t- descendent of input layers: " + ", ".join(
                self.input_ancestors
            )
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.is_output_ancestor:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(
                self.output_descendents
            )
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
        self.buffer_pass = source_entry.buffer_pass
        self.input_output_address = source_entry.input_output_address
        self.cond_branch_start_children = source_entry.cond_branch_start_children
        self.is_terminal_bool_layer = source_entry.is_terminal_bool_layer
        self.atomic_bool_val = source_entry.atomic_bool_val
        self.child_layers = []
        self.parent_layers = []
        self.orphan_layers = []

        # Module info:
        self.containing_modules_origin_nested = (
            source_entry.containing_modules_origin_nested
        )
        self.modules_exited = source_entry.modules_exited
        self.module_passes_exited = source_entry.module_passes_exited
        self.is_bottom_level_submodule_output = False
        self.bottom_level_submodule_passes_exited = set()

        # Fields specific to rolled node to fill in:
        self.edges_vary_across_passes = False
        self.child_layers_per_pass = defaultdict(list)
        self.child_passes_per_layer = defaultdict(list)
        self.parent_layers_per_pass = defaultdict(list)
        self.parent_passes_per_layer = defaultdict(list)

        # Each one will now be a list of layers, since they can vary across passes.
        self.parent_layer_arg_locs = {
            "args": defaultdict(set),
            "kwargs": defaultdict(set),
        }

    def update_data(self, source_node: TensorLogEntry):
        """Updates the data as need be.
        Args:
            source_node: the source node
        """
        if source_node.has_input_ancestor:
            self.has_input_ancestor = True
        if not any(
                [
                    self.input_output_address is None,
                    source_node.input_output_address is None,
                ]
        ):
            self.input_output_address = "".join(
                [
                    char if (source_node.input_output_address[c] == char) else "*"
                    for c, char in enumerate(self.input_output_address)
                ]
            )
            if self.input_output_address[-1] == ".":
                self.input_output_address = self.input_output_address[:-1]
            if self.input_output_address[-1] == "*":
                self.input_output_address = self.input_output_address.strip("*") + "*"

    def add_pass_info(self, source_node: TensorLogEntry):
        """Adds information about another pass of the same layer: namely, mark information about what the
        child and parent layers are for each pass.

        Args:
            source_node: Information for the source pass
        """
        # Label the layers for each pass
        child_layer_labels = [
            self.source_model_history[child].layer_label_no_pass
            for child in source_node.child_layers
        ]
        for child_layer in child_layer_labels:
            if child_layer not in self.child_layers:
                self.child_layers.append(child_layer)
            if child_layer not in self.child_layers_per_pass[source_node.pass_num]:
                self.child_layers_per_pass[source_node.pass_num].append(child_layer)

        parent_layer_labels = [
            self.source_model_history[parent].layer_label_no_pass
            for parent in source_node.parent_layers
        ]
        for parent_layer in parent_layer_labels:
            if parent_layer not in self.parent_layers:
                self.parent_layers.append(parent_layer)
            if parent_layer not in self.parent_layers_per_pass[source_node.pass_num]:
                self.parent_layers_per_pass[source_node.pass_num].append(parent_layer)

        # Label the passes for each layer, and indicate if any layers vary based on the pass.
        for child_layer in source_node.child_layers:
            child_layer_label = self.source_model_history[
                child_layer
            ].layer_label_no_pass
            if (
                    source_node.pass_num
                    not in self.child_passes_per_layer[child_layer_label]
            ):
                self.child_passes_per_layer[child_layer_label].append(
                    source_node.pass_num
                )

        for parent_layer in source_node.parent_layers:
            parent_layer_label = self.source_model_history[
                parent_layer
            ].layer_label_no_pass
            if (
                    source_node.pass_num
                    not in self.parent_passes_per_layer[parent_layer_label]
            ):
                self.parent_passes_per_layer[parent_layer_label].append(
                    source_node.pass_num
                )

        # Check if any edges vary across passes.
        if source_node.pass_num == source_node.layer_passes_total:
            pass_lists = list(self.parent_passes_per_layer.values()) + list(
                self.child_passes_per_layer.values()
            )
            pass_lens = [len(passes) for passes in pass_lists]
            if any(
                    [pass_len < source_node.layer_passes_total for pass_len in pass_lens]
            ):
                self.edges_vary_across_passes = True
            else:
                self.edges_vary_across_passes = False

        # Add submodule info:
        if source_node.is_bottom_level_submodule_output:
            self.is_bottom_level_submodule_output = True
            self.bottom_level_submodule_passes_exited.add(
                source_node.bottom_level_submodule_pass_exited
            )

        # For the parent arg locations, have a list of layers rather than single layer, since they can
        # vary across passes.

        for arg_type in ["args", "kwargs"]:
            for arg_key, layer_label in source_node.parent_layer_arg_locs[
                arg_type
            ].items():
                layer_label_no_pass = self.source_model_history[
                    layer_label
                ].layer_label_no_pass
                self.parent_layer_arg_locs[arg_type][arg_key].add(layer_label_no_pass)

    def __str__(self) -> str:
        fields_not_to_print = ["source_model_history"]
        s = ""
        for field in dir(self):
            attr = getattr(self, field)
            if (
                    not field.startswith("_")
                    and field not in fields_not_to_print
                    and not (callable(attr))
            ):
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
    DEFAULT_BG_COLOR = "white"
    BOOL_NODE_COLOR = "#F7D460"
    MAX_MODULE_PENWIDTH = 5
    MIN_MODULE_PENWIDTH = 2
    PENWIDTH_RANGE = MAX_MODULE_PENWIDTH - MIN_MODULE_PENWIDTH
    COMMUTE_FUNCS = ["add", "mul", "cat", "eq", "ne"]
    FUNCS_NOT_TO_PERTURB_IN_VALIDATION = [
        "expand_as",
        "new_zeros",
        "new_ones",
        "zero_",
        "copy_",
        "clamp",
        "fill_",
        "zeros_like",
        "ones_like",
    ]

    def __init__(
            self,
            model_name: str,
            output_device: str = "same",
            activation_postfunc: Optional[Callable] = None,
            keep_unsaved_layers: bool = True,
            save_function_args: bool = False,
            save_gradients: bool = False,
            detach_saved_tensors: bool = False,
            mark_input_output_distances: bool = True,
    ):
        """Object that stores the history of a model's forward pass.
        Both logs the history in real time, and stores a nice
        representation of the full history for the user afterward.
        """
        # Setup:
        activation_postfunc = copy.deepcopy(activation_postfunc)

        # General info
        self.model_name = model_name
        self.pass_finished = False
        self.track_tensors = False
        self.logging_mode = "exhaustive"
        self.pause_logging = False
        self.all_layers_logged = False
        self.all_layers_saved = False
        self.keep_unsaved_layers = keep_unsaved_layers
        self.activation_postfunc = activation_postfunc
        self.current_function_call_barcode = None
        self.random_seed_used = None
        self.output_device = output_device
        self.detach_saved_tensors = detach_saved_tensors
        self.save_function_args = save_function_args
        self.save_gradients = save_gradients
        self.has_saved_gradients = False
        self.mark_input_output_distances = mark_input_output_distances

        # Model structure info
        self.model_is_recurrent = False
        self.model_max_recurrent_loops = 1
        self.model_has_conditional_branching = False
        self.model_is_branching = False

        # Tensor Tracking:
        self.layer_list: List[TensorLogEntry] = []
        self.layer_list_rolled: List[RolledTensorLogEntry] = []
        self.layer_dict_main_keys: Dict[str, TensorLogEntry] = OrderedDict()
        self.layer_dict_all_keys: Dict[str, TensorLogEntry] = OrderedDict()
        self.layer_dict_rolled: Dict[str, RolledTensorLogEntry] = OrderedDict()
        self.layer_labels: List[str] = []
        self.layer_labels_w_pass: List[str] = []
        self.layer_labels_no_pass: List[str] = []
        self.layer_num_passes: Dict[str, int] = OrderedDict()
        self.raw_tensor_dict: Dict[str, TensorLogEntry] = OrderedDict()
        self.raw_tensor_labels_list: List[str] = []
        self.tensor_nums_to_save: List[int] = []
        self.tensor_counter: int = 0
        self.num_operations: int = 0
        self.raw_layer_type_counter: Dict[str, int] = defaultdict(lambda: 0)
        self.unsaved_layers_lookup_keys: Set[str] = set()

        # Mapping from raw to final layer labels:
        self.raw_to_final_layer_labels: Dict[str, str] = {}
        self.final_to_raw_layer_labels: Dict[str, str] = {}
        self.lookup_keys_to_tensor_num_dict: Dict[str, int] = {}
        self.tensor_num_to_lookup_keys_dict: Dict[int, List[str]] = defaultdict(list)

        # Special Layers:
        self.input_layers: List[str] = []
        self.output_layers: List[str] = []
        self.buffer_layers: List[str] = []
        self.buffer_num_passes: Dict = {}
        self.internally_initialized_layers: List[str] = []
        self.layers_where_internal_branches_merge_with_input: List[str] = []
        self.internally_terminated_layers: List[str] = []
        self.internally_terminated_bool_layers: List[str] = []
        self.conditional_branch_edges: List[Tuple[str, str]] = []
        self.layers_with_saved_activations: List[str] = []
        self.unlogged_layers: List[str] = []
        self.layers_with_saved_gradients: List[str] = []
        self.layers_computed_with_params: Dict[str, List] = defaultdict(list)
        self.equivalent_operations: Dict[str, set] = defaultdict(set)
        self.same_layer_operations: Dict[str, list] = defaultdict(list)

        # Tensor info:
        self.num_tensors_total: int = 0
        self.tensor_fsize_total: int = 0
        self.tensor_fsize_total_nice: str = human_readable_size(0)
        self.num_tensors_saved: int = 0
        self.tensor_fsize_saved: int = 0
        self.tensor_fsize_saved_nice: str = human_readable_size(0)

        # Param info:
        self.total_param_tensors: int = 0
        self.total_param_layers: int = 0
        self.total_params: int = 0
        self.total_params_fsize: int = 0
        self.total_params_fsize_nice: str = human_readable_size(0)

        # Module info:
        self.module_addresses: List[str] = []
        self.module_types: Dict[str, Any] = {}
        self.module_passes: List = []
        self.module_num_passes: Dict = defaultdict(lambda: 1)
        self.top_level_modules: List = []
        self.top_level_module_passes: List = []
        self.module_children: Dict = defaultdict(list)
        self.module_pass_children: Dict = defaultdict(list)
        self.module_nparams: Dict = defaultdict(lambda: 0)
        self.module_num_tensors: Dict = defaultdict(lambda: 0)
        self.module_pass_num_tensors: Dict = defaultdict(lambda: 0)
        self.module_layers: Dict = defaultdict(list)
        self.module_pass_layers: Dict = defaultdict(list)
        self.module_layer_argnames = defaultdict(list)

        # Time elapsed:
        self.pass_start_time: float = 0
        self.pass_end_time: float = 0
        self.elapsed_time_setup: float = 0
        self.elapsed_time_forward_pass: float = 0
        self.elapsed_time_cleanup: float = 0
        self.elapsed_time_total: float = 0
        self.elapsed_time_function_calls: float = 0
        self.elapsed_time_torchlens_logging: float = 0

        # Reference info
        self.func_argnames: Dict[str, tuple] = defaultdict(lambda: tuple([]))

    # ********************************************
    # ********** User-Facing Functions ***********
    # ********************************************

    def print_all_fields(self):
        """Print all data fields for ModelHistory."""
        fields_to_exclude = [
            "layer_list",
            "layer_dict_main_keys",
            "layer_dict_all_keys",
            "raw_tensor_dict",
            "decorated_to_orig_funcs_dict",
        ]

        for field in dir(self):
            attr = getattr(self, field)
            if not any(
                    [field.startswith("_"), field in fields_to_exclude, callable(attr)]
            ):
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
        fields_for_df = [
            "layer_label",
            "layer_label_w_pass",
            "layer_label_no_pass",
            "layer_label_short",
            "layer_label_w_pass_short",
            "layer_label_no_pass_short",
            "layer_type",
            "layer_type_num",
            "layer_total_num",
            "layer_passes_total",
            "pass_num",
            "operation_num",
            "tensor_shape",
            "tensor_dtype",
            "tensor_fsize",
            "tensor_fsize_nice",
            "func_applied_name",
            "func_time_elapsed",
            "function_is_inplace",
            "gradfunc",
            "is_input_layer",
            "is_output_layer",
            "is_buffer_layer",
            "is_part_of_iterable_output",
            "iterable_output_index",
            "parent_layers",
            "has_parents",
            "orig_ancestors",
            "child_layers",
            "has_children",
            "output_descendents",
            "sibling_layers",
            "has_siblings",
            "spouse_layers",
            "has_spouses",
            "initialized_inside_model",
            "min_distance_from_input",
            "max_distance_from_input",
            "min_distance_from_output",
            "max_distance_from_output",
            "computed_with_params",
            "num_params_total",
            "parent_param_shapes",
            "parent_params_fsize",
            "parent_params_fsize_nice",
            "modules_entered",
            "modules_exited",
            "is_submodule_input",
            "is_submodule_output",
            "containing_module_origin",
            "containing_modules_origin_nested",
        ]

        fields_to_change_type = {
            "layer_type_num": int,
            "layer_total_num": int,
            "layer_passes_total": int,
            "pass_num": int,
            "operation_num": int,
            "function_is_inplace": bool,
            "is_input_layer": bool,
            "is_output_layer": bool,
            "is_buffer_layer": bool,
            "is_part_of_iterable_output": bool,
            "has_parents": bool,
            "has_children": bool,
            "has_siblings": bool,
            "has_spouses": bool,
            "computed_with_params": bool,
            "num_params_total": int,
            "parent_params_fsize": int,
            "tensor_fsize": int,
            "is_submodule_input": bool,
            "is_submodule_output": bool,
        }

        model_df_dictlist = []
        for tensor_entry in self.layer_list:
            tensor_dict = {}
            for field_name in fields_for_df:
                tensor_dict[field_name] = getattr(tensor_entry, field_name)
            model_df_dictlist.append(tensor_dict)
        model_df = pd.DataFrame(model_df_dictlist)

        for field in fields_to_change_type:
            model_df[field] = model_df[field].astype(fields_to_change_type[field])

        return model_df

    ##########################
    ## Decoration Functions ##
    ##########################

    def torch_func_decorator(self, func: Callable):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Initial bookkeeping; check if it's a special function, organize the arguments.
            self.current_function_call_barcode = 0
            func_name = func.__name__
            if (
                    (func_name in funcs_not_to_log)
                    or (not self.track_tensors)
                    or self.pause_logging
            ):
                out = func(*args, **kwargs)
                return out
            all_args = list(args) + list(kwargs.values())
            arg_tensorlike = get_vars_of_type_from_obj(all_args, torch.Tensor)

            # Register any buffer tensors in the arguments.

            for t in arg_tensorlike:
                if hasattr(t, 'tl_buffer_address'):
                    self.log_source_tensor(t, 'buffer', getattr(t, 'tl_buffer_address'))

            if (func_name in print_funcs) and (len(arg_tensorlike) > 0):
                out = print_override(args[0], func_name)
                return out

            # Copy the args and kwargs in case they change in-place:
            if self.save_function_args:
                arg_copies = tuple([safe_copy(arg) for arg in args])
                kwarg_copies = {k: safe_copy(v) for k, v in kwargs.items()}
            else:
                arg_copies = args
                kwarg_copies = kwargs

            # Call the function, tracking the timing, rng states, and whether it's a nested function
            func_call_barcode = make_random_barcode()
            self.current_function_call_barcode = func_call_barcode
            start_time = time.time()
            func_rng_states = log_current_rng_states()
            out_orig = func(*args, **kwargs)
            func_time_elapsed = time.time() - start_time
            is_bottom_level_func = (
                    self.current_function_call_barcode == func_call_barcode
            )

            if func_name in ["__setitem__", "zero_", "__delitem__"]:
                out_orig = args[0]

            # Log all output tensors
            output_tensors = get_vars_of_type_from_obj(
                out_orig,
                which_type=torch.Tensor,
                subclass_exceptions=[torch.nn.Parameter],
            )

            if len(output_tensors) > 0:
                self.log_function_output_tensors(
                    func,
                    args,
                    kwargs,
                    arg_copies,
                    kwarg_copies,
                    out_orig,
                    func_time_elapsed,
                    func_rng_states,
                    is_bottom_level_func,
                )

            return out_orig

        return wrapped_func

    def decorate_pytorch(
            self, torch_module: types.ModuleType, orig_func_defs: List[Tuple]
    ) -> Dict[Callable, Callable]:
        """Mutates all PyTorch functions (TEMPORARILY!) to save the outputs of any functions
        that return Tensors, along with marking them with metadata. Returns a list of tuples that
        save the current state of the functions, such that they can be restored when done.

        args:
            torch_module: The top-level torch module (i.e., from "import torch").
                This is supplied as an argument on the off-chance that the user has imported torch
                and done their own monkey-patching.
            tensors_to_mutate: A list of tensors that will be mutated (since any tensors created
                before calling the torch mutation function will not be mutated).
            orig_func_defs: Supply a list from outside to guarantee it can be cleaned up properly.
            tensor_record: A list to which the outputs of the functions will be appended.

        returns:
            List of tuples consisting of [namespace, func_name, orig_func], sufficient
            to return torch to normal when finished, and also a dict mapping mutated functions to original functions.
        """

        # Do a pass to save the original func defs.
        self.collect_orig_func_defs(torch_module, orig_func_defs)
        decorated_func_mapper = {}

        for namespace_name, func_name in ORIG_TORCH_FUNCS:
            namespace_name_notorch = namespace_name.replace("torch.", "")
            local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
            if not hasattr(local_func_namespace, func_name):
                continue
            orig_func = getattr(local_func_namespace, func_name)
            if func_name not in self.func_argnames:
                self.get_func_argnames(orig_func, func_name)
            if getattr(orig_func, "__name__", False) == "wrapped_func":
                continue
            new_func = self.torch_func_decorator(orig_func)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(local_func_namespace, func_name, new_func)
            except (AttributeError, TypeError) as _:
                pass
            new_func.tl_is_decorated_function = True
            decorated_func_mapper[new_func] = orig_func
            decorated_func_mapper[orig_func] = new_func

        # Bolt on the identity function
        new_identity = self.torch_func_decorator(identity)
        torch.identity = new_identity

        return decorated_func_mapper

    @staticmethod
    def undecorate_pytorch(
            torch_module, orig_func_defs: List[Tuple], input_tensors: List[torch.Tensor]
    ):
        """
        Returns all PyTorch functions back to the definitions they had when mutate_pytorch was called.
        This is done for the output tensors and history_dict too to avoid ugliness. Also deletes
        the mutant versions of the functions to remove any references to old ModelHistory object.

        args:
            torch_module: The torch module object.
            orig_func_defs: List of tuples consisting of [namespace_name, func_name, orig_func], sufficient
                to regenerate the original functions.
            input_tensors: List of input tensors whose fucntions will be undecorated.
            decorated_func_mapper: Maps the decorated function to the original function
        """
        for namespace_name, func_name, orig_func in orig_func_defs:
            namespace_name_notorch = namespace_name.replace("torch.", "")
            local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decorated_func = getattr(local_func_namespace, func_name)
            del decorated_func
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(local_func_namespace, func_name, orig_func)
            except (AttributeError, TypeError) as _:
                continue
        delattr(torch, "identity")
        for input_tensor in input_tensors:
            if hasattr(input_tensor, "tl_tensor_label_raw"):
                delattr(input_tensor, "tl_tensor_label_raw")

    @staticmethod
    def undecorate_tensor(t, device: str = "cpu"):
        """Convenience function to replace the tensor with an unmutated version of itself, keeping the same data.

        Args:
            t: tensor or parameter object
            device: device to move the tensor to

        Returns:
            Unmutated tensor.
        """
        if type(t) in [torch.Tensor, torch.nn.Parameter]:
            new_t = safe_copy(t)
        else:
            new_t = t
        del t
        for attr in dir(new_t):
            if attr.startswith("tl_"):
                delattr(new_t, attr)
        new_t = clean_to(new_t, device)
        return new_t

    @staticmethod
    def collect_orig_func_defs(
            torch_module: types.ModuleType, orig_func_defs: List[Tuple]
    ):
        """Collects the original torch function definitions, so they can be restored after the logging is done.

        Args:
            torch_module: The top-level torch module
            orig_func_defs: List of tuples keeping track of the original function definitions
        """
        for namespace_name, func_name in ORIG_TORCH_FUNCS:
            namespace_name_notorch = namespace_name.replace("torch.", "")
            local_func_namespace = nested_getattr(torch_module, namespace_name_notorch)
            if not hasattr(local_func_namespace, func_name):
                continue
            orig_func = getattr(local_func_namespace, func_name)
            orig_func_defs.append((namespace_name, func_name, orig_func))

    # TODO: hard-code some of the arg names; for example truediv, getitem, etc. Can crawl through and see what isn't working
    def get_func_argnames(self, orig_func: Callable, func_name: str):
        """Attempts to get the argument names for a function, first by checking the signature, then
        by checking the documentation. Adds these names to func_argnames if it can find them,
        doesn't do anything if it can't."""
        try:
            argnames = list(inspect.signature(orig_func).parameters.keys())
            argnames = tuple([arg.replace('*', '') for arg in argnames if arg not in ['cls', 'self']])
            self.func_argnames[func_name] = argnames
            return
        except ValueError:
            pass

        docstring = orig_func.__doc__
        if (type(docstring) is not str) or (len(docstring) == 0):  # if docstring missing, skip it
            return

        open_ind, close_ind = docstring.find('('), docstring.find(')')
        argstring = docstring[open_ind + 1: close_ind]
        arg_list = argstring.split(',')
        arg_list = [arg.strip(' ') for arg in arg_list]
        argnames = []
        for arg in arg_list:
            argname = arg.split('=')[0]
            if argname in ['*', '/', '//', '']:
                continue
            argname = argname.replace('*', '')
            argnames.append(argname)
        argnames = tuple([arg for arg in argnames if arg not in ['self', 'cls']])
        self.func_argnames[func_name] = argnames
        return

    ###########################
    ##### Model Functions #####
    ###########################

    def prepare_model(
            self,
            model: nn.Module,
            module_orig_forward_funcs: Dict,
            decorated_func_mapper: Dict[Callable, Callable],
    ):
        """Adds annotations and hooks to the model, and decorates any functions in the model.

        Args:
            model: Model to prepare.
            module_orig_forward_funcs: Dict with the original forward funcs for each submodule
            decorated_func_mapper: Dictionary mapping decorated functions to original functions, so they can be restored

        Returns:
            Model with hooks and attributes added.
        """
        self.model_name = str(type(model).__name__)
        model.tl_module_address = ""
        model.tl_source_model_history = self

        module_stack = [(model, "")]  # list of tuples (name, module)

        while len(module_stack) > 0:
            module, parent_address = module_stack.pop()
            module_children = list(module.named_children())

            # Decorate any torch functions in the model:
            for func_name, func in module.__dict__.items():
                if (
                        (func_name[0:2] == "__")
                        or (not callable(func))
                        or (func not in decorated_func_mapper)
                ):
                    continue
                module.__dict__[func_name] = decorated_func_mapper[func]

            # Annotate the children with the full address.
            for c, (child_name, child_module) in enumerate(module_children):
                child_address = (
                    f"{parent_address}.{child_name}"
                    if parent_address != ""
                    else child_name
                )
                child_module.tl_module_address = child_address
                module_children[c] = (child_module, child_address)
            module_stack = module_children + module_stack

            if module == model:  # don't tag the model itself.
                continue

            module.tl_source_model_history = self
            module.tl_module_type = str(type(module).__name__)
            self.module_types[module.tl_module_address] = module.tl_module_type
            module.tl_module_pass_num = 0
            module.tl_module_pass_labels = []
            module.tl_tensors_entered_labels = []
            module.tl_tensors_exited_labels = []

            # Add decorators.

            if hasattr(module, "forward") and not hasattr(
                    module.forward, "tl_forward_call_is_decorated"
            ):
                module_orig_forward_funcs[module] = module.forward
                module.forward = self.module_forward_decorator(module.forward, module)
                module.forward.tl_forward_call_is_decorated = True

        # Mark all parameters with requires_grad = True, and mark what they were before, so they can be restored on cleanup.
        for param in model.parameters():
            param.tl_requires_grad = param.requires_grad
            param.requires_grad = True

        # And prepare any buffer tensors.
        self.prepare_buffer_tensors(model)

    def prepare_buffer_tensors(self, model: nn.Module):
        """Goes through a model and all its submodules, and prepares any "buffer" tensors: tensors
        attached to the module that aren't model parameters.

        Args:
            model: PyTorch model

        Returns:
            PyTorch model with all buffer tensors prepared and ready to track.
        """
        submodules = self.get_all_submodules(model)
        for submodule in submodules:
            attr_list = list(submodule.named_buffers()) + list(iter_accessible_attributes(submodule))
            for attribute_name, attribute in attr_list:
                if issubclass(type(attribute), torch.Tensor) and not issubclass(
                        type(attribute), torch.nn.Parameter
                ) and not hasattr(attribute, 'tl_buffer_address'):
                    if submodule.tl_module_address == "":
                        buffer_address = attribute_name
                    else:
                        buffer_address = (
                                submodule.tl_module_address + "." + attribute_name
                        )
                    setattr(attribute, 'tl_buffer_address', buffer_address)

    def module_forward_decorator(
            self, orig_forward: Callable, module: nn.Module
    ) -> Callable:
        @wraps(orig_forward)
        def decorated_forward(*args, **kwargs):
            if self.logging_mode == "fast":  # do bare minimum for logging.
                out = orig_forward(*args, **kwargs)
                output_tensors = get_vars_of_type_from_obj(
                    out, torch.Tensor, search_depth=4
                )
                for t in output_tensors:
                    # if identity module, run the function for bookkeeping
                    if module.tl_module_type.lower() == "identity":
                        t = getattr(torch, "identity")(t)
                return out

            # "Pre-hook" operations:
            module_address = module.tl_module_address
            module.tl_module_pass_num += 1
            module_pass_label = (module_address, module.tl_module_pass_num)
            module.tl_module_pass_labels.append(module_pass_label)
            input_tensors = get_vars_of_type_from_obj(
                [args, kwargs], torch.Tensor, [torch.nn.Parameter], search_depth=4
            )
            input_tensor_labels = set()
            for t in input_tensors:
                if (not hasattr(t, 'tl_tensor_label_raw')) and hasattr(t, 'tl_buffer_address'):
                    self.log_source_tensor(t, 'buffer', getattr(t, 'tl_buffer_address'))
                tensor_entry = self.raw_tensor_dict[t.tl_tensor_label_raw]
                input_tensor_labels.add(t.tl_tensor_label_raw)
                module.tl_tensors_entered_labels.append(t.tl_tensor_label_raw)
                tensor_entry.modules_entered.append(module_address)
                tensor_entry.module_passes_entered.append(module_pass_label)
                tensor_entry.is_submodule_input = True
                for arg_key, arg_val in list(enumerate(args)) + list(kwargs.items()):
                    if arg_val is t:
                        tensor_entry.modules_entered_argnames[
                            f"{module_pass_label[0]}:{module_pass_label[1]}"].append(arg_key)
                        self.module_layer_argnames[(f"{module_pass_label[0]}:"
                                                    f"{module_pass_label[1]}")].append((t.tl_tensor_label_raw, arg_key))
                tensor_entry.module_entry_exit_thread_output.append(
                    ("+", module_pass_label[0], module_pass_label[1])
                )

            # Check the buffers.
            for buffer_name, buffer_tensor in module.named_buffers():
                if hasattr(buffer_tensor, 'tl_buffer_address'):
                    continue
                if module.tl_module_address == '':
                    buffer_address = buffer_name
                else:
                    buffer_address = f"{module.tl_module_address}.{buffer_name}"
                buffer_tensor.tl_buffer_address = buffer_address
                buffer_tensor.tl_buffer_parent = buffer_tensor.tl_tensor_label_raw
                delattr(buffer_tensor, 'tl_tensor_label_raw')

            # The function call
            out = orig_forward(*args, **kwargs)

            # "Post-hook" operations:
            module_address = module.tl_module_address
            module_pass_num = module.tl_module_pass_num
            module_entry_label = module.tl_module_pass_labels.pop()
            output_tensors = get_vars_of_type_from_obj(
                out, torch.Tensor, search_depth=4
            )
            for t in output_tensors:
                # if identity module or tensor unchanged, run the identity function for bookkeeping
                if (module.tl_module_type.lower() == "identity") or (
                        t.tl_tensor_label_raw in input_tensor_labels
                ):
                    t = getattr(torch, "identity")(t)
                tensor_entry = self.raw_tensor_dict[t.tl_tensor_label_raw]
                tensor_entry.is_submodule_output = True
                tensor_entry.is_bottom_level_submodule_output = (
                    self.log_whether_exited_submodule_is_bottom_level(t, module)
                )
                tensor_entry.modules_exited.append(module_address)
                tensor_entry.module_passes_exited.append(
                    (module_address, module_pass_num)
                )
                tensor_entry.module_entry_exit_thread_output.append(
                    ("-", module_entry_label[0], module_entry_label[1])
                )
                module.tl_tensors_exited_labels.append(t.tl_tensor_label_raw)

            for (
                    t
            ) in (
                    input_tensors
            ):  # Now that module is finished, roll back the threads of all input tensors.
                tensor_entry = self.raw_tensor_dict[t.tl_tensor_label_raw]
                input_module_thread = tensor_entry.module_entry_exit_thread_output[:]
                if (
                        "+",
                        module_entry_label[0],
                        module_entry_label[1],
                ) in input_module_thread:
                    module_entry_ix = input_module_thread.index(
                        ("+", module_entry_label[0], module_entry_label[1])
                    )
                    tensor_entry.module_entry_exit_thread_output = (
                        tensor_entry.module_entry_exit_thread_output[:module_entry_ix]
                    )

            return out

        return decorated_forward

    def log_whether_exited_submodule_is_bottom_level(
            self, t: torch.Tensor, submodule: nn.Module
    ):
        """Checks whether the submodule that a tensor is leaving is a "bottom-level" submodule;
        that is, that only one tensor operation happened inside the submodule.

        Args:
            t: the tensor leaving the module
            submodule: the module that the tensor is leaving

        Returns:
            Whether the tensor operation is bottom level.
        """
        tensor_entry = self.raw_tensor_dict[getattr(t, "tl_tensor_label_raw")]
        submodule_address = submodule.tl_module_address

        if tensor_entry.is_bottom_level_submodule_output:
            return True

        # If it was initialized inside the model and nothing entered the module, it's bottom-level.
        if (
                tensor_entry.initialized_inside_model
                and len(submodule.tl_tensors_entered_labels) == 0
        ):
            tensor_entry.is_bottom_level_submodule_output = True
            tensor_entry.bottom_level_submodule_pass_exited = (
                submodule_address,
                submodule.tl_module_pass_num,
            )
            return True

        # Else, all parents must have entered the submodule for it to be a bottom-level submodule.
        for parent_label in tensor_entry.parent_layers:
            parent_tensor = self[parent_label]
            parent_modules_entered = parent_tensor.modules_entered
            if (len(parent_modules_entered) == 0) or (
                    parent_modules_entered[-1] != submodule_address
            ):
                tensor_entry.is_bottom_level_submodule_output = False
                return False

        # If it survived the above tests, it's a bottom-level submodule.
        tensor_entry.is_bottom_level_submodule_output = True
        tensor_entry.bottom_level_submodule_pass_exited = (
            submodule_address,
            submodule.tl_module_pass_num,
        )
        return True

    def get_all_submodules(
            self, model: nn.Module, is_top_level_model: bool = True
    ) -> List[nn.Module]:
        """Recursively gets list of all submodules for given module, no matter their level in the
        hierarchy; this includes the model itself.

        Args:
            model: PyTorch model.
            is_top_level_model: Whether it's the top-level model; just for the recursive logic of it.

        Returns:
            List of all submodules.
        """
        submodules = []
        if is_top_level_model:
            submodules.append(model)
        for module in model.children():
            if module not in submodules:
                submodules.append(module)
            submodules += self.get_all_submodules(module, is_top_level_model=False)
        return submodules

    def cleanup_model(
            self,
            model: nn.Module,
            module_orig_forward_funcs: Dict[nn.Module, Callable],
            decorated_func_mapper: Dict[Callable, Callable],
    ):
        """Reverses all temporary changes to the model (namely, the forward hooks and added
        model attributes) that were added for PyTorch x-ray (scout's honor; leave no trace).

        Args:
            model: PyTorch model.
            module_orig_forward_funcs: Dict containing the original, undecorated forward pass functions for
                each submodule
            decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs

        Returns:
            Original version of the model.
        """
        submodules = self.get_all_submodules(model, is_top_level_model=True)
        for submodule in submodules:
            if submodule == model:
                continue
            submodule.forward = module_orig_forward_funcs[submodule]
        self.restore_model_attributes(
            model, decorated_func_mapper=decorated_func_mapper, attribute_keyword="tl"
        )
        self.undecorate_model_tensors(model)

    @staticmethod
    def clear_hooks(hook_handles: List):
        """Takes in a list of tuples (module, hook_handle), and clears the hook at that
        handle for each module.

        Args:
            hook_handles: List of tuples (module, hook_handle)

        Returns:
            Nothing.
        """
        for hook_handle in hook_handles:
            hook_handle.remove()

    @staticmethod
    def restore_module_attributes(
            module: nn.Module,
            decorated_func_mapper: Dict[Callable, Callable],
            attribute_keyword: str = "tl",
    ):
        def del_attrs_with_prefix(module, attribute_name):
            if attribute_name.startswith(attribute_keyword):
                delattr(module, attribute_name)
                return True

        for attribute_name, attr in iter_accessible_attributes(module, short_circuit=del_attrs_with_prefix):
            if (
                    isinstance(attr, Callable)
                    and (attr in decorated_func_mapper)
                    and (attribute_name[0:2] != "__")
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    setattr(module, attribute_name, decorated_func_mapper[attr])

    def restore_model_attributes(
            self,
            model: nn.Module,
            decorated_func_mapper: Dict[Callable, Callable],
            attribute_keyword: str = "tl",
    ):
        """Recursively clears the given attribute from all modules in the model.

        Args:
            model: PyTorch model.
            decorated_func_mapper: Dict mapping between original and decorated PyTorch funcs
            attribute_keyword: Any attribute with this keyword will be cleared.

        Returns:
            Nothing.
        """
        for module in self.get_all_submodules(model):
            self.restore_module_attributes(
                module,
                decorated_func_mapper=decorated_func_mapper,
                attribute_keyword=attribute_keyword,
            )

        for param in model.parameters():
            if hasattr(param, "tl_requires_grad"):
                param.requires_grad = getattr(param, "tl_requires_grad")
                delattr(param, "tl_requires_grad")

    def undecorate_model_tensors(self, model: nn.Module):
        """Goes through a model and all its submodules, and unmutates any tensor attributes. Normally just clearing
        parameters would have done this, but some module types (e.g., batchnorm) contain attributes that are tensors,
        but not parameters.

        Args:
            model: PyTorch model

        Returns:
            PyTorch model with unmutated versions of all tensor attributes.
        """
        submodules = self.get_all_submodules(model)
        for submodule in submodules:
            for attribute_name, attribute in iter_accessible_attributes(submodule):
                if issubclass(type(attribute), torch.Tensor):
                    if not issubclass(type(attribute), torch.nn.Parameter) and hasattr(
                            attribute, "tl_tensor_label_raw"
                    ):
                        delattr(attribute, "tl_tensor_label_raw")
                        if hasattr(attribute, 'tl_buffer_address'):
                            delattr(attribute, "tl_buffer_address")
                        if hasattr(attribute, 'tl_buffer_parent'):
                            delattr(attribute, "tl_buffer_parent")
                    else:
                        remove_attributes_starting_with_str(attribute, "tl_")
                elif type(attribute) in [list, tuple, set]:
                    for item in attribute:
                        if issubclass(type(item), torch.Tensor) and hasattr(
                                item, "tl_tensor_label_raw"
                        ):
                            delattr(item, "tl_tensor_label_raw")
                            if hasattr(item, 'tl_buffer_address'):
                                delattr(item, "tl_buffer_address")
                            if hasattr(item, 'tl_buffer_parent'):
                                delattr(item, "tl_buffer_parent")
                elif type(attribute) == dict:
                    for key, val in attribute.items():
                        if issubclass(type(val), torch.Tensor) and hasattr(
                                val, "tl_tensor_label_raw"
                        ):
                            delattr(val, "tl_tensor_label_raw")
                            if hasattr(val, 'tl_buffer_address'):
                                delattr(val, "tl_buffer_address")
                            if hasattr(val, 'tl_buffer_parent'):
                                delattr(val, "tl_buffer_parent")

    def get_op_nums_from_user_labels(
            self, which_layers: Union[str, List[Union[str, int]]]
    ) -> List[int]:
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
        if which_layers == "all":
            return which_layers
        elif which_layers in [None, "none", "None", "NONE", []]:
            return []

        if type(which_layers) != list:
            which_layers = [which_layers]
        which_layers = [
            layer.lower() if (type(layer) == str) else layer for layer in which_layers
        ]

        raw_tensor_nums_to_save = set()
        for layer_key in which_layers:
            # First check if it matches a lookup key. If so, use that.
            if layer_key in self.lookup_keys_to_tensor_num_dict:
                raw_tensor_nums_to_save.add(
                    self.lookup_keys_to_tensor_num_dict[layer_key]
                )
                continue

            # If not, pull out all layers for which the key is a substring.
            keys_with_substr = [
                key for key in self.layer_dict_all_keys if layer_key in str(key)
            ]
            if len(keys_with_substr) > 0:
                for key in keys_with_substr:
                    raw_tensor_nums_to_save.add(
                        self.layer_dict_all_keys[key].realtime_tensor_num
                    )
                continue

            # If no luck, try to at least point user in right direction:

            self._give_user_feedback_about_lookup_key(layer_key, "query_multiple")

        raw_tensor_nums_to_save = sorted(list(raw_tensor_nums_to_save))
        return raw_tensor_nums_to_save

    def save_new_activations(
            self,
            model: nn.Module,
            input_args: Union[torch.Tensor, List[Any]],
            input_kwargs: Dict[Any, Any] = None,
            layers_to_save: Union[str, List] = "all",
            random_seed: Optional[int] = None,
    ):
        """Saves activations to a new input to the model, replacing existing saved activations.
        This will be much faster than the initial call to log_forward_pass (since all the of the metadata has
        already been saved), so if you wish to save the activations to many different inputs for a given model
        this is the function you should use. The one caveat is that this function assumes that the computational
        graph will be the same for the new input; if the model involves a dynamic computational graph that can change
        across inputs, and this graph changes for the new input, then this function will throw an error. In that case,
        you'll have to do a new call to log_forward_pass to log the new graph.

        Args:
            model: Model for which to save activations
            input_args: Either a single tensor input to the model, or list of input arguments.
            input_kwargs: Dict of keyword arguments to the model.
            layers_to_save: List of layers to save, using any valid lookup keys
            random_seed: Which random seed to use
        Returns:
            Nothing, but now the ModelHistory object will have saved activations for the new input.
        """
        self.logging_mode = "fast"

        # Go through and clear all existing activations.
        for tensor_log_entry in self:
            tensor_log_entry.tensor_contents = None
            tensor_log_entry.has_saved_activations = False
            tensor_log_entry.has_saved_grad = False
            tensor_log_entry.grad_contents = None

        # Reset relevant fields.
        self.layers_with_saved_activations = []
        self.layers_with_saved_gradients = []
        self.num_tensors_saved = 0
        self.tensor_fsize_saved = 0
        self.tensor_counter = 0
        self.raw_layer_type_counter = defaultdict(lambda: 0)

        # Now run and log the new inputs.
        self.run_and_log_inputs_through_model(
            model, input_args, input_kwargs, layers_to_save, random_seed
        )

    ########################################
    #### Running and logging the model #####
    ########################################

    def run_and_log_inputs_through_model(
            self,
            model: nn.Module,
            input_args: Union[torch.Tensor, List[Any]],
            input_kwargs: Dict[Any, Any] = None,
            layers_to_save: Optional[Union[str, List[Union[str, int]]]] = "all",
            random_seed: Optional[int] = None,
    ):
        """Runs input through model and logs it in ModelHistory.

        Args:
            model: Model for which to save activations
            input_args: Either a single tensor input to the model, or list of input arguments.
            input_kwargs: Dict of keyword arguments to the model.
            layers_to_save: List of tensor numbers to save
            random_seed: Which random seed to use
        Returns:
            Nothing, but now the ModelHistory object will have saved activations for the new input.
        """
        if random_seed is None:  # set random seed
            random_seed = random.randint(1, 4294967294)
        self.random_seed_used = random_seed
        set_random_seed(random_seed)

        self.tensor_nums_to_save = self.get_op_nums_from_user_labels(layers_to_save)

        if type(input_args) is tuple:
            input_args = list(input_args)
        elif (type(input_args) not in [list, tuple]) and (input_args is not None):
            input_args = [input_args]

        if not input_args:
            input_args = []

        if not input_kwargs:
            input_kwargs = {}

        if (
                type(model) == nn.DataParallel
        ):  # Unwrap model from DataParallel if relevant:
            model = model.module

        if (
                len(list(model.parameters())) > 0
        ):  # Get the model device by looking at the parameters:
            model_device = next(iter(model.parameters())).device
        else:
            model_device = "cpu"

        input_args = [copy.deepcopy(arg) for arg in input_args]
        input_arg_names = self._get_input_arg_names(model, input_args)
        input_kwargs = {key: copy.deepcopy(val) for key, val in input_kwargs.items()}

        self.pass_start_time = time.time()
        module_orig_forward_funcs = {}
        orig_func_defs = []

        try:
            (
                input_tensors,
                input_tensor_addresses,
            ) = self._fetch_label_move_input_tensors(
                input_args, input_arg_names, input_kwargs, model_device
            )
            buffer_tensors = list(model.buffers())
            tensors_to_decorate = input_tensors + buffer_tensors
            decorated_func_mapper = self.decorate_pytorch(torch, orig_func_defs)
            self.track_tensors = True
            for i, t in enumerate(input_tensors):
                self.log_source_tensor(t, "input", input_tensor_addresses[i])
            self.prepare_model(model, module_orig_forward_funcs, decorated_func_mapper)
            self.elapsed_time_setup = time.time() - self.pass_start_time
            outputs = model(*input_args, **input_kwargs)
            self.elapsed_time_forward_pass = (
                    time.time() - self.pass_start_time - self.elapsed_time_setup
            )
            self.track_tensors = False
            output_tensors_w_addresses_all = get_vars_of_type_from_obj(
                outputs,
                torch.Tensor,
                search_depth=5,
                return_addresses=True,
                allow_repeats=True,
            )
            # Remove duplicate addresses
            addresses_used = []
            output_tensors_w_addresses = []
            for entry in output_tensors_w_addresses_all:
                if entry[1] in addresses_used:
                    continue
                output_tensors_w_addresses.append(entry)
                addresses_used.append(entry[1])

            output_tensors = [t for t, _, _ in output_tensors_w_addresses]
            output_tensor_addresses = [
                addr for _, addr, _ in output_tensors_w_addresses
            ]

            for t in output_tensors:
                self.output_layers.append(t.tl_tensor_label_raw)
                self.raw_tensor_dict[t.tl_tensor_label_raw].is_output_parent = True
            tensors_to_undecorate = tensors_to_decorate + output_tensors
            self.undecorate_pytorch(torch, orig_func_defs, tensors_to_undecorate)
            self.cleanup_model(model, module_orig_forward_funcs, decorated_func_mapper)
            self.postprocess(output_tensors, output_tensor_addresses)
            decorated_func_mapper.clear()

        except (
                Exception
        ) as e:  # if anything fails, make sure everything gets cleaned up
            self.undecorate_pytorch(torch, orig_func_defs, input_tensors)
            self.cleanup_model(model, module_orig_forward_funcs, decorated_func_mapper)
            print(
                "************\nFeature extraction failed; returning model and environment to normal\n*************"
            )
            raise e

        finally:  # do garbage collection no matter what
            if 'input_args' in globals():
                del input_args
            if 'input_kwargs' in globals():
                del input_kwargs
            if 'input_tensors' in globals():
                del input_tensors
            if 'output_tensors' in globals():
                del output_tensors
            if 'outputs' in globals():
                del outputs
            torch.cuda.empty_cache()

    @staticmethod
    def _get_input_arg_names(model, input_args):
        input_arg_names = inspect.getfullargspec(model.forward).args
        if "self" in input_arg_names:
            input_arg_names.remove("self")
        input_arg_names = input_arg_names[0: len(input_args)]
        return input_arg_names

    @staticmethod
    def _fetch_label_move_input_tensors(
            input_args: List[Any],
            input_arg_names: List[str],
            input_kwargs: Dict,
            model_device: str,
    ) -> Tuple[List[torch.Tensor], List[str]]:
        """Fetches input tensors, gets their addresses, and moves them to the model device.

        Args:
            input_args: input arguments
            input_arg_names: name of input arguments
            input_kwargs: input keyword arguments
            model_device: model device

        Returns:
            input tensors and their addresses
        """
        input_arg_tensors = [
            get_vars_of_type_from_obj(
                arg, torch.Tensor, search_depth=5, return_addresses=True
            )
            for arg in input_args
        ]
        input_kwarg_tensors = [
            get_vars_of_type_from_obj(
                kwarg, torch.Tensor, search_depth=5, return_addresses=True
            )
            for kwarg in input_kwargs.values()
        ]
        for a, arg in enumerate(input_args):
            for i, (t, addr, addr_full) in enumerate(input_arg_tensors[a]):
                t_moved = t.to(model_device)
                input_arg_tensors[a][i] = (t_moved, addr, addr_full)
                if not addr_full:
                    input_args[a] = t_moved
                else:
                    nested_assign(input_args[a], addr_full, t_moved)

        for k, (key, val) in enumerate(input_kwargs.items()):
            for i, (t, addr, addr_full) in enumerate(input_kwarg_tensors[k]):
                t_moved = t.to(model_device)
                input_kwarg_tensors[k][i] = (t_moved, addr, addr_full)
                if not addr_full:
                    input_kwargs[key] = t_moved
                else:
                    nested_assign(input_kwargs[key], addr_full, t_moved)

        input_tensors = []
        input_tensor_addresses = []
        for a, arg_tensors in enumerate(input_arg_tensors):
            for t, addr, addr_full in arg_tensors:
                input_tensors.append(t)
                tensor_addr = f"input.{input_arg_names[a]}"
                if addr != "":
                    tensor_addr += f".{addr}"
                input_tensor_addresses.append(tensor_addr)

        for a, kwarg_tensors in enumerate(input_kwarg_tensors):
            for t, addr, addr_full in kwarg_tensors:
                input_tensors.append(t)
                tensor_addr = f"input.{list(input_kwargs.keys())[a]}"
                if addr != "":
                    tensor_addr += f".{addr}"
                input_tensor_addresses.append(tensor_addr)

        return input_tensors, input_tensor_addresses

    def log_source_tensor(
            self, t: torch.Tensor, source: str, extra_address: Optional[str] = None
    ):
        if self.logging_mode == "exhaustive":
            self.log_source_tensor_exhaustive(t, source, extra_address)
        elif self.logging_mode == "fast":
            self.log_source_tensor_fast(t, source)

    def log_source_tensor_exhaustive(
            self, t: torch.Tensor, source: str, extra_addr: Optional[str] = None
    ):
        """Takes in an input or buffer tensor, marks it in-place with relevant information, and
        adds it to the log.

        Args:
            t: the tensor
            source: either 'input' or 'buffer'
            extra_addr: either the buffer address or the input address
        """
        layer_type = source
        # Fetch counters and increment to be ready for next tensor to be logged
        self.tensor_counter += 1
        self.raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self.tensor_counter
        layer_type_num = self.raw_layer_type_counter[layer_type]

        tensor_label = f"{layer_type}_{layer_type_num}_raw"

        if source == "input":
            is_input_layer = True
            has_input_ancestor = True
            input_output_address = extra_addr
            is_buffer_layer = False
            buffer_address = None
            buffer_parent = None
            initialized_inside_model = False
            has_internally_initialized_ancestor = False
            input_ancestors = {tensor_label}
            internally_initialized_ancestors = set()
            operation_equivalence_type = (
                f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
            )
        elif source == "buffer":
            is_input_layer = False
            has_input_ancestor = False
            input_output_address = None
            is_buffer_layer = True
            buffer_address = extra_addr
            initialized_inside_model = True
            has_internally_initialized_ancestor = True
            internally_initialized_ancestors = {tensor_label}
            input_ancestors = set()
            operation_equivalence_type = f"buffer_{extra_addr}"
            if hasattr(t, 'tl_buffer_parent'):
                buffer_parent = t.tl_buffer_parent
            else:
                buffer_parent = None
        else:
            raise ValueError("source must be either 'input' or 'buffer'")

        fields_dict = {
            # General info:
            "tensor_label_raw": tensor_label,
            "layer_label_raw": tensor_label,
            "realtime_tensor_num": realtime_tensor_num,
            "operation_num": None,
            "source_model_history": self,
            "pass_finished": False,
            # Label Info:
            "layer_label": None,
            "layer_label_short": None,
            "layer_label_w_pass": None,
            "layer_label_w_pass_short": None,
            "layer_label_no_pass": None,
            "layer_label_no_pass_short": None,
            "layer_type": layer_type,
            "layer_type_num": layer_type_num,
            "layer_total_num": None,
            "pass_num": 1,
            "layer_passes_total": 1,
            "lookup_keys": [],
            # Saved tensor info:
            "tensor_contents": None,
            "has_saved_activations": False,
            "activation_postfunc": self.activation_postfunc,
            "detach_saved_tensor": self.detach_saved_tensors,
            "output_device": self.output_device,
            "function_args_saved": False,
            "creation_args": None,
            "creation_kwargs": None,
            "tensor_shape": tuple(t.shape),
            "tensor_dtype": t.dtype,
            "tensor_fsize": get_tensor_memory_amount(t),
            "tensor_fsize_nice": human_readable_size(get_tensor_memory_amount(t)),
            # Get-item complexities
            "was_getitem_applied": False,
            "children_tensor_versions": {},
            # Grad info:
            "grad_contents": None,
            "save_gradients": self.save_gradients,
            "has_saved_grad": False,
            "grad_shapes": None,
            "grad_dtypes": None,
            "grad_fsizes": 0,
            "grad_fsizes_nice": human_readable_size(0),
            # Function call info:
            "func_applied": None,
            "func_applied_name": "none",
            "func_call_stack": self._get_call_stack_dicts(),
            "func_time_elapsed": 0,
            "func_rng_states": log_current_rng_states(),
            "func_argnames": tuple([]),
            "num_func_args_total": 0,
            "num_position_args": 0,
            "num_keyword_args": 0,
            "func_position_args_non_tensor": [],
            "func_keyword_args_non_tensor": {},
            "func_all_args_non_tensor": [],
            "function_is_inplace": False,
            "gradfunc": "none",
            "is_part_of_iterable_output": False,
            "iterable_output_index": None,
            # Param info:
            "computed_with_params": False,
            "parent_params": [],
            "parent_param_barcodes": [],
            "parent_param_passes": {},
            "num_param_tensors": 0,
            "parent_param_shapes": [],
            "num_params_total": int(0),
            "parent_params_fsize": 0,
            "parent_params_fsize_nice": human_readable_size(0),
            # Corresponding layer info:
            "operation_equivalence_type": operation_equivalence_type,
            "equivalent_operations": self.equivalent_operations[
                operation_equivalence_type
            ],
            "same_layer_operations": [],
            # Graph info:
            "parent_layers": [],
            "has_parents": False,
            "parent_layer_arg_locs": {"args": {}, "kwargs": {}},
            "orig_ancestors": {tensor_label},
            "child_layers": [],
            "has_children": False,
            "sibling_layers": [],
            "has_siblings": False,
            "spouse_layers": [],
            "has_spouses": False,
            "is_input_layer": is_input_layer,
            "has_input_ancestor": has_input_ancestor,
            "input_ancestors": input_ancestors,
            "min_distance_from_input": None,
            "max_distance_from_input": None,
            "is_output_layer": False,
            "is_output_parent": False,
            "is_last_output_layer": False,
            "is_output_ancestor": False,
            "output_descendents": set(),
            "min_distance_from_output": None,
            "max_distance_from_output": None,
            "input_output_address": input_output_address,
            "is_buffer_layer": is_buffer_layer,
            "buffer_address": buffer_address,
            "buffer_pass": None,
            "buffer_parent": buffer_parent,
            "initialized_inside_model": initialized_inside_model,
            "has_internally_initialized_ancestor": has_internally_initialized_ancestor,
            "internally_initialized_parents": [],
            "internally_initialized_ancestors": internally_initialized_ancestors,
            "terminated_inside_model": False,
            # Conditional info:
            "is_terminal_bool_layer": False,
            "is_atomic_bool_layer": False,
            "atomic_bool_val": None,
            "in_cond_branch": False,
            "cond_branch_start_children": [],
            # Module info:
            "is_computed_inside_submodule": False,
            "containing_module_origin": None,
            "containing_modules_origin_nested": [],
            "module_nesting_depth": 0,
            "modules_entered": [],
            "modules_entered_argnames": defaultdict(list),
            "module_passes_entered": [],
            "is_submodule_input": False,
            "modules_exited": [],
            "module_passes_exited": [],
            "is_submodule_output": False,
            "is_bottom_level_submodule_output": False,
            "bottom_level_submodule_pass_exited": None,
            "module_entry_exit_threads_inputs": [],
            "module_entry_exit_thread_output": [],
        }

        self._make_tensor_log_entry(t, fields_dict, (), {}, self.activation_postfunc)

        # Tag the tensor itself with its label, and with a reference to the model history log.
        t.tl_tensor_label_raw = tensor_label

        # Log info to ModelHistory
        self.equivalent_operations[operation_equivalence_type].add(
            t.tl_tensor_label_raw
        )
        if source == "input":
            self.input_layers.append(tensor_label)
        if source == "buffer":
            self.buffer_layers.append(tensor_label)
            self.internally_initialized_layers.append(tensor_label)

    def log_source_tensor_fast(self, t: torch.Tensor, source: str):
        """NOTES TO SELF--fields to change are:
        for ModelHistory: pass timing, num tensors saved, tensor fsize
        for Tensors: tensor contents, fsize, args, kwargs, make sure to clear gradients.
        Add a minimal postprocessing thing for tallying stuff pertaining to saved tensors.
        Have some minimal checker to make sure the graph didn't change.
        """
        layer_type = source
        # Fetch counters and increment to be ready for next tensor to be logged
        self.tensor_counter += 1
        self.raw_layer_type_counter[layer_type] += 1
        layer_type_num = self.raw_layer_type_counter[layer_type]

        tensor_label_raw = f"{layer_type}_{layer_type_num}_raw"
        t.tl_tensor_label_raw = tensor_label_raw
        if tensor_label_raw in self.orphan_layers:
            return
        orig_tensor_label = self.raw_to_final_layer_labels[tensor_label_raw]
        if orig_tensor_label in self.unlogged_layers:
            return
        orig_tensor_entry = self.layer_dict_all_keys[orig_tensor_label]
        if (self.tensor_nums_to_save == "all") or (
                orig_tensor_entry.realtime_tensor_num in self.tensor_nums_to_save
        ):
            self.layers_with_saved_activations.append(orig_tensor_entry.layer_label)
            orig_tensor_entry.save_tensor_data(
                t, [], {}, self.save_function_args, self.activation_postfunc
            )

        orig_tensor_entry.tensor_shape = tuple(t.shape)
        orig_tensor_entry.tensor_dtype = t.dtype
        orig_tensor_entry.tensor_fsize = get_tensor_memory_amount(t)
        orig_tensor_entry.tensor_fsize_nice = human_readable_size(
            get_tensor_memory_amount(t)
        )

    def log_function_output_tensors(
            self,
            func: Callable,
            args: Tuple[Any],
            kwargs: Dict[str, Any],
            arg_copies: Tuple[Any],
            kwarg_copies: Dict[str, Any],
            out_orig: Any,
            func_time_elapsed: float,
            func_rng_states: Dict,
            is_bottom_level_func: bool,
    ):
        if self.logging_mode == "exhaustive":
            self.log_function_output_tensors_exhaustive(
                func,
                args,
                kwargs,
                arg_copies,
                kwarg_copies,
                out_orig,
                func_time_elapsed,
                func_rng_states,
                is_bottom_level_func,
            )
        elif self.logging_mode == "fast":
            self.log_function_output_tensors_fast(
                func,
                args,
                kwargs,
                arg_copies,
                kwarg_copies,
                out_orig,
                func_time_elapsed,
                func_rng_states,
                is_bottom_level_func,
            )

    def log_function_output_tensors_exhaustive(
            self,
            func: Callable,
            args: Tuple[Any],
            kwargs: Dict[str, Any],
            arg_copies: Tuple[Any],
            kwarg_copies: Dict[str, Any],
            out_orig: Any,
            func_time_elapsed: float,
            func_rng_states: Dict,
            is_bottom_level_func: bool,
    ):
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
        layer_type = func_name.lower().replace("_", "")
        all_args = list(args) + list(kwargs.values())

        fields_dict = (
            {}
        )  # dict storing the arguments for initializing the new log entry

        non_tensor_args = [arg for arg in args if not self._check_if_tensor_arg(arg)]
        non_tensor_kwargs = {
            key: val
            for key, val in kwargs.items()
            if not self._check_if_tensor_arg(val)
        }
        arg_tensors = get_vars_of_type_from_obj(
            all_args, torch.Tensor, [torch.nn.Parameter]
        )
        parent_layer_labels = get_attr_values_from_tensor_list(
            arg_tensors, "tl_tensor_label_raw"
        )
        parent_layer_entries = [self[label] for label in parent_layer_labels]

        # General info
        fields_dict["layer_type"] = layer_type
        fields_dict["detach_saved_tensor"] = self.detach_saved_tensors
        fields_dict["output_device"] = self.output_device

        # Grad info:
        fields_dict["grad_contents"] = None
        fields_dict["save_gradients"] = self.save_gradients
        fields_dict["has_saved_grad"] = False
        fields_dict["grad_shapes"] = None
        fields_dict["grad_dtypes"] = None
        fields_dict["grad_fsizes"] = 0
        fields_dict["grad_fsizes_nice"] = human_readable_size(0)

        # Function call info
        fields_dict["func_applied"] = func
        fields_dict["func_applied_name"] = func_name
        fields_dict["func_call_stack"] = self._get_call_stack_dicts()
        fields_dict["func_time_elapsed"] = func_time_elapsed
        fields_dict["func_rng_states"] = func_rng_states
        fields_dict["func_argnames"] = self.func_argnames[func_name.strip('_')]
        fields_dict["num_func_args_total"] = len(args) + len(kwargs)
        fields_dict["num_position_args"] = len(args)
        fields_dict["num_keyword_args"] = len(kwargs)
        fields_dict["func_position_args_non_tensor"] = non_tensor_args
        fields_dict["func_keyword_args_non_tensor"] = non_tensor_kwargs
        fields_dict["func_all_args_non_tensor"] = non_tensor_args + list(
            non_tensor_kwargs.values()
        )

        # Graph info
        parent_layer_arg_locs = self._get_parent_tensor_function_call_location(
            parent_layer_entries, args, kwargs
        )
        (
            input_ancestors,
            internally_initialized_ancestors,
        ) = self._get_ancestors_from_parents(parent_layer_entries)
        internal_parent_layer_labels = [
            label
            for label in parent_layer_labels
            if self[label].has_internally_initialized_ancestor
        ]

        fields_dict["parent_layers"] = parent_layer_labels
        fields_dict["parent_layer_arg_locs"] = parent_layer_arg_locs
        fields_dict["has_parents"] = len(fields_dict["parent_layers"]) > 0
        fields_dict["orig_ancestors"] = input_ancestors.union(
            internally_initialized_ancestors
        )
        fields_dict["child_layers"] = []
        fields_dict["has_children"] = False
        fields_dict["sibling_layers"] = []
        fields_dict["has_siblings"] = False
        fields_dict["spouse_layers"] = []
        fields_dict["has_spouses"] = False
        fields_dict["is_input_layer"] = False
        fields_dict["has_input_ancestor"] = len(input_ancestors) > 0
        fields_dict["input_ancestors"] = input_ancestors
        fields_dict["min_distance_from_input"] = None
        fields_dict["max_distance_from_input"] = None
        fields_dict["is_output_layer"] = False
        fields_dict["is_output_parent"] = False
        fields_dict["is_last_output_layer"] = False
        fields_dict["is_output_ancestor"] = False
        fields_dict["output_descendents"] = set()
        fields_dict["min_distance_from_output"] = None
        fields_dict["max_distance_from_output"] = None
        fields_dict["input_output_address"] = None
        fields_dict["is_buffer_layer"] = False
        fields_dict["buffer_address"] = None
        fields_dict["buffer_pass"] = None
        fields_dict["buffer_parent"] = None
        fields_dict["initialized_inside_model"] = len(parent_layer_labels) == 0
        fields_dict["has_internally_initialized_ancestor"] = (
                len(internally_initialized_ancestors) > 0
        )
        fields_dict["internally_initialized_parents"] = internal_parent_layer_labels
        fields_dict[
            "internally_initialized_ancestors"
        ] = internally_initialized_ancestors
        fields_dict["terminated_inside_model"] = False
        fields_dict["is_terminal_bool_layer"] = False
        fields_dict["in_cond_branch"] = False
        fields_dict["cond_branch_start_children"] = []

        # Param info
        arg_parameters = get_vars_of_type_from_obj(
            all_args, torch.nn.parameter.Parameter
        )
        parent_param_passes = self._process_parent_param_passes(arg_parameters)
        indiv_param_barcodes = list(parent_param_passes.keys())

        fields_dict["computed_with_params"] = len(parent_param_passes) > 0
        fields_dict["parent_params"] = arg_parameters
        fields_dict["parent_param_barcodes"] = indiv_param_barcodes
        fields_dict["parent_param_passes"] = parent_param_passes
        fields_dict["num_param_tensors"] = len(arg_parameters)
        fields_dict["parent_param_shapes"] = [
            tuple(param.shape) for param in arg_parameters
        ]
        fields_dict["num_params_total"] = int(
            np.sum([np.prod(shape) for shape in fields_dict["parent_param_shapes"]])
        )
        fields_dict["parent_params_fsize"] = int(
            np.sum([get_tensor_memory_amount(p) for p in arg_parameters])
        )
        fields_dict["parent_params_fsize_nice"] = human_readable_size(
            fields_dict["parent_params_fsize"]
        )

        # Module info

        containing_modules_origin_nested = self._get_input_module_info(arg_tensors)
        if len(containing_modules_origin_nested) > 0:
            is_computed_inside_submodule = True
            containing_module_origin = containing_modules_origin_nested[-1]
        else:
            is_computed_inside_submodule = False
            containing_module_origin = None

        fields_dict["is_computed_inside_submodule"] = is_computed_inside_submodule
        fields_dict["containing_module_origin"] = containing_module_origin
        fields_dict[
            "containing_modules_origin_nested"
        ] = containing_modules_origin_nested
        fields_dict["module_nesting_depth"] = len(containing_modules_origin_nested)
        fields_dict["modules_entered"] = []
        fields_dict["modules_entered_argnames"] = defaultdict(list)
        fields_dict["module_passes_entered"] = []
        fields_dict["is_submodule_input"] = False
        fields_dict["modules_exited"] = []
        fields_dict["module_passes_exited"] = []
        fields_dict["is_submodule_output"] = False
        fields_dict["is_bottom_level_submodule_output"] = False
        fields_dict["bottom_level_submodule_pass_exited"] = None
        fields_dict["module_entry_exit_threads_inputs"] = {
            p.tensor_label_raw: p.module_entry_exit_thread_output[:]
            for p in parent_layer_entries
        }
        fields_dict["module_entry_exit_thread_output"] = []

        is_part_of_iterable_output = any([issubclass(type(out_orig), cls) for cls in [list, tuple, dict, set]])
        fields_dict["is_part_of_iterable_output"] = is_part_of_iterable_output
        out_iter = make_var_iterable(out_orig)  # so we can iterate through it

        for i, out in enumerate(out_iter):
            if not self._output_should_be_logged(out, is_bottom_level_func):
                continue

            fields_dict_onetensor = {
                key: copy.copy(value) for key, value in fields_dict.items()
            }
            fields_to_deepcopy = [
                "parent_layer_arg_locs",
                "containing_modules_origin_nested",
                "parent_param_passes",
            ]
            for field in fields_to_deepcopy:
                fields_dict_onetensor[field] = copy.deepcopy(fields_dict[field])
            self._log_info_specific_to_single_function_output_tensor(
                out, i, args, kwargs, parent_param_passes, fields_dict_onetensor
            )
            self._make_tensor_log_entry(
                out,
                fields_dict=fields_dict_onetensor,
                t_args=arg_copies,
                t_kwargs=kwarg_copies,
                activation_postfunc=self.activation_postfunc,
            )
            new_tensor_entry = self[fields_dict_onetensor["tensor_label_raw"]]
            new_tensor_label = new_tensor_entry.tensor_label_raw
            self._update_tensor_family_links(new_tensor_entry)

            # Update relevant fields of ModelHistory
            # Add layer to relevant fields of ModelHistory:
            if fields_dict["initialized_inside_model"]:
                self.internally_initialized_layers.append(new_tensor_label)
            if fields_dict["has_input_ancestor"] and any(
                    [
                        (
                                self[parent_layer].has_internally_initialized_ancestor
                                and not self[parent_layer].has_input_ancestor
                        )
                        for parent_layer in fields_dict_onetensor["parent_layers"]
                    ]
            ):
                self.layers_where_internal_branches_merge_with_input.append(
                    new_tensor_label
                )

            # Tag the tensor itself with its label, and add a backward hook if saving gradients.
            out.tl_tensor_label_raw = fields_dict_onetensor["tensor_label_raw"]
            if self.save_gradients:
                self._add_backward_hook(out, out.tl_tensor_label_raw)

            # Check if parent is parent of a slice function, and deal with any complexities from that.
            if (new_tensor_entry.func_applied_name == "__getitem__") and (
                    len(new_tensor_entry.parent_layers) > 0
            ):
                self[new_tensor_entry.parent_layers[0]].was_getitem_applied = True

            for parent_label in new_tensor_entry.parent_layers:
                parent = self[parent_label]
                if all(
                        [
                            parent.was_getitem_applied,
                            parent.has_saved_activations,
                            self.save_function_args,
                        ]
                ):
                    parent_tensor_contents = self._get_parent_contents(
                        parent_label,
                        arg_copies,
                        kwarg_copies,
                        new_tensor_entry.parent_layer_arg_locs,
                    )
                    parent.children_tensor_versions[
                        new_tensor_entry.tensor_label_raw
                    ] = parent_tensor_contents

    @staticmethod
    def _get_parent_contents(
            parent_label, arg_copies, kwarg_copies, parent_layer_arg_locs
    ):
        """Utility function to get the value of a parent layer from the arguments passed to a function."""
        for pos, label in parent_layer_arg_locs["args"].items():
            if label == parent_label:
                return index_nested(arg_copies, pos)
        for argname, label in parent_layer_arg_locs["kwargs"].items():
            if label == parent_label:
                return index_nested(kwarg_copies, argname)
        raise ValueError("Parent layer not found in function arguments.")

    def log_function_output_tensors_fast(
            self,
            func: Callable,
            args: Tuple[Any],
            kwargs: Dict[str, Any],
            arg_copies: Tuple[Any],
            kwarg_copies: Dict[str, Any],
            out_orig: Any,
            func_time_elapsed: float,
            func_rng_states: Dict,
            is_bottom_level_func: bool,
    ):
        # Collect information.
        func_name = func.__name__
        layer_type = func_name.lower().replace("_", "")
        all_args = list(args) + list(kwargs.values())
        non_tensor_args = [arg for arg in args if not self._check_if_tensor_arg(arg)]
        non_tensor_kwargs = {
            key: val
            for key, val in kwargs.items()
            if not self._check_if_tensor_arg(val)
        }

        arg_tensors = get_vars_of_type_from_obj(
            all_args, torch.Tensor, [torch.nn.Parameter]
        )
        out_iter = make_var_iterable(out_orig)

        for i, out in enumerate(out_iter):
            if not self._output_should_be_logged(out, is_bottom_level_func):
                continue
            self.tensor_counter += 1
            self.raw_layer_type_counter[layer_type] += 1
            realtime_tensor_num = self.tensor_counter
            layer_type_num = self.raw_layer_type_counter[layer_type]
            tensor_label_raw = (
                f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"
            )
            if tensor_label_raw in self.orphan_layers:
                continue
            parent_layer_labels_raw = get_attr_values_from_tensor_list(
                arg_tensors, "tl_tensor_label_raw"
            )
            parent_layer_labels_orig = [
                self.raw_to_final_layer_labels[raw_label]
                for raw_label in parent_layer_labels_raw
            ]
            out.tl_tensor_label_raw = tensor_label_raw
            if tensor_label_raw not in self.raw_to_final_layer_labels:
                raise ValueError(
                    "The computational graph changed for this forward pass compared to the original "
                    "call to log_forward_pass (either due to different inputs or a different "
                    "random seed), so save_new_activations failed. Please re-run "
                    "log_forward_pass with the desired inputs."
                )
            orig_tensor_label = self.raw_to_final_layer_labels[tensor_label_raw]
            if orig_tensor_label in self.unlogged_layers:
                continue
            orig_tensor_entry = self[orig_tensor_label]

            # Check to make sure the graph didn't change.
            if any(
                    [
                        orig_tensor_entry.realtime_tensor_num != self.tensor_counter,
                        orig_tensor_entry.layer_type != layer_type,
                        orig_tensor_entry.tensor_label_raw != tensor_label_raw,
                        set(orig_tensor_entry.parent_layers)
                        != set(parent_layer_labels_orig),
                    ]
            ):
                raise ValueError(
                    "The computational graph changed for this forward pass compared to the original "
                    "call to log_forward_pass (either due to different inputs or a different "
                    "random seed), so save_new_activations failed. Please re-run "
                    "log_forward_pass with the desired inputs."
                )

            # Update any relevant fields.
            if (self.tensor_nums_to_save == "all") or (
                    orig_tensor_entry.realtime_tensor_num in self.tensor_nums_to_save
            ):
                self.layers_with_saved_activations.append(orig_tensor_entry.layer_label)
                orig_tensor_entry.save_tensor_data(
                    out,
                    arg_copies,
                    kwarg_copies,
                    self.save_function_args,
                    self.activation_postfunc,
                )
                for child_layer in orig_tensor_entry.child_layers:
                    if child_layer in self.output_layers:
                        child_output = self.layer_dict_main_keys[child_layer]
                        child_output.save_tensor_data(
                            out,
                            [],
                            {},
                            self.save_function_args,
                            self.activation_postfunc,
                        )

            orig_tensor_entry.tensor_shape = tuple(out.shape)
            orig_tensor_entry.tensor_dtype = out.dtype
            orig_tensor_entry.tensor_fsize = get_tensor_memory_amount(out)
            orig_tensor_entry.tensor_fsize_nice = human_readable_size(
                get_tensor_memory_amount(out)
            )
            orig_tensor_entry.func_time_elapsed = func_time_elapsed
            orig_tensor_entry.func_rng_states = func_rng_states
            orig_tensor_entry.func_position_args_non_tensor = non_tensor_args
            orig_tensor_entry.func_keyword_args_non_tensor = non_tensor_kwargs

    @staticmethod
    def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
        """Function to check whether to log the output of a function.

        Returns:
            True if the output should be logged, False otherwise.
        """
        if type(out) != torch.Tensor:  # only log if it's a tensor
            return False

        if (not hasattr(out, "tl_tensor_label_raw")) or is_bottom_level_func:
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
        def log_grad_to_model_history(_, g_out):
            self._log_tensor_grad(g_out, tensor_label)

        if t.grad_fn is not None:
            t.grad_fn.register_hook(log_grad_to_model_history)

    def _log_info_specific_to_single_function_output_tensor(
            self,
            t: torch.Tensor,
            i: int,
            args: Tuple[Any],
            kwargs: Dict[str, Any],
            parent_param_passes: Dict[str, int],
            fields_dict: Dict[str, Any],
    ):
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
        layer_type = fields_dict["layer_type"]
        indiv_param_barcodes = list(parent_param_passes.keys())
        self.tensor_counter += 1
        self.raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self.tensor_counter
        layer_type_num = self.raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

        if len(parent_param_passes) > 0:
            operation_equivalence_type = self._make_raw_param_group_barcode(
                indiv_param_barcodes, layer_type
            )
            fields_dict["operation_equivalence_type"] = operation_equivalence_type
            self.layers_computed_with_params[operation_equivalence_type].append(
                tensor_label_raw
            )
            fields_dict["pass_num"] = len(
                self.layers_computed_with_params[operation_equivalence_type]
            )
        else:
            operation_equivalence_type = self._get_operation_equivalence_type(
                args, kwargs, i, layer_type, fields_dict
            )
            fields_dict["operation_equivalence_type"] = operation_equivalence_type
            fields_dict["pass_num"] = 1

        self.equivalent_operations[operation_equivalence_type].add(tensor_label_raw)
        fields_dict["equivalent_operations"] = self.equivalent_operations[
            operation_equivalence_type
        ]

        fields_dict["function_is_inplace"] = hasattr(t, "tl_tensor_label_raw")
        fields_dict["gradfunc"] = type(t.grad_fn).__name__

        if fields_dict["is_part_of_iterable_output"]:
            fields_dict["iterable_output_index"] = i
        else:
            fields_dict["iterable_output_index"] = None

        if (t.dtype == torch.bool) and (t.dim()) == 0:
            fields_dict["is_atomic_bool_layer"] = True
            fields_dict["atomic_bool_val"] = t.item()
        else:
            fields_dict["is_atomic_bool_layer"] = False
            fields_dict["atomic_bool_val"] = None

        # General info
        fields_dict["tensor_label_raw"] = tensor_label_raw
        fields_dict["layer_total_num"] = None
        fields_dict["same_layer_operations"] = []
        fields_dict["realtime_tensor_num"] = realtime_tensor_num
        fields_dict["operation_num"] = None
        fields_dict["source_model_history"] = self
        fields_dict["pass_finished"] = False

        # Other labeling info
        fields_dict["layer_label"] = None
        fields_dict["layer_label_short"] = None
        fields_dict["layer_label_w_pass"] = None
        fields_dict["layer_label_w_pass_short"] = None
        fields_dict["layer_label_no_pass"] = None
        fields_dict["layer_label_no_pass_short"] = None
        fields_dict["layer_type"] = layer_type
        fields_dict["layer_label_raw"] = tensor_label_raw
        fields_dict["layer_type_num"] = layer_type_num
        fields_dict["pass_num"] = 1
        fields_dict["layer_passes_total"] = 1
        fields_dict["lookup_keys"] = []

        # Saved tensor info
        fields_dict["tensor_contents"] = None
        fields_dict["has_saved_activations"] = False
        fields_dict["activation_postfunc"] = self.activation_postfunc
        fields_dict["function_args_saved"] = False
        fields_dict["creation_args"] = None
        fields_dict["creation_kwargs"] = None
        fields_dict["tensor_shape"] = tuple(t.shape)
        fields_dict["tensor_dtype"] = t.dtype
        fields_dict["tensor_fsize"] = get_tensor_memory_amount(t)
        fields_dict["tensor_fsize_nice"] = human_readable_size(
            fields_dict["tensor_fsize"]
        )

        # Slice function complexities (i.e., what if a parent tensor is changed in place)
        fields_dict["was_getitem_applied"] = False
        fields_dict["children_tensor_versions"] = {}

        # If internally initialized, fix this information:
        if len(fields_dict["parent_layers"]) == 0:
            fields_dict["initialized_inside_model"] = True
            fields_dict["has_internally_initialized_ancestor"] = True
            fields_dict["internally_initialized_parents"] = []
            fields_dict["internally_initialized_ancestors"] = {tensor_label_raw}

    def _make_tensor_log_entry(
            self,
            t: torch.Tensor,
            fields_dict: Dict,
            t_args: Optional[Tuple] = None,
            t_kwargs: Optional[Dict] = None,
            activation_postfunc: Optional[Callable] = None,
    ):
        """
        Given a tensor, adds it to the model_history, additionally saving the activations and input
        arguments if specified. Also tags the tensor itself with its raw tensor label
        and a pointer to ModelHistory.

        Args:
            t: tensor to log
            fields_dict: dictionary of fields to log in TensorLogEntry
            t_args: Positional arguments to the function that created the tensor
            t_kwargs: Keyword arguments to the function that created the tensor
            activation_postfunc: Function to apply to activations before saving them.
        """
        if t_args is None:
            t_args = []
        if t_kwargs is None:
            t_kwargs = {}

        new_entry = TensorLogEntry(fields_dict)
        if (self.tensor_nums_to_save == "all") or (
                new_entry.realtime_tensor_num in self.tensor_nums_to_save
        ):
            new_entry.save_tensor_data(
                t, t_args, t_kwargs, self.save_function_args, activation_postfunc
            )
            self.layers_with_saved_activations.append(new_entry.tensor_label_raw)
        self.raw_tensor_dict[new_entry.tensor_label_raw] = new_entry
        self.raw_tensor_labels_list.append(new_entry.tensor_label_raw)

        return new_entry

    def _log_tensor_grad(self, grad: torch.Tensor, tensor_label_raw: str):
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
            self.layers_with_saved_gradients = sorted(
                self.layers_with_saved_gradients, key=lambda x: layer_order[x]
            )
        tensor_log_entry = self[tensor_label]
        tensor_log_entry.log_tensor_grad(grad)

    @staticmethod
    def _check_if_tensor_arg(arg: Any) -> bool:
        """Helper function to check if an argument either is a tensor or is a list/tuple containing a tensor.

        Args:
            arg: argument

        Returns:
            True if arg is or contains a tensor, false otherwise
        """
        if issubclass(type(arg), torch.Tensor):
            return True
        elif type(arg) in [list, tuple]:
            for elt in arg:
                if issubclass(type(elt), torch.Tensor):
                    return True
        else:
            return False

    def _get_parent_tensor_function_call_location(
            self,
            parent_log_entries: List[TensorLogEntry],
            args: Tuple[Any],
            kwargs: Dict[Any, Any],
    ) -> Dict:
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
        tensor_all_arg_positions = {"args": {}, "kwargs": {}}
        arg_struct_dict = {"args": args, "kwargs": kwargs}

        for parent_entry in parent_log_entries:
            for arg_type in ["args", "kwargs"]:
                arg_struct = arg_struct_dict[arg_type]
                self._find_arg_positions_for_single_parent(
                    parent_entry, arg_type, arg_struct, tensor_all_arg_positions
                )

        return tensor_all_arg_positions

    @staticmethod
    def _find_arg_positions_for_single_parent(
            parent_entry: TensorLogEntry,
            arg_type: str,
            arg_struct: Union[List, Tuple, Dict],
            tensor_all_arg_positions: Dict,
    ):
        """Helper function that finds where a single parent tensor is used in either the args or kwargs of a function,
        and updates a dict that tracks this information.

        Args:
            parent_entry: Parent tensor
            arg_type: 'args' or 'kwargs'
            arg_struct: args or kwargs
            tensor_all_arg_positions: dict tracking where the tensors are used
        """
        iterfunc_dict = {
            "args": enumerate,
            "kwargs": lambda x: x.items(),
            list: enumerate,
            tuple: enumerate,
            dict: lambda x: x.items(),
        }
        iterfunc = iterfunc_dict[arg_type]

        for arg_key, arg in iterfunc(arg_struct):
            if getattr(arg, "tl_tensor_label_raw", -1) == parent_entry.tensor_label_raw:
                tensor_all_arg_positions[arg_type][
                    arg_key
                ] = parent_entry.tensor_label_raw
            elif type(arg) in [list, tuple, dict]:
                iterfunc2 = iterfunc_dict[type(arg)]
                for sub_arg_key, sub_arg in iterfunc2(arg):
                    if (
                            getattr(sub_arg, "tl_tensor_label_raw", -1)
                            == parent_entry.tensor_label_raw
                    ):
                        tensor_all_arg_positions[arg_type][
                            (arg_key, sub_arg_key)
                        ] = parent_entry.tensor_label_raw

    @staticmethod
    def _get_ancestors_from_parents(
            parent_entries: List[TensorLogEntry],
    ) -> Tuple[Set[str], Set[str]]:
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
            internally_initialized_ancestors.update(
                parent_entry.internally_initialized_ancestors
            )
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
            self._add_sibling_labels_for_new_tensor(
                entry_to_update, self[parent_tensor_label]
            )

    def _add_sibling_labels_for_new_tensor(
            self, entry_to_update: TensorLogEntry, parent_tensor: TensorLogEntry
    ):
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
    def _process_parent_param_passes(
            arg_parameters: List[torch.nn.Parameter],
    ) -> Dict[str, int]:
        """Utility function to mark the parameters with barcodes, and log which pass they're on.

        Args:
            arg_parameters: List of arg parameters

        Returns:

        """
        parent_param_passes = {}
        for param in arg_parameters:
            if not hasattr(param, "tl_param_barcode"):
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

    def _get_operation_equivalence_type(
            self, args: Tuple, kwargs: Dict, i: int, layer_type: str, fields_dict: Dict
    ):
        arg_hash = self._get_hash_from_args(args, kwargs)
        operation_equivalence_type = f"{layer_type}_{arg_hash}"
        if fields_dict["is_part_of_iterable_output"]:
            operation_equivalence_type += f"_outindex{i}"
        if fields_dict["containing_module_origin"] is not None:
            module_str = fields_dict["containing_module_origin"][0]
            operation_equivalence_type += f"_module{module_str}"
        return operation_equivalence_type

    @staticmethod
    def _get_hash_from_args(args, kwargs):
        """
        Get a hash from the args and kwargs of a function call, excluding any tracked tensors.
        """
        args_to_hash = []
        for a, arg in enumerate(list(args) + list(kwargs.values())):
            if hasattr(arg, "tl_tensor_label_raw"):
                args_to_hash.append(f"arg{a}{arg.shape}")
            else:
                arg_iter = make_var_iterable(arg)
                for i, arg_elem in enumerate(arg_iter):
                    if not hasattr(arg_elem, "tl_tensor_label_raw") and not isinstance(
                            arg_elem, torch.nn.Parameter
                    ):
                        args_to_hash.append(arg_elem)
                    elif hasattr(arg_elem, "tl_tensor_label_raw"):
                        args_to_hash.append(f"arg{a}_iter{i}_{arg_elem.shape}")

        if len(args_to_hash) == 0:
            return "no_args"
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
        thread_modules = tensor_entry.module_entry_exit_thread_output[:]
        for thread_module in thread_modules:
            if thread_module[0] == "+":
                containing_modules.append(thread_module[1:])
            elif (thread_module[0] == "-") and (
                    thread_module[1:] in containing_modules
            ):
                containing_modules.remove(thread_module[1:])
        return containing_modules

    def _get_input_module_info(self, arg_tensors: List[torch.Tensor]):
        """Utility function to extract information about module entry/exit from input tensors.

        Args:
            arg_tensors: List of input tensors

        Returns:
            Variables with module entry/exit information
        """
        max_input_module_nesting = 0
        most_nested_containing_modules = []
        for t in arg_tensors:
            tensor_label = getattr(t, "tl_tensor_label_raw", -1)
            if tensor_label == -1:
                continue
            tensor_entry = self[tensor_label]
            containing_modules = self._update_tensor_containing_modules(tensor_entry)
            if len(containing_modules) > max_input_module_nesting:
                max_input_module_nesting = len(containing_modules)
                most_nested_containing_modules = containing_modules[:]
        return most_nested_containing_modules

    @staticmethod
    def _get_call_stack_dicts():
        call_stack = inspect.stack()
        call_stack = [
            inspect.getframeinfo(call_stack[i][0], context=19)
            for i in range(len(call_stack))
        ]
        call_stack_dicts = [
            {
                "call_fname": caller.filename,
                "call_linenum": caller.lineno,
                "function": caller.function,
                "code_context": caller.code_context,
            }
            for caller in call_stack
        ]

        for call_stack_dict in call_stack_dicts:
            if is_iterable(call_stack_dict['code_context']):
                call_stack_dict['code_context_str'] = ''.join(call_stack_dict['code_context'])
            else:
                call_stack_dict['code_context_str'] = str(call_stack_dict['code_context'])

        # Only start at the level of that first forward pass, going from shallow to deep.
        tracking = False
        filtered_dicts = []
        for d in range(len(call_stack_dicts) - 1, -1, -1):
            call_stack_dict = call_stack_dicts[d]
            if any(
                    [
                        call_stack_dict["call_fname"].endswith("model_history.py"),
                        call_stack_dict["call_fname"].endswith("torchlens/helper_funcs.py"),
                        call_stack_dict["call_fname"].endswith("torchlens/user_funcs.py"),
                        call_stack_dict["function"] == "_call_impl",
                    ]
            ):
                continue
            if call_stack_dict["function"] == "forward":
                tracking = True

            if tracking:
                filtered_dicts.append(call_stack_dict)

        return filtered_dicts

    def cleanup(self):
        """Deletes all log entries in the model."""
        for tensor_log_entry in self:
            self._remove_log_entry(tensor_log_entry, remove_references=True)
        for attr in MODEL_HISTORY_FIELD_ORDER:
            delattr(self, attr)
        torch.cuda.empty_cache()

    def _remove_log_entry(
            self, log_entry: TensorLogEntry, remove_references: bool = True
    ):
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if not attr.startswith("_") and not callable(getattr(log_entry, attr)):
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
        remove_entry_from_list(self.layers_with_saved_activations, layer_to_remove)
        remove_entry_from_list(self.layers_with_saved_gradients, layer_to_remove)
        remove_entry_from_list(
            self.layers_where_internal_branches_merge_with_input, layer_to_remove
        )

        self.conditional_branch_edges = [
            tup for tup in self.conditional_branch_edges if layer_to_remove not in tup
        ]

        # Now any nested fields.

        for group_label, group_tensors in self.layers_computed_with_params.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
        self.layers_computed_with_params = {
            k: v for k, v in self.layers_computed_with_params.items() if len(v) > 0
        }

        for group_label, group_tensors in self.equivalent_operations.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
        self.equivalent_operations = {
            k: v for k, v in self.equivalent_operations.items() if len(v) > 0
        }

        for group_label, group_tensors in self.same_layer_operations.items():
            if layer_to_remove in group_tensors:
                group_tensors.remove(layer_to_remove)
        self.same_layer_operations = {
            k: v for k, v in self.same_layer_operations.items() if len(v) > 0
        }

    # ********************************************
    # ************* Post-Processing **************
    # ********************************************

    def postprocess(
            self, output_tensors: List[torch.Tensor], output_tensor_addresses: List[str]
    ):
        """
        After the forward pass, cleans up the log into its final form.
        """
        if self.logging_mode == "fast":
            self.postprocess_fast()
            return

        # Step 1: Add dedicated output nodes

        self._add_output_layers(output_tensors, output_tensor_addresses)

        # Step 2: Trace which nodes are ancestors of output nodes

        self._find_output_ancestors()

        # Step 3: Remove orphan nodes, find nodes that don't terminate in output node

        self._remove_orphan_nodes()

        # Step 4: Find mix/max distance from input and output nodes

        if self.mark_input_output_distances:
            self._mark_input_output_distances()

        # Step 5: Starting from terminal single boolean tensors, mark the conditional branches.

        self._mark_conditional_branches()

        # Step 6: Annotate the containing modules for all internally-generated tensors (they don't know where
        # they are when they're made; have to trace breadcrumbs from tensors that came from input).

        self._fix_modules_for_internal_tensors()

        # Step 7: Fix the buffer passes and parent infomration.

        self._fix_buffer_layers()

        # Step 8: Identify all loops, mark repeated layers.

        self._assign_corresponding_tensors_to_same_layer()

        # Step 9: Go down tensor list, get the mapping from raw tensor names to final tensor names.

        self._map_raw_tensor_labels_to_final_tensor_labels()

        # Step 10: Go through and log information pertaining to all layers:
        self._log_final_info_for_all_layers()

        # Step 11: Rename the raw tensor entries in the fields of ModelHistory:
        self._rename_model_history_layer_names()
        self._trim_and_reorder_model_history_fields()

        # Step 12: And one more pass to delete unused layers from the record and do final tidying up:
        self._remove_unwanted_entries_and_log_remaining()

        # Step 13: Undecorate all saved tensors and remove saved grad_fns.
        self._undecorate_all_saved_tensors()

        # Step 14: Clear the cache after any tensor deletions for garbage collection purposes:
        torch.cuda.empty_cache()

        # Step 15: Log time elapsed.
        self._log_time_elapsed()

        # Step 16: log the pass as finished, changing the ModelHistory behavior to its user-facing version.

        self._set_pass_finished()

    def postprocess_fast(self):
        self._trim_and_reorder_model_history_fields()
        self._remove_unwanted_entries_and_log_remaining()
        self._undecorate_all_saved_tensors()
        torch.cuda.empty_cache()
        self._log_time_elapsed()
        self._set_pass_finished()

    def _add_output_layers(
            self, output_tensors: List[torch.Tensor], output_addresses: List[str]
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
            self.tensor_counter += 1
            new_output_node.tensor_label_raw = f"output_{i + 1}_raw"
            new_output_node.layer_label_raw = new_output_node.tensor_label_raw
            new_output_node.realtime_tensor_num = self.tensor_counter
            output_address = "output"
            if output_addresses[i] != "":
                output_address += f".{output_addresses[i]}"
            new_output_node.input_output_address = output_address

            # Fix function information:

            new_output_node.func_applied = identity
            new_output_node.func_applied_name = "none"
            new_output_node.func_call_stack = self._get_call_stack_dicts()
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

            self.raw_tensor_dict[new_output_node.tensor_label_raw] = new_output_node
            self.raw_tensor_labels_list.append(new_output_node.tensor_label_raw)

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
        orig_nodes = set(self.raw_tensor_labels_list)
        nodes_seen = set()
        node_stack = self.input_layers + self.output_layers
        while len(node_stack) > 0:
            tensor_label = node_stack.pop()
            nodes_seen.add(tensor_label)
            tensor_entry = self.raw_tensor_dict[tensor_label]
            if (len(tensor_entry.child_layers) == 0) and (
                    not tensor_entry.is_output_layer
            ):
                self._log_internally_terminated_tensor(tensor_label)
            for next_label in tensor_entry.child_layers + tensor_entry.parent_layers:
                if next_label not in nodes_seen:
                    node_stack.append(next_label)

        orphan_nodes = orig_nodes - nodes_seen
        self.orphan_layers = list(orphan_nodes)

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
        self._flood_graph_from_input_or_output_nodes("input")
        self._flood_graph_from_input_or_output_nodes("output")

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
            self._update_node_distance_vals(
                current_node, min_field, max_field, nodes_since_start
            )

            setattr(current_node, marker_field, True)
            getattr(current_node, layer_logging_field).add(orig_node)

            for next_node_label in getattr(current_node, forward_field):
                if self._check_whether_to_add_node_to_flood_stack(
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

    @staticmethod
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
            self._find_and_mark_same_layer_operations_starting_from_node(node)

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
            self._fetch_and_process_next_isomorphic_nodes(
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

        self._assign_and_log_isomorphic_nodes_to_same_layers(
            iso_node_groups, node_to_subgraph_dict, adjacent_subgraphs
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

        successor_nodes_dict = self._log_collisions_and_get_candidate_next_nodes(
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
            ) = self._get_next_candidate_node(successor_nodes_dict)
            if candidate_node_label is None:
                break

            new_equivalent_nodes = self._get_nodes_isomorphic_to_candidate_node(
                candidate_node_label,
                candidate_node_neighbor_type,
                candidate_node_subgraph,
                successor_nodes_dict,
            )

            # Now log this new set of isomorphic nodes.

            self._log_new_isomorphic_nodes(
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
                        self._check_and_mark_subgraph_adjacency(
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

    @staticmethod
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

    @staticmethod
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
        same_layer_node_groups = self._group_isomorphic_nodes_to_same_layers(
            iso_node_groups, node_to_subgraph_dict, adjacent_subgraphs
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
                if (not parent_node.has_input_ancestor) and (
                        parent_label not in nodes_seen
                ):
                    self._fix_modules_for_single_internal_tensor(
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
                self._fix_modules_for_single_internal_tensor(
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

    @staticmethod
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

        self.raw_tensor_labels_list.remove(buffer_to_remove.tensor_label_raw)
        self.raw_tensor_dict.pop(buffer_to_remove.tensor_label_raw)

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
        self.raw_to_final_layer_labels = raw_to_final_layer_labels
        self.final_to_raw_layer_labels = final_to_raw_layer_labels

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
            self._replace_layer_names_for_tensor_entry(tensor_entry)

            # Log the module hierarchy information:
            self._log_module_hierarchy_info_for_layer(tensor_entry)
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
                    self.raw_to_final_layer_labels[raw_name]
                    for raw_name in orig_layer_names
                ]
            )
            setattr(tensor_entry, field, new_layer_names)

        # Fix the arg locations field:
        for arg_type in ["args", "kwargs"]:
            for key, value in tensor_entry.parent_layer_arg_locs[arg_type].items():
                tensor_entry.parent_layer_arg_locs[arg_type][
                    key
                ] = self.raw_to_final_layer_labels[value]

        # Fix the field names for different children tensor versions:
        new_child_tensor_versions = {}
        for (
                child_label,
                tensor_version,
        ) in tensor_entry.children_tensor_versions.items():
            new_child_tensor_versions[
                self.raw_to_final_layer_labels[child_label]
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
        for raw_tensor_label in self.raw_tensor_labels_list:
            tensor_entry = self.raw_tensor_dict[raw_tensor_label]
            # Determine valid lookup keys and relate them to the tensor's realtime operation number:
            if tensor_entry.has_saved_activations or self.keep_unsaved_layers:
                # Add the lookup keys for the layer, to itself and to ModelHistory:
                self._add_lookup_keys_for_tensor_entry(
                    tensor_entry, i, num_logged_tensors
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
                self._trim_and_reorder_tensor_entry_fields(
                    tensor_entry
                )  # Final reformatting of fields
                i += 1
            else:
                tensors_to_remove.append(tensor_entry)
                self.unlogged_layers.append(tensor_entry.layer_label)
                self.unsaved_layers_lookup_keys.update(tensor_entry.lookup_keys)

        # Remove unused entries.
        for tensor_entry in tensors_to_remove:
            self._remove_log_entry(tensor_entry, remove_references=False)

        if (self.num_tensors_saved == len(self)) or self.keep_unsaved_layers:
            self.all_layers_logged = True
        else:
            self.all_layers_logged = False

        if self.num_tensors_saved == len(self.layer_list):
            self.all_layers_saved = True
        else:
            self.all_layers_saved = False

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
            self.lookup_keys_to_tensor_num_dict[
                lookup_key
            ] = tensor_entry.realtime_tensor_num
            self.tensor_num_to_lookup_keys_dict[
                tensor_entry.realtime_tensor_num
            ].append(lookup_key)
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
            "layers_where_internal_branches_merge_with_input",
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
                    self.raw_to_final_layer_labels[tensor_label]
                    for tensor_label in tensor_labels
                ],
            )

        new_param_tensors = {}
        for key, values in self.layers_computed_with_params.items():
            new_key = self[values[0]].layer_label
            new_param_tensors[new_key] = [
                self.raw_to_final_layer_labels[tensor_label] for tensor_label in values
            ]
        self.layers_computed_with_params = new_param_tensors

        new_equiv_operations_tensors = {}
        for key, values in self.equivalent_operations.items():
            new_equiv_operations_tensors[key] = set(
                [
                    self.raw_to_final_layer_labels[tensor_label]
                    for tensor_label in values
                ]
            )
        self.equivalent_operations = new_equiv_operations_tensors

        new_same_layer_operations = {}
        for key, values in self.same_layer_operations.items():
            new_key = self.raw_to_final_layer_labels[key]
            new_same_layer_operations[new_key] = [
                self.raw_to_final_layer_labels[tensor_label] for tensor_label in values
            ]
        self.same_layer_operations = new_same_layer_operations

        for t, (child, parent) in enumerate(self.conditional_branch_edges):
            new_child, new_parent = (
                self.raw_to_final_layer_labels[child],
                self.raw_to_final_layer_labels[parent],
            )
            self.conditional_branch_edges[t] = (new_child, new_parent)

        for module_pass, arglist in self.module_layer_argnames.items():
            inds_to_remove = []
            for a, arg in enumerate(arglist):
                raw_name = self.module_layer_argnames[module_pass][a][0]
                if raw_name not in self.raw_to_final_layer_labels:
                    inds_to_remove.append(a)
                    continue
                new_name = self.raw_to_final_layer_labels[raw_name]
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

    # ********************************************
    # *************** Visualization **************
    # ********************************************

    def render_graph(
            self,
            vis_opt: str = "unrolled",
            vis_nesting_depth: int = 1000,
            vis_outpath: str = "modelgraph",
            vis_graph_overrides: Dict = None,
            vis_node_overrides: Dict = None,
            vis_nested_node_overrides: Dict = None,
            vis_edge_overrides: Dict = None,
            vis_gradient_edge_overrides: Dict = None,
            vis_module_overrides: Dict = None,
            save_only: bool = False,
            vis_fileformat: str = "pdf",
            show_buffer_layers: bool = False,
            direction: str = "bottomup",
    ) -> None:
        """Renders the computational graph for the model.

        Args:
            vis_opt: either 'rolled' or 'unrolled'
            vis_nesting_depth: How many levels of nested modules to show; 1 for only top-level modules, 2 for two
                levels, etc.
            vis_outpath: where to store the rendered graph
            save_only: whether to only save the graph without immediately showing it
            vis_fileformat: file format to use for the rendered graph
            show_buffer_layers: whether to show the buffer layers
            direction: which way the graph should go: either 'bottomup', 'topdown', or 'leftright'

        """
        if vis_graph_overrides is None:
            vis_graph_overrides = {}
        if vis_node_overrides is None:
            vis_node_overrides = {}
        if vis_nested_node_overrides is None:
            vis_nested_node_overrides = {}
        if vis_edge_overrides is None:
            vis_edge_overrides = {}
        if vis_gradient_edge_overrides is None:
            vis_gradient_edge_overrides = {}
        if vis_module_overrides is None:
            vis_module_overrides = {}

        if not self.all_layers_logged:
            raise ValueError(
                "Must have all layers logged in order to render the graph; either save all layers,"
                "set keep_unsaved_layers to True, or use show_model_graph."
            )

        # Fix the filename if need be, to remove the extension:
        split_outpath = vis_outpath.split(".")
        if split_outpath[-1] in [
            "pdf",
            "png",
            "jpg",
            "svg",
            "jpg",
            "jpeg",
            "bmp",
            "pic",
            "tif",
            "tiff",
        ]:
            vis_outpath = ".".join(split_outpath[:-1])

        if vis_opt == "unrolled":
            entries_to_plot = self.layer_dict_main_keys
        elif vis_opt == "rolled":
            self._roll_graph()
            entries_to_plot = self.layer_dict_rolled
        else:
            raise ValueError("vis_opt must be either 'rolled' or 'unrolled'")

        if direction == "bottomup":
            rankdir = "BT"
        elif direction == "leftright":
            rankdir = "LR"
        elif direction == "topdown":
            rankdir = "TB"
        else:
            raise ValueError(
                "direction must be either 'bottomup', 'topdown', or 'leftright'."
            )

        graph_caption = (
            f"<<B>{self.model_name}</B><br align='left'/>{self.num_tensors_total} "
            f"tensors total ({self.tensor_fsize_total_nice})"
            f"<br align='left'/>{self.total_params} params total ({self.total_params_fsize_nice})<br align='left'/>>"
        )

        dot = graphviz.Digraph(
            name=self.model_name,
            comment="Computational graph for the feedforward sweep",
            format=vis_fileformat,
        )

        graph_args = {'rankdir': rankdir,
                      'label': graph_caption,
                      'labelloc': 't',
                      'labeljust': 'left',
                      'ordering': 'out'}

        for arg_name, arg_val in vis_graph_overrides.items():
            if callable(arg_val):
                graph_args[arg_name] = str(arg_val(self))
            else:
                graph_args[arg_name] = str(arg_val)

        dot.graph_attr.update(graph_args)
        dot.node_attr.update({"ordering": "out"})

        # list of edges for each subgraph; subgraphs will be created at the end.
        module_cluster_dict = defaultdict(
            lambda: {"edges": [], "has_input_ancestor": False}
        )
        collapsed_modules = set()
        edges_used = set()

        for node_barcode, node in entries_to_plot.items():
            if node.is_buffer_layer and not show_buffer_layers:
                continue
            self._add_node_to_graphviz(
                node,
                dot,
                module_cluster_dict,
                edges_used,
                vis_opt,
                collapsed_modules,
                vis_nesting_depth,
                show_buffer_layers,
                vis_node_overrides,
                vis_nested_node_overrides,
                vis_edge_overrides,
                vis_gradient_edge_overrides
            )

        # Finally, set up the subgraphs.
        self._set_up_subgraphs(dot, vis_opt, module_cluster_dict, vis_module_overrides)

        if in_notebook() and not save_only:
            display(dot)

        dot.render(vis_outpath, view=(not save_only))

    def _add_node_to_graphviz(
            self,
            node: Union[TensorLogEntry, RolledTensorLogEntry],
            graphviz_graph,
            module_edge_dict: Dict,
            edges_used: Set,
            vis_opt: str,
            collapsed_modules: Set,
            vis_nesting_depth: int = 1000,
            show_buffer_layers: bool = False,
            vis_node_overrides: Dict = None,
            vis_collapsed_node_overrides: Dict = None,
            vis_edge_overrides: Dict = None,
            vis_gradient_edge_overrides: Dict = None
    ):
        """Addes a node and its relevant edges to the graphviz figure.

        Args:
            node: node to add
            graphviz_graph: The graphviz object to add the node to.
            module_edge_dict: Dictionary of the module clusters.
            vis_opt: Whether to roll the graph or not
            vis_nesting_depth: How many levels of nested modules to show
            collapsed_modules: Labels of collapsed module nodes that have been made so far.
            show_buffer_layers: Whether to show the buffer layers
        """
        is_collapsed_module = self._check_if_collapsed_module(node, vis_nesting_depth)

        if is_collapsed_module:
            self._construct_collapsed_module_node(
                node, graphviz_graph, collapsed_modules, vis_opt, vis_nesting_depth, vis_collapsed_node_overrides
            )
            node_color = "black"
        else:
            node_color = self._construct_layer_node(
                node, graphviz_graph, show_buffer_layers, vis_opt, vis_node_overrides
            )

        self._add_edges_for_node(
            node,
            is_collapsed_module,
            vis_nesting_depth,
            node_color,
            module_edge_dict,
            edges_used,
            graphviz_graph,
            vis_opt,
            show_buffer_layers,
            vis_edge_overrides,
            vis_gradient_edge_overrides
        )

    @staticmethod
    def _check_if_collapsed_module(node, vis_nesting_depth):
        node_nesting_depth = len(node.containing_modules_origin_nested)
        if node.is_bottom_level_submodule_output:
            node_nesting_depth -= 1

        if node_nesting_depth >= vis_nesting_depth:
            return True
        else:
            return False

    def _construct_layer_node(self, node, graphviz_graph, show_buffer_layers, vis_opt, vis_node_overrides):
        # Get the address, shape, color, and line style:

        node_address, node_shape, node_color = self._get_node_address_shape_color(
            node, show_buffer_layers
        )
        node_bg_color = self._get_node_bg_color(node)

        if node.has_input_ancestor:
            line_style = "solid"
        else:
            line_style = "dashed"

        # Get the text for the node label:

        node_label = self._make_node_label(node, node_address, vis_opt)

        node_args = {'name': node.layer_label.replace(":", "pass"),
                     'label': node_label,
                     'fontcolor': node_color,
                     'color': node_color,
                     'style': f'filled,{line_style}',
                     'fillcolor': node_bg_color,
                     'shape': node_shape,
                     'ordering': 'out'
                     }
        for arg_name, arg_val in vis_node_overrides.items():
            if callable(arg_val):
                node_args[arg_name] = str(arg_val(self, node))
            else:
                node_args[arg_name] = str(arg_val)

        graphviz_graph.node(**node_args)

        if node.is_last_output_layer:
            with graphviz_graph.subgraph() as s:
                s.attr(rank="sink")
                s.node(node.layer_label.replace(":", "pass"))

        return node_color

    def _construct_collapsed_module_node(
            self, node, graphviz_graph, collapsed_modules, vis_opt, vis_nesting_depth, vis_collapsed_node_overrides
    ):
        module_address_w_pass = node.containing_modules_origin_nested[
            vis_nesting_depth - 1
            ]
        module_tuple = module_address_w_pass.split(":")
        module_output_layer = self[module_address_w_pass]
        module_output_shape = module_output_layer.tensor_shape
        module_output_fsize = module_output_layer.tensor_fsize_nice
        module_address, pass_num = module_tuple
        module_type = self.module_types[module_address]
        module_num_passes = self.module_num_passes[module_address]
        module_nparams = self.module_nparams[module_address]

        if vis_opt == "unrolled":
            node_name = "pass".join(module_tuple)
            module_num_tensors = self.module_pass_num_tensors[module_address_w_pass]
            module_has_input_ancestor = any(
                [
                    self[layer].has_input_ancestor
                    for layer in self.module_pass_layers[module_address_w_pass]
                ]
            )
        else:
            node_name = module_tuple[0]
            module_num_tensors = self.module_num_tensors[module_address]
            module_has_input_ancestor = any(
                [
                    self[layer].has_input_ancestor
                    for layer in self.module_layers[module_address]
                ]
            )

        if node_name in collapsed_modules:
            return  # collapsed node already added

        if module_num_passes == 1:
            node_title = f"<b>@{module_address}</b>"
        elif vis_opt == "unrolled" and (module_num_passes > 1):
            node_title = f"<b>@{module_address}:{pass_num}</b>"
        else:
            node_title = f"<b>@{module_address} (x{module_num_passes})</b>"

        if len(module_output_shape) > 1:
            tensor_shape_str = "x".join([str(x) for x in module_output_shape])
        elif len(node.tensor_shape) == 1:
            tensor_shape_str = f"x{module_output_shape[0]}"
        else:
            tensor_shape_str = "x1"

        if module_nparams > 0:
            bg_color = self.PARAMS_NODE_BG_COLOR
        else:
            bg_color = self.DEFAULT_BG_COLOR

        if module_has_input_ancestor:
            line_style = "solid"
        else:
            line_style = "dashed"

        node_label = (
            f"<{node_title}<br/>"
            f"{module_type}<br/>"
            f"{tensor_shape_str} ({module_output_fsize})<br/>"
            f"{module_num_tensors} layers total<br/>"
            f"{module_nparams} parameters>"
        )

        node_args = {'name': node_name,
                     'label': node_label,
                     'fontcolor': 'black',
                     'color': 'black',
                     'style': f'filled,{line_style}',
                     'fillcolor': bg_color,
                     'shape': 'box3d',
                     'ordering': 'out'
                     }

        for arg_name, arg_val in vis_collapsed_node_overrides.items():
            if callable(arg_val):
                node_args[arg_name] = str(arg_val(self, node))
            else:
                node_args[arg_name] = str(arg_val)

        graphviz_graph.node(**node_args)

    def _get_node_address_shape_color(
            self,
            node: Union[TensorLogEntry, RolledTensorLogEntry],
            show_buffer_layers: bool,
    ) -> Tuple[str, str, str]:
        """Gets the node shape, address, and color for the graphviz figure.

        Args:
            node: node to add

        Returns:
            node_address: address of the node
            node_shape: shape of the node
            node_color: color of the node
        """
        if not show_buffer_layers:
            only_non_buffer_layer = self._check_if_only_non_buffer_in_module(node)
        else:
            only_non_buffer_layer = False

        if (node.is_bottom_level_submodule_output or only_non_buffer_layer) and (
                len(node.containing_modules_origin_nested) > 0
        ):
            if type(node) == TensorLogEntry:
                module_pass_exited = node.containing_modules_origin_nested[-1]
                module, _ = module_pass_exited.split(":")
                if self.module_num_passes[module] == 1:
                    node_address = module
                else:
                    node_address = module_pass_exited
            else:
                sample_module_pass = node.containing_modules_origin_nested[-1]
                module = sample_module_pass.split(":")[0]
                node_address = module

            node_address = "<br/>@" + node_address
            node_shape = "box"
            node_color = "black"
        elif node.is_buffer_layer:
            if ((self.buffer_num_passes[node.buffer_address] == 1) or
                    (isinstance(node, RolledTensorLogEntry) and node.layer_passes_total > 1)):
                buffer_address = node.buffer_address
            else:
                buffer_address = f"{node.buffer_address}:{node.buffer_pass}"
            node_address = "<br/>@" + buffer_address
            node_shape = "box"
            node_color = self.BUFFER_NODE_COLOR
        elif node.is_output_layer or node.is_input_layer:
            node_address = "<br/>@" + node.input_output_address
            node_shape = "oval"
            node_color = "black"
        else:
            node_address = ""
            node_shape = "oval"
            node_color = "black"

        return node_address, node_shape, node_color

    def _check_if_only_non_buffer_in_module(
            self, node: Union[TensorLogEntry, RolledTensorLogEntry]
    ):
        """Utility function to check if a layer is the only non-buffer layer in the module"""
        # Check whether it leaves its module:
        if not (
                (len(node.modules_exited) > 0)
                and (len(node.containing_modules_origin_nested) > 0)
                and (
                        node.containing_modules_origin_nested[-1].split(":")[0]
                        in node.modules_exited
                )
        ):
            return False

        # Now check whether all of its parents are either buffers, or are outside the module.
        # If any aren't, return False.

        for parent_layer_label in node.parent_layers:
            if type(node) == TensorLogEntry:
                parent_layer = self[parent_layer_label]
            else:
                parent_layer = self.layer_dict_rolled[parent_layer_label]
            if (not parent_layer.is_buffer_layer) and (
                    (len(parent_layer.containing_modules_origin_nested) > 0)
                    and parent_layer.containing_modules_origin_nested[-1]
                    == node.containing_modules_origin_nested[-1]
            ):
                return False

        return True

    def _get_node_bg_color(
            self, node: Union[TensorLogEntry, RolledTensorLogEntry]
    ) -> str:
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

    def _make_node_label(
            self,
            node: Union[TensorLogEntry, RolledTensorLogEntry],
            node_address: str,
            vis_opt: str,
    ) -> str:
        """Gets the text for the graphviz node."""
        # Pass info:

        if (node.layer_passes_total > 1) and (vis_opt == "unrolled"):
            pass_label = f":{node.pass_num}"
        elif (node.layer_passes_total > 1) and (vis_opt == "rolled"):
            pass_label = f" (x{node.layer_passes_total})"
        else:
            pass_label = ""

        # Tensor shape info:

        if len(node.tensor_shape) > 1:
            tensor_shape_str = "x".join([str(x) for x in node.tensor_shape])
        elif len(node.tensor_shape) == 1:
            tensor_shape_str = f"x{node.tensor_shape[0]}"
        else:
            tensor_shape_str = "x1"

        # Layer param info:

        param_label = self._make_param_label(node)

        tensor_fsize = node.tensor_fsize_nice
        if node.layer_type in ["input", "output", "buffer"]:
            node_title = f"<b>{node.layer_type}_{node.layer_type_num}{pass_label}</b>"
        else:
            node_title = f"<b>{node.layer_type}_{node.layer_type_num}_{node.layer_total_num}{pass_label}</b>"

        if node.is_terminal_bool_layer:
            label_text = str(node.atomic_bool_val).upper()
            bool_label = f"<b><u>{label_text}:</u></b><br/><br/>"
        else:
            bool_label = ""

        node_label = (
            f"<{bool_label}{node_title}<br/>{tensor_shape_str} "
            f"({tensor_fsize}){param_label}{node_address}>"
        )

        return node_label

    @staticmethod
    def _make_param_label(node: Union[TensorLogEntry, RolledTensorLogEntry]) -> str:
        """Makes the label for parameters of a node."""
        if node.num_param_tensors == 0:
            return ""

        each_param_shape = []
        for param_shape in node.parent_param_shapes:
            if len(param_shape) > 1:
                each_param_shape.append("x".join([str(s) for s in param_shape]))
            elif len(param_shape) == 1:
                each_param_shape.append(f"x{param_shape[0]}")
            else:
                each_param_shape.append("x1")

        param_label = "<br/>params: " + ", ".join(
            [param_shape for param_shape in each_param_shape]
        )
        return param_label

    def _add_edges_for_node(
            self,
            parent_node: Union[TensorLogEntry, RolledTensorLogEntry],
            parent_is_collapsed_module: bool,
            vis_nesting_depth: int,
            node_color: str,
            module_edge_dict: Dict,
            edges_used: Set,
            graphviz_graph,
            vis_opt: str = "unrolled",
            show_buffer_layers: bool = False,
            vis_edge_overrides: Dict = None,
            vis_gradient_edge_overrides: Dict = None
    ):
        """Add the rolled-up edges for a node, marking for the edge which passes it happened for.

        Args:
            parent_node: The node to add edges for.
            parent_is_collapsed_module: Whether the node is a collapsed module node.
            vis_nesting_depth: How many levels of module nesting to show.
            node_color: Color of the node
            graphviz_graph: The graphviz graph object.
            module_edge_dict: Dictionary mapping each cluster to the edges it contains.
            edges_used: Edges used so far.
            vis_opt: Either 'unrolled' or 'rolled'
            show_buffer_layers: whether to show the buffer layers
        """
        for child_layer_label in parent_node.child_layers:
            if vis_opt == "unrolled":
                child_node = self.layer_dict_main_keys[child_layer_label]
            elif vis_opt == "rolled":
                child_node = self.layer_dict_rolled[child_layer_label]
            else:
                raise ValueError(
                    f"vis_opt must be 'unrolled' or 'rolled', not {vis_opt}"
                )

            if child_node.is_buffer_layer and not show_buffer_layers:
                continue

            if parent_node.has_input_ancestor:
                edge_style = "solid"
            else:
                edge_style = "dashed"

            if parent_is_collapsed_module:
                module_name_w_pass = parent_node.containing_modules_origin_nested[
                    vis_nesting_depth - 1
                    ]
                module_tuple = module_name_w_pass.split(":")
                if vis_opt == "unrolled":
                    tail_name = "pass".join(module_tuple)
                else:
                    tail_name = module_tuple[0]
            else:
                tail_name = parent_node.layer_label.replace(":", "pass")

            child_is_collapsed_module = self._check_if_collapsed_module(
                child_node, vis_nesting_depth
            )

            if child_is_collapsed_module:
                module_name_w_pass = child_node.containing_modules_origin_nested[
                    vis_nesting_depth - 1
                    ]
                module_tuple = module_name_w_pass.split(":")
                if vis_opt == "unrolled":
                    head_name = "pass".join(module_tuple)
                else:
                    head_name = module_tuple[0]
            else:
                head_name = child_node.layer_label.replace(":", "pass")

            both_nodes_collapsed_modules = (
                    parent_is_collapsed_module and child_is_collapsed_module
            )

            # If both child and parent are in a collapsed module of the same pass, skip the edge:
            if both_nodes_collapsed_modules:
                child_containing_modules = child_node.containing_modules_origin_nested[
                                           :
                                           ]
                parent_containing_modules = (
                    parent_node.containing_modules_origin_nested[:]
                )
                if child_node.is_bottom_level_submodule_output:
                    child_containing_modules = child_containing_modules[:-1]
                if parent_node.is_bottom_level_submodule_output:
                    parent_containing_modules = parent_containing_modules[:-1]
                if (
                        child_containing_modules[:vis_nesting_depth]
                        == parent_containing_modules[:vis_nesting_depth]
                ):
                    continue

            # Skip repeated edges:
            if (tail_name, head_name) in edges_used:
                continue
            edges_used.add((tail_name, head_name))

            edge_dict = {
                "tail_name": tail_name,
                "head_name": head_name,
                "color": node_color,
                "fontcolor": node_color,
                "style": edge_style,
                "arrowsize": ".7",
                "labelfontsize": "8",
            }

            # Mark with "if" in the case edge starts a cond branch
            if (child_layer_label in parent_node.cond_branch_start_children) and (not child_is_collapsed_module):
                edge_dict["label"] = '<<FONT POINT-SIZE="18"><b><u>IF</u></b></FONT>>'

            # Label the arguments to the next node if multiple inputs
            if not child_is_collapsed_module:
                self._label_node_arguments_if_needed(
                    parent_node, child_node, edge_dict, show_buffer_layers
                )

            # Annotate passes for rolled node edge if it varies across passes
            if vis_opt == "rolled":
                self._label_rolled_pass_nums(child_node, parent_node, edge_dict)

            for arg_name, arg_val in vis_edge_overrides.items():
                if callable(arg_val):
                    edge_dict[arg_name] = str(arg_val(self, parent_node, child_node))
                else:
                    edge_dict[arg_name] = str(arg_val)

            # Add it to the appropriate module cluster (most nested one containing both nodes)
            containing_module = self._get_lowest_containing_module_for_two_nodes(
                parent_node, child_node, both_nodes_collapsed_modules, vis_nesting_depth
            )
            if containing_module != -1:
                module_edge_dict[containing_module]["edges"].append(edge_dict)
                if parent_node.has_input_ancestor or child_node.has_input_ancestor:
                    module_edge_dict[containing_module]["has_input_ancestor"] = True
                    for module in parent_node.containing_modules_origin_nested:
                        module_edge_dict[module]["has_input_ancestor"] = True
                        if module == containing_module:
                            break
                    for module in child_node.containing_modules_origin_nested:
                        module_edge_dict[module]["has_input_ancestor"] = True
                        if module == containing_module:
                            break
            else:
                graphviz_graph.edge(**edge_dict)

            # Finally, add a backwards edge if both tensors have stored gradients.
            if vis_opt == "unrolled":
                self._add_gradient_edge(
                    parent_node,
                    child_node,
                    edge_style,
                    containing_module,
                    module_edge_dict,
                    graphviz_graph,
                    vis_gradient_edge_overrides
                )

    def _label_node_arguments_if_needed(
            self,
            parent_node: Union[TensorLogEntry, RolledTensorLogEntry],
            child_node: Union[TensorLogEntry, RolledTensorLogEntry],
            edge_dict: Dict,
            show_buffer_layers: bool = False,
    ):
        """Checks if a node has multiple non-commutative arguments, and if so, adds labels in edge_dict

        Args:
            parent_node: parent node
            child_node: child node
            edge_dict: dict of information about the edge
            show_buffer_layers: whether to show the buffer layers
        """
        if not self._check_whether_to_mark_arguments_on_edge(
                child_node, show_buffer_layers
        ):
            return

        arg_labels = []
        for arg_type in ["args", "kwargs"]:
            for arg_loc, arg_label in child_node.parent_layer_arg_locs[
                arg_type
            ].items():
                if (parent_node.layer_label == arg_label) or (
                        parent_node.layer_label in arg_label
                ):
                    arg_labels.append(f"{arg_type[:-1]} {str(arg_loc)}")

        arg_labels = "<br/>".join(arg_labels)
        arg_label = f"<<FONT POINT-SIZE='10'><b>{arg_labels}</b></FONT>>"
        if "label" not in edge_dict:
            edge_dict["label"] = arg_label
        else:
            edge_dict["label"] = edge_dict["label"][:-1] + "<br/>" + arg_label[1:]

    def _check_whether_to_mark_arguments_on_edge(
            self,
            child_node: Union[TensorLogEntry, RolledTensorLogEntry],
            show_buffer_layers: bool = False,
    ):
        if child_node.layer_type in self.COMMUTE_FUNCS:
            return False

        if type(child_node) == TensorLogEntry:
            return self._check_whether_to_mark_arguments_on_unrolled_edge(
                child_node, show_buffer_layers
            )
        elif type(child_node) == RolledTensorLogEntry:
            return self._check_whether_to_mark_arguments_on_rolled_edge(child_node)

    def _check_whether_to_mark_arguments_on_unrolled_edge(
            self, child_node: TensorLogEntry, show_buffer_layers: bool = False
    ):
        num_parents_shown = len(child_node.parent_layers)

        if not show_buffer_layers:
            num_parents_shown -= sum(
                [
                    int(self[parent].is_buffer_layer)
                    for parent in child_node.parent_layers
                ]
            )

        if num_parents_shown > 1:
            return True
        else:
            return False

    def _check_whether_to_mark_arguments_on_rolled_edge(
            self, child_node: RolledTensorLogEntry, show_buffer_layers: bool = False
    ):
        for pass_num, pass_parents in child_node.parent_layers_per_pass.items():
            num_parents_shown = len(pass_parents)
            if not show_buffer_layers:
                num_parents_shown -= sum(
                    [
                        int(self.layer_dict_rolled[parent].is_buffer_layer)
                        for parent in pass_parents
                    ]
                )
            if num_parents_shown > 1:
                return True

        return False

    @staticmethod
    def _label_rolled_pass_nums(
            child_node: RolledTensorLogEntry,
            parent_node: RolledTensorLogEntry,
            edge_dict: Dict,
    ):
        """Adds labels for the pass numbers to the edge dict for rolled nodes.

        Args:
            child_node: child node
            parent_node: parent node
            edge_dict: dictionary of edge information
        """
        parent_pass_nums = parent_node.child_passes_per_layer[child_node.layer_label]
        child_pass_nums = child_node.parent_passes_per_layer[parent_node.layer_label]
        if parent_node.edges_vary_across_passes:
            edge_dict[
                "taillabel"
            ] = f"  Out {int_list_to_compact_str(parent_pass_nums)}  "

        # Mark the head label with the argument if need be:
        if child_node.edges_vary_across_passes:
            edge_dict[
                "headlabel"
            ] = f"  In {int_list_to_compact_str(child_pass_nums)}  "

    @staticmethod
    def _get_lowest_containing_module_for_two_nodes(
            node1: Union[TensorLogEntry, RolledTensorLogEntry],
            node2: Union[TensorLogEntry, RolledTensorLogEntry],
            both_nodes_collapsed_modules: bool,
            vis_nesting_depth: int,
    ):
        """Utility function to get the lowest-level module that contains two nodes, to know where to put the edge.

        Args:
            node1: The first node.
            node2: The second node.
            vis_nesting_depth: How many levels deep to visualize.

        Returns:
            Lowest-level module pass containing two nodes.
        """
        node1_modules = node1.containing_modules_origin_nested[:]
        node2_modules = node2.containing_modules_origin_nested[:]

        if type(node1) == RolledTensorLogEntry:
            node1_modules = [module.split(":")[0] for module in node1_modules]
            node2_modules = [module.split(":")[0] for module in node2_modules]

        if node1.is_bottom_level_submodule_output:
            node1_nestmodules = node1_modules[:-1]
        else:
            node1_nestmodules = node1_modules[:]

        if node2.is_bottom_level_submodule_output:
            node2_nestmodules = node2_modules[:-1]
        else:
            node2_nestmodules = node2_modules[:]

        if (
                (len(node1_modules) == 0)
                or (len(node2_modules) == 0)
                or (node1_modules[0] != node2_modules[0])
        ):
            return -1  # no submodule contains them both.

        if node1 == node2:
            if node1.is_bottom_level_submodule_output and (len(node1_modules) == 1):
                return -1
            elif node1.is_bottom_level_submodule_output and (len(node1_modules) > 1):
                containing_module = node1_modules[-2]
            else:
                containing_module = node1_modules[-1]
            return containing_module

        if both_nodes_collapsed_modules:
            if (vis_nesting_depth == 1) or (len(node1_nestmodules) == 1):
                return -1
            if node1_modules[vis_nesting_depth - 1] == node2_modules[vis_nesting_depth - 1]:
                containing_module = node1_modules[vis_nesting_depth - 2]
                return containing_module

        containing_module = node1_modules[0]
        for m in range(min([len(node1_modules), len(node2_modules)])):
            if node1_modules[m] != node2_modules[m]:
                break
            containing_module = node1_modules[m]

        return containing_module

    def _add_gradient_edge(
            self,
            parent_layer,
            child_layer,
            edge_style,
            containing_module,
            module_edge_dict,
            graphviz_graph,
            vis_gradient_edge_overrides
    ):
        """Adds a backwards edge if both layers have saved gradients, showing the backward pass."""
        if parent_layer.has_saved_grad and child_layer.has_saved_grad:
            edge_dict = {
                "tail_name": child_layer.layer_label.replace(":", "pass"),
                "head_name": parent_layer.layer_label.replace(":", "pass"),
                "color": self.GRADIENT_ARROW_COLOR,
                "fontcolor": self.GRADIENT_ARROW_COLOR,
                "style": edge_style,
                "arrowsize": ".7",
                "labelfontsize": "8",
            }
            for arg_name, arg_val in vis_gradient_edge_overrides.items():
                if callable(arg_val):
                    edge_dict[arg_name] = str(arg_val(self, parent_layer, child_layer))
                else:
                    edge_dict[arg_name] = str(arg_val)

            if containing_module != -1:
                module_edge_dict[containing_module]["edges"].append(edge_dict)
            else:
                graphviz_graph.edge(**edge_dict)

    def _set_up_subgraphs(
            self, graphviz_graph, vis_opt: str, module_edge_dict: Dict[str, List], vis_module_overrides: Dict = None
    ):
        """Given a dictionary specifying the edges in each cluster and the graphviz graph object,
        set up the nested subgraphs and the nodes that should go inside each of them. There will be some tricky
        recursive logic to set up the nested context managers.

        Args:
            graphviz_graph: Graphviz graph object.
            vis_opt: 'rolled' or 'unrolled'
            module_edge_dict: Dictionary mapping each cluster to the list of edges it contains, with each
                edge specified as a dict with all necessary arguments for creating that edge.
        """
        if vis_opt == "unrolled":
            module_submodule_dict = self.module_pass_children.copy()
            subgraphs = self.top_level_module_passes[:]
        else:
            module_submodule_dict = self.module_children.copy()
            subgraphs = self.top_level_modules[:]

        # Get the max module nesting depth:

        max_nesting_depth = self._get_max_nesting_depth(
            subgraphs, module_edge_dict, module_submodule_dict
        )

        subgraph_stack = [[subgraph] for subgraph in subgraphs]
        nesting_depth = 0
        while len(subgraph_stack) > 0:
            parent_graph_list = subgraph_stack.pop(0)
            self._setup_subgraphs_recurse(
                graphviz_graph,
                parent_graph_list,
                module_edge_dict,
                module_submodule_dict,
                subgraph_stack,
                nesting_depth,
                max_nesting_depth,
                vis_opt,
                vis_module_overrides
            )

    def _setup_subgraphs_recurse(
            self,
            starting_subgraph,
            parent_graph_list: List,
            module_edge_dict,
            module_submodule_dict,
            subgraph_stack,
            nesting_depth,
            max_nesting_depth,
            vis_opt,
            vis_module_overrides
    ):
        """Utility function to crawl down several layers deep into nested subgraphs.

        Args:
            starting_subgraph: The subgraph we're starting from.
            parent_graph_list: List of parent graphs.
            module_edge_dict: Dict mapping each cluster to its edges.
            module_submodule_dict: Dict mapping each cluster to its subclusters.
            subgraph_stack: Stack of subgraphs to look at.
            nesting_depth: Nesting depth so far.
            max_nesting_depth: The total depth of the subgraphs.
            vis_opt: 'rolled' or 'unrolled'
        """
        subgraph_name_w_pass = parent_graph_list[nesting_depth]
        subgraph_module = subgraph_name_w_pass.split(":")[0]
        if vis_opt == "unrolled":
            cluster_name = f"cluster_{subgraph_name_w_pass.replace(':', '_pass')}"
            subgraph_name = subgraph_name_w_pass
        elif vis_opt == "rolled":
            cluster_name = f"cluster_{subgraph_module}"
            subgraph_name = subgraph_module
        else:
            raise ValueError("vis_opt must be 'rolled' or 'unrolled'")
        module_type = self.module_types[subgraph_module]
        if (self.module_num_passes[subgraph_module] > 1) and (vis_opt == "unrolled"):
            subgraph_title = subgraph_name_w_pass
        elif (self.module_num_passes[subgraph_module] > 1) and (vis_opt == "rolled"):
            subgraph_title = (
                f"{subgraph_module} (x{self.module_num_passes[subgraph_module]})"
            )
        else:
            subgraph_title = subgraph_module

        if (
                nesting_depth < len(parent_graph_list) - 1
        ):  # we haven't gotten to the bottom yet, keep going.
            with starting_subgraph.subgraph(name=cluster_name) as s:
                self._setup_subgraphs_recurse(
                    s,
                    parent_graph_list,
                    module_edge_dict,
                    module_submodule_dict,
                    subgraph_stack,
                    nesting_depth + 1,
                    max_nesting_depth,
                    vis_opt,
                    vis_module_overrides
                )

        else:  # we made it, make the subgraph and add all edges.
            with starting_subgraph.subgraph(name=cluster_name) as s:
                nesting_fraction = (
                                           max_nesting_depth - nesting_depth
                                   ) / max_nesting_depth
                pen_width = (
                        self.MIN_MODULE_PENWIDTH + nesting_fraction * self.PENWIDTH_RANGE
                )
                if module_edge_dict[subgraph_name]["has_input_ancestor"]:
                    line_style = "solid"
                else:
                    line_style = "dashed"

                module_args = {
                    'label': f"<<B>@{subgraph_title}</B><br align='left'/>({module_type})<br align='left'/>>",
                    'labelloc': 'b',
                    'style': f'filled,{line_style}',
                    'fillcolor': 'white',
                    'penwidth': str(pen_width)}

                for arg_name, arg_val in vis_module_overrides.items():
                    if callable(arg_val):
                        module_args[arg_name] = str(arg_val(self, subgraph_name))
                    else:
                        module_args[arg_name] = str(arg_val)
                s.attr(**module_args)
                subgraph_edges = module_edge_dict[subgraph_name]["edges"]
                for edge_dict in subgraph_edges:
                    s.edge(**edge_dict)
                subgraph_children = module_submodule_dict[subgraph_name_w_pass]
                for (
                        subgraph_child
                ) in subgraph_children:  # it's weird but have to go in reverse order.
                    subgraph_stack.append(parent_graph_list[:] + [subgraph_child])

    @staticmethod
    def _get_max_nesting_depth(top_modules, module_edge_dict, module_submodule_dict):
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
            module_edges = module_edge_dict[module]["edges"]
            module_submodules = module_submodule_dict[module]

            if (len(module_edges) == 0) and (
                    len(module_submodules) == 0
            ):  # can ignore if no edges and no children.
                continue
            elif (len(module_edges) > 0) and (len(module_submodules) == 0):
                max_nesting_depth = max([module_depth, max_nesting_depth])
            elif (len(module_edges) == 0) and (len(module_submodules) > 0):
                module_stack.extend(
                    [
                        (module_child, module_depth + 1)
                        for module_child in module_submodules
                    ]
                )
            else:
                max_nesting_depth = max([module_depth, max_nesting_depth])
                module_stack.extend(
                    [
                        (module_child, module_depth + 1)
                        for module_child in module_submodules
                    ]
                )
        return max_nesting_depth

    # ********************************************
    # *************** Validation *****************
    # ********************************************

    def validate_saved_activations(
            self, ground_truth_output_tensors: List[torch.Tensor], verbose: bool = False
    ) -> bool:
        """Starting from outputs and internally terminated tensors, checks whether computing their values from the saved
        values of their input tensors yields their actually saved values, and whether computing their values from
        their parent tensors yields their saved values.

        Returns:
            True if it passes the tests, False otherwise.
        """
        # First check that the ground truth output tensors are accurate:
        for i, output_layer_label in enumerate(self.output_layers):
            output_layer = self[output_layer_label]
            if not tensor_nanequal(
                    output_layer.tensor_contents,
                    ground_truth_output_tensors[i],
                    allow_tolerance=False,
            ):
                print(
                    f"The {i}th output layer, {output_layer_label}, does not match the ground truth output tensor."
                )
                return False

        # Validate the parents of each validated layer.
        validated_child_edges_for_each_layer = defaultdict(set)
        validated_layers = set(self.output_layers + self.internally_terminated_layers)
        layers_to_validate_parents_for = list(validated_layers)

        while len(layers_to_validate_parents_for) > 0:
            layer_to_validate_parents_for = layers_to_validate_parents_for.pop(0)
            parent_layers_valid = self.validate_parents_of_saved_layer(
                layer_to_validate_parents_for,
                validated_layers,
                validated_child_edges_for_each_layer,
                layers_to_validate_parents_for,
                verbose,
            )
            if not parent_layers_valid:
                return False

        if len(validated_layers) < len(self.layer_labels):
            print(
                f"All saved activations were accurate, but some layers were not reached (check that "
                f"child args logged accurately): {set(self.layer_labels) - validated_layers}"
            )
            return False

        return True

    def validate_parents_of_saved_layer(
            self,
            layer_to_validate_parents_for_label: str,
            validated_layers: Set[str],
            validated_child_edges_for_each_layer: Dict[str, Set[str]],
            layers_to_validate_parents_for: List[str],
            verbose: bool = False,
    ) -> bool:
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
        if not self._check_layer_arguments_logged_correctly(
                layer_to_validate_parents_for_label
        ):
            print(
                f"Parent arguments for layer {layer_to_validate_parents_for_label} are not logged properly; "
                f"either a parent wasn't logged as an argument, or was logged an extra time"
            )
            return False

        # Check that executing the function based on the actual saved values of the parents yields the saved
        # values of the layer itself:

        if not self._check_whether_func_on_saved_parents_yields_saved_tensor(
                layer_to_validate_parents_for_label, perturb=False
        ):
            return False

        # Check that executing the layer's function on the wrong version of the saved parent tensors
        # yields the wrong tensors, when each saved tensor is perturbed in turn:

        for perturb_layer in layer_to_validate_parents_for.parent_layers:
            if (
                    layer_to_validate_parents_for.func_applied_name
                    in self.FUNCS_NOT_TO_PERTURB_IN_VALIDATION
            ):
                continue
            if not self._check_whether_func_on_saved_parents_yields_saved_tensor(
                    layer_to_validate_parents_for_label,
                    perturb=True,
                    layers_to_perturb=[perturb_layer],
                    verbose=verbose,
            ):
                return False

        # Log that each parent layer has been validated for this source layer.

        for parent_layer_label in layer_to_validate_parents_for.parent_layers:
            parent_layer = self[parent_layer_label]
            validated_child_edges_for_each_layer[parent_layer_label].add(
                layer_to_validate_parents_for_label
            )
            if validated_child_edges_for_each_layer[parent_layer_label] == set(
                    parent_layer.child_layers
            ):
                validated_layers.add(parent_layer_label)
                if ((not parent_layer.is_input_layer) and
                        not (parent_layer.is_buffer_layer and (parent_layer.buffer_parent is None))):
                    layers_to_validate_parents_for.append(parent_layer_label)

        return True

    def _check_layer_arguments_logged_correctly(self, target_layer_label: str) -> bool:
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
        for arg_type in ["args", "kwargs"]:
            parent_layers_in_args.update(
                list(target_layer.parent_layer_arg_locs[arg_type].values())
            )
        if parent_layers_in_args != set(target_layer.parent_layers):
            return False

        argtype_dict = {
            "args": (enumerate, "creation_args"),
            "kwargs": (lambda x: x.items(), "creation_kwargs"),
        }

        # Check for each parent layer that it is logged as a saved argument when it matches an argument, and
        # is not logged when it does not match a saved argument.

        for parent_layer_label in target_layer.parent_layers:
            parent_layer = self[parent_layer_label]
            for arg_type in ["args", "kwargs"]:
                iterfunc, argtype_field = argtype_dict[arg_type]
                for key, val in iterfunc(getattr(target_layer, argtype_field)):
                    validation_correct_for_arg_and_layer = (
                        self._validate_layer_against_arg(
                            target_layer, parent_layer, arg_type, key, val
                        )
                    )
                    if not validation_correct_for_arg_and_layer:
                        return False
        return True

    def _validate_layer_against_arg(
            self, target_layer, parent_layer, arg_type, key, val
    ):
        if type(val) in [list, tuple]:
            for v, subval in enumerate(val):
                argloc_key = (key, v)
                validation_correct_for_arg_and_layer = (
                    self._check_arglocs_correct_for_arg(
                        target_layer, parent_layer, arg_type, argloc_key, subval
                    )
                )
                if not validation_correct_for_arg_and_layer:
                    return False

        elif type(val) == dict:
            for subkey, subval in val.items():
                argloc_key = (key, subkey)
                validation_correct_for_arg_and_layer = (
                    self._check_arglocs_correct_for_arg(
                        target_layer, parent_layer, arg_type, argloc_key, subval
                    )
                )
                if not validation_correct_for_arg_and_layer:
                    return False
        else:
            argloc_key = key
            validation_correct_for_arg_and_layer = self._check_arglocs_correct_for_arg(
                target_layer, parent_layer, arg_type, argloc_key, val
            )
            if not validation_correct_for_arg_and_layer:
                return False

        return True

    def _check_arglocs_correct_for_arg(
            self,
            target_layer: TensorLogEntry,
            parent_layer: TensorLogEntry,
            arg_type: str,
            argloc_key: Union[str, tuple],
            saved_arg_val: Any,
    ):
        """For a given layer and an argument to its child layer, checks that it is logged correctly:
        that is, that it's logged as an argument if it matches, and is not logged as an argument if it doesn't match.
        """
        target_layer_label = target_layer.layer_label
        parent_layer_label = parent_layer.layer_label
        if target_layer_label in parent_layer.children_tensor_versions:
            parent_activations = parent_layer.children_tensor_versions[
                target_layer_label
            ]
        else:
            parent_activations = parent_layer.tensor_contents

        if type(saved_arg_val) == torch.Tensor:
            parent_layer_matches_arg = tensor_nanequal(
                saved_arg_val, parent_activations, allow_tolerance=False
            )
        else:
            parent_layer_matches_arg = False
        parent_layer_logged_as_arg = (
                                             argloc_key in target_layer.parent_layer_arg_locs[arg_type]
                                     ) and (
                                             target_layer.parent_layer_arg_locs[arg_type][argloc_key]
                                             == parent_layer_label
                                     )

        if (
                parent_layer_matches_arg
                and (not parent_layer_logged_as_arg)
                and (parent_activations.numel() != 0)
                and (parent_activations.dtype != torch.bool)
                and (not tensor_all_nan(parent_activations))
                and (parent_activations.abs().float().mean() != 0)
                and (parent_activations.abs().float().mean() != 1)
                and not any(
            [
                torch.equal(parent_activations, self[other_parent].tensor_contents)
                for other_parent in target_layer.parent_layers
                if other_parent != parent_layer_label
            ]
        )
        ):
            print(
                f"Parent {parent_layer_label} of {target_layer_label} has activations that match "
                f"{arg_type} {argloc_key} for {target_layer_label}, but is not logged as "
                f"such in parent_layer_arg_locs."
            )
            return False

        if (not parent_layer_matches_arg) and parent_layer_logged_as_arg:
            print(
                f"Parent {parent_layer_label} of {target_layer_label} is logged as {arg_type} {argloc_key} to "
                f"{target_layer_label}, but its saved activations don't match the saved argument."
            )
            return False

        return True

    def _check_whether_func_on_saved_parents_yields_saved_tensor(
            self,
            layer_to_validate_parents_for_label: str,
            perturb: bool = False,
            layers_to_perturb: List[str] = None,
            verbose: bool = False,
    ) -> bool:
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

        if (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "__getitem__")
                and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1],
        )
        ):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "__getitem__")
                and not torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[0],
        )
        ):
            return True
        elif layer_to_validate_parents_for.func_applied_name == 'empty_like':
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "__setitem__")
                and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
                and (layer_to_validate_parents_for.creation_args[1].dtype == torch.bool)
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1],
        )
        ):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "cross_entropy")
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1],
        )
        ):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "__setitem__")
                and (type(layer_to_validate_parents_for.creation_args[1]) == tuple)
                and (
                        type(layer_to_validate_parents_for.creation_args[1][0]) == torch.Tensor
                )
                and (layer_to_validate_parents_for.creation_args[1][0].dtype == torch.bool)
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1][0],
        )
        ):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "index_select")
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[2],
        )
        ):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "lstm")
                and (torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1][0]) or
                     torch.equal(
                         self[layers_to_perturb[0]].tensor_contents,
                         layer_to_validate_parents_for.creation_args[1][1]) or
                     torch.equal(
                         self[layers_to_perturb[0]].tensor_contents,
                         layer_to_validate_parents_for.creation_args[2][0]) or
                     torch.equal(
                         self[layers_to_perturb[0]].tensor_contents,
                         layer_to_validate_parents_for.creation_args[2][1]) or
                     ((type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor) and
                      torch.equal(
                          self[layers_to_perturb[0]].tensor_contents,
                          layer_to_validate_parents_for.creation_args[1])
                     ))):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "_pad_packed_sequence")
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1]
        )):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "masked_fill_")
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[1]
        )):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "scatter_")
                and torch.equal(
            self[layers_to_perturb[0]].tensor_contents,
            layer_to_validate_parents_for.creation_args[2]
        )):
            return True
        elif (
                perturb
                and (layer_to_validate_parents_for.func_applied_name == "interpolate")
                and ((('scale_factor' in layer_to_validate_parents_for.creation_kwargs)
                      and torch.equal(
                    self[layers_to_perturb[0]].tensor_contents,
                    torch.tensor(layer_to_validate_parents_for.creation_kwargs['scale_factor'])))
                     or ((len(layer_to_validate_parents_for.creation_args) >= 3)
                         and torch.equal(
                            self[layers_to_perturb[0]].tensor_contents,
                            layer_to_validate_parents_for.creation_args[2])))
        ):
            return True

        # Prepare input arguments: keep the ones that should just be kept, perturb those that should be perturbed

        input_args = self._prepare_input_args_for_validating_layer(
            layer_to_validate_parents_for, layers_to_perturb
        )

        # set the saved rng value:
        layer_func = layer_to_validate_parents_for.func_applied
        current_rng_states = log_current_rng_states()
        set_rng_from_saved_states(layer_to_validate_parents_for.func_rng_states)
        try:
            recomputed_output = layer_func(*input_args["args"], **input_args["kwargs"])
        except:
            raise Exception(f"Invalid perturbed arguments for layer {layer_to_validate_parents_for_label}")
        set_rng_from_saved_states(current_rng_states)

        if layer_func.__name__ in [
            "__setitem__",
            "zero_",
            "__delitem__",
        ]:  # TODO: fix this
            recomputed_output = input_args["args"][0]

        if any([issubclass(type(recomputed_output), which_type) for which_type in [list, tuple]]):
            recomputed_output = recomputed_output[
                layer_to_validate_parents_for.iterable_output_index
            ]

        if (
                not (
                        tensor_nanequal(
                            recomputed_output,
                            layer_to_validate_parents_for.tensor_contents,
                            allow_tolerance=True,
                        )
                )
                and not perturb
        ):
            print(
                f"Saved activations for layer {layer_to_validate_parents_for_label} do not match the "
                f"values computed based on the parent layers {layer_to_validate_parents_for.parent_layers}."
            )
            return False

        if (
                tensor_nanequal(
                    recomputed_output,
                    layer_to_validate_parents_for.tensor_contents,
                    allow_tolerance=False,
                )
                and perturb
        ):
            return self._posthoc_perturb_check(
                layer_to_validate_parents_for, layers_to_perturb, verbose
            )

        return True

    def _prepare_input_args_for_validating_layer(
            self,
            layer_to_validate_parents_for: TensorLogEntry,
            layers_to_perturb: List[str],
    ) -> Dict:
        """Prepares the input arguments for validating the saved activations of a layer.

        Args:
            layer_to_validate_parents_for: Layer being checked.
            layers_to_perturb: Layers for which to perturb the saved activations.

        Returns:
            Dict of input arguments.
        """
        input_args = {
            "args": list(layer_to_validate_parents_for.creation_args[:]),
            "kwargs": layer_to_validate_parents_for.creation_kwargs.copy(),
        }
        input_args = self._copy_validation_args(input_args)

        # Swap in saved parent activations:

        for arg_type in ["args", "kwargs"]:
            for (
                    key,
                    parent_layer_arg,
            ) in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type].items():
                parent_layer = self[parent_layer_arg]
                if (
                        layer_to_validate_parents_for.layer_label
                        in parent_layer.children_tensor_versions
                ):
                    parent_values = parent_layer.children_tensor_versions[
                        layer_to_validate_parents_for.layer_label
                    ]
                else:
                    parent_values = parent_layer.tensor_contents
                parent_values = parent_values.detach().clone()

                if parent_layer_arg in layers_to_perturb:
                    parent_layer_func_values = self._perturb_layer_activations(
                        parent_values, layer_to_validate_parents_for.tensor_contents
                    )
                else:
                    parent_layer_func_values = parent_values

                if type(key) != tuple:
                    input_args[arg_type][key] = parent_layer_func_values
                else:
                    input_args[arg_type][key[0]] = tuple_tolerant_assign(
                        input_args[arg_type][key[0]], key[1], parent_layer_func_values
                    )

        return input_args

    @staticmethod
    def _copy_validation_args(input_args: Dict):
        new_args = []
        for i, val in enumerate(input_args["args"]):
            if type(val) == torch.Tensor:
                new_args.append(val.detach().clone())
            elif type(val) in [list, tuple, set]:
                new_iter = []
                for i2, val2 in enumerate(val):
                    if type(val2) == torch.Tensor:
                        new_iter.append(val2.detach().clone())
                    else:
                        new_iter.append(val2)
                new_args.append(type(val)(new_iter))
            else:
                new_args.append(val)
        input_args["args"] = new_args

        new_kwargs = {}
        for key, val in input_args["kwargs"].items():
            if type(val) == torch.Tensor:
                new_kwargs[key] = val.detach().clone()
            elif type(val) in [list, tuple, set]:
                new_iter = []
                for i2, val2 in enumerate(val):
                    if type(val2) == torch.Tensor:
                        new_iter.append(val2.detach().clone())
                    else:
                        new_iter.append(val2)
                new_kwargs[key] = type(val)(new_iter)
            else:
                new_kwargs[key] = val
        input_args["kwargs"] = new_kwargs
        return input_args

    @staticmethod
    def _perturb_layer_activations(
            parent_activations: torch.Tensor, output_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Perturbs the values of a saved tensor.

        Args:
            parent_activations: Tensor of activation values for the parent tensor
            output_activations: Tensor of activation values for the tensor whose parents are being tested (the output)

        Returns:
            Perturbed version of saved tensor
        """
        device = parent_activations.device
        if parent_activations.numel() == 0:
            return parent_activations.detach().clone()

        if parent_activations.dtype in [
            torch.int,
            torch.long,
            torch.short,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]:
            tensor_unique_vals = torch.unique(parent_activations)
            if len(tensor_unique_vals) > 1:
                perturbed_activations = parent_activations.detach().clone()
                while torch.equal(perturbed_activations, parent_activations):
                    perturbed_activations = torch.randint(
                        parent_activations.min(),
                        parent_activations.max() + 1,
                        size=parent_activations.shape,
                        device=device,
                    ).type(parent_activations.dtype)
            else:
                perturbed_activations = parent_activations.detach().clone()
                while torch.equal(perturbed_activations, parent_activations):
                    if torch.min(parent_activations) < 0:
                        perturbed_activations = torch.randint(
                            -10, 11, size=parent_activations.shape, device=device
                        ).type(parent_activations.dtype)
                    else:
                        perturbed_activations = torch.randint(
                            0, 11, size=parent_activations.shape, device=device
                        ).type(parent_activations.dtype)

        elif parent_activations.dtype == torch.bool:
            perturbed_activations = parent_activations.detach().clone()
            while torch.equal(perturbed_activations, parent_activations):
                perturbed_activations = torch.randint(
                    0, 2, size=parent_activations.shape, device=device
                ).bool()
        else:
            mean_output_sqrt = output_activations.detach().float().abs().mean()
            mean_output_sqrt += torch.rand(mean_output_sqrt.shape) * 100
            mean_output_sqrt *= torch.rand(mean_output_sqrt.shape)
            mean_output_sqrt.requires_grad = False
            perturbed_activations = torch.randn_like(
                parent_activations.float(), device=device
            ) * mean_output_sqrt.to(device)
            perturbed_activations = perturbed_activations.type(parent_activations.dtype)

        return perturbed_activations

    def _posthoc_perturb_check(
            self,
            layer_to_validate_parents_for: TensorLogEntry,
            layers_to_perturb: List[str],
            verbose: bool = False,
    ) -> bool:
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
        # Check if the tensor is all nans or all infinite:
        if layer_to_validate_parents_for.tensor_dtype == torch.bool:
            return True
        elif (
                (layer_to_validate_parents_for.func_applied_name == "to")
                and (len(layer_to_validate_parents_for.creation_args) > 1)
                and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
        ):
            return True
        elif (
                (layer_to_validate_parents_for.func_applied_name == "__setitem__")
                and (type(layer_to_validate_parents_for.creation_args[2]) == torch.Tensor)
                and (
                        layer_to_validate_parents_for.creation_args[0].shape
                        == layer_to_validate_parents_for.creation_args[2].shape
                )
        ):
            return True
        elif (
                layer_to_validate_parents_for.func_applied_name in ["__getitem__", "unbind"]
        ) and (
                layer_to_validate_parents_for.tensor_contents.numel() < 20
        ):  # some elements can be the same by chance
            return True
        elif (
                (layer_to_validate_parents_for.func_applied_name == "__getitem__")
                and (type(layer_to_validate_parents_for.creation_args[1]) == torch.Tensor)
                and (layer_to_validate_parents_for.creation_args[1].unique() < 20)
        ):
            return True
        elif (layer_to_validate_parents_for.func_applied_name == "max") and len(
                layer_to_validate_parents_for.creation_args
        ) > 1:
            return True
        elif (
                layer_to_validate_parents_for.func_applied_name == "max"
        ) and not torch.is_floating_point(
            layer_to_validate_parents_for.creation_args[0]
        ):
            return True
        else:
            num_inf = (
                torch.isinf(layer_to_validate_parents_for.tensor_contents.abs())
                .int()
                .sum()
            )
            num_nan = (
                torch.isnan(layer_to_validate_parents_for.tensor_contents.abs())
                .int()
                .sum()
            )
            if (num_inf == layer_to_validate_parents_for.tensor_contents.numel()) or (
                    num_nan == layer_to_validate_parents_for.tensor_contents.numel()
            ):
                return True

        arg_type_dict = {
            "args": (enumerate, "creation_args"),
            "kwargs": (lambda x: x.items(), "creation_kwargs"),
        }

        layer_to_validate_parents_for_label = layer_to_validate_parents_for.layer_label
        for arg_type in ["args", "kwargs"]:
            iterfunc, fieldname = arg_type_dict[arg_type]
            for key, val in iterfunc(getattr(layer_to_validate_parents_for, fieldname)):
                # Skip if it's the argument itself:
                if (
                        key in layer_to_validate_parents_for.parent_layer_arg_locs[arg_type]
                ) and (
                        layer_to_validate_parents_for.parent_layer_arg_locs[arg_type][key]
                ) in layers_to_perturb:
                    continue
                arg_is_special = self._check_if_arg_is_special_val(val)
                if arg_is_special:
                    if verbose:
                        print(
                            f"Activations for layer {layer_to_validate_parents_for_label} do not change when "
                            f"values for {layers_to_perturb} are changed (out of parent "
                            f"layers {layer_to_validate_parents_for.parent_layers}), but {arg_type[:-1]} {key} is "
                            f"all zeros or all-ones, so validation still succeeds..."
                        )
                    return True

        print(
            f"Activations for layer {layer_to_validate_parents_for_label} do not change when "
            f"values for {layers_to_perturb} are changed (out of parent "
            f"layers {layer_to_validate_parents_for.parent_layers}), and the other "
            f'arguments are not "special" (all-ones or all-zeros) tensors.'
        )
        return False

    @staticmethod
    def _check_if_arg_is_special_val(val: Union[torch.Tensor, Any]):
        # If it's one of the other arguments, check if it's all zeros or all ones:
        if type(val) != torch.Tensor:
            try:
                val = torch.Tensor(val)
            except:
                return True
        if torch.all(val == 0) or torch.all(val == 1) or (val.numel() == 0):
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

        keys_with_substr = [
            key for key in self.layer_dict_all_keys if str(ix) in str(key)
        ]
        if len(keys_with_substr) == 1:
            return self.layer_dict_all_keys[keys_with_substr[0]]

        self._give_user_feedback_about_lookup_key(ix, "get_one_item")

    def _give_user_feedback_about_lookup_key(self, key: Union[int, str], mode: str):
        """For __getitem__ and get_op_nums_from_user_labels, gives the user feedback about the user key
        they entered if it doesn't yield any matches.

        Args:
            key: Lookup key used by the user.
        """
        if (type(key) == int) and (
                key >= len(self.layer_list) or key < -len(self.layer_list)
        ):
            raise ValueError(
                f"You specified the layer with index {key}, but there are only {len(self.layer_list)} "
                f"layers; please specify an index in the range "
                f"-{len(self.layer_list)} - {len(self.layer_list) - 1}."
            )

        if key in self.module_addresses:
            module_num_passes = self.module_num_passes[key]
            raise ValueError(
                f"You specified output of module {key}, but it has {module_num_passes} passes; "
                f"please specify e.g. {key}:2 for the second pass of {key}."
            )

        if key.split(":")[0] in self.module_addresses:
            module, pass_num = key.split(":")
            module_num_passes = self.module_num_passes[module]
            raise ValueError(
                f"You specified module {module} pass {pass_num}, but {module} only has "
                f"{module_num_passes} passes; specify a lower number."
            )

        if key in self.layer_labels_no_pass:
            layer_num_passes = self.layer_num_passes[key]
            raise ValueError(
                f"You specified output of layer {key}, but it has {layer_num_passes} passes; "
                f"please specify e.g. {key}:2 for the second pass of {key}."
            )

        if key.split(":")[0] in self.layer_labels_no_pass:
            layer_label, pass_num = key.split(":")
            layer_num_passes = self.layer_num_passes[layer_label]
            raise ValueError(
                f"You specified layer {layer_label} pass {pass_num}, but {layer_label} only has "
                f"{layer_num_passes} passes. Specify a lower number."
            )

        raise ValueError(self._get_lookup_help_str(key, mode))

    def __iter__(self):
        """Loops through all tensors in the log."""
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
        s += (
            f"\n\tTime elapsed: {np.round(self.elapsed_time_total, 3)}s "
            f"({np.round(self.elapsed_time_torchlens_logging, 3)}s spent logging)"
        )

        # Overall model structure

        s += "\n\tStructure:"
        if self.model_is_recurrent:
            s += f"\n\t\t- recurrent (at most {self.model_max_recurrent_loops} loops)"
        else:
            s += "\n\t\t- purely feedforward, no recurrence"

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
        s += (
            f"\n\t\t- {self.num_tensors_total} total tensors ({self.tensor_fsize_total_nice}) "
            f"computed in forward pass."
        )
        s += f"\n\t\t- {self.num_tensors_saved} tensors ({self.tensor_fsize_saved_nice}) with saved activations."

        # Model parameters:

        s += (
            f"\n\tParameters: {self.total_param_layers} parameter operations ({self.total_params} params total; "
            f"{self.total_params_fsize_nice})"
        )

        # Print the module hierarchy.
        s += "\n\tModule Hierarchy:"
        s += self._module_hierarchy_str()

        # Now print all layers.
        s += "\n\tLayers"
        if self.all_layers_saved:
            s += " (all have saved activations):"
        elif self.num_tensors_saved == 0:
            s += " (no layer activations are saved):"
        else:
            s += " (* means layer has saved activations):"
        for layer_ind, layer_barcode in enumerate(self.layer_labels):
            pass_num = self.layer_dict_main_keys[layer_barcode].pass_num
            total_passes = self.layer_dict_main_keys[layer_barcode].layer_passes_total
            if total_passes > 1:
                pass_str = f" ({pass_num}/{total_passes} passes)"
            else:
                pass_str = ""

            if self.layer_dict_main_keys[layer_barcode].has_saved_activations and (
                    not self.all_layers_saved
            ):
                s += "\n\t\t* "
            else:
                s += "\n\t\t  "
            s += f"({layer_ind}) {layer_barcode} {pass_str}"

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
        s += "\n\tRaw layer labels:"
        for layer in self.raw_tensor_labels_list:
            s += f"\n\t\t{layer}"
        return s

    @staticmethod
    def pretty_print_list_w_line_breaks(lst, indent_chars: str, line_break_every=5):
        """
        Utility function to pretty print a list with line breaks, adding indent_chars every line.
        """
        s = f"\n{indent_chars}"
        for i, item in enumerate(lst):
            s += f"{item}"
            if i < len(lst) - 1:
                s += ", "
            if ((i + 1) % line_break_every == 0) and (i < len(lst) - 1):
                s += f"\n{indent_chars}"
        return s

    def _get_lookup_help_str(self, layer_label: Union[int, str], mode: str) -> str:
        """Generates a help string to be used in error messages when indexing fails."""
        sample_layer1 = random.choice(self.layer_labels_w_pass)
        sample_layer2 = random.choice(self.layer_labels_no_pass)
        if len(self.module_addresses) > 0:
            sample_module1 = random.choice(self.module_addresses)
            sample_module2 = random.choice(self.module_passes)
        else:
            sample_module1 = "features.3"
            sample_module2 = "features.4:2"
        module_str = f"(e.g., {sample_module1}, {sample_module2})"
        if mode == "get_one_item":
            msg = (
                "e.g., 'pool' will grab the maxpool2d or avgpool2d layer, 'maxpool' will grab the 'maxpool2d' "
                "layer, etc., but there must be only one such matching layer"
            )
        elif mode == "query_multiple":
            msg = (
                "e.g., 'pool' will grab all maxpool2d or avgpool2d layers, 'maxpool' will grab all 'maxpool2d' "
                "layers, etc."
            )
        else:
            raise ValueError("mode must be either get_one_item or query_multiple")
        help_str = (
            f"Layer {layer_label} not recognized; please specify either "
            f"\n\n\t1) an integer giving the ordinal position of the layer "
            f"(e.g. 2 for 3rd layer, -4 for fourth-to-last), "
            f"\n\t2) the layer label (e.g., {sample_layer1}, {sample_layer2}), "
            f"\n\t3) the module address {module_str}"
            f"\n\t4) A substring of any desired layer label ({msg})."
            f"\n\n(Label meaning: conv2d_3_4:2 means the second pass of the third convolutional layer, "
            f"and fourth layer overall in the model.)"
        )
        return help_str

    def _module_hierarchy_str(self):
        """
        Utility function to print the nested module hierarchy.
        """
        s = ""
        for module_pass in self.top_level_module_passes:
            module, pass_num = module_pass.split(":")
            s += f"\n\t\t{module}"
            if self.module_num_passes[module] > 1:
                s += f":{pass_num}"
            s += self._module_hierarchy_str_helper(module_pass, 1)
        return s

    def _module_hierarchy_str_helper(self, module_pass, level):
        """
        Helper function for _module_hierarchy_str.
        """
        s = ""
        any_grandchild_modules = any(
            [
                len(self.module_pass_children[submodule_pass]) > 0
                for submodule_pass in self.module_pass_children[module_pass]
            ]
        )
        if any_grandchild_modules or len(self.module_pass_children[module_pass]) == 0:
            for submodule_pass in self.module_pass_children[module_pass]:
                submodule, pass_num = submodule_pass.split(":")
                s += f"\n\t\t{'    ' * level}{submodule}"
                if self.module_num_passes[submodule] > 1:
                    s += f":{pass_num}"
                s += self._module_hierarchy_str_helper(submodule_pass, level + 1)
        else:
            submodule_list = []
            for submodule_pass in self.module_pass_children[module_pass]:
                submodule, pass_num = submodule_pass.split(":")
                if self.module_num_passes[submodule] == 1:
                    submodule_list.append(submodule)
                else:
                    submodule_list.append(submodule_pass)
            s += self.pretty_print_list_w_line_breaks(
                submodule_list, line_break_every=8, indent_chars=f"\t\t{'    ' * level}"
            )
        return s

    def __repr__(self):
        return self.__str__()


def run_model_and_save_specified_activations(
        model: nn.Module,
        input_args: Union[torch.Tensor, List[Any]],
        input_kwargs: Dict[Any, Any],
        layers_to_save: Optional[Union[str, List[Union[int, str]]]] = "all",
        keep_unsaved_layers: bool = True,
        output_device: str = "same",
        activation_postfunc: Optional[Callable] = None,
        mark_input_output_distances: bool = False,
        detach_saved_tensors: bool = False,
        save_function_args: bool = False,
        save_gradients: bool = False,
        random_seed: Optional[int] = None,
) -> ModelHistory:
    """Internal function that runs the given input through the given model, and saves the
    specified activations, as given by the tensor numbers (these will not be visible to the user;
    they will be generated from the nicer human-readable names and then fed in).

    Args:
        model: PyTorch model.
        input_args: Input arguments to the model's forward pass: either a single tensor, or a list of arguments.
        input_kwargs: Keyword arguments to the model's forward pass.
        layers_to_save: List of layers to save
        keep_unsaved_layers: Whether to keep layers in the ModelHistory log if they don't have saved activations.
        output_device: device where saved tensors will be stored: either 'same' to keep unchanged, or
            'cpu' or 'cuda' to move to cpu or cuda.
        activation_postfunc: Function to apply to activations before saving them (e.g., any averaging)
        mark_input_output_distances: Whether to compute the distance of each layer from the input or output.
            This is computationally expensive for large networks, so it is off by default.
        detach_saved_tensors: whether to detach the saved tensors, so they remain attached to the computational graph
        save_function_args: whether to save the arguments to each function
        save_gradients: whether to save gradients from any subsequent backward pass
        random_seed: Which random seed to use.

    Returns:
        ModelHistory object with full log of the forward pass
    """
    model_name = str(type(model).__name__)
    model_history = ModelHistory(
        model_name,
        output_device,
        activation_postfunc,
        keep_unsaved_layers,
        save_function_args,
        save_gradients,
        detach_saved_tensors,
        mark_input_output_distances,
    )
    model_history.run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, random_seed
    )
    return model_history
