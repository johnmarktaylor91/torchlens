import copy
import itertools as it
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING, Tuple, Union

import numpy as np
import torch
from torch import nn

from .helper_funcs import (_get_call_stack_dicts, get_attr_values_from_tensor_list, get_tensor_memory_amount,
                           get_vars_of_type_from_obj, human_readable_size, index_nested, log_current_rng_states,
                           make_random_barcode, make_short_barcode_from_input, make_var_iterable)
from .tensor_log import TensorLogEntry

if TYPE_CHECKING:
    from .model_history import ModelHistory


def save_new_activations(
        self: "ModelHistory",
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
    self._tensor_counter = 0
    self._raw_layer_type_counter = defaultdict(lambda: 0)

    # Now run and log the new inputs.
    self._run_and_log_inputs_through_model(
        model, input_args, input_kwargs, layers_to_save, random_seed
    )


def log_source_tensor(
        self, t: torch.Tensor, source: str, extra_address: Optional[str] = None
):
    if self.logging_mode == "exhaustive":
        log_source_tensor_exhaustive(self, t, source, extra_address)
    elif self.logging_mode == "fast":
        log_source_tensor_fast(self, t, source)


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
    self._tensor_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    realtime_tensor_num = self._tensor_counter
    layer_type_num = self._raw_layer_type_counter[layer_type]

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
        "_pass_finished": False,
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
        "func_call_stack": _get_call_stack_dicts(),
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

    _make_tensor_log_entry(self, t, fields_dict, (), {}, self.activation_postfunc)

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
    self._tensor_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    layer_type_num = self._raw_layer_type_counter[layer_type]

    tensor_label_raw = f"{layer_type}_{layer_type_num}_raw"
    t.tl_tensor_label_raw = tensor_label_raw
    if tensor_label_raw in self.orphan_layers:
        return
    orig_tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
    if orig_tensor_label in self.unlogged_layers:
        return
    orig_tensor_entry = self.layer_dict_all_keys[orig_tensor_label]
    if (self._tensor_nums_to_save == "all") or (
            orig_tensor_entry.realtime_tensor_num in self._tensor_nums_to_save
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
        log_function_output_tensors_exhaustive(
            self,
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
        log_function_output_tensors_fast(
            self,
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

    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {
        key: val
        for key, val in kwargs.items()
        if not _check_if_tensor_arg(val)
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
    fields_dict["func_call_stack"] = _get_call_stack_dicts()
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
    parent_layer_arg_locs = _get_parent_tensor_function_call_location(self,
                                                                      parent_layer_entries, args, kwargs
                                                                      )
    (
        input_ancestors,
        internally_initialized_ancestors,
    ) = _get_ancestors_from_parents(parent_layer_entries)
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
    parent_param_passes = _process_parent_param_passes(arg_parameters)
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

    containing_modules_origin_nested = _get_input_module_info(self, arg_tensors)
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
        if not _output_should_be_logged(out, is_bottom_level_func):
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
        _log_info_specific_to_single_function_output_tensor(
            self, out, i, args, kwargs, parent_param_passes, fields_dict_onetensor
        )
        _make_tensor_log_entry(
            self,
            out,
            fields_dict=fields_dict_onetensor,
            t_args=arg_copies,
            t_kwargs=kwarg_copies,
            activation_postfunc=self.activation_postfunc,
        )
        new_tensor_entry = self[fields_dict_onetensor["tensor_label_raw"]]
        new_tensor_label = new_tensor_entry.tensor_label_raw
        _update_tensor_family_links(self, new_tensor_entry)

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
            self._layers_where_internal_branches_merge_with_input.append(
                new_tensor_label
            )

        # Tag the tensor itself with its label, and add a backward hook if saving gradients.
        out.tl_tensor_label_raw = fields_dict_onetensor["tensor_label_raw"]
        if self.save_gradients:
            _add_backward_hook(self, out, out.tl_tensor_label_raw)

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
                parent_tensor_contents = _get_parent_contents(
                    parent_label,
                    arg_copies,
                    kwarg_copies,
                    new_tensor_entry.parent_layer_arg_locs,
                )
                parent.children_tensor_versions[
                    new_tensor_entry.tensor_label_raw
                ] = parent_tensor_contents


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
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {
        key: val
        for key, val in kwargs.items()
        if not _check_if_tensor_arg(val)
    }

    arg_tensors = get_vars_of_type_from_obj(
        all_args, torch.Tensor, [torch.nn.Parameter]
    )
    out_iter = make_var_iterable(out_orig)

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        self._tensor_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self._tensor_counter
        layer_type_num = self._raw_layer_type_counter[layer_type]
        tensor_label_raw = (
            f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"
        )
        if tensor_label_raw in self.orphan_layers:
            continue
        parent_layer_labels_raw = get_attr_values_from_tensor_list(
            arg_tensors, "tl_tensor_label_raw"
        )
        parent_layer_labels_orig = [
            self._raw_to_final_layer_labels[raw_label]
            for raw_label in parent_layer_labels_raw
        ]
        out.tl_tensor_label_raw = tensor_label_raw
        if tensor_label_raw not in self._raw_to_final_layer_labels:
            raise ValueError(
                "The computational graph changed for this forward pass compared to the original "
                "call to log_forward_pass (either due to different inputs or a different "
                "random seed), so save_new_activations failed. Please re-run "
                "log_forward_pass with the desired inputs."
            )
        orig_tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
        if orig_tensor_label in self.unlogged_layers:
            continue
        orig_tensor_entry = self[orig_tensor_label]

        # Check to make sure the graph didn't change.
        if any(
                [
                    orig_tensor_entry.realtime_tensor_num != self._tensor_counter,
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
        if (self._tensor_nums_to_save == "all") or (
                orig_tensor_entry.realtime_tensor_num in self._tensor_nums_to_save
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
    self._tensor_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    realtime_tensor_num = self._tensor_counter
    layer_type_num = self._raw_layer_type_counter[layer_type]
    tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

    if len(parent_param_passes) > 0:
        operation_equivalence_type = _make_raw_param_group_barcode(
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
        operation_equivalence_type = _get_operation_equivalence_type(
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
    fields_dict["_pass_finished"] = False

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
    if (self._tensor_nums_to_save == "all") or (
            new_entry.realtime_tensor_num in self._tensor_nums_to_save
    ):
        new_entry.save_tensor_data(
            t, t_args, t_kwargs, self.save_function_args, activation_postfunc
        )
        self.layers_with_saved_activations.append(new_entry.tensor_label_raw)
    self._raw_tensor_dict[new_entry.tensor_label_raw] = new_entry
    self._raw_tensor_labels_list.append(new_entry.tensor_label_raw)

    return new_entry


def _log_tensor_grad(self, grad: torch.Tensor, tensor_label_raw: str):
    """Logs the gradient for a tensor during a backward pass.

    Args:
        grad: the gradient
        tensor_label_raw: the raw tensor label

    Returns:

    """
    self.has_saved_gradients = True
    tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
    if tensor_label not in self.layers_with_saved_gradients:
        self.layers_with_saved_gradients.append(tensor_label)
        layer_order = {layer: i for i, layer in enumerate(self.layer_labels)}
        self.layers_with_saved_gradients = sorted(
            self.layers_with_saved_gradients, key=lambda x: layer_order[x]
        )
    tensor_log_entry = self[tensor_label]
    tensor_log_entry.log_tensor_grad(grad)


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
            _find_arg_positions_for_single_parent(
                parent_entry, arg_type, arg_struct, tensor_all_arg_positions
            )

    return tensor_all_arg_positions


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
        _add_sibling_labels_for_new_tensor(
            self, entry_to_update, self[parent_tensor_label]
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
        args: Tuple, kwargs: Dict, i: int, layer_type: str, fields_dict: Dict
):
    arg_hash = _get_hash_from_args(args, kwargs)
    operation_equivalence_type = f"{layer_type}_{arg_hash}"
    if fields_dict["is_part_of_iterable_output"]:
        operation_equivalence_type += f"_outindex{i}"
    if fields_dict["containing_module_origin"] is not None:
        module_str = fields_dict["containing_module_origin"][0]
        operation_equivalence_type += f"_module{module_str}"
    return operation_equivalence_type


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
        containing_modules = _update_tensor_containing_modules(tensor_entry)
        if len(containing_modules) > max_input_module_nesting:
            max_input_module_nesting = len(containing_modules)
            most_nested_containing_modules = containing_modules[:]
    return most_nested_containing_modules
