"""Functions for logging output tensors produced by decorated torch operations.

This module handles the creation and population of TensorLog entries for every
tensor produced during a forward pass.  It covers both *exhaustive* mode (full
metadata collection) and *fast* mode (re-use of a previously logged graph with
new activations).
"""

import copy
from collections import defaultdict
from math import prod
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch

from .. import _state as _st
from ..utils.introspection import (
    _get_func_call_stack,
    get_attr_values_from_tensor_list,
    get_vars_of_type_from_obj,
)
from ..utils.tensor_utils import get_tensor_memory_amount, safe_copy, tensor_nanequal
from ..utils.display import human_readable_size
from ..utils.collections import index_nested, ensure_iterable
from .flops import compute_backward_flops, compute_forward_flops
from ..data_classes.buffer_log import BufferLog
from ..data_classes.tensor_log import TensorLog

from .tensor_tracking import (
    _update_tensor_family_links,
    _locate_parent_tensors_in_args,
    _get_ancestors_from_parents,
    _add_backward_hook,
    _process_parent_param_passes,
    _make_raw_param_group_barcode,
    _get_operation_equivalence_type,
    _get_hash_from_args,
    _add_sibling_labels_for_new_tensor,
    _update_tensor_containing_modules,
)
from ..data_classes.internal_types import FuncExecutionContext
from .source_tensors import _get_input_module_info

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def log_function_output_tensors(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
):
    if self.logging_mode == "exhaustive":
        log_function_output_tensors_exhaustive(
            self,
            func,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
        )
    elif self.logging_mode == "fast":
        log_function_output_tensors_fast(
            self,
            func_name,
            args,
            kwargs,
            arg_copies,
            kwarg_copies,
            out_orig,
            exec_ctx,
            is_bottom_level_func,
        )


def _build_graph_relationship_fields(
    self,
    fields_dict: Dict[str, Any],
    parent_layer_labels: List[str],
    parent_layer_entries: List,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
) -> None:
    """Populate graph structure fields: parents, children, ancestors, buffer/IO flags."""
    parent_layer_arg_locs = _locate_parent_tensors_in_args(self, parent_layer_entries, args, kwargs)
    input_ancestors, internally_initialized_ancestors = _get_ancestors_from_parents(
        parent_layer_entries
    )
    internal_parent_layer_labels = [
        label for label in parent_layer_labels if self[label].has_internally_initialized_ancestor
    ]

    fields_dict["parent_layers"] = parent_layer_labels
    fields_dict["parent_layer_arg_locs"] = parent_layer_arg_locs
    fields_dict["has_parents"] = len(parent_layer_labels) > 0
    fields_dict["orig_ancestors"] = input_ancestors.union(internally_initialized_ancestors)
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
    fields_dict["has_internally_initialized_ancestor"] = len(internally_initialized_ancestors) > 0
    fields_dict["internally_initialized_parents"] = internal_parent_layer_labels
    fields_dict["internally_initialized_ancestors"] = internally_initialized_ancestors
    fields_dict["terminated_inside_model"] = False
    fields_dict["is_terminal_bool_layer"] = False
    fields_dict["in_cond_branch"] = False
    fields_dict["cond_branch_start_children"] = []

    is_part_of_iterable_output = any(
        issubclass(type(out_orig), cls) for cls in [list, tuple, dict, set]
    )
    fields_dict["is_part_of_iterable_output"] = is_part_of_iterable_output


def _build_param_fields(
    self,
    fields_dict: Dict[str, Any],
    all_args: List,
) -> Dict:
    """Populate parameter-involvement fields. Returns parent_param_passes dict."""
    arg_parameters = get_vars_of_type_from_obj(all_args, torch.nn.parameter.Parameter)
    parent_param_passes = _process_parent_param_passes(arg_parameters)
    indiv_param_barcodes = list(parent_param_passes.keys())

    parent_param_logs = []
    for param in arg_parameters:
        addr = getattr(param, "tl_param_address", None)
        if addr is not None and addr in self.param_logs:
            parent_param_logs.append(self.param_logs[addr])

    fields_dict["computed_with_params"] = len(parent_param_passes) > 0
    fields_dict["parent_params"] = arg_parameters
    fields_dict["parent_param_barcodes"] = indiv_param_barcodes
    fields_dict["parent_param_passes"] = parent_param_passes
    fields_dict["parent_param_logs"] = parent_param_logs
    fields_dict["num_param_tensors"] = len(arg_parameters)
    fields_dict["parent_param_shapes"] = [tuple(param.shape) for param in arg_parameters]
    fields_dict["num_params_total"] = sum(
        prod(shape) for shape in fields_dict["parent_param_shapes"]
    )
    fields_dict["num_params_trainable"] = sum(
        pl.num_params for pl in parent_param_logs if pl.trainable
    )
    fields_dict["num_params_frozen"] = sum(
        pl.num_params for pl in parent_param_logs if not pl.trainable
    )
    fields_dict["parent_params_fsize"] = sum(get_tensor_memory_amount(p) for p in arg_parameters)
    fields_dict["parent_params_fsize_nice"] = human_readable_size(
        fields_dict["parent_params_fsize"]
    )
    return parent_param_passes


def _build_module_context_fields(
    self,
    fields_dict: Dict[str, Any],
    arg_tensors: List,
    parent_layer_entries: List,
) -> None:
    """Populate module nesting, address, and input/output status fields."""
    containing_modules_origin_nested = _get_input_module_info(self, arg_tensors)
    if len(containing_modules_origin_nested) > 0:
        is_computed_inside_submodule = True
        containing_module_origin = containing_modules_origin_nested[-1]
    else:
        is_computed_inside_submodule = False
        containing_module_origin = None

    fields_dict["is_computed_inside_submodule"] = is_computed_inside_submodule
    fields_dict["containing_module_origin"] = containing_module_origin
    fields_dict["containing_modules_origin_nested"] = containing_modules_origin_nested
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
        p.tensor_label_raw: p.module_entry_exit_thread_output[:] for p in parent_layer_entries
    }
    fields_dict["module_entry_exit_thread_output"] = []


def _build_shared_fields_dict(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
) -> Tuple[Dict[str, Any], List, List, Dict]:
    """Build the fields_dict shared by all output tensors of a single function call.

    Returns:
        (fields_dict, parent_layer_entries, arg_tensors, parent_param_passes)
    """
    layer_type = func_name.lower().replace("_", "")
    all_args = list(args) + list(kwargs.values())

    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}
    arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.Parameter])
    parent_layer_labels = get_attr_values_from_tensor_list(arg_tensors, "tl_tensor_label_raw")
    parent_layer_entries = [self[label] for label in parent_layer_labels]

    fields_dict = {}

    # General info
    fields_dict["layer_type"] = layer_type
    fields_dict["detach_saved_tensor"] = self.detach_saved_tensors
    fields_dict["output_device"] = self.output_device

    # Grad info
    fields_dict["grad_contents"] = None
    fields_dict["save_gradients"] = self.save_gradients
    fields_dict["has_saved_grad"] = False
    fields_dict["grad_shape"] = None
    fields_dict["grad_dtype"] = None
    fields_dict["grad_fsize"] = 0
    fields_dict["grad_fsize_nice"] = human_readable_size(0)

    # Function call info
    fields_dict["func_applied"] = func
    fields_dict["func_applied_name"] = func_name
    fields_dict["func_call_stack"] = _get_func_call_stack(self.num_context_lines)
    fields_dict["func_time_elapsed"] = exec_ctx.time_elapsed
    fields_dict["func_rng_states"] = exec_ctx.rng_states
    fields_dict["func_autocast_state"] = exec_ctx.autocast_state
    fields_dict["func_argnames"] = _st._func_argnames.get(func_name.strip("_"), ())
    fields_dict["num_func_args_total"] = len(args) + len(kwargs)
    fields_dict["num_position_args"] = len(args)
    fields_dict["num_keyword_args"] = len(kwargs)
    fields_dict["func_position_args_non_tensor"] = non_tensor_args
    fields_dict["func_keyword_args_non_tensor"] = non_tensor_kwargs
    fields_dict["func_all_args_non_tensor"] = non_tensor_args + list(non_tensor_kwargs.values())

    _build_graph_relationship_fields(
        self, fields_dict, parent_layer_labels, parent_layer_entries, args, kwargs, out_orig
    )
    parent_param_passes = _build_param_fields(self, fields_dict, all_args)
    _build_module_context_fields(self, fields_dict, arg_tensors, parent_layer_entries)

    return fields_dict, parent_layer_entries, arg_tensors, parent_param_passes


def _classify_new_tensor_in_model_log(
    self,
    fields_dict: Dict[str, Any],
    fields_dict_onetensor: Dict[str, Any],
    new_tensor_label: str,
) -> None:
    """Update ModelLog categories (internally_initialized, merge points) for a new tensor."""
    if fields_dict["initialized_inside_model"]:
        self.internally_initialized_layers.append(new_tensor_label)
    if fields_dict["has_input_ancestor"] and any(
        (
            self[parent_layer].has_internally_initialized_ancestor
            and not self[parent_layer].has_input_ancestor
        )
        for parent_layer in fields_dict_onetensor["parent_layers"]
    ):
        self._layers_where_internal_branches_merge_with_input.append(new_tensor_label)


def _tag_tensor_and_track_variations(
    self,
    out: torch.Tensor,
    new_tensor_entry,
    fields_dict_onetensor: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
) -> None:
    """Tag the output tensor with its label, add backward hook, and track parent content variations."""
    out.tl_tensor_label_raw = fields_dict_onetensor["tensor_label_raw"]
    if self.save_gradients:
        _add_backward_hook(self, out, out.tl_tensor_label_raw)

    for parent_label in new_tensor_entry.parent_layers:
        parent = self[parent_label]
        if parent.has_saved_activations and self.save_function_args:
            parent_tensor_contents = _get_parent_contents(
                parent_label,
                arg_copies,
                kwarg_copies,
                new_tensor_entry.parent_layer_arg_locs,
            )
            if not tensor_nanequal(parent_tensor_contents, parent.tensor_contents):
                parent.children_tensor_versions[new_tensor_entry.tensor_label_raw] = (
                    parent_tensor_contents
                )
                parent.has_child_tensor_variations = True


def log_function_output_tensors_exhaustive(
    self,
    func: Callable,
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
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
        exec_ctx: Timing, RNG, and autocast state captured around the function call.
        is_bottom_level_func: whether the function is at the bottom-level of function nesting
    """
    fields_dict, parent_layer_entries, arg_tensors, parent_param_passes = _build_shared_fields_dict(
        self,
        func,
        func_name,
        args,
        kwargs,
        out_orig,
        exec_ctx,
    )

    out_iter = ensure_iterable(out_orig)

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue

        # Shallow-copy only mutable containers; immutable values (str, int, bool,
        # None, tuple, torch.dtype) don't need copying.
        fields_dict_onetensor = {}
        for key, value in fields_dict.items():
            if isinstance(value, (list, dict, set)):
                fields_dict_onetensor[key] = copy.copy(value)
            else:
                fields_dict_onetensor[key] = value
        # These nested structures need deep copies to avoid cross-tensor mutation.
        for field in (
            "parent_layer_arg_locs",
            "containing_modules_origin_nested",
            "parent_param_passes",
        ):
            fields_dict_onetensor[field] = copy.deepcopy(fields_dict[field])
        _log_output_tensor_info(
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

        _classify_new_tensor_in_model_log(
            self, fields_dict, fields_dict_onetensor, new_tensor_label
        )
        _tag_tensor_and_track_variations(
            self,
            out,
            new_tensor_entry,
            fields_dict_onetensor,
            arg_copies,
            kwarg_copies,
        )


def _get_parent_contents(parent_label, arg_copies, kwarg_copies, parent_layer_arg_locs) -> Any:
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
    func_name: str,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    arg_copies: Tuple[Any],
    kwarg_copies: Dict[str, Any],
    out_orig: Any,
    exec_ctx: FuncExecutionContext,
    is_bottom_level_func: bool,
):
    # Collect information.
    layer_type = func_name.lower().replace("_", "")
    all_args = list(args) + list(kwargs.values())
    non_tensor_args = [arg for arg in args if not _check_if_tensor_arg(arg)]
    non_tensor_kwargs = {key: val for key, val in kwargs.items() if not _check_if_tensor_arg(val)}

    arg_tensors = get_vars_of_type_from_obj(all_args, torch.Tensor, [torch.nn.Parameter])
    out_iter = ensure_iterable(out_orig)

    for i, out in enumerate(out_iter):
        if not _output_should_be_logged(out, is_bottom_level_func):
            continue
        self._tensor_counter += 1
        self._raw_layer_type_counter[layer_type] += 1
        realtime_tensor_num = self._tensor_counter
        layer_type_num = self._raw_layer_type_counter[layer_type]
        tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"
        if tensor_label_raw in self.orphan_layers:
            continue
        parent_layer_labels_raw = get_attr_values_from_tensor_list(
            arg_tensors, "tl_tensor_label_raw"
        )
        parent_layer_labels_orig = [
            self._raw_to_final_layer_labels[raw_label]
            for raw_label in parent_layer_labels_raw
            if raw_label in self._raw_to_final_layer_labels
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

        if self.save_gradients:
            _add_backward_hook(self, out, orig_tensor_label)

        # Check to make sure the graph didn't change.
        if (
            orig_tensor_entry.realtime_tensor_num != self._tensor_counter
            or orig_tensor_entry.layer_type != layer_type
            or orig_tensor_entry.tensor_label_raw != tensor_label_raw
            or set(orig_tensor_entry.parent_layers) != set(parent_layer_labels_orig)
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
                    if (
                        orig_tensor_entry.has_child_tensor_variations
                        and child_layer in orig_tensor_entry.children_tensor_versions
                    ):
                        # children_tensor_versions already has transforms applied
                        tensor_to_save = orig_tensor_entry.children_tensor_versions[child_layer]
                        child_output.tensor_contents = safe_copy(tensor_to_save)
                    else:
                        child_output.tensor_contents = safe_copy(out)
                        if self.activation_postfunc is not None:
                            child_output.tensor_contents = self.activation_postfunc(
                                child_output.tensor_contents
                            )
                    child_output.has_saved_activations = True
                    child_output.tensor_fsize = get_tensor_memory_amount(
                        child_output.tensor_contents
                    )
                    child_output.tensor_fsize_nice = human_readable_size(child_output.tensor_fsize)

        orig_tensor_entry.tensor_shape = tuple(out.shape)
        orig_tensor_entry.tensor_dtype = out.dtype
        fsize = get_tensor_memory_amount(out)
        orig_tensor_entry.tensor_fsize = fsize
        orig_tensor_entry.tensor_fsize_nice = human_readable_size(fsize)
        orig_tensor_entry.func_time_elapsed = exec_ctx.time_elapsed
        orig_tensor_entry.func_rng_states = exec_ctx.rng_states
        orig_tensor_entry.func_autocast_state = exec_ctx.autocast_state
        orig_tensor_entry.func_position_args_non_tensor = non_tensor_args
        orig_tensor_entry.func_keyword_args_non_tensor = non_tensor_kwargs


def _output_should_be_logged(out: Any, is_bottom_level_func: bool) -> bool:
    """Function to check whether to log the output of a function.

    Returns:
        True if the output should be logged, False otherwise.
    """
    if type(out) is not torch.Tensor:  # only log if it's a tensor
        return False

    if (not hasattr(out, "tl_tensor_label_raw")) or is_bottom_level_func:
        return True
    else:
        return False


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
        return False
    elif type(arg) == dict:
        for val in arg.values():
            if issubclass(type(val), torch.Tensor):
                return True
        return False
    else:
        return False


def _log_output_tensor_info(
    self,
    t: torch.Tensor,
    i: int,
    args: Tuple[Any],
    kwargs: Dict[str, Any],
    parent_param_passes: Dict[str, int],
    fields_dict: Dict[str, Any],
) -> None:
    """Logs info specific to a single output tensor (e.g. shape, equivalence type, FLOPs).

    Handles fields that differ per output tensor in a multi-output function call,
    as opposed to fields shared by all outputs of the same call.

    Args:
        t: tensor to log
        i: index of the tensor in the function output
        args: positional args to the function that created the tensor
        kwargs: keyword args to the function that created the tensor
        parent_param_passes: Dict mapping barcodes of parent params to how many passes they've seen
        fields_dict: dictionary of fields with which to initialize the new TensorLog
    """
    layer_type = fields_dict["layer_type"]
    indiv_param_barcodes = list(parent_param_passes.keys())
    self._tensor_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    realtime_tensor_num = self._tensor_counter
    layer_type_num = self._raw_layer_type_counter[layer_type]
    tensor_label_raw = f"{layer_type}_{layer_type_num}_{realtime_tensor_num}_raw"

    if len(parent_param_passes) > 0:
        operation_equivalence_type = _make_raw_param_group_barcode(indiv_param_barcodes, layer_type)
        fields_dict["operation_equivalence_type"] = operation_equivalence_type
        self.layers_computed_with_params[operation_equivalence_type].append(tensor_label_raw)
        fields_dict["pass_num"] = len(self.layers_computed_with_params[operation_equivalence_type])
    else:
        operation_equivalence_type = _get_operation_equivalence_type(
            args, kwargs, i, layer_type, fields_dict
        )
        fields_dict["operation_equivalence_type"] = operation_equivalence_type
        fields_dict["pass_num"] = 1

    self.equivalent_operations[operation_equivalence_type].add(tensor_label_raw)
    fields_dict["equivalent_operations"] = self.equivalent_operations[operation_equivalence_type]

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
    fields_dict["source_model_log"] = self
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
    fields_dict["tensor_fsize_nice"] = human_readable_size(fields_dict["tensor_fsize"])

    # FLOPs computation
    fields_dict["flops_forward"] = compute_forward_flops(
        fields_dict.get("func_applied_name"),
        fields_dict["tensor_shape"],
        fields_dict.get("parent_param_shapes", []),
        args,
        kwargs,
    )
    fields_dict["flops_backward"] = compute_backward_flops(
        fields_dict.get("func_applied_name"),
        fields_dict["flops_forward"],
    )

    # Child tensor variation tracking
    fields_dict["has_child_tensor_variations"] = False
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
    and a pointer to ModelLog.

    Args:
        t: tensor to log
        fields_dict: dictionary of fields to log in TensorLog
        t_args: Positional arguments to the function that created the tensor
        t_kwargs: Keyword arguments to the function that created the tensor
        activation_postfunc: Function to apply to activations before saving them.
    """
    if t_args is None:
        t_args = []
    if t_kwargs is None:
        t_kwargs = {}

    if fields_dict.get("is_buffer_layer"):
        new_entry = BufferLog(fields_dict)
    else:
        new_entry = TensorLog(fields_dict)
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
