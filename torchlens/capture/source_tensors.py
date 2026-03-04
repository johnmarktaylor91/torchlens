"""Functions for logging source tensors (inputs and buffers) during model tracing.

Source tensors are the starting points of the computational graph: model inputs
and module buffers. This module handles creating LayerPassLog entries for these
tensors in both exhaustive and fast logging modes.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch import nn

from ..utils.introspection import _get_func_call_stack, get_attr_values_from_tensor_list
from ..utils.tensor_utils import get_tensor_memory_amount
from ..utils.display import human_readable_size
from ..utils.rng import log_current_rng_states
from ..data_classes.buffer_log import BufferLog
from ..data_classes.layer_pass_log import LayerPassLog

from .tensor_tracking import _add_backward_hook, _update_tensor_containing_modules

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def log_source_tensor(self, t: torch.Tensor, source: str, extra_address: Optional[str] = None):
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
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    realtime_tensor_num = self._layer_counter
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
        if hasattr(t, "tl_buffer_parent"):
            buffer_parent = t.tl_buffer_parent
        else:
            buffer_parent = None
    else:
        raise ValueError("source must be either 'input' or 'buffer'")

    tensor_fsize = get_tensor_memory_amount(t)

    fields_dict = {
        # General info:
        "tensor_label_raw": tensor_label,
        "layer_label_raw": tensor_label,
        "realtime_tensor_num": realtime_tensor_num,
        "operation_num": None,
        "source_model_log": self,
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
        "tensor_fsize": tensor_fsize,
        "tensor_fsize_nice": human_readable_size(tensor_fsize),
        # Child tensor variation tracking
        "has_child_tensor_variations": False,
        "children_tensor_versions": {},
        # Grad info:
        "grad_contents": None,
        "save_gradients": self.save_gradients,
        "has_saved_grad": False,
        "grad_shape": None,
        "grad_dtype": None,
        "grad_fsize": 0,
        "grad_fsize_nice": human_readable_size(0),
        # Function call info:
        "func_applied": None,
        "func_applied_name": "none",
        "func_call_stack": _get_func_call_stack(self.num_context_lines),
        "func_time_elapsed": 0,
        "flops_forward": 0,
        "flops_backward": 0,
        "func_rng_states": log_current_rng_states(),
        "func_autocast_state": {},
        "func_argnames": (),
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
        "parent_param_logs": [],
        "num_param_tensors": 0,
        "parent_param_shapes": [],
        "num_params_total": int(0),
        "num_params_trainable": 0,
        "num_params_frozen": 0,
        "parent_params_fsize": 0,
        "parent_params_fsize_nice": human_readable_size(0),
        # Corresponding layer info:
        "operation_equivalence_type": operation_equivalence_type,
        "equivalent_operations": self.equivalent_operations[operation_equivalence_type],
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

    from .output_tensors import _make_layer_log_entry

    _make_layer_log_entry(self, t, fields_dict, (), {}, self.activation_postfunc)

    # Tag the tensor itself with its label, and with a reference to the model history log.
    t.tl_tensor_label_raw = tensor_label

    # Log info to ModelLog
    self.equivalent_operations[operation_equivalence_type].add(t.tl_tensor_label_raw)
    if source == "input":
        self.input_layers.append(tensor_label)
    if source == "buffer":
        self.buffer_layers.append(tensor_label)
        self.internally_initialized_layers.append(tensor_label)

    # Make it track gradients if relevant

    if self.save_gradients:
        _add_backward_hook(self, t, t.tl_tensor_label_raw)


def log_source_tensor_fast(self, t: torch.Tensor, source: str):
    """NOTES TO SELF--fields to change are:
    for ModelLog: pass timing, num tensors saved, tensor fsize
    for Tensors: tensor contents, fsize, args, kwargs, make sure to clear gradients.
    Add a minimal postprocessing thing for tallying stuff pertaining to saved tensors.
    Have some minimal checker to make sure the graph didn't change.
    """
    layer_type = source
    # Fetch counters and increment to be ready for next tensor to be logged
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    layer_type_num = self._raw_layer_type_counter[layer_type]

    tensor_label_raw = f"{layer_type}_{layer_type_num}_raw"
    t.tl_tensor_label_raw = tensor_label_raw
    if tensor_label_raw in self.orphan_layers:
        return
    orig_tensor_label = self._raw_to_final_layer_labels[tensor_label_raw]
    if orig_tensor_label in self.unlogged_layers:
        return
    orig_layer_entry = self.layer_dict_main_keys[orig_tensor_label]
    if (self._layer_nums_to_save == "all") or (
        orig_layer_entry.realtime_tensor_num in self._layer_nums_to_save
    ):
        self.layers_with_saved_activations.append(orig_layer_entry.layer_label)
        orig_layer_entry.save_tensor_data(
            t, [], {}, self.save_function_args, self.activation_postfunc
        )

    orig_layer_entry.tensor_shape = tuple(t.shape)
    orig_layer_entry.tensor_dtype = t.dtype
    fsize = get_tensor_memory_amount(t)
    orig_layer_entry.tensor_fsize = fsize
    orig_layer_entry.tensor_fsize_nice = human_readable_size(fsize)


def _get_input_module_info(self, arg_tensors: List[torch.Tensor]) -> List[str]:
    """Utility function to extract information about module entry/exit from input tensors.

    Args:
        arg_tensors: List of input tensors

    Returns:
        List of containing module pass strings from the most deeply nested input tensor.
    """
    max_input_module_nesting = 0
    most_nested_containing_modules = []
    for t in arg_tensors:
        tensor_label = getattr(t, "tl_tensor_label_raw", -1)
        if tensor_label == -1:
            continue
        layer_entry = self[tensor_label]
        containing_modules = _update_tensor_containing_modules(layer_entry)
        if len(containing_modules) > max_input_module_nesting:
            max_input_module_nesting = len(containing_modules)
            most_nested_containing_modules = containing_modules[:]
    return most_nested_containing_modules
