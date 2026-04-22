"""Functions for logging source tensors (inputs and buffers) during model tracing.

Source tensors are the starting points of the computational graph: model inputs
and module buffers.  This module handles creating LayerPassLog entries for these
tensors in both exhaustive and fast logging modes.

Source tensors differ from function-output tensors in several ways:
  - They have no parent layers (``parent_layers=[]``).
  - Inputs are roots with ``has_input_ancestor=True``; buffers are internally
    initialized with ``has_internally_initialized_ancestor=True``.
  - Their ``func_applied`` is None and ``func_name`` is ``"none"``.
  - Buffer labels follow ``"buffer_{N}_raw"``; input labels follow ``"input_{N}_raw"``.
  - Buffers may carry a ``tl_buffer_parent`` attribute (set during model prep)
    identifying the module that owns them.
  - Buffer entries are instantiated as ``BufferLog`` (a LayerPassLog subclass
    that adds ``name`` and ``module_address`` fields).

The ``operation_equivalence_type`` for inputs encodes shape+dtype (so inputs
with different shapes are distinct equivalence classes).  For buffers, it
encodes the buffer's module address (so the same buffer across passes is
recognized as the same layer).
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch import nn

from ..utils.introspection import _get_func_call_stack, get_attr_values_from_tensor_list
from ..utils.tensor_utils import get_tensor_memory_amount
from ..utils.rng import log_current_rng_states
from ..data_classes.buffer_log import BufferLog
from ..data_classes.layer_pass_log import LayerPassLog

from .tensor_tracking import _add_backward_hook, _update_tensor_containing_modules

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def log_source_tensor(self, t: torch.Tensor, source: str, extra_address: Optional[str] = None):
    """Dispatch source tensor logging to exhaustive or fast mode.

    Called explicitly for model inputs (from ``run_and_log_inputs_through_model``)
    and for module buffers (from the module forward decorator in model_prep.py).

    Args:
        t: The source tensor (input or buffer).
        source: ``"input"`` or ``"buffer"``.
        extra_address: For inputs, the address string (e.g. ``"input.x"``);
            for buffers, the buffer's module address (e.g. ``"encoder.bn.running_mean"``).
    """
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
    creation_order = self._layer_counter
    layer_type_num = self._raw_layer_type_counter[layer_type]

    tensor_label = f"{layer_type}_{layer_type_num}_raw"

    # Configure source-type-specific fields.
    # Inputs are graph roots with themselves as their own input ancestor.
    # Buffers are internally initialized (no input ancestry).
    if source == "input":
        is_input_layer = True
        has_input_ancestor = True
        io_role = extra_addr
        is_buffer_layer = False
        buffer_address = None
        buffer_parent = None
        is_internally_initialized = False
        has_internally_initialized_ancestor = False
        input_ancestors = {tensor_label}
        internally_initialized_ancestors = set()
        # Inputs with different shapes/dtypes get different equivalence types
        # so they're not grouped as the "same layer" by loop detection.
        operation_equivalence_type = (
            f"input_{'_'.join(tuple(str(s) for s in t.shape))}_{str(t.dtype)}"
        )
    elif source == "buffer":
        is_input_layer = False
        has_input_ancestor = False
        io_role = None
        is_buffer_layer = True
        buffer_address = extra_addr
        is_internally_initialized = True
        has_internally_initialized_ancestor = True
        internally_initialized_ancestors = {tensor_label}
        input_ancestors = set()
        # Buffer equivalence keyed by module address so the same buffer
        # is recognized as the same layer across loop iterations.
        operation_equivalence_type = f"buffer_{extra_addr}"
        # tl_buffer_parent is set during model_prep on buffers that belong
        # to a specific module; None for detached or anonymous buffers.
        if hasattr(t, "tl_buffer_parent"):
            buffer_parent = t.tl_buffer_parent
        else:
            buffer_parent = None
    else:
        raise ValueError("source must be either 'input' or 'buffer'")

    tensor_memory = get_tensor_memory_amount(t)

    fields_dict = {
        # General info:
        "tensor_label_raw": tensor_label,
        "layer_label_raw": tensor_label,
        "creation_order": creation_order,
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
        "num_passes": 1,
        "lookup_keys": [],
        # Saved tensor info:
        "activation": None,
        "has_saved_activations": False,
        "activation_postfunc": self.activation_postfunc,
        "detach_saved_tensor": self.detach_saved_tensors,
        "output_device": self.output_device,
        "args_captured": False,
        "captured_args": None,
        "captured_kwargs": None,
        "tensor_shape": tuple(t.shape),
        "tensor_dtype": t.dtype,
        "tensor_memory": tensor_memory,
        # Child tensor variation tracking
        "has_child_tensor_variations": False,
        "children_tensor_versions": {},
        # Grad info:
        "gradient": None,
        "save_gradients": self.save_gradients,
        "has_gradient": False,
        "grad_shape": None,
        "grad_dtype": None,
        "grad_memory": 0,
        # Function call info:
        "func_applied": None,
        "func_name": "none",
        "func_call_stack": _get_func_call_stack(
            self.num_context_lines, source_loading_enabled=self.save_source_context
        ),
        "func_time": 0,
        "flops_forward": 0,
        "flops_backward": 0,
        "func_rng_states": log_current_rng_states(torch_only=True) if self.save_rng_states else {},
        "func_autocast_state": {},
        "func_argnames": (),
        "num_args": 0,
        "num_positional_args": 0,
        "num_keyword_args": 0,
        "func_positional_args_non_tensor": [],
        "func_kwargs_non_tensor": {},
        "func_non_tensor_args": [],
        "func_is_inplace": False,
        "grad_fn_name": "none",
        "is_part_of_iterable_output": False,
        "iterable_output_index": None,
        # Param info:
        "parent_params": [],
        "parent_param_barcodes": [],
        "parent_param_passes": {},
        "parent_param_logs": [],
        "parent_param_shapes": [],
        "num_params_total": int(0),
        "num_params_trainable": 0,
        "num_params_frozen": 0,
        "params_memory": 0,
        # Corresponding layer info:
        "operation_equivalence_type": operation_equivalence_type,
        "equivalent_operations": self.equivalent_operations[operation_equivalence_type],
        "recurrent_group": [],
        # Graph info:
        "parent_layers": [],
        "parent_layer_arg_locs": {"args": {}, "kwargs": {}},
        "root_ancestors": {tensor_label},
        "child_layers": [],
        "has_children": False,
        "is_input_layer": is_input_layer,
        "has_input_ancestor": has_input_ancestor,
        "input_ancestors": input_ancestors,
        "min_distance_from_input": None,
        "max_distance_from_input": None,
        "is_output_layer": False,
        "feeds_output": False,
        "is_final_output": False,
        "is_output_ancestor": False,
        "output_descendants": set(),
        "min_distance_from_output": None,
        "max_distance_from_output": None,
        "io_role": io_role,
        "is_buffer_layer": is_buffer_layer,
        "buffer_address": buffer_address,
        "buffer_pass": None,
        "buffer_parent": buffer_parent,
        "is_internally_initialized": is_internally_initialized,
        "has_internally_initialized_ancestor": has_internally_initialized_ancestor,
        "internally_initialized_parents": [],
        "internally_initialized_ancestors": internally_initialized_ancestors,
        "is_internally_terminated": False,
        # Conditional info:
        "is_terminal_bool_layer": False,
        "bool_is_branch": False,
        "bool_context_kind": None,
        "bool_wrapper_kind": None,
        "bool_conditional_id": None,
        "is_scalar_bool": False,
        "scalar_bool_value": None,
        "in_cond_branch": False,
        "conditional_branch_stack": [],
        "conditional_branch_depth": 0,
        "cond_branch_start_children": [],
        "cond_branch_then_children": [],
        "cond_branch_elif_children": {},
        "cond_branch_else_children": [],
        "cond_branch_children_by_cond": {},
        # Module info:
        "containing_module": None,
        "containing_modules": [],
        "modules_entered": [],
        "modules_entered_argnames": defaultdict(list),
        "module_passes_entered": [],
        "modules_exited": [],
        "module_passes_exited": [],
        "is_submodule_output": False,
        "is_leaf_module_output": False,
        "leaf_module_pass": None,
        "module_entry_exit_threads_inputs": [],
        "module_entry_exit_thread_output": [],
        # Function config
        "func_config": {},
    }

    # Reuse the shared entry-creation logic from output_tensors.
    # Imported here (not at module level) to avoid circular imports.
    from .output_tensors import _make_layer_log_entry

    # Creates a BufferLog if is_buffer_layer=True, else LayerPassLog.
    _make_layer_log_entry(self, t, fields_dict, (), {}, self.activation_postfunc)

    # Tag the live tensor so downstream operations can find this tensor's label.
    t.tl_tensor_label_raw = tensor_label  # type: ignore[attr-defined]

    # Register in ModelLog-level tracking structures.
    self.equivalent_operations[operation_equivalence_type].add(t.tl_tensor_label_raw)  # type: ignore[attr-defined]
    if source == "input":
        self.input_layers.append(tensor_label)
    if source == "buffer":
        self.buffer_layers.append(tensor_label)
        self.internally_initialized_layers.append(tensor_label)

    # Register backward hook for gradient capture if requested.
    if self.save_gradients:
        _add_backward_hook(self, t, t.tl_tensor_label_raw)  # type: ignore[attr-defined]


def log_source_tensor_fast(self, t: torch.Tensor, source: str):
    """Fast-path source tensor logging: save new activation into existing entry.

    Mirrors the exhaustive pass's counter increments for alignment, then
    saves the tensor value and updates shape/dtype/size metadata.  Does NOT
    rebuild the fields_dict or create new log entries.
    """
    layer_type = source
    # Fetch counters and increment to be ready for next tensor to be logged
    self._layer_counter += 1
    self._raw_layer_type_counter[layer_type] += 1
    layer_type_num = self._raw_layer_type_counter[layer_type]

    # Source tensor raw labels omit the realtime_num component (unlike function
    # outputs) because source tensors are identified only by type and type_num.
    tensor_label_raw = f"{layer_type}_{layer_type_num}_raw"
    # Tag tensor for downstream fast-path ops to identify it.
    t.tl_tensor_label_raw = tensor_label_raw  # type: ignore[attr-defined]
    if tensor_label_raw in self.orphan_layers:
        return
    orig_tensor_label = self._raw_to_final_layer_labels.get(tensor_label_raw)
    if orig_tensor_label is None:
        raise ValueError(
            f"Fast-path label '{tensor_label_raw}' has no mapping in _raw_to_final_layer_labels. "
            f"This usually means the computational graph changed between the exhaustive pass "
            f"and this fast pass (e.g., dynamic control flow). Use log_forward_pass() instead."
        )
    if orig_tensor_label in self.unlogged_layers:
        return
    orig_layer_entry = self.layer_dict_main_keys[orig_tensor_label]
    if (self._layer_nums_to_save == "all") or (
        orig_layer_entry.creation_order in self._layer_nums_to_save
    ):
        self.layers_with_saved_activations.append(orig_layer_entry.layer_label)
        orig_layer_entry.save_tensor_data(
            t, [], {}, self.save_function_args, self.activation_postfunc
        )

    # Minimal graph consistency validation (#99)
    new_shape = tuple(t.shape)
    if orig_layer_entry.tensor_shape is not None and new_shape != orig_layer_entry.tensor_shape:
        import warnings

        warnings.warn(
            f"Tensor shape changed for '{orig_tensor_label}': "
            f"expected {orig_layer_entry.tensor_shape}, got {new_shape}. "
            f"The computational graph may have changed between passes."
        )
    orig_layer_entry.tensor_shape = new_shape
    orig_layer_entry.tensor_dtype = t.dtype
    memory = get_tensor_memory_amount(t)
    orig_layer_entry.tensor_memory = memory


def _get_input_module_info(self, arg_tensors: List[torch.Tensor]) -> List[str]:
    """Determine the module nesting context for a new tensor from its parents.

    Finds the most deeply nested parent tensor and returns its current
    containing-module stack (after applying any module entry/exit transitions
    recorded in its ``module_entry_exit_thread_output``).  This determines
    which module the new operation is "inside" for module-level metadata.

    Args:
        arg_tensors: List of parent tensors (input arguments to the function).

    Returns:
        List of module pass strings (e.g. ``["encoder:1", "encoder.layer1:1"]``)
        representing the nesting stack of the most deeply nested parent.
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
