"""Shared constants: field-order tuples that define the canonical set of LayerPassLog and ModelLog fields."""

import __future__
import functools
import types
from typing import List
import warnings

import torch
from torch.overrides import get_ignored_functions, get_testing_overrides

MODEL_LOG_FIELD_ORDER = [
    # General info
    "model_name",
    "_pass_finished",
    "logging_mode",
    "_all_layers_logged",
    "_all_layers_saved",
    "keep_unsaved_layers",
    "current_function_call_barcode",
    "random_seed_used",
    "detach_saved_tensors",
    "output_device",
    "save_function_args",
    "num_context_lines",
    "save_gradients",
    "has_saved_gradients",
    "activation_postfunc",
    "mark_input_output_distances",
    # Model structure info
    "model_is_recurrent",
    "model_max_recurrent_loops",
    "model_is_branching",
    "model_has_conditional_branching",
    # Layer tracking logs
    "layer_list",
    "layer_dict_main_keys",
    "layer_dict_all_keys",
    "layer_logs",
    "layer_labels",
    "layer_labels_no_pass",
    "layer_labels_w_pass",
    "layer_num_passes",
    "_raw_layer_dict",
    "_raw_layer_labels_list",
    "_layer_nums_to_save",
    "_layer_counter",
    "num_operations",
    "_raw_layer_type_counter",
    "_unsaved_layers_lookup_keys",
    # Mapping from raw to final layer labels:
    "_raw_to_final_layer_labels",
    "_final_to_raw_layer_labels",
    "_lookup_keys_to_layer_num_dict",
    "_layer_num_to_lookup_keys_dict",
    # Special layers
    "input_layers",
    "output_layers",
    "buffer_layers",
    "buffer_num_passes",
    "internally_initialized_layers",
    "_layers_where_internal_branches_merge_with_input",
    "internally_terminated_layers",
    "internally_terminated_bool_layers",
    "conditional_branch_edges",
    "layers_computed_with_params",
    "equivalent_operations",
    "layers_with_saved_activations",
    "unlogged_layers",
    "layers_with_saved_gradients",
    "orphan_layers",
    # Tensor info:
    "num_tensors_total",
    "tensor_fsize_total",
    "tensor_fsize_total_nice",
    "num_tensors_saved",
    "tensor_fsize_saved",
    "tensor_fsize_saved_nice",
    # Param info
    "param_logs",
    "total_param_tensors",
    "total_param_layers",
    "total_params",
    "total_params_trainable",
    "total_params_frozen",
    "total_params_fsize",
    "total_params_fsize_nice",
    # Time elapsed
    "pass_start_time",
    "pass_end_time",
    "elapsed_time_setup",
    "elapsed_time_forward_pass",
    "elapsed_time_cleanup",
    "elapsed_time_total",
    "elapsed_time_function_calls",
    "elapsed_time_torchlens_logging",
]

LAYER_PASS_LOG_FIELD_ORDER = [
    # General info
    "layer_label",
    "tensor_label_raw",
    "layer_label_raw",
    "operation_num",
    "realtime_tensor_num",
    "source_model_log",
    "_pass_finished",
    # Other labeling info
    "layer_label_short",
    "layer_label_w_pass",
    "layer_label_w_pass_short",
    "layer_label_no_pass",
    "layer_label_no_pass_short",
    "layer_type",
    "layer_type_num",
    "layer_total_num",
    "pass_num",
    "layer_passes_total",
    "lookup_keys",
    # Saved tensor info
    "tensor_contents",
    "has_saved_activations",
    "output_device",
    "activation_postfunc",
    "detach_saved_tensor",
    "function_args_saved",
    "creation_args",
    "creation_kwargs",
    "tensor_shape",
    "tensor_dtype",
    "tensor_fsize",
    "tensor_fsize_nice",
    # Child tensor variation tracking
    "has_child_tensor_variations",
    "children_tensor_versions",
    # Saved gradient info
    "grad_contents",
    "save_gradients",
    "has_saved_grad",
    "grad_shape",
    "grad_dtype",
    "grad_fsize",
    "grad_fsize_nice",
    # Function call info
    "func_applied",
    "func_applied_name",
    "func_call_stack",
    "func_time_elapsed",
    "flops_forward",
    "flops_backward",
    "func_rng_states",
    "func_autocast_state",
    "func_argnames",
    "num_func_args_total",
    "num_position_args",
    "num_keyword_args",
    "func_position_args_non_tensor",
    "func_keyword_args_non_tensor",
    "func_all_args_non_tensor",
    "function_is_inplace",
    "gradfunc",
    "is_part_of_iterable_output",
    "iterable_output_index",
    # Param info
    "computed_with_params",
    "parent_params",
    "parent_param_barcodes",
    "parent_param_passes",
    "parent_param_logs",
    "num_param_tensors",
    "parent_param_shapes",
    "num_params_total",
    "num_params_trainable",
    "num_params_frozen",
    "parent_params_fsize",
    "parent_params_fsize_nice",
    # Corresponding layer info
    "operation_equivalence_type",
    "equivalent_operations",
    "same_layer_operations",
    # Graph info
    "parent_layers",
    "has_parents",
    "parent_layer_arg_locs",
    "orig_ancestors",
    "child_layers",
    "has_children",
    "sibling_layers",
    "has_siblings",
    "spouse_layers",
    "has_spouses",
    "is_input_layer",
    "has_input_ancestor",
    "input_ancestors",
    "min_distance_from_input",
    "max_distance_from_input",
    "is_output_layer",
    "is_output_parent",
    "is_last_output_layer",
    "is_output_ancestor",
    "output_descendents",
    "input_output_address",
    "min_distance_from_output",
    "max_distance_from_output",
    "is_buffer_layer",
    "buffer_address",
    "buffer_pass",
    "buffer_parent",
    "initialized_inside_model",
    "has_internally_initialized_ancestor",
    "internally_initialized_parents",
    "internally_initialized_ancestors",
    "terminated_inside_model",
    # Conditional info
    "is_terminal_bool_layer",
    "is_atomic_bool_layer",
    "atomic_bool_val",
    "in_cond_branch",
    "cond_branch_start_children",
    # Module info
    "is_computed_inside_submodule",
    "containing_module_origin",
    "containing_modules_origin_nested",
    "module_nesting_depth",
    "modules_entered",
    "module_passes_entered",
    "modules_entered_argnames",
    "is_submodule_input",
    "modules_exited",
    "module_passes_exited",
    "is_submodule_output",
    "is_bottom_level_submodule_output",
    "bottom_level_submodule_pass_exited",
    "module_entry_exit_threads_inputs",
    "module_entry_exit_thread_output",
]

# Backward-compatible alias (will be removed in a future version)
TENSOR_LOG_FIELD_ORDER = LAYER_PASS_LOG_FIELD_ORDER

LAYER_LOG_FIELD_ORDER = [
    # Identity & labeling
    "layer_label",
    "layer_label_short",
    "layer_type",
    "layer_type_num",
    "layer_total_num",
    "num_passes",
    "source_model_log",
    # Function identity
    "func_applied",
    "func_applied_name",
    "function_is_inplace",
    "gradfunc",
    "func_argnames",
    "num_func_args_total",
    "num_position_args",
    "num_keyword_args",
    "is_part_of_iterable_output",
    "iterable_output_index",
    # Tensor type (representative from first pass)
    "tensor_shape",
    "tensor_dtype",
    "tensor_fsize",
    "tensor_fsize_nice",
    # Config
    "output_device",
    "activation_postfunc",
    "detach_saved_tensor",
    "save_gradients",
    # FLOPs
    "flops_forward",
    "flops_backward",
    # Param identity
    "computed_with_params",
    "parent_param_barcodes",
    "parent_param_logs",
    "num_param_tensors",
    "parent_param_shapes",
    "num_params_total",
    "num_params_trainable",
    "num_params_frozen",
    "parent_params_fsize",
    "parent_params_fsize_nice",
    # Equivalence
    "operation_equivalence_type",
    "equivalent_operations",
    # Special flags
    "is_input_layer",
    "is_output_layer",
    "is_last_output_layer",
    "is_buffer_layer",
    "buffer_address",
    "buffer_parent",
    "initialized_inside_model",
    "terminated_inside_model",
    "is_terminal_bool_layer",
    "is_atomic_bool_layer",
    "atomic_bool_val",
    # Module (static containment)
    "is_computed_inside_submodule",
    "containing_module_origin",
    "containing_modules_origin_nested",
    "module_nesting_depth",
    # Pass management
    "passes",
    "pass_labels",
]

FUNC_CALL_LOCATION_FIELD_ORDER = [
    "file",
    "line_number",
    "func_name",
    "func_signature",
    "func_docstring",
    "call_line",
    "code_context",
    "code_context_str",
    "code_context_labeled",
    "num_context_lines",
]

PARAM_LOG_FIELD_ORDER = [
    "address",
    "name",
    "shape",
    "dtype",
    "num_params",
    "fsize",
    "fsize_nice",
    "trainable",
    "is_quantized",
    "has_optimizer",
    "module_address",
    "module_type",
    "barcode",
    "num_passes",
    "layer_log_entries",
    "linked_params",
    "has_grad",
    "grad_shape",
    "grad_dtype",
    "grad_fsize",
    "grad_fsize_nice",
]

MODULE_PASS_LOG_FIELD_ORDER = [
    "module_address",
    "pass_num",
    "pass_label",
    "layers",
    "num_layers",
    "input_layers",
    "output_layers",
    "forward_args",
    "forward_kwargs",
    "call_parent",
    "call_children",
]

MODULE_LOG_FIELD_ORDER = [
    # Identity
    "address",
    "all_addresses",
    "name",
    "module_class_name",
    # Source info
    "source_file",
    "source_line",
    "class_docstring",
    "init_signature",
    "init_docstring",
    "forward_signature",
    "forward_docstring",
    # Hierarchy — address-based
    "address_parent",
    "address_children",
    "address_depth",
    # Hierarchy — call-based
    "call_parent",
    "call_children",
    "nesting_depth",
    # Pass info
    "num_passes",
    "passes",
    "pass_labels",
    # Layers
    "all_layers",
    "num_layers",
    # Parameters
    "params",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "params_fsize",
    "params_fsize_nice",
    "requires_grad",
    # Buffers
    "buffer_layers",
    # Module state
    "training_mode",
    "has_forward_hooks",
    "has_backward_hooks",
    "extra_attributes",
    "methods",
]

# Taken from https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions
IGNORED_FUNCS = [
    ("torch", "load"),
    ("torch", "as_tensor"),
    ("torch", "from_numpy"),
    ("torch", "tensor"),
    ("torch", "arange"),
    ("torch", "as_strided"),
    ("torch", "bartlett_window"),
    ("torch", "blackman_window"),
    ("torch", "cudnn_affine_grid_generator"),
    ("torch", "cudnn_batch_norm"),
    ("torch", "cudnn_convolution"),
    ("torch", "cudnn_convolution_transpose"),
    ("torch", "cudnn_convolution_relu"),
    ("torch", "cudnn_convolution_add_relu"),
    ("torch", "cudnn_grid_sampler"),
    ("torch", "cudnn_is_acceptable"),
    ("torch", "eye"),
    ("torch.fft", "fftfreq"),
    ("torch.fft", "rfftfreq"),
    ("torch", "from_file"),
    ("torch", "full"),
    ("torch", "fill_"),
    ("torch", "hamming_window"),
    ("torch", "hann_window"),
    ("torch", "kaiser_window"),
    ("torch", "linspace"),
    ("torch", "logspace"),
    ("torch", "mkldnn_adaptive_avg_pool2d"),
    ("torch", "mkldnn_convolution"),
    ("torch", "mkldnn_max_pool2d"),
    ("torch", "mkldnn_max_pool3d"),
    ("torch", "mkldnn_linear_backward_weights"),
    ("torch", "normal"),
    ("torch", "ones"),
    ("torch", "rand"),
    ("torch", "randn"),
    ("torch", "randint"),
    ("torch", "randperm"),
    ("torch", "range"),
    ("torch", "scalar_tensor"),
    ("torch", "sparse_coo_tensor"),
    ("torch", "_sparse_csr_tensor"),
    ("torch", "tril_indices"),
    ("torch", "triu_indices"),
    ("torch", "vander"),
    ("torch", "zeros"),
    ("torch.nn.functional", "upsample"),
    ("torch.nn.functional", "upsample_bilinear"),
    ("torch.nn.functional", "upsample_nearest"),
    ("torch.nn.functional", "handle_torch_function"),
    ("torch.nn.functional", "sigmoid"),
    ("torch.nn.functional", "hardsigmoid"),
    ("torch.nn.functional", "tanh"),
    ("torch.nn.init", "calculate_gain"),
    ("torch.nn.init", "uniform"),
    ("torch.nn.init", "normal"),
    ("torch.nn.init", "constant"),
    ("torch.nn.init", "eye"),
    ("torch.nn.init", "dirac"),
    ("torch.nn.init", "xavier_uniform"),
    ("torch.nn.init", "xavier_normal"),
    ("torch.nn.init", "kaiming_uniform"),
    ("torch.nn.init", "kaiming_normal"),
    ("torch.nn.init", "orthogonal"),
    ("torch.nn.init", "sparse"),
    ("torch.nn.functional", "hardswish"),
    ("torch.Tensor", "__delitem__"),
    ("torch.Tensor", "__iter__"),
    ("torch.Tensor", "__init_subclass__"),
    ("torch.Tensor", "__torch_function__"),
    ("torch.Tensor", "__new__"),
    ("torch.Tensor", "__subclasshook__"),
    ("torch.Tensor", "as_subclass"),
    ("torch.Tensor", "reinforce"),
    ("torch.Tensor", "new"),
    ("torch.Tensor", "new_tensor"),
    ("torch.Tensor", "new_empty"),
    ("torch.Tensor", "new_empty_strided"),
    ("torch.Tensor", "new_zeros"),
    ("torch.Tensor", "new_ones"),
    ("torch.Tensor", "new_full"),
    ("torch.Tensor", "_make_subclass"),
    ("torch.Tensor", "solve"),
    ("torch.Tensor", "unflatten"),
    ("torch.Tensor", "real"),
    ("torch.Tensor", "imag"),
    ("torch.Tensor", "T"),
    ("torch.Tensor", "mT"),
    ("torch.Tensor", "H"),
]


@functools.lru_cache(None)
def _get_torch_overridable_functions() -> List:
    """Return a list of (namespace_str, func_name) pairs for all torch functions
    that can be overridden via ``__torch_function__``.

    Crawls the standard torch namespaces (torch, torch.Tensor, torch.nn.functional,
    etc.) and collects every callable that is not explicitly ignored by PyTorch's
    override machinery.  The result is cached via ``lru_cache`` so the crawl only
    runs once per process.  The returned list is used to build ``ORIG_TORCH_FUNCS``,
    which drives the one-time decoration performed by ``decorate_all_once()``.
    """
    func_names = []
    tested_namespaces = [
        ("torch", torch, torch.__all__ + dir(torch._C._VariableFunctions)),
        ("torch._VF", torch._VF, dir(torch._C._VariableFunctions)),
        ("torch.functional", torch.functional, torch.functional.__all__),
        ("torch.nn.functional", torch.nn.functional, dir(torch.nn.functional)),
        ("torch.nn.init", torch.nn.init, dir(torch.nn.init)),
        ("torch.Tensor", torch.Tensor, dir(torch.Tensor)),
        ("torch.linalg", torch.linalg, dir(torch.linalg)),
        ("torch.fft", torch.fft, dir(torch.fft)),
    ]
    if hasattr(torch, "special"):
        tested_namespaces.append(("torch.special", torch.special, dir(torch.special)))
    for namespace_str, namespace, ns_funcs in tested_namespaces:
        for func_name in ns_funcs:
            ignore = False
            # ignore private functions or functions that are deleted in torch.__init__
            if namespace is not torch.Tensor:
                if func_name.startswith("__"):
                    continue
                elif (
                    func_name[0].isupper()
                ):  # Skip class-like exports (e.g. torch.Tensor, torch.Size) — only wrap functions
                    ignore = True
                elif func_name == "unique_dim":
                    continue
            else:
                if func_name == "__weakref__":
                    continue
            func = getattr(namespace, func_name)
            # Skip methods inherited from object (e.g. __hash__, __repr__ on Tensor)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                continue
            # ignore re-exported modules
            if isinstance(func, types.ModuleType):
                continue
            # ignore __future__ imports
            if isinstance(func, getattr(__future__, "_Feature")):
                continue

            if not callable(func) and hasattr(func, "__get__"):
                if ignore:
                    continue
                if func.__get__ in get_ignored_functions():
                    msg = (
                        "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                        "but still has an explicit override"
                    )
                    assert func.__get__ not in get_testing_overrides(), msg.format(
                        namespace, func.__name__
                    )
                    continue
                else:
                    func_names.append((f"{namespace_str}.{func_name}", "__get__"))
                    continue

            if not callable(func):
                continue

            if ignore:
                continue

            # cannot be overridden by __torch_function__
            if func in get_ignored_functions():
                msg = (
                    "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                    "but still has an explicit override"
                )
                assert func not in get_testing_overrides(), msg.format(namespace, func.__name__)
                continue
            func_names.append((f"{namespace_str}", func_name))
    return func_names


TORCHVISION_FUNCS = [
    ("torch.ops.torchvision.nms", "_op"),
    ("torch.ops.torchvision.deform_conv2d", "_op"),
    ("torch.ops.torchvision.ps_roi_align", "_op"),
    ("torch.ops.torchvision.ps_roi_pool", "_op"),
    ("torch.ops.torchvision.roi_align", "_op"),
    ("torch.ops.torchvision.roi_pool", "_op"),
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    OVERRIDABLE_FUNCS = _get_torch_overridable_functions()
ORIG_TORCH_FUNCS = OVERRIDABLE_FUNCS + IGNORED_FUNCS

try:
    import torchvision

    ORIG_TORCH_FUNCS += TORCHVISION_FUNCS
except ModuleNotFoundError:
    pass
