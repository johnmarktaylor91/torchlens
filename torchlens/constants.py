"""Shared constants: field-order tuples and function discovery for TorchLens.

**FIELD_ORDER lists** define the *canonical* set of fields for each data class
(ModelLog, LayerPassLog, LayerLog, etc.).  They serve two purposes:

1. **Canonical ordering** — __repr__, iteration, and serialization use these lists
   to present fields in a consistent, human-readable order.
2. **Completeness contract** — every attribute a data class exposes must appear in
   its FIELD_ORDER.  When adding a new field to a class, you MUST also add it here
   (and vice versa).  These lists are NOT used for filtering; they define the full
   set of fields.

**Function discovery** (bottom of this module) builds ``ORIG_TORCH_FUNCS``, the
master list of ``(namespace_str, func_name)`` pairs that ``decorate_all_once()``
uses to permanently wrap every torch function at import time.
"""

import __future__
import functools
import types
from typing import List
import warnings

import torch
from torch.overrides import get_ignored_functions, get_testing_overrides

# ---------------------------------------------------------------------------
# Field-order definitions
# ---------------------------------------------------------------------------
# Each list defines the complete, ordered set of fields for its data class.
# The order here controls display order in __repr__ and similar outputs.

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
    "save_source_context",
    "save_rng_states",
    "detect_loops",
    "verbose",
    "has_gradients",
    "activation_postfunc",
    "mark_input_output_distances",
    # Model structure info (is_recurrent, max_recurrent_loops,
    # is_branching, has_conditional_branching are computed @properties)
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
    "conditional_then_edges",
    "conditional_elif_edges",
    "conditional_else_edges",
    "conditional_events",
    "conditional_arm_edges",
    "conditional_edge_passes",
    "layers_with_params",
    "equivalent_operations",
    "layers_with_saved_activations",
    "unlogged_layers",
    "layers_with_saved_gradients",
    "orphan_layers",
    # Tensor info:
    "total_activation_memory",
    "num_tensors_saved",
    "saved_activation_memory",
    # Param info
    "param_logs",
    "total_param_tensors",
    "total_param_layers",
    "total_params",
    "total_params_trainable",
    "total_params_frozen",
    "total_params_memory",
    # Time elapsed
    "pass_start_time",
    "pass_end_time",
    "time_setup",
    "time_forward_pass",
    "time_cleanup",
    "time_function_calls",
]

LAYER_PASS_LOG_FIELD_ORDER = [
    # Per-pass data for a single layer execution.  One LayerPassLog exists for
    # each (layer, pass_num) pair.  Fields capture the tensor produced, the
    # function that created it, graph connectivity, module context, and more.
    #
    # General info
    "layer_label",
    "tensor_label_raw",
    "layer_label_raw",
    "operation_num",
    "creation_order",
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
    "num_passes",
    "lookup_keys",
    # Saved tensor info
    "activation",
    "has_saved_activations",
    "output_device",
    "activation_postfunc",
    "detach_saved_tensor",
    "args_captured",
    "captured_args",
    "captured_kwargs",
    "tensor_shape",
    "tensor_dtype",
    "tensor_memory",
    # Child tensor variation tracking
    "has_child_tensor_variations",
    "children_tensor_versions",
    # Saved gradient info
    "gradient",
    "save_gradients",
    "has_gradient",
    "grad_shape",
    "grad_dtype",
    "grad_memory",
    # Function call info
    "func_applied",
    "func_name",
    "func_call_stack",
    "func_time",
    "flops_forward",
    "flops_backward",
    "func_rng_states",
    "func_autocast_state",
    "func_argnames",
    "num_args",
    "num_positional_args",
    "num_keyword_args",
    "func_positional_args_non_tensor",
    "func_kwargs_non_tensor",
    "func_non_tensor_args",
    "func_is_inplace",
    "grad_fn_name",
    "is_part_of_iterable_output",
    "iterable_output_index",
    # Param info
    "parent_params",
    "parent_param_barcodes",
    "parent_param_passes",
    "parent_param_logs",
    "parent_param_shapes",
    "num_params_total",
    "num_params_trainable",
    "num_params_frozen",
    "params_memory",
    # Corresponding layer info
    "operation_equivalence_type",
    "equivalent_operations",
    "recurrent_group",
    # Graph info
    "parent_layers",
    "parent_layer_arg_locs",
    "root_ancestors",
    "child_layers",
    "has_children",
    "is_input_layer",
    "has_input_ancestor",
    "input_ancestors",
    "min_distance_from_input",
    "max_distance_from_input",
    "is_output_layer",
    "feeds_output",
    "is_final_output",
    "is_output_ancestor",
    "output_descendants",
    "io_role",
    "min_distance_from_output",
    "max_distance_from_output",
    "is_buffer_layer",
    "buffer_address",
    "buffer_pass",
    "buffer_parent",
    "is_internally_initialized",
    "has_internally_initialized_ancestor",
    "internally_initialized_parents",
    "internally_initialized_ancestors",
    "is_internally_terminated",
    # Conditional info
    "is_terminal_bool_layer",
    "bool_is_branch",
    "bool_context_kind",
    "bool_wrapper_kind",
    "bool_conditional_id",
    "is_scalar_bool",
    "scalar_bool_value",
    "in_cond_branch",
    "conditional_branch_stack",
    "conditional_branch_depth",
    "cond_branch_start_children",
    "cond_branch_then_children",
    "cond_branch_elif_children",
    "cond_branch_else_children",
    "cond_branch_children_by_cond",
    # Module info
    "containing_module",
    "containing_modules",
    "modules_entered",
    "module_passes_entered",
    "modules_entered_argnames",
    "modules_exited",
    "module_passes_exited",
    "is_submodule_output",
    "is_leaf_module_output",
    "leaf_module_pass",
    "module_entry_exit_threads_inputs",
    "module_entry_exit_thread_output",
    # Function config
    "func_config",
]

# Backward-compatible alias — LayerPassLog was formerly called TensorLog.
TENSOR_LOG_FIELD_ORDER = LAYER_PASS_LOG_FIELD_ORDER

LAYER_LOG_FIELD_ORDER = [
    # Aggregate view of a layer across all its passes.  One LayerLog per
    # unique layer; it delegates per-pass queries to its child LayerPassLogs.
    #
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
    "func_name",
    "func_is_inplace",
    "grad_fn_name",
    "func_argnames",
    "num_args",
    "num_positional_args",
    "num_keyword_args",
    "is_part_of_iterable_output",
    "iterable_output_index",
    # Tensor type (representative from first pass)
    "tensor_shape",
    "tensor_dtype",
    "tensor_memory",
    # Config
    "output_device",
    "activation_postfunc",
    "detach_saved_tensor",
    "save_gradients",
    # FLOPs
    "flops_forward",
    "flops_backward",
    # Param identity
    "parent_param_barcodes",
    "parent_param_logs",
    "parent_param_shapes",
    "num_params_total",
    "num_params_trainable",
    "num_params_frozen",
    "params_memory",
    # Equivalence
    "operation_equivalence_type",
    "equivalent_operations",
    # Special flags
    "is_input_layer",
    "is_output_layer",
    "is_final_output",
    "is_buffer_layer",
    "buffer_address",
    "buffer_parent",
    "is_internally_initialized",
    "is_internally_terminated",
    "is_terminal_bool_layer",
    "is_scalar_bool",
    "scalar_bool_value",
    "conditional_branch_stacks",
    "conditional_branch_stack_passes",
    "cond_branch_children_by_cond",
    "cond_branch_start_children",
    "cond_branch_then_children",
    "cond_branch_elif_children",
    "cond_branch_else_children",
    # Module (static containment)
    "containing_module",
    "containing_modules",
    # Function config
    "func_config",
    # Pass management
    "passes",
    "pass_labels",
]

# Metadata about where in user code a function was called (file, line, context).
# FuncCallLocation has lazy properties — source loaded on first access via linecache.
FUNC_CALL_LOCATION_FIELD_ORDER = [
    "file",
    "line_number",
    "func_name",
    "code_firstlineno",
    "code_qualname",
    "col_offset",
    "source_loading_enabled",
    "func_signature",
    "func_docstring",
    "call_line",
    "code_context",
    "source_context",
    "code_context_labeled",
    "num_context_lines",
]

# Per-parameter metadata (one ParamLog per nn.Parameter in the model).
PARAM_LOG_FIELD_ORDER = [
    "address",
    "name",
    "shape",
    "dtype",
    "num_params",
    "memory",
    "memory_str",
    "trainable",
    "is_quantized",
    "has_optimizer",
    "module_address",
    "module_type",
    "barcode",
    "num_passes",
    "used_by_layers",
    "linked_params",
    "has_grad",
    "grad_shape",
    "grad_dtype",
    "grad_memory",
    "grad_memory_str",
]

# Per-buffer metadata exported by BufferAccessor (one row per buffer address).
BUFFER_LOG_FIELD_ORDER = [
    "buffer_address",
    "name",
    "module_address",
    "buffer_pass",
    "layer_label",
    "pass_num",
    "tensor_shape",
    "tensor_dtype",
    "tensor_memory",
    "tensor_memory_str",
    "has_saved_activations",
    "has_gradient",
    "grad_shape",
    "grad_dtype",
    "grad_memory",
    "grad_memory_str",
    "buffer_parent",
    "containing_module",
    "containing_modules",
]

# Per-pass module execution data (one ModulePassLog per forward call to a module).
MODULE_PASS_LOG_FIELD_ORDER = [
    "module_address",
    "all_module_addresses",
    "is_shared_module",
    "pass_num",
    "pass_label",
    "layers",
    "num_layers",
    "input_layers",
    "output_layers",
    "forward_args_summary",
    "forward_kwargs_summary",
    "forward_args",
    "forward_kwargs",
    "call_parent",
    "call_children",
]

# Aggregate module metadata (one ModuleLog per unique nn.Module in the model).
MODULE_LOG_FIELD_ORDER = [
    # Identity
    "address",
    "all_addresses",
    "is_shared",
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
    "params_memory",
    "requires_grad",
    # Buffers
    "buffer_layers",
    # Module state
    "is_training",
    "has_forward_hooks",
    "has_backward_hooks",
    "extra_attributes",
    "methods",
]

# ---------------------------------------------------------------------------
# Function discovery for decoration
# ---------------------------------------------------------------------------
# IMPORTANT: The name "IGNORED_FUNCS" is a historical misnomer.  These are
# functions that PyTorch's __torch_function__ override mechanism *ignores*
# (i.e., they cannot be intercepted via __torch_function__), but TorchLens
# still decorates most of them.  They are listed separately because
# _get_torch_overridable_functions() intentionally skips them, so we must
# add them back into ORIG_TORCH_FUNCS explicitly.
#
# Source: https://pytorch.org/docs/stable/_modules/torch/overrides.html#get_ignored_functions
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
    ignored_funcs_set = get_ignored_functions()
    testing_overrides_set = get_testing_overrides()
    func_names = []  # accumulates (namespace_str, func_name) pairs
    # Each entry: (dotted namespace string, namespace object, list of attr names to inspect).
    # torch._VF mirrors torch._C._VariableFunctions — both are crawled so decoration
    # can patch both the public and internal references to the same underlying C++ functions.
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

            # Descriptors (properties, slots) — not directly callable but have __get__.
            # These are wrapped via their __get__ method so attribute access is intercepted.
            if not callable(func) and hasattr(func, "__get__"):
                if ignore:
                    continue
                if func.__get__ in ignored_funcs_set:
                    msg = (
                        "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                        "but still has an explicit override"
                    )
                    assert func.__get__ not in testing_overrides_set, msg.format(
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
            if func in ignored_funcs_set:
                msg = (
                    "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                    "but still has an explicit override"
                )
                assert func not in testing_overrides_set, msg.format(namespace, func.__name__)
                continue
            func_names.append((f"{namespace_str}", func_name))
    return func_names


# Torchvision ops accessed via torch.ops — decorated only if torchvision is installed.
TORCHVISION_FUNCS = [
    ("torch.ops.torchvision.nms", "_op"),
    ("torch.ops.torchvision.deform_conv2d", "_op"),
    ("torch.ops.torchvision.ps_roi_align", "_op"),
    ("torch.ops.torchvision.ps_roi_pool", "_op"),
    ("torch.ops.torchvision.roi_align", "_op"),
    ("torch.ops.torchvision.roi_pool", "_op"),
]

# Build the master function list at module load time.  Warnings are suppressed
# because some torch namespaces emit deprecation warnings during introspection.
# ORIG_TORCH_FUNCS = overridable functions + "ignored" functions (which we still
# decorate) + optional torchvision ops.  This is the complete list fed to
# decorate_all_once() in decoration/torch_funcs.py.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    OVERRIDABLE_FUNCS = _get_torch_overridable_functions()
ORIG_TORCH_FUNCS = OVERRIDABLE_FUNCS + IGNORED_FUNCS

try:
    import torchvision

    ORIG_TORCH_FUNCS += TORCHVISION_FUNCS
except ModuleNotFoundError:
    pass
