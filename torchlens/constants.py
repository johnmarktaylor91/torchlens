"""Shared constants: field-order tuples and function discovery for TorchLens.

**FIELD_ORDER lists** define the *canonical* set of fields for each data class
(Trace, OpLog, LayerLog, etc.).  They serve two purposes:

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
    "name",
    "model_name",
    "io_format_version",
    "_tracing_finished",
    "capture_mode",
    "_layers_logged",
    "_layers_saved",
    "keep_unsaved_layers",
    "keep_orphans",
    "intervention_ready",
    "capture_args_template",
    "raw_input",
    "_transform",
    "save_raw_input",
    "_current_func_barcode",
    "random_seed",
    "detach_saved_activations",
    "output_device",
    "train_mode",
    "module_filter",
    "emit_nvtx",
    "raise_on_nan",
    "trace_annotations",
    "report_values",
    "observer_spans",
    "manual_tensor_connections",
    "forward_source_file",
    "forward_source_line",
    "model_source_file",
    "model_source_line",
    "capture_cache_hit",
    "capture_cache_key",
    "capture_cache_path",
    "recording_kept",
    "streaming_pass_logs",
    "num_streamed_ops",
    "_out_hash_cache",
    "save_arg_values",
    "num_context_lines",
    "save_grads",
    "grads_to_save",
    "_grad_layer_nums_to_save",
    "grad_transform",
    "grad_transform_repr",
    "save_raw_grads",
    "save_code_context",
    "save_rng_states",
    "recurrence_detection",
    "verbose",
    "has_grads",
    "out_postfunc",
    "_out_transform_repr",
    "save_raw_outs",
    "input_annotations",
    "_source_code_blob",
    "_source_model_ref",
    "parent_run",
    "model_id",
    "model_class",
    "param_hash_quick",
    "param_hash_full",
    "input_id",
    "input_shape_hash",
    "mark_layer_depths",
    "graph_shape_hash",
    "_intervention_spec",
    "ledger",
    "last_run_ctx",
    "_has_direct_writes",
    "_warned_direct_write",
    "_warned_mutate_in_place",
    "_spec_revision",
    "_out_recipe_revision",
    "_append_sequence_id",
    "_last_hook_handle_ids",
    "run_state",
    "is_appended",
    "relationship_evidence",
    # Model structure info (is_recurrent, max_recurrent_loops,
    # is_branching, has_conditional_branching are computed @properties)
    # Layer tracking logs
    "layer_list",
    "layer_dict_main_keys",
    "layer_dict_all_keys",
    "layer_logs",
    "layer_labels",
    "op_labels",
    "layer_num_calls",
    "_raw_layer_dict",
    "_raw_layer_labels_list",
    "_layer_nums_to_save",
    "_layer_counter",
    "num_ops",
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
    "buffer_num_calls",
    "internal_source_ops",
    "_layers_where_internal_branches_merge_with_input",
    "internal_sink_ops",
    "internally_terminated_bool_ops",
    "conditional_branch_edges",
    "conditional_records",
    "conditional_arm_entry_edges",
    "conditional_edge_call_indices",
    "conditionals",
    "layers_with_params",
    "equivalent_ops",
    "ops_with_saved_outs",
    "unlogged_ops",
    "ops_with_saved_grads",
    "orphan_ops",
    # Tensor info:
    "total_out_memory",
    "total_gradient_memory",
    "autograd_saved_memory",
    "num_saved_ops",
    "saved_out_memory",
    "saved_gradient_memory",
    # Param info
    "param_logs",
    "num_param_tensors",
    "num_layers_with_params",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "param_memory",
    "total_param_gradient_memory",
    "forward_peak_memory",
    # Time elapsed
    "start_time",
    "end_time",
    "setup_duration",
    "forward_duration",
    "cleanup_duration",
    "func_calls_duration",
    # Backward pass
    "has_backward_pass",
    "grad_fn_logs",
    "grad_fn_order",
    "backward_root_grad_fn_id",
    "backward_num_calls",
    "backward_peak_memory",
    "backward_memory_backend",
]

LAYER_PASS_LOG_FIELD_ORDER = [
    # Per-pass data for a single layer execution.  One OpLog exists for
    # each (layer, call_index) pair.  Fields capture the tensor produced, the
    # function that created it, graph connectivity, module context, and more.
    #
    # General info
    "layer_label",
    "_label_raw",
    "_layer_label_raw",
    "compute_index",
    "capture_index",
    "source_trace",
    "_tracing_finished",
    "_construction_done",
    # Other labeling info
    "layer_label_short",
    "layer_label_w_pass",
    "layer_label_w_pass_short",
    "layer_label_no_pass",
    "layer_label_no_pass_short",
    "type",
    "type_index",
    "trace_index",
    "call_index",
    "num_calls",
    "lookup_keys",
    # Saved tensor info
    "out",
    "has_saved_outs",
    "output_device",
    "out_postfunc",
    "annotations",
    "interventions",
    "intervention_replaced",
    "detach_saved_activations",
    "has_saved_args",
    "saved_args",
    "saved_kwargs",
    "args_template",
    "kwargs_template",
    "shape",
    "transformed_out_shape",
    "dtype",
    "transformed_out_dtype",
    "memory",
    "transformed_out_memory",
    "bytes_delta_at_call",
    "bytes_peak_at_call",
    "transformed_out",
    "autograd_saved_memory",
    "num_autograd_saved_tensors",
    # Child tensor variation tracking
    "has_output_variations",
    "output_versions_per_child",
    # Saved grad info
    "grad",
    "transformed_grad",
    "save_grads",
    "has_grad",
    "grad_shape",
    "transformed_grad_shape",
    "grad_dtype",
    "transformed_grad_dtype",
    "grad_memory",
    "transformed_grad_memory",
    # Function call info
    "func",
    "func_call_id",
    "func_name",
    "code_context",
    "func_duration",
    "flops_forward",
    "flops_backward",
    "func_rng_states",
    "func_autocast_state",
    "arg_names",
    "num_args_total",
    "num_pos_args",
    "num_kwargs",
    "non_tensor_pos_args",
    "non_tensor_kwargs",
    "func_non_tensor_args",
    "is_inplace",
    "grad_fn_name",
    "grad_fn_id",
    "grad_fn",
    "grad_fn_log",
    "is_part_of_iterable_output",
    "multi_output_index",
    "container_path",
    "container_spec",
    # Param info
    "parent_params",
    "_param_barcodes",
    "parent_param_ops",
    "_param_logs",
    "param_shapes",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "param_memory",
    # Corresponding layer info
    "equivalence_class",
    "equivalent_ops",
    "recurrent_ops",
    # Graph info
    "parents",
    "parent_arg_positions",
    "edge_uses",
    "root_ancestors",
    "children",
    "has_children",
    "is_input",
    "has_input_ancestor",
    "input_ancestors",
    "min_distance_from_input",
    "max_distance_from_input",
    "is_output",
    "is_output_parent",
    "is_final_output",
    "has_output_descendant",
    "output_descendants",
    "is_orphan",
    "io_role",
    "min_distance_from_output",
    "max_distance_from_output",
    "is_buffer",
    "buffer_address",
    "buffer_pass",
    "buffer_parent",
    "is_internal_source",
    "has_internal_source_ancestor",
    "internal_source_parents",
    "internal_source_ancestors",
    "is_internal_sink",
    # Conditional info
    "is_terminal_bool",
    "is_terminal_conditional_bool",
    "conditional_context_kind",
    "conditional_wrapper_kind",
    "terminal_conditional_id",
    "is_scalar_bool",
    "bool_value",
    "in_conditionals",
    "terminal_bool_for",
    "is_in_conditional_body",
    "conditional_branch_stack",
    "conditional_branch_depth",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_elif_children",
    "conditional_else_children",
    "conditional_arm_children",
    # Module info
    "module",
    "_address_normalized",
    "modules",
    "fx_qualpath",
    "fx_call_index",
    "modules_entered",
    "module_ops_entered",
    "module_entry_argnames",
    "output_of_modules",
    "output_of_module_calls",
    "is_submodule_output",
    "is_atomic_module_op",
    "atomic_module_call",
    "_module_boundary_threads_inputs",
    "_module_boundary_thread_output",
    # Function config
    "func_config",
]

# Backward-compatible alias — OpLog was formerly called TensorLog.
OP_LOG_FIELD_ORDER = LAYER_PASS_LOG_FIELD_ORDER
TENSOR_LOG_FIELD_ORDER = LAYER_PASS_LOG_FIELD_ORDER

LAYER_LOG_FIELD_ORDER = [
    # Aggregate view of a layer across all its ops.  One LayerLog per
    # unique layer; it delegates per-pass queries to its child OpLogs.
    #
    # Identity & labeling
    "layer_label",
    "layer_label_short",
    "layer_type",
    "type_index",
    "trace_index",
    "num_calls",
    "source_trace",
    # Function identity
    "func",
    "func_name",
    "is_inplace",
    "grad_fn_name",
    "grad_fn_id",
    "grad_fn",
    "grad_fn_log",
    "arg_names",
    "num_args_total",
    "num_pos_args",
    "num_kwargs",
    "is_part_of_iterable_output",
    "multi_output_index",
    # Tensor type (representative from first pass)
    "shape",
    "transformed_out_shape",
    "dtype",
    "transformed_out_dtype",
    "memory",
    "transformed_out_memory",
    "transformed_out",
    "autograd_saved_memory",
    "num_autograd_saved_tensors",
    # Config
    "output_device",
    "out_postfunc",
    "annotations",
    "intervention_replaced",
    "detach_saved_activations",
    "save_grads",
    "transformed_grad",
    "transformed_grad_shape",
    "transformed_grad_dtype",
    "transformed_grad_memory",
    # FLOPs
    "flops_forward",
    "flops_backward",
    # Param identity
    "_param_barcodes",
    "_param_logs",
    "param_shapes",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "param_memory",
    # Equivalence
    "equivalence_class",
    "equivalent_ops",
    # Special flags
    "is_input",
    "is_output",
    "is_final_output",
    "is_buffer",
    "buffer_address",
    "buffer_parent",
    "is_internal_source",
    "is_internal_sink",
    "is_terminal_bool",
    "is_scalar_bool",
    "bool_value",
    "in_conditionals",
    "terminal_bool_for",
    "conditional_role_stacks",
    "conditional_branch_stack_ops",
    "conditional_arm_children",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_elif_children",
    "conditional_else_children",
    # Module (static containment)
    "module",
    "modules",
    # Function config
    "func_config",
    # Pass management
    "ops",
    "call_labels",
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
    "num_calls",
    "used_by_layers",
    "co_parent_params",
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
    "address",
    "buffer_pass",
    "layer_label",
    "call_index",
    "shape",
    "dtype",
    "memory",
    "memory_str",
    "has_saved_outs",
    "has_grad",
    "grad_shape",
    "grad_dtype",
    "grad_memory",
    "grad_memory_str",
    "buffer_parent",
    "module",
    "modules",
]

# Per-pass module execution data (one ModuleCallLog per forward call to a module).
MODULE_PASS_LOG_FIELD_ORDER = [
    "address",
    "all_addresses",
    "has_multiple_addresses",
    "call_index",
    "call_label",
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
    "has_multiple_addresses",
    "name",
    "cls",
    "class_name",
    "class_qualname",
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
    "call_depth",
    # Pass info
    "num_calls",
    "ops",
    "call_labels",
    # Layers
    "layers",
    "num_layers",
    # Parameters
    "params",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "param_memory",
    "has_trainable_params",
    # Buffers
    "buffer_layers",
    # Module state
    "is_train_mode",
    "forward_pre_hook_info",
    "forward_hook_info",
    "backward_pre_hook_info",
    "backward_hook_info",
    "full_backward_pre_hook_info",
    "full_backward_hook_info",
    "has_forward_hooks",
    "has_backward_hooks",
    "custom_attributes",
    "custom_methods",
]

GRAD_FN_PASS_LOG_FIELD_ORDER = [
    "call_index",
    "grad_inputs",
    "grad_outputs",
    "time_started",
    "time_finished",
]

GRAD_FN_LOG_FIELD_ORDER = [
    "grad_fn_id",
    "name",
    "label",
    "grad_fn_type",
    "grad_fn_type_num",
    "grad_fn_total_num",
    "module_path",
    "is_custom",
    "has_op",
    "op",
    "next_grad_fn_ids",
    "ops",
    "num_calls",
    "call_labels",
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
def _get_torch_overridable_functions() -> list[tuple[str, str]]:
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
            # Skip custom_methods inherited from object (e.g. __hash__, __repr__ on Tensor)
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
