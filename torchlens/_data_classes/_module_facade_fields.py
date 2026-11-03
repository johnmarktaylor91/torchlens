"""Stored-field tables for the ``ModuleCall`` and ``Module`` record facades.

``module.py`` installs these with ``install_record_facade`` at import time.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The M8 facades: declared stored fields become row-cell descriptors; the
# instance dict keeps the store binding, user extras, and the handful of
# names whose access paths hardcode ``__dict__`` (the template/source-trace/
# facets slots poked directly by their properties).
# ---------------------------------------------------------------------------

_MODULE_CALL_STORED_FIELDS: tuple[str, ...] = (
    "address",
    "all_addresses",
    "cls",
    "class_name",
    "class_qualname",
    "call_index",
    "label",
    "ordinal_index",
    "ops",
    "input_layers",
    "output_layers",
    "input_ops",
    "output_ops",
    "output_structure",
    "output_paths",
    "forward_args",
    "forward_kwargs",
    "inputs_before_pre_hooks",
    "inputs_after_pre_hooks",
    "forward_pre_hook_effects",
    "forward_arg_names",
    "num_forward_pos_args",
    "num_forward_kwargs",
    "num_forward_args_total",
    "forward_args_summary",
    "forward_kwargs_summary",
    "forward_duration",
    "code_context",
    "module_call_stack",
    "call_parent",
    "call_children",
)

_MODULE_STORED_FIELDS: tuple[str, ...] = (
    "address",
    "all_addresses",
    "cls",
    "class_name",
    "class_qualname",
    "class_source_file",
    "class_source_line",
    "init_source_file",
    "init_source_line",
    "forward_source_file",
    "forward_source_line",
    "class_docstring",
    "init_signature",
    "init_docstring",
    "forward_signature",
    "forward_docstring",
    "address_parent",
    "address_children",
    "address_depth",
    "call_parent",
    "call_children",
    "call_depth",
    "num_calls",
    "ops",
    "call_labels",
    "layer_labels",
    "input_ops",
    "input_layers",
    "output_ops",
    "output_layers",
    "params",
    "num_params",
    "num_params_trainable",
    "num_params_frozen",
    "buffer_layers",
    "_buffer_accessor",
    "training",
    "forward_pre_hooks",
    "forward_hooks",
    "backward_pre_hooks",
    "backward_hooks",
    "full_backward_pre_hooks",
    "full_backward_hooks",
    "custom_attributes",
    "custom_methods",
)
