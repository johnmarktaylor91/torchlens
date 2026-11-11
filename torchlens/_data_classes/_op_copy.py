"""Selective-depth Op clones (``Op._copy_with_shallow_fields``) and the output-node field sets.

Moved out of ``_op_save`` (which re-exports the copy). The output-node copy itself is
``_op_output_copy._copy_for_output``; it reads the field sets declared here.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch

from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._quantities import Bytes, Duration, Flops
from .._trace_core.alias_registry import register_shared_cells
from .._trace_core.op_store import row_clone_scope
from .._utils._record_group_table import op_attr

if TYPE_CHECKING:
    from .op import Op

_DEEPCOPY_IMMUTABLE_TYPES = (
    str,
    bytes,
    int,
    float,
    bool,
    type(None),
    torch.dtype,
    torch.device,
    Bytes,
    Duration,
    Flops,
)


_OUTPUT_NODE_REPLACED_FIELDS = frozenset(
    {
        "_edge_uses",
        "_label_raw",
        "_layer_label_raw",
        "_param_barcodes",
        "_param_logs",
        "activation_memory",
        "annotations",
        "arg_names",
        "atomic_module_call",
        "autograd_memory",
        "bytes_delta_at_call",
        "bytes_peak_at_call",
        "children",
        "code_context",
        "container_path",
        "container_spec",
        "dropped_edge_tensor_args",
        "dtype",
        "equivalence_class",
        "equivalent_ops",
        "func",
        "func_config",
        "func_duration",
        "func_name",
        "func_non_tensor_args",
        "func_rng_states",
        "grad_fn_class_name",
        "has_children",
        "has_out_variations",
        "has_output_descendant",
        "input_to_module_calls",
        "internal_source_parents",
        "is_edited",
        "edits",
        "io_role",
        "is_atomic_module",
        "is_buffer",
        "is_final_output",
        "is_input",
        "is_internal_source",
        "is_module_output",
        "is_output",
        "is_transform",
        "type",
        "module",
        "module_call_stack",
        "modules",
        "non_tensor_kwargs",
        "non_tensor_pos_args",
        "num_args_total",
        "num_autograd_tensors",
        "num_kwargs",
        "num_params",
        "num_params_frozen",
        "num_params_trainable",
        "num_passes",
        "num_pos_args",
        "out",
        "out_versions_by_child",
        "output_descendants",
        "output_of_module_calls",
        "output_of_modules",
        "param_memory",
        "param_shapes",
        "parent_arg_positions",
        "parent_param_ops",
        "parent_params",
        "parents",
        "pass_number",
        "raw_index",
        "recurrent_ops",
        "saved_args",
        "saved_kwargs",
        "shape",
        "transform_chain",
        "transform_config",
        "transform_fn_name",
        "transform_fn_qualname",
        "transform_fn_source",
        "transform_kind",
        "transformed_activation_memory",
        "transformed_out",
        "transformed_out_dtype",
        "transformed_out_shape",
        "unattributed_tensor_args",
        "var_names",
    }
)


# Derived Op fields the constructor discards: ``Op.__init__`` seeds each as ``None`` and
# keeps no cell for it, and every read recomputes it from the new row. Reading them from
# the parent during the output-node copy is dead work (about 0.3 ms per capture), so
# that copy skips them. ``arg_expressions`` is NOT here although it is discarded too:
# reading it caches the parsed source expressions on the PARENT op (the field table
# declares ``_arg_expressions_cache`` written by add_output_layers), and moving that
# parse to a later read could read a source file edited after capture.
_OUTPUT_COPY_UNREAD_DERIVED_FIELDS = frozenset(
    {
        "input_ops",
        "input_activations",
        "input_shapes",
        "input_dtypes",
        "input_memory",
        "num_inputs",
        "raw_label",
        "is_internally_initialized",
    }
)


def _copy_op_field_value(value: Any) -> Any:
    """Copy one Op field without deep-copy dispatch for immutable values.

    Parameters
    ----------
    value:
        Stored field value.

    Returns
    -------
    Any
        The original immutable value or an independent deep copy.
    """

    if isinstance(value, _DEEPCOPY_IMMUTABLE_TYPES):
        return value
    if isinstance(value, tuple) and all(
        isinstance(member, _DEEPCOPY_IMMUTABLE_TYPES) for member in value
    ):
        return value
    return copy.deepcopy(value)


def _copy_with_shallow_fields(
    op: Op,
    extra_shallow_fields: frozenset[str],
    *,
    _store: Any = None,
    unread: frozenset[str] = frozenset(),
) -> Op:
    """Clone this op while sharing fields the caller replaces immediately.

    Parameters
    ----------
    op:
        The op record this function reads and writes.
    extra_shallow_fields:
        Additional fields safe to share for this construction path.
    _store:
        Optional owning row store for the clone.
    unread:
        Derived fields the Op constructor discards and the caller does not read.

    Returns
    -------
    Op
        Selective-depth clone.
    """

    fields_dict = {}
    fields_not_to_deepcopy = {
        "func",
        "grad_fn_class_name",
        "grad_fn_handle",
        "source_trace",
        "func_rng_states",
        "saved_args",
        "saved_kwargs",
        "args_template",
        "kwargs_template",
        "parent_params",
        "out",
        "transformed_out",
        "transformed_grad",
        "out_versions_by_child",
        "container_spec",
    } | extra_shallow_fields

    # The whole-schema getattr loop is a ROW-CLONE read: under a combined step
    # audit these reads are tagged category (d) via the clone scope (legal on
    # row-creating steps, a finding elsewhere; one dict lookup when no audit is
    # armed). Clone WRITES bypass interception (adopt_row), so the shared
    # shallow cells are registered with the alias registry below.
    with row_clone_scope(object.__getattribute__(op, "_core")):
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            if field in unread:
                continue
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = _copy_op_field_value(getattr(op, op_attr(field), None))
            else:
                fields_dict[field] = getattr(op, op_attr(field), None)
    return register_shared_cells(op, type(op)(fields_dict, _store=_store))
