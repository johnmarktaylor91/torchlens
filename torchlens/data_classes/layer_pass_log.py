"""LayerPassLog: per-operation metadata for a single invocation of a layer.

Each LayerPassLog records everything about one tensor operation in the
forward pass: the output tensor itself, the function that produced it,
its parents/children in the computation graph, module containment,
parameter usage, timing, RNG state, and more.

For recurrent models, the same "layer" may execute multiple times; each
execution is a separate LayerPassLog with a distinct ``pass_num``.  The
aggregate view across passes is provided by :class:`LayerLog`.

Field categories (matching the LAYER_PASS_LOG_FIELD_ORDER in constants.py):

1. **General info** - raw/final labels, operation numbering, back-reference
   to the owning ModelLog.
2. **Label info** - human-readable labels in various formats (with/without
   pass qualifier, short form, etc.).
3. **Saved tensor info** - the tensor contents, shape, dtype, size, device
   transfer settings, activation postfunc, and function arguments.
4. **Child tensor variations** - tracks per-child input values for
   validation replay (``children_tensor_versions`` stores RAW values
   because validation compares against ``captured_args``).
5. **Gradient info** - gradient tensor and metadata (stored as a bare
   reference via ``log_tensor_grad``, not deep-copied).
6. **Function call info** - the applied function, call stack, timing,
   FLOPs, RNG state, arg metadata, grad_fn, inplace flag.
7. **Param info** - which parameters were used, their shapes and sizes.
8. **Equivalence info** - loop-detection equivalence type and groups.
9. **Graph info** - parent/child/sibling/spouse edges, input/output
   ancestry, distances, buffer/internal-init status.
10. **Conditional info** - boolean branching metadata.
11. **Module info** - module entry/exit tracking, nesting depth,
    bottom-level submodule output status.
"""

import copy
import weakref
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Tuple, Union

import torch

from .._io import (
    FieldPolicy,
    IO_FORMAT_VERSION,
    TorchLensIOError,
    default_fill_state,
    read_io_format_version,
)
from .._errors import TorchLensPostfuncError
from .._run_state import RunState
from .._training_validation import _NON_GRAD_DTYPES, TrainingModeConfigError
from ..constants import LAYER_PASS_LOG_FIELD_ORDER
from ..intervention.types import LAYER_PASS_LOG_FORK_POLICY
from ..intervention.errors import DirectActivationWriteWarning
from .._state import pause_logging
from ..utils.tensor_utils import (
    concatenate_batch_tensors,
    get_tensor_memory_amount,
    print_override,
    safe_copy,
    safe_to,
)
from ..utils.display import human_readable_size
from ..utils.display import tensor_stats_summary

_LAYER_PASS_LOG_FIELD_ORDER_SET = frozenset(LAYER_PASS_LOG_FIELD_ORDER)
_DIRECT_WRITE_GUARDED_FIELDS = frozenset(
    {
        "activation",
        "transformed_activation",
        "gradient",
        "transformed_gradient",
        "intervention_log",
    }
)
_LAYER_PASS_LOG_DEFAULT_FILL: dict[str, Any] = {
    "_source_model_log_ref": None,
    "parent_layer_log": None,
    "activation_ref": None,
    "gradient_ref": None,
    "_pending_blob_id": None,
    "_pending_transformed_activation_blob_id": None,
    "_pending_gradient_blob_id": None,
    "_pending_transformed_gradient_blob_id": None,
    "extra_data": {},
    "autograd_saved_bytes": None,
    "autograd_saved_tensor_count": None,
    "transformed_activation": None,
    "transformed_activation_shape": None,
    "transformed_activation_dtype": None,
    "transformed_activation_memory": None,
    "transformed_gradient": None,
    "transformed_gradient_shape": None,
    "transformed_gradient_dtype": None,
    "transformed_gradient_memory": None,
    "func_call_id": None,
    "output_path": (),
    "intervention_log": [],
    "container_spec": None,
    "captured_arg_template": None,
    "captured_kwarg_template": None,
    "edge_uses": [],
    "module_address_normalized": None,
    "_construction_done": True,
}
_LAYER_PASS_LOG_DEFAULT_FILL = {
    **{field_name: None for field_name in LAYER_PASS_LOG_FIELD_ORDER},
    **_LAYER_PASS_LOG_DEFAULT_FILL,
}


def _recursive_safe_copy(val: Any) -> Any:
    """Deep-copy nested structures, cloning tensors instead of using copy.deepcopy (#44)."""
    if isinstance(val, torch.Tensor):
        return safe_copy(val)
    elif isinstance(val, (list, tuple)):
        return type(val)(_recursive_safe_copy(v) for v in val)
    elif isinstance(val, dict):
        return {k: _recursive_safe_copy(v) for k, v in val.items()}
    return safe_copy(val)


def _tensor_shape_or_none(value: Any) -> tuple[int, ...] | None:
    """Return a tensor shape tuple, or ``None`` for non-tensor values."""

    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _tensor_dtype_or_none(value: Any) -> torch.dtype | None:
    """Return a tensor dtype, or ``None`` for non-tensor values."""

    return value.dtype if isinstance(value, torch.Tensor) else None


def _tensor_memory_or_none(value: Any) -> int | None:
    """Return tensor memory in bytes, or ``None`` for non-tensor values."""

    return get_tensor_memory_amount(value) if isinstance(value, torch.Tensor) else None


if TYPE_CHECKING:
    from .._io.lazy import LazyActivationRef
    from .func_call_location import FuncCallLocation
    from .layer_log import LayerLog
    from .param_log import ParamLog
    from .model_log import ModelLog


class LayerPassLog:
    """Metadata for a single tensor operation (one pass of one layer).

    Constructed from a dict whose keys must exactly match
    ``LAYER_PASS_LOG_FIELD_ORDER`` (enforced at init time).  Every
    attribute is set explicitly (not via a loop) so that IDE
    autocompletion works.

    Notable design points:

    * ``_pass_finished`` mirrors the owning ModelLog's flag. Methods like
      ``__str__`` branch on it to show raw vs final labels.
    * ``source_model_log`` is a direct reference to the owning ModelLog.
      This creates a circular reference (ModelLog -> layer_list -> entry ->
      source_model_log -> ModelLog) that is broken by ``cleanup()``.
    * ``parent_layer_log`` is a back-reference to the aggregate LayerLog
      that owns this pass.  It is set *outside* fields_dict during
      ``_build_layer_logs`` and is intentionally absent from FIELD_ORDER.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "tensor_label_raw": FieldPolicy.KEEP,
        "layer_label_raw": FieldPolicy.KEEP,
        "operation_num": FieldPolicy.KEEP,
        "creation_order": FieldPolicy.KEEP,
        "source_model_log": FieldPolicy.DROP,
        "_source_model_log_ref": FieldPolicy.WEAKREF_STRIP,
        "_pass_finished": FieldPolicy.KEEP,
        "_construction_done": FieldPolicy.DROP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "layer_label_w_pass": FieldPolicy.KEEP,
        "layer_label_w_pass_short": FieldPolicy.KEEP,
        "layer_label_no_pass": FieldPolicy.KEEP,
        "layer_label_no_pass_short": FieldPolicy.KEEP,
        "layer_type": FieldPolicy.KEEP,
        "layer_type_num": FieldPolicy.KEEP,
        "layer_total_num": FieldPolicy.KEEP,
        "pass_num": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "lookup_keys": FieldPolicy.KEEP,
        "activation": FieldPolicy.BLOB,
        "transformed_activation": FieldPolicy.BLOB,
        "has_saved_activations": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "activation_postfunc": FieldPolicy.DROP,
        "extra_data": FieldPolicy.KEEP,
        "intervention_log": FieldPolicy.DROP,
        "detach_saved_tensor": FieldPolicy.KEEP,
        "args_captured": FieldPolicy.KEEP,
        "captured_args": FieldPolicy.BLOB_RECURSIVE,
        "captured_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "captured_arg_template": FieldPolicy.DROP,
        "captured_kwarg_template": FieldPolicy.DROP,
        "tensor_shape": FieldPolicy.KEEP,
        "transformed_activation_shape": FieldPolicy.KEEP,
        "tensor_dtype": FieldPolicy.KEEP,
        "transformed_activation_dtype": FieldPolicy.KEEP,
        "tensor_memory": FieldPolicy.KEEP,
        "transformed_activation_memory": FieldPolicy.KEEP,
        "autograd_saved_bytes": FieldPolicy.KEEP,
        "autograd_saved_tensor_count": FieldPolicy.KEEP,
        "has_child_tensor_variations": FieldPolicy.KEEP,
        "children_tensor_versions": FieldPolicy.BLOB_RECURSIVE,
        "gradient": FieldPolicy.BLOB,
        "transformed_gradient": FieldPolicy.BLOB,
        "save_gradients": FieldPolicy.KEEP,
        "has_gradient": FieldPolicy.KEEP,
        "grad_shape": FieldPolicy.KEEP,
        "transformed_gradient_shape": FieldPolicy.KEEP,
        "grad_dtype": FieldPolicy.KEEP,
        "transformed_gradient_dtype": FieldPolicy.KEEP,
        "grad_memory": FieldPolicy.KEEP,
        "transformed_gradient_memory": FieldPolicy.KEEP,
        "func_applied": FieldPolicy.DROP,
        "func_call_id": FieldPolicy.KEEP,
        "func_name": FieldPolicy.KEEP,
        "func_call_stack": FieldPolicy.KEEP,
        "func_time": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "func_rng_states": FieldPolicy.BLOB_RECURSIVE,
        "func_autocast_state": FieldPolicy.KEEP,
        "func_argnames": FieldPolicy.KEEP,
        "num_args": FieldPolicy.KEEP,
        "num_positional_args": FieldPolicy.KEEP,
        "num_keyword_args": FieldPolicy.KEEP,
        "func_positional_args_non_tensor": FieldPolicy.KEEP,
        "func_kwargs_non_tensor": FieldPolicy.KEEP,
        "func_non_tensor_args": FieldPolicy.KEEP,
        "func_is_inplace": FieldPolicy.KEEP,
        "grad_fn_name": FieldPolicy.KEEP,
        "grad_fn_id": FieldPolicy.KEEP,
        "grad_fn_object": FieldPolicy.DROP,
        "corresponding_grad_fn": FieldPolicy.DROP,
        "is_part_of_iterable_output": FieldPolicy.KEEP,
        "iterable_output_index": FieldPolicy.KEEP,
        "output_path": FieldPolicy.KEEP,
        "container_spec": FieldPolicy.KEEP,
        "parent_params": FieldPolicy.KEEP,
        "parent_param_barcodes": FieldPolicy.KEEP,
        "parent_param_passes": FieldPolicy.KEEP,
        "parent_param_logs": FieldPolicy.KEEP,
        "parent_param_shapes": FieldPolicy.KEEP,
        "num_params_total": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "params_memory": FieldPolicy.KEEP,
        "operation_equivalence_type": FieldPolicy.KEEP,
        "equivalent_operations": FieldPolicy.KEEP,
        "recurrent_group": FieldPolicy.KEEP,
        "parent_layers": FieldPolicy.KEEP,
        "parent_layer_arg_locs": FieldPolicy.KEEP,
        "edge_uses": FieldPolicy.KEEP,
        "root_ancestors": FieldPolicy.KEEP,
        "child_layers": FieldPolicy.KEEP,
        "has_children": FieldPolicy.KEEP,
        "is_input_layer": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "input_ancestors": FieldPolicy.KEEP,
        "min_distance_from_input": FieldPolicy.KEEP,
        "max_distance_from_input": FieldPolicy.KEEP,
        "is_output_layer": FieldPolicy.KEEP,
        "feeds_output": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "is_output_ancestor": FieldPolicy.KEEP,
        "output_descendants": FieldPolicy.KEEP,
        "min_distance_from_output": FieldPolicy.KEEP,
        "max_distance_from_output": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "is_buffer_layer": FieldPolicy.KEEP,
        "buffer_address": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "buffer_parent": FieldPolicy.KEEP,
        "is_internally_initialized": FieldPolicy.KEEP,
        "has_internally_initialized_ancestor": FieldPolicy.KEEP,
        "internally_initialized_parents": FieldPolicy.KEEP,
        "internally_initialized_ancestors": FieldPolicy.KEEP,
        "is_internally_terminated": FieldPolicy.KEEP,
        "is_terminal_bool_layer": FieldPolicy.KEEP,
        "bool_is_branch": FieldPolicy.KEEP,
        "bool_context_kind": FieldPolicy.KEEP,
        "bool_wrapper_kind": FieldPolicy.KEEP,
        "bool_conditional_id": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "scalar_bool_value": FieldPolicy.KEEP,
        "in_cond_branch": FieldPolicy.KEEP,
        "conditional_branch_stack": FieldPolicy.KEEP,
        "conditional_branch_depth": FieldPolicy.KEEP,
        "cond_branch_start_children": FieldPolicy.KEEP,
        "cond_branch_then_children": FieldPolicy.KEEP,
        "cond_branch_elif_children": FieldPolicy.KEEP,
        "cond_branch_else_children": FieldPolicy.KEEP,
        "cond_branch_children_by_cond": FieldPolicy.KEEP,
        "containing_module": FieldPolicy.KEEP,
        "module_address_normalized": FieldPolicy.KEEP,
        "containing_modules": FieldPolicy.KEEP,
        "modules_entered": FieldPolicy.KEEP,
        "modules_entered_argnames": FieldPolicy.KEEP,
        "module_passes_entered": FieldPolicy.KEEP,
        "modules_exited": FieldPolicy.KEEP,
        "module_passes_exited": FieldPolicy.KEEP,
        "is_submodule_output": FieldPolicy.KEEP,
        "is_leaf_module_output": FieldPolicy.KEEP,
        "leaf_module_pass": FieldPolicy.KEEP,
        "module_entry_exit_threads_inputs": FieldPolicy.KEEP,
        "module_entry_exit_thread_output": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "activation_ref": FieldPolicy.DROP,
        "gradient_ref": FieldPolicy.DROP,
        "_pending_blob_id": FieldPolicy.DROP,
        "_pending_transformed_activation_blob_id": FieldPolicy.DROP,
        "_pending_gradient_blob_id": FieldPolicy.DROP,
        "_pending_transformed_gradient_blob_id": FieldPolicy.DROP,
        "parent_layer_log": FieldPolicy.DROP,
    }
    FORK_POLICY = LAYER_PASS_LOG_FORK_POLICY
    DEFAULT_FILL_STATE = _LAYER_PASS_LOG_DEFAULT_FILL
    _construction_done: bool = False

    def __getattribute__(self, name: str) -> Any:
        """Materialize lazy gradients when the public ``gradient`` field is accessed."""

        if name == "gradient":
            state = object.__getattribute__(self, "__dict__")
            gradient = state.get("gradient")
            if gradient is None and state.get("gradient_ref") is not None:
                return object.__getattribute__(self, "materialize_gradient")()
            return gradient
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Mark owning logs dirty when user code directly writes guarded fields.

        Parameters
        ----------
        name:
            Attribute being written.
        value:
            New attribute value.
        """

        construction_done = self.__dict__.get("_construction_done", False)
        if construction_done and name in _DIRECT_WRITE_GUARDED_FIELDS:
            owner = self.__dict__.get("_source_model_log_ref")
            model_log = owner() if owner is not None else None
            if model_log is not None:
                object.__setattr__(model_log, "_has_direct_writes", True)
                object.__setattr__(model_log, "run_state", RunState.DIRECT_WRITE_DIRTY)
                if not getattr(model_log, "_warned_direct_write", False):
                    warnings.warn(
                        "DirectActivationWriteWarning: direct LayerPassLog activation writes "
                        "are not recipe edits; replay/rerun propagation will overlay them.",
                        DirectActivationWriteWarning,
                        stacklevel=2,
                    )
                    object.__setattr__(model_log, "_warned_direct_write", True)
        object.__setattr__(self, name, value)

    def _internal_set(self, attr: str, value: Any) -> None:
        """Set an attribute without marking the owner dirty.

        Parameters
        ----------
        attr:
            Attribute name to set.
        value:
            Value to assign.
        """

        object.__setattr__(self, attr, value)

    def _append_tensor_from(self, other: "LayerPassLog", field_name: str) -> None:
        """Append one tensor field from another pass along batch dimension 0.

        Parameters
        ----------
        other:
            New chunk pass with a compatible tensor field.
        field_name:
            Tensor attribute name to concatenate.
        """

        current_value = getattr(self, field_name)
        other_value = getattr(other, field_name)
        if isinstance(current_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            self._internal_set(field_name, concatenate_batch_tensors(current_value, other_value))

    def __init__(self, fields_dict: Dict):
        """Initialise from a complete fields dictionary.

        Args:
            fields_dict: Dict with values for all fields defined in
                ``LAYER_PASS_LOG_FIELD_ORDER``.  Missing or extra keys
                raise ``ValueError``.
        """
        # Attributes are set explicitly (not via loop) for IDE autocompletion.
        object.__setattr__(self, "_construction_done", False)

        # Validate that fields_dict has exactly the expected keys:
        if "module_address_normalized" not in fields_dict:
            fields_dict["module_address_normalized"] = None
        fields_dict_key_set = set(fields_dict.keys())
        if fields_dict_key_set != _LAYER_PASS_LOG_FIELD_ORDER_SET:
            error_str = "Error initializing LayerPassLog:"
            missing_fields = _LAYER_PASS_LOG_FIELD_ORDER_SET - fields_dict_key_set
            extra_fields = fields_dict_key_set - _LAYER_PASS_LOG_FIELD_ORDER_SET
            if len(missing_fields) > 0:
                error_str += f"\n\t- Missing fields {', '.join(missing_fields)}"
            if len(extra_fields) > 0:
                error_str += f"\n\t- Extra fields {', '.join(extra_fields)}"
            raise ValueError(error_str)

        # General info:
        self.tensor_label_raw = fields_dict["tensor_label_raw"]
        self.layer_label_raw = fields_dict["layer_label_raw"]
        self.operation_num = fields_dict["operation_num"]
        self.creation_order = fields_dict["creation_order"]
        # Store as weakref to break circular reference (ModelLog -> layer_list -> entry -> ModelLog).
        _sml = fields_dict["source_model_log"]
        self._source_model_log_ref = weakref.ref(_sml) if _sml is not None else None
        self._pass_finished = fields_dict["_pass_finished"]

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
        self.num_passes = fields_dict["num_passes"]
        self.lookup_keys = fields_dict["lookup_keys"]

        # Saved tensor info:
        self.activation = fields_dict["activation"]
        self.transformed_activation = fields_dict["transformed_activation"]
        self.has_saved_activations = fields_dict["has_saved_activations"]
        self.output_device = fields_dict["output_device"]
        self.activation_postfunc = fields_dict["activation_postfunc"]
        self.extra_data: Dict[str, Any] = fields_dict["extra_data"]
        self.intervention_log = fields_dict["intervention_log"]
        self.detach_saved_tensor = fields_dict["detach_saved_tensor"]
        self.args_captured = fields_dict["args_captured"]
        self.captured_args = fields_dict["captured_args"]
        self.captured_kwargs = fields_dict["captured_kwargs"]
        self.captured_arg_template = fields_dict["captured_arg_template"]
        self.captured_kwarg_template = fields_dict["captured_kwarg_template"]
        self.tensor_shape = fields_dict["tensor_shape"]
        self.transformed_activation_shape = fields_dict["transformed_activation_shape"]
        self.tensor_dtype = fields_dict["tensor_dtype"]
        self.transformed_activation_dtype = fields_dict["transformed_activation_dtype"]
        self.tensor_memory = fields_dict["tensor_memory"]
        self.transformed_activation_memory = fields_dict["transformed_activation_memory"]
        self.autograd_saved_bytes: Optional[int] = fields_dict["autograd_saved_bytes"]
        self.autograd_saved_tensor_count: Optional[int] = fields_dict["autograd_saved_tensor_count"]

        # Child tensor variation tracking - stores the raw tensor values that
        # each child operation received as input.  Must store RAW values (not
        # postprocessed) because validation compares these against captured_args.
        self.has_child_tensor_variations = fields_dict["has_child_tensor_variations"]
        self.children_tensor_versions = fields_dict["children_tensor_versions"]

        # Saved gradient info - gradient is stored as a bare clone (not deep-copied)
        # via log_tensor_grad().  gradient is populated by a backward hook.
        self.gradient = fields_dict["gradient"]
        self.transformed_gradient = fields_dict["transformed_gradient"]
        self.save_gradients = fields_dict["save_gradients"]
        self.has_gradient = fields_dict["has_gradient"]
        self.grad_shape = fields_dict["grad_shape"]
        self.transformed_gradient_shape = fields_dict["transformed_gradient_shape"]
        self.grad_dtype = fields_dict["grad_dtype"]
        self.transformed_gradient_dtype = fields_dict["transformed_gradient_dtype"]
        self.grad_memory = fields_dict["grad_memory"]
        self.transformed_gradient_memory = fields_dict["transformed_gradient_memory"]

        # Function call info:
        self.func_applied = fields_dict["func_applied"]
        self.func_call_id = fields_dict["func_call_id"]
        self.func_name = fields_dict["func_name"]
        self.func_call_stack: List["FuncCallLocation"] = fields_dict["func_call_stack"]
        self.func_time = fields_dict["func_time"]
        self.flops_forward = fields_dict["flops_forward"]
        self.flops_backward = fields_dict["flops_backward"]
        self.func_rng_states = fields_dict["func_rng_states"]
        self.func_autocast_state = fields_dict["func_autocast_state"]
        self.func_argnames = fields_dict["func_argnames"]
        self.num_args = fields_dict["num_args"]
        self.num_positional_args = fields_dict["num_positional_args"]
        self.num_keyword_args = fields_dict["num_keyword_args"]
        self.func_positional_args_non_tensor = fields_dict["func_positional_args_non_tensor"]
        self.func_kwargs_non_tensor = fields_dict["func_kwargs_non_tensor"]
        self.func_non_tensor_args = fields_dict["func_non_tensor_args"]
        self.func_is_inplace = fields_dict["func_is_inplace"]
        self.grad_fn_name = fields_dict["grad_fn_name"]
        self.grad_fn_id = fields_dict["grad_fn_id"]
        self.grad_fn_object = fields_dict["grad_fn_object"]
        self.corresponding_grad_fn = fields_dict["corresponding_grad_fn"]
        self.is_part_of_iterable_output = fields_dict["is_part_of_iterable_output"]
        self.iterable_output_index = fields_dict["iterable_output_index"]
        self.output_path = fields_dict["output_path"]
        self.container_spec = fields_dict["container_spec"]

        # Param info:
        self.parent_params = fields_dict["parent_params"]
        self.parent_param_barcodes = fields_dict["parent_param_barcodes"]
        self.parent_param_passes = fields_dict["parent_param_passes"]
        self.parent_param_logs: List["ParamLog"] = fields_dict["parent_param_logs"]
        self.parent_param_shapes = fields_dict["parent_param_shapes"]
        self.num_params_total = fields_dict["num_params_total"]
        self.num_params_trainable = fields_dict["num_params_trainable"]
        self.num_params_frozen = fields_dict["num_params_frozen"]
        self.params_memory = fields_dict["params_memory"]

        # Loop-detection equivalence info:
        # operation_equivalence_type groups structurally identical operations
        # (same func + same param barcodes).  equivalent_operations holds a
        # DIRECT reference to the ModelLog-level set for this type.
        # recurrent_group is populated by loop_detection.py for layers
        # that are different passes of the same recurrent layer.
        self.operation_equivalence_type = fields_dict["operation_equivalence_type"]
        self.equivalent_operations = fields_dict["equivalent_operations"]
        self.recurrent_group = fields_dict["recurrent_group"]

        # Graph info:
        self.parent_layers = fields_dict["parent_layers"]
        self.parent_layer_arg_locs = fields_dict["parent_layer_arg_locs"]
        self.edge_uses = fields_dict["edge_uses"]
        self.root_ancestors = fields_dict["root_ancestors"]
        self.child_layers = fields_dict["child_layers"]
        self.has_children = fields_dict["has_children"]
        self.is_input_layer = fields_dict["is_input_layer"]
        self.has_input_ancestor = fields_dict["has_input_ancestor"]
        self.input_ancestors = fields_dict["input_ancestors"]
        self.min_distance_from_input = fields_dict["min_distance_from_input"]
        self.max_distance_from_input = fields_dict["max_distance_from_input"]
        self.is_output_layer = fields_dict["is_output_layer"]
        self.feeds_output = fields_dict["feeds_output"]
        self.is_final_output = fields_dict["is_final_output"]
        self.is_output_ancestor = fields_dict["is_output_ancestor"]
        self.output_descendants = fields_dict["output_descendants"]
        self.min_distance_from_output = fields_dict["min_distance_from_output"]
        self.max_distance_from_output = fields_dict["max_distance_from_output"]
        self.io_role = fields_dict["io_role"]
        self.is_buffer_layer = fields_dict["is_buffer_layer"]
        self.buffer_address = fields_dict["buffer_address"]
        self.buffer_pass = fields_dict["buffer_pass"]
        self.buffer_parent = fields_dict["buffer_parent"]
        self.is_internally_initialized = fields_dict["is_internally_initialized"]
        self.has_internally_initialized_ancestor = fields_dict[
            "has_internally_initialized_ancestor"
        ]
        self.internally_initialized_parents = fields_dict["internally_initialized_parents"]
        self.internally_initialized_ancestors = fields_dict["internally_initialized_ancestors"]
        self.is_internally_terminated = fields_dict["is_internally_terminated"]

        # Conditional info
        self.is_terminal_bool_layer = fields_dict["is_terminal_bool_layer"]
        self.bool_is_branch = fields_dict["bool_is_branch"]
        self.bool_context_kind = fields_dict["bool_context_kind"]
        self.bool_wrapper_kind = fields_dict["bool_wrapper_kind"]
        self.bool_conditional_id = fields_dict["bool_conditional_id"]
        self.is_scalar_bool = fields_dict["is_scalar_bool"]
        self.scalar_bool_value = fields_dict["scalar_bool_value"]
        self.in_cond_branch = fields_dict["in_cond_branch"]
        self.conditional_branch_stack = fields_dict["conditional_branch_stack"]
        self.conditional_branch_depth = fields_dict["conditional_branch_depth"]
        self.cond_branch_start_children = fields_dict["cond_branch_start_children"]
        self.cond_branch_then_children = fields_dict["cond_branch_then_children"]
        self.cond_branch_elif_children = fields_dict["cond_branch_elif_children"]
        self.cond_branch_else_children = fields_dict["cond_branch_else_children"]
        self.cond_branch_children_by_cond = fields_dict["cond_branch_children_by_cond"]

        # Module info
        self.containing_module = fields_dict["containing_module"]
        self.module_address_normalized = fields_dict["module_address_normalized"]
        self.containing_modules = fields_dict["containing_modules"]
        self.modules_entered = fields_dict["modules_entered"]
        self.modules_entered_argnames = fields_dict["modules_entered_argnames"]
        self.module_passes_entered = fields_dict["module_passes_entered"]
        self.modules_exited = fields_dict["modules_exited"]
        self.module_passes_exited = fields_dict["module_passes_exited"]
        self.is_submodule_output = fields_dict["is_submodule_output"]
        self.is_leaf_module_output = fields_dict["is_leaf_module_output"]
        self.leaf_module_pass = fields_dict["leaf_module_pass"]
        self.module_entry_exit_threads_inputs = fields_dict["module_entry_exit_threads_inputs"]
        self.module_entry_exit_thread_output = fields_dict["module_entry_exit_thread_output"]

        # Function config - lightweight hyperparameters always captured.
        self.func_config = fields_dict["func_config"]

        # Back-reference to the aggregate LayerLog that groups all passes of
        # this layer.  Set during postprocessing by _build_layer_logs - NOT
        # part of fields_dict or FIELD_ORDER (it's a structural link, not
        # captured data).
        self.activation_ref: Optional["LazyActivationRef"] = None
        self.gradient_ref: Optional["LazyActivationRef"] = None
        self._pending_blob_id: Optional[str] = None
        self._pending_transformed_activation_blob_id: Optional[str] = None
        self._pending_gradient_blob_id: Optional[str] = None
        self._pending_transformed_gradient_blob_id: Optional[str] = None
        self.parent_layer_log: Optional["LayerLog"] = None
        object.__setattr__(self, "_construction_done", True)

    @property
    def macs_forward(self) -> Optional[int]:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return self.flops_forward // 2 if self.flops_forward is not None else None

    @property
    def macs_backward(self) -> Optional[int]:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return self.flops_backward // 2 if self.flops_backward is not None else None

    @property
    def has_parents(self) -> bool:
        """Whether this layer has any parent layers."""
        return len(self.parent_layers) > 0

    @property
    def sibling_layers(self) -> list:
        """Layers sharing at least one parent (excluding output layers)."""
        ml = self.source_model_log
        if ml is None:
            return []
        my_label = self.layer_label if self._pass_finished else self.tensor_label_raw
        siblings = []
        seen = {my_label}
        for parent_label in self.parent_layers:
            parent = ml[parent_label]
            for child_label in parent.child_layers:
                if child_label not in seen:
                    seen.add(child_label)
                    child = ml[child_label]
                    if not child.is_output_layer:
                        siblings.append(child_label)
        return siblings

    @property
    def has_siblings(self) -> bool:
        """Whether this layer shares parents with other layers."""
        return len(self.sibling_layers) > 0

    @property
    def co_parent_layers(self) -> list:
        """Layers sharing at least one child (excluding output layers)."""
        ml = self.source_model_log
        if ml is None:
            return []
        my_label = self.layer_label if self._pass_finished else self.tensor_label_raw
        spouses = []
        seen = {my_label}
        for child_label in self.child_layers:
            child = ml[child_label]
            for parent_label in child.parent_layers:
                if parent_label not in seen:
                    seen.add(parent_label)
                    parent = ml[parent_label]
                    if not parent.is_output_layer:
                        spouses.append(parent_label)
        return spouses

    @property
    def has_co_parents(self) -> bool:
        """Whether this layer shares children with other layers."""
        return len(self.co_parent_layers) > 0

    @property
    def uses_params(self) -> bool:
        """Whether this operation used model parameters."""
        return len(self.parent_param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this operation."""
        return len(self.parent_param_barcodes)

    @property
    def is_computed_inside_submodule(self) -> bool:
        """Whether this operation was computed inside a submodule."""
        return self.containing_module is not None

    @property
    def module_nesting_depth(self) -> int:
        """Depth of module nesting for this operation."""
        return len(self.containing_modules)

    @property
    def is_submodule_input(self) -> bool:
        """Whether this operation is the first inside a submodule's forward()."""
        return len(self.modules_entered) > 0

    @property
    def tensor_memory_str(self) -> str:
        return human_readable_size(self.tensor_memory)

    @property
    def grad_memory_str(self) -> str:
        return human_readable_size(self.grad_memory)

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved activation."""

        return self.activation

    @property
    def passes(self) -> tuple["LayerPassLog", ...]:
        """Tuple containing this pass for aggregate-compatible iteration.

        Returns
        -------
        tuple[LayerPassLog, ...]
            Single-entry tuple containing this pass log.
        """

        return (self,)

    @property
    def params_memory_str(self) -> str:
        return human_readable_size(self.params_memory)

    @property
    def _streaming_label(self) -> str:
        """Best available label for sink/writer callbacks during or after postprocess.

        Returns
        -------
        str
            Pass-qualified label when available, otherwise the current layer label.
        """

        for candidate in (
            self.layer_label_w_pass,
            self.layer_label,
            self.layer_label_raw,
            self.tensor_label_raw,
        ):
            if candidate is not None:
                return str(candidate)
        return "<unknown>"

    @property
    def source_model_log(self) -> "ModelLog":
        """Back-reference to the owning ModelLog (stored as weakref)."""
        ref = self.__dict__.get("_source_model_log_ref")
        if ref is None:
            return None  # type: ignore[return-value]
        obj = ref()
        return obj  # type: ignore[return-value]

    @source_model_log.setter
    def source_model_log(self, value):
        self._source_model_log_ref = weakref.ref(value) if value is not None else None

    def materialize_activation(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Materialize this layer's saved activation from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized activation tensor.

        Raises
        ------
        TorchLensIOError
            If no activation ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> model_log = tl.load("demo_bundle", lazy=True)
        >>> tensor = model_log["linear_1_1"].materialize_activation()
        >>> tensor.shape
        torch.Size([2, 3])
        """

        if isinstance(self.activation, torch.Tensor):
            return self.activation
        if self.activation_ref is None:
            raise TorchLensIOError("no activation_ref to materialize from")
        self._internal_set("activation", self.activation_ref.materialize(map_location=map_location))
        return self.activation

    def materialize_gradient(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Materialize this layer's saved gradient from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized gradient tensor.

        Raises
        ------
        TorchLensIOError
            If no gradient ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> model_log = tl.load("demo_bundle", lazy=True)
        >>> grad = model_log["linear_1_1"].materialize_gradient()
        >>> grad.shape
        torch.Size([2, 3])
        """

        gradient = self.__dict__.get("gradient")
        if isinstance(gradient, torch.Tensor):
            return gradient
        if self.gradient_ref is None:
            raise TorchLensIOError("no gradient_ref to materialize from")
        self._internal_set("gradient", self.gradient_ref.materialize(map_location=map_location))
        return self.gradient

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = self.__dict__.copy()
        state["_source_model_log_ref"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        read_io_format_version(state, cls_name=type(self).__name__)
        default_fill_state(
            state,
            defaults=self.DEFAULT_FILL_STATE,
        )
        object.__setattr__(self, "_construction_done", False)
        state.pop("source_model_log", None)
        self.__dict__.update(state)
        object.__setattr__(self, "_construction_done", bool(state.get("_construction_done", True)))

    @property
    def activation_transform(self) -> Optional[Callable]:
        """Canonical activation transform callable used for this pass.

        Returns
        -------
        Optional[Callable]
            Transform callable, or ``None`` when activations are stored unchanged.
        """

        return self.activation_postfunc

    @activation_transform.setter
    def activation_transform(self, value: Optional[Callable]) -> None:
        """Set the canonical activation transform callable.

        Parameters
        ----------
        value:
            Transform callable, or ``None``.
        """

        self._internal_set("activation_postfunc", value)

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

    def copy(self):
        """Return a selective-depth copy of this entry.

        Most fields are ``copy.deepcopy``'d so the clone is fully independent.
        However, certain fields are shallow-copied (shared by reference) because:

        * ``func_applied``, ``grad_fn_name`` - function objects, immutable/shared.
        * ``source_model_log`` - must point to the same ModelLog instance.
        * ``func_rng_states`` - large state dicts, not mutated after capture.
        * ``captured_args``, ``captured_kwargs`` - may contain large tensors;
          deep-copying them is expensive and unnecessary.
        * ``parent_params`` - references to nn.Parameters, must stay shared.
        * ``activation``, ``children_tensor_versions`` - large tensors;
          shared references are safe since they're replaced (not mutated).

        Returns:
            A new LayerPassLog (or subclass) with the same field values.
        """
        fields_dict = {}
        fields_not_to_deepcopy = [
            "func_applied",
            "grad_fn_name",
            "grad_fn_object",
            "corresponding_grad_fn",
            "source_model_log",
            "func_rng_states",
            "captured_args",
            "captured_kwargs",
            "captured_arg_template",
            "captured_kwarg_template",
            "parent_params",
            "activation",
            "transformed_activation",
            "transformed_gradient",
            "children_tensor_versions",
        ]
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = copy.deepcopy(getattr(self, field, None))
            else:
                fields_dict[field] = getattr(self, field, None)
        copied_entry = type(self)(fields_dict)
        return copied_entry

    def save_tensor_data(
        self,
        t: torch.Tensor,
        t_args: Union[List, Tuple],
        t_kwargs: Dict,
        save_function_args: bool,
        activation_postfunc: Optional[Callable] = None,
    ) -> None:
        """Save the output tensor (and optionally args) for this operation.

        Flow:
        1. Clone the tensor via ``safe_copy`` (strips tl_ attributes to avoid
           logging the copy operation).
        2. Move to ``output_device`` if different from the tensor's current device.
        3. Apply ``activation_postfunc`` inside ``pause_logging()`` to prevent
           the postfunc's own tensor ops from being logged.
        4. Optionally deep-copy function args/kwargs via ``_recursive_safe_copy``.

        Args:
            t: The output tensor of the operation.
            t_args: Positional arguments passed to the operation.
            t_kwargs: Keyword arguments passed to the operation.
            save_function_args: Whether to deep-copy and store args/kwargs.
            activation_postfunc: Optional transform applied to the tensor
                before storing (e.g. detach, to-numpy, normalize).
        """
        model_log = self.source_model_log
        writer = getattr(model_log, "_activation_writer", None) if model_log is not None else None
        try:
            # Clone the tensor, optionally detaching from autograd graph.
            raw_activation = safe_copy(t, self.detach_saved_tensor)
            # Move to the user-requested output device if needed.
            if self.output_device not in [str(raw_activation.device), "same"]:
                raw_activation = safe_to(raw_activation, self.output_device)

            self.tensor_shape = tuple(raw_activation.shape)
            self.tensor_dtype = raw_activation.dtype
            self.tensor_memory = get_tensor_memory_amount(raw_activation)

            save_raw_activation = getattr(model_log, "save_raw_activation", True)
            store_raw = save_raw_activation or activation_postfunc is None
            self._internal_set("activation", raw_activation if store_raw else None)

            self._internal_set("transformed_activation", None)
            self.transformed_activation_shape = None
            self.transformed_activation_dtype = None
            self.transformed_activation_memory = None
            if activation_postfunc is not None:
                self._internal_set(
                    "transformed_activation",
                    self._apply_postfunc(
                        raw_activation,
                        activation_postfunc,
                        postfunc_kind="activation",
                        streaming_active=writer is not None,
                    ),
                )
                self._validate_train_mode_postfunc_output(
                    raw_activation,
                    self.transformed_activation,
                    postfunc_kind="activation",
                )
                self.transformed_activation_shape = _tensor_shape_or_none(
                    self.transformed_activation
                )
                self.transformed_activation_dtype = _tensor_dtype_or_none(
                    self.transformed_activation
                )
                self.transformed_activation_memory = _tensor_memory_or_none(
                    self.transformed_activation
                )
        except Exception as exc:
            if writer is not None:
                writer.abort(f"Failed while saving activation for {self._streaming_label}: {exc}")
                if isinstance(exc, TorchLensPostfuncError):
                    raise
                raise TorchLensIOError(
                    f"Streaming activation save failed for {self._streaming_label}."
                ) from exc
            raise

        self.has_saved_activations = True

        if model_log is not None:
            activation_sink = getattr(model_log, "_activation_sink", None)
            if activation_sink is not None and isinstance(self.activation, torch.Tensor):
                activation_sink(self._streaming_label, self.activation)

            if writer is not None and getattr(model_log, "_in_exhaustive_pass", False):
                self._stream_tensor_blob(
                    writer,
                    tensor_field="activation",
                    pending_field="_pending_blob_id",
                    kind="activation",
                )
                self._stream_tensor_blob(
                    writer,
                    tensor_field="transformed_activation",
                    pending_field="_pending_transformed_activation_blob_id",
                    kind="transformed_activation",
                )

        # Tensor args and kwargs:
        if save_function_args:
            self.args_captured = True
            self._internal_set("captured_args", [_recursive_safe_copy(arg) for arg in t_args])
            self._internal_set(
                "captured_kwargs",
                {k: _recursive_safe_copy(v) for k, v in t_kwargs.items()},
            )
        else:
            self._internal_set("captured_args", None)
            self._internal_set("captured_kwargs", None)

    def log_tensor_grad(self, grad: torch.Tensor) -> None:
        """Save the gradient tensor for this layer's output.

        Called by the backward hook registered during the forward pass.
        The gradient is ``detach().clone()``'d - a bare copy, not deep-copied -
        so it's independent of the autograd graph but cheap to store.

        Args:
            grad: The gradient tensor flowing back through this operation.
        """
        model_log = self.source_model_log
        raw_grad = grad
        self.grad_shape = tuple(raw_grad.shape)
        self.grad_dtype = raw_grad.dtype
        self.grad_memory = get_tensor_memory_amount(raw_grad)
        gradient_postfunc = getattr(model_log, "gradient_postfunc", None)
        self._internal_set("transformed_gradient", None)
        self.transformed_gradient_shape = None
        self.transformed_gradient_dtype = None
        self.transformed_gradient_memory = None
        writer = getattr(model_log, "_activation_writer", None) if model_log is not None else None
        if gradient_postfunc is not None:
            self._internal_set(
                "transformed_gradient",
                self._apply_postfunc(
                    raw_grad,
                    gradient_postfunc,
                    postfunc_kind="gradient",
                    streaming_active=writer is not None,
                ),
            )
            self._validate_train_mode_postfunc_output(
                raw_grad,
                self.transformed_gradient,
                postfunc_kind="gradient",
            )
            self.transformed_gradient_shape = _tensor_shape_or_none(self.transformed_gradient)
            self.transformed_gradient_dtype = _tensor_dtype_or_none(self.transformed_gradient)
            self.transformed_gradient_memory = _tensor_memory_or_none(self.transformed_gradient)

        save_raw_gradient = getattr(model_log, "save_raw_gradient", True)
        store_raw = save_raw_gradient or gradient_postfunc is None
        self._internal_set("gradient", raw_grad.detach().clone() if store_raw else None)
        self.has_gradient = True
        if writer is not None and getattr(model_log, "_defer_streaming_bundle_finalization", False):
            self._stream_tensor_blob(
                writer,
                tensor_field="gradient",
                pending_field="_pending_gradient_blob_id",
                kind="gradient",
            )
            self._stream_tensor_blob(
                writer,
                tensor_field="transformed_gradient",
                pending_field="_pending_transformed_gradient_blob_id",
                kind="transformed_gradient",
            )

    def _apply_postfunc(
        self,
        tensor: torch.Tensor,
        postfunc: Callable,
        *,
        postfunc_kind: str,
        streaming_active: bool,
    ) -> Any:
        """Apply a user postfunc with logging paused and rich error context."""

        try:
            with pause_logging():
                return postfunc(tensor)
        except Exception as exc:
            raise TorchLensPostfuncError(
                self._postfunc_error_message(
                    postfunc_kind=postfunc_kind,
                    tensor=tensor,
                    streaming_active=streaming_active,
                )
            ) from exc

    def _postfunc_error_message(
        self,
        *,
        postfunc_kind: str,
        tensor: torch.Tensor,
        streaming_active: bool,
    ) -> str:
        """Build context for an activation or gradient postfunc failure."""

        return (
            f"{postfunc_kind}_postfunc raised for layer {self._streaming_label} "
            f"(raw={self.layer_label_raw}, func={self.func_name}, "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"streaming_active={streaming_active})."
        )

    def _validate_train_mode_postfunc_output(
        self,
        raw_tensor: torch.Tensor,
        output: Any,
        *,
        postfunc_kind: str,
    ) -> None:
        """Validate differentiability requirements for train-mode postfunc outputs."""

        model_log = self.source_model_log
        if not getattr(model_log, "train_mode", False) or not raw_tensor.requires_grad:
            return
        if not isinstance(output, torch.Tensor):
            raise TrainingModeConfigError(
                f"{postfunc_kind}_postfunc must return a torch.Tensor while train_mode=True "
                f"for layer {self._streaming_label}."
            )
        if output.dtype in _NON_GRAD_DTYPES:
            raise TrainingModeConfigError(
                f"train_mode=True with non-grad dtype {output.dtype} on layer "
                f"{self._streaming_label}. Integer and bool dtypes cannot propagate gradients."
            )
        if output.grad_fn is None:
            raise TrainingModeConfigError(
                f"{postfunc_kind}_postfunc returned a tensor disconnected from the autograd "
                "graph (grad_fn is None) while train_mode=True. The transformed activation "
                "must remain differentiable."
            )

    def _stream_tensor_blob(
        self,
        writer: Any,
        *,
        tensor_field: str,
        pending_field: str,
        kind: str,
    ) -> None:
        """Stream one tensor field when present."""

        tensor = getattr(self, tensor_field)
        if tensor is None:
            return
        if not isinstance(tensor, torch.Tensor):
            if kind == "transformed_activation":
                message = (
                    "Streaming save requires activation_postfunc outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            elif kind == "transformed_gradient":
                message = (
                    "Streaming save requires gradient_postfunc outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            else:
                message = (
                    f"{tensor_field} expected a tensor for streaming, got {type(tensor).__name__}."
                )
            writer.abort(message)
            raise TorchLensIOError(message)
        blob_id = writer.next_blob_id()
        setattr(self, pending_field, blob_id)
        writer.write_blob(
            blob_id,
            tensor,
            kind=kind,
            label=self._streaming_label,
        )

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def get_child_layers(self):
        return [self.source_model_log[child_label] for child_label in self.child_layers]

    def get_parent_layers(self):
        return [self.source_model_log[parent_label] for parent_label in self.parent_layers]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this pass's saved activation.

        Parameters
        ----------
        method:
            Display method. ``"auto"`` chooses from tensor shape.
        **kwargs:
            Forwarded to the tensor display helper.

        Returns
        -------
        Any
            Matplotlib figure when plotting is available, otherwise a text
            fallback explaining why no plot was produced.
        """

        from ..viz._tensor_display import show_tensor

        return show_tensor(self, method=method, **kwargs)

    @property
    def params(self):
        """Access parameter metadata by address, short name, or index."""
        from .param_log import ParamAccessor

        param_dict = {pl.address: pl for pl in self.parent_param_logs}
        return ParamAccessor(param_dict)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self):
        if self._pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def _str_during_pass(self) -> str:
        """Return a human-readable summary of this tensor entry while the forward pass is still in progress."""
        s = f"Tensor {self.tensor_label_raw} (layer {self.layer_label_raw}) (PASS NOT FINISHED):"
        s += f"\n\tPass: {self.pass_num}"
        s += f"\n\tTensor info: shape {self.tensor_shape}, dtype {self.tensor_dtype}"
        s += f"\n\tComputed from params: {self.uses_params}"
        s += f"\n\tComputed in modules: {self.containing_modules}"
        s += f"\n\tOutput of modules: {self.module_passes_exited}"
        if self.is_leaf_module_output:
            s += " (bottom-level submodule output)"
        else:
            s += " (not bottom-level submodule output)"
        s += "\n\tFamily info:"
        s += f"\n\t\tParents: {self.parent_layers}"
        s += f"\n\t\tChildren: {self.child_layers}"
        s += f"\n\t\tSpouses: {self.co_parent_layers}"
        s += f"\n\t\tSiblings: {self.sibling_layers}"
        s += (
            f"\n\t\tOriginal Ancestors: {self.root_ancestors} "
            f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        )
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internally_initialized_ancestors}"
        s += (
            f"\n\t\tOutput Descendents: {self.output_descendants} "
            f"(min dist {self.min_distance_from_output} nodes, max dist {self.max_distance_from_output} nodes)"
        )
        if self.activation is not None:
            s += f"\n\tTensor contents: \n{print_override(self.activation, '__str__')}"
        return s

    def _str_after_pass(self) -> str:
        """Return a human-readable summary of this tensor entry after the forward pass has completed."""
        if self.num_passes > 1:
            pass_str = f" (pass {self.pass_num}/{self.num_passes}), "
        else:
            pass_str = ", "
        sml = self.source_model_log
        num_ops = sml.num_operations if sml is not None else "?"
        s = f"Layer {self.layer_label_no_pass}{pass_str}operation {self.operation_num}/{num_ops}:"
        s += f"\n\tOutput tensor: shape={self.tensor_shape}, dype={self.tensor_dtype}, size={self.tensor_memory_str}"
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
                f"{self.num_params_total} params total ({self.params_memory_str})"
            )
        else:
            s += "\n\tParams: no params used"
        if self.containing_module is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.containing_module}"
        if not self.is_input_layer:
            s += f"\n\tFunction: {self.func_name} (grad_fn: {self.grad_fn_name}) {module_str}"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
            s += f"\n\tTime elapsed: {self.func_time: .3E}s"
        if len(self.modules_exited) > 0:
            modules_exited_str = ", ".join(self.modules_exited)
            s += f"\n\tOutput of modules: {modules_exited_str}"
        else:
            s += "\n\tOutput of modules: none"
        if self.is_leaf_module_output:
            s += f"\n\tOutput of bottom-level module: {self.leaf_module_pass}"
        lookup_keys_str = ", ".join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents."""
        if self.activation is None:
            return ""
        else:
            s = ""
            s += f"\n\t\t{tensor_stats_summary(self.activation)}"
            tensor_size_shown = 8
            # Use logged shape, not live tensor shape (#45)
            saved_shape = (
                self.tensor_shape if self.tensor_shape is not None else self.activation.shape
            )
            # Slice first, then clone only the small slice (#73)
            if len(saved_shape) == 0:
                tensor_slice = self.activation.detach().clone()
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = self.activation[0:num_dims].detach().clone()
            elif len(saved_shape) == 2:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.activation[0:num_dims, 0:num_dims].detach().clone()
            else:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.activation.data
                for _ in range(len(saved_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims].detach().clone()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if (len(saved_shape) > 0) and (max(saved_shape) > tensor_size_shown):
                s += "..."
        return s

    def _tensor_family_str_helper(self) -> str:
        """Return a formatted string summarising parent, child, sibling, spouse, and ancestor relationships."""
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

        if len(self.co_parent_layers) > 0:
            s += "\n\t\t- shares children with layers: " + ", ".join(self.co_parent_layers)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:
            s += "\n\t\t- descendent of input layers: " + ", ".join(self.input_ancestors)
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.is_output_ancestor:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(self.output_descendants)
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

        return s

    def __repr__(self):
        return self.__str__()


# Backward-compatible alias: TensorLog was the original name for
# LayerPassLog before the LayerLog aggregate class was introduced in PR #92.
TensorLog = LayerPassLog
