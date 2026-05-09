"""OpLog: per-operation metadata for a single invocation of a layer.

Each OpLog records everything about one tensor operation in the
forward pass: the output tensor itself, the function that produced it,
its parents/children in the computation graph, module containment,
parameter usage, timing, RNG state, and more.

For recurrent models, the same "layer" may execute multiple times; each
execution is a separate OpLog with a distinct ``call_index``.  The
aggregate view across ops is provided by :class:`LayerLog`.

Field categories (matching the LAYER_PASS_LOG_FIELD_ORDER in constants.py):

1. **General info** - raw/final labels, operation numbering, back-reference
   to the owning Trace.
2. **Label info** - human-readable labels in various formats (with/without
   pass qualifier, short form, etc.).
3. **Saved tensor info** - the tensor contents, shape, dtype, size, device
   transfer settings, out postfunc, and function arguments.
4. **Child tensor variations** - tracks per-child input values for
   validation replay (``output_versions_per_child`` stores RAW values
   because validation compares against ``saved_args``).
5. **Gradient info** - grad tensor and metadata (stored as a bare
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
import hashlib
import weakref
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Tuple, Union, cast

import torch

from .._deprecations import MISSING
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
from ..intervention.types import LAYER_PASS_LOG_FIELD_FORK_POLICY
from ..intervention.errors import DirectActivationWriteWarning
from .._state import pause_logging
from ..utils.tensor_utils import (
    concatenate_batch_tensors,
    get_memory_amount,
    print_override,
    safe_copy,
    safe_to,
)
from ..utils.display import human_readable_size
from ..utils.display import tensor_stats_summary

_LAYER_PASS_LOG_FIELD_ORDER_SET = frozenset(LAYER_PASS_LOG_FIELD_ORDER)
_DIRECT_WRITE_GUARDED_FIELDS = frozenset(
    {
        "out",
        "transformed_out",
        "grad",
        "transformed_grad",
        "interventions",
    }
)
_LAYER_PASS_LOG_DEFAULT_FILL: dict[str, Any] = {
    "_source_trace_ref": None,
    "parent_layer_log": None,
    "out_ref": None,
    "grad_ref": None,
    "_pending_blob_id": None,
    "_pending_transformed_out_blob_id": None,
    "_pending_grad_blob_id": None,
    "_pending_transformed_grad_blob_id": None,
    "annotations": {},
    "autograd_saved_memory": None,
    "num_autograd_saved_tensors": None,
    "bytes_delta_at_call": None,
    "bytes_peak_at_call": None,
    "transformed_out": None,
    "transformed_out_shape": None,
    "transformed_out_dtype": None,
    "transformed_out_memory": None,
    "transformed_grad": None,
    "transformed_grad_shape": None,
    "transformed_grad_dtype": None,
    "transformed_grad_memory": None,
    "func_call_id": None,
    "container_path": (),
    "intervention_replaced": False,
    "interventions": [],
    "container_spec": None,
    "args_template": None,
    "kwargs_template": None,
    "edge_uses": [],
    "is_orphan": False,
    "_address_normalized": None,
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


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    """Return a tensor shape tuple, or ``None`` for non-tensor values."""

    return tuple(value.shape) if isinstance(value, torch.Tensor) else None


def _dtype_or_none(value: Any) -> torch.dtype | None:
    """Return a tensor dtype, or ``None`` for non-tensor values."""

    return value.dtype if isinstance(value, torch.Tensor) else None


def _memory_or_none(value: Any) -> int | None:
    """Return tensor memory in bytes, or ``None`` for non-tensor values."""

    return get_memory_amount(value) if isinstance(value, torch.Tensor) else None


def _tensor_content_hash(value: torch.Tensor) -> str:
    """Return a CPU content hash for a tensor.

    Parameters
    ----------
    value:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest.
    """

    with pause_logging():
        tensor = safe_copy(value, detach_tensor=True).cpu().contiguous()
        if tensor.dtype is torch.bfloat16:
            tensor = tensor.to(torch.float32)
        payload = tensor.numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(repr((tuple(tensor.shape), str(tensor.dtype))).encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


if TYPE_CHECKING:
    from .._io.lazy import LazyActivationRef
    from .func_call_location import FuncCallLocation
    from .layer_log import LayerLog
    from .param_log import ParamLog
    from .model_log import Trace


class OpLog:
    """Metadata for a single tensor operation (one pass of one layer).

    Constructed from a dict whose keys must exactly match
    ``LAYER_PASS_LOG_FIELD_ORDER`` (enforced at init time).  Every
    attribute is set explicitly (not via a loop) so that IDE
    autocompletion works.

    Notable design points:

    * ``_tracing_finished`` mirrors the owning Trace's flag. Methods like
      ``__str__`` branch on it to show raw vs final labels.
    * ``source_trace`` is a direct reference to the owning Trace.
      This creates a circular reference (Trace -> layer_list -> entry ->
      source_trace -> Trace) that is broken by ``cleanup()``.
    * ``parent_layer_log`` is a back-reference to the aggregate LayerLog
      that owns this pass.  It is set *outside* fields_dict during
      ``_build_layer_logs`` and is intentionally absent from FIELD_ORDER.
    * ``fx_qualpath`` and ``fx_call_index`` expose metadata that mirrors
      ``torch.fx.symbolic_trace`` naming conventions, computed independently
      using TorchLens rules.  ``fx_qualpath`` is not a lookup key, so
      ``trace[label]`` does not accept it.  Combine
      ``fx_qualpath.replace(".", "_")`` with ``fx_call_index`` when an
      FX-style name form is needed.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_label_raw": FieldPolicy.KEEP,
        "_layer_label_raw": FieldPolicy.KEEP,
        "compute_index": FieldPolicy.KEEP,
        "capture_index": FieldPolicy.KEEP,
        "source_trace": FieldPolicy.DROP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "_tracing_finished": FieldPolicy.KEEP,
        "_construction_done": FieldPolicy.DROP,
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "layer_label_w_pass": FieldPolicy.KEEP,
        "layer_label_w_pass_short": FieldPolicy.KEEP,
        "layer_label_no_pass": FieldPolicy.KEEP,
        "layer_label_no_pass_short": FieldPolicy.KEEP,
        "type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "trace_index": FieldPolicy.KEEP,
        "call_index": FieldPolicy.KEEP,
        "num_calls": FieldPolicy.KEEP,
        "lookup_keys": FieldPolicy.KEEP,
        "out": FieldPolicy.BLOB,
        "transformed_out": FieldPolicy.BLOB,
        "has_saved_outs": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "out_postfunc": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "interventions": FieldPolicy.DROP,
        "intervention_replaced": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "has_saved_args": FieldPolicy.KEEP,
        "saved_args": FieldPolicy.BLOB_RECURSIVE,
        "saved_kwargs": FieldPolicy.BLOB_RECURSIVE,
        "args_template": FieldPolicy.DROP,
        "kwargs_template": FieldPolicy.DROP,
        "shape": FieldPolicy.KEEP,
        "transformed_out_shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "transformed_out_dtype": FieldPolicy.KEEP,
        "memory": FieldPolicy.KEEP,
        "transformed_out_memory": FieldPolicy.KEEP,
        "autograd_saved_memory": FieldPolicy.KEEP,
        "num_autograd_saved_tensors": FieldPolicy.KEEP,
        "bytes_delta_at_call": FieldPolicy.KEEP,
        "bytes_peak_at_call": FieldPolicy.KEEP,
        "has_output_variations": FieldPolicy.KEEP,
        "output_versions_per_child": FieldPolicy.BLOB_RECURSIVE,
        "grad": FieldPolicy.BLOB,
        "transformed_grad": FieldPolicy.BLOB,
        "save_grads": FieldPolicy.KEEP,
        "has_grad": FieldPolicy.KEEP,
        "grad_shape": FieldPolicy.KEEP,
        "transformed_grad_shape": FieldPolicy.KEEP,
        "grad_dtype": FieldPolicy.KEEP,
        "transformed_grad_dtype": FieldPolicy.KEEP,
        "grad_memory": FieldPolicy.KEEP,
        "transformed_grad_memory": FieldPolicy.KEEP,
        "func": FieldPolicy.DROP,
        "func_call_id": FieldPolicy.KEEP,
        "func_name": FieldPolicy.KEEP,
        "code_context": FieldPolicy.KEEP,
        "func_duration": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "func_rng_states": FieldPolicy.BLOB_RECURSIVE,
        "func_autocast_state": FieldPolicy.KEEP,
        "arg_names": FieldPolicy.KEEP,
        "num_args_total": FieldPolicy.KEEP,
        "num_pos_args": FieldPolicy.KEEP,
        "num_kwargs": FieldPolicy.KEEP,
        "non_tensor_pos_args": FieldPolicy.KEEP,
        "non_tensor_kwargs": FieldPolicy.KEEP,
        "func_non_tensor_args": FieldPolicy.KEEP,
        "is_inplace": FieldPolicy.KEEP,
        "grad_fn_name": FieldPolicy.KEEP,
        "grad_fn_id": FieldPolicy.KEEP,
        "grad_fn": FieldPolicy.DROP,
        "grad_fn_log": FieldPolicy.DROP,
        "is_part_of_iterable_output": FieldPolicy.KEEP,
        "multi_output_index": FieldPolicy.KEEP,
        "container_path": FieldPolicy.KEEP,
        "container_spec": FieldPolicy.KEEP,
        "parent_params": FieldPolicy.KEEP,
        "_param_barcodes": FieldPolicy.KEEP,
        "parent_param_ops": FieldPolicy.KEEP,
        "_param_logs": FieldPolicy.KEEP,
        "param_shapes": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "equivalence_class": FieldPolicy.KEEP,
        "equivalent_ops": FieldPolicy.KEEP,
        "recurrent_ops": FieldPolicy.KEEP,
        "parents": FieldPolicy.KEEP,
        "parent_arg_positions": FieldPolicy.KEEP,
        "edge_uses": FieldPolicy.KEEP,
        "root_ancestors": FieldPolicy.KEEP,
        "children": FieldPolicy.KEEP,
        "has_children": FieldPolicy.KEEP,
        "is_input": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "input_ancestors": FieldPolicy.KEEP,
        "min_distance_from_input": FieldPolicy.KEEP,
        "max_distance_from_input": FieldPolicy.KEEP,
        "is_output": FieldPolicy.KEEP,
        "is_output_parent": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "has_output_descendant": FieldPolicy.KEEP,
        "output_descendants": FieldPolicy.KEEP,
        "is_orphan": FieldPolicy.KEEP,
        "min_distance_from_output": FieldPolicy.KEEP,
        "max_distance_from_output": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "is_buffer": FieldPolicy.KEEP,
        "buffer_address": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "buffer_parent": FieldPolicy.KEEP,
        "is_internal_source": FieldPolicy.KEEP,
        "has_internal_source_ancestor": FieldPolicy.KEEP,
        "internal_source_parents": FieldPolicy.KEEP,
        "internal_source_ancestors": FieldPolicy.KEEP,
        "is_internal_sink": FieldPolicy.KEEP,
        "is_terminal_bool": FieldPolicy.KEEP,
        "is_terminal_conditional_bool": FieldPolicy.KEEP,
        "conditional_context_kind": FieldPolicy.KEEP,
        "conditional_wrapper_kind": FieldPolicy.KEEP,
        "terminal_conditional_id": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "bool_value": FieldPolicy.KEEP,
        "in_conditionals": FieldPolicy.KEEP,
        "terminal_bool_for": FieldPolicy.KEEP,
        "is_in_conditional_body": FieldPolicy.KEEP,
        "conditional_branch_stack": FieldPolicy.KEEP,
        "conditional_branch_depth": FieldPolicy.KEEP,
        "conditional_entry_children": FieldPolicy.KEEP,
        "conditional_then_children": FieldPolicy.KEEP,
        "conditional_elif_children": FieldPolicy.KEEP,
        "conditional_else_children": FieldPolicy.KEEP,
        "conditional_arm_children": FieldPolicy.KEEP,
        "module": FieldPolicy.KEEP,
        "_address_normalized": FieldPolicy.KEEP,
        "modules": FieldPolicy.KEEP,
        "fx_qualpath": FieldPolicy.KEEP,
        "fx_call_index": FieldPolicy.KEEP,
        "modules_entered": FieldPolicy.KEEP,
        "module_entry_argnames": FieldPolicy.KEEP,
        "module_ops_entered": FieldPolicy.KEEP,
        "output_of_modules": FieldPolicy.KEEP,
        "output_of_module_calls": FieldPolicy.KEEP,
        "is_submodule_output": FieldPolicy.KEEP,
        "is_atomic_module_op": FieldPolicy.KEEP,
        "atomic_module_call": FieldPolicy.KEEP,
        "_module_boundary_threads_inputs": FieldPolicy.KEEP,
        "_module_boundary_thread_output": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "out_ref": FieldPolicy.DROP,
        "grad_ref": FieldPolicy.DROP,
        "_pending_blob_id": FieldPolicy.DROP,
        "_pending_transformed_out_blob_id": FieldPolicy.DROP,
        "_pending_grad_blob_id": FieldPolicy.DROP,
        "_pending_transformed_grad_blob_id": FieldPolicy.DROP,
        "parent_layer_log": FieldPolicy.DROP,
    }
    FIELD_FORK_POLICY = LAYER_PASS_LOG_FIELD_FORK_POLICY
    DEFAULT_FILL_STATE = _LAYER_PASS_LOG_DEFAULT_FILL
    _construction_done: bool = False

    def __getattribute__(self, name: str) -> Any:
        """Materialize lazy grads when the public ``grad`` field is accessed."""

        if name == "grad":
            state = object.__getattribute__(self, "__dict__")
            grad = state.get("grad")
            if grad is None and state.get("grad_ref") is not None:
                return object.__getattribute__(self, "materialize_grad")()
            return grad
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
            owner = self.__dict__.get("_source_trace_ref")
            trace = owner() if owner is not None else None
            if trace is not None:
                object.__setattr__(trace, "_has_direct_writes", True)
                object.__setattr__(trace, "run_state", RunState.DIRECT_WRITE_DIRTY)
                if not getattr(trace, "_warned_direct_write", False):
                    warnings.warn(
                        "DirectActivationWriteWarning: direct OpLog out writes "
                        "are not recipe edits; replay/rerun propagation will overlay them.",
                        DirectActivationWriteWarning,
                        stacklevel=2,
                    )
                    object.__setattr__(trace, "_warned_direct_write", True)
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

    def _append_tensor_from(self, other: "OpLog", field_name: str) -> None:
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

    def __init__(self, fields_dict: Dict[str, Any]) -> None:
        """Initialise from a complete fields dictionary.

        Args:
            fields_dict: Dict with values for all fields defined in
                ``LAYER_PASS_LOG_FIELD_ORDER``.  Missing or extra keys
                raise ``ValueError``.
        """
        # Attributes are set explicitly (not via loop) for IDE autocompletion.
        object.__setattr__(self, "_construction_done", False)

        # Validate that fields_dict has exactly the expected keys:
        if "_address_normalized" not in fields_dict:
            fields_dict["_address_normalized"] = None
        if "fx_qualpath" not in fields_dict:
            fields_dict["fx_qualpath"] = None
        if "fx_call_index" not in fields_dict:
            fields_dict["fx_call_index"] = 0
        fields_dict_key_set = set(fields_dict.keys())
        if fields_dict_key_set != _LAYER_PASS_LOG_FIELD_ORDER_SET:
            error_str = "Error initializing OpLog:"
            missing_fields = _LAYER_PASS_LOG_FIELD_ORDER_SET - fields_dict_key_set
            extra_fields = fields_dict_key_set - _LAYER_PASS_LOG_FIELD_ORDER_SET
            if len(missing_fields) > 0:
                error_str += f"\n\t- Missing fields {', '.join(missing_fields)}"
            if len(extra_fields) > 0:
                error_str += f"\n\t- Extra fields {', '.join(extra_fields)}"
            raise ValueError(error_str)

        # General info:
        self._label_raw = fields_dict["_label_raw"]
        self._layer_label_raw = fields_dict["_layer_label_raw"]
        self.compute_index = fields_dict["compute_index"]
        self.capture_index = fields_dict["capture_index"]
        # Store as weakref to break circular reference (Trace -> layer_list -> entry -> Trace).
        _sml = fields_dict["source_trace"]
        self._source_trace_ref = weakref.ref(_sml) if _sml is not None else None
        self._tracing_finished = fields_dict["_tracing_finished"]

        # Label info:
        self.layer_label = fields_dict["layer_label"]
        self.layer_label_short = fields_dict["layer_label_short"]
        self.layer_label_w_pass = fields_dict["layer_label_w_pass"]
        self.layer_label_w_pass_short = fields_dict["layer_label_w_pass_short"]
        self.layer_label_no_pass = fields_dict["layer_label_no_pass"]
        self.layer_label_no_pass_short = fields_dict["layer_label_no_pass_short"]
        self.type = fields_dict["type"]
        self.type_index = fields_dict["type_index"]
        self.trace_index = fields_dict["trace_index"]
        self.call_index = fields_dict["call_index"]
        self.num_calls = fields_dict["num_calls"]
        self.lookup_keys = fields_dict["lookup_keys"]

        # Saved tensor info:
        self.out = fields_dict["out"]
        self.transformed_out = fields_dict["transformed_out"]
        self.has_saved_outs = fields_dict["has_saved_outs"]
        self.output_device = fields_dict["output_device"]
        self.out_postfunc = fields_dict["out_postfunc"]
        self.annotations: Dict[str, Any] = fields_dict["annotations"]
        self.interventions = fields_dict["interventions"]
        self.intervention_replaced = fields_dict["intervention_replaced"]
        self.detach_saved_activations = fields_dict["detach_saved_activations"]
        self.has_saved_args = fields_dict["has_saved_args"]
        self.saved_args = fields_dict["saved_args"]
        self.saved_kwargs = fields_dict["saved_kwargs"]
        self.args_template = fields_dict["args_template"]
        self.kwargs_template = fields_dict["kwargs_template"]
        self.shape = fields_dict["shape"]
        self.transformed_out_shape = fields_dict["transformed_out_shape"]
        self.dtype = fields_dict["dtype"]
        self.transformed_out_dtype = fields_dict["transformed_out_dtype"]
        self.memory = fields_dict["memory"]
        self.transformed_out_memory = fields_dict["transformed_out_memory"]
        self.autograd_saved_memory: Optional[int] = fields_dict["autograd_saved_memory"]
        self.num_autograd_saved_tensors: Optional[int] = fields_dict["num_autograd_saved_tensors"]
        self.bytes_delta_at_call: Optional[int] = fields_dict["bytes_delta_at_call"]
        self.bytes_peak_at_call: Optional[int] = fields_dict["bytes_peak_at_call"]

        # Child tensor variation tracking - stores the raw tensor values that
        # each child operation received as input.  Must store RAW values (not
        # postprocessed) because validation compares these against saved_args.
        self.has_output_variations = fields_dict["has_output_variations"]
        self.output_versions_per_child = fields_dict["output_versions_per_child"]

        # Saved grad info - grad is stored as a bare clone (not deep-copied)
        # via log_tensor_grad().  grad is populated by a backward hook.
        self.grad = fields_dict["grad"]
        self.transformed_grad = fields_dict["transformed_grad"]
        self.save_grads = fields_dict["save_grads"]
        self.has_grad = fields_dict["has_grad"]
        self.grad_shape = fields_dict["grad_shape"]
        self.transformed_grad_shape = fields_dict["transformed_grad_shape"]
        self.grad_dtype = fields_dict["grad_dtype"]
        self.transformed_grad_dtype = fields_dict["transformed_grad_dtype"]
        self.grad_memory = fields_dict["grad_memory"]
        self.transformed_grad_memory = fields_dict["transformed_grad_memory"]

        # Function call info:
        self.func = fields_dict["func"]
        self.func_call_id = fields_dict["func_call_id"]
        self.func_name = fields_dict["func_name"]
        self.code_context: List["FuncCallLocation"] = fields_dict["code_context"]
        self.func_duration = fields_dict["func_duration"]
        self.flops_forward = fields_dict["flops_forward"]
        self.flops_backward = fields_dict["flops_backward"]
        self.func_rng_states = fields_dict["func_rng_states"]
        self.func_autocast_state = fields_dict["func_autocast_state"]
        self.arg_names = fields_dict["arg_names"]
        self.num_args_total = fields_dict["num_args_total"]
        self.num_pos_args = fields_dict["num_pos_args"]
        self.num_kwargs = fields_dict["num_kwargs"]
        self.non_tensor_pos_args = fields_dict["non_tensor_pos_args"]
        self.non_tensor_kwargs = fields_dict["non_tensor_kwargs"]
        self.func_non_tensor_args = fields_dict["func_non_tensor_args"]
        self.is_inplace = fields_dict["is_inplace"]
        self.grad_fn_name = fields_dict["grad_fn_name"]
        self.grad_fn_id = fields_dict["grad_fn_id"]
        self.grad_fn = fields_dict["grad_fn"]
        self.grad_fn_log = fields_dict["grad_fn_log"]
        self.is_part_of_iterable_output = fields_dict["is_part_of_iterable_output"]
        self.multi_output_index = fields_dict["multi_output_index"]
        self.container_path = fields_dict["container_path"]
        self.container_spec = fields_dict["container_spec"]

        # Param info:
        self.parent_params = fields_dict["parent_params"]
        self._param_barcodes = fields_dict["_param_barcodes"]
        self.parent_param_ops = fields_dict["parent_param_ops"]
        self._param_logs: List["ParamLog"] = fields_dict["_param_logs"]
        self.param_shapes = fields_dict["param_shapes"]
        self.num_params = fields_dict["num_params"]
        self.num_params_trainable = fields_dict["num_params_trainable"]
        self.num_params_frozen = fields_dict["num_params_frozen"]
        self.param_memory = fields_dict["param_memory"]

        # Loop-detection equivalence info:
        # equivalence_class groups structurally identical operations
        # (same func + same param barcodes).  equivalent_ops holds a
        # DIRECT reference to the Trace-level set for this type.
        # recurrent_ops is populated by loop_detection.py for layers
        # that are different ops of the same recurrent layer.
        self.equivalence_class = fields_dict["equivalence_class"]
        self.equivalent_ops = fields_dict["equivalent_ops"]
        self.recurrent_ops = fields_dict["recurrent_ops"]

        # Graph info:
        self.parents = fields_dict["parents"]
        self.parent_arg_positions = fields_dict["parent_arg_positions"]
        self.edge_uses = fields_dict["edge_uses"]
        self.root_ancestors = fields_dict["root_ancestors"]
        self.children = fields_dict["children"]
        self.has_children = fields_dict["has_children"]
        self.is_input = fields_dict["is_input"]
        self.has_input_ancestor = fields_dict["has_input_ancestor"]
        self.input_ancestors = fields_dict["input_ancestors"]
        self.min_distance_from_input = fields_dict["min_distance_from_input"]
        self.max_distance_from_input = fields_dict["max_distance_from_input"]
        self.is_output = fields_dict["is_output"]
        self.is_output_parent = fields_dict["is_output_parent"]
        self.is_final_output = fields_dict["is_final_output"]
        self.has_output_descendant = fields_dict["has_output_descendant"]
        self.output_descendants = fields_dict["output_descendants"]
        self.is_orphan: bool = fields_dict["is_orphan"]
        self.min_distance_from_output = fields_dict["min_distance_from_output"]
        self.max_distance_from_output = fields_dict["max_distance_from_output"]
        self.io_role = fields_dict["io_role"]
        self.is_buffer = fields_dict["is_buffer"]
        self.buffer_address = fields_dict["buffer_address"]
        self.buffer_pass = fields_dict["buffer_pass"]
        self.buffer_parent = fields_dict["buffer_parent"]
        self.is_internal_source = fields_dict["is_internal_source"]
        self.has_internal_source_ancestor = fields_dict["has_internal_source_ancestor"]
        self.internal_source_parents = fields_dict["internal_source_parents"]
        self.internal_source_ancestors = fields_dict["internal_source_ancestors"]
        self.is_internal_sink = fields_dict["is_internal_sink"]

        # Conditional info
        self.is_terminal_bool = fields_dict["is_terminal_bool"]
        self.is_terminal_conditional_bool = fields_dict["is_terminal_conditional_bool"]
        self.conditional_context_kind = fields_dict["conditional_context_kind"]
        self.conditional_wrapper_kind = fields_dict["conditional_wrapper_kind"]
        self.terminal_conditional_id = fields_dict["terminal_conditional_id"]
        self.is_scalar_bool = fields_dict["is_scalar_bool"]
        self.bool_value = fields_dict["bool_value"]
        self.in_conditionals = fields_dict["in_conditionals"]
        self.terminal_bool_for = fields_dict["terminal_bool_for"]
        self.is_in_conditional_body = fields_dict["is_in_conditional_body"]
        self.conditional_branch_stack = fields_dict["conditional_branch_stack"]
        self.conditional_branch_depth = fields_dict["conditional_branch_depth"]
        self.conditional_entry_children = fields_dict["conditional_entry_children"]
        self.conditional_then_children = fields_dict["conditional_then_children"]
        self.conditional_elif_children = fields_dict["conditional_elif_children"]
        self.conditional_else_children = fields_dict["conditional_else_children"]
        self.conditional_arm_children = fields_dict["conditional_arm_children"]

        # Module info
        self.module = fields_dict["module"]
        self._address_normalized = fields_dict["_address_normalized"]
        self.modules = fields_dict["modules"]
        self.fx_qualpath: Optional[str] = fields_dict["fx_qualpath"]
        self.fx_call_index: int = fields_dict["fx_call_index"]
        self.modules_entered = fields_dict["modules_entered"]
        self.module_entry_argnames = fields_dict["module_entry_argnames"]
        self.module_ops_entered = fields_dict["module_ops_entered"]
        self.output_of_modules = fields_dict["output_of_modules"]
        self.output_of_module_calls = fields_dict["output_of_module_calls"]
        self.is_submodule_output = fields_dict["is_submodule_output"]
        self.is_atomic_module_op = fields_dict["is_atomic_module_op"]
        self.atomic_module_call = fields_dict["atomic_module_call"]
        self._module_boundary_threads_inputs = fields_dict["_module_boundary_threads_inputs"]
        self._module_boundary_thread_output = fields_dict["_module_boundary_thread_output"]

        # Function config - lightweight hyperparameters always captured.
        self.func_config = fields_dict["func_config"]

        # Back-reference to the aggregate LayerLog that groups all ops of
        # this layer.  Set during postprocessing by _build_layer_logs - NOT
        # part of fields_dict or FIELD_ORDER (it's a structural link, not
        # captured data).
        self.out_ref: Optional["LazyActivationRef"] = None
        self.grad_ref: Optional["LazyActivationRef"] = None
        self._pending_blob_id: Optional[str] = None
        self._pending_transformed_out_blob_id: Optional[str] = None
        self._pending_grad_blob_id: Optional[str] = None
        self._pending_transformed_grad_blob_id: Optional[str] = None
        self.parent_layer_log: Optional["LayerLog"] = None
        object.__setattr__(self, "_construction_done", True)

    @property
    def layer_type(self) -> str:
        """Return the operation type token used by existing internal callers."""

        return cast(str, self.type)

    @layer_type.setter
    def layer_type(self, value: str) -> None:
        """Set the operation type token through the legacy internal name."""

        self.type = value

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
        return len(self.parents) > 0

    @property
    def siblings(self) -> list[str]:
        """Layers sharing at least one parent (excluding output layers)."""
        ml = self.source_trace
        if ml is None:
            return []
        my_label = self.layer_label if self._tracing_finished else self._label_raw
        siblings = []
        seen = {my_label}
        for parent_label in self.parents:
            parent = ml[parent_label]
            for child_label in parent.children:
                if child_label not in seen:
                    seen.add(child_label)
                    child = ml[child_label]
                    if not child.is_output:
                        siblings.append(child_label)
        return siblings

    @property
    def has_siblings(self) -> bool:
        """Whether this layer shares parents with other layers."""
        return len(self.siblings) > 0

    @property
    def co_parents(self) -> list[str]:
        """Layers sharing at least one child (excluding output layers)."""
        ml = self.source_trace
        if ml is None:
            return []
        my_label = self.layer_label if self._tracing_finished else self._label_raw
        spouses = []
        seen = {my_label}
        for child_label in self.children:
            child = ml[child_label]
            for parent_label in child.parents:
                if parent_label not in seen:
                    seen.add(parent_label)
                    parent = ml[parent_label]
                    if not parent.is_output:
                        spouses.append(parent_label)
        return spouses

    @property
    def has_co_parents(self) -> bool:
        """Whether this layer shares children with other layers."""
        return len(self.co_parents) > 0

    @property
    def is_in_conditional(self) -> bool:
        """Whether this op participates in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this op computes a conditional arm condition."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this op is in a conditional arm body."""

        if self.has_output_descendant and not self.conditional_entry_children:
            return False
        return bool(self.__dict__.get("_is_in_conditional_body", False)) or any(
            role.role == "body" for role in self.in_conditionals or []
        )

    @is_in_conditional_body.setter
    def is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during postprocessing."""

        self.__dict__["_is_in_conditional_body"] = value

    @is_in_conditional_body.deleter
    def is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        self.__dict__.pop("_is_in_conditional_body", None)

    @property
    def conditional_depth(self) -> int:
        """Number of distinct conditionals this op participates in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

    @property
    def uses_params(self) -> bool:
        """Whether this operation used model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this operation."""
        return len(self._param_barcodes)

    @property
    def in_submodule(self) -> bool:
        """Whether this operation was computed inside a submodule."""
        return self.module is not None

    @property
    def module_call_depth(self) -> int:
        """Depth of module nesting for this operation."""
        return len(self.modules)

    @property
    def is_submodule_input(self) -> bool:
        """Whether this operation is the first inside a submodule's forward()."""
        return len(self.modules_entered) > 0

    @property
    def memory_str(self) -> str:
        """Return out tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable out memory amount.
        """
        return human_readable_size(self.memory)

    @property
    def grad_memory_str(self) -> str:
        """Return grad tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable grad memory amount.
        """
        return human_readable_size(self.grad_memory)

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved out."""

        return self.out

    @property
    def ops(self) -> tuple["OpLog", ...]:
        """Tuple containing this pass for aggregate-compatible iteration.

        Returns
        -------
        tuple[OpLog, ...]
            Single-entry tuple containing this pass log.
        """

        return (self,)

    @property
    def param_memory_str(self) -> str:
        """Return parameter tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable parameter memory amount.
        """
        return human_readable_size(self.param_memory)

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
            self._layer_label_raw,
            self._label_raw,
        ):
            if candidate is not None:
                return str(candidate)
        return "<unknown>"

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref)."""
        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            return None  # type: ignore[return-value]
        obj = ref()
        return cast("Trace", obj)

    @source_trace.setter
    def source_trace(self, value: "Trace | None") -> None:
        """Set the owning Trace back-reference.

        Parameters
        ----------
        value:
            Owning model log, or ``None`` to clear the reference.
        """
        self._source_trace_ref = weakref.ref(value) if value is not None else None

    def _source_trace_or_error(self) -> "Trace":
        """Return the owning Trace, or raise a detached-log error.

        Returns
        -------
        Trace
            Source Trace that owns this operation log.

        Raises
        ------
        AttributeError
            If this operation log is detached from its source Trace.
        """

        ref = self.__dict__.get("_source_trace_ref")
        source = ref() if ref is not None else None
        if source is None or getattr(source, "_loaded_from_bundle", False):
            raise AttributeError(
                "This OpLog is detached from its source Trace "
                "(perhaps loaded from disk or after cleanup). "
                "Use trace.do(label, transform) directly."
            )
        return cast("Trace", source)

    def do(
        self,
        transform: Any,
        *,
        model: Any = None,
        x: Any = None,
        engine: Any = MISSING,
        confirm_mutation: Any = MISSING,
        strict: Any = MISSING,
        intervention: Any = None,
    ) -> "Trace":
        """Apply an intervention to this op through the owning Trace.

        Parameters
        ----------
        transform:
            Transform or hook to apply to this operation's output.
        model:
            Model required when ``engine="rerun"``.
        x:
            Input required when ``engine="rerun"``.
        engine:
            ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.
        strict:
            Whether selector and propagation checks should raise.
        intervention:
            Grouped intervention options.

        Returns
        -------
        Trace
            Source Trace after applying the intervention.
        """

        return self._source_trace_or_error().do(
            self.layer_label,
            transform,
            model=model,
            x=x,
            engine=engine,
            confirm_mutation=confirm_mutation,
            strict=strict,
            intervention=intervention,
        )

    def set(
        self,
        value: Any,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Set this op's out recipe through the owning Trace.

        Parameters
        ----------
        value:
            Static replacement value or one-shot callable.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Trace
            Source Trace with a stale intervention recipe.
        """

        return self._source_trace_or_error().set(
            self.layer_label,
            value,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )

    def attach_hooks(
        self,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
        """Attach sticky hooks to this op through the owning Trace.

        Parameters
        ----------
        hook:
            Hook or helper to attach to this operation.
        *extra_hooks:
            Additional hooks to compose on this operation in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress root mutation warnings when intentionally mutating.

        Returns
        -------
        Any
            Trace or scoped removable hook handle, matching ``Trace.attach_hooks``.
        """

        return self._source_trace_or_error().attach_hooks(
            self.layer_label,
            hook,
            *extra_hooks,
            strict=strict,
            prepend=prepend,
            confirm_mutation=confirm_mutation,
        )

    def materialize_out(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Materialize this layer's saved out from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized out tensor.

        Raises
        ------
        TorchLensIOError
            If no out ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> trace = tl.load("demo_bundle", lazy=True)
        >>> tensor = trace["linear_1_1"].materialize_out()
        >>> tensor.shape
        torch.Size([2, 3])
        """

        if isinstance(self.out, torch.Tensor):
            return self.out
        if self.out_ref is None:
            raise TorchLensIOError("no out_ref to materialize from")
        self._internal_set("out", self.out_ref.materialize(map_location=map_location))
        return cast(torch.Tensor, self.out)

    def materialize_grad(
        self,
        *,
        map_location: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Materialize this layer's saved grad from a lazy bundle ref.

        Parameters
        ----------
        map_location:
            Target device for the materialized tensor.

        Returns
        -------
        torch.Tensor
            Materialized grad tensor.

        Raises
        ------
        TorchLensIOError
            If no grad ref is available for this layer.

        Examples
        --------
        >>> import torchlens as tl
        >>> trace = tl.load("demo_bundle", lazy=True)
        >>> grad = trace["linear_1_1"].materialize_grad()
        >>> grad.shape
        torch.Size([2, 3])
        """

        grad = self.__dict__.get("grad")
        if isinstance(grad, torch.Tensor):
            return grad
        if self.grad_ref is None:
            raise TorchLensIOError("no grad_ref to materialize from")
        self._internal_set("grad", self.grad_ref.materialize(map_location=map_location))
        return cast(torch.Tensor, self.grad)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state["func"] = None
        state["grad_fn"] = None
        state["grad_fn_log"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        read_io_format_version(state, cls_name=type(self).__name__)
        old_key_map = {
            "tensor_label_raw": "_label_raw",
            "operation_num": "op_num",
            "activation": "out",
            "transformed_activation": "transformed_out",
            "has_saved_activations": "has_saved_outs",
            "activation_postfunc": "out_postfunc",
            "activation_shape": "shape",
            "transformed_activation_shape": "transformed_out_shape",
            "activation_dtype": "dtype",
            "transformed_activation_dtype": "transformed_out_dtype",
            "activation_memory": "memory",
            "transformed_activation_memory": "transformed_out_memory",
            "is_part_of_iterable_output": "is_part_of_iterable_output",
            "iterable_output_index": "multi_output_index",
            "grad_fn_object": "grad_fn",
            "corresponding_grad_fn": "grad_fn_log",
            "is_input_layer": "is_input",
            "is_output_layer": "is_output",
            "is_output_ancestor": "has_output_descendant",
            "is_buffer_layer": "is_buffer",
            "internally_initialized": "is_internal_source",
            "internally_terminated": "is_internal_sink",
            "parent_param_barcodes": "_param_barcodes",
            "module_passes_entered": "module_ops_entered",
            "modules_exited": "output_of_modules",
            "module_passes_exited": "output_of_module_calls",
            "is_leaf_module_output": "is_atomic_module_op",
            "leaf_module_pass": "atomic_module_call",
            "module_entry_exit_threads_inputs": "_module_boundary_thread_inputs",
            "module_entry_exit_thread_output": "_module_boundary_thread_output",
            "activation_ref": "out_ref",
            "gradient_ref": "grad_ref",
        }
        for old_key, new_key in old_key_map.items():
            if new_key not in state and old_key in state:
                state[new_key] = state.pop(old_key)
        default_fill_state(
            state,
            defaults=self.DEFAULT_FILL_STATE,
        )
        object.__setattr__(self, "_construction_done", False)
        state.pop("source_trace", None)
        self.__dict__.update(state)
        object.__setattr__(self, "_construction_done", bool(state.get("_construction_done", True)))

    @property
    def out_transform(self) -> Optional[Callable[..., Any]]:
        """Canonical out transform callable used for this pass.

        Returns
        -------
        Optional[Callable]
            Transform callable, or ``None`` when outs are stored unchanged.
        """

        return cast("Callable[..., Any] | None", self.out_postfunc)

    @out_transform.setter
    def out_transform(self, value: Optional[Callable[..., Any]]) -> None:
        """Set the canonical out transform callable.

        Parameters
        ----------
        value:
            Transform callable, or ``None``.
        """

        self._internal_set("out_postfunc", value)

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

    def copy(self) -> "OpLog":
        """Return a selective-depth copy of this entry.

        Most fields are ``copy.deepcopy``'d so the clone is fully independent.
        However, certain fields are shallow-copied (shared by reference) because:

        * ``func``, ``grad_fn_name`` - function objects, immutable/shared.
        * ``source_trace`` - must point to the same Trace instance.
        * ``func_rng_states`` - large state dicts, not mutated after capture.
        * ``saved_args``, ``saved_kwargs`` - may contain large tensors;
          deep-copying them is expensive and unnecessary.
        * ``parent_params`` - references to nn.Parameters, must stay shared.
        * ``out``, ``output_versions_per_child`` - large tensors;
          shared references are safe since they're replaced (not mutated).

        Returns:
            A new OpLog (or subclass) with the same field values.
        """
        fields_dict = {}
        fields_not_to_deepcopy = [
            "func",
            "grad_fn_name",
            "grad_fn",
            "grad_fn_log",
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
            "output_versions_per_child",
        ]
        for field in LAYER_PASS_LOG_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = copy.deepcopy(getattr(self, field, None))
            else:
                fields_dict[field] = getattr(self, field, None)
        copied_entry = type(self)(fields_dict)
        return copied_entry

    def save_activation(
        self,
        t: torch.Tensor,
        t_args: Union[List[Any], Tuple[Any, ...]],
        t_kwargs: Dict[str, Any],
        save_arg_values: bool,
        out_postfunc: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Save the output tensor (and optionally args) for this operation.

        Flow:
        1. Clone the tensor via ``safe_copy`` (strips tl_ attributes to avoid
           logging the copy operation).
        2. Move to ``output_device`` if different from the tensor's current device.
        3. Apply ``out_postfunc`` inside ``pause_logging()`` to prevent
           the postfunc's own tensor ops from being logged.
        4. Optionally deep-copy function args/kwargs via ``_recursive_safe_copy``.

        Args:
            t: The output tensor of the operation.
            t_args: Positional arguments passed to the operation.
            t_kwargs: Keyword arguments passed to the operation.
            save_arg_values: Whether to deep-copy and store args/kwargs.
            out_postfunc: Optional transform applied to the tensor
                before storing (e.g. detach, to-numpy, normalize).
        """
        trace = self.source_trace
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        try:
            # Clone the tensor, optionally detaching from autograd graph.
            raw_out = safe_copy(t, self.detach_saved_activations)
            # Move to the user-requested output device if needed.
            if self.output_device not in [str(raw_out.device), "same"]:
                raw_out = safe_to(raw_out, self.output_device)

            self.shape = tuple(raw_out.shape)
            self.dtype = raw_out.dtype
            self.memory = get_memory_amount(raw_out)

            save_raw_outs = getattr(trace, "save_raw_outs", True)
            store_raw = save_raw_outs or out_postfunc is None
            if trace is not None and store_raw and not getattr(trace, "save_arg_values", False):
                hash_cache = getattr(trace, "_out_hash_cache", None)
                if hash_cache is None:
                    hash_cache = {}
                    setattr(trace, "_out_hash_cache", hash_cache)
                out_hash = _tensor_content_hash(raw_out)
                if out_hash in hash_cache:
                    self.annotations["dedup_out_hash"] = out_hash
                    self.annotations["dedup_reference_label"] = hash_cache[out_hash][0]
                    raw_out = hash_cache[out_hash][1]
                else:
                    hash_cache[out_hash] = (self._layer_label_raw, raw_out)
            self._internal_set("out", raw_out if store_raw else None)

            self._internal_set("transformed_out", None)
            self.transformed_out_shape = None
            self.transformed_out_dtype = None
            self.transformed_out_memory = None
            if out_postfunc is not None:
                self._internal_set(
                    "transformed_out",
                    self._apply_postfunc(
                        raw_out,
                        out_postfunc,
                        postfunc_kind="out",
                        streaming_active=writer is not None,
                    ),
                )
                self._validate_train_mode_postfunc_output(
                    raw_out,
                    self.transformed_out,
                    postfunc_kind="out",
                )
                self.transformed_out_shape = _shape_or_none(self.transformed_out)
                self.transformed_out_dtype = _dtype_or_none(self.transformed_out)
                self.transformed_out_memory = _memory_or_none(self.transformed_out)
        except Exception as exc:
            if writer is not None:
                writer.abort(f"Failed while saving out for {self._streaming_label}: {exc}")
                if isinstance(exc, TorchLensPostfuncError):
                    raise
                raise TorchLensIOError(
                    f"Streaming out save failed for {self._streaming_label}."
                ) from exc
            raise

        self.has_saved_outs = True

        if trace is not None:
            out_sink = getattr(trace, "_out_sink", None)
            if out_sink is not None and isinstance(self.out, torch.Tensor):
                out_sink(self._streaming_label, self.out)

            if writer is not None and getattr(trace, "_in_exhaustive_pass", False):
                self._stream_tensor_blob(
                    writer,
                    tensor_field="out",
                    pending_field="_pending_blob_id",
                    kind="out",
                )
                self._stream_tensor_blob(
                    writer,
                    tensor_field="transformed_out",
                    pending_field="_pending_transformed_out_blob_id",
                    kind="transformed_out",
                )

        # Tensor args and kwargs:
        if save_arg_values:
            self.has_saved_args = True
            self._internal_set("saved_args", [_recursive_safe_copy(arg) for arg in t_args])
            self._internal_set(
                "saved_kwargs",
                {k: _recursive_safe_copy(v) for k, v in t_kwargs.items()},
            )
        else:
            self._internal_set("saved_args", None)
            self._internal_set("saved_kwargs", None)

    def log_tensor_grad(self, grad: torch.Tensor) -> None:
        """Save the grad tensor for this layer's output.

        Called by the backward hook registered during the forward pass.
        The grad is ``detach().clone()``'d - a bare copy, not deep-copied -
        so it's independent of the autograd graph but cheap to store.

        Args:
            grad: The grad tensor flowing back through this operation.
        """
        trace = self.source_trace
        raw_grad = grad
        self.grad_shape = tuple(raw_grad.shape)
        self.grad_dtype = raw_grad.dtype
        self.grad_memory = get_memory_amount(raw_grad)
        grad_transform = getattr(trace, "grad_transform", None)
        self._internal_set("transformed_grad", None)
        self.transformed_grad_shape = None
        self.transformed_grad_dtype = None
        self.transformed_grad_memory = None
        writer = getattr(trace, "_out_writer", None) if trace is not None else None
        if grad_transform is not None:
            self._internal_set(
                "transformed_grad",
                self._apply_postfunc(
                    raw_grad,
                    grad_transform,
                    postfunc_kind="grad",
                    streaming_active=writer is not None,
                ),
            )
            self._validate_train_mode_postfunc_output(
                raw_grad,
                self.transformed_grad,
                postfunc_kind="grad",
            )
            self.transformed_grad_shape = _shape_or_none(self.transformed_grad)
            self.transformed_grad_dtype = _dtype_or_none(self.transformed_grad)
            self.transformed_grad_memory = _memory_or_none(self.transformed_grad)

        save_raw_grads = getattr(trace, "save_raw_grads", True)
        store_raw = save_raw_grads or grad_transform is None
        self._internal_set("grad", raw_grad.detach().clone() if store_raw else None)
        self.has_grad = True
        if writer is not None and getattr(trace, "_defer_streaming_bundle_finalization", False):
            self._stream_tensor_blob(
                writer,
                tensor_field="grad",
                pending_field="_pending_grad_blob_id",
                kind="grad",
            )
            self._stream_tensor_blob(
                writer,
                tensor_field="transformed_grad",
                pending_field="_pending_transformed_grad_blob_id",
                kind="transformed_grad",
            )

    def _apply_postfunc(
        self,
        tensor: torch.Tensor,
        postfunc: Callable[..., Any],
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
        """Build context for an out or grad postfunc failure."""

        return (
            f"{postfunc_kind}_postfunc raised for layer {self._streaming_label} "
            f"(raw={self._layer_label_raw}, func={self.func_name}, "
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

        trace = self.source_trace
        if not getattr(trace, "train_mode", False) or not raw_tensor.requires_grad:
            return
        if not isinstance(output, torch.Tensor):
            raise TrainingModeConfigError(
                f"{postfunc_kind}_postfunc must return a torch.Tensor while train_mode=True "
                f"for layer {self._streaming_label}."
            )
        if output.dtype in _NON_GRAD_DTYPES:
            raise TrainingModeConfigError(
                f"train_mode=True with non-grad dtype {output.dtype} on layer "
                f"{self._streaming_label}. Integer and bool dtypes cannot propagate grads."
            )
        if output.grad_fn is None:
            raise TrainingModeConfigError(
                f"{postfunc_kind}_postfunc returned a tensor disconnected from the autograd "
                "graph (grad_fn is None) while train_mode=True. The transformed out "
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
            if kind == "transformed_out":
                message = (
                    "Streaming save requires out_postfunc outputs to be torch.Tensor "
                    f"instances, but layer {self._streaming_label} produced "
                    f"{type(tensor).__name__}."
                )
            elif kind == "transformed_grad":
                message = (
                    "Streaming save requires grad_transform outputs to be torch.Tensor "
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

    def get_children(self) -> list["OpLog"]:
        """Return child OpLog objects for this pass.

        Returns
        -------
        list[OpLog]
            Child ops resolved through the owning model log.
        """
        return [self.source_trace[child_label] for child_label in self.children]

    def get_parents(self) -> list["OpLog"]:
        """Return parent OpLog objects for this pass.

        Returns
        -------
        list[OpLog]
            Parent ops resolved through the owning model log.
        """
        return [self.source_trace[parent_label] for parent_label in self.parents]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this pass's saved out.

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
    def params(self) -> Any:
        """Access parameter metadata by address, short name, or index."""
        from .param_log import ParamAccessor

        param_dict = {pl.address: pl for pl in self._param_logs}
        return ParamAccessor(param_dict)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        if self._tracing_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def _str_during_pass(self) -> str:
        """Return a human-readable summary of this tensor entry while the forward pass is still in progress."""
        s = f"Tensor {self._label_raw} (layer {self._layer_label_raw}) (PASS NOT FINISHED):"
        s += f"\n\tPass: {self.call_index}"
        s += f"\n\tTensor info: shape {self.shape}, dtype {self.dtype}"
        s += f"\n\tComputed from params: {self.uses_params}"
        s += f"\n\tComputed in modules: {self.modules}"
        s += f"\n\tOutput of modules: {self.output_of_module_calls}"
        if self.is_atomic_module_op:
            s += " (bottom-level submodule output)"
        else:
            s += " (not bottom-level submodule output)"
        s += "\n\tFamily info:"
        s += f"\n\t\tParents: {self.parents}"
        s += f"\n\t\tChildren: {self.children}"
        s += f"\n\t\tSpouses: {self.co_parents}"
        s += f"\n\t\tSiblings: {self.siblings}"
        s += (
            f"\n\t\tOriginal Ancestors: {self.root_ancestors} "
            f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        )
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internal_source_ancestors}"
        s += (
            f"\n\t\tOutput Descendents: {self.output_descendants} "
            f"(min dist {self.min_distance_from_output} nodes, max dist {self.max_distance_from_output} nodes)"
        )
        if self.out is not None:
            s += f"\n\tTensor contents: \n{print_override(self.out, '__str__')}"
        return s

    def _str_after_pass(self) -> str:
        """Return a human-readable summary of this tensor entry after the forward pass has completed."""
        if self.num_calls > 1:
            pass_str = f" (pass {self.call_index}/{self.num_calls}), "
        else:
            pass_str = ", "
        sml = self.source_trace
        num_ops = sml.num_ops if sml is not None else "?"
        s = f"Layer {self.layer_label_no_pass}{pass_str}operation {self.compute_index}/{num_ops}:"
        s += f"\n\tOutput tensor: shape={self.shape}, dype={self.dtype}, size={self.memory_str}"
        if not self.has_saved_outs:
            s += " (not saved)"
        s += self._tensor_contents_str_helper()
        s += self._tensor_family_str_helper()
        if len(self.param_shapes) > 0:
            params_shapes_str = ", ".join(str(param_shape) for param_shape in self.param_shapes)
            s += (
                f"\n\tParams: Computed from params with shape {params_shapes_str}; "
                f"{self.num_params} params total ({self.param_memory_str})"
            )
        else:
            s += "\n\tParams: no params used"
        if self.module is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.module}"
        if not self.is_input:
            s += f"\n\tFunction: {self.func_name} (grad_fn: {self.grad_fn_name}) {module_str}"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
            s += f"\n\tTime elapsed: {self.func_duration: .3E}s"
        if len(self.output_of_modules) > 0:
            output_of_modules_str = ", ".join(self.output_of_modules)
            s += f"\n\tOutput of modules: {output_of_modules_str}"
        else:
            s += "\n\tOutput of modules: none"
        if self.is_atomic_module_op:
            s += f"\n\tOutput of bottom-level module: {self.atomic_module_call}"
        lookup_keys_str = ", ".join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents."""
        if self.out is None:
            return ""
        else:
            s = ""
            s += f"\n\t\t{tensor_stats_summary(self.out)}"
            tensor_size_shown = 8
            # Use logged shape, not live tensor shape (#45)
            saved_shape = self.shape if self.shape is not None else self.out.shape
            # Slice first, then clone only the small slice (#73)
            if len(saved_shape) == 0:
                tensor_slice = self.out.detach().clone()
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = self.out[0:num_dims].detach().clone()
            elif len(saved_shape) == 2:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.out[0:num_dims, 0:num_dims].detach().clone()
            else:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.out.data
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
        if len(self.parents) > 0:
            s += "\n\t\t- parent layers: " + ", ".join(self.parents)
        else:
            s += "\n\t\t- no parent layers"

        if len(self.children) > 0:
            s += "\n\t\t- child layers: " + ", ".join(self.children)
        else:
            s += "\n\t\t- no child layers"

        if len(self.siblings) > 0:
            s += "\n\t\t- shares parents with layers: " + ", ".join(self.siblings)
        else:
            s += "\n\t\t- shares parents with no other layers"

        if len(self.co_parents) > 0:
            s += "\n\t\t- shares children with layers: " + ", ".join(self.co_parents)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:
            s += "\n\t\t- descendent of input layers: " + ", ".join(self.input_ancestors)
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.has_output_descendant:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(self.output_descendants)
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

        return s

    def __repr__(self) -> str:
        return self.__str__()


# Backward-compatible alias: TensorLog was the original name for
# OpLog before the LayerLog aggregate class was introduced in PR #92.
TensorLog = OpLog
