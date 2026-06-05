"""Layer and LayerAccessor: aggregate per-layer metadata and dict-like accessor.

Layer groups one or more Op entries that represent the same
logical layer across recurrent ops.  For non-recurrent models (the
common case), every Layer wraps exactly one Op.

**Delegation pattern**: For single-pass layers, per-pass fields (out,
grad, step_index, etc.) are accessible directly on the Layer
via ``_single_pass_or_error()`` and ``__getattr__`` delegation to ``ops[0]``.
For multi-pass layers, accessing these fields raises ``ValueError`` (NOT
``AttributeError``) directing the user to ``layer_log.ops[N].field``.

Why ValueError instead of AttributeError: Python's property protocol treats
``AttributeError`` from a ``@property`` as "attribute doesn't exist" and falls
through to ``__getattr__``.  Using ``ValueError`` avoids this trap and gives
the user a clear error message.

**_build_layer_logs merge rules** (in postprocess/layer_log.py):
When merging multiple ops into one Layer, these aggregate fields are merged:
  - ``has_input_ancestor``: OR across ops
  - ``io_role``: character-level merge of "I", "O", "IO" strings
  - ``is_atomic_module``: OR across ops
  - ``is_in_conditional_body``: OR across ops
  - ``conditional_role_stacks`` / ``conditional_branch_stack_ops``:
    unique per-pass stack signatures and their pass numbers
  - ``conditional_arm_children`` and derived child views:
    pass-stripped ordered unions across ops
All other 78+ fields use the first pass's values only.
``output_of_modules`` / ``output_of_module_calls`` are NOT updated across ops
(correct because same-layer grouping requires identical structural position).
"""

import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

from .._deprecations import MISSING
from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from ..quantities import Bytes, Duration, Flops, Macs, as_bytes, as_flops, as_macs
from ._accessor_base import Accessor

if TYPE_CHECKING:
    import pandas as pd

    from .op import Op
    from .trace import Trace
    from .param import Param


class OpAccessor(Accessor["Op"]):
    """Scoped dict-like accessor for the Op entries owned by one Layer."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(self, ops: Dict[int, "Op"] | None = None) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        ops:
            Mapping from 1-based pass index to Op.
        """

        ops = ops or {}
        super().__init__(ops, item_list=[op for _, op in sorted(ops.items())])

    def __getitem__(self, key: int | str) -> "Op":
        """Return an Op by 0-based position or pass-qualified label."""

        if isinstance(key, int):
            return self._list[key]
        resolved = self._resolve_substring(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"Op '{key}' not found in scoped Layer ops.")

    def __setitem__(self, key: int, value: "Op") -> None:
        """Set an Op by 1-based pass index."""

        self._dict[key] = value
        self._list = [op for _, op in sorted(self._dict.items())]

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to an Op."""

        if isinstance(key, int):
            return -len(self._list) <= key < len(self._list)
        if isinstance(key, str):
            try:
                self[key]
            except KeyError:
                return False
            return True
        return False

    # This scoped accessor intentionally iterates call-index keys, unlike the generic
    # Trace-level accessors that iterate log values.
    def __iter__(self) -> Iterator[int]:  # type: ignore[override]
        """Iterate call-index keys."""

        return iter(self._dict)

    def get(self, key: int, default: "Op | None" = None) -> "Op | None":
        """Return an Op by call index, or default."""

        return self._dict.get(key, default)

    def _resolve_substring(self, key: str) -> "Op | None":
        """Resolve Op by any scoped layer-label variant."""
        if len(self._dict) == 1:
            only_op = next(iter(self._dict.values()))
            if key in {
                only_op.layer_label,
                only_op.layer_label_short,
                only_op.layer_label,
                only_op.layer_label_short,
            }:
                return only_op
        for op_log in self._dict.values():
            if key in {
                op_log.layer_label,
                op_log.label,
                op_log.layer_label,
                op_log.layer_label_short,
                op_log.label_short,
                op_log.layer_label_short,
            }:
                return op_log
        parent_matches = [
            op_log
            for op_log in self._dict.values()
            if key in {op_log.layer_label, op_log.layer_label_short}
        ]
        if len(parent_matches) > 1:
            parent_label = parent_matches[0].layer_label
            raise ValueError(
                f"Layer '{parent_label}' has {len(parent_matches)} ops. Use a 0-based "
                "integer position or a pass-qualified label like "
                f"'{parent_label}:1'."
            )
        return None


class Layer:
    """Aggregate per-layer metadata for a logged model operation.

    Groups one or more Op objects (one per invocation of this layer).
    For non-recurrent models, every Layer has exactly one pass.

    Aggregate fields (function identity, param identity, flags, module containment)
    live directly on Layer.  Per-pass fields (outs, graph edges,
    execution state, grads) live on the Op objects in ``self.ops``.

    For single-pass layers, per-pass fields are accessible directly via
    ``__getattr__`` delegation (e.g. ``layer_log.out`` transparently
    reads from ``ops[0].out``).
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "layer_type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "step_index": FieldPolicy.KEEP,
        "ordinal_index": FieldPolicy.KEEP,
        "raw_index": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "func": FieldPolicy.DROP,
        "func_name": FieldPolicy.KEEP,
        "func_qualname": FieldPolicy.KEEP,
        "is_inplace": FieldPolicy.KEEP,
        "grad_fn_class_name": FieldPolicy.KEEP,
        "grad_fn_class_qualname": FieldPolicy.KEEP,
        "grad_fn_object_id": FieldPolicy.KEEP,
        "grad_fn_handle": FieldPolicy.DROP,
        "grad_fn": FieldPolicy.DROP,
        "arg_names": FieldPolicy.KEEP,
        "num_args_total": FieldPolicy.KEEP,
        "num_pos_args": FieldPolicy.KEEP,
        "num_kwargs": FieldPolicy.KEEP,
        "in_multi_output": FieldPolicy.KEEP,
        "multi_output_index": FieldPolicy.KEEP,
        "multi_output_name": FieldPolicy.KEEP,
        "shape": FieldPolicy.KEEP,
        "transformed_out_shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "transformed_out_dtype": FieldPolicy.KEEP,
        "activation_memory": FieldPolicy.KEEP,
        "transformed_activation_memory": FieldPolicy.KEEP,
        "transformed_out": FieldPolicy.BLOB,
        "autograd_memory": FieldPolicy.KEEP,
        "total_autograd_memory": FieldPolicy.KEEP,
        "num_autograd_tensors": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "activation_transform": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "intervention_replaced": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "save_gradients": FieldPolicy.KEEP,
        "transformed_grad": FieldPolicy.BLOB,
        "transformed_grad_shape": FieldPolicy.KEEP,
        "transformed_grad_dtype": FieldPolicy.KEEP,
        "transformed_gradient_memory": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "_param_barcodes": FieldPolicy.KEEP,
        "_param_logs": FieldPolicy.KEEP,
        "param_shapes": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "total_param_memory": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "equivalence_class": FieldPolicy.KEEP,
        "equivalent_ops": FieldPolicy.KEEP,
        "is_input": FieldPolicy.KEEP,
        "is_output": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "is_buffer": FieldPolicy.KEEP,
        "address": FieldPolicy.KEEP,
        "buffer_source": FieldPolicy.KEEP,
        "buffer_write_kind": FieldPolicy.KEEP,
        "buffer_value_changed": FieldPolicy.KEEP,
        "buffer_replay_validated": FieldPolicy.KEEP,
        "buffer_source_func_name": FieldPolicy.KEEP,
        "is_internal_source": FieldPolicy.KEEP,
        "is_internal_sink": FieldPolicy.KEEP,
        "is_terminal_bool": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "bool_value": FieldPolicy.KEEP,
        "in_conditionals": FieldPolicy.KEEP,
        "terminal_bool_for": FieldPolicy.KEEP,
        "is_in_conditional_body": FieldPolicy.KEEP,
        "conditional_role_stacks": FieldPolicy.KEEP,
        "conditional_branch_stack_ops": FieldPolicy.KEEP,
        "conditional_arm_children": FieldPolicy.KEEP,
        "module": FieldPolicy.KEEP,
        "modules": FieldPolicy.KEEP,
        "output_of_modules": FieldPolicy.KEEP,
        "output_of_module_calls": FieldPolicy.KEEP,
        "conditional_entry_children": FieldPolicy.KEEP,
        "conditional_then_children": FieldPolicy.KEEP,
        "conditional_elif_children": FieldPolicy.KEEP,
        "conditional_else_children": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "is_atomic_module": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
        "call_labels": FieldPolicy.KEEP,
    }

    def __init__(self, first_pass: "Op") -> None:
        """Initialize from the first pass of this layer.

        Args:
            first_pass: The Op for pass 1 of this layer.
        """
        # Identity & labeling
        self.layer_label = first_pass.layer_label
        self.layer_label_short = first_pass.layer_label_short
        self.layer_type = first_pass.layer_type
        self.type_index = first_pass.type_index
        self.step_index = first_pass.step_index
        self.ordinal_index = first_pass.ordinal_index
        self.raw_index = first_pass.raw_index
        self.num_passes = first_pass.num_passes
        # Store as weakref to break circular reference (Trace -> layer_logs -> Layer -> Trace).
        _sml = first_pass.source_trace
        self._source_trace_ref: weakref.ReferenceType["Trace"] | None = (
            weakref.ref(_sml) if _sml is not None else None
        )

        # Function identity
        self.func = first_pass.func
        self.func_name = first_pass.func_name
        self.func_qualname = first_pass.func_qualname
        self.is_inplace = first_pass.is_inplace
        self.grad_fn_class_name = first_pass.grad_fn_class_name
        self.grad_fn_class_qualname = first_pass.grad_fn_class_qualname
        self.grad_fn_object_id = first_pass.grad_fn_object_id
        self.grad_fn_handle = first_pass.grad_fn_handle
        self.grad_fn = first_pass.grad_fn
        self.arg_names = first_pass.arg_names
        self.num_args_total = first_pass.num_args_total
        self.num_pos_args = first_pass.num_pos_args
        self.num_kwargs = first_pass.num_kwargs
        self.in_multi_output = first_pass.in_multi_output
        self.multi_output_index = first_pass.multi_output_index
        self.multi_output_name = first_pass.multi_output_name

        # Tensor type (representative from first pass)
        self.shape = first_pass.shape
        self.transformed_out_shape = first_pass.transformed_out_shape
        self.dtype = first_pass.dtype
        self.transformed_out_dtype = first_pass.transformed_out_dtype
        self.activation_memory: Bytes | None = as_bytes(first_pass.activation_memory)
        self.transformed_activation_memory: Bytes | None = as_bytes(
            first_pass.transformed_activation_memory
        )
        self.autograd_memory: Bytes | None = as_bytes(first_pass.autograd_memory)
        self.total_autograd_memory: Bytes | None = as_bytes(first_pass.autograd_memory)
        self.num_autograd_tensors: Optional[int] = first_pass.num_autograd_tensors

        # Config
        self.output_device = first_pass.output_device
        self.activation_transform = first_pass.activation_transform
        self.annotations: Dict[str, Any] = {}
        self.intervention_replaced = first_pass.intervention_replaced
        self.detach_saved_activations = first_pass.detach_saved_activations
        self.save_gradients = first_pass.save_gradients
        self.transformed_grad_shape = first_pass.transformed_grad_shape
        self.transformed_grad_dtype = first_pass.transformed_grad_dtype
        self.transformed_gradient_memory: Bytes | None = as_bytes(
            first_pass.transformed_gradient_memory
        )

        # FLOPs
        self.flops_forward = as_flops(first_pass.flops_forward)
        self.flops_backward = as_flops(first_pass.flops_backward)

        # Param identity
        self._param_barcodes = first_pass._param_barcodes
        self._param_logs: List["Param"] = first_pass._param_logs
        self.param_shapes = first_pass.param_shapes
        self.num_params = first_pass.num_params
        self.num_params_trainable = first_pass.num_params_trainable
        self.num_params_frozen = first_pass.num_params_frozen
        self.total_param_memory: Bytes = Bytes(first_pass.param_memory or 0)

        # Function config
        self.func_config = first_pass.func_config

        # Equivalence
        self.equivalence_class = first_pass.equivalence_class
        self.equivalent_ops = first_pass.equivalent_ops

        # Special flags
        self.is_input = first_pass.is_input
        self.is_output = first_pass.is_output
        self.is_final_output = first_pass.is_final_output
        self.is_buffer = first_pass.is_buffer
        self.address = first_pass.address
        self.buffer_source = first_pass.buffer_source
        self.buffer_write_kind = first_pass.buffer_write_kind
        self.buffer_value_changed = first_pass.buffer_value_changed
        self.buffer_replay_validated = first_pass.buffer_replay_validated
        self.buffer_source_func_name = first_pass.buffer_source_func_name
        self.is_internal_source = first_pass.is_internal_source
        self.is_internal_sink = first_pass.is_internal_sink
        self.is_terminal_bool = first_pass.is_terminal_bool
        self.is_scalar_bool = first_pass.is_scalar_bool
        self.bool_value = first_pass.bool_value
        self.in_conditionals = first_pass.in_conditionals
        self.terminal_bool_for = first_pass.terminal_bool_for
        self.is_in_conditional_body = first_pass.is_in_conditional_body
        self.conditional_role_stacks: List[List[Tuple[int, str]]] = []
        self.conditional_branch_stack_ops: Dict[Tuple[Tuple[int, str], ...], List[int]] = {}
        self.conditional_arm_children: Dict[int, Dict[str, List[str]]] = {}

        # Module (static containment)
        self.module = first_pass.module
        self.modules = first_pass.modules

        # Fields stored as aggregate for vis compatibility.
        # Initialized from first pass.  For multi-pass layers, _build_layer_logs
        # merges only has_input_ancestor (OR), io_role (char-merge),
        # and is_atomic_module (OR).  All others keep first-pass values.
        self.output_of_modules = first_pass.output_of_modules
        self.output_of_module_calls = first_pass.output_of_module_calls
        self.conditional_entry_children = first_pass.conditional_entry_children
        self.conditional_then_children = first_pass.conditional_then_children
        self.conditional_elif_children = first_pass.conditional_elif_children
        self.conditional_else_children = first_pass.conditional_else_children
        self.has_input_ancestor = first_pass.has_input_ancestor
        self.io_role = first_pass.io_role
        self.buffer_pass = first_pass.buffer_pass
        self.is_atomic_module = first_pass.is_atomic_module

        # Pass management
        self.ops = OpAccessor()
        self.call_labels: List[str] = []

    @property
    def macs_forward(self) -> Optional[Macs]:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_forward // 2 if self.flops_forward is not None else None)

    @property
    def macs_backward(self) -> Optional[Macs]:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return as_macs(self.flops_backward // 2 if self.flops_backward is not None else None)

    @property
    def total_activation_memory(self) -> Bytes:
        """Sum activation memory across all Ops in this Layer."""

        return Bytes(sum(int(op.activation_memory or 0) for op in self.ops.values()))

    @property
    def total_gradient_memory(self) -> Bytes:
        """Sum gradient memory across all Ops in this Layer."""

        return Bytes(sum(int(op.gradient_memory or 0) for op in self.ops.values()))

    @property
    def flops_total(self) -> Flops:
        """Representative total FLOPs for this Layer."""

        return Flops((self.flops_forward or 0) + (self.flops_backward or 0))

    @property
    def total_flops_forward(self) -> Flops:
        """Sum forward FLOPs across all Ops in this Layer."""

        return Flops(sum((op.flops_forward or 0) for op in self.ops.values()))

    @property
    def total_flops_backward(self) -> Flops:
        """Sum backward FLOPs across all Ops in this Layer."""

        return Flops(sum((op.flops_backward or 0) for op in self.ops.values()))

    @property
    def total_flops_total(self) -> Flops:
        """Sum total FLOPs across all Ops in this Layer."""

        return Flops(self.total_flops_forward + self.total_flops_backward)

    @property
    def macs_total(self) -> Macs:
        """Representative total MACs for this Layer."""

        return Macs(self.flops_total // 2)

    @property
    def total_macs_forward(self) -> Macs:
        """Sum forward MACs across all Ops in this Layer."""

        return Macs(self.total_flops_forward // 2)

    @property
    def total_macs_backward(self) -> Macs:
        """Sum backward MACs across all Ops in this Layer."""

        return Macs(self.total_flops_backward // 2)

    @property
    def total_macs_total(self) -> Macs:
        """Sum total MACs across all Ops in this Layer."""

        return Macs(self.total_flops_total // 2)

    @property
    def param_names(self) -> list[str]:
        """Return short names of parameters used by this Layer."""

        return [param.name for param in self._param_logs]

    @property
    def param_dtypes(self) -> list[Any]:
        """Return dtypes of parameters used by this Layer."""

        return [param.dtype for param in self._param_logs]

    @property
    def uses_params(self) -> bool:
        """Whether this layer uses model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this layer."""
        return len(self._param_barcodes)

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors used by this Layer."""

        return sum(1 for param in self._param_logs if param.is_trainable)

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors used by this Layer."""

        return sum(1 for param in self._param_logs if not param.is_trainable)

    @property
    def has_trainable_params(self) -> bool:
        """Whether this Layer uses at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self) -> bool:
        """Whether this Layer uses at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def is_compute_layer(self) -> bool:
        """Whether this Layer's representative Op is a compute Op."""

        return bool(self.ops and self.ops[0].is_compute_op)

    @property
    def is_orphan(self) -> bool:
        """Whether any Op in this Layer is disconnected from the main graph."""

        return any(op.is_orphan for op in self.ops.values())

    @property
    def num_ops(self) -> int:
        """Number of Ops aggregated by this Layer."""

        return len(self.ops)

    @property
    def fx_label(self) -> str | None:
        """Return a torch.fx-style label for a single-Op Layer."""

        return cast(str | None, self._single_pass_or_error("fx_label"))

    @property
    def fx_qualpath(self) -> str | None:
        """Return the FX-style qualified path for a single-Op Layer."""

        return cast(str | None, self._single_pass_or_error("fx_qualpath"))

    @property
    def fx_call_index(self) -> int:
        """Return the FX-style call index for a single-Op Layer."""

        return cast(int, self._single_pass_or_error("fx_call_index"))

    @property
    def in_submodule(self) -> bool:
        """Whether this layer was computed inside a submodule."""
        return self.module is not None

    @property
    def module_call_depth(self) -> int:
        """Depth of module nesting for this layer."""
        return len(self.modules)

    @property
    def op_labels(self) -> List[str]:
        """Op labels belonging to this Layer (glossary name for ``call_labels``)."""

        return self.call_labels

    @property
    def is_buffer_source(self) -> bool:
        """Whether this Layer represents a buffer overwrite boundary.

        Glossary name for the stored ``is_buffer`` flag; aggregate over the
        Layer (true when its representative Op is a buffer source).
        """

        return bool(self.is_buffer)

    @property
    def buffer_overwrite_index(self) -> Any:
        """Which overwrite of the buffer this Layer represents.

        Glossary name for the stored ``buffer_pass`` index.
        """

        return self.buffer_pass

    @property
    def is_module_input(self) -> bool:
        """Whether this Layer's representative Op feeds into at least one ModuleCall.

        Delegates from ``Op.is_module_input`` semantics for single-Op Layers.
        Raises ``ValueError`` for multi-Op Layers.
        """

        return cast(bool, self._single_pass_or_error("is_module_input"))

    @property
    def source_trace(self) -> "Trace":
        """Back-reference to the owning Trace (stored as weakref)."""
        ref = self.__dict__.get("_source_trace_ref")
        if ref is None:
            return None  # type: ignore[return-value]
        obj = ref()
        if obj is None:
            raise RuntimeError("Trace has been garbage-collected.")
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

    @property
    def trace(self) -> "Trace":
        """Alias for the owning Trace back-reference."""

        return self.source_trace

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        read_tlspec_version(state, cls_name=type(self).__name__)
        if "ops" not in state and "passes" in state:
            state["ops"] = state.pop("passes")
        if "call_labels" not in state and "pass_labels" in state:
            state["call_labels"] = state.pop("pass_labels")
        default_fill_state(
            state,
            defaults={
                "_source_trace_ref": None,
                "annotations": {},
                "autograd_memory": None,
                "total_autograd_memory": None,
                "num_autograd_tensors": None,
                "transformed_out": None,
                "transformed_out_shape": None,
                "transformed_out_dtype": None,
                "transformed_activation_memory": None,
                "transformed_grad": None,
                "transformed_grad_shape": None,
                "transformed_grad_dtype": None,
                "transformed_gradient_memory": None,
            },
        )
        if "activation_memory" not in state and "memory" in state:
            state["activation_memory"] = state.pop("memory")
        for field_name in (
            "activation_memory",
            "transformed_activation_memory",
            "autograd_memory",
            "total_autograd_memory",
            "transformed_gradient_memory",
            "total_param_memory",
        ):
            if state.get(field_name) is not None:
                state[field_name] = Bytes(state[field_name])
        self.__dict__.update(state)

    # ********************************************
    # ******* Single-pass delegation *************
    # ********************************************
    # For single-pass layers, per-pass fields are transparently accessible
    # on the Layer itself.  For multi-pass layers, attempting to access
    # these fields raises ValueError directing the user to a specific pass.

    def _single_pass_or_error(self, field_name: str) -> Any:
        """Access a per-pass field, requiring exactly one pass.

        Raises ValueError (not AttributeError) for multi-pass layers.
        Using ValueError avoids the Python property/__getattr__ trap:
        if a @property raises AttributeError, Python silently treats
        the attribute as missing and falls through to __getattr__.
        """
        if self.num_passes > 1:
            raise ValueError(
                f"Layer '{self.layer_label}' has {self.num_passes} ops. "
                f"Access '{field_name}' on a specific pass: "
                f"log['{self.layer_label}'].ops[0].{field_name}"
            )
        return getattr(self.ops[0], field_name)

    @property
    def out(self) -> Any:
        """Return the saved out for a single-pass layer.

        Returns
        -------
        Any
            Saved out from the only pass.
        """
        return self._single_pass_or_error("out")

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved out on single-pass layers."""

        return self._single_pass_or_error("tensor")

    @property
    def transformed_out(self) -> Any:
        """Transformed out on single-pass layers."""

        return self._single_pass_or_error("transformed_out")

    @property
    def has_saved_activation(self) -> bool:
        """Return whether the single pass has a saved out.

        Returns
        -------
        bool
            ``True`` when an out was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_saved_activation"))

    @property
    def saved_args(self) -> Any:
        """Return captured positional arguments for a single-pass layer.

        Returns
        -------
        Any
            Captured positional arguments from the only pass.
        """
        return self._single_pass_or_error("saved_args")

    @property
    def saved_kwargs(self) -> Any:
        """Return captured keyword arguments for a single-pass layer.

        Returns
        -------
        Any
            Captured keyword arguments from the only pass.
        """
        return self._single_pass_or_error("saved_kwargs")

    @property
    def grad(self) -> Any:
        """Return the saved grad for a single-pass layer.

        Returns
        -------
        Any
            Saved grad from the only pass.
        """
        return self._single_pass_or_error("grad")

    @property
    def transformed_grad(self) -> Any:
        """Transformed grad on single-pass layers."""

        return self._single_pass_or_error("transformed_grad")

    @property
    def has_grad(self) -> bool:
        """Return whether the single pass has a saved grad.

        Returns
        -------
        bool
            ``True`` when a grad was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_grad"))

    @property
    def code_context(self) -> Any:
        """Return the captured call stack for a single-pass layer.

        Returns
        -------
        Any
            Function call stack from the only pass.
        """
        return self._single_pass_or_error("code_context")

    @property
    def func_duration(self) -> Duration:
        """Return function execution time for a single-pass layer.

        Returns
        -------
        Duration
            Function timing value from the only pass.
        """
        return cast(Duration, self._single_pass_or_error("func_duration"))

    @property
    def total_func_duration(self) -> Duration:
        """Sum of function-call duration across all Ops in this Layer."""

        return Duration(sum(float(op.func_duration or 0) for op in self.ops.values()))

    @property
    def func_rng_states(self) -> Any:
        """Return RNG states captured for a single-pass layer.

        Returns
        -------
        Any
            RNG state snapshot from the only pass.
        """
        return self._single_pass_or_error("func_rng_states")

    @property
    def pass_index(self) -> int:
        """Return the pass number for a single-pass layer.

        Returns
        -------
        int
            Pass number from the only pass.
        """
        return cast(int, self._single_pass_or_error("pass_index"))

    @property
    def lookup_keys(self) -> list[str]:
        """Return lookup keys for a single-pass layer.

        Returns
        -------
        list[str]
            Lookup keys from the only pass.
        """
        return cast(list[str], self._single_pass_or_error("lookup_keys"))

    # ********************************************
    # ***** Aggregate graph properties ***********
    # ********************************************
    # Graph-edge properties compute the union across all ops, returning
    # no-pass labels (i.e. Layer-level identifiers).  This gives a
    # complete picture of which layers are connected across all recurrent
    # iterations.  Order is preserved (first-seen insertion order).

    @property
    def children(self) -> list[str]:
        """Union of child layers (no-pass labels) across all ops."""
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.children:
                no_pass = self.source_trace[label].layer_label
                if no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def parents(self) -> list[str]:
        """Union of parent layers (no-pass labels) across all ops."""
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.parents:
                no_pass = self.source_trace[label].layer_label
                if no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def has_children(self) -> bool:
        """Return whether any pass has child layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph children.
        """
        return any(p.has_children for p in self.ops.values())

    @property
    def has_parents(self) -> bool:
        """Return whether any pass has parent layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph parents.
        """
        return any(p.has_parents for p in self.ops.values())

    @property
    def num_parents(self) -> int:
        """Number of distinct parent Layers feeding this Layer."""

        return len(self.parents)

    @property
    def num_children(self) -> int:
        """Number of distinct child Layers fed by this Layer."""

        return len(self.children)

    @property
    def siblings(self) -> list[str]:
        """Union of sibling layers (no-pass labels) across all ops."""
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.siblings:
                no_pass = self.source_trace[label].layer_label
                if no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def has_siblings(self) -> bool:
        """Return whether any pass has sibling layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph siblings.
        """
        return any(p.has_siblings for p in self.ops.values())

    @property
    def co_parents(self) -> list[str]:
        """Union of spouse layers (no-pass labels) across all ops."""
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.co_parents:
                no_pass = self.source_trace[label].layer_label
                if no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def has_co_parents(self) -> bool:
        """Return whether any pass has co-parent layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph co-parents.
        """
        return any(p.has_co_parents for p in self.ops.values())

    @property
    def is_in_conditional(self) -> bool:
        """Whether this layer participates in any conditional role."""

        return bool(self.in_conditionals)

    @property
    def is_in_conditional_evaluation(self) -> bool:
        """Whether this layer computes a conditional arm condition."""

        return any(role.role == "evaluation" for role in self.in_conditionals or [])

    @property
    def is_in_conditional_body(self) -> bool:
        """Whether this layer is in a conditional arm body."""

        if getattr(self, "has_output_descendant", False) and not self.conditional_entry_children:
            return False
        return bool(self.__dict__.get("_is_in_conditional_body", False)) or any(
            role.role == "body" for role in self.in_conditionals or []
        )

    @is_in_conditional_body.setter
    def is_in_conditional_body(self, value: bool) -> None:
        """Set the cached conditional-body predicate used during aggregation."""

        self.__dict__["_is_in_conditional_body"] = value

    @is_in_conditional_body.deleter
    def is_in_conditional_body(self) -> None:
        """Delete the cached conditional-body predicate during cleanup."""

        self.__dict__.pop("_is_in_conditional_body", None)

    @property
    def conditional_depth(self) -> int:
        """Number of distinct conditionals this layer participates in."""

        return len({role.conditional_id for role in self.in_conditionals or []})

    @property
    def _tracing_finished(self) -> bool:
        sml = self.source_trace
        if sml is None:
            return True
        return sml._tracing_finished

    # ********************************************
    # ****** Convenience properties **************
    # ********************************************

    @property
    def label(self) -> str:
        """For single-pass layers, return the pass-qualified label."""
        return cast(str, self._single_pass_or_error("label"))

    @property
    def label_short(self) -> str:
        """For single-pass layers, return the short pass-qualified label."""
        return cast(str, self._single_pass_or_error("label_short"))

    @property
    def params(self) -> Any:
        """Access parameter metadata by address, short name, or index."""
        from .param import ParamAccessor

        param_dict = {pl.address: pl for pl in self._param_logs}
        return ParamAccessor(param_dict)

    # ********************************************
    # **** Rolled-vis computed properties ********
    # ********************************************
    # These provide per-pass edge tracking for rolled (recurrence-aware)
    # graph visualization.  Computed on-the-fly from the ops dict.
    # Used by the visualization renderer to draw pass-annotated edges.

    @property
    def children_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: child layer labels (no-pass) for each pass."""
        result = {}
        for call_index, pass_log in self.ops.items():
            children = []
            for label in pass_log.children:
                no_pass = self.source_trace[label].layer_label
                if no_pass not in children:
                    children.append(no_pass)
            result[call_index] = children
        return result

    @property
    def parents_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: parent layer labels (no-pass) for each pass."""
        result = {}
        for call_index, pass_log in self.ops.items():
            parents = []
            for label in pass_log.parents:
                no_pass = self.source_trace[label].layer_label
                if no_pass not in parents:
                    parents.append(no_pass)
            result[call_index] = parents
        return result

    @property
    def child_ops_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each child layer, which ops connect to it."""
        from collections import defaultdict

        result: defaultdict[str, list[int]] = defaultdict(list)
        for call_index, pass_log in self.ops.items():
            for label in pass_log.children:
                no_pass = self.source_trace[label].layer_label
                if call_index not in result[no_pass]:
                    result[no_pass].append(call_index)
        return dict(result)

    @property
    def parent_ops_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each parent layer, which ops connect from it."""
        from collections import defaultdict

        result: defaultdict[str, list[int]] = defaultdict(list)
        for call_index, pass_log in self.ops.items():
            for label in pass_log.parents:
                no_pass = self.source_trace[label].layer_label
                if call_index not in result[no_pass]:
                    result[no_pass].append(call_index)
        return dict(result)

    @property
    def edges_vary_across_ops(self) -> bool:
        """Whether graph edges differ across ops."""
        if self.num_passes <= 1:
            return False
        all_pass_lists = list(self.child_ops_per_layer.values()) + list(
            self.parent_ops_per_layer.values()
        )
        return any(len(ops) < self.num_passes for ops in all_pass_lists)

    @property
    def leaf_module_ops(self) -> set[Any]:
        """Set of module ops exited across all ops."""
        result = set()
        for pass_log in self.ops.values():
            if pass_log.is_atomic_module:
                result.add(pass_log.atomic_module_call)
        return result

    @property
    def parent_arg_positions(self) -> dict[str, dict[Any, str]]:
        """Merged parent_arg_positions across ops (set-union).

        For single-pass layers, delegates to ops[0].
        For multi-pass, merges arg locs using set-union of no-pass labels.
        """
        if self.num_passes == 1:
            return cast(dict[str, dict[Any, str]], self.ops[0].parent_arg_positions)
        from collections import defaultdict

        result: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
        for pass_log in self.ops.values():
            for arg_type in ["args", "kwargs"]:
                for arg_key, layer_label in pass_log.parent_arg_positions[arg_type].items():
                    no_pass = self.source_trace[layer_label].layer_label
                    if arg_key not in result[arg_type]:
                        result[arg_type][arg_key] = no_pass
        return result

    # ********************************************
    # ******* Fallback __getattr__ ***************
    # ********************************************

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute lookup: delegates to ops[0] for single-pass layers.

        Only called when normal attribute lookup has already failed (Python's
        ``__getattr__`` protocol).  For single-pass layers, transparently
        forwards to the underlying Op, enabling code like
        ``layer_log.func_rng_states`` without needing an explicit property.

        Private attributes (starting with '_') are never delegated — they
        raise AttributeError immediately to avoid infinite recursion with
        ``self.__dict__`` access.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        if name in {
            "param_memory",
        }:
            raise AttributeError(name)
        ops = self.__dict__.get("ops")
        if ops and len(ops) == 1:
            try:
                return getattr(ops[0], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ********************************************
    # ************ User-facing custom_methods ***********
    # ********************************************

    def get_children(self) -> list["Layer"]:
        """Return child Layer objects for this layer.

        Returns
        -------
        list[Layer]
            Child layers resolved through the owning model log.
        """
        return [self.source_trace[child_label] for child_label in self.children]

    def get_parents(self) -> list["Layer"]:
        """Return parent Layer objects for this layer.

        Returns
        -------
        list[Layer]
            Parent layers resolved through the owning model log.
        """
        return [self.source_trace[parent_label] for parent_label in self.parents]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this layer's saved out.

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

    def _source_trace_or_error(self) -> "Trace":
        """Return the owning Trace, or raise a detached-log error.

        Returns
        -------
        Trace
            Source Trace that owns this layer log.

        Raises
        ------
        AttributeError
            If this layer log is detached from its source Trace.
        """

        ref = self.__dict__.get("_source_trace_ref")
        source = ref() if ref is not None else None
        if source is None or getattr(source, "_loaded_from_bundle", False):
            raise AttributeError(
                "This Layer is detached from its source Trace "
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
        """Apply an intervention to this layer through the owning Trace.

        Parameters
        ----------
        transform:
            Transform or hook to apply to this layer's output.
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
        """Set this layer's out recipe through the owning Trace.

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
        """Attach sticky hooks to this layer through the owning Trace.

        Parameters
        ----------
        hook:
            Hook or helper to attach to this layer.
        *extra_hooks:
            Additional hooks to compose on this layer in left-to-right order.
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

    def to_pandas(self) -> "pd.DataFrame":
        """Export this Layer as a one-row pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame ordered by ``LAYER_LOG_FIELD_ORDER``.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e
        from ..constants import LAYER_LOG_FIELD_ORDER

        row = {field_name: getattr(self, field_name) for field_name in LAYER_LOG_FIELD_ORDER}
        return pd.DataFrame([row], columns=LAYER_LOG_FIELD_ORDER)

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        if not self._tracing_finished:
            return f"Layer({self.layer_label}) (pass not finished)"
        s = f"Layer {self.layer_label}:"
        if self.num_passes > 1:
            s += f" ({self.num_passes} ops)"
        s += f"\n\tOutput tensor: shape={self.shape}, dtype={self.dtype}, size={self.activation_memory}"
        if not self.is_input:
            s += f"\n\tFunction: {self.func_name} (grad_fn_handle: {self.grad_fn_class_name})"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
        if self.module is not None:
            s += f"\n\tComputed inside module: {self.module}"
        if len(self.param_shapes) > 0:
            params_shapes_str = ", ".join(str(ps) for ps in self.param_shapes)
            s += (
                f"\n\tParams: {params_shapes_str}; "
                f"{self.num_params} total ({self.total_param_memory})"
            )
        s += "\n\tRelated Layers:"
        s += f"\n\t\t- parents: {', '.join(self.parents) or 'none'}"
        s += f"\n\t\t- children: {', '.join(self.children) or 'none'}"
        if self.num_passes > 1:
            s += f"\n\tPasses: {', '.join(self.call_labels)}"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return cast(int, self.num_passes)


class LayerAccessor(Accessor["Layer"]):
    """Dict-like accessor for Layer objects.

    Supports indexing by:
    * **layer label** (str) -- exact match against no-pass label.
    * **ordinal index** (int) -- position in execution order.
    * **pass notation** (str ``"conv2d_1_1:2"``) -- strips the pass
      suffix and returns the parent Layer.

    Available as ``trace.layers``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        layer_logs: Dict[str, "Layer"],
        source_trace: Optional["Trace"] = None,
    ) -> None:
        source_ref = weakref.ref(source_trace) if source_trace is not None else None
        super().__init__(layer_logs, source_ref=source_ref)

    def _resolve_pass_qualified(self, key: str) -> "Layer | None":
        """Resolve ``layer_label:pass`` notation to the parent Layer."""
        base, _, pass_str = key.rpartition(":")
        try:
            int(pass_str)
        except ValueError:
            return None
        return self._resolve_substring(base)

    def _resolve_substring(self, key: str) -> "Layer | None":
        """Resolve exact long or short Layer labels."""
        if key in self._dict:
            return self._dict[key]
        matches = [
            layer
            for layer in self._list
            if key in {layer.layer_label, layer.layer_label, layer.layer_label_short}
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Layer lookup '{key}' is ambiguous across {len(matches)} Layers. "
                "Use the full Layer label."
            )
        return None

    def _suggest(self, key: str) -> list[str]:
        """Return similar layer labels from the source Trace."""
        source_ref = getattr(self, "_source_ref", None)
        source = source_ref() if source_ref is not None else None
        if source is not None and hasattr(source, "find_layers"):
            return source.find_layers(str(key))
        return []

    def by_operator(self, operator: str | None = None) -> Dict[str, int] | List[str]:
        """Group layers by Torch operator name.

        Parameters
        ----------
        operator:
            Optional operator name. When supplied, matching layer labels are returned.

        Returns
        -------
        Dict[str, int] | List[str]
            Counts by operator, or labels for one operator.
        """

        if operator is not None:
            return [
                layer.layer_label
                for layer in self._list
                if (layer.func_name or layer.layer_type) == operator
            ]
        counts: Dict[str, int] = {}
        for layer in self._list:
            key = str(layer.func_name or layer.layer_type)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_module(self, module: str | None = None) -> Dict[str, int] | List[str]:
        """Group layers by containing module address.

        Parameters
        ----------
        module:
            Optional module address. When supplied, matching layer labels are returned.

        Returns
        -------
        Dict[str, int] | List[str]
            Counts by module, or labels for one module.
        """

        if module is not None:
            return [
                layer.layer_label
                for layer in self._list
                if layer.module == module or module in getattr(layer, "modules", [])
            ]
        counts: Dict[str, int] = {}
        for layer in self._list:
            key = str(layer.module or "self")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def by_module_and_operator(
        self,
        module: str | None = None,
        operator: str | None = None,
    ) -> Dict[Tuple[str, str], int] | List[str]:
        """Group layers by module and operator.

        Parameters
        ----------
        module:
            Optional module address filter.
        operator:
            Optional operator-name filter.

        Returns
        -------
        Dict[Tuple[str, str], int] | List[str]
            Counts by ``(module, operator)`` or labels matching both filters.
        """

        if module is not None and operator is not None:
            return [
                layer.layer_label
                for layer in self._list
                if (layer.module == module or module in getattr(layer, "modules", []))
                and (layer.func_name or layer.layer_type) == operator
            ]
        counts: Dict[Tuple[str, str], int] = {}
        for layer in self._list:
            key = (str(layer.module or "self"), str(layer.func_name or layer.layer_type))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def total(self) -> int:
        """Return the number of aggregate layers.

        Returns
        -------
        int
            Number of layer logs.
        """

        return len(self)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "LayerAccessor({})"
        items = []
        for ll in self._list:
            items.append(
                f"  '{ll.layer_label}': {ll.func_name or 'input'} "
                f"(shape={list(ll.shape) if ll.shape else '?'}, "
                f"ops={ll.num_passes})"
            )
        inner = "\n".join(items)
        return f"LayerAccessor({len(self)} layers):\n{inner}"

    def to_pandas(self) -> "pd.DataFrame":
        """One row per unique layer (aggregate view)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        rows = []
        for ll in self._list:
            rows.append(
                {
                    "layer_label": ll.layer_label,
                    "layer_type": ll.layer_type,
                    "func_name": ll.func_name,
                    "shape": ll.shape,
                    "dtype": ll.dtype,
                    "activation_memory": ll.activation_memory,
                    "num_passes": ll.num_passes,
                    "num_params": ll.num_params,
                    "module": ll.module,
                    "is_input": ll.is_input,
                    "is_output": ll.is_output,
                    "is_buffer": ll.is_buffer,
                }
            )
        return pd.DataFrame(rows)
