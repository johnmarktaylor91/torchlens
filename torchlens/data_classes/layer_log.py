"""LayerLog and LayerAccessor: aggregate per-layer metadata and dict-like accessor.

LayerLog groups one or more OpLog entries that represent the same
logical layer across recurrent ops.  For non-recurrent models (the
common case), every LayerLog wraps exactly one OpLog.

**Delegation pattern**: For single-pass layers, per-pass fields (out,
grad, compute_index, etc.) are accessible directly on the LayerLog
via ``_single_pass_or_error()`` and ``__getattr__`` delegation to ``ops[1]``.
For multi-pass layers, accessing these fields raises ``ValueError`` (NOT
``AttributeError``) directing the user to ``layer_log.ops[N].field``.

Why ValueError instead of AttributeError: Python's property protocol treats
``AttributeError`` from a ``@property`` as "attribute doesn't exist" and falls
through to ``__getattr__``.  Using ``ValueError`` avoids this trap and gives
the user a clear error message.

**_build_layer_logs merge rules** (in postprocess/layer_log.py):
When merging multiple ops into one LayerLog, these aggregate fields are merged:
  - ``has_input_ancestor``: OR across ops
  - ``io_role``: character-level merge of "I", "O", "IO" strings
  - ``is_atomic_module_op``: OR across ops
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
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

from .._deprecations import MISSING
from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..utils.display import human_readable_size
from ._accessor_base import Accessor

if TYPE_CHECKING:
    import pandas as pd

    from .op_log import OpLog
    from .model_log import Trace
    from .param_log import ParamLog


class OpAccessor(Accessor["OpLog"]):
    """Scoped dict-like accessor for the OpLog entries owned by one LayerLog."""

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(self, ops: Dict[int, "OpLog"] | None = None) -> None:
        """Initialize the accessor.

        Parameters
        ----------
        ops:
            Mapping from 1-based call index to OpLog.
        """

        super().__init__(ops or {})

    def __getitem__(self, key: int | str) -> "OpLog":
        """Return an OpLog by call index or pass-qualified label."""

        if isinstance(key, int):
            return self._dict[key]
        resolved = self._resolve_substring(key)
        if resolved is not None:
            return resolved
        raise KeyError(f"Op '{key}' not found in scoped LayerLog ops.")

    def __setitem__(self, key: int, value: "OpLog") -> None:
        """Set an OpLog by call index."""

        self._dict[key] = value

    def __contains__(self, key: object) -> bool:
        """Return whether key resolves to an OpLog."""

        if isinstance(key, int):
            return key in self._dict
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

    def get(self, key: int, default: "OpLog | None" = None) -> "OpLog | None":
        """Return an OpLog by call index, or default."""

        return self._dict.get(key, default)

    def _resolve_substring(self, key: str) -> "OpLog | None":
        """Resolve OpLog by any scoped layer-label variant."""
        for op_log in self._dict.values():
            if key in {
                op_log.layer_label,
                op_log.layer_label_w_pass,
                op_log.layer_label_no_pass,
                op_log.layer_label_short,
                op_log.layer_label_w_pass_short,
                op_log.layer_label_no_pass_short,
            }:
                return op_log
        return None


class LayerLog:
    """Aggregate per-layer metadata for a logged model operation.

    Groups one or more OpLog objects (one per invocation of this layer).
    For non-recurrent models, every LayerLog has exactly one pass.

    Aggregate fields (function identity, param identity, flags, module containment)
    live directly on LayerLog.  Per-pass fields (outs, graph edges,
    execution state, grads) live on the OpLog objects in ``self.ops``.

    For single-pass layers, per-pass fields are accessible directly via
    ``__getattr__`` delegation (e.g. ``layer_log.out`` transparently
    reads from ``ops[1].out``).
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_is_in_conditional_body": FieldPolicy.KEEP,
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "layer_type": FieldPolicy.KEEP,
        "type_index": FieldPolicy.KEEP,
        "trace_index": FieldPolicy.KEEP,
        "num_calls": FieldPolicy.KEEP,
        "_source_trace_ref": FieldPolicy.WEAKREF_STRIP,
        "func": FieldPolicy.DROP,
        "func_name": FieldPolicy.KEEP,
        "is_inplace": FieldPolicy.KEEP,
        "grad_fn_name": FieldPolicy.KEEP,
        "grad_fn_id": FieldPolicy.KEEP,
        "grad_fn": FieldPolicy.DROP,
        "grad_fn_log": FieldPolicy.DROP,
        "arg_names": FieldPolicy.KEEP,
        "num_args_total": FieldPolicy.KEEP,
        "num_pos_args": FieldPolicy.KEEP,
        "num_kwargs": FieldPolicy.KEEP,
        "is_part_of_iterable_output": FieldPolicy.KEEP,
        "multi_output_index": FieldPolicy.KEEP,
        "multi_output_role": FieldPolicy.KEEP,
        "shape": FieldPolicy.KEEP,
        "transformed_out_shape": FieldPolicy.KEEP,
        "dtype": FieldPolicy.KEEP,
        "transformed_out_dtype": FieldPolicy.KEEP,
        "memory": FieldPolicy.KEEP,
        "transformed_out_memory": FieldPolicy.KEEP,
        "transformed_out": FieldPolicy.BLOB,
        "autograd_saved_memory": FieldPolicy.KEEP,
        "num_autograd_saved_tensors": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "out_postfunc": FieldPolicy.DROP,
        "annotations": FieldPolicy.KEEP,
        "intervention_replaced": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "save_grads": FieldPolicy.KEEP,
        "transformed_grad": FieldPolicy.BLOB,
        "transformed_grad_shape": FieldPolicy.KEEP,
        "transformed_grad_dtype": FieldPolicy.KEEP,
        "transformed_grad_memory": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "_param_barcodes": FieldPolicy.KEEP,
        "_param_logs": FieldPolicy.KEEP,
        "param_shapes": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "param_memory": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "equivalence_class": FieldPolicy.KEEP,
        "equivalent_ops": FieldPolicy.KEEP,
        "is_input": FieldPolicy.KEEP,
        "is_output": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "is_buffer": FieldPolicy.KEEP,
        "buffer_address": FieldPolicy.KEEP,
        "buffer_parent": FieldPolicy.KEEP,
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
        "is_atomic_module_op": FieldPolicy.KEEP,
        "ops": FieldPolicy.KEEP,
        "call_labels": FieldPolicy.KEEP,
    }

    def __init__(self, first_pass: "OpLog") -> None:
        """Initialize from the first pass of this layer.

        Args:
            first_pass: The OpLog for pass 1 of this layer.
        """
        # Identity & labeling
        self.layer_label = first_pass.layer_label_no_pass
        self.layer_label_short = first_pass.layer_label_no_pass_short
        self.layer_type = first_pass.layer_type
        self.type_index = first_pass.type_index
        self.trace_index = first_pass.trace_index
        self.num_calls = first_pass.num_calls
        # Store as weakref to break circular reference (Trace -> layer_logs -> LayerLog -> Trace).
        _sml = first_pass.source_trace
        self._source_trace_ref: weakref.ReferenceType["Trace"] | None = (
            weakref.ref(_sml) if _sml is not None else None
        )

        # Function identity
        self.func = first_pass.func
        self.func_name = first_pass.func_name
        self.is_inplace = first_pass.is_inplace
        self.grad_fn_name = first_pass.grad_fn_name
        self.grad_fn_id = first_pass.grad_fn_id
        self.grad_fn = first_pass.grad_fn
        self.grad_fn_log = first_pass.grad_fn_log
        self.arg_names = first_pass.arg_names
        self.num_args_total = first_pass.num_args_total
        self.num_pos_args = first_pass.num_pos_args
        self.num_kwargs = first_pass.num_kwargs
        self.is_part_of_iterable_output = first_pass.is_part_of_iterable_output
        self.multi_output_index = first_pass.multi_output_index
        self.multi_output_role = first_pass.multi_output_role

        # Tensor type (representative from first pass)
        self.shape = first_pass.shape
        self.transformed_out_shape = first_pass.transformed_out_shape
        self.dtype = first_pass.dtype
        self.transformed_out_dtype = first_pass.transformed_out_dtype
        self.memory = first_pass.memory
        self.transformed_out_memory = first_pass.transformed_out_memory
        self.autograd_saved_memory: Optional[int] = first_pass.autograd_saved_memory
        self.num_autograd_saved_tensors: Optional[int] = first_pass.num_autograd_saved_tensors

        # Config
        self.output_device = first_pass.output_device
        self.out_postfunc = first_pass.out_postfunc
        self.annotations: Dict[str, Any] = {}
        self.intervention_replaced = first_pass.intervention_replaced
        self.detach_saved_activations = first_pass.detach_saved_activations
        self.save_grads = first_pass.save_grads
        self.transformed_grad_shape = first_pass.transformed_grad_shape
        self.transformed_grad_dtype = first_pass.transformed_grad_dtype
        self.transformed_grad_memory = first_pass.transformed_grad_memory

        # FLOPs
        self.flops_forward = first_pass.flops_forward
        self.flops_backward = first_pass.flops_backward

        # Param identity
        self._param_barcodes = first_pass._param_barcodes
        self._param_logs: List["ParamLog"] = first_pass._param_logs
        self.param_shapes = first_pass.param_shapes
        self.num_params = first_pass.num_params
        self.num_params_trainable = first_pass.num_params_trainable
        self.num_params_frozen = first_pass.num_params_frozen
        self.param_memory = first_pass.param_memory

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
        self.buffer_address = first_pass.buffer_address
        self.buffer_parent = first_pass.buffer_parent
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
        # and is_atomic_module_op (OR).  All others keep first-pass values.
        self.output_of_modules = first_pass.output_of_modules
        self.output_of_module_calls = first_pass.output_of_module_calls
        self.conditional_entry_children = first_pass.conditional_entry_children
        self.conditional_then_children = first_pass.conditional_then_children
        self.conditional_elif_children = first_pass.conditional_elif_children
        self.conditional_else_children = first_pass.conditional_else_children
        self.has_input_ancestor = first_pass.has_input_ancestor
        self.io_role = first_pass.io_role
        self.buffer_pass = first_pass.buffer_pass
        self.is_atomic_module_op = first_pass.is_atomic_module_op

        # Pass management
        self.ops = OpAccessor()
        self.call_labels: List[str] = []

    @property
    def out_transform(self) -> Any:
        """Canonical out transform callable inherited from the first pass.

        Returns
        -------
        Any
            Transform callable, or ``None`` when outs are stored unchanged.
        """

        return self.out_postfunc

    @out_transform.setter
    def out_transform(self, value: Any) -> None:
        """Set the canonical out transform callable.

        Parameters
        ----------
        value:
            Transform callable, or ``None``.
        """

        self.out_postfunc = value

    @property
    def macs_forward(self) -> Optional[int]:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return self.flops_forward // 2 if self.flops_forward is not None else None

    @property
    def macs_backward(self) -> Optional[int]:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return self.flops_backward // 2 if self.flops_backward is not None else None

    @property
    def uses_params(self) -> bool:
        """Whether this layer uses model parameters."""
        return len(self._param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this layer."""
        return len(self._param_barcodes)

    @property
    def in_submodule(self) -> bool:
        """Whether this layer was computed inside a submodule."""
        return self.module is not None

    @property
    def module_call_depth(self) -> int:
        """Depth of module nesting for this layer."""
        return len(self.modules)

    @property
    def memory_str(self) -> str:
        """Return the out tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable tensor memory amount.
        """
        return human_readable_size(self.memory)

    @property
    def param_memory_str(self) -> str:
        """Return the parameter tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable parameter memory amount.
        """
        return human_readable_size(self.param_memory)

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

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with weakrefs stripped."""
        state = self.__dict__.copy()
        state["_source_trace_ref"] = None
        state["io_format_version"] = IO_FORMAT_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state produced by ``__getstate__``."""
        read_io_format_version(state, cls_name=type(self).__name__)
        if "ops" not in state and "passes" in state:
            state["ops"] = state.pop("passes")
        if "call_labels" not in state and "pass_labels" in state:
            state["call_labels"] = state.pop("pass_labels")
        default_fill_state(
            state,
            defaults={
                "_source_trace_ref": None,
                "annotations": {},
                "autograd_saved_memory": None,
                "num_autograd_saved_tensors": None,
                "transformed_out": None,
                "transformed_out_shape": None,
                "transformed_out_dtype": None,
                "transformed_out_memory": None,
                "transformed_grad": None,
                "transformed_grad_shape": None,
                "transformed_grad_dtype": None,
                "transformed_grad_memory": None,
            },
        )
        self.__dict__.update(state)

    # ********************************************
    # ******* Single-pass delegation *************
    # ********************************************
    # For single-pass layers, per-pass fields are transparently accessible
    # on the LayerLog itself.  For multi-pass layers, attempting to access
    # these fields raises ValueError directing the user to a specific pass.

    def _single_pass_or_error(self, field_name: str) -> Any:
        """Access a per-pass field, requiring exactly one pass.

        Raises ValueError (not AttributeError) for multi-pass layers.
        Using ValueError avoids the Python property/__getattr__ trap:
        if a @property raises AttributeError, Python silently treats
        the attribute as missing and falls through to __getattr__.
        """
        if self.num_calls > 1:
            raise ValueError(
                f"Layer '{self.layer_label}' has {self.num_calls} ops. "
                f"Access '{field_name}' on a specific pass: "
                f"log['{self.layer_label}'].ops[1].{field_name}"
            )
        return getattr(self.ops[1], field_name)

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
    def has_saved_outs(self) -> bool:
        """Return whether the single pass has a saved out.

        Returns
        -------
        bool
            ``True`` when an out was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_saved_outs"))

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
    def func_duration(self) -> Any:
        """Return function execution time for a single-pass layer.

        Returns
        -------
        Any
            Function timing value from the only pass.
        """
        return self._single_pass_or_error("func_duration")

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
    def compute_index(self) -> int:
        """Return the operation number for a single-pass layer.

        Returns
        -------
        int
            Operation number from the only pass.
        """
        return cast(int, self._single_pass_or_error("compute_index"))

    @property
    def call_index(self) -> int:
        """Return the pass number for a single-pass layer.

        Returns
        -------
        int
            Pass number from the only pass.
        """
        return cast(int, self._single_pass_or_error("call_index"))

    @property
    def capture_index(self) -> int:
        """Return the creation-order index for a single-pass layer.

        Returns
        -------
        int
            Creation-order value from the only pass.
        """
        return cast(int, self._single_pass_or_error("capture_index"))

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
    # no-pass labels (i.e. LayerLog-level identifiers).  This gives a
    # complete picture of which layers are connected across all recurrent
    # iterations.  Order is preserved (first-seen insertion order).

    @property
    def children(self) -> list[str]:
        """Union of child layers (no-pass labels) across all ops."""
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.children:
                no_pass = self.source_trace[label].layer_label_no_pass
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
                no_pass = self.source_trace[label].layer_label_no_pass
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
    def siblings(self) -> list[str]:
        """Union of sibling layers (no-pass labels) across all ops."""
        result = []
        seen = set()
        for pass_log in self.ops.values():
            for label in pass_log.siblings:
                no_pass = self.source_trace[label].layer_label_no_pass
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
                no_pass = self.source_trace[label].layer_label_no_pass
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
    def layer_label_no_pass(self) -> str:
        """Alias so code expecting layer_label_no_pass works on LayerLog."""
        return cast(str, self.layer_label)

    @property
    def layer_label_no_pass_short(self) -> str:
        """Alias so code expecting layer_label_no_pass_short works on LayerLog."""
        return cast(str, self.layer_label_short)

    @property
    def layer_label_w_pass(self) -> str:
        """For single-pass layers, return the pass-qualified label."""
        return cast(str, self._single_pass_or_error("layer_label_w_pass"))

    @property
    def layer_label_w_pass_short(self) -> str:
        """For single-pass layers, return the short pass-qualified label."""
        return cast(str, self._single_pass_or_error("layer_label_w_pass_short"))

    @property
    def params(self) -> Any:
        """Access parameter metadata by address, short name, or index."""
        from .param_log import ParamAccessor

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
                no_pass = self.source_trace[label].layer_label_no_pass
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
                no_pass = self.source_trace[label].layer_label_no_pass
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
                no_pass = self.source_trace[label].layer_label_no_pass
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
                no_pass = self.source_trace[label].layer_label_no_pass
                if call_index not in result[no_pass]:
                    result[no_pass].append(call_index)
        return dict(result)

    @property
    def edges_vary_across_ops(self) -> bool:
        """Whether graph edges differ across ops."""
        if self.num_calls <= 1:
            return False
        all_pass_lists = list(self.child_ops_per_layer.values()) + list(
            self.parent_ops_per_layer.values()
        )
        return any(len(ops) < self.num_calls for ops in all_pass_lists)

    @property
    def leaf_module_ops(self) -> set[Any]:
        """Set of module ops exited across all ops."""
        result = set()
        for pass_log in self.ops.values():
            if pass_log.is_atomic_module_op:
                result.add(pass_log.atomic_module_call)
        return result

    @property
    def parent_arg_positions(self) -> dict[str, dict[Any, str]]:
        """Merged parent_arg_positions across ops (set-union).

        For single-pass layers, delegates to ops[1].
        For multi-pass, merges arg locs using set-union of no-pass labels.
        """
        if self.num_calls == 1:
            return cast(dict[str, dict[Any, str]], self.ops[1].parent_arg_positions)
        from collections import defaultdict

        result: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
        for pass_log in self.ops.values():
            for arg_type in ["args", "kwargs"]:
                for arg_key, layer_label in pass_log.parent_arg_positions[arg_type].items():
                    no_pass = self.source_trace[layer_label].layer_label_no_pass
                    if arg_key not in result[arg_type]:
                        result[arg_type][arg_key] = no_pass
        return result

    # ********************************************
    # ******* Fallback __getattr__ ***************
    # ********************************************

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute lookup: delegates to ops[1] for single-pass layers.

        Only called when normal attribute lookup has already failed (Python's
        ``__getattr__`` protocol).  For single-pass layers, transparently
        forwards to the underlying OpLog, enabling code like
        ``layer_log.func_rng_states`` without needing an explicit property.

        Private attributes (starting with '_') are never delegated — they
        raise AttributeError immediately to avoid infinite recursion with
        ``self.__dict__`` access.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        ops = self.__dict__.get("ops")
        if ops and len(ops) == 1 and 1 in ops:
            try:
                return getattr(ops[1], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ********************************************
    # ************ User-facing custom_methods ***********
    # ********************************************

    def get_children(self) -> list["LayerLog"]:
        """Return child LayerLog objects for this layer.

        Returns
        -------
        list[LayerLog]
            Child layers resolved through the owning model log.
        """
        return [self.source_trace[child_label] for child_label in self.children]

    def get_parents(self) -> list["LayerLog"]:
        """Return parent LayerLog objects for this layer.

        Returns
        -------
        list[LayerLog]
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
                "This LayerLog is detached from its source Trace "
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

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        if not self._tracing_finished:
            return f"LayerLog({self.layer_label}) (pass not finished)"
        s = f"Layer {self.layer_label}:"
        if self.num_calls > 1:
            s += f" ({self.num_calls} ops)"
        s += f"\n\tOutput tensor: shape={self.shape}, dtype={self.dtype}, size={self.memory_str}"
        if not self.is_input:
            s += f"\n\tFunction: {self.func_name} (grad_fn: {self.grad_fn_name})"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
        if self.module is not None:
            s += f"\n\tComputed inside module: {self.module}"
        if len(self.param_shapes) > 0:
            params_shapes_str = ", ".join(str(ps) for ps in self.param_shapes)
            s += (
                f"\n\tParams: {params_shapes_str}; "
                f"{self.num_params} total ({self.param_memory_str})"
            )
        s += "\n\tRelated Layers:"
        s += f"\n\t\t- parents: {', '.join(self.parents) or 'none'}"
        s += f"\n\t\t- children: {', '.join(self.children) or 'none'}"
        if self.num_calls > 1:
            s += f"\n\tPasses: {', '.join(self.call_labels)}"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return cast(int, self.num_calls)


class LayerAccessor(Accessor[Union["LayerLog", "OpLog"]]):
    """Dict-like accessor for LayerLog objects.

    Supports indexing by:
    * **layer label** (str) -- exact match against no-pass label.
    * **ordinal index** (int) -- position in execution order.
    * **pass notation** (str ``"conv2d_1_1:2"``) -- returns the OpLog
      for a specific pass of a multi-pass layer.

    Available as ``trace.layers``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        layer_logs: Dict[str, "LayerLog"],
        source_trace: Optional["Trace"] = None,
    ) -> None:
        source_ref = weakref.ref(source_trace) if source_trace is not None else None
        super().__init__(layer_logs, source_ref=source_ref)

    def _resolve_pass_qualified(self, key: str) -> "OpLog | None":
        """Resolve ``layer_label:pass`` notation to an OpLog."""
        base, _, pass_str = key.rpartition(":")
        if base in self._dict:
            try:
                call_index = int(pass_str)
                return self._dict[base].ops[call_index]
            except (ValueError, KeyError):
                return None
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
                f"ops={ll.num_calls})"
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
                    "memory_str": ll.memory_str,
                    "num_calls": ll.num_calls,
                    "num_params": ll.num_params,
                    "module": ll.module,
                    "is_input": ll.is_input,
                    "is_output": ll.is_output,
                    "is_buffer": ll.is_buffer,
                }
            )
        return pd.DataFrame(rows)

    def to_csv(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the layer table to CSV.

        Parameters
        ----------
        filepath:
            Output CSV path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_csv``.
        """
        self.to_pandas().to_csv(filepath, index=False, **kwargs)

    def to_parquet(self, filepath: str | PathLike[str], **kwargs: Any) -> None:
        """Write the layer table to Parquet.

        Parameters
        ----------
        filepath:
            Output Parquet path.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_parquet``.

        Raises
        ------
        ImportError
            If ``pyarrow`` is unavailable.
        """
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "to_parquet requires pyarrow. Install with: pip install torchlens[io]"
            ) from exc
        from ..export import _parquet_safe_dataframe

        _parquet_safe_dataframe(self.to_pandas()).to_parquet(filepath, **kwargs)

    def to_json(
        self,
        filepath: str | PathLike[str],
        *,
        orient: Literal["split", "records", "index", "columns", "values", "table"] = "records",
        **kwargs: Any,
    ) -> None:
        """Write the layer table to JSON.

        Parameters
        ----------
        filepath:
            Output JSON path.
        orient:
            JSON orientation passed to ``DataFrame.to_json``.
        **kwargs:
            Additional keyword arguments forwarded to ``DataFrame.to_json``.
        """
        self.to_pandas().to_json(filepath, orient=orient, **kwargs)
