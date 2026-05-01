"""LayerLog and LayerAccessor: aggregate per-layer metadata and dict-like accessor.

LayerLog groups one or more LayerPassLog entries that represent the same
logical layer across recurrent passes.  For non-recurrent models (the
common case), every LayerLog wraps exactly one LayerPassLog.

**Delegation pattern**: For single-pass layers, per-pass fields (activation,
gradient, operation_num, etc.) are accessible directly on the LayerLog
via ``_single_pass_or_error()`` and ``__getattr__`` delegation to ``passes[1]``.
For multi-pass layers, accessing these fields raises ``ValueError`` (NOT
``AttributeError``) directing the user to ``layer_log.passes[N].field``.

Why ValueError instead of AttributeError: Python's property protocol treats
``AttributeError`` from a ``@property`` as "attribute doesn't exist" and falls
through to ``__getattr__``.  Using ``ValueError`` avoids this trap and gives
the user a clear error message.

**_build_layer_logs merge rules** (in postprocess/layer_log.py):
When merging multiple passes into one LayerLog, these aggregate fields are merged:
  - ``has_input_ancestor``: OR across passes
  - ``io_role``: character-level merge of "I", "O", "IO" strings
  - ``is_leaf_module_output``: OR across passes
  - ``in_cond_branch``: OR across passes
  - ``conditional_branch_stacks`` / ``conditional_branch_stack_passes``:
    unique per-pass stack signatures and their pass numbers
  - ``cond_branch_children_by_cond`` and derived child views:
    pass-stripped ordered unions across passes
All other 78+ fields use the first pass's values only.
``modules_exited`` / ``module_passes_exited`` are NOT updated across passes
(correct because same-layer grouping requires identical structural position).
"""

import weakref
from collections.abc import Iterator
from os import PathLike
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

from .._io import FieldPolicy, IO_FORMAT_VERSION, default_fill_state, read_io_format_version
from ..utils.display import human_readable_size

if TYPE_CHECKING:
    import pandas as pd

    from .layer_pass_log import LayerPassLog
    from .model_log import ModelLog
    from .param_log import ParamLog


class LayerLog:
    """Aggregate per-layer metadata for a logged model operation.

    Groups one or more LayerPassLog objects (one per invocation of this layer).
    For non-recurrent models, every LayerLog has exactly one pass.

    Aggregate fields (function identity, param identity, flags, module containment)
    live directly on LayerLog.  Per-pass fields (activations, graph edges,
    execution state, gradients) live on the LayerPassLog objects in ``self.passes``.

    For single-pass layers, per-pass fields are accessible directly via
    ``__getattr__`` delegation (e.g. ``layer_log.activation`` transparently
    reads from ``passes[1].activation``).
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "layer_label": FieldPolicy.KEEP,
        "layer_label_short": FieldPolicy.KEEP,
        "layer_type": FieldPolicy.KEEP,
        "layer_type_num": FieldPolicy.KEEP,
        "layer_total_num": FieldPolicy.KEEP,
        "num_passes": FieldPolicy.KEEP,
        "_source_model_log_ref": FieldPolicy.WEAKREF_STRIP,
        "func_applied": FieldPolicy.DROP,
        "func_name": FieldPolicy.KEEP,
        "func_is_inplace": FieldPolicy.KEEP,
        "grad_fn_name": FieldPolicy.KEEP,
        "grad_fn_id": FieldPolicy.KEEP,
        "grad_fn_object": FieldPolicy.DROP,
        "corresponding_grad_fn": FieldPolicy.DROP,
        "func_argnames": FieldPolicy.KEEP,
        "num_args": FieldPolicy.KEEP,
        "num_positional_args": FieldPolicy.KEEP,
        "num_keyword_args": FieldPolicy.KEEP,
        "is_part_of_iterable_output": FieldPolicy.KEEP,
        "iterable_output_index": FieldPolicy.KEEP,
        "tensor_shape": FieldPolicy.KEEP,
        "transformed_activation_shape": FieldPolicy.KEEP,
        "tensor_dtype": FieldPolicy.KEEP,
        "transformed_activation_dtype": FieldPolicy.KEEP,
        "tensor_memory": FieldPolicy.KEEP,
        "transformed_activation_memory": FieldPolicy.KEEP,
        "transformed_activation": FieldPolicy.BLOB,
        "autograd_saved_bytes": FieldPolicy.KEEP,
        "autograd_saved_tensor_count": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "activation_postfunc": FieldPolicy.DROP,
        "extra_data": FieldPolicy.KEEP,
        "detach_saved_tensor": FieldPolicy.KEEP,
        "save_gradients": FieldPolicy.KEEP,
        "transformed_gradient": FieldPolicy.BLOB,
        "transformed_gradient_shape": FieldPolicy.KEEP,
        "transformed_gradient_dtype": FieldPolicy.KEEP,
        "transformed_gradient_memory": FieldPolicy.KEEP,
        "flops_forward": FieldPolicy.KEEP,
        "flops_backward": FieldPolicy.KEEP,
        "parent_param_barcodes": FieldPolicy.KEEP,
        "parent_param_logs": FieldPolicy.KEEP,
        "parent_param_shapes": FieldPolicy.KEEP,
        "num_params_total": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "params_memory": FieldPolicy.KEEP,
        "func_config": FieldPolicy.BLOB_RECURSIVE,
        "operation_equivalence_type": FieldPolicy.KEEP,
        "equivalent_operations": FieldPolicy.KEEP,
        "is_input_layer": FieldPolicy.KEEP,
        "is_output_layer": FieldPolicy.KEEP,
        "is_final_output": FieldPolicy.KEEP,
        "is_buffer_layer": FieldPolicy.KEEP,
        "buffer_address": FieldPolicy.KEEP,
        "buffer_parent": FieldPolicy.KEEP,
        "is_internally_initialized": FieldPolicy.KEEP,
        "is_internally_terminated": FieldPolicy.KEEP,
        "is_terminal_bool_layer": FieldPolicy.KEEP,
        "is_scalar_bool": FieldPolicy.KEEP,
        "scalar_bool_value": FieldPolicy.KEEP,
        "in_cond_branch": FieldPolicy.KEEP,
        "conditional_branch_stacks": FieldPolicy.KEEP,
        "conditional_branch_stack_passes": FieldPolicy.KEEP,
        "cond_branch_children_by_cond": FieldPolicy.KEEP,
        "containing_module": FieldPolicy.KEEP,
        "containing_modules": FieldPolicy.KEEP,
        "modules_exited": FieldPolicy.KEEP,
        "module_passes_exited": FieldPolicy.KEEP,
        "cond_branch_start_children": FieldPolicy.KEEP,
        "cond_branch_then_children": FieldPolicy.KEEP,
        "cond_branch_elif_children": FieldPolicy.KEEP,
        "cond_branch_else_children": FieldPolicy.KEEP,
        "has_input_ancestor": FieldPolicy.KEEP,
        "io_role": FieldPolicy.KEEP,
        "buffer_pass": FieldPolicy.KEEP,
        "is_leaf_module_output": FieldPolicy.KEEP,
        "passes": FieldPolicy.KEEP,
        "pass_labels": FieldPolicy.KEEP,
    }

    def __init__(self, first_pass: "LayerPassLog") -> None:
        """Initialize from the first pass of this layer.

        Args:
            first_pass: The LayerPassLog for pass 1 of this layer.
        """
        # Identity & labeling
        self.layer_label = first_pass.layer_label_no_pass
        self.layer_label_short = first_pass.layer_label_no_pass_short
        self.layer_type = first_pass.layer_type
        self.layer_type_num = first_pass.layer_type_num
        self.layer_total_num = first_pass.layer_total_num
        self.num_passes = first_pass.num_passes
        # Store as weakref to break circular reference (ModelLog -> layer_logs -> LayerLog -> ModelLog).
        _sml = first_pass.source_model_log
        self._source_model_log_ref: weakref.ReferenceType["ModelLog"] | None = (
            weakref.ref(_sml) if _sml is not None else None
        )

        # Function identity
        self.func_applied = first_pass.func_applied
        self.func_name = first_pass.func_name
        self.func_is_inplace = first_pass.func_is_inplace
        self.grad_fn_name = first_pass.grad_fn_name
        self.grad_fn_id = first_pass.grad_fn_id
        self.grad_fn_object = first_pass.grad_fn_object
        self.corresponding_grad_fn = first_pass.corresponding_grad_fn
        self.func_argnames = first_pass.func_argnames
        self.num_args = first_pass.num_args
        self.num_positional_args = first_pass.num_positional_args
        self.num_keyword_args = first_pass.num_keyword_args
        self.is_part_of_iterable_output = first_pass.is_part_of_iterable_output
        self.iterable_output_index = first_pass.iterable_output_index

        # Tensor type (representative from first pass)
        self.tensor_shape = first_pass.tensor_shape
        self.transformed_activation_shape = first_pass.transformed_activation_shape
        self.tensor_dtype = first_pass.tensor_dtype
        self.transformed_activation_dtype = first_pass.transformed_activation_dtype
        self.tensor_memory = first_pass.tensor_memory
        self.transformed_activation_memory = first_pass.transformed_activation_memory
        self.autograd_saved_bytes: Optional[int] = first_pass.autograd_saved_bytes
        self.autograd_saved_tensor_count: Optional[int] = first_pass.autograd_saved_tensor_count

        # Config
        self.output_device = first_pass.output_device
        self.activation_postfunc = first_pass.activation_postfunc
        self.extra_data: Dict[str, Any] = {}
        self.detach_saved_tensor = first_pass.detach_saved_tensor
        self.save_gradients = first_pass.save_gradients
        self.transformed_gradient_shape = first_pass.transformed_gradient_shape
        self.transformed_gradient_dtype = first_pass.transformed_gradient_dtype
        self.transformed_gradient_memory = first_pass.transformed_gradient_memory

        # FLOPs
        self.flops_forward = first_pass.flops_forward
        self.flops_backward = first_pass.flops_backward

        # Param identity
        self.parent_param_barcodes = first_pass.parent_param_barcodes
        self.parent_param_logs: List["ParamLog"] = first_pass.parent_param_logs
        self.parent_param_shapes = first_pass.parent_param_shapes
        self.num_params_total = first_pass.num_params_total
        self.num_params_trainable = first_pass.num_params_trainable
        self.num_params_frozen = first_pass.num_params_frozen
        self.params_memory = first_pass.params_memory

        # Function config
        self.func_config = first_pass.func_config

        # Equivalence
        self.operation_equivalence_type = first_pass.operation_equivalence_type
        self.equivalent_operations = first_pass.equivalent_operations

        # Special flags
        self.is_input_layer = first_pass.is_input_layer
        self.is_output_layer = first_pass.is_output_layer
        self.is_final_output = first_pass.is_final_output
        self.is_buffer_layer = first_pass.is_buffer_layer
        self.buffer_address = first_pass.buffer_address
        self.buffer_parent = first_pass.buffer_parent
        self.is_internally_initialized = first_pass.is_internally_initialized
        self.is_internally_terminated = first_pass.is_internally_terminated
        self.is_terminal_bool_layer = first_pass.is_terminal_bool_layer
        self.is_scalar_bool = first_pass.is_scalar_bool
        self.scalar_bool_value = first_pass.scalar_bool_value
        self.in_cond_branch = first_pass.in_cond_branch
        self.conditional_branch_stacks: List[List[Tuple[int, str]]] = []
        self.conditional_branch_stack_passes: Dict[Tuple[Tuple[int, str], ...], List[int]] = {}
        self.cond_branch_children_by_cond: Dict[int, Dict[str, List[str]]] = {}

        # Module (static containment)
        self.containing_module = first_pass.containing_module
        self.containing_modules = first_pass.containing_modules

        # Fields stored as aggregate for vis compatibility.
        # Initialized from first pass.  For multi-pass layers, _build_layer_logs
        # merges only has_input_ancestor (OR), io_role (char-merge),
        # and is_leaf_module_output (OR).  All others keep first-pass values.
        self.modules_exited = first_pass.modules_exited
        self.module_passes_exited = first_pass.module_passes_exited
        self.cond_branch_start_children = first_pass.cond_branch_start_children
        self.cond_branch_then_children = first_pass.cond_branch_then_children
        self.cond_branch_elif_children = first_pass.cond_branch_elif_children
        self.cond_branch_else_children = first_pass.cond_branch_else_children
        self.has_input_ancestor = first_pass.has_input_ancestor
        self.io_role = first_pass.io_role
        self.buffer_pass = first_pass.buffer_pass
        self.is_leaf_module_output = first_pass.is_leaf_module_output

        # Pass management
        self.passes: Dict[int, "LayerPassLog"] = {}
        self.pass_labels: List[str] = []

    @property
    def activation_transform(self) -> Any:
        """Canonical activation transform callable inherited from the first pass.

        Returns
        -------
        Any
            Transform callable, or ``None`` when activations are stored unchanged.
        """

        return self.activation_postfunc

    @activation_transform.setter
    def activation_transform(self, value: Any) -> None:
        """Set the canonical activation transform callable.

        Parameters
        ----------
        value:
            Transform callable, or ``None``.
        """

        self.activation_postfunc = value

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
        return len(self.parent_param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this layer."""
        return len(self.parent_param_barcodes)

    @property
    def is_computed_inside_submodule(self) -> bool:
        """Whether this layer was computed inside a submodule."""
        return self.containing_module is not None

    @property
    def module_nesting_depth(self) -> int:
        """Depth of module nesting for this layer."""
        return len(self.containing_modules)

    @property
    def tensor_memory_str(self) -> str:
        """Return the activation tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable tensor memory amount.
        """
        return human_readable_size(self.tensor_memory)

    @property
    def params_memory_str(self) -> str:
        """Return the parameter tensor size in human-readable units.

        Returns
        -------
        str
            Human-readable parameter memory amount.
        """
        return human_readable_size(self.params_memory)

    @property
    def source_model_log(self) -> "ModelLog":
        """Back-reference to the owning ModelLog (stored as weakref)."""
        ref = self.__dict__.get("_source_model_log_ref")
        if ref is None:
            return None  # type: ignore[return-value]
        obj = ref()
        if obj is None:
            raise RuntimeError("ModelLog has been garbage-collected.")
        return cast("ModelLog", obj)

    @source_model_log.setter
    def source_model_log(self, value: "ModelLog | None") -> None:
        """Set the owning ModelLog back-reference.

        Parameters
        ----------
        value:
            Owning model log, or ``None`` to clear the reference.
        """
        self._source_model_log_ref = weakref.ref(value) if value is not None else None

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
            defaults={
                "_source_model_log_ref": None,
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
        if self.num_passes > 1:
            raise ValueError(
                f"Layer '{self.layer_label}' has {self.num_passes} passes. "
                f"Access '{field_name}' on a specific pass: "
                f"log['{self.layer_label}'].passes[1].{field_name}"
            )
        return getattr(self.passes[1], field_name)

    @property
    def activation(self) -> Any:
        """Return the saved activation for a single-pass layer.

        Returns
        -------
        Any
            Saved activation from the only pass.
        """
        return self._single_pass_or_error("activation")

    @property
    def tensor(self) -> Any:
        """Alias for the raw saved activation on single-pass layers."""

        return self._single_pass_or_error("tensor")

    @property
    def transformed_activation(self) -> Any:
        """Transformed activation on single-pass layers."""

        return self._single_pass_or_error("transformed_activation")

    @property
    def has_saved_activations(self) -> bool:
        """Return whether the single pass has a saved activation.

        Returns
        -------
        bool
            ``True`` when an activation was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_saved_activations"))

    @property
    def captured_args(self) -> Any:
        """Return captured positional arguments for a single-pass layer.

        Returns
        -------
        Any
            Captured positional arguments from the only pass.
        """
        return self._single_pass_or_error("captured_args")

    @property
    def captured_kwargs(self) -> Any:
        """Return captured keyword arguments for a single-pass layer.

        Returns
        -------
        Any
            Captured keyword arguments from the only pass.
        """
        return self._single_pass_or_error("captured_kwargs")

    @property
    def gradient(self) -> Any:
        """Return the saved gradient for a single-pass layer.

        Returns
        -------
        Any
            Saved gradient from the only pass.
        """
        return self._single_pass_or_error("gradient")

    @property
    def transformed_gradient(self) -> Any:
        """Transformed gradient on single-pass layers."""

        return self._single_pass_or_error("transformed_gradient")

    @property
    def has_gradient(self) -> bool:
        """Return whether the single pass has a saved gradient.

        Returns
        -------
        bool
            ``True`` when a gradient was saved for the only pass.
        """
        return cast(bool, self._single_pass_or_error("has_gradient"))

    @property
    def func_call_stack(self) -> Any:
        """Return the captured call stack for a single-pass layer.

        Returns
        -------
        Any
            Function call stack from the only pass.
        """
        return self._single_pass_or_error("func_call_stack")

    @property
    def func_time(self) -> Any:
        """Return function execution time for a single-pass layer.

        Returns
        -------
        Any
            Function timing value from the only pass.
        """
        return self._single_pass_or_error("func_time")

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
    def operation_num(self) -> int:
        """Return the operation number for a single-pass layer.

        Returns
        -------
        int
            Operation number from the only pass.
        """
        return cast(int, self._single_pass_or_error("operation_num"))

    @property
    def pass_num(self) -> int:
        """Return the pass number for a single-pass layer.

        Returns
        -------
        int
            Pass number from the only pass.
        """
        return cast(int, self._single_pass_or_error("pass_num"))

    @property
    def creation_order(self) -> int:
        """Return the creation-order index for a single-pass layer.

        Returns
        -------
        int
            Creation-order value from the only pass.
        """
        return cast(int, self._single_pass_or_error("creation_order"))

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
    # Graph-edge properties compute the union across all passes, returning
    # no-pass labels (i.e. LayerLog-level identifiers).  This gives a
    # complete picture of which layers are connected across all recurrent
    # iterations.  Order is preserved (first-seen insertion order).

    @property
    def child_layers(self) -> list[str]:
        """Union of child layers (no-pass labels) across all passes."""
        result = []
        seen = set()
        for pass_log in self.passes.values():
            for label in pass_log.child_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def parent_layers(self) -> list[str]:
        """Union of parent layers (no-pass labels) across all passes."""
        result = []
        seen = set()
        for pass_log in self.passes.values():
            for label in pass_log.parent_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
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
        return any(p.has_children for p in self.passes.values())

    @property
    def has_parents(self) -> bool:
        """Return whether any pass has parent layers.

        Returns
        -------
        bool
            ``True`` when at least one pass has graph parents.
        """
        return any(p.has_parents for p in self.passes.values())

    @property
    def sibling_layers(self) -> list[str]:
        """Union of sibling layers (no-pass labels) across all passes."""
        result = []
        seen = set()
        for pass_log in self.passes.values():
            for label in pass_log.sibling_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
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
        return any(p.has_siblings for p in self.passes.values())

    @property
    def co_parent_layers(self) -> list[str]:
        """Union of spouse layers (no-pass labels) across all passes."""
        result = []
        seen = set()
        for pass_log in self.passes.values():
            for label in pass_log.co_parent_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
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
        return any(p.has_co_parents for p in self.passes.values())

    @property
    def _pass_finished(self) -> bool:
        sml = self.source_model_log
        if sml is None:
            return True
        return sml._pass_finished

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

        param_dict = {pl.address: pl for pl in self.parent_param_logs}
        return ParamAccessor(param_dict)

    # ********************************************
    # **** Rolled-vis computed properties ********
    # ********************************************
    # These provide per-pass edge tracking for rolled (recurrence-aware)
    # graph visualization.  Computed on-the-fly from the passes dict.
    # Used by the visualization renderer to draw pass-annotated edges.

    @property
    def child_layers_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: child layer labels (no-pass) for each pass."""
        result = {}
        for pass_num, pass_log in self.passes.items():
            children = []
            for label in pass_log.child_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if no_pass not in children:
                    children.append(no_pass)
            result[pass_num] = children
        return result

    @property
    def parent_layers_per_pass(self) -> dict[int, list[str]]:
        """Dict[int, List[str]]: parent layer labels (no-pass) for each pass."""
        result = {}
        for pass_num, pass_log in self.passes.items():
            parents = []
            for label in pass_log.parent_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if no_pass not in parents:
                    parents.append(no_pass)
            result[pass_num] = parents
        return result

    @property
    def child_passes_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each child layer, which passes connect to it."""
        from collections import defaultdict

        result: defaultdict[str, list[int]] = defaultdict(list)
        for pass_num, pass_log in self.passes.items():
            for label in pass_log.child_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if pass_num not in result[no_pass]:
                    result[no_pass].append(pass_num)
        return dict(result)

    @property
    def parent_passes_per_layer(self) -> dict[str, list[int]]:
        """Dict[str, List[int]]: for each parent layer, which passes connect from it."""
        from collections import defaultdict

        result: defaultdict[str, list[int]] = defaultdict(list)
        for pass_num, pass_log in self.passes.items():
            for label in pass_log.parent_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if pass_num not in result[no_pass]:
                    result[no_pass].append(pass_num)
        return dict(result)

    @property
    def edges_vary_across_passes(self) -> bool:
        """Whether graph edges differ across passes."""
        if self.num_passes <= 1:
            return False
        all_pass_lists = list(self.child_passes_per_layer.values()) + list(
            self.parent_passes_per_layer.values()
        )
        return any(len(passes) < self.num_passes for passes in all_pass_lists)

    @property
    def leaf_module_passes(self) -> set[Any]:
        """Set of module passes exited across all passes."""
        result = set()
        for pass_log in self.passes.values():
            if pass_log.is_leaf_module_output:
                result.add(pass_log.leaf_module_pass)
        return result

    @property
    def parent_layer_arg_locs(self) -> dict[str, dict[Any, str]]:
        """Merged parent_layer_arg_locs across passes (set-union).

        For single-pass layers, delegates to passes[1].
        For multi-pass, merges arg locs using set-union of no-pass labels.
        """
        if self.num_passes == 1:
            return cast(dict[str, dict[Any, str]], self.passes[1].parent_layer_arg_locs)
        from collections import defaultdict

        result: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
        for pass_log in self.passes.values():
            for arg_type in ["args", "kwargs"]:
                for arg_key, layer_label in pass_log.parent_layer_arg_locs[arg_type].items():
                    no_pass = self.source_model_log[layer_label].layer_label_no_pass
                    if arg_key not in result[arg_type]:
                        result[arg_type][arg_key] = no_pass
        return result

    # ********************************************
    # ******* Fallback __getattr__ ***************
    # ********************************************

    def __getattr__(self, name: str) -> Any:
        """Fallback attribute lookup: delegates to passes[1] for single-pass layers.

        Only called when normal attribute lookup has already failed (Python's
        ``__getattr__`` protocol).  For single-pass layers, transparently
        forwards to the underlying LayerPassLog, enabling code like
        ``layer_log.func_rng_states`` without needing an explicit property.

        Private attributes (starting with '_') are never delegated — they
        raise AttributeError immediately to avoid infinite recursion with
        ``self.__dict__`` access.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        passes = self.__dict__.get("passes")
        if passes and len(passes) == 1 and 1 in passes:
            try:
                return getattr(passes[1], name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ********************************************
    # ************ User-facing methods ***********
    # ********************************************

    def get_child_layers(self) -> list["LayerLog"]:
        """Return child LayerLog objects for this layer.

        Returns
        -------
        list[LayerLog]
            Child layers resolved through the owning model log.
        """
        return [self.source_model_log[child_label] for child_label in self.child_layers]

    def get_parent_layers(self) -> list["LayerLog"]:
        """Return parent LayerLog objects for this layer.

        Returns
        -------
        list[LayerLog]
            Parent layers resolved through the owning model log.
        """
        return [self.source_model_log[parent_label] for parent_label in self.parent_layers]

    def show(
        self,
        method: Literal["auto", "heatmap", "channels", "rgb", "hist"] = "auto",
        **kwargs: Any,
    ) -> Any:
        """Display this layer's saved activation.

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

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self) -> str:
        if not self._pass_finished:
            return f"LayerLog({self.layer_label}) (pass not finished)"
        s = f"Layer {self.layer_label}:"
        if self.num_passes > 1:
            s += f" ({self.num_passes} passes)"
        s += (
            f"\n\tOutput tensor: shape={self.tensor_shape}, "
            f"dtype={self.tensor_dtype}, size={self.tensor_memory_str}"
        )
        if not self.is_input_layer:
            s += f"\n\tFunction: {self.func_name} (grad_fn: {self.grad_fn_name})"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
        if self.containing_module is not None:
            s += f"\n\tComputed inside module: {self.containing_module}"
        if len(self.parent_param_shapes) > 0:
            params_shapes_str = ", ".join(str(ps) for ps in self.parent_param_shapes)
            s += (
                f"\n\tParams: {params_shapes_str}; "
                f"{self.num_params_total} total ({self.params_memory_str})"
            )
        s += "\n\tRelated Layers:"
        s += f"\n\t\t- parents: {', '.join(self.parent_layers) or 'none'}"
        s += f"\n\t\t- children: {', '.join(self.child_layers) or 'none'}"
        if self.num_passes > 1:
            s += f"\n\tPasses: {', '.join(self.pass_labels)}"
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return cast(int, self.num_passes)


class LayerAccessor:
    """Dict-like accessor for LayerLog objects.

    Supports indexing by:
    * **layer label** (str) -- exact match against no-pass label.
    * **ordinal index** (int) -- position in execution order.
    * **pass notation** (str ``"conv2d_1_1:2"``) -- returns the LayerPassLog
      for a specific pass of a multi-pass layer.

    Available as ``model_log.layers``.
    """

    PORTABLE_STATE_SPEC: dict[str, FieldPolicy] = {
        "_dict": FieldPolicy.KEEP,
        "_list": FieldPolicy.KEEP,
        "_source_ref": FieldPolicy.WEAKREF_STRIP,
    }

    def __init__(
        self,
        layer_logs: Dict[str, "LayerLog"],
        source_model_log: Optional["ModelLog"] = None,
    ) -> None:
        self._dict = layer_logs  # no-pass label -> LayerLog
        self._list = list(layer_logs.values())  # execution-order list
        # Store as weakref to avoid preventing ModelLog GC.
        self._source_ref = weakref.ref(source_model_log) if source_model_log is not None else None

    def __getitem__(self, key: Union[int, str]) -> Union["LayerLog", "LayerPassLog"]:
        """Return a LayerLog by label or index, or a LayerPassLog by pass label.

        Pass notation ``"conv2d_1_1:2"`` splits on the last colon, looks up
        the base LayerLog, and returns ``layer_log.passes[2]``.
        """
        if isinstance(key, int):
            return self._list[key]
        if key in self._dict:
            return self._dict[key]
        # Try pass notation: "conv2d_1_1:2" -> LayerLog.passes[2]
        if ":" in key and self._source_ref is not None:
            base, _, pass_str = key.rpartition(":")
            if base in self._dict:
                try:
                    pass_num = int(pass_str)
                    return self._dict[base].passes[pass_num]
                except (ValueError, KeyError):
                    pass
        suggestions = []
        source = self._source_ref() if self._source_ref is not None else None
        if source is not None and hasattr(source, "suggest"):
            suggestions = source.suggest(str(key))
        if suggestions:
            suggestion_str = ", ".join(repr(item) for item in suggestions)
            raise KeyError(f"Layer '{key}' not found. Did you mean {suggestion_str}?")
        raise KeyError(f"Layer '{key}' not found.")

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __dir__(self) -> List[str]:
        """Return Python attributes plus layer labels for tab completion.

        Returns
        -------
        List[str]
            Attribute names and valid layer labels.
        """

        return sorted(set(super().__dir__()) | set(self._dict.keys()))

    def _ipython_key_completions_(self) -> List[str]:
        """Return layer labels for IPython ``obj[...]`` completion.

        Returns
        -------
        List[str]
            Valid layer labels.
        """

        return list(self._dict.keys())

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator["LayerLog"]:
        """Iterate over LayerLog objects in execution order."""
        return iter(self._list)

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
                if layer.containing_module == module
                or module in getattr(layer, "containing_modules", [])
            ]
        counts: Dict[str, int] = {}
        for layer in self._list:
            key = str(layer.containing_module or "self")
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
                if (
                    layer.containing_module == module
                    or module in getattr(layer, "containing_modules", [])
                )
                and (layer.func_name or layer.layer_type) == operator
            ]
        counts: Dict[Tuple[str, str], int] = {}
        for layer in self._list:
            key = (str(layer.containing_module or "self"), str(layer.func_name or layer.layer_type))
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
                f"(shape={list(ll.tensor_shape) if ll.tensor_shape else '?'}, "
                f"passes={ll.num_passes})"
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
                    "tensor_shape": ll.tensor_shape,
                    "tensor_dtype": ll.tensor_dtype,
                    "tensor_memory_str": ll.tensor_memory_str,
                    "num_passes": ll.num_passes,
                    "num_params_total": ll.num_params_total,
                    "containing_module": ll.containing_module,
                    "is_input_layer": ll.is_input_layer,
                    "is_output_layer": ll.is_output_layer,
                    "is_buffer_layer": ll.is_buffer_layer,
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
