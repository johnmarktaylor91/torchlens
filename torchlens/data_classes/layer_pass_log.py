"""LayerPassLog: per-operation metadata for a single invocation of a layer.

Each LayerPassLog records everything about one tensor operation in the
forward pass: the output tensor itself, the function that produced it,
its parents/children in the computation graph, module containment,
parameter usage, timing, RNG state, and more.

For recurrent models, the same "layer" may execute multiple times; each
execution is a separate LayerPassLog with a distinct ``pass_num``.  The
aggregate view across passes is provided by :class:`LayerLog`.

Field categories (matching the LAYER_PASS_LOG_FIELD_ORDER in constants.py):

1. **General info** — raw/final labels, operation numbering, back-reference
   to the owning ModelLog.
2. **Label info** — human-readable labels in various formats (with/without
   pass qualifier, short form, etc.).
3. **Saved tensor info** — the tensor contents, shape, dtype, size, device
   transfer settings, activation postfunc, and function arguments.
4. **Child tensor variations** — tracks per-child input values for
   validation replay (``children_tensor_versions`` stores RAW values
   because validation compares against ``creation_args``).
5. **Gradient info** — gradient tensor and metadata (stored as a bare
   reference via ``log_tensor_grad``, not deep-copied).
6. **Function call info** — the applied function, call stack, timing,
   FLOPs, RNG state, arg metadata, grad_fn, inplace flag.
7. **Param info** — which parameters were used, their shapes and sizes.
8. **Equivalence info** — loop-detection equivalence type and groups.
9. **Graph info** — parent/child/sibling/spouse edges, input/output
   ancestry, distances, buffer/internal-init status.
10. **Conditional info** — boolean branching metadata.
11. **Module info** — module entry/exit tracking, nesting depth,
    bottom-level submodule output status.
"""

import copy
import weakref
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch

from ..constants import LAYER_PASS_LOG_FIELD_ORDER
from .._state import pause_logging
from ..utils.tensor_utils import get_tensor_memory_amount, print_override, safe_copy, safe_to
from ..utils.display import human_readable_size

_LAYER_PASS_LOG_FIELD_ORDER_SET = frozenset(LAYER_PASS_LOG_FIELD_ORDER)


def _recursive_safe_copy(val):
    """Deep-copy nested structures, cloning tensors instead of using copy.deepcopy (#44)."""
    if isinstance(val, torch.Tensor):
        return safe_copy(val)
    elif isinstance(val, (list, tuple)):
        return type(val)(_recursive_safe_copy(v) for v in val)
    elif isinstance(val, dict):
        return {k: _recursive_safe_copy(v) for k, v in val.items()}
    return safe_copy(val)


if TYPE_CHECKING:
    from .func_call_location import FuncCallLocation
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

    def __init__(self, fields_dict: Dict):
        """Initialise from a complete fields dictionary.

        Args:
            fields_dict: Dict with values for all fields defined in
                ``LAYER_PASS_LOG_FIELD_ORDER``.  Missing or extra keys
                raise ``ValueError``.
        """
        # Attributes are set explicitly (not via loop) for IDE autocompletion.

        # Validate that fields_dict has exactly the expected keys:
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
        self.realtime_tensor_num = fields_dict["realtime_tensor_num"]
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
        self.layer_passes_total = fields_dict["layer_passes_total"]
        self.lookup_keys = fields_dict["lookup_keys"]

        # Saved tensor info:
        self.tensor_contents = fields_dict["tensor_contents"]
        self.has_saved_activations = fields_dict["has_saved_activations"]
        self.output_device = fields_dict["output_device"]
        self.activation_postfunc = fields_dict["activation_postfunc"]
        self.detach_saved_tensor = fields_dict["detach_saved_tensor"]
        self.function_args_saved = fields_dict["function_args_saved"]
        self.creation_args = fields_dict["creation_args"]
        self.creation_kwargs = fields_dict["creation_kwargs"]
        self.tensor_shape = fields_dict["tensor_shape"]
        self.tensor_dtype = fields_dict["tensor_dtype"]
        self.tensor_fsize = fields_dict["tensor_fsize"]

        # Child tensor variation tracking — stores the raw tensor values that
        # each child operation received as input.  Must store RAW values (not
        # postprocessed) because validation compares these against creation_args.
        self.has_child_tensor_variations = fields_dict["has_child_tensor_variations"]
        self.children_tensor_versions = fields_dict["children_tensor_versions"]

        # Saved gradient info — gradient is stored as a bare clone (not deep-copied)
        # via log_tensor_grad().  grad_contents is populated by a backward hook.
        self.grad_contents = fields_dict["grad_contents"]
        self.save_gradients = fields_dict["save_gradients"]
        self.has_saved_grad = fields_dict["has_saved_grad"]
        self.grad_shape = fields_dict["grad_shape"]
        self.grad_dtype = fields_dict["grad_dtype"]
        self.grad_fsize = fields_dict["grad_fsize"]

        # Function call info:
        self.func_applied = fields_dict["func_applied"]
        self.func_applied_name = fields_dict["func_applied_name"]
        self.func_call_stack: List["FuncCallLocation"] = fields_dict["func_call_stack"]
        self.func_time_elapsed = fields_dict["func_time_elapsed"]
        self.flops_forward = fields_dict["flops_forward"]
        self.flops_backward = fields_dict["flops_backward"]
        self.func_rng_states = fields_dict["func_rng_states"]
        self.func_autocast_state = fields_dict["func_autocast_state"]
        self.func_argnames = fields_dict["func_argnames"]
        self.num_func_args_total = fields_dict["num_func_args_total"]
        self.num_position_args = fields_dict["num_position_args"]
        self.num_keyword_args = fields_dict["num_keyword_args"]
        self.func_position_args_non_tensor = fields_dict["func_position_args_non_tensor"]
        self.func_keyword_args_non_tensor = fields_dict["func_keyword_args_non_tensor"]
        self.func_all_args_non_tensor = fields_dict["func_all_args_non_tensor"]
        self.function_is_inplace = fields_dict["function_is_inplace"]
        self.gradfunc = fields_dict["gradfunc"]
        self.is_part_of_iterable_output = fields_dict["is_part_of_iterable_output"]
        self.iterable_output_index = fields_dict["iterable_output_index"]

        # Param info:
        self.parent_params = fields_dict["parent_params"]
        self.parent_param_barcodes = fields_dict["parent_param_barcodes"]
        self.parent_param_passes = fields_dict["parent_param_passes"]
        self.parent_param_logs: List["ParamLog"] = fields_dict["parent_param_logs"]
        self.parent_param_shapes = fields_dict["parent_param_shapes"]
        self.num_params_total = fields_dict["num_params_total"]
        self.num_params_trainable = fields_dict["num_params_trainable"]
        self.num_params_frozen = fields_dict["num_params_frozen"]
        self.parent_params_fsize = fields_dict["parent_params_fsize"]

        # Loop-detection equivalence info:
        # operation_equivalence_type groups structurally identical operations
        # (same func + same param barcodes).  equivalent_operations holds a
        # DIRECT reference to the ModelLog-level set for this type.
        # same_layer_operations is populated by loop_detection.py for layers
        # that are different passes of the same recurrent layer.
        self.operation_equivalence_type = fields_dict["operation_equivalence_type"]
        self.equivalent_operations = fields_dict["equivalent_operations"]
        self.same_layer_operations = fields_dict["same_layer_operations"]

        # Graph info:
        self.parent_layers = fields_dict["parent_layers"]
        self.parent_layer_arg_locs = fields_dict["parent_layer_arg_locs"]
        self.orig_ancestors = fields_dict["orig_ancestors"]
        self.child_layers = fields_dict["child_layers"]
        self.has_children = fields_dict["has_children"]
        self.is_input_layer = fields_dict["is_input_layer"]
        self.has_input_ancestor = fields_dict["has_input_ancestor"]
        self.input_ancestors = fields_dict["input_ancestors"]
        self.min_distance_from_input = fields_dict["min_distance_from_input"]
        self.max_distance_from_input = fields_dict["max_distance_from_input"]
        self.is_output_layer = fields_dict["is_output_layer"]
        self.is_output_parent = fields_dict["is_output_parent"]
        self.is_last_output_layer = fields_dict["is_last_output_layer"]
        self.is_output_ancestor = fields_dict["is_output_ancestor"]
        self.output_descendents = fields_dict["output_descendents"]
        self.min_distance_from_output = fields_dict["min_distance_from_output"]
        self.max_distance_from_output = fields_dict["max_distance_from_output"]
        self.input_output_address = fields_dict["input_output_address"]
        self.is_buffer_layer = fields_dict["is_buffer_layer"]
        self.buffer_address = fields_dict["buffer_address"]
        self.buffer_pass = fields_dict["buffer_pass"]
        self.buffer_parent = fields_dict["buffer_parent"]
        self.initialized_inside_model = fields_dict["initialized_inside_model"]
        self.has_internally_initialized_ancestor = fields_dict[
            "has_internally_initialized_ancestor"
        ]
        self.internally_initialized_parents = fields_dict["internally_initialized_parents"]
        self.internally_initialized_ancestors = fields_dict["internally_initialized_ancestors"]
        self.terminated_inside_model = fields_dict["terminated_inside_model"]

        # Conditional info
        self.is_terminal_bool_layer = fields_dict["is_terminal_bool_layer"]
        self.is_atomic_bool_layer = fields_dict["is_atomic_bool_layer"]
        self.atomic_bool_val = fields_dict["atomic_bool_val"]
        self.in_cond_branch = fields_dict["in_cond_branch"]
        self.cond_branch_start_children = fields_dict["cond_branch_start_children"]
        self.cond_branch_then_children = fields_dict["cond_branch_then_children"]

        # Module info
        self.containing_module_origin = fields_dict["containing_module_origin"]
        self.containing_modules_origin_nested = fields_dict["containing_modules_origin_nested"]
        self.modules_entered = fields_dict["modules_entered"]
        self.modules_entered_argnames = fields_dict["modules_entered_argnames"]
        self.module_passes_entered = fields_dict["module_passes_entered"]
        self.modules_exited = fields_dict["modules_exited"]
        self.module_passes_exited = fields_dict["module_passes_exited"]
        self.is_submodule_output = fields_dict["is_submodule_output"]
        self.is_bottom_level_submodule_output = fields_dict["is_bottom_level_submodule_output"]
        self.bottom_level_submodule_pass_exited = fields_dict["bottom_level_submodule_pass_exited"]
        self.module_entry_exit_threads_inputs = fields_dict["module_entry_exit_threads_inputs"]
        self.module_entry_exit_thread_output = fields_dict["module_entry_exit_thread_output"]

        # Function config — lightweight hyperparameters always captured.
        self.func_config = fields_dict["func_config"]

        # Back-reference to the aggregate LayerLog that groups all passes of
        # this layer.  Set during postprocessing by _build_layer_logs — NOT
        # part of fields_dict or FIELD_ORDER (it's a structural link, not
        # captured data).
        self.parent_layer_log = None

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
    def spouse_layers(self) -> list:
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
    def has_spouses(self) -> bool:
        """Whether this layer shares children with other layers."""
        return len(self.spouse_layers) > 0

    @property
    def computed_with_params(self) -> bool:
        """Whether this operation used model parameters."""
        return len(self.parent_param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this operation."""
        return len(self.parent_param_barcodes)

    @property
    def is_computed_inside_submodule(self) -> bool:
        """Whether this operation was computed inside a submodule."""
        return self.containing_module_origin is not None

    @property
    def module_nesting_depth(self) -> int:
        """Depth of module nesting for this operation."""
        return len(self.containing_modules_origin_nested)

    @property
    def is_submodule_input(self) -> bool:
        """Whether this operation is the first inside a submodule's forward()."""
        return len(self.modules_entered) > 0

    @property
    def tensor_fsize_nice(self) -> str:
        return human_readable_size(self.tensor_fsize)

    @property
    def grad_fsize_nice(self) -> str:
        return human_readable_size(self.grad_fsize)

    @property
    def parent_params_fsize_nice(self) -> str:
        return human_readable_size(self.parent_params_fsize)

    @property
    def source_model_log(self) -> "ModelLog":
        """Back-reference to the owning ModelLog (stored as weakref)."""
        ref = self.__dict__.get("_source_model_log_ref")
        if ref is None:
            return None  # type: ignore[return-value]
        obj = ref()
        if obj is None:
            raise RuntimeError("ModelLog has been garbage-collected.")
        return obj

    @source_model_log.setter
    def source_model_log(self, value):
        self._source_model_log_ref = weakref.ref(value) if value is not None else None

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    def print_all_fields(self):
        """Print all data fields in the layer."""
        fields_to_exclude = ["source_model_log", "func_rng_states"]

        for field in dir(self):
            attr = getattr(self, field)
            if not any([field.startswith("_"), field in fields_to_exclude, callable(attr)]):
                print(f"{field}: {attr}")

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

    def copy(self):
        """Return a selective-depth copy of this entry.

        Most fields are ``copy.deepcopy``'d so the clone is fully independent.
        However, certain fields are shallow-copied (shared by reference) because:

        * ``func_applied``, ``gradfunc`` — function objects, immutable/shared.
        * ``source_model_log`` — must point to the same ModelLog instance.
        * ``func_rng_states`` — large state dicts, not mutated after capture.
        * ``creation_args``, ``creation_kwargs`` — may contain large tensors;
          deep-copying them is expensive and unnecessary.
        * ``parent_params`` — references to nn.Parameters, must stay shared.
        * ``tensor_contents``, ``children_tensor_versions`` — large tensors;
          shared references are safe since they're replaced (not mutated).

        Returns:
            A new LayerPassLog (or subclass) with the same field values.
        """
        fields_dict = {}
        fields_not_to_deepcopy = [
            "func_applied",
            "gradfunc",
            "source_model_log",
            "func_rng_states",
            "creation_args",
            "creation_kwargs",
            "parent_params",
            "tensor_contents",
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
    ):
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
        # Clone the tensor, optionally detaching from autograd graph.
        self.tensor_contents = safe_copy(t, self.detach_saved_tensor)
        # Move to the user-requested output device if needed.
        if self.output_device not in [str(self.tensor_contents.device), "same"]:
            self.tensor_contents = safe_to(self.tensor_contents, self.output_device)
        # Apply user's postfunc with logging paused so postfunc's own ops
        # (e.g. .mean(), .float()) don't get logged as model operations.
        if activation_postfunc is not None:
            with pause_logging():
                self.tensor_contents = activation_postfunc(self.tensor_contents)

        self.has_saved_activations = True

        # Tensor args and kwargs:
        if save_function_args:
            self.function_args_saved = True
            self.creation_args = [_recursive_safe_copy(arg) for arg in t_args]
            self.creation_kwargs = {k: _recursive_safe_copy(v) for k, v in t_kwargs.items()}
        else:
            self.creation_args = None
            self.creation_kwargs = None

    def log_tensor_grad(self, grad: torch.Tensor):
        """Save the gradient tensor for this layer's output.

        Called by the backward hook registered during the forward pass.
        The gradient is ``detach().clone()``'d — a bare copy, not deep-copied —
        so it's independent of the autograd graph but cheap to store.

        Args:
            grad: The gradient tensor flowing back through this operation.
        """
        self.grad_contents = grad.detach().clone()
        self.has_saved_grad = True
        self.grad_shape = grad.shape
        self.grad_dtype = grad.dtype
        self.grad_fsize = get_tensor_memory_amount(grad)

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def get_child_layers(self):
        return [self.source_model_log[child_label] for child_label in self.child_layers]

    def get_parent_layers(self):
        return [self.source_model_log[parent_label] for parent_label in self.parent_layers]

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
        s += f"\n\tComputed from params: {self.computed_with_params}"
        s += f"\n\tComputed in modules: {self.containing_modules_origin_nested}"
        s += f"\n\tOutput of modules: {self.module_passes_exited}"
        if self.is_bottom_level_submodule_output:
            s += " (bottom-level submodule output)"
        else:
            s += " (not bottom-level submodule output)"
        s += "\n\tFamily info:"
        s += f"\n\t\tParents: {self.parent_layers}"
        s += f"\n\t\tChildren: {self.child_layers}"
        s += f"\n\t\tSpouses: {self.spouse_layers}"
        s += f"\n\t\tSiblings: {self.sibling_layers}"
        s += (
            f"\n\t\tOriginal Ancestors: {self.orig_ancestors} "
            f"(min dist {self.min_distance_from_input} nodes, max dist {self.max_distance_from_input} nodes)"
        )
        s += f"\n\t\tInput Ancestors: {self.input_ancestors}"
        s += f"\n\t\tInternal Ancestors: {self.internally_initialized_ancestors}"
        s += (
            f"\n\t\tOutput Descendents: {self.output_descendents} "
            f"(min dist {self.min_distance_from_output} nodes, max dist {self.max_distance_from_output} nodes)"
        )
        if self.tensor_contents is not None:
            s += f"\n\tTensor contents: \n{print_override(self.tensor_contents, '__str__')}"
        return s

    def _str_after_pass(self) -> str:
        """Return a human-readable summary of this tensor entry after the forward pass has completed."""
        if self.layer_passes_total > 1:
            pass_str = f" (pass {self.pass_num}/{self.layer_passes_total}), "
        else:
            pass_str = ", "
        sml = self.source_model_log
        num_ops = sml.num_operations if sml is not None else "?"
        s = f"Layer {self.layer_label_no_pass}{pass_str}operation {self.operation_num}/{num_ops}:"
        s += f"\n\tOutput tensor: shape={self.tensor_shape}, dype={self.tensor_dtype}, size={self.tensor_fsize_nice}"
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
                f"{self.num_params_total} params total ({self.parent_params_fsize_nice})"
            )
        else:
            s += "\n\tParams: no params used"
        if self.containing_module_origin is None:
            module_str = "\n\tComputed inside module: not computed inside a module"
        else:
            module_str = f"\n\tComputed inside module: {self.containing_module_origin}"
        if not self.is_input_layer:
            s += f"\n\tFunction: {self.func_applied_name} (grad_fn: {self.gradfunc}) {module_str}"
            if self.func_config:
                config_str = ", ".join(f"{k}={v}" for k, v in self.func_config.items())
                s += f"\n\tConfig: {config_str}"
            s += f"\n\tTime elapsed: {self.func_time_elapsed: .3E}s"
        if len(self.modules_exited) > 0:
            modules_exited_str = ", ".join(self.modules_exited)
            s += f"\n\tOutput of modules: {modules_exited_str}"
        else:
            s += "\n\tOutput of modules: none"
        if self.is_bottom_level_submodule_output:
            s += f"\n\tOutput of bottom-level module: {self.bottom_level_submodule_pass_exited}"
        lookup_keys_str = ", ".join([str(key) for key in self.lookup_keys])
        s += f"\n\tLookup keys: {lookup_keys_str}"

        return s

    def _tensor_contents_str_helper(self) -> str:
        """Returns short, readable string for the tensor contents."""
        if self.tensor_contents is None:
            return ""
        else:
            s = ""
            tensor_size_shown = 8
            # Use logged shape, not live tensor shape (#45)
            saved_shape = (
                self.tensor_shape if self.tensor_shape is not None else self.tensor_contents.shape
            )
            # Slice first, then clone only the small slice (#73)
            if len(saved_shape) == 0:
                tensor_slice = self.tensor_contents.detach().clone()
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = self.tensor_contents[0:num_dims].detach().clone()
            elif len(saved_shape) == 2:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.tensor_contents[0:num_dims, 0:num_dims].detach().clone()
            else:
                num_dims = min(tensor_size_shown, saved_shape[-2], saved_shape[-1])
                tensor_slice = self.tensor_contents.data
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

        if len(self.spouse_layers) > 0:
            s += "\n\t\t- shares children with layers: " + ", ".join(self.spouse_layers)
        else:
            s += "\n\t\t- shares children with no other layers"

        if self.has_input_ancestor:
            s += "\n\t\t- descendent of input layers: " + ", ".join(self.input_ancestors)
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.is_output_ancestor:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(self.output_descendents)
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

        return s

    def __repr__(self):
        return self.__str__()


# Backward-compatible alias: TensorLog was the original name for
# LayerPassLog before the LayerLog aggregate class was introduced in PR #92.
TensorLog = LayerPassLog
