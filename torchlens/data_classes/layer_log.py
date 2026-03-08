"""LayerLog and LayerAccessor: aggregate per-layer metadata and dict-like accessor.

LayerLog groups one or more LayerPassLog entries that represent the same
logical layer across recurrent passes.  For non-recurrent models (the
common case), every LayerLog wraps exactly one LayerPassLog.

**Delegation pattern**: For single-pass layers, per-pass fields (tensor_contents,
grad_contents, operation_num, etc.) are accessible directly on the LayerLog
via ``_single_pass_or_error()`` and ``__getattr__`` delegation to ``passes[1]``.
For multi-pass layers, accessing these fields raises ``ValueError`` (NOT
``AttributeError``) directing the user to ``layer_log.passes[N].field``.

Why ValueError instead of AttributeError: Python's property protocol treats
``AttributeError`` from a ``@property`` as "attribute doesn't exist" and falls
through to ``__getattr__``.  Using ``ValueError`` avoids this trap and gives
the user a clear error message.

**_build_layer_logs merge rules** (in postprocess/layer_log.py):
When merging multiple passes into one LayerLog, only 3 fields are merged:
  - ``has_input_ancestor``: OR across passes
  - ``input_output_address``: character-level merge of "I", "O", "IO" strings
  - ``is_bottom_level_submodule_output``: OR across passes
All other 78+ fields use the first pass's values only.
``modules_exited`` / ``module_passes_exited`` are NOT updated across passes
(correct because same-layer grouping requires identical structural position).
"""

import weakref
from typing import Dict, List, Optional, Union, TYPE_CHECKING

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
    ``__getattr__`` delegation (e.g. ``layer_log.tensor_contents`` transparently
    reads from ``passes[1].tensor_contents``).
    """

    def __init__(self, first_pass: "LayerPassLog"):
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
        self.num_passes = first_pass.layer_passes_total
        # Store as weakref to break circular reference (ModelLog -> layer_logs -> LayerLog -> ModelLog).
        _sml = first_pass.source_model_log
        self._source_model_log_ref: Optional[weakref.ref] = (
            weakref.ref(_sml) if _sml is not None else None
        )

        # Function identity
        self.func_applied = first_pass.func_applied
        self.func_applied_name = first_pass.func_applied_name
        self.function_is_inplace = first_pass.function_is_inplace
        self.gradfunc = first_pass.gradfunc
        self.func_argnames = first_pass.func_argnames
        self.num_func_args_total = first_pass.num_func_args_total
        self.num_position_args = first_pass.num_position_args
        self.num_keyword_args = first_pass.num_keyword_args
        self.is_part_of_iterable_output = first_pass.is_part_of_iterable_output
        self.iterable_output_index = first_pass.iterable_output_index

        # Tensor type (representative from first pass)
        self.tensor_shape = first_pass.tensor_shape
        self.tensor_dtype = first_pass.tensor_dtype
        self.tensor_fsize = first_pass.tensor_fsize

        # Config
        self.output_device = first_pass.output_device
        self.activation_postfunc = first_pass.activation_postfunc
        self.detach_saved_tensor = first_pass.detach_saved_tensor
        self.save_gradients = first_pass.save_gradients

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
        self.parent_params_fsize = first_pass.parent_params_fsize

        # Equivalence
        self.operation_equivalence_type = first_pass.operation_equivalence_type
        self.equivalent_operations = first_pass.equivalent_operations

        # Special flags
        self.is_input_layer = first_pass.is_input_layer
        self.is_output_layer = first_pass.is_output_layer
        self.is_last_output_layer = first_pass.is_last_output_layer
        self.is_buffer_layer = first_pass.is_buffer_layer
        self.buffer_address = first_pass.buffer_address
        self.buffer_parent = first_pass.buffer_parent
        self.initialized_inside_model = first_pass.initialized_inside_model
        self.terminated_inside_model = first_pass.terminated_inside_model
        self.is_terminal_bool_layer = first_pass.is_terminal_bool_layer
        self.is_atomic_bool_layer = first_pass.is_atomic_bool_layer
        self.atomic_bool_val = first_pass.atomic_bool_val

        # Module (static containment)
        self.containing_module_origin = first_pass.containing_module_origin
        self.containing_modules_origin_nested = first_pass.containing_modules_origin_nested

        # Fields stored as aggregate for vis compatibility.
        # Initialized from first pass.  For multi-pass layers, _build_layer_logs
        # merges only has_input_ancestor (OR), input_output_address (char-merge),
        # and is_bottom_level_submodule_output (OR).  All others keep first-pass values.
        self.modules_exited = first_pass.modules_exited
        self.module_passes_exited = first_pass.module_passes_exited
        self.cond_branch_start_children = first_pass.cond_branch_start_children
        self.has_input_ancestor = first_pass.has_input_ancestor
        self.input_output_address = first_pass.input_output_address
        self.buffer_pass = first_pass.buffer_pass
        self.is_bottom_level_submodule_output = first_pass.is_bottom_level_submodule_output

        # Pass management
        self.passes: Dict[int, "LayerPassLog"] = {}
        self.pass_labels: List[str] = []

    @property
    def macs_forward(self) -> Optional[int]:
        """Forward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return self.flops_forward // 2 if self.flops_forward is not None else None

    @property
    def macs_backward(self) -> Optional[int]:
        """Backward MACs (multiply-accumulate ops). 1 MAC = 2 FLOPs."""
        return self.flops_backward // 2 if self.flops_backward is not None else None

    @property
    def computed_with_params(self) -> bool:
        """Whether this layer uses model parameters."""
        return len(self.parent_param_barcodes) > 0

    @property
    def num_param_tensors(self) -> int:
        """Number of parameter tensors used by this layer."""
        return len(self.parent_param_barcodes)

    @property
    def is_computed_inside_submodule(self) -> bool:
        """Whether this layer was computed inside a submodule."""
        return self.containing_module_origin is not None

    @property
    def module_nesting_depth(self) -> int:
        """Depth of module nesting for this layer."""
        return len(self.containing_modules_origin_nested)

    @property
    def tensor_fsize_nice(self) -> str:
        return human_readable_size(self.tensor_fsize)

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
    # ******* Single-pass delegation *************
    # ********************************************
    # For single-pass layers, per-pass fields are transparently accessible
    # on the LayerLog itself.  For multi-pass layers, attempting to access
    # these fields raises ValueError directing the user to a specific pass.

    def _single_pass_or_error(self, field_name):
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
    def tensor_contents(self):
        return self._single_pass_or_error("tensor_contents")

    @property
    def has_saved_activations(self):
        return self._single_pass_or_error("has_saved_activations")

    @property
    def creation_args(self):
        return self._single_pass_or_error("creation_args")

    @property
    def creation_kwargs(self):
        return self._single_pass_or_error("creation_kwargs")

    @property
    def grad_contents(self):
        return self._single_pass_or_error("grad_contents")

    @property
    def has_saved_grad(self):
        return self._single_pass_or_error("has_saved_grad")

    @property
    def func_call_stack(self):
        return self._single_pass_or_error("func_call_stack")

    @property
    def func_time_elapsed(self):
        return self._single_pass_or_error("func_time_elapsed")

    @property
    def func_rng_states(self):
        return self._single_pass_or_error("func_rng_states")

    @property
    def operation_num(self):
        return self._single_pass_or_error("operation_num")

    @property
    def pass_num(self):
        return self._single_pass_or_error("pass_num")

    @property
    def realtime_tensor_num(self):
        return self._single_pass_or_error("realtime_tensor_num")

    @property
    def lookup_keys(self):
        return self._single_pass_or_error("lookup_keys")

    # ********************************************
    # ***** Aggregate graph properties ***********
    # ********************************************
    # Graph-edge properties compute the union across all passes, returning
    # no-pass labels (i.e. LayerLog-level identifiers).  This gives a
    # complete picture of which layers are connected across all recurrent
    # iterations.  Order is preserved (first-seen insertion order).

    @property
    def child_layers(self):
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
    def parent_layers(self):
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
    def has_children(self):
        return any(p.has_children for p in self.passes.values())

    @property
    def has_parents(self):
        return any(p.has_parents for p in self.passes.values())

    @property
    def sibling_layers(self):
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
    def has_siblings(self):
        return any(p.has_siblings for p in self.passes.values())

    @property
    def spouse_layers(self):
        """Union of spouse layers (no-pass labels) across all passes."""
        result = []
        seen = set()
        for pass_log in self.passes.values():
            for label in pass_log.spouse_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if no_pass not in seen:
                    seen.add(no_pass)
                    result.append(no_pass)
        return result

    @property
    def has_spouses(self):
        return any(p.has_spouses for p in self.passes.values())

    @property
    def _pass_finished(self):
        sml = self.source_model_log
        if sml is None:
            return True
        return sml._pass_finished

    # ********************************************
    # ****** Convenience properties **************
    # ********************************************

    @property
    def layer_passes_total(self):
        """Alias for num_passes, matching LayerPassLog field name."""
        return self.num_passes

    @property
    def layer_label_no_pass(self):
        """Alias so code expecting layer_label_no_pass works on LayerLog."""
        return self.layer_label

    @property
    def layer_label_no_pass_short(self):
        """Alias so code expecting layer_label_no_pass_short works on LayerLog."""
        return self.layer_label_short

    @property
    def layer_label_w_pass(self):
        """For single-pass layers, return the pass-qualified label."""
        return self._single_pass_or_error("layer_label_w_pass")

    @property
    def layer_label_w_pass_short(self):
        """For single-pass layers, return the short pass-qualified label."""
        return self._single_pass_or_error("layer_label_w_pass_short")

    @property
    def params(self):
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
    def child_layers_per_pass(self):
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
    def parent_layers_per_pass(self):
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
    def child_passes_per_layer(self):
        """Dict[str, List[int]]: for each child layer, which passes connect to it."""
        from collections import defaultdict

        result = defaultdict(list)
        for pass_num, pass_log in self.passes.items():
            for label in pass_log.child_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if pass_num not in result[no_pass]:
                    result[no_pass].append(pass_num)
        return dict(result)

    @property
    def parent_passes_per_layer(self):
        """Dict[str, List[int]]: for each parent layer, which passes connect from it."""
        from collections import defaultdict

        result = defaultdict(list)
        for pass_num, pass_log in self.passes.items():
            for label in pass_log.parent_layers:
                no_pass = self.source_model_log[label].layer_label_no_pass
                if pass_num not in result[no_pass]:
                    result[no_pass].append(pass_num)
        return dict(result)

    @property
    def edges_vary_across_passes(self):
        """Whether graph edges differ across passes."""
        if self.num_passes <= 1:
            return False
        all_pass_lists = list(self.child_passes_per_layer.values()) + list(
            self.parent_passes_per_layer.values()
        )
        return any(len(passes) < self.num_passes for passes in all_pass_lists)

    @property
    def bottom_level_submodule_passes_exited(self):
        """Set of module passes exited across all passes."""
        result = set()
        for pass_log in self.passes.values():
            if pass_log.is_bottom_level_submodule_output:
                result.add(pass_log.bottom_level_submodule_pass_exited)
        return result

    @property
    def parent_layer_arg_locs(self):
        """Merged parent_layer_arg_locs across passes (set-union).

        For single-pass layers, delegates to passes[1].
        For multi-pass, merges arg locs using set-union of no-pass labels.
        """
        if self.num_passes == 1:
            return self.passes[1].parent_layer_arg_locs
        from collections import defaultdict

        result = {"args": {}, "kwargs": {}}
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

    def __getattr__(self, name):
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

    def print_all_fields(self):
        """Print all data fields in the layer."""
        fields_to_exclude = ["source_model_log", "func_rng_states"]
        for field in dir(self):
            attr = getattr(self, field)
            if not any([field.startswith("_"), field in fields_to_exclude, callable(attr)]):
                print(f"{field}: {attr}")

    def get_child_layers(self):
        return [self.source_model_log[child_label] for child_label in self.child_layers]

    def get_parent_layers(self):
        return [self.source_model_log[parent_label] for parent_label in self.parent_layers]

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self):
        if not self._pass_finished:
            return f"LayerLog({self.layer_label}) (pass not finished)"
        s = f"Layer {self.layer_label}:"
        if self.num_passes > 1:
            s += f" ({self.num_passes} passes)"
        s += (
            f"\n\tOutput tensor: shape={self.tensor_shape}, "
            f"dtype={self.tensor_dtype}, size={self.tensor_fsize_nice}"
        )
        if not self.is_input_layer:
            s += f"\n\tFunction: {self.func_applied_name} (grad_fn: {self.gradfunc})"
        if self.containing_module_origin is not None:
            s += f"\n\tComputed inside module: {self.containing_module_origin}"
        if len(self.parent_param_shapes) > 0:
            params_shapes_str = ", ".join(str(ps) for ps in self.parent_param_shapes)
            s += (
                f"\n\tParams: {params_shapes_str}; "
                f"{self.num_params_total} total ({self.parent_params_fsize_nice})"
            )
        s += "\n\tRelated Layers:"
        s += f"\n\t\t- parents: {', '.join(self.parent_layers) or 'none'}"
        s += f"\n\t\t- children: {', '.join(self.child_layers) or 'none'}"
        if self.num_passes > 1:
            s += f"\n\tPasses: {', '.join(self.pass_labels)}"
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.num_passes


class LayerAccessor:
    """Dict-like accessor for LayerLog objects.

    Supports indexing by:
    * **layer label** (str) -- exact match against no-pass label.
    * **ordinal index** (int) -- position in execution order.
    * **pass notation** (str ``"conv2d_1_1:2"``) -- returns the LayerPassLog
      for a specific pass of a multi-pass layer.

    Available as ``model_log.layers``.
    """

    def __init__(
        self,
        layer_logs: Dict[str, "LayerLog"],
        source_model_log: Optional["ModelLog"] = None,
    ):
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
        raise KeyError(f"Layer '{key}' not found. Valid labels: {list(self._dict.keys())[:10]}...")

    def __contains__(self, key) -> bool:
        return key in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        """Iterate over LayerLog objects in execution order."""
        return iter(self._list)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "LayerAccessor({})"
        items = []
        for ll in self._list:
            items.append(
                f"  '{ll.layer_label}': {ll.func_applied_name or 'input'} "
                f"(shape={list(ll.tensor_shape) if ll.tensor_shape else '?'}, "
                f"passes={ll.num_passes})"
            )
        inner = "\n".join(items)
        return f"LayerAccessor({len(self)} layers):\n{inner}"

    def to_pandas(self) -> "pd.DataFrame":
        """One row per unique layer (aggregate view)."""
        import pandas as pd

        rows = []
        for ll in self._list:
            rows.append(
                {
                    "layer_label": ll.layer_label,
                    "layer_type": ll.layer_type,
                    "func_applied_name": ll.func_applied_name,
                    "tensor_shape": ll.tensor_shape,
                    "tensor_dtype": ll.tensor_dtype,
                    "tensor_fsize_nice": ll.tensor_fsize_nice,
                    "num_passes": ll.num_passes,
                    "num_params_total": ll.num_params_total,
                    "containing_module_origin": ll.containing_module_origin,
                    "is_input_layer": ll.is_input_layer,
                    "is_output_layer": ll.is_output_layer,
                    "is_buffer_layer": ll.is_buffer_layer,
                }
            )
        return pd.DataFrame(rows)
