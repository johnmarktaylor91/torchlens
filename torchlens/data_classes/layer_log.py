"""LayerLog: aggregate per-layer metadata grouping one or more LayerPassLog entries."""

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
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
        self.source_model_log: "ModelLog" = first_pass.source_model_log

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
        self.tensor_fsize_nice = first_pass.tensor_fsize_nice

        # Config
        self.output_device = first_pass.output_device
        self.activation_postfunc = first_pass.activation_postfunc
        self.detach_saved_tensor = first_pass.detach_saved_tensor
        self.save_gradients = first_pass.save_gradients

        # FLOPs
        self.flops_forward = first_pass.flops_forward
        self.flops_backward = first_pass.flops_backward

        # Param identity
        self.computed_with_params = first_pass.computed_with_params
        self.parent_param_barcodes = first_pass.parent_param_barcodes
        self.parent_param_logs: List["ParamLog"] = first_pass.parent_param_logs
        self.num_param_tensors = first_pass.num_param_tensors
        self.parent_param_shapes = first_pass.parent_param_shapes
        self.num_params_total = first_pass.num_params_total
        self.num_params_trainable = first_pass.num_params_trainable
        self.num_params_frozen = first_pass.num_params_frozen
        self.parent_params_fsize = first_pass.parent_params_fsize
        self.parent_params_fsize_nice = first_pass.parent_params_fsize_nice

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
        self.is_computed_inside_submodule = first_pass.is_computed_inside_submodule
        self.containing_module_origin = first_pass.containing_module_origin
        self.containing_modules_origin_nested = first_pass.containing_modules_origin_nested
        self.module_nesting_depth = first_pass.module_nesting_depth

        # Fields stored as aggregate for vis compatibility
        # (initialized from first pass, may be updated in _build_layer_logs for multi-pass)
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

    # ********************************************
    # ******* Single-pass delegation *************
    # ********************************************

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
        return self.source_model_log._pass_finished

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
    # These provide per-pass edge tracking for visualization.
    # They are computed on-the-fly from the passes dict.

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

        result = {"args": defaultdict(set), "kwargs": defaultdict(set)}
        for pass_log in self.passes.values():
            for arg_type in ["args", "kwargs"]:
                for arg_key, layer_label in pass_log.parent_layer_arg_locs[arg_type].items():
                    no_pass = self.source_model_log[layer_label].layer_label_no_pass
                    result[arg_type][arg_key].add(no_pass)
        return result

    # ********************************************
    # ******* Fallback __getattr__ ***************
    # ********************************************

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails.
        # For single-pass layers, delegate to passes[1].
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
