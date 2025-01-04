import copy
from collections import defaultdict
from typing import Callable, Dict, List, Optional, TYPE_CHECKING, Tuple, Union

import torch

from .constants import TENSOR_LOG_ENTRY_FIELD_ORDER
from .helper_funcs import clean_to, get_tensor_memory_amount, human_readable_size, print_override, safe_copy

if TYPE_CHECKING:
    from .model_history import ModelHistory


class TensorLogEntry:
    def __init__(self, fields_dict: Dict):
        """Object that stores information about a single tensor operation in the forward pass,
        including metadata and the tensor itself (if specified). Initialized by passing in a dictionary with
        values for all fields.
        Args:
            fields_dict: Dict with values for all fields in TensorLogEntry.
        """
        # Note: this all has to be tediously initialized instead of a for-loop in order for
        # autocomplete features to work well. But, this also serves as a reference for all attributes
        # of a tensor log entry.

        # Check that fields_dict contains all fields for TensorLogEntry:
        field_order_set = set(TENSOR_LOG_ENTRY_FIELD_ORDER)
        fields_dict_key_set = set(fields_dict.keys())
        if fields_dict_key_set != field_order_set:
            error_str = "Error initializing TensorLogEntry:"
            missing_fields = field_order_set - fields_dict_key_set
            extra_fields = fields_dict_key_set - field_order_set
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
        self.source_model_history: "ModelHistory" = fields_dict["source_model_history"]
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
        self.tensor_fsize_nice = fields_dict["tensor_fsize_nice"]

        # Dealing with getitem complexities
        self.was_getitem_applied = fields_dict["was_getitem_applied"]
        self.children_tensor_versions = fields_dict["children_tensor_versions"]

        # Saved gradient info
        self.grad_contents = fields_dict["grad_contents"]
        self.save_gradients = fields_dict["save_gradients"]
        self.has_saved_grad = fields_dict["has_saved_grad"]
        self.grad_shapes = fields_dict["grad_shapes"]
        self.grad_dtypes = fields_dict["grad_dtypes"]
        self.grad_fsizes = fields_dict["grad_fsizes"]
        self.grad_fsizes_nice = fields_dict["grad_fsizes_nice"]

        # Function call info:
        self.func_applied = fields_dict["func_applied"]
        self.func_applied_name = fields_dict["func_applied_name"]
        self.func_call_stack = fields_dict["func_call_stack"]
        self.func_time_elapsed = fields_dict["func_time_elapsed"]
        self.func_rng_states = fields_dict["func_rng_states"]
        self.func_argnames = fields_dict["func_argnames"]
        self.num_func_args_total = fields_dict["num_func_args_total"]
        self.num_position_args = fields_dict["num_position_args"]
        self.num_keyword_args = fields_dict["num_keyword_args"]
        self.func_position_args_non_tensor = fields_dict[
            "func_position_args_non_tensor"
        ]
        self.func_keyword_args_non_tensor = fields_dict["func_keyword_args_non_tensor"]
        self.func_all_args_non_tensor = fields_dict["func_all_args_non_tensor"]
        self.function_is_inplace = fields_dict["function_is_inplace"]
        self.gradfunc = fields_dict["gradfunc"]
        self.is_part_of_iterable_output = fields_dict["is_part_of_iterable_output"]
        self.iterable_output_index = fields_dict["iterable_output_index"]

        # Param info:
        self.computed_with_params = fields_dict["computed_with_params"]
        self.parent_params = fields_dict["parent_params"]
        self.parent_param_barcodes = fields_dict["parent_param_barcodes"]
        self.parent_param_passes = fields_dict["parent_param_passes"]
        self.num_param_tensors = fields_dict["num_param_tensors"]
        self.parent_param_shapes = fields_dict["parent_param_shapes"]
        self.num_params_total = fields_dict["num_params_total"]
        self.parent_params_fsize = fields_dict["parent_params_fsize"]
        self.parent_params_fsize_nice = fields_dict["parent_params_fsize_nice"]

        # Corresponding layer info:
        self.operation_equivalence_type = fields_dict["operation_equivalence_type"]
        self.equivalent_operations = fields_dict["equivalent_operations"]
        self.same_layer_operations = fields_dict["same_layer_operations"]

        # Graph info:
        self.parent_layers = fields_dict["parent_layers"]
        self.has_parents = fields_dict["has_parents"]
        self.parent_layer_arg_locs = fields_dict["parent_layer_arg_locs"]
        self.orig_ancestors = fields_dict["orig_ancestors"]
        self.child_layers = fields_dict["child_layers"]
        self.has_children = fields_dict["has_children"]
        self.sibling_layers = fields_dict["sibling_layers"]
        self.has_siblings = fields_dict["has_siblings"]
        self.spouse_layers = fields_dict["spouse_layers"]
        self.has_spouses = fields_dict["has_spouses"]
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
        self.internally_initialized_parents = fields_dict[
            "internally_initialized_parents"
        ]
        self.internally_initialized_ancestors = fields_dict[
            "internally_initialized_ancestors"
        ]
        self.terminated_inside_model = fields_dict["terminated_inside_model"]

        # Conditional info
        self.is_terminal_bool_layer = fields_dict["is_terminal_bool_layer"]
        self.is_atomic_bool_layer = fields_dict["is_atomic_bool_layer"]
        self.atomic_bool_val = fields_dict["atomic_bool_val"]
        self.in_cond_branch = fields_dict["in_cond_branch"]
        self.cond_branch_start_children = fields_dict["cond_branch_start_children"]

        # Module info
        self.is_computed_inside_submodule = fields_dict["is_computed_inside_submodule"]
        self.containing_module_origin = fields_dict["containing_module_origin"]
        self.containing_modules_origin_nested = fields_dict[
            "containing_modules_origin_nested"
        ]
        self.module_nesting_depth = fields_dict["module_nesting_depth"]
        self.modules_entered = fields_dict["modules_entered"]
        self.modules_entered_argnames = fields_dict["modules_entered_argnames"]
        self.module_passes_entered = fields_dict["module_passes_entered"]
        self.is_submodule_input = fields_dict["is_submodule_input"]
        self.modules_exited = fields_dict["modules_exited"]
        self.module_passes_exited = fields_dict["module_passes_exited"]
        self.is_submodule_output = fields_dict["is_submodule_output"]
        self.is_bottom_level_submodule_output = fields_dict[
            "is_bottom_level_submodule_output"
        ]
        self.bottom_level_submodule_pass_exited = fields_dict[
            "bottom_level_submodule_pass_exited"
        ]
        self.module_entry_exit_threads_inputs = fields_dict[
            "module_entry_exit_threads_inputs"
        ]
        self.module_entry_exit_thread_output = fields_dict[
            "module_entry_exit_thread_output"
        ]

    # ********************************************
    # *********** User-Facing Functions **********
    # ********************************************

    def print_all_fields(self):
        """Print all data fields in the layer."""
        fields_to_exclude = ["source_model_history", "func_rng_states"]

        for field in dir(self):
            attr = getattr(self, field)
            if not any(
                    [field.startswith("_"), field in fields_to_exclude, callable(attr)]
            ):
                print(f"{field}: {attr}")

    # ********************************************
    # ************* Logging Functions ************
    # ********************************************

    def copy(self):
        """Return a copy of itself.

        Returns:
            Copy of itself.
        """
        fields_dict = {}
        fields_not_to_deepcopy = [
            "func_applied",
            "gradfunc",
            "source_model_history",
            "func_rng_states",
            "creation_args",
            "creation_kwargs",
            "parent_params",
            "tensor_contents",
            "children_tensor_versions",
        ]
        for field in TENSOR_LOG_ENTRY_FIELD_ORDER:
            if field not in fields_not_to_deepcopy:
                fields_dict[field] = copy.deepcopy(getattr(self, field))
            else:
                fields_dict[field] = getattr(self, field)
        copied_entry = TensorLogEntry(fields_dict)
        return copied_entry

    def save_tensor_data(
            self,
            t: torch.Tensor,
            t_args: Union[List, Tuple],
            t_kwargs: Dict,
            save_function_args: bool,
            activation_postfunc: Optional[Callable] = None,
    ):
        """Saves the tensor data for a given tensor operation.

        Args:
            t: the tensor.
            t_args: tensor positional arguments for the operation
            t_kwargs: tensor keyword arguments for the operation
            save_function_args: whether to save the arguments to the function
            activation_postfunc: function to apply to activations before saving them
        """
        # The tensor itself:
        self.tensor_contents = safe_copy(t, self.detach_saved_tensor)
        if self.output_device not in [str(self.tensor_contents.device), "same"]:
            self.tensor_contents = clean_to(self.tensor_contents, self.output_device)
        if activation_postfunc is not None:
            self.source_model_history._pause_logging = True
            self.tensor_contents = activation_postfunc(self.tensor_contents)
            self.source_model_history._pause_logging = False

        self.has_saved_activations = True

        # Tensor args and kwargs:
        if save_function_args:
            self.function_args_saved = True
            creation_args = []
            for arg in t_args:
                if type(arg) == list:
                    creation_args.append([safe_copy(a) for a in arg])
                else:
                    creation_args.append(safe_copy(arg))

            creation_kwargs = {}
            for key, value in t_kwargs.items():
                if type(value) == list:
                    creation_kwargs[key] = [safe_copy(v) for v in value]
                else:
                    creation_kwargs[key] = safe_copy(value)
            self.creation_args = creation_args
            self.creation_kwargs = creation_kwargs
        else:
            self.creation_args = None
            self.creation_kwargs = None

    def log_tensor_grad(self, grad: torch.Tensor):
        """Logs the gradient for a tensor to the log entry

        Args:
            grad: The gradient to save.
        """
        self.grad_contents = grad
        self.has_saved_grad = True
        self.grad_shapes = [g.shape for g in grad]
        self.grad_dtypes = [g.dtype for g in grad]
        self.grad_fsizes = [get_tensor_memory_amount(g) for g in grad]
        self.grad_fsizes_nice = [
            human_readable_size(get_tensor_memory_amount(g)) for g in grad
        ]

    # ********************************************
    # ************* Fetcher Functions ************
    # ********************************************

    def get_child_layers(self):
        return [
            self.source_model_history[child_label] for child_label in self.child_layers
        ]

    def get_parent_layers(self):
        return [
            self.source_model_history[parent_label]
            for parent_label in self.parent_layers
        ]

    # ********************************************
    # ************* Built-in Methods *************
    # ********************************************

    def __str__(self):
        if self._pass_finished:
            return self._str_after_pass()
        else:
            return self._str_during_pass()

    def _str_during_pass(self):
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

    def _str_after_pass(self):
        if self.layer_passes_total > 1:
            pass_str = f" (pass {self.pass_num}/{self.layer_passes_total}), "
        else:
            pass_str = ", "
        s = (
            f"Layer {self.layer_label_no_pass}"
            f"{pass_str}operation {self.operation_num}/"
            f"{self.source_model_history.num_operations}:"
        )
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
            s += (
                f"\n\tFunction: {self.func_applied_name} (grad_fn: {self.gradfunc}) "
                f"{module_str}"
            )
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
            saved_shape = self.tensor_contents.shape
            if len(saved_shape) == 0:
                tensor_slice = self.tensor_contents
            elif len(saved_shape) == 1:
                num_dims = min(tensor_size_shown, saved_shape[0])
                tensor_slice = self.tensor_contents[0:num_dims]
            elif len(saved_shape) == 2:
                num_dims = min([tensor_size_shown, saved_shape[-2], saved_shape[-1]])
                tensor_slice = self.tensor_contents[0:num_dims, 0:num_dims]
            else:
                num_dims = min(
                    [tensor_size_shown, self.tensor_shape[-2], self.tensor_shape[-1]]
                )
                tensor_slice = self.tensor_contents.data.clone()
                for _ in range(len(saved_shape) - 2):
                    tensor_slice = tensor_slice[0]
                tensor_slice = tensor_slice[0:num_dims, 0:num_dims]
            tensor_slice = tensor_slice.detach()
            tensor_slice.requires_grad = False
            s += f"\n\t\t{str(tensor_slice)}"
            if (len(saved_shape) > 0) and (max(saved_shape) > tensor_size_shown):
                s += "..."
        return s

    def _tensor_family_str_helper(self) -> str:
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
            s += "\n\t\t- descendent of input layers: " + ", ".join(
                self.input_ancestors
            )
        else:
            s += "\n\t\t- tensor was created de novo inside the model (not computed from input)"

        if self.is_output_ancestor:
            s += "\n\t\t- ancestor of output layers: " + ", ".join(
                self.output_descendents
            )
        else:
            s += "\n\t\t- tensor is not an ancestor of the model output; it terminates within the model"

        return s

    def __repr__(self):
        return self.__str__()


class RolledTensorLogEntry:
    def __init__(self, source_entry: TensorLogEntry):
        """Stripped-down version TensorLogEntry that only encodes the information needed to plot the model
        in its rolled-up form.

        Args:
            source_entry: The source TensorLogEntry from which the rolled node is constructed
        """
        # Label & general info
        self.layer_label = source_entry.layer_label_no_pass
        self.layer_type = source_entry.layer_type
        self.layer_type_num = source_entry.layer_type_num
        self.layer_total_num = source_entry.layer_total_num
        self.layer_passes_total = source_entry.layer_passes_total
        self.source_model_history = source_entry.source_model_history

        # Saved tensor info
        self.tensor_shape = source_entry.tensor_shape
        self.tensor_fsize_nice = source_entry.tensor_fsize_nice

        # Param info:
        self.computed_with_params = source_entry.computed_with_params
        self.parent_param_shapes = source_entry.parent_param_shapes
        self.num_param_tensors = source_entry.num_param_tensors

        # Graph info
        self.is_input_layer = source_entry.is_input_layer
        self.has_input_ancestor = source_entry.has_input_ancestor
        self.is_output_layer = source_entry.is_output_layer
        self.is_last_output_layer = source_entry.is_last_output_layer
        self.is_buffer_layer = source_entry.is_buffer_layer
        self.buffer_address = source_entry.buffer_address
        self.buffer_pass = source_entry.buffer_pass
        self.input_output_address = source_entry.input_output_address
        self.cond_branch_start_children = source_entry.cond_branch_start_children
        self.is_terminal_bool_layer = source_entry.is_terminal_bool_layer
        self.atomic_bool_val = source_entry.atomic_bool_val
        self.child_layers = []
        self.parent_layers = []
        self.orphan_layers = []

        # Module info:
        self.containing_modules_origin_nested = (
            source_entry.containing_modules_origin_nested
        )
        self.modules_exited = source_entry.modules_exited
        self.module_passes_exited = source_entry.module_passes_exited
        self.is_bottom_level_submodule_output = False
        self.bottom_level_submodule_passes_exited = set()

        # Fields specific to rolled node to fill in:
        self.edges_vary_across_passes = False
        self.child_layers_per_pass = defaultdict(list)
        self.child_passes_per_layer = defaultdict(list)
        self.parent_layers_per_pass = defaultdict(list)
        self.parent_passes_per_layer = defaultdict(list)

        # Each one will now be a list of layers, since they can vary across passes.
        self.parent_layer_arg_locs = {
            "args": defaultdict(set),
            "kwargs": defaultdict(set),
        }

    def update_data(self, source_node: TensorLogEntry):
        """Updates the data as need be.
        Args:
            source_node: the source node
        """
        if source_node.has_input_ancestor:
            self.has_input_ancestor = True
        if not any(
                [
                    self.input_output_address is None,
                    source_node.input_output_address is None,
                ]
        ):
            self.input_output_address = "".join(
                [
                    char if (source_node.input_output_address[c] == char) else "*"
                    for c, char in enumerate(self.input_output_address)
                ]
            )
            if self.input_output_address[-1] == ".":
                self.input_output_address = self.input_output_address[:-1]
            if self.input_output_address[-1] == "*":
                self.input_output_address = self.input_output_address.strip("*") + "*"

    def add_pass_info(self, source_node: TensorLogEntry):
        """Adds information about another pass of the same layer: namely, mark information about what the
        child and parent layers are for each pass.

        Args:
            source_node: Information for the source pass
        """
        # Label the layers for each pass
        child_layer_labels = [
            self.source_model_history[child].layer_label_no_pass
            for child in source_node.child_layers
        ]
        for child_layer in child_layer_labels:
            if child_layer not in self.child_layers:
                self.child_layers.append(child_layer)
            if child_layer not in self.child_layers_per_pass[source_node.pass_num]:
                self.child_layers_per_pass[source_node.pass_num].append(child_layer)

        parent_layer_labels = [
            self.source_model_history[parent].layer_label_no_pass
            for parent in source_node.parent_layers
        ]
        for parent_layer in parent_layer_labels:
            if parent_layer not in self.parent_layers:
                self.parent_layers.append(parent_layer)
            if parent_layer not in self.parent_layers_per_pass[source_node.pass_num]:
                self.parent_layers_per_pass[source_node.pass_num].append(parent_layer)

        # Label the passes for each layer, and indicate if any layers vary based on the pass.
        for child_layer in source_node.child_layers:
            child_layer_label = self.source_model_history[
                child_layer
            ].layer_label_no_pass
            if (
                    source_node.pass_num
                    not in self.child_passes_per_layer[child_layer_label]
            ):
                self.child_passes_per_layer[child_layer_label].append(
                    source_node.pass_num
                )

        for parent_layer in source_node.parent_layers:
            parent_layer_label = self.source_model_history[
                parent_layer
            ].layer_label_no_pass
            if (
                    source_node.pass_num
                    not in self.parent_passes_per_layer[parent_layer_label]
            ):
                self.parent_passes_per_layer[parent_layer_label].append(
                    source_node.pass_num
                )

        # Check if any edges vary across passes.
        if source_node.pass_num == source_node.layer_passes_total:
            pass_lists = list(self.parent_passes_per_layer.values()) + list(
                self.child_passes_per_layer.values()
            )
            pass_lens = [len(passes) for passes in pass_lists]
            if any(
                    [pass_len < source_node.layer_passes_total for pass_len in pass_lens]
            ):
                self.edges_vary_across_passes = True
            else:
                self.edges_vary_across_passes = False

        # Add submodule info:
        if source_node.is_bottom_level_submodule_output:
            self.is_bottom_level_submodule_output = True
            self.bottom_level_submodule_passes_exited.add(
                source_node.bottom_level_submodule_pass_exited
            )

        # For the parent arg locations, have a list of layers rather than single layer, since they can
        # vary across passes.

        for arg_type in ["args", "kwargs"]:
            for arg_key, layer_label in source_node.parent_layer_arg_locs[
                arg_type
            ].items():
                layer_label_no_pass = self.source_model_history[
                    layer_label
                ].layer_label_no_pass
                self.parent_layer_arg_locs[arg_type][arg_key].add(layer_label_no_pass)

    def __str__(self) -> str:
        fields_not_to_print = ["source_model_history"]
        s = ""
        for field in dir(self):
            attr = getattr(self, field)
            if (
                    not field.startswith("_")
                    and field not in fields_not_to_print
                    and not (callable(attr))
            ):
                s += f"{field}: {attr}\n"
        return s

    def __repr__(self):
        return self.__str__()
