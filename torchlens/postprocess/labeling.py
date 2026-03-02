"""Steps 9-12: Label mapping, final info logging, renaming, cleanup, and lookup keys."""

import warnings
from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

from ..constants import MODEL_LOG_FIELD_ORDER, TENSOR_LOG_FIELD_ORDER
from ..helper_funcs import human_readable_size
from ..data_classes.tensor_log import TensorLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _map_raw_tensor_labels_to_final_tensor_labels(self):
    """
    Determines the final label for each tensor, and stores this mapping as a dictionary
    in order to then go through and rename everything in the next preprocessing step.
    """
    raw_to_final_layer_labels = {}
    final_to_raw_layer_labels = {}
    layer_type_counter = defaultdict(lambda: 1)
    layer_total_counter = 1
    for tensor_log_entry in self:
        layer_type = tensor_log_entry.layer_type
        pass_num = tensor_log_entry.pass_num
        if pass_num == 1:
            layer_type_num = layer_type_counter[layer_type]
            layer_type_counter[layer_type] += 1
            if layer_type in ["input", "buffer"]:
                layer_total_num = 0
            else:
                layer_total_num = layer_total_counter
                layer_total_counter += 1

        else:  # inherit layer numbers from first pass of the layer
            first_pass_tensor = self[tensor_log_entry.same_layer_operations[0]]
            layer_type_num = first_pass_tensor.layer_type_num
            if layer_type in ["input", "buffer"]:
                layer_total_num = 0
            else:
                layer_total_num = first_pass_tensor.layer_total_num
        tensor_log_entry.layer_type_num = layer_type_num
        tensor_log_entry.layer_total_num = layer_total_num

        if layer_type not in ["input", "output", "buffer"]:
            tensor_log_entry.layer_label_w_pass = (
                f"{layer_type}_{layer_type_num}_{layer_total_num}:{pass_num}"
            )
            tensor_log_entry.layer_label_no_pass = (
                f"{layer_type}_{layer_type_num}_{layer_total_num}"
            )
        else:
            tensor_log_entry.layer_label_w_pass = f"{layer_type}_{layer_type_num}:{pass_num}"
            tensor_log_entry.layer_label_no_pass = f"{layer_type}_{layer_type_num}"

        tensor_log_entry.layer_label_w_pass_short = f"{layer_type}_{layer_type_num}:{pass_num}"
        tensor_log_entry.layer_label_no_pass_short = f"{layer_type}_{layer_type_num}"
        if tensor_log_entry.layer_passes_total == 1:
            tensor_log_entry.layer_label = tensor_log_entry.layer_label_no_pass
            tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_no_pass_short
        else:
            tensor_log_entry.layer_label = tensor_log_entry.layer_label_w_pass
            tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_w_pass_short
        raw_to_final_layer_labels[tensor_log_entry.tensor_label_raw] = tensor_log_entry.layer_label
        final_to_raw_layer_labels[tensor_log_entry.layer_label] = tensor_log_entry.tensor_label_raw
    self._raw_to_final_layer_labels = raw_to_final_layer_labels
    self._final_to_raw_layer_labels = final_to_raw_layer_labels


def _log_final_info_for_all_layers(self):
    """
    Goes through all layers (before discarding unsaved ones), and logs final info about the model
    and the layers that pertains to all layers (not just saved ones).
    """
    unique_layers_seen = set()  # to avoid double-counting params of recurrent layers
    operation_num = 1
    for t, tensor_entry in enumerate(self):
        if tensor_entry.layer_type in ["input", "buffer"]:
            tensor_entry.operation_num = 0
        elif tensor_entry.layer_type == "output":
            tensor_entry.operation_num = None  # fix later
        else:
            tensor_entry.operation_num = operation_num
            self.num_operations += 1
            operation_num += 1

        # Replace any layer names with their final names:
        _replace_layer_names_for_tensor_entry(self, tensor_entry)

        # Log the module hierarchy information:
        _log_module_hierarchy_info_for_layer(self, tensor_entry)
        if tensor_entry.bottom_level_submodule_pass_exited is not None:
            submodule_pass_nice_name = ":".join(
                [str(i) for i in tensor_entry.bottom_level_submodule_pass_exited]
            )
            tensor_entry.bottom_level_submodule_pass_exited = submodule_pass_nice_name

        # Tally the tensor sizes:
        self.tensor_fsize_total += tensor_entry.tensor_fsize

        # Tally the parameter sizes:
        if tensor_entry.layer_label_no_pass not in unique_layers_seen:  # only count params once
            if tensor_entry.computed_with_params:
                self.total_param_layers += 1
            self.total_params += tensor_entry.num_params_total
            self.total_params_trainable += tensor_entry.num_params_trainable
            self.total_params_frozen += tensor_entry.num_params_frozen
            self.total_param_tensors += tensor_entry.num_param_tensors
            self.total_params_fsize += tensor_entry.parent_params_fsize
            # Tally for modules, too.
            for module_name, _ in tensor_entry.containing_modules_origin_nested:
                self.module_nparams[module_name] += tensor_entry.num_params_total
                self.module_nparams_trainable[module_name] += tensor_entry.num_params_trainable
                self.module_nparams_frozen[module_name] += tensor_entry.num_params_frozen

        unique_layers_seen.add(tensor_entry.layer_label_no_pass)

        # Tally elapsed time:

        self.elapsed_time_function_calls += tensor_entry.func_time_elapsed

        # Update model structural information:
        if len(tensor_entry.child_layers) > 1:
            self.model_is_branching = True
        if tensor_entry.layer_passes_total > self.model_max_recurrent_loops:
            self.model_is_recurrent = True
            self.model_max_recurrent_loops = tensor_entry.layer_passes_total
        if tensor_entry.in_cond_branch:
            self.model_has_conditional_branching = True

    for layer in self.output_layers:
        self[layer].operation_num = self.num_operations

    # Extract the module hierarchy information
    for module in self.top_level_module_passes:
        module_no_pass = module.split(":")[0]
        if module_no_pass not in self.top_level_modules:
            self.top_level_modules.append(module_no_pass)

    for module_parent, module_children in self.module_pass_children.items():
        module_parent_nopass = module_parent.split(":")[0]
        for module_child in module_children:
            module_child_nopass = module_child.split(":")[0]
            if module_child_nopass not in self.module_children[module_parent_nopass]:
                self.module_children[module_parent_nopass].append(module_child_nopass)

    self.num_tensors_total = len(self)

    # Save the nice versions of the filesize fields:
    self.tensor_fsize_total_nice = human_readable_size(self.tensor_fsize_total)
    self.total_params_fsize_nice = human_readable_size(self.total_params_fsize)


def _replace_layer_names_for_tensor_entry(self, tensor_entry: TensorLog):
    """
    Replaces all layer names in the fields of a TensorLog with their final
    layer names.

    Args:
        tensor_entry: TensorLog to replace layer names for.
    """
    list_fields_to_rename = [
        "parent_layers",
        "orig_ancestors",
        "child_layers",
        "sibling_layers",
        "spouse_layers",
        "input_ancestors",
        "output_descendents",
        "internally_initialized_parents",
        "internally_initialized_ancestors",
        "cond_branch_start_children",
        "equivalent_operations",
        "same_layer_operations",
    ]
    for field in list_fields_to_rename:
        orig_layer_names = getattr(tensor_entry, field)
        field_type = type(orig_layer_names)
        new_layer_names = field_type(
            [self._raw_to_final_layer_labels[raw_name] for raw_name in orig_layer_names]
        )
        setattr(tensor_entry, field, new_layer_names)

    # Fix the arg locations field:
    for arg_type in ["args", "kwargs"]:
        for key, value in tensor_entry.parent_layer_arg_locs[arg_type].items():
            tensor_entry.parent_layer_arg_locs[arg_type][key] = self._raw_to_final_layer_labels[
                value
            ]

    # Fix the field names for different children tensor versions:
    new_child_tensor_versions = {}
    for (
        child_label,
        tensor_version,
    ) in tensor_entry.children_tensor_versions.items():
        new_child_tensor_versions[self._raw_to_final_layer_labels[child_label]] = tensor_version
    tensor_entry.children_tensor_versions = new_child_tensor_versions


def _log_module_hierarchy_info_for_layer(self, tensor_entry: TensorLog):
    """
    Logs the module hierarchy information for a single layer.

    Args:
        tensor_entry: Log entry to mark the module hierarchy info for.
    """
    containing_module_pass_label = None
    for m, module_pass_label in enumerate(tensor_entry.containing_modules_origin_nested):
        module_name, module_pass = module_pass_label
        module_pass_nice_label = f"{module_name}:{module_pass}"
        self.module_num_tensors[module_name] += 1
        self.module_pass_num_tensors[module_pass_nice_label] += 1
        if tensor_entry.layer_label not in self.module_layers[module_name]:
            self.module_layers[module_name].append(tensor_entry.layer_label)
        if tensor_entry.layer_label not in self.module_pass_layers[module_pass_nice_label]:
            self.module_pass_layers[module_pass_nice_label].append(tensor_entry.layer_label)
        if (m == 0) and (module_pass_nice_label not in self.top_level_module_passes):
            self.top_level_module_passes.append(module_pass_nice_label)
        else:
            if (containing_module_pass_label is not None) and (
                module_pass_nice_label
                not in self.module_pass_children[containing_module_pass_label]
            ):
                self.module_pass_children[containing_module_pass_label].append(
                    module_pass_nice_label
                )
        containing_module_pass_label = module_pass_nice_label
        if self.module_num_passes[module_name] < module_pass:
            self.module_num_passes[module_name] = module_pass
        if module_name not in self.module_addresses:
            self.module_addresses.append(module_name)
        if module_pass_label not in self.module_passes:
            self.module_passes.append(module_pass_nice_label)
    tensor_entry.module_nesting_depth = len(tensor_entry.containing_modules_origin_nested)


def _remove_unwanted_entries_and_log_remaining(self):
    """Removes entries from ModelLog that we don't want in the final saved output,
    and logs information about the remaining entries.
    """
    tensors_to_remove = []
    # Quick loop to count how many tensors are saved:
    for tensor_entry in self:
        if tensor_entry.has_saved_activations:
            self.num_tensors_saved += 1

    if self.keep_unsaved_layers:
        num_logged_tensors = len(self)
    else:
        num_logged_tensors = self.num_tensors_saved

    self.layer_list = []
    self.layer_dict_main_keys = {}
    self.layer_labels = []
    self.layer_labels_no_pass = []
    self.layer_labels_w_pass = []
    self.layer_num_passes = {}

    i = 0
    for raw_tensor_label in self._raw_tensor_labels_list:
        tensor_entry = self._raw_tensor_dict[raw_tensor_label]
        # Determine valid lookup keys and relate them to the tensor's realtime operation number:
        if getattr(tensor_entry, "has_saved_activations", False) or self.keep_unsaved_layers:
            # Add the lookup keys for the layer, to itself and to ModelLog:
            _add_lookup_keys_for_tensor_entry(self, tensor_entry, i, num_logged_tensors)

            # Log all information:
            self.layer_list.append(tensor_entry)
            self.layer_dict_main_keys[tensor_entry.layer_label] = tensor_entry
            self.layer_labels.append(tensor_entry.layer_label)
            self.layer_labels_no_pass.append(tensor_entry.layer_label_no_pass)
            self.layer_labels_w_pass.append(tensor_entry.layer_label_w_pass)
            self.layer_num_passes[tensor_entry.layer_label] = tensor_entry.layer_passes_total
            if tensor_entry.has_saved_activations:
                self.tensor_fsize_saved += tensor_entry.tensor_fsize
            _trim_and_reorder_tensor_entry_fields(tensor_entry)  # Final reformatting of fields
            i += 1
        else:
            tensors_to_remove.append(tensor_entry)
            self.unlogged_layers.append(tensor_entry.layer_label)
            self._unsaved_layers_lookup_keys.update(tensor_entry.lookup_keys)

    # Remove unused entries.
    for tensor_entry in tensors_to_remove:
        self._remove_log_entry(tensor_entry, remove_references=False)

    if (self.num_tensors_saved == len(self)) or self.keep_unsaved_layers:
        self._all_layers_logged = True
    else:
        self._all_layers_logged = False

    if self.num_tensors_saved == len(self.layer_list):
        self._all_layers_saved = True
    else:
        self._all_layers_saved = False

    # Make the saved tensor filesize pretty:
    self.tensor_fsize_saved_nice = human_readable_size(self.tensor_fsize_saved)


def _add_lookup_keys_for_tensor_entry(
    self, tensor_entry: TensorLog, tensor_index: int, num_tensors_to_keep: int
):
    """Adds the user-facing lookup keys for a TensorLog, both to itself
    and to the ModelLog top-level record.

    Args:
        tensor_entry: TensorLog to get the lookup keys for.
    """
    # The "default" keys: including the pass if multiple passes, excluding if one pass.
    lookup_keys_for_tensor = [
        tensor_entry.layer_label,
        tensor_entry.layer_label_short,
        tensor_index,
        tensor_index - num_tensors_to_keep,
    ]

    # If just one pass, also allow indexing by pass label.
    if tensor_entry.layer_passes_total == 1:
        lookup_keys_for_tensor.extend(
            [tensor_entry.layer_label_w_pass, tensor_entry.layer_label_w_pass_short]
        )

    # Relabel the module passes if the first pass:
    if self.logging_mode == "exhaustive":
        tensor_entry.module_passes_exited = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in tensor_entry.module_passes_exited
        ]
        tensor_entry.module_passes_entered = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in tensor_entry.module_passes_entered
        ]
        if tensor_entry.containing_module_origin is not None:
            tensor_entry.containing_module_origin = ":".join(
                [str(i) for i in tensor_entry.containing_module_origin]
            )
        tensor_entry.containing_modules_origin_nested = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in tensor_entry.containing_modules_origin_nested
        ]
        if (tensor_entry.containing_module_origin is None) and len(
            tensor_entry.containing_modules_origin_nested
        ) > 0:
            tensor_entry.containing_module_origin = tensor_entry.containing_modules_origin_nested[
                -1
            ]

    # Allow indexing by modules exited as well:
    for module_pass in tensor_entry.module_passes_exited:
        module_name, _ = module_pass.split(":")
        lookup_keys_for_tensor.append(f"{module_pass}")
        if self.module_num_passes[module_name] == 1:
            lookup_keys_for_tensor.append(f"{module_name}")

    # Allow using buffer/input/output address as key, too:
    if tensor_entry.is_buffer_layer:
        if self.buffer_num_passes[tensor_entry.buffer_address] == 1:
            lookup_keys_for_tensor.append(tensor_entry.buffer_address)
        lookup_keys_for_tensor.append(f"{tensor_entry.buffer_address}:{tensor_entry.buffer_pass}")
    elif tensor_entry.is_input_layer or tensor_entry.is_output_layer:
        lookup_keys_for_tensor.append(tensor_entry.input_output_address)

    lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

    # Log in both the tensor and in the ModelLog object.
    tensor_entry.lookup_keys = lookup_keys_for_tensor
    for lookup_key in lookup_keys_for_tensor:
        if lookup_key not in self._lookup_keys_to_tensor_num_dict:
            self._lookup_keys_to_tensor_num_dict[lookup_key] = tensor_entry.realtime_tensor_num
            self.layer_dict_all_keys[lookup_key] = tensor_entry
        self._tensor_num_to_lookup_keys_dict[tensor_entry.realtime_tensor_num].append(lookup_key)


def _trim_and_reorder_tensor_entry_fields(tensor_entry: TensorLog):
    """
    Sorts the fields in TensorLog into their desired order, and trims any
    fields that aren't useful after the pass.
    """
    old_dict = tensor_entry.__dict__
    new_dir_dict = OrderedDict()
    for field in TENSOR_LOG_FIELD_ORDER:
        if field in old_dict:
            new_dir_dict[field] = old_dict[field]
    for field, value in old_dict.items():
        if field not in new_dir_dict:
            new_dir_dict[field] = value
    tensor_entry.__dict__ = new_dir_dict


def _rename_model_history_layer_names(self):
    """Renames all the metadata fields in ModelLog with the final layer names, replacing the
    realtime debugging names.
    """
    list_fields_to_rename = [
        "input_layers",
        "output_layers",
        "buffer_layers",
        "internally_initialized_layers",
        "_layers_where_internal_branches_merge_with_input",
        "internally_terminated_layers",
        "internally_terminated_bool_layers",
        "layers_with_saved_gradients",
        "layers_with_saved_activations",
    ]
    for field in list_fields_to_rename:
        tensor_labels = getattr(self, field)
        setattr(
            self,
            field,
            [self._raw_to_final_layer_labels[tensor_label] for tensor_label in tensor_labels],
        )

    new_param_tensors = {}
    for key, values in self.layers_computed_with_params.items():
        new_key = self[values[0]].layer_label
        new_param_tensors[new_key] = [
            self._raw_to_final_layer_labels[tensor_label] for tensor_label in values
        ]
    self.layers_computed_with_params = new_param_tensors

    new_equiv_operations_tensors = {}
    for key, values in self.equivalent_operations.items():
        new_equiv_operations_tensors[key] = set(
            [self._raw_to_final_layer_labels[tensor_label] for tensor_label in values]
        )
    self.equivalent_operations = new_equiv_operations_tensors

    for t, (child, parent) in enumerate(self.conditional_branch_edges):
        new_child, new_parent = (
            self._raw_to_final_layer_labels[child],
            self._raw_to_final_layer_labels[parent],
        )
        self.conditional_branch_edges[t] = (new_child, new_parent)

    for module_pass, arglist in self.module_layer_argnames.items():
        inds_to_remove = []
        for a, arg in enumerate(arglist):
            raw_name = self.module_layer_argnames[module_pass][a][0]
            if raw_name not in self._raw_to_final_layer_labels:
                inds_to_remove.append(a)
                continue
            new_name = self._raw_to_final_layer_labels[raw_name]
            argname = self.module_layer_argnames[module_pass][a][1]
            self.module_layer_argnames[module_pass][a] = (new_name, argname)
        self.module_layer_argnames[module_pass] = [
            self.module_layer_argnames[module_pass][i]
            for i in range(len(arglist))
            if i not in inds_to_remove
        ]


def _trim_and_reorder_model_history_fields(self):
    """
    Sorts the fields in ModelLog into their desired order, and trims any
    fields that aren't useful after the pass.
    """
    new_dir_dict = OrderedDict()
    for field in MODEL_LOG_FIELD_ORDER:
        new_dir_dict[field] = getattr(self, field)
    for field in dir(self):
        if field.startswith("_"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_dir_dict[field] = getattr(self, field)
    self.__dict__ = new_dir_dict
