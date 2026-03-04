"""Steps 9-12: Label mapping, final info logging, renaming, cleanup, and lookup keys."""

from collections import OrderedDict, defaultdict
from typing import TYPE_CHECKING

from ..constants import MODEL_LOG_FIELD_ORDER, LAYER_PASS_LOG_FIELD_ORDER
from ..utils.display import human_readable_size
from ..data_classes.layer_pass_log import LayerPassLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


def _map_raw_labels_to_final_labels(self) -> None:
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


def _log_final_info_for_all_layers(self) -> None:
    """
    Goes through all layers (before discarding unsaved ones), and logs final info about the model
    and the layers that pertains to all layers (not just saved ones).
    """
    unique_layers_seen = set()  # to avoid double-counting params of recurrent layers
    operation_num = 1
    mbd = self._module_build_data

    # Shadow sets for O(1) membership checks in _log_module_hierarchy_info_for_layer.
    # Lists are kept as primary storage (insertion order matters for downstream consumers),
    # but linear `in` checks on lists are expensive for large models.
    _shadow_sets = {
        "module_layers": defaultdict(set),
        "module_pass_layers": defaultdict(set),
        "top_level_module_passes": set(),
        "module_pass_children": defaultdict(set),
        "module_addresses": set(),
        "module_passes": set(),
    }

    for t, layer_entry in enumerate(self):
        if layer_entry.layer_type in ["input", "buffer"]:
            layer_entry.operation_num = 0
        elif layer_entry.layer_type == "output":
            layer_entry.operation_num = None  # fix later
        else:
            layer_entry.operation_num = operation_num
            self.num_operations += 1
            operation_num += 1

        # Replace any layer names with their final names:
        _replace_layer_names_for_layer_entry(self, layer_entry)

        # Log the module hierarchy information:
        _log_module_hierarchy_info_for_layer(self, layer_entry, _shadow_sets)
        if layer_entry.bottom_level_submodule_pass_exited is not None:
            submodule_pass_nice_name = ":".join(
                [str(i) for i in layer_entry.bottom_level_submodule_pass_exited]
            )
            layer_entry.bottom_level_submodule_pass_exited = submodule_pass_nice_name

        # Tally the tensor sizes:
        self.tensor_fsize_total += layer_entry.tensor_fsize

        # Tally the parameter sizes:
        if layer_entry.layer_label_no_pass not in unique_layers_seen:  # only count params once
            if layer_entry.computed_with_params:
                self.total_param_layers += 1
            self.total_params += layer_entry.num_params_total
            self.total_params_trainable += layer_entry.num_params_trainable
            self.total_params_frozen += layer_entry.num_params_frozen
            self.total_param_tensors += layer_entry.num_param_tensors
            self.total_params_fsize += layer_entry.parent_params_fsize
            # Tally for modules, too.
            for module_name, _ in layer_entry.containing_modules_origin_nested:
                mbd["module_nparams"][module_name] += layer_entry.num_params_total
                mbd["module_nparams_trainable"][module_name] += layer_entry.num_params_trainable
                mbd["module_nparams_frozen"][module_name] += layer_entry.num_params_frozen

        unique_layers_seen.add(layer_entry.layer_label_no_pass)

        # Tally elapsed time:

        self.elapsed_time_function_calls += layer_entry.func_time_elapsed

        # Update model structural information:
        if len(layer_entry.child_layers) > 1:
            self.model_is_branching = True
        if layer_entry.layer_passes_total > self.model_max_recurrent_loops:
            self.model_is_recurrent = True
            self.model_max_recurrent_loops = layer_entry.layer_passes_total
        if layer_entry.in_cond_branch:
            self.model_has_conditional_branching = True

    _finalize_output_operation_nums(self)
    _build_module_hierarchy_dicts(self)

    self.num_tensors_total = len(self)
    self.tensor_fsize_total_nice = human_readable_size(self.tensor_fsize_total)
    self.total_params_fsize_nice = human_readable_size(self.total_params_fsize)


def _finalize_output_operation_nums(self) -> None:
    """Assign operation_num to output layers (deferred until total is known)."""
    for layer in self.output_layers:
        self[layer].operation_num = self.num_operations


def _build_module_hierarchy_dicts(self) -> None:
    """Derive top_level_modules and module_children from their pass-level counterparts."""
    mbd = self._module_build_data
    for module in mbd["top_level_module_passes"]:
        module_no_pass = module.split(":")[0]
        if module_no_pass not in mbd["top_level_modules"]:
            mbd["top_level_modules"].append(module_no_pass)

    for module_parent, module_children in mbd["module_pass_children"].items():
        module_parent_nopass = module_parent.split(":")[0]
        for module_child in module_children:
            module_child_nopass = module_child.split(":")[0]
            if module_child_nopass not in mbd["module_children"][module_parent_nopass]:
                mbd["module_children"][module_parent_nopass].append(module_child_nopass)


_LIST_FIELDS_TO_RENAME = [
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


def _replace_layer_names_for_layer_entry(self, layer_entry: LayerPassLog) -> None:
    """
    Replaces all layer names in the fields of a LayerPassLog with their final
    layer names.

    Args:
        layer_entry: LayerPassLog to replace layer names for.
    """
    mapping = self._raw_to_final_layer_labels
    d = layer_entry.__dict__

    for field in _LIST_FIELDS_TO_RENAME:
        orig = d.get(field)
        if not orig:
            continue
        if isinstance(orig, list):
            d[field] = [mapping[raw] for raw in orig]
        else:  # set
            d[field] = type(orig)(mapping[raw] for raw in orig)

    # Fix the arg locations field:
    arg_locs = d.get("parent_layer_arg_locs")
    if arg_locs:
        for arg_type in ("args", "kwargs"):
            sub = arg_locs[arg_type]
            for key, value in sub.items():
                sub[key] = mapping[value]

    # Fix the field names for different children tensor versions:
    ctv = d.get("children_tensor_versions")
    if ctv:
        d["children_tensor_versions"] = {
            mapping[child_label]: tensor_version for child_label, tensor_version in ctv.items()
        }


def _log_module_hierarchy_info_for_layer(
    self, layer_entry: LayerPassLog, _shadow_sets: dict
) -> None:
    """
    Logs the module hierarchy information for a single layer.

    Args:
        layer_entry: Log entry to mark the module hierarchy info for.
        _shadow_sets: Shadow sets for O(1) membership checks.
    """
    _module_labels_seen = _shadow_sets["module_layers"]
    _module_pass_labels_seen = _shadow_sets["module_pass_layers"]
    _top_level_module_passes_seen = _shadow_sets["top_level_module_passes"]
    _module_pass_children_seen = _shadow_sets["module_pass_children"]
    _module_addresses_seen = _shadow_sets["module_addresses"]
    _module_passes_seen = _shadow_sets["module_passes"]
    mbd = self._module_build_data

    containing_module_pass_label = None
    layer_label = layer_entry.layer_label
    for module_index, module_pass_label in enumerate(layer_entry.containing_modules_origin_nested):
        module_name, module_pass = module_pass_label
        module_pass_nice_label = f"{module_name}:{module_pass}"
        mbd["module_num_tensors"][module_name] += 1
        mbd["module_pass_num_tensors"][module_pass_nice_label] += 1
        if layer_label not in _module_labels_seen[module_name]:
            _module_labels_seen[module_name].add(layer_label)
            mbd["module_layers"][module_name].append(layer_label)
        if layer_label not in _module_pass_labels_seen[module_pass_nice_label]:
            _module_pass_labels_seen[module_pass_nice_label].add(layer_label)
            mbd["module_pass_layers"][module_pass_nice_label].append(layer_label)
        if (module_index == 0) and (module_pass_nice_label not in _top_level_module_passes_seen):
            _top_level_module_passes_seen.add(module_pass_nice_label)
            mbd["top_level_module_passes"].append(module_pass_nice_label)
        else:
            if (containing_module_pass_label is not None) and (
                module_pass_nice_label
                not in _module_pass_children_seen[containing_module_pass_label]
            ):
                _module_pass_children_seen[containing_module_pass_label].add(module_pass_nice_label)
                mbd["module_pass_children"][containing_module_pass_label].append(
                    module_pass_nice_label
                )
        containing_module_pass_label = module_pass_nice_label
        if mbd["module_num_passes"][module_name] < module_pass:
            mbd["module_num_passes"][module_name] = module_pass
        if module_name not in _module_addresses_seen:
            _module_addresses_seen.add(module_name)
            mbd["module_addresses"].append(module_name)
        if module_pass_nice_label not in _module_passes_seen:
            _module_passes_seen.add(module_pass_nice_label)
            mbd["module_passes"].append(module_pass_nice_label)
    layer_entry.module_nesting_depth = len(layer_entry.containing_modules_origin_nested)


def _remove_unwanted_entries_and_log_remaining(self) -> None:
    """Removes entries from ModelLog that we don't want in the final saved output,
    and logs information about the remaining entries.
    """
    layers_to_remove = []
    # Quick loop to count how many tensors are saved:
    for layer_entry in self:
        if layer_entry.has_saved_activations:
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
    for raw_tensor_label in self._raw_layer_labels_list:
        layer_entry = self._raw_layer_dict[raw_tensor_label]
        # Determine valid lookup keys and relate them to the tensor's realtime operation number:
        if getattr(layer_entry, "has_saved_activations", False) or self.keep_unsaved_layers:
            # Add the lookup keys for the layer, to itself and to ModelLog:
            _add_lookup_keys_for_layer_entry(self, layer_entry, i, num_logged_tensors)

            # Log all information:
            self.layer_list.append(layer_entry)
            self.layer_dict_main_keys[layer_entry.layer_label] = layer_entry
            self.layer_labels.append(layer_entry.layer_label)
            self.layer_labels_no_pass.append(layer_entry.layer_label_no_pass)
            self.layer_labels_w_pass.append(layer_entry.layer_label_w_pass)
            self.layer_num_passes[layer_entry.layer_label] = layer_entry.layer_passes_total
            if layer_entry.has_saved_activations:
                self.tensor_fsize_saved += layer_entry.tensor_fsize
            _trim_and_reorder_layer_entry_fields(layer_entry)  # Final reformatting of fields
            i += 1
        else:
            layers_to_remove.append(layer_entry)
            self.unlogged_layers.append(layer_entry.layer_label)
            self._unsaved_layers_lookup_keys.update(layer_entry.lookup_keys)

    # Remove unused entries.
    for layer_entry in layers_to_remove:
        self._remove_log_entry(layer_entry, remove_references=False)

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


def _add_lookup_keys_for_layer_entry(
    self, layer_entry: LayerPassLog, tensor_index: int, num_tensors_to_keep: int
) -> None:
    """Adds the user-facing lookup keys for a LayerPassLog, both to itself
    and to the ModelLog top-level record.

    Args:
        layer_entry: LayerPassLog to get the lookup keys for.
        tensor_index: Zero-based position of this tensor in the final ordered layer list.
        num_tensors_to_keep: Total number of tensors that will be kept, used to compute
            negative-index lookup keys.
    """
    # The "default" keys: including the pass if multiple passes, excluding if one pass.
    lookup_keys_for_tensor = [
        layer_entry.layer_label,
        layer_entry.layer_label_short,
        tensor_index,
        tensor_index - num_tensors_to_keep,
    ]

    # If just one pass, also allow indexing by pass label.
    if layer_entry.layer_passes_total == 1:
        lookup_keys_for_tensor.extend(
            [layer_entry.layer_label_w_pass, layer_entry.layer_label_w_pass_short]
        )

    # Relabel the module passes if the first pass:
    if self.logging_mode == "exhaustive":
        layer_entry.module_passes_exited = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in layer_entry.module_passes_exited
        ]
        layer_entry.module_passes_entered = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in layer_entry.module_passes_entered
        ]
        if layer_entry.containing_module_origin is not None:
            layer_entry.containing_module_origin = ":".join(
                [str(i) for i in layer_entry.containing_module_origin]
            )
        layer_entry.containing_modules_origin_nested = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in layer_entry.containing_modules_origin_nested
        ]
        if (layer_entry.containing_module_origin is None) and len(
            layer_entry.containing_modules_origin_nested
        ) > 0:
            layer_entry.containing_module_origin = layer_entry.containing_modules_origin_nested[-1]

    # Allow indexing by modules exited as well:
    for module_pass in layer_entry.module_passes_exited:
        module_name, _ = module_pass.split(":")
        lookup_keys_for_tensor.append(f"{module_pass}")
        if self._module_build_data["module_num_passes"][module_name] == 1:
            lookup_keys_for_tensor.append(f"{module_name}")

    # Allow using buffer/input/output address as key, too:
    if layer_entry.is_buffer_layer:
        if self.buffer_num_passes[layer_entry.buffer_address] == 1:
            lookup_keys_for_tensor.append(layer_entry.buffer_address)
        lookup_keys_for_tensor.append(f"{layer_entry.buffer_address}:{layer_entry.buffer_pass}")
    elif layer_entry.is_input_layer or layer_entry.is_output_layer:
        lookup_keys_for_tensor.append(layer_entry.input_output_address)

    lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

    # Log in both the tensor and in the ModelLog object.
    layer_entry.lookup_keys = lookup_keys_for_tensor
    for lookup_key in lookup_keys_for_tensor:
        if lookup_key not in self._lookup_keys_to_layer_num_dict:
            self._lookup_keys_to_layer_num_dict[lookup_key] = layer_entry.realtime_tensor_num
            self.layer_dict_all_keys[lookup_key] = layer_entry
        self._layer_num_to_lookup_keys_dict[layer_entry.realtime_tensor_num].append(lookup_key)


def _trim_and_reorder_layer_entry_fields(layer_entry: LayerPassLog) -> None:
    """
    Sorts the fields in LayerPassLog into their desired order, and trims any
    fields that aren't useful after the pass.
    """
    old_dict = layer_entry.__dict__
    new_dir_dict = OrderedDict()
    for field in LAYER_PASS_LOG_FIELD_ORDER:
        if field in old_dict:
            new_dir_dict[field] = old_dict[field]
    for field, value in old_dict.items():
        if field not in new_dir_dict:
            new_dir_dict[field] = value
    layer_entry.__dict__ = new_dir_dict


def _rename_model_history_layer_names(self) -> None:
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

    mla = self._module_build_data["module_layer_argnames"]
    for module_pass, arglist in mla.items():
        inds_to_remove = set()
        for a, arg in enumerate(arglist):
            raw_name = mla[module_pass][a][0]
            if raw_name not in self._raw_to_final_layer_labels:
                inds_to_remove.add(a)
                continue
            new_name = self._raw_to_final_layer_labels[raw_name]
            argname = mla[module_pass][a][1]
            mla[module_pass][a] = (new_name, argname)
        mla[module_pass] = [
            mla[module_pass][i] for i in range(len(arglist)) if i not in inds_to_remove
        ]


def _trim_and_reorder_model_history_fields(self) -> None:
    """
    Sorts the fields in ModelLog into their desired order, and trims any
    fields that aren't useful after the pass.
    """
    new_dir_dict = OrderedDict()
    for field in MODEL_LOG_FIELD_ORDER:
        new_dir_dict[field] = getattr(self, field)
    for field, value in self.__dict__.items():
        if field.startswith("_") and field not in new_dir_dict:
            new_dir_dict[field] = value
    self.__dict__ = new_dir_dict
