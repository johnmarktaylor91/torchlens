"""Steps 8-11: Label mapping, final info logging, renaming, cleanup, and lookup keys.

Step 8 (_map_raw_labels_to_final_labels): Assigns human-readable labels to each
    tensor. Label format: ``{layer_type}_{type_num}_{total_num}:{call_index}`` for
    regular layers, or ``{layer_type}_{type_num}:{call_index}`` for input/output/buffer.
    The ``:call_index`` suffix is omitted when num_calls == 1. For multi-pass
    layers (pass > 1), layer_type and type_index are INHERITED from the first
    pass to guarantee label consistency within recurrent_ops groups.

Step 9 (_log_final_info_for_layers): Logs operation numbers, module hierarchy,
    param/size tallies, and structural flags. Populates _module_build_data dicts
    that Step 11 and Step 16 depend on. MUST run before Step 11 because
    _add_lookup_keys_for_layer_entry needs module_num_calls data.

Step 10 (_rename_model_history_layer_names + _trim_and_reorder_model_history_fields):
    Renames all raw labels (e.g., "cos_3_raw") to final labels in both Trace-level
    fields and Op fields, then reorders Trace fields into canonical order.

Step 11 (_remove_unwanted_entries_and_log_remaining): Removes unsaved layers (unless
    keep_unsaved_layers=True), builds lookup key mappings (integer index, label,
    module path, buffer/input/output address), and logs remaining layer metadata.
"""

import weakref
from collections import defaultdict
from dataclasses import fields, is_dataclass, replace
from typing import Any, Dict, List, TYPE_CHECKING

from ..constants import MODEL_LOG_FIELD_ORDER, LAYER_PASS_LOG_FIELD_ORDER
from ..intervention.types import ParentRef
from ..utils.display import human_readable_size
from ..data_classes.op_log import Op

if TYPE_CHECKING:
    from ..data_classes.model_log import Trace


def _map_raw_labels_to_final_labels(self: "Trace") -> None:
    """Step 8: Build the raw-to-final label mapping for all tensors.

    Iterates through all tensors in order and assigns each a human-readable label.
    Label format conventions:
    - Regular layers: ``{layer_type}_{type_num}_{total_num}:{call_index}``
      e.g., ``conv2d_3_12:2`` (3rd conv2d overall, 12th layer total, pass 2)
    - Input/output/buffer: ``{layer_type}_{type_num}:{call_index}``
      e.g., ``input_1:1`` (total_num is always 0 for these types)
    - When num_calls == 1, the ``:call_index`` suffix is omitted in
      the default ``layer_label`` (but ``layer_label_w_pass`` always includes it).

    For multi-pass layers (call_index > 1), ``layer_type`` AND ``type_index``
    are INHERITED from the first pass's tensor to guarantee that all members of
    a recurrent_ops group share the same ``layer_label_no_pass``. Using
    the entry's own layer_type would cause label mismatches when ops have
    different layer types (e.g., SSD300 train with getitem ops {1,3} gap).

    Stores the bidirectional mapping in ``self._raw_to_final_layer_labels`` and
    ``self._final_to_raw_layer_labels``.
    """
    raw_to_final_layer_labels = {}
    final_to_raw_layer_labels = {}
    layer_type_counter: defaultdict[str, int] = defaultdict(lambda: 1)
    layer_total_counter = 1  # Sequential counter for non-input/buffer/output layers.
    for tensor_log_entry in self:
        layer_type = tensor_log_entry.layer_type
        call_index = tensor_log_entry.call_index
        if call_index == 1:
            # First pass: assign new type_num and total_num.
            type_index = layer_type_counter[layer_type]
            layer_type_counter[layer_type] += 1
            if layer_type in ["input", "buffer"]:
                trace_index = 0  # Input/buffer don't get a total order number.
            else:
                trace_index = layer_total_counter
                layer_total_counter += 1

        else:
            # Pass > 1: INHERIT layer_type and numbers from the first pass.
            # This ensures all ops of the same layer share layer_label_no_pass.
            first_pass_tensor = self[tensor_log_entry.recurrent_ops[0]]
            layer_type = first_pass_tensor.layer_type
            type_index = first_pass_tensor.type_index
            if layer_type in ["input", "buffer"]:
                trace_index = 0
            else:
                trace_index = first_pass_tensor.trace_index
        tensor_log_entry.type_index = type_index
        tensor_log_entry.trace_index = trace_index

        if layer_type not in ["input", "output", "buffer"]:
            tensor_log_entry.layer_label_w_pass = (
                f"{layer_type}_{type_index}_{trace_index}:{call_index}"
            )
            tensor_log_entry.layer_label_no_pass = f"{layer_type}_{type_index}_{trace_index}"
        else:
            tensor_log_entry.layer_label_w_pass = f"{layer_type}_{type_index}:{call_index}"
            tensor_log_entry.layer_label_no_pass = f"{layer_type}_{type_index}"

        tensor_log_entry.layer_label_w_pass_short = f"{layer_type}_{type_index}:{call_index}"
        tensor_log_entry.layer_label_no_pass_short = f"{layer_type}_{type_index}"
        if tensor_log_entry.num_calls == 1:
            tensor_log_entry.layer_label = tensor_log_entry.layer_label_no_pass
            tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_no_pass_short
        else:
            tensor_log_entry.layer_label = tensor_log_entry.layer_label_w_pass
            tensor_log_entry.layer_label_short = tensor_log_entry.layer_label_w_pass_short
        raw_to_final_layer_labels[tensor_log_entry._label_raw] = tensor_log_entry.layer_label
        final_to_raw_layer_labels[tensor_log_entry.layer_label] = tensor_log_entry._label_raw
    self._raw_to_final_layer_labels = raw_to_final_layer_labels
    self._final_to_raw_layer_labels = final_to_raw_layer_labels


def _log_final_info_for_layers(self: "Trace") -> None:
    """Step 9: Log final metadata for all layers and build module hierarchy.

    Iterates through all layers (before unsaved ones are discarded in Step 11)
    and computes:
    - Operation numbers (sequential, excluding input/buffer/output).
    - Replaces raw labels with final labels in each Op's fields.
    - Module hierarchy information (_module_build_data dicts).
    - Cumulative tallies: tensor sizes, param counts, elapsed time.
    - Structural flags: branching, recurrence, conditional branching.

    MUST run before Step 11 because ``_add_lookup_keys_for_layer_entry``
    (called in Step 11) needs module_num_calls data populated here.

    Uses shadow sets for O(1) membership checks to avoid expensive linear
    ``in`` checks on lists for large models.
    """
    unique_layers_seen = set()  # to avoid double-counting params of recurrent layers
    compute_index = 1
    mbd = self._module_build_data

    # Shadow sets for O(1) membership checks in _log_module_hierarchy_info_for_layer.
    # Lists are kept as primary storage (insertion order matters for downstream consumers),
    # but linear `in` checks on lists are expensive for large models.
    _shadow_sets: dict[str, Any] = {
        "module_layers": defaultdict(set),
        "module_pass_layers": defaultdict(set),
        "top_level_module_ops": set(),
        "module_pass_children": defaultdict(set),
        "addresses": set(),
        "module_ops": set(),
    }

    for t, layer_entry in enumerate(self):
        if layer_entry.layer_type in ["input", "buffer"]:
            layer_entry.compute_index = 0
        elif layer_entry.layer_type == "output":
            layer_entry.compute_index = None  # fix later
        else:
            layer_entry.compute_index = compute_index
            self.num_ops += 1
            compute_index += 1

        # Replace any layer names with their final names:
        _replace_layer_names_for_layer_entry(self, layer_entry)

        # Log the module hierarchy information:
        _log_module_hierarchy_info_for_layer(self, layer_entry, _shadow_sets)
        if layer_entry.atomic_module_call is not None:
            submodule_pass_nice_name = ":".join([str(i) for i in layer_entry.atomic_module_call])
            layer_entry.atomic_module_call = submodule_pass_nice_name

        # Tally the tensor sizes:
        self.total_out_memory += layer_entry.memory

        # Tally the parameter sizes:
        if layer_entry.layer_label_no_pass not in unique_layers_seen:  # only count params once
            if layer_entry.uses_params:
                self.num_layers_with_params += 1
            self.num_params += layer_entry.num_params
            self.num_params_trainable += layer_entry.num_params_trainable
            self.num_params_frozen += layer_entry.num_params_frozen
            self.num_param_tensors += layer_entry.num_param_tensors
            self.param_memory += layer_entry.param_memory
            # Tally for modules, too.
            for module_name, _ in layer_entry.modules:
                mbd["module_nparams"][module_name] += layer_entry.num_params
                mbd["module_nparams_trainable"][module_name] += layer_entry.num_params_trainable
                mbd["module_nparams_frozen"][module_name] += layer_entry.num_params_frozen

        unique_layers_seen.add(layer_entry.layer_label_no_pass)

        # Tally elapsed time:

        self.func_calls_duration += layer_entry.func_duration

    _compute_fx_qualpaths(self)
    _finalize_output_compute_indexs(self)
    _build_module_hierarchy_dicts(self)


def _module_address_for_fx(module_call_label: Any) -> str:
    """Return the module address portion of a module call label.

    Parameters
    ----------
    module_call_label
        Module call encoded as ``(address, call_index)`` or as the
        postprocessed ``"address:call_index"`` string.

    Returns
    -------
    str
        The structural module address without the call index.
    """

    if isinstance(module_call_label, str):
        module_address, _module_pass = module_call_label.rsplit(":", 1)
        return module_address
    module_address, _module_pass = module_call_label
    return str(module_address)


def _compute_fx_qualpaths(self: "Trace") -> None:
    """Compute FX-style structural qualpath metadata for every Op.

    ``modules`` stores cumulative module addresses, so the most nested module
    address already contains the full dotted path.

    Parameters
    ----------
    self
        Trace being postprocessed.
    """

    fx_call_counts: dict[str, int] = {}
    for layer_entry in self:
        module_calls = getattr(layer_entry, "modules", None) or []
        op_type_segment = str(layer_entry.func_name or layer_entry.layer_type or "").lower()
        if not module_calls or not op_type_segment:
            layer_entry.fx_qualpath = None
            layer_entry.fx_call_index = 0
            continue

        module_address = _module_address_for_fx(module_calls[-1])
        fx_qualpath = f"{module_address}.{op_type_segment}"
        layer_entry.fx_qualpath = fx_qualpath
        layer_entry.fx_call_index = fx_call_counts.get(fx_qualpath, 0)
        fx_call_counts[fx_qualpath] = layer_entry.fx_call_index + 1


def _finalize_output_compute_indexs(self: "Trace") -> None:
    """Assign compute_index to output layers (deferred until total is known)."""
    for layer in self.output_layers:
        self[layer].compute_index = self.num_ops


def _build_module_hierarchy_dicts(self: "Trace") -> None:
    """Derive top_level_modules and module_children from their pass-level counterparts."""
    mbd = self._module_build_data
    for module in mbd["top_level_module_ops"]:
        module_no_pass = module.split(":")[0]
        if module_no_pass not in mbd["top_level_modules"]:
            mbd["top_level_modules"].append(module_no_pass)

    for module_parent, module_children in mbd["module_pass_children"].items():
        module_parent_nopass = module_parent.split(":")[0]
        for module_child in module_children:
            module_child_nopass = module_child.split(":")[0]
            if module_child_nopass not in mbd["module_children"][module_parent_nopass]:
                mbd["module_children"][module_parent_nopass].append(module_child_nopass)


# Fields in Op that contain raw labels needing rename.
# These may be lists or sets — the rename logic handles both via type(orig).
_LIST_FIELDS_TO_RENAME = [
    "parents",
    "root_ancestors",
    "children",
    "input_ancestors",
    "output_descendants",
    "internal_source_parents",
    "internal_source_ancestors",
    "conditional_entry_children",
    "conditional_then_children",
    "conditional_else_children",
    "equivalent_ops",
    "recurrent_ops",
]


def _rename_elif_children(
    conditional_elif_children: Dict[int, List[str]],
    mapping: Dict[str, str],
) -> Dict[int, List[str]]:
    """Rename labels inside ``conditional_elif_children``.

    Args:
        conditional_elif_children: Mapping from elif index to child labels.
        mapping: Raw-to-final layer-label mapping.

    Returns:
        A new dict with all child labels rewritten.
    """
    return {
        elif_ix: [mapping[layer_label] for layer_label in child_labels]
        for elif_ix, child_labels in conditional_elif_children.items()
    }


def _rename_children_by_cond(
    conditional_arm_children: Dict[int, Dict[str, List[str]]],
    mapping: Dict[str, str],
) -> Dict[int, Dict[str, List[str]]]:
    """Rename labels inside ``conditional_arm_children``.

    Args:
        conditional_arm_children: ``cond_id -> branch_kind -> child labels``.
        mapping: Raw-to-final layer-label mapping.

    Returns:
        A new dict with every nested child label rewritten.
    """
    return {
        cond_id: {
            branch_kind: [mapping[layer_label] for layer_label in child_labels]
            for branch_kind, child_labels in branch_children.items()
        }
        for cond_id, branch_children in conditional_arm_children.items()
    }


def _replace_layer_names_for_layer_entry(self: "Trace", layer_entry: Op) -> None:
    """Replace all raw labels in a Op's fields with final labels.

    Handles three categories of fields:
    1. List/set fields (parents, children, etc.): creates NEW objects
       via type(orig)([comprehension]) — no shared-set corruption possible.
    2. parent_arg_positions dict: renames values in-place.
    3. output_versions_per_child dict: renames keys.

    Args:
        layer_entry: Op to rename labels for.
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
    arg_locs = d.get("parent_arg_positions")
    if arg_locs:
        for arg_type in ("args", "kwargs"):
            sub = arg_locs[arg_type]
            for key, value in sub.items():
                sub[key] = mapping[value]

    # Fix the field names for different children tensor versions:
    ctv = d.get("output_versions_per_child")
    if ctv:
        d["output_versions_per_child"] = {
            mapping[child_label]: tensor_version for child_label, tensor_version in ctv.items()
        }

    elif_children = d.get("conditional_elif_children")
    if elif_children:
        d["conditional_elif_children"] = _rename_elif_children(elif_children, mapping)

    children_by_cond = d.get("conditional_arm_children")
    if children_by_cond:
        d["conditional_arm_children"] = _rename_children_by_cond(children_by_cond, mapping)

    if d.get("edge_uses"):
        d["edge_uses"] = [_rename_label_dataclass(record, mapping) for record in d["edge_uses"]]

    if d.get("args_template") is not None:
        d["args_template"] = _rename_template_parent_refs(d["args_template"], mapping)
    if d.get("kwargs_template") is not None:
        d["kwargs_template"] = _rename_template_parent_refs(d["kwargs_template"], mapping)
    if d.get("interventions"):
        d["interventions"] = [
            _rename_label_dataclass(record, mapping) for record in d["interventions"]
        ]


def _rename_template_parent_refs(value: Any, mapping: Dict[str, str]) -> Any:
    """Rename ``ParentRef.parent_label`` leaves in replay templates.

    Args:
        value: Template or nested template component.
        mapping: Raw-to-final layer-label mapping.

    Returns:
        Template value with parent references rewritten where possible.
    """

    if isinstance(value, ParentRef):
        return replace(value, parent_label=mapping.get(value.parent_label, value.parent_label))
    if isinstance(value, tuple):
        return tuple(_rename_template_parent_refs(item, mapping) for item in value)
    if isinstance(value, list):
        return [_rename_template_parent_refs(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _rename_template_parent_refs(item, mapping) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _rename_template_parent_refs(getattr(value, field.name), mapping)
            for field in fields(value)
            if hasattr(value, field.name)
        }
        return replace(value, **updates)
    return value


def _rename_label_dataclass(value: Any, mapping: Dict[str, str]) -> Any:
    """Rename known label fields on frozen intervention dataclasses.

    Args:
        value: Dataclass-like record or nested container.
        mapping: Raw-to-final layer-label mapping.

    Returns:
        Record with label-bearing string fields rewritten.
    """

    label_fields = {
        "parent_label",
        "child_label",
        "target_label",
        "call_label",
        "site_label",
        "layer_label",
    }
    if isinstance(value, tuple):
        return tuple(_rename_label_dataclass(item, mapping) for item in value)
    if isinstance(value, list):
        return [_rename_label_dataclass(item, mapping) for item in value]
    if isinstance(value, dict):
        return {
            _rename_label_dataclass(key, mapping): _rename_label_dataclass(item, mapping)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        updates = {}
        for field in fields(value):
            if not hasattr(value, field.name):
                continue
            field_value = getattr(value, field.name)
            if field.name in label_fields and isinstance(field_value, str):
                updates[field.name] = mapping.get(field_value, field_value)
            else:
                updates[field.name] = _rename_label_dataclass(field_value, mapping)
        return replace(value, **updates)
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _log_module_hierarchy_info_for_layer(
    self: "Trace", layer_entry: Op, _shadow_sets: dict[str, Any]
) -> None:
    """Populate module hierarchy data for a single layer in _module_build_data.

    For each module in the layer's modules, records:
    - Module tensor counts (per-module and per-pass).
    - Module-to-layer mappings.
    - Top-level module ops and parent-child relationships.
    - Module addresses and pass labels.

    Uses shadow sets for O(1) dedup checks (the primary lists maintain insertion
    order for downstream consumers, but linear ``in`` checks on lists are O(n)).

    Args:
        layer_entry: Log entry to process.
        _shadow_sets: Dict of shadow sets mirroring _module_build_data lists.
    """
    _module_labels_seen = _shadow_sets["module_layers"]
    _module_call_labels_seen = _shadow_sets["module_pass_layers"]
    _top_level_module_ops_seen = _shadow_sets["top_level_module_ops"]
    _module_pass_children_seen = _shadow_sets["module_pass_children"]
    _addresses_seen = _shadow_sets["addresses"]
    _module_ops_seen = _shadow_sets["module_ops"]
    mbd = self._module_build_data

    parent_call_label = None
    layer_label = layer_entry.layer_label
    for module_index, raw_module_call_label in enumerate(layer_entry.modules):
        if isinstance(raw_module_call_label, str):
            module_name, module_pass = raw_module_call_label.rsplit(":", 1)
            module_pass = int(module_pass)  # type: ignore[assignment]
        else:
            module_name, module_pass = raw_module_call_label
        module_pass_nice_label = f"{module_name}:{module_pass}"
        mbd["module_num_tensors"][module_name] += 1
        mbd["module_call_index_tensors"][module_pass_nice_label] += 1
        if layer_label not in _module_labels_seen[module_name]:
            _module_labels_seen[module_name].add(layer_label)
            mbd["module_layers"][module_name].append(layer_label)
        if layer_label not in _module_call_labels_seen[module_pass_nice_label]:
            _module_call_labels_seen[module_pass_nice_label].add(layer_label)
            mbd["module_pass_layers"][module_pass_nice_label].append(layer_label)
        if (module_index == 0) and (module_pass_nice_label not in _top_level_module_ops_seen):
            _top_level_module_ops_seen.add(module_pass_nice_label)
            mbd["top_level_module_ops"].append(module_pass_nice_label)
        else:
            if (parent_call_label is not None) and (
                module_pass_nice_label not in _module_pass_children_seen[parent_call_label]
            ):
                _module_pass_children_seen[parent_call_label].add(module_pass_nice_label)
                mbd["module_pass_children"][parent_call_label].append(module_pass_nice_label)
        parent_call_label = module_pass_nice_label
        if mbd["module_num_calls"][module_name] < module_pass:
            mbd["module_num_calls"][module_name] = module_pass
        if module_name not in _addresses_seen:
            _addresses_seen.add(module_name)
            mbd["addresses"].append(module_name)
        if module_pass_nice_label not in _module_ops_seen:
            _module_ops_seen.add(module_pass_nice_label)
            mbd["module_ops"].append(module_pass_nice_label)


def _remove_unwanted_entries_and_log_remaining(self: "Trace") -> None:
    """Step 11: Remove unsaved layers, build lookup keys, finalize layer lists.

    Unless ``keep_unsaved_layers=True``, removes Op entries that don't
    have saved outs. For each retained entry:
    - Builds lookup keys (integer index, label, module path, address).
    - Adds to layer_list, layer_dict_main_keys, and label lists.
    - Reorders fields via _trim_and_reorder_layer_entry_fields.

    Sets _layers_logged and _layers_saved flags for downstream
    consumers (e.g., visualization guards for keep_unsaved_layers=False).
    """
    layers_to_remove = []
    # Quick loop to count how many tensors are saved:
    for layer_entry in self:
        if getattr(layer_entry, "is_orphan", False):
            continue
        if layer_entry.has_saved_outs:
            self.num_saved_ops += 1

    retained_call_group_labels = _labels_in_replay_ready_call_groups_to_retain(self)

    if self.keep_unsaved_layers:
        num_logged_tensors = sum(
            1 for layer_entry in self if not getattr(layer_entry, "is_orphan", False)
        )
    else:
        num_logged_tensors = sum(
            1
            for layer_entry in self
            if not getattr(layer_entry, "is_orphan", False)
            and (
                layer_entry.has_saved_outs or layer_entry.layer_label in retained_call_group_labels
            )
        )

    self.layer_list = []
    self.layer_dict_main_keys = {}
    self.layer_labels = []
    self.layer_labels = []
    self.op_labels = []
    self.layer_num_calls = {}

    i = 0
    for raw_tensor_label in self._raw_layer_labels_list:
        layer_entry = self._raw_layer_dict[raw_tensor_label]
        if getattr(layer_entry, "is_orphan", False):
            continue
        # Determine valid lookup keys and relate them to the tensor's realtime operation number:
        should_keep_for_replay = layer_entry.layer_label in retained_call_group_labels
        if (
            getattr(layer_entry, "has_saved_outs", False)
            or self.keep_unsaved_layers
            or should_keep_for_replay
        ):
            # Add the lookup keys for the layer, to itself and to Trace:
            _add_lookup_keys_for_layer_entry(self, layer_entry, i, num_logged_tensors)

            # Log all information:
            self.layer_list.append(layer_entry)
            self.layer_dict_main_keys[layer_entry.layer_label] = layer_entry
            self.layer_labels.append(layer_entry.layer_label)
            self.layer_labels.append(layer_entry.layer_label_no_pass)
            self.op_labels.append(layer_entry.layer_label_w_pass)
            self.layer_num_calls[layer_entry.layer_label_no_pass] = layer_entry.num_calls
            if layer_entry.has_saved_outs:
                self.saved_out_memory += layer_entry.memory
            i += 1
        else:
            layers_to_remove.append(layer_entry)
            self.unlogged_ops.append(layer_entry.layer_label)
            self._unsaved_layers_lookup_keys.update(layer_entry.lookup_keys)

    # Remove unused entries and scrub any label-bearing references they left behind.
    if layers_to_remove:
        self._batch_remove_log_entries(layers_to_remove, remove_references=True)

    num_non_orphan_ops = sum(
        1 for layer_entry in self if not getattr(layer_entry, "is_orphan", False)
    )
    if (self.num_saved_ops == num_non_orphan_ops) or self.keep_unsaved_layers:
        self._layers_logged = True
    else:
        self._layers_logged = False

    if self.num_saved_ops == len(self.layer_list):
        self._layers_saved = True
    else:
        self._layers_saved = False


def _labels_in_replay_ready_call_groups_to_retain(self: "Trace") -> set[str]:
    """Return labels in same-call groups that must remain replay-addressable.

    Args:
        self: Trace being postprocessed.

    Returns:
        Labels to keep atomically even when their out was not saved.
    """

    if not getattr(self, "intervention_ready", False):
        return set()

    call_groups: dict[int, list[Op]] = defaultdict(list)
    for layer_entry in self:
        if getattr(layer_entry, "is_orphan", False):
            continue
        func_call_id = getattr(layer_entry, "func_call_id", None)
        if func_call_id is not None:
            call_groups[func_call_id].append(layer_entry)

    labels_to_keep = {
        layer_entry.layer_label
        for layer_entry in self
        if layer_entry.has_saved_outs and not getattr(layer_entry, "is_orphan", False)
    }
    label_to_entry = {layer_entry.layer_label: layer_entry for layer_entry in self}
    changed = True
    while changed:
        changed = False
        for layer_label in list(labels_to_keep):
            layer_entry = label_to_entry.get(layer_label)
            if layer_entry is None:
                continue
            dependency_labels = _replay_dependency_labels(layer_entry)
            for dependency_label in dependency_labels:
                if dependency_label in label_to_entry and dependency_label not in labels_to_keep:
                    labels_to_keep.add(dependency_label)
                    changed = True
            func_call_id = getattr(layer_entry, "func_call_id", None)
            if func_call_id is None:
                continue
            for sibling in call_groups[func_call_id]:
                if sibling.layer_label not in labels_to_keep:
                    labels_to_keep.add(sibling.layer_label)
                    changed = True
    return labels_to_keep


def _replay_dependency_labels(layer_entry: Op) -> set[str]:
    """Return final labels this entry's replay metadata depends on.

    Args:
        layer_entry: Layer entry to inspect.

    Returns:
        Parent labels referenced by templates, edge provenance, or graph edges.
    """

    dependency_labels = set(layer_entry.parents)
    for edge in getattr(layer_entry, "edge_uses", []):
        dependency_labels.add(edge.parent_label)
    for template in (layer_entry.args_template, layer_entry.kwargs_template):
        for parent_ref in _collect_parent_refs(template):
            dependency_labels.add(parent_ref.parent_label)
    return dependency_labels


def _collect_parent_refs(value: Any) -> list[ParentRef]:
    """Collect ``ParentRef`` leaves from a nested replay template.

    Args:
        value: Template or nested component.

    Returns:
        Parent references found under ``value``.
    """

    if isinstance(value, ParentRef):
        return [value]
    if isinstance(value, tuple):
        refs: list[ParentRef] = []
        for item in value:
            refs.extend(_collect_parent_refs(item))
        return refs
    if isinstance(value, list):
        refs = []
        for item in value:
            refs.extend(_collect_parent_refs(item))
        return refs
    if isinstance(value, dict):
        refs = []
        for item in value.values():
            refs.extend(_collect_parent_refs(item))
        return refs
    if is_dataclass(value) and not isinstance(value, type):
        refs = []
        for field in fields(value):
            if hasattr(value, field.name):
                refs.extend(_collect_parent_refs(getattr(value, field.name)))
        return refs
    return []


def _add_lookup_keys_for_layer_entry(
    self: "Trace", layer_entry: Op, tensor_index: int, num_tensors_to_keep: int
) -> None:
    """Build user-facing lookup keys for a Op and register them.

    Keys allow users to access layers via Trace[key]. Multiple key types:
    - String labels: layer_label, layer_label_short, with/without pass suffix.
    - Integer indices: positive (0-based) and negative (Python-style).
    - Module paths: module_pass labels for modules exited by this layer.
    - Addresses: buffer_address, input/output address.

    Also reformats output_of_module_calls/entered and modules
    from (name, pass) tuples to "name:pass" strings (exhaustive mode only).

    Args:
        layer_entry: Op to build lookup keys for.
        tensor_index: Zero-based position in the final ordered layer list.
        num_tensors_to_keep: Total number of retained tensors (for negative indices).
    """
    # The "default" keys: including the pass if multiple ops, excluding if one pass.
    lookup_keys_for_tensor = [
        layer_entry.layer_label,
        layer_entry.layer_label_short,
        tensor_index,
        tensor_index - num_tensors_to_keep,
    ]

    # If just one pass, also allow indexing by pass label.
    if layer_entry.num_calls == 1:
        lookup_keys_for_tensor.extend(
            [layer_entry.layer_label_w_pass, layer_entry.layer_label_w_pass_short]
        )

    # Relabel the module ops if the first pass:
    if self.capture_mode == "exhaustive":
        layer_entry.output_of_module_calls = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in layer_entry.output_of_module_calls
        ]
        layer_entry.module_ops_entered = [
            f"{module_name}:{module_pass}"
            for module_name, module_pass in layer_entry.module_ops_entered
        ]
        if layer_entry.module is not None:
            layer_entry.module = ":".join([str(i) for i in layer_entry.module])
        layer_entry.modules = [
            f"{module_name}:{module_pass}" for module_name, module_pass in layer_entry.modules
        ]
        if (layer_entry.module is None) and len(layer_entry.modules) > 0:
            layer_entry.module = layer_entry.modules[-1]

    # Allow indexing by modules exited as well:
    for module_pass in layer_entry.output_of_module_calls:
        module_name, _ = module_pass.split(":")
        lookup_keys_for_tensor.append(f"{module_pass}")
        if self._module_build_data["module_num_calls"][module_name] == 1:
            lookup_keys_for_tensor.append(f"{module_name}")

    # Allow using buffer/input/output address as key, too:
    if layer_entry.is_buffer:
        if self.buffer_num_calls[layer_entry.buffer_address] == 1:
            lookup_keys_for_tensor.append(layer_entry.buffer_address)
        lookup_keys_for_tensor.append(f"{layer_entry.buffer_address}:{layer_entry.buffer_pass}")
    elif layer_entry.is_input or layer_entry.is_output:
        lookup_keys_for_tensor.append(layer_entry.io_role)

    lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

    # Log in both the tensor and in the Trace object.
    layer_entry.lookup_keys = lookup_keys_for_tensor
    for lookup_key in lookup_keys_for_tensor:
        if lookup_key not in self._lookup_keys_to_layer_num_dict:
            self._lookup_keys_to_layer_num_dict[lookup_key] = layer_entry.capture_index
            self.layer_dict_all_keys[lookup_key] = layer_entry
        self._layer_num_to_lookup_keys_dict[layer_entry.capture_index].append(lookup_key)


def _trim_and_reorder_layer_entry_fields(layer_entry: Op) -> None:
    """Reorder Op fields into canonical display order.

    PRESERVES all fields — this function only reorders, it does NOT strip
    any data. Fields listed in LAYER_PASS_LOG_FIELD_ORDER come first (in that
    order), followed by any remaining non-callable fields not in the order list.
    Callable attributes (custom_methods) are excluded from the reordered dict.
    """
    old_dict = layer_entry.__dict__
    new_dir_dict = {}
    # First: fields in canonical order.
    for field in LAYER_PASS_LOG_FIELD_ORDER:
        if field in old_dict:
            new_dir_dict[field] = old_dict[field]
    # Second: any remaining fields not in the order list (preserves all data).
    # The callable check skips bound custom_methods, but must not skip weakref.ref
    # objects (which are callable but are data, not custom_methods).
    for field, value in old_dict.items():
        if field not in new_dir_dict and (not callable(value) or isinstance(value, weakref.ref)):
            new_dir_dict[field] = value
    layer_entry.__dict__ = new_dir_dict


def _rename_model_history_layer_names(self: "Trace") -> None:
    """Step 10: Rename raw labels to final labels in all Trace-level fields.

    Updates list fields (input_layers, output_layers, etc.), dict fields
    (layers_with_params, equivalent_ops), conditional branch
    edges, and module layer argument names. Creates NEW container objects
    (no shared-set corruption) — see set() constructor in equivalent_ops
    rename.
    """
    list_fields_to_rename = [
        "input_layers",
        "output_layers",
        "buffer_layers",
        "internal_source_ops",
        "internal_sink_ops",
        "internally_terminated_bool_ops",
        "ops_with_saved_grads",
        "ops_with_saved_outs",
    ]
    for field in list_fields_to_rename:
        tensor_labels = getattr(self, field)
        setattr(
            self,
            field,
            [self._raw_to_final_layer_labels[tensor_label] for tensor_label in tensor_labels],
        )

    new_param_tensors = {}
    for key, values in self.layers_with_params.items():
        new_key = self[values[0]].layer_label
        new_param_tensors[new_key] = [
            self._raw_to_final_layer_labels[tensor_label] for tensor_label in values
        ]
    self.layers_with_params = new_param_tensors

    new_equiv_operations_tensors: dict[Any, set[str]] = {}
    for key, equiv_values in self.equivalent_ops.items():
        new_equiv_operations_tensors[key] = set(
            [self._raw_to_final_layer_labels[tensor_label] for tensor_label in equiv_values]
        )
    self.equivalent_ops = new_equiv_operations_tensors

    for t, (child, parent) in enumerate(self.conditional_branch_edges):
        new_child, new_parent = (
            self._raw_to_final_layer_labels[child],
            self._raw_to_final_layer_labels[parent],
        )
        self.conditional_branch_edges[t] = (new_child, new_parent)

    self.conditional_arm_entry_edges = {
        (cond_id, branch_kind): [
            (
                self._raw_to_final_layer_labels[parent],
                self._raw_to_final_layer_labels[child],
            )
            for parent, child in arm_edges
        ]
        for (cond_id, branch_kind), arm_edges in self.conditional_arm_entry_edges.items()
    }

    self.conditional_edge_call_indices = {
        (
            self._raw_to_final_layer_labels[parent],
            self._raw_to_final_layer_labels[child],
            cond_id,
            branch_kind,
        ): call_indexs
        for (
            parent,
            child,
            cond_id,
            branch_kind,
        ), call_indexs in self.conditional_edge_call_indices.items()
    }

    for conditional_event in self.conditional_records:
        conditional_event.bool_layers = [
            self._raw_to_final_layer_labels[layer_label]
            for layer_label in conditional_event.bool_layers
        ]

    self.ledger = _rename_label_dataclass(
        getattr(self, "ledger", []), self._raw_to_final_layer_labels
    )
    if getattr(self, "_intervention_spec", None) is not None:
        self._intervention_spec = _rename_label_dataclass(
            self._intervention_spec, self._raw_to_final_layer_labels
        )

    mla = self._module_build_data["module_layer_argnames"]
    for module_pass, arglist in mla.items():
        new_arglist = []
        for raw_name, argname in arglist:
            if raw_name in self._raw_to_final_layer_labels:
                new_arglist.append((self._raw_to_final_layer_labels[raw_name], argname))
        mla[module_pass] = new_arglist


def _trim_and_reorder_model_history_fields(self: "Trace") -> None:
    """Reorder Trace fields into canonical display order.

    Like ``_trim_and_reorder_layer_entry_fields``, this PRESERVES all fields.
    Public fields listed in MODEL_LOG_FIELD_ORDER come first, followed by any
    private fields (starting with ``_``) not already in the order list.
    """
    new_dir_dict = {}
    for field in MODEL_LOG_FIELD_ORDER:
        new_dir_dict[field] = getattr(self, field)
    # Preserve all remaining fields not in the canonical order (private/internal
    # fields AND runtime-config attributes like ``verbose``).
    for field, value in self.__dict__.items():
        if field not in new_dir_dict:
            new_dir_dict[field] = value
    self.__dict__ = new_dir_dict
