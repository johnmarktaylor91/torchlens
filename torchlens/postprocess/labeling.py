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

Step 10 (_rename_model_history_layer_names):
    Renames all raw labels (e.g., "cos_3_raw") to final labels in both Trace-level
    fields and Op fields.

Step 11 (_build_lookup_keys_and_finalize_retained_layers): Builds lookup key mappings (integer
    index, label, module path, buffer/input/output address), and logs retained layer metadata.
"""

from collections import defaultdict
from dataclasses import fields, is_dataclass, replace
from typing import Any, Dict, List, TYPE_CHECKING

from ..data_classes.op import Op
from ..intervention.types import ParentRef

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _map_raw_labels_to_final_labels(self: "Trace") -> None:
    """Step 8: Build the raw-to-final label mapping for all tensors.

    Iterates through all tensors in order and assigns each a human-readable label.
    Label format conventions:
    - Regular layers: ``{layer_type}_{type_num}_{total_num}:{call_index}``
      e.g., ``conv2d_3_12:2`` (3rd conv2d overall, 12th layer total, pass 2)
    - Input/output/buffer: ``{layer_type}_{type_num}:{call_index}``
      e.g., ``input_1:1`` (total_num is always 0 for these types)
    - When num_calls == 1, the ``:call_index`` suffix is omitted in
      the default ``layer_label`` (but ``label`` always includes it).

    For multi-pass layers (call_index > 1), ``layer_type`` AND ``type_index``
    are INHERITED from the first pass's tensor to guarantee that all members of
    a recurrent_ops group share the same ``layer_label``. Using
    the entry's own layer_type would cause label mismatches when ops have
    different layer types (e.g., SSD300 train with getitem ops {1,3} gap).

    Stores the bidirectional mapping in ``self._raw_to_final_layer_labels`` and
    ``self._final_to_raw_layer_labels``.
    """
    raw_to_final_layer_labels = {}
    raw_to_final_parent_layer_labels = {}
    raw_to_final_op_labels = {}
    final_to_raw_layer_labels = {}
    layer_type_counter: defaultdict[str, int] = defaultdict(lambda: 1)
    layer_total_counter = 1  # Sequential counter for non-input/buffer/output layers.
    for tensor_log_entry in self:
        layer_type = tensor_log_entry.layer_type
        pass_index = tensor_log_entry.pass_index
        if pass_index == 1:
            # First pass: assign new type_num and total_num.
            type_index = layer_type_counter[layer_type]
            layer_type_counter[layer_type] += 1
            if layer_type in ["input", "buffer"]:
                step_index = 0  # Input/buffer don't get a total order number.
            else:
                step_index = layer_total_counter
                layer_total_counter += 1

        else:
            # Pass > 1: INHERIT layer_type and numbers from the first pass.
            # This ensures all ops of the same layer share layer_label.
            first_pass_tensor = self[tensor_log_entry.recurrent_ops[0]]
            layer_type = first_pass_tensor.layer_type
            type_index = first_pass_tensor.type_index
            if layer_type in ["input", "buffer"]:
                step_index = 0
            else:
                step_index = first_pass_tensor.step_index
        tensor_log_entry.type_index = type_index
        tensor_log_entry.step_index = step_index

        if layer_type not in ["input", "output", "buffer"]:
            tensor_log_entry.label = f"{layer_type}_{type_index}_{step_index}:{pass_index}"
            tensor_log_entry.layer_label = f"{layer_type}_{type_index}_{step_index}"
        else:
            tensor_log_entry.label = f"{layer_type}_{type_index}:{pass_index}"
            tensor_log_entry.layer_label = f"{layer_type}_{type_index}"

        tensor_log_entry.label_short = f"{layer_type}_{type_index}:{pass_index}"
        tensor_log_entry.layer_label_short = f"{layer_type}_{type_index}"
        final_lookup_label = (
            tensor_log_entry.layer_label
            if tensor_log_entry.num_passes == 1
            else tensor_log_entry.label
        )
        raw_to_final_layer_labels[tensor_log_entry._label_raw] = final_lookup_label
        raw_to_final_parent_layer_labels[tensor_log_entry._label_raw] = tensor_log_entry.layer_label
        raw_to_final_op_labels[tensor_log_entry._label_raw] = tensor_log_entry.label
        final_to_raw_layer_labels[final_lookup_label] = tensor_log_entry._label_raw
    self._raw_to_final_layer_labels = raw_to_final_layer_labels
    self._raw_to_final_parent_layer_labels = raw_to_final_parent_layer_labels
    self._raw_to_final_op_labels = raw_to_final_op_labels
    self._backward_projection_event_count = None
    self._final_to_raw_layer_labels = final_to_raw_layer_labels


def _log_final_info_for_layers(self: "Trace") -> None:
    """Step 9: Log final metadata for all layers and build module hierarchy.

    Iterates through all layers before Step 11 finalizes retained layer containers
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
    step_index = 1
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
        _normalize_io_role_flags(layer_entry)
        if layer_entry.layer_type in ["input", "buffer"]:
            layer_entry.step_index = 0
        elif layer_entry.layer_type == "output":
            layer_entry.step_index = None  # fix later
        else:
            layer_entry.step_index = step_index
            self.num_ops += 1
            step_index += 1

        # Replace any layer names with their final names:
        _replace_layer_names_for_layer_entry(self, layer_entry)

        # Log the module hierarchy information:
        _log_module_hierarchy_info_for_layer(self, layer_entry, _shadow_sets)
        if layer_entry.atomic_module_call is not None:
            submodule_pass_nice_name = ":".join([str(i) for i in layer_entry.atomic_module_call])
            layer_entry.atomic_module_call = submodule_pass_nice_name

        # Tally the tensor sizes:
        self.total_activation_memory += layer_entry.activation_memory

        # Tally the parameter sizes:
        if layer_entry.layer_label not in unique_layers_seen:  # only count params once
            if layer_entry.uses_params:
                self.num_layers_with_params += 1
            self.num_params += layer_entry.num_params
            self.num_params_trainable += layer_entry.num_params_trainable
            self.num_params_frozen += layer_entry.num_params_frozen
            self.num_param_tensors += layer_entry.num_param_tensors
            self.total_param_memory += layer_entry.param_memory
            # Tally for modules, too.
            for module_name, _ in layer_entry.modules:
                mbd["module_nparams"][module_name] += layer_entry.num_params
                mbd["module_nparams_trainable"][module_name] += layer_entry.num_params_trainable
                mbd["module_nparams_frozen"][module_name] += layer_entry.num_params_frozen

        unique_layers_seen.add(layer_entry.layer_label)

        # Tally elapsed time:

        self.func_calls_duration += layer_entry.func_duration

    _compute_fx_qualpaths(self)
    _finalize_output_compute_indexs(self)
    _build_module_hierarchy_dicts(self)


def _normalize_io_role_flags(layer_entry: Op) -> None:
    """Synchronize input/output/buffer booleans with ``layer_type``.

    Args:
        layer_entry: Op whose role flags should be normalized.
    """

    layer_type = layer_entry.layer_type
    layer_entry.is_input = layer_type == "input"
    layer_entry.is_output = layer_type == "output"
    layer_entry.is_buffer = layer_type == "buffer"


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
    """Assign step_index to output layers (deferred until total is known)."""
    for layer in self.output_layers:
        self[layer].step_index = self.num_ops


def _build_module_hierarchy_dicts(self: "Trace") -> None:
    """Derive top_level_modules and module_children from their pass-level counterparts."""
    mbd = self._module_build_data
    for module in mbd["top_level_module_ops"]:
        module_no_pass = module.split(":")[0]
        if module_no_pass == "self":
            continue
        if module_no_pass not in mbd["top_level_modules"]:
            mbd["top_level_modules"].append(module_no_pass)

    for module_parent, module_children in mbd["module_pass_children"].items():
        module_parent_nopass = module_parent.split(":")[0]
        for module_child in module_children:
            module_child_nopass = module_child.split(":")[0]
            if module_child_nopass == module_parent_nopass:
                continue
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
    "op_equivalence_classes",
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
    3. out_versions_by_child dict: renames keys.

    Args:
        layer_entry: Op to rename labels for.
    """
    mapping = self._raw_to_final_layer_labels
    layer_mapping = self._raw_to_final_parent_layer_labels
    op_mapping = self._raw_to_final_op_labels

    def set_entry_field(field_name: str, value: Any) -> None:
        """Set a renamed entry field on real Ops or lightweight test doubles."""

        internal_set = getattr(layer_entry, "_internal_set", None)
        if internal_set is not None:
            internal_set(field_name, value)
            return
        setattr(layer_entry, field_name, value)

    for field in _LIST_FIELDS_TO_RENAME:
        orig = getattr(layer_entry, field, None)
        if not orig:
            continue
        if field.startswith("conditional_"):
            field_mapping = layer_mapping
        elif field in {"recurrent_ops", "op_equivalence_classes"}:
            field_mapping = op_mapping
        else:
            field_mapping = mapping
        if isinstance(orig, list):
            set_entry_field(field, [field_mapping[raw] for raw in orig])
        else:  # set
            set_entry_field(field, type(orig)(field_mapping[raw] for raw in orig))

    # Fix the arg locations field:
    arg_locs = getattr(layer_entry, "parent_arg_positions", None)
    if arg_locs:
        for arg_type in ("args", "kwargs"):
            sub = arg_locs[arg_type]
            for key, value in sub.items():
                sub[key] = mapping[value]

    # Fix the field names for different children tensor versions:
    ctv = getattr(layer_entry, "out_versions_by_child", None)
    if ctv:
        set_entry_field(
            "out_versions_by_child",
            {mapping[child_label]: tensor_version for child_label, tensor_version in ctv.items()},
        )

    elif_children = getattr(layer_entry, "conditional_elif_children", None)
    if elif_children:
        set_entry_field(
            "conditional_elif_children", _rename_elif_children(elif_children, layer_mapping)
        )

    children_by_cond = getattr(layer_entry, "conditional_arm_children", None)
    if children_by_cond:
        set_entry_field(
            "conditional_arm_children", _rename_children_by_cond(children_by_cond, layer_mapping)
        )

    edge_uses = getattr(layer_entry, "_edge_uses", None)
    if edge_uses:
        set_entry_field(
            "_edge_uses", [_rename_label_dataclass(record, mapping) for record in edge_uses]
        )

    args_template = getattr(layer_entry, "args_template", None)
    if args_template is not None:
        set_entry_field("args_template", _rename_template_parent_refs(args_template, mapping))
    kwargs_template = getattr(layer_entry, "kwargs_template", None)
    if kwargs_template is not None:
        set_entry_field("kwargs_template", _rename_template_parent_refs(kwargs_template, mapping))
    interventions = getattr(layer_entry, "interventions", None)
    if interventions:
        set_entry_field(
            "interventions",
            [_rename_label_dataclass(record, mapping) for record in interventions],
        )


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
    op_label = layer_entry.label
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
        if op_label not in _module_call_labels_seen[module_pass_nice_label]:
            _module_call_labels_seen[module_pass_nice_label].add(op_label)
            mbd["module_pass_layers"][module_pass_nice_label].append(op_label)
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


def _build_lookup_keys_and_finalize_retained_layers(self: "Trace") -> None:
    """Step 11: Build lookup keys and finalize retained layer lists.

    Parameters
    ----------
    self:
        Trace whose raw layer entries should be promoted into final containers.

    Returns
    -------
    None
        Mutates final layer lookup containers in place.
    """

    # Quick loop to count how many tensors are saved:
    for layer_entry in self:
        if getattr(layer_entry, "is_orphan", False):
            continue
        if layer_entry.has_saved_activation:
            self.num_saved_ops += 1

    num_logged_tensors = sum(
        1 for layer_entry in self if not getattr(layer_entry, "is_orphan", False)
    )

    self.layer_list = []
    self.layer_dict_main_keys = {}
    self.layer_labels = []
    self.op_labels = []
    self.layer_num_calls = {}

    i = 0
    for raw_tensor_label in self._raw_layer_labels_list:
        layer_entry = self._raw_layer_dict[raw_tensor_label]
        if getattr(layer_entry, "is_orphan", False):
            continue
        # Add the lookup keys for the layer, to itself and to Trace:
        _add_lookup_keys_for_layer_entry(self, layer_entry, i, num_logged_tensors)

        # Log all information:
        self.layer_list.append(layer_entry)
        self.layer_dict_main_keys[layer_entry.label] = layer_entry
        if layer_entry.layer_label not in self.layer_labels:
            self.layer_labels.append(layer_entry.layer_label)
        self.op_labels.append(layer_entry.label)
        self.layer_num_calls[layer_entry.layer_label] = layer_entry.num_passes
        if layer_entry.has_saved_activation:
            self.saved_activation_memory += layer_entry.activation_memory
        i += 1

    self._layers_logged = True

    if self.num_saved_ops == len(self.layer_list):
        self._layers_saved = True
    else:
        self._layers_saved = False


def _replay_dependency_labels(layer_entry: Op) -> set[str]:
    """Return final labels this entry's replay metadata depends on.

    Args:
        layer_entry: Layer entry to inspect.

    Returns:
        Parent labels referenced by templates, edge provenance, or graph edges.
    """

    dependency_labels = set(layer_entry.parents)
    for edge in getattr(layer_entry, "_edge_uses", []):
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
    - Addresses: address, input/output address.

    Also reformats output_of_module_calls/entered and modules
    from (name, pass) tuples to "name:pass" strings (metadata-building modes only).

    Args:
        layer_entry: Op to build lookup keys for.
        tensor_index: Zero-based position in the final ordered layer list.
        num_tensors_to_keep: Total number of retained tensors (for negative indices).
    """
    # The "default" keys: including the pass if multiple ops, excluding if one pass.
    lookup_keys_for_tensor = [
        layer_entry.layer_label,
        layer_entry.layer_label_short,
        layer_entry.label,
        layer_entry.label_short,
        tensor_index,
        tensor_index - num_tensors_to_keep,
    ]
    layer_entry.ordinal_index = tensor_index

    # Relabel the module ops if this pass built module metadata:
    if self.capture_mode in {"exhaustive", "predicate"}:
        layer_entry.output_of_module_calls = [
            _module_call_label(module_call) for module_call in layer_entry.output_of_module_calls
        ]
        layer_entry.input_to_module_calls = [
            _module_call_label(module_call) for module_call in layer_entry.input_to_module_calls
        ]
        if layer_entry.module is not None:
            layer_entry.module = _module_call_label(layer_entry.module)
        layer_entry.modules = [
            _module_call_label(module_call) for module_call in layer_entry.modules
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
        if self.buffer_num_calls[layer_entry.address] == 1:
            lookup_keys_for_tensor.append(layer_entry.address)
        lookup_keys_for_tensor.append(f"{layer_entry.address}:{layer_entry.buffer_pass}")
    elif layer_entry.is_input or layer_entry.is_output:
        lookup_keys_for_tensor.append(layer_entry.io_role)

    if layer_entry.fx_label is not None:
        lookup_keys_for_tensor.append(layer_entry.fx_label)

    lookup_keys_for_tensor = sorted(lookup_keys_for_tensor, key=str)

    # Log in both the tensor and in the Trace object.
    layer_entry.lookup_keys = lookup_keys_for_tensor
    for lookup_key in lookup_keys_for_tensor:
        if lookup_key not in self._lookup_keys_to_layer_num_dict:
            self._lookup_keys_to_layer_num_dict[lookup_key] = layer_entry.raw_index
            self.layer_dict_all_keys[lookup_key] = layer_entry
        self._layer_num_to_lookup_keys_dict[layer_entry.raw_index].append(lookup_key)


def _module_call_label(module_call: Any) -> str:
    """Return ``module:pass`` text for tuple or already-string module calls.

    Args:
        module_call: Module call represented as ``(address, pass)`` or text.

    Returns:
        Canonical module-call label.
    """

    if isinstance(module_call, str):
        return module_call
    module_name, module_pass = module_call
    return f"{module_name}:{module_pass}"


def _rename_model_history_layer_names(self: "Trace") -> None:
    """Step 10: Rename raw labels to final labels in all Trace-level fields.

    Updates list fields (input_layers, output_layers, etc.), dict fields
    (layers_with_params, op_equivalence_classes), conditional branch
    edges, and module layer argument names. Creates NEW container objects
    (no shared-set corruption) — see set() constructor in op_equivalence_classes
    rename.
    """
    layer_list_fields_to_rename = [
        "input_layers",
        "output_layers",
        "buffer_layers",
    ]
    for field in layer_list_fields_to_rename:
        tensor_labels = getattr(self, field)
        setattr(
            self,
            field,
            [self._raw_layer_dict[tensor_label].layer_label for tensor_label in tensor_labels],
        )

    op_list_fields_to_rename = [
        "internal_source_ops",
        "internal_sink_ops",
        "internally_terminated_bool_ops",
    ]
    for field in op_list_fields_to_rename:
        tensor_labels = getattr(self, field)
        setattr(
            self,
            field,
            [self._raw_to_final_op_labels[tensor_label] for tensor_label in tensor_labels],
        )

    new_param_tensors = {}
    for key, values in self.layers_with_params.items():
        new_key = self[values[0]].layer_label
        new_param_tensors[new_key] = [
            self._raw_to_final_parent_layer_labels[tensor_label] for tensor_label in values
        ]
    self.layers_with_params = new_param_tensors

    saved_layers = [
        layer_entry
        for layer_entry in self.layer_list
        if getattr(layer_entry, "has_saved_activation", False)
        and not getattr(layer_entry, "is_orphan", False)
    ]
    self.num_saved_layers = len({layer_entry.layer_label for layer_entry in saved_layers})
    saved_labels = {layer_entry.layer_label for layer_entry in saved_layers}
    self.num_saved_module_calls = sum(
        1
        for module_call in getattr(self, "module_calls", [])
        if any(label in saved_labels for label in getattr(module_call, "layers", []))
    )

    new_equiv_operations_tensors: dict[Any, set[str]] = {}
    for key, equiv_values in self.op_equivalence_classes.items():
        new_equiv_operations_tensors[key] = set(
            [self._raw_to_final_op_labels[tensor_label] for tensor_label in equiv_values]
        )
    self.op_equivalence_classes = new_equiv_operations_tensors

    for t, (child, parent) in enumerate(self.conditional_branch_edges):
        new_child, new_parent = (
            self._raw_to_final_parent_layer_labels[child],
            self._raw_to_final_parent_layer_labels[parent],
        )
        self.conditional_branch_edges[t] = (new_child, new_parent)

    self.conditional_arm_entry_edges = {
        (cond_id, branch_kind): [
            (
                self._raw_to_final_parent_layer_labels[parent],
                self._raw_to_final_parent_layer_labels[child],
            )
            for parent, child in arm_edges
        ]
        for (cond_id, branch_kind), arm_edges in self.conditional_arm_entry_edges.items()
    }

    conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    for (
        parent,
        child,
        cond_id,
        branch_kind,
    ), call_indexs in self.conditional_edge_call_indices.items():
        edge_key = (
            self._raw_to_final_parent_layer_labels[parent],
            self._raw_to_final_parent_layer_labels[child],
            cond_id,
            branch_kind,
        )
        for call_index in call_indexs:
            if call_index not in conditional_edge_call_indices[edge_key]:
                conditional_edge_call_indices[edge_key].append(call_index)
    self.conditional_edge_call_indices = dict(conditional_edge_call_indices)

    for conditional_event in self.conditional_records:
        conditional_event.bool_layers = [
            self._raw_to_final_parent_layer_labels[layer_label]
            for layer_label in conditional_event.bool_layers
        ]

    self.state_history = _rename_label_dataclass(
        getattr(self, "state_history", []), self._raw_to_final_layer_labels
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
