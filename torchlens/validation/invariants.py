"""Metadata invariant checks for ModelLog and its sub-objects.

Single entry point: ``check_metadata_invariants(model_log)`` runs all checks
and raises ``MetadataInvariantError`` on the first failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


class MetadataInvariantError(ValueError):
    """Raised when a metadata invariant check fails."""

    def __init__(self, check_name: str, message: str):
        super().__init__(f"[{check_name}] {message}")
        self.check_name = check_name


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_metadata_invariants(model_log: "ModelLog") -> bool:
    """Run all metadata invariant checks on a completed ModelLog.

    Raises MetadataInvariantError on the first failure.
    Returns True if all checks pass.
    """
    _check_model_log_self_consistency(model_log)
    _check_special_layer_lists(model_log)
    _check_graph_topology(model_log)
    _check_layer_pass_log_fields(model_log)
    _check_recurrence_invariants(model_log)
    _check_branching_invariants(model_log)
    _check_layer_pass_to_layer_log_xrefs(model_log)
    _check_module_layer_containment(model_log)
    _check_module_hierarchy(model_log)
    _check_param_xrefs(model_log)
    _check_buffer_xrefs(model_log)
    _check_equivalence_symmetry(model_log)
    return True


# ---------------------------------------------------------------------------
# A. ModelLog self-consistency
# ---------------------------------------------------------------------------


def _check_model_log_self_consistency(ml: "ModelLog") -> None:
    name = "model_log_self_consistency"

    # layer_labels vs layer_list length
    if len(ml.layer_labels) != len(ml.layer_list):
        raise MetadataInvariantError(
            name,
            f"len(layer_labels)={len(ml.layer_labels)} != len(layer_list)={len(ml.layer_list)}",
        )

    # No duplicate labels
    if len(ml.layer_labels) != len(set(ml.layer_labels)):
        dupes = [lbl for lbl in ml.layer_labels if ml.layer_labels.count(lbl) > 1]
        raise MetadataInvariantError(name, f"Duplicate layer_labels: {set(dupes)}")

    # num_operations counts computational layers (excludes input, output, buffer)
    excluded = set(ml.input_layers) | set(ml.output_layers) | set(ml.buffer_layers)
    expected_ops = len([lbl for lbl in ml.layer_labels_no_pass if lbl not in excluded])
    if ml.num_operations != expected_ops:
        raise MetadataInvariantError(
            name,
            f"num_operations={ml.num_operations} != expected computational layers={expected_ops}",
        )

    # Param counts: total_param_tensors counts per unique layer (may double-count
    # shared params), while len(param_logs) is unique parameter count.
    # total_param_tensors >= len(param_logs) is the valid relationship.
    if ml.total_param_tensors < len(ml.param_logs):
        raise MetadataInvariantError(
            name,
            f"total_param_tensors={ml.total_param_tensors} < len(param_logs)={len(ml.param_logs)}",
        )

    param_sum = sum(p.num_params for p in ml.param_logs)
    if param_sum > ml.total_params:
        raise MetadataInvariantError(
            name,
            f"sum(param.num_params)={param_sum} > total_params={ml.total_params} "
            f"(unique params exceed total — possible accounting error)",
        )

    if ml.total_params_trainable + ml.total_params_frozen != ml.total_params:
        raise MetadataInvariantError(
            name,
            f"trainable({ml.total_params_trainable}) + frozen({ml.total_params_frozen}) "
            f"!= total({ml.total_params})",
        )

    # At least one output layer
    if len(ml.output_layers) == 0:
        raise MetadataInvariantError(name, "No output layers found")

    # Timing
    if ml.elapsed_time_total < 0:
        raise MetadataInvariantError(name, f"elapsed_time_total={ml.elapsed_time_total} < 0")
    if ml.pass_start_time > ml.pass_end_time:
        raise MetadataInvariantError(
            name,
            f"pass_start_time={ml.pass_start_time} > pass_end_time={ml.pass_end_time}",
        )

    # Tensor counts
    if ml.num_tensors_total < ml.num_tensors_saved:
        raise MetadataInvariantError(
            name,
            f"num_tensors_total={ml.num_tensors_total} < num_tensors_saved={ml.num_tensors_saved}",
        )


# ---------------------------------------------------------------------------
# B. Special layer lists ↔ LayerPassLog flags
# ---------------------------------------------------------------------------

_SPECIAL_LIST_FLAG_PAIRS = [
    ("input_layers", "is_input_layer"),
    ("output_layers", "is_output_layer"),
    ("buffer_layers", "is_buffer_layer"),
    ("internally_initialized_layers", "initialized_inside_model"),
    ("internally_terminated_layers", "terminated_inside_model"),
]


def _check_special_layer_lists(ml: "ModelLog") -> None:
    name = "special_layer_lists"
    label_set = set(ml.layer_labels)

    for list_attr, flag_attr in _SPECIAL_LIST_FLAG_PAIRS:
        special_list = getattr(ml, list_attr)
        special_set = set(special_list)

        # All entries must be valid labels
        missing = special_set - label_set
        if missing:
            raise MetadataInvariantError(
                name, f"{list_attr} contains labels not in layer_labels: {missing}"
            )

        # Forward: every label in the list has the flag set
        for label in special_list:
            lpl = ml[label]
            if not getattr(lpl, flag_attr):
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} is in {list_attr} but {flag_attr}=False",
                )

        # Reverse: every layer with the flag is in the list
        for lpl in ml.layer_list:
            if getattr(lpl, flag_attr) and lpl.layer_label not in special_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer {lpl.layer_label} has {flag_attr}=True but is not in {list_attr}",
                )


# ---------------------------------------------------------------------------
# C. Graph topology
# ---------------------------------------------------------------------------


def _check_graph_topology(ml: "ModelLog") -> None:
    name = "graph_topology"
    label_set = set(ml.layer_labels)
    output_set = set(ml.output_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Parent-child bidirectionality
        for p in lpl.parent_layers:
            if p not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has parent {p} not in layer_labels"
                )
            parent = ml[p]
            if label not in parent.child_layers:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {p} as parent, but {p} does not list {label} as child",
                )

        for c in lpl.child_layers:
            if c not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has child {c} not in layer_labels"
                )
            child = ml[c]
            if label not in child.parent_layers:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {c} as child, but {c} does not list {label} as parent",
                )

        # Boolean flag consistency
        # Note: has_children/has_parents/has_siblings/has_spouses are set during
        # capture and don't account for output layers added during postprocessing.
        # So we exclude output layers from the child count for this check.
        non_output_children = [c for c in lpl.child_layers if c not in output_set]
        if lpl.has_children != (len(non_output_children) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_children={lpl.has_children} but "
                f"non-output child_layers={non_output_children}",
            )
        if not lpl.is_output_layer and lpl.has_parents != (len(lpl.parent_layers) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_parents={lpl.has_parents} but "
                f"len(parent_layers)={len(lpl.parent_layers)}",
            )
        if lpl.has_siblings != (len(lpl.sibling_layers) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_siblings={lpl.has_siblings} but "
                f"len(sibling_layers)={len(lpl.sibling_layers)}",
            )
        if lpl.has_spouses != (len(lpl.spouse_layers) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_spouses={lpl.has_spouses} but "
                f"len(spouse_layers)={len(lpl.spouse_layers)}",
            )

        # Input layers have no parents
        if lpl.is_input_layer and len(lpl.parent_layers) > 0:
            raise MetadataInvariantError(
                name,
                f"Input layer {label} has parent_layers={lpl.parent_layers}",
            )

        # children_tensor_versions keys subset of child_layers
        ctv_keys = set(lpl.children_tensor_versions.keys())
        child_set = set(lpl.child_layers)
        extra = ctv_keys - child_set
        if extra:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: children_tensor_versions has keys not in child_layers: {extra}",
            )


# ---------------------------------------------------------------------------
# D. LayerPassLog field consistency
# ---------------------------------------------------------------------------


def _check_layer_pass_log_fields(ml: "ModelLog") -> None:
    name = "layer_pass_log_fields"

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Tensor shape/dtype consistency when activations are saved
        if lpl.has_saved_activations and lpl.tensor_contents is not None:
            actual_shape = tuple(lpl.tensor_contents.shape)
            if lpl.tensor_shape != actual_shape:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: tensor_shape={lpl.tensor_shape} != "
                    f"actual shape={actual_shape}",
                )
            if lpl.tensor_dtype != lpl.tensor_contents.dtype:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: tensor_dtype={lpl.tensor_dtype} != "
                    f"actual dtype={lpl.tensor_contents.dtype}",
                )

        # Pass numbering
        if lpl.pass_num < 1:
            raise MetadataInvariantError(name, f"Layer {label}: pass_num={lpl.pass_num} < 1")
        if lpl.layer_passes_total < lpl.pass_num:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: layer_passes_total={lpl.layer_passes_total} "
                f"< pass_num={lpl.pass_num}",
            )

        # Function applied (non-input, non-buffer, non-output layers)
        if not (lpl.is_input_layer or lpl.is_buffer_layer or lpl.is_output_layer):
            if not callable(lpl.func_applied):
                raise MetadataInvariantError(name, f"Layer {label}: func_applied is not callable")
            if not lpl.func_applied_name:
                raise MetadataInvariantError(name, f"Layer {label}: func_applied_name is empty")

        # Operation numbering (input/buffer layers have operation_num=0)
        if not (lpl.is_input_layer or lpl.is_buffer_layer):
            if lpl.operation_num is not None and lpl.operation_num < 1:
                raise MetadataInvariantError(
                    name, f"Layer {label}: operation_num={lpl.operation_num} < 1"
                )
        if lpl.realtime_tensor_num < 1:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: realtime_tensor_num={lpl.realtime_tensor_num} < 1",
            )

        # Module nesting depth
        if lpl.module_nesting_depth != len(lpl.containing_modules_origin_nested):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_nesting_depth={lpl.module_nesting_depth} != "
                f"len(containing_modules_origin_nested)="
                f"{len(lpl.containing_modules_origin_nested)}",
            )

        # Label format: pass-qualified label has ":" iff multi-pass
        if lpl.layer_passes_total > 1 and ":" not in lpl.layer_label_w_pass:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: multi-pass but layer_label_w_pass="
                f"'{lpl.layer_label_w_pass}' has no ':'",
            )
        if ":" in lpl.layer_label_no_pass:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: layer_label_no_pass='{lpl.layer_label_no_pass}' contains ':'",
            )


# ---------------------------------------------------------------------------
# E. Recurrence / loop invariants
# ---------------------------------------------------------------------------


def _check_recurrence_invariants(ml: "ModelLog") -> None:
    name = "recurrence_invariants"

    any_recurrent = any(v > 1 for v in ml.layer_num_passes.values())
    if ml.model_is_recurrent != any_recurrent:
        raise MetadataInvariantError(
            name,
            f"model_is_recurrent={ml.model_is_recurrent} but "
            f"any layer has >1 pass = {any_recurrent}",
        )

    if ml.model_is_recurrent:
        expected_max = max(ml.layer_num_passes.values())
        if ml.model_max_recurrent_loops != expected_max:
            raise MetadataInvariantError(
                name,
                f"model_max_recurrent_loops={ml.model_max_recurrent_loops} != "
                f"max(layer_num_passes)={expected_max}",
            )

    # Per-layer pass consistency: layer_num_passes keys may be pass-qualified
    # (e.g. 'linear_1_1:1') for nested loops, or plain no-pass labels.
    # Validate that each key exists in layer_dict_main_keys.
    main_keys = set(ml.layer_dict_main_keys.keys())
    for label_key, num_passes in ml.layer_num_passes.items():
        if label_key not in main_keys:
            raise MetadataInvariantError(
                name,
                f"layer_num_passes key '{label_key}' not in layer_dict_main_keys",
            )

    # For top-level (no-pass) layer_logs, verify pass dict consistency
    for no_pass_label, ll in ml.layer_logs.items():
        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.passes.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"LayerLog '{no_pass_label}' passes keys={actual_keys} != expected {expected_keys}",
            )


# ---------------------------------------------------------------------------
# F. Branching invariants
# ---------------------------------------------------------------------------


def _check_branching_invariants(ml: "ModelLog") -> None:
    name = "branching_invariants"
    any_branching = any(len(lpl.child_layers) > 1 for lpl in ml.layer_list)
    if ml.model_is_branching != any_branching:
        raise MetadataInvariantError(
            name,
            f"model_is_branching={ml.model_is_branching} but "
            f"any layer has >1 child = {any_branching}",
        )


# ---------------------------------------------------------------------------
# G. LayerPassLog ↔ LayerLog cross-references
# ---------------------------------------------------------------------------


def _check_layer_pass_to_layer_log_xrefs(ml: "ModelLog") -> None:
    name = "layer_pass_layer_log_xrefs"

    for ll_label, ll in ml.layer_logs.items():
        if ll.layer_label != ll_label:
            raise MetadataInvariantError(
                name,
                f"LayerLog key '{ll_label}' != LayerLog.layer_label='{ll.layer_label}'",
            )

        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.passes.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"LayerLog '{ll_label}' passes keys={actual_keys} != expected {expected_keys}",
            )

        for pass_num, lpl in ll.passes.items():
            if lpl.pass_num != pass_num:
                raise MetadataInvariantError(
                    name,
                    f"LayerLog '{ll_label}' pass key={pass_num} but "
                    f"LayerPassLog.pass_num={lpl.pass_num}",
                )
            if lpl.layer_label_no_pass != ll.layer_label:
                raise MetadataInvariantError(
                    name,
                    f"LayerPassLog '{lpl.layer_label}' layer_label_no_pass="
                    f"'{lpl.layer_label_no_pass}' != "
                    f"parent LayerLog.layer_label='{ll.layer_label}'",
                )
            if lpl.parent_layer_log is not ll:
                raise MetadataInvariantError(
                    name,
                    f"LayerPassLog '{lpl.layer_label}' parent_layer_log "
                    f"does not point to its LayerLog '{ll_label}'",
                )


# ---------------------------------------------------------------------------
# H. Module ↔ Layer containment
# ---------------------------------------------------------------------------


def _check_module_layer_containment(ml: "ModelLog") -> None:
    name = "module_layer_containment"
    mod_accessor = ml.modules
    label_set = set(ml.layer_labels)

    for mod_log in mod_accessor:
        addr = mod_log.address

        # ModuleLog.all_layers labels exist in layer_logs
        for lbl in mod_log.all_layers:
            if lbl not in ml.layer_logs:
                raise MetadataInvariantError(
                    name,
                    f"ModuleLog '{addr}' all_layers contains '{lbl}' not in model_log.layer_logs",
                )

        if mod_log.num_layers != len(mod_log.all_layers):
            raise MetadataInvariantError(
                name,
                f"ModuleLog '{addr}': num_layers={mod_log.num_layers} != "
                f"len(all_layers)={len(mod_log.all_layers)}",
            )

        # ModulePassLog checks
        for pass_num, mpl in mod_log.passes.items():
            for lbl in mpl.layers:
                if lbl not in label_set:
                    raise MetadataInvariantError(
                        name,
                        f"ModulePassLog '{addr}:{pass_num}' layers contains "
                        f"'{lbl}' not in layer_labels",
                    )

            if mpl.num_layers != len(mpl.layers):
                raise MetadataInvariantError(
                    name,
                    f"ModulePassLog '{addr}:{pass_num}': "
                    f"num_layers={mpl.num_layers} != len(layers)={len(mpl.layers)}",
                )

            # input/output layers subset of layers
            mpl_layer_set = set(mpl.layers)
            for sub_attr in ("input_layers", "output_layers"):
                sub_list = getattr(mpl, sub_attr)
                sub_set = set(sub_list)
                extra = sub_set - mpl_layer_set
                if extra:
                    raise MetadataInvariantError(
                        name,
                        f"ModulePassLog '{addr}:{pass_num}' "
                        f"{sub_attr} has labels not in layers: {extra}",
                    )

    # Reverse check: layer's containing_module_origin exists in modules
    for lpl in ml.layer_list:
        cmo = lpl.containing_module_origin
        if cmo:
            # containing_module_origin may include pass suffix (e.g. 'fc:1')
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            try:
                mod = mod_accessor[cmo_addr]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' containing_module_origin='{cmo}' "
                    f"(addr='{cmo_addr}') not found in module accessor",
                )
            if lpl.layer_label_no_pass not in mod.all_layers:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' (no_pass='{lpl.layer_label_no_pass}') "
                    f"not in ModuleLog '{cmo_addr}'.all_layers",
                )


# ---------------------------------------------------------------------------
# I. Module hierarchy consistency
# ---------------------------------------------------------------------------


def _check_module_hierarchy(ml: "ModelLog") -> None:
    name = "module_hierarchy"
    mod_accessor = ml.modules

    # Root module exists
    try:
        mod_accessor["self"]
    except (KeyError, IndexError):
        raise MetadataInvariantError(name, "'self' module not found in module accessor")

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Address hierarchy bidirectional
        if mod_log.address_parent is not None:
            try:
                parent = mod_accessor[mod_log.address_parent]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"ModuleLog '{addr}' address_parent='{mod_log.address_parent}' "
                    f"not in module accessor",
                )
            if addr not in parent.address_children:
                raise MetadataInvariantError(
                    name,
                    f"ModuleLog '{addr}' has address_parent='{mod_log.address_parent}' "
                    f"but parent doesn't list it in address_children",
                )

        for child_addr in mod_log.address_children:
            try:
                child = mod_accessor[child_addr]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"ModuleLog '{addr}' address_children contains '{child_addr}' "
                    f"not in module accessor",
                )
            if child.address_parent != addr:
                raise MetadataInvariantError(
                    name,
                    f"ModuleLog '{addr}' lists '{child_addr}' as address_child, "
                    f"but child's address_parent='{child.address_parent}'",
                )

        # Module pass consistency
        if len(mod_log.passes) != mod_log.num_passes:
            raise MetadataInvariantError(
                name,
                f"ModuleLog '{addr}': len(passes)={len(mod_log.passes)} != "
                f"num_passes={mod_log.num_passes}",
            )

        expected_keys = set(range(1, mod_log.num_passes + 1))
        actual_keys = set(mod_log.passes.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"ModuleLog '{addr}' pass keys={actual_keys} != expected {expected_keys}",
            )

        # Call hierarchy: parent exists
        for pass_num, mpl in mod_log.passes.items():
            if mpl.call_parent is not None:
                try:
                    mod_accessor[mpl.call_parent]
                except (KeyError, IndexError):
                    raise MetadataInvariantError(
                        name,
                        f"ModulePassLog '{addr}:{pass_num}' call_parent="
                        f"'{mpl.call_parent}' not in module accessor",
                    )
            for cc in mpl.call_children:
                try:
                    mod_accessor[cc]
                except (KeyError, IndexError):
                    raise MetadataInvariantError(
                        name,
                        f"ModulePassLog '{addr}:{pass_num}' call_children "
                        f"contains '{cc}' not in module accessor",
                    )


# ---------------------------------------------------------------------------
# J. Param ↔ Layer ↔ Module cross-references
# ---------------------------------------------------------------------------


def _check_param_xrefs(ml: "ModelLog") -> None:
    name = "param_xrefs"
    label_set = set(ml.layer_labels)
    mod_accessor = ml.modules

    for param in ml.param_logs:
        # layer_log_entries exist
        for lbl in param.layer_log_entries:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"ParamLog '{param.address}' layer_log_entries contains "
                    f"'{lbl}' not in layer_labels",
                )

        # module_address exists
        try:
            mod_accessor[param.module_address]
        except (KeyError, IndexError):
            raise MetadataInvariantError(
                name,
                f"ParamLog '{param.address}' module_address='{param.module_address}' "
                f"not in module accessor",
            )

    # computed_with_params forward check
    for lpl in ml.layer_list:
        if lpl.computed_with_params:
            if not lpl.parent_param_logs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' has computed_with_params=True "
                    f"but parent_param_logs is empty",
                )

    # layers_computed_with_params labels exist
    for param_addr, layer_labels in ml.layers_computed_with_params.items():
        for lbl in layer_labels:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"layers_computed_with_params['{param_addr}'] contains "
                    f"'{lbl}' not in layer_labels",
                )


# ---------------------------------------------------------------------------
# K. Buffer cross-references
# ---------------------------------------------------------------------------


def _check_buffer_xrefs(ml: "ModelLog") -> None:
    name = "buffer_xrefs"
    label_set = set(ml.layer_labels)

    for lbl in ml.buffer_layers:
        if lbl not in label_set:
            raise MetadataInvariantError(
                name, f"buffer_layers contains '{lbl}' not in layer_labels"
            )

    # Check BufferLog objects via buffer accessor
    if hasattr(ml, "_buffer_accessor") and ml._buffer_accessor is not None:
        for buf in ml.buffers:
            if not buf.buffer_address:
                raise MetadataInvariantError(
                    name,
                    f"BufferLog '{buf.layer_label}' has empty buffer_address",
                )
            # module_address references a valid module
            try:
                ml.modules[buf.module_address]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"BufferLog '{buf.layer_label}' module_address="
                    f"'{buf.module_address}' not in module accessor",
                )


# ---------------------------------------------------------------------------
# L. Equivalence group symmetry
# ---------------------------------------------------------------------------


def _check_equivalence_symmetry(ml: "ModelLog") -> None:
    name = "equivalence_symmetry"
    label_set = set(ml.layer_labels)

    # equivalent_operations is keyed by equivalence type descriptors (not layer labels),
    # with values being sets of layer labels in that equivalence group.
    for eq_type, equiv_set in ml.equivalent_operations.items():
        if not isinstance(equiv_set, set):
            raise MetadataInvariantError(
                name,
                f"equivalent_operations['{eq_type}'] is not a set",
            )
        for label in equiv_set:
            if label not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"equivalent_operations['{eq_type}'] contains '{label}' not in layer_labels",
                )

    # Each layer that appears in any equivalence group should exist
    all_equiv_labels = set()
    for equiv_set in ml.equivalent_operations.values():
        all_equiv_labels.update(equiv_set)
    # All labels in equivalence groups are a subset of no-pass labels
    no_pass_set = set(ml.layer_labels_no_pass)
    extra = all_equiv_labels - no_pass_set
    if extra:
        raise MetadataInvariantError(
            name,
            f"equivalent_operations contains labels not in layer_labels_no_pass: {extra}",
        )
