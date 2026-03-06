"""Metadata invariant checks for ModelLog and its sub-objects.

Single entry point: ``check_metadata_invariants(model_log)`` runs all checks
and raises ``MetadataInvariantError`` on the first failure.

**Phase 1 -- Structural invariants (checks A-L):**
  A. ModelLog self-consistency (counts, timing, label uniqueness)
  B. Special layer lists match per-layer boolean flags
  C. Graph topology (parent-child bidirectionality, boolean flag consistency)
  D. LayerPassLog field consistency (shape, dtype, pass numbering, nesting)
  E. Recurrence / loop invariants (model_is_recurrent, pass dicts)
  F. Branching invariants (model_is_branching)
  G. LayerPassLog <-> LayerLog cross-references (pass numbering, back-pointers)
  H. Module <-> Layer containment (all_layers, module pass layers, reverse check)
  I. Module hierarchy (address parent-child bidirectionality, pass consistency)
  J. Param cross-references (ParamLog -> layer, computed_with_params flag)
  K. Buffer cross-references (buffer_layers list, BufferLog module references)
  L. Equivalence group symmetry (equivalent_operations labels are valid)

**Phase 2 -- Semantic invariants (checks M-R):**
  M. Graph ordering (realtime_tensor_num uniqueness/monotonicity, topological order, no raw labels)
  N. Loop detection invariants (same_layer_operations symmetry, func identity, param sharing, pass numbering)
  O. Distance / reachability (min <= max, input/output layer distances == 0, ancestor/descendent consistency)
  P. Graph connectivity (non-input non-buffer layers have parents, orphans removed)
  Q. Module containment logic (address acyclicity, depth consistency, nested path ordering)
  R. Lookup key bidirectionality (forward/reverse dicts, raw/final label maps)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog
    from ..data_classes.module_log import ModuleLog


class MetadataInvariantError(ValueError):
    """Raised when a metadata invariant check fails.

    Embeds the check name (e.g., ``"graph_topology"``) in the message prefix
    and stores it as an attribute for programmatic inspection in tests.
    """

    def __init__(self, check_name: str, message: str):
        super().__init__(f"[{check_name}] {message}")
        self.check_name = check_name


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_metadata_invariants(model_log: "ModelLog") -> bool:
    """Run all 18 metadata invariant checks (A-R) on a completed ModelLog.

    Checks run in dependency order: Phase 1 structural checks first (A-L),
    then Phase 2 semantic checks (M-R).  Raises ``MetadataInvariantError``
    on the first failure, so later checks can assume earlier ones passed.

    Returns True if all checks pass.
    """
    # --- Phase 1: structural invariants (A-L) ---
    _check_model_log_self_consistency(model_log)  # A
    _check_special_layer_lists(model_log)  # B
    _check_graph_topology(model_log)  # C
    _check_layer_pass_log_fields(model_log)  # D
    _check_recurrence_invariants(model_log)  # E
    _check_branching_invariants(model_log)  # F
    _check_layer_pass_to_layer_log_xrefs(model_log)  # G
    _check_module_layer_containment(model_log)  # H
    _check_module_hierarchy(model_log)  # I
    _check_param_xrefs(model_log)  # J
    _check_buffer_xrefs(model_log)  # K
    _check_equivalence_symmetry(model_log)  # L

    # --- Phase 2: semantic invariants (M-R) ---
    _check_graph_ordering(model_log)  # M
    _check_loop_detection_invariants(model_log)  # N
    _check_distance_invariants(model_log)  # O
    _check_graph_connectivity(model_log)  # P
    _check_module_containment_logic(model_log)  # Q
    _check_lookup_key_consistency(model_log)  # R
    return True


# ---------------------------------------------------------------------------
# A. ModelLog self-consistency
# ---------------------------------------------------------------------------


def _check_model_log_self_consistency(ml: "ModelLog") -> None:
    """Check A: ModelLog aggregate counts and metadata are internally consistent.

    Validates:
    - layer_labels length matches layer_list length, no duplicates.
    - num_operations == count of computational (non-input, non-output,
      non-buffer) layers.
    - Param counts (total, trainable, frozen) are consistent and sum correctly.
      Uses deduplication by layer_label_no_pass to match labeling.py logic.
    - At least one output layer exists.
    - Timing values are non-negative and ordered.
    - Tensor counts: total >= saved.
    """
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

    # num_operations counts computational layers only (excludes input, output,
    # buffer).  We check per-layer flags instead of comparing against label
    # sets because buffer_layers stores pass-qualified labels while
    # layer_labels_no_pass strips the pass suffix -- they use different formats.
    expected_ops = sum(
        1
        for lpl in ml.layer_list
        if not (lpl.is_input_layer or lpl.is_output_layer or lpl.is_buffer_layer)
    )
    if ml.num_operations != expected_ops:
        raise MetadataInvariantError(
            name,
            f"num_operations={ml.num_operations} != expected computational layers={expected_ops}",
        )

    # Param counts must be deduplicated by layer_label_no_pass because
    # multi-pass layers share the same params -- counting each pass would
    # double-count.  This matches the summation logic in labeling.py:116-122.
    seen_no_pass: set[str] = set()
    expected_param_sum = 0
    expected_total_params = 0
    for lpl in ml.layer_list:
        if lpl.layer_label_no_pass not in seen_no_pass:
            expected_param_sum += lpl.num_param_tensors
            expected_total_params += lpl.num_params_total
            seen_no_pass.add(lpl.layer_label_no_pass)
    if ml.total_param_tensors != expected_param_sum:
        raise MetadataInvariantError(
            name,
            f"total_param_tensors={ml.total_param_tensors} != "
            f"sum(unique num_param_tensors)={expected_param_sum}",
        )
    if ml.total_params != expected_total_params:
        raise MetadataInvariantError(
            name,
            f"total_params={ml.total_params} != "
            f"sum(unique num_params_total)={expected_total_params}",
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
    """Check B: special layer lists (input, output, buffer, etc.) match per-layer boolean flags.

    For each (list_attr, flag_attr) pair, verifies bidirectional consistency:
    - Forward: every label in the list has the flag set on its LayerPassLog.
    - Reverse: every LayerPassLog with the flag set appears in the list.
    """
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
    """Check C: parent-child edge bidirectionality and boolean flag consistency.

    Validates:
    - Every parent edge has a corresponding child edge (and vice versa).
    - has_children/has_parents/has_siblings/has_spouses flags match actual counts.
      Note: has_children excludes output layers (added during postprocessing,
      not during capture when the flag was set).
    - Input layers have no parents.
    - children_tensor_versions keys are a subset of child_layers.
    """
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
    """Check D: per-layer field consistency (shape, dtype, pass numbering, func, nesting).

    Validates:
    - Saved tensor shape/dtype match actual tensor_contents (when saved).
    - Pass numbering: pass_num >= 1, layer_passes_total >= pass_num.
    - Computational layers have callable func_applied and non-empty func_applied_name.
    - operation_num >= 1 for non-input/non-buffer layers.
    - module_nesting_depth matches len(containing_modules_origin_nested).
    - Label format: pass-qualified label has ':' iff multi-pass; no-pass label never has ':'.
    """
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
    """Check E: recurrence / loop invariants.

    Validates:
    - model_is_recurrent == True iff any layer has >1 pass.
    - model_max_recurrent_loops matches the maximum pass count.
    - layer_num_passes keys are valid no-pass labels.
    - LayerLog.passes dict keys are contiguous {1..N}.
    """
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

    # Per-layer pass consistency: layer_num_passes is keyed by no-pass labels.
    # Validate that each key exists in layer_labels_no_pass.
    no_pass_labels = set(ml.layer_labels_no_pass)
    for label_key, num_passes in ml.layer_num_passes.items():
        if label_key not in no_pass_labels:
            raise MetadataInvariantError(
                name,
                f"layer_num_passes key '{label_key}' not in layer_labels_no_pass",
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
    """Check F: model_is_branching matches whether any layer has >1 child."""
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
    """Check G: LayerPassLog <-> LayerLog cross-references.

    Validates:
    - LayerLog key matches its layer_label.
    - passes dict keys are contiguous {1..N}.
    - Each LayerPassLog's pass_num matches its dict key.
    - Each LayerPassLog's layer_label_no_pass matches the parent LayerLog's label.
    - parent_layer_log back-pointer is identity-equal to the LayerLog.
    """
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
    """Check H: Module <-> Layer containment consistency.

    Validates forward and reverse directions:
    - Forward: ModuleLog.all_layers labels exist in layer_logs; num_layers matches.
      ModulePassLog.layers labels exist; input/output_layers subset of layers.
    - Reverse: each layer's containing_module_origin points to a valid module
      that lists the layer in its all_layers.
    """
    name = "module_layer_containment"
    mod_accessor = ml.modules
    label_set = set(ml.layer_labels)
    no_pass_set = set(ml.layer_labels_no_pass)

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
        # mpl.layers may contain pass-qualified labels OR no-pass labels
        # (e.g., root module in recurrent models uses no-pass labels).
        for pass_num, mpl in mod_log.passes.items():
            for lbl in mpl.layers:
                if lbl not in label_set and lbl not in no_pass_set:
                    raise MetadataInvariantError(
                        name,
                        f"ModulePassLog '{addr}:{pass_num}' layers contains "
                        f"'{lbl}' not in layer_labels or layer_labels_no_pass",
                    )

            if mpl.num_layers != len(mpl.layers):
                raise MetadataInvariantError(
                    name,
                    f"ModulePassLog '{addr}:{pass_num}': "
                    f"num_layers={mpl.num_layers} != len(layers)={len(mpl.layers)}",
                )

            # input/output layers subset of layers (using both pass-qualified
            # and no-pass labels to handle recurrent models)
            mpl_layer_set = set(mpl.layers)
            valid_set = mpl_layer_set | label_set | no_pass_set
            for sub_attr in ("input_layers", "output_layers"):
                sub_list = getattr(mpl, sub_attr)
                sub_set = set(sub_list)
                extra = sub_set - valid_set
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
            if lpl.layer_label_no_pass not in mod.all_layers:  # type: ignore[union-attr]
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' (no_pass='{lpl.layer_label_no_pass}') "
                    f"not in ModuleLog '{cmo_addr}'.all_layers",
                )


# ---------------------------------------------------------------------------
# I. Module hierarchy consistency
# ---------------------------------------------------------------------------


def _check_module_hierarchy(ml: "ModelLog") -> None:
    """Check I: module address tree consistency and pass structure.

    Validates:
    - Root module 'self' exists.
    - Address parent-child bidirectionality (with exemptions for shared
      modules, where aliases may diverge from the primary path).
    - Container modules (ModuleList) that were never called may not have
      ModuleLogs -- skip rather than error.
    - Pass dict keys are contiguous {1..N} and match num_passes.
    - call_parent and call_children reference valid modules.
    """
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
                parent: ModuleLog = mod_accessor[mod_log.address_parent]  # type: ignore[assignment]
            except (KeyError, IndexError):
                # Parent module may be a container (ModuleList, ModuleDict)
                # that is never called during the forward pass, so no
                # ModuleLog exists.  Skip rather than error.
                parent = None  # type: ignore[assignment]
            if parent is not None and addr not in parent.address_children:
                # For shared modules, addr may be an alias that the parent
                # lists under a different address prefix.  Check if any of the
                # parent's address_children resolve to the same ModuleLog.
                if not mod_log.is_shared:
                    raise MetadataInvariantError(
                        name,
                        f"ModuleLog '{addr}' has address_parent='{mod_log.address_parent}' "
                        f"but parent doesn't list it in address_children",
                    )

        for child_addr in mod_log.address_children:
            try:
                child: ModuleLog = mod_accessor[child_addr]  # type: ignore[assignment]
            except (KeyError, IndexError):
                # Static children may not have been invoked during the forward
                # pass, so no ModuleLog exists.  Skip rather than error.
                continue
            if child.address_parent != addr:
                # For shared modules (same nn.Module registered under multiple
                # addresses), the child's address_parent refers to its primary
                # alias's parent, which may differ from the current parent addr.
                # This is expected — there is only one ModuleLog per module
                # instance, so address_parent always reflects the primary path.
                if not child.is_shared:
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
    """Check J: Param <-> Layer <-> Module cross-references.

    Validates:
    - ParamLog.layer_log_entries labels are valid layer labels.
    - computed_with_params == True implies parent_param_logs is non-empty.
    - layers_computed_with_params values are valid layer labels.
    """
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

        # module_address exists (skip for conditional models where the module
        # was never called, e.g. MoE routing that skips some experts)
        try:
            mod_accessor[param.module_address]
        except (KeyError, IndexError):
            pass  # Module was never invoked during forward pass

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
    """Check K: buffer layer and BufferLog cross-references.

    Validates:
    - buffer_layers list entries are valid layer labels.
    - BufferLog objects have non-empty buffer_address and valid module_address.
    """
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
    """Check L: equivalent_operations groups reference valid layer labels.

    Validates:
    - Each equivalence set value is actually a set.
    - All labels in equivalence sets exist in layer_labels (pass-qualified
      or no-pass, since recurrent models may use either form).
    """
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

    # Each layer that appears in any equivalence group should exist.
    # Labels may be no-pass or pass-qualified (for recurrent models).
    all_equiv_labels = set()
    for equiv_set in ml.equivalent_operations.values():
        all_equiv_labels.update(equiv_set)
    no_pass_set = set(ml.layer_labels_no_pass)
    pass_set = set(ml.layer_labels)
    valid_set = no_pass_set | pass_set
    extra = all_equiv_labels - valid_set
    if extra:
        raise MetadataInvariantError(
            name,
            f"equivalent_operations contains labels not in layer_labels: {extra}",
        )


# ---------------------------------------------------------------------------
# M. Graph ordering invariants
# ---------------------------------------------------------------------------

# Raw labels (e.g., "l_42") are internal identifiers assigned during capture
# and must be replaced by human-readable labels during postprocessing.
_RAW_LABEL_PATTERN = re.compile(r"^l_\d+$")


def _check_graph_ordering(ml: "ModelLog") -> None:
    """Check M: graph ordering invariants.

    Validates:
    - realtime_tensor_num is unique across all layers and monotonically
      increasing in layer_list order.
    - operation_num is unique among computational layers (non-input, non-buffer,
      non-output).
    - Topological order: every parent's realtime_tensor_num < child's.
    - No raw labels (``l_\\d+``) survive postprocessing.
    """
    name = "graph_ordering"

    # realtime_tensor_num uniqueness and monotonicity
    seen_rt_nums: dict[int, str] = {}
    prev_rt = -1
    for lpl in ml.layer_list:
        rt = lpl.realtime_tensor_num
        if rt in seen_rt_nums:
            raise MetadataInvariantError(
                name,
                f"Duplicate realtime_tensor_num={rt}: '{seen_rt_nums[rt]}' and '{lpl.layer_label}'",
            )
        seen_rt_nums[rt] = lpl.layer_label
        if rt <= prev_rt:
            raise MetadataInvariantError(
                name,
                f"realtime_tensor_num not monotonically increasing: "
                f"{prev_rt} then {rt} at '{lpl.layer_label}'",
            )
        prev_rt = rt

    # operation_num uniqueness among computational layers
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)
    output_set = set(ml.output_layers)
    seen_op_nums: dict[int, str] = {}
    for lpl in ml.layer_list:
        label = lpl.layer_label
        if label in input_set or label in buffer_set or label in output_set:
            continue
        op = lpl.operation_num
        if op is not None:
            if op in seen_op_nums:
                raise MetadataInvariantError(
                    name,
                    f"Duplicate operation_num={op}: '{seen_op_nums[op]}' and '{label}'",
                )
            seen_op_nums[op] = label

    # Topological order: parent.realtime_tensor_num < child.realtime_tensor_num
    rt_map = {lpl.layer_label: lpl.realtime_tensor_num for lpl in ml.layer_list}
    for lpl in ml.layer_list:
        for p in lpl.parent_layers:
            if rt_map.get(p, -1) >= lpl.realtime_tensor_num:
                raise MetadataInvariantError(
                    name,
                    f"Topological violation: parent '{p}' (rt={rt_map.get(p)}) "
                    f">= child '{lpl.layer_label}' (rt={lpl.realtime_tensor_num})",
                )

    # No raw labels survive postprocessing
    for label in ml.layer_labels:
        if _RAW_LABEL_PATTERN.match(label):
            raise MetadataInvariantError(name, f"Raw label '{label}' survived postprocessing")


# ---------------------------------------------------------------------------
# N. Layer equivalence / loop detection invariants
# ---------------------------------------------------------------------------


def _check_loop_detection_invariants(ml: "ModelLog") -> None:
    """Check N: loop detection / same_layer_operations invariants.

    Validates per-layer:
    - same_layer_operations is non-empty and includes self.
    - Symmetry: all members agree on the same group.
    - All members share: layer_label_no_pass, operation_equivalence_type,
      func_applied_name (for computational layers).
    - layer_passes_total == len(same_layer_operations).
    - Pass numbering within group is contiguous {1..N}.

    Validates cross-layer:
    - Parameter sharing rule: layers with same (func_applied_name,
      sorted(parent_param_barcodes)) must share layer_label_no_pass.
    - Equivalence group consistency: all members of a same_layer_operations
      group belong to the same ModelLog.equivalent_operations set.

    Note: subgraph-level adjacency (Rule 3 from loop_detection.py) cannot
    be verified post-hoc from metadata alone.
    """
    name = "loop_detection"
    label_set = set(ml.layer_labels)

    # Build same-layer groups from the authoritative same_layer_operations lists
    # Key: frozenset of labels, Value: list of LayerPassLogs in the group
    groups_seen: dict[frozenset, list] = {}

    for lpl in ml.layer_list:
        slo = lpl.same_layer_operations
        if not slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' has empty same_layer_operations",
            )

        # All members in same_layer_operations must exist
        for member in slo:
            if member not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' same_layer_operations contains "
                    f"'{member}' not in layer_labels",
                )

        # Self-inclusion
        if lpl.layer_label not in slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' not in its own same_layer_operations",
            )

        # Symmetry: all members agree on the group
        for member_label in slo:
            member = ml[member_label]
            if set(member.same_layer_operations) != set(slo):
                raise MetadataInvariantError(
                    name,
                    f"Asymmetric same_layer_operations: '{lpl.layer_label}' has "
                    f"{sorted(slo)} but '{member_label}' has "
                    f"{sorted(member.same_layer_operations)}",
                )

        # All members share layer_label_no_pass
        for member_label in slo:
            member = ml[member_label]
            if member.layer_label_no_pass != lpl.layer_label_no_pass:
                raise MetadataInvariantError(
                    name,
                    f"same_layer_operations inconsistency: '{lpl.layer_label}' "
                    f"(no_pass='{lpl.layer_label_no_pass}') and '{member_label}' "
                    f"(no_pass='{member.layer_label_no_pass}') differ",
                )

        # All members share operation_equivalence_type
        for member_label in slo:
            member = ml[member_label]
            if member.operation_equivalence_type != lpl.operation_equivalence_type:
                raise MetadataInvariantError(
                    name,
                    f"same_layer_operations type mismatch: '{lpl.layer_label}' "
                    f"type='{lpl.operation_equivalence_type}' vs '{member_label}' "
                    f"type='{member.operation_equivalence_type}'",
                )

        # All members share func_applied_name (for computational layers)
        if not (lpl.is_input_layer or lpl.is_buffer_layer or lpl.is_output_layer):
            for member_label in slo:
                member = ml[member_label]
                if member.func_applied_name != lpl.func_applied_name:
                    raise MetadataInvariantError(
                        name,
                        f"same_layer_operations func mismatch: '{lpl.layer_label}' "
                        f"func='{lpl.func_applied_name}' vs '{member_label}' "
                        f"func='{member.func_applied_name}'",
                    )

        # layer_passes_total == len(same_layer_operations)
        if lpl.layer_passes_total != len(slo):
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}': layer_passes_total={lpl.layer_passes_total} "
                f"!= len(same_layer_operations)={len(slo)}",
            )

        # Pass numbering: unique {1..N}
        group_key = frozenset(slo)
        if group_key not in groups_seen:
            pass_nums = []
            for member_label in slo:
                member = ml[member_label]
                pass_nums.append(member.pass_num)
            expected = set(range(1, len(slo) + 1))
            actual = set(pass_nums)
            if actual != expected:
                raise MetadataInvariantError(
                    name,
                    f"Pass numbering for group {sorted(slo)}: expected {expected}, got {actual}",
                )
            groups_seen[group_key] = slo

    # Rule 1: Parameter sharing invariant.
    # Layers with the same func_applied_name AND identical sorted(parent_param_barcodes)
    # must share the same layer_label_no_pass (they are the same "layer" across
    # different passes).  The func_applied_name is included in the grouping key
    # because different operations (e.g., isinf, expand, nantonum) can consume
    # the same parameter tensor without being the same logical layer.
    param_groups: dict[tuple, list] = defaultdict(list)
    for lpl in ml.layer_list:
        if lpl.computed_with_params and lpl.parent_param_barcodes:
            key = (lpl.func_applied_name, tuple(sorted(lpl.parent_param_barcodes)))
            param_groups[key].append(lpl)

    for param_key, layers in param_groups.items():
        if len(layers) > 1:
            no_pass_labels = {lpl.layer_label_no_pass for lpl in layers}
            if len(no_pass_labels) > 1:
                raise MetadataInvariantError(
                    name,
                    f"Param sharing violation: layers with same param barcodes "
                    f"{param_key} have different layer_label_no_pass: {no_pass_labels}",
                )

    # Equivalence group ↔ same_layer consistency: all members of a
    # same_layer_operations group must belong to the same equivalence set.
    # Note: ModelLog.equivalent_operations keys use the pre-module-suffix type
    # (from loop_detection), while per-layer operation_equivalence_type has
    # a module suffix appended by control_flow.py. So we check group membership
    # consistency, not exact key matching.
    no_pass_to_equiv_key: dict[str, str] = {}
    for eq_type, equiv_set in ml.equivalent_operations.items():
        for label in equiv_set:
            no_pass_to_equiv_key[label] = eq_type

    for group_key in groups_seen:
        slo = list(group_key)
        if len(slo) <= 1:
            continue
        # All members of a same-layer group should be in the same equivalence set
        equiv_keys = set()
        for member_label in slo:
            member = ml[member_label]
            no_pass = member.layer_label_no_pass
            if no_pass in no_pass_to_equiv_key:
                equiv_keys.add(no_pass_to_equiv_key[no_pass])
        if len(equiv_keys) > 1:
            raise MetadataInvariantError(
                name,
                f"same_layer_operations group {sorted(slo)} spans multiple "
                f"equivalence types: {equiv_keys}",
            )

    # Note: subgraph-level adjacency (Rule 3) is verified during BFS in
    # loop_detection.py and cannot be reconstructed from post-hoc data.
    # Non-param multi-pass groups may be the ONLY multi-pass group in a model
    # (e.g., param-free loops like repeated addition), so we cannot require
    # connection to other multi-pass layers.  The checks above (func identity,
    # equiv type, pass numbering, symmetry, param sharing) are sufficient.


# ---------------------------------------------------------------------------
# O. Distance / reachability invariants
# ---------------------------------------------------------------------------


def _check_distance_invariants(ml: "ModelLog") -> None:
    """Check O: distance and reachability invariants.

    Only runs when ``mark_input_output_distances`` was enabled during logging.

    Validates:
    - min_distance <= max_distance for both input and output distances.
    - Input layers have distance_from_input == 0.
    - Output layers have distance_from_output == 0.
    - has_input_ancestor <-> input_ancestors is non-empty.
    - is_output_ancestor <-> output_descendents is non-empty.
    - input_ancestors subset of input_layers; output_descendents subset of
      output_layers.
    """
    if not ml.mark_input_output_distances:
        return

    name = "distance_invariants"
    input_set = set(ml.input_layers)
    output_set = set(ml.output_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # min <= max for input distances
        if lpl.min_distance_from_input is not None and lpl.max_distance_from_input is not None:
            if lpl.min_distance_from_input > lpl.max_distance_from_input:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_from_input="
                    f"{lpl.min_distance_from_input} > max={lpl.max_distance_from_input}",
                )

        # min <= max for output distances
        if lpl.min_distance_from_output is not None and lpl.max_distance_from_output is not None:
            if lpl.min_distance_from_output > lpl.max_distance_from_output:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_from_output="
                    f"{lpl.min_distance_from_output} > max={lpl.max_distance_from_output}",
                )

        # Input layers: distance from input == 0
        if label in input_set:
            if lpl.min_distance_from_input != 0 or lpl.max_distance_from_input != 0:
                raise MetadataInvariantError(
                    name,
                    f"Input layer '{label}': distance_from_input should be 0, got "
                    f"min={lpl.min_distance_from_input}, max={lpl.max_distance_from_input}",
                )

        # Output layers: distance from output == 0
        if label in output_set:
            if lpl.min_distance_from_output != 0 or lpl.max_distance_from_output != 0:
                raise MetadataInvariantError(
                    name,
                    f"Output layer '{label}': distance_from_output should be 0, got "
                    f"min={lpl.min_distance_from_output}, max={lpl.max_distance_from_output}",
                )

        # has_input_ancestor ↔ input_ancestors non-empty
        has_ancestors = len(lpl.input_ancestors) > 0
        if lpl.has_input_ancestor != has_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_input_ancestor={lpl.has_input_ancestor} but "
                f"len(input_ancestors)={len(lpl.input_ancestors)}",
            )

        # is_output_ancestor ↔ output_descendents non-empty
        has_descendents = len(lpl.output_descendents) > 0
        if lpl.is_output_ancestor != has_descendents:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': is_output_ancestor={lpl.is_output_ancestor} but "
                f"len(output_descendents)={len(lpl.output_descendents)}",
            )

        # input_ancestors subset of input_layers
        extra_ancestors = lpl.input_ancestors - input_set
        if extra_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': input_ancestors contains labels not in "
                f"input_layers: {extra_ancestors}",
            )

        # output_descendents subset of output_layers
        extra_desc = lpl.output_descendents - output_set
        if extra_desc:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': output_descendents contains labels not in "
                f"output_layers: {extra_desc}",
            )


# ---------------------------------------------------------------------------
# P. Graph connectivity invariants
# ---------------------------------------------------------------------------


def _check_graph_connectivity(ml: "ModelLog") -> None:
    """Check P: graph connectivity invariants.

    Validates:
    - Every non-input, non-buffer, non-internally-initialized, non-output
      layer has at least one parent (no dangling computational nodes).
    - orphan_layers (removed during postprocessing) do NOT appear in the
      active layer_labels (they were pruned from the graph).
    """
    name = "graph_connectivity"
    label_set = set(ml.layer_labels)
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Non-input, non-buffer, non-internally-initialized layers must have parents
        if (
            label not in input_set
            and label not in buffer_set
            and not lpl.initialized_inside_model
            and not lpl.is_output_layer
            and len(lpl.parent_layers) == 0
        ):
            raise MetadataInvariantError(
                name,
                f"Layer '{label}' has no parents but is not input, buffer, "
                f"internally initialized, or output",
            )

    # orphan_layers is a subset of all known labels (pre-removal)
    orphan_set = set(ml.orphan_layers)
    # Orphans should NOT appear in the active layer_list (they were removed)
    orphan_in_list = orphan_set & label_set
    if orphan_in_list:
        raise MetadataInvariantError(
            name,
            f"orphan_layers contains labels still in layer_labels: {orphan_in_list}",
        )


# ---------------------------------------------------------------------------
# Q. Module containment logical consistency
# ---------------------------------------------------------------------------


def _check_module_containment_logic(ml: "ModelLog") -> None:
    """Check Q: module containment logical consistency.

    Validates:
    - Address tree is acyclic (walking address_parent chain reaches None
      without revisiting a node).
    - Root module 'self' has address_depth == 0; others have
      address_depth == addr.count('.') + 1.
    - Per-layer containing_modules_origin_nested:
      - Last element matches containing_module_origin.
      - Nesting depth is monotonically increasing through the path.
    """
    name = "module_containment_logic"
    mod_accessor = ml.modules

    # Build set of known module addresses
    known_addrs = set()
    for mod_log in mod_accessor:
        known_addrs.add(mod_log.address)

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Address tree acyclicity: walk address_parent to root
        visited = set()
        current = addr
        while current is not None:
            if current in visited:
                raise MetadataInvariantError(
                    name,
                    f"Cycle in address_parent chain starting from '{addr}': revisited '{current}'",
                )
            visited.add(current)
            try:
                parent_mod: ModuleLog = mod_accessor[current]  # type: ignore[assignment]
            except (KeyError, IndexError):
                break
            current = parent_mod.address_parent

        # Address depth consistency
        if addr == "self":
            if mod_log.address_depth != 0:
                raise MetadataInvariantError(
                    name,
                    f"Root module 'self' has address_depth={mod_log.address_depth}, expected 0",
                )
        else:
            expected_depth = addr.count(".") + 1
            if mod_log.address_depth != expected_depth:
                raise MetadataInvariantError(
                    name,
                    f"ModuleLog '{addr}': address_depth={mod_log.address_depth} "
                    f"!= expected {expected_depth} (addr.count('.')+1)",
                )

    # Per-layer: containing_modules_origin_nested path validity
    # Format after postprocessing: list of "addr:pass" strings, ordered from
    # outermost enclosing submodule to innermost. Does NOT include "self".
    for lpl in ml.layer_list:
        nested = lpl.containing_modules_origin_nested
        if not nested:
            continue

        # Leaf consistency: last element matches containing_module_origin
        if lpl.containing_module_origin is not None:
            if nested[-1] != lpl.containing_module_origin:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': last nested module '{nested[-1]}' "
                    f"!= containing_module_origin '{lpl.containing_module_origin}'",
                )

        # Path validity: each element's address should be deeper than the
        # previous (basic nesting depth ordering).  We do NOT check that each
        # element's address_parent chains to the previous — the nested path
        # records execution-time module context which can diverge from the
        # static address tree for complex models (e.g., ModuleList indexing).
        for i in range(1, len(nested)):
            child_addr = nested[i].split(":")[0] if ":" in nested[i] else nested[i]
            parent_addr = nested[i - 1].split(":")[0] if ":" in nested[i - 1] else nested[i - 1]
            # At minimum, the child should be deeper in the module tree
            if child_addr.count(".") < parent_addr.count("."):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': nested path depth decreases at "
                    f"index {i}: '{child_addr}' is shallower than '{parent_addr}'",
                )


# ---------------------------------------------------------------------------
# R. Lookup key bidirectionality
# ---------------------------------------------------------------------------


def _check_lookup_key_consistency(ml: "ModelLog") -> None:
    """Check R: lookup key bidirectional consistency.

    Validates:
    - _lookup_keys_to_layer_num_dict (forward: key->num) and
      _layer_num_to_lookup_keys_dict (reverse: num->[keys]) are consistent:
      forward[key]=num implies key in reverse[num], and vice versa.
    - _raw_to_final_layer_labels and _final_to_raw_layer_labels are inverse
      bijections.
    - All final labels in the raw->final map exist in layer_labels.
    """
    name = "lookup_key_consistency"

    # _lookup_keys_to_layer_num_dict maps key→num (last assigned wins).
    # _layer_num_to_lookup_keys_dict maps num→[keys] (accumulates all assignments).
    # Forward → reverse: for every (key, num) in forward, key must be in reverse[num].
    fwd = ml._lookup_keys_to_layer_num_dict
    rev = ml._layer_num_to_lookup_keys_dict

    for key, num in fwd.items():
        if num not in rev:
            raise MetadataInvariantError(
                name,
                f"_lookup_keys_to_layer_num_dict['{key}']={num} but "
                f"{num} not in _layer_num_to_lookup_keys_dict",
            )
        if key not in rev[num]:
            raise MetadataInvariantError(
                name,
                f"_lookup_keys_to_layer_num_dict['{key}']={num} but "
                f"'{key}' not in _layer_num_to_lookup_keys_dict[{num}]",
            )

    # Reverse → forward: every key in reverse must exist in forward (but may
    # point to a different num if the key was reassigned to a later layer).
    for num, keys in rev.items():
        for key in keys:
            if key not in fwd:
                raise MetadataInvariantError(
                    name,
                    f"_layer_num_to_lookup_keys_dict[{num}] has '{key}' but "
                    f"'{key}' not in _lookup_keys_to_layer_num_dict",
                )

    # _raw_to_final_layer_labels ↔ _final_to_raw_layer_labels
    raw_fwd = ml._raw_to_final_layer_labels
    raw_rev = ml._final_to_raw_layer_labels

    for raw, final in raw_fwd.items():
        if final not in raw_rev:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels['{raw}']='{final}' but "
                f"'{final}' not in _final_to_raw_layer_labels",
            )
        if raw_rev[final] != raw:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels['{raw}']='{final}' but "
                f"_final_to_raw_layer_labels['{final}']='{raw_rev[final]}'",
            )

    for final, raw in raw_rev.items():
        if raw not in raw_fwd:
            raise MetadataInvariantError(
                name,
                f"_final_to_raw_layer_labels['{final}']='{raw}' but "
                f"'{raw}' not in _raw_to_final_layer_labels",
            )

    # All final labels are valid layer labels
    label_set = set(ml.layer_labels)
    for final in raw_fwd.values():
        if final not in label_set:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels maps to '{final}' which is not in layer_labels",
            )
