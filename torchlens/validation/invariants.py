"""Metadata invariant checks for ``ModelLog`` and its sub-objects.

Single entry point: ``check_metadata_invariants(model_log)`` runs all checks
and raises ``MetadataInvariantError`` on the first failure.

**Phase 1 -- Structural invariants:**
  A. ModelLog self-consistency (counts, timing, label uniqueness)
  B. Special layer lists match per-layer boolean flags
  C. Graph topology (parent-child bidirectionality, boolean flag consistency)
  D. LayerPassLog field consistency (shape, dtype, pass numbering, nesting)
  E. Recurrence / loop invariants (is_recurrent, pass dicts)
  F. Branching invariants (is_branching)
  F2. Conditional metadata invariants (13 conditional consistency checks)
  G. LayerPassLog <-> LayerLog cross-references (pass numbering, back-pointers)
  H. Module <-> Layer containment (all_layers, module pass layers, reverse check)
  I. Module hierarchy (address parent-child bidirectionality, pass consistency)
  J. Param cross-references (ParamLog -> layer, uses_params flag)
  K. Buffer cross-references (buffer_layers list, BufferLog module references)
  L. Equivalence group symmetry (equivalent_operations labels are valid)

**Phase 2 -- Semantic invariants:**
  M. Graph ordering (creation_order uniqueness/monotonicity, topological order, no raw labels)
  N. Loop detection invariants (recurrent_group symmetry, func identity, param sharing, pass numbering)
  O. Distance / reachability (min <= max, input/output layer distances == 0, ancestor/descendent consistency)
  P. Graph connectivity (non-input non-buffer layers have parents, orphans removed)
  Q. Module containment logic (address acyclicity, depth consistency, nested path ordering)
  R. Lookup key bidirectionality (forward/reverse dicts, raw/final label maps)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors._base import ValidationError

if TYPE_CHECKING:
    from ..data_classes.layer_log import LayerLog
    from ..data_classes.layer_pass_log import LayerPassLog
    from ..data_classes.model_log import ModelLog
    from ..data_classes.module_log import ModuleLog


class MetadataInvariantError(ValidationError, ValueError):
    """Raised when a metadata invariant check fails.

    Embeds the check name (e.g., ``"graph_topology"``) in the message prefix
    and stores it as an attribute for programmatic inspection in tests.
    """

    def __init__(self, check_name: str, message: str) -> None:
        """Initialize a metadata invariant failure.

        Parameters
        ----------
        check_name:
            Invariant check group name.
        message:
            Human-readable failure detail.
        """

        super().__init__(f"[{check_name}] {message}")
        self.check_name = check_name


@dataclass(frozen=True)
class InvariantResult:
    """Structured result for an individual metadata invariant.

    Attributes
    ----------
    name:
        Invariant check name.
    passed:
        Whether the invariant passed.
    message:
        Optional diagnostic message.
    """

    name: str
    passed: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_metadata_invariants(model_log: "ModelLog") -> bool:
    """Run all metadata invariant checks on a completed ``ModelLog``.

    Checks run in dependency order: Phase 1 structural checks first, then
    Phase 2 semantic checks. Raises ``MetadataInvariantError`` on the first
    failure, so later checks can assume earlier ones passed.

    Parameters
    ----------
    model_log:
        Postprocessed model log to validate.

    Returns
    -------
    bool
        ``True`` if all invariants pass.
    """
    # --- Phase 1: structural invariants (A-L) ---
    _check_model_log_self_consistency(model_log)  # A
    _check_special_layer_lists(model_log)  # B
    _check_graph_topology(model_log)  # C
    _check_layer_pass_log_fields(model_log)  # D
    _check_recurrence_invariants(model_log)  # E
    _check_branching_invariants(model_log)  # F
    _check_conditional_invariants(model_log)  # F2
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
    check_func_call_id_invariant(model_log)  # S
    return True


def check_func_call_id_invariant(model_log: "ModelLog") -> InvariantResult:
    """Invariant S: func_call_id consistency.

    For intervention-ready logs, every non-synthetic function output must have a
    ``func_call_id``. Outputs from the same call must agree on function identity,
    call stack, captured templates, and container spec; each output in the group
    must have a unique ``output_path``; and same-call groups must not span
    incompatible pass labels. Synthetic input, output, and buffer nodes are
    exempt.

    Parameters
    ----------
    model_log:
        Postprocessed model log to validate.

    Returns
    -------
    InvariantResult
        Passing result when no inconsistency is found.
    """

    name = "func_call_id_consistency"
    if not getattr(model_log, "intervention_ready", False):
        return InvariantResult(name=name, passed=True)

    groups: dict[int, list["LayerPassLog"]] = defaultdict(list)
    for layer in model_log.layer_list:
        if _is_func_call_id_exempt(layer):
            continue
        if layer.func_call_id is None:
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} has no func_call_id",
            )
        groups[layer.func_call_id].append(layer)

    for func_call_id, group in groups.items():
        reference = group[0]
        expected_signature = _func_call_group_signature(reference)
        output_paths = []
        pass_nums = set()
        no_pass_labels = set()
        for layer in group:
            if _func_call_group_signature(layer) != expected_signature:
                raise MetadataInvariantError(
                    name,
                    f"func_call_id {func_call_id} has incompatible call metadata",
                )
            output_path = tuple(getattr(layer, "output_path", ()) or ())
            if output_path in output_paths:
                raise MetadataInvariantError(
                    name,
                    f"func_call_id {func_call_id} has duplicate output_path {output_path!r}",
                )
            output_paths.append(output_path)
            pass_nums.add(layer.pass_num)
            no_pass_labels.add(layer.layer_label_no_pass)
        if len(pass_nums) > 1 and len(no_pass_labels) > 1:
            raise MetadataInvariantError(
                name,
                f"func_call_id {func_call_id} spans incompatible pass labels",
            )
    return InvariantResult(name=name, passed=True)


def _is_func_call_id_exempt(layer: "LayerPassLog") -> bool:
    """Return whether a layer is exempt from Invariant S.

    Parameters
    ----------
    layer:
        Layer pass to classify.

    Returns
    -------
    bool
        Whether the layer is synthetic input/output/buffer metadata.
    """

    if layer.is_input_layer or layer.is_output_layer or layer.is_buffer_layer:
        return True
    func_applied = getattr(layer, "func_applied", None)
    func_name = str(getattr(layer, "func_name", "")).lower()
    return func_applied in {"input", "output", "buffer"} or func_name in {
        "input",
        "output",
        "buffer",
        "none",
    }


def _func_call_group_signature(layer: "LayerPassLog") -> tuple[object, ...]:
    """Return comparable same-call metadata for Invariant S.

    Parameters
    ----------
    layer:
        Layer pass to summarize.

    Returns
    -------
    tuple[object, ...]
        Stable comparison tuple.
    """

    return (
        layer.func_name,
        tuple(repr(location) for location in (layer.func_call_stack or ())),
        repr(layer.captured_arg_template),
        repr(layer.captured_kwarg_template),
        repr(layer.container_spec),
    )


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
    if ml.time_total < 0:
        raise MetadataInvariantError(name, f"time_total={ml.time_total} < 0")
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
    ("internally_initialized_layers", "is_internally_initialized"),
    ("internally_terminated_layers", "is_internally_terminated"),
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
    - has_children/has_parents/has_siblings/has_co_parents flags match actual counts.
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
        # Note: has_children/has_parents/has_siblings/has_co_parents are set during
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
        if lpl.has_co_parents != (len(lpl.co_parent_layers) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_co_parents={lpl.has_co_parents} but "
                f"len(co_parent_layers)={len(lpl.co_parent_layers)}",
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
    - Saved tensor shape/dtype match actual activation (when saved).
    - Pass numbering: pass_num >= 1, num_passes >= pass_num.
    - Computational layers have callable func_applied and non-empty func_name.
    - operation_num >= 1 for non-input/non-buffer layers.
    - module_nesting_depth matches len(containing_modules).
    - Label format: pass-qualified label has ':' iff multi-pass; no-pass label never has ':'.
    """
    name = "layer_pass_log_fields"

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Tensor shape/dtype consistency when activations are saved
        if lpl.has_saved_activations and lpl.activation is not None:
            actual_shape = tuple(lpl.activation.shape)
            if lpl.tensor_shape != actual_shape:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: tensor_shape={lpl.tensor_shape} != "
                    f"actual shape={actual_shape}",
                )
            if lpl.tensor_dtype != lpl.activation.dtype:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: tensor_dtype={lpl.tensor_dtype} != "
                    f"actual dtype={lpl.activation.dtype}",
                )

        # Pass numbering
        if lpl.pass_num < 1:
            raise MetadataInvariantError(name, f"Layer {label}: pass_num={lpl.pass_num} < 1")
        if lpl.num_passes < lpl.pass_num:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: num_passes={lpl.num_passes} < pass_num={lpl.pass_num}",
            )

        # Function applied (non-input, non-buffer, non-output layers)
        if not (lpl.is_input_layer or lpl.is_buffer_layer or lpl.is_output_layer):
            if not callable(lpl.func_applied):
                raise MetadataInvariantError(name, f"Layer {label}: func_applied is not callable")
            if not lpl.func_name:
                raise MetadataInvariantError(name, f"Layer {label}: func_name is empty")

        # Operation numbering (input/buffer/output bookkeeping layers have operation_num=0)
        if not (lpl.is_input_layer or lpl.is_buffer_layer or lpl.is_output_layer):
            if lpl.operation_num is not None and lpl.operation_num < 1:
                raise MetadataInvariantError(
                    name, f"Layer {label}: operation_num={lpl.operation_num} < 1"
                )
        if lpl.creation_order < 1:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: creation_order={lpl.creation_order} < 1",
            )

        # Module nesting depth
        if lpl.module_nesting_depth != len(lpl.containing_modules):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_nesting_depth={lpl.module_nesting_depth} != "
                f"len(containing_modules)="
                f"{len(lpl.containing_modules)}",
            )

        # Label format: pass-qualified label has ":" iff multi-pass
        if lpl.num_passes > 1 and ":" not in lpl.layer_label_w_pass:
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
    - is_recurrent == True iff any layer has >1 pass.
    - max_recurrent_loops matches the maximum pass count.
    - layer_num_passes keys are valid no-pass labels.
    - LayerLog.passes dict keys are contiguous {1..N}.
    """
    name = "recurrence_invariants"

    any_recurrent = any(v > 1 for v in ml.layer_num_passes.values())
    if ml.is_recurrent != any_recurrent:
        raise MetadataInvariantError(
            name,
            f"is_recurrent={ml.is_recurrent} but any layer has >1 pass = {any_recurrent}",
        )

    if ml.is_recurrent:
        expected_max = max(ml.layer_num_passes.values())
        if ml.max_recurrent_loops != expected_max:
            raise MetadataInvariantError(
                name,
                f"max_recurrent_loops={ml.max_recurrent_loops} != "
                f"max(layer_num_passes)={expected_max}",
            )

    # Per-layer pass consistency: layer_num_passes is keyed by no-pass labels.
    # Validate that each key exists in layer_labels_no_pass, and that the
    # recorded count matches the actual LayerLog.num_passes.
    no_pass_labels = set(ml.layer_labels_no_pass)
    for label_key, num_passes in ml.layer_num_passes.items():
        if label_key not in no_pass_labels:
            raise MetadataInvariantError(
                name,
                f"layer_num_passes key '{label_key}' not in layer_labels_no_pass",
            )
        if label_key in ml.layer_logs:
            actual = ml.layer_logs[label_key].num_passes
            if num_passes != actual:
                raise MetadataInvariantError(
                    name,
                    f"layer_num_passes['{label_key}']={num_passes} != LayerLog.num_passes={actual}",
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
    """Check F: is_branching matches whether any layer has >1 child."""
    name = "branching_invariants"
    any_branching = any(len(lpl.child_layers) > 1 for lpl in ml.layer_list)
    if ml.is_branching != any_branching:
        raise MetadataInvariantError(
            name,
            f"is_branching={ml.is_branching} but any layer has >1 child = {any_branching}",
        )


# ---------------------------------------------------------------------------
# F2. Conditional metadata invariants
# ---------------------------------------------------------------------------


def _fail_conditional_invariant(check_name: str, number: int, message: str) -> None:
    """Raise a numbered conditional metadata invariant failure.

    Parameters
    ----------
    check_name:
        ``MetadataInvariantError.check_name`` value for this check family.
    number:
        Conditional invariant number from the Phase 6 plan.
    message:
        Human-readable failure details.
    """

    raise MetadataInvariantError(check_name, f"Invariant {number}: {message}")


def _strip_pass_suffix(layer_label: str) -> str:
    """Return a layer label without any trailing ``:pass_num`` suffix.

    Parameters
    ----------
    layer_label:
        Layer label that may include a pass suffix.

    Returns
    -------
    str
        Pass-stripped label.
    """

    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return label_parts[0]
    return layer_label


def _get_label_pass_num(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:pass_num`` suffix.

    Returns
    -------
    int
        Parsed pass number, or ``1`` when no suffix is present.
    """

    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return int(label_parts[1])
    return 1


def _append_unique(values: list[str], value: str) -> None:
    """Append ``value`` to ``values`` only if not already present.

    Parameters
    ----------
    values:
        Ordered list being built.
    value:
        Candidate value to append.
    """

    if value not in values:
        values.append(value)


def _is_prefix_stack(prefix: list[tuple[int, str]], full: list[tuple[int, str]]) -> bool:
    """Return whether one conditional branch stack prefixes another.

    Parameters
    ----------
    prefix:
        Candidate prefix stack.
    full:
        Candidate full stack.

    Returns
    -------
    bool
        ``True`` if ``prefix`` matches the first ``len(prefix)`` entries of
        ``full``.
    """

    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def _expected_layer_pass_child_views(
    cond_branch_children_by_cond: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project pass-level child views from the primary conditional structure.

    Parameters
    ----------
    cond_branch_children_by_cond:
        Primary ``cond_id -> branch_kind -> child labels`` mapping on a
        ``LayerPassLog``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views.
    """

    then_children = sorted(
        {
            child_label
            for branch_children in cond_branch_children_by_cond.values()
            for child_label in branch_children.get("then", [])
        }
    )
    elif_children: dict[int, list[str]] = {}
    grouped_elif_children: dict[int, set[str]] = defaultdict(set)
    for branch_children in cond_branch_children_by_cond.values():
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            grouped_elif_children[elif_index].update(child_labels)
    for elif_index, elif_label_set in sorted(grouped_elif_children.items()):
        elif_children[elif_index] = sorted(elif_label_set)

    else_children = sorted(
        {
            child_label
            for branch_children in cond_branch_children_by_cond.values()
            for child_label in branch_children.get("else", [])
        }
    )
    return then_children, elif_children, else_children


def _expected_layer_log_child_views(
    cond_branch_children_by_cond: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project aggregate child views from a ``LayerLog`` primary structure.

    Parameters
    ----------
    cond_branch_children_by_cond:
        Aggregate ``cond_id -> branch_kind -> child labels`` mapping on a
        ``LayerLog``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views preserving first-seen order.
    """

    then_children: list[str] = []
    elif_children: dict[int, list[str]] = {}
    else_children: list[str] = []
    for branch_children in cond_branch_children_by_cond.values():
        for child_label in branch_children.get("then", []):
            _append_unique(then_children, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            expected_children = elif_children.setdefault(elif_index, [])
            for child_label in child_labels:
                _append_unique(expected_children, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique(else_children, child_label)
    return then_children, elif_children, else_children


def _expected_layer_log_child_union(
    layer_log: "LayerLog",
) -> dict[int, dict[str, list[str]]]:
    """Build the expected aggregate ``cond_branch_children_by_cond`` for a ``LayerLog``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being validated.

    Returns
    -------
    dict[int, dict[str, list[str]]]
        Pass-stripped union of every pass-level child list.
    """

    expected_children_by_cond: dict[int, dict[str, list[str]]] = {}
    for pass_num in sorted(layer_log.passes):
        pass_log = layer_log.passes[pass_num]
        for conditional_id, branch_children in pass_log.cond_branch_children_by_cond.items():
            merged_branch_children = expected_children_by_cond.setdefault(conditional_id, {})
            for branch_kind, child_labels in branch_children.items():
                merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
                for child_label in child_labels:
                    _append_unique(merged_child_labels, _strip_pass_suffix(child_label))
    return expected_children_by_cond


def _valid_conditional_child_labels(ml: "ModelLog") -> set[str]:
    """Return the set of valid labels for conditional child references.

    Parameters
    ----------
    ml:
        Model log being validated.

    Returns
    -------
    set[str]
        Union of pass-level labels and aggregate ``LayerLog`` keys.
    """

    return set(ml.layer_labels) | set(ml.layer_logs)


def _check_conditional_invariants(ml: "ModelLog") -> None:
    """Check F2: conditional metadata invariants added in Phase 6.

    Parameters
    ----------
    ml:
        Model log being validated.
    """

    name = "conditional_invariants"
    layer_label_set = set(ml.layer_labels)
    valid_child_labels = _valid_conditional_child_labels(ml)
    event_id_set = {event.id for event in ml.conditional_events}
    branch_context_kinds = {"if_test", "elif_test", "ifexp"}
    wrapped_context_kinds = branch_context_kinds | {"bool_cast"}

    # Invariant 1: conditional_arm_edges ↔ cond_branch_children_by_cond.
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_edges.items():
        for parent_label, child_label in edge_list:
            if parent_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_edges[{(conditional_id, branch_kind)}] references "
                    f"missing parent layer {parent_label!r}",
                )
            parent_layer = ml[parent_label]
            branch_children = parent_layer.cond_branch_children_by_cond.get(conditional_id, {}).get(
                branch_kind, []
            )
            if child_label not in branch_children:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_edges[{(conditional_id, branch_kind)}] includes edge "
                    f"({parent_label!r}, {child_label!r}) but "
                    f"{parent_label}.cond_branch_children_by_cond[{conditional_id}][{branch_kind!r}]="
                    f"{branch_children}",
                )

    for parent_layer in ml.layer_list:
        for conditional_id, branch_children in parent_layer.cond_branch_children_by_cond.items():
            for branch_kind, child_labels in branch_children.items():
                model_edges = ml.conditional_arm_edges.get((conditional_id, branch_kind), [])
                for child_label in child_labels:
                    if (parent_layer.layer_label, child_label) not in model_edges:
                        _fail_conditional_invariant(
                            name,
                            1,
                            f"{parent_layer.layer_label}.cond_branch_children_by_cond"
                            f"[{conditional_id}][{branch_kind!r}] includes {child_label!r} "
                            f"but conditional_arm_edges[{(conditional_id, branch_kind)}]={model_edges}",
                        )

    # Invariant 2: per-layer derived views are exact projections of the primary structures.
    for layer in ml.layer_list:
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_pass_child_views(layer.cond_branch_children_by_cond)
        )
        if layer.cond_branch_then_children != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.cond_branch_then_children={layer.cond_branch_then_children} != "
                f"expected projection {expected_then_children}",
            )
        if layer.cond_branch_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.cond_branch_elif_children={layer.cond_branch_elif_children} != "
                f"expected projection {expected_elif_children}",
            )
        if layer.cond_branch_else_children != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.cond_branch_else_children={layer.cond_branch_else_children} != "
                f"expected projection {expected_else_children}",
            )

    for layer_log in ml.layer_logs.values():
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_log_child_views(layer_log.cond_branch_children_by_cond)
        )
        if layer_log.cond_branch_then_children != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"LayerLog {layer_log.layer_label}.cond_branch_then_children="
                f"{layer_log.cond_branch_then_children} != expected projection "
                f"{expected_then_children}",
            )
        if layer_log.cond_branch_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"LayerLog {layer_log.layer_label}.cond_branch_elif_children="
                f"{layer_log.cond_branch_elif_children} != expected projection "
                f"{expected_elif_children}",
            )
        if layer_log.cond_branch_else_children != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"LayerLog {layer_log.layer_label}.cond_branch_else_children="
                f"{layer_log.cond_branch_else_children} != expected projection "
                f"{expected_else_children}",
            )

    # Invariant 3: every child label in conditional child views exists in the log.
    for layer in ml.layer_list:
        for field_name, child_labels in (
            ("cond_branch_start_children", layer.cond_branch_start_children),
            ("cond_branch_then_children", layer.cond_branch_then_children),
            ("cond_branch_else_children", layer.cond_branch_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.{field_name} contains missing child label "
                        f"{child_label!r}",
                    )
        for elif_index, child_labels in layer.cond_branch_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.cond_branch_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for layer_log in ml.layer_logs.values():
        for field_name, child_labels in (
            ("cond_branch_start_children", layer_log.cond_branch_start_children),
            ("cond_branch_then_children", layer_log.cond_branch_then_children),
            ("cond_branch_else_children", layer_log.cond_branch_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"LayerLog {layer_log.layer_label}.{field_name} contains missing child "
                        f"label {child_label!r}",
                    )
        for elif_index, child_labels in layer_log.cond_branch_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"LayerLog {layer_log.layer_label}.cond_branch_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for parent_label, child_label in ml.conditional_branch_edges:
        if child_label not in valid_child_labels:
            _fail_conditional_invariant(
                name,
                3,
                f"ModelLog.conditional_branch_edges contains missing child label {child_label!r} "
                f"for parent {parent_label!r}",
            )
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_edges.items():
        for parent_label, child_label in edge_list:
            if child_label not in valid_child_labels:
                _fail_conditional_invariant(
                    name,
                    3,
                    f"ModelLog.conditional_arm_edges contains missing child label "
                    f"{child_label!r} for edge {(conditional_id, branch_kind, parent_label)}",
                )

    # Invariant 4: bool classification fields are mutually consistent.
    for layer in ml.layer_list:
        expected_is_branch = layer.bool_context_kind in branch_context_kinds
        if layer.bool_is_branch != expected_is_branch:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has bool_is_branch={layer.bool_is_branch} but "
                f"bool_context_kind={layer.bool_context_kind!r}",
            )
        if layer.bool_is_branch and layer.bool_conditional_id is None:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has bool_is_branch=True but bool_conditional_id is None",
            )
        if layer.bool_context_kind is not None and not layer.is_terminal_bool_layer:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has bool_context_kind={layer.bool_context_kind!r} but "
                f"is_terminal_bool_layer=False",
            )
        if (
            layer.bool_wrapper_kind is not None
            and layer.bool_context_kind not in wrapped_context_kinds
        ):
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has bool_wrapper_kind={layer.bool_wrapper_kind!r} but "
                f"bool_context_kind={layer.bool_context_kind!r}",
            )

    # Invariant 5: every referenced cond_id corresponds to a ConditionalEvent.
    referenced_cond_ids: set[int] = set()
    for layer in ml.layer_list:
        referenced_cond_ids.update(
            conditional_id for conditional_id, _ in layer.conditional_branch_stack
        )
        if layer.bool_conditional_id is not None:
            referenced_cond_ids.add(layer.bool_conditional_id)
        referenced_cond_ids.update(layer.cond_branch_children_by_cond)
    for layer_log in ml.layer_logs.values():
        referenced_cond_ids.update(
            conditional_id
            for branch_stack in layer_log.conditional_branch_stacks
            for conditional_id, _ in branch_stack
        )
        referenced_cond_ids.update(layer_log.cond_branch_children_by_cond)
    referenced_cond_ids.update(conditional_id for conditional_id, _ in ml.conditional_arm_edges)

    for conditional_id in sorted(referenced_cond_ids):
        if conditional_id not in event_id_set:
            _fail_conditional_invariant(
                name,
                5,
                f"Referenced cond_id {conditional_id} has no matching ConditionalEvent.id "
                f"in ModelLog.conditional_events",
            )

    # Invariant 6: parent->child stacks are monotone by prefix relation.
    for parent_layer in ml.layer_list:
        for child_label in parent_layer.child_layers:
            child_layer = ml[child_label]
            if parent_layer.pass_num != child_layer.pass_num:
                continue
            if parent_layer.conditional_branch_stack == child_layer.conditional_branch_stack:
                continue
            if _is_prefix_stack(
                parent_layer.conditional_branch_stack, child_layer.conditional_branch_stack
            ):
                continue
            if _is_prefix_stack(
                child_layer.conditional_branch_stack, parent_layer.conditional_branch_stack
            ):
                continue
            _fail_conditional_invariant(
                name,
                6,
                f"Edge ({parent_layer.layer_label!r}, {child_label!r}) has non-prefix "
                f"conditional stacks parent={parent_layer.conditional_branch_stack} "
                f"child={child_layer.conditional_branch_stack}",
            )

    # Invariant 7: elif keys are contiguous on ConditionalEvent.
    for event in ml.conditional_events:
        for field_name, mapping in (
            ("branch_ranges", event.branch_ranges),
            ("branch_test_spans", event.branch_test_spans),
        ):
            elif_indices = sorted(
                int(key.split("_", 1)[1]) for key in mapping if key.startswith("elif_")
            )
            if elif_indices != list(range(1, len(elif_indices) + 1)):
                _fail_conditional_invariant(
                    name,
                    7,
                    f"ConditionalEvent id={event.id} {field_name} has non-contiguous elif keys "
                    f"{elif_indices}",
                )

    # Invariant 8: ConditionalEvent.bool_layers back-reference to the event id.
    for event in ml.conditional_events:
        for bool_label in event.bool_layers:
            if bool_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers contains missing label "
                    f"{bool_label!r}",
                )
            bool_layer = ml[bool_label]
            if bool_layer.bool_conditional_id != event.id:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers includes {bool_label!r} but "
                    f"{bool_label}.bool_conditional_id={bool_layer.bool_conditional_id}",
                )

    # Invariant 9: LayerLog conditional aggregate views match pass-level data.
    for layer_log in ml.layer_logs.values():
        expected_stack_order: list[list[tuple[int, str]]] = []
        expected_stack_passes: dict[tuple[tuple[int, str], ...], list[int]] = {}
        for pass_num in sorted(layer_log.passes):
            pass_log = layer_log.passes[pass_num]
            stack_signature = tuple(pass_log.conditional_branch_stack)
            if stack_signature not in expected_stack_passes:
                expected_stack_order.append(list(pass_log.conditional_branch_stack))
                expected_stack_passes[stack_signature] = []
            expected_stack_passes[stack_signature].append(pass_num)

        if layer_log.conditional_branch_stacks != expected_stack_order:
            _fail_conditional_invariant(
                name,
                9,
                f"LayerLog {layer_log.layer_label}.conditional_branch_stacks="
                f"{layer_log.conditional_branch_stacks} != expected {expected_stack_order}",
            )
        if layer_log.conditional_branch_stack_passes != expected_stack_passes:
            _fail_conditional_invariant(
                name,
                9,
                f"LayerLog {layer_log.layer_label}.conditional_branch_stack_passes="
                f"{layer_log.conditional_branch_stack_passes} != expected "
                f"{expected_stack_passes}",
            )

    # Invariant 10: conditional_edge_passes exactly matches unrolled arm edges.
    actual_unrolled_edges: set[tuple[str, str, int, str, int]] = set()
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_edges.items():
        for parent_label, child_label in edge_list:
            actual_unrolled_edges.add(
                (
                    _strip_pass_suffix(parent_label),
                    _strip_pass_suffix(child_label),
                    conditional_id,
                    branch_kind,
                    _get_label_pass_num(child_label),
                )
            )

    for edge_key, pass_nums in ml.conditional_edge_passes.items():
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        if pass_nums != sorted(pass_nums) or len(pass_nums) != len(set(pass_nums)):
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_edge_passes[{edge_key}] has unsorted or duplicate pass list "
                f"{pass_nums}",
            )
        for pass_num in pass_nums:
            actual_edge = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
                pass_num,
            )
            if actual_edge not in actual_unrolled_edges:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_passes[{edge_key}] includes pass {pass_num} but "
                    f"conditional_arm_edges has no matching unrolled edge",
                )

    for actual_edge in sorted(actual_unrolled_edges):
        parent_no_pass, child_no_pass, conditional_id, branch_kind, pass_num = actual_edge
        edge_key = (parent_no_pass, child_no_pass, conditional_id, branch_kind)
        if pass_num not in ml.conditional_edge_passes.get(edge_key, []):
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_arm_edges implies unrolled edge {actual_edge} but "
                f"conditional_edge_passes[{edge_key}]={ml.conditional_edge_passes.get(edge_key)}",
            )

    # Invariant 11: no transient _bool_conditional_key remains after step 5c.
    for layer in ml.layer_list:
        if hasattr(layer, "_bool_conditional_key"):
            _fail_conditional_invariant(
                name,
                11,
                f"{layer.layer_label} still has transient _bool_conditional_key attribute",
            )

    # Invariant 12: LayerLog cond_branch_children_by_cond is the exact pass union.
    for layer_log in ml.layer_logs.values():
        expected_children_by_cond = _expected_layer_log_child_union(layer_log)
        if layer_log.cond_branch_children_by_cond != expected_children_by_cond:
            _fail_conditional_invariant(
                name,
                12,
                f"LayerLog {layer_log.layer_label}.cond_branch_children_by_cond="
                f"{layer_log.cond_branch_children_by_cond} != expected pass union "
                f"{expected_children_by_cond}",
            )

    # Invariant 13: legacy IF-view conditional_branch_edges ↔ start-children.
    for parent_label, bool_label in ml.conditional_branch_edges:
        if parent_label not in layer_label_set:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges references missing parent layer {parent_label!r}",
            )
        parent_layer = ml[parent_label]
        if bool_label not in parent_layer.cond_branch_start_children:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges includes ({parent_label!r}, {bool_label!r}) but "
                f"{parent_label}.cond_branch_start_children={parent_layer.cond_branch_start_children}",
            )

    for parent_layer in ml.layer_list:
        for bool_label in parent_layer.cond_branch_start_children:
            if (parent_layer.layer_label, bool_label) not in ml.conditional_branch_edges:
                _fail_conditional_invariant(
                    name,
                    13,
                    f"{parent_layer.layer_label}.cond_branch_start_children includes "
                    f"{bool_label!r} but conditional_branch_edges={ml.conditional_branch_edges}",
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
    - Reverse: each layer's containing_module points to a valid module
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

    # Reverse check: layer's containing_module exists in modules
    for lpl in ml.layer_list:
        cmo = lpl.containing_module
        if cmo:
            # containing_module may include pass suffix (e.g. 'fc:1')
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            try:
                mod = mod_accessor[cmo_addr]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' containing_module='{cmo}' "
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
    - ParamLog.used_by_layers labels are valid layer labels.
    - uses_params == True implies parent_param_logs is non-empty.
    - layers_with_params values are valid layer labels.
    """
    name = "param_xrefs"
    label_set = set(ml.layer_labels)
    mod_accessor = ml.modules

    for param in ml.param_logs:
        # used_by_layers exist
        for lbl in param.used_by_layers:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"ParamLog '{param.address}' used_by_layers contains "
                    f"'{lbl}' not in layer_labels",
                )

        # module_address exists (skip for conditional models where the module
        # was never called, e.g. MoE routing that skips some experts)
        try:
            mod_accessor[param.module_address]
        except (KeyError, IndexError):
            pass  # Module was never invoked during forward pass

    # uses_params forward check
    for lpl in ml.layer_list:
        if lpl.uses_params:
            if not lpl.parent_param_logs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' has uses_params=True "
                    f"but parent_param_logs is empty",
                )

    # layers_with_params labels exist
    for param_addr, layer_labels in ml.layers_with_params.items():
        for lbl in layer_labels:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"layers_with_params['{param_addr}'] contains '{lbl}' not in layer_labels",
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
            # module_address references a valid module or an ancestor does.
            # Buffers may live on modules that were never entered during the
            # forward pass (e.g. anchor_generator creates buffers but they're
            # consumed by parent code), or in non-module containers
            # (ParameterList, top-level attrs like "rope"). Accept the buffer
            # if any ancestor module is in the accessor.
            addr = buf.module_address
            found_ancestor = addr in ml.modules
            while not found_ancestor and "." in addr:
                addr = addr.rsplit(".", 1)[0]
                found_ancestor = addr in ml.modules
            if not found_ancestor and "" not in ml.modules:
                raise MetadataInvariantError(
                    name,
                    f"BufferLog '{buf.layer_label}' module_address="
                    f"'{buf.module_address}' — no ancestor found in "
                    f"module accessor",
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
    - creation_order is unique across all layers and monotonically
      increasing in layer_list order.
    - operation_num is unique among computational layers (non-input, non-buffer,
      non-output).
    - Topological order: every parent's creation_order < child's.
    - No raw labels (``l_\\d+``) survive postprocessing.
    """
    name = "graph_ordering"

    # creation_order uniqueness and monotonicity
    seen_rt_nums: dict[int, str] = {}
    prev_rt = -1
    for lpl in ml.layer_list:
        rt = lpl.creation_order
        if rt in seen_rt_nums:
            raise MetadataInvariantError(
                name,
                f"Duplicate creation_order={rt}: '{seen_rt_nums[rt]}' and '{lpl.layer_label}'",
            )
        seen_rt_nums[rt] = lpl.layer_label
        if rt <= prev_rt:
            raise MetadataInvariantError(
                name,
                f"creation_order not monotonically increasing: "
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

    # Topological order: parent.creation_order < child.creation_order
    rt_map = {lpl.layer_label: lpl.creation_order for lpl in ml.layer_list}
    for lpl in ml.layer_list:
        for p in lpl.parent_layers:
            if rt_map.get(p, -1) >= lpl.creation_order:
                raise MetadataInvariantError(
                    name,
                    f"Topological violation: parent '{p}' (rt={rt_map.get(p)}) "
                    f">= child '{lpl.layer_label}' (rt={lpl.creation_order})",
                )

    # No raw labels survive postprocessing
    for label in ml.layer_labels:
        if _RAW_LABEL_PATTERN.match(label):
            raise MetadataInvariantError(name, f"Raw label '{label}' survived postprocessing")


# ---------------------------------------------------------------------------
# N. Layer equivalence / loop detection invariants
# ---------------------------------------------------------------------------


def _check_loop_detection_invariants(ml: "ModelLog") -> None:
    """Check N: loop detection / recurrent_group invariants.

    Validates per-layer:
    - recurrent_group is non-empty and includes self.
    - Symmetry: all members agree on the same group.
    - All members share: layer_label_no_pass, operation_equivalence_type,
      func_name (for computational layers).
    - num_passes == len(recurrent_group).
    - Pass numbering within group is contiguous {1..N}.

    Validates cross-layer:
    - Parameter sharing rule: layers with same (func_name,
      sorted(parent_param_barcodes)) must share layer_label_no_pass.
    - Equivalence group consistency: all members of a recurrent_group
      group belong to the same ModelLog.equivalent_operations set.

    Note: subgraph-level adjacency (Rule 3 from loop_detection.py) cannot
    be verified post-hoc from metadata alone.
    """
    name = "loop_detection"
    label_set = set(ml.layer_labels)

    # Build same-layer groups from the authoritative recurrent_group lists
    # Key: frozenset of labels, Value: list of LayerPassLogs in the group
    groups_seen: dict[frozenset[str], list[str]] = {}

    for lpl in ml.layer_list:
        slo = lpl.recurrent_group
        if not slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' has empty recurrent_group",
            )

        # All members in recurrent_group must exist
        for member in slo:
            if member not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' recurrent_group contains "
                    f"'{member}' not in layer_labels",
                )

        # Self-inclusion
        if lpl.layer_label not in slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' not in its own recurrent_group",
            )

        # Symmetry: all members agree on the group
        for member_label in slo:
            member = ml[member_label]
            if set(member.recurrent_group) != set(slo):
                raise MetadataInvariantError(
                    name,
                    f"Asymmetric recurrent_group: '{lpl.layer_label}' has "
                    f"{sorted(slo)} but '{member_label}' has "
                    f"{sorted(member.recurrent_group)}",
                )

        # All members share layer_label_no_pass
        for member_label in slo:
            member = ml[member_label]
            if member.layer_label_no_pass != lpl.layer_label_no_pass:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_group inconsistency: '{lpl.layer_label}' "
                    f"(no_pass='{lpl.layer_label_no_pass}') and '{member_label}' "
                    f"(no_pass='{member.layer_label_no_pass}') differ",
                )

        # All members share operation_equivalence_type
        for member_label in slo:
            member = ml[member_label]
            if member.operation_equivalence_type != lpl.operation_equivalence_type:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_group type mismatch: '{lpl.layer_label}' "
                    f"type='{lpl.operation_equivalence_type}' vs '{member_label}' "
                    f"type='{member.operation_equivalence_type}'",
                )

        # All members share func_name (for computational layers)
        if not (lpl.is_input_layer or lpl.is_buffer_layer or lpl.is_output_layer):
            for member_label in slo:
                member = ml[member_label]
                if member.func_name != lpl.func_name:
                    raise MetadataInvariantError(
                        name,
                        f"recurrent_group func mismatch: '{lpl.layer_label}' "
                        f"func='{lpl.func_name}' vs '{member_label}' "
                        f"func='{member.func_name}'",
                    )

        # num_passes == len(recurrent_group)
        if lpl.num_passes != len(slo):
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}': num_passes={lpl.num_passes} "
                f"!= len(recurrent_group)={len(slo)}",
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
    # Layers with the same func_name AND identical sorted(parent_param_barcodes)
    # must share the same layer_label_no_pass (they are the same "layer" across
    # different passes).  The func_name is included in the grouping key
    # because different operations (e.g., isinf, expand, nantonum) can consume
    # the same parameter tensor without being the same logical layer.
    param_groups: dict[tuple[str, tuple[str, ...]], list[LayerPassLog]] = defaultdict(list)
    for lpl in ml.layer_list:
        if lpl.uses_params and lpl.parent_param_barcodes:
            key = (lpl.func_name, tuple(sorted(lpl.parent_param_barcodes)))
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
    # recurrent_group group must belong to the same equivalence set.
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
                f"recurrent_group group {sorted(slo)} spans multiple "
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
    - is_output_ancestor <-> output_descendants is non-empty.
    - input_ancestors subset of input_layers; output_descendants subset of
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

        # is_output_ancestor ↔ output_descendants non-empty
        has_descendents = len(lpl.output_descendants) > 0
        if lpl.is_output_ancestor != has_descendents:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': is_output_ancestor={lpl.is_output_ancestor} but "
                f"len(output_descendants)={len(lpl.output_descendants)}",
            )

        # input_ancestors subset of input_layers
        extra_ancestors = lpl.input_ancestors - input_set
        if extra_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': input_ancestors contains labels not in "
                f"input_layers: {extra_ancestors}",
            )

        # output_descendants subset of output_layers
        extra_desc = lpl.output_descendants - output_set
        if extra_desc:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': output_descendants contains labels not in "
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
            and not lpl.is_internally_initialized
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
    - Per-layer containing_modules (call nesting stack):
      - Last element matches containing_module.
      - Every element is a known module address.
      - No duplicate addresses (can't be inside the same module twice).
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
        visited: set[str] = set()
        current: str | None = addr
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

    # Per-layer: containing_modules path validity
    # Format after postprocessing: list of "addr:pass" strings, ordered from
    # outermost enclosing submodule to innermost. Does NOT include "self".
    for lpl in ml.layer_list:
        nested = lpl.containing_modules
        if not nested:
            continue

        # Leaf consistency: last element matches containing_module
        if lpl.containing_module is not None:
            if nested[-1] != lpl.containing_module:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': last nested module '{nested[-1]}' "
                    f"!= containing_module '{lpl.containing_module}'",
                )

        # Path validity: each element must be a known module, and no
        # duplicate addresses (a module can't appear twice in the same call
        # stack).  We do NOT check address depth ordering — address depth
        # (position in the nn.Module tree) is independent of call nesting
        # depth (position on the forward() call stack).  A module at a
        # shallow address can be called from inside a deeply-addressed
        # module's forward(), e.g., encoder.blocks.1.1.attention calling
        # encoder.attention_structure.sin_dropout.
        seen_addrs = set()
        for entry in nested:
            addr = entry.split(":")[0] if ":" in entry else entry
            if addr not in known_addrs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': nested path contains unknown "
                    f"module address '{addr}'",
                )
            if addr in seen_addrs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': duplicate module address "
                    f"'{addr}' in nested path {nested}",
                )
            seen_addrs.add(addr)


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
