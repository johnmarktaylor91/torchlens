"""Metadata invariant checks for ``Trace`` and its sub-objects.

Single entry point: ``check_metadata_invariants(trace)`` runs all checks
and raises ``MetadataInvariantError`` on the first failure.

**Phase 1 -- Structural invariants:**
  A. Trace self-consistency (counts, timing, label uniqueness)
  B. Special layer lists match per-layer boolean flags
  C. Graph topology (parent-child bidirectionality, boolean flag consistency)
  D. Op field consistency (shape, dtype, pass numbering, nesting)
  E. Recurrence / loop invariants (is_recurrent, pass dicts)
  F. Branching invariants (is_branching)
  F2. Conditional metadata invariants (15 conditional consistency checks)
  G. Op <-> Layer cross-references (pass numbering, back-pointers)
  H. Module <-> Layer containment (layers, module pass layers, reverse check)
  I. Module hierarchy (address parent-child bidirectionality, pass consistency)
  J. Param cross-references (Param -> layer, uses_params flag)
  K. Buffer cross-references (buffer_layers list, Buffer module references)
  L. Equivalence group symmetry (op_equivalence_classes labels are valid)

**Phase 2 -- Semantic invariants:**
  M. Graph ordering (raw_index uniqueness/monotonicity, topological order, no raw labels)
  N. Loop detection invariants (recurrent_ops symmetry, func identity, param sharing, pass numbering)
  O. Distance / reachability (min <= max, input/output layer distances == 0, ancestor/descendent consistency)
  P. Graph connectivity (non-input non-buffer layers have parents, orphans removed)
  Q. Module containment logic (address acyclicity, depth consistency, nested path ordering)
  R. Lookup key bidirectionality (forward/reverse dicts, raw/final label maps)
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ..errors._base import ValidationError

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from ..data_classes.module import Module


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


def check_metadata_invariants(trace: "Trace") -> bool:
    """Run all metadata invariant checks on a completed ``Trace``.

    Checks run in dependency order: Phase 1 structural checks first, then
    Phase 2 semantic checks. Raises ``MetadataInvariantError`` on the first
    failure, so later checks can assume earlier ones passed.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Returns
    -------
    bool
        ``True`` if all invariants pass.
    """
    # --- Phase 1: structural invariants (A-L) ---
    _check_trace_self_consistency(trace)  # A
    _check_backward_graph_invariants(trace)  # T
    _check_special_layer_lists(trace)  # B
    _check_graph_topology(trace)  # C
    _check_op_log_fields(trace)  # D
    _check_recurrence_invariants(trace)  # E
    _check_branching_invariants(trace)  # F
    _check_conditional_invariants(trace)  # F2
    _check_layer_pass_to_layer_log_xrefs(trace)  # G
    _check_module_layer_containment(trace)  # H
    _check_module_hierarchy(trace)  # I
    _check_param_xrefs(trace)  # J
    _check_buffer_xrefs(trace)  # K
    _check_equivalence_symmetry(trace)  # L

    # --- Phase 2: semantic invariants (M-R) ---
    _check_graph_ordering(trace)  # M
    _check_loop_detection_invariants(trace)  # N
    _check_distance_invariants(trace)  # O
    _check_graph_connectivity(trace)  # P
    _check_module_containment_logic(trace)  # Q
    _check_lookup_key_consistency(trace)  # R
    check_func_call_id_invariant(trace)  # S
    return True


def check_func_call_id_invariant(trace: "Trace") -> InvariantResult:
    """Invariant S: func_call_id consistency.

    For intervention-ready logs, every non-synthetic function output must have a
    ``func_call_id``. Outputs from the same call must agree on function identity,
    call stack, captured templates, and container spec; each output in the group
    must have a unique ``container_path``; and same-call groups must not span
    incompatible pass labels. Synthetic input, output, and buffer nodes are
    exempt.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Returns
    -------
    InvariantResult
        Passing result when no inconsistency is found.
    """

    name = "func_call_id_consistency"
    if not getattr(trace, "intervention_ready", False):
        return InvariantResult(name=name, passed=True)

    groups: dict[int, list["Op"]] = defaultdict(list)
    for layer in trace.layer_list:
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
        container_paths = []
        pass_indices = set()
        no_call_labels = set()
        for layer in group:
            if _func_call_group_signature(layer) != expected_signature:
                raise MetadataInvariantError(
                    name,
                    f"func_call_id {func_call_id} has incompatible call metadata",
                )
            container_path = tuple(getattr(layer, "container_path", ()) or ())
            if container_path in container_paths:
                raise MetadataInvariantError(
                    name,
                    f"func_call_id {func_call_id} has duplicate container_path {container_path!r}",
                )
            container_paths.append(container_path)
            pass_indices.add(layer.pass_index)
            no_call_labels.add(layer.layer_label)
        if len(pass_indices) > 1 and len(no_call_labels) > 1:
            raise MetadataInvariantError(
                name,
                f"func_call_id {func_call_id} spans incompatible pass labels",
            )
    return InvariantResult(name=name, passed=True)


def _check_backward_graph_invariants(trace: "Trace") -> None:
    """Check T: backward grad-fn metadata consistency.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Raises
    ------
    MetadataInvariantError
        If backward metadata is internally inconsistent.
    """

    name = "backward_graph_invariants"
    sync_projection = getattr(trace, "_sync_backward_projection_if_needed", None)
    if callable(sync_projection):
        sync_projection()
    if not trace.grad_fn_logs:
        return

    grad_fn_ids = set(trace.grad_fn_logs)
    order_ids = set(trace.grad_fn_order)
    if not order_ids <= grad_fn_ids:
        missing = sorted(order_ids - grad_fn_ids)
        raise MetadataInvariantError(name, f"grad_fn_order contains unknown ids {missing!r}")

    root_ids = trace.backward_root_grad_fn_object_ids
    if not isinstance(root_ids, list):
        raise MetadataInvariantError(
            name,
            f"backward_root_grad_fn_object_ids must be a list, got {type(root_ids).__name__}",
        )
    missing_root_ids = [root_id for root_id in root_ids if root_id not in trace.grad_fn_logs]
    if missing_root_ids:
        raise MetadataInvariantError(
            name,
            f"backward_root_grad_fn_object_ids {missing_root_ids!r} are not present in grad_fn_logs",
        )

    valid_pass_indices = set(getattr(trace, "backward_pass_logs", {}).keys())
    layer_labels = set(trace.layer_labels)
    for grad_fn_object_id, grad_fn_handle in trace.grad_fn_logs.items():
        if not re.fullmatch(r"[a-z0-9_]+_back_[1-9]\d*_[1-9]\d*", grad_fn_handle.label):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label!r} does not match backward-native label grammar",
            )
        if grad_fn_handle.has_op and grad_fn_handle.op_label not in layer_labels:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} points to missing layer {grad_fn_handle.op_label!r}",
            )
        membership_source = getattr(grad_fn_handle, "module_membership_source", None)
        if membership_source not in {None, "paired", "inferred"}:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has invalid module_membership_source "
                f"{membership_source!r}",
            )
        if membership_source is None:
            if grad_fn_handle.module_address is not None or grad_fn_handle.modules:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has module containment without a source",
                )
        elif grad_fn_handle.module_address is None or not grad_fn_handle.modules:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has incomplete {membership_source!r} module containment",
            )
        op = grad_fn_handle.op
        if grad_fn_handle.has_op != (op is not None):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has inconsistent has_op/op fields",
            )
        if grad_fn_handle.grad_fn_object_id != grad_fn_object_id:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} stored id {grad_fn_handle.grad_fn_object_id!r} under {grad_fn_object_id!r}",
            )
        creator_object_id = getattr(grad_fn_handle, "creator_object_id", None)
        if creator_object_id is not None:
            creator = trace.grad_fn_logs.get(creator_object_id)
            if creator is None:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} points to missing creator id {creator_object_id!r}",
                )
            if grad_fn_handle.origin_backward_pass not in valid_pass_indices:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has invalid origin backward pass "
                    f"{grad_fn_handle.origin_backward_pass!r}",
                )
            if creator.order is not None and grad_fn_handle.order is not None:
                expected_order = creator.order + 1
                if grad_fn_handle.order != expected_order:
                    raise MetadataInvariantError(
                        name,
                        f"{grad_fn_handle.label} order {grad_fn_handle.order!r} does not "
                        f"match creator order + 1 ({expected_order!r})",
                    )
        call_ordinals = sorted(grad_fn_handle.calls.keys())
        if call_ordinals != list(range(1, len(call_ordinals) + 1)):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has non-dense local call ordinals {call_ordinals!r}",
            )
        for ordinal, call in grad_fn_handle.calls.items():
            if call.ordinal != ordinal or call.call_index != ordinal:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label}:{ordinal} has inconsistent local ordinal fields",
                )
            if call.backward_pass_index is None:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label}:{ordinal} is missing backward_pass_index",
                )

    for layer in trace.layer_list:
        grad_fn_object_id = layer.grad_fn_object_id
        if grad_fn_object_id is None or layer.grad_fn is None:
            continue
        if grad_fn_object_id not in trace.grad_fn_logs:
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} points to missing grad_fn_handle id "
                f"{grad_fn_object_id!r}",
            )

    expected_saved_grad_labels = {layer.label for layer in trace.layer_list if layer.has_grad}
    saved_grad_labels = {op.label for op in trace.saved_grad_ops}
    if saved_grad_labels != expected_saved_grad_labels:
        raise MetadataInvariantError(
            name,
            "saved_grad_ops does not match layers with saved grad tensors",
        )

    expected_pass_indices = list(range(1, trace.num_backward_passes + 1))
    actual_pass_indices = sorted(getattr(trace, "backward_pass_logs", {}).keys())
    if actual_pass_indices != expected_pass_indices:
        raise MetadataInvariantError(
            name,
            f"backward_pass_logs keys {actual_pass_indices!r} are not dense "
            f"1..{trace.num_backward_passes}",
        )
    valid_pass_indices = set(actual_pass_indices)
    for pass_index, backward_pass in getattr(trace, "backward_pass_logs", {}).items():
        if backward_pass.pass_index != pass_index:
            raise MetadataInvariantError(
                name,
                f"BackwardPass stored index {backward_pass.pass_index!r} under {pass_index!r}",
            )
        for call in backward_pass.grad_fn_calls:
            if call.backward_pass_index != pass_index:
                raise MetadataInvariantError(
                    name,
                    f"{call.call_label} is attached to pass {pass_index} but records "
                    f"pass {call.backward_pass_index}",
                )
    for grad_fn_handle in trace.grad_fn_logs.values():
        for call in grad_fn_handle.calls.values():
            if call.backward_pass_index not in valid_pass_indices:
                raise MetadataInvariantError(
                    name,
                    f"{call.call_label} references missing backward pass "
                    f"{call.backward_pass_index}",
                )


def _is_func_call_id_exempt(layer: "Op") -> bool:
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

    if layer.is_input or layer.is_output or layer.is_buffer:
        return True
    func = getattr(layer, "func", None)
    func_name = str(getattr(layer, "func_name", "")).lower()
    return func in {"input", "output", "buffer"} or func_name in {
        "input",
        "output",
        "buffer",
        "none",
    }


def _func_call_group_signature(layer: "Op") -> tuple[object, ...]:
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
        tuple(repr(location) for location in (layer.code_context or ())),
        repr(layer.args_template),
        repr(layer.kwargs_template),
        repr(layer.container_spec),
    )


# ---------------------------------------------------------------------------
# A. Trace self-consistency
# ---------------------------------------------------------------------------


def _check_trace_self_consistency(ml: "Trace") -> None:
    """Check A: Trace aggregate counts and metadata are internally consistent.

    Validates:
    - layer_labels length matches layer_list length, no duplicates.
    - num_ops == count of computational (non-input, non-output,
      non-buffer) layers.
    - Param counts (total, trainable, frozen) are consistent and sum correctly.
      Uses deduplication by layer_label to match labeling.py logic.
    - At least one output layer exists.
    - Timing values are non-negative and ordered.
    - Tensor counts: total >= saved.
    """
    name = "trace_self_consistency"

    # op_labels vs layer_list length
    if len(ml.op_labels) != len(ml.layer_list):
        raise MetadataInvariantError(
            name,
            f"len(op_labels)={len(ml.op_labels)} != len(layer_list)={len(ml.layer_list)}",
        )

    # No duplicate labels
    if len(ml.op_labels) != len(set(ml.op_labels)):
        dupes = [lbl for lbl in ml.op_labels if ml.op_labels.count(lbl) > 1]
        raise MetadataInvariantError(name, f"Duplicate op_labels: {set(dupes)}")

    # num_ops counts computational layers only (excludes input, output,
    # buffer).  We check per-layer flags instead of comparing against label
    # sets because buffer_layers stores pass-qualified labels while
    # layer_labels strips the pass suffix -- they use different formats.
    expected_ops = sum(
        1 for lpl in ml.layer_list if not (lpl.is_input or lpl.is_output or lpl.is_buffer)
    )
    if ml.num_ops != expected_ops:
        raise MetadataInvariantError(
            name,
            f"num_ops={ml.num_ops} != expected computational layers={expected_ops}",
        )

    # Param counts must be deduplicated by layer_label because
    # multi-pass layers share the same params -- counting each pass would
    # double-count.  This matches the summation logic in labeling.py:116-122.
    seen_no_pass: set[str] = set()
    expected_param_sum = 0
    expected_num_params = 0
    for lpl in ml.layer_list:
        if lpl.layer_label not in seen_no_pass:
            expected_param_sum += lpl.num_param_tensors
            expected_num_params += lpl.num_params
            seen_no_pass.add(lpl.layer_label)
    if ml.num_param_tensors != expected_param_sum:
        raise MetadataInvariantError(
            name,
            f"num_param_tensors={ml.num_param_tensors} != "
            f"sum(unique num_param_tensors)={expected_param_sum}",
        )
    if ml.num_params != expected_num_params:
        raise MetadataInvariantError(
            name,
            f"num_params={ml.num_params} != sum(unique num_params)={expected_num_params}",
        )

    if ml.num_params_trainable + ml.num_params_frozen != ml.num_params:
        raise MetadataInvariantError(
            name,
            f"trainable({ml.num_params_trainable}) + frozen({ml.num_params_frozen}) "
            f"!= total({ml.num_params})",
        )

    # At least one output layer
    if len(ml.output_layers) == 0:
        raise MetadataInvariantError(name, "No output layers found")

    # Timing
    if ml.capture_duration < 0:
        raise MetadataInvariantError(name, f"capture_duration={ml.capture_duration} < 0")
    if ml.capture_start_time > ml.capture_end_time:
        raise MetadataInvariantError(
            name,
            f"capture_start_time={ml.capture_start_time} > capture_end_time={ml.capture_end_time}",
        )

    # Tensor counts
    if ml.num_tensors < ml.num_saved_ops:
        raise MetadataInvariantError(
            name,
            f"num_tensors={ml.num_tensors} < num_saved_ops={ml.num_saved_ops}",
        )


# ---------------------------------------------------------------------------
# B. Special layer lists ↔ Op flags
# ---------------------------------------------------------------------------

_SPECIAL_LIST_FLAG_PAIRS = [
    ("input_layers", "is_input", "layer"),
    ("output_layers", "is_output", "layer"),
    ("buffer_layers", "is_buffer", "layer"),
    ("internal_source_ops", "is_internal_source", "op"),
    ("internal_sink_ops", "is_internal_sink", "op"),
]


def _check_special_layer_lists(ml: "Trace") -> None:
    """Check B: special layer lists (input, output, buffer, etc.) match per-layer boolean flags.

    For each (list_attr, flag_attr) pair, verifies bidirectional consistency:
    - Forward: every label in the list has the flag set on its Op.
    - Reverse: every Op with the flag set appears in the list.
    """
    name = "special_layer_lists"
    for list_attr, flag_attr, label_kind in _SPECIAL_LIST_FLAG_PAIRS:
        special_list = getattr(ml, list_attr)
        special_set = set(special_list)
        label_set = set(ml.op_labels if label_kind == "op" else ml.layer_labels)
        label_field = "op_labels" if label_kind == "op" else "layer_labels"

        # All entries must be valid labels
        missing = special_set - label_set
        if missing:
            raise MetadataInvariantError(
                name, f"{list_attr} contains labels not in {label_field}: {missing}"
            )

        # Forward: every label in the list has the flag set
        for label in special_list:
            lpl = ml[label]
            if not getattr(lpl, flag_attr):
                raise MetadataInvariantError(
                    name,
                    f"{label_kind.title()} {label} is in {list_attr} but {flag_attr}=False",
                )

        # Reverse: every layer/op with the flag is in the list.
        for lpl in ml.layer_list:
            label = lpl.label if label_kind == "op" else lpl.layer_label
            if getattr(lpl, flag_attr) and label not in special_set:
                raise MetadataInvariantError(
                    name,
                    f"{label_kind.title()} {label} has {flag_attr}=True but is not in {list_attr}",
                )


# ---------------------------------------------------------------------------
# C. Graph topology
# ---------------------------------------------------------------------------


def _check_graph_topology(ml: "Trace") -> None:
    """Check C: parent-child edge bidirectionality and boolean flag consistency.

    Validates:
    - Every parent edge has a corresponding child edge (and vice versa).
    - has_children/has_parents/has_siblings/has_co_parents flags match actual counts.
      Note: has_children excludes output layers (added during postprocessing,
      not during capture when the flag was set).
    - Input layers have no parents.
    - out_versions_by_child keys are a subset of children.
    """
    name = "graph_topology"
    label_set = set(ml.layer_labels) | set(ml.op_labels)
    output_set = set(ml.output_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Parent-child bidirectionality
        for p in lpl.parents:
            parent = ml[p]
            if p not in label_set and parent.layer_label not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has parent {p} not in layer_labels"
                )
            if label not in parent.children and lpl.label not in parent.children:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {p} as parent, but {p} does not list {label} as child",
                )

        for c in lpl.children:
            child = ml[c]
            if c not in label_set and child.layer_label not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has child {c} not in layer_labels"
                )
            if label not in child.parents and lpl.label not in child.parents:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {c} as child, but {c} does not list {label} as parent",
                )

        # Boolean flag consistency
        # Note: has_children/has_parents/has_siblings/has_co_parents are set during
        # capture and don't account for output layers added during postprocessing.
        # So we exclude output layers from the child count for this check.
        non_output_children = [
            c for c in lpl.children if c not in output_set and ml[c].layer_label not in output_set
        ]
        if lpl.has_children != (len(non_output_children) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_children={lpl.has_children} but "
                f"non-output children={non_output_children}",
            )
        if not lpl.is_output and lpl.has_parents != (len(lpl.parents) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_parents={lpl.has_parents} but len(parents)={len(lpl.parents)}",
            )
        if lpl.has_siblings != (len(lpl.siblings) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_siblings={lpl.has_siblings} but "
                f"len(siblings)={len(lpl.siblings)}",
            )
        if lpl.has_co_parents != (len(lpl.co_parents) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_co_parents={lpl.has_co_parents} but "
                f"len(co_parents)={len(lpl.co_parents)}",
            )

        # Input layers have no parents
        if lpl.is_input and len(lpl.parents) > 0:
            raise MetadataInvariantError(
                name,
                f"Input layer {label} has parents={lpl.parents}",
            )

        # out_versions_by_child keys subset of children
        ctv_keys = set(lpl.out_versions_by_child.keys())
        child_set = set(lpl.children)
        extra = ctv_keys - child_set
        if extra:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: out_versions_by_child has keys not in children: {extra}",
            )


# ---------------------------------------------------------------------------
# D. Op field consistency
# ---------------------------------------------------------------------------


def _check_op_log_fields(ml: "Trace") -> None:
    """Check D: per-layer field consistency (shape, dtype, pass numbering, func, nesting).

    Validates:
    - Saved tensor shape/dtype match actual out (when saved).
    - Pass numbering: pass_index >= 1, num_passes >= pass_index.
    - Computational layers have callable func and non-empty func_name.
    - step_index >= 1 for non-input/non-buffer layers.
    - module_call_depth matches len(modules).
    - Label format: pass-qualified label has ':' iff multi-pass; no-pass label never has ':'.
    """
    name = "op_log_fields"

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Tensor shape/dtype consistency when outs are saved
        if lpl.has_saved_activation and lpl.out is not None:
            actual_shape = tuple(lpl.out.shape)
            if lpl.shape != actual_shape:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: shape={lpl.shape} != actual shape={actual_shape}",
                )
            if lpl.dtype != lpl.out.dtype:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: dtype={lpl.dtype} != actual dtype={lpl.out.dtype}",
                )

        # Pass numbering
        if lpl.pass_index < 1:
            raise MetadataInvariantError(name, f"Layer {label}: pass_index={lpl.pass_index} < 1")
        if lpl.num_passes < lpl.pass_index:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: num_passes={lpl.num_passes} < pass_index={lpl.pass_index}",
            )

        # A GENUINE raw-forward-hook output replacement is legitimately
        # functionless: the user substituted an opaque tensor for a module's
        # output, so there is no torch function to validate. This exemption is
        # deliberately narrow -- it must NOT cover auto-synthesized placeholders
        # during plain capture (a previous band-aid widened it to silence the
        # vmap-built attention mask, disarming this tripwire).
        is_functionless_replacement = (
            lpl.func_name == "intervention_replacement"
            and getattr(lpl, "intervention_replaced", False)
            and not getattr(lpl, "is_internal_source", False)
        )

        # An internally generated *source* tensor whose construction TorchLens
        # could not trace (e.g. an attention mask built inside torch.vmap) is a
        # genuine functionless graph source, exactly like a buffer: func is None
        # and func_name is "none". Traced ops that merely have an internal-source
        # ancestor still carry a real callable func and are NOT exempted here.
        is_functionless_internal_source = (
            getattr(lpl, "is_internal_source", False) and lpl.func is None
        )

        # Function applied (non-input, non-buffer, non-output, non-source,
        # non-hook-replacement layers).
        if not (
            lpl.is_input
            or lpl.is_buffer
            or lpl.is_output
            or is_functionless_internal_source
            or is_functionless_replacement
        ):
            if not callable(lpl.func):
                raise MetadataInvariantError(name, f"Layer {label}: func is not callable")
            if not lpl.func_name:
                raise MetadataInvariantError(name, f"Layer {label}: func_name is empty")

        # Operation numbering (input/buffer/output bookkeeping layers have step_index=0)
        if not (lpl.is_input or lpl.is_buffer or lpl.is_output):
            if lpl.step_index is not None and lpl.step_index < 1:
                raise MetadataInvariantError(
                    name, f"Layer {label}: step_index={lpl.step_index} < 1"
                )
        if lpl.raw_index < 1:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: raw_index={lpl.raw_index} < 1",
            )

        # Module nesting depth
        if lpl.module_call_depth != len(lpl.modules):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_call_depth={lpl.module_call_depth} != "
                f"len(modules)="
                f"{len(lpl.modules)}",
            )

        # Label format: pass-qualified label has ":" iff multi-pass
        if lpl.num_passes > 1 and ":" not in lpl.label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: multi-pass but label='{lpl.label}' has no ':'",
            )
        if ":" in lpl.layer_label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: layer_label='{lpl.layer_label}' contains ':'",
            )


# ---------------------------------------------------------------------------
# E. Recurrence / loop invariants
# ---------------------------------------------------------------------------


def _check_recurrence_invariants(ml: "Trace") -> None:
    """Check E: recurrence / loop invariants.

    Validates:
    - is_recurrent == True iff any layer has >1 pass.
    - max_layer_op_count matches the maximum pass count.
    - layer_num_calls keys are valid no-pass labels.
    - Layer.ops dict keys are contiguous {1..N}.
    """
    name = "recurrence_invariants"

    any_recurrent = any(v > 1 for v in ml.layer_num_calls.values())
    if ml.is_recurrent != any_recurrent:
        raise MetadataInvariantError(
            name,
            f"is_recurrent={ml.is_recurrent} but any layer has >1 pass = {any_recurrent}",
        )

    if ml.is_recurrent:
        expected_max = max(ml.layer_num_calls.values())
        if ml.max_layer_op_count != expected_max:
            raise MetadataInvariantError(
                name,
                f"max_layer_op_count={ml.max_layer_op_count} != "
                f"max(layer_num_calls)={expected_max}",
            )

    # Per-layer pass consistency: layer_num_calls is keyed by no-pass labels.
    # Validate that each key exists in layer_labels, and that the
    # recorded count matches the actual Layer.num_passes.
    no_call_labels = set(ml.layer_labels)
    for label_key, num_calls in ml.layer_num_calls.items():
        if label_key not in no_call_labels:
            raise MetadataInvariantError(
                name,
                f"layer_num_calls key '{label_key}' not in layer_labels",
            )
        if label_key in ml.layer_logs:
            actual = ml.layer_logs[label_key].num_passes
            if num_calls != actual:
                raise MetadataInvariantError(
                    name,
                    f"layer_num_calls['{label_key}']={num_calls} != Layer.num_passes={actual}",
                )

    # For top-level (no-pass) layer_logs, verify pass dict consistency
    for no_call_label, ll in ml.layer_logs.items():
        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{no_call_label}' ops keys={actual_keys} != expected {expected_keys}",
            )


# ---------------------------------------------------------------------------
# F. Branching invariants
# ---------------------------------------------------------------------------


def _check_branching_invariants(ml: "Trace") -> None:
    """Check F: is_branching matches whether any layer has >1 child."""
    name = "branching_invariants"
    any_branching = any(len(lpl.children) > 1 for lpl in ml.layer_list)
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
    """Return a layer label without any trailing ``:call_index`` suffix.

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


def _get_label_call_index(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:call_index`` suffix.

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
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project pass-level child views from the primary conditional structure.

    Parameters
    ----------
    conditional_arm_children:
        Primary ``cond_id -> branch_kind -> child labels`` mapping on a
        ``Op``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views.
    """

    then_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("then", [])
        }
    )
    elif_children: dict[int, list[str]] = {}
    grouped_elif_children: dict[int, set[str]] = defaultdict(set)
    for branch_children in conditional_arm_children.values():
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
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("else", [])
        }
    )
    return then_children, elif_children, else_children


def _expected_layer_log_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project aggregate child views from a ``Layer`` primary structure.

    Parameters
    ----------
    conditional_arm_children:
        Aggregate ``cond_id -> branch_kind -> child labels`` mapping on a
        ``Layer``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views preserving first-seen order.
    """

    then_children: list[str] = []
    elif_children: dict[int, list[str]] = {}
    else_children: list[str] = []
    for branch_children in conditional_arm_children.values():
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
    layer_log: "Layer",
) -> dict[int, dict[str, list[str]]]:
    """Build the expected aggregate ``conditional_arm_children`` for a ``Layer``.

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
    for call_index, pass_log in sorted(layer_log.ops.items()):
        for conditional_id, branch_children in pass_log.conditional_arm_children.items():
            merged_branch_children = expected_children_by_cond.setdefault(conditional_id, {})
            for branch_kind, child_labels in branch_children.items():
                merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
                for child_label in child_labels:
                    _append_unique(merged_child_labels, _strip_pass_suffix(child_label))
    return expected_children_by_cond


def _valid_conditional_child_labels(ml: "Trace") -> set[str]:
    """Return the set of valid labels for conditional child references.

    Parameters
    ----------
    ml:
        Model log being validated.

    Returns
    -------
    set[str]
        Union of pass-level labels and aggregate ``Layer`` keys.
    """

    return set(ml.layer_labels) | set(ml.layer_logs)


def _check_conditional_invariants(ml: "Trace") -> None:
    """Check F2: conditional metadata invariants added in Phase 6.

    Parameters
    ----------
    ml:
        Model log being validated.
    """

    name = "conditional_invariants"
    layer_label_set = set(ml.layer_labels)
    valid_child_labels = _valid_conditional_child_labels(ml)
    event_id_set = {event.id for event in ml.conditional_records}
    branch_context_kinds = {"if_test", "elif_test", "ifexp"}
    wrapped_context_kinds = branch_context_kinds | {"bool_cast"}

    # Invariant 1: conditional_arm_entry_edges ↔ conditional_arm_children.
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            if parent_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] references "
                    f"missing parent layer {parent_label!r}",
                )
            parent_layer = ml.layer_logs[parent_label]
            branch_children = parent_layer.conditional_arm_children.get(conditional_id, {}).get(
                branch_kind, []
            )
            if child_label not in branch_children:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] includes edge "
                    f"({parent_label!r}, {child_label!r}) but "
                    f"{parent_label}.conditional_arm_children[{conditional_id}][{branch_kind!r}]="
                    f"{branch_children}",
                )

    for parent_layer in ml.layer_logs.values():
        for conditional_id, branch_map in parent_layer.conditional_arm_children.items():
            for branch_kind, child_labels in branch_map.items():
                model_edges = ml.conditional_arm_entry_edges.get((conditional_id, branch_kind), [])
                for child_label in child_labels:
                    if (parent_layer.layer_label, child_label) not in model_edges:
                        _fail_conditional_invariant(
                            name,
                            1,
                            f"{parent_layer.layer_label}.conditional_arm_children"
                            f"[{conditional_id}][{branch_kind!r}] includes {child_label!r} "
                            f"but conditional_arm_entry_edges[{(conditional_id, branch_kind)}]={model_edges}",
                        )

    # Invariant 2: per-layer derived views are exact projections of the primary structures.
    for layer in ml.layer_list:
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_pass_child_views(layer.conditional_arm_children)
        )
        if layer.conditional_then_children != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_then_children={layer.conditional_then_children} != "
                f"expected projection {expected_then_children}",
            )
        if layer.conditional_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_elif_children={layer.conditional_elif_children} != "
                f"expected projection {expected_elif_children}",
            )
        if layer.conditional_else_children != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_else_children={layer.conditional_else_children} != "
                f"expected projection {expected_else_children}",
            )

    for layer_log in ml.layer_logs.values():
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_log_child_views(layer_log.conditional_arm_children)
        )
        if layer_log.conditional_then_children != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_then_children="
                f"{layer_log.conditional_then_children} != expected projection "
                f"{expected_then_children}",
            )
        if layer_log.conditional_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_elif_children="
                f"{layer_log.conditional_elif_children} != expected projection "
                f"{expected_elif_children}",
            )
        if layer_log.conditional_else_children != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_else_children="
                f"{layer_log.conditional_else_children} != expected projection "
                f"{expected_else_children}",
            )

    # Invariant 3: every child label in conditional child views exists in the log.
    for layer in ml.layer_list:
        for field_name, child_labels in (
            ("conditional_entry_children", layer.conditional_entry_children),
            ("conditional_then_children", layer.conditional_then_children),
            ("conditional_else_children", layer.conditional_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.{field_name} contains missing child label "
                        f"{child_label!r}",
                    )
        for elif_index, child_labels in layer.conditional_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.conditional_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for layer_log in ml.layer_logs.values():
        for field_name, child_labels in (
            ("conditional_entry_children", layer_log.conditional_entry_children),
            ("conditional_then_children", layer_log.conditional_then_children),
            ("conditional_else_children", layer_log.conditional_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"Layer {layer_log.layer_label}.{field_name} contains missing child "
                        f"label {child_label!r}",
                    )
        for elif_index, child_labels in layer_log.conditional_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"Layer {layer_log.layer_label}.conditional_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for parent_label, child_label in ml.conditional_branch_edges:
        if child_label not in valid_child_labels:
            _fail_conditional_invariant(
                name,
                3,
                f"Trace.conditional_branch_edges contains missing child label {child_label!r} "
                f"for parent {parent_label!r}",
            )
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            if child_label not in valid_child_labels:
                _fail_conditional_invariant(
                    name,
                    3,
                    f"Trace.conditional_arm_entry_edges contains missing child label "
                    f"{child_label!r} for edge {(conditional_id, branch_kind, parent_label)}",
                )

    # Invariant 4: bool classification fields are mutually consistent.
    for layer in ml.layer_list:
        expected_is_branch = layer.conditional_context_kind in branch_context_kinds
        if layer.is_terminal_conditional_bool != expected_is_branch:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has is_terminal_conditional_bool={layer.is_terminal_conditional_bool} but "
                f"conditional_context_kind={layer.conditional_context_kind!r}",
            )
        if layer.is_terminal_conditional_bool and layer.terminal_conditional_id is None:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has is_terminal_conditional_bool=True but terminal_conditional_id is None",
            )
        if layer.conditional_context_kind is not None and not layer.is_terminal_bool:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has conditional_context_kind={layer.conditional_context_kind!r} but "
                f"is_terminal_bool=False",
            )
        if (
            layer.conditional_wrapper_kind is not None
            and layer.conditional_context_kind not in wrapped_context_kinds
        ):
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has conditional_wrapper_kind={layer.conditional_wrapper_kind!r} but "
                f"conditional_context_kind={layer.conditional_context_kind!r}",
            )

    # Invariant 5: every referenced cond_id corresponds to a ConditionalEvent.
    referenced_cond_ids: set[int] = set()
    for layer in ml.layer_list:
        referenced_cond_ids.update(
            conditional_id for conditional_id, _ in layer.conditional_branch_stack
        )
        if layer.terminal_conditional_id is not None:
            referenced_cond_ids.add(layer.terminal_conditional_id)
        referenced_cond_ids.update(layer.conditional_arm_children)
    for layer_log in ml.layer_logs.values():
        referenced_cond_ids.update(
            conditional_id
            for branch_stack in layer_log.conditional_role_stacks
            for conditional_id, _ in branch_stack
        )
        referenced_cond_ids.update(layer_log.conditional_arm_children)
    referenced_cond_ids.update(
        conditional_id for conditional_id, _ in ml.conditional_arm_entry_edges
    )

    for conditional_id in sorted(referenced_cond_ids):
        if conditional_id not in event_id_set:
            _fail_conditional_invariant(
                name,
                5,
                f"Referenced cond_id {conditional_id} has no matching ConditionalEvent.id "
                f"in Trace.conditional_records",
            )

    # Invariant 6: parent->child stacks are monotone by prefix relation.
    for parent_op in ml.layer_list:
        for child_label in parent_op.children:
            child_layer = ml[child_label]
            if parent_op.pass_index != child_layer.pass_index:
                continue
            if parent_op.conditional_branch_stack == child_layer.conditional_branch_stack:
                continue
            if _is_prefix_stack(
                parent_op.conditional_branch_stack, child_layer.conditional_branch_stack
            ):
                continue
            if _is_prefix_stack(
                child_layer.conditional_branch_stack, parent_op.conditional_branch_stack
            ):
                continue
            _fail_conditional_invariant(
                name,
                6,
                f"Edge ({parent_op.layer_label!r}, {child_label!r}) has non-prefix "
                f"conditional stacks parent={parent_op.conditional_branch_stack} "
                f"child={child_layer.conditional_branch_stack}",
            )

    # Invariant 7: elif keys are contiguous on ConditionalEvent.
    for event in ml.conditional_records:
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
    for event in ml.conditional_records:
        for bool_label in event.bool_layers:
            if bool_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers contains missing label "
                    f"{bool_label!r}",
                )
            try:
                bool_layer = ml.ops[bool_label]
            except (KeyError, ValueError, TypeError):
                bool_layer = ml[bool_label]
            bool_ops = (
                list(cast("Layer", bool_layer).ops.values())
                if hasattr(bool_layer, "ops") and not hasattr(bool_layer, "terminal_conditional_id")
                else [bool_layer]
            )
            mismatched_bool_ops = [
                op for op in bool_ops if getattr(op, "terminal_conditional_id", None) != event.id
            ]
            if mismatched_bool_ops:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers includes {bool_label!r} but "
                    f"{bool_label}.terminal_conditional_id="
                    f"{getattr(mismatched_bool_ops[0], 'terminal_conditional_id', None)}",
                )

    # Invariant 9: Layer conditional aggregate views match pass-level data.
    for layer_log in ml.layer_logs.values():
        expected_stack_order: list[list[tuple[int, str]]] = []
        expected_stack_ops: dict[tuple[tuple[int, str], ...], list[int]] = {}
        for call_index, pass_log in sorted(layer_log.ops.items()):
            stack_signature = tuple(pass_log.conditional_branch_stack)
            if stack_signature not in expected_stack_ops:
                expected_stack_order.append(list(pass_log.conditional_branch_stack))
                expected_stack_ops[stack_signature] = []
            expected_stack_ops[stack_signature].append(call_index)

        if layer_log.conditional_role_stacks != expected_stack_order:
            _fail_conditional_invariant(
                name,
                9,
                f"Layer {layer_log.layer_label}.conditional_role_stacks="
                f"{layer_log.conditional_role_stacks} != expected {expected_stack_order}",
            )
        if layer_log.conditional_branch_stack_ops != expected_stack_ops:
            _fail_conditional_invariant(
                name,
                9,
                f"Layer {layer_log.layer_label}.conditional_branch_stack_ops="
                f"{layer_log.conditional_branch_stack_ops} != expected "
                f"{expected_stack_ops}",
            )

    # Invariant 10: rolled conditional_edge_call_indices reference known
    # layer-level arm-entry edges. Exact pass lists live only in
    # conditional_edge_call_indices after the label remap.
    actual_arm_edges: set[tuple[str, str, int, str]] = set()
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            actual_arm_edges.add(
                (
                    _strip_pass_suffix(parent_label),
                    _strip_pass_suffix(child_label),
                    conditional_id,
                    branch_kind,
                )
            )

    for edge_key, call_indexs in ml.conditional_edge_call_indices.items():
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        if call_indexs != sorted(call_indexs) or len(call_indexs) != len(set(call_indexs)):
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_edge_call_indices[{edge_key}] has unsorted or duplicate pass list "
                f"{call_indexs}",
            )
        for call_index in call_indexs:
            if call_index < 1:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_call_indices[{edge_key}] includes invalid pass {call_index}",
                )
            actual_edge = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if actual_edge not in actual_arm_edges:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_call_indices[{edge_key}] includes pass metadata but "
                    f"conditional_arm_entry_edges has no matching layer edge",
                )

    for actual_edge in sorted(actual_arm_edges):
        parent_no_pass, child_no_pass, conditional_id, branch_kind = actual_edge
        edge_key = (parent_no_pass, child_no_pass, conditional_id, branch_kind)
        if edge_key not in ml.conditional_edge_call_indices:
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_arm_entry_edges implies rolled edge {actual_edge} but "
                f"conditional_edge_call_indices[{edge_key}]={ml.conditional_edge_call_indices.get(edge_key)}",
            )

    # Invariant 11: no transient _bool_conditional_key remains after step 5c.
    for layer in ml.layer_list:
        if hasattr(layer, "_bool_conditional_key"):
            _fail_conditional_invariant(
                name,
                11,
                f"{layer.layer_label} still has transient _bool_conditional_key attribute",
            )

    # Invariant 12: Layer conditional_arm_children is the exact pass union.
    for layer_log in ml.layer_logs.values():
        expected_children_by_cond = _expected_layer_log_child_union(layer_log)
        if layer_log.conditional_arm_children != expected_children_by_cond:
            _fail_conditional_invariant(
                name,
                12,
                f"Layer {layer_log.layer_label}.conditional_arm_children="
                f"{layer_log.conditional_arm_children} != expected pass union "
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
        parent_layer = ml.layer_logs[parent_label]
        if bool_label not in parent_layer.conditional_entry_children:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges includes ({parent_label!r}, {bool_label!r}) but "
                f"{parent_label}.conditional_entry_children={parent_layer.conditional_entry_children}",
            )

    for parent_layer in ml.layer_logs.values():
        for bool_label in parent_layer.conditional_entry_children:
            if (parent_layer.layer_label, bool_label) not in ml.conditional_branch_edges:
                _fail_conditional_invariant(
                    name,
                    13,
                    f"{parent_layer.layer_label}.conditional_entry_children includes "
                    f"{bool_label!r} but conditional_branch_edges={ml.conditional_branch_edges}",
                )

    # Invariant 14: conditional arm-entry edges correspond to real graph edges.
    # Invariants 1-2 tie the THEN/ELIF/ELSE child views to the arm-entry edges;
    # this check closes the loop by tying those edges to the actual rolled
    # parent->child topology, so conditional edge metadata can never reference
    # an edge that does not exist in the graph.
    rolled_graph_edges: set[tuple[str, str]] = set()
    for layer in ml.layer_list:
        for child_label in layer.children:
            rolled_graph_edges.add((layer.layer_label, _strip_pass_suffix(child_label)))
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            rolled_edge = (
                _strip_pass_suffix(parent_label),
                _strip_pass_suffix(child_label),
            )
            if rolled_edge not in rolled_graph_edges:
                _fail_conditional_invariant(
                    name,
                    14,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] includes "
                    f"({parent_label!r}, {child_label!r}) but the graph has no "
                    f"{rolled_edge[0]} -> {rolled_edge[1]} edge",
                )

    # Invariant 15: per-op conditional-branch membership records agree.
    # ``conditional_branch_stack`` is the canonical per-op record of arm
    # membership; the depth counter and any 'body' roles in
    # ``in_conditionals`` must be consistent with it.
    event_id_by_bool_label: dict[str, int] = {}
    for event in ml.conditional_records:
        for bool_label in event.bool_layers:
            event_id_by_bool_label[bool_label] = event.id
    event_id_by_conditional_id: dict[str, int] = {}
    branch_kind_by_conditional_arm: dict[tuple[str, int], str] = {}
    for conditional in getattr(ml, "conditionals", []) or []:
        for arm in conditional.arms:
            if arm.terminal_bool_op_label is not None:
                event_id = event_id_by_bool_label.get(arm.terminal_bool_op_label)
                if event_id is not None:
                    event_id_by_conditional_id[conditional.id] = event_id
                    break
        for arm_index, arm in enumerate(conditional.arms):
            if arm.kind == "elif":
                branch_kind_by_conditional_arm[(conditional.id, arm_index)] = f"elif_{arm_index}"
            else:
                branch_kind_by_conditional_arm[(conditional.id, arm_index)] = arm.kind
    stack_entries_by_layer_label: dict[str, set[tuple[int, str]]] = {}
    for layer in ml.layer_list:
        stack_entries_by_layer_label.setdefault(layer.layer_label, set()).update(
            layer.conditional_branch_stack
        )

    for layer in ml.layer_list:
        if layer.conditional_branch_depth != len(layer.conditional_branch_stack):
            _fail_conditional_invariant(
                name,
                15,
                f"{layer.label} has conditional_branch_depth="
                f"{layer.conditional_branch_depth} but len(conditional_branch_stack)="
                f"{len(layer.conditional_branch_stack)}",
            )
        has_body_role = any(role.role == "body" for role in (layer.in_conditionals or []))
        if has_body_role and not layer.conditional_branch_stack:
            _fail_conditional_invariant(
                name,
                15,
                f"{layer.label} has a 'body' conditional role in in_conditionals but an "
                f"empty conditional_branch_stack",
            )
        stack_entries = set(layer.conditional_branch_stack)
        for role in layer.in_conditionals or []:
            if role.role != "body":
                continue
            expected_event_id = event_id_by_conditional_id.get(role.conditional_id)
            if expected_event_id is None:
                _fail_conditional_invariant(
                    name,
                    15,
                    f"{layer.label} has body role conditional_id={role.conditional_id!r} "
                    f"but no matching conditional event was found",
                )
            expected_branch_kind = branch_kind_by_conditional_arm.get(
                (role.conditional_id, role.arm_index), role.arm_kind
            )
            expected_stack_entry = (expected_event_id, expected_branch_kind)
            layer_stack_entries = stack_entries_by_layer_label.get(layer.layer_label, set())
            if (
                expected_stack_entry not in stack_entries
                and expected_stack_entry not in layer_stack_entries
            ):
                _fail_conditional_invariant(
                    name,
                    15,
                    f"{layer.label} has body role conditional_id={role.conditional_id!r} "
                    f"arm_kind={role.arm_kind!r} but conditional_branch_stack="
                    f"{layer.conditional_branch_stack}; expected entry {expected_stack_entry}",
                )


# ---------------------------------------------------------------------------
# G. Op ↔ Layer cross-references
# ---------------------------------------------------------------------------


def _check_layer_pass_to_layer_log_xrefs(ml: "Trace") -> None:
    """Check G: Op <-> Layer cross-references.

    Validates:
    - Layer key matches its layer_label.
    - ops dict keys are contiguous {1..N}.
    - Each Op's call_index matches its dict key.
    - Each Op's layer_label matches the parent Layer's label.
    """
    name = "layer_pass_layer_log_xrefs"

    for ll_label, ll in ml.layer_logs.items():
        if ll.layer_label != ll_label:
            raise MetadataInvariantError(
                name,
                f"Layer key '{ll_label}' != Layer.layer_label='{ll.layer_label}'",
            )

        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{ll_label}' ops keys={actual_keys} != expected {expected_keys}",
            )

        for call_index, lpl in ll.ops.items():
            if lpl.pass_index != call_index:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{ll_label}' pass key={call_index} but Op.pass_index={lpl.pass_index}",
                )
            if lpl.layer_label != ll.layer_label:
                raise MetadataInvariantError(
                    name,
                    f"Op '{lpl.layer_label}' layer_label="
                    f"'{lpl.layer_label}' != "
                    f"parent Layer.layer_label='{ll.layer_label}'",
                )


# ---------------------------------------------------------------------------
# H. Module ↔ Layer containment
# ---------------------------------------------------------------------------


def _check_module_layer_containment(ml: "Trace") -> None:
    """Check H: Module <-> Layer containment consistency.

    Validates forward and reverse directions:
    - Forward: Module.layer_labels exist in layer_logs; num_layers matches.
      ModuleCall.ops labels exist; input/output_layers subset of ops.
    - Reverse: each layer's module points to a valid module
      that lists the layer in its layers.
    """
    name = "module_layer_containment"
    mod_accessor = ml.modules
    label_set = set(ml.op_labels)
    no_pass_set = set(ml.layer_labels)
    all_layer_label_set = label_set | no_pass_set
    module_layer_label_sets = {
        mod_log.address: set(mod_log.layer_labels) for mod_log in mod_accessor
    }

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Module.layer_labels exist in layer_logs
        for lbl in mod_log.layer_labels:
            if lbl not in ml.layer_logs:
                raise MetadataInvariantError(
                    name,
                    f"Module '{addr}' layers contains '{lbl}' not in trace.layer_logs",
                )

        if mod_log.num_layers != len(mod_log.layer_labels):
            raise MetadataInvariantError(
                name,
                f"Module '{addr}': num_layers={mod_log.num_layers} != "
                f"len(layer_labels)={len(mod_log.layer_labels)}",
            )

        # ModuleCall checks
        # mpl.ops may contain pass-qualified labels OR no-pass labels
        # (e.g., root module in recurrent models uses no-pass labels).
        for call_index, mpl in mod_log.ops.items():
            for lbl in mpl.ops:
                if lbl not in label_set and lbl not in no_pass_set:
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' ops contains "
                        f"'{lbl}' not in layer_labels or layer_labels",
                    )

            if mpl.num_layers != len(mpl.ops):
                raise MetadataInvariantError(
                    name,
                    f"ModuleCall '{addr}:{call_index}': "
                    f"num_layers={mpl.num_layers} != len(ops)={len(mpl.ops)}",
                )

            # input/output layers subset of ops (using both pass-qualified
            # and no-pass labels to handle recurrent models)
            for sub_attr in ("input_layers", "output_layers"):
                sub_list = getattr(mpl, sub_attr)
                sub_set = set(sub_list)
                extra = sub_set - all_layer_label_set
                if extra:
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' "
                        f"{sub_attr} has labels not in layers: {extra}",
                    )

    # Reverse check: layer's module exists in modules
    for lpl in ml.layer_list:
        cmo = lpl.module
        if cmo:
            # module may include pass suffix (e.g. 'fc:1')
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            try:
                mod_accessor[cmo_addr]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' module='{cmo}' "
                    f"(addr='{cmo_addr}') not found in module accessor",
                )
            module_layer_labels = module_layer_label_sets.get(cmo_addr, set())
            if lpl.layer_label not in module_layer_labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' (no_pass='{lpl.layer_label}') "
                    f"not in Module '{cmo_addr}'.layers",
                )


# ---------------------------------------------------------------------------
# I. Module hierarchy consistency
# ---------------------------------------------------------------------------


def _check_module_hierarchy(ml: "Trace") -> None:
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
                parent: Module = mod_accessor[mod_log.address_parent]  # type: ignore[assignment]
            except (KeyError, IndexError):
                # Parent module may be a container (ModuleList, ModuleDict)
                # that is never called during the forward pass, so no
                # Module exists.  Skip rather than error.
                parent = None  # type: ignore[assignment]
            if parent is not None and addr not in parent.address_children:
                # For shared modules, addr may be an alias that the parent
                # lists under a different address prefix.  Check if any of the
                # parent's address_children resolve to the same Module.
                if not mod_log.has_multiple_addresses:
                    raise MetadataInvariantError(
                        name,
                        f"Module '{addr}' has address_parent='{mod_log.address_parent}' "
                        f"but parent doesn't list it in address_children",
                    )

        for child_addr in mod_log.address_children:
            try:
                child: Module = mod_accessor[child_addr]  # type: ignore[assignment]
            except (KeyError, IndexError):
                # Static children may not have been invoked during the forward
                # pass, so no Module exists.  Skip rather than error.
                continue
            if child.address_parent != addr:
                # For shared modules (same nn.Module registered under multiple
                # addresses), the child's address_parent refers to its primary
                # alias's parent, which may differ from the current parent addr.
                # This is expected — there is only one Module per module
                # instance, so address_parent always reflects the primary path.
                if not child.has_multiple_addresses:
                    raise MetadataInvariantError(
                        name,
                        f"Module '{addr}' lists '{child_addr}' as address_child, "
                        f"but child's address_parent='{child.address_parent}'",
                    )

        # Module pass consistency
        if len(mod_log.ops) != mod_log.num_calls:
            raise MetadataInvariantError(
                name,
                f"Module '{addr}': len(ops)={len(mod_log.ops)} != num_calls={mod_log.num_calls}",
            )

        expected_keys = set(range(1, mod_log.num_calls + 1))
        actual_keys = set(mod_log.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Module '{addr}' pass keys={actual_keys} != expected {expected_keys}",
            )

        # Call hierarchy: parent exists
        for call_index, mpl in mod_log.ops.items():
            if mpl.call_parent is not None:
                try:
                    mod_accessor[mpl.call_parent]
                except (KeyError, IndexError):
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' call_parent="
                        f"'{mpl.call_parent}' not in module accessor",
                    )
            for cc in mpl.call_children:
                try:
                    mod_accessor[cc]
                except (KeyError, IndexError):
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' call_children "
                        f"contains '{cc}' not in module accessor",
                    )


# ---------------------------------------------------------------------------
# J. Param ↔ Layer ↔ Module cross-references
# ---------------------------------------------------------------------------


def _check_param_xrefs(ml: "Trace") -> None:
    """Check J: Param <-> Layer <-> Module cross-references.

    Validates:
    - Param.used_by_ops labels are valid op labels.
    - Param.used_by_layers labels are valid layer labels.
    - uses_params == True implies _param_logs is non-empty.
    - layers_with_params values are valid layer labels.
    """
    name = "param_xrefs"
    label_set = set(ml.layer_labels)
    op_label_set = set(ml.op_labels)
    mod_accessor = ml.modules

    for param in ml.param_logs:
        for lbl in param.used_by_ops:
            if lbl not in op_label_set:
                raise MetadataInvariantError(
                    name,
                    f"Param '{param.address}' used_by_ops contains '{lbl}' not in op_labels",
                )
        # used_by_layers exist
        for lbl in param.used_by_layers:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Param '{param.address}' used_by_layers contains '{lbl}' not in layer_labels",
                )

        # address exists (skip for conditional models where the module
        # was never called, e.g. MoE routing that skips some experts)
        try:
            mod_accessor[param.address]
        except (KeyError, IndexError):
            pass  # Module was never invoked during forward pass

    # uses_params forward check
    for lpl in ml.layer_list:
        if lpl.uses_params:
            if not lpl._param_logs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' has uses_params=True but _param_logs is empty",
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


def _check_buffer_xrefs(ml: "Trace") -> None:
    """Check K: buffer layer and Buffer cross-references.

    Validates:
    - buffer_layers list entries are valid layer labels.
    - Buffer objects have non-empty address and valid address.
    """
    name = "buffer_xrefs"
    label_set = set(ml.layer_labels)

    for lbl in ml.buffer_layers:
        if lbl not in label_set:
            raise MetadataInvariantError(
                name, f"buffer_layers contains '{lbl}' not in layer_labels"
            )

    # Check Buffer objects via buffer accessor
    if hasattr(ml, "_buffer_accessor") and ml._buffer_accessor is not None:
        for buf in ml.buffers:
            if not buf.address:
                raise MetadataInvariantError(
                    name,
                    f"Buffer '{buf.layer_label}' has empty address",
                )
            if not buf.versions:
                raise MetadataInvariantError(
                    name,
                    f"Buffer '{buf.address}' has no version nodes",
                )
            # address references a valid module or an ancestor does.
            # Buffers may live on modules that were never entered during the
            # forward pass (e.g. anchor_generator creates buffers but they're
            # consumed by parent code), or in non-module containers
            # (ParameterList, top-level attrs like "rope"). Accept the buffer
            # if any ancestor module is in the accessor.
            addr = buf.address
            found_ancestor = addr in ml.modules
            while not found_ancestor and "." in addr:
                addr = addr.rsplit(".", 1)[0]
                found_ancestor = addr in ml.modules
            if not found_ancestor and "" not in ml.modules:
                raise MetadataInvariantError(
                    name,
                    f"Buffer '{buf.layer_label}' address="
                    f"'{buf.address}' — no ancestor found in "
                    f"module accessor",
                )


# ---------------------------------------------------------------------------
# L. Equivalence group symmetry
# ---------------------------------------------------------------------------


def _check_equivalence_symmetry(ml: "Trace") -> None:
    """Check L: op_equivalence_classes groups reference valid Op labels.

    Validates:
    - Each equivalence set value is actually a set.
    - All labels in equivalence sets exist in op_labels.
    """
    name = "equivalence_symmetry"
    label_set = set(ml.op_labels)

    # op_equivalence_classes is keyed by equivalence type descriptors (not Op labels),
    # with values being sets of Op labels in that equivalence group.
    for eq_type, equiv_set in ml.op_equivalence_classes.items():
        if not isinstance(equiv_set, set):
            raise MetadataInvariantError(
                name,
                f"op_equivalence_classes['{eq_type}'] is not a set",
            )
        for label in equiv_set:
            if label not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"op_equivalence_classes['{eq_type}'] contains '{label}' not in op_labels",
                )

    # Each Op that appears in any equivalence group should exist.
    all_equiv_labels = set()
    for equiv_set in ml.op_equivalence_classes.values():
        all_equiv_labels.update(equiv_set)
    extra = all_equiv_labels - label_set
    if extra:
        raise MetadataInvariantError(
            name,
            f"op_equivalence_classes contains labels not in op_labels: {extra}",
        )


# ---------------------------------------------------------------------------
# M. Graph ordering invariants
# ---------------------------------------------------------------------------

# Raw labels (e.g., "l_42") are internal identifiers assigned during capture
# and must be replaced by human-readable labels during postprocessing.
_RAW_LABEL_PATTERN = re.compile(r"^l_\d+$")


def _check_graph_ordering(ml: "Trace") -> None:
    """Check M: graph ordering invariants.

    Validates:
    - raw_index is unique across all layers and monotonically
      increasing in layer_list order.
    - step_index is unique among computational layers (non-input, non-buffer,
      non-output).
    - Topological order: every parent's raw_index < child's.
    - No raw labels (``l_\\d+``) survive postprocessing.
    """
    name = "graph_ordering"

    # raw_index uniqueness and monotonicity
    seen_rt_nums: dict[int, str] = {}
    prev_rt = -1
    for lpl in ml.layer_list:
        rt = lpl.raw_index
        if rt in seen_rt_nums:
            raise MetadataInvariantError(
                name,
                f"Duplicate raw_index={rt}: '{seen_rt_nums[rt]}' and '{lpl.layer_label}'",
            )
        seen_rt_nums[rt] = lpl.layer_label
        if rt <= prev_rt:
            raise MetadataInvariantError(
                name,
                f"raw_index not monotonically increasing: "
                f"{prev_rt} then {rt} at '{lpl.layer_label}'",
            )
        prev_rt = rt

    # step_index uniqueness among computational layers
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)
    output_set = set(ml.output_layers)
    seen_op_nums: dict[int, str] = {}
    for lpl in ml.layer_list:
        label = lpl.layer_label
        if label in input_set or label in buffer_set or label in output_set:
            continue
        op = lpl.step_index
        if op is not None:
            if op in seen_op_nums:
                raise MetadataInvariantError(
                    name,
                    f"Duplicate step_index={op}: '{seen_op_nums[op]}' and '{label}'",
                )
            seen_op_nums[op] = label

    # Topological order: parent.raw_index < child.raw_index
    rt_map = {lpl.layer_label: lpl.raw_index for lpl in ml.layer_list}
    for lpl in ml.layer_list:
        for p in lpl.parents:
            if rt_map.get(p, -1) >= lpl.raw_index:
                raise MetadataInvariantError(
                    name,
                    f"Topological violation: parent '{p}' (rt={rt_map.get(p)}) "
                    f">= child '{lpl.layer_label}' (rt={lpl.raw_index})",
                )

    # No raw labels survive postprocessing
    for label in ml.layer_labels:
        if _RAW_LABEL_PATTERN.match(label):
            raise MetadataInvariantError(name, f"Raw label '{label}' survived postprocessing")


# ---------------------------------------------------------------------------
# N. Layer equivalence / loop detection invariants
# ---------------------------------------------------------------------------


def _check_loop_detection_invariants(ml: "Trace") -> None:
    """Check N: loop detection / recurrent_ops invariants.

    Validates per-layer:
    - recurrent_ops is non-empty and includes self.
    - Symmetry: all members agree on the same group.
    - All members share: layer_label, equivalence_class,
      func_name (for computational layers).
    - num_passes == len(recurrent_ops).
    - Pass numbering within group is contiguous {1..N}.

    Validates cross-layer:
    - Parameter sharing rule: layers with same (func_name,
      sorted(_param_barcodes)) must share layer_label.
    - Equivalence group consistency: all members of a recurrent_ops
      group belong to the same Trace.op_equivalence_classes set.

    Note: subgraph-level adjacency (Rule 3 from loop_detection.py) cannot
    be verified post-hoc from metadata alone.
    """
    name = "loop_detection"
    label_set = set(ml.op_labels)

    # Build same-layer groups from the authoritative recurrent_ops lists
    # Key: frozenset of labels, Value: list of OpLogs in the group
    groups_seen: dict[frozenset[str], list[str]] = {}

    for lpl in ml.layer_list:
        slo = lpl.recurrent_ops
        if not slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' has empty recurrent_ops",
            )

        # All members in recurrent_ops must exist
        for member in slo:
            if member not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' recurrent_ops contains '{member}' not in op_labels",
                )

        # Self-inclusion
        if lpl.label not in slo:
            raise MetadataInvariantError(
                name,
                f"Op '{lpl.label}' not in its own recurrent_ops",
            )

        # Symmetry: all members agree on the group
        for member_label in slo:
            member = ml[member_label]
            if set(member.recurrent_ops) != set(slo):
                raise MetadataInvariantError(
                    name,
                    f"Asymmetric recurrent_ops: '{lpl.layer_label}' has "
                    f"{sorted(slo)} but '{member_label}' has "
                    f"{sorted(member.recurrent_ops)}",
                )

        # All members share layer_label
        for member_label in slo:
            member = ml[member_label]
            if member.layer_label != lpl.layer_label:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_ops inconsistency: '{lpl.layer_label}' "
                    f"(no_pass='{lpl.layer_label}') and '{member_label}' "
                    f"(no_pass='{member.layer_label}') differ",
                )

        # All members share equivalence_class
        for member_label in slo:
            member = ml[member_label]
            if member.equivalence_class != lpl.equivalence_class:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_ops type mismatch: '{lpl.layer_label}' "
                    f"type='{lpl.equivalence_class}' vs '{member_label}' "
                    f"type='{member.equivalence_class}'",
                )

        # All members share func_name (for computational layers)
        if not (lpl.is_input or lpl.is_buffer or lpl.is_output):
            for member_label in slo:
                member = ml[member_label]
                if member.func_name != lpl.func_name:
                    raise MetadataInvariantError(
                        name,
                        f"recurrent_ops func mismatch: '{lpl.layer_label}' "
                        f"func='{lpl.func_name}' vs '{member_label}' "
                        f"func='{member.func_name}'",
                    )

        # num_passes == len(recurrent_ops)
        if lpl.num_passes != len(slo):
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}': num_passes={lpl.num_passes} "
                f"!= len(recurrent_ops)={len(slo)}",
            )

        # Pass numbering: unique {1..N}
        group_key = frozenset(slo)
        if group_key not in groups_seen:
            pass_indices = []
            for member_label in slo:
                member = ml[member_label]
                pass_indices.append(member.pass_index)
            expected = set(range(1, len(slo) + 1))
            actual = set(pass_indices)
            if actual != expected:
                raise MetadataInvariantError(
                    name,
                    f"Pass numbering for group {sorted(slo)}: expected {expected}, got {actual}",
                )
            groups_seen[group_key] = slo

    # Rule 1: Parameter sharing invariant.
    # Layers with the same func_name, identical sorted(_param_barcodes), and the
    # same output-specific equivalence class must share layer_label. The
    # equivalence class keeps distinct outputs of a multi-output parameterized op
    # from being collapsed into one logical layer.
    param_groups: dict[tuple[str, tuple[str, ...], str], list[Op]] = defaultdict(list)
    for lpl in ml.layer_list:
        if lpl.uses_params and lpl._param_barcodes:
            key = (lpl.func_name, tuple(sorted(lpl._param_barcodes)), lpl.equivalence_class)
            param_groups[key].append(lpl)

    for param_key, layers in param_groups.items():
        if len(layers) > 1:
            no_call_labels = {lpl.layer_label for lpl in layers}
            if len(no_call_labels) > 1:
                raise MetadataInvariantError(
                    name,
                    f"Param sharing violation: layers with same param barcodes "
                    f"{param_key} have different layer_label: {no_call_labels}",
                )

    # Equivalence group ↔ same_layer consistency: all members of a
    # recurrent_ops group must belong to the same equivalence set.
    # Note: Trace.op_equivalence_classes keys use the pre-module-suffix type
    # (from loop_detection), while per-layer equivalence_class has
    # a module suffix appended by control_flow.py. So we check group membership
    # consistency, not exact key matching.
    op_label_to_equiv_key: dict[str, str] = {}
    for eq_type, equiv_set in ml.op_equivalence_classes.items():
        for label in equiv_set:
            op_label_to_equiv_key[label] = eq_type

    for group_key in groups_seen:
        slo = list(group_key)
        if len(slo) <= 1:
            continue
        # All members of a same-layer group should be in the same equivalence set
        equiv_keys = set()
        for member_label in slo:
            member = ml[member_label]
            if member.label in op_label_to_equiv_key:
                equiv_keys.add(op_label_to_equiv_key[member.label])
        equiv_stems = {re.sub(r"_outindex\d+$", "", equiv_key) for equiv_key in equiv_keys}
        if len(equiv_keys) > 1 and len(equiv_stems) > 1:
            raise MetadataInvariantError(
                name,
                f"recurrent_ops group {sorted(slo)} spans multiple equivalence types: {equiv_keys}",
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


def _check_distance_invariants(ml: "Trace") -> None:
    """Check O: distance and reachability invariants.

    Only runs when ``mark_layer_depths`` was enabled during logging.

    Validates:
    - min_distance <= max_distance for both input and output distances.
    - Input layers have distance_from_input == 0.
    - Output layers have distance_from_output == 0.
    - has_input_ancestor <-> input_ancestors is non-empty.
    - has_output_descendant <-> output_descendants is non-empty.
    - input_ancestors subset of input_layers; output_descendants subset of
      output_layers.
    """
    if not ml.mark_layer_depths:
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
        if lpl.min_distance_to_output is not None and lpl.max_distance_to_output is not None:
            if lpl.min_distance_to_output > lpl.max_distance_to_output:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_to_output="
                    f"{lpl.min_distance_to_output} > max={lpl.max_distance_to_output}",
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
            if lpl.min_distance_to_output != 0 or lpl.max_distance_to_output != 0:
                raise MetadataInvariantError(
                    name,
                    f"Output layer '{label}': distance_from_output should be 0, got "
                    f"min={lpl.min_distance_to_output}, max={lpl.max_distance_to_output}",
                )

        # has_input_ancestor ↔ input_ancestors non-empty
        has_ancestors = len(lpl.input_ancestors) > 0
        if lpl.has_input_ancestor != has_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_input_ancestor={lpl.has_input_ancestor} but "
                f"len(input_ancestors)={len(lpl.input_ancestors)}",
            )

        # has_output_descendant ↔ output_descendants non-empty
        has_descendents = len(lpl.output_descendants) > 0
        if lpl.has_output_descendant != has_descendents:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_output_descendant={lpl.has_output_descendant} but "
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


def _check_graph_connectivity(ml: "Trace") -> None:
    """Check P: graph connectivity invariants.

    Validates:
    - Every non-input, non-buffer, non-internally-initialized, non-output
      layer has at least one parent (no dangling computational nodes).
    - _orphan_labels (removed during postprocessing) do NOT appear in the
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
            and not lpl.is_internal_source
            and not lpl.is_output
            and len(lpl.parents) == 0
        ):
            raise MetadataInvariantError(
                name,
                f"Layer '{label}' has no parents but is not input, buffer, "
                f"internally initialized, or output",
            )

    # _orphan_labels is a subset of all known labels (pre-removal)
    orphan_set = set(ml._orphan_labels)
    # Orphans should NOT appear in the active layer_list (they were removed)
    orphan_in_list = orphan_set & label_set
    if orphan_in_list:
        raise MetadataInvariantError(
            name,
            f"_orphan_labels contains labels still in layer_labels: {orphan_in_list}",
        )


# ---------------------------------------------------------------------------
# Q. Module containment logical consistency
# ---------------------------------------------------------------------------


def _check_module_containment_logic(ml: "Trace") -> None:
    """Check Q: module containment logical consistency.

    Validates:
    - Address tree is acyclic (walking address_parent chain reaches None
      without revisiting a node).
    - Root module 'self' has address_depth == 0; others have
      address_depth == addr.count('.') + 1.
    - Per-layer modules (call nesting stack):
      - Last element matches module.
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
                parent_mod: Module = mod_accessor[current]  # type: ignore[assignment]
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
                    f"Module '{addr}': address_depth={mod_log.address_depth} "
                    f"!= expected {expected_depth} (addr.count('.')+1)",
                )

    # Per-layer: modules path validity
    # Format after postprocessing: list of "addr:pass" strings, ordered from
    # outermost enclosing submodule to innermost. Does NOT include "self".
    for lpl in ml.layer_list:
        nested = lpl.modules
        if not nested:
            continue

        # Leaf consistency: last element matches module
        if lpl.module is not None:
            if nested[-1] != lpl.module:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': last nested module '{nested[-1]}' "
                    f"!= module '{lpl.module}'",
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


def _check_lookup_key_consistency(ml: "Trace") -> None:
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

    # All final labels are valid lookup labels. Multi-pass raw labels map to
    # pass-qualified Op labels, while single-pass raw labels may map to Layer
    # labels for compatibility lookup.
    label_set = set(ml.layer_labels) | set(ml.op_labels)
    for final in raw_fwd.values():
        if final not in label_set:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels maps to '{final}' which is not a valid label",
            )
