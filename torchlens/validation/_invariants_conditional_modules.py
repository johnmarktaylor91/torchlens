"""Conditional membership and module hierarchy invariants."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data_classes.module import Module


if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _check_module_call_boundary_and_tree,
        _fail_conditional_invariant,
        _strip_pass_suffix,
    )

__all__ = (
    "_check_conditional_arm_edges_match_graph",
    "_check_conditional_branch_membership_records",
    "_check_conditional_public_accessor_summary",
    "_check_layer_pass_to_layer_log_xrefs",
    "_check_pass_count_consistency",
    "_check_module_layer_containment",
    "_check_module_hierarchy",
)


def _check_conditional_arm_edges_match_graph(
    ml: Trace,
    name: str,
) -> None:
    """Check conditional arm-entry edges correspond to graph edges.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

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


def _check_conditional_branch_membership_records(
    ml: Trace,
    name: str,
) -> None:
    """Check per-op conditional branch membership records agree.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

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


def _check_conditional_public_accessor_summary(
    ml: Trace,
    name: str,
) -> None:
    """Check public conditional ids and fired-arm summaries are honest.

    Parameters
    ----------
    ml:
        Trace containing finalized public conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    conditionals = list(getattr(ml, "conditionals", []) or [])
    conditional_ids = [conditional.id for conditional in conditionals]
    if len(conditional_ids) != len(set(conditional_ids)):
        _fail_conditional_invariant(
            name,
            16,
            f"Trace.conditionals contains duplicate ids: {conditional_ids}",
        )

    for conditional in conditionals:
        fired_arm_indices = [index for index, arm in enumerate(conditional.arms) if arm.fired]
        if len(fired_arm_indices) == 1:
            fired_arm_index = fired_arm_indices[0]
            expected_kind = conditional.arms[fired_arm_index].kind
            if conditional.fired_arm_index != fired_arm_index:
                _fail_conditional_invariant(
                    name,
                    16,
                    f"{conditional.id} has fired_arm_index={conditional.fired_arm_index} but "
                    f"the only fired arm is index {fired_arm_index}",
                )
            if conditional.fired_arm_kind != expected_kind:
                _fail_conditional_invariant(
                    name,
                    16,
                    f"{conditional.id} has fired_arm_kind={conditional.fired_arm_kind!r} but "
                    f"the only fired arm kind is {expected_kind!r}",
                )
            continue

        if conditional.fired_arm_index is not None or conditional.fired_arm_kind is not None:
            _fail_conditional_invariant(
                name,
                16,
                f"{conditional.id} has fired_arm_index={conditional.fired_arm_index} and "
                f"fired_arm_kind={conditional.fired_arm_kind!r} despite "
                f"{len(fired_arm_indices)} fired arms",
            )


def _check_layer_pass_to_layer_log_xrefs(ml: Trace) -> None:
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


def _check_pass_count_consistency(ml: Trace) -> None:
    """Check pass-count consistency across multi-pass layer records.

    Parameters
    ----------
    ml:
        Trace whose per-layer pass maps should be checked.

    Raises
    ------
    MetadataInvariantError
        If layer call counts, op maps, and op pass metadata disagree.
    """

    name = "pass_count_consistency"
    layer_num_calls = getattr(ml, "layer_num_calls", {}) or {}
    for layer_label, layer_log in ml.layer_logs.items():
        ops = getattr(layer_log, "ops", {}) or {}
        expected_keys = set(range(1, getattr(layer_log, "num_passes", 0) + 1))
        # OpAccessor iteration yields Ops (C02 basis fix); the pass-index key
        # map stays reachable through the explicit dict protocol.
        actual_keys = set(ops.keys()) if hasattr(ops, "keys") else set(ops)
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{layer_label}' ops keys {sorted(actual_keys)!r} do not match "
                f"num_passes={getattr(layer_log, 'num_passes', None)!r}",
            )
        for pass_index, op in ops.items():
            if getattr(op, "pass_index", None) != pass_index:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' stores op with pass_index="
                    f"{getattr(op, 'pass_index', None)!r} under key {pass_index!r}",
                )
            if getattr(op, "num_passes", None) != getattr(layer_log, "num_passes", None):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' op {getattr(op, 'label', pass_index)!r} has "
                    f"num_passes={getattr(op, 'num_passes', None)!r}, expected "
                    f"{getattr(layer_log, 'num_passes', None)!r}",
                )
        if layer_label in layer_num_calls and layer_num_calls[layer_label] != layer_log.num_passes:
            raise MetadataInvariantError(
                name,
                f"layer_num_calls[{layer_label!r}]={layer_num_calls[layer_label]!r} "
                f"!= Layer.num_passes={layer_log.num_passes!r}",
            )


def _check_module_layer_containment(ml: Trace) -> None:
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
                        f"'{lbl}' not in op_labels or layer_labels",
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


def _check_module_hierarchy(ml: Trace) -> None:
    """Check I: module address tree consistency and pass structure.

    Precondition contract: rich ModuleCall checks run only for materialized
    real ModuleCall records in called modules. Static uncalled child modules
    are still handled by the existing address-level exceptions. ModuleCall
    boundary inputs are allowed to be external to the call because they are
    commonly produced in the parent scope; only output ops are required to be
    produced inside the call's own ``ops`` list.

    Validates:
    - Root module 'self' exists.
    - Address parent-child bidirectionality (with exemptions for shared
      modules, where aliases may diverge from the primary path).
    - Container modules (ModuleList) that were never called may not have
      ModuleLogs -- skip rather than error.
    - Pass dict keys are contiguous {1..N} and match num_passes.
    - call_parent and call_children reference valid modules.
    - ModuleCall labels, output boundaries, call-tree links, stacks, and
      output structures are internally consistent.
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
                parent: Module = mod_accessor[mod_log.address_parent]
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
                child: Module = mod_accessor[child_addr]
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

            _check_module_call_boundary_and_tree(ml, mpl, name)
