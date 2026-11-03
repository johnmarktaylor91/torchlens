"""Leaf helpers of ``finalization._build_conditional_records``.

Public conditional ids, the bool op that decided a branch, role-label merging and
the evaluation entry edge. This module imports nothing from the package at runtime.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import ConditionalRoleRef, Trace


def _make_public_conditional_ids(condition_events: list[Any]) -> dict[int, str]:
    """Build unique public ids for conditional records.

    Parameters
    ----------
    condition_events:
        Conditional events in trace order.

    Returns
    -------
    dict[int, str]
        Mapping from dense ``ConditionalEvent.id`` to the public conditional id
        exposed through ``Trace.graph.conditionals`` and role refs.
    """

    id_counts: dict[str, int] = defaultdict(int)
    base_ids_by_event_id: dict[int, str] = {}
    for event in condition_events:
        terminal_bool_label = event.bool_layers[0] if event.bool_layers else str(event.id)
        base_id = f"cond_{terminal_bool_label}"
        base_ids_by_event_id[event.id] = base_id
        id_counts[base_id] += 1

    return {
        event.id: (
            base_ids_by_event_id[event.id]
            if id_counts[base_ids_by_event_id[event.id]] == 1
            else f"{base_ids_by_event_id[event.id]}__event_{event.id}"
        )
        for event in condition_events
    }


def _resolve_conditional_bool_op(
    self: Trace,
    terminal_bool: str,
    witnessed_indices: list[int],
    bool_layers_raw: list[str] | None,
) -> Any | None:
    """Resolve the op whose ``bool_value`` backs one arm's single evaluation.

    Parameters
    ----------
    self:
        Trace whose lookup keys resolve conditional bool ops.
    terminal_bool:
        Renamed public label of the arm's terminal bool layer.
    witnessed_indices:
        Indices into the owning event's ``bool_layers`` witnessed for this
        arm's test. Only single-witness arms reach this resolver.
    bool_layers_raw:
        Index-aligned RAW labels stashed by phase 5c, or ``None`` for
        legacy/degraded events.

    Returns
    -------
    Any | None
        The exact evaluating op. The raw label is preferred because it is
        unique per pass: a rolled multi-pass bool layer renames every pass to
        ONE base label, and an unqualified ``layer_dict_all_keys`` lookup
        resolves last-writer-wins to an arbitrary pass (round-24 condbranch
        seal, S3). Falls back to the renamed label for legacy events.
    """

    if (
        bool_layers_raw is not None
        and len(witnessed_indices) == 1
        and 0 <= witnessed_indices[0] < len(bool_layers_raw)
    ):
        raw_op = self._layer_dict_all_keys.get(bool_layers_raw[witnessed_indices[0]])
        if raw_op is not None:
            return raw_op
    return self._layer_dict_all_keys.get(terminal_bool)


def _merge_conditional_role_labels(
    role_labels_by_cond_arm: dict[tuple[str, int, str], list[str]],
    role_key: tuple[str, int, str],
    labels: list[str],
) -> None:
    """Merge one conditional-role label list without dropping prior members.

    Parameters
    ----------
    role_labels_by_cond_arm:
        Mapping from ``(conditional_id, arm_index, role)`` to layer labels.
    role_key:
        Key naming the arm-role bucket to update.
    labels:
        Labels to merge into the bucket.
    """

    merged = dict.fromkeys([*role_labels_by_cond_arm.get(role_key, []), *labels])
    role_labels_by_cond_arm[role_key] = list(merged)


def _find_conditional_evaluation_entry_edge(
    self: Trace,
    bool_layers: list[str],
) -> tuple[str, str] | None:
    """Return the main-graph entry edge for a conditional evaluation chain.

    Parameters
    ----------
    self:
        Trace whose finalized layer graph contains the conditional evaluation.
    bool_layers:
        Bool-producing layer labels for one conditional event, in evaluation order.

    Returns
    -------
    tuple[str, str] | None
        Edge from the upstream main-graph parent into the first bool-producing
        layer, or ``None`` when no distinct upstream parent is available.
    """

    if not bool_layers:
        return None

    first_bool_label = bool_layers[0]
    if first_bool_label not in self._layer_dict_all_keys:
        return None

    visited_labels: set[str] = set()
    parent_queue: deque[str] = deque(self._layer_dict_all_keys[first_bool_label].parents)
    while parent_queue:
        parent_label = parent_queue.popleft()
        if parent_label in visited_labels:
            continue
        visited_labels.add(parent_label)
        if parent_label not in self._layer_dict_all_keys:
            continue

        parent_layer = self._layer_dict_all_keys[parent_label]
        if parent_label != first_bool_label and parent_layer.graph.has_output_descendant:
            return (parent_label, first_bool_label)
        parent_queue.extend(parent_layer.parents)

    return None


def _roles_by_label(
    conditionals: list[Any],
    role_labels_by_cond_arm: dict[tuple[str, int, str], list[str]],
    role_ref: type[ConditionalRoleRef],
) -> dict[str, list[ConditionalRoleRef]]:
    """Return each layer label's conditional roles (``role_ref`` instances) per arm bucket.

    Inverted role index (r8 R60-7): the per-layer triple loop with an innermost
    list-membership check was O(N_layers x Sum|role lists|) whenever a
    conditional's entry-edge children were numerous (an ``if`` in a hot unrolled
    loop). One pass over the role buckets builds the same per-layer role lists in
    the same (conditional, arm, evaluation-then-body) order; per-bucket label
    dedup mirrors the old one-role-per-membership semantics.
    """

    roles_by_label: dict[str, list[ConditionalRoleRef]] = {}
    for conditional in conditionals:
        for arm_index, arm in enumerate(conditional.arms):
            for role in ("evaluation", "body"):
                bucket = role_labels_by_cond_arm.get((conditional.id, arm_index, role), [])
                bucket_seen: set[str] = set()
                for label in bucket:
                    if label in bucket_seen:
                        continue
                    bucket_seen.add(label)
                    roles_by_label.setdefault(label, []).append(
                        role_ref(conditional.id, arm_index, arm.kind, role)
                    )
    return roles_by_label


def roles_held_by_pass(
    op: Op,
    layer_roles: list[ConditionalRoleRef],
    arm_stack_entries: dict[tuple[str, int], tuple[int, str]],
) -> list[ConditionalRoleRef]:
    """Return the conditional roles one pass of a layer actually holds.

    Role buckets are keyed by the rolled layer label, so every pass of a layer
    sees the roles of all its passes. A rolled layer can mix passes inside and
    outside an arm body (the same ``lamda < 0`` evaluated as an ``if`` test and
    again in that ``if``'s body, rolled as one layer). A ``body`` role is kept
    only for a pass whose own ``conditional_branch_stack`` holds that arm, the
    per-pass record of arm membership.

    Parameters
    ----------
    op:
        One pass of a layer.
    layer_roles:
        Roles collected for the pass's rolled layer label.
    arm_stack_entries:
        ``(conditional_id, arm_index)`` to the ``(event_id, branch_kind)``
        branch-stack entry of that arm.

    Returns
    -------
    list[ConditionalRoleRef]
        The roles this pass holds, in the layer's role order.
    """

    if not layer_roles:
        return []
    pass_stack = set(op.control_flow.branch_stack or ())
    return [
        role
        for role in layer_roles
        if role.role != "body"
        or arm_stack_entries.get((role.conditional_id, role.arm_index)) in pass_stack
    ]
