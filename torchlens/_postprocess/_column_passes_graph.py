"""Column-view bodies of the graph passes: step 3's orphan flood and step 5's arms.

Each function here keeps its facade body as the fallback and runs the column
body when every raw op shares one row store and raw-label lookups resolve
through the raw layer dict (no live capture-event index, tracing not finished),
which is what ``Trace.__getitem__`` and ``Trace.__iter__`` do during the pass.
Both bodies visit the same rows in the same order and perform the same reads
and writes, so the stored result is identical (checked by per-pass store
fingerprints over the corpus and by ``tests/test_column_passes_graph.py``).

The pass entry points keep their names in ``graph_traversal`` and
``control_flow`` (``_executor`` calls them and tests patch them); only the
per-op loops live here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING, Any, NamedTuple

from .._utils._record_group_table import record_member
from ._branch_attribution import _code_context_position_key
from ._column_passes import _column_lookup, _row_or_refuse

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

__all__ = (
    "assign_conditional_branch_stacks",
    "attribute_forward_edges",
    "FloodView",
    "flood_kept_siblings",
    "flood_orphan_reachability",
    "flood_view",
    "get_gained_branch_entries",
    "materialize_derived_views",
    "reset_and_flood_if_edges",
)


def _row(trace: Trace, rows: dict[str, int], label: str) -> int:
    """Return ``label``'s row; an unknown label raises ``trace[label]``'s refusal."""

    row = rows.get(label)
    return _row_or_refuse(trace, rows, label) if row is None else row


def _graph_lookup(trace: Trace) -> Any:
    """Return ``(columns, rows_by_label)`` when the column bodies apply, else ``None``.

    The single seam every body here consults (tests force the facade through it).
    """

    return _column_lookup(trace)


# ---------------------------------------------------------------------------
# Step 3: bidirectional reachability flood
# ---------------------------------------------------------------------------


class FloodView(NamedTuple):
    """Row-keyed accessors of the step 3 floods: column readers, or the facade's.

    ``key`` maps a raw label to what the other accessors take (a row of the
    shared store, or the label itself when the column view declines).
    """

    key: Callable[[str], Any]
    children: Callable[[Any], Any]
    parents: Callable[[Any], Any]
    is_output: Callable[[Any], Any]
    mark_sink: Callable[[Any, bool], None]
    is_scalar_bool: Callable[[Any], Any]
    mark_terminal_bool: Callable[[Any, bool], None]
    func_call_id: Callable[[Any], Any]


def flood_view(trace: Trace) -> FloodView:
    """Return the step 3 accessors: column readers when they apply, else facade reads.

    Both read and write the same fields of the same op (``trace[label]``
    resolves through the raw layer dict during the pass), so the floods built
    on either store the same result.
    """

    lookup = _graph_lookup(trace)
    if lookup is not None:
        columns, rows = lookup
        return FloodView(
            rows.__getitem__,  # KeyError(label), as the raw layer dict raises
            columns.getter("children"),
            columns.getter("parents"),
            columns.getter("is_output"),
            columns.setter("is_internal_sink"),
            columns.getter("is_scalar_bool"),
            columns.setter("is_terminal_bool"),
            columns.getter("func_call_id", None),
        )
    raw = trace._raw_graph_ws.raw_layer_dict

    def key(label: str) -> str:
        """Return ``label`` once the raw layer dict knows it (its ``KeyError`` otherwise)."""

        raw[label]
        return label

    def mark_terminal_bool(label: str, value: bool) -> None:
        """Write the terminal-bool flag through the facade's storage name."""

        trace[label]._control_flow_is_terminal_bool = value

    return FloodView(
        key,
        lambda label: raw[label].children,
        lambda label: raw[label].parents,
        lambda label: raw[label].is_output,
        lambda label, value: setattr(trace[label], "is_internal_sink", value),
        lambda label: trace[label].control_flow.is_scalar_bool,
        mark_terminal_bool,
        lambda label: record_member(raw[label], "func_call_id", None),
    )


def flood_orphan_reachability(trace: Trace, node_stack: list[str], view: FloodView) -> set[str]:
    """Flood both edge directions from ``node_stack`` and log internal sinks.

    A visited non-output node with no children is logged as an internally
    terminated tensor: its sink flag is set and it is appended once to
    ``trace.internal_sink_ops`` (and, when it is a scalar bool, to
    ``trace.internally_terminated_bool_ops`` with its terminal flag set).

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    node_stack:
        Seed labels (inputs, outputs and the other flood seeds); consumed.
    view:
        The pass's accessors (:func:`flood_view`).

    Returns
    -------
    set[str]
        Every label the flood reached.
    """

    # Cognitive complexity is over 15 on purpose (hot path, once per op): the
    # sink logging is inlined, not called per visited node.
    # Shadow sets over the two trace-level list[str] sink ledgers: the former
    # per-node `label not in <list>` scans were O(k^2) in sink count.
    # The lists stay the portable source of truth.
    seen_sink_labels = set(trace.internal_sink_ops)
    seen_terminated_bool_labels = set(trace.internally_terminated_bool_ops)
    key, children, parents, is_output, mark_sink, is_scalar_bool, mark_terminal_bool, _ = view
    internal_sink_ops = trace.internal_sink_ops
    terminated_bool_ops = trace.internally_terminated_bool_ops
    nodes_seen: set[str] = set()
    while node_stack:
        tensor_label = node_stack.pop()
        nodes_seen.add(tensor_label)
        row = key(tensor_label)
        child_labels = children(row)
        if len(child_labels) == 0 and not is_output(row):
            mark_sink(row, True)
            if tensor_label not in seen_sink_labels:
                seen_sink_labels.add(tensor_label)
                internal_sink_ops.append(tensor_label)
                if is_scalar_bool(row) and tensor_label not in seen_terminated_bool_labels:
                    seen_terminated_bool_labels.add(tensor_label)
                    terminated_bool_ops.append(tensor_label)
                    mark_terminal_bool(row, True)
        # Follow BOTH directions to ensure full bidirectional reachability.
        for next_label in child_labels + parents(row):
            if next_label not in nodes_seen:
                node_stack.append(next_label)
    return nodes_seen


def flood_kept_siblings(trace: Trace, added: list[str], seen: set[str], view: FloodView) -> None:
    """Flood from siblings kept with their call group, through not-yet-seen nodes only.

    Applies the main flood's rule to what it never visited: a non-output node
    with no children is logged as an internal sink. ``seen`` grows in place.
    """

    seen_sinks = set(trace.internal_sink_ops)
    seen_bools = set(trace.internally_terminated_bool_ops)
    stack = list(added)
    while stack:
        label = stack.pop()
        row = view.key(label)
        child_labels = view.children(row)
        if len(child_labels) == 0 and not view.is_output(row):
            view.mark_sink(row, True)
            if label not in seen_sinks:
                seen_sinks.add(label)
                trace.internal_sink_ops.append(label)
                if view.is_scalar_bool(row) and label not in seen_bools:
                    seen_bools.add(label)
                    trace.internally_terminated_bool_ops.append(label)
                    view.mark_terminal_bool(row, True)
        for next_label in child_labels + view.parents(row):
            if next_label not in seen:
                seen.add(next_label)
                stack.append(next_label)


# ---------------------------------------------------------------------------
# Step 5d: backward IF-edge flood
# ---------------------------------------------------------------------------


def reset_and_flood_if_edges(trace: Trace, branch_bool_labels: list[str]) -> None:
    """Phase 5d: reset per-op IF state, then backward-flood IF edges from branch bools.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    branch_bool_labels:
        Branch-participating terminal bool labels, in execution order.
    """

    trace._graph_conditional_branch_edges = []
    lookup = _graph_lookup(trace)
    if lookup is None:
        _reset_and_flood_if_edges_facade(trace, branch_bool_labels)
        return
    columns, rows = lookup
    _reset_if_state_column(columns, rows)
    _flood_if_edges_column(trace, columns, rows, branch_bool_labels)


def _reset_if_state_column(columns: Any, rows: dict[str, int]) -> None:
    """Clear every non-orphan op's IF entry children and conditional-body flag."""

    is_orphan = columns.getter("is_orphan", False)
    put_entry_children = columns.setter("conditional_entry_children")
    # ``Op.is_in_conditional_body`` is a property whose setter stores this slot.
    put_in_body = columns.setter("_is_in_conditional_body")
    for row in rows.values():  # ``Trace.__iter__`` order during the pass
        if is_orphan(row):
            continue
        put_entry_children(row, [])
        put_in_body(row, False)


def _flood_if_edges_column(
    trace: Trace, columns: Any, rows: dict[str, int], branch_bool_labels: list[str]
) -> None:
    """Column body of the backward IF-edge flood from the branch bools.

    Cognitive complexity is over 15 on purpose (hot path): the row lookups
    are inlined so the per-op loop makes no Python call the facade avoids.
    """

    entry_children = columns.getter("conditional_entry_children")
    put_in_body = columns.setter("_is_in_conditional_body")
    parents = columns.getter("parents")
    has_output_descendant = columns.getter("has_output_descendant")
    nodes_seen: set[str] = set()
    node_stack = branch_bool_labels.copy()
    while node_stack:
        node_label = node_stack.pop()
        row = rows.get(node_label)
        if row is None:
            row = _row(trace, rows, node_label)
        if node_label in nodes_seen:
            continue
        for parent_label in parents(row):
            parent_row = rows.get(parent_label)
            if parent_row is None:
                parent_row = _row(trace, rows, parent_label)
            if has_output_descendant(parent_row):
                entry_children(parent_row).append(node_label)
                put_in_body(parent_row, False)
                nodes_seen.add(parent_label)
                trace._graph_conditional_branch_edges.append((parent_label, node_label))
            else:
                if parent_label in nodes_seen:
                    continue
                put_in_body(parent_row, True)
                node_stack.append(parent_label)

        nodes_seen.add(node_label)


def _reset_and_flood_if_edges_facade(trace: Trace, branch_bool_labels: list[str]) -> None:
    """Facade body of :func:`reset_and_flood_if_edges` for non-column traces."""

    for layer in trace:
        if record_member(layer, "is_orphan", False):
            continue
        layer._control_flow_entry_children = []
        layer._control_flow_is_in_conditional_body = False

    nodes_seen: set[str] = set()
    node_stack = branch_bool_labels.copy()
    while node_stack:
        node_label = node_stack.pop()
        node = trace[node_label]
        if node_label in nodes_seen:
            continue
        for parent_label in node.parents:
            parent_layer = trace[parent_label]
            if parent_layer.graph.has_output_descendant:
                parent_layer.control_flow.entry_children.append(node_label)
                parent_layer._control_flow_is_in_conditional_body = False
                nodes_seen.add(parent_label)
                trace._graph_conditional_branch_edges.append((parent_label, node_label))
            else:
                if parent_label in nodes_seen:
                    continue
                parent_layer._control_flow_is_in_conditional_body = True
                node_stack.append(parent_label)

        nodes_seen.add(node_label)


# ---------------------------------------------------------------------------
# Step 5e: branch stacks and forward arm edges
# ---------------------------------------------------------------------------


def assign_conditional_branch_stacks(
    trace: Trace,
    events_by_key: dict[Any, Any],
    translate: Callable[[Any, dict[Any, Any]], list[tuple[int, str]]],
) -> None:
    """Give every non-orphan op its conditional branch stack and reset its arm children.

    Ops from the same call site carry the same frame positions, and the
    attribution reads nothing else from a frame, so each distinct stack is
    attributed once per pass; every op still gets its own list.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    events_by_key:
        Structural-to-dense conditional event lookup created in phase 5c.
    translate:
        ``control_flow._translate_conditional_stack``, passed by the caller so a
        patch of that module attribute is honored.
    """

    lookup = _graph_lookup(trace)
    if lookup is None:
        _assign_conditional_branch_stacks_facade(trace, events_by_key, translate)
        return
    columns, rows = lookup
    is_orphan = columns.getter("is_orphan", False)
    code_context_of = columns.getter("code_context")
    put_stack = columns.setter("conditional_branch_stack")
    stack_of = columns.getter("conditional_branch_stack")
    put_depth = columns.setter("conditional_branch_depth")
    put_arm_children = columns.setter("conditional_arm_children")
    translated_by_stack: dict[tuple[tuple[Any, ...], ...], list[tuple[int, str]]] = {}
    for layer_label in trace._raw_graph_ws.raw_layer_labels_list:
        row = rows.get(layer_label)
        if row is None:
            row = _row(trace, rows, layer_label)
        if is_orphan(row):
            continue
        code_context = code_context_of(row)
        stack_key = _code_context_position_key(code_context)
        translated = translated_by_stack.get(stack_key)
        if translated is None:
            translated = translate(code_context, events_by_key)
            translated_by_stack[stack_key] = translated
        put_stack(row, list(translated))
        put_depth(row, len(stack_of(row)))
        put_arm_children(row, {})


def _assign_conditional_branch_stacks_facade(
    trace: Trace,
    events_by_key: dict[Any, Any],
    translate: Callable[[Any, dict[Any, Any]], list[tuple[int, str]]],
) -> None:
    """Facade body of :func:`assign_conditional_branch_stacks` for non-column traces."""

    translated_by_stack: dict[tuple[tuple[Any, ...], ...], list[tuple[int, str]]] = {}
    for layer_label in trace._raw_graph_ws.raw_layer_labels_list:
        layer = trace[layer_label]
        if record_member(layer, "is_orphan", False):
            continue
        code_context = layer.source.frames
        stack_key = _code_context_position_key(code_context)
        translated = translated_by_stack.get(stack_key)
        if translated is None:
            translated = translate(code_context, events_by_key)
            translated_by_stack[stack_key] = translated
        layer._control_flow_branch_stack = list(translated)
        layer._control_flow_branch_depth = len(layer.control_flow.branch_stack)
        layer._control_flow_arm_children = {}


def get_gained_branch_entries(
    parent_stack: list[tuple[int, str]],
    child_stack: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return child stack entries gained across one forward edge.

    Parameters
    ----------
    parent_stack:
        Parent operation branch stack, ordered outer-to-inner.
    child_stack:
        Child operation branch stack, ordered outer-to-inner.

    Returns
    -------
    List[Tuple[int, str]]
        Entries present in the child's stack beyond the shared prefix with the
        parent, preserving outer-to-inner order.
    """

    shared_prefix_len = 0
    max_shared = min(len(parent_stack), len(child_stack))
    while shared_prefix_len < max_shared:
        if parent_stack[shared_prefix_len] != child_stack[shared_prefix_len]:
            break
        shared_prefix_len += 1
    return child_stack[shared_prefix_len:]


def attribute_forward_edges(trace: Trace) -> None:
    """Phase 5e body: attribute every forward edge that enters a conditional arm.

    Records the entered arm's child on the parent, the raw entry edge per
    ``(conditional_id, branch_kind)`` and the child's pass number per call-site
    edge, then stores both trace-level maps.

    Parameters
    ----------
    trace:
        Trace being postprocessed, after :func:`assign_conditional_branch_stacks`.
    """

    entry_edges: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    call_indices: dict[tuple[str, str, int, str], list[int]] = defaultdict(list)
    lookup = _graph_lookup(trace)
    if lookup is None:
        _attribute_forward_edges_facade(trace, entry_edges, call_indices)
    else:
        _attribute_forward_edges_column(trace, lookup, entry_edges, call_indices)
    trace._graph_conditional_arm_entry_edges = dict(entry_edges)
    trace._graph_conditional_edge_call_indices = dict(call_indices)


def _attribute_forward_edges_column(
    trace: Trace,
    lookup: Any,
    entry_edges: dict[tuple[int, str], list[tuple[str, str]]],
    call_indices: dict[tuple[str, str, int, str], list[int]],
) -> None:
    """Column body of :func:`attribute_forward_edges`.

    Cognitive complexity is over 15 on purpose (hot path, once per edge): the
    row lookups and the empty-stack skip are inlined rather than called.
    """

    columns, rows = lookup
    is_orphan = columns.getter("is_orphan", False)
    children = columns.getter("children")
    stack_of = columns.getter("conditional_branch_stack")
    arm_children_of = columns.getter("conditional_arm_children")
    label_raw_of = columns.getter("_label_raw")
    pass_number_of = columns.getter("pass_number")
    for parent_label in trace._raw_graph_ws.raw_layer_labels_list:
        parent_row = rows.get(parent_label)
        if parent_row is None:
            parent_row = _row(trace, rows, parent_label)
        if is_orphan(parent_row):
            continue
        for child_label in children(parent_row):
            child_row = rows.get(child_label)
            if child_row is None:
                child_row = _row(trace, rows, child_label)
            if is_orphan(child_row):
                continue
            child_stack = stack_of(child_row)
            if not child_stack:  # an empty child stack gains nothing
                continue
            gained = get_gained_branch_entries(stack_of(parent_row), child_stack)
            for conditional_id, branch_kind in gained:
                arm_children_of(parent_row).setdefault(conditional_id, {}).setdefault(
                    branch_kind, []
                ).append(child_label)
                entry_edges[(conditional_id, branch_kind)].append(
                    (label_raw_of(parent_row), label_raw_of(child_row))
                )
                call_indices[
                    (
                        label_raw_of(parent_row).split(":", 1)[0],
                        label_raw_of(child_row).split(":", 1)[0],
                        conditional_id,
                        branch_kind,
                    )
                ].append(pass_number_of(child_row))


def _attribute_forward_edges_facade(
    trace: Trace,
    entry_edges: dict[tuple[int, str], list[tuple[str, str]]],
    call_indices: dict[tuple[str, str, int, str], list[int]],
) -> None:
    """Facade body of :func:`attribute_forward_edges` for non-column traces."""

    for parent_label in trace._raw_graph_ws.raw_layer_labels_list:
        parent_layer = trace[parent_label]
        if record_member(parent_layer, "is_orphan", False):
            continue
        for child_label in parent_layer.children:
            child_layer = trace[child_label]
            if record_member(child_layer, "is_orphan", False):
                continue
            gained_entries = get_gained_branch_entries(
                parent_layer.control_flow.branch_stack,
                child_layer.control_flow.branch_stack,
            )
            for conditional_id, branch_kind in gained_entries:
                parent_layer.control_flow.arm_children.setdefault(conditional_id, {}).setdefault(
                    branch_kind, []
                ).append(child_label)
                entry_edges[(conditional_id, branch_kind)].append(
                    (parent_layer._label_raw, child_layer._label_raw)
                )
                call_indices[
                    (
                        parent_layer._label_raw.split(":", 1)[0],
                        child_layer._label_raw.split(":", 1)[0],
                        conditional_id,
                        branch_kind,
                    )
                ].append(child_layer.pass_number)


# ---------------------------------------------------------------------------
# Step 5f: derived per-arm child views
# ---------------------------------------------------------------------------


def _then_children(arm_children: dict[int, dict[str, list[str]]]) -> list[str]:
    """Return the sorted distinct THEN-arm children across every conditional."""

    return sorted(
        set(
            chain.from_iterable(
                branch_children.get("then", []) for branch_children in arm_children.values()
            )
        )
    )


def _elif_children(arm_children: dict[int, dict[str, list[str]]]) -> dict[int, list[str]]:
    """Return the sorted ELIF-arm children keyed by elif index."""

    elif_children: dict[int, set[str]] = defaultdict(set)
    for branch_children in arm_children.values():
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            elif_children[elif_index].update(child_labels)
    return {
        elif_index: sorted(child_labels)
        for elif_index, child_labels in sorted(elif_children.items())
    }


def _else_children(arm_children: dict[int, dict[str, list[str]]]) -> list[str]:
    """Return the sorted distinct ELSE-arm children across every conditional."""

    return sorted(
        set(
            chain.from_iterable(
                branch_children.get("else", []) for branch_children in arm_children.values()
            )
        )
    )


def materialize_derived_views(trace: Trace) -> None:
    """Phase 5f: Rebuild compatibility views derived from primary conditional data.

    Parameters
    ----------
    trace:
        Trace being postprocessed.
    """

    trace._graph_conditional_edge_call_indices = {
        key: sorted(set(call_indexs))
        for key, call_indexs in trace._graph_conditional_edge_call_indices.items()
    }
    lookup = _graph_lookup(trace)
    if lookup is None:
        for layer_label in trace._raw_graph_ws.raw_layer_labels_list:
            layer = trace[layer_label]
            if record_member(layer, "is_orphan", False):
                continue
            layer._control_flow_then_children = _then_children(layer.control_flow.arm_children)
            layer._control_flow_elif_children = _elif_children(layer.control_flow.arm_children)
            layer._control_flow_else_children = _else_children(layer.control_flow.arm_children)
        return
    columns, rows = lookup
    is_orphan = columns.getter("is_orphan", False)
    arm_children_of = columns.getter("conditional_arm_children")
    put_then = columns.setter("conditional_then_children")
    put_elif = columns.setter("conditional_elif_children")
    put_else = columns.setter("conditional_else_children")
    for layer_label in trace._raw_graph_ws.raw_layer_labels_list:
        row = rows.get(layer_label)
        if row is None:
            row = _row(trace, rows, layer_label)
        if is_orphan(row):
            continue
        arm_children = arm_children_of(row)
        if not arm_children:  # what the three builders return for no arms, fresh per op
            put_then(row, [])
            put_elif(row, {})
            put_else(row, [])
            continue
        put_then(row, _then_children(arm_children))
        put_elif(row, _elif_children(arm_children))
        put_else(row, _else_children(arm_children))
