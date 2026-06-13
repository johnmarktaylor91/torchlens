"""Backend-neutral recurrence grouping model and service.

This module contains the graph-only inputs needed by Step 7 loop grouping. Backend
finishers are responsible for adapting their postprocess state into this model,
passing only data-flow edges in ``data_parents`` and ``data_children``.
"""

import heapq
import itertools as it
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Set, Tuple


FrontierNodes = OrderedDict[str, dict[str, deque[str]]]


@dataclass(frozen=True)
class RecurrenceNode:
    """Graph-only node input for recurrence grouping.

    Parameters
    ----------
    label:
        Stable backend-local node label.
    raw_order:
        Capture order used for deterministic traversal and pass ordering.
    equivalence_key:
        Backend-provided structural key used for isomorphic matching.
    equivalent_labels:
        Backend-provided candidate labels considered equivalent enough to seed
        an expansion round. Torch supplies its existing ``equivalent_ops`` set.
    data_parents:
        Parent labels connected by value/data edges only.
    data_children:
        Child labels connected by value/data edges only.
    layer_label:
        Current raw layer label assignment for this node.
    recurrent_labels:
        Current recurrent-op assignment, if any, used by conservative merge guards.
    uses_params:
        Whether this operation uses learned parameters.
    func_name:
        Function name used for same-function plus same-params merging.
    param_barcodes:
        Stable parameter identifiers used for same-params merging.
    retain:
        Whether the node should be retained in grouping.
    pruned:
        Whether the node has been pruned from the user-visible graph.
    """

    label: str
    raw_order: int
    equivalence_key: str
    equivalent_labels: tuple[str, ...]
    data_parents: tuple[str, ...]
    data_children: tuple[str, ...]
    layer_label: str
    recurrent_labels: tuple[str, ...]
    uses_params: bool
    func_name: str
    param_barcodes: tuple[str, ...]
    retain: bool = True
    pruned: bool = False


@dataclass(frozen=True)
class RecurrenceGroupingGraph:
    """Backend-neutral graph input for recurrence grouping.

    Parameters
    ----------
    nodes:
        Mapping from node label to recurrence node data.
    raw_labels:
        Labels in raw capture order.
    source_labels:
        Input and internally initialized source labels used to seed traversal.
    eligible_labels:
        Labels considered by grouping. Backends may retain pruned nodes in
        ``nodes`` for diagnostics, but only eligible labels participate.
    """

    nodes: Mapping[str, RecurrenceNode]
    raw_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    eligible_labels: tuple[str, ...]


@dataclass(frozen=True)
class RecurrenceAssignment:
    """Computed recurrence assignment for one eligible node.

    Parameters
    ----------
    layer_label:
        Raw layer label leader assigned to this node.
    recurrent_labels:
        Raw labels in this layer, sorted by raw capture order.
    pass_index:
        One-indexed pass number within ``recurrent_labels``.
    num_passes:
        Number of passes represented by this layer.
    equivalence_key:
        Canonical structural key assigned to the grouped nodes.
    """

    layer_label: str
    recurrent_labels: tuple[str, ...]
    pass_index: int
    num_passes: int
    equivalence_key: str


@dataclass
class _MutableRecurrenceNode:
    """Mutable working copy of a neutral recurrence node."""

    label: str
    raw_order: int
    equivalence_key: str
    equivalent_labels: tuple[str, ...]
    data_parents: tuple[str, ...]
    data_children: tuple[str, ...]
    layer_label: str
    recurrent_labels: list[str]
    uses_params: bool
    func_name: str
    param_barcodes: tuple[str, ...]


@dataclass
class SubgraphInfo:
    """Track nodes belonging to one isomorphic subgraph.

    Parameters
    ----------
    starting_node:
        Label of the equivalent operation that anchors this subgraph.
    param_nodes:
        Labels in this subgraph that use learned parameters.
    node_set:
        Labels already assigned to this subgraph.
    """

    starting_node: str
    param_nodes: Set[str] = field(default_factory=set)
    node_set: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Register the starting node in this subgraph."""
        self.node_set.add(self.starting_node)


@dataclass
class IsomorphicExpansionState:
    """Mutable state threaded through BFS isomorphic-subgraph expansion."""

    iso_node_groups: OrderedDict[str, list[str]]
    node_to_iso_leader: OrderedDict[str, str]
    subgraph_info: Dict[str, SubgraphInfo]
    node_to_subgraph: Dict[str, SubgraphInfo]
    adjacent_subgraphs: Dict[str, set[str]]
    node_stack: deque[list[str]]


@dataclass
class _GroupingWorkspace:
    """Mutable recurrence grouping workspace built from a neutral graph."""

    nodes: dict[str, _MutableRecurrenceNode]
    raw_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    eligible_labels: set[str]

    @classmethod
    def from_graph(cls, graph: RecurrenceGroupingGraph) -> "_GroupingWorkspace":
        """Build a mutable workspace from a neutral recurrence graph.

        Parameters
        ----------
        graph:
            Backend-neutral recurrence graph.

        Returns
        -------
        _GroupingWorkspace
            Mutable grouping workspace.
        """
        eligible = set(graph.eligible_labels)
        nodes = {
            label: _MutableRecurrenceNode(
                label=node.label,
                raw_order=node.raw_order,
                equivalence_key=node.equivalence_key,
                equivalent_labels=tuple(node.equivalent_labels),
                data_parents=tuple(node.data_parents),
                data_children=tuple(node.data_children),
                layer_label=node.layer_label,
                recurrent_labels=list(node.recurrent_labels),
                uses_params=node.uses_params,
                func_name=node.func_name,
                param_barcodes=tuple(node.param_barcodes),
            )
            for label, node in graph.nodes.items()
            if label in eligible and node.retain and not node.pruned
        }
        return cls(
            nodes=nodes,
            raw_labels=tuple(label for label in graph.raw_labels if label in nodes),
            source_labels=tuple(label for label in graph.source_labels if label in nodes),
            eligible_labels=set(nodes),
        )

    def equivalent_labels(self, label: str) -> tuple[str, ...]:
        """Return eligible equivalent labels for ``label``.

        Parameters
        ----------
        label:
            Node label to inspect.

        Returns
        -------
        tuple[str, ...]
            Equivalent labels present in the grouping workspace.
        """
        node = self.nodes[label]
        return tuple(equiv for equiv in node.equivalent_labels if equiv in self.nodes)

    def assignments(self) -> dict[str, RecurrenceAssignment]:
        """Return computed assignments for every eligible node.

        Returns
        -------
        dict[str, RecurrenceAssignment]
            Assignments keyed by node label.
        """
        return {
            label: RecurrenceAssignment(
                layer_label=node.layer_label,
                recurrent_labels=tuple(node.recurrent_labels),
                pass_index=index + 1,
                num_passes=len(node.recurrent_labels),
                equivalence_key=node.equivalence_key,
            )
            for label, node in self.nodes.items()
            for index, recurrent_label in enumerate(node.recurrent_labels)
            if recurrent_label == label
        }


def group_recurrent_nodes(graph: RecurrenceGroupingGraph) -> dict[str, RecurrenceAssignment]:
    """Group recurrent nodes in a backend-neutral graph.

    Parameters
    ----------
    graph:
        Backend-neutral recurrence graph.

    Returns
    -------
    dict[str, RecurrenceAssignment]
        Final recurrence assignments keyed by eligible node label.
    """
    workspace = _GroupingWorkspace.from_graph(graph)
    _detect_and_label_workspace_loops(workspace)
    return workspace.assignments()


def _detect_and_label_workspace_loops(workspace: _GroupingWorkspace) -> None:
    """Detect loops and assign recurrence groups in a mutable workspace.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    None
        Mutates ``workspace``.
    """
    sort_keys = {label: workspace.nodes[label].raw_order for label in workspace.raw_labels}
    node_heap = [(sort_keys[label], label) for label in workspace.source_labels]
    heapq.heapify(node_heap)
    heap_seen = set(workspace.source_labels)
    equivalence_keys_seen: set[str] = set()

    while node_heap:
        _, node_label = heapq.heappop(node_heap)
        node = workspace.nodes[node_label]
        node_equivalence_key = node.equivalence_key

        if node_equivalence_key in equivalence_keys_seen:
            continue
        equivalence_keys_seen.add(node_equivalence_key)

        equivalent_labels = workspace.equivalent_labels(node_label)
        for equiv_label in equivalent_labels:
            for child in workspace.nodes[equiv_label].data_children:
                if child not in heap_seen and child in workspace.nodes:
                    heap_seen.add(child)
                    heapq.heappush(node_heap, (sort_keys[child], child))

        if len(equivalent_labels) == 1:
            node.recurrent_labels = [node_label]
            continue

        if len(equivalent_labels) == len(node.recurrent_labels):
            continue

        _expand_isomorphic_subgraphs(workspace, node_label)

    _rebuild_pass_assignments(workspace)


def _rebuild_pass_assignments(workspace: _GroupingWorkspace) -> None:
    """Rebuild recurrent labels and pass counts from authoritative layer labels.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    None
        Mutates each node's recurrent label assignment.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for label in workspace.raw_labels:
        node = workspace.nodes[label]
        groups[node.layer_label].append(label)

    for members in groups.values():
        members_sorted = sorted(members, key=lambda label: workspace.nodes[label].raw_order)
        for member_label in members_sorted:
            workspace.nodes[member_label].recurrent_labels = members_sorted


def _expand_isomorphic_subgraphs(workspace: _GroupingWorkspace, node_label: str) -> None:
    """Expand isomorphic subgraphs from one equivalent operation group.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    node_label:
        Label whose equivalent group seeds expansion.

    Returns
    -------
    None
        Mutates ``workspace`` assignments.
    """
    node = workspace.nodes[node_label]
    equivalent_operation_starting_labels = sorted(workspace.equivalent_labels(node_label))
    if not equivalent_operation_starting_labels:
        return

    sg_info: dict[str, SubgraphInfo] = {}
    for starting_label in equivalent_operation_starting_labels:
        sg_info[starting_label] = SubgraphInfo(starting_node=starting_label)
        if node.uses_params:
            sg_info[starting_label].param_nodes.add(starting_label)

    state = IsomorphicExpansionState(
        iso_node_groups=OrderedDict(
            {equivalent_operation_starting_labels[0]: equivalent_operation_starting_labels}
        ),
        node_to_iso_leader=OrderedDict(
            {
                label: equivalent_operation_starting_labels[0]
                for label in equivalent_operation_starting_labels
            }
        ),
        subgraph_info=sg_info,
        node_to_subgraph=OrderedDict(
            {label: sg_info[label] for label in equivalent_operation_starting_labels}
        ),
        adjacent_subgraphs={},
        node_stack=deque([equivalent_operation_starting_labels[:]]),
    )

    is_first_node = True
    while state.node_stack:
        isomorphic_nodes = sorted(state.node_stack.popleft())
        if len(isomorphic_nodes) == 1:
            continue
        _advance_bfs_frontier(workspace, isomorphic_nodes, state, is_first_node)
        is_first_node = False

    _refine_iso_groups(workspace, state)
    _finalize_layer_assignments(workspace, state)


def _refine_iso_groups(
    workspace: _GroupingWorkspace,
    state: IsomorphicExpansionState,
) -> None:
    """Split iso groups whose members do not share directional neighbor signatures.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state``.
    """
    for group_leader, members in list(state.iso_node_groups.items()):
        if len(members) <= 1:
            continue

        member_neighbor_isos: dict[str, set[tuple[str, str]]] = {}
        for member_label in members:
            member_node = workspace.nodes[member_label]
            neighbor_groups: set[tuple[str, str]] = set()
            for child in member_node.data_children:
                if child in state.node_to_iso_leader:
                    neighbor_groups.add(("child", state.node_to_iso_leader[child]))
            for parent in member_node.data_parents:
                if parent in state.node_to_iso_leader:
                    neighbor_groups.add(("parent", state.node_to_iso_leader[parent]))
            member_neighbor_isos[member_label] = neighbor_groups

        uf_parent = {member: member for member in members}

        def find(x: str, uf: dict[str, str] = uf_parent) -> str:
            """Return the union-find root for a group member."""
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def union(x: str, y: str, uf: dict[str, str] = uf_parent) -> None:
            """Merge the union-find sets for two group members."""
            rx, ry = find(x), find(y)
            if rx != ry:
                uf[rx] = ry

        reverse_index: dict[tuple[str, str], list[str]] = defaultdict(list)
        for member_label in members:
            for neighbor_key in member_neighbor_isos[member_label]:
                reverse_index[neighbor_key].append(member_label)
        for members_with_key in reverse_index.values():
            if len(members_with_key) > 1:
                first = members_with_key[0]
                for other in members_with_key[1:]:
                    union(first, other)

        components: dict[str, list[str]] = defaultdict(list)
        for member in members:
            components[find(member)].append(member)

        if len(components) <= 1:
            continue

        del state.iso_node_groups[group_leader]
        for comp_members in components.values():
            sorted_members = sorted(comp_members)
            new_leader = sorted_members[0]
            state.iso_node_groups[new_leader] = sorted_members
            for member in sorted_members:
                state.node_to_iso_leader[member] = new_leader


def _advance_bfs_frontier(
    workspace: _GroupingWorkspace,
    current_iso_nodes: list[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> None:
    """Process one BFS frontier step.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    current_iso_nodes:
        Labels occupying the same structural position across subgraphs.
    state:
        Mutable isomorphic expansion state.
    is_first_node:
        Whether this is the first expansion step from the starting nodes.

    Returns
    -------
    None
        Mutates ``state``.
    """
    frontier_nodes = _collect_frontier_and_detect_adjacency(
        workspace,
        current_iso_nodes,
        state,
        is_first_node,
    )

    while True:
        (
            candidate_node_label,
            candidate_node_neighbor_type,
            candidate_node_subgraph,
        ) = _pop_frontier_node(frontier_nodes)
        if candidate_node_label is None:
            break

        new_equivalent_nodes = _find_isomorphic_matches(
            workspace,
            candidate_node_label,
            candidate_node_neighbor_type,  # type: ignore[arg-type]
            candidate_node_subgraph,  # type: ignore[arg-type]
            frontier_nodes,
        )

        _register_isomorphic_group(workspace, new_equivalent_nodes, state)


def _collect_frontier_and_detect_adjacency(
    workspace: _GroupingWorkspace,
    current_iso_nodes: list[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> FrontierNodes:
    """Collect frontier nodes and detect inter-subgraph adjacency.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    current_iso_nodes:
        Labels occupying the same structural position across subgraphs.
    state:
        Mutable isomorphic expansion state.
    is_first_node:
        Whether to expand only children.

    Returns
    -------
    FrontierNodes
        Candidate frontier nodes by subgraph and neighbor direction.
    """
    node_types_to_use = ["children"] if is_first_node else ["children", "parents"]
    frontier_nodes: FrontierNodes = OrderedDict()

    for node_label in current_iso_nodes:
        node = workspace.nodes[node_label]
        node_subgraph = state.node_to_subgraph[node_label]
        node_subgraph_label = node_subgraph.starting_node
        subgraph_successor_nodes: dict[str, deque[str]] = {
            "children": deque(),
            "parents": deque(),
        }
        added_neighbors: set[str] = set()
        for node_type in node_types_to_use:
            neighbor_labels = node.data_children if node_type == "children" else node.data_parents
            for neighbor_label in neighbor_labels:
                if neighbor_label not in workspace.nodes:
                    continue
                if neighbor_label in node_subgraph.node_set:
                    continue
                if neighbor_label in state.node_to_subgraph:
                    _record_subgraph_adjacency(node_label, neighbor_label, state)
                elif neighbor_label not in added_neighbors:
                    subgraph_successor_nodes[node_type].append(neighbor_label)
                    added_neighbors.add(neighbor_label)
        frontier_nodes[node_subgraph_label] = subgraph_successor_nodes

    return frontier_nodes


def _record_subgraph_adjacency(
    node_label: str,
    neighbor_label: str,
    state: IsomorphicExpansionState,
) -> None:
    """Mark two subgraphs as adjacent in the expansion state.

    Parameters
    ----------
    node_label:
        Current node label.
    neighbor_label:
        Neighbor label already assigned to another subgraph.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state.adjacent_subgraphs``.
    """
    node_subgraph = state.node_to_subgraph[node_label]
    node_subgraph_label = node_subgraph.starting_node
    neighbor_subgraph = state.node_to_subgraph[neighbor_label]
    neighbor_subgraph_label = neighbor_subgraph.starting_node

    neighbor_iso_group = state.node_to_iso_leader[neighbor_label]
    nodes_isomorphic_to_neighbor_node = state.iso_node_groups[neighbor_iso_group]
    if len(node_subgraph.node_set.intersection(nodes_isomorphic_to_neighbor_node)) == 0:
        return

    adj = state.adjacent_subgraphs
    if (node_subgraph_label in adj) and (neighbor_subgraph_label in adj):
        if adj[node_subgraph_label] is not adj[neighbor_subgraph_label]:
            merged = adj[node_subgraph_label] | adj[neighbor_subgraph_label]
            for subgraph_key in merged:
                adj[subgraph_key] = merged
        return
    if (node_subgraph_label in adj) and (neighbor_subgraph_label not in adj):
        adj[node_subgraph_label].add(neighbor_subgraph_label)
        adj[neighbor_subgraph_label] = adj[node_subgraph_label]
    elif (node_subgraph_label not in adj) and (neighbor_subgraph_label in adj):
        adj[neighbor_subgraph_label].add(node_subgraph_label)
        adj[node_subgraph_label] = adj[neighbor_subgraph_label]
    else:
        new_adj_set = {node_subgraph_label, neighbor_subgraph_label}
        adj[node_subgraph_label] = new_adj_set
        adj[neighbor_subgraph_label] = new_adj_set


def _pop_frontier_node(
    frontier_nodes: FrontierNodes,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pop the next frontier candidate for isomorphic matching.

    Parameters
    ----------
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        Candidate label, neighbor type, and subgraph label, or all ``None``.
    """
    for subgraph_label, neighbor_type in it.product(frontier_nodes, ["children", "parents"]):
        subgraph_neighbors = frontier_nodes[subgraph_label][neighbor_type]
        if len(subgraph_neighbors) > 0:
            candidate_node_label = subgraph_neighbors.popleft()
            return candidate_node_label, neighbor_type, subgraph_label
    return None, None, None


def _find_isomorphic_matches(
    workspace: _GroupingWorkspace,
    candidate_node_label: str,
    candidate_node_neighbor_type: str,
    candidate_node_subgraph: str,
    frontier_nodes: FrontierNodes,
) -> list[tuple[str, str]]:
    """Find candidate-equivalent nodes across other subgraph frontiers.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    candidate_node_label:
        Candidate node label from one subgraph.
    candidate_node_neighbor_type:
        Neighbor direction, ``"children"`` or ``"parents"``.
    candidate_node_subgraph:
        Starting-node label for the candidate's subgraph.
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    list[tuple[str, str]]
        Matched ``(node_label, subgraph_label)`` pairs.
    """
    candidate_node = workspace.nodes[candidate_node_label]
    candidate_node_equivalence_key = candidate_node.equivalence_key
    new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
    for subgraph_label in frontier_nodes:
        if subgraph_label == candidate_node_subgraph:
            continue
        other_subgraph_nodes = frontier_nodes[subgraph_label][candidate_node_neighbor_type]
        for comparison_index, comparison_node_label in enumerate(other_subgraph_nodes):
            comparison_node = workspace.nodes[comparison_node_label]
            if comparison_node.equivalence_key == candidate_node_equivalence_key:
                del other_subgraph_nodes[comparison_index]
                new_equivalent_nodes.append((comparison_node_label, subgraph_label))
                break
    new_equivalent_nodes = sorted(new_equivalent_nodes, key=lambda item: item[0])

    seen_labels: set[str] = set()
    dupe_labels: set[str] = set()
    for node in new_equivalent_nodes:
        if node[0] in seen_labels:
            dupe_labels.add(node[0])
        seen_labels.add(node[0])
    if dupe_labels:
        new_equivalent_nodes = [node for node in new_equivalent_nodes if node[0] not in dupe_labels]
    return new_equivalent_nodes


def _register_isomorphic_group(
    workspace: _GroupingWorkspace,
    new_isomorphic_nodes: list[tuple[str, str]],
    state: IsomorphicExpansionState,
) -> None:
    """Register a newly discovered iso group.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    new_isomorphic_nodes:
        ``(node_label, subgraph_label)`` pairs in the new group.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state``.
    """
    if len(new_isomorphic_nodes) == 0:
        return
    iso_group_label = new_isomorphic_nodes[0][0]
    equivalent_node_labels = [tup[0] for tup in new_isomorphic_nodes]
    state.iso_node_groups[iso_group_label] = equivalent_node_labels[:]
    for node_label in equivalent_node_labels:
        state.node_to_iso_leader[node_label] = iso_group_label
    for node_label, node_subgraph in new_isomorphic_nodes:
        node = workspace.nodes[node_label]
        state.subgraph_info[node_subgraph].node_set.add(node_label)
        if node.uses_params:
            state.subgraph_info[node_subgraph].param_nodes.add(node_label)
        state.node_to_subgraph[node_label] = state.subgraph_info[node_subgraph]
    state.node_stack.append(equivalent_node_labels)


def _finalize_layer_assignments(
    workspace: _GroupingWorkspace,
    state: IsomorphicExpansionState,
) -> None:
    """Assign same-layer labels from iso groups, parameters, and adjacency.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``workspace`` assignments.
    """
    merged_layer_groups = _merge_iso_groups_to_layers(
        workspace, state.iso_node_groups, state.node_to_subgraph, state.adjacent_subgraphs
    )

    for layer_label, layer_nodes_set in merged_layer_groups.items():
        layer_nodes = sorted(
            list(layer_nodes_set), key=lambda layer: workspace.nodes[layer].raw_order
        )
        if len(layer_nodes) < max(
            [len(workspace.nodes[layer].recurrent_labels) for layer in layer_nodes]
        ):
            continue
        canonical_equiv_type = workspace.nodes[layer_nodes[0]].equivalence_key
        for pass_index, grouped_node_label in enumerate(layer_nodes):
            node = workspace.nodes[grouped_node_label]
            node.layer_label = layer_label
            node.recurrent_labels = layer_nodes
            node.equivalence_key = canonical_equiv_type


def _merge_iso_groups_to_layers(
    workspace: _GroupingWorkspace,
    iso_node_groups: Dict[str, list[str]],
    node_to_subgraph: Dict[str, SubgraphInfo],
    adjacent_subgraphs: Dict[str, set[str]],
) -> Dict[str, Set[str]]:
    """Merge iso groups into same-layer groups using union-find.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    iso_node_groups:
        Iso-group leader to member labels.
    node_to_subgraph:
        Node label to subgraph info.
    adjacent_subgraphs:
        Subgraph adjacency union-find map.

    Returns
    -------
    dict[str, set[str]]
        Merged layer groups with at least two members.
    """
    uf_parent: Dict[str, str] = {}

    def find(x: str) -> str:
        """Return the union-find root for a node label."""
        if x not in uf_parent:
            uf_parent[x] = x
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def union(x: str, y: str) -> None:
        """Merge two union-find sets."""
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx > ry:
                rx, ry = ry, rx
            uf_parent[ry] = rx

    all_iso_nodes: set[str] = set()
    for iso_nodes_orig in iso_node_groups.values():
        all_iso_nodes.update(iso_nodes_orig)

    sg_param_types: dict[str, frozenset[str]] = {}
    for iso_nodes_orig in iso_node_groups.values():
        for node_label in iso_nodes_orig:
            sg = node_to_subgraph[node_label]
            sg_label = sg.starting_node
            if sg_label not in sg_param_types:
                sg_param_types[sg_label] = frozenset(
                    workspace.nodes[pnode].equivalence_key for pnode in sg.param_nodes
                )

    for iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(iso_nodes_orig)
        for node1_label, node2_label in it.combinations(iso_nodes, 2):
            node1_subgraph_label = node_to_subgraph[node1_label].starting_node
            node2_subgraph_label = node_to_subgraph[node2_label].starting_node
            overlapping_param_types = (
                sg_param_types[node1_subgraph_label] & sg_param_types[node2_subgraph_label]
            )
            subgraphs_are_adjacent = (
                node1_subgraph_label in adjacent_subgraphs
                and node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label]
            )
            if overlapping_param_types or subgraphs_are_adjacent:
                union(node1_label, node2_label)

    param_barcode_groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for node_label in all_iso_nodes:
        node = workspace.nodes[node_label]
        if node.uses_params and node.param_barcodes:
            barcode_key = (node.func_name, tuple(sorted(node.param_barcodes)))
            param_barcode_groups[barcode_key].append(node_label)

    for barcode_key, nodes_with_same_params in param_barcode_groups.items():
        if len(nodes_with_same_params) > 1:
            first = nodes_with_same_params[0]
            for other in nodes_with_same_params[1:]:
                union(first, other)

    merged_layer_groups: Dict[str, Set[str]] = defaultdict(set)
    for node_label in all_iso_nodes:
        root = find(node_label)
        merged_layer_groups[root].add(node_label)

    return {leader: nodes for leader, nodes in merged_layer_groups.items() if len(nodes) > 1}


__all__ = [
    "RecurrenceAssignment",
    "RecurrenceGroupingGraph",
    "RecurrenceNode",
    "group_recurrent_nodes",
]
