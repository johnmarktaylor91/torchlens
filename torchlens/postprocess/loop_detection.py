"""Step 8: Loop detection, isomorphic subgraph expansion, and layer assignment."""

import heapq
import itertools as it
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ..data_classes.layer_pass_log import LayerPassLog

if TYPE_CHECKING:
    from ..data_classes.model_log import ModelLog


@dataclass
class SubgraphInfo:
    """Info about one isomorphic subgraph rooted at a starting node."""

    starting_node: str
    param_nodes: Set[str] = None
    node_set: Set[str] = None

    def __post_init__(self):
        if self.param_nodes is None:
            self.param_nodes = set()
        if self.node_set is None:
            self.node_set = set()
        self.node_set.add(self.starting_node)


@dataclass
class IsomorphicExpansionState:
    """Mutable state threaded through the BFS isomorphic-subgraph expansion.

    Created once in ``_expand_isomorphic_subgraphs`` and passed to every
    helper (``_advance_bfs_frontier``, ``_collect_frontier_and_detect_adjacency``,
    ``_register_isomorphic_group``, ``_record_subgraph_adjacency``,
    ``_refine_iso_groups``, ``_finalize_layer_assignments``).
    """

    iso_node_groups: OrderedDict
    node_to_iso_leader: OrderedDict
    subgraph_info: Dict[str, SubgraphInfo]
    node_to_subgraph: Dict[str, SubgraphInfo]
    adjacent_subgraphs: Dict[str, set]
    node_stack: deque


def _detect_and_label_loops(self) -> None:
    """
    Post-processing function that yokes together operations corresponding to the same layer, based on
    the following rule:
    1) Operations invoking the same parameters are always assigned to the same layer.
    2) Any contiguous operations surrounding repeated parameters are assigned to the same layer
        (e.g., if a ReLU always follows every pass of an FC layer, then all instances of that ReLU
        operation are considered part of the same layer; continue for all such contiguous
        equivalent operations)
    3) Any groups of contiguous operations that "loop" back to back, irrespective of whether
        they include a parameter or not (e.g., in ABCABCABC, then all As count as the same layer, all
        Bs count as the same layer, and all Cs cound as the same layer, but if a D or an F were inserted
        between these triplets, they would no longer be grouped together, since the repeats
        are no longer contiguous)
    It works by starting from root nodes, and starting from the earliest one, going forward one node at a time,
    and checking if there are equivalent operations. If so, it builds forward one node at a time, until
    it no longer finds equivalent operations. If these subgraphs include a parameter node, these nodes
    are then grouped together no matter what. If they don't, they're only grouped together if contiguous.
    To allow for the possibility that a node might have more "equivalent" layers as a subset of some bigger
    subgraph, then while advancing forward, the function checks the number of equivalent layers it has been
    assigned is equal to the number of operations of that type. If so, it's definitely found everything;
    if not, it runs the procedure again to check if more equivalent operations can be found.
    """
    # Pre-compute sort keys to avoid repeated attribute lookups in the heap.
    _sort_keys = {label: self[label].realtime_tensor_num for label in self._raw_layer_labels_list}

    initial_labels = self.input_layers + self.internally_initialized_layers
    node_heap = [(_sort_keys[label], label) for label in initial_labels]
    heapq.heapify(node_heap)
    heap_seen = set(initial_labels)

    operation_equivalence_types_seen = set()
    while node_heap:
        # Grab the earliest node in the stack, add its children in sorted order to the stack in advance.
        _, node_label = heapq.heappop(node_heap)
        node = self[node_label]
        node_operation_equivalence_type = node.operation_equivalence_type

        # If we've already checked the nodes of this operation equivalence type as starting nodes, continue:
        if node_operation_equivalence_type in operation_equivalence_types_seen:
            continue
        operation_equivalence_types_seen.add(node_operation_equivalence_type)
        for equiv_op in node.equivalent_operations:
            for child in self[equiv_op].child_layers:
                if child not in heap_seen:
                    heap_seen.add(child)
                    heapq.heappush(node_heap, (_sort_keys[child], child))

        # If no equivalent operations for this node, skip it; it's the only operation for this "layer"
        if len(node.equivalent_operations) == 1:
            node.same_layer_operations = [node_label]
            continue

        # If we've already found the same-layer tensors for this node, and it equals the number of
        # equivalent operations, skip it; the work is already done:
        if len(node.equivalent_operations) == len(node.same_layer_operations):
            continue

        # Else, start from this node and any equivalent operations, and work forward, finding
        # more equivalent operations:
        _expand_isomorphic_subgraphs(self, node)

    # Cleanup: rebuild same_layer_operations and pass numbers from layer_label_raw.
    # Multiple rounds of _expand_isomorphic_subgraphs can reassign a node to a new group
    # while leaving stale references in the old group's members. Rebuilding from
    # layer_label_raw (the authoritative group key) fixes this.
    _rebuild_pass_assignments(self)


def _rebuild_pass_assignments(self) -> None:
    """Rebuild same_layer_operations and pass numbers from layer_label_raw.

    Multiple rounds of _expand_isomorphic_subgraphs can reassign a node to a new group
    (via _finalize_layer_assignments) while leaving stale references in the old group's
    same_layer_operations. This function groups all tensors by their current layer_label_raw
    (the authoritative group key) and rebuilds consistent pass assignments.
    """
    groups = defaultdict(list)
    for entry in self:
        groups[entry.layer_label_raw].append(entry.tensor_label_raw)

    for raw_label, members in groups.items():
        members_sorted = sorted(members, key=lambda x: self[x].realtime_tensor_num)
        for pass_index, member_label in enumerate(members_sorted):
            member = self[member_label]
            member.same_layer_operations = members_sorted
            member.pass_num = pass_index + 1
            member.layer_passes_total = len(members_sorted)


def _expand_isomorphic_subgraphs(self, node: LayerPassLog) -> None:
    """Starting from a given node in the graph, starts from all equivalent operations (e.g., cos, add 5, etc.),
    and crawls forward, finding and marking corresponding operations until there are none left.
    At the end of this, nodes that have the same position with respect to the original node
    are labeled as the same layer either if 1) the subgraph contains a parameter node,
    or 2) the nodes belong to adjacent subgraphs.

    Args:
        node: node to start from
    """
    equivalent_operation_starting_labels = sorted(list(node.equivalent_operations))

    sg_info = {}
    for starting_label in equivalent_operation_starting_labels:
        sg_info[starting_label] = SubgraphInfo(starting_node=starting_label)
        if node.computed_with_params:
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
        _advance_bfs_frontier(self, isomorphic_nodes, state, is_first_node)
        is_first_node = False

    _refine_iso_groups(self, state)
    _finalize_layer_assignments(self, state)


def _refine_iso_groups(
    self,
    state: IsomorphicExpansionState,
) -> None:
    """Refine iso node groups by splitting groups where members have
    non-overlapping directional neighborhoods.

    When operations share the same equivalence type but occupy different
    structural positions (e.g., sin(x) in the main loop body vs sin(y) in
    a conditional branch), the BFS expansion assigns their neighbors to
    different iso groups. This function detects such cases by checking
    direction-aware neighbor iso groups: two members are connected if they
    share at least one (direction, iso_leader) pair. Connected components
    within each group become the refined sub-groups.
    """
    for group_leader, members in list(state.iso_node_groups.items()):
        if len(members) <= 1:
            continue

        # For each member, compute direction-aware neighbor iso set
        member_neighbor_isos = {}
        for member_label in members:
            member_node = self[member_label]
            neighbor_groups = set()
            for child in member_node.child_layers:
                if child in state.node_to_iso_leader:
                    neighbor_groups.add(("child", state.node_to_iso_leader[child]))
            for parent in member_node.parent_layers:
                if parent in state.node_to_iso_leader:
                    neighbor_groups.add(("parent", state.node_to_iso_leader[parent]))
            member_neighbor_isos[member_label] = neighbor_groups

        # Connected components via union-find: members sharing at least
        # one directional neighbor iso group are connected
        uf_parent = {member: member for member in members}

        def find(x, uf=uf_parent):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def union(x, y, uf=uf_parent):
            rx, ry = find(x), find(y)
            if rx != ry:
                uf[rx] = ry

        for member1, member2 in it.combinations(members, 2):
            if member_neighbor_isos[member1] & member_neighbor_isos[member2]:
                union(member1, member2)

        # Collect components
        components = defaultdict(list)
        for member in members:
            components[find(member)].append(member)

        if len(components) <= 1:
            continue  # No split needed

        # Split the group: remove old, create new sub-groups
        del state.iso_node_groups[group_leader]
        for comp_members in components.values():
            sorted_members = sorted(comp_members)
            new_leader = sorted_members[0]
            state.iso_node_groups[new_leader] = sorted_members
            for member in sorted_members:
                state.node_to_iso_leader[member] = new_leader


def _advance_bfs_frontier(
    self,
    current_iso_nodes: List[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> None:
    """Takes a set of isomorphic nodes, finds all sets of isomorphic successor nodes,
    then processes them and adds them to the stack.

    Args:
        current_iso_nodes: Current set of isomorphic nodes to get the next nodes from.
        state: Mutable BFS expansion state (iso groups, subgraph info, adjacency, stack).
        is_first_node: Whether it's the first node in the subgraph; if so, just do children, not parents to start.
    """
    frontier_nodes = _collect_frontier_and_detect_adjacency(
        self,
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
            self,
            candidate_node_label,
            candidate_node_neighbor_type,
            candidate_node_subgraph,
            frontier_nodes,
        )

        _register_isomorphic_group(self, new_equivalent_nodes, state)


def _collect_frontier_and_detect_adjacency(
    self,
    current_iso_nodes: List[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> Dict[str, Dict[str, List[str]]]:
    """Checks all parent and children nodes for overlap with nodes already added
    to subgraphs (either the same subgraph or another one), logs any adjacency among subgraphs,
    and returns a dict with the candidate successor nodes from each subgraph.

    Returns:
        Dict with the candidate next nodes for each subgraph.
    """
    node_type_fields = {"children": "child_layers", "parents": "parent_layers"}
    if is_first_node:
        node_types_to_use = ["children"]
    else:
        node_types_to_use = ["children", "parents"]

    frontier_nodes = OrderedDict()
    for node_label in current_iso_nodes:
        node = self[node_label]
        node_subgraph = state.node_to_subgraph[node_label]
        node_subgraph_label = node_subgraph.starting_node
        subgraph_successor_nodes = {"children": [], "parents": []}
        for node_type in node_types_to_use:
            node_type_field = node_type_fields[node_type]
            for neighbor_label in getattr(node, node_type_field):
                if neighbor_label in node_subgraph.node_set:
                    continue
                elif neighbor_label in state.node_to_subgraph:
                    _record_subgraph_adjacency(node_label, neighbor_label, state)
                else:
                    subgraph_successor_nodes[node_type].append(neighbor_label)
        frontier_nodes[node_subgraph_label] = subgraph_successor_nodes

    return frontier_nodes


def _record_subgraph_adjacency(
    node_label: str,
    neighbor_label: str,
    state: IsomorphicExpansionState,
) -> None:
    """Updates the adjacency status of two subgraphs."""
    node_subgraph = state.node_to_subgraph[node_label]
    node_subgraph_label = node_subgraph.starting_node
    neighbor_subgraph = state.node_to_subgraph[neighbor_label]
    neighbor_subgraph_label = neighbor_subgraph.starting_node

    # Subgraphs are adjacent if the node in the neighboring subgraph has an
    # isomorphic node in the current subgraph.
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
    elif (node_subgraph_label in adj) and (neighbor_subgraph_label not in adj):
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
    frontier_nodes: Dict[str, Dict[str, List[str]]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Helper function to grab the next candidate node to consider out of the possible successor nodes.

    Args:
        frontier_nodes: Dict of successor nodes from the set of subgraphs being considered

    Returns:

    """
    for subgraph_label, neighbor_type in it.product(frontier_nodes, ["children", "parents"]):
        subgraph_neighbors = frontier_nodes[subgraph_label][neighbor_type]
        if len(subgraph_neighbors) > 0:
            candidate_node_label = subgraph_neighbors.pop(0)
            candidate_node_neighbor_type = neighbor_type
            candidate_node_subgraph = subgraph_label
            return (
                candidate_node_label,
                candidate_node_neighbor_type,
                candidate_node_subgraph,
            )
    return None, None, None


def _find_isomorphic_matches(
    self,
    candidate_node_label: str,
    candidate_node_neighbor_type: str,
    candidate_node_subgraph: str,
    frontier_nodes: Dict[str, Dict[str, List[str]]],
) -> List[Tuple[str, str]]:
    """Finds nodes that are isomorphic with a candidate node.

    Args:
        candidate_node_label: Label of candidate node
        candidate_node_neighbor_type: Whether the candidate node is a child or parent node
        candidate_node_subgraph: Subgraph of the candidate node
        frontier_nodes: Dict keeping track of possible successor nodes

    Returns:
        List of nodes isomorphic with the candidate node
    """
    candidate_node = self[candidate_node_label]
    candidate_node_operation_equivalence_type = candidate_node.operation_equivalence_type
    new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
    for subgraph_label in frontier_nodes:
        if subgraph_label == candidate_node_subgraph:  # ignore same subgraph
            continue
        other_subgraph_nodes = frontier_nodes[subgraph_label][candidate_node_neighbor_type]
        for comparison_index, comparison_node_label in enumerate(other_subgraph_nodes):
            comparison_node = self[comparison_node_label]
            if (
                comparison_node.operation_equivalence_type
                == candidate_node_operation_equivalence_type
            ):
                new_equivalent_nodes.append(
                    (other_subgraph_nodes.pop(comparison_index), subgraph_label)
                )
                break  # only add one node per subgraph at most
    new_equivalent_nodes = sorted(new_equivalent_nodes, key=lambda x: x[0])

    # Remove any collisions to the SAME node:
    node_labels = [node[0] for node in new_equivalent_nodes]
    new_equivalent_nodes = [
        node for node in new_equivalent_nodes if node_labels.count(node[0]) == 1
    ]
    return new_equivalent_nodes


def _register_isomorphic_group(
    self,
    new_isomorphic_nodes: List[Tuple[str, str]],
    state: IsomorphicExpansionState,
) -> None:
    """Takes a new set of equivalent nodes, and logs them as equivalent, adds them to their subgraphs,
    and adds them to the stack.

    Args:
        new_isomorphic_nodes: Current set of isomorphic nodes to get the next nodes from.
        state: Mutable BFS expansion state.
    """
    if len(new_isomorphic_nodes) > 0:
        iso_group_label = new_isomorphic_nodes[0][0]
        equivalent_node_labels = [tup[0] for tup in new_isomorphic_nodes]
        state.iso_node_groups[iso_group_label] = equivalent_node_labels[:]
        for node_label in equivalent_node_labels:
            state.node_to_iso_leader[node_label] = iso_group_label
        for node_label, node_subgraph in new_isomorphic_nodes:
            node = self[node_label]
            state.subgraph_info[node_subgraph].node_set.add(node_label)
            if node.computed_with_params:
                state.subgraph_info[node_subgraph].param_nodes.add(node_label)
            state.node_to_subgraph[node_label] = state.subgraph_info[node_subgraph]
        state.node_stack.append(equivalent_node_labels)


def _finalize_layer_assignments(
    self,
    state: IsomorphicExpansionState,
) -> None:
    """After extending the subgraphs to maximum size and identifying adjacent subgraphs,
    goes through and labels the layers as corresponding to each other. The rule is that nodes will be
    labeled as corresponding if 1) they are isomorphic with respect to the starting node, and
    2) the subgraphs either contain a param node, or are adjacent.

    Args:
        state: Mutable BFS expansion state.
    """
    merged_layer_groups = _merge_iso_groups_to_layers(
        self, state.iso_node_groups, state.node_to_subgraph, state.adjacent_subgraphs
    )

    # Finally, label the nodes corresponding to the same layer.
    for layer_label, layer_nodes in merged_layer_groups.items():
        # Skip if the new layer asssignment reduces the number of equivalent layers.
        if len(layer_nodes) < max(
            [len(self[layer].same_layer_operations) for layer in layer_nodes]
        ):
            continue
        # convert to list and sort
        layer_nodes = sorted(list(layer_nodes), key=lambda layer: self[layer].realtime_tensor_num)
        # Unify operation_equivalence_type: when param-sharing merges nodes
        # from different iso groups (e.g., shared module called from different
        # parent blocks), their equivalence types may differ due to module path
        # suffixes.  Use the first node's type as canonical.
        canonical_equiv_type = self[layer_nodes[0]].operation_equivalence_type
        for pass_index, node_label in enumerate(layer_nodes):
            node = self[node_label]
            node.layer_label_raw = layer_label
            node.same_layer_operations = layer_nodes
            node.pass_num = pass_index + 1
            node.layer_passes_total = len(layer_nodes)
            node.operation_equivalence_type = canonical_equiv_type


def _merge_iso_groups_to_layers(
    self,
    iso_node_groups: Dict[str, List[str]],
    node_to_subgraph: Dict[str, SubgraphInfo],
    adjacent_subgraphs: Dict[str, set],
) -> Dict[str, Set[str]]:
    """Merge isomorphic node groups into same-layer groups based on shared params or subgraph adjacency.

    Uses union-find with path compression to correctly handle transitive merging.
    Two merge passes:

    1. **Within iso-groups** (Rules 2+3): For each pair of isomorphic nodes, merge when
       their subgraphs share param operation equivalence types or are topologically adjacent.
    2. **Cross iso-groups** (Rule 1): Unconditionally merge ALL nodes that share identical
       ``parent_param_barcodes``, regardless of iso-group membership.  This ensures the
       fundamental invariant that param-sharing operations are always the same layer.

    Args:
        iso_node_groups: Dict mapping each iso-group leader label to the list of node labels in the group.
        node_to_subgraph: Dict mapping each node label to its ``SubgraphInfo``.
        adjacent_subgraphs: Dict mapping each subgraph's starting-node label to the set of adjacent
            subgraph starting-node labels (union-find merged sets).

    Returns:
        Dict mapping each merged-layer leader label to the set of node labels assigned to that layer.
    """
    # -- Union-find with path compression --
    uf_parent: Dict[str, str] = {}

    def _find(x: str) -> str:
        if x not in uf_parent:
            uf_parent[x] = x
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]  # path compression
            x = uf_parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            # Use lexicographic order so the earliest node is always the root
            if rx > ry:
                rx, ry = ry, rx
            uf_parent[ry] = rx

    # Collect all iso-group node labels
    all_iso_nodes = set()
    for iso_nodes_orig in iso_node_groups.values():
        all_iso_nodes.update(iso_nodes_orig)

    # Pass 1: Merge within iso-groups (Rules 2+3 — subgraph adjacency / param type overlap)
    for iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(iso_nodes_orig)
        for node1_label, node2_label in it.combinations(iso_nodes, 2):
            node1_subgraph = node_to_subgraph[node1_label]
            node2_subgraph = node_to_subgraph[node2_label]
            node1_subgraph_label = node1_subgraph.starting_node
            node2_subgraph_label = node2_subgraph.starting_node
            node1_param_types = [
                self[pnode].operation_equivalence_type for pnode in node1_subgraph.param_nodes
            ]
            node2_param_types = [
                self[pnode].operation_equivalence_type for pnode in node2_subgraph.param_nodes
            ]
            overlapping_param_types = set(node1_param_types).intersection(set(node2_param_types))
            subgraphs_are_adjacent = (
                node1_subgraph_label in adjacent_subgraphs
                and node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label]
            )
            if (len(overlapping_param_types) > 0) or subgraphs_are_adjacent:
                _union(node1_label, node2_label)

    # Pass 2: Merge across iso-groups by identical param barcodes (Rule 1 — unconditional)
    # If two nodes anywhere in the model share the same sorted(parent_param_barcodes),
    # they MUST be the same layer regardless of iso-group membership.
    param_barcode_groups: Dict[tuple, list] = defaultdict(list)
    for node_label in all_iso_nodes:
        node = self[node_label]
        if node.computed_with_params and node.parent_param_barcodes:
            barcode_key = tuple(sorted(node.parent_param_barcodes))
            param_barcode_groups[barcode_key].append(node_label)

    for barcode_key, nodes_with_same_params in param_barcode_groups.items():
        if len(nodes_with_same_params) > 1:
            first = nodes_with_same_params[0]
            for other in nodes_with_same_params[1:]:
                _union(first, other)

    # Collect final groups from union-find
    merged_layer_groups: Dict[str, Set[str]] = defaultdict(set)
    for node_label in all_iso_nodes:
        root = _find(node_label)
        merged_layer_groups[root].add(node_label)

    # Only return groups with 2+ members (single nodes aren't merged layers)
    return {leader: nodes for leader, nodes in merged_layer_groups.items() if len(nodes) > 1}
