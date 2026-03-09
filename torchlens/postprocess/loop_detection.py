"""Step 8: Loop detection, isomorphic subgraph expansion, and layer assignment.

This module identifies repeated operations (loops, recurrence) in the computational
graph and assigns them to the same "layer" with multiple passes. For example, if a
model calls sin() 8 times in a loop, those 8 operations become one layer with 8 passes.

ALGORITHM OVERVIEW (5 phases):
==============================

Phase 1 — Entry (``_detect_and_label_loops``):
    Iterates nodes in realtime order via a min-heap BFS from inputs. For each
    unprocessed ``operation_equivalence_type``, calls ``_expand_isomorphic_subgraphs``.
    The ``operation_equivalence_types_seen`` set ensures each equivalence type is
    processed exactly once, providing convergence. After all rounds, calls
    ``_rebuild_pass_assignments`` to fix stale cross-references.

Phase 2 — BFS Expansion (``_expand_isomorphic_subgraphs``):
    Given a node with equivalent operations, starts one subgraph per equivalent
    node and expands them in lockstep via BFS. At each frontier step:
    - Collects children (and parents, after the first step) of all current iso nodes.
    - Detects adjacency when a frontier node belongs to another subgraph.
    - Matches frontier nodes across subgraphs by operation_equivalence_type.
    - Registers matched nodes as new iso groups and adds them to the BFS queue.
    Expansion continues until no more isomorphic matches are found.

Phase 3 — Iso Group Refinement (``_refine_iso_groups``):
    The initial BFS puts ALL same-equivalence-type operations into one iso group
    and never splits them. If structurally unrelated operations share an equivalence
    type (e.g., sin(x) in a loop body vs sin(y) in a branch), the BFS wrongly
    treats them as isomorphic. Refinement splits such groups using direction-aware
    neighbor connectivity: two members stay together only if they share at least
    one (direction, neighbor_iso_leader) pair. Connected components within each
    group become the refined sub-groups.

Phase 4 — Layer Assignment (``_finalize_layer_assignments`` + ``_merge_iso_groups_to_layers``):
    Groups iso nodes into same-layer sets using union-find. Two merge passes:
    Pass 1 (Rules 2+3): Within each iso group, merge nodes whose subgraphs share
        parameter equivalence types OR are topologically adjacent.
    Pass 2 (Rule 1): Unconditionally merge ALL nodes with identical
        (func_name, sorted(parent_param_barcodes)) — the fundamental
        invariant that same function + same params = same layer.

Phase 5 — Cleanup (``_rebuild_pass_assignments``):
    Multiple expansion rounds can reassign a node to a new group while leaving
    stale recurrent_group in old group members. This function groups all
    tensors by their authoritative ``layer_label_raw`` and rebuilds consistent
    pass assignments. This is NOT just defensive cleanup — it is a NECESSARY
    correctness step because Step 6's module suffix mutation causes the same
    equivalence group to be processed multiple times, and conflicting assignments
    from different rounds must be reconciled.

CORE INVARIANT:
    Two operations may only be assigned to the same layer if their subgraphs either:
    1. Share parameter equivalence types (same learned weights), OR
    2. Are adjacent — connected via chains of equivalent operations in the graph.
    If neither condition holds, operations MUST remain separate layers.

ADJACENCY TRACKING:
    ``adjacent_subgraphs`` is a union-find Dict[str, set] where each subgraph label
    maps to a shared set object. When two subgraphs are found adjacent, their sets
    are merged. All labels in a connected component point to the SAME set object.
"""

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
    """Tracks the set of nodes belonging to one isomorphic subgraph.

    Each subgraph is anchored by a starting_node (one of the equivalent operations
    that initiated the BFS). As the BFS expands, new nodes are added to node_set.
    param_nodes tracks which nodes in this subgraph use learned parameters — this
    is used during layer assignment to determine if subgraphs share parameter types.
    """

    starting_node: str
    param_nodes: Set[str] = None  # type: ignore[assignment]
    node_set: Set[str] = None  # type: ignore[assignment]

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
    helper function. This centralizes all mutable state to avoid excessive
    parameter passing.

    Attributes:
        iso_node_groups: Maps each iso-group leader label to the list of member
            labels. Nodes in the same iso group occupy the same structural position
            across different subgraphs (iterations of a loop).
        node_to_iso_leader: Reverse map — each node label to its group's leader.
        subgraph_info: Maps each subgraph's starting-node label to its SubgraphInfo.
        node_to_subgraph: Maps each node label to the SubgraphInfo it belongs to.
        adjacent_subgraphs: Union-find structure for subgraph adjacency. Maps each
            subgraph label to a SHARED set of all adjacent subgraph labels. Multiple
            keys may point to the same set object (union-find semantics).
        node_stack: BFS queue of iso-node lists to process next. Each entry is a
            list of node labels that are isomorphic to each other.
    """

    iso_node_groups: OrderedDict
    node_to_iso_leader: OrderedDict
    subgraph_info: Dict[str, SubgraphInfo]
    node_to_subgraph: Dict[str, SubgraphInfo]
    adjacent_subgraphs: Dict[str, set]
    node_stack: deque


def _group_by_shared_params(self) -> None:
    """Lightweight same-param grouping without full loop detection.

    Groups operations that share identical ``(func_name,
    sorted(parent_param_barcodes))`` into the same layer. This is Rule 1
    (the fundamental invariant) from the full loop detection algorithm,
    without the expensive isomorphic subgraph expansion (Phases 2-3).

    Operations without parameters remain as individual single-pass layers.
    Sets ``layer_label_raw``, ``recurrent_group``, ``pass_num``,
    and ``num_passes`` on each LayerPassLog.
    """
    param_barcode_groups: Dict[tuple, list] = defaultdict(list)
    for label in self._raw_layer_labels_list:
        node = self[label]
        if node.uses_params and node.parent_param_barcodes:
            key = (node.func_name, tuple(sorted(node.parent_param_barcodes)))
            param_barcode_groups[key].append(label)

    # For multi-member groups, set layer_label_raw to the first member's raw label.
    for key, members in param_barcode_groups.items():
        if len(members) > 1:
            leader = min(members, key=lambda x: self[x].creation_order)
            leader_raw = self[leader].layer_label_raw
            for label in members:
                self[label].layer_label_raw = leader_raw

    # Rebuild recurrent_group, pass_num, num_passes from layer_label_raw.
    _rebuild_pass_assignments(self)


def _detect_and_label_loops(self) -> None:
    """Phase 1: Entry point for loop detection.

    Iterates nodes in realtime order (earliest first) via a min-heap BFS starting
    from input and internally-initialized nodes. For each node, checks whether its
    ``operation_equivalence_type`` has already been processed. If not, and the node
    has equivalent operations (same function, args, shape, dtype, module path),
    initiates BFS expansion to find isomorphic subgraphs.

    Three grouping rules (enforced in Phase 4):
    1. **Same function + same params = same layer** (unconditional merge).
    2. **Contiguous operations around shared params** are grouped (subgraph
       adjacency or overlapping param types).
    3. **Adjacent param-free loops** (ABCABC patterns) are grouped when subgraphs
       are topologically adjacent.

    The ``operation_equivalence_types_seen`` set ensures each equivalence type is
    processed exactly once, which guarantees convergence. However, Step 6's module
    suffix mutation can cause the same underlying equivalence group to appear under
    different suffixed types — this means the same group may be processed multiple
    times (once per module-path variant). ``_rebuild_pass_assignments`` (Phase 5)
    reconciles any resulting conflicts.
    """
    # Pre-compute sort keys to avoid repeated attribute lookups in the heap.
    _sort_keys = {label: self[label].creation_order for label in self._raw_layer_labels_list}

    # Seed the heap with root nodes (inputs + internally-initialized tensors).
    initial_labels = self.input_layers + self.internally_initialized_layers
    node_heap = [(_sort_keys[label], label) for label in initial_labels]
    heapq.heapify(node_heap)
    heap_seen = set(initial_labels)

    # Track which equivalence types have been processed to avoid redundant work.
    operation_equivalence_types_seen = set()
    while node_heap:
        _, node_label = heapq.heappop(node_heap)
        node = self[node_label]
        node_operation_equivalence_type = node.operation_equivalence_type

        # Dedup: skip if this equivalence type has already been processed.
        if node_operation_equivalence_type in operation_equivalence_types_seen:
            continue
        operation_equivalence_types_seen.add(node_operation_equivalence_type)

        # Push children of ALL equivalent operations onto the heap to ensure
        # downstream nodes are eventually visited, even if this node is skipped.
        for equiv_op in node.equivalent_operations:
            for child in self[equiv_op].child_layers:
                if child not in heap_seen:
                    heap_seen.add(child)
                    heapq.heappush(node_heap, (_sort_keys[child], child))

        # Singleton: only one operation of this type, no loop possible.
        if len(node.equivalent_operations) == 1:
            node.recurrent_group = [node_label]
            continue

        # Already fully resolved by a previous expansion round.
        if len(node.equivalent_operations) == len(node.recurrent_group):
            continue

        # Multiple equivalent operations exist — expand isomorphic subgraphs
        # to determine which ones belong to the same layer.
        _expand_isomorphic_subgraphs(self, node)

    # Phase 5: Rebuild pass assignments from authoritative layer_label_raw.
    # This is NECESSARY (not just defensive) because multiple expansion rounds
    # can leave stale recurrent_group references. See module docstring.
    _rebuild_pass_assignments(self)


def _rebuild_pass_assignments(self) -> None:
    """Phase 5: Rebuild recurrent_group and pass numbers from layer_label_raw.

    WHY NECESSARY: Multiple rounds of ``_expand_isomorphic_subgraphs`` can reassign
    a node to a new group (via ``_finalize_layer_assignments``) while leaving stale
    references in the old group's ``recurrent_group``. Without this cleanup,
    duplicate layer:pass labels would cause dict overwrites during renaming, leading
    to validation failures (e.g., wrong tensor looked up for children_tensor_versions).

    Additionally, Step 6's module suffix mutation means the same equivalence group
    can be processed multiple times (once per module-path variant), creating
    conflicting assignments that must be reconciled.

    HOW IT WORKS: Groups all tensors by their current ``layer_label_raw`` (the
    authoritative group key set by ``_finalize_layer_assignments``), sorts each
    group by realtime order, and rebuilds ``recurrent_group``, ``pass_num``,
    and ``num_passes`` from scratch. O(n), runs once.
    """
    groups = defaultdict(list)
    for entry in self:
        groups[entry.layer_label_raw].append(entry.tensor_label_raw)

    for raw_label, members in groups.items():
        members_sorted = sorted(members, key=lambda x: self[x].creation_order)
        for pass_index, member_label in enumerate(members_sorted):
            member = self[member_label]
            member.recurrent_group = members_sorted
            member.pass_num = pass_index + 1
            member.num_passes = len(members_sorted)


def _expand_isomorphic_subgraphs(self, node: LayerPassLog) -> None:
    """Phase 2: BFS expansion of isomorphic subgraphs from equivalent operations.

    Given a node with multiple equivalent operations, creates one subgraph per
    equivalent node and expands them in lockstep via BFS. The expansion discovers
    which downstream (and upstream) operations are isomorphic across subgraphs.

    Algorithm:
    1. INITIALIZE: All equivalent operations start as one iso group. Each anchors
       its own SubgraphInfo. The BFS queue is seeded with this initial group.
    2. BFS LOOP: Pop an iso-node list from the queue. For each set:
       a. Collect frontier nodes (children, and parents after the first step).
       b. Detect adjacency when a frontier node belongs to another subgraph.
       c. Match frontier nodes across subgraphs by operation_equivalence_type.
       d. Register matched nodes as new iso groups, add to subgraphs, push to queue.
    3. The first BFS step only expands children (not parents) to avoid immediately
       backtracking to the starting nodes' parents.
    4. After BFS completes: refine iso groups (Phase 3) and assign layers (Phase 4).

    Args:
        node: The node whose equivalent_operations set will be expanded.
    """
    equivalent_operation_starting_labels = sorted(list(node.equivalent_operations))

    # Create one SubgraphInfo per starting node.
    sg_info = {}
    for starting_label in equivalent_operation_starting_labels:
        sg_info[starting_label] = SubgraphInfo(starting_node=starting_label)
        if node.uses_params:
            sg_info[starting_label].param_nodes.add(starting_label)

    # Initialize BFS state: all starting nodes form one iso group.
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

    # BFS expansion loop.
    is_first_node = True
    while state.node_stack:
        isomorphic_nodes = sorted(state.node_stack.popleft())
        if len(isomorphic_nodes) == 1:
            continue  # Singleton groups can't produce isomorphic matches.
        _advance_bfs_frontier(self, isomorphic_nodes, state, is_first_node)
        is_first_node = False

    # Phase 3: Refine iso groups to split structurally unrelated operations.
    _refine_iso_groups(self, state)
    # Phase 4: Assign layers based on param sharing and adjacency.
    _finalize_layer_assignments(self, state)


def _refine_iso_groups(
    self,
    state: IsomorphicExpansionState,
) -> None:
    """Phase 3: Refine iso groups by splitting structurally unrelated members.

    WHY NEEDED: The BFS starts with ALL same-equivalence-type operations in one
    iso group and never splits them during expansion. If sin(x) in a loop body
    and sin(y) in a conditional branch share the same equivalence type, the BFS
    wrongly treats them as isomorphic. During adjacency checks, a subgraph asks
    "does the other subgraph contain an isomorphic equivalent of this neighbor?"
    and gets a false positive because the unrefined group includes both.

    HOW IT WORKS:
    1. For each member of a group, compute its direction-aware neighbor signature:
       {(direction, iso_leader)} where direction is "child" or "parent".
    2. Two members are "connected" if their signatures overlap (share at least
       one (direction, iso_leader) pair).
    3. Union-find builds connected components within the group.
    4. If multiple components exist, split the group into sub-groups.

    EXAMPLE: sin(x) in a loop has neighbors {("parent", linear_leader), ("child",
    relu_leader)}. sin(y) in a branch has {("parent", const_leader), ("child",
    multiply_leader)}. Zero overlap -> split -> never merged into same layer.
    """
    # Iterate over a snapshot (list()) because we modify iso_node_groups during iteration.
    for group_leader, members in list(state.iso_node_groups.items()):
        if len(members) <= 1:
            continue

        # For each member, compute its direction-aware neighbor iso signature.
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

        # Union-find to build connected components: members sharing at least
        # one directional neighbor iso group are connected and stay in the same
        # refined group. Members with zero overlap are split apart.
        uf_parent = {member: member for member in members}

        def find(x, uf=uf_parent):
            while uf[x] != x:
                uf[x] = uf[uf[x]]  # path compression
                x = uf[x]
            return x

        def union(x, y, uf=uf_parent):
            rx, ry = find(x), find(y)
            if rx != ry:
                uf[rx] = ry

        # Reverse-index approach: union members sharing a neighbor key.
        # O(members × avg_neighbors) instead of O(members²).
        _reverse_index = defaultdict(list)
        for member_label in members:
            for neighbor_key in member_neighbor_isos[member_label]:
                _reverse_index[neighbor_key].append(member_label)
        for members_with_key in _reverse_index.values():
            if len(members_with_key) > 1:
                first = members_with_key[0]
                for other in members_with_key[1:]:
                    union(first, other)

        components = defaultdict(list)
        for member in members:
            components[find(member)].append(member)

        if len(components) <= 1:
            continue  # All members are connected; no split needed.

        # Split: remove the old group and create new sub-groups, one per component.
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
    """Process one BFS step: collect frontier, match isomorphic nodes, register groups.

    For a set of isomorphic nodes (one per subgraph), collects their frontier
    (children, and parents after the first step), detects subgraph adjacency
    where frontiers overlap with existing subgraphs, then iteratively pops
    candidate nodes and finds isomorphic matches across subgraphs.

    Args:
        current_iso_nodes: Labels of nodes occupying the same structural position
            across subgraphs (one per subgraph).
        state: Mutable BFS expansion state.
        is_first_node: If True, only expand children (not parents) to avoid
            immediately backtracking into the starting nodes' parents.
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
            candidate_node_neighbor_type,  # type: ignore[arg-type]
            candidate_node_subgraph,  # type: ignore[arg-type]
            frontier_nodes,
        )

        _register_isomorphic_group(self, new_equivalent_nodes, state)


def _collect_frontier_and_detect_adjacency(
    self,
    current_iso_nodes: List[str],
    state: IsomorphicExpansionState,
    is_first_node: bool,
) -> Dict[str, Dict[str, List[str]]]:
    """Collect frontier nodes and detect inter-subgraph adjacency.

    For each iso node, examines its children (and parents, unless is_first_node)
    and classifies each neighbor:
    - Already in the SAME subgraph: skip (already processed).
    - Already in a DIFFERENT subgraph: check for adjacency (does the current
      subgraph contain an isomorphic equivalent of the neighbor?). If yes, mark
      the two subgraphs as adjacent via union-find.
    - Not in any subgraph: add to this subgraph's frontier candidates.

    Returns:
        Dict mapping subgraph_label -> {"children": [...], "parents": [...]},
        where each list contains candidate node labels for isomorphic matching.
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
        subgraph_successor_nodes: Dict[str, List[str]] = {"children": [], "parents": []}
        added_neighbors = set()  # #148: prevent shared neighbor double-add
        for node_type in node_types_to_use:
            node_type_field = node_type_fields[node_type]
            for neighbor_label in getattr(node, node_type_field):
                if neighbor_label in node_subgraph.node_set:
                    continue
                elif neighbor_label in state.node_to_subgraph:
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
    """Mark two subgraphs as adjacent in the union-find structure.

    Two subgraphs are adjacent if a frontier node from one subgraph reaches a
    node in another subgraph, AND the other subgraph contains an isomorphic
    equivalent of that neighbor node (checked via iso_node_groups intersection).

    The adjacency dict uses a union-find pattern with shared set objects:
    - Both already tracked (different sets): MERGE the two sets, update ALL
      entries to point to the merged set. This was a critical bug fix (PR #61) —
      previously returned without merging, breaking transitive adjacency.
    - One tracked, one new: add the new one to the existing set.
    - Neither tracked: create a new shared set for both.
    """
    node_subgraph = state.node_to_subgraph[node_label]
    node_subgraph_label = node_subgraph.starting_node
    neighbor_subgraph = state.node_to_subgraph[neighbor_label]
    neighbor_subgraph_label = neighbor_subgraph.starting_node

    # Only mark adjacency if the current subgraph contains an isomorphic
    # equivalent of the neighbor node (validates structural correspondence).
    neighbor_iso_group = state.node_to_iso_leader[neighbor_label]
    nodes_isomorphic_to_neighbor_node = state.iso_node_groups[neighbor_iso_group]
    if len(node_subgraph.node_set.intersection(nodes_isomorphic_to_neighbor_node)) == 0:
        return

    adj = state.adjacent_subgraphs
    if (node_subgraph_label in adj) and (neighbor_subgraph_label in adj):
        # Both already in adjacency dict — merge sets if different.
        if adj[node_subgraph_label] is not adj[neighbor_subgraph_label]:
            merged = adj[node_subgraph_label] | adj[neighbor_subgraph_label]
            for subgraph_key in merged:
                adj[subgraph_key] = merged  # All keys point to the SAME merged set.
        return
    elif (node_subgraph_label in adj) and (neighbor_subgraph_label not in adj):
        adj[node_subgraph_label].add(neighbor_subgraph_label)
        adj[neighbor_subgraph_label] = adj[node_subgraph_label]  # Share same set object.
    elif (node_subgraph_label not in adj) and (neighbor_subgraph_label in adj):
        adj[neighbor_subgraph_label].add(node_subgraph_label)
        adj[node_subgraph_label] = adj[neighbor_subgraph_label]  # Share same set object.
    else:
        # Neither tracked yet — create a new shared set.
        new_adj_set = {node_subgraph_label, neighbor_subgraph_label}
        adj[node_subgraph_label] = new_adj_set
        adj[neighbor_subgraph_label] = new_adj_set


def _pop_frontier_node(
    frontier_nodes: Dict[str, Dict[str, List[str]]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pop the next candidate node from the frontier for isomorphic matching.

    Iterates through subgraphs and neighbor types (children first, then parents),
    returning the first available candidate. Returns (None, None, None) when the
    frontier is exhausted.

    Returns:
        Tuple of (node_label, neighbor_type, subgraph_label), or all-None if empty.
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
    """Find nodes isomorphic to a candidate across other subgraphs' frontiers.

    For each other subgraph's frontier (same neighbor type: children or parents),
    finds the first node with matching operation_equivalence_type. At most one
    match per subgraph is taken, ensuring one-to-one correspondence.

    SAFETY NOTE on pop-during-iteration: The inner loop calls
    ``other_subgraph_nodes.pop(comparison_index)`` which modifies the list
    during iteration. This is safe ONLY because ``break`` follows immediately
    after the pop. Removing the break would cause list corruption.

    After collecting matches, removes collisions where two subgraphs matched
    the same physical node (can happen with shared nodes in diamond topologies).

    Args:
        candidate_node_label: Label of the candidate node from one subgraph.
        candidate_node_neighbor_type: "children" or "parents".
        candidate_node_subgraph: Starting-node label of the candidate's subgraph.
        frontier_nodes: Dict of frontier candidates per subgraph per neighbor type.

    Returns:
        List of (node_label, subgraph_label) tuples for all matched isomorphic nodes.
    """
    candidate_node = self[candidate_node_label]
    candidate_node_operation_equivalence_type = candidate_node.operation_equivalence_type
    new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
    for subgraph_label in frontier_nodes:
        if subgraph_label == candidate_node_subgraph:  # Skip same subgraph.
            continue
        other_subgraph_nodes = frontier_nodes[subgraph_label][candidate_node_neighbor_type]
        for comparison_index, comparison_node_label in enumerate(other_subgraph_nodes):
            comparison_node = self[comparison_node_label]
            if (
                comparison_node.operation_equivalence_type
                == candidate_node_operation_equivalence_type
            ):
                # Pop removes this node from the frontier so it won't be matched again.
                # MUST break immediately after pop — see safety note above.
                new_equivalent_nodes.append(
                    (other_subgraph_nodes.pop(comparison_index), subgraph_label)
                )
                break  # One match per subgraph only.
    new_equivalent_nodes = sorted(new_equivalent_nodes, key=lambda x: x[0])

    # Remove collisions: if the same node appears in multiple (node, subgraph) tuples,
    # discard all occurrences to avoid assigning one node to multiple subgraphs.
    _seen_labels = set()
    _dupe_labels = set()
    for node in new_equivalent_nodes:
        if node[0] in _seen_labels:
            _dupe_labels.add(node[0])
        _seen_labels.add(node[0])
    if _dupe_labels:
        new_equivalent_nodes = [
            node for node in new_equivalent_nodes if node[0] not in _dupe_labels
        ]
    return new_equivalent_nodes


def _register_isomorphic_group(
    self,
    new_isomorphic_nodes: List[Tuple[str, str]],
    state: IsomorphicExpansionState,
) -> None:
    """Register a set of isomorphic nodes as a new iso group.

    Updates all BFS state structures:
    - Creates a new iso group (leader = first node label, alphabetically).
    - Maps each node to the group leader in node_to_iso_leader.
    - Adds each node to its subgraph's node_set (and param_nodes if applicable).
    - Maps each node to its SubgraphInfo in node_to_subgraph.
    - Pushes the node labels onto the BFS queue for further expansion.

    Args:
        new_isomorphic_nodes: List of (node_label, subgraph_label) tuples.
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
            if node.uses_params:
                state.subgraph_info[node_subgraph].param_nodes.add(node_label)
            state.node_to_subgraph[node_label] = state.subgraph_info[node_subgraph]
        state.node_stack.append(equivalent_node_labels)


def _finalize_layer_assignments(
    self,
    state: IsomorphicExpansionState,
) -> None:
    """Phase 4: Assign same-layer labels based on iso groups, params, and adjacency.

    After BFS expansion and iso group refinement, determines which iso-group
    members should become the same layer. Uses ``_merge_iso_groups_to_layers``
    (union-find) to merge nodes whose subgraphs share parameter equivalence
    types OR are topologically adjacent.

    Guard: if a new layer assignment would REDUCE the number of equivalent layers
    for any member node (compared to its current recurrent_group), the
    assignment is skipped. This prevents later expansion rounds from fragmenting
    groups established by earlier rounds.

    For each accepted group, sets layer_label_raw, recurrent_group, pass_num,
    num_passes, and unifies operation_equivalence_type to the canonical
    (first node's) type.

    Args:
        state: Mutable BFS expansion state.
    """
    merged_layer_groups = _merge_iso_groups_to_layers(
        self, state.iso_node_groups, state.node_to_subgraph, state.adjacent_subgraphs
    )

    # Finally, label the nodes corresponding to the same layer.
    for layer_label, layer_nodes in merged_layer_groups.items():
        # Skip if the new layer asssignment reduces the number of equivalent layers.
        if len(layer_nodes) < max([len(self[layer].recurrent_group) for layer in layer_nodes]):
            continue
        # convert to list and sort
        layer_nodes = sorted(list(layer_nodes), key=lambda layer: self[layer].creation_order)  # type: ignore[assignment]
        # Unify operation_equivalence_type: when param-sharing merges nodes
        # from different iso groups (e.g., shared module called from different
        # parent blocks), their equivalence types may differ due to module path
        # suffixes.  Use the first node's type as canonical.
        canonical_equiv_type = self[layer_nodes[0]].operation_equivalence_type  # type: ignore[index]
        for pass_index, node_label in enumerate(layer_nodes):
            node = self[node_label]
            node.layer_label_raw = layer_label
            node.recurrent_group = layer_nodes
            node.pass_num = pass_index + 1
            node.num_passes = len(layer_nodes)
            node.operation_equivalence_type = canonical_equiv_type


def _merge_iso_groups_to_layers(
    self,
    iso_node_groups: Dict[str, List[str]],
    node_to_subgraph: Dict[str, SubgraphInfo],
    adjacent_subgraphs: Dict[str, set],
) -> Dict[str, Set[str]]:
    """Merge iso groups into same-layer groups using union-find with path compression.

    Implements the CORE INVARIANT via two merge passes:

    Pass 1 — Within iso-groups (Rules 2+3):
        For each pair of nodes in the same iso group, merge if their subgraphs:
        - Share overlapping parameter equivalence types (same learned weights used
          in both subgraphs), OR
        - Are topologically adjacent (connected in the adjacency union-find).

    Pass 2 — Cross iso-groups (Rule 1):
        Unconditionally merge ALL nodes that share identical
        ``(func_name, sorted(parent_param_barcodes))``, regardless of
        iso-group membership. This ensures the fundamental invariant: operations
        applying the same function to the same parameters are ALWAYS the same layer.
        Note: different functions on the same params (e.g., __getitem__ vs __add__)
        are NOT merged — the func_name must match.

    Union-find uses lexicographic ordering so the earliest node label is always
    the root, ensuring deterministic group leaders.

    Args:
        iso_node_groups: Maps each iso-group leader to its member labels.
        node_to_subgraph: Maps each node label to its SubgraphInfo.
        adjacent_subgraphs: Union-find adjacency structure (shared set objects).

    Returns:
        Dict mapping each merged-layer leader to the set of node labels in that layer.
        Only groups with 2+ members are returned.
    """
    # Union-find with path compression. Lexicographic ordering ensures the
    # earliest node label is always the root, making group leaders deterministic.
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
            if rx > ry:
                rx, ry = ry, rx  # Lexicographic: smaller label becomes root.
            uf_parent[ry] = rx

    # Collect all iso-group node labels
    all_iso_nodes = set()
    for iso_nodes_orig in iso_node_groups.values():
        all_iso_nodes.update(iso_nodes_orig)

    # Pre-compute param types per subgraph for O(1) lookup in the pair loop (O10).
    _sg_param_types: Dict[str, frozenset] = {}
    for iso_nodes_orig in iso_node_groups.values():
        for node_label in iso_nodes_orig:
            sg = node_to_subgraph[node_label]
            sg_label = sg.starting_node
            if sg_label not in _sg_param_types:
                _sg_param_types[sg_label] = frozenset(
                    self[pnode].operation_equivalence_type for pnode in sg.param_nodes
                )

    # PASS 1: Within iso-groups — merge nodes whose subgraphs share param types or are adjacent.
    for iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(iso_nodes_orig)
        for node1_label, node2_label in it.combinations(iso_nodes, 2):
            node1_subgraph_label = node_to_subgraph[node1_label].starting_node
            node2_subgraph_label = node_to_subgraph[node2_label].starting_node
            overlapping_param_types = (
                _sg_param_types[node1_subgraph_label] & _sg_param_types[node2_subgraph_label]
            )
            subgraphs_are_adjacent = (
                node1_subgraph_label in adjacent_subgraphs
                and node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label]
            )
            if overlapping_param_types or subgraphs_are_adjacent:
                _union(node1_label, node2_label)

    # PASS 2: Cross iso-groups — unconditionally merge by (func, params) identity.
    # This is the FUNDAMENTAL INVARIANT: same function + same params = same layer,
    # regardless of structural position or iso-group membership.
    param_barcode_groups: Dict[tuple, list] = defaultdict(list)
    for node_label in all_iso_nodes:
        node = self[node_label]
        if node.uses_params and node.parent_param_barcodes:
            barcode_key = (node.func_name, tuple(sorted(node.parent_param_barcodes)))
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
