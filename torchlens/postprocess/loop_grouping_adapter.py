"""Backend-neutral recurrence grouping model and service.

This module contains the graph-only inputs needed by Step 7 loop grouping. Backend
finishers are responsible for adapting their postprocess state into this model,
passing only data-flow edges in ``data_parents`` and ``data_children``.
"""

import heapq
import itertools as it
import warnings
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from ..errors._base import TorchLensWarning

FrontierNodes = OrderedDict[str, dict[str, deque[str]]]

# Backend marker ``func_name`` for pseudo-ops (model inputs and outputs). Pseudo-ops
# are never recurrent passes of anything -- a user input or model output executes once
# by definition -- so the topological param-free grouping pass excludes them outright.
_PSEUDO_FUNC_NAME = "none"
_MIN_PARAM_FREE_LOOP_BODY_OPS = 2

# Distinct-target reachability queries a monotone-lane source may answer through
# the exact bounded per-pair BFS before it is treated as dense and its full
# descendant mask is materialized (see :class:`_ReachabilityCache`). The
# post-prefilter cohort sweeps demand about one distinct target per source, so
# sparse sources never pay a graph-wide traversal, while genuinely dense demand
# reaches the O(1) mask lane after two bounded probes.
_DENSE_SOURCE_DISTINCT_QUERIES = 2

# Node-count ceiling for the whole-graph descendant-mask batch DP (r8 R60-3):
# the batch stores one N-bit int per node, an O(N^2/8)-byte table -- ~32MB at
# the ceiling, but ~1.25GB at 100k nodes and ~125GB at 1M, reached by a plain
# straight-line loop (the common case). Above the ceiling, dense sources keep
# the demand-driven per-source BFS: each cached mask is only N/8 bytes and the
# total stays proportional to actual dense demand.
_DENSE_BATCH_MAX_NODES = 16_384

# Byte budget for demand-driven per-source descendant masks above which the
# cache stops RETAINING new masks (queries stay exact through the bounded
# per-pair BFS lane). Bounds worst-case cache memory when pathological demand
# makes every source dense on a huge graph.
_DESCENDANT_MASK_BYTE_BUDGET = 64 * 2**20

# Work budget for the param-free refinement fixpoint (r8 R60-5): the loop
# terminates (refinement only splits), but the round count is O(N) and each
# round is O(E), so a deep unrolled chain paid O(N*E). The budget counts
# member-signature rebuilds across rounds; realistic graphs converge far
# below it (a 5k-op chain costs ~25M visits). On exhaustion every remaining
# multi-member candidate class DISSOLVES to singletons -- an unconverged
# partition may still be too coarse, and over-splitting (no recurrence claim)
# is the honest direction -- with a one-per-capture warning.
_PF_FIXPOINT_WORK_BUDGET = 50_000_000

# Reachability-probe ceiling for one equal-signature cohort's pair triangle
# (r8 R60-4): mutually-unreachable siblings sharing one signature (``ys =
# [x * i for i in range(N)]``) ran the FULL O(k^2) triangle -- ~5e11 probes at
# 1M ops -- because the ``distinct_roots == 1`` early break only helps when
# unions fire. Genuine loop cohorts chain through the consecutive-pair window
# and exit at one root long before the ceiling; a cohort that exhausts it is
# overwhelmingly parallel streams (whose correct outcome IS no union), and the
# one-per-capture warning disclosures the bounded sweep.
_PF_COHORT_PAIR_PROBE_CEILING = 250_000

# A slot color is the site identity a parent contributes to a param-free op's
# signature: ``("param", call_identity)`` for parameterized calls, ``("anchor",
# (equivalence_key, output_slot))`` for anchored (module-bound or buffer) ops,
# ``("class", leader_label)`` for unanchored param-free ops (their CURRENT topological
# class), and ``("ext", label)`` for everything external (pseudo-ops, pruned or
# non-eligible parents). Colors deliberately carry NO pass index: repeated passes of
# one site contribute one color, which is what lets lockstep loop iterations produce
# EQUAL signatures.
_SlotColor = tuple[str, object]


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
        an expansion round. Torch supplies its existing ``equivalent_ops`` set,
        filtered to members sharing this node's ``equivalence_key``: when a
        backend subdivides a raw equivalence class (torch's argsig split), the
        membership here must agree with the split, or foreign-key members seed
        cross-contaminated isomorphic expansions (deep-hunt L2).
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
    output_slot: int | None = None
    """Zero-based output slot for multi-output operations, ``None`` for single-output.

    Co-outputs of ONE call (``h, c = lstm_cell(...)``) occupy distinct output slots.
    Distinct slots of the same call are distinct layers, never sequential passes of
    each other: an N-step ``nn.LSTMCell`` loop is one N-pass h-layer plus one N-pass
    c-layer, mirroring how ``torch.max`` values/indices already split. Same-function,
    same-parameter merging must therefore never fuse nodes across output slots."""
    module_site: tuple[str, ...] | None = None
    """Module ADDRESS stack (addresses only, no pass numbers), or ``None`` if unknown.

    Exact parameter identity must never override module identity: two DISTINCT modules
    that deliberately share a weight tensor (a tied ``encoder``/``decoder`` pair) are
    two semantic sites, not two passes of one recurrent layer. A genuinely reused
    module (ALBERT-style, one ``nn.Module`` called N times) keeps ONE address across
    calls and still groups. ``None`` (backends that do not supply the field) preserves
    the historical parameter-only behavior."""
    arg_signature: str | None = None
    """Structural fingerprint of the call's NON-TENSOR arguments, or ``None`` if unknown.

    Same-parameter identity must not override call semantics: two ``F.conv2d`` calls
    sharing one kernel but differing in ``padding``/``stride``/``dilation``/``groups``
    are different operations, never recurrent passes of one layer. The fingerprint
    deliberately excludes tensor arguments AND their shapes so genuine variable-length
    recurrence (a loop whose activations shrink each step) keeps one signature across
    passes. ``None`` preserves the historical behavior."""
    site_key: str | None = None
    """Portable ``site_key_v1`` structural-position identity (:mod:`._site_key`).

    Minted by each producer's node builder from raw records in execution order;
    policy-independent (identical whether grouping runs, degrades, or is off)."""
    recurrence_anchored: bool = False
    """Whether this op has a reused persistent identity that anchors genuine recurrence.

    Anchored ops are calls of a named submodule (non-empty ``modules``) or stateful buffer
    nodes (``is_buffer``). Repeating such an identity -- one ``nn.ReLU`` called four times, a
    buffer rewritten each iteration -- is real recurrence even when the body is a single op,
    so the param-free minimum-body-size guard does not apply to them. Bare functional ops
    (``torch.tanh(x)`` in the parent forward) are not anchored and must clear the body-size
    bar to be grouped, which rejects the ``y = tanh(x); z = tanh(y)`` false positive."""


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
    site_key:
        This NODE's portable structural-position key (op-granular, never a
        group fact: site-spanning groups carry distinct member keys).
    """

    layer_label: str
    recurrent_labels: tuple[str, ...]
    pass_index: int
    num_passes: int
    equivalence_key: str
    site_key: str | None = None


_ParamCallIdentity = tuple[
    str,
    tuple[str, ...],
    int | None,
    tuple[str, ...] | None,
    str | None,
]


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
    output_slot: int | None = None
    recurrence_anchored: bool = False
    module_site: tuple[str, ...] | None = None
    arg_signature: str | None = None
    site_key: str | None = None


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
    param_nodes: set[str] = field(default_factory=set)
    node_set: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Register the starting node in this subgraph."""
        self.node_set.add(self.starting_node)


@dataclass
class IsomorphicExpansionState:
    """Mutable state threaded through BFS isomorphic-subgraph expansion."""

    iso_node_groups: OrderedDict[str, list[str]]
    node_to_iso_leader: OrderedDict[str, str]
    subgraph_info: dict[str, SubgraphInfo]
    node_to_subgraph: dict[str, SubgraphInfo]
    adjacent_subgraphs: dict[str, set[str]]
    node_stack: deque[list[str]]


@dataclass
class _GroupingWorkspace:
    """Mutable recurrence grouping workspace built from a neutral graph."""

    nodes: dict[str, _MutableRecurrenceNode]
    raw_labels: tuple[str, ...]
    source_labels: tuple[str, ...]
    eligible_labels: set[str]

    @classmethod
    def from_graph(
        cls: type["_GroupingWorkspace"],
        graph: RecurrenceGroupingGraph,
    ) -> "_GroupingWorkspace":
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
                output_slot=node.output_slot,
                recurrence_anchored=node.recurrence_anchored,
                module_site=node.module_site,
                arg_signature=node.arg_signature,
                site_key=node.site_key,
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

        Notes
        -----
        All members of a finalized group share ONE ``recurrent_labels`` list object
        (:func:`_rebuild_pass_assignments`), so the tuple and the label-to-index map
        are memoized per shared list -- one shared tuple per group instead of one
        private P-tuple per member, and an O(1) index lookup instead of an O(P)
        self-scan. The memo is keyed by ``id()`` of lists kept alive by
        ``self.nodes`` for the duration of the call and is local to this call, so
        no cross-run staleness is possible. ``equivalence_key`` stays
        member-specific.
        """
        shared_groups: dict[int, tuple[tuple[str, ...], dict[str, int]]] = {}
        assignments: dict[str, RecurrenceAssignment] = {}
        for label, node in self.nodes.items():
            group = shared_groups.get(id(node.recurrent_labels))
            if group is None:
                members = tuple(node.recurrent_labels)
                group = (members, {member: index for index, member in enumerate(members)})
                shared_groups[id(node.recurrent_labels)] = group
            members, index_by_label = group
            index = index_by_label.get(label)
            if index is None:
                continue
            assignments[label] = RecurrenceAssignment(
                layer_label=node.layer_label,
                recurrent_labels=members,
                pass_index=index + 1,
                num_passes=len(members),
                equivalence_key=node.equivalence_key,
                site_key=node.site_key,
            )
        return assignments


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

    _assign_param_free_layers(workspace)
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
            dict.fromkeys(
                equivalent_operation_starting_labels, equivalent_operation_starting_labels[0]
            )
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

        # Directly chained ANCHORED members are consecutive passes of one reused
        # persistent identity (one ``nn.ReLU`` module applied to its own output) and
        # must survive refinement together. Without this, a group of exactly two has
        # asymmetric endpoint signatures (one member only feeds the group, the other
        # is only fed by it), gets split into singletons here, and the singletons
        # produce zero merge candidates downstream -- so the documented
        # ``recurrence_anchored`` bypass never ran for its canonical two-call case.
        # Groups of three or more survive because middle members carry both
        # directional signatures. Bare functional chains (``tanh(tanh(x))``) are not
        # anchored and still split, preserving the straight-chain false-positive
        # guard.
        member_set = set(members)
        for member_label in members:
            member_node = workspace.nodes[member_label]
            if not member_node.recurrence_anchored:
                continue
            for neighbor in member_node.data_children:
                if neighbor in member_set and workspace.nodes[neighbor].recurrence_anchored:
                    union(member_label, neighbor)

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

    # Direction-major drain: every subgraph's children before any parents, so a
    # loop-carried value shared between a child frontier and a parent frontier
    # is deterministically absorbed through its child position first and its
    # parent-side appearance records iteration adjacency at pop time. Callees
    # only ever REMOVE frontier entries (isomorphic matches delete from OTHER
    # subgraphs' same-direction deques; nothing refills a bucket), so draining
    # each live deque in place visits candidates in exactly the order the
    # first-non-empty-bucket rescan in :func:`_pop_frontier_node` produced,
    # without restarting the bucket-product scan after every pop.
    key_index = _build_frontier_key_index(workspace, frontier_nodes)
    for candidate_node_neighbor_type in ("children", "parents"):
        direction_index = key_index[candidate_node_neighbor_type]
        for candidate_node_subgraph in frontier_nodes:
            bucket = frontier_nodes[candidate_node_subgraph][candidate_node_neighbor_type]
            while bucket:
                candidate_node_label = bucket.popleft()
                _discard_from_key_index(
                    direction_index,
                    workspace.nodes[candidate_node_label].equivalence_key,
                    candidate_node_subgraph,
                )

                if candidate_node_label in state.node_to_subgraph:
                    # The candidate was absorbed into another subgraph after this
                    # frontier was collected (a loop-carried value: child of body
                    # ``i``, parent of body ``i + 1``). It cannot be matched again,
                    # but its presence on this subgraph's frontier is exactly the
                    # evidence that consecutive iterations are directly adjacent --
                    # record that instead.
                    _record_frontier_adjacency(
                        candidate_node_subgraph,
                        candidate_node_label,
                        state,
                    )
                    continue

                new_equivalent_nodes = _find_isomorphic_matches(
                    workspace,
                    candidate_node_label,
                    candidate_node_neighbor_type,
                    candidate_node_subgraph,
                    frontier_nodes,
                    direction_index,
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

    _canonicalize_frontier(workspace, frontier_nodes)
    return frontier_nodes


def _canonicalize_frontier(
    workspace: _GroupingWorkspace,
    frontier_nodes: FrontierNodes,
) -> None:
    """Make frontier contents a canonical function of the graph, not capture order.

    Grouping must be a well-defined function of the captured DAG: two mathematically
    identical models differing only in independent-sibling statement order (or a
    cross-backend feed ordering children differently) must produce the SAME layer
    partition. Two canonicalizations enforce this:

    * Every frontier deque is sorted by ``raw_order``, so which same-key neighbor an
      isomorphic match absorbs no longer depends on the incidental order of a node's
      ``data_children`` tuple.
    * A label appearing in the SAME direction of MORE THAN ONE subgraph's frontier is
      a single node shared between candidate loop bodies at the same relative
      position. One node cannot be a per-iteration isomorphic copy in two bodies at
      once, so such labels are dropped from that direction instead of being greedily
      absorbed by whichever subgraph popped first (the previous behavior, which let
      sibling capture order fabricate or miss loops and inflate the body-size
      guard). A label shared across OPPOSITE directions is kept: a loop-carried
      value is simultaneously a child of body ``i`` and a parent of body ``i + 1``,
      and dropping it would sever the chain that makes consecutive iterations
      adjacent (the pop path records that adjacency instead).

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    None
        Mutates ``frontier_nodes`` in place.
    """
    for direction in ("children", "parents"):
        label_counts: Counter[str] = Counter()
        for direction_buckets in frontier_nodes.values():
            label_counts.update(direction_buckets[direction])
        shared_labels = {label for label, count in label_counts.items() if count > 1}
        for direction_buckets in frontier_nodes.values():
            bucket = direction_buckets[direction]
            kept = [label for label in bucket if label not in shared_labels]
            kept.sort(key=lambda label: workspace.nodes[label].raw_order)
            direction_buckets[direction] = deque(kept)


def _record_subgraph_adjacency(
    node_label: str,
    neighbor_label: str,
    state: IsomorphicExpansionState,
) -> None:
    """Mark two subgraphs as DIRECTLY adjacent in the expansion state.

    Adjacency is pairwise, never transitive. Consecutive iterations of one loop are
    directly adjacent and still chain into a single layer through union-find in
    :func:`_merge_iso_groups_to_layers`; recording transitively merged adjacency
    SETS (the previous behavior) instead let two subgraphs many hops apart -- e.g.
    the first ``tanh`` of loop A and the last ``tanh`` of a chained but distinct
    loop B -- count as "adjacent" and merge into one incoherent recurrent layer.

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
    node_subgraph_label = state.node_to_subgraph[node_label].starting_node
    _record_frontier_adjacency(node_subgraph_label, neighbor_label, state)


def _record_frontier_adjacency(
    subgraph_label: str,
    neighbor_label: str,
    state: IsomorphicExpansionState,
) -> None:
    """Record direct adjacency between a subgraph and a neighbor's subgraph.

    Parameters
    ----------
    subgraph_label:
        Starting-node label of the subgraph whose frontier met the neighbor.
    neighbor_label:
        Neighbor label already assigned to a subgraph.
    state:
        Mutable isomorphic expansion state.

    Returns
    -------
    None
        Mutates ``state.adjacent_subgraphs``.
    """
    node_subgraph = state.subgraph_info[subgraph_label]
    neighbor_subgraph_label = state.node_to_subgraph[neighbor_label].starting_node
    if neighbor_subgraph_label == subgraph_label:
        return

    neighbor_iso_group = state.node_to_iso_leader[neighbor_label]
    nodes_isomorphic_to_neighbor_node = state.iso_node_groups[neighbor_iso_group]
    if len(node_subgraph.node_set.intersection(nodes_isomorphic_to_neighbor_node)) == 0:
        return

    adj = state.adjacent_subgraphs
    adj.setdefault(subgraph_label, set()).add(neighbor_subgraph_label)
    adj.setdefault(neighbor_subgraph_label, set()).add(subgraph_label)


def _pop_frontier_node(
    frontier_nodes: FrontierNodes,
) -> tuple[str | None, str | None, str | None]:
    """Pop the next frontier candidate for isomorphic matching.

    Parameters
    ----------
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        Candidate label, neighbor type, and subgraph label, or all ``None``.

    Notes
    -----
    Iteration is direction-major (every subgraph's children before any parents) so
    that a loop-carried value shared between a child frontier and a parent frontier
    is deterministically absorbed through its child position first; its parent-side
    appearance then records iteration adjacency at pop time.

    This helper is the reference specification of the frontier pop order. The
    production loop in :func:`_advance_bfs_frontier` drains the live deques
    directly in the identical order (buckets only ever shrink during a frontier
    step, so a single direction-major pass is pop-for-pop equivalent) instead of
    restarting this scan from the first bucket after every pop.
    """
    for neighbor_type, subgraph_label in it.product(["children", "parents"], frontier_nodes):
        subgraph_neighbors = frontier_nodes[subgraph_label][neighbor_type]
        if len(subgraph_neighbors) > 0:
            candidate_node_label = subgraph_neighbors.popleft()
            return candidate_node_label, neighbor_type, subgraph_label
    return None, None, None


def _build_frontier_key_index(
    workspace: _GroupingWorkspace,
    frontier_nodes: FrontierNodes,
) -> dict[str, dict[str, dict[str, int]]]:
    """Index one frontier step's labels by direction and equivalence key.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    frontier_nodes:
        Candidate frontier nodes by subgraph and neighbor direction.

    Returns
    -------
    dict[str, dict[str, dict[str, int]]]
        ``direction -> equivalence_key -> {subgraph_label: count}``, with subgraph
        labels in frontier order. Buckets only ever shrink during a frontier step,
        so the index is maintained by :func:`_discard_from_key_index` at the two
        removal points (candidate pop, isomorphic-match deletion) and lets
        :func:`_find_isomorphic_matches` visit only key-bearing subgraphs instead
        of scanning every subgraph's deque per candidate.
    """
    key_index: dict[str, dict[str, dict[str, int]]] = {"children": {}, "parents": {}}
    for direction, direction_index in key_index.items():
        for subgraph_label, direction_buckets in frontier_nodes.items():
            for label in direction_buckets[direction]:
                subgraph_counts = direction_index.setdefault(
                    workspace.nodes[label].equivalence_key, {}
                )
                subgraph_counts[subgraph_label] = subgraph_counts.get(subgraph_label, 0) + 1
    return key_index


def _discard_from_key_index(
    direction_index: dict[str, dict[str, int]],
    equivalence_key: str,
    subgraph_label: str,
) -> None:
    """Record one label's removal from a subgraph's frontier deque.

    Parameters
    ----------
    direction_index:
        One direction's ``equivalence_key -> {subgraph_label: count}`` index.
    equivalence_key:
        Equivalence key of the removed label.
    subgraph_label:
        Subgraph whose deque the label was removed from.
    """
    subgraph_counts = direction_index[equivalence_key]
    remaining = subgraph_counts[subgraph_label] - 1
    if remaining:
        subgraph_counts[subgraph_label] = remaining
    else:
        del subgraph_counts[subgraph_label]


def _find_isomorphic_matches(
    workspace: _GroupingWorkspace,
    candidate_node_label: str,
    candidate_node_neighbor_type: str,
    candidate_node_subgraph: str,
    frontier_nodes: FrontierNodes,
    direction_index: dict[str, dict[str, int]],
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
    direction_index:
        This direction's ``equivalence_key -> {subgraph_label: count}`` index over
        ``frontier_nodes`` (:func:`_build_frontier_key_index`).

    Returns
    -------
    list[tuple[str, str]]
        Matched ``(node_label, subgraph_label)`` pairs.

    Notes
    -----
    Only subgraphs whose frontier still holds the candidate's equivalence key are
    visited. This is exact: within one call, subgraphs are independent -- each
    contributes at most its FIRST key-matching node from its own deque, no
    cross-subgraph state exists, and the result is sorted by node label -- so
    skipping subgraphs with zero key-bearing nodes (where the scan would find
    nothing and mutate nothing) returns the identical set with identical frontier
    mutations.
    """
    candidate_node = workspace.nodes[candidate_node_label]
    candidate_node_equivalence_key = candidate_node.equivalence_key
    new_equivalent_nodes = [(candidate_node_label, candidate_node_subgraph)]
    for subgraph_label in list(direction_index.get(candidate_node_equivalence_key, ())):
        if subgraph_label == candidate_node_subgraph:
            continue
        other_subgraph_nodes = frontier_nodes[subgraph_label][candidate_node_neighbor_type]
        for comparison_index, comparison_node_label in enumerate(other_subgraph_nodes):
            comparison_node = workspace.nodes[comparison_node_label]
            if comparison_node.equivalence_key == candidate_node_equivalence_key:
                del other_subgraph_nodes[comparison_index]
                _discard_from_key_index(
                    direction_index,
                    candidate_node_equivalence_key,
                    subgraph_label,
                )
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
        layer_nodes = sorted(layer_nodes_set, key=lambda layer: workspace.nodes[layer].raw_order)
        if len(layer_nodes) < max(
            [len(workspace.nodes[layer].recurrent_labels) for layer in layer_nodes]
        ):
            continue
        canonical_equiv_type = workspace.nodes[layer_nodes[0]].equivalence_key
        for _pass_index, grouped_node_label in enumerate(layer_nodes):
            node = workspace.nodes[grouped_node_label]
            node.layer_label = layer_label
            node.recurrent_labels = layer_nodes
            node.equivalence_key = canonical_equiv_type


class _ReachabilityCache:
    """Per-grouping-run reachability cache backing :func:`_seed_reaches`.

    The bare-op fixpoint's entry-admission and cohort sweeps issue reachability
    queries whose demand SHAPE varies by regime. Before the entry-admission
    prefilter emptied the degenerate pair triangle, O(N) distinct sources each
    answered O(N) queries, so materializing the full descendant set per source
    was the right trade. After it, the surviving phase-1 sweeps issue roughly
    ONE distinct query per source, and an unconditional full-graph BFS per
    source is itself the quadratic term. The cache therefore adapts per source:

    * The first :data:`_DENSE_SOURCE_DISTINCT_QUERIES` distinct targets of a
      source are answered with the exact historical raw-order-bounded per-pair
      BFS (identical pairs hit the pair memo first). Sparse sources never pay
      a graph-wide traversal.
    * A source demanding more distinct targets is dense: its FULL descendant
      set is materialized and every later query for it is an O(1) membership
      test. The first dense source triggers ONE batch DP pass that builds ALL
      masks from child masks in reverse topological order -- one int-OR per
      edge instead of one full BFS per dense source -- guarded by a strict
      insertion-order topology check with the historical per-source BFS as the
      fallback (:meth:`_batch_build_descendant_masks`).

    Descendant sets are stored as per-node-index bitmask ints, not label sets,
    keeping the whole cache in the low-megabytes range at 1,024-step traces.

    The historical query bounds its search window by the destination's
    ``raw_order``; the unbounded per-source set gives the identical answer only
    because capture order is topological, which makes the bound pure pruning
    (every node on a directed path to the destination finishes at or below the
    destination's ``raw_order``). That premise is verified once per run with a
    single O(E) edge scan; a workspace carrying any raw-order-violating edge
    falls back to the exact historical bounded per-pair BFS for every query.

    Workspace topology (the node set, ``data_children`` tuples, and
    ``raw_order``) is frozen once :meth:`_GroupingWorkspace.from_graph` builds
    the workspace -- grouping only mutates label/assignment fields -- so cached
    descendant sets can never go stale within the run that owns the cache.
    """

    __slots__ = (
        "_workspace",
        "_order_monotone",
        "_bit_index",
        "_descendant_bits",
        "_pair_memo",
        "_sparse_query_counts",
        "_batch_attempted",
    )

    def __init__(self, workspace: _GroupingWorkspace) -> None:
        self._workspace = workspace
        self._order_monotone: bool | None = None
        self._bit_index: dict[str, int] = {}
        self._descendant_bits: dict[str, int] = {}
        self._pair_memo: dict[tuple[str, str], bool] = {}
        self._sparse_query_counts: dict[str, int] = {}
        self._batch_attempted = False

    def reaches_from_earlier(self, src_label: str, dst_label: str) -> bool:
        """Return whether ``src_label`` reaches ``dst_label`` along data edges.

        Parameters
        ----------
        src_label:
            Source label; callers orient the pair so this is the seed whose
            ``raw_order`` does not exceed the destination's.
        dst_label:
            Destination label.

        Returns
        -------
        bool
            ``True`` when ``dst_label`` is ``src_label`` or lies downstream
            of it.
        """
        if src_label == dst_label:
            return True
        if self._order_monotone is None:
            self._prepare()
        if not self._order_monotone:
            return self._bounded_pair_query(src_label, dst_label)
        mask = self._descendant_bits.get(src_label)
        if mask is None:
            queries_seen = self._sparse_query_counts.get(src_label, 0)
            if queries_seen < _DENSE_SOURCE_DISTINCT_QUERIES:
                cached = self._pair_memo.get((src_label, dst_label))
                if cached is not None:
                    return cached
                self._sparse_query_counts[src_label] = queries_seen + 1
                return self._bounded_pair_query(src_label, dst_label)
            if self._mask_budget_exhausted():
                # Mask retention budget spent (r8 R60-3): stay on the exact
                # bounded pair lane instead of growing the cache further.
                return self._bounded_pair_query(src_label, dst_label)
            mask = self._acquire_descendant_mask(src_label)
        return (mask >> self._bit_index[dst_label]) & 1 == 1

    def _prepare(self) -> None:
        """Index the frozen node set and verify the raw-order edge invariant."""
        nodes = self._workspace.nodes
        self._bit_index = {label: index for index, label in enumerate(nodes)}
        self._order_monotone = all(
            child not in nodes or nodes[child].raw_order >= node.raw_order
            for node in nodes.values()
            for child in node.data_children
        )

    def _acquire_descendant_mask(self, src_label: str) -> int:
        """Return the mask for a dense source, batch-building all masks once.

        The first dense source attempts the whole-graph batch DP; when the
        strict insertion-order topology premise fails, that source and every
        later dense source keep the exact historical per-source BFS.
        """
        if not self._batch_attempted:
            self._batch_attempted = True
            # Node-count ceiling (r8 R60-3): the batch stores one N-bit int
            # per node -- an O(N^2/8)-byte table that reaches ~1.25GB at 100k
            # nodes on a plain straight-line loop. Above the ceiling the
            # demand-driven per-source BFS keeps memory proportional to
            # actual dense demand.
            if len(self._workspace.nodes) <= _DENSE_BATCH_MAX_NODES:
                self._batch_build_descendant_masks()
        mask = self._descendant_bits.get(src_label)
        if mask is None:
            mask = self._build_descendant_mask(src_label)
        return mask

    def _mask_budget_exhausted(self) -> bool:
        """Return whether retained per-source masks have spent the byte budget."""
        node_count = len(self._bit_index)
        if node_count == 0:
            return False
        mask_bytes = node_count // 8 + 1
        return len(self._descendant_bits) * mask_bytes >= _DESCENDANT_MASK_BYTE_BUDGET

    def _batch_build_descendant_masks(self) -> None:
        """Build every descendant mask in one reverse-insertion-order DP pass.

        ``mask[node] = OR over eligible children c of (bit(c) | mask[c])``
        computes all masks with one int-OR per edge, replacing one full BFS per
        dense source. The DP consumes children's masks while walking the node
        mapping in reverse, so it requires every eligible data child to appear
        STRICTLY AFTER its parent in the mapping. ``_order_monotone`` is not
        sufficient evidence for that -- it accepts ``raw_order`` ties, and a
        tied edge inserted child-first would silently drop the child's own
        descendants from the parent's mask. On any violating edge the batch is
        abandoned (no partial state) and dense sources fall back to the exact
        per-source BFS.
        """
        nodes = self._workspace.nodes
        bit_index = self._bit_index
        for label, node in nodes.items():
            position = bit_index[label]
            for child in node.data_children:
                child_position = bit_index.get(child)
                if child_position is not None and child_position <= position:
                    return
        masks: dict[str, int] = {}
        for label in reversed(nodes):
            mask = 0
            for child in nodes[label].data_children:
                child_position = bit_index.get(child)
                if child_position is not None:
                    mask |= (1 << child_position) | masks[child]
            masks[label] = mask
        self._descendant_bits = masks

    def _build_descendant_mask(self, src_label: str) -> int:
        """Run one full BFS from ``src_label`` and cache its descendant bitmask."""
        nodes = self._workspace.nodes
        bit_index = self._bit_index
        mask = 0
        stack = [src_label]
        seen = {src_label}
        while stack:
            for child in nodes[stack.pop()].data_children:
                if child in seen or child not in nodes:
                    continue
                seen.add(child)
                stack.append(child)
                mask |= 1 << bit_index[child]
        self._descendant_bits[src_label] = mask
        return mask

    def _bounded_pair_query(self, src_label: str, dst_label: str) -> bool:
        """Answer one query with the historical raw-order-bounded per-pair BFS."""
        key = (src_label, dst_label)
        cached = self._pair_memo.get(key)
        if cached is not None:
            return cached
        nodes = self._workspace.nodes
        dst_order = nodes[dst_label].raw_order
        stack = [src_label]
        seen = {src_label}
        found = False
        while stack:
            current = stack.pop()
            if current == dst_label:
                found = True
                break
            for child in nodes[current].data_children:
                child_node = nodes.get(child)
                if child_node is None or child in seen or child_node.raw_order > dst_order:
                    continue
                seen.add(child)
                stack.append(child)
        self._pair_memo[key] = found
        return found


def _seed_reaches(
    workspace: _GroupingWorkspace,
    node1_label: str,
    node2_label: str,
    memo: _ReachabilityCache,
) -> bool:
    """Return whether one seed reaches the other along directed data edges.

    Capture order is a topological order (parents precede children), so
    reachability is only possible from the earlier-captured seed to the later one,
    and the search window is bounded by the later seed's ``raw_order``.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    node1_label:
        First seed label.
    node2_label:
        Second seed label.
    memo:
        Per-merge cache of resolved reachability queries.

    Returns
    -------
    bool
        ``True`` when a directed data path connects the two seeds.
    """
    src_label, dst_label = node1_label, node2_label
    if workspace.nodes[src_label].raw_order > workspace.nodes[dst_label].raw_order:
        src_label, dst_label = dst_label, src_label
    return memo.reaches_from_earlier(src_label, dst_label)


def _param_call_identity(node: _MutableRecurrenceNode) -> _ParamCallIdentity:
    """Return the full call identity two parameterized ops must share to be one layer.

    Recurrent passes of ONE layer are repeated executions of the SAME call: same
    function, same parameters, same output slot, same module address, and same
    non-tensor structural arguments. Exact parameter identity alone must not
    override the other axes:

    * ``module_site`` -- two DISTINCT modules deliberately sharing a weight tensor
      (tied ``encoder``/``decoder``) are two semantic sites, never a false 2-pass
      recurrent layer. A genuinely reused module keeps one address and still groups.
    * ``arg_signature`` -- one kernel applied with ``padding=0`` and ``padding=1``
      is two different operations with different output structure, not recurrence.

    ``None`` values (backends that do not supply the newer fields) compare equal to
    each other, preserving the historical parameter-only behavior for those feeds.

    Parameters
    ----------
    node:
        Parameterized workspace node.

    Returns
    -------
    _ParamCallIdentity
        Hashable identity tuple.
    """

    return (
        node.func_name,
        tuple(sorted(node.param_barcodes)),
        node.output_slot,
        node.module_site,
        node.arg_signature,
    )


def _topology_anchor_ancestry(workspace: _GroupingWorkspace) -> dict[str, bool]:
    """Return whether each node descends from a parameterized or anchored site.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace in topological capture order.

    Returns
    -------
    dict[str, bool]
        Node label to whether its upstream data topology contains a persistent
        parameterized or anchored site.
    """
    has_anchor: dict[str, bool] = {}
    for label in workspace.raw_labels:
        node = workspace.nodes[label]
        has_anchor[label] = (
            (node.uses_params and bool(node.param_barcodes))
            or node.recurrence_anchored
            or any(has_anchor.get(parent, False) for parent in node.data_parents)
        )
    return has_anchor


def _param_free_adjacency_merge_allowed(
    workspace: _GroupingWorkspace,
    node1_label: str,
    node2_label: str,
    subgraph_a: SubgraphInfo,
    subgraph_b: SubgraphInfo,
) -> bool:
    """Return whether an adjacency-only, parameter-free merge is a real loop.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    node1_label:
        Seed label of the first candidate subgraph.
    node2_label:
        Seed label of the second candidate subgraph.
    subgraph_a:
        First candidate loop-body subgraph.
    subgraph_b:
        Structurally isomorphic partner subgraph.

    Returns
    -------
    bool
        ``True`` for a reused persistent identity or when both repeated bodies
        span at least two operations. The size floor rejects a straight chain
        of adjacent same-function calls as false single-op recurrence.
    """
    if (
        workspace.nodes[node1_label].recurrence_anchored
        or workspace.nodes[node2_label].recurrence_anchored
    ):
        return True
    return min(len(subgraph_a.node_set), len(subgraph_b.node_set)) >= (
        _MIN_PARAM_FREE_LOOP_BODY_OPS
    )


# ---------------------------------------------------------------------------
# Topological parameter-free grouping
# ---------------------------------------------------------------------------
#
# Which N-pass layer (or singleton) a bare (unanchored, parameter-free) op belongs
# to is a DERIVED quantity: it follows deterministically from the already-solved
# parameterized/anchored grouping plus data-flow topology. The pass below computes
# it as a coarsest-fixpoint partition:
#
# * Every bare op gets a SIGNATURE: the multiset of its parents' slot colors
#   (parameterized call identity / anchored key / bare-op class / external label).
#   Colors carry no pass index, so lockstep iterations of one source-code site
#   produce EQUAL signatures, while a site whose parameterized direct parents
#   differ in LAYER (``enc`` versus ``dec``) is split by construction -- the exact
#   DAG evidence the r22/r23/r24 seals demanded.
# * Classes start maximally coarse (one class per equivalence key and output slot)
#   and are only ever SPLIT, so every division is backed by definite evidence and
#   iteration converges. Signatures reference the classes themselves (a ``tanh``
#   fed by an ``add`` carries the add CLASS as its color), so sibling ops in one
#   loop body partition in lockstep by construction.
# * The one honest same-site signature difference is the LOOP ENTRY: pass 1 reads
#   state produced before the loop, later passes read it through the feedback
#   wire. Entry pairs are re-admitted through a guarded carry-slot exemption
#   (:func:`_pf_entry_union_allowed`) instead of context-set heuristics. Entry
#   matching is stream-local: a connected equal-signature cohort is already a
#   realized site, while disconnected identical entries may each pair with the
#   one continuation whose carry they reach.
# * Direct parameterized/anchored consumers provide the second side of topology
#   when a multi-parent view is ambiguous. Only complete consumer sites (every
#   pass fed by the same-key universe) can decide a boundary, preventing a
#   terminal next-loop consumer from stealing the preceding loop's last call.
# * Loop-invariant-fed repeats (a factory op or a recompute of a pre-loop value
#   inside the body) carry no sequencing evidence on the parent side; they are
#   sequenced through their CONSUMERS (:func:`_pf_child_route_allows`).


def _reaches_forward(
    workspace: _GroupingWorkspace,
    src_label: str,
    dst_label: str,
    memo: _ReachabilityCache,
) -> bool:
    """Return whether ``src_label`` reaches ``dst_label`` along directed data edges.

    Unlike :func:`_seed_reaches` this query is DIRECTIONAL and reflexive: the
    loop-carried-dependence certificate requires that the later call's carry
    parent was computed FROM the earlier call (possibly the earlier call itself),
    never merely that the two are connected in some order.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    src_label:
        Candidate producer label.
    dst_label:
        Candidate consumer label.
    memo:
        Shared cache of resolved reachability queries.

    Returns
    -------
    bool
        ``True`` when ``dst_label`` is ``src_label`` or lies downstream of it.
    """
    if src_label == dst_label:
        return True
    if workspace.nodes[dst_label].raw_order < workspace.nodes[src_label].raw_order:
        return False
    return _seed_reaches(workspace, src_label, dst_label, memo)


def _pf_slot_color(
    workspace: _GroupingWorkspace,
    label: str,
    class_of: dict[str, str],
) -> _SlotColor:
    """Return the site-identity color a node contributes as a signature slot.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    label:
        Node label to color (need not be eligible; unknown labels are external).
    class_of:
        Current bare-op class assignment (label to class leader).

    Returns
    -------
    _SlotColor
        Kind-tagged site identity. Parameterized calls color by their full call
        identity (all passes of one layer share a color); anchored ops by their
        module-suffixed equivalence key; bare ops by their CURRENT topological
        class; everything else (pseudo-ops, pruned or non-eligible parents) by
        its own label.
    """
    node = workspace.nodes.get(label)
    if node is None:
        return ("ext", label)
    if node.uses_params and node.param_barcodes:
        return ("param", _param_call_identity(node))
    if node.recurrence_anchored:
        return ("anchor", (node.equivalence_key, node.output_slot))
    leader = class_of.get(label)
    if leader is not None:
        return ("class", leader)
    return ("ext", label)


def _pf_realization_counts(workspace: _GroupingWorkspace) -> dict[_SlotColor, int]:
    """Count how many calls realize each parameterized/anchored slot color.

    A color realized ONCE is a one-shot site (a pre-loop embedding, an init); a
    color realized several times is itself a recurring site. The distinction is
    load-bearing for entry admission: a genuine loop entry's odd parent is the
    one-shot pre-loop producer, never a realized recurrent site -- an op whose
    odd parent recurs sits BESIDE a loop (an interior or a boundary), and
    admitting it is exactly the historical cross-boundary straddle.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    dict[_SlotColor, int]
        Realization counts for ``("param", ...)`` and ``("anchor", ...)`` colors.
    """
    counts: dict[_SlotColor, int] = defaultdict(int)
    for node in workspace.nodes.values():
        if node.uses_params and node.param_barcodes:
            counts[("param", _param_call_identity(node))] += 1
        elif node.recurrence_anchored:
            counts[("anchor", (node.equivalence_key, node.output_slot))] += 1
    return dict(counts)


def _pf_direct_site_passes(
    workspace: _GroupingWorkspace,
) -> dict[str, tuple[_SlotColor, int]]:
    """Return the site color and pass rank of every direct topology anchor.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    dict[str, tuple[_SlotColor, int]]
        Parameterized and anchored node labels mapped to their pass-blind site
        color and one-based capture-order rank within that site.
    """
    site_members: dict[_SlotColor, list[str]] = defaultdict(list)
    for label in workspace.raw_labels:
        node = workspace.nodes[label]
        if node.uses_params and node.param_barcodes:
            site_members[("param", _param_call_identity(node))].append(label)
        elif node.recurrence_anchored:
            site_members[("anchor", (node.equivalence_key, node.output_slot))].append(label)
    return {
        label: (color, pass_index)
        for color, members in site_members.items()
        for pass_index, label in enumerate(members, start=1)
    }


def _pf_consumer_site_frame(
    workspace: _GroupingWorkspace,
    label: str,
    direct_site_passes: dict[str, tuple[_SlotColor, int]],
) -> dict[_SlotColor, tuple[int, ...]]:
    """Return direct parameterized/anchored consumer sites with pass ranks.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    label:
        Bare-op label whose consumer flank should be described.
    direct_site_passes:
        Direct topology anchors from :func:`_pf_direct_site_passes`.

    Returns
    -------
    dict[_SlotColor, tuple[int, ...]]
        Consumer-site colors mapped to their sorted pass ranks. Bare and pseudo
        consumers are omitted because only persistent direct sites provide an
        independent boundary witness.
    """
    ranks: dict[_SlotColor, list[int]] = defaultdict(list)
    for child in workspace.nodes[label].data_children:
        site_pass = direct_site_passes.get(child)
        if site_pass is None:
            continue
        color, pass_index = site_pass
        ranks[color].append(pass_index)
    return {color: tuple(sorted(pass_indices)) for color, pass_indices in ranks.items()}


def _pf_consumer_sites_advance(
    earlier: dict[_SlotColor, tuple[int, ...]],
    later: dict[_SlotColor, tuple[int, ...]],
) -> bool:
    """Return whether equal direct consumer sites advance pass-wise.

    Parameters
    ----------
    earlier:
        Consumer-site frame of the earlier bare call.
    later:
        Consumer-site frame of the later bare call.

    Returns
    -------
    bool
        ``True`` when both calls feed the same non-empty direct consumer-site
        multiset, no consumer pass regresses, and at least one pass advances.
    """
    if not earlier or earlier.keys() != later.keys():
        return False
    if any(len(earlier[color]) != len(later[color]) for color in earlier):
        return False
    paired_ranks = (
        (earlier_rank, later_rank)
        for color in earlier
        # The per-color length guard above already returned False on any
        # mismatch, so every color's rank lists pair exactly.
        for earlier_rank, later_rank in zip(earlier[color], later[color], strict=True)
    )
    comparisons = list(paired_ranks)
    return all(later_rank >= earlier_rank for earlier_rank, later_rank in comparisons) and any(
        later_rank > earlier_rank for earlier_rank, later_rank in comparisons
    )


def _pf_child_route_allows(
    workspace: _GroupingWorkspace,
    node1_label: str,
    node2_label: str,
    class_of: dict[str, str],
    reach_memo: _ReachabilityCache,
) -> bool:
    """Return whether two loop-invariant-fed calls sequence through their consumers.

    A per-iteration factory op (``torch.ones(...)`` in the body) or a recompute of
    a loop-invariant value (``torch.tanh(x)`` on the unchanging input) has no
    parent-side sequencing evidence: its parents are external or absent, and no
    data path connects one repetition to the next. Its repetition is still real
    -- one consumer per iteration -- so sequencing is certified on the consumer
    side instead: the two calls feed DISJOINT consumers of IDENTICAL site colors,
    and the earlier call's consumer flows into the later call's consumer.
    Requiring disjointness rejects sibling inits (the ``h``/``c`` zeros of one
    LSTMCell feed the SAME call and are two distinct values, not two passes).

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    node1_label:
        Earlier candidate call.
    node2_label:
        Later candidate call.
    class_of:
        Current bare-op class assignment.
    reach_memo:
        Shared reachability cache.

    Returns
    -------
    bool
        ``True`` when consumer topology certifies consecutive repetition.
    """
    children1 = [
        child for child in workspace.nodes[node1_label].data_children if child in workspace.nodes
    ]
    children2 = [
        child for child in workspace.nodes[node2_label].data_children if child in workspace.nodes
    ]
    if not children1 or not children2:
        return False
    if set(children1) & set(children2):
        return False
    colors1 = Counter(_pf_slot_color(workspace, child, class_of) for child in children1)
    colors2 = Counter(_pf_slot_color(workspace, child, class_of) for child in children2)
    if colors1 != colors2:
        return False
    return any(
        _reaches_forward(workspace, child1, child2, reach_memo)
        for child1 in children1
        for child2 in children2
    )


def _pf_entry_union_allowed(
    workspace: _GroupingWorkspace,
    entry_label: str,
    target_label: str,
    signatures: dict[str, Counter],
    parent_colors: dict[str, list[tuple[str, _SlotColor]]],
    consumer_site_frames: dict[str, dict[_SlotColor, tuple[int, ...]]],
    complete_consumer_sites: set[_SlotColor],
    cohort_sizes: dict[frozenset, int],
    realizations: dict[_SlotColor, int],
    reach_memo: _ReachabilityCache,
) -> bool:
    """Return whether a loop-entry call may join a later same-key call's site.

    On pass 1 a recurrent bare op reads state produced OUTSIDE the loop; every
    later pass reads it through the feedback wire. The two signatures therefore
    differ in exactly ONE slot -- the carry slot -- and agree everywhere else.
    Admission demands, each condition independently load-bearing:

    * **Single-slot difference.** Signatures agreeing on all but one matched
      slot. Two differing slots mean a differing FLANK as well -- positive
      evidence of a different site (the n1=1 peeled chain), never an entry.
    * **One-shot odd parent or consumer continuation.** A recurring entry-side
      odd color normally marks a neighboring site, not an entry. The exception
      is independent consumer-side proof that both calls feed advancing passes
      of the same direct parameterized/anchored site while retaining a shared
      direct flank. The flank requirement keeps unary post-operations attached
      to the site that produced them.
    * **Carry certificate.** The target's odd parent is computed FROM the entry
      (reflexively): the differing slot really is the loop feedback, not an
      unrelated topology change.
    * **Recurrence evidence.** Either a parameterized/anchored flank survives in
      the agreeing remainder (the loop beacon whose passes the pair rides), or
      -- for flankless (unary-style) entries -- the target's site is realized:
      its equal-signature cohort has two or more members, or its odd parent is
      a parameterized site realized at least twice. A bare two-op chain
      (``tanh(tanh(x))``) has neither and stays split.

    The entry-sweep prefilter in :func:`_pf_partition_class` mirrors the three
    terminal recurrence-evidence arms above with per-target necessary
    conditions; changing any arm here requires updating that prefilter (and
    the oracle battery in ``tests/test_loop_grouping_adapter.py``) in
    lockstep, or admissible pairs may be silently skipped.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    entry_label:
        Earlier call whose context has not yet saturated.
    target_label:
        Later same-key call.
    signatures:
        Current signature (parent color multiset) per candidate label.
    parent_colors:
        Per-label ``(parent_label, color)`` pairs backing the signatures.
    consumer_site_frames:
        Direct parameterized/anchored consumer sites and pass ranks per label.
    complete_consumer_sites:
        Direct consumer sites whose every pass is fed by this same-key universe.
    cohort_sizes:
        Class-local member count per signature.
    realizations:
        Parameterized/anchored color realization counts.
    reach_memo:
        Shared reachability cache.

    Returns
    -------
    bool
        ``True`` when every admission condition holds.
    """
    signature_entry = signatures[entry_label]
    signature_target = signatures[target_label]
    odd_entry = signature_entry - signature_target
    odd_target = signature_target - signature_entry
    if sum(odd_entry.values()) != 1 or sum(odd_target.values()) != 1:
        return False
    entry_odd_color = next(iter(odd_entry))
    target_odd_color = next(iter(odd_target))
    agreeing_remainder = signature_entry & signature_target
    consumer_continuation = (
        any(color[0] in ("param", "anchor") for color in agreeing_remainder)
        and set(consumer_site_frames[entry_label]) <= complete_consumer_sites
        and set(consumer_site_frames[target_label]) <= complete_consumer_sites
        and _pf_consumer_sites_advance(
            consumer_site_frames[entry_label], consumer_site_frames[target_label]
        )
    )
    entry_site_is_recurring = entry_odd_color[0] in ("param", "anchor") and (
        realizations.get(entry_odd_color, 0) != 1
    )
    if entry_site_is_recurring and not consumer_continuation:
        return False
    carry_certified = any(
        parent_label in workspace.nodes
        and _reaches_forward(workspace, entry_label, parent_label, reach_memo)
        for parent_label, color in parent_colors[target_label]
        if color == target_odd_color
    )
    if not carry_certified:
        return False
    if any(color[0] in ("param", "anchor") for color in agreeing_remainder):
        return True
    if cohort_sizes.get(frozenset(signature_target.items()), 0) >= 2:
        return True
    return target_odd_color[0] in ("param", "anchor") and realizations.get(target_odd_color, 0) >= 2


def _pf_partition_class(
    workspace: _GroupingWorkspace,
    members: list[str],
    signatures: dict[str, Counter],
    parent_colors: dict[str, list[tuple[str, _SlotColor]]],
    consumer_site_frames: dict[str, dict[_SlotColor, tuple[int, ...]]],
    complete_consumer_sites: set[_SlotColor],
    class_of: dict[str, str],
    realizations: dict[_SlotColor, int],
    reach_memo: _ReachabilityCache,
) -> list[list[str]]:
    """Partition one same-key candidate class into topological site groups.

    Members with EQUAL signatures union when data flow connects them (loop
    iterations always connect through the carry; parallel streams never do) or,
    for loop-invariant-fed repeats, when consumer topology certifies sequencing.
    Entry calls that are not already members of a connected equal-signature
    cohort union with their EARLIEST reachable admissible target only. This is
    stream-local rather than a global signature census: disconnected identical
    entries can each pair with their own continuation, while a realized cohort
    cannot bridge a later site. Adoption edges with out-degree one form an
    in-forest, so one entry can never bridge two mutually split sites.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.
    members:
        Class member labels sorted by raw capture order.
    signatures:
        Current signature per candidate label.
    parent_colors:
        Per-label ``(parent_label, color)`` pairs backing the signatures.
    consumer_site_frames:
        Direct parameterized/anchored consumer sites and pass ranks per label.
    complete_consumer_sites:
        Direct consumer sites whose every pass is fed by this same-key universe.
    class_of:
        Current bare-op class assignment.
    realizations:
        Parameterized/anchored color realization counts.
    reach_memo:
        Shared reachability cache.

    Returns
    -------
    list[list[str]]
        New member groups, each sorted by raw capture order.
    """
    parent_map = {member: member for member in members}

    def find(label: str) -> str:
        """Return the union-find root for a member label."""
        while parent_map[label] != label:
            parent_map[label] = parent_map[parent_map[label]]
            label = parent_map[label]
        return label

    def union(label1: str, label2: str) -> None:
        """Merge the union-find sets of two member labels."""
        root1, root2 = find(label1), find(label2)
        if root1 != root2:
            parent_map[root2] = root1

    member_signature_keys = [frozenset(signatures[member].items()) for member in members]
    cohorts: dict[frozenset, list[str]] = defaultdict(list)
    # member_signature_keys is built by comprehension over members, so the two
    # pair exactly by construction.
    for member, signature_key in zip(members, member_signature_keys, strict=True):
        cohorts[signature_key].append(member)
    cohort_sizes = {signature: len(cohort) for signature, cohort in cohorts.items()}

    for signature, cohort in cohorts.items():
        if len(cohort) < 2:
            continue
        invariant_fed = all(color[0] == "ext" for color, _ in signature)
        # Live distinct-root counter: every union below merges two roots that both
        # belong to this cohort, so it lowers the cohort's distinct-root count by
        # exactly one. Once a single root remains, every remaining pair can only
        # hit the root-equality short-circuit (whose sole effect is idempotent
        # path compression), so breaking out is output-identical while avoiding
        # the O(len(cohort)^2) tail of already-unified pairs.
        distinct_roots = len({find(member) for member in cohort})
        # strict=False is deliberate: zip(seq, seq[1:]) is the consecutive-pairs
        # sliding window, which is ragged by construction (n-1 pairs from n
        # items). strict=True here would raise on every non-empty cohort.
        pair_iter = it.chain(zip(cohort, cohort[1:], strict=False), it.combinations(cohort, 2))
        pair_probes = 0
        for member1, member2 in pair_iter:
            if find(member1) == find(member2):
                continue
            pair_probes += 1
            if pair_probes > _PF_COHORT_PAIR_PROBE_CEILING:
                # Bounded sweep (r8 R60-4): see the ceiling's rationale. The
                # consecutive-pair window has already run in full, so genuine
                # loop chains are unified; what remains is the exhaustive
                # cross-check over (overwhelmingly parallel) siblings.
                warnings.warn(
                    "TorchLens recurrence grouping hit the pair-probe ceiling "
                    f"({_PF_COHORT_PAIR_PROBE_CEILING}) on an equal-signature "
                    f"cohort of {len(cohort)} ops; non-consecutive recurrence "
                    "links past the ceiling are not unified and those ops stay "
                    "ungrouped (structure and payloads are unaffected).",
                    TorchLensWarning,
                    stacklevel=2,
                )
                break
            consumers1 = consumer_site_frames[member1]
            consumers2 = consumer_site_frames[member2]
            if (
                sum(count for _, count in signature) >= 2
                and consumers1
                and consumers2
                and set(consumers1) <= complete_consumer_sites
                and set(consumers2) <= complete_consumer_sites
                and consumers1.keys() != consumers2.keys()
            ):
                continue
            if (
                _reaches_forward(workspace, member1, member2, reach_memo)
                or invariant_fed
                and _pf_child_route_allows(workspace, member1, member2, class_of, reach_memo)
            ):
                union(member1, member2)
                distinct_roots -= 1
                if distinct_roots == 1:
                    break

    equal_component_sizes = Counter(find(member) for member in members)
    realized_equal_members = {
        member for member in members if equal_component_sizes[find(member)] > 1
    }

    # Entry-admission prefilter. :func:`_pf_entry_union_allowed` returns ``True``
    # only through one of three terminal recurrence-evidence arms, and each arm
    # carries a necessary condition on the pair that is checkable from per-member
    # indexes without touching the pair:
    #
    # * a surviving param/anchor flank in the agreeing remainder needs entry and
    #   target to SHARE a param/anchor color;
    # * a realized target cohort needs the target's equal-signature cohort to be
    #   non-singleton;
    # * a realized target odd parent needs the target to carry a param/anchor
    #   color with two or more realizations.
    #
    # A target failing all three conditions can never be admitted for any entry,
    # so walking only qualifying targets is admission-identical while emptying
    # the pair triangle in the degenerate long-loop regime, where every
    # signature is a distinct singleton cohort of bare colors. Index lists are
    # ascending by construction, so per-entry suffixes come from one bisect.
    member_param_anchor_colors = [
        [color for color in signatures[member] if color[0] in ("param", "anchor")]
        for member in members
    ]
    shared_color_positions: dict[_SlotColor, list[int]] = defaultdict(list)
    target_side_positions: list[int] = []
    for position, signature_key in enumerate(member_signature_keys):
        param_anchor_colors = member_param_anchor_colors[position]
        for color in param_anchor_colors:
            shared_color_positions[color].append(position)
        if cohort_sizes[signature_key] >= 2 or any(
            realizations.get(color, 0) >= 2 for color in param_anchor_colors
        ):
            target_side_positions.append(position)

    for index, entry in enumerate(members):
        if entry in realized_equal_members:
            continue
        entry_signature = member_signature_keys[index]
        candidate_positions = set(
            target_side_positions[bisect_right(target_side_positions, index) :]
        )
        for color in member_param_anchor_colors[index]:
            positions = shared_color_positions[color]
            candidate_positions.update(positions[bisect_right(positions, index) :])
        for position in sorted(candidate_positions):
            target = members[position]
            if member_signature_keys[position] == entry_signature:
                continue
            if find(entry) == find(target):
                break
            if _pf_entry_union_allowed(
                workspace,
                entry,
                target,
                signatures,
                parent_colors,
                consumer_site_frames,
                complete_consumer_sites,
                cohort_sizes,
                realizations,
                reach_memo,
            ):
                # The unfiltered sweep breaks at the FIRST later
                # differing-signature member whose root already equals the
                # entry's -- even a non-candidate -- abandoning any later
                # admissible target. (Same-signature members are skipped
                # without a root check, so they never block.) Scan the skipped
                # gap once before admitting; this runs only on admissions, so
                # the degenerate regime never pays it.
                gap_blocked = any(
                    member_signature_keys[skipped] != entry_signature
                    and find(members[skipped]) == find(entry)
                    for skipped in range(index + 1, position)
                )
                if not gap_blocked:
                    union(entry, target)
                break

    groups: dict[str, list[str]] = defaultdict(list)
    for member in members:
        groups[find(member)].append(member)
    return [
        sorted(group, key=lambda label: workspace.nodes[label].raw_order)
        for group in groups.values()
    ]


def _assign_param_free_layers(workspace: _GroupingWorkspace) -> None:
    """Assign every bare param-free op's layer from its parents' solved grouping.

    Runs after all isomorphic expansion rounds, when parameterized and anchored
    grouping is final. Candidate classes start maximally coarse (one class per
    equivalence key and output slot over all bare non-pseudo ops) and are
    refined to a fixpoint: each round snapshots colors, partitions every
    multi-member class (:func:`_pf_partition_class`), and applies the splits;
    signatures reference classes, so a split propagates to consumers on the
    next round and sibling ops in one loop body settle in lockstep. Refinement
    only splits, so the result is the coarsest partition consistent with the
    evidence and iteration terminates.

    After the partition stabilizes, a self-evidence gate dissolves classes whose
    steady members (all but the earliest call) are fed exclusively by the class
    itself: a bare single-op self-chain (``for: h = tanh(h)`` and the straight
    ``tanh(tanh(tanh(x)))`` chain alike) carries no recurrence evidence beyond
    its own repetition and stays single-pass -- the historical minimum-body-size
    policy, restated topologically. Dissolution changes consumers' colors, so
    the fixpoint reruns until no class dissolves.

    Parameters
    ----------
    workspace:
        Mutable grouping workspace.

    Returns
    -------
    None
        Mutates each bare op's ``layer_label``.
    """
    anchor_ancestry = _topology_anchor_ancestry(workspace)
    universe: dict[tuple[str, int | None], list[str]] = OrderedDict()
    for label in workspace.raw_labels:
        node = workspace.nodes[label]
        if node.uses_params and node.param_barcodes:
            continue
        if node.recurrence_anchored:
            continue
        if node.func_name == _PSEUDO_FUNC_NAME:
            continue
        if not anchor_ancestry[label]:
            # Pure parameter-free graphs retain the historical isomorphic-
            # subgraph grouping path. Their topology has no persistent site
            # colors for the direct-site fixpoint to propagate, and forcing
            # them through it fragments nested motifs or fuses sequential ones.
            continue
        universe.setdefault((node.equivalence_key, node.output_slot), []).append(label)

    universe_labels = {label for labels in universe.values() for label in labels}
    realizations = _pf_realization_counts(workspace)
    direct_site_passes = _pf_direct_site_passes(workspace)
    consumer_site_frames = {
        label: _pf_consumer_site_frame(workspace, label, direct_site_passes)
        for labels in universe.values()
        for label in labels
    }
    complete_consumer_sites_by_key: dict[tuple[str, int | None], set[_SlotColor]] = defaultdict(set)
    for key, labels in universe.items():
        observed_passes: dict[_SlotColor, set[int]] = defaultdict(set)
        for label in labels:
            for color, pass_indices in consumer_site_frames[label].items():
                observed_passes[color].update(pass_indices)
        complete_consumer_sites_by_key[key] = {
            color
            for color, pass_indices in observed_passes.items()
            if pass_indices == set(range(1, realizations[color] + 1))
        }
    reach_memo = _ReachabilityCache(workspace)

    classes: OrderedDict[str, list[str]] = OrderedDict()
    class_key: dict[str, tuple[str, int | None]] = {}
    fixed_class_of = {
        label: workspace.nodes[label].layer_label
        for label in workspace.raw_labels
        if label not in universe_labels
        and not (workspace.nodes[label].uses_params and workspace.nodes[label].param_barcodes)
        and not workspace.nodes[label].recurrence_anchored
        and workspace.nodes[label].func_name != _PSEUDO_FUNC_NAME
    }
    class_of: dict[str, str] = dict(fixed_class_of)
    for key, labels in universe.items():
        leader = labels[0]
        classes[leader] = list(labels)
        class_key[leader] = key
        for label in labels:
            class_of[label] = leader

    fixpoint_work = 0
    while True:
        if fixpoint_work > _PF_FIXPOINT_WORK_BUDGET:
            # Budget exhausted (r8 R60-5): dissolve every remaining
            # multi-member class -- the partition may still be too coarse,
            # and a coarse class would CLAIM recurrence the evidence has not
            # settled, so the honest degradation is no claim at all.
            warnings.warn(
                "TorchLens recurrence grouping exhausted its refinement "
                f"budget ({_PF_FIXPOINT_WORK_BUDGET} member visits) before "
                "converging; the affected parameter-free ops stay ungrouped "
                "(structure and payloads are unaffected).",
                TorchLensWarning,
                stacklevel=2,
            )
            for leader, members in list(classes.items()):
                if len(members) < 2:
                    continue
                del classes[leader]
                key = class_key.pop(leader)
                for member in members:
                    classes[member] = [member]
                    class_key[member] = key
                    class_of[member] = member
            break
        signatures: dict[str, Counter] = {}
        parent_colors: dict[str, list[tuple[str, _SlotColor]]] = {}
        for leader, members in classes.items():
            fixpoint_work += len(members)
            for member in members:
                pairs = [
                    (parent, _pf_slot_color(workspace, parent, class_of))
                    for parent in workspace.nodes[member].data_parents
                ]
                parent_colors[member] = pairs
                signature = Counter(color for _, color in pairs)
                signatures[member] = signature

        changed = False
        new_classes: OrderedDict[str, list[str]] = OrderedDict()
        new_class_key: dict[str, tuple[str, int | None]] = {}
        for leader, members in classes.items():
            if len(members) < 2:
                new_classes[leader] = members
                new_class_key[leader] = class_key[leader]
                continue
            groups = _pf_partition_class(
                workspace,
                members,
                signatures,
                parent_colors,
                consumer_site_frames,
                complete_consumer_sites_by_key[class_key[leader]],
                class_of,
                realizations,
                reach_memo,
            )
            if len(groups) > 1:
                changed = True
            for group in sorted(groups, key=lambda group_: workspace.nodes[group_[0]].raw_order):
                new_leader = group[0]
                new_classes[new_leader] = group
                new_class_key[new_leader] = class_key[leader]

        if changed:
            classes = new_classes
            class_key = new_class_key
            class_of = dict(fixed_class_of)
            class_of.update(
                {label: leader for leader, members in classes.items() for label in members}
            )
            continue

        dissolved = False
        for leader, members in list(classes.items()):
            if len(members) < 2:
                continue
            steady_colors: set[_SlotColor] = set()
            for member in members[1:]:
                steady_colors.update(signatures[member])
            if steady_colors and steady_colors <= {("class", leader)}:
                del classes[leader]
                key = class_key.pop(leader)
                for member in members:
                    classes[member] = [member]
                    class_key[member] = key
                    class_of[member] = member
                dissolved = True
        if not dissolved:
            break

    for leader, members in classes.items():
        for member in members:
            workspace.nodes[member].layer_label = leader if len(members) > 1 else member


def _merge_iso_groups_to_layers(
    workspace: _GroupingWorkspace,
    iso_node_groups: dict[str, list[str]],
    node_to_subgraph: dict[str, SubgraphInfo],
    adjacent_subgraphs: dict[str, set[str]],
) -> dict[str, set[str]]:
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
    uf_parent: dict[str, str] = {}

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

    reach_memo = _ReachabilityCache(workspace)
    anchor_ancestry = _topology_anchor_ancestry(workspace)

    for _iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(
            iso_nodes_orig, key=lambda node_label: workspace.nodes[node_label].raw_order
        )
        # Consecutive pairs first: in a genuine loop they carry the unions, so the
        # full pairwise sweep afterwards short-circuits on shared union-find roots
        # instead of re-deriving (and re-checking reachability for) distant pairs.
        # Live distinct-root counter: every union below merges two roots that both
        # belong to this iso group, lowering the group's distinct-root count by
        # exactly one. Once a single root remains, every remaining pair can only
        # hit the root-equality short-circuit (whose sole effect is idempotent
        # path compression), so breaking out is output-identical while avoiding
        # the O(len(iso_nodes)^2) tail of already-unified pairs. Groups arriving
        # fully unified (from earlier groups' unions) skip the sweep outright.
        distinct_roots = len({find(node_label) for node_label in iso_nodes})
        if distinct_roots == 1:
            continue
        combination_nodes = iso_nodes
        if all(
            not workspace.nodes[node_label].uses_params
            and not workspace.nodes[node_label].recurrence_anchored
            for node_label in iso_nodes
        ):
            # Every pair in an all-bare group deterministically falls into the
            # bare branch below, whose ONLY union arm additionally requires BOTH
            # endpoints to be free of anchored ancestry. A pair with an
            # ancestry-carrying endpoint is therefore a guaranteed no-op
            # ``continue``, so restricting the combinations triangle to the
            # ancestry-free members removes only provably inert pairs: the union
            # sequence -- and with it the final partition and its min-label
            # roots -- is identical, while a mixed-ancestry group's O(N^2) tail
            # collapses to its handful of eligible members. With fewer than two
            # ancestry-free members no pair anywhere in the group (consecutive
            # or combination) can union, so the whole sweep is skipped; those
            # anchored-descendant bare ops belong to the two-sided param-free
            # fixpoint, which assigns them after all parameterized/anchored
            # groups are final.
            combination_nodes = [
                node_label for node_label in iso_nodes if not anchor_ancestry[node_label]
            ]
            if len(combination_nodes) < 2:
                continue
        # Bucketed candidate sweep instead of the historical full
        # ``it.combinations(combination_nodes, 2)`` triangle. Every union arm
        # below preconditions on subgraph adjacency (bare and anchored arms)
        # or on the two subgraphs sharing parametric body content (the
        # weight-tied arm, which alone may also union through ``_seed_reaches``
        # without adjacency). A pair outside those buckets is a guaranteed
        # ``continue`` in every branch, and the union order cannot change the
        # partition (min-root union; the root-equality short-circuit is
        # idempotent), so enumerating only the bucketed superset is
        # output-identical. The full triangle made torch capture O(N^2) in
        # the largest structurally-identical op group -- a plain param-free
        # feed-forward chain walked all C(N,2) pairs (35% of capture CPU at
        # 3200 ops) and gained zero grouping for it, because neither
        # short-circuit can fire when no unions ever happen.
        order_index = {node_label: index for index, node_label in enumerate(iso_nodes)}
        nodes_by_subgraph: dict[str, list[str]] = defaultdict(list)
        for node_label in combination_nodes:
            nodes_by_subgraph[node_to_subgraph[node_label].starting_node].append(node_label)
        candidate_sg_pairs: set[tuple[str, str]] = set()
        for sg1 in nodes_by_subgraph:
            self_adjacent = sg1 in adjacent_subgraphs and sg1 in adjacent_subgraphs[sg1]
            if len(nodes_by_subgraph[sg1]) > 1 and (self_adjacent or sg_param_types.get(sg1)):
                candidate_sg_pairs.add((sg1, sg1))
            for sg2 in adjacent_subgraphs.get(sg1, ()):
                if sg2 != sg1 and sg2 in nodes_by_subgraph:
                    candidate_sg_pairs.add((min(sg1, sg2), max(sg1, sg2)))
        subgraphs_by_param_type: dict[str, list[str]] = defaultdict(list)
        for sg1 in nodes_by_subgraph:
            for param_type in sg_param_types.get(sg1, frozenset()):
                subgraphs_by_param_type[param_type].append(sg1)

        def _weight_tied_sg_pairs(
            subgraphs_by_param_type: dict[str, list[str]] = subgraphs_by_param_type,
            candidate_sg_pairs: set[tuple[str, str]] = candidate_sg_pairs,
        ) -> Iterator[tuple[str, str]]:
            """Yield the shared-param-type subgraph triangle LAZILY (r8 R29).

            The eager ``candidate_sg_pairs.update(combinations(...))`` build
            materialized O(G^2) tuples per tied-weight iso group (~2M pairs
            at G=2048) BEFORE the consumer's ``distinct_roots == 1`` break
            could prune anything; generating them behind the same generator
            keeps the transient cost proportional to pairs actually
            consumed. The seen set grows only with consumption, and dedup
            against the adjacency-derived eager pairs preserves the exact
            historical pair universe (order may differ, which is sound: the
            min-root union makes the final partition order-independent, and
            this order is still deterministic).
            """

            emitted: set[tuple[str, str]] = set()
            for shared_subgraphs in subgraphs_by_param_type.values():
                for pair in it.combinations(sorted(shared_subgraphs), 2):
                    if pair in candidate_sg_pairs or pair in emitted:
                        continue
                    emitted.add(pair)
                    yield pair

        def _bucketed_candidate_pairs(
            candidate_sg_pairs: set[tuple[str, str]] = candidate_sg_pairs,
            nodes_by_subgraph: dict[str, list[str]] = nodes_by_subgraph,
            order_index: dict[str, int] = order_index,
        ) -> Iterator[tuple[str, str]]:
            """Yield candidate pairs oriented by capture order (earlier first)."""

            for sg1, sg2 in it.chain(sorted(candidate_sg_pairs), _weight_tied_sg_pairs()):
                if sg1 == sg2:
                    members = sorted(nodes_by_subgraph[sg1], key=order_index.__getitem__)
                    yield from it.combinations(members, 2)
                    continue
                for node1_label in nodes_by_subgraph[sg1]:
                    for node2_label in nodes_by_subgraph[sg2]:
                        if order_index[node1_label] < order_index[node2_label]:
                            yield node1_label, node2_label
                        else:
                            yield node2_label, node1_label

        # strict=False is deliberate: the consecutive-pairs sliding window is
        # ragged by construction (see the cohort sweep above).
        # Relation-driven candidates replace the C(N,2) triangle (r5 b4-opus
        # F29-A, measured: the full triangle walked on a plain feed-forward
        # chain of N identical bare ops -- 35% of capture CPU at 3200 ops,
        # rising as N^2, for ZERO produced groupings). Every union arm below
        # requires the pair's subgraphs to be ADJACENT or (the parameterized
        # arm only) to share at least one param type, so pairs outside those
        # two relations are provably inert ``continue``s: enumerating only
        # related pairs preserves the union transitive closure -- and with it
        # the final partition and its min-label roots -- exactly.
        pair_iter = it.chain(
            zip(iso_nodes, iso_nodes[1:], strict=False),
            _bucketed_candidate_pairs(),
        )
        for node1_label, node2_label in pair_iter:
            if find(node1_label) == find(node2_label):
                continue
            node1_subgraph_label = node_to_subgraph[node1_label].starting_node
            node2_subgraph_label = node_to_subgraph[node2_label].starting_node
            subgraphs_are_adjacent = (
                node1_subgraph_label in adjacent_subgraphs
                and node2_subgraph_label in adjacent_subgraphs[node1_subgraph_label]
            )
            node1 = workspace.nodes[node1_label]
            node2 = workspace.nodes[node2_label]
            if (
                node1.uses_params
                and node2.uses_params
                and _param_call_identity(node1) != _param_call_identity(node2)
            ):
                # Two parameterized ops that differ in module address, output slot,
                # or non-tensor call structure are different semantic sites; no
                # adjacency or shared-parameter evidence may unite them as passes
                # of one recurrent layer. Identity equality is transitive, so
                # allowed unions can never chain around this veto.
                continue
            pair_anchored = node1.recurrence_anchored or node2.recurrence_anchored
            if not (node1.uses_params or node2.uses_params or pair_anchored):
                if (
                    not anchor_ancestry[node1_label]
                    and not anchor_ancestry[node2_label]
                    and subgraphs_are_adjacent
                    and _param_free_adjacency_merge_allowed(
                        workspace,
                        node1_label,
                        node2_label,
                        node_to_subgraph[node1_label],
                        node_to_subgraph[node2_label],
                    )
                ):
                    # With no persistent topology anchor anywhere upstream,
                    # retain the historical whole-body isomorphism rule. This
                    # is the only evidence available for pure nested/chained
                    # motifs and is insulated from parameterized boundaries.
                    union(node1_label, node2_label)
                    distinct_roots -= 1
                    if distinct_roots == 1:
                        break
                # Anchored-descendant bare ops are never decided by local
                # subgraph adjacency; the two-sided fixpoint assigns them after
                # all parameterized/anchored groups are final.
                continue
            overlapping_param_types = (
                sg_param_types[node1_subgraph_label] & sg_param_types[node2_subgraph_label]
            )
            if overlapping_param_types:
                # Shared parametric body content marks two iterations of one
                # weight-tied loop -- but only when the iterations actually CHAIN:
                # directly adjacent, or one seed feeds the other through the graph
                # (interleaved bodies, e.g. tied-linear/mul/tied-linear/log). A
                # shared weight captured inside two structurally-disjoint parallel
                # branches is NOT recurrence between the branches' ops; without the
                # connectivity requirement, independent terminal reductions on
                # parallel streams were grouped as spurious recurrent passes.
                if subgraphs_are_adjacent or _seed_reaches(
                    workspace, node1_label, node2_label, reach_memo
                ):
                    union(node1_label, node2_label)
                    distinct_roots -= 1
                    if distinct_roots == 1:
                        break
            elif subgraphs_are_adjacent and pair_anchored:
                # A reused persistent identity -- a submodule call repeated across
                # iterations (one ``nn.ReLU`` invoked four times) or a stateful
                # buffer rewritten each pass -- is real recurrence even with a
                # single-op body. Bare param-free pairs never reach this branch
                # (they are skipped above for the topological pass).
                union(node1_label, node2_label)
                distinct_roots -= 1
                if distinct_roots == 1:
                    break

    param_barcode_groups: dict[_ParamCallIdentity, list[str]] = defaultdict(list)
    for node_label in all_iso_nodes:
        node = workspace.nodes[node_label]
        if node.uses_params and node.param_barcodes:
            # The FULL call identity is the key, not bare parameter identity. The
            # output slot is part of it: co-outputs of one call (h and c of an
            # LSTMCell) share function name and parameters but are DISTINCT layers,
            # not sequential passes of each other (omitting the slot doubled
            # ``num_passes`` for every multi-output recurrent cell). The module
            # address and non-tensor arg signature are equally part of it: tied
            # distinct modules and same-kernel/different-padding convolutions were
            # unconditionally unioned here into false recurrent layers.
            param_barcode_groups[_param_call_identity(node)].append(node_label)

    for _identity_key, nodes_with_same_params in param_barcode_groups.items():
        if len(nodes_with_same_params) > 1:
            first = nodes_with_same_params[0]
            for other in nodes_with_same_params[1:]:
                union(first, other)

    merged_layer_groups: dict[str, set[str]] = defaultdict(set)
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
