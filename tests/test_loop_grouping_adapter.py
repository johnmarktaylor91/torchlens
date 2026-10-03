"""Tests for backend-neutral loop grouping adapter."""

import cProfile
import itertools as it
import pstats
import random
from collections import Counter, defaultdict
from typing import Any

import example_models
import pytest
import torch

import torchlens.postprocess.loop_grouping_adapter as lga
from torchlens import trace as trace_fn
from torchlens.postprocess.loop_grouping_adapter import (
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)


def _raw_label(trace: Any, final_label: str) -> str:
    """Return the raw label corresponding to a final trace label.

    Parameters
    ----------
    trace:
        TorchLens trace with raw/final label maps.
    final_label:
        Final layer or op label.

    Returns
    -------
    str
        Raw label for ``final_label``.
    """
    return trace._final_to_raw_layer_labels[final_label]


def _raw_recurrent_member_sets(trace: Any) -> set[frozenset[str]]:
    """Collect expected recurrent member sets from a torch recurrent fixture.

    Parameters
    ----------
    trace:
        TorchLens trace for a recurrent torch model.

    Returns
    -------
    set[frozenset[str]]
        Raw-label recurrent groups with at least two members.
    """
    member_sets: set[frozenset[str]] = set()
    for op in trace.ops:
        if len(op.recurrent_ops) <= 1:
            continue
        raw_members = frozenset(_raw_label(trace, label) for label in op.recurrent_ops)
        member_sets.add(raw_members)
    return member_sets


def _neutral_graph_from_torch_recurrent_fixture(trace: Any) -> RecurrenceGroupingGraph:
    """Build a label-consistent neutral graph from a finalized torch trace.

    Parameters
    ----------
    trace:
        TorchLens trace for a recurrent torch model.

    Returns
    -------
    RecurrenceGroupingGraph
        Neutral graph using raw labels consistently for nodes and data edges.
    """
    nodes: dict[str, RecurrenceNode] = {}
    raw_labels: list[str] = []
    raw_label_set = {op._label_raw for op in trace.ops}
    # ``equivalent_ops`` holds finalized OP labels (pass-qualified), which are keyed
    # differently from the final LAYER labels in ``_final_to_raw_layer_labels`` --
    # map them back to raw labels through the ops themselves.
    op_label_to_raw = {op.label: op._label_raw for op in trace.ops}

    for op in trace.ops:
        raw_label = op._label_raw
        raw_labels.append(raw_label)
        nodes[raw_label] = RecurrenceNode(
            label=raw_label,
            raw_order=op.raw_index,
            equivalence_key=op.equivalence_class,
            equivalent_labels=tuple(
                op_label_to_raw[equiv_label]
                for equiv_label in op.equivalent_ops
                if equiv_label in op_label_to_raw
            ),
            data_parents=tuple(
                _raw_label(trace, parent)
                for parent in op.parents
                if parent in trace.layer_dict_all_keys
            ),
            data_children=tuple(
                _raw_label(trace, child)
                for child in op.children
                if child in trace.layer_dict_all_keys
            ),
            layer_label=raw_label,
            recurrent_labels=(),
            uses_params=bool(op.uses_params),
            func_name=op.func_name,
            param_barcodes=tuple(op._param_barcodes),
            retain=raw_label in raw_label_set,
            pruned=False,
        )

    return RecurrenceGroupingGraph(
        nodes=nodes,
        raw_labels=tuple(raw_labels),
        source_labels=tuple(_raw_label(trace, label) for label in trace.input_layers),
        eligible_labels=tuple(raw_labels),
    )


def test_neutral_loop_grouping_matches_torch_recurrent_fixture() -> None:
    """Neutral grouping service reproduces torch recurrent member sets."""
    torch.manual_seed(0)
    traced = trace_fn(example_models.RecurrentParamsSimple(), torch.rand(5, 5))

    graph = _neutral_graph_from_torch_recurrent_fixture(traced)
    assignments = group_recurrent_nodes(graph)
    actual_groups = {
        frozenset(assignment.recurrent_labels)
        for assignment in assignments.values()
        if assignment.num_passes > 1
    }

    assert actual_groups == _raw_recurrent_member_sets(traced)
    assert all("control" not in node.data_parents for node in graph.nodes.values())


class _ChainedTiedRecurrentNet(torch.nn.Module):
    """Tied linear + tanh applied ``num_steps`` times: one big chained loop."""

    def __init__(self, num_steps: int, dim: int = 16) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.tied = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for _ in range(self.num_steps):
            hidden = torch.tanh(self.tied(hidden))
        return hidden


class _ParallelStreamsNet(torch.nn.Module):
    """Disjoint parallel repeats: multi-root groups that must stay split.

    Two disconnected param-free ``relu(x) + 1`` chains form one candidate
    class that legitimately partitions into TWO recurrent layers, and a
    shared linear applied to three disjoint slices feeds three distinct
    single-pass heads. Both are adversarial for any pairwise-sweep early
    exit: the sweep must keep running while multiple roots remain.
    """

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stream_a = x[:, 0]
        stream_b = x[:, 1]
        for _ in range(5):
            stream_a = torch.relu(stream_a) + 1
        for _ in range(5):
            stream_b = torch.relu(stream_b) + 1
        head_relu = torch.relu(self.shared(x[:, 0]))
        head_sigmoid = torch.sigmoid(self.shared(x[:, 1]))
        head_tanh = torch.tanh(self.shared(x[:, 2]))
        return (
            stream_a.sum() + stream_b.sum() + head_relu.sum() + head_sigmoid.sum() + head_tanh.sum()
        )


def _trace_with_adapter_find_calls(model: torch.nn.Module, x: torch.Tensor) -> tuple[Any, int]:
    """Trace ``model`` and count union-find ``find`` calls in the adapter."""
    profiler = cProfile.Profile()
    profiler.enable()
    traced = trace_fn(model, x)
    profiler.disable()
    find_calls = sum(
        call_count
        for (file_name, _line, func_name), (call_count, *_rest) in pstats.Stats(
            profiler
        ).stats.items()
        if func_name == "find" and file_name.endswith("loop_grouping_adapter.py")
    )
    return traced, find_calls


def test_pairwise_sweep_early_exit_keeps_chained_grouping_subquadratic() -> None:
    """Chained tied loop groups identically while the pair sweep stays subquadratic.

    Grouping both iso groups (tied linear and tanh, 64 members each) must not
    pay the historical O(members^2) already-unified pair tail: the quadratic
    sweep cost ~23k adapter ``find`` calls at 64 steps, the early-exit version
    ~3k. The bound is a call count, not a timing, so it is load-robust.
    """
    torch.manual_seed(0)
    num_steps = 64
    traced, find_calls = _trace_with_adapter_find_calls(
        _ChainedTiedRecurrentNet(num_steps), torch.rand(2, 16)
    )

    recurrent_passes = {op.layer_label: op.num_passes for op in traced.ops if op.num_passes > 1}
    assert len(recurrent_passes) == 2
    assert set(recurrent_passes.values()) == {num_steps}
    assert find_calls < 12_000


class _BareChain(torch.nn.Module):
    """A plain param-free feed-forward chain: the quadratic sweep's worst case."""

    def __init__(self, num_steps: int) -> None:
        super().__init__()
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_steps):
            x = torch.relu(x)
        return x


def test_bare_chain_sweep_is_linear_in_group_size() -> None:
    """A param-free chain must not pay the C(N,2) pairwise triangle.

    b4-opus r5 (F29-A): all N chain relus land in ONE iso group with no
    adjacency, no params, and no anchors, so no union can ever fire and
    neither documented short-circuit engages -- the full triangle ran
    (~161k adapter ``find`` calls at 400 ops, exactly 2*C(N,2)+O(N)) and
    bought zero grouping. The bucketed candidate sweep enumerates only
    pairs a union arm could accept, so the same capture stays linear
    (~1.2k finds). Call counts, not timings, so the bound is load-robust.
    """
    torch.manual_seed(0)
    num_steps = 400
    traced, find_calls = _trace_with_adapter_find_calls(_BareChain(num_steps), torch.rand(4))

    # Grouping oracle: a bare chain mints NO recurrent layers (layers == ops).
    assert all(op.num_passes == 1 for op in traced.ops)
    assert traced.num_ops == num_steps
    # Pre-fix this is >= 2*C(400,2) = 159,600; the bucketed sweep needs O(N).
    assert find_calls < 12_000


def test_pairwise_sweep_early_exit_preserves_multi_root_group_membership() -> None:
    """Multi-root candidate groups keep exact historical membership.

    A premature pairwise-sweep exit would either merge the two disconnected
    param-free streams into one layer or fail to accumulate each stream's
    five passes; the shared-weight heads must stay three distinct
    single-pass layers fed by one three-pass linear layer.
    """
    torch.manual_seed(0)
    traced, _ = _trace_with_adapter_find_calls(_ParallelStreamsNet(), torch.rand(2, 3, 8))

    num_passes = {op.layer_label: op.num_passes for op in traced.ops}
    relu_stream_layers = sorted(
        label for label, passes in num_passes.items() if label.startswith("relu") and passes == 5
    )
    assert len(relu_stream_layers) == 2
    add_stream_layers = sorted(
        label for label, passes in num_passes.items() if label.startswith("add") and passes == 5
    )
    assert len(add_stream_layers) == 2
    shared_linear_passes = {
        passes for label, passes in num_passes.items() if label.startswith("linear")
    }
    assert shared_linear_passes == {3}
    for head in ("sigmoid", "tanh"):
        head_passes = [p for label, p in num_passes.items() if label.startswith(head)]
        assert head_passes == [1]


class _ExplicitCatCellRNN(torch.nn.Module):
    """Explicit ``tanh(cell(cat))`` unrolled loop: the degenerate entry regime."""

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.cell = torch.nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            hidden = torch.tanh(self.cell(torch.cat([x[:, step], hidden], dim=1)))
        return hidden


class _AddCellRNN(torch.nn.Module):
    """Explicit ``tanh(w(h) + x_t)`` unrolled loop: the add-cell battery regime."""

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.w = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            hidden = torch.tanh(self.w(hidden) + x[:, step])
        return hidden


class _HandLSTM(torch.nn.Module):
    """Hand-rolled LSTM cell loop: entry admissions genuinely fire here."""

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.wi = torch.nn.Linear(2 * dim, dim)
        self.wf = torch.nn.Linear(2 * dim, dim)
        self.wo = torch.nn.Linear(2 * dim, dim)
        self.wg = torch.nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(x.shape[0], x.shape[2])
        cell = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            joint = torch.cat([x[:, step], hidden], dim=1)
            gate_i = torch.sigmoid(self.wi(joint))
            gate_f = torch.sigmoid(self.wf(joint))
            gate_o = torch.sigmoid(self.wo(joint))
            gate_g = torch.tanh(self.wg(joint))
            cell = gate_f * cell + gate_i * gate_g
            hidden = gate_o * torch.tanh(cell)
        return hidden


def _unfiltered_pf_partition_class_oracle(
    workspace: Any,
    members: list,
    signatures: dict,
    parent_colors: dict,
    consumer_site_frames: dict,
    complete_consumer_sites: set,
    class_of: dict,
    realizations: dict,
    reach_memo: Any,
) -> list:
    """Verbatim pre-prefilter ``_pf_partition_class``: the byte-identity oracle.

    This is the unfiltered O(members^2) entry sweep exactly as shipped before
    the entry-admission prefilter, kept as a reference implementation so the
    prefiltered production sweep can be asserted union-identical on every real
    partition call issued while tracing recurrent fixtures.
    """
    parent_map = {member: member for member in members}

    def find(label: str) -> str:
        while parent_map[label] != label:
            parent_map[label] = parent_map[parent_map[label]]
            label = parent_map[label]
        return label

    def union(label1: str, label2: str) -> None:
        root1, root2 = find(label1), find(label2)
        if root1 != root2:
            parent_map[root2] = root1

    cohorts: dict = defaultdict(list)
    for member in members:
        cohorts[frozenset(signatures[member].items())].append(member)
    cohort_sizes = {signature: len(cohort) for signature, cohort in cohorts.items()}

    for signature, cohort in cohorts.items():
        if len(cohort) < 2:
            continue
        invariant_fed = all(color[0] == "ext" for color, _ in signature)
        distinct_roots = len({find(member) for member in cohort})
        pair_iter = it.chain(zip(cohort, cohort[1:]), it.combinations(cohort, 2))
        for member1, member2 in pair_iter:
            if find(member1) == find(member2):
                continue
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
                lga._reaches_forward(workspace, member1, member2, reach_memo)
                or invariant_fed
                and lga._pf_child_route_allows(workspace, member1, member2, class_of, reach_memo)
            ):
                union(member1, member2)
                distinct_roots -= 1
                if distinct_roots == 1:
                    break

    equal_component_sizes = Counter(find(member) for member in members)
    realized_equal_members = {
        member for member in members if equal_component_sizes[find(member)] > 1
    }
    for index, entry in enumerate(members):
        if entry in realized_equal_members:
            continue
        entry_signature = frozenset(signatures[entry].items())
        for target in members[index + 1 :]:
            if frozenset(signatures[target].items()) == entry_signature:
                continue
            if find(entry) == find(target):
                break
            if lga._pf_entry_union_allowed(
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
                union(entry, target)
                break

    groups: dict = defaultdict(list)
    for member in members:
        groups[find(member)].append(member)
    return [
        sorted(group, key=lambda label: workspace.nodes[label].raw_order)
        for group in groups.values()
    ]


def test_entry_sweep_prefilter_matches_unfiltered_oracle(monkeypatch: Any) -> None:
    """Prefiltered entry sweep is partition-identical to the unfiltered oracle.

    Every real ``_pf_partition_class`` call issued while tracing the battery
    (degenerate cat-cell, add-cell, admissions-firing hand LSTM, and the
    multi-root parallel-streams adversary) must return exactly the groups the
    verbatim pre-prefilter sweep returns, in the same order.
    """
    production = lga._pf_partition_class
    compared_calls = {"count": 0}

    def comparing_partition(*args: Any, **kwargs: Any) -> list:
        produced = production(*args, **kwargs)
        oracle = _unfiltered_pf_partition_class_oracle(*args, **kwargs)
        assert produced == oracle
        compared_calls["count"] += 1
        return produced

    monkeypatch.setattr(lga, "_pf_partition_class", comparing_partition)
    torch.manual_seed(0)
    trace_fn(_ExplicitCatCellRNN(32), torch.rand(2, 32, 8))
    trace_fn(_AddCellRNN(32), torch.rand(2, 32, 8))
    trace_fn(_HandLSTM(24), torch.rand(2, 24, 8))
    trace_fn(_ParallelStreamsNet(), torch.rand(2, 3, 8))
    assert compared_calls["count"] > 0


def test_entry_sweep_prefilter_empties_degenerate_pair_triangle(monkeypatch: Any) -> None:
    """The degenerate long-loop regime issues ZERO entry-admission pair calls.

    Every cat-cell signature is a distinct singleton cohort of bare colors, so
    all three admission arms are structurally unreachable: the prefilter must
    empty the pair triangle outright (the pre-prefilter sweep paid C(N-1, 2)
    calls here) while grouping still assigns every loop op its full pass
    count. The hand LSTM guards the other direction: admissions still fire
    through the prefilter.
    """
    real_allowed = lga._pf_entry_union_allowed
    outcomes: list[bool] = []

    def spying_allowed(*args: Any, **kwargs: Any) -> bool:
        allowed = real_allowed(*args, **kwargs)
        outcomes.append(allowed)
        return allowed

    monkeypatch.setattr(lga, "_pf_entry_union_allowed", spying_allowed)
    torch.manual_seed(0)
    num_steps = 32
    traced = trace_fn(_ExplicitCatCellRNN(num_steps), torch.rand(2, num_steps, 8))
    assert outcomes == []
    recurrent_passes = {op.layer_label: op.num_passes for op in traced.ops if op.num_passes > 1}
    assert set(recurrent_passes.values()) == {num_steps}

    outcomes.clear()
    trace_fn(_HandLSTM(24), torch.rand(2, 24, 8))
    assert any(outcomes)


class _TwinCatCellRNN(torch.nn.Module):
    """Two independent cat-cell loops: a mixed-ancestry multi-root bare group.

    Each stream's FIRST ``cat`` consumes only the raw input and a fresh zeros
    tensor (no anchored ancestry) while every later ``cat`` consumes the
    previous ``tanh(linear(...))`` (anchored ancestry), so the ``cat``
    candidate group mixes ancestry-free and ancestry-carrying members with
    MULTIPLE surviving roots and nonconsecutive eligible pairs -- adversarial
    for any pair-space restriction in the iso-group merge sweep.
    """

    def __init__(self, num_steps: int, dim: int = 8) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.cell_a = torch.nn.Linear(2 * dim, dim)
        self.cell_b = torch.nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_a = torch.zeros(x.shape[0], x.shape[2])
        hidden_b = torch.zeros(x.shape[0], x.shape[2])
        for step in range(self.num_steps):
            hidden_a = torch.tanh(self.cell_a(torch.cat([x[:, step], hidden_a], dim=1)))
        for step in range(self.num_steps):
            hidden_b = torch.tanh(self.cell_b(torch.cat([x[:, step], hidden_b], dim=1)))
        return hidden_a + hidden_b


def _unrestricted_merge_iso_groups_oracle(
    workspace: Any,
    iso_node_groups: dict,
    node_to_subgraph: dict,
    adjacent_subgraphs: dict,
) -> dict:
    """Verbatim pre-restriction ``_merge_iso_groups_to_layers``: the identity oracle.

    This is the full-triangle sweep exactly as shipped before the bare-group
    combinations restriction (every pair of an all-bare group enumerated, with
    the historical all-anchored group skip), kept as a reference so the
    restricted production sweep can be asserted merge-identical on every real
    call issued while tracing adversarial recurrent fixtures.
    """
    uf_parent: dict = {}

    def find(x: str) -> str:
        if x not in uf_parent:
            uf_parent[x] = x
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx > ry:
                rx, ry = ry, rx
            uf_parent[ry] = rx

    all_iso_nodes: set = set()
    for iso_nodes_orig in iso_node_groups.values():
        all_iso_nodes.update(iso_nodes_orig)

    sg_param_types: dict = {}
    for iso_nodes_orig in iso_node_groups.values():
        for node_label in iso_nodes_orig:
            sg = node_to_subgraph[node_label]
            sg_label = sg.starting_node
            if sg_label not in sg_param_types:
                sg_param_types[sg_label] = frozenset(
                    workspace.nodes[pnode].equivalence_key for pnode in sg.param_nodes
                )

    reach_memo = lga._ReachabilityCache(workspace)
    anchor_ancestry = lga._topology_anchor_ancestry(workspace)

    for _iso_group_label, iso_nodes_orig in iso_node_groups.items():
        iso_nodes = sorted(
            iso_nodes_orig, key=lambda node_label: workspace.nodes[node_label].raw_order
        )
        distinct_roots = len({find(node_label) for node_label in iso_nodes})
        if distinct_roots == 1:
            continue
        if all(
            not workspace.nodes[node_label].uses_params
            and not workspace.nodes[node_label].recurrence_anchored
            and anchor_ancestry[node_label]
            for node_label in iso_nodes
        ):
            continue
        pair_iter = it.chain(zip(iso_nodes, iso_nodes[1:]), it.combinations(iso_nodes, 2))
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
                and lga._param_call_identity(node1) != lga._param_call_identity(node2)
            ):
                continue
            pair_anchored = node1.recurrence_anchored or node2.recurrence_anchored
            if not (node1.uses_params or node2.uses_params or pair_anchored):
                if (
                    not anchor_ancestry[node1_label]
                    and not anchor_ancestry[node2_label]
                    and subgraphs_are_adjacent
                    and lga._param_free_adjacency_merge_allowed(
                        workspace,
                        node1_label,
                        node2_label,
                        node_to_subgraph[node1_label],
                        node_to_subgraph[node2_label],
                    )
                ):
                    union(node1_label, node2_label)
                    distinct_roots -= 1
                    if distinct_roots == 1:
                        break
                continue
            overlapping_param_types = (
                sg_param_types[node1_subgraph_label] & sg_param_types[node2_subgraph_label]
            )
            if overlapping_param_types:
                if subgraphs_are_adjacent or lga._seed_reaches(
                    workspace, node1_label, node2_label, reach_memo
                ):
                    union(node1_label, node2_label)
                    distinct_roots -= 1
                    if distinct_roots == 1:
                        break
            elif subgraphs_are_adjacent and pair_anchored:
                union(node1_label, node2_label)
                distinct_roots -= 1
                if distinct_roots == 1:
                    break

    param_barcode_groups: dict = defaultdict(list)
    for node_label in all_iso_nodes:
        node = workspace.nodes[node_label]
        if node.uses_params and node.param_barcodes:
            param_barcode_groups[lga._param_call_identity(node)].append(node_label)

    for _identity_key, nodes_with_same_params in param_barcode_groups.items():
        if len(nodes_with_same_params) > 1:
            first = nodes_with_same_params[0]
            for other in nodes_with_same_params[1:]:
                union(first, other)

    merged_layer_groups: dict = defaultdict(set)
    for node_label in all_iso_nodes:
        root = find(node_label)
        merged_layer_groups[root].add(node_label)

    return {leader: nodes for leader, nodes in merged_layer_groups.items() if len(nodes) > 1}


@pytest.mark.heavy
def test_bare_group_restriction_matches_unrestricted_merge_oracle(monkeypatch: Any) -> None:
    """Restricted iso-group merge is merge-identical to the full-triangle oracle.

    Every real ``_merge_iso_groups_to_layers`` call issued while tracing the
    adversarial battery (mixed-ancestry cat-cell, twin multi-root cat-cell
    streams, admissions-firing hand LSTM, and the pure param-free
    parallel-streams net whose bare groups have NO anchored ancestry and must
    keep their full triangle) must return exactly the merged groups -- same
    min-label roots, same member sets -- the unrestricted sweep returns.
    """
    production = lga._merge_iso_groups_to_layers
    compared_calls = {"count": 0}

    def comparing_merge(*args: Any, **kwargs: Any) -> dict:
        produced = production(*args, **kwargs)
        oracle = _unrestricted_merge_iso_groups_oracle(*args, **kwargs)
        assert produced == oracle
        compared_calls["count"] += 1
        return produced

    monkeypatch.setattr(lga, "_merge_iso_groups_to_layers", comparing_merge)
    torch.manual_seed(0)
    trace_fn(_ExplicitCatCellRNN(32), torch.rand(2, 32, 8))
    trace_fn(_TwinCatCellRNN(16), torch.rand(2, 16, 8))
    trace_fn(_HandLSTM(24), torch.rand(2, 24, 8))
    trace_fn(_ParallelStreamsNet(), torch.rand(2, 3, 8))
    assert compared_calls["count"] > 0


def _reachability_workspace(
    labels: list[str],
    raw_orders: dict[str, int],
    children: dict[str, tuple[str, ...]],
) -> Any:
    """Build a minimal grouping workspace exposing only reachability topology."""
    nodes = {
        label: lga._MutableRecurrenceNode(
            label=label,
            raw_order=raw_orders[label],
            equivalence_key="key",
            equivalent_labels=(),
            data_parents=(),
            data_children=children.get(label, ()),
            layer_label=label,
            recurrent_labels=[],
            uses_params=False,
            func_name="func",
            param_barcodes=(),
        )
        for label in labels
    }
    return lga._GroupingWorkspace(
        nodes=nodes,
        raw_labels=tuple(sorted(labels, key=lambda label: raw_orders[label])),
        source_labels=(labels[0],),
        eligible_labels=set(labels),
    )


def _random_monotone_workspace(seed: int, num_nodes: int, edge_probability: float) -> Any:
    """Build a random forward-edge DAG workspace in topological insertion order."""
    rng = random.Random(seed)
    labels = [f"n{index:03d}" for index in range(num_nodes)]
    children = {
        labels[i]: tuple(
            labels[j] for j in range(i + 1, num_nodes) if rng.random() < edge_probability
        )
        for i in range(num_nodes)
    }
    raw_orders = {label: index for index, label in enumerate(labels)}
    return _reachability_workspace(labels, raw_orders, children)


def _reference_reaches(workspace: Any, src_label: str, dst_label: str) -> bool:
    """Plain unbounded DFS reachability: the reachability ground truth."""
    if src_label == dst_label:
        return True
    stack = [src_label]
    seen = {src_label}
    while stack:
        for child in workspace.nodes[stack.pop()].data_children:
            if child == dst_label:
                return True
            if child not in seen and child in workspace.nodes:
                seen.add(child)
                stack.append(child)
    return False


@pytest.mark.smoke
def test_adaptive_reachability_matches_reference_on_random_dags() -> None:
    """Sparse, dense, and repeated queries all match ground-truth reachability.

    The query mix drives every adaptive lane: one distinct target per source
    (the post-prefilter sparse shape, answered by the bounded pair query),
    repeated identical pairs (the pair memo), and dense sources with many
    distinct targets (mask materialization through the batch DP). Dense
    demand must actually materialize masks -- the adaptive policy defers the
    full-mask regime, never disables it.
    """
    for seed in range(12):
        workspace = _random_monotone_workspace(seed, num_nodes=40, edge_probability=0.08)
        labels = list(workspace.nodes)
        cache = lga._ReachabilityCache(workspace)
        rng = random.Random(1000 + seed)
        queries: list[tuple[str, str]] = []
        for index in range(len(labels) - 1):
            queries.append((labels[index], labels[index + 1]))
        for src_label in rng.sample(labels[: len(labels) // 2], 3):
            src_order = workspace.nodes[src_label].raw_order
            later = [label for label in labels if workspace.nodes[label].raw_order >= src_order]
            for dst_label in rng.sample(later, min(8, len(later))):
                queries.append((src_label, dst_label))
        queries.extend(queries[:10])
        for src_label, dst_label in queries:
            assert cache.reaches_from_earlier(src_label, dst_label) == _reference_reaches(
                workspace, src_label, dst_label
            ), (seed, src_label, dst_label)
        assert cache._batch_attempted
        assert len(cache._descendant_bits) == len(labels)


@pytest.mark.smoke
def test_batch_dp_masks_equal_per_source_bfs_masks() -> None:
    """The reverse-topological batch DP builds bit-identical descendant masks."""
    for seed in (0, 1, 2):
        workspace = _random_monotone_workspace(seed, num_nodes=60, edge_probability=0.06)
        batch_cache = lga._ReachabilityCache(workspace)
        batch_cache._prepare()
        batch_cache._batch_build_descendant_masks()
        bfs_cache = lga._ReachabilityCache(workspace)
        bfs_cache._prepare()
        assert batch_cache._descendant_bits, "batch DP unexpectedly abandoned"
        for label in workspace.nodes:
            assert batch_cache._descendant_bits[label] == bfs_cache._build_descendant_mask(label), (
                seed,
                label,
            )


@pytest.mark.smoke
def test_tied_insertion_order_falls_back_to_per_source_bfs() -> None:
    """A raw-order tie inserted child-first abandons the batch DP, not correctness.

    ``_order_monotone`` accepts ``raw_order`` ties, but the batch DP requires
    strict insertion-order topology: with the tied child inserted BEFORE its
    parent, a naive DP would drop the child's own descendants from the
    parent's mask. The guard must abandon the batch and serve the dense
    source through the exact per-source BFS, including the grand-descendant
    reached through the tied edge.
    """
    labels = ["tied_child", "tied_parent", "grandchild", "detached"]
    raw_orders = {"tied_parent": 0, "tied_child": 0, "grandchild": 1, "detached": 2}
    children = {
        "tied_parent": ("tied_child",),
        "tied_child": ("grandchild",),
    }
    workspace = _reachability_workspace(labels, raw_orders, children)
    cache = lga._ReachabilityCache(workspace)

    assert cache.reaches_from_earlier("tied_parent", "tied_child")
    assert cache.reaches_from_earlier("tied_parent", "grandchild")
    assert not cache.reaches_from_earlier("tied_parent", "detached")
    assert cache._batch_attempted
    assert set(cache._descendant_bits) == {"tied_parent"}
    assert cache.reaches_from_earlier("tied_parent", "grandchild")


@pytest.mark.smoke
def test_non_monotone_workspace_keeps_bounded_pair_lane() -> None:
    """A raw-order-violating edge disables masks entirely, answers stay exact."""
    labels = ["late_parent", "early_child", "tail"]
    raw_orders = {"late_parent": 5, "early_child": 1, "tail": 6}
    children = {"late_parent": ("early_child",), "early_child": ("tail",)}
    workspace = _reachability_workspace(labels, raw_orders, children)
    cache = lga._ReachabilityCache(workspace)

    for _ in range(4):
        assert cache.reaches_from_earlier("early_child", "tail")
        assert not cache.reaches_from_earlier("early_child", "late_parent")
    assert cache._order_monotone is False
    assert not cache._descendant_bits
    assert not cache._batch_attempted


def test_assignments_pool_one_recurrent_labels_tuple_per_group() -> None:
    """Assignments equal the historical per-node scan and pool one tuple per group."""
    torch.manual_seed(0)
    traced = trace_fn(_ExplicitCatCellRNN(24), torch.rand(2, 24, 8))
    graph = _neutral_graph_from_torch_recurrent_fixture(traced)
    workspace = lga._GroupingWorkspace.from_graph(graph)
    lga._detect_and_label_workspace_loops(workspace)
    assignments = workspace.assignments()

    expected = {
        label: lga.RecurrenceAssignment(
            layer_label=node.layer_label,
            recurrent_labels=tuple(node.recurrent_labels),
            pass_index=index + 1,
            num_passes=len(node.recurrent_labels),
            equivalence_key=node.equivalence_key,
        )
        for label, node in workspace.nodes.items()
        for index, recurrent_label in enumerate(node.recurrent_labels)
        if recurrent_label == label
    }
    assert assignments == expected
    assert list(assignments) == list(expected)

    instances_by_group: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for assignment in assignments.values():
        if assignment.num_passes > 1:
            instances_by_group[assignment.recurrent_labels].append(assignment.recurrent_labels)
    assert instances_by_group
    for members, instances in instances_by_group.items():
        assert len(instances) == len(members)
        assert all(instance is instances[0] for instance in instances)


def test_merge_iso_groups_visits_linear_pairs_on_an_unrelated_chain() -> None:
    """The pair sweep must not walk C(N,2) over an ungroupable iso group (R29/F29-A).

    Every union arm requires subgraph adjacency or shared param types, yet
    the sweep walked the full ``it.combinations`` triangle over a plain
    chain of N identical bare ops -- 35% of capture CPU at 3200 ops, rising
    as N^2, with ZERO groupings produced (the documented short-circuits
    never fire because no union ever happens). Candidate enumeration is now
    relation-driven; an all-unrelated group must cost O(N), pinned here by
    counting node_to_subgraph lookups (two per visited pair).
    """

    from torchlens.postprocess.loop_grouping_adapter import (
        SubgraphInfo,
        _GroupingWorkspace,
        _merge_iso_groups_to_layers,
        _MutableRecurrenceNode,
    )

    n = 300
    labels = [f"relu_{index}" for index in range(n)]
    nodes = {}
    for index, label in enumerate(labels):
        nodes[label] = _MutableRecurrenceNode(
            label=label,
            raw_order=index,
            equivalence_key="relu",
            equivalent_labels=tuple(labels),
            data_parents=(labels[index - 1],) if index else (),
            data_children=(labels[index + 1],) if index < n - 1 else (),
            layer_label=label,
            recurrent_labels=[],
            uses_params=False,
            func_name="relu",
            param_barcodes=(),
        )
    workspace = _GroupingWorkspace(
        nodes=nodes,
        raw_labels=tuple(labels),
        source_labels=(labels[0],),
        eligible_labels=set(labels),
    )

    class _CountingDict(dict):
        lookups = 0

        def __getitem__(self, key):
            type(self).lookups += 1
            return super().__getitem__(key)

    node_to_subgraph = _CountingDict({label: SubgraphInfo(starting_node=label) for label in labels})
    # NO adjacency and NO param types: no pair anywhere in the group can
    # union, exactly the plain feed-forward chain shape.
    merged = _merge_iso_groups_to_layers(
        workspace,
        {"relu_0": list(labels)},
        node_to_subgraph,
        {},
    )
    assert merged == {}, "an unrelated bare chain must produce no merged groups"
    assert _CountingDict.lookups < 20 * n, (
        f"pair sweep walked {_CountingDict.lookups} node_to_subgraph lookups for "
        f"{n} unrelated nodes -- the C(N,2) triangle is back (expected O(N))"
    )
