"""Tests for backend-neutral loop grouping adapter."""

import itertools as it
import random
from collections import Counter, defaultdict
from typing import Any

import pytest
import torch
from support.structural_grouper import legacy_structural_grouper  # noqa: F401

import torchlens._postprocess.loop_grouping_adapter as lga
from torchlens import trace as trace_fn

# This suite drives the structural grouper (grouper_v1), which serves captures
# without call-site evidence; torch captures group by call site
# (tests/test_grouping_call_sites.py).
pytestmark = pytest.mark.usefixtures("legacy_structural_grouper")


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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/loop_grouping_adapter.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/loop_grouping_adapter.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/loop_grouping_adapter.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/loop_grouping_adapter.py.
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
