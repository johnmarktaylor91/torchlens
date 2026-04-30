"""Tests for the multi_trace subpackage (TraceBundle + supergraph).

Covers the 21 required cases from the Phase 1 spec plus a handful of edge
cases discovered during implementation.  Each test is self-contained --
fixtures live in the test file itself rather than in conftest.py because
the multi_trace tests are the only callers.
"""

from __future__ import annotations

import warnings
from typing import List

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.multi_trace import (
    NodeView,
    TopologyDiff,
    TraceBundle,
    bundle,
    compare_topology,
)
from torchlens.multi_trace.metrics import (
    METRIC_REGISTRY,
    cosine_distance,
    relative_l1_scalar,
    resolve_metric,
)

pytestmark = pytest.mark.skip(
    reason="Phase 9 redesign: TraceBundle public semantics were replaced by tl.bundle Bundle."
)


# ---------------------------------------------------------------------------
# Tiny model fixtures
# ---------------------------------------------------------------------------


class _LinearNet(nn.Module):
    """Small static model used for shared-topology tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class _BranchNet(nn.Module):
    """Conditionally branching model used for divergent-topology tests.

    Picks between ``relu`` and ``sigmoid`` based on the sign of the input
    mean, producing distinct supergraph nodes on each branch.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.mean() > 0:
            return torch.relu(x)
        return torch.sigmoid(x)


class _ScalarOutNet(nn.Module):
    """Model whose output is a 0-d tensor (sum reduction)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).sum()


def _trace_linear(n_traces: int, batch: int = 2) -> List[tl.ModelLog]:
    model = _LinearNet()
    return [tl.log_forward_pass(model, torch.rand(batch, 4)) for _ in range(n_traces)]


def _trace_branch_pair() -> tuple[tl.ModelLog, tl.ModelLog]:
    model = _BranchNet()
    ml_pos = tl.log_forward_pass(model, torch.ones(2, 4))
    ml_neg = tl.log_forward_pass(model, -torch.ones(2, 4))
    return ml_pos, ml_neg


# ---------------------------------------------------------------------------
# 1. Construction tests
# ---------------------------------------------------------------------------


def test_bundle_construction_shared_topology() -> None:
    traces = _trace_linear(3)
    b = TraceBundle(traces)
    assert len(b) == 3
    assert b.is_shared_topology is True
    assert len(b.universal_nodes) == len(b.nodes)
    assert b.coverage("trace_0") == pytest.approx(1.0)
    assert b.coverage("trace_2") == pytest.approx(1.0)


def test_bundle_construction_divergent_topology() -> None:
    ml_pos, ml_neg = _trace_branch_pair()
    b = TraceBundle([ml_pos, ml_neg])
    assert b.is_shared_topology is False
    selective = b.selective_nodes()
    assert len(selective) > 0, "Expected at least the relu/sigmoid branch nodes to be selective"
    # Both branch ops must appear somewhere in the supergraph.
    nodes = b.nodes
    assert any("relu" in name for name in nodes), f"relu not in supergraph nodes: {nodes}"
    assert any("sigmoid" in name for name in nodes), f"sigmoid not in supergraph nodes: {nodes}"


# ---------------------------------------------------------------------------
# 2. Node view tests
# ---------------------------------------------------------------------------


def test_node_view_shared_topology_stacked() -> None:
    traces = _trace_linear(3, batch=2)
    b = TraceBundle(traces)
    # Pick a node that all traces traverse: relu_1_2 on first call.
    nodes = b.nodes
    relu_nodes = [n for n in nodes if "relu" in n]
    assert relu_nodes, f"no relu in nodes: {nodes}"
    target = relu_nodes[0]
    view = b[target]
    assert isinstance(view, NodeView)
    activations = view.activations
    assert len(activations) == 3
    assert all(t is not None for t in activations)
    stacked = view.activation
    assert stacked.shape[0] == 2 * 3  # batch=2 across 3 traces
    assert stacked.shape[1:] == activations[0].shape[1:]


def test_node_view_divergent_topology_list() -> None:
    ml_pos, ml_neg = _trace_branch_pair()
    b = TraceBundle([ml_pos, ml_neg])
    # Find the relu node -- only ml_pos traverses it.
    relu_node_names = [n for n in b.nodes if "relu" in n]
    assert relu_node_names, f"no relu in {b.nodes}"
    relu_view = b[relu_node_names[0]]
    assert isinstance(relu_view, NodeView)
    assert len(relu_view.activations) < len(b)
    with pytest.raises(ValueError) as exc:
        _ = relu_view.activation
    msg = str(exc.value)
    assert "not all" in msg.lower() or "missing" in msg.lower()


def test_node_view_traces_set() -> None:
    ml_pos, ml_neg = _trace_branch_pair()
    b = TraceBundle([ml_pos, ml_neg], names=["pos", "neg"])
    relu_nodes = [n for n in b.nodes if "relu" in n]
    sigmoid_nodes = [n for n in b.nodes if "sigmoid" in n]
    assert relu_nodes and sigmoid_nodes
    assert b[relu_nodes[0]].traces == {"pos"}
    assert b[sigmoid_nodes[0]].traces == {"neg"}


# ---------------------------------------------------------------------------
# 3. Diff tests
# ---------------------------------------------------------------------------


def test_diff_cosine_pairwise() -> None:
    traces = _trace_linear(3)
    b = TraceBundle(traces)
    target = next(n for n in b.nodes if "relu" in n)
    matrix = b[target].diff(metric="cosine")
    assert matrix.shape == (3, 3)
    # Diagonal must be 0 (or near-zero for floating point).
    for i in range(3):
        assert float(matrix[i, i]) == pytest.approx(0.0, abs=1e-6)
    # Symmetric
    for i in range(3):
        for j in range(3):
            assert float(matrix[i, j]) == pytest.approx(float(matrix[j, i]), abs=1e-6)


def test_diff_scalar_fallback() -> None:
    model = _ScalarOutNet()
    traces = [tl.log_forward_pass(model, torch.rand(2, 4)) for _ in range(3)]
    b = TraceBundle(traces)
    # output_1 is a scalar (sum reduction).
    output_node = next(n for n in b.nodes if n.startswith("output"))
    view = b[output_node]
    matrix = view.diff(metric="cosine")
    assert matrix.shape == (3, 3)
    assert torch.isnan(matrix).any().item() is False
    # Diagonal still zero.
    for i in range(3):
        assert float(matrix[i, i]) == pytest.approx(0.0, abs=1e-6)


def test_diff_to_specific_other() -> None:
    traces = _trace_linear(3)
    b = TraceBundle(traces, names=["a", "b", "c"])
    target = next(n for n in b.nodes if "relu" in n)
    row = b[target].diff(other="a", metric="cosine")
    assert row.shape == (1, 3)
    # diff to self should be 0.
    a_idx = b.names.index("a")
    assert float(row[0, a_idx]) == pytest.approx(0.0, abs=1e-6)


def test_diff_on_gradient() -> None:
    model = _LinearNet()
    traces: List[tl.ModelLog] = []
    for _ in range(2):
        x = torch.rand(2, 4, requires_grad=True)
        ml = tl.log_forward_pass(model, x, save_gradients=True, train_mode=True)
        loss = ml.layer_logs[ml.output_layers[0]].activation.sum()
        loss.backward()
        traces.append(ml)
    b = TraceBundle(traces)
    assert b.has_gradients is True
    target = next(n for n in b.nodes if "relu" in n)
    matrix = b[target].diff(metric="cosine", on="gradient")
    assert matrix.shape == (2, 2)
    assert float(matrix[0, 0]) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Aggregate / ranking tests
# ---------------------------------------------------------------------------


def test_aggregate_mean() -> None:
    traces = _trace_linear(3)
    b = TraceBundle(traces)
    target = next(n for n in b.nodes if "relu" in n)
    mean = b[target].aggregate("mean")
    # Manual: average of the three per-trace activations along axis 0.
    activations = b[target].activations
    stacked = torch.stack([a.to(torch.float32) for a in activations], dim=0)
    expected = stacked.mean(dim=0)
    assert torch.allclose(mean, expected, atol=1e-6)


def test_most_changed() -> None:
    # Construct three traces with a node guaranteed to vary heavily.
    model = _LinearNet()
    inputs = [torch.zeros(2, 4), torch.ones(2, 4) * 5.0, torch.ones(2, 4) * -5.0]
    traces = [tl.log_forward_pass(model, x) for x in inputs]
    b = TraceBundle(traces)
    ranked = b.most_changed(top_k=10, metric="cosine")
    assert len(ranked) > 0
    # Strictly non-increasing scores.
    for prior, latter in zip(ranked, ranked[1:]):
        assert prior[1] >= latter[1] - 1e-9
    # Each score should be a finite number.
    for _, score in ranked:
        assert score == score  # NaN check
        assert score >= -1e-9


# ---------------------------------------------------------------------------
# 5. Groups + coverage tests
# ---------------------------------------------------------------------------


def test_groups() -> None:
    ml_pos1, ml_neg = _trace_branch_pair()
    # Add a second positive trace so 'cats' has 2 members and 'dogs' has 1.
    model = _BranchNet()
    ml_pos2 = tl.log_forward_pass(model, torch.ones(2, 4) * 0.5)
    b = TraceBundle(
        [ml_pos1, ml_pos2, ml_neg],
        names=["cat_01", "cat_02", "dog_01"],
        groups={"cats": ["cat_01", "cat_02"], "dogs": ["dog_01"]},
    )
    cat_only = b.selective_nodes(group="cats")
    dog_only = b.selective_nodes(group="dogs")
    # Cats traversed relu, dogs traversed sigmoid.
    relu_nodes = [n for n in cat_only if "relu" in n]
    sigmoid_nodes = [n for n in dog_only if "sigmoid" in n]
    assert relu_nodes, f"expected relu node in cats-only nodes: {cat_only}"
    assert sigmoid_nodes, f"expected sigmoid node in dogs-only nodes: {dog_only}"


def test_coverage() -> None:
    ml_pos, ml_neg = _trace_branch_pair()
    b = TraceBundle([ml_pos, ml_neg], names=["pos", "neg"])
    cov_pos = b.coverage("pos")
    cov_neg = b.coverage("neg")
    # Each trace traversed only its own branch, so neither has full coverage.
    assert 0.0 < cov_pos < 1.0
    assert 0.0 < cov_neg < 1.0


# ---------------------------------------------------------------------------
# 6. Edge case + assertion tests
# ---------------------------------------------------------------------------


def test_empty_bundle_errors() -> None:
    with pytest.raises(ValueError):
        TraceBundle([])


def test_single_trace_bundle() -> None:
    traces = _trace_linear(1)
    b = TraceBundle(traces)
    assert len(b) == 1
    assert b.is_shared_topology is True
    target = next(n for n in b.nodes if "relu" in n)
    # most_changed returns nothing because no node has 2+ traversers.
    ranked = b.most_changed(top_k=5)
    assert ranked == []
    # diff on a single-trace node yields a 1x1 zero matrix.
    matrix = b[target].diff(metric="cosine")
    assert matrix.shape == (1, 1)
    assert float(matrix[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_assert_shared_topology() -> None:
    ml_pos, ml_neg = _trace_branch_pair()
    b_div = TraceBundle([ml_pos, ml_neg])
    with pytest.raises(ValueError):
        b_div.assert_shared_topology()
    b_shared = TraceBundle(_trace_linear(2))
    # Should not raise.
    b_shared.assert_shared_topology()


# ---------------------------------------------------------------------------
# 7. Topology comparison tests
# ---------------------------------------------------------------------------


def test_topology_compare_identical() -> None:
    traces = _trace_linear(1)
    diff = compare_topology(traces[0], traces[0])
    assert isinstance(diff, TopologyDiff)
    assert diff.is_identical is True
    assert diff.unmatched_a == []
    assert diff.unmatched_b == []
    assert len(diff.matched) == len(traces[0].layer_logs)


def test_topology_compare_divergent() -> None:
    ml_pos, ml_neg = _trace_branch_pair()
    diff = compare_topology(ml_pos, ml_neg)
    assert diff.is_identical is False
    # ml_pos has the relu node only; ml_neg has the sigmoid node only.
    pos_only = set(diff.unmatched_a)
    neg_only = set(diff.unmatched_b)
    assert any("relu" in n for n in pos_only), f"unmatched_a missing relu: {pos_only}"
    assert any("sigmoid" in n for n in neg_only), f"unmatched_b missing sigmoid: {neg_only}"
    # Matched should include the shared input/output/mean/gt nodes.
    assert len(diff.matched) > 0


# ---------------------------------------------------------------------------
# 8. Repr + factory + smoke
# ---------------------------------------------------------------------------


def test_repr() -> None:
    traces = _trace_linear(3)
    b = TraceBundle(traces)
    r = repr(b)
    assert "N=3" in r
    assert "supergraph nodes" in r
    assert "shared_topology=True" in r
    assert "has_gradients=" in r


@pytest.mark.smoke
def test_smoke_basic_bundle() -> None:
    traces = _trace_linear(2)
    b = bundle(traces)
    assert len(b) == 2
    assert b.is_shared_topology is True
    target = next(n for n in b.nodes if "relu" in n)
    view = b[target]
    assert isinstance(view, NodeView)
    assert view.activation.shape[0] >= 2


def test_factory_function() -> None:
    traces = _trace_linear(2)
    b1 = bundle(traces)
    b2 = TraceBundle(traces)
    assert isinstance(b1, TraceBundle)
    assert b1.names == b2.names
    assert b1.nodes == b2.nodes


# ---------------------------------------------------------------------------
# Bonus / edge-case coverage (additive, beyond the spec's 21)
# ---------------------------------------------------------------------------


def test_metric_registry_contains_expected_keys() -> None:
    assert "cosine" in METRIC_REGISTRY
    assert "relative_l2" in METRIC_REGISTRY
    assert "pearson" in METRIC_REGISTRY


def test_resolve_metric_callable_passthrough() -> None:
    sentinel = lambda a, b: torch.tensor(7.0)  # noqa: E731
    assert resolve_metric(sentinel) is sentinel
    with pytest.raises(ValueError):
        resolve_metric("not_a_metric")


def test_invalid_indexing() -> None:
    b = TraceBundle(_trace_linear(2))
    with pytest.raises(KeyError):
        _ = b["does_not_exist"]
    with pytest.raises(TypeError):
        _ = b[1.5]  # type: ignore[arg-type]


def test_groups_unknown_member_rejected() -> None:
    traces = _trace_linear(2)
    with pytest.raises(ValueError):
        TraceBundle(traces, names=["a", "b"], groups={"x": ["c"]})


def test_names_mismatch_raises() -> None:
    traces = _trace_linear(2)
    with pytest.raises(ValueError):
        TraceBundle(traces, names=["a"])
    with pytest.raises(ValueError):
        TraceBundle(traces, names=["a", "a"])  # duplicates


def test_large_bundle_warning() -> None:
    # Build a fake list of MLs by reusing a single trace; the warning fires
    # purely on length.
    one = _trace_linear(1)[0]
    fake = [one] * 101
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TraceBundle(fake)
    assert any("100" in str(w.message) or "stats" in str(w.message).lower() for w in caught)


def test_metric_helpers_sanity() -> None:
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    cos_d = cosine_distance(a, b).item()
    assert cos_d == pytest.approx(1.0, abs=1e-6)
    same = cosine_distance(a, a).item()
    assert same == pytest.approx(0.0, abs=1e-6)
    # Scalar fallback
    s1 = torch.tensor(2.0)
    s2 = torch.tensor(3.0)
    val = relative_l1_scalar(s1, s2).item()
    assert val == pytest.approx(0.5, abs=1e-6)
