"""``Bundle.compare`` on a fork of the same model: recurrence, masks, shapes.

Three regressions from the behaviour net's ``bundle-validate`` cells:

* a recurrent model folds its passes into layers, so the layer-level edge
  set carries feedback edges; the supergraph must still order its nodes
  instead of refusing the member as cyclic;
* two members holding the same ``-inf`` attention mask must read distance
  ``0.0`` at that site, not ``nan``;
* operands of different shapes refuse (typed) naming the site.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.bundle._deltas import _distance_value
from torchlens.intervention._topology.topology import (
    _preferential_topological_order,
    build_supergraph,
)
from torchlens.intervention.errors import BundleComparisonError


class _Recurrent(nn.Module):
    """Three unrolled steps; each step's output feeds the next step's add."""

    def __init__(self) -> None:
        """Initialize modules."""
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent steps from a single-pass gate.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final hidden state.
        """
        gate = torch.sigmoid(x)
        hidden = gate
        for _ in range(3):
            hidden = torch.tanh(self.lin(gate) + hidden)
        return hidden


class _Masked(nn.Module):
    """Self-attention scores under an additive ``-inf`` causal mask."""

    def __init__(self) -> None:
        """Initialize modules."""
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the rows of ``x`` under a causal mask.

        Parameters
        ----------
        x:
            Input tensor of shape ``(rows, 4)``.

        Returns
        -------
        torch.Tensor
            Attended values.
        """
        hidden = self.lin(x)
        mask = torch.full((x.shape[0], x.shape[0]), float("-inf")).triu(1)
        scores = hidden @ hidden.transpose(0, 1) + mask
        return torch.softmax(scores, dim=-1) @ hidden


class _SubstringLabels(nn.Module):
    """Ten ops whose first label (``t_1_1``) is a substring of the last (``cat_1_10``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose, eight elementwise ops, transpose back, concatenate.

        Parameters
        ----------
        x:
            Input tensor of shape ``(rows, cols)``.

        Returns
        -------
        torch.Tensor
            Concatenation of the processed rows and the input.
        """
        hidden = x.t()
        hidden = hidden + 1
        hidden = hidden * 2
        hidden = torch.relu(hidden)
        hidden = hidden - 1
        hidden = torch.tanh(hidden)
        hidden = hidden / 2
        hidden = torch.sigmoid(hidden)
        hidden = hidden.t()
        return torch.cat([hidden, x], dim=0)


def _no_pass(label: str) -> str:
    head, sep, tail = label.rpartition(":")
    return head if sep and tail.isdigit() else label


def _trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def _first_interior_op(trace: tl.Trace) -> str:
    """Return the op label of the first saved single-pass op with parents."""

    for op in trace.saved_ops:
        single_pass = getattr(op.layer, "num_passes", 1) == 1
        if op.parents and op.dtype is not None and op.dtype.is_floating_point and single_pass:
            return op.label
    raise AssertionError("no single-pass interior op")


def _descendants(trace: tl.Trace, label: str) -> set[str]:
    seen: set[str] = set()
    frontier = [label]
    while frontier:
        current = frontier.pop()
        for child in trace[current].children or ():
            child_label = _no_pass(
                child if isinstance(child, str) else getattr(child, "label", str(child))
            )
            if child_label not in seen:
                seen.add(child_label)
                frontier.append(child_label)
    return seen


def _compare_fork(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, str, dict[str, float]]:
    trace = _trace(model, x)
    label = _first_interior_op(trace)
    fork = trace.fork()
    fork.do(tl.label(label), tl.zero_ablate())
    pair = tl.Bundle({"base": trace, "edited": fork}, baseline="base")
    rows = {
        _no_pass(name): value
        for name, value in pair.compare("base", "edited", metric="relative_l2")
    }
    return trace, _no_pass(label), rows


# --- F5: a recurrent member is not a cyclic supergraph -------------------------


def test_recurrent_member_alone_builds_an_ordered_supergraph() -> None:
    """Feedback edges stay in ``edges`` (drawing) but do not drive the order."""

    torch.manual_seed(0)
    trace = _trace(_Recurrent(), torch.randn(2, 4))
    graph = build_supergraph([trace], ["base"])

    assert set(graph.topological_order) == set(graph.nodes)
    adds = [name for name in graph.nodes if name.startswith("add_")]
    tanhs = [name for name in graph.nodes if name.startswith("tanh_")]
    assert len(adds) == 1 and len(tanhs) == 1
    position = {name: idx for idx, name in enumerate(graph.topological_order)}
    assert position[adds[0]] < position[tanhs[0]]
    assert (tanhs[0], adds[0]) in graph.edges  # the recurrence feedback is kept


def test_recurrent_fork_compares_with_the_edited_site_first() -> None:
    """A fork of a recurrent model merges and compares like any other fork."""

    torch.manual_seed(0)
    trace, site, rows = _compare_fork(_Recurrent(), torch.randn(2, 4))

    assert site in rows
    assert rows[site] == pytest.approx(1.0, abs=1e-5)
    assert all(math.isfinite(value) for value in rows.values())
    cone = _descendants(trace, site)
    outside = {name: value for name, value in rows.items() if name != site and name not in cone}
    assert all(value == 0.0 for value in outside.values()), outside


def test_contradictory_member_orders_refuse_typed() -> None:
    """Members that disagree on operation order refuse with a coded error."""

    edges = {("a", "b"): {"left"}, ("b", "a"): {"right"}}
    with pytest.raises(BundleComparisonError) as info:
        _preferential_topological_order({"a", "b", "c"}, edges, ["a", "b", "c"])

    exc = info.value
    assert isinstance(exc, ValueError)
    assert exc.fields["code"] == "bundle_gate_model_axis_unmet"
    assert exc.fields["operation"] == "supergraph"
    assert exc.fields["cyclic_edges"] == ["a->b (left)", "b->a (right)"]
    assert "Remedy" in str(exc)


# --- F6: identical ``-inf`` payloads read 0.0, not nan ------------------------


@pytest.mark.parametrize("metric", ["relative_l2", "cosine", "pearson"])
def test_identical_payload_with_inf_reads_zero(metric: str) -> None:
    """Equal tensors holding ``-inf`` are unchanged under every registered metric."""

    mask = torch.full((4, 4), float("-inf")).triu(1)
    assert _distance_value(mask, mask.clone(), metric) == 0.0
    with_nan = torch.tensor([1.0, float("nan"), 3.0])
    assert _distance_value(with_nan, with_nan.clone(), metric) == 0.0
    changed = _distance_value(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]), metric)
    assert math.isfinite(changed) and changed > 0.0


def test_shared_non_finite_entries_are_left_out_of_the_metric() -> None:
    """Masked positions both operands agree on do not turn the distance into nan."""

    reference = torch.tensor([1.0, float("-inf"), 3.0])
    candidate = torch.tensor([2.0, float("-inf"), 3.0])
    assert _distance_value(reference, candidate, "relative_l2") == pytest.approx(1 / math.sqrt(10))
    disagreeing = torch.tensor([1.0, 2.0, 3.0])
    assert math.isnan(_distance_value(reference, disagreeing, "relative_l2"))


def test_identical_payload_shortcut_leaves_user_callables_alone() -> None:
    """A user metric is evaluated as given, even on identical operands."""

    calls: list[int] = []

    def constant(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return torch.tensor(0.5)

    assert _distance_value(torch.ones(3), torch.ones(3), constant) == 0.5
    assert calls == [1]


def test_masked_fork_reports_no_distance_outside_the_cone() -> None:
    """The ``-inf`` mask sites outside the edit's cone read 0.0 in the fork."""

    torch.manual_seed(0)
    trace, site, rows = _compare_fork(_Masked(), torch.randn(3, 4))

    assert site in rows
    assert rows[site] == pytest.approx(1.0, abs=1e-5)
    mask_sites = [name for name in rows if name.startswith(("full_", "triu_"))]
    assert mask_sites, sorted(rows)
    assert all(math.isfinite(value) for value in rows.values()), rows
    cone = _descendants(trace, site)
    outside = {name: value for name, value in rows.items() if name != site and name not in cone}
    assert set(mask_sites) <= set(outside)
    assert all(value == 0.0 for value in outside.values()), outside
    assert all(rows[site] >= value for value in outside.values())


def test_compare_keeps_a_site_whose_label_is_a_substring_of_another() -> None:
    """``t_1_1`` is resolved exactly in the other member, not as a substring of ``cat_1_10``."""

    torch.manual_seed(0)
    trace, site, rows = _compare_fork(_SubstringLabels(), torch.randn(3, 4))

    assert site == "t_1_1"
    superstrings = [name for name in trace._layer_logs if name != site and site in name]
    assert superstrings == ["cat_1_10"], list(trace._layer_logs)
    assert site in rows, sorted(rows)
    assert rows[site] == pytest.approx(1.0, abs=1e-5)
    assert "cat_1_10" in rows


# --- F8: different shapes refuse naming the site ------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (torch.ones(2, 3), torch.ones(3, 2)),  # equal element counts, different shapes
        (torch.ones(2, 3), torch.ones(2, 4)),  # different element counts
    ],
)
def test_shape_mismatch_refuses_typed_naming_the_site(
    left: torch.Tensor, right: torch.Tensor
) -> None:
    """Shape-mismatched operands refuse with ``metric_shape_mismatch`` and the site."""

    with pytest.raises(InvalidArgumentError) as info:
        _distance_value(left, right, "cosine", site="view_1_1")

    exc = info.value
    assert exc.fields["code"] == "metric_shape_mismatch"
    assert exc.fields["site"] == "view_1_1"
    assert exc.fields["reference_shape"] == tuple(left.shape)
    assert exc.fields["candidate_shape"] == tuple(right.shape)
    assert "view_1_1" in str(exc)
    assert exc.fields["remedy"]


def test_scalar_like_pair_is_not_a_shape_mismatch() -> None:
    """A 0-d and a 1-element tensor still compare as scalars."""

    assert _distance_value(torch.tensor(3.0), torch.tensor([3.0]), "cosine") == 0.0
