"""Bundle site reads pair ops across a capture-time module edit.

A ``tl.module(...)`` edit is recorded as its own ``interventionreplacement`` op
between the module's last op and its consumers, so every later op is
renumbered. Bundle site reads used to join members by literal label and
silently dropped exactly the ops the edit changed: ``most_changed`` reported
only the model output as changed, and the module's last op (which correctly
keeps its pre-edit value) looked like evidence that the edit never applied.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens.intervention._topology.topology import compare_topology


class _EditedBlockNet(nn.Module):
    """Linear, ReLU, then a (Linear, ReLU) block ``mlp_6``, then a Linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.mlp_6 = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.out = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.mlp_6(torch.relu(self.inp(x))))


def _independent_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1 - F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


@pytest.fixture
def clean_and_ablated():
    # A trace holds its model only weakly, and the Bundle gate derives the
    # identity relationship these reads need from that live reference. Yield
    # so the model stays alive through the test; returning would leave it to
    # whichever garbage collection runs next.
    torch.manual_seed(0)
    model = _EditedBlockNet().eval()
    x = torch.randn(1, 8)
    with torch.no_grad():
        module_out = model.mlp_6(torch.relu(model.inp(x)))
        expected_ablated_output = model.out(torch.zeros_like(module_out))
    clean = tl.trace(model, x)
    ablated = tl.trace(model, x, intervene=tl.when(tl.module("mlp_6"), tl.zero_ablate()))
    yield clean, ablated, module_out, expected_ablated_output


def _head(trace):
    return [op for op in trace.layer_list if op.func_name == "linear"][-1]


def _edit_ops(trace):
    return [op for op in trace.layer_list if op.func_name == "intervention_replacement"]


def test_module_edit_is_recorded_as_its_own_op(clean_and_ablated) -> None:
    clean, ablated, module_out, expected_ablated_output = clean_and_ablated
    edits = _edit_ops(ablated)
    assert len(edits) == 1
    assert edits[0].is_edited is True
    assert torch.equal(edits[0].out, torch.zeros_like(module_out))
    assert len(edits[0].parents) == 1
    module_last_op = ablated[edits[0].parents[0]]
    # The module's last op keeps the value it computed, before the edit.
    assert torch.equal(module_last_op.out, module_out)
    assert torch.allclose(ablated["output_1"].out, expected_ablated_output)
    # The head is renumbered: a literal-label join cannot pair it.
    assert _head(clean).layer_label != _head(ablated).layer_label


def test_alignment_sees_through_the_edit_op(clean_and_ablated) -> None:
    clean, ablated, _module_out, _expected = clean_and_ablated
    diff = compare_topology(clean, ablated)
    assert (_head(clean).layer_label, _head(ablated).layer_label) in diff.matched
    assert diff.unmatched_a == []
    assert diff.unmatched_b == [_edit_ops(ablated)[0].layer_label]


def test_site_reads_score_the_renumbered_ops(clean_and_ablated) -> None:
    clean, ablated, _module_out, _expected = clean_and_ablated
    bundle = tl.Bundle({"clean": clean, "ablated": ablated}, baseline="clean")
    head_label = _head(clean).layer_label
    expected = _independent_cosine_distance(_head(clean).out, _head(ablated).out)
    assert expected > 0.01

    ranked = dict(bundle.rank_changes(k=len(clean.layer_list)))
    assert set(ranked) == {op.layer_label for op in clean.layer_list}
    assert ranked[head_label] == pytest.approx(expected, rel=1e-4, abs=1e-6)

    pair_rows = dict(bundle.diff_pair("clean", "ablated"))
    assert set(pair_rows) == {op.layer_label for op in clean.layer_list}
    assert pair_rows[head_label] == pytest.approx(expected, rel=1e-4, abs=1e-6)
