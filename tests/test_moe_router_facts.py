"""Structural routing identity and saved-output routing-table regressions."""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import PayloadUnavailableError


class _ToyMoE(nn.Module):
    """Four experts with a top-two softmax gate and token scatter."""

    def __init__(self, *, dictionary: bool = False) -> None:
        super().__init__()
        self.chooser = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.chooser.weight.copy_(torch.eye(4))
        experts = [nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 4)) for _ in range(4)]
        self.bank = (
            nn.ModuleDict(dict(zip(("z", "a", "q", "b"), experts)))
            if dictionary
            else nn.ModuleList(experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.chooser(x).softmax(dim=-1).topk(2, dim=-1)
        result = torch.zeros_like(x)
        experts = self.bank.values() if isinstance(self.bank, nn.ModuleDict) else self.bank
        for expert_id, expert in enumerate(experts):
            token, slot = torch.where(indices == expert_id)
            values = expert(x[token]) * weights[token, slot].unsqueeze(-1)
            result.index_add_(0, token, values)
        return result


def _input() -> torch.Tensor:
    """Route every expert twice without random tie breaking."""
    return torch.tensor(
        [[4.0, 3.0, 1.0, 0.0], [0.0, 4.0, 3.0, 1.0], [1.0, 0.0, 4.0, 3.0], [3.0, 1.0, 0.0, 4.0]]
    )


@pytest.mark.parametrize("dictionary", [False, True])
def test_router_identity_and_expert_selection(dictionary: bool) -> None:
    """Identity comes from dataflow and container order, never module names."""
    model = _ToyMoE(dictionary=dictionary)
    trace = tl.trace(model, _input())
    routers = [op for op in trace.ops if op.is_router]
    assert {"linear", "softmax", "topk"} <= {op.func_name for op in routers}
    assert all(op.expert is None for op in routers)
    keys = ["z", "a", "q", "b"] if dictionary else list(map(str, range(4)))
    for expert_id, key in enumerate(keys):
        members = [op for op in trace.ops if f"bank.{key}:1" in op.modules]
        assert members and all(op.expert == expert_id for op in members)
        selected = trace[tl.where(lambda op: op.expert == expert_id)]
        assert set(selected.labels) == {op.layer_label for op in members}
    assert not hasattr(trace.ops[0], "__dict__")


def test_routing_table_matches_independent_gate_oracle() -> None:
    """Table values and token loads match an ordinary uncaptured gate call."""
    model = _ToyMoE()
    x = _input()
    expected_weights, expected_indices = model.chooser(x).softmax(-1).topk(2, dim=-1)
    trace = tl.trace(model, x)
    assert len(trace.routing_table) == 1
    row = trace.routing_table[0]
    assert row["expert_indices"] == expected_indices.tolist()
    torch.testing.assert_close(torch.tensor(row["gate_weights"]), expected_weights)
    assert row["tokens_per_expert"] == [2, 2, 2, 2]
    assert row["load"] == [0.25] * 4
    assert row["token_shape"] == (4,)
    assert trace.modules["self"].routing_table == trace.routing_table
    assert trace.modules["chooser"].routing_table is None
    assert trace.modules["bank.0"].routing_table is None
    assert trace.validate_forward_pass()


def test_unsaved_routing_outputs_refuse_but_identity_remains() -> None:
    """Metadata-only captures preserve identity and refuse missing numbers."""
    model = _ToyMoE()
    trace = tl.trace(model, _input(), save="none")
    assert any(op.is_router for op in trace.ops)
    assert {op.expert for op in trace.ops} == {None, 0, 1, 2, 3}
    with pytest.raises(PayloadUnavailableError) as caught:
        _ = trace.routing_table
    assert caught.value.fields["code"] == "moe_routing_activation_missing"
    assert caught.value.fields["activation"] in trace.ops
    assert "save='all'" in str(caught.value)


def test_non_moe_has_no_identity_or_table() -> None:
    """Repeated modules alone are insufficient to infer routing."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    trace = tl.trace(model, _input())
    assert all(not op.is_router and op.expert is None for op in trace.ops)
    assert trace.routing_table == []
    assert all(module.routing_table is None for module in trace.modules)


def test_unrelated_topk_does_not_turn_ensemble_into_moe() -> None:
    """A gate-shaped side calculation does not prove expert indexing."""

    class Ensemble(_ToyMoE):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            auxiliary = self.chooser(x).softmax(-1).topk(2, dim=-1).values.sum()
            return sum(expert(x) for expert in self.bank) + auxiliary

    model = Ensemble()
    trace = tl.trace(model, _input())
    assert all(not op.is_router and op.expert is None for op in trace.ops)
    assert trace.routing_table == []


def test_reused_block_has_one_routing_row_per_call() -> None:
    """Repeated invocation preserves distinct token selections."""

    class Reused(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = _ToyMoE()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x) + self.block(-x)

    model = Reused()
    trace = tl.trace(model, _input())
    rows = trace.routing_table
    assert len(rows) == 2
    assert rows[0]["expert_indices"] != rows[1]["expert_indices"]
    assert trace.modules["block"].routing_table == rows
