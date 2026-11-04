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
        selected = trace[tl.where(lambda op, expert_id=expert_id: op.expert == expert_id)]
        assert set(selected.labels) == {op.label for op in members}
    assert not hasattr(trace.ops[0], "__dict__")


def test_routing_table_matches_independent_gate_oracle() -> None:
    """Table values and token loads match an ordinary uncaptured gate call."""
    model = _ToyMoE()
    x = _input()
    expected_weights, expected_indices = model.chooser(x).softmax(-1).topk(2, dim=-1)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True))
    assert len(trace.routing_table) == 1
    row = trace.routing_table[0]
    assert row["module"] == "self"
    assert row["expert_indices"] == expected_indices.tolist()
    torch.testing.assert_close(torch.tensor(row["gate_weights"]), expected_weights)
    assert row["tokens_per_expert"] == [2, 2, 2, 2]
    assert row["load"] == [0.25] * 4
    assert row["token_shape"] == (4,)
    assert trace.modules["self"].routing_table == trace.routing_table
    assert trace.modules["chooser"].routing_table is None
    assert trace.modules["bank.0"].routing_table is None
    assert trace.validate_forward_pass(model(x))


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


def test_fused_experts_keep_table_without_single_expert_identity() -> None:
    """Batched expert computation reads the same saved routing outputs."""

    class Fused(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(4, 4)
            self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(4)])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weights, indices = self.gate(x).softmax(-1).topk(2, dim=-1)
            matrices = torch.stack([expert.weight for expert in self.experts])
            selected = matrices[indices]
            values = torch.einsum("tkoi,ti->tko", selected, x)
            return (values * weights.unsqueeze(-1)).sum(1)

    model = Fused()
    x = _input()
    expected_weights, expected_indices = model.gate(x).softmax(-1).topk(2, dim=-1)
    trace = tl.trace(model, x)
    (row,) = trace.routing_table
    assert row["expert_indices"] == expected_indices.tolist()
    torch.testing.assert_close(torch.tensor(row["gate_weights"]), expected_weights)
    fused = [op for op in trace.ops if op.func_name == "einsum"]
    assert fused and all(op.expert is None for op in fused)


@pytest.mark.parametrize("keepdim", [False, True])
def test_argmax_with_saved_probability_gather(keepdim: bool) -> None:
    """Top-one routing takes probabilities from the saved gather output."""

    class TopOne(_ToyMoE):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            probabilities = self.chooser(x).softmax(-1)
            selected = probabilities.argmax(dim=-1, keepdim=keepdim)
            indices = selected if keepdim else selected.unsqueeze(-1)
            weights = probabilities.gather(-1, indices)
            result = torch.zeros_like(x)
            for number, expert in enumerate(self.bank):
                token, slot = torch.where(indices == number)
                result.index_add_(0, token, expert(x[token]) * weights[token, slot, None])
            return result

    model = TopOne()
    trace = tl.trace(model, _input())
    (row,) = trace.routing_table
    assert row["expert_indices"] == [[0], [1], [2], [3]]
    assert row["tokens_per_expert"] == [1, 1, 1, 1]


def test_selected_logits_then_softmax() -> None:
    """Routing can normalize the selected logits instead of the whole bank."""

    class SelectedLogits(_ToyMoE):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits, indices = self.chooser(x).topk(2, dim=-1)
            weights = logits.softmax(-1)
            result = torch.zeros_like(x)
            for number, expert in enumerate(self.bank):
                token, slot = torch.where(indices == number)
                result.index_add_(0, token, expert(x[token]) * weights[token, slot, None])
            return result

    model = SelectedLogits()
    x = _input()
    logits, indices = model.chooser(x).topk(2, dim=-1)
    trace = tl.trace(model, x)
    (row,) = trace.routing_table
    assert row["expert_indices"] == indices.tolist()
    torch.testing.assert_close(torch.tensor(row["gate_weights"]), logits.softmax(-1))


def test_fused_parameter_bank_and_existing_unit_selection() -> None:
    """A stacked parameter bank needs no new name to select expert slices."""

    class FusedBank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Linear(4, 4)
            self.matrices = nn.Parameter(torch.randn(4, 4, 4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            weights, indices = self.router(x).softmax(-1).topk(2, dim=-1)
            values = torch.einsum("tkoi,ti->tko", self.matrices[indices], x)
            return (values * weights.unsqueeze(-1)).sum(1)

    model = FusedBank()
    trace = tl.trace(model, _input())
    (row,) = trace.routing_table
    fused = next(op for op in trace.ops if op.func_name == "einsum")
    assert fused.expert is None
    mask = (torch.tensor(row["expert_indices"]) == 1).unsqueeze(-1).expand(fused.shape)
    selected = trace[tl.units(fused.layer_label, mask)]
    assert selected.labels == (fused.label,)
    assert torch.equal(selected[0].mask, mask)


def test_capture_does_not_discover_expert_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-time features must not add MoE discovery to an ordinary capture."""
    import torchlens._data_classes._moe_structure as structure

    def unexpected(*args: object) -> None:
        raise AssertionError("routing discovery ran during capture")

    monkeypatch.setattr(structure, "_layouts", unexpected)
    model = _ToyMoE()
    trace = tl.trace(model, _input())
    assert len(trace.ops) > 0


def test_expert_writer_uses_reserved_coordinate_kind() -> None:
    """The interpretability writer classification fills its expert row kind."""
    from torchlens.mechinterp._residual import _writer_kind

    model = _ToyMoE()
    trace = tl.trace(model, _input())
    writer = next(op for op in trace.ops if "bank.2:1" in op.modules)
    assert _writer_kind(trace, writer) == "expert"
    assert writer.expert == 2
