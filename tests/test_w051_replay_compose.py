"""Tier-(ii) edits COMPOSE and never silently revert (AUD-CODE 2.9b / 4.10).

Before the fix EDGE-kind tier-(ii) entries were excluded from the cone
re-splice, so any later recompute of the child (a second edge edit on
another arg, an upstream act edit, a param edit reaching the child)
silently reverted the substitution while the store, stamp, and audit row
kept asserting it. Chained ``do(tl.params(p), scale(0.5))`` REPLACED
(0.5) instead of composing (0.25) as the act path does.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens as tl

ATOL = 1e-6


class _Two(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.fc1(x))
        b = torch.tanh(self.fc2(x))
        return torch.add(a, b) * 2.0


@pytest.fixture(scope="module")
def two():
    torch.manual_seed(0)
    model = _Two().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True),
    )
    try:
        yield model, x, trace
    finally:
        trace.cleanup()


def _parts(model, x):
    with torch.no_grad():
        return torch.relu(model.fc1(x)), torch.tanh(model.fc2(x))


def _add(trace):
    return next(op for op in trace.layer_list if op.func_name == "add")


def _edge(trace, position):
    add = _add(trace)
    return next(
        e
        for e in trace.edges
        if e.child_label in (add.label, add.layer_label) and e.arg_path == (position,)
    )


def _validate(fork, truth):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fork.validate_forward_pass(truth)


def test_two_edge_edits_on_one_child_both_hold(two) -> None:
    model, x, trace = two
    fork = trace.fork()
    fork.do(_edge(trace, 0).__selection__(), tl.zero_ablate())
    fork.do(_edge(trace, 1).__selection__(), tl.zero_ablate())
    add = fork[_add(trace).label]
    assert bool((add.out == 0).all())
    assert len(add.ops[0].edge_substitutions) == 2
    assert _validate(fork, torch.zeros(2, 4)) is True


def test_edge_edit_survives_upstream_act_edit(two) -> None:
    model, x, trace = two
    _a, b = _parts(model, x)
    fork = trace.fork()
    fork.do(_edge(trace, 0).__selection__(), tl.zero_ablate())
    assert torch.allclose(fork[_add(trace).label].out, b, atol=ATOL)
    fork.do(tl.label("tanh_1_4"), tl.scale(0.5))
    assert torch.allclose(fork[_add(trace).label].out, 0.5 * b, atol=ATOL)
    assert _validate(fork, b) is True


@pytest.mark.parametrize("param_first", [False, True])
def test_edge_edit_and_param_edit_compose_in_either_order(two, param_first: bool) -> None:
    model, x, trace = two
    fork = trace.fork()
    with torch.no_grad():
        b_no_bias = torch.tanh(x @ model.fc2.weight.T)
    steps = [
        lambda: fork.do(_edge(trace, 0).__selection__(), tl.zero_ablate()),
        lambda: fork.do(tl.params("fc2.bias"), tl.zero_ablate()),
    ]
    for step in steps[::-1] if param_first else steps:
        step()
    assert torch.allclose(fork[_add(trace).label].out, b_no_bias, atol=ATOL)
    assert _validate(fork, 2.0 * b_no_bias) is True


def test_same_edge_edited_twice_composes_and_keeps_both_fires(two) -> None:
    model, x, trace = two
    a, b = _parts(model, x)
    fork = trace.fork()
    edge = _edge(trace, 0)
    fork.do(edge.__selection__(), tl.scale(0.5))
    fork.do(edge.__selection__(), tl.scale(0.5))
    add = fork[_add(trace).label].ops[0]
    assert torch.allclose(add.out, 0.25 * a + b, atol=ATOL)
    fires = [r for r in add.interventions if r.edge_address is not None]
    assert len(fires) == 2
    store_value = add.edge_substitutions[(edge.arg_kind, edge.arg_path)]["value"]
    assert torch.allclose(store_value, 0.25 * a, atol=ATOL)
    assert _validate(fork, 2.0 * (0.25 * a + b)) is True


def test_chained_param_scaling_composes_and_never_writes_the_live_param(two) -> None:
    model, x, trace = two
    live_before = model.fc2.weight.detach().clone()
    fork = trace.fork()
    fork.do(tl.params("fc2.weight"), tl.scale(0.5))
    fork.do(tl.params("fc2.weight"), tl.scale(0.5))
    with torch.no_grad():
        a = torch.relu(model.fc1(x))
        expected_add = a + torch.tanh(x @ (0.25 * model.fc2.weight).T + model.fc2.bias)
    assert torch.allclose(fork[_add(trace).label].out, expected_add, atol=ATOL)
    assert torch.equal(model.fc2.weight.detach(), live_before)
    consumer = fork["linear_2_3"].ops[0]
    assert len([r for r in consumer.interventions if r.edge_address is not None]) == 2
    assert _validate(fork, 2.0 * expected_add) is True
