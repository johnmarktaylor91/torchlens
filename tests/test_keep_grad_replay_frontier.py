"""``keep_grad`` under differentiable replay discloses that nothing upstream is reached.

``push(replay=ReplayOptions(differentiable=True))`` re-enters each edited site value as
a fresh leaf, so a graph-keeping edit passes the keep_grad check there yet no parameter
upstream of the site gets a gradient. keep_grad says so once per session with the code
``keep_grad_replay_frontier``; ``fork.do`` (which does reach upstream) stays silent, and
the live door the warning names does train the upstream parameters.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.intervention import _gradient_contract, keep_grad

_CODE = "keep_grad_replay_frontier"
_X = torch.randn(2, 4, generator=torch.Generator().manual_seed(5))


class _Net(nn.Module):
    """``lin2(lin1(x))``: the edit sits on ``lin1``'s output."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.lin1(x))


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_gradient_contract, "_WARNED_CODES", set())


def _ready(model: nn.Module) -> Any:
    return tl.trace(model, _X, capture=tl.options.CaptureOptions(intervention_ready=True))


def _codes(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(record.message.fields.get("code"))
        for record in caught
        if isinstance(record.message, TorchLensWarning)
    ]


def _push_differentiable(model: nn.Module) -> Any:
    replay = tl.options.ReplayOptions(
        hooks={tl.label("linear_1_1"): keep_grad(tl.scale(2.0))}, differentiable=True
    )
    return _ready(model).push(replay=replay)


def test_differentiable_replay_discloses_once_per_session() -> None:
    model = _Net()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replayed = _push_differentiable(model)
        _push_differentiable(model)
    assert _codes(caught).count(_CODE) == 1
    message = next(str(record.message) for record in caught if _CODE in _codes([record]))
    assert "linear_1_1" in message and "scale" in message
    # The disclosure is true: the output has a graph, yet lin1 gets no gradient.
    replayed[replayed.output_layers[0]].out.sum().backward()
    assert model.lin1.weight.grad is None


def test_fork_do_stays_silent_and_reaches_upstream() -> None:
    model = _Net()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork = _ready(model).fork()
        fork.do("linear_1_1", keep_grad(tl.scale(2.0)))
    assert _CODE not in _codes(caught)
    fork[fork.output_layers[0]].out.sum().backward()
    assert model.lin1.weight.grad is not None


def test_the_named_live_doors_train_upstream_parameters() -> None:
    model = _Net()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _push_differentiable(model)
    remedy = next(record.message.fields["remedy"] for record in caught if _CODE in _codes([record]))
    assert ".bind(model)(x)" in remedy and "tl.trace(model, x, intervene=" in remedy
    bound = tl.when(tl.module("lin1"), keep_grad(tl.scale(2.0))).bind(model)
    bound(_X).sum().backward()
    assert model.lin1.weight.grad is not None
    assert torch.count_nonzero(model.lin1.weight.grad) > 0
    model.zero_grad(set_to_none=True)
    trace = tl.trace(model, _X, intervene=tl.when(tl.module("lin1"), tl.scale(2.0)))
    trace[trace.output_layers[0]].out.sum().backward()
    assert model.lin1.weight.grad is not None
