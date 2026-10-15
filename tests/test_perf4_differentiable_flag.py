"""The replay run context's ``differentiable`` flag describes the replay that just ran.

A trace that came out of differentiable replay carries ``last_run["differentiable"]``
True. A later plain replay of it (or of a fork of it) must report False, and hooks
firing inside that plain replay must read False too, so ``keep_grad`` does not
disclose a replay frontier that does not exist. A differentiable replay of a trace
that came out of plain replay reports True.
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


def _ready() -> Any:
    return tl.trace(_Net(), _X, capture=tl.options.CaptureOptions(intervention_ready=True))


def _codes(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(record.message.fields.get("code"))
        for record in caught
        if isinstance(record.message, TorchLensWarning)
    ]


def _push(trace: Any, *, differentiable: bool) -> Any:
    replay = tl.options.ReplayOptions(
        hooks={tl.label("linear_1_1"): keep_grad(tl.scale(2.0))}, differentiable=differentiable
    )
    return trace.push(replay=replay)


def test_plain_replay_after_differentiable_replay_reports_false() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replayed = _push(_ready(), differentiable=True)
    assert replayed.last_run["differentiable"] is True
    assert "frontier" in replayed.last_run
    _gradient_contract._WARNED_CODES.clear()
    fork = replayed.fork()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fork.do("linear_1_1", keep_grad(tl.scale(2.0)))
    assert fork.last_run["differentiable"] is False
    assert "frontier" not in fork.last_run
    assert _CODE not in _codes(caught), "a plain replay has no frontier to disclose"
    # The source keeps its own record (the run context is rebound, never mutated).
    assert replayed.last_run["differentiable"] is True


def test_plain_push_on_the_replayed_trace_itself_reports_false() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replayed = _push(_ready(), differentiable=True)
    _gradient_contract._WARNED_CODES.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _push(replayed, differentiable=False)
    assert replayed.last_run["differentiable"] is False
    assert _CODE not in _codes(caught)


def test_differentiable_replay_after_plain_replay_reports_true() -> None:
    plain = _ready().fork()
    plain.do("linear_1_1", tl.scale(3.0))
    assert plain.last_run["differentiable"] is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replayed = _push(plain, differentiable=True)
    assert replayed.last_run["differentiable"] is True
    assert _codes(caught).count(_CODE) == 1, "the disclosure reads the same flag"
