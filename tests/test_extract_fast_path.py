"""``tl.extract`` takes a hooked plain forward only when it returns what a full capture returns.

Every case compares the fast result with the full capture's result
(``_extract_layers_with_trace``) bit for bit, and checks which path ran.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as _tl_pkg
import torchlens as tl
from torchlens.errors import InvalidArgumentError


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x)) + x


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.shared = nn.Linear(6, 6)
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(torch.softmax(x, dim=-1))


class _Twice(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.fc(x) + torch.rand_like(x))


class _Pair(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 2, x + 1


class _PairNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pair = _Pair()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.pair(x)
        return a - b


def _full(model: nn.Module, x: Any, layers: Any) -> dict[str, torch.Tensor]:
    return _tl_pkg._extract_layers_with_trace(model, x, layers)[1]


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    ran: list[int] = []
    real = _tl_pkg._extract_layers_with_trace

    def counting(*args: Any, **kwargs: Any) -> Any:
        ran.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_tl_pkg, "_extract_layers_with_trace", counting)
    return ran


def _same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key
        assert not a[key].requires_grad


@pytest.mark.parametrize(
    "layers",
    [
        ["blocks.0", "blocks.1.fc", "head"],
        {"first": "blocks.0", "last": "head"},
    ],
)
def test_module_outputs_take_the_fast_path_and_match_the_capture(
    monkeypatch: pytest.MonkeyPatch, layers: Any
) -> None:
    torch.manual_seed(0)
    model, x = _Net().eval(), torch.randn(2, 6)
    expected = _full(model, x, layers)
    ran = _spy(monkeypatch)
    got = tl.extract(model, x, layers)
    assert ran == []
    _same(got, expected)


@pytest.mark.parametrize(
    "case",
    ["called_twice", "never_called", "tuple_output", "train_mode", "kwargs_input"],
)
def test_ineligible_requests_run_the_full_capture(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6)
    model: nn.Module
    if case == "called_twice":
        model, layers = _Twice().eval(), ["fc"]
    elif case == "never_called":
        model, layers = _Net().eval(), ["shared", "head"]
    elif case == "tuple_output":
        model, layers = _PairNet().eval(), ["pair"]
    elif case == "train_mode":
        model, layers = _Net().train(), ["blocks.0"]
    else:
        model, layers = _Net().eval(), ["blocks.0"]
    inputs: Any = [x] if case == "kwargs_input" else x
    ran = _spy(monkeypatch)
    if case == "never_called":
        with pytest.raises(InvalidArgumentError):
            tl.extract(model, inputs, layers)
        assert ran == [1]
        return
    torch.manual_seed(1)
    got = tl.extract(model, inputs, layers)
    assert ran == [1]
    torch.manual_seed(1)
    _same(got, _full(model, inputs, layers))


def test_a_fallback_restores_the_random_state() -> None:
    torch.manual_seed(0)
    model, x = _Twice().eval(), torch.randn(2, 6)
    torch.manual_seed(5)
    got = tl.extract(model, x, ["fc"])
    torch.manual_seed(5)
    _same(got, _full(model, x, ["fc"]))


def test_unknown_names_keep_the_capture_refusal() -> None:
    model, x = _Net().eval(), torch.randn(2, 6)
    with pytest.raises(InvalidArgumentError):
        tl.extract(model, x, ["blocks.0", "no_such_module"])


def test_op_labels_and_attached_outs_use_the_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    model, x = _Net().eval(), torch.randn(2, 6)
    label = next(op.label for op in tl.trace(model, x).layers if op.func_name == "softmax")
    ran = _spy(monkeypatch)
    tl.extract(model, x, [label])
    attached = tl.extract(model, x, ["blocks.0"], detach_saved_activations=False)
    assert ran == [1, 1]
    assert attached["blocks.0"].requires_grad
