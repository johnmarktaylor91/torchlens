"""``tl.pluck`` takes a hooked plain forward when it returns what the capture returns.

An exact module address on an eval-mode model with one tensor (or text) input
runs one forward with a hook on that module; an exact module address as
``stop_after`` ends that forward from a hook after the module's first call.
Every case compares the result with the traced pluck bit for bit and checks
which path ran; every other request keeps the traced path.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._extraction import _front_door
from torchlens.errors import InvalidArgumentError, TorchLensWarning

#: Module names whose forward ran, in order (module level: a capture restores model attributes).
_RAN: list[str] = []


class _Tracked(nn.Module):
    """A linear layer that records each forward in ``_RAN``."""

    def __init__(self, name: str, width: int) -> None:
        super().__init__()
        self.tag = name
        self.fc = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        _RAN.append(self.tag)
        return torch.tanh(self.fc(x))


class _Chain(nn.Module):
    """Three tracked stages, a ReLU module and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.Sequential(_Tracked("a", 6), _Tracked("b", 6), _Tracked("c", 6))
        self.relu = nn.ReLU()
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.relu(self.stages(x)))


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.fc(self.pool(torch.relu(self.bn(self.conv(x)))).flatten(1))


class _CharTokenizer:
    def __call__(self, text: str, return_tensors: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 16 for c in text]]))


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.block = nn.Sequential(nn.Linear(8, 8), nn.GELU())
        self.head = nn.Linear(8, 16)
        self.tokenizer = _CharTokenizer()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(self.block(self.emb(ids)))


def _model_and_input(name: str) -> tuple[nn.Module, Any]:
    torch.manual_seed(0)
    if name == "chain":
        return _Chain().eval(), torch.randn(2, 6)
    if name == "conv":
        model = _ConvNet().eval()
        with torch.no_grad():
            model.bn.running_mean.uniform_(-0.5, 0.5)
            model.bn.running_var.uniform_(0.5, 1.5)
        return model, torch.randn(2, 3, 8, 8)
    return _TinyLM().eval(), "hello there"


def _traced(monkeypatch: pytest.MonkeyPatch, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Today's pluck: the fast path switched off for one call."""

    with monkeypatch.context() as patch:
        patch.setattr(_front_door, "try_module_output_pluck", lambda *_a, **_k: None)
        return tl.pluck(*args, **kwargs)


def _spy(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    ran: list[int] = []
    real = _front_door._capture_trace

    def counting(*args: Any, **kwargs: Any) -> Any:
        ran.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_front_door, "_capture_trace", counting)
    return ran


def _same(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(got, want)
    assert got.requires_grad == want.requires_grad


def _no_hooks(model: nn.Module) -> bool:
    return all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize(
    ("name", "layer", "stop_after"),
    [
        ("chain", "stages.1", None),
        ("chain", "stages.1.fc", None),
        ("chain", "head", None),
        ("chain", "stages.0", "stages.1"),
        ("chain", "stages.1", "stages.1"),
        ("chain", "stages.1.fc", "stages"),
        ("conv", "bn", None),
        ("conv", "conv", "bn"),
        ("lm", "block", None),
        ("lm", "emb", "block.0"),
    ],
)
def test_module_address_takes_the_fast_path_and_matches_the_capture(
    monkeypatch: pytest.MonkeyPatch, name: str, layer: str, stop_after: str | None
) -> None:
    model, x = _model_and_input(name)
    want = _traced(monkeypatch, model, x, layer, stop_after=stop_after)
    ran = _spy(monkeypatch)
    got = tl.pluck(model, x, layer, stop_after=stop_after)
    assert ran == [], "the eligible request ran a capture"
    _same(got, want)
    assert _no_hooks(model)


@pytest.mark.parametrize("grad", [True, False])
def test_fast_pluck_keeps_the_capture_grad_state(
    monkeypatch: pytest.MonkeyPatch, grad: bool
) -> None:
    model, x = _model_and_input("chain")
    with torch.set_grad_enabled(grad):
        want = _traced(monkeypatch, model, x, "stages.1")
        got = tl.pluck(model, x, "stages.1")
    _same(got, want)


def test_stop_after_ends_the_forward_early() -> None:
    model, x = _model_and_input("chain")
    _RAN.clear()
    tl.pluck(model, x, "stages.0", stop_after="stages.1")
    assert _RAN == ["a", "b"], "the forward ran past the stop module"
    assert _no_hooks(model)
    _RAN.clear()
    tl.pluck(model, x, "stages.0")
    assert _RAN == ["a", "b", "c"]


def test_fast_pluck_installs_no_torch_wrappers() -> None:
    model, x = _model_and_input("chain")
    before = (torch.tanh, torch.Tensor.__add__, torch.nn.functional.linear)
    tl.pluck(model, x, "stages.0", stop_after="stages.1")
    assert (torch.tanh, torch.Tensor.__add__, torch.nn.functional.linear) == before


@pytest.mark.parametrize(
    "case",
    [
        "train_mode",
        "op_label",
        "function_named_stop",
        "selector_stop",
        "callable_stop",
        "list_input",
    ],
)
def test_ineligible_requests_keep_the_traced_path(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    model, x = _model_and_input("chain")
    layer: str = "stages.0"
    stop_after: Any = None
    inputs: Any = x
    if case == "train_mode":
        model.train()
    elif case == "op_label":
        layer = next(op.label for op in tl.trace(model, x).layers if op.func_name == "tanh")
    elif case == "function_named_stop":
        stop_after = "relu"  # also a function name: the capture may halt on F.relu first
    elif case == "selector_stop":
        stop_after = tl.module("stages.1")
    elif case == "callable_stop":
        stop_after = lambda ctx: False  # noqa: E731
    else:
        inputs = [x]
    # A callable that never fires: the capture warns, on both paths alike.
    warns = case == "callable_stop"
    with pytest.warns(TorchLensWarning, match="never fired") if warns else nullcontext():
        torch.manual_seed(1)
        want = _traced(monkeypatch, model, inputs, layer, stop_after=stop_after)
    ran = _spy(monkeypatch)
    with pytest.warns(TorchLensWarning, match="never fired") if warns else nullcontext():
        torch.manual_seed(1)
        got = tl.pluck(model, inputs, layer, stop_after=stop_after)
    assert ran == [1]
    _same(got, want)


def test_ambient_stop_site_keeps_the_traced_path(monkeypatch: pytest.MonkeyPatch) -> None:
    model, x = _model_and_input("chain")
    ran = _spy(monkeypatch)
    with tl.experimental.stop_after("stages.1"):
        got = tl.pluck(model, x, "stages.0")
    assert ran == [1]
    with tl.experimental.stop_after("stages.1"):
        want = _traced(monkeypatch, model, x, "stages.0")
    _same(got, want)


@pytest.mark.parametrize(
    ("layer", "stop_after"),
    [("stages.2", "stages.0"), ("stages.0", "no_such")],
)
def test_unservable_stop_requests_fall_back_to_the_capture_refusal(
    monkeypatch: pytest.MonkeyPatch, layer: str, stop_after: str
) -> None:
    """A stop before the layer, or a stop that is not a module, gets the capture's own answer."""

    model, x = _model_and_input("chain")
    with pytest.raises(Exception) as traced_error:  # noqa: PT011 -- compared below
        _traced(monkeypatch, model, x, layer, stop_after=stop_after)
    ran = _spy(monkeypatch)
    with pytest.raises(type(traced_error.value)) as fast_error:
        tl.pluck(model, x, layer, stop_after=stop_after)
    assert ran == [1]
    assert str(fast_error.value) == str(traced_error.value)
    assert _no_hooks(model)


def test_stop_module_that_never_runs_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Skips(_Chain):
        def __init__(self) -> None:
            super().__init__()
            self.unused = nn.Linear(6, 6)

    torch.manual_seed(0)
    model, x = _Skips().eval(), torch.randn(2, 6)
    ran = _spy(monkeypatch)
    with pytest.raises(InvalidArgumentError):
        tl.pluck(model, x, "stages.0", stop_after="unused")
    assert ran == [1]
    assert _no_hooks(model)


def test_a_fallback_restores_the_random_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial hooked forward that falls back leaves the random state as the capture alone."""

    class _Noisy(_Chain):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
            return super().forward(x + torch.rand_like(x))

    torch.manual_seed(0)
    model, x = _Noisy().eval(), torch.randn(2, 6)
    torch.manual_seed(5)
    with pytest.raises(Exception):  # noqa: B017,PT011 -- stop fires before the layer
        tl.pluck(model, x, "stages.2", stop_after="stages.0")
    after = torch.rand(4)
    torch.manual_seed(5)
    with pytest.raises(Exception):  # noqa: B017,PT011
        _traced(monkeypatch, model, x, "stages.2", stop_after="stages.0")
    assert torch.equal(after, torch.rand(4))
