"""TorchLens's direct tensor gradient-hook install matches ``torch.Tensor.register_hook``.

Capture installs its gradient hook on a plain tensor through the C class's
getsets (``_tensor_grad_hook.register_plain_tensor_hook``) instead of calling
``register_hook``, which reads shadowed properties during capture. These tests
pin torch's method body that the direct install repeats, and check that hooks
from both routes share one dict, fire in registration order, pass gradients on
and remove cleanly. Every expected value comes from torch itself, never from
another TorchLens result.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

import torchlens as tl
from torchlens.backends.torch._tensor_grad_hook import (
    add_plain_tensor_hook,
    register_plain_tensor_hook,
    register_tensor_grad_hook,
    takes_plain_hook_install,
)
from torchlens.options import CaptureOptions

# The statements of ``torch.Tensor.register_hook`` the direct install repeats, in order.
_PINNED_STATEMENTS = (
    "if has_torch_function_unary(self):",
    "if not self.requires_grad:",
    "if self._backward_hooks is None:",
    "self._backward_hooks = OrderedDict()",
    "if self.grad_fn is not None:",
    "self.grad_fn._register_hook_dict(self)",
    "handle = RemovableHandle(self._backward_hooks)",
    "self._backward_hooks[handle.id] = hook",
    "return handle",
)


def _register_hook_body() -> list[str]:
    """Return the stripped code lines of ``torch.Tensor.register_hook`` from torch's file."""

    lines = Path(inspect.getfile(torch._tensor)).read_text().splitlines()
    start = lines.index("    def register_hook(self, hook):")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("    def ") or line.startswith("    @"):
            break
        body.append(line.strip())
    return body


def _recording_hook(order: list[str], name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a hook that appends ``name`` to ``order`` and passes the gradient on."""

    def hook(grad: torch.Tensor) -> torch.Tensor:
        order.append(name)
        return grad

    return hook


def test_torch_register_hook_body_is_the_one_the_direct_install_repeats() -> None:
    """A torch change to ``register_hook`` fails here before the routes can drift."""

    body = _register_hook_body()
    positions = [body.index(statement) for statement in _PINNED_STATEMENTS]
    assert positions == sorted(positions)
    assert [line for line in body if line][-1] == "return handle"


def test_removable_handle_id_counter_is_the_one_the_handle_free_route_advances() -> None:
    """``RemovableHandle`` still keys hooks from its class counter, advanced by one per handle."""

    source = inspect.getsource(RemovableHandle.__init__)
    assert "self.id = RemovableHandle.next_id" in source
    assert "RemovableHandle.next_id += 1" in source


def test_plain_install_refuses_like_torch() -> None:
    """A tensor outside autograd gets torch's own error from both routes."""

    with pytest.raises(RuntimeError) as torch_error:
        torch.zeros(3).register_hook(lambda grad: grad)
    with pytest.raises(RuntimeError) as plain_error:
        register_plain_tensor_hook(torch.zeros(3), lambda grad: grad)
    with pytest.raises(RuntimeError) as add_error:
        add_plain_tensor_hook(torch.zeros(3), lambda grad: grad)
    assert str(plain_error.value) == str(torch_error.value) == str(add_error.value)


@pytest.mark.parametrize("plain_first", [True, False])
def test_both_routes_share_one_dict_in_registration_order(plain_first: bool) -> None:
    """Hooks from either route fire in registration order and remove cleanly."""

    x = torch.randn(4, requires_grad=True)
    y = x * 2
    order: list[str] = []
    routes = ["plain", "torch", "plain", "torch"] if plain_first else ["torch", "plain"] * 2
    handles = []
    for index, route in enumerate(routes):
        hook = _recording_hook(order, f"{route}{index}")
        handles.append(
            register_plain_tensor_hook(y, hook) if route == "plain" else y.register_hook(hook)
        )
    ids = [handle.id for handle in handles]
    assert ids == sorted(ids)
    assert list(y._backward_hooks) == ids
    handles[1].remove()
    y.sum().backward()
    assert order == [f"{route}{index}" for index, route in enumerate(routes) if index != 1]
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def _register_by_route(y: torch.Tensor, route: str, hook: Callable[..., object]) -> int | None:
    """Register ``hook`` on ``y`` by ``route``; return the handle id, or None for ``add``."""

    if route == "add":
        assert add_plain_tensor_hook(y, hook)
        return None
    handle = register_plain_tensor_hook(y, hook) if route == "plain" else y.register_hook(hook)
    return handle.id


@pytest.mark.parametrize(
    "routes",
    [
        ("plain", "torch", "add", "torch", "add"),
        ("torch", "add", "plain", "add", "torch"),
        ("add", "add", "torch", "plain", "torch"),
    ],
)
def test_handle_free_route_shares_the_dict_in_registration_order(routes: tuple[str, ...]) -> None:
    """Hooks from every route take increasing keys in one dict and fire in registration order."""

    x = torch.randn(4, requires_grad=True)
    y = x * 2
    order: list[str] = []
    ids = [
        _register_by_route(y, route, _recording_hook(order, f"{route}{index}"))
        for index, route in enumerate(routes)
    ]
    keys = list(y._backward_hooks)
    assert keys == sorted(keys)
    assert len(keys) == len(routes)
    assert all(
        key == hook_id for key, hook_id in zip(keys, ids, strict=True) if hook_id is not None
    )
    y.sum().backward()
    assert order == [f"{route}{index}" for index, route in enumerate(routes)]
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_a_handle_removes_only_its_own_hook_beside_handle_free_hooks() -> None:
    """A torch handle's ``remove()`` leaves the hooks the handle-free route added."""

    x = torch.randn(4, requires_grad=True)
    y = x * 2
    order: list[str] = []
    assert add_plain_tensor_hook(y, _recording_hook(order, "add0"))
    handle = y.register_hook(_recording_hook(order, "torch1"))
    assert add_plain_tensor_hook(y, _recording_hook(order, "add2"))
    handle.remove()
    y.sum().backward()
    assert order == ["add0", "add2"]


@pytest.mark.parametrize("leaf", [True, False])
def test_plain_install_passes_its_return_on(leaf: bool) -> None:
    """The hook's return replaces the gradient, on a leaf and on an op output, in order."""

    x = torch.randn(4, requires_grad=True)
    hooked = x if leaf else x * 1
    register_plain_tensor_hook(hooked, lambda grad: grad * 3)
    assert add_plain_tensor_hook(hooked, lambda grad: grad * 5)
    (hooked * x.detach()).sum().backward()
    assert torch.equal(x.grad, x.detach() * 3 * 5)


class _Sub(torch.Tensor):
    """A tensor subclass: it keeps torch's ``register_hook`` call."""


class _PassThroughMode(TorchFunctionMode):
    """An active torch function mode that runs every function unchanged."""

    def __torch_function__(
        self,
        func: Callable[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        return func(*args, **(kwargs or {}))


def test_subclass_and_function_mode_keep_torchs_register_hook() -> None:
    """Only a plain tensor with no ``__torch_function__`` override takes the direct install."""

    x = torch.randn(4, requires_grad=True)
    sub = (x * 2).as_subclass(_Sub)
    assert takes_plain_hook_install(x * 2)
    assert not takes_plain_hook_install(sub)
    assert not add_plain_tensor_hook(sub, lambda grad: grad)
    assert sub._backward_hooks is None
    with _PassThroughMode():
        assert not takes_plain_hook_install(x * 2)
        assert not add_plain_tensor_hook(x * 2, lambda grad: grad)
    order: list[str] = []

    def no_slot_prehook(*_args: object) -> object:
        raise AssertionError("a tensor site never takes the node pre-hook")

    register_tensor_grad_hook(sub, _recording_hook(order, "sub"), no_slot_prehook)
    sub.sum().backward()
    assert order == ["sub"]
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


class _LinearTanh(nn.Module):
    """Linear followed by tanh, so the linear output's gradient is not constant."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


def _double(grad: torch.Tensor) -> torch.Tensor:
    """Double a gradient."""

    return grad * 2


def _hook_output_with(hook: Callable[[torch.Tensor], torch.Tensor]) -> Callable[..., None]:
    """Return a forward hook that registers ``hook`` on the output and returns None."""

    def forward_hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        output.register_hook(hook)

    return forward_hook


def test_capture_hook_fires_before_a_user_hook_registered_later_in_forward() -> None:
    """TorchLens's hook, registered at the op, sees the gradient before the user's hook edits it."""

    torch.manual_seed(0)
    model = _LinearTanh()
    x = torch.randn(3, 4)
    held: dict[str, torch.Tensor] = {}

    def retain(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        output.retain_grad()
        held["out"] = output

    handle = model.lin.register_forward_hook(retain)
    model(x).sum().backward()
    handle.remove()
    raw_grad = held["out"].grad
    assert raw_grad is not None
    model.zero_grad()
    eager_weight_grad = _weight_grad_with_doubling_hook(model, x)
    calls: list[int] = []

    def double_and_count(grad: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return grad * 2

    user_handle = model.lin.register_forward_hook(_hook_output_with(double_and_count))
    trace = tl.trace(model, x, capture=CaptureOptions(save_grads=True))
    user_handle.remove()
    model.zero_grad()
    trace.backward(trace[trace.output_layers[0]].out.sum())
    sites = list(trace.find_sites(tl.module("lin")))
    assert len(sites) == 1
    assert calls == [1]
    assert torch.equal(sites[0].grad, raw_grad)
    assert torch.equal(model.lin.weight.grad, eager_weight_grad)
    trace.cleanup()


def _weight_grad_with_doubling_hook(model: _LinearTanh, x: torch.Tensor) -> torch.Tensor:
    """Return the eager weight gradient with the linear output's gradient doubled."""

    handle = model.lin.register_forward_hook(_hook_output_with(_double))
    model(x).sum().backward()
    handle.remove()
    weight_grad = model.lin.weight.grad
    assert weight_grad is not None
    return weight_grad.clone()
