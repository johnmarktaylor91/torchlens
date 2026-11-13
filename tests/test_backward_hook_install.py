"""TorchLens's direct tensor gradient-hook install matches ``torch.Tensor.register_hook``.

Capture installs its gradient hook on a plain tensor through the C class's
getsets (``_torch_compat.get_direct_tensor_hook_install``, behind
``HAS_DIRECT_TENSOR_HOOK_INSTALL``) instead of calling ``register_hook``, which
reads shadowed properties during capture. When the flag is false capture keeps
``register_hook`` with real handles. These tests run on both routes (the
``route`` fixture forces the flag off for ``fallback``), check torch's method body
the direct install repeats, and check that hooks share one dict, fire in
registration order, pass gradients on and remove cleanly. Every expected value
comes from torch itself, never from another TorchLens result.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterator

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

import torchlens as tl
from torchlens._utils import _torch_compat as tc
from torchlens._utils._tensor_hook_install import REGISTER_HOOK_BODY, register_hook_body_matches
from torchlens.backends.torch._tensor_grad_hook import (
    add_plain_tensor_hook,
    register_plain_tensor_hook,
    register_tensor_grad_hook,
    takes_plain_hook_install,
)
from torchlens.options import CaptureOptions


@pytest.fixture(params=["direct", "fallback"])
def route(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Run a test on the direct install, then with ``HAS_DIRECT_TENSOR_HOOK_INSTALL`` forced off."""

    if request.param == "direct":
        if tc.get_direct_tensor_hook_install() is None:
            pytest.skip("this torch does not admit the direct install (fallback covers it)")
    else:
        tc.get_direct_tensor_hook_install()
        monkeypatch.setattr(tc, "HAS_DIRECT_TENSOR_HOOK_INSTALL", False)
    yield request.param


def _no_slot_prehook(*_args: object) -> object:
    raise AssertionError("a tensor site never takes the node pre-hook")


def _capture_add(y: torch.Tensor, hook: Callable[..., object], route: str) -> None:
    """Add ``hook`` as capture does: handle-free when admitted, else ``register_hook``."""

    added = add_plain_tensor_hook(y, hook)
    assert added == (route == "direct")
    if not added:
        register_tensor_grad_hook(y, hook, _no_slot_prehook)


def _recording_hook(order: list[str], name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a hook that appends ``name`` to ``order`` and passes the gradient on."""

    def hook(grad: torch.Tensor) -> torch.Tensor:
        order.append(name)
        return grad

    return hook


def test_flag_follows_torchs_register_hook_body() -> None:
    """Where torch's ``register_hook`` body differs, the flag is false and capture falls back."""

    matches = register_hook_body_matches(torch.Tensor.register_hook)
    direct = tc.get_direct_tensor_hook_install(force_probe=True)
    assert tc.get_torch_capability_snapshot()["HAS_DIRECT_TENSOR_HOOK_INSTALL"] == (
        direct is not None
    )
    if not matches:
        assert direct is None
        assert not tc.HAS_DIRECT_TENSOR_HOOK_INSTALL
        x = torch.randn(2, requires_grad=True) * 2
        assert not takes_plain_hook_install(x)
        assert not add_plain_tensor_hook(x, lambda grad: grad)
        return
    assert direct is not None
    assert tc.HAS_DIRECT_TENSOR_HOOK_INSTALL


def _same_steps_older_spelling(self, hook):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
    """Docstrings, imports, line breaks and ``hooks.RemovableHandle`` do not count."""
    from torch.utils import hooks

    if has_torch_function_unary(self):  # noqa: F821
        return handle_torch_function(Tensor.register_hook, (self,), self, hook)  # noqa: F821
    if not self.requires_grad:
        raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
    if self._backward_hooks is None:
        self._backward_hooks = OrderedDict()
        if self.grad_fn is not None:
            self.grad_fn._register_hook_dict(self)
    handle = hooks.RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle


def _one_step_moved(self, hook):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
    if has_torch_function_unary(self):  # noqa: F821
        return handle_torch_function(Tensor.register_hook, (self,), self, hook)  # noqa: F821
    if not self.requires_grad:
        raise RuntimeError("cannot register a hook on a tensor that doesn't require gradient")
    if self._backward_hooks is None:
        self._backward_hooks = OrderedDict()
    if self.grad_fn is not None:
        self.grad_fn._register_hook_dict(self)
    handle = RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle


def test_body_check_ignores_spelling_and_catches_a_moved_step() -> None:
    """The comparison is by statement: torch 2.1's spelling matches, a moved step does not."""

    assert "RemovableHandle(self._backward_hooks)" in REGISTER_HOOK_BODY
    assert register_hook_body_matches(_same_steps_older_spelling)
    assert not register_hook_body_matches(_one_step_moved)
    assert not register_hook_body_matches(len)


def test_import_does_not_probe() -> None:
    """``import torchlens`` leaves the direct install unprobed: detection waits for first capture."""

    code = (
        "import torchlens; from torchlens._utils import _torch_compat as tc; "
        "print(tc._DIRECT_TENSOR_HOOK_PROBED, tc.HAS_DIRECT_TENSOR_HOOK_INSTALL)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300
    )
    assert out.stdout.split() == ["False", "False"]


def test_removable_handle_id_counter_is_the_one_the_handle_free_route_advances() -> None:
    """``RemovableHandle`` still keys hooks from its class counter, advanced by one per handle."""

    source = inspect.getsource(RemovableHandle.__init__)
    assert "self.id = RemovableHandle.next_id" in source
    assert "RemovableHandle.next_id += 1" in source


def test_plain_install_refuses_like_torch(route: str) -> None:
    """A tensor outside autograd gets torch's own error from every route."""

    with pytest.raises(RuntimeError) as torch_error:
        torch.zeros(3).register_hook(lambda grad: grad)
    with pytest.raises(RuntimeError) as plain_error:
        register_plain_tensor_hook(torch.zeros(3), lambda grad: grad)
    with pytest.raises(RuntimeError) as add_error:
        _capture_add(torch.zeros(3), lambda grad: grad, route)
    assert str(plain_error.value) == str(torch_error.value) == str(add_error.value)


@pytest.mark.parametrize("plain_first", [True, False])
def test_both_routes_share_one_dict_in_registration_order(plain_first: bool, route: str) -> None:
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


def _register_by_route(
    y: torch.Tensor, route: str, hook: Callable[..., object], install: str
) -> int | None:
    """Register ``hook`` on ``y`` by ``route``; return the handle id, or None for ``add``."""

    if route == "add":
        _capture_add(y, hook, install)
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
def test_handle_free_route_shares_the_dict_in_registration_order(
    routes: tuple[str, ...], route: str
) -> None:
    """Hooks from every route take increasing keys in one dict and fire in registration order."""

    x = torch.randn(4, requires_grad=True)
    y = x * 2
    order: list[str] = []
    ids = [
        _register_by_route(y, step, _recording_hook(order, f"{step}{index}"), route)
        for index, step in enumerate(routes)
    ]
    keys = list(y._backward_hooks)
    assert keys == sorted(keys)
    assert len(keys) == len(routes)
    assert all(
        key == hook_id for key, hook_id in zip(keys, ids, strict=True) if hook_id is not None
    )
    y.sum().backward()
    assert order == [f"{step}{index}" for index, step in enumerate(routes)]
    assert torch.equal(x.grad, torch.full_like(x, 2.0))


def test_a_handle_removes_only_its_own_hook_beside_handle_free_hooks(route: str) -> None:
    """A torch handle's ``remove()`` leaves the hooks the handle-free route added."""

    x = torch.randn(4, requires_grad=True)
    y = x * 2
    order: list[str] = []
    _capture_add(y, _recording_hook(order, "add0"), route)
    handle = y.register_hook(_recording_hook(order, "torch1"))
    _capture_add(y, _recording_hook(order, "add2"), route)
    handle.remove()
    y.sum().backward()
    assert order == ["add0", "add2"]


@pytest.mark.parametrize("leaf", [True, False])
def test_plain_install_passes_its_return_on(leaf: bool, route: str) -> None:
    """The hook's return replaces the gradient, on a leaf and on an op output, in order."""

    x = torch.randn(4, requires_grad=True)
    hooked = x if leaf else x * 1
    register_plain_tensor_hook(hooked, lambda grad: grad * 3)
    _capture_add(hooked, lambda grad: grad * 5, route)
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


def test_subclass_and_function_mode_keep_torchs_register_hook(route: str) -> None:
    """Only a plain tensor with no ``__torch_function__`` override takes the direct install."""

    x = torch.randn(4, requires_grad=True)
    sub = (x * 2).as_subclass(_Sub)
    assert takes_plain_hook_install(x * 2) == (route == "direct")
    assert not takes_plain_hook_install(sub)
    assert not add_plain_tensor_hook(sub, lambda grad: grad)
    assert sub._backward_hooks is None
    with _PassThroughMode():
        assert not takes_plain_hook_install(x * 2)
        assert not add_plain_tensor_hook(x * 2, lambda grad: grad)
    order: list[str] = []
    register_tensor_grad_hook(sub, _recording_hook(order, "sub"), _no_slot_prehook)
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


def test_capture_hook_fires_before_a_user_hook_registered_later_in_forward(route: str) -> None:
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


def _capture_grads(
    model: _LinearTanh, x: torch.Tensor
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Capture with a user gradient hook; return every saved gradient and the hook firing order."""

    order: list[str] = []

    def user_hook(grad: torch.Tensor) -> torch.Tensor:
        order.append("user")
        return grad * 2

    def capture_spy(grad: torch.Tensor) -> torch.Tensor:
        order.append("leaf")
        return grad

    handle = model.lin.register_forward_hook(_hook_output_with(user_hook))
    leaf = x.clone().requires_grad_(True)
    leaf.register_hook(capture_spy)
    trace = tl.trace(model, leaf, capture=CaptureOptions(save_grads=True))
    handle.remove()
    model.zero_grad()
    trace.backward(trace[trace.output_layers[0]].out.sum())
    grads = {
        label: trace[label].grad.clone()
        for label in trace.layer_labels
        if trace[label].grad is not None
    }
    grads["lin.weight"] = model.lin.weight.grad.clone()
    trace.cleanup()
    return grads, order


def test_forced_off_flag_captures_identical_gradients_and_hook_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``HAS_DIRECT_TENSOR_HOOK_INSTALL`` forced off, capture saves the same gradients."""

    if tc.get_direct_tensor_hook_install() is None:
        pytest.skip("this torch does not admit the direct install; nothing to compare")
    torch.manual_seed(0)
    model = _LinearTanh()
    x = torch.randn(3, 4)
    direct_grads, direct_order = _capture_grads(model, x)
    monkeypatch.setattr(tc, "HAS_DIRECT_TENSOR_HOOK_INSTALL", False)
    fallback_grads, fallback_order = _capture_grads(model, x)
    assert direct_order == fallback_order == ["user", "leaf"]
    assert direct_grads.keys() == fallback_grads.keys()
    assert len(direct_grads) > 2
    for label, grad in direct_grads.items():
        assert torch.equal(grad, fallback_grads[label]), label
