"""Selector ``save_grads`` records gradients at ops whose output is written in place.

An in-place op logs its output against a safe copy whose ``grad_fn`` is a
dead-end ``CloneBackward`` node; the live tensor the model keeps using carries
the real gradient. With ``save_grads=True`` the gradient hook goes on the live
tensor during the forward. A selector spelling (``tl.module(...)``,
``tl.label(...)``) defers hook installation until the forward ends, and the
deferred install used to hook only the dead-end copy, so the op recorded no
gradient: a module ending in ``ReLU(inplace=True)`` (the torchvision ResNet
``BasicBlock``) came back with ``op.grad is None``.

Every test compares the recorded gradient with an eager reference taken by a
plain tensor hook or ``retain_grad``, never with another TorchLens result.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions


def _module_output_reference(model: nn.Module, x: torch.Tensor, address: str) -> torch.Tensor:
    """Return the gradient a ``retain_grad`` forward hook sees at ``address``."""

    held: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        output.retain_grad()
        held["out"] = output

    handle = model.get_submodule(address).register_forward_hook(hook)
    try:
        model(x).sum().backward()
    finally:
        handle.remove()
    grad = held["out"].grad
    assert grad is not None
    return grad


def _module_site_grad(
    model: nn.Module, x: torch.Tensor, address: str, save_grads: object
) -> torch.Tensor | None:
    """Trace ``model``, run a managed backward and return the module output's grad."""

    trace = tl.trace(model, x, capture=CaptureOptions(save_grads=save_grads))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    sites = list(trace.find_sites(tl.module(address)))
    assert len(sites) == 1
    return sites[0].grad


class _EndsInPlace(nn.Module):
    """Linear followed by one in-place op that writes the module's output."""

    def __init__(self, family: str) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)
        self.family = family

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin(x)
        if self.family == "relu_":
            return out.relu_()
        if self.family == "add_":
            return out.add_(x)
        if self.family == "mul_":
            return out.mul_(x)
        if self.family == "masked_fill_":
            return out.masked_fill_(x > 0, 0.5)
        if self.family == "iadd":
            out += x
            return out
        raise AssertionError(self.family)


class _Wrapper(nn.Module):
    """Parent model: the in-place module sits between two out-of-place ops."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.inner(x)))


class _BasicBlock(nn.Module):
    """ResNet ``BasicBlock`` shape: residual ``+=`` then ``ReLU(inplace=True)``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class _TinyResNet(nn.Module):
    """Stem, a two-block ``layer2`` stage ending in ``relu_``, pooled head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Sequential(_BasicBlock(8), _BasicBlock(8))
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(self.relu(self.bn1(self.conv1(x))))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))


@pytest.mark.parametrize("family", ["relu_", "add_", "mul_", "masked_fill_", "iadd"])
@pytest.mark.parametrize("spelling", ["selector", "all"])
def test_module_ending_in_place_records_the_retain_grad_gradient(
    family: str, spelling: str
) -> None:
    """Each in-place family at a module output records the eager gradient."""

    torch.manual_seed(0)
    model = _Wrapper(_EndsInPlace(family)).eval()
    x = torch.randn(2, 6)
    reference = _module_output_reference(copy.deepcopy(model), x, "inner")
    save_grads = tl.module("inner") if spelling == "selector" else True
    grad = _module_site_grad(model, x, "inner", save_grads)
    assert grad is not None
    assert torch.allclose(grad, reference)


@pytest.mark.parametrize("address", ["layer2", "layer2.0", "layer2.1"])
def test_resnet_basic_block_stage_records_the_retain_grad_gradient(address: str) -> None:
    """The ResNet stage and each block, all ending in ``relu_``, record their gradient."""

    torch.manual_seed(0)
    model = _TinyResNet().eval()
    x = torch.randn(1, 3, 8, 8)
    reference = _module_output_reference(copy.deepcopy(model), x, address)
    grad = _module_site_grad(model, x, address, tl.module(address))
    assert grad is not None
    assert grad.shape == (1, 8, 8, 8)
    assert torch.allclose(grad, reference)


class _WrittenThreeTimes(nn.Module):
    """One tensor written in place three times inside the module body."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(
        self, x: torch.Tensor, probe: Callable[[torch.Tensor, str], None] | None = None
    ) -> torch.Tensor:
        y = self.lin(x)
        y.add_(x)
        if probe is not None:
            probe(y, "add_")
        y.mul_(x)
        if probe is not None:
            probe(y, "mul_")
        y.relu_()
        if probe is not None:
            probe(y, "relu_")
        return torch.tanh(y)


def _per_version_reference(model: _WrittenThreeTimes, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return each in-place version's gradient from a hook placed at that version."""

    grads: dict[str, torch.Tensor] = {}

    def probe(t: torch.Tensor, name: str) -> None:
        t.register_hook(lambda g: grads.__setitem__(name, g.clone()))

    model(x, probe).sum().backward()
    return grads


@pytest.mark.parametrize("func_name", ["add_", "mul_", "relu_"])
def test_each_in_place_write_to_one_tensor_records_its_own_gradient(func_name: str) -> None:
    """A selected earlier write gets its version's gradient, not the last write's."""

    torch.manual_seed(0)
    model = _WrittenThreeTimes().eval()
    x = torch.randn(2, 6)
    reference = _per_version_reference(copy.deepcopy(model), x)
    assert not torch.allclose(reference["add_"], reference["relu_"])
    trace = tl.trace(model, x, capture=CaptureOptions(save_grads=tl.func(func_name)))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    sites = [op for op in trace if getattr(op, "func_name", None) == func_name]
    assert len(sites) == 1
    assert sites[0].grad is not None
    assert torch.allclose(sites[0].grad, reference[func_name])


class _WritesThroughView(nn.Module):
    """An in-place write to a view changes the base the module returns."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        y[:, :3].mul_(2.0)
        return y


def test_in_place_write_through_a_view_records_the_base_gradient() -> None:
    """The module output is the base written through a view; its gradient is recorded."""

    torch.manual_seed(0)
    model = _Wrapper(_WritesThroughView()).eval()
    x = torch.randn(2, 6)
    reference = _module_output_reference(copy.deepcopy(model), x, "inner")
    grad = _module_site_grad(model, x, "inner", tl.module("inner"))
    assert grad is not None
    assert torch.allclose(grad, reference)


class _AddsParameterView(nn.Module):
    """The in-place op's operand is a view of a leaf Parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 6)
        self.shift = nn.Parameter(torch.randn(6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).add_(self.shift.view(1, 6))


def test_in_place_op_with_a_leaf_parameter_view_operand() -> None:
    """The output gradient and the Parameter's gradient both match eager."""

    torch.manual_seed(0)
    model = _Wrapper(_AddsParameterView()).eval()
    x = torch.randn(2, 6)
    eager = copy.deepcopy(model)
    reference = _module_output_reference(eager, x, "inner")
    eager_shift_grad = eager.inner.shift.grad
    assert eager_shift_grad is not None
    grad = _module_site_grad(model, x, "inner", tl.module("inner"))
    assert grad is not None
    assert torch.allclose(grad, reference)
    assert model.inner.shift.grad is not None
    assert torch.allclose(model.inner.shift.grad, eager_shift_grad)


def test_unselected_in_place_ops_record_no_gradient() -> None:
    """The fix hooks only selected ops: the other in-place writes stay gradient-free."""

    torch.manual_seed(0)
    model = _WrittenThreeTimes().eval()
    trace = tl.trace(model, torch.randn(2, 6), capture=CaptureOptions(save_grads=tl.func("mul_")))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    with_grad = [op.func_name for op in trace if getattr(op, "grad", None) is not None]
    assert with_grad == ["mul_"]


def test_resnet_stage_backward_validation_passes() -> None:
    """Backward validation of the in-place ResNet stage passes unchanged."""

    torch.manual_seed(0)
    assert tl.validate(_TinyResNet().eval(), torch.randn(1, 3, 8, 8), scope="backward")
