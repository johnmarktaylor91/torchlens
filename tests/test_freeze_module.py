"""Phase 5a tests for experimental freeze controls."""

from __future__ import annotations

import torch

import torchlens as tl


def test_freeze_module_restores_requires_grad_and_gradients() -> None:
    """``freeze_module`` should restore parameter training state."""

    layer = torch.nn.Linear(2, 2)
    for parameter in layer.parameters():
        parameter.grad = torch.ones_like(parameter)

    with tl.experimental.freeze_module(layer):
        assert all(not parameter.requires_grad for parameter in layer.parameters())
        assert all(parameter.grad is None for parameter in layer.parameters())

    assert all(parameter.requires_grad for parameter in layer.parameters())
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_freeze_module_does_not_change_forward_output() -> None:
    """Freezing parameters should not change observer-style forward outputs."""

    layer = torch.nn.Linear(2, 2)
    x = torch.randn(1, 2)
    before = layer(x)
    with tl.experimental.freeze_module(layer):
        during = layer(x)
    after = layer(x)

    assert torch.equal(before, during)
    assert torch.equal(before, after)
