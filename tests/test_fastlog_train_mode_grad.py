"""Regression tests for fastlog train-mode autograd retention."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def test_train_mode_grad_flows_through_saved_tensor() -> None:
    """A loss built from a train-mode fastlog payload can backpropagate."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
    x = torch.randn(2, 4, requires_grad=True)

    recording = tl.fastlog.record(
        model,
        x,
        keep_op=lambda ctx: ctx.func_name == "linear",
        train_mode=True,
    )
    saved = next(record.ram_payload for record in recording if record.ram_payload is not None)

    assert saved is not None
    assert saved.grad_fn is not None
    saved.sum().backward()
    assert x.grad is not None
    assert any(param.grad is not None for param in model.parameters())
