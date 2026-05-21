"""Smoke tests for train-mode capture APIs."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


@pytest.mark.smoke
def test_trace_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """trace backward_ready keeps saved outs differentiable."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        backward_ready=True,
        random_seed=0,
    )
    saved = trace[trace.output_layers[0]].out

    assert trace.backward_ready is True
    assert saved.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    saved.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
    trace.cleanup()


@pytest.mark.smoke
def test_save_new_outs_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """save_new_outs backward_ready override keeps replay outs differentiable."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        detach_saved_activations=True,
        random_seed=0,
    )
    trace.save_new_outs(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        backward_ready=True,
        random_seed=0,
    )
    saved = trace[trace.output_layers[0]].out

    assert saved.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    saved.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
    trace.cleanup()


@pytest.mark.smoke
def test_fastlog_record_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """fastlog record backward_ready keeps recorded payloads differentiable."""

    recording = tl.fastlog.record(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        backward_ready=True,
    )
    payload = next(record.ram_payload for record in recording if record.ram_payload is not None)

    assert payload.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    payload.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
