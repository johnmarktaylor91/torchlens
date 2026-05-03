"""Smoke tests for train-mode capture APIs."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


@pytest.mark.smoke
def test_trace_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """trace train_mode keeps saved activations differentiable."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = trace[trace.output_layers[0]].activation

    assert trace.train_mode is True
    assert saved.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    saved.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
    trace.cleanup()


@pytest.mark.smoke
def test_save_new_activations_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """save_new_activations train_mode override keeps replay activations differentiable."""

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        detach_saved_tensors=True,
        random_seed=0,
    )
    trace.save_new_activations(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = trace[trace.output_layers[0]].activation

    assert saved.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    saved.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
    trace.cleanup()


@pytest.mark.smoke
def test_fastlog_record_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """fastlog record train_mode keeps recorded payloads differentiable."""

    recording = tl.fastlog.record(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
    )
    payload = next(record.ram_payload for record in recording if record.ram_payload is not None)

    assert payload.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    payload.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
