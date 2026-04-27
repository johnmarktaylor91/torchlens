"""Smoke tests for train-mode capture APIs."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


@pytest.mark.smoke
def test_log_forward_pass_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """log_forward_pass train_mode keeps saved activations differentiable."""

    model_log = tl.log_forward_pass(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = model_log[model_log.output_layers[0]].activation

    assert model_log.train_mode is True
    assert saved.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    saved.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
    model_log.cleanup()


@pytest.mark.smoke
def test_save_new_activations_train_mode_basic(two_layer_mlp: TwoLayerMlp) -> None:
    """save_new_activations train_mode override keeps replay activations differentiable."""

    model_log = tl.log_forward_pass(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        detach_saved_tensors=True,
        random_seed=0,
    )
    model_log.save_new_activations(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        train_mode=True,
        random_seed=0,
    )
    saved = model_log[model_log.output_layers[0]].activation

    assert saved.grad_fn is not None
    two_layer_mlp.zero_grad(set_to_none=True)
    saved.sum().backward()
    assert any(param.grad is not None for param in two_layer_mlp.parameters())
    model_log.cleanup()
