"""Regression tests for train-mode activation retention."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


def _assert_backward_populates_grad(model: torch.nn.Module, loss: torch.Tensor) -> None:
    """Assert a saved activation can backpropagate to model parameters."""

    model.zero_grad(set_to_none=True)
    loss.sum().backward()
    assert any(param.grad is not None for param in model.parameters())


@pytest.mark.smoke
def test_a1_save_new_activations_output_layer_keeps_grad(two_layer_mlp: TwoLayerMlp) -> None:
    """save_new_activations keeps the fast-postprocessed output attached."""

    x = torch.randn(3, 4, requires_grad=True)
    model_log = tl.log_forward_pass(two_layer_mlp, x, random_seed=0)
    output_label = model_log.output_layers[0]

    model_log.save_new_activations(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        layers_to_save=[output_label],
        random_seed=0,
    )

    saved = model_log[output_label].activation
    assert saved.grad_fn is not None
    _assert_backward_populates_grad(two_layer_mlp, saved)
    model_log.cleanup()


@pytest.mark.smoke
def test_a1_two_pass_selective_save_output_layer_keeps_grad(two_layer_mlp: TwoLayerMlp) -> None:
    """Selective two-pass capture keeps the output-layer activation attached."""

    probe_log = tl.log_forward_pass(two_layer_mlp, torch.randn(3, 4), random_seed=0)
    output_label = probe_log.output_layers[0]
    probe_log.cleanup()

    model_log = tl.log_forward_pass(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        layers_to_save=[output_label],
        random_seed=0,
    )

    saved = model_log[output_label].activation
    assert saved.grad_fn is not None
    _assert_backward_populates_grad(two_layer_mlp, saved)
    model_log.cleanup()
