"""Regression tests for train-mode out retention."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from .conftest import TwoLayerMlp


def _assert_backward_populates_grad(model: torch.nn.Module, loss: torch.Tensor) -> None:
    """Assert a saved out can backpropagate to model parameters."""

    model.zero_grad(set_to_none=True)
    loss.sum().backward()
    assert any(param.grad is not None for param in model.parameters())


@pytest.mark.smoke
def test_a1_save_new_outs_output_layer_keeps_grad(two_layer_mlp: TwoLayerMlp) -> None:
    """save_new_outs keeps the fast-postprocessed output attached."""

    x = torch.randn(3, 4, requires_grad=True)
    trace = tl.trace(two_layer_mlp, x, random_seed=0)
    output_label = trace.output_layers[0]

    trace.save_new_outs(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        layers_to_save=[output_label],
        random_seed=0,
    )

    saved = trace[output_label].out
    assert saved.grad_fn is not None
    _assert_backward_populates_grad(two_layer_mlp, saved)
    trace.cleanup()


@pytest.mark.smoke
def test_a1_two_pass_selective_save_output_layer_keeps_grad(two_layer_mlp: TwoLayerMlp) -> None:
    """Selective two-pass capture keeps the output-layer out attached."""

    probe_log = tl.trace(two_layer_mlp, torch.randn(3, 4), random_seed=0)
    output_label = probe_log.output_layers[0]
    probe_log.cleanup()

    trace = tl.trace(
        two_layer_mlp,
        torch.randn(3, 4, requires_grad=True),
        layers_to_save=[output_label],
        random_seed=0,
    )

    saved = trace[output_label].out
    assert saved.grad_fn is not None
    _assert_backward_populates_grad(two_layer_mlp, saved)
    trace.cleanup()
