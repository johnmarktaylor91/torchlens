"""Tests for inference-only trace capture."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._training_validation import TrainingModeConfigError
from torchlens.errors import ConfigurationError


class _TinyParamModel(nn.Module):
    """Small parameterized module whose output normally has autograd history."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a differentiable forward pass.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        return torch.relu(self.fc(x)).sum()


def _model_and_input() -> tuple[_TinyParamModel, torch.Tensor]:
    """Return a deterministic model/input pair.

    Returns
    -------
    tuple[_TinyParamModel, torch.Tensor]
        Model and input tensor that both participate in autograd.
    """

    torch.manual_seed(0)
    return _TinyParamModel(), torch.randn(4, 3, requires_grad=True)


def test_inference_only_discards_saved_output_grad_fn() -> None:
    """Inference-only capture saves outputs without autograd history."""

    model, x = _model_and_input()
    trace = tl.trace(model, x, layers_to_save="all", inference_only=True)
    try:
        assert trace.inference_only is True
        assert trace[trace.output_layers[0]].out.grad_fn is None
    finally:
        trace.cleanup()


def test_default_capture_retains_saved_output_grad_fn() -> None:
    """Default capture keeps the existing grad-fn behavior."""

    model, x = _model_and_input()
    trace = tl.trace(model, x, layers_to_save="all")
    try:
        assert trace.inference_only is False
        assert trace[trace.output_layers[0]].out.grad_fn is not None
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("kwargs", "flag_name"),
    [
        ({"backward_ready": True}, "backward_ready"),
        ({"save_grads": True}, "save_grads"),
        ({"intervention_ready": True}, "intervention_ready"),
    ],
)
def test_inference_only_rejects_backward_related_capture_flags(
    kwargs: dict[str, object], flag_name: str
) -> None:
    """Inference-only capture rejects explicit backward-related options."""

    model, x = _model_and_input()
    with pytest.raises(TrainingModeConfigError, match=flag_name):
        tl.trace(model, x, inference_only=True, **kwargs)


def test_inference_only_allows_default_tensor_grad_hook_flag() -> None:
    """The moot tensor-hook flag does not conflict with inference-only capture."""

    model, x = _model_and_input()
    trace = tl.trace(model, x, inference_only=True, capture_tensor_grad_hooks=True)
    try:
        assert trace.inference_only is True
    finally:
        trace.cleanup()


def test_inference_only_deferred_backward_raises_clear_error() -> None:
    """Deferred backward is rejected because inference-only has no autograd graph."""

    model, x = _model_and_input()
    trace = tl.trace(model, x, layers_to_save="all", inference_only=True)
    try:
        loss = model(x)
        with pytest.raises(ConfigurationError, match="autograd graph was discarded"):
            trace.log_backward(loss)
    finally:
        trace.cleanup()
