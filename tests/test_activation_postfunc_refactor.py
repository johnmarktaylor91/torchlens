"""Regression coverage for raw and transformed postfunc storage."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    """Small model with a non-trivial intermediate activation."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        return torch.relu(self.linear(x))


def _first_transformed_layer(model_log: tl.ModelLog) -> Any:
    """Return the first layer with a transformed activation."""

    for label in model_log.layer_labels:
        layer = model_log[label]
        if layer.transformed_activation is not None:
            return layer
    raise AssertionError("No transformed activation found.")


def _output_loss(model_log: tl.ModelLog) -> torch.Tensor:
    """Return scalar loss from the logged output activation."""

    return model_log[model_log.output_layers[0]].activation.sum()


@pytest.mark.smoke
def test_activation_postfunc_keeps_raw_tensor_and_transformed_metadata() -> None:
    """Activation postfunc stores raw and transformed tensors separately."""

    x = torch.randn(2, 4)
    model_log = tl.log_forward_pass(_TinyModel(), x, activation_postfunc=lambda t: t.mean())
    layer = _first_transformed_layer(model_log)

    assert layer.tensor is layer.activation
    assert layer.activation.shape == layer.tensor_shape
    assert layer.transformed_activation.shape == torch.Size([])
    assert layer.transformed_activation_shape == ()
    assert layer.tensor_shape != layer.transformed_activation_shape
    assert layer.tensor_dtype == layer.activation.dtype
    assert layer.transformed_activation_dtype == layer.transformed_activation.dtype
    assert layer.transformed_activation_memory == layer.transformed_activation.nelement() * (
        layer.transformed_activation.element_size()
    )


def test_transformed_activation_absent_without_postfunc() -> None:
    """No activation postfunc leaves transformed activation fields empty."""

    model_log = tl.log_forward_pass(_TinyModel(), torch.randn(2, 4))

    assert all(model_log[label].transformed_activation is None for label in model_log.layer_labels)


def test_save_raw_activation_false_keeps_raw_metadata_only() -> None:
    """save_raw_activation=False drops raw tensor storage but preserves raw metadata."""

    model_log = tl.log_forward_pass(
        _TinyModel(),
        torch.randn(2, 4),
        activation_postfunc=lambda t: t.mean(),
        save_raw_activation=False,
    )
    layer = _first_transformed_layer(model_log)

    assert layer.activation is None
    assert layer.tensor is None
    assert layer.tensor_shape is not None
    assert layer.tensor_dtype is not None
    assert layer.tensor_memory > 0
    assert layer.transformed_activation is not None


def test_gradient_postfunc_keeps_raw_gradient_and_transformed_metadata() -> None:
    """Gradient postfunc stores raw and transformed gradients separately."""

    model_log = tl.log_forward_pass(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        gradients_to_save="all",
        gradient_postfunc=lambda t: t.mean(),
    )
    model_log.log_backward(_output_loss(model_log))
    layer = next(model_log[label] for label in model_log.layers_with_saved_gradients)

    assert layer.gradient is not None
    assert layer.grad_shape == tuple(layer.gradient.shape)
    assert layer.transformed_gradient is not None
    assert layer.transformed_gradient_shape == ()
    assert layer.transformed_gradient_dtype == layer.transformed_gradient.dtype
    assert layer.transformed_gradient_memory == layer.transformed_gradient.nelement() * (
        layer.transformed_gradient.element_size()
    )


def test_save_raw_gradient_false_keeps_raw_metadata_only() -> None:
    """save_raw_gradient=False drops raw gradient storage but preserves raw metadata."""

    model_log = tl.log_forward_pass(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        gradients_to_save="all",
        gradient_postfunc=lambda t: t.mean(),
        save_raw_gradient=False,
    )
    model_log.log_backward(_output_loss(model_log))
    layer = next(model_log[label] for label in model_log.layers_with_saved_gradients)

    assert layer.gradient is None
    assert layer.grad_shape is not None
    assert layer.grad_dtype is not None
    assert layer.grad_memory > 0
    assert layer.transformed_gradient is not None


def test_train_mode_activation_postfunc_detach_rejected() -> None:
    """Detached train-mode activation transforms are rejected."""

    with pytest.raises(tl.TrainingModeConfigError, match="disconnected from the autograd graph"):
        tl.log_forward_pass(
            _TinyModel(),
            torch.randn(2, 4, requires_grad=True),
            train_mode=True,
            activation_postfunc=lambda t: t.detach(),
        )


def test_train_mode_activation_postfunc_int_rejected() -> None:
    """Integer train-mode activation transforms are rejected."""

    with pytest.raises(tl.TrainingModeConfigError, match="non-grad dtype"):
        tl.log_forward_pass(
            _TinyModel(),
            torch.randn(2, 4, requires_grad=True),
            train_mode=True,
            activation_postfunc=lambda t: t.to(torch.int64),
        )


def test_train_mode_activation_postfunc_connected_passes() -> None:
    """Differentiable train-mode activation transforms are accepted."""

    model_log = tl.log_forward_pass(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        train_mode=True,
        activation_postfunc=lambda t: t * 2,
    )

    assert _first_transformed_layer(model_log).transformed_activation.grad_fn is not None


def test_postfunc_error_has_context_and_cause() -> None:
    """Postfunc failures include layer, op, shape, dtype, and original cause."""

    def _raise(_: torch.Tensor) -> torch.Tensor:
        """Raise a sentinel error."""

        raise ValueError("sentinel")

    with pytest.raises(tl.TorchLensPostfuncError) as exc_info:
        tl.log_forward_pass(_TinyModel(), torch.randn(2, 4), activation_postfunc=_raise)

    message = str(exc_info.value)
    assert "activation_postfunc raised" in message
    assert "layer" in message
    assert "func=" in message
    assert "shape=" in message
    assert "dtype=torch.float32" in message
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_portable_save_roundtrip_preserves_transformed_activation(tmp_path: Path) -> None:
    """Portable save/load preserves transformed activation fields."""

    bundle_path = tmp_path / "postfunc_bundle.tl"
    model_log = tl.log_forward_pass(_TinyModel(), torch.randn(2, 4), activation_postfunc=torch.mean)
    layer = _first_transformed_layer(model_log)

    tl.save(model_log, bundle_path)
    restored = tl.load(bundle_path)
    restored_layer = restored[layer.layer_label]

    assert torch.equal(restored_layer.transformed_activation, layer.transformed_activation)
    assert restored_layer.transformed_activation_shape == layer.transformed_activation_shape
    assert restored_layer.transformed_activation_dtype == layer.transformed_activation_dtype
    assert restored_layer.transformed_activation_memory == layer.transformed_activation_memory


def test_postfunc_type_aliases_exported() -> None:
    """Activation and gradient postfunc aliases are importable from torchlens."""

    from torchlens import ActivationPostfunc, GradientPostfunc

    assert ActivationPostfunc is not None
    assert GradientPostfunc is not None


def test_streaming_preserves_transformed_activation(tmp_path: Path) -> None:
    """Streaming bundles write transformed activation fields."""

    bundle_path = tmp_path / "streamed_postfunc.tl"
    model_log = tl.log_forward_pass(
        _TinyModel(),
        torch.randn(2, 4),
        activation_postfunc=torch.mean,
        save_activations_to=bundle_path,
        layers_to_save="all",
    )
    layer = _first_transformed_layer(model_log)
    restored = tl.load(bundle_path)
    restored_layer = restored[layer.layer_label]

    assert torch.equal(restored_layer.transformed_activation, layer.transformed_activation)
