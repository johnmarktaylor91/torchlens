"""Regression coverage for raw and transformed transform storage."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions, StorageOptions


class _TinyModel(nn.Module):
    """Small model with a non-trivial intermediate out."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        return torch.relu(self.linear(x))


def _first_transformed_layer(trace: tl.Trace) -> Any:
    """Return the first layer with a transformed out."""

    for label in trace.layer_labels:
        layer = trace[label]
        if layer.payload.transformed_out is not None:
            return layer
    raise AssertionError("No transformed out found.")


def _output_loss(trace: tl.Trace) -> torch.Tensor:
    """Return scalar loss from the logged output out."""

    return trace[trace.output_layers[0]].out.sum()


def test_grad_transform_keeps_raw_grad_and_transformed_metadata() -> None:
    """Gradient transform stores raw and transformed grads separately."""

    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        capture=CaptureOptions(save_grads="all", grad_transform=lambda t: t.mean()),
    )
    trace.backward(_output_loss(trace))
    layer = next(trace[label] for label in trace.saved_grad_ops.keys())

    assert layer.grad is not None
    assert layer.autograd.grad_shape == tuple(layer.grad.shape)
    assert layer.payload.transformed_grad is not None
    assert layer.payload.transformed_grad_shape == ()
    assert layer.payload.transformed_grad_dtype == layer.payload.transformed_grad.dtype
    assert layer.transformed_gradient_memory == layer.payload.transformed_grad.nelement() * (
        layer.payload.transformed_grad.element_size()
    )


def test_save_raw_grads_false_keeps_raw_metadata_only() -> None:
    """save_raw_grads=False drops raw grad storage but preserves raw metadata."""

    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        capture=CaptureOptions(
            save_grads="all", grad_transform=lambda t: t.mean(), save_raw_grads=False
        ),
    )
    trace.backward(_output_loss(trace))
    layer = next(trace[label] for label in trace.saved_grad_ops.keys())

    assert layer.grad is None
    assert layer.autograd.grad_shape is not None
    assert layer.autograd.grad_dtype is not None
    assert layer.gradient_memory > 0
    assert layer.payload.transformed_grad is not None


def test_train_mode_activation_transform_detach_rejected() -> None:
    """Detached train-mode out transforms are rejected."""

    with pytest.raises(
        tl.errors.TrainingModeConfigError, match="disconnected from the autograd graph"
    ):
        tl.trace(
            _TinyModel(),
            torch.randn(2, 4, requires_grad=True),
            capture=CaptureOptions(backward_ready=True, activation_transform=lambda t: t.detach()),
        )


def test_transform_error_has_context_and_cause() -> None:
    """Postfunc failures include layer, op, shape, dtype, and original cause."""

    def _raise(_: torch.Tensor) -> torch.Tensor:
        """Raise a sentinel error."""

        raise ValueError("sentinel")

    with pytest.raises(tl.errors.UserTransformError) as exc_info:
        tl.trace(
            _TinyModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(activation_transform=_raise),
        )

    message = str(exc_info.value)
    assert "activation_transform raised" in message
    assert "layer" in message
    assert "func=" in message
    assert "shape=" in message
    assert "dtype=torch.float32" in message
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_portable_save_roundtrip_preserves_transformed_out(tmp_path: Path) -> None:
    """Portable save/load preserves transformed out fields."""

    bundle_path = tmp_path / "transform_bundle.tl"
    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(activation_transform=torch.mean),
    )
    layer = _first_transformed_layer(trace)

    trace.save(bundle_path)
    restored = tl.load(bundle_path)
    restored_layer = restored[layer.layer_label]

    assert torch.equal(restored_layer.payload.transformed_out, layer.payload.transformed_out)
    assert restored_layer.payload.transformed_out_shape == layer.payload.transformed_out_shape
    assert restored_layer.payload.transformed_out_dtype == layer.payload.transformed_out_dtype
    assert (
        restored_layer.cost.transformed_activation_memory
        == layer.cost.transformed_activation_memory
    )


def test_streaming_preserves_transformed_out(tmp_path: Path) -> None:
    """Streaming bundles write transformed out fields."""

    bundle_path = tmp_path / "streamed_transform.tl"
    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4),
        capture=CaptureOptions(activation_transform=torch.mean),
        save="all",
        storage=StorageOptions(path=bundle_path),
    )
    layer = _first_transformed_layer(trace)
    restored = tl.load(bundle_path)
    restored_layer = restored[layer.layer_label]

    assert torch.equal(restored_layer.payload.transformed_out, layer.payload.transformed_out)
