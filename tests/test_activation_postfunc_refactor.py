"""Regression coverage for raw and transformed postfunc storage."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


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
        if layer.transformed_out is not None:
            return layer
    raise AssertionError("No transformed out found.")


def _output_loss(trace: tl.Trace) -> torch.Tensor:
    """Return scalar loss from the logged output out."""

    return trace[trace.output_layers[0]].out.sum()


@pytest.mark.smoke
def test_out_postfunc_keeps_raw_tensor_and_transformed_metadata() -> None:
    """Activation postfunc stores raw and transformed tensors separately."""

    x = torch.randn(2, 4)
    trace = tl.trace(_TinyModel(), x, out_postfunc=lambda t: t.mean())
    layer = _first_transformed_layer(trace)

    assert layer.tensor is layer.out
    assert layer.out.shape == layer.shape
    assert layer.transformed_out.shape == torch.Size([])
    assert layer.transformed_out_shape == ()
    assert layer.shape != layer.transformed_out_shape
    assert layer.dtype == layer.out.dtype
    assert layer.transformed_out_dtype == layer.transformed_out.dtype
    assert layer.transformed_activation_memory == layer.transformed_out.nelement() * (
        layer.transformed_out.element_size()
    )


def test_transformed_out_absent_without_postfunc() -> None:
    """No out postfunc leaves transformed out fields empty."""

    trace = tl.trace(_TinyModel(), torch.randn(2, 4))

    assert all(trace[label].transformed_out is None for label in trace.layer_labels)


def test_save_raw_outs_false_keeps_raw_metadata_only() -> None:
    """save_raw_outs=False drops raw tensor storage but preserves raw metadata."""

    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4),
        out_postfunc=lambda t: t.mean(),
        save_raw_outs=False,
    )
    layer = _first_transformed_layer(trace)

    assert layer.out is None
    assert layer.tensor is None
    assert layer.shape is not None
    assert layer.dtype is not None
    assert layer.memory > 0
    assert layer.transformed_out is not None


def test_grad_transform_keeps_raw_grad_and_transformed_metadata() -> None:
    """Gradient postfunc stores raw and transformed grads separately."""

    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        gradients_to_save="all",
        gradient_transform=lambda t: t.mean(),
    )
    trace.log_backward(_output_loss(trace))
    layer = next(trace[label] for label in trace.saved_grad_ops.keys())

    assert layer.grad is not None
    assert layer.grad_shape == tuple(layer.grad.shape)
    assert layer.transformed_grad is not None
    assert layer.transformed_grad_shape == ()
    assert layer.transformed_grad_dtype == layer.transformed_grad.dtype
    assert layer.transformed_gradient_memory == layer.transformed_grad.nelement() * (
        layer.transformed_grad.element_size()
    )


def test_save_raw_grads_false_keeps_raw_metadata_only() -> None:
    """save_raw_gradients=False drops raw grad storage but preserves raw metadata."""

    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        gradients_to_save="all",
        gradient_transform=lambda t: t.mean(),
        save_raw_gradients=False,
    )
    trace.log_backward(_output_loss(trace))
    layer = next(trace[label] for label in trace.saved_grad_ops.keys())

    assert layer.grad is None
    assert layer.grad_shape is not None
    assert layer.grad_dtype is not None
    assert layer.gradient_memory > 0
    assert layer.transformed_grad is not None


def test_train_mode_out_postfunc_detach_rejected() -> None:
    """Detached train-mode out transforms are rejected."""

    with pytest.raises(tl.TrainingModeConfigError, match="disconnected from the autograd graph"):
        tl.trace(
            _TinyModel(),
            torch.randn(2, 4, requires_grad=True),
            backward_ready=True,
            out_postfunc=lambda t: t.detach(),
        )


def test_train_mode_out_postfunc_int_rejected() -> None:
    """Integer train-mode out transforms are rejected."""

    with pytest.raises(tl.TrainingModeConfigError, match="non-grad dtype"):
        tl.trace(
            _TinyModel(),
            torch.randn(2, 4, requires_grad=True),
            backward_ready=True,
            out_postfunc=lambda t: t.to(torch.int64),
        )


def test_train_mode_out_postfunc_connected_ops() -> None:
    """Differentiable train-mode out transforms are accepted."""

    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4, requires_grad=True),
        backward_ready=True,
        out_postfunc=lambda t: t * 2,
    )

    assert _first_transformed_layer(trace).transformed_out.grad_fn is not None


def test_postfunc_error_has_context_and_cause() -> None:
    """Postfunc failures include layer, op, shape, dtype, and original cause."""

    def _raise(_: torch.Tensor) -> torch.Tensor:
        """Raise a sentinel error."""

        raise ValueError("sentinel")

    with pytest.raises(tl.TorchLensPostfuncError) as exc_info:
        tl.trace(_TinyModel(), torch.randn(2, 4), out_postfunc=_raise)

    message = str(exc_info.value)
    assert "out_postfunc raised" in message
    assert "layer" in message
    assert "func=" in message
    assert "shape=" in message
    assert "dtype=torch.float32" in message
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_portable_save_roundtrip_preserves_transformed_out(tmp_path: Path) -> None:
    """Portable save/load preserves transformed out fields."""

    bundle_path = tmp_path / "postfunc_bundle.tl"
    trace = tl.trace(_TinyModel(), torch.randn(2, 4), out_postfunc=torch.mean)
    layer = _first_transformed_layer(trace)

    tl.save(trace, bundle_path)
    restored = tl.load(bundle_path)
    restored_layer = restored[layer.layer_label]

    assert torch.equal(restored_layer.transformed_out, layer.transformed_out)
    assert restored_layer.transformed_out_shape == layer.transformed_out_shape
    assert restored_layer.transformed_out_dtype == layer.transformed_out_dtype
    assert restored_layer.transformed_activation_memory == layer.transformed_activation_memory


def test_postfunc_type_aliases_exported() -> None:
    """Activation and grad postfunc aliases are importable from torchlens."""

    from torchlens import ActivationPostfunc, GradientPostfunc

    assert ActivationPostfunc is not None
    assert GradientPostfunc is not None


def test_streaming_preserves_transformed_out(tmp_path: Path) -> None:
    """Streaming bundles write transformed out fields."""

    bundle_path = tmp_path / "streamed_postfunc.tl"
    trace = tl.trace(
        _TinyModel(),
        torch.randn(2, 4),
        out_postfunc=torch.mean,
        save_outs_to=bundle_path,
        layers_to_save="all",
    )
    layer = _first_transformed_layer(trace)
    restored = tl.load(bundle_path)
    restored_layer = restored[layer.layer_label]

    assert torch.equal(restored_layer.transformed_out, layer.transformed_out)
