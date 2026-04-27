"""Smoke tests for backward gradient disk streaming."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

pytest.importorskip("safetensors")

import torchlens as tl
from torchlens._io.manifest import Manifest


class _TinyStreamingBackwardModel(nn.Module):
    """Small model used by backward streaming tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.fc2(torch.relu(self.fc1(x)))


def _logged_backward_stream(
    bundle_path: Path,
    *,
    keep_gradients_in_memory: bool = True,
) -> tuple[tl.ModelLog, dict[str, torch.Tensor]]:
    """Capture a backward pass with streamed gradients.

    Parameters
    ----------
    bundle_path:
        Destination bundle path.
    keep_gradients_in_memory:
        Whether gradients should remain in memory after bundle finalization.

    Returns
    -------
    tuple[tl.ModelLog, dict[str, torch.Tensor]]
        Captured log and eager gradient clones keyed by saved layer label.
    """

    torch.manual_seed(0)
    model = _TinyStreamingBackwardModel()
    inputs = torch.randn(2, 3, requires_grad=True)
    model_log = tl.log_forward_pass(
        model,
        inputs,
        layers_to_save="all",
        save_gradients=True,
        save_gradients_to=bundle_path,
        keep_gradients_in_memory=keep_gradients_in_memory,
    )
    model_log.log_backward(model_log[model_log.output_layers[0]].activation.sum())
    expected = (
        {
            label: model_log[label].gradient.detach().clone()
            for label in model_log.layers_with_saved_gradients
        }
        if keep_gradients_in_memory
        else {}
    )
    return model_log, expected


@pytest.mark.smoke
def test_log_backward_streams_gradients_to_disk(tmp_path: Path) -> None:
    """Gradient streaming should write gradient blobs into the bundle."""

    bundle_path = tmp_path / "backward_stream.tl"
    model_log, _expected = _logged_backward_stream(bundle_path)
    manifest = Manifest.read(bundle_path / "manifest.json")

    assert bundle_path.exists()
    assert model_log.has_backward_log
    assert any(entry.kind == "gradient" for entry in manifest.tensors)
    assert len(model_log.layers_with_saved_gradients) > 0


@pytest.mark.smoke
def test_lazy_load_gradient_from_bundle(tmp_path: Path) -> None:
    """Lazy-loaded gradients should materialize on field access."""

    bundle_path = tmp_path / "backward_stream.tl"
    _model_log, expected = _logged_backward_stream(bundle_path)
    lazy_log = tl.load(bundle_path, lazy=True)
    label = next(iter(expected))
    layer = lazy_log[label]

    assert layer.__dict__["gradient"] is None
    assert layer.gradient_ref is not None
    assert torch.equal(layer.gradient, expected[label])
    assert isinstance(layer.__dict__["gradient"], torch.Tensor)


@pytest.mark.smoke
def test_bundle_save_load_roundtrip_with_backward(tmp_path: Path) -> None:
    """Bundle save/load should preserve backward metadata and cross-references."""

    source_path = tmp_path / "backward_stream.tl"
    model_log, expected = _logged_backward_stream(source_path)
    roundtrip_path = tmp_path / "roundtrip.tl"

    tl.save(model_log, roundtrip_path)
    restored = tl.load(roundtrip_path, lazy=True)
    label = next(iter(expected))
    grad_fn = next(grad_fn for grad_fn in restored.grad_fns if grad_fn.corresponding_layer)

    assert restored.has_backward_log is True
    assert restored.backward_num_passes == model_log.backward_num_passes
    assert restored.backward_memory_backend == model_log.backward_memory_backend
    assert len(restored.grad_fn_logs) == len(model_log.grad_fn_logs)
    assert restored.grad_fn_order == model_log.grad_fn_order
    assert grad_fn.corresponding_layer.corresponding_grad_fn is grad_fn
    assert torch.equal(restored[label].gradient, expected[label])


@pytest.mark.smoke
def test_train_mode_disk_save_rejected_for_gradients(tmp_path: Path) -> None:
    """Training mode should reject gradient disk streaming."""

    model = _TinyStreamingBackwardModel()
    inputs = torch.randn(2, 3, requires_grad=True)

    with pytest.raises(ValueError, match="gradient disk saves"):
        tl.log_forward_pass(
            model,
            inputs,
            layers_to_save="all",
            save_gradients=True,
            save_gradients_to=tmp_path / "bad.tl",
            train_mode=True,
        )


@pytest.mark.smoke
def test_keep_gradients_in_memory_false(tmp_path: Path) -> None:
    """Explicit gradient eviction should keep lazy refs and drop live tensors."""

    bundle_path = tmp_path / "backward_stream.tl"
    model_log, _expected = _logged_backward_stream(
        bundle_path,
        keep_gradients_in_memory=False,
    )
    layer = model_log[model_log.layers_with_saved_gradients[0]]

    assert layer.__dict__["gradient"] is None
    assert layer.gradient_ref is not None
    assert isinstance(layer.gradient, torch.Tensor)
