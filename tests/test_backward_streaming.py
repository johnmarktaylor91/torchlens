"""Smoke tests for backward grad disk streaming."""

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
    retain_in_memory: bool = True,
) -> tuple[tl.Trace, dict[str, torch.Tensor]]:
    """Capture a backward pass with streamed grads.

    Parameters
    ----------
    bundle_path:
        Destination bundle path.
    retain_in_memory:
        Whether streamed payloads should remain in memory after bundle finalization.

    Returns
    -------
    tuple[tl.Trace, dict[str, torch.Tensor]]
        Captured log and eager grad clones keyed by saved layer label.
    """

    torch.manual_seed(0)
    model = _TinyStreamingBackwardModel()
    inputs = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        inputs,
        layers_to_save="all",
        save_grads=True,
        storage=tl.to_disk(bundle_path, retain_in_memory=retain_in_memory),
    )
    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    expected = (
        {label: trace[label].grad.detach().clone() for label in trace.saved_grad_ops.keys()}
        if retain_in_memory
        else {}
    )
    return trace, expected


@pytest.mark.smoke
def test_log_backward_streams_grads_to_disk(tmp_path: Path) -> None:
    """Gradient streaming should write grad blobs into the bundle."""

    bundle_path = tmp_path / "backward_stream.tl"
    trace, _expected = _logged_backward_stream(bundle_path)
    manifest = Manifest.read(bundle_path / "manifest.json")

    assert bundle_path.exists()
    assert trace.has_backward_pass
    assert any(entry.kind == "grad" for entry in manifest.tensors)
    assert len(trace.saved_grad_ops) > 0


@pytest.mark.smoke
def test_lazy_load_grad_from_bundle(tmp_path: Path) -> None:
    """Lazy-loaded grads should materialize on field access."""

    bundle_path = tmp_path / "backward_stream.tl"
    _trace, expected = _logged_backward_stream(bundle_path)
    lazy_log = tl.load(bundle_path, lazy=True)
    label = next(iter(expected))
    layer = lazy_log[label]

    assert layer._slot("grad") is None
    assert layer.grad_ref is not None
    assert torch.equal(layer.grad, expected[label])
    assert isinstance(layer._slot("grad"), torch.Tensor)


@pytest.mark.smoke
def test_bundle_save_load_roundtrip_with_backward(tmp_path: Path) -> None:
    """Bundle save/load should preserve backward metadata and cross-references."""

    source_path = tmp_path / "backward_stream.tl"
    trace, expected = _logged_backward_stream(source_path)
    roundtrip_path = tmp_path / "roundtrip.tl"

    tl.save(trace, roundtrip_path)
    restored = tl.load(roundtrip_path, lazy=True)
    label = next(iter(expected))
    grad_fn_handle = next(
        grad_fn_handle for grad_fn_handle in restored.grad_fns if grad_fn_handle.op
    )

    assert restored.has_backward_pass is True
    assert restored.num_backward_passes == trace.num_backward_passes
    assert restored.backward_memory_backend == trace.backward_memory_backend
    assert len(restored.grad_fn_logs) == len(trace.grad_fn_logs)
    assert restored.grad_fn_order == trace.grad_fn_order
    assert grad_fn_handle.op.grad_fn_handle is grad_fn_handle
    assert torch.equal(restored[label].grad, expected[label])


@pytest.mark.smoke
def test_train_mode_disk_save_rejected_for_grads(tmp_path: Path) -> None:
    """Training mode should reject grad disk streaming."""

    model = _TinyStreamingBackwardModel()
    inputs = torch.randn(2, 3, requires_grad=True)

    with pytest.raises(ValueError, match="disk-backed gradient storage"):
        tl.trace(
            model,
            inputs,
            layers_to_save="all",
            save_grads=True,
            storage=tl.to_disk(tmp_path / "bad.tl"),
            backward_ready=True,
        )


@pytest.mark.smoke
def test_retain_in_memory_false_for_grad_streaming(tmp_path: Path) -> None:
    """Explicit grad eviction should keep lazy refs and drop live tensors."""

    bundle_path = tmp_path / "backward_stream.tl"
    trace, _expected = _logged_backward_stream(
        bundle_path,
        retain_in_memory=False,
    )
    layer = trace[list(trace.saved_grad_ops.keys())[0]]

    assert layer._slot("grad") is None
    assert layer.grad_ref is not None
    assert isinstance(layer.grad, torch.Tensor)
