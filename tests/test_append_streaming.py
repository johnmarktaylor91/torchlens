"""Append rerun regression tests for active streaming traces."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import AppendStreamingUnsupportedError
from torchlens.options import CaptureOptions, ReplayOptions, StorageOptions


class _StreamingAppendModel(nn.Module):
    """Small shape-preserving model for streaming append tests."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.linear(x)


@pytest.mark.smoke
def test_append_rerun_on_streaming_trace_raises(tmp_path: Path) -> None:
    """Append rejects traces with a live bundle stream writer."""

    model = _StreamingAppendModel().eval()
    trace = tl.trace(
        model,
        torch.randn(1, 3),
        capture=CaptureOptions(
            layers_to_save="all",
            save_grads=True,
            replayable=True,
        ),
        storage=tl.to_disk(tmp_path / "streaming_bundle.tl"),
    )

    try:
        with pytest.raises(AppendStreamingUnsupportedError, match="bundle_path streaming"):
            trace.run(model, torch.randn(1, 3), replay=ReplayOptions(append=True))
    finally:
        # The deferred grad stream keeps its writer thread until backward finalizes it.
        trace.log_backward(trace[trace.output_layers[0]].out.sum())


def test_append_rerun_on_callback_streaming_trace_raises() -> None:
    """Append rejects traces with a live out sink callback."""

    received: list[tuple[str, torch.Tensor]] = []

    def _sink(label: str, tensor: torch.Tensor) -> None:
        """Collect streamed tensors.

        Parameters
        ----------
        label:
            Layer label.
        tensor:
            Streamed tensor value.
        """

        received.append((label, tensor))

    model = _StreamingAppendModel().eval()
    trace = tl.trace(
        model,
        torch.randn(1, 3),
        capture=CaptureOptions(layers_to_save="all", replayable=True),
        streaming=StorageOptions(on_save=_sink),
    )

    assert received
    with pytest.raises(AppendStreamingUnsupportedError, match="on_save streaming"):
        trace.run(model, torch.randn(1, 3), replay=ReplayOptions(append=True))


def test_append_rerun_on_loaded_streaming_trace_works(tmp_path: Path) -> None:
    """Loaded traces drop live streaming handles and can append normally."""

    model = _StreamingAppendModel().eval()
    bundle_path = tmp_path / "streaming_bundle.tl"
    tl.trace(
        model,
        torch.randn(1, 3),
        capture=CaptureOptions(layers_to_save="all", replayable=True),
        storage=tl.to_disk(bundle_path),
    )
    loaded = tl.load(bundle_path)

    loaded.run(model, torch.randn(1, 3), replay=ReplayOptions(append=True))

    assert loaded.is_appended is True
    assert loaded._append_sequence_id == 1
