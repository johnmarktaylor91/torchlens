"""Append rerun regression tests for active streaming traces."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import AppendStreamingNotSupportedError


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
        save_grads_to=tmp_path / "streaming_bundle.tl",
        layers_to_save="all",
        intervention_ready=True,
    )

    with pytest.raises(AppendStreamingNotSupportedError, match="bundle_path streaming"):
        trace.rerun(model, torch.randn(1, 3), append=True)


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
        out_sink=_sink,
        layers_to_save="all",
        intervention_ready=True,
    )

    assert received
    with pytest.raises(AppendStreamingNotSupportedError, match="out_callback streaming"):
        trace.rerun(model, torch.randn(1, 3), append=True)


def test_append_rerun_on_loaded_streaming_trace_works(tmp_path: Path) -> None:
    """Loaded traces drop live streaming handles and can append normally."""

    model = _StreamingAppendModel().eval()
    bundle_path = tmp_path / "streaming_bundle.tl"
    tl.trace(
        model,
        torch.randn(1, 3),
        save_outs_to=bundle_path,
        layers_to_save="all",
        intervention_ready=True,
    )
    loaded = tl.load(bundle_path)

    loaded.rerun(model, torch.randn(1, 3), append=True)

    assert loaded.is_appended is True
    assert loaded._append_sequence_id == 1
