"""Tests for BackwardPass backward call-site metadata."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.func_call_location import FuncCallLocation


class _ContextModel(nn.Module):
    """Small model for backward call-context tests."""

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

        return self.fc(x).relu().sum()


def _trace_and_loss() -> tuple[tl.Trace, torch.Tensor]:
    """Return a trace and loss suitable for deferred backward.

    Returns
    -------
    tuple[tl.Trace, torch.Tensor]
        Trace and saved-output loss.
    """

    torch.manual_seed(0)
    model = _ContextModel()
    x = torch.randn(4, 3, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save="all", save_grads=True)
    return trace, trace[trace.output_layers[0]].out


def test_backward_call_context_points_to_user_log_backward_call() -> None:
    """BackwardPass stores the test frame that called trace.log_backward."""

    trace, loss = _trace_and_loss()
    try:
        expected_line = inspect.currentframe().f_lineno + 1
        trace.log_backward(loss)
        backward_pass = trace.last_backward_pass

        assert backward_pass is not None
        location = backward_pass.backward_call_context
        assert isinstance(location, FuncCallLocation)
        assert Path(location.file).resolve() == Path(__file__).resolve()
        assert location.func_name == "test_backward_call_context_points_to_user_log_backward_call"
        assert location.line_number == expected_line
        assert Path(location.file).parent.name == "tests"
    finally:
        trace.cleanup()


def test_backward_call_context_survives_tlspec_round_trip(tmp_path: Path) -> None:
    """Portable save/load preserves BackwardPass.backward_call_context."""

    trace, loss = _trace_and_loss()
    bundle_path = tmp_path / "backward_context.tlspec"
    try:
        expected_line = inspect.currentframe().f_lineno + 1
        trace.log_backward(loss)
        tl.save(trace, bundle_path, overwrite=True)
        loaded = tl.load(bundle_path)

        backward_pass = loaded.last_backward_pass
        assert backward_pass is not None
        location = backward_pass.backward_call_context
        assert isinstance(location, FuncCallLocation)
        assert Path(location.file).resolve() == Path(__file__).resolve()
        assert location.func_name == "test_backward_call_context_survives_tlspec_round_trip"
        assert location.line_number == expected_line
    finally:
        trace.cleanup()


@pytest.mark.skip(
    reason=(
        "No stable public fixture currently forces orphan hook-only implicit backward without "
        "TorchLens' wrapped backward triggers."
    )
)
def test_implicit_backward_call_context_is_none() -> None:
    """Implicit backward passes have no explicit Python TorchLens call site."""
