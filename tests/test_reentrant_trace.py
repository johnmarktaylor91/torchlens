"""Regression tests for nested trace refusal."""

import pytest
import torch
from torch import nn

import torchlens as tl


class _NestedTraceModel(nn.Module):
    """Module that attempts to start a nested TorchLens trace."""

    def __init__(self) -> None:
        """Initialize the inner module used by the nested trace."""

        super().__init__()
        self.inner = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attempt a nested trace from inside the active forward pass."""

        tl.trace(self.inner, x)
        return x


def test_nested_trace_raises_typed_error_with_active_model_name() -> None:
    """Nested tracing should raise the public typed re-entrancy error."""

    with pytest.raises(tl.ReentrantTraceError, match="NestedTraceModel"):
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))


def test_reentrant_trace_error_is_exported() -> None:
    """The re-entrancy exception should be part of the top-level public surface."""

    assert tl.ReentrantTraceError.__name__ == "ReentrantTraceError"
    assert "ReentrantTraceError" in tl.__all__
