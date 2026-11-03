"""Tests for Dynamo graph-break correlation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_compat
from torchlens.errors import GraphBreaksNormalizationError, GraphBreaksUnavailableError


class _BreakFreeModel(nn.Module):
    """Small fully traceable model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply traceable tensor operations only.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Rectified affine tensor.
        """

        return torch.relu(x + 1)


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_graph_breaks.py.
def test_graph_breaks_unavailable_capability_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing explain capability fails with the public typed error."""

    monkeypatch.setattr(_torch_compat, "HAS_DYNAMO_EXPLAIN", False)

    with pytest.raises(GraphBreaksUnavailableError, match="unavailable") as exc_info:
        tl.debug.graph_breaks(_BreakFreeModel(), torch.ones(2, 3))
    assert exc_info.value.code == "graph_breaks_unavailable"


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_dynamo_explain_shapes.py, debug/_graph_breaks.py.
def test_graph_breaks_unknown_runtime_shape_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized explain output fails typed instead of attribute-crashing."""

    monkeypatch.setattr(_torch_compat, "HAS_DYNAMO_EXPLAIN", True)
    monkeypatch.setattr(_torch_compat, "run_dynamo_explain", lambda *_args, **_kwargs: object())

    with pytest.raises(GraphBreaksNormalizationError, match="unsupported") as excinfo:
        tl.debug.graph_breaks(_BreakFreeModel(), torch.ones(2, 3))
    assert excinfo.value.fields["code"] == "graph_breaks_normalization_failed"
