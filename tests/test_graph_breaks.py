"""Tests for Dynamo graph-break correlation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_compat
from torchlens.debug import GraphBreaksNormalizationError, GraphBreaksUnavailableError


class _ExplicitBreakModel(nn.Module):
    """Small model with one guaranteed user-inserted Dynamo break."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run operations on both sides of an explicit graph break.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scaled tensor.
        """

        value = x + 1
        torch._dynamo.graph_break()
        return value * 2


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


@pytest.mark.skipif(
    not _torch_compat.HAS_DYNAMO_EXPLAIN,
    reason="torch._dynamo.explain unavailable",
)
def test_graph_breaks_reports_source_and_safe_correlation() -> None:
    """A guaranteed break retains its user source and correlation accounting."""

    report = tl.report.graph_breaks(_ExplicitBreakModel(), torch.ones(2, 3))

    assert len(report.breaks) >= 1
    graph_break = report.breaks[0]
    assert graph_break.source_file == __file__
    assert graph_break.line_number is not None
    assert "graph_break" in graph_break.reason
    assert graph_break.matched_op_labels or graph_break.unmatched_reason is not None


@pytest.mark.skipif(
    not _torch_compat.HAS_DYNAMO_EXPLAIN,
    reason="torch._dynamo.explain unavailable",
)
def test_graph_breaks_break_free_model_is_empty() -> None:
    """A fully traceable model returns an empty break list."""

    report = tl.report.graph_breaks(_BreakFreeModel(), torch.ones(2, 3))

    assert report.breaks == ()


def test_graph_breaks_unavailable_capability_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing explain capability fails with the public typed error."""

    monkeypatch.setattr(_torch_compat, "HAS_DYNAMO_EXPLAIN", False)

    with pytest.raises(GraphBreaksUnavailableError, match="unavailable") as exc_info:
        tl.report.graph_breaks(_BreakFreeModel(), torch.ones(2, 3))
    assert exc_info.value.code == "graph_breaks_unavailable"


def test_graph_breaks_unknown_runtime_shape_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognized explain output fails typed instead of attribute-crashing."""

    monkeypatch.setattr(_torch_compat, "HAS_DYNAMO_EXPLAIN", True)
    monkeypatch.setattr(_torch_compat, "run_dynamo_explain", lambda *_args, **_kwargs: object())

    with pytest.raises(GraphBreaksNormalizationError, match="unsupported") as excinfo:
        tl.report.graph_breaks(_BreakFreeModel(), torch.ones(2, 3))
    assert excinfo.value.fields["code"] == "graph_breaks_normalization_failed"
