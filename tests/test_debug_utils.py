"""Tests for ``tl.debug`` diagnostic helpers."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class NanModel(nn.Module):
    """Tiny model that creates a known non-finite op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by zero at a predictable operation."""

        y = x - x
        return y / y


class SparseNanModel(nn.Module):
    """Tiny model with an unsaved birth op and a saved downstream output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create Inf and propagate it through a subsequent add operation.

        Parameters
        ----------
        x:
            Finite input tensor.

        Returns
        -------
        torch.Tensor
            Non-finite output.
        """

        y = x + 1
        born_nonfinite = y / (y - y)
        return born_nonfinite + 1


def test_find_nan_live_reports_op_module_and_test_location() -> None:
    """Live find_nan stops at the non-finite operation with a user location."""

    result = tl.debug.find_nan(nn.Sequential(NanModel()), torch.ones(2, 3))

    assert result.found is True
    assert result.op is not None
    assert result.func_name == "__truediv__"
    assert result.module_address == "0"
    assert result.bad_tensors == ("output",)
    assert result.source_line is not None
    assert __file__ in result.source_line
    assert "FindNanResult(found=True" in repr(result)


def test_trace_find_nan_agrees_with_live_capture() -> None:
    """Post-hoc full-save find_nan agrees with live detection."""

    model = NanModel()
    live = tl.debug.find_nan(model, torch.ones(2, 3))
    post_hoc = tl.trace(model, torch.ones(2, 3)).find_nan()

    assert post_hoc.found is True
    assert post_hoc.func_name == live.func_name == "__truediv__"
    assert post_hoc.kind == live.kind == "nan"
    assert post_hoc.source_line is not None


def test_trace_find_nan_names_sparse_uncertainty_zone() -> None:
    """Sparse scans describe unsaved ancestors instead of overclaiming a birth op."""

    trace = tl.trace(SparseNanModel(), torch.ones(2, 3), save=tl.func("add"))
    result = trace.find_nan()

    assert result.found is True
    assert result.scope == "first among saved tensors"
    assert result.func_name == "__add__"
    assert any("truediv" in label for label in result.uncertainty_zone)
    assert "uncertainty zone" in result.message


def test_find_nan_live_clean_model_returns_clear_result() -> None:
    """Live find_nan returns a no-finding result without raising."""

    result = tl.debug.find_nan(nn.ReLU(), torch.ones(2, 3))

    assert result.found is False
    assert result.kind == "none"
    assert "found=False" in repr(result)


def test_bisect_nan_finds_first_nonfinite_op() -> None:
    """bisect_nan returns the first saved op with NaN or Inf output."""

    trace = tl.trace(NanModel(), torch.randn(2, 3))

    result = tl.debug.bisect_nan(trace)

    assert result.found is True
    assert result.op is not None
    assert result.label == result.op.layer_label
    assert result.kind in {"nan", "inf", "nan+inf"}
    assert "truediv" in str(result.label)
    assert result.source_line is not None


def test_bisect_nan_no_nonfinite_case() -> None:
    """bisect_nan returns a clear no-finding result."""

    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()).eval(), torch.randn(2, 3))

    result = tl.debug.bisect_nan(trace)

    assert result.found is False
    assert result.op is None
    assert result.kind == "none"
    assert result.message == "No NaN/Inf found in saved activations."


def test_bisect_nan_unsaved_region_message() -> None:
    """Selective saves produce an actionable wider-save message."""

    trace = tl.trace(NanModel(), torch.randn(2, 3), save=tl.func("add"))

    result = tl.debug.bisect_nan(trace)

    assert result.found is False
    assert "no saved activation for the suspect region" in result.message
    assert "Re-trace" in result.message


@pytest.mark.parametrize("metric", ["flops", "memory", "duration"])
def test_hot_path_returns_ranked_dataframe(metric: str) -> None:
    """hot_path returns a sorted DataFrame for each metric."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    trace = tl.trace(model, torch.randn(3, 4))

    frame = tl.debug.hot_path(trace, by=metric)  # type: ignore[arg-type]

    assert list(frame.columns) == ["source_file:line", "op_count", "total_cost", "pct_total"]
    assert len(frame) > 0
    assert frame["total_cost"].tolist() == sorted(frame["total_cost"].tolist(), reverse=True)
    assert "excluded_missing_metric_count" in frame.attrs


def test_hot_path_rejects_missing_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    """hot_path raises the standard tabular-extra error when pandas is absent."""

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Raise ImportError for pandas only."""

        if name == "pandas":
            raise ImportError("blocked pandas")
        return real_import(name, *args, **kwargs)

    trace = tl.trace(nn.Sequential(nn.Linear(2, 2)).eval(), torch.randn(1, 2))
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pip install torchlens\[tabular\]"):
        tl.debug.hot_path(trace)


def test_hot_path_reports_missing_metric_exclusions() -> None:
    """hot_path reports how many ops lacked the requested metric."""

    trace = tl.trace(nn.Sequential(nn.ReLU()).eval(), torch.randn(1, 2))
    first_compute = next(op for op in trace.layer_list if int(op.step_index or 0) > 0)
    first_compute.flops_forward = None

    frame = tl.debug.hot_path(trace, by="flops")

    assert frame.attrs["excluded_missing_metric_count"] >= 1
