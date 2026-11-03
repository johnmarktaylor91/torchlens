"""Tests for ``tl.debug`` diagnostic helpers."""

from __future__ import annotations

import inspect

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


def _user_nan_function(x: torch.Tensor) -> torch.Tensor:
    """Produce a NaN from a user-defined call site.

    Parameters
    ----------
    x:
        Finite model input.

    Returns
    -------
    torch.Tensor
        Tensor containing NaN values.
    """

    zero = x - x
    return zero / zero


class UserFunctionNanModel(nn.Module):
    """Model that delegates its non-finite operation to user code."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the user-defined non-finite operation.

        Parameters
        ----------
        x:
            Model input.

        Returns
        -------
        torch.Tensor
            User-function output.
        """

        return _user_nan_function(x)


# Kept by the shrink-r4 coverage proof: sole branch coverage of backends/torch/model_prep.py, debug/_nan.py and a sole campaign mutation kill.
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
    assert "FindNanReport(found=True" in repr(result)


# Kept: reads source statically or runs a subprocess, so per-id coverage cannot measure it (shrink rule 3).
def test_find_nan_live_reports_nearest_user_function_location() -> None:
    """Live find_nan reports the function containing the offending operation."""

    result = tl.debug.find_nan(UserFunctionNanModel(), torch.ones(2, 3))
    source, start_line = inspect.getsourcelines(_user_nan_function)
    expected_line = start_line + source.index("    return zero / zero\n")

    assert result.found is True
    assert result.source_line == f"{__file__}:{expected_line}"


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_nan.py and a sole campaign mutation kill.
def test_trace_find_nan_names_sparse_uncertainty_zone() -> None:
    """Sparse scans describe unsaved ancestors instead of overclaiming a birth op."""

    trace = tl.trace(SparseNanModel(), torch.ones(2, 3), save=tl.func("add"))
    result = trace.find_nan()

    assert result.found is True
    assert result.scope == "first among saved tensors"
    assert result.func_name == "__add__"
    assert any("truediv" in label for label in result.uncertainty_zone)
    assert "uncertainty zone" in result.message


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_nan.py.
def test_bisect_nan_unsaved_region_message() -> None:
    """Selective saves produce an actionable wider-save message.

    The selector must MATCH a real site: ``NanModel`` computes only ``sub`` and
    ``div``, so ``tl.func("add")`` matched nothing, capture legitimately warned
    that the save selector selected no activations, and the suite's
    error-on-warning policy failed the test. ``tl.func("sub")`` saves the finite
    ``x - x`` and leaves the non-finite ``y / y`` unsaved, which is the actual
    scenario named here -- a genuine selective save whose saved region misses the
    suspect op, not a save that selected nothing at all.
    """

    trace = tl.trace(NanModel(), torch.randn(2, 3), save=tl.func("sub"))

    result = tl.debug.bisect_nan(trace)

    assert result.found is False
    assert "no saved activation for the suspect region" in result.message
    assert "Re-trace" in result.message


# Kept by the shrink-r4 coverage proof: a sole campaign mutation kill.
@pytest.mark.parametrize(
    "metric",
    [
        "memory",
    ],
)
def test_hot_path_returns_ranked_dataframe(metric: str) -> None:
    """hot_path returns a sorted DataFrame for each metric."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    trace = tl.trace(model, torch.randn(3, 4))

    frame = tl.debug.hot_path(trace, by=metric).to_pandas()  # type: ignore[arg-type]

    assert list(frame.columns) == ["source_file:line", "op_count", "total_cost", "pct_total"]
    assert len(frame) > 0
    assert frame["total_cost"].tolist() == sorted(frame["total_cost"].tolist(), reverse=True)
    assert "excluded_missing_metric_count" in frame.attrs
