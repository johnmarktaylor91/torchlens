"""Tests for the fp64 precision bisector in torchlens.debug."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class CancellingModel(nn.Module):
    """Model with an engineered fp32 catastrophic-cancellation site."""

    def __init__(self) -> None:
        """Initialize the surrounding well-conditioned layers."""

        super().__init__()
        self.pre = nn.Linear(8, 8)
        self.post = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Push values to 1e7 and cancel back, destroying fp32 mantissa bits."""

        y = self.pre(x)
        big = y + 1e7
        cancelled = big - 1e7
        return self.post(cancelled)


class CleanModel(nn.Module):
    """Well-conditioned model that should agree with the fp64 reference."""

    def __init__(self) -> None:
        """Initialize two linear layers."""

        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear / relu / linear."""

        return self.b(torch.relu(self.a(x)))


def test_bisect_precision_finds_engineered_cancellation() -> None:
    """The first divergent op is the cancellation subtraction, not its ancestors."""

    torch.manual_seed(0)
    model = CancellingModel().eval()
    result = tl.debug.bisect_precision(model, torch.randn(2, 8))

    assert result.found is True
    assert result.func_name == "__sub__"
    assert result.max_rel_err > 0.1
    labels = [row.label for row in result.rows]
    assert labels.index(result.label) >= 1
    for row in result.rows[: labels.index(result.label)]:
        assert row.diverged is False
    assert "First precision divergence" in result.message


def test_bisect_precision_clean_model_reports_no_divergence() -> None:
    """A well-conditioned model stays within dtype-derived tolerance everywhere."""

    torch.manual_seed(0)
    model = CleanModel().eval()
    result = tl.debug.bisect_precision(model, torch.randn(2, 8))

    assert result.found is False
    assert result.label is None
    assert result.rows
    assert all(row.diverged is False for row in result.rows)
    assert result.skipped == ()


def test_bisect_precision_leaves_caller_model_and_rng_untouched() -> None:
    """Both runs happen on deep copies under a forked RNG."""

    torch.manual_seed(0)
    model = CleanModel().eval()
    x = torch.randn(2, 8)
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    rng_before = torch.random.get_rng_state()
    tl.debug.bisect_precision(model, x)

    assert torch.equal(rng_before, torch.random.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])
    assert next(model.parameters()).dtype == torch.float32


def test_bisect_precision_flags_stochastic_first_divergence() -> None:
    """A train-mode dropout divergence is flagged as stochastic and taught."""

    torch.manual_seed(0)

    class Droppy(nn.Module):
        """Linear followed by dropout."""

        def __init__(self) -> None:
            """Initialize the layer and dropout."""

            super().__init__()
            self.a = nn.Linear(8, 8)
            self.drop = nn.Dropout(0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run linear then dropout."""

            return self.drop(self.a(x))

    model = Droppy().train()
    result = tl.debug.bisect_precision(model, torch.randn(4, 8))
    if result.found:
        first_row = next(row for row in result.rows if row.label == result.label)
        assert first_row.stochastic is True
        assert "eval mode" in result.message


@pytest.mark.smoke
def test_bisect_precision_explicit_tolerances_override_dtype_defaults() -> None:
    """Explicit rtol/atol replace per-dtype derivation on every row."""

    torch.manual_seed(0)
    model = CleanModel().eval()
    result = tl.debug.bisect_precision(model, torch.randn(2, 8), rtol=0.5, atol=1.0)

    assert all(row.rtol == 0.5 and row.atol == 1.0 for row in result.rows)
    assert result.found is False
