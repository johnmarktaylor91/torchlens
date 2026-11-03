"""Tests for the Trace summary feature."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from torch import nn

import torchlens as tl


class TinySummaryModel(nn.Module):
    """Small model with stable top-level module names for summary tests."""

    def __init__(self) -> None:
        """Initialize the test model."""
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 8 * 8, 5, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        return self.fc(x)


class RecurrentSummaryModel(nn.Module):
    """Small model with a repeated module for multi-pass summaries."""

    def __init__(self) -> None:
        """Initialize the repeated layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same layer more than once."""

        for _ in range(3):
            x = self.linear(x)
        return x


@pytest.fixture()
def tiny_summary_log() -> Generator[tl.Trace, None, None]:
    """Return a metadata-only log for the tiny summary model."""
    model = TinySummaryModel()
    x = torch.randn(1, 3, 8, 8)
    log = tl.trace(model, x, save=None)
    try:
        yield log
    finally:
        log.cleanup()


def test_small_model_default_output_golden(tiny_summary_log: tl.Trace) -> None:
    """Bare summary() renders the auto ladder; the preamble lives in provenance()."""
    rebuilt = tiny_summary_log.summary()
    # The rebuilt default (summary memo 3.4): hybrid view, disclosure line,
    # hairline table, labeled footer; the preamble moved to provenance().
    assert rebuilt.startswith("TinySummaryModel | input (1, 3, 8, 8) float32")
    assert "view: hybrid, all 4 ops" in rebuilt
    assert "conv (Conv2d)" in rebuilt
    assert "Params   1,388 declared" in rebuilt
    assert "TorchLens Discoverability Summary" in tiny_summary_log.provenance._summary()


@pytest.mark.parametrize("kwargs", [{"level": "op"}], ids=["compute", "op"])
def test_summary_handles_multi_pass_layers(kwargs: dict[str, str]) -> None:
    """Compute and op views render a repeated layer without the multi-pass accessor error."""

    log = tl.trace(
        RecurrentSummaryModel(),
        torch.randn(1, 3),
        save=None,
    )
    try:
        summary_text = log.summary(**kwargs)
    finally:
        log.cleanup()

    assert summary_text.startswith("RecurrentSummaryModel")
    assert "linear" in summary_text


def test_custom_columns_selection(tiny_summary_log: tl.Trace) -> None:
    """columns= drives the primary table columns."""
    full = str(tiny_summary_log.summary())
    narrow = str(tiny_summary_log.summary(columns=["name", "params"]))
    assert "fwd flops" in full
    assert "fwd flops" not in narrow
    assert "params" in narrow
