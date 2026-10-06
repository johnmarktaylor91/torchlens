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
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))
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
    assert "TorchLens Discoverability Summary" in tiny_summary_log.provenance()


def test_summary_discloses_unknown_flops_operations() -> None:
    """Summary reports unknown operations excluded from aggregate FLOP totals."""

    class _PadModel(nn.Module):
        """Model containing an intentionally unregistered pad operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pad the final dimension."""

            return torch.nn.functional.pad(x, (1, 1))

    log = tl.trace(_PadModel(), torch.randn(2, 3))

    # The default footer names the unknown op, the lower-bound status, and
    # the remedy (summary memo 3.5).
    rebuilt = log.summary()
    assert "pad" in rebuilt
    assert "lower bounds" in rebuilt
    assert "remedy: torchlens.capture.flops.register_op_rule" in rebuilt


@pytest.mark.parametrize("training", [True, False])
def test_batchnorm_module_summary_uses_real_output_shape_and_dtype(training: bool) -> None:
    """BatchNorm summary rows use module outputs instead of trailing running-stat buffers."""

    class _BatchNormModel(nn.Module):
        """Small BatchNorm model with captured running-stat buffer operations."""

        def __init__(self) -> None:
            """Initialize the BatchNorm layer."""

            super().__init__()
            self.norm = nn.BatchNorm2d(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Normalize one image batch."""

            return self.norm(x)

    model = _BatchNormModel()
    model.train(training)
    log = tl.trace(model, torch.randn(2, 3, 4, 4))

    for report in (log.summary(), log.summary(view="compute")):
        norm_lines = [line for line in str(report).splitlines() if "norm (BatchNorm2d)" in line]
        assert norm_lines, str(report)
        assert all("(2, 3, 4, 4)" in line for line in norm_lines), norm_lines
        assert not any("(3)" in line for line in norm_lines), norm_lines


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"level": "module"}, {"level": "op"}, {"view": "compute"}],
    ids=["default", "module", "op", "compute"],
)
def test_grammar_axes_render(tiny_summary_log: tl.Trace, kwargs: dict[str, str]) -> None:
    """Every row grain and column bundle renders the model header."""
    summary_text = tiny_summary_log.summary(**kwargs)
    assert summary_text.startswith("TinySummaryModel | input (1, 3, 8, 8) float32")


@pytest.mark.parametrize("kwargs", [{"view": "compute"}, {"level": "op"}], ids=["compute", "op"])
def test_summary_handles_multi_pass_layers(kwargs: dict[str, str]) -> None:
    """Compute and op views render a repeated layer without the multi-pass accessor error."""

    log = tl.trace(
        RecurrentSummaryModel(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save=None),
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


def test_op_level_renders_op_rows(tiny_summary_log: tl.Trace) -> None:
    """level='op' renders one row per executed op pass."""
    summary_text = tiny_summary_log.summary(level="op")
    assert "conv2d" in summary_text
    assert "linear" in summary_text


def test_repr_remains_short_for_resnet18() -> None:
    """The compact repr should stay comfortably below the sprint cap."""
    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18()
    x = torch.randn(1, 3, 64, 64)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        rendered = repr(log)
    finally:
        log.cleanup()
    assert len(rendered) < 1200


def test_torchlens_summary_wrapper() -> None:
    """The top-level wrapper should return the rendered summary string."""
    model = TinySummaryModel()
    x = torch.randn(1, 3, 8, 8)
    summary_text = tl.visualization.summary(model, x)
    assert isinstance(summary_text, str)
    assert summary_text.startswith("TinySummaryModel | input (1, 3, 8, 8) float32")
