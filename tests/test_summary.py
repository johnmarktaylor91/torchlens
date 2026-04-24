"""Tests for the ModelLog summary feature."""

from __future__ import annotations

from typing import Generator

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


@pytest.fixture()
def tiny_summary_log() -> Generator[tl.ModelLog, None, None]:
    """Return a metadata-only log for the tiny summary model."""
    model = TinySummaryModel()
    x = torch.randn(1, 3, 8, 8)
    log = tl.log_forward_pass(model, x, layers_to_save=None)
    try:
        yield log
    finally:
        log.cleanup()


def test_small_model_default_output_golden(tiny_summary_log: tl.ModelLog) -> None:
    """Default overview output should match the expected compact golden text."""
    summary_text = tiny_summary_log.summary()
    assert summary_text == (
        "Model: TinySummaryModel\n"
        "+-------------------+--------------+--------+-------+\n"
        "| Layer             | Output Shape | Params | Train |\n"
        "+-------------------+--------------+--------+-------+\n"
        "| input             | [1,3,8,8]    | 0      | -     |\n"
        "| conv (Conv2d)     | [1,4,8,8]    | 108    | yes   |\n"
        "| relu (ReLU)       | [1,4,8,8]    | 0      | -     |\n"
        "| flatten (Flatten) | [1,256]      | 0      | -     |\n"
        "| fc (Linear)       | [1,5]        | 1.3 K  | yes   |\n"
        "| output            | [1,5]        | -      | -     |\n"
        "+-------------------+--------------+--------+-------+\n"
        "Params: 1,388 unique; trainable: 1,388\n"
        "Ops: 4 total\n"
        "Saved activations: 0 B\n"
        "Forward FLOPs: 19.2 K  MACs: 9.6 K"
    )


@pytest.mark.parametrize(
    "level, expected_fragment",
    [
        ("overview", "Model: TinySummaryModel"),
        ("graph", "Graph Summary: TinySummaryModel"),
        ("memory", "Memory Summary: TinySummaryModel"),
        ("control_flow", "Control-Flow Summary: TinySummaryModel"),
        ("compute", "Compute Summary: TinySummaryModel"),
        ("cost", "Compute Summary: TinySummaryModel"),
    ],
)
def test_all_level_options(
    tiny_summary_log: tl.ModelLog,
    level: str,
    expected_fragment: str,
) -> None:
    """Every supported level should render without error."""
    summary_text = tiny_summary_log.summary(level=level)  # type: ignore[arg-type]
    assert expected_fragment in summary_text


def test_custom_fields_selection(tiny_summary_log: tl.ModelLog) -> None:
    """Custom field selection should drive the primary table columns."""
    summary_text = tiny_summary_log.summary(fields=["name", "params"])
    assert "| Layer             | Params |" in summary_text
    assert "Output Shape" not in summary_text


def test_show_ops_true_dumps_op_level_rows(tiny_summary_log: tl.ModelLog) -> None:
    """show_ops=True should append operation-level rows."""
    summary_text = tiny_summary_log.summary(show_ops=True, mode="rolled")
    assert "Operations:" in summary_text
    assert "conv2d_1_1" in summary_text
    assert "linear_1_4" in summary_text


def test_repr_remains_short_for_resnet18() -> None:
    """The compact repr should stay comfortably below the sprint cap."""
    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18()
    x = torch.randn(1, 3, 64, 64)
    log = tl.log_forward_pass(model, x, layers_to_save=None)
    try:
        rendered = repr(log)
    finally:
        log.cleanup()
    assert len(rendered) < 1200


def test_torchlens_summary_wrapper() -> None:
    """The top-level wrapper should return the rendered summary string."""
    model = TinySummaryModel()
    x = torch.randn(1, 3, 8, 8)
    summary_text = tl.summary(model, x)
    assert isinstance(summary_text, str)
    assert "Model: TinySummaryModel" in summary_text
