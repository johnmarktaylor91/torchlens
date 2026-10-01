"""Tests for the Trace summary feature."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._summary_internal._builder import _entry_name


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
    """Bare summary() renders the rebuilt auto ladder; the legacy preset
    spelling keeps the historical compact golden text byte-stable."""
    rebuilt = tiny_summary_log.summary()
    # The rebuilt default (summary memo 3.4): hybrid view, disclosure line,
    # hairline table, labeled footer; the preamble moved to provenance().
    assert rebuilt.startswith("TinySummaryModel | input (1, 3, 8, 8) float32")
    assert "view: hybrid, all 4 ops" in rebuilt
    assert "conv (Conv2d)" in rebuilt
    assert "Params   1,388 declared" in rebuilt
    assert "TorchLens Discoverability Summary" in tiny_summary_log.provenance()
    summary_text = tiny_summary_log.summary(level="overview")
    assert "TorchLens Discoverability Summary" in summary_text
    assert summary_text.endswith(
        "Model: TinySummaryModel\n"
        "+-------------------+--------------+--------+-------+\n"
        "| Module (type)     | Output Shape | Params | Train |\n"
        "+-------------------+--------------+--------+-------+\n"
        "| input             | [1,3,8,8]    | -      | -     |\n"
        "| conv (Conv2d)     | [1,4,8,8]    | 108    | yes   |\n"
        "| relu (ReLU)       | [1,4,8,8]    | 0      | -     |\n"
        "| flatten (Flatten) | [1,256]      | 0      | -     |\n"
        "| fc (Linear)       | [1,5]        | 1.3 K  | yes   |\n"
        "| output            | [1,5]        | -      | -     |\n"
        "+-------------------+--------------+--------+-------+\n"
        "Params: 1,388 unique (parameter identity); trainable: 1,388 (100.0%); frozen: 0\n"
        "Ops: 4 total\n"
        "Edges: 5 total\n"
        "Branching factor: 1.00\n"
        "Saved outs: 0 B\n"
        "Forward FLOPs: 16.6 KFLOPs  MACs: 8.19 KMACs\n"
        "Unknown-FLOPs ops: 0\n"
        "FLOP convention: fma=2 (one multiply-accumulate = 2 FLOPs); "
        "MACs are true multiply-accumulate counts.\n"
        "Health: NOT-CHECKED (6 op output(s) unexamined; see trace.health_facts)"
    )


def test_summary_discloses_unknown_flops_operations() -> None:
    """Summary reports unknown operations excluded from aggregate FLOP totals."""

    class _PadModel(nn.Module):
        """Model containing an intentionally unregistered pad operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pad the final dimension."""

            return torch.nn.functional.pad(x, (1, 1))

    log = tl.trace(_PadModel(), torch.randn(2, 3))

    legacy_expected = (
        "Unknown-FLOPs ops: 1 (pad x1; excluded from FLOP/MAC totals; "
        "remedy: torchlens.capture.flops.register_op_rule)"
    )
    # The rebuilt default footer names the unknown op, the lower-bound
    # status, and the remedy (summary memo 3.5); the legacy preset keeps
    # its historical wording byte-stable through the compatibility table.
    rebuilt = log.summary()
    assert "pad" in rebuilt
    assert "lower bounds" in rebuilt
    assert "remedy: torchlens.capture.flops.register_op_rule" in rebuilt
    assert legacy_expected in log.summary(level="compute")


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

    overview = log.summary(level="overview")
    compute = log.summary(level="compute")

    assert "norm (BatchNorm2d) | [2,3,4,4]" in overview
    assert "| norm " in compute
    assert "float32" in compute
    assert "norm (BatchNorm2d) | [3]" not in overview


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
    tiny_summary_log: tl.Trace,
    level: str,
    expected_fragment: str,
) -> None:
    """Every supported level should render without error."""
    summary_text = tiny_summary_log.summary(level=level)  # type: ignore[arg-type]
    assert expected_fragment in summary_text


@pytest.mark.parametrize("level", ["compute", "cost"])
def test_compute_summary_handles_multi_pass_layers(level: str) -> None:
    """Compute and cost summaries should aggregate repeated-layer timing."""

    log = tl.trace(
        RecurrentSummaryModel(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    try:
        summary_text = log.summary(level=level)  # type: ignore[arg-type]
    finally:
        log.cleanup()

    assert "Compute Summary: RecurrentSummaryModel" in summary_text
    assert "| linear " in summary_text
    assert " s ms" not in summary_text


def test_memory_summary_names_recurrent_layers_with_pass_count() -> None:
    """Memory summary row names should show recurrent layer multiplicity.

    The multiplicity is spelled per aggregation mode, and the two spellings are
    not interchangeable. A rolled table has one row per LAYER, so ``xN`` reads
    correctly. An unrolled table -- which ``mode="auto"`` selects for a
    recurrent trace -- has one row per PASS, where ``xN`` would claim N calls
    on each of N rows; those rows carry the pass-qualified label instead.
    """

    log = tl.trace(
        RecurrentSummaryModel(),
        torch.randn(1, 3),
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    try:
        rolled_text = log.summary(level="memory", mode="rolled")
        auto_text = log.summary(level="memory")
    finally:
        log.cleanup()

    assert "linear_1_1 (x3 passes)" in rolled_text
    assert "linear_1_1 (x3 passes)" not in auto_text
    for pass_num in (1, 2, 3):
        assert f"linear_1_1:{pass_num}" in auto_text


def test_summary_entry_name_uses_layer_num_passes() -> None:
    """Recurrent layer names use ``num_passes`` rather than nonexistent call counts."""

    entry = type(
        "FakeLayer",
        (),
        {"layer_label": "linear_1_1", "num_passes": 3, "ops": {1: object()}},
    )()

    assert _entry_name(entry) == "linear_1_1 (x3 passes)"


def test_custom_fields_selection(tiny_summary_log: tl.Trace) -> None:
    """Custom field selection should drive the primary table columns."""
    summary_text = tiny_summary_log.summary(fields=["name", "params"])
    assert "| Module (type)     | Params |" in summary_text
    assert "Output Shape" not in summary_text


def test_show_ops_true_dumps_op_level_rows(tiny_summary_log: tl.Trace) -> None:
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
    legacy_text = tl.visualization.summary(model, x, level="overview")
    assert "Model: TinySummaryModel" in legacy_text
