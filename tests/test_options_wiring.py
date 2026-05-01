"""Entry-point wiring tests for grouped option classes."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._deprecations import _WARNED_DEPRECATIONS
from torchlens.options import CaptureOptions, SaveOptions, VisualizationOptions


class _TinyModel(nn.Module):
    """Small deterministic model for option-wiring tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.fc1 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.fc1(x)


@pytest.fixture(autouse=True)
def clear_deprecation_state() -> None:
    """Reset once-per-process deprecation state between tests."""

    _WARNED_DEPRECATIONS.clear()


def _input() -> torch.Tensor:
    """Return a stable model input."""

    torch.manual_seed(0)
    return torch.randn(1, 3)


def _capture_summary(log: Any) -> tuple[list[str], int]:
    """Return stable fields for comparing captures."""

    return (list(log.layer_logs.keys()), int(log.num_tensors_saved))


def test_log_forward_pass_capture_options_equivalent_to_individual_kwargs() -> None:
    """Grouped capture options should preserve individual-kwarg behavior."""

    grouped = tl.log_forward_pass(
        _TinyModel(),
        _input(),
        capture=CaptureOptions(layers_to_save="all"),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            individual = tl.log_forward_pass(_TinyModel(), _input(), layers_to_save="all")
        try:
            assert _capture_summary(grouped) == _capture_summary(individual)
        finally:
            individual.cleanup()
    finally:
        grouped.cleanup()


def test_log_forward_pass_capture_conflict_raises() -> None:
    """Same capture field supplied both ways should fail early."""

    with pytest.raises(ValueError, match="conflicting capture options"):
        tl.log_forward_pass(
            _TinyModel(),
            _input(),
            capture=CaptureOptions(layers_to_save="all"),
            layers_to_save="none",
        )


def test_log_forward_pass_save_conflict_raises() -> None:
    """Same save field supplied both ways should fail early."""

    with pytest.raises(ValueError, match="conflicting save options"):
        tl.log_forward_pass(
            _TinyModel(),
            _input(),
            save=SaveOptions(save_raw_activation=True),
            save_raw_activation=False,
        )


def test_log_forward_pass_individual_capture_kwarg_warns() -> None:
    """Individual capture kwargs should warn during the migration window."""

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        log = tl.log_forward_pass(_TinyModel(), _input(), layers_to_save="all")
    try:
        messages = [str(record.message) for record in records]
        assert any(
            "layers_to_save" in message and "capture.layers_to_save" in message
            for message in messages
        )
    finally:
        log.cleanup()


def test_visualization_canonical_kwargs_route_without_deprecation() -> None:
    """Canonical visualization kwargs should route to render settings without warnings."""

    options = VisualizationOptions(view="rolled", depth=2, layout="dot", node_style="profiling")
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        assert options.view == "rolled"
        assert options.depth == 2
        assert options.layout == "dot"
        assert options.node_style == "profiling"
    assert [record for record in records if issubclass(record.category, DeprecationWarning)] == []
