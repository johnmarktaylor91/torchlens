"""Entry-point wiring tests for grouped option classes.

The flat option kwargs were removed by the 2026-08-19 shim-removal lane, so
the grouped objects are the ONE spelling: these tests pin that the grouped
door works warning-free and that the removed flat spellings refuse as unknown
keywords instead of silently doing nothing.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions, VisualizationOptions


class _TinyModel(nn.Module):
    """Small deterministic model for option-wiring tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.fc1 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

        return self.fc1(x)


def _input() -> torch.Tensor:
    """Return a stable model input."""

    torch.manual_seed(0)
    return torch.randn(1, 3)


def _capture_summary(log: Any) -> tuple[list[str], int]:
    """Return stable fields for comparing captures."""

    return (list(log.layer_logs.keys()), int(log.num_saved_ops))


def test_trace_grouped_capture_options_route_warning_free() -> None:
    """The grouped capture door works and emits no deprecation warnings."""

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        log = tl.trace(
            _TinyModel(),
            _input(),
            capture=CaptureOptions(layers_to_save="all"),
        )
    try:
        labels, saved = _capture_summary(log)
        assert labels
        assert saved > 0
        deprecations = [
            record for record in records if issubclass(record.category, DeprecationWarning)
        ]
        assert deprecations == []
    finally:
        log.cleanup()


@pytest.mark.parametrize(
    "flat_kwargs",
    [
        {"layers_to_save": "none"},
        {"random_seed": 1},
        {"verbose": True},
        {"save_raw_activations": False},
        {"save_outs_to": "somewhere"},
    ],
    ids=["layers_to_save", "random_seed", "verbose", "save_raw_activations", "save_outs_to"],
)
def test_removed_flat_trace_kwargs_refuse(flat_kwargs: dict[str, Any]) -> None:
    """The removed flat trace kwargs raise TypeError as unknown keywords."""

    with pytest.raises(TypeError):
        tl.trace(_TinyModel(), _input(), **flat_kwargs)


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
