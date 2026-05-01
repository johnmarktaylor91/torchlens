"""Read-compatibility tests for TorchLens 2.16.0 ``.tlspec`` fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.types import HelperSpec, InterventionSpec
from torchlens.options import CaptureOptions, VisualizationOptions
from torchlens.validation import check_spec_compat

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tlspec_v2_16"


class TinyCNN(nn.Module):
    """Small CNN used to regenerate in-memory fixture counterparts."""

    def __init__(self) -> None:
        """Initialize the tiny CNN layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class TinyTransformer(nn.Module):
    """Small transformer fixture model."""

    def __init__(self) -> None:
        """Initialize the tiny transformer layers."""

        super().__init__()
        self.proj = nn.Linear(5, 6)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=6,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(6)
        self.out = nn.Linear(6, 4)
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        x = self.proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.out(x[:, -1, :])


def _tiny_cnn_fixture(seed: int) -> tuple[TinyCNN, torch.Tensor]:
    """Build a deterministic tiny CNN and input.

    Parameters
    ----------
    seed:
        Random seed used for weights and input.

    Returns
    -------
    tuple[TinyCNN, torch.Tensor]
        Model and input tensor.
    """

    torch.manual_seed(seed)
    return TinyCNN().eval(), torch.randn(2, 1, 4, 4)


def _tiny_transformer_fixture(seed: int) -> tuple[TinyTransformer, torch.Tensor]:
    """Build a deterministic tiny transformer and input.

    Parameters
    ----------
    seed:
        Random seed used for weights and input.

    Returns
    -------
    tuple[TinyTransformer, torch.Tensor]
        Model and input tensor.
    """

    torch.manual_seed(seed)
    return TinyTransformer().eval(), torch.randn(2, 3, 5, dtype=torch.float64)


def _capture_cnn(seed: int, *, intervention_ready: bool = False) -> tl.ModelLog:
    """Capture the tiny CNN counterpart for one fixture.

    Parameters
    ----------
    seed:
        Fixture seed.
    intervention_ready:
        Whether to include intervention replay metadata.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    model, x = _tiny_cnn_fixture(seed)
    return tl.log_forward_pass(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=intervention_ready,
            layers_to_save="all",
            random_seed=0,
        ),
        visualization=VisualizationOptions(view="none"),
    )


def _capture_transformer(seed: int) -> tl.ModelLog:
    """Capture the tiny transformer counterpart for one fixture.

    Parameters
    ----------
    seed:
        Fixture seed.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    model, x = _tiny_transformer_fixture(seed)
    return tl.log_forward_pass(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", random_seed=0),
        visualization=VisualizationOptions(view="none"),
    )


def _build_intervention_counterpart(seed: int) -> tl.ModelLog:
    """Build an in-memory intervention counterpart.

    Parameters
    ----------
    seed:
        Fixture seed.

    Returns
    -------
    tl.ModelLog
        Model log with the zero-ablation recipe attached.
    """

    log = _capture_cnn(seed, intervention_ready=True)
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    return log


def _assert_modellog_matches(live_log: tl.ModelLog, loaded_log: tl.ModelLog) -> None:
    """Assert that loaded ModelLog state matches a fresh in-memory capture.

    Parameters
    ----------
    live_log:
        Freshly captured model log.
    loaded_log:
        Loaded fixture model log.
    """

    assert loaded_log.model_name == live_log.model_name
    assert [layer.layer_label for layer in loaded_log.layer_list] == [
        layer.layer_label for layer in live_log.layer_list
    ]
    for live_layer, loaded_layer in zip(live_log.layer_list, loaded_log.layer_list):
        if isinstance(live_layer.activation, torch.Tensor):
            assert isinstance(loaded_layer.activation, torch.Tensor)
            assert loaded_layer.activation.dtype == live_layer.activation.dtype
            assert loaded_layer.activation.shape == live_layer.activation.shape
            assert torch.equal(loaded_layer.activation, live_layer.activation)


def _assert_intervention_matches(
    fixture_spec: InterventionSpec,
    live_log: tl.ModelLog,
) -> None:
    """Assert that a loaded intervention spec matches a live counterpart.

    Parameters
    ----------
    fixture_spec:
        Intervention spec loaded from a 2.16.0 fixture.
    live_log:
        Fresh in-memory log with the same recipe.
    """

    live_spec = live_log._intervention_spec
    assert fixture_spec.target_value_specs == live_spec.target_value_specs
    assert fixture_spec.hook_specs == live_spec.hook_specs
    assert fixture_spec.records == live_spec.records
    fixture_helper = fixture_spec.target_value_specs[0].value
    live_helper = live_spec.target_value_specs[0].value
    assert isinstance(fixture_helper, HelperSpec)
    assert isinstance(live_helper, HelperSpec)
    activation = torch.randn(2, 3)
    assert torch.equal(
        fixture_helper()(activation, hook=None),
        live_helper()(activation, hook=None),
    )

    compat = check_spec_compat(fixture_spec, live_log)
    assert compat.outcome in {"EXACT", "COMPATIBLE_WITH_CONFIRMATION"}
    assert compat.targets_resolve_identically is True


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("fixture_name", "expected_format", "expected_type", "seed"),
    [
        ("F1_intervention_default.tlspec", "v2.16_intervention", InterventionSpec, 1101),
        ("F2_modellog_tiny_cnn.tlspec", "v2.16_modellog_portable", tl.ModelLog, 1102),
        ("F3_modellog_tiny_transformer.tlspec", "v2.16_modellog_portable", tl.ModelLog, 1103),
        ("F4_intervention_audit.tlspec", "v2.16_intervention", InterventionSpec, 1104),
        (
            "F5_intervention_executable_with_callables.tlspec",
            "v2.16_intervention",
            InterventionSpec,
            1105,
        ),
        ("F6_intervention_portable.tlspec", "v2.16_intervention", InterventionSpec, 1106),
    ],
)
def test_v2_16_tlspec_fixture_loads_and_matches_in_memory_counterpart(
    fixture_name: str,
    expected_format: str,
    expected_type: type[Any],
    seed: int,
) -> None:
    """Every 2.16.0 golden fixture should detect, load, and match a live object."""

    fixture_path = FIXTURE_ROOT / fixture_name
    assert tl.io.detect_tlspec_format(fixture_path) == expected_format

    loaded = tl.load(fixture_path)
    assert isinstance(loaded, expected_type)
    if isinstance(loaded, InterventionSpec):
        live_log = _build_intervention_counterpart(seed)
        _assert_intervention_matches(loaded, live_log)
    elif fixture_name == "F2_modellog_tiny_cnn.tlspec":
        _assert_modellog_matches(_capture_cnn(seed), loaded)
    else:
        _assert_modellog_matches(_capture_transformer(seed), loaded)
