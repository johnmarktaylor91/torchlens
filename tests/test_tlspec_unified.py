"""Tests for Phase 11 unified ``.tlspec`` writer and reader behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions, VisualizationOptions
from torchlens.validation import validate_tlspec


class UnifiedTinyModel(nn.Module):
    """Tiny model with one intervention-friendly operation."""

    def __init__(self) -> None:
        """Initialize the tiny model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

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

        return torch.relu(self.linear(x))


def _captured_log(*, intervention_ready: bool = False) -> tl.ModelLog:
    """Create a deterministic captured model log.

    Parameters
    ----------
    intervention_ready:
        Whether to include intervention replay metadata.

    Returns
    -------
    tl.ModelLog
        Captured log.
    """

    torch.manual_seed(2100)
    return tl.log_forward_pass(
        UnifiedTinyModel().eval(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=intervention_ready,
            layers_to_save="all",
            random_seed=0,
        ),
        visualization=VisualizationOptions(view="none"),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one manifest JSON object.

    Parameters
    ----------
    path:
        ``.tlspec`` path.

    Returns
    -------
    dict[str, Any]
        Decoded manifest.
    """

    data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.mark.smoke
@pytest.mark.parametrize("level", ["audit", "executable_with_callables", "portable"])
def test_unified_modellog_round_trips_per_save_level(tmp_path: Path, level: str) -> None:
    """ModelLog.save writes unified manifests that load polymorphically."""

    log = _captured_log()
    path = tmp_path / f"model_{level}.tlspec"

    log.save(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.ModelLog)
    assert manifest["kind"] == "model_log"
    assert manifest["tlspec_version"] == 1
    assert manifest["save_level"] == level
    assert [layer.layer_label for layer in loaded.layer_list] == [
        layer.layer_label for layer in log.layer_list
    ]


@pytest.mark.smoke
@pytest.mark.parametrize("level", ["audit", "executable_with_callables", "portable"])
def test_unified_bundle_round_trips_per_save_level(tmp_path: Path, level: str) -> None:
    """Bundle.save writes unified manifests that load as Bundle objects."""

    log = _captured_log()
    bundle = tl.bundle({"baseline": log}, baseline="baseline")
    path = tmp_path / f"bundle_{level}.tlspec"

    bundle.save(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Bundle)
    assert manifest["kind"] == "bundle"
    assert manifest["save_level"] == level
    assert loaded.names == ["baseline"]
    assert isinstance(loaded["baseline"], tl.ModelLog)


@pytest.mark.smoke
@pytest.mark.parametrize("level", ["audit", "executable_with_callables", "portable"])
def test_unified_intervention_round_trips_per_save_level(tmp_path: Path, level: str) -> None:
    """Intervention saves now emit the full unified manifest field set."""

    log = _captured_log(intervention_ready=True)
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    path = tmp_path / f"intervention_{level}.tlspec"

    log.save_intervention(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, InterventionSpec)
    assert manifest["kind"] == "intervention"
    assert manifest["tlspec_version"] == 1
    assert manifest["format_version"] == "1"
    assert manifest["save_level"] == level
    assert loaded.metadata["save_level"] == level


@pytest.mark.smoke
def test_inspect_tlspec_returns_unified_manifest(tmp_path: Path) -> None:
    """Manifest inspection returns parsed unified metadata."""

    path = tmp_path / "inspect.tlspec"
    _captured_log().save(path)

    manifest = tl.io.inspect_tlspec(path)

    assert manifest["kind"] == "model_log"
    assert manifest["body_format"] == "safetensors"
    assert isinstance(manifest["sites"], list)


@pytest.mark.smoke
def test_validate_tlspec_rejects_missing_unified_field(tmp_path: Path) -> None:
    """Schema validation fails closed for malformed unified manifests."""

    path = tmp_path / "invalid.tlspec"
    _captured_log().save(path)
    manifest = _read_manifest(path)
    manifest.pop("model_fingerprint")
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_fingerprint"):
        validate_tlspec(path)
