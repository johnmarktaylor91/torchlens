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
from torchlens.options import CaptureOptions
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


def _captured_log(*, intervention_ready: bool = False) -> tl.Trace:
    """Create a deterministic captured model log.

    Parameters
    ----------
    intervention_ready:
        Whether to include intervention replay metadata.

    Returns
    -------
    tl.Trace
        Captured log.
    """

    torch.manual_seed(2100)
    return tl.trace(
        UnifiedTinyModel().eval(),
        torch.randn(2, 3),
        capture=CaptureOptions(
            intervention_ready=intervention_ready,
            layers_to_save="all",
            random_seed=0,
        ),
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
    """Trace.save writes unified manifests that load polymorphically."""

    log = _captured_log()
    path = tmp_path / f"model_{level}.tlspec"

    log.save(path, level=level)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert isinstance(loaded, tl.Trace)
    assert manifest["kind"] == "trace"
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
    assert isinstance(loaded["baseline"], tl.Trace)


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

    assert manifest["kind"] == "trace"
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


@pytest.mark.smoke
def test_unified_manifest_records_backward_summary(tmp_path: Path) -> None:
    """Trace.save records backward fields and gradient blob kinds in the manifest."""

    torch.manual_seed(2101)
    model = UnifiedTinyModel().eval()
    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(model, x, save_grads=True)
    log.log_backward(log[log.output_layers[0]].out.sum())
    path = tmp_path / "backward.tlspec"

    log.save(path)
    validate_tlspec(path)
    loaded = tl.load(path)
    manifest = _read_manifest(path)

    assert loaded.has_backward_pass is True
    assert loaded.backward_passes.for_pass(1).pass_index == 1
    assert manifest["backward_summary"]["has_backward_pass"] is True
    assert manifest["backward_summary"]["num_backward_passes"] == log.num_backward_passes
    assert manifest["backward_summary"]["num_grad_fns"] == log.num_grad_fns
    assert manifest["backward_summary"]["gradient_blob_count"] >= 1
    assert "grad" in manifest["backward_summary"]["gradient_blob_kinds"]
    assert any(entry["intended_use"] == "grad" for entry in manifest["body_index"])
    assert (
        "removed backward gradient configuration fields"
        in (manifest["backward_summary"]["old_bundle_policy"])
    )


@pytest.mark.smoke
def test_validate_tlspec_rejects_bad_backward_blob_kind(tmp_path: Path) -> None:
    """Schema validation rejects malformed backward blob kinds."""

    torch.manual_seed(2102)
    model = UnifiedTinyModel().eval()
    x = torch.randn(2, 3, requires_grad=True)
    log = tl.trace(model, x, save_grads=True)
    log.log_backward(log[log.output_layers[0]].out.sum())
    path = tmp_path / "bad_backward_kind.tlspec"
    log.save(path)
    manifest = _read_manifest(path)
    manifest["backward_summary"]["gradient_blob_kinds"] = ["grad", "legacy_grad"]
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gradient_blob_kinds"):
        validate_tlspec(path)
