"""Tests for the Phase 11.0 intervention ``.tlspec`` transitional writer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.save import load_intervention_spec as legacy_load_intervention_spec
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions, VisualizationOptions


class _ReluModel(nn.Module):
    """Small model with one portable intervention site."""

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

        return torch.relu(x) + 1


def _intervention_log() -> tl.ModelLog:
    """Create a deterministic intervention-ready log.

    Returns
    -------
    tl.ModelLog
        Model log with a zero-ablation recipe attached.
    """

    torch.manual_seed(1700)
    x = torch.randn(2, 3)
    log = tl.log_forward_pass(
        _ReluModel(),
        x,
        capture=CaptureOptions(intervention_ready=True, random_seed=0),
        visualization=VisualizationOptions(view="none"),
    )
    log.set(tl.func("relu"), tl.zero_ablate(), confirm_mutation=True)
    return log


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object.

    Parameters
    ----------
    path:
        JSON path.

    Returns
    -------
    dict[str, Any]
        Decoded JSON object.
    """

    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one JSON object.

    Parameters
    ----------
    path:
        JSON path.
    data:
        JSON object.
    """

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _copy_as_unified(source: Path, dest: Path) -> None:
    """Copy a transitional spec and mark it as unified.

    Parameters
    ----------
    source:
        Transitional source path.
    dest:
        Unified destination path.
    """

    shutil.copytree(source, dest)
    manifest = _read_json(dest / "manifest.json")
    manifest["tlspec_version"] = 1
    _write_json(dest / "manifest.json", manifest)


@pytest.mark.smoke
def test_transitional_intervention_writer_adds_kind_without_removing_legacy_fields(
    tmp_path: Path,
) -> None:
    """The writer should emit ``kind`` while preserving 2.16.0 fields."""

    path = tmp_path / "transitional.tlspec"
    _intervention_log().save_intervention(path, level="portable")

    manifest = _read_json(path / "manifest.json")
    assert manifest["kind"] == "intervention"
    assert manifest["format_version"] == "1"
    assert manifest["tensor_entries"] == []
    assert tl.io.detect_tlspec_format(path) == "v2.16_intervention_with_kind"


@pytest.mark.smoke
def test_new_loader_reads_transitional_intervention_spec(tmp_path: Path) -> None:
    """The polymorphic loader should read transitional intervention specs."""

    path = tmp_path / "transitional.tlspec"
    _intervention_log().save_intervention(path, level="portable")

    loaded = tl.load(path)

    assert isinstance(loaded, InterventionSpec)
    assert loaded.metadata["save_level"] == "portable"


@pytest.mark.smoke
def test_legacy_intervention_reader_ignores_transitional_kind(tmp_path: Path) -> None:
    """The direct 2.16.0 intervention reader should ignore ``manifest.kind``."""

    path = tmp_path / "transitional.tlspec"
    _intervention_log().save_intervention(path, level="executable_with_callables")

    loaded = legacy_load_intervention_spec(path)

    assert isinstance(loaded, InterventionSpec)
    assert loaded.metadata["save_level"] == "executable_with_callables"


@pytest.mark.smoke
def test_new_loader_reads_intervention_with_unified_manifest_marker(tmp_path: Path) -> None:
    """The Phase 11.0 loader should dispatch unified intervention manifests by kind."""

    transitional_path = tmp_path / "transitional.tlspec"
    unified_path = tmp_path / "unified.tlspec"
    _intervention_log().save_intervention(transitional_path, level="portable")
    _copy_as_unified(transitional_path, unified_path)

    assert tl.io.detect_tlspec_format(unified_path) == "v2.0_unified"
    loaded = tl.load(unified_path)

    assert isinstance(loaded, InterventionSpec)
    assert loaded.metadata["save_level"] == "portable"
