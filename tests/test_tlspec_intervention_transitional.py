"""Current intervention saves load; transitional 2.x layouts refuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from example_models import TinyReluAdd as _ReluModel

import torchlens as tl
from torchlens._io import TLSPEC_VERSION
from torchlens.errors import ArtifactTooOldError
from torchlens.intervention.save import load_intervention_spec as legacy_load_intervention_spec
from torchlens.intervention.types import InterventionSpec
from torchlens.options import CaptureOptions


def _interventions() -> tl.Trace:
    """Create a deterministic intervention-ready log.

    Returns
    -------
    tl.Trace
        Model log with a zero-ablation recipe attached.
    """

    torch.manual_seed(1700)
    x = torch.randn(2, 3)
    log = tl.trace(
        _ReluModel(),
        x,
        capture=CaptureOptions(replayable=True, seed=0),
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


def _remove_unified_marker(path: Path) -> None:
    """Convert a unified intervention manifest into transitional form.

    Parameters
    ----------
    path:
        Intervention spec directory.
    """

    manifest = _read_json(path / "manifest.json")
    manifest.pop("tlspec_version", None)
    manifest.pop("format", None)
    recipe = _read_json(path / "intervention.json")
    recipe["format_version"] = "2"
    _write_json(path / "spec.json", recipe)
    (path / "intervention.json").unlink()
    _write_json(path / "manifest.json", manifest)


def test_intervention_writer_emits_current_format(
    tmp_path: Path,
) -> None:
    """The writer emits a 3.0 manifest and recipe, with no legacy payload.

    Parameters
    ----------
    tmp_path:
        Parent of the saved intervention artifact.
    """

    path = tmp_path / "unified.tlspec"
    _interventions().save_intervention(path, level="portable")

    manifest = _read_json(path / "manifest.json")
    assert manifest["tlspec_version"] == TLSPEC_VERSION
    assert manifest["kind"] == "intervention"
    assert manifest["format_version"] == 3
    assert _read_json(path / "intervention.json")["format_version"] == 3
    assert not (path / "spec.json").exists()
    assert tl.io.detect_tlspec_format(path) == "v3"


def test_new_loader_refuses_transitional_intervention_spec(tmp_path: Path) -> None:
    """The polymorphic loader refuses the 2.x transitional layout.

    Parameters
    ----------
    tmp_path:
        Parent of the saved intervention artifact.
    """

    path = tmp_path / "transitional.tlspec"
    _interventions().save_intervention(path, level="portable")
    _remove_unified_marker(path)

    assert tl.io.detect_tlspec_format(path) == "pre_3"
    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"] == "intervention format_version=2"


def test_direct_intervention_reader_refuses_transitional_kind(tmp_path: Path) -> None:
    """The direct reader also refuses an intervention layout written by 2.x.

    Parameters
    ----------
    tmp_path:
        Parent of the saved intervention artifact.
    """

    path = tmp_path / "transitional.tlspec"
    _interventions().save_intervention(path, level="executable_with_callables")
    _remove_unified_marker(path)

    with pytest.raises(ArtifactTooOldError) as excinfo:
        legacy_load_intervention_spec(path)
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"


def test_new_loader_reads_intervention_with_unified_manifest_marker(tmp_path: Path) -> None:
    """The Phase 11.0 loader should dispatch unified intervention manifests by kind."""

    unified_path = tmp_path / "unified.tlspec"
    _interventions().save_intervention(unified_path, level="portable")

    assert tl.io.detect_tlspec_format(unified_path) == "v3"
    loaded = tl.load(unified_path)

    assert isinstance(loaded, InterventionSpec)
    assert loaded.metadata["save_level"] == "portable"
