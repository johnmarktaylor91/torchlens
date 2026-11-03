"""Every kept artifact family saved by torchlens 2.x refuses at ``tl.load`` with one typed error.

Torchlens 3 carries no reader for files saved by 2.x. ``tests/goldens/legacy/`` keeps one small
real artifact per format family (see its README), and this module loads each of them and
checks the one refusal: the 2.x origin and the remedy are named, the code is
``artifact_version_below_floor``, and the artifact path travels in the structured fields where
the loader has it. Two forged variants cover the door a restamped manifest uses: a 2.x payload
whose manifest names the current format refuses at the unpickler (the pickled module path), and
a stamp below the oldest known format refuses at the manifest. The 2.16 intervention spec
format (``format_version`` 1) is its own family: it has no ``tlspec_version`` and refuses at
its spec stamp, at ``tl.load`` and at ``tl.validation.validate_tlspec``. So is the v1
dataset-extraction layout (schema ``tl_extract_manifest_v1``, torchlens 2.34 and earlier):
every extraction reader and a resumed extraction refuse it at the manifest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from _refuses_as_2x import assert_refuses_as_2x
from example_models import TinyReluAdd

import torchlens as tl
from torchlens.dataset_extraction import load_extraction, open_extraction, relabel_extraction
from torchlens.errors import ArtifactTooOldError
from torchlens.validation import validate_tlspec

_TESTS = Path(__file__).resolve().parent
_LEGACY = _TESTS / "goldens" / "legacy"

#: family id -> (artifact path, the manifest stamp it must carry, or None for a stampless format).
_FAMILIES: dict[str, tuple[Path, int | None]] = {
    "tlspec_v1": (_LEGACY / "tlspec_v1.tlspec", 1),
    "tlspec_v6_trace": (_LEGACY / "tlspec_v6_trace.tlspec", 6),
    "tlspec_v6_runnable": (_LEGACY / "tlspec_v6_runnable.tlspec", 6),
    "tlspec_v7": (_LEGACY / "tlspec_v7.tlspec", 7),
    "tlspec_v9": (_LEGACY / "tlspec_v9.tlspec", 9),
    "modellog_v2_16_genuine": (_LEGACY / "modellog_v2_16_genuine.tlspec", None),
    "modellog_v2_16_modern_spelling": (
        _TESTS / "fixtures" / "tlspec_v2_16" / "F2_modellog_tiny_cnn.tlspec",
        2,
    ),
}


def _manifest(path: Path) -> dict:
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def test_genuine_modellog_refusal_names_the_one_release_that_reads_it() -> None:
    """The 2.16 ModelLog refusal also names the one release that reads that format."""

    path, _ = _FAMILIES["modellog_v2_16_genuine"]
    error = assert_refuses_as_2x(path)
    assert "torchlens==2.17.0" in error.fields["remedy"]


_INTERVENTION_V2_16 = _LEGACY / "intervention_v2_16.tlspec"


def test_intervention_spec_v2_16_refuses_as_2x() -> None:
    """The kept 2.16 intervention spec refuses at load and at validation with the one error."""

    assert _INTERVENTION_V2_16.exists(), "frozen 2.16 intervention spec missing"
    spec = json.loads((_INTERVENTION_V2_16 / "spec.json").read_text(encoding="utf-8"))
    assert spec["format_version"] == "1", "the artifact must stay a format_version-1 spec"
    assert tl.io.detect_tlspec_format(_INTERVENTION_V2_16) == "v2.16_intervention"

    error = assert_refuses_as_2x(_INTERVENTION_V2_16)
    assert error.fields["observed"] == "intervention format_version=1"
    assert error.fields["path"] == str(_INTERVENTION_V2_16)
    assert "saved again with torchlens 2.36.x is written as format_version 2" in str(error)

    with pytest.raises(ArtifactTooOldError) as excinfo:
        validate_tlspec(_INTERVENTION_V2_16)
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"] == "intervention format_version=1"


_EXTRACTION_V1 = _LEGACY / "extraction_v1"


def _assert_extraction_v1_refusal(error: ArtifactTooOldError, path: Path) -> None:
    message = str(error)
    assert "saved by torchlens 2.x" in message, message
    assert "load it with torchlens 2.36.x" in message, message
    assert error.fields["code"] == "artifact_version_below_floor"
    assert error.fields["observed"] == "extraction manifest schema=tl_extract_manifest_v1"
    assert error.fields["path"] == str(path)


def test_extraction_v1_refuses_as_2x_at_every_reader(tmp_path: Path) -> None:
    """The kept v1 extraction manifest refuses at tl.load and at each extraction reader."""

    manifest = json.loads((_EXTRACTION_V1 / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "tl_extract_manifest_v1", "the artifact must stay a v1 manifest"
    assert str(manifest["torchlens_version"]).startswith("2."), "written by torchlens 2.x"
    work = tmp_path / "extraction_v1"
    shutil.copytree(_EXTRACTION_V1, work)

    _assert_extraction_v1_refusal(assert_refuses_as_2x(work), work)
    for reader in (load_extraction, open_extraction):
        with pytest.raises(ArtifactTooOldError) as excinfo:
            reader(work)
        _assert_extraction_v1_refusal(excinfo.value, work)
    with pytest.raises(ArtifactTooOldError) as excinfo:
        relabel_extraction(work, ["a", "b", "c"])
    _assert_extraction_v1_refusal(excinfo.value, work)
    assert sorted(p.name for p in work.iterdir()) == ["manifest.json"], "nothing migrated in place"


def test_resuming_an_extraction_into_a_v1_artifact_refuses_as_2x(tmp_path: Path) -> None:
    """``extract_dataset(..., resume=True)`` over a v1 artifact refuses before any forward."""

    work = tmp_path / "extraction_v1"
    shutil.copytree(_EXTRACTION_V1, work)
    model = TinyReluAdd().eval()
    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.extract_dataset(
            model,
            torch.randn(3, 3),
            {"relu": "relu"},
            batch_size=2,
            path=work,
            progress=False,
            resume=True,
        )
    _assert_extraction_v1_refusal(excinfo.value, work)
    assert sorted(p.name for p in work.iterdir()) == ["manifest.json"], "nothing written"


def test_v2_extraction_saved_by_2x_refuses_at_every_reader_under_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v2 extraction manifest stamped by torchlens 2.x refuses at the direct readers too.

    ``tl.load`` refuses it at the root manifest; the extraction readers share
    ``load_manifest``, which raises the same one 2.x refusal once the running release is 3.
    """

    manifest = json.loads((_EXTRACTION_V1 / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema"] = "tl_extract_manifest_v2"
    manifest["torchlens_version"] = "2.36.3"
    work = tmp_path / "extraction_v2_saved_by_2x"
    work.mkdir()
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(tl, "__version__", "3.0.0")

    for reader in (load_extraction, open_extraction):
        with pytest.raises(ArtifactTooOldError) as excinfo:
            reader(work)
        message = str(excinfo.value)
        assert "saved by torchlens 2.x" in message, message
        assert "load it with torchlens 2.36.x" in message, message
        assert excinfo.value.fields["code"] == "artifact_version_below_floor"
        assert excinfo.value.fields["observed"] == "torchlens_version=2.36.3"
        assert excinfo.value.fields["path"] == str(work)
    assert sorted(p.name for p in work.iterdir()) == ["manifest.json"], "nothing written"
