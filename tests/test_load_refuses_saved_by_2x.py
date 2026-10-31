"""Every kept artifact family saved by torchlens 2.x refuses at ``tl.load`` with one typed error.

Torchlens 3 carries no reader for files saved by 2.x. ``tests/goldens/legacy/`` keeps one small
real artifact per format family (see its README), and this module loads each of them and
checks the one refusal: the 2.x origin and the remedy are named, the code is
``artifact_version_below_floor``, and the artifact path travels in the structured fields where
the loader has it. Two forged variants cover the door a restamped manifest uses: a 2.x payload
whose manifest names the current format refuses at the unpickler (the pickled module path), and
a stamp below the oldest known format refuses at the manifest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from _refuses_as_2x import assert_refuses_as_2x

from torchlens._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION

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


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_family_refuses_as_2x(family: str) -> None:
    """The artifact exists, is the family it claims, and refuses with the one 2.x error."""

    path, stamp = _FAMILIES[family]
    assert path.exists(), f"frozen {family} artifact missing"
    manifest = _manifest(path)
    assert manifest.get("tlspec_version") == stamp, f"{family} must stay a stamp-{stamp} artifact"
    assert str(manifest["torchlens_version"]).startswith("2."), family

    error = assert_refuses_as_2x(path)

    assert error.fields["observed"], family
    assert error.fields["floor_tlspec_version"] == MIN_TLSPEC_VERSION


def test_genuine_modellog_refusal_names_the_one_release_that_reads_it() -> None:
    """The 2.16 ModelLog refusal also names the one release that reads that format."""

    path, _ = _FAMILIES["modellog_v2_16_genuine"]
    error = assert_refuses_as_2x(path)
    assert "torchlens==2.17.0" in error.fields["remedy"]


def test_restamped_2x_payload_refuses_at_the_unpickler(tmp_path: Path) -> None:
    """A 2.x payload under a manifest that names the current format still refuses as 2.x.

    The module path the payload pickled is the evidence: no manifest edit can make a 2.x
    pickle look current.
    """

    source, _ = _FAMILIES["tlspec_v9"]
    work = tmp_path / "restamped.tlspec"
    shutil.copytree(source, work)
    manifest = _manifest(work)
    manifest["tlspec_version"] = TLSPEC_VERSION
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    error = assert_refuses_as_2x(work)

    assert str(error.fields["observed"]).startswith("module_path=torchlens."), error.fields


@pytest.mark.parametrize("stamp", [0, 1, MIN_TLSPEC_VERSION - 1])
def test_stamp_below_the_oldest_known_format_refuses(tmp_path: Path, stamp: int) -> None:
    """A stamp older than any format this release names refuses at the manifest."""

    source, _ = _FAMILIES["tlspec_v9"]
    work = tmp_path / f"stamp_{stamp}.tlspec"
    shutil.copytree(source, work)
    manifest = _manifest(work)
    manifest["tlspec_version"] = stamp
    (work / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    error = assert_refuses_as_2x(work)

    assert error.fields["observed"] == f"tlspec_version={stamp}"
