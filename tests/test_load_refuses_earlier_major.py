"""A file a release before the current saved format wrote refuses at ``tl.load``.

The current saved format (tlspec 11) starts at ``FIRST_FORMAT3_RELEASE``; no reader for
the field names and package paths of earlier releases is carried. The load door refuses
a directory whose root manifest stamps an earlier release up front: a 2.x stamp below the
floor takes the one 2.x refusal every other 2.x door raises, and an older major refuses
with the same error class naming the saving release and the two ways out (that major's
releases read it, or capture again). The floor compares the full release stamp, so a
2.37.1 stamp refuses under a 2.38 release of the same major.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from test_merged_load_identity import _saved_two_rank
from torch import nn

import torchlens as tl
from torchlens._io.format_contract import (
    FIRST_FORMAT3_RELEASE,
    MIN_TORCHLENS_VERSION_TEXT,
    _release_parts,
    raise_if_saved_by_older_major,
)
from torchlens._io.writer_contract import writer_contract
from torchlens.dataset_extraction import load_extraction, open_extraction
from torchlens.errors import ArtifactTooOldError

_PIN = 'pip install "torchlens<2.38"'


def _saved_trace(tmp_path: Path) -> tuple[Path, str]:
    """Save a small trace and return its path and the release stamped in its manifest."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Sequential(nn.Linear(3, 2), nn.ReLU()), torch.randn(1, 3))
    path = tmp_path / "trace.tlspec"
    trace.save(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return path, str(manifest["torchlens_version"])


def _saved_merged(tmp_path: Path) -> Path:
    """Save a two-rank merged directory (root manifest ``merged-directory``)."""

    return _saved_two_rank(tmp_path)


def _saved_fastlog(tmp_path: Path) -> Path:
    """Record to disk only (root manifest ``fastlog-directory``)."""

    path = tmp_path / "recording.tlfast"
    tl.record(
        nn.Sequential(nn.Linear(3, 2), nn.ReLU()),
        torch.ones(1, 3),
        storage=tl.options.StorageOptions(path=path, retain_in_memory=False),
        recording=tl.options.RecordingOptions(default_op=True),
    )
    return path


def _saved_extraction(tmp_path: Path) -> Path:
    """Extract one layer over a two-row batch (an extraction manifest)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
    address = tl.trace(model, torch.randn(2, 3)).layers[-1].label
    path = tmp_path / "dataset"
    tl.extract_dataset(model, torch.randn(2, 3), [address], batch_size=2, path=path, progress=False)
    return path


_SAVERS = {
    "trace": lambda tmp_path: _saved_trace(tmp_path)[0],
    "merged": _saved_merged,
    "fastlog": _saved_fastlog,
    "extraction": _saved_extraction,
}


def _stamp(path: Path) -> str:
    """Return the release stamped in a saved directory's root manifest."""

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["torchlens_version"])


def _restamp(path: Path, saved: str) -> None:
    """Rewrite the release stamp in a saved directory's root manifest."""

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["torchlens_version"] = saved
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _assert_below_floor(error: ArtifactTooOldError, saved: str, path: Path) -> None:
    """Check the structured fields and remedy of a below-floor refusal."""

    assert error.fields["code"] == "artifact_version_below_floor"
    assert error.fields["observed"] == f"torchlens_version={saved}"
    assert error.fields["floor_torchlens_version"] == FIRST_FORMAT3_RELEASE
    assert error.fields["path"] == str(path)
    assert _PIN in str(error), str(error)
    assert _PIN in error.fields["remedy"]


def test_the_floor_is_2_38_0_and_this_release_is_at_or_above_it() -> None:
    """The floor, the writer-contract floor text and the package version agree."""

    assert FIRST_FORMAT3_RELEASE == "2.38.0"
    assert MIN_TORCHLENS_VERSION_TEXT == FIRST_FORMAT3_RELEASE
    assert writer_contract()["min_torchlens_version"] == FIRST_FORMAT3_RELEASE
    running = _release_parts(tl.__version__)
    floor = _release_parts(FIRST_FORMAT3_RELEASE)
    assert running is not None and floor is not None
    assert running >= floor, "this release would refuse its own saves"


@pytest.mark.parametrize("saved", ["2.37.1", "2.37.99", "2.36.3", "2.0.0", "2", "2.37"])
def test_a_2x_stamp_below_the_floor_refuses(saved: str) -> None:
    """Every 2.x stamp below 2.38.0 refuses with the one 2.x refusal."""

    with pytest.raises(ArtifactTooOldError) as excinfo:
        raise_if_saved_by_older_major(saved, "/a/path", running_version=tl.__version__)
    _assert_below_floor(excinfo.value, saved, Path("/a/path"))


@pytest.mark.parametrize(
    "saved", ["2.38.0", "2.38.0rc1", "2.38.0.dev3", "2.38.1", "2.39.0", "3.0.0", "10.0"]
)
def test_a_stamp_at_or_above_the_floor_passes(saved: str) -> None:
    """A stamp at or above 2.38.0 (pre-release suffixes ignored) is not refused here."""

    raise_if_saved_by_older_major(saved, "/a/path", running_version=tl.__version__)


@pytest.mark.parametrize("saved", [None, 237, "", "v2.37.1", "unknown"])
def test_an_unparseable_stamp_is_left_to_its_own_checks(saved: object) -> None:
    """A missing or unparseable stamp returns untouched (other typed checks own it)."""

    raise_if_saved_by_older_major(saved, "/a/path", running_version=tl.__version__)


@pytest.mark.parametrize("kind", sorted(_SAVERS))
def test_a_root_manifest_stamped_2_37_1_refuses_at_load(tmp_path: Path, kind: str) -> None:
    """Trace, merged, fastlog and extraction directories stamped 2.37.1 refuse at ``tl.load``."""

    path = _SAVERS[kind](tmp_path)
    _restamp(path, "2.37.1")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)

    _assert_below_floor(excinfo.value, "2.37.1", path)


@pytest.mark.parametrize("kind", sorted(_SAVERS))
def test_a_root_manifest_stamped_2_38_0_loads(tmp_path: Path, kind: str) -> None:
    """The same directories saved by this release carry its stamp and load."""

    path = _SAVERS[kind](tmp_path)
    assert _stamp(path) == tl.__version__ == "3.0.3"

    assert tl.load(path) is not None


def test_the_direct_extraction_readers_refuse_a_2_37_1_stamp(tmp_path: Path) -> None:
    """``load_extraction`` and ``open_extraction`` refuse a 2.37.1 stamp like ``tl.load``."""

    path = _saved_extraction(tmp_path)
    load_extraction(path)
    open_extraction(path)
    _restamp(path, "2.37.1")

    for reader in (load_extraction, open_extraction):
        with pytest.raises(ArtifactTooOldError) as excinfo:
            reader(path)
        _assert_below_floor(excinfo.value, "2.37.1", path)


def test_file_saved_by_torchlens_2_refuses_at_load(tmp_path: Path) -> None:
    """A trace stamped by an earlier 2.x release takes the one 2.x refusal."""

    path, _ = _saved_trace(tmp_path)
    _restamp(path, "2.36.3")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)

    message = str(excinfo.value)
    assert "saved by torchlens 2.37 and earlier" in message
    assert "load it with torchlens 2.36.x" in message
    assert "capture the model again" in message
    assert (
        "this torchlens release does not read files saved by torchlens 2.37 and earlier" in message
    )
    assert "torchlens 3" not in message
    _assert_below_floor(excinfo.value, "2.36.3", path)


def test_file_saved_by_an_earlier_major_release_refuses_at_load(tmp_path: Path) -> None:
    """A major older than 2 refuses with the same error class, naming the saver."""

    path, _ = _saved_trace(tmp_path)
    _restamp(path, "1.4.0")

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)

    message = str(excinfo.value)
    assert "saved by torchlens 1.4.0" in message
    assert "torchlens<2" in message
    assert "capture the model again" in message
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"] == "torchlens_version=1.4.0"
    assert excinfo.value.fields["floor_torchlens_version"] == FIRST_FORMAT3_RELEASE
    assert excinfo.value.fields["path"] == str(path)
