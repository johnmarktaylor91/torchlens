"""Lane A08 row gate: the harvested genuine-release corpus still round-trips.

The persistence-honesty fixes (WT1 A-IV items 17-22 + W3) must not change how
LAWFUL existing artifacts load: every in-window genuine artifact (writers
v2.31.0 through unreleased main) loads, and the below-floor v2.16.0 ModelLog
bundle keeps its GOVERNED typed refusal (drop-not-resurrect). Loaded
attestation statuses can only be COMPLETE (a genuine attested writer) or
UNATTESTED (legacy finished artifacts) -- never a laundered upgrade or a new
refusal introduced by this lane.

The corpus is the ecosystem panel's harvest (sha256 ``3429a84f...``), committed
as ``tests/release_goldens/genuine_release_artifacts.tar.gz``;
``TORCHLENS_HARVESTED_CORPUS`` overrides the location. Absent bytes SKIP loudly.
"""

from __future__ import annotations

import hashlib
import os
import tarfile
from pathlib import Path

import pytest
from _oracle_env import expect_bundle_minor_version_mismatch

import torchlens as tl
from torchlens._io import ArtifactVersionBelowFloorError
from torchlens.capture.outcome import CaptureStatus

pytestmark = [pytest.mark.smoke]

_CORPUS_SHA256 = (  # content digest, not a credential
    "3429a84fbd406d374afde5b50f6a8be0867ada77b728d6e88309713c6eadcdb3"  # pragma: allowlist secret
)
_COMMITTED_CORPUS = Path(__file__).parent / "release_goldens" / "genuine_release_artifacts.tar.gz"

#: writer -> loadability expectation. The rehydration floor is torchlens 2.33
#: (tlspec 6); the 2.31/2.32 rows exercise whichever side of the governed
#: window they fall on in the CURRENT runtime, refusing ONLY with the three
#: governed compatibility types -- never an untyped crash or a non-governed
#: refusal introduced by a behavioral change.
_IN_WINDOW = ("art_v2.33.0_portable", "art_v2.34.1_portable", "art_main_portable")
_BELOW_FLOOR = ("art_v2.16.0_portable",)
_WINDOW_EDGE = ("art_v2.31.0_portable", "art_v2.32.4_portable", "art_v2.31.0_audit")


def _corpus_tarball() -> Path | None:
    override = os.environ.get("TORCHLENS_HARVESTED_CORPUS")
    for candidate in ([Path(override)] if override else []) + [_COMMITTED_CORPUS]:
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    tarball = _corpus_tarball()
    if tarball is None:
        pytest.skip(
            "harvested corpus not present (set TORCHLENS_HARVESTED_CORPUS); "
            "the A08 lane gate runs this with the corpus available"
        )
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert digest == _CORPUS_SHA256, "harvested corpus bytes do not match the recorded digest"
    target = tmp_path_factory.mktemp("harvested_corpus")
    with tarfile.open(tarball, "r:gz") as archive:
        archive.extractall(target, filter="data")
    return target


@pytest.mark.parametrize("artifact", _IN_WINDOW)
def test_in_window_artifacts_load(corpus_dir, artifact):
    with expect_bundle_minor_version_mismatch():
        loaded = tl.load(corpus_dir / artifact)
    assert loaded.layer_list, artifact
    assert loaded.outcome.status in (CaptureStatus.COMPLETE, CaptureStatus.UNATTESTED)


@pytest.mark.parametrize("artifact", _BELOW_FLOOR)
def test_below_floor_artifacts_keep_their_governed_refusal(corpus_dir, artifact):
    with pytest.raises(ArtifactVersionBelowFloorError):
        tl.load(corpus_dir / artifact)


@pytest.mark.parametrize("artifact", _WINDOW_EDGE)
def test_window_edge_artifacts_load_or_refuse_governed(corpus_dir, artifact):
    from torchlens._io import (
        ArtifactRuntimeIncompatibleError,
        ArtifactVersionAboveRuntimeError,
    )

    try:
        with expect_bundle_minor_version_mismatch():
            loaded = tl.load(corpus_dir / artifact)
    except (
        ArtifactVersionBelowFloorError,
        ArtifactVersionAboveRuntimeError,
        ArtifactRuntimeIncompatibleError,
    ):
        return
    assert loaded.layer_list, artifact
    assert loaded.outcome.status in (CaptureStatus.COMPLETE, CaptureStatus.UNATTESTED)
