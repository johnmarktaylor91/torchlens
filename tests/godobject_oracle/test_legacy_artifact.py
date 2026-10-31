"""Frozen legacy-artifact gate: files saved by torchlens 2.x refuse at load, clearly.

The two committed artifacts under ``goldens/`` were written by torchlens 2.34.1 at the
pre-columnar baseline (tree ``db2bc7a5``) and are NEVER regenerated: they ARE the 2.x
format. TorchLens 3 carries no reader for 2.x files (their field names and package paths
changed in the 3.0 rename), so loading either one raises one typed error that names the
2.x origin and the two ways out: save it again with torchlens 2.36.x, or capture again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import torchlens as tl
from torchlens.errors import ArtifactTooOldError

_GOLDEN_DIR = Path(__file__).resolve().parent / "goldens"

_ANALYSIS_ARTIFACT = _GOLDEN_DIR / "legacy_baseline_cnn.tlspec"
_RUNNABLE_ARTIFACT = _GOLDEN_DIR / "legacy_baseline_cnn_runnable.tlspec"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "artifact", [_ANALYSIS_ARTIFACT, _RUNNABLE_ARTIFACT], ids=["analysis", "runnable"]
)
def test_legacy_artifact_refuses_with_the_2x_remedy(artifact: Path) -> None:
    """A 2.x artifact refuses typed at load, naming its origin and the remedy."""

    assert artifact.exists(), "frozen legacy artifact missing"

    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(str(artifact))

    message = str(excinfo.value)
    assert "saved by torchlens 2.x" in message
    assert "save it again with torchlens 2.36.x" in message
    assert "capture the model again" in message
    assert excinfo.value.fields["code"] == "artifact_version_below_floor"
    assert excinfo.value.fields["observed"].startswith("module_path=torchlens.")
